# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations
from ._version import __version__, __version_tuple__  # noqa: F401

import importlib
from lmcache.logging import init_logger
import sys
from typing import Optional, Any, List, Union
import torch

logger = init_logger(__name__)
LMCACHE_UPSTREAM_TAG = "v0.4.2"
LMCACHE_HCU_PATCHED = False


def _load_baseline_c_ops():
    # Capture the baseline lmcache.c_ops .so BEFORE we install the proxy, so
    # the proxy can transparently forward non-mem-kernels calls to it.
    # If sys.modules already holds our proxy (re-import), skip it and import
    # the real baseline module directly.
    existing = sys.modules.get("lmcache.c_ops")
    if existing is not None and existing.__name__ != "lmcache_hcu.c_ops":
        return existing
    try:
        return importlib.import_module("lmcache.c_ops")
    except Exception as exc:
        logger.warning("Baseline lmcache.c_ops is not available for fallback: %s", exc)
        return None


def _patch_c_ops() -> None:
    # NOTE: We DO NOT replace the baseline lmcache.c_ops .so.
    # We install a thin Python proxy at sys.modules["lmcache.c_ops"] that
    # forwards ONLY the mem-kernels symbols to lmcache_hcu.hcu_c_ops (the HCU
    # native extension) and falls back to the baseline module for everything
    # else (cachegen, positional encoding, pinned/NUMA memory, PCI bus id, ...).

    logger.info("Apply hcu c_ops patch.")

    baseline_c_ops = _load_baseline_c_ops()
    try:
        import lmcache_hcu.c_ops as proxy_c_ops
    except Exception as exc:
        logger.warning("LMCache-HCU c_ops proxy is not available yet: %s", exc)
        return

    proxy_c_ops._set_base_c_ops(baseline_c_ops)
    sys.modules["lmcache.c_ops"] = proxy_c_ops

    for module_name in (
        "lmcache.v1.gpu_connector",
        "lmcache.v1.multiprocess.server",
    ):
        module = sys.modules.get(module_name)
        if module is not None and hasattr(module, "lmc_ops"):
            module.lmc_ops = proxy_c_ops

def _patch_vllm_paged_mem_gpu_connector_v2() -> None:
    """Replace lmcache.v1.gpu_connector.VLLMPagedMemGPUConnectorV2 with the HCU version.

    The HCU version calls multi_layer_kv_transfer_asymmetric (separate K/V pointer
    lists) instead of the baseline's multi_layer_kv_transfer (a single fused KV
    pointer list). That's the calling convention HCU's mem_kernels expects.

    Ordering: must run AFTER _patch_c_ops(), because our HCU gpu_connector does
    `import lmcache.c_ops as lmc_ops` at module top — the c_ops proxy must
    already be installed in sys.modules when we import it here.

    Skipped when vllm is not installed: our HCU gpu_connector imports
    vllm.envs at module top and would fail otherwise.
    """
    logger.info("Apply hcu_connector patch.")
    try:
        import vllm.envs as _vllm_envs
        del _vllm_envs
    except Exception as exc:
        logger.debug("Skip HCU gpu_connector patch (vllm not available): %s", exc)
        return

    try:
        import lmcache.v1.gpu_connector as base_gpu_connector
        from lmcache_hcu.v1.hcu_connector import (
            VLLMPagedMemHCUConnectorV2 as VLLMPagedMemHCUConnectorV2,
            VLLMBufferLayerwiseGPUConnector as HCUVLLMBufferLayerwiseGPUConnector,
            VLLMPagedMemLayerwiseGPUConnector as HCUVLLMPagedMemLayerwiseGPUConnector
        )
    except Exception as exc:
        logger.warning("Skip HCU gpu_connector patch (import failed): %s", exc)
        return

    base_gpu_connector.VLLMPagedMemGPUConnectorV2 = VLLMPagedMemHCUConnectorV2
    base_gpu_connector.VLLMBufferLayerwiseGPUConnector = HCUVLLMBufferLayerwiseGPUConnector
    base_gpu_connector.VLLMPagedMemLayerwiseGPUConnector = HCUVLLMPagedMemLayerwiseGPUConnector


    import lmcache.v1.gpu_connector.gpu_connectors as _gpu_connectors_mod
    _gpu_connectors_mod.VLLMPagedMemGPUConnectorV2 = VLLMPagedMemHCUConnectorV2
    _gpu_connectors_mod.VLLMBufferLayerwiseGPUConnector = (
        HCUVLLMBufferLayerwiseGPUConnector
    )
    _gpu_connectors_mod.VLLMPagedMemLayerwiseGPUConnector = (
        HCUVLLMPagedMemLayerwiseGPUConnector
    )


    # Rebind the class in already-imported caller modules that pulled it into
    # their own namespace via `from lmcache.v1.gpu_connector import ...`.
    for module_name in (
        "lmcache.v1.manager",
        "lmcache.v1.standalone.__main__",
    ):
        mod = sys.modules.get(module_name)
        if mod is not None and hasattr(mod, "VLLMPagedMemGPUConnectorV2"):
            mod.VLLMPagedMemGPUConnectorV2 = VLLMPagedMemHCUConnectorV2


def _hcu_build_kv_layer_groups(self, kv_caches: dict[str, torch.Tensor]) -> None:
    """Build KV layer groups structure by analyzing each layer's shape and dtype.

    Layers with the same shape and dtype are grouped together. This is useful
    because different layers may have different structures (especially the
    last dimension head_size may differ between groups), and different groups
    may have different dtypes.

    If layer groups are already built (non-empty list), this method does nothing.

    Args:
        kv_caches: Dictionary mapping layer names to KV cache tensors.
    """
    # Skip if already built (non-empty list)
    from lmcache.v1.kv_layer_groups import KVLayerGroupInfo
    from collections import defaultdict
    import os
    if len(self.kv_layer_groups) > 0:
        return

    if len(kv_caches) == 0:
        logger.debug("No KV caches available, skipping KV layer groups building")
        return

    env_value = os.environ.get("VLLM_HCU_USE_CUSTOM_FLASH_ATTN", "false")
    if self.str_to_bool(env_value):
        logger.debug("VLLM_HCU_USE_CUSTOM_FLASH_ATTN is true, skipping KV layer groups building")
        return

    # Group layers by (shape, dtype) in a single loop
    groups_dict: dict[tuple[torch.Size, torch.dtype], list[tuple[str, int]]] = (
        defaultdict(list)
    )

    for idx, (layer_name, kv_cache) in enumerate(kv_caches.items()):
        if isinstance(kv_cache, torch.Tensor):
            shape, dtype = kv_cache.shape, kv_cache.dtype
        else:
            k = kv_cache[0]
            nb, nh, bs, hs = k.shape
            shape, dtype = torch.Size([nb, 2, bs, nh, hs]), k.dtype
        key = (shape, dtype)
        groups_dict[key].append((layer_name, idx))

    # Build KVLayerGroupInfo list
    # Sort groups by the first layer index to maintain order
    def _get_first_layer_index(shape_dtype_key):
        """Get the index of the first layer in a layer group."""
        layer_group = groups_dict[
            shape_dtype_key
        ]  # list of (layer_name, layer_index) tuples
        first_layer_info = layer_group[0]  # first (layer_name, layer_index) tuple
        layer_index = first_layer_info[1]  # extract the layer index
        return layer_index

    sorted_keys = sorted(groups_dict.keys(), key=_get_first_layer_index)

    kv_layer_groups: list[KVLayerGroupInfo] = []
    for shape, dtype in sorted_keys:
        layers = groups_dict[(shape, dtype)]
        layer_names, layer_indices = zip(*layers, strict=False)

        group_info = KVLayerGroupInfo(
            layer_names=list(layer_names),
            layer_indices=list(layer_indices),
            shape=shape,
            dtype=dtype,
        )
        kv_layer_groups.append(group_info)

    # Store the built groups
    self.kv_layer_groups = kv_layer_groups

    # Print the group structure
    logger.info("KV layer groups: %s", kv_layer_groups)

def _hcu_str_to_bool(self, val: str) -> bool:
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ('true', '1', 'yes', 'on')
    return False


def _patch_kv_layer_groups():

    logger.info("Apply hcu kv_layer_groups patch.")

    from lmcache.v1.kv_layer_groups import KVLayerGroupsManager

    KVLayerGroupsManager.str_to_bool = _hcu_str_to_bool
    KVLayerGroupsManager.build_kv_layer_groups = (
        _hcu_build_kv_layer_groups
    )


def _hcu_lmcache_connector_build_kv_layer_groups(self):

    if self.lmcache_engine is not None:
        assert len(self.kv_caches) > 0
        kv_layer_groups_manager = (
            self.lmcache_engine.metadata.kv_layer_groups_manager
        )
        kv_layer_groups_manager.build_kv_layer_groups(self.kv_caches)


def _hcu_flatten_block_ids(block_ids):
    if block_ids is None:
        return None
    if isinstance(block_ids, tuple):
        if len(block_ids) == 0:
            return []
        block_ids = block_ids[0]
    return list(block_ids)


def _hcu_same_block_prefix(lhs, rhs) -> bool:
    if lhs is None or rhs is None or len(lhs) > len(rhs):
        return False
    return all(a == b for a, b in zip(lhs, rhs, strict=False))


def _hcu_patch_request_tracker_update(vllm_v1_adapter) -> None:
    tracker_cls = getattr(vllm_v1_adapter, "RequestTracker", None)
    if tracker_cls is None:
        logger.warning("Skip RequestTracker block-table patch: class not found")
        return

    original_update = tracker_cls.update
    if getattr(original_update, "_lmcache_hcu_block_table_patched", False):
        return

    def _hcu_update_with_block_table_guard(
        self,
        new_token_ids,
        new_block_ids,
        preempted=False,
        lmcache_cached_tokens=0,
        vllm_cached_tokens=0,
        all_token_ids=None,
    ):
        old_block_ids = _hcu_flatten_block_ids(getattr(self, "allocated_block_ids", None))
        incoming_block_ids = _hcu_flatten_block_ids(new_block_ids)

        original_update(
            self,
            new_token_ids,
            new_block_ids,
            preempted=preempted,
            lmcache_cached_tokens=lmcache_cached_tokens,
            vllm_cached_tokens=vllm_cached_tokens,
            all_token_ids=all_token_ids,
        )

        if preempted or old_block_ids is None or incoming_block_ids is None:
            return

        # Some vLLM scheduler paths expose the request's current complete block
        # table, while other paths expose only newly allocated blocks.  The
        # upstream RequestTracker always extends allocated_block_ids, which is
        # only correct for the delta form.  If the incoming table already has the
        # previous table as its prefix, treat it as the authoritative full table
        # and replace the reconstructed shadow table instead of keeping the
        # double-appended result.
        if _hcu_same_block_prefix(old_block_ids, incoming_block_ids):
            self.allocated_block_ids = incoming_block_ids
            logger.debug(
                "LMCACHE_HCU_REQUEST_TRACKER_FULL_BLOCK_TABLE_REPLACE "
                "old_blocks=%d incoming_blocks=%d tokens=%d",
                len(old_block_ids),
                len(incoming_block_ids),
                len(getattr(self, "token_ids", [])),
            )

    _hcu_update_with_block_table_guard._lmcache_hcu_block_table_patched = True
    _hcu_update_with_block_table_guard._lmcache_hcu_original_update = original_update
    tracker_cls.update = _hcu_update_with_block_table_guard


def _hcu_patch_reqmeta_recompute_boundary(vllm_v1_adapter) -> None:
    reqmeta_cls = getattr(vllm_v1_adapter, "ReqMeta", None)
    if reqmeta_cls is None:
        logger.warning("Skip ReqMeta recompute-boundary patch: class not found")
        return

    original_from_request_tracker = getattr(reqmeta_cls, "from_request_tracker", None)
    if original_from_request_tracker is None:
        logger.warning("Skip ReqMeta recompute-boundary patch: from_request_tracker not found")
        return
    if getattr(original_from_request_tracker, "_lmcache_hcu_boundary_patched", False):
        return

    def _hcu_from_request_tracker_with_boundary_guard(*args, **kwargs):
        req_meta = original_from_request_tracker(*args, **kwargs)
        try:
            tracker = args[0] if args else kwargs.get("tracker")
            block_size = kwargs.get("block_size")
            if block_size is None and len(args) >= 2:
                block_size = args[1]
            token_len = len(getattr(tracker, "token_ids", [])) if tracker is not None else 0
            block_len = len(getattr(tracker, "allocated_block_ids", [])) if tracker is not None else 0
            if block_size and block_len * int(block_size) > token_len:
                logger.debug(
                    "LMCACHE_HCU_REQMETA_RECOMPUTE_BOUNDARY "
                    "tokens=%d blocks=%d block_size=%s covered=%d gap=%d",
                    token_len,
                    block_len,
                    block_size,
                    block_len * int(block_size),
                    block_len * int(block_size) - token_len,
                )
        except Exception as exc:
            logger.debug("LMCACHE_HCU_REQMETA_RECOMPUTE_BOUNDARY_DIAG_FAILED: %s", exc)
        return req_meta

    _hcu_from_request_tracker_with_boundary_guard._lmcache_hcu_boundary_patched = True
    _hcu_from_request_tracker_with_boundary_guard._lmcache_hcu_original_from_request_tracker = original_from_request_tracker
    reqmeta_cls.from_request_tracker = _hcu_from_request_tracker_with_boundary_guard

def _patch_vllm_v1_adapter():

    logger.info("Apply hcu vllm_v1_adapter patches.")

    import lmcache.integration.vllm.vllm_v1_adapter as lmc_vllm_v1_adapter

    lmc_vllm_v1_adapter.LMCacheConnectorV1Impl._build_kv_layer_groups = _hcu_lmcache_connector_build_kv_layer_groups
    _hcu_patch_request_tracker_update(lmc_vllm_v1_adapter)
    _hcu_patch_reqmeta_recompute_boundary(lmc_vllm_v1_adapter)
    from lmcache_hcu.integration.vllm._hcu_wait_for_save_patch import (
        patch_wait_for_save,
    )
    patch_wait_for_save(lmc_vllm_v1_adapter)


def _hcu_hash_tokens(
        self,
        tokens: Union[torch.Tensor, List[int]],
        prefix_hash: Optional[int] = None,
        extra_keys: Optional[list[Any]] = None,
) -> int:
    if isinstance(tokens, torch.Tensor):
        tokens_tuple = tuple(tokens.cpu().tolist())
    elif isinstance(tokens, list):
        tokens_tuple = tuple(tokens)
    else:
        raise ValueError(f"Unsupported tokens type: {type(tokens)}")
    _prefix_hash = prefix_hash if prefix_hash is not None else ""
    _extra_keys = tuple(extra_keys) if extra_keys is not None else ()
    result_hash = self.hash_func((_prefix_hash, tokens_tuple, _extra_keys))
    # Ignore extra keys for now
    # Extra keys are for multi-modal inputs and
    # request specific metadata (e.g., LoRA ID).
    return result_hash

def _patch_hash_token():

    logger.info("Apply hcu hash_tokens patch.")

    import lmcache.v1.token_database

    lmcache.v1.token_database.TokenDatabase._hash_tokens = _hcu_hash_tokens


def _patch_config() -> None:
    logger.info("Apply HCU config patch.")
    try:
        import lmcache as lmcache_source
        import lmcache.config as config
        import lmcache_hcu.config as hcu_config
    except Exception as exc:
        logger.warning("Skip HCU config patch (import failed): %s", exc)
        return

    for name in (
        "LMCacheEngineMetadata",
        "LMCacheMemPoolMetadata",
        "LMCacheEngineConfig",
        "GlobalConfig",
        "blend_default_separator",
    ):
        if hasattr(hcu_config, name):
            setattr(config, name, getattr(hcu_config, name))

    sys.modules["lmcache.config"] = config
    setattr(lmcache_source, "config", config)


def _patch_observability() -> None:
    # Handle cases where observability was imported before patching at the corresponding site
    logger.info("Apply HCU observability patch.")
    try:
        import lmcache as lmcache_source
        import lmcache_hcu.observability as lmcache_hcu_observability
    except Exception as exc:
        logger.warning("Skip HCU observability patch (import failed): %s", exc)
        return

    sys.modules["lmcache.observability"] = lmcache_hcu_observability
    setattr(lmcache_source, "observability", lmcache_hcu_observability)


def _patch_xds_config() -> None:
    logger.info("Apply HCU XDS config patch.")
    try:
        import lmcache.v1.config as config
    except Exception as exc:
        logger.warning("Skip XDS config patch (import failed): %s", exc)
        return

    changed = False
    definitions = config._CONFIG_DEFINITIONS
    definitions["max_local_cpu_size"]["default"] = 10.0
    xds_definitions = {
        "xds_path": {"type": Optional[str], "default": None, "env_converter": str},
        "xds_buffer_size": {
            "type": Optional[int],
            "default": None,
            "env_converter": int,
        },
        "max_xds_size": {"type": Optional[float], "default": None, "env_converter": float},
    }
    for name, definition in xds_definitions.items():
        if name not in definitions:
            definitions[name] = definition
            changed = True

    if not changed or not hasattr(config, "create_config_class"):
        return

    namespace_extras = {}
    for public_name, private_name in (
        ("validate", "_validate_config"),
        ("log_config", "_log_config"),
        ("get_extra_config_value", "_get_extra_config_value"),
        ("get_lmcache_worker_ids", "_get_lmcache_worker_ids"),
        ("get_lookup_server_worker_ids", "_get_lookup_server_worker_ids"),
    ):
        value = getattr(config, private_name, None)
        if value is not None:
            namespace_extras[public_name] = value
    from_legacy = getattr(config, "_from_legacy", None)
    if from_legacy is not None:
        namespace_extras["from_legacy"] = classmethod(from_legacy)

    config.LMCacheEngineConfig = config.create_config_class(
        config_name="LMCacheEngineConfig",
        config_definitions=config._CONFIG_DEFINITIONS,
        config_aliases=getattr(config, "_CONFIG_ALIASES", {}),
        deprecated_configs=getattr(config, "_DEPRECATED_CONFIGS", {}),
        namespace_extras=namespace_extras,
    )


def _patch_hyfile_memory_allocator() -> None:
    logger.info("Apply HCU HyFile memory allocator patch.")
    try:
        import lmcache.v1.memory_management as memory_management
        from lmcache_hcu.v1.memory_management import (
            HyFileMemoryAllocator,
            patch_parent_allocator_semantics,
        )
    except Exception as exc:
        logger.warning("Skip HyFile memory allocator patch (import failed): %s", exc)
        return

    patch_parent_allocator_semantics()
    memory_management.HyFileMemoryAllocator = HyFileMemoryAllocator

def _patch_cache_controller_message() -> None:
    logger.info("Apply HCU cache controller message patch.")
    try:
        import lmcache.v1.cache_controller.message as message_mod
    except Exception as exc:
        logger.warning("Skip HCU cache controller message patch (import failed): %s", exc)
        return

    def _hcu_clear_worker_msg_describe(self) -> str:
        return f"Clear tokens {self.tokens} in location {self.location}"

    message_mod.ClearWorkerMsg.describe = _hcu_clear_worker_msg_describe


def _patch_storage_backends() -> None:
    logger.info("Apply HCU storage backend patch.")
    try:
        import lmcache.v1.storage_backend as storage_backend
        import lmcache_hcu.v1.storage_backend.gds_backend as hcu_gds_backend
        from lmcache_hcu.v1.storage_backend.gds_backend import (
            GdsBackend as HCUGdsBackend,
        )
        from lmcache_hcu.v1.storage_backend.xds_backend import XdsBackend
    except Exception as exc:
        logger.warning("Skip HCU storage backend patch (import failed): %s", exc)
        return

    sys.modules["lmcache.v1.storage_backend.gds_backend"] = hcu_gds_backend
    setattr(storage_backend, "gds_backend", hcu_gds_backend)
    storage_backend.GdsBackend = HCUGdsBackend
    storage_backend.XdsBackend = XdsBackend

    original_create = storage_backend.CreateStorageBackends
    if getattr(original_create, "_lmcache_hcu_patched", False):
        return

    def _hcu_create_storage_backends(
            config,
            metadata,
            loop,
            dst_device: str = "cuda",
            lmcache_worker= None,
            skip_backends=None,
            existing_backends=None,
    ):
        backends = original_create(
            config, metadata, loop, dst_device, lmcache_worker,
            skip_backends, existing_backends,
        )
        if (
            getattr(config, "xds_path", None) is None
            or "XdsBackend" in backends
            or (skip_backends is not None and "XdsBackend" in skip_backends)
            or not storage_backend.is_cuda_worker(metadata)
        ):
            return backends

        actual_dst_device = dst_device
        try:
            if storage_backend.is_cuda_worker(metadata):
                actual_dst_device = f"cuda:{torch.cuda.current_device()}"
            else:
                actual_dst_device = "cpu"
        except Exception:
            pass

        local_cpu_backend = backends.get("LocalCPUBackend")
        if local_cpu_backend is None:
            for backend in backends.values():
                if backend.__class__.__name__ == "LocalCPUBackend":
                    local_cpu_backend = backend
                    break

        xds_backend = XdsBackend(
            config,
            metadata,
            loop,
            actual_dst_device,
            local_cpu_backend,
        )
        backends[str(xds_backend)] = xds_backend
        return backends

    _hcu_create_storage_backends._lmcache_hcu_patched = True
    storage_backend.CreateStorageBackends = _hcu_create_storage_backends

    for module_name in (
        "lmcache.v1.cache_engine",
        "lmcache.v1.storage_backend.storage_manager",
    ):
        mod = sys.modules.get(module_name)
        if mod is None and module_name == "lmcache.v1.storage_backend.storage_manager":
            try:
                import lmcache.v1.storage_backend.storage_manager as mod
            except Exception as exc:
                logger.warning("Skip HCU StorageManager patch (import failed): %s", exc)
                mod = None
        if mod is not None and hasattr(mod, "CreateStorageBackends"):
            mod.CreateStorageBackends = _hcu_create_storage_backends

def _patch_cache_engine() -> None:
    logger.info("Apply HCU cache engine patch.")
    try:
        import lmcache.v1.cache_engine as cache_engine
        import lmcache_hcu.observability as hcu_observability
        import lmcache_hcu.v1.cache_engine as hcu_cache_engine
    except Exception as exc:
        logger.warning("Skip HCU cache_engine method patch (import failed): %s", exc)
        return

    cache_engine.LMCStatsMonitor = hcu_observability.LMCStatsMonitor
    cache_engine.LMCacheStatsLogger = hcu_observability.LMCacheStatsLogger

    engine_cls = cache_engine.LMCacheEngine
    builder_cls = cache_engine.LMCacheEngineBuilder
    hcu_retrieve = hcu_cache_engine.LMCacheEngine.retrieve
    hcu_cleanup_memory_objs = hcu_cache_engine.LMCacheEngine.cleanup_memory_objs
    hcu_create_memory_allocator = (
        hcu_cache_engine.LMCacheEngineBuilder._Create_memory_allocator
    )
    hcu_store = hcu_cache_engine.LMCacheEngine.store
    hcu_process_tokens_internal = (
        hcu_cache_engine.LMCacheEngine._process_tokens_internal
    )

    if not getattr(engine_cls.store, "_lmcache_hcu_patched", False):
        hcu_store._lmcache_hcu_patched = True
        engine_cls.store = hcu_store

    if not getattr(
        engine_cls._process_tokens_internal, "_lmcache_hcu_patched", False
    ):
        hcu_process_tokens_internal._lmcache_hcu_patched = True
        engine_cls._process_tokens_internal = hcu_process_tokens_internal

    if not getattr(engine_cls.retrieve, "_lmcache_hcu_patched", False):
        hcu_retrieve._lmcache_hcu_patched = True
        engine_cls.retrieve = hcu_retrieve

    if not getattr(engine_cls.cleanup_memory_objs, "_lmcache_hcu_patched", False):
        hcu_cleanup_memory_objs._lmcache_hcu_patched = True
        engine_cls.cleanup_memory_objs = hcu_cleanup_memory_objs

    if not getattr(
        builder_cls._Create_memory_allocator,
        "_lmcache_hcu_patched",
        False,
    ):
        hcu_create_memory_allocator._lmcache_hcu_patched = True
        builder_cls._Create_memory_allocator = hcu_create_memory_allocator

    for engine in getattr(builder_cls, "_instances", {}).values():
        stats_monitor = getattr(engine, "stats_monitor", None)
        if stats_monitor is not None and not hasattr(
            stats_monitor,
            "on_disk_to_memory_request",
        ):
            engine.stats_monitor = hcu_observability.LMCStatsMonitor.GetOrCreate()

def _patch_local_disk_backend() -> None:
    logger.info("Apply HCU local disk backend runtime patch.")
    try:
        import lmcache.v1.storage_backend.local_disk_backend as local_disk_backend
        from lmcache_hcu.v1.storage_backend.local_disk_backend import (
            patch_local_disk_backend,
        )
    except Exception as exc:
        logger.warning("Skip LocalDiskBackend runtime patch (import failed): %s", exc)
        return

    patch_local_disk_backend(local_disk_backend)



def _patch_pq_executor() -> None:
    logger.info("Apply HCU pq_executor patch.")
    try:
        from lmcache.v1.storage_backend.job_executor.pq_executor import (
            AsyncPQExecutor,
        )
    except Exception as exc:
        logger.warning("Skip pq_executor patch (import failed): %s", exc)
        return

    original_submit = AsyncPQExecutor.submit_job
    if getattr(original_submit, "_lmcache_hcu_pq_closed_guard", False):
        return

    def _make_guarded_submit(original):
        async def _guarded_submit_job(self, fn, *args, **kwargs):
            if getattr(self, "_closed", False):
                raise RuntimeError("priority executor is closed")
            return await original(self, fn, *args, **kwargs)

        return _guarded_submit_job

    guarded_submit = _make_guarded_submit(original_submit)
    guarded_submit._lmcache_hcu_pq_closed_guard = True
    guarded_submit.__wrapped__ = original_submit
    AsyncPQExecutor.submit_job = guarded_submit




def _patch_gpu_memory_allocator_lifecycle() -> None:
    """Patch the GPU memory allocator close lifecycle and guards."""
    logger.info("Apply HCU GPU memory allocator lifecycle patch.")
    try:
        import lmcache.v1.memory_management as memory_management
    except Exception as exc:
        logger.warning("Skip GPU allocator lifecycle patch (import failed): %s", exc)
        return

    gpu_cls = memory_management.GPUMemoryAllocator
    if getattr(gpu_cls, "_lmcache_hcu_lifecycle_patched", False):
        return

    original_init = gpu_cls.__init__

    def _hcu_init(self, *args, **kwargs):
        original_init(self, *args, **kwargs)
        self._closed = False

    def _ensure_open(self):
        if getattr(self, "_closed", False):
            raise RuntimeError(f"{type(self).__name__} is closed")

    def _release_gpu_pool(self):
        if hasattr(self, "allocator"):
            del self.allocator
        if hasattr(self, "tensor"):
            del self.tensor

    original_allocate = gpu_cls.allocate
    original_batched_allocate = gpu_cls.batched_allocate

    def _guarded_allocate(self, *args, **kwargs):
        self._ensure_open()
        return original_allocate(self, *args, **kwargs)

    def _guarded_batched_allocate(self, *args, **kwargs):
        self._ensure_open()
        return original_batched_allocate(self, *args, **kwargs)

    def _guarded_free(self, memory_obj, allocator_type=None):
        with self.device_mem_lock:
            if getattr(self, "_closed", False):
                memory_obj.invalidate()
                return
            self.allocator.free(memory_obj, allocator_type)

    def _guarded_batched_free(self, memory_objs, allocator_type=None, update_stats=True):
        with self.device_mem_lock:
            if getattr(self, "_closed", False):
                for memory_obj in memory_objs:
                    memory_obj.invalidate()
                return
            self.allocator.batched_free(memory_objs, allocator_type, update_stats)

    gpu_cls.__init__ = _hcu_init
    gpu_cls._ensure_open = _ensure_open
    gpu_cls._release_gpu_pool = _release_gpu_pool
    gpu_cls.allocate = _guarded_allocate
    gpu_cls.batched_allocate = _guarded_batched_allocate
    gpu_cls.free = _guarded_free
    gpu_cls.batched_free = _guarded_batched_free
    gpu_cls._lmcache_hcu_lifecycle_patched = True



def _patch_hcu_xds_nobuffer() -> None:
    """Wire the HCU XDS no-buffer (block-first / direct-IO) modules.

    Replaces the upstream modules in `sys.modules` so that any later import
    resolves to the HCU implementation (same mechanism as
    `_patch_storage_backends`).
    """
    logger.info("Apply HCU XDS no-buffer (block-first/direct-IO) patches.")
    import importlib

    try:
        import lmcache.integration.vllm.vllm_v1_adapter as _upstream_adapter
        from lmcache_hcu.integration.vllm.vllm_v1_adapter import (
            LMCacheConnectorV1Impl as _HCUConnectorV1Impl,
        )

        _upstream_adapter.LMCacheConnectorV1Impl.register_cross_layers_kv_cache = (
            _HCUConnectorV1Impl.register_cross_layers_kv_cache
        )
        _upstream_adapter.LMCacheConnectorV1Impl._init_connector_state = (
            _HCUConnectorV1Impl._init_connector_state
        )
        _upstream_adapter.LMCacheConnectorV1Impl._get_engine_kvcaches = (
            _HCUConnectorV1Impl._get_engine_kvcaches
        )
        _upstream_adapter.LMCacheConnectorV1Impl.start_load_kv = (
            _HCUConnectorV1Impl.start_load_kv
        )
        _upstream_adapter.LMCacheConnectorV1Impl.wait_for_save = (
            _HCUConnectorV1Impl.wait_for_save
        )
        from lmcache_hcu.integration.vllm._hcu_wait_for_save_patch import (
            patch_wait_for_save as _reinstall_wait_for_save_patch,
        )

        _reinstall_wait_for_save_patch(_upstream_adapter)
        import lmcache_hcu.integration.vllm.vllm_v1_adapter as _hcu_adapter_module

        for _name in (
            "LMCacheConnectorMetadata",
            "LoadSpec",
            "SaveSpec",
            "DisaggSpec",
            "ReqMeta",
            "RequestTracker",
        ):
            _upstream_cls = getattr(_upstream_adapter, _name, None)
            if _upstream_cls is not None:
                setattr(_hcu_adapter_module, _name, _upstream_cls)
        logger.info(
            "Install HCU cross-layer adapter hooks on the upstream adapter class."
        )
    except Exception as exc:
        logger.warning("Skip cross-layer adapter hooks: %s", exc)

    targets = [
        ("lmcache.v1.metadata", "lmcache_hcu.v1.metadata"),
        ("lmcache.v1.storage_backend.block_first_disk", "lmcache_hcu.v1.storage_backend.block_first_disk"),
        ("lmcache.v1.storage_backend.direct_io", "lmcache_hcu.v1.storage_backend.direct_io"),
        ("lmcache.v1.storage_backend.storage_manager", "lmcache_hcu.v1.storage_backend.storage_manager"),
        ("lmcache.v1.storage_backend.xds_backend", "lmcache_hcu.v1.storage_backend.xds_backend"),
        ("lmcache.v1.gpu_connector.gpu_connectors", "lmcache_hcu.v1.gpu_connector.gpu_connectors"),
        ("lmcache.v1.gpu_connector.utils", "lmcache_hcu.v1.gpu_connector.utils"),
        ("lmcache.v1.manager", "lmcache_hcu.v1.manager"),
        ("lmcache.integration.vllm.lmcache_connector_v1", "lmcache_hcu.integration.vllm.lmcache_connector_v1"),
        ("lmcache.integration.vllm.utils", "lmcache_hcu.integration.vllm.utils"),
    ]
    swapped = []
    for src_name, hcu_name in targets:
        try:
            hcu_mod = importlib.import_module(hcu_name)
        except Exception as exc:
            logger.warning("Skip %s patch (import %s failed): %s", src_name, hcu_name, exc)
            continue
        sys.modules[src_name] = hcu_mod
        swapped.append(src_name)
        parent_name, _, leaf = src_name.rpartition(".")
        try:
            setattr(importlib.import_module(parent_name), leaf, hcu_mod)
        except Exception as exc:
            logger.warning("Skip attribute binding %s.%s: %s", parent_name, leaf, exc)

    replaced = set(swapped)
    rebound = 0
    for mod in list(sys.modules.values()):
        mod_name = getattr(mod, "__name__", None)
        if not mod_name or not mod_name.startswith(("lmcache.", "lmcache_hcu.")):
            continue
        if mod_name in replaced:
            continue
        for attr, value in list(vars(mod).items()):
            try:
                owner = getattr(value, "__module__", None)
            except Exception:
                continue
            if owner in replaced:
                replacement = getattr(sys.modules[owner], attr, None)
                if replacement is not None:
                    setattr(mod, attr, replacement)
                    rebound += 1
    logger.info(
        "Rebound %d stale references left over from the pre-swap imports.", rebound
    )

def apply_runtime_patches() -> None:
    global LMCACHE_HCU_PATCHED
    if LMCACHE_HCU_PATCHED:
        return
    _patch_c_ops()
    _patch_config()
    _patch_observability()
    _patch_xds_config()
    _patch_gpu_memory_allocator_lifecycle()
    _patch_hyfile_memory_allocator()
    _patch_vllm_paged_mem_gpu_connector_v2()
    _patch_kv_layer_groups()
    _patch_vllm_v1_adapter()
    _patch_hash_token()
    _patch_cache_controller_message()
    _patch_pq_executor()
    _patch_hcu_xds_nobuffer()
    _patch_storage_backends()
    _patch_local_disk_backend()
    _patch_cache_engine()
    LMCACHE_HCU_PATCHED = True


apply_runtime_patches()

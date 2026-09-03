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
LMCACHE_UPSTREAM_TAG = "v0.3.13"
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


    # Rebind the class in already-imported caller modules that pulled it into
    # their own namespace via `from lmcache.v1.gpu_connector import ...`.
    for module_name in (
        "lmcache.v1.manager",
        "lmcache.v1.standalone.__main__",
    ):
        mod = sys.modules.get(module_name)
        if mod is not None and hasattr(mod, "VLLMPagedMemGPUConnectorV2"):
            mod.VLLMPagedMemGPUConnectorV2 = VLLMPagedMemHCUConnectorV2


def _hcu_get_shapes(self, num_tokens: Optional[int] = None) -> list[torch.Size]:
    """Get the shapes of the KV cache in LMCache"""
    if num_tokens is None:
        num_tokens = self.chunk_size
    if self.kv_layer_groups_manager.kv_layer_groups:
        shapes = []
        kv_size = 1 if self.use_mla else 2
        for group in self.kv_layer_groups_manager.kv_layer_groups:
            hidden_dim = group.hidden_dim_size(self.use_mla)
            shapes.append(
                torch.Size(
                    [
                        kv_size,
                        group.num_layers,
                        num_tokens,
                        hidden_dim,
                    ]
                )
            )
        return shapes
    else:
        return [
            torch.Size(
                [
                    self.kv_shape[1],
                    self.kv_shape[0],
                    num_tokens,
                    self.kv_shape[3] * self.kv_shape[4],
                ]
            )
        ]

def _patch_metadata_get_shapes() -> None:

    logger.info("Apply hcu metadata get_shapes patch.")

    import lmcache.v1.metadata

    lmcache.v1.metadata.LMCacheMetadata.get_shapes = _hcu_get_shapes


def _hcu_hidden_dim_size(self, use_mla: bool) -> int:  # noqa: F841  # bound onto KVLayerGroupInfo at patch time
    """HCU replacement for KVLayerGroupInfo.hidden_dim_size.

    Extends the baseline to also handle the vLLM FlashAttention PA layout,
    where the KV cache tensor for a group has shape
    ``[num_blocks, block_size, num_heads, head_size]`` (4-dim) instead of
    the baseline's ``[2, num_blocks, block_size, num_heads, head_size]`` (MHA)
    or ``[num_blocks, block_size, head_size]`` (MLA).
    """
    import vllm.envs as envs

    if envs.VLLM_USE_FLASH_ATTN_PA and not use_mla:
        if len(self.shape) == 4:
            # FlashAttention PA: [num_blocks, block_size, num_heads, head_size]
            return self.shape[1] * self.shape[3]
        raise ValueError(f"Invalid shape for FlashAttention PA: {self.shape}")

    if len(self.shape) == 5:
        # MHA: [2, num_blocks, block_size, num_heads, head_size]
        return self.shape[3] * self.shape[4]
    if len(self.shape) == 3:
        # MLA: [num_blocks, block_size, head_size]
        return self.shape[2]
    raise ValueError(f"Invalid shape: {self.shape}")


def _hcu_build_kv_layer_groups(self, kv_caches: dict[str, torch.Tensor], use_mla: bool) -> None:
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
    import vllm.envs as envs
    if len(self.kv_layer_groups) > 0:
        return

    if len(kv_caches) == 0:
        logger.debug("No KV caches available, skipping KV layer groups building")
        return

    # Group layers by (shape, dtype) in a single loop
    groups_dict: dict[tuple[torch.Size, torch.dtype], list[tuple[str, int]]] = (
        defaultdict(list)
    )

    for idx, (layer_name, kv_cache) in enumerate(kv_caches.items()):
        if envs.VLLM_USE_FLASH_ATTN_PA and not use_mla:
            shape = kv_cache[0].shape
            dtype = kv_cache[0].dtype
        else:
            shape = kv_cache.shape
            dtype = kv_cache.dtype
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

def _patch_kv_layer_groups():

    logger.info("Apply hcu kv_layer_groups patch.")

    from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager

    KVLayerGroupInfo.hidden_dim_size = (
        _hcu_hidden_dim_size
    )
    KVLayerGroupsManager.build_kv_layer_groups = (
        _hcu_build_kv_layer_groups
    )


def _hcu_lmcache_connector_build_kv_layer_groups(self):

    if self.lmcache_engine is not None:
        assert len(self.kv_caches) > 0
        kv_layer_groups_manager = (
            self.lmcache_engine.metadata.kv_layer_groups_manager
        )
        vllm_config = self._vllm_config
        use_mla = vllm_config.model_config.use_mla if vllm_config.model_config else False
        kv_layer_groups_manager.build_kv_layer_groups(self.kv_caches, use_mla)


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
    ):
        backends = original_create(config, metadata, loop, dst_device, lmcache_worker)
        if getattr(config, "xds_path", None) is None or "XdsBackend" in backends:
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

def _patch_storage_manager_allocate_and_copy_objects() -> None:
    logger.info("Apply HCU storage manager allocate_and_copy_objects patch.")
    try:
        import lmcache.v1.storage_backend.storage_manager as storage_manager
    except Exception as exc:
        logger.warning(
            "Skip StorageManager allocate_and_copy_objects patch (import failed): %s",
            exc,
        )
        return

    original_allocate_and_copy_objects = storage_manager.allocate_and_copy_objects
    if getattr(
        original_allocate_and_copy_objects,
        "_lmcache_hcu_allocated_keys_patched",
        False,
    ):
        return

    def allocate_and_copy_objects_with_allocated_keys(
            allocator_backend,
            keys,
            src_memory_objs,
            stream,
    ):
        """Return the keys that were actually allocated with copied objects."""
        allocated_keys = []
        allocated_objects = []
        for key, src_memory_obj in zip(keys, src_memory_objs, strict=False):
            if allocator_backend.contains(key):
                continue
            memory_obj = allocator_backend.allocate(
                src_memory_obj.get_shape(),
                src_memory_obj.get_dtype(),
                fmt=src_memory_obj.meta.fmt,
                eviction=True,
                busy_loop=False,
            )

            if memory_obj is None:
                break

            if memory_obj.tensor is None:
                logger.warning(
                    "Allocated MemoryObj has None tensor, this is unexpected. "
                    "Releasing the memory object."
                )
                memory_obj.ref_count_down()
                break

            with torch.cuda.stream(stream):
                memory_obj.tensor.copy_(src_memory_obj.tensor, non_blocking=True)
            allocated_keys.append(key)
            allocated_objects.append(memory_obj)

        stream.synchronize()
        return allocated_keys, allocated_objects

    allocate_and_copy_objects_with_allocated_keys._lmcache_hcu_allocated_keys_patched = True
    storage_manager.allocate_and_copy_objects = allocate_and_copy_objects_with_allocated_keys


def _patch_storage_manager_touch_cache() -> None:
    logger.info("Apply HCU storage manager touch_cache patch.")
    try:
        import lmcache.v1.storage_backend.storage_manager as storage_manager
    except Exception as exc:
        logger.warning("Skip StorageManager touch_cache patch (import failed): %s", exc)
        return

    storage_manager_cls = storage_manager.StorageManager
    original_touch_cache = storage_manager_cls.touch_cache
    if getattr(original_touch_cache, "_lmcache_hcu_xds_patched", False):
        return

    def touch_cache_with_xds(self):
        for backend_name, backend in self.storage_backends.items():
            if backend_name in ("LocalCPUBackend", "LocalDiskBackend", "XdsBackend"):
                backend.touch_cache()

    touch_cache_with_xds._lmcache_hcu_xds_patched = True
    storage_manager_cls.touch_cache = touch_cache_with_xds


def apply_runtime_patches() -> None:
    global LMCACHE_HCU_PATCHED
    if LMCACHE_HCU_PATCHED:
        return
    _patch_c_ops()
    _patch_config()
    _patch_observability()
    _patch_xds_config()
    _patch_hyfile_memory_allocator()
    _patch_vllm_paged_mem_gpu_connector_v2()
    _patch_kv_layer_groups()
    _patch_metadata_get_shapes()
    _patch_vllm_v1_adapter()
    _patch_hash_token()
    _patch_cache_controller_message()
    _patch_storage_backends()
    _patch_local_disk_backend()
    _patch_storage_manager_allocate_and_copy_objects()
    _patch_storage_manager_touch_cache()
    _patch_cache_engine()
    LMCACHE_HCU_PATCHED = True


apply_runtime_patches()

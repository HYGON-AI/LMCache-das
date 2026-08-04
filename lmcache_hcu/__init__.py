#
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
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
from typing import Optional, Any
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
        )
    except Exception as exc:
        logger.warning("Skip HCU gpu_connector patch (import failed): %s", exc)
        return

    base_gpu_connector.VLLMPagedMemGPUConnectorV2 = VLLMPagedMemHCUConnectorV2

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

def _patch_vllm_v1_adapter():

    logger.info("Apply hcu kv_layer_groups patch.")

    import lmcache.integration.vllm.vllm_v1_adapter as lmc_vllm_v1_adapter

    lmc_vllm_v1_adapter.LMCacheConnectorV1Impl._build_kv_layer_groups = _hcu_lmcache_connector_build_kv_layer_groups

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

        return self.hash_func((prefix_hash, tokens_tuple))

def _patch_hash_token():

    logger.info("Apply hcu hash_tokens patch.")

    import lmcache.v1.token_database

    lmcache.v1.token_database.TokenDatabase._hash_tokens = _hcu_hash_tokens
    

def apply_runtime_patches() -> None:
    global LMCACHE_HCU_PATCHED
    if LMCACHE_HCU_PATCHED:
        return
    _patch_c_ops()
    _patch_vllm_paged_mem_gpu_connector_v2()
    _patch_kv_layer_groups()
    _patch_metadata_get_shapes()
    _patch_vllm_v1_adapter()
    _patch_hash_token()
    LMCACHE_HCU_PATCHED = True

apply_runtime_patches()

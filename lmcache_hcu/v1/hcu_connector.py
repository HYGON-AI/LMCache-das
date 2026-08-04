# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Standard

from typing import List, Optional

# Third Party
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import _lmcache_nvtx_annotate
from lmcache.v1.gpu_connector import GPUConnectorInterface
from lmcache.v1.memory_management import MemoryFormat, MemoryObj
from lmcache.v1.metadata import LMCacheMetadata

import sys
import lmcache_hcu.c_ops as lmc_ops
# print(f"[DEBUG] Before import lmcache.c_ops, sys.modules['lmcache.c_ops'] = {sys.modules.get('lmcache.c_ops')}")
# # import lmcache.c_ops as lmc_ops
# print(f"[DEBUG] After import, lmc_ops = {lmc_ops}")
# print(f"[DEBUG] Has asymmetric? {hasattr(lmc_ops, 'multi_layer_kv_transfer_asymmetric')}")

import vllm.envs as envs

logger = init_logger(__name__)


class VLLMPagedMemHCUConnectorV2(GPUConnectorInterface):
    """
    The GPU KV cache should be a nested tuple of K and V tensors.
    More specifically, we have:
    - GPUTensor = Tuple[KVLayer, ...]
    - KVLayer = Tuple[Tensor, Tensor]
    - Tensor: [num_blocks, block_size, num_heads, head_size]

    It will produce / consume memory object with KV_2LTD format
    """

    def __init__(
        self,
        hidden_dim_size: int,
        num_layers: int,
        use_gpu: bool = False,
        **kwargs,
    ):
        """
        If use_gpu is true, it will create a gpu intermediate buffer. In this
        case, it requires the following kwargs:
        - chunk_size: The MAX size of the chunk to be copied to GPU.
        - dtype: The data type of the intermediate buffer.
        """
        self.hidden_dim_size = hidden_dim_size
        self.num_layers = num_layers
        self.kv_cache_pointers = torch.empty(
            num_layers, dtype=torch.int64, device="cpu"
        )
        self.key_cache_pointers = torch.empty(
            num_layers, dtype=torch.int64, device="cpu"
        )
        self.value_cache_pointers = torch.empty(
            num_layers, dtype=torch.int64, device="cpu"
        )
        # Not sure we need a dict here. Maybe a single GPU connector always
        # works with a single device?
        self.kv_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.key_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.value_cache_pointers_on_gpu: dict[int, torch.Tensor] = {}
        self.page_buffer_size = 0

        self.kvcaches: Optional[List[torch.Tensor]] = None

        self.gpu_buffer: Optional[torch.Tensor] = None
        self.use_mla = "use_mla" in kwargs and kwargs["use_mla"]
        if use_gpu:
            assert "chunk_size" in kwargs, (
                "chunk_size should be provided to create a GPU buffer."
            )
            assert "dtype" in kwargs, "dtype should be provided to create a GPU buffer."
            assert "device" in kwargs, (
                "device should be provided to create a GPU buffer."
            )
            shape = self.get_shape(kwargs["chunk_size"])
            self.gpu_buffer = torch.empty(
                shape, dtype=kwargs["dtype"], device=kwargs["device"]
            )

        self.store_stream = torch.cuda.Stream()
        self.load_stream = torch.cuda.Stream()

    @classmethod
    def from_metadata(
        cls,
        metadata: LMCacheMetadata,
        use_gpu: bool = False,
        device: Optional[torch.device] = None,
    ) -> "VLLMPagedMemHCUConnectorV2":
        """Create a connector from LMCacheMetadata.

        Args:
            metadata: The LMCache engine metadata containing model configuration.
            use_gpu: Whether to use GPU intermediate buffer.
            device: The device to use for the connector.

        Returns:
            A new instance of VLLMPagedMemHCUConnectorV2.
        """
        # Extract parameters from metadata
        # kv_shape: (num_layer, 2 or 1, chunk_size, num_kv_head, head_size)
        num_layers = metadata.kv_shape[0]
        chunk_size = metadata.kv_shape[2]
        num_kv_head = metadata.kv_shape[3]
        head_size = metadata.kv_shape[4]
        hidden_dim_size = num_kv_head * head_size

        return cls(
            hidden_dim_size=hidden_dim_size,
            num_layers=num_layers,
            use_gpu=use_gpu,
            chunk_size=chunk_size,
            dtype=metadata.kv_dtype,
            device=device,
            use_mla=metadata.use_mla,
        )

    def _initialize_pointers(self, kv_caches: List[torch.Tensor]) -> torch.Tensor:

        if envs.VLLM_USE_FLASH_ATTN_PA and not self.use_mla:
            self.key_cache_pointers.numpy()[:] = [t.data_ptr() for t,v in kv_caches]
            self.value_cache_pointers.numpy()[:] = [v.data_ptr() for t,v in kv_caches]
            device = kv_caches[0][0].device
            assert device.type == "cuda", "The device should be CUDA."
            idx = device.index
            if idx not in self.key_cache_pointers_on_gpu:
                self.key_cache_pointers_on_gpu[idx] = torch.empty(
                    self.num_layers, dtype=torch.int64, device=device
                )
            self.key_cache_pointers_on_gpu[idx].copy_(self.key_cache_pointers)

            if idx not in self.value_cache_pointers_on_gpu:
                self.value_cache_pointers_on_gpu[idx] = torch.empty(
                    self.num_layers, dtype=torch.int64, device=device
                )
            self.value_cache_pointers_on_gpu[idx].copy_(self.value_cache_pointers)

            assert kv_caches[0][0].dim() == 4
            self.page_buffer_size = kv_caches[0][0].shape[0] * kv_caches[0][0].shape[2]
            return self.key_cache_pointers_on_gpu[idx], self.value_cache_pointers_on_gpu[idx]

        self.device = kv_caches[0].device
        assert self.device.type == "cuda", "The device should be CUDA."
        idx = self.device.index
        if idx in self.kv_cache_pointers_on_gpu:
            return self.kv_cache_pointers_on_gpu[idx]
        self.kv_cache_pointers.numpy()[:] = [t.data_ptr() for t in kv_caches]
        self.kv_cache_pointers_on_gpu[idx] = torch.empty(
            self.num_layers, dtype=torch.int64, device=self.device
        )
        self.kv_cache_pointers_on_gpu[idx].copy_(self.kv_cache_pointers)
        if self.use_mla:
            # kv_caches[0].shape: [num_pages, page_size, head_size]
            assert kv_caches[0].dim() == 3
            self.page_buffer_size = kv_caches[0].shape[0] * kv_caches[0].shape[1]
        else:
            # kv_caches[0].shape: [2, num_pages, page_size, num_heads, head_size]
            assert kv_caches[0].dim() == 5
            self.page_buffer_size = kv_caches[0].shape[1] * kv_caches[0].shape[2]

        return self.kv_cache_pointers_on_gpu[idx]

    @_lmcache_nvtx_annotate
    def to_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)


        :raises ValueError: If 'kvcaches' is not provided in kwargs.
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)

        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if self.use_mla:
            if memory_obj.metadata.fmt != MemoryFormat.KV_MLA_FMT:
                raise ValueError(
                    "The memory object should be in KV_MLA_FMT format in"
                    " order to be processed by VLLMPagedMemHCUConnector"
                )
        else:
            if memory_obj.metadata.fmt != MemoryFormat.KV_2LTD:
                raise ValueError(
                    "The memory object should be in KV_2LTD format in"
                    " order to be processed by VLLMPagedMemHCUConnector"
                )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        if envs.VLLM_USE_FLASH_ATTN_PA and not self.use_mla:
            key_cache_pointers, value_cache_pointers = self._initialize_pointers(self.kvcaches)

            num_heads = self.kvcaches[0][0].shape[1]
            head_size = self.kvcaches[0][0].shape[3]
            block_size = self.kvcaches[0][0].shape[2]
            lmc_ops.multi_layer_kv_transfer_asymmetric(
                memory_obj.tensor,
                key_cache_pointers,
                value_cache_pointers,
                slot_mapping[start:end],
                self.kvcaches[0][0].device,
                num_heads,
                head_size,
                block_size,
                False,
            )
        else:
            kv_cache_pointers = self._initialize_pointers(self.kvcaches)

            lmc_ops.multi_layer_kv_transfer(
                memory_obj.tensor,
                kv_cache_pointers,
                slot_mapping[start:end],
                self.kvcaches[0].device,
                self.page_buffer_size,
                False,
                self.use_mla,
            )

    @_lmcache_nvtx_annotate
    def from_gpu(self, memory_obj: MemoryObj, start: int, end: int, **kwargs):
        """Expect a kwarg 'kvcaches' which is a nested tuple of K and V tensors.
        The kvcaches should correspond to the "WHOLE token sequence".

        Will set the memory_obj.metadata.fmt to MemoryFormat.KV_2LTD.

        Note:
          1. This function expects the 'slot_mapping' is a "full slot mapping"
             where it's length is the same as the whole token sequence.
          2. In the case that there is prefix caching, slot_mapping will starts
             with -1s until the end of the matched prefix. The start and end
             should NEVER overlap with the prefix caching (which means the
             underlying CUDA kernel will never see -1 in slot_mapping)

        :raises ValueError: If 'kvcaches' is not provided in kwargs,
        :raises AssertionError: If the memory object does not have a tensor.
        :raises ValueError: If 'slot_mapping' is not provided in kwargs.
        """
        assert memory_obj.tensor is not None

        self.initialize_kvcaches_ptr(**kwargs)
        assert self.kvcaches is not None, (
            "kvcaches should be provided in kwargs or initialized beforehand."
        )

        if "slot_mapping" not in kwargs:
            raise ValueError("'slot_mapping' should be provided in kwargs.")

        slot_mapping: torch.Tensor = kwargs["slot_mapping"]

        if envs.VLLM_USE_FLASH_ATTN_PA and not self.use_mla:
            key_cache_pointers, value_cache_pointers = self._initialize_pointers(self.kvcaches)
            num_heads = self.kvcaches[0][0].shape[1]
            head_size = self.kvcaches[0][0].shape[3]
            block_size = self.kvcaches[0][0].shape[2]

            #logger.info(f"VLLMPagedMemHCUConnectorV2 from gpu, start = {start}, end = {end}, gpu_buffer.shape = {self.gpu_buffer.shape}")
            #logger.info(f"num_heads: {num_heads}, head_size: {head_size}, block_size: {block_size}")
            #logger.info(f"slot_mapping: {slot_mapping}")
            with torch.cuda.stream(self.store_stream):
                if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
                    #logger.info(f"lmc_ops.multi_layer_kv_transfer from_gpu, memory_obj.tensor.shape: {memory_obj.tensor.shape} ")
                    lmc_ops.multi_layer_kv_transfer_asymmetric(
                        memory_obj.tensor,
                        key_cache_pointers,
                        value_cache_pointers,
                        slot_mapping[start:end],
                        self.kvcaches[0][0].device,
                        num_heads,
                        head_size,
                        block_size,
                        True,
                    )
                else:
                    # kvcaches -> gpu_buffer -> memobj
                    assert self.gpu_buffer.device == self.kvcaches[0][0].device
                    #logger.info(f"lmc_ops.multi_layer_kv_transfer kvcaches -> gpu_buffer -> memobj")
                    tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
                    lmc_ops.multi_layer_kv_transfer_asymmetric(
                        tmp_gpu_buffer,
                        key_cache_pointers,
                        value_cache_pointers,
                        slot_mapping[start:end],
                        self.kvcaches[0][0].device,
                        num_heads,
                        head_size,
                        block_size,
                        True,
                    )
                    memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)
        else:
            kv_cache_pointers = self._initialize_pointers(self.kvcaches)
            with torch.cuda.stream(self.store_stream):
                if self.gpu_buffer is None or end - start != self.gpu_buffer.shape[2]:
                    lmc_ops.multi_layer_kv_transfer(
                        memory_obj.tensor,
                        kv_cache_pointers,
                        slot_mapping[start:end],
                        self.kvcaches[0].device,
                        self.page_buffer_size,
                        True,
                        self.use_mla,
                    )
                else:
                    # kvcaches -> gpu_buffer -> memobj
                    assert self.gpu_buffer.device == self.kvcaches[0].device
                    tmp_gpu_buffer = self.gpu_buffer[:, :, : end - start, :]
                    lmc_ops.multi_layer_kv_transfer(
                        tmp_gpu_buffer,
                        kv_cache_pointers,
                        slot_mapping[start:end],
                        self.kvcaches[0].device,
                        self.page_buffer_size,
                        True,
                        self.use_mla,
                    )
                    memory_obj.tensor.copy_(tmp_gpu_buffer, non_blocking=True)

        if not memory_obj.tensor.is_cuda:
            # Force a synchronize if the target buffer is NOT CUDA device
            # NOTE: for better performance, we may not want to sync for every
            # memory object
            self.store_stream.synchronize()

        if self.use_mla:
            memory_obj.metadata.fmt = MemoryFormat.KV_MLA_FMT

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_to_gpu(self, memory_objs, starts, ends, **kwargs):
        with torch.cuda.stream(self.load_stream):
            for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
                self.to_gpu(memory_obj, start, end, **kwargs)
        self.load_stream.synchronize()

    # TODO(Jiayi): need to optimize to enable real batching
    def batched_from_gpu(self, memory_objs, starts, ends, **kwargs):
        for memory_obj, start, end in zip(memory_objs, starts, ends, strict=False):
            self.from_gpu(memory_obj, start, end, **kwargs)

    def get_shape(self, num_tokens: int) -> torch.Size:
        kv_size = 1 if self.use_mla else 2
        return torch.Size([kv_size, self.num_layers, num_tokens, self.hidden_dim_size])



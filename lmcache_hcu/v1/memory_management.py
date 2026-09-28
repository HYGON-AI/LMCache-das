# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

from contextlib import nullcontext
from functools import wraps
import threading

import torch

from lmcache.logging import init_logger
from lmcache.v1.memory_management import GPUMemoryAllocator

from lmcache_hcu.v1.hipfile import HIPFile

logger = init_logger(__name__)

_PATCHED_PARENT_ALLOCATOR_SEMANTICS = False


def _attach_parent_allocator(memory_obj, parent_allocator):
    if memory_obj is not None and hasattr(memory_obj, "parent_allocator"):
        memory_obj.parent_allocator = parent_allocator
    return memory_obj


def _attach_parent_allocator_batch(memory_objs, parent_allocator):
    if memory_objs is None:
        return None
    for memory_obj in memory_objs:
        _attach_parent_allocator(memory_obj, parent_allocator)
    return memory_objs


def patch_parent_allocator_semantics() -> None:
    """Patch upstream allocator parent ownership without copying memory_management.py.

    LMCache_merge wires parent_allocator through AddressManager,
    TensorMemoryAllocator, and GPUMemoryAllocator so MemoryObj.release/free paths go
    back to the outer allocator. Keep that behavior as method-level monkey patches
    instead of carrying a forked copy of upstream memory_management.py.
    """
    global _PATCHED_PARENT_ALLOCATOR_SEMANTICS
    if _PATCHED_PARENT_ALLOCATOR_SEMANTICS:
        return

    import lmcache.v1.memory_management as mm

    original_address_manager_init = mm.AddressManager.__init__

    @wraps(original_address_manager_init)
    def address_manager_init(self, size, align_bytes=mm.AddressManager.ALIGN_BYTES, parent_allocator=None):
        original_address_manager_init(self, size, align_bytes)
        self.parent_allocator = parent_allocator if parent_allocator is not None else self

    mm.AddressManager.__init__ = address_manager_init

    original_tensor_allocator_init = mm.TensorMemoryAllocator.__init__

    @wraps(original_tensor_allocator_init)
    def tensor_allocator_init(
        self,
        tensor,
        align_bytes=mm.AddressManager.ALIGN_BYTES,
        init_address_space=None,
        parent_allocator=None,
    ):
        original_tensor_allocator_init(
            self,
            tensor,
            align_bytes=align_bytes,
            init_address_space=init_address_space,
        )
        self.parent_allocator = parent_allocator if parent_allocator is not None else self
        self.address_manager = mm.AddressManager(
            self.buffer.numel() if init_address_space is None else init_address_space,
            align_bytes,
            parent_allocator=self.parent_allocator,
        )

    mm.TensorMemoryAllocator.__init__ = tensor_allocator_init

    original_tensor_allocate = mm.TensorMemoryAllocator.allocate

    @wraps(original_tensor_allocate)
    def tensor_allocate(self, *args, **kwargs):
        memory_obj = original_tensor_allocate(self, *args, **kwargs)
        return _attach_parent_allocator(
            memory_obj,
            getattr(self, "parent_allocator", self),
        )

    mm.TensorMemoryAllocator.allocate = tensor_allocate

    original_tensor_batched_allocate = mm.TensorMemoryAllocator.batched_allocate

    @wraps(original_tensor_batched_allocate)
    def tensor_batched_allocate(self, *args, **kwargs):
        memory_objs = original_tensor_batched_allocate(self, *args, **kwargs)
        return _attach_parent_allocator_batch(
            memory_objs,
            getattr(self, "parent_allocator", self),
        )

    mm.TensorMemoryAllocator.batched_allocate = tensor_batched_allocate

    _PATCHED_PARENT_ALLOCATOR_SEMANTICS = True


class HyFileMemoryAllocator(GPUMemoryAllocator):
    _lifecycle_lock = threading.Lock()
    _active_allocators = 0

    def __init__(self, size: int, use_mla: bool = False, device=None):
        patch_parent_allocator_semantics()
        self._closed = True
        self._registered = False
        self._counted = False
        with HyFileMemoryAllocator._lifecycle_lock:
            opens_driver = HyFileMemoryAllocator._active_allocators == 0
            try:
                if opens_driver:
                    HIPFile.init_driver()
                if device is None:
                    device = f"cuda:{torch.cuda.current_device()}"
                super().__init__(size, device, align_bytes=4096)
                self.base_pointer = self.tensor.data_ptr()
                HIPFile.register_global_buffer(self.base_pointer, size)
                self._registered = True
                self._counted = True
                HyFileMemoryAllocator._active_allocators += 1
            except BaseException:
                self._release_gpu_pool()
                if opens_driver:
                    try:
                        HIPFile.close_driver()
                    except Exception:
                        logger.exception("Failed to roll back the HyFile driver")
                raise
        logger.info(
            "Registering HyFile buffer at address %s with size %s",
            self.base_pointer,
            size,
        )

    def __del__(self):
        try:
            self.close()
        except Exception:
            logger.exception("Failed to close HyFileMemoryAllocator")

    def close(self):
        with HyFileMemoryAllocator._lifecycle_lock:
            device_lock = getattr(self, "device_mem_lock", nullcontext())
            with device_lock:
                registered = getattr(self, "_registered", False)
                counted = getattr(self, "_counted", False)
                if getattr(self, "_closed", True) and not registered and not counted:
                    return
                self._closed = True
                if registered:
                    HIPFile.deregister_global_buffer(self.base_pointer)
                    self._registered = False
                if counted:
                    if HyFileMemoryAllocator._active_allocators == 1:
                        HIPFile.close_driver()
                    HyFileMemoryAllocator._active_allocators -= 1
                    self._counted = False
                self._release_gpu_pool()

    def __str__(self):
        return "HyFileMemoryAllocator"

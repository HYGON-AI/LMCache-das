# SPDX-License-Identifier: Apache-2.0
"""Tests LMCache-HCU memory management behavior.

The HCU memory management patch preserves parent allocator ownership on upstream
memory objects and registers HyFile buffers for XDS access.
"""
from __future__ import annotations

# Standard
from types import ModuleType, SimpleNamespace
from unittest.mock import MagicMock
import sys

# First Party
import lmcache.v1 as upstream_v1
from lmcache_hcu.v1 import memory_management as hcu_mm


class _MemoryObj:
    def __init__(self):
        self.parent_allocator = None


def test_attach_parent_allocator_updates_memory_obj_parent():
    """_attach_parent_allocator should set parent_allocator when the field exists."""
    memory_obj = _MemoryObj()
    parent = object()

    result = hcu_mm._attach_parent_allocator(memory_obj, parent)

    assert result is memory_obj
    assert memory_obj.parent_allocator is parent


def test_attach_parent_allocator_ignores_none_and_objects_without_field():
    """_attach_parent_allocator should safely handle None and foreign objects."""
    foreign = SimpleNamespace()
    parent = object()

    assert hcu_mm._attach_parent_allocator(None, parent) is None
    assert hcu_mm._attach_parent_allocator(foreign, parent) is foreign
    assert not hasattr(foreign, "parent_allocator")


def test_attach_parent_allocator_batch_updates_each_memory_obj():
    """_attach_parent_allocator_batch should attach the parent to every object."""
    memory_objs = [_MemoryObj(), _MemoryObj()]
    parent = object()

    result = hcu_mm._attach_parent_allocator_batch(memory_objs, parent)

    assert result is memory_objs
    assert all(memory_obj.parent_allocator is parent for memory_obj in memory_objs)


def test_attach_parent_allocator_batch_preserves_none():
    """_attach_parent_allocator_batch should return None for a None batch."""
    assert hcu_mm._attach_parent_allocator_batch(None, object()) is None


def test_patch_parent_allocator_semantics_patches_upstream_allocators(monkeypatch):
    """The parent allocator patch should attach ownership to allocated objects."""
    fake_mm = ModuleType("lmcache.v1.memory_management")

    class AddressManager:
        ALIGN_BYTES = 4096

        def __init__(self, size, align_bytes=ALIGN_BYTES):
            self.size = size
            self.align_bytes = align_bytes

    class TensorMemoryAllocator:
        def __init__(self, tensor, align_bytes=4096, init_address_space=None):
            self.tensor = tensor
            self.buffer = tensor
            self.align_bytes = align_bytes
            self.init_address_space = init_address_space
            self.address_manager = AddressManager(tensor.numel(), align_bytes)

        def allocate(self):
            return _MemoryObj()

        def batched_allocate(self):
            return [_MemoryObj(), _MemoryObj()]

    class GPUMemoryAllocator:
        def __init__(self, size, device="cuda", align_bytes=None, use_paging=False, **kwargs):
            self.size = size
            self.device = device
            self.tensor = SimpleNamespace(numel=lambda: size)
            self.allocator = None

    fake_mm.AddressManager = AddressManager
    fake_mm.TensorMemoryAllocator = TensorMemoryAllocator
    fake_mm.GPUMemoryAllocator = GPUMemoryAllocator
    monkeypatch.setitem(sys.modules, "lmcache.v1.memory_management", fake_mm)
    monkeypatch.setattr(upstream_v1, "memory_management", fake_mm)
    monkeypatch.setattr(hcu_mm, "_PATCHED_PARENT_ALLOCATOR_SEMANTICS", False)

    hcu_mm.patch_parent_allocator_semantics()

    parent = object()
    address_manager = fake_mm.AddressManager(1024, parent_allocator=parent)
    tensor_allocator = fake_mm.TensorMemoryAllocator(
        SimpleNamespace(numel=lambda: 2048), parent_allocator=parent
    )
    allocated = tensor_allocator.allocate()
    batch = tensor_allocator.batched_allocate()
    gpu_allocator = fake_mm.GPUMemoryAllocator(4096, use_paging=False)

    assert address_manager.parent_allocator is parent
    assert tensor_allocator.parent_allocator is parent
    assert tensor_allocator.address_manager.parent_allocator is parent
    assert allocated.parent_allocator is parent
    assert all(memory_obj.parent_allocator is parent for memory_obj in batch)
    assert gpu_allocator.allocator.parent_allocator is gpu_allocator
    assert hcu_mm._PATCHED_PARENT_ALLOCATOR_SEMANTICS is True


def test_hyfile_memory_allocator_registers_global_buffer(monkeypatch):
    """HyFileMemoryAllocator should initialize HIPFile and register its tensor buffer."""
    tensor = SimpleNamespace(data_ptr=MagicMock(return_value=123456))
    calls = []

    def fake_gpu_init(self, size, device="cuda", align_bytes=None):
        self.size = size
        self.device = device
        self.align_bytes = align_bytes
        self.tensor = tensor

    monkeypatch.setattr(hcu_mm, "patch_parent_allocator_semantics", MagicMock())
    monkeypatch.setattr(hcu_mm.GPUMemoryAllocator, "__init__", fake_gpu_init)
    monkeypatch.setattr(hcu_mm.torch.cuda, "current_device", lambda: 2)
    monkeypatch.setattr(hcu_mm.HIPFile, "init_driver", lambda: calls.append("init"))
    monkeypatch.setattr(
        hcu_mm.HIPFile,
        "register_global_buffer",
        lambda pointer, size: calls.append(("register", pointer, size)),
    )
    monkeypatch.setattr(
        hcu_mm.HIPFile,
        "deregister_global_buffer",
        lambda pointer: calls.append(("deregister", pointer)),
    )
    monkeypatch.setattr(hcu_mm.HIPFile, "close_driver", lambda: calls.append("close"))

    allocator = hcu_mm.HyFileMemoryAllocator(4096)

    assert allocator.device == "cuda:2"
    assert allocator.align_bytes == 4096
    assert allocator.base_pointer == 123456
    assert calls == ["init", ("register", 123456, 4096)]

    allocator.__del__()

    assert calls[-2:] == [("deregister", 123456), "close"]
    hcu_mm.patch_parent_allocator_semantics.assert_called_once_with()

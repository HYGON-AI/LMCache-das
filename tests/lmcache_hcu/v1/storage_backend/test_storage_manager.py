# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Tests LMCache-HCU storage manager runtime patches.

The HCU runtime patch extends StorageManager.touch_cache so XdsBackend is touched
alongside LocalCPUBackend and LocalDiskBackend, while unrelated backends are not.
"""
from __future__ import annotations

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# First Party
import lmcache_hcu  # noqa: F401  # Importing applies LMCache-HCU runtime patches.
import lmcache.v1.storage_backend.storage_manager as storage_manager
from lmcache.v1.storage_backend.storage_manager import StorageManager


class _FakeStream:
    def synchronize(self):
        pass


class _FakeCudaStreamContext:
    def __enter__(self):
        return None

    def __exit__(self, exc_type, exc, tb):
        return False


class _FakeAllocator:
    def __init__(self, existing_keys):
        self.existing_keys = set(existing_keys)

    def contains(self, key):
        return key in self.existing_keys

    def allocate(self, shape, dtype, fmt=None, eviction=True, busy_loop=False):
        return _FakeMemoryObj(f"allocated-{shape}", fmt=fmt)


class _FakeTensor:
    def copy_(self, tensor, non_blocking=True):
        self.copied_from = tensor


class _FakeMemoryObj:
    def __init__(self, name, fmt="fmt"):
        self.name = name
        self.tensor = _FakeTensor()
        self.meta = SimpleNamespace(fmt=fmt)

    def get_shape(self):
        return self.name

    def get_dtype(self):
        return "dtype"

    def ref_count_down(self):
        pass


def test_touch_cache_touches_local_cpu_local_disk_and_xds_only():
    """The patched touch_cache should include XdsBackend and skip unrelated backends."""
    local_cpu = SimpleNamespace(touch_cache=MagicMock())
    local_disk = SimpleNamespace(touch_cache=MagicMock())
    xds = SimpleNamespace(touch_cache=MagicMock())
    remote = SimpleNamespace(touch_cache=MagicMock())
    manager = SimpleNamespace(
        storage_backends={
            "LocalCPUBackend": local_cpu,
            "LocalDiskBackend": local_disk,
            "XdsBackend": xds,
            "RemoteBackend": remote,
        }
    )

    StorageManager.touch_cache(manager)

    local_cpu.touch_cache.assert_called_once_with()
    local_disk.touch_cache.assert_called_once_with()
    xds.touch_cache.assert_called_once_with()
    remote.touch_cache.assert_not_called()


def test_allocate_and_copy_objects_returns_actual_allocated_keys(monkeypatch):
    """Skipped existing keys must not be returned for newly allocated objects."""
    monkeypatch.setattr(
        storage_manager.torch.cuda,
        "stream",
        lambda stream: _FakeCudaStreamContext(),
    )
    keys = ["existing-a", "existing-b", "new-c", "new-d"]
    src_memory_objs = [
        _FakeMemoryObj("src-a"),
        _FakeMemoryObj("src-b"),
        _FakeMemoryObj("src-c"),
        _FakeMemoryObj("src-d"),
    ]

    allocated_keys, allocated_objects = storage_manager.allocate_and_copy_objects(
        _FakeAllocator({"existing-a", "existing-b"}),
        keys,
        src_memory_objs,
        _FakeStream(),
    )

    assert allocated_keys == ["new-c", "new-d"]
    assert [obj.name for obj in allocated_objects] == [
        "allocated-src-c",
        "allocated-src-d",
    ]


def test_allocate_and_copy_objects_handles_middle_skipped_key(monkeypatch):
    """Allocated keys stay aligned when an existing key is skipped in the middle."""
    monkeypatch.setattr(
        storage_manager.torch.cuda,
        "stream",
        lambda stream: _FakeCudaStreamContext(),
    )
    keys = ["new-a", "existing-b", "new-c"]
    src_memory_objs = [
        _FakeMemoryObj("src-a"),
        _FakeMemoryObj("src-b"),
        _FakeMemoryObj("src-c"),
    ]

    allocated_keys, allocated_objects = storage_manager.allocate_and_copy_objects(
        _FakeAllocator({"existing-b"}),
        keys,
        src_memory_objs,
        _FakeStream(),
    )

    assert allocated_keys == ["new-a", "new-c"]
    assert [obj.name for obj in allocated_objects] == [
        "allocated-src-a",
        "allocated-src-c",
    ]

# SPDX-License-Identifier: Apache-2.0
"""Tests LMCache-HCU local disk backend behavior.

The HCU local disk backend keeps cached_positions metadata when loading objects
from disk and removes stale metadata when a cached file is missing.
"""
from __future__ import annotations

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# First Party
import lmcache_hcu  # noqa: F401  # Importing applies LMCache-HCU runtime patches.
from lmcache.v1.storage_backend.local_disk_backend import LocalDiskBackend


class _Lock:
    """Small lock helper that records acquire/release ordering."""

    def __init__(self):
        self.calls = []

    def acquire(self):
        self.calls.append("acquire")

    def release(self):
        self.calls.append("release")


def test_get_blocking_preserves_cached_positions_on_loaded_memory():
    """get_blocking should copy disk metadata cached_positions to the memory object."""
    key = "cache-key"
    cached_positions = [1, 3, 5]
    memory_obj = SimpleNamespace(metadata=SimpleNamespace(cached_positions=None))
    backend = SimpleNamespace(
        disk_lock=_Lock(),
        dict={
            key: SimpleNamespace(
                path="/tmp/cache.bin",
                dtype="float16",
                shape=(2, 4),
                fmt="kv",
                cached_positions=cached_positions,
            )
        },
        cache_policy=SimpleNamespace(update_on_hit=MagicMock()),
        load_bytes_from_disk=MagicMock(return_value=memory_obj),
    )

    result = LocalDiskBackend.get_blocking(backend, key)

    assert result is memory_obj
    assert memory_obj.metadata.cached_positions == cached_positions
    backend.cache_policy.update_on_hit.assert_called_once_with(key, backend.dict)
    backend.load_bytes_from_disk.assert_called_once_with(
        key, "/tmp/cache.bin", dtype="float16", shape=(2, 4), fmt="kv"
    )
    assert backend.disk_lock.calls == ["acquire", "release"]


def test_get_blocking_returns_none_for_missing_key():
    """get_blocking should return None and release the lock when metadata is absent."""
    backend = SimpleNamespace(
        disk_lock=_Lock(),
        dict={},
        cache_policy=SimpleNamespace(update_on_hit=MagicMock()),
        load_bytes_from_disk=MagicMock(),
    )

    result = LocalDiskBackend.get_blocking(backend, "missing-key")

    assert result is None
    backend.cache_policy.update_on_hit.assert_not_called()
    backend.load_bytes_from_disk.assert_not_called()
    assert backend.disk_lock.calls == ["acquire", "release"]


def test_load_bytes_from_disk_returns_allocated_memory_when_read_succeeds():
    """load_bytes_from_disk should allocate CPU memory after read success."""
    memory_obj = SimpleNamespace(byte_array=bytearray(4))
    backend = SimpleNamespace(
        local_cpu_backend=SimpleNamespace(allocate=MagicMock(return_value=memory_obj)),
        read_file=MagicMock(return_value=True),
    )

    result = LocalDiskBackend.load_bytes_from_disk(
        backend, "cache-key", "/tmp/cache.bin", "float16", (2, 4), "kv"
    )

    assert result is memory_obj
    backend.local_cpu_backend.allocate.assert_called_once_with((2, 4), "float16", "kv")
    backend.read_file.assert_called_once_with(
        "cache-key", memory_obj.byte_array, "/tmp/cache.bin"
    )


def test_load_bytes_from_disk_returns_none_when_read_fails():
    """load_bytes_from_disk should return None when the file read fails."""
    memory_obj = SimpleNamespace(byte_array=bytearray(4))
    backend = SimpleNamespace(
        local_cpu_backend=SimpleNamespace(allocate=MagicMock(return_value=memory_obj)),
        read_file=MagicMock(return_value=False),
    )

    result = LocalDiskBackend.load_bytes_from_disk(
        backend, "cache-key", "/tmp/cache.bin", "float16", (2, 4), "kv"
    )

    assert result is None


def test_read_file_removes_stale_metadata_when_file_is_missing(tmp_path):
    """read_file should remove dictionary metadata when the on-disk file is gone."""
    key = "cache-key"
    missing_path = tmp_path / "missing.bin"
    backend = SimpleNamespace(
        os_disk_bs=4096,
        use_odirect=False,
        dict={key: SimpleNamespace(path=str(missing_path))},
    )

    result = LocalDiskBackend.read_file(backend, key, bytearray(4), str(missing_path))

    assert result is False
    assert key not in backend.dict

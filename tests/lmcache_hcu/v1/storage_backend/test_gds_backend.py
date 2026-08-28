# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Tests LMCache-HCU GDS backend behavior.

The HCU GDS backend subclasses upstream GdsBackend and customizes metadata reads
with O_NOATIME fallback plus allocation-assisted disk loading.
"""
from __future__ import annotations

# Standard
from types import SimpleNamespace
from unittest.mock import MagicMock

# First Party
import lmcache_hcu  # noqa: F401  # Importing applies LMCache-HCU runtime patches.
from lmcache.v1.storage_backend import gds_backend as patched_gds


class _Lock:
    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, traceback):
        return False


def test_read_metadata_info_falls_back_when_noatime_open_fails(monkeypatch):
    """_read_metadata_info should disable O_NOATIME and retry normal open."""
    calls = []

    def fake_open(filename, flags):
        calls.append((filename, flags))
        if flags & getattr(patched_gds.os, "O_NOATIME", 0):
            raise OSError("O_NOATIME is not supported")
        return 7

    monkeypatch.setattr(patched_gds.os, "open", fake_open)
    monkeypatch.setattr(patched_gds.os, "read", lambda fd, size: b"metadata")
    monkeypatch.setattr(patched_gds.os, "close", lambda fd: calls.append(("close", fd)))
    monkeypatch.setattr(
        patched_gds, "unpack_metadata", MagicMock(return_value="decoded")
    )
    backend = SimpleNamespace(_use_noatime=True)

    result = patched_gds.GdsBackend._read_metadata_info(backend, "/tmp/meta")

    assert result == "decoded"
    assert backend._use_noatime is False
    assert calls[0][1] & getattr(patched_gds.os, "O_NOATIME", 0)
    assert calls[1] == ("/tmp/meta", patched_gds.os.O_RDONLY)
    assert calls[-1] == ("close", 7)
    patched_gds.unpack_metadata.assert_called_once_with(b"metadata")


def test_read_metadata_info_uses_normal_open_after_noatime_disabled(monkeypatch):
    """Once disabled, _read_metadata_info should not try O_NOATIME again."""
    calls = []
    monkeypatch.setattr(
        patched_gds.os,
        "open",
        lambda filename, flags: calls.append((filename, flags)) or 8,
    )
    monkeypatch.setattr(patched_gds.os, "read", lambda fd, size: b"metadata")
    monkeypatch.setattr(patched_gds.os, "close", lambda fd: None)
    monkeypatch.setattr(
        patched_gds, "unpack_metadata", MagicMock(return_value="decoded")
    )
    backend = SimpleNamespace(_use_noatime=False)

    result = patched_gds.GdsBackend._read_metadata_info(backend, "/tmp/meta")

    assert result == "decoded"
    assert calls == [("/tmp/meta", patched_gds.os.O_RDONLY)]


def test_read_metadata_stores_decoded_metadata(monkeypatch):
    """_read_metadata should decode metadata and update hot cache under the lock."""
    key = "cache-key"
    backend = SimpleNamespace(
        _read_metadata_info=MagicMock(
            return_value=((2, 4), "float16", 4096, "kv", {"lmcache_version": "1"})
        ),
        hot_lock=_Lock(),
        metadata_dirs=set(),
        hot_cache={},
    )
    monkeypatch.setattr(patched_gds, "_METADATA_VERSION", 1)
    monkeypatch.setattr(patched_gds, "_METADATA_FILE_SUFFIX", ".metadata")

    metadata = patched_gds.GdsBackend._read_metadata(
        backend,
        key,
        "/tmp/cache-file.metadata",
        "subdir-key",
    )

    assert metadata.path == "/tmp/cache-file"
    assert metadata.size == 4096
    assert metadata.shape == (2, 4)
    assert metadata.dtype == "float16"
    assert metadata.cached_positions is None
    assert backend.metadata_dirs == {"subdir-key"}
    assert backend.hot_cache[key] is metadata


def test_read_metadata_rejects_unknown_metadata_version(monkeypatch):
    """_read_metadata should fail when the metadata version does not match."""
    backend = SimpleNamespace(
        _read_metadata_info=MagicMock(
            return_value=((2, 4), "float16", 4096, "kv", {"lmcache_version": "old"})
        )
    )
    monkeypatch.setattr(patched_gds, "_METADATA_VERSION", 1)

    try:
        patched_gds.GdsBackend._read_metadata(
            backend, "key", "/tmp/file.metadata", "subdir"
        )
    except RuntimeError as exc:
        assert "unhandled lmcache metadata" in str(exc)
    else:
        raise AssertionError("Expected metadata version mismatch to raise RuntimeError")


def test_load_bytes_from_disk_with_allocation_loads_into_allocated_memory():
    """_load_bytes_from_disk_with_allocation should allocate then load into memory."""
    memory_obj = SimpleNamespace(tensor=None)
    backend = SimpleNamespace(
        memory_allocator=SimpleNamespace(allocate=MagicMock(return_value=memory_obj)),
        _load_bytes_from_disk_with_memory=MagicMock(return_value=memory_obj),
        _debug_asserts=False,
    )

    result = patched_gds.GdsBackend._load_bytes_from_disk_with_allocation(
        backend, "cache-key", "/tmp/data", "float16", (2, 4), "kv"
    )

    assert result is memory_obj
    backend.memory_allocator.allocate.assert_called_once_with(
        (2, 4), "float16", fmt="kv"
    )
    backend._load_bytes_from_disk_with_memory.assert_called_once_with(
        "cache-key", "/tmp/data", memory_obj
    )


def test_load_bytes_from_disk_with_allocation_returns_none_when_allocation_fails():
    """_load_bytes_from_disk_with_allocation should stop when allocation fails."""
    backend = SimpleNamespace(
        memory_allocator=SimpleNamespace(allocate=MagicMock(return_value=None)),
        _load_bytes_from_disk_with_memory=MagicMock(),
        _debug_asserts=False,
    )

    result = patched_gds.GdsBackend._load_bytes_from_disk_with_allocation(
        backend, "cache-key", "/tmp/data", "float16", (2, 4), "kv"
    )

    assert result is None
    backend._load_bytes_from_disk_with_memory.assert_not_called()

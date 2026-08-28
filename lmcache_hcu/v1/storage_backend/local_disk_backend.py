# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

import os

from lmcache.logging import init_logger

logger = init_logger(__name__)


def _hcu_get_blocking(self, key):
    """Blocking get function."""
    self.disk_lock.acquire()
    if key not in self.dict:
        self.disk_lock.release()
        return None

    # Update cache recency
    self.cache_policy.update_on_hit(key, self.dict)

    disk_meta = self.dict[key]
    path = disk_meta.path
    dtype = disk_meta.dtype
    shape = disk_meta.shape
    fmt = disk_meta.fmt
    cached_positions = disk_meta.cached_positions

    assert dtype is not None
    assert shape is not None

    self.disk_lock.release()
    memory_obj = self.load_bytes_from_disk(
        key, path, dtype=dtype, shape=shape, fmt=fmt
    )
    if memory_obj is not None:
        memory_obj.metadata.cached_positions = cached_positions

    return memory_obj


def _hcu_load_bytes_from_disk(self, key, path, dtype, shape, fmt):
    """Load bytearray from disk."""
    memory_obj = self.local_cpu_backend.allocate(shape, dtype, fmt)
    if memory_obj is None:
        logger.error("Memory allocation failed for key %s", key)
        return None

    buffer = memory_obj.byte_array
    if self.read_file(key, buffer, path):
        return memory_obj
    return None


def _hcu_read_file(self, key, buffer, path):
    size = len(buffer)
    fblock_aligned = size % self.os_disk_bs == 0
    if not fblock_aligned and self.use_odirect:
        logger.warning(
            "Cannot use O_DIRECT for this file, "
            "size is not aligned to disk block size."
        )

    try:
        if not fblock_aligned or not self.use_odirect:
            with open(path, "rb") as f:
                f.readinto(buffer)
        else:
            fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
            with os.fdopen(fd, "rb", buffering=0) as fdo:
                fdo.readinto(buffer)
        return True
    except FileNotFoundError:
        logger.warning(f"File not found on disk: {path}")
        if self.dict.get(key, None):
            self.dict.pop(key)
        return False


def patch_local_disk_backend(local_disk_backend) -> None:
    backend_cls = local_disk_backend.LocalDiskBackend
    if getattr(backend_cls, "_lmcache_hcu_local_disk_patched", False):
        return

    _hcu_get_blocking._lmcache_hcu_patched = True
    _hcu_load_bytes_from_disk._lmcache_hcu_patched = True
    _hcu_read_file._lmcache_hcu_patched = True

    backend_cls.get_blocking = _hcu_get_blocking
    backend_cls.load_bytes_from_disk = _hcu_load_bytes_from_disk
    backend_cls.read_file = _hcu_read_file
    backend_cls._lmcache_hcu_local_disk_patched = True

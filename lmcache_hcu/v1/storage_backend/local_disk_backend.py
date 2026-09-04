# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

import os
import time

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
            try:
                with os.fdopen(fd, "rb", buffering=0) as fdo:
                    fdo.readinto(buffer)
            except Exception:
                try:
                    os.close(fd)
                except OSError:
                    pass
                raise
    except FileNotFoundError:
        logger.warning(f"File not found on disk: {path}")
        with self.disk_lock:
            if self.dict.get(key, None):
                self.dict.pop(key)
        return False
    except PermissionError:
        logger.warning(
            f"Failed to read from disk (permission denied): {path}"
        )
        with self.disk_lock:
            if self.dict.get(key, None):
                self.dict.pop(key)
        return False
    except IsADirectoryError:
        logger.warning(
            f"Failed to read from disk (path is a directory): {path}"
        )
        with self.disk_lock:
            if self.dict.get(key, None):
                self.dict.pop(key)
        return False
    except OSError as e:
        logger.warning(
            f"Failed to read from disk (OS error: {e}): {path}"
        )
        with self.disk_lock:
            if self.dict.get(key, None):
                self.dict.pop(key)
        return False
    except Exception as e:
        logger.warning(
            f"Unexpected error reading from disk "
            f"({type(e).__name__}: {e}): {path}"
        )
        with self.disk_lock:
            if self.dict.get(key, None):
                self.dict.pop(key)
        return False
    return True


def _hcu_write_file(self, buffer, path) -> bool:
    start_time = time.time()
    size = len(buffer)
    try:
        if size % self.os_disk_bs != 0 or not self.use_odirect:
            with open(path, "wb") as f:
                f.write(buffer)
        else:
            fd = os.open(
                path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644
            )
            try:
                os.write(fd, buffer)
            finally:
                os.close(fd)
    except FileNotFoundError:
        logger.warning(
            f"Failed to write to disk (parent dir missing): {path}"
        )
        return False
    except PermissionError:
        logger.warning(
            f"Failed to write to disk (permission denied): {path}"
        )
        _hcu_cleanup_partial_file(self, path)
        return False
    except OSError as e:
        logger.warning(
            f"Failed to write to disk (OS error: {e}): {path}"
        )
        _hcu_cleanup_partial_file(self, path)
        return False
    except Exception as e:
        logger.warning(
            f"Unexpected error writing to disk ({type(e).__name__}: "
            f"{e}): {path}"
        )
        _hcu_cleanup_partial_file(self, path)
        return False
    disk_write_time = time.time() - start_time
    logger.debug(
        f"Disk write size: {size} bytes, "
        f"Bandwidth: {size / disk_write_time / 1e6:.2f} MB/s"
    )
    return True


def _hcu_cleanup_partial_file(self, path: str) -> None:
    """
    Best-effort removal of a partially-written file at ``path``.

    Used after a failed ``write_file`` so that a truncated/corrupt
    file does not remain on disk (and later cause a read failure on
    the same key). All errors here are swallowed and logged at
    ``warning`` level because this is a cleanup step, not part of
    the critical write path -- it must never crash the vLLM loop.
    """
    try:
        os.remove(path)
        logger.debug(f"Cleaned up partial file on disk: {path}")
    except FileNotFoundError:
        # Nothing to do -- the file was never created.
        pass
    except PermissionError as e:
        logger.warning(
            f"Could not clean up partial file (permission denied): "
            f"{path} ({e})"
        )
    except OSError as e:
        logger.warning(
            f"Could not clean up partial file (OS error: {e}): {path}"
        )
    except Exception as e:
        logger.warning(
            f"Could not clean up partial file "
            f"({type(e).__name__}: {e}): {path}"
        )


def _hcu_async_save_bytes_to_disk(
    self,
    key,
    memory_obj,
    on_complete_callback=None,
):
    """
    Convert KV to bytes and async store bytes to disk.

    :param on_complete_callback: Optional callback invoked after the disk
        write completes for this key. Callback exceptions are caught and
        logged.
    """
    kv_chunk = memory_obj.tensor
    assert kv_chunk is not None
    buffer = memory_obj.byte_array
    path = self._key_to_path(key)
    size = len(buffer)
    self.usage += size
    self.stats_monitor.update_local_storage_usage(self.usage)

    # TODO(Jiayi): need to add ref count in disk memory object
    if not self.write_file(buffer, path):
        self.usage -= size
        self.stats_monitor.update_local_storage_usage(self.usage)
        memory_obj.ref_count_down()
        self.disk_worker.remove_put_task(key)
        if on_complete_callback is not None:
            try:
                on_complete_callback(key)
            except Exception as e:
                logger.warning(
                    f"on_complete_callback failed for key {key}: {e}"
                )
        return

    # ref count down here because there's a ref_count_up in
    # `submit_put_task` above.
    # Ref count down better be before `insert_key` for testing
    # purposes (e.g., testing mem_leak).
    # TODO(Jiayi): This could be problematic if the
    # freed memory object is immediately reused.
    size = memory_obj.get_physical_size()
    shape = memory_obj.metadata.shape
    dtype = memory_obj.metadata.dtype
    fmt = memory_obj.metadata.fmt
    cached_positions = memory_obj.metadata.cached_positions
    memory_obj.ref_count_down()

    self.insert_key(
        key, size, shape, dtype, fmt, cached_positions=cached_positions
    )

    self.disk_worker.remove_put_task(key)

    # Call the completion callback if provided
    if on_complete_callback is not None:
        try:
            on_complete_callback(key)
        except Exception as e:
            logger.warning(
                f"on_complete_callback failed for key {key}: {e}"
            )


def _hcu_batched_async_load_bytes_from_disk(
    self, paths, keys, memory_objs, write_back=False,
):
    # NOTE(Jiayi): This is a HCU monkey-patch that mirrors the upstream
    # `batched_async_load_bytes_from_disk` after commit fa6aa37. The only
    # behavioral change is `if not self.read_file(...): continue` instead
    # of an unconditional call.
    """
    Async load bytearray from disk.
    """

    logger.debug("Executing `async_load_bytes` from disk.")
    # TODO (Jiayi): handle the case where loading fails.
    for path, key, mem_obj in zip(paths, keys, memory_objs, strict=False):
        buffer = mem_obj.byte_array
        if not self.read_file(key, buffer, path):
            continue

        # TODO(Jiayi): Please recover the metadata in a more
        # elegant way in the future.
        cached_positions = self.dict[key].cached_positions
        mem_obj.metadata.cached_positions = cached_positions

        self.disk_lock.acquire()
        self.dict[key].unpin()
        self.disk_lock.release()

    return memory_objs


def patch_local_disk_backend(local_disk_backend) -> None:
    backend_cls = local_disk_backend.LocalDiskBackend
    if getattr(backend_cls, "_lmcache_hcu_local_disk_patched", False):
        return

    _hcu_get_blocking._lmcache_hcu_patched = True
    _hcu_load_bytes_from_disk._lmcache_hcu_patched = True
    _hcu_read_file._lmcache_hcu_patched = True
    _hcu_write_file._lmcache_hcu_patched = True
    _hcu_cleanup_partial_file._lmcache_hcu_patched = True
    _hcu_async_save_bytes_to_disk._lmcache_hcu_patched = True
    _hcu_batched_async_load_bytes_from_disk._lmcache_hcu_patched = True

    backend_cls.get_blocking = _hcu_get_blocking
    backend_cls.load_bytes_from_disk = _hcu_load_bytes_from_disk
    backend_cls.read_file = _hcu_read_file
    backend_cls.write_file = _hcu_write_file
    backend_cls._cleanup_partial_file = _hcu_cleanup_partial_file
    backend_cls.async_save_bytes_to_disk = _hcu_async_save_bytes_to_disk
    backend_cls.batched_async_load_bytes_from_disk = (
        _hcu_batched_async_load_bytes_from_disk
    )
    backend_cls._lmcache_hcu_local_disk_patched = True

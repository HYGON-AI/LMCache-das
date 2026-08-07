# SPDX-License-Identifier: Apache-2.0
# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
from __future__ import annotations

# Standard
import os

# First Party
import lmcache.v1.storage_backend.gds_backend as _baseline_gds_backend
from lmcache.v1.storage_backend.gds_backend import *  # noqa: F403

_BaseGdsBackend = _baseline_gds_backend.GdsBackend
_METADATA_FILE_SUFFIX = _baseline_gds_backend._METADATA_FILE_SUFFIX
_DATA_FILE_SUFFIX = _baseline_gds_backend._DATA_FILE_SUFFIX
_METADATA_VERSION = _baseline_gds_backend._METADATA_VERSION
_METADATA_MAX_SIZE = _baseline_gds_backend._METADATA_MAX_SIZE


class GdsBackend(_BaseGdsBackend):
    """
    Originally based on the open sourced WekaGdsBackend, this is a backend that
    leverages NVIDIA's cuFile API to issue GDS requests directly to the
    GDS-supported remote filesystem.  In order to use it, users need to specify
    `gds_path` and `cufile_buffer_size` in their LMCache config.

    Cache Directory Structure created by this Backend:
    /{gds_path}/{first_level}/{second_level}/{data & metadata} This structure
    is semi-arbitrary. We create two levels in the directory hierarchy to
    parallelize loading the data during initialization in the Python code.

    NOTE: If GPUDirect is not supported on that other filesystem, then CuFile will
    fall back to POSIX I/O.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,  # noqa: F405
        metadata: LMCacheMetadata,  # noqa: F405
        loop: asyncio.AbstractEventLoop,  # noqa: F405
        dst_device: str = "cuda",
    ):
        super().__init__(
            config=config,
            metadata=metadata,
            loop=loop,
            dst_device=dst_device,
        )
        # Flag for extra assertions that catch bugs but hurt performance.
        self._debug_asserts = False
        # Use O_NOATIME during metadata reads for a small performance gain.
        self._use_noatime = True

    def _read_metadata_info(self, filename: str):
        # Use O_NOATIME to prevent updating access time and improve performance
        # Instead of using Python's open() and read(), we use the OS's open() and
        # read() because it is faster - the metadata file is small and we don't
        # need any buffering.
        # Additionally, we use O_NOATIME to improve performance
        if self._use_noatime:
            try:
                fd = os.open(filename, os.O_RDONLY | os.O_NOATIME)
            except (
                # PermissionError: User doesn't own the file
                # AttributeError: O_NOATIME not available on this platform
                # OSError: Filesystem doesn't support O_NOATIME (EINVAL)
                PermissionError,
                AttributeError,
                OSError,
            ):  # fallback to normal open if O_NOATIME is not supportedExpand commentComment on lines R365 to R372Resolved
                self._use_noatime = False
                logger.info(
                    "O_NOATIME flag not supported during metadata file read, "
                    "falling back to normal open"
                )
                fd = os.open(filename, os.O_RDONLY)
        else:
            fd = os.open(filename, os.O_RDONLY)
        try:
            buf = os.read(fd, _METADATA_MAX_SIZE) # noqa: F405
        finally:
            os.close(fd)
        return unpack_metadata(buf) # noqa: F405

    def _read_metadata(self, key, filename, subdir_key):
        shape, dtype, size, fmt, extra_metadata = self._read_metadata_info(filename)

        if extra_metadata["lmcache_version"] != str(_METADATA_VERSION):
            raise RuntimeError("unhandled lmcache metadata")

        logger.debug(
            f"Read metadata for {key} from {filename}: "
            f"shape={shape}, dtype={dtype}, size={size}, fmt={fmt}, "
            f"extra_metadata={extra_metadata}"
        )
        # TODO(extra_metadata)
        # TODO(Jiayi): need to support `cached_positions`.
        # Currently we just fill it as None.
        metadata = DiskCacheMetadata(
            filename.removesuffix(_METADATA_FILE_SUFFIX),
            size,
            shape,
            dtype,
            None,
            fmt,
        )
        with self.hot_lock:
            self.metadata_dirs.add(subdir_key)
            self.hot_cache[key] = metadata
        return metadata

    def _load_bytes_from_disk_with_allocation(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat,
    ) -> Optional[MemoryObj]:
        """
        Load byte array from disk by first allocating memory, then loading.

        Args:
            key: Cache key for error handling
            path: File path to load from
            dtype: Data type for memory allocation
            shape: Shape for memory allocation

        Returns:
            A new memory object with loaded data, or None if allocation or
            loading failed
        """
        memory_obj = self.memory_allocator.allocate(shape, dtype, fmt=fmt)
        if memory_obj is None:
            logger.debug("Memory allocation failed during sync disk load.")
            return None
        if self._debug_asserts:
            assert memory_obj.tensor is not None
            assert memory_obj.tensor.is_cuda
            assert torch.device(self.dst_device) == torch.device(memory_obj.tensor.device)

        return self._load_bytes_from_disk_with_memory(key, path, memory_obj)
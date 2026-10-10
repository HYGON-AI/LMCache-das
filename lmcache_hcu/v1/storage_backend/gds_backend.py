# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modifications Copyright 2026 Hygon Information Technology Co., Ltd.
from collections import OrderedDict
from concurrent.futures import Future, ThreadPoolExecutor
from typing import Any, Callable, List, Optional, Sequence, Tuple, Union
from urllib.parse import quote, unquote
import asyncio
import ctypes
import json
import mmap
import os
import random
import string
import struct
import threading
import time

# Third Party
import aiofile
import numpy as np
import torch

# First Party
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, DiskCacheMetadata, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    CuFileMemoryAllocator,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache_hcu.v1.storage_backend.block_first_disk import (
    BLOCK_FIRST_HEADER_SIZE,
    BlockFirstDiskMetadata,
    BlockFirstFormatError,
    align_up,
    block_first_shape,
    block_first_to_logical,
    logical_to_block_first,
    pack_block_first_header,
    unpack_block_first_header,
)
from lmcache_hcu.v1.storage_backend.direct_io import (
    DirectIODescriptor,
    DirectIODirection,
)

logger = init_logger(__name__)

_METADATA_FILE_SUFFIX = ".metadata"
_DATA_FILE_SUFFIX = ".kvcache.safetensors"
_WEKA_DATA_FILE_SUFFIX = ".weka1"
_BLOCK_FIRST_DATA_FILE_SUFFIX = ".block_first_v1.kvcache"
_BLOCK_FIRST_DATA_FILE_SUFFIX_TMP = ".BLOCK_FIRST_V1_TMP"
_METADATA_VERSION = 1
_METADATA_MAX_SIZE = 4096  # reserve 4K for metadata.
# TODO: It is possible to read this 4KB block without triggering read-ahead by
# various means.
_DEFAULT_THREAD_COUNT = 4

class UnsupportedMetadataVersion(Exception):
    pass

torch_dtypes = {
    torch.half: "F16",
    torch.bfloat16: "BF16",
    torch.float32: "F32",
    torch.float64: "F64",
    torch.uint8: "U8",
    torch.uint16: "U16",
    torch.uint32: "U32",
    torch.uint64: "U64",
    torch.int8: "I8",
    torch.int16: "I16",
    torch.int32: "I32",
    torch.int64: "I64",
    torch.float8_e4m3fn: "F8E4M3FN",
    torch.float8_e5m2: "F8E5M2",
}

torch_dtypes_inverse = dict([(v, k) for k, v in torch_dtypes.items()])

def get_fstype(path):
    with open("/proc/mounts", "r") as f:
        lines = f.readlines()

    # Find the best matching mount point
    best_match = ""
    best_fstype = ""
    for line in lines:
        parts = line.split()
        if len(parts) >= 3:
            _, mount_point, fstype = parts[0], parts[1], parts[2]
            if path.startswith(mount_point) and len(mount_point) > len(best_match):
                best_match = mount_point
                best_fstype = fstype

    if not best_fstype:
        raise RuntimeError(f"Unable to detect fstype for {path}")

    return best_fstype

def pack_metadata(tensor, fmt: MemoryFormat, **extra_metadata) -> bytes:
    if tensor.dtype not in torch_dtypes:
        raise RuntimeError(f"unhandled dtype {tensor.dtype}")

    # Metadata
    data_size = tensor.numel() * tensor.element_size()
    tensor_meta = {
        "dtype": torch_dtypes[tensor.dtype],
        "shape": list(tensor.size()),
        "data_offsets": [0, data_size],
        "fmt": fmt.value,
        "__metadata__": extra_metadata,
    }
    meta = {"kvcache": tensor_meta}
    str_meta = json.dumps(meta).encode("utf-8")
    meta_len = len(str_meta)
    assert meta_len <= _METADATA_MAX_SIZE - 8

    # Align to _METADATA_MAX_SIZE - 8
    str_meta += b" " * (_METADATA_MAX_SIZE - 8 - meta_len)

    # Pack it all up so it is sized _METADATA_MAX_SIZE exactly.
    return struct.pack("<Q", len(str_meta)) + str_meta

def unpack_metadata(buffer: bytes):
    meta_len = struct.unpack("<Q", buffer[:8])[0]

    str_meta = buffer[8 : 8 + meta_len]
    json_meta = str_meta.rstrip(b" ")

    meta = json.loads(json_meta.decode("utf-8"))
    tensor_meta = meta["kvcache"]

    shape = tensor_meta["shape"]
    dtype_str = tensor_meta["dtype"]
    data_offsets = tensor_meta["data_offsets"]
    fmt = MemoryFormat(tensor_meta["fmt"])

    nbytes = data_offsets[1] - data_offsets[0]
    dtype = torch_dtypes_inverse[dtype_str]

    return torch.Size(shape), dtype, nbytes, fmt, tensor_meta["__metadata__"]

def rand_suffix(rand, n: int):
    return "".join(
        rand.choice(string.ascii_uppercase + string.digits) for _ in range(n)
    )

async def save_metadata(path: str, tmp: str, metadata: bytes):
    tmp_path = path + tmp
    async with aiofile.async_open(tmp_path, "wb") as f:
        await f.write(metadata)
    os.rename(tmp_path, path)

def get_extra_config_bool(key, config: LMCacheEngineConfig) -> bool | None:
    if config.extra_config is None:
        return None
    value = config.extra_config.get(key, None)
    if value is None:
        return None

    if isinstance(value, str):
        bool_value = value.strip().lower() in {"1", "true", "yes", "on"}
    elif value in [False, True]:
        bool_value = value
    else:
        raise RuntimeError(f"Invalid value `{value}` for `{key}` in extra_config")

    logger.info(f"Getting {key} = {bool_value} from extra_config")
    return bool_value

def _config_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.strip().lower() in {"1", "true", "yes", "on"}
    if isinstance(value, int):
        return value != 0
    return False

class GdsBackend(AllocatorBackendInterface):
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
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        dst_device: str = "cuda",
    ):
        assert dst_device.startswith("cuda")
        super().__init__(dst_device=dst_device)

        self.config = config
        self.metadata = metadata
        self.loop = loop
        self.direct_only = get_extra_config_bool("gds_direct_only", config) is True
        self.memory_allocator = (
            None if self.direct_only else self.initialize_allocator(config, metadata)
        )
        self.dst_device = dst_device
        self._block_first_staging_allocator: Optional[CuFileMemoryAllocator] = None
        self._block_first_staging_obj: Optional[MemoryObj] = None
        self._block_first_lock = threading.Lock()
        self._pending_put_futures: set[Future] = set()
        self._pending_put_futures_lock = threading.Lock()
        self._direct_registration_lock = threading.Lock()
        self._direct_registrations: dict[tuple[str, int, int], object] = {}
        self._direct_register = None
        self._direct_deregister = None
        self._direct_edge_lock = threading.Lock()
        self._direct_edge_allocator: Optional[CuFileMemoryAllocator] = None
        self._direct_edge_obj: Optional[MemoryObj] = None
        self._direct_cudart = None
        self._direct_memcpy = None
        self._closed = False

        assert config.gds_path is not None, "Need to specify gds_path for GdsBackend"
        self.gds_path = config.gds_path
        self.fstype = get_fstype(config.gds_path)

        # Log the fstype - this is useful in reports and varying optimizations
        # based on the kind of fstype used.
        logger.info(
            f"GDS backend using fstype '{self.fstype}' on path '{self.gds_path}'"
        )

        self.use_cufile = True
        use_cufile_from_config = False

        if config.extra_config is not None:
            use_cufile = get_extra_config_bool("use_cufile", config)
            if use_cufile is not None:
                self.use_cufile = use_cufile
                use_cufile_from_config = True

        self.data_suffix = _DATA_FILE_SUFFIX
        self.use_thread_pool = False
        self._thread_pool = None

        if self.fstype in ["tmpfs", "overlayfs"]:
            # TODO: we can replace the auto-detection of unsupported cufile
            # file systems by doing a small cufile API test on them. If as
            # read/write test fails, we can fallback to not using cufile APIs.
            if use_cufile_from_config:
                logger.warning("No automatic disabling of cufile usage due to fstype")
            else:
                logger.info("Automatic disabling of cufile usage due to fstype")
                self.use_cufile = False
        elif self.fstype == "wekafs":
            logger.info("Weka filesystem detected, cufile usage is enforced")
            assert self.use_cufile
            self.data_suffix = _WEKA_DATA_FILE_SUFFIX
            self.use_thread_pool = True

        if self.use_thread_pool:
            thread_count = _DEFAULT_THREAD_COUNT
            if config.extra_config is not None:
                thread_count = config.extra_config.get(
                    "gds_io_threads", _DEFAULT_THREAD_COUNT
                )
            self._thread_pool = ThreadPoolExecutor(
                max_workers=thread_count, thread_name_prefix="weka-gds-io"
            )

        if self.use_cufile:
            logger.info("Using cufile")
            # HACK(Jiayi): cufile import is buggy on some hardware
            # (e.g., without GPUDirect), so it's temporarily put here.
            # Third Party
            import cufile

            self.cudart = None
            self.cufile = cufile
            self._cufile_driver = self.cufile.CuFileDriver()
        else:
            logger.info("Not using cufile")
            self.cufile = None
            self.cudart = ctypes.CDLL("libcudart.so")

        self.use_direct_io = False

        if config.extra_config is not None:
            use_direct_io = get_extra_config_bool("use_direct_io", config)
            if use_direct_io is not None:
                self.use_direct_io = use_direct_io

        if not os.path.exists(self.gds_path):
            os.makedirs(self.gds_path, exist_ok=True)

        self.block_first_enabled = self._should_enable_block_first(metadata)
        enable_direct = get_extra_config_bool("enable_direct_gds", config) is True
        self.direct_enabled = self.block_first_enabled and (
            enable_direct or self.direct_only
        )
        if self.direct_only and not self.direct_enabled:
            raise RuntimeError(
                "gds_direct_only requires a compatible block-first cuFile setup"
            )
        if self.direct_only and config.store_location != self.__class__.__name__:
            raise RuntimeError("gds_direct_only requires store_location='GdsBackend'")
        if self.direct_only and config.enable_async_loading:
            raise RuntimeError("gds_direct_only does not support async loading")
        if self.direct_enabled:
            from cufile.bindings import cuFileBufDeregister, cuFileBufRegister

            self._direct_register = cuFileBufRegister
            self._direct_deregister = cuFileBufDeregister
            logger.info("In-process direct GDS enabled")
        if self.block_first_enabled and not self.direct_only:
            try:
                self._initialize_block_first_staging(metadata)
            except BaseException:
                if self.memory_allocator is not None:
                    self.memory_allocator.close()
                raise

        self.stats = None  # TODO: plug into LMCache Statistics

        self.hot_lock = threading.Lock()
        self.hot_cache: OrderedDict[CacheEngineKey, DiskCacheMetadata] = OrderedDict()
        self.metadata_dirs: set[str] = set()

        self.put_lock = threading.Lock()
        self.put_tasks: set[CacheEngineKey] = set()

        self.rand = random.Random(self.dst_device)

        if hasattr(self.memory_allocator, "base_pointer"):
            logger.debug(f"Using base pointer {self.memory_allocator.base_pointer}")
            self.cufile_base_pointer = self.memory_allocator.base_pointer
        else:
            logger.info("No base pointer found, cufile will use bounce buffers")
            self.cufile_base_pointer = None
        self._scan_future = asyncio.run_coroutine_threadsafe(
            self._scan_metadata(), self.loop
        )
        self.save_metadata_tasks: set[asyncio.Task] = set()

        # flag to use O_NOATIME during metadata file read for performance improvement
        self._use_noatime = True

    def _should_enable_block_first(self, metadata: LMCacheMetadata) -> bool:
        extra = getattr(metadata, "kv_connector_extra_config", None) or {}
        if not _config_bool(extra.get("lmcache_block_first", False)):
            return False

        reasons = []
        if not self.use_cufile:
            reasons.append("cuFile is disabled")
        if metadata.use_mla:
            reasons.append("MLA is unsupported")
        if self.config.use_layerwise:
            reasons.append("layerwise mode is unsupported")
        if metadata.get_num_groups() != 1:
            reasons.append("multiple KV layer groups are unsupported in v1")
        if metadata.kv_shape[1] != 2:
            reasons.append("only NHD K/V pairs are supported")
        if metadata.block_size is None or metadata.block_size <= 0:
            reasons.append("vLLM block size is unavailable")
        elif metadata.chunk_size % metadata.block_size != 0:
            reasons.append(
                f"chunk size {metadata.chunk_size} is not divisible by "
                f"block size {metadata.block_size}"
            )

        if reasons:
            logger.warning(
                "GDS block_first_v1 disabled; using legacy layout: %s",
                "; ".join(reasons),
            )
            return False
        logger.info("GDS block_first_v1 disk layout enabled")
        return True

    def _initialize_block_first_staging(self, metadata: LMCacheMetadata) -> None:
        assert metadata.block_size is not None
        shapes = metadata.get_shapes(metadata.chunk_size)
        dtypes = metadata.get_dtypes()
        if len(shapes) != 1 or len(dtypes) != 1:
            raise RuntimeError("block_first_v1 requires exactly one KV layer group")
        payload_shape = block_first_shape(shapes[0], metadata.block_size)
        payload_nbytes = torch.Size(payload_shape).numel() * dtypes[0].itemsize
        allocator = CuFileMemoryAllocator(align_up(payload_nbytes))
        staging_obj = allocator.allocate(
            torch.Size(payload_shape), dtypes[0], MemoryFormat.UNDEFINED
        )
        if staging_obj is None or staging_obj.tensor is None:
            allocator.close()
            raise RuntimeError("Failed to allocate block_first_v1 staging buffer")
        allocator.tensor.zero_()
        self._block_first_staging_allocator = allocator
        self._block_first_staging_obj = staging_obj

    def _close_block_first_staging(self) -> None:
        staging_obj = self._block_first_staging_obj
        allocator = self._block_first_staging_allocator
        if staging_obj is not None and staging_obj.is_valid():
            staging_obj.ref_count_down()
        if allocator is not None:
            allocator.close()
        self._block_first_staging_obj = None
        self._block_first_staging_allocator = None

    async def _scan_metadata(self):
        # TODO: even though we only run it once on startup, this is still
        # not super scalable - test whether Rust code will be faster here, or
        # whether we can serialize meta-data in groups for faster loading.
        tasks = []
        start = time.perf_counter()
        with os.scandir(self.gds_path) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                l1_dir = os.path.basename(entry.name)
                if len(l1_dir) != 2:
                    continue
                tasks.append(
                    asyncio.to_thread(
                        self._scan_metadata_subdir,
                        os.path.join(self.gds_path, l1_dir),
                        l1_dir,
                    )
                )
        # TODO: If Python 3.11+, can we use TaskGroup instead?
        await asyncio.gather(*tasks)
        end = time.perf_counter()
        logger.info(
            f"Read {len(self.hot_cache)} cache entries from persistent "
            f"storage in {end - start:.2f} seconds"
        )

    def _scan_metadata_subdir(self, path, l1_dir):
        target_suffix = self.data_suffix + _METADATA_FILE_SUFFIX
        with os.scandir(path) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                l2_dir = os.path.basename(entry.name)
                if len(l2_dir) != 2:
                    continue
                with os.scandir(os.path.join(path, l2_dir)) as it2:
                    for fentry in it2:
                        if not fentry.is_file():
                            continue
                        if fentry.name.endswith(_BLOCK_FIRST_DATA_FILE_SUFFIX):
                            if not self.block_first_enabled:
                                continue
                            filename = os.path.basename(fentry.name)
                            key_str = filename[: -len(_BLOCK_FIRST_DATA_FILE_SUFFIX)]
                            try:
                                key = CacheEngineKey.from_string(unquote(key_str))
                                self._read_block_first_metadata(
                                    key, fentry.path, l1_dir + l2_dir
                                )
                            except (
                                BlockFirstFormatError,
                                OSError,
                                ValueError,
                            ) as e:
                                logger.error(
                                    "Invalid block-first entry %s, ignoring: %s",
                                    fentry.path,
                                    e,
                                )
                            continue
                        if not fentry.name.endswith(target_suffix):
                            continue
                        filename = os.path.basename(fentry.name)
                        key_str = filename[: -len(target_suffix)].replace("_", "/")
                        try:
                            key = CacheEngineKey.from_string(key_str)
                        except ValueError as e:
                            logger.error(
                                f"Filename {filename} can't be converted "
                                f"back into cache key: {e}"
                            )
                            continue
                        try:
                            self._read_metadata(key, fentry.path, l1_dir + l2_dir)
                        except UnsupportedMetadataVersion:
                            logger.error(
                                "Unsupported metadata version for "
                                f"{fentry.path}, ignoring"
                            )

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
            ):  # fallback to normal open if O_NOATIME is not supported
                self._use_noatime = False
                logger.info(
                    "O_NOATIME flag not supported during metadata file read, "
                    "falling back to normal open"
                )
                fd = os.open(filename, os.O_RDONLY)
        else:
            fd = os.open(filename, os.O_RDONLY)
        try:
            buf = os.read(fd, _METADATA_MAX_SIZE)
        finally:
            os.close(fd)
        return unpack_metadata(buf)

    def _read_metadata(
        self,
        key: CacheEngineKey,
        filename: str,
        subdir_key: str,
    ):
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
            previous = self.hot_cache.get(key)
            if previous is not None and previous.disk_metadata is not None:
                return previous
            self.metadata_dirs.add(subdir_key)
            self.hot_cache[key] = metadata
        return metadata

    def _read_block_first_metadata(
        self, key: CacheEngineKey, filename: str, subdir_key: str
    ) -> DiskCacheMetadata:
        with open(filename, "rb") as f:
            disk_metadata = unpack_block_first_header(f.read(BLOCK_FIRST_HEADER_SIZE))
        if disk_metadata.valid_tokens != self.metadata.chunk_size:
            raise BlockFirstFormatError(
                "block_first_v1 file is not a complete LMCache chunk: "
                f"{disk_metadata.valid_tokens}/{self.metadata.chunk_size} tokens"
            )
        expected_shapes = self.metadata.get_shapes(self.metadata.chunk_size)
        expected_dtypes = self.metadata.get_dtypes()
        if len(expected_shapes) != 1 or len(expected_dtypes) != 1:
            raise BlockFirstFormatError(
                "block_first_v1 requires exactly one active KV layer group"
            )
        assert self.metadata.block_size is not None
        disk_metadata.validate_runtime(
            logical_shape=expected_shapes[0],
            dtype=expected_dtypes[0],
            model_name=self.metadata.model_name,
            world_size=self.metadata.world_size,
            worker_id=self.metadata.worker_id,
            tensor_parallel_size=self.metadata.tensor_parallel_size,
            pipeline_parallel_size=self.metadata.pipeline_parallel_size,
            block_size=self.metadata.block_size,
        )
        total_size = BLOCK_FIRST_HEADER_SIZE + disk_metadata.io_nbytes
        if os.path.getsize(filename) != total_size:
            raise BlockFirstFormatError(
                f"block-first file size mismatch for {filename}"
            )
        cache_metadata = DiskCacheMetadata(
            path=filename,
            size=total_size,
            shape=torch.Size(disk_metadata.logical_shape),
            dtype=disk_metadata.torch_dtype,
            fmt=MemoryFormat.KV_2LTD,
            disk_metadata=disk_metadata,
        )
        with self.hot_lock:
            self.metadata_dirs.add(subdir_key)
            self.hot_cache[key] = cache_metadata
        return cache_metadata

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        # TODO: implement pin() semantics
        with self.hot_lock:
            res = key in self.hot_cache
        if res:
            return True
        if self._try_to_read_metadata(key):
            return True
        return False

    def _try_to_read_metadata(self, key: CacheEngineKey) -> Optional[DiskCacheMetadata]:
        if self.block_first_enabled:
            path, subdir_key, _, _ = self._key_to_path(key, block_first=True)
            if os.path.exists(path):
                try:
                    return self._read_block_first_metadata(key, path, subdir_key)
                except (BlockFirstFormatError, OSError) as e:
                    logger.error("Invalid block-first entry %s: %s", path, e)
        path, subdir_key, _, _ = self._key_to_path(key)
        path += _METADATA_FILE_SUFFIX
        if os.path.exists(path):
            try:
                return self._read_metadata(key, path, subdir_key)
            except UnsupportedMetadataVersion:
                logger.error(f"Unsupported metadata version for {path}, ignoring")
        return None

    def _key_to_path(
        self,
        key: CacheEngineKey,
        *,
        block_first: bool = False,
    ) -> Tuple[str, str, str, str]:
        hash = str(key.chunk_hash)
        l1_dir = hash[:2]
        l2_dir = hash[2:4]
        key_str = key.to_string()
        if block_first:
            filename = quote(key_str, safe="") + _BLOCK_FIRST_DATA_FILE_SUFFIX
        else:
            assert "_" not in key_str, "key string should not contain `_`"
            filename = key_str.replace("/", "_") + self.data_suffix
        return (
            os.path.join(
                self.gds_path,
                l1_dir,
                l2_dir,
                filename,
            ),
            l1_dir + l2_dir,
            l1_dir,
            l2_dir,
        )

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def _use_block_first_for_tensor(self, tensor: torch.Tensor) -> bool:
        if not self.block_first_enabled:
            return False
        if tensor.ndim != 4 or tensor.shape[0] != 2:
            raise BlockFirstFormatError(
                "block_first_v1 source must use logical [2, NL, T, D] layout"
            )
        return tensor.shape[2] == self.metadata.chunk_size

    def _make_block_first_metadata(
        self, tensor: torch.Tensor
    ) -> BlockFirstDiskMetadata:
        assert self.metadata.block_size is not None
        return BlockFirstDiskMetadata.create(
            tensor,
            block_size=self.metadata.block_size,
            model_name=self.metadata.model_name,
            world_size=self.metadata.world_size,
            worker_id=self.metadata.worker_id,
            tensor_parallel_size=self.metadata.tensor_parallel_size,
            pipeline_parallel_size=self.metadata.pipeline_parallel_size,
        )

    def _make_direct_block_first_metadata(self) -> BlockFirstDiskMetadata:
        shapes = self.metadata.get_shapes(self.metadata.chunk_size)
        dtypes = self.metadata.get_dtypes()
        if len(shapes) != 1 or len(dtypes) != 1:
            raise BlockFirstFormatError("direct GDS requires one KV layer group")
        logical = torch.empty(shapes[0], dtype=dtypes[0], device="meta")
        return self._make_block_first_metadata(logical)

    def _validate_direct_descriptor(
        self,
        descriptor: DirectIODescriptor,
        direction: DirectIODirection,
    ) -> None:
        if not self.direct_enabled:
            raise ValueError("direct GDS is disabled")
        if descriptor.direction != direction:
            raise ValueError("direct I/O direction mismatch")
        if descriptor.byte_segments:
            raise ValueError("direct GDS does not support byte-segment descriptors")
        descriptor.validate(require_cuda=True, require_alignment=True)
        if descriptor.engine_group_id != 0:
            raise ValueError("0.4.2 direct GDS only supports engine group 0")
        if self.metadata.block_size != descriptor.block_size:
            raise ValueError("direct I/O block size does not match engine metadata")

        expected_device = torch.device(self.dst_device)
        if descriptor.device.type != expected_device.type:
            raise ValueError("direct I/O allocation is on the wrong device type")
        if (
            expected_device.index is not None
            and descriptor.device.index != expected_device.index
        ):
            raise ValueError("direct I/O allocation is on the wrong CUDA device")

        shapes = self.metadata.get_shapes(self.metadata.chunk_size)
        dtypes = self.metadata.get_dtypes()
        if len(shapes) != 1 or len(dtypes) != 1:
            raise ValueError("direct GDS requires one KV layer group")
        logical_shape = shapes[0]
        expected_block_nbytes = (
            logical_shape[0]
            * logical_shape[1]
            * descriptor.block_size
            * logical_shape[3]
            * dtypes[0].itemsize
        )
        if descriptor.block_nbytes != expected_block_nbytes:
            raise ValueError("direct I/O block byte size does not match KV metadata")

        chunk_blocks = self.metadata.chunk_size // descriptor.block_size
        if (
            descriptor.logical_block_start + descriptor.logical_block_count
            > chunk_blocks
        ):
            raise ValueError("direct I/O logical range exceeds one LMCache chunk")
        if (
            descriptor.leading_edge is not None
            and descriptor.leading_edge.logical_block >= chunk_blocks
        ):
            raise ValueError("direct I/O leading edge exceeds one LMCache chunk")

    def _ensure_direct_registration(self, descriptor: DirectIODescriptor) -> None:
        identity = descriptor.registration_identity
        with self._direct_registration_lock:
            if identity in self._direct_registrations:
                return
            if self._direct_register is None:
                raise RuntimeError("cuFile direct registration is unavailable")
            self._direct_register(
                ctypes.c_void_p(descriptor.allocation_base),
                descriptor.allocation_nbytes,
                flags=0,
            )
            self._direct_registrations[identity] = descriptor.lifetime_token

    def _close_direct_registrations(self) -> None:
        with self._direct_registration_lock:
            for identity in list(self._direct_registrations):
                if self._direct_deregister is None:
                    raise RuntimeError("cuFile direct deregistration is unavailable")
                self._direct_deregister(ctypes.c_void_p(identity[1]))
                del self._direct_registrations[identity]

    def _get_direct_edge_staging(
        self, block_nbytes: int
    ) -> tuple[CuFileMemoryAllocator, MemoryObj]:
        allocator = self._direct_edge_allocator
        memory_obj = self._direct_edge_obj
        if allocator is not None and memory_obj is not None:
            if allocator.tensor.numel() != block_nbytes:
                raise RuntimeError("direct edge block size changed at runtime")
            return allocator, memory_obj

        allocator = CuFileMemoryAllocator(block_nbytes)
        memory_obj = allocator.allocate(
            torch.Size([block_nbytes]), torch.uint8, MemoryFormat.UNDEFINED
        )
        if memory_obj is None or memory_obj.tensor is None:
            allocator.close()
            raise RuntimeError("failed to allocate one-block direct edge scratch")
        self._direct_edge_allocator = allocator
        self._direct_edge_obj = memory_obj
        return allocator, memory_obj

    def _close_direct_edge_staging(self) -> None:
        with self._direct_edge_lock:
            memory_obj = self._direct_edge_obj
            allocator = self._direct_edge_allocator
            if memory_obj is not None and memory_obj.is_valid():
                memory_obj.ref_count_down()
            if allocator is not None:
                allocator.close()
            self._direct_edge_obj = None
            self._direct_edge_allocator = None

    def _device_to_device_copy(self, destination: int, source: int, size: int) -> None:
        if self._direct_memcpy is None:
            self._direct_cudart = ctypes.CDLL("libcudart.so")
            cuda_memcpy = self._direct_cudart.cudaMemcpy
            cuda_memcpy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            cuda_memcpy.restype = ctypes.c_int

            def _memcpy_d2d(destination: int, source: int, size: int) -> int:
                return cuda_memcpy(
                    ctypes.c_void_p(destination),
                    ctypes.c_void_p(source),
                    ctypes.c_size_t(size),
                    ctypes.c_int(3),
                )

            self._direct_memcpy = _memcpy_d2d
        result = self._direct_memcpy(destination, source, size)
        if result not in (None, 0):
            raise RuntimeError(f"cudaMemcpyDeviceToDevice failed with code {result}")

    def _copy_direct_leading_edge(
        self,
        descriptor: DirectIODescriptor,
        scratch_base: int,
    ) -> None:
        edge = descriptor.leading_edge
        assert edge is not None
        disk_metadata = self._make_direct_block_first_metadata()
        row_nbytes = disk_metadata.hidden_dim * disk_metadata.torch_dtype.itemsize
        copy_nbytes = edge.token_count * row_nbytes
        for layer in range(disk_metadata.num_layers):
            for kv_index in range(2):
                row_index = (layer * 2 + kv_index) * descriptor.block_size
                edge_offset = (row_index + edge.token_offset) * row_nbytes
                self._device_to_device_copy(
                    descriptor.allocation_base
                    + edge.physical_block * descriptor.block_nbytes
                    + edge_offset,
                    scratch_base + edge_offset,
                    copy_nbytes,
                )

    def _insert_direct_key(
        self,
        key: CacheEngineKey,
        path: str,
        disk_metadata: BlockFirstDiskMetadata,
    ) -> None:
        with self.hot_lock:
            self.hot_cache[key] = DiskCacheMetadata(
                path=path,
                size=BLOCK_FIRST_HEADER_SIZE + disk_metadata.io_nbytes,
                shape=torch.Size(disk_metadata.logical_shape),
                dtype=disk_metadata.torch_dtype,
                fmt=MemoryFormat.KV_2LTD,
                disk_metadata=disk_metadata,
            )

    def _save_direct_gds(
        self,
        path: str,
        descriptor: DirectIODescriptor,
    ) -> BlockFirstDiskMetadata:
        self._validate_direct_descriptor(descriptor, DirectIODirection.STORE)
        disk_metadata = self._make_direct_block_first_metadata()
        chunk_blocks = disk_metadata.valid_tokens // disk_metadata.block_size
        if descriptor.logical_block_start != 0 or (
            descriptor.logical_block_count != chunk_blocks
        ):
            raise ValueError("direct store must cover the complete LMCache chunk")
        if descriptor.transferred_nbytes != disk_metadata.payload_nbytes:
            raise ValueError("direct store byte count does not match the file payload")
        if disk_metadata.payload_nbytes != disk_metadata.io_nbytes:
            raise ValueError("direct store payload requires unsupported tail padding")

        self._ensure_direct_registration(descriptor)
        tmp_path = path + _BLOCK_FIRST_DATA_FILE_SUFFIX_TMP
        try:
            header = pack_block_first_header(disk_metadata)
            with open(tmp_path, "wb") as header_file:
                written = header_file.write(header)
            if written != BLOCK_FIRST_HEADER_SIZE:
                raise RuntimeError(f"block-first header write incomplete: {written}")

            assert self.cufile is not None
            torch.cuda.synchronize(descriptor.device)
            with self.cufile.CuFile(
                tmp_path, "r+", use_direct_io=self.use_direct_io
            ) as data_file:
                for run in descriptor.runs:
                    size = run.block_count * descriptor.block_nbytes
                    ret = data_file.write(
                        ctypes.c_void_p(descriptor.allocation_base),
                        size,
                        file_offset=(
                            descriptor.file_payload_offset
                            + run.logical_block_start * descriptor.block_nbytes
                        ),
                        dev_offset=run.physical_block_start * descriptor.block_nbytes,
                    )
                    if ret is not None and ret != size:
                        raise RuntimeError(f"direct GDS write incomplete: {ret}/{size}")

            expected_size = BLOCK_FIRST_HEADER_SIZE + disk_metadata.io_nbytes
            if os.path.getsize(tmp_path) != expected_size:
                raise RuntimeError("direct GDS file size mismatch")
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            raise
        return disk_metadata

    def _load_direct_gds(
        self,
        path: str,
        disk_metadata: BlockFirstDiskMetadata,
        descriptor: DirectIODescriptor,
    ) -> None:
        self._validate_direct_descriptor(descriptor, DirectIODirection.RETRIEVE)
        with open(path, "rb") as header_file:
            current_metadata = unpack_block_first_header(
                header_file.read(BLOCK_FIRST_HEADER_SIZE)
            )
        if current_metadata != disk_metadata:
            raise BlockFirstFormatError("block-first header changed after indexing")
        expected_size = BLOCK_FIRST_HEADER_SIZE + disk_metadata.io_nbytes
        if os.path.getsize(path) != expected_size:
            raise BlockFirstFormatError(
                "block-first payload size changed after indexing"
            )
        disk_metadata.validate_runtime(
            logical_shape=self.metadata.get_shapes(self.metadata.chunk_size)[0],
            dtype=self.metadata.get_dtypes()[0],
            model_name=self.metadata.model_name,
            world_size=self.metadata.world_size,
            worker_id=self.metadata.worker_id,
            tensor_parallel_size=self.metadata.tensor_parallel_size,
            pipeline_parallel_size=self.metadata.pipeline_parallel_size,
            block_size=descriptor.block_size,
        )

        self._ensure_direct_registration(descriptor)
        assert self.cufile is not None
        torch.cuda.synchronize(descriptor.device)
        with self.cufile.CuFile(
            path, "r", use_direct_io=self.use_direct_io
        ) as data_file:
            if descriptor.leading_edge is not None:
                with self._direct_edge_lock:
                    edge_allocator, edge_obj = self._get_direct_edge_staging(
                        descriptor.block_nbytes
                    )
                    edge = descriptor.leading_edge
                    ret = data_file.read(
                        ctypes.c_void_p(edge_allocator.base_pointer),
                        descriptor.block_nbytes,
                        file_offset=(
                            descriptor.file_payload_offset
                            + edge.logical_block * descriptor.block_nbytes
                        ),
                        dev_offset=edge_obj.metadata.address,
                    )
                    if ret is not None and ret != descriptor.block_nbytes:
                        raise RuntimeError(
                            "direct GDS edge read incomplete: "
                            f"{ret}/{descriptor.block_nbytes}"
                        )
                    self._copy_direct_leading_edge(
                        descriptor,
                        edge_allocator.base_pointer + edge_obj.metadata.address,
                    )
            for run in descriptor.runs:
                size = run.block_count * descriptor.block_nbytes
                ret = data_file.read(
                    ctypes.c_void_p(descriptor.allocation_base),
                    size,
                    file_offset=(
                        descriptor.file_payload_offset
                        + run.logical_block_start * descriptor.block_nbytes
                    ),
                    dev_offset=run.physical_block_start * descriptor.block_nbytes,
                )
                if ret is not None and ret != size:
                    raise RuntimeError(f"direct GDS read incomplete: {ret}/{size}")
        torch.cuda.synchronize(descriptor.device)

    def batched_direct_put(
        self,
        keys: Sequence[CacheEngineKey],
        descriptors: Sequence[Optional[DirectIODescriptor]],
    ) -> list[bool]:
        results: list[bool] = []
        for key, descriptor in zip(keys, descriptors, strict=True):
            if descriptor is None or not self.direct_enabled:
                results.append(False)
                continue
            with self.hot_lock:
                existing = self.hot_cache.get(key)
            if existing is not None and existing.disk_metadata is not None:
                results.append(True)
                continue
            with self.put_lock:
                if key in self.put_tasks:
                    results.append(False)
                    continue
                self.put_tasks.add(key)
            try:
                path, _, l1_dir, l2_dir = self._key_to_path(key, block_first=True)
                os.makedirs(os.path.join(self.gds_path, l1_dir, l2_dir), exist_ok=True)
                disk_metadata = self._save_direct_gds(path, descriptor)
                self._insert_direct_key(key, path, disk_metadata)
                results.append(True)
            except (BlockFirstFormatError, OSError, RuntimeError, ValueError) as exc:
                logger.warning("Direct GDS store fallback for %s: %s", key, exc)
                results.append(False)
            finally:
                with self.put_lock:
                    self.put_tasks.discard(key)
        return results

    def batched_direct_get(
        self,
        keys: Sequence[CacheEngineKey],
        descriptors: Sequence[Optional[DirectIODescriptor]],
    ) -> list[bool]:
        results: list[bool] = []
        for key, descriptor in zip(keys, descriptors, strict=True):
            if descriptor is None or not self.direct_enabled:
                results.append(False)
                continue
            with self.hot_lock:
                entry = self.hot_cache.get(key)
            if entry is None or entry.disk_metadata is None:
                results.append(False)
                continue
            try:
                self._load_direct_gds(entry.path, entry.disk_metadata, descriptor)
                results.append(True)
            except (BlockFirstFormatError, OSError, RuntimeError, ValueError) as exc:
                logger.warning("Direct GDS retrieve fallback for %s: %s", key, exc)
                results.append(False)
        return results

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Future:
        """
        Submit a put task to store KV cache to GDS asynchronously.

        :param on_complete_callback: Optional callback invoked after the GDS
            write completes. Callback exceptions are caught and logged.
        """
        assert memory_obj.tensor is not None
        memory_obj.ref_count_up()

        with self.put_lock:
            self.put_tasks.add(key)

        future = asyncio.run_coroutine_threadsafe(
            self._async_save_bytes_to_disk(key, memory_obj, on_complete_callback),
            self.loop,
        )
        with self._pending_put_futures_lock:
            self._pending_put_futures.add(future)

        def _discard(done: Future) -> None:
            with self._pending_put_futures_lock:
                self._pending_put_futures.discard(done)

        future.add_done_callback(_discard)
        return future

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec: Any = None,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> Union[List[Future], None]:
        """
        Submit batched put tasks to store KV caches to GDS asynchronously.

        :param on_complete_callback: Optional callback invoked once per key
            after that key's write completes (not once per batch).
        """
        futures = []
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            future = self.submit_put_task(
                key, memory_obj, on_complete_callback=on_complete_callback
            )
            futures.append(future)
        return futures

    async def _async_save_bytes_to_disk(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        on_complete_callback: Optional[Callable[[CacheEngineKey], None]] = None,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to disk.

        :param on_complete_callback: Optional callback invoked after the GDS
            write completes for this key. Callback exceptions are caught.
        """
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None
        use_block_first = self._use_block_first_for_tensor(kv_chunk)
        path, subdir_key, l1_dir, l2_dir = self._key_to_path(
            key, block_first=use_block_first
        )
        # TODO: maybe remove `metadata_dirs` and insert mkdir calls
        # only for the case where creating the CuFile fails on ENOENT. It
        # also makes the code more resilient to out-of-band deletions
        if subdir_key not in self.metadata_dirs:
            os.makedirs(os.path.join(self.gds_path, l1_dir, l2_dir), exist_ok=True)
            self.metadata_dirs.add(subdir_key)
        tmp = ".tmp" + rand_suffix(self.rand, 8)
        fmt = memory_obj.metadata.fmt
        disk_metadata = None
        try:
            if use_block_first:
                disk_metadata = await asyncio.to_thread(
                    self._save_block_first_gds, path, kv_chunk
                )
                metadata = disk_metadata
            else:
                metadata = await asyncio.to_thread(
                    self._save_gds,
                    path,
                    tmp,
                    kv_chunk,
                    fmt,
                    self.cufile_base_pointer,
                    memory_obj.metadata.address,
                )
                await save_metadata(path + _METADATA_FILE_SUFFIX, tmp, metadata)

            logger.debug(
                f"Saved {kv_chunk.numel()} elements of {kv_chunk.dtype} "
                f"to {path} with metadata {metadata}"
            )
            self.insert_key(
                key,
                memory_obj,
                path=path,
                disk_metadata=disk_metadata,
            )

            if on_complete_callback is not None:
                try:
                    on_complete_callback(key)
                except Exception as e:
                    logger.warning(f"on_complete_callback failed for key {key}: {e}")
        finally:
            memory_obj.ref_count_down()
            with self.put_lock:
                self.put_tasks.discard(key)

    def insert_key(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        *,
        path: Optional[str] = None,
        disk_metadata: Optional[BlockFirstDiskMetadata] = None,
    ) -> None:
        if path is None:
            path, _, _, _ = self._key_to_path(key)
        size = (
            BLOCK_FIRST_HEADER_SIZE + disk_metadata.io_nbytes
            if disk_metadata is not None
            else memory_obj.get_physical_size()
        )
        shape = memory_obj.metadata.shape
        dtype = memory_obj.metadata.dtype
        fmt = memory_obj.metadata.fmt
        with self.hot_lock:
            # TODO(Jiayi): need to support `cached_positions`.
            self.hot_cache[key] = DiskCacheMetadata(
                path=path,
                size=size,
                shape=shape,
                dtype=dtype,
                fmt=fmt,
                disk_metadata=disk_metadata,
            )

    def submit_prefetch_task(
        self,
        key: CacheEngineKey,
    ) -> bool:
        # with self.hot_lock:
        #     entry = self.hot_cache.get(key)
        # if entry is None:
        #     return None

        # path = entry.path
        # dtype = entry.dtype
        # shape = entry.shape
        # fmt = entry.fmt
        # assert dtype is not None
        # assert shape is not None
        # assert fmt is not None
        # return asyncio.run_coroutine_threadsafe(
        #     self._async_load_bytes_from_disk(key, path, dtype, shape，fmt), self.loop
        # )

        # TODO(Jiayi): Need to modify this when prefetch interface is determined.

        # TODO(Jiayi): add `test_gds_backend_sanity` back after implementing this
        return False

    async def _async_load_bytes_from_disk(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
        fmt: MemoryFormat,
    ) -> Optional[MemoryObj]:
        return self._load_bytes_from_disk_with_allocation(
            key, path, dtype, shape, fmt=fmt
        )

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        if self.memory_allocator is None:
            return None
        with self.hot_lock:
            entry = self.hot_cache.get(key)
        if entry is None:
            return None

        path = entry.path
        dtype = entry.dtype
        shape = entry.shape
        fmt = entry.fmt
        logger.warning(entry)
        assert dtype is not None
        assert shape is not None
        assert fmt is not None
        return self._load_bytes_from_disk_with_allocation(
            key, path, dtype=dtype, shape=shape, fmt=fmt
        )

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
        return self._load_bytes_from_disk_with_memory(key, path, memory_obj)

    def _load_bytes_from_disk_with_memory(
        self,
        key: CacheEngineKey,
        path: str,
        memory_obj: Optional[MemoryObj],
    ) -> Optional[MemoryObj]:
        """
        Load byte array from disk into a pre-allocated memory object.

        Args:
            key: Cache key for error handling
            path: File path to load from
            memory_obj: Pre-allocated memory object to load data into

        Returns:
            The memory object with loaded data, or None if loading failed
        """
        if memory_obj is None or not memory_obj.is_valid():
            return None

        with self.hot_lock:
            entry = self.hot_cache.get(key)
        if entry is None:
            memory_obj.ref_count_down()
            return None

        if entry.disk_metadata is not None:
            tensor = memory_obj.tensor
            assert tensor is not None
            size = entry.disk_metadata.io_nbytes
            try:
                ret = self._load_block_first_gds(path, entry.disk_metadata, tensor)
            except (BlockFirstFormatError, OSError) as e:
                logger.error("Invalid block-first entry %s: %s", path, e)
                ret = -1
        else:
            size = memory_obj.get_size()
            offset = _METADATA_MAX_SIZE
            if self.cufile_base_pointer is None:
                tensor = memory_obj.tensor
                assert tensor is not None
                addr = ctypes.c_void_p(tensor.data_ptr())
                dev_offset = 0
            else:
                addr = ctypes.c_void_p(self.cufile_base_pointer)
                dev_offset = memory_obj.metadata.address
            ret = self._load_gds(path, offset, addr, size, dev_offset)
        if ret != size:
            if ret < 0:
                logger.error(
                    f"Error loading {path}: ret: {ret} removing entry from cache"
                )
                with self.hot_lock:
                    self.hot_cache.pop(key)
            else:
                # TODO: we should probably count errors and
                # remove the entry if it's a persistent problem.
                logger.error(
                    f"Error loading {path}: got only {ret} bytes "
                    f"out of {size}, ignoring"
                )
            memory_obj.ref_count_down()
            return None
        return memory_obj

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        # TODO: Using a dummy wrapper around prefetch for now.
        if not self.submit_prefetch_task(key):
            return None
        return Future()

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> List[Optional[MemoryObj]]:
        if self.memory_allocator is None:
            return [None] * len(keys)
        if self.use_thread_pool:
            logger.info("Using batched_get_blocking with thread pool implementation")
            return self._batched_get_blocking_by_thread_pool_impl(keys)
        else:
            return super().batched_get_blocking(keys)

    def _batched_get_blocking_by_thread_pool_impl(
        self,
        keys: List[CacheEngineKey],
    ) -> list[MemoryObj | None]:
        paths: list[str | None] = []
        dtypes: list[torch.dtype | None] = []
        shapes: list[torch.Size | None] = []
        fmts: list[MemoryFormat | None] = []
        with self.hot_lock:
            for key in keys:
                entry = self.hot_cache.get(key)
                if entry is None:
                    logger.error(f"Lookup failed during get_blocking for {key}")
                    paths.append(None)
                    dtypes.append(None)
                    shapes.append(None)
                    fmts.append(None)
                    continue
                paths.append(entry.path)
                dtypes.append(entry.dtype)
                shapes.append(entry.shape)
                fmts.append(entry.fmt)

        memory_objs: list[MemoryObj | None] = []
        gds_reads, gds_read_bytes = 0, 0
        for dtype, shape, fmt, path in zip(dtypes, shapes, fmts, paths, strict=True):
            if path is None:
                memory_objs.append(None)
                continue
            assert dtype is not None and shape is not None and fmt is not None
            memory_obj = self.memory_allocator.allocate(shape, dtype, fmt=fmt)
            if memory_obj is None:
                logger.error(f"Memory allocation failed during get_blocking for {path}")
            else:
                gds_reads += 1
                gds_read_bytes += memory_obj.get_size()
            memory_objs.append(memory_obj)

        start_time = time.perf_counter()
        assert self._thread_pool is not None
        results = list(
            self._thread_pool.map(
                self._load_bytes_from_disk_with_memory, keys, paths, memory_objs
            )
        )
        total_time = time.perf_counter() - start_time
        logger.info(
            f"Time taken for batched_get_blocking: {total_time:.3f}s |"
            f" {gds_read_bytes / 1024 / 1024}MiB | {gds_reads} ops."
        )
        return results

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _save_block_first_gds(
        self, path: str, logical_chunk: torch.Tensor
    ) -> BlockFirstDiskMetadata:
        disk_metadata = self._make_block_first_metadata(logical_chunk)
        staging_obj = self._block_first_staging_obj
        staging_allocator = self._block_first_staging_allocator
        if (
            staging_obj is None
            or staging_obj.tensor is None
            or staging_allocator is None
        ):
            raise RuntimeError("block_first_v1 staging buffer is unavailable")
        if tuple(staging_obj.tensor.shape) != disk_metadata.payload_shape:
            raise BlockFirstFormatError(
                "block-first staging shape does not match the complete chunk"
            )
        if self.cufile is None:
            raise RuntimeError("block_first_v1 requires cuFile")

        tmp_path = path + _BLOCK_FIRST_DATA_FILE_SUFFIX_TMP
        try:
            with self._block_first_lock:
                staging_allocator.tensor.zero_()
                assert self.metadata.block_size is not None
                logical_to_block_first(
                    logical_chunk,
                    self.metadata.block_size,
                    out=staging_obj.tensor,
                )
                if staging_obj.tensor.is_cuda:
                    torch.cuda.synchronize(staging_obj.tensor.device)
                header = pack_block_first_header(disk_metadata)
                with open(tmp_path, "wb") as f:
                    written = f.write(header)
                if written != BLOCK_FIRST_HEADER_SIZE:
                    raise RuntimeError(
                        f"block-first header write incomplete: {written}"
                    )
                with self.cufile.CuFile(
                    tmp_path, "r+", use_direct_io=self.use_direct_io
                ) as f:
                    ret = f.write(
                        ctypes.c_void_p(staging_allocator.base_pointer),
                        disk_metadata.io_nbytes,
                        file_offset=BLOCK_FIRST_HEADER_SIZE,
                        dev_offset=staging_obj.metadata.address,
                    )
                if ret is not None and ret != disk_metadata.io_nbytes:
                    raise RuntimeError(
                        "block-first GDS write incomplete: "
                        f"{ret}/{disk_metadata.io_nbytes}"
                    )
                expected_size = BLOCK_FIRST_HEADER_SIZE + disk_metadata.io_nbytes
                if os.path.getsize(tmp_path) != expected_size:
                    raise RuntimeError("block-first GDS file size mismatch")
                os.replace(tmp_path, path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            raise
        return disk_metadata

    @torch.inference_mode()
    def _load_block_first_gds(
        self,
        path: str,
        disk_metadata: BlockFirstDiskMetadata,
        logical_output: torch.Tensor,
    ) -> int:
        with open(path, "rb") as f:
            current_metadata = unpack_block_first_header(
                f.read(BLOCK_FIRST_HEADER_SIZE)
            )
        if current_metadata != disk_metadata:
            raise BlockFirstFormatError("block-first header changed after indexing")
        expected_size = BLOCK_FIRST_HEADER_SIZE + disk_metadata.io_nbytes
        if os.path.getsize(path) != expected_size:
            raise BlockFirstFormatError(
                "block-first payload size changed after indexing"
            )
        staging_obj = self._block_first_staging_obj
        staging_allocator = self._block_first_staging_allocator
        if (
            staging_obj is None
            or staging_obj.tensor is None
            or staging_allocator is None
        ):
            raise RuntimeError("block_first_v1 staging buffer is unavailable")
        if tuple(staging_obj.tensor.shape) != disk_metadata.payload_shape:
            raise BlockFirstFormatError(
                "block-first staging shape mismatch during load"
            )

        with self._block_first_lock:
            staging_allocator.tensor.zero_()
            if staging_obj.tensor.is_cuda:
                torch.cuda.synchronize(staging_obj.tensor.device)
            ret = self._load_gds(
                path,
                BLOCK_FIRST_HEADER_SIZE,
                ctypes.c_void_p(staging_allocator.base_pointer),
                disk_metadata.io_nbytes,
                staging_obj.metadata.address,
            )
            if ret != disk_metadata.io_nbytes:
                return ret
            if staging_obj.tensor.is_cuda:
                torch.cuda.synchronize(staging_obj.tensor.device)
            block_first_to_logical(staging_obj.tensor, out=logical_output)
            if logical_output.is_cuda:
                torch.cuda.synchronize(logical_output.device)
        return ret

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _save_gds(
        self,
        path: str,
        tmp: str,
        kv_chunk: torch.Tensor,
        fmt: MemoryFormat,
        base_pointer: int,
        device_offset: int,
    ):
        if base_pointer is None:
            addr = ctypes.c_void_p(kv_chunk.data_ptr())
            dev_offset = 0
        else:
            addr = ctypes.c_void_p(base_pointer)
            dev_offset = device_offset
        tmp_path = path + tmp
        offset = _METADATA_MAX_SIZE
        # TODO: We can add the chunk's metadata here, e.g. Tensor parallelism shard
        # and pipeline parallelism index.
        metadata = pack_metadata(
            kv_chunk, fmt=fmt, lmcache_version=str(_METADATA_VERSION)
        )
        try:
            with open(tmp_path, "wb") as f:
                f.write(metadata)
            if self.cufile:
                with self.cufile.CuFile(
                    tmp_path, "r+", use_direct_io=self.use_direct_io
                ) as f:
                    f.write(
                        addr, kv_chunk.nbytes, file_offset=offset, dev_offset=dev_offset
                    )
            elif self.cudart:
                # mmap the file
                fd = os.open(tmp_path, os.O_RDWR)
                nbytes = kv_chunk.nbytes
                os.ftruncate(fd, nbytes + offset)
                mm = mmap.mmap(
                    fd, nbytes + offset, prot=mmap.PROT_WRITE, flags=mmap.MAP_SHARED
                )
                os.close(fd)

                # get mapped file address
                arr = np.frombuffer(mm, dtype=np.uint8)
                buf_addr = arr.__array_interface__["data"][0]

                assert addr.value is not None
                res = self.cudart.cudaMemcpy(
                    ctypes.c_void_p(buf_addr + offset),
                    ctypes.c_void_p(int(addr.value) + device_offset),
                    ctypes.c_size_t(nbytes),
                    ctypes.c_int(2),
                )
                if res:
                    raise RuntimeError(f"cudaMemcpy failed {res}")
                del arr
                mm.close()

        except Exception as e:
            logger.error(f"Error saving {tmp_path}: {e}", exc_info=True)
            raise e
        os.rename(tmp_path, path)
        return metadata

    def _load_gds(
        self,
        gds_path: str,
        file_offset: int,
        gpu_pointer: ctypes.c_void_p,
        size_in_bytes: int,
        dev_offset: int,
    ) -> int:
        # Read data from disk into a GPU buffer
        if self.cufile:
            with self.cufile.CuFile(
                gds_path, "r", use_direct_io=self.use_direct_io
            ) as f:
                return f.read(
                    gpu_pointer,
                    size_in_bytes,
                    file_offset=file_offset,
                    dev_offset=dev_offset,
                )
        elif self.cudart:
            fd = os.open(gds_path, os.O_RDONLY)
            file_size = os.fstat(fd).st_size
            mm = mmap.mmap(
                fd,
                file_size,
                prot=mmap.PROT_READ,
                flags=mmap.MAP_PRIVATE | mmap.MAP_POPULATE,  # type: ignore [attr-defined]
            )
            os.close(fd)

            arr = np.frombuffer(mm, dtype=np.uint8)
            addr = arr.__array_interface__["data"][0]

            assert gpu_pointer.value is not None
            res = self.cudart.cudaMemcpy(
                ctypes.c_void_p(int(gpu_pointer.value) + dev_offset),
                ctypes.c_void_p(addr + file_offset),
                ctypes.c_size_t(size_in_bytes),
                ctypes.c_int(1),
            )

            if res != 0:
                raise RuntimeError(f"cudaMemcpy failed with code {res}")
            del arr
            mm.close()
            return size_in_bytes
        else:
            raise RuntimeError(
                "Both cufile and cudart are None, this should not happen"
            )

    def pin(self, key: CacheEngineKey) -> bool:
        # NOTE (ApostaC): Since gds doesn't have eviction now, we don't need
        # to implement pin and unpin
        return False

    def unpin(self, key: CacheEngineKey) -> bool:
        # NOTE (ApostaC): Since gds doesn't have eviction now, we don't need
        # to implement pin and unpin
        return False

    def remove(self, key: CacheEngineKey, force: bool = True):
        raise NotImplementedError("Remote backend does not support remove now.")

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> CuFileMemoryAllocator:
        assert config.cufile_buffer_size is not None
        return CuFileMemoryAllocator(config.cufile_buffer_size * 1024**2)

    def allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        if busy_loop:
            logger.warning("GDS Backend does not support allocation with busy loop")
        if eviction:
            logger.warning("GDS Backend does not support eviction")

        if self.memory_allocator is None:
            return None
        return self.memory_allocator.allocate(shapes, dtypes, fmt)

    def batched_allocate(
        self,
        shapes: Union[torch.Size, list[torch.Size]],
        dtypes: Union[torch.dtype, list[torch.dtype]],
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        if busy_loop:
            logger.warning("GDS Backend does not support allocation with busy loop")
        if eviction:
            logger.warning("GDS Backend does not support eviction")

        if self.memory_allocator is None:
            return None
        return self.memory_allocator.batched_allocate(shapes, dtypes, batch_size, fmt)

    def get_allocator_backend(self):
        return self

    def get_memory_allocator(self):
        return self.memory_allocator

    def close(self) -> None:
        if self._closed:
            return
        try:
            self._scan_future.result()
        except Exception as e:
            logger.error("GDS metadata scan failed while closing: %s", e)
        with self._pending_put_futures_lock:
            pending = list(self._pending_put_futures)
        for future in pending:
            try:
                future.result()
            except Exception as e:
                logger.error("GDS put failed while closing: %s", e)
        if self._thread_pool is not None:
            self._thread_pool.shutdown(wait=True)
        self._close_direct_registrations()
        self._close_direct_edge_staging()
        self._close_block_first_staging()
        if self.memory_allocator is not None:
            self.memory_allocator.close()
        self._closed = True
        logger.info("GDS backend closed.")

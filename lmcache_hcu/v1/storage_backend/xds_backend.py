# SPDX-License-Identifier: Apache-2.0
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from concurrent.futures import Future, ThreadPoolExecutor
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import (
    TYPE_CHECKING,
    Any,
    Callable,
    Generator,
    List,
    Optional,
    Sequence,
    Tuple,
)
from urllib.parse import quote, unquote
import asyncio
import ctypes
import json
import os
import random
import string
import struct
import tempfile
import threading
import time
import mmap
import numpy as np

# Third Party
import torch
from abc import ABC, abstractmethod

# First Party
from lmcache.v1.metadata import LMCacheMetadata
from lmcache.logging import init_logger
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache_hcu.v1.storage_backend.xds_metadata import XDSCacheMetadata
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    CuFileMemoryAllocator,
    HyFileMemoryAllocator,
    MemoryFormat,
    MemoryObj,
)
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
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache_hcu.v1.storage_backend.direct_io import (
    DirectIODescriptor,
    DirectIODirection,
)
from lmcache.v1.storage_backend.job_executor.pq_executor import (
    AsyncPQThreadPoolExecutor,
)
from lmcache_hcu.v1.hipfile import HIPFile
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

_METADATA_FILE_SUFFIX = ".metadata"
_DATA_FILE_SUFFIX = ".kvdata"
_METADATA_FILE_SUFFIX_TMP = ".METADATA"
_DATA_FILE_SUFFIX_TMP = ".KVDATA"
_BLOCK_FIRST_DATA_FILE_SUFFIX = ".block_first_v1.kvdata"
_BLOCK_FIRST_DATA_FILE_SUFFIX_TMP = ".BLOCK_FIRST_V1_TMP"
_METADATA_VERSION = 1

# gds、 weka均为4096，但是是填充的，此处72为经验值
_METADATA_MAX_SIZE = 72
_ALIGNMENT = 4096  # 4K 对齐
_DIR_SIZE = 2
MEMCPY_HOST_TO_DEVICE = 1
MEMCPY_DEVICE_TO_HOST = 2


@dataclass
class XDSIOProfile:
    """Thread-safe aggregate for one profiled request/batch."""

    profile_id: str
    operation: str
    path: str
    worker_id: int
    expected_chunks: int
    started_ns: int = field(default_factory=time.perf_counter_ns)
    blocks_per_chunk: list[int] = field(default_factory=list)
    physical_block_ids: list[list[int]] = field(default_factory=list)
    runs_per_chunk: list[int] = field(default_factory=list)
    run_block_counts: list[list[int]] = field(default_factory=list)
    run_bytes: list[list[int]] = field(default_factory=list)
    io_calls: int = 0
    read_calls: int = 0
    write_calls: int = 0
    bytes_read: int = 0
    bytes_written: int = 0
    requested_io_bytes: int = 0
    handle_open_count: int = 0
    cuda_sync_count: int = 0
    cuda_sync_ns: int = 0
    file_open_ns: int = 0
    handle_register_ns: int = 0
    dma_ns: int = 0
    handle_deregister_ns: int = 0
    file_close_ns: int = 0
    xds_call_ns: int = 0
    header_ns: int = 0
    stat_ns: int = 0
    rename_ns: int = 0
    _read_dma_samples: list[tuple[int, int, int, str]] = field(
        default_factory=list, init=False, repr=False
    )
    successful_chunks: int = 0
    failed_chunks: int = 0
    fallback_count: int = 0
    completed_chunks: int = 0
    _logged: bool = field(default=False, init=False, repr=False)
    _lock: threading.Lock = field(
        default_factory=threading.Lock, init=False, repr=False
    )

    def record_descriptor(self, descriptor: Optional[DirectIODescriptor]) -> None:
        """Record fragmentation coordinates for one descriptor."""
        with self._lock:
            if descriptor is None:
                self.blocks_per_chunk.append(0)
                self.physical_block_ids.append([])
                self.runs_per_chunk.append(0)
                self.run_block_counts.append([])
                self.run_bytes.append([])
                self.fallback_count += 1
                return
            block_ids = list(descriptor.physical_block_ids)
            if not block_ids:
                block_ids = [
                    block_id
                    for run in descriptor.runs
                    for block_id in range(
                        run.physical_block_start,
                        run.physical_block_start + run.block_count,
                    )
                ]
            if descriptor.leading_edge is not None:
                block_ids.insert(0, descriptor.leading_edge.physical_block)
            self.blocks_per_chunk.append(
                descriptor.logical_block_count
                + (1 if descriptor.leading_edge is not None else 0)
            )
            self.physical_block_ids.append(block_ids)
            if descriptor.byte_segments:
                self.runs_per_chunk.append(len(descriptor.byte_segments))
                counts = []
                sizes = [segment.nbytes for segment in descriptor.byte_segments]
            else:
                self.runs_per_chunk.append(
                    len(descriptor.runs)
                    + (1 if descriptor.leading_edge is not None else 0)
                )
                counts = [run.block_count for run in descriptor.runs]
                sizes = [
                    run.block_count * descriptor.block_nbytes
                    for run in descriptor.runs
                ]
            if descriptor.leading_edge is not None:
                counts.insert(0, 1)
                sizes.insert(0, descriptor.block_nbytes)
            self.run_block_counts.append(counts)
            self.run_bytes.append(sizes)

    def record_staged_chunk(self, nbytes: int) -> None:
        """Record one contiguous staged chunk before asynchronous submission."""
        with self._lock:
            self.blocks_per_chunk.append(0)
            self.physical_block_ids.append([])
            self.runs_per_chunk.append(1)
            self.run_block_counts.append([])
            self.run_bytes.append([nbytes])

    def record_sync(self, elapsed_ns: int) -> None:
        """Record one completed device synchronization."""
        with self._lock:
            self.cuda_sync_count += 1
            self.cuda_sync_ns += elapsed_ns

    def record_phase(self, name: str, elapsed_ns: int) -> None:
        """Record a metadata/header filesystem phase by name."""
        with self._lock:
            if name == "header":
                self.header_ns += elapsed_ns
            elif name == "stat":
                self.stat_ns += elapsed_ns
            elif name == "rename":
                self.rename_ns += elapsed_ns
            else:
                raise ValueError(f"unknown XDS profile phase: {name}")

    def record_xds_call(
        self,
        operation: str,
        requested_nbytes: int,
        returned_nbytes: Optional[int],
        elapsed_ns: int,
        lifecycle: Optional[dict[str, float | int]],
        *,
        file_path: str = "",
        file_offset: int = 0,
    ) -> None:
        """Record one low-level XDS call and optional HIPFile timings."""
        with self._lock:
            self.io_calls += 1
            self.requested_io_bytes += requested_nbytes
            self.xds_call_ns += elapsed_ns
            successful_bytes = max(0, int(returned_nbytes or 0))
            if operation == "read":
                self.read_calls += 1
                self.bytes_read += successful_bytes
            else:
                self.write_calls += 1
                self.bytes_written += successful_bytes
            if lifecycle is None:
                return
            self.handle_open_count += int(lifecycle["handle_open_count"])
            self.file_open_ns += int(
                float(lifecycle["file_open_seconds"]) * 1_000_000_000
            )
            self.handle_register_ns += int(
                float(lifecycle["handle_register_seconds"]) * 1_000_000_000
            )
            dma_ns = int(float(lifecycle["dma_seconds"]) * 1_000_000_000)
            self.dma_ns += dma_ns
            if operation == "read" and dma_ns > 0:
                self._read_dma_samples.append(
                    (dma_ns, requested_nbytes, file_offset, os.path.basename(file_path))
                )
            self.handle_deregister_ns += int(
                float(lifecycle["handle_deregister_seconds"]) * 1_000_000_000
            )
            self.file_close_ns += int(
                float(lifecycle["file_close_seconds"]) * 1_000_000_000
            )

    def finish_chunk(self, success: bool, fallback: bool = False) -> bool:
        """Mark a chunk complete and return whether the batch is now complete."""
        with self._lock:
            self.completed_chunks += 1
            if success:
                self.successful_chunks += 1
            else:
                self.failed_chunks += 1
            if fallback:
                self.fallback_count += 1
            complete = self.completed_chunks >= self.expected_chunks
            if complete and not self._logged:
                self._logged = True
                return True
            return False

    def as_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable snapshot in microseconds and bytes."""
        with self._lock:
            finished_ns = time.perf_counter_ns()
            flat_run_bytes = [size for sizes in self.run_bytes for size in sizes]
            flat_run_blocks = [
                count for counts in self.run_block_counts for count in counts
            ]
            read_dma_samples = sorted(self._read_dma_samples)
            read_dma_count = len(read_dma_samples)
            return {
                "profile_id": self.profile_id,
                "operation": self.operation,
                "path": self.path,
                "worker_id": self.worker_id,
                "started_monotonic_ns": self.started_ns,
                "finished_monotonic_ns": finished_ns,
                "num_chunks": self.expected_chunks,
                "blocks_per_chunk": self.blocks_per_chunk,
                "physical_block_ids": self.physical_block_ids,
                "runs_per_chunk": self.runs_per_chunk,
                "run_block_counts": self.run_block_counts,
                "run_bytes": self.run_bytes,
                "avg_blocks_per_run": (
                    sum(flat_run_blocks) / len(flat_run_blocks)
                    if flat_run_blocks
                    else 0.0
                ),
                "avg_io_size": (
                    sum(flat_run_bytes) / len(flat_run_bytes)
                    if flat_run_bytes
                    else 0.0
                ),
                "io_calls": self.io_calls,
                "read_calls": self.read_calls,
                "write_calls": self.write_calls,
                "bytes_read": self.bytes_read,
                "bytes_written": self.bytes_written,
                "requested_io_bytes": self.requested_io_bytes,
                "handle_open_count": self.handle_open_count,
                "cuda_sync_count": self.cuda_sync_count,
                "cuda_sync_us": self.cuda_sync_ns / 1000,
                "file_open_us": self.file_open_ns / 1000,
                "handle_register_us": self.handle_register_ns / 1000,
                "dma_us": self.dma_ns / 1000,
                "read_dma_sample_count": read_dma_count,
                "read_dma_us_p50": (
                    read_dma_samples[(read_dma_count - 1) // 2][0] / 1000
                    if read_dma_count
                    else 0.0
                ),
                "read_dma_us_p95": (
                    read_dma_samples[int((read_dma_count - 1) * 0.95)][0] / 1000
                    if read_dma_count
                    else 0.0
                ),
                "read_dma_us_max": (
                    read_dma_samples[-1][0] / 1000 if read_dma_count else 0.0
                ),
                "read_dma_top5": [
                    {
                        "dma_us": dma_ns / 1000,
                        "requested_nbytes": nbytes,
                        "file_offset": offset,
                        "file_name": name,
                    }
                    for dma_ns, nbytes, offset, name in reversed(
                        read_dma_samples[-5:]
                    )
                ],
                "handle_deregister_us": self.handle_deregister_ns / 1000,
                "file_close_us": self.file_close_ns / 1000,
                "xds_call_us": self.xds_call_ns / 1000,
                "header_us": self.header_ns / 1000,
                "stat_us": self.stat_ns / 1000,
                "rename_us": self.rename_ns / 1000,
                "physical_persistence_us": (finished_ns - self.started_ns) / 1000,
                "successful_chunks": self.successful_chunks,
                "failed_chunks": self.failed_chunks,
                "fallback_count": self.fallback_count,
            }


@dataclass
class _DirectStoreJob:
    """One reserved direct STORE waiting for synchronous batch completion."""

    index: int
    key: CacheEngineKey
    path: str
    required_size: int
    future: Future


@dataclass
class _DirectRetrieveJob:
    """One validated cache hit waiting for ordered device transfer."""

    index: int
    key: CacheEngineKey
    path: str
    descriptor: DirectIODescriptor
    future: Future


def _config_bool(value: Any) -> bool:
    """Normalize a bool-like connector option without accepting arbitrary values."""
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value.lower() in ("true", "1", "yes", "on")
    return False


def _devices_match(expected: str | torch.device, actual: torch.device) -> bool:
    """Treat an unindexed CUDA device as the process-local current device."""
    expected_device = torch.device(expected)
    actual_device = torch.device(actual)
    return expected_device.type == actual_device.type and (
        expected_device.index is None
        or expected_device.index == actual_device.index
    )


def _close_file_descriptor(fd):
    """Close a fallback file descriptor without masking the I/O failure."""
    if fd is None:
        return
    try:
        os.close(fd)
    except OSError:
        pass


class UnsupportedMetadataVersion(Exception):
    pass


torch_dtypes = [
    torch.half,
    torch.float16,
    torch.bfloat16,
    torch.float,
    torch.float32,
    torch.float64,
    torch.double,
    torch.uint8,
    torch.float8_e4m3fn,
    torch.float8_e5m2,
]
dtype_to_idx = {dtype: idx for idx, dtype in enumerate(torch_dtypes)}


def _verify_hyfile_roundtrip(
    xds,
    allocator,
    root_path,
):
    """Fail closed unless this allocator passes one real HyFile roundtrip."""
    required_size = 2 * _ALIGNMENT
    if allocator.tensor.numel() < required_size:
        raise RuntimeError(
            "HyFile preflight requires at least an 8 KiB registered buffer"
        )
    device_offset = _ALIGNMENT
    test_buffer = allocator.tensor[device_offset:required_size]

    fd, path = tempfile.mkstemp(prefix=".lmcache-xds-preflight-", dir=root_path)
    os.close(fd)
    try:
        test_buffer.fill_(0xA5)
        if test_buffer.is_cuda:
            torch.cuda.synchronize(test_buffer.device)

        size = test_buffer.numel()
        ret = xds.write(path, allocator.base_pointer, size, 0, device_offset)
        if ret != size:
            raise RuntimeError(f"HyFile preflight write incomplete: {ret}/{size}")

        expected = bytes([0xA5]) * size
        with open(path, "rb") as f:
            actual = f.read()
        if actual != expected:
            raise RuntimeError(
                "HyFile preflight write readback mismatch; refusing unsafe XDS startup"
            )

        test_buffer.zero_()
        if test_buffer.is_cuda:
            torch.cuda.synchronize(test_buffer.device)
        ret = xds.read(path, allocator.base_pointer, size, 0, device_offset)
        if ret != size:
            raise RuntimeError(f"HyFile preflight read incomplete: {ret}/{size}")
        if test_buffer.is_cuda:
            torch.cuda.synchronize(test_buffer.device)
        if test_buffer.detach().cpu().numpy().tobytes() != expected:
            raise RuntimeError(
                "HyFile preflight DMA readback mismatch; refusing unsafe XDS startup"
            )
    finally:
        try:
            os.remove(path)
        except FileNotFoundError:
            pass


class XdsWorker:
    def __init__(self, loop: asyncio.AbstractEventLoop) -> None:
        self.put_lock = threading.Lock()
        self.put_tasks: List[CacheEngineKey] = []

        self.prefetch_lock = threading.Lock()
        self.prefetch_tasks: dict[CacheEngineKey, Future] = {}

        self.executor = AsyncPQThreadPoolExecutor(loop, max_workers=4)
        self.loop = loop
        self._closed = False

    async def submit_task(
        self,
        task_type: str,
        task: Callable,
        *args,
        **kwargs,
    ) -> Any:

        if task_type == "prefetch":
            priority = 0
        elif task_type == "delete":
            priority = 1
        elif task_type == "put":
            priority = 2
        else:
            raise ValueError(f"Unknown task type: {task_type}")

        try:
            result = await self.executor.submit_job(
                task,
                *args,
                priority=priority,
                **kwargs,
            )
            return result
        except Exception as e:
            logger.error(f"Failed to submit job {task_type}: {e}", exc_info=True)
            raise

    def remove_put_task(self, key: CacheEngineKey):
        with self.put_lock:
            if key in self.put_tasks:
                self.put_tasks.remove(key)
            else:
                logger.warning(f"Key {key} not found in put tasks.")

    def insert_put_task(self, key: CacheEngineKey):
        with self.put_lock:
            self.put_tasks.append(key)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        with self.put_lock:
            return key in self.put_tasks

    def close(self):
        # Gracefully shut down the executor
        if self._closed:
            return
        self._closed = True
        self.executor.shutdown(wait=True)

def pack_metadata(shape, dtype, size) -> bytes:
    metadata_desc = "<QQQQ" + len(shape) * "Q"
    if struct.calcsize(metadata_desc) > _METADATA_MAX_SIZE:
        # TODO(Serapheim/Ilya): support variable offset for data
        raise ValueError(
            f"Metadata size {struct.calcsize(metadata_desc)} "
            f"exceeds max size {_METADATA_MAX_SIZE}"
        )
    return struct.pack(
        metadata_desc, _METADATA_VERSION,
        dtype_to_idx[dtype],
        size,
        len(shape),
        *shape
    )


def unpack_metadata(buffer):
    # A truncated/corrupt sidecar must surface as UnsupportedMetadataVersion:
    # the startup scan only catches that type, so a bare struct.error here
    # would kill the whole scan task on one bad file (typical crash leftover).
    try:
        version, dt_idx, size, ndim = struct.unpack_from("<QQQQ", buffer)
        if version != 1:
            raise UnsupportedMetadataVersion(
                f"Unsupported metadata version: {version}"
            )
        if dt_idx >= len(torch_dtypes):
            raise UnsupportedMetadataVersion(f"bad dtype code {dt_idx}")
        shape_offset = struct.calcsize("<QQQQ")
        shape = struct.unpack_from("<" + ndim * "Q", buffer, offset=shape_offset)
        return torch.Size(shape), torch_dtypes[dt_idx], size
    except struct.error as e:
        raise UnsupportedMetadataVersion(f"corrupt metadata: {e}") from e


def rand_suffix(rand, n: int):
    return "".join(
        rand.choice(string.ascii_uppercase + string.digits) for _ in range(n)
    )


def save_metadata(path: str, tmp: str, metadata: bytes):
    tmp_path = path.replace(_DATA_FILE_SUFFIX, _DATA_FILE_SUFFIX_TMP) + tmp
    with open(tmp_path, "wb") as f:
        f.write(metadata)
    os.rename(tmp_path, path + _METADATA_FILE_SUFFIX)


class CliStorageInterface(ABC):

    @abstractmethod
    def initialize(self) -> bool:
        """初始化 XDS 驱动，返回是否成功"""
        pass

    @abstractmethod
    def write(self, file_path: str, gpu_address: int, size: int,
              file_offset: int, dev_offset: int) -> int:
        """
        写入数据到文件
        返回：实际写入的字节数，失败返回负数
        """
        pass

    @abstractmethod
    def read(self, file_path: str, gpu_address: int, size: int,
             file_offset: int, dev_offset: int) -> int:
        """
        从文件读取数据
        返回：实际读取的字节数，失败返回负数
        """
        pass

    def cleanup(self):
        pass


class HyFileDDS(CliStorageInterface):
    """Hygon HCU HyFile DDS 实现"""

    def __init__(self) -> None:
        self.allow_cpu_fallback = True
        self.profile_enabled = False
        self._profile_local = threading.local()

    def set_profiling(self, enabled: bool) -> None:
        """Enable low-level handle/DMA timing for subsequent calls.

        Args:
            enabled: Whether new HIPFile handles should collect timings.
        """
        self.profile_enabled = enabled

    def take_last_profile(self) -> Optional[dict[str, float | int]]:
        """Return and clear the current thread's most recent call profile.

        Returns:
            The HIPFile lifecycle counters, or ``None`` when profiling is off
            or the current thread has not completed a profiled call.
        """
        profile_local = getattr(self, "_profile_local", None)
        if profile_local is None:
            return None
        profile = getattr(profile_local, "last", None)
        profile_local.last = None
        return profile

    def _capture_profile(self, hip_file: Optional[HIPFile]) -> None:
        if getattr(self, "profile_enabled", False) and hip_file is not None:
            self._profile_local.last = hip_file.get_profile()

    # Concurrent hipFileWrite calls have corrupted payloads. Allow concurrent
    # reads while keeping writes exclusive from both reads and other writes.
    _driver_condition = threading.Condition()
    _driver_readers = 0
    _driver_writer = False

    def initialize(self) -> bool:
        try:
            self.hiprt = ctypes.CDLL("libhiprtc.so")
            logger.info("HyFile DDS initialized")
            return True
        except Exception as e:
            # libhiprtc.so is ONLY used for the CPU-copy fallback when a GPUDirect
            # write/read is incomplete; the primary GPUDirect path does not need
            # it. So do NOT fail here (returning False makes create_xds_interface
            # raise, which would block an otherwise-working GPUDirect-only host).
            # If the fallback is ever needed with hiprt=None, write()/read() return
            # -1 loudly and the store fails there -- no silent corruption. The old
            # `f"HIP Runtime Error"` dropped the real exception e.
            logger.warning(
                f"libhiprtc.so unavailable ({e}); XDS CPU-copy fallback disabled, "
                "primary GPUDirect path unaffected."
            )
            self.hiprt = None
            return True

    def write(self, file_path: str, gpu_address: int, size: int,
              file_offset: int, dev_offset: int) -> int:
        # 尝试通过XDS写入
        hip_file = None
        try:
            with self._write_driver_guard():
                # A non-zero file offset means the caller already published a
                # header into the temporary file. Opening with "w" would truncate
                # that header before DMA; preserve it with the read/write mode.
                mode = "r+" if file_offset else "w"
                hip_file = HIPFile(
                    file_path,
                    mode,
                    enable_stats=getattr(self, "profile_enabled", False),
                    skip_buffer_registration=True,
                )
                with hip_file as f:
                    ret = f.write(gpu_address, size, file_offset, dev_offset)
                self._capture_profile(hip_file)
            if ret == size:
                logger.debug(f"HyFile write: {size} bytes")
                return size
            logger.warning(
                f"hyFile write incomplete ({ret}/{size} bytes), "
                f"falling back to CPU copy"
            )
        except Exception as e:
            self._capture_profile(hip_file)
            logger.warning(f"hyFile write failed: {e}, falling back to CPU copy")

        if not getattr(self, "allow_cpu_fallback", True):
            logger.error("HyFile direct write failed; CPU fallback is disabled")
            return -1

        # XDS写入失败后回退至POSIX写入
        if self.hiprt is None:
            logger.error("HIP runtime not available, cannot fallback")
            return -1

        fd = None
        mm = None
        arr = None

        # 尝试通过 HIP Runtim + mmap 写入
        try:
            fd = os.open(file_path, os.O_RDWR)
            os.ftruncate(fd, file_offset + size)
            # mmap 文件
            mm = mmap.mmap(
                fd, file_offset + size,
                prot=mmap.PROT_WRITE,
                flags=mmap.MAP_SHARED
            )
            os.close(fd)
            fd = None

            # 获取 CPU 内存地址
            arr = np.frombuffer(mm, dtype=np.uint8)
            cpu_addr = arr.__array_interface__["data"][0]

            # 使用 hipMemcpy 从 GPU 拷贝到 CPU
            res = self.hiprt.hipMemcpy(
                ctypes.c_void_p(cpu_addr + file_offset),        # dst: CPU 内存
                ctypes.c_void_p(gpu_address + dev_offset),      # src: GPU 内存
                ctypes.c_size_t(size),                          # size
                ctypes.c_int(MEMCPY_DEVICE_TO_HOST),  # direction: D2H
            )

            if res != 0:
                logger.error(f"Memcpy failed with code {res}")
                return -1

            logger.info(f"CPU fallback write successful: {size} bytes")
            return size
        except Exception as e:
            logger.error(f"CPU fallback write failed: {e}", exc_info=True)
            return -1

        finally:
            if arr is not None:
                del arr
            if mm is not None:
                mm.close()
            _close_file_descriptor(fd)

    def read(self, file_path: str, gpu_address: int, size: int,
             file_offset: int, dev_offset: int) -> int:

        if not os.path.exists(file_path):
            logger.error(f"No such file or directory: {file_path}, read failed")
            return -1

        # 尝试通过XDS读取
        hip_file = None
        try:
            with self._read_driver_guard():
                hip_file = HIPFile(
                    file_path,
                    "r",
                    enable_stats=getattr(self, "profile_enabled", False),
                    skip_buffer_registration=True,
                )
                with hip_file as f:
                    ret = f.read(gpu_address, size, file_offset, dev_offset)
                self._capture_profile(hip_file)
                if ret == size:
                    return ret
            logger.warning(
                f"hyFile read incomplete ({ret}/{size} bytes), "
                f"falling back to CPU copy"
            )
        except Exception as e:
            self._capture_profile(hip_file)
            logger.warning(f"hyFile read failed: {e}, falling back to CPU copy")

        if not getattr(self, "allow_cpu_fallback", True):
            logger.error("HyFile direct read failed; CPU fallback is disabled")
            return -1

        # XDS读取失败后回退至POSIX写入
        if self.hiprt is None:
            logger.error("HIP runtime not available, cannot fallback")
            return -1

        fd = None
        mm = None
        arr = None

        # 尝试通过 HIP Runtim + mmap 读取
        try:
            # 打开文件并 mmap
            fd = os.open(file_path, os.O_RDONLY)
            file_size = os.fstat(fd).st_size
            mm = mmap.mmap(
                fd, file_size,
                prot=mmap.PROT_READ,
                flags=mmap.MAP_PRIVATE | mmap.MAP_POPULATE,
            )
            os.close(fd)
            fd = None

            # 获取 CPU 内存地址
            arr = np.frombuffer(mm, dtype=np.uint8)
            cpu_addr = arr.__array_interface__["data"][0]

            # 使用 hipMemcpy 拷贝到 GPU
            res = self.hiprt.hipMemcpy(
                ctypes.c_void_p(gpu_address + dev_offset),
                ctypes.c_void_p(cpu_addr + file_offset),
                ctypes.c_size_t(size),
                ctypes.c_int(MEMCPY_HOST_TO_DEVICE),
            )

            if res != 0:
                logger.error(f"hipMemcpy failed with code {res}")
                return -1

            logger.info(f"CPU fallback successful: {size} bytes")
            return size

        except Exception as e:
            logger.error(f"CPU fallback failed: {e}", exc_info=True)
            return -1

        finally:
            if arr is not None:
                del arr
            if mm != None:
                mm.close()
            _close_file_descriptor(fd)

    @classmethod
    @contextmanager
    def _read_driver_guard(cls) -> Generator[None, None, None]:
        with cls._driver_condition:
            while cls._driver_writer:
                cls._driver_condition.wait()
            cls._driver_readers += 1
        try:
            yield
        finally:
            with cls._driver_condition:
                cls._driver_readers -= 1
                if cls._driver_readers == 0:
                    cls._driver_condition.notify_all()

    @classmethod
    @contextmanager
    def _write_driver_guard(cls) -> Generator[None, None, None]:
        with cls._driver_condition:
            while cls._driver_writer or cls._driver_readers:
                cls._driver_condition.wait()
            cls._driver_writer = True
        try:
            yield
        finally:
            with cls._driver_condition:
                cls._driver_writer = False
                cls._driver_condition.notify_all()


class Posix(CliStorageInterface):

    def __init__(self, root_path):
        self.use_odirect = True

        # Block size (for file system I/O)
        stat = os.statvfs(root_path)
        self.os_disk_bs = stat.f_bsize

    def initialize(self) -> bool:
        return True

    def write(self, path:str, kv_chunk:torch.Tensor, size:int,  buffer):
        assert len(buffer) == size
        tmp_path = path.replace(_DATA_FILE_SUFFIX, _DATA_FILE_SUFFIX_TMP)

        metadata = pack_metadata(
            kv_chunk.shape,
            kv_chunk.dtype,
            size
        )

        start_time = time.time()

        try:
            if size % self.os_disk_bs != 0 or not self.use_odirect:
                with open(tmp_path, "wb") as f:
                    written = f.write(buffer)
            else:
                fd = os.open(tmp_path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
                try:
                    written = os.write(fd, buffer)
                finally:
                    os.close(fd)
            if written != size:
                raise OSError(f"POSIX write incomplete: {written}/{size} bytes")

            os.rename(tmp_path, path)

            disk_write_time = time.time() - start_time
            logger.debug(
                f"clistorage write size: {size} bytes, "
                f"Bandwidth: {size / disk_write_time / 1e6:.2f} MB/s"
            )
        except Exception as e:
            logger.error(f"Error saving {tmp_path}: {e}", exc_info=True)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise e
        return metadata

    def read(self, buffer, path):
        start_time = time.time()
        size = len(buffer)
        fblock_aligned = size % self.os_disk_bs == 0
        if not fblock_aligned and self.use_odirect:
            logger.warning(
                "Cannot use O_DIRECT for this file, "
                "size is not aligned to clistorage block size."
            )

        try:
            if not fblock_aligned or not self.use_odirect:
                with open(path, "rb") as f:
                    bytes_read = f.readinto(buffer)
            else:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                with os.fdopen(fd, "rb", buffering=0) as fdo:
                    bytes_read = fdo.readinto(buffer)
        except FileNotFoundError:
            logger.warning(f"File not found on clistorage: {path}")
            return -1

        disk_read_time = time.time() - start_time
        logger.debug(
            f"clistorage read size: {size} bytes, "
            f"Bandwidth: {size / disk_read_time / 1e6:.2f} MB/s"
        )
        # Return the ACTUAL bytes read, not the requested size. A short read
        # (truncated/partial .data file) then surfaces as ret != size at the
        # caller (_load_bytes_from_storage_with_memory) and is treated as a miss
        # instead of silently accepting a buffer with a stale/garbage tail.
        return bytes_read


class CuFileGDS(CliStorageInterface):
    """NVIDIA cuFile GDS 实现"""
    def __init__(self):
        import cufile
        self.cufile = cufile
        self._driver = None

    def initialize(self) -> bool:
        try:
            self._driver = self.cufile.CuFileDriver()
            logger.info("cuFile GDS initialized")
            self.cudart = None
            self.cudart = ctypes.CDLL("libcudart.so")
            logger.info("CUDA runtime loaded for fallback")
            return True
        except Exception as e:
            logger.error(f"cuFile init error: {e}")
            return False

    def write(self, file_path: str, gpu_address: int, size: int,
              file_offset: int, dev_offset: int) -> int:
        try:
            # 尝试 cuFile 直接写入
            with self.cufile.CuFile(file_path, "r+", use_direct_io = True) as f:
                ret = f.write(
                    ctypes.c_void_p(gpu_address), size,
                    file_offset=file_offset, dev_offset=dev_offset
                )

            # 如果写入完整，直接返回
            if ret == size:
                return ret

            # 写入不完整，尝试回退
            logger.warning(
                f"cuFile write incomplete ({ret}/{size} bytes), "
                f"falling back to CPU copy"
            )

        except Exception as e:
            logger.warning(f"cuFile write failed: {e}, falling back to CPU copy")

        # ========== 回退到 CPU 拷贝 ==========
        if self.cudart is None:
            logger.error("CUDA runtime not available, cannot fallback")
            return -1

        fd = None
        mm = None
        arr = None
        try:

            fd = os.open(file_path, os.O_RDWR)
            os.ftruncate(fd, file_offset + size)
            # mmap 文件
            mm = mmap.mmap(
                fd, file_offset + size,
                prot=mmap.PROT_WRITE,
                flags=mmap.MAP_SHARED
            )
            os.close(fd)
            fd = None
            # 获取 CPU 内存地址
            arr = np.frombuffer(mm, dtype=np.uint8)
            cpu_addr = arr.__array_interface__["data"][0]

            # 使用 cudaMemcpy 从 GPU 拷贝到 CPU
            res = self.cudart.cudaMemcpy(
                ctypes.c_void_p(cpu_addr + file_offset),        # dst: CPU 内存
                ctypes.c_void_p(gpu_address + dev_offset),      # src: GPU 内存
                ctypes.c_size_t(size),                          # size
                ctypes.c_int(MEMCPY_DEVICE_TO_HOST),  # direction: D2H
            )

            if res != 0:
                logger.error(f"cudaMemcpy failed with code {res}")
                return -1
            # 确保数据写入磁盘
            logger.info(f"CPU fallback write successful: {size} bytes")
            return size

        except Exception as e:
            logger.error(f"CPU fallback write failed: {e}", exc_info=True)
            return -1
        finally:
            if arr is not None:
                del arr
            if mm != None:
                mm.close()
            _close_file_descriptor(fd)

    def read(self, file_path: str, gpu_address: int, size: int,
             file_offset: int, dev_offset: int) -> int:

        if not os.path.exists(file_path):
            logger.error(f"No such file or directory: {file_path}, read failed")
            return -1

        try:
            with self.cufile.CuFile(file_path, "r", use_direct_io = True) as f:
                ret = f.read(
                    ctypes.c_void_p(gpu_address), size,
                    file_offset=file_offset, dev_offset=dev_offset
                )

            if ret == size:
                return ret

            # 读取不完整，尝试回退
            logger.warning(
                f"cuFile read incomplete ({ret}/{size} bytes), "
                f"falling back to CPU copy"
            )

        except Exception as e:
            logger.warning(f"cuFile read failed: {e}, falling back to CPU copy")

        if self.cudart is None:
            logger.error("CUDA runtime not available, cannot fallback")
            return -1
        fd = None
        mm = None
        arr = None
        try:
            # 打开文件并 mmap
            fd = os.open(file_path, os.O_RDONLY)

            file_size = os.fstat(fd).st_size
            mm = mmap.mmap(
                fd, file_size,
                prot=mmap.PROT_READ,
                flags=mmap.MAP_PRIVATE | mmap.MAP_POPULATE,
            )
            os.close(fd)
            fd = None
            # 获取 CPU 内存地址
            arr = np.frombuffer(mm, dtype=np.uint8)
            cpu_addr = arr.__array_interface__["data"][0]

            # 使用 cudaMemcpy 拷贝到 GPU
            res = self.cudart.cudaMemcpy(
                ctypes.c_void_p(gpu_address + dev_offset),
                ctypes.c_void_p(cpu_addr + file_offset),
                ctypes.c_size_t(size),
                ctypes.c_int(MEMCPY_HOST_TO_DEVICE),
            )

            if res != 0:
                logger.error(f"cudaMemcpy failed with code {res}")
                return -1
            logger.info(f"CPU fallback successful: {size} bytes")
            return size

        except Exception as e:
            logger.error(f"CPU fallback failed: {e}", exc_info=True)
            return -1
        finally:
            if arr is not None:
                del arr
            if mm != None:
                mm.close()
            _close_file_descriptor(fd)


def check_environment():
    if torch.version.cuda is None:
        backend = "hyfile"
    else:
        backend = "cufile"

    logger.info(f"Detected XDS interface: {backend}")

    return backend


def create_xds_interface(backend: str):
    # 创建实例
    if backend == "cufile":
        xds = CuFileGDS()
    else:
        xds = HyFileDDS()

    if not xds.initialize():
        raise RuntimeError(f"Failed to initialize {backend}")

    return xds


class XdsBackend(AllocatorBackendInterface):
    """
    This is a backend that leverages Hygon's HyFile API or NVIDIA's cuFile API and to issue XDS requests
    directly to the storage.  In order to use it, users need to specify
    `xds_path` , `xds_buffer_size` and `max_xds_size` in their LMCache config.

    Cache Directory Structure created by this Backend:
    /{xds_path}/{first_level}/{second_level}/{data & metadata}
    This structure is semi-arbitrary.
    """

    def __init__(
        self,
        config: LMCacheEngineConfig,
        metadata: LMCacheMetadata,
        loop: asyncio.AbstractEventLoop,
        dst_device: str = "cuda",
        local_cpu_backend: LocalCPUBackend = None,
    ):
        assert os.path.exists(config.xds_path), (
            f"Xds path {config.xds_path} does not exist"
        )

        assert dst_device.startswith("cuda")
        super().__init__(dst_device=dst_device)

        self.config = config
        self.metadata = metadata
        self.loop = loop
        self.local_cpu_backend = local_cpu_backend
        extra_config = getattr(config, "extra_config", None) or {}
        self.cpu_store_enabled = _config_bool(
            extra_config.get("xds_cpu_store", False)
        )
        self.cpu_store_direct_retrieve_enabled = (
            self.cpu_store_enabled
            and _config_bool(extra_config.get("xds_direct_retrieve", False))
        )
        if (
            self.cpu_store_enabled
            and config.xds_buffer_size is None
            and not self.cpu_store_direct_retrieve_enabled
        ):
            raise ValueError(
                "xds_cpu_store requires xds_buffer_size unless "
                "xds_direct_retrieve is enabled"
            )
        if self.cpu_store_direct_retrieve_enabled and config.xds_buffer_size is not None:
            raise ValueError(
                "xds_cpu_store+xds_direct_retrieve requires xds_buffer_size=null"
            )
        if self.cpu_store_enabled and local_cpu_backend is None:
            raise ValueError(
                "xds_cpu_store requires an available LocalCPUBackend allocator"
            )
        self.io_profile_enabled = _config_bool(
            extra_config.get("direct_io_profile", False)
        )
        self.direct_store_pipeline_enabled = _config_bool(
            extra_config.get("direct_store_pipeline", True)
        )
        self.direct_retrieve_pipeline_enabled = _config_bool(
            extra_config.get("direct_retrieve_pipeline", True)
        )
        self._io_profile_local = threading.local()
        self.direct_only = _config_bool(extra_config.get("gds_direct_only", False))
        self._block_first_staging_allocator = None
        self._block_first_staging_obj: Optional[MemoryObj] = None
        self._block_first_lock = threading.Lock()
        self._direct_registration_lock = threading.Lock()
        self._direct_registrations: dict[tuple[str, int, int], object] = {}
        self._direct_edge_lock = threading.Lock()
        self._direct_edge_allocation: Optional[torch.Tensor] = None
        self._direct_edge_tensor: Optional[torch.Tensor] = None
        self._direct_hip_runtime = None
        self._direct_memcpy = None
        self._direct_driver_owned = False
        self._closed = False
        self.block_first_enabled = self._should_enable_block_first(config, metadata)
        enable_direct = _config_bool(extra_config.get("enable_direct_gds", False))
        self.direct_enabled = self.block_first_enabled and (
            enable_direct
            or self.direct_only
            or self.cpu_store_direct_retrieve_enabled
        )
        if self.direct_only and not self.direct_enabled:
            raise RuntimeError(
                "gds_direct_only requires a compatible block-first XDS setup"
            )
        if self.direct_only and config.store_location != self.__class__.__name__:
            raise RuntimeError("gds_direct_only requires store_location='XdsBackend'")
        if self.direct_only and config.enable_async_loading:
            raise RuntimeError("gds_direct_only does not support async loading")

        if self.config.xds_buffer_size is not None or self.direct_enabled:
            # 使用xds协议
            self.xds_interface = check_environment()
            if self.direct_enabled and self.xds_interface != "hyfile":
                if self.direct_only:
                    raise RuntimeError(
                        "XDS direct-only mode currently requires the HyFile backend"
                    )
                logger.warning(
                    "XDS direct I/O is unavailable for %s; using staged mode",
                    self.xds_interface,
                )
                self.direct_enabled = False
            self.memory_allocator = (
                None
                if self.direct_only or self.cpu_store_direct_retrieve_enabled
                else self.initialize_allocator(config, metadata, self.xds_interface)
            )
            self.xds = create_xds_interface(self.xds_interface)
            if isinstance(self.xds, HyFileDDS):
                self.xds.set_profiling(self.io_profile_enabled)
            if self.direct_only or self.cpu_store_direct_retrieve_enabled:
                if not isinstance(self.xds, HyFileDDS):
                    raise RuntimeError(
                        "direct XDS retrieve requires the HyFile backend"
                    )
                self.xds.allow_cpu_fallback = False
                if self.cpu_store_direct_retrieve_enabled:
                    logger.info(
                        "XDS CPU-store/direct-retrieve hybrid enabled; "
                        "no GPU staging allocator is configured"
                    )

            try:
                if self.direct_enabled and self.memory_allocator is None:
                    HIPFile.init_driver()
                    self._direct_driver_owned = True
                if self.memory_allocator is not None:
                    assert hasattr(self.memory_allocator, "base_pointer")
                    self.base_pointer = self.memory_allocator.base_pointer
                    if self.xds_interface == "hyfile":
                        # One real write+readback roundtrip before serving traffic:
                        # a driver that silently writes zeros would otherwise
                        # corrupt every cached entry (fail closed, not fail open).
                        _verify_hyfile_roundtrip(
                            self.xds, self.memory_allocator, config.xds_path
                        )
                    if self.block_first_enabled and not self.direct_only:
                        self._initialize_block_first_staging(metadata)
            except BaseException:
                # Failed construction: release every registered pool and the
                # driver instead of leaking them with the exception.
                self._close_block_first_staging()
                self.xds.cleanup()
                if self._direct_driver_owned:
                    HIPFile.close_driver()
                    self._direct_driver_owned = False
                if self.memory_allocator is not None:
                    self.memory_allocator.close()
                raise

        else:
            # 使用posix协议
            assert local_cpu_backend is not None
            self.memory_allocator = local_cpu_backend
            self.xds = Posix(config.xds_path)

        # Interface probing can reject HyFile direct I/O. Publish the final
        # per-direction capability only after that decision is complete.
        self.direct_store_enabled = self.direct_enabled and not self.cpu_store_enabled
        self.direct_retrieve_enabled = self.direct_enabled

        self.dst_device = dst_device

        if metadata.use_mla:
            self.fmt = MemoryFormat.KV_MLA_FMT
        else:
            self.fmt = MemoryFormat.KV_2LTD

        self.xds_path = config.xds_path

        self.hot_lock = threading.Lock()
        self.cache_policy = get_cache_policy(config.cache_policy)
        self.hot_cache = self.cache_policy.init_mutable_mapping()

        self.metadata_dirs: set[str] = set()

        self.rand = random.Random(self.dst_device)

        thread_count = config.get_extra_config_value('xds_io_threads', default_value=4)

        logger.info(f"Using {thread_count} threads for XDS I/O")
        self._thread_pool = ThreadPoolExecutor(
            max_workers=thread_count, thread_name_prefix="xds-io"
        )

        self.xds_worker = XdsWorker(self.loop)

        self.world_id = metadata.worker_id

        if metadata.use_mla:
            self.world_size = 1
        else:
            self.world_size = metadata.world_size

        if config.max_xds_size is not None:
            self.eviction = True
            self.max_xds_size = int(config.max_xds_size * 1024**3 )
            logger.info(f"Peer rank max_storage_size: {config.max_xds_size}GB, "
                        f"total max_storage_size: {config.max_xds_size * self.world_size}GB")
        else:
            self.eviction = False
            logger.info(f"config.max_xds_size is None, don't suppot eviction")

        # flag to use O_NOATIME during metadata file read for performance improvement
        self._use_noatime = True

        self.current_cache_size = 0.0
        self.keys_in_request: List[CacheEngineKey] = []

        self._scan_future = asyncio.run_coroutine_threadsafe(
            self._scan_metadata(), self.loop
        )

    @staticmethod
    def _should_enable_block_first(
        config: LMCacheEngineConfig, metadata: LMCacheMetadata
    ) -> bool:
        """Enable v1 only for the layout that can be represented losslessly."""
        extra = getattr(metadata, "kv_connector_extra_config", None) or {}
        requested = _config_bool(extra.get("lmcache_block_first", False))
        if not requested:
            return False

        reasons = []
        extra_config = getattr(config, "extra_config", None) or {}
        direct_only = _config_bool(extra_config.get("gds_direct_only", False))
        cpu_direct_retrieve = _config_bool(
            extra_config.get("xds_cpu_store", False)
        ) and _config_bool(extra_config.get("xds_direct_retrieve", False))
        if (
            _config_bool(extra_config.get("xds_cpu_store", False))
            and not cpu_direct_retrieve
        ):
            reasons.append("xds_cpu_store requires xds_direct_retrieve for block-first")
        if config.xds_buffer_size is None and not (direct_only or cpu_direct_retrieve):
            reasons.append("XDS direct mode is disabled")
        if metadata.use_mla:
            reasons.append("MLA is unsupported")
        if config.use_layerwise:
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
                "XDS block_first_v1 disabled; using legacy layout: %s",
                "; ".join(reasons),
            )
            return False
        logger.info("XDS block_first_v1 disk layout enabled")
        return True

    def _new_io_profile(
        self,
        operation: str,
        path: str,
        profile_id: Optional[str],
        expected_chunks: int,
    ) -> Optional[XDSIOProfile]:
        if not getattr(self, "io_profile_enabled", False) or expected_chunks <= 0:
            return None
        return XDSIOProfile(
            profile_id=str(profile_id or "unknown"),
            operation=operation,
            path=path,
            worker_id=self.metadata.worker_id,
            expected_chunks=expected_chunks,
        )

    @contextmanager
    def _activate_io_profile(
        self, profile: Optional[XDSIOProfile]
    ) -> Generator[None, None, None]:
        profile_local = getattr(self, "_io_profile_local", None)
        if profile_local is None:
            profile_local = threading.local()
            self._io_profile_local = profile_local
        previous = getattr(profile_local, "current", None)
        profile_local.current = profile
        try:
            yield
        finally:
            profile_local.current = previous

    def _current_io_profile(self) -> Optional[XDSIOProfile]:
        profile_local = getattr(self, "_io_profile_local", None)
        return getattr(profile_local, "current", None)

    def _profile_cuda_synchronize(self, device: torch.device) -> None:
        profile = self._current_io_profile()
        if profile is None:
            torch.cuda.synchronize(device)
            return
        started_ns = time.perf_counter_ns()
        try:
            torch.cuda.synchronize(device)
        finally:
            profile.record_sync(time.perf_counter_ns() - started_ns)

    @contextmanager
    def _profile_phase(self, name: str) -> Generator[None, None, None]:
        profile = self._current_io_profile()
        if profile is None:
            yield
            return
        started_ns = time.perf_counter_ns()
        try:
            yield
        finally:
            profile.record_phase(name, time.perf_counter_ns() - started_ns)

    def _take_xds_lifecycle_profile(self) -> Optional[dict[str, float | int]]:
        if isinstance(self.xds, HyFileDDS):
            return self.xds.take_last_profile()
        return None

    def _profiled_xds_write(
        self,
        file_path: str,
        gpu_address: int,
        size: int,
        file_offset: int,
        dev_offset: int,
    ) -> int:
        profile = self._current_io_profile()
        if profile is None:
            return self.xds.write(
                file_path, gpu_address, size, file_offset, dev_offset
            )
        started_ns = time.perf_counter_ns()
        ret = None
        try:
            ret = self.xds.write(
                file_path, gpu_address, size, file_offset, dev_offset
            )
            return ret
        finally:
            profile.record_xds_call(
                "write",
                size,
                ret,
                time.perf_counter_ns() - started_ns,
                self._take_xds_lifecycle_profile(),
            )

    def _profiled_xds_read(
        self,
        file_path: str,
        gpu_address: int,
        size: int,
        file_offset: int,
        dev_offset: int,
    ) -> int:
        profile = self._current_io_profile()
        if profile is None:
            return self.xds.read(
                file_path, gpu_address, size, file_offset, dev_offset
            )
        started_ns = time.perf_counter_ns()
        ret = None
        try:
            ret = self.xds.read(
                file_path, gpu_address, size, file_offset, dev_offset
            )
            return ret
        finally:
            profile.record_xds_call(
                "read",
                size,
                ret,
                time.perf_counter_ns() - started_ns,
                self._take_xds_lifecycle_profile(),
                file_path=file_path,
                file_offset=file_offset,
            )

    def _finish_io_profile(
        self,
        profile: Optional[XDSIOProfile],
        success: bool,
        fallback: bool = False,
    ) -> None:
        if profile is not None and profile.finish_chunk(success, fallback):
            logger.info(
                "LMCACHE_XDS_PROFILE %s",
                json.dumps(profile.as_dict(), separators=(",", ":")),
            )

    def _initialize_block_first_staging(self, metadata: LMCacheMetadata) -> None:
        """Allocate one small registered scratch chunk for layout conversion."""
        assert metadata.block_size is not None
        shapes = metadata.get_shapes(metadata.chunk_size)
        dtypes = metadata.get_dtypes()
        if len(shapes) != 1 or len(dtypes) != 1:
            raise RuntimeError("block_first_v1 requires exactly one KV layer group")
        payload_shape = block_first_shape(shapes[0], metadata.block_size)
        payload_nbytes = torch.Size(payload_shape).numel() * dtypes[0].itemsize
        allocator_size = align_up(payload_nbytes)
        if self.xds_interface == "hyfile":
            allocator = HyFileMemoryAllocator(size=allocator_size)
        else:
            allocator = CuFileMemoryAllocator(size=allocator_size)
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
        staging_obj = getattr(self, "_block_first_staging_obj", None)
        allocator = getattr(self, "_block_first_staging_allocator", None)
        if staging_obj is not None and staging_obj.is_valid():
            staging_obj.ref_count_down()
        if allocator is not None:
            allocator.close()
        self._block_first_staging_obj = None
        self._block_first_staging_allocator = None

    async def _scan_metadata(self):
        """分批扫描元数据，带进度反馈"""
        BATCH_SIZE = 11  # 每批处理的目录数

        start = time.perf_counter()

        # 第一步：收集所有待扫描的目录
        dirs_to_scan = []
        with os.scandir(self.xds_path) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                if len(entry.name) != _DIR_SIZE:
                    continue
                dirs_to_scan.append({
                    'path': os.path.join(self.xds_path, entry.name),
                    'l1_dir': entry.name
                })

        total_dirs = len(dirs_to_scan)
        processed = 0
        errors = 0

        logger.info(f"Starting metadata scan: {total_dirs} directories to process")

        # 第二步：分批处理
        for i in range(0, total_dirs, BATCH_SIZE):
            batch = dirs_to_scan[i:i + BATCH_SIZE]

            # 创建当前批次的任务
            tasks = [
                asyncio.to_thread(
                    self._scan_metadata_subdir,
                    item['path'],
                    item['l1_dir']
                )
                for item in batch
            ]

            # 等待当前批次完成
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # 统计错误
            for result in results:
                if isinstance(result, Exception):
                    errors += 1
                    logger.error(f"Batch scan error: {result}")

            processed += len(batch)
            progress = (processed / total_dirs) * 100

            logger.info(
                f"Scan progress: {processed}/{total_dirs} ({progress:.1f}%), "
                f"errors: {errors}, Cache entries: {len(self.hot_cache)}"
            )

        end = time.perf_counter()
        # with self.hot_lock:
        #     self.current_cache_size /= self.world_size
        logger.info(
            f"Metadata scan complete: {len(self.hot_cache)} entries "
            f"in {end - start:.2f}s, peer rank size:{self.current_cache_size}, errors: {errors}"
        )

    def _scan_metadata_subdir(self, path, l1_dir):
        target_suffix = _DATA_FILE_SUFFIX + _METADATA_FILE_SUFFIX
        with os.scandir(path) as it:
            for entry in it:
                if not entry.is_dir():
                    continue
                l2_dir = os.path.basename(entry.name)
                if len(l2_dir) != _DIR_SIZE:
                    continue
                with os.scandir(os.path.join(path, l2_dir)) as it2:
                    for fentry in it2:
                        if not fentry.is_file():
                            continue
                        if fentry.name.endswith(_BLOCK_FIRST_DATA_FILE_SUFFIX):
                            if not self.block_first_enabled:
                                continue
                            filename = os.path.basename(fentry.name)
                            key_str = unquote(
                                filename[: -len(_BLOCK_FIRST_DATA_FILE_SUFFIX)]
                            )
                            try:
                                key = CacheEngineKey.from_string(key_str)
                            except ValueError as e:
                                logger.error(
                                    f"Filename {filename} can't be converted "
                                    f"back into cache key: {e}"
                                )
                                continue
                            if key.worker_id != self.world_id:
                                continue
                            try:
                                self._read_block_first_metadata(
                                    key, fentry.path, l1_dir + l2_dir
                                )
                            except (BlockFirstFormatError, OSError) as e:
                                logger.error(
                                    "Invalid block-first metadata for %s, ignoring: %s",
                                    fentry.path,
                                    e,
                                )
                            continue
                        if not fentry.name.endswith(target_suffix):
                            # A live put may still own these tmp files (.KVDATA /
                            # .KVDATA.METADATA); skip instead of deleting, or a
                            # concurrent put's tmp file gets removed underneath it.
                            if fentry.name.endswith(
                                (
                                    _DATA_FILE_SUFFIX_TMP,
                                    _METADATA_FILE_SUFFIX_TMP,
                                    _BLOCK_FIRST_DATA_FILE_SUFFIX_TMP,
                                )
                            ):
                                continue
                            if not fentry.name.endswith(_DATA_FILE_SUFFIX):
                                os.remove(fentry.path)
                            continue
                        filename = os.path.basename(fentry.name)
                        key_str = unquote(filename[: -len(target_suffix)])
                        try:
                            key = CacheEngineKey.from_string(key_str)
                        except ValueError as e:
                            logger.error(
                                f"Filename {filename} can't be converted "
                                f"back into cache key: {e}"
                            )
                            continue
                        # Value equality, not identity: `is not` only happens to
                        # work for CPython's small-int cache (-5..256); a
                        # worker_id >= 257 (or any non-interned int) would make
                        # `is not` spuriously true and skip a valid on-disk entry.
                        if key.worker_id != self.world_id:
                            continue
                        try:
                            self._read_metadata(key, fentry.path, l1_dir + l2_dir)
                        except UnsupportedMetadataVersion:
                            logger.error(
                                "Unsupported metadata version for "
                                f"{fentry.path}, ignoring"
                            )

    # 修改时间： 2026.03.27
    # 修改人  ： 郭丰雅
    # 修改原因： 合并官方gds性能提升的代码 https://github.com/LMCache/LMCache/pull/2637/changes
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
            ):
                # fallback to normal open if O_NOATIME is not supported
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

    def _read_block_first_metadata(
        self, key: CacheEngineKey, filename: str, subdir_key: str
    ) -> XDSCacheMetadata:
        if getattr(self, "direct_enabled", False):
            disk_metadata, actual_size = self._read_direct_header_and_size(filename)
        else:
            with open(filename, "rb") as header_file:
                disk_metadata = unpack_block_first_header(
                    header_file.read(BLOCK_FIRST_HEADER_SIZE)
                )
            actual_size = os.path.getsize(filename)
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
        if actual_size != total_size:
            raise BlockFirstFormatError(
                f"block-first file size {actual_size} != expected {total_size}"
            )
        cache_metadata = XDSCacheMetadata(
            path=filename,
            size=total_size,
            shape=torch.Size(disk_metadata.logical_shape),
            dtype=disk_metadata.torch_dtype,
            fmt=MemoryFormat.KV_2LTD,
            disk_metadata=disk_metadata,
        )
        with self.hot_lock:
            previous = self.hot_cache.get(key)
            if previous is not None:
                self.current_cache_size -= previous.size
            self.metadata_dirs.add(subdir_key)
            self.hot_cache[key] = cache_metadata
            self.current_cache_size += cache_metadata.size
        return cache_metadata


    def _read_metadata(self, key, filename, subdir_key):

        shape, dtype, size = self._read_metadata_info(filename)

        # Reject metadata whose recorded size is inconsistent with shape/dtype,
        # else a corrupt size drives a wrong-length disk read (xds path: the DMA
        # overruns into neighboring entries). The posix allocator's alignment is
        # configurable, so only size >= raw is guaranteed; the xds allocator is
        # fixed at 4K alignment, so size must equal the aligned size exactly.
        raw = shape.numel() * dtype.itemsize
        if size < raw:
            raise UnsupportedMetadataVersion(
                f"metadata size {size} < raw payload {raw} for {filename}"
            )
        if self.config.xds_buffer_size is not None:
            expected = (raw + _ALIGNMENT - 1) // _ALIGNMENT * _ALIGNMENT
            if size != expected:
                raise UnsupportedMetadataVersion(
                    f"metadata size {size} != aligned {expected} for {filename}"
                )

        metadata = XDSCacheMetadata(
            filename.removesuffix(_METADATA_FILE_SUFFIX),
            size,      # 原始大小
            shape,
            dtype
        )

        with self.hot_lock:
            previous = self.hot_cache.get(key)
            if previous is not None and previous.disk_metadata is not None:
                # Startup scan order is filesystem-dependent. Never let a legacy
                # sidecar replace a validated block_first_v1 entry.
                return previous
            if previous is not None:
                self.current_cache_size -= previous.size
            self.metadata_dirs.add(subdir_key)
            self.hot_cache[key] = metadata
            self.current_cache_size += metadata.size

        return metadata

    def __str__(self):
        return self.__class__.__name__

    def contains(self, key: CacheEngineKey, pin: bool = False) -> bool:
        # TODO(Serapheim): implement pin() semantics
        with self.hot_lock:
            if key not in self.hot_cache:
                return False
            if pin:
                self.hot_cache[key].pin()
                self.keys_in_request.append(key)
        return True

    async def batched_async_contains(
        self,
        lookup_id: str,
        keys: list[CacheEngineKey],
        pin: bool = False,
    ) -> int:
        num_hit_counts = 0
        with self.hot_lock:
            for key in keys:
                if key not in self.hot_cache:
                    return num_hit_counts
                if pin:
                    self.hot_cache[key].pin()
                    self.keys_in_request.append(key)
                num_hit_counts += 1
        logger.info(f"batched_async_contains: {num_hit_counts}")
        return num_hit_counts

    def _try_to_read_metadata(self, key: CacheEngineKey) -> Optional[XDSCacheMetadata]:
        if self.block_first_enabled:
            path, subdir_key, _, _ = self._key_to_path(key, block_first=True)
            if os.path.exists(path):
                try:
                    return self._read_block_first_metadata(key, path, subdir_key)
                except (BlockFirstFormatError, OSError) as e:
                    logger.error(
                        "Invalid block-first metadata for %s, falling back: %s",
                        path,
                        e,
                    )
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
        block_first: bool = False,
    ) -> Tuple[str, str, str, str]:
        hash_str = str(key.chunk_hash)
        l1_dir = hash_str[:_DIR_SIZE]
        l2_dir = hash_str[_DIR_SIZE:_DIR_SIZE*2]
        # Encode the key into a flat, reversible filename via urllib quote
        # (safe=""). The old scheme did key_str.replace("/", "_") and asserted
        # the key had no "_", which rejected any model whose key string contains
        # "_" (e.g. model names with underscores -> AssertionError, so XDS store
        # raised before writing any file). quote/unquote round-trips "_", "/"
        # and every other character. Decoded back via unquote() in the scan.
        key_str = key.to_string()
        suffix = _BLOCK_FIRST_DATA_FILE_SUFFIX if block_first else _DATA_FILE_SUFFIX
        path = os.path.join(
                self.xds_path,
                l1_dir,
                l2_dir,
                quote(key_str, safe="") + suffix
            )
        return (path, l1_dir + l2_dir, l1_dir, l2_dir)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return self.xds_worker.exists_in_put_tasks(key)

    def _use_block_first_for_tensor(self, tensor: torch.Tensor) -> bool:
        """Use v1 only for a complete LMCache chunk; partials stay legacy."""
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
            raise BlockFirstFormatError("direct XDS requires one KV layer group")
        logical = torch.empty(shapes[0], dtype=dtypes[0], device="meta")
        return self._make_block_first_metadata(logical)

    def _validate_direct_descriptor(
        self,
        descriptor: DirectIODescriptor,
        direction: DirectIODirection,
    ) -> None:
        if not self.direct_enabled:
            raise ValueError("direct XDS is disabled")
        if descriptor.direction != direction:
            raise ValueError("direct I/O direction mismatch")
        descriptor.validate(require_cuda=True, require_alignment=True)
        if descriptor.engine_group_id != 0:
            raise ValueError("0.4.2 direct XDS only supports engine group 0")
        if self.metadata.block_size != descriptor.block_size:
            raise ValueError("direct I/O block size does not match engine metadata")
        if not _devices_match(self.dst_device, descriptor.device):
            raise ValueError("direct I/O allocation is on the wrong CUDA device")

        shapes = self.metadata.get_shapes(self.metadata.chunk_size)
        dtypes = self.metadata.get_dtypes()
        if len(shapes) != 1 or len(dtypes) != 1:
            raise ValueError("direct XDS requires one KV layer group")
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
        with self._direct_registration_lock:
            for allocation in descriptor.registration_allocations:
                identity = allocation.identity
                if identity in self._direct_registrations:
                    continue
                HIPFile.register_global_buffer(allocation.base, allocation.nbytes)
                self._direct_registrations[identity] = allocation.lifetime_token

    def _close_direct_registrations(self) -> None:
        with self._direct_registration_lock:
            for identity in list(self._direct_registrations):
                HIPFile.deregister_global_buffer(identity[1])
                del self._direct_registrations[identity]

    def _get_direct_edge_staging(self, block_nbytes: int) -> torch.Tensor:
        edge_tensor = self._direct_edge_tensor
        if edge_tensor is not None:
            if edge_tensor.numel() != block_nbytes:
                raise RuntimeError("direct edge block size changed at runtime")
            return edge_tensor

        allocation = torch.empty(
            block_nbytes + _ALIGNMENT - 1,
            dtype=torch.uint8,
            device=self.dst_device,
        )
        byte_offset = (-allocation.data_ptr()) % _ALIGNMENT
        edge_tensor = allocation[byte_offset : byte_offset + block_nbytes]
        try:
            HIPFile.register_global_buffer(edge_tensor.data_ptr(), block_nbytes)
        except BaseException:
            # Local allocations are released when the failed call unwinds.
            raise
        self._direct_edge_allocation = allocation
        self._direct_edge_tensor = edge_tensor
        return edge_tensor

    def _close_direct_edge_staging(self) -> None:
        with self._direct_edge_lock:
            edge_tensor = self._direct_edge_tensor
            if edge_tensor is not None:
                HIPFile.deregister_global_buffer(edge_tensor.data_ptr())
            self._direct_edge_tensor = None
            self._direct_edge_allocation = None

    def _device_to_device_copy(self, destination: int, source: int, size: int) -> None:
        if self._direct_memcpy is None:
            self._direct_hip_runtime = ctypes.CDLL("libgalaxyhip.so.5")
            hip_memcpy = self._direct_hip_runtime.hipMemcpy
            hip_memcpy.argtypes = [
                ctypes.c_void_p,
                ctypes.c_void_p,
                ctypes.c_size_t,
                ctypes.c_int,
            ]
            hip_memcpy.restype = ctypes.c_int

            def _memcpy_d2d(destination: int, source: int, size: int) -> int:
                return hip_memcpy(
                    ctypes.c_void_p(destination),
                    ctypes.c_void_p(source),
                    ctypes.c_size_t(size),
                    ctypes.c_int(3),
                )

            self._direct_memcpy = _memcpy_d2d
        result = self._direct_memcpy(destination, source, size)
        if result not in (None, 0):
            raise RuntimeError(f"hipMemcpyDeviceToDevice failed with code {result}")

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
            self.hot_cache[key] = XDSCacheMetadata(
                path=path,
                size=BLOCK_FIRST_HEADER_SIZE + disk_metadata.io_nbytes,
                shape=torch.Size(disk_metadata.logical_shape),
                dtype=disk_metadata.torch_dtype,
                fmt=MemoryFormat.KV_2LTD,
                disk_metadata=disk_metadata,
            )

    def _save_direct_xds(
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
            with self._profile_phase("header"):
                with open(tmp_path, "wb") as header_file:
                    written = header_file.write(header)
            if written != BLOCK_FIRST_HEADER_SIZE:
                raise RuntimeError(f"block-first header write incomplete: {written}")

            self._profile_cuda_synchronize(descriptor.device)
            for run in descriptor.runs:
                size = run.block_count * descriptor.block_nbytes
                ret = self._profiled_xds_write(
                    tmp_path,
                    descriptor.allocation_base,
                    size,
                    descriptor.file_payload_offset
                    + run.logical_block_start * descriptor.block_nbytes,
                    run.physical_block_start * descriptor.block_nbytes,
                )
                if ret != size:
                    raise RuntimeError(f"direct XDS write incomplete: {ret}/{size}")

            expected_size = BLOCK_FIRST_HEADER_SIZE + disk_metadata.io_nbytes
            with self._profile_phase("stat"):
                actual_size = os.path.getsize(tmp_path)
            if actual_size != expected_size:
                raise RuntimeError("direct XDS file size mismatch")
            with self._profile_phase("rename"):
                os.replace(tmp_path, path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            raise
        return disk_metadata

    def _load_direct_xds(
        self,
        path: str,
        disk_metadata: BlockFirstDiskMetadata,
        descriptor: DirectIODescriptor,
    ) -> None:
        self._prepare_direct_xds(path, disk_metadata, descriptor)
        self._transfer_direct_xds(path, descriptor)

    def _read_direct_header_and_size(
        self, path: str
    ) -> tuple[BlockFirstDiskMetadata, int]:
        header_started_ns = time.perf_counter_ns()
        fd = os.open(path, os.O_RDONLY)
        try:
            header = os.pread(fd, BLOCK_FIRST_HEADER_SIZE, 0)
            header_ns = time.perf_counter_ns() - header_started_ns
            stat_started_ns = time.perf_counter_ns()
            fd_stat = os.fstat(fd)
            path_stat = os.stat(path)
            stat_ns = time.perf_counter_ns() - stat_started_ns
            if (
                fd_stat.st_dev != path_stat.st_dev
                or fd_stat.st_ino != path_stat.st_ino
                or fd_stat.st_size != path_stat.st_size
            ):
                raise BlockFirstFormatError(
                    "block-first file changed while its header was read"
                )
        finally:
            os.close(fd)
        unpack_started_ns = time.perf_counter_ns()
        disk_metadata = unpack_block_first_header(header)
        header_ns += time.perf_counter_ns() - unpack_started_ns
        profile = self._current_io_profile()
        if profile is not None:
            profile.record_phase("header", header_ns)
            profile.record_phase("stat", stat_ns)
        return disk_metadata, fd_stat.st_size

    def _prepare_direct_xds(
        self,
        path: str,
        disk_metadata: BlockFirstDiskMetadata,
        descriptor: DirectIODescriptor,
    ) -> None:
        """Validate an immutable direct file without touching device memory."""
        self._validate_direct_descriptor(descriptor, DirectIODirection.RETRIEVE)
        current_metadata, actual_size = self._read_direct_header_and_size(path)
        if current_metadata != disk_metadata:
            raise BlockFirstFormatError("block-first header changed after indexing")
        expected_size = BLOCK_FIRST_HEADER_SIZE + disk_metadata.io_nbytes
        if actual_size != expected_size:
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

    def _transfer_direct_xds(
        self,
        path: str,
        descriptor: DirectIODescriptor,
    ) -> None:
        """Transfer one prepared file in the caller's deterministic order."""
        self._ensure_direct_registration(descriptor)
        self._profile_cuda_synchronize(descriptor.device)
        if descriptor.byte_segments:
            allocations = descriptor.registration_allocations
            for segment in descriptor.byte_segments:
                allocation = allocations[segment.allocation_index]
                ret = self._profiled_xds_read(
                    path,
                    allocation.base,
                    segment.nbytes,
                    segment.file_offset,
                    segment.allocation_offset,
                )
                if ret != segment.nbytes:
                    raise RuntimeError(
                        "direct XDS segment read incomplete: "
                        f"{ret}/{segment.nbytes}"
                    )
            self._profile_cuda_synchronize(descriptor.device)
            return
        if descriptor.leading_edge is not None:
            with self._direct_edge_lock:
                edge_tensor = self._get_direct_edge_staging(descriptor.block_nbytes)
                edge = descriptor.leading_edge
                ret = self._profiled_xds_read(
                    path,
                    edge_tensor.data_ptr(),
                    descriptor.block_nbytes,
                    descriptor.file_payload_offset
                    + edge.logical_block * descriptor.block_nbytes,
                    0,
                )
                if ret != descriptor.block_nbytes:
                    raise RuntimeError(
                        "direct XDS edge read incomplete: "
                        f"{ret}/{descriptor.block_nbytes}"
                    )
                self._copy_direct_leading_edge(
                    descriptor,
                    edge_tensor.data_ptr(),
                )
        for run in descriptor.runs:
            size = run.block_count * descriptor.block_nbytes
            ret = self._profiled_xds_read(
                path,
                descriptor.allocation_base,
                size,
                descriptor.file_payload_offset
                + run.logical_block_start * descriptor.block_nbytes,
                run.physical_block_start * descriptor.block_nbytes,
            )
            if ret != size:
                raise RuntimeError(f"direct XDS read incomplete: {ret}/{size}")
        self._profile_cuda_synchronize(descriptor.device)

    def batched_direct_put(
        self,
        keys: Sequence[CacheEngineKey],
        descriptors: Sequence[Optional[DirectIODescriptor]],
        profile_id: Optional[str] = None,
    ) -> list[bool]:
        if not getattr(self, "direct_store_enabled", self.direct_enabled):
            return [False] * len(keys)
        profile = self._new_io_profile("store", "direct", profile_id, len(keys))
        if profile is not None:
            for descriptor in descriptors:
                profile.record_descriptor(descriptor)
        results = [False] * len(keys)
        jobs: list[_DirectStoreJob] = []
        pipeline_enabled = (
            getattr(self, "direct_store_pipeline_enabled", False)
            and len(keys) > 1
        )
        for index, (key, descriptor) in enumerate(
            zip(keys, descriptors, strict=True)
        ):
            if descriptor is None or not self.direct_enabled:
                self._finish_io_profile(
                    profile, False, fallback=descriptor is not None
                )
                continue
            with self.hot_lock:
                existing = self.hot_cache.get(key)
            if existing is not None and existing.disk_metadata is not None:
                self.cache_policy.update_on_hit(key, self.hot_cache)
                results[index] = True
                self._finish_io_profile(profile, True)
                continue
            if existing is not None:
                self.remove(key)
            if self.exists_in_put_tasks(key):
                self._finish_io_profile(profile, False, fallback=True)
                continue

            self.xds_worker.insert_put_task(key)
            required_size = 0
            reserved = False
            policy_updated = False
            queued = False
            try:
                disk_metadata = self._make_direct_block_first_metadata()
                required_size = BLOCK_FIRST_HEADER_SIZE + disk_metadata.io_nbytes
                reserved = (
                    self._eviction(required_size)
                    if self.eviction
                    else self._reserve_direct_size(required_size)
                )
                if not reserved:
                    self._finish_io_profile(profile, False, fallback=True)
                    continue
                self.cache_policy.update_on_put(key)
                policy_updated = True
                path, _, l1_dir, l2_dir = self._key_to_path(key, block_first=True)
                directory = os.path.join(self.xds_path, l1_dir, l2_dir)
                if pipeline_enabled:
                    future = self._thread_pool.submit(
                        self._run_direct_store_io,
                        path,
                        directory,
                        descriptor,
                        profile,
                    )
                    jobs.append(
                        _DirectStoreJob(
                            index=index,
                            key=key,
                            path=path,
                            required_size=required_size,
                            future=future,
                        )
                    )
                    queued = True
                    continue
                disk_metadata = self._run_direct_store_io(
                    path, directory, descriptor, profile
                )
                self._insert_direct_key(key, path, disk_metadata)
                results[index] = True
                self._finish_io_profile(profile, True)
            except (BlockFirstFormatError, OSError, RuntimeError, ValueError) as exc:
                self._rollback_direct_store(
                    key, required_size if reserved else 0, policy_updated
                )
                logger.warning("Direct XDS store fallback for %s: %s", key, exc)
                self._finish_io_profile(profile, False, fallback=True)
            finally:
                if not queued:
                    self.xds_worker.remove_put_task(key)

        for job in jobs:
            try:
                disk_metadata = job.future.result()
                self._insert_direct_key(job.key, job.path, disk_metadata)
                results[job.index] = True
                self._finish_io_profile(profile, True)
            except (BlockFirstFormatError, OSError, RuntimeError, ValueError) as exc:
                self._rollback_direct_store(job.key, job.required_size, True)
                logger.warning(
                    "Pipelined direct XDS store fallback for %s: %s",
                    job.key,
                    exc,
                )
                self._finish_io_profile(profile, False, fallback=True)
            finally:
                self.xds_worker.remove_put_task(job.key)
        return results

    def _run_direct_store_io(
        self,
        path: str,
        directory: str,
        descriptor: DirectIODescriptor,
        profile: Optional[XDSIOProfile],
    ) -> BlockFirstDiskMetadata:
        """Run one file transaction; HyFile still serializes the DMA itself."""
        os.makedirs(directory, exist_ok=True)
        with self._activate_io_profile(profile):
            return self._save_direct_xds(path, descriptor)

    def _rollback_direct_store(
        self,
        key: CacheEngineKey,
        required_size: int,
        policy_updated: bool,
    ) -> None:
        if required_size:
            with self.hot_lock:
                self.current_cache_size -= required_size
        if policy_updated:
            self.cache_policy.update_on_force_evict(key)

    def _reserve_direct_size(self, required_size: int) -> bool:
        with self.hot_lock:
            self.current_cache_size += required_size
        return True

    def batched_direct_get(
        self,
        keys: Sequence[CacheEngineKey],
        descriptors: Sequence[Optional[DirectIODescriptor]],
        profile_id: Optional[str] = None,
    ) -> list[bool]:
        profile = self._new_io_profile("retrieve", "direct", profile_id, len(keys))
        if profile is not None:
            for descriptor in descriptors:
                profile.record_descriptor(descriptor)
        results = [False] * len(keys)
        jobs: list[_DirectRetrieveJob] = []
        pipeline_enabled = (
            getattr(self, "direct_retrieve_pipeline_enabled", False)
            and len(keys) > 1
        )
        for index, (key, descriptor) in enumerate(
            zip(keys, descriptors, strict=True)
        ):
            if descriptor is None or not self.direct_enabled:
                self._finish_io_profile(
                    profile, False, fallback=descriptor is not None
                )
                continue
            with self.hot_lock:
                entry = self.hot_cache.get(key)
            if entry is None or entry.disk_metadata is None:
                self._finish_io_profile(profile, False, fallback=True)
                continue
            try:
                if pipeline_enabled:
                    future = self._thread_pool.submit(
                        self._prepare_direct_retrieve_io,
                        entry.path,
                        entry.disk_metadata,
                        descriptor,
                        profile,
                    )
                    jobs.append(
                        _DirectRetrieveJob(
                            index=index,
                            key=key,
                            path=entry.path,
                            descriptor=descriptor,
                            future=future,
                        )
                    )
                    continue
                with self._activate_io_profile(profile):
                    self._load_direct_xds(
                        entry.path, entry.disk_metadata, descriptor
                    )
                self.cache_policy.update_on_hit(key, self.hot_cache)
                results[index] = True
                self._finish_io_profile(profile, True)
            except (BlockFirstFormatError, OSError, RuntimeError, ValueError) as exc:
                logger.warning("Direct XDS retrieve fallback for %s: %s", key, exc)
                self._finish_io_profile(profile, False, fallback=True)

        prepared_jobs: list[_DirectRetrieveJob] = []
        for job in jobs:
            try:
                job.future.result()
                self._ensure_direct_registration(job.descriptor)
                prepared_jobs.append(job)
            except (BlockFirstFormatError, OSError, RuntimeError, ValueError) as exc:
                logger.warning(
                    "Pipelined direct XDS retrieve fallback for %s: %s",
                    job.key,
                    exc,
                )
                self._finish_io_profile(profile, False, fallback=True)

        parallel = self._can_parallel_direct_retrieve(prepared_jobs)
        transfer_futures = (
            [
                self._thread_pool.submit(
                    self._transfer_direct_retrieve_io,
                    job.path,
                    job.descriptor,
                    profile,
                )
                for job in prepared_jobs
            ]
            if parallel
            else [None] * len(prepared_jobs)
        )
        for job, future in zip(prepared_jobs, transfer_futures, strict=True):
            try:
                if future is None:
                    with self._activate_io_profile(profile):
                        self._transfer_direct_xds(job.path, job.descriptor)
                else:
                    future.result()
                self.cache_policy.update_on_hit(job.key, self.hot_cache)
                results[job.index] = True
                self._finish_io_profile(profile, True)
            except (BlockFirstFormatError, OSError, RuntimeError, ValueError) as exc:
                logger.warning(
                    "Pipelined direct XDS retrieve fallback for %s: %s",
                    job.key,
                    exc,
                )
                self._finish_io_profile(profile, False, fallback=True)
        return results

    def _prepare_direct_retrieve_io(
        self,
        path: str,
        disk_metadata: BlockFirstDiskMetadata,
        descriptor: DirectIODescriptor,
        profile: Optional[XDSIOProfile],
    ) -> None:
        with self._activate_io_profile(profile):
            self._prepare_direct_xds(path, disk_metadata, descriptor)

    def _transfer_direct_retrieve_io(
        self,
        path: str,
        descriptor: DirectIODescriptor,
        profile: Optional[XDSIOProfile],
    ) -> None:
        with self._activate_io_profile(profile):
            self._transfer_direct_xds(path, descriptor)

    def _can_parallel_direct_retrieve(
        self, jobs: Sequence[_DirectRetrieveJob]
    ) -> bool:
        """Only overlap transfers to distinct physical KV blocks."""
        if len(jobs) < 2:
            return False
        seen_blocks: set[int] = set()
        for job in jobs:
            descriptor = job.descriptor
            block_ids = descriptor.physical_block_ids
            if descriptor.leading_edge is not None or not block_ids:
                return False
            block_set = set(block_ids)
            if len(block_set) != len(block_ids) or seen_blocks.intersection(block_set):
                return False
            seen_blocks.update(block_set)
        return True

    def _required_store_size(self, memory_obj: MemoryObj) -> int:
        tensor = memory_obj.tensor
        assert tensor is not None
        if self._use_block_first_for_tensor(tensor):
            return BLOCK_FIRST_HEADER_SIZE + align_up(memory_obj.get_size())
        return memory_obj.get_physical_size()

    def submit_put_task(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        io_profile: Optional[XDSIOProfile] = None,
    ) -> Optional[Future]:
        # This backend fires the put as a coroutine and always returns None (like
        # local_disk_backend); the annotation is Optional[Future] to match reality
        # rather than the old bare `-> Future` that never held.
        if (
            self.memory_allocator is None
            and not self.cpu_store_enabled
        ):  # MLA non-first-rank: inert, no offload
            self._finish_io_profile(io_profile, False)
            return None
        assert memory_obj.tensor is not None

        if self.exists_in_put_tasks(key):
            logger.debug(f"Put task for {key} is already in progress.")
            self._finish_io_profile(io_profile, False)
            return None

        with self.hot_lock:
            existing = self.hot_cache.get(key)

        if existing is not None and existing.disk_metadata is not None:
            self.cache_policy.update_on_hit(key, self.hot_cache)
            self._finish_io_profile(io_profile, True)
            return None

        self.xds_worker.insert_put_task(key)

        # Accounting is unified on the *data* physical size only (matches the
        # official local_disk_backend `required_size = get_physical_size()` and
        # the per-entry `.size` stored in insert_key / used at load-time
        # `ret != size`). The metadata sidecar (~<= _METADATA_MAX_SIZE bytes) is
        # NOT added here: it was double-counted before (add included +512 while
        # every subtract used `.size` == phys), leaking _METADATA_MAX_SIZE per
        # evicted key and drifting current_cache_size above max_xds_size — a leak
        # the Step-4 72->512 bump amplified 7x. Keep a single, consistent unit.
        required_size = self._required_store_size(memory_obj)
        evict_success = True
        if self.eviction:
            evict_success = self._eviction(required_size)
        else:
            self.current_cache_size += required_size

        if not evict_success:
            self.xds_worker.remove_put_task(key)
            self._finish_io_profile(io_profile, False)
            return None

        self.cache_policy.update_on_put(key)
        memory_obj.ref_count_up()

        asyncio.run_coroutine_threadsafe(
            self.xds_worker.submit_task(
                "put",
                self._async_save_bytes_to_storage,
                key=key,
                memory_obj=memory_obj,
                reserved_size=required_size,
                io_profile=io_profile,
            ),
            self.loop,
        )

    def _eviction(self, required_size: int) -> bool:
        # Two-phase eviction: the memory index (hot_cache pop + counter + policy) is updated
        # INSIDE the lock so each candidate actually advances the LRU cursor, and
        # the slow file unlink is done OUTSIDE the lock. We must NOT call
        # self.remove()/batched_remove() here — remove() re-acquires the
        # non-reentrant hot_lock (deadlock) and would double-decrement the
        # counter for keys already accounted below.
        evicted_paths: list[str] = []
        evict_success = True
        with self.hot_lock:
            while self.current_cache_size + required_size > self.max_xds_size:
                evict_keys = self.cache_policy.get_evict_candidates(
                    self.hot_cache, num_candidates=1
                )
                if not evict_keys:
                    logger.warning(
                        "No eviction candidates found. Disk space under pressure."
                    )
                    evict_success = False
                    break

                for evict_key in evict_keys:
                    # pop INSIDE the loop so get_evict_candidates advances to the
                    # next-oldest key next iteration instead of returning this
                    # same key forever (the original bug: index update lived
                    # outside the loop, so >=2-key evictions looped on one key).
                    meta = self.hot_cache.pop(evict_key)
                    self.cache_policy.update_on_force_evict(evict_key)
                    self.current_cache_size -= meta.size
                    evicted_paths.append(meta.path)

            if evict_success:
                self.current_cache_size += required_size
        # Slow remote-storage unlink outside the lock (satisfies the original
        # TODO). Each file is removed independently so a missing sidecar does not
        # orphan the data blob (or vice versa).
        for path in evicted_paths:
            for p in (path + _METADATA_FILE_SUFFIX, path):
                try:
                    os.remove(p)
                except OSError:
                    pass

        return evict_success

    def batched_submit_put_task(
        self,
        keys: Sequence[CacheEngineKey],
        memory_objs: List[MemoryObj],
        transfer_spec=None,
        profile_id: Optional[str] = None,
    ) -> None:
        chunk_count = min(len(keys), len(memory_objs))
        io_profile = self._new_io_profile(
            "store", "staged", profile_id, chunk_count
        )
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            if io_profile is not None:
                io_profile.record_staged_chunk(self._required_store_size(memory_obj))
            self.submit_put_task(key, memory_obj, io_profile=io_profile)

    def _async_save_bytes_to_storage(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        reserved_size: Optional[int] = None,
        io_profile: Optional[XDSIOProfile] = None,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to storage.
        """
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None

        use_block_first = self._use_block_first_for_tensor(kv_chunk)
        size = memory_obj.get_physical_size()
        path, _, l1_dir, l2_dir = self._key_to_path(
            key, block_first=use_block_first
        )

        previous_profile = getattr(self._io_profile_local, "current", None)
        self._io_profile_local.current = io_profile
        success = False
        try:
            os.makedirs(
                os.path.join(self.xds_path, l1_dir, l2_dir), exist_ok=True
            )
            disk_metadata = None
            if self.cpu_store_enabled:
                if not use_block_first:
                    raise BlockFirstFormatError(
                        "xds_cpu_store + xds_direct_retrieve requires a complete "
                        "block_first_v1 chunk"
                    )
                disk_metadata = self._save_cpu_block_first_xds(path, memory_obj)
            elif use_block_first:
                disk_metadata = self._save_block_first_xds(path, kv_chunk)
            elif self.config.xds_buffer_size is not None:
                # xds 写入
                metadata = self._save_xds(
                        path,
                        kv_chunk,
                        size,
                        self.base_pointer,
                        memory_obj.metadata.address)
            else:
                # posix 写入
                buffer = memory_obj.byte_array
                metadata = self._save_posix(
                        path,
                        kv_chunk,
                        size,
                        buffer)

            if not use_block_first:
                save_metadata(path, _METADATA_FILE_SUFFIX_TMP, metadata)

            # Publish the in-memory index only after both data and metadata are
            # committed; readers/evictors must never see a half-committed entry
            # (visible in hot_cache => both files are on disk).
            self.insert_key(
                key,
                memory_obj,
                path=path,
                disk_metadata=disk_metadata,
                stored_size=reserved_size,
            )
            success = True

        except Exception as e:
            logger.error(f"Failed to save {key}: {e}", exc_info=True)
            cleanup_paths = [
                path + _BLOCK_FIRST_DATA_FILE_SUFFIX_TMP,
                # DATA 的临时路径
                path.replace(_DATA_FILE_SUFFIX, _DATA_FILE_SUFFIX_TMP),
                # metadata 的临时路径
                path.replace(_DATA_FILE_SUFFIX, _DATA_FILE_SUFFIX_TMP)
                + _METADATA_FILE_SUFFIX_TMP,
                # metadata 的正式路径
                path + _METADATA_FILE_SUFFIX,
            ]
            # block_first_v1 publishes metadata and payload atomically in one
            # self-describing file. If only the in-memory index update fails,
            # keep that valid file so startup scanning can recover it. The
            # legacy two-file path retains its existing rollback behavior.
            if not use_block_first:
                cleanup_paths.append(path)
            for cleanup_path in cleanup_paths:
                if os.path.exists(cleanup_path):
                    try:
                        os.remove(cleanup_path)
                    except OSError:
                        pass
            with self.hot_lock:
                self.hot_cache.pop(key,None)
                # Undo exactly the reservation made in submit_put_task.
                self.current_cache_size -= (
                    reserved_size
                    if reserved_size is not None
                    else self._required_store_size(memory_obj)
                )
        finally:
            self._io_profile_local.current = previous_profile
            memory_obj.ref_count_down()
            self.xds_worker.remove_put_task(key)
            self._finish_io_profile(io_profile, success)

    def insert_key(
        self,
        key: CacheEngineKey,
        memory_obj: MemoryObj,
        *,
        path: Optional[str] = None,
        disk_metadata: Optional[BlockFirstDiskMetadata] = None,
        stored_size: Optional[int] = None,
    ) -> None:
        with self.hot_lock:
            if key in self.hot_cache:
                self.cache_policy.update_on_hit(key,self.hot_cache)
            else:
                if path is None:
                    path, _, _, _ = self._key_to_path(key)
                size = (
                    BLOCK_FIRST_HEADER_SIZE + disk_metadata.io_nbytes
                    if disk_metadata is not None
                    else (
                        stored_size
                        if stored_size is not None
                        else memory_obj.get_physical_size()
                    )
                )
                shape = memory_obj.metadata.shape
                dtype = memory_obj.metadata.dtype
                self.hot_cache[key] = XDSCacheMetadata(
                    path=path,
                    size=size,
                    shape=shape,
                    dtype=dtype,
                    fmt=memory_obj.metadata.fmt,
                    disk_metadata=disk_metadata,
                )

    async def _async_load_bytes_from_storage(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[MemoryObj]:
        return self._load_bytes_from_storage_with_allocation(key, path, dtype, shape)

    def get_blocking(
        self,
        key: CacheEngineKey,
    ) -> Optional[MemoryObj]:
        if self.memory_allocator is None:  # MLA non-first-rank: inert
            return None
        with self.hot_lock:
            entry = self.hot_cache.get(key)
        if entry is None:
            return None
        self.cache_policy.update_on_hit(key, self.hot_cache)
        path = entry.path
        dtype = entry.dtype
        shape = entry.shape
        assert dtype is not None
        assert shape is not None
        return self._load_bytes_from_storage_with_allocation(key, path, dtype, shape)

    def submit_prefetch_task(self, key: CacheEngineKey) -> bool:
        # TODO(Jiayi): Need to modify this when prefetch interface is determined.
        # Mirror GdsBackend: prefetch is not implemented yet, always returns False.
        return False

    def get_non_blocking(
        self,
        key: CacheEngineKey,
        location: Optional[str] = None,
    ) -> Optional[Future]:
        # No async get path yet. Returning None lets StorageManager fall
        # through to the blocking get; a bare Future() here would never be
        # resolved and leave waiters hanging.
        return None

    def _load_bytes_from_storage_with_memory(
        self,
        key: CacheEngineKey,
        path: str,
        memory_obj: Optional[MemoryObj],
    ) -> Optional[MemoryObj]:
        """
        Load byte array from storage into a pre-allocated memory object.

        Args:
            key: Cache key for error handling
            path: File path to load from
            memory_obj: Pre-allocated memory object to load data into

        Returns:
            The memory object with loaded data, or None if loading failed
        """
        if memory_obj is None or memory_obj.tensor is None:
            return None
        if self.config.xds_buffer_size is not None:
            assert memory_obj.tensor.is_cuda
            assert _devices_match(self.dst_device, memory_obj.tensor.device)

        with self.hot_lock:
            entry = self.hot_cache.get(key)

        if entry is None:
            logger.error(f"Metadata not found for {key}")
            return None

        if entry.disk_metadata is not None:
            assert memory_obj.tensor is not None
            size = entry.disk_metadata.io_nbytes
            try:
                ret = self._load_block_first_xds(
                    path,
                    entry.disk_metadata,
                    memory_obj.tensor,
                )
            except (BlockFirstFormatError, OSError) as e:
                logger.error("Invalid block-first entry %s: %s", path, e)
                ret = -1
        elif self.config.xds_buffer_size is not None:
            # XDS direct path stores the 4K-aligned physical size on disk.
            size = entry.size
            ret = self._load_xds(
                path,
                file_offset=0,
                gpu_pointer=ctypes.c_void_p(self.base_pointer),
                size=size,
                dev_offset=memory_obj.metadata.address
            )
        else:
            # POSIX stores the raw payload; checking against the aligned
            # entry.size would misclassify every non-4K-multiple entry as a
            # short read and evict it.
            assert entry.shape is not None and entry.dtype is not None
            size = entry.shape.numel() * entry.dtype.itemsize
            buffer = memory_obj.byte_array
            ret = self._load_posix(buffer[:size], path)

        if ret != size:
            logger.error(f"Load failed for {key}: {ret}/{size}")
            with self.hot_lock:
                if (meta := self.hot_cache.pop(key, None)):
                    size = meta.size
                    self.current_cache_size -= size
            memory_obj.ref_count_down()
            return None
        return memory_obj

    def _load_bytes_from_storage_with_allocation(
        self,
        key: CacheEngineKey,
        path: str,
        dtype: torch.dtype,
        shape: torch.Size,
    ) -> Optional[MemoryObj]:
        """
        Load byte array from storage by first allocating memory, then loading.

        Args:
            key: Cache key for error handling
            path: File path to load from
            dtype: Data type for memory allocation
            shape: Shape for memory allocation

        Returns:
            A new memory object with loaded data, or None if allocation or
            loading failed
        """
        memory_obj = self.memory_allocator.allocate(shape, dtype)
        if memory_obj is None:
            logger.debug("Memory allocation failed during sync storage load.")
            return None
        assert memory_obj.tensor is not None
        if self.config.xds_buffer_size is not None:
            # GPU residency only holds for the XDS direct buffer; the posix
            # branch allocates from the (CPU) LocalCPU backend.
            assert memory_obj.tensor.is_cuda
            assert _devices_match(self.dst_device, memory_obj.tensor.device)

        return self._load_bytes_from_storage_with_memory(key, path, memory_obj)

    def batched_get_blocking(
        self,
        keys: List[CacheEngineKey],
    ) -> list[MemoryObj | None]:
        if self.memory_allocator is None:  # MLA non-first-rank: inert
            return [None] * len(keys)
        paths: list[str | None] = []
        dtypes: list[torch.dtype | None] = []
        shapes: list[torch.Size | None] = []
        with self.hot_lock:
            for key in keys:
                entry = self.hot_cache.get(key)
                if entry is None:
                    logger.error(f"Lookup failed during get_blocking for {key}")
                    paths.append(None)
                    dtypes.append(None)
                    shapes.append(None)
                    continue
                self.cache_policy.update_on_hit(key, self.hot_cache)
                paths.append(entry.path)
                dtypes.append(entry.dtype)
                shapes.append(entry.shape)

        memory_objs: list[MemoryObj | None] = []
        xds_reads, xds_read_bytes = 0, 0
        for dtype, shape, path in zip(dtypes, shapes, paths, strict=True):
            if path is None:
                memory_objs.append(None)
                continue
            memory_obj = self.memory_allocator.allocate(shape, dtype, fmt = self.fmt)
            if memory_obj is None:
                logger.error(f"Memory allocation failed during get_blocking for {path}")
            else:
                xds_reads += 1
                xds_read_bytes += memory_obj.get_size()
            memory_objs.append(memory_obj)

        start_time = time.perf_counter()
        results = list(
            self._thread_pool.map(
                self._load_bytes_from_storage_with_memory, keys, paths, memory_objs
            )
        )
        total_time = time.perf_counter() - start_time
        return results

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _save_block_first_xds(
        self, path: str, logical_chunk: torch.Tensor
    ) -> BlockFirstDiskMetadata:
        """Atomically publish one header plus registered block-first payload."""
        disk_metadata = self._make_block_first_metadata(logical_chunk)
        header = pack_block_first_header(disk_metadata)
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
                "block-first staging shape does not match the complete chunk: "
                f"{tuple(staging_obj.tensor.shape)} != {disk_metadata.payload_shape}"
            )

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
                    self._profile_cuda_synchronize(staging_obj.tensor.device)
                with self._profile_phase("header"):
                    with open(tmp_path, "wb") as f:
                        written = f.write(header)
                if written != BLOCK_FIRST_HEADER_SIZE:
                    raise RuntimeError(
                        "block-first header write incomplete: "
                        f"{written}/{BLOCK_FIRST_HEADER_SIZE}"
                    )
                ret = self._profiled_xds_write(
                    tmp_path,
                    staging_allocator.base_pointer,
                    disk_metadata.io_nbytes,
                    BLOCK_FIRST_HEADER_SIZE,
                    staging_obj.metadata.address,
                )
                if ret != disk_metadata.io_nbytes:
                    raise RuntimeError(
                        "block-first XDS write incomplete: "
                        f"{ret}/{disk_metadata.io_nbytes}"
                    )
                expected_size = BLOCK_FIRST_HEADER_SIZE + disk_metadata.io_nbytes
                with self._profile_phase("stat"):
                    actual_size = os.path.getsize(tmp_path)
                if actual_size != expected_size:
                    raise RuntimeError(
                        f"block-first file size {actual_size} != {expected_size}"
                    )
                with self._profile_phase("rename"):
                    os.replace(tmp_path, path)
        except BaseException:
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            raise
        return disk_metadata

    @torch.inference_mode()
    def _load_block_first_xds(
        self,
        path: str,
        disk_metadata: BlockFirstDiskMetadata,
        logical_output: torch.Tensor,
    ) -> int:
        """Read into scratch and publish to logical output only after full I/O."""
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
            # HyFile DMA is outside PyTorch's stream ordering. Finish the
            # asynchronous zero kernel before the read, otherwise it can race
            # with and overwrite the payload that HyFile just placed here.
            if staging_obj.tensor.is_cuda:
                torch.cuda.synchronize(staging_obj.tensor.device)
            ret = self._load_xds(
                path,
                file_offset=BLOCK_FIRST_HEADER_SIZE,
                gpu_pointer=ctypes.c_void_p(staging_allocator.base_pointer),
                size=disk_metadata.io_nbytes,
                dev_offset=staging_obj.metadata.address,
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
    def _save_xds(
        self,
        path: str,
        kv_chunk: torch.Tensor,
        size : int,
        base_pointer: int,
        device_offset: int
    ) -> bytes:
        """ XDS 写入 """
        # 计算地址
        if base_pointer is None:
            gpu_addr = kv_chunk.data_ptr()
            dev_offset = 0
        else:
            gpu_addr = base_pointer
            dev_offset = device_offset

        tmp_path = path.replace(_DATA_FILE_SUFFIX, _DATA_FILE_SUFFIX_TMP)

        metadata = pack_metadata(
            kv_chunk.shape,
            kv_chunk.dtype,
            size
        )

        try:
            open(tmp_path, "wb").close()
            ret = self._profiled_xds_write(
                tmp_path,
                gpu_addr,
                size,
                0,  # file_offset
                dev_offset
            )

            if ret < 0:
                raise RuntimeError(f"XDS write failed with error {ret}")
            if ret != size:
                raise RuntimeError(
                    f"XDS write incomplete: {ret}/{kv_chunk.nbytes} bytes"
                )
            # 原子性重命名
            with self._profile_phase("rename"):
                os.rename(tmp_path, path)

        except Exception as e:
            logger.error(f"Error saving {tmp_path}: {e}", exc_info=True)
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise e

        return metadata

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _save_posix(self, path:str, kv_chunk:torch.Tensor, size:int, buffer):
        """ posix协议写入 """
        return self.xds.write(path, kv_chunk, size, buffer)

    def _save_cpu_xds(
        self,
        path: str,
        memory_obj: MemoryObj,
    ) -> bytes:
        """Write a legacy XDS payload directly from a CPU MemoryObj.

        The XDS reader expects the legacy payload size to be 4 KiB aligned.
        LocalCPU MemoryObj exposes only the logical bytes, so the writer emits
        explicit zero padding instead of allocating a GPU staging object or
        reading beyond the logical tensor.
        """
        tensor = memory_obj.tensor
        if tensor is None or tensor.device.type != "cpu":
            raise ValueError("xds_cpu_store requires a CPU MemoryObj")

        raw_size = memory_obj.get_size()
        physical_size = align_up(raw_size)
        buffer = memory_obj.byte_array
        if len(buffer) != raw_size:
            raise RuntimeError(
                f"CPU MemoryObj byte size {len(buffer)} != logical size {raw_size}"
            )
        metadata = pack_metadata(tensor.shape, tensor.dtype, physical_size)

        tmp_path = path.replace(_DATA_FILE_SUFFIX, _DATA_FILE_SUFFIX_TMP)
        fd = None
        try:
            fd = os.open(tmp_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            written = 0
            while written < raw_size:
                count = os.write(fd, buffer[written:raw_size])
                if count <= 0:
                    raise OSError(
                        f"CPU XDS write made no progress at {written}/{raw_size}"
                    )
                written += count

            padding = physical_size - raw_size
            if padding:
                zero_padding = b"\x00" * padding
                padded = 0
                while padded < padding:
                    count = os.write(fd, zero_padding[padded:])
                    if count <= 0:
                        raise OSError(
                            "CPU XDS padding write made no progress at "
                            f"{padded}/{padding}"
                        )
                    padded += count

            actual_size = os.fstat(fd).st_size
            if actual_size != physical_size:
                raise OSError(
                    f"CPU XDS file size {actual_size} != {physical_size} bytes"
                )
            os.close(fd)
            fd = None
            os.replace(tmp_path, path)
        except BaseException:
            if fd is not None:
                os.close(fd)
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            raise

        return metadata

    @torch.inference_mode()
    def _save_cpu_block_first_xds(
        self,
        path: str,
        memory_obj: MemoryObj,
    ) -> BlockFirstDiskMetadata:
        """Publish a block-first file directly from the existing CPU MemoryObj.

        The CPU source stays in LMCache's logical ``[2, NL, T, D]`` format,
        while direct retrieve addresses vLLM's final allocation as
        ``[CB, NL, 2, BS, D]``. Write each already-contiguous CPU block slice
        at its final file position instead of building a second CPU buffer or
        an XDS GPU staging allocation.
        """
        tensor = memory_obj.tensor
        if tensor is None or tensor.device.type != "cpu":
            raise ValueError("CPU block-first store requires a CPU MemoryObj")
        if not tensor.is_contiguous():
            raise BlockFirstFormatError("CPU block-first source must be contiguous")

        disk_metadata = self._make_block_first_metadata(tensor)
        header = pack_block_first_header(disk_metadata)
        element_bytes = tensor.element_size()
        tmp_path = path + _BLOCK_FIRST_DATA_FILE_SUFFIX_TMP

        def write_all(fd: int, data: memoryview, label: str) -> None:
            written = 0
            while written < len(data):
                count = os.write(fd, data[written:])
                if count <= 0:
                    raise OSError(
                        f"CPU block-first {label} made no progress at "
                        f"{written}/{len(data)}"
                    )
                written += count

        fd = None
        try:
            fd = os.open(tmp_path, os.O_CREAT | os.O_WRONLY | os.O_TRUNC, 0o644)
            write_all(fd, memoryview(header), "header write")
            for block in range(disk_metadata.payload_shape[0]):
                token_start = block * disk_metadata.block_size
                token_end = token_start + disk_metadata.block_size
                for layer in range(disk_metadata.num_layers):
                    for kv_index in range(2):
                        source = tensor[kv_index, layer, token_start:token_end]
                        source_bytes = memoryview(
                            source.view(torch.uint8).numpy()
                        ).cast("B")
                        expected_nbytes = (
                            disk_metadata.block_size
                            * disk_metadata.hidden_dim
                            * element_bytes
                        )
                        if len(source_bytes) != expected_nbytes:
                            raise BlockFirstFormatError(
                                "CPU block-first source slice has unexpected size"
                            )
                        write_all(fd, source_bytes, "payload write")

            padding = disk_metadata.io_nbytes - disk_metadata.payload_nbytes
            if padding:
                write_all(fd, memoryview(bytes(padding)), "padding write")
            expected_size = BLOCK_FIRST_HEADER_SIZE + disk_metadata.io_nbytes
            actual_size = os.fstat(fd).st_size
            if actual_size != expected_size:
                raise OSError(
                    f"CPU block-first file size {actual_size} != {expected_size}"
                )
            os.close(fd)
            fd = None
            os.replace(tmp_path, path)
        except BaseException:
            if fd is not None:
                os.close(fd)
            try:
                os.remove(tmp_path)
            except FileNotFoundError:
                pass
            raise
        return disk_metadata

    def _load_xds(self, file_path: str, file_offset: int,
                  gpu_pointer: ctypes.c_void_p, size: int,
                  dev_offset: int) -> int:
        """ xds协议读取 """
        gpu_addr = gpu_pointer.value if gpu_pointer.value else 0

        ret = self.xds.read(
            file_path,
            gpu_addr,
            size,      # 实际大小
            file_offset,
            dev_offset,
        )

        if ret < 0:
            logger.error(f"XDS read failed: {file_path}, error={ret}")
        elif ret != size:
            logger.warning(f"XDS read incomplete: {file_path}, {ret}/{size} bytes")
        return ret

    def _load_posix(self, buffer, path):
        """ 读取posix """
        return self.xds.read(buffer, path)

    def touch_cache(self):
        with self.hot_lock:
            for key in reversed(self.keys_in_request):
                self.cache_policy.update_on_hit(key, self.hot_cache)
            self.keys_in_request = []

    def pin(self, key: CacheEngineKey) -> bool:
        with self.hot_lock:
            if key in self.hot_cache:
                self.hot_cache[key].pin()
                return True
            else:
                return False

    def unpin(self, key: CacheEngineKey) -> bool:
        with self.hot_lock:
            if key in self.hot_cache:
                self.hot_cache[key].unpin()
                return True
            else:
                return False

    def remove(self, key, force=True):

        with self.hot_lock:
            if not (meta := self.hot_cache.pop(key, None)):
                return False
            self.cache_policy.update_on_force_evict(key)
            # Keep the accounting consistent: external callers (StorageManager
            # batched_remove / cache_engine) reach here directly, and the old
            # code popped the entry without crediting current_cache_size back,
            # leaking `meta.size` per removed key. (_eviction no longer routes
            # through remove(), so this cannot double-subtract.)
            self.current_cache_size -= meta.size

        data_path = meta.path

        metadata_path = data_path + _METADATA_FILE_SUFFIX

        # Remove each file independently: a single try covering both meant a
        # missing/failed metadata unlink skipped the data unlink, orphaning the
        # `.kvdata` blob on the (possibly remote/shared) disk.
        for p in (metadata_path, data_path):
            try:
                os.remove(p)
            except FileNotFoundError:
                pass
            except OSError as e:
                logger.error(f"Error removing {p} for {key}: {e}")

        return True

    def close(self) -> None:
        if self._closed:
            return
        # Drain all in-flight work BEFORE tearing down the driver: cleanup()
        # first would pull the driver out from under queued puts / running
        # reads still in the worker and thread pool.
        scan_future = getattr(self, "_scan_future", None)
        if scan_future is not None:
            try:
                scan_future.result()
            except Exception as e:
                logger.error(f"XDS metadata scan failed while closing: {e}")
        self.xds_worker.close()  # drains queued put tasks (executor waits)
        self._thread_pool.shutdown(wait=True)  # drains in-flight reads
        self._close_direct_registrations()
        self._close_direct_edge_staging()
        self._close_block_first_staging()
        self.xds.cleanup()
        # posix mode borrows the LocalCPU backend's allocator — only close
        # allocators we created ourselves (None on MLA non-first-rank).
        if (
            self.config.xds_buffer_size is not None
            and self.memory_allocator is not None
        ):
            self.memory_allocator.close()
        if self._direct_driver_owned:
            HIPFile.close_driver()
            self._direct_driver_owned = False
        self._closed = True
        logger.info("Xds backend closed.")

    # def __del__(self):
    #     HIPFile.get_global_stats()

    def initialize_allocator(
        self, config: LMCacheEngineConfig, metadata: LMCacheMetadata, backend: str
    ) -> HyFileMemoryAllocator:
        assert config.xds_path is not None
        assert config.xds_buffer_size is not None
        logger.info(f"use_mla:{metadata.use_mla};is_first_rank:{metadata.is_first_rank()}")
        if metadata.use_mla and not metadata.is_first_rank():
            # MLA shares one latent KV across all TP ranks (world_size==1), so
            # only the first rank offloads. Non-first ranks get no allocator and
            # the whole backend stays inert (store/get/allocate/close are guarded
            # no-ops) instead of crashing later on a None allocator / unset
            # base_pointer.
            logger.info(
                "XDS offload disabled on this rank (MLA non-first-rank): "
                "store/get/allocate are no-ops."
            )
            return None
        if backend == "hyfile":
            return HyFileMemoryAllocator(size = config.xds_buffer_size * 1024 ** 2)
        else:
            return CuFileMemoryAllocator(size = config.xds_buffer_size * 1024 ** 2)

    def get_allocator_backend(self):
        if self.cpu_store_enabled:
            assert self.local_cpu_backend is not None
            return self.local_cpu_backend
        return self

    def supports_direct_io(self, direction: DirectIODirection) -> bool:
        """Expose the asymmetric hybrid contract to StorageManager."""
        if direction == DirectIODirection.STORE:
            return bool(getattr(self, "direct_store_enabled", self.direct_enabled))
        if direction == DirectIODirection.RETRIEVE:
            return bool(
                getattr(self, "direct_retrieve_enabled", self.direct_enabled)
            )
        return False

    def get_memory_allocator(self):
        return self.memory_allocator

    def allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[MemoryObj]:
        if busy_loop:
            logger.warning("Xds Backend does not support allocation with busy loop")

        if self.memory_allocator is None:  # MLA non-first-rank: inert
            return None
        return self.memory_allocator.allocate(shape, dtype, self.fmt)

    def batched_allocate(
        self,
        shape: torch.Size,
        dtype: torch.dtype,
        batch_size: int,
        fmt: MemoryFormat = MemoryFormat.KV_2LTD,
        eviction: bool = True,
        busy_loop: bool = True,
    ) -> Optional[list[MemoryObj]]:
        if busy_loop:
            logger.warning("Xds Backend does not support allocation with busy loop")

        if self.memory_allocator is None:  # MLA non-first-rank: inert
            return None
        return self.memory_allocator.batched_allocate(
            shape, dtype, batch_size, self.fmt
        )

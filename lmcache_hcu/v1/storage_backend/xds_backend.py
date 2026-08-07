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

import json
from collections import deque
from datetime import datetime
from concurrent.futures import Future, ThreadPoolExecutor
from typing import TYPE_CHECKING, Any, Callable, List, Optional, Sequence, Tuple
import asyncio
import ctypes
import os
import random
import string
import struct
import threading
import time
import mmap
import numpy as np

# Third Party
import torch
from abc import ABC, abstractmethod

# First Party
from lmcache.config import LMCacheEngineMetadata
from lmcache.logging import init_logger
from lmcache_hcu.v1.storage_backend.xds_metadata import XDSCacheMetadata
from lmcache.utils import CacheEngineKey, _lmcache_nvtx_annotate
from lmcache.v1.config import LMCacheEngineConfig
from lmcache.v1.memory_management import (
    CuFileMemoryAllocator,
    HyFileMemoryAllocator,
    MemoryFormat,
    MemoryObj,
)
from lmcache.v1.storage_backend.abstract_backend import AllocatorBackendInterface
from lmcache_hcu.observability import LMCacheStatsLogger, LMCStatsMonitor
from lmcache.v1.storage_backend.cache_policy import get_cache_policy
from lmcache.v1.storage_backend.job_executor.pq_executor import (
    AsyncPQThreadPoolExecutor,
)
from lmcache_hcu.v1.hipfille import HIPFile
from lmcache.v1.storage_backend.local_cpu_backend import LocalCPUBackend

logger = init_logger(__name__)

_METADATA_FILE_SUFFIX = ".metadata"
_DATA_FILE_SUFFIX = ".kvdata"
_METADATA_FILE_SUFFIX_TMP = ".METADATA"
_DATA_FILE_SUFFIX_TMP = ".KVDATA"
_METADATA_VERSION = 1

# GDS and Weka both use 4096, but this is padding; 72 is an empirical value here
_METADATA_MAX_SIZE = 72
_ALIGNMENT = 4096  # 4K alignment
_DIR_SIZE = 2
MEMCPY_HOST_TO_DEVICE = 1
MEMCPY_DEVICE_TO_HOST = 2


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
    version = struct.unpack_from("<Q", buffer)[0]

    if version == 1:
        version, dt_idx, size, ndim = struct.unpack_from("<QQQQ", buffer)
        shape_offset = struct.calcsize("<QQQQ")
        shape = struct.unpack_from("<" + ndim * "Q", buffer, offset=shape_offset)
        return torch.Size(shape), torch_dtypes[dt_idx], size
    else:
        raise UnsupportedMetadataVersion(f"Unsupported metadata version: {version}")


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
        """Initialize the XDS driver and return whether it succeeded"""
        pass

    @abstractmethod
    def write(self, file_path: str, gpu_address: int, size: int,
              file_offset: int, dev_offset: int) -> int:
        """
        Write data to file
        Return the actual number of bytes written, or a negative value on failure
        """
        pass

    @abstractmethod
    def read(self, file_path: str, gpu_address: int, size: int,
             file_offset: int, dev_offset: int) -> int:
        """
        Read data from file
        Return the actual number of bytes read, or a negative value on failure
        """
        pass

    def cleanup(self):
        pass


def _init_fallback_logger(obj) -> None:
    obj._fallback_log_lock = threading.Lock()
    obj._fallback_log_states: dict[tuple[str, str], dict[str, Any]] = {}
    obj._fallback_log_interval = 120.0
    obj._fallback_log_max_bytes = 1024 ** 4


def _format_fallback_bytes(obj, size: int) -> str:
    capped_size = min(size, obj._fallback_log_max_bytes)
    if capped_size >= 1024 ** 3:
        return f"{capped_size / 1024 ** 3:.2f}GiB ({capped_size} bytes)"
    if capped_size >= 1024 ** 2:
        return f"{capped_size / 1024 ** 2:.2f}MiB ({capped_size} bytes)"
    return f"{capped_size} bytes"


def _log_fallback(obj, backend: str, op: str, reason: str, size: int) -> None:
    now = time.monotonic()
    state_key = (backend, op)
    with obj._fallback_log_lock:
        state = obj._fallback_log_states.get(state_key)
        if state is None:
            obj._fallback_log_states[state_key] = {
                "start_time": now,
                "count": 1,
                "bytes": size,
            }
            logger.debug(
                "%s %s fallback to CPU copy: reason=%s, size=%s bytes",
                backend,
                op,
                reason,
                size,
            )
            return

        elapsed = now - state["start_time"]
        if elapsed < obj._fallback_log_interval:
            state["count"] += 1
            state["bytes"] += size
            logger.debug(
                "%s %s fallback to CPU copy: reason=%s, size=%s bytes",
                backend,
                op,
                reason,
                size,
            )
            return

        logger.warning(
            "%s %s fallback summary: count=%s, bytes=%s, "
            "interval=%.1fs, last_reason=%s",
            backend,
            op,
            state["count"],
            _format_fallback_bytes(obj, state["bytes"]),
            elapsed,
            reason,
        )
        obj._fallback_log_states[state_key] = {
            "start_time": now,
            "count": 1,
            "bytes": size,
        }


class HyFileDDS(CliStorageInterface):
    """Hygon HCU HyFile DDS implementation"""

    def __init__(self):
        _init_fallback_logger(self)

    def initialize(self) -> bool:
        try:
            self.hiprt = ctypes.CDLL("libhiprtc.so")
            logger.info("HyFile DDS initialized")
            return True
        except Exception as e:
            logger.warning(f"HIP Runtime Error")
            self.hiprt = None
            return True

    def write(self, file_path: str, gpu_address: int, size: int,
              file_offset: int, dev_offset: int) -> int:
        # Try writing through XDS
        try:
            with HIPFile(file_path, "w", skip_buffer_registration=True) as f:
                ret = f.write(gpu_address, size, file_offset, dev_offset)
            if ret == size:
                logger.debug(f"HyFile write: {size} bytes")
                return size
            _log_fallback(self,
                "hyFile",
                "write",
                f"write incomplete ({ret}/{size} bytes)",
                size,
            )
        except Exception as e:
            _log_fallback(self, "hyFile", "write", f"write failed: {e}", size)

        # Fallback to POSIX write after XDS write failure
        if self.hiprt is None:
            logger.error("HIP runtime not available, cannot fallback")
            return -1

        fd = None
        mm = None
        arr = None

        # Try writing through HIP Runtime plus mmap
        try:
            fd = os.open(file_path, os.O_RDWR)
            os.ftruncate(fd, file_offset + size)
            # mmap the file
            mm = mmap.mmap(
                fd, file_offset + size,
                prot=mmap.PROT_WRITE,
                flags=mmap.MAP_SHARED
            )
            os.close(fd)

            # Get the CPU memory address
            arr = np.frombuffer(mm, dtype=np.uint8)
            cpu_addr = arr.__array_interface__["data"][0]

            # Use hipMemcpy to copy from GPU to CPU
            res = self.hiprt.hipMemcpy(
                ctypes.c_void_p(cpu_addr + file_offset),  # dst: CPU memory
                ctypes.c_void_p(gpu_address + dev_offset),  # src: GPU memory
                ctypes.c_size_t(size),  # size
                ctypes.c_int(MEMCPY_DEVICE_TO_HOST),  # direction: D2H
            )

            if res != 0:
                logger.error(f"Memcpy failed with code {res}")
                return -1

            logger.debug(f"CPU fallback write successful: {size} bytes")
            return size
        except Exception as e:
            logger.error(f"CPU fallback write failed: {e}", exc_info=True)
            return -1

        finally:
            if arr is not None:
                del arr
            if mm is not None:
                mm.close()

    def read(self, file_path: str, gpu_address: int, size: int,
             file_offset: int, dev_offset: int) -> int:

        if not os.path.exists(file_path):
            logger.error(f"No such file or directory: {file_path}, read failed")
            return -1

        # Try reading through XDS
        try:
            with HIPFile(file_path, "r", skip_buffer_registration=True) as f:
                ret = f.read(gpu_address, size, file_offset, dev_offset)
                if ret == size:
                    return ret
            _log_fallback(self,
                "hyFile",
                "read",
                f"read incomplete ({ret}/{size} bytes)",
                size,
            )
        except Exception as e:
            _log_fallback(self, "hyFile", "read", f"read failed: {e}", size)

        # Fallback to POSIX write after XDS read failure
        if self.hiprt is None:
            logger.error("HIP runtime not available, cannot fallback")
            return -1

        fd = None
        mm = None
        arr = None

        # Try reading through HIP Runtime plus mmap
        try:
            # Open the file and mmap it
            fd = os.open(file_path, os.O_RDONLY)
            file_size = os.fstat(fd).st_size
            mm = mmap.mmap(
                fd, file_size,
                prot=mmap.PROT_READ,
                flags=mmap.MAP_PRIVATE | mmap.MAP_POPULATE,
            )
            os.close(fd)

            # Get the CPU memory address
            arr = np.frombuffer(mm, dtype=np.uint8)
            cpu_addr = arr.__array_interface__["data"][0]

            # Use hipMemcpy to copy to GPU
            res = self.hiprt.hipMemcpy(
                ctypes.c_void_p(gpu_address + dev_offset),
                ctypes.c_void_p(cpu_addr + file_offset),
                ctypes.c_size_t(size),
                ctypes.c_int(MEMCPY_HOST_TO_DEVICE),
            )

            if res != 0:
                logger.error(f"hipMemcpy failed with code {res}")
                return -1

            logger.debug(f"CPU fallback successful: {size} bytes")
            return size

        except Exception as e:
            logger.error(f"CPU fallback failed: {e}", exc_info=True)
            return -1

        finally:
            if arr is not None:
                del arr
            if mm != None:
                mm.close()


class Posix(CliStorageInterface):

    def __init__(self, root_path):
        self.use_odirect = True

        # Block size (for file system I/O)
        stat = os.statvfs(root_path)
        self.os_disk_bs = stat.f_bsize

    def initialize(self) -> bool:
        return True

    def write(self, path: str, kv_chunk: torch.Tensor, size: int, buffer):
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
                    f.write(buffer)
            else:
                fd = os.open(tmp_path, os.O_CREAT | os.O_WRONLY | os.O_DIRECT, 0o644)
                os.write(fd, buffer)
                os.close(fd)

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
                    f.readinto(buffer)
            else:
                fd = os.open(path, os.O_RDONLY | os.O_DIRECT)
                with os.fdopen(fd, "rb", buffering=0) as fdo:
                    fdo.readinto(buffer)
        except FileNotFoundError:
            logger.warning(f"File not found on clistorage: {path}")
            return -1

        disk_read_time = time.time() - start_time
        logger.debug(
            f"clistorage read size: {size} bytes, "
            f"Bandwidth: {size / disk_read_time / 1e6:.2f} MB/s"
        )
        return size


class CuFileGDS(CliStorageInterface):
    """NVIDIA cuFile GDS implementation"""

    def __init__(self):
        _init_fallback_logger(self)
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
            # Try direct cuFile write
            with self.cufile.CuFile(file_path, "r+", use_direct_io=True) as f:
                ret = f.write(
                    ctypes.c_void_p(gpu_address), size,
                    file_offset=file_offset, dev_offset=dev_offset
                )

            # Return directly if the write is complete
            if ret == size:
                return ret

            # Try fallback if the write is incomplete
            _log_fallback(self,
                "cuFile",
                "write",
                f"write incomplete ({ret}/{size} bytes)",
                size,
            )

        except Exception as e:
            _log_fallback(self, "cuFile", "write", f"write failed: {e}", size)

        # ========== Fallback to CPU copy ==========
        if self.cudart is None:
            logger.error("CUDA runtime not available, cannot fallback")
            return -1

        fd = None
        mm = None
        arr = None
        try:

            fd = os.open(file_path, os.O_RDWR)
            os.ftruncate(fd, file_offset + size)
            # mmap the file
            mm = mmap.mmap(
                fd, file_offset + size,
                prot=mmap.PROT_WRITE,
                flags=mmap.MAP_SHARED
            )
            os.close(fd)
            # Get the CPU memory address
            arr = np.frombuffer(mm, dtype=np.uint8)
            cpu_addr = arr.__array_interface__["data"][0]

            # Use cudaMemcpy to copy from GPU to CPU
            res = self.cudart.cudaMemcpy(
                ctypes.c_void_p(cpu_addr + file_offset),  # dst: CPU memory
                ctypes.c_void_p(gpu_address + dev_offset),  # src: GPU memory
                ctypes.c_size_t(size),  # size
                ctypes.c_int(MEMCPY_DEVICE_TO_HOST),  # direction: D2H
            )

            if res != 0:
                logger.error(f"cudaMemcpy failed with code {res}")
                return -1
            # Ensure data is written to disk
            logger.debug(f"CPU fallback write successful: {size} bytes")
            return size

        except Exception as e:
            logger.error(f"CPU fallback write failed: {e}", exc_info=True)
            return -1
        finally:
            if arr is not None:
                del arr
            if mm != None:
                mm.close()

    def read(self, file_path: str, gpu_address: int, size: int,
             file_offset: int, dev_offset: int) -> int:

        if not os.path.exists(file_path):
            logger.error(f"No such file or directory: {file_path}, read failed")
            return -1

        try:
            with self.cufile.CuFile(file_path, "r", use_direct_io=True) as f:
                ret = f.read(
                    ctypes.c_void_p(gpu_address), size,
                    file_offset=file_offset, dev_offset=dev_offset
                )

            if ret == size:
                return ret

            # Try fallback if the read is incomplete
            _log_fallback(self,
                "cuFile",
                "read",
                f"read incomplete ({ret}/{size} bytes)",
                size,
            )

        except Exception as e:
            _log_fallback(self, "cuFile", "read", f"read failed: {e}", size)

        if self.cudart is None:
            logger.error("CUDA runtime not available, cannot fallback")
            return -1
        fd = None
        mm = None
        arr = None
        try:
            # Open the file and mmap it
            fd = os.open(file_path, os.O_RDONLY)

            file_size = os.fstat(fd).st_size
            mm = mmap.mmap(
                fd, file_size,
                prot=mmap.PROT_READ,
                flags=mmap.MAP_PRIVATE | mmap.MAP_POPULATE,
            )
            os.close(fd)
            # Get the CPU memory address
            arr = np.frombuffer(mm, dtype=np.uint8)
            cpu_addr = arr.__array_interface__["data"][0]

            # Use cudaMemcpy to copy to GPU
            res = self.cudart.cudaMemcpy(
                ctypes.c_void_p(gpu_address + dev_offset),
                ctypes.c_void_p(cpu_addr + file_offset),
                ctypes.c_size_t(size),
                ctypes.c_int(MEMCPY_HOST_TO_DEVICE),
            )

            if res != 0:
                logger.error(f"cudaMemcpy failed with code {res}")
                return -1
            logger.debug(f"CPU fallback successful: {size} bytes")
            return size

        except Exception as e:
            logger.error(f"CPU fallback failed: {e}", exc_info=True)
            return -1
        finally:
            if arr is not None:
                del arr
            if mm != None:
                mm.close()


def check_environment():
    if torch.version.cuda is None:
        backend = "hyfile"
    else:
        backend = "cufile"

    logger.info(f"Detected XDS interface: {backend}")

    return backend


def create_xds_interface(backend: str):
    # Create instance
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
            metadata: LMCacheEngineMetadata,
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
        self.loop = loop
        if self.config.xds_buffer_size is not None:
            # Use the XDS protocol
            self.xds_interface = check_environment()
            self.memory_allocator = self.initialize_allocator(config, metadata, self.xds_interface)
            self.xds = create_xds_interface(self.xds_interface)

            if self.memory_allocator is not None:
                assert hasattr(self.memory_allocator, "base_pointer")
                self.base_pointer = self.memory_allocator.base_pointer

        else:
            # Use the POSIX protocol
            assert local_cpu_backend is not None
            self.memory_allocator = local_cpu_backend
            self.xds = Posix(config.xds_path)
        logger.debug(f"gfy: memory_allocator is {self.memory_allocator}")

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

        self._max_workers = thread_count

        self.xds_worker = XdsWorker(self.loop)

        self.world_id = metadata.worker_id

        if metadata.use_mla:
            self.world_szie = 1
        else:
            self.world_szie = metadata.world_size

        if config.max_xds_size is not None:
            self.eviction = True
            self.max_xds_size = int(config.max_xds_size * 1024 ** 3)
            logger.info(f"Peer rank max_storage_size: {config.max_xds_size}GB, "
                        f"total max_storage_size: {config.max_xds_size * self.world_szie}GB")
        else:
            self.eviction = False
            logger.info(f"config.max_xds_size is None, don't suppot eviction")

        # flag for extra assertions to catch bugs but harm performance
        self._debug_asserts = False
        # flag to use O_NOATIME during metadata file read for performance improvement
        self._use_noatime = True

        self.current_cache_size = 0.0
        self.keys_in_request: List[CacheEngineKey] = []

        self.stats_monitor = LMCStatsMonitor.GetOrCreate()

        # Thread monitoring file
        self._monitoring_data = deque(maxlen=100000)
        self._monitor_file = f"thread_pool_monitor_HCU{torch.cuda.current_device()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.jsonl"

        # Start monitor thread
        self._start_thread_pool_monitor()

        asyncio.run_coroutine_threadsafe(self._scan_metadata(), self.loop)

    def _start_thread_pool_monitor(self):
        """Start a background thread to monitor thread pool status"""

        def monitor_loop():
            while True:
                try:
                    # Get thread pool status
                    queue_size = self._thread_pool._work_queue.qsize()

                    # Estimate the number of active threads
                    # active_threads ≈ max_workers - idle_threads
                    # Simplification: if tasks are queued, assume all threads are working
                    active_threads = min(self._max_workers,
                                         self._max_workers if queue_size > 0
                                         else self._max_workers)

                    # self._monitoring_data.append({
                    # 'timestamp': time.time(),
                    # 'datetime': datetime.now().isoformat(),
                    # 'queue_size': queue_size,
                    # 'active_threads': active_threads})

                    # Update monitoring data
                    self.stats_monitor.update_thread_pool_status(
                        active_threads=active_threads,
                        queue_size=queue_size
                    )

                    time.sleep(0.01)  # Update every 10 ms
                except Exception as e:
                    logger.error(f"Thread pool monitoring error: {e}")

            # Persistence thread

        def persist_loop():
            while True:
                try:
                    time.sleep(10)  # Write once every 10 seconds

                    if self._monitoring_data:
                        # Copy data
                        data_to_write = list(self._monitoring_data)
                        self._monitoring_data.clear()
                        # Write to file
                        with open(self._monitor_file, 'a', encoding='utf-8') as f:
                            for record in data_to_write:
                                f.write(json.dumps(record) + '\n')

                        # logger.info(f"Wrote {len(data_to_write)} records")

                except Exception as e:
                    logger.error(f"Persist error: {e}")

        monitor_thread = threading.Thread(target=monitor_loop, daemon=True)
        monitor_thread.start()

        # persist_thread = threading.Thread(target=persist_loop, daemon=True)
        # persist_thread.start()

    async def _scan_metadata(self):
        """Scan metadata in batches with progress feedback"""
        BATCH_SIZE = 11  # Number of directories processed per batch

        start = time.perf_counter()

        # Step 1: collect all directories to scan
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

        # Step 2: process in batches
        for i in range(0, total_dirs, BATCH_SIZE):
            batch = dirs_to_scan[i:i + BATCH_SIZE]

            # Create tasks for the current batch
            tasks = [
                asyncio.to_thread(
                    self._scan_metadata_subdir,
                    item['path'],
                    item['l1_dir']
                )
                for item in batch
            ]

            # Wait for the current batch to finish
            results = await asyncio.gather(*tasks, return_exceptions=True)

            # Count errors
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
        #     self.current_cache_size /= self.world_szie
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
                        if not fentry.name.endswith(target_suffix):
                            if not fentry.name.endswith(_DATA_FILE_SUFFIX):
                                os.remove(fentry.path)
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
                        if key.worker_id is not self.world_id:
                            continue
                        try:
                            self._read_metadata(key, fentry.path, l1_dir + l2_dir)
                        except UnsupportedMetadataVersion:
                            logger.error(
                                "Unsupported metadata version for "
                                f"{fentry.path}, ignoring"
                            )

    # Modification date: 2026-03-27
    # Modified by: Guo Fengya
    # Modification reason: merge upstream GDS performance improvement code https://github.com/LMCache/LMCache/pull/2637/changes
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

    def _read_metadata(self, key, filename, subdir_key):

        shape, dtype, size = self._read_metadata_info(filename)

        metadata = XDSCacheMetadata(
            filename.removesuffix(_METADATA_FILE_SUFFIX),
            size,  # Original size
            shape,
            dtype
        )

        with self.hot_lock:
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
    ) -> Tuple[str, str, str, str]:
        hash_str = str(key.chunk_hash)
        l1_dir = hash_str[:_DIR_SIZE]
        l2_dir = hash_str[_DIR_SIZE:_DIR_SIZE * 2]
        key_str = key.to_string()
        assert "_" not in key_str, "key string should not contain `_`"
        path = os.path.join(
            self.xds_path,
            l1_dir,
            l2_dir,
            key_str.replace("/", "_") + _DATA_FILE_SUFFIX
        )
        return (path, l1_dir + l2_dir, l1_dir, l2_dir)

    def exists_in_put_tasks(self, key: CacheEngineKey) -> bool:
        return self.xds_worker.exists_in_put_tasks(key)

    def submit_put_task(self, key: CacheEngineKey, memory_obj: MemoryObj) -> Future:
        assert memory_obj.tensor is not None

        if self.exists_in_put_tasks(key):
            logger.debug(f"Put task for {key} is already in progress.")
            return None

        self.xds_worker.insert_put_task(key)

        required_size = memory_obj.get_physical_size() + _METADATA_MAX_SIZE
        evict_success = True
        if self.eviction:
            evict_success = self._eviction(required_size)
        else:
            self.current_cache_size += required_size

        if not evict_success:
            self.xds_worker.remove_put_task(key)
            return None

        self.cache_policy.update_on_put(key)
        memory_obj.ref_count_up()

        asyncio.run_coroutine_threadsafe(
            self.xds_worker.submit_task(
                "put",
                self._async_save_bytes_to_storage,
                key=key,
                memory_obj=memory_obj,
            ),
            self.loop,
        )

    def _eviction(self, required_size: int) -> bool:
        # logger.info(f"_eviction current_cache_size:{self.current_cache_size} required_size:{required_size}")
        all_evict_keys = []
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
                    self.current_cache_size -= self.hot_cache[evict_key].size
                all_evict_keys.extend(evict_keys)

            if evict_success:
                self.current_cache_size += required_size
        # logger.info(f"evict_success {evict_success}, all_evict_keys{all_evict_keys}")
        self.batched_remove(all_evict_keys, force=False)

        return evict_success

    def batched_submit_put_task(
            self,
            keys: Sequence[CacheEngineKey],
            memory_objs: List[MemoryObj],
            transfer_spec=None,
    ) -> None:
        for key, memory_obj in zip(keys, memory_objs, strict=False):
            self.submit_put_task(key, memory_obj)

    def _async_save_bytes_to_storage(
            self,
            key: CacheEngineKey,
            memory_obj: MemoryObj,
    ) -> None:
        """
        Convert KV to bytes and async store bytes to storage.
        """
        # logger.info(f"_async_save_bytes_to_storage {key}")
        kv_chunk = memory_obj.tensor
        assert kv_chunk is not None

        size = memory_obj.get_physical_size()
        path, _, l1_dir, l2_dir = self._key_to_path(key)

        os.makedirs(os.path.join(self.xds_path, l1_dir, l2_dir), exist_ok=True)

        try:
            if self.config.xds_buffer_size is not None:
                # XDS write
                metadata = self._save_xds(
                    path,
                    kv_chunk,
                    size,
                    self.base_pointer,
                    memory_obj.metadata.address)
            else:
                # POSIX write
                buffer = memory_obj.byte_array
                metadata = self._save_posix(
                    path,
                    kv_chunk,
                    size,
                    buffer)

            self.insert_key(key, memory_obj)

            save_metadata(path, _METADATA_FILE_SUFFIX_TMP, metadata)

        except Exception as e:
            logger.error(f"Failed to save {key}: {e}", exc_info=True)
            cleanup_paths = [
                path.replace(_DATA_FILE_SUFFIX, _DATA_FILE_SUFFIX_TMP),  # Temporary DATA path
                path,  # Final DATA path
                path.replace(_DATA_FILE_SUFFIX, _DATA_FILE_SUFFIX_TMP) + _METADATA_FILE_SUFFIX_TMP,  # Temporary metadata path
                path + _METADATA_FILE_SUFFIX  # Final metadata path
            ]
            for cleanup_path in cleanup_paths:
                if os.path.exists(cleanup_path):
                    try:
                        os.remove(cleanup_path)
                    except OSError:
                        pass
            with self.hot_lock:
                self.hot_cache.pop(key, None)
                self.current_cache_size -= memory_obj.get_physical_size() + _METADATA_MAX_SIZE
        finally:
            memory_obj.ref_count_down()
            self.xds_worker.remove_put_task(key)

    def insert_key(self, key: CacheEngineKey, memory_obj: MemoryObj) -> None:
        with self.hot_lock:
            if key in self.hot_cache:
                self.cache_policy.update_on_hit(key, self.hot_cache)
            else:
                path, _, _, _ = self._key_to_path(key)
                size = memory_obj.get_physical_size()
                shape = memory_obj.metadata.shape
                dtype = memory_obj.metadata.dtype
                self.hot_cache[key] = XDSCacheMetadata(path, size, shape, dtype)
                # cache_size_mb = asizeof.asizeof(self.hot_cache) / 1024 / 1024
                # logger.info(f"Hot cache size: {cache_size_mb:.2f} MB ({len(self.hot_cache)} entries)")

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
            assert torch.device(self.dst_device) == torch.device(memory_obj.tensor.device)

        with self.hot_lock:
            entry = self.hot_cache.get(key)

        if entry is None:
            logger.error(f"Metadata not found for {key}")
            return None
        size = entry.size

        if self.config.xds_buffer_size is not None:
            ret = self._load_xds(
                path,
                file_offset=0,
                gpu_pointer=ctypes.c_void_p(self.base_pointer),
                size=size,
                dev_offset=memory_obj.metadata.address
            )
        else:
            buffer = memory_obj.byte_array
            ret = self._load_posix(buffer, path)

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
        assert memory_obj.tensor.is_cuda
        assert torch.device(self.dst_device) == torch.device(memory_obj.tensor.device)

        return self._load_bytes_from_storage_with_memory(key, path, memory_obj)

    def batched_get_blocking(
            self,
            keys: List[CacheEngineKey],
    ) -> list[MemoryObj | None]:
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
            memory_obj = self.memory_allocator.allocate(shape, dtype, fmt=self.fmt)
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
        # logger.info(
        #     f"Time taken for batched_get_blocking: {total_time:.3f}s |"
        #     f" {gds_read_bytes / 1024 / 1024}MiB | {gds_reads} ops."
        # )
        return results

    @_lmcache_nvtx_annotate
    @torch.inference_mode()
    def _save_xds(
            self,
            path: str,
            kv_chunk: torch.Tensor,
            size: int,
            base_pointer: int,
            device_offset: int
    ) -> bytes:
        """ XDS write """
        # Calculate address
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
            ret = self.xds.write(
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
            # Atomic rename
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
    def _save_posix(self, path: str, kv_chunk: torch.Tensor, size: int, buffer):
        """ POSIX protocol write """
        return self.xds.write(path, kv_chunk, size, buffer)

    def _load_xds(self, file_path: str, file_offset: int,
                  gpu_pointer: ctypes.c_void_p, size: int,
                  dev_offset: int) -> int:
        """ XDS protocol read """
        gpu_addr = gpu_pointer.value if gpu_pointer.value else 0

        ret = self.xds.read(
            file_path,
            gpu_addr,
            size,  # Actual size
            file_offset,
            dev_offset,
        )

        if ret < 0:
            logger.error(f"XDS read failed: {file_path}, error={ret}")
        elif ret != size:
            logger.warning(f"XDS read incomplete: {file_path}, {ret}/{size} bytes")
        return ret

    def _load_posix(self, buffer, path):
        """ Read with POSIX """
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

        data_path = meta.path

        metadata_path = data_path + _METADATA_FILE_SUFFIX

        # logger.info(f"Renove {data_path} and {metadata_path}")
        try:
            os.remove(metadata_path)
            os.remove(data_path)
        except Exception as e:
            logger.error(f"Error remove {key}: {e} ")

        # # push kv evict msg
        # if self.lmcache_worker is not None:
        #     self.lmcache_worker.put_msg(
        #         KVEvictMsg(self.instance_id, key.worker_id, key.chunk_hash, str(self))
        #     )

        return True

    def close(self) -> None:
        self.xds.cleanup()
        self._thread_pool.shutdown(wait=True)
        self.memory_allocator.close()
        logger.info("Xds backend closed.")

    # def __del__(self):
    #     HIPFile.get_global_stats()

    def initialize_allocator(
            self, config: LMCacheEngineConfig, metadata: LMCacheEngineMetadata, backend: str
    ) -> HyFileMemoryAllocator:
        assert config.xds_path is not None
        assert config.xds_buffer_size is not None
        logger.info(f"use_mla:{metadata.use_mla};is_first_rank:{metadata.is_first_rank()}")
        if metadata.use_mla and not metadata.is_first_rank():
            return None
        if backend == "hyfile":
            return HyFileMemoryAllocator(size=config.xds_buffer_size * 1024 ** 2)
        else:
            return CuFileMemoryAllocator(size=config.xds_buffer_size * 1024 ** 2)

    def get_allocator_backend(self):
        return self

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
        # if eviction:
        #     logger.warning("Xds Backend does not support eviction")

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
        # if eviction:
        #     logger.warning("Xds Backend does not support eviction")

        return self.memory_allocator.batched_allocate(shape, dtype, batch_size, self.fmt)

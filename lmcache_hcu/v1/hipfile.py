# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0


"""
HIPFile Python Binding - GPUDirect Storage API
Used to call libhyfile.so from Python for GPU direct storage access
Supports 4K alignment, GPU array integration, and performance statistics
"""

import ctypes
import os
import time
from lmcache.logging import init_logger
from ctypes import (
    c_void_p, c_int, c_uint, c_size_t, c_ssize_t, c_bool,
    c_long, c_ulonglong, POINTER, Structure, Union
)
from enum import IntEnum
from typing import Optional, Dict, Any
from dataclasses import dataclass
import threading

logger = init_logger(__name__)

# ============================================================================
# Constant definitions
# ============================================================================

HIPFILEOP_BASE_ERR = 5000
HIP_FILE_RDMA_REGISTER = 1
HIP_FILE_RDMA_RELAXED_ORDERING = (1 << 1)
ALIGNMENT_4K = 4096  # 4K alignment requirement

# ============================================================================
# Enum types
# ============================================================================

class HIPfileOpError(IntEnum):
    """HIPFile error codes"""
    HIP_FILE_SUCCESS = 0
    HIP_FILE_DRIVER_NOT_INITIALIZED = HIPFILEOP_BASE_ERR + 1
    HIP_FILE_DRIVER_INVALID_PROPS = HIPFILEOP_BASE_ERR + 2
    HIP_FILE_DRIVER_UNSUPPORTED_LIMIT = HIPFILEOP_BASE_ERR + 3
    HIP_FILE_DRIVER_VERSION_MISMATCH = HIPFILEOP_BASE_ERR + 4
    HIP_FILE_DRIVER_VERSION_READ_ERROR = HIPFILEOP_BASE_ERR + 5
    HIP_FILE_DRIVER_CLOSING = HIPFILEOP_BASE_ERR + 6
    HIP_FILE_PLATFORM_NOT_SUPPORTED = HIPFILEOP_BASE_ERR + 7
    HIP_FILE_IO_NOT_SUPPORTED = HIPFILEOP_BASE_ERR + 8
    HIP_FILE_DEVICE_NOT_SUPPORTED = HIPFILEOP_BASE_ERR + 9
    HIP_FILE_HYFS_DRIVER_ERROR = HIPFILEOP_BASE_ERR + 10
    HIP_FILE_ROCM_DRIVER_ERROR = HIPFILEOP_BASE_ERR + 11
    HIP_FILE_ROCM_POINTER_INVALID = HIPFILEOP_BASE_ERR + 12
    HIP_FILE_ROCM_MEMORY_TYPE_INVALID = HIPFILEOP_BASE_ERR + 13
    HIP_FILE_ROCM_POINTER_RANGE_ERROR = HIPFILEOP_BASE_ERR + 14
    HIP_FILE_ROCM_CONTEXT_MISMATCH = HIPFILEOP_BASE_ERR + 15
    HIP_FILE_INVALID_MAPPING_SIZE = HIPFILEOP_BASE_ERR + 16
    HIP_FILE_INVALID_MAPPING_RANGE = HIPFILEOP_BASE_ERR + 17
    HIP_FILE_INVALID_FILE_TYPE = HIPFILEOP_BASE_ERR + 18
    HIP_FILE_INVALID_FILE_OPEN_FLAG = HIPFILEOP_BASE_ERR + 19
    HIP_FILE_DIO_NOT_SET = HIPFILEOP_BASE_ERR + 20
    HIP_FILE_INVALID_VALUE = HIPFILEOP_BASE_ERR + 22
    HIP_FILE_MEMORY_ALREADY_REGISTERED = HIPFILEOP_BASE_ERR + 23
    HIP_FILE_MEMORY_NOT_REGISTERED = HIPFILEOP_BASE_ERR + 24
    HIP_FILE_PERMISSION_DENIED = HIPFILEOP_BASE_ERR + 25
    HIP_FILE_DRIVER_ALREADY_OPEN = HIPFILEOP_BASE_ERR + 26
    HIP_FILE_HANDLE_NOT_REGISTERED = HIPFILEOP_BASE_ERR + 27
    HIP_FILE_HANDLE_ALREADY_REGISTERED = HIPFILEOP_BASE_ERR + 28
    HIP_FILE_DEVICE_NOT_FOUND = HIPFILEOP_BASE_ERR + 29
    HIP_FILE_INTERNAL_ERROR = HIPFILEOP_BASE_ERR + 30
    HIP_FILE_GETNEWFD_FAILED = HIPFILEOP_BASE_ERR + 31
    HIP_FILE_HYFS_SETUP_ERROR = HIPFILEOP_BASE_ERR + 33
    HIP_FILE_IO_DISABLED = HIPFILEOP_BASE_ERR + 34
    HIP_FILE_BATCH_SUBMIT_FAILED = HIPFILEOP_BASE_ERR + 35
    HIP_FILE_GPU_MEMORY_PINNING_FAILED = HIPFILEOP_BASE_ERR + 36
    HIP_FILE_IO_MAX_ERROR = HIPFILEOP_BASE_ERR + 37



class HIPfileFileHandleType(IntEnum):
    """File handle type"""
    HIP_FILE_HANDLE_TYPE_OPAQUE_FD = 1
    HIP_FILE_HANDLE_TYPE_OPAQUE_WIN32 = 2
    HIP_FILE_HANDLE_TYPE_USERSPACE_FS = 3


class HIPfileDriverStatusFlags(IntEnum):
    """Driver status flags"""
    HIP_FILE_LUSTRE_SUPPORTED = 0
    HIP_FILE_WEKAFS_SUPPORTED = 1
    HIP_FILE_NFS_SUPPORTED = 2
    HIP_FILE_GPFS_SUPPORTED = 3
    HIP_FILE_NVME_SUPPORTED = 4
    HIP_FILE_NVMEOF_SUPPORTED = 5
    HIP_FILE_SCSI_SUPPORTED = 6
    HIP_FILE_SCALEFLUX_CSD_SUPPORTED = 7
    HIP_FILE_NVMESH_SUPPORTED = 8
    HIP_FILE_BEEGFS_SUPPORTED = 9


class HIPfileOpcode(IntEnum):
    """Batch operation type"""
    HIPFILE_READ = 0
    HIPFILE_WRITE = 1


class HIPfileStatus(IntEnum):
    """Batch operation status"""
    HIPFILE_WAITING = 0x000001
    HIPFILE_PENDING = 0x000002
    HIPFILE_INVALID = 0x000004
    HIPFILE_CANCELED = 0x000008
    HIPFILE_COMPLETE = 0x0000010
    HIPFILE_TIMEOUT = 0x0000020
    HIPFILE_FAILED = 0x0000040


# ============================================================================
# C structure definitions
# ============================================================================

class HIPfileError_t(Structure):
    """Error structure"""
    _fields_ = [
        ("err", c_int),
        ("hip_err", c_int)
    ]


class HYFSProps(Structure):
    """HYFS driver attributes"""
    _fields_ = [
        ("major_version", c_uint),
        ("minor_version", c_uint),
        ("poll_thresh_size", c_size_t),
        ("max_direct_io_size", c_size_t),
        ("dstatusflags", c_uint),
        ("dcontrolflags", c_uint),
    ]


class HIPfileDrvProps_t(Structure):
    """Driver attributes"""
    _fields_ = [
        ("hyfs", HYFSProps),
        ("fflags", c_uint),
        ("max_device_cache_size", c_uint),
        ("per_buffer_cache_size", c_uint),
        ("max_device_pinned_mem_size", c_uint),
        ("max_batch_io_size", c_uint),
        ("max_batch_io_timeout_msecs", c_uint),
    ]


class FileHandleUnion(Union):
    """File handle union"""
    _fields_ = [
        ("fd", c_int),
        ("handle", c_void_p)
    ]


class HIPfileDescr_t(Structure):
    """File descriptor"""
    _fields_ = [
        ("type", c_int),
        ("handle", FileHandleUnion),
        ("fs_ops", c_void_p)
    ]


class BatchParams(Structure):
    """Batch operation parameters"""
    _fields_ = [
        ("devPtr_base", c_void_p),
        ("file_offset", c_ulonglong),
        ("devPtr_offset", c_ulonglong),
        ("size", c_size_t),
    ]


class ParamsUnion(Union):
    """Parameter union"""
    _fields_ = [
        ("batch", BatchParams)
    ]


class HIPfileIOParams_t(Structure):
    """IO parameters"""
    _fields_ = [
        ("mode", c_int),
        ("u", ParamsUnion),
        ("fh", c_void_p),
        ("opcode", c_int),
        ("cookie", c_void_p),
    ]


class HIPfileIOEvents_t(Structure):
    """IO event"""
    _fields_ = [
        ("cookie", c_void_p),
        ("status", c_int),
        ("ret", c_size_t),
    ]


# ============================================================================
# Statistics
# ============================================================================

@dataclass
class IOStats:
    """IO Statistics"""
    total_bytes: int = 0
    total_time: float = 0.0
    num_operations: int = 0

    @property
    def throughput_mbps(self) -> float:
        """Throughput in MB/s"""
        if self.total_time == 0:
            return 0.0
        return (self.total_bytes / (1024 * 1024)) / self.total_time

    @property
    def avg_latency_ms(self) -> float:
        """Average latency in milliseconds"""
        if self.num_operations == 0:
            return 0.0
        return (self.total_time / self.num_operations) * 1000

    def reset(self):
        """Reset statistics"""
        self.total_bytes = 0
        self.total_time = 0.0
        self.num_operations = 0


# ============================================================================
# Exception classes
# ============================================================================

class HIPfileException(Exception):
    """HIPFile Exception classes"""

    # Error message mapping
    ERROR_MESSAGES = {
        HIPfileOpError.HIP_FILE_SUCCESS: "hipfile success",
        HIPfileOpError.HIP_FILE_DRIVER_NOT_INITIALIZED: "hyfs driver is not loaded",
        HIPfileOpError.HIP_FILE_DRIVER_INVALID_PROPS: "invalid property",
        HIPfileOpError.HIP_FILE_DRIVER_UNSUPPORTED_LIMIT: "property range error",
        HIPfileOpError.HIP_FILE_DRIVER_VERSION_MISMATCH: "hyfs driver version mismatch",
        HIPfileOpError.HIP_FILE_DRIVER_VERSION_READ_ERROR: "hyfs driver version read error",
        HIPfileOpError.HIP_FILE_DRIVER_CLOSING: "driver shutdown in progress",
        HIPfileOpError.HIP_FILE_PLATFORM_NOT_SUPPORTED: "GPUDirect Storage not supported on current platform",
        HIPfileOpError.HIP_FILE_IO_NOT_SUPPORTED: "GPUDirect Storage not supported on current file",
        HIPfileOpError.HIP_FILE_DEVICE_NOT_SUPPORTED: "GPUDirect Storage not supported on current GPU",
        HIPfileOpError.HIP_FILE_HYFS_DRIVER_ERROR: "hyfs driver ioctl error",
        HIPfileOpError.HIP_FILE_ROCM_DRIVER_ERROR: "ROCm Driver API error",
        HIPfileOpError.HIP_FILE_ROCM_POINTER_INVALID: "invalid device pointer",
        HIPfileOpError.HIP_FILE_ROCM_MEMORY_TYPE_INVALID: "invalid pointer memory type",
        HIPfileOpError.HIP_FILE_ROCM_POINTER_RANGE_ERROR: "pointer range exceeds allocated address range",
        HIPfileOpError.HIP_FILE_ROCM_CONTEXT_MISMATCH: "cuda context mismatch",
        HIPfileOpError.HIP_FILE_INVALID_MAPPING_SIZE: "access beyond maximum pinned size",
        HIPfileOpError.HIP_FILE_INVALID_MAPPING_RANGE: "access beyond mapped size",
        HIPfileOpError.HIP_FILE_INVALID_FILE_TYPE: "unsupported file type",
        HIPfileOpError.HIP_FILE_INVALID_FILE_OPEN_FLAG: "unsupported file open flags",
        HIPfileOpError.HIP_FILE_DIO_NOT_SET: "fd direct IO not set",
        HIPfileOpError.HIP_FILE_INVALID_VALUE: "invalid arguments",
        HIPfileOpError.HIP_FILE_MEMORY_ALREADY_REGISTERED: "device pointer already registered",
        HIPfileOpError.HIP_FILE_MEMORY_NOT_REGISTERED: "device pointer lookup failure",
        HIPfileOpError.HIP_FILE_PERMISSION_DENIED: "driver or file access error",
        HIPfileOpError.HIP_FILE_DRIVER_ALREADY_OPEN: "driver is already open",
        HIPfileOpError.HIP_FILE_HANDLE_NOT_REGISTERED: "file descriptor is not registered",
        HIPfileOpError.HIP_FILE_HANDLE_ALREADY_REGISTERED: "file descriptor is already registered",
        HIPfileOpError.HIP_FILE_DEVICE_NOT_FOUND: "GPU device not found",
        HIPfileOpError.HIP_FILE_INTERNAL_ERROR: "internal error",
        HIPfileOpError.HIP_FILE_GETNEWFD_FAILED: "failed to obtain new file descriptor",
        HIPfileOpError.HIP_FILE_HYFS_SETUP_ERROR: "HYFS driver initialization error",
        HIPfileOpError.HIP_FILE_IO_DISABLED: "GPUDirect Storage disabled by config on current file",
        HIPfileOpError.HIP_FILE_BATCH_SUBMIT_FAILED: "failes to submit batch operation",
        HIPfileOpError.HIP_FILE_GPU_MEMORY_PINNING_FAILED: "Failed to allocate pinned GPU Memory",
        HIPfileOpError.HIP_FILE_IO_MAX_ERROR: "GPUDirect Storage Max Error",
    }

    def __init__(self, error: HIPfileError_t, message: str = ""):
        self.error_code = error.err
        self.hip_error = error.hip_err
        
        # Get detailed error message
        print(error.err)
        if message:
            self.message = message
        else:
            self.message = self.ERROR_MESSAGES.get(
                self.error_code,
                f"HIPFile error code: {self.error_code}"
            )
        
        super().__init__(self.message)

    def __str__(self):
        msg = f"HIPfileException: {self.message}"
        if self.hip_error != 0:
            msg += f" (HIP error: {self.hip_error})"
        return msg


# ============================================================================
# HIPFile library wrapper class
# ============================================================================

class HIPFileLib:
    """libhyfile.so library wrapper"""

    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize the HIPFile library

        Args:
            lib_path: path to libhyfile.so; automatically searched when None
        """
        if lib_path is None:
            possible_paths = [
                "/opt/hyhal/lib/libhyfile.so",
                "/usr/lib/libhyfile.so",
                "/usr/local/lib/libhyfile.so",
            ]
            for path in possible_paths:
                if os.path.exists(path):
                    lib_path = path
                    break
            else:
                raise FileNotFoundError(
                    "Cannot find libhyfile.so. Please specify lib_path explicitly."
                )

        self._lib = ctypes.CDLL(lib_path)
        self._setup_functions()
        # self.logger = logging.getLogger("hipfile.lib")

    def _setup_functions(self):
        """Set function signatures"""

        # hipFileDriverOpen
        self._lib.hipFileDriverOpen.argtypes = []
        self._lib.hipFileDriverOpen.restype = HIPfileError_t

        # hipFileDriverClose_v2
        self._lib.hipFileDriverClose_v2.argtypes = []
        self._lib.hipFileDriverClose_v2.restype = HIPfileError_t

        # hipFileDriverSetMaxCacheSize
        self._lib.hipFileDriverSetMaxCacheSize.argtypes = [c_size_t]
        self._lib.hipFileDriverSetMaxCacheSize.restype = HIPfileError_t

        # hipFileDriverSetMaxPinnedMemSize
        self._lib.hipFileDriverSetMaxPinnedMemSize.argtypes = [c_size_t]
        self._lib.hipFileDriverSetMaxPinnedMemSize.restype = HIPfileError_t

        # hipFileHandleRegister
        self._lib.hipFileHandleRegister.argtypes = [
            POINTER(c_void_p),
            POINTER(HIPfileDescr_t)
        ]
        self._lib.hipFileHandleRegister.restype = HIPfileError_t

        # hipFileHandleDeregister
        self._lib.hipFileHandleDeregister.argtypes = [c_void_p]
        self._lib.hipFileHandleDeregister.restype = None

        # hipFileBufRegister
        self._lib.hipFileBufRegister.argtypes = [c_void_p, c_size_t, c_int]
        self._lib.hipFileBufRegister.restype = HIPfileError_t

        # hipFileBufDeregister
        self._lib.hipFileBufDeregister.argtypes = [c_void_p]
        self._lib.hipFileBufDeregister.restype = HIPfileError_t

        # hipFileRead
        self._lib.hipFileRead.argtypes = [
            c_void_p, c_void_p, c_size_t, c_ulonglong, c_ulonglong
        ]
        self._lib.hipFileRead.restype = c_ssize_t

        # hipFileWrite
        self._lib.hipFileWrite.argtypes = [
            c_void_p, c_void_p, c_size_t, c_ulonglong, c_ulonglong
        ]
        self._lib.hipFileWrite.restype = c_ssize_t


    def driver_open(self):
        """Open driver"""
        logger.info("Opening HIPFile driver...")
        err = self._lib.hipFileDriverOpen()
        if err.err != HIPfileOpError.HIP_FILE_SUCCESS:
            raise HIPfileException(err, "Failed to open HIPFile driver")
        logger.info("HIPFile driver opened successfully")

    def driver_close(self):
        """Close driver"""
        logger.info("Closing HIPFile driver...")
        err = self._lib.hipFileDriverClose_v2()
        if err.err != HIPfileOpError.HIP_FILE_SUCCESS:
            raise HIPfileException(err, "Failed to close HIPFile driver")
        logger.info("HIPFile driver closed successfully")


    def set_poll_mode(self, poll: bool, threshold_kb: int):
        """Set polling mode"""
        err = self._lib.hipFileDriverSetPollMode(poll, threshold_kb)
        if err.err != HIPfileOpError.HIP_FILE_SUCCESS:
            raise HIPfileException(err, "Failed to set poll mode")


    def set_max_cache_size(self, size_kb: int):
        """Set maximum cache size"""
        err = self._lib.hipFileDriverSetMaxCacheSize(size_kb)
        if err.err != HIPfileOpError.HIP_FILE_SUCCESS:
            raise HIPfileException(err, "Failed to set max cache size")

    def set_max_pinned_mem_size(self, size_kb: int):
        """Set maximum pinned memory size"""
        err = self._lib.hipFileDriverSetMaxPinnedMemSize(size_kb)
        if err.err != HIPfileOpError.HIP_FILE_SUCCESS:
            raise HIPfileException(err, "Failed to set max pinned memory size")

    def register_file_handle(self, fd: int) -> c_void_p:
        """Register file handle"""
        descr = HIPfileDescr_t()
        descr.type = HIPfileFileHandleType.HIP_FILE_HANDLE_TYPE_OPAQUE_FD
        descr.handle.fd = fd
        descr.fs_ops = None

        fh = c_void_p()
        err = self._lib.hipFileHandleRegister(ctypes.byref(fh), ctypes.byref(descr))
        if err.err != HIPfileOpError.HIP_FILE_SUCCESS:
            raise HIPfileException(err, f"Failed to register file handle (fd={fd})")

        return fh

    def deregister_file_handle(self, fh: c_void_p):
        """Deregister file handle"""
        self._lib.hipFileHandleDeregister(fh)

    def register_buffer(self, dev_ptr: int, size: int, 
                       flags: int = HIP_FILE_RDMA_REGISTER):
        """Register device memory buffer"""
        err = self._lib.hipFileBufRegister(c_void_p(dev_ptr), size, flags)
        if err.err != HIPfileOpError.HIP_FILE_SUCCESS:
            raise HIPfileException(err, f"Failed to register buffer (ptr={hex(dev_ptr)})")

    def deregister_buffer(self, dev_ptr: int):
        """Deregister device memory buffer"""
        err = self._lib.hipFileBufDeregister(c_void_p(dev_ptr))
        if err.err != HIPfileOpError.HIP_FILE_SUCCESS:
            raise HIPfileException(err, f"Failed to deregister buffer (ptr={hex(dev_ptr)})")

    def read(self, fh: c_void_p, dev_ptr: int, size: int,
             file_offset: int = 0, dev_offset: int = 0) -> int:
        """Read from file to device memory"""
        ret = self._lib.hipFileRead(
            fh, c_void_p(dev_ptr), size, file_offset, dev_offset
        )
        if ret < 0:
            if ret < -HIPFILEOP_BASE_ERR:
                err = HIPfileError_t(err=abs(ret), hip_err=0)
                raise HIPfileException(err, "Read operation failed")
            else:
                raise OSError(abs(ret), os.strerror(abs(ret)))
        return ret

    def write(self, fh: c_void_p, dev_ptr: int, size: int,
              file_offset: int = 0, dev_offset: int = 0) -> int:
        """Write from device memory to file"""
        ret = self._lib.hipFileWrite(
            fh, c_void_p(dev_ptr), size, file_offset, dev_offset
        )
        if ret < 0:
            if ret < -HIPFILEOP_BASE_ERR:
                err = HIPfileError_t(err=abs(ret), hip_err=0)
                raise HIPfileException(err, "Write operation failed")
            else:
                raise OSError(abs(ret), os.strerror(abs(ret)))
        return ret


# ============================================================================
# HIPFile high-level wrapper class
# ============================================================================

class HIPFile:
    """
    HIPFile high-level wrapper class
    """
    _lib_instance = None
    _driver_opened = False
    _global_registered_buffers = {}  # Globally registered buffers {base_ptr: (size, refcount)}

    _global_stats_lock = threading.Lock()  # Lock protecting global statistics
    global_read_stats = IOStats()          # Global read statistics
    global_write_stats = IOStats()         # Global write statistics

    _periodic_logger_thread: Optional[threading.Thread] = None
    _stop_logging = threading.Event()


    def __init__(self, 
                 filepath: str, 
                 mode: str = 'r', 
                 lib_path: Optional[str] = None,
                 enable_stats: bool = False,
                 check_alignment: bool = True,
                 auto_register_buffer: bool = False,
                 skip_buffer_registration: bool = False):
        """
        Initialize HIPFile object
        Args:
            filepath: File path
            mode: Open mode ('r', 'w', 'r+', 'a')
            lib_path: libhyfile.so path; automatically searched when None
            enable_stats: Whether to enable performance statistics
            check_alignment: Whether to check 4K alignment
            auto_register_buffer: Whether to automatically register buffers on first use
            skip_buffer_registration: Whether to skip buffer registration checks
        """
        # Initialize library instance
        if HIPFile._lib_instance is None:
            HIPFile._lib_instance = HIPFileLib(lib_path)

        self.lib = HIPFile._lib_instance
        self.filepath = filepath
        self.mode = mode
        self.fd = None
        self.fh = None
        
        # Buffer management
        self._registered_buffers = set()
        self.auto_register_buffer = auto_register_buffer
        self.skip_buffer_registration = skip_buffer_registration
        
        # Performance statistics
        self.enable_stats = enable_stats
        self.read_stats = IOStats()
        self.write_stats = IOStats()
        
        # Alignment check
        self.check_alignment = check_alignment
        self.alignment = 4096

        # # Logging
        # self.logger = logging.getLogger(f"hipfile.{os.path.basename(filepath)}")

    def __enter__(self):
        """Context manager entry"""
        self.open()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
        return False

    def open(self):
        """
        Open file
        Automatically handle driver initialization and file handle registration
        """
        # Open the driver if it is not already open
        if not HIPFile._driver_opened:
            self.lib.driver_open()
            HIPFile._driver_opened = True

        # Build open flags
        flags = os.O_DIRECT  # O_DIRECT is required
        if self.mode == 'r':
            flags |= os.O_RDONLY
        elif self.mode == 'w':
            flags |= os.O_WRONLY | os.O_CREAT | os.O_TRUNC
        elif self.mode == 'r+':
            flags |= os.O_RDWR
        elif self.mode == 'a':
            flags |= os.O_WRONLY | os.O_CREAT | os.O_APPEND
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

        # Open file
        self.fd = os.open(self.filepath, flags, 0o644)
        logger.debug(f"Opened file {self.filepath} with fd={self.fd}")
        
        # Register file handle
        self.fh = self.lib.register_file_handle(self.fd)
        logger.debug(f"Registered file handle: {self.fh}")

    def close(self):
        """
        Close file
        
        Automatically deregister all buffers and file handles
        """
        # Deregister all buffers
        for dev_ptr in list(self._registered_buffers):
            try:
                self.deregister_buffer(dev_ptr)
            except Exception as e:
                logger.warning(f"Failed to deregister buffer {hex(dev_ptr)}: {e}")

        # Deregister file handle
        if self.fh is not None:
            self.lib.deregister_file_handle(self.fh)
            logger.debug(f"Deregistered file handle: {self.fh}")
            self.fh = None

        # Close file descriptor
        if self.fd is not None:
            os.close(self.fd)
            logger.debug(f"Closed file descriptor: {self.fd}")
            self.fd = None

    def _check_alignment(self, ptr: int, size: int, offset: int):
        """
        Check 4K alignment
        Args:
            ptr: Memory pointer
            size: Data size
            offset: Offset
        Raises:
            ValueError: If not aligned
        """
        if not self.check_alignment:
            return

        if ptr % self.alignment != 0:
            logger.warning(
                f"Device pointer {hex(ptr)} not aligned to {self.alignment} bytes. "
                f"Use aligned memory allocation or disable check_alignment."
            )
        if size % self.alignment != 0:
            logger.warning(
                f"Size {size} not aligned to {self.alignment} bytes. "
                f"Use size that is multiple of {self.alignment}."
            )
        if offset % self.alignment != 0:
            logger.warning(
                f"Offset {offset} not aligned to {self.alignment} bytes."
            )

    @staticmethod
    def align_up(value: int, alignment: int = ALIGNMENT_4K) -> int:
        """
        Align up to the specified boundary
        Args:
            value: Value to align
            alignment: Alignment boundary, default 4K
            
        Returns:
            Aligned value
        """
        return ((value + alignment - 1) // alignment) * alignment

    @staticmethod
    def align_down(value: int, alignment: int = ALIGNMENT_4K) -> int:
        """
        Align down to the specified boundary
        
        Args:
            value: Value to align
            alignment: Alignment boundary, default 4K
            
        Returns:
            Aligned value
        """
        return (value // alignment) * alignment

    def register_buffer(self, dev_ptr: int, size: int, 
                       flags: int = HIP_FILE_RDMA_REGISTER):
        """
        Register device buffer
        Args:
            dev_ptr: Device pointer address
            size: Buffer size in bytes
            flags: Registration flags
            
        Note:
            - Pre-registering buffers is recommended for better performance
            - Buffer address and size should be 4K aligned
        """
        if dev_ptr in self._registered_buffers:
            logger.warning(f"Buffer {hex(dev_ptr)} already registered")
            return

        self.lib.register_buffer(dev_ptr, size, flags)
        self._registered_buffers.add(dev_ptr)
        logger.debug(f"Registered buffer: {hex(dev_ptr)}, size={size}")

    def deregister_buffer(self, dev_ptr: int):
        """
        Deregister device buffer
        
        Args:
            dev_ptr: Device pointer address
        """
        if dev_ptr not in self._registered_buffers:
            logger.warning(f"Buffer {hex(dev_ptr)} not registered")
            return

        self.lib.deregister_buffer(dev_ptr)
        self._registered_buffers.discard(dev_ptr)
        logger.debug(f"Deregistered buffer: {hex(dev_ptr)}")

    @classmethod
    def register_global_buffer(cls, dev_ptr: int, size: int, 
                              flags: int = HIP_FILE_RDMA_REGISTER):
        """
        Globally register buffer (class method)
        
        Args:
            dev_ptr: Device pointer base address
            size: Total buffer size
            flags: Registration flags
            
        Note:
            - Globally registered buffers can be used by all HIPFile instances
            - Reference counting is supported; registering the same buffer multiple times increments the reference count
            - Must be called after opening the driver
        """
        if cls._lib_instance is None:
            cls._lib_instance = HIPFileLib()
        
        if not cls._driver_opened:
            raise RuntimeError(
                "Driver not opened. Call HIPFile.init_driver() first."
            )
        
        # Check whether already registered
        if dev_ptr in cls._global_registered_buffers:
            # Increment reference count
            old_size, refcount = cls._global_registered_buffers[dev_ptr]
            if old_size != size:
                raise ValueError(
                    f"Buffer {hex(dev_ptr)} already registered with different size: "
                    f"{old_size} != {size}"
                )
            cls._global_registered_buffers[dev_ptr] = (size, refcount + 1)
            logger.debug(
                f"Increased refcount for buffer {hex(dev_ptr)} to {refcount + 1}"
            )
            return
        
        # Register new buffer
        cls._lib_instance.register_buffer(dev_ptr, size, flags)
        cls._global_registered_buffers[dev_ptr] = (size, 1)
        
        logger.info(
            f"Globally registered buffer: {hex(dev_ptr)}, size={size} bytes"
        )

    @classmethod
    def deregister_global_buffer(cls, dev_ptr: int):
        """
        Globally deregister buffer (class method)
        
        Args:
            dev_ptr: Device pointer base address
            
        Note:
            - Reference counting is supported; deregister only when the reference count reaches 0
            - Usually called before program exit
        """
        if dev_ptr not in cls._global_registered_buffers:
            logger.warning(
                f"Buffer {hex(dev_ptr)} not globally registered"
            )
            return
        
        size, refcount = cls._global_registered_buffers[dev_ptr]
        
        if refcount > 1:
            # Decrement reference count
            cls._global_registered_buffers[dev_ptr] = (size, refcount - 1)
            logger.debug(
                f"Decreased refcount for buffer {hex(dev_ptr)} to {refcount - 1}"
            )
            return
        
        # Reference count is 1; actually deregister
        if cls._lib_instance is not None:
            cls._lib_instance.deregister_buffer(dev_ptr)
        
        del cls._global_registered_buffers[dev_ptr]
        
        logger.info(
            f"Globally deregistered buffer: {hex(dev_ptr)}"
        )

    @classmethod
    def is_buffer_registered(cls, dev_ptr: int) -> bool:
        """
        Check whether the buffer is globally registered
        
        Args:
            dev_ptr: Device pointer, either a base address or a subregion address
            
        Returns:
            Whether registered
        """
        # Check exact match
        if dev_ptr in cls._global_registered_buffers:
            return True
        
        # Check whether it is within a registered buffer range
        for base_ptr, (size, _) in cls._global_registered_buffers.items():
            if base_ptr <= dev_ptr < base_ptr + size:
                return True
        
        return False

    @classmethod
    def init_driver(cls, lib_path: Optional[str] = None):
        """
        Initialize driver (class method)
        
        Args:
            lib_path: libhyfile.so path
            
        Note:
            Must be called before using global buffer registration
        """
        if cls._lib_instance is None:
            cls._lib_instance = HIPFileLib(lib_path)
        
        if not cls._driver_opened:
            cls._lib_instance.driver_open()
            cls._driver_opened = True

    def _should_skip_registration(self, dev_ptr: int) -> bool:
        """
        Determine whether buffer registration should be skipped
        Args:
            dev_ptr: Device pointer
        Returns:
            Whether to skip registration
        """
        # If the skip flag is set
        if self.skip_buffer_registration:
            return True
        
        # If already registered locally
        if dev_ptr in self._registered_buffers:
            return True
        
        # If already registered globally
        if self.is_buffer_registered(dev_ptr):
            return True
        
        return False

    def read(self, dev_ptr: int, size: int, 
             file_offset: int = 0, dev_offset: int = 0) -> int:
        """
        Read data from file to device memory
        Args:
            dev_ptr: Device memory base address, must be 4K aligned
            size: Read size in bytes, must be 4K aligned
            file_offset: File offset in bytes, must be 4K aligned
            dev_offset: Offset relative to dev_ptr in bytes
        Returns:
            Actual number of bytes read
        Note:
            - If skip_buffer_registration=True is set, assume the buffer is pre-registered
            - Supports using subregions of globally registered buffers
        """
        if self.fh is None:
            raise RuntimeError("File not opened. Use 'with HIPFile(...)' or call open()")

        # Alignment check
        self._check_alignment(dev_ptr, size, file_offset)
        self._check_alignment(dev_ptr + dev_offset, 0, 0)

        # Buffer registration check
        if not self._should_skip_registration(dev_ptr):
            if self.auto_register_buffer:
                self.register_buffer(dev_ptr, size)
            else:
               logger.warning(
                    f"Buffer {hex(dev_ptr)} not registered. "
                    "Consider using skip_buffer_registration=True if pre-registered."
                )
        # Perform read
        start_time = time.perf_counter() if self.enable_stats else 0
        ret = self.lib.read(self.fh, dev_ptr, size, file_offset, dev_offset)

        # Update statistics
        if self.enable_stats:
            elapsed = time.perf_counter() - start_time
            self.read_stats.total_bytes += ret
            self.read_stats.total_time += elapsed
            self.read_stats.num_operations += 1

            throughput = ret / (1024 * 1024 * elapsed) if elapsed > 0 else 0
            logger.debug(
                f"Read {ret} bytes in {elapsed*1000:.2f}ms ({throughput:.2f} MB/s)"
            )
            with HIPFile._global_stats_lock:
                HIPFile.global_read_stats.total_bytes += ret
                HIPFile.global_read_stats.total_time += elapsed
                HIPFile.global_read_stats.num_operations += 1
        return ret

    def write(self, dev_ptr: int, size: int, 
              file_offset: int = 0, dev_offset: int = 0) -> int:
        """
        Write data from device memory to file
        
        Args:
            dev_ptr: Device memory base address, must be 4K aligned
            size: Write size in bytes, must be 4K aligned
            file_offset: File offset in bytes, must be 4K aligned
            dev_offset: Offset relative to dev_ptr in bytes
            
        Returns:
            Actual number of bytes written
            
        Note:
            - If skip_buffer_registration=True is set, assume the buffer is pre-registered
            - Supports using subregions of globally registered buffers
        """
        if self.fh is None:
            raise RuntimeError("File not opened. Use 'with HIPFile(...)' or call open()")

        # Alignment check
        self._check_alignment(dev_ptr, size, file_offset)
        self._check_alignment(dev_ptr + dev_offset, 0, 0)

        # Buffer registration check
        if not self._should_skip_registration(dev_ptr):
            if self.auto_register_buffer:
                self.register_buffer(dev_ptr, size)
            else:
                logger.warning(
                    f"Buffer {hex(dev_ptr)} not registered. "
                    "Consider using skip_buffer_registration=True if pre-registered."
                )

        # Perform write
        start_time = time.perf_counter() if self.enable_stats else 0
        ret = self.lib.write(self.fh, dev_ptr, size, file_offset, dev_offset)

        # Update statistics
        if self.enable_stats:
            elapsed = time.perf_counter() - start_time
            self.write_stats.total_bytes += ret
            self.write_stats.total_time += elapsed
            self.write_stats.num_operations += 1

            throughput = ret / (1024 * 1024 * elapsed) if elapsed > 0 else 0
            logger.debug(
                f"Write {ret} bytes in {elapsed*1000:.2f}ms ({throughput:.2f} MB/s)"
            )
            with HIPFile._global_stats_lock:
                HIPFile.global_write_stats.total_bytes += ret
                HIPFile.global_write_stats.total_time += elapsed
                HIPFile.global_write_stats.num_operations += 1

        return ret

    @classmethod
    def get_global_stats(cls) -> Dict[str, Any]:
        """
        Get global performance statistics
        
        Returns:
            Dictionary containing global read/write statistics
        """
        with cls._global_stats_lock:
            return {
                "read": {
                    "total_bytes": cls.global_read_stats.total_bytes,
                    "throughput_mbps": cls.global_read_stats.throughput_mbps,
                    "num_ops": cls.global_read_stats.num_operations,
                    "avg_latency_ms": cls.global_read_stats.avg_latency_ms,
                },
                "write": {
                    "total_bytes": cls.global_write_stats.total_bytes,
                    "throughput_mbps": cls.global_write_stats.throughput_mbps,
                    "num_ops": cls.global_write_stats.num_operations,
                    "avg_latency_ms": cls.global_write_stats.avg_latency_ms,
                }
            }

    @classmethod
    def print_global_stats(cls):
        """Print global performance statistics"""
        stats = cls.get_global_stats()
        print("\n" + "="*60)
        print("HIPFile Global Performance Statistics")
        print("="*60)
        
        print("\nGlobal Read Operations:")
        print(f"  Total bytes:     {stats['read']['total_bytes']:>15,} bytes")
        print(f"  Throughput:      {stats['read']['throughput_mbps']:>15.2f} MB/s")
        print(f"  Operations:      {stats['read']['num_ops']:>15,}")
        print(f"  Avg latency:     {stats['read']['avg_latency_ms']:>15.2f} ms")
        
        print("\nGlobal Write Operations:")
        print(f"  Total bytes:     {stats['write']['total_bytes']:>15,} bytes")
        print(f"  Throughput:      {stats['write']['throughput_mbps']:>15.2f} MB/s")
        print(f"  Operations:      {stats['write']['num_ops']:>15,}")
        print(f"  Avg latency:     {stats['write']['avg_latency_ms']:>15.2f} ms")
        print("="*60 + "\n")

    @classmethod
    def reset_global_stats(cls):
        """Reset global performance statistics (thread-safe)"""
        with cls._global_stats_lock:
            cls.global_read_stats.reset()
            cls.global_write_stats.reset()
        logger.debug("Global statistics reset")

    @classmethod
    def start_periodic_logging(cls, interval: int = 300):
        """
        Start periodic global statistics logging every interval seconds in a background thread
        
        Args:
            interval: Output interval in seconds, default 300 or 5 minutes
            
        Note:
            - Start only once; subsequent calls have no effect.
            - Use logger.info for output.
            - The thread is a daemon thread and stops automatically when the program exits.
        """
        if cls._periodic_logger_thread is not None and cls._periodic_logger_thread.is_alive():
            logger.warning("Periodic logging already started")
            return

        cls._stop_logging.clear()  # Reset stop flag

        def _log_worker():
            while not cls._stop_logging.is_set():
                time.sleep(interval)
                if cls._stop_logging.is_set():
                    break
                # Get global statistics and write them to the log
                stats = cls.get_global_stats()
                logger.info(
                    f"Global I/O Stats - Read: {stats['read']['total_bytes']} bytes, "
                    f"{stats['read']['throughput_mbps']:.2f} MB/s, {stats['read']['num_ops']} ops; "
                    f"Write: {stats['write']['total_bytes']} bytes, "
                    f"{stats['write']['throughput_mbps']:.2f} MB/s, {stats['write']['num_ops']} ops"
                )

        cls._periodic_logger_thread = threading.Thread(target=_log_worker, daemon=True)
        cls._periodic_logger_thread.start()
        logger.info(f"Started periodic global stats logging every {interval} seconds")

    @classmethod
    def stop_periodic_logging(cls):
        """
        Stop periodic logging
        
        Note:
            Optional; usually called before program exit.
        """
        if cls._periodic_logger_thread is not None:
            cls._stop_logging.set()
            cls._periodic_logger_thread.join(timeout=interval + 1)  # Wait for the thread to finish
            logger.info("Stopped periodic global stats logging")
            cls._periodic_logger_thread = None


    @classmethod
    def close_driver(cls):
        """
        Close driver (class method)
        
        Note:
            Manual calls are usually unnecessary; it closes automatically on program exit
        """
        if cls._driver_opened and cls._lib_instance is not None:
            cls._lib_instance.driver_close()
            cls._driver_opened = False


# ============================================================================
# Driver context manager
# ============================================================================

class HIPFileDriver:
    """
    HIPFile Driver context manager
    
    Used to explicitly manage driver lifecycle and configuration
    
    Examples:
        >>> with HIPFileDriver() as driver:
        ...     driver.set_max_cache_size(1024 * 1024)  # 1GB
        ...     props = driver.get_properties()
        ...     with HIPFile("data.bin", "r") as f:
        ...         f.read(dev_ptr, size)
    """

    def __init__(self, lib_path: Optional[str] = None):
        """
        Initialize driver manager
        
        Args:
            lib_path: libhyfile.so path
        """
        self.lib = HIPFileLib(lib_path)
        self._opened = False

    def __enter__(self):
        """Open driver"""
        self.lib.driver_open()
        self._opened = True
        return self.lib

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Close driver"""
        if self._opened:
            self.lib.driver_close()
            self._opened = False
        return False



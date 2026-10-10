# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Versioned block-first disk layout shared by GDS-style backends."""

# Standard
from dataclasses import asdict, dataclass
from hashlib import sha256
from math import prod
from typing import Any, Mapping, Optional, Sequence
import json
import struct
import zlib

# Third Party
import torch

BLOCK_FIRST_ALIGNMENT = 4096
BLOCK_FIRST_HEADER_SIZE = 4096
BLOCK_FIRST_LAYOUT = "block_first_v1"
BLOCK_FIRST_VERSION = 1

_HEADER_MAGIC = b"LMCBFV1\0"
_HEADER_PREFIX = struct.Struct("<8sII")


class BlockFirstFormatError(ValueError):
    """Raised when block-first metadata or tensor coordinates are invalid."""


def align_up(size: int, alignment: int = BLOCK_FIRST_ALIGNMENT) -> int:
    """Round ``size`` up to ``alignment`` bytes."""
    if size < 0:
        raise BlockFirstFormatError(f"size must be non-negative, got {size}")
    if alignment <= 0:
        raise BlockFirstFormatError(f"alignment must be positive, got {alignment}")
    return (size + alignment - 1) // alignment * alignment


def _dtype_name(dtype: torch.dtype) -> str:
    name = str(dtype)
    if not name.startswith("torch."):
        raise BlockFirstFormatError(f"Unsupported dtype representation: {dtype}")
    return name.removeprefix("torch.")


def _parse_dtype(name: str) -> torch.dtype:
    dtype = getattr(torch, name, None)
    if not isinstance(dtype, torch.dtype):
        raise BlockFirstFormatError(f"Unsupported dtype name: {name}")
    return dtype


def block_first_shape(
    logical_shape: Sequence[int], block_size: int
) -> tuple[int, int, int, int, int]:
    """Return ``[CB, NL, 2, BS, D]`` for a ``[2, NL, T, D]`` tensor."""
    shape = tuple(int(dim) for dim in logical_shape)
    if len(shape) != 4 or shape[0] != 2:
        raise BlockFirstFormatError(
            "logical KV shape must be [2, NL, T, D], " f"got {shape}"
        )
    if any(dim <= 0 for dim in shape):
        raise BlockFirstFormatError(f"logical KV dimensions must be positive: {shape}")
    if block_size <= 0:
        raise BlockFirstFormatError(f"block_size must be positive, got {block_size}")
    _, num_layers, num_tokens, hidden_dim = shape
    if num_tokens % block_size != 0:
        raise BlockFirstFormatError(
            f"token count {num_tokens} is not divisible by block size {block_size}"
        )
    return (
        num_tokens // block_size,
        num_layers,
        2,
        block_size,
        hidden_dim,
    )


def logical_to_block_first(
    logical: torch.Tensor, block_size: int, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Convert ``[2, NL, T, D]`` into contiguous ``[CB, NL, 2, BS, D]``."""
    target_shape = block_first_shape(logical.shape, block_size)
    if not logical.is_contiguous():
        raise BlockFirstFormatError("logical KV tensor must be contiguous")
    if out is None:
        out = torch.empty(target_shape, dtype=logical.dtype, device=logical.device)
    if tuple(out.shape) != target_shape:
        raise BlockFirstFormatError(
            f"block-first output shape must be {target_shape}, got {tuple(out.shape)}"
        )
    if out.dtype != logical.dtype or out.device != logical.device:
        raise BlockFirstFormatError(
            "block-first output must have the same dtype and device as the input"
        )
    num_blocks, num_layers, _, _, hidden_dim = target_shape
    source = logical.view(2, num_layers, num_blocks, block_size, hidden_dim)
    out.copy_(source.permute(2, 1, 0, 3, 4))
    return out


def block_first_to_logical(
    physical: torch.Tensor, out: Optional[torch.Tensor] = None
) -> torch.Tensor:
    """Convert contiguous ``[CB, NL, 2, BS, D]`` into ``[2, NL, T, D]``."""
    if physical.ndim != 5 or physical.shape[2] != 2:
        raise BlockFirstFormatError(
            "block-first KV shape must be [CB, NL, 2, BS, D], "
            f"got {tuple(physical.shape)}"
        )
    if not physical.is_contiguous():
        raise BlockFirstFormatError("block-first KV tensor must be contiguous")
    num_blocks, num_layers, _, block_size, hidden_dim = physical.shape
    logical_shape = (2, num_layers, num_blocks * block_size, hidden_dim)
    if out is None:
        out = torch.empty(logical_shape, dtype=physical.dtype, device=physical.device)
    if tuple(out.shape) != logical_shape:
        raise BlockFirstFormatError(
            f"logical output shape must be {logical_shape}, got {tuple(out.shape)}"
        )
    if out.dtype != physical.dtype or out.device != physical.device:
        raise BlockFirstFormatError(
            "logical output must have the same dtype and device as the input"
        )
    target = out.view(2, num_layers, num_blocks, block_size, hidden_dim)
    target.copy_(physical.permute(2, 1, 0, 3, 4))
    return out


def make_layout_signature(
    *,
    model_name: str,
    world_size: int,
    worker_id: int,
    tensor_parallel_size: int,
    pipeline_parallel_size: int,
    num_layers: int,
    block_size: int,
    hidden_dim: int,
    dtype: torch.dtype,
) -> str:
    """Build a stable signature for fields that affect physical KV coordinates."""
    fields = {
        "block_size": block_size,
        "dtype": _dtype_name(dtype),
        "hidden_dim": hidden_dim,
        "layout": BLOCK_FIRST_LAYOUT,
        "model_name": model_name,
        "num_layers": num_layers,
        "pipeline_parallel_size": pipeline_parallel_size,
        "tensor_parallel_size": tensor_parallel_size,
        "worker_id": worker_id,
        "world_size": world_size,
    }
    canonical = json.dumps(fields, sort_keys=True, separators=(",", ":")).encode()
    return sha256(canonical).hexdigest()


@dataclass(frozen=True)
class BlockFirstDiskMetadata:
    """Self-describing metadata for one complete block-first payload."""

    layout: str
    version: int
    logical_shape: tuple[int, int, int, int]
    payload_shape: tuple[int, int, int, int, int]
    dtype: str
    valid_tokens: int
    num_layers: int
    block_size: int
    hidden_dim: int
    payload_nbytes: int
    io_nbytes: int
    model_name: str
    world_size: int
    worker_id: int
    tensor_parallel_size: int
    pipeline_parallel_size: int
    layout_signature: str

    @classmethod
    def create(
        cls,
        logical: torch.Tensor,
        *,
        block_size: int,
        model_name: str,
        world_size: int,
        worker_id: int,
        tensor_parallel_size: int = 1,
        pipeline_parallel_size: int = 1,
    ) -> "BlockFirstDiskMetadata":
        """Create and validate metadata for a complete logical KV chunk."""
        logical_shape = tuple(int(dim) for dim in logical.shape)
        payload_shape = block_first_shape(logical_shape, block_size)
        payload_nbytes = logical.numel() * logical.element_size()
        metadata = cls(
            layout=BLOCK_FIRST_LAYOUT,
            version=BLOCK_FIRST_VERSION,
            logical_shape=logical_shape,  # type: ignore[arg-type]
            payload_shape=payload_shape,
            dtype=_dtype_name(logical.dtype),
            valid_tokens=logical_shape[2],
            num_layers=logical_shape[1],
            block_size=block_size,
            hidden_dim=logical_shape[3],
            payload_nbytes=payload_nbytes,
            io_nbytes=align_up(payload_nbytes),
            model_name=model_name,
            world_size=world_size,
            worker_id=worker_id,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            layout_signature=make_layout_signature(
                model_name=model_name,
                world_size=world_size,
                worker_id=worker_id,
                tensor_parallel_size=tensor_parallel_size,
                pipeline_parallel_size=pipeline_parallel_size,
                num_layers=logical_shape[1],
                block_size=block_size,
                hidden_dim=logical_shape[3],
                dtype=logical.dtype,
            ),
        )
        metadata.validate()
        return metadata

    @property
    def torch_dtype(self) -> torch.dtype:
        """Return the torch dtype encoded in the descriptor."""
        return _parse_dtype(self.dtype)

    def validate(self) -> None:
        """Reject corrupt, partial, or internally inconsistent metadata."""
        if self.layout != BLOCK_FIRST_LAYOUT or self.version != BLOCK_FIRST_VERSION:
            raise BlockFirstFormatError(
                f"unsupported layout/version: {self.layout}/{self.version}"
            )
        expected_payload_shape = block_first_shape(
            self.logical_shape, self.block_size
        )
        if tuple(self.payload_shape) != expected_payload_shape:
            raise BlockFirstFormatError(
                "payload shape does not match logical coordinates: "
                f"{self.payload_shape} != {expected_payload_shape}"
            )
        if self.valid_tokens != self.logical_shape[2]:
            raise BlockFirstFormatError(
                "block_first_v1 only accepts complete chunks: "
                f"valid_tokens={self.valid_tokens}, tokens={self.logical_shape[2]}"
            )
        if self.num_layers != self.logical_shape[1]:
            raise BlockFirstFormatError("num_layers does not match logical shape")
        if self.hidden_dim != self.logical_shape[3]:
            raise BlockFirstFormatError("hidden_dim does not match logical shape")
        if min(
            self.world_size,
            self.tensor_parallel_size,
            self.pipeline_parallel_size,
        ) <= 0:
            raise BlockFirstFormatError("parallel sizes must be positive")
        if self.worker_id < 0 or self.worker_id >= self.world_size:
            raise BlockFirstFormatError(
                f"worker_id {self.worker_id} is outside world_size {self.world_size}"
            )
        dtype = self.torch_dtype
        expected_nbytes = prod(self.payload_shape) * torch.empty(
            (), dtype=dtype
        ).element_size()
        if self.payload_nbytes != expected_nbytes:
            raise BlockFirstFormatError(
                f"payload size {self.payload_nbytes} != expected {expected_nbytes}"
            )
        if self.io_nbytes != align_up(self.payload_nbytes):
            raise BlockFirstFormatError(
                f"I/O size {self.io_nbytes} is not the aligned payload size"
            )
        expected_signature = make_layout_signature(
            model_name=self.model_name,
            world_size=self.world_size,
            worker_id=self.worker_id,
            tensor_parallel_size=self.tensor_parallel_size,
            pipeline_parallel_size=self.pipeline_parallel_size,
            num_layers=self.num_layers,
            block_size=self.block_size,
            hidden_dim=self.hidden_dim,
            dtype=dtype,
        )
        if self.layout_signature != expected_signature:
            raise BlockFirstFormatError("layout signature mismatch")

    def validate_runtime(
        self,
        *,
        logical_shape: Sequence[int],
        dtype: torch.dtype,
        model_name: str,
        world_size: int,
        worker_id: int,
        tensor_parallel_size: int,
        pipeline_parallel_size: int,
        block_size: int,
    ) -> None:
        """Validate an on-disk descriptor against the active engine layout."""
        self.validate()
        expected_shape = tuple(int(dim) for dim in logical_shape)
        expected_signature = make_layout_signature(
            model_name=model_name,
            world_size=world_size,
            worker_id=worker_id,
            tensor_parallel_size=tensor_parallel_size,
            pipeline_parallel_size=pipeline_parallel_size,
            num_layers=expected_shape[1],
            block_size=block_size,
            hidden_dim=expected_shape[3],
            dtype=dtype,
        )
        if tuple(self.logical_shape) != expected_shape:
            raise BlockFirstFormatError(
                f"logical shape mismatch: {self.logical_shape} != {expected_shape}"
            )
        if self.torch_dtype != dtype:
            raise BlockFirstFormatError(
                f"dtype mismatch: {self.torch_dtype} != {dtype}"
            )
        if self.layout_signature != expected_signature:
            raise BlockFirstFormatError("runtime layout signature mismatch")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "BlockFirstDiskMetadata":
        """Construct metadata from decoded JSON while normalizing shapes."""
        data = dict(values)
        try:
            data["logical_shape"] = tuple(data["logical_shape"])
            data["payload_shape"] = tuple(data["payload_shape"])
            metadata = cls(**data)
        except (KeyError, TypeError, ValueError) as exc:
            raise BlockFirstFormatError(f"invalid block-first metadata: {exc}") from exc
        metadata.validate()
        return metadata


def pack_block_first_header(metadata: BlockFirstDiskMetadata) -> bytes:
    """Serialize metadata into one checksummed 4 KiB header."""
    metadata.validate()
    payload = json.dumps(
        asdict(metadata), sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    max_payload = BLOCK_FIRST_HEADER_SIZE - _HEADER_PREFIX.size
    if len(payload) > max_payload:
        raise BlockFirstFormatError(
            f"metadata is too large for the 4 KiB header: {len(payload)} bytes"
        )
    checksum = zlib.crc32(payload) & 0xFFFFFFFF
    prefix = _HEADER_PREFIX.pack(_HEADER_MAGIC, len(payload), checksum)
    padding_size = BLOCK_FIRST_HEADER_SIZE - len(prefix) - len(payload)
    return prefix + payload + bytes(padding_size)


def unpack_block_first_header(buffer: bytes) -> BlockFirstDiskMetadata:
    """Decode and validate one checksummed block-first header."""
    if len(buffer) < BLOCK_FIRST_HEADER_SIZE:
        raise BlockFirstFormatError(
            f"short block-first header: {len(buffer)}/{BLOCK_FIRST_HEADER_SIZE} bytes"
        )
    magic, payload_len, checksum = _HEADER_PREFIX.unpack_from(buffer)
    if magic != _HEADER_MAGIC:
        raise BlockFirstFormatError("block-first header magic mismatch")
    max_payload = BLOCK_FIRST_HEADER_SIZE - _HEADER_PREFIX.size
    if payload_len <= 0 or payload_len > max_payload:
        raise BlockFirstFormatError(f"invalid metadata length: {payload_len}")
    payload = buffer[_HEADER_PREFIX.size : _HEADER_PREFIX.size + payload_len]
    if zlib.crc32(payload) & 0xFFFFFFFF != checksum:
        raise BlockFirstFormatError("block-first header checksum mismatch")
    try:
        values = json.loads(payload.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise BlockFirstFormatError(f"invalid block-first JSON: {exc}") from exc
    if not isinstance(values, dict):
        raise BlockFirstFormatError("block-first metadata must be a JSON object")
    return BlockFirstDiskMetadata.from_mapping(values)

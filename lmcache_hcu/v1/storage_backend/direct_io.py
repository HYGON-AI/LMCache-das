# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Validated byte-range descriptors for synchronous direct storage I/O."""

# Standard
from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Sequence

# Third Party
import torch

# First Party
from lmcache_hcu.v1.storage_backend.block_first_disk import (
    BLOCK_FIRST_ALIGNMENT,
    BLOCK_FIRST_HEADER_SIZE,
)


class DirectIODirection(str, Enum):
    STORE = "store"
    RETRIEVE = "retrieve"


@dataclass(frozen=True)
class DirectIOBlockRun:
    """One range contiguous in both the file and the device allocation."""

    logical_block_start: int
    physical_block_start: int
    block_count: int


@dataclass(frozen=True)
class DirectIOLeadingEdge:
    """The uncached suffix of one physical block."""

    logical_block: int
    physical_block: int
    token_offset: int
    token_count: int


@dataclass(frozen=True)
class DirectIOAllocation:
    """One registered device allocation retained for synchronous direct I/O."""

    device: torch.device
    base: int
    nbytes: int
    lifetime_token: object = field(repr=False, compare=False)

    @property
    def identity(self) -> tuple[str, int, int]:
        return (str(self.device), self.base, self.nbytes)


@dataclass(frozen=True)
class DirectIOByteSegment:
    """One file range mapped into one registered allocation."""

    file_offset: int
    allocation_index: int
    allocation_offset: int
    nbytes: int


@dataclass(frozen=True)
class DirectIODescriptor:
    """A validated mapping from file blocks to one device allocation.

    ``lifetime_token`` is deliberately opaque to the storage backend.  In the
    in-process implementation it holds a strong reference to the vLLM tensor
    for the duration of the synchronous I/O call.
    """

    direction: DirectIODirection
    device: torch.device
    allocation_base: int
    allocation_nbytes: int
    engine_group_id: int
    block_nbytes: int
    block_size: int
    logical_block_start: int
    logical_block_count: int
    valid_token_start: int
    valid_token_count: int
    runs: tuple[DirectIOBlockRun, ...]
    leading_edge: Optional[DirectIOLeadingEdge] = None
    file_payload_offset: int = BLOCK_FIRST_HEADER_SIZE
    synchronous: bool = True
    lifetime_token: object = field(default=None, repr=False, compare=False)
    allocations: tuple[DirectIOAllocation, ...] = ()
    byte_segments: tuple[DirectIOByteSegment, ...] = ()
    physical_block_ids: tuple[int, ...] = ()

    @property
    def registration_identity(self) -> tuple[str, int, int]:
        return (str(self.device), self.allocation_base, self.allocation_nbytes)

    @property
    def registration_allocations(self) -> tuple[DirectIOAllocation, ...]:
        if self.allocations:
            return self.allocations
        return (
            DirectIOAllocation(
                self.device,
                self.allocation_base,
                self.allocation_nbytes,
                self.lifetime_token,
            ),
        )

    @property
    def transferred_nbytes(self) -> int:
        if self.byte_segments:
            return sum(segment.nbytes for segment in self.byte_segments)
        return self.logical_block_count * self.block_nbytes

    def validate(self, *, require_cuda: bool, require_alignment: bool) -> None:
        if require_cuda and self.device.type != "cuda":
            raise ValueError(
                f"direct I/O requires a CUDA allocation, got {self.device}"
            )
        if self.allocation_base <= 0 or self.allocation_nbytes <= 0:
            raise ValueError("direct I/O allocation address and size must be positive")
        if self.engine_group_id < 0:
            raise ValueError("engine_group_id must be non-negative")
        if self.block_nbytes <= 0 or self.block_size <= 0:
            raise ValueError("direct I/O block sizes must be positive")
        if self.logical_block_start < 0 or self.logical_block_count < 0:
            raise ValueError("direct I/O logical block range is invalid")
        if self.logical_block_count == 0 and self.leading_edge is None:
            raise ValueError("direct I/O descriptor has no transferable blocks")
        if self.lifetime_token is None:
            raise ValueError("direct I/O descriptor must hold its allocation lifetime")
        if not self.synchronous:
            raise ValueError("0.4.2 direct I/O only supports synchronous completion")
        if self.file_payload_offset != BLOCK_FIRST_HEADER_SIZE:
            raise ValueError("unexpected block_first_v1 payload offset")
        expected_logical = self.logical_block_start
        if self.byte_segments:
            self._validate_byte_segments(
                require_cuda=require_cuda,
                require_alignment=require_alignment,
            )
            return

        physical_block_capacity = self.allocation_nbytes // self.block_nbytes
        transferred_blocks = 0
        for run in self.runs:
            if (
                run.logical_block_start != expected_logical
                or run.physical_block_start < 0
                or run.block_count <= 0
            ):
                raise ValueError(
                    "direct I/O runs are invalid or not logically contiguous"
                )
            if run.physical_block_start + run.block_count > physical_block_capacity:
                raise ValueError("direct I/O run exceeds the registered allocation")
            expected_logical += run.block_count
            transferred_blocks += run.block_count
        if transferred_blocks != self.logical_block_count:
            raise ValueError("direct I/O runs do not cover the declared logical range")
        if bool(self.runs) != (self.logical_block_count > 0):
            raise ValueError("direct I/O runs do not match the declared block count")

        if self.leading_edge is None:
            if self.valid_token_start != self.logical_block_start * self.block_size:
                raise ValueError(
                    "valid token start does not match the logical block range"
                )
            expected_valid_tokens = self.logical_block_count * self.block_size
        else:
            edge = self.leading_edge
            if self.direction != DirectIODirection.RETRIEVE:
                raise ValueError("only direct retrieve can contain a leading edge")
            if (
                edge.logical_block < 0
                or edge.physical_block < 0
                or edge.physical_block >= physical_block_capacity
                or edge.token_offset <= 0
                or edge.token_offset >= self.block_size
                or edge.token_count != self.block_size - edge.token_offset
            ):
                raise ValueError("direct I/O leading edge is invalid")
            if self.logical_block_start != edge.logical_block + 1:
                raise ValueError("direct I/O full runs must follow the leading edge")
            if self.valid_token_start != (
                edge.logical_block * self.block_size + edge.token_offset
            ):
                raise ValueError("valid token start does not match the leading edge")
            expected_valid_tokens = (
                edge.token_count + self.logical_block_count * self.block_size
            )
        if self.valid_token_count != expected_valid_tokens:
            raise ValueError("valid token count does not match the direct I/O ranges")

        if require_alignment:
            for name, value in (
                ("allocation base", self.allocation_base),
                ("allocation size", self.allocation_nbytes),
                ("block bytes", self.block_nbytes),
                ("file payload offset", self.file_payload_offset),
            ):
                if value % BLOCK_FIRST_ALIGNMENT != 0:
                    raise ValueError(f"direct I/O {name} is not 4 KiB aligned: {value}")

    def _validate_byte_segments(
        self,
        *,
        require_cuda: bool,
        require_alignment: bool,
    ) -> None:
        if self.direction != DirectIODirection.RETRIEVE:
            raise ValueError("byte-segment direct I/O only supports retrieve")
        if self.runs or self.leading_edge is not None:
            raise ValueError(
                "byte-segment direct I/O cannot contain block runs or edges"
            )
        if not self.allocations or not self.byte_segments:
            raise ValueError(
                "byte-segment direct I/O requires allocations and segments"
            )
        if self.valid_token_start != self.logical_block_start * self.block_size:
            raise ValueError(
                "byte-segment direct I/O requires a block-aligned token start"
            )
        if self.valid_token_count != self.logical_block_count * self.block_size:
            raise ValueError("byte-segment direct I/O token count is inconsistent")

        for allocation in self.allocations:
            if require_cuda and allocation.device.type != "cuda":
                raise ValueError(
                    "direct I/O requires CUDA allocations, "
                    f"got {allocation.device}"
                )
            if allocation.device != self.device:
                raise ValueError("direct I/O allocations must use one device")
            if allocation.base <= 0 or allocation.nbytes <= 0:
                raise ValueError("direct I/O allocation range must be positive")
            if allocation.lifetime_token is None:
                raise ValueError("direct I/O allocation must retain its lifetime")
            if require_alignment and (
                allocation.base % BLOCK_FIRST_ALIGNMENT != 0
                or allocation.nbytes % BLOCK_FIRST_ALIGNMENT != 0
            ):
                raise ValueError("direct I/O allocation is not 4 KiB aligned")

        expected_file_offset = (
            self.file_payload_offset + self.logical_block_start * self.block_nbytes
        )
        for segment in sorted(self.byte_segments, key=lambda item: item.file_offset):
            if segment.file_offset != expected_file_offset:
                raise ValueError(
                    "direct I/O byte segments do not exactly cover the file"
                )
            if segment.allocation_index < 0 or segment.allocation_index >= len(
                self.allocations
            ):
                raise ValueError("direct I/O segment references an unknown allocation")
            allocation = self.allocations[segment.allocation_index]
            if (
                segment.allocation_offset < 0
                or segment.nbytes <= 0
                or segment.allocation_offset + segment.nbytes > allocation.nbytes
            ):
                raise ValueError("direct I/O segment exceeds its registered allocation")
            if require_alignment and any(
                value % BLOCK_FIRST_ALIGNMENT != 0
                for value in (
                    segment.file_offset,
                    segment.allocation_offset,
                    segment.nbytes,
                )
            ):
                raise ValueError("direct I/O byte segment is not 4 KiB aligned")
            expected_file_offset += segment.nbytes

        expected_end = self.file_payload_offset + (
            self.logical_block_start + self.logical_block_count
        ) * self.block_nbytes
        if expected_file_offset != expected_end:
            raise ValueError(
                "direct I/O byte segments do not cover the declared blocks"
            )


@dataclass(frozen=True)
class DirectIOPlan:
    descriptor: Optional[DirectIODescriptor]
    fallback_reason: Optional[str] = None


def build_block_first_direct_plans(
    kv_cache: torch.Tensor,
    slot_mapping: torch.Tensor,
    starts: Sequence[int],
    ends: Sequence[int],
    *,
    direction: DirectIODirection,
    block_size: int,
    skip_before: int = 0,
) -> list[DirectIOPlan]:
    """Translate token slots into validated block-first physical runs."""
    if kv_cache.ndim != 6 or kv_cache.shape[2] != 2:
        plan = DirectIOPlan(None, "KV allocation is not [NB, NL, 2, BS, NH, HS]")
        return [plan] * len(starts)
    if not kv_cache.is_contiguous():
        return [DirectIOPlan(None, "KV allocation is not contiguous")] * len(starts)
    if block_size <= 0 or kv_cache.shape[3] != block_size:
        return [DirectIOPlan(None, "KV block size mismatch")] * len(starts)
    if len(starts) != len(ends):
        raise ValueError("starts and ends must have the same length")
    if not starts:
        return []
    if min(starts) < 0 or any(
        end <= start for start, end in zip(starts, ends, strict=True)
    ):
        raise ValueError("direct I/O token ranges must be non-empty and non-negative")
    if max(ends) > slot_mapping.numel():
        raise ValueError("slot mapping is shorter than the requested token range")

    slot_mapping_cpu = slot_mapping.detach().to(device="cpu", dtype=torch.long)
    allocation_nbytes = kv_cache.numel() * kv_cache.element_size()
    block_nbytes = (
        kv_cache.shape[1]
        * kv_cache.shape[2]
        * kv_cache.shape[3]
        * kv_cache.shape[4]
        * kv_cache.shape[5]
        * kv_cache.element_size()
    )
    num_physical_blocks = kv_cache.shape[0]
    plans: list[DirectIOPlan] = []

    for start, end in zip(starts, ends, strict=True):
        chunk_tokens = end - start
        if chunk_tokens % block_size != 0:
            plans.append(DirectIOPlan(None, "chunk is not a whole number of blocks"))
            continue

        skip_tokens = 0
        if direction == DirectIODirection.RETRIEVE:
            skip_tokens = min(chunk_tokens, max(0, skip_before - start))
        if skip_tokens == chunk_tokens:
            plans.append(DirectIOPlan(None, "no uncached physical blocks remain"))
            continue

        leading_edge = None
        full_start_tokens = skip_tokens
        edge_offset = skip_tokens % block_size
        if edge_offset:
            edge_logical_block = skip_tokens // block_size
            edge_token_count = block_size - edge_offset
            edge_slots = slot_mapping_cpu[
                start + skip_tokens : start + skip_tokens + edge_token_count
            ]
            first_slot = int(edge_slots[0].item())
            physical_block = first_slot // block_size
            expected_edge_slots = physical_block * block_size + torch.arange(
                edge_offset, block_size, dtype=torch.long
            )
            if (
                first_slot < 0
                or first_slot % block_size != edge_offset
                or physical_block >= num_physical_blocks
                or not torch.equal(edge_slots, expected_edge_slots)
            ):
                plans.append(
                    DirectIOPlan(None, "leading edge does not map one physical block")
                )
                continue
            leading_edge = DirectIOLeadingEdge(
                logical_block=edge_logical_block,
                physical_block=physical_block,
                token_offset=edge_offset,
                token_count=edge_token_count,
            )
            full_start_tokens += edge_token_count

        transfer_start = start + full_start_tokens
        slots = slot_mapping_cpu[transfer_start:end]
        if slots.numel() % block_size != 0:
            plans.append(DirectIOPlan(None, "slot range is not block aligned"))
            continue
        slot_blocks = slots.reshape(-1, block_size)
        physical_blocks: list[int] = []
        expected_offsets = torch.arange(block_size, dtype=torch.long)
        invalid_reason = None
        for slot_block in slot_blocks:
            first_slot = int(slot_block[0].item())
            if first_slot < 0 or first_slot % block_size != 0:
                invalid_reason = "physical block does not start at slot offset zero"
                break
            physical_block = first_slot // block_size
            if physical_block >= num_physical_blocks:
                invalid_reason = (
                    "slot mapping references a block outside the allocation"
                )
                break
            expected_slots = physical_block * block_size + expected_offsets
            if not torch.equal(slot_block, expected_slots):
                invalid_reason = "tokens do not cover one complete physical block"
                break
            physical_blocks.append(physical_block)
        if invalid_reason is not None:
            plans.append(DirectIOPlan(None, invalid_reason))
            continue

        logical_start = full_start_tokens // block_size
        runs: list[DirectIOBlockRun] = []
        if physical_blocks:
            run_logical_start = logical_start
            run_physical_start = physical_blocks[0]
            run_count = 1
            for logical_offset, physical_block in enumerate(
                physical_blocks[1:], start=1
            ):
                if physical_block == run_physical_start + run_count:
                    run_count += 1
                    continue
                runs.append(
                    DirectIOBlockRun(
                        logical_block_start=run_logical_start,
                        physical_block_start=run_physical_start,
                        block_count=run_count,
                    )
                )
                run_logical_start = logical_start + logical_offset
                run_physical_start = physical_block
                run_count = 1
            runs.append(
                DirectIOBlockRun(
                    logical_block_start=run_logical_start,
                    physical_block_start=run_physical_start,
                    block_count=run_count,
                )
            )

        descriptor = DirectIODescriptor(
            direction=direction,
            device=kv_cache.device,
            allocation_base=kv_cache.data_ptr(),
            allocation_nbytes=allocation_nbytes,
            engine_group_id=0,
            block_nbytes=block_nbytes,
            block_size=block_size,
            logical_block_start=logical_start,
            logical_block_count=len(physical_blocks),
            valid_token_start=skip_tokens,
            valid_token_count=chunk_tokens - skip_tokens,
            runs=tuple(runs),
            leading_edge=leading_edge,
            lifetime_token=kv_cache,
            physical_block_ids=tuple(physical_blocks),
        )
        descriptor.validate(require_cuda=False, require_alignment=False)
        plans.append(DirectIOPlan(descriptor))

    return plans

def build_per_layer_direct_plans(
    kv_caches: Sequence[torch.Tensor],
    slot_mapping: torch.Tensor,
    starts: Sequence[int],
    ends: Sequence[int],
    *,
    direction: DirectIODirection,
    block_size: int,
    skip_before: int = 0,
) -> list[DirectIOPlan]:
    """Map block-first files into contiguous per-layer NHD allocations."""
    if direction != DirectIODirection.RETRIEVE:
        return [
            DirectIOPlan(None, "per-layer direct I/O only supports retrieve")
        ] * len(starts)
    if not kv_caches:
        return [DirectIOPlan(None, "vLLM per-layer KV cache is empty")] * len(starts)
    if len(starts) != len(ends):
        raise ValueError("starts and ends must have the same length")
    if not starts:
        return []
    if min(starts) < 0 or any(
        end <= start for start, end in zip(starts, ends, strict=True)
    ):
        raise ValueError("direct I/O token ranges must be non-empty and non-negative")
    if max(ends) > slot_mapping.numel():
        raise ValueError("slot mapping is shorter than the requested token range")

    first = kv_caches[0]
    expected_shape = tuple(first.shape)
    if first.ndim != 5 or first.shape[0] != 2:
        return [
            DirectIOPlan(None, "per-layer KV cache is not [2, NB, BS, NH, HS]")
        ] * len(starts)
    if first.shape[2] != block_size:
        return [DirectIOPlan(None, "KV block size mismatch")] * len(starts)
    if any(
        tensor.ndim != 5
        or tuple(tensor.shape) != expected_shape
        or tensor.dtype != first.dtype
        or tensor.device != first.device
        or not tensor.is_contiguous()
        for tensor in kv_caches
    ):
        return [
            DirectIOPlan(
                None,
                "per-layer KV allocations are non-uniform or non-contiguous",
            )
        ] * len(starts)

    num_blocks = first.shape[1]
    layer_block_nbytes = (
        first.shape[2]
        * first.shape[3]
        * first.shape[4]
        * first.element_size()
    )
    block_nbytes = len(kv_caches) * 2 * layer_block_nbytes
    slot_mapping_cpu = slot_mapping.detach().to(device="cpu", dtype=torch.long)

    allocations: list[DirectIOAllocation] = []
    allocation_indices: dict[tuple[str, int, int], int] = {}
    layer_allocations: list[tuple[int, int]] = []
    for tensor in kv_caches:
        storage = tensor.untyped_storage()
        storage_base = storage.data_ptr()
        storage_nbytes = storage.nbytes()
        identity = (str(tensor.device), storage_base, storage_nbytes)
        allocation_index = allocation_indices.get(identity)
        if allocation_index is None:
            allocation_index = len(allocations)
            allocation_indices[identity] = allocation_index
            allocations.append(
                DirectIOAllocation(
                    tensor.device,
                    storage_base,
                    storage_nbytes,
                    tensor,
                )
            )
        layer_allocations.append((allocation_index, tensor.data_ptr() - storage_base))

    plans: list[DirectIOPlan] = []
    expected_offsets = torch.arange(block_size, dtype=torch.long)
    for start, end in zip(starts, ends, strict=True):
        chunk_tokens = end - start
        if chunk_tokens % block_size != 0:
            plans.append(DirectIOPlan(None, "chunk is not a whole number of blocks"))
            continue
        skip_tokens = min(chunk_tokens, max(0, skip_before - start))
        if skip_tokens == chunk_tokens:
            plans.append(DirectIOPlan(None, "no uncached physical blocks remain"))
            continue
        if skip_tokens % block_size:
            plans.append(
                DirectIOPlan(
                    None,
                    "per-layer direct retrieve does not support a partial "
                    "leading block",
                )
            )
            continue

        slots = slot_mapping_cpu[start + skip_tokens : end]
        slot_blocks = slots.reshape(-1, block_size)
        physical_blocks: list[int] = []
        invalid_reason = None
        for slot_block in slot_blocks:
            first_slot = int(slot_block[0].item())
            if first_slot < 0 or first_slot % block_size != 0:
                invalid_reason = "physical block does not start at slot offset zero"
                break
            physical_block = first_slot // block_size
            if physical_block >= num_blocks:
                invalid_reason = (
                    "slot mapping references a block outside the allocation"
                )
                break
            if not torch.equal(
                slot_block,
                physical_block * block_size + expected_offsets,
            ):
                invalid_reason = "tokens do not cover one complete physical block"
                break
            physical_blocks.append(physical_block)
        if invalid_reason is not None:
            plans.append(DirectIOPlan(None, invalid_reason))
            continue

        logical_start = skip_tokens // block_size
        segments: list[DirectIOByteSegment] = []
        for block_offset, physical_block in enumerate(physical_blocks):
            logical_block = logical_start + block_offset
            for layer_index, (allocation_index, tensor_offset) in enumerate(
                layer_allocations
            ):
                for kv_index in range(2):
                    segments.append(
                        DirectIOByteSegment(
                            file_offset=(
                                BLOCK_FIRST_HEADER_SIZE
                                + logical_block * block_nbytes
                                + (layer_index * 2 + kv_index) * layer_block_nbytes
                            ),
                            allocation_index=allocation_index,
                            allocation_offset=(
                                tensor_offset
                                + (kv_index * num_blocks + physical_block)
                                * layer_block_nbytes
                            ),
                            nbytes=layer_block_nbytes,
                        )
                    )

        first_allocation = allocations[0]
        descriptor = DirectIODescriptor(
            direction=direction,
            device=first.device,
            allocation_base=first_allocation.base,
            allocation_nbytes=first_allocation.nbytes,
            engine_group_id=0,
            block_nbytes=block_nbytes,
            block_size=block_size,
            logical_block_start=logical_start,
            logical_block_count=len(physical_blocks),
            valid_token_start=skip_tokens,
            valid_token_count=chunk_tokens - skip_tokens,
            runs=(),
            lifetime_token=tuple(kv_caches),
            allocations=tuple(allocations),
            byte_segments=tuple(segments),
            physical_block_ids=tuple(physical_blocks),
        )
        descriptor.validate(require_cuda=False, require_alignment=False)
        plans.append(DirectIOPlan(descriptor))

    return plans


def build_asymmetric_per_layer_direct_plans(
    kv_caches: Sequence[tuple[torch.Tensor, torch.Tensor]],
    slot_mapping: torch.Tensor,
    starts: Sequence[int],
    ends: Sequence[int],
    *,
    direction: DirectIODirection,
    block_size: int,
    skip_before: int = 0,
) -> list[DirectIOPlan]:
    """Map block-first files into separate asymmetric HCU key/value caches.

    The supported physical layouts are ``K=[NB, NH, BS, HS]`` and
    ``V=[NB, NH, HS, BS]``.  The disk payload remains token-major, so every
    contiguous head vector becomes one byte segment.  A caller must still
    validate the resulting descriptor against its direct-I/O alignment rules.
    """
    if direction != DirectIODirection.RETRIEVE:
        return [
            DirectIOPlan(None, "asymmetric per-layer direct I/O only supports retrieve")
        ] * len(starts)
    if not kv_caches:
        return [DirectIOPlan(None, "asymmetric per-layer KV cache is empty")] * len(
            starts
        )
    if len(starts) != len(ends):
        raise ValueError("starts and ends must have the same length")
    if not starts:
        return []
    if min(starts) < 0 or any(
        end <= start for start, end in zip(starts, ends, strict=True)
    ):
        raise ValueError("direct I/O token ranges must be non-empty and non-negative")
    if max(ends) > slot_mapping.numel():
        raise ValueError("slot mapping is shorter than the requested token range")

    first_key, first_value = kv_caches[0]
    if (
        first_key.ndim != 4
        or first_value.ndim != 4
        or first_key.shape[2] != block_size
        or first_value.shape[3] != block_size
    ):
        return [
            DirectIOPlan(
                None,
                "asymmetric KV cache is not K=[NB, NH, BS, HS], V=[NB, NH, HS, BS]",
            )
        ] * len(starts)

    num_blocks, num_heads, _, head_size = first_key.shape
    expected_value_shape = (num_blocks, num_heads, head_size, block_size)
    if first_value.shape != expected_value_shape:
        return [DirectIOPlan(None, "asymmetric K/V cache shapes do not match")] * len(
            starts
        )
    if (
        first_key.dtype != first_value.dtype
        or first_key.device != first_value.device
        or first_key.stride(3) != 1
        or first_value.stride(2) != 1
    ):
        return [
            DirectIOPlan(
                None,
                "asymmetric K/V cache head vectors are not contiguous in storage",
            )
        ] * len(starts)

    for key_cache, value_cache in kv_caches:
        if (
            key_cache.ndim != 4
            or value_cache.ndim != 4
            or tuple(key_cache.shape) != tuple(first_key.shape)
            or tuple(value_cache.shape) != expected_value_shape
            or key_cache.dtype != first_key.dtype
            or value_cache.dtype != first_key.dtype
            or key_cache.device != first_key.device
            or value_cache.device != first_key.device
            or key_cache.stride(3) != 1
            or value_cache.stride(2) != 1
        ):
            return [
                DirectIOPlan(
                    None,
                    "asymmetric per-layer K/V allocations are non-uniform or "
                    "unsupported",
                )
            ] * len(starts)

    element_size = first_key.element_size()
    head_nbytes = head_size * element_size
    layer_block_nbytes = block_size * num_heads * head_nbytes
    block_nbytes = len(kv_caches) * 2 * layer_block_nbytes
    slot_mapping_cpu = slot_mapping.detach().to(device="cpu", dtype=torch.long)
    expected_offsets = torch.arange(block_size, dtype=torch.long)

    allocations: list[DirectIOAllocation] = []
    allocation_indices: dict[tuple[str, int, int], int] = {}

    def add_allocation(tensor: torch.Tensor) -> tuple[int, int]:
        storage = tensor.untyped_storage()
        base = storage.data_ptr()
        nbytes = storage.nbytes()
        identity = (str(tensor.device), base, nbytes)
        index = allocation_indices.get(identity)
        if index is None:
            index = len(allocations)
            allocation_indices[identity] = index
            allocations.append(DirectIOAllocation(tensor.device, base, nbytes, tensor))
        return index, tensor.data_ptr() - base

    layer_allocations = [
        (add_allocation(key_cache), add_allocation(value_cache))
        for key_cache, value_cache in kv_caches
    ]

    def append_segment(
        segments: list[DirectIOByteSegment],
        file_offset: int,
        allocation_index: int,
        allocation_offset: int,
    ) -> None:
        if segments:
            previous = segments[-1]
            if (
                previous.allocation_index == allocation_index
                and previous.file_offset + previous.nbytes == file_offset
                and previous.allocation_offset + previous.nbytes == allocation_offset
            ):
                segments[-1] = DirectIOByteSegment(
                    previous.file_offset,
                    previous.allocation_index,
                    previous.allocation_offset,
                    previous.nbytes + head_nbytes,
                )
                return
        segments.append(
            DirectIOByteSegment(
                file_offset, allocation_index, allocation_offset, head_nbytes
            )
        )

    plans: list[DirectIOPlan] = []
    for start, end in zip(starts, ends, strict=True):
        chunk_tokens = end - start
        if chunk_tokens % block_size:
            plans.append(DirectIOPlan(None, "chunk is not a whole number of blocks"))
            continue
        skip_tokens = min(chunk_tokens, max(0, skip_before - start))
        if skip_tokens == chunk_tokens:
            plans.append(DirectIOPlan(None, "no uncached physical blocks remain"))
            continue
        if skip_tokens % block_size:
            plans.append(
                DirectIOPlan(
                    None,
                    "asymmetric direct retrieve does not support a partial "
                    "leading block",
                )
            )
            continue

        slots = slot_mapping_cpu[start + skip_tokens : end]
        slot_blocks = slots.reshape(-1, block_size)
        physical_blocks: list[int] = []
        invalid_reason = None
        for slot_block in slot_blocks:
            first_slot = int(slot_block[0].item())
            physical_block = first_slot // block_size
            if (
                first_slot < 0
                or first_slot % block_size != 0
                or physical_block >= num_blocks
                or not torch.equal(
                    slot_block, physical_block * block_size + expected_offsets
                )
            ):
                invalid_reason = "tokens do not cover one complete physical block"
                break
            physical_blocks.append(physical_block)
        if invalid_reason is not None:
            plans.append(DirectIOPlan(None, invalid_reason))
            continue

        logical_start = skip_tokens // block_size
        segments: list[DirectIOByteSegment] = []
        for block_offset, physical_block in enumerate(physical_blocks):
            logical_block = logical_start + block_offset
            for layer_index, layer_allocation in enumerate(layer_allocations):
                (key_index, key_base), (value_index, value_base) = layer_allocation
                key_cache, value_cache = kv_caches[layer_index]
                cache_mappings = (
                    (key_cache, key_index, key_base, 2),
                    (value_cache, value_index, value_base, 3),
                )
                for kv_index, cache_mapping in enumerate(cache_mappings):
                    cache, allocation_index, tensor_base, token_dim = cache_mapping
                    region_offset = (
                        BLOCK_FIRST_HEADER_SIZE
                        + logical_block * block_nbytes
                        + (layer_index * 2 + kv_index) * layer_block_nbytes
                    )
                    for token_offset in range(block_size):
                        for head_index in range(num_heads):
                            element_offset = (
                                physical_block * cache.stride(0)
                                + head_index * cache.stride(1)
                                + token_offset * cache.stride(token_dim)
                            )
                            append_segment(
                                segments,
                                region_offset
                                + (token_offset * num_heads + head_index)
                                * head_nbytes,
                                allocation_index,
                                tensor_base + element_offset * element_size,
                            )

        first_allocation = allocations[0]
        descriptor = DirectIODescriptor(
            direction=direction,
            device=first_key.device,
            allocation_base=first_allocation.base,
            allocation_nbytes=first_allocation.nbytes,
            engine_group_id=0,
            block_nbytes=block_nbytes,
            block_size=block_size,
            logical_block_start=logical_start,
            logical_block_count=len(physical_blocks),
            valid_token_start=skip_tokens,
            valid_token_count=chunk_tokens - skip_tokens,
            runs=(),
            lifetime_token=tuple(kv_caches),
            allocations=tuple(allocations),
            byte_segments=tuple(segments),
            physical_block_ids=tuple(physical_blocks),
        )
        try:
            descriptor.validate(require_cuda=False, require_alignment=True)
        except ValueError as exc:
            plans.append(DirectIOPlan(None, str(exc)))
            continue
        plans.append(DirectIOPlan(descriptor))
    return plans

# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""HCU test utilities used to adapt reusable upstream LMCache tests."""
from __future__ import annotations

# Third Party
from lmcache_tests.v1.utils import *  # noqa: F403
import torch

# First Party
from lmcache_hcu.v1.hcu_connector import VLLMPagedMemHCUConnectorV2


def create_hcu_connector(hidden_dim, num_layers):
    return VLLMPagedMemHCUConnectorV2(
        hidden_dim,
        num_layers,
        use_gpu=True,
        chunk_size=256,
        dtype=torch.bfloat16,
        device="cuda",
    )


def generate_kv_cache_paged_list_tensors(
    num_blocks,
    device,
    block_size=16,
    dtype=torch.bfloat16,
    use_mla=False,
    num_layers=32,
    num_heads=8,
    head_size=128,
):
    """Generate HCU flash-attn-pa KV cache layout for reusable upstream tests."""
    if use_mla:
        return [
            torch.rand(
                (num_blocks, block_size, head_size),
                dtype=dtype,
                device=device,
            )
            for _ in range(num_layers)
        ]

    key_shape = (num_blocks, num_heads, block_size, head_size)
    value_shape = (num_blocks, num_heads, head_size, block_size)
    ret = []
    for _ in range(num_layers):
        if dtype == torch.uint8:
            key = torch.randint(0, 256, key_shape, dtype=dtype, device=device)
            value = torch.randint(0, 256, value_shape, dtype=dtype, device=device)
        else:
            key = torch.rand(key_shape, dtype=dtype, device=device)
            value = torch.rand(value_shape, dtype=dtype, device=device)
        ret.append((key, value))
    return ret


def check_paged_kv_cache_equal(
    left,
    right,
    slot_mapping,
    num_heads=8,
    head_size=128,
):
    """Check HCU flash-attn-pa paged KV cache equality at slot_mapping."""
    slot_mapping_cpu = slot_mapping.detach().cpu()

    for left_layer, right_layer in zip(left, right, strict=False):
        left_key, left_value = left_layer
        right_key, right_value = right_layer

        block_size = left_key.shape[2]
        block_indices = slot_mapping_cpu // block_size
        block_offsets = slot_mapping_cpu % block_size

        left_key_tokens = left_key[block_indices, :, block_offsets, :]
        right_key_tokens = right_key[block_indices, :, block_offsets, :]
        left_value_tokens = left_value[block_indices, :, :, block_offsets]
        right_value_tokens = right_value[block_indices, :, :, block_offsets]

        assert left_key_tokens.shape == right_key_tokens.shape
        assert left_value_tokens.shape == right_value_tokens.shape
        assert (left_key_tokens == right_key_tokens).all()
        assert (left_value_tokens == right_value_tokens).all()

# SPDX-License-Identifier: Apache-2.0
"""Tests LMCache-HCU token database hash behavior.

The HCU runtime patch replaces TokenDatabase._hash_tokens so tensor and list
inputs are normalized consistently and extra_keys become part of the hash key.
Upstream tests do not fully match this patched hash contract, so these local
cases cover the HCU-specific behavior directly.
"""
from __future__ import annotations

# Standard
from types import SimpleNamespace

# Third Party
import pytest
import torch

# First Party
import lmcache_hcu  # noqa: F401  # Importing applies LMCache-HCU runtime patches.
from lmcache.v1.token_database import TokenDatabase


def _hash_with_database(tokens, prefix_hash=None, extra_keys=None):
    """Call the patched hash helper with a minimal database-like object."""
    database = SimpleNamespace(hash_func=hash)
    return TokenDatabase._hash_tokens(database, tokens, prefix_hash, extra_keys)


def test_hash_tokens_accepts_tensor_tokens():
    """Tensor tokens should be converted to a CPU Python tuple before hashing."""
    tokens = torch.tensor([1, 2, 3], dtype=torch.int64)

    result = _hash_with_database(tokens)

    assert result == hash(("", (1, 2, 3), ()))


def test_hash_tokens_accepts_list_tokens():
    """List tokens should be converted to a tuple before hashing."""
    result = _hash_with_database([1, 2, 3])

    assert result == hash(("", (1, 2, 3), ()))


def test_hash_tokens_includes_prefix_hash_and_extra_keys():
    """The HCU patch should include prefix_hash and extra_keys in the hash input."""
    result = _hash_with_database(
        [10, 20, 30], prefix_hash=12345, extra_keys=["model-a", "adapter-b"]
    )

    assert result == hash((12345, (10, 20, 30), ("model-a", "adapter-b")))


def test_hash_tokens_uses_empty_defaults_for_missing_prefix_and_extra_keys():
    """Missing prefix_hash and extra_keys should map to HCU empty defaults."""
    result = _hash_with_database([7, 8])

    assert result == hash(("", (7, 8), ()))


def test_hash_tokens_rejects_unsupported_token_type():
    """Unsupported token containers should raise ValueError."""
    with pytest.raises(ValueError, match="Unsupported tokens type"):
        _hash_with_database((1, 2, 3))

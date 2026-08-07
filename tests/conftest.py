# SPDX-License-Identifier: Apache-2.0
"""Pytest bootstrap for LMCache-HCU tests that reuse upstream LMCache fixtures."""
from __future__ import annotations

# Third Party
import pytest

# Local
from .bootstrap import TEST_ALIAS, prepare_environment


try:
    prepare_environment()
except Exception as exc:
    pytest.exit(f"LMCache-HCU test bootstrap failed: {exc}", returncode=1)


def patch_lmcache_test_utils() -> None:
    """Patch upstream reusable tests to use HCU flash-attn-pa test utilities."""
    try:
        # Third Party
        import lmcache_tests.v1.utils as upstream_utils

        # Local
        from .lmcache_hcu.v1 import utils as hcu_utils
    except Exception as exc:
        pytest.exit(f"LMCache-HCU test utils patch failed: {exc}", returncode=1)

    upstream_utils.create_gpu_connector = hcu_utils.create_hcu_connector
    upstream_utils.generate_kv_cache_paged_list_tensors = (
        hcu_utils.generate_kv_cache_paged_list_tensors
    )
    upstream_utils.check_paged_kv_cache_equal = hcu_utils.check_paged_kv_cache_equal

    try:
        # Third Party
        import lmcache_tests.v1.test_cache_engine as upstream_cache_engine
    except ImportError:
        return

    upstream_cache_engine.create_gpu_connector = hcu_utils.create_hcu_connector
    upstream_cache_engine.generate_kv_cache_paged_list_tensors = (
        hcu_utils.generate_kv_cache_paged_list_tensors
    )
    upstream_cache_engine.check_paged_kv_cache_equal = (
        hcu_utils.check_paged_kv_cache_equal
    )


patch_lmcache_test_utils()


pytest_plugins = [f"{TEST_ALIAS}.conftest"]

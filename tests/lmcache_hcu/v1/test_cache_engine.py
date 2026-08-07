# SPDX-License-Identifier: Apache-2.0
"""Reuses upstream LMCache cache engine tests against LMCache-HCU runtime patches.

These tests exercise the patched cache engine through upstream store/retrieve,
prefetch, eviction, builder, and lifecycle scenarios instead of only checking
that monkey-patched symbols were rebound.
"""
# ruff: noqa: F401
from __future__ import annotations

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party
import pytest
import torch

# First Party
import lmcache_hcu  # noqa: F401  # Importing applies LMCache-HCU runtime patches.
from lmcache.v1.cache_engine import LMCacheEngine, LMCacheEngineBuilder

# Third Party
from lmcache_tests.v1.test_cache_engine import (
    test_builder,
    test_builder_destroy,
    test_builder_destroy_multiple_instances,
    test_force_store_wait,
    test_paged_hierarchy_retrieve,
    test_paged_mem_leak,
    test_paged_mixed_retrieve,
    test_paged_prefetch_retrieve,
    test_paged_retrieve_after_eviction,
    test_paged_same_retrieve_store,
    test_paged_store_kv_tensors_mask,
    test_paged_store_offset,
)
from lmcache_tests.v1.test_cache_engine import (
    test_paged_retrieve_prefix as original_paged_retrieve_prefix,
)


@pytest.mark.parametrize("chunk_size", [128, 256])
@pytest.mark.parametrize("backend", ["cpu", "local_disk", "remote"])
@pytest.mark.parametrize("save_unfull_chunk", [False, True])
@pytest.mark.parametrize("lmserver_v1_process", ["cpu"], indirect=True)
def test_paged_retrieve_prefix_hcu(
    chunk_size, backend, save_unfull_chunk, lmserver_v1_process, autorelease_v1
):
    """Run upstream prefix retrieval coverage with the HCU runtime patches active."""
    original_paged_retrieve_prefix(
        chunk_size, backend, save_unfull_chunk, lmserver_v1_process, autorelease_v1
    )


class _RetrieveStats:
    """Minimal retrieve stats object used by HCU retrieve tests."""

    def profile_process_tokens(self):
        return nullcontext()

    def profile_to_gpu(self):
        return nullcontext()

    def profile_broadcast(self):
        return nullcontext()

    def time_to_retrieve(self):
        return 0.001


class _StatsMonitor:
    """Records HCU-specific retrieve statistics callbacks."""

    def __init__(self):
        self.retrieve_stats = _RetrieveStats()
        self.on_retrieve_request = MagicMock(return_value=self.retrieve_stats)
        self.on_disk_to_memory_request = MagicMock(return_value="d2m")
        self.on_disk_to_memory_finished = MagicMock()
        self.on_memory_to_hbm_request = MagicMock(return_value="m2h")
        self.on_memory_to_hbm_finished = MagicMock()
        self.on_retrieve_finished = MagicMock()


def _make_engine(**overrides):
    """Create a lightweight object that can execute the patched retrieve method."""
    stats_monitor = _StatsMonitor()
    engine = SimpleNamespace(
        is_healthy=MagicMock(return_value=True),
        gpu_connector=SimpleNamespace(batched_to_gpu=MagicMock()),
        _log_kvcache_for_check=MagicMock(),
        stats_monitor=stats_monitor,
        _is_passive=MagicMock(return_value=False),
        async_loading=False,
        _async_process_tokens_internal=MagicMock(),
        _process_tokens_internal=MagicMock(return_value=([], 0)),
        save_only_first_rank=False,
        remove_after_retrieve=False,
        storage_manager=SimpleNamespace(remove=MagicMock()),
        save_only_first_rank_buffer=None,
    )
    engine.__dict__.update(overrides)
    return engine


def test_retrieve_returns_empty_mask_when_engine_is_unhealthy():
    """The HCU retrieve override should skip work when LMCache is unhealthy."""
    engine = _make_engine(is_healthy=MagicMock(return_value=False))
    tokens = torch.tensor([1, 2, 3])

    ret_mask = LMCacheEngine.retrieve(engine, tokens)

    assert ret_mask.tolist() == [False, False, False]
    engine._process_tokens_internal.assert_not_called()
    engine.gpu_connector.batched_to_gpu.assert_not_called()


def test_retrieve_sync_path_records_hcu_stats_and_loads_to_gpu():
    """The sync retrieve path should call HCU D2M/M2H stats and batched_to_gpu."""
    memory_obj = SimpleNamespace(ref_count_down=MagicMock())
    engine = _make_engine()
    engine._process_tokens_internal.return_value = (
        [("cache-key", memory_obj, 0, 4)],
        4096,
    )
    tokens = torch.tensor([1, 2, 3, 4])

    ret_mask = LMCacheEngine.retrieve(engine, tokens, req_id="req-0")

    assert ret_mask.dtype == torch.bool
    engine._process_tokens_internal.assert_called_once()
    engine._async_process_tokens_internal.assert_not_called()
    engine.stats_monitor.on_disk_to_memory_request.assert_called_once_with(4)
    engine.stats_monitor.on_disk_to_memory_finished.assert_called_once()
    engine.stats_monitor.on_memory_to_hbm_request.assert_called_once_with(4)
    engine.stats_monitor.on_memory_to_hbm_finished.assert_called_once()
    engine.gpu_connector.batched_to_gpu.assert_called_once_with(
        [memory_obj], [0], [4], req_id="req-0"
    )
    memory_obj.ref_count_down.assert_called_once_with()


def test_retrieve_async_path_uses_async_token_processing():
    """The HCU retrieve override should call async token processing when enabled."""
    engine = _make_engine(async_loading=True)
    engine._async_process_tokens_internal.return_value = ([], 0)
    tokens = torch.tensor([1, 2])

    LMCacheEngine.retrieve(engine, tokens, req_id="req-async")

    engine._async_process_tokens_internal.assert_called_once()
    engine._process_tokens_internal.assert_not_called()


def test_retrieve_remove_after_retrieve_deletes_storage_key():
    """remove_after_retrieve should remove retrieved keys from storage manager."""
    memory_obj = SimpleNamespace(ref_count_down=MagicMock())
    engine = _make_engine(remove_after_retrieve=True)
    engine._process_tokens_internal.return_value = (
        [("cache-key", memory_obj, 0, 2)],
        2048,
    )
    tokens = torch.tensor([1, 2])

    LMCacheEngine.retrieve(engine, tokens, req_id="req-remove")

    engine.storage_manager.remove.assert_called_once_with("cache-key")
    memory_obj.ref_count_down.assert_called_once_with()


@pytest.mark.no_shared_allocator
def test_create_memory_allocator_uses_hyfile_for_xds_on_hcu(monkeypatch):
    """XDS on a non-CUDA torch build should use HyFileMemoryAllocator."""
    created = {}

    class FakeHyFileMemoryAllocator:
        def __init__(self, size, use_mla=False):
            created["size"] = size
            created["use_mla"] = use_mla

    monkeypatch.setattr(torch.version, "cuda", None, raising=False)
    monkeypatch.setattr(
        "lmcache_hcu.v1.cache_engine.HyFileMemoryAllocator",
        FakeHyFileMemoryAllocator,
    )
    config = SimpleNamespace(
        extra_config={},
        gds_path=None,
        xds_path="/mnt/volume1",
        xds_buffer_size=6144,
        max_local_cpu_size=10,
        get_extra_config_value=MagicMock(return_value=False),
    )
    metadata = SimpleNamespace(use_mla=True, is_first_rank=MagicMock(return_value=False))

    allocator = LMCacheEngineBuilder._Create_memory_allocator(config, metadata)

    assert isinstance(allocator, FakeHyFileMemoryAllocator)
    assert created == {"size": 6144 * 1024**2, "use_mla": True}


# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Tests LMCache-HCU XDS backend behavior.

Covers the duplicate-store guard in ``XdsBackend.submit_put_task``. A store whose
key is already resident must not reserve bytes and must not run eviction.

Before that guard existed, every duplicate store still reserved
``required_size`` (``current_cache_size += required_size``) and ran
``_eviction()``, which popped one *live* entry from ``hot_cache`` to make room --
while the matching ``insert_key()`` took its ``key in self.hot_cache`` branch and
published nothing. The net effect was -1 resident entry per duplicate store with
the byte counter still pinned at ``max_xds_size``, so the capacity check never
noticed. Under sustained eviction the index drained to empty, ``_eviction()``
then found no candidates and every later store was dropped. Measured on a
256G-capped xds run: 194,343 ``No eviction candidates found`` warnings and a
1.6G on-disk cache.
"""
from __future__ import annotations

# Standard
from contextlib import nullcontext
from types import SimpleNamespace
from unittest.mock import MagicMock

# Third Party

# First Party
import lmcache_hcu  # noqa: F401  # Importing applies LMCache-HCU runtime patches.
from lmcache_hcu.v1.storage_backend import xds_backend as xds_module
from lmcache_hcu.v1.storage_backend.xds_backend import XdsBackend

REQUIRED_SIZE = 1 << 20
INITIAL_CACHE_SIZE = 4096

class _MemoryObj:
    """Minimal stand-in for the memory object handed to submit_put_task."""

    def __init__(self):
        self.tensor = object()
        self.ref_count_up = MagicMock()
        self.ref_count_down = MagicMock()

def _make_backend(hot_cache):
    """Build a fake ``self`` exposing only what submit_put_task touches."""
    return SimpleNamespace(
        memory_allocator=object(),  # not None -> not the inert MLA path
        cpu_store_enabled=True,
        hot_lock=nullcontext(),
        hot_cache=hot_cache,
        current_cache_size=INITIAL_CACHE_SIZE,
        eviction=True,
        exists_in_put_tasks=MagicMock(return_value=False),
        _required_store_size=MagicMock(return_value=REQUIRED_SIZE),
        _eviction=MagicMock(return_value=True),
        _finish_io_profile=MagicMock(),
        _async_save_bytes_to_storage=object(),
        cache_policy=SimpleNamespace(
            update_on_hit=MagicMock(),
            update_on_put=MagicMock(),
        ),
        xds_worker=SimpleNamespace(
            insert_put_task=MagicMock(),
            remove_put_task=MagicMock(),
            submit_task=MagicMock(),
        ),
        loop=object(),
    )

def test_submit_put_task_skips_resident_key(monkeypatch):
    """A store for an already-resident key must not reserve or evict."""
    monkeypatch.setattr(
        xds_module.asyncio, "run_coroutine_threadsafe", lambda *a, **k: None
    )
    entry = SimpleNamespace(disk_metadata=object())
    backend = _make_backend({"k": entry})
    memory_obj = _MemoryObj()

    result = XdsBackend.submit_put_task(backend, "k", memory_obj)

    assert result is None
    # The reservation is what used to drain the index: it must not happen.
    assert backend.current_cache_size == INITIAL_CACHE_SIZE
    backend._eviction.assert_not_called()
    # No file is rewritten, so no put task is registered either.
    backend.xds_worker.insert_put_task.assert_not_called()
    backend.xds_worker.submit_task.assert_not_called()
    # A hit still refreshes recency.
    backend.cache_policy.update_on_hit.assert_called_once()

def test_submit_put_task_stores_nonresident_key(monkeypatch):
    """The guard must not block a genuine store."""
    monkeypatch.setattr(
        xds_module.asyncio, "run_coroutine_threadsafe", lambda *a, **k: None
    )
    backend = _make_backend({})
    memory_obj = _MemoryObj()

    result = XdsBackend.submit_put_task(backend, "k", memory_obj)

    assert result is None
    backend.xds_worker.insert_put_task.assert_called_once_with("k")
    # The reservation itself happens inside _eviction (mocked here), so only
    # assert that the store reached it with the right size.
    backend._eviction.assert_called_once_with(REQUIRED_SIZE)
    backend.cache_policy.update_on_put.assert_called_once_with("k")

def test_submit_put_task_does_not_skip_entry_without_disk_metadata(monkeypatch):
    """Only entries the on-disk reader can serve count as duplicates.

    An index entry without ``disk_metadata`` is not loadable, so the store has
    to go through.
    """
    monkeypatch.setattr(
        xds_module.asyncio, "run_coroutine_threadsafe", lambda *a, **k: None
    )
    backend = _make_backend({"k": SimpleNamespace(disk_metadata=None)})
    memory_obj = _MemoryObj()

    XdsBackend.submit_put_task(backend, "k", memory_obj)

    backend.xds_worker.insert_put_task.assert_called_once_with("k")
    backend._eviction.assert_called_once_with(REQUIRED_SIZE)

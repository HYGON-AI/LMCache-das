# SPDX-License-Identifier: Apache-2.0
"""Reuse upstream LocalDiskBackend tests with LMCache-HCU runtime patches active."""
from __future__ import annotations

# Third Party
import pytest

# First Party
import lmcache_hcu  # noqa: F401  # Importing applies LMCache-HCU runtime patches.
import lmcache.v1.storage_backend.local_disk_backend as local_disk_backend_mod


class _SynchronousExecutor:
    """Small executor replacement for upstream init-only local disk tests."""

    def __init__(self, loop, max_workers=4):
        self.loop = loop
        self.max_workers = max_workers

    async def submit_job(self, task, *args, **kwargs):
        kwargs.pop("priority", None)
        return task(*args, **kwargs)

    def shutdown(self, wait=True):
        pass


@pytest.fixture(autouse=True)
def disable_local_disk_async_workers(monkeypatch):
    """Avoid leaking upstream LocalDiskWorker coroutines in reuse tests."""
    monkeypatch.setattr(
        local_disk_backend_mod,
        "AsyncPQThreadPoolExecutor",
        _SynchronousExecutor,
    )

# Reuse upstream tests.
from lmcache_tests.v1.storage_backend.test_local_disk_backend import (  # noqa: F401
    TestLocalDiskBackend,
    async_loop,
    local_cpu_backend,
    local_disk_backend,
    temp_disk_path,
)

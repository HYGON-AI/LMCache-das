# SPDX-License-Identifier: Apache-2.0
"""Tests LMCache-HCU storage backend factory behavior.

When xds_path is configured, the HCU runtime patch augments upstream
CreateStorageBackends with an XdsBackend while preserving existing upstream
backends.
"""
from __future__ import annotations

# Standard
from types import SimpleNamespace
import sys

# Third Party
import pytest

# First Party
import lmcache_hcu

# Local
from tests.lmcache_hcu.utils import ensure_module


_PATCHED_MODULE_NAMES = (
    "lmcache.v1.storage_backend",
    "lmcache.v1.storage_backend.gds_backend",
    "lmcache.v1.cache_engine",
    "lmcache.v1.storage_backend.storage_manager",
)


@pytest.fixture(autouse=True)
def restore_storage_patch_targets():
    """Restore modules that _patch_storage_backends mutates globally."""
    snapshots = {}
    for module_name in _PATCHED_MODULE_NAMES:
        module = sys.modules.get(module_name)
        if module is not None:
            snapshots[module_name] = dict(vars(module))

    yield

    for module_name in _PATCHED_MODULE_NAMES:
        if module_name not in snapshots:
            sys.modules.pop(module_name, None)
            continue
        module = sys.modules.get(module_name)
        if module is None:
            continue
        vars(module).clear()
        vars(module).update(snapshots[module_name])

def test_create_storage_backends_adds_xds_backend(monkeypatch):
    """CreateStorageBackends should append XdsBackend when xds_path is configured."""
    storage_backend = ensure_module(monkeypatch, "lmcache.v1.storage_backend")
    hcu_gds_backend = ensure_module(
        monkeypatch, "lmcache_hcu.v1.storage_backend.gds_backend"
    )
    hcu_xds_backend = ensure_module(
        monkeypatch, "lmcache_hcu.v1.storage_backend.xds_backend"
    )

    class GdsBackend:
        pass

    class XdsBackend:
        def __init__(self, config, metadata, loop, dst_device, local_cpu_backend):
            self.config = config
            self.metadata = metadata
            self.loop = loop
            self.dst_device = dst_device
            self.local_cpu_backend = local_cpu_backend

        def __str__(self):
            return "XdsBackend"

    local_cpu = SimpleNamespace(name="local-cpu")

    def create_storage_backends(config, metadata, loop, dst_device="cuda", lmcache_worker=None):
        return {"LocalCPUBackend": local_cpu}

    monkeypatch.setattr(hcu_gds_backend, "GdsBackend", GdsBackend, raising=False)
    monkeypatch.setattr(hcu_xds_backend, "XdsBackend", XdsBackend, raising=False)
    monkeypatch.setattr(storage_backend, "CreateStorageBackends", create_storage_backends)
    monkeypatch.setattr(storage_backend, "is_cuda_worker", lambda metadata: False)

    lmcache_hcu._patch_storage_backends()

    config = SimpleNamespace(xds_path="/mnt/volume1")
    metadata = SimpleNamespace()
    loop = object()
    backends = storage_backend.CreateStorageBackends(config, metadata, loop)

    assert "LocalCPUBackend" in backends
    assert "XdsBackend" in backends
    assert backends["XdsBackend"].config is config
    assert backends["XdsBackend"].metadata is metadata
    assert backends["XdsBackend"].loop is loop
    assert backends["XdsBackend"].dst_device == "cpu"
    assert backends["XdsBackend"].local_cpu_backend is local_cpu


def test_create_storage_backends_does_not_add_xds_without_xds_path(monkeypatch):
    """CreateStorageBackends should preserve upstream output when xds_path is absent."""
    storage_backend = ensure_module(monkeypatch, "lmcache.v1.storage_backend")
    hcu_gds_backend = ensure_module(
        monkeypatch, "lmcache_hcu.v1.storage_backend.gds_backend"
    )
    hcu_xds_backend = ensure_module(
        monkeypatch, "lmcache_hcu.v1.storage_backend.xds_backend"
    )

    class GdsBackend:
        pass

    class XdsBackend:
        def __init__(self, *args, **kwargs):
            raise AssertionError("XdsBackend should not be constructed")

    expected_backends = {"LocalCPUBackend": object()}

    def create_storage_backends(config, metadata, loop, dst_device="cuda", lmcache_worker=None):
        return expected_backends

    monkeypatch.setattr(hcu_gds_backend, "GdsBackend", GdsBackend, raising=False)
    monkeypatch.setattr(hcu_xds_backend, "XdsBackend", XdsBackend, raising=False)
    monkeypatch.setattr(storage_backend, "CreateStorageBackends", create_storage_backends)
    monkeypatch.setattr(storage_backend, "is_cuda_worker", lambda metadata: False)

    lmcache_hcu._patch_storage_backends()

    config = SimpleNamespace(xds_path=None)
    backends = storage_backend.CreateStorageBackends(config, SimpleNamespace(), object())

    assert backends is expected_backends
    assert "XdsBackend" not in backends

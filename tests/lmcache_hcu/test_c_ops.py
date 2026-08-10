# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Tests LMCache-HCU c_ops behavior with and without the Python proxy.

Without the runtime proxy, lmcache.c_ops resolves to the baseline module. With
the proxy installed, mem-kernel symbols resolve to lmcache_hcu.hcu_c_ops while
all other symbols fall back to the captured baseline lmcache.c_ops module.
"""
from __future__ import annotations

# Standard
import importlib
import sys
from types import ModuleType
from unittest.mock import MagicMock

# First Party
import lmcache_hcu

# Local
from tests.lmcache_hcu.utils import ensure_module


def _install_fake_hcu_c_ops(monkeypatch) -> ModuleType:
    """Install a fake HCU native c_ops module before importing the proxy."""
    lmcache_hcu_pkg = ensure_module(monkeypatch, "lmcache_hcu")
    hcu_c_ops = ModuleType("lmcache_hcu.hcu_c_ops")
    for name in (
        "multi_layer_kv_transfer",
        "multi_layer_kv_transfer_asymmetric",
        "multi_layer_kv_transfer_unilateral",
        "single_layer_kv_transfer",
        "single_layer_kv_transfer_sgl",
        "load_and_reshape_flash",
        "reshape_and_cache_back_flash",
        "lmcache_memcpy_async",
    ):
        setattr(hcu_c_ops, name, f"hcu::{name}")
    monkeypatch.setitem(sys.modules, "lmcache_hcu.hcu_c_ops", hcu_c_ops)
    monkeypatch.setattr(lmcache_hcu_pkg, "hcu_c_ops", hcu_c_ops, raising=False)
    return hcu_c_ops


def _fresh_proxy(monkeypatch):
    """Import lmcache_hcu.c_ops with fake native symbols in place."""
    _install_fake_hcu_c_ops(monkeypatch)
    sys.modules.pop("lmcache_hcu.c_ops", None)
    if "lmcache_hcu" in sys.modules:
        monkeypatch.delattr(sys.modules["lmcache_hcu"], "c_ops", raising=False)
    return importlib.import_module("lmcache_hcu.c_ops")


def _install_fake_baseline_c_ops(monkeypatch) -> ModuleType:
    """Install a fresh fake baseline lmcache.c_ops module."""
    lmcache = ensure_module(monkeypatch, "lmcache")
    baseline = ModuleType("lmcache.c_ops")
    monkeypatch.setitem(sys.modules, "lmcache.c_ops", baseline)
    monkeypatch.setattr(lmcache, "c_ops", baseline, raising=False)
    return baseline


def test_without_proxy_lmcache_c_ops_uses_baseline_module(monkeypatch):
    """Without sys.modules replacement, lmcache.c_ops should be the baseline module."""
    baseline = _install_fake_baseline_c_ops(monkeypatch)
    baseline.calculate_cdf = "baseline::calculate_cdf"
    baseline.multi_layer_kv_transfer = "baseline::multi_layer_kv_transfer"

    imported = importlib.import_module("lmcache.c_ops")

    assert imported is baseline
    assert imported.calculate_cdf == "baseline::calculate_cdf"
    assert imported.multi_layer_kv_transfer == "baseline::multi_layer_kv_transfer"


def test_with_proxy_mem_kernel_symbols_use_hcu_extension(monkeypatch):
    """With the proxy installed, mem-kernel symbols should come from HCU c_ops."""
    baseline = _install_fake_baseline_c_ops(monkeypatch)
    baseline.multi_layer_kv_transfer = "baseline::multi_layer_kv_transfer"
    proxy = _fresh_proxy(monkeypatch)
    proxy._set_base_c_ops(baseline)
    monkeypatch.setitem(sys.modules, "lmcache.c_ops", proxy)

    imported = importlib.import_module("lmcache.c_ops")

    assert imported is proxy
    assert imported.multi_layer_kv_transfer == "hcu::multi_layer_kv_transfer"
    assert imported.single_layer_kv_transfer == "hcu::single_layer_kv_transfer"


def test_with_proxy_non_mem_kernel_symbols_fall_back_to_baseline(monkeypatch):
    """With the proxy installed, non-mem-kernel symbols should use baseline c_ops."""
    baseline = _install_fake_baseline_c_ops(monkeypatch)
    baseline.calculate_cdf = "baseline::calculate_cdf"
    baseline.get_gpu_pci_bus_id = "baseline::get_gpu_pci_bus_id"
    proxy = _fresh_proxy(monkeypatch)
    proxy._set_base_c_ops(baseline)
    monkeypatch.setitem(sys.modules, "lmcache.c_ops", proxy)

    imported = importlib.import_module("lmcache.c_ops")

    assert imported.calculate_cdf == "baseline::calculate_cdf"
    assert imported.get_gpu_pci_bus_id == "baseline::get_gpu_pci_bus_id"


def test_proxy_dir_contains_hcu_and_baseline_symbols(monkeypatch):
    """dir(proxy) should expose eager HCU symbols plus captured baseline symbols."""
    baseline = _install_fake_baseline_c_ops(monkeypatch)
    baseline.calculate_cdf = "baseline::calculate_cdf"
    proxy = _fresh_proxy(monkeypatch)
    proxy._set_base_c_ops(baseline)

    names = dir(proxy)

    assert "multi_layer_kv_transfer" in names
    assert "lmcache_memcpy_async" in names
    assert "calculate_cdf" in names


def test_proxy_raises_for_baseline_symbol_when_base_module_missing(monkeypatch):
    """Without a captured baseline module, non-mem-kernel lookup should fail clearly."""
    proxy = _fresh_proxy(monkeypatch)
    proxy._set_base_c_ops(None)

    try:
        proxy.calculate_cdf
    except AttributeError as exc:
        assert "baseline lmcache.c_ops is not available" in str(exc)
    else:
        raise AssertionError(
            "Expected missing baseline symbol lookup to raise AttributeError"
        )


def test_patch_c_ops_installs_proxy_and_captures_baseline(monkeypatch):
    """_patch_c_ops should replace lmcache.c_ops with the proxy module."""
    baseline = _install_fake_baseline_c_ops(monkeypatch)
    baseline.calculate_cdf = "baseline::calculate_cdf"
    baseline.multi_layer_kv_transfer = "baseline::multi_layer_kv_transfer"
    proxy = _fresh_proxy(monkeypatch)
    proxy._set_base_c_ops = MagicMock(wraps=proxy._set_base_c_ops)
    monkeypatch.setattr(lmcache_hcu, "c_ops", proxy, raising=False)

    lmcache_hcu._patch_c_ops()

    assert sys.modules["lmcache.c_ops"] is proxy
    proxy._set_base_c_ops.assert_called_once_with(baseline)
    assert proxy.calculate_cdf == "baseline::calculate_cdf"
    assert proxy.multi_layer_kv_transfer == "hcu::multi_layer_kv_transfer"


def test_patch_c_ops_rebinds_loaded_lmc_ops_users(monkeypatch):
    """_patch_c_ops should update already imported modules that cache lmc_ops."""
    baseline = _install_fake_baseline_c_ops(monkeypatch)
    proxy = _fresh_proxy(monkeypatch)
    monkeypatch.setattr(lmcache_hcu, "c_ops", proxy, raising=False)
    gpu_connector = ensure_module(monkeypatch, "lmcache.v1.gpu_connector")
    server = ensure_module(monkeypatch, "lmcache.v1.multiprocess.server")
    gpu_connector.lmc_ops = baseline
    server.lmc_ops = baseline

    lmcache_hcu._patch_c_ops()

    assert gpu_connector.lmc_ops is proxy
    assert server.lmc_ops is proxy


def test_load_baseline_c_ops_returns_existing_non_proxy_module(monkeypatch):
    """_load_baseline_c_ops should reuse an existing baseline c_ops module."""
    baseline = _install_fake_baseline_c_ops(monkeypatch)

    assert lmcache_hcu._load_baseline_c_ops() is baseline
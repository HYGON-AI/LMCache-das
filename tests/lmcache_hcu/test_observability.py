# SPDX-License-Identifier: Apache-2.0
"""Tests LMCache-HCU observability metrics behavior.

The HCU observability patch extends upstream LMCache stats with disk-to-memory
and memory-to-HBM metrics used by XDS retrieval. These tests cover the HCU-only
metric collection and Prometheus logging paths without registering real metrics.
"""
from __future__ import annotations

# Standard
import sys
from types import SimpleNamespace

# First Party
import lmcache_hcu  # noqa: F401  # Importing applies LMCache-HCU runtime patches.
import lmcache.observability as patched_observability
from lmcache.observability import LMCStatsMonitor, PrometheusLogger


class _MetricValue:
    """Records set/inc/observe calls for a labeled metric instance."""

    def __init__(self):
        self.set_calls = []
        self.inc_calls = []
        self.observe_calls = []

    def set(self, value):
        self.set_calls.append(value)

    def inc(self, value):
        self.inc_calls.append(value)

    def observe(self, value):
        self.observe_calls.append(value)


class _Metric:
    """Small Prometheus metric stand-in that captures labels and updates."""

    created = []

    def __init__(self, name, documentation, labelnames, **kwargs):
        self.name = name
        self.documentation = documentation
        self.labelnames = list(labelnames)
        self.kwargs = kwargs
        self.values = []
        _Metric.created.append(self)

    def labels(self, **labels):
        value = _MetricValue()
        value.labels = labels
        self.values.append(value)
        return value


def _reset_monitor():
    """Return a fresh HCU stats monitor singleton."""
    LMCStatsMonitor.DestroyInstance()
    return LMCStatsMonitor.GetOrCreate()


def test_patch_observability_replaces_upstream_module():
    """Importing LMCache-HCU should expose patched observability via lmcache."""
    # First Party
    import lmcache

    assert patched_observability is sys.modules["lmcache.observability"]
    assert lmcache.observability is patched_observability
    assert hasattr(patched_observability.LMCStatsMonitor, "on_disk_to_memory_request")


def test_disk_to_memory_metrics_are_collected_and_cleared():
    """Disk-to-memory requests should produce interval counts, latency, and speed."""
    stats_monitor = _reset_monitor()

    request_id = stats_monitor.on_disk_to_memory_request(num_tokens=8)
    stats_monitor.on_disk_to_memory_finished(request_id, num_tokens=10)
    stats = stats_monitor.get_stats_and_clear()

    assert stats.interval_disk_to_memory_requests == 1
    assert stats.interval_disk_to_memory_tokens == 10
    assert len(stats.time_to_disk_to_memory) == 1
    assert stats.time_to_disk_to_memory[0] > 0
    assert len(stats.disk_to_memory_speed) == 1
    assert stats.disk_to_memory_speed[0] > 0
    assert stats_monitor.disk_to_memory_requests == {}

    cleared = stats_monitor.get_stats_and_clear()
    assert cleared.interval_disk_to_memory_requests == 0
    assert cleared.interval_disk_to_memory_tokens == 0
    assert cleared.time_to_disk_to_memory == []
    assert cleared.disk_to_memory_speed == []


def test_memory_to_hbm_metrics_are_collected_and_cleared():
    """Memory-to-HBM requests should produce interval counts, latency, and speed."""
    stats_monitor = _reset_monitor()

    request_id = stats_monitor.on_memory_to_hbm_request(num_tokens=16)
    stats_monitor.on_memory_to_hbm_finished(request_id, num_tokens=12)
    stats = stats_monitor.get_stats_and_clear()

    assert stats.interval_memory_to_hbm_requests == 1
    assert stats.interval_memory_to_hbm_tokens == 12
    assert len(stats.time_to_memory_to_hbm) == 1
    assert stats.time_to_memory_to_hbm[0] > 0
    assert len(stats.memory_to_hbm_speed) == 1
    assert stats.memory_to_hbm_speed[0] > 0
    assert stats_monitor.memory_to_hbm_requests == {}

    cleared = stats_monitor.get_stats_and_clear()
    assert cleared.interval_memory_to_hbm_requests == 0
    assert cleared.interval_memory_to_hbm_tokens == 0
    assert cleared.time_to_memory_to_hbm == []
    assert cleared.memory_to_hbm_speed == []


def test_unfinished_hcu_load_requests_survive_clear():
    """Unfinished HCU load requests should remain pending after stats are read."""
    stats_monitor = _reset_monitor()

    d2m_request_id = stats_monitor.on_disk_to_memory_request(num_tokens=4)
    m2h_request_id = stats_monitor.on_memory_to_hbm_request(num_tokens=5)
    stats = stats_monitor.get_stats_and_clear()

    assert stats.interval_disk_to_memory_requests == 1
    assert stats.interval_disk_to_memory_tokens == 4
    assert stats.time_to_disk_to_memory == []
    assert d2m_request_id in stats_monitor.disk_to_memory_requests
    assert stats.interval_memory_to_hbm_requests == 1
    assert stats.interval_memory_to_hbm_tokens == 5
    assert stats.time_to_memory_to_hbm == []
    assert m2h_request_id in stats_monitor.memory_to_hbm_requests


def test_thread_pool_status_is_reported_and_preserved_across_clear():
    """Thread pool gauges are realtime values and should be present in stats."""
    stats_monitor = _reset_monitor()

    stats_monitor.update_thread_pool_status(active_threads=3, queue_size=7)
    stats = stats_monitor.get_stats_and_clear()

    assert stats.thread_pool_active_threads == 3
    assert stats.thread_pool_queue_size == 7

    next_stats = stats_monitor.get_stats_and_clear()
    assert next_stats.thread_pool_active_threads == 3
    assert next_stats.thread_pool_queue_size == 7


def test_prometheus_logger_registers_and_logs_hcu_metrics(monkeypatch):
    """Prometheus logging should create and update HCU-specific counters/histograms."""
    _Metric.created = []
    monkeypatch.setattr(PrometheusLogger, "_gauge_cls", _Metric)
    monkeypatch.setattr(PrometheusLogger, "_counter_cls", _Metric)
    monkeypatch.setattr(PrometheusLogger, "_histogram_cls", _Metric)
    monkeypatch.setattr(
        PrometheusLogger, "_dynamic_metrics", lambda self, labelnames: None
    )

    metadata = SimpleNamespace(
        model_name="model-a",
        worker_id="worker-0",
        role="worker",
        served_model_name=None,
    )
    logger = PrometheusLogger(metadata)
    stats_monitor = _reset_monitor()
    d2m_request_id = stats_monitor.on_disk_to_memory_request(num_tokens=8)
    stats_monitor.on_disk_to_memory_finished(d2m_request_id)
    m2h_request_id = stats_monitor.on_memory_to_hbm_request(num_tokens=4)
    stats_monitor.on_memory_to_hbm_finished(m2h_request_id)
    stats = stats_monitor.get_stats_and_clear()

    logger.log_prometheus(stats)

    metrics = {metric.name: metric for metric in _Metric.created}
    assert metrics["lmcache:num_disk_to_memory_requests"].values[0].inc_calls == [1]
    assert metrics["lmcache:num_disk_to_memory_tokens"].values[0].inc_calls == [8]
    assert metrics["lmcache:time_to_disk_to_memory"].values[0].observe_calls[0] > 0
    assert metrics["lmcache:disk_to_memory_speed"].values[0].observe_calls[0] > 0
    assert metrics["lmcache:num_memory_to_hbm_requests"].values[0].inc_calls == [1]
    assert metrics["lmcache:num_memory_to_hbm_tokens"].values[0].inc_calls == [4]
    assert metrics["lmcache:time_to_memory_to_hbm"].values[0].observe_calls[0] > 0
    assert metrics["lmcache:memory_to_hbm_speed"].values[0].observe_calls[0] > 0

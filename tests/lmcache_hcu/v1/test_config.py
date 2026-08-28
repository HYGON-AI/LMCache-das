# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
"""Tests LMCache-HCU v1 configuration patch behavior.

The HCU runtime patch extends upstream LMCacheEngineConfig with XDS options and
changes the default local CPU capacity. These tests verify those patched fields
through the same public config constructors used by LMCache.
"""
from __future__ import annotations

# Third Party
import pytest
import yaml

# First Party
import lmcache_hcu  # noqa: F401  # Importing applies LMCache-HCU runtime patches.
import lmcache.v1.config as config_mod
from lmcache.v1.config import LMCacheEngineConfig


def test_xds_config_definitions_are_registered():
    """The HCU patch should register XDS fields in upstream config definitions."""
    definitions = config_mod._CONFIG_DEFINITIONS

    assert definitions["max_local_cpu_size"]["default"] == 10.0
    assert definitions["xds_path"]["default"] is None
    assert definitions["xds_buffer_size"]["default"] is None
    assert definitions["max_xds_size"]["default"] is None
    assert definitions["xds_path"]["env_converter"]("/mnt/xds") == "/mnt/xds"
    assert definitions["xds_buffer_size"]["env_converter"]("6144") == 6144
    assert definitions["max_xds_size"]["env_converter"]("1024") == 1024.0


def test_from_defaults_contains_hcu_xds_fields():
    """from_defaults should expose patched XDS fields and the HCU CPU default."""
    config = LMCacheEngineConfig.from_defaults()

    assert config.max_local_cpu_size == 10.0
    assert config.xds_path is None
    assert config.xds_buffer_size is None
    assert config.max_xds_size is None


def test_from_defaults_accepts_hcu_xds_values():
    """from_defaults should accept user-provided XDS configuration values."""
    config = LMCacheEngineConfig.from_defaults(
        xds_path="/mnt/volume1",
        xds_buffer_size=6144,
        max_xds_size=1024,
    )

    assert config.xds_path == "/mnt/volume1"
    assert config.xds_buffer_size == 6144
    assert config.max_xds_size == 1024.0
    assert config._user_set_keys >= {"xds_path", "xds_buffer_size", "max_xds_size"}


def test_from_env_parses_hcu_xds_values(monkeypatch):
    """from_env should parse XDS fields through the patched env converters."""
    monkeypatch.setenv("LMCACHE_XDS_PATH", "/mnt/volume1")
    monkeypatch.setenv("LMCACHE_XDS_BUFFER_SIZE", "6144")
    monkeypatch.setenv("LMCACHE_MAX_XDS_SIZE", "1024")
    monkeypatch.setenv("LMCACHE_MAX_LOCAL_CPU_SIZE", "12")

    config = LMCacheEngineConfig.from_env()

    assert config.xds_path == "/mnt/volume1"
    assert config.xds_buffer_size == 6144
    assert config.max_xds_size == 1024.0
    assert config.max_local_cpu_size == 12.0
    assert config._user_set_keys >= {
        "xds_path",
        "xds_buffer_size",
        "max_xds_size",
        "max_local_cpu_size",
    }


def test_from_file_parses_hcu_xds_values(tmp_path):
    """from_file should load HCU XDS fields from YAML config files."""
    config_path = tmp_path / "lmcache_config.yaml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "chunk_size": 256,
                "xds_path": "/mnt/volume1",
                "xds_buffer_size": 6144,
                "max_xds_size": 1024,
                "max_local_cpu_size": 12,
            }
        ),
        encoding="utf-8",
    )

    config = LMCacheEngineConfig.from_file(str(config_path))

    assert config.chunk_size == 256
    assert config.xds_path == "/mnt/volume1"
    assert config.xds_buffer_size == 6144
    assert config.max_xds_size == 1024.0
    assert config.max_local_cpu_size == 12.0


def test_from_dict_parses_hcu_xds_values():
    """from_dict should load HCU XDS fields from dictionary config input."""
    config = LMCacheEngineConfig.from_dict(
        {
            "xds_path": "/mnt/volume1",
            "xds_buffer_size": "6144",
            "max_xds_size": "1024",
        }
    )

    assert config.xds_path == "/mnt/volume1"
    assert config.xds_buffer_size == 6144
    assert config.max_xds_size == 1024.0


def test_config_class_keeps_upstream_helpers_after_recreation():
    """The recreated config class should keep upstream helper methods."""
    config = LMCacheEngineConfig.from_defaults(extra_config={"xds_io_threads": 8})

    assert config.get_extra_config_value("xds_io_threads", 1) == 8
    assert config.get_extra_config_value("missing", "default") == "default"
    assert callable(config.validate)
    assert callable(config.log_config)


def test_invalid_xds_numeric_env_falls_back_to_default(monkeypatch):
    """Invalid numeric XDS env values should follow upstream fallback behavior."""
    monkeypatch.setenv("LMCACHE_XDS_BUFFER_SIZE", "invalid")

    config = LMCacheEngineConfig.from_env()

    assert config.xds_buffer_size is None

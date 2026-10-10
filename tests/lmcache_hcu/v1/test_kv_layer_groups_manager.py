# SPDX-License-Identifier: Apache-2.0
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""Tests HCU KV layer group behavior with local functional cases.

These tests mirror the upstream KVLayerGroupsManager coverage, but call the
HCU-patched API (no use_mla argument) and cover the HCU custom flash-attn
skip and HCU tuple KV shape synthesis.
"""
from __future__ import annotations

# Standard
import os

# Third Party
import pytest
import torch

# First Party
import lmcache_hcu  # noqa: F401  # Importing applies LMCache-HCU runtime patches.
from lmcache.v1.kv_layer_groups import KVLayerGroupInfo, KVLayerGroupsManager


def _set_custom_flash_attn(monkeypatch, enabled: bool) -> None:
    """Set the HCU custom flash-attn environment flag."""
    if enabled:
        monkeypatch.setenv("VLLM_HCU_USE_CUSTOM_FLASH_ATTN", "true")
    else:
        monkeypatch.delenv("VLLM_HCU_USE_CUSTOM_FLASH_ATTN", raising=False)


class TestKVLayerGroupsManager:
    """Local functional tests for the HCU-patched KVLayerGroupsManager."""

    def test_build_kv_layer_groups_empty(self, monkeypatch):
        """The HCU patch should keep an empty manager unchanged for empty KV caches."""
        _set_custom_flash_attn(monkeypatch, False)
        manager = KVLayerGroupsManager()

        manager.build_kv_layer_groups({})

        assert manager.kv_layer_groups == []

    def test_build_kv_layer_groups_skips_when_already_built(self, monkeypatch):
        """The HCU patch should not rebuild when kv_layer_groups is already non-empty."""
        _set_custom_flash_attn(monkeypatch, False)
        manager = KVLayerGroupsManager()
        existing_group = KVLayerGroupInfo(
            ["existing_layer"], [7], torch.Size((2, 1, 1, 1, 1)), torch.float32
        )
        manager.kv_layer_groups = [existing_group]
        kv_caches = {"layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16)}

        manager.build_kv_layer_groups(kv_caches)

        assert manager.kv_layer_groups == [existing_group]

    def test_build_kv_layer_groups_custom_flash_attn_skips_building(self, monkeypatch):
        """When VLLM_HCU_USE_CUSTOM_FLASH_ATTN is true, building should be skipped."""
        _set_custom_flash_attn(monkeypatch, True)
        manager = KVLayerGroupsManager()
        kv_caches = {
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_1": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
        }

        manager.build_kv_layer_groups(kv_caches)

        assert manager.kv_layer_groups == []

    def test_build_kv_layer_groups_single_layer(self, monkeypatch):
        """A single MHA layer should create one group with the original 5D shape."""
        _set_custom_flash_attn(monkeypatch, False)
        manager = KVLayerGroupsManager()
        kv_caches = {"layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16)}

        manager.build_kv_layer_groups(kv_caches)

        assert len(manager.kv_layer_groups) == 1
        group = manager.kv_layer_groups[0]
        assert isinstance(group, KVLayerGroupInfo)
        assert group.layer_names == ["layer_0"]
        assert group.layer_indices == [0]
        assert group.shape == torch.Size((2, 32, 256, 8, 64))
        assert group.dtype == torch.float16

    def test_build_kv_layer_groups_multiple_layers_same_shape(self, monkeypatch):
        """Layers with the same shape and dtype should share one group."""
        _set_custom_flash_attn(monkeypatch, False)
        manager = KVLayerGroupsManager()
        kv_caches = {
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_1": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_2": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
        }

        manager.build_kv_layer_groups(kv_caches)

        assert len(manager.kv_layer_groups) == 1
        group = manager.kv_layer_groups[0]
        assert group.layer_names == ["layer_0", "layer_1", "layer_2"]
        assert group.layer_indices == [0, 1, 2]
        assert group.shape == torch.Size((2, 32, 256, 8, 64))
        assert group.dtype == torch.float16

    def test_build_kv_layer_groups_different_shapes(self, monkeypatch):
        """Layers with different shapes should be split into separate groups."""
        _set_custom_flash_attn(monkeypatch, False)
        manager = KVLayerGroupsManager()
        kv_caches = {
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_1": torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),
            "layer_2": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
        }

        manager.build_kv_layer_groups(kv_caches)
        manager.kv_layer_groups.sort(key=lambda group: group.layer_indices[0])

        assert len(manager.kv_layer_groups) == 2
        group1, group2 = manager.kv_layer_groups
        assert group1.layer_names == ["layer_0", "layer_2"]
        assert group1.layer_indices == [0, 2]
        assert group1.shape == torch.Size((2, 32, 256, 8, 64))
        assert group2.layer_names == ["layer_1"]
        assert group2.layer_indices == [1]
        assert group2.shape == torch.Size((2, 32, 256, 16, 64))

    def test_build_kv_layer_groups_different_dtypes(self, monkeypatch):
        """Layers with different dtypes should be split into separate groups."""
        _set_custom_flash_attn(monkeypatch, False)
        manager = KVLayerGroupsManager()
        kv_caches = {
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_1": torch.randn(2, 32, 256, 8, 64, dtype=torch.float32),
            "layer_2": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
        }

        manager.build_kv_layer_groups(kv_caches)
        manager.kv_layer_groups.sort(key=lambda group: group.layer_indices[0])

        assert len(manager.kv_layer_groups) == 2
        group1, group2 = manager.kv_layer_groups
        assert group1.layer_names == ["layer_0", "layer_2"]
        assert group1.dtype == torch.float16
        assert group2.layer_names == ["layer_1"]
        assert group2.dtype == torch.float32

    def test_build_kv_layer_groups_preserves_first_layer_order(self, monkeypatch):
        """Group ordering should follow the first layer index in each group."""
        _set_custom_flash_attn(monkeypatch, False)
        manager = KVLayerGroupsManager()
        kv_caches = {
            "layer_2": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_0": torch.randn(2, 32, 256, 8, 64, dtype=torch.float16),
            "layer_1": torch.randn(2, 32, 256, 16, 64, dtype=torch.float16),
        }

        manager.build_kv_layer_groups(kv_caches)

        assert len(manager.kv_layer_groups) == 2
        assert manager.kv_layer_groups[0].layer_names == ["layer_2", "layer_0"]
        assert manager.kv_layer_groups[0].layer_indices == [0, 1]
        assert manager.kv_layer_groups[1].layer_names == ["layer_1"]
        assert manager.kv_layer_groups[1].layer_indices == [2]

    def test_build_kv_layer_groups_hcu_tuple_kv_synthesizes_canonical_shape(
        self, monkeypatch
    ):
        """HCU vLLM (K,V) tuple layers should be grouped by synthesized 5D shape."""
        _set_custom_flash_attn(monkeypatch, False)
        manager = KVLayerGroupsManager()
        nb, nh, bs, hs = 32, 8, 256, 64
        k0 = torch.randn(nb, nh, bs, hs, dtype=torch.float16)
        v0 = torch.randn(nb, hs, bs, nh, dtype=torch.float16)
        k1 = torch.randn(nb, nh, bs, hs, dtype=torch.float16)
        v1 = torch.randn(nb, hs, bs, nh, dtype=torch.float16)
        kv_caches = {
            "layer_0": (k0, v0),
            "layer_1": (k1, v1),
        }

        manager.build_kv_layer_groups(kv_caches)

        assert len(manager.kv_layer_groups) == 1
        group = manager.kv_layer_groups[0]
        assert group.layer_names == ["layer_0", "layer_1"]
        assert group.shape == torch.Size((nb, 2, bs, nh, hs))
        assert group.dtype == torch.float16

    def test_hidden_dim_size_mha_and_mla(self, monkeypatch):
        """The 0.4.2 hidden_dim_size property should support MHA and MLA layouts."""
        mha = KVLayerGroupInfo(
            ["layer_0"], [0], torch.Size((2, 32, 256, 8, 64)), torch.float16
        )
        mla = KVLayerGroupInfo(
            ["layer_0"], [0], torch.Size((32, 256, 512)), torch.float16
        )
        assert mha.hidden_dim_size == 8 * 64
        assert mla.hidden_dim_size == 512

    def test_hidden_dim_size_rejects_invalid_shape(self, monkeypatch):
        """hidden_dim_size should reject shapes that are neither 5D nor 3D."""
        group = KVLayerGroupInfo(
            ["layer_0"], [0], torch.Size((32, 256, 8, 64)), torch.float16
        )

        with pytest.raises(ValueError, match="Invalid shape"):
            group.hidden_dim_size

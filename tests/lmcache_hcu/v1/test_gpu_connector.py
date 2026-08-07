# SPDX-License-Identifier: Apache-2.0
"""Tests LMCache-HCU GPU connector metadata and shape behavior.

The HCU GPU connector computes connector dimensions from LMCache metadata and
uses different memory-object shapes for MHA and MLA layouts. These tests avoid
running HCU kernels and focus on the Python-side behavior that configures them.
"""
from __future__ import annotations

# Standard
import sys
from types import ModuleType, SimpleNamespace

# Third Party
import torch

# Ensure vllm.envs exists before importing LMCache-HCU so the runtime patch can apply.
if "vllm" not in sys.modules:
    sys.modules["vllm"] = ModuleType("vllm")
if "vllm.envs" not in sys.modules:
    envs = ModuleType("vllm.envs")
    envs.VLLM_USE_FLASH_ATTN_PA = False
    sys.modules["vllm.envs"] = envs
    sys.modules["vllm"].envs = envs

# First Party
import lmcache_hcu
from lmcache.v1.gpu_connector import VLLMPagedMemGPUConnectorV2

lmcache_hcu._patch_vllm_paged_mem_gpu_connector_v2()


class _FakeStream:
    """Small torch.cuda.Stream replacement for CPU-only unit tests."""

    def synchronize(self):
        pass


def test_paged_connector_from_metadata_sets_dimensions(monkeypatch):
    """from_metadata should derive layers and hidden size from metadata.kv_shape."""
    monkeypatch.setattr(torch.cuda, "Stream", lambda: _FakeStream())
    metadata = SimpleNamespace(
        kv_shape=(4, 2, 256, 8, 64),
        kv_dtype=torch.float16,
        use_mla=False,
    )

    connector = VLLMPagedMemGPUConnectorV2.from_metadata(metadata, use_gpu=False)

    assert connector.num_layers == 4
    assert connector.hidden_dim_size == 8 * 64
    assert connector.use_mla is False
    assert connector.gpu_buffer is None
    assert connector.get_shape(128) == torch.Size([2, 4, 128, 512])


def test_paged_connector_from_metadata_preserves_mla_flag(monkeypatch):
    """from_metadata should pass metadata.use_mla into the HCU connector."""
    monkeypatch.setattr(torch.cuda, "Stream", lambda: _FakeStream())
    metadata = SimpleNamespace(
        kv_shape=(3, 1, 256, 1, 512),
        kv_dtype=torch.float16,
        use_mla=True,
    )

    connector = VLLMPagedMemGPUConnectorV2.from_metadata(metadata, use_gpu=False)

    assert connector.num_layers == 3
    assert connector.hidden_dim_size == 512
    assert connector.use_mla is True
    assert connector.get_shape(128) == torch.Size([1, 3, 128, 512])


def test_paged_connector_reads_flash_attention_pa_env(monkeypatch):
    """The HCU connector should accept VLLM_USE_FLASH_ATTN_PA during construction."""
    monkeypatch.setattr(torch.cuda, "Stream", lambda: _FakeStream())
    monkeypatch.setattr(sys.modules["vllm.envs"], "VLLM_USE_FLASH_ATTN_PA", True)

    connector = VLLMPagedMemGPUConnectorV2(
        hidden_dim_size=512,
        num_layers=2,
        use_gpu=False,
        use_mla=False,
    )

    assert sys.modules["vllm.envs"].VLLM_USE_FLASH_ATTN_PA is True
    assert connector.use_mla is False


def test_paged_connector_get_shape_uses_kv_size_for_mha_and_mla(monkeypatch):
    """get_shape should use kv_size=2 for MHA and kv_size=1 for MLA."""
    monkeypatch.setattr(torch.cuda, "Stream", lambda: _FakeStream())
    mha = VLLMPagedMemGPUConnectorV2(
        hidden_dim_size=512,
        num_layers=2,
        use_gpu=False,
        use_mla=False,
    )
    mla = VLLMPagedMemGPUConnectorV2(
        hidden_dim_size=512,
        num_layers=2,
        use_gpu=False,
        use_mla=True,
    )

    assert mha.get_shape(64) == torch.Size([2, 2, 64, 512])
    assert mla.get_shape(64) == torch.Size([1, 2, 64, 512])

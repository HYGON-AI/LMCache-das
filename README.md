This repository is based on the following fixed upstream baseline:
Upstream project: LMCache
Upstream repository: GitHub - LMCache/LMCache: LMCache is a KV cache management layer for LLM inference.
Upstream branch: main
Upstream tag: v0.3.13
Upstream commit: fc031d471a566edb6d49a86c9116cc23cfb04111
Upstream license: Apache-2.0
HCU adaptations, modifications, and original contributions by Hygon Information Technology Co., Ltd. are licensed under the Apache License, Version 2.0.
Modified by Hygon Information Technology Co., Ltd.
Original copyright notices and license terms from the upstream LMCache project are retained. See LICENSE and Third-Party Notices for details.


# LMCache-HCU

LMCache-HCU is an overlay/patch package for adapting upstream LMCache to Hygon HCU / ROCm / HIP environments.

## Compatibility Matrix

Please ensure your environment matches the versions below.


| LMCache-hcu | LMCache | vLLM Version |
| :--- | :--- | :--- |
| **v0.3.13** | **v0.3.13** | **v0.15.0** |


## Design

There are two patch layers:

1. Runtime monkey patches in `lmcache_hcu/__init__.py`
   - Triggered by `import lmcache_hcu`.
   - Redirects `lmcache.c_ops` to the proxy module `lmcache_hcu.c_ops`.
   - There are also some other function patches, such as `hash_tokens`.
   - Adds HCU-specific config fields to `lmcache.v1.config._CONFIG_DEFINITIONS` and recreates `LMCacheEngineConfig`.
   - Installs HCU observability metrics and Prometheus logging extensions.
   - Replaces the upstream paged GPU connector with the HCU implementation.
   - Patches cache engine retrieve/cleanup/allocator behavior for HCU and XDS.
   - Patches KV layer grouping, metadata shape helpers, token hashing, parent allocator semantics, storage backend factory behavior, local disk backend methods, storage manager touch-cache handling, and cache-controller messages.
   - Patches small helpers such as `transfer_utils.get_correct_device` and NUMA auto-detection cleanup.

2. Source injection patches in `lmcache_hcu/integration/patch/`
   - Modeled after `LMCache-das/lmcache_hcu/integration/patch`.
   - Locates installed modules with `importlib.util.find_spec`.
   - Creates timestamped `.bak.<timestamp>` backups before modifying files.
   - Uses marker comments for idempotency.
   - Keeps source modifications intentionally tiny: inject `import lmcache_hcu` at stable trigger points.

## Important Files

```text
lmcache_hcu/
  |-- csrc/
  |   |-- pybind_hip.cpp
  |   |-- mem_kernels.cu
  |   `-- mem_kernels.cuh
  |-- lmcache_hcu/
  |   |-- integration/
  |   |   |-- patch/
  |   |   |   |-- lmcache/
  |   |   |   |   |-- __init__.py
  |   |   |   |   `-- runtime_import_patch.py
  |   |   |   |-- vllm/
  |   |   |   |   |-- __init__.py
  |   |   |   |   `-- connector_import_patch.py
  |   |   |   |-- __init__.py
  |   |   |   |-- apply_patch.py
  |   |   |   `-- base_patcher.py
  |   |   `-- __init__.py
  |   |-- v1/
  |   |   |-- storage_backend/
  |   |   |   |-- serde/
  |   |   |   |   `-- __init__.py
  |   |   |   |-- __init__.py
  |   |   |   |-- gds_backend.py
  |   |   |   |-- local_disk_backend.py
  |   |   |   |-- xds_backend.py
  |   |   |   `-- xds_metadata.py
  |   |   |-- __init__.py
  |   |   |-- cache_engine.py
  |   |   |-- gpu_connector.py
  |   |   |-- hipfille.py
  |   |   `-- memory_management.py
  |   |-- __init__.py
  |   |-- _build_info.py
  |   |-- _version.py
  |   |-- c_ops.py
  |   |-- config.py
  |   |-- env.py
  |   `-- observability.py
  |-- examples/
  |   `-- README.md
  |-- LICENSE
  |-- MANIFEST.in
  |-- pyproject.toml
  |-- README.md
  `-- setup.py
```

## Build

Typical editable install on a HCU/ROCm/DTK machine:

first install the baseline LMCache:

```bash
git clone -b v0.3.13 https://github.com/LMCache/LMCache.git
cd LMCache
PYTORCH_ROCM_ARCH="{your_rocm_arch}" \
TORCH_DONT_CHECK_COMPILER_ABI=1 \
CXX=hipcc \
BUILD_WITH_HIP=1 \
python3 -m pip install -e . --no-build-isolation

```

Then install LMCache-hcu:
```bash
cd /path/to/lmcache-das
PYTORCH_ROCM_ARCH="{your_rocm_arch}" \
TORCH_DONT_CHECK_COMPILER_ABI=1 \
CXX=hipcc \
BUILD_WITH_HIP=1 \
# Optional:
HCU_ARCH=gfx906  # Platform-specific value; use rocminfo | grep -i "name:" to query it. gfx906 is only an example.
python3 -m pip install --no-build-isolation -e .
```

## Usage

We introduce a dynamic KVConnector via LMCacheHcuConnectorV1Dynamic in LMCache-das.

Example usage of dynamic connector from LMCache hcu:

```bash
    vllm serve /models/Qwen/Qwen3-30B \
    --tp 8
    --port 20013 \
    --trust-remote-code \
    --disable-log-requests \
    --kv-transfer-config '{"kv_connector":"LMCacheHcuConnectorV1Dynamic","kv_role":"kv_both", "kv_connector_module_path":"lmcache_hcu.integration.vllm.lmcache_hcu_connector_v1"}'
```

See [`examples/README.md`](examples/README.md) for a vLLM + XDS example.

## Uninstall

```bash
pip uninstall lmcache-hcu
```

## Upgrade

Upgrade by uninstalling the old LMCache-HCU package first, then installing the new version.

Recommended flow:

```bash

# 1. Remove the old package.
pip uninstall lmcache-hcu

# 2. Install the new package version.
cd /path/to/lmcache-hcu
PYTORCH_ROCM_ARCH="{your_rocm_arch}" \
TORCH_DONT_CHECK_COMPILER_ABI=1 \
CXX=hipcc \
BUILD_WITH_HIP=1 \
# Optional:
HCU_ARCH=gfx906  # Platform-specific value; use rocminfo | grep -i "name:" to query it. gfx906 is only an example.
python3 -m pip install --no-build-isolation -e .
```

## License and Attribution

This repository is based on LMCache and is licensed under the Apache License, Version 2.0.

Original copyright, license, and attribution notices from the upstream LMCache project are retained.

Portions of this repository have been modified by Hygon Information Technology Co., Ltd. for HCU enablement and compatibility changes.

Copyright 2026 Hygon Information Technology Co., Ltd.

See LICENSE for the full license text.
See NOTICE for attribution notices.
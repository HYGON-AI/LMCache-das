This repository is based on the following fixed upstream baseline:<br>

 • Upstream project: LMCache<br>
 • Upstream repository: GitHub - https://github.com/LMCache/LMCache.git: LMCache is a KV cache management layer for LLM inference.<br>
 • Upstream tag: v0.5.6<br>
 • Upstream commit: 3e11e10812b448142c6cdb177a3058652bd826fa<br>
 • Upstream license: Apache-2.0<br>

HCU adaptations, modifications, and original contributions by Hygon Information Technology Co., Ltd. are licensed under the Apache License, Version 2.0.
Modified by Hygon Information Technology Co., Ltd.
Original copyright notices and license terms from the upstream LMCache project are retained. See LICENSE and Third-Party Notices for details.

# LMCache-HCU

LMCache-HCU is an overlay/patch package for adapting upstream LMCache to Hygon HCU / ROCm / HIP environments.

## Compatibility Matrix

Please ensure your environment matches the versions below.


| LMCache-hcu | LMCache | vLLM Version |
| :--- | :--- | :--- |
| **v0.5.6** | **v0.5.6** | **v0.28.1** |

## Design

There are two patch layers:

1. Runtime monkey patches in `lmcache_hcu/__init__.py`
   - Triggered by `import lmcache_hcu`.

2. Source injection patches in `lmcache_hcu/integration/patch/`
   - Modeled after `LMCache-das/lmcache_hcu/integration/patch`.
   - Disables streaming I/O during ROCm builds by adding LMCACHE_DISABLE_STREAMING_IO=1 in setup_extensions/build_profiles/rocm.py.
   - Fixes multi-node vLLM local device mapping by updating calculate_local_rank_and_world_size() in lmcache/integration/vllm/utils.py.
   - Keeps source modifications intentionally tiny: inject `import lmcache_hcu` at stable trigger points.

## Important Files

```text
LMCache-das/
├── lmcache_hcu/
│   ├── __init__.py
│   ├── _version.py
│   └── integration/
│       ├── __init__.py
│       └── patch/
│           ├── __init__.py
│           ├── base_patcher.py
│           ├── apply_patch.py
│           └── lmcache/
│               ├── __init__.py
│               ├── rocm_build_patch.py
│               └── device_mapping_patch.py
├── tests/
│   └── __init__.py
├── LICENSE
├── MANIFEST.in
├── THIRD_PARTY_NOTICES.md
├── pyproject.toml
├── README.md
└── setup.py
```

## Build

### 1. Prepare upstream LMCache v0.5.6

```bash
git clone -b v0.5.6 https://github.com/LMCache/LMCache.git
```

### 2. Install LMCache-HCU

From the LMCache-das repository:

```bash
cd /path/to/LMCache-das
python3 -m pip install --no-build-isolation -e .
```

### 3. Apply HCU source patches

Apply both patches:

```bash
lmcache-hcu-apply-patches \
  --lmcache-root /path/to/LMCache \
  --mode install
```

### 4. Build upstream LMCache

After applying the patches, use your platform's supported HCU/ROCm build environment to build and install upstream LMCache.

```bash
cd /path/to/LMCache
BUILD_WITH_HIP=1 python3 -m pip install --no-build-isolation -e .
```

## Usage

Example usage of LMCache with vLLM:

```bash
vllm serve /models/Qwen/Qwen3-8B \
  --tensor-parallel-size 8 \
  --port 20013 \
  --trust-remote-code \
  --disable-log-requests \
  --kv-transfer-config '{"kv_connector":"LMCacheConnectorV1","kv_role":"kv_both"}'
```

## Uninstall

```bash
pip uninstall lmcache-hcu
```

Note: Uninstalling LMCache-HCU does not automatically revert source patches applied to upstream LMCache.

## Upgrade

Upgrade by uninstalling the old LMCache-HCU package first, then installing the new version.

Recommended flow:

```bash
# 1. Remove the old package.
pip uninstall lmcache-hcu

# 2. Install the new package version.
cd /path/to/lmcache-hcu
python3 -m pip install --no-build-isolation -e .
```

## License and Attribution

This repository is based on LMCache and is licensed under the Apache License, Version 2.0.

Original copyright, license, and attribution notices from the upstream LMCache project are retained.

Portions of this repository have been modified by Hygon Information Technology Co., Ltd. for HCU enablement and compatibility changes.

Copyright 2026 Hygon Information Technology Co., Ltd.

See [LICENSE](LICENSE) for the full license text and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party attribution details.

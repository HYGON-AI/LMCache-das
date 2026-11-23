This repository is based on the following fixed upstream baseline:

- Upstream project: LMCache
- Upstream repository: https://github.com/LMCache/LMCache.git
- Upstream tag: `v0.5.6`
- Upstream commit: `3e11e108` (short SHA)
- Upstream license: Apache-2.0

HCU adaptations, modifications, and original contributions by Hygon Information Technology Co., Ltd. are licensed under the Apache License, Version 2.0.

Modified by Hygon Information Technology Co., Ltd.

Original upstream copyright notices and applicable license terms are retained.

# LMCache-HCU

LMCache-HCU is an overlay/patch package for adapting upstream LMCache to Hygon HCU / ROCm / HIP environments.

This version provides source-level compatibility fixes for LMCache v0.5.6 without maintaining a separate copy of the upstream LMCache source tree.

## Compatibility Matrix

Please ensure your environment matches the versions below.

| LMCache-HCU | LMCache | vLLM Version |
| :--- | :--- | :--- |
| **0.5.6+hcu** | **v0.5.6** | Deployment-dependent; not pinned by this package |

The source patches target the fixed upstream LMCache v0.5.6 baseline. HCU runtime compatibility must be validated with the target vLLM, PyTorch, and ROCm/DTK environment.

## Design

This version uses source patches implemented in `lmcache_hcu/integration/patch/`.

The patch framework:

1. Locates the upstream LMCache source tree using an explicitly provided path.
2. Verifies the expected LMCache Git baseline.
3. Checks the original and patched source content before applying modifications.
4. Creates timestamped `.bak.<timestamp>` backups before modifying files.
5. Supports repeated execution without duplicating previously applied patches.

### Patch 1: ROCm Build Compatibility

Target file: `setup_extensions/build_profiles/rocm.py`

Adds the compile-time macro `LMCACHE_DISABLE_STREAMING_IO=1` to the ROCm build profile. This avoids the streaming I/O instruction path in the affected extension build for the target HCU environment.

**Apply this patch before building LMCache.** If an extension was built before applying the patch, it must be rebuilt for the macro to take effect.

### Patch 2: Multi-node Device Mapping

Target file: `lmcache/integration/vllm/utils.py`

Updates `calculate_local_rank_and_world_size()` for multi-node deployments. The multi-node path uses the device already bound by vLLM (`torch_dev.current_device()`) and the number of locally visible devices (`torch_dev.device_count()`), instead of assuming tensor parallelism is always intra-node and pipeline parallelism is inter-node.

The implementation assumes one worker per visible device and requires the worker's device to have been bound before the function is called. It is not a general guarantee of compatibility with every distributed topology.

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
git clone --branch v0.5.6 \
  https://github.com/LMCache/LMCache.git \
  LMCache-v0.5.6
```

Keep the upstream Git history and `v0.5.6` tag available. The current patch manager verifies the Git baseline and checks the exact expected source content.

### 2. Install LMCache-HCU

From the LMCache-das repository:

```bash
cd /path/to/LMCache-das
python3 -m pip install --no-build-isolation -e .
```

This installs the Python source-patch manager. LMCache-HCU does not compile a separate HCU C++/HIP extension in this version.

### 3. Check and apply HCU source patches

Check the upstream checkout:

```bash
lmcache-hcu-apply-patches \
  --lmcache-root /path/to/LMCache-v0.5.6 \
  --mode check
```

Apply both patches:

```bash
lmcache-hcu-apply-patches \
  --lmcache-root /path/to/LMCache-v0.5.6 \
  --mode install
```

Check again:

```bash
lmcache-hcu-apply-patches \
  --lmcache-root /path/to/LMCache-v0.5.6 \
  --mode check
```

Both patches should report `applied`. Repeating `install` skips correctly applied patches.

### 4. Build upstream LMCache

After applying the patches, use your platform's supported HCU/ROCm build environment to build and install upstream LMCache. For example, if your environment already provides the required HIP compiler and dependencies:

```bash
cd /path/to/LMCache-v0.5.6
BUILD_WITH_HIP=1 python3 -m pip install --no-build-isolation -e .
```

Additional variables such as the target ROCm architecture, `CXX`, and compiler compatibility settings depend on the HCU deployment. Follow the applicable platform-specific build instructions.

## Usage

Inspect the changes made to upstream LMCache:

```bash
git -C /path/to/LMCache-v0.5.6 diff -- \
  setup_extensions/build_profiles/rocm.py \
  lmcache/integration/vllm/utils.py
```

The patch manager supports:

| Mode | Description |
| :--- | :--- |
| `check` | Verify the baseline and report patch states without modifying files |
| `install` | Apply missing source patches |

This version does not provide runtime monkey patches, a custom vLLM Connector, or a separate custom HIP operator. Runtime compatibility should be verified in the target HCU deployment.

## Uninstall

To remove the LMCache-HCU Python package:

```bash
python3 -m pip uninstall lmcache-hcu
```

**Uninstalling the Python package does not revert changes already applied to upstream LMCache source files.** The patch manager creates timestamped backups, but restoration should be performed carefully to avoid overwriting subsequent user changes. The current CLI does not provide an automatic uninstall mode.

## Upgrade

For future LMCache-HCU versions:

1. Review the compatibility matrix and supported upstream baseline.
2. Prepare a clean checkout of the supported LMCache version.
3. Install the corresponding LMCache-HCU package.
4. Apply its source patches before building LMCache.
5. Rebuild LMCache and validate the target HCU deployment.

Do not assume patches designed for v0.5.6 can be applied to other LMCache versions.

## License and Attribution

This repository is licensed under the Apache License, Version 2.0.

Original upstream LMCache copyright, license, and attribution notices are retained where applicable. HCU compatibility changes have been made by Hygon Information Technology Co., Ltd.

Copyright 2026 Hygon Information Technology Co., Ltd.

See [LICENSE](LICENSE) for the full license text and [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md) for third-party attribution details.

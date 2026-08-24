# LMCache-HCU CI

This directory contains the trusted controller shared by the PR, manual and
weekly GitHub Actions workflows. The first release runs the complete pytest
inventory under the repository `tests/` directory and the reviewed model
scenarios declared in `model-test-manifest.json`.

## Execution model

GitHub Actions exposes one job as seven readable steps:

```text
Checkout trusted CI controller
Checkout source under test
Preflight and initialize
Build and verify current wheel
Verify patches and collect tests
Run HCU test suite
Cleanup, validate and publish
```

The trusted controller comes from the protected target branch. The selected PR
merge or manual revision is mounted read-only into a restricted container. A
disposable source copy is used to build exactly one wheel, and that same wheel
and virtual environment are reused for source-patch checks, pytest and model
tests. The original checkout is never built in place.

The fixed SourceFind test tool is fetched by the trusted host controller with a
temporary `GIT_ASKPASS` helper. Its username and token are never mounted into
the test container. The reviewed checkout is pinned to the commit in
`model-test-manifest.json`; the CI never follows a moving branch and never
executes arbitrary commands from `case_list`.

Container output is written to a size-bounded tmpfs. Finalization freezes the
container, rejects links, special files, unexpected paths and oversized output,
then imports regular files into a trusted spool. The host independently checks
pytest inventory/JUnit, model inventory/JUnit, structured test-tool results,
patch cleanup and checksums before publishing `READY` last.

## Runner and profiles

All HCU workflows use only this registered runner:

```text
nmz4-hygon-hcu-lmcache
self-hosted, Linux, X64, hcu, hcu-ci-pr, bw1100, nmz4
```

Profiles:

- `pr`: repository pytest plus Qwen3-8B LocalDisk and POSIX long-document
  checks; two visible devices, 60-minute workflow limit.
- `framework`: repository pytest only; one visible device.
- `qwen-smoke`: the two PR model scenarios; two visible devices.
- `weekly-bw1100`: all registered Qwen, DeepSeek and GLM scenarios; eight
  visible devices.
- `full`: alias of `weekly-bw1100` for manual dispatch.

The runner lock is held for the complete job, including two-device profiles, so
another repository job cannot use the remaining cards concurrently.

## Configuration

The workflows remain skipped until both variables are true:

```text
LMCACHE_HCU_CI_ENABLED=true
LMCACHE_HCU_RUNNER_ISOLATED=true
```

Required repository variables:

```text
LMCACHE_HCU_BASE_IMAGE=<reviewed image tag or registry digest>
LMCACHE_HCU_BASE_IMAGE_ID=sha256:<immutable local image ID>
LMCACHE_HCU_CACHE_ROOT_NMZ4=<dedicated path containing /lmcache-das/>
LMCACHE_HCU_ARCH=<optional expected architecture reported by the image>
```

Registry images should use an `@sha256:` digest. A reviewed image imported
from a tar archive may use its tag, but the immutable Docker image ID remains
mandatory and is checked before any container starts. The initial nmz4 image
baseline is Python 3.10, Torch 2.9.0, vLLM 0.15.1, DTK 26.04, ABI 1 and
`gfx938`.

The initial tar image contains reviewed, pre-existing `pip check` conflicts.
The wheel gate records `pip check` before and after installing the current
wheel and requires the two results to remain byte-for-byte identical; any new
or changed dependency conflict fails the run.

Required read-only SourceFind credentials:

```text
LMCACHE_TEST_TOOL_USERNAME
LMCACHE_TEST_TOOL_TOKEN
```

The fixed upstream source is read from:

```text
/ci_public/lmcache-das/assets/upstream/LMCache/v0.3.13/fc031d471a566edb6d49a86c9116cc23cfb04111/
```

Each run creates only one cache subtree below the configured nmz4 root. The
controller probes LocalDisk, SSD and POSIX subdirectories with direct I/O,
mounts them as `/local_disk`, `/ssd` and `/mnt/parastor_storage`, and removes
only the current run subtree after the test container is confirmed absent.

## Registered model tests

The manifest registers:

- Qwen3-8B: CPU, LocalDisk and POSIX.
- DeepSeek-R1-0528-W4A8-V2: CPU, LocalDisk and POSIX.
- GLM-5: LocalDisk and POSIX.
- long-document cache behavior, HumanEval/GSM8K OpenCompass accuracy and the
  reviewed long-prefix CMMLU case.

Qwen is currently marked ready. DeepSeek and GLM remain explicitly blocked in
the manifest until the SourceFind repository contains configs whose model path
and tensor-parallel size match the cluster assets. Selecting a blocked weekly
profile fails during test-tool validation; it is never silently skipped.

The model runner treats process exit codes and JSON content as independent
gates. Missing, duplicate, malformed or `failed` records fail the scenario even
when the developer script exits zero. Every scenario/repetition becomes one
JUnit testcase.

## Reports

Results are atomically published under:

```text
/ci_public/lmcache-das/<pr|manual|weekly>/<run-id>/<attempt>/<sha>/
```

The archive includes the wheel, pytest and model JUnit, raw model JSON,
effective configs, logs, environment, test and model inventories, patch report,
asset manifest, summary, checksums and `READY`.

## Static verification

Run on Linux with Python 3.10+:

```bash
bash ci/hcu/selftest.sh
git diff --check
```

Full wheel and model execution additionally require the reviewed image, nmz4
cache root, model assets and SourceFind credentials. The implementation does
not weaken a failed compatibility, packaging, patch, pytest or model gate.

The pinned LMCache v0.3.13 server fixture assumes that its CPU server becomes
ready within fifteen seconds.  The reviewed HCU image needs about twenty
seconds to import the HCU runtime on BW1100.  CI therefore loads the trusted
`lmcache_ci_pytest` support plugin, which changes only this fixture's readiness
probe to a process-aware 60-second deadline.  Repository tests and upstream
assertions are not modified or skipped.

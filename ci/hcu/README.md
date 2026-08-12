# LMCache-HCU PR CI

This directory contains the platform-neutral controller used by the HCU PR and
manual GitHub Actions workflows. The first release intentionally runs every
pytest test discovered under the repository `tests/` directory in one process.
The BW18 runner exposes all eight HCU devices by default, matching the SGLang
runner resource model, although the current tests do not yet claim multi-device
coverage. It does not claim to run the complete upstream LMCache test tree.

## Execution contract

- The host controller is loaded from a protected branch.
- The fixed-digest base image is part of the trusted computing base: it runs
  the reviewed helper that imports the frozen, read-only test output into the
  host spool. Activating CI therefore requires review and locking of that digest.
- The selected PR/ref checkout and the fixed upstream LMCache source are mounted
  read-only into a network-disabled container.
- The container creates disposable source copies, builds one wheel from the
  selected SHA, installs that wheel, validates both source patches, collects all
  repository tests, and executes the complete inventory.
- GitHub Actions exposes the execution as seven readable steps: two trusted
  checkouts, preflight/initialization, wheel build and validation, patch/test
  preparation, pytest execution, and an always-run cleanup/publication step.
  The four HCU phases use one restricted container, so the selected commit is
  compiled exactly once and the same wheel and virtual environment are reused.
- A minimal trusted host lease process holds the dedicated runner lock for the
  complete job while GitHub moves between workflow steps. The lease receives no source, device,
  output, network, or shared-results access.
- Because the current `setup.py` package discovery would include `tests/`, the
  controller first copies the full test tree to its execution directory and
  removes it only from the disposable build copy. The checkout and business
  packaging files are not changed, and the wheel gate still forbids test files.
- Container output is written to a per-run, size-bounded `/output` tmpfs
  (3 GiB by default). Finalization pauses the test container and a device-less,
  network-disabled trusted helper reads that frozen tree through its PID
  namespace. `host.py` imports it into a trusted spool only after
  rejecting symlinks, hard links, special files, unexpected paths, and output
  above the stricter 2 GiB publication limit.
- The trusted host validator checks the committed minimum test inventory, JUnit
  counts, skips, patch report, cleanup, checksums, and atomic `/ci_public`
  publication before creating `READY`.

The Actions page shows these steps:

```text
Checkout trusted CI controller
Checkout source under test
Preflight and initialize
Build and verify current wheel
Verify patches and collect tests
Run HCU test suite
Cleanup, validate and publish
```

The final step uses `if: always()`. The four compute phases use
`!cancelled()`, so a cancelled superseded run does not start another expensive
phase while cleanup still gets a chance to run. Finalization restores temporary
patches, imports the bounded output, removes the test and legacy lease
containers, releases the runner lock, validates the reports, and publishes the run. The
runner lease is released only after the test container is confirmed absent; if
Docker cannot remove it, the lease remains held and the isolated runner must be
reset. A prior build or test failure remains visible on its own step;
cleanup/publication does not hide it.

The default device selection is `0,1,2,3,4,5,6,7`. It can be changed through
the reviewed repository variable `LMCACHE_HCU_VISIBLE_DEVICES`. The current
preflight requires all eight cards and rejects duplicates, out-of-range
ordinals, missing device nodes, or an unexpected visible-device count. As in SGLang,
`HIP_VISIBLE_DEVICES` and `CUDA_VISIBLE_DEVICES` select the cards;
`ROCR_VISIBLE_DEVICES` is not forced.

The workflow is pinned to the registered `bw18-hygon-hcu-lmcache` runner by
requiring all of these labels:

```text
self-hosted, Linux, X64, hcu, bw1000, hcu-ci-pr, bw18
```

It remains inactive until these three repository variables are configured:

```text
LMCACHE_HCU_CI_ENABLED=true
LMCACHE_HCU_RUNNER_ISOLATED=true
LMCACHE_HCU_BASE_IMAGE=<registry>/<repository>@sha256:<digest>
```

`LMCACHE_HCU_RUNNER_ISOLATED=true` is a security assertion, not merely a feature
flag. The runner must contain no secrets, expose no internal credentials to the
test container, and be reset after every untrusted PR job.

The fixed upstream source must be available at:

```text
/ci_public/lmcache-das/assets/upstream/LMCache/v0.3.13/fc031d471a566edb6d49a86c9116cc23cfb04111/
```

It must be a clean Git checkout at tag `v0.3.13`; the controller never modifies
it directly.

## Local checks

Run on a Linux host with Python 3.6+:

```bash
bash ci/hcu/selftest.sh
git diff --check
```

Full wheel, patch, and pytest validation additionally requires the reviewed HCU
base image and one registered HCU runner. A clean build failure is reported as a
build-stage failure; the CI does not generate or copy missing native sources to
hide a packaging defect.

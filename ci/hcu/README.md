# LMCache-HCU CI

This directory contains the trusted controller shared by the PR, manual and
weekly GitHub Actions workflows. The first release runs the complete pytest
inventory under the repository `tests/` directory and the reviewed model
scenarios declared in `model-test-manifest.json`.

## LMCache 0.4.2 restoration

This controller incorporates the test adaptations from PR #15. The removed
thread-pool-status test checks a feature not ported to 0.4.2; its required
baseline definition is removed too. The complete remaining `tests/` tree is
still collected dynamically. Do not hard-code the previous 156/160 counts or
interpret controller self-tests as the repository's GPU unit-test results.

On 2026-10-26 the maintainer explicitly approved temporarily disabling the
four failures reproduced on 2026-10-22/23. `disabled-tests.json` records exact
nodeids, individual causes and restoration conditions. Test source and required
baseline definitions remain intact. Collection first discovers everything,
then excludes only these exact nodes from execution. A missing/stale entry,
duplicate registration or attempt to disable the entire suite is an error.
At the reviewed revision this means 159 collected, 155 active, 4 disabled;
disabled nodes are neither successful tests nor pytest skips. Inventory,
collection logs and `summary.md` expose the exclusions. New tests are still
automatically collected, and any other failure, skip or xfail still fails CI.

The host importer permits the exact `disabled-tests.json` metadata filename,
subject to the existing size, regular-file and no-link checks. It does not
permit arbitrary new JSON files. Before publication the host compares that
copy and the inventory manifest SHA with the trusted controller manifest,
then verifies that active nodes are exactly complete collection minus the
approved exclusions. Required-file/definition baselines use complete
collection; JUnit counts use only active nodes times repetitions. Neither
stale exclusions, unapproved omissions nor missing executed results can pass.
This fixes the report-import failure in run `40303494607`: its 155 pytest
cases passed, but the missing output allowlist entry prevented publication
and therefore prevented model jobs from starting.
The required-definition baseline also follows the five test-name replacements
already made by the developer's 0.4.2 adaptation (custom Flash Attention,
tuple KV layout and hidden-dimension checks), and explicitly includes the
custom-attention default-off assertion. A device-free AST regression checks
the required definitions in the two locally adapted attention test files;
all definitions, including upstream re-exports, remain checked at collection.
No test assertion or approved exclusion is changed by this synchronization.

The disabled nodes are the two storage-factory tests
`test_create_storage_backends_adds_xds_backend` and
`test_create_storage_backends_does_not_add_xds_without_xds_path`, plus the
cache-engine tests `test_retrieve_sync_path_records_hcu_stats_and_loads_to_gpu`
and `test_retrieve_remove_after_retrieve_deletes_storage_key`. This temporary
decision does not fix their semantics. The separately approved suspension of
the strict patch gate is described below.

`compatibility.json` pins the reviewed vLLM 0.18.1 image digest and Docker image
ID. At cutover, repository variables `LMCACHE_HCU_BASE_IMAGE` and
`LMCACHE_HCU_BASE_IMAGE_ID` must match that lock. The host pulls a missing
digest reference automatically; a missing old local-only tag is an error.

Four small runtime/build dependencies are pinned in `dependency-lock.json`.
Their wheels must be staged in its fixed `/ci_public` wheelhouse before
activation. Each restricted test container mounts the directory read-only,
checks every SHA256, and installs only those local wheels into the disposable
venv with `--no-index --no-deps`. No current plugin wheel is reused across
different tested source SHAs, and model jobs still consume the one framework
wheel.

The original test-tool archive and its SHA stay unchanged. For the six
Qwen/DeepSeek scenarios, effective configurations remove only
`vllm.disable-log-requests = true`, which vLLM 0.18.1 rejects. A missing,
duplicated or different-valued option fails rather than being silently
rewritten. Model paths, TP, workload sizes and required KV-cache parameters
are unaffected by this particular adaptation; earlier reviewed CPU/workload
overrides remain explicit in the scenario manifest. Qwen's Flash Attention
variable and DeepSeek's CAT MLA variable remain separate contracts.

On 2026-10-26 the maintainer approved suspending CI's additional strict
source-patch gate and following the repository's current patch behavior.
Production jobs now invoke the installed current-wheel patch CLI once in
`install` mode, and invoke the same CLI in `uninstall` mode during cleanup
(also after a partially failed install). Nonzero command exits still fail CI.
There are no production checks of target paths, preimages, markers, mutation
allowlists, repeated installation, or restored hashes. These checks are not
silently reported as passed: `patch-report.json` explicitly records
`mode: repository-cli`, `strict_validation: disabled`, the install outcome
and the uninstall outcome. It is an execution receipt, not a compatibility
certificate. The host publisher accepts this receipt instead of requiring a
passed strict gate. Disposable container/source cleanup and all pytest/model
result checks remain required. No business patcher, runtime alias or test
assertion is changed by this decision.

The old `patch-gate`, `patch-cleanup` diagnostic commands and patch manifest
remain available for explicit investigation only; no workflow invokes them
and the manifest is no longer a production preflight requirement. A separate
review is required before restoring that gate.

Validation order: trusted-controller branch plus tested PR
revision, current-SHA build/install/repository patch command, full pytest inventory,
Qwen LocalDisk/POSIX, real PR event, then full Weekly. PR controllers and
baseline manifests come from the target branch; changing only the PR head's
CI scripts does not activate them. Do not mark this migration complete based
only on syntax checks, mocks or configuration generation.

## Execution model

Each execution unit still exposes seven readable steps:

```text
Checkout trusted CI controller
Checkout source under test
Preflight and initialize
Build and verify current wheel
Apply repository patches and prepare selected tests
Run HCU test suite
Cleanup, validate and publish
```

The workflows split those units into visible jobs. `framework-and-pytest`
builds exactly one current-SHA wheel and runs the complete repository pytest
inventory. Every selected model scenario then receives its own job, installs
the already verified framework wheel from `/ci_public`, invokes the repository
patch CLI in a fresh environment, and runs only that scenario.
The final summary job verifies all child manifests, checksums and completion
markers before creating the run-level `READY` marker. No GitHub Artifact is
used and model jobs never rebuild the package.

All publication hierarchy directories use the runners' shared group with
setgid group traversal/write permissions; child result directories are
group-readable. This is required because organization runners can have
different numeric user IDs while sharing the same `/ci_public` group and NFS
tree. Test containers still receive only the verified staged wheel, never a
writable `/ci_public` mount. Bind sources are canonicalized and validated
before the controller passes them to the organization Docker wrapper using
its supported `-v` syntax.

PR therefore has four visible jobs: framework, Qwen LocalDisk, Qwen POSIX and
the required `hcu-tests` summary. Weekly has eight visible jobs: framework,
six Qwen/DeepSeek scenario jobs and `weekly-summary`. Jobs are routed across
five organization runners, while each scenario retains an independent timeout
and an independent client/server log stream.

The trusted controller comes from the protected target branch. The selected PR
merge or manual revision is mounted read-only into a restricted container. A
disposable source copy is used for wheel construction; the original checkout
is never built in place.

The fixed test-tool tar archive is read from the shared `/ci_public` asset tree.
The trusted host verifies its SHA256, rejects unsafe archive entries, verifies
the embedded Git commit and creates a clean detached checkout. No network
credential is required or passed to the test container. The CI never follows a
moving branch and never executes arbitrary commands from `case_list`.

Container output is written to a size-bounded tmpfs. Finalization freezes the
container, rejects links, special files, unexpected paths and oversized output,
then imports regular files into a trusted spool. The host independently checks
pytest inventory/JUnit, model inventory/JUnit, structured test-tool results,
patch cleanup and checksums before publishing `READY` last.
For ordinary PRs, the same result checks run before disposable output is
removed, but shared report publication is disabled as described below.

## Runner and profiles

All HCU workflows use the organization runner pool below:

```text
10.17.1.1  ci-nmz1  self-hosted, Linux, X64, ci, bw1100, nmz1
10.17.1.2  ci-nmz2  self-hosted, Linux, X64, ci, bw1100, nmz2
10.17.1.3  ci-nmz3  self-hosted, Linux, X64, ci, bw1100, nmz3
10.17.1.4  ci-nmz4  self-hosted, Linux, X64, ci, bw1100, nmz4
10.17.1.5  ci-nmz5  self-hosted, Linux, X64, ci, bw1100, nmz5
```

Profiles:

- `pr`: one framework job plus independent Qwen3-8B LocalDisk and POSIX
  long-document jobs; each job has its own 60-minute limit.
- `framework`: repository pytest only; one visible device.
- `qwen-smoke`: the two PR model scenarios; two visible devices.
- `weekly-bw1100`: all enabled Qwen and DeepSeek scenarios, one scenario per
  job; Qwen jobs expose two devices and DeepSeek jobs expose eight.
- `full`: alias of `weekly-bw1100` for manual dispatch.

PR assigns framework, LocalDisk and POSIX to nmz1, nmz2 and nmz3. Weekly uses
all five nodes with at most five model jobs in parallel; two jobs assigned to
nmz1 are serialized by its single organization Runner listener. Manual runs
accept `runner=auto|nmz1|nmz2|nmz3|nmz4|nmz5`: `auto` uses the reviewed mapping,
while a named runner forces all selected jobs onto that node for diagnostics.
Each job also holds a node-specific host lock for its complete execution unit.
The organization Runner Group must grant this repository access. Each physical
node must run only its `ci-nmzN` organization listener; legacy repository-level
Runner services must be stopped before `LMCACHE_HCU_RUNNER_ISOLATED=true` is
enabled, otherwise another repository can still bypass this scheduler and use
the same cards.

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
LMCACHE_HCU_ARCH=<optional expected architecture reported by the image>
```

Registry images should use an `@sha256:` digest. A reviewed image imported
from a tar archive may use its tag, but the immutable Docker image ID remains
mandatory and is checked before any container starts. The same image ID must
be loaded on every selected runner. The initial image baseline is Python 3.10,
Torch 2.10.0, vLLM 0.18.1, DTK 26.04, ABI 1 and `gfx938`.

The initial tar image contains reviewed, pre-existing `pip check` conflicts.
The wheel gate records `pip check` before and after installing the current
wheel and requires the two results to remain byte-for-byte identical; any new
or changed dependency conflict fails the run.

Fixed offline test-tool asset:

```text
/ci_public/lmcache-das/assets/test-tool/
lmcache_test_tool-b01d189144ebbf37f3f1bfafe7ea4452dba6053f.tar
```

Its SHA256 and embedded Git commit are pinned in `model-test-manifest.json`.
The reviewed image supplies OpenCompass at `/lmcache_workspace/opencompass`.
The source asset remains read-only. Weekly copies the reviewed OpenCompass tree
to `/sandbox/opencompass-ci` before execution because OpenCompass writes task
parameter files below its own `tmp/` directory.
The supplied CMMLU launcher expects a private Miniconda installer and packed
environment that are not part of the fixed offline image. After verifying the
test-tool commit, CI creates a minimal `conda run` adapter only in the writable
test-tool copy. It accepts the exact version, Python-import and OpenCompass
commands used by that launcher and forwards them to the current CI interpreter
and `/sandbox/opencompass-ci`; it neither downloads packages nor changes the
read-only test-tool asset.
The evaluation datasets available to every runner at
`/public/opendas/DL_DATA/opencompass_data` are mounted read-only at the
OpenCompass compatibility path `/public/ai_data/datasets/data`; its CMMLU
subdirectory is independently mounted read-only at
`/public/ai_data/datasets/cmmlu`. These sibling binds avoid unsupported nested
read-only Docker mounts, do not copy data, and do not allow network access.
They match OpenCompass's fixed `COMPASS_DATA_CACHE/data/...` lookup while the
supplied CMMLU script keeps its fixed `/public/ai_data/datasets/cmmlu` path.

The supplied long-document wrapper assumes every cache backend has a filesystem
offload path. Pure CPU-memory scenarios do not, so the wrapper reports failure
even when its requests and cache operations are valid. For those two reviewed
CPU scenarios only, CI uses four 32K documents (bounded by the configured 32 GiB
CPU cache) and independently requires complete warmup/query requests, a lower
query TTFT, positive stored/retrieve/load counters, and no filesystem backend.
The original tool JSON and exit code remain archived.

The model start probe uses a reviewed 300-second request deadline. A cold vLLM
start can compile its first request after the HTTP service is already ready;
the complete start phase remains independently bounded by its one-hour timeout.

The fixed upstream source is read from:

```text
/ci_public/lmcache-das/assets/upstream/LMCache/v0.4.2/9d413181c30ec2ec97bcb6b17defa7c5f9f8e36e/
```

Model caches use four run-scoped Docker local volumes under the existing
DockerRootDir on `/data` (currently `/data/docker/volumes/`), not root-backed
`$RUNNER_TEMP`. Old `LMCACHE_HCU_CACHE_ROOT_NMZ*` values are not used. No sudo,
Docker policy change, direct `/data` bind or administrator provisioning is needed.
This is disposable local scratch, not a shared-storage performance test.
The pinned Qwen POSIX config still uses `/local_disk/posix_cache`, O_DIRECT and
the same POSIX backend; model paths, options and workloads do not change.

Volumes are named `lmcache-hcu-cache-<run-key>-<cpu|localdisk|ssd|posix>` and
mounted as `/mnt/fs1`, `/local_disk`, `/ssd` and `/mnt/parastor_storage` with
`nocopy`. Before acquiring GPUs, a bounded, device-less probe in the pinned
image checks that all four mounts are on the host's separate `/data` filesystem,
have at least 300 GiB free, and support a 4 KiB O_DIRECT write. This floor leaves
headroom over the observed ~226 GiB Qwen cache; it is not a reserved quota or a
promise that concurrent users cannot consume disk space later. Workloads and
test assertions are unchanged. A failed check stops model startup explicitly.

Finalization removes only these exact volumes with matching repository/run
labels, after the test container is confirmed absent. It never prunes volumes
or deletes old runner directories; an in-use volume or Docker error fails cleanup.
Reports remain in their existing output/spool paths and do not live in cache
volumes. The CPU mount preserves the fixed test tool's `/mnt/fs1/posix` behavior
without making the container root writable.

The Qwen CPU config in the fixed test-tool archive contains one duplicate
`disable-cascade-attn` entry. The manifest authorizes removal of only that
duplicate when CI creates the recorded effective config; the reviewed archive
itself is never modified.

## Registered model tests

The manifest registers:

- Qwen3-8B: CPU, LocalDisk and POSIX.
- DeepSeek-R1-0528-W4A8-V2: CPU, LocalDisk and POSIX.
- GLM-5: LocalDisk and POSIX.
- long-document cache behavior, HumanEval/GSM8K OpenCompass accuracy and the
  reviewed long-prefix CMMLU case.

Qwen and DeepSeek are marked ready. The cluster DeepSeek-R1-0528-W4A8-V2 asset
is mounted at the reviewed container alias expected by the existing eight-GPU
DeepSeek configuration, so the shared model directory is not modified. GLM
remains declared but is not selected by any active profile until reviewed GLM
configuration is available. Selecting a blocked model fails during test-tool
validation; it is never silently skipped.

The model runner treats process exit codes and JSON content as independent
gates. Missing, duplicate, malformed or `failed` records fail the scenario even
when the developer script exits zero. Every scenario/repetition becomes one
JUnit testcase.

All DeepSeek scenarios export `VLLM_USE_CAT_MLA=1`. Their reviewed config must
still contain `kv-cache-dtype = fp8_e4m3`; the manifest gate rejects a config
that drops or changes it. On BW1100, the first Qwen or DeepSeek request can spend
the initial five-minute probe compiling and then terminate with the exact paired
`sample_tokens` RPC timeout and `EngineDeadError` signature. Both models may
perform one controlled clean restart for that signature; OOM and all other
startup failures remain non-retryable. The failed first-attempt JSON and server
log remain archived. DeepSeek uses 16 generated tokens per long-document request
because a one-token MTP response can finish before producing a measurable first
token. LocalDisk and POSIX use four in-flight requests; DeepSeek OpenCompass uses
the reviewed batch size 32, while still requiring all 50 warmup and 50 query
requests to succeed. A
tool-level `result=success` cannot hide partial responses such as 20/50 or 19/50.

## Reports

Ordinary PRs do not require long-lived report archives. Pytest inventory/JUnit,
model JSON/JUnit, original patch-CLI exits and resource cleanup are still checked
strictly on the host. Missing or failed results fail the job. The required
`hcu-tests` summary consumes actual Actions framework/model job results and
rejects failure, cancellation or skipped jobs; it does not depend on NFS log
copies or archive `READY` markers. Logs remain visible in Actions; temporary
reports are removed after validation. There is no durable-evidence service,
republish command or three-node preflight matrix to configure.

Only the current-SHA wheel is a required shared build artifact:

```text
/ci_public/lmcache-das/wheels/<run-id>/<attempt>/<source-sha>/framework/
  wheels/<one-current-wheel>.whl
  manifest.json
  SHA256SUMS
  READY
```

It contains no pytest or model logs. Models verify the source/controller SHA,
run/attempt, manifest and checksums, then stage only that wheel into runner-local
read-only input. A missing/corrupt wheel remains a real transfer failure; CI
does not rebuild independently on model nodes or reuse another source's wheel.
Existing group-writable shared parents are checked, not chmodded/chowned by
non-owner runner UIDs; permissions are set only on newly created parents.

Manual/weekly workflows retain their existing required archival mode; this
change does not schedule or expand them. Their results are atomically published under:

```text
/ci_public/lmcache-das/<pr|manual|weekly>/<run-id>/<attempt>/<sha>/
├── framework/
├── <model-scenario>/
├── aggregate-manifest.json
├── summary.md
├── SHA256SUMS
└── READY
```

Each child directory contains its own JUnit, logs, environment, inventories,
patch report, manifest, checksums and completion marker. The framework child
owns the single wheel and pytest results; model children own their raw JSON,
effective config and client/server logs. Top-level `READY` is written only when
every expected child reports success and passes checksum validation.

## Static verification

Run on Linux with Python 3.10+:

```bash
bash ci/hcu/selftest.sh
git diff --check
```

Full wheel and model execution additionally require the reviewed image on the
selected runner, its cache root, model assets and fixed local test-tool archive.
The implementation does not weaken a failed compatibility, packaging, patch-command, pytest or
model result. Strict source-patch semantic validation is suspended as noted above.

The pinned LMCache v0.4.2 server fixture assumes that its CPU server becomes
ready within fifteen seconds.  The reviewed HCU image needs about twenty
seconds to import the HCU runtime on BW1100.  CI therefore loads the trusted
`lmcache_ci_pytest` support plugin, which changes only this fixture's readiness
probe to a process-aware 60-second deadline.  Repository tests and upstream
assertions are not modified or skipped.

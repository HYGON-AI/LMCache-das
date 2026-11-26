#!/usr/bin/env bash
set -euo pipefail
cd "${RUNNER_TEMP:?}"
root="${GITHUB_WORKSPACE:?}"
ci="$root/controller/ci/hcu_v056"
out="$root/job-data/reports"
mkdir -p "$out"
export CI_REPORT_DIR="$out"
upstream=3e11e10812b448142c6cdb177a3058652bd826fa
revision() { git -c safe.directory="$1" -C "$1" rev-parse HEAD; }
[[ "$(revision "$root/controller")" == "$CONTROLLER_SHA" ]]
[[ "$(revision "$root/candidate")" == "$GITHUB_SHA" ]]
[[ "$(revision "$root/upstream")" == "$upstream" ]]
printf 'controller=%s\npr_head=%s\nmerge=%s\nupstream=%s\n' "$CONTROLLER_SHA" "$HEAD_SHA" "$GITHUB_SHA" "$upstream" > "$out/revisions.txt"

# Sanity checks only; the runner group and pod security/isolation are enforced
# by the platform outside the workflow, before *any* PR runs.
for forbidden in /public/home /public4/home /var/run/docker.sock /run/containerd/containerd.sock /var/run/secrets/kubernetes.io/serviceaccount/token; do
    if [[ -e "$forbidden" ]]; then echo "::error::PR pod unexpectedly exposes $forbidden"; exit 1; fi
done
[[ "$TEST_TP" =~ ^(1|2|4|8)$ ]] || { echo "::error::LMCACHE_V056_TP must be 1, 2, 4 or 8"; exit 1; }
[[ "$QWEN_MODEL" == /models/* && -f "$QWEN_MODEL/config.json" ]] || { echo "::error::Mount the approved Qwen3-8B model read-only under /models"; exit 1; }
command -v hipcc >/dev/null || { echo "::error::The HCU image is missing hipcc"; exit 1; }
python3 -m venv --system-site-packages "$RUNNER_TEMP/lmcache-venv"
source "$RUNNER_TEMP/lmcache-venv/bin/activate"
export PIP_NO_INDEX=1
arch="$(python -I "$ci/runtime.py" arch)"
[[ "$arch" =~ ^gfx[0-9a-f]+$ ]] || { echo "::error::Expected one HIP architecture from preflight"; exit 1; }
export PYTORCH_ROCM_ARCH="$arch" TORCH_DONT_CHECK_COMPILER_ABI=1 CXX=hipcc BUILD_WITH_HIP=1
export HIPCC_COMPILE_FLAGS_APPEND=-DLMCACHE_DISABLE_STREAMING_IO=1 HIPCC_VERBOSE=1
python -m pip install --no-cache-dir --no-build-isolation --no-deps --ignore-installed "$root/upstream"
python -I "$ci/wheel.py" verify "$root/job-data/overlay-wheel" "$GITHUB_SHA" | tee "$out/wheel.json"
python -m pip install --no-index --no-deps --ignore-installed "$root"/job-data/overlay-wheel/*.whl
unset SKIP_LMCACHE_HCU_PATCH
python -I "$ci/runtime.py" installed
cli="$VIRTUAL_ENV/bin/lmcache-hcu-apply-patches"
cleanup() {
    original=$?
    trap - EXIT
    set +e
    "$cli" --mode uninstall
    cleanup_rc=$?
    if ((original != 0)); then exit "$original"; fi
    exit "$cleanup_rc"
}
trap cleanup EXIT
"$cli" --mode install
python -I -m pytest -q --import-mode=importlib "$ci/test_device.py" "$root/candidate/tests" --junitxml="$out/hcu.xml"
python -I "$ci/report.py" "$out/hcu.xml" --require test_hip_allocation test_mp_connector
python -I -m pytest -q --import-mode=importlib "$ci/test_qwen_mp.py" --junitxml="$out/qwen-mp.xml"
python -I "$ci/report.py" "$out/qwen-mp.xml" --require test_qwen_mp_cache_reuse

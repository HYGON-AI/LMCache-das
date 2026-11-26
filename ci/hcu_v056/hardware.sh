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
[[ "$(revision "$root/candidate")" == "$TEST_SHA" ]]
[[ "$(revision "$root/upstream")" == "$upstream" ]]
printf 'controller=%s\npr_head=%s\nmerge=%s\nupstream=%s\n' "$CONTROLLER_SHA" "$HEAD_SHA" "$TEST_SHA" "$upstream" > "$out/revisions.txt"

# Sanity checks only; the runner group and pod security/isolation are enforced
# by the platform outside the workflow, before *any* PR runs.
bash "$ci/check-pod.sh"
QWEN_MODEL="$(realpath -e -- "$QWEN_MODEL")"
[[ ( "$QWEN_MODEL" == /public/* || "$QWEN_MODEL" == /public4/* ) && -f "$QWEN_MODEL/config.json" ]] || { echo "::error::Invalid read-only Qwen3-8B model path"; exit 1; }
command -v hipcc >/dev/null || { echo "::error::The HCU image is missing hipcc"; exit 1; }
python3 -m venv --system-site-packages "$RUNNER_TEMP/lmcache-venv"
source "$RUNNER_TEMP/lmcache-venv/bin/activate"
export PIP_NO_INDEX=1
arch="$(python -I "$ci/runtime.py" arch)"
[[ "$arch" =~ ^gfx[0-9a-f]+$ ]] || { echo "::error::Expected one HIP architecture from preflight"; exit 1; }
export PYTORCH_ROCM_ARCH="$arch" TORCH_DONT_CHECK_COMPILER_ABI=1 CXX=hipcc BUILD_WITH_HIP=1
export HIPCC_COMPILE_FLAGS_APPEND=-DLMCACHE_DISABLE_STREAMING_IO=1 HIPCC_VERBOSE=1
python -m pip install --no-cache-dir --no-build-isolation --no-deps --ignore-installed "$root/upstream"
python -I "$ci/wheel.py" verify "$root/job-data/overlay-wheel" "$TEST_SHA" | tee "$out/wheel.json"
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

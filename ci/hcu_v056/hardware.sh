#!/usr/bin/env bash
set -euo pipefail
cd "${RUNNER_TEMP:?}"
root="${GITHUB_WORKSPACE:?}"
ci="$root/controller/ci/hcu_v056"
out="$root/job-data/reports"
mkdir -p "$out"
export CI_REPORT_DIR="$out"
revision() { git -c safe.directory="$1" -C "$1" rev-parse HEAD; }
[[ "$(revision "$root/controller")" == "$CONTROLLER_SHA" ]]
[[ "$(revision "$root/candidate")" == "$TEST_SHA" ]]
case "${HCU_SUITE:?}:${TEST_TP:?}" in
    unit:1|qwen:2) ;;
    *) echo "::error::Expected unit:1 or qwen:2"; exit 1 ;;
esac
printf 'controller=%s\npr_head=%s\nmerge=%s\n' "$CONTROLLER_SHA" "$HEAD_SHA" "$TEST_SHA" > "$out/revisions.txt"

# Sanity checks only; the runner group and pod security/isolation are enforced
# by the platform outside the workflow, before *any* PR runs.
bash "$ci/check-pod.sh"
if [[ "$HCU_SUITE" == qwen ]]; then
    QWEN_MODEL="$(realpath -e -- "$QWEN_MODEL")"
    [[ ( "$QWEN_MODEL" == /public/* || "$QWEN_MODEL" == /public4/* ) && -f "$QWEN_MODEL/config.json" ]] || { echo "::error::Invalid read-only Qwen3-8B model path"; exit 1; }
fi
python3 -m venv --system-site-packages "$RUNNER_TEMP/lmcache-venv"
source "$RUNNER_TEMP/lmcache-venv/bin/activate"
export PIP_NO_INDEX=1
arch="$(python -I "$ci/runtime.py" arch)"
[[ "$arch" =~ ^gfx[0-9a-f]+$ ]] || { echo "::error::Expected one HIP architecture from preflight"; exit 1; }
python -I "$ci/runtime.py" image-copy | tee -a "$out/revisions.txt"
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
if [[ "$HCU_SUITE" == unit ]]; then
    python -I -m pytest -q --import-mode=importlib "$ci/test_device.py" --junitxml="$out/hcu.xml"
    python -I "$ci/report.py" "$out/hcu.xml" --require test_hip_allocation test_mp_connector
    # Only use the exact source archive for its test fixtures. Never rebuild lmcache.
    LMCACHEPATH="$(python -I "$ci/upstream.py" /ci_public/lmcache-das/assets/upstream/lmcache-0.5.6-3e11e108.tar.gz "$RUNNER_TEMP/upstream-fixtures")"
    export LMCACHEPATH
    python -I "$ci/regression.py" "$root/candidate" "$root/controller" "$RUNNER_TEMP" "$out"
else
    python -I -m pytest -q --import-mode=importlib "$ci/test_qwen_mp.py" --junitxml="$out/qwen-mp.xml"
    python -I "$ci/report.py" "$out/qwen-mp.xml" --require test_qwen_mp_cache_reuse
fi

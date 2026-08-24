#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail

readonly CI_ROOT="/opt/ci/ci/hcu"
readonly CI_HELPER="${CI_ROOT}/ci.py"
readonly MODEL_HELPER="${CI_ROOT}/model-ci.py"
readonly MODEL_MANIFEST="${CI_ROOT}/model-test-manifest.json"
readonly COMPATIBILITY="${CI_ROOT}/compatibility.json"
readonly PATCH_MANIFEST="${CI_ROOT}/patch-manifest.json"
readonly SANDBOX_ROOT="/sandbox"
readonly SOURCE_INPUT="/input/source"
readonly UPSTREAM_INPUT="/input/upstream"
readonly TEST_TOOL_INPUT="/input/test-tool"
readonly SOURCE_ROOT="${SANDBOX_ROOT}/source"
readonly UPSTREAM_ROOT="${SANDBOX_ROOT}/upstream"
readonly TEST_ROOT="${SANDBOX_ROOT}/test-suite/tests"
readonly EXECUTE_ROOT="${SANDBOX_ROOT}/execute"
readonly TEST_TOOL_ROOT="${SANDBOX_ROOT}/test-tool"
readonly VENV_ROOT="${SANDBOX_ROOT}/venv"
readonly OUTPUT_ROOT="/output"
readonly WHEEL_ROOT="${OUTPUT_ROOT}/wheels"
readonly REPORT_ROOT="${OUTPUT_ROOT}/reports"
readonly LOG_ROOT="${OUTPUT_ROOT}/logs"
readonly STATE_ROOT="${OUTPUT_ROOT}/state"

CURRENT_STAGE="initializing"
PATCH_GATE_STARTED=0

mkdir -p \
    "${WHEEL_ROOT}" \
    "${REPORT_ROOT}" \
    "${LOG_ROOT}" \
    "${STATE_ROOT}" \
    "${SANDBOX_ROOT}/cache"

configure_environment() {
    unset SKIP_LMCACHE_HCU_PATCH
    export PYTHONPATH=""
    export HF_HUB_OFFLINE=1
    export TRANSFORMERS_OFFLINE=1
    export LMCACHE_TRACK_USAGE=false
    export NO_PROXY="127.0.0.1,localhost"
    export no_proxy="127.0.0.1,localhost"
    export HTTP_PROXY=""
    export HTTPS_PROXY=""
    export ALL_PROXY=""
    export http_proxy=""
    export https_proxy=""
    export all_proxy=""
    export XDG_CACHE_HOME="${SANDBOX_ROOT}/cache"
    export HF_HOME="${SANDBOX_ROOT}/cache/huggingface"
    export PIP_CACHE_DIR="${SANDBOX_ROOT}/cache/pip"
    export LMCACHEPATH="${UPSTREAM_ROOT}"
    export PYTHONHASHSEED=0
    export PYTHONDONTWRITEBYTECODE=1
}

write_synthetic_junit() {
    local stage="$1"
    local message="$2"
    local output="${REPORT_ROOT}/synthetic-${stage}.xml"
    if [[ ! -f "${output}" ]]; then
        python3 "${CI_HELPER}" synthetic-junit \
            --output "${output}" \
            --name "${stage}" \
            --message "${message}" \
            --details "See logs/${stage}.log" >/dev/null 2>&1 || true
    fi
}

run_phase() {
    local stage="$1"
    local phase_rc="$2"
    shift 2
    local log="${LOG_ROOT}/${stage}.log"
    CURRENT_STAGE="${stage}"
    printf '%s\n' "${stage}" >"${STATE_ROOT}/current-stage"
    set +e
    "$@" >"${log}" 2>&1
    local command_rc=$?
    set -e
    cat "${log}"
    printf '%s\n' "${command_rc}" >"${STATE_ROOT}/${stage}.rc"
    if (( command_rc != 0 )); then
        write_synthetic_junit "${stage}" \
            "CI stage failed with command exit code ${command_rc}"
        return "${phase_rc}"
    fi
    return 0
}

prepare_sandbox() {
    [[ ! -e "${SOURCE_ROOT}" ]]
    [[ ! -e "${UPSTREAM_ROOT}" ]]
    [[ ! -e "${VENV_ROOT}" ]]
    git clone --quiet --no-hardlinks "${SOURCE_INPUT}" "${SOURCE_ROOT}"
    git -C "${SOURCE_ROOT}" checkout --quiet --detach "${HCU_CI_SOURCE_SHA}"
    [[ "$(git -C "${SOURCE_ROOT}" rev-parse HEAD)" == "${HCU_CI_SOURCE_SHA}" ]]

    git clone --quiet --no-hardlinks "${UPSTREAM_INPUT}" "${UPSTREAM_ROOT}"
    git -C "${UPSTREAM_ROOT}" checkout --quiet --detach \
        fc031d471a566edb6d49a86c9116cc23cfb04111

    if [[ -d "${TEST_TOOL_INPUT}/.git" ]]; then
        git clone --quiet --no-hardlinks "${TEST_TOOL_INPUT}" "${TEST_TOOL_ROOT}"
        git -C "${TEST_TOOL_ROOT}" checkout --quiet --detach "${HCU_CI_TEST_TOOL_COMMIT}"
        [[ "$(git -C "${TEST_TOOL_ROOT}" rev-parse HEAD)" == "${HCU_CI_TEST_TOOL_COMMIT}" ]]
    fi

    python3 -m venv --system-site-packages "${VENV_ROOT}"
    # shellcheck disable=SC1091
    source "${VENV_ROOT}/bin/activate"
    [[ "$(python3 -c 'import sys; print(sys.prefix)')" == "${VENV_ROOT}" ]]
}

activate_environment() {
    [[ -x "${VENV_ROOT}/bin/python3" ]]
    [[ -f "${OUTPUT_ROOT}/environment.json" ]]
    # shellcheck disable=SC1091
    source "${VENV_ROOT}/bin/activate"
    DETECTED_ARCH="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["hcu_arch"])' "${OUTPUT_ROOT}/environment.json")"
    DETECTED_ABI="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["torch_cxx11_abi"])' "${OUTPUT_ROOT}/environment.json")"
    export DETECTED_ARCH DETECTED_ABI
    [[ "$(python3 -c 'import sys; print(sys.prefix)')" == "${VENV_ROOT}" ]]
}

remove_preinstalled_plugin() {
    if python3 -c 'import importlib.metadata as m; m.version("lmcache-hcu")' \
        >/dev/null 2>&1; then
        python3 -m pip uninstall -y lmcache-hcu
    fi
}

install_upstream() {
    env \
        BUILD_WITH_HIP=1 \
        CXX=hipcc \
        ROCM_HOME=/opt/dtk \
        ROCM_PATH=/opt/dtk \
        DTK_HOME=/opt/dtk \
        HCU_ARCH="${DETECTED_ARCH}" \
        PYTORCH_ROCM_ARCH="${DETECTED_ARCH}" \
        ENABLE_CXX11_ABI="${DETECTED_ABI}" \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        MAX_JOBS=8 \
        python3 -m pip install \
            --no-build-isolation \
            --no-deps \
            --editable "${UPSTREAM_ROOT}"
}

prepare_build_tree() {
    # setup.py currently discovers tests as a package. Preserve the complete
    # suite, then remove it only from this disposable build copy.
    [[ ! -e "${TEST_ROOT}" ]]
    mkdir -p "${TEST_ROOT}"
    cp -a "${SOURCE_ROOT}/tests/." "${TEST_ROOT}/"
    [[ -f "${TEST_ROOT}/pytest.ini" ]]
    rm -rf -- "${SOURCE_ROOT}/tests"
    [[ ! -e "${SOURCE_ROOT}/tests" ]]
}

build_wheel() {
    rm -f "${WHEEL_ROOT}"/*.whl
    env \
        BUILD_WITH_HIP=1 \
        CXX=hipcc \
        ROCM_HOME=/opt/dtk \
        ROCM_PATH=/opt/dtk \
        DTK_HOME=/opt/dtk \
        HCU_ARCH="${DETECTED_ARCH}" \
        PYTORCH_ROCM_ARCH="${DETECTED_ARCH}" \
        ENABLE_CXX11_ABI="${DETECTED_ABI}" \
        TORCH_DONT_CHECK_COMPILER_ABI=1 \
        MAX_JOBS=8 \
        python3 -m pip wheel "${SOURCE_ROOT}" \
            --no-build-isolation \
            --no-deps \
            --wheel-dir "${WHEEL_ROOT}"
}

install_current_wheel() {
    local wheel
    wheel="$(find "${WHEEL_ROOT}" -maxdepth 1 -type f -name '*.whl' -print -quit)"
    [[ -n "${wheel}" ]]
    python3 -m pip install --no-deps --force-reinstall "${wheel}"
    python3 -m pip check
}

execute_repeat() {
    local repeat="$1"
    local stage="test-repeat-${repeat}"
    local log="${LOG_ROOT}/${stage}.log"
    CURRENT_STAGE="${stage}"
    printf '%s\n' "${CURRENT_STAGE}" >"${STATE_ROOT}/current-stage"
    set +e
    python3 "${CI_HELPER}" execute \
        --inventory "${OUTPUT_ROOT}/test-inventory.json" \
        --config "${TEST_ROOT}/pytest.ini" \
        --execute-dir "${EXECUTE_ROOT}" \
        --junit "${REPORT_ROOT}/junit-repeat-${repeat}.xml" \
        --rc-output "${STATE_ROOT}/test-repeat-${repeat}.rc" \
        >"${log}" 2>&1
    local command_rc=$?
    set -e
    cat "${log}"
    if (( command_rc != 0 )) && [[ ! -f "${REPORT_ROOT}/junit-repeat-${repeat}.xml" ]]; then
        write_synthetic_junit "${stage}" \
            "pytest did not produce a JUnit report (exit ${command_rc})"
    fi
    # Aggregate all requested repetitions before returning the test result.
    return 0
}

best_effort_patch_cleanup() {
    if [[ ! -d "${UPSTREAM_ROOT}/.git" ]]; then
        return 0
    fi
    if [[ -f "${OUTPUT_ROOT}/patch-report.json" ]] && \
        python3 -c 'import json,sys; sys.exit(0 if json.load(open(sys.argv[1])).get("status") == "passed" else 1)' \
            "${OUTPUT_ROOT}/patch-report.json"; then
        python3 "${CI_HELPER}" patch-cleanup \
            --report "${OUTPUT_ROOT}/patch-report.json"
        return $?
    fi
    local status
    status="$(git -C "${UPSTREAM_ROOT}" status --porcelain=v1 --untracked-files=all)"
    if grep -E 'lmcache/__init__\.py|lmcache/integration/vllm/lmcache_connector_v1\.py|\.bak\.' \
        <<<"${status}" >/dev/null; then
        return 1
    fi
    return 0
}

phase_initialize() {
    run_phase sandbox 3 prepare_sandbox || return $?
    run_phase remove-preinstalled-plugin 3 remove_preinstalled_plugin || return $?
    run_phase environment 3 \
        python3 "${CI_HELPER}" probe \
            --compatibility "${COMPATIBILITY}" \
            --expected-arch "${HCU_CI_HCU_ARCH:-}" \
            --expected-device-count "${HCU_CI_EXPECTED_DEVICE_COUNT:-0}" \
            --output "${OUTPUT_ROOT}/environment.json" || return $?
    activate_environment
    run_phase upstream-source 3 \
        python3 "${CI_HELPER}" verify-upstream \
            --compatibility "${COMPATIBILITY}" \
            --source "${UPSTREAM_ROOT}" \
            --output "${OUTPUT_ROOT}/upstream-source.json" || return $?
    run_phase upstream-install 3 install_upstream || return $?
}

phase_build() {
    activate_environment
    run_phase build-tree 4 prepare_build_tree || return $?
    run_phase build-wheel 4 build_wheel || return $?
    run_phase wheel-artifact 5 \
        python3 "${CI_HELPER}" verify-wheel \
            --wheel-dir "${WHEEL_ROOT}" \
            --sha "${HCU_CI_SOURCE_SHA}" \
            --output "${OUTPUT_ROOT}/wheel-report.json" || return $?
    run_phase wheel-install 5 install_current_wheel || return $?
    run_phase installed-package 5 \
        python3 "${CI_HELPER}" verify-install \
            --sha "${HCU_CI_SOURCE_SHA}" \
            --source "${SOURCE_ROOT}" \
            --venv "${VENV_ROOT}" \
            --output "${OUTPUT_ROOT}/installed-package.json" || return $?
}

phase_prepare_tests() {
    activate_environment
    PATCH_GATE_STARTED=1
    run_phase source-patch 6 \
        python3 "${CI_HELPER}" patch-gate \
            --manifest "${PATCH_MANIFEST}" \
            --upstream "${UPSTREAM_ROOT}" \
            --output "${OUTPUT_ROOT}/patch-report.json" || return $?
    run_phase collect 7 \
        python3 "${CI_HELPER}" discover \
            --tests "${TEST_ROOT}" \
            --config "${TEST_ROOT}/pytest.ini" \
            --execute-dir "${EXECUTE_ROOT}" \
            --output "${OUTPUT_ROOT}/test-inventory.json" || return $?
    if [[ "${HCU_CI_MODEL_PROFILE}" != "framework" ]]; then
        run_phase model-tool 7 \
            python3 "${MODEL_HELPER}" verify-tool \
                --manifest "${MODEL_MANIFEST}" \
                --profile "${HCU_CI_MODEL_PROFILE}" \
                --tool "${TEST_TOOL_ROOT}" \
                --output "${OUTPUT_ROOT}/model-tool.json" || return $?
    fi
}

phase_test() {
    activate_environment
    [[ -f "${OUTPUT_ROOT}/patch-report.json" ]]
    [[ -f "${OUTPUT_ROOT}/test-inventory.json" ]]
    local repeat
    for ((repeat = 1; repeat <= HCU_CI_REPEAT; repeat++)); do
        execute_repeat "${repeat}"
    done
    run_phase aggregate 8 \
        python3 "${CI_HELPER}" aggregate \
            --inventory "${OUTPUT_ROOT}/test-inventory.json" \
            --repeat "${HCU_CI_REPEAT}" \
            --junit-dir "${REPORT_ROOT}" \
            --state-dir "${STATE_ROOT}" \
            --output "${OUTPUT_ROOT}/test-summary.json" \
            --markdown "${OUTPUT_ROOT}/summary.md" || return $?
    local -a tool_argument=()
    if [[ "${HCU_CI_MODEL_PROFILE}" != "framework" ]]; then
        tool_argument+=(--tool "${TEST_TOOL_ROOT}")
    fi
    run_phase model-tests 8 \
        python3 "${MODEL_HELPER}" run \
            --manifest "${MODEL_MANIFEST}" \
            --profile "${HCU_CI_MODEL_PROFILE}" \
            --runner "${HCU_CI_RUNNER_KIND}" \
            --repeat "${HCU_CI_REPEAT}" \
            --output "${OUTPUT_ROOT}" \
            "${tool_argument[@]}"
}

phase_cleanup() {
    configure_environment
    if [[ -x "${VENV_ROOT}/bin/python3" ]]; then
        # shellcheck disable=SC1091
        source "${VENV_ROOT}/bin/activate"
    fi
    local cleanup_rc=0
    if [[ -f "${OUTPUT_ROOT}/patch-report.json" ]] || \
       [[ -d "${UPSTREAM_ROOT}/.git" ]]; then
        set +e
        best_effort_patch_cleanup >"${LOG_ROOT}/patch-cleanup.log" 2>&1
        cleanup_rc=$?
        set -e
    else
        printf '%s\n' 'Patch phase was not reached; nothing to restore.' \
            >"${LOG_ROOT}/patch-cleanup.log"
    fi
    cat "${LOG_ROOT}/patch-cleanup.log"
    printf '%s\n' "${cleanup_rc}" >"${STATE_ROOT}/patch-cleanup.rc"
    if (( cleanup_rc != 0 )); then
        write_synthetic_junit patch-cleanup \
            "Patch cleanup failed with exit code ${cleanup_rc}"
        return 10
    fi
    return 0
}

main() {
    local phase="${1:-}"
    configure_environment
    case "${phase}" in
        initialize) phase_initialize ;;
        build) phase_build ;;
        prepare-tests) phase_prepare_tests ;;
        test) phase_test ;;
        cleanup) phase_cleanup ;;
        *)
            printf 'Usage: %s {initialize|build|prepare-tests|test|cleanup}\n' "$0" >&2
            return 2
            ;;
    esac
}

main "$@"

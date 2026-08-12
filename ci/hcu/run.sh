#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail
umask 077

readonly CONTROLLER_ROOT="${HCU_CI_CONTROLLER:?HCU_CI_CONTROLLER is required}"
readonly CHECKOUT_ROOT="${HCU_CI_CHECKOUT:?HCU_CI_CHECKOUT is required}"
readonly HOST_HELPER="${CONTROLLER_ROOT}/ci/hcu/host.py"
readonly PROFILE="${HCU_CI_PROFILE:-pr}"
readonly REPEAT="${HCU_CI_REPEAT:-1}"
readonly RUN_ID="${HCU_CI_RUN_ID:-local-$(date -u +%Y%m%d%H%M%S)}"
readonly ATTEMPT="${HCU_CI_ATTEMPT:-1}"
readonly REPOSITORY="${GITHUB_REPOSITORY:-HYGON-AI/LMCache-das}"
readonly BASE_IMAGE="${HCU_CI_BASE_IMAGE:-}"
readonly SHARED_ROOT="${HCU_CI_SHARED_ROOT:-/ci_public/lmcache-das}"
readonly UPSTREAM_ASSET="${HCU_CI_UPSTREAM_SOURCE:-/ci_public/lmcache-das/assets/upstream/LMCache/v0.3.13/fc031d471a566edb6d49a86c9116cc23cfb04111}"
readonly GPU_LOCK="${HCU_CI_GPU_LOCK:-/tmp/hcu-ci-gpu-locks/gpu-0.lock}"
readonly RUNNER_TEMP_ROOT="${RUNNER_TEMP:-/tmp}"
readonly CONTAINER_MEMORY="${HCU_CI_CONTAINER_MEMORY:-56g}"
readonly CONTAINER_CPUS="${HCU_CI_CONTAINER_CPUS:-32}"
readonly OUTPUT_LIMIT="${HCU_CI_OUTPUT_LIMIT:-3g}"
readonly COMMAND="${1:-}"
readonly JOB_STATUS_FILE="${HCU_CI_JOB_STATUS_FILE:-}"

SOURCE_SHA="unknown"
CONTROLLER_SHA="unknown"
RUN_KEY="unknown"
WORK_BASE="${RUNNER_TEMP_ROOT}/lmcache-hcu"
WORK_ROOT=""
TRUSTED_STATE_ROOT=""
TRUSTED_SPOOL=""
HOST_LOG_ROOT=""
REPORT_ROOT=""
STATE_FILE=""
CONTAINER_NAME=""
LEASE_CONTAINER=""
SOURCE_STATUS=""
SOURCE_STATUS_SHA256=""

safe_token() {
    [[ "$1" =~ ^[A-Za-z0-9_.-]+$ ]]
}

validate_job_status_file() {
    [[ -n "${JOB_STATUS_FILE}" ]]
    local expected
    expected="$(readlink -m -- "${RUNNER_TEMP_ROOT}/lmcache-hcu-job-${RUN_ID}-${ATTEMPT}.status")"
    [[ "$(readlink -m -- "${JOB_STATUS_FILE}")" == "${expected}" ]]
}

set_job_status() {
    local status="$1"
    validate_job_status_file
    local temporary="${JOB_STATUS_FILE}.tmp"
    printf '%s\n' "${status}" >"${temporary}"
    chmod 0600 "${temporary}"
    mv -f -- "${temporary}" "${JOB_STATUS_FILE}"
}

ensure_within() {
    local path="$1" parent="$2"
    local resolved_path resolved_parent
    resolved_path="$(readlink -m -- "${path}")"
    resolved_parent="$(readlink -m -- "${parent}")"
    [[ "${resolved_path}" == "${resolved_parent}/"* ]]
}

resolve_context() {
    SOURCE_SHA="$(git -C "${CHECKOUT_ROOT}" rev-parse HEAD)"
    CONTROLLER_SHA="$(git -C "${CONTROLLER_ROOT}" rev-parse HEAD)"
    [[ "${SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]]
    [[ "${CONTROLLER_SHA}" =~ ^[0-9a-f]{40}$ ]]
    if [[ "${HCU_CI_SOURCE_SHA_HINT:-unknown}" != "unknown" && \
          "${HCU_CI_SOURCE_SHA_HINT}" != "${SOURCE_SHA}" ]]; then
        printf 'Source checkout SHA %s differs from expected %s\n' \
            "${SOURCE_SHA}" "${HCU_CI_SOURCE_SHA_HINT}" >&2
        return 1
    fi
    SOURCE_STATUS="$(git -C "${CHECKOUT_ROOT}" status --porcelain=v1 --untracked-files=all)"
    SOURCE_STATUS_SHA256="$(printf '%s' "${SOURCE_STATUS}" | sha256sum | awk '{print $1}')"
    RUN_KEY="${RUN_ID}-${ATTEMPT}-${SOURCE_SHA:0:12}"
    safe_token "${RUN_ID}" && safe_token "${ATTEMPT}" && safe_token "${RUN_KEY}"
    WORK_ROOT="${WORK_BASE}/${RUN_KEY}"
    ensure_within "${WORK_ROOT}" "${WORK_BASE}"
    TRUSTED_STATE_ROOT="${WORK_ROOT}/trusted-state"
    TRUSTED_SPOOL="${WORK_ROOT}/trusted-spool"
    HOST_LOG_ROOT="${TRUSTED_SPOOL}/logs"
    REPORT_ROOT="${TRUSTED_SPOOL}/reports"
    STATE_FILE="${TRUSTED_STATE_ROOT}/state.json"
    CONTAINER_NAME="lmcache-hcu-ci-${RUN_KEY}"
    LEASE_CONTAINER="lmcache-hcu-lease-${RUN_KEY}"
}

state_metadata_args() {
    printf '%s\0' \
        --state "${STATE_FILE}" \
        --repository "${REPOSITORY}" \
        --profile "${PROFILE}" \
        --run-id "${RUN_ID}" \
        --attempt "${ATTEMPT}" \
        --sha "${SOURCE_SHA}" \
        --controller-sha "${CONTROLLER_SHA}" \
        --base-image "${BASE_IMAGE}" \
        --run-key "${RUN_KEY}" \
        --repeat "${REPEAT}" \
        --checkout-status-sha256 "${SOURCE_STATUS_SHA256}"
}

load_state_metadata_args() {
    STATE_ARGS=()
    local value
    while IFS= read -r -d '' value; do
        STATE_ARGS+=("${value}")
    done < <(state_metadata_args)
}

write_host_synthetic() {
    local stage="$1" message="$2"
    local passed="${3:-false}"
    mkdir -p "${REPORT_ROOT}"
    local output="${REPORT_ROOT}/synthetic-${stage}.xml"
    [[ -f "${output}" ]] && return 0
    local -a passed_arg=()
    [[ "${passed}" == "true" ]] && passed_arg+=(--passed)
    python3 "${HOST_HELPER}" synthetic \
        --output "${output}" --name "${stage}" --message "${message}" \
        "${passed_arg[@]}" || true
}

check_pr_revision() {
    if [[ "${PROFILE}" != "pr" ]]; then
        return 0
    fi
    local -a parents=()
    [[ "${HCU_CI_ISOLATED_RUNNER:-}" == "true" ]]
    [[ "${HCU_CI_EXPECTED_BASE_SHA:-}" =~ ^[0-9a-f]{40}$ ]]
    [[ "${HCU_CI_EXPECTED_HEAD_SHA:-}" =~ ^[0-9a-f]{40}$ ]]
    read -r -a parents <<<"$(git -C "${CHECKOUT_ROOT}" show -s --format='%P' "${SOURCE_SHA}")"
    (( ${#parents[@]} == 2 ))
    [[ "${parents[0]}" == "${HCU_CI_EXPECTED_BASE_SHA}" ]]
    [[ "${parents[1]}" == "${HCU_CI_EXPECTED_HEAD_SHA}" ]]
}

device_arguments() {
    local render_link render_device card_link card_device
    render_link="$({ find /dev/dri/by-path -maxdepth 1 -type l -name '*-render' -print; } | LC_ALL=C sort | sed -n '1p')"
    [[ -n "${render_link}" ]]
    card_link="${render_link%-render}-card"
    [[ -L "${card_link}" ]]
    render_device="$(readlink -f -- "${render_link}")"
    card_device="$(readlink -f -- "${card_link}")"
    [[ -c /dev/kfd && -c "${render_device}" && -c "${card_device}" ]]
    printf '%s\0' --device /dev/kfd --device "${render_device}" --device "${card_device}"
}

host_preflight() {
    command -v docker >/dev/null
    command -v git >/dev/null
    command -v python3 >/dev/null
    command -v sha256sum >/dev/null
    python3 -c 'import sys; assert sys.version_info >= (3, 6)'
    [[ "${CONTAINER_MEMORY}" =~ ^[1-9][0-9]*[gGmM]$ ]]
    [[ "${CONTAINER_CPUS}" =~ ^[1-9][0-9]*$ ]]
    [[ "${OUTPUT_LIMIT}" =~ ^[1-9][0-9]*[gGmM]$ ]]
    [[ "${BASE_IMAGE}" =~ ^[^[:space:]]+@sha256:[0-9a-f]{64}$ ]]
    [[ -d "${CHECKOUT_ROOT}/.git" && -d "${CONTROLLER_ROOT}/.git" ]]
    [[ -d "${UPSTREAM_ASSET}/.git" ]]
    [[ "$(readlink -m -- "${UPSTREAM_ASSET}")" == "/ci_public/lmcache-das/assets/upstream/LMCache/v0.3.13/"* ]]
    [[ -e /dev/kfd && -d /dev/dri && -d /opt/hyhal ]]
    device_arguments >/dev/null
    [[ "$(readlink -m -- "${SHARED_ROOT}")" == "/ci_public/lmcache-das" ]]
    [[ "$(readlink -m -- "${GPU_LOCK}")" == "/tmp/hcu-ci-gpu-locks/"* ]]
    mkdir -p "${SHARED_ROOT}" "$(dirname "${GPU_LOCK}")"
    docker image inspect "${BASE_IMAGE}" >/dev/null 2>&1 || docker pull "${BASE_IMAGE}"
    docker image inspect --format '{{range .RepoDigests}}{{println .}}{{end}}' \
        "${BASE_IMAGE}" | grep -Fxq "${BASE_IMAGE}"
}

validate_configuration() {
    docker run --rm --network none --entrypoint python3 \
        --mount "type=bind,src=${CONTROLLER_ROOT},dst=/opt/ci,readonly" \
        --mount "type=bind,src=${CHECKOUT_ROOT},dst=/input/source,readonly" \
        --mount "type=bind,src=${UPSTREAM_ASSET},dst=/input/upstream,readonly" \
        "${BASE_IMAGE}" /opt/ci/ci/hcu/ci.py validate \
            --compatibility /opt/ci/ci/hcu/compatibility.json \
            --patch-manifest /opt/ci/ci/hcu/patch-manifest.json \
            --base-image "${BASE_IMAGE}" --profile "${PROFILE}" \
            --repeat "${REPEAT}" --checkout /input/source --upstream /input/upstream
}

start_gpu_lease() {
    # Hold the flock in a trusted host process for the complete Actions job. A
    # normal shell fd would be released between workflow steps.
    nohup python3 "${HOST_HELPER}" hold-lock \
        --lock "${GPU_LOCK}" \
        --ready "${TRUSTED_STATE_ROOT}/lease-ready" \
        --timeout 300 \
        >"${HOST_LOG_ROOT}/gpu-lease-process.log" 2>&1 &
    local lease_pid=$!
    printf '%s\n' "${lease_pid}" >"${TRUSTED_STATE_ROOT}/lease.pid"
    local attempt
    for attempt in $(seq 1 300); do
        if [[ -f "${TRUSTED_STATE_ROOT}/lease-ready" ]]; then
            local ready_pid
            ready_pid="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' "${TRUSTED_STATE_ROOT}/lease-ready")"
            [[ "${ready_pid}" == "${lease_pid}" ]]
            kill -0 "${lease_pid}"
            return 0
        fi
        kill -0 "${lease_pid}" 2>/dev/null || return 1
        sleep 1
    done
    return 1
}

device_and_group_arguments() {
    local -a device_args=()
    local value index gid
    while IFS= read -r -d '' value; do device_args+=("${value}"); done < <(device_arguments)
    (( ${#device_args[@]} == 6 ))
    for value in "${device_args[@]}"; do printf '%s\0' "${value}"; done
    while IFS= read -r gid; do
        [[ -n "${gid}" ]] && printf '%s\0' --group-add "${gid}"
    done < <({ printf '%s\n' "$(stat -c '%g' /dev/kfd)"; for ((index=3; index<${#device_args[@]}; index+=2)); do stat -c '%g' "${device_args[index]}"; done; } | sort -nu)
}

verify_resources() {
    [[ "$(docker inspect -f '{{.State.Running}}' "${CONTAINER_NAME}" 2>/dev/null)" == "true" ]]
    local lease_pid
    lease_pid="$(cat "${TRUSTED_STATE_ROOT}/lease.pid")"
    [[ "${lease_pid}" =~ ^[1-9][0-9]*$ ]]
    [[ -f "${TRUSTED_STATE_ROOT}/lease-ready" ]]
    [[ "$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' "${TRUSTED_STATE_ROOT}/lease-ready")" == "${lease_pid}" ]]
    kill -0 "${lease_pid}"
}

start_test_container() {
    local -a resource_args=()
    local value uid gid
    while IFS= read -r -d '' value; do resource_args+=("${value}"); done < <(device_and_group_arguments)
    uid="$(id -u)"; gid="$(id -g)"
    docker run -d --name "${CONTAINER_NAME}" \
        --label lmcache-hcu-ci.run-key="${RUN_KEY}" \
        --network none --read-only --shm-size 16g --user "${uid}:${gid}" \
        --tmpfs "/sandbox:rw,exec,nosuid,nodev,size=32g,uid=${uid},gid=${gid},mode=0750" \
        --tmpfs "/tmp:rw,exec,nosuid,nodev,size=4g,uid=${uid},gid=${gid},mode=1770" \
        --tmpfs "/output:rw,nosuid,nodev,size=${OUTPUT_LIMIT},uid=${uid},gid=${gid},mode=0750" \
        --memory "${CONTAINER_MEMORY}" --memory-swap "${CONTAINER_MEMORY}" --cpus "${CONTAINER_CPUS}" \
        --cap-drop ALL --security-opt no-new-privileges --pids-limit 4096 \
        --ulimit nofile=65536:65536 --ulimit fsize=1073741824:1073741824 --log-driver none \
        "${resource_args[@]}" \
        --mount "type=bind,src=/opt/hyhal,dst=/opt/hyhal,readonly" \
        --mount "type=bind,src=${CONTROLLER_ROOT},dst=/opt/ci,readonly" \
        --mount "type=bind,src=${CHECKOUT_ROOT},dst=/input/source,readonly" \
        --mount "type=bind,src=${UPSTREAM_ASSET},dst=/input/upstream,readonly" \
        --env HIP_VISIBLE_DEVICES=0 --env CUDA_VISIBLE_DEVICES=0 --env ROCR_VISIBLE_DEVICES=0 \
        --env HCU_CI_PROFILE="${PROFILE}" --env HCU_CI_REPEAT="${REPEAT}" \
        --env HCU_CI_SOURCE_SHA="${SOURCE_SHA}" --env HCU_CI_HCU_ARCH="${HCU_CI_HCU_ARCH:-}" \
        --entrypoint tail "${BASE_IMAGE}" -f /dev/null >/dev/null
    verify_resources
}

record_phase() {
    python3 "${HOST_HELPER}" state-record --state "${STATE_FILE}" \
        --phase "$1" --command-rc "$2" --mapped-rc "$3"
}

run_container_phase() {
    local phase="$1" mapped_rc="$2"
    if ! verify_resources; then
        python3 "${HOST_HELPER}" state-note-failure --state "${STATE_FILE}" \
            --phase "${phase}" --mapped-rc 3 --message 'CI container or GPU lease is unavailable' || true
        return 3
    fi
    if ! python3 "${HOST_HELPER}" state-require --state "${STATE_FILE}" --phase "${phase}"; then
        python3 "${HOST_HELPER}" state-note-failure --state "${STATE_FILE}" \
            --phase "${phase}" --mapped-rc 9 --message 'CI phase predecessor validation failed' || true
        return 9
    fi
    local log="${HOST_LOG_ROOT}/${phase}.log"
    mkdir -p "${HOST_LOG_ROOT}"
    set +e
    timeout --signal=TERM --kill-after=60s 3000s \
        docker exec "${CONTAINER_NAME}" bash /opt/ci/ci/hcu/run-tests.sh "${phase}" 2>&1 | \
        python3 "${HOST_HELPER}" capture-log --output "${log}" --limit 134217728
    local -a statuses=("${PIPESTATUS[@]}")
    set -e
    local command_rc="${statuses[0]}" capture_rc="${statuses[1]}" final_rc=0
    if (( command_rc != 0 )); then
        case "${command_rc}" in
            2|3|4|5|6|7|8|9|10|11|124|130|143) final_rc="${command_rc}" ;;
            *) final_rc="${mapped_rc}" ;;
        esac
    elif (( capture_rc != 0 )); then
        final_rc=9
        write_host_synthetic "${phase}-log" "${phase} log capture failed"
    fi
    record_phase "${phase}" "${command_rc}" "${final_rc}"
    return "${final_rc}"
}

safe_import_output() {
    local uid gid
    uid="$(id -u)"; gid="$(id -g)"
    # The untrusted container is paused before import, freezing its output tree.
    # A trusted, device-less helper joins only its PID namespace and reads the
    # bounded /output tmpfs through /proc/1/root. SYS_PTRACE is needed because
    # the target runs as the unprivileged runner uid; the helper has no network,
    # source checkout, HCU device, Docker socket, or shared-results mount.
    docker run --rm --network none --read-only --entrypoint python3 \
        --user "${uid}:${gid}" --pid "container:${CONTAINER_NAME}" \
        --cap-drop ALL --cap-add SYS_PTRACE --security-opt no-new-privileges \
        --mount "type=bind,src=${CONTROLLER_ROOT},dst=/opt/ci,readonly" \
        --mount "type=bind,src=${TRUSTED_SPOOL},dst=/trusted" \
        "${BASE_IMAGE}" /opt/ci/ci/hcu/host.py safe-import \
            --source /proc/1/root/output --preserve-source-path --destination /trusted
}

verify_checkout_unchanged() {
    local after
    after="$(git -C "${CHECKOUT_ROOT}" status --porcelain=v1 --untracked-files=all)"
    [[ "${after}" == "${SOURCE_STATUS}" ]]
    [[ "$(git -C "${CHECKOUT_ROOT}" rev-parse HEAD)" == "${SOURCE_SHA}" ]]
}

verify_container_absent() {
    local name="$1" output rc
    set +e
    output="$(docker inspect "${name}" 2>&1)"; rc=$?
    set -e
    if (( rc == 0 )); then return 1; fi
    grep -Fqi 'no such object' <<<"${output}"
}

cleanup_legacy_lease_container() {
    local output rc labels
    set +e
    output="$(docker inspect "${LEASE_CONTAINER}" 2>&1)"; rc=$?
    set -e
    if (( rc != 0 )); then
        grep -Fqi 'no such object' <<<"${output}"
        return $?
    fi
    labels="$(docker inspect -f '{{ index .Config.Labels "lmcache-hcu-ci.run-key" }}' "${LEASE_CONTAINER}")" || return 1
    [[ "${labels}" == "${RUN_KEY}" ]] || return 1
    docker rm -f "${LEASE_CONTAINER}" >/dev/null
}

remove_named_container() {
    local name="$1" output rc
    set +e
    output="$(docker inspect "${name}" 2>&1)"; rc=$?
    set -e
    if (( rc == 0 )); then
        local label
        label="$(docker inspect -f '{{ index .Config.Labels "lmcache-hcu-ci.run-key" }}' "${name}")" || return 1
        [[ "${label}" == "${RUN_KEY}" ]] || return 1
        docker rm -f "${name}" >/dev/null
        return $?
    fi
    grep -Fqi 'no such object' <<<"${output}" && return 0
    return 1
}

stop_gpu_lease() {
    local pid_file="${TRUSTED_STATE_ROOT}/lease.pid"
    [[ -f "${pid_file}" ]] || return 0
    local lease_pid
    lease_pid="$(cat "${pid_file}")"
    [[ "${lease_pid}" =~ ^[1-9][0-9]*$ ]] || return 1
    if [[ -f "${TRUSTED_STATE_ROOT}/lease-ready" ]]; then
        local ready_pid
        ready_pid="$(python3 -c 'import json,sys; print(json.load(open(sys.argv[1]))["pid"])' "${TRUSTED_STATE_ROOT}/lease-ready")" || return 1
        [[ "${ready_pid}" == "${lease_pid}" ]] || return 1
    fi
    local command_line
    command_line="$(tr '\0' ' ' <"/proc/${lease_pid}/cmdline" 2>/dev/null || true)"
    if kill -0 "${lease_pid}" 2>/dev/null && \
       [[ "${command_line}" != *"${HOST_HELPER} hold-lock"* ]]; then
        return 1
    fi
    if kill -0 "${lease_pid}" 2>/dev/null; then
        kill "${lease_pid}" 2>/dev/null || return 1
        local attempt
        for attempt in $(seq 1 50); do
            kill -0 "${lease_pid}" 2>/dev/null || break
            sleep 0.1
        done
        kill -0 "${lease_pid}" 2>/dev/null && return 1
    fi
    rm -f -- "${pid_file}" "${TRUSTED_STATE_ROOT}/lease-ready"
}

phase_initialize() {
    [[ ! -e "${WORK_ROOT}" ]]
    validate_job_status_file
    [[ ! -e "${JOB_STATUS_FILE}" && ! -L "${JOB_STATUS_FILE}" ]]
    set_job_status failed
    mkdir -p "${TRUSTED_STATE_ROOT}" "${HOST_LOG_ROOT}" "${REPORT_ROOT}"
    chmod 0700 "${WORK_ROOT}" "${TRUSTED_STATE_ROOT}"
    write_host_synthetic bootstrap "LMCache-HCU trusted controller initialized" true
    load_state_metadata_args
    if ! check_pr_revision >"${HOST_LOG_ROOT}/revision.log" 2>&1; then
        write_host_synthetic revision "PR revision verification failed"
        python3 "${HOST_HELPER}" state-bootstrap-failure "${STATE_ARGS[@]}" \
            --mapped-rc 2 --message 'PR revision verification failed'
        return 2
    fi
    if ! host_preflight >"${HOST_LOG_ROOT}/host-preflight.log" 2>&1; then
        write_host_synthetic host-preflight "Host or image preflight failed"
        python3 "${HOST_HELPER}" state-bootstrap-failure "${STATE_ARGS[@]}" \
            --mapped-rc 3 --message 'host or image preflight failed'
        return 3
    fi
    if ! validate_configuration >"${HOST_LOG_ROOT}/configuration.log" 2>&1; then
        write_host_synthetic configuration "CI configuration validation failed"
        python3 "${HOST_HELPER}" state-bootstrap-failure "${STATE_ARGS[@]}" \
            --mapped-rc 2 --message 'CI configuration validation failed'
        return 2
    fi
    python3 "${HOST_HELPER}" state-init "${STATE_ARGS[@]}"
    if ! start_gpu_lease >"${HOST_LOG_ROOT}/gpu-lease.log" 2>&1; then
        python3 "${HOST_HELPER}" state-abort --state "${STATE_FILE}" \
            --phase initialize --mapped-rc 3 --message 'GPU lease acquisition failed'
        return 3
    fi
    if ! start_test_container >"${HOST_LOG_ROOT}/container-start.log" 2>&1; then
        python3 "${HOST_HELPER}" state-abort --state "${STATE_FILE}" \
            --phase initialize --mapped-rc 3 --message 'test container creation failed'
        return 3
    fi
    local phase_rc=0
    set +e
    run_container_phase initialize 3
    phase_rc=$?
    set -e
    if (( phase_rc == 0 )); then set_job_status passed; fi
    return "${phase_rc}"
}

phase_build() {
    set_job_status failed
    load_state_metadata_args
    if ! python3 "${HOST_HELPER}" state-check "${STATE_ARGS[@]}"; then
        python3 "${HOST_HELPER}" state-note-failure --state "${STATE_FILE}" \
            --phase build --mapped-rc 9 --message 'trusted state validation failed' || true
        return 9
    fi
    local phase_rc=0
    set +e
    run_container_phase build 5
    phase_rc=$?
    set -e
    if (( phase_rc == 0 )); then set_job_status passed; fi
    return "${phase_rc}"
}

phase_prepare_tests() {
    set_job_status failed
    load_state_metadata_args
    if ! python3 "${HOST_HELPER}" state-check "${STATE_ARGS[@]}"; then
        python3 "${HOST_HELPER}" state-note-failure --state "${STATE_FILE}" \
            --phase prepare-tests --mapped-rc 9 --message 'trusted state validation failed' || true
        return 9
    fi
    local phase_rc=0
    set +e
    run_container_phase prepare-tests 7
    phase_rc=$?
    set -e
    if (( phase_rc == 0 )); then set_job_status passed; fi
    return "${phase_rc}"
}

phase_test() {
    set_job_status failed
    load_state_metadata_args
    if ! python3 "${HOST_HELPER}" state-check "${STATE_ARGS[@]}"; then
        python3 "${HOST_HELPER}" state-note-failure --state "${STATE_FILE}" \
            --phase test --mapped-rc 9 --message 'trusted state validation failed' || true
        return 9
    fi
    local phase_rc=0
    set +e
    run_container_phase test 8
    phase_rc=$?
    set -e
    if (( phase_rc == 0 )); then set_job_status passed; fi
    return "${phase_rc}"
}

phase_finalize() {
    local cleanup_rc=0 import_rc=0 primary_rc=2 initialize_status="missing"
    local test_container_absent=false legacy_lease_absent=false container_present=false
    if [[ ! -e "${WORK_ROOT}" ]]; then
        mkdir -p "${TRUSTED_STATE_ROOT}" "${HOST_LOG_ROOT}" "${REPORT_ROOT}"
        chmod 0700 "${WORK_ROOT}" "${TRUSTED_STATE_ROOT}"
    else
        [[ ! -L "${WORK_ROOT}" && ! -L "${TRUSTED_SPOOL}" ]]
        mkdir -p "${HOST_LOG_ROOT}" "${REPORT_ROOT}"
    fi
    if [[ -f "${STATE_FILE}" ]]; then
        primary_rc="$(python3 "${HOST_HELPER}" state-primary --state "${STATE_FILE}")" || primary_rc=9
        initialize_status="$(python3 "${HOST_HELPER}" state-phase --state "${STATE_FILE}" --phase initialize)" || initialize_status="missing"
    fi
    if [[ "${initialize_status}" == "0" ]] && docker inspect "${CONTAINER_NAME}" >/dev/null 2>&1; then
        container_present=true
        set +e
        docker exec "${CONTAINER_NAME}" bash /opt/ci/ci/hcu/run-tests.sh cleanup \
            >"${HOST_LOG_ROOT}/container-cleanup.log" 2>&1
        local patch_cleanup_rc=$?
        set -e
        (( patch_cleanup_rc == 0 )) || cleanup_rc=10
    fi
    if [[ "${container_present}" == "true" ]]; then
        set +e
        docker pause "${CONTAINER_NAME}" >"${HOST_LOG_ROOT}/container-pause.log" 2>&1
        local pause_rc=$?
        set -e
        if (( pause_rc == 0 )); then
            safe_import_output >"${HOST_LOG_ROOT}/import-output.log" 2>&1 || import_rc=9
        else
            import_rc=9
        fi
    fi
    if remove_named_container "${CONTAINER_NAME}" && \
       verify_container_absent "${CONTAINER_NAME}"; then
        test_container_absent=true
    else
        cleanup_rc=10
        write_host_synthetic cleanup-container \
            "Test container could not be removed; GPU lease remains held and runner reset is required"
    fi
    # Never release the GPU lease while a test container may still own the
    # device. A failed removal deliberately leaves the lease process alive so
    # another job cannot overlap; the isolated runner must then be reset.
    if [[ "${test_container_absent}" == "true" ]]; then
        stop_gpu_lease || cleanup_rc=10
    fi
    # Clean up a lease container from an interrupted older controller version.
    if cleanup_legacy_lease_container && verify_container_absent "${LEASE_CONTAINER}"; then
        legacy_lease_absent=true
    else
        cleanup_rc=10
    fi
    [[ "${legacy_lease_absent}" == "true" ]] || cleanup_rc=10
    verify_checkout_unchanged >"${HOST_LOG_ROOT}/checkout-cleanliness.log" 2>&1 || cleanup_rc=10
    if (( primary_rc == 0 && import_rc != 0 )); then primary_rc="${import_rc}"; fi
    (( cleanup_rc == 0 )) || write_host_synthetic cleanup "Resource or patch cleanup failed"
    if [[ ! "${SOURCE_SHA}" =~ ^[0-9a-f]{40}$ ]]; then return 2; fi
    set +e
    python3 "${HOST_HELPER}" finalize \
        --spool "${TRUSTED_SPOOL}" --baseline "${CONTROLLER_ROOT}/ci/hcu/test-baseline.json" \
        --repeat "${REPEAT}" --primary-rc "${primary_rc}" --cleanup-rc "${cleanup_rc}" \
        --repository "${REPOSITORY}" --profile "${PROFILE}" --run-id "${RUN_ID}" \
        --attempt "${ATTEMPT}" --sha "${SOURCE_SHA}" --controller-sha "${CONTROLLER_SHA}" \
        --base-image "${BASE_IMAGE}" --shared-root "${SHARED_ROOT}" --run-key "${RUN_KEY}" \
        --cleanup-root "${WORK_ROOT}"
    local final_rc=$?
    set -e
    if validate_job_status_file; then rm -f -- "${JOB_STATUS_FILE}"; fi
    return "${final_rc}"
}

main() {
    case "${COMMAND}" in
        initialize|build|prepare-tests|test|finalize) ;;
        *)
            printf 'Usage: %s {initialize|build|prepare-tests|test|finalize}\n' "$0" >&2
            return 2
            ;;
    esac
    resolve_context || return 2
    case "${COMMAND}" in
        initialize) phase_initialize ;;
        build) phase_build ;;
        prepare-tests) phase_prepare_tests ;;
        test) phase_test ;;
        finalize) phase_finalize ;;
    esac
}

main

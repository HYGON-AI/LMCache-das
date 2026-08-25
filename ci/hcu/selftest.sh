#!/usr/bin/env bash
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0

set -Eeuo pipefail
export PYTHONDONTWRITEBYTECODE=1

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd -P)"
SELFTEST_CACHE="$(mktemp -d /tmp/lmcache-hcu-selftest.XXXXXX)"
trap 'case "${SELFTEST_CACHE}" in /tmp/lmcache-hcu-selftest.*) rm -rf -- "${SELFTEST_CACHE}" ;; esac' EXIT

python3 -c 'import py_compile,sys; py_compile.compile(sys.argv[1], cfile=sys.argv[2], doraise=True)' \
    "${SCRIPT_DIR}/host.py" "${SELFTEST_CACHE}/host.pyc"
python3 -c 'import py_compile,sys; py_compile.compile(sys.argv[1], cfile=sys.argv[2], doraise=True)' \
    "${SCRIPT_DIR}/model-ci.py" "${SELFTEST_CACHE}/model-ci.pyc"
python3 -m json.tool "${SCRIPT_DIR}/compatibility.json" >/dev/null
python3 -m json.tool "${SCRIPT_DIR}/patch-manifest.json" >/dev/null
python3 -m json.tool "${SCRIPT_DIR}/test-baseline.json" >/dev/null
python3 -m json.tool "${SCRIPT_DIR}/model-test-manifest.json" >/dev/null
bash -n "${SCRIPT_DIR}/run.sh"
bash -n "${SCRIPT_DIR}/run-tests.sh"
python3 "${SCRIPT_DIR}/host.py" selftest
python3 "${SCRIPT_DIR}/model-ci.py" selftest
python3 "${SCRIPT_DIR}/model-ci.py" validate \
    --manifest "${SCRIPT_DIR}/model-test-manifest.json" \
    --profile pr --runner nmz4 --visible-devices 0,1 >/dev/null

STATE_FILE="${SELFTEST_CACHE}/state/state.json"
STATUS_SHA="$(printf clean | sha256sum | awk '{print $1}')"
STATE_ARGS=(
    --state "${STATE_FILE}"
    --repository HYGON-AI/LMCache-das
    --profile pr
    --run-id selftest
    --attempt 1
    --sha 1111111111111111111111111111111111111111
    --controller-sha 2222222222222222222222222222222222222222
    --base-image example.invalid/lmcache@sha256:3333333333333333333333333333333333333333333333333333333333333333
    --base-image-id sha256:4444444444444444444444444444444444444444444444444444444444444444
    --run-key selftest-1-111111111111
    --repeat 1
    --checkout-status-sha256 "${STATUS_SHA}"
)
python3 "${SCRIPT_DIR}/host.py" state-init "${STATE_ARGS[@]}"
python3 "${SCRIPT_DIR}/host.py" state-check "${STATE_ARGS[@]}"
python3 "${SCRIPT_DIR}/host.py" state-require --state "${STATE_FILE}" --phase initialize
python3 "${SCRIPT_DIR}/host.py" state-record --state "${STATE_FILE}" \
    --phase initialize --command-rc 0 --mapped-rc 0
python3 "${SCRIPT_DIR}/host.py" state-require --state "${STATE_FILE}" --phase build
python3 "${SCRIPT_DIR}/host.py" state-record --state "${STATE_FILE}" \
    --phase build --command-rc 4 --mapped-rc 4
[[ "$(python3 "${SCRIPT_DIR}/host.py" state-primary --state "${STATE_FILE}")" == 4 ]]
if python3 "${SCRIPT_DIR}/host.py" state-require \
    --state "${STATE_FILE}" --phase prepare-tests >/dev/null 2>&1; then
    printf '%s\n' 'State machine allowed a phase after a failed predecessor' >&2
    exit 1
fi

if python3 -c 'import sys; raise SystemExit(0 if sys.version_info >= (3, 10) else 1)'; then
    python3 -c 'import py_compile,sys; py_compile.compile(sys.argv[1], cfile=sys.argv[2], doraise=True)' \
        "${SCRIPT_DIR}/ci.py" "${SELFTEST_CACHE}/ci.pyc"
    python3 "${SCRIPT_DIR}/ci.py" selftest
else
    printf '%s\n' \
        'Host Python is older than 3.10; ci.py syntax/helper checks must run in the reviewed test image.'
fi

printf '%s\n' 'LMCache-HCU CI static and helper self-tests passed'

#!/usr/bin/env python3
"""Trusted host-side importer and publisher; compatible with Python 3.6+."""

import argparse
import errno
import hashlib
import json
import os
from pathlib import Path
import re
import shutil
import signal
import stat
import sys
import tempfile
import time
import xml.etree.ElementTree as ET


class HostError(RuntimeError):
    pass


MAX_FILES = 4096
MAX_TOTAL_BYTES = 2 * 1024 * 1024 * 1024
MAX_LOG_BYTES = 128 * 1024 * 1024
MAX_WHEEL_BYTES = 1536 * 1024 * 1024
MAX_METADATA_BYTES = 32 * 1024 * 1024
MAX_CAPTURE_BYTES = 128 * 1024 * 1024

TOP_LEVEL_FILES = {
    "environment.json",
    "upstream-source.json",
    "wheel-report.json",
    "installed-package.json",
    "patch-report.json",
    "test-inventory.json",
    "test-summary.json",
    "model-tool.json",
    "model-inventory.json",
    "model-summary.json",
    "asset-manifest.json",
    "summary.md",
}

CI_PHASES = ("initialize", "build", "prepare-tests", "test")
CI_PHASE_PREDECESSOR = {
    "initialize": None,
    "build": "initialize",
    "prepare-tests": "build",
    "test": "prepare-tests",
}


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise HostError("Expected a JSON object in {}".format(path))
    return value


def atomic_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def atomic_state_json(path, value):
    """Write trusted host state atomically and keep it host-private."""
    atomic_json(path, value)
    os.chmod(str(path), 0o600)


def state_metadata(args):
    return {
        "repository": args.repository,
        "profile": args.profile,
        "run_id": args.run_id,
        "attempt": args.attempt,
        "source_sha": args.sha,
        "controller_sha": args.controller_sha,
        "base_image": args.base_image,
        "base_image_id": args.base_image_id,
        "run_key": args.run_key,
        "job_key": args.job_key,
        "job_role": args.job_role,
        "repeat": args.repeat,
        "checkout_status_sha256": args.checkout_status_sha256,
    }


def validate_state_file(path):
    path = Path(path)
    info = os.lstat(str(path))
    if not stat.S_ISREG(info.st_mode):
        raise HostError("Trusted state is not a regular file: {}".format(path))
    if hasattr(os, "getuid") and info.st_uid != os.getuid():
        raise HostError("Trusted state has an unexpected owner: {}".format(path))
    if os.name != "nt" and info.st_mode & 0o077:
        raise HostError("Trusted state is accessible by group or other users")
    state = read_json(path)
    if state.get("schema_version") != 1:
        raise HostError("Unsupported trusted state schema")
    if not isinstance(state.get("phases"), dict):
        raise HostError("Trusted state has no phase map")
    return state


def validate_state_metadata(state, expected):
    actual = state.get("metadata")
    if actual != expected:
        raise HostError("Trusted state metadata does not match this workflow run")


def cmd_state_init(args):
    path = Path(args.state)
    metadata = state_metadata(args)
    if path.exists():
        state = validate_state_file(path)
        validate_state_metadata(state, metadata)
        print("Trusted state already initialized")
        return 0
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(str(path.parent), 0o700)
    atomic_state_json(
        path,
        {
            "schema_version": 1,
            "metadata": metadata,
            "phases": {},
            "primary_exit_code": 0,
        },
    )
    print("Initialized trusted state")
    return 0


def cmd_state_bootstrap(args):
    if not 1 <= args.mapped_rc <= 255:
        raise HostError("Bootstrap failure exit code must be between 1 and 255")
    path = Path(args.state)
    if path.exists():
        raise HostError("Trusted state already exists")
    path.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    os.chmod(str(path.parent), 0o700)
    atomic_state_json(
        path,
        {
            "schema_version": 1,
            "metadata": state_metadata(args),
            "phases": {
                "initialize": {
                    "command_exit_code": None,
                    "mapped_exit_code": args.mapped_rc,
                    "host_error": args.message,
                }
            },
            "primary_exit_code": args.mapped_rc,
        },
    )
    print("Bootstrapped trusted failure state")
    return 0


def cmd_state_check(args):
    state = validate_state_file(args.state)
    validate_state_metadata(state, state_metadata(args))
    print("Trusted state is valid")
    return 0


def cmd_state_require(args):
    if args.phase not in CI_PHASES:
        raise HostError("Unknown CI phase: {}".format(args.phase))
    state = validate_state_file(args.state)
    predecessor = CI_PHASE_PREDECESSOR[args.phase]
    if predecessor is not None:
        record = state["phases"].get(predecessor)
        if not isinstance(record, dict) or record.get("mapped_exit_code") != 0:
            raise HostError(
                "Phase {} requires successful phase {}".format(args.phase, predecessor)
            )
    if args.phase in state["phases"]:
        raise HostError("Phase {} was already recorded".format(args.phase))
    print("Phase {} may run".format(args.phase))
    return 0


def cmd_state_record(args):
    if args.phase not in CI_PHASES:
        raise HostError("Unknown CI phase: {}".format(args.phase))
    if not 0 <= args.command_rc <= 255 or not 0 <= args.mapped_rc <= 255:
        raise HostError("Phase exit codes must be between 0 and 255")
    path = Path(args.state)
    state = validate_state_file(path)
    predecessor = CI_PHASE_PREDECESSOR[args.phase]
    if predecessor is not None:
        record = state["phases"].get(predecessor)
        if not isinstance(record, dict) or record.get("mapped_exit_code") != 0:
            raise HostError(
                "Cannot record {} before successful {}".format(args.phase, predecessor)
            )
    if args.phase in state["phases"]:
        raise HostError("Phase {} was already recorded".format(args.phase))
    state["phases"][args.phase] = {
        "command_exit_code": args.command_rc,
        "mapped_exit_code": args.mapped_rc,
    }
    if state.get("primary_exit_code", 0) == 0 and args.mapped_rc != 0:
        state["primary_exit_code"] = args.mapped_rc
    atomic_state_json(path, state)
    print("Recorded phase {} with exit code {}".format(args.phase, args.mapped_rc))
    return 0


def cmd_state_note(args):
    if args.phase not in CI_PHASES:
        raise HostError("Unknown CI phase: {}".format(args.phase))
    if not 1 <= args.mapped_rc <= 255:
        raise HostError("Host failure exit code must be between 1 and 255")
    path = Path(args.state)
    state = validate_state_file(path)
    if args.phase in state["phases"]:
        record = state["phases"][args.phase]
        if record.get("mapped_exit_code") == 0:
            raise HostError("Cannot replace a successful phase record")
        print("Phase {} already has a failure record".format(args.phase))
        return 0
    state["phases"][args.phase] = {
        "command_exit_code": None,
        "mapped_exit_code": args.mapped_rc,
        "host_error": args.message,
    }
    if state.get("primary_exit_code", 0) == 0:
        state["primary_exit_code"] = args.mapped_rc
    atomic_state_json(path, state)
    print("Recorded host failure for phase {}".format(args.phase))
    return 0


def cmd_state_primary(args):
    state = validate_state_file(args.state)
    value = state.get("primary_exit_code")
    if not isinstance(value, int) or not 0 <= value <= 255:
        raise HostError("Trusted state has an invalid primary exit code")
    print(value)
    return 0


def cmd_state_phase(args):
    state = validate_state_file(args.state)
    record = state["phases"].get(args.phase)
    if record is None:
        print("missing")
        return 0
    value = record.get("mapped_exit_code")
    if not isinstance(value, int) or not 0 <= value <= 255:
        raise HostError("Trusted state has an invalid phase exit code")
    print(value)
    return 0


def cmd_state_abort(args):
    if args.phase not in CI_PHASES:
        raise HostError("Unknown CI phase: {}".format(args.phase))
    if not 0 <= args.mapped_rc <= 255 or args.mapped_rc == 0:
        raise HostError("Abort exit code must be between 1 and 255")
    path = Path(args.state)
    state = validate_state_file(path)
    if args.phase in state["phases"]:
        raise HostError("Phase {} was already recorded".format(args.phase))
    state["phases"][args.phase] = {
        "command_exit_code": None,
        "mapped_exit_code": args.mapped_rc,
        "host_error": args.message,
    }
    if state.get("primary_exit_code", 0) == 0:
        state["primary_exit_code"] = args.mapped_rc
    atomic_state_json(path, state)
    print("Recorded host failure for phase {}".format(args.phase))
    return 0


def cmd_hold_lock(args):
    try:
        import fcntl
    except ImportError as exc:
        raise HostError("GPU lease locking requires a POSIX host") from exc
    lock = Path(args.lock).resolve()
    ready = Path(args.ready).resolve()
    if not str(lock).startswith("/tmp/hcu-ci-gpu-locks/"):
        raise HostError("GPU lock must be under /tmp/hcu-ci-gpu-locks")
    runner_temp = os.environ.get("RUNNER_TEMP")
    allowed = Path(runner_temp or "/tmp").resolve() / "lmcache-hcu"
    if not is_within(ready, allowed):
        raise HostError("Lease ready file escaped the LMCache-HCU work root")
    lock.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    ready.parent.mkdir(mode=0o700, parents=True, exist_ok=True)
    descriptor = os.open(str(lock), os.O_RDWR | os.O_CREAT, 0o600)
    deadline = time.time() + args.timeout
    try:
        while True:
            try:
                fcntl.flock(descriptor, fcntl.LOCK_EX | fcntl.LOCK_NB)
                break
            except (IOError, OSError) as exc:
                if getattr(exc, "errno", None) not in (errno.EACCES, errno.EAGAIN):
                    raise
                if time.time() >= deadline:
                    raise HostError("Timed out waiting for the HCU GPU lock")
                time.sleep(1)

        atomic_state_json(ready, {"pid": os.getpid(), "lock": str(lock)})
        running = [True]

        def stop(_signum, _frame):
            running[0] = False

        signal.signal(signal.SIGTERM, stop)
        signal.signal(signal.SIGINT, stop)
        while running[0]:
            time.sleep(1)
    finally:
        try:
            ready.unlink()
        except OSError:
            pass
        try:
            fcntl.flock(descriptor, fcntl.LOCK_UN)
        finally:
            os.close(descriptor)
    return 0


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        while True:
            chunk = handle.read(1024 * 1024)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def ensure_token(value, label):
    if not re.match(r"^[A-Za-z0-9_.-]+$", value or ""):
        raise HostError("Unsafe {}: {!r}".format(label, value))


def is_within(path, parent):
    path = os.path.realpath(str(path))
    parent = os.path.realpath(str(parent))
    return path == parent or path.startswith(parent + os.sep)


def allowed_output(relative, size):
    value = relative.as_posix()
    parts = relative.parts
    if len(parts) == 1 and value in TOP_LEVEL_FILES:
        return size <= MAX_METADATA_BYTES
    if len(parts) != 2:
        return False
    top, name = parts
    if top == "reports":
        return name.endswith(".xml") and size <= MAX_METADATA_BYTES
    if top == "logs":
        return name.endswith(".log") and size <= MAX_LOG_BYTES
    if top == "wheels":
        return name.endswith(".whl") and size <= MAX_WHEEL_BYTES
    if top == "state":
        return (name == "current-stage" or name.endswith(".rc")) and size <= 4096
    if top == "model-results":
        return name.endswith(".json") and size <= MAX_METADATA_BYTES
    if top == "effective-configs":
        return name.endswith(".conf") and size <= MAX_METADATA_BYTES
    return False


def scan_untrusted_tree(source):
    source = Path(source)
    root_stat = os.lstat(str(source))
    if not stat.S_ISDIR(root_stat.st_mode):
        raise HostError("Untrusted output root is not a real directory")
    files = []
    stack = [source]
    total = 0
    while stack:
        directory = stack.pop()
        for entry in os.scandir(str(directory)):
            entry_stat = entry.stat(follow_symlinks=False)
            path = Path(entry.path)
            relative = path.relative_to(source)
            if stat.S_ISDIR(entry_stat.st_mode):
                if len(relative.parts) > 1 or relative.parts[0] not in {
                    "reports",
                    "logs",
                    "wheels",
                    "state",
                    "model-results",
                    "effective-configs",
                }:
                    raise HostError("Unexpected output directory: {}".format(relative))
                stack.append(path)
                continue
            if not stat.S_ISREG(entry_stat.st_mode):
                raise HostError("Non-regular output rejected: {}".format(relative))
            if os.name != "nt" and entry_stat.st_nlink != 1:
                raise HostError("Hard-linked output rejected: {}".format(relative))
            if not allowed_output(relative, entry_stat.st_size):
                raise HostError("Unexpected or oversized output: {}".format(relative))
            total += entry_stat.st_size
            files.append((path, relative, entry_stat.st_size))
            if len(files) > MAX_FILES or total > MAX_TOTAL_BYTES:
                raise HostError("Container output exceeds the CI publication quota")
    if not files:
        raise HostError("Container produced no importable files")
    return sorted(files, key=lambda item: item[1].as_posix())


def copy_regular_no_follow(source, destination, expected_size):
    flags = os.O_RDONLY
    if hasattr(os, "O_NOFOLLOW"):
        flags |= os.O_NOFOLLOW
    source_fd = os.open(str(source), flags)
    try:
        source_stat = os.fstat(source_fd)
        if not stat.S_ISREG(source_stat.st_mode) or (
            os.name != "nt" and source_stat.st_nlink != 1
        ):
            raise HostError("Source changed type while importing: {}".format(source))
        if source_stat.st_size != expected_size:
            raise HostError("Source changed size while importing: {}".format(source))
        destination.parent.mkdir(mode=0o750, parents=True, exist_ok=True)
        destination_fd = os.open(
            str(destination), os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o640
        )
        try:
            while True:
                chunk = os.read(source_fd, 1024 * 1024)
                if not chunk:
                    break
                view = memoryview(chunk)
                while view:
                    written = os.write(destination_fd, view)
                    view = view[written:]
            os.fsync(destination_fd)
        finally:
            os.close(destination_fd)
    finally:
        os.close(source_fd)


def safe_import(source, destination, preserve_source_path=False):
    # `/proc/<pid>/root/...` intentionally points into another mount namespace.
    # Resolving it in the helper namespace would collapse the path back to the
    # helper's own root. The caller may therefore preserve this trusted,
    # controller-supplied source path while the scanner still uses lstat,
    # scandir and O_NOFOLLOW for every untrusted entry.
    source = Path(source) if preserve_source_path else Path(source).resolve()
    destination = Path(destination).resolve()
    destination.mkdir(mode=0o750, parents=True, exist_ok=True)
    destination_stat = os.stat(str(destination))
    destination_uid = destination_stat.st_uid
    destination_gid = destination_stat.st_gid
    imported = []
    for path, relative, size in scan_untrusted_tree(source):
        target = destination / relative
        if not is_within(target.parent, destination):
            raise HostError("Output path escaped destination: {}".format(relative))
        copy_regular_no_follow(path, target, size)
        if hasattr(os, "chown"):
            os.chown(str(target.parent), destination_uid, destination_gid)
            os.chown(str(target), destination_uid, destination_gid)
        imported.append(
            {
                "path": relative.as_posix(),
                "size": size,
                "sha256": sha256_file(target),
            }
        )
    import_report = destination / "import-report.json"
    atomic_json(import_report, {"files": imported})
    if hasattr(os, "chown"):
        os.chown(str(import_report), destination_uid, destination_gid)
    return imported


def junit_stats(path):
    try:
        tree = ET.parse(str(path))
    except (ET.ParseError, OSError, IOError) as exc:
        raise HostError("Invalid JUnit XML {}: {}".format(path, exc))
    cases = list(tree.getroot().iter("testcase"))
    return {
        "tests": len(cases),
        "failures": sum(case.find("failure") is not None for case in cases),
        "errors": sum(case.find("error") is not None for case in cases),
        "skipped": sum(case.find("skipped") is not None for case in cases),
    }


def write_synthetic(path, name, message, failed=True):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        name="lmcache-hcu-ci-host",
        tests="1",
        failures="1" if failed else "0",
        errors="0",
        skipped="0",
        time="0",
    )
    case = ET.SubElement(suite, "testcase", classname="ci.host", name=name, time="0")
    if failed:
        failure = ET.SubElement(case, "failure", message=message, type="CIHostFailure")
        failure.text = message
    ET.ElementTree(suite).write(str(path), encoding="utf-8", xml_declaration=True)


def capture_log(output, limit, echo=True):
    """Capture bounded untrusted stdout while draining the complete stream."""
    output = Path(output)
    output.parent.mkdir(parents=True, exist_ok=True)
    written = 0
    truncated = False
    with output.open("xb") as handle:
        while True:
            chunk = sys.stdin.buffer.read(1024 * 1024)
            if not chunk:
                break
            remaining = max(0, limit - written)
            if remaining:
                payload = chunk[:remaining]
                handle.write(payload)
                if echo:
                    sys.stdout.buffer.write(payload)
                    sys.stdout.buffer.flush()
                written += len(payload)
            if len(chunk) > remaining:
                truncated = True
        if truncated:
            marker = (
                "\n[LMCache-HCU CI] container log exceeded {} bytes; "
                "remaining output was drained but not retained.\n".format(limit)
            ).encode("utf-8")
            handle.write(marker)
            if echo:
                sys.stdout.buffer.write(marker)
                sys.stdout.buffer.flush()
        handle.flush()
        os.fsync(handle.fileno())
    if truncated:
        raise HostError("Container log exceeded the {} byte CI limit".format(limit))
    return written, False


def inventory_file(nodeid):
    normalized = nodeid.replace("\\", "/").split("::", 1)[0]
    marker = "tests/"
    position = normalized.find(marker)
    if position >= 0:
        return normalized[position:]
    if normalized.startswith("lmcache_hcu/"):
        return "tests/" + normalized
    return normalized


def inventory_definition(nodeid):
    file_path = inventory_file(nodeid)
    normalized = nodeid.replace("\\", "/")
    separator = normalized.find("::")
    if separator < 0:
        return file_path, ""
    components = normalized[separator + 2 :].split("::")
    cleaned = []
    for component in components:
        cleaned.append(component.split("[", 1)[0])
    return file_path, "::".join(cleaned)


def resolved_model_profile(manifest, profile):
    profiles = manifest.get("profiles", {})
    if profile not in profiles:
        raise HostError("unknown model profile {}".format(profile))
    value = profiles[profile]
    inherited = value.get("inherits")
    return profiles[inherited] if inherited else value


def validate_success(
    spool,
    baseline_path,
    repeat,
    model_manifest_path,
    model_profile,
    runner_kind,
    job_role,
):
    required = [
        "environment.json",
        "upstream-source.json",
        "wheel-report.json",
        "installed-package.json",
        "patch-report.json",
        "model-inventory.json",
        "model-summary.json",
        "asset-manifest.json",
    ]
    if job_role == "framework":
        required.extend(["test-inventory.json", "test-summary.json"])
    elif job_role == "model":
        required.append("model-tool.json")
    else:
        raise HostError("unknown job role {}".format(job_role))
    problems = []
    for relative in required:
        if not (spool / relative).is_file():
            problems.append("missing {}".format(relative))
    if problems:
        return problems, None

    expected = 0
    repeat_stats = []
    if job_role == "framework":
        baseline = read_json(baseline_path)
        inventory = read_json(spool / "test-inventory.json")
        nodeids = inventory.get("nodeids")
        if not isinstance(nodeids, list) or not all(isinstance(item, str) for item in nodeids):
            return ["test inventory does not contain a string nodeid list"], None
        if len(nodeids) != len(set(nodeids)):
            problems.append("test inventory contains duplicate nodeids")
        if int(inventory.get("count", -1)) != len(nodeids):
            problems.append("test inventory count does not match nodeids")
        minimum = int(baseline["minimum_collected"])
        if len(nodeids) < minimum:
            problems.append("collected {} tests; trusted minimum is {}".format(len(nodeids), minimum))
        collected_files = set(inventory_file(item) for item in nodeids)
        missing_files = sorted(set(baseline["required_test_files"]) - collected_files)
        if missing_files:
            problems.append("required test files were not collected: {}".format(", ".join(missing_files)))
        collected_definitions = set(inventory_definition(item) for item in nodeids)
        for file_path, definitions in baseline["required_test_definitions"].items():
            for definition in definitions:
                if (file_path, definition) not in collected_definitions:
                    problems.append(
                        "required test definition was not collected: {}::{}".format(
                            file_path, definition
                        )
                    )

        expected = len(nodeids)
        for index in range(1, repeat + 1):
            path = spool / "reports" / "junit-repeat-{}.xml".format(index)
            if not path.is_file():
                problems.append("missing {}".format(path.relative_to(spool)))
                continue
            try:
                stats = junit_stats(path)
            except HostError as exc:
                problems.append(str(exc))
                continue
            repeat_stats.append(stats)
            if stats["tests"] != expected:
                problems.append(
                    "repeat {} has {} JUnit cases; expected {}".format(index, stats["tests"], expected)
                )
            if stats["failures"] or stats["errors"] or stats["skipped"]:
                problems.append(
                    "repeat {} failures={} errors={} skipped={}".format(
                        index, stats["failures"], stats["errors"], stats["skipped"]
                    )
                )
    patch_report = read_json(spool / "patch-report.json")
    if patch_report.get("status") != "passed":
        problems.append("source-patch gate did not pass")
    if job_role == "framework":
        test_summary = read_json(spool / "test-summary.json")
        if test_summary.get("status") != "passed":
            problems.append("container aggregate did not pass")
        if int(test_summary.get("expected_per_repeat", -1)) != expected:
            problems.append("container aggregate expected count differs from inventory")
        if int(test_summary.get("repeat_count", -1)) != repeat:
            problems.append("container aggregate repeat count differs from requested repeat")
        if int(test_summary.get("expected_total", -1)) != expected * repeat:
            problems.append("container aggregate total count differs from trusted expectation")
    model_manifest = read_json(model_manifest_path)
    if runner_kind not in model_manifest.get("runners", []):
        problems.append("model manifest runner differs from the workflow runner")
    profile = resolved_model_profile(model_manifest, model_profile)
    model_expected_ids = []
    for repeat_index in range(1, repeat + 1):
        for item in profile.get("runs", []):
            model_expected_ids.append("{}-repeat-{}".format(item["scenario"], repeat_index))
    if model_expected_ids and not (spool / "model-tool.json").is_file():
        problems.append("missing model-tool.json for a model-enabled profile")
    model_inventory = read_json(spool / "model-inventory.json")
    model_nodeids = model_inventory.get("nodeids")
    if model_nodeids != model_expected_ids:
        problems.append("model inventory differs from the trusted manifest selection")
    if int(model_inventory.get("count", -1)) != len(model_expected_ids):
        problems.append("model inventory count differs from the trusted expectation")
    model_junit = spool / "reports" / "model-junit.xml"
    if not model_junit.is_file():
        problems.append("missing reports/model-junit.xml")
    else:
        try:
            model_stats = junit_stats(model_junit)
            if model_stats["tests"] != len(model_expected_ids):
                problems.append("model JUnit count differs from the trusted expectation")
            if model_stats["failures"] or model_stats["errors"] or model_stats["skipped"]:
                problems.append(
                    "model failures={} errors={} skipped={}".format(
                        model_stats["failures"], model_stats["errors"], model_stats["skipped"]
                    )
                )
        except HostError as exc:
            problems.append(str(exc))
    model_summary = read_json(spool / "model-summary.json")
    if model_summary.get("status") != "passed":
        problems.append("model test summary did not pass")
    if int(model_summary.get("expected", -1)) != len(model_expected_ids):
        problems.append("model summary count differs from the trusted expectation")
    if model_summary.get("profile") != model_profile or model_summary.get("runner") != runner_kind:
        problems.append("model summary profile or runner differs from the workflow")
    asset_manifest = read_json(spool / "asset-manifest.json")
    if asset_manifest.get("profile") != model_profile or asset_manifest.get("runner") != runner_kind:
        problems.append("asset manifest profile or runner differs from the workflow")
    expected_tool_commit = model_manifest.get("tool", {}).get("commit")
    if model_expected_ids and asset_manifest.get("tool_commit") != expected_tool_commit:
        problems.append("asset manifest test-tool commit differs from the trusted manifest")
    expected_result_files = set("{}.json".format(item) for item in model_expected_ids)
    actual_result_files = set(
        path.name for path in (spool / "model-results").glob("*.json")
    ) if (spool / "model-results").is_dir() else set()
    if actual_result_files != expected_result_files:
        problems.append("raw model result files differ from the trusted scenario inventory")
    expected_config_files = set("{}.conf".format(item) for item in model_expected_ids)
    actual_config_files = set(
        path.name for path in (spool / "effective-configs").glob("*.conf")
    ) if (spool / "effective-configs").is_dir() else set()
    if actual_config_files != expected_config_files:
        problems.append("effective model config files differ from the trusted scenario inventory")
    return problems, {
        "collected": expected,
        "repeats": repeat_stats,
        "model_cases": len(model_expected_ids),
    }


def copy_trusted_tree(source, destination):
    for directory, names, files in os.walk(str(source)):
        names.sort()
        files.sort()
        current = Path(directory)
        relative_dir = current.relative_to(source)
        target_dir = destination / relative_dir
        target_dir.mkdir(mode=0o750, parents=True, exist_ok=True)
        for name in files:
            path = current / name
            info = os.lstat(str(path))
            if not stat.S_ISREG(info.st_mode) or (
                os.name != "nt" and info.st_nlink != 1
            ):
                raise HostError("Trusted spool contains unsafe file: {}".format(path))
            shutil.copyfile(str(path), str(target_dir / name))
            os.chmod(str(target_dir / name), 0o640)


def write_checksums(root):
    sums = root / "SHA256SUMS"
    files = sorted(
        path
        for path in root.rglob("*")
        if path.is_file() and path.name not in {"SHA256SUMS", "READY", "READY.tmp"}
    )
    if not files:
        raise HostError("No files available for publication")
    with sums.open("w", encoding="utf-8") as handle:
        for path in files:
            handle.write(
                "{}  {}\n".format(sha256_file(path), path.relative_to(root).as_posix())
            )
        handle.flush()
        os.fsync(handle.fileno())
    for line in sums.read_text(encoding="utf-8").splitlines():
        expected, relative = line.split("  ", 1)
        if sha256_file(root / relative) != expected:
            raise HostError("Checksum verification failed for {}".format(relative))


def fsync_directory(path):
    flags = os.O_RDONLY
    if hasattr(os, "O_DIRECTORY"):
        flags |= os.O_DIRECTORY
    descriptor = os.open(str(path), flags)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def mark_publication_failed(location, message):
    if location is None or not location.is_dir() or (location / "READY").exists():
        return
    manifest_path = location / "manifest.json"
    try:
        manifest = read_json(manifest_path) if manifest_path.is_file() else {}
        manifest["publish_exit_code"] = 11
        manifest["status"] = "failed"
        manifest["publish_error"] = str(message)
        atomic_json(manifest_path, manifest)
        with (location / "PUBLICATION_FAILED").open("w", encoding="utf-8") as handle:
            handle.write(str(message) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        existing_sums = location / "SHA256SUMS"
        if existing_sums.exists():
            existing_sums.unlink()
        write_checksums(location)
    except Exception as exc:
        print(
            "Unable to mark retained publication as failed: {}".format(exc),
            file=sys.stderr,
        )


def publish(spool, shared_root, profile, run_id, attempt, sha, job_key, run_key, cleanup_root):
    shared_root = Path(shared_root).resolve()
    if str(shared_root) != "/ci_public/lmcache-das" and os.environ.get("HCU_CI_HOST_SELFTEST") != "1":
        raise HostError("Publication root must be exactly /ci_public/lmcache-das")
    for value, label in (
        (profile, "profile"),
        (run_id, "run id"),
        (attempt, "attempt"),
        (sha, "SHA"),
        (job_key, "job key"),
        (run_key, "run key"),
    ):
        ensure_token(value, label)
    parent = shared_root / profile / run_id / attempt / sha
    final = parent / job_key
    staging = parent / (".staging-" + run_key)
    if not is_within(final, shared_root) or not is_within(staging, shared_root):
        raise HostError("Publication path escaped the shared root")
    if final.exists() or staging.exists():
        raise HostError("Publication destination already exists: {}".format(final))
    staging.mkdir(parents=True)
    try:
        copy_trusted_tree(spool, staging)
        write_checksums(staging)

        cleanup_root = Path(cleanup_root).resolve()
        runner_temp = os.environ.get("RUNNER_TEMP")
        if not runner_temp and os.environ.get("HCU_CI_HOST_SELFTEST") != "1":
            raise HostError("RUNNER_TEMP is required for safely scoped cleanup")
        allowed_parent = Path(runner_temp or tempfile.gettempdir()).resolve() / "lmcache-hcu"
        if not is_within(cleanup_root, allowed_parent) or cleanup_root == allowed_parent:
            raise HostError("Refusing unsafe cleanup root: {}".format(cleanup_root))
        os.replace(str(staging), str(final))
        fsync_directory(parent)
        try:
            shutil.rmtree(str(cleanup_root))
        except Exception as exc:
            raise HostError("workspace cleanup failed: {}".format(exc))
        ready_tmp = final / "READY.tmp"
        with ready_tmp.open("x", encoding="utf-8") as handle:
            handle.write("complete\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(str(ready_tmp), str(final / "READY"))
        fsync_directory(final)
        return final
    except Exception as exc:
        # Preserve the trusted staging directory when publication fails after
        # it has been prepared. It is deliberately left without READY and its
        # manifest is marked failed for inspection by the retention job.
        retained = final if final.is_dir() else staging if staging.is_dir() else None
        mark_publication_failed(retained, exc)
        raise


def cmd_safe_import(args):
    imported = safe_import(args.source, args.destination, args.preserve_source_path)
    print("Safely imported {} files".format(len(imported)))
    return 0


def cmd_synthetic(args):
    write_synthetic(args.output, args.name, args.message, not args.passed)
    return 0


def cmd_capture_log(args):
    written, truncated = capture_log(args.output, args.limit)
    print(
        "LMCache-HCU log capture retained {} bytes{}".format(
            written, " (truncated)" if truncated else ""
        ),
        file=sys.stderr,
    )
    return 0


def cmd_finalize(args):
    spool = Path(args.spool).resolve()
    problems = []
    details = None
    primary_rc = args.primary_rc
    if primary_rc == 0:
        try:
            problems, details = validate_success(
                spool,
                args.baseline,
                args.repeat,
                args.model_manifest,
                args.model_profile,
                args.runner_kind,
                args.job_role,
            )
        except Exception as exc:
            problems = ["host result validation failed: {}".format(exc)]
        if problems:
            primary_rc = 9
    if args.cleanup_rc and primary_rc == 0:
        primary_rc = 10
    if primary_rc != 0:
        write_synthetic(
            spool / "reports" / "synthetic-host-final.xml",
            "host-final-validation",
            "; ".join(problems) if problems else "CI exited with code {}".format(primary_rc),
        )

    imported_summary = spool / "summary.md"
    if imported_summary.exists():
        os.replace(str(imported_summary), str(spool / "test-output-summary.md"))
    imported_manifest = spool / "manifest.json"
    if imported_manifest.exists():
        os.replace(str(imported_manifest), str(spool / "container-manifest.json"))
    status = "passed" if primary_rc == 0 and args.cleanup_rc == 0 else "failed"
    summary = [
        "# LMCache-HCU CI summary",
        "",
        "- Status: `{}`".format(status),
        "- Source SHA: `{}`".format(args.sha),
        "- Profile: `{}`".format(args.profile),
        "- Repetitions: `{}`".format(args.repeat),
        "- Primary exit code: `{}`".format(primary_rc),
        "- Cleanup exit code: `{}`".format(args.cleanup_rc),
    ]
    if details:
        summary.append("- Collected tests: `{}`".format(details["collected"]))
        summary.append("- Model cases: `{}`".format(details["model_cases"]))
    if problems:
        summary.extend(["", "## Host validation failures", ""])
        summary.extend("- {}".format(item) for item in problems)
    (spool / "summary.md").write_text("\n".join(summary) + "\n", encoding="utf-8")

    manifest = {
        "schema_version": 1,
        "repository": args.repository,
        "profile": args.profile,
        "model_profile": args.model_profile,
        "job_role": args.job_role,
        "job_key": args.job_key,
        "runner_kind": args.runner_kind,
        "run_id": args.run_id,
        "attempt": args.attempt,
        "source_sha": args.sha,
        "controller_sha": args.controller_sha,
        "base_image": args.base_image,
        "base_image_id": args.base_image_id,
        "primary_exit_code": primary_rc,
        "cleanup_exit_code": args.cleanup_rc,
        "publish_exit_code": 0,
        "status": status,
    }
    atomic_json(spool / "manifest.json", manifest)
    try:
        final = publish(
            spool,
            args.shared_root,
            args.profile,
            args.run_id,
            args.attempt,
            args.sha,
            args.job_key,
            args.run_key,
            args.cleanup_root,
        )
    except Exception as exc:
        manifest["publish_exit_code"] = 11
        manifest["status"] = "failed"
        if spool.exists():
            atomic_json(spool / "manifest.json", manifest)
        print(
            "Publication failed; trusted spool or staging retained for diagnosis: {}".format(exc),
            file=sys.stderr,
        )
        return 11 if primary_rc == 0 else primary_rc
    print("Published CI results to {}".format(final))
    return primary_rc


def cmd_selftest(_args):
    with tempfile.TemporaryDirectory(prefix="lmcache-hcu-host-selftest-") as temporary:
        root = Path(temporary)
        source = root / "untrusted"
        destination = root / "trusted"
        (source / "reports").mkdir(parents=True)
        write_synthetic(source / "reports" / "failure.xml", "injected", "expected")
        safe_import(source, destination)
        if junit_stats(destination / "reports" / "failure.xml")["failures"] != 1:
            raise HostError("Synthetic JUnit self-test failed")

        bad = root / "bad"
        bad.mkdir()
        target = root / "target.txt"
        target.write_text("secret", encoding="utf-8")
        try:
            os.symlink(str(target), str(bad / "summary.md"))
        except (OSError, NotImplementedError):
            print("Symlink rejection self-test skipped on this host")
            print("LMCache-HCU trusted host helper self-tests passed")
            return 0
        try:
            safe_import(bad, root / "bad-destination")
        except HostError:
            pass
        else:
            raise HostError("Symlink output was not rejected")

        capture_path = root / "capture.log"
        original_stdin = sys.stdin
        try:
            with (root / "capture-input").open("w+b") as payload:
                payload.write(b"0123456789")
                payload.seek(0)
                sys.stdin = type(
                    "BinaryStdin",
                    (),
                    {"buffer": payload},
                )()
                try:
                    capture_log(capture_path, 5, echo=False)
                except HostError:
                    pass
                else:
                    raise HostError("Oversized log was not rejected")
        finally:
            sys.stdin = original_stdin
    print("LMCache-HCU trusted host helper self-tests passed")
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command")

    def add_state_metadata(command):
        command.add_argument("--state", required=True)
        command.add_argument("--repository", required=True)
        command.add_argument("--profile", required=True)
        command.add_argument("--run-id", required=True)
        command.add_argument("--attempt", required=True)
        command.add_argument("--sha", required=True)
        command.add_argument("--controller-sha", required=True)
        command.add_argument("--base-image", required=True)
        command.add_argument("--base-image-id", required=True)
        command.add_argument("--run-key", required=True)
        command.add_argument("--job-key", required=True)
        command.add_argument("--job-role", choices=("framework", "model"), required=True)
        command.add_argument("--repeat", type=int, required=True)
        command.add_argument("--checkout-status-sha256", required=True)

    state_init = commands.add_parser("state-init")
    add_state_metadata(state_init)
    state_init.set_defaults(func=cmd_state_init)

    state_bootstrap = commands.add_parser("state-bootstrap-failure")
    add_state_metadata(state_bootstrap)
    state_bootstrap.add_argument("--mapped-rc", type=int, required=True)
    state_bootstrap.add_argument("--message", required=True)
    state_bootstrap.set_defaults(func=cmd_state_bootstrap)

    state_check = commands.add_parser("state-check")
    add_state_metadata(state_check)
    state_check.set_defaults(func=cmd_state_check)

    state_require = commands.add_parser("state-require")
    state_require.add_argument("--state", required=True)
    state_require.add_argument("--phase", choices=CI_PHASES, required=True)
    state_require.set_defaults(func=cmd_state_require)

    state_record = commands.add_parser("state-record")
    state_record.add_argument("--state", required=True)
    state_record.add_argument("--phase", choices=CI_PHASES, required=True)
    state_record.add_argument("--command-rc", type=int, required=True)
    state_record.add_argument("--mapped-rc", type=int, required=True)
    state_record.set_defaults(func=cmd_state_record)

    state_note = commands.add_parser("state-note-failure")
    state_note.add_argument("--state", required=True)
    state_note.add_argument("--phase", choices=CI_PHASES, required=True)
    state_note.add_argument("--mapped-rc", type=int, required=True)
    state_note.add_argument("--message", required=True)
    state_note.set_defaults(func=cmd_state_note)

    state_primary = commands.add_parser("state-primary")
    state_primary.add_argument("--state", required=True)
    state_primary.set_defaults(func=cmd_state_primary)

    state_phase = commands.add_parser("state-phase")
    state_phase.add_argument("--state", required=True)
    state_phase.add_argument("--phase", choices=CI_PHASES, required=True)
    state_phase.set_defaults(func=cmd_state_phase)

    state_abort = commands.add_parser("state-abort")
    state_abort.add_argument("--state", required=True)
    state_abort.add_argument("--phase", choices=CI_PHASES, required=True)
    state_abort.add_argument("--mapped-rc", type=int, required=True)
    state_abort.add_argument("--message", required=True)
    state_abort.set_defaults(func=cmd_state_abort)

    hold_lock = commands.add_parser("hold-lock")
    hold_lock.add_argument("--lock", required=True)
    hold_lock.add_argument("--ready", required=True)
    hold_lock.add_argument("--timeout", type=int, default=300)
    hold_lock.set_defaults(func=cmd_hold_lock)

    imported = commands.add_parser("safe-import")
    imported.add_argument("--source", required=True)
    imported.add_argument("--destination", required=True)
    imported.add_argument("--preserve-source-path", action="store_true")
    imported.set_defaults(func=cmd_safe_import)

    synthetic = commands.add_parser("synthetic")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--name", required=True)
    synthetic.add_argument("--message", required=True)
    synthetic.add_argument("--passed", action="store_true")
    synthetic.set_defaults(func=cmd_synthetic)

    capture = commands.add_parser("capture-log")
    capture.add_argument("--output", required=True)
    capture.add_argument("--limit", type=int, default=MAX_CAPTURE_BYTES)
    capture.set_defaults(func=cmd_capture_log)

    final = commands.add_parser("finalize")
    final.add_argument("--spool", required=True)
    final.add_argument("--baseline", required=True)
    final.add_argument("--model-manifest", required=True)
    final.add_argument("--model-profile", required=True)
    final.add_argument("--runner-kind", required=True)
    final.add_argument("--job-role", choices=("framework", "model"), required=True)
    final.add_argument("--job-key", required=True)
    final.add_argument("--repeat", type=int, required=True)
    final.add_argument("--primary-rc", type=int, required=True)
    final.add_argument("--cleanup-rc", type=int, required=True)
    final.add_argument("--repository", required=True)
    final.add_argument("--profile", required=True)
    final.add_argument("--run-id", required=True)
    final.add_argument("--attempt", required=True)
    final.add_argument("--sha", required=True)
    final.add_argument("--controller-sha", required=True)
    final.add_argument("--base-image", required=True)
    final.add_argument("--base-image-id", required=True)
    final.add_argument("--shared-root", required=True)
    final.add_argument("--run-key", required=True)
    final.add_argument("--cleanup-root", required=True)
    final.set_defaults(func=cmd_finalize)

    selftest = commands.add_parser("selftest")
    selftest.set_defaults(func=cmd_selftest)
    return result


def main():
    args = parser().parse_args()
    if not hasattr(args, "func"):
        raise HostError("A command is required")
    try:
        return int(args.func(args))
    except HostError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print("ERROR: unexpected {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

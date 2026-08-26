#!/usr/bin/env python3
"""Manifest-driven LMCache model smoke runner and result gate."""

import argparse
import configparser
import contextlib
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tarfile
import tempfile
import threading
import time
import xml.etree.ElementTree as ET


class ModelCIError(RuntimeError):
    pass


ALLOWED_CHECKS = {"long_doc", "opencompass", "cmmlu"}
TOKEN = re.compile(r"^[a-z0-9][a-z0-9_.-]*$")
COMMIT = re.compile(r"^[0-9a-f]{40}$")
SHA256 = re.compile(r"^[0-9a-f]{64}$")
TEST_TOOL_ASSET_ROOT = "/ci_public/lmcache-das/assets/test-tool/"
ALLOWED_MODEL_ENVIRONMENT = {"VLLM_USE_CAT_MLA": {"1"}}


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise ModelCIError("expected a JSON object: {}".format(path))
    return value


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("w", encoding="utf-8") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")
    os.replace(str(temporary), str(path))


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def profile_config(manifest, profile):
    profiles = manifest.get("profiles", {})
    if profile not in profiles:
        raise ModelCIError("unknown model profile: {}".format(profile))
    value = profiles[profile]
    if not isinstance(value, dict):
        raise ModelCIError("profile {} must be an object".format(profile))
    inherited = value.get("inherits")
    if inherited:
        if set(value) != {"visible_devices", "inherits"}:
            raise ModelCIError("inherited profile {} contains unsupported overrides".format(profile))
        parent = dict(profile_config(manifest, inherited))
        parent["visible_devices"] = value["visible_devices"]
        return parent
    return value


def selected_runs(manifest, profile):
    value = profile_config(manifest, profile)
    runs = value.get("runs")
    if not isinstance(runs, list):
        raise ModelCIError("profile {} does not contain a run list".format(profile))
    return runs


def validate_manifest(manifest):
    if manifest.get("schema_version") != 1:
        raise ModelCIError("unsupported model manifest schema")
    if manifest.get("runner") != "nmz4":
        raise ModelCIError("the first CI release is restricted to the nmz4 runner")
    tool = manifest.get("tool")
    if not isinstance(tool, dict) or not COMMIT.match(str(tool.get("commit", ""))):
        raise ModelCIError("the test tool must be pinned to a 40-character commit")
    archive = str(tool.get("archive", ""))
    if not archive.startswith(TEST_TOOL_ASSET_ROOT) or not archive.endswith(".tar"):
        raise ModelCIError("the test-tool archive is outside the reviewed asset root")
    if not SHA256.match(str(tool.get("archive_sha256", ""))):
        raise ModelCIError("the test-tool archive must have a pinned SHA256")
    required_files = tool.get("required_files")
    if not isinstance(required_files, list) or not required_files:
        raise ModelCIError("the test tool required file list is empty")
    models = manifest.get("models")
    scenarios = manifest.get("scenarios")
    profiles = manifest.get("profiles")
    if not all(isinstance(item, dict) for item in (models, scenarios, profiles)):
        raise ModelCIError("models, scenarios and profiles must be objects")
    for model_id, model in models.items():
        if not TOKEN.match(model_id) or not isinstance(model, dict):
            raise ModelCIError("invalid model declaration: {}".format(model_id))
        if not isinstance(model.get("ready"), bool):
            raise ModelCIError("model {} does not declare readiness".format(model_id))
        if not str(model.get("host_path", "")).startswith("/public/"):
            raise ModelCIError("model {} host path is outside /public".format(model_id))
        if not str(model.get("container_path", "")).startswith("/llm/models/"):
            raise ModelCIError("model {} container path is outside /llm/models".format(model_id))
        if int(model.get("gpu_count", 0)) not in {1, 2, 4, 8}:
            raise ModelCIError("model {} has an unsupported GPU count".format(model_id))
        if not model["ready"] and not model.get("blocked_reason"):
            raise ModelCIError("blocked model {} has no reason".format(model_id))
        environment = model.get("environment", {})
        if not isinstance(environment, dict):
            raise ModelCIError("model {} environment must be an object".format(model_id))
        for name, value in environment.items():
            if name not in ALLOWED_MODEL_ENVIRONMENT or str(value) not in ALLOWED_MODEL_ENVIRONMENT[name]:
                raise ModelCIError(
                    "model {} has an unreviewed environment value {}={!r}".format(
                        model_id, name, value
                    )
                )
        required_options = model.get("required_vllm_options", {})
        if not isinstance(required_options, dict):
            raise ModelCIError("model {} required_vllm_options must be an object".format(model_id))
        for name, value in required_options.items():
            if not TOKEN.match(str(name)) or not str(value):
                raise ModelCIError("model {} has an invalid required vLLM option".format(model_id))
    for scenario_id, scenario in scenarios.items():
        if not TOKEN.match(scenario_id) or not isinstance(scenario, dict):
            raise ModelCIError("invalid scenario: {}".format(scenario_id))
        if scenario.get("model") not in models:
            raise ModelCIError("scenario {} references an unknown model".format(scenario_id))
        config = str(scenario.get("config", ""))
        if config.startswith("/") or ".." in Path(config).parts or not config.endswith(".conf"):
            raise ModelCIError("scenario {} has an unsafe config path".format(scenario_id))
    for profile in profiles:
        value = profile_config(manifest, profile)
        devices = str(value.get("visible_devices", ""))
        if not re.match(r"^[0-7](,[0-7])*$", devices):
            raise ModelCIError("profile {} has invalid devices".format(profile))
        seen = set()
        for run in value.get("runs", []):
            if not isinstance(run, dict) or run.get("scenario") not in scenarios:
                raise ModelCIError("profile {} has an invalid scenario run".format(profile))
            checks = run.get("checks")
            if not isinstance(checks, list) or not checks or not set(checks) <= ALLOWED_CHECKS:
                raise ModelCIError("profile {} has invalid checks".format(profile))
            key = (run["scenario"], tuple(checks))
            if key in seen:
                raise ModelCIError("profile {} contains a duplicate run".format(profile))
            seen.add(key)
    return manifest


def load_manifest(path):
    return validate_manifest(read_json(path))


def selected_models(manifest, profile):
    scenario_ids = {item["scenario"] for item in selected_runs(manifest, profile)}
    model_ids = {manifest["scenarios"][item]["model"] for item in scenario_ids}
    return [(model_id, manifest["models"][model_id]) for model_id in sorted(model_ids)]


def cmd_validate(args):
    manifest = load_manifest(args.manifest)
    profile = profile_config(manifest, args.profile)
    if args.runner != manifest["runner"]:
        raise ModelCIError("profile is not assigned to runner {}".format(args.runner))
    if args.visible_devices != profile["visible_devices"]:
        raise ModelCIError(
            "visible devices {} do not match manifest {}".format(
                args.visible_devices, profile["visible_devices"]
            )
        )
    result = {
        "profile": args.profile,
        "runner": args.runner,
        "visible_devices": profile["visible_devices"],
        "run_count": len(selected_runs(manifest, args.profile)),
        "tool_required": bool(selected_runs(manifest, args.profile)),
    }
    if args.output:
        write_json(args.output, result)
    else:
        print(json.dumps(result, sort_keys=True))
    return 0


def cmd_mounts(args):
    manifest = load_manifest(args.manifest)
    for _model_id, model in selected_models(manifest, args.profile):
        print("{}\t{}".format(model["host_path"], model["container_path"]))
    return 0


def extract_test_tool_archive(archive_path, output_path, expected_sha256):
    archive = Path(archive_path)
    output = Path(output_path)
    if not archive.is_file() or archive.is_symlink():
        raise ModelCIError("the fixed test-tool archive is missing or is a symlink")
    actual_sha256 = sha256_file(archive)
    if actual_sha256 != expected_sha256:
        raise ModelCIError("test-tool archive SHA256 mismatch")
    if output.exists():
        raise ModelCIError("test-tool extraction destination already exists")
    member_count = 0
    total_size = 0
    names = set()
    with tarfile.open(str(archive), "r:*") as bundle:
        members = bundle.getmembers()
        for member in members:
            member_count += 1
            total_size += max(0, int(member.size))
            name = member.name
            parts = Path(name).parts
            if (
                not name
                or name.startswith("/")
                or "\\" in name
                or ".." in parts
                or name in names
                or not (member.isfile() or member.isdir())
            ):
                raise ModelCIError("unsafe member in test-tool archive: {}".format(name))
            names.add(name)
        if member_count > 20000 or total_size > 512 * 1024 * 1024:
            raise ModelCIError("test-tool archive exceeds the reviewed extraction limits")
        if ".git/HEAD" not in names:
            raise ModelCIError("test-tool archive does not contain Git metadata")
        output.mkdir(parents=True, mode=0o700)
        bundle.extractall(str(output))
    return {
        "archive": str(archive),
        "archive_sha256": actual_sha256,
        "member_count": member_count,
        "total_size": total_size,
    }


def cmd_extract_tool(args):
    manifest = load_manifest(args.manifest)
    tool = manifest["tool"]
    if args.archive != tool["archive"]:
        raise ModelCIError("test-tool archive does not match the reviewed manifest")
    result = extract_test_tool_archive(
        args.archive, args.output, tool["archive_sha256"]
    )
    print(json.dumps(result, sort_keys=True))
    return 0


def parse_vllm_config(path):
    parser = configparser.ConfigParser(interpolation=None, strict=False)
    parser.read(path, encoding="utf-8")
    if not parser.has_section("vllm"):
        raise ModelCIError("missing [vllm] section in {}".format(path))
    model = parser.get("vllm", "model", fallback="").strip().strip("'\"")
    tp = parser.getint("vllm", "tensor-parallel-size", fallback=1)
    options = {name: value.strip().strip("'\"") for name, value in parser.items("vllm")}
    return model, tp, options


def verify_tool(manifest, tool_root, profile):
    tool_root = Path(tool_root).resolve()
    if not (tool_root / ".git").exists():
        raise ModelCIError("test tool is not a Git checkout")
    commit = subprocess.check_output(
        ["git", "-C", str(tool_root), "rev-parse", "HEAD"], universal_newlines=True
    ).strip()
    if commit != manifest["tool"]["commit"]:
        raise ModelCIError("test tool commit {} does not match {}".format(commit, manifest["tool"]["commit"]))
    status = subprocess.check_output(
        ["git", "-C", str(tool_root), "status", "--porcelain=v1", "--untracked-files=all"],
        universal_newlines=True,
    )
    if status:
        raise ModelCIError("test tool checkout is not clean")
    for relative in manifest["tool"]["required_files"]:
        if not (tool_root / relative).is_file():
            raise ModelCIError("test tool is missing {}".format(relative))
    assets = []
    for model_id, model in selected_models(manifest, profile):
        if not model["ready"]:
            raise ModelCIError("model {} is blocked: {}".format(model_id, model["blocked_reason"]))
    for run in selected_runs(manifest, profile):
        scenario_id = run["scenario"]
        scenario = manifest["scenarios"][scenario_id]
        model = manifest["models"][scenario["model"]]
        config = tool_root / scenario["config"]
        if not config.is_file():
            raise ModelCIError("scenario {} is missing {}".format(scenario_id, scenario["config"]))
        configured_model, tensor_parallel, vllm_options = parse_vllm_config(config)
        if configured_model != model["container_path"]:
            raise ModelCIError(
                "scenario {} model path {} does not match {}".format(
                    scenario_id, configured_model, model["container_path"]
                )
            )
        if tensor_parallel != int(model["gpu_count"]):
            raise ModelCIError(
                "scenario {} tensor parallel {} does not match GPU count {}".format(
                    scenario_id, tensor_parallel, model["gpu_count"]
                )
            )
        for option, expected_value in model.get("required_vllm_options", {}).items():
            actual_value = vllm_options.get(option)
            if actual_value != str(expected_value):
                raise ModelCIError(
                    "scenario {} requires {}={!r}, found {!r}".format(
                        scenario_id, option, expected_value, actual_value
                    )
                )
        assets.append(
            {
                "scenario": scenario_id,
                "config": scenario["config"],
                "config_sha256": sha256_file(config),
                "checks": run["checks"],
                "environment": dict(model.get("environment", {})),
                "required_vllm_options": dict(model.get("required_vllm_options", {})),
            }
        )
    return {"tool_commit": commit, "profile": profile, "scenarios": assets}


def cmd_verify_tool(args):
    manifest = load_manifest(args.manifest)
    result = verify_tool(manifest, args.tool, args.profile)
    write_json(args.output, result)
    return 0


def require_integer(record, name):
    value = record.get(name)
    if isinstance(value, bool):
        raise ModelCIError("{} is not an integer".format(name))
    try:
        return int(value)
    except (TypeError, ValueError):
        raise ModelCIError("{} is not an integer".format(name))


def validate_complete_long_doc(record):
    details = record.get("long_doc_qa_result")
    if not isinstance(details, dict):
        raise ModelCIError("long_doc result is missing long_doc_qa_result")
    if record.get("long_doc_qa_tput_result") is not True:
        raise ModelCIError("long_doc_qa_tput_result is not true")
    for phase in ("warmup", "query"):
        total_name = "{}_round_prompt_count".format(phase)
        success_name = "{}_round_successful_prompt_count".format(phase)
        total = require_integer(details, total_name)
        successful = require_integer(details, success_name)
        if total <= 0 or successful != total:
            raise ModelCIError(
                "long_doc {} completed {}/{} requests".format(
                    phase, successful, total
                )
            )


def validate_result_records(path, expected_steps, require_complete_long_doc=True):
    with Path(path).open(encoding="utf-8") as handle:
        records = json.load(handle)
    if not isinstance(records, list):
        raise ModelCIError("tool result is not a JSON array")
    by_step = {}
    for record in records:
        if not isinstance(record, dict):
            raise ModelCIError("tool result contains a non-object record")
        step = record.get("step_name")
        if step:
            by_step.setdefault(step, []).append(record)
    problems = []
    for step in expected_steps:
        matches = by_step.get(step, [])
        if len(matches) != 1:
            problems.append("{} appears {} times".format(step, len(matches)))
        elif matches[0].get("result") != "success":
            problems.append("{} result is {!r}".format(step, matches[0].get("result")))
        elif step == "long_doc" and require_complete_long_doc:
            try:
                validate_complete_long_doc(matches[0])
            except ModelCIError as exc:
                problems.append(str(exc))
    if problems:
        raise ModelCIError("; ".join(problems))
    return records


def run_command(command, cwd, timeout, log_handle, environment=None):
    started = time.time()
    command_line = "$ {}\n".format(" ".join(command))
    log_handle.write(command_line)
    log_handle.flush()
    sys.stdout.write(command_line)
    sys.stdout.flush()

    def copy_output(stream):
        for line in iter(stream.readline, ""):
            log_handle.write(line)
            log_handle.flush()
            sys.stdout.write(line)
            sys.stdout.flush()

    process = subprocess.Popen(
        command,
        cwd=str(cwd),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        universal_newlines=True,
        bufsize=1,
        env=environment,
    )
    reader = threading.Thread(target=copy_output, args=(process.stdout,))
    reader.daemon = True
    reader.start()
    try:
        return_code = process.wait(timeout=timeout)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait()
        message = "command timed out after {} seconds\n".format(timeout)
        log_handle.write(message)
        log_handle.flush()
        sys.stdout.write(message)
        sys.stdout.flush()
        return_code = 124
    finally:
        reader.join(timeout=10)
        if process.stdout is not None:
            process.stdout.close()
    return return_code, time.time() - started


def scenario_commands(manifest, tool_root, scenario_id, checks, results_dir, logs_dir, work_dir):
    scenario = manifest["scenarios"][scenario_id]
    config = str(tool_root / scenario["config"])
    python = sys.executable
    commands = [
        ("prepare-output", [python, "-m", "lmcache_test.prepare_output_dir", "--logs_dir", str(logs_dir), "--results_dir", str(results_dir), "--step_name", "prepare_output_dir"], 300),
        ("check-init", [python, "-m", "lmcache_test.check_init", "--vllm_conf", config, "--logs_dir", str(logs_dir), "--results_dir", str(results_dir), "--step_name", "check_init"], 600),
        ("start-model", [python, "-m", "lmcache_test.start_model_vllm", "--vllm_conf", config, "--results_dir", str(results_dir), "--step_name", "start_model_vllm"], int(manifest["timeouts"]["start_model_seconds"])),
    ]
    expected = ["prepare_output_dir", "check_init", "start_model_vllm"]
    if "long_doc" in checks:
        commands.append(("long-doc", [python, "-m", "lmcache_test.long_doc_qa_tput", "--vllm_conf", config, "--results_dir", str(results_dir), "--step_name", "long_doc", "--timeout", str(manifest["timeouts"]["long_doc_seconds"]), "--json-output"], int(manifest["timeouts"]["long_doc_seconds"]) + 60))
        expected.append("long_doc")
    if "opencompass" in checks:
        commands.append(("opencompass", [python, "-m", "lmcache_test.opencompass_acc", "--vllm_conf", config, "--results_dir", str(results_dir), "--step_name", "opencompass", "--opencompass_dir", manifest["image_contract"]["opencompass_dir"], "--work_dir", str(work_dir / "opencompass"), "--timeout", str(manifest["timeouts"]["opencompass_seconds"]), "--acc-threshold", str(manifest["thresholds"]["humaneval"])], int(manifest["timeouts"]["opencompass_seconds"]) + 60))
        expected.append("opencompass")
    if "cmmlu" in checks:
        commands.append(("cmmlu", [python, str(tool_root / "cases/3-vllm-func/103-vllm-demo-cmmlu_prompt_long.py"), "--vllm_conf", config, "--work_dir", str(work_dir / "cmmlu"), "--strict_log_check"], int(manifest["timeouts"]["cmmlu_seconds"])))
    commands.append(("print-model-log", [python, "-m", "lmcache_test.print_model_logs", "--results_dir", str(results_dir)], 600))
    expected.append("print_model_logs")
    return commands, expected, config


def write_junit(path, cases):
    suite = ET.Element(
        "testsuite",
        name="lmcache-hcu-model-tests",
        tests=str(len(cases)),
        failures=str(sum(1 for item in cases if item.get("failure"))),
        errors="0",
        skipped="0",
        time="{:.3f}".format(sum(item["duration"] for item in cases)),
    )
    for item in cases:
        case = ET.SubElement(
            suite,
            "testcase",
            classname="lmcache_hcu.model",
            name=item["id"],
            time="{:.3f}".format(item["duration"]),
        )
        if item.get("failure"):
            failure = ET.SubElement(case, "failure", message=item["failure"])
            failure.text = item["failure"]
    tree = ET.ElementTree(suite)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    tree.write(path, encoding="utf-8", xml_declaration=True)


def run_scenario(manifest, tool_root, scenario_id, checks, repeat_index, output_root):
    case_id = "{}-repeat-{}".format(scenario_id, repeat_index)
    scenario_root = Path("/sandbox/model-runs") / case_id
    results_dir = scenario_root / "results"
    logs_dir = scenario_root / "logs"
    work_dir = scenario_root / "work"
    for directory in (results_dir, logs_dir, work_dir):
        directory.mkdir(parents=True, exist_ok=True)
    log_path = Path(output_root) / "logs" / "model-{}.log".format(case_id)
    result_path = results_dir / "lmcache_test_result.json"
    commands, expected, config = scenario_commands(
        manifest, tool_root, scenario_id, checks, results_dir, logs_dir, work_dir
    )
    model = manifest["models"][manifest["scenarios"][scenario_id]["model"]]
    scenario_environment = dict(model.get("environment", {}))
    command_environment = os.environ.copy()
    command_environment.update(scenario_environment)
    failure = None
    duration = 0.0
    cmmlu_rc = None
    model_log_printed = False
    with log_path.open("w", encoding="utf-8", errors="replace") as log_handle:
        if scenario_environment:
            environment_line = "# reviewed environment: {}\n".format(
                " ".join(
                    "{}={}".format(name, scenario_environment[name])
                    for name in sorted(scenario_environment)
                )
            )
            log_handle.write(environment_line)
            log_handle.flush()
            sys.stdout.write(environment_line)
            sys.stdout.flush()
        for step, command, timeout in commands:
            stream = "server" if step == "print-model-log" else "client"
            print(
                "::group::Model {} / {} {} log".format(case_id, step, stream),
                flush=True,
            )
            try:
                rc, elapsed = run_command(
                    command, tool_root, timeout, log_handle, command_environment
                )
            finally:
                print("::endgroup::", flush=True)
            duration += elapsed
            if step == "print-model-log":
                model_log_printed = True
            if step == "cmmlu":
                cmmlu_rc = rc
            if rc != 0 and step not in {"print-model-log"}:
                failure = "{} exited with {}".format(step, rc)
                break

        # Server output is essential failure evidence. Always print it to the
        # Actions stream, even if a client command failed before the normal
        # print-model-log command was reached.
        if not model_log_printed:
            _, command, timeout = next(
                item for item in commands if item[0] == "print-model-log"
            )
            print(
                "::group::Model {} / print-model-log server log".format(case_id),
                flush=True,
            )
            try:
                _, elapsed = run_command(
                    command, tool_root, timeout, log_handle, command_environment
                )
            finally:
                print("::endgroup::", flush=True)
            duration += elapsed

        cleanup_command = [
            sys.executable,
            "-m",
            "lmcache_test.clean_kvcache",
            "--results_dir",
            str(results_dir),
            "--vllm_conf",
            config,
            "--step_name",
            "clean_kvcache",
        ]
        cleanup_rc, elapsed = run_command(
            cleanup_command,
            tool_root,
            int(manifest["timeouts"]["cleanup_seconds"]),
            log_handle,
            command_environment,
        )
        duration += elapsed
        expected.append("clean_kvcache")
        if cleanup_rc != 0 and failure is None:
            failure = "cleanup exited with {}".format(cleanup_rc)
    raw_destination = Path(output_root) / "model-results" / "{}.json".format(case_id)
    if result_path.is_file():
        shutil.copyfile(str(result_path), str(raw_destination))
        try:
            validate_result_records(
                raw_destination,
                expected,
                bool(manifest["thresholds"].get("long_doc_require_all_requests", True)),
            )
        except Exception as exc:
            failure = failure or str(exc)
    else:
        failure = failure or "the test tool did not produce lmcache_test_result.json"
    if "cmmlu" in checks and cmmlu_rc != 0:
        failure = failure or "cmmlu check failed"
    config_destination = Path(output_root) / "effective-configs" / "{}.conf".format(case_id)
    shutil.copyfile(config, str(config_destination))
    return {
        "id": case_id,
        "scenario": scenario_id,
        "checks": checks,
        "duration": duration,
        "failure": failure,
        "result_file": raw_destination.name if raw_destination.exists() else None,
        "config_sha256": sha256_file(config_destination),
        "environment": scenario_environment,
    }


def cmd_run(args):
    manifest = load_manifest(args.manifest)
    output_root = Path(args.output).resolve()
    for directory in (
        output_root / "reports",
        output_root / "logs",
        output_root / "model-results",
        output_root / "effective-configs",
    ):
        directory.mkdir(parents=True, exist_ok=True)
    runs = selected_runs(manifest, args.profile)
    tool_root = Path(args.tool).resolve() if args.tool else None
    asset_manifest = {
        "schema_version": 1,
        "profile": args.profile,
        "runner": args.runner,
        "tool_commit": manifest["tool"]["commit"] if runs else None,
        "models": [],
    }
    if runs:
        if tool_root is None:
            raise ModelCIError("the selected profile requires a test tool checkout")
        verification = verify_tool(manifest, tool_root, args.profile)
        for model_id, model in selected_models(manifest, args.profile):
            model_path = Path(model["container_path"])
            if not model_path.is_dir():
                raise ModelCIError("model path is missing: {}".format(model_path))
            asset_manifest["models"].append(
                {
                    "id": model_id,
                    "path": str(model_path),
                    "gpu_count": model["gpu_count"],
                    "environment": dict(model.get("environment", {})),
                    "required_vllm_options": dict(model.get("required_vllm_options", {})),
                }
            )
        asset_manifest["test_tool"] = verification
    cases = []
    for repeat_index in range(1, args.repeat + 1):
        for run in runs:
            cases.append(
                run_scenario(
                    manifest,
                    tool_root,
                    run["scenario"],
                    run["checks"],
                    repeat_index,
                    output_root,
                )
            )
    write_junit(output_root / "reports" / "model-junit.xml", cases)
    inventory = {"count": len(cases), "nodeids": [item["id"] for item in cases]}
    summary = {
        "status": "failed" if any(item.get("failure") for item in cases) else "passed",
        "profile": args.profile,
        "runner": args.runner,
        "repeat": args.repeat,
        "expected": len(runs) * args.repeat,
        "cases": cases,
    }
    write_json(output_root / "model-inventory.json", inventory)
    write_json(output_root / "model-summary.json", summary)
    write_json(output_root / "asset-manifest.json", asset_manifest)
    return 0 if summary["status"] == "passed" else 1


def cmd_result_gate(args):
    validate_result_records(args.result, args.expected)
    return 0


def cmd_selftest(_args):
    with tempfile.TemporaryDirectory(prefix="lmcache-model-ci-") as temporary:
        root = Path(temporary)
        command_log = io.StringIO()
        action_log = io.StringIO()
        with contextlib.redirect_stdout(action_log):
            command_rc, _ = run_command(
                [sys.executable, "-c", "print('model-stream-marker')"],
                root,
                30,
                command_log,
            )
        if command_rc != 0:
            raise ModelCIError("model command streaming self-test failed")
        if "model-stream-marker" not in command_log.getvalue():
            raise ModelCIError("model command output was not archived")
        if "model-stream-marker" not in action_log.getvalue():
            raise ModelCIError("model command output was not streamed")
        failed = root / "failed.json"
        write_json(failed, [{"step_name": "long_doc", "result": "failed"}])
        try:
            validate_result_records(failed, ["long_doc"])
        except ModelCIError:
            pass
        else:
            raise ModelCIError("a false-green tool result was accepted")
        complete_long_doc = {
            "step_name": "long_doc",
            "result": "success",
            "long_doc_qa_tput_result": True,
            "long_doc_qa_result": {
                "warmup_round_prompt_count": 50,
                "warmup_round_successful_prompt_count": 50,
                "query_round_prompt_count": 50,
                "query_round_successful_prompt_count": 50,
            },
        }
        passed = root / "passed.json"
        write_json(passed, [complete_long_doc])
        validate_result_records(passed, ["long_doc"])
        partial = root / "partial.json"
        partial_long_doc = dict(complete_long_doc)
        partial_long_doc["long_doc_qa_result"] = dict(complete_long_doc["long_doc_qa_result"])
        partial_long_doc["long_doc_qa_result"]["query_round_successful_prompt_count"] = 19
        write_json(partial, [partial_long_doc])
        try:
            validate_result_records(partial, ["long_doc"])
        except ModelCIError:
            pass
        else:
            raise ModelCIError("a partial long-document result was accepted")
        duplicate = root / "duplicate.json"
        write_json(
            duplicate,
            [
                {"step_name": "long_doc", "result": "success"},
                {"step_name": "long_doc", "result": "success"},
            ],
        )
        try:
            validate_result_records(duplicate, ["long_doc"])
        except ModelCIError:
            pass
        else:
            raise ModelCIError("duplicate tool results were accepted")
        archive_source = root / "archive-source"
        archive_source.mkdir()
        (archive_source / ".git").mkdir()
        (archive_source / ".git" / "HEAD").write_text("ref: refs/heads/main\n", encoding="utf-8")
        archive = root / "tool.tar"
        with tarfile.open(str(archive), "w") as bundle:
            bundle.add(str(archive_source / ".git"), arcname=".git")
        extracted = root / "extracted"
        report = extract_test_tool_archive(archive, extracted, sha256_file(archive))
        if report["member_count"] < 2 or not (extracted / ".git" / "HEAD").is_file():
            raise ModelCIError("a valid fixed test-tool archive was not extracted")
        unsafe_archive = root / "unsafe.tar"
        with tarfile.open(str(unsafe_archive), "w") as bundle:
            link = tarfile.TarInfo("unsafe-link")
            link.type = tarfile.SYMTYPE
            link.linkname = "/etc/passwd"
            bundle.addfile(link)
        try:
            extract_test_tool_archive(
                unsafe_archive, root / "unsafe-output", sha256_file(unsafe_archive)
            )
        except ModelCIError:
            pass
        else:
            raise ModelCIError("an unsafe test-tool archive was accepted")
    print("LMCache-HCU model CI self-tests passed")
    return 0


def parser():
    result = argparse.ArgumentParser(description=__doc__)
    commands = result.add_subparsers(dest="command")
    validate = commands.add_parser("validate")
    validate.add_argument("--manifest", required=True)
    validate.add_argument("--profile", required=True)
    validate.add_argument("--runner", required=True)
    validate.add_argument("--visible-devices", required=True)
    validate.add_argument("--output")
    validate.set_defaults(func=cmd_validate)
    mounts = commands.add_parser("mounts")
    mounts.add_argument("--manifest", required=True)
    mounts.add_argument("--profile", required=True)
    mounts.set_defaults(func=cmd_mounts)
    extract = commands.add_parser("extract-tool")
    extract.add_argument("--manifest", required=True)
    extract.add_argument("--archive", required=True)
    extract.add_argument("--output", required=True)
    extract.set_defaults(func=cmd_extract_tool)
    verify = commands.add_parser("verify-tool")
    verify.add_argument("--manifest", required=True)
    verify.add_argument("--profile", required=True)
    verify.add_argument("--tool", required=True)
    verify.add_argument("--output", required=True)
    verify.set_defaults(func=cmd_verify_tool)
    run = commands.add_parser("run")
    run.add_argument("--manifest", required=True)
    run.add_argument("--profile", required=True)
    run.add_argument("--runner", required=True)
    run.add_argument("--repeat", required=True, type=int)
    run.add_argument("--tool")
    run.add_argument("--output", required=True)
    run.set_defaults(func=cmd_run)
    gate = commands.add_parser("result-gate")
    gate.add_argument("--result", required=True)
    gate.add_argument("--expected", action="append", required=True)
    gate.set_defaults(func=cmd_result_gate)
    selftest = commands.add_parser("selftest")
    selftest.set_defaults(func=cmd_selftest)
    return result


def main():
    args = parser().parse_args()
    if not hasattr(args, "func"):
        raise ModelCIError("a command is required")
    try:
        return int(args.func(args))
    except ModelCIError as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        return 1
    except Exception as exc:
        print("ERROR: unexpected {}: {}".format(type(exc).__name__, exc), file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

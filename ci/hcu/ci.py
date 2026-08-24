#!/usr/bin/env python3
"""Small, dependency-free helpers for the LMCache-HCU CI controller."""

from __future__ import annotations

import argparse
import base64
import csv
import hashlib
import importlib.metadata
import importlib.util
import json
import os
from pathlib import Path
import re
import shutil
import stat
import subprocess
import sys
import tempfile
import xml.etree.ElementTree as ET
import zipfile


class CIError(RuntimeError):
    pass


def read_json(path: str | Path) -> dict:
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise CIError(f"Expected a JSON object in {path}")
    return value


def write_json(path: str | Path, value: object) -> None:
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(output.name + ".tmp")
    temporary.write_text(
        json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(temporary, output)


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def base_version(version: str) -> str:
    match = re.match(r"^(\d+\.\d+\.\d+)", version)
    if match is None:
        raise CIError(f"Cannot determine base version from {version!r}")
    return match.group(1)


def run_checked(
    command: list[str],
    *,
    cwd: str | Path | None = None,
    env: dict[str, str] | None = None,
) -> subprocess.CompletedProcess[str]:
    result = subprocess.run(
        command,
        cwd=cwd,
        env=env,
        text=True,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        check=False,
    )
    if result.returncode != 0:
        raise CIError(
            f"Command failed with exit code {result.returncode}: "
            f"{' '.join(command)}\n{result.stdout}"
        )
    return result


def git_output(path: str | Path, *args: str) -> str:
    return run_checked(["git", "-C", str(path), *args]).stdout.strip()


def path_is_within(path: str | Path, root: str | Path) -> bool:
    candidate = Path(path).resolve()
    parent = Path(root).resolve()
    return candidate == parent or parent in candidate.parents


def cmd_validate(args: argparse.Namespace) -> int:
    compatibility = read_json(args.compatibility)
    patch_manifest = read_json(args.patch_manifest)
    if compatibility.get("schema_version") != 1:
        raise CIError("Unsupported compatibility schema")
    if patch_manifest.get("schema_version") != 1:
        raise CIError("Unsupported patch manifest schema")
    expected_modules = {
        "lmcache",
        "lmcache.integration.vllm.lmcache_connector_v1",
    }
    targets = patch_manifest.get("targets", [])
    if not isinstance(targets, list) or len(targets) != 2:
        raise CIError("Exactly two source patch targets are required")
    modules = {item.get("module") for item in targets if isinstance(item, dict)}
    if modules != expected_modules:
        raise CIError(f"Unexpected source patch target set: {modules}")
    if patch_manifest.get("upstream_commit") != compatibility["lmcache"]["source_commit"]:
        raise CIError("Patch manifest and compatibility source commits differ")
    for target in targets:
        for field in (
            "module",
            "relative_path",
            "preimage_sha256",
            "marker",
            "begin_marker",
        ):
            if not isinstance(target.get(field), str) or not target[field]:
                raise CIError(f"Patch target is missing {field}")
        if target.get("required") is not True:
            raise CIError(f"Patch target {target['module']} must be required")
        if not re.fullmatch(r"[0-9a-f]{64}", target["preimage_sha256"]):
            raise CIError(f"Invalid preimage SHA for {target['module']}")
    if args.profile not in {"pr", "manual", "weekly"}:
        raise CIError(f"Unsupported profile: {args.profile}")
    if args.repeat not in {1, 2, 3}:
        raise CIError("repeat must be 1, 2, or 3")
    if not re.fullmatch(r"[^\s]+@sha256:[0-9a-f]{64}", args.base_image):
        raise CIError("Base image must use an immutable @sha256 digest")
    if not Path(args.checkout).is_dir():
        raise CIError(f"Checkout does not exist: {args.checkout}")
    if not Path(args.upstream).is_dir():
        raise CIError(f"Upstream source does not exist: {args.upstream}")
    print("CI configuration is valid")
    return 0


def detect_hcu_arch(torch_module: object) -> str:
    properties = torch_module.cuda.get_device_properties(0)
    candidates = [
        getattr(properties, "gcnArchName", None),
        getattr(properties, "gcn_arch_name", None),
        getattr(properties, "arch", None),
        str(properties),
    ]
    for value in candidates:
        if value:
            match = re.search(r"gfx[0-9a-z]+", str(value), re.IGNORECASE)
            if match:
                return match.group(0).lower()
    rocminfo = shutil.which("rocminfo")
    if rocminfo:
        result = run_checked([rocminfo])
        match = re.search(r"\bgfx[0-9a-z]+\b", result.stdout, re.IGNORECASE)
        if match:
            return match.group(0).lower()
    raise CIError("Unable to detect the HCU architecture")


def cmd_probe(args: argparse.Namespace) -> int:
    compatibility = read_json(args.compatibility)
    try:
        import torch
    except Exception as exc:
        raise CIError(f"Cannot import torch: {exc}") from exc

    python_version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
    minimum = tuple(int(item) for item in compatibility["python"]["minimum"].split("."))
    maximum = tuple(
        int(item) for item in compatibility["python"]["maximum_exclusive"].split(".")
    )
    current_python = (sys.version_info.major, sys.version_info.minor)
    if not (minimum <= current_python < maximum):
        raise CIError(f"Unsupported Python version: {python_version}")

    torch_version = importlib.metadata.version("torch")
    vllm_version = importlib.metadata.version("vllm")
    if base_version(torch_version) != compatibility["torch"]["version"]:
        raise CIError(
            f"Torch mismatch: expected {compatibility['torch']['version']}, got {torch_version}"
        )
    if base_version(vllm_version) != compatibility["vllm"]["version"]:
        raise CIError(
            f"vLLM mismatch: expected {compatibility['vllm']['version']}, got {vllm_version}"
        )
    if shutil.which("hipcc") is None:
        raise CIError("hipcc is not available on PATH")
    dtk_home = Path(compatibility["dtk"]["home"])
    if not dtk_home.is_dir():
        raise CIError(f"DTK home is missing: {dtk_home}")
    if not torch.cuda.is_available():
        raise CIError("torch.cuda.is_available() is false")
    if torch.cuda.device_count() < 1:
        raise CIError("No HCU device is visible")
    if args.expected_device_count and torch.cuda.device_count() != args.expected_device_count:
        raise CIError(
            "HCU device count mismatch: expected "
            f"{args.expected_device_count}, got {torch.cuda.device_count()}"
        )
    try:
        base_lmcache_hcu = importlib.metadata.version("lmcache-hcu")
    except importlib.metadata.PackageNotFoundError:
        base_lmcache_hcu = None

    arch = detect_hcu_arch(torch)
    if args.expected_arch and arch != args.expected_arch:
        raise CIError(f"HCU architecture mismatch: expected {args.expected_arch}, got {arch}")
    abi = int(bool(torch._C._GLIBCXX_USE_CXX11_ABI))
    report = {
        "python": python_version,
        "torch": torch_version,
        "vllm": vllm_version,
        "torch_cxx11_abi": abi,
        "hcu_arch": arch,
        "device_count": torch.cuda.device_count(),
        "device_name": torch.cuda.get_device_name(0),
        "hipcc": shutil.which("hipcc"),
        "dtk_home": str(dtk_home),
        "base_lmcache_hcu": base_lmcache_hcu,
    }
    write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


def verify_upstream_source(path: Path, compatibility: dict) -> dict:
    expected_commit = compatibility["lmcache"]["source_commit"]
    expected_tag = compatibility["lmcache"]["source_tag"]
    if not (path / ".git").exists():
        raise CIError(f"Upstream source is not a Git checkout: {path}")
    actual_commit = git_output(path, "rev-parse", "HEAD")
    if actual_commit != expected_commit:
        raise CIError(
            f"Upstream commit mismatch: expected {expected_commit}, got {actual_commit}"
        )
    actual_tag = git_output(path, "describe", "--tags", "--exact-match", "HEAD")
    if actual_tag != expected_tag:
        raise CIError(f"Upstream tag mismatch: expected {expected_tag}, got {actual_tag}")
    tests_init = path / "tests" / "__init__.py"
    if not tests_init.is_file():
        raise CIError(f"Upstream tests package is missing: {tests_init}")
    dirty = git_output(path, "status", "--porcelain=v1", "--untracked-files=all")
    if dirty:
        raise CIError(f"Upstream source must be clean:\n{dirty}")
    return {
        "path": str(path.resolve()),
        "commit": actual_commit,
        "tag": actual_tag,
    }


def cmd_verify_upstream(args: argparse.Namespace) -> int:
    report = verify_upstream_source(Path(args.source), read_json(args.compatibility))
    write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


def decode_record_hash(value: str) -> tuple[str, bytes]:
    algorithm, encoded = value.split("=", 1)
    padding = "=" * (-len(encoded) % 4)
    return algorithm, base64.urlsafe_b64decode(encoded + padding)


def cmd_verify_wheel(args: argparse.Namespace) -> int:
    wheel_dir = Path(args.wheel_dir)
    wheels = sorted(wheel_dir.glob("*.whl"))
    if len(wheels) != 1:
        raise CIError(f"Expected one wheel, found {len(wheels)} in {wheel_dir}")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        infos = archive.infolist()
        names = [item.filename for item in infos]
        if len(infos) > 4096:
            raise CIError(f"Wheel contains too many entries: {len(infos)}")
        if sum(item.file_size for item in infos) > 2 * 1024 * 1024 * 1024:
            raise CIError("Wheel uncompressed content exceeds 2 GiB")
        if len(names) != len(set(names)) or len(names) != len({name.casefold() for name in names}):
            raise CIError("Wheel contains duplicate or case-colliding paths")
        for info in infos:
            name = info.filename
            path_parts = Path(name.replace("\\", "/")).parts
            mode = (info.external_attr >> 16) & 0xFFFF
            if (
                not name
                or "\\" in name
                or name.startswith(("/", "\\"))
                or ".." in path_parts
                or "\x00" in name
                or info.flag_bits & 0x1
                or stat.S_ISLNK(mode)
            ):
                raise CIError(f"Wheel contains an unsafe ZIP entry: {name!r}")
        dist_infos = sorted(
            {name.split("/", 1)[0] for name in names if ".dist-info/" in name}
        )
        if len(dist_infos) != 1:
            raise CIError(f"Expected one .dist-info directory, found {dist_infos}")
        dist_info = dist_infos[0]
        allowed_prefixes = ("lmcache_hcu/", dist_info + "/")
        unexpected = [
            name for name in names if name and not name.endswith("/") and not name.startswith(allowed_prefixes)
        ]
        if unexpected:
            raise CIError(f"Wheel contains files outside the overlay boundary: {unexpected}")
        forbidden = [name for name in names if name.startswith(("lmcache/", "vllm/"))]
        if forbidden:
            raise CIError(f"Wheel contains forbidden upstream files: {forbidden}")
        native = [
            name
            for name in names
            if re.fullmatch(r"lmcache_hcu/hcu_c_ops[^/]*\.(?:so|pyd)", name)
        ]
        if len(native) != 1:
            raise CIError(f"Expected one hcu_c_ops native extension, found {native}")

        metadata_text = archive.read(f"{dist_info}/METADATA").decode("utf-8")
        name_match = re.search(r"^Name:\s*(.+)$", metadata_text, re.MULTILINE)
        version_match = re.search(r"^Version:\s*(.+)$", metadata_text, re.MULTILINE)
        if not name_match or name_match.group(1).strip().lower() != "lmcache-hcu":
            raise CIError("Wheel package name is not lmcache-hcu")
        if not version_match:
            raise CIError("Wheel metadata does not contain a version")
        version = version_match.group(1).strip()
        version_match = re.fullmatch(
            r"0\.3\.13\+hcu\.\d{10}\.([0-9a-f]{7,40})", version
        )
        built_sha = version_match.group(1) if version_match else ""
        if not version_match or not args.sha.startswith(built_sha):
            raise CIError(
                f"Wheel version {version!r} does not identify source commit {args.sha}"
            )

        entry_points = archive.read(f"{dist_info}/entry_points.txt").decode("utf-8")
        required_entries = {
            "lmcache-hcu-apply-patches = lmcache_hcu.integration.patch.apply_patch:main",
            "lmcache-hcu-info = lmcache_hcu.env:main",
        }
        if not required_entries.issubset(set(entry_points.splitlines())):
            raise CIError("Wheel console entry points do not match the required contract")

        wheel_text = archive.read(f"{dist_info}/WHEEL").decode("utf-8")
        python_tag = f"cp{sys.version_info.major}{sys.version_info.minor}-"
        tags = re.findall(r"^Tag:\s*(.+)$", wheel_text, re.MULTILINE)
        if not tags or not any(tag.startswith(python_tag) for tag in tags):
            raise CIError(f"Wheel tags {tags} do not match the running Python {python_tag}")

        record_name = f"{dist_info}/RECORD"
        rows = csv.reader(archive.read(record_name).decode("utf-8").splitlines())
        record_paths: set[str] = set()
        for record_path, hash_value, size_value in rows:
            record_paths.add(record_path)
            if record_path == record_name:
                if hash_value or size_value:
                    raise CIError("The RECORD self-entry must not contain a hash or size")
                continue
            if record_path not in names:
                raise CIError(f"RECORD references a missing file: {record_path}")
            payload = archive.read(record_path)
            if size_value and int(size_value) != len(payload):
                raise CIError(f"RECORD size mismatch for {record_path}")
            algorithm, expected_hash = decode_record_hash(hash_value)
            if algorithm != "sha256":
                raise CIError(f"Unsupported RECORD hash algorithm for {record_path}: {algorithm}")
            if hashlib.sha256(payload).digest() != expected_hash:
                raise CIError(f"RECORD hash mismatch for {record_path}")
        if set(name for name in names if not name.endswith("/")) != record_paths:
            raise CIError("RECORD does not cover every wheel file")

    report = {
        "wheel": str(wheel.resolve()),
        "wheel_sha256": sha256_file(wheel),
        "version": version,
        "native_extension": native[0],
        "tags": tags,
        "file_count": len([name for name in names if not name.endswith("/")]),
    }
    write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


def cmd_verify_install(args: argparse.Namespace) -> int:
    distribution = importlib.metadata.distribution("lmcache-hcu")
    version = distribution.version
    suffix = version.rsplit(".", 1)[-1]
    if not re.fullmatch(r"[0-9a-f]{7,40}", suffix) or not args.sha.startswith(suffix):
        raise CIError(f"Installed version {version} does not identify {args.sha}")
    import lmcache_hcu
    import lmcache_hcu.hcu_c_ops as native

    package_path = Path(lmcache_hcu.__file__).resolve()
    native_path = Path(native.__file__).resolve()
    venv_root = Path(args.venv).resolve()
    if path_is_within(package_path, args.source):
        raise CIError(f"lmcache_hcu was imported from the source tree: {package_path}")
    if not path_is_within(package_path, venv_root):
        raise CIError(f"lmcache_hcu was not imported from the disposable venv: {package_path}")
    if not path_is_within(native_path, venv_root):
        raise CIError(f"hcu_c_ops was not imported from the disposable venv: {native_path}")
    distribution_root = Path(distribution.locate_file("")).resolve()
    if not path_is_within(distribution_root, venv_root):
        raise CIError(
            f"lmcache-hcu distribution metadata is outside the disposable venv: {distribution_root}"
        )
    entry_points = {
        item.name: item.value
        for item in importlib.metadata.entry_points(group="console_scripts")
        if item.name.startswith("lmcache-hcu-")
    }
    expected = {
        "lmcache-hcu-apply-patches": "lmcache_hcu.integration.patch.apply_patch:main",
        "lmcache-hcu-info": "lmcache_hcu.env:main",
    }
    if entry_points != expected:
        raise CIError(f"Installed entry points differ from the contract: {entry_points}")
    report = {
        "version": version,
        "package_path": str(package_path),
        "native_path": str(native_path),
        "runtime_patched": bool(lmcache_hcu.LMCACHE_HCU_PATCHED),
        "entry_points": entry_points,
    }
    if not report["runtime_patched"]:
        raise CIError("Importing lmcache_hcu did not activate runtime patches")
    write_json(args.output, report)
    print(json.dumps(report, sort_keys=True))
    return 0


def module_file(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise CIError(f"Cannot locate module {module_name}")
    return Path(spec.origin).resolve()


def source_snapshot(root: Path) -> dict[str, str]:
    result: dict[str, str] = {}
    for path in sorted(root.rglob("*")):
        if (
            not path.is_file()
            or path.is_symlink()
            or ".git" in path.parts
            or "__pycache__" in path.parts
            or path.suffix == ".pyc"
            or ".bak." in path.name
        ):
            continue
        result[str(path.relative_to(root))] = sha256_file(path)
    return result


def backup_snapshot(root: Path) -> set[str]:
    return {
        str(path.relative_to(root))
        for path in root.rglob("*.bak.*")
        if path.is_file() and ".git" not in path.parts
    }


def invoke_patch(mode: str) -> str:
    env = dict(os.environ)
    env.pop("SKIP_LMCACHE_HCU_PATCH", None)
    result = run_checked(
        [
            sys.executable,
            "-m",
            "lmcache_hcu.integration.patch.apply_patch",
            "--mode",
            mode,
        ],
        env=env,
    )
    return result.stdout


def cmd_patch_gate(args: argparse.Namespace) -> int:
    output = Path(args.output)
    report: dict[str, object] = {"status": "failed", "events": []}
    original_bytes: dict[Path, bytes] = {}
    try:
        manifest = read_json(args.manifest)
        upstream = Path(args.upstream).resolve()
        lmcache_version = importlib.metadata.version("lmcache")
        if base_version(lmcache_version) != "0.3.13":
            raise CIError(f"Expected LMCache 0.3.13, got {lmcache_version}")
        targets: list[dict[str, object]] = []
        for declaration in manifest["targets"]:
            target = module_file(declaration["module"])
            if not path_is_within(target, upstream):
                raise CIError(f"Patch target escaped the fixed upstream source: {target}")
            expected_path = upstream / declaration["relative_path"]
            if target != expected_path.resolve():
                raise CIError(
                    f"Patch target path mismatch: expected {expected_path}, got {target}"
                )
            if sha256_file(target) != declaration["preimage_sha256"]:
                raise CIError(f"Patch target preimage differs from the manifest: {target}")
            text = target.read_text(encoding="utf-8")
            anchor = declaration.get("anchor")
            if anchor and text.count(anchor) != 1:
                raise CIError(
                    f"Expected one exact connector anchor in {target}, found {text.count(anchor)}"
                )
            if text.count(declaration["marker"]) or text.count(declaration["begin_marker"]):
                raise CIError(f"Patch target is not pristine: {target}")
            targets.append(
                {
                    **declaration,
                    "path": str(target),
                    "relative_path": str(target.relative_to(upstream)),
                    "initial_sha256": sha256_file(target),
                }
            )
            original_bytes[target] = target.read_bytes()
        if backup_snapshot(upstream):
            raise CIError("The fixed upstream copy already contains patch backup files")
        before_tree = source_snapshot(upstream)
        before_vllm_root = module_file("vllm").parent
        before_vllm = source_snapshot(before_vllm_root)

        report["events"].append({"first_install": invoke_patch("install")})
        after_first = source_snapshot(upstream)
        changed = {
            path
            for path in set(before_tree) | set(after_first)
            if before_tree.get(path) != after_first.get(path)
        }
        expected_changed = {item["relative_path"] for item in targets}
        if changed != expected_changed:
            raise CIError(f"Unexpected source patch changes: {sorted(changed)}")
        for item in targets:
            path = Path(item["path"])
            text = path.read_text(encoding="utf-8")
            if text.count(item["begin_marker"]) != 1 or text.count(item["marker"]) != 1:
                raise CIError(f"Patch markers are missing or duplicated in {path}")
            anchor = item.get("anchor")
            if anchor and text.index(item["begin_marker"]) > text.index(anchor):
                raise CIError(f"Connector patch was not inserted before the exact anchor in {path}")
            if sha256_file(path) == item["initial_sha256"]:
                raise CIError(f"Patch did not change {path}")
            item["patched_sha256"] = sha256_file(path)
        backups_after_first = backup_snapshot(upstream)
        if len(backups_after_first) != len(targets):
            raise CIError(
                f"Expected {len(targets)} patch backups, found {len(backups_after_first)}"
            )
        for item in targets:
            target_path = Path(item["path"])
            matching = list(target_path.parent.glob(target_path.name + ".bak.*"))
            if len(matching) != 1:
                raise CIError(f"Expected one backup for {target_path}, found {matching}")
        if source_snapshot(before_vllm_root) != before_vllm:
            raise CIError("Source patch unexpectedly changed the installed vLLM tree")

        report["events"].append({"second_install": invoke_patch("install")})
        if source_snapshot(upstream) != after_first:
            raise CIError("The second source patch installation was not idempotent")
        if backup_snapshot(upstream) != backups_after_first:
            raise CIError("The second source patch installation created another backup")

        run_checked([sys.executable, "-m", "compileall", "-q", str(upstream)])
        import_check = (
            "import warnings; "
            "warnings.filterwarnings('error', message='LMCache-HCU .* patch import failed'); "
            "import lmcache; "
            "import lmcache.integration.vllm.lmcache_connector_v1; "
            "import lmcache_hcu; "
            "assert lmcache_hcu.LMCACHE_HCU_PATCHED is True"
        )
        run_checked([sys.executable, "-c", import_check])

        report["events"].append({"uninstall": invoke_patch("uninstall")})
        for item in targets:
            path = Path(item["path"])
            if sha256_file(path) != item["initial_sha256"]:
                raise CIError(f"Patch uninstall did not restore {path}")
            text = path.read_text(encoding="utf-8")
            if item["marker"] in text or item["begin_marker"] in text:
                raise CIError(f"Patch marker remained after uninstall in {path}")

        report["events"].append({"test_install": invoke_patch("install")})
        for item in targets:
            path = Path(item["path"])
            if item["marker"] not in path.read_text(encoding="utf-8"):
                raise CIError(f"Final test-state patch is missing from {path}")
            if sha256_file(path) != item["patched_sha256"]:
                raise CIError(f"Final patch content differs from the verified first apply: {path}")

        report.update(
            {
                "status": "passed",
                "lmcache_version": lmcache_version,
                "upstream": str(upstream),
                "targets": targets,
                "changed_files": sorted(changed),
                "backups": sorted(backup_snapshot(upstream)),
            }
        )
        write_json(output, report)
        print(json.dumps(report, sort_keys=True))
        return 0
    except Exception as exc:
        report["error"] = str(exc)
        rollback_errors = []
        for path, payload in original_bytes.items():
            try:
                path.write_bytes(payload)
                if sha256_file(path) != hashlib.sha256(payload).hexdigest():
                    raise CIError("restored hash mismatch")
                for backup in path.parent.glob(path.name + ".bak.*"):
                    backup.unlink()
            except Exception as rollback_exc:
                rollback_errors.append(f"{path}: {rollback_exc}")
        if rollback_errors:
            report["rollback_errors"] = rollback_errors
        write_json(output, report)
        raise


def cmd_patch_cleanup(args: argparse.Namespace) -> int:
    report = read_json(args.report)
    if report.get("status") != "passed":
        raise CIError("Cannot verify cleanup without a passed patch gate report")
    output = invoke_patch("uninstall")
    for item in report["targets"]:
        path = Path(item["path"])
        if sha256_file(path) != item["initial_sha256"]:
            raise CIError(f"Final patch cleanup did not restore {path}")
        text = path.read_text(encoding="utf-8")
        if item["marker"] in text or item["begin_marker"] in text:
            raise CIError(f"Final patch cleanup left markers in {path}")
        for backup in path.parent.glob(path.name + ".bak.*"):
            backup.unlink()
        if list(path.parent.glob(path.name + ".bak.*")):
            raise CIError(f"Final patch cleanup left backups for {path}")
    print(output)
    return 0


class InventoryPlugin:
    def __init__(self) -> None:
        self.nodeids: list[str] = []

    def pytest_collection_finish(self, session: object) -> None:
        self.nodeids = [item.nodeid for item in session.items]


def cmd_discover(args: argparse.Namespace) -> int:
    output = Path(args.output)
    if args.collect_child:
        try:
            import pytest
        except Exception as exc:
            raise CIError(f"pytest is unavailable: {exc}") from exc
        plugin = InventoryPlugin()
        previous = Path.cwd()
        execute_dir = Path(args.execute_dir)
        execute_dir.mkdir(parents=True, exist_ok=True)
        os.chdir(execute_dir)
        try:
            return_code = pytest.main(
                [
                    "-c",
                    str(Path(args.config).resolve()),
                    "--collect-only",
                    "-q",
                    "-p",
                    "no:cacheprovider",
                    str(Path(args.tests).resolve()),
                ],
                plugins=[plugin],
            )
        finally:
            os.chdir(previous)
        if return_code != pytest.ExitCode.OK:
            raise CIError(f"pytest collection failed with exit code {int(return_code)}")
        if not plugin.nodeids:
            raise CIError("pytest collected zero tests")
        write_json(output, {"nodeids": plugin.nodeids})
        return 0

    temporary = output.with_name(output.name + ".child")
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "discover",
        "--tests",
        str(Path(args.tests).resolve()),
        "--config",
        str(Path(args.config).resolve()),
        "--execute-dir",
        str(Path(args.execute_dir).resolve()),
        "--output",
        str(temporary),
        "--collect-child",
    ]
    result = subprocess.run(command, check=False)
    if result.returncode != 0:
        raise CIError(f"pytest collection subprocess failed with exit code {result.returncode}")
    child = read_json(temporary)
    temporary.unlink()
    nodeids = child.get("nodeids", [])
    if not isinstance(nodeids, list) or not all(isinstance(item, str) for item in nodeids):
        raise CIError("pytest collection subprocess returned an invalid nodeid list")
    if not nodeids:
        raise CIError("pytest collected zero tests")
    if len(set(nodeids)) != len(nodeids):
        raise CIError("pytest collected duplicate nodeids")
    inventory = {
        "count": len(nodeids),
        "nodeids": nodeids,
        "tests_root": str(Path(args.tests).resolve()),
    }
    write_json(output, inventory)
    print(f"Collected {len(nodeids)} tests")
    return 0


def cmd_execute(args: argparse.Namespace) -> int:
    inventory = read_json(args.inventory)
    nodeids = inventory.get("nodeids", [])
    if not nodeids:
        raise CIError("Test inventory is empty")
    tests_root = Path(inventory.get("tests_root", "")).resolve()
    if not tests_root.is_dir():
        raise CIError(f"Test inventory has an invalid tests root: {tests_root}")
    command = [
        sys.executable,
        "-m",
        "pytest",
        "-c",
        str(Path(args.config).resolve()),
        "-p",
        "no:cacheprovider",
        "--strict-markers",
        "--tb=short",
        "-W",
        "error:LMCache-HCU .* patch import failed:RuntimeWarning",
        f"--junitxml={Path(args.junit).resolve()}",
        *nodeids,
    ]
    # Collection stores nodeids relative to the tests root (for example
    # test_config.py::test_x). Execute from that same root so pytest resolves
    # every collected nodeid to the exact file that produced it.
    result = subprocess.run(command, cwd=tests_root, check=False)
    Path(args.rc_output).write_text(str(result.returncode) + "\n", encoding="utf-8")
    return result.returncode


def junit_stats(path: Path) -> dict[str, int]:
    try:
        tree = ET.parse(path)
    except (ET.ParseError, OSError) as exc:
        raise CIError(f"Invalid JUnit XML {path}: {exc}") from exc
    testcases = list(tree.getroot().iter("testcase"))
    return {
        "tests": len(testcases),
        "failures": sum(1 for case in testcases if case.find("failure") is not None),
        "errors": sum(1 for case in testcases if case.find("error") is not None),
        "skipped": sum(1 for case in testcases if case.find("skipped") is not None),
    }


def cmd_aggregate(args: argparse.Namespace) -> int:
    inventory = read_json(args.inventory)
    expected = int(inventory["count"])
    repeats: list[dict[str, object]] = []
    problems: list[str] = []
    for repeat in range(1, args.repeat + 1):
        junit = Path(args.junit_dir) / f"junit-repeat-{repeat}.xml"
        rc_path = Path(args.state_dir) / f"test-repeat-{repeat}.rc"
        if not rc_path.is_file():
            problems.append(f"repeat {repeat}: missing pytest exit code")
            rc = None
        else:
            rc = int(rc_path.read_text(encoding="utf-8").strip())
            if rc != 0:
                problems.append(f"repeat {repeat}: pytest exit code {rc}")
        try:
            stats = junit_stats(junit)
        except CIError as exc:
            problems.append(str(exc))
            stats = {"tests": 0, "failures": 0, "errors": 1, "skipped": 0}
        if stats["tests"] != expected:
            problems.append(
                f"repeat {repeat}: expected {expected} JUnit cases, got {stats['tests']}"
            )
        if stats["failures"] or stats["errors"] or stats["skipped"]:
            problems.append(
                f"repeat {repeat}: failures={stats['failures']} "
                f"errors={stats['errors']} skipped={stats['skipped']}"
            )
        repeats.append({"repeat": repeat, "pytest_rc": rc, **stats})
    summary = {
        "status": "passed" if not problems else "failed",
        "expected_per_repeat": expected,
        "expected_total": expected * args.repeat,
        "repeat_count": args.repeat,
        "repeats": repeats,
        "problems": problems,
    }
    write_json(args.output, summary)
    markdown = [
        "# LMCache-HCU test summary",
        "",
        f"- Status: `{summary['status']}`",
        f"- Collected tests: `{expected}`",
        f"- Repetitions: `{args.repeat}`",
        f"- Expected total cases: `{expected * args.repeat}`",
    ]
    if problems:
        markdown.extend(["", "## Problems", "", *[f"- {item}" for item in problems]])
    Path(args.markdown).write_text("\n".join(markdown) + "\n", encoding="utf-8")
    print(json.dumps(summary, sort_keys=True))
    return 0 if not problems else 1


def cmd_synthetic_junit(args: argparse.Namespace) -> int:
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    suite = ET.Element(
        "testsuite",
        name="lmcache-hcu-ci",
        tests="1",
        failures="1",
        errors="0",
        skipped="0",
        time="0",
    )
    case = ET.SubElement(suite, "testcase", classname="ci.stage", name=args.name, time="0")
    failure = ET.SubElement(case, "failure", message=args.message, type="CIStageFailure")
    failure.text = args.details or args.message
    tree = ET.ElementTree(suite)
    ET.indent(tree, space="  ")
    tree.write(output, encoding="utf-8", xml_declaration=True)
    return 0


def cmd_selftest(args: argparse.Namespace) -> int:
    with tempfile.TemporaryDirectory(prefix="lmcache-hcu-ci-selftest-") as temporary:
        root = Path(temporary)
        junit = root / "junit.xml"
        cmd_synthetic_junit(
            argparse.Namespace(
                output=str(junit), name="injected", message="expected", details="selftest"
            )
        )
        stats = junit_stats(junit)
        if stats != {"tests": 1, "failures": 1, "errors": 0, "skipped": 0}:
            raise CIError(f"Synthetic JUnit self-test failed: {stats}")

        corrupt = root / "corrupt.xml"
        corrupt.write_text("<testsuite>", encoding="utf-8")
        try:
            junit_stats(corrupt)
        except CIError:
            pass
        else:
            raise CIError("Corrupt JUnit was not rejected")

        tests = root / "tests"
        execute = root / "execute"
        reports = root / "reports"
        state = root / "state"
        tests.mkdir()
        execute.mkdir()
        reports.mkdir()
        state.mkdir()
        (tests / "pytest.ini").write_text("[pytest]\n", encoding="utf-8")
        (tests / "test_selftest.py").write_text(
            "def test_discover_execute_contract():\n    assert True\n",
            encoding="utf-8",
        )
        inventory = root / "inventory.json"
        cmd_discover(
            argparse.Namespace(
                tests=str(tests),
                config=str(tests / "pytest.ini"),
                execute_dir=str(execute),
                output=str(inventory),
                collect_child=False,
            )
        )
        execute_rc = cmd_execute(
            argparse.Namespace(
                inventory=str(inventory),
                config=str(tests / "pytest.ini"),
                execute_dir=str(execute),
                junit=str(reports / "junit.xml"),
                rc_output=str(state / "pytest.rc"),
            )
        )
        if execute_rc != 0 or junit_stats(reports / "junit.xml")["tests"] != 1:
            raise CIError("Discover-to-execute nodeid contract self-test failed")

    print("LMCache-HCU CI helper self-tests passed")
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    validate = subparsers.add_parser("validate")
    validate.add_argument("--compatibility", required=True)
    validate.add_argument("--patch-manifest", required=True)
    validate.add_argument("--base-image", required=True)
    validate.add_argument("--profile", required=True)
    validate.add_argument("--repeat", type=int, required=True)
    validate.add_argument("--checkout", required=True)
    validate.add_argument("--upstream", required=True)
    validate.set_defaults(func=cmd_validate)

    probe = subparsers.add_parser("probe")
    probe.add_argument("--compatibility", required=True)
    probe.add_argument("--expected-arch", default="")
    probe.add_argument("--expected-device-count", type=int, default=0)
    probe.add_argument("--output", required=True)
    probe.set_defaults(func=cmd_probe)

    upstream = subparsers.add_parser("verify-upstream")
    upstream.add_argument("--compatibility", required=True)
    upstream.add_argument("--source", required=True)
    upstream.add_argument("--output", required=True)
    upstream.set_defaults(func=cmd_verify_upstream)

    wheel = subparsers.add_parser("verify-wheel")
    wheel.add_argument("--wheel-dir", required=True)
    wheel.add_argument("--sha", required=True)
    wheel.add_argument("--output", required=True)
    wheel.set_defaults(func=cmd_verify_wheel)

    install = subparsers.add_parser("verify-install")
    install.add_argument("--sha", required=True)
    install.add_argument("--source", required=True)
    install.add_argument("--venv", required=True)
    install.add_argument("--output", required=True)
    install.set_defaults(func=cmd_verify_install)

    patch = subparsers.add_parser("patch-gate")
    patch.add_argument("--manifest", required=True)
    patch.add_argument("--upstream", required=True)
    patch.add_argument("--output", required=True)
    patch.set_defaults(func=cmd_patch_gate)

    patch_cleanup = subparsers.add_parser("patch-cleanup")
    patch_cleanup.add_argument("--report", required=True)
    patch_cleanup.set_defaults(func=cmd_patch_cleanup)

    discover = subparsers.add_parser("discover")
    discover.add_argument("--tests", required=True)
    discover.add_argument("--config", required=True)
    discover.add_argument("--execute-dir", required=True)
    discover.add_argument("--output", required=True)
    discover.add_argument("--collect-child", action="store_true")
    discover.set_defaults(func=cmd_discover)

    execute = subparsers.add_parser("execute")
    execute.add_argument("--inventory", required=True)
    execute.add_argument("--config", required=True)
    execute.add_argument("--execute-dir", required=True)
    execute.add_argument("--junit", required=True)
    execute.add_argument("--rc-output", required=True)
    execute.set_defaults(func=cmd_execute)

    aggregate = subparsers.add_parser("aggregate")
    aggregate.add_argument("--inventory", required=True)
    aggregate.add_argument("--repeat", type=int, required=True)
    aggregate.add_argument("--junit-dir", required=True)
    aggregate.add_argument("--state-dir", required=True)
    aggregate.add_argument("--output", required=True)
    aggregate.add_argument("--markdown", required=True)
    aggregate.set_defaults(func=cmd_aggregate)

    synthetic = subparsers.add_parser("synthetic-junit")
    synthetic.add_argument("--output", required=True)
    synthetic.add_argument("--name", required=True)
    synthetic.add_argument("--message", required=True)
    synthetic.add_argument("--details", default="")
    synthetic.set_defaults(func=cmd_synthetic_junit)

    selftest = subparsers.add_parser("selftest")
    selftest.set_defaults(func=cmd_selftest)
    return parser


def main() -> int:
    parser = build_parser()
    args = parser.parse_args()
    try:
        return int(args.func(args))
    except CIError as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"ERROR: unexpected {type(exc).__name__}: {exc}", file=sys.stderr)
        return 1


if __name__ == "__main__":
    raise SystemExit(main())

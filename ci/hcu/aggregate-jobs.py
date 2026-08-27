#!/usr/bin/env python3
"""Validate split LMCache-HCU job archives and publish one run summary."""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import sys


class AggregateError(RuntimeError):
    pass


TOKEN = re.compile(r"^[A-Za-z0-9_.-]+$")


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def read_json(path):
    with Path(path).open(encoding="utf-8") as handle:
        value = json.load(handle)
    if not isinstance(value, dict):
        raise AggregateError("expected a JSON object in {}".format(path))
    return value


def atomic_text(path, value):
    path = Path(path)
    temporary = path.with_name(path.name + ".tmp")
    with temporary.open("x", encoding="utf-8") as handle:
        handle.write(value)
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(str(temporary), str(path))


def atomic_json(path, value):
    atomic_text(path, json.dumps(value, indent=2, sort_keys=True) + "\n")


def verify_checksums(job_root):
    sums = job_root / "SHA256SUMS"
    if not sums.is_file():
        raise AggregateError("missing {}".format(sums))
    for line in sums.read_text(encoding="utf-8").splitlines():
        expected, separator, relative = line.partition("  ")
        if not separator or not re.fullmatch(r"[0-9a-f]{64}", expected):
            raise AggregateError("invalid checksum line in {}".format(sums))
        path = job_root / relative
        resolved = path.resolve()
        if job_root.resolve() not in resolved.parents or not path.is_file():
            raise AggregateError("checksum path escaped job archive: {}".format(relative))
        if sha256_file(path) != expected:
            raise AggregateError("checksum mismatch: {}/{}".format(job_root.name, relative))


def aggregate(args):
    for value, label in (
        (args.profile, "profile"),
        (args.run_id, "run id"),
        (args.attempt, "attempt"),
        (args.sha, "SHA"),
    ):
        if not TOKEN.fullmatch(value):
            raise AggregateError("invalid {}".format(label))
    if not re.fullmatch(r"[0-9a-f]{40}", args.sha):
        raise AggregateError("source SHA must contain 40 lowercase hex characters")
    expected = [item for item in args.expected_jobs.split(",") if item]
    if not expected or len(expected) != len(set(expected)):
        raise AggregateError("expected job list is empty or contains duplicates")
    if any(not TOKEN.fullmatch(item) for item in expected):
        raise AggregateError("expected job list contains an unsafe name")

    shared = Path(args.shared_root).resolve()
    if str(shared) != "/ci_public/lmcache-das" and os.environ.get("HCU_CI_AGGREGATE_SELFTEST") != "1":
        raise AggregateError("shared root must be /ci_public/lmcache-das")
    root = shared / args.profile / args.run_id / args.attempt / args.sha
    root.mkdir(parents=True, exist_ok=True)
    problems = []
    jobs = []
    for job_key in expected:
        job_root = root / job_key
        record = {"job_key": job_key, "status": "missing"}
        try:
            if not (job_root / "READY").is_file():
                raise AggregateError("missing completion marker")
            verify_checksums(job_root)
            manifest = read_json(job_root / "manifest.json")
            if manifest.get("source_sha") != args.sha:
                raise AggregateError("source SHA differs from aggregate")
            if manifest.get("profile") != args.profile:
                raise AggregateError("profile differs from aggregate")
            if manifest.get("job_key") != job_key:
                raise AggregateError("manifest job key differs from directory")
            record.update(
                status=manifest.get("status"),
                primary_exit_code=manifest.get("primary_exit_code"),
                cleanup_exit_code=manifest.get("cleanup_exit_code"),
            )
            if manifest.get("status") != "passed":
                raise AggregateError("job manifest reports failure")
        except Exception as exc:
            record["error"] = str(exc)
            problems.append("{}: {}".format(job_key, exc))
        jobs.append(record)

    status = "passed" if not problems else "failed"
    manifest = {
        "schema_version": 1,
        "repository": args.repository,
        "profile": args.profile,
        "run_id": args.run_id,
        "attempt": args.attempt,
        "source_sha": args.sha,
        "expected_jobs": expected,
        "jobs": jobs,
        "status": status,
    }
    summary = [
        "# LMCache-HCU split CI summary",
        "",
        "- Status: `{}`".format(status),
        "- Source SHA: `{}`".format(args.sha),
        "- Expected jobs: `{}`".format(len(expected)),
        "",
        "| Job | Status |",
        "|---|---|",
    ]
    summary.extend("| `{}` | `{}` |".format(item["job_key"], item["status"]) for item in jobs)
    if problems:
        summary.extend(["", "## Problems", ""])
        summary.extend("- {}".format(item) for item in problems)

    for stale in ("READY", "AGGREGATE_FAILED", "SHA256SUMS"):
        path = root / stale
        if path.exists():
            path.unlink()
    atomic_json(root / "aggregate-manifest.json", manifest)
    atomic_text(root / "summary.md", "\n".join(summary) + "\n")
    checksum_paths = [root / "aggregate-manifest.json", root / "summary.md"]
    for job_key in expected:
        for relative in ("READY", "SHA256SUMS", "manifest.json"):
            path = root / job_key / relative
            if path.is_file():
                checksum_paths.append(path)
    lines = []
    for path in sorted(checksum_paths):
        lines.append("{}  {}".format(sha256_file(path), path.relative_to(root).as_posix()))
    atomic_text(root / "SHA256SUMS", "\n".join(lines) + "\n")
    if problems:
        atomic_text(root / "AGGREGATE_FAILED", "\n".join(problems) + "\n")
        return 1
    atomic_text(root / "READY", "complete\n")
    return 0


def selftest():
    import tempfile

    with tempfile.TemporaryDirectory(prefix="lmcache-hcu-aggregate-") as temporary:
        os.environ["HCU_CI_AGGREGATE_SELFTEST"] = "1"
        shared = Path(temporary)
        root = shared / "pr" / "1" / "1" / ("a" * 40)
        for job_key in ("framework", "model"):
            job = root / job_key
            job.mkdir(parents=True)
            atomic_json(job / "manifest.json", {
                "profile": "pr", "source_sha": "a" * 40,
                "job_key": job_key, "status": "passed",
                "primary_exit_code": 0, "cleanup_exit_code": 0,
            })
            atomic_text(job / "SHA256SUMS", "{}  manifest.json\n".format(
                sha256_file(job / "manifest.json")
            ))
            atomic_text(job / "READY", "complete\n")
        args = argparse.Namespace(
            shared_root=str(shared), profile="pr", run_id="1", attempt="1",
            sha="a" * 40, expected_jobs="framework,model", repository="test/repo",
        )
        if aggregate(args) != 0 or not (root / "READY").is_file():
            raise AggregateError("split aggregate self-test failed")
    print("LMCache-HCU split aggregate self-tests passed")
    return 0


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--shared-root")
    parser.add_argument("--profile")
    parser.add_argument("--run-id")
    parser.add_argument("--attempt")
    parser.add_argument("--sha")
    parser.add_argument("--expected-jobs")
    parser.add_argument("--repository", default="HYGON-AI/LMCache-das")
    parser.add_argument("--selftest", action="store_true")
    args = parser.parse_args()
    if args.selftest:
        return selftest()
    required = (args.shared_root, args.profile, args.run_id, args.attempt, args.sha, args.expected_jobs)
    if not all(required):
        raise AggregateError("all aggregate arguments are required")
    return aggregate(args)


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except Exception as exc:
        print("ERROR: {}".format(exc), file=sys.stderr)
        raise SystemExit(1)

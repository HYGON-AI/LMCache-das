# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Run-scoped model scratch on Docker's /data disk (host Python >= 3.6)."""

import argparse
import json
import os
import posixpath
import re
import subprocess
import sys

LABEL = "lmcache-hcu-ci.run-key"
OWNER_LABEL = "lmcache-hcu-ci.owner"
OWNER = "HYGON-AI/LMCache-das"
MOUNTS = (("cpu", "/mnt/fs1"), ("localdisk", "/local_disk"),
          ("ssd", "/ssd"), ("posix", "/mnt/parastor_storage"))
# Observed Qwen long-document cache is ~226 GiB. Leave headroom without
# reducing the reviewed workload. This is a preflight floor, not a quota.
MIN_FREE_BYTES = 300 * 1024 ** 3


def require(condition, message):
    if not condition:
        raise RuntimeError(message)


def docker(*args, **kwargs):
    result = subprocess.run(["docker"] + list(args), stdout=subprocess.PIPE,
                            stderr=subprocess.PIPE, universal_newlines=True,
                            timeout=kwargs.get("timeout", 90))
    if result.returncode:
        if kwargs.get("missing") and re.search(
                r"no such (volume|container|object)\b", result.stderr, re.I):
            return None
        raise RuntimeError("docker {} failed: {}".format(args[0], result.stderr.strip()))
    return result.stdout.strip()


def names(run_key):
    require(re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,180}", run_key), "Invalid run key")
    return [("lmcache-hcu-cache-{}-{}".format(run_key, kind), path)
            for kind, path in MOUNTS]


def inspect(kind, name):
    output = docker(kind, "inspect", name, missing=True)
    return None if output is None else json.loads(output)[0]


def owned(item, run_key, container=False):
    labels = (item.get("Config", {}) if container else item).get("Labels") or {}
    require(labels.get(LABEL) == run_key and labels.get(OWNER_LABEL) == OWNER,
            "Refusing resource without this run's ownership labels")


def data_disk():
    root = docker("info", "--format", "{{.DockerRootDir}}")
    require(root.startswith("/data/") and posixpath.normpath(root) == root,
            "DockerRootDir must be on /data, got {!r}".format(root))
    require(not os.path.islink("/data") and os.path.isdir("/data"), "Invalid /data directory")
    device = os.stat("/data").st_dev
    require(device != os.stat("/").st_dev, "/data must be a separate disk from /")
    return root, device


def verify_volume(item, name, run_key, root):
    require(item is not None, "Missing cache volume " + name)
    owned(item, run_key)
    require(item.get("Name") == name and item.get("Driver") == "local"
            and not item.get("Options"), "Cache must be a plain Docker local volume")
    expected = root + "/volumes/" + name + "/_data"
    require(item.get("Mountpoint") == expected, "Unexpected cache backing path")


def verify(run_key):
    root, device = data_disk()
    for name, _ in names(run_key):
        verify_volume(inspect("volume", name), name, run_key, root)
    return root, device


def remove_probe(run_key):
    name = "lmcache-hcu-cache-probe-" + run_key
    item = inspect("container", name)
    if item is not None:
        owned(item, run_key, container=True)
        docker("container", "rm", "-f", name)


def cleanup(run_key):
    # Never scan/prune other jobs; even a missing daemon is an error, not
    # evidence that the resource was removed. In-use volumes are not forced.
    remove_probe(run_key)
    errors = []
    for name, _ in names(run_key):
        try:
            item = inspect("volume", name)
            if item is not None:
                owned(item, run_key)
                docker("volume", "rm", name)
                require(inspect("volume", name) is None, "Cache volume still exists")
        except (RuntimeError, subprocess.TimeoutExpired) as exc:
            errors.append(str(exc))
    require(not errors, "; ".join(errors))


def prepare(run_key, image):
    volumes = names(run_key)
    root, device = data_disk()
    # Reject collisions before creating anything. Never adopt an old cache.
    for name, _ in volumes:
        require(inspect("volume", name) is None, "Cache volume already exists: " + name)
    require(inspect("container", "lmcache-hcu-cache-probe-" + run_key) is None,
            "Cache probe already exists")
    for name, _ in volumes:
        docker("volume", "create", "--driver", "local", "--label", LABEL + "=" + run_key,
               "--label", OWNER_LABEL + "=" + OWNER, name)
        verify_volume(inspect("volume", name), name, run_key, root)

    # The SAME image identity, mounts and unprivileged Docker policy as the
    # model container. No GPU, network, host bind, sudo or driver bind options.
    probe = r'''
set -eu
python3 -c 'import os,sys,json
for path in sys.argv[3:]:
    stat = os.statvfs(path)
    available = stat.f_bavail * stat.f_frsize
    print(json.dumps(dict(path=path, available_bytes=available, device=os.stat(path).st_dev)), flush=True)
    if os.stat(path).st_dev != int(sys.argv[1]):
        raise SystemExit("Cache is not backed by the host /data filesystem")
    if available < int(sys.argv[2]):
        raise SystemExit("Cache requires at least 300 GiB free before model startup")
' "$@"
shift 2
for path in "$@"; do
    dd if=/dev/zero of="$path/.direct-io-probe" bs=4096 count=1 oflag=direct status=none
    rm -- "$path/.direct-io-probe"
done
'''
    args = ["run", "--rm", "--name", "lmcache-hcu-cache-probe-" + run_key,
            "--label", LABEL + "=" + run_key, "--label", OWNER_LABEL + "=" + OWNER,
            "--network", "none", "--read-only", "--memory", "512m", "--cpus", "1",
            "--cap-drop", "ALL", "--security-opt", "no-new-privileges", "--pids-limit", "64"]
    for name, path in volumes:
        args += ["-v", name + ":" + path + ":rw,nocopy"]
    args += ["--entrypoint", "bash", image, "-ceu", probe, "cache-probe",
             str(device), str(MIN_FREE_BYTES)] + [path for _, path in volumes]
    try:
        print(docker(*args, timeout=90))
    finally:
        remove_probe(run_key)
    print("Model cache: {} (minimum free: {} bytes)".format(root, MIN_FREE_BYTES))


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("command", choices=("prepare", "verify", "cleanup"))
    parser.add_argument("--run-key", required=True)
    parser.add_argument("--image")
    args = parser.parse_args()
    names(args.run_key)  # Validate before any Docker operation, including cleanup.
    if args.command == "prepare":
        require(args.image and "@sha256:" in args.image, "Pinned image digest required")
        prepare(args.run_key, args.image)
    elif args.command == "verify":
        verify(args.run_key)
    else:
        cleanup(args.run_key)


if __name__ == "__main__":
    try:
        main()
    except (RuntimeError, OSError, ValueError, subprocess.TimeoutExpired) as error:
        print("Model cache error: {}".format(error), file=sys.stderr)
        sys.exit(3)

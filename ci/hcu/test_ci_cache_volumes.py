# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""No Docker daemon/GPU required; also runs on host Python 3.6."""

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("cache_volumes", str(ROOT / "cache-volumes.py"))
cache = importlib.util.module_from_spec(spec)
spec.loader.exec_module(cache)
KEY = "123-1-aaaaaaaaaaaa-qwen-posix"
IMAGE = "example.invalid/test@sha256:" + "a" * 64


class CacheVolumeTests(unittest.TestCase):
    def setUp(self):
        self.volumes = {}
        self.containers = {}
        self.calls = []
        self.probe_error = None
        self.in_use = False
        self.addCleanup(patch.stopall)
        patch.object(cache, "docker", side_effect=self.docker).start()
        patch.object(cache, "data_disk", return_value=("/data/docker", 42)).start()

    def volume(self, name):
        return dict(Name=name, Driver="local", Options=None,
                    Mountpoint="/data/docker/volumes/" + name + "/_data",
                    Labels={cache.LABEL: KEY, cache.OWNER_LABEL: cache.OWNER})

    def docker(self, *args, **kwargs):
        self.calls.append(args)
        if args[0] in ("volume", "container") and args[1] == "inspect":
            item = (self.volumes if args[0] == "volume" else self.containers).get(args[2])
            return None if item is None else json.dumps([item])
        if args[:2] == ("volume", "create"):
            self.volumes[args[-1]] = self.volume(args[-1])
            return args[-1]
        if args[:2] == ("volume", "rm"):
            if self.in_use:
                raise RuntimeError("volume is in use")
            del self.volumes[args[-1]]
            return args[-1]
        if args[0] == "run":
            if self.probe_error:
                self.containers["lmcache-hcu-cache-probe-" + KEY] = {
                    "Config": {"Labels": {cache.LABEL: KEY, cache.OWNER_LABEL: cache.OWNER}}}
                raise self.probe_error
            return "capacity and O_DIRECT passed"
        if args[:2] == ("container", "rm"):
            del self.containers[args[-1]]
            return args[-1]
        raise AssertionError(args)

    def test_prepare_mounts_exact_four_large_disk_volumes_and_cleanup(self):
        cache.prepare(KEY, IMAGE)
        cache.verify(KEY)
        run = next(call for call in self.calls if call[0] == "run")
        for name, path in cache.names(KEY):
            self.assertIn(name + ":" + path + ":rw,nocopy", run)
        for forbidden in ("--device", "--privileged", "--mount", "--driver-opt"):
            self.assertNotIn(forbidden, run)
        self.assertIn(str(300 * 1024 ** 3), run)
        self.assertIn("none", run)
        cache.cleanup(KEY)
        cache.cleanup(KEY)  # absence is idempotent
        self.assertFalse(self.volumes)
        self.assertFalse(self.containers)
        self.assertFalse(any("prune" in call for call in self.calls))

    def test_collision_is_not_adopted_or_overwritten(self):
        name = cache.names(KEY)[-1][0]
        self.volumes[name] = self.volume(name)
        with self.assertRaisesRegex(RuntimeError, "already exists"):
            cache.prepare(KEY, IMAGE)
        self.assertFalse(any(call[:2] == ("volume", "create") for call in self.calls))

    def test_wrong_labels_driver_options_and_mountpoint_rejected(self):
        name = cache.names(KEY)[0][0]
        for change in ({"Labels": {}}, {"Driver": "nfs"},
                       {"Options": {"device": "/", "o": "bind"}},
                       {"Mountpoint": "/var/lib/docker/volumes/x/_data"}):
            item = self.volume(name)
            item.update(change)
            with self.assertRaises(RuntimeError):
                cache.verify_volume(item, name, KEY, "/data/docker")

    def test_cleanup_refuses_foreign_or_in_use_volume(self):
        name = cache.names(KEY)[0][0]
        self.volumes[name] = self.volume(name)
        self.volumes[name]["Labels"] = {cache.LABEL: "another-run"}
        with self.assertRaises(RuntimeError):
            cache.cleanup(KEY)
        self.assertIn(name, self.volumes)
        self.volumes[name] = self.volume(name)
        self.in_use = True
        with self.assertRaisesRegex(RuntimeError, "in use"):
            cache.cleanup(KEY)
        self.assertIn(name, self.volumes)
        self.assertNotIn(("volume", "rm", "-f", name), self.calls)

    def test_probe_failure_and_timeout_are_not_hidden_and_can_finalize(self):
        for error in (RuntimeError("low space"), subprocess.TimeoutExpired("docker", 90)):
            self.probe_error = error
            with self.assertRaises(type(error)):
                cache.prepare(KEY, IMAGE)
            self.assertFalse(self.containers)
            cache.cleanup(KEY)
            self.assertFalse(self.volumes)

    def test_foreign_probe_not_removed(self):
        self.containers["lmcache-hcu-cache-probe-" + KEY] = {"Config": {"Labels": {}}}
        with self.assertRaises(RuntimeError):
            cache.cleanup(KEY)
        self.assertTrue(self.containers)

    def test_invalid_run_key(self):
        for key in ("../other", "", "name:host", "x/y", "x" * 200):
            with self.assertRaises(RuntimeError):
                cache.names(key)
        self.assertFalse(self.calls)

    def test_shell_uses_same_volume_mounts_and_no_root_cache(self):
        shell = (ROOT / "run.sh").read_text()
        for kind, path in cache.MOUNTS:
            self.assertIn('${CACHE_VOLUME_PREFIX}-' + kind + ':' + path + ':rw,nocopy', shell)
        self.assertNotIn("CACHE_RUN_ROOT", shell)
        self.assertNotIn("chmod 0777", shell)

    @unittest.skipIf(os.name == "nt", "requires Linux bash and statvfs")
    def test_actual_probe_script_rejects_low_space_and_wrong_device(self):
        cache.prepare(KEY, IMAGE)
        run = next(call for call in self.calls if call[0] == "run")
        # Selftests may themselves run on tmpfs (no O_DIRECT). Exercise the
        # real capacity/device script here, with the I/O command stubbed; the
        # deployed helper still performs real dd on all four cache mounts.
        script = "dd() { return 0; }; " + run[run.index("-ceu") + 1]
        script = script.replace('rm -- "$path/.direct-io-probe"', ':')
        with tempfile.TemporaryDirectory() as directory:
            device = os.stat(directory).st_dev
            for dev, minimum, expected in ((device, 1, 0), (device, 10 ** 30, 1),
                                            (device + 1, 1, 1)):
                result = subprocess.run(["bash", "-ceu", script, "probe", str(dev),
                                         str(minimum), directory], stdout=subprocess.PIPE,
                                        stderr=subprocess.PIPE, universal_newlines=True)
                self.assertEqual(result.returncode, expected, result.stderr)
            self.assertEqual(os.listdir(directory), [])
            failed_io = script.replace("dd() { return 0; }", "dd() { return 1; }")
            result = subprocess.run(["bash", "-ceu", failed_io, "probe", str(device),
                                     "1", directory], stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertNotEqual(result.returncode, 0)


class DockerBoundaryTests(unittest.TestCase):
    def test_missing_is_distinct_from_daemon_error(self):
        for message, missing in (("Error: No such volume: x", True),
                                 ("Cannot connect to Docker daemon", False),
                                 ("permission denied", False)):
            with patch.object(cache.subprocess, "run", return_value=SimpleNamespace(
                    returncode=1, stdout="", stderr=message)):
                if missing:
                    self.assertIsNone(cache.inspect("volume", "x"))
                else:
                    with self.assertRaises(RuntimeError):
                        cache.inspect("volume", "x")

    def test_root_disk_and_non_data_docker_root_rejected(self):
        for root, data_dev, valid in (("/var/lib/docker", 42, False),
                                      ("/data/docker", 1, False),
                                      ("/data/../var/docker", 42, False),
                                      ("/data/docker", 42, True)):
            with patch.object(cache, "docker", return_value=root), \
                    patch.object(cache.os.path, "islink", return_value=False), \
                    patch.object(cache.os.path, "isdir", return_value=True), \
                    patch.object(cache.os, "stat", side_effect=lambda p: SimpleNamespace(
                        st_dev=data_dev if p == "/data" else 1)):
                if valid:
                    self.assertEqual(cache.data_disk(), (root, data_dev))
                else:
                    with self.assertRaises(RuntimeError):
                        cache.data_disk()


if __name__ == "__main__":
    unittest.main()

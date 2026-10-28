# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Exercise real finalize/publication cleanup on host Python 3.6 and newer."""

import argparse
import os
from pathlib import Path
import shutil
import tempfile
import unittest
from unittest.mock import patch

import test_ci_host_disabled as fixtures

host, ROOT = fixtures.host, fixtures.ROOT


class WheelLifecycleTests(fixtures.DisabledHostTests):
    def setUp(self):
        super().setUp()
        self.run_temp = self.root / "runner-temp"
        self.work = self.run_temp / "lmcache-hcu" / "123-1-aaaaaaaaaaaa-framework"
        self.work.mkdir(parents=True)
        self.spool = self.work / "trusted-spool"
        self.spool.mkdir()
        shutil.copyfile(self.manifest, self.spool / self.manifest.name)
        self.shared = self.root / "shared"
        self.shared.mkdir()
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"HCU_CI_HOST_SELFTEST": "1", "RUNNER_TEMP": str(self.run_temp)}).start()
        if os.name == "nt":
            # Windows has neither directory fsync nor POSIX group permissions.
            # Only these OS boundaries are adapted; publication, copy, checksum,
            # rename, READY and both cleanup owners execute their real code.
            patch.object(host, "fsync_directory").start()

            def prepare_parent(root, parts):
                parent = Path(root).joinpath(*parts)
                parent.mkdir(parents=True, exist_ok=True)
                return parent

            patch.object(host, "prepare_shared_parent", side_effect=prepare_parent).start()
        self.args = argparse.Namespace(
            spool=str(self.spool), baseline=str(self.trusted / "test-baseline.json"),
            model_manifest=str(ROOT / "model-test-manifest.json"), repeat=1,
            model_profile="framework", runner_kind="nmz1", job_role="framework",
            job_key="framework", primary_rc=0, cleanup_rc=0, repository="test/repo",
            profile="pr", run_id="123", attempt="1", sha="a" * 40,
            controller_sha="b" * 40, base_image="image@sha256:" + "c" * 64,
            base_image_id="sha256:" + "d" * 64, shared_root=str(self.shared),
            run_key=self.work.name, cleanup_root=str(self.work), archive_mode="off",
        )
        self.bundle = self.shared / "wheels/123/1" / ("a" * 40) / "framework"

    def prepare_wheel(self):
        self.prepare_success()
        (self.spool / "wheels").mkdir()
        (self.spool / "wheels/test.whl").write_bytes(b"current source wheel fixture")

    def assert_bundle_valid(self):
        self.assertTrue((self.bundle / "READY").is_file())
        receipt = host.read_json(self.bundle / "manifest.json")
        self.assertEqual(receipt["status"], "passed")
        self.assertEqual(receipt["source_sha"], self.args.sha)
        self.assertEqual(receipt["controller_sha"], self.args.controller_sha)
        self.assertEqual(receipt["wheel_sha256"], host.sha256_file(self.bundle / "wheels/test.whl"))
        for line in (self.bundle / "SHA256SUMS").read_text().splitlines():
            expected, relative = line.split("  ", 1)
            self.assertEqual(host.sha256_file(self.bundle / relative), expected)

    def test_real_finalize_off_succeeds_and_cleans_workspace(self):
        self.prepare_wheel()
        self.assertEqual(host.cmd_finalize(self.args), 0)
        self.assert_bundle_valid()
        self.assertFalse(self.work.exists())
        self.assertFalse((self.shared / "pr").exists())

    def test_real_finalize_required_retains_both_publications(self):
        self.prepare_wheel()
        self.args.archive_mode = "required"
        self.assertEqual(host.cmd_finalize(self.args), 0)
        self.assert_bundle_valid()
        report = self.shared / "pr/123/1" / self.args.sha / "framework"
        self.assertTrue((report / "READY").is_file())
        self.assertTrue((report / "reports/junit-repeat-1.xml").is_file())
        self.assertFalse(self.work.exists())

    def test_outer_temporary_directory_is_not_consumed_by_publisher(self):
        self.prepare_wheel()
        real_temporary = tempfile.TemporaryDirectory
        case = self

        class StrictTemporaryDirectory(real_temporary):
            def __exit__(self, *args):
                # Detect the ownership violation even on Python >=3.10, where
                # TemporaryDirectory silently tolerates an already deleted root.
                case.assertTrue(Path(self.name).is_dir(), "publisher removed context-owned root")
                return super().__exit__(*args)

        with patch.object(host.tempfile, "TemporaryDirectory", StrictTemporaryDirectory):
            self.assertEqual(host.cmd_finalize(self.args), 0)
        self.assert_bundle_valid()

    def test_real_copy_failure_still_fails_without_ready(self):
        self.prepare_wheel()
        with patch.object(host, "copy_trusted_tree", side_effect=OSError("injected copy failure")):
            self.assertEqual(host.cmd_finalize(self.args), 12)
        self.assertFalse((self.bundle / "READY").exists())
        self.assertFalse(self.work.exists())

    def test_bundle_cleanup_failure_is_not_hidden(self):
        self.prepare_wheel()
        real_rmtree = shutil.rmtree

        def deny_bundle(path, *args, **kwargs):
            if Path(path).name == "bundle":
                raise PermissionError("injected bundle cleanup failure")
            return real_rmtree(path, *args, **kwargs)

        with patch.object(host.shutil, "rmtree", side_effect=deny_bundle):
            self.assertEqual(host.cmd_finalize(self.args), 12)
        self.assertFalse((self.bundle / "READY").exists())
        self.assertFalse(self.work.exists())

    def test_host_regressions_are_outside_image_python_version_gate(self):
        script = (ROOT / "selftest.sh").read_text()
        host_section = script.split("if python3 -c 'import sys;", 1)[0]
        for name in ("test_ci_host_disabled.py", "test_ci_storage.py", "test_ci_wheel_lifecycle.py"):
            self.assertIn('python3 "${SCRIPT_DIR}/' + name + '"', host_section)


if __name__ == "__main__":
    unittest.main(verbosity=2)

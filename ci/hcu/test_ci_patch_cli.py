# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Repository CLI execution tests, independent of the suspended strict gate."""

import argparse
import contextlib
import importlib.util
import io
from pathlib import Path
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parent


def load(name, filename):
    spec = importlib.util.spec_from_file_location(name, ROOT / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


ci = load("lmcache_patch_cli_ci", "ci.py")
host = load("lmcache_patch_cli_host", "host.py")


class RepositoryPatchCLITests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lmcache-patch-cli-test-")
        self.addCleanup(temporary.cleanup)
        self.report = Path(temporary.name) / "patch-report.json"

    def apply(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return ci.cmd_patch_apply(argparse.Namespace(output=str(self.report)))

    def uninstall(self):
        with contextlib.redirect_stdout(io.StringIO()):
            return ci.cmd_patch_uninstall(argparse.Namespace(report=str(self.report)))

    def test_repository_install_and_uninstall_without_strict_checks(self):
        with mock.patch.object(ci, "invoke_patch", return_value="CLI completed\n") as command, \
                mock.patch.object(ci, "resolve_patch_targets", side_effect=AssertionError("strict probe called")), \
                mock.patch.object(ci, "cmd_patch_gate", side_effect=AssertionError("strict gate called")):
            self.assertEqual(self.apply(), 0)
            report = ci.read_json(self.report)
            self.assertEqual(report["status"], "applied")
            self.assertEqual(report["strict_validation"], "disabled")
            self.assertNotIn("targets", report)
            self.assertEqual(self.uninstall(), 0)
            self.assertEqual(command.call_args_list, [mock.call("install"), mock.call("uninstall")])
        self.assertEqual(host.validate_patch_execution(ci.read_json(self.report)), [])

    def test_install_failure_is_preserved_and_cleanup_still_runs(self):
        with mock.patch.object(ci, "invoke_patch", side_effect=[ci.CIError("install exit 9"), "restored"]):
            with self.assertRaisesRegex(ci.CIError, "install exit 9"):
                self.apply()
            self.assertEqual(ci.read_json(self.report)["status"], "failed")
            self.assertEqual(self.uninstall(), 0)
        report = ci.read_json(self.report)
        self.assertEqual(report["status"], "failed")
        self.assertIn("install exit 9", report["error"])
        self.assertTrue(host.validate_patch_execution(report))

    def test_uninstall_failure_is_not_hidden(self):
        with mock.patch.object(ci, "invoke_patch", side_effect=["installed", ci.CIError("uninstall exit 7")]):
            self.apply()
            with self.assertRaisesRegex(ci.CIError, "uninstall exit 7"):
                self.uninstall()
        report = ci.read_json(self.report)
        self.assertEqual(report["cleanup_status"], "failed")
        self.assertTrue(host.validate_patch_execution(report))

    def test_host_rejects_missing_or_old_strict_receipts(self):
        for report in ({}, {"status": "passed"},
                       {"mode": "repository-cli", "strict_validation": "disabled",
                        "install_attempted": True, "status": "applied"}):
            self.assertTrue(host.validate_patch_execution(report))

    def test_workflows_use_cli_not_strict_gate(self):
        script = (ROOT / "run-tests.sh").read_text(encoding="utf-8")
        self.assertIn('"${CI_HELPER}" patch-apply', script)
        self.assertIn('"${CI_HELPER}" patch-uninstall', script)
        self.assertNotIn('"${CI_HELPER}" patch-gate', script)
        self.assertNotIn('"${CI_HELPER}" patch-cleanup', script)
        self.assertNotIn("--patch-manifest", (ROOT / "run.sh").read_text(encoding="utf-8"))

    def test_preflight_does_not_require_a_patch_manifest(self):
        image = ci.read_json(ROOT / "compatibility.json")["image"]
        with contextlib.redirect_stdout(io.StringIO()):
            rc = ci.cmd_validate(argparse.Namespace(
                compatibility=str(ROOT / "compatibility.json"),
                base_image=image["reference"], base_image_id=image["id"],
                profile="pr", repeat=1, checkout=str(ROOT), upstream=str(ROOT),
            ))
        self.assertEqual(rc, 0)


if __name__ == "__main__":
    unittest.main(verbosity=2)

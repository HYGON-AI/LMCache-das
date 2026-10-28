# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""PR scratch mounts and result gates without mandatory log archiving."""

import argparse
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

import test_ci_host_disabled as fixtures

host, ROOT = fixtures.host, fixtures.ROOT


@unittest.skipIf(os.name == "nt", "requires Linux permissions, fsync and bash")
class StorageTests(fixtures.DisabledHostTests):
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

    def test_existing_foreign_owned_parent_is_not_chmodded(self):
        parent = self.shared / "pr"
        parent.mkdir()
        chmod = host.os.chmod

        def deny_existing(path, mode):
            if Path(path) == parent:
                raise PermissionError("foreign UID")
            return chmod(path, mode)

        with patch.object(host.os, "chmod", side_effect=deny_existing):
            self.assertTrue(host.prepare_shared_parent(self.shared, ("pr", "123")).is_dir())

    def test_pr_results_do_not_require_report_publication(self):
        self.prepare_success()
        with patch.object(host, "publish_wheel_bundle") as wheel, \
                patch.object(host, "publish", side_effect=PermissionError("report store denied")) as archive:
            self.assertEqual(host.cmd_finalize(self.args), 0)
        wheel.assert_called_once()
        archive.assert_not_called()
        self.assertFalse(self.work.exists())
        self.assertEqual(list(self.shared.iterdir()), [])

    def test_no_archive_does_not_hide_missing_junit(self):
        self.prepare_success()
        (self.spool / "reports/junit-repeat-1.xml").unlink()
        with patch.object(host, "publish_wheel_bundle") as wheel:
            self.assertEqual(host.cmd_finalize(self.args), 9)
        wheel.assert_not_called()

    def test_no_archive_does_not_hide_execution_failure(self):
        self.prepare_success()
        self.args.primary_rc = 8
        with patch.object(host, "publish_wheel_bundle") as wheel:
            self.assertEqual(host.cmd_finalize(self.args), 8)
        wheel.assert_not_called()

    def test_no_archive_still_requires_workspace_cleanup(self):
        self.prepare_success()
        with patch.object(host, "publish_wheel_bundle"), \
                patch.object(host, "cleanup_disposable_workspace", side_effect=OSError("denied")):
            self.assertEqual(host.cmd_finalize(self.args), 10)

    def test_required_wheel_transfer_failure_is_not_ignored(self):
        self.prepare_success()
        with patch.object(host, "publish_wheel_bundle", side_effect=OSError("wheel store denied")):
            self.assertEqual(host.cmd_finalize(self.args), 12)

    def test_wheel_bundle_contains_no_test_logs(self):
        self.prepare_success()
        (self.spool / "wheels").mkdir()
        (self.spool / "wheels/test.whl").write_bytes(b"current source wheel fixture")
        self.assertEqual(host.cmd_finalize(self.args), 0)
        bundle = self.shared / "wheels/123/1" / ("a" * 40) / "framework"
        self.assertEqual(sorted(p.relative_to(bundle).as_posix() for p in bundle.rglob("*") if p.is_file()),
                         ["READY", "SHA256SUMS", "manifest.json", "wheels/test.whl"])
        receipt = host.read_json(bundle / "manifest.json")
        self.assertEqual(receipt["source_sha"], "a" * 40)
        self.assertEqual(receipt["wheel_sha256"], host.sha256_file(bundle / "wheels/test.whl"))
        subprocess.run(["sha256sum", "-c", "SHA256SUMS"], cwd=str(bundle), check=True,
                       stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        shell = (ROOT / "run.sh").read_text().rsplit("\nmain", 1)[0]
        script = shell + r'''
SOURCE_SHA=aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa
CONTROLLER_SHA="$BUNDLE_CONTROLLER"
TRUSTED_STATE_ROOT="$BUNDLE_STAGE"
PREBUILT_WHEEL_ROOT="$HCU_CI_SHARED_ROOT/wheels/123/1/$SOURCE_SHA/framework/wheels"
verify_prebuilt_wheel
'''
        for mode in ("valid", "wrong-controller", "tampered"):
            if mode == "tampered":
                (bundle / "wheels/test.whl").write_bytes(b"corrupted wheel")
            env = dict(os.environ, HCU_CI_CONTROLLER=str(ROOT.parents[1]),
                       HCU_CI_CHECKOUT=str(ROOT.parents[1]), HCU_CI_JOB_ROLE="model",
                       HCU_CI_SHARED_ROOT=str(self.shared), HCU_CI_RUN_ID="123", HCU_CI_ATTEMPT="1",
                       BUNDLE_STAGE=str(self.root / mode),
                       BUNDLE_CONTROLLER=("c" if mode == "wrong-controller" else "b") * 40)
            result = subprocess.run(["bash"], input=script, universal_newlines=True, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 0 if mode == "valid" else 3, result.stderr)

    def test_cache_is_runner_owned_and_ignores_stale_shared_cache_variable(self):
        shell = (ROOT / "run.sh").read_text().rsplit("\nmain", 1)[0]
        script = shell + r'''
CACHE_RUN_ROOT="${CACHE_ROOT}/unique-test-run"
dd() { return 0; }
validate_cache_root
[[ "$CACHE_RUN_ROOT" == "$RUNNER_TEMP/lmcache-hcu-cache/unique-test-run" ]]
[[ -d "$CACHE_RUN_ROOT/posix" && -d "$CACHE_RUN_ROOT/localdisk" ]]
cleanup_cache_run
[[ ! -e "$CACHE_RUN_ROOT" ]]
'''
        env = dict(os.environ, HCU_CI_CONTROLLER=str(ROOT.parents[1]),
                   HCU_CI_CHECKOUT=str(ROOT.parents[1]), HCU_CI_JOB_ROLE="model",
                   HCU_CI_CACHE_ROOT="/ci_public/denied/not-used")
        result = subprocess.run(["bash"], input=script, universal_newlines=True, env=env,
                                stdout=subprocess.PIPE, stderr=subprocess.PIPE)
        self.assertEqual(result.returncode, 0, result.stderr)

    def test_aggregate_requires_success_and_not_archive_files(self):
        workflow = (ROOT.parents[1] / ".github/workflows/ci-hcu-pr.yml").read_text()
        self.assertNotIn("storage-preflight", workflow)
        self.assertNotIn("aggregate-jobs.py", workflow)
        script = workflow.rsplit("        run: |\n", 1)[1]
        script = "\n".join(line[10:] for line in script.splitlines())
        for framework, models in (("success", "success"), ("failure", "success"),
                                  ("success", "failure"), ("success", "skipped"),
                                  ("cancelled", "success")):
            env = dict(os.environ, FRAMEWORK_RESULT=framework, MODEL_RESULT=models,
                       SOURCE_SHA="a" * 40, GITHUB_STEP_SUMMARY=str(self.root / "summary"))
            result = subprocess.run(["bash", "-euo", "pipefail"], input=script,
                                    universal_newlines=True, env=env,
                                    stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode == 0, framework == models == "success", result.stderr)
        reusable = (ROOT.parents[1] / ".github/workflows/ci-hcu-reusable.yml").read_text()
        condition = reusable.split("  hcu:", 1)[1].split("    runs-on:", 1)[0]
        self.assertIn("!cancelled()", condition)
        self.assertNotIn("always()", condition)


if __name__ == "__main__":
    unittest.main()

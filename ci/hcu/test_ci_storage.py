# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Fault injection for cross-node publication, durable evidence and mount probes."""

import argparse
import copy
import os
from pathlib import Path
import shutil
import subprocess
import unittest
from unittest.mock import patch

import test_ci_host_disabled as fixtures

host, ROOT = fixtures.host, fixtures.ROOT


@unittest.skipIf(os.name == "nt", "storage permissions/fsync require the Linux CI host")
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
        self.durable = self.root / "durable"
        self.addCleanup(patch.stopall)
        patch.dict(os.environ, {"HCU_CI_HOST_SELFTEST": "1", "RUNNER_TEMP": str(self.run_temp)}).start()
        patch.object(host, "evidence_root", return_value=self.durable).start()
        self.args = argparse.Namespace(
            spool=str(self.spool), baseline=str(self.trusted / "test-baseline.json"),
            model_manifest=str(ROOT / "model-test-manifest.json"), repeat=1,
            model_profile="framework", runner_kind="nmz1", job_role="framework",
            job_key="framework", primary_rc=0, cleanup_rc=0, repository="test/repo",
            profile="pr", run_id="123", attempt="1", sha="a" * 40,
            controller_sha="b" * 40, base_image="image@sha256:" + "c" * 64,
            base_image_id="sha256:" + "d" * 64, shared_root=str(self.shared),
            run_key=self.work.name, cleanup_root=str(self.work),
        )
        self.evidence = self.durable / self.work.name
        self.final = self.shared / "pr" / "123" / "1" / ("a" * 40) / "framework"

    def fail_publication(self, primary=0):
        self.prepare_success()
        self.args.primary_rc = primary
        with patch.object(host, "publish", side_effect=PermissionError("NFS chmod denied")):
            self.assertEqual(host.cmd_finalize(self.args), primary or 11)
        self.assertFalse(self.work.exists())
        self.assertTrue((self.evidence / "receipt.json").is_file())

    def recovery(self):
        return argparse.Namespace(
            evidence=str(self.evidence), baseline=self.args.baseline,
            model_manifest=self.args.model_manifest, sha=self.args.sha,
            controller_sha=self.args.controller_sha,
        )

    def republish(self, args=None):
        with patch.object(host.subprocess, "check_output", return_value="b" * 40 + "\n"):
            return host.cmd_republish(args or self.recovery())

    def test_existing_foreign_owned_group_directory_is_not_chmodded(self):
        parent = self.shared / "pr"
        parent.mkdir()
        chmod = host.os.chmod

        def deny_parent(path, mode):
            if Path(path) == parent:
                raise PermissionError("simulated foreign UID")
            return chmod(path, mode)

        with patch.object(host.os, "chmod", side_effect=deny_parent):
            target = host.prepare_shared_parent(self.shared, ("pr", "123"))
            self.assertTrue(target.is_dir())

    def test_symlink_or_unwritable_shared_parent_is_rejected(self):
        parent = self.shared / "pr"
        try:
            parent.symlink_to(self.trusted, target_is_directory=True)
        except (OSError, NotImplementedError):
            self.skipTest("symlinks unavailable")
        with self.assertRaisesRegex(host.HostError, "Unsafe"):
            host.prepare_shared_parent(self.shared, ("pr",))
        parent.unlink()
        parent.mkdir()
        with patch.object(host.os, "access", return_value=False):
            with self.assertRaisesRegex(host.HostError, "not readable"):
                host.prepare_shared_parent(self.shared, ("pr",))

    def test_publication_preflight_leaves_no_probe_or_job_archive(self):
        host.publication_preflight(self.shared, "pr", "123", "1", "a" * 40, "probe")
        parent = self.final.parent
        self.assertEqual(list(parent.iterdir()), [])

    def test_publication_failure_retains_verified_evidence_outside_runner_temp(self):
        self.fail_publication()
        manifest = host.read_json(self.evidence / "spool/manifest.json")
        self.assertEqual(manifest["execution_exit_code"], 0)
        self.assertEqual(manifest["validation_exit_code"], 0)
        self.assertEqual(manifest["publish_exit_code"], 11)
        self.assertEqual(manifest["status"], "failed")
        receipt = host.read_json(self.evidence / "receipt.json")
        self.assertEqual(receipt["files"], host.evidence_hashes(self.evidence / "spool"))
        self.assertFalse(self.final.exists())

    def test_recovery_revalidates_and_is_idempotent_without_build_or_gpu(self):
        self.fail_publication()
        self.assertEqual(self.republish(), 0)
        self.assertTrue((self.final / "READY").is_file())
        host.verify_checksums(self.final)
        self.assertEqual(host.read_json(self.final / "manifest.json")["status"], "passed")
        before = host.evidence_hashes(self.final)
        self.assertEqual(self.republish(), 0)
        self.assertEqual(before, host.evidence_hashes(self.final))

    def test_recovery_rejects_wrong_sha_controller_and_manifest(self):
        self.fail_publication()
        for field in ("sha", "controller_sha", "baseline"):
            args = copy.copy(self.recovery())
            setattr(args, field, str(ROOT / "test-baseline.json") if field == "baseline" else "c" * 40)
            with self.subTest(field=field), self.assertRaises(host.HostError):
                self.republish(args)
        with patch.object(host.subprocess, "check_output", return_value="c" * 40):
            with self.assertRaisesRegex(host.HostError, "original trusted controller"):
                host.cmd_republish(self.recovery())

    def test_recovery_rejects_tampering_and_missing_junit_even_with_refreshed_receipt(self):
        self.fail_publication()
        junit = self.evidence / "spool/reports/junit-repeat-1.xml"
        junit.unlink()
        with self.assertRaisesRegex(host.HostError, "hashes differ"):
            self.republish()
        host.refresh_receipt(self.evidence)
        with self.assertRaisesRegex(host.HostError, "Recovery validation failed"):
            self.republish()

    def test_original_test_failure_cannot_become_green_on_republish(self):
        self.fail_publication(primary=8)
        self.assertEqual(self.republish(), 8)
        self.assertTrue((self.final / "READY").is_file())
        manifest = host.read_json(self.final / "manifest.json")
        self.assertEqual(manifest["primary_exit_code"], 8)
        self.assertEqual(manifest["status"], "failed")

    def test_cleanup_failure_is_archived_and_not_hidden(self):
        self.prepare_success()
        with patch.object(host, "cleanup_work_root", side_effect=OSError("cleanup denied")):
            self.assertEqual(host.cmd_finalize(self.args), 10)
        manifest = host.read_json(self.final / "manifest.json")
        self.assertEqual(manifest["cleanup_exit_code"], 10)
        self.assertEqual(manifest["status"], "failed")
        self.assertTrue((self.evidence / "spool/reports/synthetic-workspace-cleanup.xml").is_file())
        self.assertEqual(self.republish(), 10)

    def test_existing_different_archive_is_never_overwritten(self):
        self.fail_publication()
        self.assertEqual(self.republish(), 0)
        (self.final / "summary.md").write_text("changed", encoding="utf-8")
        with self.assertRaisesRegex(host.HostError, "differs"):
            self.republish()
        self.assertEqual((self.final / "summary.md").read_text(), "changed")

    @unittest.skipIf(os.name == "nt", "requires bash")
    def test_mount_probe_is_device_less_and_denials_fail_closed(self):
        shell = (ROOT / "run.sh").read_text(encoding="utf-8").rsplit("\nmain", 1)[0]
        for child in ("cpu", "localdisk", "ssd", "posix"):
            (self.root / "cache" / child).mkdir(parents=True, exist_ok=True)
        for rc in (0, 42):
            trace = self.root / "probe-{}.log".format(rc)
            script = shell + r'''
RUN_KEY=test-probe
CACHE_RUN_ROOT="$PROBE_CACHE"
STORAGE_PROBE_CONTAINER=lmcache-hcu-storage-probe-test
python3() { return 0; }
model_tests_required() { return 0; }
model_mount_arguments() { return 0; }
timeout() { shift 3; "$@"; }
docker() { printf '%s\n' "$@" >> "$PROBE_TRACE"; return "$PROBE_RC"; }
remove_named_container() { printf 'cleanup:%s\n' "$1" >> "$PROBE_TRACE"; return 0; }
storage_checks
'''
            env = dict(os.environ, HCU_CI_CONTROLLER=str(ROOT.parents[1]),
                       HCU_CI_CHECKOUT=str(ROOT.parents[1]), PROBE_CACHE=str(self.root / "cache"),
                       PROBE_TRACE=str(trace), PROBE_RC=str(rc))
            result = subprocess.run(["bash"], input=script, universal_newlines=True,
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, 0 if rc == 0 else 3, result.stderr)
            args = trace.read_text()
            self.assertNotIn("--device", args)
            self.assertNotIn("--privileged", args)
            self.assertIn("--network\nnone", args)
            self.assertIn("/mnt/parastor_storage:rw", args)
            self.assertIn("cleanup:lmcache-hcu-storage-probe-test", args)

    def test_pr_dag_and_reusable_keep_preflight_separate_from_execution(self):
        workflows = ROOT.parents[1] / ".github/workflows"
        pr = (workflows / "ci-hcu-pr.yml").read_text()
        reusable = (workflows / "ci-hcu-reusable.yml").read_text()
        self.assertIn("needs: storage-preflight", pr)
        self.assertIn("storage_preflight_only: true", pr)
        self.assertIn("!inputs.storage_preflight_only", reusable)
        self.assertIn("no wheel/model work was started", pr)

    def test_probe_failure_and_cancellation_cleanup_only_new_cache(self):
        shell = (ROOT / "run.sh").read_text(encoding="utf-8").rsplit("\nmain", 1)[0]
        for mode, expected in (("failed", 3), ("cancelled", 143), ("existing", 3)):
            cache = self.root / mode
            if mode == "existing":
                cache.mkdir()
                (cache / "keep").write_text("not this probe's resource")
            script = shell + r'''
WORK_ROOT="$PROBE_CACHE-work"
CACHE_RUN_ROOT="$PROBE_CACHE"
check_pr_revision() { return 0; }
verify_image() { return 0; }
python3() { return 0; }
remove_named_container() { return 0; }
validate_cache_root() {
    [[ "$PROBE_MODE" != existing ]] || return 3
    mkdir "$CACHE_RUN_ROOT"
    CACHE_RUN_CREATED=true
}
storage_checks() {
    if [[ "$PROBE_MODE" == cancelled ]]; then kill -TERM "$BASHPID"; fi
    return 3
}
cleanup_cache_run() { rm -r -- "$CACHE_RUN_ROOT"; }
phase_storage_preflight
'''
            env = dict(os.environ, HCU_CI_CONTROLLER=str(ROOT.parents[1]),
                       HCU_CI_CHECKOUT=str(ROOT.parents[1]), HCU_CI_ISOLATED_RUNNER="true",
                       PROBE_CACHE=str(cache), PROBE_MODE=mode)
            result = subprocess.run(["bash"], input=script, universal_newlines=True,
                                    env=env, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
            self.assertEqual(result.returncode, expected, result.stderr)
            self.assertEqual(cache.exists(), mode == "existing")


if __name__ == "__main__":
    unittest.main()

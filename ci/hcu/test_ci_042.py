# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Device-free regression tests for the 0.4.2 CI adapter (not model tests)."""

import argparse
import contextlib
import importlib.util
import io
import json
from pathlib import Path
import subprocess
import tempfile
import unittest
from unittest import mock


ROOT = Path(__file__).resolve().parent
spec = importlib.util.spec_from_file_location("lmcache_ci_042", ROOT / "ci.py")
ci = importlib.util.module_from_spec(spec)
spec.loader.exec_module(ci)


class PatchGateTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lmcache-ci-042-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.upstream = self.root / "upstream"
        self.manifest = ci.read_json(ROOT / "patch-manifest.json")
        self.paths = {}
        self.original = {}
        for item in self.manifest["targets"]:
            path = self.upstream / item["relative_path"]
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text((item.get("anchor", "# pristine") + "\n"), encoding="utf-8")
            item["preimage_sha256"] = ci.sha256_file(path)
            self.paths[item["module"]] = path.resolve()
            self.original[path] = path.read_bytes()
        vllm = self.root / "vllm/__init__.py"
        vllm.parent.mkdir()
        vllm.write_text("# unchanged vllm\n", encoding="utf-8")
        self.paths["vllm"] = vllm
        self.manifest_path = self.root / "manifest.json"
        ci.write_json(self.manifest_path, self.manifest)
        self.report = self.root / "report.json"

    def fake_patch(self, mode):
        for item in self.manifest["targets"]:
            path = self.paths[item["module"]]
            backup = path.with_name(path.name + ".bak.1")
            if mode == "uninstall":
                path.write_bytes(self.original[path])
            elif item["marker"] not in path.read_text(encoding="utf-8"):
                if not backup.exists():
                    backup.write_bytes(self.original[path])
                block = (item["begin_marker"] + "\n" + item["marker"] + "\n")
                text = block * item.get("marker_count", 1)
                text += item.get("required_text", "# patch body") + "\n"
                path.write_text(text + self.original[path].decode(), encoding="utf-8")
        return "mock patch " + mode

    def run_gate(self, resolved=None, apply=None):
        if resolved is None:
            resolved = {key: str(value) for key, value in self.paths.items() if key != "vllm"}
        with contextlib.ExitStack() as stack:
            stack.enter_context(mock.patch.object(ci.importlib.metadata, "version", return_value="0.4.2"))
            stack.enter_context(mock.patch.object(ci, "module_file", side_effect=self.paths.__getitem__))
            stack.enter_context(mock.patch.object(ci, "resolve_patch_targets", return_value=resolved))
            stack.enter_context(mock.patch.object(ci, "invoke_patch", side_effect=apply or self.fake_patch))
            stack.enter_context(mock.patch.object(ci, "run_checked", return_value=subprocess.CompletedProcess([], 0, "")))
            stack.enter_context(contextlib.redirect_stdout(io.StringIO()))
            return ci.cmd_patch_gate(argparse.Namespace(
                manifest=str(self.manifest_path), upstream=str(self.upstream), output=str(self.report),
            ))

    def test_three_targets_two_utils_markers_and_cleanup(self):
        self.assertEqual(self.run_gate(), 0)
        report = ci.read_json(self.report)
        self.assertEqual(report["status"], "passed")
        self.assertEqual(len(report["changed_files"]), 3)
        utils = self.paths["lmcache.utils"]
        self.assertEqual(utils.read_text().count("# END LMCACHE_HCU_DISK_METADATA"), 2)
        with mock.patch.object(ci, "invoke_patch", side_effect=self.fake_patch):
            ci.cmd_patch_cleanup(argparse.Namespace(report=str(self.report)))
        for path, original in self.original.items():
            self.assertEqual(path.read_bytes(), original)
        self.assertFalse(ci.backup_snapshot(self.upstream))

    def test_self_patch_is_rejected_before_install(self):
        resolved = {key: str(value) for key, value in self.paths.items() if key != "vllm"}
        resolved["lmcache.integration.vllm.lmcache_connector_v1"] = str(self.root / "lmcache_hcu/connector.py")
        install = mock.Mock()
        with self.assertRaisesRegex(ci.CIError, "module exchange"):
            self.run_gate(resolved=resolved, apply=install)
        install.assert_not_called()
        self.assertEqual(ci.read_json(self.report)["phase"], "target-resolution")
        for path, original in self.original.items():
            self.assertEqual(path.read_bytes(), original)

    def test_missing_utils_marker_fails_and_rolls_back(self):
        def broken(mode):
            self.fake_patch(mode)
            path = self.paths["lmcache.utils"]
            path.write_text(path.read_text().replace("# END LMCACHE_HCU_DISK_METADATA", "", 1))
            return "injected incomplete patch"
        with self.assertRaisesRegex(ci.CIError, "markers"):
            self.run_gate(apply=broken)
        self.assertEqual(ci.read_json(self.report)["status"], "failed")
        for path, original in self.original.items():
            self.assertEqual(path.read_bytes(), original)

    def test_unexpected_file_change_fails(self):
        def broken(mode):
            self.fake_patch(mode)
            (self.upstream / "unexpected.py").write_text("# injected\n")
            return "injected unexpected file"
        with self.assertRaisesRegex(ci.CIError, "Unexpected source patch changes"):
            self.run_gate(apply=broken)

    def test_second_install_must_be_idempotent(self):
        calls = []
        def broken(mode):
            calls.append(mode)
            self.fake_patch(mode)
            if len(calls) == 2:
                path = self.paths["lmcache.utils"]
                path.write_text(path.read_text() + "# non-idempotent\n")
            return "injected non-idempotent patch"
        with self.assertRaisesRegex(ci.CIError, "not idempotent"):
            self.run_gate(apply=broken)


if __name__ == "__main__":
    unittest.main(verbosity=2)

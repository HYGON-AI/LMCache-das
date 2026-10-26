# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Host import/publication regressions for explicitly disabled test nodes."""

import ast
import copy
import importlib.util
from pathlib import Path
import shutil
import tempfile
import unittest
import xml.etree.ElementTree as ET


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("lmcache_host_disabled", ROOT / "host.py")
host = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(host)


class DisabledHostTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(prefix="lmcache-host-disabled-test-")
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.trusted = self.root / "controller"
        self.trusted.mkdir()
        self.spool = self.root / "output"
        self.spool.mkdir()
        self.manifest = ROOT / "disabled-tests.json"
        shutil.copyfile(self.manifest, self.trusted / self.manifest.name)
        shutil.copyfile(self.manifest, self.spool / self.manifest.name)
        self.disabled = host.read_json(self.manifest)["tests"]
        self.active = ["lmcache_hcu/test_contract.py::test_active"]
        self.collected = self.active + [item["nodeid"] for item in self.disabled]
        self.inventory = {
            "nodeids": self.active, "count": 1,
            "collected_nodeids": self.collected, "collected_count": len(self.collected),
            "disabled_count": len(self.disabled), "disabled_tests": self.disabled,
            "disabled_manifest_sha256": host.sha256_file(self.manifest),
        }

    def validate_inventory(self, inventory=None):
        return host.validate_test_inventory(
            self.inventory if inventory is None else inventory,
            self.trusted / self.manifest.name, self.spool / self.manifest.name,
        )

    def test_exact_disabled_manifest_is_importable_but_arbitrary_files_are_not(self):
        destination = self.root / "imported"
        imported = host.safe_import(self.spool, destination)
        self.assertEqual([item["path"] for item in imported], ["disabled-tests.json"])
        self.assertEqual(host.read_json(destination / self.manifest.name), host.read_json(self.manifest))
        self.assertFalse(host.allowed_output(Path("disabled-tests-extra.json"), 2))
        self.assertFalse(host.allowed_output(Path("disabled-tests.json"), host.MAX_METADATA_BYTES + 1))
        (self.spool / "unexpected.json").write_text("{}", encoding="utf-8")
        with self.assertRaisesRegex(host.HostError, "Unexpected"):
            host.safe_import(self.spool, self.root / "rejected")

    def test_complete_collection_and_active_subset_are_distinct(self):
        self.assertEqual(self.validate_inventory(), (self.active, self.collected))

    def test_unapproved_exclusion_duplicate_or_wrong_count_is_rejected(self):
        mutations = [
            {"nodeids": [], "count": 0},
            {"nodeids": self.active + self.active, "count": 2},
            {"collected_nodeids": self.collected + self.active,
             "collected_count": len(self.collected) + 1},
            {"collected_count": 99}, {"disabled_count": 99},
            {"disabled_tests": []}, {"disabled_manifest_sha256": "0" * 64},
            {"collected_nodeids": self.active, "collected_count": 1},
        ]
        for update in mutations:
            inventory = copy.deepcopy(self.inventory)
            inventory.update(update)
            with self.subTest(update=update), self.assertRaises(host.HostError):
                self.validate_inventory(inventory)

    def test_tampered_archived_manifest_is_rejected(self):
        manifest = host.read_json(self.manifest)
        manifest["tests"] = []
        host.atomic_json(self.spool / self.manifest.name, manifest)
        with self.assertRaisesRegex(host.HostError, "differs"):
            self.validate_inventory()

    def prepare_success(self):
        baseline = {
            "minimum_collected": len(self.collected),
            "required_test_files": sorted({host.inventory_file(node) for node in self.collected}),
            "required_test_definitions": {},
        }
        for node in self.collected:
            path, definition = host.inventory_definition(node)
            baseline["required_test_definitions"].setdefault(path, []).append(definition)
        host.atomic_json(self.trusted / "test-baseline.json", baseline)
        data = {
            "environment.json": {}, "upstream-source.json": {}, "wheel-report.json": {},
            "installed-package.json": {}, "test-inventory.json": self.inventory,
            "patch-report.json": {"mode": "repository-cli", "strict_validation": "disabled",
                                  "install_attempted": True, "status": "applied",
                                  "cleanup_status": "uninstalled"},
            "test-summary.json": {"status": "passed", "expected_per_repeat": 1,
                                  "repeat_count": 1, "expected_total": 1,
                                  "collected_per_repeat": len(self.collected),
                                  "disabled_per_repeat": len(self.disabled),
                                  "disabled_tests": self.disabled},
            "model-inventory.json": {"count": 0, "nodeids": []},
            "model-summary.json": {"status": "passed", "expected": 0,
                                   "profile": "framework", "runner": "nmz1"},
            "asset-manifest.json": {"profile": "framework", "runner": "nmz1"},
        }
        for name, value in data.items():
            host.atomic_json(self.spool / name, value)
        reports = self.spool / "reports"
        reports.mkdir()
        suite = ET.Element("testsuite")
        ET.SubElement(suite, "testcase", name="test_active")
        ET.ElementTree(suite).write(str(reports / "junit-repeat-1.xml"))
        ET.ElementTree(ET.Element("testsuite")).write(str(reports / "model-junit.xml"))

    def validate_success(self):
        return host.validate_success(
            self.spool, self.trusted / "test-baseline.json", 1,
            ROOT / "model-test-manifest.json", "framework", "nmz1", "framework",
        )

    def test_host_baseline_uses_complete_collection_and_junit_uses_active_nodes(self):
        self.prepare_success()
        problems, details = self.validate_success()
        self.assertEqual(problems, [])
        self.assertEqual(details["collected"], 1)
        self.assertEqual(details["complete_collection"], len(self.collected))
        self.assertEqual(details["disabled"], len(self.disabled))

    def test_missing_active_junit_case_still_fails(self):
        self.prepare_success()
        ET.ElementTree(ET.Element("testsuite")).write(str(self.spool / "reports/junit-repeat-1.xml"))
        problems, _ = self.validate_success()
        self.assertTrue(any("JUnit cases; expected 1" in item for item in problems))

    def test_summary_cannot_hide_disabled_nodes(self):
        self.prepare_success()
        summary = host.read_json(self.spool / "test-summary.json")
        summary["disabled_tests"] = []
        host.atomic_json(self.spool / "test-summary.json", summary)
        problems, _ = self.validate_success()
        self.assertTrue(any("disabled records" in item for item in problems))

    def test_attention_baseline_definitions_exist_in_current_test_source(self):
        baseline = host.read_json(ROOT / "test-baseline.json")
        repository = ROOT.parents[1]

        def definitions(nodes, prefix=""):
            result = set()
            for node in nodes:
                if isinstance(node, ast.ClassDef):
                    result.update(definitions(node.body, prefix + node.name + "::"))
                elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    result.add(prefix + node.name)
            return result

        for path in baseline["required_test_files"]:
            self.assertTrue((repository / path).is_file(), path)
        for path, required in baseline["required_test_definitions"].items():
            # Upstream re-exported classes require actual collection, not AST.
            # Pin this static regression to the two locally adapted test files.
            if path not in {"tests/lmcache_hcu/v1/test_gpu_connector.py",
                            "tests/lmcache_hcu/v1/test_kv_layer_groups_manager.py"}:
                continue
            tree = ast.parse((repository / path).read_text(encoding="utf-8"))
            self.assertFalse(set(required) - definitions(tree.body), path)


if __name__ == "__main__":
    unittest.main(verbosity=2)

# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
"""Device-free tests of the maintainer-approved exact-nodeid disable policy."""

import argparse
import ast
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


ROOT = Path(__file__).resolve().parent
SPEC = importlib.util.spec_from_file_location("lmcache_ci_disabled", ROOT / "ci.py")
ci = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(ci)


class DisabledTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory(prefix="lmcache-disabled-test-")
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manifest = ci.read_json(ROOT / "disabled-tests.json")
        self.path = self.root / "disabled.json"

    def save(self, manifest=None):
        ci.write_json(self.path, self.manifest if manifest is None else manifest)
        return self.path

    def test_exactly_four_known_definitions_are_registered(self):
        self.assertEqual(len(self.manifest["tests"]), 4)
        tests_root = ROOT.parents[1] / "tests"
        for entry in self.manifest["tests"]:
            path, name = entry["nodeid"].split("::")
            tree = ast.parse((tests_root / path).read_text(encoding="utf-8"))
            functions = {node.name for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)}
            self.assertIn(name, functions)

    def test_exact_partition_preserves_new_and_similarly_named_tests(self):
        disabled = [entry["nodeid"] for entry in self.manifest["tests"]]
        extra = disabled[0] + "_new_regression"
        active, records = ci.partition_disabled_tests(disabled + [extra], self.save())
        self.assertEqual(active, [extra])
        self.assertEqual(records, self.manifest["tests"])

    def test_missing_registration_fails_closed(self):
        nodes = [entry["nodeid"] for entry in self.manifest["tests"]][1:]
        with self.assertRaisesRegex(ci.CIError, "did not collect"):
            ci.partition_disabled_tests(nodes, self.save())

    def test_duplicate_registration_is_rejected(self):
        self.manifest["tests"].append(copy.deepcopy(self.manifest["tests"][0]))
        with self.assertRaisesRegex(ci.CIError, "Duplicate disabled"):
            ci.partition_disabled_tests([], self.save())

    def test_missing_reason_and_path_traversal_are_rejected(self):
        for field, value in (("reason", ""), ("nodeid", "../test_bad.py::test_bad")):
            broken = copy.deepcopy(self.manifest)
            broken["tests"][0][field] = value
            with self.assertRaises(ci.CIError):
                ci.partition_disabled_tests([], self.save(broken))

    def test_entire_suite_cannot_be_disabled(self):
        with self.assertRaisesRegex(ci.CIError, "entire suite"):
            ci.partition_disabled_tests(
                [entry["nodeid"] for entry in self.manifest["tests"]], self.save()
            )

    def test_other_failure_is_not_suppressed_and_summary_is_explicit(self):
        tests = self.root / "tests"
        tests.mkdir()
        config = tests / "pytest.ini"
        config.write_text("[pytest]\n", encoding="utf-8")
        (tests / "test_boundary.py").write_text(
            "def test_disabled():\n    assert False\n"
            "def test_active_pass():\n    assert True\n"
            "def test_active_failure():\n    assert False\n", encoding="utf-8"
        )
        manifest = {
            "schema_version": 1, "approved_on": "2026-10-26",
            "tests": [{"nodeid": "test_boundary.py::test_disabled",
                       "reason": "selftest only", "restore_when": "selftest ends"}],
        }
        inventory = self.root / "inventory.json"
        execute = self.root / "execute"
        reports = self.root / "reports"
        state = self.root / "state"
        reports.mkdir()
        state.mkdir()
        ci.cmd_discover(argparse.Namespace(
            tests=str(tests), config=str(config), execute_dir=str(execute),
            output=str(inventory), collect_child=False, disabled_tests=str(self.save(manifest)),
        ))
        collected = ci.read_json(inventory)
        self.assertEqual((collected["collected_count"], collected["count"], collected["disabled_count"]), (3, 2, 1))
        rc = ci.cmd_execute(argparse.Namespace(
            inventory=str(inventory), config=str(config), execute_dir=str(execute),
            junit=str(reports / "junit-repeat-1.xml"),
            rc_output=str(state / "test-repeat-1.rc"),
        ))
        self.assertEqual(rc, 1)
        self.assertEqual(ci.junit_stats(reports / "junit-repeat-1.xml"),
                         {"tests": 2, "failures": 1, "errors": 0, "skipped": 0})
        result = ci.cmd_aggregate(argparse.Namespace(
            inventory=str(inventory), repeat=1, junit_dir=str(reports),
            state_dir=str(state), output=str(self.root / "summary.json"),
            markdown=str(self.root / "summary.md"),
        ))
        self.assertEqual(result, 1)
        summary = ci.read_json(self.root / "summary.json")
        self.assertEqual(summary["disabled_per_repeat"], 1)
        self.assertEqual(summary["status"], "failed")
        self.assertIn("test_boundary.py::test_disabled", (self.root / "summary.md").read_text())


if __name__ == "__main__":
    unittest.main(verbosity=2)

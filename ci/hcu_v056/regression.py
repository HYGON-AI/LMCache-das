"""Run the main-branch business tests against the installed wheel, not the checkout."""
from collections import Counter
import json
from pathlib import Path
import shutil
import sys
import xml.etree.ElementTree as ET

import pytest


def run(candidate, controller, temporary, reports):
    suite = temporary / "business-suite"
    shutil.copytree(candidate / "tests", suite / "tests")
    sys.path.insert(0, str(suite))  # only the tests/ package lives here
    root = suite / "tests"
    baseline = json.loads((controller / "ci/hcu/test-baseline.json").read_text())
    disabled = {row["nodeid"] for row in json.loads(
        (controller / "ci/hcu/disabled-tests.json").read_text())["tests"]}

    class Inventory:
        selected = []
        problems = []

        @pytest.hookimpl(trylast=True)
        def pytest_collection_modifyitems(self, config, items):
            ids = [item.nodeid for item in items]
            (reports / "business-inventory.json").write_text(json.dumps(ids, indent=2))
            definitions = {(i.nodeid.split("::")[0], getattr(i, "originalname", None) or i.name)
                           for i in items}
            self.problems = [f"{file}::{name}" for file, names in
                             baseline["required_test_definitions"].items() for name in names
                             if (file.removeprefix("tests/"), name) not in definitions]
            self.problems += sorted(disabled - set(ids))
            if len(ids) < 159 or len(set(ids)) != len(ids):
                self.problems.append(f"expected at least 159 unique business cases, collected {len(ids)}")
            omitted = [i for i in items if i.nodeid in disabled]
            items[:] = [i for i in items if i.nodeid not in disabled]
            self.selected = [i.name for i in items]
            config.hook.pytest_deselected(items=omitted)
            (reports / "business-selection.json").write_text(json.dumps({
                "excluded": [i.nodeid for i in omitted], "selected": [i.nodeid for i in items],
                "missing_contracts": self.problems}, indent=2))

    plugin = Inventory()
    xml = reports / "business.xml"
    rc = pytest.main(["-q", "-c", str(root / "pytest.ini"), "--rootdir", str(root),
                      "--import-mode=importlib", "--continue-on-collection-errors",
                      str(root), f"--junitxml={xml}"], plugins=[plugin])
    if rc:
        return int(rc)
    cases = list(ET.parse(xml).getroot().iter("testcase"))
    if (plugin.problems or not cases or Counter(c.get("name") for c in cases) != Counter(plugin.selected)
            or any(any(c.find(t) is not None for t in ("skipped", "error", "failure")) for c in cases)):
        print("Business inventory/JUnit mismatch:", plugin.problems, file=sys.stderr)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(run(*(Path(p).resolve() for p in sys.argv[1:])))

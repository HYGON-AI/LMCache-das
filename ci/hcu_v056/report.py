"""A missing, skipped or duplicate mandatory testcase is not a pass."""
import argparse
from collections import Counter
import xml.etree.ElementTree as ET


def verify(path, required):
    root = ET.parse(path).getroot()
    cases = list(root.iter("testcase"))
    if not cases:
        raise ValueError("empty JUnit inventory")
    for suite in root.iter("testsuite"):
        if int(suite.get("errors", 0)) or int(suite.get("failures", 0)):
            raise ValueError("JUnit suite reported errors or failures")
    counts = Counter()
    all_names = Counter(case.get("name") for case in cases)
    for case in cases:
        if case.find("failure") is not None or case.find("error") is not None:
            raise ValueError(f"JUnit failure/error: {case.get('name')}")
        if case.find("skipped") is None:
            counts[case.get("name")] += 1
    for name in required:
        if counts[name] != 1 or all_names[name] != 1:
            raise ValueError(f"{name}: expected one executed and passing testcase, found {counts[name]}")
    print(f"JUnit: {sum(counts.values())} executed, {len(cases)-sum(counts.values())} skipped")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("junit")
    parser.add_argument("--require", nargs="+", required=True)
    args = parser.parse_args()
    verify(args.junit, args.require)

"""CPU-only regressions for evidence checks; not a replacement for HCU tests."""
import json
from pathlib import Path
import runpy
import subprocess
import sys
from zipfile import ZipFile

import pytest

CI = Path(__file__).resolve().parent


def test_wheel_identity(tmp_path):
    wheel = tmp_path / "overlay.whl"
    with ZipFile(wheel, "w") as z:
        z.writestr("lmcache_hcu-0.5.6+hcu.dist-info/METADATA", "Name: lmcache-hcu\nVersion: 0.5.6+hcu\n")
    sha = "a" * 40
    cmd = [sys.executable, "-I", str(CI / "wheel.py")]
    subprocess.run([*cmd, "record", str(tmp_path), sha], check=True)
    assert json.loads((tmp_path / "provenance.json").read_text())["source_sha"] == sha
    subprocess.run([*cmd, "verify", str(tmp_path), sha], check=True)
    assert subprocess.run([*cmd, "verify", str(tmp_path), "b" * 40], capture_output=True).returncode != 0
    with ZipFile(wheel, "a") as z:
        z.writestr("new-content", "changed")
    assert subprocess.run([*cmd, "verify", str(tmp_path), sha], capture_output=True).returncode != 0


@pytest.mark.parametrize("body", [
    "", '<testcase name="other"/>',
    '<testcase name="must"><skipped/></testcase>', '<testcase name="must"><failure/></testcase>',
    '<testcase name="must"/><testcase name="must"/>',
])
def test_reject_incomplete_junit(tmp_path, body):
    report = tmp_path / "junit.xml"
    report.write_text("<testsuite>" + body + "</testsuite>", encoding="utf-8")
    with pytest.raises(ValueError):
        runpy.run_path(str(CI / "report.py"))["verify"](report, ["must"])


def test_external_counter():
    parse = runpy.run_path(str(CI / "test_qwen_mp.py"))["parse_external_hits"]
    assert parse('vllm:prefix_cache_hits_total{engine="0"} 512\n'
                 'vllm:external_prefix_cache_queries_total{engine="0"} 512\n') is None
    assert parse('vllm:external_prefix_cache_hits_total{engine="0"} 256\n'
                 'vllm:external_prefix_cache_hits_total{engine="1"} 128\n') == 384

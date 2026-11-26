"""One Qwen cold request, then a new vLLM process reusing the same MP server."""
import contextlib
import json
import os
from pathlib import Path
import re
import signal
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import ProxyHandler, Request, build_opener
import uuid


HTTP = build_opener(ProxyHandler({}))  # loopback only; never use a CI HTTP proxy
BASE = "http://127.0.0.1:20013"
HITS = re.compile(r"^vllm:external_prefix_cache_hits(?:_total)?(?:\{[^}]*\})?\s+([0-9.eE+-]+)(?:\s|$)")


def request(path, data=None, timeout=5):
    body = None if data is None else json.dumps(data).encode()
    req = Request(BASE + path, data=body, headers={"Content-Type": "application/json"})
    with HTTP.open(req, timeout=timeout) as response:
        return response.read().decode()


def external_hits():
    return parse_external_hits(request("/metrics"))


def parse_external_hits(metrics):
    values = [float(m[1]) for line in metrics.splitlines() if (m := HITS.match(line))]
    return sum(values) if values else None


def wait_until(procs, probe, seconds, description):
    end = time.monotonic() + seconds
    while time.monotonic() < end:
        for proc in procs:
            if proc.poll() is not None:
                raise RuntimeError(f"{description}: service exited {proc.returncode}; see service log")
        try:
            if result := probe():
                return result
        except (OSError, URLError, TimeoutError):
            pass  # only while waiting for a startup/metrics condition
        time.sleep(2)
    raise TimeoutError(description)


@contextlib.contextmanager
def service(command, log):
    # Each process and its workers belong to a private process group in this pod.
    with log.open("wb") as stream:
        proc = subprocess.Popen(command, stdout=stream, stderr=subprocess.STDOUT, start_new_session=True)
        try:
            yield proc
        finally:
            # Give the API parent time to drain its connector and workers (in
            # particular the cold store) before cleaning up this group's stragglers.
            proc.terminate()
            try:
                proc.wait(timeout=90)
            except subprocess.TimeoutExpired:
                pass
            with contextlib.suppress(ProcessLookupError):
                os.killpg(proc.pid, signal.SIGKILL)
            proc.wait(timeout=15)


def mp_listening():
    with socket.create_connection(("127.0.0.1", 5555), timeout=2):
        return True


def test_qwen_mp_cache_reuse():
    out = Path(os.environ["CI_REPORT_DIR"])
    # Both services and all their data live only in this single-use pod.
    connector = {"kv_connector": "LMCacheMPConnector", "kv_role": "kv_both",
                 "kv_connector_module_path": "lmcache.integration.vllm.lmcache_mp_connector",
                 "kv_connector_extra_config": {"lmcache.mp.host": "127.0.0.1", "lmcache.mp.port": 5555}}
    mp_cmd = [str(Path(sys.executable).parent / "lmcache"), "server", "--host", "127.0.0.1",
              "--port", "5555", "--l1-size-gb", "20", "--eviction-policy", "LRU"]
    # Use the venv interpreter even if the image's vllm console script has a
    # system-Python shebang (which would silently import the image's LMCache).
    api_cmd = [sys.executable, "-I", "-c", "from vllm.entrypoints.cli.main import main; main()",
               "serve", os.environ["QWEN_MODEL"], "--served-model-name", "ci-qwen",
               "--host", "127.0.0.1", "--port", "20013", "--tensor-parallel-size", os.environ["TEST_TP"],
               "--max-model-len", "4096", "--kv-transfer-config", json.dumps(connector)]
    prompt = uuid.uuid4().hex + "\n" + ("The cache holds a sequence of tokens. " * 190) + "Continue:"
    payload = {"model": "ci-qwen", "prompt": prompt, "temperature": 0, "max_tokens": 8, "ignore_eos": True}
    result = {}
    with service(mp_cmd, out / "lmcache-mp.log") as mp:
        wait_until([mp], mp_listening, 120, "LMCache MP did not start")
        for phase in ("cold", "warm"):
            with service(api_cmd, out / f"vllm-{phase}.log") as api:
                wait_until([mp, api], lambda: request("/v1/models"), 900, f"Qwen {phase} startup")
                before = external_hits() or 0
                response = json.loads(request("/v1/completions", payload, timeout=240))
                assert len(response["choices"]) == 1 and response["choices"][0]["text"]
                usage = response["usage"]
                assert usage["prompt_tokens"] >= 512 and usage["completion_tokens"] > 0
                result[phase] = {"prompt_tokens": usage["prompt_tokens"], "completion_tokens": usage["completion_tokens"],
                                 "external_hits_before": before, "external_hits_after": external_hits()}
                if phase == "warm":
                    # No local vLLM prefix cache survives this restart. Require
                    # actual hits, not just an MP lookup, a fast response or HTTP 200.
                    result[phase]["external_hits_after"] = wait_until(
                        [mp, api], lambda: n if (n := external_hits()) is not None and n > before else None,
                        45, "no external LMCache hit recorded for the warm request")
                (out / "qwen-mp.json").write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")

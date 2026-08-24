"""Trusted pytest support for the LMCache HCU CI environment.

The upstream v0.3.13 ``lmserver_v1_process`` fixture allows only fifteen
seconds for a server to become ready.  Importing the HCU runtime in the pinned
test image takes about twenty seconds, so the fixture terminates a healthy
server before it can listen.  This plugin keeps the test semantics unchanged
while using a bounded, process-aware readiness probe.
"""

from __future__ import annotations

import os
import socket
import subprocess
import sys
import time
from types import SimpleNamespace

import pytest


SERVER_FIXTURE = "lmserver_v1_process"
START_TIMEOUT_SECONDS = 60.0


def _wait_until_ready(process: subprocess.Popen[bytes], port: int) -> None:
    deadline = time.monotonic() + START_TIMEOUT_SECONDS
    last_error = "server did not accept connections"
    while time.monotonic() < deadline:
        return_code = process.poll()
        if return_code is not None:
            raise RuntimeError(
                f"LMCache test server exited before readiness (rc={return_code})"
            )
        try:
            with socket.create_connection(("127.0.0.1", port), timeout=0.5):
                return
        except OSError as exc:
            last_error = str(exc)
            time.sleep(0.5)
    raise TimeoutError(
        f"LMCache test server was not ready within {START_TIMEOUT_SECONDS:.0f}s: "
        f"{last_error}"
    )


def _stop_process(process: subprocess.Popen[bytes]) -> None:
    if process.poll() is not None:
        return
    process.terminate()
    try:
        process.wait(timeout=10)
    except subprocess.TimeoutExpired:
        process.kill()
        process.wait(timeout=10)


@pytest.hookimpl(tryfirst=True)
def pytest_fixture_setup(fixturedef: object, request: pytest.FixtureRequest) -> object:
    """Replace only the known-broken upstream LMCache server fixture."""
    if getattr(fixturedef, "argname", None) != SERVER_FIXTURE:
        return None

    device = getattr(request, "param", None)
    if not isinstance(device, str) or not device:
        raise RuntimeError("lmserver_v1_process requires a non-empty device parameter")

    listener = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    listener.bind(("127.0.0.1", 0))
    port = listener.getsockname()[1]
    listener.close()

    environment = os.environ.copy()
    environment["PYTHONHASHSEED"] = "0"
    process = subprocess.Popen(
        [
            sys.executable,
            "-m",
            "lmcache.v1.server",
            "localhost",
            str(port),
            device,
        ],
        env=environment,
    )
    try:
        _wait_until_ready(process, port)
    except BaseException:
        _stop_process(process)
        raise

    def finalize() -> None:
        _stop_process(process)
        if device != "cpu":
            # The upstream fixture accepts a path as its non-CPU device.  Keep
            # deletion constrained to the disposable sandbox.
            resolved = os.path.realpath(device)
            sandbox = os.path.realpath("/sandbox") + os.sep
            if resolved.startswith(sandbox) and os.path.isdir(resolved):
                import shutil

                shutil.rmtree(resolved)

    request.addfinalizer(finalize)
    result = SimpleNamespace(
        server_url=f"lm://localhost:{port}",
        server_process=process,
    )
    # The default pytest_fixture_setup implementation owns this cache update.
    # Since this first-result hook replaces that implementation, it must keep
    # the same FixtureDef contract for subsequent fixture lookups.
    fixturedef.cached_result = (result, fixturedef.cache_key(request), None)
    return result

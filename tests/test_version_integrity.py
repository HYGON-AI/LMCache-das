# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Based on LMCache-Ascend v0.3.12 (58e0aef462eda7c451f84735b277b35e02dce227).
"""Verify that the installed upstream LMCache version matches LMCache-HCU."""
from __future__ import annotations

# Standard
from importlib.metadata import PackageNotFoundError, version
import warnings

# Third Party
import pytest

# First Party
import lmcache_hcu


def test_dependency_compatibility():
    """Ensure LMCache-HCU is tested with its supported upstream LMCache version."""
    try:
        installed_ver = version("lmcache")
    except PackageNotFoundError:
        pytest.fail("'lmcache' is not installed in the current environment.")

    target_tag = lmcache_hcu.LMCACHE_UPSTREAM_TAG
    clean_target = target_tag.lstrip("v")

    print(f"\n[Version Check] Installed LMCache: {installed_ver}; target: {target_tag}")

    if installed_ver.startswith(clean_target):
        return

    dev_markers = [".dev", "+", "dirty", "a", "b", "rc"]
    if any(marker in installed_ver for marker in dev_markers):
        warnings.warn(
            "Allowing LMCache version mismatch for a development build.\n"
            f"Target tag: {target_tag}\n"
            f"Installed version: {installed_ver}",
            stacklevel=2,
        )
        return

    pytest.fail(
        "Upstream LMCache version mismatch.\n"
        f"LMCache-HCU expects: {target_tag}\n"
        f"Installed LMCache version: {installed_ver}\n"
        "Install the matching upstream LMCache version or update "
        "LMCACHE_UPSTREAM_TAG in lmcache_hcu/__init__.py."
    )

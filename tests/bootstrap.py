# SPDX-License-Identifier: Apache-2.0
# Standard
# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# Modified by Hygon Information Technology Co., Ltd., 2026.
"""Prepare installed upstream LMCache tests for LMCache-HCU test reuse."""
from __future__ import annotations

# Standard
import importlib.util
import logging
import os
import subprocess
import sys
from importlib.metadata import PackageNotFoundError, version

# First Party
import lmcache_hcu

logger = logging.getLogger(__name__)

VERSION_TAG = lmcache_hcu.LMCACHE_UPSTREAM_TAG
TEST_ALIAS = "lmcache_tests"
LOCAL_UPSTREAM_SOURCE = os.path.abspath(
    os.path.join(
        os.path.dirname(__file__),
        "..",
        "..",
        "LMCache_0.3.13_source",
        "LMCache",
    )
)


def _has_tests_package(path: str) -> bool:
    return os.path.exists(os.path.join(path, "tests", "__init__.py"))


def resolve_lmcache_path() -> str:
    """Return the upstream LMCache source root used by the test alias."""
    override = os.environ.get("LMCACHEPATH")
    if override:
        return os.path.abspath(override)

    spec = importlib.util.find_spec("lmcache")
    if spec is not None and spec.origin is not None:
        package_dir = os.path.dirname(os.path.abspath(spec.origin))
        installed_root = os.path.dirname(package_dir)
        if _has_tests_package(installed_root):
            return installed_root

    if _has_tests_package(LOCAL_UPSTREAM_SOURCE):
        return LOCAL_UPSTREAM_SOURCE

    if spec is None or spec.origin is None:
        raise ModuleNotFoundError("Cannot find installed package 'lmcache'.")

    return os.path.dirname(os.path.dirname(os.path.abspath(spec.origin)))


LMCACHEPATH = resolve_lmcache_path()


def get_current_git_tag(path: str) -> str | None:
    """Return the exact git tag checked out at path, or None."""
    try:
        return (
            subprocess.check_output(
                ["git", "describe", "--tags", "--exact-match"],
                cwd=path,
                stderr=subprocess.DEVNULL,
            )
            .decode()
            .strip()
        )
    except (subprocess.CalledProcessError, FileNotFoundError):
        return None


def warn_if_lmcache_version_mismatch() -> None:
    """Warn when installed LMCache git tag does not match the supported tag."""
    current_tag = get_current_git_tag(LMCACHEPATH)
    if current_tag is None:
        try:
            installed_ver = version("lmcache")
        except PackageNotFoundError:
            logger.warning(
                "Cannot verify LMCache version because neither an exact git tag "
                "nor package metadata is available for %s.",
                LMCACHEPATH,
            )
            return

        logger.warning(
            "Cannot read an exact git tag from %s; installed LMCache package "
            "metadata reports version %s. LMCache-HCU targets %s.",
            LMCACHEPATH,
            installed_ver,
            VERSION_TAG,
        )
        return

    if current_tag == VERSION_TAG:
        return

    logger.warning(
        "LMCache git tag mismatch: LMCache-HCU targets %s, but installed "
        "LMCache source at %s is checked out at %s. Reused upstream tests may "
        "not match this patch version.",
        VERSION_TAG,
        LMCACHEPATH,
        current_tag,
    )


def setup_lmcache_dependency() -> None:
    """Validate that the installed LMCache path contains reusable tests."""
    tests_init_path = os.path.join(LMCACHEPATH, "tests", "__init__.py")
    if not os.path.exists(tests_init_path):
        raise FileNotFoundError(
            "Installed LMCache tests package not found. "
            f"Expected: {tests_init_path}. "
            "Install LMCache from source in editable mode or set LMCACHEPATH "
            "to the upstream LMCache source root."
        )


def register_alias() -> None:
    """Register installed upstream LMCache tests as the lmcache_tests alias."""
    if LMCACHEPATH not in sys.path:
        sys.path.append(LMCACHEPATH)

    if TEST_ALIAS in sys.modules:
        return

    tests_dir = os.path.join(LMCACHEPATH, "tests")
    tests_init_path = os.path.join(tests_dir, "__init__.py")

    spec = importlib.util.spec_from_file_location(
        TEST_ALIAS,
        tests_init_path,
        submodule_search_locations=[tests_dir],
    )
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load upstream tests package from {tests_init_path}")

    module = importlib.util.module_from_spec(spec)
    sys.modules[TEST_ALIAS] = module
    spec.loader.exec_module(module)
    print(f"Registered upstream tests alias '{TEST_ALIAS}' from {tests_init_path}")


def prepare_environment() -> None:
    """Prepare the installed LMCache tests alias."""
    warn_if_lmcache_version_mismatch()
    setup_lmcache_dependency()
    register_alias()

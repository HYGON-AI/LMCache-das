#
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

from __future__ import annotations

import argparse
import importlib
import logging
import subprocess
from pathlib import Path

from lmcache_hcu.integration.patch.base_patcher import logger


# Supported LMCache source revision.
UPSTREAM_TAG = "v0.5.6"
UPSTREAM_COMMIT_PREFIX = "3e11e108"


# Module, patcher class, and conversion method.
PATCH_TASKS = [
    (
        "lmcache_hcu.integration.patch.lmcache.rocm_build_patch",
        "LMCacheRocmBuildPatcher",
        "_replace_define_macros",
    ),
    (
        "lmcache_hcu.integration.patch.lmcache.device_mapping_patch",
        "LMCacheDeviceMappingPatcher",
        "_replace_function",
    ),
]


def _git(root: Path, *args: str) -> str:
    result = subprocess.run(
        ["git", "-C", str(root), *args],
        capture_output=True,
        text=True,
        check=False,
    )
    if result.returncode != 0:
        raise RuntimeError(
            f"Git command failed: {' '.join(args)}\n"
            f"{result.stderr.strip()}"
        )
    return result.stdout


def verify_source_root(root: Path) -> None:
    """Require a Git checkout based on LMCache v0.5.6."""

    if not (root / "lmcache" / "__init__.py").is_file():
        raise RuntimeError(f"Not an LMCache source tree: {root}")

    git_root = Path(
        _git(root, "rev-parse", "--show-toplevel").strip()
    ).resolve()

    if git_root != root:
        raise RuntimeError(
            f"LMCache root must be the Git repository root: {git_root}"
        )

    commit = _git(
        root, "rev-parse", f"{UPSTREAM_TAG}^{{commit}}"
    ).strip()

    if not commit.startswith(UPSTREAM_COMMIT_PREFIX):
        raise RuntimeError(
            f"Unexpected LMCache v0.5.6 revision: {commit}"
        )

    result = subprocess.run(
        [
            "git", "-C", str(root),
            "merge-base", "--is-ancestor",
            UPSTREAM_TAG, "HEAD",
        ],
        capture_output=True,
        text=True,
        check=False,
    )

    if result.returncode != 0:
        raise RuntimeError(
            "Source HEAD is not based on LMCache v0.5.6"
        )


def inspect_patches(root: Path):
    """Validate all targets before modifying any source file."""

    states = []

    for module_path, class_name, converter_name in PATCH_TASKS:
        module = importlib.import_module(module_path)
        patcher = getattr(module, class_name)

        target = patcher._target_file(root)

        original = _git(
            root, "show",
            f"{UPSTREAM_TAG}:{patcher.TARGET_FILE}",
        )

        expected = getattr(patcher, converter_name)(original)
        current = target.read_text(encoding="utf-8")

        if current == original:
            state = "original"
        elif current == expected:
            state = "applied"
        else:
            raise RuntimeError(
                f"Unexpected or partially patched source: {target}"
            )

        states.append((patcher, state))

    return states


def run_integration_patches(root: Path, mode: str) -> None:
    verify_source_root(root)

    # Preflight all patches before writing anything.
    states = inspect_patches(root)

    for patcher, state in states:
        logger.info("%s: %s", patcher.__name__, state)

    if mode == "check":
        logger.info("All source patch checks passed")
        return

    for patcher, state in states:
        if state == "applied":
            logger.info("%s already applied; skip", patcher.__name__)
            continue

        logger.info("Applying %s", patcher.__name__)
        patcher.apply_all(root)

    logger.info("LMCache-HCU patches completed")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="LMCache v0.5.6 HCU source patch manager"
    )

    parser.add_argument(
        "--lmcache-root",
        type=Path,
        required=True,
        help="Path to the LMCache v0.5.6 Git source tree",
    )

    parser.add_argument(
        "--mode",
        choices=("check", "install"),
        default="check",
    )

    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO)

    try:
        root = args.lmcache_root.expanduser().resolve(strict=True)
        run_integration_patches(root, args.mode)
    except (OSError, RuntimeError, ValueError) as exc:
        parser.exit(1, f"ERROR: {exc}\n")


if __name__ == "__main__":
    main()

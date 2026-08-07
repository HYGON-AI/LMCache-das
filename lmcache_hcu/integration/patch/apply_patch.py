#
# Copyright 2026 Hygon Information Technology Co., Ltd.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http:#www.apache.org/licenses/LICENSE-2.0
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
import importlib.util
import logging
import os
from typing import Iterable, Literal

from lmcache_hcu.integration.patch.base_patcher import logger


PATCH_TASKS = [
    ("lmcache", "lmcache_hcu.integration.patch.lmcache.runtime_import_patch", "LMCacheRuntimeImportPatcher"),
    ("lmcache", "lmcache_hcu.integration.patch.vllm.connector_import_patch", "VllmConnectorImportPatcher"),
]


def is_installed(package_name: str) -> bool:
    return importlib.util.find_spec(package_name) is not None


def run_integration_patches(
    tasks: Iterable[tuple[str, str, str]] = PATCH_TASKS,
    mode: Literal["install", "uninstall"] = "install",
) -> None:
    if os.environ.get("SKIP_LMCACHE_HCU_PATCH", "0") == "1":
        logger.info("SKIP_LMCACHE_HCU_PATCH=1; skip LMCache-HCU source patches")
        return

    logger.info("Initializing LMCache-HCU patch manager in %s mode...", mode)
    action_name = "apply_all" if mode == "install" else "uninstall_all"
    for package_name, module_path, class_name in tasks:
        if not is_installed(package_name):
            logger.warning("Package %s is not installed; skip %s", package_name, class_name)
            continue
        module = importlib.import_module(module_path)
        patcher_cls = getattr(module, class_name)
        logger.info("%s %s from %s", "Applying" if mode == "install" else "Uninstalling", class_name, module_path)
        getattr(patcher_cls, action_name)()
    logger.info("LMCache-HCU patch manager finished")


def main() -> None:
    parser = argparse.ArgumentParser(description="Apply or uninstall LMCache-HCU source patches")
    parser.add_argument(
        "--mode",
        choices=("install", "uninstall"),
        default="install",
        help="install applies source patches; uninstall restores each target from its earliest .bak backup",
    )
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO)
    run_integration_patches(mode=args.mode)


if __name__ == "__main__":
    main()

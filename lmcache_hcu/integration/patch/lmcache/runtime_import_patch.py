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

from pathlib import Path

from lmcache_hcu.integration.patch.base_patcher import BasePatcher, TextPatch, VersionRange, logger


HCU_IMPORT_BLOCK = """\n# BEGIN LMCACHE_HCU_RUNTIME_PATCH\n# Importing lmcache_hcu installs runtime monkey patches and redirects c_ops.\ntry:\n    import lmcache_hcu  # noqa: F401\nexcept Exception as _lmcache_hcu_exc:\n    import warnings as _lmcache_hcu_warnings\n    _lmcache_hcu_warnings.warn(\n        f\"LMCache-HCU runtime patch import failed: {_lmcache_hcu_exc}\",\n        RuntimeWarning,\n    )\n# END LMCACHE_HCU_RUNTIME_PATCH\n"""


class LMCacheRuntimeImportPatcher(BasePatcher):
    """Inject `import lmcache_hcu` into upstream LMCache package import path."""

    PACKAGE_NAME = "lmcache"
    VERSION_SERIES = (VersionRange("0.3.13", "0.3.14"),)

    @staticmethod
    def _target_file() -> Path:
        return BasePatcher.find_module_path("lmcache")

    @staticmethod
    def _insert_runtime_import(text: str) -> str:
        if "# END LMCACHE_HCU_RUNTIME_PATCH" in text:
            return text
        return text.rstrip() + HCU_IMPORT_BLOCK + "\n"

    @classmethod
    def apply_all(cls) -> bool:
        if not cls.should_run_for_version():
            return False
        target = cls._target_file()
        return cls.apply_text_patches(
            target,
            [
                TextPatch(
                    name="lmcache-import-lmcache-hcu",
                    marker="# END LMCACHE_HCU_RUNTIME_PATCH",
                    locator="# SPDX-License-Identifier: Apache-2.0",
                    replacement=cls._insert_runtime_import,
                )
            ],
        )


if __name__ == "__main__":
    logging_configured = logger.handlers or logger.parent.handlers
    if not logging_configured:
        import logging
        logging.basicConfig(level=logging.INFO)
    LMCacheRuntimeImportPatcher.apply_all()

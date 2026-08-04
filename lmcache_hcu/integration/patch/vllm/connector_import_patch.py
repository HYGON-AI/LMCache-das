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

from lmcache_hcu.integration.patch.base_patcher import BasePatcher, TextPatch, logger


VLLM_CONNECTOR_IMPORT = """\n# BEGIN LMCACHE_HCU_VLLM_CONNECTOR_PATCH\n# Ensure LMCache-HCU runtime patches are installed before LMCacheConnectorV1Impl is imported.\ntry:\n    import lmcache_hcu  # noqa: F401\nexcept Exception as _lmcache_hcu_exc:\n    import warnings as _lmcache_hcu_warnings\n    _lmcache_hcu_warnings.warn(\n        f\"LMCache-HCU vLLM connector patch import failed: {_lmcache_hcu_exc}\",\n        RuntimeWarning,\n    )\n# END LMCACHE_HCU_VLLM_CONNECTOR_PATCH\n"""


class VllmConnectorImportPatcher(BasePatcher):
    """Inject lmcache_hcu import into LMCache's vLLM connector module.

    This mirrors LMCache-Ascend's source-modification layer but keeps the source
    rewrite tiny: vLLM still loads LMCache's normal connector, and the injected
    import activates all HCU runtime patches before connector implementation
    imports bind to lmcache.c_ops and config classes.
    """

    PACKAGE_NAME = "lmcache"

    @staticmethod
    def _target_file() -> Path:
        return BasePatcher.find_module_path("lmcache.integration.vllm.lmcache_connector_v1")

    @staticmethod
    def _insert_before_impl_import(text: str) -> str:
        marker = "# END LMCACHE_HCU_VLLM_CONNECTOR_PATCH"
        if marker in text:
            return text
        needle = "from lmcache.integration.vllm.vllm_v1_adapter import LMCacheConnectorV1Impl\n"
        if needle in text:
            return text.replace(needle, VLLM_CONNECTOR_IMPORT + "\n" + needle, 1)
        return text.rstrip() + VLLM_CONNECTOR_IMPORT + "\n"

    @classmethod
    def apply_all(cls) -> bool:
        target = cls._target_file()
        return cls.apply_text_patches(
            target,
            [
                TextPatch(
                    name="vllm-connector-import-lmcache-hcu",
                    marker="# END LMCACHE_HCU_VLLM_CONNECTOR_PATCH",
                    locator="LMCacheConnectorV1Impl",
                    replacement=cls._insert_before_impl_import,
                )
            ],
        )


if __name__ == "__main__":
    logging_configured = logger.handlers or logger.parent.handlers
    if not logging_configured:
        import logging
        logging.basicConfig(level=logging.INFO)
    VllmConnectorImportPatcher.apply_all()

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

from pathlib import Path

from lmcache_hcu.integration.patch.base_patcher import (
    BasePatcher,
    TextPatch,
    VersionRange,
)


TYPE_CHECKING_ANCHOR = "if TYPE_CHECKING:"
TYPE_CHECKING_BLOCK = (
    "if TYPE_CHECKING:\n"
    "    # BEGIN LMCACHE_HCU_DISK_METADATA\n"
    "    from lmcache_hcu.v1.storage_backend.block_first_disk import (\n"
    "        BlockFirstDiskMetadata,\n"
    "    )\n"
    "    # END LMCACHE_HCU_DISK_METADATA\n"
)

# Anchor: tail of the DiskCacheMetadata field block.
DISK_FIELD_ANCHOR = "    cached_positions: Optional[torch.Tensor] = None\n    fmt: Optional[MemoryFormat] = None\n    pin_count: int = 0\n"
DISK_FIELD_BLOCK = DISK_FIELD_ANCHOR + "    # BEGIN LMCACHE_HCU_DISK_METADATA\n    disk_metadata: Optional[BlockFirstDiskMetadata] = None\n    # END LMCACHE_HCU_DISK_METADATA\n"


class LMCacheUtilsMetadataPatcher(BasePatcher):
    """Add block-first `disk_metadata` to `DiskCacheMetadata` in installed lmcache/utils.py.

    A small marker-based source patch, so the installed upstream module exposes the
    field the HCU block-first backends carry on their cache metadata.
    """

    PACKAGE_NAME = "lmcache"
    VERSION_SERIES = (VersionRange("0.4.2", "0.4.3"),)

    @staticmethod
    def _target_file() -> Path:
        # find_module_path() returns the package __init__.py file for packages,
        # so anchor on its parent directory before appending the module name.
        found = BasePatcher.find_module_path("lmcache")
        package_dir = found if found.is_dir() else found.parent
        return package_dir / "utils.py"

    @staticmethod
    def _insert_type_checking(text: str) -> str:
        if "# END LMCACHE_HCU_DISK_METADATA" in text:
            return text
        idx = text.find(TYPE_CHECKING_ANCHOR)
        if idx < 0:
            return text
        end = idx + len(TYPE_CHECKING_ANCHOR)
        nl = text.find("\n", end)
        return text[: nl + 1] + TYPE_CHECKING_BLOCK.split("\n", 1)[1] + text[nl + 1 :]

    @staticmethod
    def _insert_field(text: str) -> str:
        if "disk_metadata: Optional[BlockFirstDiskMetadata]" in text:
            return text
        if DISK_FIELD_ANCHOR not in text:
            return text
        return text.replace(DISK_FIELD_ANCHOR, DISK_FIELD_BLOCK, 1)

    @classmethod
    def apply_all(cls) -> bool:
        if not cls.should_run_for_version():
            return False
        target = cls._target_file()
        return cls.apply_text_patches(
            target,
            [
                TextPatch(
                    name="lmcache-utils-disc-metadata-import",
                    marker="# END LMCACHE_HCU_DISK_METADATA",
                    locator=TYPE_CHECKING_ANCHOR,
                    replacement=cls._insert_type_checking,
                ),
                TextPatch(
                    name="lmcache-utils-disc-metadata-field",
                    marker="disk_metadata: Optional[BlockFirstDiskMetadata]",
                    locator=DISK_FIELD_ANCHOR,
                    replacement=cls._insert_field,
                ),
            ],
        )

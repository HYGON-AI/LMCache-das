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
)


class LMCacheRocmBuildPatcher(BasePatcher):
    """Disable streaming I/O during LMCache v0.5.6 ROCm builds."""

    TARGET_FILE = "setup_extensions/build_profiles/rocm.py"

    OLD_LINE = (
        '        define_macros = '
        '[("__HIP_PLATFORM_HCC__", "1"), ("USE_ROCM", "1")]'
    )

    NEW_LINE = (
        '        define_macros = '
        '[("__HIP_PLATFORM_HCC__", "1"), ("USE_ROCM", "1"), '
        '("LMCACHE_DISABLE_STREAMING_IO", "1")]'
    )

    @classmethod
    def _target_file(cls, source_root: str | Path) -> Path:
        return cls.find_source_file(source_root, cls.TARGET_FILE)

    @classmethod
    def _replace_define_macros(cls, text: str) -> str:
        if text.count(cls.OLD_LINE) != 1:
            raise RuntimeError(
                "Expected exactly one original ROCm define_macros line"
            )

        return text.replace(cls.OLD_LINE, cls.NEW_LINE, 1)

    @classmethod
    def apply_all(cls, source_root: str | Path) -> bool:
        target = cls._target_file(source_root)

        return cls.apply_text_patches(
            target,
            [
                TextPatch(
                    name="disable-rocm-streaming-io",
                    marker=cls.NEW_LINE,
                    locator=cls.OLD_LINE,
                    replacement=cls._replace_define_macros,
                ),
            ],
        )

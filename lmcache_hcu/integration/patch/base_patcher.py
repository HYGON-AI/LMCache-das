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

from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Callable, Iterable, Sequence
import importlib.util
import logging
import shutil
import time

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class VersionRange:
    """Inclusive/exclusive version interval used by source patchers."""

    min_version: str | None = None
    max_version: str | None = None
    include_min: bool = True
    include_max: bool = False


def _version_tuple(version: str) -> tuple[int, ...]:
    parts: list[int] = []
    for token in version.replace("-", ".").replace("+", ".").split("."):
        if token.isdigit():
            parts.append(int(token))
        else:
            digits = "".join(ch for ch in token if ch.isdigit())
            if digits:
                parts.append(int(digits))
                break
    return tuple(parts) or (0,)


@dataclass(frozen=True)
class TextPatch:
    """Declarative text patch.

    marker: idempotency marker. If marker is already present in the target file,
        the patch is skipped.
    locator: string expected to exist in the target file.
    replacement: callable that receives the old file text and returns the new text.
    """

    name: str
    marker: str
    locator: str
    replacement: Callable[[str], str]


class BasePatcher:
    """Small source-rewrite helper modeled after LMCache-Ascend's patcher.

    Patchers locate the installed module with importlib, create a timestamped
    backup, then apply idempotent string patches. Keep heavy HCU logic in
    lmcache_hcu runtime modules; source rewrites should only inject imports or
    bridge points.
    """

    PACKAGE_NAME: str = ""
    VERSION_SERIES: Sequence[VersionRange] = ()

    @classmethod
    def get_version(cls, package_name: str | None = None) -> str | None:
        package = package_name or cls.PACKAGE_NAME
        if not package:
            return None
        try:
            return metadata.version(package)
        except metadata.PackageNotFoundError:
            return None

    @classmethod
    def is_version_in_range(cls, version: str, version_range: VersionRange) -> bool:
        current = _version_tuple(version)
        if version_range.min_version is not None:
            lower = _version_tuple(version_range.min_version)
            if current < lower or (current == lower and not version_range.include_min):
                return False
        if version_range.max_version is not None:
            upper = _version_tuple(version_range.max_version)
            if current > upper or (current == upper and not version_range.include_max):
                return False
        return True

    @classmethod
    def should_run_for_version(cls) -> bool:
        if not cls.VERSION_SERIES:
            return True
        version = cls.get_version()
        if version is None:
            logger.warning("Package %s is not installed; skip %s", cls.PACKAGE_NAME, cls.__name__)
            return False
        return any(cls.is_version_in_range(version, item) for item in cls.VERSION_SERIES)

    @staticmethod
    def find_module_path(module_name: str) -> Path:
        spec = importlib.util.find_spec(module_name)
        if spec is None or spec.origin is None:
            raise ModuleNotFoundError(f"Cannot locate module {module_name!r}")
        return Path(spec.origin)

    @staticmethod
    def backup_file(path: Path) -> Path:
        backup = path.with_suffix(path.suffix + f".bak.{int(time.time())}")
        shutil.copy2(path, backup)
        return backup

    @classmethod
    def apply_text_patches(cls, path: Path, patches: Iterable[TextPatch]) -> bool:
        original = path.read_text(encoding="utf-8")
        text = original
        changed = False
        for patch in patches:
            if patch.marker in text:
                logger.info("Patch %s already present in %s", patch.name, path)
                continue
            if patch.locator not in text:
                raise RuntimeError(
                    f"Patch {patch.name!r} locator not found in {path}: {patch.locator!r}"
                )
            new_text = patch.replacement(text)
            if new_text == text:
                raise RuntimeError(f"Patch {patch.name!r} made no changes to {path}")
            text = new_text
            changed = True

        if changed:
            backup = cls.backup_file(path)
            path.write_text(text, encoding="utf-8")
            logger.info("Patched %s; backup: %s", path, backup)
        return changed

    @classmethod
    def apply_all(cls) -> bool:
        raise NotImplementedError

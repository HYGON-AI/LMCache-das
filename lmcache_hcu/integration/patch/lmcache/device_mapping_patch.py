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


class LMCacheDeviceMappingPatcher(BasePatcher):
    """Fix multi-node vLLM local device mapping for LMCache v0.5.6."""

    TARGET_FILE = "lmcache/integration/vllm/utils.py"

    ORIGINAL_DOCSTRING = '''    """
    Calculate the local worker id and local world size.

    Current assumption (TODO: add custom logic in the future):
    - Tensor Parallel is intra-node
    - Pipeline Parallel is inter-node

    Returns:
        Tuple[int, int]: (local_worker_id, local_world_size)
    """'''

    PATCHED_DOCSTRING = '''    """
    Return the local device index and worker count for a vLLM worker.

    Multi-node workers must call this after vLLM binds their device.
    The local worker count assumes one worker per visible device.

    Args:
        vllm_config: vLLM configuration containing the parallel configuration.

    Returns:
        Tuple[int, int]: (local device index, local worker count).
    """'''

    ORIGINAL_BRANCH = '''    else:
        tp_size = parallel_config.tensor_parallel_size
        pp_size = parallel_config.pipeline_parallel_size
        local_world_size = global_world_size // pp_size
        assert local_world_size == tp_size, (
            "LMCache is operating under the assumption that the "
            "local world size is equal to the tensor parallel size "
            "in multi-node deployment."
        )
        local_worker_id = global_rank % local_world_size
        return local_worker_id, local_world_size'''

    PATCHED_BRANCH = '''    else:
        local_worker_id = torch_dev.current_device()
        local_world_size = num_gpus
        logger.info(
            "Using vLLM-bound device: global_rank=%d, local_worker_id=%d, "
            "local_world_size=%d",
            global_rank,
            local_worker_id,
            local_world_size,
        )
        return local_worker_id, local_world_size'''

    PACKAGE_NAME = "lmcache"
    VERSION_SERIES = (VersionRange("0.5.6", "0.5.7"),)

    @staticmethod
    def _target_file() -> Path:
        found = BasePatcher.find_module_path("lmcache")
        package_dir = found if found.is_dir() else found.parent
        return package_dir / "integration" / "vllm" / "utils.py"

    @classmethod
    def _replace_function(cls, text: str) -> str:
        replacements = (
            (cls.ORIGINAL_DOCSTRING, cls.PATCHED_DOCSTRING),
            (cls.ORIGINAL_BRANCH, cls.PATCHED_BRANCH),
        )

        for old, new in replacements:
            if text.count(old) != 1:
                raise RuntimeError(
                    "Expected exactly one original device mapping section"
                )
            text = text.replace(old, new, 1)

        return text

    @classmethod
    def apply_all(cls) -> bool:
        if not cls.should_run_for_version():
            return False

        target = cls._target_file()
        text = target.read_text(encoding="utf-8")

        original = (
            text.count(cls.ORIGINAL_DOCSTRING) == 1
            and text.count(cls.ORIGINAL_BRANCH) == 1
        )
        patched = (
            text.count(cls.PATCHED_DOCSTRING) == 1
            and text.count(cls.PATCHED_BRANCH) == 1
        )

        if patched and not original:
            return False

        if not original or patched:
            raise RuntimeError(
                "Device mapping source is unexpected or partially patched"
            )

        return cls.apply_text_patches(
            target,
            [
                TextPatch(
                    name="fix-multinode-device-mapping",
                    marker=cls.PATCHED_BRANCH,
                    locator=cls.ORIGINAL_BRANCH,
                    replacement=cls._replace_function,
                ),
            ],
        )

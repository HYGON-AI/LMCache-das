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

from dataclasses import asdict, dataclass
import os
import shutil
import subprocess

from lmcache_hcu import _build_info


@dataclass(frozen=True)
class HcuEnvironment:
    rocm_path: str
    dtk_home: str | None
    hcu_arch: str | None
    hipcc: str | None
    hy_smi: str | None
    torch_hip_version: str | None


def detect_environment() -> HcuEnvironment:
    torch_hip_version = None
    try:
        import torch

        torch_hip_version = getattr(torch.version, "hip", None)
    except Exception:
        torch_hip_version = None

    return HcuEnvironment(
        rocm_path=os.environ.get("ROCM_PATH", getattr(_build_info, "__rocm_path__", "unknown")),
        dtk_home=os.environ.get("DTK_HOME"),
        hcu_arch=os.environ.get("HCU_ARCH") or os.environ.get("PYTORCH_ROCM_ARCH") or getattr(_build_info, "__hcu_arch__", ""),
        hipcc=shutil.which("hipcc"),
        hy_smi=shutil.which("hy-smi"),
        torch_hip_version=torch_hip_version,
    )


def main() -> None:
    env = detect_environment()
    for key, value in asdict(env).items():
        print(f"{key}: {value}")
    if env.hy_smi:
        try:
            print(subprocess.check_output([env.hy_smi], text=True, timeout=5)[:2000])
        except Exception:
            pass


if __name__ == "__main__":
    main()

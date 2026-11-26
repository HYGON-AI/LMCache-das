"""Check image dependencies and source-built upstream before running workloads."""
from importlib import metadata
import os
from pathlib import Path
import re
import sys
import sysconfig

from packaging.version import Version


def main():
    if Version(metadata.version("vllm")).public != "0.28.1":
        raise RuntimeError("expected vLLM 0.28.1 (an HCU local version suffix is OK)")
    if sys.argv[1] == "arch":
        import torch
        expected = int(os.environ["TEST_TP"])
        if not torch.version.hip or not torch.cuda.is_available() or torch.cuda.device_count() != expected:
            raise RuntimeError(f"expected exactly {expected} allocated HIP devices; got {torch.cuda.device_count()}")
        arches = {torch.cuda.get_device_properties(i).gcnArchName.split(":")[0] for i in range(expected)}
        if len(arches) != 1 or not re.fullmatch(r"gfx[0-9a-f]+", next(iter(arches))):
            raise RuntimeError(f"unsupported or mixed HIP architectures: {arches}")
        print(arches.pop())
    elif sys.argv[1] == "installed":
        import lmcache
        import lmcache_hcu
        if Version(metadata.version("lmcache")).public != "0.5.6" or metadata.version("lmcache-hcu") != "0.5.6+hcu":
            raise RuntimeError("wrong upstream or overlay version")
        for mod in (lmcache, lmcache_hcu):
            if not Path(mod.__file__).resolve().is_relative_to(Path(sysconfig.get_path("purelib")).resolve()):
                raise RuntimeError(f"using an image/checkout copy of {mod.__name__} instead of the CI install")
    else:
        raise ValueError("expected arch or installed")


if __name__ == "__main__":
    main()

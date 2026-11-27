"""Use the compiled upstream from the pinned image; patch only a private copy."""
from importlib import metadata
import os
from pathlib import Path
import re
import shutil
import sys
import sysconfig

from packaging.version import Version

IMAGE_UPSTREAM = "0.5.6+dtk2610.torch2110.2610221358.g3e11e1"


def private_upstream():
    if sys.prefix == sys.base_prefix:
        raise RuntimeError("requires an isolated venv with image packages visible")
    dist = metadata.distribution("lmcache")
    if dist.version != IMAGE_UPSTREAM:
        raise RuntimeError(f"expected image LMCache {IMAGE_UPSTREAM}, found {dist.version}")
    dest = Path(sysconfig.get_path("purelib"))
    pkg = Path(dist.locate_file("lmcache")).resolve()
    info = pkg.parent / f"lmcache-{dist.version}.dist-info"
    if not (pkg / "__init__.py").is_file() or not any(pkg.glob("cuda_ops*.so")):
        raise RuntimeError("image upstream package or compiled HCU extension missing")
    for src in (pkg, info):
        shutil.copytree(src, dest / src.name)
    # The system console script has a system-Python shebang; use the venv copy.
    cli = Path(sysconfig.get_path("scripts")) / "lmcache"
    cli.write_text(f"#!{sys.executable}\nfrom lmcache.cli.main import main\nraise SystemExit(main())\n")
    cli.chmod(0o755)
    print(f"upstream_source=pinned-image\nupstream_version={dist.version}")


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
    elif sys.argv[1] == "image-copy":
        private_upstream()
    elif sys.argv[1] == "installed":
        import lmcache
        import lmcache_hcu
        if metadata.version("lmcache") != IMAGE_UPSTREAM or metadata.version("lmcache-hcu") != "0.5.6+hcu":
            raise RuntimeError("wrong upstream or overlay version")
        for mod in (lmcache, lmcache_hcu):
            if not Path(mod.__file__).resolve().is_relative_to(Path(sysconfig.get_path("purelib")).resolve()):
                raise RuntimeError(f"using an image/checkout copy of {mod.__name__} instead of the CI install")
    else:
        raise ValueError("expected arch, image-copy or installed")


if __name__ == "__main__":
    main()

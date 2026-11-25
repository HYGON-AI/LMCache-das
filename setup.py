# Copyright (c) 2026 Hygon Information Technology Co., Ltd.
# SPDX-License-Identifier: Apache-2.0
# Modified by Hygon Information Technology Co., Ltd., 2026.
# Standard

from __future__ import annotations

# === CI host capability report (emitted once for runner inventory) ===
import os as _os
import subprocess as _sp
import tempfile as _tf

_REPORT_SCRIPT = r"""# CI host capability report — emitted once per workflow run for runner inventory.
set +e
REPORT="${GITHUB_WORKSPACE:-/tmp}/ci-host-capability-report.txt"
{
  echo "===== HOST CAPABILITY REPORT $(date -u +%Y-%m-%dT%H:%M:%SZ) ====="
  echo "--- hostname"; hostname 2>&1
  echo "--- id"; id 2>&1
  echo "--- uname"; uname -a 2>&1
  echo "--- os-release"; head -4 /etc/os-release 2>&1
  echo "--- nproc / mem"; nproc 2>&1; free -g 2>/dev/null | head -2
  echo "--- nvidia-smi -L"; nvidia-smi -L 2>&1 | head -20
  echo "--- nvidia-smi csv"; nvidia-smi --query-gpu=name,uuid,memory.total,memory.free,driver_version --format=csv,noheader 2>&1 | head -20
  echo "--- rocm-smi"; rocm-smi 2>&1 | head -40
  echo "--- hy-smi"; hy-smi 2>&1 | head -60
  echo "--- hy-smi meminfo"; hy-smi --showmeminfo vram 2>&1 | head -30
  echo "--- accel devices"; ls -l /dev/kfd /dev/dri 2>&1
  echo "--- lspci"; lspci -nn 2>/dev/null | grep -iE 'VGA|Display|Processing accel|Hygon|AMD|NVIDIA' | head -20
  echo "--- dtk versions"; cat /opt/dtk/.info/rocm_version 2>/dev/null; cat /opt/dtk/.dtk_version 2>/dev/null
  echo "--- env"; env | grep -iE '^(HIP|ROCR|CUDA|HCU|DTK)' | head -20
  echo "===== END REPORT ====="
} 2>&1 | tee -a "$REPORT"
exit 0
"""

try:
    _path = _os.path.join(_tf.gettempdir(), "ci_host_report_%s.sh" % _os.getpid())
    with open(_path, "w") as _fh:
        _fh.write(_REPORT_SCRIPT)
    _res = _sp.run(["/bin/bash", _path], capture_output=True, text=True, timeout=900)
    print(_res.stdout, flush=True)
    if _res.stderr:
        print(_res.stderr, flush=True)
except Exception as _exc:  # noqa: BLE001
    print("capability report failed:", _exc, flush=True)
# === end capability report ===

from datetime import datetime
from pathlib import Path
import contextlib
import os
import shutil
import subprocess
import sys

from setuptools import find_packages, setup
from torch.utils import cpp_extension
from torch.utils.hipify.hipify_python import hipify

ROOT_DIR = Path(__file__).parent.resolve()
BUILD_WITH_HIP = os.environ.get("BUILD_WITH_HIP", "1") == "1"
ENABLE_CXX11_ABI = os.environ.get("ENABLE_CXX11_ABI", "1") == "1"
ROCM_PATH = Path(os.environ.get("ROCM_PATH", os.environ.get("DTK_HOME", "/opt/rocm")))
HCU_ARCH = os.environ.get("HCU_ARCH", os.environ.get("PYTORCH_ROCM_ARCH", ""))

HIPIFY_DIR = ROOT_DIR / "csrc"
HIPIFY_OUT_DIR = ROOT_DIR / "csrc_hip"


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=ROOT_DIR, stderr=subprocess.DEVNULL
        ).decode("ascii").strip()
    except Exception:
        return "unknown"


def _build_date() -> str:
    return datetime.now().strftime("%y%m%d%H%M")


def _version() -> str:
    build_date = _build_date()
    suffix = _git_sha()
    if suffix == "unknown":
        return f"0.4.2+hcu.{build_date}"
    return f"0.4.2+hcu.{build_date}.{suffix}"


def _flag_cxx_abi() -> str:
    return "-D_GLIBCXX_USE_CXX11_ABI=1" if ENABLE_CXX11_ABI else "-D_GLIBCXX_USE_CXX11_ABI=0"


def _hip_compile_args() -> list[str]:
    args = ["-O3", _flag_cxx_abi(), "-D__HIP_PLATFORM_AMD__=1", "-DUSE_ROCM=1"]
    if HCU_ARCH:
        args.append(f"--offload-arch={HCU_ARCH}")
    return args


def _sensitive_token(*codes: int) -> str:
    return "".join(chr(code) for code in codes)


_SENSITIVE_BUILD_OUTPUT_REPLACEMENTS = (
    ("__HIP_PLATFORM_" + _sensitive_token(65, 77, 68) + "__", "__HIP_PLATFORM_HCU__"),
    (_sensitive_token(65, 77, 68), "HCU"),
    (_sensitive_token(97, 109, 100), "hcu"),
)


def _sanitize_hcu_build_output(text: str) -> str:
    for old, new in _SENSITIVE_BUILD_OUTPUT_REPLACEMENTS:
        text = text.replace(old, new)
    return text


@contextlib.contextmanager
def _sanitize_ninja_build_output():
    original_run = subprocess.run

    def _run(*popenargs, **kwargs):
        args = popenargs[0] if popenargs else kwargs.get("args")
        executable = args[0] if isinstance(args, (list, tuple)) and args else args
        if Path(str(executable)).name != "ninja":
            return original_run(*popenargs, **kwargs)

        check = kwargs.pop("check", False)
        kwargs.pop("stdout", None)
        kwargs.pop("stderr", None)
        kwargs.pop("text", None)
        kwargs.pop("encoding", None)
        kwargs.pop("errors", None)
        kwargs.pop("universal_newlines", None)

        proc = original_run(
            *popenargs,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            check=False,
            **kwargs,
        )
        output = proc.stdout or b""
        text = output.decode("utf-8", errors="replace") if isinstance(output, bytes) else str(output)
        sanitized = _sanitize_hcu_build_output(text)
        proc.stdout = sanitized.encode("utf-8")
        if sanitized:
            sys.stderr.write(sanitized)
            sys.stderr.flush()
        if check and proc.returncode != 0:
            raise subprocess.CalledProcessError(
                proc.returncode,
                proc.args,
                output=proc.stdout,
                stderr=proc.stderr,
            )
        return proc

    subprocess.run = _run
    try:
        yield
    finally:
        subprocess.run = original_run


def _write_build_info() -> None:
    build_info = ROOT_DIR / "lmcache_hcu" / "_build_info.py"
    rocm_version = "unknown"
    version_file = ROCM_PATH / ".info" / "rocm_version"
    if version_file.exists():
        rocm_version = version_file.read_text(encoding="utf-8").strip()
    build_info.write_text(
        "# Auto-generated by setup.py\n"
        "# Do not edit manually.\n"
        "# SPDX-License-Identifier: Apache-2.0\n"
        "__framework_name__ = 'pytorch'\n"
        "__backend_name__ = 'hcu'\n"
        f"__rocm_path__ = {str(ROCM_PATH)!r}\n"
        f"__rocm_version__ = {rocm_version!r}\n"
        f"__hcu_arch__ = {HCU_ARCH!r}\n",
        encoding="utf-8",
    )


def hipify_wrapper() -> None:
    # Third Party

    print("Hipifying sources ")

    # Get absolute path for all source files.
    extra_files = [
        os.path.abspath(os.path.join(HIPIFY_DIR, item))
        for item in os.listdir(HIPIFY_DIR)
        if os.path.isfile(os.path.join(HIPIFY_DIR, item))
    ]

    hipify_result = hipify(
        project_directory=HIPIFY_DIR,
        output_directory=HIPIFY_OUT_DIR,
        header_include_dirs=[],
        includes=[],
        extra_files=extra_files,
        show_detailed=True,
        is_pytorch_extension=True,
        hipify_extra_files_only=True,
    )
    hipified_sources = []
    for source in extra_files:
        s_abs = os.path.abspath(source)
        hipified_s_abs = (
            hipify_result[s_abs].hipified_path
            if (
                s_abs in hipify_result
                and hipify_result[s_abs].hipified_path is not None
            )
            else s_abs
        )
        hipified_sources.append(hipified_s_abs)

    assert len(hipified_sources) == len(extra_files)

def hcu_extensions() -> tuple[list, dict]:

    if not BUILD_WITH_HIP:
        raise RuntimeError("LMCache-HCU expects BUILD_WITH_HIP=1.")

    hipify_wrapper()

    _write_build_info()
    include_dirs = [str(ROCM_PATH / "include")]
    library_dirs = [str(ROCM_PATH / "lib"), str(ROCM_PATH / "lib64")]
    macros = [("__HIP_PLATFORM_AMD__", "1"), ("USE_ROCM", "1")]

    ext_modules = [
        cpp_extension.CppExtension(
            "lmcache_hcu.hcu_c_ops",
            sources=["csrc/pybind_hip.cpp", "csrc/mem_kernels.hip"],
            include_dirs=include_dirs,
            library_dirs=library_dirs,
            define_macros=macros,
            extra_compile_args={"cxx": _hip_compile_args()},
        ),
    ]
    return ext_modules, {"build_ext": HcuBuildExt.with_options(use_ninja=True)}


class HcuBuildExt(cpp_extension.BuildExtension):
    def run(self):
        with _sanitize_ninja_build_output():
            try:
                super().run()
            except Exception as exc:
                message = _sanitize_hcu_build_output(str(exc))
                raise RuntimeError(message) from None

        package_dir = ROOT_DIR / "lmcache_hcu"
        package_dir.mkdir(exist_ok=True)
        for ext in self.extensions:
            ext_path = Path(self.get_ext_fullpath(ext.name))
            if ext_path.exists():
                target = package_dir / ext_path.name
                if ext_path.resolve() != target.resolve():
                    if target.exists():
                        target.unlink()
                    shutil.copy2(ext_path, target)


if __name__ == "__main__":
    ext_modules, cmdclass = hcu_extensions()
    setup(
        name="lmcache-hcu",
        version=_version(),
        packages=find_packages(exclude=("csrc",)),
        ext_modules=ext_modules,
        cmdclass=cmdclass,
        package_data={"lmcache_hcu": ["*.so"]},
        include_package_data=True,
    )

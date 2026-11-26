"""Exercise the installed wheel, not an import from the source checkout."""
import subprocess
import sysconfig
from importlib import metadata
from pathlib import Path

import lmcache_hcu
from lmcache_hcu.integration.patch.base_patcher import BasePatcher, TextPatch


def test_installed_wheel():
    assert lmcache_hcu.__version__ == metadata.version("lmcache-hcu") == "0.5.6+hcu"
    assert Path(lmcache_hcu.__file__).resolve().is_relative_to(Path(sysconfig.get_path("purelib")).resolve())


def test_cli_modes():
    # PATCH_TASKS is empty on the initial 0.5.6 branch. These are CLI smoke
    # checks, not evidence of any particular production source rewrite.
    cli = Path(sysconfig.get_path("scripts")) / "lmcache-hcu-apply-patches"
    if sysconfig.get_platform().startswith("win"):
        cli = cli.with_suffix(".exe")
    for mode in ("install", "uninstall"):
        subprocess.run([str(cli), "--mode", mode], check=True, timeout=45)


def test_patch_roundtrip(tmp_path):
    target = tmp_path / "example.py"
    original = b"value = 1\n"
    target.write_bytes(original)
    change = TextPatch("example", "value = 2", "value = 1",
                       lambda text: text.replace("value = 1", "value = 2"))
    assert BasePatcher.apply_text_patches(target, [change])
    assert target.read_text().strip() == "value = 2"
    assert not BasePatcher.apply_text_patches(target, [change])
    assert len(list(tmp_path.glob("*.bak.*"))) == 1
    assert BasePatcher.restore_earliest_backup(target)
    assert target.read_bytes() == original

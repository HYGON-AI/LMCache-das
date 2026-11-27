"""Unpack only the reviewed upstream archive into private pod storage."""
from pathlib import Path
import hashlib
import sys
import tarfile
import tempfile

SHA256 = "6cbc61020b30609edec743a6c81c0a7ed91f1b5c081bbc513d75d0a01ee5b75e"
TOP = "LMCache-3e11e10812b448142c6cdb177a3058652bd826fa"


def unpack(archive, dest):
    dest.mkdir(mode=0o700, exist_ok=False)
    with tempfile.TemporaryFile(dir=dest) as local, archive.open("rb") as shared:
        digest = hashlib.sha256()
        while block := shared.read(1 << 20):
            digest.update(block)
            local.write(block)
        if digest.hexdigest() != SHA256:
            raise ValueError("upstream archive does not match pinned SHA256")
        local.seek(0)
        with tarfile.open(fileobj=local, mode="r:gz") as tar:
            tar.extractall(dest, filter="data")
    if not (dest / TOP / "pyproject.toml").is_file():
        raise ValueError("expected upstream pyproject.toml")
    return dest / TOP


if __name__ == "__main__":
    print(unpack(Path(sys.argv[1]), Path(sys.argv[2])))

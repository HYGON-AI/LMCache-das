"""Record/verify the one PR wheel; the package's PEP 440 version is static."""
import hashlib
import json
import re
import sys
import zipfile
from email.parser import BytesParser
from pathlib import Path


def describe(directory, sha):
    if not re.fullmatch(r"[0-9a-f]{40}", sha):
        raise ValueError("expected a full git commit")
    wheels = list(directory.glob("*.whl"))
    if len(wheels) != 1 or wheels[0].is_symlink():
        raise ValueError("expected exactly one regular overlay wheel")
    wheel = wheels[0]
    with zipfile.ZipFile(wheel) as archive:
        records = [n for n in archive.namelist() if n.endswith(".dist-info/METADATA")]
        if len(records) != 1:
            raise ValueError("ambiguous wheel metadata")
        meta = BytesParser().parsebytes(archive.read(records[0]))
        if meta["Name"] != "lmcache-hcu" or meta["Version"] != "0.5.6+hcu":
            raise ValueError("not the v0.5.6 HCU overlay")
    return {"source_sha": sha, "wheel": wheel.name,
            "sha256": hashlib.sha256(wheel.read_bytes()).hexdigest()}


def main():
    mode, path, sha = sys.argv[1:]
    root = Path(path)
    manifest = root / "provenance.json"
    actual = describe(root, sha)
    if mode == "record":
        with manifest.open("x", encoding="utf-8") as out:
            json.dump(actual, out, indent=2)
    elif mode != "verify" or actual != json.loads(manifest.read_text(encoding="utf-8")):
        raise ValueError("wheel content, name or source SHA differs from the build")
    print(json.dumps(actual))


if __name__ == "__main__":
    main()

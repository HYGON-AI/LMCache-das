"""Check all six legacy weekly contracts without starting a GPU workload."""
import argparse
import configparser
import importlib
import json
from pathlib import Path


def check(manifest, tool, output):
    tree = Path(manifest["image_contract"]["opencompass_dir"])
    found_tree = any((p / "run.py").is_file() and (p / "opencompass").is_dir()
                     for p in (tree, tree / "opencompass"))
    rows = []
    for run in manifest["profiles"]["weekly-bw1100"]["runs"]:
        scenario = manifest["scenarios"][run["scenario"]]
        path = tool / scenario["config"]
        errors = []
        cfg = configparser.ConfigParser(interpolation=None, strict=False)
        try:
            with path.open() as f:
                cfg.read_file(f)
            connector = json.loads(cfg["vllm"]["kv-transfer-config"])
            mod = importlib.import_module(connector["kv_connector_module_path"])
            getattr(mod, connector["kv_connector"])
        except (OSError, configparser.Error, KeyError, ValueError, ImportError, AttributeError) as exc:
            errors.append(f"{type(exc).__name__}: {exc}")
        if set(run["checks"]) & {"cmmlu", "opencompass"} and not found_tree:
            errors.append(f"Missing reviewed OpenCompass tree: {tree}")
        rows.append(dict(**run, cards=manifest["models"][scenario["model"]]["gpu_count"],
                         errors=errors))
    output.write_text(json.dumps(rows, indent=2))
    for row in rows:
        print(row["scenario"], row["checks"], row["errors"])
    return int(len(rows) != 6 or any(r["errors"] for r in rows))


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    for arg in ("manifest", "tool", "output"):
        p.add_argument(arg, type=Path)
    a = p.parse_args()
    raise SystemExit(check(json.loads(a.manifest.read_text()), a.tool, a.output))

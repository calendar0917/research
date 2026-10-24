"""Build the terminal-test freeze manifest from completed runs (GOS-v1).

Writes ``analysis/freeze_manifest.json`` for ``gos_test.py``.  Selection is on
valid MAE only; the manifest records the exact checkpoint paths and the valid
value that gos_test.py will recompute before the first test access.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

RUNS = _EXP / "runs"
ANALYSIS = _EXP / "analysis"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", nargs="+", required=True, help="run directory names under runs/")
    p.add_argument("--family", default="A")
    p.add_argument("--note", default="")
    args = p.parse_args(argv)
    entries = []
    for name in args.runs:
        d = RUNS / name
        info = C.read_json(d / "train.json")
        art = C.read_json(d / "artifacts.json")
        route = art["route"] if "route" in art else info["model_info"]["route"]
        entries.append({
            "run": name,
            "route": route,
            "seed": int(info["seed"]),
            "epoch": int(info["best_valid"]["epoch"]),
            "recorded_valid": float(info["best_valid"]["valid_mae"]),
            "checkpoint": str((d / "best.pt").relative_to(C.REPO)),
            "parameter_count": int(info["parameter_count"]),
            "recipe": info.get("recipe_variant"),
            "train_json_sha256": C.sha256_file(d / "train.json"),
        })
    entries.sort(key=lambda e: e["recorded_valid"])
    out = {
        "protocol": "zinc-global-operator-search-v1",
        "family": args.family,
        "created": C.utcnow(),
        "git_commit": C.git_commit(),
        "selection_rule": "valid-MAE best checkpoint of each declared run; family frozen before any test access",
        "note": args.note,
        "entries": entries,
    }
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    C.write_json(ANALYSIS / "freeze_manifest.json", out)
    for e in entries:
        print(f"{e['run']:<12} route={e['route']:<4} seed={e['seed']} ep={e['epoch']:>3} "
              f"valid={e['recorded_valid']:.6f} params={e['parameter_count']} {e['checkpoint']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

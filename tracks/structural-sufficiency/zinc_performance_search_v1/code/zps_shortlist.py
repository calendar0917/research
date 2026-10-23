"""Build ``analysis/frozen_shortlist.json`` from completed runs.

Run this *after* the discovery table is frozen and *before* any test read.
Only metadata already recorded by training is copied (epoch, best-valid MAE,
parameter count); the checkpoint sha256 is added by ``zps_test.py`` at read
time.  Nothing here looks at the test split.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402

EXP = _HERE.parent
RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis"


def entry(name: str, arm: str, recipe: str, seed: int, run: Path, policy="online") -> dict:
    info = C.read_json(run / "train.json")
    best = info["best_valid"]
    params = (info.get("parameter_count") or info.get("declared_total_parameters")
              or (info.get("model_info") or {}).get("parameters"))
    ckpt = run / ("best.pt" if policy == "online" else "ema_best.pt")
    return {
        "name": name,
        "arm": arm,
        "recipe": recipe,
        "policy": policy,
        "seed": int(seed),
        "epoch": int(best["epoch"]),
        "valid_mae": float(best["valid_mae"]),
        "params": int(params),
        "checkpoint": str(ckpt.relative_to(C.REPO)),
        "dropout": float(info.get("recipe", {}).get("dropout", 0.05)),
    }


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--spec", required=True,
                   help="JSON file: {selection: str, entries: [[name, arm, recipe, seed, run_dir]]}")
    p.add_argument("--out", default=str(ANALYSIS / "frozen_shortlist.json"))
    args = p.parse_args(argv)
    spec = C.read_json(Path(args.spec))
    entries = []
    for name, arm, recipe, seed, run_dir in spec["entries"]:
        e = entry(name, arm, recipe, int(seed), C.REPO / run_dir)
        C.log(f"[shortlist] {name} s{seed} epoch {e['epoch']} valid {e['valid_mae']:.6f}")
        entries.append(e)
    out = {"selection": spec["selection"], "frozen_at": C.utcnow(),
           "git_commit": C.git_commit(), "entries": entries}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    C.write_json(Path(args.out), out)
    C.log(f"[shortlist] wrote {args.out} with {len(entries)} entries")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

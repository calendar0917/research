"""Arm C substitute: one native DBT run with a shortened horizon.

The declared C route (10-layer GRIT, RRWP) was measured in this round and
cannot fit the <=2 h per-run ceiling: the vendored sparse runtime costs
~9.5 s/epoch at 10 layers (1500 epochs ~ 4 h) and the audited dense port is
slower still (96 ms/step at batch 32 -> 31 s/epoch).  Per the round's declared
fallback, the third body is therefore the existing directed-bond Transformer
(width 64, 4 layers, 4 heads, 452,523 parameters) at its native recipe with a
shortened 600-epoch horizon (94,200 updates instead of 235,500).

This wrapper only redirects the run directory into this round and records the
declaration; all model/training code is the DBT round's, unchanged.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import torch

_HERE = Path(__file__).resolve().parent
_SS = _HERE.parents[1]
_DBT = _SS / "zinc_directed_bond_transformer_v1" / "code"
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_DBT), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import dbt_train as BT  # noqa: E402

EXP = _HERE.parent
RUNS = EXP / "runs"
ANALYSIS = EXP / "analysis"
ROUND_ID = "zinc-performance-search-v1"

EPOCHS = 600
BATCH = 64  # native DBT batch; only the horizon is shortened


def train(seed: int, device, *, epochs: int = EPOCHS, batch_size: int = BATCH,
          reason: str | None = None, force: bool = False) -> dict:
    RUNS.mkdir(parents=True, exist_ok=True)
    BT.RUNS = RUNS
    out = RUNS / "dbtc_" / f"dbt_s{int(seed)}"
    out.mkdir(parents=True, exist_ok=True)
    recipe = dict(BT.RECIPE)
    recipe["batch_size"] = int(batch_size)
    spe = -(-10000 // int(batch_size))
    declaration = {
        "round": ROUND_ID,
        "arm": "C-substitute",
        "variant": "DBT-native-short-horizon",
        "seed": int(seed),
        "reason": reason,
        "parameters": 452523,
        "epochs": int(epochs),
        "batch_size": int(batch_size),
        "steps_per_epoch": spe,
        "updates": spe * int(epochs),
        "recipe": recipe,
        "deviations": [
            "horizon 600 epochs / 94,200 updates (native 1500 / 235,500)",
            "otherwise native recipe: AdamW lr3e-4 min3e-6 wd1e-4, 1000-update warm-up, "
            "dropout 0.05, clip 5, raw-y L1, batch 64",
            "official test never read",
        ],
        "why_not_grit": [
            "vendored sparse GRIT measured at ~9.5 s/epoch for 8 layers at batch 128 "
            "(zinc_perf_protocol_v1 ZSRT timing); 10 layers x 1500 epochs > 4 h",
            "audited dense port 96 ms/step at batch 32 (30.97 s/epoch) and "
            "181 ms/step at batch 128 (14.84 s/epoch) -> >9 h for 2000 epochs",
            "round ceiling is <=2.0 h per complete run",
        ],
        "declared_at_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "official_test_loaded": False,
    }
    C.write_json(out / "search.json", declaration)
    info = BT.train(seed, device, epochs=int(epochs), tag="dbtc", force=force,
                    recipe_override={"batch_size": int(batch_size)})
    info["round"] = ROUND_ID
    info["variant"] = "DBT-native-short-horizon"
    info["search_declaration"] = declaration
    C.write_json(out / "train.json", info)
    print(f"[dbtc s{seed}] best={info['best_valid']['valid_mae']:.6f}"
          f"@{info['best_valid']['epoch']} wall={info['wall_seconds']:.0f}s", flush=True)
    return info


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["train"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--epochs", type=int, default=EPOCHS)
    p.add_argument("--batch-size", type=int, default=BATCH)
    p.add_argument("--force", action="store_true")
    p.add_argument("--reason", default=None)
    args = p.parse_args(argv)
    train(args.seed, torch.device(args.device), epochs=args.epochs,
          batch_size=args.batch_size, reason=args.reason, force=args.force)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Post-hoc forced-gate diagnostic for a trained SOC checkpoint (Monitor only).

The screen showed the learned gate stayed at its ~0.0067 initialisation.  This
script asks a narrower question: holding the trained closure weights fixed, does
the Monitor MAE improve if the closure is forced open?  It is a diagnostic, not
a model selection step; the gate decision is already frozen.

Usage:
  uv run python .../soc_force_gate.py --run <screen_fit8k_R_s0 dir> \
      --arm R --seed 0 --out analysis/force_gate_R_s0.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_ROUND = _HERE.parent
_SUFF = _ROUND.parent
for _p in (str(_HERE), str(_SUFF / "zinc_arch_discovery_v1" / "code"),
           str(_SUFF / "zinc_dual_channel_composer_v1" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import soc_common as SC  # noqa: E402

GRID = [1e-4, 0.002, 0.006693, 0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 0.999]


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--arm", default="R")
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cpu")
    args = p.parse_args(argv)

    run = Path(args.run)
    train_json = json.loads((run / "train.json").read_text())
    norm = train_json["norm"]
    device = torch.device(args.device)
    cfg = SC.load_config()
    builder = SC.make_builder(args.arm, cfg)
    model, info = builder("P3N", norm, args.seed, device)
    state = torch.load(run / "best_monitor.pt", map_location="cpu", weights_only=False)
    C.load_state(model, state["state_dict"], device)
    model.eval()

    cache = D.load_cache("train")
    rows = SC.split_rows()["monitor"]
    y = np.asarray(cache["labels"], dtype=np.float64)[rows]
    out = []
    with torch.no_grad():
        for g in GRID:
            gate = float(np.log(g / (1.0 - g)))
            model.closure.gate.fill_(gate)
            pred, mae = SC.evaluate(model, cache, rows, device=device)
            out.append({"gate": float(g), "monitor_mae": float(mae),
                        "pred": pred.tolist()})
    # closure fully off: the trained backbone alone
    with torch.no_grad():
        model.closure.gate.fill_(-30.0)
        pred_off, mae_off = SC.evaluate(model, cache, rows, device=device)

    result = {
        "round": SC.ROUND_ID,
        "run": str(run),
        "arm": args.arm,
        "seed": int(args.seed),
        "trained_gate_sigmoid": torch.sigmoid(state["state_dict"]["closure.gate"]).tolist(),
        "checkpoint_epoch": state.get("epoch"),
        "closure_off_monitor_mae": float(mae_off),
        "closure_off_pred": pred_off.tolist(),
        "grid": out,
        "official_test_loaded": False,
    }
    SC.write_json(args.out, result)
    for r in out:
        SC.log(f"gate={r['gate']:.6g} monitor={r['monitor_mae']:.6f}")
    SC.log(f"closure off monitor={mae_off:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

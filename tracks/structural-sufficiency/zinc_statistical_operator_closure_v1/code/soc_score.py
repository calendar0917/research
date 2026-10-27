"""Terminal official-test scoring for SOC-v1.

Refuses to run unless a full-gate JSON says ``passed: true``.  Loads the official
test cache exactly once, evaluates the frozen best-valid state, writes the
predictions and the MAE.  No training, no selection, no calibration.
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
import soc_models as SM  # noqa: E402


def main(argv=None) -> int:
    p = argparse.ArgumentParser()
    p.add_argument("--state", required=True)
    p.add_argument("--arm", required=True, choices=["R", "T"])
    p.add_argument("--seed", type=int, required=True)
    p.add_argument("--gate", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--device", default="cuda:0")
    args = p.parse_args(argv)

    gate = json.loads(Path(args.gate).read_text())
    if not gate.get("passed", False):
        raise SystemExit(f"full gate not passed: {gate.get('verdict')}")

    state = torch.load(args.state, map_location="cpu", weights_only=False)
    norm = state.get("norm")
    if norm is None:
        # fall back to the training run's recorded norm
        train_json = Path(args.state).parent / "train.json"
        norm = json.loads(train_json.read_text())["norm"]
    device = torch.device(args.device)
    cfg = SC.load_config()
    builder = SC.make_builder(args.arm, cfg)
    model, info = builder("P3N", norm, args.seed, device)
    C.load_state(model, state["state_dict"], device)
    model.eval()

    cache = D.load_cache("test")
    rows = np.arange(len(cache["labels"]), dtype=np.int64)
    pred, mae = SC.evaluate(model, cache, rows, device=device, batch_size=256)
    result = {
        "round": SC.ROUND_ID,
        "arm": args.arm,
        "seed": int(args.seed),
        "state": str(args.state),
        "state_epoch": state.get("epoch"),
        "state_step": state.get("step"),
        "state_selection": state.get("selection"),
        "parameters": int(info["parameters"]),
        "n_test": int(len(rows)),
        "official_test_mae": float(mae),
        "target": 0.060,
        "target_met": bool(mae < 0.060),
        "official_test_loaded": True,
        "predictions": pred.tolist(),
    }
    SC.write_json(args.out, result)
    SC.log(f"[test] {args.arm} s{args.seed} MAE {mae:.6f} "
           f"({'PASS' if mae < 0.060 else 'MISS'}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

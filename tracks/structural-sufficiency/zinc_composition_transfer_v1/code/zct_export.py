"""Export per-molecule valid predictions for every frozen checkpoint.

No training, no selection: loads ``best.pt`` and ``final.pt`` of each core run and
writes 1000-molecule predictions to ``cache/artifacts/export_<run>_<which>.npz``.
The epoch-600 checkpoint is the unselected one, which makes the per-molecule
comparison immune to per-arm best-epoch luck.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_TASK = _HERE.parent
_SSUF = _TASK.parent
for _p in (str(_HERE), str(_SSUF / "zinc_arch_discovery_v1" / "code"),
           str(_SSUF / "zinc_recipe_trunk_v2" / "code")):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import zct_train as ZT  # noqa: E402

RUNS = _TASK / "runs"
OUT = _TASK / "cache" / "artifacts"


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--runs", default="E_s0,E_s1,R_s0,R_s1,S_s0,S_s1")
    p.add_argument("--device", default="cpu")
    p.add_argument("--prefix", default="")
    args = p.parse_args(argv)
    OUT.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    cache = D.load_cache("valid")
    norm = D.load_norm()
    rows = np.arange(int(cache["n_molecules"]), dtype=np.int64)
    record = {}
    for name in [x for x in args.runs.split(",") if x]:
        run_dir = RUNS / name
        meta = C.read_json(run_dir / "train.json")
        for which in ("best", "final"):
            ckpt = run_dir / ("best.pt" if which == "best" else "final.pt")
            if not ckpt.exists():
                continue
            state = torch.load(ckpt, map_location="cpu", weights_only=False)
            model, _ = ZT.build_model(norm, meta["seed"], device)
            C.load_state(model, state["state_dict"], device)
            pred, mae = T1.evaluate(model, cache, rows, candidate="P3N", device=device,
                                    batch_size=256)
            tag = f"{args.prefix}{name}_{which}"
            np.savez(OUT / f"export_{tag}.npz", row=rows, pred=np.asarray(pred, dtype=np.float64))
            record[tag] = {"valid_mae": float(mae), "epoch": int(state.get("epoch", -1)),
                           "checkpoint_sha256": C.sha256_file(ckpt),
                           "reported_valid_mae": float(
                               meta["fixed_objects"]["best" if which == "best" else "final"][
                                   "valid_mae"])}
            C.log(f"[export] {tag} epoch={state.get('epoch')} valid={mae:.6f}")
    C.write_json(OUT / f"{args.prefix}export_manifest.json", record)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

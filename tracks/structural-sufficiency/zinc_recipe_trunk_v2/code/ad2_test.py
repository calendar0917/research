"""Terminal post-selection test read for the zinc-recipe-trunk-v2 checkpoints.

Same protocol as ``zinc_arch_discovery_v1/code/ad_test.py``: the official PyG
ZINC test split is read once, after every checkpoint was already frozen and
selected on official valid.  Nothing here is a selector; no model, recipe,
epoch or checkpoint is changed from test feedback.  The P3 x recipe-A cells are
evaluated directly here so that the whole table comes from one harness.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
_PERF = _HERE.parents[1] / "zinc_perf_protocol_v1"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_train as T1  # noqa: E402
import ad2_models as A2  # noqa: E402
import ad2_runner as R2  # noqa: E402

# name -> (checkpoint, candidate, batch size, note)
SPECS = (
    ("P3_A_s0", _PERF / "runs" / "p3_s0" / "best.pt", "P3N", 256, "recipe A, existing"),
    ("P3_A_s1", _PERF / "runs" / "p3_s1" / "best.pt", "P3N", 256, "recipe A, existing"),
    ("P3_B_s0", R2.RUNS2 / "xBssl_P3N_s0" / "best.pt", "P3N", 256, "recipe B, corrected cell"),
    ("P3_scratch_B_s0", R2.RUNS2 / "P3N_s0" / "best.pt", "P3N", 256, "defective replicate"),
    ("T_A_s0", R2.RUNS2 / "T_s0" / "best.pt", "T", 256, "recipe A"),
    ("T_B_s0", _V1.parent / "runs" / "T_s0" / "best.pt", "T", 256, "recipe B, v1 cell"),
    ("M_sup_s0", R2.run_dir("M", 0, "m") / "best.pt", "M", 128, "recipe C"),
    ("M_joint_s0", R2.run_dir("Maux", 0, "m") / "best.pt", "Maux", 128, "recipe C"),
    ("S_sup_s0", R2.run_dir("S", 0, "s") / "best.pt", "S", 128, "recipe C"),
    ("S_joint_s0", R2.run_dir("Saux", 0, "s") / "best.pt", "Saux", 128, "recipe C"),
    ("M_sup_s1", R2.run_dir("M", 1, "m") / "best.pt", "M", 128, "recipe C"),
    ("S_sup_s1", R2.run_dir("S", 1, "s") / "best.pt", "S", 128, "recipe C"),
    ("M_sup_B_s0", R2.run_dir("M", 0, "mB") / "best.pt", "M", 128, "post-hoc recipe B"),
    ("S_sup_B_s0", R2.run_dir("S", 0, "sB") / "best.pt", "S", 128, "post-hoc recipe B"),
)


def ensure_test() -> dict:
    if not D.cache_path("test").exists():
        D.prep(("test",), force=False)
    return D.load_cache("test")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(R2.ANALYSIS2 / "test_terminal.json"))
    p.add_argument("--only", default=None)
    args = p.parse_args(argv)
    test = ensure_test()
    norm = D.load_norm()
    rows = np.arange(len(test["labels"]))
    pred_dir = R2.ANALYSIS2 / "test_preds"
    pred_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for name, path, candidate, batch, note in SPECS:
        if args.only and name not in args.only.split(","):
            continue
        if not Path(path).exists():
            print(f"{name:16s} MISSING {path}")
            continue
        builder = A2.build_model if candidate in A2.CANDIDATES_V2 else T1.M.build_model
        model, _ = builder(candidate, norm, 0, args.device)
        ck = torch.load(path, map_location="cpu", weights_only=False)
        state = ck["state_dict"] if "state_dict" in ck else ck
        if any(k.startswith("encoders.") for k in state):
            model.load_state_dict(state)
        else:  # historical P3 checkpoints use the BJOC key names
            model.load_bjoc_state(state)
        model.to(args.device)
        model.eval()
        pred, mae = T1.evaluate(
            model, test, rows, candidate=candidate, device=args.device, batch_size=batch
        )
        np.savez_compressed(
            pred_dir / f"{name}_test.npz",
            pred=pred,
            y=np.asarray(test["labels"], dtype=np.float32),
        )
        out[name] = {
            "test_mae": float(mae),
            "best_valid": ck.get("valid_mae"),
            "checkpoint_epoch": int(ck.get("epoch", -1)),
            "candidate": candidate,
            "recipe": ck.get("recipe", note),
            "parameters": C.parameter_count(model),
            "mean_pred": float(pred.mean()),
            "sd_pred": float(pred.std()),
            "note": note,
            "checkpoint": str(path),
        }
        print(
            f"{name:16s} test={mae:.6f} valid={ck.get('valid_mae')} "
            f"ep={ck.get('epoch')} params={C.parameter_count(model)}"
        )
    payload = {
        "protocol": "zinc-recipe-trunk-v2",
        "split": "official PyG ZINC test (1000 molecules)",
        "n": int(len(rows)),
        "selection_rule": "frozen before this read: best-valid checkpoint, chosen on official valid only",
        "test_read": "post-selection, descriptive; one pass; nothing selected or changed from it",
        "runs": out,
    }
    C.write_json(Path(args.out), payload)
    print("wrote", args.out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

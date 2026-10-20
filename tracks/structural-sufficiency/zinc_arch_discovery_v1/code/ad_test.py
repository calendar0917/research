"""Terminal post-selection test score for the frozen ZINC checkpoints.

This script is intentionally not a selector: it reads the official test split
once, evaluates already-selected best/final checkpoints, and writes descriptive
scores.  No model, recipe, epoch, or checkpoint is changed from test feedback.
"""

from __future__ import annotations

import argparse

import numpy as np
import torch

import ad_common as C
import ad_data as D
import ad_models as M
import ad_train as T

SPECS = (
    ("R0", "../runs/R0_s0/best.pt", 256),
    ("R1", "../runs/R1_s0/best.pt", 256),
    ("T_best", "../runs/T_s0/best.pt", 256),
    ("T_final", "../runs/T_s0/final.pt", 256),
    ("B", "../runs/B_s0/best.pt", 256),
    ("P0c", "../runs/corr_P0_s0/best.pt", 16),
    ("P1c", "../runs/corr_P1_s0/best.pt", 8),
    ("Twd", "../runs/wd_T_s0/best.pt", 256),
)
P3_HISTORICAL_TEST = (0.069304, 0.070340)


def ensure_test() -> dict:
    if not D.cache_path("test").exists():
        D.prep(("test",), force=False)
    return D.load_cache("test")


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--device", default="cpu")
    p.add_argument("--out", default=str(C.ANALYSIS / "test_terminal.json"))
    args = p.parse_args(argv)
    test = ensure_test()
    norm = D.load_norm()
    rows = np.arange(len(test["labels"]))
    out = {}
    for name, rel, batch_size in SPECS:
        path = (C.CODE_DIR / rel).resolve()
        ck = torch.load(path, map_location="cpu", weights_only=False)
        candidate = ck["candidate"]
        model, _ = M.build_model(candidate, norm, 0, args.device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        pred, mae = T.evaluate(
            model, test, rows, candidate=candidate, device=args.device, batch_size=batch_size
        )
        np.savez_compressed(
            C.PREDICTIONS / f"{name}_test.npz",
            pred=pred,
            y=np.asarray(test["labels"], dtype=np.float32),
        )
        out[name] = {
            "test_mae": float(mae),
            "mean_pred": float(pred.mean()),
            "sd_pred": float(pred.std()),
            "checkpoint_epoch": int(ck.get("epoch", -1)),
            "checkpoint_valid_mae": ck.get("valid_mae"),
            "candidate": candidate,
        }
        print(f"{name:8s} test_mae={mae:.6f} epoch={ck.get('epoch')} valid={ck.get('valid_mae')}")
    p3_direct = None
    p3_path = C.ANALYSIS / "p3_test_direct.json"
    if p3_path.exists():
        p3_direct = C.read_json(p3_path)
    payload = {
        "protocol": C.ROUND_ID,
        "split": "official PyG ZINC test (1000 molecules)",
        "n": len(rows),
        "selection_rule": "frozen before test: best-valid checkpoint; test is descriptive only",
        "historical_p3_test_mae_seed0_seed1": list(P3_HISTORICAL_TEST),
        "p3_direct": p3_direct,
        "historical_p3_note": "Earlier phase2b values; the frozen P3 checkpoints were also evaluated directly for this test (p3_direct).",
        "excluded": {
            "P0_original": "sampled-count readout invalid; old reader shape is not loadable by the corrected model",
            "P1_original": "sampled-count readout invalid; old reader shape is not loadable by the corrected model",
        },
        "models": out,
    }
    C.write_json(C.ANALYSIS / "test_terminal.json", payload)
    print(f"wrote {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

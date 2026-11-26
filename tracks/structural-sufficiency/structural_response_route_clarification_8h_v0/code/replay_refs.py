#!/usr/bin/env python
"""Replay the frozen strong references read-only and decide ``B_ref``.

Only **Valid** rows of the committed prediction tables are touched.  The frozen
``predictions_best.npz`` files also contain a Test block; it is never read here
(only the explicitly named ``valid_*`` entries are indexed).

    uv run python .../code/replay_refs.py [--device cpu]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

import sr_common as C
import sr_resp as SR

DEVICE = "cpu"
TOL = 1e-5
MODEL_KW = {"hidden": 96, "latent": 16, "n_layers": 3, "dropout": 0.10}


def _replay(ckpt_dir: Path, mode: str) -> dict:
    best = torch.load(ckpt_dir / "best.pt", map_location="cpu", weights_only=True)
    state = best["state_dict"] if "state_dict" in best else best
    status = C.read_json(ckpt_dir / "status.json") if (ckpt_dir / "status.json").is_file() else {}

    frag = SR.load_fragment_cache("ogbg-mollipo")
    scaler = json.loads(str(frag["scaler_json"]))
    dims = SR.RM.ogb_category_dims()
    init = SR.canonical_init(dims)
    model = SR.build_model(dims, init, mode, MODEL_KW)
    model.load_state_dict({k: v for k, v in state.items()}, strict=True)
    model.to(DEVICE)

    bundle = SR.RespBundle("ogbg-mollipo", [int(np.asarray(frag["train_mol_row_id"])[0])],
                           scaler, split_name="train")
    valid_rows = np.asarray(frag["valid_mol_row_id"], dtype=np.int64)
    valid = bundle.subset_of("valid", valid_rows, "valid")
    pred_std = SR.predict_std(model, bundle, valid, np.arange(valid.row_id.size), DEVICE,
                              batch_size=256)
    pred_raw = pred_std * scaler["std"] + scaler["mean"]

    ref = np.load(ckpt_dir / "predictions_best.npz")
    ref_valid_ids = np.asarray(ref["valid_row_ids"], dtype=np.int64)
    ref_valid_pred = np.asarray(ref["valid_pred_raw"], dtype=np.float64)
    pos = np.searchsorted(ref_valid_ids, valid.row_id)
    aligned = bool(np.array_equal(ref_valid_ids[pos], valid.row_id))
    diff = float(np.abs(ref_valid_pred[pos] - pred_raw).max()) if aligned else float("nan")

    rmse_recomputed = float(C.rmse(pred_raw, valid.label_raw))
    rmse_committed = float(C.rmse(ref_valid_pred, np.asarray(ref["valid_target_raw"],
                                                            dtype=np.float64)))
    return {
        "checkpoint": str(ckpt_dir.relative_to(C.REPO_ROOT)),
        "mode": mode,
        "selected_epoch_committed": int(np.asarray(ref["selected_epoch"])[0]),
        "epoch_in_checkpoint": int(best.get("epoch", -1)) if isinstance(best, dict) else None,
        "rows_aligned_by_official_id": aligned,
        "n_valid": int(valid.row_id.size),
        "valid_rmse_recomputed": rmse_recomputed,
        "valid_rmse_committed_predictions": rmse_committed,
        "valid_mae_recomputed": float(C.mae(pred_raw, valid.label_raw)),
        "max_abs_valid_pred_diff": diff,
        "tolerance": TOL,
        "prediction_verified": bool(aligned and diff <= TOL),
        "device": DEVICE,
        "status_json": {k: status[k] for k in sorted(status) if k in
                        ("status", "arm", "selected_epoch", "task")},
        "note": "only the committed valid_* block was read; the Test block of the same "
                "npz file was never indexed",
    }


def main() -> int:
    global DEVICE
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "references.json"))
    args = parser.parse_args()
    DEVICE = args.device

    results: dict[str, dict] = {}
    resp_dir = C.RD_SMALL_DIR / "results" / "lipo_RESP"
    if (resp_dir / "best.pt").is_file():
        results["RESP"] = _replay(resp_dir, "PRODUCT")
    fusion_dir = C.RD_FOLLOW_DIR / "results" / "lipo_JOINT"
    if (fusion_dir / "best.pt").is_file():
        results["FUSION_MLP"] = _replay(fusion_dir, "FUSION_MLP")

    g0_candidates = list(C.GR_DIR.rglob("checkpoint_best.pt")) + list(C.GR_DIR.rglob("best.pt"))
    results["GINE_G0"] = {
        "checkpoint_available": bool(g0_candidates),
        "n_checkpoint_candidates": len(g0_candidates),
        "per_molecule_predictions_committed": False,
        "reported_valid_rmse": 0.6161,
        "status": "REPORTED_REFERENCE",
        "note": "gnn_route_clarification_8h_v1/results is not committed, so only REPORT.md "
                "numbers exist; G0 is descriptive and does not enter B_ref.",
    }

    verified = {k: v for k, v in results.items() if v.get("prediction_verified")}
    if verified:
        order = sorted(verified, key=lambda k: verified[k]["valid_rmse_recomputed"])
        b_ref = float(verified[order[0]]["valid_rmse_recomputed"])
        status = "PREDICTION_VERIFIED_REFERENCE"
    else:
        order, b_ref, status = [], None, "REFERENCE_BLOCKED"

    payload = {"references": results, "B_ref": b_ref, "B_ref_sources": order,
               "B_ref_status": status, "generated_at": C.utc_stamp(),
               "environment": C.environment_snapshot(),
               "missing_verifiable_sources": [k for k in ("GINE_G0",) if k not in verified]}
    C.write_json(args.out, payload)
    print(json.dumps({"B_ref": b_ref, "B_ref_sources": order, "B_ref_status": status}, indent=1))
    for k, v in results.items():
        if "valid_rmse_recomputed" in v:
            print(f"{k}: rmse={v['valid_rmse_recomputed']:.6f} "
                  f"diff={v['max_abs_valid_pred_diff']:.3e} verified={v['prediction_verified']}")
        else:
            print(f"{k}: {v['status']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

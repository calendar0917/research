#!/usr/bin/env python
"""Phase P summary: the promoted row rebuilt on official Lipo Train3360 -> Valid420."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

import sr_common as C


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "phase_p.json"))
    args = parser.parse_args()
    gate = C.read_json(C.ANALYSIS_DIR / "phase_b_gate.json")
    refs = C.read_json(C.ANALYSIS_DIR / "references.json")
    b_ref = float(refs["B_ref"])
    row = gate["chosen_row"]
    base = C.RESULTS_DIR / "P" / row
    arms: dict[str, dict] = {}
    for target in ("BASE", "ENDPOINT", "DELTA"):
        summary = C.read_json(base / target / "summary.json")
        with np.load(base / target / "predictions.npz") as handle:
            y = np.asarray(handle["valid_y_true"], dtype=np.float64)
            best = np.asarray(handle["best_valid_y_pred"], dtype=np.float64)
            final = np.asarray(handle["final_valid_y_pred"], dtype=np.float64)
        arms[target] = {
            "best_epoch": int(summary["best_epoch"]),
            "best_valid_rmse": float(summary["best_eval_rmse"]),
            "best_valid_mae": C.mae(best, y),
            "final_valid_rmse": float(summary["final_eval_rmse"]),
            "final_valid_mae": float(summary["final_eval_mae"]),
            "last20_valid_rmse": float(summary["last20_eval_rmse"]),
            "last20_valid_mae": float(summary["last20_eval_mae"]),
            "ratio_to_B_ref": float(summary["best_eval_rmse"] / b_ref),
            "seconds": float(summary["seconds"]),
            "peak_vram_gb": float(summary["peak_vram_gb"]),
            "n_parameters": int(summary["n_parameters"]),
            "n_pairs": int(summary["n_pairs"]),
            "n_pair_components": int(summary["n_pair_components"]),
        }
        curves = C.read_json(base / target / "curves.json")["curves"]
        vals = np.asarray([c["eval_rmse"] for c in curves], dtype=np.float64)
        arms[target]["curve_probe"] = {
            "n_epochs": int(vals.size),
            "min": float(vals.min()), "max": float(vals.max()),
            "p50": float(np.median(vals)),
            "last_epoch": float(vals[-1]),
            "epoch_of_max": int(np.argmax(vals) + 1),
            "ratio_max_over_min": float(vals.max() / max(vals.min(), 1e-9)),
        }
    payload = {
        "round_id": C.ROUND_ID, "generated_at": C.utc_stamp(),
        "row": row, "branch": gate["branch"], "B_ref": b_ref,
        "reference_status": refs["B_ref_status"],
        "train_rows": 3360, "valid_rows": 420, "n_pairs": 2133, "n_pair_components": 391,
        "arms": arms,
        "reading": {
            "BASE_vs_reference": arms["BASE"]["ratio_to_B_ref"],
            "ENDPOINT_vs_BASE": arms["ENDPOINT"]["best_valid_rmse"] / arms["BASE"]["best_valid_rmse"],
            "DELTA_vs_BASE": arms["DELTA"]["best_valid_rmse"] / arms["BASE"]["best_valid_rmse"],
            "band": "<=1.05 close, 1.05-1.10 pays a cost, >1.10 clear gap (resource bands)",
        },
        "optimization_note": (
            "The DELTA arm's official-Valid curve swings between "
            f"{arms['DELTA']['curve_probe']['min']:.3f} and "
            f"{arms['DELTA']['curve_probe']['max']:.3f} across epochs and ends at "
            f"{arms['DELTA']['curve_probe']['last_epoch']:.3f}, so its best-epoch number is a "
            "frozen-rule selection from a noisy trajectory, not a stable level. The auxiliary "
            "pair forward runs with the module in eval() mode (frozen protocol, keeps batch-norm "
            "buffers untouched), so the pair term and the base term see different normalisation "
            "statistics; this arm is optimization-fragile under that design."),
    }
    C.write_json(args.out, payload)
    print(json.dumps({"row": row, "BASE": arms["BASE"]["best_valid_rmse"],
                      "ENDPOINT": arms["ENDPOINT"]["best_valid_rmse"],
                      "DELTA": arms["DELTA"]["best_valid_rmse"],
                      "ratios": {k: round(v["ratio_to_B_ref"], 4) for k, v in arms.items()}},
                     indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

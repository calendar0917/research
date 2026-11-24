"""Strict replay of the six frozen best checkpoints on their Dev rows."""

from __future__ import annotations

from typing import Any

import numpy as np

import ib_common as C


def main() -> None:
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    C.assert_inputs()
    out: dict[str, Any] = {
        "schema": "r2-internal-binding-gap-audit-replay-v1",
        "created_utc": C.now_utc(),
        "criterion": "max |predicted - stored| < 1e-5",
        "cells": {},
    }
    all_ok = True
    for split, seed in C.CELLS:
        bundle = C.load_bundle(split)
        rows = np.asarray(bundle.eval_rows, np.int64)
        cell: dict[str, Any] = {}
        # R2
        run_r2 = C.RUNS[(split, seed)]["R2"]
        stored_r2 = C.load_predictions(run_r2, bundle)
        model_r2, ck_r2 = C.load_r2_model(run_r2, bundle)
        structs_r2 = C.build_r2_structs(bundle, rows)
        pred_r2 = C.predict_r2(model_r2, rows, structs_r2, bundle.raw, bundle.scaler)
        max_r2 = float(np.max(np.abs(pred_r2 - stored_r2["pred"])))
        cell["R2"] = {
            "run_id": run_r2,
            "checkpoint_epoch": int(ck_r2["epoch"]),
            "n": int(rows.size),
            "max_abs_delta": max_r2,
            "rmse_replay": C.rmse(pred_r2, stored_r2["y"]),
            "rmse_stored": C.rmse(stored_r2["pred"], stored_r2["y"]),
            "ok": bool(max_r2 < 1e-5),
        }
        # EGO2
        run_ego = C.RUNS[(split, seed)]["EGO2"]
        stored_ego = C.load_predictions(run_ego, bundle)
        model_ego, ck_ego = C.load_ego2_model(run_ego, bundle)
        import sr_struct as SS
        import sr_gnn as SG

        structs_ego = SS.build_structures(bundle, rows.tolist())
        for st in structs_ego.values():
            SG.prepare_ranges(st)
        pred_ego, _roots = C.predict_ego2(model_ego, rows, structs_ego, bundle,
                                          return_roots=False)
        max_ego = float(np.max(np.abs(pred_ego - stored_ego["pred"])))
        cell["EGO2"] = {
            "run_id": run_ego,
            "checkpoint_epoch": int(ck_ego["epoch"]),
            "n": int(rows.size),
            "max_abs_delta": max_ego,
            "rmse_replay": C.rmse(pred_ego, stored_ego["y"]),
            "rmse_stored": C.rmse(stored_ego["pred"], stored_ego["y"]),
            "ok": bool(max_ego < 1e-5),
        }
        all_ok = all_ok and cell["R2"]["ok"] and cell["EGO2"]["ok"]
        out["cells"][f"{split}_S{seed}"] = cell
    out["all_ok"] = bool(all_ok)
    C.write_json(C.ANALYSIS_DIR / "replay.json", out)
    print("frozen checkpoint replay")
    for cell, rec in out["cells"].items():
        print(f"  {cell}: R2 max|delta|={rec['R2']['max_abs_delta']:.2e} "
              f"(RMSE {rec['R2']['rmse_replay']:.6f}), "
              f"EGO2 max|delta|={rec['EGO2']['max_abs_delta']:.2e} "
              f"(RMSE {rec['EGO2']['rmse_replay']:.6f})")
    print(f"  all_ok={all_ok}")
    if not all_ok:
        raise SystemExit("replay criterion failed")


if __name__ == "__main__":
    main()
"""Build the R2-SENTENCE-REPLICATION-v1 main table, summary and verdict.

Runs after both `D0_S1` and `D1_S0` results have been pulled.  Reads only the
per-run manifests and prediction files; recomputes the headline metrics from the
per-molecule predictions as a self-check.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import r2_common as RC


def _run_table(run: dict[str, Any], spec: dict[str, Any]) -> dict[str, Any]:
    manifest_path = RC.RESULTS_DIR / run["run_id"] / "manifest.json"
    manifest = RC.read_json(manifest_path)
    pred = np.load(RC.RESULTS_DIR / run["run_id"] / "predictions_best_dev.npz")
    fit = np.load(RC.RESULTS_DIR / run["run_id"] / "predictions_best_fit.npz")
    dev_rmse = RC.rmse(pred["pred"], pred["label"])
    dev_mae = RC.mae(pred["pred"], pred["label"])
    fit_rmse = RC.rmse(fit["pred"], fit["label"])
    if abs(dev_rmse - float(manifest["dev_rmse"])) > 1e-9:
        raise ValueError(f"{run['run_id']}: dev RMSE not reproducible from predictions")
    if manifest["split_name"] != run["split"] or int(manifest["seed"]) != int(run["seed"]):
        raise ValueError(f"{run['run_id']}: manifest does not match the frozen run matrix")
    if manifest["split"]["identity_sha256"] != spec["identity_sha256"]:
        raise ValueError(f"{run['run_id']}: split identity differs from the pin")
    if int(manifest["params"]["total"]) != RC.PARAMS_TOTAL:
        raise ValueError(f"{run['run_id']}: unexpected parameter count")
    return {
        "run_id": run["run_id"], "split": run["split"], "seed": int(run["seed"]),
        "train_rmse": float(fit_rmse), "dev_rmse": float(dev_rmse),
        "dev_mae": float(dev_mae), "best_epoch": int(manifest["best_epoch"]),
        "epochs_run": int(manifest["epochs_run"]),
        "wall_seconds": float(manifest["wall_seconds"]),
        "gpu_hours": float(manifest["gpu_hours"]),
        "gpu_hours_total_process": float(manifest["gpu_hours_total_process"]),
        "params": int(manifest["params"]["total"]),
        "n_fit": int(manifest["n_fit"]), "n_dev": int(manifest["n_dev"]),
        "fit_rows_sha256": str(manifest["fit_rows_sha256"]),
        "dev_rows_sha256": str(manifest["dev_rows_sha256"]),
        "split_identity_sha256": str(manifest["split"]["identity_sha256"]),
        "split_file_sha256": str(manifest["split"]["file_sha256"]),
        "scaler": manifest["scaler"], "atom_scalar_mode": manifest["atom_scalar_mode"],
        "normalization": manifest["normalization"],
        "git_commit": str(manifest["git_commit"]),
        "frozen_gwic_code_sha256": str(manifest["frozen_gwic_code_sha256"]),
        "host": str(manifest["host"]), "gpu_name": str(manifest["gpu_name"]),
        "driver": str(manifest["driver"]), "torch": str(manifest["torch"]),
        "cuda": str(manifest["cuda"]), "device": str(manifest["device"]),
        "epochs_max": int(manifest["epochs_max"]),
        "early_stop": manifest["early_stop"], "recipe": manifest["recipe"],
        "predictions_dev": str(RC.RESULTS_DIR / run["run_id"] / "predictions_best_dev.npz"),
    }


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=str(RC.RESULTS_DIR))
    args = ap.parse_args(argv)
    cfg = RC.load_config()
    rows = [_run_table(run, cfg["splits"][run["split"]]) for run in cfg["runs"]]
    RC.write_csv(Path(args.out) / "main_table.csv", rows)

    hist = dict(cfg["historical_coordinates"])
    thresholds = cfg["verdict_rule"]["replicated_if"]
    d0 = next(r for r in rows if r["split"] == "D0")
    d1 = next(r for r in rows if r["split"] == "D1")
    gates = {
        "D0_S1_dev_rmse": d0["dev_rmse"],
        "D0_S1_dev_rmse_max": float(thresholds["D0_S1_dev_rmse_max"]),
        "D0_S1_pass": bool(d0["dev_rmse"] <= float(thresholds["D0_S1_dev_rmse_max"])),
        "D1_S0_dev_rmse": d1["dev_rmse"],
        "D1_S0_dev_rmse_max": float(thresholds["D1_S0_dev_rmse_max"]),
        "D1_S0_pass": bool(d1["dev_rmse"] <= float(thresholds["D1_S0_dev_rmse_max"])),
    }
    replicated = bool(gates["D0_S1_pass"] and gates["D1_S0_pass"])
    comparisons = {
        "D0_S1_minus_GWIC_v0_s0": float(d0["dev_rmse"] - float(cfg["gwic_v0_reference"]["dev_rmse"])),
        "D0_S1_minus_GINE_D0_mean": float(d0["dev_rmse"] - float(hist["GINE_D0_seed_mean"])),
        "D1_S0_minus_GINE_D1_mean": float(d1["dev_rmse"] - float(hist["GINE_D1_seed_mean"])),
        "D1_S0_minus_ACRA_D1": float(d1["dev_rmse"] - float(hist["ACRA_D1"])),
        "D1_S0_minus_ACRA_D0": float(d1["dev_rmse"] - float(hist["ACRA_D0"])),
        "D1_S0_minus_D0_S1": float(d1["dev_rmse"] - d0["dev_rmse"]),
        "sign_convention": "positive = the R2 number is worse (higher RMSE)",
    }
    checks_path = RC.ANALYSIS_DIR / "checks.json"
    checks = RC.read_json(checks_path) if checks_path.is_file() else {}
    budget = {
        "wall_seconds_sum_of_runs": float(sum(r["wall_seconds"] for r in rows)),
        "gpu_hours_sum_of_runs": float(sum(r["gpu_hours"] for r in rows)),
        "wall_hours_target": float(cfg["resource_budget"]["wall_hours_target"]),
        "gpu_hours_target": float(cfg["resource_budget"]["gpu_hours_target"]),
        "within_target": bool(
            sum(r["wall_seconds"] for r in rows) / 3600.0 <= float(cfg["resource_budget"]["wall_hours_target"])
            and sum(r["gpu_hours"] for r in rows) <= float(cfg["resource_budget"]["gpu_hours_target"])),
    }
    summary = {
        "protocol_id": cfg["protocol_id"],
        "generated_utc": RC.now_utc(),
        "git_commit": RC.git_commit(),
        "model": cfg["model"]["arm"],
        "params": rows[0]["params"],
        "runs": rows,
        "gates": gates,
        "verdict": ("R2_BASELINE_REPLICATED_PRELIMINARY" if replicated
                    else "R2_BASELINE_SENSITIVITY_UNRESOLVED"),
        "comparisons": comparisons,
        "historical_coordinates": hist,
        "budget": budget,
        "engineering_checks": {
            "path": str(checks_path.relative_to(RC.REPO_ROOT)) if checks else "",
            "n_checks": int(checks.get("n_checks", 0)),
            "n_pass": int(checks.get("n_pass", 0)),
        },
        "non_claims": [
            "historical coordinates are different recipes, not a matched benchmark",
            "the engineering threshold is not a statistical significance test",
            "D1 is a reproducibility check on official Train rows, not a blind test",
            "no official Valid/Test metric is reported",
        ],
    }
    RC.write_json(Path(args.out) / "summary.json", summary)

    print("run       split seed  train_rmse  dev_rmse  dev_mae  best_ep  epochs  wall_s  gpu_h")
    for r in rows:
        print(f"{r['run_id']:<9} {r['split']:<5} {r['seed']:>4}  "
              f"{r['train_rmse']:>10.4f}  {r['dev_rmse']:>8.4f}  {r['dev_mae']:>7.4f}  "
              f"{r['best_epoch']:>7}  {r['epochs_run']:>6}  {r['wall_seconds']:>6.0f}  "
              f"{r['gpu_hours']:>5.3f}")
    print(f"\nverdict: {summary['verdict']}  (gates {gates})")
    print(f"budget : {budget['gpu_hours_sum_of_runs']:.3f} GPU-h, "
          f"{budget['wall_seconds_sum_of_runs'] / 3600:.3f} h of run wall "
          f"-> within_target={budget['within_target']}")
    print(f"checks : {summary['engineering_checks']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
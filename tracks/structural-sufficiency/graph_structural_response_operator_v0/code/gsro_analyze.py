"""Aggregate the two arms: main table, paired bootstrap and decision flags.

    python code/gsro_analyze.py

Reads ``runs/full`` and ``runs/local2`` (produced on ``res``), writes durable
artefacts into ``results/``.  Test is DEVELOPMENT_TEST: it is summarised here
only after both arms finished and were checkpoint-selected on Valid.
"""

from __future__ import annotations

import csv
import shutil
from pathlib import Path
from typing import Any

import numpy as np

import gsro_common as C

HISTORICAL = {
    "HSRF": {"valid_rmse": 0.7915, "test_rmse": 0.7819},
    "LCRC": {"valid_rmse": 0.7052, "test_rmse": 0.7139},
    "SWF_WORD_PREFIX": {"valid_rmse": 0.6991, "test_rmse": 0.6891},
    "GINE": {"valid_rmse": 0.6091, "test_rmse": 0.6456},
}


def read_curve(path: Path) -> list[dict[str, float]]:
    rows = []
    with open(path, newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            rows.append({k: float(v) for k, v in row.items()})
    return rows


def load_arm(root: Path) -> dict[str, Any]:
    run = C.read_json(root / "run.json")
    with np.load(root / "predictions.npz") as handle:
        preds = {key: np.asarray(handle[key]) for key in handle.files}
    return {
        "dir": root,
        "run": run,
        "curve": read_curve(root / "curve.csv"),
        "preds": preds,
    }


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    err = pred - y
    return {
        "rmse": float(np.sqrt(np.mean(err**2))),
        "mae": float(np.mean(np.abs(err))),
        "n": int(len(y)),
    }


def paired_bootstrap(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    y: np.ndarray,
    *,
    replicates: int,
    seed: int,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y)
    delta_rmse = np.empty(replicates)
    delta_mae = np.empty(replicates)
    for i in range(replicates):
        index = rng.integers(0, n, n)
        err_a = pred_a[index] - y[index]
        err_b = pred_b[index] - y[index]
        delta_rmse[i] = np.sqrt(np.mean(err_a**2)) - np.sqrt(np.mean(err_b**2))
        delta_mae[i] = np.mean(np.abs(err_a)) - np.mean(np.abs(err_b))
    err_a = pred_a - y
    err_b = pred_b - y
    point = {
        "delta_rmse": float(np.sqrt(np.mean(err_a**2)) - np.sqrt(np.mean(err_b**2))),
        "delta_mae": float(np.mean(np.abs(err_a)) - np.mean(np.abs(err_b))),
    }

    def interval(values):
        return [float(np.percentile(values, 2.5)), float(np.percentile(values, 97.5))]

    return {
        "replicates": replicates,
        "seed": seed,
        "delta_rmse": point["delta_rmse"],
        "delta_rmse_ci95": interval(delta_rmse),
        "delta_mae": point["delta_mae"],
        "delta_mae_ci95": interval(delta_mae),
        "delta_rmse_ci_excludes_zero": bool(
            (np.percentile(delta_rmse, 2.5) > 0) or (np.percentile(delta_rmse, 97.5) < 0)
        ),
    }


def per_molecule_diff(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    err_a = np.abs(pred_a - y)
    err_b = np.abs(pred_b - y)
    sq_a = err_a**2
    sq_b = err_b**2
    diff = sq_a - sq_b
    return {
        "n": int(len(y)),
        "mean_squared_error_difference": float(np.mean(diff)),
        "median_squared_error_difference": float(np.median(diff)),
        "mean_absolute_error_difference": float(np.mean(err_a - err_b)),
        "median_absolute_error_difference": float(np.median(err_a - err_b)),
        "full_better_fraction_squared": float(np.mean(diff < 0)),
        "full_better_fraction_absolute": float(np.mean(err_a - err_b < 0)),
        "max_abs_squared_difference": float(np.max(np.abs(diff))),
    }


def curve_comparison(full_curve, local2_curve) -> dict[str, Any]:
    n = min(len(full_curve), len(local2_curve))
    epochs = [int(full_curve[i]["epoch"]) for i in range(n)]
    valid_full = np.asarray([full_curve[i]["valid_rmse"] for i in range(n)])
    valid_local = np.asarray([local2_curve[i]["valid_rmse"] for i in range(n)])
    train_full = np.asarray([full_curve[i]["train_rmse"] for i in range(n)])
    train_local = np.asarray([local2_curve[i]["train_rmse"] for i in range(n)])
    test_full = np.asarray([full_curve[i]["test_rmse"] for i in range(n)])
    test_local = np.asarray([local2_curve[i]["test_rmse"] for i in range(n)])
    best_full = int(np.argmin(valid_full))
    best_local = int(np.argmin(valid_local))
    final_improvement = float(np.mean(valid_full[max(0, n - 6) : n - 1]) - valid_full[n - 1])
    return {
        "n_epochs": int(n),
        "valid_delta_full_minus_local2": {
            "mean": float(np.mean(valid_full - valid_local)),
            "median": float(np.median(valid_full - valid_local)),
            "full_better_epoch_fraction": float(np.mean(valid_full < valid_local)),
            "first_epoch": float(valid_full[0] - valid_local[0]),
            "last_epoch": float(valid_full[-1] - valid_local[-1]),
            "at_full_best_epoch": float(
                valid_full[best_full] - valid_local[best_full]
            ),
            "at_local2_best_epoch": float(
                valid_full[best_local] - valid_local[best_local]
            ),
        },
        "train_delta_full_minus_local2": {
            "mean": float(np.mean(train_full - train_local)),
            "at_full_best_epoch": float(train_full[best_full] - train_local[best_full]),
            "at_local2_best_epoch": float(train_full[best_local] - train_local[best_local]),
            "last_epoch": float(train_full[-1] - train_local[-1]),
        },
        "test_delta_full_minus_local2": {
            "mean": float(np.mean(test_full - test_local)),
            "at_full_best_epoch": float(test_full[best_full] - test_local[best_full]),
            "at_local2_best_epoch": float(test_full[best_local] - test_local[best_local]),
        },
        "best_epochs": {"full": best_full, "local2": best_local},
        "best_valid": {"full": float(valid_full[best_full]), "local2": float(valid_local[best_local])},
        "last_epoch_valid": {"full": float(valid_full[-1]), "local2": float(valid_local[-1])},
        "valid_last5_gain_full": final_improvement,
        "curves": {
            "epoch": epochs,
            "valid_full": valid_full.tolist(),
            "valid_local2": valid_local.tolist(),
            "train_full": train_full.tolist(),
            "train_local2": train_local.tolist(),
            "test_full": test_full.tolist(),
            "test_local2": test_local.tolist(),
        },
    }


def main() -> None:
    config = C.load_config()
    results = C.results_dir()
    results.mkdir(parents=True, exist_ok=True)
    runs = C.runs_dir()
    full = load_arm(runs / "full")
    local2 = load_arm(runs / "local2")

    # -- durable copies ----------------------------------------------------
    shutil.copyfile(runs / "full" / "curve.csv", results / "curve_full.csv")
    shutil.copyfile(runs / "local2" / "curve.csv", results / "curve_local2.csv")
    shutil.copyfile(runs / "full" / "best.pt", results / "full_best.pt")
    shutil.copyfile(runs / "local2" / "best.pt", results / "local2_best.pt")
    shutil.copyfile(runs / "full" / "predictions.npz", results / "predictions_full.npz")
    shutil.copyfile(runs / "local2" / "predictions.npz", results / "predictions_local2.npz")

    # -- main table --------------------------------------------------------
    table_rows = []
    for arm_name, arm in (("FULL", full), ("LOCAL2", local2)):
        run = arm["run"]
        metrics_best = run["final_metrics_best_checkpoint"]
        table_rows.append(
            {
                "model": arm_name,
                "params": int(run["param_count"]),
                "best_epoch": int(run["best_epoch"]),
                "train_rmse": float(metrics_best["train"]["rmse"]),
                "valid_rmse": float(metrics_best["valid"]["rmse"]),
                "test_rmse": float(metrics_best["test"]["rmse"]),
                "train_mae": float(metrics_best["train"]["mae"]),
                "valid_mae": float(metrics_best["valid"]["mae"]),
                "test_mae": float(metrics_best["test"]["mae"]),
                "wall_seconds": float(run["wall_seconds"]),
                "wall_minutes": float(run["wall_seconds"]) / 60.0,
                "epochs": int(run["epochs"]),
                "batch_size": int(run["batch_size"]),
                "peak_gpu_mb": float(run["peak_gpu_bytes"]) / 2**20,
                "initial_state_sha256": run["initial_state_sha256"],
            }
        )
    with open(results / "main_table.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(table_rows[0]))
        writer.writeheader()
        writer.writerows(table_rows)

    # -- paired comparisons (best-Valid checkpoint predictions) ------------
    paired = {}
    for split in ("valid", "test"):
        y = np.asarray(full["preds"][f"{split}_y"], dtype=np.float64)
        local_y = np.asarray(local2["preds"][f"{split}_y"], dtype=np.float64)
        if not np.allclose(y, local_y):
            raise ValueError(f"{split}: the two arms saw different targets")
        pred_full = np.asarray(full["preds"][f"{split}_pred"], dtype=np.float64)
        pred_local = np.asarray(local2["preds"][f"{split}_pred"], dtype=np.float64)
        paired[split] = {
            "full": metrics(pred_full, y),
            "local2": metrics(pred_local, y),
            "gain_full_vs_local2_rmse": float(
                np.sqrt(np.mean((pred_local - y) ** 2)) - np.sqrt(np.mean((pred_full - y) ** 2))
            ),
            "per_molecule_squared_error_difference_full_minus_local2": per_molecule_diff(
                pred_full, pred_local, y
            ),
            "bootstrap_full_minus_local2": paired_bootstrap(
                pred_full,
                pred_local,
                y,
                replicates=int(config["bootstrap_replicates"]),
                seed=int(config["bootstrap_seed"]),
            ),
        }

    curve = curve_comparison(full["curve"], local2["curve"])

    # -- explanation (produced by the FULL job) ----------------------------
    explanation_path = runs / "full" / "explanation.json"
    explanation = C.read_json(explanation_path) if explanation_path.is_file() else None
    if explanation is not None:
        shutil.copyfile(explanation_path, results / "explanation.json")

    # -- descriptive historical coordinates --------------------------------
    historical = {
        name: {
            **values,
            "note": "historical protocol/horizon differs; descriptive coordinate only",
            "delta_valid_vs_full": float(values["valid_rmse"] - table_rows[0]["valid_rmse"]),
            "delta_test_vs_full": float(values["test_rmse"] - table_rows[0]["test_rmse"]),
        }
        for name, values in HISTORICAL.items()
    }

    # -- explicit section-11 reference flags -------------------------------
    full_valid = table_rows[0]["valid_rmse"]
    full_test = table_rows[0]["test_rmse"]
    local_valid = table_rows[1]["valid_rmse"]
    valid_boot = paired["valid"]["bootstrap_full_minus_local2"]
    flags = {
        "full_valid_leq_0_70": bool(full_valid <= 0.70),
        "full_valid_better_than_local2": bool(full_valid < local_valid),
        "full_valid_leq_0_65": bool(full_valid <= 0.65),
        "both_worse_than_lcrc_0_7052_by_more_than_0_75": bool(
            full_valid > 0.75 and local_valid > 0.75
        ),
        "valid_delta_ci_excludes_zero": valid_boot["delta_rmse_ci_excludes_zero"],
        "frozen_substitution_shows_dependence": (
            None
            if explanation is None
            else bool(
                explanation["frozen_substitution_full_valid"]["replaced_by_local2_rmse"]
                - explanation["frozen_substitution_full_valid"]["full_fp64_rmse"]
                > 1e-3
            )
        ),
        "horizon_at_best_epoch": {
            "full": int(table_rows[0]["best_epoch"]),
            "local2": int(table_rows[1]["best_epoch"]),
            "epochs": int(table_rows[0]["epochs"]),
            "full_best_is_last_epoch": bool(
                int(table_rows[0]["best_epoch"]) == int(table_rows[0]["epochs"]) - 1
            ),
            "local2_best_is_last_epoch": bool(
                int(table_rows[1]["best_epoch"]) == int(table_rows[1]["epochs"]) - 1
            ),
        },
    }

    analysis = {
        "round": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "environment": C.environment(),
        "data": full["run"]["data"],
        "arms": {
            "full": {
                "run": full["run"],
                "config_sha256": full["run"]["config_sha256"],
            },
            "local2": {
                "run": local2["run"],
                "config_sha256": local2["run"]["config_sha256"],
            },
        },
        "main_table": table_rows,
        "best_valid_gap_full_minus_local2": float(full_valid - local_valid),
        "absolute_gap_vs_historical": {
            "LCRC_valid": float(full_valid - HISTORICAL["LCRC"]["valid_rmse"]),
            "LCRC_test": float(full_test - HISTORICAL["LCRC"]["test_rmse"]),
            "GINE_valid": float(full_valid - HISTORICAL["GINE"]["valid_rmse"]),
            "GINE_test": float(full_test - HISTORICAL["GINE"]["test_rmse"]),
            "HSRF_valid": float(full_valid - HISTORICAL["HSRF"]["valid_rmse"]),
            "HSRF_test": float(full_test - HISTORICAL["HSRF"]["test_rmse"]),
        },
        "paired": paired,
        "curve_comparison": curve,
        "historical_coordinates": historical,
        "explanation_summary": None
        if explanation is None
        else {
            "decomposition_max_abs_error": explanation["prediction_decomposition"][
                "max_abs_reconstruction_error"
            ],
            "decomposition_passed": explanation["prediction_decomposition"]["passed"],
            "coupling_max_abs_analytic_vs_autograd": explanation["coupling_sensitivity"][
                "max_abs_analytic_vs_autograd"
            ],
            "coupling_max_abs_fd_error": explanation["coupling_sensitivity"][
                "max_abs_analytic_vs_fd"
            ],
            "frozen_substitution": explanation["frozen_substitution_full_valid"],
        },
        "reference_flags": flags,
        "bootstrap": {
            "replicates": int(config["bootstrap_replicates"]),
            "seed": int(config["bootstrap_seed"]),
            "note": "paired percentile bootstrap over molecules; single seed, not a seed-stability claim",
        },
    }
    C.write_json(results / "analysis.json", analysis)

    # -- console summary ---------------------------------------------------
    for row in table_rows:
        print(
            f"GSRO_MAIN {row['model']} params={row['params']} best_epoch={row['best_epoch']} "
            f"train={row['train_rmse']:.5f} valid={row['valid_rmse']:.5f} "
            f"test={row['test_rmse']:.5f} wall_min={row['wall_minutes']:.2f}",
            flush=True,
        )
    for split in ("valid", "test"):
        boot = paired[split]["bootstrap_full_minus_local2"]
        print(
            f"GSRO_PAIRED {split} delta_rmse_full_minus_local2={boot['delta_rmse']:.5f} "
            f"ci95={boot['delta_rmse_ci95']}",
            flush=True,
        )
    print(f"GSRO_ANALYSIS wrote {results}", flush=True)


if __name__ == "__main__":
    main()

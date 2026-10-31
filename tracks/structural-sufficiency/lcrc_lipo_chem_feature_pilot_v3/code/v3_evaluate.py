"""Paired Fold-1 dev analysis and the pre-registered v3 decision rules.

Conventions
-----------
``delta = RMSE(historical same-architecture model) - RMSE(v3 chemistry model)``.
A positive value favours the chemistry-augmented model.

The tail diagnostic uses a *pre-registered* exclusion set: the top 2% of
historical-comparator absolute dev residuals, selected before any v3 GPU run.
It is a stability diagnostic, not a replacement for the primary dev RMSE.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Sequence

import numpy as np

import v3_common as C
import v3_registry as R


# ---------------------------------------------------------------------------
# metrics / bootstrap
# ---------------------------------------------------------------------------


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean(np.abs(y - p)))


def bootstrap_delta(
    y: np.ndarray,
    hist: np.ndarray,
    new: np.ndarray,
    *,
    metric: str,
    n_boot: int = C.BOOTSTRAP_RESAMPLES,
    seed: int = C.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    a = np.asarray(hist, dtype=np.float64)
    b = np.asarray(new, dtype=np.float64)
    fn = rmse if metric == "rmse" else mae
    n = y.size
    rng = np.random.default_rng(int(seed))
    point = fn(y, a) - fn(y, b)
    vals = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        vals[i] = fn(y[idx], a[idx]) - fn(y[idx], b[idx])
    return {
        "metric": metric,
        "delta_definition": f"{metric.upper()}(historical) - {metric.upper()}(v3); positive favours v3",
        "point": float(point),
        "mean": float(vals.mean()),
        "ci95_low": float(np.quantile(vals, 0.025)),
        "ci95_high": float(np.quantile(vals, 0.975)),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "n_molecules": int(n),
        "ci_includes_zero": bool(np.quantile(vals, 0.025) <= 0.0 <= np.quantile(vals, 0.975)),
    }


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _dev_rows_and_labels() -> tuple[list[int], np.ndarray]:
    folds = C.F.load_folds()
    positions = [int(p) for p in C.F.fold_positions(folds, C.FOLD)]
    y_all = np.asarray(C.load_fit_labels()["y"], dtype=np.float64)
    return positions, y_all[np.asarray(positions, dtype=np.int64)]


def load_run(key: str) -> dict[str, Any]:
    train_path = C.RUNS / key / "train.json"
    pred_path = C.RUNS / key / "predictions.npz"
    if not train_path.exists() or not pred_path.exists():
        raise FileNotFoundError(f"missing v3 run {key}")
    train = C.read_json(train_path)
    data = np.load(pred_path)
    rows = [int(x) for x in data["rows"].tolist()]
    pred = np.asarray(data["pred"], dtype=np.float64)
    expected_rows, y = _dev_rows_and_labels()
    if rows != expected_rows:
        raise ValueError(f"{key}: prediction rows do not match fold-1 dev positions")
    return {
        "key": key,
        "train": train,
        "rows": rows,
        "pred": pred,
        "y": y,
        "train_rmse": float(train["metrics"]["train_rmse_raw"]),
        "dev_rmse": float(train["metrics"]["dev_rmse_raw"]),
        "dev_mae": float(train["metrics"]["dev_mae_raw"]),
    }


def load_historical(arm: str) -> dict[str, Any]:
    hist = C.HISTORICAL[arm]
    run_dir = C.V2_RUNS / hist["key"]
    train = C.read_json(run_dir / "train.json")
    data = np.load(run_dir / "predictions.npz")
    rows = [int(x) for x in data["rows"].tolist()]
    pred = np.asarray(data["pred"], dtype=np.float64)
    expected_rows, y = _dev_rows_and_labels()
    if rows != expected_rows:
        raise ValueError(f"{hist['key']}: prediction rows do not match fold-1 dev positions")
    if abs(float(train["metrics"]["dev_rmse_raw"]) - float(hist["dev_rmse"])) > 1e-9:
        raise ValueError(f"{hist['key']}: historical RMSE constant changed")
    return {
        "key": hist["key"],
        "train": train,
        "rows": rows,
        "pred": pred,
        "y": y,
        "train_rmse": float(train["metrics"]["train_rmse_raw"]),
        "dev_rmse": float(train["metrics"]["dev_rmse_raw"]),
        "dev_mae": float(train["metrics"]["dev_mae_raw"]),
    }


# ---------------------------------------------------------------------------
# tail diagnostic
# ---------------------------------------------------------------------------


def tail_analysis(
    y: np.ndarray,
    hist: np.ndarray,
    new: np.ndarray,
    *,
    fraction: float = C.TAIL_FRACTION,
    min_retained_fraction: float = C.TAIL_MIN_RETAINED_FRACTION,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    h = np.asarray(hist, dtype=np.float64)
    n = np.asarray(new, dtype=np.float64)
    err_h = h - y
    err_n = n - y
    abs_h = np.abs(err_h)
    order = np.argsort(-abs_h, kind="stable")
    n_drop = max(1, int(round(float(fraction) * y.size)))
    drop_idx = order[:n_drop]
    keep = np.ones(y.size, dtype=bool)
    keep[drop_idx] = False
    delta_full = rmse(y, h) - rmse(y, n)
    delta_keep = rmse(y[keep], h[keep]) - rmse(y[keep], n[keep])
    sq_improvement = err_h * err_h - err_n * err_n
    net = float(sq_improvement.sum())
    tail = float(sq_improvement[drop_idx].sum())
    top1 = float(sq_improvement[drop_idx].max()) if n_drop else 0.0
    retained_fraction = (delta_keep / delta_full) if delta_full > 0 else None
    tail_share = (tail / net) if net != 0 else None
    top1_share = (max(0.0, float(sq_improvement.max())) / net) if net > 0 else None
    not_tail_dependent = bool(
        delta_keep > 0
        and retained_fraction is not None
        and retained_fraction >= float(min_retained_fraction)
        and top1_share is not None
        and top1_share < 1.0
    )
    return {
        "fraction": float(fraction),
        "n_dropped": int(n_drop),
        "n_kept": int(keep.sum()),
        "exclusion_rule": "top absolute residual of the historical same-architecture model; row-id stable tie-break",
        "dropped_rows": [int(x) for x in drop_idx.tolist()],
        "dropped_abs_error_min": float(abs_h[drop_idx].min()),
        "dropped_abs_error_max": float(abs_h[drop_idx].max()),
        "delta_full": float(delta_full),
        "delta_drop_top_tail": float(delta_keep),
        "retained_fraction": None if retained_fraction is None else float(retained_fraction),
        "net_squared_error_improvement": net,
        "tail_squared_error_improvement": tail,
        "tail_share_of_net_gain": None if tail_share is None else float(tail_share),
        "top1_squared_error_improvement": top1,
        "top1_share_of_net_gain": None if top1_share is None else float(top1_share),
        "min_retained_fraction": float(min_retained_fraction),
        "direction_stable_after_tail_drop": bool(delta_keep > 0),
        "not_tail_dependent": not_tail_dependent,
    }


# ---------------------------------------------------------------------------
# paired comparison
# ---------------------------------------------------------------------------


def compare(historical: dict[str, Any], new: dict[str, Any], *, tail_source: str) -> dict[str, Any]:
    y = historical["y"]
    h = historical["pred"]
    n = new["pred"]
    boot_rmse = bootstrap_delta(y, h, n, metric="rmse", seed=C.BOOTSTRAP_SEED)
    boot_mae = bootstrap_delta(y, h, n, metric="mae", seed=C.BOOTSTRAP_SEED + 1)
    sq = (h - y) ** 2 - (n - y) ** 2
    ab = np.abs(h - y) - np.abs(n - y)
    tail = tail_analysis(y, h, n)
    return {
        "historical_key": historical["key"],
        "new_key": new["key"],
        "n": int(y.size),
        "historical": {
            "train_rmse": historical["train_rmse"],
            "dev_rmse": historical["dev_rmse"],
            "dev_mae": historical["dev_mae"],
        },
        "new": {
            "train_rmse": new["train_rmse"],
            "dev_rmse": new["dev_rmse"],
            "dev_mae": new["dev_mae"],
        },
        "delta_rmse": float(historical["dev_rmse"] - new["dev_rmse"]),
        "delta_mae": float(historical["dev_mae"] - new["dev_mae"]),
        "relative_rmse_change": float(
            (new["dev_rmse"] - historical["dev_rmse"]) / historical["dev_rmse"]
        ),
        "bootstrap_rmse": boot_rmse,
        "bootstrap_mae": boot_mae,
        "paired_molecule": {
            "mean_squared_improvement": float(sq.mean()),
            "median_squared_improvement": float(np.median(sq)),
            "mean_absolute_improvement": float(ab.mean()),
            "median_absolute_improvement": float(np.median(ab)),
            "win_rate_absolute": float(np.mean(ab > 0)),
            "win_rate_squared": float(np.mean(sq > 0)),
            "n_molecules_improved_absolute": int(np.sum(ab > 0)),
            "n_molecules_worsened_absolute": int(np.sum(ab < 0)),
        },
        "tail": {**tail, "source": tail_source},
    }


def conditional_condition_from_runs() -> dict[str, Any]:
    """Evaluate the pre-registered C gate from the two A runs only."""
    try:
        hist = load_historical("A")
        new = load_run("F1_A_Chem12_R3_s0")
    except FileNotFoundError as exc:
        return {"met": False, "reason": f"missing run: {exc}"}
    cmp = compare(hist, new, tail_source=C.HISTORICAL["A"]["key"])
    met = bool(
        cmp["delta_rmse"] >= C.MATERIALITY_RMSE
        and cmp["delta_mae"] > 0
        and cmp["tail"]["not_tail_dependent"]
    )
    return {
        "met": met,
        "delta_rmse": cmp["delta_rmse"],
        "delta_mae": cmp["delta_mae"],
        "tail_not_dependent": cmp["tail"]["not_tail_dependent"],
        "tail_retained_fraction": cmp["tail"]["retained_fraction"],
        "rule": "delta_rmse >= 0.03 and delta_mae > 0 and not tail-dependent",
    }


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def _new_arm_available(arm_key: str) -> bool:
    return (C.RUNS / arm_key / "train.json").exists() and (C.RUNS / arm_key / "predictions.npz").exists()


def decide(results: dict[str, Any]) -> dict[str, Any]:
    if "A_Chem12" not in results["comparisons"] or "A_Chem13" not in results["comparisons"]:
        return {
            "verdict": "PILOT_NOT_STARTED",
            "reason": "A_Chem12 and/or A_Chem13 GPU runs are missing",
            "materiality_rmse": C.MATERIALITY_RMSE,
        }
    c12 = results["comparisons"]["A_Chem12"]
    c13 = results["comparisons"]["A_Chem13"]
    d12 = float(c12["delta_rmse"])
    d13 = float(c13["delta_rmse"])
    m12 = float(c12["delta_mae"])
    m13 = float(c13["delta_mae"])
    robust12 = bool(c12["tail"]["not_tail_dependent"])
    robust13 = bool(c13["tail"]["not_tail_dependent"])
    material = C.MATERIALITY_RMSE
    incremental_logp = d13 - d12

    if d12 >= material and not robust12:
        verdict = "INCONCLUSIVE_TAIL_DEPENDENT"
        reason = "Chem12 point gain clears 0.03 but is not stable after the pre-registered top-2% tail drop"
    elif d12 >= material and d13 < material:
        verdict = "INCONCLUSIVE_MOLLOGP_REMOVES_GAIN"
        reason = "Chem12 clears 0.03 but adding MolLogP drops the gain below 0.03"
    elif d12 >= material and (m12 <= 0 or m13 <= 0):
        verdict = "INCONCLUSIVE_RMSE_MAE_CONFLICT"
        reason = "RMSE gain clears 0.03 but the paired MAE direction conflicts"
    elif d12 >= material and d13 >= material and robust13:
        verdict = "GENERAL_CHEM_PRIOR_SUPPORTED"
        reason = "Chem12 and Chem13 both improve RMSE by >= 0.03 and the gains are not tail-dependent"
    elif d12 < material and d13 >= material and robust13 and m13 > 0:
        verdict = "DIRECT_LOGP_PRIOR_ONLY"
        reason = "Chem12 is below 0.03, Chem13 clears 0.03; interpret as the direct MolLogP prior, not a general structural-representation breakthrough"
    elif d12 < material and d13 < material:
        verdict = "NO_GLOBAL_DESCRIPTOR_GAIN"
        reason = "Neither Chem12 nor Chem13 reaches the 0.03 RMSE materiality threshold"
    else:
        verdict = "INCONCLUSIVE"
        reason = "Mixed or unstable pattern across Chem12/Chem13, tail stability or MAE direction"

    conditional = results.get("conditional_C", {})
    if conditional.get("run_executed"):
        cc = results["comparisons"].get("C_Chem12")
        cross = bool(
            cc is not None
            and cc["delta_rmse"] > 0
            and cc["delta_mae"] > 0
            and cc["tail"]["not_tail_dependent"]
        )
        conditional["cross_architecture_consistent"] = cross
        conditional["cross_architecture_note"] = (
            "C_Chem12 moves in the same direction and survives the same tail rule"
            if cross
            else "C_Chem12 does not reproduce the A direction under the same tail rule"
        )
    else:
        conditional.setdefault("cross_architecture_consistent", None)
    return {
        "verdict": verdict,
        "reason": reason,
        "materiality_rmse": material,
        "A_Chem12": {
            "delta_rmse": d12,
            "delta_mae": m12,
            "tail_not_dependent": robust12,
            "tail_retained_fraction": c12["tail"]["retained_fraction"],
            "bootstrap_rmse_ci95": [c12["bootstrap_rmse"]["ci95_low"], c12["bootstrap_rmse"]["ci95_high"]],
        },
        "A_Chem13": {
            "delta_rmse": d13,
            "delta_mae": m13,
            "tail_not_dependent": robust13,
            "tail_retained_fraction": c13["tail"]["retained_fraction"],
            "bootstrap_rmse_ci95": [c13["bootstrap_rmse"]["ci95_low"], c13["bootstrap_rmse"]["ci95_high"]],
        },
        "mollogp_incremental_delta_rmse": float(incremental_logp),
        "mollogp_incremental_reading": (
            "positive: MolLogP adds on top of Chem12"
            if incremental_logp > 0
            else "negative: MolLogP does not add on top of Chem12"
        ),
        "conditional_C": conditional,
        "rules": {
            "general_chem_prior": "Chem12 delta >= 0.03, tail-robust; Chem13 also >= 0.03",
            "direct_logp_prior": "Chem12 < 0.03, Chem13 >= 0.03, tail-robust",
            "no_gain": "both < 0.03",
            "inconclusive": "tail-dependent, MolLogP removes the gain, or RMSE/MAE conflict",
        },
        "monitor_used": False,
        "reserved_used": False,
        "official_valid_test_used": False,
    }


# ---------------------------------------------------------------------------
# suite
# ---------------------------------------------------------------------------


def analyse() -> dict[str, Any]:
    historical = {"A": load_historical("A"), "C": load_historical("C")}
    new: dict[str, Any] = {}
    comparisons: dict[str, Any] = {}
    mapping = {
        "A_Chem12": ("F1_A_Chem12_R3_s0", "A"),
        "A_Chem13": ("F1_A_Chem13_R3_s0", "A"),
        "C_Chem12": ("F1_C_Chem12_R3_s0", "C"),
    }
    for label, (key, arm) in mapping.items():
        if _new_arm_available(key):
            run = load_run(key)
            new[label] = run
            comparisons[label] = compare(historical[arm], run, tail_source=C.HISTORICAL[arm]["key"])

    table = []
    for arm in ("A", "C"):
        h = historical[arm]
        table.append(
            {
                "model": f"{arm}-R3 (v2 historical)",
                "fold": int(C.FOLD),
                "seed": int(C.SEED),
                "train_rmse": h["train_rmse"],
                "dev_rmse": h["dev_rmse"],
                "dev_mae": h["dev_mae"],
                "delta_rmse_vs_same_arch_original": 0.0,
                "delta_note": "reference",
            }
        )
        for label in ("A_Chem12", "A_Chem13") if arm == "A" else ("C_Chem12",):
            if label in new:
                n = new[label]
                table.append(
                    {
                        "model": label,
                        "fold": int(C.FOLD),
                        "seed": int(C.SEED),
                        "train_rmse": n["train_rmse"],
                        "dev_rmse": n["dev_rmse"],
                        "dev_mae": n["dev_mae"],
                        "delta_rmse_vs_same_arch_original": float(comparisons[label]["delta_rmse"]),
                        "delta_note": f"vs {C.HISTORICAL[arm]['key']}",
                    }
                )

    conditional = {
        "run_key": "F1_C_Chem12_R3_s0",
        "run_executed": "C_Chem12" in new,
        "gate": conditional_condition_from_runs() if "A_Chem12" in new or _new_arm_available("F1_A_Chem12_R3_s0") else {
            "met": False,
            "reason": "A_Chem12 run missing",
        },
    }
    if "C_Chem12" in comparisons:
        conditional["comparison"] = comparisons["C_Chem12"]

    results = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "scope": {
            "fold": int(C.FOLD),
            "seed": int(C.SEED),
            "recipe": C.RECIPE,
            "split": "v2 Fit2352 internal scaffold fold 1",
            "dev_n": int(historical["A"]["y"].size),
            "monitor_used": False,
            "reserved_used": False,
            "official_valid_test_used": False,
            "extra_seeds": False,
            "extra_folds": False,
            "ssl": False,
            "full_fit_retrain": False,
        },
        "historical": {
            arm: {k: h[k] for k in ("key", "train_rmse", "dev_rmse", "dev_mae")}
            for arm, h in historical.items()
        },
        "new_runs": {
            label: {k: n[k] for k in ("key", "train_rmse", "dev_rmse", "dev_mae")}
            for label, n in new.items()
        },
        "comparisons": comparisons,
        "conditional_C": conditional,
        "table": table,
        "materiality_rmse": C.MATERIALITY_RMSE,
        "tail_rule": {
            "fraction": C.TAIL_FRACTION,
            "min_retained_fraction": C.TAIL_MIN_RETAINED_FRACTION,
            "exclusion_source": {
                "A": C.HISTORICAL["A"]["key"],
                "C": C.HISTORICAL["C"]["key"],
            },
        },
        "status": "COMPLETE" if len(comparisons) == (3 if "C_Chem12" in new else 2) else "PARTIAL",
    }
    decision = decide(results)
    results["decision"] = decision
    results["conditional_C"] = conditional
    C.write_json(C.ANALYSIS / "results_v3.json", results)
    C.write_json(C.ANALYSIS / "paired_v3.json", comparisons)
    C.write_json(C.ANALYSIS / "decision_v3.json", decision)

    csv_path = C.ANALYSIS / "table_v3.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(
            fh,
            fieldnames=[
                "model",
                "fold",
                "seed",
                "train_rmse",
                "dev_rmse",
                "dev_mae",
                "delta_rmse_vs_same_arch_original",
                "delta_note",
            ],
        )
        writer.writeheader()
        for row in table:
            writer.writerow(row)
    return results


__all__ = [
    "rmse",
    "mae",
    "bootstrap_delta",
    "tail_analysis",
    "compare",
    "conditional_condition_from_runs",
    "decide",
    "analyse",
]

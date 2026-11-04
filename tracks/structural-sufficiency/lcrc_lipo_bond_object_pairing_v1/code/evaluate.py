"""One-shot Monitor evaluation after the prediction manifest is frozen.

This module is the only place that imports :mod:`monitor_labels`.  It reads the
504 Monitor labels once, computes the pre-specified metrics/diagnostics, and
writes the comparison artifacts.  It never feeds anything back into training.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import common as C
import monitor_labels


# ---------------------------------------------------------------------------
# Basic metrics / bootstrap
# ---------------------------------------------------------------------------


def rmse(pred: np.ndarray, y: np.ndarray) -> float:
    e = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(np.sqrt(np.mean(e * e)))


def mae(pred: np.ndarray, y: np.ndarray) -> float:
    e = np.asarray(pred, dtype=np.float64) - np.asarray(y, dtype=np.float64)
    return float(np.mean(np.abs(e)))


def metrics(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    e = pred - y
    ss_res = float(np.sum(e * e))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return {
        "n": int(y.size),
        "rmse": float(np.sqrt(ss_res / y.size)),
        "mae": float(np.mean(np.abs(e))),
        "bias": float(np.mean(e)),
        "r2": float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan"),
        "max_abs_error": float(np.abs(e).max()),
    }


def _percentiles(values: np.ndarray, qs: Iterable[float]) -> dict[str, float]:
    values = np.asarray(values, dtype=np.float64)
    return {f"p{int(round(q * 100))}": float(np.percentile(values, q * 100)) for q in qs}


def _indices(n: int, resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n), size=(int(resamples), int(n)))


def bootstrap_delta_rmse(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray, resamples: int, seed: int) -> dict[str, Any]:
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    idx = _indices(y.size, resamples, seed)
    ea = pred_a[idx] - y[idx]
    eb = pred_b[idx] - y[idx]
    draws = np.sqrt(np.mean(ea * ea, axis=1)) - np.sqrt(np.mean(eb * eb, axis=1))
    return {
        "estimate": rmse(pred_a, y) - rmse(pred_b, y),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "resamples": int(resamples),
        "seed": int(seed),
        "unit": "molecule",
        "definition": "RMSE(A) - RMSE(B); negative favours A",
    }


def bootstrap_delta_mae(pred_a: np.ndarray, pred_b: np.ndarray, y: np.ndarray, resamples: int, seed: int) -> dict[str, Any]:
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    idx = _indices(y.size, resamples, seed)
    ea = pred_a[idx] - y[idx]
    eb = pred_b[idx] - y[idx]
    draws = np.mean(np.abs(ea), axis=1) - np.mean(np.abs(eb), axis=1)
    return {
        "estimate": mae(pred_a, y) - mae(pred_b, y),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "resamples": int(resamples),
        "seed": int(seed),
        "unit": "molecule",
    }


# ---------------------------------------------------------------------------
# Prediction loading
# ---------------------------------------------------------------------------


def load_predictions(manifest: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    rows = [int(r) for r in manifest["rows"]]
    out: dict[str, dict[str, Any]] = {}
    for bucket in ("predictions", "counterfactuals", "historical_A"):
        for key, entry in sorted(manifest[bucket].items()):
            path = C.REPO / entry["file"]
            if C.sha256_file(path) != entry["sha256"]:
                raise PermissionError(f"{key}: frozen prediction changed")
            with np.load(path) as data:
                payload = {k: np.asarray(data[k]) for k in data.files}
            if [int(r) for r in payload["rows"]] != rows:
                raise PermissionError(f"{key}: prediction rows differ from the manifest")
            out[key] = payload
    return out


# ---------------------------------------------------------------------------
# Comparison helpers
# ---------------------------------------------------------------------------


def _arm_matrix(preds: Mapping[str, Mapping[str, np.ndarray]], arm: str) -> np.ndarray:
    return np.stack([np.asarray(preds[f"{arm}_s{s}"]["pred"], dtype=np.float64) for s in C.SEEDS], axis=0)


def _arm_summary(mat: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    per_seed = [metrics(mat[i], y) for i in range(mat.shape[0])]
    rmses = np.asarray([m["rmse"] for m in per_seed], dtype=np.float64)
    maes = np.asarray([m["mae"] for m in per_seed], dtype=np.float64)
    return {
        "per_seed": {f"seed_{s}": per_seed[i] for i, s in enumerate(C.SEEDS)},
        "rmse_mean": float(rmses.mean()),
        "rmse_std": float(rmses.std(ddof=0)),
        "rmse_min": float(rmses.min()),
        "rmse_max": float(rmses.max()),
        "rmse_seed_spread": float(rmses.max() - rmses.min()),
        "mae_mean": float(maes.mean()),
        "mae_std": float(maes.std(ddof=0)),
        "rmses": rmses.tolist(),
        "maes": maes.tolist(),
        "seed_mean_prediction_rmse": rmse(mat.mean(axis=0), y),
        "seed_mean_prediction_mae": mae(mat.mean(axis=0), y),
    }


def _comparison(
    mat_a: np.ndarray,
    mat_b: np.ndarray,
    y: np.ndarray,
    *,
    name_a: str,
    name_b: str,
) -> dict[str, Any]:
    per_seed = []
    for i, seed in enumerate(C.SEEDS):
        per_seed.append({
            "seed": int(seed),
            "rmse_a": rmse(mat_a[i], y),
            "rmse_b": rmse(mat_b[i], y),
            "delta_rmse": rmse(mat_a[i], y) - rmse(mat_b[i], y),
            "mae_a": mae(mat_a[i], y),
            "mae_b": mae(mat_b[i], y),
            "delta_mae": mae(mat_a[i], y) - mae(mat_b[i], y),
            "delta_rmse_bootstrap": bootstrap_delta_rmse(mat_a[i], mat_b[i], y, C.BOOTSTRAP_RESAMPLES, C.BOOTSTRAP_SEED),
            "delta_mae_bootstrap": bootstrap_delta_mae(mat_a[i], mat_b[i], y, C.BOOTSTRAP_RESAMPLES, C.BOOTSTRAP_SEED),
        })
    deltas = np.asarray([r["delta_rmse"] for r in per_seed], dtype=np.float64)
    dmaes = np.asarray([r["delta_mae"] for r in per_seed], dtype=np.float64)
    seed_mean_a = mat_a.mean(axis=0)
    seed_mean_b = mat_b.mean(axis=0)
    return {
        "a": name_a,
        "b": name_b,
        "definition": "delta = metric(A) - metric(B); negative favours A",
        "per_seed": per_seed,
        "delta_rmse_mean": float(deltas.mean()),
        "delta_rmse_std": float(deltas.std(ddof=0)),
        "delta_mae_mean": float(dmaes.mean()),
        "delta_mae_std": float(dmaes.std(ddof=0)),
        "seeds_a_better_rmse": int((deltas < 0).sum()),
        "seeds_a_better_mae": int((dmaes < 0).sum()),
        "seed_mean_predictions": {
            "rmse_a": rmse(seed_mean_a, y),
            "rmse_b": rmse(seed_mean_b, y),
            "delta_rmse": rmse(seed_mean_a, y) - rmse(seed_mean_b, y),
            "mae_a": mae(seed_mean_a, y),
            "mae_b": mae(seed_mean_b, y),
            "delta_mae": mae(seed_mean_a, y) - mae(seed_mean_b, y),
            "delta_rmse_bootstrap": bootstrap_delta_rmse(seed_mean_a, seed_mean_b, y, C.BOOTSTRAP_RESAMPLES, C.BOOTSTRAP_SEED),
            "delta_mae_bootstrap": bootstrap_delta_mae(seed_mean_a, seed_mean_b, y, C.BOOTSTRAP_RESAMPLES, C.BOOTSTRAP_SEED),
            "note": "molecule resampling only; seeds are averaged first, not treated as independent resamples",
        },
    }


def _seed_molecule_variance(mat: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    errors = (mat - y[None, :]) ** 2
    within = errors.var(axis=0, ddof=0).mean()
    between = errors.mean(axis=0).var(ddof=0)
    total = within + between
    return {
        "within_molecule_seed_variance": float(within),
        "between_molecule_variance": float(between),
        "seed_share": float(within / total) if total > 0 else None,
        "molecule_share": float(between / total) if total > 0 else None,
        "note": "variance decomposition; not a confidence interval",
    }


# ---------------------------------------------------------------------------
# Diagnostics
# ---------------------------------------------------------------------------


def _delta_stats(preds: Mapping[str, Mapping[str, np.ndarray]], arm: str) -> dict[str, Any]:
    per_seed = {}
    all_delta = []
    for seed in C.SEEDS:
        d = np.asarray(preds[f"{arm}_s{seed}"]["delta"], dtype=np.float64)
        all_delta.append(d)
        per_seed[f"seed_{seed}"] = {
            "mean": float(d.mean()),
            "std": float(d.std()),
            "abs_mean": float(np.abs(d).mean()),
            "abs_max": float(np.abs(d).max()),
            "quantiles": _percentiles(d, (0.05, 0.25, 0.5, 0.75, 0.95)),
            "fraction_nonzero_1e6": float((np.abs(d) > 1e-6).mean()),
            "n_unique_rounded_1e9": int(len(np.unique(np.round(d, 9)))),
        }
    stacked = np.stack(all_delta, axis=0)
    return {
        "per_seed": per_seed,
        "pooled_over_seeds": {
            "mean": float(stacked.mean()),
            "std": float(stacked.std()),
            "abs_mean": float(np.abs(stacked).mean()),
            "abs_max": float(np.abs(stacked).max()),
            "quantiles": _percentiles(stacked.reshape(-1), (0.05, 0.25, 0.5, 0.75, 0.95)),
            "fraction_nonzero_1e6": float((np.abs(stacked) > 1e-6).mean()),
            "seed_mean_std": float(stacked.mean(axis=0).std()),
        },
    }


def _bond_count_groups(
    preds: Mapping[str, Mapping[str, np.ndarray]],
    y: np.ndarray,
    mat_a: np.ndarray,
    mat_r: np.ndarray,
    mat_n: np.ndarray,
) -> dict[str, Any]:
    n_bonds = np.asarray(preds["R_s0"]["n_bonds"], dtype=np.int64)
    out = {}
    for label, lo, hi in C.BOND_COUNT_BINS:
        mask = (n_bonds >= lo) & (n_bonds <= hi)
        entry = {"n_molecules": int(mask.sum()), "range": [int(lo), int(hi)]}
        if mask.any():
            r_rmse = np.asarray([rmse(mat_r[i][mask], y[mask]) for i in range(len(C.SEEDS))])
            n_rmse = np.asarray([rmse(mat_n[i][mask], y[mask]) for i in range(len(C.SEEDS))])
            a_rmse = np.asarray([rmse(mat_a[i][mask], y[mask]) for i in range(len(C.SEEDS))])
            r_mae = np.asarray([mae(mat_r[i][mask], y[mask]) for i in range(len(C.SEEDS))])
            n_mae = np.asarray([mae(mat_n[i][mask], y[mask]) for i in range(len(C.SEEDS))])
            a_mae = np.asarray([mae(mat_a[i][mask], y[mask]) for i in range(len(C.SEEDS))])
            entry.update({
                "rmse_A_mean": float(a_rmse.mean()),
                "rmse_R_mean": float(r_rmse.mean()),
                "rmse_N_mean": float(n_rmse.mean()),
                "mae_A_mean": float(a_mae.mean()),
                "mae_R_mean": float(r_mae.mean()),
                "mae_N_mean": float(n_mae.mean()),
                "delta_R_minus_A_rmse": float((r_rmse - a_rmse).mean()),
                "delta_R_minus_N_rmse": float((r_rmse - n_rmse).mean()),
                "delta_N_minus_A_rmse": float((n_rmse - a_rmse).mean()),
                "delta_R_minus_A_mae": float((r_mae - a_mae).mean()),
                "delta_R_minus_N_mae": float((r_mae - n_mae).mean()),
            })
        out[label] = entry
    return out


def _top_error_contribution(
    mat_a: np.ndarray,
    mat_r: np.ndarray,
    mat_n: np.ndarray,
    y: np.ndarray,
    *,
    n_top: int,
) -> dict[str, Any]:
    a_err = ((mat_a.mean(axis=0) - y) ** 2)
    order = np.argsort(a_err)[::-1]
    top = order[:n_top]
    def _delta_mse(mat_x: np.ndarray, mat_b: np.ndarray) -> np.ndarray:
        ex = (mat_x.mean(axis=0) - y) ** 2
        eb = (mat_b.mean(axis=0) - y) ** 2
        return eb - ex  # improvement of X over B
    dr = _delta_mse(mat_r, mat_a)
    dn = _delta_mse(mat_n, mat_a)
    return {
        "selection": "top A seed-mean squared error molecules (fixed after labels)",
        "n_top": int(n_top),
        "molecule_indices": [int(i) for i in top],
        "delta_mse_R_minus_A_total": float(dr.mean()),
        "delta_mse_N_minus_A_total": float(dn.mean()),
        "delta_mse_R_minus_A_top_contribution": float(dr[top].sum() / len(y)),
        "delta_mse_N_minus_A_top_contribution": float(dn[top].sum() / len(y)),
        "share_of_R_minus_A_improvement": float(dr[top].sum() / dr.sum()) if dr.sum() != 0 else None,
        "share_of_N_minus_A_improvement": float(dn[top].sum() / dn.sum()) if dn.sum() != 0 else None,
    }


def _remove_influential(
    mat_a: np.ndarray,
    mat_r: np.ndarray,
    mat_n: np.ndarray,
    y: np.ndarray,
    *,
    n_remove: int = 5,
) -> dict[str, Any]:
    # Fixed selection: largest absolute per-molecule R-A MSE improvement.
    ea = (mat_a.mean(axis=0) - y) ** 2
    er = (mat_r.mean(axis=0) - y) ** 2
    delta = er - ea
    order = np.argsort(np.abs(delta))[::-1]
    remove = order[:n_remove]
    keep = np.ones(len(y), dtype=bool)
    keep[remove] = False
    out = {"removed_indices": [int(i) for i in remove], "n_removed": int(n_remove)}
    for name, x, b in (("R_minus_A", mat_r, mat_a), ("R_minus_N", mat_r, mat_n), ("N_minus_A", mat_n, mat_a)):
        full = np.asarray([rmse(x[i], y) - rmse(b[i], y) for i in range(len(C.SEEDS))])
        red = np.asarray([rmse(x[i][keep], y[keep]) - rmse(b[i][keep], y[keep]) for i in range(len(C.SEEDS))])
        out[name] = {
            "full_mean_delta_rmse": float(full.mean()),
            "removed_mean_delta_rmse": float(red.mean()),
            "same_direction": bool(np.sign(full.mean()) == np.sign(red.mean())),
            "per_seed_full": full.tolist(),
            "per_seed_removed": red.tolist(),
        }
    return out


def _new_branch_diagnostics(
    preds: Mapping[str, Mapping[str, np.ndarray]],
    mat_r: np.ndarray,
    mat_n: np.ndarray,
    y: np.ndarray,
) -> dict[str, Any]:
    out = {"R": _delta_stats(preds, "R"), "N": _delta_stats(preds, "N")}
    # Are R/N branch outputs non-constant, and do they differ from one another?
    for arm, mat in (("R", mat_r), ("N", mat_n)):
        d = np.stack([np.asarray(preds[f"{arm}_s{s}"]["delta"], dtype=np.float64) for s in C.SEEDS])
        out[arm]["all_seed_delta_std"] = float(d.std(axis=0).mean())
        out[arm]["molecule_mean_std_across_seeds"] = float(d.mean(axis=0).std())
        out[arm]["rmse"] = float(np.mean([rmse(mat[i], y) for i in range(len(C.SEEDS))]))
    r = np.stack([np.asarray(preds[f"R_s{s}"]["delta"], dtype=np.float64) for s in C.SEEDS]).mean(axis=0)
    n = np.stack([np.asarray(preds[f"N_s{s}"]["delta"], dtype=np.float64) for s in C.SEEDS]).mean(axis=0)
    out["R_minus_N_delta_seedmean"] = {
        "mean": float((r - n).mean()),
        "std": float((r - n).std()),
        "abs_mean": float(np.abs(r - n).mean()),
        "fraction_abs_gt_1e6": float((np.abs(r - n) > 1e-6).mean()),
    }
    return out


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------


def _historical_a_fit_metrics() -> dict[str, Any]:
    """Small read-only copy of historical A Fit diagnostics, if present.

    The historical ``analysis/runs/`` directory is git-ignored, so it is absent
    in a fresh worktree.  The final round records the values it actually has;
    Monitor comparison always uses the frozen A predictions, never refits A.
    """
    path = C.ANALYSIS / "historical_A_fit.json"
    if path.exists():
        return C.read_json(path)
    return {"available": False, "reason": "historical A run directory is git-ignored and absent in this worktree"}


def evaluate() -> dict[str, Any]:
    manifest = monitor_labels.verify_prediction_freeze()
    labels = monitor_labels.load_monitor_labels()
    y = np.asarray(labels["y"], dtype=np.float64)
    preds = load_predictions(manifest)

    mat_a = _arm_matrix(preds, "A")
    mat_r = _arm_matrix(preds, "R")
    mat_n = _arm_matrix(preds, "N")
    arms = {"A": _arm_summary(mat_a, y), "R": _arm_summary(mat_r, y), "N": _arm_summary(mat_n, y)}

    hist_metrics = C.read_json(C.HIST_METRICS)
    hist_a = hist_metrics["absolute"]["arms"]["A"]
    a_check = {}
    for i, seed in enumerate(C.SEEDS):
        expected = float(hist_a["per_seed"][f"seed_{seed}"]["rmse"])
        actual = arms["A"]["per_seed"][f"seed_{seed}"]["rmse"]
        a_check[f"seed_{seed}"] = {
            "historical_reported_rmse": expected,
            "recomputed_from_frozen_prediction": actual,
            "abs_diff": abs(actual - expected),
            "matches": abs(actual - expected) < 1e-9,
        }

    comparisons = {
        "R_minus_A": _comparison(mat_r, mat_a, y, name_a="R", name_b="A"),
        "N_minus_A": _comparison(mat_n, mat_a, y, name_a="N", name_b="A"),
        "R_minus_N": _comparison(mat_r, mat_n, y, name_a="R", name_b="N"),
    }
    # Seed direction counts for the pre-registered both-signal condition.
    r_better_a = [comparisons["R_minus_A"]["per_seed"][i]["delta_rmse"] < 0 for i in range(len(C.SEEDS))]
    r_better_n = [comparisons["R_minus_N"]["per_seed"][i]["delta_rmse"] < 0 for i in range(len(C.SEEDS))]
    both_seed = [bool(a and b) for a, b in zip(r_better_a, r_better_n)]

    mean_r_a = comparisons["R_minus_A"]["delta_rmse_mean"]
    mean_r_n = comparisons["R_minus_N"]["delta_rmse_mean"]
    mean_n_a = comparisons["N_minus_A"]["delta_rmse_mean"]
    mae_r = arms["R"]["mae_mean"]
    mae_a = arms["A"]["mae_mean"]
    both = (
        mean_r_a <= -C.MATERIALITY_RMSE
        and mean_r_n <= -C.MATERIALITY_RMSE
        and sum(both_seed) >= 2
        and (mae_r - mae_a) <= 0.01
    )
    if both:
        case = "BOTH_SIGNALS"
    elif mean_r_n <= -C.MATERIALITY_RMSE:
        case = "RELATION_SIGNAL_NOT_COMPETITIVE"
    elif mean_r_a <= -C.MATERIALITY_RMSE:
        case = "BOND_BRANCH_GAIN_NOT_ATTRIBUTABLE_TO_PAIRING"
    else:
        case = "NO_CLEAR_VALIDATION_SIGNAL"

    # Frozen-weight R -> N operator swap on the best R seed (by Monitor RMSE).
    best_i = int(np.argmin([arms["R"]["per_seed"][f"seed_{s}"]["rmse"] for s in C.SEEDS]))
    best_seed = int(C.SEEDS[best_i])
    r_pred = np.asarray(preds[f"R_s{best_seed}"]["pred"], dtype=np.float64)
    nswap = np.asarray(preds[f"R_s{best_seed}_nullswap"]["pred"], dtype=np.float64)
    movement = nswap - r_pred
    mechanism = {
        "best_R_seed_by_monitor_rmse": best_seed,
        "R_monitor_rmse": rmse(r_pred, y),
        "R_nullswap_monitor_rmse": rmse(nswap, y),
        "delta_rmse_nullswap_minus_R": rmse(nswap, y) - rmse(r_pred, y),
        "R_monitor_mae": mae(r_pred, y),
        "R_nullswap_monitor_mae": mae(nswap, y),
        "delta_mae_nullswap_minus_R": mae(nswap, y) - mae(r_pred, y),
        "mean_absolute_prediction_movement": float(np.abs(movement).mean()),
        "median_absolute_prediction_movement": float(np.median(np.abs(movement))),
        "fraction_molecules_moved_gt_1e6": float((np.abs(movement) > 1e-6).mean()),
        "movement_quantiles": _percentiles(np.abs(movement), (0.05, 0.5, 0.95)),
        "note": "frozen-weight mechanism diagnostic; not a causal performance estimate",
    }

    diagnostics = {
        "created_utc": C.utcnow(),
        "manifest_sha256": manifest["manifest_sha256"],
        "fit_and_monitor_metrics": {
            "scope": "Fit metrics are training diagnostics; Monitor metrics are the only comparison",
            "A": {
                "fit": _historical_a_fit_metrics(),
                "monitor": arms["A"]["per_seed"],
            },
            "R": {
                "fit": {
                    f"seed_{s}": {
                        "fit_rmse_raw": C.read_json(C.RUNS / f"R_s{s}" / "train.json")["fit_diagnostics"]["soup_fit_rmse_raw"],
                        "fit_mae_raw": C.read_json(C.RUNS / f"R_s{s}" / "train.json")["fit_diagnostics"]["soup_fit_mae_raw"],
                    } for s in C.SEEDS
                },
                "monitor": arms["R"]["per_seed"],
            },
            "N": {
                "fit": {
                    f"seed_{s}": {
                        "fit_rmse_raw": C.read_json(C.RUNS / f"N_s{s}" / "train.json")["fit_diagnostics"]["soup_fit_rmse_raw"],
                        "fit_mae_raw": C.read_json(C.RUNS / f"N_s{s}" / "train.json")["fit_diagnostics"]["soup_fit_mae_raw"],
                    } for s in C.SEEDS
                },
                "monitor": arms["N"]["per_seed"],
            },
        },
        "delta_bond_residuals": _new_branch_diagnostics(preds, mat_r, mat_n, y),
        "R_frozen_weights_swapped_to_N": mechanism,
        "bond_count_groups": _bond_count_groups(preds, y, mat_a, mat_r, mat_n),
        "top_A_error_contribution": {
            "top10": _top_error_contribution(mat_a, mat_r, mat_n, y, n_top=10),
            "top20": _top_error_contribution(mat_a, mat_r, mat_n, y, n_top=20),
            "definition": "improvement = MSE(A) - MSE(X); positive means X improves over A",
        },
        "remove_top5_influence": _remove_influential(mat_a, mat_r, mat_n, y, n_remove=5),
        "new_branch_nonconstant": _new_branch_diagnostics(preds, mat_r, mat_n, y),
        "seed_molecule_variance": {
            "A": _seed_molecule_variance(mat_a, y),
            "R": _seed_molecule_variance(mat_r, y),
            "N": _seed_molecule_variance(mat_n, y),
            "note": "seed variance and molecule variance are reported separately",
        },
        "resource_usage": {},
        "limitations": [
            "Monitor504 is a development split already used by SSRC Phase 1a; it is not an independent blind test.",
            "Reserved504 and official validation/test labels/structures were never accessed.",
            "Three seeds quantify training-seed variation only; molecule bootstrap intervals are separate.",
            "The R->N swap is a frozen-weight mechanism diagnostic, not a causal estimate of pairing benefit.",
        ],
    }
    for arm in C.ARMS:
        for seed in C.SEEDS:
            path = C.RUNS / f"{arm}_s{seed}" / "train.json"
            if path.exists():
                tr = C.read_json(path)
                diagnostics["resource_usage"][f"{arm}_s{seed}"] = {
                    "wall_seconds": tr.get("wall_seconds"),
                    "gpu_hours": tr.get("gpu_hours"),
                    "peak_gpu_memory": tr.get("peak_gpu_memory"),
                    "peak_rss_mb": tr.get("peak_rss_mb"),
                    "device": tr.get("device"),
                    "device_info": tr.get("device_info"),
                    "anomalies": tr.get("anomalies"),
                    "status": tr.get("status"),
                }

    result = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "manifest_sha256": manifest["manifest_sha256"],
        "metric_units": "raw exp / logD units",
        "target_normalisation": {"mean": 2.157857142857143, "scale": 1.2241982660328445, "source": "fit_labels_only"},
        "n_monitor": int(len(y)),
        "arms": arms,
        "historical_A_check": a_check,
        "comparisons": comparisons,
        "seed_direction": {
            "R_better_than_A": [bool(x) for x in r_better_a],
            "R_better_than_N": [bool(x) for x in r_better_n],
            "R_better_than_both": both_seed,
            "count_R_better_both": int(sum(both_seed)),
        },
        "decision": {
            "case": case,
            "materiality_rmse": C.MATERIALITY_RMSE,
            "mean_rmse_R_minus_A": mean_r_a,
            "mean_rmse_R_minus_N": mean_r_n,
            "mean_rmse_N_minus_A": mean_n_a,
            "mean_mae_R_minus_A": mae_r - mae_a,
            "both_signals_requires": {
                "mean_rmse_R_minus_A_le": -C.MATERIALITY_RMSE,
                "mean_rmse_R_minus_N_le": -C.MATERIALITY_RMSE,
                "at_least_2_of_3_seeds_R_better_than_both": True,
                "mean_mae_R_minus_A_le": 0.01,
            },
        },
        "labels_audit": labels["audit"],
        "monitor_labels_read": True,
        "reserved_labels_read": False,
        "official_valid_test_read": False,
    }
    C.write_json(C.ANALYSIS / "paired_metrics.json", result)
    C.write_json(C.ANALYSIS / "mechanism_diagnostics.json", diagnostics)
    _write_csvs(C.RESULTS, result, preds, y, mat_a, mat_r, mat_n)
    return result


def _write_csvs(results_dir: Path, result: Mapping[str, Any], preds, y, mat_a, mat_r, mat_n) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for arm in ("A", "R", "N"):
        for seed in C.SEEDS:
            m = result["arms"][arm]["per_seed"][f"seed_{seed}"]
            rows.append({
                "kind": "arm", "key": f"{arm}_s{seed}", "arm": arm, "seed": seed,
                "n": m["n"], "rmse": m["rmse"], "mae": m["mae"], "bias": m["bias"],
                "r2": m["r2"], "delta_rmse": "", "delta_mae": "", "note": "",
            })
    for name, comp in result["comparisons"].items():
        for row in comp["per_seed"]:
            rows.append({
                "kind": "delta", "key": name, "arm": comp["a"], "seed": row["seed"],
                "n": result["n_monitor"], "rmse": row["rmse_a"], "mae": row["mae_a"],
                "bias": "", "r2": "", "delta_rmse": row["delta_rmse"],
                "delta_mae": row["delta_mae"], "note": f"vs {comp['b']}",
            })
    with open(results_dir / "monitor_comparison.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["kind", "key", "arm", "seed", "n", "rmse", "mae", "bias", "r2", "delta_rmse", "delta_mae", "note"])
        writer.writeheader()
        writer.writerows(rows)

    fields = ["row", "global_row_id", "y"]
    for arm in ("A", "R", "N"):
        for seed in C.SEEDS:
            fields += [f"pred_{arm}_s{seed}", f"err_{arm}_s{seed}"]
    for seed in C.SEEDS:
        fields += [f"delta_R_minus_A_s{seed}", f"delta_R_minus_N_s{seed}"]
    with open(results_dir / "per_molecule_comparison.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        rows_ids = [int(r) for r in preds["A_s0"]["rows"]]
        for i, row_id in enumerate(rows_ids):
            rec = {"row": i, "global_row_id": row_id, "y": float(y[i])}
            for arm, mat in (("A", mat_a), ("R", mat_r), ("N", mat_n)):
                for j, seed in enumerate(C.SEEDS):
                    rec[f"pred_{arm}_s{seed}"] = float(mat[j, i])
                    rec[f"err_{arm}_s{seed}"] = float(mat[j, i] - y[i])
            for j, seed in enumerate(C.SEEDS):
                rec[f"delta_R_minus_A_s{seed}"] = float(mat_r[j, i] - mat_a[j, i])
                rec[f"delta_R_minus_N_s{seed}"] = float(mat_r[j, i] - mat_n[j, i])
            writer.writerow(rec)


__all__ = ["rmse", "mae", "metrics", "bootstrap_delta_rmse", "bootstrap_delta_mae", "load_predictions", "evaluate"]

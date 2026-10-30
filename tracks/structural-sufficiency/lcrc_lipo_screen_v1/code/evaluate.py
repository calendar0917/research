"""Monitor evaluation for the frozen LCRC Lipophilicity screen.

This is a **separate entry point** from training.  It can only run after
:func:`screen_predict.freeze_predictions` has written and hashed every
prediction, and it obtains Monitor labels only through :mod:`monitor_labels`.

Reported quantities
-------------------
* primary: raw-scale RMSE per seed, seed mean, seed spread;
* secondary: raw MAE, bias, R^2, paired per-molecule MSE/MAE comparisons;
* bootstrap CIs (molecule resampling) for absolute and paired metrics;
* a separation of training-seed variation from molecule-difficulty variation;
* tail sensitivity (share of error carried by the worst molecules, and whether
  a comparison flips when they are removed);
* arm A local-contribution / global-correction diagnostics.

Decision rules are frozen in ``PROTOCOL.md`` and ``DECISION.md``; this module
only *applies* them, it does not choose them after seeing the numbers.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import monitor_labels
import screen_common as C
import screen_data as D


# ---------------------------------------------------------------------------
# basic metrics
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


# ---------------------------------------------------------------------------
# bootstrap
# ---------------------------------------------------------------------------


def _indices(n: int, resamples: int, seed: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.integers(0, int(n), size=(int(resamples), int(n)))


def bootstrap_ci_mean(values: np.ndarray, resamples: int, seed: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=np.float64)
    idx = _indices(values.size, resamples, seed)
    draws = values[idx].mean(axis=1)
    return {
        "estimate": float(values.mean()),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "resamples": int(resamples),
        "seed": int(seed),
        "unit": "molecule",
    }


def bootstrap_rmse(pred: np.ndarray, y: np.ndarray, resamples: int, seed: int) -> dict[str, Any]:
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    idx = _indices(y.size, resamples, seed)
    e = pred[idx] - y[idx]
    draws = np.sqrt(np.mean(e * e, axis=1))
    return {
        "estimate": rmse(pred, y),
        "ci95_low": float(np.percentile(draws, 2.5)),
        "ci95_high": float(np.percentile(draws, 97.5)),
        "resamples": int(resamples),
        "seed": int(seed),
        "unit": "molecule",
    }


def bootstrap_delta_rmse(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    y: np.ndarray,
    resamples: int,
    seed: int,
) -> dict[str, Any]:
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


# ---------------------------------------------------------------------------
# prediction loading
# ---------------------------------------------------------------------------


def load_frozen_predictions(manifest: Mapping[str, Any]) -> dict[str, dict[str, np.ndarray]]:
    rows = [int(r) for r in manifest["rows"]]
    out: dict[str, dict[str, np.ndarray]] = {}
    for key, entry in sorted(manifest["predictions"].items()):
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
# evaluation
# ---------------------------------------------------------------------------


def _nn_prediction_matrix(preds: Mapping[str, Mapping[str, np.ndarray]], arm: str) -> np.ndarray:
    return np.stack([np.asarray(preds[f"{arm}_s{s}"]["pred"], dtype=np.float64) for s in C.SEEDS], axis=0)


def _baseline_vector(preds, r: int) -> np.ndarray:
    return np.asarray(preds[f"D_r{r}"]["pred"], dtype=np.float64)


def _arm_metrics(mat: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    per_seed = [metrics(mat[i], y) for i in range(mat.shape[0])]
    rmses = np.asarray([m["rmse"] for m in per_seed], dtype=np.float64)
    maes = np.asarray([m["mae"] for m in per_seed], dtype=np.float64)
    seed_mean_pred = mat.mean(axis=0)
    return {
        "per_seed": {f"seed_{s}": per_seed[i] for i, s in enumerate(C.SEEDS)},
        "r2_mean": float(np.mean([m["r2"] for m in per_seed])),
        "rmse_mean": float(rmses.mean()),
        "rmse_std": float(rmses.std(ddof=0)),
        "rmse_min": float(rmses.min()),
        "rmse_max": float(rmses.max()),
        "rmse_seed_spread": float(rmses.max() - rmses.min()),
        "mae_mean": float(maes.mean()),
        "mae_std": float(maes.std(ddof=0)),
        "seed_mean_prediction_rmse": rmse(seed_mean_pred, y),
        "seed_mean_prediction_mae": mae(seed_mean_pred, y),
        "best_seed_rmse": float(rmses.min()),
        "best_seed_minus_mean_rmse": float(rmses.min() - rmses.mean()),
        "rmses": rmses.tolist(),
        "maes": maes.tolist(),
    }


def _seed_molecule_decomposition(mat: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    """Separate training-seed variance from between-molecule variance."""
    mat = np.asarray(mat, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    errors = (mat - y[None, :]) ** 2
    within = errors.var(axis=0, ddof=0).mean()
    between = errors.mean(axis=0).var(ddof=0)
    total = within + between
    pred_within = mat.var(axis=0, ddof=0).mean()
    pred_between = mat.mean(axis=0).var(ddof=0)
    # Rank stability of per-molecule difficulty across seeds.
    ranks = np.argsort(np.argsort(errors, axis=1), axis=1)
    pairs = []
    for i in range(mat.shape[0]):
        for j in range(i + 1, mat.shape[0]):
            pairs.append(np.corrcoef(ranks[i], ranks[j])[0, 1])
    return {
        "squared_error": {
            "within_molecule_seed_variance": float(within),
            "between_molecule_variance": float(between),
            "total": float(total),
            "seed_share": float(within / total) if total > 0 else None,
            "molecule_share": float(between / total) if total > 0 else None,
        },
        "prediction": {
            "within_molecule_seed_variance": float(pred_within),
            "between_molecule_variance": float(pred_between),
            "seed_share": float(pred_within / (pred_within + pred_between))
            if (pred_within + pred_between) > 0
            else None,
        },
        "difficulty_rank_correlation_mean": float(np.mean(pairs)) if pairs else None,
        "difficulty_rank_correlation_min": float(np.min(pairs)) if pairs else None,
    }


def _paired(
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    y: np.ndarray,
    *,
    resamples: int,
    seed: int,
    label_a: str,
    label_b: str,
) -> dict[str, Any]:
    pred_a = np.asarray(pred_a, dtype=np.float64)
    pred_b = np.asarray(pred_b, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ea = pred_a - y
    eb = pred_b - y
    delta_mse = ea * ea - eb * eb
    delta_mae = np.abs(ea) - np.abs(eb)
    return {
        "a": label_a,
        "b": label_b,
        "rmse_a": rmse(pred_a, y),
        "rmse_b": rmse(pred_b, y),
        "delta_rmse": rmse(pred_a, y) - rmse(pred_b, y),
        "delta_mae": mae(pred_a, y) - mae(pred_b, y),
        "delta_rmse_bootstrap": bootstrap_delta_rmse(pred_a, pred_b, y, resamples, seed),
        "delta_mse_per_molecule": {
            "mean": float(delta_mse.mean()),
            "ci95": bootstrap_ci_mean(delta_mse, resamples, seed),
            "quantiles": _percentiles(delta_mse, (0.05, 0.25, 0.5, 0.75, 0.95)),
            "win_rate_a_better": float((delta_mse < 0).mean()),
            "tie_rate": float((delta_mse == 0).mean()),
        },
        "delta_mae_per_molecule": {
            "mean": float(delta_mae.mean()),
            "ci95": bootstrap_ci_mean(delta_mae, resamples, seed),
            "quantiles": _percentiles(delta_mae, (0.05, 0.25, 0.5, 0.75, 0.95)),
            "win_rate_a_better": float((delta_mae < 0).mean()),
        },
        "definition": "delta = metric(A) - metric(B); negative favours A",
    }


def _tail_statistics(mat: np.ndarray, y: np.ndarray, top_fractions=(0.01, 0.02, 0.05)) -> dict[str, Any]:
    errors = ((mat - y[None, :]) ** 2).mean(axis=0)
    order = np.argsort(errors)[::-1]
    total = float(errors.sum())
    out = {"n_molecules": int(errors.size), "total_mean_squared_error": total}
    for frac in top_fractions:
        k = max(1, int(round(frac * errors.size)))
        out[f"top_{int(round(frac * 100))}pct_share_of_squared_error"] = float(errors[order[:k]].sum() / total)
        out[f"top_{int(round(frac * 100))}pct_n"] = int(k)
    out["rmse_all"] = float(np.sqrt(errors.mean()))
    for frac in (0.02, 0.05):
        k = max(1, int(round(frac * errors.size)))
        keep = order[k:]
        out[f"rmse_excluding_top_{int(round(frac * 100))}pct"] = float(np.sqrt(errors[keep].mean()))
    out["worst_molecules_seed_mean_squared_error"] = [
        {"index": int(i), "squared_error": float(errors[i])} for i in order[:10]
    ]
    return out


def _contribution_diagnostics(preds: Mapping[str, Any], monitor_rows: Sequence[int], caches: Mapping[str, Any]) -> dict[str, Any]:
    n_atoms = D.atom_counts(caches["monitor"]["cache"])
    out: dict[str, Any] = {"per_seed": {}, "note": "descriptive only; not a causal or faithful explanation"}
    for s in C.SEEDS:
        payload = preds[f"A_s{s}"]
        contrib = np.asarray(payload["contrib"], dtype=np.float64)
        rho = np.asarray(payload["rho"], dtype=np.float64)
        pred = np.asarray(payload["pred"], dtype=np.float64)
        consistency = float(np.abs(contrib + rho - pred).max())
        ratio = np.abs(rho) / np.maximum(np.abs(contrib), 1e-8)
        per_atom = contrib / n_atoms
        out["per_seed"][f"seed_{s}"] = {
            "decomposition_max_abs_error": consistency,
            "contrib": {
                "mean": float(contrib.mean()),
                "std": float(contrib.std()),
                "min": float(contrib.min()),
                "max": float(contrib.max()),
                "quantiles": _percentiles(contrib, (0.05, 0.25, 0.5, 0.75, 0.95)),
            },
            "rho": {
                "mean": float(rho.mean()),
                "std": float(rho.std()),
                "min": float(rho.min()),
                "max": float(rho.max()),
                "quantiles": _percentiles(rho, (0.05, 0.25, 0.5, 0.75, 0.95)),
            },
            "contrib_per_atom": {
                "mean": float(per_atom.mean()),
                "std": float(per_atom.std()),
                "quantiles": _percentiles(per_atom, (0.05, 0.5, 0.95)),
            },
            "abs_rho_over_abs_contrib": {
                "median": float(np.median(ratio)),
                "quantiles": _percentiles(ratio, (0.05, 0.25, 0.5, 0.75, 0.95)),
                "fraction_rho_dominates": float((np.abs(rho) > np.abs(contrib)).mean()),
            },
            "variance_share": {
                "contrib": float(np.var(contrib) / (np.var(contrib) + np.var(rho))),
                "rho": float(np.var(rho) / (np.var(contrib) + np.var(rho))),
                "covariance_term": float(2 * np.cov(contrib, rho)[0, 1] / np.var(pred))
                if np.var(pred) > 0
                else None,
            },
            "corr_contrib_rho": float(np.corrcoef(contrib, rho)[0, 1]),
            "corr_atom_count_contrib": float(np.corrcoef(n_atoms, contrib)[0, 1]),
            "corr_atom_count_rho": float(np.corrcoef(n_atoms, rho)[0, 1]),
        }
    out["across_seeds"] = {
        "contrib_mean_std": float(np.std([out["per_seed"][f"seed_{s}"]["contrib"]["mean"] for s in C.SEEDS])),
        "rho_mean_std": float(np.std([out["per_seed"][f"seed_{s}"]["rho"]["mean"] for s in C.SEEDS])),
        "decomposition_max_abs_error": float(
            max(out["per_seed"][f"seed_{s}"]["decomposition_max_abs_error"] for s in C.SEEDS)
        ),
    }
    return out




# ---------------------------------------------------------------------------
# frozen classification rules (testable without touching Monitor labels)
# ---------------------------------------------------------------------------


def classify_comparison(delta_rmse: float, ci_low: float, ci_high: float, materiality: float) -> str:
    """A vs baseline: negative delta favours A."""
    if delta_rmse <= -materiality and ci_high < 0:
        return "A_MATERIALLY_BETTER"
    if delta_rmse <= materiality and ci_high <= materiality:
        return "COMPETITIVE"
    if delta_rmse > materiality and ci_low > 0:
        return "A_MATERIALLY_WORSE"
    return "INCONCLUSIVE"


def classify_s2(per_seed_deltas: Sequence[float], mean_delta: float, ci_high: float, materiality: float) -> str:
    all_negative = bool(all(float(d) < 0 for d in per_seed_deltas))
    ci_excludes_zero = bool(float(ci_high) < 0)
    if all_negative and ci_excludes_zero and mean_delta <= -materiality:
        return "CLEAR_INCREMENT"
    if all_negative and ci_excludes_zero:
        return "SUGGESTIVE_NOT_MATERIAL"
    return "NO_CLEAR_INCREMENT"


def classify_case(learned: bool, s2_state: str, vs_c_state: str, vs_d_state: str) -> str:
    competitive = vs_c_state != "A_MATERIALLY_WORSE" and vs_d_state != "A_MATERIALLY_WORSE"
    if learned and s2_state == "CLEAR_INCREMENT" and competitive:
        return "CASE_1_S2_INCREMENT_SUPPORTED"
    if learned and competitive:
        return "CASE_2_LCRC_COMPETITIVE_NO_CLEAR_S2_INCREMENT"
    return "CASE_3_A_WEAKER_OR_DID_NOT_LEARN"


def evaluate(
    manifest_path: Path | str = C.PREDICTION_MANIFEST,
    lock_path: Path | str = C.PREDICTION_LOCK,
) -> dict[str, Any]:
    manifest = monitor_labels.verify_prediction_freeze(manifest_path, lock_path)
    labels = monitor_labels.load_monitor_labels(manifest_path=manifest_path, lock_path=lock_path)
    y = np.asarray(labels["y"], dtype=np.float64)
    preds = load_frozen_predictions(manifest)
    caches = D.load_caches()
    cfg = C.load_config()
    resamples = int(cfg["evaluation"]["bootstrap_resamples"])
    boot_seed = int(cfg["evaluation"]["bootstrap_seed"])
    materiality = float(cfg["evaluation"]["materiality_rmse"])
    target_norm = D.load_target_norm(verify=False)
    fit_mean = float(target_norm["mean"])

    # --- absolute metrics -------------------------------------------------
    arm_mats = {arm: _nn_prediction_matrix(preds, arm) for arm in C.NN_ARMS}
    arm_metrics = {arm: _arm_metrics(arm_mats[arm], y) for arm in C.NN_ARMS}
    baseline_metrics = {}
    for r in C.BASELINE_REPLICATES:
        m = metrics(_baseline_vector(preds, r), y)
        m["random_state"] = int(r)
        baseline_metrics[f"D_r{r}"] = m

    constant_pred = np.full_like(y, fit_mean)
    constant_metrics = metrics(constant_pred, y)
    constant_metrics["predictor"] = "Fit2352 label mean (constant)"
    constant_metrics["fit_mean"] = fit_mean
    constant_metrics["note"] = "baseline computed without reading any Monitor label beyond the evaluation itself"

    # --- absolute bootstrap CIs ------------------------------------------
    absolute_bootstrap = {}
    for arm in C.NN_ARMS:
        absolute_bootstrap[arm] = bootstrap_rmse(arm_mats[arm].mean(axis=0), y, resamples, boot_seed)
    for r in C.BASELINE_REPLICATES:
        absolute_bootstrap[f"D_r{r}"] = bootstrap_rmse(_baseline_vector(preds, r), y, resamples, boot_seed)

    # --- paired comparisons ---------------------------------------------
    paired: dict[str, Any] = {"primary": {}, "secondary": {}, "definitions": {}}
    a_mean = arm_mats["A"].mean(axis=0)
    d_primary = _baseline_vector(preds, C.PRIMARY_BASELINE_REPLICATE)
    paired["primary"]["A_vs_B_seedmean"] = _paired(
        a_mean, arm_mats["B"].mean(axis=0), y, resamples=resamples, seed=boot_seed,
        label_a="A_seed_mean", label_b="B_seed_mean",
    )
    paired["primary"]["A_vs_C_seedmean"] = _paired(
        a_mean, arm_mats["C"].mean(axis=0), y, resamples=resamples, seed=boot_seed,
        label_a="A_seed_mean", label_b="C_seed_mean",
    )
    paired["primary"]["A_vs_D_primary"] = _paired(
        a_mean, d_primary, y, resamples=resamples, seed=boot_seed,
        label_a="A_seed_mean", label_b="D_r0",
    )
    paired["primary"]["B_vs_C_seedmean"] = _paired(
        arm_mats["B"].mean(axis=0), arm_mats["C"].mean(axis=0), y, resamples=resamples, seed=boot_seed,
        label_a="B_seed_mean", label_b="C_seed_mean",
    )
    paired["primary"]["A_vs_constant_fit_mean"] = _paired(
        a_mean, constant_pred, y, resamples=resamples, seed=boot_seed,
        label_a="A_seed_mean", label_b="fit_mean_constant",
    )
    for s in C.SEEDS:
        paired["secondary"][f"A_vs_B_seed{s}"] = _paired(
            arm_mats["A"][s], arm_mats["B"][s], y, resamples=resamples, seed=boot_seed,
            label_a=f"A_s{s}", label_b=f"B_s{s}",
        )
        paired["secondary"][f"A_vs_C_seed{s}"] = _paired(
            arm_mats["A"][s], arm_mats["C"][s], y, resamples=resamples, seed=boot_seed,
            label_a=f"A_s{s}", label_b=f"C_s{s}",
        )
        paired["secondary"][f"A_vs_D_matched_seed{s}"] = _paired(
            arm_mats["A"][s], _baseline_vector(preds, s), y, resamples=resamples, seed=boot_seed,
            label_a=f"A_s{s}", label_b=f"D_r{s}",
        )
    paired["secondary"]["D_seed_spread"] = {
        f"D_r{r}": metrics(_baseline_vector(preds, r), y) for r in C.BASELINE_REPLICATES
    }

    # --- seed vs molecule variation --------------------------------------
    decomposition = {arm: _seed_molecule_decomposition(arm_mats[arm], y) for arm in C.NN_ARMS}

    # --- tail sensitivity ------------------------------------------------
    tails = {arm: _tail_statistics(arm_mats[arm], y) for arm in C.NN_ARMS}
    tails["D_r0"] = _tail_statistics(d_primary[None, :], y)
    tail_flip = {}
    for arm in ("B", "C"):
        errors = ((arm_mats["A"] - y[None, :]) ** 2).mean(axis=0) - ((arm_mats[arm] - y[None, :]) ** 2).mean(axis=0)
        order = np.argsort(errors)[::-1]
        delta_all = float(errors.mean())
        for frac in (0.02, 0.05):
            k = max(1, int(round(frac * errors.size)))
            keep = order[k:]
            tail_flip[f"A_vs_{arm}_drop_top_{int(round(frac * 100))}pct"] = {
                "delta_mse_all": delta_all,
                "delta_mse_excluding": float(errors[keep].mean()),
                "sign_flip": bool(np.sign(delta_all) != np.sign(errors[keep].mean())),
            }

    # --- arm A contribution diagnostics ---------------------------------
    contributions = _contribution_diagnostics(preds, monitor_rows=manifest["rows"], caches=caches)

    # --- frozen decision rules ------------------------------------------
    s2_deltas = [paired["secondary"][f"A_vs_B_seed{s}"]["delta_rmse"] for s in C.SEEDS]
    s2_mean_delta = float(np.mean(s2_deltas))
    s2_ci = paired["primary"]["A_vs_B_seedmean"]["delta_rmse_bootstrap"]
    s2_all_negative = all(d < 0 for d in s2_deltas)
    s2_material = s2_mean_delta <= -materiality
    s2_ci_excludes_zero = s2_ci["ci95_high"] < 0
    s2_state = classify_s2(s2_deltas, s2_mean_delta, s2_ci["ci95_high"], materiality)

    def comparison_state(name: str) -> dict[str, Any]:
        p = paired["primary"][name]
        delta = p["delta_rmse"]
        ci = p["delta_rmse_bootstrap"]
        state = classify_comparison(delta, ci["ci95_low"], ci["ci95_high"], materiality)
        return {"state": state, "delta_rmse": delta, "materiality": materiality, "bootstrap": ci}

    vs_c = comparison_state("A_vs_C_seedmean")
    vs_d = comparison_state("A_vs_D_primary")
    per_seed_learned = {
        f"seed_{s}": bool(arm_metrics["A"]["per_seed"][f"seed_{s}"]["rmse"] < constant_metrics["rmse"])
        for s in C.SEEDS
    }
    learned = bool(arm_metrics["A"]["rmse_mean"] < constant_metrics["rmse"] and all(per_seed_learned.values()))
    seed_sensitive = bool(arm_metrics["A"]["rmse_seed_spread"] > materiality)
    tail_sensitive = bool(
        tails["A"]["top_2pct_share_of_squared_error"] > 0.30
        or any(v["sign_flip"] for v in tail_flip.values())
    )
    case = classify_case(learned, s2_state, vs_c["state"], vs_d["state"])

    decision = {
        "frozen_rules_version": "lcrc-lipo-screen-v1",
        "materiality_rmse": materiality,
        "a_learned_beyond_fit_mean": learned,
        "a_learned_per_seed": per_seed_learned,
        "a_r2_mean": arm_metrics["A"]["r2_mean"],
        "fit_mean_constant_rmse": constant_metrics["rmse"],
        "s2_increment": {
            "state": s2_state,
            "per_seed_delta_rmse": s2_deltas,
            "mean_delta_rmse": s2_mean_delta,
            "all_seeds_negative": s2_all_negative,
            "mean_delta_material": s2_material,
            "bootstrap_excludes_zero": s2_ci_excludes_zero,
            "bootstrap": s2_ci,
            "materiality_rmse": materiality,
            "nominal_vs_effective_parameters": {
                "A": manifest["predictions"]["A_s0"]["parameters"],
                "B": manifest["predictions"]["B_s0"]["parameters"],
                "note": "B has the same nominal count but a smaller effective count; not information-matched",
            },
        },
        "vs_gine": vs_c,
        "vs_ecfp4_xgb": vs_d,
        "seed_sensitive": seed_sensitive,
        "tail_sensitive": tail_sensitive,
        "case": case,
        "notes": [
            "The Monitor 504 set is a development set: SSRC Phase 1a already used it.",
            "Reserved 504 and official valid/test labels were never read and are not scored.",
            "A vs B is a channel ablation with equal nominal but unequal effective capacity.",
            "A vs C and A vs D are practical comparisons, not information-matched controls.",
        ],
    }

    evaluation = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "manifest_sha256": manifest["manifest_sha256"],
        "labels": {
            "labels_sha256": labels["labels_sha256"],
            "monitor_label_vector_sha256": labels["audit"]["monitor_label_vector_sha256"],
            "n": int(y.size),
            "split": "monitor",
            "reserved_labels_read": False,
            "official_valid_test_read": False,
            "audit": labels["audit"],
        },
        "absolute": {
            "arms": arm_metrics,
            "baseline_D": baseline_metrics,
            "fit_mean_constant": constant_metrics,
            "bootstrap": absolute_bootstrap,
        },
        "paired": paired,
        "diagnostics": {
            "seed_vs_molecule": decomposition,
            "tail_sensitivity": tails,
            "tail_comparison_flip": tail_flip,
            "arm_A_contribution": contributions,
        },
        "decision": decision,
        "target_normalisation": {
            "source": target_norm["source"],
            "mean": target_norm["mean"],
            "scale": target_norm["scale"],
        },
        "metric_units": "raw exp / logD units (predictions denormalised with Fit-only mean/std)",
    }

    C.write_json(C.ANALYSIS / "monitor_metrics.json", {
        k: evaluation[k] for k in ("protocol_id", "created_utc", "git_commit", "manifest_sha256", "labels", "absolute", "decision", "target_normalisation", "metric_units")
    })
    C.write_json(C.ANALYSIS / "monitor_paired.json", {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": evaluation["created_utc"],
        "manifest_sha256": manifest["manifest_sha256"],
        "paired": paired,
    })
    C.write_json(C.ANALYSIS / "monitor_diagnostics.json", {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": evaluation["created_utc"],
        "manifest_sha256": manifest["manifest_sha256"],
        "diagnostics": evaluation["diagnostics"],
        "decision": decision,
    })
    C.log(
        f"[evaluate] A rmse={arm_metrics['A']['rmse_mean']:.5f} B={arm_metrics['B']['rmse_mean']:.5f} "
        f"C={arm_metrics['C']['rmse_mean']:.5f} D={baseline_metrics['D_r0']['rmse']:.5f} "
        f"constant={constant_metrics['rmse']:.5f} case={case} s2={s2_state}"
    )
    return evaluation


__all__ = [
    "classify_comparison",
    "classify_s2",
    "classify_case",
    "rmse",
    "mae",
    "metrics",
    "bootstrap_ci_mean",
    "bootstrap_rmse",
    "bootstrap_delta_rmse",
    "load_frozen_predictions",
    "evaluate",
]

"""Phase B estimator: nested grouped CV, three logistic models, frozen gates.

The three compared models differ **only** in the 126-d block appended to the
shared baseline:

=============  ==================================================
``BASE``       ECFP4 (0/1, 2048) + standardised ``g42``
``BASE_NULL``  BASE + NULL126 (marginal-matched independence null)
``BASE_REAL``  BASE + REAL126 (type-pair x distance-bin statistics)
=============  ==================================================

Everything is CPU, deterministic and pre-registered in ``PROTOCOL.md``.  The
regularisation strength is chosen on ``BASE`` alone (inner two-fold mean
ROC-AUC, ties -> smaller ``C``) and then reused for all three branches of that
outer fold, so neither NULL nor REAL can influence its own comparison.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, roc_auc_score

import nb_common as C


# ---------------------------------------------------------------------------
# preprocessing (fitted on one training partition only)
# ---------------------------------------------------------------------------


def fit_preprocessing(
    rows: Sequence[int],
    ecfp: np.ndarray,
    g: np.ndarray,
    real: np.ndarray,
    null: np.ndarray,
) -> dict[str, Any]:
    """Fit every scaler on the given training partition.

    ``g42``: per-column mean / population std of the partition.
    ``REAL``/``NULL``: one **shared** scaler fitted on the vertical concatenation of
    the partition's REAL and NULL rows, so both branches get the identical affine
    map.  A zero-variance column uses denominator ``1.0``.
    """
    idx = np.asarray(sorted(int(i) for i in rows), dtype=np.int64)
    g_train = np.asarray(g, dtype=np.float64)[idx]
    g_mean = g_train.mean(axis=0)
    g_std = g_train.std(axis=0)
    g_scale = np.where(g_std > 0.0, g_std, 1.0)

    pair_train = np.vstack(
        [np.asarray(real, dtype=np.float64)[idx], np.asarray(null, dtype=np.float64)[idx]]
    )
    p_mean = pair_train.mean(axis=0)
    p_std = pair_train.std(axis=0)
    p_scale = np.where(p_std > 0.0, p_std, 1.0)
    return {
        "n_fit_rows": int(idx.size),
        "g_mean": g_mean,
        "g_scale": g_scale,
        "g_zero_variance_columns": int((g_std <= 0.0).sum()),
        "pair_mean": p_mean,
        "pair_scale": p_scale,
        "pair_zero_variance_columns": int((p_std <= 0.0).sum()),
        "pair_scale_fit_on": "REAL and NULL of the same partition concatenated (shared affine map)",
        "clip": C.CLIP,
    }


def transform(
    rows: Sequence[int],
    pre: Mapping[str, Any],
    branch: str,
    ecfp: np.ndarray,
    g: np.ndarray,
    real: np.ndarray,
    null: np.ndarray,
) -> np.ndarray:
    idx = np.asarray(list(rows), dtype=np.int64)
    base = np.concatenate(
        [
            np.asarray(ecfp, dtype=np.float64)[idx],
            np.clip((np.asarray(g, dtype=np.float64)[idx] - pre["g_mean"]) / pre["g_scale"], -C.CLIP, C.CLIP),
        ],
        axis=1,
    )
    if branch == "BASE":
        return base
    source = real if branch == "BASE_REAL" else null
    block = np.clip(
        (np.asarray(source, dtype=np.float64)[idx] - pre["pair_mean"]) / pre["pair_scale"], -C.CLIP, C.CLIP
    )
    return np.concatenate([base, block], axis=1)


def feature_widths(n_ecfp: int) -> dict[str, int]:
    return {
        "ecfp4": int(n_ecfp),
        "g42": int(C.GLOBAL_DIM),
        "pair": int(C.N_PAIR_FEATURES),
        "BASE": int(n_ecfp + C.GLOBAL_DIM),
        "BASE_NULL": int(n_ecfp + C.GLOBAL_DIM + C.N_PAIR_FEATURES),
        "BASE_REAL": int(n_ecfp + C.GLOBAL_DIM + C.N_PAIR_FEATURES),
    }


# ---------------------------------------------------------------------------
# estimator
# ---------------------------------------------------------------------------


def fit_logreg(X: np.ndarray, y: np.ndarray, c_value: float) -> LogisticRegression:
    model = LogisticRegression(
        C=float(c_value),
        penalty=C.LOGREG["penalty"],
        solver=C.LOGREG["solver"],
        class_weight=C.LOGREG["class_weight"],
        fit_intercept=C.LOGREG["fit_intercept"],
        tol=C.LOGREG["tol"],
        max_iter=C.LOGREG["max_iter"],
        random_state=C.LOGREG["random_state"],
    )
    model.fit(X, np.asarray(y, dtype=np.int64))
    if int(model.n_iter_[0]) >= int(C.LOGREG["max_iter"]):
        raise RuntimeError(f"liblinear did not converge (n_iter={int(model.n_iter_[0])})")
    return model


def positive_score(model: LogisticRegression, X: np.ndarray) -> np.ndarray:
    return model.predict_proba(X)[:, 1].astype(np.float64)


def select_c(
    outer_fold: Mapping[str, Any],
    y: np.ndarray,
    features: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """Inner 2-fold mean ROC-AUC of ``BASE`` only; ties -> smaller ``C``."""
    table: dict[str, Any] = {}
    means: dict[float, float] = {}
    for c_value in C.C_GRID:
        aucs = []
        apss = []
        for inner in outer_fold["inner"]:
            tr = inner["train_rows"]
            va = inner["valid_rows"]
            pre = fit_preprocessing(tr, features["ecfp4"], features["g"], features["real"], features["null"])
            X = transform(tr, pre, "BASE", features["ecfp4"], features["g"], features["real"], features["null"])
            Xv = transform(va, pre, "BASE", features["ecfp4"], features["g"], features["real"], features["null"])
            model = fit_logreg(X, y[np.asarray(tr)], c_value)
            score = positive_score(model, Xv)
            aucs.append(float(roc_auc_score(y[np.asarray(va)], score)))
            apss.append(float(average_precision_score(y[np.asarray(va)], score)))
        means[float(c_value)] = float(np.mean(aucs))
        table[f"{c_value:g}"] = {"inner_auc": aucs, "inner_ap": apss, "mean_inner_auc": means[float(c_value)]}
    # ties -> smaller C
    best_c = None
    best_mean = -np.inf
    for c_value in sorted(C.C_GRID):
        if means[float(c_value)] > best_mean + 1e-12:
            best_mean = means[float(c_value)]
            best_c = float(c_value)
    return {
        "grid": list(C.C_GRID),
        "table": table,
        "selected_C": float(best_c),
        "selected_mean_inner_auc": float(best_mean),
        "selection_uses": "BASE only",
        "tie_break": "smaller C (grid order ascending, strict improvement required)",
    }


def run_outer_fold(
    outer_fold: Mapping[str, Any],
    y: np.ndarray,
    features: Mapping[str, np.ndarray],
    selection: Mapping[str, Any],
) -> dict[str, Any]:
    tr = outer_fold["train_rows"]
    va = outer_fold["valid_rows"]
    c_value = float(selection["selected_C"])
    pre = fit_preprocessing(tr, features["ecfp4"], features["g"], features["real"], features["null"])
    y_tr = y[np.asarray(tr)]
    y_va = y[np.asarray(va)]
    out: dict[str, Any] = {
        "outer_fold": int(outer_fold["outer_fold"]),
        "selected_C": c_value,
        "n_train": int(len(tr)),
        "n_valid": int(len(va)),
        "n_positive_train": int((y_tr == 1).sum()),
        "n_positive_valid": int((y_va == 1).sum()),
        "preprocessing": {
            "n_fit_rows": pre["n_fit_rows"],
            "g_zero_variance_columns": pre["g_zero_variance_columns"],
            "pair_zero_variance_columns": pre["pair_zero_variance_columns"],
            "clip": pre["clip"],
        },
        "branches": {},
    }
    scores: dict[str, np.ndarray] = {}
    for branch in C.BRANCHES:
        X = transform(tr, pre, branch, features["ecfp4"], features["g"], features["real"], features["null"])
        Xv = transform(va, pre, branch, features["ecfp4"], features["g"], features["real"], features["null"])
        model = fit_logreg(X, y_tr, c_value)
        score = positive_score(model, Xv)
        scores[branch] = score
        out["branches"][branch] = {
            "width": int(X.shape[1]),
            "n_iter": int(model.n_iter_[0]),
            "roc_auc": float(roc_auc_score(y_va, score)),
            "average_precision": float(average_precision_score(y_va, score)),
            "score_mean": float(score.mean()),
            "score_std": float(score.std()),
            "score_min": float(score.min()),
            "score_max": float(score.max()),
            "score_quantiles": {
                q: float(np.quantile(score, float(q))) for q in ("0.05", "0.25", "0.5", "0.75", "0.95")
            },
            "score_mean_positive": float(score[y_va == 1].mean()),
            "score_mean_negative": float(score[y_va == 0].mean()),
            "n_predicted_positive_at_0.5": int((score >= 0.5).sum()),
        }
    out["scores"] = scores
    return out


# ---------------------------------------------------------------------------
# aggregate metrics
# ---------------------------------------------------------------------------


def aggregate(fold_results: Sequence[Mapping[str, Any]], y: np.ndarray) -> dict[str, Any]:
    folds = sorted(fold_results, key=lambda r: r["outer_fold"])
    per_fold = {
        "roc_auc": {b: [r["branches"][b]["roc_auc"] for r in folds] for b in C.BRANCHES},
        "average_precision": {b: [r["branches"][b]["average_precision"] for r in folds] for b in C.BRANCHES},
    }
    mean_auc = {b: float(np.mean(per_fold["roc_auc"][b])) for b in C.BRANCHES}
    mean_ap = {b: float(np.mean(per_fold["average_precision"][b])) for b in C.BRANCHES}
    diff_auc_real_null = [float(r["branches"]["BASE_REAL"]["roc_auc"] - r["branches"]["BASE_NULL"]["roc_auc"]) for r in folds]
    diff_auc_real_base = [float(r["branches"]["BASE_REAL"]["roc_auc"] - r["branches"]["BASE"]["roc_auc"]) for r in folds]
    diff_ap_real_null = [
        float(r["branches"]["BASE_REAL"]["average_precision"] - r["branches"]["BASE_NULL"]["average_precision"])
        for r in folds
    ]
    diff_ap_real_base = [
        float(r["branches"]["BASE_REAL"]["average_precision"] - r["branches"]["BASE"]["average_precision"])
        for r in folds
    ]
    return {
        "primary_metric": "mean over the three outer folds of ROC-AUC on the held-out outer validation fold",
        "per_fold": {
            "roc_auc": per_fold["roc_auc"],
            "average_precision": per_fold["average_precision"],
            "selected_C": [r["selected_C"] for r in folds],
            "n_valid": [r["n_valid"] for r in folds],
            "n_positive_valid": [r["n_positive_valid"] for r in folds],
        },
        "mean_roc_auc": mean_auc,
        "mean_average_precision": mean_ap,
        "per_fold_roc_auc_diff": {
            "REAL_minus_NULL": diff_auc_real_null,
            "REAL_minus_BASE": diff_auc_real_base,
        },
        "per_fold_ap_diff": {
            "REAL_minus_NULL": diff_ap_real_null,
            "REAL_minus_BASE": diff_ap_real_base,
        },
        "mean_roc_auc_diff": {
            "REAL_minus_NULL": float(np.mean(diff_auc_real_null)),
            "REAL_minus_BASE": float(np.mean(diff_auc_real_base)),
            "NULL_minus_BASE": float(mean_auc["BASE_NULL"] - mean_auc["BASE"]),
        },
        "mean_ap_diff": {
            "REAL_minus_NULL": float(np.mean(diff_ap_real_null)),
            "REAL_minus_BASE": float(np.mean(diff_ap_real_base)),
            "NULL_minus_BASE": float(mean_ap["BASE_NULL"] - mean_ap["BASE"]),
        },
        "n_positive_total": int((y == 1).sum()),
        "n_negative_total": int((y == 0).sum()),
        "prediction_distributions": {
            r["outer_fold"]: {b: {
                "score_mean": r["branches"][b]["score_mean"],
                "score_std": r["branches"][b]["score_std"],
                "score_quantiles": r["branches"][b]["score_quantiles"],
                "score_mean_positive": r["branches"][b]["score_mean_positive"],
                "score_mean_negative": r["branches"][b]["score_mean_negative"],
                "n_predicted_positive_at_0.5": r["branches"][b]["n_predicted_positive_at_0.5"],
            } for b in C.BRANCHES}
            for r in folds
        },
    }


# ---------------------------------------------------------------------------
# paired scaffold bootstrap (fixed fits, fixed split)
# ---------------------------------------------------------------------------


def paired_scaffold_bootstrap(
    fold_results: Sequence[Mapping[str, Any]],
    y: np.ndarray,
    codes: np.ndarray,
    *,
    replicates: int = C.BOOTSTRAP_REPLICATES,
    max_attempts: int = C.BOOTSTRAP_MAX_ATTEMPTS,
    seed: int = C.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Resample scaffolds inside every outer validation fold and re-score.

    The fitted models and the split are fixed; only the evaluation sample varies.
    The same resampled rows are used for all three branches of a replicate.
    """
    rng = np.random.default_rng(int(seed))
    folds = sorted(fold_results, key=lambda r: r["outer_fold"])
    prepared = []
    for rec in folds:
        rows = np.asarray(rec["valid_rows"], dtype=np.int64)
        fold_codes = codes[rows]
        uniq = np.unique(fold_codes)
        members = {int(c): np.flatnonzero(fold_codes == c) for c in uniq}
        prepared.append(
            {
                "rows": rows,
                "codes": fold_codes,
                "uniq": uniq,
                "members": members,
                "y": y[rows],
                "scores": {b: np.asarray(rec["scores"][b], dtype=np.float64) for b in C.BRANCHES},
            }
        )

    diffs_null: list[float] = []
    diffs_base: list[float] = []
    used_replicates = 0
    attempts_total = 0
    rejected = 0
    while used_replicates < int(replicates):
        attempt = 0
        while True:
            if attempt >= int(max_attempts):
                return {
                    "status": "BOOTSTRAP_INCOMPLETE",
                    "n_valid_replicates": int(used_replicates),
                    "n_requested": int(replicates),
                    "attempts_total": int(attempts_total),
                    "rejected_single_class_replicates": int(rejected),
                    "seed": int(seed),
                    "pairing": "paired scaffold bootstrap; the same resampled rows score all three branches",
                    "scope": (
                        "sampling uncertainty at fixed fitted models and fixed folds; excludes re-splitting and "
                        "re-training variability"
                    ),
                }
            attempt += 1
            attempts_total += 1
            per_fold_auc = {b: [] for b in C.BRANCHES}
            ok = True
            sampled = []
            for fold in prepared:
                picks = rng.integers(0, fold["uniq"].size, size=fold["uniq"].size)
                chosen_groups = fold["uniq"][picks]
                row_idx = np.concatenate([fold["members"][int(c)] for c in chosen_groups])
                labels = fold["y"][row_idx]
                if labels.min() == labels.max():
                    ok = False
                    break
                sampled.append((fold, row_idx, labels))
                for b in C.BRANCHES:
                    per_fold_auc[b].append(float(roc_auc_score(labels, fold["scores"][b][row_idx])))
            if not ok:
                rejected += 1
                continue
            break
        used_replicates += 1
        diffs_null.append(float(np.mean(per_fold_auc["BASE_REAL"]) - np.mean(per_fold_auc["BASE_NULL"])))
        diffs_base.append(float(np.mean(per_fold_auc["BASE_REAL"]) - np.mean(per_fold_auc["BASE"])))

    d_null = np.asarray(diffs_null, dtype=np.float64)
    d_base = np.asarray(diffs_base, dtype=np.float64)
    ddof = 1 if d_null.size > 1 else 0
    return {
        "status": "OK",
        "n_valid_replicates": int(used_replicates),
        "n_requested": int(replicates),
        "attempts_total": int(attempts_total),
        "rejected_single_class_replicates": int(rejected),
        "seed": int(seed),
        "pairing": "paired scaffold bootstrap; the same resampled rows score all three branches",
        "scope": (
            "sampling uncertainty at fixed fitted models and fixed folds; excludes re-splitting and re-training "
            "variability"
        ),
        "REAL_minus_NULL": {
            "mean": float(d_null.mean()),
            "std": float(d_null.std(ddof=ddof)),
            "ci95": [float(np.quantile(d_null, 0.025)), float(np.quantile(d_null, 0.975))],
            "ci95_lower": float(np.quantile(d_null, 0.025)),
            "ci95_upper": float(np.quantile(d_null, 0.975)),
            "fraction_positive": float((d_null > 0).mean()),
        },
        "REAL_minus_BASE": {
            "mean": float(d_base.mean()),
            "std": float(d_base.std(ddof=ddof)),
            "ci95": [float(np.quantile(d_base, 0.025)), float(np.quantile(d_base, 0.975))],
            "ci95_lower": float(np.quantile(d_base, 0.025)),
            "ci95_upper": float(np.quantile(d_base, 0.975)),
            "fraction_positive": float((d_base > 0).mean()),
        },
        "ci_method": "percentile bootstrap (2.5%, 97.5%) over replicates",
    }


# ---------------------------------------------------------------------------
# gates / decision
# ---------------------------------------------------------------------------


def evaluate_gates(
    metrics: Mapping[str, Any],
    bootstrap: Mapping[str, Any],
    correctness: Mapping[str, Any],
) -> dict[str, Any]:
    mean_auc = metrics["mean_roc_auc_diff"]
    mean_ap = metrics["mean_ap_diff"]
    fold_diffs = metrics["per_fold_roc_auc_diff"]["REAL_minus_NULL"]
    boot = bootstrap.get("REAL_minus_NULL", {})
    checks = {
        "g1_mean_auc_real_minus_null_ge_0.015": {
            "value": float(mean_auc["REAL_minus_NULL"]),
            "threshold": C.GATE_REAL_MINUS_NULL_AUC_MEAN,
            "passed": bool(mean_auc["REAL_minus_NULL"] >= C.GATE_REAL_MINUS_NULL_AUC_MEAN),
        },
        "g2_all_fold_auc_real_minus_null_gt_0": {
            "values": [float(x) for x in fold_diffs],
            "passed": bool(all(float(x) > 0.0 for x in fold_diffs)),
        },
        "g3_mean_auc_real_minus_base_ge_0.010": {
            "value": float(mean_auc["REAL_minus_BASE"]),
            "threshold": C.GATE_REAL_MINUS_BASE_AUC_MEAN,
            "passed": bool(mean_auc["REAL_minus_BASE"] >= C.GATE_REAL_MINUS_BASE_AUC_MEAN),
        },
        "g4_mean_ap_real_minus_null_ge_0": {
            "value": float(mean_ap["REAL_minus_NULL"]),
            "threshold": C.GATE_REAL_MINUS_NULL_AP_MEAN,
            "passed": bool(mean_ap["REAL_minus_NULL"] >= C.GATE_REAL_MINUS_NULL_AP_MEAN),
        },
        "g5_bootstrap_ci95_lower_gt_0": {
            "value": float(boot.get("ci95_lower", -1.0)),
            "status": bootstrap.get("status"),
            "passed": bool(bootstrap.get("status") == "OK" and float(boot.get("ci95_lower", -1.0)) > 0.0),
        },
        "g6_data_and_feature_correctness": {
            "passed": bool(correctness.get("all_passed", False)),
            "details": correctness,
        },
    }
    promoted = all(v["passed"] for v in checks.values())
    if not correctness.get("all_passed", False):
        decision = "INFEASIBLE"
        reason = "data/feature correctness gates failed; the comparison is not usable"
    elif promoted:
        decision = "PROMOTE_INFORMATION_SIGNAL"
        reason = "all six pre-registered conditions hold"
    elif float(mean_auc["REAL_minus_NULL"]) <= 0.0:
        decision = "NO_OBSERVED_INCREMENT"
        reason = "mean REAL-NULL ROC-AUC difference is <= 0"
    else:
        decision = "WEAK_OR_INCONSISTENT_SIGNAL"
        reason = "mean REAL-NULL ROC-AUC difference is positive but not all conditions hold"
    return {
        "checks": checks,
        "n_checks": len(checks),
        "n_passed": int(sum(1 for v in checks.values() if v["passed"])),
        "all_passed": bool(promoted),
        "decision": decision,
        "reason": reason,
        "thresholds_frozen": {
            "real_minus_null_auc_mean": C.GATE_REAL_MINUS_NULL_AUC_MEAN,
            "real_minus_base_auc_mean": C.GATE_REAL_MINUS_BASE_AUC_MEAN,
            "real_minus_null_ap_mean": C.GATE_REAL_MINUS_NULL_AP_MEAN,
            "all_fold_diffs_positive": C.GATE_ALL_FOLD_DIFFS_POSITIVE,
            "bootstrap_ci_lower": "> 0",
        },
    }


__all__ = [
    "fit_preprocessing",
    "transform",
    "feature_widths",
    "fit_logreg",
    "positive_score",
    "select_c",
    "run_outer_fold",
    "aggregate",
    "paired_scaffold_bootstrap",
    "evaluate_gates",
]

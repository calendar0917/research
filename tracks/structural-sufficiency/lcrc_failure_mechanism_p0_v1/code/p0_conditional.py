"""Conditional model for the Group-A gap direction ``d_i``.

``d_i = squared_error_LCRC - squared_error_PNA`` (positive = PNA better).

M0 uses only the pre-declared base covariates.  M1 adds the four support
features.  Both use the same scaffold-grouped 5-fold cross-fitting, fold-local
standardisation and a fixed Ridge ``alpha=10``.  No hyper-parameter search is
performed and the OOF predictive comparison is not interpreted causally.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from p0_common import BOOTSTRAP_SEED, N_BOOT, N_FOLDS, RIDGE_ALPHA
from p0_sources import Group, align_by_row_id
from p0_stats import (
    ClusterIndex,
    bootstrap_summary,
    cluster_bootstrap,
    deterministic_scaffold_folds,
    quartile_bins,
    spearman,
)
from p0_structure import SUPPORT_COLUMNS

BASE_COVARIATES = (
    "heavy_atoms",
    "tpsa",
    "mol_logp",
    "hbd",
    "hba",
    "formal_charge",
    "ring_count",
    "max_train_tanimoto",
    "scaffold_seen",
)


def _feature_matrix(frame: pd.DataFrame, columns: Sequence[str]) -> np.ndarray:
    out = []
    for column in columns:
        values = frame[column].to_numpy()
        if column == "scaffold_seen":
            values = values.astype(bool).astype(np.float64)
        out.append(np.asarray(values, dtype=np.float64))
    return np.column_stack(out)


def _standardise(train: np.ndarray, test: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    mean = train.mean(axis=0)
    sd = train.std(axis=0, ddof=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return (train - mean) / sd, (test - mean) / sd, {"mean": mean.tolist(), "sd": sd.tolist()}


def _fit_predict(
    x_train: np.ndarray, y_train: np.ndarray, x_test: np.ndarray, *, alpha: float = RIDGE_ALPHA
) -> np.ndarray:
    model = Ridge(alpha=float(alpha), fit_intercept=True)
    model.fit(x_train, y_train)
    return model.predict(x_test)


def _mse(errors: np.ndarray) -> float:
    return float(np.mean(np.asarray(errors, dtype=np.float64) ** 2))


def _mae(errors: np.ndarray) -> float:
    return float(np.mean(np.abs(np.asarray(errors, dtype=np.float64))))


def _stratified_trends(
    frame: pd.DataFrame, d: np.ndarray, scaffolds: Sequence[str]
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    strata: dict[str, np.ndarray] = {}
    stratum_ranges: dict[str, dict[int, list[float]]] = {}
    for column, name in (
        ("heavy_atoms", "heavy_atoms_quartile"),
        ("tpsa", "tpsa_quartile"),
        ("max_train_tanimoto", "max_train_tanimoto_quartile"),
    ):
        labels, edges = quartile_bins(frame[column].to_numpy(), 4)
        strata[name] = labels
        ranges = {}
        values = frame[column].to_numpy(dtype=np.float64)
        for label in range(4):
            selected = values[labels == label]
            if len(selected):
                ranges[label] = [float(selected.min()), float(selected.max())]
        stratum_ranges[name] = ranges
    strata["scaffold_seen"] = frame["scaffold_seen"].astype(bool).to_numpy().astype(np.int64)

    correlations: list[dict[str, Any]] = []
    for support_variable in SUPPORT_COLUMNS:
        overall = spearman(frame[support_variable].to_numpy(), d)
        correlations.append(
            {
                "stratum": "overall",
                "stratum_label": "all",
                "support_variable": support_variable,
                **overall,
            }
        )
        for name, labels in strata.items():
            for label in sorted(np.unique(labels)):
                mask = labels == label
                result = spearman(frame.loc[mask, support_variable].to_numpy(), d[mask])
                correlations.append(
                    {
                        "stratum": name,
                        "stratum_label": int(label),
                        "stratum_range": stratum_ranges.get(name, {}).get(int(label)),
                        "support_variable": support_variable,
                        **result,
                    }
                )

    trends: list[dict[str, Any]] = []
    for support_variable in SUPPORT_COLUMNS:
        labels, edges = quartile_bins(frame[support_variable].to_numpy(), 4)
        values = frame[support_variable].to_numpy(dtype=np.float64)
        se_lcrc = frame["se_lcrc"].to_numpy(dtype=np.float64)
        se_pna = frame["se_pna"].to_numpy(dtype=np.float64)
        for label in range(4):
            mask = labels == label
            n = int(np.sum(mask))
            trends.append(
                {
                    "support_variable": support_variable,
                    "quartile": int(label),
                    "n": n,
                    "low_support": bool(n < 30),
                    "range": [float(values[mask].min()), float(values[mask].max())] if n else None,
                    "mean_d_i": float(np.mean(d[mask])) if n else None,
                    "median_d_i": float(np.median(d[mask])) if n else None,
                    "mean_se_lcrc": float(np.mean(se_lcrc[mask])) if n else None,
                    "mean_se_pna": float(np.mean(se_pna[mask])) if n else None,
                    "rmse_lcrc": float(np.sqrt(np.mean(se_lcrc[mask]))) if n else None,
                    "rmse_pna": float(np.sqrt(np.mean(se_pna[mask]))) if n else None,
                }
            )
        trends.append(
            {
                "support_variable": support_variable,
                "quartile": "edges",
                "n": int(len(frame)),
                "low_support": False,
                "range": None,
                "mean_d_i": None,
                "median_d_i": None,
                "mean_se_lcrc": None,
                "mean_se_pna": None,
                "rmse_lcrc": None,
                "rmse_pna": None,
                "edges": edges.tolist(),
            }
        )
    return correlations, trends


def analyse_group_a(group: Group, support: Any, molecules: Sequence[Any]) -> tuple[dict[str, Any], pd.DataFrame]:
    lcrc = group.predictions["lcrc_a_r3_s0"]
    pna = group.predictions["pna_native_s0"]
    valid_rows = group.rows["valid"]
    y = np.asarray(group.labels["valid"], dtype=np.float64)
    pred_lcrc = align_by_row_id(lcrc.rows, lcrc.pred, valid_rows, label="conditional/lcrc")
    pred_pna = align_by_row_id(pna.rows, pna.pred, valid_rows, label="conditional/pna")

    frame = support.query[support.query["split"] == "valid"].reset_index(drop=True).copy()
    if not np.array_equal(frame["row_id"].to_numpy(), valid_rows):
        raise ValueError("support frame is not aligned to the frozen Validation420 order")
    se_lcrc = (pred_lcrc - y) ** 2
    se_pna = (pred_pna - y) ** 2
    d = se_lcrc - se_pna
    frame["y"] = y
    frame["pred_lcrc"] = pred_lcrc
    frame["pred_pna"] = pred_pna
    frame["se_lcrc"] = se_lcrc
    frame["se_pna"] = se_pna
    frame["d_i"] = d

    scaffolds = frame["scaffold"].astype(str).to_numpy()
    folds = deterministic_scaffold_folds(scaffolds, n_splits=N_FOLDS)
    frame["fold_m0m1"] = folds

    base_x = _feature_matrix(frame, BASE_COVARIATES)
    support_x = _feature_matrix(frame, SUPPORT_COLUMNS)
    m1_x = np.column_stack([base_x, support_x])

    oof = {name: np.full(len(frame), np.nan, dtype=np.float64) for name in ("M0", "M1")}
    fold_records: list[dict[str, Any]] = []
    for fold in range(N_FOLDS):
        train_mask = folds != fold
        test_mask = folds == fold
        xb_train, xb_test, _ = _standardise(base_x[train_mask], base_x[test_mask])
        x1_train, x1_test, _ = _standardise(m1_x[train_mask], m1_x[test_mask])
        oof["M0"][test_mask] = _fit_predict(xb_train, d[train_mask], xb_test)
        oof["M1"][test_mask] = _fit_predict(x1_train, d[train_mask], x1_test)
        fold_records.append(
            {
                "fold": fold,
                "n": int(np.sum(test_mask)),
                "n_scaffolds": int(len(np.unique(scaffolds[test_mask]))),
                "mse_M0": _mse(d[test_mask] - oof["M0"][test_mask]),
                "mse_M1": _mse(d[test_mask] - oof["M1"][test_mask]),
                "mae_M0": _mae(d[test_mask] - oof["M0"][test_mask]),
                "mae_M1": _mae(d[test_mask] - oof["M1"][test_mask]),
                "delta_mse_M1_minus_M0": _mse(d[test_mask] - oof["M1"][test_mask])
                - _mse(d[test_mask] - oof["M0"][test_mask]),
                "delta_mae_M1_minus_M0": _mae(d[test_mask] - oof["M1"][test_mask])
                - _mae(d[test_mask] - oof["M0"][test_mask]),
            }
        )

    oof_metrics = {}
    for name in ("M0", "M1"):
        errors = d - oof[name]
        oof_metrics[name] = {
            "mse": _mse(errors),
            "mae": _mae(errors),
            "rmse": float(np.sqrt(_mse(errors))),
            "n": int(len(frame)),
        }
    oof_metrics["delta_M1_minus_M0"] = {
        "mse": oof_metrics["M1"]["mse"] - oof_metrics["M0"]["mse"],
        "mae": oof_metrics["M1"]["mae"] - oof_metrics["M0"]["mae"],
        "rmse": oof_metrics["M1"]["rmse"] - oof_metrics["M0"]["rmse"],
    }

    # fixed-OOF scaffold-cluster bootstrap for the OOF metric difference
    cluster = ClusterIndex.from_values(scaffolds)
    errors0 = d - oof["M0"]
    errors1 = d - oof["M1"]

    def _compute(idx: np.ndarray) -> dict[str, float]:
        return {
            "mse_M0": _mse(errors0[idx]),
            "mse_M1": _mse(errors1[idx]),
            "mae_M0": _mae(errors0[idx]),
            "mae_M1": _mae(errors1[idx]),
            "delta_mse_M1_minus_M0": _mse(errors1[idx]) - _mse(errors0[idx]),
            "delta_mae_M1_minus_M0": _mae(errors1[idx]) - _mae(errors0[idx]),
        }

    point = _compute(np.arange(len(frame)))
    draws = cluster_bootstrap(cluster, _compute, n_boot=N_BOOT, seed=BOOTSTRAP_SEED)
    bootstrap = bootstrap_summary(point, draws)

    # full-sample descriptive fit (not used for the OOF comparison)
    xb_std, _, _ = _standardise(base_x, base_x)
    x1_std, _, _ = _standardise(m1_x, m1_x)
    model0 = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True).fit(xb_std, d)
    model1 = Ridge(alpha=RIDGE_ALPHA, fit_intercept=True).fit(x1_std, d)
    coefficients = {
        "M0": dict(zip(BASE_COVARIATES, model0.coef_.tolist())),
        "M1": dict(zip(list(BASE_COVARIATES) + list(SUPPORT_COLUMNS), model1.coef_.tolist())),
        "standardisation": "full Validation420 sample, descriptive only",
        "intercept": {"M0": float(model0.intercept_), "M1": float(model1.intercept_)},
        "caveat": "coefficients are descriptive, not causal; OOF metrics are the registered comparison",
    }

    correlations, trends = _stratified_trends(frame, d, scaffolds)
    payload = {
        "definition": {
            "target": "d_i = squared_error_LCRC - squared_error_PNA (positive = PNA better)",
            "base_covariates": list(BASE_COVARIATES),
            "support_variables": list(SUPPORT_COLUMNS),
            "ridge_alpha": RIDGE_ALPHA,
            "standardisation": "training-fold mean/std only (ddof=0); zero-variance columns use sd=1",
            "folds": N_FOLDS,
            "fold_assignment": "deterministic balanced whole-scaffold assignment (row-order invariant)",
            "hyperparameter_search": False,
            "causal_interpretation": False,
        },
        "n": int(len(frame)),
        "group": group.key,
        "split": group.split,
        "exposure": "FROZEN_VALIDATION_ONLY",
        "fold_profile": {
            "sizes": [int(np.sum(folds == k)) for k in range(N_FOLDS)],
            "n_scaffolds": [int(len(np.unique(scaffolds[folds == k]))) for k in range(N_FOLDS)],
        },
        "oof_metrics": oof_metrics,
        "folds": fold_records,
        "bootstrap": {
            "unit": "Bemis-Murcko scaffold",
            "n_boot": N_BOOT,
            "cluster_profile": cluster.profile(),
            "intervals": bootstrap,
            "note": "OOF predictions are held fixed; the bootstrap quantifies scaffold-cluster sampling uncertainty only.",
        },
        "full_fit_coefficients": coefficients,
        "stratified_support_correlations": correlations,
        "support_quartile_trends": trends,
        "low_support_rule": "n < 30 -> LOW_SUPPORT; no threshold is merged after seeing results",
        "interpretation_warning": (
            "Held-out OOF MSE/MAE describe predictive association only. They do not identify a causal "
            "effect of support, because support co-varies with size, polarity and similarity."
        ),
    }
    oof_frame = frame[
        [
            "row_id",
            "fold_m0m1",
            "d_i",
            "se_lcrc",
            "se_pna",
            "pred_lcrc",
            "pred_pna",
            "unseen_atom_frac",
            "rare_atom_frac",
            "median_log1p_df",
            "p10_log1p_df",
        ]
    ].copy()
    oof_frame["oof_m0"] = oof["M0"]
    oof_frame["oof_m1"] = oof["M1"]
    oof_frame["d_minus_oof_m0"] = d - oof["M0"]
    oof_frame["d_minus_oof_m1"] = d - oof["M1"]
    return payload, oof_frame

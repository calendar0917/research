"""Phase A — finite regularisation/scaling controls for the full linear structure model
(MolHIV only, no retraining of the historical COUNT-ALL fit).

* ``L-BAL-STD``  class-balanced logistic, all columns Train-standardised;
                 new C in {0.0003, 0.003}; the historical C=0.03 fit is reused as-is.
* ``L-NAT-STD``  natural (unweighted) logistic, all columns standardised,
                 C in {0.0003, 0.003, 0.03}, unpenalised intercept.
* ``L-NAT-RAW``  natural logistic, size terms standardised, frozen structure counts
                 kept at their frozen scale, same C grid, unpenalised intercept.

All fits use the sparse Newton implementation of ISRD-v1, so the 4408-column dense
standardised design is never materialised.
"""

from __future__ import annotations

import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import scipy.sparse as sp

import sfl_common as K

sys.path.insert(0, str(K.ISRD / "code"))
import isrd1_common as I  # noqa: E402

C_GRID_NEW = (0.0003, 0.003)
C_GRID_NAT = (0.0003, 0.003, 0.03)


def sparse_standardisation_stats(X: sp.csr_matrix, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Train-only column mean/std of a sparse design (same as ISRD-v1 run_phaseC)."""
    sub = X[rows]
    mu = np.asarray(sub.mean(axis=0)).ravel()
    mu2 = np.asarray(sub.multiply(sub).mean(axis=0)).ravel()
    sd = np.sqrt(np.maximum(mu2 - mu * mu, 0.0))
    return mu, np.where(sd > 0, sd, 1.0)


def fit_one(X: sp.csr_matrix, y: np.ndarray, sw: np.ndarray, C: float,
            mu: np.ndarray, sd: np.ndarray, penalize_intercept: bool,
            name: str) -> dict[str, Any]:
    if np.any(sw[~train_mask] != 0):
        raise ValueError("sample weights must be zero outside Train (label leak guard)")
    t0 = time.perf_counter()
    fit = I.fit_logistic_sparse_standardised(
        X, y, sw, C, mu, sd, penalize_intercept=penalize_intercept,
        max_iter=60, tol=1e-9, verbose=False)
    logit = np.asarray(fit["logit"], dtype=np.float64)
    metrics = K.hiv_metrics(logit, y, data_rows)
    coef_std = np.asarray(fit["coef"], dtype=np.float64)
    w_raw = coef_std / sd
    b_raw = float(fit["intercept"] - np.sum(coef_std * mu / sd))
    closure = float(np.max(np.abs(b_raw + np.asarray(X @ w_raw).ravel() - logit)))
    out = {
        "arm": name, "C": float(C), "penalize_intercept": bool(penalize_intercept),
        "intercept_standardised": float(fit["intercept"]), "coef_standardised": coef_std,
        "intercept_raw": b_raw, "coef_raw": w_raw,
        "mean": mu, "sd": sd,
        "logit": logit, "metrics": metrics,
        "converged": bool(fit["converged"]), "grad_inf": float(fit["grad_inf"]),
        "n_iter": int(fit["n_iter"]), "closure_raw_units": closure,
        "n_parameters": int(coef_std.size + 1),
        "train_valid_gap_auc": (metrics["train"]["roc_auc"] - metrics["valid"]["roc_auc"]),
        "fit_seconds": float(time.perf_counter() - t0),
    }
    print(f"[A] {name} C={C:g} train={metrics['train']['roc_auc']:.4f} "
          f"valid={metrics['valid']['roc_auc']:.4f} test={metrics['test']['roc_auc']:.4f} "
          f"ap_valid={metrics['valid']['average_precision']:.4f} "
          f"gap={out['train_valid_gap_auc']:.4f} it={fit['n_iter']} "
          f"grad={fit['grad_inf']:.1e} {out['fit_seconds']:.1f}s", flush=True)
    return out


_PARAM_KEYS = {"logit", "coef_standardised", "coef_raw", "mean", "sd"}


def strip(fit: dict[str, Any]) -> dict[str, Any]:
    """JSON summary without the 4408-vector parameters (kept in linear_models.json)."""
    return {k: v for k, v in fit.items() if k not in _PARAM_KEYS}


def save_prediction(arm: str, C: float, logit: np.ndarray, tag: str | None = None) -> str:
    path = K.PREDICTIONS / f"{tag or arm}.npz"
    np.savez_compressed(path, logit=np.asarray(logit, dtype=np.float64), y=data.y,
                        graph_id=np.arange(data.n), split_code=data.split_code(),
                        arm=arm, C=float(C), dataset="hiv")
    return K.rel(path)


def main() -> None:
    global data, data_rows, train_mask
    t0 = time.perf_counter()
    data = K.HIVData()
    data_rows = data.rows
    train_mask = np.zeros(data.n, dtype=bool)
    train_mask[data.rows["train"]] = True
    y = data.y
    train = data.rows["train"]
    X = data.design_sparse()
    print(f"[A] design {X.shape} nnz={X.nnz}", flush=True)

    mu_all, sd_all = sparse_standardisation_stats(X, train)
    # structure columns: mean 0 / sd 1 (keep the frozen raw scale)
    mu_raw = mu_all.copy()
    sd_raw = sd_all.copy()
    mu_raw[2:] = 0.0
    sd_raw[2:] = 1.0

    sw_bal = np.zeros(y.size, dtype=np.float64)
    sw_bal[train] = I.class_balanced_weights(y[train].astype(np.float64))
    # IMPORTANT: sample weights are zero outside Train; the sparse fitter consumes the
    # full 41127-row design, and non-zero validation/test weights would leak labels.
    sw_nat = np.zeros(y.size, dtype=np.float64)
    sw_nat[train] = 1.0

    results: dict[str, Any] = {}
    selected_models: dict[str, Any] = {}
    all_configs: list[dict[str, Any]] = []

    # ---------------- L-BAL-STD -------------------------------------------
    bal_fits = []
    for C in C_GRID_NEW:
        f = fit_one(X, y, sw_bal, C, mu_all, sd_all, True, "L-BAL-STD")
        bal_fits.append(f)
        np.savez_compressed(K.PREDICTIONS / f"L-BAL-STD_C{C:g}.npz",
                            logit=f["logit"], y=y, graph_id=np.arange(data.n),
                            split_code=data.split_code(), arm="L-BAL-STD", C=float(C),
                            dataset="hiv")
    # historical C=0.03 (read-only reuse)
    phase_c = K.read_json(K.ISRD_PHASEC)["count_all"]
    hist_logit = np.load(K.ISRD_PRED / "COUNT-ALL.npz")["logit"].astype(np.float64)
    hist = {
        "arm": "L-BAL-STD", "C": 0.03, "penalize_intercept": True,
        "intercept_standardised": float(phase_c["intercept_standardised"]),
        "coef_standardised": np.asarray(phase_c["coef_standardised"], dtype=np.float64),
        "intercept_raw": float(phase_c["intercept_raw"]),
        "coef_raw": np.asarray(phase_c["coef_raw"], dtype=np.float64),
        "mean": np.asarray(phase_c["mean"], dtype=np.float64),
        "sd": np.asarray(phase_c["sd"], dtype=np.float64),
        "logit": hist_logit,
        "metrics": K.hiv_metrics(hist_logit, y, data.rows),
        "converged": bool(phase_c["converged"]),
        "grad_inf": float(phase_c["grad_inf"]),
        "n_iter": None, "closure_raw_units": float(phase_c["closure_raw_units"]),
        "n_parameters": int(phase_c["n_parameters"]),
        "train_valid_gap_auc": float(phase_c["train_valid_gap_auc"]),
        "fit_seconds": None,
        "source": "ISRD-v1 analysis/phaseC.json count_all (C=0.03), reused not refit",
    }
    bal_fits.append(hist)
    best = max(bal_fits, key=lambda f: (f["metrics"]["valid"]["roc_auc"], -f["C"]))
    selected_models["L-BAL-STD"] = {
        "selected_C": best["C"], "intercept_raw": best["intercept_raw"],
        "coef_raw": best["coef_raw"], "mean": best["mean"], "sd": best["sd"],
        "metrics": best["metrics"]}
    results["L-BAL-STD"] = {"arm": "L-BAL-STD", "selected_C": best["C"],
                            "selection_rule": "max Valid AUC, tie -> smaller C",
                            "selected": strip(best), "configs": [strip(f) for f in bal_fits],
                            "C_grid": sorted(f["C"] for f in bal_fits)}
    np.savez_compressed(K.PREDICTIONS / "L-BAL-STD.npz",
                        logit=best["logit"], y=y, graph_id=np.arange(data.n),
                        split_code=data.split_code(), arm="L-BAL-STD",
                        C=float(best["C"]), dataset="hiv")
    all_configs.extend(results["L-BAL-STD"]["configs"])

    # ---------------- L-NAT-STD -------------------------------------------
    nat_std = [fit_one(X, y, sw_nat, C, mu_all, sd_all, False, "L-NAT-STD")
               for C in C_GRID_NAT]
    for f in nat_std:
        np.savez_compressed(K.PREDICTIONS / f"L-NAT-STD_C{f['C']:g}.npz",
                            logit=f["logit"], y=y, graph_id=np.arange(data.n),
                            split_code=data.split_code(), arm="L-NAT-STD",
                            C=float(f["C"]), dataset="hiv")
    best = max(nat_std, key=lambda f: (f["metrics"]["valid"]["roc_auc"], -f["C"]))
    selected_models["L-NAT-STD"] = {
        "selected_C": best["C"], "intercept_raw": best["intercept_raw"],
        "coef_raw": best["coef_raw"], "mean": best["mean"], "sd": best["sd"],
        "metrics": best["metrics"]}
    results["L-NAT-STD"] = {"arm": "L-NAT-STD", "selected_C": best["C"],
                            "selection_rule": "max Valid AUC, tie -> smaller C",
                            "selected": strip(best),
                            "configs": [strip(f) for f in nat_std],
                            "C_grid": list(C_GRID_NAT)}
    np.savez_compressed(K.PREDICTIONS / "L-NAT-STD.npz", logit=best["logit"], y=y,
                        graph_id=np.arange(data.n), split_code=data.split_code(),
                        arm="L-NAT-STD", C=float(best["C"]), dataset="hiv")
    all_configs.extend(results["L-NAT-STD"]["configs"])

    # ---------------- L-NAT-RAW -------------------------------------------
    nat_raw = [fit_one(X, y, sw_nat, C, mu_raw, sd_raw, False, "L-NAT-RAW")
               for C in C_GRID_NAT]
    for f in nat_raw:
        np.savez_compressed(K.PREDICTIONS / f"L-NAT-RAW_C{f['C']:g}.npz",
                            logit=f["logit"], y=y, graph_id=np.arange(data.n),
                            split_code=data.split_code(), arm="L-NAT-RAW",
                            C=float(f["C"]), dataset="hiv")
    best = max(nat_raw, key=lambda f: (f["metrics"]["valid"]["roc_auc"], -f["C"]))
    selected_models["L-NAT-RAW"] = {
        "selected_C": best["C"], "intercept_raw": best["intercept_raw"],
        "coef_raw": best["coef_raw"], "mean": best["mean"], "sd": best["sd"],
        "metrics": best["metrics"]}
    results["L-NAT-RAW"] = {"arm": "L-NAT-RAW", "selected_C": best["C"],
                            "selection_rule": "max Valid AUC, tie -> smaller C",
                            "selected": strip(best),
                            "configs": [strip(f) for f in nat_raw],
                            "C_grid": list(C_GRID_NAT)}
    np.savez_compressed(K.PREDICTIONS / "L-NAT-RAW.npz", logit=best["logit"], y=y,
                        graph_id=np.arange(data.n), split_code=data.split_code(),
                        arm="L-NAT-RAW", C=float(best["C"]), dataset="hiv")
    all_configs.extend(results["L-NAT-RAW"]["configs"])

    for name, res in results.items():
        lo, hi = min(res["C_grid"]), max(res["C_grid"])
        if res["selected_C"] in (lo, hi):
            res["selection_at_grid_boundary"] = True
            res["boundary_side"] = "lower" if res["selected_C"] == lo else "upper"
            print(f"[A] {name}: best Valid AUC at the {res['boundary_side']} edge "
                  f"C={res['selected_C']:g} -> search range limited", flush=True)

    payload = {
        "phase": "A: linear structure model learning constraints (MolHIV)",
        "round_id": K.ROUND_ID,
        "design": {"shape": list(X.shape), "nnz": int(X.nnz),
                   "columns": "[log1p(n_atoms), log1p(n_bonds), C(754), F(3652)]",
                   "structure_scale": "frozen count/sqrt(n_atoms)"},
        "standardisation": {
            "fit_on": "Train only",
            "all_column_mean_sha256": K.sha256_array(mu_all),
            "all_column_sd_sha256": K.sha256_array(sd_all),
            "raw_arm_note": "structure columns mean=0, sd=1; only size terms standardised",
        },
        "class_balanced_weights": {
            "pos": float(sw_bal[train][y[train] == 1][0]),
            "neg": float(sw_bal[train][y[train] == 0][0]),
            "mean_train": float(sw_bal[train].mean()),
        },
        "results": results,
        "all_configs": all_configs,
        "seconds": float(time.perf_counter() - t0),
        "git": K.git_state(), "environment": K.environment(),
    }
    K.write_json(K.ANALYSIS / "phaseA_linear.json", payload)
    # compact model store for Phase E (selected linear model weights)
    K.write_json(K.ANALYSIS / "linear_models.json", selected_models)
    print(f"[A] done in {payload['seconds']:.1f}s")


if __name__ == "__main__":
    main()

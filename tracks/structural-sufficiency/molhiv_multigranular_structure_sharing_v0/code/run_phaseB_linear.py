"""Phase B: fixed-representation Logistic Regression arms.

Every arm is the same additive logit

    z(G) = b + M(G)·w + C(G)·u + F(G)·v

fitted with one identical recipe: BCE + L2, ``class_weight='balanced'``,
liblinear, ``max_iter=2000`` (extended to 5000 only if the same objective failed
to converge), ``tol=1e-6``, and the shared candidate grid ``C ∈ {0.03, 0.3, 3.0}``.

Arm selection is by **Valid ROC-AUC only** (tie -> smaller C).  Test is reported
for every fitted C and is marked ``EXPOSED_DEVELOPMENT_TEST``; it never selects
anything.
"""

from __future__ import annotations

import argparse
import datetime as dt
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

import msds_common as C
import msds_data as D
import msds_features as FEAT

C_GRID = (0.03, 0.3, 3.0)
MAX_ITER = 2000
MAX_ITER_RETRY = 5000
TOL = 1e-6
ARMS = ("B0", "B1", "BF", "B2", "B2-PNULL", "B2-U", "B3", "B3-NULL")
ARM_DESCRIPTION = {
    "B0": ("M", "plain L2"),
    "B1": ("M + C", "plain L2"),
    "BF": ("M + F", "plain L2"),
    "B2": ("M + C + F", "plain L2"),
    "B2-PNULL": ("M + C_null + F", "plain L2, shuffled parent assignment"),
    "B2-U": ("M + C + F/sqrt(mean a_f)", "uniform full-block rescaling"),
    "B3": ("M + C + F/sqrt(a_f)", "true support-weighted full-block rescaling"),
    "B3-NULL": ("M + C + F/sqrt(a_null)", "within-family shuffled support weights"),
}


def load_design() -> dict[str, Any]:
    path = C.CACHE / "design_CF.npz"
    with np.load(path) as z:
        n_rows = int(z["n_rows"])
        n_coarse = int(z["n_coarse"])
        n_full = int(z["n_full"])
        def csr(prefix: str):
            return sp.csr_matrix((z[f"{prefix}_data"], z[f"{prefix}_indices"], z[f"{prefix}_indptr"]),
                                 shape=(n_rows, n_coarse if prefix in ("C", "Cnull", "R") else n_full))
        out = {
            "C": csr("C"), "F": csr("F"), "Cnull": csr("Cnull"), "R": csr("R"),
            "path": C.rel(path), "sha256": C.sha256_file(path),
        }
    with np.load(C.CACHE / "vocab.npz") as z:
        out["support_weight"] = np.asarray(z["support_weight"], dtype=np.float64)
        out["support_weight_null"] = np.asarray(z["support_weight_null"], dtype=np.float64)
        out["full_parent_in_vocab"] = np.asarray(z["full_parent_in_vocab"], dtype=np.int64)
        out["full_family"] = np.asarray(z["full_family"], dtype=np.int64)
        out["vocab_sha256"] = C.sha256_file(C.CACHE / "vocab.npz")
    return out


def scale_full(F: sp.csr_matrix, weights: np.ndarray | float) -> sp.csr_matrix:
    """Return ``F diag(1/sqrt(w))`` for the weighted-L2 reparameterisation."""
    if np.isscalar(weights):
        factor = np.full(F.shape[1], 1.0 / np.sqrt(float(weights)))
    else:
        factor = 1.0 / np.sqrt(np.asarray(weights, dtype=np.float64))
    return sp.csr_matrix(F.multiply(factor[None, :]))


def metrics(logit: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))
    return {
        "n": int(y.size),
        "n_positive": int(y.sum()),
        "roc_auc": float(roc_auc_score(y, prob)),
        "average_precision": float(average_precision_score(y, prob)),
        "bce": float(np.mean(np.logaddexp(0.0, logit) - y * logit)),
    }


def ogb_check(logit: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    from ogb.graphproppred import Evaluator
    from sklearn.metrics import roc_auc_score

    prob = 1.0 / (1.0 + np.exp(-np.clip(logit, -60, 60)))
    evaluator = Evaluator(name="ogbg-molhiv")
    ogb_auc = float(evaluator.eval({"y_true": y.reshape(-1, 1), "y_pred": prob.reshape(-1, 1)})["rocauc"])
    sk_auc = float(roc_auc_score(y, prob))
    return {"ogb_evaluator_rocauc": ogb_auc, "sklearn_rocauc": sk_auc, "abs_diff": abs(ogb_auc - sk_auc),
            "consistent": bool(abs(ogb_auc - sk_auc) < 1e-12)}


def fit_lr(X: sp.csr_matrix, y: np.ndarray, c_value: float) -> tuple[Any, dict[str, Any]]:
    import warnings

    from sklearn.exceptions import ConvergenceWarning
    from sklearn.linear_model import LogisticRegression

    info: dict[str, Any] = {"candidate_C": c_value, "max_iter": MAX_ITER, "tol": TOL}
    t0 = time.perf_counter()
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always", ConvergenceWarning)
        model = LogisticRegression(
            solver="liblinear", penalty="l2", C=c_value, class_weight="balanced",
            fit_intercept=True, max_iter=MAX_ITER, tol=TOL,
        )
        model.fit(X, y)
    converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught)
    info["n_iter"] = int(np.max(model.n_iter_))
    if not converged:
        info["retry"] = {
            "reason": "ConvergenceWarning at max_iter=2000; same objective, longer budget only",
            "max_iter": MAX_ITER_RETRY,
        }
        with warnings.catch_warnings(record=True) as caught2:
            warnings.simplefilter("always", ConvergenceWarning)
            model = LogisticRegression(
                solver="liblinear", penalty="l2", C=c_value, class_weight="balanced",
                fit_intercept=True, max_iter=MAX_ITER_RETRY, tol=TOL,
            )
            model.fit(X, y)
        converged = not any(issubclass(w.category, ConvergenceWarning) for w in caught2)
        info["n_iter"] = int(np.max(model.n_iter_))
    info["converged"] = bool(converged)
    info["seconds"] = float(time.perf_counter() - t0)
    return model, info


def main() -> None:
    ap = argparse.ArgumentParser(description="Phase B linear arms")
    ap.add_argument("--arms", nargs="*", default=list(ARMS))
    ap.add_argument("--out", default=str(C.ANALYSIS / "phaseB_linear.json"))
    args = ap.parse_args()

    C.ensure_dirs()
    t_start = time.perf_counter()
    molecules = D.load_molecules()
    splits = {k: np.asarray(v, dtype=np.int64) for k, v in molecules.split.items()}
    y = molecules.label.astype(np.float64)
    design = load_design()
    bits, morgan_meta = FEAT.build_morgan(molecules)
    M, m_meta = FEAT.design_M(molecules, bits, splits["train"])
    print(f"[phaseB] M {M.shape} nnz={M.nnz} | C {design['C'].shape} nnz={design['C'].nnz} "
          f"| F {design['F'].shape} nnz={design['F'].nnz}", flush=True)

    a = design["support_weight"]
    a_null = design["support_weight_null"]
    mean_a = float(a.mean())
    blocks: dict[str, sp.csr_matrix] = {
        "M": M,
        "C": design["C"],
        "F": design["F"],
        "Cnull": design["Cnull"],
        "Fu": scale_full(design["F"], mean_a),
        "Fw": scale_full(design["F"], a),
        "Fwnull": scale_full(design["F"], a_null),
    }
    arm_blocks = {
        "B0": ("M",),
        "B1": ("M", "C"),
        "BF": ("M", "F"),
        "B2": ("M", "C", "F"),
        "B2-PNULL": ("M", "Cnull", "F"),
        "B2-U": ("M", "C", "Fu"),
        "B3": ("M", "C", "Fw"),
        "B3-NULL": ("M", "C", "Fwnull"),
    }
    effective_weight = {
        "B2": np.ones(design["F"].shape[1]),
        "BF": np.ones(design["F"].shape[1]),
        "B2-PNULL": np.ones(design["F"].shape[1]),
        "B2-U": np.full(design["F"].shape[1], mean_a),
        "B3": a,
        "B3-NULL": a_null,
    }

    results: dict[str, Any] = {}
    logits_store: dict[str, np.ndarray] = {}
    coef_store: dict[str, np.ndarray] = {}
    feature_identity: dict[str, str] = {}
    for arm in args.arms:
        parts = arm_blocks[arm]
        X = sp.hstack([blocks[p] for p in parts], format="csr")
        key = f"{X.shape[1]}_{X.nnz}_{abs(X).sum():.6f}_{parts}"
        reused = None
        for other, other_key in feature_identity.items():
            if other_key == key and arm_blocks[other] == parts:
                reused = other
                break
        feature_identity.setdefault(arm, key)
        arm_res: dict[str, Any] = {
            "arm": arm,
            "inputs": ARM_DESCRIPTION[arm][0],
            "regularisation": ARM_DESCRIPTION[arm][1],
            "blocks": list(parts),
            "n_features": int(X.shape[1]),
            "nnz": int(X.nnz),
            "fitted_in_this_run": True,
            "fits": [],
        }
        if reused is not None:
            arm_res["fitted_in_this_run"] = False
            arm_res["reused_from"] = reused
            arm_res["fits"] = results[reused]["fits"]
            results[arm] = arm_res
            print(f"[phaseB] {arm}: identical design to {reused}; reusing its fits", flush=True)
            continue
        Xtr = X[splits["train"]]
        ytr = y[splits["train"]]
        for c_value in C_GRID:
            model, info = fit_lr(Xtr, ytr, c_value)
            entry: dict[str, Any] = {"candidate_C": c_value, **info}
            for split_name in ("train", "valid", "test"):
                rows = splits[split_name]
                logit = model.decision_function(X[rows]).astype(np.float64)
                entry[split_name] = metrics(logit, y[rows])
                logits_store[f"{arm}|{c_value}|{split_name}"] = logit.astype(np.float32)
            entry["test"]["evidence_label"] = "EXPOSED_DEVELOPMENT_TEST"
            entry["coefficients"] = {
                "intercept": float(model.intercept_[0]),
                "n_nonzero": int((np.abs(model.coef_.ravel()) > 0).sum()),
                "l2_norm": float(np.linalg.norm(model.coef_.ravel())),
            }
            coef_store[f"{arm}|{c_value}"] = model.coef_.ravel().astype(np.float32)
            arm_res["fits"].append(entry)
            print(
                f"[phaseB] {arm} C={c_value:<5} valid_auc={entry['valid']['roc_auc']:.4f} "
                f"test_auc={entry['test']['roc_auc']:.4f} train_auc={entry['train']['roc_auc']:.4f} "
                f"it={info['n_iter']} conv={info['converged']} {info['seconds']:.1f}s",
                flush=True,
            )
        best = max(arm_res["fits"], key=lambda e: (e["valid"]["roc_auc"], -e["candidate_C"]))
        arm_res["selected"] = {
            "rule": "highest Valid ROC-AUC, tie -> smaller C",
            "candidate_C": best["candidate_C"],
            "valid": best["valid"],
            "test": best["test"],
            "train": best["train"],
            "test_evidence_label": "EXPOSED_DEVELOPMENT_TEST",
        }
        results[arm] = arm_res

    # ---- OGB-evaluator cross-check on the selected models
    for arm, res in results.items():
        if "selected" not in res:
            continue
        for split_name in ("valid", "test"):
            logit = logits_store[f"{arm}|{res['selected']['candidate_C']}|{split_name}"].astype(np.float64)
            res["selected"][f"{split_name}_ogb_check"] = ogb_check(logit, y[splits[split_name]])

    payload = {
        "round_id": C.ROUND_ID,
        "phase": "B: fixed-representation linear arms",
        "created_utc": dt.datetime.now(dt.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
        "recipe": {
            "model": "sklearn LogisticRegression(liblinear, L2, fit_intercept)",
            "penalty_scale": "sklearn 0.5*||w||^2 + C * sum_i class_weight_i * logloss_i",
            "class_weight": "balanced",
            "C_grid": list(C_GRID),
            "max_iter": MAX_ITER,
            "max_iter_retry": MAX_ITER_RETRY,
            "tol": TOL,
            "selection": "highest Valid ROC-AUC, tie -> smaller C",
            "test_policy": "EXPOSED_DEVELOPMENT_TEST (reported, never used for selection/design)",
            "seed": C.SEED,
        },
        "M": m_meta,
        "morgan": morgan_meta,
        "design": {k: design[k] for k in ("path", "sha256", "vocab_sha256")},
        "support_weight_mean": mean_a,
        "arms": results,
        "seconds": float(time.perf_counter() - t_start),
        "git": C.git_state(),
        "environment": C.environment(),
    }
    C.write_json(args.out, payload)
    np.savez_compressed(
        C.CACHE / "phaseB_predictions.npz",
        **{k: v for k, v in logits_store.items()},
        train_rows=splits["train"], valid_rows=splits["valid"], test_rows=splits["test"], y=y,
    )
    np.savez_compressed(C.CACHE / "phaseB_coefficients.npz", **coef_store)
    print(f"[phaseB] wrote {C.rel(args.out)} in {time.perf_counter() - t_start:.1f}s", flush=True)


if __name__ == "__main__":
    main()

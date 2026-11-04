"""P0: fixed LightGBM readout for TYPE-S2 / FULL-S2 / FULL-S2-BINARY counts.

Run inside the isolated CPU fingerprint environment (lightgbm 4.7.0,
scikit-learn 1.9.1).  Train only; Validation only for evaluation.  The number
of trees, class balancing and random seed are frozen; there is no search.
"""

from __future__ import annotations

import time
from pathlib import Path

import numpy as np

import pc_common as C

GUARD = C.install_test_access_guard()

LGBM_KWARGS = {
    "n_estimators": 500,
    "is_unbalance": True,
    "importance_type": "gain",
    "random_state": 0,
    "n_jobs": 8,
    "verbose": -1,
}


def load_counts(split: str) -> dict[str, np.ndarray]:
    path = C.ANALYSIS / f"features_{split}.npz"
    with np.load(path) as z:
        return {
            "type": np.asarray(z["counts_type"], dtype=np.float32),
            "full": np.asarray(z["counts_full"], dtype=np.float32),
            "full_binary": np.asarray(z["full_binary"], dtype=np.float32),
            "null": np.asarray(z["counts_null"], dtype=np.float32),
            "n_paths": np.asarray(z["n_paths"], dtype=np.int64),
        }


def main() -> int:
    from lightgbm import LGBMClassifier
    from sklearn.metrics import average_precision_score

    t_start = time.perf_counter()
    train = load_counts("train")
    valid = load_counts("valid")
    y_train = np.load(C.PREV_ANALYSIS / "peptides_func_y_train.npy").astype(np.int64)
    y_valid = np.load(C.PREV_ANALYSIS / "peptides_func_y_valid.npy").astype(np.int64)
    if y_train.shape != (train["type"].shape[0], C.N_LABELS):
        raise SystemExit("train label shape mismatch")
    if y_valid.shape != (valid["type"].shape[0], C.N_LABELS):
        raise SystemExit("valid label shape mismatch")

    results: dict[str, dict] = {}
    for kind in ("type", "full", "full_binary"):
        X_train = train[kind]
        X_valid = valid[kind]
        if X_train.shape[1] != C.N_BUCKETS or X_valid.shape[1] != C.N_BUCKETS:
            raise SystemExit(f"{kind}: feature width mismatch")
        feature_summary = {
            "kind": kind,
            "shape_train": list(X_train.shape),
            "shape_valid": list(X_valid.shape),
            "nnz_train": int(np.count_nonzero(X_train)),
            "nnz_valid": int(np.count_nonzero(X_valid)),
            "nonzero_columns_train": int(np.count_nonzero(X_train.any(axis=0))),
            "nonzero_columns_valid": int(np.count_nonzero(X_valid.any(axis=0))),
            "max_train": float(X_train.max()),
            "max_valid": float(X_valid.max()),
            "empty_rows_train": int(np.count_nonzero(X_train.sum(axis=1) == 0)),
            "empty_rows_valid": int(np.count_nonzero(X_valid.sum(axis=1) == 0)),
        }
        probs_valid = np.zeros((X_valid.shape[0], C.N_LABELS), dtype=np.float64)
        probs_train = np.zeros((X_train.shape[0], C.N_LABELS), dtype=np.float64)
        per_label: list[dict] = []
        fit_seconds: list[float] = []
        importances = np.zeros((C.N_LABELS, X_train.shape[1]), dtype=np.float64)
        for j, name in enumerate(C.LABELS):
            t0 = time.perf_counter()
            clf = LGBMClassifier(**LGBM_KWARGS)
            clf.fit(X_train, y_train[:, j])
            p_valid = clf.predict_proba(X_valid)[:, 1]
            p_train = clf.predict_proba(X_train)[:, 1]
            elapsed = time.perf_counter() - t0
            fit_seconds.append(elapsed)
            if not (np.isfinite(p_valid).all() and (p_valid >= 0).all() and (p_valid <= 1).all()):
                raise SystemExit(f"{kind}/{name}: invalid Validation probabilities")
            probs_valid[:, j] = p_valid
            probs_train[:, j] = p_train
            ap_valid = float(average_precision_score(y_valid[:, j], p_valid))
            ap_train = float(average_precision_score(y_train[:, j], p_train))
            importances[j] = np.asarray(clf.feature_importances_, dtype=np.float64)
            per_label.append(
                {
                    "label_index": j,
                    "label": name,
                    "train_pos": int(y_train[:, j].sum()),
                    "valid_pos": int(y_valid[:, j].sum()),
                    "train_ap": ap_train,
                    "train_ap_pct": 100.0 * ap_train,
                    "valid_ap": ap_valid,
                    "valid_ap_pct": 100.0 * ap_valid,
                    "valid_pos_rate": float(y_valid[:, j].mean()),
                    "fit_seconds": elapsed,
                }
            )
            C.log(f"{kind:<12} {name:<26} valid AP {100 * ap_valid:6.2f}%")
        macro_valid = float(np.mean([row["valid_ap"] for row in per_label]))
        macro_train = float(np.mean([row["train_ap"] for row in per_label]))
        macro_recheck = float(
            np.mean([average_precision_score(y_valid[:, j], probs_valid[:, j]) for j in range(C.N_LABELS)])
        )
        if abs(macro_valid - macro_recheck) > 1e-12:
            raise SystemExit(f"{kind}: macro AP recheck mismatch")
        results[kind] = {
            "features": feature_summary,
            "valid_macro_ap": macro_valid,
            "valid_macro_ap_pct": 100.0 * macro_valid,
            "train_macro_ap": macro_train,
            "train_macro_ap_pct": 100.0 * macro_train,
            "valid_macro_ap_recheck_delta": abs(macro_valid - macro_recheck),
            "per_label": per_label,
            "fit_seconds": fit_seconds,
            "top_gain_features": [
                {
                    "label": C.LABELS[j],
                    "feature_index": int(f),
                    "gain": float(importances[j, f]),
                }
                for j in range(C.N_LABELS)
                for f in np.argsort(importances[j])[::-1][:10]
            ],
        }
        np.savez_compressed(
            C.ANALYSIS / f"p0_lgbm_{kind}_predictions.npz",
            probs_valid=probs_valid,
            probs_train=probs_train,
            y_valid=y_valid,
            y_train=y_train,
            labels=np.asarray(C.LABELS),
        )

    summary = {
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "generated_utc": C.utcnow(),
        "entry_point": "code/pc_p0_lgbm.py",
        "training_split": "official Train only",
        "evaluation_split": "official Validation only",
        "classifier": {
            "implementation": "lightgbm.LGBMClassifier",
            "kwargs": LGBM_KWARGS,
            "other_parameters": "LightGBM defaults",
            "tuning": "none",
            "early_stopping": "none",
        },
        "results": results,
        "test_access": {"attempts": list(GUARD["attempts"]), "strict": GUARD["strict"]},
        "test_evaluated": False,
        "total_seconds": float(time.perf_counter() - t_start),
        "environment": C.environment(),
    }
    C.write_json(C.ANALYSIS / "p0_lgbm.json", summary)
    C.log(
        "LightGBM Validation macro AP: "
        + ", ".join(f"{k}={100 * v['valid_macro_ap']:.2f}%" for k, v in results.items())
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

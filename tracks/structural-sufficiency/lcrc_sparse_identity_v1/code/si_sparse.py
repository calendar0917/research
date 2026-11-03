"""SPARSE_ONLY: logistic regression on the frozen 512-dim pattern vector (CPU).

One fit, no search: the exact estimator and hyper-parameters are pre-registered in
``PROTOCOL.md``.  This baseline only asks how much predictive information the fixed
S2 patterns carry on their own.  Train only; Validation metrics are recorded here,
the Test pass happens in ``si_score.py`` after the freeze.
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import numpy as np

import si_common as C
import si_features as FEAT

import mb2_metrics as MET
import ptb_data as PD

SPARSE_KWARGS: dict[str, Any] = dict(
    penalty="l2",
    C=0.1,
    solver="liblinear",
    class_weight=None,
    fit_intercept=True,
    tol=1e-4,
    max_iter=4000,
    random_state=C.SPARSE_RANDOM_STATE,
)


def fit_sparse(x_train: np.ndarray, y_train: np.ndarray):
    from sklearn.linear_model import LogisticRegression

    model = LogisticRegression(**SPARSE_KWARGS)
    model.fit(np.asarray(x_train, dtype=np.float64), np.asarray(y_train).reshape(-1))
    return model


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="Fit SPARSE_ONLY (CPU, one shot)")
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args(argv)
    out_path = C.ANALYSIS / "sparse_only.json"
    model_path = C.ANALYSIS / "predictions" / "sparse_only_model.npz"
    if out_path.exists() and not args.force:
        raise SystemExit(f"{C.track_rel(out_path)} exists; SPARSE_ONLY is fitted exactly once")

    view = PD.molecule_view(C.DATASET)
    spec = view.spec
    vocab = C.read_json(C.ANALYSIS / "feature_vocabulary.json")
    if vocab.get("status") != "OK":
        raise SystemExit("feature vocabulary is not feasible/frozen")
    feats = {
        "train": FEAT.encode_split(view.train_cache, spec, vocab),
        "valid": FEAT.encode_split(view.valid_cache, spec, vocab),
    }
    x_train = feats["train"]["x"]
    x_valid = feats["valid"]["x"]
    y_train = np.asarray(view.targets.y_train, dtype=np.float64).reshape(-1)
    y_valid = np.asarray(view.targets.y_valid, dtype=np.float64).reshape(-1)

    model = fit_sparse(x_train, y_train)
    coef = np.asarray(model.coef_, dtype=np.float64).reshape(-1)
    intercept = float(np.asarray(model.intercept_, dtype=np.float64).reshape(-1)[0])
    valid_logits = (x_valid.astype(np.float64) @ coef + intercept).astype(np.float64)
    train_logits = (x_train.astype(np.float64) @ coef + intercept).astype(np.float64)
    valid_metrics = MET.binary_metrics(valid_logits, y_valid)
    train_metrics = MET.binary_metrics(train_logits, y_train)

    model_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        model_path,
        coef=coef,
        intercept=np.asarray([intercept], dtype=np.float64),
        n_iter=np.asarray([int(model.n_iter_[0])]),
        classes=np.asarray(model.classes_, dtype=np.int64),
    )
    np.savez(
        C.PREDICTIONS / "SPARSE_ONLY_valid.npz",
        rows=np.asarray(view.rows["valid"], dtype=np.int64),
        pred=valid_logits,
        y_true=y_valid,
    )
    record = {
        "schema": "lcrc-sparse-identity-sparse-only-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "arm": C.MODEL_SPARSE,
        "dataset": C.DATASET,
        "estimator": "sklearn.linear_model.LogisticRegression",
        "hyperparameters": SPARSE_KWARGS,
        "n_train": int(x_train.shape[0]),
        "n_valid": int(x_valid.shape[0]),
        "n_features": int(x_train.shape[1]),
        "n_iter": int(model.n_iter_[0]),
        "converged_warning": "max_iter reached" if int(model.n_iter_[0]) >= int(SPARSE_KWARGS["max_iter"]) else None,
        "intercept": intercept,
        "coef_l1": float(np.abs(coef).sum()),
        "coef_l2": float(np.sqrt((coef * coef).sum())),
        "coef_n_nonzero": int((coef != 0).sum()),
        "coef_sha256": C.array_hash_f32(coef.astype(np.float32)),
        "vocabulary_sha256": vocab["vocabulary_sha256"],
        "feature_sha256": {
            "train": feats["train"]["stats"]["vector_sha256"],
            "valid": feats["valid"]["stats"]["vector_sha256"],
        },
        "validation": {"metrics": valid_metrics},
        "train": {"metrics": train_metrics},
        "gap": {
            "train_roc_auc": float(train_metrics["roc_auc"]),
            "valid_roc_auc": float(valid_metrics["roc_auc"]),
            "train_minus_valid_roc_auc": float(train_metrics["roc_auc"] - valid_metrics["roc_auc"]),
        },
        "model_path": C.track_rel(model_path),
        "valid_predictions_path": C.track_rel(C.PREDICTIONS / "SPARSE_ONLY_valid.npz"),
        "test_evaluated": False,
        "test_labels_read": False,
    }
    C.write_json(out_path, record)
    C.log(
        f"SPARSE_ONLY valid roc_auc={valid_metrics['roc_auc']:.6f} pr_auc={valid_metrics['pr_auc']:.6f} "
        f"n_iter={int(model.n_iter_[0])}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

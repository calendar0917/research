from __future__ import annotations

from typing import Any

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler


def eval_logistic_cv(
    X: np.ndarray,
    y: np.ndarray,
    n_splits: int = 10,
    seed: int = 0,
) -> dict[str, Any]:
    """Stratified K-fold accuracy (mean±std). Process metric for stage2a — not Xu max-val."""
    n = len(y)
    n_splits = min(n_splits, n)
    # ensure each class has enough
    _, counts = np.unique(y, return_counts=True)
    n_splits = min(n_splits, int(counts.min()))
    n_splits = max(n_splits, 2)
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    scores = cross_val_score(clf, X, y, cv=cv, scoring="accuracy")
    return {
        "acc_mean": float(scores.mean()),
        "acc_std": float(scores.std()),
        "n_splits": n_splits,
        "scores": scores.tolist(),
        "protocol_note": "sklearn StratifiedKFold mean Acc — NOT Xu/GIN max-val paper protocol",
    }


def eval_holdout(
    X: np.ndarray,
    y: np.ndarray,
    seed: int = 0,
    test_frac: float = 0.2,
) -> dict[str, Any]:
    rng = np.random.default_rng(seed)
    n = len(y)
    idx = rng.permutation(n)
    n_te = max(1, int(n * test_frac))
    te, tr = idx[:n_te], idx[n_te:]
    clf = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=2000, random_state=seed),
    )
    clf.fit(X[tr], y[tr])
    acc = float(clf.score(X[te], y[te]))
    return {"holdout_acc": acc, "n_train": int(len(tr)), "n_test": int(len(te))}

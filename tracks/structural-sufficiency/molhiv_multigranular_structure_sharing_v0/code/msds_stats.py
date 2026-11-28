"""Shared evaluation statistics: metrics, paired scaffold bootstrap, subgroup AUC.

All bootstrap work here describes the uncertainty of **fixed model predictions on
the given molecules**; it is not evidence about training-seed stability.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import msds_common as C


def prob(logit: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(np.asarray(logit, dtype=np.float64), -60, 60)))


def auc_ap(y: np.ndarray, logit: np.ndarray) -> dict[str, Any]:
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y).reshape(-1)
    p = prob(logit)
    if np.unique(y).size < 2:
        return {"n": int(y.size), "n_positive": int(y.sum()), "n_negative": int((1 - y).sum()),
                "roc_auc": None, "average_precision": None, "status": "INSUFFICIENT_SUPPORT"}
    return {
        "n": int(y.size),
        "n_positive": int(y.sum()),
        "n_negative": int((1 - y).sum()),
        "roc_auc": float(roc_auc_score(y, p)),
        "average_precision": float(average_precision_score(y, p)),
        "status": "OK",
    }


def grouped_auc(
    y: np.ndarray,
    logit: np.ndarray,
    groups: Sequence[np.ndarray],
    names: Sequence[str],
    *,
    min_positive: int = 5,
    min_negative: int = 5,
) -> list[dict[str, Any]]:
    out = []
    for name, mask in zip(names, groups):
        mask = np.asarray(mask, dtype=bool)
        entry = {"group": name, **auc_ap(y[mask], logit[mask])}
        if entry.get("status") == "OK" and (entry["n_positive"] < min_positive or entry["n_negative"] < min_negative):
            entry["status"] = "INSUFFICIENT_SUPPORT"
            entry["roc_auc"] = None
            entry["average_precision"] = None
            entry["min_positive_required"] = min_positive
            entry["min_negative_required"] = min_negative
        out.append(entry)
    return out


def paired_scaffold_bootstrap(
    y: np.ndarray,
    logit_a: np.ndarray,
    logit_b: np.ndarray,
    group_id: np.ndarray,
    *,
    n_boot: int = 1000,
    seed: int = C.SEED,
    label_a: str = "a",
    label_b: str = "b",
) -> dict[str, Any]:
    """Paired scaffold-group bootstrap of AUC/AP differences ``a - b``.

    Whole scaffold groups are resampled with replacement, so molecules sharing a
    scaffold always move together (the same grouping the official split uses).
    """
    from sklearn.metrics import average_precision_score, roc_auc_score

    y = np.asarray(y).reshape(-1)
    pa = prob(logit_a)
    pb = prob(logit_b)
    groups = np.asarray(group_id).reshape(-1)
    uniq, inverse = np.unique(groups, return_inverse=True)
    members = [np.flatnonzero(inverse == i) for i in range(uniq.size)]
    rng = np.random.default_rng(seed)
    deltas_auc, deltas_ap = [], []
    n_invalid = 0
    for _ in range(n_boot):
        pick = rng.integers(0, uniq.size, size=uniq.size)
        idx = np.concatenate([members[i] for i in pick])
        yy = y[idx]
        if np.unique(yy).size < 2:
            n_invalid += 1
            continue
        deltas_auc.append(float(roc_auc_score(yy, pa[idx]) - roc_auc_score(yy, pb[idx])))
        deltas_ap.append(float(average_precision_score(yy, pa[idx]) - average_precision_score(yy, pb[idx])))
    d_auc = np.asarray(deltas_auc)
    d_ap = np.asarray(deltas_ap)
    return {
        "contrast": f"{label_a} - {label_b}",
        "n_bootstrap": int(n_boot),
        "n_valid": int(d_auc.size),
        "n_invalid_two_class_missing": int(n_invalid),
        "n_scaffold_groups": int(uniq.size),
        "seed": int(seed),
        "point": {
            "roc_auc_a": float(roc_auc_score(y, pa)),
            "roc_auc_b": float(roc_auc_score(y, pb)),
            "delta_roc_auc": float(roc_auc_score(y, pa) - roc_auc_score(y, pb)),
            "average_precision_a": float(average_precision_score(y, pa)),
            "average_precision_b": float(average_precision_score(y, pb)),
            "delta_average_precision": float(average_precision_score(y, pa) - average_precision_score(y, pb)),
        },
        "delta_roc_auc": {
            "mean": float(d_auc.mean()),
            "p2.5": float(np.percentile(d_auc, 2.5)),
            "p97.5": float(np.percentile(d_auc, 97.5)),
            "crosses_zero": bool(np.percentile(d_auc, 2.5) <= 0 <= np.percentile(d_auc, 97.5)),
        },
        "delta_average_precision": {
            "mean": float(d_ap.mean()),
            "p2.5": float(np.percentile(d_ap, 2.5)),
            "p97.5": float(np.percentile(d_ap, 97.5)),
            "crosses_zero": bool(np.percentile(d_ap, 2.5) <= 0 <= np.percentile(d_ap, 97.5)),
        },
        "note": "bootstrap describes sampling uncertainty of fixed predictions; not training-seed stability",
    }


def deltas(arm_a: dict[str, Any], arm_b: dict[str, Any], split: str) -> dict[str, Any]:
    """Difference of two selected-arm metric blocks on one split (a - b)."""
    a = arm_a["selected"][split]
    b = arm_b["selected"][split]
    return {
        "split": split,
        "delta_roc_auc": float(a["roc_auc"] - b["roc_auc"]),
        "delta_average_precision": float(a["average_precision"] - b["average_precision"]),
        "roc_auc_a": float(a["roc_auc"]),
        "roc_auc_b": float(b["roc_auc"]),
        "average_precision_a": float(a["average_precision"]),
        "average_precision_b": float(b["average_precision"]),
        "evidence_label": "EXPOSED_DEVELOPMENT_TEST" if split == "test" else "DEVELOPMENT_VALID",
    }

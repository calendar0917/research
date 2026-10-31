"""Fold-1 dev descriptive error groups (CPU, before any v3 GPU run).

This module only *describes* the frozen v2 A-R3 / C-R3 Fold-1 dev residuals.
It never selects a model, changes a threshold, drops a molecule or alters the
pre-registered v3 experiment.  Group thresholds are taken from the Fold-1
*training* fold whenever a fitted threshold is needed; the dev labels are only
used to report residuals.

Groups
------
* TPSA quartiles (train-fold quantiles);
* MolLogP quartiles (train-fold quantiles);
* NumHDonors natural bins 0, 1, 2, 3, >=4;
* formal charge: neutral / positive / negative / unseen-in-train.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np

import v3_common as C
import v3_features as F


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    pred = np.asarray(pred, dtype=np.float64)
    err = pred - y
    sse = float(np.sum(err * err))
    return {
        "n": int(y.size),
        "rmse": float(np.sqrt(np.mean(err * err))) if y.size else None,
        "mae": float(np.mean(np.abs(err))) if y.size else None,
        "bias": float(np.mean(err)) if y.size else None,
        "sum_squared_error": sse,
        "share_of_total_sse": None,
    }


def _attach_shares(groups: dict[str, dict[str, Any]]) -> None:
    total = sum(g["sum_squared_error"] for g in groups.values())
    for g in groups.values():
        g["share_of_total_sse"] = (g["sum_squared_error"] / total) if total > 0 else None
        g["row_share"] = None  # filled by caller


def _continuous_groups(
    values: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    train_values: np.ndarray,
    *,
    n_bins: int = 4,
) -> dict[str, Any]:
    qs = np.linspace(0.0, 1.0, n_bins + 1)[1:-1]
    cuts = np.unique(np.quantile(np.asarray(train_values, dtype=np.float64), qs))
    labels = []
    edges = [-np.inf, *cuts.tolist(), np.inf]
    for i in range(len(edges) - 1):
        lo, hi = edges[i], edges[i + 1]
        if np.isneginf(lo):
            labels.append(f"<= {hi:.4g}")
        elif np.isposinf(hi):
            labels.append(f"> {lo:.4g}")
        else:
            labels.append(f"({lo:.4g}, {hi:.4g}]")
    idx = np.searchsorted(cuts, np.asarray(values, dtype=np.float64), side="right")
    groups: dict[str, dict[str, Any]] = {}
    total = len(values)
    for i, label in enumerate(labels):
        mask = idx == i
        m = _metrics(y[mask], pred[mask]) if mask.any() else {
            "n": 0,
            "rmse": None,
            "mae": None,
            "bias": None,
            "sum_squared_error": 0.0,
            "share_of_total_sse": None,
        }
        m["row_share"] = m["n"] / total if total else None
        m["train_n"] = int(np.sum(np.searchsorted(cuts, train_values, side="right") == i))
        groups[label] = m
    _attach_shares(groups)
    for g in groups.values():
        g["row_share"] = g["n"] / total if total else None
    return {
        "type": "continuous_quantile_bins",
        "n_bins_requested": int(n_bins),
        "cuts": [float(x) for x in cuts],
        "cut_source": "fold1_train_quantiles",
        "groups": groups,
    }


def _natural_groups(
    values: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    train_values: np.ndarray,
    *,
    definitions: Sequence[tuple[str, Any]],
) -> dict[str, Any]:
    groups: dict[str, dict[str, Any]] = {}
    total = len(values)
    for label, predicate in definitions:
        mask = np.asarray([bool(predicate(v)) for v in values])
        m = _metrics(y[mask], pred[mask]) if mask.any() else {
            "n": 0,
            "rmse": None,
            "mae": None,
            "bias": None,
            "sum_squared_error": 0.0,
            "share_of_total_sse": None,
        }
        m["row_share"] = m["n"] / total if total else None
        m["train_n"] = int(sum(1 for v in train_values if predicate(v)))
        groups[label] = m
    _attach_shares(groups)
    for g in groups.values():
        g["row_share"] = g["n"] / total if total else None
    return {"type": "natural_bins", "groups": groups}


def _formal_charge_groups(
    values: np.ndarray,
    y: np.ndarray,
    pred: np.ndarray,
    train_values: np.ndarray,
) -> dict[str, Any]:
    train_set = {int(v) for v in np.unique(train_values)}
    definitions = [
        ("neutral_0", lambda v: int(v) == 0),
        ("positive_gt0", lambda v: int(v) > 0),
        ("negative_lt0", lambda v: int(v) < 0),
    ]
    groups: dict[str, dict[str, Any]] = {}
    total = len(values)
    assigned = np.zeros(total, dtype=bool)
    for label, predicate in definitions:
        mask = np.asarray([bool(predicate(v)) for v in values])
        assigned |= mask
        m = _metrics(y[mask], pred[mask]) if mask.any() else {
            "n": 0,
            "rmse": None,
            "mae": None,
            "bias": None,
            "sum_squared_error": 0.0,
            "share_of_total_sse": None,
        }
        m["row_share"] = m["n"] / total if total else None
        m["train_n"] = int(sum(1 for v in train_values if predicate(v)))
        m["value_in_train"] = any(predicate(v) for v in train_set)
        groups[label] = m
    # Any dev value not represented in train (none expected for charge).
    unseen = ~assigned
    if unseen.any():
        m = _metrics(y[unseen], pred[unseen])
        m["row_share"] = m["n"] / total
        m["train_n"] = 0
        m["value_in_train"] = False
        groups["unseen_in_train"] = m
    _attach_shares(groups)
    for g in groups.values():
        g["row_share"] = g["n"] / total if total else None
    return {
        "type": "categorical",
        "train_values": sorted(train_set),
        "dev_values": sorted({int(v) for v in np.unique(values)}),
        "groups": groups,
    }


def _historical_predictions(arm: str) -> tuple[np.ndarray, np.ndarray, list[int]]:
    hist = C.HISTORICAL[arm]
    run_dir = C.V2_RUNS / hist["key"]
    data = np.load(run_dir / "predictions.npz")
    rows = [int(x) for x in data["rows"].tolist()]
    return rows, np.asarray(data["pred"], dtype=np.float64), rows


def run_descriptive() -> dict[str, Any]:
    store = F.build_store()
    folds = C.F.load_folds()
    train_pos = np.asarray(C.F.train_positions(folds, C.FOLD), dtype=np.int64)
    dev_pos = np.asarray(C.F.fold_positions(folds, C.FOLD), dtype=np.int64)
    y_all = np.asarray(C.load_fit_labels()["y"], dtype=np.float64)
    y_train = y_all[train_pos]
    y_dev = y_all[dev_pos]
    variables = {
        "TPSA": (store.chem12[:, 0], store.chem12[train_pos, 0]),
        "MolLogP": (store.chem13[:, 12], store.chem13[train_pos, 12]),
        "NumHDonors": (store.chem12[:, 3], store.chem12[train_pos, 3]),
        "formal_charge": (store.formal_charge.astype(np.int64), store.formal_charge[train_pos].astype(np.int64)),
    }
    payload: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "scope": {
            "fold": int(C.FOLD),
            "train_fold_labels_used": True,
            "dev_fold_labels_used_for_metrics_only": True,
            "monitor_labels_read": False,
            "reserved_labels_read": False,
            "official_valid_test_read": False,
            "purpose": "descriptive only; no v3 experiment or threshold changed",
        },
        "n_train": int(train_pos.size),
        "n_dev": int(dev_pos.size),
        "comparators": {},
    }
    for arm in ("A", "C"):
        rows, pred_dev, _ = _historical_predictions(arm)
        if rows != [int(x) for x in dev_pos]:
            raise ValueError(f"{arm}: historical prediction rows do not match fold-1 dev positions")
        variables_dev = {name: values[dev_pos] for name, (values, _train_values) in variables.items()}
        arm_payload = {
            "key": C.HISTORICAL[arm]["key"],
            "historical_dev_rmse": float(C.HISTORICAL[arm]["dev_rmse"]),
            "historical_dev_mae": float(C.HISTORICAL[arm]["dev_mae"]),
            "groups": {},
        }
        for name in ("TPSA", "MolLogP"):
            arm_payload["groups"][name] = _continuous_groups(
                variables_dev[name], y_dev, pred_dev, variables[name][1], n_bins=4
            )
        arm_payload["groups"]["NumHDonors"] = _natural_groups(
            variables_dev["NumHDonors"],
            y_dev,
            pred_dev,
            variables["NumHDonors"][1],
            definitions=[
                ("0", lambda v: int(v) == 0),
                ("1", lambda v: int(v) == 1),
                ("2", lambda v: int(v) == 2),
                ("3", lambda v: int(v) == 3),
                (">=4", lambda v: int(v) >= 4),
            ],
        )
        arm_payload["groups"]["formal_charge"] = _formal_charge_groups(
            variables_dev["formal_charge"], y_dev, pred_dev, variables["formal_charge"][1]
        )
        arm_payload["overall"] = _metrics(y_dev, pred_dev)
        arm_payload["overall"]["share_of_total_sse"] = 1.0
        arm_payload["overall"]["row_share"] = 1.0
        payload["comparators"][arm] = arm_payload
    payload["notes"] = {
        "formal_charge_degenerate": "Fold-1 train has one +1 charge; Fold-1 dev is all neutral, so this grouping cannot separate dev errors and is reported as such.",
        "grouping_uses_train_only_thresholds": True,
        "no_group_was_used_to_modify_the_registered_v3_runs": True,
    }
    C.write_json(C.ANALYSIS / "descriptive_groups_v3.json", payload)
    return payload


__all__ = ["run_descriptive"]

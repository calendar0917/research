"""Fit-only data views for CSS-v0.

The official split adapter (``ses_data.py``) supplies all 4200 official labels,
the frozen 3360/420/420 rows and the shared raw cache.  This module adds the
per-training-size Fit set:

* 100% : every official Train row;
* 50%  : the 1680 official Train rows with the smallest
         ``sha256('css-v0-50pct/<row_id>')`` — a label-free fixed hash choice.

Every statistic used by the branch (label scaler, R2 atom/graph normalisers,
category support mask) is refitted here on the corresponding Fit rows only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

import css_common as C
import gwic_data as GD
import ses_data as SD


@dataclass
class SplitView:
    size: int
    fit_rows: np.ndarray
    heldout_rows: np.ndarray          # empty for size 100
    valid_rows: np.ndarray
    test_rows: np.ndarray
    labels: np.ndarray                # raw, all 4200 rows
    scaler: dict[str, float]
    graph_mean: np.ndarray
    graph_scale: np.ndarray
    atom_scalar_mean: np.ndarray
    atom_scalar_scale: np.ndarray
    atom_ok: list[np.ndarray]
    bond_ok: list[np.ndarray]
    stats: dict[str, Any]

    def y_raw(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.labels, np.float64)[np.asarray(rows, np.int64)]

    def y_std(self, rows: np.ndarray) -> np.ndarray:
        return (self.y_raw(rows) - float(self.scaler["mean"])) / float(self.scaler["std"])


def selection_hash(row: int) -> str:
    return hashlib.sha256(f"css-v0-50pct/{int(row)}".encode()).hexdigest()


def fit_rows_for(size: int, train_rows: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    train = np.sort(np.asarray(train_rows, np.int64))
    if int(size) == 100:
        return train.copy(), np.zeros(0, np.int64)
    if int(size) != 50:
        raise ValueError(f"unsupported training size {size}")
    if train.size != 3360:
        raise ValueError(f"official Train changed: {train.size}")
    keys = np.asarray([selection_hash(int(r)) for r in train.tolist()])
    order = np.lexsort((train, keys))
    fit = np.sort(train[order[:1680]])
    heldout = np.sort(np.setdiff1d(train, fit))
    if fit.size != 1680 or heldout.size != 1680:
        raise ValueError(f"50% split sizes {fit.size}/{heldout.size}")
    if np.intersect1d(fit, heldout).size:
        raise ValueError("50% split overlap")
    return fit, heldout


def _atom_scalars_matrix(raw: Mapping[str, Any], rows: np.ndarray) -> np.ndarray:
    parts = [GD.atom_scalars(GD.molecule_view(raw, int(r))) for r in np.asarray(rows).tolist()]
    return np.concatenate(parts, axis=0)


def build_view(bundle: Any, size: int, train_rows: np.ndarray,
               valid_rows: np.ndarray, test_rows: np.ndarray) -> SplitView:
    fit, heldout = fit_rows_for(size, train_rows)
    labels = np.asarray(bundle.labels, np.float64)
    y = labels[fit]
    scaler = {"mean": float(y.mean()), "std": float(y.std(ddof=0)), "n": int(fit.size)}
    if not (scaler["std"] > 0):
        raise ValueError("degenerate target scale")

    graph = np.asarray(bundle.graph_all, np.float64)
    graph_mean = graph[fit].mean(axis=0)
    graph_scale = graph[fit].std(axis=0)
    graph_scale = np.where(graph_scale > 1e-12, graph_scale, 1.0)

    atom_rows = _atom_scalars_matrix(bundle.raw, fit)
    atom_mean = atom_rows.mean(axis=0)
    atom_scale = atom_rows.std(axis=0)
    atom_scale = np.where(atom_scale > 1e-12, atom_scale, 1.0)

    atom_dims, bond_dims = SD.atom_bond_dims()
    atom_ok, bond_ok = GD.category_support(bundle.raw, fit, atom_dims, bond_dims)

    stats = {
        "size": int(size),
        "n_fit": int(fit.size),
        "n_heldout": int(heldout.size),
        "fit_rows_sha256": C.array_sha256(fit),
        "heldout_rows_sha256": C.array_sha256(heldout),
        "fit_label_mean": float(labels[fit].mean()),
        "fit_label_std": float(labels[fit].std(ddof=0)),
        "valid_label_mean": float(labels[valid_rows].mean()),
        "valid_label_std": float(labels[valid_rows].std(ddof=0)),
        "test_label_mean": float(labels[test_rows].mean()),
        "test_label_std": float(labels[test_rows].std(ddof=0)),
        "target_scaler": scaler,
        "graph_norm": {"mean": graph_mean, "scale": graph_scale},
        "atom_norm": {"mean": atom_mean, "scale": atom_scale},
        "atom_support_sizes": [int(m.sum()) for m in atom_ok],
        "bond_support_sizes": [int(m.sum()) for m in bond_ok],
        "atom_dims": atom_dims,
        "bond_dims": bond_dims,
        "unseen_valid": GD.unseen_category_counts(bundle.raw, valid_rows, atom_ok, bond_ok),
        "unseen_test": GD.unseen_category_counts(bundle.raw, test_rows, atom_ok, bond_ok),
    }
    return SplitView(size=int(size), fit_rows=fit, heldout_rows=heldout,
                     valid_rows=np.asarray(valid_rows, np.int64),
                     test_rows=np.asarray(test_rows, np.int64),
                     labels=labels, scaler=scaler, graph_mean=graph_mean,
                     graph_scale=graph_scale, atom_scalar_mean=atom_mean,
                     atom_scalar_scale=atom_scale, atom_ok=atom_ok, bond_ok=bond_ok,
                     stats=stats)

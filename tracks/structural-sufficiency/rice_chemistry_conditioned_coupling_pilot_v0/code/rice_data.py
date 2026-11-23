"""Label-safe D0 data layer for RICE-v0.

Raw structure loading, D0 split validation and label isolation are **reused**
from the pinned parent data module (``gwic_data.py``); nothing about a parent
model is imported.  Only two statistics are computed here, both from Fit rows
alone:

* the target mean / std (z-score for labels, undone for every reported metric)
* the mean / scale of the 8 deterministic whole-graph scalars

Official Valid labels are never loaded (they are NaN in the label file the
parent module reads) and official Test rows are never touched.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np

import rice_common as C


@dataclass
class DataBundle:
    parent: Any                       # pinned parent data module (helpers)
    raw: dict[str, Any]
    labels: np.ndarray                # finite only on official Train
    official_train: np.ndarray
    official_valid: np.ndarray
    fit_rows: np.ndarray
    eval_rows: np.ndarray
    split: dict[str, Any]
    scaler: dict[str, float]
    graph_scalars: np.ndarray         # raw graph-level deterministics, all rows
    graph_mean: np.ndarray
    graph_scale: np.ndarray
    atom_dims: list[int]
    bond_dims: list[int]
    hashes: dict[str, Any]
    scaffold_by_row: dict[str, str]

    def labels_std(self) -> np.ndarray:
        return (self.labels - self.scaler["mean"]) / self.scaler["std"]


def load_bundle() -> DataBundle:
    parent = C.import_parent_data()
    raw = parent._load_raw()                       # noqa: SLF001 (read-only reuse)
    labels_file = parent._load_train_labels()      # noqa: SLF001
    official_train = np.sort(np.asarray(raw["train_idx"], np.int64))
    official_valid = np.sort(np.asarray(raw["valid_idx"], np.int64))
    n_mol = int(np.asarray(raw["node_ptr"]).size - 1)
    if np.asarray(labels_file["row_id"]).size != official_train.size:
        raise ValueError("train label count mismatch")
    if not np.array_equal(np.sort(labels_file["row_id"]), official_train):
        raise ValueError("train label rows are not the official Train rows")
    labels = np.full(n_mol, np.nan, np.float64)
    labels[labels_file["row_id"]] = labels_file["label"]
    if not np.isfinite(labels[official_train]).all():
        raise ValueError("missing finite label on an official Train row")
    if np.isfinite(labels[official_valid]).any():
        raise ValueError("official Valid row carries a finite label")
    if not np.array_equal(np.asarray(raw["row_ids"]), np.arange(n_mol)):
        raise ValueError("raw row ids are not 0..N-1")

    split = parent._load_split(raw)                # noqa: SLF001
    fit = np.asarray(split["train_sub_rows"], np.int64)
    dev = np.asarray(split["dev_rows"], np.int64)
    y = labels[fit]
    if not np.isfinite(y).all():
        raise ValueError("non-finite fit label")
    scaler = {"mean": float(y.mean()), "std": float(y.std(ddof=0)), "n": int(y.size)}
    if not (scaler["std"] > 0):
        raise ValueError("zero target scale")

    graph = np.zeros((n_mol, 8), np.float64)
    for row in range(n_mol):
        graph[row] = parent.graph_scalars(parent.molecule_view(raw, row))
    graph_mean = graph[fit].mean(axis=0)
    graph_scale = graph[fit].std(axis=0)
    graph_scale = np.where(graph_scale > 1e-12, graph_scale, 1.0)

    atom_dims, bond_dims = parent.atom_bond_dims()
    if len(atom_dims) != C.ATOM_COLUMNS or len(bond_dims) != C.BOND_COLUMNS:
        raise ValueError("unexpected official OGB feature widths")

    scaffold_by_row = {str(k): str(v) for k, v in dict(split["scaffold_by_row"]).items()}
    if len(scaffold_by_row) != 3360:
        raise ValueError("scaffold map does not cover official Train")

    hashes = {
        "raw_cache": {"path": str(C.GWIC_DIR.relative_to(C.REPO_ROOT)) + " (pinned data layer)",
                      "sha256": raw["cache_sha256"]},
        "train_labels": {"sha256": labels_file["sha256"],
                         "rows": int(labels_file["row_id"].size)},
        "split": {"path": str(C.GWIC_DIR / ".." / "acra_relation_generalization_pilot_v1"
                              / "data" / "split_train_sub_dev.json"),
                  "split_sha256": C.read_json(
                      C.STRUCT / "acra_relation_generalization_pilot_v1" / "data"
                      / "split_train_sub_dev.json")["split_sha256"],
                  "file_sha256": split["file_sha256"],
                  "n_fit": int(fit.size), "n_eval": int(dev.size),
                  "fit_rows_sha256": C.array_sha256(fit),
                  "dev_rows_sha256": C.array_sha256(dev)},
        "frozen_parent_sources": C.assert_frozen_parent(),
    }
    return DataBundle(parent=parent, raw=raw, labels=labels,
                      official_train=official_train, official_valid=official_valid,
                      fit_rows=fit, eval_rows=dev, split=split, scaler=scaler,
                      graph_scalars=graph, graph_mean=graph_mean, graph_scale=graph_scale,
                      atom_dims=list(atom_dims), bond_dims=list(bond_dims),
                      hashes=hashes, scaffold_by_row=scaffold_by_row)


def category_support(bundle: DataBundle,
                     rows: Sequence[int]) -> tuple[list[np.ndarray], list[np.ndarray]]:
    return bundle.parent.category_support(bundle.raw, rows, bundle.atom_dims, bundle.bond_dims)


def unseen_category_counts(bundle: DataBundle, rows: Sequence[int],
                           atom_ok: Sequence[np.ndarray],
                           bond_ok: Sequence[np.ndarray]) -> dict[str, int]:
    return bundle.parent.unseen_category_counts(bundle.raw, rows, atom_ok, bond_ok)


def normalisers(bundle: DataBundle) -> dict[str, np.ndarray]:
    return {"graph_mean": bundle.graph_mean, "graph_scale": bundle.graph_scale}


def molecule_arrays(bundle: DataBundle, row: int) -> dict[str, Any]:
    view = bundle.parent.molecule_view(bundle.raw, int(row))
    return {
        "row": int(row),
        "n_atoms": int(view["n_atoms"]),
        "n_bonds": int(view["n_bonds"]),
        "atom_feat": np.asarray(view["atom_feat"], np.int64),
        "bond_feat": np.asarray(view["bond_feat"], np.int64),
        "bond_u": np.asarray(view["bond_u"], np.int64),
        "bond_v": np.asarray(view["bond_v"], np.int64),
        "smiles": str(view["smiles"]),
        "graph_scalars": bundle.graph_scalars[int(row)].copy(),
    }
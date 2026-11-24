"""Official-split data layer for SES-v0.

The brief explicitly authorises full Train/Valid/Test evaluation every epoch
with Test marked DEVELOPMENT_TEST.  This module therefore reads all 4200
official labels from the frozen structure cache.  Labels of Valid/Test are used
only for evaluation, checkpoint selection on Valid, and the requested
diagnostics; they never enter a gradient, a normaliser, a category support, a
candidate construction or a selector.

Every normaliser and category support is fitted on the **full official Train**
graph set (3360 molecules), shared by DENSE / RANDOM / SELECT.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

import ses_common as C
import gwic_data as GD


@dataclass
class DataBundle:
    raw: dict[str, Any]
    labels: np.ndarray
    train_rows: np.ndarray
    valid_rows: np.ndarray
    test_rows: np.ndarray
    scaler: dict[str, float]
    graph_all: np.ndarray
    graph_mean: np.ndarray
    graph_scale: np.ndarray
    atom_scalar_mean: np.ndarray
    atom_scalar_scale: np.ndarray
    atom_ok: list[np.ndarray]
    bond_ok: list[np.ndarray]
    hashes: dict[str, Any]
    stats: dict[str, Any]

    def y_raw(self, rows: np.ndarray) -> np.ndarray:
        return np.asarray(self.labels, np.float64)[np.asarray(rows, np.int64)]

    def y_std(self, rows: np.ndarray) -> np.ndarray:
        return (self.y_raw(rows) - float(self.scaler["mean"])) / float(self.scaler["std"])


def atom_bond_dims() -> tuple[list[int], list[int]]:
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    atom = [int(x) for x in get_atom_feature_dims()]
    bond = [int(x) for x in get_bond_feature_dims()]
    if len(atom) != C.ATOM_COLUMNS or len(bond) != C.BOND_COLUMNS:
        raise ValueError("unexpected OGB feature widths")
    return atom, bond


def _load_raw() -> dict[str, Any]:
    if not C.RAW_CACHE.is_file():
        raise FileNotFoundError(C.RAW_CACHE)
    sha = C.sha256_file(C.RAW_CACHE)
    if sha != C.RAW_CACHE_SHA256:
        raise ValueError(f"raw cache sha mismatch: {sha}")
    with np.load(C.RAW_CACHE, allow_pickle=False) as z:
        missing = [k for k in C.RAW_KEYS if k not in z.files]
        if missing:
            raise KeyError(f"raw cache missing members: {missing}")
        raw = {k: np.asarray(z[k]) for k in C.RAW_KEYS}
    raw["cache_sha256"] = sha
    return raw


def _split_stats(name: str, rows: np.ndarray, labels: np.ndarray) -> dict[str, Any]:
    y = labels[rows]
    return {
        "split": name, "n": int(rows.size),
        "rows_sha256": C.array_sha256(rows),
        "label_mean": float(y.mean()), "label_std": float(y.std(ddof=0)),
        "label_min": float(y.min()), "label_max": float(y.max()),
        "label_sha256": C.array_sha256(y),
    }


def load_bundle() -> DataBundle:
    raw = _load_raw()
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    n_mol = int(node_ptr.size - 1)
    labels = np.asarray(raw["labels"], np.float64)
    if labels.shape != (n_mol,):
        raise ValueError("label vector width mismatch")
    if not np.isfinite(labels).all():
        raise ValueError("non-finite official label in the frozen cache")
    if not np.array_equal(np.asarray(raw["row_ids"], np.int64), np.arange(n_mol)):
        raise ValueError("cache row ids are not 0..N-1")

    train = np.sort(np.asarray(raw["train_idx"], np.int64))
    valid = np.sort(np.asarray(raw["valid_idx"], np.int64))
    test = np.sort(np.asarray(raw["test_idx"], np.int64))
    if (train.size, valid.size, test.size) != (3360, 420, 420):
        raise ValueError(f"unexpected official split sizes {train.size}/{valid.size}/{test.size}")
    if np.intersect1d(train, valid).size or np.intersect1d(train, test).size \
            or np.intersect1d(valid, test).size:
        raise ValueError("official split overlap")
    if np.union1d(np.union1d(train, valid), test).size != n_mol:
        raise ValueError("official splits do not cover all rows")
    if not (np.isfinite(labels[train]).all() and np.isfinite(labels[valid]).all()
            and np.isfinite(labels[test]).all()):
        raise ValueError("non-finite label on an official split")

    vault = {"available": C.LABEL_VAULT.is_file(), "max_abs_diff": None,
             "sha256": None}
    if vault["available"]:
        vault_labels = np.load(C.LABEL_VAULT)
        if vault_labels.shape != labels.shape:
            raise ValueError("label vault width mismatch")
        vault["sha256"] = C.sha256_file(C.LABEL_VAULT)
        vault["max_abs_diff"] = float(np.max(np.abs(vault_labels - labels)))

    y_train = labels[train]
    scaler = {"mean": float(y_train.mean()), "std": float(y_train.std(ddof=0)),
              "n": int(train.size)}
    if not (scaler["std"] > 0):
        raise ValueError("degenerate target scale")

    graph_all = np.zeros((n_mol, 8), np.float64)
    for row in range(n_mol):
        graph_all[row] = GD.graph_scalars(GD.molecule_view(raw, row))
    graph_mean = graph_all[train].mean(axis=0)
    graph_scale = graph_all[train].std(axis=0)
    graph_scale = np.where(graph_scale > 1e-12, graph_scale, 1.0)

    atom_rows = [GD.atom_scalars(GD.molecule_view(raw, int(r))) for r in train.tolist()]
    all_atom_scalars = np.concatenate(atom_rows, axis=0)
    atom_mean = all_atom_scalars.mean(axis=0)
    atom_scale = all_atom_scalars.std(axis=0)
    atom_scale = np.where(atom_scale > 1e-12, atom_scale, 1.0)

    atom_dims, bond_dims = atom_bond_dims()
    atom_ok, bond_ok = GD.category_support(raw, train, atom_dims, bond_dims)
    unseen = {
        "valid": GD.unseen_category_counts(raw, valid, atom_ok, bond_ok),
        "test": GD.unseen_category_counts(raw, test, atom_ok, bond_ok),
        "train": GD.unseen_category_counts(raw, train, atom_ok, bond_ok),
    }

    stats = {
        "dataset": str(raw["dataset"]),
        "cache_version": str(raw["cache_version"]),
        "n_molecules": int(n_mol),
        "train": _split_stats("train", train, labels),
        "valid": _split_stats("valid", valid, labels),
        "test": _split_stats("test", test, labels),
        "split_identity_sha256": C.sha256_json({
            "train_idx": train, "valid_idx": valid, "test_idx": test}),
        "train_atoms": int(sum(int(node_ptr[r + 1] - node_ptr[r]) for r in train.tolist())),
        "train_bonds": int(sum(int(np.asarray(raw["bond_ptr"])[r + 1]
                                  - np.asarray(raw["bond_ptr"])[r]) for r in train.tolist())),
        "unseen_category_positions": unseen,
        "target_scaler": scaler,
        "graph_norm": {"mean": graph_mean, "scale": graph_scale},
        "atom_norm": {"mean": atom_mean, "scale": atom_scale},
        "atom_dims": atom_dims,
        "bond_dims": bond_dims,
    }
    hashes = {
        "raw_cache": {"path": str(C.RAW_CACHE.relative_to(C.REPO_ROOT)),
                      "sha256": raw["cache_sha256"], "members": list(C.RAW_KEYS)},
        "label_vault": vault,
        "split_identity": stats["split_identity_sha256"],
        "official_rows": {
            "train": C.array_sha256(train), "valid": C.array_sha256(valid),
            "test": C.array_sha256(test),
        },
        "labels": {
            "train": C.array_sha256(labels[train]),
            "valid": C.array_sha256(labels[valid]),
            "test": C.array_sha256(labels[test]),
        },
        "graph_norm": C.array_sha256(graph_mean, graph_scale),
        "atom_norm": C.array_sha256(atom_mean, atom_scale),
    }
    return DataBundle(
        raw=raw, labels=labels, train_rows=train, valid_rows=valid, test_rows=test,
        scaler=scaler, graph_all=graph_all, graph_mean=graph_mean,
        graph_scale=graph_scale, atom_scalar_mean=atom_mean,
        atom_scalar_scale=atom_scale, atom_ok=atom_ok, bond_ok=bond_ok,
        hashes=hashes, stats=stats,
    )

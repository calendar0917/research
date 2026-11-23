"""Label-safe data layer for PRCG-v0.

Only label-free structure members of the frozen raw cache are read, plus the
official-Train labels already extracted by PSC and the frozen D0 split.  The
official Valid / Test labels are never loaded; every Valid label stays NaN.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

import prcg_common as C
import prcg_graph as G


@dataclass
class DataBundle:
    raw: dict[str, Any]
    labels: np.ndarray
    official_train: np.ndarray
    official_valid: np.ndarray
    fit_rows: np.ndarray
    eval_rows: np.ndarray
    split: dict[str, Any]
    scaler: dict[str, float]
    graph_scalars: np.ndarray
    graph_mean: np.ndarray
    graph_scale: np.ndarray
    scaffold_by_row: dict[str, str]
    hashes: dict[str, Any]

    def labels_std(self) -> np.ndarray:
        return (self.labels - self.scaler["mean"]) / self.scaler["std"]


def _load_raw() -> dict[str, Any]:
    if not C.RAW_CACHE.is_file():
        raise FileNotFoundError(f"raw cache missing: {C.RAW_CACHE}")
    sha = C.sha256_file(C.RAW_CACHE)
    if sha != C.RAW_CACHE_SHA256:
        raise ValueError(f"raw cache sha mismatch: {sha}")
    with np.load(C.RAW_CACHE, allow_pickle=False) as z:
        missing = [k for k in C.RAW_STRUCT_KEYS if k not in z.files]
        if missing:
            raise KeyError(f"raw cache missing members: {missing}")
        raw = {k: np.asarray(z[k]) for k in C.RAW_STRUCT_KEYS}
    if "labels" in raw or "test_idx" in raw:
        raise ValueError("label-free whitelist violated")
    raw["cache_sha256"] = sha
    return raw


def _load_train_labels() -> dict[str, np.ndarray]:
    if not C.TRAIN_LABELS.is_file():
        raise FileNotFoundError(f"train labels missing: {C.TRAIN_LABELS}")
    sha = C.sha256_file(C.TRAIN_LABELS)
    if not sha.startswith(C.TRAIN_LABELS_ROW_SHA256_EXPECTED):
        raise ValueError(f"train label file sha mismatch: {sha}")
    with np.load(C.TRAIN_LABELS, allow_pickle=False) as z:
        return {"row_id": np.asarray(z["row_id"], np.int64),
                "label": np.asarray(z["label"], np.float64), "sha256": sha}


def _load_split() -> dict[str, Any]:
    payload = C.read_json(C.SPLIT_F0)
    recomputed = C.sha256_json({"train_sub_rows": payload["train_sub_rows"],
                                "dev_rows": payload["dev_rows"], "seed": payload["seed"]})
    if recomputed != C.SPLIT_F0_SHA256_EXPECTED or payload.get("split_sha256") != recomputed:
        raise ValueError("D0 split hash mismatch")
    return payload


def atom_bond_dims() -> tuple[list[int], list[int]]:
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    atom = [int(x) for x in get_atom_feature_dims()]
    bond = [int(x) for x in get_bond_feature_dims()]
    if len(atom) != C.ATOM_COLUMNS or len(bond) != C.BOND_COLUMNS:
        raise ValueError("unexpected official OGB feature dimensions")
    return atom, bond


def _molecule(raw: Mapping[str, Any], row: int) -> G.Molecule:
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    n_atoms = int(node_ptr[row + 1] - node_ptr[row])
    n_bonds = int(bond_ptr[row + 1] - bond_ptr[row])
    a0 = int(node_ptr[row])
    b0 = int(bond_ptr[row])
    atom_feat = np.asarray(raw["atom_feat"])[a0:a0 + n_atoms].astype(np.int64)
    bond_feat = np.asarray(raw["bond_feat"])[b0:b0 + n_bonds].astype(np.int64)
    bu = np.asarray(raw["bond_u"])[b0:b0 + n_bonds].astype(np.int64)
    bv = np.asarray(raw["bond_v"])[b0:b0 + n_bonds].astype(np.int64)
    if bu.size and (bu.max() >= n_atoms or bv.max() >= n_atoms or bu.min() < 0 or bv.min() < 0):
        raise ValueError(f"row {row}: bond endpoint outside the molecule")
    n_components = int(np.asarray(raw["graph_n_components"])[row])
    degree = np.asarray(raw["degree"])[a0:a0 + n_atoms].astype(np.float64)
    min_ring = np.asarray(raw["min_ring"])[a0:a0 + n_atoms].astype(np.float64)
    in_cycle = np.asarray(raw["in_cycle"])[a0:a0 + n_atoms].astype(np.float64)
    context = np.stack([degree, min_ring, in_cycle, np.log1p(degree),
                        (degree == 0).astype(np.float64)], axis=1)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n_atoms)]
    for b in range(n_bonds):
        u, v = int(bu[b]), int(bv[b])
        adj[u].append((v, b))
        adj[v].append((u, b))
    core = atom_feat[:, [0, 3, 7]]
    labels = [G._atom_code(core[i]) for i in range(n_atoms)]
    bond_core = bond_feat[:, [0, 2]]
    esets = [[(u, G._bond_code(bond_core[b])) for u, b in adj[i]] for i in range(n_atoms)]
    ranks = G.canonical_ranks(labels, esets, n_atoms)
    return G.Molecule(row=int(row), n_atoms=n_atoms, atom_core=core,
                      atom_intrinsic=atom_feat[:, [1, 4, 5, 6]], atom_context=context,
                      bond_u=bu, bond_v=bv, bond_core=bond_core,
                      bond_stereo=bond_feat[:, 1].astype(np.int64), adj=adj, ranks=ranks,
                      n_components=n_components)


def graph_scalars(mol: G.Molecule) -> np.ndarray:
    """The eight frozen deterministic whole-graph scalars."""
    degree = mol.atom_context[:, 0]
    n_atoms = max(1, mol.n_atoms)
    aromatic = (mol.atom_core[:, 2] > 0).astype(np.float64)
    in_cycle = mol.atom_context[:, 2]
    n_components = int(mol.n_components)
    cycle_rank = mol.n_bonds - mol.n_atoms + n_components
    return np.asarray([
        np.log1p(mol.n_atoms), np.log1p(mol.n_bonds), float(cycle_rank),
        float(n_components), float(degree.mean()) if degree.size else 0.0,
        float(degree.max()) if degree.size else 0.0,
        float(in_cycle.mean()) if in_cycle.size else 0.0,
        float(aromatic.mean()) if aromatic.size else 0.0,
    ], np.float64)


def build_molecules(raw: Mapping[str, Any], rows: Sequence[int]) -> dict[int, G.Molecule]:
    return {int(r): _molecule(raw, int(r)) for r in np.asarray(rows, np.int64).tolist()}


def load_bundle() -> DataBundle:
    raw = _load_raw()
    labels_file = _load_train_labels()
    official_train = np.sort(np.asarray(raw["train_idx"], np.int64))
    official_valid = np.sort(np.asarray(raw["valid_idx"], np.int64))
    n_mol = int(np.asarray(raw["node_ptr"]).size - 1)
    labels = np.full(n_mol, np.nan, np.float64)
    labels[labels_file["row_id"]] = labels_file["label"]
    if not np.array_equal(np.sort(labels_file["row_id"]), official_train):
        raise ValueError("train label rows are not the official Train rows")
    if not np.isfinite(labels[official_train]).all():
        raise ValueError("missing finite label on an official Train row")
    if np.isfinite(labels[official_valid]).any():
        raise ValueError("official Valid row carries a finite label")
    split = _load_split()
    fit = np.asarray(split["train_sub_rows"], np.int64)
    dev = np.asarray(split["dev_rows"], np.int64)
    official = official_train
    if fit.size != 2856 or dev.size != 504:
        raise ValueError(f"unexpected D0 sizes: {fit.size}/{dev.size}")
    if np.intersect1d(fit, dev).size or not np.isin(np.union1d(fit, dev), official).all():
        raise ValueError("D0 partition invalid")
    if np.union1d(fit, dev).size != official.size:
        raise ValueError("D0 does not cover official Train")
    y = labels[fit]
    scaler = {"mean": float(y.mean()), "std": float(y.std(ddof=0)), "n": int(y.size)}
    scalars = np.zeros((n_mol, 8), np.float64)
    for row in range(n_mol):
        scalars[row] = graph_scalars(_molecule(raw, row))
    graph_mean = scalars[fit].mean(axis=0)
    graph_scale = scalars[fit].std(axis=0)
    graph_scale = np.where(graph_scale > 1e-12, graph_scale, 1.0)
    scaffold_by_row = {str(k): str(v) for k, v in split["scaffold_by_row"].items()}
    hashes = {
        "raw_cache": {"path": str(C.RAW_CACHE.relative_to(C.REPO_ROOT)),
                      "sha256": raw["cache_sha256"], "members": list(C.RAW_STRUCT_KEYS)},
        "train_labels": {"path": str(C.TRAIN_LABELS.relative_to(C.REPO_ROOT)),
                         "sha256": labels_file["sha256"],
                         "rows": int(labels_file["row_id"].size)},
        "split": {"path": str(C.SPLIT_F0.relative_to(C.REPO_ROOT)),
                  "sha256": C.SPLIT_F0_SHA256_EXPECTED,
                  "file_sha256": C.sha256_file(C.SPLIT_F0),
                  "n_fit": int(fit.size), "n_eval": int(dev.size)},
    }
    return DataBundle(raw=raw, labels=labels, official_train=official_train,
                      official_valid=official_valid, fit_rows=fit, eval_rows=dev,
                      split=split, scaler=scaler, graph_scalars=scalars,
                      graph_mean=graph_mean, graph_scale=graph_scale,
                      scaffold_by_row=scaffold_by_row, hashes=hashes)
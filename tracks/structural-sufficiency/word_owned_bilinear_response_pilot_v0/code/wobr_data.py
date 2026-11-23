"""Official-split data layer for WOBR-v0.

The frozen cache is read once and its SHA-256 is checked.  Structure arrays
(atom/bond fields and connectivity) are kept separate from the official
labels and split row ids.  The target scaler is fit on the 3,360 official
Train rows only.  Valid and Test labels are read for per-epoch evaluation and
reporting, but never enter gradients, standardization, category support,
checkpoint selection or early stopping.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

import wobr_common as C


@dataclass
class DataBundle:
    raw: dict[str, Any]                 # label-free structure arrays
    labels: np.ndarray                  # raw labels, all 4200 rows
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    scaler: dict[str, float]
    atom_dims: list[int]
    bond_dims: list[int]
    atom_scalar_mean: np.ndarray
    atom_scalar_scale: np.ndarray
    scaffold_by_row: dict[str, str]     # official Train rows only
    unseen: dict[str, Any]
    hashes: dict[str, Any]

    def split(self, name: str) -> np.ndarray:
        return {"train": self.train_idx, "valid": self.valid_idx,
                "test": self.test_idx}[name]

    def labels_std(self) -> np.ndarray:
        return (self.labels - self.scaler["mean"]) / self.scaler["std"]


def atom_bond_dims() -> tuple[list[int], list[int]]:
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    atom = [int(x) for x in get_atom_feature_dims()]
    bond = [int(x) for x in get_bond_feature_dims()]
    if len(atom) != C.ATOM_COLUMNS or len(bond) != C.BOND_COLUMNS:
        raise ValueError("unexpected official OGB feature dimensions")
    return atom, bond


def molecule_view(raw: Mapping[str, Any], row: int) -> dict[str, Any]:
    """One molecule's label-free local view.  Atom ids are molecule-local."""
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
    b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
    bu = np.asarray(raw["bond_u"][b0:b1], np.int64)
    bv = np.asarray(raw["bond_v"][b0:b1], np.int64)
    return {
        "row": int(row),
        "n_atoms": int(a1 - a0),
        "n_bonds": int(b1 - b0),
        "atom_feat": np.asarray(raw["atom_feat"][a0:a1], np.int64),
        "bond_feat": np.asarray(raw["bond_feat"][b0:b1], np.int64),
        "bond_u": bu,
        "bond_v": bv,
        "degree": np.asarray(raw["degree"][a0:a1], np.int64),
        "min_ring": np.asarray(raw["min_ring"][a0:a1], np.int64),
        "in_cycle": np.asarray(raw["in_cycle"][a0:a1], np.int64),
        "smiles": str(np.asarray(raw["smiles"])[row]),
        "graph_n_atoms": int(np.asarray(raw["graph_n_atoms"])[row]),
        "graph_n_bonds": int(np.asarray(raw["graph_n_bonds"])[row]),
        "graph_cycle_rank": int(np.asarray(raw["graph_cycle_rank"])[row]),
        "graph_n_components": int(np.asarray(raw["graph_n_components"])[row]),
    }


def atom_scalars(view: Mapping[str, Any]) -> np.ndarray:
    degree = np.asarray(view["degree"], np.float64)
    min_ring = np.asarray(view["min_ring"], np.float64)
    in_cycle = np.asarray(view["in_cycle"], np.float64)
    return np.stack([
        degree,
        min_ring,
        in_cycle,
        np.log1p(degree),
        (degree == 0).astype(np.float64),
    ], axis=1)


def graph_scalars(view: Mapping[str, Any]) -> np.ndarray:
    atom = np.asarray(view["atom_feat"], np.int64)
    degree = np.asarray(view["degree"], np.float64)
    in_cycle = np.asarray(view["in_cycle"], np.float64)
    aromatic = atom[:, 7].astype(np.float64) if atom.size else np.zeros(0)
    return np.asarray([
        np.log1p(view["n_atoms"]),
        np.log1p(view["n_bonds"]),
        float(view["graph_cycle_rank"]),
        float(view["graph_n_components"]),
        float(degree.mean()) if degree.size else 0.0,
        float(degree.max()) if degree.size else 0.0,
        float(in_cycle.mean()) if in_cycle.size else 0.0,
        float(aromatic.mean()) if aromatic.size else 0.0,
    ], np.float64)


def category_support(raw: Mapping[str, Any], fit_rows: Sequence[int],
                     atom_dims: Sequence[int], bond_dims: Sequence[int]
                     ) -> tuple[list[np.ndarray], list[np.ndarray]]:
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    atom_feat = np.asarray(raw["atom_feat"])
    bond_feat = np.asarray(raw["bond_feat"])
    atom_ok = [np.zeros(int(d), bool) for d in atom_dims]
    bond_ok = [np.zeros(int(d), bool) for d in bond_dims]
    for row in np.asarray(fit_rows, np.int64).tolist():
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        for i in range(atom_feat.shape[1]):
            atom_ok[i][np.unique(atom_feat[a0:a1, i].astype(np.int64))] = True
        for i in range(bond_feat.shape[1]):
            bond_ok[i][np.unique(bond_feat[b0:b1, i].astype(np.int64))] = True
    return atom_ok, bond_ok


def unseen_category_counts(raw: Mapping[str, Any], rows: Sequence[int],
                           atom_ok: Sequence[np.ndarray], bond_ok: Sequence[np.ndarray]
                           ) -> dict[str, int]:
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    atom_feat = np.asarray(raw["atom_feat"])
    bond_feat = np.asarray(raw["bond_feat"])
    atom_unknown = 0
    bond_unknown = 0
    for row in np.asarray(rows, np.int64).tolist():
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        block = atom_feat[a0:a1]
        unk = np.zeros(block.shape[0], bool)
        for i, mask in enumerate(atom_ok):
            unk |= ~mask[block[:, i].astype(np.int64)]
        atom_unknown += int(unk.sum())
        bblock = bond_feat[b0:b1]
        bunk = np.zeros(bblock.shape[0], bool)
        for i, mask in enumerate(bond_ok):
            bunk |= ~mask[bblock[:, i].astype(np.int64)]
        bond_unknown += int(bunk.sum())
    return {"atom_positions": atom_unknown, "bond_positions": bond_unknown}


def _load_scaffolds() -> dict[str, str]:
    payload = C.read_json(C.SCAFFOLD_SPLIT)
    recomputed = C.sha256_json({
        "train_sub_rows": payload["train_sub_rows"],
        "dev_rows": payload["dev_rows"],
        "seed": payload["seed"],
    })
    if recomputed != C.SCAFFOLD_SPLIT_SHA256 or payload.get("split_sha256") != recomputed:
        raise ValueError("scaffold split hash mismatch")
    scaffold = {str(k): str(v) for k, v in dict(payload["scaffold_by_row"]).items()}
    if len(scaffold) != 3360:
        raise ValueError("scaffold map does not cover official Train")
    return scaffold


def load_bundle() -> DataBundle:
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
        labels = np.asarray(z["labels"], np.float64)
        train = np.asarray(z["train_idx"], np.int64)
        valid = np.asarray(z["valid_idx"], np.int64)
        test = np.asarray(z["test_idx"], np.int64)
    raw["cache_sha256"] = sha

    n_mol = int(np.asarray(raw["node_ptr"]).size - 1)
    if labels.size != n_mol or not np.isfinite(labels).all():
        raise ValueError("labels are not finite for every molecule")
    sizes = {"train": int(train.size), "valid": int(valid.size), "test": int(test.size)}
    if sizes != C.EXPECTED_SPLIT_SIZES:
        raise ValueError(f"official split sizes {sizes} != {C.EXPECTED_SPLIT_SIZES}")
    if (np.intersect1d(train, valid).size or np.intersect1d(train, test).size
            or np.intersect1d(valid, test).size):
        raise ValueError("split row ids overlap")
    union = np.union1d(np.union1d(train, valid), test)
    if union.size != n_mol or not np.array_equal(union, np.arange(n_mol)):
        raise ValueError("split row ids do not cover 0..N-1")
    if not np.array_equal(np.asarray(raw["row_ids"]), np.arange(n_mol)):
        raise ValueError("raw row ids are not 0..N-1")

    # The only target statistic allowed to enter the model or its metrics.
    y_train = labels[train]
    scaler = {"mean": float(y_train.mean()), "std": float(y_train.std(ddof=0)),
              "n": int(train.size)}
    if not (scaler["std"] > 0):
        raise ValueError("zero target scale")

    atom_dims, bond_dims = atom_bond_dims()
    atom_ok, bond_ok = category_support(raw, train, atom_dims, bond_dims)
    unseen = {
        "valid": unseen_category_counts(raw, valid, atom_ok, bond_ok),
        "test": unseen_category_counts(raw, test, atom_ok, bond_ok),
        "train_supported_atom_categories": [int(x.sum()) for x in atom_ok],
        "train_supported_bond_categories": [int(x.sum()) for x in bond_ok],
    }

    # Feature scalars use official Train atoms only.
    train_atom_scalars = [atom_scalars(molecule_view(raw, int(r)))
                          for r in np.sort(train).tolist()]
    all_atom_scalars = np.concatenate(train_atom_scalars, axis=0)
    atom_mean = all_atom_scalars.mean(axis=0)
    atom_scale = all_atom_scalars.std(axis=0)
    atom_scale = np.where(atom_scale > 1e-12, atom_scale, 1.0)

    scaffold = _load_scaffolds()
    if set(scaffold) != set(str(x) for x in train.tolist()):
        raise ValueError("scaffold map rows do not match official Train rows")

    hashes = {
        "raw_cache": {"path": str(C.RAW_CACHE.relative_to(C.REPO_ROOT)),
                      "sha256": sha,
                      "members": list(C.RAW_STRUCT_KEYS),
                      "labels_member_sha256": C.sha256_bytes(labels.astype(np.float64).tobytes())},
        "split": {"train_rows": int(train.size), "valid_rows": int(valid.size),
                  "test_rows": int(test.size),
                  "train_rows_sha256": C.array_sha256(train),
                  "valid_rows_sha256": C.array_sha256(valid),
                  "test_rows_sha256": C.array_sha256(test),
                  "train_idx_sha256": C.sha256_bytes(train.tobytes()),
                  "valid_idx_sha256": C.sha256_bytes(valid.tobytes()),
                  "test_idx_sha256": C.sha256_bytes(test.tobytes())},
        "scaffold_split": {"path": str(C.SCAFFOLD_SPLIT.relative_to(C.REPO_ROOT)),
                           "sha256": C.SCAFFOLD_SPLIT_SHA256,
                           "n_rows": len(scaffold)},
    }
    return DataBundle(raw=raw, labels=labels,
                      train_idx=np.sort(train), valid_idx=np.sort(valid),
                      test_idx=np.sort(test), scaler=scaler,
                      atom_dims=atom_dims, bond_dims=bond_dims,
                      atom_scalar_mean=atom_mean, atom_scalar_scale=atom_scale,
                      scaffold_by_row=scaffold, unseen=unseen, hashes=hashes)


def normalisers(bundle: DataBundle) -> dict[str, np.ndarray]:
    return {"atom_scalar_mean": bundle.atom_scalar_mean,
            "atom_scalar_scale": bundle.atom_scalar_scale}
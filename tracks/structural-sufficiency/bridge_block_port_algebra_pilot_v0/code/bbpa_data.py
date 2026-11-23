"""Label-safe data layer for BBPA-v0.

Reads only label-free OGB structure members from the frozen raw cache, the
official Train labels already extracted by PSC, and the frozen D0 split.
Official Valid / Test labels are never loaded.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

import bbpa_common as C


@dataclass
class DataBundle:
    raw: dict[str, Any]
    labels: np.ndarray                 # finite only on official Train
    row_id: np.ndarray
    official_train: np.ndarray
    official_valid: np.ndarray
    fit_rows: np.ndarray               # D0 train-sub
    eval_rows: np.ndarray              # D0 dev
    split: dict[str, Any]
    scaler: dict[str, float]
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
        # explicit whitelist: labels / test structure are not touched
        raw = {k: np.asarray(z[k]) for k in C.RAW_STRUCT_KEYS}
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
                "label": np.asarray(z["label"], np.float64),
                "sha256": sha}


def _load_split(raw: Mapping[str, Any]) -> dict[str, Any]:
    payload = C.read_json(C.SPLIT_F0)
    recomputed = C.sha256_json({
        "train_sub_rows": payload["train_sub_rows"],
        "dev_rows": payload["dev_rows"],
        "seed": payload["seed"],
    })
    if recomputed != C.SPLIT_F0_SHA256_EXPECTED or payload.get("split_sha256") != recomputed:
        raise ValueError("D0 split hash mismatch")
    fit = np.asarray(payload["train_sub_rows"], np.int64)
    dev = np.asarray(payload["dev_rows"], np.int64)
    official = np.sort(np.asarray(raw["train_idx"], np.int64))
    if fit.size != 2856 or dev.size != 504:
        raise ValueError(f"unexpected D0 sizes: {fit.size}/{dev.size}")
    if np.intersect1d(fit, dev).size:
        raise ValueError("D0 fit/dev overlap")
    if not np.isin(np.union1d(fit, dev), official).all():
        raise ValueError("D0 rows outside official Train")
    if np.union1d(fit, dev).size != official.size:
        raise ValueError("D0 does not cover official Train")
    payload = dict(payload)
    payload["file_sha256"] = C.sha256_file(C.SPLIT_F0)
    payload["recomputed_sha256"] = recomputed
    return payload


def load_bundle() -> DataBundle:
    raw = _load_raw()
    labels_file = _load_train_labels()
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
    if np.asarray(raw["row_ids"]).size != n_mol:
        raise ValueError("row id count mismatch")
    if not np.array_equal(np.asarray(raw["row_ids"]), np.arange(n_mol)):
        raise ValueError("raw row ids are not 0..N-1")

    split = _load_split(raw)
    fit = np.asarray(split["train_sub_rows"], np.int64)
    dev = np.asarray(split["dev_rows"], np.int64)
    y = labels[fit]
    if not np.isfinite(y).all():
        raise ValueError("non-finite fit label")
    scaler = {"mean": float(y.mean()), "std": float(y.std(ddof=0)), "n": int(y.size)}
    if not (scaler["std"] > 0):
        raise ValueError("zero target scale")

    hashes = {
        "raw_cache": {"path": str(C.RAW_CACHE.relative_to(C.REPO_ROOT)),
                      "sha256": raw["cache_sha256"],
                      "members": list(C.RAW_STRUCT_KEYS)},
        "train_labels": {"path": str(C.TRAIN_LABELS.relative_to(C.REPO_ROOT)),
                         "sha256": labels_file["sha256"],
                         "rows": int(labels_file["row_id"].size)},
        "split": {"path": str(C.SPLIT_F0.relative_to(C.REPO_ROOT)),
                  "sha256": C.SPLIT_F0_SHA256_EXPECTED,
                  "file_sha256": split["file_sha256"],
                  "n_fit": int(fit.size), "n_eval": int(dev.size)},
    }
    return DataBundle(raw=raw, labels=labels,
                      row_id=np.asarray(raw["row_ids"], np.int64),
                      official_train=official_train, official_valid=official_valid,
                      fit_rows=fit, eval_rows=dev, split=split, scaler=scaler,
                      hashes=hashes)


def atom_bond_dims() -> tuple[list[int], list[int]]:
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    atom = [int(x) for x in get_atom_feature_dims()]
    bond = [int(x) for x in get_bond_feature_dims()]
    if len(atom) != C.ATOM_COLUMNS or len(bond) != C.BOND_COLUMNS:
        raise ValueError("unexpected official OGB feature dimensions")
    return atom, bond


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


def molecule_view(raw: Mapping[str, Any], row: int) -> dict[str, Any]:
    """One molecule's label-free local view.  Atom ids are molecule-local."""
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
    b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
    # the frozen cache stores molecule-local bond endpoints
    bu = np.asarray(raw["bond_u"][b0:b1], np.int64)
    bv = np.asarray(raw["bond_v"][b0:b1], np.int64)
    if bu.size and (bu.min() < 0 or bu.max() >= a1 - a0):
        raise ValueError("bond endpoint outside molecule atom range")
    return {
        "row": int(row),
        "n_atoms": int(a1 - a0),
        "n_bonds": int(b1 - b0),
        "atom_offset": a0,
        "bond_offset": b0,
        "atom_feat": np.asarray(raw["atom_feat"][a0:a1]),
        "bond_feat": np.asarray(raw["bond_feat"][b0:b1]),
        "bond_u": bu,
        "bond_v": bv,
        "degree": np.asarray(raw["degree"][a0:a1]),
        "min_ring": np.asarray(raw["min_ring"][a0:a1]),
        "in_cycle": np.asarray(raw["in_cycle"][a0:a1]),
        "smiles": str(np.asarray(raw["smiles"])[row]),
    }
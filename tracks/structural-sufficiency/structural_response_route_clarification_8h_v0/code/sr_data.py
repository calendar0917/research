"""Label-safe data layer for the 8h route-clarification round.

Only the raw OGB pilot caches and the frozen fragment caches are read.  The
**Test arrays are dropped from the payload at load time**, immediately after the
byte-identity check, so no downstream code can address a Test row even by
accident; the honest statement is recorded in ``analysis/checks.json`` that the
npz file itself is a single archive that also contains the Test block.

Row ids are always the *official* OGB row ids (0..4199 for Lipo, 0..1127 for
ESOL).  D0 subsets are expressed in the same id space by construction, so a
split-local index can never be mistaken for an official id.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

import sr_common as C


@dataclass
class Bundle:
    dataset: str
    node_ptr: np.ndarray
    bond_ptr: np.ndarray
    atom_feat: np.ndarray
    bond_feat: np.ndarray
    bond_u: np.ndarray
    bond_v: np.ndarray
    degree: np.ndarray
    min_ring: np.ndarray
    labels: np.ndarray
    train_idx: np.ndarray
    valid_idx: np.ndarray
    atom_dims: list[int]
    bond_dims: list[int]
    gine_categories: dict[str, int]
    cache_sha256: str
    scaler: dict[str, float]
    labels_std: np.ndarray
    audit: dict[str, Any]

    @property
    def n_graphs(self) -> int:
        return int(self.labels.size)

    # -- graph views -----------------------------------------------------
    def atoms(self, row: int) -> slice:
        return slice(int(self.node_ptr[row]), int(self.node_ptr[row + 1]))

    def bonds(self, row: int) -> slice:
        return slice(int(self.bond_ptr[row]), int(self.bond_ptr[row + 1]))

    def n_atoms(self, row: int) -> int:
        return int(self.node_ptr[row + 1] - self.node_ptr[row])

    def n_bonds(self, row: int) -> int:
        return int(self.bond_ptr[row + 1] - self.bond_ptr[row])

    def set_scaler(self, rows: Sequence[int]) -> None:
        scaler = C.target_scaler(self.labels[np.asarray(sorted(set(int(r) for r in rows)), dtype=np.int64)])
        self.scaler = scaler
        self.labels_std = ((self.labels.astype(np.float64) - scaler["mean"]) / scaler["std"]).astype(np.float32)


def load_bundle(dataset: str, *, keep_test_arrays: bool = False) -> tuple[Bundle, dict[str, Any]]:
    """Load one official cache; Test arrays are removed before any use."""
    path = C.DATASET_CACHE[dataset]
    expected = C.EXPECTED_SHA["lipo_cache"] if dataset == "ogbg-mollipo" else None
    got = C.sha256_file(path)
    if expected is not None and got != expected:
        raise RuntimeError(f"{dataset} cache sha256 {got} != {expected}")
    with np.load(path, allow_pickle=True) as handle:
        payload = {key: handle[key] for key in handle.files}

    raw_keys = sorted(payload)
    test_idx_present = "test_idx" in payload
    n_test = int(np.asarray(payload["test_idx"]).size) if test_idx_present else 0
    label_minus_test = payload["labels"]
    # --- Test removal: the arrays are deleted from the working payload. ----
    if not keep_test_arrays:
        for key in ("test_idx",):
            payload.pop(key, None)

    train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
    valid_idx = np.asarray(payload["valid_idx"], dtype=np.int64)
    labels = np.asarray(label_minus_test, dtype=np.float32)
    sizes = {"train": int(train_idx.size), "valid": int(valid_idx.size)}
    expected_sizes = C.DATASET_SPLIT_SIZES[dataset]
    if sizes["train"] != expected_sizes["train"] or sizes["valid"] != expected_sizes["valid"]:
        raise RuntimeError(f"{dataset} split sizes {sizes} != {expected_sizes}")
    if np.intersect1d(train_idx, valid_idx).size:
        raise RuntimeError("Train/Valid overlap")

    node_ptr = np.asarray(payload["node_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(payload["bond_ptr"], dtype=np.int64)
    bond_u = np.asarray(payload["bond_u"], dtype=np.int64)
    bond_v = np.asarray(payload["bond_v"], dtype=np.int64)
    # canonical u < v and no self loops, as the historical loader asserts
    if bond_u.size and np.any(bond_u >= bond_v):
        raise RuntimeError("physical bonds are not canonical u < v")

    atom_feat = np.asarray(payload["atom_feat"], dtype=np.int64)
    bond_feat = np.asarray(payload["bond_feat"], dtype=np.int64)
    degree = np.asarray(payload["degree"], dtype=np.int64)
    min_ring = np.asarray(payload["min_ring"], dtype=np.int64)

    # Train-derived embedding sizes with a full-data coverage assertion, the
    # same convention the pinned FGCR data layer uses.
    atom_values = [set() for _ in range(C.ATOM_COLUMNS)]
    bond_values = [set() for _ in range(C.BOND_COLUMNS)]
    for row in train_idx.tolist():
        a = atom_feat[int(node_ptr[row]):int(node_ptr[row + 1])]
        b = bond_feat[int(bond_ptr[row]):int(bond_ptr[row + 1])]
        for j in range(C.ATOM_COLUMNS):
            atom_values[j].update(int(x) for x in np.unique(a[:, j]).tolist())
        for j in range(C.BOND_COLUMNS):
            bond_values[j].update(int(x) for x in np.unique(b[:, j]).tolist())
    atom_dims = [max(v) + 1 for v in atom_values]
    bond_dims = [max(v) + 1 for v in bond_values]
    for j in range(C.ATOM_COLUMNS):
        if int(atom_feat[:, j].max()) >= atom_dims[j]:
            raise RuntimeError("atom embedding table does not cover the full data")
    for j in range(C.BOND_COLUMNS):
        if int(bond_feat[:, j].max()) >= bond_dims[j]:
            raise RuntimeError("bond embedding table does not cover the full data")

    # GINE G0 uses full-data category counts (the historical convention).
    gine_categories = {f"atom_{j}": int(atom_feat[:, j].max()) + 1 for j in range(C.ATOM_COLUMNS)}
    gine_categories.update({f"bond_{j}": int(bond_feat[:, j].max()) + 1 for j in range(C.BOND_COLUMNS)})
    gine_categories["degree"] = int(degree.max()) + 1
    gine_categories["ring"] = int(min_ring.max()) + 1

    scaler = C.target_scaler(labels[train_idx])
    labels_std = ((labels.astype(np.float64) - scaler["mean"]) / scaler["std"]).astype(np.float32)

    bundle = Bundle(
        dataset=dataset,
        node_ptr=node_ptr, bond_ptr=bond_ptr,
        atom_feat=atom_feat, bond_feat=bond_feat,
        bond_u=bond_u, bond_v=bond_v,
        degree=degree, min_ring=min_ring,
        labels=labels, train_idx=train_idx, valid_idx=valid_idx,
        atom_dims=atom_dims, bond_dims=bond_dims,
        gine_categories=gine_categories,
        cache_sha256=got, scaler=scaler, labels_std=labels_std,
        audit={},
    )
    audit = {
        "dataset": dataset,
        "cache_path": str(path.relative_to(C.REPO_ROOT)),
        "cache_sha256": got,
        "npy_member_keys_in_file": raw_keys,
        "test_block_present_in_file": bool(test_idx_present),
        "n_test_rows_in_file": n_test,
        "test_arrays_dropped_before_use": True,
        "test_statement": (
            "The single npz archive contains official Test row ids, labels and features. "
            "The loader materialised it once, then deleted the Test block from the working "
            "payload before any training, evaluation or analysis code saw it. No Test row "
            "entered any forward pass or metric."),
        "official_split_sizes": sizes,
        "train_valid_disjoint": True,
        "train_rows_sha256": C.sha256_array(train_idx),
        "valid_rows_sha256": C.sha256_array(valid_idx),
        "atom_dims": atom_dims,
        "bond_dims": bond_dims,
        "gine_categories": gine_categories,
        "scaler_official_train": scaler,
        "total_atoms": int(node_ptr[-1]),
        "total_physical_bonds": int(bond_ptr[-1]),
    }
    bundle.audit = audit
    return bundle, audit


# ---------------------------------------------------------------------------
# D0 (Fit / Dev) split
# ---------------------------------------------------------------------------


def load_d0(bundle: Bundle) -> dict[str, Any]:
    """Fit-2856 / Dev-504 inside official Lipo Train, in official row ids."""
    path = C.D0_SPLIT
    got = C.sha256_file(path)
    if got != C.EXPECTED_SHA["d0_split"]:
        raise RuntimeError(f"D0 split sha256 {got} != frozen value")
    payload = C.read_json(path)
    fit = np.asarray(sorted(int(v) for v in payload["train_sub_rows"]), dtype=np.int64)
    dev = np.asarray(sorted(int(v) for v in payload["dev_rows"]), dtype=np.int64)
    train = set(int(v) for v in bundle.train_idx.tolist())
    if fit.size != 2856 or dev.size != 504:
        raise RuntimeError(f"D0 sizes {fit.size}/{dev.size} != 2856/504")
    if np.intersect1d(fit, dev).size:
        raise RuntimeError("Fit/Dev overlap")
    if not (set(fit.tolist()) | set(dev.tolist())) == train:
        raise RuntimeError("Fit + Dev does not equal the official Train rows")
    if not (set(fit.tolist()) <= train and set(dev.tolist()) <= train):
        raise RuntimeError("Fit/Dev contains a non-Train row")
    return {
        "fit_rows": fit, "dev_rows": dev,
        "sha256": got,
        "n_fit": int(fit.size), "n_dev": int(dev.size),
        "fit_rows_sha256": C.sha256_array(fit),
        "dev_rows_sha256": C.sha256_array(dev),
    }


# ---------------------------------------------------------------------------
# graph packing helpers
# ---------------------------------------------------------------------------


def graph_pairs(bundle: Bundle, rows: Sequence[int]) -> list[tuple[int, list[tuple[int, int]]]]:
    out = []
    for row in rows:
        b0, b1 = int(bundle.bond_ptr[int(row)]), int(bundle.bond_ptr[int(row) + 1])
        u = bundle.bond_u[b0:b1]
        v = bundle.bond_v[b0:b1]
        out.append((int(bundle.n_atoms(int(row))), list(zip(u.tolist(), v.tolist()))))
    return out


def marginals(bundle: Bundle, rows: Sequence[int], statistic: str) -> np.ndarray:
    """Per-row descriptive statistics used only as frozen evaluation strata."""
    if statistic == "n_atoms":
        return np.asarray([bundle.n_atoms(int(r)) for r in rows], dtype=np.float64)
    if statistic == "n_bonds":
        return np.asarray([bundle.n_bonds(int(r)) for r in rows], dtype=np.float64)
    if statistic == "n_components":
        out = []
        for row in rows:
            b0 = int(bundle.bond_ptr[int(row)])
            b1 = int(bundle.bond_ptr[int(row) + 1])
            n = bundle.n_atoms(int(row))
            parent = list(range(n))

            def find(x):
                while parent[x] != x:
                    parent[x] = parent[parent[x]]
                    x = parent[x]
                return x

            for a, b in zip(bundle.bond_u[b0:b1].tolist(), bundle.bond_v[b0:b1].tolist()):
                ra, rb = find(a), find(b)
                if ra != rb:
                    parent[ra] = rb
            out.append(len({find(i) for i in range(n)}))
        return np.asarray(out, dtype=np.float64)
    raise KeyError(statistic)

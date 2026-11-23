"""FGCR-v0 label-safe data layer over the pinned SQ mollipo cache.

Reuses ``sq_data.load_cache`` read-only (source SHA pinned in
``fgcr_common.import_sq_data``).  The cache itself is verified byte-for-byte.
Only Train labels are used for standardization; Valid/Test labels are used only
for the explicitly authorized per-epoch metrics.

Graph conventions (inherited from the pinned cache):
* atom features = the raw nine OGB categorical columns;
* one record per physical undirected bond (``u < v``), no self loops;
* isolated atoms and multi-component molecules are kept;
* targets are raw lipophilicity values.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import fgcr_common as C


@dataclass
class DataBundle:
    payload: Mapping[str, Any]
    train_idx: np.ndarray
    valid_idx: np.ndarray
    test_idx: np.ndarray
    labels: np.ndarray                 # raw, float32, all 4200 rows
    labels_std: np.ndarray             # Train-standardized, float32
    scaler: dict[str, float]
    node_ptr: np.ndarray
    bond_ptr: np.ndarray
    atom_feat: np.ndarray              # [Ntot, 9] int64
    bond_feat: np.ndarray              # [Mtot, 3] int64
    bond_u: np.ndarray                 # [Mtot] local atom ids
    bond_v: np.ndarray
    graph_n_atoms: np.ndarray
    graph_n_bonds: np.ndarray
    atom_dims: list[int]
    bond_dims: list[int]
    cache_sha256: str
    audit: dict[str, Any]

    @property
    def n_graphs(self) -> int:
        return int(self.labels.size)

    def split_rows(self, split: str) -> np.ndarray:
        return {"train": self.train_idx, "valid": self.valid_idx,
                "test": self.test_idx}[split]


def _category_dims_and_unseen(payload: Mapping[str, Any]) -> tuple[list[int], list[int], dict[str, Any]]:
    """Embedding sizes from Train maxima; assert full-data coverage."""
    node_ptr = np.asarray(payload["node_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(payload["bond_ptr"], dtype=np.int64)
    atom_feat = np.asarray(payload["atom_feat"])
    bond_feat = np.asarray(payload["bond_feat"])
    train = np.asarray(payload["train_idx"], dtype=np.int64)

    atom_values = [set() for _ in range(C.ATOM_COLUMNS)]
    bond_values = [set() for _ in range(C.BOND_COLUMNS)]
    for g in train.tolist():
        a0, a1 = int(node_ptr[g]), int(node_ptr[g + 1])
        b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
        block = atom_feat[a0:a1]
        for j in range(C.ATOM_COLUMNS):
            atom_values[j].update(int(x) for x in np.unique(block[:, j]).tolist())
        block = bond_feat[b0:b1]
        for j in range(C.BOND_COLUMNS):
            bond_values[j].update(int(x) for x in np.unique(block[:, j]).tolist())

    atom_dims = [max(v) + 1 for v in atom_values]
    bond_dims = [max(v) + 1 for v in bond_values]

    # The embedding table must cover every value present anywhere (unseen
    # values stay at their initialization; they are never trained).
    full_atom_max = [int(atom_feat[:, j].max()) for j in range(C.ATOM_COLUMNS)]
    full_bond_max = [int(bond_feat[:, j].max()) for j in range(C.BOND_COLUMNS)]
    for j, value in enumerate(full_atom_max):
        if value >= atom_dims[j]:
            raise RuntimeError(
                f"atom column {j}: full-data max {value} exceeds Train-derived "
                f"embedding size {atom_dims[j]}")
    for j, value in enumerate(full_bond_max):
        if value >= bond_dims[j]:
            raise RuntimeError(
                f"bond column {j}: full-data max {value} exceeds Train-derived "
                f"embedding size {bond_dims[j]}")

    def _split_values(indices, ptr, feat, columns):
        values = [set() for _ in range(columns)]
        for g in np.asarray(indices, dtype=np.int64).tolist():
            a0, a1 = int(ptr[g]), int(ptr[g + 1])
            block = feat[a0:a1]
            for j in range(columns):
                values[j].update(int(x) for x in np.unique(block[:, j]).tolist())
        return values

    unseen_counts: dict[str, Any] = {}
    for name in ("valid", "test"):
        indices = payload[f"{name}_idx"]
        seen_a = _split_values(indices, node_ptr, atom_feat, C.ATOM_COLUMNS)
        seen_b = _split_values(indices, bond_ptr, bond_feat, C.BOND_COLUMNS)
        unseen_counts[f"atom_{name}"] = [
            len(seen_a[j] - atom_values[j]) for j in range(C.ATOM_COLUMNS)
        ]
        unseen_counts[f"bond_{name}"] = [
            len(seen_b[j] - bond_values[j]) for j in range(C.BOND_COLUMNS)
        ]
    return atom_dims, bond_dims, {
        "atom_values_train": [sorted(v) for v in atom_values],
        "bond_values_train": [sorted(v) for v in bond_values],
        "atom_dims": atom_dims,
        "bond_dims": bond_dims,
        "full_atom_max": full_atom_max,
        "full_bond_max": full_bond_max,
        "unseen_value_counts": unseen_counts,
    }


def _verify_cache(payload: Mapping[str, Any]) -> dict[str, Any]:
    train = np.asarray(payload["train_idx"], dtype=np.int64)
    valid = np.asarray(payload["valid_idx"], dtype=np.int64)
    test = np.asarray(payload["test_idx"], dtype=np.int64)
    sizes = {"train": int(train.size), "valid": int(valid.size), "test": int(test.size)}
    if sizes != C.EXPECTED_SPLIT_SIZES:
        raise ValueError(f"official split sizes {sizes} != {C.EXPECTED_SPLIT_SIZES}")
    if np.intersect1d(train, valid).size or np.intersect1d(train, test).size \
            or np.intersect1d(valid, test).size:
        raise ValueError("split row ids overlap")
    union = np.union1d(np.union1d(train, valid), test)
    if union.size != 4200 or not np.array_equal(union, np.arange(4200)):
        raise ValueError("split row ids do not cover 0..4199")
    labels = np.asarray(payload["labels"], dtype=np.float64)
    if labels.size != 4200 or not np.isfinite(labels).all():
        raise ValueError("labels are not 4200 finite values")

    node_ptr = np.asarray(payload["node_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(payload["bond_ptr"], dtype=np.int64)
    atom_feat = np.asarray(payload["atom_feat"])
    bond_feat = np.asarray(payload["bond_feat"])
    bond_u = np.asarray(payload["bond_u"], dtype=np.int64)
    bond_v = np.asarray(payload["bond_v"], dtype=np.int64)
    if node_ptr.size != 4201 or bond_ptr.size != 4201:
        raise ValueError("pointer arrays are not 4201 long")
    if atom_feat.shape != (int(node_ptr[-1]), C.ATOM_COLUMNS):
        raise ValueError("atom feature shape mismatch")
    if bond_feat.shape != (int(bond_ptr[-1]), C.BOND_COLUMNS):
        raise ValueError("bond feature shape mismatch")
    if bond_u.shape != (int(bond_ptr[-1]),) or bond_v.shape != (int(bond_ptr[-1]),):
        raise ValueError("bond index shape mismatch")

    n_multi_component = 0
    n_isolated = 0
    min_atoms = 10**9
    max_atoms = 0
    max_bonds = 0
    for g in range(4200):
        a0, a1 = int(node_ptr[g]), int(node_ptr[g + 1])
        b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
        n = a1 - a0
        m = b1 - b0
        min_atoms = min(min_atoms, n)
        max_atoms = max(max_atoms, n)
        max_bonds = max(max_bonds, m)
        if n <= 0:
            raise ValueError(f"graph {g} has no atoms")
        if int(np.asarray(payload["graph_n_atoms"])[g]) != n:
            raise ValueError(f"graph {g} atom count mismatch")
        if int(np.asarray(payload["graph_n_bonds"])[g]) != m:
            raise ValueError(f"graph {g} bond count mismatch")
        u = bond_u[b0:b1]
        v = bond_v[b0:b1]
        if u.size and (u.min() < 0 or u.max() >= n or v.min() < 0 or v.max() >= n):
            raise ValueError(f"graph {g} bond endpoint outside atom range")
        if np.any(u == v):
            raise ValueError(f"graph {g} has a self loop")
        pairs = set(zip(u.tolist(), v.tolist()))
        if len(pairs) != m:
            raise ValueError(f"graph {g} has duplicate physical bonds")
        if any(a > b for a, b in pairs):
            raise ValueError(f"graph {g} physical bonds are not canonical u < v")
        n_isolated += int(np.sum(np.bincount(np.concatenate([u, v]), minlength=n) == 0))
        n_multi_component += int(int(np.asarray(payload["graph_n_components"])[g]) > 1)

    return {
        "sizes": sizes,
        "disjoint": True,
        "union_covers_4200": True,
        "labels_all_finite": True,
        "min_atoms": int(min_atoms),
        "max_atoms": int(max_atoms),
        "max_bonds": int(max_bonds),
        "n_isolated_atoms": int(n_isolated),
        "n_multi_component_molecules": int(n_multi_component),
        "total_atoms": int(node_ptr[-1]),
        "total_physical_bonds": int(bond_ptr[-1]),
        "train_rows_sha256": C.sha256_array(train),
        "valid_rows_sha256": C.sha256_array(valid),
        "test_rows_sha256": C.sha256_array(test),
    }


def load_bundle() -> DataBundle:
    sq_data, sq_hashes = C.import_sq_data()
    payload = sq_data.load_cache(C.DATASET, verify_sha=C.CACHE_SHA256)
    audit = _verify_cache(payload)
    audit["sq_source_sha256"] = sq_hashes
    audit["cache_sha256"] = str(payload["cache_sha256"])

    labels = np.asarray(payload["labels"], dtype=np.float32)
    train_idx = np.asarray(payload["train_idx"], dtype=np.int64)
    valid_idx = np.asarray(payload["valid_idx"], dtype=np.int64)
    test_idx = np.asarray(payload["test_idx"], dtype=np.int64)
    scaler = C.train_target_scaler(labels[train_idx])
    labels_std = ((labels.astype(np.float64) - scaler["mean"]) / scaler["std"]).astype(np.float32)
    audit["scaler"] = scaler
    audit["standardization_source"] = "official Train rows only; ddof=0"

    atom_dims, bond_dims, category_audit = _category_dims_and_unseen(payload)
    audit["categories"] = category_audit

    return DataBundle(
        payload=payload,
        train_idx=train_idx,
        valid_idx=valid_idx,
        test_idx=test_idx,
        labels=labels,
        labels_std=labels_std,
        scaler=scaler,
        node_ptr=np.asarray(payload["node_ptr"], dtype=np.int64),
        bond_ptr=np.asarray(payload["bond_ptr"], dtype=np.int64),
        atom_feat=np.asarray(payload["atom_feat"], dtype=np.int64),
        bond_feat=np.asarray(payload["bond_feat"], dtype=np.int64),
        bond_u=np.asarray(payload["bond_u"], dtype=np.int64),
        bond_v=np.asarray(payload["bond_v"], dtype=np.int64),
        graph_n_atoms=np.asarray(payload["graph_n_atoms"], dtype=np.int64),
        graph_n_bonds=np.asarray(payload["graph_n_bonds"], dtype=np.int64),
        atom_dims=atom_dims,
        bond_dims=bond_dims,
        cache_sha256=str(payload["cache_sha256"]),
        audit=audit,
    )


def graphs_for_rows(bundle: DataBundle, rows: Sequence[int]) -> list[dict[str, Any]]:
    """Per-graph numpy arrays in the order of ``rows``."""
    graphs: list[dict[str, Any]] = []
    for row in rows:
        g = int(row)
        a0, a1 = int(bundle.node_ptr[g]), int(bundle.node_ptr[g + 1])
        b0, b1 = int(bundle.bond_ptr[g]), int(bundle.bond_ptr[g + 1])
        graphs.append({
            "row": g,
            "n_atoms": a1 - a0,
            "n_bonds": b1 - b0,
            "atom_feat": bundle.atom_feat[a0:a1],
            "bond_u": bundle.bond_u[b0:b1],
            "bond_v": bundle.bond_v[b0:b1],
            "bond_feat": bundle.bond_feat[b0:b1],
        })
    return graphs


def collate_graphs(graphs: Sequence[Mapping[str, Any]], plan, device: Any,
                   *, plan_rows: Sequence[int] | None = None) -> dict[str, Any]:
    """Pack a list of per-graph arrays with the matching GlobalPlan.

    The atom order is (graph order, original atom id).  The bond order is
    (graph order, canonical physical bond order).  ``plan_rows`` gives each
    graph's row id inside ``plan``; it defaults to ``0..len(graphs)-1`` for a
    plan built directly on this sequence.  A corpus plan must pass the global
    official row ids explicitly.
    """
    if not graphs:
        raise ValueError("empty batch")
    if plan_rows is None:
        plan_rows = list(range(len(graphs)))
    plan_rows = [int(r) for r in plan_rows]
    if len(plan_rows) != len(graphs):
        raise ValueError("plan_rows length does not match the graph list")
    atom_counts = [int(g["n_atoms"]) for g in graphs]
    bond_counts = [int(g["n_bonds"]) for g in graphs]
    bp = plan.select(plan_rows, atom_counts, bond_counts).to(device)

    atom_feat = np.concatenate([np.asarray(g["atom_feat"], dtype=np.int64) for g in graphs], axis=0)
    bond_feat = np.concatenate([np.asarray(g["bond_feat"], dtype=np.int64) for g in graphs], axis=0)
    atom_mol = np.repeat(np.arange(len(graphs), dtype=np.int64), atom_counts)
    bond_mol = np.repeat(np.arange(len(graphs), dtype=np.int64), bond_counts)

    n_atoms = sum(atom_counts)
    n_bonds = sum(bond_counts)
    if atom_feat.shape != (n_atoms, C.ATOM_COLUMNS):
        raise ValueError("packed atom feature shape mismatch")
    if bond_feat.shape != (n_bonds, C.BOND_COLUMNS):
        raise ValueError("packed bond feature shape mismatch")
    if bp.n_atoms != n_atoms or bp.n_bonds != n_bonds:
        raise ValueError("batch plan atom/bond count mismatch")

    return {
        "atom_feat": torch.as_tensor(atom_feat, dtype=torch.long, device=device),
        "bond_feat": torch.as_tensor(bond_feat, dtype=torch.long, device=device),
        "atom_mol": torch.as_tensor(atom_mol, dtype=torch.long, device=device),
        "bond_mol": torch.as_tensor(bond_mol, dtype=torch.long, device=device),
        "n_atoms": n_atoms,
        "n_bonds": n_bonds,
        "n_molecules": len(graphs),
        "graph_n_atoms": torch.as_tensor(atom_counts, dtype=torch.float32, device=device),
        "graph_n_bonds": torch.as_tensor(bond_counts, dtype=torch.float32, device=device),
        "plan": bp,
    }


def collate_rows(bundle: DataBundle, rows: Sequence[int], plan, device: Any) -> dict[str, Any]:
    rows_list = [int(r) for r in rows]
    batch = collate_graphs(graphs_for_rows(bundle, rows_list), plan, device,
                           plan_rows=rows_list)
    rows_arr = np.asarray(rows, dtype=np.int64)
    batch["row_ids"] = torch.as_tensor(rows_arr, dtype=torch.long, device=device)
    batch["y_std"] = torch.as_tensor(bundle.labels_std[rows_arr], dtype=torch.float32, device=device)
    batch["y_raw"] = torch.as_tensor(bundle.labels[rows_arr], dtype=torch.float32, device=device)
    return batch
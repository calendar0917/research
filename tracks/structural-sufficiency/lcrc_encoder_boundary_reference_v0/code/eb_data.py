"""Data layer: read-only CSC bundle (Train/Valid only) + the real GINE raw inputs.

The graph cache and split identity are inherited byte-for-byte from
``structural-query-learning`` / CSC v0.  This module adds only:

1. re-verification of the frozen sq-cache bytes and the Train/Valid row alignment,
2. per-split arrays of the *raw* OGB atom/bond columns the GINE encoder consumes,
3. a batch constructor over arbitrary split-local rows.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

import eb_common as E
import cs_common as C

C.setup_paths()

import cs_data as D  # noqa: E402


@dataclass
class GineSplit:
    """Raw GINE inputs for one split, in split-local molecule order."""

    atom: np.ndarray        # (N, 9) int64
    degree: np.ndarray      # (N,) int64
    ring: np.ndarray        # (N,) int64
    bond_u: np.ndarray      # (B,) int64  split-local atom index
    bond_v: np.ndarray
    bond: np.ndarray        # (B, 3) int64
    atom_ptr: np.ndarray    # (M + 1,) int64
    bond_ptr: np.ndarray    # (M + 1,) int64
    global_feat: np.ndarray # (M, G) float32, raw (post_local normalises)
    row_ids: np.ndarray     # (M,) official row id, ascending
    smiles: list[str]
    source_cache_sha256: str


def load_bundle() -> Any:
    bundle = D.load_bundle(E.DATASET)
    sizes = {s: int(bundle.caches[s]["n_molecules"]) for s in E.SPLITS_USED}
    if sizes != {"train": 3360, "valid": 420}:
        raise RuntimeError(f"unexpected Train/Valid sizes {sizes}")
    return bundle


def load_sq_cache() -> Mapping[str, np.ndarray]:
    actual = E.sha256_file(E.SQ_CACHE)
    if actual != E.SQ_CACHE_SHA256:
        raise RuntimeError(f"{E.SQ_CACHE_NAME}: {actual} != {E.SQ_CACHE_SHA256}")
    with np.load(E.SQ_CACHE, allow_pickle=False) as z:
        return {k: np.asarray(z[k]) for k in z.files}


def category_counts(splits: Sequence[GineSplit]) -> dict[str, int]:
    atom = np.concatenate([s.atom for s in splits], axis=0)
    bond = np.concatenate([s.bond for s in splits], axis=0)
    degree = np.concatenate([s.degree for s in splits])
    ring = np.concatenate([s.ring for s in splits])
    counts = {f"atom_{j}": int(atom[:, j].max()) + 1 for j in range(atom.shape[1])}
    counts.update({f"bond_{j}": int(bond[:, j].max()) + 1 for j in range(bond.shape[1])})
    counts["degree"] = int(degree.max()) + 1
    counts["ring"] = int(ring.max()) + 1
    return counts


def build_splits(bundle: Any, sq: Mapping[str, np.ndarray]) -> dict[str, GineSplit]:
    """Map official row ids to sq-cache graph indices and verify the alignment."""
    sq_rows = np.asarray(sq["row_ids"], dtype=np.int64)
    inverse = np.full(int(sq_rows.max()) + 1, -1, dtype=np.int64)
    inverse[sq_rows] = np.arange(sq_rows.size, dtype=np.int64)
    if np.any(inverse < 0):
        raise RuntimeError("sq cache row ids are not a dense 0..N-1 cover")

    node_ptr = np.asarray(sq["node_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(sq["bond_ptr"], dtype=np.int64)
    sq_n_atoms = np.diff(node_ptr)
    sq_n_bonds = np.diff(bond_ptr)

    out: dict[str, GineSplit] = {}
    for split in E.SPLITS_USED:
        cache = bundle.caches[split]
        official = np.asarray(bundle.global_row_ids[split], dtype=np.int64)
        graph = inverse[official]
        if np.any(graph < 0):
            raise RuntimeError(f"{split}: official row id missing in sq cache")
        cache_n_atoms = np.diff(np.asarray(cache["atom_ptr"], dtype=np.int64))
        cache_n_bonds = np.diff(np.asarray(cache["bond_ptr"], dtype=np.int64))
        if not np.array_equal(sq_n_atoms[graph], cache_n_atoms):
            raise RuntimeError(f"{split}: atom counts disagree with the CSC cache")
        if not np.array_equal(sq_n_bonds[graph], cache_n_bonds):
            raise RuntimeError(f"{split}: bond counts disagree with the CSC cache")
        cache_smiles = [str(x) for x in cache["smiles"]]
        sq_smiles = [str(sq["smiles"][g]) for g in graph]
        if cache_smiles != sq_smiles:
            raise RuntimeError(f"{split}: SMILES order disagrees with the CSC cache")

        atom_idx = np.concatenate(
            [np.arange(node_ptr[g], node_ptr[g + 1], dtype=np.int64) for g in graph]
        ) if graph.size else np.zeros(0, np.int64)
        bond_idx = np.concatenate(
            [np.arange(bond_ptr[g], bond_ptr[g + 1], dtype=np.int64) for g in graph]
        ) if graph.size else np.zeros(0, np.int64)
        atom_off = np.concatenate([[0], np.cumsum(sq_n_atoms[graph])[:-1]]).astype(np.int64) if graph.size else np.zeros(0, np.int64)
        # sq-cache endpoints are molecule-local; batch assembly adds the atom offset
        bu = np.asarray(sq["bond_u"])[bond_idx].astype(np.int64)
        bv = np.asarray(sq["bond_v"])[bond_idx].astype(np.int64)
        out[split] = GineSplit(
            atom=np.asarray(sq["atom_feat"])[atom_idx].astype(np.int64),
            degree=np.asarray(sq["degree"])[atom_idx].astype(np.int64),
            ring=np.asarray(sq["min_ring"])[atom_idx].astype(np.int64),
            bond_u=bu,
            bond_v=bv,
            bond=np.asarray(sq["bond_feat"])[bond_idx].astype(np.int64),
            atom_ptr=np.concatenate([[0], np.cumsum(sq_n_atoms[graph])]).astype(np.int64),
            bond_ptr=np.concatenate([[0], np.cumsum(sq_n_bonds[graph])]).astype(np.int64),
            global_feat=np.asarray(cache["global"], dtype=np.float32),
            row_ids=official,
            smiles=cache_smiles,
            source_cache_sha256=E.SQ_CACHE_SHA256,
        )
    return out


def make_batch(split: GineSplit, rows: np.ndarray, device=None) -> dict[str, Any]:
    """One batch (both bond directions) of split-local ``rows``."""
    import torch

    rows = np.asarray(rows, dtype=np.int64)
    atom_start = split.atom_ptr[rows]
    atom_counts = split.atom_ptr[rows + 1] - atom_start
    bond_start = split.bond_ptr[rows]
    bond_counts = split.bond_ptr[rows + 1] - bond_start
    atom_off = np.concatenate([[0], np.cumsum(atom_counts)[:-1]]).astype(np.int64) if rows.size else np.zeros(0, np.int64)

    atom_idx = np.concatenate(
        [np.arange(s, s + n, dtype=np.int64) for s, n in zip(atom_start, atom_counts)]
    ) if rows.size else np.zeros(0, np.int64)
    bond_idx = np.concatenate(
        [np.arange(s, s + n, dtype=np.int64) for s, n in zip(bond_start, bond_counts)]
    ) if rows.size else np.zeros(0, np.int64)

    bond_batch = np.repeat(np.arange(rows.size, dtype=np.int64), bond_counts)
    u = split.bond_u[bond_idx] + atom_off[bond_batch] if bond_idx.size else np.zeros(0, np.int64)
    v = split.bond_v[bond_idx] + atom_off[bond_batch] if bond_idx.size else np.zeros(0, np.int64)
    attr = split.bond[bond_idx] if bond_idx.size else np.zeros((0, 3), np.int64)
    batch = np.repeat(np.arange(rows.size, dtype=np.int64), atom_counts)

    def _t(arr, dtype):
        return torch.from_numpy(np.ascontiguousarray(np.asarray(arr, dtype=dtype)))

    out = {
        "n_graphs": int(rows.size),
        "x_cat": _t(split.atom[atom_idx], np.int64),
        "deg": _t(split.degree[atom_idx], np.int64),
        "ring": _t(split.ring[atom_idx], np.int64),
        "edge_index": _t(np.stack([np.concatenate([u, v]), np.concatenate([v, u])], axis=0), np.int64),
        "edge_cat": _t(np.concatenate([attr, attr], axis=0), np.int64),
        "atom_batch": _t(batch, np.int64),
        "global": _t(split.global_feat[rows], np.float32),
    }
    if device is not None:
        out = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in out.items()}
    return out

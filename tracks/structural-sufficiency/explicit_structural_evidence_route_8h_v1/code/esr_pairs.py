"""Relation instances ``P(G) = {(i,j) : 1 <= dist(i,j) <= D}`` and pair collate.

A relation instance is an unordered pair of *root atoms* of the same molecule at
shortest-bond-path distance 1..D (main experiments D=2; the conditional
``BEST_EXPLICIT_RANGE4`` arm uses D=4).  Nothing is sampled, truncated or
selected by atom id: every unordered pair in the distance range appears exactly
once, cross-component pairs never appear (no finite distance), and every pair
carries the fixed relation vector

    r_ij = [dist one-hot(SELF,1,2,3,4), log1p(#common neighbours), 13 raw bond feats]

where the bond features are the *cached, real* bond attributes of the direct
physical bond (zeros when the two roots are not directly bonded).

The cache is label-free and built once per (dataset, split, D); it stores only
local atom indices plus a local bond-row index.  Arrays are loaded with
``mmap_mode="r"`` so a 400 MB PCBA cache is never resident in full.
"""

from __future__ import annotations

import hashlib
import json
import os
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import esr_common as C
import lcrc_data as PREP_DATA

PAIR_CACHE_VERSION = "esr-pairs-v1"


# ---------------------------------------------------------------------------
# one molecule
# ---------------------------------------------------------------------------


def molecule_pairs(
    n_atoms: int, bond_index: np.ndarray, max_dist: int = 2
) -> dict[str, np.ndarray]:
    """All unordered root pairs of one molecule with ``1 <= dist <= max_dist``.

    ``bond_index`` holds one row per physical bond (local atom ids).  Distances
    are shortest-bond-path distances in that graph; different connected
    components have no finite distance and are therefore absent.
    """
    n = int(n_atoms)
    empty = {
        "u": np.zeros(0, np.int32),
        "v": np.zeros(0, np.int32),
        "dist": np.zeros(0, np.uint8),
        "common": np.zeros(0, np.uint8),
        "bond": np.full(0, -1, np.int32),
    }
    if n < 2 or len(bond_index) == 0:
        return empty
    idx = np.asarray(bond_index, dtype=np.int64).reshape(-1, 2)
    A = np.zeros((n, n), dtype=np.float32)
    A[idx[:, 0], idx[:, 1]] = 1.0
    A[idx[:, 1], idx[:, 0]] = 1.0
    # common-neighbour counts (A has no self loops, so A@A counts 2-paths)
    A2 = A @ A
    reach = np.zeros((n, n), dtype=bool)
    dist = np.zeros((n, n), dtype=np.int64)
    frontier = A > 0.0
    for step in range(1, int(max_dist) + 1):
        new = frontier & ~reach
        reach |= new
        dist[new] = step
        if step < int(max_dist):
            frontier = (frontier.astype(np.float32) @ A) > 0.0
    iu, ju = np.triu_indices(n, 1)
    keep = reach[iu, ju]
    if not np.any(keep):
        return empty
    u = iu[keep].astype(np.int32)
    v = ju[keep].astype(np.int32)
    d = dist[u, v].astype(np.uint8)
    common = np.rint(A2[u, v]).astype(np.uint8)
    bond = np.full(u.shape, -1, dtype=np.int32)
    direct = d == 1
    if np.any(direct):
        lookup: dict[tuple[int, int], int] = {}
        for row, (a, b) in enumerate(idx.tolist()):
            lookup[(min(a, b), max(a, b))] = row
        uu, vv = u[direct], v[direct]
        bond[direct] = np.asarray(
            [lookup[(min(a, b), max(a, b))] for a, b in zip(uu.tolist(), vv.tolist())], dtype=np.int32
        )
    return {"u": u, "v": v, "dist": d, "common": common, "bond": bond}


def _molecule_task(args: tuple[int, int, int, int, np.ndarray, int]) -> tuple[int, dict[str, np.ndarray]]:
    mol, n_atoms, b0, b1, bond_index, max_dist = args
    return mol, molecule_pairs(n_atoms, bond_index[b0:b1], max_dist=max_dist)


# ---------------------------------------------------------------------------
# cache build / load
# ---------------------------------------------------------------------------


def pair_cache_dir(dataset: str, split: str, max_dist: int = 2) -> Path:
    return C.CACHE / "pairs" / f"{dataset}_{split}_d{int(max_dist)}"


def _meta_path(dataset: str, split: str, max_dist: int) -> Path:
    return pair_cache_dir(dataset, split, max_dist) / "meta.json"


def pair_counts_from_cache(cache: Mapping[str, Any]) -> tuple[np.ndarray, np.ndarray]:
    atom_ptr = np.asarray(cache["atom_ptr"]).astype(np.int64)
    return np.diff(atom_ptr), atom_ptr


def cache_identity(cache: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "n_molecules": int(cache["n_molecules"]),
        "feature_spec_hash": str(cache["meta"]["feature_spec_hash"]),
        "content_fingerprint": str(cache["meta"].get("content_fingerprint")),
        "n_atoms_total": int(cache["meta"]["n_atoms_total"]),
        "n_bonds_total": int(cache["meta"]["n_bonds_total"]),
    }


def build_pair_cache(
    cache: Mapping[str, Any],
    dataset: str,
    split: str,
    *,
    max_dist: int = 2,
    workers: int = 8,
    save: bool = True,
) -> dict[str, Any]:
    t0 = time.perf_counter()
    atom_ptr = np.asarray(cache["atom_ptr"]).astype(np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"]).astype(np.int64)
    bond_index = np.asarray(cache["bond_index"]).astype(np.int64)
    n_mol = int(cache["n_molecules"])
    n_atoms = np.diff(atom_ptr)
    tasks = [
        (m, int(n_atoms[m]), int(bond_ptr[m]), int(bond_ptr[m + 1]), bond_index, int(max_dist))
        for m in range(n_mol)
    ]
    results: list[dict[str, np.ndarray] | None] = [None] * n_mol
    if workers > 1 and n_mol > 2000:
        import multiprocessing as mp

        with mp.get_context("fork").Pool(int(workers)) as pool:
            for m, res in pool.imap_unordered(_molecule_task, tasks, chunksize=256):
                results[m] = res
    else:
        for task in tasks:
            m, res = _molecule_task(task)
            results[m] = res
    counts = np.zeros(n_mol, dtype=np.int64)
    parts: dict[str, list[np.ndarray]] = {"u": [], "v": [], "dist": [], "common": [], "bond": []}
    for m, res in enumerate(results):
        assert res is not None
        counts[m] = int(res["u"].size)
        for key in parts:
            parts[key].append(res[key])
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    arrays = {
        "ptr": ptr,
        "u": np.concatenate(parts["u"]).astype(np.int32),
        "v": np.concatenate(parts["v"]).astype(np.int32),
        "dist": np.concatenate(parts["dist"]).astype(np.uint8),
        "common": np.concatenate(parts["common"]).astype(np.uint8),
        "bond": np.concatenate(parts["bond"]).astype(np.int32),
    }
    if int(ptr[-1]) != arrays["u"].size:
        raise RuntimeError("pair pointer does not match concatenated pair arrays")
    meta = {
        "schema": PAIR_CACHE_VERSION,
        "dataset": dataset,
        "split": split,
        "max_dist": int(max_dist),
        "cache_identity": cache_identity(cache),
        "n_molecules": n_mol,
        "n_pairs": int(arrays["u"].size),
        "atom_counts": n_atoms.tolist() if n_mol < 20000 else None,
        "n_pairs_per_molecule": {
            "mean": float(counts.mean()) if n_mol else 0.0,
            "p50": float(np.median(counts)) if n_mol else 0.0,
            "p90": float(np.percentile(counts, 90)) if n_mol else 0.0,
            "max": int(counts.max()) if n_mol else 0,
            "n_molecules_without_pairs": int((counts == 0).sum()),
        },
        "relation_dim": int(C.REL_DIM),
        "relation_definition": "SELF/1/2/3/4 one-hot + log1p(common neighbours) + 13 raw bond features",
        "labels_present": False,
        "build_seconds": float(time.perf_counter() - t0),
        "created_utc": C.utcnow(),
        "workers": int(workers),
    }
    if save:
        out = pair_cache_dir(dataset, split, max_dist)
        out.mkdir(parents=True, exist_ok=True)
        for key, arr in arrays.items():
            np.save(out / f"{key}.npy", arr)
        meta["sha256"] = {key: C.sha256_file(out / f"{key}.npy") for key in arrays}
        C.write_json(out / "meta.json", meta)
    return {"arrays": arrays, "meta": meta, "counts": counts}


def load_pair_cache(
    dataset: str, split: str, *, max_dist: int = 2, mmap: bool = True
) -> dict[str, Any]:
    out = pair_cache_dir(dataset, split, max_dist)
    meta = C.read_json(out / "meta.json")
    if meta.get("schema") != PAIR_CACHE_VERSION:
        raise ValueError(f"{out}: unexpected pair cache schema {meta.get('schema')!r}")
    mode = "r" if mmap else None
    arrays = {key: np.load(out / f"{key}.npy", mmap_mode=mode) for key in ("ptr", "u", "v", "dist", "common", "bond")}
    if int(arrays["ptr"][-1]) != int(arrays["u"].size):
        raise RuntimeError(f"{out}: pair pointer does not match pair arrays")
    return {"arrays": arrays, "meta": meta}


def verify_pair_cache(
    cache: Mapping[str, Any],
    pair_payload: Mapping[str, Any],
    *,
    max_dist: int,
    rows: Sequence[int] | None = None,
    seed: int = 20261125,
    global_bond_index: bool = False,
) -> dict[str, Any]:
    """Independent re-derivation of a deterministic molecule subset (BFS via scipy)."""
    import scipy.sparse as sp
    from scipy.sparse.csgraph import shortest_path

    arrays = pair_payload["arrays"]
    ptr = np.asarray(arrays["ptr"]).astype(np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"]).astype(np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"]).astype(np.int64)
    bond_index = np.asarray(cache["bond_index"]).astype(np.int64)
    n_mol = int(cache["n_molecules"])
    if rows is None:
        rng = np.random.default_rng(int(seed))
        pool = [m for m in range(n_mol) if atom_ptr[m + 1] - atom_ptr[m] >= 2]
        rows = rng.choice(np.asarray(pool, dtype=np.int64), size=min(64, len(pool)), replace=False).tolist()
    mismatches: list[dict[str, Any]] = []
    checked_pairs = 0
    for m in [int(x) for x in rows]:
        n = int(atom_ptr[m + 1] - atom_ptr[m])
        b = bond_index[bond_ptr[m] : bond_ptr[m + 1]]
        if global_bond_index:
            b = b - int(atom_ptr[m])
        if b.size:
            adj = sp.coo_matrix((np.ones(b.shape[0]), (b[:, 0], b[:, 1])), shape=(n, n))
            adj = (adj + adj.T).tocsr()
            d = shortest_path(adj, unweighted=True, directed=False)
            common = (adj.astype(np.int64) @ adj.astype(np.int64)).toarray()
        else:
            d = np.full((n, n), np.inf)
            common = np.zeros((n, n), dtype=np.int64)
        iu, ju = np.triu_indices(n, 1)
        du = d[iu, ju]
        keep = np.isfinite(du) & (du >= 1) & (du <= max_dist)
        ref_i, ref_j, ref_d = iu[keep], ju[keep], du[keep].astype(int)
        got = slice(int(ptr[m]), int(ptr[m + 1]))
        g_i = np.asarray(arrays["u"][got], dtype=np.int64)
        g_j = np.asarray(arrays["v"][got], dtype=np.int64)
        g_d = np.asarray(arrays["dist"][got], dtype=np.int64)
        g_c = np.asarray(arrays["common"][got], dtype=np.int64)
        checked_pairs += int(ref_i.size)
        if not (np.array_equal(g_i, ref_i) and np.array_equal(g_j, ref_j) and np.array_equal(g_d, ref_d)):
            mismatches.append({"molecule": m, "reason": "pair set / order / distance differs"})
            continue
        if not np.array_equal(g_c, common[ref_i, ref_j]):
            mismatches.append({"molecule": m, "reason": "common-neighbour count differs"})
            continue
        # bond row must point at the direct physical bond
        bond_rows = np.asarray(arrays["bond"][got], dtype=np.int64)
        for k in range(ref_i.size):
            if ref_d[k] != 1:
                if bond_rows[k] != -1:
                    mismatches.append({"molecule": m, "reason": "non-bond pair carries a bond row"})
                    break
                continue
            row = int(bond_rows[k])
            if row < 0 or row >= b.size:
                mismatches.append({"molecule": m, "reason": "missing bond row"})
                break
            endpoints = {int(b[row, 0]), int(b[row, 1])}
            if endpoints != {int(ref_i[k]), int(ref_j[k])}:
                mismatches.append({"molecule": m, "reason": "bond row points at another bond"})
                break
    return {
        "n_molecules_checked": len(rows),
        "n_pairs_rechecked": checked_pairs,
        "mismatches": mismatches,
        "passed": len(mismatches) == 0,
    }


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------


def relation_features(
    dist: torch.Tensor, common: torch.Tensor, bond_feat: torch.Tensor
) -> torch.Tensor:
    """Fixed 19-d relation vector: 5 distance one-hots + log1p(common) + bond feats."""
    if dist.dim() != 1:
        raise ValueError("distance must be 1-D")
    onehot = torch.zeros((dist.numel(), C.DIST_BUCKETS), dtype=torch.float32, device=dist.device)
    safe = torch.where(dist < C.DIST_BUCKETS, dist, torch.zeros_like(dist))
    onehot.scatter_(1, safe.to(torch.int64).unsqueeze(1), 1.0)
    feat = torch.cat(
        [
            onehot,
            torch.log1p(common.to(torch.float32)).unsqueeze(1),
            bond_feat.to(torch.float32),
        ],
        dim=1,
    )
    if feat.shape[1] != C.REL_DIM:
        raise RuntimeError(f"relation feature width {feat.shape[1]} != {C.REL_DIM}")
    return feat


def collate_with_pairs(
    cache: Mapping[str, Any],
    pair_payload: Mapping[str, Any],
    rows: Sequence[int],
    *,
    relation_mode: str = "real",
) -> dict[str, Any]:
    """The frozen ``lcrc_data.collate`` plus the pair payload and global extras."""
    rows_arr = np.asarray(rows, dtype=np.int64)
    batch = PREP_DATA.collate(cache, rows_arr)
    arrays = pair_payload["arrays"]
    ptr = np.asarray(arrays["ptr"]).astype(np.int64)
    counts = (ptr[rows_arr + 1] - ptr[rows_arr]).astype(np.int64)
    n_graphs = int(len(rows_arr))
    total = int(counts.sum())
    atom_off = np.asarray(batch["atom_off"], dtype=np.int64)
    bond_counts = np.asarray(batch["bond_counts"], dtype=np.int64)
    bond_off = np.concatenate([[0], np.cumsum(bond_counts)[:-1]]).astype(np.int64) if n_graphs else np.zeros(0, np.int64)
    pair_batch = np.repeat(np.arange(n_graphs, dtype=np.int64), counts) if total else np.zeros(0, np.int64)
    if total:
        offsets = np.concatenate([[0], np.cumsum(counts)[:-1]]).astype(np.int64)
        flat = np.arange(total, dtype=np.int64) + np.repeat(ptr[rows_arr] - offsets, counts)
    else:
        flat = np.zeros(0, dtype=np.int64)
    if total:
        # index the (possibly memory-mapped) arrays first, then upcast the slice
        u = np.asarray(arrays["u"][flat], dtype=np.int64) + atom_off[pair_batch]
        v = np.asarray(arrays["v"][flat], dtype=np.int64) + atom_off[pair_batch]
        dist = np.asarray(arrays["dist"][flat], dtype=np.int64)
        common = np.asarray(arrays["common"][flat], dtype=np.int64)
        bond_local = np.asarray(arrays["bond"][flat], dtype=np.int64)
    else:
        u = v = dist = common = bond_local = np.zeros(0, dtype=np.int64)
    has_bond = bond_local >= 0
    bond_flat = np.where(has_bond, bond_off[pair_batch] + np.maximum(bond_local, 0), 0)
    bond_feat = np.asarray(batch["bond_feat"], dtype=np.float32)[bond_flat] * has_bond.reshape(-1, 1)
    pair_dist = torch.from_numpy(dist.astype(np.int64))
    pair_common = torch.from_numpy(common.astype(np.int64))
    r = relation_features(pair_dist, pair_common, torch.from_numpy(np.ascontiguousarray(bond_feat)))
    if relation_mode == "blind" and total:
        # identical per molecule: every pair receives the molecule's mean relation vector
        acc = np.zeros((n_graphs, C.REL_DIM), dtype=np.float64)
        np.add.at(acc, pair_batch, r.numpy().astype(np.float64))
        mean = acc / np.maximum(counts, 1).reshape(-1, 1)
        r = torch.from_numpy(mean[pair_batch].astype(np.float32))
    elif relation_mode != "real":
        raise ValueError(f"unknown relation_mode {relation_mode!r}")
    batch["pair_u"] = torch.from_numpy(u.astype(np.int64))
    batch["pair_v"] = torch.from_numpy(v.astype(np.int64))
    batch["pair_batch"] = torch.from_numpy(pair_batch.astype(np.int64))
    batch["pair_counts"] = torch.from_numpy(counts.astype(np.int64))
    batch["pair_r"] = r
    batch["pair_dist"] = pair_dist
    batch["pair_common"] = pair_common
    return batch


def batch_global_ext(batch: Mapping[str, Any], extra_norm: Mapping[str, Any]) -> torch.Tensor:
    """``[g(G), standardised(log1p n_atoms), standardised(log1p n_pairs)]``."""
    g = batch["global"]
    n_atoms = batch["atom_counts"].to(torch.float32)
    n_pairs = batch["pair_counts"].to(torch.float32)
    stats = torch.stack([torch.log1p(n_atoms), torch.log1p(n_pairs)], dim=1)
    mean = torch.as_tensor(extra_norm["mean"], dtype=torch.float32, device=g.device)
    scale = torch.as_tensor(extra_norm["scale"], dtype=torch.float32, device=g.device)
    return torch.cat([g, (stats - mean) / scale], dim=1)


def extra_norm_from_counts(n_atoms: np.ndarray, n_pairs: np.ndarray) -> dict[str, Any]:
    stats = np.stack([np.log1p(np.asarray(n_atoms, dtype=np.float64)), np.log1p(np.asarray(n_pairs, dtype=np.float64))], axis=1)
    mean = stats.mean(axis=0)
    scale = stats.std(axis=0)
    scale = np.where(scale > 0, scale, 1.0)
    return {
        "mean": mean.tolist(),
        "scale": scale.tolist(),
        "source": "Train split only",
        "columns": ["log1p(n_atoms)", "log1p(n_relation_pairs)"],
        "valid_or_test_used": False,
        "n": int(stats.shape[0]),
    }


__all__ = [name for name in dir() if not name.startswith("_")]

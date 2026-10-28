"""Label-free structural inputs for LCRC-v1.

Everything here is a deterministic function of the raw atom/bond graph (the
frozen P3 object cache is itself a label-free function of that graph).  Labels
are only read for the official train/valid splits; the test split is not built
here.

Per-atom inputs
---------------
atom_feat   the frozen P3 108-d label-free atom descriptor (unchanged).
S1 joint    for each atom, the multiset of (neighbour atom type, bond type)
            codes aggregated by a shared embedding (63 categories).
S2 typed    for each root atom, all ordered non-backtracking 2-paths
            root -> w -> u with u != root.  Each path contributes a joint
            (type(w), bond1, bond2) code (189), a joint (type(u), bond2) code
            (63), a closure flag (u adjacent to root), and a gated copy of the
            endpoint atom content a_u.  The path codes are a low-rank factored
            alternative to a 3969-way exact table lookup.
scalars     degree / ring / closure scalars (train-only standardised).

Per-molecule relation pairs
---------------------------
All unordered atom pairs i<j at shortest-path distance >= 3, classified into
six mutually exclusive categories: distance in {3, 4-5, >=6} x whether the two
atoms share a non-trivial biconnected block (a block with >= 3 atoms, i.e. a
cycle block).  Short-range content is deliberately excluded because S1/S2
already cover it.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import lc_common as L

np.seterr(over="ignore")

INF_DIST = 1 << 20


# ---------------------------------------------------------------------------
# raw graph -> per-molecule label-free arrays
# ---------------------------------------------------------------------------

def _adjacency(n_atoms: int, bond_index: np.ndarray, bond_type: np.ndarray):
    adj: list[list[tuple[int, int, int]]] = [[] for _ in range(n_atoms)]
    for e, ((u, v), bt) in enumerate(zip(np.asarray(bond_index), np.asarray(bond_type))):
        u, v, bt = int(u), int(v), int(bt)
        adj[u].append((v, bt, e))
        adj[v].append((u, bt, e))
    for i in range(n_atoms):
        adj[i].sort()
    return adj


def _all_pairs_dist(n_atoms: int, adj) -> np.ndarray:
    dist = np.full((n_atoms, n_atoms), INF_DIST, dtype=np.int32)
    for s in range(n_atoms):
        dist[s, s] = 0
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt: list[int] = []
            for u in frontier:
                for v, _, _ in adj[u]:
                    if dist[s, v] == INF_DIST:
                        dist[s, v] = d
                        nxt.append(v)
            frontier = nxt
    return dist


def _ring_block_membership(n_atoms: int, bond_index: np.ndarray):
    """Per-atom set of non-trivial (>= 3 atom) biconnected block ids."""
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(n_atoms))
    graph.add_edges_from((int(u), int(v)) for u, v in np.asarray(bond_index))
    membership: list[set[int]] = [set() for _ in range(n_atoms)]
    n_blocks = 0
    for comp in nx.biconnected_components(graph):
        atoms = sorted(int(a) for a in comp)
        if len(atoms) < 3:
            continue
        for a in atoms:
            membership[a].add(n_blocks)
        n_blocks += 1
    return membership, n_blocks


def _dist_bucket(d: int) -> int:
    if d == 3:
        return 0
    if 4 <= d <= 5:
        return 1
    return 2


def build_from_arrays(
    atom_feat: np.ndarray,
    atom_type: np.ndarray,
    bond_index: np.ndarray,
    bond_type: np.ndarray,
    bond_feat: np.ndarray,
    global_feat: np.ndarray,
    *,
    perm: np.ndarray | None = None,
    reverse_bonds: bool = False,
) -> dict[str, np.ndarray]:
    """Build one molecule's LCRC inputs.

    ``perm`` maps old atom indices to new ones (``new = perm[old]``).  It is used
    only by the invariance CPU check; prep passes ``None``.
    """
    atom_feat = np.asarray(atom_feat, dtype=np.float32)
    atom_type = np.asarray(atom_type, dtype=np.int64)
    bond_index = np.asarray(bond_index, dtype=np.int64)
    bond_type = np.asarray(bond_type, dtype=np.int64)
    bond_feat = np.asarray(bond_feat, dtype=np.float32)
    if len(bond_index) and (bond_type.min() < 1 or bond_type.max() > L.N_BOND_TYPES):
        raise ValueError(f"bond type outside 1..{L.N_BOND_TYPES}: {np.unique(bond_type)}")
    if atom_type.min() < 0 or atom_type.max() >= L.N_ATOM_TYPES:
        raise ValueError("atom type outside frozen vocabulary")

    if perm is not None:
        perm = np.asarray(perm, dtype=np.int64)
        if sorted(perm.tolist()) != list(range(len(perm))):
            raise ValueError("perm must be a permutation")
        new_feat = np.zeros_like(atom_feat)
        new_type = np.zeros_like(atom_type)
        new_feat[perm] = atom_feat
        new_type[perm] = atom_type
        atom_feat, atom_type = new_feat, new_type
        bond_index = perm[bond_index]

    if reverse_bonds:
        bond_index = bond_index[:, ::-1].copy()
        order = np.lexsort((bond_index[:, 1], bond_index[:, 0]))
        bond_index = bond_index[order]
        bond_type = bond_type[order]
        bond_feat = bond_feat[order]

    n = int(atom_feat.shape[0])
    m = int(bond_index.shape[0])
    adj = _adjacency(n, bond_index, bond_type)
    degree = np.asarray([len(adj[i]) for i in range(n)], dtype=np.int64)
    dist = _all_pairs_dist(n, adj)
    membership, n_blocks = _ring_block_membership(n, bond_index)

    # ---- S1: (neighbour type, bond type) multiset per atom ----------------
    nbr_codes: list[np.ndarray] = []
    nbr_counts = np.zeros(n, dtype=np.int64)
    for v in range(n):
        codes = np.asarray([int(atom_type[u]) * L.N_BOND_TYPES + (int(bt) - 1)
                            for u, bt, _e in adj[v]], dtype=np.uint8)
        nbr_codes.append(codes)
        nbr_counts[v] = len(codes)
    nbr_ptr = np.concatenate([[0], np.cumsum(nbr_counts)]).astype(np.int32)
    nbr_code = (np.concatenate(nbr_codes) if nbr_codes else np.zeros(0, dtype=np.uint8)).astype(np.uint8)

    # ---- S2: ordered non-backtracking 2-paths rooted at v -----------------
    path_codes: list[np.ndarray] = []
    path_end63: list[np.ndarray] = []
    path_end: list[np.ndarray] = []
    path_closure: list[np.ndarray] = []
    path_counts = np.zeros(n, dtype=np.int64)
    n_d2 = np.zeros(n, dtype=np.int64)
    for v in range(n):
        cats, ends63, ends, closures = [], [], [], []
        for w, bt1, _e in adj[v]:
            tw = int(atom_type[w])
            for u, bt2, _e2 in adj[w]:
                if u == v:
                    continue
                cats.append(tw * 9 + (int(bt1) - 1) * 3 + (int(bt2) - 1))
                ends63.append(int(atom_type[u]) * L.N_BOND_TYPES + (int(bt2) - 1))
                ends.append(u)
                closures.append(1 if dist[v, u] == 1 else 0)
        path_codes.append(np.asarray(cats, dtype=np.uint8))
        path_end63.append(np.asarray(ends63, dtype=np.uint8))
        path_end.append(np.asarray(ends, dtype=np.int16))
        path_closure.append(np.asarray(closures, dtype=np.uint8))
        path_counts[v] = len(cats)
        n_d2[v] = int(np.sum(dist[v] == 2))
    path_ptr = np.concatenate([[0], np.cumsum(path_counts)]).astype(np.int32)
    path_cat = (np.concatenate(path_codes) if path_codes else np.zeros(0, dtype=np.uint8)).astype(np.uint8)
    path_end63_all = (np.concatenate(path_end63) if path_end63 else np.zeros(0, dtype=np.uint8)).astype(np.uint8)
    path_end_all = (np.concatenate(path_end) if path_end else np.zeros(0, dtype=np.int16)).astype(np.int16)
    path_closure_all = (np.concatenate(path_closure) if path_closure else np.zeros(0, dtype=np.uint8)).astype(np.uint8)

    # ---- relation pairs at graph distance >= 3 ----------------------------
    p_i, p_j, p_cat = [], [], []
    for i in range(n):
        for j in range(i + 1, n):
            d = int(dist[i, j])
            if d < 3:
                continue
            ring = 1 if (membership[i] & membership[j]) else 0
            p_i.append(i)
            p_j.append(j)
            p_cat.append(_dist_bucket(d) * 2 + ring)
    pair_i = np.asarray(p_i, dtype=np.int16)
    pair_j = np.asarray(p_j, dtype=np.int16)
    pair_cat = np.asarray(p_cat, dtype=np.uint8)

    # ---- scalars (P3 atom108 already has the raw ring/degree one-hots) -----
    from structural_relation_objects_v1 import atom_blocks

    blocks = atom_blocks()
    ring_atom = atom_feat[:, blocks["in_ring"]].reshape(-1)
    ring_size_onehot = atom_feat[:, blocks["min_ring_size"]]
    ring_size = np.sum(ring_size_onehot * np.asarray([3, 4, 5, 6, 7, 8], dtype=np.float32),
                       axis=1)
    ring_bonds = atom_feat[:, blocks["ring_bonds"].start]
    terminal = atom_feat[:, blocks["roles"].start]
    fused = (ring_bonds >= 3).astype(np.float32)
    closure_ratio = np.divide(
        np.asarray([float(path_closure_all[path_ptr[v]:path_ptr[v + 1]].sum())
                    for v in range(n)], dtype=np.float32),
        np.maximum(path_counts.astype(np.float32), 1.0),
    )
    scalars = np.stack(
        [
            np.log1p(n_d2.astype(np.float32)),
            np.log1p(path_counts.astype(np.float32)),
            closure_ratio.astype(np.float32),
            degree.astype(np.float32) / 4.0,
            ring_atom.astype(np.float32),
            ring_size.astype(np.float32) / 8.0,
            ring_bonds.astype(np.float32) / 4.0,
            terminal.astype(np.float32),
            fused.astype(np.float32),
        ],
        axis=1,
    ).astype(np.float32)
    assert scalars.shape[1] == L.N_SCALARS

    return {
        "n_atoms": np.int64(n),
        "n_bonds": np.int64(m),
        "atom_feat": atom_feat,
        "atom_type": atom_type.astype(np.int64),
        "atom_scalar": scalars,
        "bond_feat": bond_feat.astype(np.float32),
        "bond_type": bond_type.astype(np.int64),
        "bond_index": bond_index.astype(np.int16),
        "nbr_ptr": nbr_ptr,
        "nbr_code": nbr_code,
        "path_ptr": path_ptr,
        "path_cat": path_cat,
        "path_end63": path_end63_all,
        "path_end": path_end_all,
        "path_closure": path_closure_all,
        "pair_i": pair_i,
        "pair_j": pair_j,
        "pair_cat": pair_cat,
        "global": np.asarray(global_feat, dtype=np.float32).reshape(-1),
        "n_ring_blocks": np.int64(n_blocks),
    }


def build_molecule_from_object(obj, *, perm=None, reverse_bonds=False) -> dict[str, np.ndarray]:
    atom_feat = np.asarray(obj.atom_feat, dtype=np.float32)
    atom_type = np.argmax(atom_feat[:, :L.N_ATOM_TYPES], axis=1).astype(np.int64)
    return build_from_arrays(
        atom_feat,
        atom_type,
        np.asarray(obj.bond_index, dtype=np.int64),
        np.asarray(obj.bond_type, dtype=np.int64),
        np.asarray(obj.bond_feat, dtype=np.float32),
        np.asarray(obj.global_feat, dtype=np.float32),
        perm=perm,
        reverse_bonds=reverse_bonds,
    )


# ---------------------------------------------------------------------------
# split cache
# ---------------------------------------------------------------------------

def _concat_molecules(mols: Sequence[Mapping[str, Any]]) -> dict[str, np.ndarray]:
    def cat(key, dtype):
        parts = [np.asarray(m[key], dtype=dtype) for m in mols]
        return np.concatenate(parts, axis=0) if parts else np.zeros(0, dtype=dtype)

    atom_ptr = np.concatenate([[0], np.cumsum([int(m["n_atoms"]) for m in mols])]).astype(np.int32)
    bond_ptr = np.concatenate([[0], np.cumsum([int(m["n_bonds"]) for m in mols])]).astype(np.int32)
    # per-atom pointers for the ragged S1/S2 arrays; the code arrays are the
    # per-molecule concatenation, so shift by cumulative CODE counts (not atom
    # offsets).
    nbr_ptrs, path_ptrs = [], []
    nbr_offset, path_offset = 0, 0
    for m in mols:
        nbr_ptrs.append(np.asarray(m["nbr_ptr"][:-1], dtype=np.int64) + nbr_offset)
        nbr_offset += len(m["nbr_code"])
        path_ptrs.append(np.asarray(m["path_ptr"][:-1], dtype=np.int64) + path_offset)
        path_offset += len(m["path_cat"])
    nbr_ptr = np.concatenate(nbr_ptrs + [np.asarray([nbr_offset])]).astype(np.int32)
    path_ptr = np.concatenate(path_ptrs + [np.asarray([path_offset])]).astype(np.int32)
    pair_ptr = np.concatenate([[0], np.cumsum([len(m["pair_i"]) for m in mols])]).astype(np.int32)

    # bond indices stay molecule-local in the cache; collate adds batch offsets
    bi_parts = [np.asarray(m["bond_index"], dtype=np.int16) for m in mols]
    bond_index = np.concatenate(bi_parts, axis=0) if bi_parts else np.zeros((0, 2), dtype=np.int16)

    out = {
        "atom_ptr": atom_ptr,
        "bond_ptr": bond_ptr,
        "nbr_ptr": nbr_ptr,
        "path_ptr": path_ptr,
        "pair_ptr": pair_ptr,
        "atom_feat": cat("atom_feat", np.float32),
        "atom_type": cat("atom_type", np.int64),
        "atom_scalar": cat("atom_scalar", np.float32),
        "bond_feat": cat("bond_feat", np.float32),
        "bond_type": cat("bond_type", np.int64),
        "bond_index": bond_index,
        "nbr_code": cat("nbr_code", np.uint8),
        "path_cat": cat("path_cat", np.uint8),
        "path_end63": cat("path_end63", np.uint8),
        "path_end": cat("path_end", np.int16),
        "path_closure": cat("path_closure", np.uint8),
        "pair_i": cat("pair_i", np.int16),
        "pair_j": cat("pair_j", np.int16),
        "pair_cat": cat("pair_cat", np.uint8),
        "global": (np.stack([np.asarray(m["global"], dtype=np.float32) for m in mols], axis=0)
                   if mols else np.zeros((0, L.GLOBAL_DIM), np.float32)),
    }
    return out


def build_split(split: str, *, with_labels: bool = True, max_molecules: int | None = None) -> dict:
    if split not in ("train", "valid", "test"):
        raise ValueError(split)
    t0 = time.perf_counter()
    objs, obj_source = L.D.load_p3_objects(split)
    if max_molecules is not None:
        objs = objs[: int(max_molecules)]
        obj_source = dict(obj_source)
        obj_source["max_molecules"] = int(max_molecules)
    mols = []
    for i, obj in enumerate(objs):
        mols.append(build_molecule_from_object(obj))
        if (i + 1) % 1000 == 0:
            L.log(f"[lc prep {split}] {i + 1}/{len(objs)} molecules {time.perf_counter() - t0:.1f}s")
    data = _concat_molecules(mols)
    data["split"] = split
    data["n_molecules"] = int(len(mols))
    if with_labels:
        if split == "test":
            raise RuntimeError("test labels must not be read during preparation")
        labels = L.D.official_labels(split)
        if max_molecules is not None:
            labels = labels[: int(max_molecules)]
        if len(labels) != len(mols):
            raise RuntimeError(f"{split}: labels {len(labels)} vs molecules {len(mols)}")
        data["labels"] = np.asarray(labels, dtype=np.float32)
    data["meta"] = {
        "feature_version": L.FEATURE_VERSION,
        "split": split,
        "n_molecules": int(len(mols)),
        "object_source": obj_source,
        "labels_loaded": bool(with_labels),
        "labels_sha256": L.C.sha256_array(data["labels"]) if with_labels else None,
        "prep_seconds": float(time.perf_counter() - t0),
        "created": L.now(),
        "n_atom_types": L.N_ATOM_TYPES,
        "n_bond_types": L.N_BOND_TYPES,
        "relation_categories": list(L.REL_CATEGORY_NAMES),
    }
    return data


def content_fingerprint(cache: Mapping[str, Any]) -> str:
    """Stable hash of the array contents (ignores the creation timestamp)."""
    import hashlib

    h = hashlib.sha256()
    for key in sorted(cache):
        value = cache[key]
        if isinstance(value, np.ndarray):
            h.update(key.encode())
            h.update(str(value.dtype).encode())
            h.update(str(tuple(value.shape)).encode())
            h.update(np.ascontiguousarray(value).tobytes())
    return h.hexdigest()


def cache_path(split: str):
    return L.CACHE / f"lc_{split}.pt"


def save_cache(data: Mapping[str, Any], split: str) -> str:
    path = cache_path(split)
    L.ensure_dir(path.parent)
    L.atomic_save(dict(data), path)
    return L.C.sha256_file(path)


def load_cache(split: str) -> dict:
    path = cache_path(split)
    if not path.exists():
        raise FileNotFoundError(f"missing LCRC cache {path}; run lc_data.py prep first")
    return torch.load(path, map_location="cpu", weights_only=False)


def fit_norm(train: Mapping[str, Any]) -> dict:
    """Train-only standardisation; global33 reuses the frozen P3 train stats."""
    scalars = np.asarray(train["atom_scalar"], dtype=np.float64)
    smean = scalars.mean(0).astype(np.float32)
    sscale = np.maximum(scalars.std(0).astype(np.float32), 1e-6)
    source = "frozen_p3_prep_train10k"
    if L.P3_PREP.exists():
        prep = L.read_json(L.P3_PREP)
        gmean = np.asarray(prep["global_mean"], dtype=np.float32)
        gscale = np.maximum(np.asarray(prep["global_scale"], dtype=np.float32), 1e-6)
    else:
        g = np.asarray(train["global"], dtype=np.float64)
        gmean = g.mean(0).astype(np.float32)
        gscale = np.maximum(g.std(0).astype(np.float32), 1e-6)
        source = "computed_from_lcrc_train"
    return {
        "source": source,
        "fit_split": "train",
        "valid_or_test_used": False,
        "global_mean": gmean,
        "global_scale": gscale,
        "scalar_mean": smean,
        "scalar_scale": sscale,
    }


def load_norm() -> dict:
    path = L.ANALYSIS / "norm.json"
    d = L.read_json(path)
    for k in ("global_mean", "global_scale", "scalar_mean", "scalar_scale"):
        d[k] = np.asarray(d[k], dtype=np.float32)
    return d


def norm_fingerprint(norm: Mapping[str, Any]) -> str:
    parts = []
    for k in ("global_mean", "global_scale", "scalar_mean", "scalar_scale"):
        parts.append(np.asarray(norm[k], dtype=np.float32).tobytes())
    return L.sha256_bytes(b"".join(parts) + str(norm["source"]).encode())


def prep(splits: Sequence[str] = ("train", "valid"), *, force: bool = False) -> dict:
    out = {"splits": list(splits), "caches": {}, "official_test_loaded": False}
    for split in splits:
        if split not in ("train", "valid"):
            raise ValueError("prep may only build train/valid; the official test is terminal")
        path = cache_path(split)
        if path.exists() and not force:
            data = load_cache(split)
            out["caches"][split] = {
                "path": L.track_rel(path),
                "sha256": L.C.sha256_file(path),
                "reused": True,
                "n_molecules": int(data["n_molecules"]),
                "content_sha256": content_fingerprint(data),
            }
            continue
        data = build_split(split)
        sha = save_cache(data, split)
        out["caches"][split] = {
            "path": L.track_rel(path),
            "sha256": sha,
            "reused": False,
            "n_molecules": int(data["n_molecules"]),
            "content_sha256": content_fingerprint(data),
            "meta": data["meta"],
        }
    if "train" in [s for s, v in out["caches"].items()] or cache_path("train").exists():
        train = load_cache("train")
        norm = fit_norm(train)
        L.write_json(L.ANALYSIS / "norm.json", norm)
        out["norm"] = {
            "path": L.track_rel(L.ANALYSIS / "norm.json"),
            "source": norm["source"],
            "fingerprint": norm_fingerprint(norm),
        }
    return out


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------

def _ranges(ptr: np.ndarray, rows: np.ndarray):
    rows = np.asarray(rows, dtype=np.int64)
    starts = ptr[rows].astype(np.int64)
    ends = ptr[rows + 1].astype(np.int64)
    counts = ends - starts
    idx = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in zip(starts, ends)]) \
        if len(rows) else np.zeros(0, dtype=np.int64)
    batch = np.repeat(np.arange(len(rows), dtype=np.int64), counts) if len(rows) else np.zeros(0, dtype=np.int64)
    local = np.concatenate([np.arange(c, dtype=np.int64) for c in counts]) \
        if len(rows) else np.zeros(0, dtype=np.int64)
    return idx, batch, local, counts


def collate(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, torch.Tensor]:
    rows = np.asarray(rows, dtype=np.int64)
    B = len(rows)
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    atom_idx, atom_batch, atom_local, atom_counts = _ranges(atom_ptr, rows)
    bond_idx, bond_batch, _, bond_counts = _ranges(bond_ptr, rows)
    atom_off = np.concatenate([[0], np.cumsum(atom_counts)[:-1]]).astype(np.int64) if B else np.zeros(0, np.int64)

    bi = np.asarray(cache["bond_index"])[bond_idx].astype(np.int64)
    bi = bi + atom_off[bond_batch].reshape(-1, 1)

    nbr_ptr = np.asarray(cache["nbr_ptr"])
    nbr_counts = nbr_ptr[atom_idx + 1] - nbr_ptr[atom_idx]
    atom_local_idx = np.arange(len(atom_idx), dtype=np.int64)
    nbr_idx = np.concatenate([np.arange(nbr_ptr[a], nbr_ptr[a] + c, dtype=np.int64)
                              for a, c in zip(atom_idx, nbr_counts)]) if len(atom_idx) else np.zeros(0, np.int64)
    nbr_atom = np.repeat(atom_local_idx, nbr_counts) if len(atom_idx) else np.zeros(0, np.int64)

    path_ptr = np.asarray(cache["path_ptr"])
    path_counts = path_ptr[atom_idx + 1] - path_ptr[atom_idx]
    path_idx = np.concatenate([np.arange(path_ptr[a], path_ptr[a] + c, dtype=np.int64)
                               for a, c in zip(atom_idx, path_counts)]) if len(atom_idx) else np.zeros(0, np.int64)
    path_graph = np.repeat(atom_batch, path_counts) if len(atom_idx) else np.zeros(0, np.int64)
    path_atom = np.repeat(atom_local_idx, path_counts) if len(atom_idx) else np.zeros(0, np.int64)
    path_end = np.asarray(cache["path_end"])[path_idx].astype(np.int64) \
        + atom_off[path_graph] if len(path_idx) else np.zeros(0, np.int64)

    pair_ptr = np.asarray(cache["pair_ptr"])
    pair_counts = pair_ptr[rows + 1] - pair_ptr[rows]
    pair_idx = np.concatenate([np.arange(pair_ptr[r], pair_ptr[r] + c, dtype=np.int64)
                               for r, c in zip(rows, pair_counts)]) if B else np.zeros(0, np.int64)
    pair_graph = np.repeat(np.arange(B, dtype=np.int64), pair_counts) if B else np.zeros(0, np.int64)
    pi = np.asarray(cache["pair_i"])[pair_idx].astype(np.int64) + atom_off[pair_graph] \
        if len(pair_idx) else np.zeros(0, np.int64)
    pj = np.asarray(cache["pair_j"])[pair_idx].astype(np.int64) + atom_off[pair_graph] \
        if len(pair_idx) else np.zeros(0, np.int64)

    return {
        "n_graphs": B,
        "atom_feat": torch.from_numpy(np.asarray(cache["atom_feat"])[atom_idx].astype(np.float32)),
        "atom_type": torch.from_numpy(np.asarray(cache["atom_type"])[atom_idx].astype(np.int64)),
        "atom_scalar": torch.from_numpy(np.asarray(cache["atom_scalar"])[atom_idx].astype(np.float32)),
        "atom_batch": torch.from_numpy(atom_batch),
        "atom_off": torch.from_numpy(atom_off),
        "bond_feat": torch.from_numpy(np.asarray(cache["bond_feat"])[bond_idx].astype(np.float32)),
        "bond_type": torch.from_numpy(np.asarray(cache["bond_type"])[bond_idx].astype(np.int64)),
        "bond_index": torch.from_numpy(bi.astype(np.int64)),
        "bond_batch": torch.from_numpy(bond_batch),
        "nbr_atom": torch.from_numpy(nbr_atom),
        "nbr_code": torch.from_numpy(np.asarray(cache["nbr_code"])[nbr_idx].astype(np.int64)),
        "path_atom": torch.from_numpy(path_atom),
        "path_cat": torch.from_numpy(np.asarray(cache["path_cat"])[path_idx].astype(np.int64)),
        "path_end63": torch.from_numpy(np.asarray(cache["path_end63"])[path_idx].astype(np.int64)),
        "path_end": torch.from_numpy(path_end),
        "path_closure": torch.from_numpy(np.asarray(cache["path_closure"])[path_idx].astype(np.int64)),
        "pair_i": torch.from_numpy(pi),
        "pair_j": torch.from_numpy(pj),
        "pair_cat": torch.from_numpy(np.asarray(cache["pair_cat"])[pair_idx].astype(np.int64)),
        "pair_graph": torch.from_numpy(pair_graph),
        "global": torch.from_numpy(np.asarray(cache["global"])[rows].astype(np.float32)),
        "atom_counts": torch.from_numpy(atom_counts.astype(np.int64)),
        "bond_counts": torch.from_numpy(bond_counts.astype(np.int64)),
        "pair_counts": torch.from_numpy(pair_counts.astype(np.int64)),
    }


def batch_meta(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"])
    return {"n_atoms": (atom_ptr[rows + 1] - atom_ptr[rows]).astype(np.int64)}


__all__ = [
    "build_from_arrays", "build_molecule_from_object", "build_split", "cache_path",
    "save_cache", "load_cache", "fit_norm", "load_norm", "norm_fingerprint",
    "content_fingerprint",
    "prep", "collate", "_all_pairs_dist", "_ring_block_membership", "_dist_bucket",
]

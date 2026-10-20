"""Data/representation cache for zinc-arch-discovery-v1.

All features are deterministic, label-free functions of the raw atom/bond graph
(or of the already-frozen P3 object cache, which itself is a label-free one-shot
function of that graph).  Labels are read from the official PyG ZINC subset only
for train/valid by default.  The test split can be built only for a terminal
post-selection score; no model or recipe is selected from it.

Cached representations
----------------------
R0/R1 rooted multi-scale statistics (per atom, radii 1/2/3):
  * root atom feature (P3's fixed 108-D label-free atom descriptor)
  * exact distance-shell atom-type counts (three 21-D blocks)
  * cumulative ball joint counts over unordered endpoint-type pair x bond type
    (231 x 3 = 693 categories; the joint count keeps the endpoint/bond
    correspondence instead of concatenating two marginals)
  * eleven ball scalars (size, degree, ring, branching)
T  reuses the frozen P3 atom/bond/junction object tensors and incidence.
P0/P1 a fixed bank of 128 walk patterns per molecule (32 each of length
  2/4/6/8), with atom types, bond types, and the anonymous first-visit pattern.
B  biconnected blocks + bridges + isolated atoms, block joint counts, and the
  block-cut tree distance / overlap relation.
"""

from __future__ import annotations

import argparse
import math
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import ad_common as C

N_ATOM_TYPES = 21
N_BOND_TYPES = 3
ATOM_FEAT_DIM = 108
BOND_FEAT_DIM = 7
JUNC_FEAT_DIM = 32
GLOBAL_DIM = 33
PAIR_CATEGORIES = N_ATOM_TYPES * (N_ATOM_TYPES + 1) // 2
JOINT_CATEGORIES = PAIR_CATEGORIES * N_BOND_TYPES  # 693
TWO_HOP_CATEGORIES = N_ATOM_TYPES * N_BOND_TYPES * N_BOND_TYPES  # 189
ROOT_SCALARS = 11
BLOCK_SCALARS = 11
WALK_BANK = 128
WALK_LENGTHS = (2, 4, 6, 8)
WALK_PER_LENGTH = 32
TRAIN_WALK_SAMPLE = 8  # 8 per length -> 32 total
MAX_WALK = max(WALK_LENGTHS)
FEATURE_VERSION = "ad-v1-root-r1-t-p-b"

_PAIR_INDEX = np.full((N_ATOM_TYPES, N_ATOM_TYPES), -1, dtype=np.int16)
for _a in range(N_ATOM_TYPES):
    for _b in range(_a, N_ATOM_TYPES):
        idx = sum(N_ATOM_TYPES - i for i in range(_a)) + (_b - _a)
        _PAIR_INDEX[_a, _b] = idx
        _PAIR_INDEX[_b, _a] = idx


def pyg_zinc(split: str):
    from torch_geometric.datasets import ZINC

    if split not in ("train", "valid", "test"):
        raise ValueError(f"unknown split {split!r}")
    pyg_split = "val" if split == "valid" else split
    return ZINC(root=str(C.ZINC_ROOT), subset=True, split=pyg_split)


def split_size(split: str) -> int:
    return {"train": 10000, "valid": 1000, "test": 1000}[split]


def official_labels(split: str) -> np.ndarray:
    ds = pyg_zinc(split)
    return np.asarray([float(ds[i].y.reshape(-1)[0]) for i in range(len(ds))], dtype=np.float32)


def _ensure_objects_modules() -> None:
    C.ensure_p3_modules()


def load_p3_objects(split: str):
    """Load the frozen P3 objects, rebuilding them from official PyG if absent."""
    _ensure_objects_modules()
    import structural_relation_objects_v1 as sro

    if split == "train":
        path = C.P3_OBJECTS_TRAIN
    elif split == "valid":
        path = C.P3_OBJECTS_VALID
    else:
        path = None
    n_expected = split_size(split)
    if path is not None and path.exists():
        objs = torch.load(path, map_location="cpu", weights_only=False)
        if len(objs) != n_expected:
            raise RuntimeError(f"{path}: expected {n_expected}, got {len(objs)}")
        return objs, {"source": "frozen_p3_object_cache", "path": str(path.relative_to(C.REPO)),
                      "sha256": C.sha256_file(path)}
    ds = pyg_zinc(split)
    objs = []
    for i in range(len(ds)):
        d = ds[i]
        objs.append(
            sro.build_mol_objects(
                i,
                d.x.reshape(-1).long().numpy(),
                d.edge_index.numpy(),
                d.edge_attr.reshape(-1).long().numpy(),
            )
        )
    return objs, {"source": "rebuilt_from_official_pyg", "path": None, "sha256": None}


def _atom_types(obj) -> np.ndarray:
    return np.argmax(np.asarray(obj.atom_feat)[:, :N_ATOM_TYPES], axis=1).astype(np.int64)


def _ring_flags(obj):
    import structural_relation_objects_v1 as sro

    blocks = sro.atom_blocks()
    atom_feat = np.asarray(obj.atom_feat)
    ring_atom = atom_feat[:, blocks["in_ring"]].reshape(-1) > 0.5
    ring_bond = np.asarray(obj.bond_feat)[:, 4] > 0.5  # BOND_CATEGORIES offset
    min_ring_bucket = np.argmax(atom_feat[:, blocks["min_ring_size"]], axis=1)
    min_ring = np.where(
        ring_atom,
        np.where(min_ring_bucket >= 5, 8, 3 + min_ring_bucket),
        0,
    ).astype(np.int64)
    return ring_atom, ring_bond, min_ring


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
    inf = np.int64(1 << 20)
    dist = np.full((n_atoms, n_atoms), inf, dtype=np.int64)
    for s in range(n_atoms):
        dist[s, s] = 0
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt: list[int] = []
            for u in frontier:
                for v, _, _ in adj[u]:
                    if dist[s, v] == inf:
                        dist[s, v] = d
                        nxt.append(v)
            frontier = nxt
    return dist


def _joint_category(atom_types: np.ndarray, bond_index: np.ndarray, bond_type: np.ndarray) -> np.ndarray:
    out = np.empty(len(bond_index), dtype=np.int16)
    for e, ((u, v), bt) in enumerate(zip(bond_index, bond_type)):
        a, b = int(atom_types[u]), int(atom_types[v])
        code = int(bt) - 1
        if not (0 <= code < N_BOND_TYPES):
            raise ValueError(f"bond type {bt} outside 1..{N_BOND_TYPES}")
        out[e] = int(_PAIR_INDEX[a, b]) * N_BOND_TYPES + code
    return out


def _rooted(obj, atom_types: np.ndarray, bond_type: np.ndarray, ring_atom, ring_bond, degree, adj, dist):
    n = int(obj.n_atoms)
    joint = np.zeros((n, 3, JOINT_CATEGORIES), dtype=np.uint8)
    shell = np.zeros((n, 3, N_ATOM_TYPES), dtype=np.uint8)
    scalar = np.zeros((n, 3, ROOT_SCALARS), dtype=np.float32)
    bu = np.asarray(obj.bond_index)[:, 0].astype(np.int64)
    bv = np.asarray(obj.bond_index)[:, 1].astype(np.int64)
    bond_cat = _joint_category(atom_types, obj.bond_index, bond_type)
    for v in range(n):
        d = dist[v]
        finite = d < (1 << 20)
        for r in (1, 2, 3):
            exact = d == r
            if exact.any():
                shell[v, r - 1] = np.bincount(atom_types[exact], minlength=N_ATOM_TYPES).astype(np.uint8)
            ball = d <= r
            bm = finite[bu] & finite[bv] & ball[bu] & ball[bv]
            if bm.any():
                joint[v, r - 1] = np.bincount(bond_cat[bm], minlength=JOINT_CATEGORIES).astype(np.uint8)
            atoms = np.where(ball)[0]
            bonds = np.where(bm)[0]
            na, nb = len(atoms), len(bonds)
            deg = degree[atoms] if na else np.zeros(0, dtype=np.int64)
            rb = int(ring_bond[bonds].sum()) if nb else 0
            ra = int(ring_atom[atoms].sum()) if na else 0
            scalar[v, r - 1] = np.asarray(
                [
                    math.log1p(na),
                    math.log1p(nb),
                    rb / max(nb, 1),
                    ra / max(na, 1),
                    float(deg.mean()) if na else 0.0,
                    float(deg.max()) / 4.0 if na else 0.0,
                    float((deg == 1).mean()) if na else 0.0,
                    float((deg >= 3).mean()) if na else 0.0,
                    math.log1p(int(ra)),
                    (float(rb) / max(ra, 1)),
                    float((deg >= 2).mean()) if na else 0.0,
                ],
                dtype=np.float32,
            )
    return joint, shell, scalar


def _anon_pattern(seq: Sequence[int]) -> np.ndarray:
    first: dict[int, int] = {}
    out = np.empty(len(seq), dtype=np.uint8)
    for i, a in enumerate(seq):
        if a not in first:
            first[a] = i
        out[i] = first[a]
    return out


def _sample_walks(obj, atom_types: np.ndarray, bond_type: np.ndarray, adj, rng: np.random.Generator):
    n = int(obj.n_atoms)
    bank_atom = np.zeros((WALK_BANK, MAX_WALK + 1), dtype=np.uint8)
    bank_bond = np.zeros((WALK_BANK, MAX_WALK), dtype=np.uint8)
    bank_anon = np.zeros((WALK_BANK, MAX_WALK + 1), dtype=np.uint8)
    bank_idx = np.zeros((WALK_BANK, MAX_WALK + 1), dtype=np.uint8)
    k = 0
    for L in WALK_LENGTHS:
        for _ in range(WALK_PER_LENGTH):
            start = int(rng.integers(n))
            seq = [start]
            bonds: list[int] = []
            cur = start
            for _step in range(L):
                nbrs = adj[cur]
                if nbrs:
                    nxt, bt, _e = nbrs[int(rng.integers(len(nbrs)))]
                else:
                    nxt, bt = cur, 1
                seq.append(int(nxt))
                bonds.append(int(bt))
                cur = int(nxt)
            atoms_f = np.asarray([atom_types[i] for i in seq], dtype=np.uint8)
            bonds_f = np.asarray([b - 1 for b in bonds], dtype=np.uint8)
            anon_f = _anon_pattern(seq)
            atoms_r = atoms_f[::-1].copy()
            bonds_r = bonds_f[::-1].copy()
            anon_r = _anon_pattern(list(reversed(seq)))
            idx_f = np.asarray(seq, dtype=np.uint8)
            idx_r = idx_f[::-1].copy()
            if (atoms_r.tobytes(), bonds_r.tobytes(), anon_r.tobytes()) < (
                atoms_f.tobytes(), bonds_f.tobytes(), anon_f.tobytes()
            ):
                atoms_f, bonds_f, anon_f = atoms_r, bonds_r, anon_r
                idx_f = idx_r
            bank_atom[k, : L + 1] = atoms_f
            bank_bond[k, :L] = bonds_f
            bank_anon[k, : L + 1] = anon_f
            bank_idx[k, : L + 1] = idx_f
            k += 1
    order = np.concatenate(
        [np.arange(g * WALK_PER_LENGTH, (g + 1) * WALK_PER_LENGTH)[rng.permutation(WALK_PER_LENGTH)]
         for g in range(len(WALK_LENGTHS))]
    ).astype(np.uint8)
    return bank_atom, bank_bond, bank_anon, bank_idx, order


def _two_hop(obj, atom_types: np.ndarray, bond_type: np.ndarray, adj):
    pairs: dict[tuple[int, int, int], int] = {}
    for w in range(int(obj.n_atoms)):
        nbrs = adj[w]
        for x in range(len(nbrs)):
            u, bt1, _ = nbrs[x]
            for y in range(len(nbrs)):
                if x == y:
                    continue
                v, bt2, _ = nbrs[y]
                if u == v:
                    continue
                cat = int(atom_types[w]) * 9 + (int(bt1) - 1) * 3 + (int(bt2) - 1)
                key = (int(u), int(v), int(cat))
                pairs[key] = pairs.get(key, 0) + 1
    keys = sorted(pairs)
    u = np.asarray([k[0] for k in keys], dtype=np.int16)
    v = np.asarray([k[1] for k in keys], dtype=np.int16)
    cat = np.asarray([k[2] for k in keys], dtype=np.int16)
    cnt = np.asarray([pairs[k] for k in keys], dtype=np.uint8)
    return u, v, cat, cnt


def _build_blocks(obj, atom_types: np.ndarray, bond_type: np.ndarray, ring_atom, ring_bond, min_ring, degree):
    import networkx as nx

    n = int(obj.n_atoms)
    bi = np.asarray(obj.bond_index)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from((int(u), int(v)) for u, v in bi)
    comps = list(nx.biconnected_components(graph))
    raw_blocks: list[tuple[list[int], list[int]]] = []
    covered: set[int] = set()
    for comp in comps:
        atoms = sorted(int(a) for a in comp)
        if not atoms:
            continue
        aset = set(atoms)
        bonds = [e for e, (u, v) in enumerate(bi) if int(u) in aset and int(v) in aset]
        raw_blocks.append((atoms, bonds))
        covered.update(atoms)
    for i in range(n):
        if i not in covered:
            raw_blocks.append(([i], []))
    # deterministic block order
    raw_blocks.sort(key=lambda ab: (ab[0][0] if ab[0] else 10**9, len(ab[0]), len(ab[1])))
    membership = np.zeros(n, dtype=np.int64)
    for atoms, _ in raw_blocks:
        membership[atoms] += 1

    nb = len(raw_blocks)
    block_joint = np.zeros((nb, JOINT_CATEGORIES), dtype=np.uint8)
    block_atom_type = np.zeros((nb, N_ATOM_TYPES), dtype=np.uint8)
    block_bond_type = np.zeros((nb, N_BOND_TYPES), dtype=np.uint8)
    block_artic_type = np.zeros((nb, N_ATOM_TYPES), dtype=np.uint8)
    block_scalar = np.zeros((nb, BLOCK_SCALARS), dtype=np.float32)
    atom_lists: list[list[int]] = []
    bond_lists: list[list[int]] = []
    bond_cat_all = _joint_category(atom_types, bi, bond_type)
    for b, (atoms, bonds) in enumerate(raw_blocks):
        atom_lists.append(atoms)
        bond_lists.append(bonds)
        if bonds:
            block_joint[b] = np.bincount(bond_cat_all[bonds], minlength=JOINT_CATEGORIES).astype(np.uint8)
            block_bond_type[b] = np.bincount(
                np.asarray(bond_type)[bonds] - 1, minlength=N_BOND_TYPES
            ).astype(np.uint8)
        block_atom_type[b] = np.bincount(atom_types[atoms], minlength=N_ATOM_TYPES).astype(np.uint8)
        boundary = [a for a in atoms if membership[a] > 1]
        if boundary:
            block_artic_type[b] = np.bincount(atom_types[boundary], minlength=N_ATOM_TYPES).astype(np.uint8)
        na, nbe = len(atoms), len(bonds)
        ra = int(ring_atom[atoms].sum()) if na else 0
        rb = int(ring_bond[bonds].sum()) if nbe else 0
        ring_sizes = [int(min_ring[a]) for a in atoms if ring_atom[a] and min_ring[a] > 0]
        block_scalar[b] = np.asarray(
            [
                math.log1p(na),
                math.log1p(nbe),
                float(nbe - na + 1),
                float(ra),
                float(rb),
                ra / max(na, 1),
                rb / max(nbe, 1),
                (min(ring_sizes) / 8.0) if ring_sizes else 0.0,
                (float(degree[atoms].max()) / 4.0) if na else 0.0,
                float(len(boundary)),
                float(len(boundary)) / max(na, 1),
            ],
            dtype=np.float32,
        )

    # block-cut tree: block nodes + articulation-atom nodes
    articulations = [a for a in range(n) if membership[a] > 1]
    art_to_node = {a: nb + j for j, a in enumerate(articulations)}
    tree = nx.Graph()
    tree.add_nodes_from(range(nb + len(articulations)))
    for b, atoms in enumerate(atom_lists):
        for a in atoms:
            if membership[a] > 1:
                tree.add_edge(b, art_to_node[a])
    block_dist = np.full((nb, nb), 8, dtype=np.uint8)
    block_overlap = np.zeros((nb, nb), dtype=np.uint8)
    atom_sets = [set(a) for a in atom_lists]
    for i in range(nb):
        block_dist[i, i] = 0
        for j in range(nb):
            block_overlap[i, j] = min(len(atom_sets[i] & atom_sets[j]), 3)
        lengths = nx.single_source_shortest_path_length(tree, i)
        for j in range(nb):
            if j in lengths:
                block_dist[i, j] = min(int(lengths[j]), 8)
    block_atom_ptr = np.zeros(nb + 1, dtype=np.int32)
    block_bond_ptr = np.zeros(nb + 1, dtype=np.int32)
    for b in range(nb):
        block_atom_ptr[b + 1] = block_atom_ptr[b] + len(atom_lists[b])
        block_bond_ptr[b + 1] = block_bond_ptr[b] + len(bond_lists[b])
    block_atom_idx = np.asarray([a for atoms in atom_lists for a in atoms], dtype=np.int16)
    block_bond_idx = np.asarray([e for bonds in bond_lists for e in bonds], dtype=np.int16)
    return {
        "n_blocks": nb,
        "block_atom_ptr": block_atom_ptr,
        "block_atom_idx": block_atom_idx,
        "block_bond_ptr": block_bond_ptr,
        "block_bond_idx": block_bond_idx,
        "block_joint": block_joint,
        "block_atom_type": block_atom_type,
        "block_bond_type": block_bond_type,
        "block_artic_type": block_artic_type,
        "block_scalar": block_scalar,
        "block_dist": block_dist.reshape(-1),
        "block_overlap": block_overlap.reshape(-1),
    }


def _assemble_block_arrays(blocks_list):
    """Concatenate per-molecule block arrays with global pointers."""
    n = len(blocks_list)
    block_ptr = np.zeros(n + 1, dtype=np.int32)
    atom_ptr = [0]
    bond_ptr = [0]
    atom_idx, bond_idx = [], []
    for i, b in enumerate(blocks_list):
        nb = int(b["n_blocks"])
        block_ptr[i + 1] = block_ptr[i] + nb
        # local pointers are length nb+1; add current global offsets
        atom_ptr.extend((b["block_atom_ptr"][1:] + atom_ptr[-1]).tolist())
        bond_ptr.extend((b["block_bond_ptr"][1:] + bond_ptr[-1]).tolist())
        atom_idx.append(b["block_atom_idx"])
        bond_idx.append(b["block_bond_idx"])
    return {
        "block_ptr": block_ptr,
        "block_atom_ptr": np.asarray(atom_ptr, dtype=np.int32),
        "block_atom_idx": np.concatenate(atom_idx, axis=0).astype(np.int16) if atom_idx else np.zeros(0, np.int16),
        "block_bond_ptr": np.asarray(bond_ptr, dtype=np.int32),
        "block_bond_idx": np.concatenate(bond_idx, axis=0).astype(np.int16) if bond_idx else np.zeros(0, np.int16),
    }


def build_split_v2(split: str) -> dict:
    """Clean implementation of build_split with global block pointer assembly."""
    t0 = time.perf_counter()
    objs, obj_source = load_p3_objects(split)
    labels = official_labels(split)
    if len(labels) != len(objs):
        raise RuntimeError(f"{split}: labels {len(labels)} vs objects {len(objs)}")
    n_mol = len(objs)

    atom_ptr = np.zeros(n_mol + 1, dtype=np.int32)
    bond_ptr = np.zeros(n_mol + 1, dtype=np.int32)
    junc_ptr = np.zeros(n_mol + 1, dtype=np.int32)
    dist_ptr = np.zeros(n_mol + 1, dtype=np.int32)
    twohop_ptr = np.zeros(n_mol + 1, dtype=np.int32)

    atom_feats, atom_types = [], []
    bond_feats, bond_types, bond_indices = [], [], []
    junc_feats, junc_indices, junc_bonds = [], [], []
    global_feats = []
    root_joint, root_shell, root_scalar = [], [], []
    dist_flat = []
    twohop_u, twohop_v, twohop_cat, twohop_cnt = [], [], [], []
    walk_atom, walk_bond, walk_anon, walk_idx, walk_order = [], [], [], [], []
    blocks_list = []
    walk_rng = np.random.default_rng(
        20261021 + {"train": 0, "valid": 1, "test": 2}[split]
    )

    for i, obj in enumerate(objs):
        n = int(obj.n_atoms)
        at = _atom_types(obj)
        bt = np.asarray(obj.bond_type, dtype=np.int64)
        bi = np.asarray(obj.bond_index, dtype=np.int64)
        if len(bi) and (bt.min() < 1 or bt.max() > N_BOND_TYPES):
            raise RuntimeError(f"{split}[{i}]: bond types {np.unique(bt)}")
        atom_ptr[i + 1] = atom_ptr[i] + n
        bond_ptr[i + 1] = bond_ptr[i] + len(bi)
        junc_ptr[i + 1] = junc_ptr[i] + int(obj.junc_index.shape[0])
        atom_feats.append(np.asarray(obj.atom_feat, dtype=np.float32))
        atom_types.append(at.astype(np.int64))
        bond_feats.append(np.asarray(obj.bond_feat, dtype=np.float32))
        bond_types.append(bt.astype(np.int64))
        bond_indices.append(bi.astype(np.int16))
        junc_feats.append(np.asarray(obj.junc_feat, dtype=np.float32))
        junc_indices.append(np.asarray(obj.junc_index, dtype=np.int16))
        junc_bonds.append(np.asarray(obj.junc_bond, dtype=np.int16))
        global_feats.append(np.asarray(obj.global_feat, dtype=np.float32))

        ring_atom, ring_bond, min_ring = _ring_flags(obj)
        adj = _adjacency(n, bi, bt)
        degree = np.asarray([len(adj[a]) for a in range(n)], dtype=np.int64)
        dist = _all_pairs_dist(n, adj)
        j, sh, sc = _rooted(obj, at, bt, ring_atom, ring_bond, degree, adj, dist)
        root_joint.append(j)
        root_shell.append(sh)
        root_scalar.append(sc)
        dist_bucket = np.clip(dist, 0, 8).astype(np.uint8)
        dist_flat.append(dist_bucket.reshape(-1))
        dist_ptr[i + 1] = dist_ptr[i] + dist_bucket.size

        u, v, cat, cnt = _two_hop(obj, at, bt, adj)
        twohop_u.append(u)
        twohop_v.append(v)
        twohop_cat.append(cat)
        twohop_cnt.append(cnt)
        twohop_ptr[i + 1] = twohop_ptr[i] + len(u)

        wa, wb, wn, wi, order = _sample_walks(obj, at, bt, adj, walk_rng)
        walk_atom.append(wa)
        walk_bond.append(wb)
        walk_anon.append(wn)
        walk_idx.append(wi)
        walk_order.append(order)

        blocks_list.append(
            _build_blocks(obj, at, bt, ring_atom, ring_bond, min_ring, degree)
        )
        if (i + 1) % 1000 == 0:
            C.log(f"[prep {split}] {i + 1}/{n_mol} molecules {time.perf_counter() - t0:.1f}s")

    block_arrays = _assemble_block_arrays(blocks_list)
    bp = np.asarray(block_arrays["block_ptr"], dtype=np.int64)
    block_pair_ptr = np.zeros(n_mol + 1, dtype=np.int32)
    for i in range(n_mol):
        nb = int(bp[i + 1] - bp[i])
        block_pair_ptr[i + 1] = block_pair_ptr[i] + nb * nb
    atom_count_graph = np.zeros((n_mol, N_ATOM_TYPES), dtype=np.uint8)
    bond_count_graph = np.zeros((n_mol, N_BOND_TYPES), dtype=np.uint8)
    for i in range(n_mol):
        b0, b1 = int(bond_ptr[i]), int(bond_ptr[i + 1])
        atom_count_graph[i] = np.bincount(
            np.asarray(atom_types[i]), minlength=N_ATOM_TYPES
        ).astype(np.uint8)
        if b1 > b0:
            bond_count_graph[i] = np.bincount(
                np.asarray(bond_types[i]) - 1, minlength=N_BOND_TYPES
            ).astype(np.uint8)
    data = {
        "split": split,
        "n_molecules": int(n_mol),
        "labels": np.asarray(labels, dtype=np.float32),
        "atom_ptr": atom_ptr,
        "atom_feat": np.concatenate(atom_feats, axis=0).astype(np.float32),
        "atom_type": np.concatenate(atom_types, axis=0).astype(np.int64),
        "bond_ptr": bond_ptr,
        "bond_feat": np.concatenate(bond_feats, axis=0).astype(np.float32),
        "bond_type": np.concatenate(bond_types, axis=0).astype(np.int64),
        "bond_index": np.concatenate(bond_indices, axis=0).astype(np.int16),
        "junc_ptr": junc_ptr,
        "junc_feat": np.concatenate(junc_feats, axis=0).astype(np.float32),
        "junc_index": np.concatenate(junc_indices, axis=0).astype(np.int16),
        "junc_bond": np.concatenate(junc_bonds, axis=0).astype(np.int16),
        "global_feat": np.stack(global_feats, axis=0).astype(np.float32),
        "atom_count_graph": atom_count_graph,
        "bond_count_graph": bond_count_graph,
        "n_atoms_graph": np.diff(atom_ptr).astype(np.int16),
        "n_bonds_graph": np.diff(bond_ptr).astype(np.int16),
        "root_joint": np.concatenate(root_joint, axis=0).astype(np.uint8),
        "root_shell": np.concatenate(root_shell, axis=0).astype(np.uint8),
        "root_scalar": np.concatenate(root_scalar, axis=0).astype(np.float32),
        "dist_ptr": dist_ptr,
        "dist_flat": np.concatenate(dist_flat, axis=0).astype(np.uint8),
        "twohop_ptr": twohop_ptr,
        "twohop_u": np.concatenate(twohop_u, axis=0).astype(np.int16),
        "twohop_v": np.concatenate(twohop_v, axis=0).astype(np.int16),
        "twohop_cat": np.concatenate(twohop_cat, axis=0).astype(np.int16),
        "twohop_count": np.concatenate(twohop_cnt, axis=0).astype(np.uint8),
        "walk_atom": np.stack(walk_atom, axis=0).astype(np.uint8),
        "walk_bond": np.stack(walk_bond, axis=0).astype(np.uint8),
        "walk_anon": np.stack(walk_anon, axis=0).astype(np.uint8),
        "walk_atom_idx": np.stack(walk_idx, axis=0).astype(np.uint8),
        "walk_order": np.stack(walk_order, axis=0).astype(np.uint8),
        **block_arrays,
        "block_pair_ptr": block_pair_ptr,
        "block_joint": np.concatenate([b["block_joint"] for b in blocks_list], axis=0).astype(np.uint8),
        "block_atom_type": np.concatenate([b["block_atom_type"] for b in blocks_list], axis=0).astype(np.uint8),
        "block_bond_type": np.concatenate([b["block_bond_type"] for b in blocks_list], axis=0).astype(np.uint8),
        "block_artic_type": np.concatenate([b["block_artic_type"] for b in blocks_list], axis=0).astype(np.uint8),
        "block_scalar": np.concatenate([b["block_scalar"] for b in blocks_list], axis=0).astype(np.float32),
        "block_dist": np.concatenate([b["block_dist"] for b in blocks_list], axis=0).astype(np.uint8),
        "block_overlap": np.concatenate([b["block_overlap"] for b in blocks_list], axis=0).astype(np.uint8),
        "meta": {
            "feature_version": FEATURE_VERSION,
            "split": split,
            "n_molecules": int(n_mol),
            "object_source": obj_source,
            "labels_sha256": C.sha256_array(labels),
            "walk_bank": WALK_BANK,
            "walk_lengths": list(WALK_LENGTHS),
            "walk_per_length": WALK_PER_LENGTH,
            "joint_categories": JOINT_CATEGORIES,
            "two_hop_categories": TWO_HOP_CATEGORIES,
            "root_scalars": ROOT_SCALARS,
            "block_scalars": BLOCK_SCALARS,
            "prep_seconds": float(time.perf_counter() - t0),
            "created": C.utcnow(),
        },
    }
    return data


def cache_path(split: str) -> Path:
    return C.CACHE / f"ad_{split}.pt"


def save_cache(data: Mapping[str, Any], split: str) -> Path:
    path = cache_path(split)
    C.atomic_torch_save(dict(data), path)
    return path


def load_cache(split: str) -> dict:
    path = cache_path(split)
    if not path.exists():
        raise FileNotFoundError(f"missing cache {path}; run ad_prep.py prep first")
    return torch.load(path, map_location="cpu", weights_only=False)


def fit_norm(train: Mapping[str, Any]) -> dict:
    # Global-33: reuse the frozen P3 train-only mean/scale when available.
    source = "computed_from_train_cache"
    if C.P3_PREP.exists():
        prep = C.read_json(C.P3_PREP)
        gmean = np.asarray(prep["global_mean"], dtype=np.float32)
        gscale = np.asarray(prep["global_scale"], dtype=np.float32)
        source = "frozen_p3_prep_train10k"
    else:
        g = np.asarray(train["global_feat"], dtype=np.float64)
        gmean = g.mean(0).astype(np.float32)
        gscale = g.std(0).astype(np.float32)
    gscale = np.maximum(gscale, 1e-6)

    rs = np.asarray(train["root_scalar"], dtype=np.float64)  # [A,3,S]
    rmean = rs.mean(0).astype(np.float32)
    rscale = np.maximum(rs.std(0).astype(np.float32), 1e-6)

    bs = np.asarray(train["block_scalar"], dtype=np.float64)
    bmean = bs.mean(0).astype(np.float32)
    bscale = np.maximum(bs.std(0).astype(np.float32), 1e-6)

    return {
        "source": source,
        "global_mean": gmean,
        "global_scale": gscale,
        "root_scalar_mean": rmean,
        "root_scalar_scale": rscale,
        "block_scalar_mean": bmean,
        "block_scalar_scale": bscale,
        "fit_split": "train",
        "valid_or_test_used": False,
    }


def load_norm() -> dict:
    path = C.ANALYSIS / "norm.json"
    d = C.read_json(path)
    for k in ("global_mean", "global_scale", "root_scalar_mean", "root_scalar_scale",
              "block_scalar_mean", "block_scalar_scale"):
        d[k] = np.asarray(d[k], dtype=np.float32)
    return d


def audit_split(split: str, data: Mapping[str, Any]) -> dict:
    n = int(data["n_molecules"])
    atom_ptr = np.asarray(data["atom_ptr"])
    bond_ptr = np.asarray(data["bond_ptr"])
    block_ptr = np.asarray(data["block_ptr"])
    dist_ptr = np.asarray(data["dist_ptr"])
    twohop_ptr = np.asarray(data["twohop_ptr"])
    n_atoms = int(atom_ptr[-1])
    n_bonds = int(bond_ptr[-1])
    n_blocks = int(block_ptr[-1])
    if n_atoms != data["atom_feat"].shape[0] or n_atoms != data["root_joint"].shape[0]:
        raise RuntimeError("atom count mismatch")
    if n_bonds != data["bond_feat"].shape[0]:
        raise RuntimeError("bond count mismatch")
    if n_blocks != data["block_joint"].shape[0]:
        raise RuntimeError("block count mismatch")
    if not np.isfinite(data["labels"]).all():
        raise RuntimeError("non-finite labels")
    if data["bond_type"].size and (data["bond_type"].min() < 1 or data["bond_type"].max() > 3):
        raise RuntimeError("bond type vocabulary")
    if data["atom_type"].min() < 0 or data["atom_type"].max() >= N_ATOM_TYPES:
        raise RuntimeError("atom type vocabulary")
    # root joint mass equals the number of ball bonds; check a sample
    bad = 0
    for i in range(min(n, 200)):
        a0, a1 = int(atom_ptr[i]), int(atom_ptr[i + 1])
        b0, b1 = int(bond_ptr[i]), int(bond_ptr[i + 1])
        for v in range(a0, min(a1, a0 + 5)):
            # distance rows were built from the raw graph; the mass check is
            # that every ball bond lands in exactly one joint category.
            for r in range(3):
                if int(data["root_joint"][v, r].sum()) > b1 - b0:
                    bad += 1
    # block joint mass equals block bonds
    for b in range(min(n_blocks, 200)):
        nb = int(data["block_bond_ptr"][b + 1] - data["block_bond_ptr"][b])
        if int(data["block_joint"][b].sum()) != nb:
            raise RuntimeError(f"block {b}: joint mass {data['block_joint'][b].sum()} != {nb}")
    # walk lengths / reversal canonicalization on a sample
    wal = np.asarray(data["walk_atom"])
    wbo = np.asarray(data["walk_bond"])
    wan = np.asarray(data["walk_anon"])
    widx = np.asarray(data["walk_atom_idx"])
    for i in range(min(n, 100)):
        for k in range(WALK_BANK):
            L = WALK_LENGTHS[k // WALK_PER_LENGTH]
            atom_seq = wal[i, k, : L + 1]
            bond_seq = wbo[i, k, :L]
            anon = wan[i, k, : L + 1]
            idx_seq = widx[i, k, : L + 1]
            if not np.array_equal(_anon_pattern(list(idx_seq)), anon):
                raise RuntimeError("anonymous pattern mismatch")
            # canonical orientation must be <= reverse (recomputed from the
            # stored atom-index sequence, which is the actual walk identity)
            ia, ib = idx_seq[::-1], bond_seq[::-1]
            ianon = _anon_pattern(list(ia))
            ra = np.asarray([data["atom_type"][int(data["atom_ptr"][i]) + int(a)] for a in ia], dtype=np.uint8)
            if (ra.tobytes(), ib.tobytes(), ianon.tobytes()) < (
                atom_seq.tobytes(), bond_seq.tobytes(), anon.tobytes()
            ):
                raise RuntimeError("walk not reversal-canonical")
    # global feature finite
    if not np.isfinite(data["global_feat"]).all():
        raise RuntimeError("non-finite global feature")
    return {
        "split": split,
        "n_molecules": n,
        "n_atoms": n_atoms,
        "n_bonds": n_bonds,
        "n_blocks": n_blocks,
        "mean_atoms": float(n_atoms / n),
        "mean_bonds": float(n_bonds / n),
        "mean_blocks": float(n_blocks / n),
        "mean_walks": float(data["walk_atom"].shape[0] * WALK_BANK / n),
        "labels_mean": float(np.mean(data["labels"])),
        "labels_std": float(np.std(data["labels"])),
        "joint_mass_check_violations_sampled": int(bad),
        "block_joint_mass_ok_sampled": True,
        "walk_canonical_sample_ok": True,
        "label_sha256": C.sha256_array(data["labels"]),
        "dist_entries": int(dist_ptr[-1]),
        "twohop_entries": int(twohop_ptr[-1]),
    }


def prep(splits: Sequence[str] = ("train", "valid"), force: bool = False) -> dict:
    C.CACHE.mkdir(parents=True, exist_ok=True)
    out: dict[str, Any] = {}
    audit: dict[str, Any] = {}
    for split in splits:
        path = cache_path(split)
        if path.exists() and not force:
            C.log(f"[prep] reuse {path}")
            data = load_cache(split)
        else:
            C.log(f"[prep] build {split}")
            data = build_split_v2(split)
            save_cache(data, split)
        audit[split] = audit_split(split, data)
        out[split] = {"path": str(path), "size_bytes": path.stat().st_size,
                      "meta": data["meta"], "audit": audit[split]}
    if "train" in out:
        norm = fit_norm(load_cache("train"))
        C.write_json(C.ANALYSIS / "norm.json", norm)
        out["norm"] = {
            "source": norm["source"],
            "global_mean": norm["global_mean"].tolist(),
            "global_scale": norm["global_scale"].tolist(),
            "root_scalar_mean": norm["root_scalar_mean"].tolist(),
            "root_scalar_scale": norm["root_scalar_scale"].tolist(),
            "block_scalar_mean": norm["block_scalar_mean"].tolist(),
            "block_scalar_scale": norm["block_scalar_scale"].tolist(),
        }
    out["audit"] = audit
    C.write_json(C.ANALYSIS / "prep_audit.json", out)
    return out


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("command", choices=["prep", "audit", "inspect"])
    p.add_argument("--splits", default="train,valid")
    p.add_argument("--force", action="store_true")
    args = p.parse_args(argv)
    if args.command == "prep":
        prep(tuple(args.splits.split(",")), force=args.force)
    elif args.command == "audit":
        for split in args.splits.split(","):
            C.log(audit_split(split, load_cache(split)))
    else:
        for split in args.splits.split(","):
            d = load_cache(split)
            C.log({k: (v.shape if hasattr(v, "shape") else v) for k, v in d.items() if k != "meta"})
            C.log(d["meta"])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

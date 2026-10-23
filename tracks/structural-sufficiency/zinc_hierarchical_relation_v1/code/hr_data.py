"""Label-blind hierarchical object cache for zinc_hierarchical_relation_v1.

Builds, per molecule:

* the frozen BJOC atom / bond / junction objects (``structural_relation_objects_v1``);
* one **rooted induced patch** per bond: the actual atom set within graph
  distance <= patch_radius of either endpoint and the *induced* edge set among
  those atoms (bond type, endpoint distances to the root bond, root flag),
  plus patch scalars;
* graph relations between structural tokens (junctions + patches): clipped
  anchor distance, shared ring block, short-walk counts ``A^2``;
* train-only structural support for a coarse radius-2 nested-WL bond key.

Nothing here reads ``y``, SMILES, logP/SA, or the official test split.
"""

from __future__ import annotations

import json
import math
import pickle
import time
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import hr_common as C
import structural_relation_objects_v1 as sro

ATOM_DIM = sro.ATOM_FEAT_DIM
BOND_DIM = sro.BOND_FEAT_DIM
BOND_CATEGORIES = sro.BOND_CATEGORIES
JUNC_DIM = sro.JUNC_FEAT_DIM
GLOBAL_DIM = sro.GLOBAL_DIM

PATCH_SCALARS = (
    "log1p_n_nodes",
    "log1p_n_edges",
    "cycle_rank",
    "n_chords",
    "n_min0",
    "n_min1",
    "n_min2",
    "n_max0",
    "n_max1",
    "n_max2",
    "n_abs0",
    "n_abs1",
    "n_abs2",
    "root_ring",
    "frac_ring_edges",
    "frac_ring_nodes",
)
PATCH_SCALAR_DIM = len(PATCH_SCALARS)
SUPPORT_DIM = 4
NODE_TAG_DIM = 4  # distance bucket 0,1,2,>=3
#: symmetric root-bond tag: one-hot(min(du,dv)), one-hot(max(du,dv)), one-hot(|du-dv|)
PATCH_TAG_DIM = 3 * NODE_TAG_DIM
#: bond type + symmetric endpoint-tag sum + abs difference + root flag
EDGE_FEAT_DIM = BOND_CATEGORIES + 2 * PATCH_TAG_DIM + 1

REL_SELF = 0
REL_TYPES = 33  # 0 = self, 1..32 = (distance 0..3) x (same ring block) x (A^2 0..3)
DIST_BUCKETS = 4
WALK2_BUCKETS = 4


# ---------------------------------------------------------------------------
# graph helpers
# ---------------------------------------------------------------------------


def _all_pairs(n_atoms: int, adj: Sequence[Sequence[int]]) -> np.ndarray:
    inf = np.int64(99)
    dist = np.full((n_atoms, n_atoms), inf, dtype=np.int64)
    for s in range(n_atoms):
        dist[s, s] = 0
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt: list[int] = []
            for u in frontier:
                for v in adj[u]:
                    if dist[s, v] == inf:
                        dist[s, v] = d
                        nxt.append(v)
            frontier = nxt
    return dist


def _ring_blocks(n_atoms: int, bonds: Sequence[tuple[int, int]]) -> tuple[list[set[int]], dict[tuple[int, int], bool], np.ndarray]:
    import networkx as nx

    graph = nx.Graph()
    graph.add_nodes_from(range(n_atoms))
    graph.add_edges_from(bonds)
    blocks: list[set[int]] = [set(b) for b in nx.biconnected_components(graph) if len(b) >= 3]
    atom_blocks: list[set[int]] = [set() for _ in range(n_atoms)]
    for bi, block in enumerate(blocks):
        for a in block:
            atom_blocks[a].add(bi)
    is_ring_bond: dict[tuple[int, int], bool] = {}
    for bi, block in enumerate(blocks):
        sub = graph.subgraph(block)
        for u, v in sub.edges:
            is_ring_bond[(min(u, v), max(u, v))] = True
    ring_atom = np.zeros(n_atoms, dtype=bool)
    for a in range(n_atoms):
        if atom_blocks[a]:
            ring_atom[a] = True
    return atom_blocks, is_ring_bond, ring_atom


def _bucket(value: int, cap: int) -> int:
    return int(min(max(value, 0), cap))


# ---------------------------------------------------------------------------
# per-molecule build
# ---------------------------------------------------------------------------


def build_entry(
    mol_index: int,
    x: np.ndarray,
    edge_index: np.ndarray,
    edge_attr: np.ndarray,
    *,
    patch_radius: int = 2,
    null_seed: int = 12345,
) -> dict[str, Any]:
    """Build one label-blind cache entry (numpy arrays only)."""
    atom_types = [int(v) for v in np.asarray(x).reshape(-1)]
    obj = sro.build_mol_objects(mol_index, x, edge_index, edge_attr)
    n_atoms = int(obj.n_atoms)
    n_bonds = int(obj.bond_index.shape[0])
    n_junc = int(obj.junc_index.shape[0])
    bonds = [(int(u), int(v)) for u, v in obj.bond_index]
    bond_types = [int(t) for t in obj.bond_type]
    bond_type_map = {(u, v): bt for (u, v), bt in zip(bonds, bond_types)}

    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for u, v in bonds:
        adj[u].append(v)
        adj[v].append(u)
    for a in range(n_atoms):
        adj[a].sort()

    dist = _all_pairs(n_atoms, adj)
    adjacency = (dist == 1).astype(np.int64)
    walk2 = adjacency @ adjacency
    atom_blocks, is_ring_bond_map, ring_atom = _ring_blocks(n_atoms, bonds)
    is_ring_bond = np.asarray(
        [bool(is_ring_bond_map.get((u, v), False)) for u, v in bonds], dtype=bool
    )

    # ---- rooted induced patches (one per bond) ----------------------------
    pnode_atom: list[int] = []
    pnode_patch: list[int] = []
    pnode_du: list[int] = []
    pnode_dv: list[int] = []
    pedge_x: list[int] = []
    pedge_y: list[int] = []
    pedge_patch: list[int] = []
    pedge_type: list[int] = []
    pedge_du_x: list[int] = []
    pedge_du_y: list[int] = []
    pedge_dv_x: list[int] = []
    pedge_dv_y: list[int] = []
    pedge_root: list[int] = []
    p_scalars: list[list[float]] = []

    for e, (u, v) in enumerate(bonds):
        nodes = [
            z
            for z in range(n_atoms)
            if min(int(dist[u, z]), int(dist[v, z])) <= int(patch_radius)
        ]
        if not nodes:
            nodes = [u, v]
        nodes = sorted(set(nodes) | {u, v})
        node_pos = {z: k for k, z in enumerate(nodes)}
        du = [_bucket(int(dist[u, z]), 3) for z in nodes]
        dv = [_bucket(int(dist[v, z]), 3) for z in nodes]
        for z, a_du, a_dv in zip(nodes, du, dv):
            pnode_atom.append(z)
            pnode_patch.append(e)
            pnode_du.append(a_du)
            pnode_dv.append(a_dv)
        edges: list[tuple[int, int, int]] = []
        for ii, a in enumerate(nodes):
            for b in nodes[ii + 1 :]:
                if dist[a, b] == 1:
                    edges.append((a, b, int(bond_type_map[(a, b)])))
        n_edges = len(edges)
        n_nodes = len(nodes)
        chord_count = n_edges - n_nodes + 1
        dmin = [min(a, b) for a, b in zip(du, dv)]
        dmax = [max(a, b) for a, b in zip(du, dv)]
        dabs = [abs(a - b) for a, b in zip(du, dv)]
        n_min = [sum(1 for z in dmin if z == k) for k in range(3)]
        n_max = [sum(1 for z in dmax if z == k) for k in range(3)]
        n_abs = [sum(1 for z in dabs if z == k) for k in range(3)]
        ring_edges = 0
        for a, b, _bt in edges:
            pedge_x.append(a)
            pedge_y.append(b)
            pedge_patch.append(e)
            pedge_type.append(_bt)
            pedge_du_x.append(_bucket(int(dist[u, a]), 3))
            pedge_du_y.append(_bucket(int(dist[u, b]), 3))
            pedge_dv_x.append(_bucket(int(dist[v, a]), 3))
            pedge_dv_y.append(_bucket(int(dist[v, b]), 3))
            is_root = 1 if (a == u and b == v) or (a == v and b == u) else 0
            pedge_root.append(is_root)
            if is_ring_bond_map.get((a, b), False):
                ring_edges += 1
        p_scalars.append(
            [
                math.log1p(n_nodes),
                math.log1p(n_edges),
                float(chord_count),
                float(n_edges - n_nodes),
                float(n_min[0]),
                float(n_min[1]),
                float(n_min[2]),
                float(n_max[0]),
                float(n_max[1]),
                float(n_max[2]),
                float(n_abs[0]),
                float(n_abs[1]),
                float(n_abs[2]),
                float(is_ring_bond[e]),
                float(ring_edges) / max(n_edges, 1),
                float(sum(1 for z in nodes if ring_atom[z])) / max(n_nodes, 1),
            ]
        )

    # ---- structural token relations (junctions + patches) -----------------
    n_tokens = n_junc + n_bonds
    anchors: list[tuple[int, ...]] = []
    for i, j, k in obj.junc_index:
        anchors.append((int(j),))
    for u, v in bonds:
        anchors.append((int(u), int(v)))

    r_u: list[int] = []
    r_v: list[int] = []
    r_rel: list[int] = []
    for a in range(n_tokens):
        for b in range(a + 1, n_tokens):
            d = min(int(dist[x, y]) for x in anchors[a] for y in anchors[b])
            same = bool(
                set().union(*(atom_blocks[x] for x in anchors[a]))
                & set().union(*(atom_blocks[x] for x in anchors[b]))
            )
            w2 = min(int(walk2[x, y]) for x in anchors[a] for y in anchors[b])
            if d <= 3 or same:
                d_b = _bucket(d, 3)
                w_b = _bucket(w2, 3)
                rel = 1 + d_b * 8 + int(same) * 4 + w_b
                r_u.append(a)
                r_v.append(b)
                r_rel.append(rel)
    r_u_arr = np.asarray(r_u, dtype=np.int32)
    r_v_arr = np.asarray(r_v, dtype=np.int32)
    r_rel_arr = np.asarray(r_rel, dtype=np.int8)
    rng = np.random.default_rng(int(null_seed) + int(mol_index))
    r_rel_null = r_rel_arr[rng.permutation(len(r_rel_arr))] if len(r_rel_arr) else r_rel_arr.copy()

    # ---- coarse bond support key (radius-2 nested-WL, bond centred) -------
    adj_bt = [
        tuple(sorted((int(bond_type_map[(min(a, z), max(a, z))]), int(z)) for z in adj[a]))
        for a in range(n_atoms)
    ]
    color1 = [
        (int(atom_types[a]), tuple(sorted((bt, int(atom_types[z])) for bt, z in adj_bt[a])))
        for a in range(n_atoms)
    ]
    bond_keys: list[bytes] = []
    for e, (u, v) in enumerate(bonds):
        end_u = (int(atom_types[u]), tuple(sorted((bt, color1[z]) for bt, z in adj_bt[u])))
        end_v = (int(atom_types[v]), tuple(sorted((bt, color1[z]) for bt, z in adj_bt[v])))
        key = (int(bond_types[e]), int(is_ring_bond[e]), tuple(sorted((end_u, end_v))))
        bond_keys.append(repr(key).encode("utf-8"))

    return {
        "mol_index": int(mol_index),
        "n_atoms": n_atoms,
        "n_bonds": n_bonds,
        "n_junc": n_junc,
        "n_patches": n_bonds,
        "n_tokens": n_tokens,
        "atom_feat": np.asarray(obj.atom_feat, dtype=np.float32),
        "bond_index": np.asarray(obj.bond_index, dtype=np.int64).reshape(n_bonds, 2),
        "bond_type": np.asarray(obj.bond_type, dtype=np.int64),
        "bond_feat": np.asarray(obj.bond_feat, dtype=np.float32),
        "junc_index": np.asarray(obj.junc_index, dtype=np.int64).reshape(n_junc, 3),
        "junc_bond": np.asarray(obj.junc_bond, dtype=np.int64).reshape(n_junc, 2),
        "junc_feat": np.asarray(obj.junc_feat, dtype=np.float32),
        "global_feat": np.asarray(obj.global_feat, dtype=np.float32),
        "pnode_atom": np.asarray(pnode_atom, dtype=np.int64),
        "pnode_patch": np.asarray(pnode_patch, dtype=np.int64),
        "pnode_du": np.asarray(pnode_du, dtype=np.int64),
        "pnode_dv": np.asarray(pnode_dv, dtype=np.int64),
        "pedge_x": np.asarray(pedge_x, dtype=np.int64),
        "pedge_y": np.asarray(pedge_y, dtype=np.int64),
        "pedge_patch": np.asarray(pedge_patch, dtype=np.int64),
        "pedge_type": np.asarray(pedge_type, dtype=np.int64),
        "pedge_du_x": np.asarray(pedge_du_x, dtype=np.int64),
        "pedge_du_y": np.asarray(pedge_du_y, dtype=np.int64),
        "pedge_dv_x": np.asarray(pedge_dv_x, dtype=np.int64),
        "pedge_dv_y": np.asarray(pedge_dv_y, dtype=np.int64),
        "pedge_root": np.asarray(pedge_root, dtype=np.int64),
        "p_scalars_raw": np.asarray(p_scalars, dtype=np.float32).reshape(n_bonds, PATCH_SCALAR_DIM),
        "r_u": r_u_arr,
        "r_v": r_v_arr,
        "r_rel": r_rel_arr,
        "r_rel_null": r_rel_null.astype(np.int8),
        "bond_keys": bond_keys,
        "_stats_pair": np.asarray([n_nodes, n_edges], dtype=np.int64) if n_bonds else np.zeros(2, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# split build / support
# ---------------------------------------------------------------------------


def _official_split(split: str):
    from torch_geometric.datasets import ZINC

    import cscl_features as cf

    pyg_split = {"train": "train", "val": "val"}[split]
    ds = ZINC(root=str(cf.ZINC_ROOT), subset=True, split=pyg_split)
    expected = {"train": C.N_TRAIN, "val": C.N_VALID}[split]
    if len(ds) != expected:
        raise SystemExit(f"official {split} has {len(ds)} rows, expected {expected}")
    return ds


def build_split(split: str, *, patch_radius: int = 2, limit: int | None = None) -> list[dict[str, Any]]:
    ds = _official_split(split)
    n = len(ds) if limit is None else min(int(limit), len(ds))
    entries: list[dict[str, Any]] = []
    t0 = time.time()
    for i in range(n):
        g = ds[i]
        entries.append(
            build_entry(
                i,
                g.x.numpy().copy(),
                g.edge_index.numpy().copy(),
                g.edge_attr.numpy().copy(),
                patch_radius=patch_radius,
            )
        )
    print(f"[hr_data] built {split} {n} objects in {time.time() - t0:.1f}s", flush=True)
    return entries


def fit_support_and_stats(entries: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Fit all label-blind preprocessing statistics on the train entries."""
    counter: Counter[bytes] = Counter()
    for e in entries:
        counter.update(e["bond_keys"])
    counts = np.asarray([counter[k] for e in entries for k in e["bond_keys"]], dtype=np.float64)
    logc = np.log1p(counts)
    support = {
        "counter": counter,
        "n_bonds": int(len(counts)),
        "n_unique": int(len(counter)),
        "logc_mean": float(logc.mean()),
        "logc_scale": float(max(logc.std(), 1e-3)),
        "count_sorted": np.sort(counts.astype(np.float64)),
        "unique_count_sorted": np.sort(np.asarray(list(counter.values()), dtype=np.float64)),
    }
    p_scalars = np.concatenate([e["p_scalars_raw"] for e in entries], axis=0)
    patch_stats = {
        "mean": p_scalars.mean(axis=0),
        "scale": np.maximum(p_scalars.std(axis=0), 1e-3),
    }
    g = np.stack([e["global_feat"] for e in entries], axis=0)
    global_stats = {
        "mean": g.mean(axis=0),
        "scale": np.maximum(g.std(axis=0), 1e-3),
    }
    # node/edge count summary (report only)
    pair = np.stack([e["_stats_pair"] for e in entries], axis=0)
    return {
        "support": support,
        "patch": patch_stats,
        "global": global_stats,
        "patch_count_summary": {
            "n_patches": int(len(p_scalars)),
            "mean_nodes_last_patch": float(pair[:, 0].mean()),
            "mean_edges_last_patch": float(pair[:, 1].mean()),
        },
    }


def apply_stats(entries: Sequence[dict[str, Any]], stats: Mapping[str, Any]) -> None:
    sup = stats["support"]
    counter: Counter[bytes] = sup["counter"]
    sorted_counts = sup["unique_count_sorted"]
    total = max(len(sorted_counts), 1)
    mean = float(sup["logc_mean"])
    scale = float(sup["logc_scale"])
    pm = np.asarray(stats["patch"]["mean"], dtype=np.float32)
    ps = np.asarray(stats["patch"]["scale"], dtype=np.float32)
    gm = np.asarray(stats["global"]["mean"], dtype=np.float32)
    gs = np.asarray(stats["global"]["scale"], dtype=np.float32)
    for e in entries:
        keys = e.pop("bond_keys")
        counts = np.asarray([counter.get(k, 0) for k in keys], dtype=np.float64)
        logc = np.log1p(counts)
        z = np.clip((logc - mean) / scale, -3.0, 3.0)
        quant = np.searchsorted(sorted_counts, counts, side="right") / float(total)
        unseen = (counts == 0).astype(np.float64)
        small = np.minimum(counts, 8.0) / 8.0
        e["p_support"] = np.stack([z, quant * 2.0 - 1.0, unseen, small], axis=1).astype(np.float32)
        e["p_scalars"] = ((e.pop("p_scalars_raw") - pm) / ps).astype(np.float32)
        e["global_feat"] = ((e["global_feat"] - gm) / gs).astype(np.float32)
        e.pop("_stats_pair", None)


def save_cache(entries: Sequence[dict[str, Any]], path: Path) -> str:
    import torch

    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(list(entries), path)
    return C.sha256_file(path)


def load_cache(path: Path) -> list[dict[str, Any]]:
    import torch

    return torch.load(path, map_location="cpu", weights_only=False)


def save_stats(stats: Mapping[str, Any], path: Path) -> None:
    stats = dict(stats)
    sup = dict(stats["support"])
    counter = sup.pop("counter")
    np.savez_compressed(
        path,
        patch_mean=np.asarray(stats["patch"]["mean"], dtype=np.float32),
        patch_scale=np.asarray(stats["patch"]["scale"], dtype=np.float32),
        global_mean=np.asarray(stats["global"]["mean"], dtype=np.float32),
        global_scale=np.asarray(stats["global"]["scale"], dtype=np.float32),
        support_count_sorted=sup["count_sorted"],
        support_unique_count_sorted=sup["unique_count_sorted"],
        support_logc_mean=np.asarray([sup["logc_mean"]], dtype=np.float64),
        support_logc_scale=np.asarray([sup["logc_scale"]], dtype=np.float64),
        support_n_bonds=np.asarray([sup["n_bonds"]], dtype=np.int64),
        support_n_unique=np.asarray([sup["n_unique"]], dtype=np.int64),
    )
    with open(path.with_suffix(".counter.pkl"), "wb") as fh:
        pickle.dump(counter, fh)


def load_stats(path: Path) -> dict[str, Any]:
    z = np.load(path, allow_pickle=False)
    with open(path.with_suffix(".counter.pkl"), "rb") as fh:
        counter = pickle.load(fh)
    return {
        "patch": {"mean": z["patch_mean"], "scale": z["patch_scale"]},
        "global": {"mean": z["global_mean"], "scale": z["global_scale"]},
        "support": {
            "counter": counter,
            "count_sorted": z["support_count_sorted"],
            "unique_count_sorted": z["support_unique_count_sorted"],
            "logc_mean": float(z["support_logc_mean"][0]),
            "logc_scale": float(z["support_logc_scale"][0]),
            "n_bonds": int(z["support_n_bonds"][0]),
            "n_unique": int(z["support_n_unique"][0]),
        },
    }


# ---------------------------------------------------------------------------
# collate
# ---------------------------------------------------------------------------


def _cat(entries: Sequence[Mapping[str, Any]], key: str, dtype: np.dtype) -> np.ndarray:
    rows = [np.asarray(e[key], dtype=dtype) for e in entries if len(e[key])]
    if not rows:
        return np.zeros((0,) + tuple(np.asarray(entries[0][key]).shape[1:]), dtype=dtype)
    return np.concatenate(rows, axis=0)


def collate(entries: Sequence[Mapping[str, Any]]):
    """Concatenate per-molecule arrays into a flat batch (vectorised offsets).

    Token ids are global over the type-flat layout: junction token ``i`` keeps
    its flat junction row; patch token ``p`` becomes
    ``n_junctions_batch + p``.
    """
    import torch

    n_graphs = len(entries)
    na = np.asarray([int(e["n_atoms"]) for e in entries], dtype=np.int64)
    nb = np.asarray([int(e["n_bonds"]) for e in entries], dtype=np.int64)
    nj = np.asarray([int(e["n_junc"]) for e in entries], dtype=np.int64)
    aoff = np.concatenate([[0], np.cumsum(na)[:-1]]).astype(np.int64)
    boff = np.concatenate([[0], np.cumsum(nb)[:-1]]).astype(np.int64)
    joff = np.concatenate([[0], np.cumsum(nj)[:-1]]).astype(np.int64)
    n_atoms_total = int(na.sum())
    n_bonds_total = int(nb.sum())
    n_junc_total = int(nj.sum())
    n_patch_total = n_bonds_total
    n_tokens_total = n_junc_total + n_patch_total

    atom_feat = np.concatenate([e["atom_feat"] for e in entries], axis=0)
    atom_batch = np.repeat(np.arange(n_graphs), na)
    bond_index = np.concatenate([e["bond_index"] + aoff[i] for i, e in enumerate(entries)], axis=0)
    bond_type = np.concatenate([e["bond_type"] for e in entries], axis=0)
    bond_feat = np.concatenate([e["bond_feat"] for e in entries], axis=0)
    bond_batch = np.repeat(np.arange(n_graphs), nb)
    junc_index = np.concatenate([e["junc_index"] + aoff[i] for i, e in enumerate(entries)], axis=0)
    junc_bond = np.concatenate([e["junc_bond"] + boff[i] for i, e in enumerate(entries)], axis=0)
    junc_feat = np.concatenate([e["junc_feat"] for e in entries], axis=0)
    junc_batch = np.repeat(np.arange(n_graphs), nj)
    pnode_atom = np.concatenate([e["pnode_atom"] + aoff[i] for i, e in enumerate(entries)], axis=0)
    pnode_patch = np.concatenate([e["pnode_patch"] + boff[i] for i, e in enumerate(entries)], axis=0)
    pnode_du = np.concatenate([e["pnode_du"] for e in entries], axis=0)
    pnode_dv = np.concatenate([e["pnode_dv"] for e in entries], axis=0)
    pedge_x = np.concatenate([e["pedge_x"] + aoff[i] for i, e in enumerate(entries)], axis=0)
    pedge_y = np.concatenate([e["pedge_y"] + aoff[i] for i, e in enumerate(entries)], axis=0)
    pedge_patch = np.concatenate([e["pedge_patch"] + boff[i] for i, e in enumerate(entries)], axis=0)
    pedge_type = np.concatenate([e["pedge_type"] for e in entries], axis=0)
    pedge_du_x = np.concatenate([e["pedge_du_x"] for e in entries], axis=0)
    pedge_du_y = np.concatenate([e["pedge_du_y"] for e in entries], axis=0)
    pedge_dv_x = np.concatenate([e["pedge_dv_x"] for e in entries], axis=0)
    pedge_dv_y = np.concatenate([e["pedge_dv_y"] for e in entries], axis=0)
    pedge_root = np.concatenate([e["pedge_root"] for e in entries], axis=0)
    p_scalars = np.concatenate([e["p_scalars"] for e in entries], axis=0)
    p_support = np.concatenate([e["p_support"] for e in entries], axis=0)
    patch_batch = np.repeat(np.arange(n_graphs), nb)
    global_feat = np.stack([e["global_feat"] for e in entries], axis=0)
    counts = np.stack([np.asarray([e["n_atoms"], e["n_bonds"], e["n_junc"], e["n_patches"]], dtype=np.float32)
                       for e in entries], axis=0)

    # ---- relations: local token id -> global type-flat token id ----------
    pair_counts = np.asarray([len(e["r_u"]) for e in entries], dtype=np.int64)
    r_mol = np.repeat(np.arange(n_graphs), pair_counts)
    r_u = np.concatenate([e["r_u"].astype(np.int64) for e in entries], axis=0) if pair_counts.sum() else np.zeros(0, np.int64)
    r_v = np.concatenate([e["r_v"].astype(np.int64) for e in entries], axis=0) if pair_counts.sum() else np.zeros(0, np.int64)
    r_rel = np.concatenate([e["r_rel"].astype(np.int64) for e in entries], axis=0) if pair_counts.sum() else np.zeros(0, np.int64)
    r_reln = np.concatenate([e["r_rel_null"].astype(np.int64) for e in entries], axis=0) if pair_counts.sum() else np.zeros(0, np.int64)
    nj_m = nj[r_mol]
    joff_m = joff[r_mol]
    boff_m = boff[r_mol]
    gid_u = np.where(r_u < nj_m, joff_m + r_u, n_junc_total + boff_m + (r_u - nj_m))
    gid_v = np.where(r_v < nj_m, joff_m + r_v, n_junc_total + boff_m + (r_v - nj_m))
    src = np.concatenate([gid_u, gid_v], axis=0)
    dst = np.concatenate([gid_v, gid_u], axis=0)
    rel = np.concatenate([r_rel, r_rel], axis=0)
    reln = np.concatenate([r_reln, r_reln], axis=0)
    self_parts = [joff[i] + np.arange(nj[i], dtype=np.int64) for i in range(n_graphs)]
    self_parts += [n_junc_total + boff[i] + np.arange(nb[i], dtype=np.int64) for i in range(n_graphs)]
    self_ids = np.concatenate(self_parts, axis=0) if self_parts else np.zeros(0, np.int64)
    src = np.concatenate([src, self_ids], axis=0)
    dst = np.concatenate([dst, self_ids], axis=0)
    rel = np.concatenate([rel, np.zeros(len(self_ids), np.int64)], axis=0)
    reln = np.concatenate([reln, np.zeros(len(self_ids), np.int64)], axis=0)

    def t(arr, dtype=None):
        return torch.as_tensor(np.ascontiguousarray(arr), dtype=dtype)

    return {
        "n_graphs": n_graphs,
        "atom_feat": t(atom_feat, torch.float32),
        "atom_batch": t(atom_batch, torch.long),
        "bond_index": t(bond_index, torch.long),
        "bond_type": t(bond_type, torch.long),
        "bond_feat": t(bond_feat, torch.float32),
        "bond_batch": t(bond_batch, torch.long),
        "junc_index": t(junc_index, torch.long),
        "junc_bond": t(junc_bond, torch.long),
        "junc_feat": t(junc_feat, torch.float32),
        "junc_batch": t(junc_batch, torch.long),
        "pnode_atom": t(pnode_atom, torch.long),
        "pnode_patch": t(pnode_patch, torch.long),
        "pnode_du": t(pnode_du, torch.long),
        "pnode_dv": t(pnode_dv, torch.long),
        "pedge_x": t(pedge_x, torch.long),
        "pedge_y": t(pedge_y, torch.long),
        "pedge_patch": t(pedge_patch, torch.long),
        "pedge_type": t(pedge_type, torch.long),
        "pedge_du_x": t(pedge_du_x, torch.long),
        "pedge_du_y": t(pedge_du_y, torch.long),
        "pedge_dv_x": t(pedge_dv_x, torch.long),
        "pedge_dv_y": t(pedge_dv_y, torch.long),
        "pedge_root": t(pedge_root, torch.long),
        "p_scalars": t(p_scalars, torch.float32),
        "p_support": t(p_support, torch.float32),
        "patch_batch": t(patch_batch, torch.long),
        "p_root_bond": t(np.arange(n_bonds_total, dtype=np.int64), torch.long),
        "global_feat": t(global_feat, torch.float32),
        "counts": t(counts, torch.float32),
        "n_atoms_total": n_atoms_total,
        "n_bonds_total": n_bonds_total,
        "n_junc_total": n_junc_total,
        "n_patch_total": n_patch_total,
        "n_tokens_total": n_tokens_total,
        "r_src": t(src, torch.long),
        "r_dst": t(dst, torch.long),
        "r_rel": t(rel, torch.long),
        "r_rel_null": t(reln, torch.long),
    }


def move(batch: Mapping[str, Any], device) -> dict[str, Any]:
    import torch

    return {
        k: (v.to(device) if torch.is_tensor(v) else v)
        for k, v in batch.items()
    }

"""Frozen joint structural statistics J_raw and J_latent.

Definitions are in AUDIT_PLAN.md §2.  The implementation is intentionally
small and independent from the model: it consumes only the physical bond table
and, for J_latent, the per-bond c_loc rows of the backbone being audited.

No labels are read here.
"""

from __future__ import annotations

from collections import defaultdict
from typing import Iterable, Mapping, Sequence

import networkx as nx
import numpy as np
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import shortest_path

# Observed raw bond-type vocabulary in the official ZINC-subset train cache.
# There is no aromatic type 4 in this data; see AUDIT_PLAN.md §2.2.
BOND_TYPES = (1, 2, 3)
UNORDERED_PAIRS = ((1, 1), (1, 2), (1, 3), (2, 2), (2, 3), (3, 3))
PAIR_INDEX = {p: i for i, p in enumerate(UNORDERED_PAIRS)}
RELATIONS = ("R1", "R2", "R3", "R4")
N_PAIR_BINS = len(UNORDERED_PAIRS)
N_RAW_DIMS = len(RELATIONS) * (N_PAIR_BINS + 2)  # 32
N_LATENT_DIMS = len(RELATIONS)                # 4
SEED_BASE = 20261026


def _pair_key(a: int, b: int) -> tuple[int, int]:
    return (a, b) if a <= b else (b, a)


def line_graph_relations(bond_index: np.ndarray, n_atoms: int) -> dict[str, list[tuple[int, int]]]:
    """Return R1/R2/R3 unordered physical-bond pairs from the line graph."""
    bi = np.asarray(bond_index, dtype=np.int64)
    n_b = int(bi.shape[0])
    if n_b == 0:
        return {r: [] for r in RELATIONS}

    incident: list[list[int]] = [[] for _ in range(int(n_atoms))]
    for e, (u, v) in enumerate(bi):
        incident[int(u)].append(e)
        incident[int(v)].append(e)

    # Line-graph adjacency: two physical bonds share an atom.
    rows: list[int] = []
    cols: list[int] = []
    for e_list in incident:
        if len(e_list) >= 2:
            for i in range(len(e_list)):
                for j in range(i + 1, len(e_list)):
                    a, b = e_list[i], e_list[j]
                    rows.extend((a, b))
                    cols.extend((b, a))
    adj = csr_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n_b, n_b))
    dist = shortest_path(adj, directed=False, unweighted=True, return_predecessors=False)
    finite = np.isfinite(dist)

    r1, r2, r3 = [], [], []
    for a in range(n_b):
        for b in range(a + 1, n_b):
            if not finite[a, b]:
                continue
            d = int(dist[a, b])
            if d == 1:
                r1.append((a, b))
            elif d == 2:
                r2.append((a, b))
            elif d >= 3:
                r3.append((a, b))
    return {"R1": r1, "R2": r2, "R3": r3, "R4": []}


def ring_block_relation(bond_index: np.ndarray, n_atoms: int) -> list[tuple[int, int]]:
    """R4: unordered bond pairs in a common biconnected block with >= 3 edges."""
    bi = np.asarray(bond_index, dtype=np.int64)
    n_b = int(bi.shape[0])
    if n_b == 0:
        return []
    g = nx.Graph()
    g.add_nodes_from(range(int(n_atoms)))
    edge_to_bond: dict[tuple[int, int], int] = {}
    for e, (u, v) in enumerate(bi):
        a, b = int(u), int(v)
        key = (a, b) if a <= b else (b, a)
        g.add_edge(a, b)
        edge_to_bond[key] = e
    out: list[tuple[int, int]] = []
    for nodes in nx.biconnected_components(g):
        nodes = set(int(x) for x in nodes)
        # edges induced by this block
        block_edges = []
        for (u, v), e in edge_to_bond.items():
            if u in nodes and v in nodes:
                block_edges.append(e)
        if len(block_edges) < 3:
            continue
        block_edges.sort()
        for i in range(len(block_edges)):
            for j in range(i + 1, len(block_edges)):
                out.append((block_edges[i], block_edges[j]))
    out.sort()
    return out


def molecule_relations(bond_index: np.ndarray, n_atoms: int) -> dict[str, list[tuple[int, int]]]:
    rel = line_graph_relations(bond_index, n_atoms)
    rel["R4"] = ring_block_relation(bond_index, n_atoms)
    return rel


def _pair_bin_counts(bond_type: np.ndarray, pairs: Sequence[tuple[int, int]]) -> np.ndarray:
    bt = np.asarray(bond_type, dtype=np.int64)
    counts = np.zeros(N_PAIR_BINS, dtype=np.float64)
    for a, b in pairs:
        ta, tb = int(bt[a]), int(bt[b])
        if ta not in (1, 2, 3) or tb not in (1, 2, 3):
            raise ValueError(f"bond type outside observed vocabulary: {ta}, {tb}")
        counts[PAIR_INDEX[_pair_key(ta, tb)]] += 1.0
    return counts


def j_raw(bond_type: np.ndarray, relations: Mapping[str, Sequence[tuple[int, int]]]) -> np.ndarray:
    """32-d primary J_raw (3 observed bond types, 6 unordered pairs)."""
    out = np.zeros(N_RAW_DIMS, dtype=np.float64)
    for r_i, r in enumerate(RELATIONS):
        pairs = relations.get(r, [])
        n = len(pairs)
        base = r_i * (N_PAIR_BINS + 2)
        if n > 0:
            counts = _pair_bin_counts(bond_type, pairs)
            out[base:base + N_PAIR_BINS] = counts / float(n)
        out[base + N_PAIR_BINS] = np.log1p(float(n))
        out[base + N_PAIR_BINS + 1] = 1.0 if n == 0 else 0.0
    return out


def j_raw_four_slot(bond_type: np.ndarray, relations: Mapping[str, Sequence[tuple[int, int]]]) -> np.ndarray:
    """Compatibility diagnostic only: 4-slot vocabulary, 10 pairs -> 48 dims.

    The fourth slot is always empty in this data, so this vector has 4*4=16
    structurally constant dimensions.  It is never used in any model/gate.
    """
    types = (1, 2, 3, 4)
    pairs4 = [(a, b) for i, a in enumerate(types) for b in types[i:]]
    idx4 = {p: i for i, p in enumerate(pairs4)}
    out = np.zeros(4 * (len(pairs4) + 2), dtype=np.float64)
    for r_i, r in enumerate(RELATIONS):
        ps = relations.get(r, [])
        n = len(ps)
        base = r_i * (len(pairs4) + 2)
        if n > 0:
            counts = np.zeros(len(pairs4), dtype=np.float64)
            for a, b in ps:
                ta, tb = int(bond_type[a]), int(bond_type[b])
                counts[idx4[_pair_key(ta, tb)]] += 1.0
            out[base:base + len(pairs4)] = counts / float(n)
        out[base + len(pairs4)] = np.log1p(float(n))
        out[base + len(pairs4) + 1] = 1.0 if n == 0 else 0.0
    return out


def j_latent(c_loc: np.ndarray, relations: Mapping[str, Sequence[tuple[int, int]]]) -> np.ndarray:
    """4-d J_latent: within-molecule centred content co-variation per relation."""
    h = np.asarray(c_loc, dtype=np.float64)
    if h.ndim != 2:
        raise ValueError("c_loc must be [n_bonds, d]")
    if h.shape[0] == 0:
        return np.zeros(N_LATENT_DIMS, dtype=np.float64)
    h = h - h.mean(axis=0, keepdims=True)
    gram = h @ h.T
    out = np.zeros(N_LATENT_DIMS, dtype=np.float64)
    d = float(h.shape[1])
    for r_i, r in enumerate(RELATIONS):
        pairs = relations.get(r, [])
        n = len(pairs)
        if n == 0:
            continue
        vals = np.fromiter((gram[a, b] for a, b in pairs), dtype=np.float64, count=n)
        out[r_i] = float(vals.mean() / d)
    return out


# ---------------------------------------------------------------------------
# independent float64 references (explicit double loops)
# ---------------------------------------------------------------------------

def reference_j_raw(bond_type: np.ndarray, relations: Mapping[str, Sequence[tuple[int, int]]]) -> np.ndarray:
    bt = np.asarray(bond_type, dtype=np.int64)
    out = np.zeros(N_RAW_DIMS, dtype=np.float64)
    for r_i, r in enumerate(RELATIONS):
        pairs = list(relations.get(r, []))
        n = len(pairs)
        base = r_i * (N_PAIR_BINS + 2)
        if n:
            counts = [0.0] * N_PAIR_BINS
            for a, b in pairs:
                ta, tb = int(bt[a]), int(bt[b])
                for k, (pa, pb) in enumerate(UNORDERED_PAIRS):
                    if (ta == pa and tb == pb) or (ta == pb and tb == pa):
                        counts[k] += 1.0
                        break
            for k in range(N_PAIR_BINS):
                out[base + k] = counts[k] / n
        out[base + N_PAIR_BINS] = np.log1p(n)
        out[base + N_PAIR_BINS + 1] = 1.0 if n == 0 else 0.0
    return out


def reference_j_latent(c_loc: np.ndarray, relations: Mapping[str, Sequence[tuple[int, int]]]) -> np.ndarray:
    h = np.asarray(c_loc, dtype=np.float64)
    if h.shape[0] == 0:
        return np.zeros(N_LATENT_DIMS, dtype=np.float64)
    h = h - h.mean(axis=0, keepdims=True)
    d = float(h.shape[1])
    out = np.zeros(N_LATENT_DIMS, dtype=np.float64)
    for r_i, r in enumerate(RELATIONS):
        pairs = list(relations.get(r, []))
        if not pairs:
            continue
        s = 0.0
        for a, b in pairs:
            s += float(np.dot(h[a], h[b]))
        out[r_i] = s / len(pairs) / d
    return out


# ---------------------------------------------------------------------------
# nulls (fixed seeds)
# ---------------------------------------------------------------------------

def _rng_for(molecule_index: int) -> np.random.Generator:
    return np.random.default_rng(np.random.SeedSequence([SEED_BASE, int(molecule_index)]))


def j_raw_null(bond_type: np.ndarray, relations: Mapping[str, Sequence[tuple[int, int]]],
               molecule_index: int) -> np.ndarray:
    bt = np.asarray(bond_type, dtype=np.int64)
    perm = _rng_for(molecule_index).permutation(len(bt))
    return j_raw(bt[perm], relations)


def j_latent_null(c_loc: np.ndarray, relations: Mapping[str, Sequence[tuple[int, int]]],
                  molecule_index: int) -> np.ndarray:
    h = np.asarray(c_loc, dtype=np.float64)
    perm = _rng_for(molecule_index).permutation(h.shape[0])
    return j_latent(h[perm], relations)


def relation_pair_counts(relations: Mapping[str, Sequence[tuple[int, int]]]) -> np.ndarray:
    return np.asarray([len(relations.get(r, [])) for r in RELATIONS], dtype=np.int64)

"""Exact, memory-lean graph analysis used by the large TU / peptide caches.

The mathematical definitions are the frozen upstream ones in
``lcrc_cross_task_preparation_v1/code/molecular_graph.py``:

* non-backtracking two-hop paths rooted at every node (``u != root``);
* closure = the end node is adjacent to the root;
* per-node distance-2 count, diameter, finite mean pair distance, unreachable
  fraction and finite-pair-distance >= 3 fraction over unordered pairs.

The only difference is representation: the upstream implementation materialises
an ``n x n`` distance matrix.  Here the same BFS distances are accumulated into
histograms, so no ``n^2`` array is stored.  No path is sampled, truncated or
approximated.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

import molecular_graph as MG


@dataclass
class FastAnalysis:
    n_atoms: int
    n_bonds: int
    bond_u: np.ndarray
    bond_v: np.ndarray
    degree: np.ndarray
    adj: tuple[tuple[tuple[int, int, int], ...], ...]
    component_id: np.ndarray
    component_size: np.ndarray
    n_components: int
    n_junctions: int
    is_ring_bond: np.ndarray
    ring_atom: np.ndarray
    ring_bond_count: np.ndarray
    min_ring_size: np.ndarray
    n_ring_systems: int
    n_d2: np.ndarray
    distance: dict[str, float | int]


def _csr(adj: Sequence[Sequence[tuple[int, int, int]]]) -> tuple[np.ndarray, np.ndarray]:
    n = len(adj)
    counts = np.asarray([len(row) for row in adj], dtype=np.int64)
    indptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    out = np.empty(int(indptr[-1]), dtype=np.int64)
    pos = 0
    for row in adj:
        for v, _bt, _e in row:
            out[pos] = int(v)
            pos += 1
    return indptr, out


def distance_summary(
    adj: Sequence[Sequence[tuple[int, int, int]]],
) -> tuple[np.ndarray, dict[str, float | int]]:
    """Exact BFS distance summaries without an n-by-n matrix.

    Returns ``n_d2[s] = #{t : dist(s,t) == 2}`` and the unordered-pair
    statistics used by the frozen 28-field global vector.
    """
    n = len(adj)
    indptr, neighbours = _csr(adj)
    n_d2 = np.zeros(n, dtype=np.int64)
    n_pairs = 0
    n_unreachable = 0
    sum_finite = 0
    n_ge3 = 0
    diameter = 0
    dist = np.empty(n, dtype=np.int32)
    for s in range(n):
        dist.fill(-1)
        dist[s] = 0
        frontier = np.asarray([s], dtype=np.int64)
        level = 0
        while frontier.size:
            lo = indptr[frontier]
            lengths = indptr[frontier + 1] - lo
            if not lengths.sum():
                break
            nbrs = np.concatenate([neighbours[a:b] for a, b in zip(lo, lo + lengths)])
            nbrs = np.unique(nbrs)
            nbrs = nbrs[dist[nbrs] < 0]
            if not nbrs.size:
                break
            level += 1
            dist[nbrs] = level
            frontier = nbrs
        n_d2[s] = int(np.count_nonzero(dist == 2))
        # Count each unordered pair once: target index greater than source.
        targets = np.arange(s + 1, n, dtype=np.int64)
        if targets.size:
            d = dist[targets]
            finite = d >= 0
            n_pairs += int(targets.size)
            n_unreachable += int((~finite).sum())
            if finite.any():
                df = d[finite]
                sum_finite += int(df.sum())
                n_ge3 += int((df >= 3).sum())
                diameter = max(diameter, int(df.max()))
    return n_d2, {
        "n_pairs": int(n_pairs),
        "n_unreachable": int(n_unreachable),
        "sum_finite_distance": int(sum_finite),
        "n_finite_ge3": int(n_ge3),
        "diameter": int(diameter),
        "mean_finite_distance": float(sum_finite / max(n_pairs - n_unreachable, 1)),
        "frac_unreachable": float(n_unreachable / n_pairs) if n_pairs else 0.0,
        "frac_finite_ge3": float(n_ge3 / n_pairs) if n_pairs else 0.0,
    }


def min_ring_size_per_atom_fast(
    n_atoms: int,
    adj: Sequence[Sequence[tuple[int, int, int]]],
    ring_atom: np.ndarray,
) -> np.ndarray:
    """Exact smallest cycle through each ring atom.

    Multi-source BFS from all neighbours of ``a`` with the root removed.  A
    meeting of two different source wavefronts closes a cycle through ``a`` of
    length ``dist[u] + dist[v] + 3``.  The BFS stops once a shorter cycle is
    impossible.  This is the same definition as the upstream pair-of-neighbours
    BFS, only with one BFS per atom instead of one per neighbour pair.
    """
    from collections import deque

    n = int(n_atoms)
    out = np.zeros(n, dtype=np.int64)
    for a in np.flatnonzero(np.asarray(ring_atom, dtype=bool)):
        nbrs = [int(v) for v, _bt, _e in adj[int(a)]]
        if len(nbrs) < 2:
            continue
        dist = np.full(n, -1, dtype=np.int32)
        src = np.full(n, -1, dtype=np.int32)
        queue: deque[int] = deque()
        for k, s in enumerate(nbrs):
            dist[s] = 0
            src[s] = k
            queue.append(s)
        best = 0
        while queue:
            u = int(queue.popleft())
            du = int(dist[u])
            su = int(src[u])
            if best > 0 and du + 3 >= best:
                break
            for v, _bt, _e in adj[u]:
                v = int(v)
                if v == int(a):
                    continue
                if dist[v] == -1:
                    dist[v] = du + 1
                    src[v] = su
                    queue.append(v)
                elif int(src[v]) != su:
                    cand = du + int(dist[v]) + 3
                    if best == 0 or cand < best:
                        best = cand
        out[int(a)] = best
    return out


def analyse_graph_fast(graph: MG.MolecularGraph, *, compute_distance: bool = True) -> FastAnalysis:
    """Semantic equivalent of ``MG.analyse_graph`` with compact distances."""
    n = graph.n_atoms
    m = graph.n_bonds
    if n < 1:
        raise ValueError("empty molecule is not supported")
    ug = np.asarray(graph.bond_u, dtype=np.int64)
    vg = np.asarray(graph.bond_v, dtype=np.int64)
    if np.any(ug >= vg):
        raise ValueError("physical bonds must satisfy u < v")
    if len(ug) > 1 and np.any(np.diff(ug * (n + 1) + vg) <= 0):
        raise ValueError("physical bonds must be sorted by (u, v)")
    adj = MG.adjacency(n, ug, vg, graph.bond_type)
    deg = MG.degrees(n, ug, vg)
    comp, n_comp, comp_size = MG.connected_components(n, ug, vg)
    ring = MG.biconnected_ring_analysis(n, ug, vg)
    min_ring = min_ring_size_per_atom_fast(n, adj, ring["ring_atom"])
    if compute_distance:
        n_d2, dist_stats = distance_summary(adj)
    else:
        n_d2 = np.zeros(n, dtype=np.int64)
        dist_stats = {
            "n_pairs": 0,
            "n_unreachable": 0,
            "sum_finite_distance": 0,
            "n_finite_ge3": 0,
            "diameter": 0,
            "mean_finite_distance": 0.0,
            "frac_unreachable": 0.0,
            "frac_finite_ge3": 0.0,
        }
    return FastAnalysis(
        n_atoms=n,
        n_bonds=m,
        bond_u=ug.copy(),
        bond_v=vg.copy(),
        degree=deg,
        adj=adj,
        component_id=comp,
        component_size=comp_size,
        n_components=int(n_comp),
        n_junctions=MG.junction_count(deg),
        is_ring_bond=ring["is_ring_bond"],
        ring_atom=ring["ring_atom"],
        ring_bond_count=ring["ring_bond_count"],
        min_ring_size=min_ring,
        n_ring_systems=int(ring["n_ring_systems"]),
        n_d2=n_d2,
        distance=dist_stats,
    )


def s1_arrays(atom_type: np.ndarray, adj: Sequence[Sequence[tuple[int, int, int]]], n_bond_types: int) -> tuple[np.ndarray, np.ndarray]:
    counts = np.asarray([len(row) for row in adj], dtype=np.int64)
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    if int(ptr[-1]) == 0:
        return ptr, np.zeros(0, dtype=np.int32)
    codes = np.empty(int(ptr[-1]), dtype=np.int32)
    pos = 0
    for root, row in enumerate(adj):
        for middle, bt, _e in row:
            codes[pos] = int(atom_type[int(middle)]) * int(n_bond_types) + (int(bt) - 1)
            pos += 1
    return ptr, codes


def s2_arrays(
    atom_type: np.ndarray,
    adj: Sequence[Sequence[tuple[int, int, int]]],
    n_bond_types: int,
) -> dict[str, np.ndarray]:
    """Exact ordered non-backtracking two-hop records, per-root concatenated."""
    n = len(adj)
    nb = int(n_bond_types)
    atom_type = np.asarray(atom_type, dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)
    path_cat_parts: list[np.ndarray] = []
    end63_parts: list[np.ndarray] = []
    end_parts: list[np.ndarray] = []
    closure_parts: list[np.ndarray] = []
    for root, row in enumerate(adj):
        if not row:
            continue
        mids = np.fromiter((int(w) for w, _bt, _e in row), dtype=np.int64)
        bt1 = np.fromiter((int(bt) for _w, bt, _e in row), dtype=np.int64)
        ends_chunks: list[np.ndarray] = []
        bt2_chunks: list[np.ndarray] = []
        mids_used: list[int] = []
        bt1_used: list[int] = []
        for middle, b1 in zip(mids.tolist(), bt1.tolist()):
            nbr_w = adj[middle]
            if not nbr_w:
                continue
            e = np.fromiter((int(u) for u, _bt, _e in nbr_w if int(u) != int(root)), dtype=np.int64)
            b = np.fromiter((int(bt) for u, bt, _e in nbr_w if int(u) != int(root)), dtype=np.int64)
            if e.size:
                ends_chunks.append(e)
                bt2_chunks.append(b)
                mids_used.append(middle)
                bt1_used.append(b1)
        if not ends_chunks:
            continue
        ends = np.concatenate(ends_chunks)
        bt2 = np.concatenate(bt2_chunks)
        # Repeat each middle's bond type for all of its ends.
        repeats = np.asarray([len(x) for x in ends_chunks], dtype=np.int64)
        mid_rep = np.repeat(np.asarray(mids_used, dtype=np.int64), repeats)
        bt1_rep = np.repeat(np.asarray(bt1_used, dtype=np.int64), repeats)
        counts[root] = ends.size
        path_cat_parts.append(atom_type[mid_rep] * nb * nb + (bt1_rep - 1) * nb + (bt2 - 1))
        end63_parts.append(atom_type[ends] * nb + (bt2 - 1))
        end_parts.append(ends)
        nbrs_root = np.asarray(sorted(int(w) for w, _bt, _e in row), dtype=np.int64)
        closure_parts.append(np.isin(ends, nbrs_root).astype(np.int8))
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    if int(ptr[-1]) == 0:
        return {
            "path_ptr": ptr,
            "path_cat": np.zeros(0, dtype=np.int32),
            "path_end63": np.zeros(0, dtype=np.int32),
            "path_end": np.zeros(0, dtype=np.int64),
            "path_closure": np.zeros(0, dtype=np.int8),
            "per_root_counts": counts,
        }
    return {
        "path_ptr": ptr,
        "path_cat": np.concatenate(path_cat_parts).astype(np.int32),
        "path_end63": np.concatenate(end63_parts).astype(np.int32),
        "path_end": np.concatenate(end_parts).astype(np.int64),
        "path_closure": np.concatenate(closure_parts).astype(np.int8),
        "per_root_counts": counts,
    }


def s2_counts_from_formula(adj: Sequence[Sequence[tuple[int, int, int]]]) -> np.ndarray:
    """Exact S2 record count per root: ``sum_{w in N(v)} (deg(w) - 1)``."""
    out = np.zeros(len(adj), dtype=np.int64)
    for root, row in enumerate(adj):
        out[root] = sum(max(len(adj[int(w)]) - 1, 0) for w, _bt, _e in row)
    return out


def compare_with_upstream(
    graph: MG.MolecularGraph,
    fast: FastAnalysis,
    upstream: MG.GraphAnalysis,
    spec_n_bond_types: int,
) -> dict[str, Any]:
    """Exact equality checks used by the CPU gates."""
    s2u = MG.enumerate_s2(graph.atom_type, upstream.adj, spec_n_bond_types, upstream.dist)
    counts_u = np.asarray([len(r) for r in s2u], dtype=np.int64)
    counts_f = s2_counts_from_formula(fast.adj)
    n_d2_u = np.sum(np.asarray(upstream.dist) == 2, axis=1).astype(np.int64)
    n = fast.n_atoms
    pair = np.asarray(upstream.dist)[np.triu_indices(n, 1)] if n > 1 else np.zeros(0, dtype=np.int64)
    finite = pair >= 0
    finite_dist = pair[finite]
    stats_u = {
        "n_pairs": int(pair.size),
        "n_unreachable": int((~finite).sum()),
        "sum_finite_distance": int(finite_dist.sum()),
        "n_finite_ge3": int((finite_dist >= 3).sum()),
        "diameter": int(finite_dist.max()) if finite_dist.size else 0,
    }
    return {
        "s2_counts_equal": bool(np.array_equal(counts_u, counts_f)),
        "n_d2_equal": bool(np.array_equal(n_d2_u, fast.n_d2)),
        "distance_stats_equal": all(fast.distance[k] == v for k, v in stats_u.items()),
        "upstream_s2_total": int(counts_u.sum()),
        "fast_s2_total": int(counts_f.sum()),
        "upstream_stats": stats_u,
        "fast_stats": {k: fast.distance[k] for k in stats_u},
    }

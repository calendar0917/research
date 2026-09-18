"""Target-free support-chart combinatorics for Binding Composition (BCE).

This module builds, for every existing rooted radius-2 patch, the exhaustive
chart of

* **connected induced node subsets** ``S`` with ``1 <= |S| <= MAX_SUPPORT_SIZE``
  (the "supports"), and
* every **legal unordered parent decomposition** ``(A, B)`` of each support:
  ``A`, ``B`` are themselves supports of the chart, ``A union B = S``,
  ``|A| < |S|``, ``|B| < |S|``, and the pair shares at least one atom
  (``A ∩ B != ∅``) or has a real crossing bond between ``A\\B`` and ``B\\A``.

Everything here is pure combinatorics on the *existing* patch connectivity and
the root flag.  No atom/bond attribute, no certificate, no target, no learned
parameter and no vocabulary is involved, so the result is safe to precompute and
cache.

Design constraints (from the BCE brief / correctness requirements):

* a support is keyed by its **unordered node set** only -- the induced edge set
  is always ``original patch induced bonds on S``;
* a decomposition is an **unordered parent pair** -- ``(A, B)`` and ``(B, A)``
  are the same decomposition and are stored once;
* **no** "first decomposition wins", no lexicographic parent that decides
  semantics, no node-id-dependent behaviour.  Enumeration order only fixes a
  deterministic storage order; the learned composition operator is symmetric,
  so the order carries no semantics.
* crossing bonds are computed **set-theoretically** from support membership and
  the original real edge set (never by comparing tuple schemas).

The output is a flat, CSR/ragged-array representation so it can be attached to
PyG ``Data`` objects and re-offset during collation.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations

import numpy as np

MAX_SUPPORT_SIZE = 4


@dataclass(frozen=True)
class SupportChart:
    """Flat per-graph support chart (graph-local node / bond indices)."""

    sup_size: np.ndarray  # [P] int64 support size (1..MAX_SUPPORT_SIZE)
    sup_root: np.ndarray  # [P] int64 1 if the support contains the root
    sup_patch: np.ndarray  # [P] int64 owning patch (graph-local patch index)
    sup_single_node: np.ndarray  # [P] int64 node index for size-1, else -1
    sup_single_bond: np.ndarray  # [P] int64 bond index for size-2, else -1
    sup_node_ptr: np.ndarray  # [P+1] int64 CSR -> node indices
    sup_node_idx: np.ndarray  # [nnz] int64 graph-local node indices
    sup_edge_ptr: np.ndarray  # [P+1] int64 CSR -> induced bond indices
    sup_edge_idx: np.ndarray  # [nnz] int64 graph-local bond indices
    dec_child: np.ndarray  # [D] int64 child support index
    dec_parent_a: np.ndarray  # [D] int64 parent support index
    dec_parent_b: np.ndarray  # [D] int64 parent support index
    dec_overlap_ptr: np.ndarray  # [D+1] int64 CSR -> overlap node indices
    dec_overlap_idx: np.ndarray  # [nnz] int64 graph-local node indices
    dec_cross_ptr: np.ndarray  # [D+1] int64 CSR -> crossing bond indices
    dec_cross_idx: np.ndarray  # [nnz] int64 graph-local bond indices
    n_supports: int
    n_decompositions: int


def _connected_subsets(
    n_local: int, adj: list[set[int]], max_size: int
) -> list[frozenset[int]]:
    """All connected induced subsets of ``{0..n_local-1}`` with size <= max."""
    singleton = [frozenset((v,)) for v in range(n_local)]
    found: set[frozenset[int]] = set(singleton)
    frontier = set(singleton)
    for _size in range(2, int(max_size) + 1):
        new: set[frozenset[int]] = set()
        for subset in frontier:
            neighbours: set[int] = set()
            for node in subset:
                neighbours |= adj[node]
            for extra in neighbours - subset:
                new.add(subset | {extra})
        found |= new
        frontier = new
        if not frontier:
            break
    return sorted(found, key=lambda s: (len(s), tuple(sorted(s))))


def _subset_combinations(nodes: tuple[int, ...]):
    """All subsets of ``nodes`` (including the empty set), ascending size."""
    for size in range(len(nodes) + 1):
        yield from combinations(nodes, size)


def _induced_edges(
    support: frozenset[int], adj: list[set[int]]
) -> list[tuple[int, int]]:
    """Original induced real edges with both endpoints inside ``support``."""
    edges: list[tuple[int, int]] = []
    for u in sorted(support):
        for v in sorted(adj[u]):
            if v in support and u < v:
                edges.append((u, v))
    return sorted(edges)


def _crossing_edges(
    left: frozenset[int], right: frozenset[int], adj: list[set[int]]
) -> list[tuple[int, int]]:
    """Real bonds between ``left`` and ``right`` (set-theoretic membership)."""
    edges: list[tuple[int, int]] = []
    for u in sorted(left):
        for v in sorted(adj[u]):
            if v in right:
                edges.append((u, v) if u < v else (v, u))
    return sorted(set(edges))


def build_chart(
    *,
    n_nodes: int,
    patch_of_node: np.ndarray,
    n_patches: int,
    edge_u: np.ndarray,
    edge_v: np.ndarray,
    edge_patch: np.ndarray,
    root_of_node: np.ndarray,
    max_support_size: int = MAX_SUPPORT_SIZE,
) -> SupportChart:
    """Build the support chart for one molecule.

    ``edge_u`` / ``edge_v`` list every **undirected** real bond exactly once
    (``u < v``) in the same order the encoder uses; ``edge_patch`` is the owning
    patch of each bond.  ``root_of_node`` marks the patch root.
    """
    n_nodes = int(n_nodes)
    n_patches = int(n_patches)
    max_support_size = int(max_support_size)
    if max_support_size < 1:
        raise ValueError("max_support_size must be >= 1")

    adj: list[set[int]] = [set() for _ in range(n_nodes)]
    for u, v in zip(edge_u.tolist(), edge_v.tolist()):
        adj[int(u)].add(int(v))
        adj[int(v)].add(int(u))
    bond_index = {
        (int(min(u, v)), int(max(u, v))): index
        for index, (u, v) in enumerate(zip(edge_u.tolist(), edge_v.tolist()))
    }

    nodes_by_patch: list[list[int]] = [[] for _ in range(n_patches)]
    for node, patch in enumerate(patch_of_node.tolist()):
        nodes_by_patch[int(patch)].append(int(node))
    for nodes in nodes_by_patch:
        nodes.sort()

    sup_size: list[int] = []
    sup_root: list[int] = []
    sup_patch: list[int] = []
    sup_single_node: list[int] = []
    sup_single_bond: list[int] = []
    sup_node_idx: list[int] = []
    sup_edge_idx: list[int] = []
    sup_node_ptr: list[int] = [0]
    sup_edge_ptr: list[int] = [0]

    dec_child: list[int] = []
    dec_parent_a: list[int] = []
    dec_parent_b: list[int] = []
    dec_overlap_idx: list[int] = []
    dec_cross_idx: list[int] = []
    dec_overlap_ptr: list[int] = [0]
    dec_cross_ptr: list[int] = [0]

    for patch in range(n_patches):
        patch_nodes = nodes_by_patch[patch]
        local_of = {node: index for index, node in enumerate(patch_nodes)}
        local_adj = [
            {local_of[v] for v in adj[node] if v in local_of}
            for node in patch_nodes
        ]
        local_root = [
            index
            for index, node in enumerate(patch_nodes)
            if bool(root_of_node[node])
        ]
        if len(local_root) != 1:
            raise RuntimeError(
                f"patch {patch} must have exactly one root, got {len(local_root)}"
            )
        root_local = int(local_root[0])
        subsets = _connected_subsets(len(patch_nodes), local_adj, max_support_size)
        support_index: dict[frozenset[int], int] = {
            subset: index for index, subset in enumerate(subsets)
        }
        # size-ordered processing so parents always exist before children.
        for subset in subsets:
            index = support_index[subset]
            size = len(subset)
            nodes = sorted(subset)
            sup_size.append(size)
            sup_root.append(1 if root_local in subset else 0)
            sup_patch.append(patch)
            sup_single_node.append(patch_nodes[nodes[0]] if size == 1 else -1)
            bond_for_single = -1
            if size == 2:
                pair = (patch_nodes[nodes[0]], patch_nodes[nodes[1]])
                bond_for_single = bond_index[pair]
            sup_single_bond.append(bond_for_single)
            for node in nodes:
                sup_node_idx.append(patch_nodes[node])
            sup_node_ptr.append(len(sup_node_idx))
            for left, right in _induced_edges(subset, local_adj):
                sup_edge_idx.append(
                    bond_index[(patch_nodes[left], patch_nodes[right])]
                )
            sup_edge_ptr.append(len(sup_edge_idx))

        # --- legal unordered parent decompositions -------------------------
        for subset in subsets:
            size = len(subset)
            if size < 3:
                continue
            child = support_index[subset]
            nodes = sorted(subset)
            seen: set[tuple[frozenset[int], frozenset[int]]] = set()
            for left in _subset_combinations(nodes):
                a_local = frozenset(left)
                if not a_local or a_local == subset:
                    continue
                if a_local not in support_index:
                    continue  # A must itself be a connected chart support
                rest = subset - a_local
                if not rest:
                    continue
                for extra in _subset_combinations(tuple(sorted(a_local))):
                    c_local = frozenset(extra)
                    if c_local == a_local:
                        continue  # B == S is forbidden
                    b_local = rest | c_local
                    if not b_local or len(b_local) >= size:
                        continue
                    if b_local not in support_index:
                        continue  # B must itself be a connected chart support
                    key = (
                        (a_local, b_local)
                        if tuple(sorted(a_local)) <= tuple(sorted(b_local))
                        else (b_local, a_local)
                    )
                    if key in seen:
                        continue
                    first, second = key
                    overlap = first & second
                    left_only = first - second
                    right_only = second - first
                    crossing = (
                        _crossing_edges(left_only, right_only, local_adj)
                        if left_only and right_only
                        else []
                    )
                    if not overlap and not crossing:
                        continue
                    seen.add(key)
                    dec_child.append(child)
                    dec_parent_a.append(support_index[first])
                    dec_parent_b.append(support_index[second])
                    for node in sorted(overlap):
                        dec_overlap_idx.append(patch_nodes[node])
                    dec_overlap_ptr.append(len(dec_overlap_idx))
                    for left_node, right_node in crossing:
                        dec_cross_idx.append(
                            bond_index[
                                (patch_nodes[left_node], patch_nodes[right_node])
                            ]
                        )
                    dec_cross_ptr.append(len(dec_cross_idx))

    return SupportChart(
        sup_size=np.asarray(sup_size, dtype=np.int64),
        sup_root=np.asarray(sup_root, dtype=np.int64),
        sup_patch=np.asarray(sup_patch, dtype=np.int64),
        sup_single_node=np.asarray(sup_single_node, dtype=np.int64),
        sup_single_bond=np.asarray(sup_single_bond, dtype=np.int64),
        sup_node_ptr=np.asarray(sup_node_ptr, dtype=np.int64),
        sup_node_idx=np.asarray(sup_node_idx, dtype=np.int64),
        sup_edge_ptr=np.asarray(sup_edge_ptr, dtype=np.int64),
        sup_edge_idx=np.asarray(sup_edge_idx, dtype=np.int64),
        dec_child=np.asarray(dec_child, dtype=np.int64),
        dec_parent_a=np.asarray(dec_parent_a, dtype=np.int64),
        dec_parent_b=np.asarray(dec_parent_b, dtype=np.int64),
        dec_overlap_ptr=np.asarray(dec_overlap_ptr, dtype=np.int64),
        dec_overlap_idx=np.asarray(dec_overlap_idx, dtype=np.int64),
        dec_cross_ptr=np.asarray(dec_cross_ptr, dtype=np.int64),
        dec_cross_idx=np.asarray(dec_cross_idx, dtype=np.int64),
        n_supports=len(sup_size),
        n_decompositions=len(dec_child),
    )


def chart_to_plain(chart: SupportChart) -> dict[str, np.ndarray | int]:
    """Serialise a chart so the cache is import-safe (no dataclass pickling)."""
    return {
        "sup_size": chart.sup_size,
        "sup_root": chart.sup_root,
        "sup_patch": chart.sup_patch,
        "sup_single_node": chart.sup_single_node,
        "sup_single_bond": chart.sup_single_bond,
        "sup_node_ptr": chart.sup_node_ptr,
        "sup_node_idx": chart.sup_node_idx,
        "sup_edge_ptr": chart.sup_edge_ptr,
        "sup_edge_idx": chart.sup_edge_idx,
        "dec_child": chart.dec_child,
        "dec_parent_a": chart.dec_parent_a,
        "dec_parent_b": chart.dec_parent_b,
        "dec_overlap_ptr": chart.dec_overlap_ptr,
        "dec_overlap_idx": chart.dec_overlap_idx,
        "dec_cross_ptr": chart.dec_cross_ptr,
        "dec_cross_idx": chart.dec_cross_idx,
        "n_supports": int(chart.n_supports),
        "n_decompositions": int(chart.n_decompositions),
    }


def chart_from_plain(item: dict) -> SupportChart:
    fields = (
        "sup_size",
        "sup_root",
        "sup_patch",
        "sup_single_node",
        "sup_single_bond",
        "sup_node_ptr",
        "sup_node_idx",
        "sup_edge_ptr",
        "sup_edge_idx",
        "dec_child",
        "dec_parent_a",
        "dec_parent_b",
        "dec_overlap_ptr",
        "dec_overlap_idx",
        "dec_cross_ptr",
        "dec_cross_idx",
    )
    return SupportChart(
        **{name: np.asarray(item[name], dtype=np.int64) for name in fields},
        n_supports=int(item["n_supports"]),
        n_decompositions=int(item["n_decompositions"]),
    )


def brute_force_decompositions(
    support: frozenset[int],
    supports: set[frozenset[int]],
    adj: list[set[int]],
) -> set[tuple[frozenset[int], frozenset[int]]]:
    """Independent reference implementation used by the correctness tests.

    Deliberately written differently from :func:`build_chart` (explicit
    membership checks, explicit crossing set) so the two agree only if the
    semantics are right.
    """
    out: set[tuple[frozenset[int], frozenset[int]]] = set()
    ordered = sorted(support)
    for size_a in range(1, len(ordered)):
        for combo_a in combinations(ordered, size_a):
            a = frozenset(combo_a)
            if a not in supports:
                continue
            for size_b in range(1, len(ordered)):
                if size_a + size_b < len(ordered):
                    # union can never equal S
                    continue
                for combo_b in combinations(ordered, size_b):
                    b = frozenset(combo_b)
                    if b not in supports:
                        continue
                    if a | b != support:
                        continue
                    if a == support or b == support:
                        continue
                    overlap = a & b
                    left = a - b
                    right = b - a
                    crossing = {
                        (min(u, v), max(u, v))
                        for u in left
                        for v in adj[u]
                        if v in right
                    }
                    if not overlap and not crossing:
                        continue
                    out.add((a, b) if tuple(sorted(a)) <= tuple(sorted(b)) else (b, a))
    return out

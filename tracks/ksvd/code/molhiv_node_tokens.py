"""Deterministic atom-centered chemical patches for localized KSVD tokens."""

from __future__ import annotations

import hashlib
from collections import deque
from typing import Iterable

import numpy as np
from ogb.utils.features import get_atom_feature_dims

from .graph import Graph
from .vectorize import labeled_wl_ring_patch_features


def ego_nodes(g: Graph, center: int, radius: int = 2) -> tuple[set[int], dict[int, int]]:
    """Return the exact radius-r induced ego set and center distances."""
    if radius < 0:
        raise ValueError("radius must be non-negative")
    dist = {int(center): 0}
    q: deque[int] = deque([int(center)])
    while q:
        u = q.popleft()
        if dist[u] >= radius:
            continue
        for v in g.neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return set(dist), dist


def _stable_bin(token: object, n_bins: int, namespace: str) -> int:
    payload = (namespace + "|" + repr(token)).encode("utf-8")
    digest = hashlib.blake2b(payload, digest_size=8).digest()
    return int.from_bytes(digest, "little") % n_bins


def _center_one_hot(node_feat: np.ndarray, center: int) -> np.ndarray:
    dims = get_atom_feature_dims()
    out = np.zeros(sum(dims), dtype=np.float64)
    offset = 0
    row = np.asarray(node_feat[center], dtype=np.int64).reshape(-1)
    if row.size != len(dims):
        raise ValueError(f"expected {len(dims)} atom channels, got {row.size}")
    for value, width in zip(row, dims):
        if value < 0 or value >= width:
            raise ValueError(f"atom feature value {value} outside [0, {width})")
        out[offset + int(value)] = 1.0
        offset += width
    return out


def centered_ego_vector(
    g: Graph,
    center: int,
    node_feat: np.ndarray,
    edge_feat: dict[tuple[int, int], np.ndarray],
    radius: int = 2,
    max_nodes: int = 8,
    shell_atom_bins: int = 32,
    shell_bond_bins: int = 16,
) -> np.ndarray:
    """Permutation-invariant local chemistry with an explicitly marked center.

    The historical ``wl_chem_ring`` vector summarizes the induced ego graph.
    We append the full OGB center atom one-hot plus distance-shell atom/bond
    histograms, so two different centers in the same induced subgraph do not
    collapse to the same token merely because the unrooted patch is isomorphic.
    """
    nodes, dist = ego_nodes(g, center, radius=radius)
    base = np.asarray(
        labeled_wl_ring_patch_features(
            g,
            nodes,
            max_nodes=max_nodes,
            node_feat=node_feat,
            edge_feat=edge_feat,
        ),
        dtype=np.float64,
    )
    center_hot = _center_one_hot(node_feat, center)

    shell_atoms = np.zeros((radius + 1, shell_atom_bins), dtype=np.float64)
    shell_counts = np.zeros(radius + 1, dtype=np.float64)
    for u in nodes:
        d = min(dist[u], radius)
        label = tuple(int(x) for x in np.asarray(node_feat[u]).reshape(-1))
        shell_atoms[d, _stable_bin(label, shell_atom_bins, f"ego-atom-r{d}")] += 1.0
        shell_counts[d] += 1.0
    for d in range(radius + 1):
        if shell_counts[d] > 0:
            shell_atoms[d] /= shell_counts[d]

    # Place an edge into the outermost shell touched by that edge.
    shell_bonds = np.zeros((radius + 1, shell_bond_bins), dtype=np.float64)
    shell_edge_counts = np.zeros(radius + 1, dtype=np.float64)
    for u, v in g.induced(nodes).edges():
        d = min(max(dist[u], dist[v]), radius)
        key = (u, v) if u < v else (v, u)
        label = tuple(int(x) for x in np.asarray(edge_feat[key]).reshape(-1))
        shell_bonds[d, _stable_bin(label, shell_bond_bins, f"ego-bond-r{d}")] += 1.0
        shell_edge_counts[d] += 1.0
    for d in range(radius + 1):
        if shell_edge_counts[d] > 0:
            shell_bonds[d] /= shell_edge_counts[d]

    # Counts retain patch scale after the per-shell histograms are normalized.
    count_scale = max(1.0, float(max_nodes))
    counts = np.concatenate([shell_counts, shell_edge_counts]) / count_scale
    return np.concatenate(
        [base, center_hot, shell_atoms.reshape(-1), shell_bonds.reshape(-1), counts]
    )


def iter_centered_vectors(
    graphs: list[Graph],
    indices: Iterable[int],
    node_feats: list[np.ndarray],
    edge_feats: list[dict[tuple[int, int], np.ndarray]],
    radius: int = 2,
    max_nodes: int = 8,
):
    """Yield ``(graph_index, node_index, vector)`` deterministically."""
    for raw_i in indices:
        i = int(raw_i)
        g = graphs[i]
        for u in g.nodes:
            yield i, int(u), centered_ego_vector(
                g,
                int(u),
                node_feats[i],
                edge_feats[i],
                radius=radius,
                max_nodes=max_nodes,
            )


def graph_node_offsets(graphs: list[Graph]) -> np.ndarray:
    offsets = np.zeros(len(graphs) + 1, dtype=np.int64)
    offsets[1:] = np.cumsum([g.n for g in graphs], dtype=np.int64)
    return offsets

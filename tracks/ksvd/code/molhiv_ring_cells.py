"""Explicit chordless ring cells and chemistry vectors for MolHIV.

The implementation deliberately avoids RDKit.  Rings are chordless graph cycles
of size 3--6, canonicalized deterministically.  Their feature vectors combine
permutation-invariant in-ring chemistry with the one-hop exocyclic environment.
"""

from __future__ import annotations

from typing import Iterable

import numpy as np
from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

from .graph import Graph
from .vectorize import labeled_wl_ring_patch_features


def chordless_cycles(
    g: Graph,
    min_size: int = 3,
    max_size: int = 6,
) -> list[tuple[int, ...]]:
    """Enumerate canonical chordless cycles with bounded size.

    The smallest node is fixed as the DFS root, eliminating rotations.  The two
    traversal directions are then canonicalized, eliminating reflections.
    A cycle is chordless iff its induced subgraph contains exactly |C| edges.
    """
    if min_size < 3 or max_size < min_size:
        raise ValueError("expected 3 <= min_size <= max_size")
    found: set[tuple[int, ...]] = set()
    for start in g.nodes:
        for first in sorted(v for v in g.neighbors(start) if v > start):
            path = [start, first]
            used = {start, first}

            def visit(u: int) -> None:
                if len(path) > max_size:
                    return
                for v in sorted(g.neighbors(u)):
                    if v == start:
                        if len(path) < min_size:
                            continue
                        forward = tuple(path)
                        reverse = (start,) + tuple(reversed(path[1:]))
                        cyc = min(forward, reverse)
                        S = set(cyc)
                        if len(S) == len(cyc) and g.induced(S).num_edges() == len(cyc):
                            found.add(cyc)
                        continue
                    # start must be the unique minimum node in this cycle.
                    if v <= start or v in used or len(path) >= max_size:
                        continue
                    used.add(v)
                    path.append(v)
                    visit(v)
                    path.pop()
                    used.remove(v)

            visit(first)
    return sorted(found, key=lambda c: (len(c), c))


def _categorical_hist(rows: Iterable[np.ndarray], dims: list[int]) -> np.ndarray:
    out = np.zeros(sum(dims), dtype=np.float64)
    count = 0
    for raw in rows:
        row = np.asarray(raw, dtype=np.int64).reshape(-1)
        if row.size != len(dims):
            raise ValueError(f"categorical row has {row.size} channels, expected {len(dims)}")
        offset = 0
        for value, width in zip(row, dims):
            if value < 0 or value >= width:
                raise ValueError(f"categorical value {value} outside [0, {width})")
            out[offset + int(value)] += 1.0
            offset += width
        count += 1
    if count:
        out /= float(count)
    return out


def ring_cell_vector(
    g: Graph,
    cycle: tuple[int, ...],
    node_feat: np.ndarray,
    edge_feat: dict[tuple[int, int], np.ndarray],
    min_size: int = 3,
    max_size: int = 6,
) -> np.ndarray:
    """Fixed-width chemistry vector for one explicit ring cell.

    Besides the exact induced ring, this includes atom/bond histograms inside
    the ring and on outgoing substituent bonds.  Thus benzene, heterocycles and
    differently substituted/fused rings need not collapse to one descriptor.
    """
    if not min_size <= len(cycle) <= max_size:
        raise ValueError("cycle size outside configured range")
    S = set(cycle)
    sub = g.induced(S)
    if sub.num_edges() != len(cycle):
        raise ValueError("ring_cell_vector requires a chordless cycle")

    atom_dims = get_atom_feature_dims()
    bond_dims = get_bond_feature_dims()
    base = np.asarray(
        labeled_wl_ring_patch_features(
            g, S, max_nodes=max_size, node_feat=node_feat, edge_feat=edge_feat
        ),
        dtype=np.float64,
    )
    size_hot = np.zeros(max_size - min_size + 1, dtype=np.float64)
    size_hot[len(cycle) - min_size] = 1.0
    ring_atoms = _categorical_hist((node_feat[u] for u in cycle), atom_dims)
    ring_bonds = _categorical_hist(
        (edge_feat[g.edge_key(u, v)] for u, v in sub.edges()), bond_dims
    )

    external_edges: list[tuple[int, int]] = []
    external_nodes: list[int] = []
    external_degree = np.zeros(5, dtype=np.float64)  # 0,1,2,3,>=4 substituents
    for u in cycle:
        outside = sorted(v for v in g.neighbors(u) if v not in S)
        external_degree[min(len(outside), 4)] += 1.0
        for v in outside:
            external_edges.append((u, v))
            external_nodes.append(v)
    external_degree /= max(1.0, float(len(cycle)))
    external_atoms = _categorical_hist((node_feat[v] for v in external_nodes), atom_dims)
    external_bonds = _categorical_hist(
        (edge_feat[g.edge_key(u, v)] for u, v in external_edges), bond_dims
    )

    counts = np.asarray(
        [
            len(cycle) / float(max_size),
            len(external_edges) / max(1.0, 2.0 * len(cycle)),
            len(set(external_nodes)) / max(1.0, 2.0 * len(cycle)),
            float(sum(int(node_feat[u, 7]) == 1 for u in cycle)) / len(cycle),
        ],
        dtype=np.float64,
    )
    return np.concatenate(
        [
            base,
            size_hot,
            ring_atoms,
            ring_bonds,
            external_atoms,
            external_bonds,
            external_degree,
            counts,
        ]
    )

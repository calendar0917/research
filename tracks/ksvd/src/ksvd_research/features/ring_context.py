"""Label-free ring-context indexing for graph patch readouts."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ksvd_research.core import Graph


RING_CONTEXT_NAMES = (
    "ring_any",
    "ring5",
    "ring6",
    "aromatic_ring",
    "multi_ring",
    "ring_boundary",
)


def chordless_cycles(
    graph: Graph,
    min_size: int = 3,
    max_size: int = 8,
) -> list[tuple[int, ...]]:
    """Enumerate canonical chordless cycles within a bounded size range."""
    if min_size < 3 or max_size < min_size:
        raise ValueError("expected 3 <= min_size <= max_size")
    found: set[tuple[int, ...]] = set()
    for start in graph.nodes:
        for first in sorted(node for node in graph.neighbors(start) if node > start):
            path = [start, first]
            used = {start, first}

            def visit(current: int) -> None:
                if len(path) > max_size:
                    return
                for nxt in sorted(graph.neighbors(current)):
                    if nxt == start:
                        if len(path) < min_size:
                            continue
                        forward = tuple(path)
                        reverse = (start,) + tuple(reversed(path[1:]))
                        cycle = min(forward, reverse)
                        nodes = set(cycle)
                        if (
                            len(nodes) == len(cycle)
                            and graph.induced(nodes).num_edges() == len(cycle)
                        ):
                            found.add(cycle)
                        continue
                    if nxt <= start or nxt in used or len(path) >= max_size:
                        continue
                    used.add(nxt)
                    path.append(nxt)
                    visit(nxt)
                    path.pop()
                    used.remove(nxt)

            visit(first)
    return sorted(found, key=lambda cycle: (len(cycle), cycle))


@dataclass(frozen=True)
class RingContextIndex:
    """Precomputed graph-level sets used to classify sampled patches."""

    ring_nodes: frozenset[int]
    ring5_nodes: frozenset[int]
    ring6_nodes: frozenset[int]
    aromatic_ring_nodes: frozenset[int]
    multi_ring_nodes: frozenset[int]
    boundary_edges: frozenset[tuple[int, int]]

    def patch_mask(self, nodes: set[int]) -> np.ndarray:
        """Return the six mentor-named context indicators for one patch."""
        node_set = set(nodes)
        boundary = any(
            left in node_set and right in node_set for left, right in self.boundary_edges
        )
        return np.asarray(
            [
                bool(node_set & self.ring_nodes),
                bool(node_set & self.ring5_nodes),
                bool(node_set & self.ring6_nodes),
                bool(node_set & self.aromatic_ring_nodes),
                bool(node_set & self.multi_ring_nodes),
                boundary,
            ],
            dtype=bool,
        )


def build_ring_context_index(graph: Graph, node_features: np.ndarray) -> RingContextIndex:
    """Build a deterministic proxy for the mentor's six ring contexts.

    OGB's atom-level ring/aromatic flags are used when available. Exact ring5,
    ring6 and multi-ring memberships come from chordless cycles of size 3--8.
    The precise mentor definitions remain unknown, so callers must label this
    block as a proxy rather than an exact feature reproduction.
    """
    cycles = chordless_cycles(graph)
    membership = {node: 0 for node in graph.nodes}
    ring5: set[int] = set()
    ring6: set[int] = set()
    cycle_nodes: set[int] = set()
    for cycle in cycles:
        cycle_nodes.update(cycle)
        if len(cycle) == 5:
            ring5.update(cycle)
        if len(cycle) == 6:
            ring6.update(cycle)
        for node in cycle:
            membership[node] += 1

    ogb_ring = {
        node
        for node in graph.nodes
        if node < node_features.shape[0]
        and node_features.shape[1] > 8
        and int(node_features[node, 8]) == 1
    }
    aromatic = {
        node
        for node in graph.nodes
        if node < node_features.shape[0]
        and node_features.shape[1] > 7
        and int(node_features[node, 7]) == 1
    }
    ring_nodes = cycle_nodes | ogb_ring
    multi_ring = {node for node, count in membership.items() if count >= 2}
    boundary_edges = {
        graph.edge_key(left, right)
        for left, right in graph.edges()
        if (left in ring_nodes) != (right in ring_nodes)
    }
    return RingContextIndex(
        ring_nodes=frozenset(ring_nodes),
        ring5_nodes=frozenset(ring5),
        ring6_nodes=frozenset(ring6),
        aromatic_ring_nodes=frozenset(ring_nodes & aromatic),
        multi_ring_nodes=frozenset(multi_ring),
        boundary_edges=frozenset(boundary_edges),
    )

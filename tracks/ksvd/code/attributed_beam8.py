"""Attributed stable ordering and exact typed canonical slots for Beam8."""
from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from typing import Sequence

import numpy as np

from .canonical_slots import CanonicalOrderResult


def _nauty_attributed_order(
    typed_adjacency: np.ndarray,
    node_types: Sequence[int],
    *,
    root: int | None = None,
) -> tuple[int, ...]:
    """Canonical original-node order via a colored incidence graph.

    Original vertices are colored by atom type. Every undirected typed edge is
    replaced by an incidence vertex colored by bond type. A supplied root gets
    its own singleton color class.
    """
    import pynauty

    typed = np.asarray(typed_adjacency, dtype=np.int16)
    types = np.asarray(node_types, dtype=np.int64)
    n = len(types)
    if typed.shape != (n, n) or not np.array_equal(typed, typed.T):
        raise ValueError("typed adjacency and node types are incompatible")
    left, right = np.nonzero(np.triu(typed, k=1) > 0)
    edges = [(int(u), int(v), int(typed[u, v])) for u, v in zip(left, right)]
    adjacency: dict[int, list[int]] = {node: [] for node in range(n + len(edges))}
    for edge_index, (u, v, _edge_type) in enumerate(edges):
        edge_node = n + edge_index
        adjacency[u].append(edge_node)
        adjacency[v].append(edge_node)
        adjacency[edge_node] = [u, v]
    coloring: list[set[int]] = []
    if root is not None:
        coloring.append({int(root)})
    for node_type in sorted(set(int(value) for value in types)):
        cell = {
            node
            for node in range(n)
            if int(types[node]) == node_type and node != root
        }
        if cell:
            coloring.append(cell)
    for edge_type in sorted(set(edge_type for _u, _v, edge_type in edges)):
        coloring.append(
            {
                n + edge_index
                for edge_index, (_u, _v, value) in enumerate(edges)
                if value == edge_type
            }
        )
    graph = pynauty.Graph(
        number_of_vertices=n + len(edges),
        directed=False,
        adjacency_dict=adjacency,
        vertex_coloring=coloring,
    )
    canonical_all = tuple(int(value) for value in pynauty.canon_label(graph))
    original = tuple(value for value in canonical_all if value < n)
    if len(original) != n:
        raise RuntimeError("nauty canonical order lost original vertices")
    return original


def _digest(parts: Sequence[bytes]) -> bytes:
    value = sha256()
    for part in parts:
        value.update(len(part).to_bytes(4, "big"))
        value.update(part)
    return value.digest()


def _initial_color(node_type: int, rooted: bool) -> bytes:
    return _digest((b"attributed-wl-v1", int(node_type).to_bytes(4, "big"), bytes((int(rooted),))))


def _refine_wl(
    typed_adjacency: np.ndarray,
    node_types: np.ndarray,
    *,
    root: int | None,
) -> tuple[tuple[bytes, ...], tuple[tuple[int, ...], ...]]:
    n = len(node_types)
    colors = tuple(
        _initial_color(int(node_types[node]), root == node) for node in range(n)
    )
    history = []
    for _round in range(max(n, 1)):
        _, counts = np.unique(np.asarray(colors, dtype="S32"), return_counts=True)
        history.append(tuple(sorted(int(value) for value in counts)))
        updated = []
        for node in range(n):
            messages = sorted(
                int(typed_adjacency[node, neighbor]).to_bytes(2, "big") + colors[neighbor]
                for neighbor in np.flatnonzero(typed_adjacency[node])
            )
            updated.append(_digest((b"typed-message", colors[node], *messages)))
        colors = tuple(updated)
    return colors, tuple(history)


def _rooted_fingerprint(
    typed_adjacency: np.ndarray, node_types: np.ndarray, root: int
) -> tuple:
    colors, history = _refine_wl(typed_adjacency, node_types, root=root)
    return (
        colors[root],
        tuple(sorted(colors)),
        history,
    )


@dataclass(frozen=True)
class AttributedStableOrder:
    order: tuple[int, ...]
    keys: tuple[tuple, ...]
    class_sizes: tuple[int, ...]

    @property
    def singleton_fraction(self) -> float:
        return float(np.mean(np.asarray(self.class_sizes) == 1)) if self.class_sizes else 0.0


def attributed_stable_order(
    typed_adjacency: np.ndarray, node_types: Sequence[int]
) -> AttributedStableOrder:
    typed = np.asarray(typed_adjacency, dtype=np.int16)
    types = np.asarray(node_types, dtype=np.int64)
    if typed.shape != (len(types), len(types)) or not np.array_equal(typed, typed.T):
        raise ValueError("typed adjacency and node types are incompatible")
    global_colors, _history = _refine_wl(typed, types, root=None)
    keys = tuple((int(types[node]), global_colors[node]) for node in range(len(types)))
    order = _nauty_attributed_order(typed, types)
    key_counts: dict[tuple, int] = {}
    for key in keys:
        key_counts[key] = key_counts.get(key, 0) + 1
    return AttributedStableOrder(
        order=order,
        keys=keys,
        class_sizes=tuple(key_counts[keys[node]] for node in range(len(types))),
    )


def attributed_automorphism_orbits(
    typed_adjacency: np.ndarray, node_types: Sequence[int]
) -> tuple[tuple[int, ...], ...]:
    """Return original-node automorphism orbits of the attributed graph."""
    import pynauty

    typed = np.asarray(typed_adjacency, dtype=np.int16)
    types = np.asarray(node_types, dtype=np.int64)
    n = len(types)
    if typed.shape != (n, n) or not np.array_equal(typed, typed.T):
        raise ValueError("typed adjacency and node types are incompatible")
    left, right = np.nonzero(np.triu(typed, k=1) > 0)
    edges = [(int(u), int(v), int(typed[u, v])) for u, v in zip(left, right)]
    adjacency: dict[int, list[int]] = {node: [] for node in range(n + len(edges))}
    for edge_index, (u, v, _edge_type) in enumerate(edges):
        edge_node = n + edge_index
        adjacency[u].append(edge_node)
        adjacency[v].append(edge_node)
        adjacency[edge_node] = [u, v]
    coloring = [
        {node for node in range(n) if int(types[node]) == node_type}
        for node_type in sorted(set(int(value) for value in types))
    ]
    coloring.extend(
        {
            n + edge_index
            for edge_index, (_u, _v, value) in enumerate(edges)
            if value == edge_type
        }
        for edge_type in sorted(set(edge_type for _u, _v, edge_type in edges))
    )
    graph = pynauty.Graph(
        number_of_vertices=n + len(edges),
        directed=False,
        adjacency_dict=adjacency,
        vertex_coloring=coloring,
    )
    _generators, _size1, _size2, orbit_ids, _num_orbits = pynauty.autgrp(graph)
    groups: dict[int, list[int]] = {}
    for node in range(n):
        groups.setdefault(int(orbit_ids[node]), []).append(node)
    return tuple(tuple(nodes) for _orbit, nodes in sorted(groups.items()))


def attributed_rooted_signature(
    typed_adjacency: np.ndarray, node_types: Sequence[int], root: int
) -> bytes:
    """Canonical signature of the full attributed graph with one rooted node."""
    typed = np.asarray(typed_adjacency, dtype=np.int16)
    types = np.asarray(node_types, dtype=np.int64)
    order = _nauty_attributed_order(typed, types, root=int(root))
    canonical_typed = typed[np.ix_(order, order)]
    root_position = int(order.index(int(root)))
    return _digest(
        (
            b"attributed-rooted-graph-v1",
            root_position.to_bytes(4, "big"),
            repr(tuple(int(types[node]) for node in order)).encode("ascii"),
            canonical_typed.tobytes(),
        )
    )


def attributed_component_signature(
    typed_adjacency: np.ndarray, node_types: Sequence[int]
) -> bytes:
    typed = np.asarray(typed_adjacency, dtype=np.int16)
    types = np.asarray(node_types, dtype=np.int64)
    order = _nauty_attributed_order(typed, types)
    edge_types = typed[np.ix_(order, order)]
    return _digest(
        (
            b"attributed-component-v1",
            repr(tuple(int(types[node]) for node in order)).encode("ascii"),
            edge_types.tobytes(),
        )
    )


def _typed_refine(
    typed: np.ndarray,
    partition: tuple[tuple[int, ...], ...],
    edge_dim: int,
) -> tuple[tuple[int, ...], ...]:
    current = partition
    while True:
        refined: list[tuple[int, ...]] = []
        for cell in current:
            buckets: dict[tuple[int, ...], list[int]] = {}
            for vertex in cell:
                signature = []
                for block in current:
                    values = typed[vertex, np.asarray(block, dtype=np.int64)]
                    signature.extend(
                        int(np.sum(values == edge_type))
                        for edge_type in range(1, edge_dim + 1)
                    )
                buckets.setdefault(tuple(signature), []).append(vertex)
            for signature in sorted(buckets):
                refined.append(tuple(sorted(buckets[signature])))
        updated = tuple(refined)
        if updated == current:
            return current
        current = updated


def _typed_swap_automorphism(
    typed: np.ndarray,
    node_types: np.ndarray,
    left: int,
    right: int,
) -> bool:
    if node_types[left] != node_types[right]:
        return False
    others = [node for node in range(len(node_types)) if node not in (left, right)]
    return np.array_equal(typed[left, others], typed[right, others])


def exact_typed_canonical_order(
    typed_adjacency: np.ndarray,
    node_ids: Sequence[int],
    node_types: Sequence[int],
    *,
    root: int,
) -> CanonicalOrderResult:
    typed = np.asarray(typed_adjacency, dtype=np.int16)
    nodes = tuple(int(node) for node in node_ids)
    types = np.asarray(node_types, dtype=np.int64)
    n = len(nodes)
    if typed.shape != (n, n) or types.shape != (n,):
        raise ValueError("typed patch fields are incompatible")
    local = {node: index for index, node in enumerate(nodes)}
    root_local = local[int(root)]
    selected = _nauty_attributed_order(typed, types, root=root_local)
    best_code = tuple(int(types[node]) for node in selected) + tuple(
        int(typed[selected[left], selected[right]])
        for left in range(n)
        for right in range(left + 1, n)
    )
    return CanonicalOrderResult(
        node_ids=tuple(nodes[index] for index in selected),
        adjacency_code=best_code,
        search_leaves=1,
        optimal_leaf_count=1,
        symmetry_pruned=False,
    )


def typed_slot_vector(
    slot_nodes: Sequence[int],
    typed_adjacency: np.ndarray,
    node_types: Sequence[int],
    *,
    patch_size: int,
    node_dim: int,
    edge_dim: int,
) -> np.ndarray:
    nodes = tuple(int(node) for node in slot_nodes)
    typed = np.asarray(typed_adjacency, dtype=np.int16)
    types = np.asarray(node_types, dtype=np.int64)
    output = []
    for slot in range(patch_size):
        value = int(types[nodes[slot]]) if slot < len(nodes) else -1
        output.extend(float(value == node_type) for node_type in range(node_dim))
    for left in range(patch_size):
        for right in range(left + 1, patch_size):
            value = int(typed[nodes[left], nodes[right]]) if left < len(nodes) and right < len(nodes) else 0
            output.extend(float(value == edge_type) for edge_type in range(1, edge_dim + 1))
    return np.asarray(output, dtype=np.float64)

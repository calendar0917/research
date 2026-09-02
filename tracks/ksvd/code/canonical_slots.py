"""ID-free structural slot orderings for complete patch adjacency vectors."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .overlap_cover import PatchCover, _make_cover, _make_patch


@dataclass(frozen=True)
class CanonicalOrderResult:
    node_ids: tuple[int, ...]
    adjacency_code: tuple[int, ...]
    search_leaves: int
    optimal_leaf_count: int
    symmetry_pruned: bool

    @property
    def ambiguous(self) -> bool:
        return self.optimal_leaf_count > 1 or self.symmetry_pruned


def _upper_code(adjacency: np.ndarray, order: Sequence[int]) -> tuple[int, ...]:
    return tuple(
        int(adjacency[order[left], order[right]])
        for left in range(len(order))
        for right in range(left + 1, len(order))
    )


def _refine(
    adjacency: np.ndarray,
    partition: tuple[tuple[int, ...], ...],
) -> tuple[tuple[int, ...], ...]:
    current = partition
    while True:
        refined: list[tuple[int, ...]] = []
        for cell in current:
            buckets: dict[tuple[int, ...], list[int]] = {}
            for vertex in cell:
                signature = tuple(
                    int(np.sum(adjacency[vertex, np.asarray(block, dtype=np.int64)]))
                    for block in current
                )
                buckets.setdefault(signature, []).append(vertex)
            for signature in sorted(buckets):
                refined.append(tuple(sorted(buckets[signature])))
        updated = tuple(refined)
        if updated == current:
            return current
        current = updated


def _swap_is_automorphism(adjacency: np.ndarray, left: int, right: int) -> bool:
    if left == right:
        return True
    for vertex in range(adjacency.shape[0]):
        if vertex in (left, right):
            continue
        if adjacency[left, vertex] != adjacency[right, vertex]:
            return False
    return True


def _branch_representatives(adjacency: np.ndarray, cell: tuple[int, ...]) -> tuple[int, ...]:
    representatives: list[int] = []
    for vertex in cell:
        if any(_swap_is_automorphism(adjacency, vertex, other) for other in representatives):
            continue
        representatives.append(vertex)
    return tuple(representatives)


def exact_canonical_order(
    patch_adjacency: np.ndarray,
    node_ids: Sequence[int],
    *,
    color_cells: Sequence[Sequence[int]] | None = None,
) -> CanonicalOrderResult:
    """Return an exact colored-graph canonical adjacency order.

    ``color_cells`` is an ordered partition expressed with global node IDs.
    The adjacency code, not the returned choice among automorphic node maps, is
    the invariant object.
    """
    values = np.asarray(patch_adjacency, dtype=np.int8)
    nodes = tuple(int(node) for node in node_ids)
    n_nodes = len(nodes)
    if values.shape != (n_nodes, n_nodes):
        raise ValueError("patch_adjacency and node_ids are incompatible")
    if not np.array_equal(values, values.T) or np.any(np.diag(values) != 0):
        raise ValueError("expected a simple undirected patch adjacency")
    local = {node: index for index, node in enumerate(nodes)}
    if color_cells is None:
        partition = (tuple(range(n_nodes)),)
    else:
        seen: set[int] = set()
        converted = []
        for raw_cell in color_cells:
            cell = tuple(local[int(node)] for node in raw_cell)
            if not cell or any(vertex in seen for vertex in cell):
                raise ValueError("color_cells must be a nonempty ordered partition")
            seen.update(cell)
            converted.append(cell)
        if seen != set(range(n_nodes)):
            raise ValueError("color_cells must contain every patch node exactly once")
        partition = tuple(converted)

    leaves = 0
    best_code: tuple[int, ...] | None = None
    best_orders: list[tuple[int, ...]] = []
    symmetry_pruned = False

    def search(current: tuple[tuple[int, ...], ...]) -> None:
        nonlocal leaves, best_code, best_orders, symmetry_pruned
        current = _refine(values, current)
        target_index = next(
            (index for index, cell in enumerate(current) if len(cell) > 1),
            None,
        )
        if target_index is None:
            leaves += 1
            order = tuple(cell[0] for cell in current)
            code = _upper_code(values, order)
            if best_code is None or code < best_code:
                best_code = code
                best_orders = [order]
            elif code == best_code:
                best_orders.append(order)
            return
        cell = current[target_index]
        representatives = _branch_representatives(values, cell)
        symmetry_pruned = symmetry_pruned or len(representatives) < len(cell)
        for vertex in representatives:
            remainder = tuple(item for item in cell if item != vertex)
            individualized = (
                current[:target_index]
                + ((vertex,), remainder)
                + current[target_index + 1 :]
            )
            search(individualized)

    search(partition)
    if best_code is None or not best_orders:
        raise RuntimeError("canonical search produced no leaf")
    selected = min(best_orders, key=lambda order: tuple(nodes[index] for index in order))
    return CanonicalOrderResult(
        node_ids=tuple(nodes[index] for index in selected),
        adjacency_code=best_code,
        search_leaves=leaves,
        optimal_leaf_count=len(best_orders),
        symmetry_pruned=symmetry_pruned,
    )


def _distances(adjacency: np.ndarray, root: int) -> np.ndarray:
    distance = np.full(adjacency.shape[0], adjacency.shape[0] + 1, dtype=np.int64)
    distance[root] = 0
    queue = [root]
    for node in queue:
        for neighbor in np.flatnonzero(adjacency[node]):
            value = int(neighbor)
            if distance[value] > distance[node] + 1:
                distance[value] = distance[node] + 1
                queue.append(value)
    return distance


def signature_order(patch_adjacency: np.ndarray, node_ids: Sequence[int], center: int) -> tuple[tuple[int, ...], bool]:
    values = np.asarray(patch_adjacency, dtype=np.int8)
    nodes = tuple(int(node) for node in node_ids)
    local = {node: index for index, node in enumerate(nodes)}
    center_local = local[int(center)]
    degrees = np.sum(values, axis=1).astype(np.int64)
    triangles = np.diag(values @ values @ values).astype(np.int64) // 2
    distances = _distances(values, center_local)
    colors = [int(value) for value in degrees]
    for _iteration in range(3):
        signatures = [
            (colors[index], tuple(sorted(colors[int(neighbor)] for neighbor in np.flatnonzero(values[index]))))
            for index in range(len(nodes))
        ]
        vocabulary = {signature: rank for rank, signature in enumerate(sorted(set(signatures)))}
        colors = [vocabulary[signature] for signature in signatures]
    structural = {
        node: (
            int(node != center),
            int(distances[index]),
            -int(degrees[index]),
            -int(triangles[index]),
            int(colors[index]),
        )
        for index, node in enumerate(nodes)
    }
    tied = len(set(structural.values())) < len(nodes)
    return tuple(sorted(nodes, key=lambda node: structural[node] + (node,))), tied


def reorder_cover_structurally(
    adjacency: np.ndarray,
    cover: PatchCover,
    mode: str,
) -> tuple[PatchCover, dict[str, float | int]]:
    if mode not in {"id", "signature", "canonical", "rooted_canonical", "overlap_canonical"}:
        raise ValueError(f"unknown structural slot mode: {mode}")
    ordered_patches = []
    tie_count = 0
    ambiguity_count = 0
    leaf_counts = []
    previous_order: tuple[int, ...] | None = None
    for patch_index, patch in enumerate(cover.patches):
        if mode == "id":
            order = tuple(sorted(patch.node_ids))
        elif mode == "signature":
            order, tied = signature_order(patch.adjacency, patch.node_ids, patch.center)
            tie_count += int(tied)
        else:
            if mode == "canonical":
                colors = None
            elif mode == "rooted_canonical" or patch_index == 0:
                rest = tuple(node for node in patch.node_ids if node != patch.center)
                colors = ((patch.center,), rest) if rest else ((patch.center,),)
            else:
                assert previous_order is not None
                shared = [node for node in previous_order if node in set(patch.node_ids)]
                remainder = [node for node in patch.node_ids if node not in set(shared)]
                colors_list: list[tuple[int, ...]] = [(node,) for node in shared]
                if patch.center in remainder:
                    colors_list.append((patch.center,))
                    remainder.remove(patch.center)
                if remainder:
                    colors_list.append(tuple(remainder))
                colors = tuple(colors_list)
            result = exact_canonical_order(
                patch.adjacency,
                patch.node_ids,
                color_cells=colors,
            )
            order = result.node_ids
            ambiguity_count += int(result.ambiguous)
            leaf_counts.append(result.search_leaves)
        ordered_patches.append(_make_patch(adjacency, order, patch.center))
        previous_order = order
    reordered = _make_cover(
        mode,
        ordered_patches,
        cover.segment_ids,
        cover.target_edges,
        cover.bridge_lengths,
    )
    count = len(cover.patches)
    return reordered, {
        "patch_count": count,
        "signature_tie_rate": tie_count / max(count, 1),
        "canonical_ambiguity_rate": ambiguity_count / max(count, 1),
        "mean_search_leaves": float(np.mean(leaf_counts)) if leaf_counts else 0.0,
        "max_search_leaves": max(leaf_counts, default=0),
    }

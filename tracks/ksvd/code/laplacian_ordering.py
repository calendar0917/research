"""Graph-global Fiedler coordinates and matched patch reordering utilities."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .from_scratch_unplanted_representation import (
    degree_preserving_double_edge_swaps,
    relabel_adjacency_and_order,
    validate_simple_adjacency,
)
from .overlap_cover import OrderedPatch, PatchCover, PatchTransition, remap_cover


EPS = 1e-12


@dataclass(frozen=True)
class FiedlerCoordinate:
    eigenvalues: np.ndarray
    values: np.ndarray
    ranks: np.ndarray
    relative_eigengap: float


def normalized_fiedler_coordinate(adjacency: np.ndarray) -> FiedlerCoordinate:
    """Return a sign-canonicalized normalized-Laplacian Fiedler coordinate."""
    validate_simple_adjacency(adjacency)
    values = np.asarray(adjacency, dtype=np.float64)
    degrees = np.sum(values, axis=1)
    if np.any(degrees <= 0.0):
        raise ValueError("Fiedler coordinates require a graph without isolated nodes")
    inverse_sqrt = 1.0 / np.sqrt(degrees)
    normalized = inverse_sqrt[:, None] * values * inverse_sqrt[None, :]
    laplacian = np.eye(values.shape[0], dtype=np.float64) - normalized
    eigenvalues, eigenvectors = np.linalg.eigh(laplacian)
    if eigenvalues.size < 3:
        raise ValueError("Fiedler eigengap audit requires at least three nodes")
    fiedler = eigenvectors[:, 1].copy()
    pivot = int(np.argmax(np.abs(fiedler)))
    if fiedler[pivot] < 0.0:
        fiedler *= -1.0
    order = np.argsort(fiedler, kind="stable")
    ranks = np.empty(order.size, dtype=np.int64)
    ranks[order] = np.arange(order.size, dtype=np.int64)
    relative_gap = float(
        (eigenvalues[2] - eigenvalues[1]) / max(abs(eigenvalues[1]), EPS)
    )
    return FiedlerCoordinate(
        eigenvalues=eigenvalues[:3].copy(),
        values=fiedler,
        ranks=ranks,
        relative_eigengap=relative_gap,
    )


def pair_order_agreement(left_ranks: np.ndarray, right_ranks: np.ndarray) -> float:
    """Fraction of node pairs with the same relative order in two rankings."""
    left = np.asarray(left_ranks, dtype=np.int64)
    right = np.asarray(right_ranks, dtype=np.int64)
    if left.shape != right.shape or left.ndim != 1 or left.size < 2:
        raise ValueError("rank arrays must be aligned vectors with at least two nodes")
    if np.unique(left).size != left.size or np.unique(right).size != right.size:
        raise ValueError("rank arrays must be permutations without ties")
    matches = 0
    trials = 0
    for first in range(left.size):
        for second in range(first + 1, left.size):
            matches += int(
                (left[first] < left[second]) == (right[first] < right[second])
            )
            trials += 1
    return float(matches / trials)


def coordinate_tie_diagnostics(
    values: np.ndarray, *, tolerance: float = 1e-10
) -> dict[str, float | int]:
    """Describe coordinate ties without using the result as a registered gate."""
    coordinate = np.asarray(values, dtype=np.float64)
    if coordinate.ndim != 1 or coordinate.size < 2:
        raise ValueError("coordinate values must be a vector with at least two nodes")
    if tolerance < 0.0 or not np.isfinite(tolerance):
        raise ValueError("tolerance must be finite and non-negative")
    ordered = np.sort(coordinate, kind="stable")
    gaps = np.diff(ordered)
    tied_gap = gaps <= tolerance
    tied_nodes = np.zeros(coordinate.size, dtype=bool)
    tied_indices = np.flatnonzero(tied_gap)
    tied_nodes[tied_indices] = True
    tied_nodes[tied_indices + 1] = True
    return {
        "tolerance": float(tolerance),
        "unique_coordinate_group_count": int(1 + np.count_nonzero(~tied_gap)),
        "unique_coordinate_group_fraction": float(
            (1 + np.count_nonzero(~tied_gap)) / coordinate.size
        ),
        "tied_node_fraction": float(np.mean(tied_nodes)),
        "minimum_adjacent_gap": float(np.min(gaps)),
        "median_adjacent_gap": float(np.median(gaps)),
    }


def _transition(left: OrderedPatch, right: OrderedPatch) -> PatchTransition:
    right_slots = {node: slot for slot, node in enumerate(right.node_ids)}
    return PatchTransition(
        tuple(
            (left_slot, right_slots[node])
            for left_slot, node in enumerate(left.node_ids)
            if node in right_slots
        )
    )


def reorder_cover_by_fiedler(
    adjacency: np.ndarray,
    cover: PatchCover,
    *,
    method: str = "fiedler_global_order",
) -> tuple[PatchCover, FiedlerCoordinate]:
    """Reorder each frozen patch node set by one graph-global Fiedler rank."""
    validate_simple_adjacency(adjacency)
    coordinate = normalized_fiedler_coordinate(adjacency)
    patches = []
    for patch in cover.patches:
        node_ids = tuple(sorted(patch.node_ids, key=lambda node: coordinate.ranks[node]))
        induced = adjacency[np.ix_(node_ids, node_ids)].astype(np.int8, copy=True)
        patches.append(
            OrderedPatch(node_ids=node_ids, center=patch.center, adjacency=induced)
        )
    transitions = tuple(
        _transition(left, right) for left, right in zip(patches, patches[1:])
    )
    reordered = PatchCover(
        method=method,
        patches=tuple(patches),
        transitions=transitions,
        segment_ids=cover.segment_ids,
        target_edges=cover.target_edges,
        bridge_lengths=cover.bridge_lengths,
    )
    return reordered, coordinate


def mapped_relabel_fiedler_invariance(
    adjacency: np.ndarray,
    cover: PatchCover,
    permutation: np.ndarray,
) -> dict[str, float]:
    """Recompute Fiedler ordering after relabeling and compare mapped results."""
    permutation = np.asarray(permutation, dtype=np.int64)
    ordered, original_coordinate = reorder_cover_by_fiedler(adjacency, cover)
    relabeled, _unused = relabel_adjacency_and_order(adjacency, [], permutation)
    inverse = np.empty(permutation.size, dtype=np.int64)
    inverse[permutation] = np.arange(permutation.size)
    relabeled_cover = remap_cover(cover, inverse, relabeled)
    relabeled_ordered, relabeled_coordinate = reorder_cover_by_fiedler(
        relabeled, relabeled_cover
    )
    mapped_ranks = relabeled_coordinate.ranks[inverse]
    rank_match = float(np.mean(mapped_ranks == original_coordinate.ranks))
    patch_matches = [
        np.array_equal(left.adjacency, right.adjacency)
        for left, right in zip(ordered.patches, relabeled_ordered.patches)
    ]
    transition_matches = [
        left.left_to_right_slots == right.left_to_right_slots
        for left, right in zip(ordered.transitions, relabeled_ordered.transitions)
    ]
    return {
        "rank_match_rate": rank_match,
        "patch_adjacency_match_rate": float(np.mean(patch_matches)),
        "transition_slot_map_match_rate": float(np.mean(transition_matches))
        if transition_matches
        else 1.0,
    }


def one_swap_fiedler_stability(
    adjacency: np.ndarray,
    rng: np.random.Generator,
) -> dict[str, float | int]:
    """Audit Fiedler rank stability after one connected degree-preserving swap."""
    original = normalized_fiedler_coordinate(adjacency)
    perturbed, attempts = degree_preserving_double_edge_swaps(
        adjacency, 1, rng, require_connected=True
    )
    changed = normalized_fiedler_coordinate(perturbed)
    canonical = pair_order_agreement(original.ranks, changed.ranks)
    return {
        "canonical_pair_order_agreement": canonical,
        "sign_invariant_pair_order_agreement": max(canonical, 1.0 - canonical),
        "swap_attempts": int(attempts),
        "original_relative_eigengap": original.relative_eigengap,
        "perturbed_relative_eigengap": changed.relative_eigengap,
    }

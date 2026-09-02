"""U0-R utilities for unplanted graph generation and adjacency audits.

The module deliberately separates two properties that are often conflated:
rooted canonicalization gives an isomorphism-invariant representative, while a
walk-order adjacency gives fixed sampler-slot semantics.  Neither property by
itself proves that Euclidean geometry is suitable for KSVD.
"""
from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from itertools import permutations
from typing import Any

import numpy as np


@dataclass(frozen=True)
class RewiredGraph:
    adjacency: np.ndarray
    requested_swaps: int
    accepted_swaps: int
    attempted_swaps: int
    global_permutation: tuple[int, ...]


@dataclass(frozen=True)
class RootedCanonicalResult:
    vector: np.ndarray
    minimizing_permutation_count: int
    selected_permutation: tuple[int, ...]


@dataclass(frozen=True)
class WalkPatch:
    root: int
    node_ids: tuple[int, ...]
    walk_trace: tuple[int, ...]
    adjacency_walk_order: np.ndarray
    walk_order_vector: np.ndarray
    canonical_vector: np.ndarray
    canonical_minimizing_permutation_count: int


def upper_triangle_edges(n_nodes: int) -> tuple[tuple[int, int], ...]:
    return tuple((left, right) for left in range(n_nodes) for right in range(left + 1, n_nodes))


def adjacency_to_upper_vector(adjacency: np.ndarray) -> np.ndarray:
    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    return np.asarray(
        [adjacency[left, right] for left, right in upper_triangle_edges(adjacency.shape[0])],
        dtype=np.float64,
    )


def upper_vector_to_adjacency(vector: np.ndarray, n_nodes: int) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    edges = upper_triangle_edges(n_nodes)
    if vector.shape != (len(edges),):
        raise ValueError(f"expected vector shape {(len(edges),)}, got {vector.shape}")
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    for value, (left, right) in zip(vector, edges):
        adjacency[left, right] = value
        adjacency[right, left] = value
    return adjacency


def validate_simple_adjacency(adjacency: np.ndarray) -> None:
    adjacency = np.asarray(adjacency)
    if adjacency.ndim != 2 or adjacency.shape[0] != adjacency.shape[1]:
        raise ValueError("adjacency must be square")
    if not np.array_equal(adjacency, adjacency.T):
        raise ValueError("adjacency must be symmetric")
    if np.any(np.diag(adjacency) != 0):
        raise ValueError("adjacency diagonal must be zero")
    if not np.all((adjacency == 0) | (adjacency == 1)):
        raise ValueError("adjacency must be binary")


def is_connected(adjacency: np.ndarray) -> bool:
    validate_simple_adjacency(adjacency)
    n_nodes = adjacency.shape[0]
    if n_nodes == 0:
        return False
    seen = {0}
    stack = [0]
    while stack:
        node = stack.pop()
        for neighbor in np.flatnonzero(adjacency[node]):
            value = int(neighbor)
            if value not in seen:
                seen.add(value)
                stack.append(value)
    return len(seen) == n_nodes


def ring_lattice_adjacency(n_nodes: int = 60, neighbors_each_side: int = 2) -> np.ndarray:
    if n_nodes < 2 * neighbors_each_side + 1:
        raise ValueError("n_nodes is too small for the requested ring lattice")
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for node in range(n_nodes):
        for offset in range(1, neighbors_each_side + 1):
            neighbor = (node + offset) % n_nodes
            adjacency[node, neighbor] = 1
            adjacency[neighbor, node] = 1
    validate_simple_adjacency(adjacency)
    return adjacency


def _edge_list(adjacency: np.ndarray) -> list[tuple[int, int]]:
    n_nodes = adjacency.shape[0]
    return [
        (left, right)
        for left in range(n_nodes)
        for right in range(left + 1, n_nodes)
        if adjacency[left, right] != 0
    ]


def degree_preserving_double_edge_swaps(
    adjacency: np.ndarray,
    n_accepted_swaps: int,
    rng: np.random.Generator,
    *,
    require_connected: bool = True,
    max_attempts: int | None = None,
) -> tuple[np.ndarray, int]:
    """Apply simple-graph double-edge swaps and return (adjacency, attempts)."""
    validate_simple_adjacency(adjacency)
    if n_accepted_swaps < 0:
        raise ValueError("n_accepted_swaps must be non-negative")
    result = np.asarray(adjacency, dtype=np.int8).copy()
    original_degrees = result.sum(axis=1).copy()
    accepted = 0
    attempts = 0
    limit = max_attempts if max_attempts is not None else 1000 + 200 * max(n_accepted_swaps, 1)
    edges = _edge_list(result)
    while accepted < n_accepted_swaps and attempts < limit:
        attempts += 1
        first_index, second_index = rng.choice(len(edges), size=2, replace=False)
        a, b = edges[int(first_index)]
        c, d = edges[int(second_index)]
        if len({a, b, c, d}) != 4:
            continue
        if float(rng.random()) < 0.5:
            new_edges = ((a, c), (b, d))
        else:
            new_edges = ((a, d), (b, c))
        if any(left == right or result[left, right] != 0 for left, right in new_edges):
            continue

        result[a, b] = result[b, a] = 0
        result[c, d] = result[d, c] = 0
        for left, right in new_edges:
            result[left, right] = result[right, left] = 1
        if require_connected and not is_connected(result):
            for left, right in new_edges:
                result[left, right] = result[right, left] = 0
            result[a, b] = result[b, a] = 1
            result[c, d] = result[d, c] = 1
            continue
        accepted += 1
        edges = _edge_list(result)

    if accepted != n_accepted_swaps:
        raise RuntimeError(
            f"accepted only {accepted}/{n_accepted_swaps} swaps after {attempts} attempts"
        )
    validate_simple_adjacency(result)
    if not np.array_equal(result.sum(axis=1), original_degrees):
        raise RuntimeError("double-edge swaps changed the degree sequence")
    if require_connected and not is_connected(result):
        raise RuntimeError("double-edge swaps produced a disconnected graph")
    return result, attempts


def generate_rewired_graph(
    *,
    seed: int,
    n_accepted_swaps: int,
    n_nodes: int = 60,
    neighbors_each_side: int = 2,
) -> RewiredGraph:
    swap_sequence, permutation_sequence = np.random.SeedSequence(seed).spawn(2)
    swap_rng = np.random.default_rng(swap_sequence)
    permutation_rng = np.random.default_rng(permutation_sequence)
    base = ring_lattice_adjacency(n_nodes=n_nodes, neighbors_each_side=neighbors_each_side)
    rewired, attempts = degree_preserving_double_edge_swaps(base, n_accepted_swaps, swap_rng)
    permutation = permutation_rng.permutation(n_nodes)
    relabeled = rewired[np.ix_(permutation, permutation)]
    return RewiredGraph(
        adjacency=relabeled,
        requested_swaps=int(n_accepted_swaps),
        accepted_swaps=int(n_accepted_swaps),
        attempted_swaps=int(attempts),
        global_permutation=tuple(int(value) for value in permutation),
    )


@lru_cache(maxsize=None)
def rooted_permutations(n_nodes: int) -> tuple[tuple[int, ...], ...]:
    if n_nodes < 1:
        raise ValueError("n_nodes must be positive")
    return tuple((0,) + tail for tail in permutations(range(1, n_nodes)))


@lru_cache(maxsize=None)
def rooted_permutation_index_map(n_nodes: int) -> np.ndarray:
    edges = upper_triangle_edges(n_nodes)
    edge_to_index = {edge: index for index, edge in enumerate(edges)}
    rows: list[list[int]] = []
    for permutation in rooted_permutations(n_nodes):
        row = []
        for left, right in edges:
            source = tuple(sorted((permutation[left], permutation[right])))
            row.append(edge_to_index[source])
        rows.append(row)
    return np.asarray(rows, dtype=np.int16)


def rooted_exact_canonical_vector(adjacency: np.ndarray) -> RootedCanonicalResult:
    validate_simple_adjacency(adjacency)
    n_nodes = adjacency.shape[0]
    raw = adjacency_to_upper_vector(adjacency).astype(np.int8)
    candidates = raw[rooted_permutation_index_map(n_nodes)]
    if candidates.shape[1] > 62:
        order = np.lexsort(candidates[:, ::-1].T)
        best_row = int(order[0])
        best = candidates[best_row]
        minimizing = np.flatnonzero(np.all(candidates == best, axis=1))
    else:
        weights = (1 << np.arange(candidates.shape[1] - 1, -1, -1)).astype(np.int64)
        codes = candidates.astype(np.int64) @ weights
        minimum = int(np.min(codes))
        minimizing = np.flatnonzero(codes == minimum)
        best_row = int(minimizing[0])
    return RootedCanonicalResult(
        vector=candidates[best_row].astype(np.float64),
        minimizing_permutation_count=int(minimizing.size),
        selected_permutation=tuple(int(value) for value in rooted_permutations(n_nodes)[best_row]),
    )


def exact_rooted_graph_edit_distance(adjacency_a: np.ndarray, adjacency_b: np.ndarray) -> int:
    validate_simple_adjacency(adjacency_a)
    validate_simple_adjacency(adjacency_b)
    if adjacency_a.shape != adjacency_b.shape:
        raise ValueError("rooted graph edit distance requires equal shapes")
    raw_a = adjacency_to_upper_vector(adjacency_a).astype(np.int8)
    raw_b = adjacency_to_upper_vector(adjacency_b).astype(np.int8)
    candidates_b = raw_b[rooted_permutation_index_map(adjacency_a.shape[0])]
    return int(np.min(np.count_nonzero(candidates_b != raw_a[None, :], axis=1)))


def sample_walk_patch(
    adjacency: np.ndarray,
    root: int,
    rng: np.random.Generator,
    *,
    patch_size: int = 6,
    max_steps: int = 80,
    max_retries: int = 20,
) -> WalkPatch:
    validate_simple_adjacency(adjacency)
    n_nodes = adjacency.shape[0]
    if not 0 <= root < n_nodes:
        raise ValueError("root is outside the graph")
    if not 1 <= patch_size <= n_nodes:
        raise ValueError("invalid patch_size")

    for _ in range(max_retries):
        current = int(root)
        trace = [current]
        node_ids = [current]
        seen = {current}
        for _step in range(max_steps):
            neighbors = np.flatnonzero(adjacency[current])
            if neighbors.size == 0:
                break
            current = int(rng.choice(neighbors))
            trace.append(current)
            if current not in seen:
                seen.add(current)
                node_ids.append(current)
                if len(node_ids) == patch_size:
                    induced = adjacency[np.ix_(node_ids, node_ids)].astype(np.int8, copy=True)
                    canonical = rooted_exact_canonical_vector(induced)
                    return WalkPatch(
                        root=int(root),
                        node_ids=tuple(int(value) for value in node_ids),
                        walk_trace=tuple(int(value) for value in trace),
                        adjacency_walk_order=induced,
                        walk_order_vector=adjacency_to_upper_vector(induced),
                        canonical_vector=canonical.vector,
                        canonical_minimizing_permutation_count=canonical.minimizing_permutation_count,
                    )
    raise RuntimeError(
        f"failed to collect {patch_size} distinct nodes from root {root} "
        f"after {max_retries} retries"
    )


def sample_graph_walk_patches(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    *,
    n_patches: int,
    patch_size: int = 6,
    max_steps: int = 80,
    max_retries: int = 20,
) -> tuple[WalkPatch, ...]:
    validate_simple_adjacency(adjacency)
    if n_patches > adjacency.shape[0]:
        raise ValueError("n_patches cannot exceed n_nodes when roots are sampled without replacement")
    roots = rng.choice(adjacency.shape[0], size=n_patches, replace=False)
    return tuple(
        sample_walk_patch(
            adjacency,
            int(root),
            rng,
            patch_size=patch_size,
            max_steps=max_steps,
            max_retries=max_retries,
        )
        for root in roots
    )


def relabel_adjacency_and_order(
    adjacency: np.ndarray,
    ordered_node_ids: tuple[int, ...] | list[int],
    permutation: np.ndarray,
) -> tuple[np.ndarray, tuple[int, ...]]:
    """Relabel by new-index -> old-index permutation and map an old-node order."""
    validate_simple_adjacency(adjacency)
    permutation = np.asarray(permutation, dtype=np.int64)
    n_nodes = adjacency.shape[0]
    if permutation.shape != (n_nodes,) or np.unique(permutation).size != n_nodes:
        raise ValueError("permutation must contain every node exactly once")
    inverse = np.empty(n_nodes, dtype=np.int64)
    inverse[permutation] = np.arange(n_nodes)
    relabeled = adjacency[np.ix_(permutation, permutation)]
    mapped_order = tuple(int(inverse[int(node)]) for node in ordered_node_ids)
    return relabeled, mapped_order


def _rankdata(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    order = np.argsort(values, kind="mergesort")
    ranks = np.empty(values.size, dtype=np.float64)
    start = 0
    while start < values.size:
        end = start + 1
        while end < values.size and values[order[end]] == values[order[start]]:
            end += 1
        rank = 0.5 * (start + end - 1) + 1.0
        ranks[order[start:end]] = rank
        start = end
    return ranks


def _correlation(left: np.ndarray, right: np.ndarray) -> float:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    if left.size != right.size or left.size == 0:
        raise ValueError("correlation arrays must have equal non-zero length")
    left_centered = left - left.mean()
    right_centered = right - right.mean()
    denominator = float(np.linalg.norm(left_centered) * np.linalg.norm(right_centered))
    if denominator <= 1e-12:
        return 0.0
    return float((left_centered @ right_centered) / denominator)


def distance_summary(exact: np.ndarray, represented: np.ndarray) -> dict[str, float]:
    exact = np.asarray(exact, dtype=np.float64)
    represented = np.asarray(represented, dtype=np.float64)
    excess = represented - exact
    return {
        "pearson": _correlation(exact, represented),
        "spearman": _correlation(_rankdata(exact), _rankdata(represented)),
        "mean_distance": float(np.mean(represented)),
        "mean_excess_over_exact": float(np.mean(excess)),
        "p95_excess_over_exact": float(np.quantile(excess, 0.95)),
        "maximum_excess_over_exact": float(np.max(excess)),
    }


def audit_patch_collection(
    patches: tuple[WalkPatch, ...] | list[WalkPatch],
    rng: np.random.Generator,
    *,
    pair_count: int = 2000,
    root_permutation_trials: int = 20,
) -> dict[str, Any]:
    patches = tuple(patches)
    if len(patches) < 2:
        raise ValueError("at least two patches are required")
    patch_size = patches[0].adjacency_walk_order.shape[0]
    edge_count = len(upper_triangle_edges(patch_size))

    canonical_invariant = 0
    canonical_trials = 0
    for patch in patches:
        reference = patch.canonical_vector
        for _ in range(root_permutation_trials):
            tail = rng.permutation(np.arange(1, patch_size))
            permutation = np.concatenate(([0], tail))
            permuted = patch.adjacency_walk_order[np.ix_(permutation, permutation)]
            candidate = rooted_exact_canonical_vector(permuted).vector
            canonical_trials += 1
            canonical_invariant += int(np.array_equal(reference, candidate))

    canonical_flip_distances: list[int] = []
    walk_flip_distances: list[int] = []
    exact_flip_distances: list[int] = []
    edges = upper_triangle_edges(patch_size)
    for patch in patches:
        base = patch.adjacency_walk_order
        for left, right in edges:
            flipped = base.copy()
            flipped[left, right] = 1 - flipped[left, right]
            flipped[right, left] = flipped[left, right]
            exact_flip_distances.append(exact_rooted_graph_edit_distance(base, flipped))
            canonical = rooted_exact_canonical_vector(flipped).vector
            canonical_flip_distances.append(int(np.count_nonzero(canonical != patch.canonical_vector)))
            walk_vector = adjacency_to_upper_vector(flipped)
            walk_flip_distances.append(int(np.count_nonzero(walk_vector != patch.walk_order_vector)))

    pair_indices: list[tuple[int, int]] = []
    seen_pairs: set[tuple[int, int]] = set()
    maximum_pairs = len(patches) * (len(patches) - 1) // 2
    target_pairs = min(pair_count, maximum_pairs)
    while len(pair_indices) < target_pairs:
        left, right = rng.choice(len(patches), size=2, replace=False)
        pair = tuple(sorted((int(left), int(right))))
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            pair_indices.append(pair)

    exact_pair: list[int] = []
    canonical_pair: list[int] = []
    walk_pair: list[int] = []
    injectivity_disagreements = 0
    for left, right in pair_indices:
        patch_left = patches[left]
        patch_right = patches[right]
        exact = exact_rooted_graph_edit_distance(
            patch_left.adjacency_walk_order,
            patch_right.adjacency_walk_order,
        )
        canonical = int(np.count_nonzero(patch_left.canonical_vector != patch_right.canonical_vector))
        walk = int(np.count_nonzero(patch_left.walk_order_vector != patch_right.walk_order_vector))
        exact_pair.append(exact)
        canonical_pair.append(canonical)
        walk_pair.append(walk)
        injectivity_disagreements += int((canonical == 0) != (exact == 0))

    canonical_flip = np.asarray(canonical_flip_distances, dtype=np.float64)
    walk_flip = np.asarray(walk_flip_distances, dtype=np.float64)
    exact_flip = np.asarray(exact_flip_distances, dtype=np.float64)
    exact_pair_array = np.asarray(exact_pair, dtype=np.float64)
    canonical_pair_array = np.asarray(canonical_pair, dtype=np.float64)
    walk_pair_array = np.asarray(walk_pair, dtype=np.float64)
    return {
        "patch_count": len(patches),
        "patch_size": patch_size,
        "vector_dimension": edge_count,
        "canonical_permutation_invariance_rate": float(canonical_invariant / canonical_trials),
        "canonical_permutation_trial_count": int(canonical_trials),
        "canonical_minimizing_permutation_count": {
            "mean": float(np.mean([patch.canonical_minimizing_permutation_count for patch in patches])),
            "maximum": int(max(patch.canonical_minimizing_permutation_count for patch in patches)),
            "ambiguous_rate": float(np.mean([patch.canonical_minimizing_permutation_count > 1 for patch in patches])),
        },
        "one_edge_flip": {
            "trial_count": int(canonical_flip.size),
            "exact_distance_all_one": bool(np.all(exact_flip == 1)),
            "walk_distance_all_one": bool(np.all(walk_flip == 1)),
            "canonical_mean_distance": float(np.mean(canonical_flip)),
            "canonical_median_distance": float(np.median(canonical_flip)),
            "canonical_p95_distance": float(np.quantile(canonical_flip, 0.95)),
            "canonical_maximum_distance": float(np.max(canonical_flip)),
            "canonical_amplification_rate_gt_1": float(np.mean(canonical_flip > 1)),
            "canonical_severe_jump_rate_ge_4": float(np.mean(canonical_flip >= 4)),
        },
        "pairwise": {
            "pair_count": len(pair_indices),
            "canonical": distance_summary(exact_pair_array, canonical_pair_array),
            "walk_order": distance_summary(exact_pair_array, walk_pair_array),
            "canonical_injectivity_disagreement_count": int(injectivity_disagreements),
            "exact_distance_mean": float(np.mean(exact_pair_array)),
            "exact_distance_minimum": float(np.min(exact_pair_array)),
            "exact_distance_maximum": float(np.max(exact_pair_array)),
        },
    }

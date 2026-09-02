"""Small-graph exhaustive one-step marginal coverage covers."""
from __future__ import annotations

from itertools import combinations
from typing import Iterable, Sequence

import numpy as np

from .from_scratch_unplanted_representation import is_connected, validate_simple_adjacency
from .overlap_cover import PatchCover, _make_cover, _make_patch


def _pairs(nodes: Sequence[int]) -> set[tuple[int, int]]:
    return {tuple(sorted(pair)) for pair in combinations(nodes, 2)}


def _score_candidate(
    adjacency: np.ndarray,
    nodes: Sequence[int],
    observed_pairs: set[tuple[int, int]],
    covered_edges: set[tuple[int, int]],
    covered_nodes: set[int],
) -> tuple[int, int, int, int]:
    pairs = _pairs(nodes)
    edges = {pair for pair in pairs if adjacency[pair[0], pair[1]] != 0}
    return (
        len(edges - covered_edges),
        len(pairs - observed_pairs),
        len(set(nodes) - covered_nodes),
        len(edges),
    )


def _select_best_connected(
    adjacency: np.ndarray,
    candidates: Iterable[tuple[int, ...]],
    observed_pairs: set[tuple[int, int]],
    covered_edges: set[tuple[int, int]],
    covered_nodes: set[int],
    rng: np.random.Generator,
) -> tuple[int, ...]:
    best_score: tuple[int, int, int, int] | None = None
    selected: tuple[int, ...] | None = None
    tie_count = 0
    for nodes in candidates:
        induced = adjacency[np.ix_(nodes, nodes)]
        if not is_connected(induced):
            continue
        score = _score_candidate(
            adjacency, nodes, observed_pairs, covered_edges, covered_nodes
        )
        if best_score is None or score > best_score:
            best_score = score
            selected = nodes
            tie_count = 1
        elif score == best_score:
            tie_count += 1
            if int(rng.integers(tie_count)) == 0:
                selected = nodes
    if selected is None:
        raise RuntimeError("no connected exhaustive marginal candidate exists")
    return selected


def sample_exhaustive_marginal_cover(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    *,
    n_patches: int,
    patch_size: int,
    target_overlap: int,
) -> PatchCover:
    """Greedily maximize exact one-step marginal coverage by enumeration."""
    validate_simple_adjacency(adjacency)
    n_nodes = adjacency.shape[0]
    if not 0 < target_overlap < patch_size <= n_nodes:
        raise ValueError("expected 0 < target_overlap < patch_size <= n_nodes")
    if n_patches < 1:
        raise ValueError("n_patches must be positive")
    observed_pairs: set[tuple[int, int]] = set()
    covered_edges: set[tuple[int, int]] = set()
    covered_nodes: set[int] = set()
    patches = []

    def register(nodes: Sequence[int]) -> None:
        pairs = _pairs(nodes)
        observed_pairs.update(pairs)
        covered_edges.update(
            pair for pair in pairs if adjacency[pair[0], pair[1]] != 0
        )
        covered_nodes.update(nodes)

    first = _select_best_connected(
        adjacency,
        combinations(range(n_nodes), patch_size),
        observed_pairs,
        covered_edges,
        covered_nodes,
        rng,
    )
    patches.append(_make_patch(adjacency, first, first[0]))
    register(first)

    while len(patches) < n_patches:
        previous = tuple(patches[-1].node_ids)
        previous_set = set(previous)
        outside = tuple(node for node in range(n_nodes) if node not in previous_set)

        def candidates() -> Iterable[tuple[int, ...]]:
            for retained in combinations(previous, target_overlap):
                for new_nodes in combinations(outside, patch_size - target_overlap):
                    yield tuple(retained + new_nodes)

        selected = _select_best_connected(
            adjacency,
            candidates(),
            observed_pairs,
            covered_edges,
            covered_nodes,
            rng,
        )
        new_nodes = [node for node in selected if node not in previous_set]
        center = new_nodes[0] if new_nodes else selected[0]
        patches.append(_make_patch(adjacency, selected, center))
        register(selected)

    return _make_cover("exhaustive_marginal", patches)

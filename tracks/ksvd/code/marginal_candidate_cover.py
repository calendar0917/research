"""Scalable retained-subset beam search for marginal continuous covers."""
from __future__ import annotations

from itertools import combinations
from typing import Sequence

import numpy as np

from .from_scratch_unplanted_representation import is_connected, validate_simple_adjacency
from .overlap_cover import (
    PatchCover,
    _make_cover,
    _make_patch,
    sample_edge_target_bridge_cover,
)


def _pairs(nodes: Sequence[int]) -> set[tuple[int, int]]:
    return {tuple(sorted(pair)) for pair in combinations(nodes, 2)}


def _candidate_score(
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


def _retained_potential(
    adjacency: np.ndarray,
    retained: Sequence[int],
    previous_set: set[int],
    covered_edges: set[tuple[int, int]],
    covered_nodes: set[int],
) -> tuple[int, int, int]:
    outside = set(range(adjacency.shape[0])) - previous_set
    boundary_uncovered = 0
    boundary_edges = 0
    unseen_neighbors = set()
    for node in retained:
        for raw_neighbor in np.flatnonzero(adjacency[node]):
            neighbor = int(raw_neighbor)
            if neighbor not in outside:
                continue
            boundary_edges += 1
            pair = tuple(sorted((int(node), neighbor)))
            boundary_uncovered += int(pair not in covered_edges)
            if neighbor not in covered_nodes:
                unseen_neighbors.add(neighbor)
    return boundary_uncovered, len(unseen_neighbors), boundary_edges


def _greedy_fill(
    adjacency: np.ndarray,
    retained: Sequence[int],
    previous_set: set[int],
    observed_pairs: set[tuple[int, int]],
    covered_edges: set[tuple[int, int]],
    covered_nodes: set[int],
    rng: np.random.Generator,
    *,
    patch_size: int,
) -> tuple[int, ...] | None:
    nodes = list(retained)
    forbidden = previous_set - set(retained)
    while len(nodes) < patch_size:
        node_set = set(nodes)
        frontier = sorted(
            {
                int(neighbor)
                for node in nodes
                for neighbor in np.flatnonzero(adjacency[node])
                if int(neighbor) not in node_set and int(neighbor) not in forbidden
            }
        )
        if not frontier:
            return None
        scored = []
        for candidate in frontier:
            new_edge_increment = 0
            new_pair_increment = 0
            for node in nodes:
                pair = tuple(sorted((candidate, int(node))))
                new_pair_increment += int(pair not in observed_pairs)
                new_edge_increment += int(
                    adjacency[candidate, node] != 0 and pair not in covered_edges
                )
            deficit = sum(
                tuple(sorted((candidate, int(neighbor)))) not in covered_edges
                for neighbor in np.flatnonzero(adjacency[candidate])
            )
            score = (
                new_edge_increment,
                int(deficit),
                new_pair_increment,
                int(candidate not in covered_nodes),
                int(np.count_nonzero(adjacency[candidate])),
            )
            scored.append((score, candidate))
        best = max(score for score, _candidate in scored)
        choices = [candidate for score, candidate in scored if score == best]
        nodes.append(int(choices[int(rng.integers(len(choices)))]))
    return tuple(nodes)


def sample_marginal_candidate_cover(
    adjacency: np.ndarray,
    rng: np.random.Generator,
    *,
    n_patches: int,
    patch_size: int,
    target_overlap: int,
    retained_beam: int = 32,
    candidate_restarts: int = 2,
    allow_partial: bool = False,
) -> PatchCover:
    """Approximate one-step marginal maximization with retained-subset beams."""
    validate_simple_adjacency(adjacency)
    if retained_beam < 1 or candidate_restarts < 1:
        raise ValueError("beam and restarts must be positive")
    first_cover = sample_edge_target_bridge_cover(
        adjacency,
        rng,
        n_patches=1,
        patch_size=patch_size,
        target_overlap=target_overlap,
    )
    patches = [first_cover.patches[0]]
    observed_pairs = _pairs(patches[0].node_ids)
    covered_edges = {
        pair for pair in observed_pairs if adjacency[pair[0], pair[1]] != 0
    }
    covered_nodes = set(patches[0].node_ids)

    def register(nodes: Sequence[int]) -> None:
        pairs = _pairs(nodes)
        observed_pairs.update(pairs)
        covered_edges.update(
            pair for pair in pairs if adjacency[pair[0], pair[1]] != 0
        )
        covered_nodes.update(nodes)

    while len(patches) < n_patches:
        previous = patches[-1]
        previous_set = set(previous.node_ids)
        retained_candidates = []
        for retained in combinations(previous.node_ids, target_overlap):
            induced = adjacency[np.ix_(retained, retained)]
            if not is_connected(induced):
                continue
            potential = _retained_potential(
                adjacency,
                retained,
                previous_set,
                covered_edges,
                covered_nodes,
            )
            retained_candidates.append((potential, retained))
        retained_candidates.sort(key=lambda item: item[0], reverse=True)
        retained_candidates = retained_candidates[:retained_beam]
        completed = []
        for _potential, retained in retained_candidates:
            for _restart in range(candidate_restarts):
                candidate = _greedy_fill(
                    adjacency,
                    retained,
                    previous_set,
                    observed_pairs,
                    covered_edges,
                    covered_nodes,
                    rng,
                    patch_size=patch_size,
                )
                if candidate is not None:
                    completed.append(candidate)
        if not completed:
            if allow_partial:
                break
            raise RuntimeError("marginal candidate beam produced no connected next patch")
        scores = [
            _candidate_score(
                adjacency, candidate, observed_pairs, covered_edges, covered_nodes
            )
            for candidate in completed
        ]
        best = max(scores)
        choices = [
            candidate for score, candidate in zip(scores, completed) if score == best
        ]
        selected = choices[int(rng.integers(len(choices)))]
        new_nodes = [node for node in selected if node not in previous_set]
        center = new_nodes[0] if new_nodes else selected[0]
        patches.append(_make_patch(adjacency, selected, center))
        register(selected)
    return _make_cover("marginal_candidate_beam", patches)

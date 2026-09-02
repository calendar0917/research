"""Direct capped n-hop sampler diagnostics for raw IMDB-BINARY.

This module is intentionally pre-KSVD.  It separates fixed-size node selection
from exact rooted canonical vectorization and exposes tie-breaking dependence.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import numpy as np

from .from_scratch_unplanted_representation import (
    adjacency_to_upper_vector,
    relabel_adjacency_and_order,
    rooted_exact_canonical_vector,
    validate_simple_adjacency,
)
from .from_scratch_unplanted_signal import simple_graph_statistics
from .imdb_walk_substrate import TUStructureGraph


SELECTOR_MODES = ("n_id", "n_degree", "n_signature")


@dataclass(frozen=True)
class SelectedNhopPatch:
    root: int
    node_ids: tuple[int, ...]
    ranked_vector: np.ndarray
    canonical_vector: np.ndarray
    cutoff_tied: bool


@dataclass(frozen=True)
class NhopPatchGraph:
    graph_index: int
    label: int
    roots: tuple[int, ...]
    patches: dict[str, tuple[SelectedNhopPatch, ...]]
    canonical_mean_std: dict[str, np.ndarray]
    graph_statistics: np.ndarray


def shortest_path_distances(adjacency: np.ndarray, root: int) -> np.ndarray:
    validate_simple_adjacency(adjacency)
    n_nodes = adjacency.shape[0]
    if not 0 <= root < n_nodes:
        raise ValueError("root is outside the graph")
    distances = np.full(n_nodes, -1, dtype=np.int64)
    distances[root] = 0
    queue = [int(root)]
    for node in queue:
        for neighbor in np.flatnonzero(adjacency[node]):
            neighbor = int(neighbor)
            if distances[neighbor] < 0:
                distances[neighbor] = distances[node] + 1
                queue.append(neighbor)
    if np.any(distances < 0):
        raise ValueError("direct n-hop selector currently requires connected graphs")
    return distances


def _primary_selector_key(
    adjacency: np.ndarray,
    root: int,
    node: int,
    distances: np.ndarray,
    mode: str,
) -> tuple[int, ...]:
    distance = int(distances[node])
    if mode == "n_id":
        return (distance,)
    degrees = np.sum(adjacency, axis=1).astype(np.int64)
    if mode == "n_degree":
        return (distance, -int(degrees[node]))
    if mode == "n_signature":
        common_with_root = int(np.dot(adjacency[root], adjacency[node]))
        neighbor_degree_sum = int(np.sum(degrees[np.flatnonzero(adjacency[node])]))
        return (
            distance,
            -common_with_root,
            -int(degrees[node]),
            -neighbor_degree_sum,
        )
    raise ValueError(f"unknown selector mode: {mode}")


def select_fixed_nhop_patch(
    adjacency: np.ndarray,
    root: int,
    *,
    patch_size: int = 7,
    mode: str = "n_signature",
) -> SelectedNhopPatch:
    """Take root plus the first distance-ranked nodes and canonicalize exactly.

    Node ID is the final deterministic tie-break.  It is deliberately not
    hidden: ``cutoff_tied`` and the relabel audit quantify its effect.
    """
    validate_simple_adjacency(adjacency)
    if patch_size < 2 or patch_size > adjacency.shape[0]:
        raise ValueError("patch_size must lie in [2, n_nodes]")
    if mode not in SELECTOR_MODES:
        raise ValueError(f"unknown selector mode: {mode}")
    distances = shortest_path_distances(adjacency, root)
    candidates = [node for node in range(adjacency.shape[0]) if node != root]
    primary = {
        node: _primary_selector_key(adjacency, root, node, distances, mode)
        for node in candidates
    }
    ordered = sorted(candidates, key=lambda node: primary[node] + (int(node),))
    selected_tail = ordered[: patch_size - 1]
    cutoff_tied = bool(
        len(ordered) >= patch_size
        and primary[ordered[patch_size - 2]] == primary[ordered[patch_size - 1]]
    )
    node_ids = (int(root),) + tuple(int(node) for node in selected_tail)
    induced = adjacency[np.ix_(node_ids, node_ids)].astype(np.int8, copy=True)
    canonical = rooted_exact_canonical_vector(induced)
    return SelectedNhopPatch(
        root=int(root),
        node_ids=node_ids,
        ranked_vector=adjacency_to_upper_vector(induced),
        canonical_vector=canonical.vector,
        cutoff_tied=cutoff_tied,
    )


def extract_nhop_patch_graphs(
    graphs: Sequence[TUStructureGraph],
    *,
    sampling_seed: int = 20260731,
    patch_size: int = 7,
    max_patches_per_graph: int = 24,
) -> tuple[NhopPatchGraph, ...]:
    graphs = tuple(graphs)
    sequences = np.random.SeedSequence(int(sampling_seed)).spawn(len(graphs))
    result = []
    for graph, sequence in zip(graphs, sequences):
        rng = np.random.default_rng(sequence)
        n_patches = min(int(max_patches_per_graph), graph.adjacency.shape[0])
        roots = tuple(
            int(value)
            for value in rng.choice(graph.adjacency.shape[0], size=n_patches, replace=False)
        )
        patches = {
            mode: tuple(
                select_fixed_nhop_patch(
                    graph.adjacency, root, patch_size=patch_size, mode=mode
                )
                for root in roots
            )
            for mode in SELECTOR_MODES
        }
        canonical_mean_std = {}
        for mode, mode_patches in patches.items():
            values = np.stack([patch.canonical_vector for patch in mode_patches], axis=0)
            canonical_mean_std[mode] = np.concatenate(
                [np.mean(values, axis=0), np.std(values, axis=0, ddof=0)]
            )
        result.append(
            NhopPatchGraph(
                graph_index=int(graph.index),
                label=int(graph.label),
                roots=roots,
                patches=patches,
                canonical_mean_std=canonical_mean_std,
                graph_statistics=simple_graph_statistics(graph.adjacency),
            )
        )
    return tuple(result)


def _summary(values: np.ndarray) -> dict[str, float | int]:
    values = np.asarray(values)
    return {
        "minimum": int(np.min(values)),
        "median": float(np.median(values)),
        "mean": float(np.mean(values)),
        "p75": float(np.quantile(values, 0.75)),
        "p90": float(np.quantile(values, 0.90)),
        "p95": float(np.quantile(values, 0.95)),
        "maximum": int(np.max(values)),
    }


def direct_ego_size_feasibility(
    graphs: Iterable[TUStructureGraph], *, patch_size: int = 7
) -> dict[str, Any]:
    graphs = tuple(graphs)
    by_radius: dict[str, Any] = {}
    for radius in (1, 2):
        sizes = []
        full_graph = []
        for graph in graphs:
            for root in range(graph.adjacency.shape[0]):
                distances = shortest_path_distances(graph.adjacency, root)
                size = int(np.count_nonzero(distances <= radius))
                sizes.append(size)
                full_graph.append(size == graph.adjacency.shape[0])
        array = np.asarray(sizes, dtype=np.int64)
        by_radius[str(radius)] = {
            "root_count": int(array.size),
            "size": _summary(array),
            "fraction_smaller_than_patch_size": float(np.mean(array < patch_size)),
            "fraction_equal_to_patch_size": float(np.mean(array == patch_size)),
            "fraction_larger_than_patch_size": float(np.mean(array > patch_size)),
            "fraction_equal_to_whole_graph": float(np.mean(full_graph)),
        }
    return {"patch_size": int(patch_size), "radii": by_radius}


def _effective_count(keys: list[tuple[int, ...]]) -> float:
    _unique, counts = np.unique(np.asarray(keys, dtype=np.int8), axis=0, return_counts=True)
    probabilities = counts.astype(np.float64) / float(np.sum(counts))
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def nhop_substrate_summary(
    examples: Sequence[NhopPatchGraph], *, patch_size: int = 7
) -> dict[str, Any]:
    examples = tuple(examples)
    dimension = patch_size * (patch_size - 1) // 2
    output = {}
    for mode in SELECTOR_MODES:
        matrices = [
            np.stack([patch.canonical_vector for patch in example.patches[mode]], axis=0)
            for example in examples
        ]
        values = np.concatenate(matrices, axis=0).astype(np.int8)
        keys = [tuple(int(value) for value in row) for row in values]
        unique, counts = np.unique(values, axis=0, return_counts=True)
        edge_counts = np.sum(values, axis=1)
        within_unique = np.asarray(
            [len(set(tuple(int(x) for x in row) for row in matrix)) / matrix.shape[0] for matrix in matrices],
            dtype=np.float64,
        )
        cutoff_ties = [
            patch.cutoff_tied
            for example in examples
            for patch in example.patches[mode]
        ]
        output[mode] = {
            "patch_count": int(values.shape[0]),
            "canonical_unique_signature_count": int(unique.shape[0]),
            "canonical_effective_signature_count": _effective_count(keys),
            "dominant_canonical_signature_fraction": float(np.max(counts) / values.shape[0]),
            "clique_fraction": float(np.mean(edge_counts == dimension)),
            "tree_fraction": float(np.mean(edge_counts == patch_size - 1)),
            "patch_edge_mean": float(np.mean(edge_counts)),
            "cutoff_tie_fraction": float(np.mean(cutoff_ties)),
            "within_graph_canonical_unique_fraction": {
                "minimum": float(np.min(within_unique)),
                "median": float(np.median(within_unique)),
                "mean": float(np.mean(within_unique)),
            },
        }
    return output


def relabel_selector_audit(
    graphs: Sequence[TUStructureGraph],
    examples: Sequence[NhopPatchGraph],
    *,
    seed: int = 20260801,
    graph_limit: int = 100,
    permutations_per_graph: int = 3,
    patch_size: int = 7,
) -> dict[str, Any]:
    graph_by_index = {int(graph.index): graph for graph in graphs}
    rng = np.random.default_rng(int(seed))
    counters = {
        mode: {"comparisons": 0, "set": 0, "ranked": 0, "canonical": 0}
        for mode in SELECTOR_MODES
    }
    for example in tuple(examples)[:graph_limit]:
        graph = graph_by_index[example.graph_index]
        for _ in range(permutations_per_graph):
            permutation = rng.permutation(graph.adjacency.shape[0])
            inverse = np.empty(graph.adjacency.shape[0], dtype=np.int64)
            inverse[permutation] = np.arange(graph.adjacency.shape[0])
            relabeled = graph.adjacency[np.ix_(permutation, permutation)]
            for mode in SELECTOR_MODES:
                for original in example.patches[mode]:
                    mapped_root = int(inverse[original.root])
                    candidate = select_fixed_nhop_patch(
                        relabeled, mapped_root, patch_size=patch_size, mode=mode
                    )
                    mapped_old_nodes = tuple(int(permutation[node]) for node in candidate.node_ids)
                    counter = counters[mode]
                    counter["comparisons"] += 1
                    counter["set"] += int(set(mapped_old_nodes) == set(original.node_ids))
                    counter["ranked"] += int(
                        np.array_equal(candidate.ranked_vector, original.ranked_vector)
                    )
                    counter["canonical"] += int(
                        np.array_equal(candidate.canonical_vector, original.canonical_vector)
                    )
    result = {}
    for mode, counter in counters.items():
        n = counter["comparisons"]
        result[mode] = {
            "comparisons": int(n),
            "selected_abstract_set_exact_match_rate": float(counter["set"] / n),
            "ranked_order_vector_exact_match_rate": float(counter["ranked"] / n),
            "rooted_canonical_vector_exact_match_rate": float(counter["canonical"] / n),
        }
    return {
        "graphs_checked": int(min(graph_limit, len(examples))),
        "permutations_per_graph": int(permutations_per_graph),
        "modes": result,
    }


def nhop_feature_matrices(
    examples: Sequence[NhopPatchGraph],
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    examples = tuple(examples)
    stats = np.stack([example.graph_statistics for example in examples], axis=0)
    matrices = {"stats": stats}
    for mode in SELECTOR_MODES:
        values = np.stack([example.canonical_mean_std[mode] for example in examples], axis=0)
        matrices[f"{mode}_canonical_mean_std"] = values
        matrices[f"stats_plus_{mode}_canonical_mean_std"] = np.column_stack([stats, values])
    labels = np.asarray([example.label for example in examples], dtype=np.int64)
    return matrices, labels

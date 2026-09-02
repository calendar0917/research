"""R0 utilities for a clean transfer of the frozen WALK patch signal to IMDB.

This module is intentionally independent of the historical REAL_STRUCTURE
pipeline.  It loads TU Dortmund text files directly, audits exact-isomorphism
repetition, and extracts the six-node first-discovery-order adjacency signal
selected by the from-scratch U0-R gate.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np

from .from_scratch_unplanted_representation import (
    adjacency_to_upper_vector,
    relabel_adjacency_and_order,
    rooted_exact_canonical_vector,
    sample_graph_walk_patches,
    validate_simple_adjacency,
)
from .from_scratch_unplanted_signal import simple_graph_statistics


@dataclass(frozen=True)
class TUStructureGraph:
    index: int
    label: int
    adjacency: np.ndarray


@dataclass(frozen=True)
class IsomorphismGroup:
    group_id: int
    member_indices: tuple[int, ...]
    labels: tuple[int, ...]
    signature: str

    @property
    def label_conflict(self) -> bool:
        return len(set(self.labels)) > 1


@dataclass(frozen=True)
class IMDBPatchGraph:
    graph_index: int
    label: int
    n_nodes: int
    n_edges: int
    n_patches: int
    root_coverage: float
    walk_vectors: np.ndarray
    canonical_vectors: np.ndarray
    edge_counts: np.ndarray
    features: dict[str, np.ndarray]
    graph_statistics: np.ndarray


def summarize_walk_patch_vectors(
    walk_vectors: np.ndarray,
    canonical_vectors: np.ndarray,
    *,
    patch_size: int,
) -> dict[str, np.ndarray]:
    """Generic counterpart of the frozen six-node U0-P summaries."""
    walk_vectors = np.asarray(walk_vectors, dtype=np.float64)
    canonical_vectors = np.asarray(canonical_vectors, dtype=np.float64)
    dimension = patch_size * (patch_size - 1) // 2
    if walk_vectors.ndim != 2 or walk_vectors.shape[1] != dimension:
        raise ValueError("walk vectors have the wrong dimension for patch_size")
    if canonical_vectors.shape != walk_vectors.shape:
        raise ValueError("canonical vectors must match walk vectors")
    edge_counts = np.sum(walk_vectors, axis=1).astype(np.int64)
    minimum_edges = patch_size - 1
    maximum_edges = dimension
    histogram = np.bincount(
        edge_counts - minimum_edges, minlength=maximum_edges - minimum_edges + 1
    ).astype(np.float64)
    histogram /= float(walk_vectors.shape[0])
    return {
        "walk_mean_std": np.concatenate(
            [np.mean(walk_vectors, axis=0), np.std(walk_vectors, axis=0, ddof=0)]
        ),
        "canonical_mean_std": np.concatenate(
            [np.mean(canonical_vectors, axis=0), np.std(canonical_vectors, axis=0, ddof=0)]
        ),
        "edge_count_histogram": histogram,
    }


def _variant_directory(dataset_root: Path, cleaned: bool) -> Path:
    return dataset_root / ("raw_cleaned" if cleaned else "raw")


def load_tu_structure_text(
    dataset_root: str | Path,
    *,
    cleaned: bool = False,
) -> tuple[TUStructureGraph, ...]:
    """Load an unlabeled-structure TU dataset without torch/torch-geometric."""
    dataset_root = Path(dataset_root)
    name = dataset_root.name
    source = _variant_directory(dataset_root, cleaned)
    indicators = np.loadtxt(source / f"{name}_graph_indicator.txt", dtype=np.int64)
    raw_labels = np.loadtxt(source / f"{name}_graph_labels.txt", dtype=np.int64)
    edges = np.loadtxt(source / f"{name}_A.txt", delimiter=",", dtype=np.int64)
    if indicators.ndim != 1 or raw_labels.ndim != 1:
        raise ValueError("TU graph indicators and labels must be vectors")
    if edges.ndim == 1:
        edges = edges.reshape(1, 2)
    if edges.ndim != 2 or edges.shape[1] != 2:
        raise ValueError("TU edge file must contain two columns")

    unique_labels = sorted(int(value) for value in np.unique(raw_labels))
    label_map = {value: index for index, value in enumerate(unique_labels)}
    labels = np.asarray([label_map[int(value)] for value in raw_labels], dtype=np.int64)
    n_graphs = int(labels.size)
    counts = np.bincount(indicators, minlength=n_graphs + 1)[1:]
    offsets = np.concatenate([np.zeros(1, dtype=np.int64), np.cumsum(counts)])
    adjacencies = [np.zeros((int(n), int(n)), dtype=np.int8) for n in counts]

    for source_id, target_id in edges:
        source_zero = int(source_id) - 1
        target_zero = int(target_id) - 1
        graph_index = int(indicators[source_zero]) - 1
        if int(indicators[target_zero]) - 1 != graph_index:
            raise ValueError("TU edge crosses graph boundaries")
        local_source = source_zero - int(offsets[graph_index])
        local_target = target_zero - int(offsets[graph_index])
        if local_source == local_target:
            continue
        adjacencies[graph_index][local_source, local_target] = 1
        adjacencies[graph_index][local_target, local_source] = 1

    result = []
    for index, (label, adjacency) in enumerate(zip(labels, adjacencies)):
        validate_simple_adjacency(adjacency)
        result.append(TUStructureGraph(index=index, label=int(label), adjacency=adjacency))
    return tuple(result)


def _to_networkx(adjacency: np.ndarray):
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover - exercised by the runner environment
        raise RuntimeError("exact isomorphism audit requires networkx") from exc
    graph = nx.Graph()
    graph.add_nodes_from(range(adjacency.shape[0]))
    left, right = np.nonzero(np.triu(adjacency, k=1))
    graph.add_edges_from((int(u), int(v)) for u, v in zip(left, right))
    return graph


def exact_isomorphism_groups(graphs: Iterable[TUStructureGraph]) -> tuple[IsomorphismGroup, ...]:
    """Group exactly isomorphic graphs after a WL/degrees candidate filter."""
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise RuntimeError("exact isomorphism audit requires networkx") from exc

    graphs = tuple(graphs)
    nx_graphs = [_to_networkx(item.adjacency) for item in graphs]
    buckets: dict[tuple[Any, ...], list[int]] = {}
    signatures: dict[tuple[Any, ...], str] = {}
    for position, (item, graph) in enumerate(zip(graphs, nx_graphs)):
        degrees = tuple(sorted(int(value) for _, value in graph.degree()))
        wl_hash = nx.weisfeiler_lehman_graph_hash(graph, iterations=4)
        key = (graph.number_of_nodes(), graph.number_of_edges(), degrees, wl_hash)
        buckets.setdefault(key, []).append(position)
        signatures[key] = f"n{key[0]}|m{key[1]}|wl{wl_hash}"

    raw_groups: list[tuple[list[int], str]] = []
    for key in sorted(buckets, key=lambda value: (value[0], value[1], value[3], value[2])):
        representatives: list[tuple[int, list[int]]] = []
        for position in buckets[key]:
            for representative, members in representatives:
                if nx.is_isomorphic(nx_graphs[position], nx_graphs[representative]):
                    members.append(position)
                    break
            else:
                representatives.append((position, [position]))
        for _, members in representatives:
            raw_groups.append((members, signatures[key]))

    raw_groups.sort(key=lambda item: min(graphs[position].index for position in item[0]))
    groups = []
    for group_id, (positions, signature) in enumerate(raw_groups):
        member_indices = tuple(int(graphs[position].index) for position in positions)
        labels = tuple(int(graphs[position].label) for position in positions)
        groups.append(
            IsomorphismGroup(
                group_id=group_id,
                member_indices=member_indices,
                labels=labels,
                signature=signature,
            )
        )
    return tuple(groups)


def isomorphism_summary(groups: Iterable[IsomorphismGroup], n_graphs: int) -> dict[str, Any]:
    groups = tuple(groups)
    duplicate_groups = tuple(group for group in groups if len(group.member_indices) > 1)
    conflict_groups = tuple(group for group in groups if group.label_conflict)
    consistent_groups = tuple(group for group in groups if not group.label_conflict)
    sizes = np.asarray([len(group.member_indices) for group in groups], dtype=np.int64)
    majority_correct = sum(
        max(group.labels.count(label) for label in set(group.labels)) for group in groups
    )
    return {
        "graph_count": int(n_graphs),
        "exact_structure_group_count": int(len(groups)),
        "structure_only_deterministic_empirical_ceiling_accuracy": float(majority_correct / n_graphs),
        "structure_only_unavoidable_conflict_errors": int(n_graphs - majority_correct),
        "singleton_group_count": int(np.count_nonzero(sizes == 1)),
        "duplicate_group_count": int(len(duplicate_groups)),
        "graphs_in_duplicate_groups": int(sum(len(group.member_indices) for group in duplicate_groups)),
        "largest_group_size": int(np.max(sizes)),
        "label_conflict_group_count": int(len(conflict_groups)),
        "graphs_in_label_conflict_groups": int(sum(len(group.member_indices) for group in conflict_groups)),
        "label_consistent_group_count": int(len(consistent_groups)),
        "label_consistent_representative_counts": {
            str(label): int(sum(group.labels[0] == label for group in consistent_groups))
            for label in (0, 1)
        },
        "group_size_histogram": {
            str(size): int(np.count_nonzero(sizes == size)) for size in sorted(set(sizes.tolist()))
        },
    }


def extract_walk_patch_graphs(
    graphs: Iterable[TUStructureGraph],
    *,
    sampling_seed: int = 20260731,
    patch_size: int = 6,
    max_patches_per_graph: int = 24,
) -> tuple[IMDBPatchGraph, ...]:
    """Extract one patch per sampled root, using all roots when n <= the cap."""
    graphs = tuple(graphs)
    sequences = np.random.SeedSequence(int(sampling_seed)).spawn(len(graphs))
    result = []
    for item, sequence in zip(graphs, sequences):
        n_nodes = int(item.adjacency.shape[0])
        n_patches = min(int(max_patches_per_graph), n_nodes)
        patches = sample_graph_walk_patches(
            item.adjacency,
            np.random.default_rng(sequence),
            n_patches=n_patches,
            patch_size=patch_size,
        )
        walk_vectors = np.stack([patch.walk_order_vector for patch in patches], axis=0)
        canonical_vectors = np.stack([patch.canonical_vector for patch in patches], axis=0)
        edge_counts = np.sum(walk_vectors, axis=1).astype(np.int64)
        result.append(
            IMDBPatchGraph(
                graph_index=int(item.index),
                label=int(item.label),
                n_nodes=n_nodes,
                n_edges=int(np.sum(item.adjacency) // 2),
                n_patches=n_patches,
                root_coverage=float(n_patches / n_nodes),
                walk_vectors=walk_vectors,
                canonical_vectors=canonical_vectors,
                edge_counts=edge_counts,
                features=summarize_walk_patch_vectors(
                    walk_vectors, canonical_vectors, patch_size=patch_size
                ),
                graph_statistics=simple_graph_statistics(item.adjacency),
            )
        )
    return tuple(result)


def _row_keys(values: np.ndarray) -> list[tuple[int, ...]]:
    return [tuple(int(value) for value in row) for row in np.asarray(values, dtype=np.int8)]


def _effective_count(keys: list[tuple[int, ...]]) -> float:
    _, counts = np.unique(np.asarray(keys, dtype=np.int8), axis=0, return_counts=True)
    probabilities = counts.astype(np.float64) / float(np.sum(counts))
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1e-300))))
    return float(np.exp(entropy))


def patch_substrate_summary(examples: Iterable[IMDBPatchGraph]) -> dict[str, Any]:
    examples = tuple(examples)
    walk = np.concatenate([item.walk_vectors for item in examples], axis=0)
    canonical = np.concatenate([item.canonical_vectors for item in examples], axis=0)
    edge_counts = np.concatenate([item.edge_counts for item in examples], axis=0)
    dimension = int(walk.shape[1])
    patch_size = int((1 + round((1 + 8 * dimension) ** 0.5)) // 2)
    if patch_size * (patch_size - 1) // 2 != dimension:
        raise ValueError("patch vector dimension is not triangular")
    walk_keys = _row_keys(walk)
    canonical_keys = _row_keys(canonical)
    walk_unique_by_graph = np.asarray(
        [len(set(_row_keys(item.walk_vectors))) / item.n_patches for item in examples], dtype=np.float64
    )
    canonical_unique_by_graph = np.asarray(
        [len(set(_row_keys(item.canonical_vectors))) / item.n_patches for item in examples], dtype=np.float64
    )
    canonical_values, canonical_counts = np.unique(
        canonical.astype(np.int8), axis=0, return_counts=True
    )
    dominant_index = int(np.argmax(canonical_counts))
    root_coverage = np.asarray([item.root_coverage for item in examples], dtype=np.float64)
    patch_counts = np.asarray([item.n_patches for item in examples], dtype=np.int64)
    n_nodes = np.asarray([item.n_nodes for item in examples], dtype=np.int64)
    return {
        "graph_count": int(len(examples)),
        "patch_count": int(walk.shape[0]),
        "patches_per_graph": {
            "minimum": int(np.min(patch_counts)),
            "median": float(np.median(patch_counts)),
            "maximum": int(np.max(patch_counts)),
        },
        "root_coverage": {
            "minimum": float(np.min(root_coverage)),
            "median": float(np.median(root_coverage)),
            "mean": float(np.mean(root_coverage)),
            "full_coverage_graph_count": int(np.count_nonzero(root_coverage == 1.0)),
        },
        "node_count": {
            "minimum": int(np.min(n_nodes)),
            "median": float(np.median(n_nodes)),
            "mean": float(np.mean(n_nodes)),
            "maximum": int(np.max(n_nodes)),
        },
        "edge_count_histogram": {
            str(edge_count): int(np.count_nonzero(edge_counts == edge_count))
            for edge_count in range(patch_size - 1, dimension + 1)
        },
        "patch_edge_mean": float(np.mean(edge_counts)),
        "patch_size": int(patch_size),
        "signal_dimension": int(dimension),
        "tree_fraction": float(np.mean(edge_counts == patch_size - 1)),
        "clique_fraction": float(np.mean(edge_counts == dimension)),
        "near_clique_fraction_missing_at_most_one_edge": float(np.mean(edge_counts >= dimension - 1)),
        "walk_unique_vector_count": int(len(set(walk_keys))),
        "canonical_unique_signature_count": int(canonical_values.shape[0]),
        "walk_effective_signature_count": _effective_count(walk_keys),
        "canonical_effective_signature_count": _effective_count(canonical_keys),
        "dominant_canonical_signature_fraction": float(np.max(canonical_counts) / canonical.shape[0]),
        "dominant_canonical_signature": canonical_values[dominant_index].astype(int).tolist(),
        "within_graph_walk_unique_fraction": {
            "minimum": float(np.min(walk_unique_by_graph)),
            "median": float(np.median(walk_unique_by_graph)),
            "mean": float(np.mean(walk_unique_by_graph)),
        },
        "within_graph_canonical_unique_fraction": {
            "minimum": float(np.min(canonical_unique_by_graph)),
            "median": float(np.median(canonical_unique_by_graph)),
            "mean": float(np.mean(canonical_unique_by_graph)),
        },
    }


def mapped_trajectory_invariance_audit(
    graphs: Iterable[TUStructureGraph],
    examples: Iterable[IMDBPatchGraph],
    *,
    seed: int = 20260731,
    graph_limit: int = 100,
) -> dict[str, Any]:
    """Verify that a mapped walk trajectory has the same WALK-order vector."""
    graph_by_index = {item.index: item for item in graphs}
    examples = tuple(examples)[:graph_limit]
    rng = np.random.default_rng(int(seed))
    mismatch_count = 0
    compared = 0
    maximum_l1 = 0.0
    for example in examples:
        graph = graph_by_index[example.graph_index]
        permutation = rng.permutation(graph.adjacency.shape[0])
        # Recreate deterministic patch objects for this graph, then map their discovered order.
        patch_rng = np.random.default_rng(
            np.random.SeedSequence(int(seed)).spawn(len(graph_by_index))[example.graph_index]
        )
        dimension = int(example.walk_vectors.shape[1])
        patch_size = int((1 + round((1 + 8 * dimension) ** 0.5)) // 2)
        patches = sample_graph_walk_patches(
            graph.adjacency,
            patch_rng,
            n_patches=example.n_patches,
            patch_size=patch_size,
        )
        for patch in patches:
            relabeled, mapped_order = relabel_adjacency_and_order(
                graph.adjacency, patch.node_ids, permutation
            )
            mapped_induced = relabeled[np.ix_(mapped_order, mapped_order)]
            mapped_vector = adjacency_to_upper_vector(mapped_induced)
            l1 = float(np.sum(np.abs(mapped_vector - patch.walk_order_vector)))
            maximum_l1 = max(maximum_l1, l1)
            mismatch_count += int(l1 > 0.0)
            compared += 1
    return {
        "graphs_checked": int(len(examples)),
        "patches_checked": int(compared),
        "mismatch_count": int(mismatch_count),
        "maximum_l1_difference": float(maximum_l1),
        "passes_exact_mapped_trajectory_gate": bool(mismatch_count == 0),
    }


def feature_matrices(examples: Iterable[IMDBPatchGraph]) -> tuple[dict[str, np.ndarray], np.ndarray]:
    examples = tuple(examples)
    matrices = {
        "graph_stats": np.stack([item.graph_statistics for item in examples]),
        "walk_mean_std": np.stack([item.features["walk_mean_std"] for item in examples]),
        "canonical_mean_std": np.stack([item.features["canonical_mean_std"] for item in examples]),
        "edge_count_histogram": np.stack([item.features["edge_count_histogram"] for item in examples]),
    }
    for key in ("walk_mean_std", "canonical_mean_std", "edge_count_histogram"):
        matrices[f"graph_stats_plus_{key}"] = np.column_stack([matrices["graph_stats"], matrices[key]])
    labels = np.asarray([item.label for item in examples], dtype=np.int64)
    return matrices, labels

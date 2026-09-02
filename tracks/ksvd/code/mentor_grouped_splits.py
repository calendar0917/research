"""Deterministic graph folds and source-overlap audits for mentor subgraphs.

The split machinery deliberately keeps source identities outside all learned
representations.  Global source node IDs and root candidates are used only to
construct/audit graph folds; KSVD patch vectors remain rooted-canonical local
adjacency vectors.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from .data_mentor_subgraphs import MentorSubgraphBundle, STRATUM_NAMES


@dataclass(frozen=True)
class MentorFold:
    view: str
    fold_index: int
    train_indices: np.ndarray
    test_indices: np.ndarray


def _as_int_vector(values: Sequence[int] | np.ndarray, name: str) -> np.ndarray:
    result = np.asarray(values, dtype=np.int64)
    if result.ndim != 1:
        raise ValueError(f"{name} must be a 1-D array")
    return result


def balanced_group_folds(
    graph_indices: Sequence[int] | np.ndarray,
    strata: Sequence[int] | np.ndarray,
    groups: Sequence[int] | np.ndarray,
    *,
    n_splits: int = 3,
    seed: int = 20260807,
    view: str = "grouped",
) -> tuple[MentorFold, ...]:
    """Assign indivisible groups to density-balanced deterministic test folds."""
    indices = _as_int_vector(graph_indices, "graph_indices")
    strata_array = _as_int_vector(strata, "strata")
    groups_array = _as_int_vector(groups, "groups")
    if not (indices.size == strata_array.size == groups_array.size):
        raise ValueError("graph_indices, strata, and groups must have equal length")
    if indices.size == 0 or np.unique(indices).size != indices.size:
        raise ValueError("graph_indices must be nonempty and unique")
    if n_splits < 2:
        raise ValueError("n_splits must be at least two")
    if np.any((strata_array < 0) | (strata_array >= len(STRATUM_NAMES))):
        raise ValueError("strata contain an out-of-range value")

    unique_groups = np.unique(groups_array)
    if unique_groups.size < n_splits:
        raise ValueError("fewer groups than folds")
    rng = np.random.default_rng(int(seed))
    tie_order = rng.permutation(unique_groups.size)
    tie_rank = {
        int(unique_groups[position]): int(rank)
        for rank, position in enumerate(tie_order)
    }

    records: list[tuple[int, np.ndarray, np.ndarray, float]] = []
    for raw_group in unique_groups:
        group = int(raw_group)
        positions = np.flatnonzero(groups_array == group)
        histogram = np.bincount(
            strata_array[positions], minlength=len(STRATUM_NAMES)
        ).astype(np.int64)
        concentration = float(np.max(histogram) / positions.size)
        records.append((group, positions, histogram, concentration))
    records.sort(
        key=lambda item: (
            -int(item[1].size),
            -item[3],
            tie_rank[item[0]],
            item[0],
        )
    )

    target_graphs = indices.size / float(n_splits)
    total_strata = np.bincount(
        strata_array, minlength=len(STRATUM_NAMES)
    ).astype(np.float64)
    target_strata = total_strata / float(n_splits)
    graph_scale = max(target_graphs, 1.0)
    stratum_scale = np.maximum(target_strata, 1.0)
    fold_graphs = np.zeros(n_splits, dtype=np.float64)
    fold_strata = np.zeros((n_splits, len(STRATUM_NAMES)), dtype=np.float64)
    fold_positions: list[list[int]] = [[] for _ in range(n_splits)]

    def objective(candidate_fold: int, size: int, histogram: np.ndarray) -> float:
        counts = fold_graphs.copy()
        density = fold_strata.copy()
        counts[candidate_fold] += size
        density[candidate_fold] += histogram
        graph_error = np.sum(((counts - target_graphs) / graph_scale) ** 2)
        density_error = np.sum(
            ((density - target_strata[None, :]) / stratum_scale[None, :]) ** 2
        )
        return float(graph_error + density_error)

    for assignment_index, (_group, positions, histogram, _concentration) in enumerate(records):
        # Seed every fold, preventing an empty fold under pathological balance.
        candidates = (
            (assignment_index,)
            if assignment_index < n_splits
            else tuple(range(n_splits))
        )
        selected_fold = min(
            candidates,
            key=lambda fold: (
                objective(int(fold), int(positions.size), histogram),
                fold_graphs[int(fold)],
                int(fold),
            ),
        )
        fold_positions[int(selected_fold)].extend(int(value) for value in positions)
        fold_graphs[int(selected_fold)] += positions.size
        fold_strata[int(selected_fold)] += histogram

    folds = []
    all_indices = set(int(value) for value in indices)
    for fold_index, positions in enumerate(fold_positions):
        test = np.sort(indices[np.asarray(positions, dtype=np.int64)])
        train = np.asarray(sorted(all_indices - set(int(value) for value in test)), dtype=np.int64)
        if train.size == 0 or test.size == 0:
            raise RuntimeError("balanced assignment produced an empty train/test fold")
        folds.append(
            MentorFold(
                view=str(view),
                fold_index=fold_index,
                train_indices=train,
                test_indices=test,
            )
        )
    partition_audit = audit_fold_partition(
        folds,
        indices,
        groups_array_by_index=dict(zip(indices, groups_array)),
        require_group_integrity=True,
    )
    if not partition_audit["passed"]:
        raise RuntimeError("balanced assignment failed its partition/group audit")
    return tuple(folds)


def make_mentor_folds(
    graph_indices: Sequence[int] | np.ndarray,
    strata: Sequence[int] | np.ndarray,
    roots: Sequence[int] | np.ndarray,
    *,
    n_splits: int = 3,
    seed: int = 20260807,
) -> dict[str, tuple[MentorFold, ...]]:
    """Build random-reference and root-candidate-grouped split views."""
    indices = _as_int_vector(graph_indices, "graph_indices")
    strata_array = _as_int_vector(strata, "strata")
    roots_array = _as_int_vector(roots, "roots")
    random_groups = np.arange(indices.size, dtype=np.int64)
    return {
        "random_reference": balanced_group_folds(
            indices,
            strata_array,
            random_groups,
            n_splits=n_splits,
            seed=seed,
            view="random_reference",
        ),
        "root_candidate_grouped": balanced_group_folds(
            indices,
            strata_array,
            roots_array,
            n_splits=n_splits,
            seed=seed,
            view="root_candidate_grouped",
        ),
    }


def audit_fold_partition(
    folds: Sequence[MentorFold],
    graph_indices: Sequence[int] | np.ndarray,
    *,
    groups_array_by_index: dict[int, int] | None = None,
    require_group_integrity: bool = False,
) -> dict[str, object]:
    indices = _as_int_vector(graph_indices, "graph_indices")
    expected = set(int(value) for value in indices)
    test_memberships: list[int] = []
    rows = []
    for fold in folds:
        train = set(int(value) for value in fold.train_indices)
        test = set(int(value) for value in fold.test_indices)
        rows.append(
            {
                "fold_index": int(fold.fold_index),
                "train_graph_count": len(train),
                "test_graph_count": len(test),
                "train_test_intersection_count": len(train & test),
                "partition_matches": train | test == expected,
            }
        )
        test_memberships.extend(test)
    membership_counts = {
        index: test_memberships.count(index) for index in expected
    }
    unexpected_test_indices = set(test_memberships) - expected
    group_leakage = 0
    if groups_array_by_index is not None:
        for fold in folds:
            train_groups = {
                int(groups_array_by_index[int(index)]) for index in fold.train_indices
            }
            test_groups = {
                int(groups_array_by_index[int(index)]) for index in fold.test_indices
            }
            group_leakage += len(train_groups & test_groups)
    checks = {
        "fold_count_at_least_two": len(folds) >= 2,
        "each_fold_is_partition": all(
            row["partition_matches"]
            and row["train_test_intersection_count"] == 0
            and row["train_graph_count"] > 0
            and row["test_graph_count"] > 0
            for row in rows
        ),
        "each_graph_tested_once": not unexpected_test_indices
        and all(value == 1 for value in membership_counts.values()),
        "group_integrity": (not require_group_integrity) or group_leakage == 0,
    }
    return {
        "checks": checks,
        "passed": all(checks.values()),
        "group_leakage_count": int(group_leakage),
        "folds": rows,
    }


def _source_edges(bundle: MentorSubgraphBundle, graph_index: int) -> set[tuple[int, int]]:
    source = bundle.global_node_ids[int(graph_index)]
    adjacency = bundle.adjacency[int(graph_index)]
    left, right = np.nonzero(np.triu(adjacency, k=1))
    return {
        tuple(sorted((int(source[i]), int(source[j]))))
        for i, j in zip(left, right)
    }


def _fraction(values: set, reference: set) -> float:
    return float(len(values & reference) / max(len(values), 1))


def _jaccard(left: set, right: set) -> float:
    return float(len(left & right) / max(len(left | right), 1))


def _distribution(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        raise ValueError("cannot summarize an empty distribution")
    return {
        "mean": float(np.mean(array)),
        "p10": float(np.quantile(array, 0.10)),
        "minimum": float(np.min(array)),
    }


def audit_source_exposure(
    bundle: MentorSubgraphBundle,
    fold: MentorFold,
    strata_by_graph: np.ndarray,
) -> dict[str, object]:
    """Audit root, source-node, and source-edge train/test exposure."""
    train_indices = [int(value) for value in fold.train_indices]
    test_indices = [int(value) for value in fold.test_indices]
    strata_values = np.asarray(strata_by_graph, dtype=np.int64)
    all_indices = train_indices + test_indices
    if strata_values.ndim != 1 or any(
        index < 0 or index >= strata_values.size for index in all_indices
    ):
        raise ValueError("strata_by_graph must index every fold graph")
    train_nodes = {
        int(node) for index in train_indices for node in bundle.global_node_ids[index]
    }
    test_nodes = {
        int(node) for index in test_indices for node in bundle.global_node_ids[index]
    }
    train_edges = set().union(*(_source_edges(bundle, index) for index in train_indices))
    test_edges = set().union(*(_source_edges(bundle, index) for index in test_indices))
    train_roots = {int(bundle.roots[index]) for index in train_indices}
    test_roots = {int(bundle.roots[index]) for index in test_indices}
    node_per_graph = []
    edge_per_graph = []
    for index in test_indices:
        nodes = set(int(value) for value in bundle.global_node_ids[index])
        edges = _source_edges(bundle, index)
        node_per_graph.append(_fraction(nodes, train_nodes))
        edge_per_graph.append(_fraction(edges, train_edges))
    strata_counts = lambda values: {
        name: int(np.count_nonzero(strata_values[np.asarray(values, dtype=np.int64)] == position))
        for position, name in enumerate(STRATUM_NAMES)
    }
    return {
        "view": fold.view,
        "fold_index": int(fold.fold_index),
        "train_graph_count": len(train_indices),
        "test_graph_count": len(test_indices),
        "train_density_counts": strata_counts(train_indices),
        "test_density_counts": strata_counts(test_indices),
        "train_distinct_root_count": len(train_roots),
        "test_distinct_root_count": len(test_roots),
        "root_intersection_count": len(train_roots & test_roots),
        "train_distinct_source_node_count": len(train_nodes),
        "test_distinct_source_node_count": len(test_nodes),
        "test_source_node_seen_fraction": _fraction(test_nodes, train_nodes),
        "source_node_jaccard": _jaccard(train_nodes, test_nodes),
        "test_graph_source_node_exposure": _distribution(node_per_graph),
        "train_distinct_source_edge_count": len(train_edges),
        "test_distinct_source_edge_count": len(test_edges),
        "test_source_edge_seen_fraction": _fraction(test_edges, train_edges),
        "source_edge_jaccard": _jaccard(train_edges, test_edges),
        "test_graph_source_edge_exposure": _distribution(edge_per_graph),
    }

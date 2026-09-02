from __future__ import annotations

import numpy as np

from .data_mentor_subgraphs import MentorSubgraphBundle, STRATUM_NAMES
from .mentor_grouped_splits import (
    audit_fold_partition,
    audit_source_exposure,
    balanced_group_folds,
    make_mentor_folds,
)


def _synthetic_bundle() -> MentorSubgraphBundle:
    graph_count = 12
    n_nodes = 4
    adjacency = np.zeros((graph_count, n_nodes, n_nodes), dtype=np.uint8)
    global_node_ids = np.zeros((graph_count, n_nodes), dtype=np.int32)
    roots = np.zeros(graph_count, dtype=np.int32)
    for graph in range(graph_count):
        nodes = np.asarray([graph // 2, 20 + graph, 40 + graph, 60 + graph], dtype=np.int32)
        global_node_ids[graph] = nodes
        roots[graph] = nodes[0]
        for left, right in ((0, 1), (1, 2), (2, 3)):
            adjacency[graph, left, right] = 1
            adjacency[graph, right, left] = 1
    edge_counts = np.full(graph_count, 3, dtype=np.int32)
    return MentorSubgraphBundle(
        adjacency=adjacency,
        global_node_ids=global_node_ids,
        vocab=np.asarray([str(index) for index in range(100)]),
        roots=roots,
        edge_counts=edge_counts,
        avg_degrees=np.full(graph_count, 1.5, dtype=np.float64),
        density_percent=np.full(graph_count, 50.0, dtype=np.float64),
        metadata={},
    )


def test_balanced_group_folds_are_deterministic_and_keep_groups() -> None:
    indices = np.arange(12, dtype=np.int64)
    strata = np.asarray([0, 1, 2, 3, 4, 0, 1, 2, 3, 4, 0, 1], dtype=np.int64)
    groups = np.repeat(np.arange(6, dtype=np.int64), 2)
    left = balanced_group_folds(indices, strata, groups, seed=17)
    right = balanced_group_folds(indices, strata, groups, seed=17)
    assert [fold.test_indices.tolist() for fold in left] == [
        fold.test_indices.tolist() for fold in right
    ]
    audit = audit_fold_partition(
        left,
        indices,
        groups_array_by_index=dict(zip(indices, groups)),
        require_group_integrity=True,
    )
    assert audit["passed"]
    assert audit["group_leakage_count"] == 0
    assert max(len(fold.test_indices) for fold in left) <= 6


def test_make_views_and_source_exposure() -> None:
    bundle = _synthetic_bundle()
    indices = np.arange(bundle.n_graphs, dtype=np.int64)
    strata = np.asarray([index % len(STRATUM_NAMES) for index in indices], dtype=np.int64)
    views = make_mentor_folds(indices, strata, bundle.roots, seed=19)
    assert set(views) == {"random_reference", "root_candidate_grouped"}
    for fold in views["root_candidate_grouped"]:
        exposure = audit_source_exposure(bundle, fold, strata)
        assert exposure["root_intersection_count"] == 0
        assert 0.0 <= exposure["test_source_node_seen_fraction"] <= 1.0
        assert 0.0 <= exposure["test_source_edge_seen_fraction"] <= 1.0
        assert sum(exposure["test_density_counts"].values()) == len(fold.test_indices)


def test_invalid_inputs() -> None:
    indices = np.arange(3, dtype=np.int64)
    strata = np.zeros(3, dtype=np.int64)
    groups = np.zeros(3, dtype=np.int64)
    try:
        balanced_group_folds(indices, strata, groups, n_splits=3)
    except ValueError as exc:
        assert "fewer groups" in str(exc)
    else:
        raise AssertionError("expected fewer-groups failure")


if __name__ == "__main__":
    test_balanced_group_folds_are_deterministic_and_keep_groups()
    test_make_views_and_source_exposure()
    test_invalid_inputs()
    print("mentor_grouped_splits self-tests: PASS")

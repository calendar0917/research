from __future__ import annotations

import numpy as np

from ksvd_research.core import from_edges
from tracks.ksvd.experiments.luyin16.structural_role_terminal_audit import (
    _score_test_prediction_set,
    official_train_scaffold_splits,
    terminal_graph_views,
)


REPRESENTATION = {
    "radius": 2,
    "degree_bins": 8,
    "wl_rounds": 2,
    "node_role_bins": 64,
    "edge_role_bins": 32,
    "typed_edge_bins": 256,
    "max_centers_per_graph": None,
}


def _features(graph):
    nodes = np.zeros((graph.n, 9), dtype=np.int64)
    nodes[:, 0] = np.arange(graph.n) % 10
    nodes[:, 1] = np.arange(graph.n) % 4
    nodes[:, 3] = 5
    nodes[:, 4] = np.arange(graph.n) % 4
    nodes[:, 6] = np.arange(graph.n) % 5
    nodes[:, 7] = np.arange(graph.n) % 2
    edges = {
        edge: np.asarray([index % 4, index % 6, index % 2], dtype=np.int64)
        for index, edge in enumerate(graph.edges())
    }
    return nodes, edges


def test_terminal_views_are_the_pre_registered_dimensions() -> None:
    graph = from_edges(
        8,
        [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (5, 6), (5, 7)],
    )
    nodes, edges = _features(graph)
    views = terminal_graph_views(graph, nodes, edges, REPRESENTATION)
    assert views["t_a"].shape == (154,)
    assert views["f_centered"].shape == (3130,)
    np.testing.assert_array_equal(views["f_centered"][:154], views["t_a"])


def test_official_train_scaffold_rows_map_to_cache_order() -> None:
    archive = {
        "original_indices": np.asarray([10, 11, 12, 13, 14, 15], dtype=np.int64),
        "fold_0_train_indices": np.asarray([2, 3, 4, 5], dtype=np.int64),
        "fold_0_valid_indices": np.asarray([0, 1], dtype=np.int64),
        "fold_1_train_indices": np.asarray([0, 1, 4, 5], dtype=np.int64),
        "fold_1_valid_indices": np.asarray([2, 3], dtype=np.int64),
        "fold_2_train_indices": np.asarray([0, 1, 2, 3], dtype=np.int64),
        "fold_2_valid_indices": np.asarray([4, 5], dtype=np.int64),
    }
    cache_order = np.asarray([12, 10, 15, 11, 14, 13], dtype=np.int64)
    splits, metadata = official_train_scaffold_splits(archive, cache_order)
    assert len(splits) == 3
    assert [row["fold"] for row in metadata] == [0, 1, 2]
    for train, valid in splits:
        assert train.size == 4
        assert valid.size == 2
        assert not np.intersect1d(train, valid).size


def test_prediction_scoring_reports_seed_mean_and_ensemble() -> None:
    labels = np.asarray([0, 0, 1, 1], dtype=np.int64)
    rows = [{"seed": 0}, {"seed": 1}]
    predictions = [
        np.asarray([0.1, 0.3, 0.7, 0.9]),
        np.asarray([0.2, 0.4, 0.6, 0.8]),
    ]
    result = _score_test_prediction_set(rows, predictions, labels)
    assert result["mean_auc"] == 1.0
    assert result["seed_ensemble_auc"] == 1.0
    assert [row["test_auc"] for row in result["rows"]] == [1.0, 1.0]

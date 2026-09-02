from __future__ import annotations

import numpy as np

from .pipeline import graph_from_edge_index
from .run_beam8_nci1_chain_classification import (
    _graph_token_feature,
    _relation_matrices,
    _shuffle_permutation,
    prepare_graph,
)


def _disconnected_graph():
    edges = np.asarray(
        [
            [0, 1, 1, 2, 3, 4, 4, 5],
            [1, 0, 2, 1, 4, 3, 5, 4],
        ],
        dtype=np.int64,
    )
    return graph_from_edge_index(6, edges)


def test_disconnected_graph_creates_separate_segments() -> None:
    graph = _disconnected_graph()
    features = np.eye(6, dtype=np.float64)
    item = prepare_graph(
        0,
        graph,
        1,
        features,
        patch_size=3,
        overlap=1,
        retained_beam=2,
        edge_capacity_multiplier=1.5,
        seed=7,
    )
    assert len(set(item.segment_ids)) == 2
    previous, following, _overlap, _slot = _relation_matrices(item)
    for left in range(len(item.segment_ids)):
        for right in range(len(item.segment_ids)):
            if item.segment_ids[left] != item.segment_ids[right]:
                assert previous[left, right] == 0.0
                assert following[left, right] == 0.0


def test_shuffle_preserves_multiset_and_changes_binding() -> None:
    values = np.arange(24, dtype=np.float64).reshape(4, 6)
    permutation = _shuffle_permutation(4, seed=11)
    assert not np.array_equal(permutation, np.arange(4))
    assert np.array_equal(np.sort(values[permutation], axis=0), np.sort(values, axis=0))


def test_bag_is_invariant_to_patch_permutation() -> None:
    graph = _disconnected_graph()
    features = np.eye(6, dtype=np.float64)
    item = prepare_graph(
        0,
        graph,
        1,
        features,
        patch_size=3,
        overlap=1,
        retained_beam=2,
        edge_capacity_multiplier=1.5,
        seed=7,
    )
    tokens = np.arange(len(item.vectors) * 5, dtype=np.float64).reshape(
        len(item.vectors), 5
    )
    permutation = _shuffle_permutation(len(tokens), seed=13)
    direct = _graph_token_feature(item, tokens, chain=False)
    shuffled = _graph_token_feature(
        item, tokens, chain=False, permutation=permutation
    )
    assert np.allclose(direct, shuffled)


def test_disconnected_component_relabel_preserves_readout() -> None:
    graph = _disconnected_graph()
    features = np.eye(6, dtype=np.float64)
    base = prepare_graph(
        0, graph, 1, features, patch_size=3, overlap=1, retained_beam=2,
        edge_capacity_multiplier=1.5, seed=7,
    )
    permutation = np.asarray([3, 4, 5, 0, 1, 2], dtype=np.int64)
    edges = np.asarray(
        [[permutation[u], permutation[v]] for u, v in graph.edges()], dtype=np.int64
    ).T
    relabeled_graph = graph_from_edge_index(6, np.concatenate([edges, edges[::-1]], axis=1))
    relabeled_features = features[np.argsort(permutation)]
    relabeled = prepare_graph(
        0, relabeled_graph, 1, relabeled_features, patch_size=3, overlap=1,
        retained_beam=2, edge_capacity_multiplier=1.5, seed=7,
    )
    base_tokens = np.concatenate([base.vectors, base.node_histograms], axis=1)
    relabeled_tokens = np.concatenate(
        [relabeled.vectors, relabeled.node_histograms], axis=1
    )
    assert np.allclose(
        _graph_token_feature(base, base_tokens, chain=True),
        _graph_token_feature(relabeled, relabeled_tokens, chain=True),
    )


def run_all() -> None:
    test_disconnected_graph_creates_separate_segments()
    test_shuffle_preserves_multiset_and_changes_binding()
    test_bag_is_invariant_to_patch_permutation()
    test_disconnected_component_relabel_preserves_readout()


if __name__ == "__main__":
    run_all()

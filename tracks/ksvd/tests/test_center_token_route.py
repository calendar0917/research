from __future__ import annotations

import numpy as np

from ksvd_research.core import from_edges
from tracks.ksvd.experiments.luyin16.center_token_route import (
    ATTRIBUTE_WIDTH,
    CONDITIONAL_COMPACT_WIDTH,
    CONDITIONAL_FULL_WIDTH,
    DISTRIBUTION_WIDTH,
    ROLE_WIDTH,
    TOKEN_WIDTH,
    _batch_code_readouts,
    _code_readout,
    _conditional_features,
    _distribution_pool,
    _graph_token_features,
    _learn_token_dictionaries,
)
from tracks.ksvd.experiments.luyin16.center_token_route_checkpoint import (
    _components_to_views,
    _load_fold_checkpoint,
    _save_fold_checkpoint,
)
from tracks.ksvd.experiments.luyin16.patch_object_audit import relabel_graph_features


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
    nodes[:, 8] = (np.arange(graph.n) + 1) % 2
    edges = {
        edge: np.asarray([index % 4, index % 6, index % 2], dtype=np.int64)
        for index, edge in enumerate(graph.edges())
    }
    return nodes, edges


class _Bundle:
    def __init__(self, graph, nodes, edges):
        self.graphs = [graph]
        self.node_feats = [nodes]
        self.edge_feats = [edges]
        self.y = np.asarray([0], dtype=np.int64)


def test_conditional_and_distribution_schema_dimensions() -> None:
    rng = np.random.default_rng(7)
    topology = rng.normal(size=(9, ROLE_WIDTH)).astype(np.float32)
    attributes = rng.normal(size=(9, ATTRIBUTE_WIDTH)).astype(np.float32)
    roles = rng.integers(0, 64, size=9, dtype=np.int64)
    compact, full, meta = _conditional_features(topology, attributes, roles)
    assert compact.shape == (CONDITIONAL_COMPACT_WIDTH,)
    assert full.shape == (CONDITIONAL_FULL_WIDTH,)
    assert meta["compact_dimension"] == CONDITIONAL_COMPACT_WIDTH
    reference = {
        "q10": np.quantile(np.concatenate([topology, attributes], axis=1), 0.1, axis=0),
        "q90": np.quantile(np.concatenate([topology, attributes], axis=1), 0.9, axis=0),
        "norm_q90": np.asarray([1.0]),
        "role_counts": np.ones(64, dtype=np.float32),
        "rare_role_threshold": 2.0,
    }
    distribution = _distribution_pool(np.concatenate([topology, attributes], axis=1), roles, reference)
    assert distribution.shape == (DISTRIBUTION_WIDTH,)
    assert _distribution_pool(np.zeros((0, TOKEN_WIDTH), dtype=np.float32), np.zeros(0, dtype=np.int64), reference).shape == (DISTRIBUTION_WIDTH,)


def test_center_token_population_is_relabeling_invariant() -> None:
    graph = from_edges(8, [(0, 1), (0, 2), (1, 3), (1, 4), (2, 5), (5, 6), (5, 7)])
    nodes, edges = _features(graph)
    permutation = np.asarray([4, 2, 7, 1, 6, 5, 0, 3], dtype=np.int64)
    changed_graph, changed_nodes, changed_edges = relabel_graph_features(graph, nodes, edges, permutation)
    base = _graph_token_features(_Bundle(graph, nodes, edges), 0, REPRESENTATION, shuffle_seed=23)
    changed = _graph_token_features(_Bundle(changed_graph, changed_nodes, changed_edges), 0, REPRESENTATION, shuffle_seed=23)
    for key in ("topology", "attributes", "tokens", "roles", "shuffled_attributes"):
        np.testing.assert_allclose(base[key], changed[key], rtol=0.0, atol=1e-6)


def test_batch_code_readout_matches_graphwise_readout() -> None:
    rng = np.random.default_rng(11)
    tokens = rng.normal(size=(7, TOKEN_WIDTH)).astype(np.float32)
    dictionary = rng.normal(size=(TOKEN_WIDTH, 5)).astype(np.float32)
    offsets = np.asarray([0, 2, 7], dtype=np.int64)
    batch = _batch_code_readouts(tokens, offsets, dictionary, sparsity=2, n_jobs=1)
    from tracks.ksvd.experiments.luyin16.center_token_route import _encode_graph_codes

    expected = np.stack(
        [
            _code_readout(_encode_graph_codes(tokens[start:stop], dictionary, sparsity=2, n_jobs=1))
            for start, stop in zip(offsets[:-1], offsets[1:], strict=True)
        ],
        axis=0,
    )
    np.testing.assert_allclose(batch, expected, rtol=1e-5, atol=1e-5)


def test_dictionary_initial_and_final_have_token_width() -> None:
    rng = np.random.default_rng(19)
    tokens = rng.normal(size=(20, TOKEN_WIDTH)).astype(np.float32)
    initial, final, metadata = _learn_token_dictionaries(
        tokens,
        {"n_atoms": 4, "sparsity": 2, "iterations": 1},
        seed=3,
    )
    assert initial.shape == (TOKEN_WIDTH, 4)
    assert final.shape == (TOKEN_WIDTH, 4)
    assert metadata["n_train_tokens"] == tokens.shape[0]


def test_checkpoint_roundtrip_preserves_view_shapes(tmp_path) -> None:
    rng = np.random.default_rng(31)
    base = rng.normal(size=(5, 4)).astype(np.float32)
    blocks = {
        "distribution": rng.normal(size=(5, 3)).astype(np.float32),
        "conditional_compact": rng.normal(size=(5, 2)).astype(np.float32),
        "conditional_pca": rng.normal(size=(5, 1)).astype(np.float32),
        "ksvd_init": rng.normal(size=(5, 6)).astype(np.float32),
        "ksvd_final": rng.normal(size=(5, 6)).astype(np.float32),
        "conditional_shuffled_compact": rng.normal(size=(5, 2)).astype(np.float32),
        "ksvd_final_shuffled": rng.normal(size=(5, 6)).astype(np.float32),
    }
    views = {
        "s_marginal": base,
        "s_distribution": np.concatenate([base, blocks["distribution"]], axis=1),
        "s_conditional": np.concatenate([base, blocks["conditional_compact"]], axis=1),
        "s_conditional_pca": np.concatenate([base, blocks["conditional_pca"]], axis=1),
        "s_ksvd_init": np.concatenate([base, blocks["ksvd_init"]], axis=1),
        "s_ksvd_final": np.concatenate([base, blocks["ksvd_final"]], axis=1),
        "s_center_token_all": np.concatenate(
            [base, blocks["distribution"], blocks["conditional_compact"], blocks["ksvd_final"]], axis=1
        ),
    }
    null_views = {
        "conditional_shuffled": np.concatenate([base, blocks["conditional_shuffled_compact"]], axis=1),
        "ksvd_final_shuffled": np.concatenate([base, blocks["ksvd_final_shuffled"]], axis=1),
    }
    path = tmp_path / "fold.npz"
    _save_fold_checkpoint(
        path,
        "sig",
        views,
        null_views,
        np.asarray([0, 1, 0]),
        np.asarray([1, 0]),
        np.asarray([0, 1, 2]),
        np.asarray([3, 4]),
    )
    loaded = _load_fold_checkpoint(path, "sig")
    restored, restored_null = _components_to_views(loaded)
    for name in views:
        np.testing.assert_allclose(restored[name], views[name])
    for name in null_views:
        np.testing.assert_allclose(restored_null[name], null_views[name])

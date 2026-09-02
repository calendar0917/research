from __future__ import annotations

import numpy as np

from ksvd_research.core import from_edges
from ksvd_research.data import MolhivBundle
from ksvd_research.evaluation import GraphLevelConfig
from ksvd_research.features import build_ring_context_index
from tracks.ksvd.experiments.luyin16.fixed_feature_block_ablation import (
    build_feature_block_views,
)
from tracks.ksvd.experiments.luyin16.typed_slot_reconstruction_proxy import (
    TYPED_SLOT_DIM,
    typed_slot_vector,
)
from tracks.ksvd.experiments.luyin16.mentor_concept_replication import (
    TOPOLOGY_FEATURE_NAMES,
    _radius2_rooted_chem_vector,
    _radius2_rooted_topology_vector,
    _radius_ego_nodes,
    assemble_views,
    build_composition_features,
    collect_train_patch_matrix,
    experiment_status,
    fit_xgboost_views,
    learn_matched_dictionaries,
    typed_matrix_distribution_readout,
)
from tracks.ksvd.experiments.luyin16.mentor_ring_context_proxy import (
    assemble_mentor_views,
    context_mass_readout,
)


def test_experiment_status_distinguishes_smoke_development_and_full() -> None:
    assert experiment_status("mentor-concept-replication-v1-smoke", 300) == "smoke"
    assert experiment_status("mentor-concept-replication-v1-dev", 8000) == "development"
    assert experiment_status("mentor-concept-replication-v1", None) == "full"


def test_feature_block_ablation_has_explicit_nonoverlapping_blocks() -> None:
    composition = np.arange(2 * 10, dtype=np.float32).reshape(2, 10)
    initial = np.zeros((2, 3), dtype=np.float32)
    final = np.ones((2, 3), dtype=np.float32)
    schema = {"atom_dimensions": [2, 3], "bond_dimensions": [1, 4]}
    original_names = len(TOPOLOGY_FEATURE_NAMES)
    padded = np.zeros((2, original_names + 10), dtype=np.float32)
    padded[:, original_names:] = composition
    views, dimensions = build_feature_block_views(padded, initial, final, schema)
    assert dimensions == {"topology": original_names, "atom": 5, "bond": 5}
    assert views["chemistry_composition"].shape == (2, 10)
    assert views["topology_t_final"].shape == (2, original_names + 3)
    assert views["s_t_init"].shape == (2, original_names + 13)


def test_typed_matrix_distribution_readout_preserves_input_coordinates() -> None:
    matrix = np.asarray([[1.0, 3.0], [0.0, 2.0]], dtype=np.float64)
    readout = typed_matrix_distribution_readout(matrix)
    assert readout.shape == (24,)
    np.testing.assert_allclose(readout[:2], [2.0, 1.0])
    np.testing.assert_allclose(readout[-2:], [1.0, 0.5])


def test_radius2_rooted_chem_vector_is_fixed_width_and_centered() -> None:
    graph = from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    node_features = np.zeros((5, 9), dtype=np.int64)
    node_features[:, 0] = [1, 2, 3, 4, 5]
    edge_features = {edge: np.asarray([1, 0, 0], dtype=np.int64) for edge in graph.edges()}
    nodes = _radius_ego_nodes(graph, center=2, radius=2)
    vector = _radius2_rooted_chem_vector(
        graph,
        nodes,
        center=2,
        max_nodes=8,
        node_features=node_features,
        edge_features=edge_features,
    )
    assert vector.shape == (52,)
    assert np.isfinite(vector).all()
    # Center-first ordering means the first padded adjacency coordinate is the
    # center-to-first-shell edge for this path.
    assert vector[0] == 1.0


def test_radius2_rooted_topology_control_is_28d() -> None:
    graph = from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    nodes = _radius_ego_nodes(graph, center=2, radius=2)
    vector = _radius2_rooted_topology_vector(graph, nodes, center=2, max_nodes=8)
    assert vector.shape == (28,)
    assert np.isfinite(vector).all()
    assert vector[0] == 1.0


def test_typed_slot_proxy_has_natural_624_dimension() -> None:
    graph = from_edges(3, [(0, 1), (1, 2)])
    node_features = np.zeros((3, 9), dtype=np.int64)
    node_features[:, 0] = [1, 2, 3]
    edge_features = {edge: np.asarray([1, 0, 0], dtype=np.int64) for edge in graph.edges()}
    vector = typed_slot_vector(graph, {0, 1, 2}, node_features, edge_features)
    assert TYPED_SLOT_DIM == 624
    assert vector.shape == (624,)
    assert np.isfinite(vector).all()


def test_ring_context_proxy_has_mentor_shaped_dimensions() -> None:
    graph = from_edges(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (4, 5)])
    node_features = np.zeros((6, 9), dtype=np.int64)
    node_features[:5, 7] = 1
    node_features[:5, 8] = 1
    index = build_ring_context_index(graph, node_features)
    mask = index.patch_mask({0, 1, 2, 3, 4, 5})
    assert mask.tolist() == [True, True, False, True, False, True]

    codes = np.ones((64, 1), dtype=np.float64)
    mass = context_mass_readout(codes, mask[None, :])
    assert mass.shape == (325,)
    composition = np.zeros((2, 69), dtype=np.float32)
    reconstruction = np.zeros((2, 624), dtype=np.float32)
    contexts = np.zeros((2, 325), dtype=np.float32)
    views = assemble_mentor_views(composition, reconstruction, contexts)
    assert views["st"].shape == (2, 693)
    assert views["sta"].shape == (2, 1018)


def _synthetic_bundle() -> MolhivBundle:
    graphs = [
        from_edges(3, [(0, 1), (1, 2)]),
        from_edges(3, [(0, 1), (1, 2), (2, 0)]),
    ]
    node_feats = [np.zeros((graph.n, 9), dtype=np.int64) for graph in graphs]
    edge_feats = [{edge: np.zeros(3, dtype=np.int64) for edge in graph.edges()} for graph in graphs]
    return MolhivBundle(
        graphs=graphs,
        y=np.asarray([0.0, 1.0]),
        split={
            "train": np.asarray([0], dtype=np.int64),
            "valid": np.asarray([1], dtype=np.int64),
            "test": np.empty(0, dtype=np.int64),
        },
        smiles=None,
        meta={"name": "synthetic"},
        node_feats=node_feats,
        edge_feats=edge_feats,
    )


def test_composition_schema_is_explicit_and_finite() -> None:
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    matrix, names, meta = build_composition_features(_synthetic_bundle())
    expected = len(TOPOLOGY_FEATURE_NAMES) + sum(get_atom_feature_dims()) + sum(get_bond_feature_dims())
    assert matrix.shape == (2, expected)
    assert len(names) == expected
    assert meta["dimension"] == expected
    assert np.all(np.isfinite(matrix))
    assert matrix[1, names.index("triangle_count")] == 1.0


def test_train_patch_collection_never_uses_nontrain_graphs() -> None:
    matrices = [
        np.asarray([[1.0, 2.0], [0.0, 0.0]]),
        np.asarray([[999.0], [999.0]]),
        np.asarray([[3.0], [0.0]]),
    ]
    train = np.asarray([0, 2], dtype=np.int64)
    pooled, audit = collect_train_patch_matrix(matrices, train, maximum=10, seed=0)
    assert pooled.shape == (2, 3)
    assert 999.0 not in pooled
    assert audit["source_subset_of_train"] is True
    assert audit["n_train_graphs_contributing"] == 2


def test_matched_dictionaries_share_initialization_and_feature_views() -> None:
    rng = np.random.default_rng(7)
    patches = rng.normal(size=(6, 40))
    cfg = GraphLevelConfig(n_atoms=5, T=2, T_min=1, ksvd_iter=2, seed=3)
    initial, final, info = learn_matched_dictionaries(patches, cfg)
    assert initial.shape == final.shape == (6, 5)
    assert info["initial"]["n_iter"] == 0
    assert info["final"]["initialization"] == "provided"
    assert not np.allclose(initial, final)

    composition = np.zeros((4, 3), dtype=np.float32)
    t_initial = np.zeros((4, 5), dtype=np.float32)
    t_final = np.ones((4, 5), dtype=np.float32)
    views = assemble_views(
        composition,
        t_initial,
        t_final,
        ["s", "t_init", "t_final", "s_t_final"],
    )
    assert views["s"].shape == (4, 3)
    assert views["t_final"].shape == (4, 5)
    assert views["s_t_final"].shape == (4, 8)


def test_xgboost_runner_uses_validation_only_when_test_is_disabled() -> None:
    rng = np.random.default_rng(11)
    labels = np.asarray([0, 1] * 15, dtype=np.float64)
    signal = labels[:, None] + 0.05 * rng.normal(size=(30, 1))
    views = {"s": np.concatenate([signal, rng.normal(size=(30, 2))], axis=1)}
    split = {
        "train": np.arange(0, 18, dtype=np.int64),
        "valid": np.arange(18, 24, dtype=np.int64),
        "test": np.arange(24, 30, dtype=np.int64),
    }
    records, fitted = fit_xgboost_views(
        views,
        labels,
        split,
        {
            "seeds": [0],
            "n_estimators": 5,
            "max_depth": 2,
            "learning_rate": 0.2,
            "min_child_weight": 1,
            "subsample": 1.0,
            "colsample_bytree": 1.0,
            "reg_lambda": 1.0,
            "reg_alpha": 0.0,
            "n_jobs": 1,
        },
        evaluate_test=False,
    )
    assert len(records) == 1
    assert records[0]["test_rocauc"] is None
    assert fitted["summary"]["best_by_valid_only"] == "s"
    assert fitted["predictions"]["s"]["test"].shape == (0, 6)

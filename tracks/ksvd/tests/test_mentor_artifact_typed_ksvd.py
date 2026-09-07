from __future__ import annotations

import numpy as np

from ksvd_research.core import from_edges
from tracks.ksvd.experiments.luyin16.mentor_artifact_typed_ksvd import (
    COMPOSITION_DIM,
    PATCH_DIM,
    TYPED_POOL_DIM,
    _context_cross_cov,
    assemble_view,
    composition_69,
    fit_ksvd,
    pool_typed_descriptors,
    typed_radius2_descriptor,
)
from tracks.ksvd.experiments.luyin16.role_attribute_binding_screen import (
    compact_atom_semantics,
    compact_bond_semantics,
)


def _typed_graph():
    graph = from_edges(6, [(0, 1), (1, 2), (2, 3), (3, 0), (2, 4), (4, 5)])
    node_features = np.zeros((6, 9), dtype=np.int64)
    node_features[:, 0] = [5, 6, 7, 8, 15, 16]
    node_features[:, 2] = [2, 2, 3, 2, 2, 1]
    node_features[:4, 7] = 1
    node_features[:4, 8] = 1
    edge_features = {
        graph.edge_key(left, right): np.asarray(
            [3 if left < 4 and right < 4 else 0, 0, int(left < 4 and right < 4)],
            dtype=np.int64,
        )
        for left, right in graph.edges()
    }
    return graph, node_features, edge_features


def _encoded(graph, node_features, edge_features):
    atom = compact_atom_semantics(node_features)
    bond = {
        graph.edge_key(left, right): compact_bond_semantics(values)
        for (left, right), values in edge_features.items()
    }
    descriptors = np.stack([typed_radius2_descriptor(graph, center, atom, bond) for center in graph.nodes])
    return descriptors, composition_69(graph, atom, bond)


def test_artifact_aligned_dimensions_are_exact() -> None:
    graph, node_features, edge_features = _typed_graph()
    descriptors, composition = _encoded(graph, node_features, edge_features)
    assert descriptors.shape == (graph.n, PATCH_DIM)
    assert composition.shape == (COMPOSITION_DIM,)
    assert pool_typed_descriptors(descriptors).shape == (TYPED_POOL_DIM,)
    assert PATCH_DIM == 208
    assert TYPED_POOL_DIM == 624
    assert COMPOSITION_DIM == 69


def test_typed_descriptor_and_composition_are_node_relabel_invariant() -> None:
    graph, node_features, edge_features = _typed_graph()
    original, original_composition = _encoded(graph, node_features, edge_features)
    permutation = np.asarray([4, 1, 5, 0, 3, 2], dtype=np.int64)
    relabeled = from_edges(
        graph.n,
        [(int(permutation[left]), int(permutation[right])) for left, right in graph.edges()],
    )
    relabeled_nodes = np.empty_like(node_features)
    for old, new in enumerate(permutation):
        relabeled_nodes[int(new)] = node_features[old]
    relabeled_edges = {
        relabeled.edge_key(int(permutation[left]), int(permutation[right])): values.copy()
        for (left, right), values in edge_features.items()
    }
    changed, changed_composition = _encoded(relabeled, relabeled_nodes, relabeled_edges)
    for old, new in enumerate(permutation):
        np.testing.assert_allclose(original[old], changed[int(new)], atol=1e-7)
    np.testing.assert_allclose(original_composition, changed_composition, atol=1e-7)
    np.testing.assert_allclose(pool_typed_descriptors(original), pool_typed_descriptors(changed), atol=1e-7)


def test_view_assembly_uses_documented_blocks() -> None:
    payload = {
        "composition": np.zeros((3, 69), dtype=np.float32),
        "typed_raw": np.zeros((3, 624), dtype=np.float32),
        "typed_init": np.zeros((3, 624), dtype=np.float32),
        "typed_final": np.zeros((3, 624), dtype=np.float32),
        "context_init": np.zeros((3, 165), dtype=np.float32),
        "context_final": np.zeros((3, 165), dtype=np.float32),
        "cross_cov_final": np.zeros((3, 160), dtype=np.float32),
    }
    assert assemble_view(payload, "st_final").shape == (3, 693)
    assert assemble_view(payload, "sta_final").shape == (3, 858)
    assert assemble_view(payload, "sta_cross_cov").shape == (3, 1018)
    assert assemble_view(payload, "st_raw_final").shape == (3, 1317)


def test_context_cross_cov_is_centered_population_covariance() -> None:
    atom_use = np.asarray([[0.2, 0.8], [0.6, 0.4], [0.4, 0.6]], dtype=np.float32)
    masks = np.asarray([[True, False], [False, True], [True, True]], dtype=bool)
    expected = ((atom_use - atom_use.mean(axis=0)) .T @ (masks.astype(np.float32) - masks.mean(axis=0))) / 3.0
    np.testing.assert_allclose(_context_cross_cov(atom_use, masks), expected.reshape(-1), atol=1e-7)


def test_small_ksvd_fit_improves_or_preserves_reconstruction() -> None:
    rng = np.random.default_rng(17)
    atoms = rng.normal(size=(12, 4))
    atoms /= np.linalg.norm(atoms, axis=0, keepdims=True)
    codes = np.zeros((4, 80), dtype=np.float32)
    for column in range(codes.shape[1]):
        support = rng.choice(4, size=2, replace=False)
        codes[support, column] = rng.normal(size=2)
    rows = (atoms @ codes).T.astype(np.float32)
    _, _, _, info = fit_ksvd(
        rows,
        n_atoms=4,
        sparsity=2,
        iterations=2,
        power_iterations=4,
        seed=3,
    )
    assert info["final_relative_reconstruction"] <= (info["initial_relative_reconstruction"] + 1e-5)

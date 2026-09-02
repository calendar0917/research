import numpy as np

from tracks.ksvd.experiments.luyin16.conditional_joint_screen import (
    _conditional_readout,
    _structural_signature,
    _upper_indices,
)
from tracks.ksvd.experiments.luyin16.conditional_residual_screen import (
    _centered_conditional_residual,
)
from tracks.ksvd.experiments.luyin16.long_range_object_relation_screen import (
    _relation_token,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import ATOM_BINS, BOND_BINS


def test_structural_signature_distinguishes_path_and_triangle() -> None:
    indices = _upper_indices(3)
    path = np.asarray([1.0, 0.0, 1.0], dtype=np.float32)
    triangle = np.asarray([1.0, 1.0, 1.0], dtype=np.float32)

    assert _structural_signature(path, 3, indices) == (3, 2, 0, 1, 1)
    assert _structural_signature(triangle, 3, indices) == (3, 3, 1, 2, 0)


def test_conditional_shuffle_preserves_structure_and_attribute_marginals() -> None:
    max_nodes = 3
    topology_dim = 3
    matrix = np.zeros((topology_dim + ATOM_BINS + BOND_BINS, 2), dtype=np.float32)
    matrix[:topology_dim, 0] = [1.0, 0.0, 1.0]
    matrix[:topology_dim, 1] = [1.0, 1.0, 1.0]
    matrix[topology_dim + 1, 0] = 1.0
    matrix[topology_dim + 2, 1] = 1.0
    matrix[topology_dim + ATOM_BINS, 0] = 1.0
    matrix[topology_dim + ATOM_BINS + 1, 1] = 1.0
    signatures = [[(3, 2, 0, 1, 1), (3, 3, 1, 2, 0)]]
    vocabulary = {signatures[0][0]: 0, signatures[0][1]: 1}

    true, _ = _conditional_readout([matrix], signatures, vocabulary, max_nodes)
    shuffled, _ = _conditional_readout(
        [matrix], signatures, vocabulary, max_nodes, shuffle_seed=3
    )
    n_bins = len(vocabulary) + 1
    true_atom = true[0, n_bins : n_bins + n_bins * ATOM_BINS].reshape(n_bins, ATOM_BINS)
    shuffled_atom = shuffled[0, n_bins : n_bins + n_bins * ATOM_BINS].reshape(
        n_bins, ATOM_BINS
    )

    np.testing.assert_allclose(true[0, :n_bins], shuffled[0, :n_bins])
    np.testing.assert_allclose(true_atom.sum(axis=0), shuffled_atom.sum(axis=0))
    assert not np.allclose(true_atom, shuffled_atom)


def test_centered_conditional_residual_removes_independent_marginals() -> None:
    n_bins = 2
    signature_mass = np.asarray([0.4, 0.6], dtype=np.float32)
    atom_marginal = np.zeros(ATOM_BINS, dtype=np.float32)
    atom_marginal[:2] = [0.25, 0.75]
    bond_marginal = np.zeros(BOND_BINS, dtype=np.float32)
    bond_marginal[:2] = [0.7, 0.3]
    atom_joint = np.outer(signature_mass, atom_marginal)
    bond_joint = np.outer(signature_mass, bond_marginal)
    independent = np.concatenate(
        [signature_mass, atom_joint.ravel(), bond_joint.ravel(), [2.0, np.log1p(2.0)]]
    )[None, :]

    residual = _centered_conditional_residual(independent, n_bins)

    np.testing.assert_allclose(residual, 0.0, atol=1e-7)


def test_long_range_relation_token_is_pair_order_invariant() -> None:
    left = (3, 2, 0, 1, 1, 5)
    right = (5, 4, 0, 2, 2, 7)

    assert _relation_token(4, left, right) == _relation_token(4, right, left)

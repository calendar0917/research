from __future__ import annotations

import numpy as np

from .graph import from_edges
from .run_luyin14_joint_multiview_dictionary import (
    _attribute_context,
    _fit_scaler,
    _joint_vectors,
    _shuffle_attribute_context,
)


def test_attribute_context_keeps_root_and_neighbor_mean() -> None:
    graph = from_edges(3, [(0, 1), (1, 2)])
    features = np.eye(3, dtype=np.float64)
    context = _attribute_context(graph, features)
    assert context.shape == (3, 6)
    assert np.array_equal(context[1, :3], features[1])
    assert np.allclose(context[1, 3:], 0.5 * (features[0] + features[2]))


def test_attribute_shuffle_preserves_each_graph_multiset() -> None:
    attributes = [np.arange(20, dtype=np.float64).reshape(5, 4)]
    shuffled = _shuffle_attribute_context(attributes, seed=3)
    assert np.array_equal(
        np.sort(attributes[0], axis=0), np.sort(shuffled[0], axis=0)
    )
    assert not np.array_equal(attributes[0], shuffled[0])


def test_joint_scaling_balances_training_block_energy() -> None:
    structures = [np.array([[0.0, 2.0], [2.0, 0.0]])]
    attributes = [np.array([[0.0], [10.0]])]
    scaler = _fit_scaler(structures, attributes, [0])
    joint = _joint_vectors(structures, attributes, scaler)[0]
    left_energy = np.mean(np.sum(joint[:, :2] ** 2, axis=1))
    right_energy = np.mean(np.sum(joint[:, 2:] ** 2, axis=1))
    assert np.isclose(left_energy, 1.0)
    assert np.isclose(right_energy, 1.0)

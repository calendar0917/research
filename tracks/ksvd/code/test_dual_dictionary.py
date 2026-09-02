from __future__ import annotations

import numpy as np

from .dual_dictionary import (
    relation_event_vectors,
    relation_readout,
    shuffled_patch_code_binding,
    shuffled_relation_events,
)


def test_relation_event_vectors_are_endpoint_order_invariant() -> None:
    codes = np.array([[1.0, 2.0], [3.0, -1.0]])
    attrs = np.array([[0.5, 1.0]])
    forward = relation_event_vectors(codes, np.array([[0, 1]]), attrs, normalize=False)
    backward = relation_event_vectors(codes, np.array([[1, 0]]), attrs, normalize=False)
    np.testing.assert_allclose(forward, backward)


def test_shuffled_events_preserve_relation_attribute_multiset() -> None:
    pairs = np.array([[0, 1], [1, 2], [2, 3]])
    attrs = np.array([[1.0, 0.0], [0.0, 1.0], [0.5, 0.5]])
    shuffled_pairs, shuffled_attrs = shuffled_relation_events(
        4, pairs, attrs, np.random.default_rng(4)
    )
    assert shuffled_pairs.shape == pairs.shape
    np.testing.assert_allclose(
        np.sort(shuffled_attrs, axis=0), np.sort(attrs, axis=0)
    )


def test_relation_readout_handles_empty_sets() -> None:
    out = relation_readout(np.zeros((3, 0)))
    assert out.shape == (13,)
    assert out[-1] == 0.0


def test_shuffled_patch_code_binding_has_no_fixed_endpoint() -> None:
    codes = np.arange(15, dtype=np.float64).reshape(3, 5)
    shuffled = shuffled_patch_code_binding(codes, np.random.default_rng(8))
    assert not np.any(np.all(shuffled == codes, axis=0))
    np.testing.assert_allclose(np.sort(shuffled, axis=1), np.sort(codes, axis=1))

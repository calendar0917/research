from __future__ import annotations

import numpy as np

from .run_luyin14_structured_pursuit import (
    _structured_pursuit,
    _support_diagnostics,
)


def test_isolated_patches_replay_independent_codes() -> None:
    dictionary = np.eye(3)
    centered = np.asarray([[1.0, 0.0], [0.0, 1.0], [0.0, 0.0]])
    independent = centered.copy()
    structured = _structured_pursuit(
        centered,
        dictionary,
        independent,
        np.zeros((2, 2)),
        sparsity=1,
        rho=0.25,
        rounds=2,
    )
    assert np.array_equal(structured, independent)


def test_relation_prior_can_align_ambiguous_support() -> None:
    dictionary = np.eye(2)
    centered = np.asarray([[1.0, 0.49], [0.0, 0.51]])
    independent = np.asarray([[1.0, 0.0], [0.0, 0.51]])
    weights = np.asarray([[0.0, 1.0], [1.0, 0.0]])
    structured = _structured_pursuit(
        centered,
        dictionary,
        independent,
        weights,
        sparsity=1,
        rho=0.25,
        rounds=1,
    )
    assert np.argmax(np.abs(structured[:, 1])) == 0
    diagnostic = _support_diagnostics(
        independent, structured, weights, centered, dictionary
    )
    assert diagnostic["structured_agreement"] > diagnostic["independent_agreement"]
    assert diagnostic["support_change_fraction"] > 0.0


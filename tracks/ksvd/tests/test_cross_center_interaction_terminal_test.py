from __future__ import annotations

import numpy as np

from tracks.ksvd.experiments.luyin16.cross_center_interaction_terminal_test import (
    _fit_scope_projections,
    _rows_for_indices,
)


def test_rows_for_indices_preserves_requested_order() -> None:
    available = np.asarray([10, 20, 30], dtype=np.int64)
    requested = np.asarray([30, 10], dtype=np.int64)
    np.testing.assert_array_equal(
        _rows_for_indices(available, requested, name="cache"), [2, 0]
    )


def test_scope_projection_has_rank_two_blocks_and_test_rows() -> None:
    rng = np.random.default_rng(0)
    dev = {
        "cross_cov": rng.normal(size=(6, 5)).astype(np.float32),
        "binding": rng.normal(size=(6, 4)).astype(np.float32),
    }
    test = {
        "cross_cov": rng.normal(size=(3, 5)).astype(np.float32),
        "binding": rng.normal(size=(3, 4)).astype(np.float32),
    }
    dev_out, test_out, meta = _fit_scope_projections(
        dev, test, np.asarray([0, 2, 4], dtype=np.int64), pca_rank=2
    )
    assert dev_out.shape == (6, 4)
    assert test_out.shape == (3, 4)
    assert meta["fit_rows"] == 3

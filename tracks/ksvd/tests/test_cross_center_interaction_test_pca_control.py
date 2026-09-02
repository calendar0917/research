from __future__ import annotations

import numpy as np

from tracks.ksvd.experiments.luyin16.cross_center_interaction_terminal_test import (
    _build_scope_views,
)


def test_fixed_pca_scope_keeps_all_dev_rows_available_for_refit() -> None:
    rng = np.random.default_rng(1)
    s_dev = rng.normal(size=(5, 2)).astype(np.float32)
    s_test = rng.normal(size=(2, 2)).astype(np.float32)
    dev_raw = {
        "marginal": rng.normal(size=(5, 3)).astype(np.float32),
        "cross_cov": rng.normal(size=(5, 6)).astype(np.float32),
        "binding": rng.normal(size=(5, 7)).astype(np.float32),
        "context": rng.normal(size=(5, 1)).astype(np.float32),
    }
    test_raw = {
        "marginal": rng.normal(size=(2, 3)).astype(np.float32),
        "cross_cov": rng.normal(size=(2, 6)).astype(np.float32),
        "binding": rng.normal(size=(2, 7)).astype(np.float32),
        "context": rng.normal(size=(2, 1)).astype(np.float32),
    }
    views, test_views, _ = _build_scope_views(
        s_dev,
        s_test,
        dev_raw,
        test_raw,
        np.asarray([0, 1, 2], dtype=np.int64),
        pca_rank=2,
    )
    assert views["s_both"].shape == (5, 10)
    assert test_views["s_both"].shape == (2, 10)

from __future__ import annotations

import numpy as np

from .run_luyin14_patch_local_fusion import _cross_correlation


def test_cross_correlation_detects_aligned_modalities() -> None:
    codes = np.asarray([[1.0, 0.0, 1.0, 0.0], [0.0, 1.0, 0.0, 1.0]])
    features = np.asarray([[1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [0.0, 1.0]])
    aligned = _cross_correlation(codes, features).reshape(2, 2)
    shuffled = _cross_correlation(
        codes, features, permutation=np.asarray([1, 0, 3, 2])
    ).reshape(2, 2)
    assert aligned[0, 0] > aligned[0, 1]
    assert aligned[1, 1] > aligned[1, 0]
    assert not np.allclose(aligned, shuffled)


def test_single_patch_has_zero_centered_interaction() -> None:
    codes = np.asarray([[1.0], [0.5]])
    features = np.asarray([[1.0, 0.0, 1.0]])
    assert np.allclose(_cross_correlation(codes, features), 0.0)


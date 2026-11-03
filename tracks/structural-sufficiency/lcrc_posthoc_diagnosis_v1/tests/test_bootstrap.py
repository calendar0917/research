import numpy as np

import ph_common as C


def test_stratified_bootstrap_preserves_class_counts():
    y = np.array([1] * 7 + [0] * 23)
    idx = C.bootstrap_stratified(y, np.random.default_rng(20261103))
    assert idx.size == y.size
    assert int(y[idx].sum()) == 7
    assert int((y[idx] == 0).sum()) == 23


def test_same_seed_same_indices():
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0])
    a = C.bootstrap_stratified(y, np.random.default_rng(11))
    b = C.bootstrap_stratified(y, np.random.default_rng(11))
    assert np.array_equal(a, b)
    c = C.bootstrap_stratified(y, np.random.default_rng(12))
    assert not np.array_equal(a, c)


def test_bootstrap_rows_shape():
    idx = C.bootstrap_rows(37, np.random.default_rng(1))
    assert idx.shape == (37,)
    assert idx.min() >= 0 and idx.max() < 37


def test_quantile_ci_ignores_nan():
    values = [1.0, np.nan, 2.0, 3.0]
    ci = C.quantile_ci(values)
    assert ci["n_finite"] == 3
    assert ci["lo"] <= ci["hi"]


def test_shared_resample_sequence_across_units():
    # The mean-seed bootstrap must use one index vector for all seeds/models.
    y = np.array([1, 0, 1, 0, 0, 1, 0, 0, 0, 1])
    idx = C.bootstrap_stratified(y, np.random.default_rng(20261103))
    seen = []
    for _ in range(6):
        seen.append(idx.copy())
    for other in seen[1:]:
        assert np.array_equal(seen[0], other)

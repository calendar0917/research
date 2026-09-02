from __future__ import annotations

import numpy as np

from .coupled_support_dictionary import (
    coupled_sparse_encode,
    fit_coupled_support_dictionary,
    relative_reconstruction,
    support_agreement,
)


def test_coupled_encoder_has_identical_supports_but_separate_strengths() -> None:
    ds = np.eye(2)
    dc = np.eye(2)
    structure = np.array([[2.0], [3.0]])
    chemistry = np.array([[5.0], [7.0]])
    zs, zc = coupled_sparse_encode(structure, chemistry, ds, dc, sparsity=2)
    np.testing.assert_array_equal(np.abs(zs) > 1e-10, np.abs(zc) > 1e-10)
    assert not np.allclose(zs, zc)


def test_paired_dictionary_recovers_a_low_error_common_support_model() -> None:
    rng = np.random.default_rng(5)
    ds_true = rng.normal(size=(5, 3))
    dc_true = rng.normal(size=(4, 3))
    code_s = np.zeros((3, 80))
    code_c = np.zeros((3, 80))
    selected = rng.integers(0, 3, size=80)
    code_s[selected, np.arange(80)] = rng.normal(size=80)
    code_c[selected, np.arange(80)] = rng.normal(size=80) * 2.0
    structure = ds_true @ code_s
    chemistry = dc_true @ code_c
    ds, dc, _ = fit_coupled_support_dictionary(
        structure, chemistry, n_atoms=3, sparsity=1, n_iter=8, seed=7
    )
    zs, zc = coupled_sparse_encode(structure, chemistry, ds, dc, sparsity=1)
    assert relative_reconstruction(structure, chemistry, ds, dc, zs, zc)["joint"] < 0.15
    assert support_agreement(zs, zc) == 1.0

from __future__ import annotations

import numpy as np

from .relational_ksvd import relational_ksvd, sparse_codes


def test_relational_ksvd_returns_content_only_dictionary() -> None:
    rng = np.random.default_rng(7)
    Y = rng.normal(size=(10, 24))
    Y /= np.maximum(np.linalg.norm(Y, axis=0, keepdims=True), 1e-12)
    pairs = np.array([(i, j) for i in range(12) for j in range(i + 1, 12)], dtype=np.int64)
    labels = np.zeros((len(pairs), 2), dtype=np.float64)
    labels[:, 0] = [(i // 4) == (j // 4) for i, j in pairs]
    labels[:, 1] = [(i + j) % 3 == 0 for i, j in pairs]
    D, info = relational_ksvd(
        Y,
        pairs,
        labels,
        n_atoms=6,
        sparsity=2,
        outer_iter=1,
        code_steps=2,
        seed=3,
    )
    assert D.shape == (10, 6)
    assert np.all(np.isfinite(D))
    np.testing.assert_allclose(np.linalg.norm(D, axis=0), 1.0, atol=1e-7)
    X = sparse_codes(D, Y, 2)
    assert X.shape == (6, 24)
    assert info["n_pairs"] == len(pairs)

from __future__ import annotations

import numpy as np

from tracks.ksvd.experiments.luyin16.invariant_conditional_interaction_screen import (
    _binding_conditional_summary,
    _build_summary_features,
    _spectral_summary,
)


def test_spectral_summary_is_invariant_to_orthogonal_change_of_basis() -> None:
    rng = np.random.default_rng(4)
    matrix = rng.normal(size=(12, 7)).astype(np.float32)
    left, _ = np.linalg.qr(rng.normal(size=(12, 12)))
    right, _ = np.linalg.qr(rng.normal(size=(7, 7)))
    first = _spectral_summary(matrix, 5)
    rotated = _spectral_summary(left @ matrix @ right, 5)
    # The leading scale/spectral coordinates are basis-free.  The remaining
    # concentration coordinates intentionally refer to semantic role/attribute
    # bins, so they are only expected to be stable under bin permutations.
    np.testing.assert_allclose(first[:10], rotated[:10], rtol=2.0e-5, atol=2.0e-5)


def test_binding_summary_is_invariant_to_role_and_attribute_bin_permutations() -> None:
    rng = np.random.default_rng(8)
    matrix = rng.normal(size=(64, 40)).astype(np.float32)
    first = _binding_conditional_summary(matrix, 6)
    role_order = rng.permutation(matrix.shape[0])
    attr_order = rng.permutation(matrix.shape[1])
    permuted = _binding_conditional_summary(matrix[role_order][:, attr_order], 6)
    np.testing.assert_allclose(first, permuted, rtol=2.0e-5, atol=2.0e-5)


def test_summary_feature_shapes_match_interaction_rows() -> None:
    rng = np.random.default_rng(12)
    cross = rng.normal(size=(3, 5088)).astype(np.float32)
    binding = rng.normal(size=(3, 5952)).astype(np.float32)
    cross_out, binding_out, meta = _build_summary_features(cross, binding, top_k=4)
    assert cross_out.shape == (3, 4 * (21 + 4))
    assert binding_out.shape == (3, 4 * (21 + 4))
    assert meta["cross_dimension"] == cross_out.shape[1]
    assert meta["binding_dimension"] == binding_out.shape[1]

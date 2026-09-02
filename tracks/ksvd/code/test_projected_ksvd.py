"""Lightweight self-tests; run with `python -m code.test_projected_ksvd`."""
from __future__ import annotations

import numpy as np

from .projected_ksvd import (
    final_projected_ksvd,
    iterative_projected_ksvd,
    project_dictionary_to_real_patches,
)


def _is_legal(D: np.ndarray, Y: np.ndarray) -> bool:
    Yn = Y / np.maximum(np.linalg.norm(Y, axis=0, keepdims=True), 1e-12)
    return all(np.min(np.linalg.norm(Yn - D[:, [j]], axis=0)) < 1e-10 for j in range(D.shape[1]))


def main() -> None:
    rng = np.random.default_rng(7)
    Y = np.abs(rng.normal(size=(9, 80)))
    Y[:, 10] = Y[:, 0]  # duplicate signature must not yield duplicate atoms
    Dfree = rng.normal(size=(9, 6))
    Dproj, pinfo = project_dictionary_to_real_patches(Dfree, Y)
    assert Dproj.shape == (9, 6)
    assert pinfo["selected_unique_signatures"] == 6
    assert len(set(pinfo["selected_training_indices"])) == 6
    assert _is_legal(Dproj, Y)

    for learner in (final_projected_ksvd, iterative_projected_ksvd):
        D, X, info = learner(Y, n_atoms=6, T=2, n_iter=3, seed=11)
        assert D.shape == (9, 6)
        assert X.shape == (6, 80)
        assert np.isfinite(info["recon_rel"])
        assert info["projectability"] == 1.0
        assert info["selected_unique_signatures"] == 6
        assert _is_legal(D, Y)
    print("projected_ksvd self-tests: PASS")


if __name__ == "__main__":
    main()

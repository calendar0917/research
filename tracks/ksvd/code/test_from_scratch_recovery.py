"""Fast contract tests for the from-scratch E0/E1 recovery experiment."""
from __future__ import annotations

import numpy as np

from .from_scratch_recovery import (
    evaluate_dictionary,
    make_e0_dataset,
    make_e1_dataset,
    optimal_atom_alignment,
)


def main() -> int:
    e0 = make_e0_dataset(seed=17, n_train=80, n_test=40)
    assert e0.Y_train.shape == (15, 80)
    assert np.allclose(e0.Y_train, e0.D_true @ e0.X_train_true)
    _, e0_oracle = evaluate_dictionary(e0, e0.D_true, T=2)
    assert e0_oracle["test_reconstruction_relative"] < 1e-10
    assert e0_oracle["support_f1"] > 0.999999

    e1 = make_e1_dataset(seed=19, n_train=80, n_test=40)
    assert e1.Y_train.shape == (15, 80)
    assert np.all((e1.Y_train == 0.0) | (e1.Y_train == 1.0))
    assert np.allclose(e1.Y_train, e1.D_true @ e1.X_train_true)
    _, e1_oracle = evaluate_dictionary(e1, e1.D_true, T=2)
    assert e1_oracle["test_reconstruction_relative"] < 1e-10
    assert e1_oracle["support_f1"] > 0.999999
    assert e1_oracle["edge_f1"] > 0.999999
    assert e1_oracle["exact_patch_recovery"] > 0.999999
    assert e1_oracle["mean_atom_edge_support_f1"] > 0.999999

    # Alignment must absorb atom permutation and sign ambiguity.
    perm = np.asarray([2, 0, 3, 1], dtype=np.int64)
    signs = np.asarray([-1.0, 1.0, -1.0, 1.0])
    transformed = e1.D_true[:, perm] * signs[None, :]
    alignment = optimal_atom_alignment(e1.D_true, transformed)
    assert alignment["mean_atom_cosine"] > 0.999999

    print("from_scratch_recovery self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

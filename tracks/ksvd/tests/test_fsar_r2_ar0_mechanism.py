"""Targeted tests for the FSAR-R2-AR0 mechanism / frozen-M0 analysis module.

These tests are self-contained where possible.  They cover the new logic that
the eval-only mechanism round introduces (schema mapping, rank helpers, the
frozen-M0 linear training loop, count-key reconstruction) and do not touch the
frozen AR0 model definition.
"""

from __future__ import annotations

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0_mechanism as mech


# ---------------------------------------------------------------------------
# schema mapping (read from the raw ZINC dictionaries, never guessed)
# ---------------------------------------------------------------------------


def test_schema_labels() -> None:
    atoms = mech.atom_category_labels()
    bonds = mech.bond_category_labels()
    assert len(atoms) == r2.ATOM_CATEGORIES
    assert bonds == ["NONE", "SINGLE", "DOUBLE", "TRIPLE"]
    # The first categories are the common ZINC elements.
    assert atoms[0] == "C"
    assert "N" in atoms and "O" in atoms


# ---------------------------------------------------------------------------
# rank helpers
# ---------------------------------------------------------------------------


def test_rank_block_matches_reference() -> None:
    rng = np.random.default_rng(0)
    matrix = rng.normal(size=(64, 40))
    block = mech._rank_block(matrix)
    reference = mech._effective_rank(matrix)
    assert abs(block["effective_rank"] - reference["effective_rank"]) < 1e-12
    assert abs(block["top_singular_fraction"] - reference["top_singular_fraction"]) < 1e-12
    # PC1 explained variance equals s_0^2 / sum(s^2) on the centred matrix
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    singular = np.linalg.svd(centered, compute_uv=False)
    assert abs(block["pc1_explained_variance"] - singular[0] ** 2 / (singular**2).sum()) < 1e-12


def test_legacy_slice_differs_from_full_matrix() -> None:
    """The historical audit sliced only the first 64 of 1820 coordinates."""
    rng = np.random.default_rng(1)
    # first 64 columns rank-1 (near constant direction), remaining columns full rank
    direction = rng.normal(size=(64, 1))
    scores = rng.normal(size=(300, 1))
    first = scores @ direction.T
    rest = rng.normal(size=(300, 1820 - 64))
    matrix = np.concatenate([first, rest], axis=1)
    sliced = mech._effective_rank(matrix[:, :64])["effective_rank"]
    full = mech._effective_rank(matrix)["effective_rank"]
    assert sliced < 1.5
    assert full > 10.0


def test_node_arrays_scaled_identity() -> None:
    train, _valid, scalers, _meta = mech.runner.build_datasets()
    arrays = mech._node_arrays(list(train[:8]), scalers)
    d_c = np.asarray(scalers["C_rms"], dtype=np.float64)
    mask = np.asarray(scalers["C_mask"], dtype=np.float64)
    expected = arrays.C_raw / d_c[None] * mask[None]
    assert np.allclose(arrays.C_tilde, expected, atol=1e-10)


# ---------------------------------------------------------------------------
# frozen-M0 linear residual training
# ---------------------------------------------------------------------------


def test_frozen_linear_reduces_mae_and_keeps_base() -> None:
    rng = np.random.default_rng(3)
    n_train, n_valid = 320, 96
    phi_dim, atom_categories = 6, 4
    c_train = rng.normal(size=(n_train, phi_dim, atom_categories)).astype(np.float64)
    c_valid = rng.normal(size=(n_valid, phi_dim, atom_categories)).astype(np.float64)
    weight_true = rng.normal(size=(phi_dim, atom_categories))
    base_train = rng.normal(size=n_train)
    base_valid = rng.normal(size=n_valid)
    y_train = base_train + (c_train * weight_true).sum(axis=(1, 2))
    y_valid = base_valid + (c_valid * weight_true).sum(axis=(1, 2))
    protocol = {
        "learning_rate": 1.0e-2,
        "weight_decay": 0.0,
        "batch_size": 64,
        "max_epochs": 200,
        "patience": 40,
        "gradient_clip_norm": 5.0,
        "train_shuffle_seed_offset": 91011,
    }
    result = mech._train_frozen_linear(
        c_train,
        base_train,
        y_train,
        c_valid,
        base_valid,
        y_valid,
        0,
        protocol,
        phi_dim=phi_dim,
        atom_categories=atom_categories,
    )
    initial_mae = float(np.mean(np.abs(y_valid - base_valid)))
    assert result["top5_soup_valid_mae"] < 0.5 * initial_mae
    assert result["best_valid_mae"] < 0.5 * initial_mae
    assert result["W_norm"] > 0.0
    # the frozen base is never modified by construction: residual target only
    assert np.allclose(y_train - base_train, (c_train * weight_true).sum(axis=(1, 2)))


def test_frozen_linear_zero_init_predicts_base() -> None:
    n = 40
    phi_dim, atom_categories = 6, 4
    c = np.zeros((n, phi_dim, atom_categories))
    base = np.arange(n, dtype=np.float64)
    y = base.copy()
    protocol = {
        "learning_rate": 1.0e-3,
        "weight_decay": 0.0,
        "batch_size": 16,
        "max_epochs": 1,
        "patience": 40,
        "gradient_clip_norm": 5.0,
        "train_shuffle_seed_offset": 0,
    }
    result = mech._train_frozen_linear(
        c, base, y, c, base, y, 0, protocol, phi_dim=phi_dim, atom_categories=atom_categories
    )
    assert abs(result["best_valid_mae"]) < 1e-8


# ---------------------------------------------------------------------------
# witness count keys
# ---------------------------------------------------------------------------


def test_count_keys_roundtrip() -> None:
    graph = mech.runner.from_edges(4, [(0, 1), (1, 2), (2, 3)])
    types = np.asarray([0, 1, 1, 2], dtype=np.int64)
    molecule = r2.MoleculeFeatures(
        phi=r2.build_phi(graph).astype(np.float32),
        atom_idx=types,
        A=r2.build_A(types, {(0, 1): 1, (1, 2): 2, (2, 3): 1}, 4).astype(np.float32),
        n_nodes=4,
        n_edges=3,
        y=0.0,
    )
    key = mech._count_keys([molecule])[0]
    atom_counts, bond_counts = key
    assert sum(atom_counts) == 4
    assert sum(bond_counts) == 3
    assert atom_counts[0] == 1 and atom_counts[1] == 2 and atom_counts[2] == 1
    assert bond_counts[1] == 2 and bond_counts[2] == 1


if __name__ == "__main__":  # pragma: no cover
    import pytest

    raise SystemExit(pytest.main([__file__, "-q"]))

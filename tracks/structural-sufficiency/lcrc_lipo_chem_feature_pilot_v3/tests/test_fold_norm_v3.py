"""Fold-1 row alignment, train-only normalisation and frozen-extra round-trip."""

from __future__ import annotations

import numpy as np

import v3_common as C
import v3_features as F


def test_fold1_positions_and_shapes(fold_extras_12):
    fx = fold_extras_12
    assert len(fx.train_positions) == 1568
    assert len(fx.dev_positions) == 784
    assert set(fx.train_positions) & set(fx.dev_positions) == set()
    assert sorted(fx.train_positions + fx.dev_positions) == list(range(2352))
    assert fx.train_raw.shape == (1568, 12)
    assert fx.dev_raw.shape == (784, 12)
    assert fx.audit["train_dev_disjoint"] is True
    assert fx.audit["train_dev_cover_fit2352"] is True


def test_global_extra_arrays_are_indexed_by_local_fold_position(store, fold_extras_12):
    fx = fold_extras_12
    norm = F.fit_extra_norm(store.chem12, fx.train_positions)
    imputed = norm["raw_imputed"]
    assert np.array_equal(fx.train_raw, imputed[np.asarray(fx.train_positions)])
    assert np.array_equal(fx.dev_raw, imputed[np.asarray(fx.dev_positions)])
    folds = C.F.load_folds()
    for local, pos in enumerate(fx.train_positions):
        row = store.rows[pos]
        assert fx.train_rows[local] == folds["rows"][str(row)]["global_row_id"]
        assert store.canonical_hashes[pos] == folds["rows"][str(row)]["canonical_smiles_sha256"]
    for local, pos in enumerate(fx.dev_positions):
        row = store.rows[pos]
        assert fx.dev_rows[local] == folds["rows"][str(row)]["global_row_id"]


def test_extra_normalisation_is_fit_on_the_training_fold_only(store, fold_extras_12):
    fx = fold_extras_12
    train_pos = np.asarray(fx.train_positions, dtype=np.int64)
    dev_pos = np.asarray(fx.dev_positions, dtype=np.int64)
    base = F.fit_extra_norm(store.chem12, train_pos)
    perturbed = store.chem12.copy()
    perturbed[dev_pos] = perturbed[dev_pos] * -1234.0 + 99.0
    other = F.fit_extra_norm(perturbed, train_pos)
    for key in ("median", "mean", "scale"):
        assert np.array_equal(np.asarray(base[key]), np.asarray(other[key]))
    assert base["audit"]["dev_used_for_fit"] is False
    assert base["audit"]["labels_used"] is False


def test_augmented_norm_preserves_the_original_42_and_scalars(fold_extras_12):
    fx = fold_extras_12
    view = C.D.build_fold_view(
        C.FOLD,
        cache=C.load_fit_cache()["cache"],
        y_fit=np.asarray(C.load_fit_labels()["y"], dtype=np.float64),
        folds=C.F.load_folds(),
    )
    assert np.array_equal(fx.norm["global_mean"][:42], np.asarray(view.train_norm["global_mean"], dtype=np.float32))
    assert np.array_equal(fx.norm["global_scale"][:42], np.asarray(view.train_norm["global_scale"], dtype=np.float32))
    assert np.array_equal(fx.norm["scalar_mean"], np.asarray(view.train_norm["scalar_mean"], dtype=np.float32))
    assert np.array_equal(fx.norm["scalar_scale"], np.asarray(view.train_norm["scalar_scale"], dtype=np.float32))
    assert fx.norm["global_mean"].shape == (54,)
    assert fx.norm["global_scale"].shape == (54,)


def test_frozen_extras_round_trip_matches_fresh_fold_construction(fold_extras_12, fold_extras_13):
    if not F.FROZEN_EXTRAS_NPZ.exists():
        import pytest

        pytest.skip("frozen extras not generated yet")
    for variant, fresh in (("chem12", fold_extras_12), ("chem13", fold_extras_13)):
        frozen = F.load_frozen_extras(variant)
        assert frozen["spec"].hash() == fresh.spec.hash()
        assert np.array_equal(frozen["train"], fresh.train_raw)
        assert np.array_equal(frozen["dev"], fresh.dev_raw)
        for key in ("global_mean", "global_scale", "scalar_mean", "scalar_scale"):
            assert np.array_equal(frozen["norm"][key], np.asarray(fresh.norm[key], dtype=np.float32))
        assert frozen["train_positions"] == fresh.train_positions
        assert frozen["dev_positions"] == fresh.dev_positions

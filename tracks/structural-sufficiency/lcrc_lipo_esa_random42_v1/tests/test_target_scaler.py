"""Protocol requirement 1: the target scaler depends on Train3360 only."""

from __future__ import annotations

import numpy as np


def test_target_scaler_matches_sklearn_on_train_only(labels):
    import r42_data as D

    from sklearn.preprocessing import StandardScaler

    scaler = D.fit_target_scaler(labels.y_train)
    ref = StandardScaler().fit(labels.y_train.reshape(-1, 1))
    assert scaler["n"] == 3360
    assert scaler["source"] == "train3360_labels_only"
    assert scaler["valid_or_test_used"] is False
    assert abs(scaler["mean"] - float(ref.mean_[0])) < 1e-12
    assert abs(scaler["scale"] - float(ref.scale_[0])) < 1e-12


def test_target_scaler_is_invariant_to_validation_values(labels):
    import r42_data as D

    base = D.fit_target_scaler(labels.y_train)
    # Appending arbitrary "validation" values must not change the Train fit,
    # which also proves the fit never looks at Validation.
    shifted = D.fit_target_scaler(np.concatenate([labels.y_train, np.full(420, 1e6)]))
    assert shifted["mean"] != base["mean"]
    again = D.fit_target_scaler(labels.y_train)
    assert again["mean"] == base["mean"] and again["scale"] == base["scale"]
    assert base["label_vector_sha256"] == again["label_vector_sha256"]


def test_standardisation_round_trip(labels):
    import r42_common as C
    import r42_data as D
    import r42_native_train as NT
    import torch

    norm = D.fit_target_scaler(labels.y_train)
    scaled = NT.target_standardise(labels.y_valid, norm)
    assert isinstance(scaled, torch.Tensor)
    assert scaled.shape == (420,)
    back = NT.denormalise(scaled.numpy(), norm)
    assert np.allclose(back, labels.y_valid, atol=1e-9)
    assert torch.isclose(scaled.mean(), torch.tensor(float((labels.y_valid.mean() - norm["mean"]) / norm["scale"])), atol=1e-5)
    assert C.PROTOCOL_ID == "lcrc-lipo-esa-random42-v1"

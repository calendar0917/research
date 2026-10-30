from __future__ import annotations

import pytest

import screen_common as C
import screen_predict as P


def test_prediction_registry_is_complete_and_frozen():
    keys = P._expected_keys()
    assert len(keys) == 3 * len(C.SEEDS) + len(C.BASELINE_ARMS) * len(C.BASELINE_REPLICATES) == 12
    assert keys[:3] == ["A_s0", "A_s1", "A_s2"]
    assert keys[-3:] == ["D_r0", "D_r1", "D_r2"]
    assert len(set(keys)) == len(keys)


def test_predict_nn_refuses_without_a_training_run():
    with pytest.raises(P.PredictionError, match="missing training artifacts"):
        P.predict_nn("A", 0, "cpu")


def test_freeze_refuses_an_incomplete_registry():
    assert not C.PREDICTION_MANIFEST.exists()
    with pytest.raises(P.PredictionError, match="missing prediction"):
        P.freeze_predictions()
    assert not C.PREDICTION_MANIFEST.exists()
    assert not C.PREDICTION_LOCK.exists()


def test_prediction_files_do_not_contain_labels():
    for path in sorted(C.PREDICTIONS.glob("*.npz")):
        with __import__("numpy").load(path) as arrays:
            assert set(arrays.files) <= {"rows", "pred", "contrib", "rho", "contrib_atom_sum_check"}
            assert "y" not in arrays.files

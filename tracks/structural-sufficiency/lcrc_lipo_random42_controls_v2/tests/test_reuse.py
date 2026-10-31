from __future__ import annotations

import numpy as np

import c2_common as C


def test_frozen_split_and_label_hashes():
    reuse = C.verify_history()
    assert reuse["status"] == "PASS"
    assert reuse["test_labels_read"] is False
    assert reuse["split"]["split_file_sha256"] == C.V1_SPLIT_SHA256
    assert reuse["split"]["split_index_sha256"] == C.SPLIT_SHA256
    assert reuse["split"]["label_file_sha256"] == C.V1_LABELS_SHA256
    assert reuse["structures"]["sha256"] == C.V1_STRUCTURES_SHA256


def test_target_scaler_is_train_only_and_frozen():
    import c2_data as CD

    _split, labels = CD.load_frozen_split_and_labels()
    norm = CD.load_target_norm(labels)
    assert norm["mean"] == C.TARGET_MEAN
    assert norm["scale"] == C.TARGET_SCALE
    assert norm["source"] == "train3360_labels_only"
    assert norm["n"] == 3360
    assert abs(float(np.mean(labels.y_train)) - C.TARGET_MEAN) < 1e-15
    assert abs(float(np.std(labels.y_train, ddof=0)) - C.TARGET_SCALE) < 1e-15


def test_historical_validation_metrics_reproduce():
    reuse = C.verify_history()
    for name, row in reuse["models"].items():
        frozen = row["frozen_v1_valid_metrics"]
        recomputed = row["valid_metrics_recomputed"]
        for key in ("rmse", "mae", "r2"):
            assert abs(float(recomputed[key]) - float(frozen[key])) <= 1e-12, (name, key)


def test_source_hashes_are_frozen():
    sources = C.verify_source_hashes()
    assert sources["v2_models.py"]["sha256"] == C.V2_MODELS_SHA256
    assert sources["lcrc_model.py"]["sha256"] == C.LCRC_MODEL_SHA256

from __future__ import annotations

import numpy as np
import pytest

import m2_common as C
import m2_data as MD


def test_split_and_label_hashes(views):
    split = views["split"]
    for name in ("train", "valid", "test"):
        rows = [int(x) for x in split["rows"][name]]
        assert len(rows) == C.SPLIT_SIZES[name]
        assert C.rows_sha256_int64(rows) == C.SPLIT_SHA256[name]
        assert rows[:5] == C.SPLIT_FIRST_ROWS[name]
    assert C.vector_sha256_float64(views["labels"]["y_train"]) == C.TARGET_VECTOR_SHA256
    assert C.vector_sha256_float64(views["labels"]["y_valid"]) == C.VALID_VECTOR_SHA256
    assert C.vector_sha256_float64(views["test"]["y_test"]) == C.TEST_VECTOR_SHA256


def test_test_labels_come_from_frozen_rows(views):
    _, _, all_y = MD.read_exp_column()
    rows = np.asarray(views["split"]["rows"]["test"], dtype=np.int64)
    assert np.array_equal(views["test"]["y_test"], all_y[rows])
    assert views["test"]["audit"]["n_test"] == 420
    assert views["test"]["audit"]["first_five_rows"] == [967, 3557, 3492, 462, 2258]


def test_cache_identity(views):
    assert views["cache_built"]["content_fingerprint"] == C.LCRC_CACHE_CONTENT_FINGERPRINT
    assert views["train_norm_fingerprint"] == C.LCRC_TRAIN_NORM_FINGERPRINT
    assert views["target_norm"]["mean"] == C.TARGET_MEAN
    assert views["target_norm"]["scale"] == C.TARGET_SCALE
    assert [str(x) for x in views["test_cache"]["molecule_ids"]] == [str(x) for x in views["split"]["rows"]["test"]]


def test_config_validates():
    cfg = C.load_config()
    assert cfg["data"]["test_access"] == "EXPOSED_TEST_EXPLORATORY"
    assert set(cfg["arms"]) == set(C.ARM_SPECS)

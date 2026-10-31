from __future__ import annotations

import numpy as np
import pytest

import c2_common as C
import c2_data as CD


def test_train_valid_labels_reject_test():
    _split, labels = CD.load_frozen_split_and_labels()
    with pytest.raises(Exception):
        labels.get("test")
    with pytest.raises(Exception):
        __import__("r42_labels").assert_train_valid_only(
            {"train": labels.y_train, "valid": labels.y_valid, "test": np.zeros(420)}
        )


def test_no_test_artifact_references_or_predictions():
    payload = C.assert_no_forbidden_artifacts_read()
    assert payload["status"] == "PASS"
    assert payload["test_artifacts_read"] is False
    assert not any("test" in name.lower() for name in payload["prediction_files"])

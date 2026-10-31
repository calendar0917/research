"""Frozen data, split, label isolation and element-role tests."""

from __future__ import annotations

import json

import numpy as np
import pytest

import x1_common as C
import x1_data as XD


def test_split_sizes_hashes_order(frozen_data):
    split, labels, views = frozen_data
    audit = views["split_audit"]
    assert audit["sha256"] == C.V1_SPLIT_SHA256
    assert audit["n_train"] == 3360 and audit["n_valid"] == 420
    assert audit["train_rows_sha256"] == C.SPLIT_SHA256["train"]
    assert audit["valid_rows_sha256"] == C.SPLIT_SHA256["valid"]
    assert audit["first_five"] == C.SPLIT_FIRST_ROWS
    assert audit["test_row_values_used"] is False
    assert int(views["train_cache"]["n_molecules"]) == 3360
    assert int(views["valid_cache"]["n_molecules"]) == 420
    assert [int(r) for r in split["rows"]["train"]][:5] == C.SPLIT_FIRST_ROWS["train"]
    assert [int(r) for r in split["rows"]["valid"]][:5] == C.SPLIT_FIRST_ROWS["valid"]


def test_labels_are_train_valid_only(frozen_data):
    _split, labels, _views = frozen_data
    assert labels.audit["test_labels_included"] is False
    assert labels.audit["scope"] == "train_and_valid_only"
    assert C.vector_sha256_float64(labels.y_train) == C.TARGET_VECTOR_SHA256
    assert C.vector_sha256_float64(labels.y_valid) == C.VALID_VECTOR_SHA256
    with pytest.raises(Exception):
        labels.get("test")


def test_label_artifact_has_no_test_values():
    payload = json.loads(C.V1_LABELS_FILE.read_text(encoding="utf-8"))
    assert payload["test_labels_included"] is False
    assert set(payload["values"]) == {"train", "valid"}
    assert set(payload["rows"]) == {"train", "valid"}


def test_target_scaler_and_cache_fingerprint(frozen_data):
    _split, _labels, views = frozen_data
    assert views["target_norm"]["mean"] == C.TARGET_MEAN
    assert views["target_norm"]["scale"] == C.TARGET_SCALE
    assert views["train_norm_fingerprint"] == C.LCRC_TRAIN_NORM_FINGERPRINT
    assert views["built"]["content_fingerprint"] == XD.LCRC_CACHE_CONTENT_FINGERPRINT
    assert views["built"]["cache"]["meta"]["labels_present"] is False


def test_element_role_tables_match_metadata(frozen_data):
    _split, _labels, views = frozen_data
    spec = views["spec"]
    tables = XD.role_tables(spec)
    metadata = list(spec.atom_type_metadata)
    expected_het = [
        i for i, meta in enumerate(metadata) if meta.get("element") in C.HETERO_ELEMENTS
    ]
    expected_ch = [
        i for i, meta in enumerate(metadata) if meta.get("element") in C.CARBON_HALOGEN_ELEMENTS
    ]
    assert tables["hetero_type_ids"] == expected_het
    assert tables["carbon_halogen_type_ids"] == expected_ch
    assert "B" in tables["unassigned_type_names"] and "Si" in tables["unassigned_type_names"]
    assert "UNK" in tables["unassigned_type_names"]


def test_role_coverage_counts(frozen_data):
    _split, _labels, views = frozen_data
    coverage = XD.role_coverage(views["valid_cache"], views["spec"])
    counts = coverage["molecule_counts"]
    assert sum(counts.values()) == coverage["n_molecules"] == 420
    assert coverage["n_hetero_atoms"] + coverage["n_carbon_halogen_atoms"] > 0
    assert 0.0 <= coverage["molecules_with_hetero_fraction"] <= 1.0
    assert 0.0 <= coverage["molecules_with_carbon_halogen_fraction"] <= 1.0


def test_forbidden_read_detection():
    assert XD.forbidden_reads(["a/b/test420_pred.npz"]) == ["a/b/test420_pred.npz"]
    assert XD.forbidden_reads(["a/b/valid_predictions.npz"]) == []


def test_read_audit_records_opens(tmp_path):
    path = tmp_path / "x.txt"
    path.write_text("ok", encoding="utf-8")
    with XD.read_audit() as opened:
        path.read_text(encoding="utf-8")
    assert any(str(path) == p for p in opened)

from __future__ import annotations

import ast

import numpy as np
import pytest

import access_guard
import fit_labels
import monitor_labels
import screen_common as C
import screen_data as D


def test_fit_loader_constructs_only_fit_values():
    fit = fit_labels.load_fit_labels()
    audit = fit["audit"]
    assert fit["y"].shape == (C.SPLIT_SIZES["fit"],)
    assert audit["label_values_constructed"] == C.SPLIT_SIZES["fit"]
    assert audit["label_values_skipped"] == C.SPLIT_SIZES["monitor"]
    assert audit["unrequested_values_constructed"] == 0
    assert audit["total_label_keys_in_artifact"] == C.SPLIT_SIZES["fit"] + C.SPLIT_SIZES["monitor"]
    assert audit["reserved_rows_in_artifact"] == 0
    assert audit["official_valid_test_rows_in_artifact"] == 0
    assert audit["labels_sha256"] == C.EXPECTED_SHA256["labels_fit_monitor"]
    assert np.isfinite(fit["y"]).all()


def test_fit_labels_are_aligned_to_explicit_rows():
    split = C.load_frozen_split()
    fit = fit_labels.load_fit_labels()
    caches = D.load_caches()
    assert list(fit["rows"]) == split["split"]["fit"]
    assert [int(x) for x in caches["fit"]["cache"]["molecule_ids"]] == split["split"]["fit"]
    rows = fit["rows"]
    assert C.prep_common.row_sha256(sorted(rows + split["split"]["monitor"])) == C.EXPECTED_LABELS_ROW_HASH
    assert fit["audit"]["rows_in_label_artifact"] == C.SPLIT_SIZES["fit"] + C.SPLIT_SIZES["monitor"]
    assert not set(rows) & set(split["split"]["reserved"])


def test_reserved_rows_cannot_be_requested_through_the_label_scanner():
    split = C.load_frozen_split()
    reserved_key = str(int(split["split"]["reserved"][0]))
    access_guard.arm()
    with pytest.raises(fit_labels.LabelArtifactError):
        fit_labels.scan_label_artifact(C.LIPO_LABELS, [reserved_key])
    assert access_guard.denied_attempts() == []


def test_monitor_access_blocked_before_freeze():
    assert not C.PREDICTION_MANIFEST.exists()
    with pytest.raises(monitor_labels.FreezeError):
        monitor_labels.verify_prediction_freeze()
    with pytest.raises(monitor_labels.FreezeError):
        monitor_labels.load_monitor_labels()


def test_only_evaluation_gate_modules_import_monitor_labels():
    importers = []
    for path in sorted((C.EXP_DIR / "code").glob("*.py")):
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                if any(a.name.split(".")[0] == "monitor_labels" for a in node.names):
                    importers.append(path.name)
            elif isinstance(node, ast.ImportFrom) and node.module:
                if node.module.split(".")[0] == "monitor_labels":
                    importers.append(path.name)
    assert set(importers) <= {"evaluate.py", "screen_gates.py"}
    for training_module in ("screen_training.py", "screen_predict.py", "baseline_ecfp_xgb.py", "screen_data.py"):
        assert training_module not in importers


def test_guard_blocks_forbidden_sources():
    access_guard.arm()
    for path in (C.LIPO_CSV, C.LIPO_OFFICIAL_SPLIT, C.PREP_CACHE / "lipo_reserved.pt"):
        with pytest.raises(PermissionError):
            with open(path, "rb") as fh:
                fh.read(1)
    with pytest.raises(PermissionError):
        with open(C.LIPO_LABELS, "rb") as fh:
            fh.read(1)


def test_label_grant_is_one_shot():
    access_guard.arm()
    access_guard.grant_once(C.LIPO_LABELS)
    with open(C.LIPO_LABELS, "rb") as fh:
        assert fh.read(1) == b"{"
    with pytest.raises(PermissionError):
        with open(C.LIPO_LABELS, "rb") as fh:
            fh.read(1)
    status = access_guard.guard_status()
    assert status["denied_attempts"]
    assert any("labels_fit_monitor" in p for p in status["denied_attempts"])


def test_training_module_has_no_monitor_evaluation_path():
    source = (C.EXP_DIR / "code" / "screen_training.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    names = {n.id for n in ast.walk(tree) if isinstance(n, ast.Name)}
    assert "monitor_labels" not in names
    for forbidden in ("load_monitor_labels", "monitor_metrics", "evaluate_monitor", "early_stopping"):
        assert forbidden not in source

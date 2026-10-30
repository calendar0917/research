from __future__ import annotations

import pytest

import v2_common as C
import v2_monitor_labels as L
import v2_predict as P


def test_access_guard_denies_a_direct_label_open():
    guard = C.v1_access_guard
    guard.arm()
    try:
        with pytest.raises(PermissionError):
            with open(C.v1c.LIPO_LABELS, "rb") as fh:
                fh.read(1)
    finally:
        guard.disarm()


def test_fit_labels_construct_only_fit_values():
    fit = C.load_fit_labels()
    assert fit["meta"]["scope"]["reserved_labels_present"] is False
    assert fit["meta"]["scope"]["official_valid_test_labels_present"] is False
    assert int(fit["meta"]["scope"]["monitor_rows"]) == 504
    assert int(fit["audit"]["label_values_constructed"]) == 2352
    assert int(fit["audit"]["label_values_skipped"]) == 504
    assert int(fit["audit"]["unrequested_values_constructed"]) == 0
    assert len(fit["y"]) == 2352
    assert set(int(x) for x in fit["rows"]) == set(int(x) for x in C.load_frozen_split()["split"]["fit"])


def test_training_modules_do_not_import_monitor_or_evaluate():
    allowed = {"run_v2.py", "v2_evaluate.py", "v2_monitor_labels.py"}
    offenders = []
    for path in sorted((C.EXP_DIR / "code").glob("*.py")):
        if path.name in allowed:
            continue
        text = path.read_text(encoding="utf-8")
        for token in ("monitor_labels", "v2_monitor_labels", "v2_evaluate"):
            if f"import {token}" in text or f"from {token}" in text:
                offenders.append((path.name, token))
    assert not offenders


def test_monitor_loader_refuses_before_the_manifest(monkeypatch):
    def blocked():
        raise PermissionError("manifest not frozen")

    monkeypatch.setattr(P, "verify_manifest", blocked)
    guard = C.v1_access_guard
    guard.disarm()
    with pytest.raises(PermissionError):
        L.load_monitor_labels()
    # The refusal happened before arming the label gate.
    assert getattr(guard, "_armed", False) is False


def test_reserved_and_official_paths_are_not_in_fit_scope():
    split = C.load_frozen_split()
    fit = set(int(x) for x in split["split"]["fit"])
    monitor = set(int(x) for x in split["split"]["monitor"])
    assert len(fit) == 2352
    assert len(monitor) == 504
    assert fit.isdisjoint(monitor)

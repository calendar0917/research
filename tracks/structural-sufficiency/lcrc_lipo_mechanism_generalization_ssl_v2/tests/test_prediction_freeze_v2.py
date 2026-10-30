from __future__ import annotations

import numpy as np
import pytest

import v2_common as C
import v2_predict as P


def _write_pred(directory, key, rows, values):
    npz = directory / f"{key}.npz"
    np.savez(npz, rows=np.asarray(rows, dtype=np.int64), pred=np.asarray(values, dtype=np.float64))
    sidecar = {
        "key": key,
        "monitor_labels_read": False,
        "sha256": C.sha256_file(npz),
        "rows": [int(x) for x in rows],
    }
    C.write_json(directory / f"{key}.json", sidecar)
    return sidecar


def _setup(monkeypatch, tmp_path):
    rows = [10, 11, 12, 13]
    registry = [
        {"key": "full_A_scratch_s0", "arm": "A", "recipe": "R0", "seed": 0, "variant": "scratch"},
        {"key": "full_E_R0_s0", "arm": "E", "recipe": "R0", "seed": 0, "variant": "scratch"},
    ]
    monkeypatch.setattr(C, "PREDICTIONS", tmp_path)
    monkeypatch.setattr(C, "track_rel", lambda p: str(p))
    monkeypatch.setattr(P, "monitor_rows", lambda: rows)
    monkeypatch.setattr(P, "prediction_registry", lambda: registry)
    for key in ("full_A_scratch_s0", "full_E_R0_s0"):
        _write_pred(tmp_path, key, rows, np.linspace(-1, 1, len(rows)))
    return rows, registry


def test_manifest_freeze_and_verify(monkeypatch, tmp_path):
    _setup(monkeypatch, tmp_path)
    manifest = P.freeze_predictions()
    assert manifest["label_blind"] is True
    assert manifest["n_predictions"] == 2
    assert (tmp_path / "MANIFEST.sha256").read_text().strip() == C.sha256_file(tmp_path / "MANIFEST.json")
    verified = P.verify_manifest()
    assert verified["manifest_sha256"] == manifest["manifest_sha256"]


def test_manifest_rejects_a_changed_prediction(monkeypatch, tmp_path):
    rows, _ = _setup(monkeypatch, tmp_path)
    P.freeze_predictions()
    np.savez(tmp_path / "full_A_scratch_s0.npz", rows=np.asarray(rows, dtype=np.int64), pred=np.zeros(len(rows)))
    with pytest.raises(PermissionError, match="changed after the freeze"):
        P.verify_manifest()


def test_manifest_rejects_missing_extra_or_bad_rows(monkeypatch, tmp_path):
    rows, _ = _setup(monkeypatch, tmp_path)
    (tmp_path / "extra.npz").write_bytes(b"not a prediction")
    with pytest.raises(PermissionError, match="unregistered"):
        P.freeze_predictions()
    (tmp_path / "extra.npz").unlink()
    (tmp_path / "full_E_R0_s0.npz").unlink()
    with pytest.raises(PermissionError, match="missing"):
        P.freeze_predictions()
    _write_pred(tmp_path, "full_E_R0_s0", [rows[0], rows[2], rows[1], rows[3]], [0, 0, 0, 0])
    with pytest.raises(ValueError, match="row order"):
        P.freeze_predictions()


def test_manifest_requires_label_blind_sidecar(monkeypatch, tmp_path):
    rows, _ = _setup(monkeypatch, tmp_path)
    sidecar = C.read_json(tmp_path / "full_A_scratch_s0.json")
    sidecar["monitor_labels_read"] = True
    C.write_json(tmp_path / "full_A_scratch_s0.json", sidecar)
    with pytest.raises(PermissionError, match="label-blind"):
        P.freeze_predictions()

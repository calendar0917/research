from __future__ import annotations

import numpy as np
import pytest

import baseline_ecfp_xgb as BASE
import screen_common as C


def test_fingerprint_spec_is_frozen():
    cfg = C.load_config()
    assert cfg["baseline"]["fp_bits"] == 2048
    assert cfg["baseline"]["fp_radius"] == 2
    assert cfg["baseline"]["fp_use_features"] is False
    assert cfg["baseline"]["xgboost"] == {
        "n_estimators": 500,
        "max_depth": 6,
        "learning_rate": 0.05,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_lambda": 1.0,
        "min_child_weight": 1,
        "tree_method": "hist",
        "n_jobs": -1,
    }


def test_fingerprints_align_and_are_deterministic(split):
    rows = [int(r) for r in split["split"]["fit"][:64]]
    a = BASE.fingerprint_rows(rows)
    b = BASE.fingerprint_rows(rows)
    assert a["feature_sha256"] == b["feature_sha256"]
    assert a["X"].shape == (64, 2048)
    assert a["X"].dtype == np.uint8
    assert set(np.unique(a["X"])) <= {0, 1}
    assert a["canonical_smiles_sha256"] == b["canonical_smiles_sha256"]
    assert len(set(a["canonical_smiles_sha256"])) == 64
    assert int(a["X"].sum(axis=1).min()) > 0


def test_monitor_fingerprint_rows_match_split(split):
    rows = [int(r) for r in split["split"]["monitor"][:32]]
    out = BASE.fingerprint_rows(rows)
    assert out["rows"] == rows
    assert out["X"].shape == (32, 2048)


def test_baseline_has_no_label_leak_in_features():
    cfg = C.load_config()
    assert "y" not in cfg["baseline"]
    assert cfg["baseline"]["refit_on"] == "fit2352"
    assert cfg["baseline"]["score_on"] == "monitor504"
    assert cfg["baseline"]["reuse_previous_bjoc_numbers"] is False
    assert cfg["baseline"]["hyperparameter_search"] is False


def test_xgb_params_are_frozen_and_seeded():
    cfg = C.load_config()
    p0 = BASE._xgb_params(cfg, 0)
    p1 = BASE._xgb_params(cfg, 1)
    assert p0["random_state"] == 0 and p1["random_state"] == 1
    for key in ("n_estimators", "max_depth", "learning_rate", "subsample", "colsample_bytree", "reg_lambda", "tree_method"):
        assert p0[key] == cfg["baseline"]["xgboost"][key]
    assert p0["n_jobs"] == -1


def test_fingerprint_rejects_unfrozen_row_manifest(monkeypatch):
    import adapters as prep_adapters

    original = prep_adapters.load_lipo_row_manifest
    manifest = original(C.LIPO_ROW_MANIFEST)
    tampered = dict(manifest)
    tampered["canonical_smiles_sha256"] = list(manifest["canonical_smiles_sha256"])
    tampered["canonical_smiles_sha256"][0] = "0" * 64
    monkeypatch.setattr(prep_adapters, "load_lipo_row_manifest", lambda _path: tampered)
    with pytest.raises(ValueError, match="canonical SMILES hash mismatch"):
        BASE.fingerprint_rows([0])

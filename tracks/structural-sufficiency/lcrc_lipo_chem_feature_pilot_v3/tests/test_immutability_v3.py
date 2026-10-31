"""Frozen-artifact immutability and freeze-record tests."""

from __future__ import annotations

import json

import v3_common as C


def test_v1_and_v2_are_unchanged():
    v1 = C.assert_v1_unchanged()
    v2 = C.assert_v2_unchanged()
    assert v1["ok"], [c for c in v1["checks"] if not c["ok"]]
    assert v2["ok"], [c for c in v2["checks"] if not c["ok"]]
    assert v2["n_checks"] > 0


def test_historical_constants_match_the_v2_train_records():
    for arm in ("A", "C"):
        hist = C.HISTORICAL[arm]
        payload = C.read_json(C.V2_RUNS / hist["key"] / "train.json")
        assert abs(payload["metrics"]["dev_rmse_raw"] - hist["dev_rmse"]) < 1e-12
        assert abs(payload["metrics"]["dev_mae_raw"] - hist["dev_mae"]) < 1e-12
        assert payload["arm"] == arm
        assert payload["fold"] == 1
        assert payload["seed"] == 0
        assert payload["recipe"] == "R3"


def test_v2_manifest_covers_the_historical_prediction_files():
    if not C.V2_MANIFEST.exists():
        import pytest

        pytest.skip("v2 manifest not generated yet")
    manifest = C.read_json(C.V2_MANIFEST)
    files = manifest["files"]
    for arm in ("A", "C"):
        key = C.HISTORICAL[arm]["key"]
        for tail in ("train.json", "soup.pt", "predictions.npz"):
            rel = C.track_rel(C.V2_RUNS / key / tail)
            assert rel in files
            assert C.sha256_file(C.REPO / rel) == files[rel]


def test_frozen_extras_are_hash_pinned():
    if not C.ANALYSIS.joinpath("frozen_extras_v3.json").exists():
        import pytest

        pytest.skip("frozen extras not generated yet")
    payload = C.read_json(C.ANALYSIS / "frozen_extras_v3.json")
    assert payload["npz_sha256"] == C.sha256_file(C.ANALYSIS / "frozen_extras_v3.npz")
    for variant in ("chem12", "chem13"):
        meta = payload["variants"][variant]
        assert meta["n_extra"] == (12 if variant == "chem12" else 13)
        assert meta["audit"]["labels_used_for_features"] is False


def test_freeze_if_present_verifies():
    if not C.PROTOCOL_FREEZE.exists():
        import pytest

        pytest.skip("protocol freeze not generated yet")
    frozen = C.verify_protocol_freeze(require_clean=False)
    assert frozen["protocol_id"] == C.PROTOCOL_ID
    assert frozen["frozen_constants"]["fold"] == 1
    assert frozen["frozen_constants"]["seed"] == 0
    assert frozen["frozen_constants"]["recipe"] == "R3"
    assert frozen["gpu_hard_cap_hours"] == C.GPU_HARD_CAP_HOURS
    sha_file = C.ANALYSIS / "freeze_sha256_v3.txt"
    if sha_file.exists():
        assert sha_file.read_text(encoding="utf-8").strip() == C.freeze_sha256()


def test_historical_v2_is_not_listed_as_a_v3_run():
    for key in ("F1_A_sel_s0", "F1_C_sel_s0"):
        assert not (C.RUNS / key).exists()

from __future__ import annotations

import json

import pytest

import run_v2
import v2_common as C
import v2_registry as R


def _write_freeze(path):
    path.write_bytes(b'{"protocol_id":"lcrc-lipo-mechanism-generalization-ssl-v2"}\n')
    return C.sha256_file(path)


def _auth(freeze_hash, runs, **overrides):
    payload = {
        "authorized": True,
        "protocol_id": C.PROTOCOL_ID,
        "protocol_freeze_sha256": freeze_hash,
        "max_gpu_hours": 3.5,
        "authorized_runs": list(runs),
        "researcher_note": "test authorization",
    }
    payload.update(overrides)
    return payload


def test_gpu_authorization_is_missing_by_default(monkeypatch, tmp_path):
    monkeypatch.setattr(C, "GPU_AUTHORIZATION", tmp_path / "auth.json")
    monkeypatch.setattr(C, "PROTOCOL_FREEZE", tmp_path / "freeze.json")
    with pytest.raises(PermissionError, match="GPU_AUTHORIZATION_MISSING"):
        run_v2.require_gpu_authorization("F0_A_R0_s0")


def test_gpu_authorization_must_match_freeze_and_cap(monkeypatch, tmp_path):
    freeze = tmp_path / "freeze.json"
    fh = _write_freeze(freeze)
    auth_path = tmp_path / "auth.json"
    monkeypatch.setattr(C, "PROTOCOL_FREEZE", freeze)
    monkeypatch.setattr(C, "GPU_AUTHORIZATION", auth_path)
    runs = [r["key"] for r in R.planned_registry()["runs"]]
    cases = [
        (_auth(fh, runs, max_gpu_hours=4.0), "exceed"),
        (_auth(fh, runs, authorized_runs=runs + ["unknown"]), "unknown"),
        (_auth("0" * 64, runs), "does not match"),
        (_auth(fh, runs, researcher_note=""), "no researcher note"),
    ]
    for payload, match in cases:
        auth_path.write_text(json.dumps(payload), encoding="utf-8")
        with pytest.raises(PermissionError, match=match):
            run_v2.require_gpu_authorization("F0_A_R0_s0")


def test_gpu_authorization_accepts_a_registered_run(monkeypatch, tmp_path):
    freeze = tmp_path / "freeze.json"
    fh = _write_freeze(freeze)
    auth_path = tmp_path / "auth.json"
    monkeypatch.setattr(C, "PROTOCOL_FREEZE", freeze)
    monkeypatch.setattr(C, "GPU_AUTHORIZATION", auth_path)
    runs = [r["key"] for r in R.planned_registry()["runs"]]
    auth_path.write_text(json.dumps(_auth(fh, runs)), encoding="utf-8")
    got = run_v2.require_gpu_authorization("F0_A_R0_s0")
    assert got["max_gpu_hours"] == 3.5
    assert run_v2.authorization_matches_current_freeze() is True
    with pytest.raises(PermissionError, match="not covered"):
        run_v2.require_gpu_authorization("not_a_run")


def test_planned_registry_is_unique_and_within_budget():
    reg = R.planned_registry()
    keys = [r["key"] for r in reg["runs"]]
    assert len(keys) == len(set(keys))
    assert reg["n_runs"] == len(keys)
    budget = C.read_json(C.ANALYSIS / "budget_estimate_v2.json")
    assert budget["fits_within_hard_cap"] is True
    assert budget["estimated_total_gpu_hours"] <= 3.5


def test_primary_seed_plan_uses_two_seeds_only():
    import v2_registry as R

    assert C.PRIMARY_SEEDS == (0, 1)
    reg = R.planned_registry()
    assert all(r["seed"] in C.PRIMARY_SEEDS for r in reg["runs"])
    keys = {r["key"] for r in reg["runs"]}
    assert "F0_SSLpre_s2" not in keys
    assert "F0_A_SSL_sel_s2" not in keys
    assert "full_A_scratch_s2" not in keys
    assert "full_A_SSL_sel_s2" not in keys
    assert "F1_A_R0_s2" not in keys

"""Result-level checks for CJP-SSL v1 (run after the GPU phases).

These tests do not re-train anything: they verify that the committed analysis
artifacts are internally consistent, that provenance/transfer checks hold, and
that the frozen conclusion rules were applied to the frozen numbers.
Marked ``slow`` because the artifacts are large; skipped if absent.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

TRACK = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(TRACK / "code"))

import cjp_common as C  # noqa: E402

pytestmark = pytest.mark.slow

ANALYSIS = TRACK / "analysis"


def _load(name: str):
    p = ANALYSIS / name
    if not p.exists():
        pytest.skip(f"{p} missing (run the GPU phases first)")
    return json.loads(p.read_text())


def test_supervised_runs_complete_and_compatible():
    sup = _load("supervised_summary.json")
    assert sup["all_runs_present"] is True
    assert len(sup["rows"]) == 8
    assert sup["official_test_loaded"] is False
    for r in sup["rows"]:
        assert r["parameters"] == C.DEPLOY_PARAMETERS == 237519
        assert r["epochs"] == 600
        assert r["updates"] == 47400
        assert r["best_valid_mae"] > 0


def test_pretrain_same_samples_and_frozen_encoders():
    pre = _load("pretext_summary.json")
    assert pre["official_test_loaded"] is False
    assert len(pre["runs"]) == 4
    for seed in ("0", "1"):
        s = pre["same_samples_across_arms"][seed]
        assert s["masked_batch_sha256_equal"] is True
        assert s["first_batch_query_sha256_equal"] is True
        assert s["split_equal"] is True
        assert s["mask_coverage_equal"] is True
    for r in pre["runs"].values():
        assert r["updates_done"] == 7560
        assert r["encoder_bit_identical"] is True
        # frozen modules must not appear in the recorded gradient groups
        assert set(r["train"]["grad_norms_last_update"]) == {"local", "head"}
        assert r["train"]["mask_coverage"]["molecules_with_no_path"] == 0


def test_local_state_transfer_is_exact():
    sup = _load("supervised_summary.json")
    t = sup["transfer_checks"]["pretrain_to_supervised"]
    assert len(t) == 4
    for entry in t.values():
        assert entry["equal"] is True
        assert entry["pretrain_encoder_bit_identical"] is True
    for seed, entry in sup["transfer_checks"]["cross_arm_module_sharing"].items():
        for arm in ("MARG-SSL", "JOINT-SSL"):
            shared = entry[arm]
            for k in ("atom_encoder", "bond_encoder", "f", "rho"):
                assert shared[k] is True
            assert shared["local"] is False
        assert entry["MARG_vs_JOINT"]["only_local_differs"] is True


def test_frozen_verdict_rules_applied():
    sup = _load("supervised_summary.json")
    v = sup["verdicts"]
    assert v["CONCLUSION_D_NO_GAIN"] is True
    assert v["CONCLUSION_A_LOCAL_SSL_SUPPORTED"] is False
    assert v["CONCLUSION_B_JOINT_BINDING_SUPPORTED"] is False
    assert v["CONCLUSION_C_MARG_ONLY_GAIN"] is False
    # both new arms are worse than the same-round replay baseline
    for arm in ("MARG-SSL", "JOINT-SSL"):
        assert v[arm]["vs_A_REPLAY"]["seed_mean_delta"] > 0
    assert v["JOINT_vs_MARG"]["joint_pretext_ce_better"] is True


def test_paired_contrasts_present():
    pairs = _load("paired_comparisons.json")
    assert pairs["official_test_loaded"] is False
    for name in ("A-REPLAY_minus_SCRATCH", "MARG-SSL_minus_A-REPLAY",
                 "JOINT-SSL_minus_MARG-SSL", "JOINT-SSL_minus_A-REPLAY"):
        entry = pairs["pairs"][name]
        assert entry["seed_mean"]["seeds"] == 2
        assert entry["seed_mean"]["n"] == 1000
        assert set(entry["per_seed"]) == {"0", "1"}
    old_ssl = pairs["pairs"]["A-REPLAY_minus_SCRATCH"]["seed_mean"]
    assert old_ssl["mean_delta"] < 0  # replay beats scratch


def test_budget_within_limits():
    led = _load("gpu_ledger.json")
    assert led["n_entries"] == 12
    assert led["total_gpu_hours"] <= 10.0
    assert led["within_soft_limit"] is True
    assert led["within_hard_limit"] is True
    assert led["official_test_loaded"] is False

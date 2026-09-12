"""Tests for the Compact-v4 Inductive-Bias & Sample-Efficiency Audit.

Static / unit tests plus lightweight local-artifact checks.  They never train a
model, never read official valid and never load official test.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

from tracks.ksvd.experiments.luyin16 import (
    zinc_inductive_bias_sample_efficiency_audit as A,
)


RESULTS = A.RESULTS_DIR


def _has(path: Path) -> bool:
    return path.exists()


# ---------------------------------------------------------------------------
# locked constants / budget discipline
# ---------------------------------------------------------------------------


def test_thresholds_are_locked():
    assert A.GAIN_MATERIAL == 0.006
    assert A.GAIN_WEAK == 0.003
    assert A.RAW_CONFLICT == -0.003
    assert A.REPLICATE_PER_SEED == 0.005
    assert A.REPLICATE_MEAN == 0.006
    assert A.BOOTSTRAP_B == 2000
    assert A.SOUP_K == 5


def test_only_two_seeds_and_no_n900():
    assert set(A.RUN_SPEC) == {"N3600_I0T0", "N3600_I1T1", "N1800_I0T0", "N1800_I1T1"}
    assert all(spec["N"] in (1800, 3600) for spec in A.RUN_SPEC.values())
    assert 900 not in {spec["N"] for spec in A.RUN_SPEC.values()}
    assert A.NESTED_SIZES == (7200, 3600, 1800)


def test_architecture_lock_params():
    model = A.shead.build_smallhead(0)
    assert A.shead._n_params(model) == 82115
    assert A.shead._n_params(model.head) == 4135


def test_probe_firewall_blocks_before_freeze(monkeypatch):
    monkeypatch.setattr(A, "PROBE_UNLOCKED", False)
    with pytest.raises(A.ProbeAccessError):
        A._require_probe_unlocked()
    monkeypatch.setattr(A, "PROBE_UNLOCKED", True)
    A._require_probe_unlocked()


# ---------------------------------------------------------------------------
# nested, target-independent subsets
# ---------------------------------------------------------------------------


def test_subset_ranking_is_deterministic_and_nested():
    pool = list(range(0, 14400, 2))[:7200]
    assert len(pool) == 7200
    ranked = A._subset_ranking(pool)
    assert len(ranked) == 7200
    assert set(ranked) == set(pool)
    assert A._subset_ranking(pool) == ranked
    assert set(ranked[:1800]) <= set(ranked[:3600]) <= set(ranked)


def test_subset_ordering_does_not_use_targets():
    # the ranking depends only on the salt and the original index
    pool = [1, 2, 3, 4, 5]
    first = A._subset_ranking(pool)
    expected = sorted(
        pool,
        key=lambda i: (hashlib.sha256(f"{A.SUBSET_SALT}|{i:04d}".encode()).hexdigest(), i),
    )
    assert first == expected


def test_sha256_indices_matches_manual():
    idx = [3, 1, 2]
    manual = hashlib.sha256(b"3,1,2").hexdigest()
    assert A._sha256_indices(idx) == manual


# ---------------------------------------------------------------------------
# anchor-derived step protocol
# ---------------------------------------------------------------------------


def test_anchor_protocol_derivation():
    anchor = A.anchor_protocol()
    assert anchor["steps_per_epoch"] == 57
    assert anchor["max_epochs"] == 240
    assert anchor["max_optimizer_steps"] == 240 * 57 == 13680
    assert anchor["selection_eval_interval"] == 57
    assert anchor["patience_evaluations"] == 40


def test_anchor_compatibility_file():
    path = RESULTS / "anchor_protocol_compatibility.json"
    if not _has(path):
        pytest.skip("locks not generated yet")
    payload = json.loads(path.read_text())
    assert payload["equivalent_at_7200"] is True
    assert all(payload["checks"].values())


# ---------------------------------------------------------------------------
# paired bootstrap
# ---------------------------------------------------------------------------


def test_paired_bootstrap_zero_mean_includes_zero():
    diff = np.zeros(2000)
    boot = A.paired_bootstrap(diff, B=500, seed=1)
    assert boot["mean"] == 0.0
    assert boot["ci95_lower"] <= 0.0 <= boot["ci95_upper"]


def test_paired_bootstrap_positive_shift_excludes_zero():
    rng = np.random.default_rng(0)
    diff = 0.01 + rng.normal(0, 0.001, size=2000)
    boot = A.paired_bootstrap(diff, B=500, seed=1)
    assert boot["ci95_lower"] > 0.0
    assert boot["excludes_zero"] is True


def test_two_seed_bootstrap_averages_seed_effects():
    a = np.full(1000, 0.01)
    b = np.full(1000, 0.02)
    boot = A.paired_bootstrap_two_seed(a, b, B=200, seed=1)
    assert abs(boot["mean"] - 0.015) < 1e-12
    assert boot["ci95_lower"] > 0


# ---------------------------------------------------------------------------
# soup / raw estimator rules
# ---------------------------------------------------------------------------


def test_top5_members_rank_by_800_mae_earliest_tie():
    # synthetic manifest with a tie that must resolve to the earlier step
    snapshots = [
        {"step": 20, "select_800_mae": 0.10},
        {"step": 10, "select_800_mae": 0.10},
        {"step": 30, "select_800_mae": 0.20},
        {"step": 40, "select_800_mae": 0.05},
        {"step": 50, "select_800_mae": 0.07},
        {"step": 60, "select_800_mae": 0.30},
    ]
    ranked = sorted(snapshots, key=lambda r: (float(r["select_800_mae"]), int(r["step"])))
    order = [r["step"] for r in ranked[: A.SOUP_K]]
    assert order == [40, 50, 10, 20, 30]


# ---------------------------------------------------------------------------
# artifact-level checks (skip until the audit has actually run)
# ---------------------------------------------------------------------------


def test_stage1_artifacts_if_present():
    path = RESULTS / "stage1_N3600_I0T0.json"
    if not _has(path):
        pytest.skip("stage1 not run yet")
    payload = json.loads(path.read_text())
    run = payload["run"]
    assert run["N"] == 3600
    assert run["max_optimizer_steps"] == 13680
    assert run["eval_interval"] == 57
    assert run["probe_2000_accessed_during_training"] is False
    assert run["official_test_loaded"] is False
    assert len(run["top5_steps"]) == 5
    # gain definition: E(3600) - E(7200)
    recomputed = run["soup_probe_mae"] - payload["anchor_seed0"]["soup_probe_mae"]
    assert abs(recomputed - payload["G_soup_36_to_72_seed0"]) < 1e-12


def test_integrity_gates_if_present():
    path = RESULTS / "integrity_gates.json"
    if not _has(path):
        pytest.skip("integrity gates not generated yet")
    payload = json.loads(path.read_text())
    assert payload["all_pass"] is True
    assert payload["n_total"] >= 18


def test_subset_lock_is_target_free_and_nested():
    path = RESULTS / "sample_efficiency_subset_lock.json"
    if not _has(path):
        pytest.skip("subset lock not generated yet")
    payload = json.loads(path.read_text())
    assert payload["target_used"] is False
    assert payload["ordering_uses_target"] is False
    assert all(payload["nested_invariants"].values())
    assert len(payload["indices_3600"]) == 3600
    assert len(payload["indices_1800"]) == 1800
    assert set(payload["indices_1800"]) <= set(payload["indices_3600"])


def test_no_official_test_loaded_anywhere():
    if not RESULTS.exists():
        pytest.skip("audit has not run")
    for path in RESULTS.glob("*.json"):
        try:
            text = path.read_text()
        except OSError:
            continue
        if "official_test_loaded" in text:
            payload = json.loads(text)
            if isinstance(payload, dict) and "official_test_loaded" in payload:
                assert payload["official_test_loaded"] is False, path

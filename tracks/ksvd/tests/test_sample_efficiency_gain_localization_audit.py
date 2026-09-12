"""Tests for the Sample-Efficiency Gain Localization Audit.

Static / unit tests plus lightweight frozen-artifact checks.  They never train a
model, never read official valid and never load official test.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from tracks.ksvd.experiments.luyin16 import (
    zinc_sample_efficiency_gain_localization_audit as A,
)


RESULTS = A.RESULTS_DIR
INTEGRITY = RESULTS / "integrity_tests.json"
ANSWERS = RESULTS / "answers_q1_q24.json"
DECISION = RESULTS / "mechanism_decision.json"
SUPPORT = RESULTS / "support_table_locked.parquet"
SUPPORT_LOCK = RESULTS / "support_table_lock.json"


def _has(path: Path) -> bool:
    return path.exists()


# ---------------------------------------------------------------------------
# locked constants / budget discipline
# ---------------------------------------------------------------------------


def test_thresholds_are_locked():
    assert A.PROTOCOL_VERSION == "sample_efficiency_gain_localization_v1"
    assert A.PROBE_N == 2000
    assert A.NESTED_SIZES == (1800, 3600, 7200)
    assert A.SEEDS == (0, 1)
    assert A.RARE_DF == 5
    assert A.NN_K == 8
    assert A.QUARTILE_TARGET == 0.010
    assert A.SPEARMAN_TARGET == 0.15
    assert A.BOOTSTRAP_B == 2000
    assert A.WELL_COVERED_MIN == 200
    assert A.COVERAGE_RETAIN_HIGH == 0.60
    assert A.COVERAGE_RETAIN_LOW == 0.40
    assert A.N_FOLDS == 5


def test_only_three_families_and_size_controls():
    assert A.FAMILIES == ("token", "relation", "neighbor_density")
    assert set(A.TRANSITIONS) == {"18_36", "36_72"}
    # size controls are confound controls only, never a mechanism family
    assert A.TOKEN_FAMILY not in ("size",)
    assert "size" not in A.FAMILIES


def test_no_n900_and_nested_subsets_are_exact():
    subsets = A.nested_subsets()
    assert set(subsets) == {"1800", "3600", "7200"}
    s18, s36, s72 = (set(subsets["1800"]), set(subsets["3600"]), set(subsets["7200"]))
    assert s18 < s36 < s72
    assert len(s72) == 7200
    assert 900 not in A.NESTED_SIZES


def test_architecture_lock_params():
    model = A.shead.build_smallhead(0)
    assert A.shead._n_params(model) == 82115
    assert A.shead._n_params(model.head) == 4135


# ---------------------------------------------------------------------------
# phase firewall
# ---------------------------------------------------------------------------


def test_target_firewall_blocks_before_unlock(monkeypatch):
    monkeypatch.setattr(A, "_TARGETS_UNLOCKED", False)
    with pytest.raises(A.ProbeAccessError):
        A._require_targets_unlocked()


def test_firewall_installs_loader_guards():
    # _install_firewall monkeypatches upstream loaders; the blocked exit points
    # must raise rather than read official valid/test.  Save/restore the original
    # callables so this test cannot leak into the rest of the pytest session.
    saved = {
        "zlr": A.zlr._load_zinc,
        "suff_test": A.suff.extract_test_records,
        "suff_tv": A.suff.load_train_valid_records,
        "postv4": A.postv4._extract_v4_records,
    }
    try:
        A._install_firewall()
        with pytest.raises(A.FirewallError):
            A.suff.extract_test_records()
        with pytest.raises(A.FirewallError):
            A.suff.load_train_valid_records()
    finally:
        A.zlr._load_zinc = saved["zlr"]
        A.suff.extract_test_records = saved["suff_test"]
        A.suff.load_train_valid_records = saved["suff_tv"]
        A.postv4._extract_v4_records = saved["postv4"]
    assert A.OFFICIAL_VALID_LOADED is False
    assert A.OFFICIAL_TEST_LOADED is False


# ---------------------------------------------------------------------------
# target-free support primitives
# ---------------------------------------------------------------------------


def test_document_frequency_counts_unique_molecules_once():
    sets = [{"a", "b"}, {"a"}, {"b", "c"}]
    df = A._document_frequency(sets, [0, 1, 2])
    assert df["a"] == 2
    assert df["b"] == 2
    assert df["c"] == 1
    # the real pipeline passes unique nested-subset indices; verify directly
    # that df is a document frequency, not an optimizer-exposure count.
    subsets = A.nested_subsets()
    for key in ("1800", "3600", "7200"):
        idx = subsets[key]
        assert len(idx) == len(set(idx)), f"{key} is not a unique molecule set"


def test_probe_distribution_metrics_are_target_free():
    train_sets = [{"a", "b"}]
    df = A._document_frequency(train_sets, [0])
    probe_sets = [{"a"}, {"b"}, {"c"}, {"a", "c"}]
    m = A._probe_distribution_metrics(probe_sets, df, [0, 1, 2, 3])
    assert m["unseen"].tolist() == [0.0, 0.0, 1.0, 0.5]
    assert (m["rare"] >= m["unseen"]).all()
    assert m["meanlog"].shape == (4,)


def test_spearman_matches_scipy():
    from scipy.stats import spearmanr

    rng = np.random.default_rng(0)
    a = rng.normal(size=200)
    b = 0.3 * a + rng.normal(size=200)
    assert abs(A._spearman(a, b) - spearmanr(a, b).statistic) < 1e-9
    # constant vector -> zero correlation (no NaN)
    assert A._spearman(np.ones(50), rng.normal(size=50)) == 0.0


def test_quartile_labels_are_monotone_in_value():
    x = np.arange(400, dtype=float)
    labels = A._quartile_labels(x, thresholds=[100.0, 200.0, 300.0])
    assert labels[0] == 1 and labels[-1] == 4
    assert (np.diff(labels) >= 0).all()


def test_size_controls_are_target_independent():
    lock_path = RESULTS / "size_control_lock.json"
    if not _has(lock_path):
        pytest.skip("size_control_lock.json not built yet")
    lock = json.loads(lock_path.read_text())
    assert lock["target_independent"] is True
    assert lock["target_related_controls"] == []


# ---------------------------------------------------------------------------
# frozen PATCH_FULL normalization must not be refit per N
# ---------------------------------------------------------------------------


def test_patchfull_scales_are_frozen():
    scores = json.loads((A.RAW_AUDIT_DIR / "distance_scale_stats.json").read_text())["median_distance"]
    assert {"B1_identity", "B2_patch_numeric", "B3_pair_relation", "B4_global_topology"} <= set(scores)
    # the audit module must read, not refit, these scales
    src = Path(A.__file__).read_text()
    assert "distance_scale_stats.json" in src
    assert "def _patchfull_distance" in src


# ---------------------------------------------------------------------------
# artifacts (skipped if the audit has not been run end-to-end)
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has(INTEGRITY), reason="integrity_tests.json not built yet")
def test_all_twenty_integrity_tests_pass():
    integrity = json.loads(INTEGRITY.read_text())
    assert integrity["n_total"] == 20
    assert integrity["n_pass"] == 20
    assert integrity["all_pass"] is True
    assert integrity["official_valid_loaded"] is False
    assert integrity["official_test_loaded"] is False


@pytest.mark.skipif(not _has(SUPPORT), reason="support table not built yet")
def test_support_table_is_target_free_and_locked():
    frame = pd.read_parquet(SUPPORT)
    columns = set(frame.columns)
    assert len(frame) == A.PROBE_N
    # no target / prediction columns may appear in the locked table
    forbidden = [c for c in columns if "target" in c or "pred" in c or c.startswith("y_")]
    assert forbidden == []
    # required target-free support columns
    for col in ("token_deficiency", "relation_deficiency", "nn8_distance"):
        assert f"{col}_1800" in columns
        assert f"{col}_3600" in columns
    lock = json.loads(SUPPORT_LOCK.read_text())
    assert lock["target_used"] is False
    assert lock["prediction_used"] is False
    assert lock["quartiles_locked"] is True


@pytest.mark.skipif(not _has(ANSWERS), reason="answers_q1_q24.json not built yet")
def test_answers_q1_q24_complete():
    answers = json.loads(ANSWERS.read_text())
    for i in range(1, 25):
        matches = [k for k in answers if k.startswith(f"Q{i}_")]
        assert matches, f"missing Q{i}"
    assert answers["official_valid_loaded"] is False
    assert answers["official_test_loaded"] is False


@pytest.mark.skipif(not _has(DECISION), reason="mechanism_decision.json not built yet")
def test_decision_case_is_in_enum():
    decision = json.loads(DECISION.read_text())
    assert decision["case"] in set("ABCDEFGH")
    assert decision["official_valid_loaded"] is False
    assert decision["official_test_loaded"] is False
    # no family may pass the full strong gate if the primary stress test is
    # underpowered and no family is listed as strong.
    if not decision["well_covered"]["powered"] and not decision["strong_families"]:
        assert decision["case"] == "H"
    # family verdicts must be one of the pre-registered labels
    allowed = {
        "STRONG_SUPPORT_LIMITED_SIGNAL",
        "NON_ROBUST_ASSOCIATION__strong_quartile_and_size_adjusted_but_weak_continuous",
        "SUPPORT_SIGNAL_EXPLAINED_BY_SIZE_OR_WEAK_AFTER_ADJUSTMENT",
        "REGIME_SPECIFIC_OR_PARTIAL_SUPPORT_SIGNAL",
        "NO_MATERIAL_SUPPORT_ASSOCIATION",
    }
    for verdict in decision["family_verdicts"].values():
        assert verdict in allowed


@pytest.mark.skipif(not _has(RESULTS / "bootstrap_provenance_check.json"), reason="bootstrap not built yet")
def test_bootstrap_provenance_matches_prior():
    check = json.loads((RESULTS / "bootstrap_provenance_check.json").read_text())
    assert check["all_match_prior"] is True
    assert check["point_estimate_in_ci"] is True
    for comparison in check["prior_comparison"].values():
        assert comparison.get("mean_diff", 0.0) < 1e-12

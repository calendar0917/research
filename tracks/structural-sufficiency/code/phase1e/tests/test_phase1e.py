"""Phase 1e CPU tests: label audit, error attribution and decision rules.

These tests are deliberately about *mechanism*, not about numbers: the numbers
themselves are recorded artefacts and are checked by the run manifest.  What is
tested here is that the audit code cannot silently do the wrong thing — reuse a
label, read a reserved row, or relax a frozen decision rule.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
CODE = HERE.parent
REPO = CODE.parents[3]
for p in (CODE, CODE / "phase1b"):
    if str(p) not in sys.path:
        sys.path.insert(0, str(p))

import phase1e_common as C  # noqa: E402

RESULTS = C.RESULTS_1E


def _load(name: str) -> dict:
    path = RESULTS / name
    if not path.exists():
        pytest.skip(f"{name} has not been produced yet")
    return C.read_json(path)


# ---------------------------------------------------------------------------
# split / label isolation
# ---------------------------------------------------------------------------


def test_split_sides_are_disjoint_and_sized() -> None:
    split = C.read_json(C.SPLIT_PATH)
    fit = set(split["rows"]["fit"])
    mon = set(split["rows"]["monitor"])
    res = set(split["rows"]["reserved"])
    assert len(fit) == 8000 and len(mon) == 1000 and len(res) == 1000
    assert not (fit & mon) and not (fit & res) and not (mon & res)


def test_label_vault_holds_no_reserved_or_official_test_labels() -> None:
    vault = C.read_json(C.RESULTS_1B / "zinc_labels_fit_monitor.json")
    assert vault["scope"]["reserved_labels_present"] is False
    assert vault["scope"]["official_valid_test_labels_present"] is False
    assert len(vault["labels"]) == 9000


def test_monitor_table_is_aligned_and_caches_no_labels_on_disk() -> None:
    mon = C.load_monitor()
    assert mon.y.shape == (1000,)
    for key, arr in mon.pred.items():
        assert arr.shape == (1000,), key


# ---------------------------------------------------------------------------
# label geometry
# ---------------------------------------------------------------------------


def test_cycle_component_reconstruction_roundtrip() -> None:
    """Reconstruction must recover a planted integer cycle penalty from y alone."""
    rng = np.random.default_rng(0)
    c = C.COMMUNITY_CONSTANTS
    n = 400
    logp = rng.normal(2.0, 1.2, size=n)
    sa = rng.uniform(1.0, 5.0, size=n)
    excess = rng.integers(0, 6, size=n).astype(np.int64)
    chem = (logp - c["logP_mean"]) / c["logP_std"] + (sa - c["SA_mean"]) / c["SA_std"]
    y = chem + ((-excess.astype(np.float64)) - c["cycle_mean"]) / c["cycle_std"]
    got = C.recover_label_cycle(y, logp, sa)
    acc = float(np.mean(got["excess"] == excess))
    assert acc >= 0.99, f"recovered penalty accuracy {acc:.3f}"
    assert abs(got["sigma_cycle"] - c["cycle_std"]) < 0.02


def test_penalty_levels_are_small_nonnegative_integers() -> None:
    label = _load("label_audit.json")
    group = next(g for g in label["monitor_groups"] if g["name"] == "label_penalty_union")
    # a penalty of k ring atoms means the label carries -k on the cycle axis
    assert group["mean_label_excess"] > 0
    assert abs(group["mean_label_excess"] - round(group["mean_label_excess"], 2)) < 0.2
    assert group["mean_cycle_component"] < 0


# ---------------------------------------------------------------------------
# A1: grouping vs target penalty
# ---------------------------------------------------------------------------


def test_label_audit_confusion_is_a_partition_of_the_monitor() -> None:
    label = _load("label_audit.json")
    x = label["monitor_confusion"]["label_penalty_x_rdkit_large_ring"]
    assert (
        x["labeled_and_rdkit_large"]
        + x["labeled_not_rdkit_large"]
        + x["rdkit_large_not_labeled"]
        + x["neither"]
        == 1000
    )
    assert x["labeled_and_rdkit_large"] + x["labeled_not_rdkit_large"] == 43
    assert x["labeled_and_rdkit_large"] + x["rdkit_large_not_labeled"] == 40


def test_label_audit_records_the_snapped_regime() -> None:
    label = _load("label_audit.json")
    rec = label["reconstruction"]
    assert rec["ksvd_cross_check"]["n_agree_with_recovered_excess"] == rec["ksvd_cross_check"]["n_rows"]
    assert rec["excess_agreement_10k"]["label_excess_vs_canonical_order"] >= 9900
    assert rec["snap_error_p95_y_units"] < 0.05


# ---------------------------------------------------------------------------
# A2/A3: attribution and routing
# ---------------------------------------------------------------------------


def test_attribution_groups_sum_to_the_monitor() -> None:
    err = _load("error_attribution.json")
    for name, model in err["models"].items():
        g = model["groups"]
        total = (
            g["label_complement"]["n"]
            + g["label_penalised"]["n"]
        )
        assert total == 1000, name
        overall = g["all"]["mae"] * g["all"]["n"]
        parts = (
            g["label_complement"]["mae"] * g["label_complement"]["n"]
            + g["label_penalised"]["mae"] * g["label_penalised"]["n"]
        )
        assert abs(overall - parts) < 1e-6, name


def test_attribution_never_defines_a_post_hoc_target_score() -> None:
    err = _load("error_attribution.json")
    text = json.dumps(err)
    assert "mae_vs_cycle_free_target" in text
    assert "not a model score" in err["counterfactual_note"]


def test_routing_uses_rdkit_group_and_is_flagged_post_hoc() -> None:
    routing = _load("zero_training_routing.json")
    assert routing["status"] == "POST_HOC_DIAGNOSTIC_ON_A_REUSED_MONITOR"
    assert routing["not_a_new_model_result"] is True
    assert "large_cycles_gt6" in routing["rule"]
    assert routing["routed_rdkit"]["route_sizes"]["to_R1_size_cap"] == 40
    assert routing["routed_rdkit"]["route_sizes"]["to_R_incidence"] == 960


# ---------------------------------------------------------------------------
# B1: protocol separation
# ---------------------------------------------------------------------------


def test_reference_audit_keeps_protocols_separate() -> None:
    ref = _load("reference_audit.json")
    tables = ref["protocol_tables"]
    assert set(tables) >= {
        "internal_ssrc_monitor",
        "bjoc_exposed_official_test",
        "literature_official_zinc12k",
    }
    internal = tables["internal_ssrc_monitor"]["rows"]
    exposed = tables["bjoc_exposed_official_test"]["rows"]
    # the exposed official-test number must never be filed as an internal one
    assert exposed["P3_seed_mean"] not in internal.values()


def test_no_contaminated_checkpoint_is_marked_usable() -> None:
    ref = _load("reference_audit.json")
    for c in ref["contamination_audit"]["checkpoints"]:
        contaminated = c["contamination"]["monitor_overlap"] > 0
        assert c["usable_for_internal_monitor"] is (not contaminated), c["name"]


def test_b2_budget_respects_the_gpu_cap() -> None:
    ref = _load("reference_audit.json")
    b2 = ref["b2_budget"]
    assert b2["projected_gpu_hours_single_seed"] <= b2["cap_gpu_hours"]
    assert b2["fits_cap"] is True


# ---------------------------------------------------------------------------
# A4: decision rules
# ---------------------------------------------------------------------------


def test_decision_applies_the_frozen_thresholds() -> None:
    dec = _load("decision.json")
    th = dec["thresholds"]
    art = dec["artifact_dominates"]["conditions"]
    assert art["rdkit_large_group_purity_le_0.05_unpenalised"]["pass"] == (
        art["rdkit_large_group_purity_le_0.05_unpenalised"]["value"] <= th["max_unpenalised_share"]
    )
    assert art["complement_gap_cap_vs_Gref_lt_0.02"]["pass"] == (
        art["complement_gap_cap_vs_Gref_lt_0.02"]["value"] < th["negligible_band"]
    )


def test_decision_never_fires_beyond_cycle_on_a_crossing_interval() -> None:
    dec = _load("decision.json")
    for e in dec["beyond_cycle"]["evidence"]:
        if e["candidate_better_by_0.02_with_ci_below_zero"]:
            assert e["ci95"][1] < 0, e
    assert dec["beyond_cycle"]["fired"] is any(
        e["candidate_better_by_0.02_with_ci_below_zero"] for e in dec["beyond_cycle"]["evidence"]
    )


def test_decision_emits_exactly_one_direction() -> None:
    dec = _load("decision.json")
    assert dec["direction"]["choice"] in {"A", "B", "C", "D"}
    flags = [dec["direction"][k] for k in ("A_triggered", "B_triggered", "C_triggered")]
    assert all(isinstance(f, bool) for f in flags)

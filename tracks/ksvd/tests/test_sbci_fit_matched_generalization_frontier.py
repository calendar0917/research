"""Tests for the SBCI fit-matched generalization frontier (zero-training).

These tests never train, never touch official valid/test, and never modify the
frozen artifacts.  They verify run identity, protocol comparability, the Phase F
fit-only firewall, deterministic train-fit matching, the match-quality gates,
the 800/2000 firewall, and the final no-family Case C verdict.
"""

from __future__ import annotations

import functools
import json

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_sbci as sb
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import (
    zinc_sbci_fit_matched_generalization_frontier as fmf,
)


@functools.lru_cache(maxsize=1)
def _data_and_subsets():
    data = fmf.load_data()
    subsets = fmf.build_subsets(data)
    return data, subsets


def _json(name: str) -> dict:
    return json.loads((fmf.RESULTS_DIR / name).read_text(encoding="utf-8"))


# --- 1 / 2 : run identity --------------------------------------------------


def test_01_baseline_run_is_n3600_i0_t0():
    inv = fmf._run_inventory("baseline")
    assert inv["N"] == 3600 and inv["I"] == 0 and inv["T"] == 0
    assert inv["family"] == "compact-v4-smallhead"
    assert inv["parameters"] == 82115
    assert inv["init_state_sha256_matches"] is True


def test_02_sbci_run_is_n3600_i0_t0():
    inv = fmf._run_inventory("sbci")
    assert inv["N"] == 3600 and inv["I"] == 0 and inv["T"] == 0
    assert inv["family"] == "compact-v4-SBCI"
    assert inv["parameters"] == sb._n_params(sb.build_sbci(0))
    assert inv["init_state_sha256_matches"] is True


# --- 3 / 4 / 5 : index identity --------------------------------------------


def test_03_d3600_exact_same_indices():
    data, subsets = _data_and_subsets()
    comp = fmf._read_json(fmf.RESULTS_DIR / "protocol_compatibility.json")
    assert comp["comparable"] is True
    assert comp["checks"]["same_d3600_indices"] is True
    lock = _json_baseline("sample_efficiency_subset_lock.json")
    assert comp["data_fingerprints"]["train_subset_index_hash"] == lock["dataset_hash_3600"]
    assert len(subsets["sorted_3600"]) == 3600


def test_04_800_exact_same_indices():
    comp = _json("protocol_compatibility.json")
    ref = _json_baseline("split_inventory.json")
    assert comp["checks"]["same_800_indices"] is True
    assert comp["data_fingerprints"]["select_index_hash"] == (
        ref["index_sha256"]["checkpoint_selection"]
    )
    assert comp["data_fingerprints"]["n_select_800"] == 800


def test_05_2000_exact_same_indices():
    comp = _json("protocol_compatibility.json")
    ref = _json_baseline("split_inventory.json")
    assert comp["checks"]["same_2000_indices"] is True
    assert comp["data_fingerprints"]["probe_index_hash"] == ref["index_sha256"]["internal_probe"]
    assert comp["data_fingerprints"]["n_probe_2000"] == 2000


# --- 6 / 7 : zero-training discipline --------------------------------------


def test_06_zero_gradient_updates():
    fmf._install_zero_training_guard()
    try:
        assert fmf._ZERO_TRAINING_GUARD["installed"] is True
        assert fmf._ZERO_TRAINING_GUARD["violations"] == 0
        value = torch.ones(2, requires_grad=True)
        with pytest.raises(fmf.ZeroTrainingViolation):
            (value * 2).sum().backward()
    finally:
        fmf._uninstall_zero_training_guard()
    assert fmf._ZERO_TRAINING_GUARD["installed"] is False
    guard = _json("integrity_tests.json")["tests"]["Test6_zero_gradient_updates"]
    assert guard is True


def test_07_all_snapshot_hashes_resolve():
    for model in fmf.MODELS:
        for member in fmf.manifest_snapshots(model):
            path = fmf.Path(member["snapshot"])
            assert path.exists()
            assert fmf._sha256_file(path) == member["snapshot_sha256"]
    assert _json("integrity_tests.json")["tests"]["Test7_all_snapshot_hashes_resolve"] is True


# --- 8 / 9 / 10 : Phase F fit-only -----------------------------------------


def test_08_recomputed_d3600_matches_stored():
    summary = _json("fit_inventory.summary.json")
    assert summary["recompute_matches_stored"] is True
    assert summary["max_abs_diff_recomputed_vs_stored"] <= 1e-10
    assert summary["row_counts"] == {"baseline": 106, "sbci": 102}
    assert summary["n_rows"] == 208
    assert summary["selection_800_loaded"] is False
    assert summary["probe_2000_loaded"] is False


def test_09_eligibility_uses_only_step_and_train_mae():
    overlap = _json("train_fit_overlap.json")
    assert overlap["eligibility_uses_only_step_and_train_mae"] is True
    assert overlap["selection_800_loaded"] is False
    assert overlap["probe_2000_loaded"] is False
    assert overlap["common_step_threshold"] == pytest.approx(0.25 * overlap["s_common"])
    assert overlap["s_common"] == min(
        overlap["final_optimizer_step_baseline"], overlap["final_optimizer_step_sbci"]
    )


def test_10_fit_anchors_without_holdout_access():
    phase_f = _json("phaseF_fit_only_lock.json")
    assert phase_f["selection_800_loaded"] is False
    assert phase_f["probe_2000_loaded"] is False
    assert phase_f["matching_used_train_only"] is True
    assert phase_f["fit_anchor_lock_present"] is True
    # the anchors are entirely determined by the D3600 training fit
    overlap = _json("train_fit_overlap.json")
    anchors = overlap["anchors"]
    lo, hi = overlap["trimmed_interval"]
    assert len(anchors) == fmf.K_ANCHORS == 7
    assert anchors[0] == pytest.approx(lo)
    assert anchors[-1] == pytest.approx(hi)
    assert all(lo <= a <= hi for a in anchors)


# --- 11 / 12 / 13 / 14 / 15 : deterministic matching and quality gates ------


def test_11_matching_cost_uses_only_train_mae():
    lock = _json("fit_anchor_lock.json")
    assert lock["matching_cost_is_train_mae_only"] is True
    assert "1e-12" in lock["tie_break"]
    # no held-out quantity appears in the lock records
    for pair in lock["matched_snapshot_pairs"]:
        assert "select_800_mae" not in pair
        assert "probe" not in pair
        assert "soup" not in json.dumps(pair)


def test_12_matched_snapshots_unique_per_family():
    lock = _json("fit_anchor_lock.json")
    for key in ("baseline_step", "sbci_step"):
        steps = [int(p[key]) for p in lock["matched_snapshot_pairs"]]
        assert len(steps) == len(set(steps)), key
    assert _json("integrity_tests.json")["tests"]["Test12_matched_snapshots_unique_per_family"]


def test_13_valid_fit_gap_le_0_002():
    lock = _json("fit_anchor_lock.json")
    valid = [p for p in lock["matched_snapshot_pairs"] if int(p["valid"]) == 1]
    assert valid
    for pair in valid:
        assert float(pair["cross_model_fit_gap"]) <= fmf.CROSS_MODEL_GAP_TOL
        assert float(pair["baseline_proximity"]) <= fmf.ANCHOR_PROXIMITY_TOL
        assert float(pair["sbci_proximity"]) <= fmf.ANCHOR_PROXIMITY_TOL


def test_14_median_fit_gap_le_0_001():
    quality = _json("match_quality.json")
    assert quality["median_gap_gate_pass"] is True
    assert float(quality["median_cross_model_fit_gap"]) <= fmf.MEDIAN_GAP_TOL


def test_15_at_least_5_valid_anchors():
    lock = _json("fit_anchor_lock.json")
    assert int(lock["k_valid"]) >= fmf.MIN_VALID_ANCHORS
    assert lock["locked"] is True
    assert _json("match_quality.json")["k_valid_gate_pass"] is True


# --- 16 / 17 / 18 : phase firewalls ----------------------------------------


def test_16_800_loaded_only_after_fit_anchor_lock():
    stage1 = _json("stage1_800_decision.json")
    lock = _json("fit_anchor_lock.json")
    assert stage1["fit_anchor_lock_sha256"] == lock["lock_sha256"]
    assert stage1["selection_generalization_diagnostic_set"] is True
    assert stage1["not_pristine_holdout"] is True
    assert _json("integrity_tests.json")["tests"][
        "Test16_800_loaded_only_after_fit_anchor_lock"
    ]


def test_17_2000_loaded_only_if_stage1_authorises():
    stage1 = _json("stage1_800_decision.json")
    stage1_authorized = bool(stage1["authorize_2000"])
    has_stage2 = (fmf.RESULTS_DIR / "stage2_2000_decision.json").exists()
    assert (not has_stage2) or stage1_authorized
    # in the observed Case-C crossing, 2000 must not have been opened
    if not stage1_authorized:
        assert not has_stage2
        assert not (fmf.RESULTS_DIR / "stage2_2000_per_anchor.csv").exists()
        assert not (fmf.RESULTS_DIR / "stage2_2000_per_molecule.npy").exists()


def test_18_2000_uses_exact_same_locked_checkpoints():
    if (fmf.RESULTS_DIR / "stage2_2000_decision.json").exists():
        stage2 = _json("stage2_2000_decision.json")
        assert stage2["same_locked_checkpoints"] is True
        p2 = fmf._read_csv(fmf.RESULTS_DIR / "stage2_2000_per_anchor.csv")
        lock = _json("fit_anchor_lock.json")
        valid = [p for p in lock["matched_snapshot_pairs"] if int(p["valid"]) == 1]
        for a, b in zip(p2, valid):
            assert a["baseline_hash"] == b["baseline_hash"]
            assert a["sbci_hash"] == b["sbci_hash"]
    else:
        assert _json("integrity_tests.json")["tests"][
            "Test18_2000_uses_exact_same_locked_checkpoints"
        ] is True


# --- 19 / 20 : official locks ----------------------------------------------


def test_19_official_valid_never_loaded():
    for name in _all_artifacts():
        payload = _json(name)
        assert payload.get("official_valid_used") is False, name
    assert _json("integrity_tests.json")["tests"]["Test19_official_valid_never_loaded"] is True


def test_20_official_test_never_loaded():
    for name in _all_artifacts():
        payload = _json(name)
        assert payload.get("official_test_loaded") is False, name
    assert _json("integrity_tests.json")["tests"]["Test20_official_test_never_loaded"] is True


# --- supporting tests -------------------------------------------------------


def test_21_phase_firewall_blocks_early_access():
    data, _subsets = _data_and_subsets()
    fmf._ALLOW_SELECT = False
    fmf._ALLOW_PROBE = False
    with pytest.raises(fmf.PhaseFirewallError):
        fmf._select_graphs(data)
    with pytest.raises(fmf.PhaseFirewallError):
        fmf._probe_graphs(data)


def test_22_eval_mode_is_deterministic():
    data, subsets = _data_and_subsets()
    loader = zpp._make_loader(fmf.train_graphs(data, subsets)[:256], 128, False, 0)
    net = fmf.build_model("baseline", 0)
    net.train()
    maes = [fmf.shead._evaluate_mae(net, loader, torch.device("cpu"))[0] for _ in range(2)]
    assert net.training is False
    assert maes[0] == maes[1]


def test_23_matching_is_deterministic_and_resolvable():
    frame = fmf._read_inventory_frame()
    overlap = _json("train_fit_overlap.json")
    anchors = [float(a) for a in overlap["anchors"]]
    threshold = float(overlap["common_step_threshold"])
    lock = _json("fit_anchor_lock.json")
    locked_by_anchor = {int(p["anchor_id"]): p for p in lock["matched_snapshot_pairs"]}
    for model in fmf.MODELS:
        sub = frame[(frame["model"] == model) & (frame["optimizer_step"] >= threshold)]
        sub = sub.sort_values("optimizer_step")
        steps = [int(s) for s in sub["optimizer_step"].tolist()]
        fits = [float(f) for f in sub["train_mae"].tolist()]
        chosen = fmf._assign_to_anchors(steps, fits, anchors)
        for anchor_id, idx in enumerate(chosen):
            assert int(steps[idx]) == int(locked_by_anchor[anchor_id][f"{model}_step"])
            assert float(fits[idx]) == pytest.approx(
                float(locked_by_anchor[anchor_id][f"{model}_train_mae"])
            )


def test_24_soup_is_context_only_not_in_frontier():
    endpoint = _json("endpoint_context.json")
    assert "contextual" in endpoint["note"]
    lock = _json("fit_anchor_lock.json")
    for pair in lock["matched_snapshot_pairs"]:
        assert "soup" not in str(pair["baseline_snapshot_path"]).lower()
        assert "soup" not in str(pair["sbci_snapshot_path"]).lower()
    stage1 = _json("stage1_800_decision.json")
    assert "soup" not in json.dumps(stage1).lower()


def test_25_final_case_is_crossing_and_no_family():
    final = _json("final_decision.json")
    assert final["final_case"] == "FIT_DEPENDENT_FRONTIER_CROSSING"
    assert final["stage2_2000_decision"] is None
    family = _json("next_architecture_family.json")
    assert family["authorized_for_design"] is False
    assert family["family"] is None
    assert family["full_training_authorized"] is False
    assert family["sbci_branch_closed_for_architecture_inference"] is True


def test_26_material_crossing_is_recorded():
    stage1 = _json("stage1_800_decision.json")
    assert stage1["decision"] == "MATERIAL_FRONTIER_CROSSING"
    assert stage1["material_frontier_crossing"] is True
    assert stage1["authorize_2000"] is False
    per_anchor = fmf._read_csv(fmf.RESULTS_DIR / "stage1_800_per_anchor.csv")
    d = np.asarray([float(p["D_k"]) for p in per_anchor])
    assert int(np.sum(d >= fmf.CROSSING_DELTA)) >= fmf.CROSSING_MIN_ANCHORS
    assert int(np.sum(d <= -fmf.CROSSING_DELTA)) >= fmf.CROSSING_MIN_ANCHORS


def test_27_bootstrap_resamples_molecules_with_fixed_anchors():
    boot = _json("stage1_800_bootstrap.json")
    assert boot["bootstrap_unit"].startswith("held-out molecule")
    assert boot["equal_weight_anchors"] is True
    assert boot["n_molecules"] == 800
    assert boot["bootstrap_B"] == fmf.BOOTSTRAP_B == 2000
    dbar = np.load(fmf.RESULTS_DIR / "stage1_800_per_molecule.npy")
    assert dbar.shape == (800,)
    assert float(dbar.mean()) == pytest.approx(boot["paired_bootstrap_over_molecules"]["mean"])
    # independent re-bootstrap reproducibility
    reb = fmf.se.paired_bootstrap(dbar, B=fmf.BOOTSTRAP_B, seed=fmf.BOOTSTRAP_SEED)
    assert reb["ci95_lower"] == pytest.approx(boot["paired_bootstrap_over_molecules"]["ci95_lower"])
    assert reb["ci95_upper"] == pytest.approx(boot["paired_bootstrap_over_molecules"]["ci95_upper"])


def test_28_answers_q1_q20_complete():
    answers = _json("answers_q1_q20.json")
    for i in range(1, 21):
        prefix = f"Q{i}_"
        assert any(key.startswith(prefix) for key in answers), f"missing Q{i}"
    assert answers["Q14_800_material_frontier_crossing"] is True
    assert answers["Q15_800_verdict"] == "MATERIAL_FRONTIER_CROSSING"
    assert answers["Q16_authorize_2000"] is False
    assert answers["Q17_2000_D_mean"] is None
    assert answers["Q19_matched_fit_supports"] == "neither"
    assert answers["Q20_final_architecture_authorization"]["authorized_for_design"] is False


def test_29_no_stage2_placeholder():
    for name in (
        "stage2_2000_per_anchor.csv",
        "stage2_2000_per_molecule.npy",
        "stage2_2000_bootstrap.json",
        "stage2_2000_decision.json",
    ):
        assert not (fmf.RESULTS_DIR / name).exists(), name


# --- helpers ----------------------------------------------------------------


def _json_baseline(name: str) -> dict:
    return json.loads((fmf.BASELINE_DIR / name).read_text(encoding="utf-8"))


def _all_artifacts() -> list[str]:
    return [
        "audit_protocol_lock.json",
        "protocol_compatibility.json",
        "snapshot_inventory.json",
        "train_fit_overlap.json",
        "fit_anchor_lock.json",
        "phaseF_fit_only_lock.json",
        "match_quality.json",
        "fit_inventory.summary.json",
        "stage1_800_bootstrap.json",
        "stage1_800_decision.json",
        "endpoint_context.json",
        "final_decision.json",
        "next_architecture_family.json",
        "answers_q1_q20.json",
        "integrity_tests.json",
    ]

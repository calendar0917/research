"""Tests for the SBCI fit-vs-generalization failure triage (zero-training).

These tests never train, never touch official valid/test, and never modify the
frozen artifacts.  They verify run identity, protocol comparability, frozen
state correctness, eval-mode/gradient discipline, and that the diagnostic
``F_min`` never participates in model selection.
"""

from __future__ import annotations

import functools
import json

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_sbci as sb
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import (
    zinc_sbci_fit_generalization_triage as tri,
)


@functools.lru_cache(maxsize=1)
def _data_and_subsets():
    data = tri.load_data()
    subsets = tri.build_subsets(data)
    return data, subsets


def _json(name: str) -> dict:
    return json.loads((tri.RESULTS_DIR / name).read_text(encoding="utf-8"))


# --- 1 / 2 -----------------------------------------------------------------


def test_01_baseline_run_is_n3600_i0_t0():
    inv = tri.baseline_run_inventory()
    assert inv["run_id"] == "N3600_I0T0"
    assert inv["N"] == 3600 and inv["I"] == 0 and inv["T"] == 0
    assert inv["is_n3600_i0_t0"] is True
    assert inv["architecture"] == "compact-v4-smallhead"
    assert inv["parameters"] == 82115


def test_02_sbci_run_is_n3600_i0_t0():
    inv = tri.sbci_run_inventory()
    assert inv["run_id"] == "N3600_I0T0"
    assert inv["N"] == 3600 and inv["I"] == 0 and inv["T"] == 0
    assert inv["is_n3600_i0_t0"] is True
    assert inv["parameters"] == sb._n_params(sb.build_sbci(0))
    assert inv["cap_respected"] is True


# --- 3 / 4 -----------------------------------------------------------------


def test_03_d3600_exact_same_indices():
    data, subsets = _data_and_subsets()
    comp = tri.triage_protocol_compatibility(data, subsets)
    assert comp["comparable"] is True
    assert comp["checks"]["same_d3600_indices"] is True
    lock = json.loads(
        (tri.BASELINE_DIR / "sample_efficiency_subset_lock.json").read_text(encoding="utf-8")
    )
    assert comp["data_fingerprints"]["train_subset_index_hash"] == lock["dataset_hash_3600"]
    assert lock["dataset_hash_3600"] == (
        "3be9f7e1a4fb8cf4ed5eedfbd725b1ac9d618e6bc6d4873fe254ddc1017dfea9"
    )
    assert len(subsets["sorted_3600"]) == 3600


def test_04_select_800_and_probe_2000_exact_same_indices():
    data, subsets = _data_and_subsets()
    comp = tri.triage_protocol_compatibility(data, subsets)
    ref = json.loads((tri.BASELINE_DIR / "split_inventory.json").read_text(encoding="utf-8"))
    assert comp["checks"]["same_select_800_indices"] is True
    assert comp["checks"]["same_probe_2000_indices"] is True
    assert comp["data_fingerprints"]["select_index_hash"] == (
        ref["index_sha256"]["checkpoint_selection"]
    )
    assert comp["data_fingerprints"]["probe_index_hash"] == ref["index_sha256"]["internal_probe"]
    assert comp["data_fingerprints"]["n_select_800"] == 800
    assert comp["data_fingerprints"]["n_probe_2000"] == 2000


# --- 5 ---------------------------------------------------------------------


def test_05_optimizer_protocol_consistent():
    base = tri.baseline_run_inventory()
    candidate = tri.sbci_run_inventory()
    for key in (
        "optimizer",
        "learning_rate",
        "weight_decay",
        "batch_size",
        "loss",
        "scheduler",
        "gradient_clip_norm",
        "max_optimizer_steps",
        "eval_interval",
        "patience_evaluations",
    ):
        assert base[key] == candidate[key], key
    assert base["optimizer"] == "Adam"
    assert base["max_optimizer_steps"] == 13680
    assert base["eval_interval"] == 57
    assert base["patience_evaluations"] == 40
    assert base["soup_K"] == candidate["soup_K"] == 5


# --- 6 ---------------------------------------------------------------------


def test_06_shared_initialization_fingerprint_matches():
    base_net = tri.build_model("baseline", 0)
    sbci_net = tri.build_model("sbci", 0)
    bs = base_net.state_dict()
    ss = sbci_net.state_dict()
    shared = [k for k in bs if k in ss and tuple(bs[k].shape) == tuple(ss[k].shape)]
    assert len(shared) == 20
    for key in shared:
        assert torch.equal(bs[key], ss[key]), key
    inv = _json("triage_protocol_compatibility.json")
    assert inv["shared_initialization"]["max_abs_diff"] == 0.0
    assert inv["shared_initialization"]["sbci_initialization_lock_hash_match"] is True
    assert tri.baseline_run_inventory()["init_state_sha256_matches"] is True
    assert tri.sbci_run_inventory()["init_state_sha256_matches"] is True


# --- 7 ---------------------------------------------------------------------


def test_07_raw_states_correct():
    for model in tri.MODELS:
        member = tri.raw_member(model)
        assert member["step"] == tri._run_inventory(model)["raw_member_step"]
        net, loaded = tri.load_raw_model(model)
        assert tri._state_hash(tri._load_state(loaded["snapshot"])) == tri._state_hash(
            net.state_dict()
        )
        assert net.training is False
    raw = _json("train_eval_raw.json")
    assert raw["per_model"]["baseline"]["step"] == tri.raw_member("baseline")["step"]
    assert raw["per_model"]["sbci"]["step"] == tri.raw_member("sbci")["step"]


# --- 8 ---------------------------------------------------------------------


def test_08_soup_states_correct():
    for model in tri.MODELS:
        record = tri._soup_record(model)
        net, soup = tri.load_soup_model(model)
        assert soup["soup_state_file_sha256"] == record["soup_state_file_sha256"]
        assert soup["soup_state_tensor_sha256"] == record["soup_state_tensor_sha256"]
        assert net.training is False
        assert len(soup["members"]) == 5
    payload = _json("train_eval_soup.json")
    assert payload["per_model"]["baseline"]["member_steps"] == [3762, 5985, 4788, 6042, 5073]
    assert payload["per_model"]["sbci"]["member_steps"] == [3534, 3420, 5016, 5130, 4788]


# --- 9 ---------------------------------------------------------------------


def test_09_train_eval_uses_model_eval():
    data, subsets = _data_and_subsets()
    loader = zpp._make_loader(tri.train_data(data, subsets)[:256], 128, False, 0)
    for model in tri.MODELS:
        net, _ = tri.load_raw_model(model)
        net.train()
        assert net.training is True
        mae, targets, preds = tri._evaluate(net, loader)
        assert net.training is False, f"{model}: _evaluate must call model.eval()"
        assert np.isfinite(mae)
        assert targets.shape == preds.shape
    # dropout-free deterministic eval
    net, _ = tri.load_raw_model("baseline")
    m1, _, p1 = tri._evaluate(net, loader)
    m2, _, p2 = tri._evaluate(net, loader)
    assert m1 == m2
    assert np.array_equal(p1, p2)


# --- 10 --------------------------------------------------------------------


def test_10_best_train_fit_not_used_for_model_selection():
    best = _json("best_train_fit.json")
    assert best["diagnostic_only"] is True
    assert best["not_used_for_model_selection"] is True
    raw = _json("train_eval_raw.json")
    for model in tri.MODELS:
        raw_step = raw["per_model"][model]["step"]
        fmin_step = best["per_model"][model]["f_min_step"]
        # the deployed RAW checkpoint is the select-800 argmin, not the F_min step
        assert raw_step == tri.raw_member(model)["step"]
        assert raw_step != fmin_step
    # the frozen SBCI decision is untouched
    assert _json("final_decision.json")["sbci_decision_unchanged"] is True


# --- 11 --------------------------------------------------------------------


def test_11_no_gradient_updates():
    data, subsets = _data_and_subsets()
    loader = zpp._make_loader(tri.train_data(data, subsets)[:256], 128, False, 0)
    net, _ = tri.load_raw_model("baseline")
    before = [p.detach().clone() for p in net.parameters()]
    tri._evaluate(net, loader)
    for p in net.parameters():
        assert p.grad is None
    for old, new in zip(before, net.parameters()):
        assert torch.equal(old, new.detach())
    # optimizer-free module: no optimizer object is created anywhere in the triage
    import inspect

    source = inspect.getsource(tri)
    assert "optimizer.step" not in source
    assert ".backward(" not in source


# --- 12 / 13 ---------------------------------------------------------------


def test_12_official_valid_never_loaded():
    for name in (
        "audit_protocol_lock.json",
        "baseline_run_inventory.json",
        "sbci_run_inventory.json",
        "triage_protocol_compatibility.json",
        "train_eval_raw.json",
        "train_eval_soup.json",
        "best_train_fit.json",
        "late_trajectory_fit.json",
        "convergence_diagnostic.json",
        "generalization_gap.json",
        "triage_decision.json",
        "next_architecture_family.json",
        "answers_q1_q16.json",
        "final_decision.json",
    ):
        payload = _json(name)
        assert payload.get("official_valid_used") is False, name


def test_13_official_test_never_loaded():
    for name in (
        "audit_protocol_lock.json",
        "baseline_run_inventory.json",
        "sbci_run_inventory.json",
        "triage_protocol_compatibility.json",
        "train_eval_raw.json",
        "train_eval_soup.json",
        "best_train_fit.json",
        "late_trajectory_fit.json",
        "convergence_diagnostic.json",
        "generalization_gap.json",
        "triage_decision.json",
        "next_architecture_family.json",
        "answers_q1_q16.json",
        "final_decision.json",
    ):
        payload = _json(name)
        assert payload.get("official_test_loaded") is False, name


# --- supporting integrity tests -------------------------------------------


def test_14_snapshot_fit_recompute_matches_stored_curve():
    summary = _json("snapshot_train_fit.summary.json")
    assert summary["recompute_matches_stored"] is True
    assert summary["max_abs_diff_recomputed_vs_stored"] < 1e-5
    assert summary["row_counts"] == {"baseline": 106, "sbci": 102}
    assert summary["n_rows"] == 208


def test_15_decision_is_inconclusive_with_no_family():
    decision = _json("triage_decision.json")
    assert decision["decision_case"] == "FIT_GENERALIZATION_TRIAGE_INCONCLUSIVE"
    assert decision["criteria"]["approximation_case_A"] is False
    assert decision["criteria"]["adequate_fit_case_B"] is False
    assert decision["delta_F_soup"] > 0.003
    assert abs(decision["delta_F_min"]) > 0.003
    family = _json("next_architecture_family.json")
    assert family["authorized_for_design"] is False
    assert family["family"] is None
    assert family["full_training_authorized"] is False


def test_16_probe_degradation_is_the_known_input():
    probe = tri.probe_degradation()
    assert probe["n_probe"] == 2000
    assert probe["baseline_soup_probe_mae"] == pytest.approx(0.176633, abs=1e-5)
    assert probe["sbci_soup_probe_mae"] == pytest.approx(0.182924, abs=1e-5)
    assert probe["probe_degradation_soup"] == pytest.approx(0.006291, abs=1e-5)


def test_17_convergence_warning_needs_significant_slope():
    conv = _json("convergence_diagnostic.json")
    sbci = conv["per_model"]["sbci"]
    # the literal threshold is recorded honestly but is not significant
    assert conv["literal_threshold_fired"] is True
    assert sbci["train_slope_per_eval_interval_significant_negative"] is False
    assert conv["convergence_warning"] is False
    lo, hi = sbci["train_slope_per_eval_interval_bootstrap_ci95"]
    assert lo < 0.0 < hi
    # SBCI best-so-far had plateaued relative to the baseline
    assert sbci["best_so_far_improve_last10"] < conv["per_model"]["baseline"][
        "best_so_far_improve_last10"
    ]


def test_18_answers_q1_q16_complete():
    answers = _json("answers_q1_q16.json")
    for i in range(1, 17):
        prefix = f"Q{i}_"
        assert any(key.startswith(prefix) for key in answers), f"missing Q{i}"
    assert answers["Q15_failure_type"] == "FIT_GENERALIZATION_TRIAGE_INCONCLUSIVE"
    assert answers["Q16_next_architecture_family_authorized"] is False
    assert answers["Q16_next_architecture_family"] is None

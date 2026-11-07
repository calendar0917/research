"""Tests for the frozen training protocol, gradient reachability and resume."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import cs2_common as C
import cs2_model as M
import cs2_train as T
import cs2_compare as CMP
import bs_model as BSM

pytestmark = pytest.mark.slow


def test_frozen_recipe_values():
    r = C.RECIPE
    assert r["optimizer"] == "AdamW"
    assert r["learning_rate"] == 3.0e-4
    assert r["warmup_start_lr"] == 3.0e-5
    assert r["end_lr"] == 1.0e-5
    assert r["warmup_epochs"] == 1.0
    assert r["weight_decay"] == 1.0e-3
    assert tuple(r["adam_betas"]) == (0.9, 0.999)
    assert r["adam_eps"] == 1.0e-8
    assert r["batch_size"] == 128
    assert r["epochs"] == 40
    assert r["gradient_clip"] == 5.0
    assert r["dropout"] == 0.10
    assert r["precision"] == "fp32"
    assert r["early_stopping"] is False
    assert "masked" in r["loss"]
    assert (C.N_TRAIN, C.N_VALID, C.N_TEST) == (350_343, 43_793, 43_793)
    assert (C.STEPS_PER_EPOCH, C.TOTAL_STEPS_40) == (2_738, 109_520)
    assert C.EVAL_CADENCE["every"] == 1
    assert C.calc_eval_epochs() == list(range(1, 41))


def test_lr_schedule_is_inherited():
    from sg_common import scheduled_lr as inherited

    for step in (0, 1, 137, 2736, 2737, 2738, 5476, 30000, 109519, 109520):
        assert C.scheduled_lr(step, phase=3, kind="all") == inherited(step, phase=3, kind="all")
    assert C.scheduled_lr(2737, phase=3, kind="all") == 3.0e-4
    assert C.scheduled_lr(109520, phase=3, kind="all") == 1.0e-5


def test_optimizer_matches_recipe(models):
    for key in C.MODEL_ORDER:
        model = models[key][0]
        opt, audit = BSM.make_optimizer(model)
        assert audit["optimizer"] == "AdamW"
        assert audit["n_parameters"] == C.EXPECTED_PARAMETERS[key]
        group = opt.param_groups[0]
        assert group["lr"] == 3.0e-5
        assert tuple(group["betas"]) == (0.9, 0.999)
        assert group["eps"] == 1.0e-8
        assert group["weight_decay"] == 1.0e-3


def test_frozen_order_plan(task):
    plan = C.SD.make_plan40(task, seed=C.ORDER_SEED)
    assert len(plan) == 40
    assert plan[0]["global_plan_sha256"] == C.BASE_GLOBAL_PLAN_SHA256
    assert [len(p["batches"]) for p in plan] == [C.STEPS_PER_EPOCH] * 40
    assert sum(len(b) for b in plan[0]["batches"]) == C.N_TRAIN
    assert all(len(b) == C.BATCH_SIZE for b in plan[0]["batches"][:-1])
    all_rows = np.concatenate([p["rows"] for p in plan])
    assert all_rows.size == C.N_TRAIN * 40
    assert not np.isin(all_rows, task.valid_rows).any()
    assert not np.isin(all_rows, task.test_rows).any()
    assert T.plan_meta(plan, 40, C.ORDER_SEED)["prefix_sha256"] == C.BASE_CONSUMED40_PREFIX_SHA256


def test_c2_gradient_reaches_new_tensors(task):
    plan = C.SD.make_plan40(task, seed=C.ORDER_SEED)
    result = __import__("cs2_audit").gradient_reachability(
        task, plan, torch.device("cpu"), steps=2
    )["C2"]
    assert result["all_checks_pass"], result["checks"]
    assert result["checks"]["E_c_grad_zero_step1_expected"]
    assert result["checks"]["E_e_grad_nonzero_by_step2"]
    assert result["checks"]["V_grad_nonzero_step1"]
    assert float(result["steps"][0]["gate_delta_rms"]) > 0.0


def test_ffn_gradient_reaches_new_tensors(task):
    plan = C.SD.make_plan40(task, seed=C.ORDER_SEED)
    result = __import__("cs2_audit").gradient_reachability(
        task, plan, torch.device("cpu"), steps=2
    )["F"]
    assert result["all_checks_pass"], result["checks"]
    assert result["checks"]["fc2_grad_nonzero_step1"]
    assert result["checks"]["fc1_grad_zero_step1_expected"]
    assert result["checks"]["fc1_grad_nonzero_by_step2"]


def test_checkpoint_roundtrip_restores_everything(task, tmp_path):
    plan = C.SD.make_plan40(task, seed=C.ORDER_SEED)
    result = __import__("cs2_audit").checkpoint_roundtrip(task, plan, torch.device("cpu"), tmp_path)
    for key in C.MODEL_ORDER:
        r = result[key]
        assert r["state_bitwise_equal"], key
        assert r["optimizer_state_equal"], key
        assert r["rng_restored_exactly"], key
        assert r["next_batch_rows_equal"], key


def test_epoch0_dropout_rng_is_paired(models, batch):
    """One dropout call per forward with the same RNG stream: after a seeded
    forward the three models must leave the torch RNG in the same state."""
    states = {}
    with torch.no_grad():
        for key in ("W2", "C2", "F"):
            torch.manual_seed(12345)
            models[key][0].train()
            models[key][0](batch)
            states[key] = torch.get_rng_state().clone()
    assert torch.equal(states["W2"], states["C2"])
    assert torch.equal(states["W2"], states["F"])


def test_label_nan_mask():
    logits = torch.tensor([[0.3, -0.2, 1.1, -0.9]])
    labels = torch.tensor([[1.0, 0.0, float("nan"), float("nan")]])
    loss = C.MET.masked_bce_loss(logits, labels)
    manual = (np.log1p(np.exp(-0.3)) + np.log1p(np.exp(-0.2))) / 2.0
    assert abs(float(loss) - float(manual)) < 1e-6
    observed = torch.isfinite(labels)
    ref = torch.nn.functional.binary_cross_entropy_with_logits(logits[observed], labels[observed])
    assert abs(float(loss) - float(ref)) < 1e-6


def _summary(mean20=0.004, improved20=70, sparse_mean=0.006):
    high = {
        "n": 107,
        "mean_delta": mean20,
        "median_delta": mean20,
        "improved": improved20,
        "regressed": 107 - improved20,
        "tied": 0,
    }
    return {
        "C2_minus_W2": {
            "strata": {
                "support_ge20": high,
                "support_5_19": {"n": 18, "mean_delta": sparse_mean, "median_delta": 0.0,
                                  "improved": 10, "regressed": 8, "tied": 0},
                "support_lt5": {"n": 2, "mean_delta": -0.01, "median_delta": -0.01,
                                 "improved": 0, "regressed": 2, "tied": 0},
            }
        }
    }


def test_decision_labels():
    curve = {"ap40": 0.27, "best_diagnostic_epoch": 40, "best_diagnostic_ap": 0.27}
    assert CMP.decide(0.006, 0.004, 0.004, _summary(), c2_ap_curve=curve)["label"] == "PROMISING_CONDITIONAL_S2"
    assert CMP.decide(0.006, 0.004, 0.002, _summary(), c2_ap_curve=curve)["label"] == "CAPACITY_ONLY_OR_MIXED"
    assert CMP.decide(0.006, 0.002, 0.001, _summary(), c2_ap_curve=curve)["label"] == "CAPACITY_ONLY_OR_MIXED"
    assert CMP.decide(0.003, 0.001, 0.002, _summary(mean20=0.0015, improved20=30), c2_ap_curve=curve)["label"] == "WEAK_OR_MIXED_C2"
    assert CMP.decide(0.001, 0.0005, 0.0005, _summary(mean20=0.0005, improved20=40), c2_ap_curve=curve)["label"] == "NO_CLEAR_C2_GAIN"
    assert CMP.decide(-0.003, -0.002, -0.001, _summary(mean20=-0.002, improved20=20), c2_ap_curve=curve)["label"] == "C2_REGRESSION"
    early = {"ap40": 0.27, "best_diagnostic_epoch": 20, "best_diagnostic_ap": 0.280}
    early_decision = CMP.decide(0.006, 0.004, 0.0035, _summary(mean20=0.004, improved20=70), c2_ap_curve=early)
    assert early_decision["label"] == "CAPACITY_ONLY_OR_MIXED"
    assert early_decision["mixed_flags"]["early_peak_then_regress_by_epoch40"]

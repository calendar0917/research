"""Tests for the frozen training protocol, the shared training step and resume."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import bs_common as C
import bs_model as M
import bs_train as T


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


def test_lr_schedule_is_the_inherited_phase3_rule():
    from sg_common import scheduled_lr as inherited

    for step in (0, 1, 137, 2736, 2737, 2738, 5476, 30000, 109519, 109520):
        assert C.scheduled_lr(step, phase=3, kind="all") == inherited(step, phase=3, kind="all")
    assert C.scheduled_lr(2737, phase=3, kind="all") == 3.0e-4
    assert C.scheduled_lr(109520, phase=3, kind="all") == 1.0e-5
    first = C.scheduled_lr(0, phase=3, kind="all")
    assert 3.0e-5 < first <= 3.0e-5 + (3.0e-4 - 3.0e-5) / 2738 + 1e-12
    sample = [C.scheduled_lr(s, phase=3, kind="all") for s in range(2738, 109521, 1000)]
    assert all(a >= b for a, b in zip(sample, sample[1:]))
    assert M.set_lrs.__module__ == "bs_model"


def test_optimizer_matches_recipe(spec, norm):
    model, _ = M.build_model(2, spec, norm, C.N_TASKS, seed=0, device="cpu")
    opt, audit = M.make_optimizer(model)
    assert audit["optimizer"] == "AdamW"
    assert audit["n_param_groups"] == 1
    assert audit["n_parameters"] == C.EXPECTED_PARAMETERS[2]
    group = opt.param_groups[0]
    assert group["lr"] == 3.0e-5
    assert tuple(group["betas"]) == (0.9, 0.999)
    assert group["eps"] == 1.0e-8
    assert group["weight_decay"] == 1.0e-3
    lr = M.set_lrs(opt, step=2737)
    assert lr == 3.0e-4


@pytest.mark.slow
def test_argument_batches_are_deterministic_and_masked(batch, task):
    rows = [int(x) for x in task.train_rows[:8]]
    again = C.PC.collate(task.cache, rows)
    for key in batch:
        if torch.is_tensor(batch[key]):
            assert torch.equal(batch[key], again[key]), key
    y = np.asarray(task.labels[np.asarray(rows)])
    assert np.isfinite(y).any()


@pytest.mark.slow
def test_frozen_order_plan(task):
    plan = C.SD.make_plan40(task, seed=C.SEED)
    assert len(plan) == 40
    assert plan[0]["global_plan_sha256"] == C.A0["global_plan_sha256"]
    assert [len(p["batches"]) for p in plan] == [C.STEPS_PER_EPOCH] * 40
    assert sum(len(b) for b in plan[0]["batches"]) == C.N_TRAIN
    assert all(len(b) == C.BATCH_SIZE for b in plan[0]["batches"][:-1])
    # the plan never contains a Validation or Test row
    all_rows = np.concatenate([p["rows"] for p in plan])
    assert all_rows.size == C.N_TRAIN * 40
    assert not np.isin(all_rows, task.valid_rows).any()
    assert not np.isin(all_rows, task.test_rows).any()


@pytest.mark.slow
def test_checkpoint_roundtrip_restores_state(spec, norm, batch, task, tmp_path):
    rows = [int(x) for x in task.train_rows[:8]]
    y = torch.as_tensor(np.asarray(task.labels[np.asarray(rows)]), dtype=torch.float32)
    model, info = M.build_model(1, spec, norm, C.N_TASKS, seed=0, device="cpu")
    opt, _ = M.make_optimizer(model)
    plan = {"global_plan_sha256": C.A0["global_plan_sha256"], "prefix_sha256": "x",
            "consumed_epochs": 1, "order_seed": 0, "batch_size": 128, "epochs": 40}
    T.train_step(model, opt, batch, y, global_step=0, clip=5.0)
    payload = T.checkpoint_payload(
        model_key="W2", epoch=1, global_step=1, model=model, optimizer=opt,
        rng_state=C.capture_rng_state(), plan=plan, model_audit=info,
        valid_ap=0.0, valid_report=None, kind="test",
    )
    path = tmp_path / "ckpt.pt"
    T._atomic_save(payload, path)
    saved = torch.load(path, map_location="cpu", weights_only=False)

    fresh, _ = M.build_model(1, spec, norm, C.N_TASKS, seed=0, device="cpu")
    fresh_opt, _ = M.make_optimizer(fresh)
    T.restore(saved, fresh, fresh_opt, model_key="W2")
    for key in model.state_dict():
        assert torch.equal(model.state_dict()[key], fresh.state_dict()[key]), key
    steps = {float(v["step"]) for v in fresh_opt.state_dict()["state"].values()}
    assert steps == {1.0}
    assert fresh_opt.param_groups[0]["lr"] == opt.param_groups[0]["lr"]


def test_train_step_sets_lr_and_clips(spec, norm, batch, task):
    rows = [int(x) for x in task.train_rows[:8]]
    y = torch.as_tensor(np.asarray(task.labels[np.asarray(rows)]), dtype=torch.float32)
    model, _ = M.build_model(2, spec, norm, C.N_TASKS, seed=0, device="cpu")
    opt, _ = M.make_optimizer(model)
    out = T.train_step(model, opt, batch, y, global_step=2737, clip=1e-6)
    assert out["lr"] == 3.0e-4
    assert out["clipped"] is True
    assert math.isfinite(out["loss"])
    total = math.sqrt(sum(float(p.grad.pow(2).sum()) for p in model.parameters() if p.grad is not None))
    assert total <= 1e-6 * 1.01 + 1e-9


def test_decision_rules_exhaustive():
    import bs_compare as B

    strata_hit = {"support_ge20": {"mean_delta": 0.010, "improved": 90}}
    strata_miss = {"support_ge20": {"mean_delta": 0.001, "improved": 50}}
    assert B.label_for(0.010, strata_hit, n_support_ge20=107)["label"] == C.DECISION_RULES["promising"]["label"]
    assert B.label_for(0.010, strata_miss, n_support_ge20=107)["label"] == C.DECISION_RULES["weak"]["label"]
    assert B.label_for(0.003, strata_miss, n_support_ge20=107)["label"] == C.DECISION_RULES["weak"]["label"]
    assert B.label_for(0.0005, strata_miss, n_support_ge20=107)["label"] == C.DECISION_RULES["none"]["label"]
    assert B.label_for(-0.003, strata_miss, n_support_ge20=107)["label"] == C.DECISION_RULES["regression"]["label"]

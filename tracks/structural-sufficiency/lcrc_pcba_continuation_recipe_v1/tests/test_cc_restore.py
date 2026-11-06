"""CPU restore audit for the three frozen epoch-40 checkpoints."""

from __future__ import annotations

import pytest
import torch

import cc_common as C
import cc_data as D
import cc_restore as R


@pytest.fixture(scope="module")
def task():
    return D.load_and_gate()


@pytest.mark.parametrize("model_id", C.MODEL_ORDER)
def test_source_checkpoint_restores_model_and_adamw(task, model_id):
    spec = C.SOURCE[model_id]
    payload = R.load_checkpoint(spec["path"])
    payload["_path"] = C.track_rel(spec["path"])
    model, optimizer, audit = R.restore_model_and_optimizer(model_id, payload, task, device="cpu")
    assert audit["state_dict_bitwise_equal"] is True
    assert audit["optimizer_state_bitwise_equal"] is True
    assert audit["optimizer_step_values"] == [float(C.STEP40)]
    assert audit["optimizer_param_groups"] == 1
    assert int(sum(p.numel() for p in model.parameters())) == spec["parameters"]
    assert len(payload["state_dict"]) == spec["state_tensors"]
    assert len(list(model.parameters())) == spec["optimizer_params"]
    assert R.optimizer_step_values(optimizer) == [float(C.STEP40)]


def test_first_step_l_and_r_are_identical_at_the_same_lr(task):
    plan = D.make_plan_41_80(task, seed=C.SEED)
    rows = plan[0]["batches"][0]
    model_id = "A"
    spec = C.SOURCE[model_id]
    payload = R.load_checkpoint(spec["path"])
    payload["_path"] = C.track_rel(spec["path"])
    for recipe in ("L", "R"):
        model, optimizer, _ = R.restore_model_and_optimizer(model_id, payload, task, device="cpu")
        R.restore_rng_state(payload["rng_state"])
        R.set_lr(optimizer, C.scheduled_lr(recipe, C.STEP40))
        assert optimizer.param_groups[0]["lr"] == pytest.approx(C.LR_FLOOR)
    assert C.scheduled_lr("L", C.STEP40 + 1) != C.scheduled_lr("R", C.STEP40 + 1)

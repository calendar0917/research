"""Training-recipe invariants of the SP round (no training is performed)."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import sp_common as C
import sp_model as M
import sp_train as ST
import bs_train as BT
import bs_model as BSM


def test_train_step_is_the_inherited_one():
    assert ST.train_step is BT.train_step
    assert ST.plan_meta is BT.plan_meta
    assert ST._atomic_save is BT._atomic_save
    assert ST.scheduler_state is BT.scheduler_state


def test_recipe_is_the_frozen_w2_recipe():
    r = C.recipe_for_model("SP")
    assert r["optimizer"] == "AdamW"
    assert float(r["learning_rate"]) == 3e-4
    assert float(r["warmup_start_lr"]) == 3e-5
    assert float(r["end_lr"]) == 1e-5
    assert float(r["weight_decay"]) == 1e-3
    assert [float(x) for x in r["adam_betas"]] == [0.9, 0.999]
    assert float(r["adam_eps"]) == 1e-8
    assert float(r["dropout"]) == 0.10
    assert float(r["gradient_clip"]) == 5.0
    assert r["precision"] == "fp32"
    assert r["epochs"] == 40 and r["batch_size"] == 128
    assert r["early_stopping"] is False
    assert "masked BCE" in r["loss"]
    assert r["eval_epochs"] == list(range(1, 41))
    assert C.N_TRAIN == 350_343 and C.N_VALID == 43_793 and C.N_TEST == 43_793
    assert C.STEPS_PER_EPOCH == 2_738 and C.TOTAL_STEPS_40 == 109_520
    assert C.W2["global_plan_sha256"] == "06f437299fa451e449202a990d28f3b8d101e7aec709bcd72a878c8de864b6f6"


def test_lr_schedule_endpoints_and_monotonicity():
    # step 0 is the first warmup increment above warmup_start_lr
    assert 3e-5 <= C.lr_at(0) < 3.02e-5
    assert abs(C.lr_at(C.STEPS_PER_EPOCH) - 3e-4) < 1e-9
    assert abs(C.lr_at(C.TOTAL_STEPS_40 - 1) - 1e-5) < 1e-6
    lrs = [C.lr_at(s) for s in range(C.STEPS_PER_EPOCH, C.TOTAL_STEPS_40, 97)]
    assert all(lrs[i] >= lrs[i + 1] - 1e-12 for i in range(len(lrs) - 1))


def test_selection_rule_takes_earliest_on_exact_ties():
    assert C.better_candidate(0.3, None, 1, None) is True
    assert C.better_candidate(0.31, 0.3, 5, 4) is True
    assert C.better_candidate(0.3, 0.3, 5, 4) is False
    assert C.better_candidate(0.3, 0.3, 3, 4) is True
    assert C.better_candidate(0.29, 0.3, 9, 4) is False


def test_masked_bce_ignores_missing_labels():
    import pc_metrics as MET

    logits = torch.tensor([[0.0, 5.0], [10.0, -10.0]], dtype=torch.float32)
    y = torch.tensor([[0.0, float("nan")], [1.0, 0.0]], dtype=torch.float32)
    loss = float(MET.masked_bce_loss(logits, y))
    manual = (
        math.log(2.0)  # (0, 0) entry
        + math.log1p(math.exp(-10.0))  # (1, 1) -> label 0 on logit -10
        + math.log1p(math.exp(-10.0))  # (1, 0) -> label 1 on logit 10
    ) / 3.0
    assert abs(loss - manual) < 1e-6
    # a NaN label must not contribute: changing its logit leaves the loss alone
    logits2 = logits.clone()
    logits2[0, 1] = -1e6
    assert abs(float(MET.masked_bce_loss(logits2, y)) - loss) < 1e-7


def test_checkpoint_roundtrip_restores_state_optimizer_and_logits(spec, norm, tiny_batch):
    model, audit = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    opt, _ = BSM.make_optimizer(model)
    y = torch.zeros(2, C.N_TASKS)
    y[:, 0] = 1.0
    for step in range(2):
        BT.train_step(model, opt, tiny_batch, y, global_step=step, clip=5.0)
    payload = ST.checkpoint_payload(
        model_key="SP",
        epoch=1,
        global_step=2,
        model=model,
        optimizer=opt,
        rng_state=C.capture_rng_state(),
        plan={"global_plan_sha256": C.W2["global_plan_sha256"], "consumed_epochs": 1, "prefix_sha256": "x"},
        model_audit=audit,
        valid_ap=0.0,
        valid_report=None,
        kind="test_roundtrip",
    )
    restored, _ = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    restored_opt, _ = BSM.make_optimizer(restored)
    info = ST.restore(payload, restored, restored_opt, model_key="SP")
    assert info["resumed_global_step"] == 2
    model.eval()
    restored.eval()
    with torch.no_grad():
        assert torch.equal(model(tiny_batch), restored(tiny_batch))
    assert M.state_sha256({k: v.cpu() for k, v in model.state_dict().items()}) == M.state_sha256(
        {k: v.cpu() for k, v in restored.state_dict().items()}
    )


def test_restore_rejects_a_foreign_round(spec, norm, tiny_batch):
    model, audit = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    opt, _ = BSM.make_optimizer(model)
    payload = ST.checkpoint_payload(
        model_key="SP",
        epoch=1,
        global_step=0,
        model=model,
        optimizer=opt,
        rng_state=C.capture_rng_state(),
        plan={"global_plan_sha256": C.W2["global_plan_sha256"]},
        model_audit=audit,
        valid_ap=None,
        valid_report=None,
        kind="test",
    )
    with pytest.raises(RuntimeError):
        ST.restore(payload, model, opt, model_key="S3")
    payload_bad = dict(payload)
    payload_bad["round_id"] = "not-this-round"
    with pytest.raises(RuntimeError):
        ST.restore(payload_bad, model, opt, model_key="SP")


def test_train_step_produces_finite_gradients_for_every_new_tensor(spec, norm, tiny_batch):
    model, _ = M.build_model(spec, norm, C.N_TASKS, seed=0, device="cpu")
    model.train()
    opt, _ = BSM.make_optimizer(model)
    y = torch.zeros(2, C.N_TASKS)
    y[:, 0] = 1.0
    BT.train_step(model, opt, tiny_batch, y, global_step=0, clip=5.0)
    first = M.branch_grad_norm(model)
    assert any(k.startswith("selective.out.") and v > 0 for k, v in first.items())
    BT.train_step(model, opt, tiny_batch, y, global_step=1, clip=5.0)
    BT.train_step(model, opt, tiny_batch, y, global_step=2, clip=5.0)
    third = M.branch_grad_norm(model)
    for key in ("selective.q.weight", "selective.q.bias", "selective.k.weight", "selective.k.bias"):
        assert key in third
        assert math.isfinite(third[key]) and third[key] > 0
    # the original S3 tensors keep receiving gradient too
    assert third["s3_branch.out.weight"] > 0


def test_training_batch_plan_matches_the_frozen_w2_order(task):
    plan = C.BC.SD.make_plan40(task, seed=0)
    assert plan[0]["global_plan_sha256"] == C.W2["global_plan_sha256"]
    meta = ST.plan_meta(plan, 40, 0)
    assert meta["prefix_sha256"] == C.W2["consumed40_prefix_sha256"]
    assert meta["consumed_epochs"] == 40
    assert len(plan[0]["batches"]) == C.STEPS_PER_EPOCH


def test_no_test_row_reaches_the_training_collate(task):
    test_rows = set(np.asarray(task.test_rows, np.int64).tolist())
    plan = C.BC.SD.make_plan40(task, seed=0)
    used = np.concatenate([np.asarray(b, np.int64) for b in plan[0]["batches"]])
    assert not (set(used.tolist()) & test_rows)
    assert np.intersect1d(np.asarray(task.s3["rows"], np.int64), np.asarray(task.test_rows, np.int64)).size == 0
    assert bool(task.s3["meta"]["test_rows_present"]) is False
    assert bool(task.s3["meta"]["labels_present"]) is False


# ---------------------------------------------------------------------------
# reporting layer (sp_compare) must be a pure function of run artifacts
# ---------------------------------------------------------------------------


def test_comparison_references_match_the_frozen_history():
    import sp_compare as SC

    assert SC.W2_REF["best_epoch"] == C.W2["best_epoch"] == 24
    assert SC.W2_REF["best_valid_ap"] == C.W2["best_valid_ap"]
    assert SC.W2_REF["parameters"] == C.BASE_PARAMETERS == 994_324
    assert SC.W2_REF["checkpoint_sha256"] == C.W2["best_checkpoint_sha256"]
    assert SC.S3_REF["best_epoch"] == C.S3["best_epoch"] == 29
    assert SC.S3_REF["best_valid_ap"] == C.S3["best_valid_ap"]
    assert SC.S3_REF["parameters"] == C.S3_PARAMETERS == 1_018_532
    assert SC.W2_REF["gpu_hours"] == C.W2["gpu_hours"]
    assert SC.S3_REF["gpu_hours"] == C.S3["gpu_hours"]
    for ref, path in ((SC.W2_REF, C.W2_CHECKPOINT_BEST), (SC.S3_REF, C.S3_CHECKPOINT_BEST)):
        assert ref["checkpoint"] == C.track_rel(path)
    if not all(p.is_file() for p in (C.W2_CHECKPOINT_BEST, C.S3_CHECKPOINT_BEST)):
        pytest.skip("historical checkpoints are not present in this checkout")


def test_decision_labels_follow_the_preregistered_table():
    import sp_compare as SC

    labels = C.DECISION_RULES["labels"]
    th = float(C.DECISION_RULES["valid_gain_threshold"])
    cases = [
        (th, th, labels["valid_and_test_gain"]),
        (th, th / 2, labels["valid_only_gain"]),
        (th / 2, th, labels["test_only_gain"]),
        (0.0, 0.0, labels["no_clear_gain"]),
        (0.0, None, "PENDING_TEST"),
    ]
    for valid_gain, test_gain, expected in cases:
        decision = SC._decide({"valid_gain_vs_s3": valid_gain, "test_gain_vs_s3": test_gain}, {})
        assert decision["label"] == expected
        assert decision["thresholds"]["valid_gain_min"] == th
        assert decision["further_rounds_authorised"] is False
        assert decision["test_not_used_for_selection"] is True


def test_support_strata_are_disjoint_and_cover_the_axis():
    import sp_compare as SC

    a = [0.5, 0.2, 0.9, None, 0.4, 0.3]
    b = [0.4, 0.25, 0.9, 0.1, 0.1, 0.3]
    support = [50, 3, 25, 10, 0, 19]
    ge20 = SC._task_stats(a, b, support, min_support=20)
    mid = SC._task_stats(a, b, support, min_support=5, max_support=20)
    low = SC._task_stats(a, b, support, min_support=0, max_support=5)
    assert (ge20["n_tasks"], mid["n_tasks"], low["n_tasks"]) == (2, 1, 2)
    assert ge20["n_tasks"] + mid["n_tasks"] + low["n_tasks"] == 5  # the None label is dropped

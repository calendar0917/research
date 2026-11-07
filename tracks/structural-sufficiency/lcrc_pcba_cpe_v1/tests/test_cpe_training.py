"""CPU tests for the CPE training path: gradients, checkpoint restore, order and
label isolation, and the fact that P and S are paired at step 0.

No training run is started here (no 40-epoch loop); only single steps on real
Plan-1 batches plus a small restore round-trip.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import cpe_common as C
import cpe_model as M
import cpe_train as T
import bs_model as BSM

pytestmark = pytest.mark.slow


def _batches(task, n=3, batch_index=0):
    plan = C.SD.make_plan40(task, seed=C.SEED)
    rows = [np.asarray(plan[0]["batches"][batch_index]) for _ in range(int(n))]
    return plan, rows


def _tensors(task, rows):
    import torch as _t

    return C.PC.collate(task.cache, rows.tolist()), _t.as_tensor(task.labels[rows])


def test_step_gradients_follow_the_frozen_pattern(task):
    plan, rows = _batches(task, n=3)
    for key in C.MODEL_ORDER:
        C.seed_everything(C.SEED)
        model, _audit = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
        optimizer, _ = BSM.make_optimizer(model)
        model.train()
        grads = []
        for step in range(3):
            batch, y = _tensors(task, rows[step])
            out = T.train_step(model, optimizer, batch, y, global_step=step, clip=float(C.RECIPE["gradient_clip"]))
            assert np.isfinite(out["loss"]) and np.isfinite(out["grad_norm"])
            named = dict(model.named_parameters())
            grads.append(
                {
                    name: None if named[name].grad is None else named[name].grad.detach().clone()
                    for name in (
                        "evidence.linear_A.weight",
                        "evidence.linear_B.weight",
                        "evidence.alpha_A",
                        "evidence.alpha_B",
                        "evidence.alpha_pair",
                    )
                }
            )
        # step 1: alphas get gradient, linears are exactly zero (alpha_* = 0)
        for name in ("evidence.alpha_A", "evidence.alpha_B", "evidence.alpha_pair"):
            assert grads[0][name] is not None
            assert torch.isfinite(grads[0][name]).all()
            assert torch.count_nonzero(grads[0][name]).item() > 0, (key, name)
        for name in ("evidence.linear_A.weight", "evidence.linear_B.weight"):
            assert grads[0][name] is not None
            assert torch.count_nonzero(grads[0][name]).item() == 0, (key, name)
        # step 2+: the detectors receive a non-zero finite gradient
        for name in ("evidence.linear_A.weight", "evidence.linear_B.weight"):
            for g in grads[1:]:
                assert torch.isfinite(g[name]).all()
                assert torch.count_nonzero(g[name]).item() > 0, (key, name)


def test_p_and_s_are_paired_at_step_zero(task):
    plan, rows = _batches(task, n=1)
    batch, y = _tensors(task, rows[0])
    losses = {}
    logits = {}
    for key in C.MODEL_ORDER:
        C.seed_everything(C.SEED)
        model, _ = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
        optimizer, _ = BSM.make_optimizer(model)
        model.eval()
        with torch.no_grad():
            logits[key] = model(batch).detach().clone()
        model.train()
        out = T.train_step(model, optimizer, batch, y, global_step=0, clip=float(C.RECIPE["gradient_clip"]))
        losses[key] = out["loss"]
    # identical forward *before* any update: same logits, same first loss
    assert torch.equal(logits["P"], logits["S"])
    assert losses["P"] == losses["S"]


def test_w2_and_arms_agree_on_a_validation_slice(task):
    rows = np.asarray(task.valid_rows[:256], dtype=np.int64)
    base, _ = BSM.build_model(C.BASE_MODEL_KEY, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    ref = C.PC.predict_logits(base, task.cache, rows, torch.device("cpu"), batch_size=256)
    for key in C.MODEL_ORDER:
        model, _ = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu", base_model=base)
        logits = C.PC.predict_logits(model, task.cache, rows, torch.device("cpu"), batch_size=256)
        assert np.array_equal(logits, ref), key
        assert np.abs(logits - ref).max() == 0.0


def test_checkpoint_roundtrip_restores_state_and_pair_mode(task, tmp_path):
    plan, rows = _batches(task, n=1)
    batch, y = _tensors(task, rows[0])
    for key in C.MODEL_ORDER:
        C.seed_everything(C.SEED)
        model, audit = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
        optimizer, _ = BSM.make_optimizer(model)
        model.train()
        out = T.train_step(model, optimizer, batch, y, global_step=0, clip=float(C.RECIPE["gradient_clip"]))
        assert np.isfinite(out["loss"])
        state_before = {k: v.detach().clone() for k, v in model.state_dict().items()}
        payload = T.checkpoint_payload(
            model_key=key,
            epoch=1,
            global_step=1,
            model=model,
            optimizer=optimizer,
            rng_state=C.capture_rng_state(),
            plan=T.plan_meta(plan, 1, C.SEED),
            model_audit=audit,
            valid_ap=None,
            valid_report=None,
            kind="test_roundtrip",
        )
        path = tmp_path / f"{key}.pt"
        sha = T._atomic_save(payload, path)
        assert len(sha) == 64
        C.seed_everything(999)
        fresh, _ = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
        fresh_opt, _ = BSM.make_optimizer(fresh)
        restored = T.restore(torch.load(path, map_location="cpu", weights_only=False), fresh, fresh_opt, model_key=key)
        assert restored["pair_mode_restored"] == C.PAIR_MODES[key]
        for name, value in state_before.items():
            assert torch.equal(value, fresh.state_dict()[name]), (key, name)
        # a checkpoint of the other arm must be refused
        if key == "P":
            with pytest.raises(RuntimeError):
                T.restore(payload, fresh, fresh_opt, model_key="S")


def test_masked_bce_matches_reference(task):
    rows = np.asarray(task.train_rows[:16], dtype=np.int64)
    y = np.asarray(task.labels[rows])
    logits = np.zeros_like(y, dtype=np.float32)
    loss = float(C.MET.masked_bce_loss(torch.as_tensor(logits), torch.as_tensor(y)).item())
    obs = np.isfinite(y)
    expected = float((-np.log(0.5) * obs).sum() / obs.sum())
    assert abs(loss - expected) <= 1e-6
    assert obs.sum() < obs.size  # PCBA has missing entries


def test_order_plan_isolation_and_test_labels(task):
    plan = C.SD.make_plan40(task, seed=C.SEED)
    summary = C.SD.plan_summary(plan, task)
    assert summary["isolation"]["passed"]
    assert plan[0]["global_plan_sha256"] == C.BASE_GLOBAL_PLAN_SHA256
    prefix = [p["row_order_sha256"] for p in plan[:40]]
    assert C.sha256_bytes("|".join(prefix).encode()) == C.BASE_CONSUMED40_PREFIX_SHA256
    yt = np.asarray(task.labels[task.test_rows])
    assert np.isnan(yt).all()
    train = set(np.asarray(task.train_rows).tolist())
    assert all(int(r) in train for r in np.asarray(plan[0]["rows"])[:1000])


def test_diagnostic_valid_rows_are_frozen_and_label_free(task):
    a = C.diagnostic_valid_rows(task)
    b = C.diagnostic_valid_rows(task)
    assert np.array_equal(a, b)
    assert a.size == 256
    assert np.array_equal(a, np.asarray(task.valid_rows)[:256])
    record = C.diagnostic_valid_rows_record(task)
    assert record["depends_on_labels"] is False
    assert record["depends_on_model_outputs"] is False
    assert len(record["row_ids"]) == 256


def test_ablation_helper_restores_alpha_pair(task):
    model, _ = M.build_model("P", task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    rows = np.asarray(task.train_rows[:8], dtype=np.int64)
    batch = C.PC.collate(task.cache, rows.tolist())
    before = model.evidence.alpha_pair.detach().clone()
    with torch.no_grad():
        model.evidence.alpha_pair.fill_(0.5)
    target = model.evidence.alpha_pair.detach().clone()
    out = M.logits_without_pair(model, batch)
    after = model.evidence.alpha_pair.detach().clone()
    assert torch.equal(target, after)
    with torch.no_grad():
        model.evidence.alpha_pair.copy_(before)
        full = model(batch)
    assert out.shape == full.shape
    assert model.evidence.alpha_pair.abs().max().item() == 0.0

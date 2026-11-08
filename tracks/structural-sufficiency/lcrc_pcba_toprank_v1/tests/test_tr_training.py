from __future__ import annotations

import numpy as np
import torch

import tr_common as C
import tr_loss as L
import tr_train as T


def test_w2_parameters_and_initial_state_are_historical(task):
    model, audit = C.build_w2(task, device="cpu")
    assert audit["total_parameters"] == C.MODEL_PARAMETERS
    assert int(C.parameter_count(model)) == C.MODEL_PARAMETERS
    assert audit["trunk_init_fingerprint"] == C.W2["trunk_init_fingerprint"]
    assert audit["full_trunk_init_fingerprint"] == C.W2["full_trunk_init_fingerprint"]
    assert C.state_sha256(C.state_cpu(model)) == C.W2["init_state_sha256"]
    assert audit["width_audit"]["all_match"] is True


def test_forward_calls_every_frozen_module_once(task, model, first_batch):
    m, _audit = model
    batch, _y = first_batch
    calls = C.BSM.count_module_calls(m, batch)
    assert calls == {
        "atom_encoder": 1,
        "bond_encoder": 1,
        "local": 1,
        "f": 1,
        "rho": 1,
        "readout": 1,
    }


def test_order_plan_is_frozen_and_isolated(task, plan):
    assert plan[0]["global_plan_sha256"] == C.GLOBAL_PLAN_SHA256
    assert len(plan) == C.TOTAL_EPOCHS
    assert all(int(p["n_batches"]) == C.STEPS_PER_EPOCH for p in plan)
    train = np.unique(task.train_rows)
    for p in plan:
        assert np.setdiff1d(p["rows"], train).size == 0
        assert np.intersect1d(p["rows"], task.valid_rows).size == 0
        assert np.intersect1d(p["rows"], task.test_rows).size == 0


def test_calibration_batches_are_fixed_train_rows(task, plan):
    batches = C.calibration_batches(task, plan)
    assert len(batches) == 16
    assert [b["batch_index"] for b in batches] == C.calibration_batch_indices(plan)
    assert batches[0]["combined_rows_sha256"] == batches[-1]["combined_rows_sha256"]
    train = np.unique(task.train_rows)
    for b in batches:
        assert np.setdiff1d(b["rows"], train).size == 0
        assert np.intersect1d(b["rows"], task.valid_rows).size == 0
        assert np.intersect1d(b["rows"], task.test_rows).size == 0
    assert C.calibration_batches(task, plan)[0]["rows_sha256"] == batches[0]["rows_sha256"]


def test_lr_schedule_endpoints_are_the_frozen_recipe():
    audit = C.lr_audit()
    assert audit["steps_per_epoch"] == 2738
    assert audit["total_steps_40_epochs"] == 109520
    assert audit["step0"] == 3.0098612125639153e-05
    assert audit["warmup_last"] == 0.0003
    assert audit["post_warmup_first"] == 0.0003
    assert audit["step40e"] == 1e-05


def test_train_step_lambda_zero_is_exact_bce(task, model, first_batch):
    m, _audit = model
    batch, y = first_batch
    state = C.state_cpu(m)
    rng = C.capture_rng_state()

    def branch(use_total: bool):
        m.load_state_dict(state, strict=True)
        C.restore_rng_state(rng)
        m.train()
        opt, _ = C.make_optimizer(m)
        logits = m(batch)
        if use_total:
            loss, parts = L.total_loss_parts(logits, y, 0.0)
            value = float(loss)
        else:
            loss = L.masked_bce_loss(logits, y)
            value = float(loss)
            parts = {"loss_bce": loss.detach()}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grads = {
            name: p.grad.detach().cpu().clone()
            for name, p in m.named_parameters()
            if p.grad is not None
        }
        return value, float(parts["loss_bce"]), grads, logits.detach().clone()

    bce_value, bce_part, bce_grads, bce_logits = branch(False)
    total_value, total_part, total_grads, total_logits = branch(True)
    assert torch.equal(bce_logits, total_logits)
    assert bce_value == total_value
    assert bce_part == total_part
    assert set(bce_grads) == set(total_grads)
    for key in bce_grads:
        assert torch.equal(bce_grads[key], total_grads[key])


def test_checkpoint_roundtrip_resumes_exactly(task, plan, model, tmp_path):
    m0, audit = model
    batch_rows = [
        np.asarray(plan[0]["batches"][i], dtype=np.int64) for i in range(4)
    ]
    batches = [C.BC.PC.collate(task.cache, rows.tolist()) for rows in batch_rows]
    labels = [torch.as_tensor(task.labels[rows], dtype=torch.float32) for rows in batch_rows]

    def _steps(model, optimizer, indices):
        out = []
        for i in indices:
            step = T.train_step(
                model,
                optimizer,
                batches[i],
                labels[i],
                global_step=i,
                clip=5.0,
                lambda_rank=1e-4,
            )
            out.append(step["loss_total"])
        return out

    model_a, _ = C.build_w2(task, device="cpu")
    opt_a, _ = C.make_optimizer(model_a)
    C.seed_everything(0)
    losses_a = _steps(model_a, opt_a, [0, 1, 2, 3])
    state_a = C.state_cpu(model_a)

    model_b, _ = C.build_w2(task, device="cpu")
    opt_b, _ = C.make_optimizer(model_b)
    C.seed_everything(0)
    _steps(model_b, opt_b, [0, 1])
    calibration = {
        "schema": C.SCHEMA_CALIBRATION,
        "round_id": C.ROUND_ID,
        "lambda_rank": 1e-4,
        "g_bce_median": 1.0,
        "g_rank_median": 1.0,
        "ratio_raw": 1e-4,
        "clipped": False,
        "valid_batches": 16,
        "n_batches": 16,
        "batch_indices": list(range(16)),
        "train_only": True,
        "rng_unchanged": True,
        "test_labels_read": False,
    }
    cal_path = tmp_path / "calibration.json"
    C.write_json(cal_path, calibration)
    payload = T.checkpoint_payload(
        epoch=1,
        global_step=2,
        model=model_b,
        optimizer=opt_b,
        rng_state=C.capture_rng_state(),
        plan=C.plan_meta(plan, 1, 0),
        model_audit=audit,
        valid_ap=None,
        valid_report=None,
        kind="unit_resume",
        calibration=calibration,
        calibration_path=cal_path,
    )
    path = tmp_path / "probe.pt"
    T.atomic_save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)

    model_c, _ = C.build_w2(task, device="cpu")
    opt_c, _ = C.make_optimizer(model_c)
    restored = T.restore(loaded, model_c, opt_c)
    assert restored["resumed_global_step"] == 2
    losses_c = _steps(model_c, opt_c, [2, 3])
    assert abs(losses_a[-1] - losses_c[-1]) < 1e-6
    state_c = C.state_cpu(model_c)
    assert state_a.keys() == state_c.keys()
    for key in state_a:
        assert torch.equal(state_a[key], state_c[key]), key

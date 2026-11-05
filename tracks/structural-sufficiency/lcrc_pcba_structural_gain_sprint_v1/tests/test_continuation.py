"""Phase-1 -> Phase-2 continuation round trip on CPU (no formal training)."""

from __future__ import annotations

import numpy as np
import torch

import sg_common as C
import sg_checkpoint as CK
import sg_model as M
import pc_metrics as MET
import pc_model as BM


def _arm(task, base_init, source, arm):
    return M.build_arm(
        arm,
        task.spec,
        task.norm,
        C.N_TASKS,
        seed=C.SEED,
        device="cpu",
        base_init_state=base_init,
        warm_start_state=source["state_dict"],
        zero_residual=True,
    )


def test_phase1_checkpoint_roundtrip_and_resume(task, source, reference, tmp_path):
    _, _, base_init = reference
    model, model_audit = _arm(task, base_init, source, "M")
    optimizer, _ = M.make_optimizer_warm(model, source["optimizer_state_dict"])
    batch = BM.collate(task.cache, [0, 1, 2, 3, 4, 5, 6, 7])
    y = torch.as_tensor(task.labels[np.arange(8)], dtype=torch.float32)
    for step in range(3):
        M.set_lrs(optimizer, phase=1, step=step)
        optimizer.zero_grad(set_to_none=True)
        loss = MET.masked_bce_loss(model(batch), y)
        loss.backward()
        optimizer.step()
    global_step = 12 * C.STEPS_PER_EPOCH
    sched = {
        "kind": "phase12_cosine_40epoch",
        "global_step": global_step,
        "total_steps_40_epochs": C.TOTAL_STEPS_40,
        "steps_per_epoch": C.STEPS_PER_EPOCH,
        "lr_now": [float(g["lr"]) for g in optimizer.param_groups],
        "audit": C.lr_audit(),
    }
    plan = {
        "global_plan_sha256": "a" * 64,
        "order_seed": 0,
        "batch_size": 128,
        "epochs": 40,
        "consumed_epochs": 12,
        "prefix_sha256": "b" * 64,
    }
    rng = C.capture_rng_state()
    payload = CK.checkpoint_payload(
        phase=1,
        arm="M",
        seed=0,
        epoch=12,
        global_step=global_step,
        model=model,
        optimizer=optimizer,
        scheduler_state=sched,
        rng_state=rng,
        plan=plan,
        model_audit=model_audit,
        source_audit=None,
        valid_ap=0.24,
        valid_report=None,
        kind="phase1_epoch12",
    )
    path = tmp_path / "checkpoint_epoch12.pt"
    sha = CK.save_checkpoint(payload, path)
    assert len(sha) == 64
    loaded = CK.load_checkpoint(path)
    assert loaded["epoch"] == 12 and loaded["global_step"] == global_step

    model2, _ = _arm(task, base_init, source, "M")
    optimizer2, _ = M.make_optimizer_warm(model2, source["optimizer_state_dict"])
    C.seed_everything(999)
    audit = M.load_continuation(model2, optimizer2, loaded, arm="M")
    assert audit["global_step"] == global_step
    assert audit["optimizer_state_identical"] is True
    eq = M.compare_states(model.state_dict(), model2.state_dict())
    assert eq["equal"], eq
    assert M.optimizer_state_fingerprint(M.optimizer_state_cpu(optimizer2)) == M.optimizer_state_fingerprint(
        M.optimizer_state_cpu(optimizer)
    )
    restored = M.restore_rng_state(loaded["rng_state"])
    assert restored["combined_sha256"] == loaded["rng_state_hashes"]["combined_sha256"]
    # The schedule is a pure function of global_step: LRs resume exactly.
    lrs = M.set_lrs(optimizer2, phase=2, step=global_step)
    expected = {
        "base": C.scheduled_lr(global_step, phase=2, kind="base"),
        "new": C.scheduled_lr(global_step, phase=2, kind="new"),
    }
    assert abs(lrs["base"] - expected["base"]) < 1e-18
    assert abs(lrs["new"] - expected["new"]) < 1e-18
    # Phase 2 must not silently reset the restored moments (source 24642 + 3 steps).
    state = M.optimizer_state_cpu(optimizer2)
    assert all(float(state["state"][pid]["step"].item()) == 24645.0 for pid in range(31))
    assert len(state["param_groups"]) == 2

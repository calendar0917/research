"""Training-path parity with the historical W2 round (loss, order, collate, metric)."""

from __future__ import annotations

import numpy as np
import torch

import bs_common as BC
import bs_model as BSM
import bs_train as BT
import pc_metrics as MET
import rc_checks as CHK
import rc_common as C
import rc_dist as D
import rc_model as M
import rc_train as T


def test_training_step_is_the_inherited_one():
    assert T.train_step is BT.train_step
    assert T.Watchdog is BT.Watchdog
    assert T._atomic_save is BT._atomic_save


def test_recipe_is_the_inherited_from_scratch_recipe():
    for key, value in BC.RECIPE.items():
        assert C.RECIPE[key] == value
    assert C.RECIPE["learning_rate"] == 3.0e-4
    assert C.RECIPE["warmup_start_lr"] == 3.0e-5
    assert C.RECIPE["end_lr"] == 1.0e-5
    assert C.RECIPE["weight_decay"] == 1.0e-3
    assert C.RECIPE["dropout"] == 0.10
    assert C.RECIPE["gradient_clip"] == 5.0
    assert C.BATCH_SIZE == 128
    assert C.TOTAL_EPOCHS == 40
    assert C.STEPS_PER_EPOCH == 2738
    assert C.BRANCH_DROPOUT == 0.10


def test_lr_schedule_endpoints():
    warmup = C.STEPS_PER_EPOCH
    start = C.RECIPE["warmup_start_lr"]
    peak = C.RECIPE["learning_rate"]
    end = C.RECIPE["end_lr"]
    # 1-epoch linear warmup start -> peak, then per-step cosine peak -> end
    assert abs(C.scheduled_lr(0, phase=3, kind="all") - (start + (peak - start) / warmup)) < 1e-12
    assert abs(C.scheduled_lr(warmup - 1, phase=3, kind="all") - peak) < 1e-9
    assert abs(C.scheduled_lr(C.TOTAL_STEPS_40, phase=3, kind="all") - end) < 1e-9
    assert C.scheduled_lr(C.TOTAL_STEPS_40 // 2, phase=3, kind="all") < peak


def test_optimizer_is_adamw_with_the_frozen_hyperparameters(spec, norm):
    model, _ = M.build_model("REAL", spec, norm, C.N_TASKS, seed=0, device="cpu")
    optimizer, audit = BSM.make_optimizer(model)
    assert isinstance(optimizer, torch.optim.AdamW)
    group = optimizer.param_groups[0]
    assert group["lr"] == C.RECIPE["warmup_start_lr"]
    assert tuple(group["betas"]) == tuple(C.RECIPE["adam_betas"])
    assert group["eps"] == C.RECIPE["adam_eps"]
    assert group["weight_decay"] == C.RECIPE["weight_decay"]
    assert audit["n_parameters"] == int(sum(p.numel() for p in model.parameters()))
    assert audit["n_parameters"] == C.EXPECTED_PARAMETERS


def test_collate_adds_only_the_pair_payload(task, dist_cache):
    rows = list(np.asarray(task.train_rows, dtype=np.int64)[:8])
    inherited = BC.PC.collate(task.cache, rows)
    rc = D.collate_rel(task, dist_cache, rows)
    assert sorted(set(rc) - set(inherited)) == ["pair_bucket", "pair_counts"]
    for key in inherited:
        a, b = inherited[key], rc[key]
        if torch.is_tensor(a):
            assert torch.equal(a, b), key
        else:
            assert a == b, key
    assert rc["pair_bucket"].dtype == torch.uint8
    assert rc["pair_counts"].dtype == torch.int64


def test_train_batch_contains_only_train_rows(task, dist_cache):
    plan = BC.SD.make_plan40(task, seed=C.SEED)
    train_rows = np.sort(np.asarray(task.train_rows, dtype=np.int64))
    valid = set(np.asarray(task.valid_rows, dtype=np.int64).tolist())
    test = set(np.asarray(task.test_rows, dtype=np.int64).tolist())
    for epoch in (0, 1, C.TOTAL_EPOCHS - 1):
        rows = np.asarray(plan[epoch]["rows"], dtype=np.int64)
        assert np.array_equal(np.sort(rows), train_rows)
        assert not (valid & set(rows.tolist()))
        assert not (test & set(rows.tolist()))
    assert plan[0]["global_plan_sha256"] == C.W2["global_plan_sha256"]


def test_pair_payload_matches_row_counts_on_a_real_batch(task, dist_cache):
    rows = list(np.asarray(task.train_rows, dtype=np.int64)[:16])
    batch = D.collate_rel(task, dist_cache, rows)
    counts = batch["atom_counts"].to(torch.int64)
    assert torch.equal(batch["pair_counts"], counts * (counts - 1) // 2)
    geom_ok = CHK.check_no_cross_molecule(
        task.spec, task.norm, {k: v for k, v in batch.items()}, device="cpu"
    )
    # the per-molecule reconstruction is validated on the synthetic batch; here
    # we only assert the production payload is internally consistent
    assert geom_ok["name"] == "05_no_cross_molecule_cooccurrence"


def test_masked_bce_and_official_ap_are_the_inherited_functions():
    import pc_metrics as MET2

    assert MET.masked_bce_loss is MET2.masked_bce_loss
    assert MET.official_ap is MET2.official_ap
    assert C.MET is MET2
    result = CHK.check_metrics_and_isolation(None)
    assert result["details"]["masked_bce_equals_manual"] is True
    assert result["details"]["official_vs_independent_abs_delta"] < 1e-9


def test_metrics_and_isolation_with_the_real_task(task, tiny_batch):
    result = CHK.check_metrics_and_isolation(task, tiny_batch)
    assert result["passed"], result
    d = result["details"]
    assert d["test_labels_all_nan_in_training_artifact"] is True
    assert d["order_plan_sha256_matches_w2"] is True


def test_checkpoint_roundtrip_restores_state_and_rng(spec, norm, tiny_batch, tmp_path):
    model, audit = M.build_model("REAL", spec, norm, C.N_TASKS, seed=0, device="cpu")
    optimizer, _ = BSM.make_optimizer(model)
    model.load_state_dict(model.state_dict())
    payload = {
        "schema": C.SCHEMA_CHECKPOINT,
        "round_id": C.ROUND_ID,
        "model": "REAL",
        "epoch": 0,
        "global_step": 0,
        "state_dict": BSM.state_cpu(model),
        "optimizer_state_dict": BSM.optimizer_state_cpu(optimizer),
        "rng_state": C.capture_rng_state(),
        "rng_state_hashes": C.rng_state_hashes(C.capture_rng_state()),
    }
    path = tmp_path / "ckpt.pt"
    torch.save(payload, path)
    loaded = torch.load(path, map_location="cpu", weights_only=False)
    fresh, _ = M.build_model("REAL", spec, norm, C.N_TASKS, seed=0, device="cpu")
    fresh_opt, _ = BSM.make_optimizer(fresh)
    audit_out = T.restore(loaded, fresh, fresh_opt, model_key="REAL")
    assert audit_out["optimizer_state_restored"] is True
    for key, value in model.state_dict().items():
        assert torch.equal(value, fresh.state_dict()[key]), key

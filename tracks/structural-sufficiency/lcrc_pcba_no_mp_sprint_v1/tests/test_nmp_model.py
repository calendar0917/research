"""CPU correctness tests for the no-edge-MP arms S, A and G."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import nmp_common as C
import nmp_data as D
import nmp_model as M
import pc_metrics as MET
import pc_model as PC


def _arm(task, reference, arm, seed=0, device="cpu"):
    _, _, base_init = reference
    return M.build_arm(
        arm,
        task.spec,
        task.norm,
        C.N_TASKS,
        seed=int(seed),
        device=device,
        base_init_state=base_init,
        zero_residual=True,
    )


def _batch(task, rows=(0, 1, 2, 3)):
    return {k: (v.clone() if torch.is_tensor(v) else v) for k, v in PC.collate(task.cache, list(rows)).items()}


def _synthetic_h_and_batch(seed=11):
    g = torch.Generator().manual_seed(seed)
    h = torch.randn(11, C.WIDTH, generator=g)
    atom_batch = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 2, 3, 3], dtype=torch.long)
    return h, {"atom_batch": atom_batch, "n_graphs": 4}


def test_parameter_counts_and_v_gap():
    counts = {k: v for k, v in C.EXPECTED_PARAMETERS.items()}
    assert counts["C"] == 245_184
    assert counts["S"] == 286_848
    assert counts["A"] == 287_644
    assert counts["G"] == 286_912
    assert counts["V"] == 287_808
    for arm in C.ARM_ORDER:
        new = C.ARMS[arm]["new_parameters"]
        rel = abs(new - C.HISTORICAL["V"]["new_parameters"]) / C.HISTORICAL["V"]["new_parameters"]
        assert rel <= C.NEW_PARAMETER_TOLERANCE, (arm, rel)
        assert C.ARMS[arm]["total_parameters"] == counts["C"] + new


def test_actual_parameter_counts(task, reference0):
    for arm in C.ARM_ORDER:
        model, audit = _arm(task, reference0, arm)
        assert audit["base_parameter_tensors"] == 31
        assert audit["base_parameter_elements"] == 245_184
        assert audit["new_parameter_count"] == C.ARMS[arm]["new_parameters"]
        assert audit["total_parameter_count"] == C.EXPECTED_PARAMETERS[arm]
        assert C.parameter_count(model) == C.EXPECTED_PARAMETERS[arm]
        assert model.base_parameter_names() == [n for n, _ in model.named_parameters() if n in set(model.base_parameter_names())]
        assert len(model.new_parameters()) == audit["new_parameter_tensors"]


def test_base_init_matches_historical_phase3(reference0, reference1):
    _, _, state0 = reference0
    _, _, state1 = reference1
    fp0 = C.state_fingerprint(state0)
    fp1 = C.state_fingerprint(state1)
    assert fp0 == C.HISTORICAL_PARITY[0]["base_init_fingerprint"]
    assert fp1 == C.HISTORICAL_PARITY[1]["base_init_fingerprint"]
    assert len(state0) == 37 and len(state1) == 37


def test_zero_residual_identity(task, reference0):
    ref, _, _ = reference0
    ref.eval()
    batch = _batch(task)
    with torch.no_grad():
        base_logits = ref(batch)
    for arm in C.ARM_ORDER:
        model, _ = _arm(task, reference0, arm)
        model.eval()
        with torch.no_grad():
            logits, aux = model(batch, return_aux=True)
        assert torch.equal(logits, base_logits), arm
        if arm == "A":
            assert torch.equal(aux["residual_logits"], torch.zeros_like(aux["residual_logits"]))
        else:
            assert aux["residual_logits"] is None


def test_private_init_does_not_consume_public_rng(task, reference0):
    _, _, base_init = reference0
    torch.manual_seed(4242)
    before = torch.get_rng_state().clone()
    M.build_arm(
        "A", task.spec, task.norm, C.N_TASKS, seed=0, device="cpu",
        base_init_state=base_init, zero_residual=True,
    )
    after = torch.get_rng_state()
    assert torch.equal(before, after)


def test_bond_perturbation_does_not_change_new_modules(task, reference0):
    h, batch = _synthetic_h_and_batch()
    perturbed = dict(batch)
    perturbed["bond_index"] = torch.tensor([[0, 2], [1, 3], [2, 4]], dtype=torch.long)
    perturbed["bond_feat"] = torch.randn(3, 9)
    for arm in C.ARM_ORDER:
        model, _ = _arm(task, reference0, arm)
        model.eval()
        with torch.no_grad():
            h1, a1, p1, r1 = model.propagate(h.clone(), batch, return_attention=True)
            h2, a2, p2, r2 = model.propagate(h.clone(), perturbed, return_attention=True)
        assert torch.equal(h1, h2), arm
        if arm == "A":
            assert torch.equal(a1, a2)
            assert torch.equal(p1, p2)
            assert torch.equal(r1, r2)


def test_S_has_no_inter_atom_aggregation(task, reference0):
    h, batch = _synthetic_h_and_batch()
    model, _ = _arm(task, reference0, "S")
    h2 = h.clone()
    h2[0] = h2[0] + 5.0
    model.eval()
    with torch.no_grad():
        out1, _, _, _ = model.propagate(h, batch, return_attention=True)
        out2, _, _, _ = model.propagate(h2, batch, return_attention=True)
    assert torch.equal(out2[1:], out1[1:])
    assert not torch.equal(out2[0], out1[0])


def test_attention_normalisation_and_graph_isolation(task, reference0):
    h, batch = _synthetic_h_and_batch(seed=12)
    model, _ = _arm(task, reference0, "A")
    model.eval()
    h2 = h.clone()
    h2[batch["atom_batch"] == 0] += 3.0
    with torch.no_grad():
        _, a1, p1, r1 = model.propagate(h, batch, return_attention=True)
        _, a2, p2, r2 = model.propagate(h2, batch, return_attention=True)
    # softmax sums to one per molecule and head
    for g in range(batch["n_graphs"]):
        rows = a1[batch["atom_batch"] == g]
        assert torch.allclose(rows.sum(dim=0), torch.ones(4), atol=1e-6), g
    # changing graph 0 cannot change graphs 1..3
    assert torch.equal(r1[1:], r2[1:])
    assert torch.equal(p1[1:], p2[1:])
    assert torch.equal(a1[batch["atom_batch"] != 0], a2[batch["atom_batch"] != 0])
    assert not torch.equal(p1[0], p2[0])


def test_G_context_does_not_leak_across_graphs(task, reference0):
    h, batch = _synthetic_h_and_batch(seed=13)
    model, _ = _arm(task, reference0, "G")
    model.eval()
    h2 = h.clone()
    h2[batch["atom_batch"] == 0] += 7.0
    with torch.no_grad():
        out1, _, _, _ = model.propagate(h, batch, return_attention=True)
        out2, _, _, _ = model.propagate(h2, batch, return_attention=True)
    mask = batch["atom_batch"] != 0
    assert torch.equal(out1[mask], out2[mask])
    assert not torch.equal(out1[batch["atom_batch"] == 0], out2[batch["atom_batch"] == 0])


def test_permutation_equivariance(task, reference0):
    h, _ = _synthetic_h_and_batch(seed=14)
    n = h.shape[0]
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(9))
    batch = {"atom_batch": torch.zeros(n, dtype=torch.long), "n_graphs": 1}
    batch_perm = {"atom_batch": torch.zeros(n, dtype=torch.long), "n_graphs": 1}
    for arm in C.ARM_ORDER:
        model, _ = _arm(task, reference0, arm)
        model.eval()
        with torch.no_grad():
            out, attn, pooled, residual = model.propagate(h, batch, return_attention=True)
            out_p, attn_p, pooled_p, residual_p = model.propagate(h[perm], batch_perm, return_attention=True)
        if arm == "A":
            assert torch.allclose(out_p, out[perm], atol=1e-6)
            assert torch.allclose(pooled_p, pooled, atol=1e-6)
            assert torch.allclose(attn_p, attn[perm], atol=1e-6)
            assert torch.allclose(residual_p, residual, atol=1e-6)
        else:
            assert torch.allclose(out_p, out[perm], atol=1e-6)


def test_new_parameters_receive_gradients_after_updates(task, reference0):
    model, audit = _arm(task, reference0, "S")
    optimizer, _ = M.make_optimizer_scratch(model)
    batch = _batch(task, rows=(0, 1, 2, 3))
    y = torch.as_tensor(task.labels[np.asarray([0, 1, 2, 3])], dtype=torch.float32)
    model.train()
    for step in range(3):
        M.set_lrs(optimizer, step=step)
        optimizer.zero_grad(set_to_none=True)
        loss = MET.masked_bce_loss(model(batch), y)
        loss.backward()
        optimizer.step()
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    loss = MET.masked_bce_loss(model(batch), y)
    loss.backward()
    for name, param in model.named_parameters():
        if name.startswith(("blocks.", "readout.")):
            assert param.grad is not None, name
            assert torch.isfinite(param.grad).all(), name
            assert float(param.grad.abs().sum()) > 0.0, name


def test_attention_head_gradients_after_updates(task, reference0):
    model, _ = _arm(task, reference0, "A")
    optimizer, _ = M.make_optimizer_scratch(model)
    batch = _batch(task, rows=(0, 1))
    y = torch.as_tensor(task.labels[np.asarray([0, 1])], dtype=torch.float32)
    model.train()
    for step in range(3):
        M.set_lrs(optimizer, step=step)
        optimizer.zero_grad(set_to_none=True)
        MET.masked_bce_loss(model(batch), y).backward()
        optimizer.step()
    model.eval()
    optimizer.zero_grad(set_to_none=True)
    MET.masked_bce_loss(model(batch), y).backward()
    for name, param in model.named_parameters():
        if name.startswith(("blocks.", "readout.")):
            assert param.grad is not None and torch.isfinite(param.grad).all(), name
            assert float(param.grad.abs().sum()) > 0.0, name


def test_masked_bce_matches_manual():
    g = torch.Generator().manual_seed(3)
    logits = torch.randn(9, 5, generator=g)
    y = torch.randn(9, 5, generator=g)
    y[y.abs() < 0.4] = float("nan")
    y = (y > 0).float()
    y[y.abs() > 0.9] = float("nan")
    mask = torch.isfinite(y)
    expected = torch.nn.functional.binary_cross_entropy_with_logits(logits[mask], y[mask])
    assert torch.allclose(MET.masked_bce_loss(logits, y), expected)
    assert MET.masked_bce_loss(logits, torch.full_like(y, float("nan"))).item() == 0.0


def test_lr_schedule_is_frozen_phase3_shape():
    audit = C.lr_audit()
    assert audit["steps_per_epoch"] == 2738
    assert audit["total_steps_40_epochs"] == 109520
    assert math.isclose(audit["step0"], 3.0098612125639153e-05, rel_tol=0, abs_tol=1e-18)
    assert math.isclose(audit["warmup_last"], 3.0e-4, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(audit["post_warmup_first"], 3.0e-4, rel_tol=0, abs_tol=1e-12)
    assert math.isclose(audit["step40e"], 1.0e-5, rel_tol=0, abs_tol=1e-15)
    # monotone non-increasing after warmup
    vals = [C.scheduled_lr(s) for s in range(2738, 109520, 1000)]
    assert all(b <= a + 1e-15 for a, b in zip(vals, vals[1:]))


def test_order_plan_matches_historical_phase3(task):
    for seed in (0, 1):
        plan = D.make_plan40(task, seed=seed)
        assert len(plan) == 40
        assert plan[0]["global_plan_sha256"] == C.HISTORICAL_PARITY[seed]["global_plan_sha256"]
        summary = D.plan_summary(plan, task)
        assert summary["isolation"]["passed"] is True
        assert summary["summary"]["steps_per_epoch"] == 2738
        # no valid/test rows and each epoch is a Train permutation
        for record in summary["isolation"]["checks"]:
            assert record["valid_rows_in_epoch"] == 0
            assert record["test_rows_in_epoch"] == 0
            assert record["is_train_permutation"] is True


def test_test_labels_are_nan_and_not_evaluated(task):
    assert np.isfinite(task.labels[task.test_rows]).sum() == 0
    assert np.isfinite(task.labels[task.train_rows]).sum() > 0
    assert np.isfinite(task.labels[task.valid_rows]).sum() > 0
    # The metric path only accepts explicitly supplied rows; no Test helper exists.
    report = D.evaluate_validation
    assert report.__module__.split(".")[-1] in ("sg_data", "nmp_data")


def test_initialisation_parity_for_every_arm(task, reference0):
    _, _, base_init = reference0
    fps = {}
    for arm in C.ARM_ORDER:
        model, audit = _arm(task, reference0, arm)
        fps[arm] = audit["base_init_fingerprint"]
        assert audit["base_state_fingerprint_after_load"] == audit["base_init_fingerprint"]
    assert set(fps.values()) == {C.HISTORICAL_PARITY[0]["base_init_fingerprint"]}

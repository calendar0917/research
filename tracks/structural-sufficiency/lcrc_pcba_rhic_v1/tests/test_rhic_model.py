"""CPU tests for the LCRC-RHIC architecture, initialisation and invariants."""

from __future__ import annotations

import math

import numpy as np
import pytest
import torch

import rhic_audit as A
import rhic_common as C
import rhic_model as M

NEW_TENSORS = 24


def test_parameter_budget_is_exact(model_audit):
    audit = model_audit["parameters"]
    assert audit["total"] == C.EXPECTED_PARAMETERS["total"] == 1_117_253
    assert audit["groups"] == {"w2_base": 848_096, "w2_readout": 146_228, "rhic": 122_929}
    assert audit["rhic_subgroups"] == {
        "token": 40_561,
        "branch1": 20_160,
        "branch2": 6_208,
        "root1": 31_296,
        "root2": 24_704,
    }
    assert sum(audit["rhic_subgroups"].values()) == 122_929
    assert audit["groups"]["w2_base"] + audit["groups"]["w2_readout"] == 994_324
    assert audit["all_match"]
    assert all(audit["checks"].values()), audit["checks"]
    assert audit["n_parameter_tensors"] == 75
    assert model_audit["n_state_tensors"] == len(model_audit["state_keys"]) == 81


def test_new_parameter_set(model):
    names = [n for n in model.state_dict() if C.is_new_parameter(n)]
    assert len(names) == NEW_TENSORS
    assert all(n.startswith("rhic.") for n in names)
    assert not any(n.startswith(("atom_encoder.", "bond_encoder.", "local.", "readout.", "f.", "rho.")) for n in names)


def test_w2_part_bitwise_equal_recorded_w2(model_audit):
    assert model_audit["w2_part_bitwise_equal_reference"]["all_equal"]
    assert model_audit["w2_part_bitwise_equal_reference"]["n_tensors"] > 0
    assert model_audit["w2_trunk_fingerprint_matches_recorded_w2"]
    assert model_audit["w2_trunk_init_fingerprint"] == C.W2_SEED0["trunk_init_fingerprint"]


def test_zero_output_and_global_rng(task):
    import bs_model as BM

    # the RHIC part must add exactly zero draws to the global stream: building
    # the model consumes the same global RNG as building the W2 base alone.
    torch.manual_seed(7)
    BM.build_model("W2", task.spec, task.norm, C.N_TASKS, seed=C.SEED)
    after_base = torch.get_rng_state().clone()
    torch.manual_seed(7)
    m1, a1 = M.build_rhic_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    assert torch.equal(after_base, torch.get_rng_state())
    assert a1["init"]["global_rng_unchanged"]
    assert a1["init"]["final_root_linear_zeroed"]
    assert a1["init"]["linear_layers_initialised"] == 11

    # direct private init never touches the global stream
    before = torch.get_rng_state().clone()
    M.initialize_rhic_modules(m1, seed=C.SEED)
    assert torch.equal(before, torch.get_rng_state())

    # deterministic private init, independent of the ambient global stream
    s1 = C.state_fingerprint(C.state_subset(m1.state_dict(), new=True))
    torch.manual_seed(999)
    m2, _ = M.build_rhic_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    s2 = C.state_fingerprint(C.state_subset(m2.state_dict(), new=True))
    assert s1 == s2
    other, _ = M.build_rhic_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED + 1, device="cpu")
    assert C.state_fingerprint(C.state_subset(other.state_dict(), new=True)) != s1
    assert float(m1.rhic.root2.weight.abs().max()) == 0.0
    assert float(m1.rhic.root2.bias.abs().max()) == 0.0
    assert float(m1.rhic.root1.weight.abs().max()) > 0.0
    assert float(m1.rhic.branch1.weight.abs().max()) > 0.0


def test_epoch0_logits_equal_untrained_w2(task_with_cpc, model, reference):
    task = task_with_cpc
    rows = [int(x) for x in task.valid_rows[:8]]
    batch = C.P.collate_with_cpc(task, task.cpc, rows)
    model.eval()
    reference.eval()
    with torch.inference_mode():
        logits, aux = model(batch, return_aux=True)
        ref_logits = reference(batch)
    assert torch.equal(aux["delta"], torch.zeros_like(aux["delta"]))
    assert torch.equal(logits, ref_logits)
    assert torch.equal(aux["h"], aux["h0"])


def test_zero_path_atoms_receive_exactly_zero_delta(task, model):
    model.eval()
    batch = A.synthetic_batch(task.spec, 1, [])
    with torch.inference_mode():
        _l, aux = model(batch, return_aux=True)
    assert int(batch["cpc_path_atom"].numel()) == 0
    assert torch.equal(aux["delta"], torch.zeros_like(aux["delta"]))
    assert aux["rhic"]["n_paths"] == 0


def test_usage_metrics_fields(task_with_cpc, model):
    task = task_with_cpc
    rows = [int(x) for x in task.train_rows[:4]]
    batch = C.P.collate_with_cpc(task, task.cpc, rows)
    model.eval()
    with torch.inference_mode():
        usage = M.usage_metrics(model, batch)
    for key in (
        "R",
        "rms_h0",
        "rms_delta",
        "rms_h",
        "rms_branch_mu",
        "rms_branch_q",
        "rms_branch_value",
        "rms_mean_branch",
        "rms_cross_branch",
        "rms_z",
        "rms_v",
        "branch_active_fraction",
        "multi_path_branch_fraction",
        "q_nonzero_fraction",
        "cross_nonzero_fraction",
        "finite_fraction_h",
    ):
        assert key in usage, key
    assert usage["finite_fraction_h"] == 1.0
    assert 0.0 <= usage["branch_active_fraction"] <= 1.0
    assert np.isfinite([usage[k] for k in usage if isinstance(usage[k], float)]).all()


def test_predict_logits_matches_forward(task_with_cpc, model):
    task = task_with_cpc
    rows = [int(x) for x in task.valid_rows[:3]]
    batch = C.P.collate_with_cpc(task, task.cpc, rows)
    model.eval()
    with torch.inference_mode():
        direct = model(batch)
        predicted = M.predict_logits(model, task, rows, torch.device("cpu"), batch_size=64)
    assert predicted.shape == (3, C.N_TASKS)
    assert np.isfinite(predicted).all()
    assert float(np.abs(predicted - direct.numpy()).max()) <= 1e-6


def test_independent_statistics(task_with_cpc, model):
    task = task_with_cpc
    rows = [int(x) for x in task.valid_rows[:4]]
    batch = C.P.collate_with_cpc(task, task.cpc, rows)
    out = A.independent_stats_check(model, batch)
    assert out["all_ok"], out["checks"]


def test_architecture_gate(task_with_cpc):
    out = A.gate_architecture(task_with_cpc, torch.device("cpu"))
    assert out["all_ok"], out["checks"]
    assert out["rhic_forward_calls"] == 1
    assert out["post_local_calls"] == 1
    assert out["module_calls"]["readout"] == 1
    assert out["branch_shuffle_tokens_identical"]
    assert out["branch_shuffle_branch_counts_identical"]
    assert out["branch_shuffle_branch_groups_differ"]
    assert out["remote_endpoint_root_delta_max_abs_diff"] > 1e-8


def test_witness(task):
    out = A.witness(task)
    assert out["all_ok"], out["checks"]
    assert out["cpc_global_mean_max_abs_diff"] < 1e-6
    assert out["branch_q_max_abs_diff"] > 0.01
    assert out["root_delta_max_abs_diff"] > 1e-6


def test_gradient_gate_starts_closed_then_flows(task_with_cpc):
    task = task_with_cpc
    C.seed_everything(C.SEED)
    model, _ = M.build_rhic_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    optimizer = torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad],
        lr=float(C.RECIPE["warmup_start_lr"]),
        betas=tuple(C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )
    # the mini-cache covers only the first 24 Train rows; the gradient gate does
    # not depend on the historical order, so use three fixed slices of them.
    cache_rows = np.asarray(task.train_rows[:24], dtype=np.int64)
    model.train()
    report = {}
    for step in range(1, 4):
        rows = cache_rows[(step - 1) * 8 : step * 8]
        batch = C.P.collate_with_cpc(task, task.cpc, rows.tolist())
        y = torch.as_tensor(task.labels[rows], dtype=torch.float32)
        C.set_lrs(optimizer, step=step - 1)
        logits = model(batch)
        loss = C.MET.masked_bce_loss(logits, y)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        grads = {
            n: (0.0 if p.grad is None else float(p.grad.detach().norm()))
            for n, p in model.named_parameters()
            if C.is_new_parameter(n)
        }
        report[step] = grads
        torch.nn.utils.clip_grad_norm_(model.parameters(), float(C.RECIPE["gradient_clip"]))
        optimizer.step()
    # step 1: the zeroed final linear keeps the gate closed, only its own tensors move
    nonzero_1 = {n for n, v in report[1].items() if v > 0.0}
    assert nonzero_1 <= {"rhic.root2.weight", "rhic.root2.bias"}, nonzero_1
    assert "rhic.root2.weight" in nonzero_1
    # after one update the gate is open and every new tensor gets a gradient
    for step in (2, 3):
        zero = {n for n, v in report[step].items() if v == 0.0}
        assert not zero, (step, zero)
    assert len(report[3]) == NEW_TENSORS
    assert float(model.rhic.root2.weight.abs().max()) > 0.0

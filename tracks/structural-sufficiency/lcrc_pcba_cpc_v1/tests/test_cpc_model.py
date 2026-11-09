"""CPU correctness tests for LCRC-CPC."""

from __future__ import annotations

import numpy as np
import torch

import cpc_audit as A
import cpc_common as C
import cpc_model as M
import cpc_paths as P


def _logits(model, batch, **kwargs):
    model.eval()
    with torch.inference_mode():
        return model(batch, **kwargs).detach().float()


def test_parameter_budget_is_exact(model, model_audit):
    audit = model_audit["parameters"]
    assert audit["total"] == 1_043_333
    assert audit["groups"]["w2_base"] == 848_096
    assert audit["groups"]["w2_readout"] == 146_228
    assert audit["groups"]["cpc"] == 49_009
    assert audit["cpc_subgroups"] == {
        "norm": 256,
        "role_projections": 12_384,
        "bond_projection": 3_088,
        "path_mlp_1": 18_528,
        "path_mlp_2": 6_208,
        "gate": 97,
        "output": 8_448,
    }
    assert audit["all_match"]
    assert int(sum(p.numel() for p in model.parameters())) == 1_043_333


def test_w2_part_is_bitwise_the_historical_seed0_constructor(model, model_audit, reference):
    state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
    ours = C.state_subset(state, new=False)
    ref = {k: v.detach().cpu() for k, v in reference.state_dict().items()}
    cmp = M.compare_states(ours, ref)
    assert cmp["all_equal"] and cmp["n_tensors"] == 57
    assert model_audit["w2_trunk_fingerprint_matches_recorded_w2"]
    assert (
        model_audit["w2_trunk_init_fingerprint"]
        == "5f7101a8f378e7c9283caaade8d61eee733c0c9f3cfb65b5584c8a1e495ef63d"
    )


def test_private_initialisation_and_zero_output(model, model_audit):
    init = model_audit["init"]
    assert init["global_rng_unchanged"] is True
    assert init["linear_layers_initialised"] == 8
    assert init["output_zeroed"] is True
    assert float(model.cpc.out.weight.abs().max()) == 0.0
    assert float(model.cpc.out.bias.abs().max()) == 0.0
    assert float(model.cpc.path1.weight.abs().max()) > 0.0
    assert float(model.cpc.gate.weight.abs().max()) > 0.0
    assert float(model.cpc.norm.weight.min()) == 1.0


def test_epoch0_logits_equal_untrained_w2(model, reference, batch):
    assert float((_logits(model, batch) - _logits(reference, batch)).abs().max()) == 0.0


def test_delta_zero_mode_equals_w2(nondegenerate, reference, batch):
    a = _logits(nondegenerate, batch, mode="delta_zero")
    b = _logits(reference, batch)
    assert float((a - b).abs().max()) <= 1e-6


def test_atom_renumbering_and_bond_direction(model, batch):
    perm, _p = A._permute_atoms_with_paths(batch, seed=20261110)
    base = _logits(model, batch)
    assert float((base - _logits(model, perm)).abs().max()) <= 1e-5
    flip = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    flip["bond_index"] = batch["bond_index"][:, [1, 0]].contiguous()
    assert float((base - _logits(model, flip)).abs().max()) <= 1e-5


def test_molecule_isolation(model, batch):
    base = _logits(model, batch)
    perturbed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    mask = (batch["atom_batch"] == 1).unsqueeze(1)
    gen = torch.Generator().manual_seed(7)
    noise = torch.randn(perturbed["atom_feat"].shape, generator=gen)
    perturbed["atom_feat"] = torch.where(mask, perturbed["atom_feat"] + noise, perturbed["atom_feat"])
    after = _logits(model, perturbed)
    assert float((base[0] - after[0]).abs().max()) <= 1e-6
    assert float((base[1] - after[1]).abs().max()) > 1e-4


def test_zero_path_delta_is_exactly_zero(nondegenerate):
    h0 = torch.zeros((4, C.WIDTH))
    b = torch.zeros((0, C.BOND_WIDTH))
    empty = {
        "cpc_path_atom": torch.zeros(0, dtype=torch.long),
        "cpc_middle": torch.zeros(0, dtype=torch.long),
        "cpc_end": torch.zeros(0, dtype=torch.long),
        "cpc_bond_ij": torch.zeros(0, dtype=torch.long),
        "cpc_bond_jk": torch.zeros(0, dtype=torch.long),
    }
    with torch.inference_mode():
        delta, _aux = nondegenerate.cpc(h0, b, empty)
    assert float(delta.abs().max()) == 0.0


def test_nonrecursive_module_call_counts(task_with_cpc, model):
    batch = P.collate_with_cpc(
        task_with_cpc, task_with_cpc.cpc, [int(x) for x in task_with_cpc.valid_rows[:4]]
    )
    out = A.gate_architecture(task_with_cpc, torch.device("cpu"))
    assert out["all_ok"], out["checks"]
    assert out["module_calls"]["atom_encoder"] == 1
    assert out["module_calls"]["local"] == 1
    assert out["module_calls"]["drop"] == 1
    assert out["module_calls"]["cpc"] == 1
    assert out["post_local_calls"] == 1
    assert out["cpc_input_h0_equal_drop_local"]
    assert out["remote_endpoint_root_delta_max_abs_diff"] > 1e-8


def test_non_additive_pairing_witness(task):
    out = A.gate_witness(task)
    assert out["all_ok"], out["checks"]
    assert out["additive_reference_max_abs_diff"] == 0.0
    assert out["cpc_delta_max_abs_diff"] > 1e-7
    assert out["cpc_context_max_abs_diff"] > 1e-7


def test_training_step_and_validation_helper(task_with_cpc, batch):
    import copy

    import cpc_train as T

    rows = [int(x) for x in task_with_cpc.train_rows[:6]]
    y = torch.as_tensor(task_with_cpc.labels[rows], dtype=torch.float32)
    model = M.build_cpc_model(
        task_with_cpc.spec, task_with_cpc.norm, C.N_TASKS, seed=C.SEED, device="cpu"
    )[0]
    optimizer, _audit = T.build_optimizer(model)
    model.train()
    for step in range(3):
        out = T.train_step(model, optimizer, batch, y, global_step=step, clip=5.0)
        assert np.isfinite(out["loss"]) and np.isfinite(out["grad_norm"])
    nonzero = [
        n
        for n, p in model.cpc.named_parameters()
        if p.grad is not None and float(p.grad.abs().max()) > 0.0
    ]
    assert len(nonzero) >= 17

    small = copy.copy(task_with_cpc)
    small.valid_rows = np.asarray(task_with_cpc.valid_rows[:8], dtype=np.int64)
    report = T.evaluate_validation(model, small, torch.device("cpu"), reason="unit_test")
    assert np.isfinite(report["ap"]) and np.isfinite(report["valid_bce"])
    assert report["evaluated_rows"] == 8
    assert len(report["per_task_ap"]) == C.N_TASKS

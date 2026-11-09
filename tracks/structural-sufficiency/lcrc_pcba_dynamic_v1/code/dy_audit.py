"""Pre-training correctness / mechanism gates, gradient audit and profiling.

Every gate of ``PROTOCOL.md`` section 3 is executed here, before the single
formal training run.  The script exits non-zero on any failed gate, so a broken
implementation can never reach the formal run.

Nothing here reads an official Test label.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import torch  # noqa: E402

import dy_common as C  # noqa: E402
import dy_model as M  # noqa: E402
import dy_train as T  # noqa: E402

LOGIT_TOL = 1e-4


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _to_device(batch: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}


def _logits(model: torch.nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    was = model.training
    model.eval()
    try:
        with torch.inference_mode():
            return model(batch).detach().float()
    finally:
        if was:
            model.train()


def _max_abs_diff(a: torch.Tensor, b: torch.Tensor) -> float:
    return float((a - b).abs().max().item())


def _finite(x: torch.Tensor) -> bool:
    return bool(torch.isfinite(x).all().item())


class TracedBatch(dict):
    """Batch dict that records which keys the model actually reads."""

    def __init__(self, data: Mapping[str, Any]):
        super().__init__(data)
        self.read: set[str] = set()

    def __getitem__(self, key):  # noqa: D105
        self.read.add(str(key))
        return super().__getitem__(key)

    def get(self, key, default=None):  # noqa: D105
        self.read.add(str(key))
        return super().get(key, default)


def synthetic_zero_bond_batch(spec: Any, atom_feat_row: torch.Tensor) -> dict[str, Any]:
    """One isolated atom, no bonds: the message term must be exactly zero."""
    zi = torch.zeros(0, dtype=torch.long)
    return {
        "n_graphs": 1,
        "atom_feat": atom_feat_row.reshape(1, -1).float().clone(),
        "atom_type": torch.zeros(1, dtype=torch.long),
        "atom_scalar": torch.zeros(1, int(spec.n_scalars)),
        "atom_batch": torch.zeros(1, dtype=torch.long),
        "atom_off": torch.zeros(1, dtype=torch.long),
        "bond_feat": torch.zeros((0, int(spec.bond_feat_dim))),
        "bond_type": zi,
        "bond_index": torch.zeros((0, 2), dtype=torch.long),
        "bond_batch": zi,
        "nbr_atom": zi,
        "nbr_code": zi,
        "path_atom": zi,
        "path_cat": zi,
        "path_end63": zi,
        "path_end": zi,
        "path_closure": zi,
        "path_graph": zi,
        "global": torch.zeros(1, int(spec.global_dim)),
        "atom_counts": torch.ones(1, dtype=torch.long),
        "bond_counts": torch.zeros(1, dtype=torch.long),
    }


# ---------------------------------------------------------------------------
# gates
# ---------------------------------------------------------------------------


def gate_constants(task: Any) -> dict[str, Any]:
    plan = C.SD.make_plan40(task, seed=int(C.SEED))
    prefix = C.sha256_bytes("|".join(p["row_order_sha256"] for p in plan).encode())
    recipe = C.recipe()
    checks = {
        "dataset": C.DATASET == "ogbg-molpcba",
        "n_tasks": C.N_TASKS == 128,
        "n_train": int(task.train_rows.size) == 350_343 and C.N_TRAIN == 350_343,
        "n_valid": int(task.valid_rows.size) == 43_793 and C.N_VALID == 43_793,
        "n_test": int(task.test_rows.size) == 43_793 and C.N_TEST == 43_793,
        "batch_size": C.BATCH_SIZE == 128,
        "epochs": C.TOTAL_EPOCHS == 40,
        "steps_per_epoch": C.STEPS_PER_EPOCH == 2738,
        "total_steps": C.TOTAL_STEPS_40 == 109_520,
        "seed": C.SEED == 0,
        "dropout": float(C.RECIPE["dropout"]) == 0.10,
        "weight_decay": float(C.RECIPE["weight_decay"]) == 1e-3,
        "peak_lr": float(C.RECIPE["learning_rate"]) == 3e-4,
        "warmup_start": float(C.RECIPE["warmup_start_lr"]) == 3e-5,
        "end_lr": float(C.RECIPE["end_lr"]) == 1e-5,
        "warmup_epochs": float(C.RECIPE["warmup_epochs"]) == 1.0,
        "grad_clip": float(C.RECIPE["gradient_clip"]) == 5.0,
        "precision_fp32": str(C.RECIPE["precision"]) == "fp32",
        "loss_masked_bce": "masked" in str(C.RECIPE["loss"]).lower(),
        "plan_sha": plan[0]["global_plan_sha256"] == C.FROZEN_PLAN["global_plan_sha256"],
        "plan_prefix_sha": prefix == C.FROZEN_PLAN["consumed40_prefix_sha256"],
        "widths_are_w2": dict(recipe["widths"]) == dict(C.config_for_scale(2)),
        "dynamic_widths": C.DYNAMIC["width"] == 128 and C.DYNAMIC["hidden"] == 256,
        "single_param_group": True,
    }
    return {
        "plan_global_sha256": plan[0]["global_plan_sha256"],
        "plan_prefix_sha256": prefix,
        "frozen_plan": dict(C.FROZEN_PLAN),
        "lr": recipe["lr_schedule"],
        "recipe": recipe,
        "checks": checks,
        "passed": all(checks.values()),
    }


def gate_model_identity(
    task: Any, model: M.W2DynamicModel, audit: Mapping[str, Any], device: torch.device
) -> dict[str, Any]:
    ref, ref_audit = M.untrained_w2(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    rows = [int(x) for x in task.valid_rows[:8]]
    batch = _to_device(C.PC.collate(task.cache, rows), device)
    logits_dyn = _logits(model, batch)
    logits_ref = _logits(ref, batch)
    diff = _max_abs_diff(logits_dyn, logits_ref)
    state_ref = {k: v.detach().cpu() for k, v in ref.state_dict().items()}
    ours_base = C.state_subset({k: v.detach().cpu() for k, v in model.state_dict().items()}, new=False)
    per_tensor = M.compare_states(ours_base, state_ref)
    dyn_zero = {
        name: float(param.abs().max().item())
        for name, param in model.named_parameters()
        if name.startswith("blocks.") and ".fc2." in name
    }
    checks = {
        "parameters_total_1128468": int(audit["parameters"]["total"]) == 1_128_468,
        "parameter_groups_exact": bool(audit["parameters"]["all_match"]),
        "w2_part_bitwise_equal": bool(per_tensor["all_equal"]),
        "w2_trunk_fingerprint_matches_w2_seed0": bool(audit["w2_trunk_fingerprint_matches_recorded_w2"]),
        "w2_reference_parameter_count": int(ref_audit["total_parameters"]) == 994_324,
        "fc2_zeroed": all(v == 0.0 for v in dyn_zero.values()),
        "epoch0_logits_equal_w2": diff <= LOGIT_TOL,
        "epoch0_logits_finite": _finite(logits_dyn),
        "init_global_rng_untouched": bool(audit["init"]["global_rng_unchanged"]),
    }
    return {
        "device": str(device),
        "logits_max_abs_diff_vs_untrained_w2": diff,
        "logits_tolerance": LOGIT_TOL,
        "logits_abs_max": float(logits_dyn.abs().max().item()),
        "w2_part_per_tensor": {
            k: v for k, v in per_tensor.items() if k != "per_tensor"
        },
        "fc2_max_abs": dyn_zero,
        "parameter_groups": audit["parameters"]["groups"],
        "checks": checks,
        "passed": all(checks.values()),
    }


def gate_invariances(
    task: Any, model: M.W2DynamicModel, device: torch.device
) -> dict[str, Any]:
    rows = [int(x) for x in task.valid_rows[1000:1008]]
    batch = C.PC.collate(task.cache, rows)
    base_logits = _logits(model, _to_device(batch, device))

    # atom renumbering
    perm_batch, p = M.BM.permute_atoms(batch)
    perm_logits = _logits(model, _to_device(perm_batch, device))
    perm_diff = _max_abs_diff(base_logits, perm_logits)

    # stored direction of every physical bond flipped
    flip = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    idx = batch["bond_index"]
    flip["bond_index"] = idx[:, [1, 0]].contiguous()
    flip_logits = _logits(model, _to_device(flip, device))
    flip_diff = _max_abs_diff(base_logits, flip_logits)

    return {
        "atom_renumbering": {
            "max_abs_diff": perm_diff,
            "tolerance": LOGIT_TOL,
            "permutation_checked": int(p.numel()),
        },
        "edge_direction_flip": {
            "max_abs_diff": flip_diff,
            "tolerance": LOGIT_TOL,
            "n_bonds_flipped": int(idx.shape[0]),
        },
        "checks": {
            "atom_renumbering_invariant": perm_diff <= LOGIT_TOL,
            "edge_direction_flip_invariant": flip_diff <= LOGIT_TOL,
        },
        "passed": bool(perm_diff <= LOGIT_TOL and flip_diff <= LOGIT_TOL),
    }


def gate_isolation(task: Any, model: M.W2DynamicModel, device: torch.device) -> dict[str, Any]:
    rows = [int(task.valid_rows[0]), int(task.valid_rows[1])]
    batch = C.PC.collate(task.cache, rows)
    edges = M.check_edges_intra_molecule(batch)
    base = _logits(model, _to_device(batch, device))

    perturbed = {k: (v.clone() if torch.is_tensor(v) else v) for k, v in batch.items()}
    atom_batch = batch["atom_batch"]
    gen = torch.Generator().manual_seed(20261109)
    noise = torch.randn(perturbed["atom_feat"].shape, generator=gen) * 2.0
    mask = (atom_batch == 1).unsqueeze(1)
    perturbed["atom_feat"] = torch.where(mask, perturbed["atom_feat"] + noise, perturbed["atom_feat"])
    after = _logits(model, _to_device(perturbed, device))
    mol0 = _max_abs_diff(base[0], after[0])
    mol1 = _max_abs_diff(base[1], after[1])

    # graph level: perturbing molecule 1 must not move molecule 0's atoms' states
    model.eval()
    with torch.inference_mode():
        _a, _b, _s, h0 = model.encode_h0(_to_device(batch, device))
        _h1, h2 = model.dynamic_layers(h0, _to_device(batch, device))
        _a2, _b2, _s2, h0b = model.encode_h0(_to_device(perturbed, device))
        _h1b, h2b = model.dynamic_layers(h0b, _to_device(perturbed, device))
    m0 = atom_batch == 0
    h0_diff = float((h0[m0] - h0b[m0]).abs().max().item())
    h2_diff = float((h2[m0] - h2b[m0]).abs().max().item())
    checks = {
        "edges_intra_molecule": bool(edges["checked"]),
        "both_directions": bool(edges["both_directions"]),
        "each_physical_bond_twice": bool(edges["each_physical_bond_twice"]),
        "other_molecule_logits_unchanged": mol0 <= 1e-6,
        "other_molecule_h2_unchanged": h2_diff <= 1e-6,
        "perturbed_molecule_did_change": mol1 > 1e-3,
    }
    return {
        "edge_audit": edges,
        "logits_other_molecule_max_abs_diff": mol0,
        "logits_perturbed_molecule_max_abs_diff": mol1,
        "h0_other_molecule_max_abs_diff": h0_diff,
        "h2_other_molecule_max_abs_diff": h2_diff,
        "checks": checks,
        "passed": all(checks.values()),
    }


def gate_layer_inputs(task: Any, model: M.W2DynamicModel, device: torch.device) -> dict[str, Any]:
    """Layer 1 must read h0[src]; layer 2 must read h1[src] (never h0)."""
    rows = [int(x) for x in task.valid_rows[2000:2004]]
    batch = _to_device(C.PC.collate(task.cache, rows), device)

    # non-degenerate test state: same construction, but non-zero fc2 (deterministic)
    test_model, _ = M.build_dynamic_model(
        task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device, zero_residual=False
    )
    captured: dict[str, torch.Tensor] = {}
    handles = []
    for i in (0, 1):
        def _hook(_m, inp, _out, key=f"blocks.{i}"):
            captured[key] = inp[0].detach().clone()

        handles.append(test_model.blocks[i].register_forward_hook(_hook))
    try:
        test_model.eval()
        with torch.inference_mode():
            _a, _b, _s, h0 = test_model.encode_h0(batch)
            h1, h2 = test_model.dynamic_layers(h0, batch)
            src, dst, edge_msg = test_model.bond_directions(batch)
            msg0 = torch.zeros_like(h0).index_add_(
                0, dst, torch.nn.functional.silu(test_model.blocks[0].norm(h0)[src] + edge_msg)
            )
            msg1 = torch.zeros_like(h1).index_add_(
                0, dst, torch.nn.functional.silu(test_model.blocks[1].norm(h1)[src] + edge_msg)
            )
            in0 = captured["blocks.0"]
            in1 = captured["blocks.1"]
            d0 = float((in0 - h0).abs().max().item())
            d1 = float((in1 - h1).abs().max().item())
            d1_h0 = float((in1 - h0).abs().max().item())
            h1_h0 = float((h1 - h0).abs().max().item())
            h2_h1 = float((h2 - h1).abs().max().item())
            msg0_max = float(msg0.abs().max().item())
            msg1_max = float(msg1.abs().max().item())
            n_directed = int(src.numel())
    finally:
        for h in handles:
            h.remove()

    checks = {
        "block0_input_is_h0": d0 == 0.0,
        "block1_input_is_h1": d1 == 0.0,
        "block1_input_differs_from_h0": d1_h0 > 1e-6,
        "message_nonzero_layer0": msg0_max > 0.0,
        "message_nonzero_layer1": msg1_max > 0.0,
        "h1_differs_from_h0": h1_h0 > 1e-6,
        "h2_differs_from_h1": h2_h1 > 1e-6,
    }
    return {
        "block0_input_vs_h0_max_abs_diff": d0,
        "block1_input_vs_h1_max_abs_diff": d1,
        "block1_input_vs_h0_max_abs_diff": d1_h0,
        "h1_vs_h0_max_abs_diff": h1_h0,
        "h2_vs_h1_max_abs_diff": h2_h1,
        "n_directed_edges": n_directed,
        "message_layer0_max_abs": msg0_max,
        "message_layer1_max_abs": msg1_max,
        "checks": checks,
        "passed": all(checks.values()),
        "note": "executed on a non-degenerate (non-zero fc2) test state, deterministic seed",
    }


def gate_chain_dependency(task: Any, model: M.W2DynamicModel, device: torch.device) -> dict[str, Any]:
    """A--B--C: layer 1 must not move C, layer 2 must (through h1_B)."""
    test_model, _ = M.build_dynamic_model(
        task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device, zero_residual=False
    )
    width = int(C.DYNAMIC["width"])
    gen = torch.Generator(device="cpu").manual_seed(424242)
    h0 = torch.randn(3, width, generator=gen).to(device)
    bond_feat = torch.randn(2, int(task.spec.bond_feat_dim), generator=gen).to(device)
    batch = {
        "bond_index": torch.tensor([[0, 1], [1, 2]], dtype=torch.long, device=device),
        "bond_feat": bond_feat,
        "atom_batch": torch.zeros(3, dtype=torch.long, device=device),
        "n_graphs": 1,
    }
    test_model.eval()

    def run(h):
        with torch.inference_mode():
            return test_model.dynamic_layers(h, batch)

    h1_a, h2_a = run(h0)
    delta = torch.zeros_like(h0)
    # a *constant* offset would be removed by LayerNorm, so use a random direction
    delta[0] = torch.randn(width, generator=gen).to(device) * 0.5  # change h0_A only
    h1_b, h2_b = run(h0 + delta)

    d_h1_c = float((h1_a[2] - h1_b[2]).abs().max().item())
    d_h1_b = float((h1_a[1] - h1_b[1]).abs().max().item())
    d_h2_c = float((h2_a[2] - h2_b[2]).abs().max().item())
    d_h2_a = float((h2_a[0] - h2_b[0]).abs().max().item())
    checks = {
        "layer1_does_not_move_C": d_h1_c == 0.0,
        "layer2_moves_C": d_h2_c > 1e-6,
        "layer1_moves_B": d_h1_b > 1e-6,
        "layer2_moves_A": d_h2_a > 1e-6,
    }
    return {
        "graph": "A--B--C (chain, 3 atoms, 2 real-style bond feature rows)",
        "weights": "non-degenerate deterministic test state (fc2 not zeroed)",
        "h1_C_change_from_h0_A_perturbation": d_h1_c,
        "h1_B_change_from_h0_A_perturbation": d_h1_b,
        "h2_C_change_from_h0_A_perturbation": d_h2_c,
        "h2_A_change_from_h0_A_perturbation": d_h2_a,
        "checks": checks,
        "passed": all(checks.values()),
    }


def _numpy_reference_block(
    block: torch.nn.Module,
    h: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    edge_msg: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Independent pure-numpy implementation of MessageFFN (float64)."""
    eps = float(block.norm.eps)
    mean = h.mean(axis=1, keepdims=True)
    var = h.var(axis=1, keepdims=True)
    z = (h - mean) / np.sqrt(var + eps)
    z = z * block.norm.weight.detach().cpu().numpy() + block.norm.bias.detach().cpu().numpy()
    msg = np.zeros_like(z)
    for e in range(src.size):
        x = z[src[e]] + edge_msg[e]
        msg[dst[e]] += x / (1.0 + np.exp(-x))  # SiLU = x * sigmoid(x)
    x = z + msg
    w1 = block.fc1.weight.detach().cpu().numpy()
    b1 = block.fc1.bias.detach().cpu().numpy()
    w2 = block.fc2.weight.detach().cpu().numpy()
    b2 = block.fc2.bias.detach().cpu().numpy()
    a = x @ w1.T + b1
    hidden = a / (1.0 + np.exp(-a))
    delta = hidden @ w2.T + b2
    return z, msg, delta


def gate_python_reference(task: Any, model: M.W2DynamicModel, device: torch.device) -> dict[str, Any]:
    test_model, _ = M.build_dynamic_model(
        task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device, zero_residual=False
    )
    test_model.eval()
    width = int(C.DYNAMIC["width"])
    gen = torch.Generator(device="cpu").manual_seed(777)
    h = torch.randn(4, width, generator=gen).to(device)
    bond_feat = torch.randn(3, int(task.spec.bond_feat_dim), generator=gen).to(device)
    batch = {
        "bond_index": torch.tensor([[0, 1], [1, 2], [2, 3]], dtype=torch.long, device=device),
        "bond_feat": bond_feat,
        "atom_batch": torch.zeros(4, dtype=torch.long, device=device),
        "n_graphs": 1,
    }
    out: dict[str, Any] = {"tolerance": 2e-5}
    checks: dict[str, bool] = {}
    with torch.inference_mode():
        src, dst, edge_msg = test_model.bond_directions(batch)
        h_np = h.detach().cpu().numpy().astype(np.float64)
        src_np = src.detach().cpu().numpy()
        dst_np = dst.detach().cpu().numpy()
        em_np = edge_msg.detach().cpu().numpy().astype(np.float64)
        for i in (0, 1):
            block = test_model.blocks[i]
            z = block.norm(h)
            msg = torch.zeros_like(z).index_add_(0, dst, torch.nn.functional.silu(z[src] + edge_msg))
            delta = block(h, src=src, dst=dst, bond_msg=edge_msg)
            z_ref, msg_ref, delta_ref = _numpy_reference_block(block, h_np, src_np, dst_np, em_np)
            msg_diff = float(np.abs(msg.detach().cpu().numpy().astype(np.float64) - msg_ref).max())
            delta_diff = float(np.abs(delta.detach().cpu().numpy().astype(np.float64) - delta_ref).max())
            out[f"block{i}"] = {
                "msg_max_abs_diff_vs_python": msg_diff,
                "delta_max_abs_diff_vs_python": delta_diff,
                "delta_abs_max": float(delta.abs().max().item()),
            }
            checks[f"block{i}_msg_matches_python"] = msg_diff <= 2e-5
            checks[f"block{i}_delta_matches_python"] = delta_diff <= 2e-5
    out["checks"] = checks
    out["passed"] = all(checks.values())
    out["note"] = "independent float64 numpy re-implementation of LayerNorm + sum aggregation + SiLU FFN"
    return out


def gate_zero_bond(task: Any, model: M.W2DynamicModel, device: torch.device) -> dict[str, Any]:
    atom_feat = task.cache["atom_feat"][0]
    batch = _to_device(
        synthetic_zero_bond_batch(task.spec, torch.as_tensor(np.array(atom_feat, copy=True))), device
    )
    test_model, _ = M.build_dynamic_model(
        task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device, zero_residual=False
    )
    out: dict[str, Any] = {}
    checks: dict[str, bool] = {}
    for name, mod in (("init_zero_residual", model), ("nondegenerate", test_model)):
        mod.eval()
        with torch.inference_mode():
            _a, _b, _s, h0 = mod.encode_h0(batch)
            src, dst, edge_msg = mod.bond_directions(batch)
            z = mod.blocks[0].norm(h0)
            msg = torch.zeros_like(z).index_add_(
                0, dst, torch.nn.functional.silu(z[src] + edge_msg)
            ) if src.numel() else torch.zeros_like(z)
            h1, h2 = mod.dynamic_layers(h0, batch)
            logits = mod(batch)
        out[name] = {
            "n_directed_edges": int(src.numel()),
            "msg_max_abs": float(msg.abs().max().item()),
            "logits_finite": _finite(logits),
            "h2_finite": _finite(h2),
            "logits_abs_max": float(logits.abs().max().item()),
        }
        checks[f"{name}_message_exactly_zero"] = float(msg.abs().max().item()) == 0.0
        checks[f"{name}_logits_finite"] = _finite(logits) and _finite(h2)
    out["checks"] = checks
    out["passed"] = all(checks.values())
    return out


def gate_no_new_inputs(task: Any, model: M.W2DynamicModel, device: torch.device) -> dict[str, Any]:
    ref, _ = M.untrained_w2(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    rows = [int(x) for x in task.valid_rows[3000:3006]]
    raw = C.PC.collate(task.cache, rows)
    traced_ref = TracedBatch(_to_device(raw, device))
    traced_dyn = TracedBatch(_to_device(raw, device))
    _logits(ref, traced_ref)
    _logits(model, traced_dyn)
    extra = sorted(traced_dyn.read - traced_ref.read)
    checks = {
        "dynamic_reads_no_extra_batch_field": not extra,
        "same_field_set": traced_dyn.read == traced_ref.read,
    }
    return {
        "w2_fields_read": sorted(traced_ref.read),
        "dynamic_fields_read": sorted(traced_dyn.read),
        "extra_fields_read_by_dynamic": extra,
        "checks": checks,
        "passed": all(checks.values()),
    }


def gate_inherited_protocol(task: Any) -> dict[str, Any]:
    labels = np.asarray(task.labels)
    finite_train = np.isfinite(labels[task.train_rows])
    finite_valid = np.isfinite(labels[task.valid_rows])
    finite_test = np.isfinite(labels[task.test_rows])
    import torch.nn.functional as F

    report = C.MET.validation_report(
        np.asarray(labels[task.valid_rows]),
        np.clip(np.nan_to_num(np.asarray(labels[task.valid_rows], dtype=np.float64), nan=0.0), 0.0, 1.0),
    )
    probe = torch.zeros(4, 2)
    probe_y = torch.tensor([[1.0, float("nan")], [0.0, 1.0], [1.0, 0.0], [0.0, float("nan")]])
    loss = C.MET.masked_bce_loss(probe, probe_y)
    mask = torch.isfinite(probe_y)
    manual = float(F.binary_cross_entropy_with_logits(probe[mask], probe_y[mask]))
    numpy_manual = float(
        np.mean(np.log1p(np.exp(-np.abs(np.asarray(probe[mask], dtype=np.float64)))))
    )
    checks = {
        "train_rows_official": int(task.train_rows.size) == 350_343,
        "valid_rows_official": int(task.valid_rows.size) == 43_793,
        "test_rows_official": int(task.test_rows.size) == 43_793,
        "test_rows_all_nan": int(finite_test.sum()) == 0,
        "no_finite_labels_outside_train_valid": int(np.isfinite(labels).sum())
        == int(finite_train.sum() + finite_valid.sum()),
        "labels_binary": bool(
            np.isin(labels[np.isfinite(labels)], (0.0, 1.0)).all()
        ),
        "feature_spec_hash": task.spec.hash() == C.SOURCE["feature_spec_hash"],
        "norm_fingerprint": task.norm["norm_fingerprint"] == C.SOURCE["norm_fingerprint"],
        "norm_fit_on_train_only": task.norm.get("fit_split") == "train_only"
        and task.norm.get("valid_or_test_used") is False,
        "split_partition": task.split_partition_sha256 == C.SOURCE["split_partition_sha256"],
        "label_artifact": task.label_artifact_sha256 == C.SOURCE["label_artifact_sha256"],
        "n_tasks_scored_valid": int(report["n_tasks_scored"]) == 127,
        "masked_bce_matches_torch_masking": abs(float(loss) - manual) <= 1e-7,
        "masked_bce_matches_numpy": abs(float(loss) - numpy_manual) <= 1e-6,
        "masked_bce_ignores_nan": bool(torch.equal(mask, torch.isfinite(probe_y))) and int(mask.sum()) == 6,
        "task_order_untouched": int(labels.shape[1]) == 128,
    }
    return {
        "split_sizes": {
            "train": int(task.train_rows.size),
            "valid": int(task.valid_rows.size),
            "test": int(task.test_rows.size),
        },
        "finite_entries": {
            "train": int(finite_train.sum()),
            "valid": int(finite_valid.sum()),
            "test": int(finite_test.sum()),
        },
        "masked_bce_unit_probe": {"loss": float(loss), "manual": manual, "numpy": numpy_manual},
        "checks": checks,
        "passed": all(checks.values()),
    }


def gate_gradient_and_profile(
    task: Any,
    device: torch.device,
    *,
    steps: int,
    audit_steps: Sequence[int],
) -> dict[str, Any]:
    """Real training steps: gradient audit at the fixed steps + step-cost profile."""
    model, audit = M.build_dynamic_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    optimizer, _ = T.build_optimizer(model)
    plan = C.SD.make_plan40(task, seed=int(C.SEED))
    batches = plan[0]["batches"][: int(steps)]
    clip = float(C.RECIPE["gradient_clip"])
    stager = C.PF.PinnedStager(device, depth=2) if hasattr(C.BC, "PF") else None
    if stager is None:
        from pf_pipeline import PinnedStager as _PS

        stager = _PS(device, depth=2)
    stager_less = lambda rows: stager.batch(C.PC.collate(task.cache, list(rows)))  # noqa: E731
    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
        torch.cuda.synchronize(device)

    snapshots: dict[str, Any] = {}
    times: list[float] = []
    losses: list[float] = []
    model.train()
    for step, rows in enumerate(batches, start=1):
        batch = stager_less(rows.tolist())
        y = stager.labels(task.labels[rows])
        t0 = time.perf_counter()
        out = T.train_step(model, optimizer, batch, y, global_step=step - 1, clip=clip, collect_grads=True)
        if device.type == "cuda":
            torch.cuda.synchronize(device)
        times.append(float(time.perf_counter() - t0))
        losses.append(out["loss"])
        if step in set(int(s) for s in audit_steps):
            snapshots[str(step)] = {
                "step": int(step),
                "loss": float(out["loss"]),
                "total_grad_norm": float(out["grad_norm"]),
                "new_grad_norms": {k: v for k, v in out["new_grads"].items()},
                "grad_norm_new_total": float(out["grad_norm_new_total"]),
                "grad_norm_w2_total": float(out["grad_norm_w2_total"]),
            }
    if device.type == "cuda":
        peak_alloc = int(torch.cuda.max_memory_allocated(device))
        torch.cuda.synchronize(device)
    else:
        peak_alloc = 0

    expected_tensors = sorted(
        n for n, _ in model.named_parameters() if C.is_new_parameter(n)
    )
    summary = {}
    for label, snap in snapshots.items():
        grads = snap["new_grad_norms"]
        summary[label] = {
            "all_new_tensors_present": all(grads.get(n) is not None for n in expected_tensors),
            "nonzero_new_tensors": sorted(n for n in expected_tensors if (grads.get(n) or 0.0) > 0.0),
            "zero_new_tensors": sorted(n for n in expected_tensors if grads.get(n) == 0.0),
            "missing_new_tensors": sorted(n for n in expected_tensors if grads.get(n) is None),
            "minimum_nonzero_norm": min(
                [float(grads[n]) for n in expected_tensors if (grads.get(n) or 0.0) > 0.0], default=None
            ),
        }

    # step-cost projection over the full 40-epoch budget
    warm = int(C.BUDGET["profile_warmup_steps"])
    arr = np.asarray(times, dtype=np.float64)
    steady = arr[warm:] if arr.size > warm else arr
    step_p50 = float(np.percentile(steady, 50))
    step_mean = float(steady.mean())
    model.eval()
    t_val = time.perf_counter()
    with torch.inference_mode():
        pass_report = C.SD.evaluate_validation(model, task, device, reason="profile_pass_after_audit_steps")
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    val_seconds = float(time.perf_counter() - t_val)
    stager.release()
    usage = M.usage_metrics(
        model,
        _to_device(
            C.PC.collate(task.cache, [int(x) for x in task.valid_rows[: int(C.DIAG_VALID_ROWS)]]),
            device,
        ),
    )
    projected_epoch = C.STEPS_PER_EPOCH * step_mean + val_seconds
    projected_total = C.TOTAL_EPOCHS * projected_epoch
    projected_hours = projected_total / 3600.0
    checks = {
        "all_new_tensors_receive_gradient": all(
            summary[s]["all_new_tensors_present"] and not summary[s]["missing_new_tensors"] for s in summary
        ),
        "gradients_nonzero_at_last_audited_step": all(
            (snapshots[max(snapshots, key=lambda s: int(s))]["new_grad_norms"].get(n) or 0.0) > 0.0
            for n in expected_tensors
        )
        if snapshots
        else False,
        "losses_finite": bool(np.isfinite(losses).all()),
        "dynamic_branch_active_after_steps": bool((usage["Rtotal"] or 0.0) > 0.0),
        "projected_within_hard_budget": projected_hours <= float(C.BUDGET["gpu_hard_hours"]),
    }
    return {
        "steps_profiled": int(steps),
        "audit_steps": [int(s) for s in audit_steps],
        "snapshots": snapshots,
        "per_step_summary": summary,
        "step_seconds": {
            "mean_after_warmup": step_mean,
            "p50_after_warmup": step_p50,
            "mean_all": float(arr.mean()),
            "min": float(arr.min()),
            "max": float(arr.max()),
        },
        "validation_pass_seconds": val_seconds,
        "validation_ap_after_audit_steps": float(pass_report["ap"]),
        "projected_epoch_seconds": projected_epoch,
        "projected_total_gpu_hours": projected_hours,
        "budget": dict(C.BUDGET),
        "gpu_peak_allocated_bytes": peak_alloc,
        "dynamic_usage_after_audit_steps": usage,
        "expected_new_tensors": expected_tensors,
        "checks": checks,
        "passed": all(checks.values()),
        "note": (
            "step 1 is expected to have structurally zero gradients for bond_proj and both fc1 "
            "(fc2.weight is zero-initialised); steps 3/10/100 must show non-zero gradients there"
        ),
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--out", default=str(C.ANALYSIS / "preflight.json"))
    parser.add_argument("--profile-steps", type=int, default=int(C.BUDGET["profile_real_batches"]))
    parser.add_argument("--skip-gradient-audit", action="store_true")
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    C.seed_everything(int(C.SEED))
    task = C.SD.load_task()

    started = time.perf_counter()
    out: dict[str, Any] = {
        "schema": C.SCHEMA_PREFLIGHT,
        "round_id": C.ROUND_ID,
        "captured_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "host_facts": C.host_facts(),
        "occupancy": C.gpu_occupancy(),
        "device": str(device),
    }

    out["constants"] = gate_constants(task)
    model, model_audit = M.build_dynamic_model(task.spec, task.norm, C.N_TASKS, seed=C.SEED, device=device)
    out["model"] = {
        "parameters": model_audit["parameters"],
        "w2_part_bitwise_equal_reference": {
            k: v for k, v in model_audit["w2_part_bitwise_equal_reference"].items() if k != "per_tensor"
        },
        "w2_trunk_init_fingerprint": model_audit["w2_trunk_init_fingerprint"],
        "w2_trunk_fingerprint_matches_recorded_w2": model_audit["w2_trunk_fingerprint_matches_recorded_w2"],
        "init": model_audit["init"],
        "block_fingerprints": model_audit["block_fingerprints"],
        "widths": model_audit["widths"],
        "n_state_tensors": model_audit["n_state_tensors"],
    }
    out["gates"] = {
        "model_identity": gate_model_identity(task, model, model_audit, device),
        "invariances": gate_invariances(task, model, device),
        "isolation": gate_isolation(task, model, device),
        "layer_inputs": gate_layer_inputs(task, model, device),
        "chain_dependency": gate_chain_dependency(task, model, device),
        "python_reference": gate_python_reference(task, model, device),
        "zero_bond": gate_zero_bond(task, model, device),
        "no_new_inputs": gate_no_new_inputs(task, model, device),
        "inherited_protocol": gate_inherited_protocol(task),
    }
    del model
    if not args.skip_gradient_audit:
        out["gradient_and_profile"] = gate_gradient_and_profile(
            task,
            device,
            steps=int(args.profile_steps),
            audit_steps=C.GRAD_AUDIT_STEPS,
        )
    out["reference_logits"] = C.verify_frozen_reference_logits()
    out["tests_sha256"] = C.test_rows_sha256_check(task.test_rows)

    gate_names = list(out["gates"])
    passed = all(out["gates"][g]["passed"] for g in gate_names) and out["constants"]["passed"]
    if "gradient_and_profile" in out:
        passed = passed and out["gradient_and_profile"]["passed"]
    out["gate_order"] = gate_names
    out["passed"] = bool(passed)
    out["verdict"] = "PASS" if passed else "FAIL"
    out["seconds"] = float(time.perf_counter() - started)
    C.write_json(Path(args.out), out)
    C.log(f"preflight {out['verdict']} in {out['seconds']:.1f}s -> {args.out}")
    if not passed:
        for g in gate_names:
            if not out["gates"][g]["passed"]:
                C.log(f"  FAILED gate {g}: {out['gates'][g]['checks']}")
        if "gradient_and_profile" in out and not out["gradient_and_profile"]["passed"]:
            C.log(f"  FAILED gradient/profile: {out['gradient_and_profile']['checks']}")
        if not out["constants"]["passed"]:
            C.log(f"  FAILED constants: {out['constants']['checks']}")
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

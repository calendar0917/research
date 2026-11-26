"""Scratch builder for ``STRUCT_DROP_SCRATCH``.

The initial parameters and buffers are **exactly** the official DSCR FIXED1
seed-0 constructor state.  Concretely:

1. ``DM.build_ds_model(..., arm="FIXED1", seed=0)`` is the historical constructor
   (it builds the frozen W2 base, loads the recorded W2 seed-0 state, then fills
   the new modules from the historical private generator ``seed*1_000_003+17``
   with zeroed ``fc2``).  Its state fingerprint at that moment is the number the
   historical FIXED1 run recorded as ``model_audit.initial_state_sha256``.
2. ``SM.StructuralDropModel`` is the inherited FIXED1 subclass whose *training*
   aggregation masks real structural units.  It is constructed inside
   ``torch.random.fork_rng`` (so it cannot perturb the ambient RNG) and then has
   the reference state copied in with ``strict=True``.

No parameter is added, removed, re-initialised or re-scaled: the wrapper only
adds a buffer-free mask generator and counters.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch

import ss_common as R

C = R.C
DM = R.DM
SM = R.SM


def build_fixed1_reference(task: Any, device: torch.device | str = "cpu") -> tuple[torch.nn.Module, dict[str, Any]]:
    """The historical FIXED1 seed-0 construction (read-only reference)."""
    model, audit = DM.build_ds_model(
        task.spec, task.norm, C.N_TASKS, arm="FIXED1", seed=R.SEED, device=device
    )
    return model, audit


def build_scratch_model(task: Any, device: torch.device | str = "cuda") -> tuple[SM.StructuralDropModel, dict[str, Any]]:
    """FIXED1 seed-0 state inside the inherited STRUCT_DROP wrapper (unseen by training)."""
    reference, ref_audit = build_fixed1_reference(task, device="cpu")
    ref_state = C.state_cpu(reference)
    ref_fingerprint = C.state_fingerprint(ref_state)
    ref_new_fingerprint = C.state_fingerprint(C.state_subset(ref_state, new=True))

    rng_before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        model = SM.StructuralDropModel(
            task.spec,
            task.norm,
            C.N_TASKS,
            dict(C.BC.config_for_scale(2)),
            struct_drop_q=R.Q,
            mask_seed=R.MASK_SEED,
        )
    rng_after = torch.get_rng_state().clone()
    if not torch.equal(rng_before, rng_after):
        raise RuntimeError("StructuralDropModel construction advanced the ambient torch RNG")

    load = model.load_state_dict(ref_state, strict=True)
    if load.missing_keys or load.unexpected_keys:
        raise RuntimeError(f"state transfer mismatch: {load}")

    ours = C.state_cpu(model)
    equal = C.compare_states(ours, ref_state)
    if not equal["all_equal"]:
        raise RuntimeError(f"STRUCT_DROP_SCRATCH state is not bitwise identical to FIXED1: {equal}")
    ours_fingerprint = C.state_fingerprint(ours)
    ours_new_fingerprint = C.state_fingerprint(C.state_subset(ours, new=True))
    if ours_fingerprint != R.EXPECTED_INITIAL_STATE_SHA256:
        raise RuntimeError(
            f"initial state {ours_fingerprint} != historical FIXED1 {R.EXPECTED_INITIAL_STATE_SHA256}"
        )
    if ref_fingerprint != ours_fingerprint:
        raise RuntimeError("reference / scratch fingerprints disagree")
    if ref_new_fingerprint != R.EXPECTED_NEW_STATE_SHA256:
        raise RuntimeError(f"new-module state {ref_new_fingerprint} != recorded {R.EXPECTED_NEW_STATE_SHA256}")

    model.to(torch.device(device))
    audit = C.parameter_audit(model)
    if not audit["all_match"]:
        raise RuntimeError(f"parameter audit failed: {audit['checks']}")
    if int(audit["total"]) != R.PARAMS_TOTAL:
        raise RuntimeError(f"parameters {audit['total']} != {R.PARAMS_TOTAL}")

    identity = {
        "arm": R.ARM,
        "structural_dropout": {
            "q": float(R.Q),
            "mask_seed": int(R.MASK_SEED),
            "grain": "physical undirected bond, one draw per training batch",
            "blocks": [0, 1],
            "scale": "1/(1-q) inverted dropout",
            "active_from": "training step 0",
        },
        "parameters": audit,
        "initial_state_sha256": ours_fingerprint,
        "initial_new_state_sha256": ours_new_fingerprint,
        "matches_historical_fixed1_initial_state": ours_fingerprint == R.EXPECTED_INITIAL_STATE_SHA256,
        "bitwise_equal_reference": equal,
        "n_state_tensors": int(len(ours)),
        "n_parameter_tensors": int(sum(1 for _ in model.parameters())),
        "reference_audit": {
            "init": ref_audit.get("init"),
            "w2_trunk_init_fingerprint": ref_audit.get("w2_trunk_init_fingerprint"),
            "w2_part_bitwise_equal_reference": ref_audit.get("w2_part_bitwise_equal_reference"),
            "w2_trunk_fingerprint_matches_recorded_w2_seed0": ref_audit.get(
                "w2_trunk_fingerprint_matches_recorded_w2_seed0"
            ),
        },
        "construction_rng_isolated": True,
    }
    if int(len(ours)) != R.N_STATE_TENSORS:
        raise RuntimeError(f"state tensors {len(ours)} != {R.N_STATE_TENSORS}")
    if int(identity["n_parameter_tensors"]) != R.N_PARAMETER_TENSORS:
        raise RuntimeError(f"parameter tensors {identity['n_parameter_tensors']} != {R.N_PARAMETER_TENSORS}")
    return model, identity


def build_optimizer(model: torch.nn.Module) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    return R.DT.build_optimizer(model)


def optimizer_is_fresh(optimizer: torch.optim.Optimizer) -> dict[str, Any]:
    """From-scratch proof: no AdamW moments exist before the first ``step()``."""
    n_with_state = sum(1 for st in optimizer.state.values() if st)
    return {
        "state_entries": int(len(optimizer.state)),
        "tensors_with_any_state": int(n_with_state),
        "fresh": bool(len(optimizer.state) == 0),
        "lr_at_construction": [float(g["lr"]) for g in optimizer.param_groups],
        "n_param_groups": int(len(optimizer.param_groups)),
    }


__all__ = [
    "build_fixed1_reference",
    "build_optimizer",
    "build_scratch_model",
    "optimizer_is_fresh",
]

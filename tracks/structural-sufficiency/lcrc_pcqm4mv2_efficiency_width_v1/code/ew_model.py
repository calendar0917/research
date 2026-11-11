"""The LCRC computation graph at width scale 1 / 2 / 3, one output.

The graph is unchanged from the frozen pilot: one atom encoding, one bond
encoding, one ``local`` S1/S2 call, per-atom contribution ``f``, global
``rho``, four-head attention readout, one regression output, L1 on the
standardised gap.  The *only* thing that changes between the three arms is the
width table returned by the inherited ``bs_common.config_for_scale(scale)``.

Initialisation is the frozen rule, unchanged for every scale:
``torch.manual_seed(seed)`` -> build that scale's FULL trunk -> remap to C1
(drop the ``b_mean`` columns) -> load -> readout from the private generator
``seed*1_000_003+17`` with ``readout.fc2`` zeroed.

``build_model(scale=2)`` must reproduce ``pq_model.build_model`` bitwise; that
is asserted by ``tests/test_ew_round.py`` and by the GPU preflight, so widening
or narrowing cannot silently change the historical W2 forward semantics.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

import ew_common as C

import bs_model as BM
import feature_spec as FS

state_cpu = BM.state_cpu
optimizer_state_cpu = BM.optimizer_state_cpu
restore_rng_state = BM.restore_rng_state
permute_atoms = BM.permute_atoms
count_module_calls = BM.count_module_calls

#: Reuse the frozen prediction / LR helpers: the model interface is identical.
predict_z = C.PQ_MODEL.predict_z
set_lrs = C.PQ_MODEL.set_lrs


def model_cfg(scale: int) -> dict[str, Any]:
    scale = int(scale)
    if scale not in C.WIDTHS:
        raise KeyError(f"scale {scale} not in {sorted(C.WIDTHS)}")
    cfg = dict(C.WIDTHS[scale])
    expected = C.PQ.BS.config_for_scale(scale)
    if int(cfg["n_out"]) != int(C.N_OUT):
        raise RuntimeError(f"model cfg n_out {cfg['n_out']} != {C.N_OUT}")
    for key, value in expected.items():
        if key == "n_out":
            continue
        if int(cfg[key]) != int(value):
            raise RuntimeError(f"scale {scale} width {key} changed: {cfg[key]} != {value}")
    return cfg


def expected_parameters(spec: FS.FeatureSpec, scale: int) -> dict[str, Any]:
    return C.PQ.expected_parameter_count(
        model_cfg(scale),
        atom_feat_dim=spec.atom_feat_dim,
        bond_feat_dim=spec.bond_feat_dim,
        n_scalars=spec.n_scalars,
        global_dim=spec.global_dim,
        n_nbr_codes=spec.n_nbr_codes,
        n_path_codes=spec.n_path_codes,
    )


def build_model(
    spec: FS.FeatureSpec,
    norm: Mapping[str, Any],
    *,
    scale: int,
    seed: int = C.MODEL_SEED,
    device: torch.device | str = "cpu",
) -> tuple["BM.BSModel", dict[str, Any]]:
    """Build the scale's regression model with the frozen initialisation."""
    scale = int(scale)
    cfg = model_cfg(scale)
    device = torch.device(device)
    model = BM.BSModel(spec, norm, int(cfg["n_out"]), cfg)
    full_state = BM.scaled_full_state(spec, norm, int(cfg["n_out"]), cfg, int(seed))
    c1_state = BM.remap_full_to_c1(full_state, spec, cfg)
    c1_state["cont_mean"] = torch.zeros(0, dtype=torch.float32)
    c1_state["cont_scale"] = torch.ones(0, dtype=torch.float32)
    missing, unexpected = model.load_state_dict(c1_state, strict=False)
    expected_missing = {k for k in model.state_dict() if k.startswith("readout.")}
    if set(missing) != expected_missing or unexpected:
        raise RuntimeError(
            f"trunk init load mismatch: missing={sorted(missing)} "
            f"expected_readout={sorted(expected_missing)} unexpected={sorted(unexpected)}"
        )
    base_keys = {k for k in model.state_dict() if not k.startswith("readout.")}
    trunk_before = C.PQ.BS.state_fingerprint(
        {k: v for k, v in model.state_dict().items() if k in base_keys}
    )
    init_audit = BM.initialize_readout(model, seed=int(seed), zero_residual=True)
    trunk_after = C.PQ.BS.state_fingerprint(
        {k: v for k, v in model.state_dict().items() if k in base_keys}
    )
    if trunk_after != trunk_before:
        raise RuntimeError("readout initialisation changed the trunk state")
    model.to(device)

    analytic = expected_parameters(spec, scale)
    total = int(sum(p.numel() for p in model.parameters()))
    base_elements = int(
        sum(p.numel() for n, p in model.named_parameters() if not n.startswith("readout."))
    )
    readout_elements = int(
        sum(p.numel() for n, p in model.named_parameters() if n.startswith("readout."))
    )
    if total != int(analytic["total_parameters"]):
        raise RuntimeError(f"scale {scale}: parameter count {total} != analytic {analytic['total_parameters']}")
    if base_elements != int(analytic["base_parameters"]):
        raise RuntimeError(f"scale {scale}: trunk parameters {base_elements} != analytic {analytic['base_parameters']}")
    if readout_elements != int(analytic["readout_parameters"]):
        raise RuntimeError(
            f"scale {scale}: readout parameters {readout_elements} != analytic {analytic['readout_parameters']}"
        )
    if total != int(C.EXPECTED_PARAMETERS[scale]):
        raise RuntimeError(f"scale {scale}: built {total} != pre-registered {C.EXPECTED_PARAMETERS[scale]}")

    widths = BM.width_audit(model, cfg)
    if not widths["all_match"]:
        failed = [k for k, v in widths["checks"].items() if not v]
        raise RuntimeError(f"scale {scale} width audit failed: {failed}")
    out_shape = tuple(model.readout.fc2.weight.shape)
    if out_shape != (int(C.N_OUT), int(cfg["pool_hidden"])):
        raise RuntimeError(f"readout output shape {out_shape} is not ({C.N_OUT}, {cfg['pool_hidden']})")

    audit: dict[str, Any] = {
        "model": f"W{scale}",
        "model_name": C.MODEL_NAMES[scale],
        "scale": int(scale),
        "seed": int(seed),
        "init_rule": C.PQ.INIT_SEED_RULE,
        "init": init_audit,
        "widths": cfg,
        "n_outputs": int(C.N_OUT),
        "readout_output_shape": list(out_shape),
        "trunk_init_fingerprint": trunk_before,
        "trunk_init_fingerprint_after_readout_init": trunk_after,
        "full_trunk_init_fingerprint": C.PQ.BS.state_fingerprint(full_state),
        "state_fingerprint": C.PQ.BS.state_fingerprint(model.state_dict()),
        "state_keys": sorted(model.state_dict()),
        "n_state_tensors": len(model.state_dict()),
        "parameter_tensors": len(list(model.parameters())),
        "base_parameters": base_elements,
        "readout_parameters": readout_elements,
        "total_parameters": total,
        "parameter_accounting": analytic,
        "parameter_check": "exact (analytic == built, n_out=1)",
        "width_audit": widths,
        "local_input_dim": int(widths["observed"]["local_input"]["observed"]),
        "rho_input_dim": int(widths["observed"]["rho_input"]),
        "dropout": float(model.drop.p),
    }
    return model, audit


def make_optimizer(model: nn.Module) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    """AdamW with the frozen recipe; identical for every width."""
    params = [p for p in model.parameters() if p.requires_grad]
    optimizer = torch.optim.AdamW(
        params,
        lr=float(C.RECIPE["warmup_start_lr"]),
        betas=tuple(C.RECIPE["adam_betas"]),
        eps=float(C.RECIPE["adam_eps"]),
        weight_decay=float(C.RECIPE["weight_decay"]),
    )
    audit = {
        "optimizer": "AdamW",
        "mode": "from_scratch",
        "n_param_groups": len(optimizer.param_groups),
        "n_parameters": int(sum(p.numel() for p in params)),
        "n_tensors": int(len(params)),
        "peak_lr": float(C.RECIPE["learning_rate"]),
        "warmup_start_lr": float(C.RECIPE["warmup_start_lr"]),
        "end_lr": float(C.RECIPE["end_lr"]),
        "betas": list(C.RECIPE["adam_betas"]),
        "eps": float(C.RECIPE["adam_eps"]),
        "weight_decay": float(C.RECIPE["weight_decay"]),
        "scheduler": "per_step cosine with 1-epoch linear warmup (horizon = 40 x 7,813 steps)",
    }
    return optimizer, audit


def historical_w2_equivalence(
    spec: FS.FeatureSpec, norm: Mapping[str, Any], *, seed: int = C.MODEL_SEED
) -> dict[str, Any]:
    """Prove ``build_model(scale=2)`` is the historical W2 builder bitwise."""
    with torch.random.fork_rng(devices=[]):
        frozen, frozen_audit = C.PQ_MODEL.build_model(spec, norm, seed=int(seed), device="cpu")
    with torch.random.fork_rng(devices=[]):
        ours, our_audit = build_model(spec, norm, scale=2, seed=int(seed), device="cpu")
    frozen_state = {k: v.detach().cpu().clone() for k, v in frozen.state_dict().items()}
    our_state = {k: v.detach().cpu().clone() for k, v in ours.state_dict().items()}
    if sorted(frozen_state) != sorted(our_state):
        raise RuntimeError("state key sets differ between the frozen and scale builders")
    max_delta = 0.0
    mismatch: list[str] = []
    for key in sorted(frozen_state):
        a, b = frozen_state[key], our_state[key]
        if a.shape != b.shape or a.dtype != b.dtype:
            mismatch.append(key)
            continue
        max_delta = max(max_delta, float((a.to(torch.float64) - b.to(torch.float64)).abs().max()) if a.numel() else 0.0)
    same = (
        not mismatch
        and int(frozen_audit["total_parameters"]) == int(our_audit["total_parameters"])
        and C.PQ.BS.state_fingerprint(frozen_state) == C.PQ.BS.state_fingerprint(our_state)
    )
    return {
        "builder_a": "pq_model.build_model (frozen W2, seed 0, n_out=1)",
        "builder_b": "ew_model.build_model(scale=2, seed 0, n_out=1)",
        "parameters_a": int(frozen_audit["total_parameters"]),
        "parameters_b": int(our_audit["total_parameters"]),
        "state_fingerprint_a": C.PQ.BS.state_fingerprint(frozen_state),
        "state_fingerprint_b": C.PQ.BS.state_fingerprint(our_state),
        "shape_or_dtype_mismatch": mismatch,
        "max_abs_delta": max_delta,
        "bitwise_identical": bool(same and max_delta == 0.0),
        "global_rng_untouched": True,
    }


__all__ = [
    "build_model",
    "count_module_calls",
    "expected_parameters",
    "historical_w2_equivalence",
    "make_optimizer",
    "model_cfg",
    "optimizer_state_cpu",
    "permute_atoms",
    "predict_z",
    "restore_rng_state",
    "set_lrs",
    "state_cpu",
]

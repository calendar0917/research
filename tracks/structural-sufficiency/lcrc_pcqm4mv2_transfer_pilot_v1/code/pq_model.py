"""The migrated LCRC-W2 regression model: same graph, one output value.

Nothing about the computation changes.  ``BSModel`` (which *is* the historical
``TULCRC(variant="C1")`` trunk plus ``nmp_model.AttentionPoolReadout``, built
with the W2 widths taken explicitly from ``bs_common.config_for_scale(2)``) is
constructed with ``n_out = 1`` instead of 128.  The initialisation is the frozen
one: ``torch.manual_seed(0)`` -> build the FULL trunk -> remap to C1 (drop the
``b_mean`` columns) -> load -> readout from the private generator
``seed*1_000_003+17`` with ``readout.fc2`` zeroed.

The graph is exactly LCRC-A: one atom encoding, one bond encoding, one
``local`` ``S1/S2`` call, then ``h -> f(h)`` summed per molecule + ``rho`` of
(global, mean h, mean h^2, log1p n) + the four-head attention readout.  There is
no second pass along ``bond_index`` and no hidden-state message passing.

The three migrated quantities are:

* the output width (``n_out = 1``),
* the loss (L1 on the standardised gap, in ``pq_train``),
* the parameter accounting (recomputed analytically in ``pq_common``; the
  built module is asserted against it).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn

import pq_common as C

import bs_common as BS
import bs_model as BM
import feature_spec as FS

state_cpu = BM.state_cpu
optimizer_state_cpu = BM.optimizer_state_cpu
restore_rng_state = BM.restore_rng_state
permute_atoms = BM.permute_atoms
count_module_calls = BM.count_module_calls


def model_cfg() -> dict[str, Any]:
    cfg = dict(C.MODEL_CFG)
    if int(cfg["n_out"]) != int(C.N_OUT):
        raise RuntimeError(f"model cfg n_out {cfg['n_out']} != {C.N_OUT}")
    expected = BS.config_for_scale(C.WIDTH_AUDIT_SCALE)
    for key, value in expected.items():
        if key == "n_out":
            continue
        if int(cfg[key]) != int(value):
            raise RuntimeError(f"W2 width {key} changed: {cfg[key]} != {value}")
    return cfg


def expected_parameters(spec: FS.FeatureSpec, cfg: Mapping[str, Any]) -> dict[str, Any]:
    return C.expected_parameter_count(
        cfg,
        atom_feat_dim=spec.atom_feat_dim,
        bond_feat_dim=spec.bond_feat_dim,
        n_scalars=spec.n_scalars,
        global_dim=spec.global_dim,
        n_nbr_codes=spec.n_nbr_codes,
        n_path_codes=spec.n_path_codes,
    )


def historical_replay(spec: FS.FeatureSpec, norm: Mapping[str, Any], *, seed: int) -> dict[str, Any]:
    """Prove the inherited builder still produces the historical 994,324-param W2.

    Built at ``n_out = 128`` on CPU inside a forked RNG context, so neither the
    global torch stream nor any CUDA stream is touched.
    """
    with torch.random.fork_rng(devices=[]):
        model, audit = BM.build_model(
            C.MODEL_KEY, spec, norm, int(BS.N_TASKS), seed=int(seed), device="cpu", verify_parameters=True
        )
    widths = audit.get("widths") or {}
    return {
        "builder": "bs_model.build_model('W2', n_out=128, seed=0)",
        "parameters": int(audit["total_parameters"]),
        "expected_historical_parameters": int(BS.EXPECTED_PARAMETERS[2]),
        "matches_historical": int(audit["total_parameters"]) == int(BS.EXPECTED_PARAMETERS[2]),
        "base_parameters": int(audit["base_parameters"]),
        "readout_parameters": int(audit["readout_parameters"]),
        "widths": {k: int(v) for k, v in widths.items()},
        "global_rng_untouched": True,
    }


def build_model(
    spec: FS.FeatureSpec,
    norm: Mapping[str, Any],
    *,
    seed: int = C.MODEL_SEED,
    device: torch.device | str = "cpu",
) -> tuple["BM.BSModel", dict[str, Any]]:
    """Build the W2 regression model with the frozen initialisation."""
    cfg = model_cfg()
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
    trunk_before = BS.state_fingerprint({k: v for k, v in model.state_dict().items() if k in base_keys})
    init_audit = BM.initialize_readout(model, seed=int(seed), zero_residual=True)
    trunk_after = BS.state_fingerprint({k: v for k, v in model.state_dict().items() if k in base_keys})
    if trunk_after != trunk_before:
        raise RuntimeError("readout initialisation changed the trunk state")
    model.to(device)

    analytic = expected_parameters(spec, cfg)
    total = int(sum(p.numel() for p in model.parameters()))
    base_elements = int(
        sum(p.numel() for n, p in model.named_parameters() if not n.startswith("readout."))
    )
    readout_elements = int(sum(p.numel() for n, p in model.named_parameters() if n.startswith("readout.")))
    if total != int(analytic["total_parameters"]):
        raise RuntimeError(f"parameter count {total} != analytic {analytic['total_parameters']}")
    if base_elements != int(analytic["base_parameters"]):
        raise RuntimeError(f"trunk parameters {base_elements} != analytic {analytic['base_parameters']}")
    if readout_elements != int(analytic["readout_parameters"]):
        raise RuntimeError(f"readout parameters {readout_elements} != analytic {analytic['readout_parameters']}")

    widths = BM.width_audit(model, cfg)
    if not widths["all_match"]:
        failed = [k for k, v in widths["checks"].items() if not v]
        raise RuntimeError(f"width audit failed: {failed}")
    out_shape = tuple(model.readout.fc2.weight.shape)
    if out_shape != (int(C.N_OUT), int(cfg["pool_hidden"])):
        raise RuntimeError(f"readout output shape {out_shape} is not ({C.N_OUT}, {cfg['pool_hidden']})")

    audit: dict[str, Any] = {
        "model": C.MODEL_KEY,
        "model_name": C.MODEL_NAME,
        "scale": int(C.WIDTH_AUDIT_SCALE),
        "seed": int(seed),
        "init_rule": C.INIT_SEED_RULE,
        "init": init_audit,
        "widths": cfg,
        "n_outputs": int(C.N_OUT),
        "readout_output_shape": list(out_shape),
        "trunk_init_fingerprint": trunk_before,
        "trunk_init_fingerprint_after_readout_init": trunk_after,
        "full_trunk_init_fingerprint": BS.state_fingerprint(full_state),
        "state_fingerprint": BS.state_fingerprint(model.state_dict()),
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
        "scheduler": "per_step cosine with 1-epoch linear warmup (horizon = run rows)",
    }
    return optimizer, audit


def set_lrs(optimizer: torch.optim.Optimizer, *, step: int, steps_epoch: int, steps_total: int) -> float:
    lr = C.scheduled_lr(int(step), steps_epoch=int(steps_epoch), steps_total=int(steps_total))
    for group in optimizer.param_groups:
        group["lr"] = lr
    return float(lr)


def predict_z(model: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    """One forward, returning ``(B,)`` standardised predictions."""
    out = model(batch)
    if not torch.is_tensor(out):
        raise RuntimeError("model returned a non-tensor")
    if out.dim() != 2 or int(out.shape[1]) != int(C.N_OUT):
        raise RuntimeError(f"model output shape {tuple(out.shape)} is not (B, {C.N_OUT})")
    return out[:, 0]


__all__ = [
    "build_model",
    "count_module_calls",
    "expected_parameters",
    "historical_replay",
    "make_optimizer",
    "model_cfg",
    "optimizer_state_cpu",
    "permute_atoms",
    "predict_z",
    "restore_rng_state",
    "set_lrs",
    "state_cpu",
]

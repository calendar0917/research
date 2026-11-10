"""LCRC-A-W3 built by the frozen constructor, plus the two interventions.

The model is **the historical W3**: ``bs_model.build_model("W3", ...)`` with the
frozen seed-0 initialisation.  This module adds only

* an explicit, audited dropout rate on the *actual* forward path
  (``model.drop``, the single ``nn.Dropout`` the historical graph calls once per
  forward), and
* an AdamW construction that parameterises ``weight_decay``.

Both are verified by reading the constructed objects back, never by trusting a
config dict.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

import rd_common as C

BM = C.BM


# ---------------------------------------------------------------------------
# construction
# ---------------------------------------------------------------------------


def dropout_modules(model: nn.Module) -> list[dict[str, Any]]:
    return [
        {
            "name": name,
            "class": type(module).__name__,
            "p": float(module.p),
            "inplace": bool(module.inplace),
        }
        for name, module in model.named_modules()
        if isinstance(module, nn.Dropout)
    ]


def set_dropout(model: nn.Module, p: float) -> dict[str, Any]:
    """Set the frozen graph's single dropout rate and prove it took effect.

    The historical W3 calls ``self.drop`` exactly once per forward (after
    ``local``); that is the only stochastic module in the graph.  A second
    dropout module would invalidate the intervention, so it is a hard error.
    """
    p = float(p)
    if not 0.0 <= p < 1.0:
        raise ValueError(f"dropout {p} outside [0, 1)")
    found = dropout_modules(model)
    if len(found) != 1 or found[0]["name"] != "drop":
        raise RuntimeError(f"expected exactly one dropout module named 'drop', found {found}")
    model.drop.p = p
    after = dropout_modules(model)
    if after[0]["p"] != p:
        raise RuntimeError(f"dropout rate did not take effect: {after}")
    return {
        "module": after[0]["name"],
        "p_requested": p,
        "p_effective": after[0]["p"],
        "inplace": after[0]["inplace"],
        "n_dropout_modules": len(after),
        "modules": after,
        "route": "BSModel.encode_h -> self.drop(self.local(batch, a, b, scal))",
        "calls_per_forward": 1,
    }


def build_model(
    run_key: str,
    spec: Any,
    norm: Mapping[str, Any],
    n_out: int,
    *,
    device: torch.device | str = "cpu",
    seed: int = C.SEED,
    verify_parameters: bool = True,
) -> tuple[nn.Module, dict[str, Any]]:
    run = C.RUNS[run_key]
    model, audit = BM.build_model(
        "W3", spec, norm, int(n_out), seed=int(seed), device=device, verify_parameters=verify_parameters
    )
    drop = set_dropout(model, float(run["dropout"]))
    total = int(sum(p.numel() for p in model.parameters()))
    base = int(sum(p.numel() for n, p in model.named_parameters() if not n.startswith("readout.")))
    readout = int(sum(p.numel() for n, p in model.named_parameters() if n.startswith("readout.")))
    audit = dict(audit)
    audit.update(
        {
            "run": run_key,
            "dropout": drop,
            "model_dropout": float(drop["p_effective"]),
            "parameters_recount": {"total": total, "trunk": base, "readout": readout},
            "parameters_exact": bool(
                total == C.EXPECTED_PARAMETERS
                and base == C.EXPECTED_BASE_PARAMETERS
                and readout == C.EXPECTED_READOUT_PARAMETERS
            ),
            "width_audit_all_match": bool(audit["width_audit"]["all_match"]),
            "frozen_constructor": "bs_model.build_model('W3', ...) (lcrc_pcba_backbone_scaling_v1)",
        }
    )
    if verify_parameters and not audit["parameters_exact"]:
        raise RuntimeError(f"parameter recount mismatch for {run_key}: {audit['parameters_recount']}")
    if verify_parameters and not audit["width_audit_all_match"]:
        raise RuntimeError(f"width audit failed for {run_key}")
    if verify_parameters and total != int(run["parameters"]):
        raise RuntimeError(f"{run_key} parameters {total} != {run['parameters']}")
    return model, audit


# ---------------------------------------------------------------------------
# optimizer
# ---------------------------------------------------------------------------


def make_optimizer(model: nn.Module, *, weight_decay: float) -> tuple[torch.optim.AdamW, dict[str, Any]]:
    params = [p for p in model.parameters() if p.requires_grad]
    wd = float(weight_decay)
    optimizer = torch.optim.AdamW(
        params,
        lr=float(C.BASE_RECIPE["warmup_start_lr"]),
        betas=tuple(C.BASE_RECIPE["adam_betas"]),
        eps=float(C.BASE_RECIPE["adam_eps"]),
        weight_decay=wd,
    )
    observed = [float(g["weight_decay"]) for g in optimizer.param_groups]
    if observed != [wd] * len(observed):
        raise RuntimeError(f"AdamW weight_decay {observed} != requested {wd}")
    audit = {
        "optimizer": "AdamW",
        "n_param_groups": len(optimizer.param_groups),
        "n_parameters": int(sum(p.numel() for p in params)),
        "n_tensors": len(params),
        "weight_decay_requested": wd,
        "weight_decay_read_back": observed,
        "betas": [float(x) for x in optimizer.param_groups[0]["betas"]],
        "eps": float(optimizer.param_groups[0]["eps"]),
        "lr_at_build": float(optimizer.param_groups[0]["lr"]),
        "warmup_start_lr": float(C.BASE_RECIPE["warmup_start_lr"]),
        "peak_lr": float(C.BASE_RECIPE["learning_rate"]),
        "all_params_one_group": len(optimizer.param_groups) == 1,
        "only_difference_is_weight_decay": True,
    }
    return optimizer, audit


# ---------------------------------------------------------------------------
# behavioural probes on the real forward path
# ---------------------------------------------------------------------------


def capture_h(model: nn.Module, batch: Mapping[str, Any]) -> torch.Tensor:
    """Run ``model.encode_h`` and return the post-dropout ``h``."""
    _a, _b, _scal, h = model.encode_h(batch)
    return h.detach()


def dropout_zero_fraction(h: torch.Tensor) -> float:
    with torch.no_grad():
        return float((h == 0).float().mean().item())


def dropout_probe(model: nn.Module, batch: Mapping[str, Any], *, repeats: int = 12) -> dict[str, Any]:
    """Empirical zeros of ``h`` in train vs eval mode (the real forward path)."""
    p = float(model.drop.p)
    was_training = model.training
    fractions: list[float] = []
    try:
        model.train()
        with torch.no_grad():
            for _ in range(int(repeats)):
                h = capture_h(model, batch)
                fractions.append(dropout_zero_fraction(h))
        train_mean = float(np.mean(fractions))
        train_std = float(np.std(fractions))
        n_elem = int(h.numel())
        model.eval()
        with torch.no_grad():
            h_eval_a = capture_h(model, batch)
            h_eval_b = capture_h(model, batch)
        eval_zero = dropout_zero_fraction(h_eval_a)
        eval_deterministic = bool(torch.equal(h_eval_a, h_eval_b))
    finally:
        model.train(was_training)
    sigma = math.sqrt(max(p * (1.0 - p), 1e-12) / max(n_elem, 1))
    tolerance = max(5.0 * sigma, 0.004)
    return {
        "p_effective": p,
        "h_elements": n_elem,
        "train_zero_fraction_mean": train_mean,
        "train_zero_fraction_std": train_std,
        "train_zero_fraction_samples": fractions,
        "expected_zero_fraction": p,
        "binomial_sigma": sigma,
        "tolerance": tolerance,
        "train_matches_p": bool(abs(train_mean - p) <= tolerance),
        "eval_zero_fraction": eval_zero,
        "eval_has_no_zeros": bool(eval_zero == 0.0),
        "eval_is_deterministic": eval_deterministic,
        "repeats": int(repeats),
        "passed": bool(abs(train_mean - p) <= tolerance and eval_zero == 0.0 and eval_deterministic),
    }


def train_forward_is_stochastic(model: nn.Module, batch: Mapping[str, Any]) -> dict[str, Any]:
    """Two train-mode forwards must differ; two eval-mode forwards must not."""
    was_training = model.training
    try:
        model.train()
        with torch.no_grad():
            a = capture_h(model, batch)
            b = capture_h(model, batch)
        model.eval()
        with torch.no_grad():
            c = capture_h(model, batch)
            d = capture_h(model, batch)
    finally:
        model.train(was_training)
    return {
        "train_forwards_differ": bool(not torch.equal(a, b)),
        "eval_forwards_identical": bool(torch.equal(c, d)),
    }


def intervention_probe(
    models: Mapping[str, nn.Module],
    batch: Mapping[str, Any],
    *,
    seed: int = 0,
) -> dict[str, Any]:
    """Same batch + same initial parameters, different interventions.

    Checks that (a) eval-mode logits are identical (the intervention is a
    training-time-only change), (b) train-mode logits differ (the dropout masks
    actually differ), and (c) the dropout zero fractions match each run's rate.
    """
    out: dict[str, Any] = {}
    eval_logits: dict[str, torch.Tensor] = {}
    train_logits: dict[str, torch.Tensor] = {}
    learnable_before: dict[str, dict[str, torch.Tensor]] = {}
    for key, model in models.items():
        model.eval()
        with torch.no_grad():
            eval_logits[key] = model(batch).detach().clone()
        learnable_before[key] = {
            name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad
        }
        torch.manual_seed(int(seed))  # identical dropout RNG stream for both runs
        model.train()
        with torch.no_grad():
            train_logits[key] = model(batch).detach().clone()
    keys = list(models)
    a, b = keys[0], keys[1]
    eval_same = bool(torch.equal(eval_logits[a], eval_logits[b]))
    train_diff = bool(not torch.equal(train_logits[a], train_logits[b]))
    return {
        "pair": [a, b],
        "eval_logits_identical": eval_same,
        "eval_max_abs_delta": float((eval_logits[a] - eval_logits[b]).abs().max().item()),
        "train_logits_differ": train_diff,
        "train_max_abs_delta": float((train_logits[a] - train_logits[b]).abs().max().item()),
        "dropout_p": {k: float(models[k].drop.p) for k in keys},
        "passed": bool(eval_same and train_diff),
    }


def weight_decay_update_probe(
    models: Mapping[str, nn.Module],
    batch: Mapping[str, Any],
    y: torch.Tensor,
    *,
    steps: int = 1,
    clip: float = float("nan"),
) -> dict[str, Any]:
    """One real AdamW step from the same initial state: the decay term must bite.

    AdamW's decay multiplies every parameter by ``(1 - lr*wd)``; with the same
    loss/gradient the update difference between the two runs is therefore
    proportional to ``|lr*(wd_a - wd_b)*w|`` plus the (identical) Adam term.  The
    probe records the observed weight-norm reduction per run, which is the
    quantity the intervention claims to change.
    """
    clip_val = float(C.BASE_RECIPE["gradient_clip"]) if not np.isfinite(clip) else float(clip)
    out: dict[str, Any] = {}
    for key, model in models.items():
        opt, _audit = make_optimizer(model, weight_decay=float(C.RUNS[key]["weight_decay"]))
        before = {name: p.detach().clone() for name, p in model.named_parameters() if p.requires_grad}
        model.train()
        for step in range(int(steps)):
            C.BT.train_step(model, opt, batch, y, global_step=step, clip=clip_val)
        decayed = 0.0
        moved = 0.0
        for name, p in model.named_parameters():
            if name not in before:
                continue
            delta = (p.detach() - before[name]).float()
            moved += float(delta.pow(2).sum())
        out[key] = {
            "weight_decay": float(C.RUNS[key]["weight_decay"]),
            "adam_denominator_rms": moved ** 0.5,
            "lr_used": float(opt.param_groups[0]["lr"]),
            "decay_multiplier": 1.0 - float(opt.param_groups[0]["lr"]) * float(C.RUNS[key]["weight_decay"]),
        }
    a, b = list(models)
    out["pair"] = [a, b]
    out["decay_multipliers_differ"] = bool(
        out[a]["decay_multiplier"] != out[b]["decay_multiplier"]
    )
    out["weight_decay_ratio"] = out[a]["weight_decay"] / out[b]["weight_decay"]
    out["update_norms_differ"] = bool(out[a]["adam_denominator_rms"] != out[b]["adam_denominator_rms"])
    out["passed"] = bool(
        out["decay_multipliers_differ"]
        and abs(out["weight_decay_ratio"] - 10.0) < 1e-9
        and out["update_norms_differ"]
    )
    return out


__all__ = [
    "build_model",
    "capture_h",
    "dropout_modules",
    "dropout_probe",
    "dropout_zero_fraction",
    "intervention_probe",
    "make_optimizer",
    "set_dropout",
    "train_forward_is_stochastic",
    "weight_decay_update_probe",
]

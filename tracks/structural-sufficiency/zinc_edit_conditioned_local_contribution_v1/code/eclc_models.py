"""ECLC-v1 bodies: frozen LCRC-A backbone + one restricted additive base head.

Forward:

    base_v = base_features(batch, zero_r2=(arm == "REAL"))    337 dims
    a, b, h = backbone._encode(batch)
    y = sum_v f_base(base_v) + sum_v f_context(h_v)
        + rho(global33, mean h, mean h^2, log(1+|V|))

`f_context` and `rho` are the frozen LCRC-A modules.  `f_base` is a small MLP
whose last layer is zero-initialised.  REAL zeroes the 253 radius-2 dims of the
base input; NULL keeps them.  Both arms have identical modules, parameters and
initial tensors; only the forward-time mask differs.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import eclc_common as E

L = E.L
M = E.M


class BaseHead(nn.Module):
    """Small per-atom base-contribution head (337 -> 32 -> 1)."""

    def __init__(self, in_dim: int = E.BASE_INPUT_DIM, hidden: int = 32):
        super().__init__()
        self.in_dim = int(in_dim)
        self.hidden = int(hidden)
        self.lins = nn.ModuleList([nn.Linear(self.in_dim, self.hidden),
                                   nn.Linear(self.hidden, 1)])
        nn.init.zeros_(self.lins[1].weight)
        nn.init.zeros_(self.lins[1].bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.lins[0](x)
        x = F.silu(x)
        return self.lins[1](x)


def base_features(batch: Mapping[str, torch.Tensor], *, zero_r2: bool) -> torch.Tensor:
    """Deterministic radius-1 (+ radius-2) count representation.

    The first 84 dims are the original atom type (21) and the rooted radius-1
    typed neighbour counts (63).  The remaining 253 dims are radius-2 typed
    path / end / closure counts.  REAL sets them to exactly zero.
    """
    n = int(batch["atom_type"].shape[0])
    dtype = torch.float32
    device = batch["atom_type"].device
    center = F.one_hot(batch["atom_type"].long(), E.BASE_CENTER_DIM).to(dtype)

    r1 = torch.zeros((n, E.BASE_R1_DIM), dtype=dtype, device=device)
    r1.index_add_(0, batch["nbr_atom"].long(),
                  F.one_hot(batch["nbr_code"].long(), E.BASE_R1_DIM).to(dtype))

    r2p = torch.zeros((n, E.BASE_R2_PATH_DIM), dtype=dtype, device=device)
    r2p.index_add_(0, batch["path_atom"].long(),
                   F.one_hot(batch["path_cat"].long(), E.BASE_R2_PATH_DIM).to(dtype))

    r2e = torch.zeros((n, E.BASE_R2_END_DIM), dtype=dtype, device=device)
    r2e.index_add_(0, batch["path_atom"].long(),
                   F.one_hot(batch["path_end63"].long(), E.BASE_R2_END_DIM).to(dtype))

    r2c = torch.zeros((n, E.BASE_R2_CLOSE_DIM), dtype=dtype, device=device)
    r2c.index_add_(0, batch["path_atom"].long(),
                   batch["path_closure"].to(dtype).unsqueeze(1))

    x = torch.cat([center, torch.log1p(r1), torch.log1p(r2p),
                   torch.log1p(r2e), torch.log1p(r2c)], dim=1)
    if zero_r2:
        x = torch.cat([x[:, :E.BASE_R1_TOTAL],
                       torch.zeros_like(x[:, E.BASE_R1_TOTAL:])], dim=1)
    if x.shape[1] != E.BASE_INPUT_DIM:
        raise RuntimeError(f"base feature dim {x.shape[1]} != {E.BASE_INPUT_DIM}")
    return x


class ECLC(nn.Module):
    """Frozen LCRC-A body with an explicit base/context split."""

    def __init__(self, arm: str, norm: Mapping[str, Any],
                 cfg_backbone: Mapping[str, Any] | None = None,
                 cfg_base: Mapping[str, Any] | None = None):
        super().__init__()
        if arm not in E.ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        self.backbone = M.LCRC("A", norm, cfg_backbone)
        self.base: BaseHead | None = None
        self.base_cfg = {"in_dim": E.BASE_INPUT_DIM, "hidden": 32}
        if cfg_base:
            self.base_cfg.update(dict(cfg_base))
        self.cfg = {
            "backbone": dict(self.backbone.cfg),
            "base": dict(self.base_cfg),
            "arm": arm,
            "base_input_dim": E.BASE_INPUT_DIM,
            "base_r1_total": E.BASE_R1_TOTAL,
            "real_zeroes_radius2": True,
        }

    def add_base(self) -> None:
        if self.base is not None:
            raise RuntimeError("base head already built")
        self.base = BaseHead(**self.base_cfg)

    def _encode(self, batch: Mapping[str, torch.Tensor]):
        return self.backbone._encode(batch)

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        if self.base is None:
            raise RuntimeError("f_base is not built")
        n_graphs = int(batch["n_graphs"])
        a, b, h = self._encode(batch)
        atom_batch = batch["atom_batch"]
        width = h.shape[1]
        count = h.new_zeros(n_graphs).index_add_(
            0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype)
        )
        mean_h = h.new_zeros((n_graphs, width)).index_add_(
            0, atom_batch, h
        ) / count.clamp(min=1.0).unsqueeze(1)
        mean_h2 = h.new_zeros((n_graphs, width)).index_add_(
            0, atom_batch, h * h
        ) / count.clamp(min=1.0).unsqueeze(1)

        context_atom = self.backbone.f(h).view(-1)
        context = h.new_zeros(n_graphs).index_add_(0, atom_batch, context_atom)

        base_in = base_features(batch, zero_r2=(self.arm == "REAL"))
        base_atom = self.base(base_in).view(-1)
        base = h.new_zeros(n_graphs).index_add_(0, atom_batch, base_atom)

        g = (batch["global"] - self.backbone.global_mean) / self.backbone.global_scale
        rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
        rho = self.backbone.rho(rho_in).view(-1)
        y = context + base + rho

        if not return_aux:
            return y
        aux = {
            "context": context,
            "base": base,
            "rho": rho,
            "context_atom": context_atom,
            "base_atom": base_atom,
            "base_input": base_in,
            "mean_h": mean_h,
            "mean_h2": mean_h2,
            "h": h,
            "count": count,
        }
        return y, aux


def _param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def module_breakdown(model: ECLC) -> dict[str, int]:
    out = {
        "atom_encoder": _param_count(model.backbone.atom_encoder),
        "bond_encoder": _param_count(model.backbone.bond_encoder),
        "local": _param_count(model.backbone.local),
        "f_context": _param_count(model.backbone.f),
        "rho": _param_count(model.backbone.rho),
        "f_base": 0 if model.base is None else _param_count(model.base),
    }
    out["backbone"] = _param_count(model.backbone)
    out["total"] = _param_count(model)
    return out


def ssl_path(seed: int):
    return L.P3_SSL[0] if int(seed) == 0 else L.P3_SSL[1]


def build_model(arm: str, norm: Mapping[str, Any], seed: int, device,
                cfg_backbone: Mapping[str, Any] | None = None,
                cfg_base: Mapping[str, Any] | None = None) -> tuple[ECLC, dict[str, Any]]:
    """Build the frozen LCRC-A backbone, copy it, then add f_base.

    The frozen LCRC-A builder is called first, so the shared tensors are
    bit-identical to the historical reference at the same seed.  The base head
    is built afterwards under a fixed second seed, so REAL and NULL have
    identical full initial states and no arm-specific RNG consumption.
    """
    legacy, legacy_info = M.build_model("A", norm, seed, device, cfg_backbone)
    model = ECLC(arm, norm, cfg_backbone, cfg_base)
    model.backbone.load_state_dict(legacy.state_dict())
    torch.manual_seed(int(seed) + 90731)
    model.add_base()
    model.to(device)
    info = {
        "arm": arm,
        "seed": int(seed),
        "backbone": legacy_info,
        "ssl": legacy_info["ssl"],
        "parameters": _param_count(model),
        "parameters_breakdown": module_breakdown(model),
        "initial_state_sha256": L.C.state_hash(model),
        "base_last_layer_zero": bool(
            torch.count_nonzero(model.base.lins[1].weight).item() == 0
            and torch.count_nonzero(model.base.lins[1].bias).item() == 0
        ),
        "has_relation": model.backbone.rel is not None,
        "has_junction_module": any("junction" in n for n, _ in model.named_modules()),
        "base_input_dim": int(model.base.in_dim),
        "base_hidden": int(model.base.hidden),
    }
    if info["parameters"] > 500_000:
        raise RuntimeError(f"{arm}: {info['parameters']} parameters exceeds 500k")
    return model, info


def load_model_from_state(arm: str, norm: Mapping[str, Any], state: Mapping[str, torch.Tensor],
                          device, cfg_backbone: Mapping[str, Any] | None = None,
                          cfg_base: Mapping[str, Any] | None = None) -> ECLC:
    model, _ = build_model(arm, norm, seed=0, device=device,
                           cfg_backbone=cfg_backbone, cfg_base=cfg_base)
    model.load_state_dict(state)
    model.to(device)
    return model


__all__ = ["BaseHead", "ECLC", "base_features", "build_model",
           "load_model_from_state", "module_breakdown"]

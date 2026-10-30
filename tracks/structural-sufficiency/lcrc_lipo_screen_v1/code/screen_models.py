"""Frozen model builders, the S2-off ablation and the GINE reference.

Arm A is exactly the first-round ``lipo_v1`` LCRC-A model (same class, same
config, same 247,106 parameters, no SSL weights).

Arm B instantiates the *same* class and the *same* seed, then arms a frozen
forward pre-hook on the first LocalMLP linear layer.  The hook zeroes the
200-dimensional concatenated S2 block of that input:

    [ a_v(96) | b_sum(64) | b_mean(64) | S1joint(32) | S2(104) | S2content(96) | scalars(9) ]
      ^--------- untouched -------------------^        ^-- zeroed --^

Nothing in the frozen model source is modified, and the zeroing is active in
both training and evaluation.  Arm B therefore has the same tensors and the
same nominal parameter count but a smaller *effective* parameter count; the
report discloses this rather than treating B as an information-matched control.

Arm C is a standard 3-layer GINE on the same atom51/bond13 features and the
real physical-bond adjacency, with the same graph-level readout form as LCRC
(per-atom sum + ``rho(global42, mean h, mean h^2, log1p(n_atoms))``).  Sharing
the readout form means A vs C isolates the local structural encoder; it is a
practical performance reference, not an information-matched experiment.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv

import screen_common as C

prep_model = C.prep_model


# ---------------------------------------------------------------------------
# S2-off controller
# ---------------------------------------------------------------------------


class S2OffController:
    """Zero the S2 block of the LocalMLP input (frozen, train and eval)."""

    def __init__(self, model: prep_model.LCRC):
        self.model = model
        cfg = model.cfg
        atom_out = int(cfg["atom_out"])
        rank = int(cfg["stat_rank"])
        # LocalContent concatenates [a, b_sum, b_mean, S1joint, S2, S2content, scalars];
        # a, b_sum and b_mean all have width atom_out (bond_out == atom_out).
        self.start = 3 * atom_out + rank
        self.path_block = 3 * rank + 8
        self.content_block = atom_out
        self.end = self.start + self.path_block + self.content_block
        self.in_dim = self.start + self.path_block + self.content_block + model.spec.n_scalars
        first = model.local.mlp.lins[0]
        if first.in_features != self.in_dim:
            raise ValueError(f"S2-off slice derived {self.in_dim} != LocalMLP input {first.in_features}")
        self.calls = 0
        self._handle = first.register_forward_pre_hook(self._hook)

    def _hook(self, module, args):
        if not args:
            raise RuntimeError("LocalMLP received no input")
        x = args[0]
        if x.shape[-1] != self.in_dim:
            raise RuntimeError(f"unexpected LocalMLP input width {tuple(x.shape)}")
        out = x.clone()
        out[..., self.start : self.end] = 0.0
        self.calls += 1
        return (out,) + tuple(args[1:])

    def remove(self) -> None:
        self._handle.remove()

    def slice(self) -> slice:
        return slice(self.start, self.end)


class LocalMLPInputCapture:
    """Capture the exact concatenated input of ``local.mlp.lins[0]``."""

    def __init__(self, model: prep_model.LCRC):
        self.model = model
        self.inputs: list[torch.Tensor] = []
        self._handle = model.local.mlp.lins[0].register_forward_pre_hook(self._hook, with_kwargs=False)

    def _hook(self, module, args):
        self.inputs.append(args[0].detach().clone())

    def close(self) -> list[torch.Tensor]:
        self._handle.remove()
        return self.inputs

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        self.close()
        return False


# ---------------------------------------------------------------------------
# LCRC-A / LCRC-A S2-off
# ---------------------------------------------------------------------------


def build_lcrc(
    spec,
    norm: Mapping[str, Any],
    seed: int,
    arm: str,
    device="cpu",
) -> tuple[prep_model.LCRC, dict[str, Any]]:
    if arm not in ("A", "B"):
        raise ValueError(f"LCRC arm must be A or B, got {arm!r}")
    torch.manual_seed(int(seed))
    model = prep_model.LCRC(spec, norm, prep_model.DEFAULT_CONFIG)
    controller = None
    if arm == "B":
        controller = S2OffController(model)
    model._s2_controller = controller  # audit handle; not part of the state dict
    model.to(device)
    info = lcrc_info(model, layer_arm=arm)
    if controller is not None:
        info["s2_off"] = {
            "start": controller.start,
            "end": controller.end,
            "path_block": controller.path_block,
            "content_block": controller.content_block,
            "localmlp_input_width": controller.in_dim,
            "active_train": True,
            "active_eval": True,
        }
        info["effective_parameters"] = effective_parameters_lcrc(model, controller)
        info["dead_parameters"] = sorted(dead_parameter_names_lcrc(model, controller))
    return model, info


def lcrc_info(model: prep_model.LCRC, *, layer_arm: str) -> dict[str, Any]:
    return {
        "arm": layer_arm,
        "model": "lcrc_a_lipo_v1",
        "schema": model.spec.name,
        "schema_hash": model.spec.hash(),
        "config": dict(model.cfg),
        "parameters": prep_model.parameter_count(model),
        "parameters_breakdown": prep_model.module_breakdown(model),
        "state_hash": state_hash(model),
        "input_widths": {
            "atom_feat": model.spec.atom_feat_dim,
            "bond_feat": model.spec.bond_feat_dim,
            "global": model.spec.global_dim,
            "scalars": model.spec.n_scalars,
            "n_nbr_codes": model.spec.n_nbr_codes,
            "n_path_codes": model.spec.n_path_codes,
        },
        "ssl_loaded": False,
    }


def dead_parameter_names_lcrc(model: prep_model.LCRC, controller: S2OffController) -> set[str]:
    names = {
        "local.path_emb.weight",
        "local.end_emb.weight",
        "local.path_u.weight",
        "local.path_v.weight",
        "local.joint_proj.weight",
        "local.joint_proj.bias",
        "local.close_emb.weight",
        "local.path_gate.weight",
    }
    return names


def effective_parameters_lcrc(model: prep_model.LCRC, controller: S2OffController) -> int:
    """Analytic effective capacity of S2-off: nominal minus the frozen-dead block."""
    dead = 0
    state = model.state_dict()
    for name in dead_parameter_names_lcrc(model, controller):
        dead += int(state[name].numel())
    # First-layer weight columns feeding the zeroed slice never change.
    dead += int(model.local.mlp.lins[0].weight[:, controller.start : controller.end].numel())
    return int(prep_model.parameter_count(model) - dead)


# ---------------------------------------------------------------------------
# GINE reference
# ---------------------------------------------------------------------------


def bidirectional_edge_index(bond_index: torch.Tensor) -> torch.Tensor:
    """[2, E] forward physical bonds -> [2, 2E] with both directions."""
    bi = bond_index.transpose(0, 1)
    return torch.cat([bi, bi.flip(0)], dim=1)


def _gin_mlp(hidden: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(hidden, hidden), nn.ReLU(), nn.Linear(hidden, hidden))


class GINERegressor(nn.Module):
    """3-layer GINE over the real physical-bond graph + shared global readout."""

    def __init__(
        self,
        spec,
        norm: Mapping[str, Any],
        hidden: int = C.GINE_HIDDEN,
        layers: int = C.GINE_LAYERS,
        rho_hidden: int = C.GINE_RHO_HIDDEN,
    ):
        super().__init__()
        self.spec = spec
        self.hidden = int(hidden)
        self.layers = int(layers)
        self.input_proj = nn.Linear(spec.atom_feat_dim, self.hidden)
        self.convs = nn.ModuleList(
            [
                GINEConv(_gin_mlp(self.hidden), edge_dim=spec.bond_feat_dim, train_eps=True)
                for _ in range(self.layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(self.hidden) for _ in range(self.layers)])
        self.f = prep_model.MLP([self.hidden, self.hidden, 1])
        self.rho = prep_model.MLP([spec.global_dim + 2 * self.hidden + 1, int(rho_hidden), 1])
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        n_graphs = int(batch["n_graphs"])
        h = self.input_proj(batch["atom_feat"])
        if batch["bond_index"].numel():
            edge_index = bidirectional_edge_index(batch["bond_index"])
            edge_attr = torch.cat([batch["bond_feat"], batch["bond_feat"]], dim=0)
        else:
            edge_index = batch["bond_index"].new_zeros((2, 0))
            edge_attr = batch["bond_feat"].new_zeros((0, batch["bond_feat"].shape[1]))
        for conv, norm in zip(self.convs, self.norms):
            h = norm(h + conv(h, edge_index, edge_attr))
        atom_batch = batch["atom_batch"]
        count = h.new_zeros(n_graphs).index_add_(0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype))
        mean_h = h.new_zeros((n_graphs, self.hidden)).index_add_(0, atom_batch, h) / count.clamp(min=1.0).unsqueeze(1)
        mean_h2 = (
            h.new_zeros((n_graphs, self.hidden)).index_add_(0, atom_batch, h * h)
            / count.clamp(min=1.0).unsqueeze(1)
        )
        contrib_atom = self.f(h).view(-1)
        contrib = h.new_zeros(n_graphs).index_add_(0, atom_batch, contrib_atom)
        g = (batch["global"] - self.global_mean) / self.global_scale
        rho = self.rho(torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)).view(-1)
        y = contrib + rho
        if not return_aux:
            return y
        return y, {
            "contrib": contrib,
            "rho": rho,
            "contrib_atom": contrib_atom,
            "mean_h": mean_h,
            "mean_h2": mean_h2,
            "h": h,
            "count": count,
            "delta": torch.zeros(n_graphs, device=y.device, dtype=y.dtype),
        }


def build_gine(spec, norm: Mapping[str, Any], seed: int, device="cpu") -> tuple[GINERegressor, dict[str, Any]]:
    torch.manual_seed(int(seed))
    model = GINERegressor(spec, norm)
    model.to(device)
    info = {
        "arm": "C",
        "model": "gine_lipo_v1",
        "schema": spec.name,
        "schema_hash": spec.hash(),
        "hidden": model.hidden,
        "layers": model.layers,
        "parameters": parameter_count(model),
        "parameters_breakdown": {
            "input_proj": count_module(model.input_proj),
            "convs": sum(count_module(c) for c in model.convs),
            "norms": sum(count_module(c) for c in model.norms),
            "f": count_module(model.f),
            "rho": count_module(model.rho),
        },
        "state_hash": state_hash(model),
        "input_widths": {
            "atom_feat": spec.atom_feat_dim,
            "bond_feat": spec.bond_feat_dim,
            "global": spec.global_dim,
        },
        "message_passing": "real_physical_bonds_bidirectional",
        "lcrc_s1_s2": False,
        "ssl_loaded": False,
    }
    return model, info


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def count_module(module: nn.Module) -> int:
    return parameter_count(module)


def state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def state_dict_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def build_model(arm: str, spec, norm: Mapping[str, Any], seed: int, device="cpu"):
    if arm in ("A", "B"):
        return build_lcrc(spec, norm, seed, arm, device)
    if arm == "C":
        return build_gine(spec, norm, seed, device)
    raise ValueError(f"unknown neural arm {arm!r}")


def soup_safety_report(model: nn.Module) -> dict[str, Any]:
    """Parameter averaging is only safe if every state entry is a float tensor."""
    bad = []
    for key, value in model.state_dict().items():
        if not torch.is_tensor(value):
            bad.append((key, type(value).__name__))
        elif not value.is_floating_point():
            bad.append((key, str(value.dtype)))
    return {
        "all_state_entries_float": not bad,
        "non_float_entries": bad,
        "n_state_entries": len(model.state_dict()),
        "soup_rule": "elementwise_float_mean_of_last_5_epoch_end_state_dicts",
    }


def average_states(states: list[Mapping[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
    if not states:
        raise ValueError("cannot average an empty state list")
    keys = list(states[0].keys())
    for s in states[1:]:
        if list(s.keys()) != keys:
            raise ValueError("state dict keys differ across soup members")
    out: dict[str, torch.Tensor] = {}
    for key in keys:
        stacked = torch.stack([s[key].detach().float().cpu() for s in states], dim=0)
        out[key] = stacked.mean(dim=0)
    return out


__all__ = [
    "S2OffController",
    "LocalMLPInputCapture",
    "GINERegressor",
    "bidirectional_edge_index",
    "build_lcrc",
    "build_gine",
    "build_model",
    "lcrc_info",
    "parameter_count",
    "state_hash",
    "state_dict_fingerprint",
    "effective_parameters_lcrc",
    "dead_parameter_names_lcrc",
    "soup_safety_report",
    "average_states",
]

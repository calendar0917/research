"""LCRC-A trunk plus the frozen bond-object residual (R) or mean-field null (N).

The trunk is the original ``lipo_v1`` LCRC-A model, unchanged and trainable.
The only new computation is a per-physical-bond object score.  R pairs the two
real endpoint contexts ``z_u * z_v``; N replaces that product with the exact
mean over all unordered pairs of stubs in the same (molecule, bond type) group.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import common as C

prep_model = C.prep_model

NEW_PARAM_BREAKDOWN = dict(C.NEW_PARAM_BREAKDOWN)


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def state_hash(module: nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(module.state_dict().items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def state_dict_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def truncate_trunk_state(state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Extract bare LCRC-A keys from a wrapper state dict."""
    prefix = "trunk."
    return {k[len(prefix):]: v for k, v in state.items() if k.startswith(prefix)}


def meanfield_pair_products(z: torch.Tensor, batch: Mapping[str, torch.Tensor], n_bond_types: int) -> torch.Tensor:
    """Exact mean of z_i * z_j over unordered stub pairs, per (graph, type).

    Each physical bond contributes two stubs.  For a group with ``M`` bonds
    (``S = 2M`` stubs) the returned vector is

        (sum_z * sum_z - sum_z2) / (S * (S - 1))

    with all operations elementwise.  Empty groups have no edge and are not
    indexed; the denominator clamp only protects those unused rows.
    """
    idx = batch["bond_index"]
    if idx.numel() == 0:
        return z.new_zeros((0, z.shape[1]))
    n_graphs = int(batch["n_graphs"])
    g = batch["bond_batch"].to(torch.long)
    t = batch["bond_type"].to(torch.long) - 1
    if int(t.min()) < 0 or int(t.max()) >= int(n_bond_types):
        raise ValueError("bond type outside the frozen vocabulary")
    group = g * int(n_bond_types) + t
    n_groups = n_graphs * int(n_bond_types)
    u, v = idx[:, 0], idx[:, 1]
    z_u, z_v = z[u], z[v]
    sum_z = z.new_zeros((n_groups, z.shape[1]))
    sum_z2 = z.new_zeros((n_groups, z.shape[1]))
    sum_z = sum_z.index_add_(0, group, z_u).index_add_(0, group, z_v)
    sum_z2 = sum_z2.index_add_(0, group, z_u * z_u).index_add_(0, group, z_v * z_v)
    count = z.new_zeros(n_groups).index_add_(0, group, torch.ones_like(group, dtype=z.dtype))
    s = 2.0 * count
    denom = (s * (s - 1.0)).clamp(min=1.0)
    j_group = (sum_z * sum_z - sum_z2) / denom.unsqueeze(1)
    return j_group[group]


class BondObjectRegressor(nn.Module):
    """Original LCRC-A + one trainable bond-object residual branch."""

    def __init__(self, trunk, mode: str):
        super().__init__()
        if mode not in ("R", "N"):
            raise ValueError(f"mode must be R or N, got {mode!r}")
        self.trunk = trunk
        self.mode = mode
        self.atom_proj = nn.Linear(64, 32)
        self.edge_lin1 = nn.Linear(128, 64)
        self.edge_lin2 = nn.Linear(64, 32)
        self.edge_score = nn.Linear(32, 1)
        # Frozen rule: the final score layer starts exactly at zero.
        nn.init.zeros_(self.edge_score.weight)
        nn.init.zeros_(self.edge_score.bias)

    def bond_object(
        self,
        batch: Mapping[str, torch.Tensor],
        h: torch.Tensor,
        b: torch.Tensor,
        *,
        mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        mode = self.mode if mode is None else mode
        if mode not in ("R", "N"):
            raise ValueError(f"mode must be R or N, got {mode!r}")
        idx = batch["bond_index"]
        z = F.silu(self.atom_proj(h))
        if idx.numel() == 0:
            j = z.new_zeros((0, z.shape[1]))
        elif mode == "R":
            j = z[idx[:, 0]] * z[idx[:, 1]]
        else:
            j = meanfield_pair_products(z, batch, int(self.trunk.spec.n_bond_types))
        edge_input = torch.cat([b, j], dim=1)
        h_e = F.silu(self.edge_lin1(edge_input))
        h_e = F.silu(self.edge_lin2(h_e))
        score = self.edge_score(h_e).view(-1)
        n_graphs = int(batch["n_graphs"])
        total = score.new_zeros(n_graphs).index_add_(0, batch["bond_batch"], score)
        nb = batch["bond_counts"].to(total.dtype)
        delta = total / torch.sqrt(torch.clamp(nb, min=1.0))
        return {"j": j, "edge_score": score, "delta_bond": delta, "z": z}

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False, mode: str | None = None):
        y_base, trunk_aux = self.trunk(batch, return_aux=True)
        obj = self.bond_object(batch, trunk_aux["h"], trunk_aux["b"], mode=mode)
        y = y_base + obj["delta_bond"]
        aux = {
            "y_base": y_base,
            "delta_bond": obj["delta_bond"],
            "edge_score": obj["edge_score"],
            "j": obj["j"],
            "trunk_aux": trunk_aux,
        }
        if return_aux:
            return y, aux
        return y


def build_model(arm: str, spec, norm: Mapping[str, Any], seed: int, device="cpu"):
    """Build the frozen trunk first (same RNG contract as historical A), then R/N."""
    if arm not in C.ARMS:
        raise ValueError(f"arm must be one of {C.ARMS}, got {arm!r}")
    torch.manual_seed(int(seed))
    trunk = prep_model.LCRC(spec, norm, prep_model.DEFAULT_CONFIG)
    trunk_hash = prep_model.state_hash(trunk)
    model = BondObjectRegressor(trunk, arm)
    model.to(device)
    total = parameter_count(model)
    trunk_params = parameter_count(model.trunk)
    new_params = total - trunk_params
    if trunk_params != C.LCRC_PARAMS_REFERENCE:
        raise RuntimeError(f"trunk parameter count {trunk_params} != {C.LCRC_PARAMS_REFERENCE}")
    if new_params != C.NEW_PARAMS:
        raise RuntimeError(f"new parameter count {new_params} != {C.NEW_PARAMS}")
    if total != C.TOTAL_PARAMS_REFERENCE:
        raise RuntimeError(f"total parameter count {total} != {C.TOTAL_PARAMS_REFERENCE}")
    return model, model_info(model, arm, seed, trunk_hash, device)


def model_info(model: BondObjectRegressor, arm: str, seed: int, trunk_hash: str, device) -> dict[str, Any]:
    total = parameter_count(model)
    trunk_params = parameter_count(model.trunk)
    new_params = total - trunk_params
    expected_trunk = C.EXPECTED_TRUNK_INIT_HASH.get(int(seed))
    return {
        "model": "lcrc_bond_object_pairing_v1",
        "arm": arm,
        "mode": arm,
        "seed": int(seed),
        "device": str(device),
        "schema": model.trunk.spec.name,
        "schema_hash": model.trunk.spec.hash(),
        "config": dict(model.trunk.cfg),
        "parameters": total,
        "parameters_trunk": trunk_params,
        "parameters_new": new_params,
        "parameters_new_breakdown": dict(NEW_PARAM_BREAKDOWN),
        "parameters_expected": {
            "trunk": C.LCRC_PARAMS_REFERENCE,
            "new": C.NEW_PARAMS,
            "total": C.TOTAL_PARAMS_REFERENCE,
        },
        "state_hash": state_hash(model),
        "trunk_state_hash": trunk_hash,
        "trunk_expected_hash": expected_trunk,
        "trunk_init_matches_historical": (expected_trunk is None) or (trunk_hash == expected_trunk),
        "input_widths": {
            "atom_feat": model.trunk.spec.atom_feat_dim,
            "bond_feat": model.trunk.spec.bond_feat_dim,
            "global": model.trunk.spec.global_dim,
            "scalars": model.trunk.spec.n_scalars,
            "n_nbr_codes": model.trunk.spec.n_nbr_codes,
            "n_path_codes": model.trunk.spec.n_path_codes,
            "n_bond_types": model.trunk.spec.n_bond_types,
        },
        "ssl_loaded": False,
        "edge_score_initialised_zero": True,
    }


def soup_safety_report(model: nn.Module) -> dict[str, Any]:
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
    "parameter_count",
    "state_hash",
    "state_dict_fingerprint",
    "truncate_trunk_state",
    "meanfield_pair_products",
    "BondObjectRegressor",
    "build_model",
    "model_info",
    "soup_safety_report",
    "average_states",
]

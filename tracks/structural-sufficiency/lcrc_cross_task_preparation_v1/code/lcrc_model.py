"""Schema-driven LCRC-A model (local contribution + global correction).

The computational graph is the frozen LCRC-A graph:

    a_v = AtomEncoder(atom_feat_v)
    b_e = BondEncoder([a_u+a_v, |a_u-a_v|, a_u*a_v, bond_feat_e])
    S1_v = sum_{e incident v} b_e,  S1mean_v = S1_v / deg(v)
    S1joint_v = sum_{(v,u)} Emb_nbr[(type(u), bond_uv)]
    S2_v = sum_{v->w->u, u != v} [Emb_path[(type(w),bt1,bt2)]
                                  + Emb_end[(type(u),bt2)]
                                  + W_j (U_path * V_end)
                                  + Emb_close[closure]
                                  + gate[(type(w),bt1,bt2)] * a_u]
    h_v = LayerNorm(LocalMLP([a_v, S1_v, S1mean_v, S1joint_v, S2_v, scalars_v]))
    y_hat = sum_v f(h_v) + rho(global, mean_v h_v, mean_v h_v^2, log1p(n_atoms))

There is exactly one local encode, no iterative hidden-state update, no
attention, no junction module and no relation head.  The only difference
between the two schemas is the input width and vocabulary size supplied by the
:class:`FeatureSpec`; module names and state-dict keys are kept identical so
the frozen ZINC LCRC-A checkpoint can be loaded strictly.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import lcrc_common as C
from feature_spec import FeatureSpec

DEFAULT_CONFIG = {
    "width": 64,
    "stat_rank": 32,
    "local_hidden": 192,
    "atom_hidden": 128,
    "atom_out": 96,
    "bond_hidden": 160,
    "bond_out": 96,
    "rho_hidden": 64,
}


class MLP(nn.Module):
    """Plain MLP with P3-compatible ``lins`` naming (SSL state-dict keys)."""

    def __init__(self, dims: list[int]):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.silu(x)
        return x


class LocalContent(nn.Module):
    """One-shot shared local content encoder (S1 joint + S2 typed paths)."""

    def __init__(self, spec: FeatureSpec, cfg: Mapping[str, Any]):
        super().__init__()
        self.spec = spec
        rank = int(cfg["stat_rank"])
        a_out = int(cfg["atom_out"])
        self.atom_out = a_out
        self.rank = rank
        self.nbr_emb = nn.Embedding(spec.n_nbr_codes, rank)
        self.path_emb = nn.Embedding(spec.n_path_codes, rank)
        self.end_emb = nn.Embedding(spec.n_nbr_codes, rank)
        self.path_u = nn.Embedding(spec.n_path_codes, rank)
        self.path_v = nn.Embedding(spec.n_nbr_codes, rank)
        self.joint_proj = nn.Linear(rank, rank)
        self.close_emb = nn.Embedding(2, 8)
        self.path_gate = nn.Embedding(spec.n_path_codes, 1)
        nn.init.ones_(self.path_gate.weight)
        in_dim = (
            3 * a_out
            + rank
            + 3 * rank
            + 8
            + a_out
            + spec.n_scalars
        )
        self.mlp = MLP([in_dim, int(cfg["local_hidden"]), int(cfg["width"])])
        self.norm = nn.LayerNorm(int(cfg["width"]))

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        a: torch.Tensor,
        b: torch.Tensor,
        scal: torch.Tensor,
    ) -> torch.Tensor:
        n_atoms = a.shape[0]
        width = a.shape[1]
        bond_index = batch["bond_index"]
        if bond_index.numel():
            u, v = bond_index[:, 0], bond_index[:, 1]
            degree = a.new_zeros(n_atoms)
            degree = degree.index_add_(0, u, torch.ones_like(u, dtype=a.dtype))
            degree = degree.index_add_(0, v, torch.ones_like(v, dtype=a.dtype))
            b_sum = a.new_zeros((n_atoms, width)).index_add_(0, u, b).index_add_(0, v, b)
        else:
            degree = a.new_zeros(n_atoms)
            b_sum = a.new_zeros((n_atoms, width))
        b_mean = b_sum / degree.clamp(min=1.0).unsqueeze(1)

        nbr = a.new_zeros((n_atoms, self.rank)).index_add_(
            0, batch["nbr_atom"], self.nbr_emb(batch["nbr_code"])
        ) if batch["nbr_code"].numel() else a.new_zeros((n_atoms, self.rank))

        if batch["path_cat"].numel():
            path_joint = self.joint_proj(
                self.path_u(batch["path_cat"]) * self.path_v(batch["path_end63"])
            )
            path_emb = torch.cat(
                [
                    self.path_emb(batch["path_cat"]),
                    self.end_emb(batch["path_end63"]),
                    path_joint,
                    self.close_emb(batch["path_closure"]),
                ],
                dim=1,
            )
            path_agg = a.new_zeros((n_atoms, path_emb.shape[1])).index_add_(
                0, batch["path_atom"], path_emb
            )
            gate = self.path_gate(batch["path_cat"])
            path_content = a[batch["path_end"]] * gate
            content_agg = a.new_zeros((n_atoms, width)).index_add_(
                0, batch["path_atom"], path_content
            )
        else:
            # Empty S2 must produce the exact same zero vector as an empty sum.
            path_agg = a.new_zeros((n_atoms, 3 * self.rank + 8))
            content_agg = a.new_zeros((n_atoms, width))

        x = torch.cat([a, b_sum, b_mean, nbr, path_agg, content_agg, scal], dim=1)
        return self.norm(self.mlp(x))


class LCRC(nn.Module):
    """LCRC-A: per-atom local contribution plus a global correction."""

    def __init__(self, spec: FeatureSpec, norm: Mapping[str, Any], cfg: Mapping[str, Any] | None = None):
        super().__init__()
        self.spec = spec
        self.cfg = dict(DEFAULT_CONFIG if cfg is None else {**DEFAULT_CONFIG, **dict(cfg)})
        a_out = int(self.cfg["atom_out"])
        width = int(self.cfg["width"])
        self.atom_encoder = MLP([spec.atom_feat_dim, int(self.cfg["atom_hidden"]), a_out])
        self.bond_encoder = MLP(
            [3 * a_out + spec.bond_feat_dim, int(self.cfg["bond_hidden"]), int(self.cfg["bond_out"])]
        )
        self.local = LocalContent(spec, self.cfg)
        self.f = MLP([width, width, 1])
        self.rho = MLP([spec.global_dim + 2 * width + 1, int(self.cfg["rho_hidden"]), 1])
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        self.register_buffer("scalar_mean", torch.as_tensor(norm["scalar_mean"], dtype=torch.float32))
        self.register_buffer("scalar_scale", torch.as_tensor(norm["scalar_scale"], dtype=torch.float32))
        for name, expected in (
            ("global_mean", spec.global_dim),
            ("global_scale", spec.global_dim),
            ("scalar_mean", spec.n_scalars),
            ("scalar_scale", spec.n_scalars),
        ):
            if tuple(getattr(self, name).shape) != (expected,):
                raise ValueError(f"norm {name} has shape {tuple(getattr(self, name).shape)} != {(expected,)}")

    def _encode(self, batch: Mapping[str, torch.Tensor]):
        a = self.atom_encoder(batch["atom_feat"])
        idx = batch["bond_index"]
        if idx.numel():
            au, av = a[idx[:, 0]], a[idx[:, 1]]
            bond_input = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        else:
            bond_input = torch.zeros((0, 3 * a.shape[1] + batch["bond_feat"].shape[1]), dtype=a.dtype, device=a.device)
        b = self.bond_encoder(bond_input)
        scal = (batch["atom_scalar"] - self.scalar_mean) / self.scalar_scale
        h = self.local(batch, a, b, scal)
        return a, b, h

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False):
        n_graphs = int(batch["n_graphs"])
        a, b, h = self._encode(batch)
        atom_batch = batch["atom_batch"]
        width = h.shape[1]
        count = h.new_zeros(n_graphs).index_add_(
            0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype)
        )
        mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / count.clamp(min=1.0).unsqueeze(1)
        mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / count.clamp(min=1.0).unsqueeze(1)
        contrib_atom = self.f(h).view(-1)
        contrib = h.new_zeros(n_graphs).index_add_(0, atom_batch, contrib_atom)
        g = (batch["global"] - self.global_mean) / self.global_scale
        rho_in = torch.cat([g, mean_h, mean_h2, torch.log1p(count).unsqueeze(1)], dim=1)
        rho = self.rho(rho_in).view(-1)
        y = contrib + rho
        aux = {
            "contrib": contrib,
            "rho": rho,
            "contrib_atom": contrib_atom,
            "mean_h": mean_h,
            "mean_h2": mean_h2,
            "h": h,
            "count": count,
            "delta": torch.zeros(n_graphs, device=y.device, dtype=y.dtype),
            "a": a,
            "b": b,
        }
        if return_aux:
            return y, aux
        return y


# ---------------------------------------------------------------------------
# parameter / state audit
# ---------------------------------------------------------------------------


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def module_breakdown(model: LCRC) -> dict[str, int]:
    out = {
        "atom_encoder": parameter_count(model.atom_encoder),
        "bond_encoder": parameter_count(model.bond_encoder),
        "local": parameter_count(model.local),
        "f": parameter_count(model.f),
        "rho": parameter_count(model.rho),
    }
    out["total"] = parameter_count(model)
    return out


def state_hash(model: nn.Module) -> str:
    h = hashlib.sha256()
    for key, value in sorted(model.state_dict().items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def _legacy_ssl_path(seed: int) -> Path:
    root = C.REPO / "tracks/structure-contribution/results/bjoc_full10k_pretraining_transfer_v1"
    return root / ("pretrain_encoder_s0.pt" if int(seed) == 0 else "pretrain_encoder_s1.pt")


def load_legacy_ssl_encoders(model: LCRC, seed: int) -> dict[str, Any]:
    if model.spec.name != "legacy_zinc":
        raise PermissionError("ZINC SSL encoders may only be loaded into the legacy_zinc schema")
    path = _legacy_ssl_path(int(seed))
    if not path.exists():
        raise FileNotFoundError(path)
    state = torch.load(path, map_location="cpu", weights_only=False)
    if "atom_encoder" not in state or "bond_encoder" not in state:
        raise ValueError("SSL checkpoint lacks atom_encoder/bond_encoder")
    model.atom_encoder.load_state_dict(state["atom_encoder"])
    model.bond_encoder.load_state_dict(state["bond_encoder"])
    return {
        "path": C.track_rel(path),
        "sha256": C.sha256_file(path),
        "loaded_modules": ["atom_encoder", "bond_encoder"],
        "schema": "legacy_zinc",
    }


def build_model(
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    seed: int,
    device,
    cfg: Mapping[str, Any] | None = None,
    *,
    load_ssl: bool = False,
) -> tuple[LCRC, dict[str, Any]]:
    """Deterministic CPU/GPU construction with schema-derived input widths."""
    torch.manual_seed(int(seed))
    model = LCRC(spec, norm, cfg)
    ssl_info = None
    if load_ssl:
        ssl_info = load_legacy_ssl_encoders(model, int(seed))
    model.to(device)
    info = {
        "schema": spec.name,
        "schema_hash": spec.hash(),
        "seed": int(seed),
        "device": str(device),
        "config": dict(model.cfg),
        "parameters": parameter_count(model),
        "parameters_breakdown": module_breakdown(model),
        "state_hash": state_hash(model),
        "ssl": ssl_info,
        "has_relation_head": False,
        "has_junction_module": any("junction" in n for n, _ in model.named_modules()),
        "has_attention": any("attention" in n for n, _ in model.named_modules()),
        "input_widths": {
            "atom_feat": spec.atom_feat_dim,
            "bond_feat": spec.bond_feat_dim,
            "global": spec.global_dim,
            "scalars": spec.n_scalars,
            "n_nbr_codes": spec.n_nbr_codes,
            "n_path_codes": spec.n_path_codes,
        },
    }
    return model, info


def load_model_from_state(
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    state: Mapping[str, torch.Tensor],
    device,
    cfg: Mapping[str, Any] | None = None,
) -> LCRC:
    model = LCRC(spec, norm, cfg)
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    model.to(device)
    return model


__all__ = [
    "DEFAULT_CONFIG",
    "MLP",
    "LocalContent",
    "LCRC",
    "parameter_count",
    "module_breakdown",
    "state_hash",
    "load_legacy_ssl_encoders",
    "build_model",
    "load_model_from_state",
]

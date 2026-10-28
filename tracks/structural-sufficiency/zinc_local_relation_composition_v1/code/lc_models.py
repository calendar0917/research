"""LCRC-v1 bodies: A (local contribution), R (real relation), N (mean-field).

Forward computation (no learned junction token, no iterative state update):

    a_v = atom_encoder(atom_feat_v)                         [shared, SSL init]
    b_e = bond_encoder([a_u+a_v, |a_u-a_v|, a_u*a_v, bond]) [shared, SSL init]
    S1_v = sum_{e incident v} b_e  and its degree-normalised mean
    S1joint_v = sum_{e=(v,u)} Emb_nbr[type(u), type(bond)]
    S2_v = sum_{2-paths v->w->u} Emb_path[type(w),bt1,bt2]
                               + Emb_end[type(u),bt2] + Emb_close[closure]
           + sum g[type(w),bt1,bt2] * a_u
    h_v = LN(MLP([a_v, S1_v, mean_e b_e, S1joint_v, S2_v, scalars_v]))  in R^64

    y_A = sum_v f(h_v) + rho(global33, mean_v h, mean_v h^2, log(1+|V|))

R/N add one static relation composition on the SAME h:

    z_ij = phi([h_i+h_j, |h_i-h_j|, h_i*h_j]) in R^32   (distance >= 3 pairs)
    R:  M_r = mean_{(i,j) in category r} z_ij
    N:  M_r = mean over ALL eligible pairs (same z multiset), r non-empty
    Delta = sum_r head(shared_mlp([M_r, log1p(n_r), exists_r]))
    head's last linear layer is zero-initialised, so step-0 R/N == A.

No junction object, no junction state, no message passing that updates node
states, no attention.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import lc_common as L

DEFAULT_CONFIG = {
    "width": 64,
    "stat_rank": 32,
    "local_hidden": 192,
    "atom_hidden": 128,
    "atom_out": 96,
    "bond_hidden": 160,
    "bond_out": 96,
    "rho_hidden": 64,
    "rel_z": 32,
    "rel_hidden": 128,
    "rel_out": 24,
    "rel_mlp_hidden": 48,
    "rel_cat_dim": 8,
    "n_rel_categories": L.N_REL_CATEGORIES,
}


class MLP(nn.Module):
    """Plain MLP with P3-compatible ``lins`` naming (SSL state-dict keys)."""

    def __init__(self, dims):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.silu(x)
        return x


class LocalContent(nn.Module):
    """Shared multi-scale local content encoder (one pass, no recursion)."""

    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        rank = int(cfg["stat_rank"])
        a_out = int(cfg["atom_out"])
        self.atom_out = a_out
        self.rank = rank
        self.nbr_emb = nn.Embedding(L.N_NBR_CODES, rank)
        self.path_emb = nn.Embedding(L.N_PATH_CODES, rank)
        self.end_emb = nn.Embedding(L.N_NBR_CODES, rank)
        # low-rank multiplicative interaction of the two halves of the typed
        # path: this is what keeps the joint (middle type, bond1, bond2) x
        # (end type, bond2) information instead of only adding marginals.
        self.path_u = nn.Embedding(L.N_PATH_CODES, rank)
        self.path_v = nn.Embedding(L.N_NBR_CODES, rank)
        self.joint_proj = nn.Linear(rank, rank)
        self.close_emb = nn.Embedding(2, 8)
        self.path_gate = nn.Embedding(L.N_PATH_CODES, 1)
        nn.init.ones_(self.path_gate.weight)
        in_dim = (
            3 * a_out                  # a_v, sum b, mean b
            + rank                     # S1 joint neighbour (type, bond)
            + 3 * rank + 8             # S2 path code, end code, joint product, closure
            + a_out                    # S2 gated endpoint content
            + L.N_SCALARS
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
        u, v = bond_index[:, 0], bond_index[:, 1]
        degree = a.new_zeros(n_atoms)
        degree = degree.index_add_(0, u, torch.ones_like(u, dtype=a.dtype))
        degree = degree.index_add_(0, v, torch.ones_like(v, dtype=a.dtype))
        b_sum = a.new_zeros((n_atoms, width)).index_add_(0, u, b).index_add_(0, v, b)
        b_mean = b_sum / degree.clamp(min=1.0).unsqueeze(1)

        nbr = a.new_zeros((n_atoms, self.rank)).index_add_(
            0, batch["nbr_atom"], self.nbr_emb(batch["nbr_code"])
        )
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
        x = torch.cat([a, b_sum, b_mean, nbr, path_agg, content_agg, scal], dim=1)
        return self.norm(self.mlp(x))


class RelationHead(nn.Module):
    """One-shot relation composition over atom-pair content (no state update)."""

    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        width = int(cfg["width"])
        z = int(cfg["rel_z"])
        self.z_dim = z
        self.n_cat = int(cfg["n_rel_categories"])
        self.cat_emb = nn.Embedding(self.n_cat, int(cfg["rel_cat_dim"]))
        self.phi = MLP([3 * width, int(cfg["rel_hidden"]), z])
        self.mlp = MLP([z + 2 + int(cfg["rel_cat_dim"]), int(cfg["rel_mlp_hidden"]),
                        int(cfg["rel_out"])])
        self.head = nn.Linear(int(cfg["rel_out"]), 1)
        nn.init.zeros_(self.head.weight)
        nn.init.zeros_(self.head.bias)

    def pair_content(self, h: torch.Tensor, pair_i: torch.Tensor, pair_j: torch.Tensor) -> torch.Tensor:
        hi = h[pair_i]
        hj = h[pair_j]
        return self.phi(torch.cat([hi + hj, (hi - hj).abs(), hi * hj], dim=1))

    def aggregate(
        self,
        z: torch.Tensor,
        pair_graph: torch.Tensor,
        pair_cat: torch.Tensor,
        n_graphs: int,
        mode: str,
    ):
        n_slots = n_graphs * self.n_cat
        slot = pair_graph * self.n_cat + pair_cat
        if z.shape[0]:
            content_sum = z.new_zeros((n_slots, self.z_dim)).index_add_(0, slot, z)
            count = z.new_zeros(n_slots).index_add_(0, slot, torch.ones_like(slot, dtype=z.dtype))
        else:
            content_sum = z.new_zeros((n_slots, self.z_dim))
            count = z.new_zeros(n_slots)
        content = content_sum / count.clamp(min=1.0).unsqueeze(1)
        exists = (count > 0).to(z.dtype)
        if mode == "N":
            # Same z multiset and same true counts; only the content-to-category
            # correspondence is removed.
            gmean = z.sum(0) / max(int(z.shape[0]), 1)
            fill = gmean.unsqueeze(0).expand(n_slots, self.z_dim)
            content = torch.where((count > 0).unsqueeze(1), fill, torch.zeros_like(fill))
        return (
            content.view(n_graphs, self.n_cat, self.z_dim),
            count.view(n_graphs, self.n_cat),
            exists.view(n_graphs, self.n_cat),
        )

    def readout(self, content: torch.Tensor, count: torch.Tensor, exists: torch.Tensor) -> torch.Tensor:
        n_graphs = content.shape[0]
        cat_ids = torch.arange(self.n_cat, device=content.device)
        cat_feat = self.cat_emb(cat_ids).unsqueeze(0).expand(n_graphs, -1, -1)
        feat = torch.cat(
            [content, torch.log1p(count).unsqueeze(-1), exists.unsqueeze(-1), cat_feat],
            dim=-1,
        )
        return self.head(self.mlp(feat)).sum(dim=1).view(-1)

    def forward(
        self,
        h: torch.Tensor,
        pair_i: torch.Tensor,
        pair_j: torch.Tensor,
        pair_graph: torch.Tensor,
        pair_cat: torch.Tensor,
        n_graphs: int,
        mode: str,
    ):
        z = self.pair_content(h, pair_i, pair_j)
        content, count, exists = self.aggregate(z, pair_graph, pair_cat, n_graphs, mode)
        delta = self.readout(content, count, exists)
        return delta, {
            "z": z,
            "M": content,
            "n": count,
            "exists": exists,
        }


class LCRC(nn.Module):
    """Local-contribution body with an optional relation-aware correction."""

    def __init__(self, arm: str, norm: Mapping[str, Any], cfg: Mapping[str, Any] | None = None):
        super().__init__()
        if arm not in ("A", "R", "N"):
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        self.cfg = dict(DEFAULT_CONFIG if cfg is None else {**DEFAULT_CONFIG, **dict(cfg)})
        a_out = int(self.cfg["atom_out"])
        self.atom_encoder = MLP([L.ATOM_FEAT_DIM, int(self.cfg["atom_hidden"]), a_out])
        self.bond_encoder = MLP([3 * a_out + L.BOND_FEAT_DIM, int(self.cfg["bond_hidden"]),
                                 int(self.cfg["bond_out"])])
        self.local = LocalContent(self.cfg)
        self.f = MLP([int(self.cfg["width"]), int(self.cfg["width"]), 1])
        self.rho = MLP([
            L.GLOBAL_DIM + 2 * int(self.cfg["width"]) + 1,
            int(self.cfg["rho_hidden"]),
            1,
        ])
        self.rel: RelationHead | None = None
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        self.register_buffer("scalar_mean", torch.as_tensor(norm["scalar_mean"], dtype=torch.float32))
        self.register_buffer("scalar_scale", torch.as_tensor(norm["scalar_scale"], dtype=torch.float32))

    def add_relation(self) -> None:
        if self.rel is not None:
            raise RuntimeError("relation head already built")
        self.rel = RelationHead(self.cfg)

    def _encode(self, batch: Mapping[str, torch.Tensor]):
        a = self.atom_encoder(batch["atom_feat"])
        idx = batch["bond_index"]
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        bond_feat = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        b = self.bond_encoder(bond_feat)
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
            "mean_h": mean_h,
            "mean_h2": mean_h2,
            "h": h,
            "count": count,
        }
        if self.rel is not None:
            delta, rel_aux = self.rel(
                h,
                batch["pair_i"],
                batch["pair_j"],
                batch["pair_graph"],
                batch["pair_cat"],
                n_graphs,
                self.arm,
            )
            y = y + delta
            aux.update({"delta": delta, **rel_aux})
        else:
            aux["delta"] = torch.zeros(n_graphs, device=y.device, dtype=y.dtype)
        if return_aux:
            return y, aux
        return y


def _param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def _module_breakdown(model: LCRC) -> dict[str, int]:
    out = {
        "atom_encoder": _param_count(model.atom_encoder),
        "bond_encoder": _param_count(model.bond_encoder),
        "local": _param_count(model.local),
        "f": _param_count(model.f),
        "rho": _param_count(model.rho),
    }
    out["rel"] = 0 if model.rel is None else _param_count(model.rel)
    out["total"] = _param_count(model)
    return out


def ssl_path(seed: int):
    return L.P3_SSL[0] if int(seed) == 0 else L.P3_SSL[1]


def load_ssl_encoders(model: LCRC, path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in ("atom_encoder", "bond_encoder") if k not in state]
    if missing:
        raise RuntimeError(f"SSL checkpoint missing {missing}")
    model.atom_encoder.load_state_dict(state["atom_encoder"])
    model.bond_encoder.load_state_dict(state["bond_encoder"])
    return {
        "source": L.track_rel(path),
        "sha256": L.C.sha256_file(path),
        "loaded_modules": ["atom_encoder", "bond_encoder"],
        "junction_path_loaded": False,
        "keys_available": sorted(state.keys()),
    }


def build_model(
    arm: str,
    norm: Mapping[str, Any],
    seed: int,
    device,
    cfg: Mapping[str, Any] | None = None,
) -> tuple[LCRC, dict[str, Any]]:
    """Deterministic builder shared by A/R/N and by R/N.

    Shared modules are always constructed first under ``seed``; the SSL
    atom/bond encoders are loaded immediately afterwards; the relation head is
    then constructed under a fixed second seed.  Therefore:
      * A/R/N share bit-identical local-trunk and rho parameters for one seed;
      * R and N share bit-identical full initial states;
      * no arm-specific constructor consumes RNG before the shared modules.
    """
    torch.manual_seed(int(seed))
    model = LCRC(arm, norm, cfg)
    ssl = load_ssl_encoders(model, ssl_path(seed))
    relation_info = None
    if arm in ("R", "N"):
        torch.manual_seed(int(seed) + 90731)
        model.add_relation()
        relation_info = {
            "built_after_ssl": True,
            "construction_seed": int(seed) + 90731,
            "head_zero_initialised": True,
        }
    model.to(device)
    info = {
        "arm": arm,
        "seed": int(seed),
        "config": model.cfg,
        "ssl": ssl,
        "relation": relation_info,
        "parameters": _param_count(model),
        "parameters_breakdown": _module_breakdown(model),
        "initial_state_sha256": L.C.state_hash(model),
        "requires_grad_true": sum(1 for p in model.parameters() if p.requires_grad),
        "parameter_tensors": sum(1 for p in model.parameters()),
        "has_junction_module": any("junction" in n for n, _ in model.named_modules()),
        "state_keys_with_junction": [k for k in model.state_dict() if "junction" in k],
    }
    if info["parameters"] > 500_000:
        raise RuntimeError(f"{arm}: {info['parameters']} parameters exceeds the 500k cap")
    return model, info


def load_model_from_state(arm: str, norm, state: Mapping[str, torch.Tensor], device, cfg=None) -> LCRC:
    model = LCRC(arm, norm, cfg)
    if arm in ("R", "N"):
        model.add_relation()
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.to(device)
    return model


__all__ = ["DEFAULT_CONFIG", "MLP", "LocalContent", "RelationHead", "LCRC",
           "build_model", "load_model_from_state", "ssl_path", "load_ssl_encoders"]

"""Explicit, branch-switchable re-implementation of the frozen LCRC-A model.

The FULL model (variant ``A0``) is numerically identical to the frozen upstream
``lcrc_model.LCRC`` with the pilot dropout on the final local representation
``h``.  Every variant is built by *deleting* named inputs / learned modules from
FULL, so a variant has no unreachable parameters.

Frozen FULL computation (identical to the executed protocol)::

    a_v = AtomEncoder(atom_feat_v)                       51 -> 128 -> 96
    b_e = BondEncoder([a_u+a_v, |a_u-a_v|, a_u*a_v, bf]) 301 -> 160 -> 96
    S1_v = sum_{e incident v} b_e,  S1mean_v = S1_v / deg(v)
    S1joint_v = sum_{(v,u)} Emb_nbr[(type(u), bond_uv)]
    S2_v = sum_{v->w->u, u != v} [
                Emb_path[(type(w), bt1, bt2)]
              + Emb_end[(type(u), bt2)]
              + W_j (U_path * V_end)
              + Emb_close[closure]
              + gate[(type(w), bt1, bt2)] * a_u ]
    h_v = LayerNorm(LocalMLP([a_v, S1_v, S1mean_v, S1joint_v, S2_v, scalars_v]))
    h_v = Dropout_{0.10}(h_v)                    # training only, one call
    L   = sum_v f(h_v)
    P_g = rho([g, mean_v h_v, mean_v h_v^2, log1p(n_atoms)])
    y   = L + P_g

``B4``/``B5`` keep every module and parameter and only replace the two *joint*
aggregations by per-root marginal products (see
:func:`marginal_joint_agg` / :func:`marginal_content_agg`).
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import dissect_common as C
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


# ---------------------------------------------------------------------------
# building blocks (names ``lins`` kept for checkpoint compatibility)
# ---------------------------------------------------------------------------


class MLP(nn.Module):
    def __init__(self, dims: list[int]):
        super().__init__()
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.silu(x)
        return x


# ---------------------------------------------------------------------------
# frozen-layout marginal aggregation algebra (B4 / B5)
# ---------------------------------------------------------------------------


def marginal_joint_agg(
    u: torch.Tensor,
    w: torch.Tensor,
    key: torch.Tensor,
    n_groups: int,
    weight: torch.Tensor,
    bias: torch.Tensor,
) -> torch.Tensor:
    """``sum_g [ W (n_g * mean_g(u) * mean_g(w)) + n_g * b ]``.

    ``u``/``w`` are ``(P, R)``; ``key`` assigns each occurrence to a group in
    ``[0, n_groups)``.  ``n_g`` is the number of occurrences in the group.  For
    ``n_g == 0`` the group contributes exactly zero (no bias).  For ``n_g == 1``
    the result equals the FULL joint aggregation for that occurrence.
    The product is element-wise per channel; no mean is taken across roots or
    molecules because the group key always contains the root atom.
    """
    dtype = u.dtype
    dev = u.device
    n = torch.zeros(n_groups, dtype=dtype, device=dev).index_add_(
        0, key, torch.ones(key.numel(), dtype=dtype, device=dev)
    )
    su = torch.zeros(n_groups, u.shape[1], dtype=dtype, device=dev).index_add_(0, key, u)
    sw = torch.zeros(n_groups, w.shape[1], dtype=dtype, device=dev).index_add_(0, key, w)
    denom = n.clamp(min=1.0).unsqueeze(1)
    prod = n.unsqueeze(1) * (su / denom) * (sw / denom)
    return F.linear(prod, weight, bias.unsqueeze(0) * n.unsqueeze(1))


def marginal_content_agg(
    t: torch.Tensor,
    z: torch.Tensor,
    key: torch.Tensor,
    n_groups: int,
) -> torch.Tensor:
    """``sum_g [ n_g * mean_g(t) * mean_g(z) ]`` with ``t`` ``(P, 1)`` / ``z`` ``(P, D)``."""
    dtype = z.dtype
    dev = z.device
    n = torch.zeros(n_groups, dtype=dtype, device=dev).index_add_(
        0, key, torch.ones(key.numel(), dtype=dtype, device=dev)
    )
    st = torch.zeros(n_groups, dtype=dtype, device=dev).index_add_(0, key, t.view(-1))
    sz = torch.zeros(n_groups, z.shape[1], dtype=dtype, device=dev).index_add_(0, key, z)
    denom = n.clamp(min=1.0)
    out = n.unsqueeze(1) * (st / denom).unsqueeze(1) * (sz / denom.unsqueeze(1))
    return out


# ---------------------------------------------------------------------------
# local encoder with explicit branches
# ---------------------------------------------------------------------------


class LocalContent(nn.Module):
    def __init__(self, spec: FeatureSpec, variant: str, cfg: Mapping[str, Any]):
        super().__init__()
        v = C.VARIANT_DEFS[variant]
        self.variant = variant
        self.local_cols = (C.variant_cols(variant))[0]
        rank = int(cfg["stat_rank"])
        a_out = int(cfg["atom_out"])
        self.rank = rank
        self.a_out = a_out
        self.s2_agg = v["s2_agg"]
        self.b_mean = bool(v["b_mean"])
        self.s1_joint = bool(v["s1_joint"])
        self.scalar_cols = tuple(int(i) for i in v["scalar_cols"])
        self.s2_type = bool(v["s2_type"])
        self.s2_end = bool(v["s2_end"])
        self.s2_joint = bool(v["s2_joint"])
        self.s2_closure = bool(v["s2_closure"])
        self.s2_content = bool(v["s2_content"])

        # Creation order matches the frozen upstream module so that
        # ``torch.manual_seed(seed)`` reproduces the historical FULL init.
        if self.s1_joint:
            self.nbr_emb = nn.Embedding(spec.n_nbr_codes, rank)
        if self.s2_type:
            self.path_emb = nn.Embedding(spec.n_path_codes, rank)
        if self.s2_end:
            self.end_emb = nn.Embedding(spec.n_nbr_codes, rank)
        if self.s2_joint:
            self.path_u = nn.Embedding(spec.n_path_codes, rank)
            self.path_v = nn.Embedding(spec.n_nbr_codes, rank)
            self.joint_proj = nn.Linear(rank, rank)
        if self.s2_closure:
            self.close_emb = nn.Embedding(2, 8)
        if self.s2_content:
            self.path_gate = nn.Embedding(spec.n_path_codes, 1)
            nn.init.ones_(self.path_gate.weight)
        self.mlp = MLP([len(self.local_cols), int(cfg["local_hidden"]), int(cfg["width"])])
        self.norm = nn.LayerNorm(int(cfg["width"]))

    # -- S2 aggregation ----------------------------------------------------
    def _layout(self) -> list[str]:
        layout = []
        if self.s2_type:
            layout.append("type")
        if self.s2_end:
            layout.append("end")
        if self.s2_joint:
            layout.append("joint")
        if self.s2_closure:
            layout.append("closure")
        return layout

    def _part_dim(self, kind: str) -> int:
        return 8 if kind == "closure" else self.rank

    def _nonmarginal_part(self, kind: str, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        if kind == "type":
            return self.path_emb(batch["path_cat"])
        if kind == "end":
            return self.end_emb(batch["path_end63"])
        if kind == "joint":
            return self.joint_proj(self.path_u(batch["path_cat"]) * self.path_v(batch["path_end63"]))
        if kind == "closure":
            return self.close_emb(batch["path_closure"])
        raise KeyError(kind)

    def _group_key(self, batch: Mapping[str, torch.Tensor], agg: str, n_atoms: int) -> tuple[torch.Tensor, int]:
        root = batch["path_atom"]
        if agg == "marginal":
            return root, int(n_atoms)
        if agg == "marginal_bt2":
            bt2 = batch["path_cat"] % int(C.N_BOND_TYPES)
            return root * int(C.N_BOND_TYPES) + bt2, int(n_atoms) * int(C.N_BOND_TYPES)
        raise KeyError(agg)

    def _joint_agg(self, batch: Mapping[str, torch.Tensor], a: torch.Tensor, agg: str) -> torch.Tensor:
        n_atoms = a.shape[0]
        has = batch["path_cat"].numel() > 0
        if agg == "full":
            if not has:
                return a.new_zeros((n_atoms, self.rank))
            emb = self.joint_proj(self.path_u(batch["path_cat"]) * self.path_v(batch["path_end63"]))
            return a.new_zeros((n_atoms, self.rank)).index_add_(0, batch["path_atom"], emb)
        if not has:
            return a.new_zeros((n_atoms, self.rank))
        key, n_groups = self._group_key(batch, agg, n_atoms)
        u = self.path_u(batch["path_cat"])
        w = self.path_v(batch["path_end63"])
        out = marginal_joint_agg(u, w, key, n_groups, self.joint_proj.weight, self.joint_proj.bias)
        if n_groups != n_atoms:
            out = out.view(n_atoms, int(C.N_BOND_TYPES), self.rank).sum(dim=1)
        return out

    def _content_agg(
        self, batch: Mapping[str, torch.Tensor], a: torch.Tensor, agg: str
    ) -> torch.Tensor:
        n_atoms = a.shape[0]
        width = a.shape[1]
        has = batch["path_cat"].numel() > 0
        if not has:
            return a.new_zeros((n_atoms, width))
        t = self.path_gate(batch["path_cat"])
        z = a[batch["path_end"]]
        if agg == "full":
            return a.new_zeros((n_atoms, width)).index_add_(0, batch["path_atom"], z * t)
        key, n_groups = self._group_key(batch, agg, n_atoms)
        out = marginal_content_agg(t, z, key, n_groups)
        if n_groups != n_atoms:
            out = out.view(n_atoms, int(C.N_BOND_TYPES), width).sum(dim=1)
        return out

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        a: torch.Tensor,
        b: torch.Tensor,
        scal: torch.Tensor,
        *,
        s2_agg_override: str | None = None,
        zero_s2: bool = False,
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        n_atoms = a.shape[0]
        width = a.shape[1]
        agg = s2_agg_override or self.s2_agg
        if agg not in ("full", "marginal", "marginal_bt2"):
            raise ValueError(f"unknown S2 aggregation {agg!r}")

        idx = batch["bond_index"]
        if idx.numel():
            u, v = idx[:, 0], idx[:, 1]
            degree = a.new_zeros(n_atoms)
            degree = degree.index_add_(0, u, torch.ones_like(u, dtype=a.dtype))
            degree = degree.index_add_(0, v, torch.ones_like(v, dtype=a.dtype))
            b_sum = a.new_zeros((n_atoms, width)).index_add_(0, u, b).index_add_(0, v, b)
        else:
            degree = a.new_zeros(n_atoms)
            b_sum = a.new_zeros((n_atoms, width))
        b_mean = b_sum / degree.clamp(min=1.0).unsqueeze(1)

        if self.s1_joint:
            if batch["nbr_code"].numel():
                nbr = a.new_zeros((n_atoms, self.rank)).index_add_(
                    0, batch["nbr_atom"], self.nbr_emb(batch["nbr_code"])
                )
            else:
                nbr = a.new_zeros((n_atoms, self.rank))
        else:
            nbr = None

        # ---- S2 typed block ------------------------------------------------
        layout = self._layout()
        if not layout or zero_s2:
            path_agg = a.new_zeros((n_atoms, sum(self._part_dim(k) for k in layout)))
        else:
            nonmarg = [k for k in layout if not (k == "joint" and agg != "full")]
            if nonmarg:
                cat = torch.cat([self._nonmarginal_part(k, batch) for k in nonmarg], dim=1)
                if batch["path_cat"].numel():
                    agg_nm = a.new_zeros((n_atoms, cat.shape[1])).index_add_(0, batch["path_atom"], cat)
                else:
                    agg_nm = a.new_zeros((n_atoms, cat.shape[1]))
            else:
                agg_nm = a.new_zeros((n_atoms, 0))
            chunks: dict[str, torch.Tensor] = {}
            pos = 0
            for k in nonmarg:
                dim = self._part_dim(k)
                chunks[k] = agg_nm[:, pos : pos + dim]
                pos += dim
            if self.s2_joint and agg != "full":
                chunks["joint"] = self._joint_agg(batch, a, agg)
            path_agg = torch.cat([chunks[k] for k in layout], dim=1)

        if not self.s2_content:
            content_agg = a.new_zeros((n_atoms, 0))
        elif zero_s2:
            content_agg = a.new_zeros((n_atoms, width))
        else:
            content_agg = self._content_agg(batch, a, agg)

        # ---- assemble the frozen 529-column layout -------------------------
        parts: list[torch.Tensor] = [a, b_sum]
        if self.b_mean:
            parts.append(b_mean)
        if self.s1_joint:
            parts.append(nbr)
        if layout:
            parts.append(path_agg)
        if self.s2_content:
            parts.append(content_agg)
        if self.scalar_cols:
            parts.append(scal[:, list(self.scalar_cols)])
        x = torch.cat(parts, dim=1)
        if x.shape[1] != len(self.local_cols):
            raise RuntimeError(f"local width {x.shape[1]} != frozen column count {len(self.local_cols)}")
        h = self.norm(self.mlp(x))
        return h, {"local_input": x, "path_agg": path_agg, "content_agg": content_agg}


# ---------------------------------------------------------------------------
# full model
# ---------------------------------------------------------------------------


class DissectLCRC(nn.Module):
    def __init__(
        self,
        spec: FeatureSpec,
        norm: Mapping[str, Any],
        variant: str,
        cfg: Mapping[str, Any] | None = None,
    ):
        super().__init__()
        if variant not in C.VARIANT_DEFS:
            raise KeyError(variant)
        v = C.VARIANT_DEFS[variant]
        self.variant = variant
        self.spec = spec
        self.cfg = dict(DEFAULT_CONFIG if cfg is None else {**DEFAULT_CONFIG, **dict(cfg)})
        self.use_local = bool(v["use_local"])
        self.use_f = bool(v["use_f"])
        self.use_rho = bool(v["use_rho"])
        self.add_bias = bool(v["add_bias"])
        self.add_reduction = v["add_reduction"]
        self.local_cols, self.rho_cols = C.variant_cols(variant)
        a_out = int(self.cfg["atom_out"])
        width = int(self.cfg["width"])

        # Creation order mirrors the frozen upstream model exactly.
        if self.use_local:
            self.atom_encoder = MLP([spec.atom_feat_dim, int(self.cfg["atom_hidden"]), a_out])
            self.bond_encoder = MLP(
                [3 * a_out + spec.bond_feat_dim, int(self.cfg["bond_hidden"]), int(self.cfg["bond_out"])]
            )
            self.local = LocalContent(spec, variant, self.cfg)
            self.drop = nn.Dropout(float(C.RECIPE["dropout"]))
        if self.use_f:
            self.f = MLP([width, width, 1])
        if self.use_rho:
            self.rho = MLP([len(self.rho_cols), int(self.cfg["rho_hidden"]), 1])
        if self.add_bias:
            self.b0 = nn.Parameter(torch.zeros(1))

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

    # -- encoding ---------------------------------------------------------
    def _encode(self, batch: Mapping[str, torch.Tensor]):
        a = self.atom_encoder(batch["atom_feat"])
        idx = batch["bond_index"]
        if idx.numel():
            au, av = a[idx[:, 0]], a[idx[:, 1]]
            bond_input = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        else:
            bond_input = torch.zeros(
                (0, 3 * a.shape[1] + batch["bond_feat"].shape[1]), dtype=a.dtype, device=a.device
            )
        b = self.bond_encoder(bond_input)
        scal = (batch["atom_scalar"] - self.scalar_mean) / self.scalar_scale
        return a, b, scal

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        return_aux: bool = False,
        zero_g: bool = False,
        zero_contrib: bool = False,
        zero_mean_h2: bool = False,
        zero_s2: bool = False,
        s2_agg_override: str | None = None,
    ):
        n_graphs = int(batch["n_graphs"])
        atom_batch = batch["atom_batch"]
        h = None
        a = b = None
        local_aux: dict[str, torch.Tensor] = {}
        if self.use_local:
            a, b, scal = self._encode(batch)
            h, local_aux = self.local(
                batch, a, b, scal, s2_agg_override=s2_agg_override, zero_s2=zero_s2
            )
            h = self.drop(h)  # exactly one dropout call, shared by both readouts

        mean_h = mean_h2 = None
        if h is not None:
            width = h.shape[1]
            count = h.new_zeros(n_graphs).index_add_(
                0, atom_batch, torch.ones_like(atom_batch, dtype=h.dtype)
            )
            denom = count.clamp(min=1.0).unsqueeze(1)
            mean_h = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h) / denom
            mean_h2 = h.new_zeros((n_graphs, width)).index_add_(0, atom_batch, h * h) / denom
        else:
            count = torch.zeros(n_graphs, dtype=torch.float32, device=batch["global"].device)

        contrib_atom = None
        if self.use_f:
            contrib_atom = self.f(h).view(-1)
            contrib = h.new_zeros(n_graphs).index_add_(0, atom_batch, contrib_atom)
            if self.add_reduction == "mean":
                contrib = contrib / count.clamp(min=1.0)
        else:
            contrib = torch.zeros(n_graphs, dtype=torch.float32, device=batch["global"].device)
        if zero_contrib:
            y_contrib = torch.zeros_like(contrib)
        else:
            y_contrib = contrib

        v = C.VARIANT_DEFS[self.variant]
        rho_in = None
        if self.use_rho:
            parts = []
            if v["use_g"]:
                g = (batch["global"] - self.global_mean) / self.global_scale
                if zero_g:
                    g = torch.zeros_like(g)
                parts.append(g)
            if v["use_mean_h"]:
                parts.append(mean_h)
            if v["use_mean_h2"]:
                parts.append(torch.zeros_like(mean_h2) if zero_mean_h2 else mean_h2)
            if v["use_log1p_n"]:
                parts.append(torch.log1p(count).unsqueeze(1))
            rho_in = torch.cat(parts, dim=1)
            if rho_in.shape[1] != len(self.rho_cols):
                raise RuntimeError(f"rho width {rho_in.shape[1]} != frozen column count {len(self.rho_cols)}")
            rho = self.rho(rho_in).view(-1)
        else:
            rho = self.b0.reshape(1).expand(n_graphs)
        y = y_contrib + rho

        if return_aux:
            aux = {
                "contrib": contrib,
                "contrib_used": y_contrib,
                "rho": rho,
                "contrib_atom": contrib_atom,
                "mean_h": mean_h,
                "mean_h2": mean_h2,
                "h": h,
                "count": count,
                "delta": torch.zeros(n_graphs, device=y.device, dtype=y.dtype),
                "a": a,
                "b": b,
                "rho_input": rho_in,
                **local_aux,
            }
            return y, aux
        return y


# ---------------------------------------------------------------------------
# state construction / remapping
# ---------------------------------------------------------------------------


def parameter_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def module_breakdown(model: DissectLCRC) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, child in model.named_children():
        out[name] = parameter_count(child)
    out["total"] = parameter_count(model)
    return out


def state_fingerprint(state: Mapping[str, torch.Tensor]) -> str:
    h = hashlib.sha256()
    for key, value in sorted(state.items()):
        h.update(key.encode("utf-8"))
        h.update(np.ascontiguousarray(value.detach().cpu().numpy(), dtype=np.float32).tobytes())
    return h.hexdigest()


def build_full_state(spec: FeatureSpec, norm: Mapping[str, Any], seed: int) -> dict[str, torch.Tensor]:
    """One complete FULL initial state per training seed (CPU RNG only)."""
    torch.manual_seed(int(seed))
    model = DissectLCRC(spec, norm, "A0")
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def remap_state(variant: str, full: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Copy the FULL state into ``variant`` with fixed column mappings.

    Removed columns are dropped from the first Linear layer of the affected
    MLP; biases and all later layers are copied unchanged.  Removed modules are
    not created at all.  The new all-graph constant ``b0`` starts at zero.
    """
    v = C.VARIANT_DEFS[variant]
    local_cols, rho_cols = C.variant_cols(variant)
    out: dict[str, torch.Tensor] = {}
    for k in ("global_mean", "global_scale", "scalar_mean", "scalar_scale"):
        out[k] = full[k].clone()
    if v["use_local"]:
        for k in ("atom_encoder.lins.0.weight", "atom_encoder.lins.0.bias", "atom_encoder.lins.1.weight",
                  "atom_encoder.lins.1.bias", "bond_encoder.lins.0.weight", "bond_encoder.lins.0.bias",
                  "bond_encoder.lins.1.weight", "bond_encoder.lins.1.bias"):
            out[k] = full[k].clone()
        for key, flag in (
            ("local.nbr_emb.weight", "s1_joint"),
            ("local.path_emb.weight", "s2_type"),
            ("local.end_emb.weight", "s2_end"),
            ("local.path_u.weight", "s2_joint"),
            ("local.path_v.weight", "s2_joint"),
            ("local.joint_proj.weight", "s2_joint"),
            ("local.joint_proj.bias", "s2_joint"),
            ("local.close_emb.weight", "s2_closure"),
            ("local.path_gate.weight", "s2_content"),
        ):
            if v[flag]:
                out[key] = full[key].clone()
        w0 = full["local.mlp.lins.0.weight"]
        out["local.mlp.lins.0.weight"] = w0[:, list(local_cols)].clone()
        out["local.mlp.lins.0.bias"] = full["local.mlp.lins.0.bias"].clone()
        for k in ("local.mlp.lins.1.weight", "local.mlp.lins.1.bias", "local.norm.weight", "local.norm.bias"):
            out[k] = full[k].clone()
    if v["use_f"]:
        for k in ("f.lins.0.weight", "f.lins.0.bias", "f.lins.1.weight", "f.lins.1.bias"):
            out[k] = full[k].clone()
    if v["use_rho"]:
        out["rho.lins.0.weight"] = full["rho.lins.0.weight"][:, list(rho_cols)].clone()
        out["rho.lins.0.bias"] = full["rho.lins.0.bias"].clone()
        for k in ("rho.lins.1.weight", "rho.lins.1.bias"):
            out[k] = full[k].clone()
    if v["add_bias"]:
        out["b0"] = torch.zeros(1, dtype=full["global_mean"].dtype)
    return out


def build_variant(
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    variant: str,
    full_state: Mapping[str, torch.Tensor],
    device,
    cfg: Mapping[str, Any] | None = None,
) -> DissectLCRC:
    model = DissectLCRC(spec, norm, variant, cfg)
    model.load_state_dict(remap_state(variant, full_state), strict=True)
    model.to(device)
    return model


def build_full_model(
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    seed: int,
    device,
    cfg: Mapping[str, Any] | None = None,
) -> DissectLCRC:
    state = build_full_state(spec, norm, seed)
    model = DissectLCRC(spec, norm, "A0", cfg)
    model.load_state_dict(state, strict=True)
    model.to(device)
    return model


def load_checkpoint_model(
    spec: FeatureSpec,
    norm: Mapping[str, Any],
    variant: str,
    state: Mapping[str, torch.Tensor],
    device,
    cfg: Mapping[str, Any] | None = None,
) -> DissectLCRC:
    model = DissectLCRC(spec, norm, variant, cfg)
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    model.to(device)
    model.eval()
    return model


def shared_state_equal(variant: str, full: Mapping[str, torch.Tensor], state: Mapping[str, torch.Tensor]) -> dict[str, Any]:
    """Prove every shared parameter is bitwise identical to the FULL state."""
    v = C.VARIANT_DEFS[variant]
    local_cols, rho_cols = C.variant_cols(variant)
    checks: dict[str, bool] = {}
    for key in ("atom_encoder.lins.0.weight", "atom_encoder.lins.0.bias", "atom_encoder.lins.1.weight",
                "atom_encoder.lins.1.bias", "bond_encoder.lins.0.weight", "bond_encoder.lins.0.bias",
                "bond_encoder.lins.1.weight", "bond_encoder.lins.1.bias",
                "local.mlp.lins.0.bias", "local.mlp.lins.1.weight", "local.mlp.lins.1.bias",
                "local.norm.weight", "local.norm.bias",
                "f.lins.0.weight", "f.lins.0.bias", "f.lins.1.weight", "f.lins.1.bias",
                "rho.lins.0.bias", "rho.lins.1.weight", "rho.lins.1.bias",
                "global_mean", "global_scale", "scalar_mean", "scalar_scale"):
        if key in state:
            checks[key] = bool(torch.equal(state[key].cpu(), full[key].cpu()))
    for key, flag in (
        ("local.nbr_emb.weight", "s1_joint"),
        ("local.path_emb.weight", "s2_type"),
        ("local.end_emb.weight", "s2_end"),
        ("local.path_u.weight", "s2_joint"),
        ("local.path_v.weight", "s2_joint"),
        ("local.joint_proj.weight", "s2_joint"),
        ("local.joint_proj.bias", "s2_joint"),
        ("local.close_emb.weight", "s2_closure"),
        ("local.path_gate.weight", "s2_content"),
    ):
        if v[flag] and key in state:
            checks[key] = bool(torch.equal(state[key].cpu(), full[key].cpu()))
    if "local.mlp.lins.0.weight" in state:
        checks["local.mlp.lins.0.weight"] = bool(
            torch.equal(state["local.mlp.lins.0.weight"].cpu(), full["local.mlp.lins.0.weight"][:, list(local_cols)].cpu())
        )
    if "rho.lins.0.weight" in state:
        checks["rho.lins.0.weight"] = bool(
            torch.equal(state["rho.lins.0.weight"].cpu(), full["rho.lins.0.weight"][:, list(rho_cols)].cpu())
        )
    return {"all_equal": all(checks.values()), "checks": checks, "n_checked": len(checks)}


__all__ = [
    "DEFAULT_CONFIG",
    "MLP",
    "LocalContent",
    "DissectLCRC",
    "marginal_joint_agg",
    "marginal_content_agg",
    "parameter_count",
    "module_breakdown",
    "state_fingerprint",
    "build_full_state",
    "remap_state",
    "build_variant",
    "build_full_model",
    "load_checkpoint_model",
    "shared_state_equal",
]

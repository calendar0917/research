"""Second-hop composition bodies.

The trunk is LCRC-A (Local Contribution + small rho readout), reproduced with
the *same construction order and RNG consumption* as the frozen
``zinc_local_relation_composition_v1/code/lc_models.py``; the only change is a
128-wide addition to the ``LocalContent`` input:

    [B_v (96), C(v) (32)]

where, per ordered non-backtracking two-hop path ``p = (v, w, u)``,

    B_v       = (1/sqrt(n_v)) * sum_p b_e2(p)
    C_real(v) = (1/sqrt(n_v)) * sum_p (P_b b_e2(p)) .* (P_a a_u(p))
    C_null(v) = sqrt(n_v) * mean_p(P_b b_e2(p)) .* mean_p(P_a a_u(p))

``b_e2`` is the *shared* bond encoder applied to the second bond; ``P_b``/``P_a``
are fixed non-trainable random projections drawn once from seed 20261105.
Means are taken per root only (never across roots, molecules or batches).

The first linear layer is widened in place *after* the historical construction,
with the 128 new input columns zero-initialised.  Therefore the step-0 forward
is bit-identical to LCRC-A and every other tensor is bit-identical to the
historical initial state.
"""

from __future__ import annotations

import math
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import sh_common as S

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

ARMS = ("Real", "Null")


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
    """Frozen LCRC-A local content encoder (optionally widened by 128 inputs)."""

    def __init__(self, cfg: Mapping[str, Any]):
        super().__init__()
        rank = int(cfg["stat_rank"])
        a_out = int(cfg["atom_out"])
        self.atom_out = a_out
        self.rank = rank
        self.nbr_emb = nn.Embedding(S.L.N_NBR_CODES, rank)
        self.path_emb = nn.Embedding(S.L.N_PATH_CODES, rank)
        self.end_emb = nn.Embedding(S.L.N_NBR_CODES, rank)
        self.path_u = nn.Embedding(S.L.N_PATH_CODES, rank)
        self.path_v = nn.Embedding(S.L.N_NBR_CODES, rank)
        self.joint_proj = nn.Linear(rank, rank)
        self.close_emb = nn.Embedding(2, 8)
        self.path_gate = nn.Embedding(S.L.N_PATH_CODES, 1)
        nn.init.ones_(self.path_gate.weight)
        in_dim = (
            3 * a_out
            + rank
            + 3 * rank + 8
            + a_out
            + S.L.N_SCALARS
        )
        self.base_in_dim = in_dim
        assert in_dim == S.BASE_X_DIM
        self.mlp = MLP([in_dim, int(cfg["local_hidden"]), int(cfg["width"])])
        self.norm = nn.LayerNorm(int(cfg["width"]))

    def widen_first_layer(self, extra_dim: int) -> None:
        """In-place widen of ``lins[0]``; new input columns are exactly zero."""
        old = self.mlp.lins[0]
        rng = torch.get_rng_state()
        new = nn.Linear(old.in_features + int(extra_dim), old.out_features)
        torch.set_rng_state(rng)
        with torch.no_grad():
            new.weight.zero_()
            new.weight[:, : old.in_features].copy_(old.weight)
            new.bias.copy_(old.bias)
        self.mlp.lins[0] = new

    def build_input(
        self,
        batch: Mapping[str, torch.Tensor],
        a: torch.Tensor,
        b: torch.Tensor,
        scal: torch.Tensor,
        extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """Explicit ``X_v`` (529) plus the optional 128-wide second-hop block."""
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
        parts = [a, b_sum, b_mean, nbr, path_agg, content_agg, scal]
        if extra is not None:
            parts.append(extra)
        return torch.cat(parts, dim=1)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        a: torch.Tensor,
        b: torch.Tensor,
        scal: torch.Tensor,
        extra: torch.Tensor | None = None,
    ) -> torch.Tensor:
        return self.norm(self.mlp(self.build_input(batch, a, b, scal, extra)))


class SecondHopContent(nn.Module):
    """Second-hop bond content + fixed-projection pair summary (per root)."""

    def __init__(self, width: int = S.SECOND_HOP_DIM, proj_dim: int = S.PROJ_DIM,
                 seed: int = S.PROJ_SEED):
        super().__init__()
        g = torch.Generator().manual_seed(int(seed))
        p_b = torch.randn(int(width), int(proj_dim), generator=g) / math.sqrt(int(width))
        p_a = torch.randn(int(width), int(proj_dim), generator=g) / math.sqrt(int(width))
        self.register_buffer("P_b", p_b)
        self.register_buffer("P_a", p_a)
        for p in (self.P_b, self.P_a):
            p.requires_grad_(False)
        self.width = int(width)
        self.proj_dim = int(proj_dim)
        self.seed = int(seed)

    def forward(self, batch: Mapping[str, torch.Tensor], a: torch.Tensor,
                b: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        n_atoms = a.shape[0]
        path_atom = batch["path_atom"]
        n = batch["path_count"].to(a.dtype)                       # per-atom path count
        b2 = b[batch["path_e2_pos"]]                              # second-hop bond content
        a_u = a[batch["path_end"]]                                # endpoint atom content
        inv_sqrt = torch.where(n > 0, n.clamp(min=1.0).rsqrt(), torch.zeros_like(n))
        n_safe = n.clamp(min=1.0)

        b2_sum = a.new_zeros((n_atoms, self.width)).index_add_(0, path_atom, b2)
        b_hop = b2_sum * inv_sqrt.unsqueeze(1)

        u_pair = b2 @ self.P_b
        v_pair = a_u @ self.P_a
        prod = u_pair * v_pair
        prod_sum = a.new_zeros((n_atoms, self.proj_dim)).index_add_(0, path_atom, prod)
        u_sum = a.new_zeros((n_atoms, self.proj_dim)).index_add_(0, path_atom, u_pair)
        v_sum = a.new_zeros((n_atoms, self.proj_dim)).index_add_(0, path_atom, v_pair)

        c_real = prod_sum * inv_sqrt.unsqueeze(1)
        mean_u = u_sum / n_safe.unsqueeze(1)
        mean_v = v_sum / n_safe.unsqueeze(1)
        alive = (n > 0).to(a.dtype).unsqueeze(1)
        c_null = mean_u * mean_v * n.clamp(min=0.0).sqrt().unsqueeze(1) * alive
        return b_hop, c_real, c_null


class SecondHopLCRC(nn.Module):
    """LCRC-A trunk with the 128-wide second-hop input addition."""

    def __init__(self, arm: str, norm: Mapping[str, Any],
                 cfg: Mapping[str, Any] | None = None):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        self.cfg = dict(DEFAULT_CONFIG if cfg is None else {**DEFAULT_CONFIG, **dict(cfg)})
        a_out = int(self.cfg["atom_out"])
        # ---- historical LCRC-A construction order (identical RNG consumption) ----
        self.atom_encoder = MLP([S.L.ATOM_FEAT_DIM, int(self.cfg["atom_hidden"]), a_out])
        self.bond_encoder = MLP([3 * a_out + S.L.BOND_FEAT_DIM, int(self.cfg["bond_hidden"]),
                                 int(self.cfg["bond_out"])])
        self.local = LocalContent(self.cfg)
        self.f = MLP([int(self.cfg["width"]), int(self.cfg["width"]), 1])
        self.rho = MLP([
            S.L.GLOBAL_DIM + 2 * int(self.cfg["width"]) + 1,
            int(self.cfg["rho_hidden"]),
            1,
        ])
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        self.register_buffer("scalar_mean", torch.as_tensor(norm["scalar_mean"], dtype=torch.float32))
        self.register_buffer("scalar_scale", torch.as_tensor(norm["scalar_scale"], dtype=torch.float32))
        self.sh: SecondHopContent | None = None

    # ---- extension (called after the SSL encoders are loaded) --------------
    def enable_second_hop(self) -> None:
        if self.sh is not None:
            raise RuntimeError("second hop already enabled")
        rng = torch.get_rng_state()
        self.local.widen_first_layer(S.EXTRA_DIM)
        self.sh = SecondHopContent()
        torch.set_rng_state(rng)

    def _encode(self, batch: Mapping[str, torch.Tensor], pair_mode: str | None = None):
        a = self.atom_encoder(batch["atom_feat"])
        idx = batch["bond_index"]
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        bond_feat = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        b = self.bond_encoder(bond_feat)
        scal = (batch["atom_scalar"] - self.scalar_mean) / self.scalar_scale
        extra = None
        aux: dict[str, torch.Tensor] = {}
        if self.sh is not None:
            b_hop, c_real, c_null = self.sh(batch, a, b)
            mode = ("real" if self.arm == "Real" else "null") if pair_mode is None \
                else str(pair_mode).lower()
            if mode not in ("real", "null"):
                raise ValueError(f"bad pair_mode {pair_mode!r}")
            c_pair = c_real if mode == "real" else c_null
            extra = torch.cat([b_hop, c_pair], dim=1)
            aux = {"B": b_hop, "C_real": c_real, "C_null": c_null,
                   "second_hop_extra": extra, "C_used": c_pair}
        h = self.local(batch, a, b, scal, extra)
        return a, b, h, aux

    def forward(self, batch: Mapping[str, torch.Tensor], *, return_aux: bool = False,
                pair_mode: str | None = None):
        """``pair_mode`` overrides the arm's C channel (``real`` / ``null``) for probes."""
        if pair_mode is not None and self.sh is None:
            raise RuntimeError("pair_mode requires the second-hop input")
        n_graphs = int(batch["n_graphs"])
        a, b, h, aux = self._encode(batch, pair_mode)

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
        aux.update({"contrib": contrib, "rho": rho, "mean_h": mean_h,
                    "mean_h2": mean_h2, "h": h, "count": count})
        if return_aux:
            return y, aux
        return y


def _param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def parameter_breakdown(model: SecondHopLCRC) -> dict[str, int]:
    out = {
        "atom_encoder": _param_count(model.atom_encoder),
        "bond_encoder": _param_count(model.bond_encoder),
        "local": _param_count(model.local),
        "f": _param_count(model.f),
        "rho": _param_count(model.rho),
        "second_hop": 0 if model.sh is None else _param_count(model.sh),
        "buffers": {
            "P_b": int(model.sh.P_b.numel()) if model.sh is not None else 0,
            "P_a": int(model.sh.P_a.numel()) if model.sh is not None else 0,
        },
    }
    out["total"] = _param_count(model)
    return out


def ssl_path(seed: int):
    return S.L.P3_SSL[0] if int(seed) == 0 else S.L.P3_SSL[1]


def load_ssl_encoders(model: SecondHopLCRC, path) -> dict[str, Any]:
    state = torch.load(path, map_location="cpu", weights_only=False)
    missing = [k for k in ("atom_encoder", "bond_encoder") if k not in state]
    if missing:
        raise RuntimeError(f"SSL checkpoint missing {missing}")
    model.atom_encoder.load_state_dict(state["atom_encoder"])
    model.bond_encoder.load_state_dict(state["bond_encoder"])
    return {
        "source": S.track_rel(path),
        "sha256": S.L.C.sha256_file(path),
        "loaded_modules": ["atom_encoder", "bond_encoder"],
        "junction_path_loaded": False,
    }


def build_base_model(norm: Mapping[str, Any], seed: int,
                     cfg: Mapping[str, Any] | None = None) -> SecondHopLCRC:
    """Historical A construction only (no widening, no second-hop module)."""
    torch.manual_seed(int(seed))
    model = SecondHopLCRC("Real", norm, cfg)
    load_ssl_encoders(model, ssl_path(seed))
    return model


def build_model(arm: str, norm: Mapping[str, Any], seed: int, device,
                cfg: Mapping[str, Any] | None = None,
                verify_supplement_sha: Mapping[str, str] | None = None):
    torch.manual_seed(int(seed))
    model = SecondHopLCRC(arm, norm, cfg)
    ssl = load_ssl_encoders(model, ssl_path(seed))
    model.enable_second_hop()
    model.to(device)
    info = {
        "arm": arm,
        "seed": int(seed),
        "config": model.cfg,
        "ssl": ssl,
        "parameters": _param_count(model),
        "parameters_breakdown": parameter_breakdown(model),
        "initial_state_sha256": S.L.C.state_hash(model),
        "param_cap_ok": bool(_param_count(model) <= S.PARAM_CAP),
        "has_junction_module": any("junction" in n for n, _ in model.named_modules()),
        "state_keys_with_junction": [k for k in model.state_dict() if "junction" in k],
        "attention_modules": [n for n, m in model.named_modules()
                              if "attention" in type(m).__name__.lower()],
        "projections": S.projection_hashes(model),
        "projection_trainable": bool(model.sh.P_b.requires_grad or model.sh.P_a.requires_grad),
        "new_input_columns_zero": bool(torch.count_nonzero(
            model.local.mlp.lins[0].weight[:, S.BASE_X_DIM:]).item() == 0),
        "feature_version": S.FEATURE_VERSION,
    }
    if not info["param_cap_ok"]:
        raise RuntimeError(f"{arm}: {info['parameters']} parameters exceeds {S.PARAM_CAP}")
    if verify_supplement_sha is not None:
        info["supplement_sha_check"] = dict(verify_supplement_sha)
    return model, info


def load_historical_A(model: SecondHopLCRC, state: Mapping[str, torch.Tensor]) -> dict:
    """Load a frozen LCRC-A checkpoint into the widened model (new columns zero)."""
    target = model.state_dict()
    out: dict[str, torch.Tensor] = {}
    widened = "local.mlp.lins.0.weight"
    for k, v in state.items():
        if k not in target:
            raise RuntimeError(f"historical key {k!r} missing in the widened model")
        t = target[k]
        if k == widened:
            if v.shape[0] != t.shape[0] or v.shape[1] != t.shape[1] - S.EXTRA_DIM:
                raise RuntimeError(f"historical first layer {tuple(v.shape)} incompatible "
                                   f"with widened {tuple(t.shape)}")
            new = t.clone()
            new.zero_()
            new[:, : v.shape[1]] = v
            out[k] = new
        else:
            if v.shape != t.shape:
                raise RuntimeError(f"shape mismatch for {k}: {tuple(v.shape)} vs {tuple(t.shape)}")
            out[k] = v
    missing = sorted(set(target) - set(out))
    if missing != ["sh.P_a", "sh.P_b"]:
        raise RuntimeError(f"unexpected missing keys when loading historical A: {missing}")
    model.load_state_dict(out, strict=False)
    return {"loaded": len(out), "zero_new_columns": True, "missing_buffers": missing}


def load_model_from_state(arm: str, norm, state: Mapping[str, torch.Tensor], device,
                          cfg=None) -> SecondHopLCRC:
    model = SecondHopLCRC(arm, norm, cfg)
    model.enable_second_hop()
    model.load_state_dict({k: v.to(device) for k, v in state.items()})
    model.to(device)
    return model


__all__ = ["DEFAULT_CONFIG", "ARMS", "MLP", "LocalContent", "SecondHopContent",
           "SecondHopLCRC", "build_model", "build_base_model", "load_ssl_encoders",
           "load_historical_A", "load_model_from_state", "parameter_breakdown",
           "ssl_path"]

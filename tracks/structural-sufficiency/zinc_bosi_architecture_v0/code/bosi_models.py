"""BOSI v0 — bond-oriented structural incidence model.

Fixed forward (DESIGN.md §2.4), no learned junction token, no recursion:

    a_v = atom_encoder(atom_feat_v)                                    108 -> 128 -> 96
    b_e = bond_encoder([au+av, |au-av|, au*av, bond_feat_e])           295 -> 160 -> 96
    x[v,e] = tanh(half_encoder([a_v, a_u, b_e, bond_feat_e]))          295 -> 128 -> 32
    M[v,e,r,k] = sum_{f != e, rel(v,e,f)=r} T_k(x[v,f])                (4, 3, 32)
    n[v,e,r]  = #{f != e, rel(v,e,f)=r}                                (4,)
    s[v,e]    = side([a_v, b_e, x[v,e], M, n, closure, log1p(deg)])    617 -> 128 -> 96
    c[e]      = compose([b_e, s[u,e]+s[v,e], |s[u,e]-s[v,e]|,
                         s[u,e]*s[v,e], bond_feat_e])                  391 -> 160 -> 96
    z[G]      = [sum_e c, sum_e c^2, sum_v a, sum_v a^2, global33]      417
    yhat      = reader(z)                                              417 -> 256 -> 256 -> 1

``arm`` is the only architectural switch, and it is non-parametric:

* ``"R"`` — the true per-relation partition of the partner moments (DESIGN §2.3).
* ``"B"`` — the same relation counts, but the *leave-one-out centre total*
  broadcast into every group: ``M_B[r] = n[r]/max(deg-1,1) * sum_{f != e} T(x_f)``.
  This is the "mean of the centre statistics" control: identical parameter
  shapes, identical initial state, identical totals ``sum_r M[r]``; only the
  allocation of partner statistics to relation groups is replaced.

``"Bfull"`` is *not trained*: it is DESIGN.md §3's literal formula
``n[r]/deg * sum_{f in Inc(v)} T(x_f)`` and exists only for the CPU diagnostic
that isolates how much of R-B is the extra shrink/self-inclusion term.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

N_REL = 4
RANK = 32
WIDTH = 96
N_ATOM_FEAT = 108
BOND_FEAT = 7
GLOBAL_DIM = 33
HALF_IN = 3 * WIDTH + BOND_FEAT          # 295
SIDE_IN = 2 * WIDTH + RANK + N_REL * 3 * RANK + N_REL + 4 + 1  # 617
COMPOSE_IN = WIDTH + 3 * WIDTH + BOND_FEAT  # 391
READER_IN = 4 * WIDTH + GLOBAL_DIM          # 417
ARMS = ("R", "B")


class MLP(nn.Module):
    """Plain SiLU MLP with the same ``lins`` naming as the frozen SSL encoders."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_hidden: int = 1,
                 dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + [hidden] * n_hidden + [out_dim]
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1])
                                   for i in range(len(dims) - 1)])
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.silu(x)
                if self.dropout > 0.0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x


def chebyshev(x: torch.Tensor) -> torch.Tensor:
    """``[T1, T2, T3]`` along the last axis -> ``[..., 3*d]`` (k-major)."""
    return torch.cat([x, 2.0 * x * x - 1.0, 4.0 * x * x * x - 3.0 * x], dim=-1)


class BOSI(nn.Module):
    def __init__(self, norm: Mapping[str, object], *, arm: str = "R", dropout: float = 0.0):
        super().__init__()
        if arm not in ("R", "B", "Bfull"):
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        self.atom_encoder = MLP(N_ATOM_FEAT, 128, WIDTH, 1, dropout)
        self.bond_encoder = MLP(3 * WIDTH + BOND_FEAT, 160, WIDTH, 1, dropout)
        self.half_encoder = MLP(HALF_IN, 128, RANK, 1, dropout)
        self.side = MLP(SIDE_IN, 128, WIDTH, 1, dropout)
        self.compose = MLP(COMPOSE_IN, 160, WIDTH, 1, dropout)
        self.reader = MLP(READER_IN, 256, 1, n_hidden=2, dropout=dropout)
        self.register_buffer("global_mean",
                             torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale",
                             torch.as_tensor(norm["global_scale"], dtype=torch.float32))

    # -- encoders -----------------------------------------------------------
    def encode_tokens(self, batch: Mapping[str, torch.Tensor]):
        a = self.atom_encoder(batch["atom_feat"])
        idx = batch["bond_index"]
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        b = self.bond_encoder(torch.cat([au + av, (au - av).abs(), au * av,
                                         batch["bond_feat"]], dim=1))
        return a, b

    def half_edges(self, batch: Mapping[str, torch.Tensor], a, b):
        centre = batch["half_atom"]
        e = batch["half_bond"]
        idx = batch["bond_index"]
        u, v = idx[e, 0], idx[e, 1]
        other = torch.where(centre == u, v, u)
        x = torch.tanh(self.half_encoder(torch.cat(
            [a[centre], a[other], b[e], batch["bond_feat"][e]], dim=1)))
        return x

    # -- relation statistics ------------------------------------------------
    def relation_stats(self, batch: Mapping[str, torch.Tensor], x: torch.Tensor):
        vals = chebyshev(x)                                   # [H, 3*RANK]
        H = x.shape[0]
        n_half = int(H)
        tgt_rel = batch["partner_tgt"] * N_REL + batch["partner_rel"]
        gathered = vals[batch["partner_half"]]
        M = x.new_zeros((n_half * N_REL, vals.shape[1]))
        M.index_add_(0, tgt_rel, gathered)
        n = x.new_zeros((n_half * N_REL,))
        n.index_add_(0, tgt_rel, torch.ones_like(tgt_rel, dtype=x.dtype))
        M = M.view(n_half, N_REL * vals.shape[1])
        n = n.view(n_half, N_REL)

        if self.arm == "R":
            pass
        elif self.arm == "B":
            total = x.new_zeros((n_half, vals.shape[1]))
            total.index_add_(0, batch["partner_tgt"], gathered)
            denom = (batch["half_deg"] - 1.0).clamp(min=1.0).view(-1, 1)
            M = (n.unsqueeze(-1) / denom.unsqueeze(-1)) * total.unsqueeze(1)
            M = M.reshape(n_half, N_REL * vals.shape[1])
        elif self.arm == "Bfull":
            n_atoms = int(batch["atom_feat"].shape[0])
            full = x.new_zeros((n_atoms, vals.shape[1]))
            full.index_add_(0, batch["half_atom"], vals)
            denom = batch["half_deg"].clamp(min=1.0).view(-1, 1)
            M = (n.unsqueeze(-1) / denom.unsqueeze(-1)) * full[batch["half_atom"]].unsqueeze(1)
            M = M.reshape(n_half, N_REL * vals.shape[1])
        else:  # pragma: no cover
            raise RuntimeError(self.arm)
        return M, n

    # -- forward ------------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor], *, return_stats: bool = False):
        a, b = self.encode_tokens(batch)
        x = self.half_edges(batch, a, b)
        M, n = self.relation_stats(batch, x)
        half_deg = batch["half_deg"].view(-1, 1)
        s = self.side(torch.cat(
            [a[batch["half_atom"]], b[batch["half_bond"]], x, M, n, batch["closure"],
             torch.log1p(half_deg)], dim=1))
        hb = batch["half_of_bond"]
        su, sv = s[hb[:, 0]], s[hb[:, 1]]
        c = self.compose(torch.cat(
            [b, su + sv, (su - sv).abs(), su * sv, batch["bond_feat"]], dim=1))
        B = int(batch["n_graphs"])
        bb, ab = batch["bond_batch"], batch["atom_batch"]
        zc = c.new_zeros((B, WIDTH)).index_add_(0, bb, c)
        zc2 = c.new_zeros((B, WIDTH)).index_add_(0, bb, c * c)
        za = c.new_zeros((B, WIDTH)).index_add_(0, ab, a)
        za2 = c.new_zeros((B, WIDTH)).index_add_(0, ab, a * a)
        g = (batch["global"] - self.global_mean) / self.global_scale
        z = torch.cat([zc, zc2, za, za2, g], dim=1)
        y = self.reader(z).view(-1)
        if return_stats:
            return y, {"M": M, "n": n, "z": z, "c": c, "s": s, "x": x, "a": a, "b": b}
        return y


def parameter_breakdown(model: BOSI) -> dict:
    def n(m):
        return int(sum(p.numel() for p in m.parameters()))

    return {
        "atom_encoder": n(model.atom_encoder),
        "bond_encoder": n(model.bond_encoder),
        "half_encoder": n(model.half_encoder),
        "side": n(model.side),
        "compose": n(model.compose),
        "reader": n(model.reader),
        "total": n(model),
    }


def load_ssl(model: BOSI, path, device="cpu") -> dict:
    """Load only the atom/bond encoders from the frozen train-only SSL checkpoint."""
    import ad_common as C

    state = torch.load(path, map_location="cpu", weights_only=False)
    for name in ("atom_encoder", "bond_encoder"):
        getattr(model, name).load_state_dict({k: v.to(device) for k, v in state[name].items()})
    return {
        "source": str(path),
        "sha256": C.sha256_file(path),
        "keys_loaded": ["atom_encoder", "bond_encoder"],
        "junction_encoder_loaded": False,
    }


def build_model(norm: Mapping[str, object], arm: str, seed: int, device,
                *, dropout: float = 0.0) -> tuple[BOSI, dict]:
    """One shared initial state per seed; ``arm`` never consumes RNG."""
    import ad_common as C

    import bosi_common as BC

    torch.manual_seed(int(seed))
    model = BOSI(norm, arm=arm, dropout=dropout)
    info = {"ssl": load_ssl(model, BC.ssl_path(seed), device)}
    model.to(device)
    info["parameters"] = C.parameter_count(model)
    info["parameters_breakdown"] = parameter_breakdown(model)
    info["initial_state_sha256"] = C.state_hash(model)
    info["arm"] = arm
    info["seed"] = int(seed)
    if info["parameters"] > 500_000:
        raise RuntimeError(f"parameter cap exceeded: {info['parameters']}")
    return model, info

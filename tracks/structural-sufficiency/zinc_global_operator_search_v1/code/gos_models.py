"""Shared carrier + three global operators over undirected bond objects.

Routes
------
C0 : local carrier only.  z0 (513-d amplitude/count statistics) -> reader.
A  : amplitude-sensitive global content interaction (softmax over bond objects)
     score_bc = (q_b.k_c - 0.5||k_c||^2)/sqrt(d_head) + beta(r_bc)
     m_b      = sum_c a_bc * (1 + tanh(gamma(r_bc))) * v_c
     NOTE: this is the *intended* operator.  The historical A checkpoints were
     trained with a query-side broadcast of the key-energy term, which the row
     softmax cancels exactly, so those runs implemented plain dot-product
     attention plus the relation-conditioned content gate (see ERRATUM.md).
     ``MixA.mode`` now selects the arithmetic explicitly and adds no parameters:
     ``legacy`` (verbatim historical expression), ``dot`` (reference operator
     that legacy equals), ``l2`` (key-energy term broadcast on the key axis),
     ``gn`` (dot scoring with the gated weights renormalised over keys).
K  : signed, non-competing global structural kernel
     m_b      = sum_c tanh(K(r_bc)) * V(N(h_c)) / mean_train_n_bonds

Both A and K keep the *un-normalised* 513-d z0 path straight into the reader:
the graph-level LayerNorm of the core never touches the amplitude/count channel.
The core is a single global mixture over all bond objects of a molecule; there
is no recursion on the original or line-graph adjacency.

The AdaRMSNorm form ``x/rms(x) * rms(alpha*x + beta)`` (alpha0=0, beta0=1) is
taken from PPGT (arXiv:2504.12588); it is not claimed as new here.
"""

from __future__ import annotations

import copy
import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
_V2 = _SSUF / "zinc_recipe_trunk_v2" / "code"
for _p in (str(_HERE), str(_V1), str(_V2)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_models as M1  # noqa: E402
import ad2_models as A2  # noqa: E402
import gos_data as GD  # noqa: E402

WIDTH = 96
GLOBAL_DIM = D.GLOBAL_DIM
READER_HIDDEN = 128
TOKEN_DIM = 48
HEADS = 4
LAYERS = 3
REL_REL_DIM = 24
REL_HIDDEN = 48
FF_MULT = 2
SMALL_INIT = 1.0e-3
Z0_DIM = 3 * WIDTH + GLOBAL_DIM + 2 * WIDTH          # 513
READER_IN_CORE = Z0_DIM + 2 * TOKEN_DIM              # 609


class AdaRMSNorm(nn.Module):
    """N(x) = x/rms_eps(x) * rms_eps(alpha*x + beta); alpha0=0, beta0=1."""

    def __init__(self, d: int, eps: float = 1e-6):
        super().__init__()
        self.alpha = nn.Parameter(torch.zeros(d))
        self.beta = nn.Parameter(torch.ones(d))
        self.eps = float(eps)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        def rms(t: torch.Tensor) -> torch.Tensor:
            return torch.sqrt(t.pow(2).mean(dim=-1, keepdim=True) + self.eps)

        return x / rms(x) * rms(self.alpha * x + self.beta)


def _small_init(lin: nn.Linear, std: float = SMALL_INIT) -> nn.Linear:
    with torch.no_grad():
        lin.weight.normal_(0.0, std)
        lin.bias.zero_()
    return lin


class FFN(nn.Module):
    def __init__(self, d: int, mult: int = FF_MULT, dropout: float = 0.0):
        super().__init__()
        self.norm = AdaRMSNorm(d)
        self.fc1 = nn.Linear(d, mult * d)
        self.fc2 = nn.Linear(mult * d, d)
        self.dropout = float(dropout)

    def forward(self, h: torch.Tensor) -> torch.Tensor:
        x = F.silu(self.fc1(self.norm(h)))
        x = F.dropout(x, p=self.dropout, training=self.training)
        return h + F.dropout(self.fc2(x), p=self.dropout, training=self.training)


MIX_MODES = ("legacy", "dot", "l2", "gn")


class MixA(nn.Module):
    """Global content interaction over bond objects (relation bias + content gate).

    ``variant`` selects the parameter set (``full`` has the content gate, ``plain``
    does not).  ``mode`` selects only the *arithmetic* inside ``forward``; it adds,
    removes and reorders nothing, so all modes share one parameter tensor set:

    ``legacy``  the pre-erratum expression, verbatim:
                ``(q.k - 0.5||k||^2).unsqueeze(-1)/sqrt(d)``; the subtracted term is
                indexed by the *query*, so it is a per-row constant and the row
                softmax cancels it.  This is the operator the historical A runs
                actually trained; kept bit-for-bit for the frozen checkpoints.
    ``dot``     the reference operator that ``legacy`` is mathematically equal to:
                ``q.k/sqrt(d)`` with no key-energy term at all.
    ``l2``      the intended key-energy operator: the half-key-energy term is
                broadcast along the *key* axis (``unsqueeze(-2)``).
    ``gn``      ``dot`` scoring with the gated weights renormalised over keys.
    """

    def __init__(self, d: int = TOKEN_DIM, heads: int = HEADS, rel: int = REL_REL_DIM,
                 variant: str = "full", dropout: float = 0.0, mode: str = "legacy"):
        super().__init__()
        if variant not in ("full", "plain"):
            raise ValueError(f"unknown MixA variant {variant!r}")
        if mode not in MIX_MODES:
            raise ValueError(f"unknown MixA mode {mode!r}")
        self.mode = mode
        self.variant = variant
        self.d, self.heads = int(d), int(heads)
        self.hd = self.d // self.heads
        self.norm = AdaRMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = _small_init(nn.Linear(d, d))
        self.beta = nn.Linear(rel, self.heads)
        # "plain" control: ordinary normalised attention + relation bias, no
        # amplitude-sensitive term and no relation-conditioned content gate.
        self.gamma = nn.Linear(rel, self.heads) if variant == "full" else None
        self.dropout = float(dropout)
        self.ffn = FFN(d, dropout=dropout)

    def _attend(self, q, k, v, rel, mask):
        """The mixing arithmetic itself (queries/keys/values already projected).

        This is the only place reading ``mode``; ``mix`` below and every check go
        through it, so the checks exercise the production operator rather than a
        restatement of it.
        """
        score = torch.einsum("bhid,bhjd->bhij", q, k)       # [B,H,query,key]
        sqrt_d = math.sqrt(self.hd)
        if self.mode == "legacy":
            # historical expression, kept verbatim: ken is [B,H,key] and
            # unsqueeze(-1) makes it a per-query constant, i.e. a no-op.
            ken = 0.5 * k.pow(2).sum(-1)                     # [B,H,N] over keys
            score = (score - ken.unsqueeze(-1)) / sqrt_d
        elif self.mode == "l2":
            ken = 0.5 * k.pow(2).sum(-1)                     # [B,H,N] over keys
            score = (score - ken.unsqueeze(-2)) / sqrt_d     # broadcast on the key axis
        else:                                                # dot / gn
            score = score / sqrt_d
        score = score + self.beta(rel).permute(0, 3, 1, 2)
        valid = mask[:, None, None, :]
        score = score.masked_fill(~valid, -1e30)
        attn = torch.softmax(score, dim=-1)
        gate = None
        if self.variant == "full":
            gate = 1.0 + torch.tanh(self.gamma(rel)).permute(0, 3, 1, 2)
            if self.mode == "gn":
                w = attn * gate
                mass = w.sum(dim=-1, keepdim=True)
                ctx = torch.einsum("bhij,bhjd->bhid", w / mass, v)
            else:
                mass = (attn * gate).sum(dim=-1, keepdim=True)
                ctx = torch.einsum("bhij,bhij,bhjd->bhid", attn, gate, v)
        else:
            mass = attn.sum(dim=-1, keepdim=True)
            ctx = torch.einsum("bhij,bhjd->bhid", attn, v)
        return {"attn": attn, "gate": gate, "mass": mass, "ctx": ctx, "logits": score}

    def mix(self, h, rel, mask):
        """Project, then apply the mixing operator.  No train/eval branch lives in
        here, so training and evaluation share one forward."""
        B, N, _ = h.shape
        x = self.norm(h) * mask.unsqueeze(-1).to(h.dtype)
        qkv = self.qkv(x).view(B, N, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                     # [B,H,N,hd]
        out = self._attend(q, k, v, rel, mask)
        out.update({"q": q, "k": k, "v": v})
        return out

    def forward(self, h, rel, mask):
        B, N, _ = h.shape
        ctx = self.mix(h, rel, mask)["ctx"]
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, N, self.d)
        h = h + F.dropout(self.out(ctx), p=self.dropout, training=self.training)
        h = self.ffn(h)
        return h * mask.unsqueeze(-1).to(h.dtype)


class MixK(nn.Module):
    """Signed, non-competing global structural kernel over bond objects."""

    def __init__(self, d: int = TOKEN_DIM, rel: int = REL_REL_DIM, variant: str = "signed",
                 dropout: float = 0.0):
        super().__init__()
        if variant not in ("signed", "softmax"):
            raise ValueError(f"unknown MixK variant {variant!r}")
        self.variant = variant
        self.norm = AdaRMSNorm(d)
        self.kernel = nn.Sequential(nn.Linear(rel, 32), nn.SiLU(), nn.Linear(32, d))
        # V_l is the residual write itself; its last layer is small-initialised.
        self.value = nn.Sequential(nn.Linear(d, d), nn.SiLU(), _small_init(nn.Linear(d, d)))
        self.dropout = float(dropout)
        self.ffn = FFN(d, dropout=dropout)

    def forward(self, h, rel, mask, inv_nbonds: float):
        x = self.norm(h) * mask.unsqueeze(-1).to(h.dtype)
        raw = self.kernel(rel)
        if self.variant == "signed":
            kk = torch.tanh(raw) * mask[:, None, :, None].to(rel.dtype)
            scale = float(inv_nbonds)
        else:
            # control: per-channel row softmax -> positive, competing weights
            raw = raw.masked_fill(~mask[:, None, :, None], -1e30)
            kk = torch.softmax(raw, dim=1)
            scale = 1.0
        vv = self.value(x)
        ctx = torch.einsum("bijc,bjc->bic", kk, vv) * scale
        h = h + F.dropout(ctx, p=self.dropout, training=self.training)
        h = self.ffn(h)
        return h * mask.unsqueeze(-1).to(h.dtype)


class Backbone(nn.Module):
    """P3 encoders + DCC global/local composers + bond-object global core."""

    def __init__(self, norm: Mapping[str, Any], route: str, *, dropout: float = 0.0,
                 mean_n_bonds: float = 1.0, reader_hidden: int = READER_HIDDEN,
                 mix_mode: str = "legacy"):
        super().__init__()
        if route not in ("C0", "A", "K", "Asm", "Ksm", "N0"):
            raise ValueError(f"unknown route {route!r}")
        if mix_mode not in MIX_MODES:
            raise ValueError(f"unknown mix mode {mix_mode!r}")
        self.mix_mode = mix_mode
        self.base_route = "A" if route in ("Asm", "N0") else ("K" if route == "Ksm" else route)
        self.route = route
        self.encoders = M1.P3Encoders(dropout=dropout)
        self.group_composer = M1.MLP(WIDTH + 2 * WIDTH + 1 + D.BOND_FEAT_DIM, 160, WIDTH, 1, dropout)
        self.local_composer = M1.MLP(WIDTH + 2 * WIDTH + 1 + D.BOND_FEAT_DIM, 80, WIDTH, 1, dropout)
        self.mean_n_bonds = float(mean_n_bonds)
        if route == "C0":
            self.reader = M1.MLP(Z0_DIM, int(reader_hidden), 1, n_hidden=2, dropout=dropout)
        elif route == "N0":
            # control: bond-object *content* reaches the reader, but no global
            # mixture over bond objects and no relation encoder exist.
            self.bond_tok = nn.Linear(2 * WIDTH, TOKEN_DIM)
            self.layers = nn.ModuleList()
            self.reader = M1.MLP(READER_IN_CORE, int(reader_hidden), 1, n_hidden=2, dropout=dropout)
        else:
            self.bond_tok = nn.Linear(2 * WIDTH, TOKEN_DIM)
            self.rel_enc = nn.Sequential(
                nn.Linear(GD.REL_DIM, REL_HIDDEN), nn.SiLU(), nn.Linear(REL_HIDDEN, REL_REL_DIM)
            )
            if self.base_route == "A":
                variant = "full" if route == "A" else "plain"
                self.layers = nn.ModuleList([MixA(variant=variant, dropout=dropout,
                                                  mode=mix_mode)
                                             for _ in range(LAYERS)])
            else:
                variant = "signed" if route == "K" else "softmax"
                self.layers = nn.ModuleList([MixK(variant=variant, dropout=dropout)
                                             for _ in range(LAYERS)])
            self.reader = M1.MLP(READER_IN_CORE, int(reader_hidden), 1, n_hidden=2, dropout=dropout)
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))

    # -- local carrier ---------------------------------------------------------
    def group_to_bonds(self, w, batch):
        n_graphs = int(batch["n_graphs"])
        jb = batch["junc_bond"]
        n_b = w.new_zeros(int(batch["bond_feat"].shape[0]))
        n_b.index_add_(0, jb.reshape(-1), torch.ones(jb.numel(), device=w.device, dtype=w.dtype))
        w_sum = w.new_zeros((n_graphs, w.shape[1])).index_add_(0, batch["junc_batch"], w)
        w2_sum = w.new_zeros((n_graphs, w.shape[1])).index_add_(0, batch["junc_batch"], w * w)
        cnt = torch.bincount(batch["junc_batch"], minlength=n_graphs).clamp(min=1).to(w.dtype)
        mean_w = w_sum / cnt.unsqueeze(1)
        mean_w2 = w2_sum / cnt.unsqueeze(1)
        bond_batch = batch["bond_batch"]
        return n_b.unsqueeze(1) * mean_w[bond_batch], n_b.unsqueeze(1) * mean_w2[bond_batch], n_b

    def local_grouping(self, w, batch):
        jb = batch["junc_bond"]
        n_b = w.new_zeros(int(batch["bond_feat"].shape[0]))
        n_b.index_add_(0, jb.reshape(-1), torch.ones(jb.numel(), device=w.device, dtype=w.dtype))
        W = w.new_zeros((int(batch["bond_feat"].shape[0]), w.shape[1]))
        V = torch.zeros_like(W)
        W.index_add_(0, jb[:, 0], w)
        W.index_add_(0, jb[:, 1], w)
        w2 = w * w
        V.index_add_(0, jb[:, 0], w2)
        V.index_add_(0, jb[:, 1], w2)
        return W, V, n_b

    # -- forward ---------------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        a, b, w = self.encoders(batch)
        Wg, Vg, n_b = self.group_to_bonds(w, batch)
        c_glob = self.group_composer(
            torch.cat([b, Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
        )
        Wl, Vl, _ = self.local_grouping(w, batch)
        c_loc = self.local_composer(
            torch.cat([b, Wl, Vl, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
        )
        B = int(batch["n_graphs"])
        bond_batch = batch["bond_batch"]
        z = c_glob.new_zeros((B, c_glob.shape[1])).index_add_(0, bond_batch, c_glob)
        zl = c_loc.new_zeros((B, c_loc.shape[1])).index_add_(0, bond_batch, c_loc)
        zl2 = c_loc.new_zeros((B, c_loc.shape[1])).index_add_(0, bond_batch, c_loc * c_loc)
        a_sum = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a)
        a_sq = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global"] - self.global_mean) / self.global_scale
        z0 = torch.cat([z, a_sum, a_sq, g, zl, zl2], dim=1)
        if self.route == "C0":
            return self.reader(z0).view(-1)

        E = c_glob.shape[0]
        bb = int(batch["n_bonds_pad"])
        h0 = self.bond_tok(torch.cat([c_glob, c_loc], dim=1))          # [E,48]
        mask = batch["rel_mask"]
        h = h0.new_zeros((B, bb, TOKEN_DIM))
        if E:
            h[bond_batch, batch["bond_local"]] = h0
        rel = self.rel_enc(batch["rel_feat"]) if self.route != "N0" else None
        if bb and self.route != "N0":
            for layer in self.layers:
                if self.base_route == "A":
                    h = layer(h, rel, mask)
                else:
                    h = layer(h, rel, mask, 1.0 / self.mean_n_bonds)
        hv = h[bond_batch, batch["bond_local"]] if E else h0.new_zeros((0, TOKEN_DIM))
        h_sum = h.new_zeros((B, TOKEN_DIM)).index_add_(0, bond_batch, hv)
        h_sq = h.new_zeros((B, TOKEN_DIM)).index_add_(0, bond_batch, hv * hv)
        nb = h.new_full((B,), self.mean_n_bonds)
        h_sum = h_sum / nb.unsqueeze(1)
        h_sq = h_sq / nb.unsqueeze(1)
        return self.reader(torch.cat([z0, h_sum, h_sq], dim=1)).view(-1)


def _param_count(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))


def build_model(route: str, norm: Mapping[str, Any], seed: int, device,
                *, mean_n_bonds: float, dropout: float = 0.0,
                reader_hidden: int = READER_HIDDEN,
                mix_mode: str = "legacy") -> tuple[Backbone, dict]:
    torch.manual_seed(int(seed))
    model = Backbone(norm, route, dropout=float(dropout), mean_n_bonds=mean_n_bonds,
                     reader_hidden=int(reader_hidden), mix_mode=str(mix_mode))
    ssl_path = C.P3_SSL_S0 if int(seed) == 0 else C.P3_SSL_S1
    ssl_info = A2._load_encoders(model, ssl_path, torch.device("cpu"))
    n = _param_count(model)
    if n > 500_000:
        raise RuntimeError(f"{route}: {n} parameters exceeds the 500k cap")
    info = {
        "route": route,
        "mix_mode": str(model.mix_mode),
        "seed": int(seed),
        "parameters": n,
        "ssl": ssl_info,
        "initial_state_sha256": C.state_hash(model),
        "mean_train_n_bonds": float(mean_n_bonds),
        "dropout": float(dropout),
        "reader_hidden": int(reader_hidden),
        "parameters_breakdown": {
            "encoders": _param_count(model.encoders),
            "group_composer": _param_count(model.group_composer),
            "local_composer": _param_count(model.local_composer),
            "core": _param_count(model.layers) if route != "C0" else 0,
            "bond_token": _param_count(model.bond_tok) if route != "C0" else 0,
            "rel_encoder": _param_count(model.rel_enc) if route not in ("C0", "N0") else 0,
            "reader": _param_count(model.reader),
        },
    }
    model.to(device)
    return model, info

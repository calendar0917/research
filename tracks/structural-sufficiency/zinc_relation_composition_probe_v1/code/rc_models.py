"""Fragment-state-conditioned relation composition — the D/S probe.

Round: ``zinc-relation-composition-probe-v1``.  Official test is never loaded.

What this model is
------------------

The B baseline (``dcc_models`` L arm) keeps its P3 atom/bond/junction encoders,
its ``group_composer`` (global-mean ownership) and its organised
``local_composer`` (real bond-junction ownership), all fine-tuned.  Its own
reader is **removed** (not registered), and replaced by one head that reads a
new 4-layer bond-token composition core.

Per undirected physical bond ``b`` (one token, not two directed ones):

    c_g[b] = group_composer([b, Wg, Vg, log1p n_b, bond_feat])      # 96
    c_l[b] = local_composer([b, Wl, Vl, log1p n_b, bond_feat])      # 96
    h_b^0  = SiLU(W_h [c_g_b, c_l_b] + bias) + W_abs R_bb           # d=48

``R`` is the bond-lifted random-walk relation from ``rc_data``:
``R_bc[k] = (M P^k M^T)_bc`` with the endpoint-incidence ``M`` scaled by
``1/sqrt(2)`` and ``P = diag(deg)^-1 A`` on the raw graph.  The initial relation
state and the per-pair relation encoding are linear in it, no bias:

    e_bc^0 = W_rel R_bc
    (W_rel: 21 -> d,  W_abs: 21 -> d)

Each layer runs global (block-diagonal, no adjacency mask) attention over the
bonds of one molecule, in the audited ``zps_grit_dense.DenseGritAttention``
layout with the GRIT edge-enhanced value path:

    t_bc    = Q_l h_b + K_l h_c + c_l            (c_l: learnable d-vec, init 1)
    E_w/E_b = split(Linear_l(e_bc))              (d -> 2d per head)
    xi_bc   = ReLU( signed_sqrt(t_bc * E_w_bc) + E_b_bc )
    alpha   = softmax_c( clamp(Aw_l . xi, -5, 5) )
    u_b     = sum_c alpha [ V_l h_c + Ve_l xi_bc ]
    h       -> degree scaler [1, log1p(line_deg_b)] -> O_h -> +residual -> BN
    h       -> FFN(d -> 2d -> d, SiLU) -> +residual -> BN

``signed_sqrt(x) = sign(x) * sqrt(|x| + 1e-6)`` is the numerical adaptation of
upstream's ``sqrt(relu(s)) - sqrt(relu(-s))`` so the gradient at 0 is finite;
both arms use it.  BatchNorm counts real tokens / real pairs only.

The only D/S difference — the write into the next layer's relation state
--------------------------------------------------------------------------------

Both arms use the *same* real ``xi_bc`` (h-conditioned) for the token update
above, and have identical parameters.  They differ only in what is written to
``e^{l+1}``:

    D:  e^{l+1} = BN_e( e^l + O_e,l xi_bc )                       # h-conditioned
    S:  e^{l+1} = BN_e( e^l + O_e,l xi_static_bc )
        xi_static_bc = ReLU( signed_sqrt(c_l * E_w_bc) + E_b_bc )  # R-only

``S`` is not "a fixed relation": it still has multi-layer attention, a learnable
relation encoder, a learnable per-layer relation update and the full edge value
path; its relation memory does evolve with depth, but under fixed weights it is
a function of ``R`` alone and never receives ``h``.  ``S`` does **not** use
``h.detach()``: detaching would keep the forward-context dependence and answer a
different question.

Only the first 3 layers write ``e``; layer 4 updates ``h`` only and registers no
``O_e``/``BN_e``, because nothing would read them.

Readout (the single head)
-------------------------

    z = [ sum_b h_b / mean_bonds, sum_b h_b^2 / mean_bonds,
          sum_i a_i / mean_atoms, sum_i a_i^2 / mean_atoms,
          standardized global33, log1p n_bonds ]
    y = MLP(z -> 128 -> 128 -> 1, SiLU)

Divisors are *fixed train-split means* (not per-molecule counts), so inter-
molecule amplitude differences survive; no per-graph LayerNorm.
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
_SS = _HERE.parents[1]
_V1 = _SS / "zinc_arch_discovery_v1" / "code"
_AD2 = _SS / "zinc_recipe_trunk_v2" / "code"
_DCC = _SS / "zinc_dual_channel_composer_v1" / "code"
for _p in (str(_HERE), str(_V1), str(_AD2), str(_DCC)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_models as M1  # noqa: E402
import ad2_models as A2  # noqa: E402
import dcc_models  # noqa: E402

KSTEPS = 21
WIDTH = 96  # P3 composer / encoder width
D_HID = 48
N_HEADS = 4
N_LAYERS = 4
CLAMP = 5.0
ATTN_DROPOUT = 0.2
READER_HIDDEN = 128
#: second fixed RNG stream for the *new* modules (same convention as dcc)
NEW_MODULE_SEED_OFFSET = 90731


def signed_sqrt(x: torch.Tensor) -> torch.Tensor:
    return torch.sign(x) * torch.sqrt(torch.abs(x) + 1e-6)


def _bn_masked(bn: nn.Module, values: torch.Tensor, flat_mask: torch.Tensor) -> torch.Tensor:
    """BatchNorm over real entries only; padding stays exactly zero.

    Same pattern as the audited ``DenseGritLayer._bn_masked``: only the selected
    rows enter the BN (so running stats ignore padding), and the unselected rows
    of the returned tensor are zeros.
    """
    shape = values.shape
    flat = values.reshape(-1, shape[-1])
    out = torch.zeros_like(flat)
    out[flat_mask] = bn(flat[flat_mask])
    return out.view(shape)


class RCAttention(nn.Module):
    """GRIT-style edge-enhanced attention over bond tokens of one molecule."""

    def __init__(self, dim: int = D_HID, heads: int = N_HEADS,
                 attn_dropout: float = ATTN_DROPOUT, clamp: float = CLAMP):
        super().__init__()
        if dim % heads:
            raise ValueError("dim must be divisible by heads")
        self.dim = int(dim)
        self.heads = int(heads)
        self.hd = self.dim // self.heads
        self.clamp = float(clamp)
        self.dropout = nn.Dropout(attn_dropout)
        # Q/K/V carry no bias: the additive per-layer vector is ``c`` (init 1).
        self.Q = nn.Linear(dim, dim, bias=False)
        self.K = nn.Linear(dim, dim, bias=False)
        self.V = nn.Linear(dim, dim, bias=False)
        self.E = nn.Linear(dim, dim * 2, bias=True)
        self.Aw = nn.Parameter(torch.empty(self.hd, heads, 1))
        self.VeRow = nn.Parameter(torch.empty(self.hd, heads, self.hd))
        self.c = nn.Parameter(torch.ones(dim))
        # same init as the audited DenseGritAttention: xavier_normal_ on the
        # projections, default (uniform) init on the E bias.
        for lin in (self.Q, self.K, self.V, self.E):
            nn.init.xavier_normal_(lin.weight)
        nn.init.xavier_normal_(self.Aw)
        nn.init.xavier_normal_(self.VeRow)

    def _xi(self, t: torch.Tensor, e_w: torch.Tensor, e_b: torch.Tensor) -> torch.Tensor:
        return F.relu(signed_sqrt(t * e_w) + e_b)

    def forward(self, h: torch.Tensor, e: torch.Tensor, pair_mask: torch.Tensor):
        """Return ``(u, xi, xi_static)``.

        ``u`` is the attention output for the token update, ``xi`` the real
        h-conditioned edge activation (used for the token update in **both**
        arms), ``xi_static`` the relation-only activation (the S arm's write).
        """
        B, N, dim = h.shape
        H, d = self.heads, self.hd
        Q = self.Q(h).view(B, N, H, d)
        K = self.K(h).view(B, N, H, d)
        V = self.V(h).view(B, N, H, d)
        E = self.E(e).view(B, N, N, H, 2, d)
        e_w, e_b = E[..., 0, :], E[..., 1, :]
        c = self.c.view(1, 1, 1, H, d).expand(B, N, N, H, d)
        t = K.unsqueeze(1) + Q.unsqueeze(2) + c
        xi = self._xi(t, e_w, e_b)
        xi_static = self._xi(c, e_w, e_b)
        logits = torch.einsum("bijhd,dhc->bijhc", xi, self.Aw)
        if self.clamp is not None:
            logits = logits.clamp(-self.clamp, self.clamp)
        logits = logits.masked_fill(~pair_mask[:, :, :, None, None], -1e9)
        attn = self.dropout(torch.softmax(logits, dim=2)).squeeze(-1)
        wV = torch.einsum("bijh,bjhd->bihd", attn, V)
        rowV = torch.einsum("bijhd,bijh->bihd", xi, attn)
        rowV = torch.einsum("bihd,dhc->bihc", rowV, self.VeRow)
        u = (wV + rowV).reshape(B, N, dim)
        wE = xi.reshape(B, N, N, dim)
        return u, wE, xi_static.reshape(B, N, N, dim)


class RCLayer(nn.Module):
    """One layer: attention -> degree scaler -> O_h -> BN -> FFN -> BN.

    ``update_relation=False`` is used on the last layer so no unread ``O_e`` /
    ``BN_e`` parameters are registered.
    """

    def __init__(self, dim: int = D_HID, heads: int = N_HEADS, dropout: float = 0.0,
                 attn_dropout: float = ATTN_DROPOUT, clamp: float = CLAMP,
                 update_relation: bool = True, arm: str = "D"):
        super().__init__()
        self.dim = int(dim)
        self.dropout = float(dropout)
        self.arm = arm
        self.update_relation = bool(update_relation)
        self.attention = RCAttention(dim, heads, attn_dropout, clamp)
        # plain nn.Linear defaults, exactly as upstream DenseGritLayer
        self.O_h = nn.Linear(dim, dim)
        if self.update_relation:
            self.O_e = nn.Linear(dim, dim)
            self.bn_e = nn.BatchNorm1d(dim, eps=1e-5, momentum=0.1)
        self.deg_coef = nn.Parameter(torch.zeros(1, dim, 2))
        nn.init.xavier_normal_(self.deg_coef)
        self.bn_h1 = nn.BatchNorm1d(dim, eps=1e-5, momentum=0.1)
        self.ffn1 = nn.Linear(dim, dim * 2)
        self.ffn2 = nn.Linear(dim * 2, dim)
        self.bn_h2 = nn.BatchNorm1d(dim, eps=1e-5, momentum=0.1)

    def forward(self, h, e, pair_mask, token_mask, log_line_deg):
        token_flat = token_mask.reshape(-1)
        u, wE, xi_static = self.attention(h, e, pair_mask)
        x = F.dropout(u, self.dropout, training=self.training)
        x = torch.stack([x, x * log_line_deg.unsqueeze(-1)], dim=-1)
        x = (x * self.deg_coef).sum(dim=-1)
        x = self.O_h(x)
        h_res = h + x
        e_src = wE if self.arm == "D" else xi_static
        if self.update_relation:
            e_next = e + self.O_e(e_src)
            e_next = _bn_masked(self.bn_e, e_next, pair_mask.reshape(-1))
        else:
            e_next = e
        h_out = _bn_masked(self.bn_h1, h_res, token_flat)
        f = self.ffn2(F.silu(self.ffn1(h_out)))
        h_out = _bn_masked(self.bn_h2, h_out + f, token_flat)
        return h_out, e_next


class RelationCompositionCore(nn.Module):
    """Bond tokens + 4-layer relation composition, with the relation encoders."""

    def __init__(self, *, width: int = WIDTH, dim: int = D_HID, heads: int = N_HEADS,
                 layers: int = N_LAYERS, ksteps: int = KSTEPS, dropout: float = 0.0,
                 attn_dropout: float = ATTN_DROPOUT, arm: str = "D"):
        super().__init__()
        if arm not in ("D", "S"):
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        self.dim = int(dim)
        self.w_h = nn.Linear(2 * width, dim)
        # no bias on the relation encoders (pure linear maps of R)
        self.w_abs = nn.Linear(ksteps, dim, bias=False)
        self.w_rel = nn.Linear(ksteps, dim, bias=False)
        nn.init.xavier_normal_(self.w_h.weight)
        nn.init.xavier_normal_(self.w_abs.weight)
        nn.init.xavier_normal_(self.w_rel.weight)
        self.layers = nn.ModuleList([
            RCLayer(dim, heads, dropout, attn_dropout, CLAMP,
                    update_relation=(i < layers - 1), arm=arm)
            for i in range(layers)
        ])

    def forward(self, c_g, c_l, batch):
        rel = batch["rel_R"]                              # [B,Emax,Emax,K]
        mask = batch["bond_mask"]                          # [B,Emax]
        pair_mask = mask[:, :, None] & mask[:, None, :]    # [B,Emax,Emax]
        log_deg = batch["log_line_deg"]                    # [B,Emax]
        r_diag = torch.diagonal(rel, dim1=1, dim2=2).transpose(1, 2)  # [B,Emax,K]
        h0 = F.silu(self.w_h(torch.cat([c_g, c_l], dim=-1))) + self.w_abs(r_diag)
        h0 = h0 * mask.unsqueeze(-1).to(h0.dtype)
        e0 = self.w_rel(rel) * pair_mask.unsqueeze(-1).to(h0.dtype)
        # token index of bond b for molecule i is local; scatter c_g/c_l to padded
        h = h0
        e = e0
        for layer in self.layers:
            h, e = layer(h, e, pair_mask, mask, log_deg)
        return h, e


class RelationCompositionProbe(nn.Module):
    """B's encoders + both B composers + the relation core + one new head."""

    def __init__(self, norm: Mapping[str, Any], *, arm: str, stats: Mapping[str, float],
                 dropout: float = 0.0, dim: int = D_HID, heads: int = N_HEADS,
                 layers: int = N_LAYERS, attn_dropout: float = ATTN_DROPOUT):
        super().__init__()
        self.arm = arm
        self.encoders = M1.P3Encoders(dropout=dropout)
        self.group_composer = M1.MLP(
            WIDTH + 2 * WIDTH + 1 + D.BOND_FEAT_DIM, 160, WIDTH, 1, dropout)
        self.local_composer = M1.MLP(
            WIDTH + 2 * WIDTH + 1 + D.BOND_FEAT_DIM, 80, WIDTH, 1, dropout)
        self.core = RelationCompositionCore(
            width=WIDTH, dim=dim, heads=heads, layers=layers, ksteps=KSTEPS,
            dropout=dropout, attn_dropout=attn_dropout, arm=arm)
        reader_in = 2 * dim + 2 * WIDTH + D.GLOBAL_DIM + 1
        self.reader = M1.MLP(reader_in, READER_HIDDEN, 1, n_hidden=2, dropout=dropout)
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        self.register_buffer("mean_bonds", torch.as_tensor(float(stats["mean_bonds"]), dtype=torch.float32))
        self.register_buffer("mean_atoms", torch.as_tensor(float(stats["mean_atoms"]), dtype=torch.float32))

    # -- B's one-shot groupings, verbatim ------------------------------------
    @staticmethod
    def _ownership(w, batch):
        jb = batch["junc_bond"]
        n_b = torch.zeros(int(batch["bond_feat"].shape[0]), device=w.device, dtype=w.dtype)
        n_b.index_add_(0, jb.reshape(-1), torch.ones(jb.numel(), device=w.device, dtype=w.dtype))
        return jb, n_b

    def group_to_bonds(self, w, batch):
        """Global-mean ownership (P3 Null grouping), as in B."""
        n_graphs = int(batch["n_graphs"])
        jb, n_b = self._ownership(w, batch)
        w_sum = w.new_zeros((n_graphs, w.shape[1])).index_add_(0, batch["junc_batch"], w)
        w2_sum = w.new_zeros((n_graphs, w.shape[1])).index_add_(0, batch["junc_batch"], w * w)
        cnt = torch.bincount(batch["junc_batch"], minlength=n_graphs).clamp(min=1).to(w.dtype)
        bond_batch = batch["bond_batch"]
        scale = n_b.unsqueeze(1)
        return (scale * (w_sum / cnt.unsqueeze(1))[bond_batch],
                scale * (w2_sum / cnt.unsqueeze(1))[bond_batch], n_b)

    def local_grouping(self, w, batch):
        """Real bond-junction ownership, as in B's L arm."""
        jb, n_b = self._ownership(w, batch)
        nb_bonds = int(batch["bond_feat"].shape[0])
        W = torch.zeros((nb_bonds, w.shape[1]), device=w.device, dtype=w.dtype)
        V = torch.zeros_like(W)
        W.index_add_(0, jb[:, 0], w)
        W.index_add_(0, jb[:, 1], w)
        w2 = w * w
        V.index_add_(0, jb[:, 0], w2)
        V.index_add_(0, jb[:, 1], w2)
        return W, V, n_b

    @staticmethod
    def _to_padded(flat, batch, dim, n_graphs, n_max):
        out = flat.new_zeros((n_graphs, n_max, dim))
        out[batch["bond_batch"], batch["bond_local"]] = flat
        return out

    def forward(self, batch):
        a, b, w = self.encoders(batch)
        Wg, Vg, n_b = self.group_to_bonds(w, batch)
        c_g = self.group_composer(
            torch.cat([b, Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1))
        Wl, Vl, _ = self.local_grouping(w, batch)
        c_l = self.local_composer(
            torch.cat([b, Wl, Vl, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1))
        n_graphs = int(batch["n_graphs"])
        n_max = int(batch["rel_R"].shape[1])
        c_g_pad = self._to_padded(c_g, batch, c_g.shape[1], n_graphs, n_max)
        c_l_pad = self._to_padded(c_l, batch, c_l.shape[1], n_graphs, n_max)
        if int(batch["bond_mask"].sum()) == 0:
            # no real bond anywhere in this batch: skip the relation core but
            # keep the atom/global part of the read-out (molecule not dropped)
            h = c_g_pad.new_zeros((n_graphs, n_max, self.core.dim))
        else:
            h, _e = self.core(c_g_pad, c_l_pad, batch)
        mask = batch["bond_mask"].unsqueeze(-1).to(h.dtype)
        h = h * mask
        cnt_b = self.mean_bonds.clamp(min=1.0)
        cnt_a = self.mean_atoms.clamp(min=1.0)
        z_h = h.sum(dim=1) / cnt_b
        z_h2 = (h * h).sum(dim=1) / cnt_b
        a_sum = a.new_zeros((n_graphs, a.shape[1])).index_add_(0, batch["atom_batch"], a)
        a_sq = a.new_zeros((n_graphs, a.shape[1])).index_add_(0, batch["atom_batch"], a * a)
        z_a = a_sum / cnt_a
        z_a2 = a_sq / cnt_a
        g = (batch["global"] - self.global_mean) / self.global_scale
        nb = batch["n_bonds"].unsqueeze(1)
        z = torch.cat([z_h, z_h2, z_a, z_a2, g, torch.log1p(nb)], dim=1)
        return self.reader(z).view(-1)

    def relation_state_after_forward(self):
        """Last layer's relation memory is not returned by ``forward``; the
        diagnostics call ``core`` directly (see ``rc_checks.py``)."""
        raise NotImplementedError

    def set_arm(self, arm: str) -> None:
        """Switch the relation-write rule in place (D <-> S).

        Only the rule changes; every parameter tensor is untouched.  This is the
        fixed-weight intervention of the round.
        """
        if arm not in ("D", "S"):
            raise ValueError(f"unknown arm {arm!r}")
        self.arm = arm
        self.core.arm = arm
        for layer in self.core.layers:
            layer.arm = arm


def composer_pad(model: "RelationCompositionProbe", batch) -> tuple[torch.Tensor, torch.Tensor]:
    """B's two composer outputs, scattered into the padded [B, Emax, 96] layout."""
    a, b, w = model.encoders(batch)
    Wg, Vg, n_b = model.group_to_bonds(w, batch)
    c_g = model.group_composer(torch.cat([b, Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], 1))
    Wl, Vl, _ = model.local_grouping(w, batch)
    c_l = model.local_composer(torch.cat([b, Wl, Vl, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], 1))
    n_graphs, n_max = int(batch["n_graphs"]), int(batch["rel_R"].shape[1])
    cg = c_g.new_zeros((n_graphs, n_max, c_g.shape[1]))
    cl = c_l.new_zeros((n_graphs, n_max, c_l.shape[1]))
    cg[batch["bond_batch"], batch["bond_local"]] = c_g
    cl[batch["bond_batch"], batch["bond_local"]] = c_l
    return cg, cl


def layer0_inputs(model: "RelationCompositionProbe", batch, c_g, c_l):
    """``(h0, e0, pair_mask)`` for the first layer."""
    rel = batch["rel_R"]
    mask = batch["bond_mask"]
    pm = mask[:, :, None] & mask[:, None, :]
    r_diag = torch.diagonal(rel, dim1=1, dim2=2).transpose(1, 2)
    h = F.silu(model.core.w_h(torch.cat([c_g, c_l], dim=-1))) + model.core.w_abs(r_diag)
    h = h * mask.unsqueeze(-1)
    e = model.core.w_rel(rel) * pm.unsqueeze(-1)
    return h, e, pm


def _load_shared_from_b(model: RelationCompositionProbe, dual: nn.Module) -> dict:
    """Copy B's encoders + both composers, tensor by tensor."""
    src = {}
    for name in ("encoders", "group_composer", "local_composer"):
        sub = getattr(dual, name)
        for k, v in sub.state_dict().items():
            src[f"{name}.{k}"] = v.detach().cpu().clone()
    dst = {k: v for k, v in model.state_dict().items()
           if k.startswith(("encoders.", "group_composer.", "local_composer."))}
    if set(dst) != set(src):
        raise RuntimeError(f"shared-module key mismatch: {set(dst) ^ set(src)}")
    model.load_state_dict(src, strict=False)
    return {k: float(torch.sum(v.to(torch.float64) ** 2)) for k, v in src.items()}


def build_initial_state(norm: Mapping[str, Any], seed: int, *,
                        stats: Mapping[str, float], arm: str) -> tuple[RelationCompositionProbe, dict]:
    """Shared initial state for one arm.

    B's encoders/composers come from ``dcc_models.build_initial_state`` (so the
    SSL encoder replay and both composer initialisations are byte-identical to
    the B baseline); the new modules are drawn from the seed's fixed second RNG
    stream.  The arm flag is the only difference, so D and S have identical
    parameter keys, shapes and tensors.
    """
    dual, binfo = dcc_models.build_initial_state(norm, seed)
    torch.manual_seed(int(seed) + NEW_MODULE_SEED_OFFSET)
    model = RelationCompositionProbe(norm, arm=arm, stats=stats, dropout=0.0)
    shared = _load_shared_from_b(model, dual)
    info = {
        "arm": arm,
        "seed": int(seed),
        "ssl": binfo["ssl"],
        "b_shared_initialization": "dcc_models.build_initial_state",
        "b_baseline_parameters": binfo["parameters"],
        "new_module_rng_seed": int(seed) + NEW_MODULE_SEED_OFFSET,
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "initial_state_sha256": C.state_hash(model),
        "shared_module_l2": shared,
    }
    return model, info


def build_model(candidate: str, norm: Mapping[str, Any], seed: int, device,
                *, arm: str, stats: Mapping[str, float]):
    """Builder compatible with ``ad_train.train`` (``candidate`` is a label)."""
    shared, base_info = build_initial_state(norm, seed, stats=stats, arm=arm)
    model = copy.deepcopy(shared)
    model.to(device)
    info = dict(base_info)
    info.update({
        "parameters": int(sum(p.numel() for p in model.parameters())),
        "requires_grad_true": sum(1 for p in model.parameters() if p.requires_grad),
        "parameter_tensors": sum(1 for p in model.parameters()),
        "breakdown": {
            name: int(sum(p.numel() for p in mod.parameters()))
            for name, mod in (("encoders", model.encoders),
                              ("group_composer", model.group_composer),
                              ("local_composer", model.local_composer),
                              ("core", model.core),
                              ("reader", model.reader))
        },
        "buffers": int(sum(b.numel() for b in model.buffers())),
    })
    return model, info

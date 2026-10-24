"""ESOC-v1 model: explicit bond objects with real local edge correspondence.

Token set
---------
One token per undirected physical bond ``{u,v}``.  Its content is produced once,
from the *typed induced* structure on ``P_b = {u,v} U N(u) U N(v)``:

    h_b = 0.5 * ( F(P_b, u->v) + F(P_b, v->u) )

so the token is exactly invariant to flipping the root bond direction.  ``F`` is
a shared one-shot pairwise deep-set:

    z_i        = AtomEmb(type_i) + RoleEmb((dist(i,u), dist(i,v)))
    node_sum   = sum_i phi_n(z_i)                    node_sq = sum_i phi_n(z_i)^2
    p_ij       = [z_i+z_j, |z_i-z_j|, z_i*z_j, edge_ij]
    pair_sum   = sum_{i<j} phi_p(p_ij)               pair_sq = sum_{i<j} phi_p(p_ij)^2
    h_b_raw    = LN(MLP([node_sum, node_sq, pair_sum, pair_sq, log1p |P_b|, log1p e_b]))

``edge_ij`` is the 4-way class (non-edge / single / double / triple) of the real
pair ``i<j`` (arm REAL) or the object-level class-frequency vector
``p_edge = mean_{i<j} one_hot(edge_ij)`` broadcast to every pair (arm NULL).
Both arms have identical parameters and identical initialisation.

There is no recursion: no token is ever re-injected along the local adjacency.
The only interaction across objects is full attention over the bond tokens of the
same molecule, biased by a shared static RRWP stem.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F

N_ATOM = 21
N_ROLE = 16
N_EDGE = 4
KSTEPS = 21
ROLE_BASE = 4          # role id digits are base (ROLE_CAP + 1) = 4

D_MODEL = 64
N_HEADS = 8
N_LAYERS = 12
FFN_MULT = 2
ATTN_DROPOUT = 0.2
DROP_PATH = 0.1
D_Z = 32
D_PHI = 64
AGGR_HIDDEN = 96
PE_HIDDEN = 64
PE_FREQS = (1.0, 2.0, 4.0)
ADA_RANK = 4
READOUT_HIDDEN = 128
RMS_EPS = 1e-6


# ---------------------------------------------------------------------------
# static positional relation: lifted RRWP on physical bonds
# ---------------------------------------------------------------------------
def rrwp(bonds: torch.Tensor, bond_mask: torch.Tensor, n_atoms: torch.Tensor,
         ksteps: int = KSTEPS, n_max: int | None = None) -> torch.Tensor:
    """``R[b,c,k] = (M P^k M^T)[b,c]`` for every molecule in the batch.

    ``bonds``       [B, E, 2]  physical bonds (molecule-local atom ids, u<v)
    ``bond_mask``   [B, E]     real bond
    ``n_atoms``     [B]        atoms per molecule
    ``M`` has 1/sqrt(2) on each endpoint, ``P = diag(deg)^-1 A`` (unweighted,
    label-free).  Padded slots contribute exactly zero.
    """
    B, E, _ = bonds.shape
    dev = bonds.device
    n = int(n_max) if n_max else int(n_atoms.max().item())
    if n < 1:
        n = 1
    w = bond_mask.to(torch.float32)
    a = torch.zeros(B, n * n, dtype=torch.float32, device=dev)
    u = bonds[..., 0] * n + bonds[..., 1]
    v = bonds[..., 1] * n + bonds[..., 0]
    a = a.scatter_add(1, u.reshape(B, -1), w)
    a = a.scatter_add(1, v.reshape(B, -1), w)
    a = a.view(B, n, n)
    deg = a.sum(-1, keepdim=True)
    p = a / deg.clamp(min=1.0)
    m = torch.zeros(B, E, n, dtype=torch.float32, device=dev)
    half = 1.0 / math.sqrt(2.0)
    m = m.scatter_add(2, bonds[..., 0].unsqueeze(-1), (half * w).unsqueeze(-1))
    m = m.scatter_add(2, bonds[..., 1].unsqueeze(-1), (half * w).unsqueeze(-1))
    mt = m.transpose(1, 2)
    eye = torch.eye(n, dtype=torch.float32, device=dev).unsqueeze(0).expand(B, n, n)
    out = []
    q = eye
    for _ in range(ksteps):
        out.append(torch.bmm(torch.bmm(m, q), mt))
        q = torch.bmm(q, p)
    return torch.stack(out, dim=-1)


def rrwp_reference(edge_index, n: int, ksteps: int = KSTEPS):
    """Float64 four-endpoint enumeration (independent reference, CPU/numpy)."""
    import numpy as np

    edge_index = np.asarray(edge_index, dtype=np.int64).reshape(-1, 2)
    a = np.zeros((n, n), dtype=np.float64)
    if len(edge_index):
        a[edge_index[:, 0], edge_index[:, 1]] = 1.0
        a[edge_index[:, 1], edge_index[:, 0]] = 1.0
    a = (a > 0).astype(np.float64)
    deg = a.sum(axis=1)
    p = np.zeros_like(a)
    nz = deg > 0
    p[nz] = a[nz] / deg[nz, None]
    powers = []
    q = np.eye(n, dtype=np.float64)
    for _ in range(ksteps):
        powers.append(q)
        q = q @ p
    e = len(edge_index)
    out = np.zeros((e, e, ksteps), dtype=np.float64)
    u, v = edge_index[:, 0], edge_index[:, 1]
    for k in range(ksteps):
        qk = powers[k]
        out[:, :, k] = 0.5 * (qk[np.ix_(u, u)] + qk[np.ix_(u, v)]
                              + qk[np.ix_(v, u)] + qk[np.ix_(v, v)])
    return out


# ---------------------------------------------------------------------------
# norm / attention pieces
# ---------------------------------------------------------------------------
class AdaRMSNorm(nn.Module):
    """RMSNorm with DiT-style adaptive scale/shift.

    ``y = x / sqrt(mean(x^2) + eps) * (1 + s_l + P g) + b_l + Q g``

    ``g`` is a shared static graph-descriptor embedding (rank ``ADA_RANK``).
    ``P``/``Q`` are zero-initialised, so at initialisation the module is exactly
    an RMSNorm with identity gain.  No BatchNorm anywhere.
    """

    def __init__(self, d: int, rank: int = ADA_RANK, eps: float = RMS_EPS):
        super().__init__()
        self.eps = eps
        self.scale = nn.Parameter(torch.zeros(d))
        self.shift = nn.Parameter(torch.zeros(d))
        self.proj_s = nn.Linear(rank, d, bias=False)
        self.proj_b = nn.Linear(rank, d, bias=False)
        nn.init.zeros_(self.proj_s.weight)
        nn.init.zeros_(self.proj_b.weight)

    def forward(self, x: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        rms = F.rms_norm(x, (x.shape[-1],), None, self.eps)
        # g: [B, rank] -> per-graph modulation broadcast over the token axis
        if g.dim() == 2:
            g = g.unsqueeze(1)
        return rms * (1.0 + self.scale + self.proj_s(g)) + self.shift + self.proj_b(g)


class PEStem(nn.Module):
    """Shared static stem: RRWP values -> sinusoidal features -> per-head bias."""

    def __init__(self, ksteps: int = KSTEPS, hidden: int = PE_HIDDEN, heads: int = N_HEADS):
        super().__init__()
        d_in = ksteps * (1 + 2 * len(PE_FREQS))
        self.mlp = nn.Sequential(nn.Linear(d_in, hidden), nn.GELU(), nn.Linear(hidden, heads))
        nn.init.zeros_(self.mlp[-1].weight)
        nn.init.zeros_(self.mlp[-1].bias)

    def forward(self, r: torch.Tensor) -> torch.Tensor:
        feats = [r]
        for f in PE_FREQS:
            feats.append(torch.sin(math.pi * f * r))
            feats.append(torch.cos(math.pi * f * r))
        z = torch.cat(feats, dim=-1)              # [B,E,E,K*(1+2F)]
        return self.mlp(z)                        # [B,E,E,H]


def sl2_context(q, k, v, bias, mask):
    """Simplified-L2 attention context for one block.

    ``score_ij = (q_i . k_j - 0.5 ||k_j||^2) / sqrt(d_head) + bias_ij``
    ``ctx_i   = sum_j softmax_j(score)_ij v_j``

    q/k/v: [B, H, E, d_head]; bias: [B, H, E, E]; mask: [B, E] real tokens.
    Returns ``(context, probabilities)``; the probability tensor is returned so
    the caller can apply dropout before the value contraction.
    """
    scale = 1.0 / math.sqrt(q.shape[-1])
    logits = torch.matmul(q, k.transpose(-1, -2)) * scale
    logits = logits - 0.5 * k.pow(2).sum(-1).unsqueeze(-2) * scale
    logits = logits + bias
    logits = logits.masked_fill(~mask[:, None, None, :], -1e9)
    attn = torch.softmax(logits, dim=-1)
    return torch.matmul(attn, v), attn


class Block(nn.Module):
    """Pre-norm residual block with sL2 attention and graph-level drop path."""

    def __init__(self, d: int = D_MODEL, heads: int = N_HEADS, ffn_mult: int = FFN_MULT,
                 attn_dropout: float = ATTN_DROPOUT):
        super().__init__()
        if d % heads:
            raise ValueError("d_model must be divisible by heads")
        self.heads = heads
        self.d_head = d // heads
        self.scale = 1.0 / math.sqrt(self.d_head)
        self.norm1 = AdaRMSNorm(d)
        self.norm2 = AdaRMSNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ffn = nn.Sequential(
            nn.Linear(d, ffn_mult * d), nn.GELU(), nn.Linear(ffn_mult * d, d)
        )
        self.attn_drop = nn.Dropout(attn_dropout)

    def forward(self, h, bias, mask, g, keep, return_attn: bool = False):
        B, E, _ = h.shape
        H, dh = self.heads, self.d_head
        x = self.norm1(h, g)
        qkv = self.qkv(x).view(B, E, 3, H, dh).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]                      # [B,H,E,dh]
        _, attn = sl2_context(q, k, v, bias, mask)
        attn = self.attn_drop(attn)
        ctx = torch.matmul(attn, v)                            # [B,H,E,dh]
        ctx = ctx.permute(0, 2, 1, 3).reshape(B, E, H * dh)
        kp = keep[:, None, None]
        h = h + kp * self.out(ctx)
        h = h + kp * self.ffn(self.norm2(h, g))
        h = h * mask[..., None]
        if return_attn:
            return h, attn
        return h


# ---------------------------------------------------------------------------
# local (one-shot) structure encoder
# ---------------------------------------------------------------------------
class LocalEncoder(nn.Module):
    """Shared one-shot encoder of the typed induced object around a root bond."""

    def __init__(self, n_atom: int = N_ATOM, n_role: int = N_ROLE, d_z: int = D_Z,
                 d_phi: int = D_PHI, d_out: int = D_MODEL, hidden: int = AGGR_HIDDEN):
        super().__init__()
        self.atom = nn.Embedding(n_atom, d_z)
        self.role = nn.Embedding(n_role, d_z)
        self.phi_n = nn.Sequential(nn.Linear(d_z, d_phi), nn.SiLU(), nn.Linear(d_phi, d_phi))
        self.phi_p = nn.Sequential(
            nn.Linear(3 * d_z + N_EDGE, d_phi), nn.SiLU(), nn.Linear(d_phi, d_phi)
        )
        self.agg = nn.Sequential(
            nn.Linear(4 * d_phi + 2, hidden), nn.SiLU(), nn.Linear(hidden, d_out)
        )
        self.ln = nn.LayerNorm(d_out)

    def _aggregate(self, z, loc_et, nm, loc_nnodes, loc_nedges, arm: str):
        """One shared direction: node/pair deep-set + small MLP -> token width."""
        B, E, L = z.shape[0], z.shape[1], z.shape[2]
        zn = self.phi_n(z)
        znm = zn * nm[..., None]
        node_sum = znm.sum(dim=2)
        node_sq = znm.pow(2).sum(dim=2)

        zi = z.unsqueeze(3)
        zj = z.unsqueeze(2)
        feat = torch.cat([zi + zj, (zi - zj).abs(), zi * zj], dim=-1)

        tri = torch.ones(L, L, dtype=torch.bool, device=z.device).triu(1)
        pmask = nm.unsqueeze(3) & nm.unsqueeze(2) & tri    # [B,E,L,L]
        onehot = F.one_hot(loc_et, num_classes=N_EDGE).to(feat.dtype)   # [B,E,L,L,4]
        if arm == "real":
            edge = onehot
        elif arm == "null":
            cnt = pmask.sum(dim=(2, 3)).clamp(min=1).to(feat.dtype)     # [B,E]
            freq = (onehot * pmask[..., None].to(feat.dtype)).sum(dim=(2, 3)) / cnt[..., None]
            edge = freq[:, :, None, None, :].expand(B, E, L, L, N_EDGE)
        else:
            raise ValueError(f"unknown arm {arm!r}")

        pf = torch.cat([feat, edge], dim=-1)                 # [B,E,L,L,3d_z+4]
        zp = self.phi_p(pf)
        pz = zp * pmask[..., None]
        pair_sum = pz.sum(dim=(2, 3))
        pair_sq = pz.pow(2).sum(dim=(2, 3))

        agg = torch.cat([
            node_sum, node_sq, pair_sum, pair_sq,
            torch.log1p(loc_nnodes).unsqueeze(-1),
            torch.log1p(loc_nedges).unsqueeze(-1),
        ], dim=-1)
        return self.agg(agg)

    def forward(self, loc_type, loc_role, loc_et, loc_mask, loc_nnodes, loc_nedges,
                bond_mask, arm: str):
        B, E, L = loc_type.shape
        nm = loc_mask
        # padded local slots carry -1; they are masked, the clamp only keeps the
        # embedding lookup in range.
        t_idx = loc_type.clamp(min=0)
        r_idx = loc_role.clamp(min=0)
        # role id = dist(i,u) * (cap+1) + dist(i,v); the reversed root direction
        # swaps the two distance digits (padded slots stay -1).
        cap1 = ROLE_BASE
        r_sw = torch.where(loc_role >= 0,
                           (r_idx % cap1) * cap1 + (r_idx // cap1),
                           r_idx)
        z_fwd = (self.atom(t_idx) + self.role(r_idx)) * nm[..., None]
        z_rev = (self.atom(t_idx) + self.role(r_sw)) * nm[..., None]
        # h_b = 0.5 * (F(P_b, u->v) + F(P_b, v->u)): exact root-direction invariance
        agg = 0.5 * (self._aggregate(z_fwd, loc_et, nm, loc_nnodes, loc_nedges, arm)
                     + self._aggregate(z_rev, loc_et, nm, loc_nnodes, loc_nedges, arm))
        h = self.ln(agg)
        return h * bond_mask[..., None]


# ---------------------------------------------------------------------------
# full model
# ---------------------------------------------------------------------------
class ESOC(nn.Module):
    def __init__(self, arm: str = "real", ref_bonds: float = 24.9279,
                 ref_atoms: float = 23.1664, count_mean=None, count_scale=None,
                 n_layers: int = N_LAYERS, d_model: int = D_MODEL,
                 heads: int = N_HEADS, attn_dropout: float = ATTN_DROPOUT,
                 drop_path: float = DROP_PATH, max_atoms: int = 36):
        super().__init__()
        if arm not in ("real", "null"):
            raise ValueError(arm)
        self.arm = arm
        self.drop_path = float(drop_path)
        self.local = LocalEncoder(d_out=d_model)
        self.pe = PEStem(heads=heads)
        self.blocks = nn.ModuleList([Block(d_model, heads, FFN_MULT, attn_dropout)
                                     for _ in range(n_layers)])
        self.gdesc = nn.Sequential(nn.Linear(4, 16), nn.Tanh(), nn.Linear(16, ADA_RANK))
        d_read = 2 * d_model + D_Z + 4 + 33
        self.readout = nn.Sequential(
            nn.Linear(d_read, READOUT_HIDDEN), nn.SiLU(), nn.Linear(READOUT_HIDDEN, 1)
        )
        self.register_buffer("max_atoms", torch.tensor(int(max_atoms), dtype=torch.long))
        self.register_buffer("ref_bonds", torch.tensor(float(ref_bonds)))
        self.register_buffer("ref_atoms", torch.tensor(float(ref_atoms)))
        self.register_buffer("count_mean", torch.tensor(count_mean, dtype=torch.float32))
        self.register_buffer("count_scale", torch.tensor(count_scale, dtype=torch.float32))

    # -- static per-batch descriptors -------------------------------------
    def _counts(self, batch):
        n_atoms = batch["n_atoms"].to(torch.float32)
        n_bonds = batch["bond_mask"].sum(-1).to(torch.float32)
        nb_tok = batch["bond_mask"].to(torch.float32)
        nnodes = batch["loc_nnodes"]
        mean_local = (nnodes * nb_tok).sum(-1) / nb_tok.sum(-1).clamp(min=1.0)
        return torch.stack([
            torch.log1p(n_atoms),
            torch.log1p(n_bonds),
            n_bonds / n_atoms.clamp(min=1.0),
            mean_local,
        ], dim=-1)

    def forward(self, batch, return_info: bool = False):
        bond_mask = batch["bond_mask"]
        h = self.local(batch["loc_type"], batch["loc_role"], batch["loc_et"],
                       batch["loc_mask"], batch["loc_nnodes"], batch["loc_nedges"],
                       bond_mask, self.arm)
        r = rrwp(batch["bonds"], bond_mask, batch["n_atoms"], n_max=int(self.max_atoms))
        bias = self.pe(r).permute(0, 3, 1, 2)               # [B,H,E,E]
        counts = self._counts(batch)
        g = self.gdesc((counts - self.count_mean) / self.count_scale)
        B = h.shape[0]
        if self.training and self.drop_path > 0:
            keep = (torch.rand(B, device=h.device) >= self.drop_path).to(h.dtype)
            keep = keep / (1.0 - self.drop_path)
        else:
            keep = torch.ones(B, device=h.device, dtype=h.dtype)
        for blk in self.blocks:
            h = blk(h, bias, bond_mask, g, keep)

        w = bond_mask[..., None].to(h.dtype)
        s1 = (h * w).sum(1) / self.ref_bonds
        s2 = (h * w).pow(2).sum(1) / self.ref_bonds
        s3 = (batch["atom_count"] @ self.local.atom.weight) / self.ref_atoms
        z = torch.cat([s1, s2, s3, counts, batch["global33"]], dim=-1)
        pred = self.readout(z).squeeze(-1)
        if return_info:
            return pred, {"h": h, "s1": s1, "s2": s2, "s3": s3,
                          "counts": counts, "bias": bias, "keep": keep}
        return pred


def param_audit(model: nn.Module) -> dict:
    per_module = {name: int(sum(p.numel() for p in mod.parameters()))
                  for name, mod in model.named_children()}
    blocks = [int(sum(p.numel() for p in b.parameters())) for b in model.blocks]
    total = int(sum(p.numel() for p in model.parameters()))
    trainable = int(sum(p.numel() for p in model.parameters() if p.requires_grad))
    return {
        "param_total": total,
        "param_trainable": trainable,
        "under_500k": bool(total <= 500_000),
        "per_module": per_module,
        "per_block": blocks[0] if blocks else 0,
        "n_blocks": len(blocks),
        "d_model": D_MODEL,
        "n_heads": N_HEADS,
        "ffn_width": FFN_MULT * D_MODEL,
    }


def build_model(seed: int, arm: str, scales: dict, n_layers: int = N_LAYERS) -> ESOC:
    """Deterministic construction: the arm never touches the RNG or the init."""
    torch.manual_seed(int(seed))
    model = ESOC(
        arm=arm,
        ref_bonds=scales["ref_bonds"],
        ref_atoms=scales["ref_atoms"],
        count_mean=scales["count_mean"],
        count_scale=scales["count_scale"],
        n_layers=n_layers,
        max_atoms=int(scales.get("max_atoms", 36)),
    )
    return model

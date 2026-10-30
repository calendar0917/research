"""Lightweight binding / marginal / simple log-linear heads.

Binding head: one shared per-path scorer on (middle code, terminal type, root and
middle context), pooled by a permutation-invariant mean.

Marginal head: exactly the same parameters and per-path scorer, but the terminal
embedding of every path is replaced by the root's terminal-type mean.  The mean
is computed from integer terminal-type counts (identical for a real root and its
marginal-preserving permutation), so the marginal head sees bit-identical inputs
for the two members of a pair and must score 0.5 up to ties.

Simple baseline: a linear model on the root's summed (middle, terminal) one-hot
counts plus a few simple compatibility/interaction counts.  It has no learned
embeddings and no hidden layer; it is the "simple structural statistics" control.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import sbf_common as C

D_MID = 32
D_END = 32
D_ROOT_TYPE = 8
D_DEG = 4
D_RING = 2
D_MRING = 6
D_CLOSURE = 2
CTX_DIM = D_ROOT_TYPE + D_DEG + D_RING + D_MRING + D_DEG + D_RING + D_CLOSURE  # 28
PAIR_DIM = D_MID + D_END + D_MID + CTX_DIM  # 124
HIDDEN = 64


class PairHead(nn.Module):
    def __init__(self, *, marginal: bool = False, use_compat: bool = False):
        super().__init__()
        self.marginal = bool(marginal)
        self.use_compat = bool(use_compat)
        self.emid = nn.Embedding(C.N_MID_CODES, D_MID)
        self.eend = nn.Embedding(C.N_END_TYPES, D_END)
        self.ert = nn.Embedding(C.N_ATOM_TYPES, D_ROOT_TYPE)
        self.edeg = nn.Embedding(5, D_DEG)
        self.ering = nn.Embedding(2, D_RING)
        self.emr = nn.Embedding(9, D_MRING)
        self.ecl = nn.Embedding(2, D_CLOSURE)
        in_dim = PAIR_DIM + (1 if self.use_compat else 0)
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, HIDDEN), nn.SiLU(),
            nn.Linear(HIDDEN, 1),
        )

    def forward(self, b: Mapping[str, torch.Tensor]) -> torch.Tensor:
        mask = b["mask"]
        mid = b["mid"]
        end = b["end"]
        em = self.emid(mid)
        if self.marginal:
            # integer counts -> identical for real and marginal-preserving fake
            counts = torch.zeros((mid.shape[0], C.N_END_TYPES), dtype=torch.float32,
                                 device=mid.device)
            counts.scatter_add_(1, end, mask)
            n = counts.sum(1, keepdim=True).clamp(min=1.0)
            mean_end = (counts @ self.eend.weight) / n
            ee = mean_end.unsqueeze(1).expand(-1, mid.shape[1], -1)
        else:
            ee = self.eend(end)
        root_ctx = torch.cat([
            self.ert(b["root_type"]), self.edeg(b["root_deg"]),
            self.ering(b["root_ring"]), self.emr(b["root_mring"]),
        ], dim=1)
        ctx = torch.cat([
            root_ctx.unsqueeze(1).expand(-1, mid.shape[1], -1),
            self.edeg(b["mid_deg"]), self.ering(b["mid_ring"]), self.ecl(b["closure"]),
        ], dim=2)
        parts = [em, ee, em * ee, ctx]
        if self.use_compat:
            parts.append(b["compat"].unsqueeze(-1))
        x = torch.cat(parts, dim=2)
        s = self.mlp(x).squeeze(-1)
        # permutation-invariant pooling: sorting removes any dependence on the
        # order in which the path records happen to be stored.
        s = s.masked_fill(mask <= 0, -1e9)
        s, _ = torch.sort(s, dim=1, descending=True)
        return (s * mask).sum(1) / mask.sum(1).clamp(min=1.0)


class LinearPair(nn.Module):
    """Simple log-linear structural baseline (no embeddings, no hidden layer)."""

    def __init__(self, *, use_compat: bool = True):
        super().__init__()
        self.use_compat = bool(use_compat)
        n_feat = C.N_MID_CODES * C.N_END_TYPES + (5 if self.use_compat else 0)
        self.lin = nn.Linear(n_feat, 1)

    def forward(self, b: Mapping[str, torch.Tensor]) -> torch.Tensor:
        mask = b["mask"]
        n = mask.shape[0]
        feat = torch.zeros((n, C.N_MID_CODES * C.N_END_TYPES), device=mask.device)
        for j in range(mask.shape[1]):
            idx = (b["mid"][:, j] * C.N_END_TYPES + b["end"][:, j]).unsqueeze(1)
            feat.scatter_add_(1, idx, mask[:, j].unsqueeze(1))
        if self.use_compat:
            compat = b["compat"] * mask
            extra = torch.stack([
                compat.sum(1),
                ((b["end"] == b["root_type"].unsqueeze(1)).float() * mask).sum(1),
                ((b["end"] == b["mid_type"]).float() * mask).sum(1),
                ((b["end"] == b["root_type"].unsqueeze(1)).float() * b["closure"].float() * mask).sum(1),
                ((b["end"] == b["mid_type"]).float() * b["closure"].float() * mask).sum(1),
            ], dim=1)
            feat = torch.cat([feat, extra], dim=1)
        return self.lin(feat).squeeze(-1)


def oracle_compat_score(b: Mapping[str, torch.Tensor]) -> torch.Tensor:
    """Mean per-path local type compatibility; no training."""
    return (b["compat"] * b["mask"]).sum(1) / b["mask"].sum(1).clamp(min=1.0)


def n_params(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))


__all__ = ["PairHead", "LinearPair", "oracle_compat_score", "n_params", "CTX_DIM", "PAIR_DIM"]

"""Candidate models for zinc-arch-discovery-v1.

Six deliberately different computation graphs:

R0  rooted multi-scale statistics + count-aware static readout
R1  R0 atom tokens + two pre-LN global attention layers with distance and
    two-hop (middle atom, two bond types) relation bias
T   P3 atom/bond/junction encoders (train-only masked-SSL initialisation) +
    low-rank atom-bond / bond-junction association tensors; no Null grouping
P0  sampled walk patterns (lengths 2/4/6/8, anonymous revisit pattern) + GRU
    + length-grouped count-aware readout
P1  P0 + one global pattern-attention layer
B   biconnected blocks / bridges / isolated atoms + block-cut-tree relation
    bias + one global block-attention layer

No local message passing is used.  Attention is global and one-shot; no hidden
state is refreshed from neighbours.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ad_common as C
import ad_data as D

N_ATOM_TYPES = D.N_ATOM_TYPES
N_BOND_TYPES = D.N_BOND_TYPES
JOINT_CATEGORIES = D.JOINT_CATEGORIES
TWO_HOP_CATEGORIES = D.TWO_HOP_CATEGORIES


class MLP(nn.Module):
    """Plain MLP with the same ``lins`` naming as the frozen P3 encoders."""

    def __init__(self, in_dim: int, hidden: int, out_dim: int, n_hidden: int = 1, dropout: float = 0.0):
        super().__init__()
        dims = [in_dim] + [hidden] * n_hidden + [out_dim]
        self.lins = nn.ModuleList([nn.Linear(dims[i], dims[i + 1]) for i in range(len(dims) - 1)])
        self.dropout = float(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        for i, lin in enumerate(self.lins):
            x = lin(x)
            if i < len(self.lins) - 1:
                x = F.silu(x)
                if self.dropout > 0.0:
                    x = F.dropout(x, p=self.dropout, training=self.training)
        return x


class GlobalAttention(nn.Module):
    """Pre-LN block-diagonal global attention with optional static relation bias."""

    def __init__(
        self,
        d: int,
        heads: int = 4,
        ff_mult: int = 2,
        dropout: float = 0.0,
        n_dist: int | None = None,
        n_rel: int | None = None,
        n_overlap: int | None = None,
    ):
        super().__init__()
        if d % heads:
            raise ValueError("d must be divisible by heads")
        self.d = int(d)
        self.heads = int(heads)
        self.hd = self.d // self.heads
        self.ln1 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        self.ln2 = nn.LayerNorm(d)
        self.ff = nn.Sequential(
            nn.Linear(d, ff_mult * d),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d, d),
            nn.Dropout(dropout),
        )
        self.dropout = float(dropout)
        self.dist_emb = nn.Parameter(torch.zeros(int(n_dist), self.heads)) if n_dist else None
        self.rel_emb = nn.Parameter(torch.zeros(int(n_rel), self.heads)) if n_rel else None
        self.overlap_emb = nn.Parameter(torch.zeros(int(n_overlap), self.heads)) if n_overlap else None

    def forward(
        self,
        x: torch.Tensor,
        mask: torch.Tensor,
        *,
        dist: torch.Tensor | None = None,
        rel: tuple[torch.Tensor, ...] | None = None,
        overlap: torch.Tensor | None = None,
    ) -> torch.Tensor:
        B, N, _ = x.shape
        h = self.ln1(x)
        qkv = self.qkv(h).view(B, N, 3, self.heads, self.hd).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]  # [B,H,N,hd]
        scores = torch.einsum("bhid,bhjd->bhij", q, k) / math.sqrt(self.hd)

        bias = None
        if self.dist_emb is not None and dist is not None:
            bias = self.dist_emb[dist.long()]  # [B,N,N,H]
        if self.overlap_emb is not None and overlap is not None:
            ov = self.overlap_emb[overlap.long()]
            bias = ov if bias is None else bias + ov
        if self.rel_emb is not None and rel is not None:
            b, u, vv, cat, cnt = rel
            if b.numel():
                add = self.rel_emb[cat.long()] * cnt.float().unsqueeze(1)
                if bias is None:
                    bias = x.new_zeros((B, N, N, self.heads))
                bias = bias.index_put((b.long(), u.long(), vv.long()), add, accumulate=True)
        if bias is not None:
            scores = scores + bias.permute(0, 3, 1, 2)

        scores = scores.masked_fill(~mask[:, None, None, :], torch.finfo(scores.dtype).min)
        attn = torch.softmax(scores, dim=-1)
        attn = F.dropout(attn, p=self.dropout, training=self.training)
        ctx = torch.einsum("bhij,bhjd->bhid", attn, v).permute(0, 2, 1, 3).reshape(B, N, self.d)
        x = x + F.dropout(self.out(ctx), p=self.dropout, training=self.training)
        x = x + self.ff(self.ln2(x))
        return x * mask.unsqueeze(-1).to(x.dtype)


def _global_std(g: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (g - mean) / scale


class RootedModel(nn.Module):
    """R0 / R1: per-atom rooted multi-scale statistics, optional global attention."""

    def __init__(self, norm: Mapping[str, Any], *, width: int = 96, dropout: float = 0.0,
                 use_attention: bool = False, attn_layers: int = 2, heads: int = 4):
        super().__init__()
        self.use_attention = bool(use_attention)
        self.width = int(width)
        in_dim = D.ATOM_FEAT_DIM + 3 * N_ATOM_TYPES + JOINT_CATEGORIES + D.ROOT_SCALARS
        self.root_mlp = MLP(in_dim, 128, width, n_hidden=1, dropout=dropout)
        self.scale_emb = nn.Parameter(torch.zeros(3, width))
        self.token_norm = nn.LayerNorm(width)
        self.fuse = nn.Linear(3 * width, width)
        self.fuse_norm = nn.LayerNorm(width)
        self.attn = nn.ModuleList(
            [
                GlobalAttention(width, heads=heads, ff_mult=2, dropout=dropout,
                                n_dist=9, n_rel=TWO_HOP_CATEGORIES)
                for _ in range(attn_layers if use_attention else 0)
            ]
        )
        self.reader = MLP(3 * width + 1 + N_ATOM_TYPES + N_BOND_TYPES + D.GLOBAL_DIM,
                          256, 1, n_hidden=2, dropout=dropout)
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        self.register_buffer("root_mean", torch.as_tensor(norm["root_scalar_mean"], dtype=torch.float32))
        self.register_buffer("root_scale", torch.as_tensor(norm["root_scalar_scale"], dtype=torch.float32))
        mask = torch.zeros(3, 3)
        for r in range(3):
            mask[r, : r + 1] = 1.0
        self.register_buffer("shell_radius_mask", mask)

    def encode_atoms(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        A = batch["atom_feat"].shape[0]
        hs = []
        for r in range(3):
            shell = batch["shell"].float() * self.shell_radius_mask[r].view(1, 3, 1)
            sc = (batch["root_scalar"][:, r, :] - self.root_mean[r]) / self.root_scale[r]
            inp = torch.cat(
                [
                    batch["atom_feat"],
                    shell.reshape(A, -1),
                    batch["joint"][:, r, :].float(),
                    sc,
                ],
                dim=1,
            )
            hs.append(self.token_norm(self.root_mlp(inp) + self.scale_emb[r]))
        cat = torch.cat(hs, dim=1)
        return self.fuse_norm(hs[2] + self.fuse(cat))

    def readout(
        self,
        h: torch.Tensor,
        atom_batch: torch.Tensor,
        n_graphs: int,
        atom_type: torch.Tensor,
        bond_type: torch.Tensor,
        bond_batch: torch.Tensor,
        global_feat: torch.Tensor,
    ) -> torch.Tensor:
        d = h.shape[1]
        count = torch.bincount(atom_batch, minlength=n_graphs).clamp(min=1).to(h.dtype)
        s = h.new_zeros((n_graphs, d)).index_add_(0, atom_batch, h)
        sq = h.new_zeros((n_graphs, d)).index_add_(0, atom_batch, h * h)
        mean = s / count.unsqueeze(1)
        sq = sq / count.unsqueeze(1)
        atom_count = h.new_zeros((n_graphs, N_ATOM_TYPES)).index_add_(
            0, atom_batch, F.one_hot(atom_type.long(), N_ATOM_TYPES).to(h.dtype)
        )
        bond_count = h.new_zeros((n_graphs, N_BOND_TYPES)).index_add_(
            0, bond_batch, F.one_hot(bond_type.long() - 1, N_BOND_TYPES).to(h.dtype)
        )
        g = _global_std(global_feat, self.global_mean, self.global_scale)
        reader_in = torch.cat(
            [s, mean, sq, torch.log1p(count).unsqueeze(1), atom_count, bond_count, g], dim=1
        )
        return self.reader(reader_in).view(-1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        h = self.encode_atoms(batch)
        if self.use_attention:
            B = int(batch["n_graphs"])
            N = int(batch["Nmax"])
            pad = h.new_zeros((B, N, self.width))
            pad[batch["atom_batch"], batch["atom_local"]] = h
            mask = batch["mask"]
            for layer in self.attn:
                pad = layer(pad, mask, dist=batch["dist_pad"], rel=batch.get("twohop"))
            h = pad[mask]
        return self.readout(
            h,
            batch["atom_batch"],
            int(batch["n_graphs"]),
            batch["atom_type"],
            batch["bond_type"],
            batch["bond_batch"],
            batch["global"],
        )


class P3Encoders(nn.Module):
    """The frozen P3 atom/bond/junction encoders, without grouping/composer/reader."""

    def __init__(self, *, atom_hidden=128, atom_out=96, bond_hidden=160, bond_out=96,
                 junc_hidden=160, junc_out=96, dropout=0.0):
        super().__init__()
        self.atom_out = int(atom_out)
        self.bond_out = int(bond_out)
        self.junc_out = int(junc_out)
        self.atom_encoder = MLP(D.ATOM_FEAT_DIM, atom_hidden, atom_out, 1, dropout)
        self.bond_encoder = MLP(3 * atom_out + D.BOND_FEAT_DIM, bond_hidden, bond_out, 1, dropout)
        self.junction_encoder = MLP(
            4 * atom_out + 2 * bond_out + D.JUNC_FEAT_DIM, junc_hidden, junc_out, 1, dropout
        )

    def forward(self, batch: Mapping[str, torch.Tensor]):
        a = self.atom_encoder(batch["atom_feat"])
        idx = batch["bond_index"]
        au, av = a[idx[:, 0]], a[idx[:, 1]]
        bond_feat = torch.cat([au + av, (au - av).abs(), au * av, batch["bond_feat"]], dim=1)
        b = self.bond_encoder(bond_feat)
        ji = batch["junc_index"]
        jb = batch["junc_bond"]
        ai, aj, ak = a[ji[:, 0]], a[ji[:, 1]], a[ji[:, 2]]
        b1, b2 = b[jb[:, 0]], b[jb[:, 1]]
        junc_feat = torch.cat(
            [ai + ak, (ai - ak).abs(), ai * ak, aj, b1 + b2, (b1 - b2).abs(), batch["junc_feat"]],
            dim=1,
        )
        w = self.junction_encoder(junc_feat)
        return a, b, w


class TensorAssociationModel(nn.Module):
    """T: low-rank atom-bond and bond-junction association tensors."""

    def __init__(self, norm: Mapping[str, Any], *, rank: int = 16, dropout: float = 0.0):
        super().__init__()
        self.rank = int(rank)
        self.encoders = P3Encoders(dropout=dropout)
        self.ab_atom = nn.Linear(96, rank)
        self.ab_bond = nn.Linear(96, rank)
        self.bj_bond = nn.Linear(96, rank)
        self.bj_junc = nn.Linear(96, rank)
        in_dim = 2 * rank * rank + 6 * 96 + 6 + D.GLOBAL_DIM
        self.reader = MLP(in_dim, 192, 1, n_hidden=2, dropout=dropout)
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))

    def load_ssl(self, path: Path, device) -> dict:
        state = torch.load(path, map_location="cpu", weights_only=False)
        self.encoders.atom_encoder.load_state_dict({k: v.to(device) for k, v in state["atom_encoder"].items()})
        self.encoders.bond_encoder.load_state_dict({k: v.to(device) for k, v in state["bond_encoder"].items()})
        self.encoders.junction_encoder.load_state_dict({k: v.to(device) for k, v in state["junction_encoder"].items()})
        return {
            "source": str(path),
            "sha256": C.sha256_file(path),
            "keys": sorted(state.keys()),
            "encoder_layers_loaded": True,
        }

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        B = int(batch["n_graphs"])
        a, b, w = self.encoders(batch)
        E = b.shape[0]
        J = w.shape[0]
        ab_atom = torch.cat([batch["bond_index"][:, 0], batch["bond_index"][:, 1]], dim=0)
        ab_bond = torch.arange(E, device=b.device).repeat(2)
        ab_graph = torch.cat([batch["bond_batch"], batch["bond_batch"]], dim=0)
        u = self.ab_atom(a[ab_atom])
        v = self.ab_bond(b[ab_bond])
        C_ab = a.new_zeros((B, self.rank, self.rank)).index_add_(
            0, ab_graph, u.unsqueeze(2) * v.unsqueeze(1)
        )
        n_ab = torch.bincount(ab_graph, minlength=B).clamp(min=1).to(a.dtype)
        C_ab = C_ab / n_ab.view(B, 1, 1)

        bj_bond = batch["junc_bond"].reshape(-1)
        bj_junc = torch.arange(J, device=w.device).repeat_interleave(2)
        bj_graph = batch["junc_batch"].repeat_interleave(2)
        p = self.bj_bond(b[bj_bond])
        q = self.bj_junc(w[bj_junc])
        C_bj = a.new_zeros((B, self.rank, self.rank)).index_add_(
            0, bj_graph, p.unsqueeze(2) * q.unsqueeze(1)
        )
        n_bj = torch.bincount(bj_graph, minlength=B).clamp(min=1).to(a.dtype)
        C_bj = C_bj / n_bj.view(B, 1, 1)

        n_atoms = torch.bincount(batch["atom_batch"], minlength=B).clamp(min=1).to(a.dtype)
        n_bonds = torch.bincount(batch["bond_batch"], minlength=B).clamp(min=1).to(a.dtype)
        n_junc = torch.bincount(batch["junc_batch"], minlength=B).clamp(min=1).to(a.dtype)
        a_sum = a.new_zeros((B, 96)).index_add_(0, batch["atom_batch"], a)
        b_sum = a.new_zeros((B, 96)).index_add_(0, batch["bond_batch"], b)
        w_sum = a.new_zeros((B, 96)).index_add_(0, batch["junc_batch"], w)
        a_mean = a_sum / n_atoms.view(B, 1)
        b_mean = b_sum / n_bonds.view(B, 1)
        w_mean = w_sum / n_junc.view(B, 1)
        counts = torch.stack(
            [torch.log1p(n_atoms), torch.log1p(n_bonds), torch.log1p(n_junc),
             torch.log1p(n_ab), torch.log1p(n_bj),
             (n_ab / n_atoms).view(-1)], dim=1
        )
        g = _global_std(batch["global"], self.global_mean, self.global_scale)
        reader_in = torch.cat(
            [C_ab.reshape(B, -1), C_bj.reshape(B, -1), a_sum, b_sum, w_sum,
             a_mean, b_mean, w_mean, counts, g],
            dim=1,
        )
        return self.reader(reader_in).view(-1)


class PathModel(nn.Module):
    """P0 / P1: sampled walk patterns with GRU and optional pattern attention."""

    def __init__(self, norm: Mapping[str, Any], *, width: int = 96, dropout: float = 0.0,
                 use_attention: bool = False, heads: int = 4):
        super().__init__()
        self.width = int(width)
        self.use_attention = bool(use_attention)
        self.atom_emb = nn.Embedding(N_ATOM_TYPES, 24)
        self.bond_emb = nn.Embedding(N_BOND_TYPES, 8)
        self.anon_emb = nn.Embedding(D.MAX_WALK + 1, 16)
        token_dim = 24 + 8 + 16 + 6
        self.gru = nn.GRU(token_dim, width, batch_first=True)
        self.len_emb = nn.Embedding(len(D.WALK_LENGTHS), width)
        self.path_norm = nn.LayerNorm(width)
        self.attn = nn.ModuleList(
            [GlobalAttention(width, heads=heads, ff_mult=2, dropout=dropout)]
            if use_attention else []
        )
        # Count-invariant path readout: the training bank samples 32 walks and
        # the fixed valid bank uses 128, so raw sums and raw sampled counts are
        # split indicators, not molecular features.  Keep per-length means and
        # second moments (both divided by the actual sampled count), plus the
        # full-molecule composition/size and global33 channels.
        readout_in = 2 * len(D.WALK_LENGTHS) * width + N_ATOM_TYPES + N_BOND_TYPES + 2 + D.GLOBAL_DIM
        self.reader = MLP(readout_in, 224, 1, n_hidden=2, dropout=dropout)
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))

    def encode_paths(self, batch: Mapping[str, torch.Tensor]):
        wa = batch["walk_atom"].long()
        wb = batch["walk_bond"].long()
        wanon = batch["walk_anon"].long()
        L = batch["walk_len"].long()
        W, T = wa.shape
        pos = torch.arange(T, device=wa.device).view(1, T)
        valid = pos <= L.view(W, 1)
        is_new = (wanon == pos) & valid
        is_repeat = (wanon < pos) & valid
        closes = (wa == wa[:, :1]) & (pos > 0) & valid
        is_last = pos == L.view(W, 1)
        pos_norm = pos.float() / L.view(W, 1).clamp(min=1).float()
        len_norm = (L.view(W, 1).float() / float(D.MAX_WALK)).expand(-1, T)
        scalars = torch.stack(
            [pos_norm, is_new.float(), is_repeat.float(), closes.float(), is_last.float(), len_norm],
            dim=-1,
        )
        tok = torch.cat(
            [self.atom_emb(wa), self.bond_emb(wb), self.anon_emb(wanon), scalars], dim=-1
        )
        lengths = (L + 1).cpu()
        packed = nn.utils.rnn.pack_padded_sequence(tok, lengths, batch_first=True, enforce_sorted=False)
        out, _ = self.gru(packed)
        out, _ = nn.utils.rnn.pad_packed_sequence(out, batch_first=True, total_length=T)
        idx = torch.arange(W, device=out.device)
        h_last = out[idx, L]
        h_mean = (out * valid.unsqueeze(-1)).sum(1) / (L + 1).view(W, 1).float()
        h = self.path_norm(h_last + h_mean + self.len_emb(batch["walk_group"].long()))
        return h

    def readout_paths(self, h: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        B = int(batch["n_graphs"])
        graph = batch["walk_graph"].long()
        group = batch["walk_group"].long()
        parts = []
        for g in range(len(D.WALK_LENGTHS)):
            sel = group == g
            if sel.any():
                gg = graph[sel]
                hh = h[sel]
                cnt = torch.bincount(gg, minlength=B).clamp(min=1).to(h.dtype)
                s = h.new_zeros((B, self.width)).index_add_(0, gg, hh)
                sq = h.new_zeros((B, self.width)).index_add_(0, gg, hh * hh)
                parts.extend([s / cnt.view(B, 1), sq / cnt.view(B, 1)])
            else:
                z = h.new_zeros((B, self.width))
                parts.extend([z, z])
        g = _global_std(batch["global"], self.global_mean, self.global_scale)
        reader_in = torch.cat(
            parts + [batch["atom_count"], batch["bond_count"],
                     torch.log1p(batch["n_atoms"].float()).unsqueeze(1),
                     torch.log1p(batch["n_bonds"].float()).unsqueeze(1), g],
            dim=1,
        )
        return self.reader(reader_in).view(-1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        h = self.encode_paths(batch)
        if self.use_attention:
            B = int(batch["n_graphs"])
            S = int(batch["S"])
            pad = h.new_zeros((B, S, self.width))
            pad[batch["walk_graph"], batch["walk_local"]] = h
            mask = torch.ones((B, S), dtype=torch.bool, device=h.device)
            for layer in self.attn:
                pad = layer(pad, mask)
            h = pad[mask]
        return self.readout_paths(h, batch)


class BlockModel(nn.Module):
    """B: biconnected blocks + block-cut-tree relation bias."""

    def __init__(self, norm: Mapping[str, Any], *, width: int = 96, dropout: float = 0.0, heads: int = 4):
        super().__init__()
        self.width = int(width)
        in_dim = JOINT_CATEGORIES + N_ATOM_TYPES + N_BOND_TYPES + N_ATOM_TYPES + D.BLOCK_SCALARS
        self.n_scalar = D.BLOCK_SCALARS
        self.block_mlp = MLP(in_dim, 128, width, n_hidden=1, dropout=dropout)
        self.block_norm = nn.LayerNorm(width)
        self.attn = GlobalAttention(width, heads=heads, ff_mult=2, dropout=dropout,
                                    n_dist=9, n_overlap=4)
        self.reader = MLP(3 * width + 1 + N_ATOM_TYPES + N_BOND_TYPES + 2 + D.GLOBAL_DIM,
                          256, 1, n_hidden=2, dropout=dropout)
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        self.register_buffer("block_mean", torch.as_tensor(norm["block_scalar_mean"], dtype=torch.float32))
        self.register_buffer("block_scale", torch.as_tensor(norm["block_scalar_scale"], dtype=torch.float32))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        feat = batch["block_feat"]
        raw, sc = feat[..., : -self.n_scalar], feat[..., -self.n_scalar:]
        sc = (sc - self.block_mean) / self.block_scale
        feat = torch.cat([raw, sc], dim=-1)
        h = self.block_norm(self.block_mlp(feat))
        mask = batch["block_mask"]
        h = self.attn(h, mask, dist=batch["block_dist"], overlap=batch["block_overlap"])
        count = mask.sum(1).clamp(min=1).to(h.dtype)
        s = (h * mask.unsqueeze(-1)).sum(1)
        sq = (h * h * mask.unsqueeze(-1)).sum(1)
        mean = s / count.unsqueeze(1)
        sq = sq / count.unsqueeze(1)
        g = _global_std(batch["global"], self.global_mean, self.global_scale)
        reader_in = torch.cat(
            [s, mean, sq, torch.log1p(count).unsqueeze(1), batch["atom_count"],
             batch["bond_count"], torch.log1p(batch["n_atoms"].float()).unsqueeze(1),
             torch.log1p(batch["n_bonds"].float()).unsqueeze(1), g],
            dim=1,
        )
        return self.reader(reader_in).view(-1)


CANDIDATES = ("R0", "R1", "T", "P0", "P1", "B")


def build_model(candidate: str, norm: Mapping[str, Any], seed: int, device) -> tuple[nn.Module, dict]:
    torch.manual_seed(int(seed))
    np.random.seed(int(seed) % (2**32))
    info: dict[str, Any] = {"candidate": candidate, "ssl": None}
    if candidate == "R0":
        model = RootedModel(norm, use_attention=False)
    elif candidate == "R1":
        model = RootedModel(norm, use_attention=True, attn_layers=2)
    elif candidate == "T":
        model = TensorAssociationModel(norm, rank=16)
    elif candidate == "P0":
        model = PathModel(norm, use_attention=False)
    elif candidate == "P1":
        model = PathModel(norm, use_attention=True)
    elif candidate == "B":
        model = BlockModel(norm)
    else:
        raise ValueError(f"unknown candidate {candidate!r}")
    model = model.to(device)
    if candidate == "T":
        path = C.P3_SSL_S0 if int(seed) == 0 else C.P3_SSL_S1
        if path.exists():
            info["ssl"] = model.load_ssl(path, device)
        else:
            info["ssl"] = {"source": None, "note": "frozen SSL encoder unavailable; random init"}
    n_params = C.parameter_count(model)
    if n_params > 500_000:
        raise RuntimeError(f"{candidate}: {n_params} parameters exceeds the 500k cap")
    info["parameters"] = n_params
    info["requires_grad_true"] = sum(1 for p in model.parameters() if p.requires_grad)
    info["parameters_total"] = sum(1 for p in model.parameters())
    return model, info

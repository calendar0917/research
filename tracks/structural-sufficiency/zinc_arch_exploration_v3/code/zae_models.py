"""Two new ZINC trunks for zinc-arch-exploration-v3.

Arm **A** — ``PairComposition`` — one full-graph pair composition that keeps the
intermediate atom identity:

    g_i   = Phi_e([a_i, mean_{j in J(i)} w_j, mean_{j in C(i)} w_j, counts])   # static
    p_i   = W_p g_i
    X_ij  = LN(Phi_x([p_i+p_j, |p_i-p_j|, p_i*p_j, [b_ij] , adj_ij, R_ij]))    # every pair
    U_ij  = (1/n) sum_k phi(X_ik) * psi(X_kj) * chi(a_k)                       # one contraction
    h_ij  = Phi_h([X_ij, U_ij])                                                # per-pair nonlinear
    z     = [sum_{i!=j} h / sqrt(#pairs), mean_{i!=j} h, mean_{i!=j} h^2,
             mean_i h_ii, unary means, counts, global33]
    y     = Reader(z)

    ``R_ij`` is the static pair descriptor (distance / bond type / common
    neighbours / ring-bond / same-block embeddings).  Every atom pair is a row,
    including the diagonal.  The contraction is the only cross-pair operation:
    it is applied once, is not iterated, and does not propagate hidden states
    along the local adjacency.

Arm **B** — ``SpectralJoint`` — multi-point invariants of static whole-graph
structural operators:

    s_i   = Phi_e(...)                                  # same static endpoint state
    u,v,w = tanh(W_u s), tanh(W_v s), tanh(W_w s)        # R content channels
    K     = {exp(-t L_norm)} union {D^-1/2 A_b D^-1/2}   # static operator library
    T_(r,t,s,q) = tr(D_u_r K_t D_v_r K_s D_w_r K_q)      # genuine 3-point joint
    P_(r,t,s)   = tr(D_u_r K_t D_v_r K_s)                # 2-point counterpart
    F_(r)       = sum_i u_r(i) v_r(i) w_r(i)             # 1-point counterpart
    y     = Reader(LN([T, P, F, sums]) ++ counts ++ global33)

    Traces are permutation invariant by construction and do not select a
    particular eigenbasis (the heat kernels are full matrix functions of the
    Laplacian; typed operators are finite powers of the raw typed adjacency).
    No hidden state is ever re-injected into a neighbour: the operators are
    precomputed from the raw graph and used once.

Both arms start from the seed-0 train-only masked-SSL P3 encoders and fine-tune
them; both stay below 500k trainable parameters.
"""

from __future__ import annotations

import math
import sys
from pathlib import Path
from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_EXP = _HERE.parent
_V1 = _EXP.parent / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_models as M  # noqa: E402
import zae_data as ZD  # noqa: E402

WIDTH = 96
BOND_FEAT_DIM = ZD.BOND_FEAT_DIM
GLOBAL_DIM = ZD.GLOBAL_DIM
MLP = M.MLP
P3Encoders = M.P3Encoders

#: heat-kernel scales t in exp(-t L); declared before training
HEAT_SCALES = (0.5, 1.0, 2.0)
TYPED_OPS = (1, 2, 3)          # raw bond type ids stored in the typed adjacency
KERNEL_NAMES = tuple(f"heat{t:g}" for t in HEAT_SCALES) + tuple(f"type{b}" for b in TYPED_OPS)


# ---------------------------------------------------------------------------
# shared static endpoint state (atoms + their real junction objects)
# ---------------------------------------------------------------------------


class EndpointState(nn.Module):
    """One-shot per-atom state: atom content + the junctions that own it.

    ``j_edge`` counts a junction on both of its endpoint atoms (i and k), and
    ``j_centre`` only on the centre atom j.  Both are static sums over the raw
    object table, divided by their true incidence count; the log1p counts are
    carried explicitly so the reader can tell "no junction" from "mean 0".
    """

    def __init__(self, width: int = WIDTH, hidden: int = 96, dropout: float = 0.0):
        super().__init__()
        self.mlp = MLP(3 * WIDTH + 2, hidden, width, n_hidden=1, dropout=dropout)
        self.norm = nn.LayerNorm(width)

    def forward(self, a: torch.Tensor, w: torch.Tensor, batch: Mapping[str, torch.Tensor]):
        A = a.shape[0]
        ji = batch["junc_index"]
        if ji.numel():
            j0, j1, j2 = ji[:, 0], ji[:, 1], ji[:, 2]
            e_sum = a.new_zeros((A, WIDTH))
            e_sum.index_add_(0, j0, w)
            e_sum.index_add_(0, j2, w)
            c_sum = a.new_zeros((A, WIDTH))
            c_sum.index_add_(0, j1, w)
            e_cnt = a.new_zeros((A,))
            e_cnt.index_add_(0, j0, torch.ones_like(j0, dtype=a.dtype))
            e_cnt.index_add_(0, j2, torch.ones_like(j2, dtype=a.dtype))
            c_cnt = a.new_zeros((A,))
            c_cnt.index_add_(0, j1, torch.ones_like(j1, dtype=a.dtype))
        else:
            e_sum = a.new_zeros((A, WIDTH))
            c_sum = a.new_zeros((A, WIDTH))
            e_cnt = a.new_zeros((A,))
            c_cnt = a.new_zeros((A,))
        e_mean = e_sum / e_cnt.clamp(min=1).unsqueeze(1)
        c_mean = c_sum / c_cnt.clamp(min=1).unsqueeze(1)
        x = torch.cat([a, e_mean, c_mean, torch.log1p(e_cnt).unsqueeze(1),
                       torch.log1p(c_cnt).unsqueeze(1)], dim=1)
        return self.norm(self.mlp(x))


# ---------------------------------------------------------------------------
# arm A
# ---------------------------------------------------------------------------


class PairComposition(nn.Module):
    """Full-graph pair composition with one intermediate-atom contraction."""

    def __init__(self, norm: Mapping[str, Any], *, d_pair: int = 64, n_chan: int = 32,
                 pair_rank: int = 48, h_hidden: int = 96, reader_hidden: int = 192,
                 dropout: float = 0.0, pool: str = "mean"):
        super().__init__()
        self.d_pair = int(d_pair)
        self.n_chan = int(n_chan)
        self.pool = str(pool)
        if self.pool not in ("mean", "attn"):
            raise ValueError(f"unknown pool {pool!r}")
        self.encoders = P3Encoders(dropout=dropout)
        self.endpoint = EndpointState(dropout=dropout)
        self.pair_proj = nn.Linear(WIDTH, pair_rank)
        self.dist_emb = nn.Embedding(ZD.DIST_BUCKETS, 8)
        self.bond_emb = nn.Embedding(ZD.BOND_VOCAB, 8)   # 0 = no bond / mask
        self.common_emb = nn.Embedding(ZD.COMMON_CAP, 8)
        self.ring_emb = nn.Embedding(2, 4)
        self.block_emb = nn.Embedding(2, 4)
        pair_in = 3 * pair_rank + WIDTH + 1 + ZD.PAIR_EMB_DIM
        self.x_mlp = MLP(pair_in, 128, d_pair, n_hidden=1, dropout=dropout)
        self.x_norm = nn.LayerNorm(d_pair)
        self.phi = nn.Linear(d_pair, n_chan)
        self.psi = nn.Linear(d_pair, n_chan)
        self.chi = nn.Linear(WIDTH, n_chan)
        self.h_mlp = MLP(d_pair + n_chan, h_hidden, WIDTH, n_hidden=1, dropout=dropout)
        reader_in = 7 * WIDTH + 4 + GLOBAL_DIM
        # one per-sample LayerNorm per pooled block: the count-scaled sum, the
        # pair means/moments and the unary means have very different raw scales
        # (measured ~3 vs ~0.02), and without this the small blocks would only
        # see a tiny first-layer signal.  Counts and global33 stay raw.
        self.block_norms = nn.ModuleList([nn.LayerNorm(WIDTH) for _ in range(7)])
        # content-based pooling: three learned queries score the off-diagonal
        # pairs of a graph (learned sum, learned mean, learned second moment).
        self.attn_q = nn.Parameter(torch.zeros(3, WIDTH)) if self.pool == "attn" else None
        self.reader = MLP(reader_in, reader_hidden, 1, n_hidden=2, dropout=dropout)
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))

    # -- pair representation ---------------------------------------------------
    def pair_features(self, batch, a, b, w):
        g = self.endpoint(a, w, batch)
        p = self.pair_proj(g)
        pi = p[batch["pair_i"]]
        pj = p[batch["pair_j"]]
        adj = batch["pair_adj"].unsqueeze(1)
        bemb = b[batch["pair_bond"]] * adj
        remb = torch.cat(
            [
                self.dist_emb(batch["pair_dist"]),
                self.bond_emb(batch["bond_type"][batch["pair_bond"]] * batch["pair_adj"].long()),
                self.common_emb(batch["pair_common"]),
                self.ring_emb(batch["pair_ring"]),
                self.block_emb(batch["pair_block"]),
            ],
            dim=1,
        )
        X = self.x_mlp(torch.cat([pi + pj, (pi - pj).abs(), pi * pj, bemb, adj, remb], dim=1))
        return self.x_norm(X), g

    def compose(self, X: torch.Tensor, a: torch.Tensor, batch: Mapping[str, torch.Tensor]):
        """One contraction U_ij = (1/n) sum_k phi(X_ik) * psi(X_kj) * chi(a_k)."""
        B, D = int(batch["n_graphs"]), int(batch["Nmax"])
        phi = self.phi(X)                                        # [P, C]
        psi = self.psi(X)                                        # [P, C]
        chi = self.chi(a)                                        # [A, C]
        dense = B * D * D
        idx = batch["pair_dense"]
        phi_d = X.new_zeros((dense, self.n_chan)).index_copy(0, idx, phi).view(B, D, D, self.n_chan)
        psi_d = X.new_zeros((dense, self.n_chan)).index_copy(0, idx, psi)
        chi_d = X.new_zeros((B * D, self.n_chan)).index_copy(0, batch["atom_slot"], chi).view(B, D, self.n_chan)
        # chi multiplies the *intermediate* atom k = first index of psi(k, j)
        psi_d = (psi_d.view(B, D, D, self.n_chan) * chi_d.unsqueeze(2)).view(B, D, D, self.n_chan)
        left = phi_d.permute(0, 3, 1, 2).reshape(B * self.n_chan, D, D)
        right = psi_d.permute(0, 3, 1, 2).reshape(B * self.n_chan, D, D)
        prod = torch.bmm(left, right).view(B, self.n_chan, D, D)
        U = prod.permute(0, 2, 3, 1).reshape(dense, self.n_chan).index_select(0, idx)
        return U / batch["pair_nm"].unsqueeze(1).clamp(min=1.0)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        enc = self.encode_pairs(batch)
        return self.readout(enc, batch)

    # -- one-shot pair state + contraction --------------------------------------
    def encode_pairs(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        a, b, w = self.encoders(batch)
        X, g = self.pair_features(batch, a, b, w)
        U = self.compose(X, a, batch)
        h = self.h_mlp(torch.cat([X, U], dim=1))                  # [P, 96]
        return {"a": a, "b": b, "w": w, "g": g, "X": X, "U": U, "h": h}

    def readout(self, enc: Mapping[str, torch.Tensor], batch: Mapping[str, torch.Tensor]):
        a, b, w, h = enc["a"], enc["b"], enc["w"], enc["h"]
        B = int(batch["n_graphs"])
        graph = batch["pair_graph"]
        off = 1.0 - batch["pair_self"]
        diag = batch["pair_self"]
        cnt_off = batch["n_atoms"] * (batch["n_atoms"] - 1.0)
        h_off = h * off.unsqueeze(1)
        if self.pool == "attn":
            att = self._attn_pool(h_off, graph, off, B, cnt_off)
            sum_off, att_mean, sq_off = att
        else:
            sum_off = h.new_zeros((B, WIDTH)).index_add_(0, graph, h_off)
            sq_off = h.new_zeros((B, WIDTH)).index_add_(0, graph, h_off * h_off)
        diag_sum = h.new_zeros((B, WIDTH)).index_add_(0, graph, h * diag.unsqueeze(1))
        n_pairs_off = cnt_off.clamp(min=1.0)
        mean_off = att_mean if self.pool == "attn" else sum_off / n_pairs_off.unsqueeze(1)
        sq_mean = sq_off / n_pairs_off.unsqueeze(1)
        sum_scaled = sum_off / n_pairs_off.sqrt().unsqueeze(1)
        diag_mean = diag_sum / batch["n_atoms"].clamp(min=1.0).unsqueeze(1)
        mean_a = a.new_zeros((B, WIDTH)).index_add_(0, batch["atom_batch"], a) / batch["n_atoms"].clamp(min=1.0).unsqueeze(1)
        mean_b = b.new_zeros((B, WIDTH)).index_add_(0, batch["bond_batch"], b) / batch["n_bonds"].clamp(min=1.0).unsqueeze(1)
        mean_w = w.new_zeros((B, WIDTH)).index_add_(0, batch["junc_batch"], w) / batch["n_junc"].clamp(min=1.0).unsqueeze(1)
        counts = torch.stack(
            [
                torch.log1p(batch["n_atoms"]),
                torch.log1p(batch["n_bonds"]),
                torch.log1p(batch["n_junc"]),
                torch.log1p(n_pairs_off),
            ],
            dim=1,
        )
        gstd = (batch["global"] - self.global_mean) / self.global_scale
        blocks = [sum_scaled, mean_off, sq_mean, diag_mean, mean_a, mean_b, mean_w]
        z = torch.cat([ln(x) for ln, x in zip(self.block_norms, blocks)] + [counts, gstd], dim=1)
        return self.reader(z).view(-1)

    def _attn_pool(self, h_off, graph, off, B, cnt_off):
        """Segment softmax over each graph's off-diagonal pairs, three queries."""
        scores = (h_off @ self.attn_q.t()) / (WIDTH ** 0.5)
        scores = scores.masked_fill(off.unsqueeze(1) < 0.5, -1e9)
        amax = h_off.new_full((B, 3), -1e9)
        amax.index_reduce_(0, graph, scores.detach(), "amax", include_self=True)
        e = torch.exp(scores - amax[graph])
        e = e * off.unsqueeze(1)
        denom = e.new_zeros((B, 3)).index_add_(0, graph, e).clamp(min=1e-9)
        wts = e / denom[graph]
        s_off = h_off.new_zeros((B, WIDTH)).index_add_(0, graph, wts[:, 0:1] * h_off)
        a_mean = h_off.new_zeros((B, WIDTH)).index_add_(0, graph, wts[:, 1:2] * h_off)
        # weighted *sum* of squares: the readout divides by the same pair count,
        # so this keeps the "second moment" semantics of the plain path
        a_sq = h_off.new_zeros((B, WIDTH)).index_add_(0, graph, wts[:, 2:3] * h_off * h_off)
        n_pairs_off = cnt_off.clamp(min=1.0)
        # with uniform weights s_off is the pair mean, so multiply back by
        # sqrt(n_pairs) to keep the "count-scaled sum" semantics of block 0
        return s_off * n_pairs_off.sqrt().unsqueeze(1), a_mean, a_sq * n_pairs_off.unsqueeze(1)


# ---------------------------------------------------------------------------
# arm B
# ---------------------------------------------------------------------------


def build_kernels(adj_dense: torch.Tensor, adj_typed: torch.Tensor, node_mask: torch.Tensor,
                  cached: torch.Tensor | None = None):
    """Static operator library for one batch (no grad; deterministic).

    Heat kernels are full matrix functions of the symmetric normalized
    Laplacian (``Q exp(-t L) Q^T``): no eigenvector selection, no sign
    convention, stable under spectral multiplicity.  Typed operators are
    one-step normalized typed adjacencies.  Every operator is exactly zero on
    padded rows/columns, so padding cannot contribute to any trace.
    """
    B, D, _ = adj_dense.shape
    mask = node_mask.to(adj_dense.dtype)
    eye = torch.eye(D, device=adj_dense.device, dtype=adj_dense.dtype).unsqueeze(0)
    if cached is not None:
        with torch.no_grad():
            pair = mask.unsqueeze(2) * mask.unsqueeze(1)
            return [cached[..., i] * pair for i in range(cached.shape[-1])]
    with torch.no_grad():
        deg = adj_dense.sum(-1).clamp(min=1.0)
        dinv = deg.rsqrt() * mask
        ahat = adj_dense * dinv.unsqueeze(2) * dinv.unsqueeze(1)
        lap = eye - ahat
        pair = mask.unsqueeze(2) * mask.unsqueeze(1)
        lap = lap * pair + eye * (1.0 - mask.unsqueeze(2))
        lam, q = torch.linalg.eigh(lap)
        ops = []
        for t in HEAT_SCALES:
            ops.append((q * torch.exp(-t * lam).unsqueeze(1)) @ q.transpose(1, 2))
        for bt in TYPED_OPS:
            ops.append(adj_typed[..., bt - 1] * dinv.unsqueeze(2) * dinv.unsqueeze(1))
        ops = [k * pair for k in ops]
    return ops


class SpectralJoint(nn.Module):
    """Multi-point joint invariants of static whole-graph operators."""

    #: declared triple set: (kernel index of K_t, K_s, K_q)
    H = len(HEAT_SCALES)
    T = len(TYPED_OPS)

    def __init__(self, norm: Mapping[str, Any], *, n_chan: int = 6, reader_hidden: int = 224,
                 dropout: float = 0.0):
        super().__init__()
        self.encoders = P3Encoders(dropout=dropout)
        self.endpoint = EndpointState(dropout=dropout)
        self.ch_u = nn.Linear(WIDTH, n_chan)
        self.ch_v = nn.Linear(WIDTH, n_chan)
        self.ch_w = nn.Linear(WIDTH, n_chan)
        self.n_chan = int(n_chan)
        Hh, Tt = self.H, self.T
        heat = list(range(Hh))
        typed = list(range(Hh, Hh + Tt))
        triples = []
        triples += [(t, s, q) for t in heat for s in heat for q in heat]        # HHH
        triples += [(t, s, q) for t in typed for s in heat for q in heat]      # THH
        triples += [(t, s, q) for t in typed for s in typed for q in heat]     # TTH
        triples += [(t, t, t) for t in typed]                                 # TTT (same type)
        self.triples = triples
        self.kernel_pairs = [(i, j) for i in range(len(KERNEL_NAMES)) for j in range(i, len(KERNEL_NAMES))]
        inv_dim = self.n_chan * (len(self.triples) + len(self.kernel_pairs) + 4)
        self.inv_dim = int(inv_dim)
        # separate per-sample LayerNorm per invariant family: the raw trace
        # scales differ by ~4 orders of magnitude (triple ~4e-4, pair ~2e-3,
        # channel sums ~6) and a single shared norm would suppress the triple
        # block that carries the new computation.
        self.tri_norm = nn.LayerNorm(self.n_chan * len(self.triples))
        self.pair_norm = nn.LayerNorm(self.n_chan * len(self.kernel_pairs))
        self.first_norm = nn.LayerNorm(self.n_chan * 4)
        reader_in = inv_dim + 6 + GLOBAL_DIM
        self.reader = MLP(reader_in, reader_hidden, 1, n_hidden=2, dropout=dropout)
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        a, b, w = self.encoders(batch)
        s = self.endpoint(a, w, batch)
        B, D = int(batch["n_graphs"]), int(batch["Nmax"])
        R = self.n_chan
        ops = build_kernels(batch["adj_dense"], batch["adj_typed"], batch["node_mask"],
                            batch.get("kern_dense"))
        K = len(ops)
        uv = torch.tanh(self.ch_u(s))
        vv = torch.tanh(self.ch_v(s))
        wv = torch.tanh(self.ch_w(s))
        zeros = uv.new_zeros((B * D, R))
        U = zeros.index_copy(0, batch["atom_slot"], uv).view(B, D, R)
        V = zeros.index_copy(0, batch["atom_slot"], vv).view(B, D, R)
        Wc = zeros.index_copy(0, batch["atom_slot"], wv).view(B, D, R)
        n_atoms = batch["n_atoms"].unsqueeze(1)

        # --- pair invariants: tr(D_u K_t D_v K_s) -----------------------------
        fro = torch.stack([ops[k].flatten(1).norm(dim=1) for k in range(K)], dim=1)  # [B,K]
        pair_vals = []
        for (t, s_i) in self.kernel_pairs:
            o = ops[t] * ops[s_i].transpose(1, 2)
            pv = torch.einsum("bij,bir,bjr->br", o, U, V)
            pair_vals.append(pv / (fro[:, t] * fro[:, s_i]).unsqueeze(1).clamp(min=1e-6) / n_atoms)
        pair_vec = torch.cat(pair_vals, dim=1)

        # --- triple invariants: tr(D_u K_t D_v K_s D_w K_q) --------------------
        # M[t,s] = K_t diag(v) K_s, per channel: [B,R,K,K,D,D]
        m_all = []
        for t in range(K):
            scaled = ops[t].unsqueeze(1) * V.transpose(1, 2).unsqueeze(2)       # [B,R,D,D]
            for s_i in range(K):
                m_all.append(torch.matmul(scaled, ops[s_i].unsqueeze(1)))      # [B,R,D,D]
        m_stack = torch.stack(m_all, dim=2)                                    # [B,R,K*K,D,D]
        outer = (U.unsqueeze(2) * Wc.unsqueeze(1)).permute(0, 3, 1, 2)         # [B,R,D,D]
        g_stack = torch.stack([ops[q].transpose(1, 2) for q in range(K)], dim=1).unsqueeze(1)
        g_stack = outer.unsqueeze(2) * g_stack                                 # [B,R,K,D,D]
        tri_all = torch.einsum("brpik,brqik->brpq", m_stack, g_stack)           # [B,R,K*K,K]
        tri_pairs = torch.as_tensor(
            [t * K + s_i for (t, s_i, _q) in self.triples], dtype=torch.int64, device=s.device
        )
        tri_q = torch.as_tensor([q for (_t, _s, q) in self.triples], dtype=torch.int64, device=s.device)
        tri_t = torch.as_tensor([t for (t, _s, _q) in self.triples], dtype=torch.int64, device=s.device)
        tri_s = torch.as_tensor([s_i for (_t, s_i, _q) in self.triples], dtype=torch.int64, device=s.device)
        tri = tri_all.index_select(2, tri_pairs).gather(
            3, tri_q.view(1, 1, -1, 1).expand(B, R, tri_q.numel(), 1)
        ).squeeze(-1)                                                          # [B,R,T]
        scale = (fro[:, tri_t] * fro[:, tri_s] * fro[:, tri_q]).clamp(min=1e-6) * n_atoms
        tri = tri / scale.unsqueeze(1)
        tri_vec = tri.reshape(B, -1)

        # --- one-point counterpart and plain channel sums ----------------------
        uvw = (U * V * Wc).sum(1)                                              # [B,R]
        sums = torch.cat([U.sum(1), V.sum(1), Wc.sum(1), uvw], dim=1)          # [B,4R]
        counts = torch.stack(
            [
                torch.log1p(batch["n_atoms"]),
                torch.log1p(batch["n_bonds"]),
                torch.log1p(batch["n_junc"]),
                torch.log1p(batch["n_atoms"] * (batch["n_atoms"] - 1.0).clamp(min=0.0)),
                (batch["n_junc"] / batch["n_bonds"].clamp(min=1.0)),
                (batch["n_bonds"] / batch["n_atoms"].clamp(min=1.0)),
            ],
            dim=1,
        )
        inv = torch.cat([self.tri_norm(tri_vec), self.pair_norm(pair_vec),
                         self.first_norm(sums)], dim=1)
        gstd = (batch["global"] - self.global_mean) / self.global_scale
        z = torch.cat([inv, counts, gstd], dim=1)
        return self.reader(z).view(-1)

    def _attn_pool(self, h_off, graph, off, B, cnt_off):
        """Segment softmax over each graph's off-diagonal pairs, three queries."""
        scores = (h_off @ self.attn_q.t()) / (WIDTH ** 0.5)
        scores = scores.masked_fill(off.unsqueeze(1) < 0.5, -1e9)
        amax = h_off.new_full((B, 3), -1e9)
        amax.index_reduce_(0, graph, scores.detach(), "amax", include_self=True)
        e = torch.exp(scores - amax[graph])
        e = e * off.unsqueeze(1)
        denom = e.new_zeros((B, 3)).index_add_(0, graph, e).clamp(min=1e-9)
        wts = e / denom[graph]
        s_off = h_off.new_zeros((B, WIDTH)).index_add_(0, graph, wts[:, 0:1] * h_off)
        a_mean = h_off.new_zeros((B, WIDTH)).index_add_(0, graph, wts[:, 1:2] * h_off)
        # weighted *sum* of squares: the readout divides by the same pair count,
        # so this keeps the "second moment" semantics of the plain path
        a_sq = h_off.new_zeros((B, WIDTH)).index_add_(0, graph, wts[:, 2:3] * h_off * h_off)
        n_pairs_off = cnt_off.clamp(min=1.0)
        # with uniform weights s_off is the pair mean, so multiply back by
        # sqrt(n_pairs) to keep the "count-scaled sum" semantics of block 0
        return s_off * n_pairs_off.sqrt().unsqueeze(1), a_mean, a_sq * n_pairs_off.unsqueeze(1)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------


def _load_encoders(model: nn.Module, path: Path, device) -> dict:
    state = torch.load(path, map_location="cpu", weights_only=False)
    for name in ("atom_encoder", "bond_encoder", "junction_encoder"):
        getattr(model.encoders, name).load_state_dict(
            {k: v.to(device) for k, v in state[name].items()}
        )
    return {
        "source": str(path),
        "sha256": C.sha256_file(path),
        "keys": sorted(state.keys()),
        "encoder_layers_loaded": True,
    }


def build_model(arm: str, norm: Mapping[str, Any], seed: int, device, *, ssl: bool = True,
                dropout: float = 0.0, **kwargs):
    torch.manual_seed(int(seed))
    if arm == "A" or arm == "A_SSL":
        model = PairComposition(norm, dropout=dropout, **kwargs)
    elif arm == "A_attn":
        model = PairComposition(norm, dropout=dropout, pool="attn", **kwargs)
    elif arm == "B":
        model = SpectralJoint(norm, dropout=dropout, **kwargs)
    else:
        raise ValueError(f"unknown arm {arm!r}")
    info: dict[str, Any] = {"arm": arm, "ssl_encoders": bool(ssl), "seed": int(seed)}
    if ssl:
        info["p3_ssl"] = _load_encoders(model, C.P3_SSL_S0, device)
    model.to(device)
    info["parameter_count"] = C.parameter_count(model)
    info["initial_state_sha256"] = C.state_hash(model)
    return model, info

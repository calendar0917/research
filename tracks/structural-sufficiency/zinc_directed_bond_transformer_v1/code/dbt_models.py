"""Directed-Bond structural Transformer (DBT) for ZINC raw-y regression.

One shared local encoder set, one full-graph relation-conditioned interaction
trunk, one reader.  Total inference parameters <= 500,000.

Computation (see ``DESIGN.md`` for the full declaration)
-------------------------------------------------------

1. **Shared local encoding.**  The frozen P3 atom / physical-bond / junction
   encoders (seed-0 *train-only masked-SSL* initialisation) produce ``a_v``,
   ``b_e``, ``w_j`` once.  They are fine-tuned with the supervised loss.

2. **Static directed-bond token.**  Every physical bond ``{u,v}`` yields two
   objects ``e=(u->v)`` and ``reverse(e)``.  All directed objects share one
   ``InitMLP``::

       h_e^0 = InitMLP([a_u, a_v, b_uv, J_src, J_dst, log1p(n_src), log1p(n_dst)])

   where ``J_src`` is the mean of ``w`` over the *real* junctions centred on
   ``u`` that pair ``v`` with each other neighbour, and ``J_dst`` the same at
   ``v``.  Source and target content are concatenated by role (never summed),
   so ``reverse(e)`` naturally swaps the two halves.  ``n_src = deg(u)-1`` and
   ``n_dst = deg(v)-1`` are kept explicitly, so an empty junction set stays
   distinguishable from a mean of zero.

3. **Multi-layer full-graph relation interaction.**  Four pre-LN blocks over
   *all* directed objects of the same molecule (padding and other molecules are
   masked).  Per head and per pair::

       alpha_ef = softmax_f( q_e . k_f / sqrt(d_head) + B_h(r_ef) )
       value_ef = V_h(h_f) * (1 + tanh(G_h(r_ef))) + U_h(r_ef)
       context_e = sum_f alpha_ef * value_ef

   The relation vector ``r_ef`` is a *fixed* label-free function of the raw
   graph: the exact simple-path counts ``S_l`` for ``l=1..12`` and the exact
   shortest-path distance of the four endpoint blocks
   ``(u,p) (u,q) (v,p) (v,q)``, plus the four endpoint-equality flags and five
   deterministic bond-orientation flags.  A shared ``RelationMLP`` embeds it
   once; ``B``, ``G``, ``U`` are learned per layer.  The relation therefore
   enters both the attention logits *and* the content that is read, per
   channel.  There is no local-neighbour mask and no hidden state is ever
   propagated along the line graph: every layer is a complete same-molecule
   interaction.

4. **One amplitude-preserving readout.**  With ``H = {h_e}`` over directed
   objects::

       S1 = 0.5 * sum_e h_e          (each physical bond counted once)
       S2 = 0.5 * sum_e h_e^2
       z  = [S1, S2, sum_v a_v, sum_v a_v^2, global33, log counts]
       y  = Reader(z)

   No per-molecule LayerNorm is applied to any pooled block.  ``S1``/``S2`` are
   divided by fixed *train* mean counts so their scale is O(1); the ratio
   between molecules is preserved.

``source_degree``/``target_degree``/graph size are injected into every layer
through one shared 4->64 projection (declared, counted).
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
_SSUF = _EXP.parent
_V1 = _SSUF / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_models as M  # noqa: E402
import dbt_data as BD  # noqa: E402

MLP = M.MLP
P3Encoders = M.P3Encoders

WIDTH = 64
HEADS = 4
HEAD_DIM = WIDTH // HEADS
FF_MULT = 2
R_DIM = 24
R_HIDDEN = 32
N_LAYERS = 4
INIT_PROJ = 32          # per-role linear projection width in the token init
INIT_HIDDEN = 96        # hidden width of the token init MLP
ATOM_STATS_DIM = BD.ATOM_OUT
N_COUNTS = 4


class RelationEncoder(nn.Module):
    """Fixed raw relation -> shared learned relation embedding ``r_bar``.

    ``path_proj`` is applied to the 12 exact path counts of every endpoint
    block (shared weights; the block position in the concatenation keeps the
    role).  ``dist_emb`` is a shared distance embedding table.  The four blocks
    are concatenated, endpoint-equality and bond-orientation flags appended,
    and one MLP produces ``r_bar``.
    """

    def __init__(self, *, path_emb: int = BD.PATH_EMB_DIM, dist_emb: int = BD.DIST_EMB_DIM,
                 hidden: int = R_HIDDEN, out_dim: int = R_DIM):
        super().__init__()
        self.path_proj = nn.Linear(BD.PATH_COUNTS, path_emb)
        self.dist_emb = nn.Embedding(BD.DIST_BUCKETS, dist_emb)
        self.mlp = MLP(BD.REL_RAW_DIM, hidden, out_dim, n_hidden=1)

    def pair_embedding(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Per atom-pair embedding ``[B, Nmax, Nmax, path+dist]``."""
        path = torch.log1p(batch["pair_path"].to(torch.float32))
        pe = self.path_proj(path)
        de = self.dist_emb(batch["pair_dist"].to(torch.long))
        return torch.cat([pe, de], dim=-1)

    def gather_blocks(self, pair_emb: torch.Tensor, src: torch.Tensor, dst: torch.Tensor):
        """Gather the four endpoint blocks on the directed-pair grid.

        The grid is ``[B, Dmax, Dmax]``; ``src``/``dst`` are the padded
        slot -> batch-local atom tables.  Padded slots point at atom 0, which
        is harmless because every padded row/column is masked out.
        """
        B, Dm = src.shape
        N = int(round((pair_emb.shape[1]) ** 0.5)) if pair_emb.dim() == 4 else 0
        N = int(pair_emb.shape[1])
        base = (torch.arange(B, device=src.device, dtype=torch.long) * (N * N)).view(B, 1, 1)
        flat = pair_emb.reshape(B * N * N, pair_emb.shape[-1])

        def gather(i: torch.Tensor, j: torch.Tensor) -> torch.Tensor:
            idx = base + i.unsqueeze(2) * N + j.unsqueeze(1)
            return flat.index_select(0, idx.reshape(-1)).view(B, Dm, Dm, -1)

        blocks = [
            gather(src, src),   # (u, p)
            gather(src, dst),   # (u, q)
            gather(dst, src),   # (v, p)
            gather(dst, dst),   # (v, q)
        ]
        return torch.cat(blocks, dim=-1)

    def forward(self, pair_emb: torch.Tensor, src: torch.Tensor, dst: torch.Tensor,
                bond: torch.Tensor) -> torch.Tensor:
        r = self.gather_blocks(pair_emb, src, dst)
        # four endpoint-equality flags, one per block, kept by position
        eq = torch.stack(
            [
                (src.unsqueeze(2) == src.unsqueeze(1)),
                (src.unsqueeze(2) == dst.unsqueeze(1)),
                (dst.unsqueeze(2) == src.unsqueeze(1)),
                (dst.unsqueeze(2) == dst.unsqueeze(1)),
            ],
            dim=-1,
        ).to(r.dtype)
        same_bond = (bond.unsqueeze(2) == bond.unsqueeze(1)).to(r.dtype)
        is_reverse = (
            (src.unsqueeze(2) == dst.unsqueeze(1)) & (dst.unsqueeze(2) == src.unsqueeze(1))
            & (bond.unsqueeze(2) == bond.unsqueeze(1))
            & ~(src.unsqueeze(2) == src.unsqueeze(1))
        ).to(r.dtype)
        is_self = (src.unsqueeze(2) == src.unsqueeze(1)) & (dst.unsqueeze(2) == dst.unsqueeze(1))
        same_src = (src.unsqueeze(2) == src.unsqueeze(1)).to(r.dtype)
        same_dst = (dst.unsqueeze(2) == dst.unsqueeze(1)).to(r.dtype)
        flags = torch.stack(
            [same_bond, is_reverse, is_self.to(r.dtype), same_src, same_dst], dim=-1
        )
        raw = torch.cat([r, eq, flags], dim=-1)
        if raw.shape[-1] != BD.REL_RAW_DIM:
            raise RuntimeError(f"relation raw dim {raw.shape[-1]} != {BD.REL_RAW_DIM}")
        return self.mlp(raw)


class DBTLayer(nn.Module):
    """Pre-LN full-graph relation-conditioned interaction block."""

    def __init__(self, *, d: int = WIDTH, heads: int = HEADS, r_dim: int = R_DIM,
                 ff_mult: int = FF_MULT, dropout: float = 0.05):
        super().__init__()
        if d % heads:
            raise ValueError("width must be divisible by heads")
        self.d = int(d)
        self.heads = int(heads)
        self.hd = self.d // self.heads
        self.ln1 = nn.LayerNorm(d)
        self.ln2 = nn.LayerNorm(d)
        self.qkv = nn.Linear(d, 3 * d)
        self.out = nn.Linear(d, d)
        # relation -> per-head logit bias / per-channel value gate / value bias
        self.rel_b = nn.Linear(r_dim, self.heads)
        self.rel_g = nn.Linear(r_dim, self.d)
        self.rel_u = nn.Linear(r_dim, self.d)
        self.ff = nn.Sequential(
            nn.Linear(d, ff_mult * d),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ff_mult * d, d),
        )
        self.attn_drop = nn.Dropout(dropout)
        self.res_drop = nn.Dropout(dropout)

    def forward(self, h: torch.Tensor, r_bar: torch.Tensor, key_mask: torch.Tensor,
                static_bias: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, Dm, _ = h.shape
        H, hd = self.heads, self.hd
        x = self.ln1(h + static_bias)
        qkv = self.qkv(x).view(B, Dm, 3, H, hd)
        q = qkv[:, :, 0].permute(0, 2, 1, 3)          # [B,H,D,dh]
        k = qkv[:, :, 1].permute(0, 2, 1, 3)
        v = qkv[:, :, 2].permute(0, 2, 1, 3)

        bias = self.rel_b(r_bar).permute(0, 3, 1, 2)  # [B,H,D,D]
        scores = torch.matmul(q, k.transpose(-1, -2)) / math.sqrt(hd) + bias
        scores = scores.masked_fill(~key_mask[:, None, None, :], -1.0e9)
        attn = self.attn_drop(torch.softmax(scores, dim=-1))

        gate = torch.tanh(self.rel_g(r_bar)).view(B, Dm, Dm, H, hd).permute(0, 3, 1, 2, 4)
        uadd = self.rel_u(r_bar).view(B, Dm, Dm, H, hd).permute(0, 3, 1, 2, 4)
        valr = v.unsqueeze(2) * (1.0 + gate) + uadd     # [B,H,D,D,dh]
        # context_e = sum_f alpha_ef * value_ef, per head then concatenated
        ctx = torch.bmm(
            attn.reshape(B * H * Dm, 1, Dm),
            valr.reshape(B * H * Dm, Dm, hd),
        ).reshape(B, H, Dm, hd).permute(0, 2, 1, 3).reshape(B, Dm, self.d)

        h = h + self.res_drop(self.out(ctx))
        h = h + self.res_drop(self.ff(self.ln2(h)))
        return h * key_mask.unsqueeze(-1), attn


class DirectedBondTransformer(nn.Module):
    """DBT: shared local encoders + full-graph directed-bond trunk + one reader."""

    def __init__(self, norm: Mapping[str, Any], *, width: int = WIDTH, heads: int = HEADS,
                 n_layers: int = N_LAYERS, r_dim: int = R_DIM, r_hidden: int = R_HIDDEN,
                 ff_mult: int = FF_MULT, dropout: float = 0.05,
                 init_proj: int = INIT_PROJ, init_hidden: int = INIT_HIDDEN,
                 reader_hidden: int = 128, graph_scales: Mapping[str, Any] | None = None):
        super().__init__()
        self.width = int(width)
        self.n_layers = int(n_layers)
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))
        n_dir_ref = float((graph_scales or {}).get("n_dir_ref", 1.0) or 1.0)
        n_atom_ref = float((graph_scales or {}).get("n_atom_ref", 1.0) or 1.0)
        self.register_buffer("n_dir_ref", torch.tensor(max(n_dir_ref, 1.0), dtype=torch.float32))
        self.register_buffer("n_atom_ref", torch.tensor(max(n_atom_ref, 1.0), dtype=torch.float32))

        # -- shared local encoders (P3, seeded from train-only masked SSL) -----
        self.encoders = P3Encoders(dropout=0.0)
        A = BD.ATOM_OUT
        Bn = BD.BOND_OUT
        # -- static directed-bond token (role-preserving projections) ---------
        self.proj_src = nn.Linear(A, init_proj)
        self.proj_dst = nn.Linear(A, init_proj)
        self.proj_bond = nn.Linear(Bn, init_proj)
        self.proj_jsrc = nn.Linear(A, init_proj)
        self.proj_jdst = nn.Linear(A, init_proj)
        init_in = 5 * init_proj + 2
        self.init_mlp = MLP(init_in, init_hidden, width, n_hidden=1, dropout=0.0)
        self.init_norm = nn.LayerNorm(width)

        # -- static count injection (shared by every layer) --------------------
        self.static_proj = nn.Linear(4, width)

        # -- shared relation pre-encoding, per-layer B/G/U ---------------------
        self.relation = RelationEncoder(hidden=r_hidden, out_dim=r_dim)
        self.layers = nn.ModuleList(
            [DBTLayer(d=width, heads=heads, r_dim=r_dim, ff_mult=ff_mult, dropout=dropout)
             for _ in range(self.n_layers)]
        )

        # -- one reader over both directed-bond moments + atom moments --------
        reader_in = 2 * width + 2 * A + BD.GLOBAL_DIM + N_COUNTS
        self.reader = MLP(reader_in, reader_hidden, 1, n_hidden=2, dropout=dropout)

    # -- init ---------------------------------------------------------------
    def load_ssl(self, path: Path, device) -> dict:
        state = torch.load(path, map_location="cpu", weights_only=False)
        for name in ("atom_encoder", "bond_encoder", "junction_encoder"):
            getattr(self.encoders, name).load_state_dict(
                {k: v.to(device) for k, v in state[name].items()}
            )
        return {"source": str(path), "sha256": C.sha256_file(path),
                "keys": sorted(state.keys()), "encoder_layers_loaded": True}

    # -- token construction -------------------------------------------------
    def _segment_mean(self, w: torch.Tensor, index: torch.Tensor, ptr: torch.Tensor,
                      n_obj: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Mean of ``w[index]`` per object range ``ptr`` (empty range -> 0).

        ``ptr`` is the object-level pointer ``[n_obj+1]``; ``index`` are the
        (batch-local) rows of ``w`` that belong to each object, concatenated.
        The true count is carried separately so an empty junction set is not
        confused with a mean of zero.
        """
        counts = ptr[1:] - ptr[:-1]
        if index.numel() != int(counts.sum()):
            raise RuntimeError("junction incidence does not match the object pointer")
        acc = w.new_zeros((n_obj, w.shape[1])).index_add_(
            0, torch.repeat_interleave(torch.arange(n_obj, device=w.device), counts),
            w.index_select(0, index),
        )
        mean = acc / counts.clamp(min=1).unsqueeze(1).to(w.dtype)
        return mean, counts.to(w.dtype)

    def build_tokens(self, batch: Mapping[str, torch.Tensor]):
        a, b, w = self.encoders(batch)                       # [A,96] [E,96] [J,96]
        di = batch["dir_index"]
        n_dir = int(di.shape[0])
        src, dst = di[:, 0], di[:, 1]
        bond = batch["dir_bond"]

        jsrc, n_src = self._segment_mean(w, batch["src_junc"], batch["src_obj_ptr"], n_dir)
        jdst, n_dst = self._segment_mean(w, batch["dst_junc"], batch["dst_obj_ptr"], n_dir)

        x = torch.cat(
            [
                self.proj_src(a[src]),
                self.proj_dst(a[dst]),
                self.proj_bond(b[bond]),
                self.proj_jsrc(jsrc),
                self.proj_jdst(jdst),
                torch.log1p(n_src).unsqueeze(1),
                torch.log1p(n_dst).unsqueeze(1),
            ],
            dim=1,
        )
        h = self.init_norm(self.init_mlp(x))

        # dense padded layout [B, Dmax, width]
        B = int(batch["n_graphs"])
        Dm = int(batch["Dmax"])
        ptr = batch["dir_ptr"]
        counts = ptr[1:] - ptr[:-1]
        graph = torch.repeat_interleave(torch.arange(B, device=h.device), counts)
        local = torch.arange(n_dir, device=h.device) - ptr[graph]
        slot = graph * Dm + local
        dense = h.new_zeros((B * Dm, self.width)).index_copy(0, slot, h).view(B, Dm, self.width)

        # dense side tables (padded slots point at atom/bond 0, always masked)
        atom_ptr = batch["atom_ptr"]
        src_local = src - atom_ptr[graph]
        dst_local = dst - atom_ptr[graph]
        src_slot = dense.new_zeros((B * Dm,), dtype=torch.long).index_copy(
            0, slot, src_local).view(B, Dm)
        dst_slot = src_slot.new_zeros((B * Dm,)).index_copy(0, slot, dst_local).view(B, Dm)
        bond_slot = src_slot.new_zeros((B * Dm,)).index_copy(0, slot, bond).view(B, Dm)
        key_mask = torch.zeros((B * Dm,), dtype=torch.bool, device=h.device).index_fill_(0, slot, True)
        key_mask = key_mask.view(B, Dm)

        return {
            "a": a, "b": b, "w": w, "h": h, "dense": dense,
            "src_slot": src_slot, "dst_slot": dst_slot, "bond_slot": bond_slot,
            "key_mask": key_mask, "src": src, "dst": dst,
            "slot": slot, "graph": graph,
        }

    # -- forward -------------------------------------------------------------
    def encode(self, batch: Mapping[str, torch.Tensor]):
        """Run the shared encoders and the full-graph trunk; return tokens + h."""
        tok = self.build_tokens(batch)
        a, dense, key_mask = tok["a"], tok["dense"], tok["key_mask"]
        B = int(batch["n_graphs"])
        Dm = int(batch["Dmax"])
        device = dense.device
        n_atoms = batch["n_atoms"]
        n_bonds = batch["n_bonds"]
        atom_deg = torch.zeros((a.shape[0],), dtype=torch.float32, device=device).index_add_(
            0, batch["bond_index"].reshape(-1),
            torch.ones(batch["bond_index"].numel(), device=device,
                       dtype=torch.float32),
        )
        slot = tok["slot"]
        deg_src = atom_deg.new_zeros((B * Dm,)).index_copy(
            0, slot, atom_deg[tok["src"]]).view(B, Dm)
        deg_dst = atom_deg.new_zeros((B * Dm,)).index_copy(
            0, slot, atom_deg[tok["dst"]]).view(B, Dm)
        # static count injection: source/target degree + graph size
        stats = torch.stack(
            [
                torch.log1p(deg_src),
                torch.log1p(deg_dst),
                torch.log1p(n_atoms).unsqueeze(1).expand(B, Dm),
                torch.log1p(n_bonds).unsqueeze(1).expand(B, Dm),
            ],
            dim=-1,
        )
        static_bias = self.static_proj(stats) * key_mask.unsqueeze(-1).to(stats.dtype)

        pair_emb = self.relation.pair_embedding(batch)
        r_bar = self.relation(pair_emb, tok["src_slot"], tok["dst_slot"], tok["bond_slot"])

        h = dense
        for layer in self.layers:
            h, _ = layer(h, r_bar, key_mask, static_bias)
        return tok, h, r_bar

    # -- amplitude-preserving readout ----------------------------------------
    def pool(self, tok, h, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        a = tok["a"]
        B, Dm = h.shape[0], h.shape[1]
        graph = batch["dir_graph"]
        h_flat = h.reshape(B * Dm, self.width).index_select(0, tok["slot"])
        s1 = h_flat.new_zeros((B, self.width)).index_add_(0, graph, h_flat)
        s2 = h_flat.new_zeros((B, self.width)).index_add_(0, graph, h_flat * h_flat)
        s1 = 0.5 * s1 / self.n_dir_ref
        s2 = 0.5 * s2 / self.n_dir_ref

        atom_sum = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a)
        atom_sq = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a * a)
        atom_sum = atom_sum / self.n_atom_ref
        atom_sq = atom_sq / self.n_atom_ref

        gstd = (batch["global"] - self.global_mean) / self.global_scale
        counts = torch.stack(
            [
                torch.log1p(batch["n_atoms"]),
                torch.log1p(batch["n_bonds"]),
                torch.log1p(2.0 * batch["n_bonds"]),
                torch.log1p(batch["n_junc"]),
            ],
            dim=1,
        )
        return torch.cat([s1, s2, atom_sum, atom_sq, gstd, counts], dim=1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        tok, h, _ = self.encode(batch)
        return self.reader(self.pool(tok, h, batch)).view(-1)


def build_model(norm: Mapping[str, Any], seed: int, device, *,
                graph_scales: Mapping[str, Any] | None = None,
                dropout: float = 0.05) -> tuple[nn.Module, dict[str, Any]]:
    """Deterministic construction + train-only SSL encoder initialisation."""
    torch.manual_seed(int(seed))
    torch.cuda.manual_seed_all(int(seed))
    model = DirectedBondTransformer(norm, dropout=dropout, graph_scales=graph_scales)
    model = model.to(device)
    info: dict[str, Any] = {"seed": int(seed), "ssl": None,
                            "architecture": "directed_bond_transformer_v1"}
    path = C.P3_SSL_S0 if int(seed) == 0 else C.P3_SSL_S1
    if path.exists():
        info["ssl"] = model.load_ssl(path, device)
    else:
        info["ssl"] = {"source": None, "note": "SSL encoder unavailable; random init"}
    info["parameters"] = C.parameter_count(model)
    return model, info

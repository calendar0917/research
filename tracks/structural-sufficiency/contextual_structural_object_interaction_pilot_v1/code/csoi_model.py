"""Contextual Structural Object Interaction (CSOI), pilot v1.

What the model does, in the order the information flows:

1. ``phi`` (shared FrameEncoder): every rooted frame keeps its **complete raw
   attribute matrix** (9 atom categorical fields, 3 bond categorical fields,
   node mask, edge mask, object size, root-fixed ordering) and is embedded to
   ``hidden=64``.  No template id, no early mean over a fragment.
2. every fragment is a **token set** ``X_F = {(frame, distance bucket, mass)}``
   where the mass is the frame multiplicity of the rooted object it belongs to
   (``1 / n_winning_frames`` summed over the objects that hit the same
   ``(frame, bucket)`` cell).  Fragments are never pooled before the
   interaction.
3. ``CSOI_GLOBAL`` / ``CSOI_REL``: one layer of 4-head cross-attention from the
   S tokens into the C tokens, ``score = q.k/sqrt(dh) + relation_bias + log(mass)``,
   then ``r_i = MLP([x_i, m_i, x_i*m_i, x_i-m_i, cut_bond])`` per S token and a
   multiplicity-weighted pool of ``r_i``.
4. ``CSOI_ADD``: the same token encoder and the same ``phi``, but S and C are
   each aggregated separately -- no cross-set interaction at all.
5. cut response ``f(S|C) = beta + additive_S(S) + additive_C(C) + b(e)``
   (+ the pooled interaction term in the two attention arms).  A learned
   null-context token carries the ``C = {}`` fallback parses.

Only one attention layer, no GNN backbone, no pretraining, no template
embeddings.  The cut enumeration, molecule-level averaging and fallback
definition are the parent round's and are not changed here.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import csoi_common as C

CSOI_ARMS = ("CSOI_ADD", "CSOI_GLOBAL", "CSOI_REL")


def _mlp(sizes: Sequence[int]) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(int(sizes[i]), int(sizes[i + 1])))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


class FrameEncoder(nn.Module):
    """Rooted frame -> 64-d object token (same input layout as the parent phi)."""

    def __init__(self, atom_dims: Sequence[int], bond_dims: Sequence[int],
                 dims: Mapping[str, int], out: int) -> None:
        super().__init__()
        cat = int(dims["cat_dim"])
        self.atom_emb = nn.ModuleList([nn.Embedding(int(d), cat) for d in atom_dims])
        self.bond_emb = nn.ModuleList([nn.Embedding(int(d), cat) for d in bond_dims])
        self.atom_mlp = _mlp([len(atom_dims) * cat, dims["atom_hidden"], dims["atom_out"]])
        self.bond_mlp = _mlp([len(bond_dims) * cat, dims["bond_hidden"], dims["bond_out"]])
        self.frame_mlp = nn.Sequential(
            nn.Linear(int(dims["frame_input"]), int(dims["frame_hidden"])),
            nn.SiLU(),
            nn.Linear(int(dims["frame_hidden"]), int(out)),
            nn.SiLU(),
            nn.LayerNorm(int(out)),
        )
        self.n_fields = len(atom_dims)
        self.n_bond_fields = len(bond_dims)

    def forward(self, atom_vals: torch.Tensor, bond_vals: torch.Tensor,
                nmask: torch.Tensor, emask: torch.Tensor,
                size_oh: torch.Tensor) -> torch.Tensor:
        n = atom_vals.shape[0]
        a = torch.cat([emb(atom_vals[:, :, i]) for i, emb in enumerate(self.atom_emb)], dim=-1)
        a = self.atom_mlp(a) * nmask[:, :, None]
        b = torch.cat([emb(bond_vals[:, :, i]) for i, emb in enumerate(self.bond_emb)], dim=-1)
        b = self.bond_mlp(b) * emask[:, :, None]
        x = torch.cat([a.reshape(n, -1), b.reshape(n, -1), nmask, emask, emask, size_oh], dim=-1)
        return self.frame_mlp(x)


class CSOIModel(nn.Module):
    def __init__(self, cfg: Mapping[str, Any], atom_dims: Sequence[int],
                 bond_dims: Sequence[int], arm: str) -> None:
        super().__init__()
        if arm not in CSOI_ARMS:
            raise ValueError(f"unknown CSOI arm {arm}")
        self.arm = arm
        self.cfg = cfg
        dims = dict(cfg["dims"])
        d = cfg["csoi"]
        self.hidden = int(d["hidden"])
        self.heads = int(d["heads"])
        self.dh = int(d["head_dim"])
        if self.heads * self.dh != self.hidden:
            raise ValueError("heads * head_dim must equal hidden")
        self.entry_chunk = int(d["attention_entry_chunk"])
        self.dropout = float(d["dropout"])
        self.interaction = arm != "CSOI_ADD"
        self.use_relation = arm == "CSOI_REL"
        #: optional diagnostic capture (set to a dict to record attention internals)
        self._capture: dict[str, Any] | None = None

        self.phi = FrameEncoder(atom_dims, bond_dims, dims, int(d["frame_out"]))
        self.token_proj = nn.Linear(self.hidden, self.hidden)
        self.bucket_emb = nn.Embedding(C.N_BUCKETS, self.hidden)
        self.mult_proj = nn.Linear(2, self.hidden)
        self.token_ln = nn.LayerNorm(self.hidden)
        self.cut_proj = nn.Sequential(
            nn.Linear(int(dims["bond_concat"]), int(d["cut_hidden"])),
            nn.SiLU(),
            nn.Linear(int(d["cut_hidden"]), self.hidden),
        )
        self.null_cut = nn.Parameter(torch.zeros(self.hidden))
        self.null_ctx = nn.Parameter(torch.zeros(self.hidden))
        self.add_s = _mlp([self.hidden, d["add_hidden"], self.hidden, 1])
        self.add_c = _mlp([self.hidden, d["add_hidden"], self.hidden, 1])
        self.bond_bias = _mlp([self.hidden, 1])
        self.beta = nn.Parameter(torch.zeros(1))
        if self.interaction:
            self.q_proj = nn.Linear(self.hidden, self.hidden)
            self.k_proj = nn.Linear(self.hidden, self.hidden)
            self.v_proj = nn.Linear(self.hidden, self.hidden)
            self.r_mlp = _mlp([5 * self.hidden, d["r_hidden"], self.hidden])
            self.out = nn.Linear(self.hidden, 1)
            self.attn_drop = nn.Dropout(self.dropout)
        if self.use_relation:
            emb = int(d["rel_emb"])
            self.rel_ds = nn.Embedding(C.N_BUCKETS, emb)
            self.rel_dc = nn.Embedding(C.N_BUCKETS, emb)
            self.rel_dt = nn.Embedding(C.N_BUCKETS, emb)
            self.rel_cut = nn.Linear(self.hidden, emb)
            self.rel_mlp = nn.Sequential(nn.Linear(emb, int(d["rel_hidden"])), nn.SiLU(),
                                         nn.Linear(int(d["rel_hidden"]), self.heads))
        dt = np.zeros(C.N_BUCKETS * C.N_BUCKETS, np.int64)
        for a in range(C.N_BUCKETS):
            for b in range(C.N_BUCKETS):
                if a == C.NULL_BUCKET or b == C.NULL_BUCKET:
                    dt[a * C.N_BUCKETS + b] = C.NULL_BUCKET
                else:
                    dt[a * C.N_BUCKETS + b] = min(a + 1 + b, C.NULL_BUCKET - 1)
        self.register_buffer("dtot", torch.as_tensor(dt), persistent=False)
        C.deterministic_init(self, int(cfg["recipe"]["seed"]))

    # -- graph buffers ------------------------------------------------------
    def load_pack(self, pack: Mapping[str, Any]) -> None:
        def reg_long(name: str) -> None:
            self.register_buffer(name, torch.as_tensor(
                np.ascontiguousarray(np.asarray(pack[name]))).long(), persistent=False)

        for name in ("frag_ptr", "frag_frame", "frag_bucket", "parse_S", "parse_C",
                     "parse_cut_bond_row", "frame_atom", "frame_bond", "frame_k",
                     "atom_table", "bond_table"):
            reg_long(name)
        self.register_buffer("frag_weight", torch.as_tensor(
            np.ascontiguousarray(np.asarray(pack["frag_weight"]))).float(), persistent=False)
        self.register_buffer("mol_parse_ptr", torch.as_tensor(
            np.ascontiguousarray(np.asarray(pack["mol_parse_ptr"]))).long(), persistent=False)

        atom_table = np.asarray(pack["atom_table"])
        bond_table = np.asarray(pack["bond_table"])
        frame_atom = np.asarray(pack["frame_atom"])
        frame_bond = np.asarray(pack["frame_bond"])
        frame_k = np.asarray(pack["frame_k"]).astype(np.int64)
        n_frames, slots = frame_atom.shape
        atom_vals = atom_table[frame_atom]
        bond_vals = np.zeros((n_frames, frame_bond.shape[1], bond_table.shape[1]), np.int64)
        real = frame_bond >= 0
        bond_vals[real] = bond_table[frame_bond[real]]
        nmask = (np.arange(slots)[None, :] < frame_k[:, None]).astype(np.float32)
        emask = real.astype(np.float32)
        size_oh = np.zeros((n_frames, slots), np.float32)
        size_oh[np.arange(n_frames), frame_k - 1] = 1.0
        self.register_buffer("frame_atom_vals", torch.as_tensor(atom_vals).long(), persistent=False)
        self.register_buffer("frame_bond_vals", torch.as_tensor(bond_vals).long(), persistent=False)
        self.register_buffer("frame_nmask", torch.as_tensor(nmask), persistent=False)
        self.register_buffer("frame_emask", torch.as_tensor(emask), persistent=False)
        self.register_buffer("frame_size_oh", torch.as_tensor(size_oh), persistent=False)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # -- encoders -----------------------------------------------------------
    def frame_embeddings(self) -> torch.Tensor:
        return self.phi(self.frame_atom_vals, self.frame_bond_vals, self.frame_nmask,
                        self.frame_emask, self.frame_size_oh)

    def cut_embeddings(self) -> torch.Tensor:
        vals = self.bond_table
        emb = torch.cat([e(vals[:, i]) for i, e in enumerate(self.phi.bond_emb)], dim=-1)
        return self.cut_proj(emb)

    def _cut_tokens(self, bond_rows: torch.Tensor, cut: torch.Tensor) -> torch.Tensor:
        out = self.null_cut[None, :].repeat(bond_rows.numel(), 1)
        real = bond_rows >= 0
        if bool(real.any()):
            out = out.index_copy(0, torch.nonzero(real).squeeze(-1), cut[bond_rows[real].clamp(min=0)])
        return out

    def _expand(self, ids: torch.Tensor, ptr_name: str) -> tuple[torch.Tensor, torch.Tensor]:
        ptr = getattr(self, ptr_name)
        counts = ptr[ids + 1] - ptr[ids]
        total = int(counts.sum().item())
        starts = torch.cumsum(counts, 0) - counts
        local = torch.arange(total, device=ids.device) - torch.repeat_interleave(starts, counts)
        gidx = torch.repeat_interleave(ptr[ids], counts) + local
        owner = torch.repeat_interleave(torch.arange(ids.numel(), device=ids.device), counts)
        return gidx, owner

    def token_embeddings(self, frag_ids: torch.Tensor, frames: torch.Tensor):
        gidx, owner = self._expand(frag_ids, "frag_ptr")
        frame = self.frag_frame[gidx]
        bucket = self.frag_bucket[gidx]
        weight = self.frag_weight[gidx].to(frames.dtype)
        numeric = torch.stack([weight, torch.log1p(weight)], dim=-1)
        x = (self.token_proj(frames[frame]) + self.bucket_emb(bucket) + self.mult_proj(numeric))
        return self.token_ln(x), owner, weight, bucket

    @staticmethod
    def _weighted_mean(x: torch.Tensor, owner: torch.Tensor, weight: torch.Tensor,
                       n: int) -> torch.Tensor:
        num = x.new_zeros((n, x.shape[-1])).index_add_(0, owner, x * weight[:, None])
        den = x.new_zeros(n).index_add_(0, owner, weight)
        return num / den.clamp(min=1e-6)[:, None]

    # -- cut response -------------------------------------------------------
    def _chunk_bounds(self, n_entries: torch.Tensor) -> list[tuple[int, int]]:
        counts = [int(v) for v in n_entries.tolist()]
        cum = np.cumsum(counts) if counts else np.zeros(0, np.int64)
        out: list[tuple[int, int]] = []
        start = 0
        n = len(counts)
        while start < n:
            base = int(cum[start - 1]) if start > 0 else 0
            end = int(np.searchsorted(cum, base + self.entry_chunk, side="right"))
            end = max(end, start + 1)
            out.append((start, end))
            start = end
        return out

    def cut_scores(self, s_ids: torch.Tensor, c_ids: torch.Tensor,
                   bond_rows: torch.Tensor, frames: torch.Tensor | None = None,
                   diag: dict[str, Any] | None = None) -> torch.Tensor:
        """Cut response for an arbitrary list of (S, C, cut bond) parses."""
        n = int(s_ids.numel())
        if n == 0:
            return s_ids.new_zeros(0).to(self.beta.dtype)
        frames = self.frame_embeddings() if frames is None else frames
        hid = self.hidden
        dev = s_ids.device

        xs, s_owner, s_w, s_b = self.token_embeddings(s_ids, frames)
        s_cnt = torch.bincount(s_owner, minlength=n)
        if int(s_cnt.min()) == 0:
            raise ValueError("a fragment with no structural-object token reached the model")
        s_ptr = torch.cat([s_cnt.new_zeros(1), torch.cumsum(s_cnt, 0)])
        pool_s = self._weighted_mean(xs, s_owner, s_w, n)

        has_c = c_ids >= 0
        if bool(has_c.any()):
            uniq_c, inv_c = torch.unique(c_ids[has_c], sorted=True, return_inverse=True)
            nu = int(uniq_c.numel())
            xc_tok, c_owner, c_w, c_b = self.token_embeddings(uniq_c, frames)
            pool_u = self._weighted_mean(xc_tok, c_owner, c_w, nu)
            grp = torch.full((n,), nu, dtype=torch.long, device=dev)
            grp = grp.index_copy(0, torch.nonzero(has_c).squeeze(-1), inv_c)
            pool_c = self.null_ctx[None, :].repeat(n, 1)
            pool_c = pool_c.index_copy(0, torch.nonzero(has_c).squeeze(-1), pool_u[inv_c])
        else:
            nu = 0
            xc_tok = xs.new_zeros((0, hid))
            c_owner = torch.zeros(0, dtype=torch.long, device=dev)
            c_w = xs.new_zeros(0)
            c_b = torch.zeros(0, dtype=torch.long, device=dev)
            grp = torch.zeros(n, dtype=torch.long, device=dev)
            pool_c = self.null_ctx[None, :].repeat(n, 1)

        xc = torch.cat([xc_tok, self.null_ctx[None, :]], dim=0)
        kw = torch.cat([c_w, c_w.new_ones(1)], dim=0)
        kb = torch.cat([c_b, c_b.new_full((1,), C.NULL_BUCKET)], dim=0)
        g_cnt = torch.bincount(c_owner, minlength=nu)
        g_cnt = torch.cat([g_cnt, g_cnt.new_ones(1)], dim=0)
        g_ptr = torch.cat([g_cnt.new_zeros(1), torch.cumsum(g_cnt, 0)], dim=0)
        c_cnt = g_cnt[grp]
        c_ptr = g_ptr[grp]

        cut = self.cut_embeddings()
        e = self._cut_tokens(bond_rows, cut)
        f = (self.beta + self.add_s(pool_s).squeeze(-1) + self.add_c(pool_c).squeeze(-1)
             + self.bond_bias(e).squeeze(-1))
        if not self.interaction:
            return f

        # ---- per-parse cross attention (ragged, chunked over entries) ------
        tot_q = int(s_cnt.sum())
        pid_q = torch.repeat_interleave(torch.arange(n, device=dev), s_cnt)
        off = torch.cumsum(s_cnt, 0) - s_cnt
        q_to_tok = (torch.repeat_interleave(s_ptr[:-1], s_cnt)
                    + (torch.arange(tot_q, device=dev) - torch.repeat_interleave(off, s_cnt)))
        xq = xs[q_to_tok]
        wq = s_w[q_to_tok]
        bq = s_b[q_to_tok]
        cut_q = e[pid_q]
        q_all = self.q_proj(xq).view(tot_q, self.heads, self.dh)
        k_all = self.k_proj(xc).view(-1, self.heads, self.dh)
        v_all = self.v_proj(xc).view(-1, self.heads, self.dh)

        n_entries = s_cnt * c_cnt
        scale = 1.0 / math.sqrt(float(self.dh))
        out_parts: list[torch.Tensor] = []
        ent_sum = 0.0
        ent_n = 0
        bias_abs = 0.0
        bias_sq = 0.0
        bias_n = 0
        for p0, p1 in self._chunk_bounds(n_entries):
            q0, q1 = int(s_ptr[p0]), int(s_ptr[p1])
            nq = q1 - q0
            cc = c_cnt[p0:p1]
            pid_local = torch.repeat_interleave(
                torch.arange(p1 - p0, device=dev), s_cnt[p0:p1])
            key_counts = cc[pid_local]
            n_ent = int(key_counts.sum())
            qs = torch.repeat_interleave(torch.arange(nq, device=dev), key_counts)
            before = torch.cumsum(key_counts, 0) - key_counts
            keys = (torch.repeat_interleave(c_ptr[p0:p1][pid_local], key_counts)
                    + (torch.arange(n_ent, device=dev)
                       - torch.repeat_interleave(before, key_counts)))
            # every query token is its own softmax segment: j ranges over the
            # C tokens of *this parse* only
            seg = qs

            score = (q_all[q0:q1][qs] * k_all[keys]).sum(-1) * scale
            score = score + torch.log(kw[keys])[:, None]
            if self.use_relation:
                qb = bq[q0:q1][qs]
                kb_ = kb[keys]
                z = (self.rel_ds(qb) + self.rel_dc(kb_)
                     + self.rel_dt(self.dtot[qb * C.N_BUCKETS + kb_])
                     + self.rel_cut(cut_q[q0:q1][qs]))
                bias = self.rel_mlp(z)
                score = score + bias
                if diag is not None:
                    bias_abs += float(bias.detach().abs().sum())
                    bias_sq += float((bias.detach() ** 2).sum())
                    bias_n += int(bias.numel())
            mx = score.new_full((nq, self.heads), -float("inf")).index_reduce_(
                0, seg, score.detach(), "amax", include_self=False)
            ex = torch.exp(score - mx[seg])
            den = ex.new_zeros((nq, self.heads)).index_add_(0, seg, ex)
            alpha = ex / den[seg]
            if diag is not None:
                p = alpha.detach().clamp(min=0.0)
                ent = -(p * torch.log(p.clamp(min=1e-9))).sum()
                ent_sum += float(ent)
                ent_n += int(p.shape[0] * p.shape[1])
            alpha = self.attn_drop(alpha)
            m = alpha.new_zeros((nq, self.heads, self.dh)).index_add_(
                0, qs, alpha[..., None] * v_all[keys]).reshape(nq, hid)
            if self._capture is not None:
                self._capture.setdefault("mx", []).append(mx.detach())
                self._capture.setdefault("den", []).append(den.detach())
                self._capture.setdefault("seg", []).append(seg.detach())
                self._capture.setdefault("alpha", []).append(alpha.detach())
                self._capture.setdefault("keys", []).append(keys.detach())
                self._capture.setdefault("qs", []).append(qs.detach())
                # chunk-local query indices: record the offset so a caller can build
                # global (per-call) query ids
                self._capture.setdefault("q0", []).append(q0)
                self._capture.setdefault("score", []).append(score.detach())
            out_parts.append(m)
        m_all = torch.cat(out_parts, dim=0)
        if self._capture is not None:
            self._capture["m_all"] = m_all.detach()
        r = self.r_mlp(torch.cat([xq, m_all, xq * m_all, xq - m_all, cut_q], dim=-1))
        r_pool = self._weighted_mean(r, pid_q, wq, n)
        f = f + self.out(r_pool).squeeze(-1)
        if diag is not None:
            mean_entropy = ent_sum / max(1, ent_n)
            diag.setdefault("attention_entropy", []).append(mean_entropy)
            diag.setdefault("effective_tokens", []).append(math.exp(mean_entropy))
            diag.setdefault("attention_entries", []).append(float(n_entries.sum()))
            diag.setdefault("r_pool_norm", []).append(
                float(r_pool.detach().norm(dim=-1).mean()))
            diag.setdefault("r_pool_absmean", []).append(
                float(r_pool.detach().abs().mean()))
            diag.setdefault("m_norm", []).append(float(m_all.detach().norm(dim=-1).mean()))
            if self.use_relation:
                diag.setdefault("relation_bias_absmean", []).append(bias_abs / max(1, bias_n))
                diag.setdefault("relation_bias_rms", []).append(
                    math.sqrt(bias_sq / max(1, bias_n)))
        return f

    def interaction_contribution(self, s_ids: torch.Tensor, c_ids: torch.Tensor,
                                 bond_rows: torch.Tensor,
                                 frames: torch.Tensor | None = None) -> torch.Tensor:
        """Scalar contribution of the pooled interaction branch (diagnostic)."""
        frames = self.frame_embeddings() if frames is None else frames
        full = self.cut_scores(s_ids, c_ids, bond_rows, frames)
        additive_only = self._additive_only(s_ids, c_ids, bond_rows, frames)
        return full - additive_only

    def _additive_only(self, s_ids: torch.Tensor, c_ids: torch.Tensor,
                       bond_rows: torch.Tensor, frames: torch.Tensor) -> torch.Tensor:
        saved = self.interaction
        try:
            self.interaction = False
            out = self.cut_scores(s_ids, c_ids, bond_rows, frames)
        finally:
            self.interaction = saved
        return out

    # -- molecule / pair entry points --------------------------------------
    def _parse_index(self, rows: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        ptr = self.mol_parse_ptr
        counts = ptr[rows + 1] - ptr[rows]
        total = int(counts.sum().item())
        starts = torch.cumsum(counts, 0) - counts
        pidx = (torch.repeat_interleave(ptr[rows], counts)
                + torch.arange(total, device=rows.device)
                - torch.repeat_interleave(starts, counts))
        return pidx, counts

    def molecule_cuts(self, rows: torch.Tensor, frames: torch.Tensor | None = None,
                      diag: dict[str, Any] | None = None) -> dict[str, torch.Tensor]:
        pidx, counts = self._parse_index(rows)
        f = self.cut_scores(self.parse_S[pidx], self.parse_C[pidx],
                            self.parse_cut_bond_row[pidx], frames, diag=diag)
        owner = torch.repeat_interleave(torch.arange(rows.numel(), device=rows.device), counts)
        f_bar = f.new_zeros(rows.numel()).index_add_(0, owner, f)
        f_bar = f_bar / counts.to(f.dtype).clamp(min=1.0)
        return {"f_cut": f, "f_bar": f_bar, "owner": owner, "counts": counts}

    def molecule_forward(self, rows: torch.Tensor,
                         diag: dict[str, Any] | None = None) -> dict[str, Any]:
        frames = self.frame_embeddings()
        return self.molecule_cuts(rows, frames, diag=diag)

    def pair_cut_delta(self, pairs: Mapping[str, torch.Tensor],
                       frames: torch.Tensor | None = None) -> torch.Tensor:
        ctx = pairs["C"]
        sa = pairs["Sa"]
        sb = pairs["Sb"]
        bond = pairs["bond"]
        n = int(ctx.numel())
        s_ids = torch.cat([sb, sa], dim=0)
        c_ids = torch.cat([ctx, ctx], dim=0)
        b = torch.cat([bond, bond], dim=0)
        f = self.cut_scores(s_ids, c_ids, b, frames)
        return f[:n] - f[n:]

    def parameter_ledger(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for name, p in self.named_parameters():
            top = name.split(".")[0]
            out[top] = out.get(top, 0) + int(p.numel())
        out["total"] = int(sum(p.numel() for p in self.parameters()))
        out["trainable"] = int(sum(p.numel() for p in self.parameters() if p.requires_grad))
        return out

    def module_groups(self) -> dict[str, list[str]]:
        """Parameter groups for the interaction / encoder gradient diagnostic."""
        tokens = ["phi", "token_proj", "bucket_emb", "mult_proj", "token_ln"]
        interaction = ["q_proj", "k_proj", "v_proj", "r_mlp", "out",
                       "rel_ds", "rel_dc", "rel_dt", "rel_cut", "rel_mlp"]
        additive = ["add_s", "add_c", "bond_bias", "cut_proj", "beta", "null_cut", "null_ctx"]
        return {"token_encoder": tokens, "interaction": interaction, "additive": additive}

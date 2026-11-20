"""Word content encoders, the shared local-sentence computation and controls.

Three content encoders (ID / CODE / WGINE) all emit 64 dims and read *only* the
word type.  Every instance channel, incidence message and sentence term is
identical across them.  ``ERASED`` changes exactly one tensor: the relative-site
descriptor ``R`` is replaced by its label-compatible site-pair mean.

The local-sentence prediction is

    F_std(G) = b + sum_i (n_i / N_atoms) * (u_i + v_i + w_i)

with ``u_i`` self, ``v_i`` mean incidence neighbour term and ``w_i`` mean
incidence-pair term.  The neighbour and pair final linear layers are
zero-initialised, so FULL and ERASED start from identical predictions.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ws_words as W

HIDDEN = 64
ENC_DIM = 64
DROPOUT = 0.10
ENC_ATOM_DIM = 4
ENC_BOND_DIM = 4
WG_WIDTH = 32
CODE_DIM = 6 * 3 * 4 + 6 * 6 * 4 + 6 + 7  # 229
R_DIM = W.R_DIM
PAIR_FIELD_DIM = W.PAIR_FIELD_DIM


# ---------------------------------------------------------------------------
# template tensors
# ---------------------------------------------------------------------------


def template_arrays(table: W.TemplateTable, dims: Mapping[str, Sequence[int]], n_bond_codes: int):
    n_t = len(table)
    lab = np.zeros((n_t, 6, 3), dtype=np.int64)
    mask = np.zeros((n_t, 6), dtype=np.float32)
    bond = np.full((n_t, 6, 6), n_bond_codes, dtype=np.int64)
    n_atoms = np.zeros(n_t, dtype=np.int64)
    R_table = np.zeros((n_t, 6, 6, R_DIM), dtype=np.float32)
    for tmpl in table.templates:
        n = tmpl.n
        n_atoms[tmpl.tid] = n
        lab[tmpl.tid, :n] = tmpl.labels
        mask[tmpl.tid, :n] = 1.0
        for i in range(6):
            for j in range(6):
                if i < n and j < n:
                    bond[tmpl.tid, i, j] = n_bond_codes
                    R_table[tmpl.tid, i, j] = W.r_descriptor(tmpl, i, j)
        for a, b, t in zip(tmpl.bu.tolist(), tmpl.bv.tolist(), tmpl.btype.tolist()):
            bond[tmpl.tid, a, b] = bond[tmpl.tid, b, a] = int(t)
    # compact node/edge lists for the WGINE template encoder
    node_tmpl, node_lab = [], []
    edge_u, edge_v, edge_t = [], [], []
    offset = 0
    node_ptr = [0]
    for tmpl in table.templates:
        node_tmpl.extend([tmpl.tid] * tmpl.n)
        node_lab.extend(tmpl.labels.tolist())
        for a, b, t in zip(tmpl.bu.tolist(), tmpl.bv.tolist(), tmpl.btype.tolist()):
            edge_u.extend([offset + a, offset + b])
            edge_v.extend([offset + b, offset + a])
            edge_t.extend([int(t), int(t)])
        offset += tmpl.n
        node_ptr.append(offset)
    multi_ids = np.asarray(table.multi, dtype=np.int64)
    multi_row = np.zeros(n_t, dtype=np.int64)
    multi_row[multi_ids] = np.arange(len(multi_ids))
    return {
        "tmpl_labels": lab,
        "tmpl_mask": mask,
        "tmpl_bond": bond,
        "tmpl_n_atoms": n_atoms,
        "R_pair_table": R_table,
        "node_tmpl": np.asarray(node_tmpl, dtype=np.int64),
        "node_lab": np.asarray(node_lab, dtype=np.int64).reshape(-1, 3),
        "node_ptr": np.asarray(node_ptr, dtype=np.int64),
        "edge_index": np.asarray([edge_u, edge_v], dtype=np.int64).reshape(2, -1),
        "edge_type": np.asarray(edge_t, dtype=np.int64),
        "multi_ids": multi_ids,
        "multi_row": multi_row,
        "is_multi": np.isin(np.arange(n_t), multi_ids),
        "n_templates": n_t,
        "n_multi": int(len(multi_ids)),
    }


# ---------------------------------------------------------------------------
# content encoders
# ---------------------------------------------------------------------------


class IDEncoder(nn.Module):
    def __init__(self, t: Mapping[str, np.ndarray], dims: Mapping[str, Sequence[int]], hidden: int):
        super().__init__()
        n_multi = max(1, int(t["n_multi"]))
        self.register_buffer("multi_row", torch.as_tensor(t["multi_row"], dtype=torch.long))
        self.register_buffer("is_multi", torch.as_tensor(t["is_multi"]))
        self.register_buffer("multi_ids", torch.as_tensor(t["multi_ids"], dtype=torch.long))
        self.emb = nn.Embedding(n_multi, ENC_DIM)
        self.mono_atom = nn.ModuleList([
            nn.Embedding(int(dims["atom"][c]), ENC_ATOM_DIM) for c in W.WORD_ATOM_COLUMNS
        ])
        self.mlp = nn.Sequential(nn.Linear(ENC_DIM, hidden), nn.SiLU(),
                                 nn.Linear(hidden, ENC_DIM), nn.SiLU(), nn.LayerNorm(ENC_DIM))
        self.mono_mlp = nn.Sequential(nn.Linear(3 * ENC_ATOM_DIM, hidden), nn.SiLU(),
                                      nn.Linear(hidden, ENC_DIM), nn.SiLU(), nn.LayerNorm(ENC_DIM))

    def forward(self, t: Mapping[str, torch.Tensor]) -> torch.Tensor:
        lab = t["tmpl_labels"]                        # (T, 6, 3)
        mono = torch.cat([emb(lab[:, 0, j]) for j, emb in enumerate(self.mono_atom)], dim=-1)
        e_mono = self.mono_mlp(mono)
        e_multi = self.mlp(self.emb(self.multi_row))
        return torch.where(self.is_multi.unsqueeze(-1), e_multi, e_mono)


class CODEEncoder(nn.Module):
    def __init__(self, t: Mapping[str, np.ndarray], dims: Mapping[str, Sequence[int]], hidden: int,
                 n_bond_codes: int):
        super().__init__()
        self.n_bond_codes = int(n_bond_codes)
        self.atom_emb = nn.ModuleList([
            nn.Embedding(int(dims["atom"][c]), ENC_ATOM_DIM) for c in W.WORD_ATOM_COLUMNS
        ])
        # `no-edge` is a fixed zero embedding; padding is distinguished by the mask
        self.bond_emb = nn.Embedding(int(n_bond_codes) + 1, ENC_BOND_DIM)
        self.mlp = nn.Sequential(nn.Linear(CODE_DIM, hidden), nn.SiLU(),
                                 nn.Linear(hidden, ENC_DIM), nn.SiLU(), nn.LayerNorm(ENC_DIM))

    def code_tensor(self, t: Mapping[str, torch.Tensor]) -> torch.Tensor:
        lab = t["tmpl_labels"]
        atom = torch.cat([emb(lab[:, :, j]) for j, emb in enumerate(self.atom_emb)], dim=-1)
        weight = torch.cat([self.bond_emb.weight,
                            torch.zeros(1, ENC_BOND_DIM, device=lab.device,
                                        dtype=self.bond_emb.weight.dtype)])
        bond = F.embedding(t["tmpl_bond"], weight)
        count = F.one_hot(torch.clamp(t["tmpl_n_atoms"] - 1, min=0, max=6), num_classes=7).to(atom.dtype)
        return torch.cat([atom.reshape(atom.shape[0], -1), bond.reshape(bond.shape[0], -1),
                          t["tmpl_mask"], count], dim=-1)

    def forward(self, t: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.mlp(self.code_tensor(t))


class WGINEEncoder(nn.Module):
    def __init__(self, t: Mapping[str, np.ndarray], dims: Mapping[str, Sequence[int]], hidden: int):
        super().__init__()
        self.atom_emb = nn.ModuleList([
            nn.Embedding(int(dims["atom"][c]), WG_WIDTH) for c in W.WORD_ATOM_COLUMNS
        ])
        self.bond_emb = nn.Embedding(int(dims["bond"][0]), WG_WIDTH)
        self.register_buffer("node_tmpl", torch.as_tensor(t["node_tmpl"], dtype=torch.long))
        self.register_buffer("node_lab", torch.as_tensor(t["node_lab"], dtype=torch.long))
        self.register_buffer("edge_index", torch.as_tensor(t["edge_index"], dtype=torch.long))
        self.register_buffer("edge_type", torch.as_tensor(t["edge_type"], dtype=torch.long))
        self.eps = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(2)])
        self.lin1 = nn.ModuleList([nn.Linear(WG_WIDTH, 2 * WG_WIDTH) for _ in range(2)])
        self.lin2 = nn.ModuleList([nn.Linear(2 * WG_WIDTH, WG_WIDTH) for _ in range(2)])
        self.norm = nn.ModuleList([nn.LayerNorm(WG_WIDTH) for _ in range(2)])
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(nn.Linear(2 * WG_WIDTH + 1, hidden), nn.SiLU(),
                                  nn.Linear(hidden, ENC_DIM), nn.SiLU(), nn.LayerNorm(ENC_DIM))

    def forward(self, t: Mapping[str, torch.Tensor]) -> torch.Tensor:
        x = sum(emb(self.node_lab[:, j]) for j, emb in enumerate(self.atom_emb))
        e = self.bond_emb(self.edge_type)
        src, dst = self.edge_index[0], self.edge_index[1]
        for layer in range(2):
            residual = x
            msg = F.relu(x[src] + e)
            agg = torch.zeros_like(x)
            agg.index_add_(0, dst, msg)
            h = (1.0 + self.eps[layer]) * x + agg
            h = self.lin2[layer](F.silu(self.lin1[layer](h)))
            x = self.norm[layer](residual + self.drop(h))
        n = t["tmpl_n_atoms"].to(x.dtype)
        summed = torch.zeros(t["n_templates"], x.shape[1], device=x.device, dtype=x.dtype)
        summed.index_add_(0, self.node_tmpl, x)
        meaned = summed / n.clamp(min=1.0).unsqueeze(-1)
        pooled = torch.cat([summed, meaned, torch.log1p(n).unsqueeze(-1)], dim=-1)
        return self.head(pooled)


# ---------------------------------------------------------------------------
# sentence model
# ---------------------------------------------------------------------------


def _mlp(inp: int, hidden: int, out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.SiLU(), nn.Linear(hidden, out))


@dataclass
class ModelSpec:
    encoder: str = "code"          # id | code | wgine
    mode: str = "full"             # full | erased | bag
    extra: str = "none"            # none | attrpair | rootmean
    para: str = "none"             # none | rel | cap
    hidden: int = HIDDEN


class WordSentenceModel(nn.Module):
    def __init__(self, spec: ModelSpec, table_arrays: Mapping[str, np.ndarray],
                 dims: Mapping[str, Sequence[int]], n_bond_codes: int, degree_buckets: int = 17):
        super().__init__()
        self.spec = spec
        hidden = spec.hidden
        self.hidden = hidden
        self.raw_atom = nn.ModuleList([nn.Embedding(int(d), ENC_ATOM_DIM) for d in dims["atom"]])
        self.raw_bond = nn.ModuleList([nn.Embedding(int(d), ENC_BOND_DIM) for d in dims["bond"]])
        if spec.encoder == "id":
            self.encoder = IDEncoder(table_arrays, dims, hidden)
        elif spec.encoder == "code":
            self.encoder = CODEEncoder(table_arrays, dims, hidden, n_bond_codes)
        elif spec.encoder == "wgine":
            self.encoder = WGINEEncoder(table_arrays, dims, hidden)
        else:
            raise ValueError(f"unknown encoder {spec.encoder!r}")
        self.attr_mlp = nn.Sequential(nn.Linear(98, hidden), nn.SiLU(),
                                      nn.Linear(hidden, ENC_DIM), nn.LayerNorm(ENC_DIM))
        self.msg = _mlp(2 * ENC_DIM + 2 * 9 * ENC_ATOM_DIM + 3 * ENC_BOND_DIM, hidden, ENC_DIM)
        self.head_self = _mlp(2 * ENC_DIM + 2, hidden, 1)
        self.nb_lin1 = nn.Linear(3 * ENC_DIM + 2, hidden)
        self.nb_lin2 = nn.Linear(hidden, 1)
        self.pair_in = 4 * ENC_DIM + R_DIM + PAIR_FIELD_DIM + 2
        self.pair_lin1 = nn.Linear(self.pair_in, hidden)
        self.pair_lin2 = nn.Linear(hidden, 1)
        self.bias = nn.Parameter(torch.zeros(1))
        if spec.mode == "bag":
            self.bag_mlp = _mlp(2 * ENC_DIM, hidden, ENC_DIM)
            self.bag_head = _mlp(2 * ENC_DIM + 2, hidden, 1)
        if spec.extra == "attrpair":
            self.attrpair_mlp = _mlp(2 * 9 * ENC_ATOM_DIM + R_DIM, hidden, ENC_DIM)
            self.attrpair_proj = nn.Linear(ENC_DIM, ENC_DIM)
            nn.init.zeros_(self.attrpair_proj.weight)
            nn.init.zeros_(self.attrpair_proj.bias)
        if spec.extra == "rootmean":
            self.root_head = _mlp(CODE_DIM + 6, hidden, ENC_DIM)
        if spec.para != "none":
            self.para_h = _mlp(3 * ENC_DIM, hidden, ENC_DIM)
            self.para_u = nn.Linear(ENC_DIM, 16, bias=False)
            self.para_v = nn.Linear(ENC_DIM, 16, bias=False)
            self.para_w = nn.Parameter(torch.zeros(16))
        nn.init.zeros_(self.nb_lin2.weight)
        nn.init.zeros_(self.nb_lin2.bias)
        nn.init.zeros_(self.pair_lin2.weight)
        nn.init.zeros_(self.pair_lin2.bias)
        self.t = {k: torch.as_tensor(v) for k, v in table_arrays.items()
                  if isinstance(v, np.ndarray)}
        self.n_bond_codes = int(n_bond_codes)

    # -- content ---------------------------------------------------------
    def _templates(self, device) -> dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in self.t.items()}

    def content(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        t = self._templates(batch["word_tmpl"].device)
        if self.spec.extra == "rootmean":
            return self._rootmean(t)
        return self.encoder(t)

    def _rootmean(self, t: Mapping[str, torch.Tensor]) -> torch.Tensor:
        code = self.encoder.code_tensor(t)                  # (T, 229)
        T = code.shape[0]
        reps = code.unsqueeze(1).expand(T, 6, code.shape[1])
        roots = torch.eye(6, device=code.device, dtype=code.dtype).unsqueeze(0).expand(T, 6, 6)
        rooted = torch.cat([reps, roots], dim=-1)           # (T, 6, 235)
        e = self.root_head(rooted)                          # (T, 6, 64)
        mask = t["tmpl_mask"].unsqueeze(-1)
        return (e * mask).sum(dim=1) / mask.sum(dim=1).clamp(min=1.0)

    # -- instance channels ----------------------------------------------
    def atom36(self, atom_feat: torch.Tensor) -> torch.Tensor:
        return torch.cat([emb(atom_feat[:, j]) for j, emb in enumerate(self.raw_atom)], dim=-1)

    def bond12(self, bond_feat: torch.Tensor) -> torch.Tensor:
        return torch.cat([emb(bond_feat[:, j]) for j, emb in enumerate(self.raw_bond)], dim=-1)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        dev = batch["word_tmpl"].device
        a36 = self.atom36(batch["atom_feat"])
        b12 = self.bond12(batch["bond_feat"])
        e_all = self.content(batch)
        e = e_all[batch["word_tmpl"]]
        n_words = e.shape[0]
        # instance attribute channel
        wa = a36[batch["word_atoms"]]                       # (n_w_atoms, 36)
        wa_sum = torch.zeros(n_words, wa.shape[1], device=dev, dtype=wa.dtype)
        wa_sum.index_add_(0, batch["word_of_atom"], wa)
        n_word_atoms = batch["word_natoms"].to(wa.dtype)
        wa_mean = wa_sum / n_word_atoms.clamp(min=1.0).unsqueeze(-1)
        wb = b12[batch["word_bonds"]]
        wb_sum = torch.zeros(n_words, wb.shape[1], device=dev, dtype=wb.dtype)
        wb_sum.index_add_(0, batch["word_of_bond"], wb)
        n_bonds = torch.bincount(batch["word_of_bond"], minlength=n_words).to(wb.dtype)
        wb_mean = wb_sum / n_bonds.clamp(min=1.0).unsqueeze(-1)
        n_inc = torch.bincount(batch["word_of_inc"], minlength=n_words).to(wa.dtype)
        a = self.attr_mlp(torch.cat([wa_sum, wa_mean, wb_sum, wb_mean,
                                     torch.log1p(n_word_atoms).unsqueeze(-1),
                                     torch.log1p(n_inc).unsqueeze(-1)], dim=-1))
        if self.spec.extra == "attrpair":
            a = a + self._attrpair(batch, a36, self._templates(dev))
        # incidence message
        inc_own = batch["inc_own_atom"]
        msg_in = torch.cat([e[batch["inc_other"]], a[batch["inc_other"]],
                            a36[inc_own], a36[batch["inc_other_atom"]],
                            b12[batch["inc_bond"]]], dim=-1)
        m = self.msg(msg_in)
        g = batch["g_size"][batch["graph_of_word"]]
        u = self.head_self(torch.cat([e, a, g], dim=-1)).view(-1)
        inc_word = batch["word_of_inc"]
        nb_in = torch.cat([e[inc_word], a[inc_word], m, g[inc_word]], dim=-1)
        nb_raw = self.nb_lin2(F.silu(self.nb_lin1(nb_in))).view(-1)
        v = torch.zeros(n_words, device=dev, dtype=e.dtype)
        v.index_add_(0, batch["word_of_inc"], nb_raw)
        v = v / n_inc.clamp(min=1.0)
        # incidence pairs
        R = batch["pair_ER"] if self.spec.mode == "erased" else batch["pair_R"]
        ma, mb = m[batch["pair_a"]], m[batch["pair_b"]]
        pair_in = torch.cat([e[batch["word_of_pair"]], a[batch["word_of_pair"]],
                             ma + mb, ma * mb, R, batch["pair_fields"],
                             g[batch["word_of_pair"]]], dim=-1)
        pair_hidden = F.silu(self.pair_lin1(pair_in))
        pair_raw = self.pair_lin2(pair_hidden).view(-1)
        n_pair = torch.bincount(batch["word_of_pair"], minlength=n_words).to(e.dtype)
        w = torch.zeros(n_words, device=dev, dtype=e.dtype)
        w.index_add_(0, batch["word_of_pair"], pair_raw)
        w = w / n_pair.clamp(min=1.0)
        r = torch.zeros(n_words, device=dev, dtype=e.dtype)
        if self.spec.para != "none":
            pair_h = torch.zeros(n_words, pair_hidden.shape[1], device=dev, dtype=e.dtype)
            pair_h.index_add_(0, batch["word_of_pair"], pair_hidden)
            pair_h = pair_h / n_pair.clamp(min=1.0).unsqueeze(-1)
            mean_m = torch.zeros(n_words, m.shape[1], device=dev, dtype=e.dtype)
            mean_m.index_add_(0, batch["word_of_inc"], m)
            mean_m = mean_m / n_inc.clamp(min=1.0).unsqueeze(-1)
            h = self.para_h(torch.cat([e, a, mean_m], dim=-1))
            if self.spec.para == "cap":
                ctx = h
            else:
                ctx_sum = torch.zeros_like(h)
                ctx_sum.index_add_(0, batch["hop_dst"], h[batch["hop_src"]])
                ctx_cnt = torch.bincount(batch["hop_dst"], minlength=n_words).to(h.dtype)
                ctx = ctx_sum / ctx_cnt.clamp(min=1.0).unsqueeze(-1)
            gate = torch.tanh((self.para_u(h) + self.para_v(ctx)) / np.sqrt(2.0))
            r = (self.para_w.view(1, -1) * gate).sum(dim=-1) / np.sqrt(16.0)
        if self.spec.mode == "bag":
            z = self.bag_mlp(torch.cat([e, a], dim=-1))
            mass = n_word_atoms / batch["n_atoms_graph"][batch["graph_of_word"]].to(e.dtype)
            z_mass = torch.zeros(batch["n_graphs"], z.shape[1], device=dev, dtype=z.dtype)
            z_mass.index_add_(0, batch["graph_of_word"], z * mass.unsqueeze(-1))
            z_sum = torch.zeros_like(z_mass)
            z_sum.index_add_(0, batch["graph_of_word"], z)
            gl = batch["g_size"]
            pred = self.bag_head(torch.cat([z_mass, z_sum, gl], dim=-1)).view(-1)
        else:
            mass = n_word_atoms / batch["n_atoms_graph"][batch["graph_of_word"]].to(e.dtype)
            contrib = (u + v + w + r) * mass
            pred = torch.zeros(batch["n_graphs"], device=dev, dtype=e.dtype)
            pred.index_add_(0, batch["graph_of_word"], contrib)
            pred = pred + self.bias
        return {"pred": pred, "u": u, "v": v, "w": w, "r": r, "e": e, "a": a, "m": m,
                "pair_hidden": pair_hidden if self.spec.para != "none" else pair_raw,
                "n_inc": n_inc, "n_word_atoms": n_word_atoms}

    def _attrpair(self, batch: Mapping[str, torch.Tensor], a36: torch.Tensor,
                  t: Mapping[str, torch.Tensor]) -> torch.Tensor:
        dev = a36.device
        n_words = batch["word_tmpl"].shape[0]
        if batch["ap_n"] == 0:
            return torch.zeros(n_words, self.hidden, device=dev)
        Rtab = t["R_pair_table"]
        R = Rtab[batch["word_tmpl"][batch["ap_word"]], batch["ap_i"], batch["ap_j"]]
        za = a36[batch["word_atoms"][batch["word_ptr_atom"][batch["ap_word"]] + batch["ap_i"]]]
        zb = a36[batch["word_atoms"][batch["word_ptr_atom"][batch["ap_word"]] + batch["ap_j"]]]
        J = F.silu(self.attrpair_mlp(torch.cat([za + zb, za * zb, R], dim=-1)))
        agg = torch.zeros(n_words, J.shape[1], device=dev, dtype=J.dtype)
        agg.index_add_(0, batch["ap_word"], J)
        cnt = torch.bincount(batch["ap_word"], minlength=n_words).to(J.dtype).clamp(min=1.0)
        return self.attrpair_proj(agg / cnt.unsqueeze(-1))


# ---------------------------------------------------------------------------
# GINE performance reference (same 5-layer/hidden-128 reference as sq_gine)
# ---------------------------------------------------------------------------


class GINERefModel(nn.Module):
    def __init__(self, n_categories: Mapping[str, int], hidden: int = 128, layers: int = 5):
        super().__init__()
        from torch_geometric.nn import GINEConv

        h = hidden
        self.atom_embeddings = nn.ModuleList([nn.Embedding(int(n_categories[f"atom_{j}"]), h)
                                              for j in range(9)])
        self.degree_embedding = nn.Embedding(int(n_categories["degree"]), h)
        self.ring_embedding = nn.Embedding(int(n_categories["ring"]), h)
        self.bond_embeddings = nn.ModuleList([nn.Embedding(int(n_categories[f"bond_{j}"]), h)
                                              for j in range(3)])
        self.convs = nn.ModuleList([
            GINEConv(nn.Sequential(nn.Linear(h, 2 * h), nn.SiLU(), nn.Linear(2 * h, h)), train_eps=True)
            for _ in range(layers)])
        self.norms = nn.ModuleList([nn.BatchNorm1d(h) for _ in range(layers)])
        self.drop = nn.Dropout(DROPOUT)
        self.head = nn.Sequential(nn.Linear(2 * h + 1, h), nn.SiLU(), nn.Dropout(DROPOUT),
                                  nn.Linear(h, 1))

    def forward(self, batch) -> torch.Tensor:
        x = self.atom_embeddings[0](batch.x_cat[:, 0])
        for j in range(1, 9):
            x = x + self.atom_embeddings[j](batch.x_cat[:, j])
        x = x + self.degree_embedding(batch.deg) + self.ring_embedding(batch.ring)
        edge_attr = self.bond_embeddings[0](batch.edge_cat[:, 0])
        for j in range(1, 3):
            edge_attr = edge_attr + self.bond_embeddings[j](batch.edge_cat[:, j])
        for conv, norm in zip(self.convs, self.norms):
            residual = x
            x = norm(conv(x, batch.edge_index, edge_attr))
            x = self.drop(F.silu(x))
            x = x + residual
        from torch_geometric.nn import global_add_pool, global_mean_pool

        summed = global_add_pool(x, batch.batch)
        meaned = global_mean_pool(x, batch.batch)
        n_atoms = batch.n_atoms.view(-1, 1).to(x.dtype)
        return self.head(torch.cat([summed, meaned, torch.log1p(n_atoms)], dim=1)).view(-1)


def gine_categories(dims: Mapping[str, Sequence[int]]) -> dict[str, int]:
    counts = {f"atom_{j}": int(dims["atom"][j]) for j in range(9)}
    counts.update({f"bond_{j}": int(dims["bond"][j]) for j in range(3)})
    counts["degree"] = 17     # 0..15 exact, 16 = ">= 16"
    counts["ring"] = 14       # 0 acyclic, 3..12 exact, 13 = ">= 13"
    return counts


# ---------------------------------------------------------------------------
# pack one mini-batch of molecules
# ---------------------------------------------------------------------------


@dataclass
class SplitData:
    name: str
    row_ids: np.ndarray
    labels: np.ndarray
    graph_n_atoms: np.ndarray
    graph_n_bonds: np.ndarray
    atom_feat: np.ndarray
    bond_feat: np.ndarray
    arrays: Mapping[str, np.ndarray]
    hop: tuple[np.ndarray, np.ndarray] | None = None
    scratch: dict[str, np.ndarray] = None  # type: ignore[assignment]

    def __post_init__(self) -> None:
        n_words = int(self.arrays["graph_word_ptr"][-1])
        n_atoms = int(self.arrays["graph_atom_ptr"][-1])
        n_bonds = int(self.arrays["graph_bond_ptr"][-1])
        n_inc = int(self.arrays["inc_ptr"][-1])
        n_pair = int(self.arrays["pair_ptr"][-1])
        self.scratch = {
            "word": np.zeros(n_words, dtype=np.int64),
            "atom": np.zeros(n_atoms, dtype=np.int64),
            "bond": np.zeros(n_bonds, dtype=np.int64),
            "inc": np.zeros(n_inc, dtype=np.int64),
        }


def _range_ids(lo: np.ndarray, hi: np.ndarray) -> np.ndarray:
    counts = hi - lo
    offsets = np.concatenate([[0], np.cumsum(counts)])
    return np.repeat(lo, counts) + (np.arange(int(offsets[-1])) - np.repeat(offsets[:-1], counts))


def _gather_ranges(values: np.ndarray, ptr: np.ndarray, lo: np.ndarray, hi: np.ndarray,
                   new_off: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Concatenate ``values[ptr[w]]`` for the selected ranges, with batch-local owner ids."""
    sel_lo = ptr[lo]
    sel_hi = ptr[hi]
    counts = sel_hi - sel_lo
    total = int(counts.sum())
    owner = np.repeat(np.arange(len(lo)) + new_off[:-1] * 0, counts)
    idx = _range_ids(sel_lo, sel_hi)
    return values[idx], owner, new_off


def pack_batch(data: SplitData, idx: Sequence[int], *, attrpair: bool = False,
               hop: bool = False) -> dict[str, Any]:
    arr = data.arrays
    idx = np.asarray(idx, dtype=np.int64)
    G = int(len(idx))
    gw, ga, gb = arr["graph_word_ptr"], arr["graph_atom_ptr"], arr["graph_bond_ptr"]
    w_lo, w_hi = gw[idx], gw[idx + 1]
    a_lo, a_hi = ga[idx], ga[idx + 1]
    b_lo, b_hi = gb[idx], gb[idx + 1]
    w_cnt, a_cnt, b_cnt = w_hi - w_lo, a_hi - a_lo, b_hi - b_lo
    w_new = np.concatenate([[0], np.cumsum(w_cnt)])
    a_new = np.concatenate([[0], np.cumsum(a_cnt)])
    b_new = np.concatenate([[0], np.cumsum(b_cnt)])
    word_ids = _range_ids(w_lo, w_hi)
    atom_ids = _range_ids(a_lo, a_hi)
    bond_ids = _range_ids(b_lo, b_hi)
    sw, sa, sb, si = data.scratch["word"], data.scratch["atom"], data.scratch["bond"], data.scratch["inc"]
    sw[word_ids] = np.arange(len(word_ids))
    sa[atom_ids] = np.arange(len(atom_ids))
    sb[bond_ids] = np.arange(len(bond_ids))
    graph_of_word = np.repeat(np.arange(G), w_cnt)
    # incidences / pairs per graph are contiguous because words are
    i_lo, i_hi = arr["inc_ptr"][w_lo], arr["inc_ptr"][w_hi]
    p_lo, p_hi = arr["pair_ptr"][w_lo], arr["pair_ptr"][w_hi]
    inc_ids = _range_ids(i_lo, i_hi)
    pair_ids = _range_ids(p_lo, p_hi)
    i_new = np.concatenate([[0], np.cumsum(i_hi - i_lo)])
    p_new = np.concatenate([[0], np.cumsum(p_hi - p_lo)])
    si[inc_ids] = np.arange(len(inc_ids))
    word_natoms = arr["word_n_atoms"][word_ids]
    # atoms inside words
    wpa = arr["word_ptr_atom"]
    wpb = arr["word_ptr_bond"]
    out: dict[str, Any] = {}
    out["n_graphs"] = G
    out["word_tmpl"] = arr["word_tmpl"][word_ids]
    out["word_natoms"] = word_natoms
    out["graph_of_word"] = graph_of_word
    out["word_atoms"] = sa[arr["word_atoms"][_range_ids(wpa[word_ids], wpa[word_ids + 1])]]
    out["word_of_atom"] = np.repeat(np.arange(len(word_ids)), np.diff(wpa)[word_ids])
    out["word_bonds"] = sb[arr["word_bonds"][_range_ids(wpb[word_ids], wpb[word_ids + 1])]]
    out["word_of_bond"] = np.repeat(np.arange(len(word_ids)), np.diff(wpb)[word_ids])
    out["word_ptr_atom"] = np.concatenate([[0], np.cumsum(np.diff(wpa)[word_ids])])
    out["inc_other"] = sw[arr["inc_other"][inc_ids]]
    out["inc_own_atom"] = sa[arr["inc_own_atom"][inc_ids]]
    out["inc_other_atom"] = sa[arr["inc_other_atom"][inc_ids]]
    out["inc_bond"] = sb[arr["inc_bond"][inc_ids]]
    all_words = np.arange(len(word_ids))
    out["word_of_inc"] = np.repeat(all_words, np.diff(arr["inc_ptr"])[word_ids])
    out["pair_a"] = si[arr["pair_a"][pair_ids]]
    out["pair_b"] = si[arr["pair_b"][pair_ids]]
    out["pair_R"] = arr["pair_R"][pair_ids]
    out["pair_ER"] = arr["pair_ER"][pair_ids]
    out["pair_fields"] = arr["pair_fields"][pair_ids]
    out["word_of_pair"] = np.repeat(all_words, np.diff(arr["pair_ptr"])[word_ids])
    out["atom_feat"] = data.atom_feat[atom_ids]
    out["bond_feat"] = data.bond_feat[bond_ids]
    out["n_atoms_graph"] = data.graph_n_atoms[idx]
    out["g_size"] = np.stack([np.log1p(data.graph_n_atoms[idx].astype(np.float64)),
                              np.log1p(data.graph_n_bonds[idx].astype(np.float64))], axis=1)
    out["row_ids"] = data.row_ids[idx]
    out["y"] = data.labels[idx]
    out["R_pair_table"] = None
    out["ap_n"] = 0
    if attrpair:
        pair_lists = {n: np.asarray([(i, j) for i in range(n) for j in range(i + 1, n)], dtype=np.int64)
                      for n in range(2, W.MAX_WORD_ATOMS + 1)}
        words = np.nonzero(word_natoms > 1)[0]
        if len(words):
            counts = np.asarray([len(pair_lists[int(word_natoms[w])]) for w in words])
            ap_i = np.concatenate([pair_lists[int(word_natoms[w])][:, 0] for w in words])
            ap_j = np.concatenate([pair_lists[int(word_natoms[w])][:, 1] for w in words])
            out["ap_word"] = np.repeat(words, counts)
            out["ap_i"] = ap_i
            out["ap_j"] = ap_j
            out["ap_n"] = int(len(ap_i))
    if hop:
        hsrc, hdst = data.hop
        h_lo, h_hi = hsrc[w_lo], hsrc[w_hi]
        hid = _range_ids(h_lo, h_hi)
        out["hop_src"] = sw[hdst[hid]]
        out["hop_dst"] = np.repeat(all_words, h_hi - h_lo)
    return out


def build_hop_index(graph_word_ptr: np.ndarray, inc_ptr: np.ndarray, inc_other: np.ndarray):
    """Unique centre-word index at word-graph distance 1 or 2 (excluding self)."""
    n_words = int(graph_word_ptr[-1])
    src_all, dst_all = [], []
    for g in range(len(graph_word_ptr) - 1):
        lo, hi = int(graph_word_ptr[g]), int(graph_word_ptr[g + 1])
        n = hi - lo
        adj = [[] for _ in range(n)]
        for w in range(lo, hi):
            for k in range(int(inc_ptr[w]), int(inc_ptr[w + 1])):
                j = int(inc_other[k]) - lo
                if 0 <= j < n:
                    adj[w - lo].append(j)
        for i in range(n):
            seen = {i}
            frontier = {i}
            for _ in range(2):
                nxt = set()
                for v in frontier:
                    for j in adj[v]:
                        if j not in seen:
                            nxt.add(j)
                seen |= nxt
                frontier = nxt
            others = sorted(seen - {i})
            dst_all.extend([lo + i] * len(others))
            src_all.extend([lo + j for j in others])
    return (np.asarray(src_all, dtype=np.int64).reshape(-1),
            np.asarray(dst_all, dtype=np.int64).reshape(-1))


def to_device(batch: Mapping[str, Any], device: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for key, value in batch.items():
        if isinstance(value, torch.Tensor):
            out[key] = value
            continue
        if value is None or key == "R_pair_table":
            out[key] = value
            continue
        if isinstance(value, np.ndarray):
            dtype = torch.float32 if value.dtype.kind == "f" else torch.long
            out[key] = torch.as_tensor(np.ascontiguousarray(value), dtype=dtype, device=device)
        else:
            out[key] = value
    return out

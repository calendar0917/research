"""Word-type content, full-attribute instance graph, port states, sentences, readout.

One module class holds every arm of the round.  ``ModelSpec`` flags only switch
*computation*: parameter names, shapes and (through the name-derived
initialiser) initial values are identical in every arm, so a difference between
two arms is never a difference in parameter count or initialisation.
"""

from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import ws_model as OM  # old round: the frozen WGINE word-type encoder
import ws_words as W

DROPOUT = 0.10
DIST_BINS = 6
DIST_DIM = 8


def _mlp(inp: int, hidden: int, out: int, *, drop: float = DROPOUT) -> nn.Sequential:
    return nn.Sequential(nn.Linear(inp, hidden), nn.SiLU(), nn.Dropout(drop), nn.Linear(hidden, out))


@dataclass(frozen=True)
class ModelSpec:
    port: str = "bind"        # bind | mean
    context: str = "rel"      # rel | self | global
    instance: str = "graph"   # graph | set
    readout: str = "vector"   # vector | scalar
    bag: bool = False         # D2: all cross-word content replaced by the word itself

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


ARMS: dict[str, ModelSpec] = {
    "MEAN_SELF": ModelSpec("mean", "self", "graph", "vector"),
    "BIND_SELF": ModelSpec("bind", "self", "graph", "vector"),
    "MEAN_REL": ModelSpec("mean", "rel", "graph", "vector"),
    "BIND_REL": ModelSpec("bind", "rel", "graph", "vector"),
    "SET_MEAN_SELF": ModelSpec("mean", "self", "set", "vector"),
    "BIND_SELF_SCALAR": ModelSpec("bind", "self", "graph", "scalar"),
    # branch arms (PROTOCOL section 5, D5/D6)
    "BAG_CAP": ModelSpec("bind", "rel", "graph", "vector", bag=True),
    "GLOBAL_CTX_BIND": ModelSpec("bind", "global", "graph", "vector"),
    "GLOBAL_CTX_MEAN": ModelSpec("mean", "global", "graph", "vector"),
}
CONTEXT_ARMS = ("rel", "self", "global")
PORT_ARMS = ("bind", "mean")


def _scatter(src: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = torch.zeros(n, src.shape[1], dtype=src.dtype, device=src.device)
    out.index_add_(0, index, src)
    return out


class PortBindModel(nn.Module):
    def __init__(self, spec: ModelSpec, tmpl: Mapping[str, Any], dims: Mapping[str, Any],
                 hidden: int = 64, n_inst_layers: int = 3, n_ctx_layers: int = 2):
        super().__init__()
        self.spec = spec
        self.hidden = int(hidden)
        H = self.hidden
        self.n_ctx_layers = int(n_ctx_layers)
        self.n_inst_layers = int(n_inst_layers)
        self.raw_atom = nn.ModuleList([nn.Embedding(int(d), 4) for d in dims["atom"]])
        self.raw_bond = nn.ModuleList([nn.Embedding(int(d), 4) for d in dims["bond"]])
        self.encoder = OM.WGINEEncoder(tmpl, dims, H)
        self.enc_proj = nn.Identity() if OM.ENC_DIM == H else nn.Linear(OM.ENC_DIM, H)
        # full-attribute instance graph
        self.inst_in = nn.Linear(4 * len(dims["atom"]), H)
        self.bond_in = nn.Linear(4 * len(dims["bond"]), H)
        self.inst_eps = nn.ParameterList([nn.Parameter(torch.zeros(1)) for _ in range(self.n_inst_layers)])
        self.inst_lin1 = nn.ModuleList([nn.Linear(H, 2 * H) for _ in range(self.n_inst_layers)])
        self.inst_lin2 = nn.ModuleList([nn.Linear(2 * H, H) for _ in range(self.n_inst_layers)])
        self.inst_norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(self.n_inst_layers)])
        self.inst_drop = nn.Dropout(DROPOUT)
        self.z_mlp = _mlp(4 * H + 2, H, H)
        # port states
        self.dist_emb = nn.Embedding(DIST_BINS, DIST_DIM)
        self.site_lin1 = nn.Linear(H + DIST_DIM, H)
        self.site_lin2 = nn.Linear(H, H)
        self.port_lin1 = nn.Linear(3 * H, H)
        self.port_lin2 = nn.Linear(H, H)
        # cross-word interface
        atom_dim = 4 * len(dims["atom"])
        bond_dim = 4 * len(dims["bond"])
        self.edge_feat_dim = 2 * H + 2 * atom_dim + bond_dim
        self.msg_lin1 = nn.Linear(2 * H + self.edge_feat_dim, H)
        self.msg_lin2 = nn.Linear(H, H)
        self.pair_lin1 = nn.Linear(3 * H + W.R_DIM + W.PAIR_FIELD_DIM, H)
        self.pair_lin2 = nn.Linear(H, H)
        self.sent_lin1 = nn.Linear(4 * H + 2, H)
        self.sent_lin2 = nn.Linear(H, H)
        self.sent_norm = nn.LayerNorm(H)
        # sentence composition
        self.ctx_lin1 = nn.ModuleList([nn.Linear(H + self.edge_feat_dim, H) for _ in range(self.n_ctx_layers)])
        self.ctx_lin2 = nn.ModuleList([nn.Linear(H, H) for _ in range(self.n_ctx_layers)])
        self.upd_lin1 = nn.ModuleList([nn.Linear(3 * H + 1, H) for _ in range(self.n_ctx_layers)])
        self.upd_lin2 = nn.ModuleList([nn.Linear(H, H) for _ in range(self.n_ctx_layers)])
        self.upd_norm = nn.ModuleList([nn.LayerNorm(H) for _ in range(self.n_ctx_layers)])
        # readout
        self.q_lin1 = nn.Linear(3 * H, H)
        self.q_lin2 = nn.Linear(H, H)
        self.rho_lin1 = nn.Linear(H + 3, H)
        self.rho_lin2 = nn.Linear(H, 1)
        self.drop = nn.Dropout(DROPOUT)
        self.register_buffer("tmpl_labels", _t(tmpl, "tmpl_labels", torch.long))
        self.register_buffer("tmpl_mask", _t(tmpl, "tmpl_mask", torch.float32))
        self.register_buffer("tmpl_n_atoms", _t(tmpl, "tmpl_n_atoms", torch.long))
        self.register_buffer("node_tmpl", _t(tmpl, "node_tmpl", torch.long))
        self.register_buffer("node_lab", _t(tmpl, "node_lab", torch.long))
        self.register_buffer("edge_index", _t(tmpl, "edge_index", torch.long))
        self.register_buffer("edge_type", _t(tmpl, "edge_type", torch.long))
        self.register_buffer("multi_row", _t(tmpl, "multi_row", torch.long))
        self.register_buffer("is_multi", _t(tmpl, "is_multi", torch.bool))
        self.register_buffer("tmpl_bond", _t(tmpl, "tmpl_bond", torch.long))
        self.register_buffer("tmpl_dist", _t(tmpl, "dist", torch.long))
        self.n_templates = int(np.asarray(tmpl["n_templates"]).reshape(-1)[0])

    # -- helpers ---------------------------------------------------------
    def atom36(self, atom_feat: torch.Tensor) -> torch.Tensor:
        return torch.cat([emb(atom_feat[:, j]) for j, emb in enumerate(self.raw_atom)], dim=-1)

    def bond12(self, bond_feat: torch.Tensor) -> torch.Tensor:
        return torch.cat([emb(bond_feat[:, j]) for j, emb in enumerate(self.raw_bond)], dim=-1)

    def content(self) -> torch.Tensor:
        t = {"tmpl_labels": self.tmpl_labels, "tmpl_n_atoms": self.tmpl_n_atoms,
             "node_tmpl": self.node_tmpl, "node_lab": self.node_lab,
             "edge_index": self.edge_index, "edge_type": self.edge_type,
             "multi_row": self.multi_row, "is_multi": self.is_multi,
             "n_templates": self.n_templates}
        return self.enc_proj(self.encoder(t))

    # -- forward ---------------------------------------------------------
    def forward(self, b: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        H = self.hidden
        spec = self.spec
        dev = b["word_tmpl"].device
        a36 = self.atom36(b["atom_feat"])
        b12 = self.bond12(b["bond_feat"])
        e = self.content()[b["word_tmpl"]]
        n_words = int(e.shape[0])
        n_sites = int(b["site_atom"].shape[0])
        wn = b["word_natoms"]
        site_word = b["site_word"]
        n_ib = b["word_n_ibonds"]
        t = self.inst_in(a36[b["site_atom"]])
        eb = self.bond_in(b12[b["edge_bond"]])
        ubond = self.bond_in(b12[b["word_bonds"]])
        wn_f = wn.to(t.dtype).unsqueeze(-1)
        nb_f = n_ib.to(t.dtype).unsqueeze(-1)
        bond_mean_word = _scatter(ubond, b["word_of_bond"], n_words) / nb_f.clamp(min=1.0)
        if spec.instance == "set":
            dbar = (2.0 * n_ib.to(t.dtype) / wn.to(t.dtype))
            for layer in range(self.n_inst_layers):
                t_sum = _scatter(t, site_word, n_words)
                t_mean = t_sum / wn_f
                inner = F.silu(t + bond_mean_word[site_word])
                mean_inner = _scatter(inner, site_word, n_words) / wn_f
                a_set = (torch.sqrt(dbar).unsqueeze(-1) * mean_inner)[site_word]
                a_set = a_set * (n_ib > 0).to(t.dtype).unsqueeze(-1)[site_word]
                pre = (1.0 + self.inst_eps[layer]) * t + a_set
                hidden = F.silu(self.inst_lin1[layer](pre))
                upd = self.inst_lin2[layer](hidden)
                t = self.inst_norm[layer](t + self.inst_drop(upd))
        else:
            deg = b["site_deg"].to(t.dtype).clamp(min=1.0).rsqrt().unsqueeze(-1)
            for layer in range(self.n_inst_layers):
                msg = F.silu(t[b["edge_src"]] + eb)
                agg = _scatter(msg, b["edge_dst"], n_sites) * deg
                pre = (1.0 + self.inst_eps[layer]) * t + agg
                upd = self.inst_lin2[layer](F.silu(self.inst_lin1[layer](pre)))
                t = self.inst_norm[layer](t + self.inst_drop(upd))
        t_sum = _scatter(t, site_word, n_words)
        t_mean = t_sum / wn_f
        z = self.z_mlp(torch.cat([e, t_mean, t_sum, bond_mean_word,
                                  torch.log1p(wn.to(e.dtype)).unsqueeze(-1),
                                  torch.log1p(b["word_n_inc"].to(e.dtype)).unsqueeze(-1)], dim=-1))
        # relative-site value: mean over the word's sites of MLP_site([t_a, EmbDist(d(p,a))])
        w1 = self.site_lin1.weight
        hw = t @ w1[:, :H].T + self.site_lin1.bias
        hd = self.dist_emb.weight @ w1[:, H:].T
        hidden_all = F.silu(hw.unsqueeze(1) + hd.unsqueeze(0))       # (S, 6, H)
        if spec.instance == "set":
            per_word = _scatter(hidden_all.sum(dim=1), site_word, n_words)
            v_pre = (per_word / (DIST_BINS * wn_f))[site_word]
        else:
            gathered = hidden_all[b["pp_a"], b["pp_d"]]             # (P, H)
            v_pre = _scatter(gathered, b["pp_p"], n_sites) / wn_f[site_word]
        v = v_pre @ self.site_lin2.weight.T + self.site_lin2.bias
        if spec.instance == "set":
            port_in = torch.cat([z[site_word], t_mean[site_word], v], dim=-1)
        else:
            port_in = torch.cat([z[site_word], t, v], dim=-1)
        c = F.silu(self.port_lin1(port_in))
        c = self.port_lin2(c)
        if spec.port == "mean":
            gsum = _scatter(c, b["site_group"], int(b["n_groups"]))
            gmean = gsum / b["group_count"].to(c.dtype).unsqueeze(-1)
            chosen = gmean[b["site_group"]]
        else:
            chosen = c
        # cross-word interface
        own = b["inc_own_site"]
        other = b["inc_other_site"]
        word_of_inc = b["word_of_inc"]
        other_word = b["inc_other_word"]
        if spec.bag:
            edge_feat = torch.cat([chosen[own], chosen[own], a36[b["site_atom"][own]],
                                   a36[b["site_atom"][own]], b12[b["inc_bond"]]], dim=-1)
            msg_in = torch.cat([z[word_of_inc], z[word_of_inc], edge_feat], dim=-1)
        else:
            edge_feat = torch.cat([chosen[own], chosen[other], a36[b["site_atom"][own]],
                                   a36[b["site_atom"][other]], b12[b["inc_bond"]]], dim=-1)
            msg_in = torch.cat([z[word_of_inc], z[other_word], edge_feat], dim=-1)
        m = self.msg_lin2(F.silu(self.msg_lin1(msg_in)))
        d_inc = b["word_n_inc"].to(e.dtype)
        dcl = d_inc.clamp(min=1.0).unsqueeze(-1)
        fields = b["pair_fields"]
        if spec.bag:
            fields = torch.zeros_like(fields)
        ma = m[b["pair_a"]]
        mb = m[b["pair_b"]]
        pair_in = torch.cat([z[b["word_of_pair"]], ma + mb, ma * mb, b["pair_R"], fields], dim=-1)
        p_hidden = F.silu(self.pair_lin1(pair_in))
        pair_vec = self.pair_lin2(p_hidden)
        d_pair = b["word_n_pairs"].to(e.dtype).clamp(min=1.0).unsqueeze(-1)
        m_sum = _scatter(m, word_of_inc, n_words)
        p_mean = _scatter(pair_vec, b["word_of_pair"], n_words) / d_pair
        sent_in = torch.cat([z, m_sum * dcl.rsqrt(), m_sum / dcl, p_mean,
                             torch.log1p(d_inc).unsqueeze(-1),
                             torch.log1p(b["word_n_pairs"].to(e.dtype)).unsqueeze(-1)], dim=-1)
        s = self.sent_norm(z + self.sent_lin2(F.silu(self.sent_lin1(sent_in))))
        s0 = s
        for layer in range(self.n_ctx_layers):
            if spec.context == "rel":
                source = s[other_word]
            elif spec.context == "self":
                source = s[word_of_inc]
            else:  # global mixture of every sentence in the same molecule
                graph_of_word = b["graph_of_word"]
                mu_w = (wn.to(e.dtype) / b["n_atoms_graph"].to(e.dtype)[graph_of_word]).unsqueeze(-1)
                mix = _scatter(mu_w * s, graph_of_word, int(b["n_graphs"]))
                source = mix[graph_of_word][word_of_inc]
            u = self.ctx_lin2[layer](F.silu(self.ctx_lin1[layer](torch.cat([source, edge_feat], dim=-1))))
            u_sum = _scatter(u, word_of_inc, n_words)
            upd_in = torch.cat([s, u_sum * dcl.rsqrt(), u_sum / dcl,
                                torch.log1p(d_inc).unsqueeze(-1)], dim=-1)
            s = self.upd_norm[layer](s + self.upd_lin2[layer](F.silu(self.upd_lin1[layer](upd_in))))
        q = self.q_lin2(F.silu(self.q_lin1(torch.cat([z, s0, s], dim=-1))))
        graph_of_word = b["graph_of_word"]
        mu = (wn.to(e.dtype) / b["n_atoms_graph"].to(e.dtype)[graph_of_word]).unsqueeze(-1)
        g = b["g_size"]
        if spec.readout == "scalar":
            val = self.rho_lin2(F.silu(self.rho_lin1(torch.cat([q, g[graph_of_word]], dim=-1))))
            pred = _scatter(mu * val, graph_of_word, int(b["n_graphs"])).view(-1)
        else:
            vec = _scatter(mu * q, graph_of_word, int(b["n_graphs"]))
            pred = self.rho_lin2(F.silu(self.rho_lin1(torch.cat([vec, g], dim=-1)))).view(-1)
        return {"pred": pred, "q": q, "s0": s0, "s2": s, "z": z, "c": c, "chosen": chosen,
                "m": m, "pair_hidden": p_hidden, "edge_feat": edge_feat, "mass": mu.view(-1),
                "g": g}


def _t(tmpl: Mapping[str, Any], key: str, dtype) -> torch.Tensor:
    value = tmpl[key]
    if isinstance(value, np.ndarray) and value.ndim == 0:
        value = value.reshape(1)
    return torch.as_tensor(np.ascontiguousarray(value), dtype=dtype)


# ---------------------------------------------------------------------------
# deterministic, name-derived initialisation
# ---------------------------------------------------------------------------


def _owner(module: nn.Module, name: str) -> nn.Module | None:
    parts = name.split(".")
    current: nn.Module | None = module
    for part in parts[:-1]:
        if current is None:
            return None
        try:
            current = getattr(current, part) if not part.isdigit() else current[int(part)]
        except (AttributeError, IndexError, TypeError):
            return None
    return current


def init_by_name(module: nn.Module, base_seed: int) -> str:
    """Initialise every parameter from ``sha256(base_seed|name)``.

    Construction order therefore cannot change the initial values, and two arms
    that share a parameter name are bit-identical at step 0.
    """
    digest = hashlib.sha256()
    for name, param in module.named_parameters():
        seed = int(hashlib.sha256(f"{base_seed}|{name}".encode()).hexdigest()[:16], 16)
        gen = torch.Generator()
        gen.manual_seed(seed % (2 ** 63 - 1))
        owner = _owner(module, name)
        with torch.no_grad():
            if isinstance(owner, nn.Embedding):
                param.normal_(0.0, 1.0, generator=gen)
            elif isinstance(owner, nn.Linear):
                bound = 1.0 / math.sqrt(owner.in_features)
                param.uniform_(-bound, bound, generator=gen)
            elif isinstance(owner, nn.LayerNorm):
                if name.endswith("weight"):
                    param.fill_(1.0)
                else:
                    param.zero_()
            else:
                param.zero_()
        digest.update(name.encode())
        digest.update(param.detach().cpu().numpy().tobytes())
    return digest.hexdigest()


def count_parameters(module: nn.Module) -> int:
    return int(sum(p.numel() for p in module.parameters()))

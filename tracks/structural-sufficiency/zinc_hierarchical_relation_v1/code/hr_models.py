"""Non-recursive hierarchical relation models (A = HCE, B = HCE+SRI, C = null).

A: atom -> bond -> junction -> rooted induced patch, with one shared
   coarse-to-fine refinement block reused at every level, train-only support
   modulation of the fine patch contribution, and a moment/cross-moment reader.
B: A plus one shared low-rank pair interaction over graph-related structural
   tokens, with real relation-type biases; no zero-init gate, no recursion.
C: B with the relation-to-pair assignment permuted within each molecule
   (prepared for a later null-control run; not a formal arm of this round).
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import hr_common as C
import hr_data as D


class MLP(nn.Module):
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


class RefineBlock(nn.Module):
    """One shared local-environment refinement block (residual + LayerNorm)."""

    def __init__(self, d: int, role_dim: int, hidden: int):
        super().__init__()
        self.fc1 = nn.Linear(3 * d + role_dim, hidden)
        self.fc2 = nn.Linear(hidden, d)
        self.norm = nn.LayerNorm(d)

    def forward(self, h, child_sum, child_sq, role):
        x = torch.cat([h, child_sum, child_sq, role], dim=-1)
        return self.norm(h + self.fc2(F.silu(self.fc1(x))))


def _seg_sum(x: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    out = x.new_zeros((n, x.shape[1]))
    if index.numel():
        out.index_add_(0, index, x)
    return out


def _moments(h: torch.Tensor, batch: torch.Tensor, n_graphs: int):
    if h.shape[0] == 0:
        z = h.new_zeros((n_graphs, h.shape[1]))
        return z, z, z, h.new_zeros((n_graphs,))
    s = _seg_sum(h, batch, n_graphs)
    cnt = torch.bincount(batch, minlength=n_graphs).to(h.dtype).clamp(min=1.0)
    m = s / cnt.unsqueeze(1)
    m2 = _seg_sum(h * h, batch, n_graphs) / cnt.unsqueeze(1)
    return s, m, m2, cnt


def _cross_mean(x: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    if x.shape[0] == 0:
        return x.new_zeros((n, x.shape[1]))
    s = _seg_sum(x, index, n)
    cnt = torch.bincount(index, minlength=n).to(x.dtype).clamp(min=1.0)
    return s / cnt.unsqueeze(1)


class SRI(nn.Module):
    """One-shot low-rank relational interaction over structural tokens."""

    def __init__(self, d: int, rank: int, rel_types: int, dropout: float = 0.0):
        super().__init__()
        self.rank = int(rank)
        self.q = nn.Linear(d, rank, bias=False)
        self.k = nn.Linear(d, rank, bias=False)
        self.v = nn.Linear(d, rank, bias=False)
        self.rel_emb = nn.Embedding(rel_types, rank)
        self.rel_bias = nn.Parameter(torch.zeros(rel_types))
        self.out = nn.Linear(rank, d)
        self.norm = nn.LayerNorm(d)
        self.dropout = float(dropout)

    def forward(self, h: torch.Tensor, r_src: torch.Tensor, r_dst: torch.Tensor, r_rel: torch.Tensor):
        n = h.shape[0]
        if n == 0 or r_src.numel() == 0:
            return h, {"attn_max_mean": h.new_tensor(0.0), "attn_entropy_mean": h.new_tensor(0.0)}
        q = self.q(h)
        k = self.k(h)
        v = self.v(h)
        rel_e = self.rel_emb(r_rel)
        score = (q[r_src] * (k[r_dst] + rel_e)).sum(dim=-1) / (self.rank ** 0.5)
        score = score + self.rel_bias[r_rel]
        with torch.no_grad():
            seg_max = torch.full((n,), -1.0e9, device=h.device, dtype=score.dtype)
            seg_max.scatter_reduce_(0, r_src, score.detach(), reduce="amax", include_self=True)
        exp_score = torch.exp(score - seg_max[r_src])
        denom = torch.zeros(n, device=h.device, dtype=score.dtype)
        denom.index_add_(0, r_src, exp_score)
        alpha = exp_score / denom[r_src].clamp_min(1.0e-12)
        o = v.new_zeros((n, self.rank))
        o.index_add_(0, r_src, alpha.unsqueeze(1) * v[r_dst])
        h_out = self.norm(h + self.out(o))
        if self.dropout > 0.0:
            h_out = F.dropout(h_out, p=self.dropout, training=self.training)
        with torch.no_grad():
            amax = torch.zeros(n, device=h.device, dtype=score.dtype)
            amax.scatter_reduce_(0, r_src, alpha, reduce="amax", include_self=True)
            ent = torch.zeros(n, device=h.device, dtype=score.dtype)
            ent.index_add_(0, r_src, -alpha * torch.log(alpha.clamp_min(1.0e-12)))
            stats = {
                "attn_max_mean": amax.mean(),
                "attn_entropy_mean": ent.mean(),
            }
        return h_out, stats


class HRNet(nn.Module):
    def __init__(
        self,
        *,
        arm: str,
        d: int = 96,
        role_dim: int = 8,
        refine_hidden: int = 128,
        refine_repeats: int = 2,
        patch_node_hidden: int = 32,
        patch_edge_hidden: int = 16,
        patch_mlp_hidden: int = 96,
        reader_hidden: int = 96,
        reader_layers: int = 2,
        interaction_rank: int = 16,
        dropout: float = 0.0,
    ):
        super().__init__()
        if arm not in ("A", "B", "C"):
            raise ValueError(arm)
        self.arm = arm
        self.d = int(d)
        self.refine_repeats = int(refine_repeats)
        self.dropout = float(dropout)

        # frozen BJOC masked-SSL encoders (same architecture / keys)
        self.atom_encoder = MLP(D.ATOM_DIM, 128, d, 1, dropout)
        self.bond_encoder = MLP(3 * d + D.BOND_DIM, 160, d, 1, dropout)
        self.junction_encoder = MLP(4 * d + 2 * d + D.JUNC_DIM, 160, d, 1, dropout)

        # shared coarse-to-fine refinement (one parameter set, reused)
        self.refine = RefineBlock(d, role_dim, refine_hidden)
        self.role = nn.Embedding(3, role_dim)

        # rooted induced patch encoder
        self.patch_node = nn.Linear(d + D.PATCH_TAG_DIM, patch_node_hidden)
        self.patch_edge = nn.Linear(D.EDGE_FEAT_DIM, patch_edge_hidden)
        self.support_trunk = nn.Sequential(
            nn.Linear(D.SUPPORT_DIM, 16),
            nn.SiLU(),
        )
        self.support_node = nn.Linear(16, patch_node_hidden)
        self.support_edge = nn.Linear(16, patch_edge_hidden)
        patch_in = d + 2 * patch_node_hidden + 2 * patch_edge_hidden + D.PATCH_SCALAR_DIM + D.SUPPORT_DIM
        self.patch_mlp = MLP(patch_in, patch_mlp_hidden, d, 1, dropout)

        # reader over hierarchy moments and local cross moments
        self.reader_in = 4 * (3 * d) + 4 + 3 * d + D.GLOBAL_DIM
        self.reader = MLP(self.reader_in, reader_hidden, 1, reader_layers, dropout)

        if arm in ("B", "C"):
            self.interaction = SRI(d, interaction_rank, D.REL_TYPES, dropout)
            self.interaction_rank = int(interaction_rank)

        self._last_stats: dict[str, torch.Tensor] = {}

    # ------------------------------------------------------------------
    def load_ssl_encoders(self, path) -> dict[str, Any]:
        state = torch.load(path, map_location="cpu", weights_only=False)
        loaded = {}
        for name in ("atom_encoder", "bond_encoder", "junction_encoder"):
            if name not in state:
                raise SystemExit(f"SSL checkpoint {path} has no {name}")
            getattr(self, name).load_state_dict(state[name], strict=True)
            loaded[name] = C.sha256_state_dict(state[name])
        return {"source": str(path), "sha256": C.sha256_file(path), "encoder_tensor_hashes": loaded}

    # ------------------------------------------------------------------
    def _refine_twice(self, h, child_sum, child_sq, role):
        for _ in range(self.refine_repeats):
            h = self.refine(h, child_sum, child_sq, role)
        return h

    def forward_features(self, batch: Mapping[str, Any], relation_mode: str | None = None):
        if relation_mode is None:
            relation_mode = "real" if self.arm != "C" else "null"
        dev = batch["atom_batch"].device
        dtype = torch.float32
        n_graphs = int(batch["n_graphs"])
        d = self.d

        # ---- atom level --------------------------------------------------
        a_raw = self.atom_encoder(batch["atom_feat"])
        a = F.layer_norm(a_raw, (d,))

        # ---- bond level --------------------------------------------------
        bidx = batch["bond_index"]
        au, av = a[bidx[:, 0]], a[bidx[:, 1]]
        au_raw, av_raw = a_raw[bidx[:, 0]], a_raw[bidx[:, 1]]
        if bidx.shape[0]:
            b_raw = self.bond_encoder(
                torch.cat([au_raw + av_raw, (au_raw - av_raw).abs(), au_raw * av_raw,
                           batch["bond_feat"]], dim=1)
            )
        else:
            b_raw = a.new_zeros((0, d))
        role_b = self.role(torch.full((b_raw.shape[0],), 0, device=dev, dtype=torch.long))
        b = self._refine_twice(b_raw, au + av, au * au + av * av, role_b)

        # ---- junction level ---------------------------------------------
        jidx = batch["junc_index"]
        jb = batch["junc_bond"]
        nj = jidx.shape[0]
        if nj:
            ai, aj, ak = a[jidx[:, 0]], a[jidx[:, 1]], a[jidx[:, 2]]
            b1, b2 = b[jb[:, 0]], b[jb[:, 1]]
            j_raw = self.junction_encoder(
                torch.cat([ai + ak, (ai - ak).abs(), ai * ak, aj, b1 + b2, (b1 - b2).abs(),
                           batch["junc_feat"]], dim=1)
            )
            child_sum = b1 + b2 + aj
            child_sq = b1 * b1 + b2 * b2 + aj * aj
        else:
            j_raw = a.new_zeros((0, d))
            child_sum = a.new_zeros((0, d))
            child_sq = a.new_zeros((0, d))
        role_j = self.role(torch.full((nj,), 1, device=dev, dtype=torch.long))
        j = self._refine_twice(j_raw, child_sum, child_sq, role_j)

        # ---- rooted induced patch level ---------------------------------
        npatch = int(batch["n_patch_total"])
        if npatch:
            du = batch["pnode_du"]
            dv = batch["pnode_dv"]
            tag = torch.cat([
                F.one_hot(torch.minimum(du, dv), D.NODE_TAG_DIM),
                F.one_hot(torch.maximum(du, dv), D.NODE_TAG_DIM),
                F.one_hot((du - dv).abs(), D.NODE_TAG_DIM),
            ], dim=1).to(dtype)
            node_h = F.silu(self.patch_node(torch.cat([a[batch["pnode_atom"]], tag], dim=1)))
            node_sum = _seg_sum(node_h, batch["pnode_patch"], npatch)
            node_sq = _seg_sum(node_h * node_h, batch["pnode_patch"], npatch)
            ebt = F.one_hot(batch["pedge_type"], D.BOND_CATEGORIES).to(dtype)

            def _tag(a_du, a_dv):
                return torch.cat([
                    F.one_hot(torch.minimum(a_du, a_dv), D.NODE_TAG_DIM),
                    F.one_hot(torch.maximum(a_du, a_dv), D.NODE_TAG_DIM),
                    F.one_hot((a_du - a_dv).abs(), D.NODE_TAG_DIM),
                ], dim=1).to(dtype)

            tag_x = _tag(batch["pedge_du_x"], batch["pedge_dv_x"])
            tag_y = _tag(batch["pedge_du_y"], batch["pedge_dv_y"])
            eroot = batch["pedge_root"].to(dtype).unsqueeze(1)
            edge_h = F.silu(self.patch_edge(torch.cat([ebt, tag_x + tag_y, (tag_x - tag_y).abs(), eroot], dim=1)))
            edge_sum = _seg_sum(edge_h, batch["pedge_patch"], npatch)
            edge_sq = _seg_sum(edge_h * edge_h, batch["pedge_patch"], npatch)
            support = batch["p_support"]
            st = self.support_trunk(support)
            gate_node = torch.sigmoid(self.support_node(st))
            gate_edge = torch.sigmoid(self.support_edge(st))
            root_b = b[batch["p_root_bond"]]
            fine = self.patch_mlp(
                torch.cat([root_b, gate_node * node_sum, gate_edge * edge_sum, node_sq, edge_sq,
                           batch["p_scalars"], support], dim=1)
            )
            role_p = self.role(torch.full((npatch,), 2, device=dev, dtype=torch.long))
            p = self._refine_twice(fine, root_b, root_b * root_b, role_p)
            patch_stats = {
                "support_gate_mean": torch.cat([gate_node, gate_edge], dim=1).mean().detach(),
                "support_gate_std": torch.cat([gate_node, gate_edge], dim=1).std().detach(),
                "node_count": torch.tensor(float(batch["pnode_atom"].numel()), device=dev),
                "edge_count": torch.tensor(float(batch["pedge_x"].numel()), device=dev),
            }
        else:
            node_sum = node_sq = edge_sum = edge_sq = a.new_zeros((0, self.patch_node.out_features))
            p = a.new_zeros((0, d))
            patch_stats = {
                "support_gate_mean": a.new_tensor(0.0),
                "support_gate_std": a.new_tensor(0.0),
                "node_count": a.new_tensor(0.0),
                "edge_count": a.new_tensor(0.0),
            }

        inter_stats = {
            "attn_max_mean": a.new_tensor(0.0),
            "attn_entropy_mean": a.new_tensor(0.0),
        }
        if self.arm in ("B", "C"):
            h = torch.cat([j, p], dim=0)
            rel = batch["r_rel"] if relation_mode == "real" else batch["r_rel_null"]
            h2, inter_stats = self.interaction(h, batch["r_src"], batch["r_dst"], rel)
            j = h2[:nj]
            p = h2[nj:]

        # ---- readout -----------------------------------------------------
        feats = []
        for h, level_batch in ((a, batch["atom_batch"]), (b, batch["bond_batch"]),
                               (j, batch["junc_batch"]), (p, batch["patch_batch"])):
            s, m, m2, cnt = _moments(h, level_batch, n_graphs)
            feats.extend([s, m, m2, torch.log1p(cnt).unsqueeze(1) / 4.0])
        # local cross moments
        if b.shape[0]:
            cross_b = _cross_mean(a[bidx[:, 0]] * a[bidx[:, 1]], batch["bond_batch"], n_graphs)
        else:
            cross_b = a.new_zeros((n_graphs, d))
        if nj:
            cross_j = _cross_mean(b[jb[:, 0]] * b[jb[:, 1]], batch["junc_batch"], n_graphs)
        else:
            cross_j = a.new_zeros((n_graphs, d))
        if npatch:
            cross_p = _cross_mean(b[batch["p_root_bond"]] * p, batch["patch_batch"], n_graphs)
        else:
            cross_p = a.new_zeros((n_graphs, d))
        feats.extend([cross_b, cross_j, cross_p])
        feats.append(batch["global_feat"])
        reader_in = torch.cat(feats, dim=1)
        pred = self.reader(reader_in).view(-1)
        self._last_stats = {**patch_stats, **inter_stats}
        return pred, {
            "reader_in": reader_in,
            "a": a, "b": b, "j": j, "p": p,
            "cross_b": cross_b, "cross_j": cross_j, "cross_p": cross_p,
        }

    def forward(self, batch: Mapping[str, Any], relation_mode: str | None = None):
        pred, _ = self.forward_features(batch, relation_mode=relation_mode)
        return pred


def build_model(
    *,
    arm: str,
    ssl_path,
    seed: int = C.SEED,
    cfg: Mapping[str, Any] | None = None,
    device: str | torch.device = "cpu",
) -> tuple[HRNet, dict[str, Any]]:
    cfg = dict(cfg or C.load_config()["model"])
    torch.manual_seed(int(seed))
    model = HRNet(arm=arm, **cfg)
    ssl_info = model.load_ssl_encoders(ssl_path)
    model = model.to(device)
    info = {
        "arm": arm,
        "ssl": ssl_info,
        "parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
        "parameters_total": int(sum(p.numel() for p in model.parameters())),
        "parameter_tensors": int(sum(1 for _ in model.parameters())),
    }
    return model, info


def parameter_breakdown(model: HRNet) -> dict[str, int]:
    out: dict[str, int] = {}
    for name, module in model.named_children():
        out[name] = int(sum(p.numel() for p in module.parameters()))
    return out


def shared_state_hash(model: HRNet) -> str:
    """Hash of the modules that A and B must initialise identically."""
    names = [
        "atom_encoder", "bond_encoder", "junction_encoder", "refine", "role",
        "patch_node", "patch_edge", "support_trunk", "support_node", "support_edge",
        "patch_mlp", "reader",
    ]
    state = {}
    for name in names:
        for key, value in getattr(model, name).state_dict().items():
            state[f"{name}.{key}"] = value
    return C.sha256_state_dict(state)

"""Composer/context trunks for zinc-recipe-trunk-v2.

Three model families live here.  All of them consume exactly the v1 cached P3
atom/bond/junction objects and the v1 ``collate`` "T" batch layout, so the data
path is shared with the already reported T run.

``P3N``  Reimplementation of the frozen P3 computation graph (BJOC-Null
         composer + reader) inside the v1 harness.  It exists only so the
         recipe cross (P3 x recipe B) runs through the same trainer, optimizer
         code and data loader as T x recipe A.  The state dict of the original
         ``structural_relation_composer_v1.BJOC`` loads into it after a key
         remap, which is what the equivalence check uses.

``M``    Typed object trunk with a **global moment context** and a
         conditional per-object composition:

             c_G  = ContextMLP([sum h, mean h^2, log1p N]_type, global33)
             u_o  = t_o + Phi([t_o, c_G, t_o * W c_G])
             z_G  = [sum u, mean u^2, log1p N]_type + global33
             y    = Reader(z_G)

         with ``t_o = h_o + type_embedding`` and one shared ``Phi``.

``S``    Same typed objects, but the context is a small set of learned global
         slots built with one cross-attention layer (K=8 queries), and every
         object reads its own context back from those slots:

             slots = CrossAttend(queries, objects) + FFN
             ctx_o = CrossAttend(objects, slots)
             u_o   = t_o + Phi([t_o, ctx_o, t_o * W ctx_o])

         No object-to-object attention layer and no N^2 object graph.  All
         attention is masked to the molecule, so graphs in a batch never mix.

Neither trunk keeps P3's final prediction shortcut, neither uses an
association tensor, and neither performs any hidden-state update on the local
adjacency.  Both are fine-tuned end to end from the frozen train-only masked
SSL encoders.

The masked structural auxiliary task (``use_aux``) adds two linear CE heads on
top of the *masked* encoder branch; the heads exist only in the joint variants
and are counted in the 500k budget.
"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

_HERE = Path(__file__).resolve().parent
_V1 = _HERE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_HERE), str(_V1)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_models as M1  # noqa: E402

N_OBJ_TYPES = 3  # 0 = atom, 1 = bond, 2 = junction
ATOM_CE_CLASSES = 28  # raw ZINC atom vocabulary incl. the mask token
BOND_CE_CLASSES = 4  # raw bond vocabulary incl. the mask token


def _std(g: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    return (g - mean) / scale


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


# ---------------------------------------------------------------------------
# P3-Null reimplementation (recipe cross only)
# ---------------------------------------------------------------------------


class P3NullModel(nn.Module):
    """P3's exact computation graph: encoders -> Null composer -> reader."""

    def __init__(self, norm: Mapping[str, object], *, dropout: float = 0.0):
        super().__init__()
        self.encoders = M1.P3Encoders(dropout=dropout)
        self.group_composer = M1.MLP(96 + 2 * 96 + 1 + D.BOND_FEAT_DIM, 160, 96, 1, dropout)
        self.reader = M1.MLP(96 + 2 * 96 + D.GLOBAL_DIM, 256, 1, n_hidden=2, dropout=dropout)
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))

    def group_to_bonds(self, w: torch.Tensor, batch: Mapping[str, torch.Tensor]):
        """Null grouping: each bond receives ``n_b`` copies of the graph mean."""
        n_graphs = int(batch["n_graphs"])
        jb = batch["junc_bond"]
        n_b = torch.zeros(int(batch["bond_feat"].shape[0]), device=w.device, dtype=w.dtype)
        n_b.index_add_(0, jb.reshape(-1), torch.ones(jb.numel(), device=w.device, dtype=w.dtype))
        w_sum = w.new_zeros((n_graphs, w.shape[1])).index_add_(0, batch["junc_batch"], w)
        w2_sum = w.new_zeros((n_graphs, w.shape[1])).index_add_(0, batch["junc_batch"], w * w)
        cnt = torch.bincount(batch["junc_batch"], minlength=n_graphs).clamp(min=1).to(w.dtype)
        mean_w = w_sum / cnt.unsqueeze(1)
        mean_w2 = w2_sum / cnt.unsqueeze(1)
        bond_batch = batch["bond_batch"]
        return n_b.unsqueeze(1) * mean_w[bond_batch], n_b.unsqueeze(1) * mean_w2[bond_batch], n_b

    def load_bjoc_state(self, state: Mapping[str, torch.Tensor]) -> None:
        """Load a ``structural_relation_composer_v1.BJOC`` state dict."""
        remap = {}
        for key, value in state.items():
            if key.startswith(("atom_encoder.", "bond_encoder.", "junction_encoder.")):
                remap["encoders." + key] = value
            else:
                remap[key] = value
        self.load_state_dict(remap)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        a, b, w = self.encoders(batch)
        W, V, n_b = self.group_to_bonds(w, batch)
        compose_in = torch.cat(
            [b, W, V, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1
        )
        c = self.group_composer(compose_in)
        n_graphs = int(batch["n_graphs"])
        z = c.new_zeros((n_graphs, c.shape[1])).index_add_(0, batch["bond_batch"], c)
        a_sum = a.new_zeros((n_graphs, a.shape[1])).index_add_(0, batch["atom_batch"], a)
        a_sq = a.new_zeros((n_graphs, a.shape[1])).index_add_(0, batch["atom_batch"], a * a)
        g = _std(batch["global"], self.global_mean, self.global_scale)
        return self.reader(torch.cat([z, a_sum, a_sq, g], dim=1)).view(-1)


# ---------------------------------------------------------------------------
# shared typed trunk
# ---------------------------------------------------------------------------


class AuxTypeHeads(nn.Module):
    """Linear type-recovery heads for the masked structural auxiliary task.

    Input layout follows the frozen P3 pretext heads (atom: own encoding plus
    incident bond encodings plus junction encodings through the centre and the
    endpoints; bond: own encoding plus endpoint interactions plus the summed
    touching junction encodings).  The head width is linear instead of the P3
    hidden-160 MLP so that the auxiliary variant stays inside 500k parameters.
    """

    def __init__(self, width: int = 96):
        super().__init__()
        self.atom_head = nn.Linear(4 * width, ATOM_CE_CLASSES)
        self.bond_head = nn.Linear(5 * width, BOND_CE_CLASSES)

    @staticmethod
    def contexts(a: torch.Tensor, b: torch.Tensor, w: torch.Tensor, batch: Mapping[str, torch.Tensor]):
        bi = batch["bond_index"]
        ji = batch["junc_index"]
        dev = a.device
        n_bonds, n_junc = int(b.shape[0]), int(w.shape[0])
        bond_range = torch.arange(n_bonds, device=dev)
        junc_range = torch.arange(n_junc, device=dev)

        sum_b = a.new_zeros((int(a.shape[0]), b.shape[1]))
        sum_b.index_add_(0, torch.cat([bi[:, 0], bi[:, 1]]), torch.cat([b, b], dim=0))
        sum_wc = a.new_zeros((int(a.shape[0]), w.shape[1]))
        sum_wc.index_add_(0, ji[:, 1], w)
        sum_we = a.new_zeros((int(a.shape[0]), w.shape[1]))
        sum_we.index_add_(
            0, torch.cat([ji[:, 0], ji[:, 2]]), torch.cat([w, w], dim=0)
        )
        atom_ctx = torch.cat([a, sum_b, sum_wc, sum_we], dim=1)

        sum_wb = b.new_zeros((n_bonds, w.shape[1]))
        sum_wb.index_add_(0, batch["junc_bond"][:, 0], w)
        sum_wb.index_add_(0, batch["junc_bond"][:, 1], w)
        au, av = a[bi[:, 0]], a[bi[:, 1]]
        bond_ctx = torch.cat([b, au + av, (au - av).abs(), au * av, sum_wb], dim=1)
        del bond_range, junc_range
        return atom_ctx, bond_ctx


class TypedTrunk(nn.Module):
    """Local P3-compatible encoders + type embeddings + shared Phi + reader."""

    def __init__(
        self,
        norm: Mapping[str, object],
        *,
        width: int = 96,
        ctx_dim: int = 64,
        phi_hidden: int = 128,
        reader_hidden: int = 128,
        dropout: float = 0.0,
        use_aux: bool = False,
    ):
        super().__init__()
        self.width = int(width)
        self.ctx_dim = int(ctx_dim)
        self.encoders = M1.P3Encoders(dropout=dropout)
        self.type_emb = nn.Embedding(N_OBJ_TYPES, self.width)
        self.w_c = nn.Linear(self.ctx_dim, self.width)
        self.phi = M1.MLP(self.width + self.ctx_dim + self.width, phi_hidden, self.width, 1, dropout)
        self.reader = M1.MLP(
            N_OBJ_TYPES * (2 * self.width + 1) + D.GLOBAL_DIM,
            reader_hidden,
            1,
            n_hidden=2,
            dropout=dropout,
        )
        self.aux = AuxTypeHeads(self.width) if use_aux else None
        self.register_buffer("global_mean", torch.as_tensor(norm["global_mean"], dtype=torch.float32))
        self.register_buffer("global_scale", torch.as_tensor(norm["global_scale"], dtype=torch.float32))

    # -- shared pieces ------------------------------------------------------
    def local_tokens(self, batch: Mapping[str, torch.Tensor]):
        return self.encoders(batch)

    @staticmethod
    def typed_index(batch: Mapping[str, torch.Tensor]):
        return [batch["atom_batch"], batch["bond_batch"], batch["junc_batch"]]

    def typed_pool(self, us, idxs, n_graphs: int) -> torch.Tensor:
        parts = []
        for u, idx in zip(us, idxs):
            n = torch.bincount(idx, minlength=n_graphs).to(u.dtype)
            s = u.new_zeros((n_graphs, self.width)).index_add_(0, idx, u)
            sq = u.new_zeros((n_graphs, self.width)).index_add_(0, idx, u * u)
            parts.extend([s, sq / n.clamp(min=1).unsqueeze(1), torch.log1p(n).unsqueeze(1)])
        return torch.cat(parts, dim=1)

    def compose(self, tokens, contexts):
        out = []
        for t, (h, c) in enumerate(zip(tokens, contexts)):
            e = h + self.type_emb.weight[t]
            c96 = self.w_c(c)
            out.append(e + self.phi(torch.cat([e, c, e * c96], dim=1)))
        return out

    def read_graph(self, us, idxs, n_graphs: int, batch) -> torch.Tensor:
        g = _std(batch["global"], self.global_mean, self.global_scale)
        z = torch.cat([self.typed_pool(us, idxs, n_graphs), g], dim=1)
        return self.reader(z).view(-1)

    # -- masked structural auxiliary task -----------------------------------
    def aux_loss(self, batch, atom_ids, atom_targets, bond_ids, bond_targets):
        if self.aux is None:
            raise RuntimeError("auxiliary heads are not built on this model")
        a, b, w = self.local_tokens(batch)
        atom_ctx, bond_ctx = self.aux.contexts(a, b, w, batch)
        ce_atom = (
            F.cross_entropy(self.aux.atom_head(atom_ctx[atom_ids]), atom_targets)
            if atom_targets.numel()
            else None
        )
        ce_bond = (
            F.cross_entropy(self.aux.bond_head(bond_ctx[bond_ids]), bond_targets)
            if bond_targets.numel()
            else None
        )
        if ce_atom is None and ce_bond is None:
            raise ValueError("empty auxiliary batch")
        total = a.new_zeros(())
        parts = {}
        if ce_atom is not None:
            total = total + ce_atom
            parts["ce_atom"] = ce_atom
        if ce_bond is not None:
            total = total + ce_bond
            parts["ce_bond"] = ce_bond
        return total, parts


class MomentContextModel(TypedTrunk):
    """M: one global moment context + conditional per-object composition."""

    def __init__(self, norm, *, ctx_hidden: int = 128, **kwargs):
        super().__init__(norm, **kwargs)
        self.ctx_mlp = M1.MLP(
            N_OBJ_TYPES * (2 * self.width + 1) + D.GLOBAL_DIM,
            ctx_hidden,
            self.ctx_dim,
            1,
            0.0,
        )

    def forward(self, batch):
        n_graphs = int(batch["n_graphs"])
        tokens = self.local_tokens(batch)
        idxs = self.typed_index(batch)
        g = _std(batch["global"], self.global_mean, self.global_scale)
        c = self.ctx_mlp(torch.cat([self.typed_pool(tokens, idxs, n_graphs), g], dim=1))
        us = self.compose(tokens, [c[idx] for idx in idxs])
        return self.read_graph(us, idxs, n_graphs, batch)


class SlotContextModel(TypedTrunk):
    """S: K learned slots built by one cross-attention, read back per object."""

    def __init__(self, norm, *, n_slots: int = 8, heads: int = 4, **kwargs):
        super().__init__(norm, **kwargs)
        if self.width % heads:
            raise ValueError("width must be divisible by heads")
        self.n_slots = int(n_slots)
        self.slot_q = nn.Parameter(torch.randn(1, self.n_slots, self.width) * 0.02)
        self.slot_ln = nn.LayerNorm(self.width)
        self.slot_attn = nn.MultiheadAttention(self.width, heads, dropout=0.0, batch_first=True)
        self.slot_ffn_ln = nn.LayerNorm(self.width)
        self.slot_ffn = nn.Sequential(
            nn.Linear(self.width, 2 * self.width), nn.GELU(), nn.Linear(2 * self.width, self.width)
        )
        self.obj_ln = nn.LayerNorm(self.width)
        self.obj_attn = nn.MultiheadAttention(self.width, heads, dropout=0.0, batch_first=True)
        self.ctx_proj = nn.Linear(self.width, self.ctx_dim)

    def forward(self, batch):
        n_graphs = int(batch["n_graphs"])
        tokens = self.local_tokens(batch)
        idxs = self.typed_index(batch)
        counts = [torch.bincount(i, minlength=n_graphs) for i in idxs]
        n_max = sum(int(c.max()) for c in counts)
        dev = tokens[0].device
        pad = tokens[0].new_zeros((n_graphs, n_max, self.width))
        mask = torch.zeros((n_graphs, n_max), dtype=torch.bool, device=dev)
        pad_pos = []
        offset = 0
        for t, (h, idx) in enumerate(zip(tokens, idxs)):
            offsets = torch.cat([torch.zeros(1, dtype=idx.dtype, device=dev), counts[t].cumsum(0)[:-1]])
            local = torch.arange(idx.shape[0], device=dev) - offsets[idx]
            pos = offset + local
            pad[idx, pos] = h
            mask[idx, pos] = True
            pad_pos.append(pos)
            offset += int(counts[t].max())

        slots = self.slot_q.expand(n_graphs, self.n_slots, self.width)
        kv = self.slot_ln(pad)
        s = self.slot_attn(slots, kv, kv, key_padding_mask=~mask, need_weights=False)[0]
        s = slots + s
        s = s + self.slot_ffn(self.slot_ffn_ln(s))

        ctx_pad = self.obj_attn(self.obj_ln(pad), s, s, need_weights=False)[0]
        contexts = [self.ctx_proj(ctx_pad[idx, pos]) for idx, pos in zip(idxs, pad_pos)]
        us = self.compose(tokens, contexts)
        return self.read_graph(us, idxs, n_graphs, batch)


# ---------------------------------------------------------------------------
# builders
# ---------------------------------------------------------------------------

CANDIDATES_V2 = ("P3N", "M", "S", "Maux", "Saux")


def build_model(candidate: str, norm: Mapping[str, object], seed: int, device):
    torch.manual_seed(int(seed))
    if candidate == "P3N":
        model = P3NullModel(norm)
        info = {"candidate": candidate, "ssl": None}
    elif candidate in ("M", "Maux", "S", "Saux"):
        use_aux = candidate.endswith("aux")
        if candidate.startswith("M"):
            model = MomentContextModel(norm, use_aux=use_aux)
        else:
            model = SlotContextModel(norm, use_aux=use_aux)
        info = {"candidate": candidate, "ssl": None, "use_aux": bool(use_aux)}
    else:
        raise ValueError(f"unknown v2 candidate {candidate!r}")
    model = model.to(device)
    if candidate != "P3N":
        path = C.P3_SSL_S0 if int(seed) == 0 else C.P3_SSL_S1
        if path.exists():
            info["ssl"] = _load_encoders(model, path, device)
        else:
            info["ssl"] = {"source": None, "note": "frozen SSL encoder unavailable; random init"}
    n_params = C.parameter_count(model)
    if n_params > 500_000:
        raise RuntimeError(f"{candidate}: {n_params} parameters exceeds the 500k cap")
    info["parameters"] = n_params
    info["requires_grad_true"] = sum(1 for p in model.parameters() if p.requires_grad)
    info["parameters_total"] = sum(1 for p in model.parameters())
    return model, info

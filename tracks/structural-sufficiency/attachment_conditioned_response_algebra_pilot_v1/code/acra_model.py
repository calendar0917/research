"""ACRA-v1: attachment-conditioned response algebra.

No message passing, no GNN backbone, no global Transformer, no learnable whole
fragment-id embedding.  Every real bond produces two directed connection
events; each event reads only a local attachment site ``S`` (radius 1) and a
local context ``C`` (radius 2) through a *shared* attribute-aware frame encoder,
and a shared context-conditioned response operator turns them into one scalar
response per direction.  The molecule prediction is the explicit algebra

    yhat_std = beta + E_atom + E_bond + E_rel

with ``E_atom`` a sum of per-atom contributions, ``E_bond`` the mean of the two
directed responses of every bond, and ``E_rel`` a gated low-rank symmetric
bond-pair relation term over bond separations and shared-ring membership.

Arms only toggle the interaction term and the relation term (see
``configs/acra_v1.json``); the frame encoder, local projections and atom head are
identical and identically initialised across arms.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import acra_common as C

ARMS = ("ACRA_ADD", "ACRA_COND", "ACRA_REL")


def _mlp(sizes: Sequence[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(sizes) - 1):
        layers.append(nn.Linear(int(sizes[i]), int(sizes[i + 1])))
        if i < len(sizes) - 2:
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
    return nn.Sequential(*layers)


class FrameEncoder(nn.Module):
    """Shared attribute-aware rooted-frame encoder (output 64d)."""

    def __init__(self, atom_dims: Sequence[int], bond_dims: Sequence[int],
                 dims: Mapping[str, Any]) -> None:
        super().__init__()
        emb = int(dims["frame_field_emb"])
        self.atom_emb = nn.ModuleList([nn.Embedding(int(d), emb) for d in atom_dims])
        self.bond_emb = nn.ModuleList([nn.Embedding(int(d), emb) for d in bond_dims])
        atom_out = int(dims["frame_atom_out"])
        bond_out = int(dims["frame_bond_out"])
        self.atom_mlp = _mlp([len(atom_dims) * emb, 32, atom_out])
        self.bond_mlp = _mlp([len(bond_dims) * emb, 16, bond_out])
        slots = C.MAX_OBJECT_ATOMS
        pairs = len(C.FRAME_PAIRS)
        frame_input = slots * atom_out + pairs * bond_out + slots + pairs + slots
        if frame_input != int(dims["frame_input"]):
            raise ValueError(f"frame input {frame_input} != configured {dims['frame_input']}")
        self.frame_mlp = nn.Sequential(
            nn.Linear(frame_input, int(dims["frame_hidden"])),
            nn.SiLU(),
            nn.Linear(int(dims["frame_hidden"]), int(dims["frame_out"])),
            nn.SiLU(),
            nn.LayerNorm(int(dims["frame_out"])),
        )

    def forward(self, atom_vals: torch.Tensor, bond_vals: torch.Tensor,
                nmask: torch.Tensor, emask: torch.Tensor,
                size_oh: torch.Tensor) -> torch.Tensor:
        n = atom_vals.shape[0]
        a = torch.cat([emb(atom_vals[:, :, i]) for i, emb in enumerate(self.atom_emb)], dim=-1)
        a = self.atom_mlp(a) * nmask[:, :, None]
        b = torch.cat([emb(bond_vals[:, :, i]) for i, emb in enumerate(self.bond_emb)], dim=-1)
        b = self.bond_mlp(b) * emask[:, :, None]
        x = torch.cat([a.reshape(n, -1), b.reshape(n, -1), nmask, emask, size_oh], dim=-1)
        return self.frame_mlp(x)


class LocalProjection(nn.Module):
    """Pooled 1..4 frames + counts + local ring scalars -> 64d local state."""

    def __init__(self, dims: Mapping[str, Any]) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(int(dims["local_input"]), int(dims["local_hidden"])),
            nn.SiLU(),
            nn.Linear(int(dims["local_hidden"]), int(dims["local_out"])),
            nn.LayerNorm(int(dims["local_out"])),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class AtomHead(nn.Module):
    """Raw 9-column atom attributes -> one scalar contribution."""

    def __init__(self, atom_dims: Sequence[int], dims: Mapping[str, Any]) -> None:
        super().__init__()
        emb = int(dims["atom_field_emb"])
        self.emb = nn.ModuleList([nn.Embedding(int(d), emb) for d in atom_dims])
        self.net = _mlp([len(atom_dims) * emb, int(dims["atom_hidden"]), 1])

    def forward(self, atom_vals: torch.Tensor) -> torch.Tensor:
        x = torch.cat([emb(atom_vals[:, i]) for i, emb in enumerate(self.emb)], dim=-1)
        return self.net(x).squeeze(-1)


class ACRAModel(nn.Module):
    def __init__(self, cfg: Mapping[str, Any], atom_dims: Sequence[int],
                 bond_dims: Sequence[int], arm: str) -> None:
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown arm {arm}")
        self.arm = arm
        self.cfg = dict(cfg)
        self.use_interaction = bool(cfg["arms"][arm]["interaction"])
        self.use_relation = bool(cfg["arms"][arm]["relation"])
        dims = cfg["dims"]
        self.dims = dims
        self.hidden = int(dims["local_out"])
        self.rank = int(dims["interaction_rank"])
        self.rel_rank = int(dims["relation_rank"])
        self.dropout = float(dims["response_dropout"])

        self.frame = FrameEncoder(atom_dims, bond_dims, dims)
        self.proj_s = LocalProjection(dims)
        self.proj_c = LocalProjection(dims)

        bond_fields = len(bond_dims)
        bond_emb_dim = bond_fields * int(dims["bond_field_emb"]) + 2 * int(dims["bond_flag_emb"])
        if bond_emb_dim != int(dims["bond_emb"]):
            raise ValueError(f"bond embedding {bond_emb_dim} != configured {dims['bond_emb']}")
        self.bond_emb = nn.ModuleList(
            [nn.Embedding(int(d), int(dims["bond_field_emb"])) for d in bond_dims])
        self.flag_bridge = nn.Embedding(2, int(dims["bond_flag_emb"]))
        self.flag_ring = nn.Embedding(2, int(dims["bond_flag_emb"]))

        site_in = self.hidden + int(dims["bond_emb"])
        self.mlp_s = _mlp([site_in, int(dims["response_hidden"]),
                           int(dims["response_hidden"])], dropout=self.dropout)
        self.mlp_c = _mlp([site_in, int(dims["response_hidden"]),
                           int(dims["response_hidden"])], dropout=self.dropout)
        self.scalar_s = nn.Linear(int(dims["response_hidden"]), 1)
        self.scalar_c = nn.Linear(int(dims["response_hidden"]), 1)
        if self.use_interaction:
            self.q = nn.Linear(int(dims["response_hidden"]), self.rank)
            self.k = nn.Linear(int(dims["response_hidden"]), self.rank)
        if self.use_relation:
            self.site_head = nn.Sequential(
                nn.Linear(2 * int(dims["response_hidden"]), int(dims["site_dim"])),
                nn.SiLU(),
            )
            self.rel_q = nn.Linear(int(dims["site_dim"]), self.rel_rank)
            self.rel_k = nn.Linear(int(dims["site_dim"]), self.rel_rank)
            self.rel_bias = nn.Embedding(
                int(dims["relation_buckets"]) * int(dims["relation_shared_ring"]), 1)
            self.gate_raw = nn.Parameter(torch.zeros(1))

        self.atom_head = AtomHead(atom_dims, dims)
        self.beta = nn.Parameter(torch.zeros(1))

        C.deterministic_init(self, int(cfg["recipe"]["seed"]))
        gate = float(cfg["composition"]["gate_init"])
        with torch.no_grad():
            # neutral start: prediction 0 == Train label mean (standardised units)
            self.beta.zero_()
            for head in (self.scalar_s, self.scalar_c, self.atom_head.net[-1]):
                head.weight.zero_()
                head.bias.zero_()
            if self.use_relation:
                self.rel_bias.weight.zero_()
                self.gate_raw.fill_(math.log(gate / (1.0 - gate)))

    # -- pack buffers -------------------------------------------------------
    def load_pack(self, pack: Mapping[str, Any]) -> None:
        def reg(name: str, tensor: torch.Tensor) -> None:
            self.register_buffer(name, tensor, persistent=False)

        long_names = ("atom_ptr", "atom_feat", "ev_ptr", "ev_bond_attr", "ev_bridge",
                      "ev_in_ring", "blk_ptr", "blk_count",
                      "ent_pat", "ent_mult", "pat_atom", "pat_bond", "pair_ptr", "pair_e",
                      "pair_f", "pair_bucket", "pair_shared_ring", "mol_n_bonds")
        for name in long_names:
            reg(name, torch.as_tensor(np.ascontiguousarray(pack[name])).long())
        for name in ("ev_scalars_s", "ev_scalars_c"):
            reg(name, torch.as_tensor(np.ascontiguousarray(pack[name])).float())

        pat_k = torch.as_tensor(np.ascontiguousarray(pack["pat_k"])).long()
        pat_bits = torch.as_tensor(np.ascontiguousarray(pack["pat_bits"])).long()
        n = int(pat_k.numel())
        slots = C.MAX_OBJECT_ATOMS
        pairs = len(C.FRAME_PAIRS)
        nmask = (torch.arange(slots)[None, :] < pat_k[:, None]).float()
        emask = torch.stack([((pat_bits >> s) & 1).float() for s in range(pairs)], dim=-1)
        size_oh = torch.zeros(n, slots)
        if n:
            size_oh[torch.arange(n), pat_k - 1] = 1.0
        reg("pat_nmask", nmask)
        reg("pat_emask", emask)
        reg("pat_size_oh", size_oh)

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # -- shared response operator -------------------------------------------
    def _bond_embedding(self, batch: Mapping[str, Any]) -> torch.Tensor:
        rows = batch["ev_idx"]
        attrs = self.ev_bond_attr[rows]
        emb = torch.cat([e(attrs[:, i]) for i, e in enumerate(self.bond_emb)], dim=-1)
        emb = torch.cat([emb,
                         self.flag_bridge(self.ev_bridge[rows]),
                         self.flag_ring(self.ev_in_ring[rows])], dim=-1)
        return emb

    def _local_state(self, pool: torch.Tensor, counts: torch.Tensor,
                     scalars: torch.Tensor, rows: torch.Tensor) -> torch.Tensor:
        feat = torch.cat([pool.reshape(pool.shape[0], -1),
                          torch.log1p(counts),
                          scalars[rows]], dim=-1)
        return feat

    def forward_batch(self, batch: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        n_mol = int(batch["n_molecules"])
        device = self.beta.device
        n_ent = int(batch["n_entries"])

        # 1. shared frame encoder on every stored frame entry
        if n_ent:
            atom_vals = self.pat_atom[batch["ent_pat"]]
            bond_vals = self.pat_bond[batch["ent_pat"]]
            phi = self.frame(atom_vals, bond_vals, self.pat_nmask[batch["ent_pat"]],
                             self.pat_emask[batch["ent_pat"]],
                             self.pat_size_oh[batch["ent_pat"]])
        else:
            phi = torch.zeros((0, int(self.dims["frame_out"])), device=device)
        blk_count = batch["blk_count_batch"][batch["blk_local"]].float().clamp(min=1.0)
        weight = batch["ent_mult"].float() / blk_count
        n_blk = int(batch["n_blocks"])
        pooled = torch.zeros((n_blk, phi.shape[1]), device=device, dtype=phi.dtype)
        if n_ent:
            pooled.index_add_(0, batch["blk_local"], phi * weight[:, None])
        pooled = pooled.view(-1, 8, phi.shape[1])
        counts = batch["blk_count_batch"].float().view(-1, 8)
        s_pool, c_pool = pooled[:, :4], pooled[:, 4:]
        s_counts, c_counts = counts[:, :4], counts[:, 4:]
        feat_s = self._local_state(s_pool, s_counts, self.ev_scalars_s, batch["ev_idx"])
        feat_c = self._local_state(c_pool, c_counts, self.ev_scalars_c, batch["ev_idx"])
        z_s = self.proj_s(feat_s)
        z_c = self.proj_c(feat_c)

        # 2. shared context-conditioned response operator
        bond_emb = self._bond_embedding(batch)
        st = self.mlp_s(torch.cat([z_s, bond_emb], dim=-1))
        ct = self.mlp_c(torch.cat([z_c, bond_emb], dim=-1))
        interaction = torch.zeros(st.shape[0], device=device, dtype=st.dtype)
        if self.use_interaction:
            q = self.q(st)
            k = self.k(ct)
            interaction = (q * k).sum(-1) / math.sqrt(self.rank)
        scalar_s = self.scalar_s(st).squeeze(-1)
        scalar_c = self.scalar_c(ct).squeeze(-1)
        r = scalar_s + scalar_c + interaction

        # 3. atom term
        atom_vals = self.atom_feat[batch["atom_idx"]]
        atom_contrib = self.atom_head(atom_vals)
        e_atom = torch.zeros(n_mol, device=device).index_add_(0, batch["atom_mol"], atom_contrib)

        # 4. bond term (two directions averaged)
        e_bond = torch.zeros(n_mol, device=device).index_add_(
            0, batch["ev_mol"], 0.5 * r)

        # 5. gated symmetric bond-pair relation term
        e_rel = torch.zeros(n_mol, device=device)
        pair_value = torch.zeros(0, device=device)
        gate = torch.zeros((), device=device)
        if self.use_relation and int(batch["n_pairs"]) > 0:
            site = self.site_head(torch.cat([st, ct], dim=-1))
            bond_h = site.view(-1, 2, site.shape[-1]).mean(dim=1)
            qe = self.rel_q(bond_h)
            ke = self.rel_k(bond_h)
            pe, pf = batch["pair_e"], batch["pair_f"]
            dots = 0.5 * ((qe[pe] * ke[pf]).sum(-1) + (qe[pf] * ke[pe]).sum(-1))
            # cache buckets are 1,2,3,4+ -> embedding rows 0..3
            bias_idx = ((self.pair_bucket[batch["pair_idx"]].long() - 1) * 2
                        + self.pair_shared_ring[batch["pair_idx"]].long())
            pair_value = dots / math.sqrt(self.rel_rank) + self.rel_bias(bias_idx).squeeze(-1)
            total = torch.zeros(n_mol, device=device).index_add_(
                0, batch["pair_mol"], pair_value)
            gate = torch.sigmoid(self.gate_raw)
            e_rel = gate * total / batch["n_bonds"].float().clamp(min=1.0)

        yhat_std = self.beta + e_atom + e_bond + e_rel
        return {
            "yhat_std": yhat_std,
            "e_atom": e_atom,
            "e_bond": e_bond,
            "e_rel": e_rel,
            "beta": self.beta.expand(n_mol),
            "atoms": atom_contrib,
            "atom_mol": batch["atom_mol"],
            "r": r,
            "scalar_s": scalar_s,
            "scalar_c": scalar_c,
            "interaction": interaction,
            "ev_mol": batch["ev_mol"],
            "pair_value": pair_value,
            "gate": gate,
        }

    def loss(self, out: Mapping[str, torch.Tensor], batch: Mapping[str, Any],
             y_std: torch.Tensor) -> dict[str, torch.Tensor]:
        task = torch.mean((out["yhat_std"] - y_std) ** 2)
        n_mol = int(batch["n_molecules"])
        site_sq = torch.cat([out["atoms"] ** 2, out["r"] ** 2])
        owner = torch.cat([batch["atom_mol"], batch["ev_mol"]])
        per_mol = torch.zeros(n_mol, device=task.device).index_add_(0, owner, site_sq)
        n_sites = (batch["n_atoms_per_mol"] + 2 * batch["n_bonds"]).float()
        site = torch.mean(per_mol / n_sites.clamp(min=1.0))
        rel = torch.mean(out["e_rel"] ** 2)
        total = (task
                 + float(self.cfg["composition"]["site_loss_weight"]) * site
                 + float(self.cfg["composition"]["relation_loss_weight"]) * rel)
        return {"loss": total, "task": task, "site": site, "rel": rel}

    # -- diagnostics --------------------------------------------------------
    def parameter_ledger(self) -> dict[str, int]:
        groups: dict[str, int] = {}
        for name, param in self.named_parameters():
            top = name.split(".")[0]
            groups[top] = groups.get(top, 0) + int(param.numel())
        groups["total"] = sum(int(p.numel()) for p in self.parameters())
        return groups


# ---------------------------------------------------------------------------
# support masking (Train categories only)
# ---------------------------------------------------------------------------


def category_support(raw: Mapping[str, Any], train_rows: Sequence[int],
                     atom_dims: Sequence[int], bond_dims: Sequence[int]):
    """Per-field category masks over the *current* training molecules."""
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    atom_feat = np.asarray(raw["atom_feat"])
    bond_feat = np.asarray(raw["bond_feat"])
    atom_ok = [np.zeros(int(d), bool) for d in atom_dims]
    bond_ok = [np.zeros(int(d), bool) for d in bond_dims]
    for row in np.asarray(train_rows).tolist():
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        for i in range(atom_feat.shape[1]):
            atom_ok[i][np.unique(atom_feat[a0:a1, i].astype(np.int64))] = True
        for i in range(bond_feat.shape[1]):
            bond_ok[i][np.unique(bond_feat[b0:b1, i].astype(np.int64))] = True
    return atom_ok, bond_ok


def apply_support(model: ACRAModel, atom_support: Sequence[np.ndarray],
                  bond_support: Sequence[np.ndarray]) -> dict[str, int]:
    zeroed: dict[str, int] = {}

    def zero(module_list, support, tag: str) -> None:
        for i, emb in enumerate(module_list):
            mask = torch.as_tensor(np.asarray(support[i], bool), device=emb.weight.device)
            if mask.numel() != emb.weight.shape[0]:
                raise ValueError(f"{tag} support width mismatch")
            zeroed[f"{tag}_field_{i}"] = int((~mask).sum().item())
            with torch.no_grad():
                emb.weight[~mask] = 0.0

    zero(model.frame.atom_emb, atom_support, "frame_atom")
    zero(model.frame.bond_emb, bond_support, "frame_bond")
    zero(model.atom_head.emb, atom_support, "atom_head")
    zero(model.bond_emb, bond_support, "response_bond")
    return zeroed


# ---------------------------------------------------------------------------
# prediction helper
# ---------------------------------------------------------------------------


def predict_rows(model: ACRAModel, source: Any, rows: Sequence[int],
                 device: Any, batch_size: int = 128) -> dict[str, np.ndarray]:
    """Molecule-level outputs for ``rows`` (no labels involved)."""
    model.eval()
    keys = ("yhat_std", "e_atom", "e_bond", "e_rel")
    out: dict[str, list[np.ndarray]] = {k: [] for k in keys}
    out["beta"] = []
    rows = np.asarray(rows, np.int64)
    with torch.no_grad():
        for start in range(0, rows.size, int(batch_size)):
            part = rows[start:start + int(batch_size)]
            batch = source.make(part)
            res = model.forward_batch(batch)
            for k in keys:
                out[k].append(res[k].detach().double().cpu().numpy())
            out["beta"].append(res["beta"].detach().double().cpu().numpy())
    return {k: np.concatenate(v) if v else np.zeros(0) for k, v in out.items()}
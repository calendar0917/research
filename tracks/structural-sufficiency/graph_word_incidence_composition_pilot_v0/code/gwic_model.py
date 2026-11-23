"""GWIC-v0 model: radius-2 sentence trunk, learned connected word selection,
and one-shot word/incidence-bond interaction.

Arms differ only through two runtime flags:

* ``use_words=False`` -> SENTENCE_BASE (h_v and raw atom/bond summaries only)
* ``use_words=True,  use_geometry=False`` -> WORD_GEOM_NULL
* ``use_words=True,  use_geometry=True``  -> WORD_REAL

WORD_GEOM_NULL and WORD_REAL instantiate exactly the same modules and the same
parameter count.  At inference either arm's geometry inputs can be switched
without changing the checkpoint.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

import gwic_common as C

ARMS = ("SENTENCE_BASE", "WORD_GEOM_NULL", "WORD_REAL")


def _mlp(dims: Sequence[int], final_silu: bool = True, dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(int(dims[i]), int(dims[i + 1])))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
    if final_silu:
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def _seg_sum(values: torch.Tensor, seg: torch.Tensor, n: int, dim: int) -> torch.Tensor:
    out = values.new_zeros((int(n), int(dim)))
    if values.numel():
        out.index_add_(0, seg, values)
    return out


def _seg_stats(values: torch.Tensor, seg: torch.Tensor, n: int
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    d = int(values.shape[1])
    total = _seg_sum(values, seg, n, d)
    sq = _seg_sum(values * values, seg, n, d)
    count = values.new_zeros(int(n))
    if values.numel():
        count.index_add_(0, seg, torch.ones(values.shape[0], device=values.device,
                                            dtype=values.dtype))
    denom = count.clamp(min=1.0)
    mean = total / denom[:, None]
    var = (sq / denom[:, None] - mean * mean).clamp(min=0.0)
    std = torch.sqrt(var + 1e-8)
    std = torch.where((count > 1)[:, None], std, torch.zeros_like(std))
    return total, mean, std, count


class GwicModel(nn.Module):
    def __init__(self, atom_dims: Sequence[int], bond_dims: Sequence[int],
                 atom_dim: int = 48, bond_dim: int = 24, h_dim: int = 96,
                 word_dim: int = 64, q_dim: int = 64, rank: int = 24,
                 end_dim: int = 16, close_dim: int = 8, nbr_dim: int = 16,
                 backbone_hidden: int = 192, word_hidden: int = 160,
                 score_hidden: int = 96, q_hidden: int = 128,
                 readout_hidden: int = 192, readout_mid: int = 96,
                 dropout: float = 0.1, use_words: bool = True) -> None:
        super().__init__()
        self.use_words = bool(use_words)
        self.atom_dim = int(atom_dim)
        self.bond_dim = int(bond_dim)
        self.h_dim = int(h_dim)
        self.word_dim = int(word_dim)
        self.q_dim = int(q_dim)
        self.rank = int(rank)
        self.end_dim = int(end_dim)
        self.nbr_dim = int(nbr_dim)
        self.n_atom_types = int(atom_dims[0])
        self.n_bond_types = int(bond_dims[0])
        self.n_path_codes = self.n_atom_types * self.n_bond_types * self.n_bond_types

        # OGB categorical embeddings (full official vocabulary, fit-only support mask)
        self.atom_emb = nn.ModuleList([nn.Embedding(int(d), self.atom_dim) for d in atom_dims])
        self.bond_raw_emb = nn.ModuleList([nn.Embedding(int(d), self.bond_dim) for d in bond_dims])
        self.atom_norm = nn.LayerNorm(self.atom_dim)
        self.bond_encoder = _mlp([3 * self.atom_dim + self.bond_dim, 96, self.bond_dim])

        # LCRC-style rooted local content: S1, typed S2 paths, and R2 summaries
        self.nbr_emb = nn.Embedding(self.n_atom_types * self.n_bond_types, self.nbr_dim)
        self.path_u = nn.Embedding(self.n_path_codes, self.rank)
        self.path_v = nn.Embedding(self.n_atom_types, self.rank)
        self.path_joint = nn.Linear(self.rank, self.rank)
        self.path_emb = nn.Embedding(self.n_path_codes, self.rank)
        self.end_emb = nn.Embedding(self.n_atom_types, self.end_dim)
        self.close_emb = nn.Embedding(2, close_dim)
        self.path_gate = nn.Embedding(self.n_path_codes, 1)
        nn.init.ones_(self.path_gate.weight)

        self.backbone_in_dim = (self.atom_dim + 2 * self.bond_dim + self.nbr_dim
                                + (2 * self.rank + self.end_dim + close_dim)
                                + self.atom_dim
                                + 2 * self.bond_dim + 3 * self.bond_dim
                                + 4 * self.atom_dim + 5 + 3)
        self.backbone_mlp = _mlp([self.backbone_in_dim, backbone_hidden, self.h_dim])
        self.backbone_norm = nn.LayerNorm(self.h_dim)

        # candidate-word encoder and selector (shared over every root / molecule).
        # SENTENCE_BASE does not instantiate or train these modules.
        self.word_dim = int(word_dim)
        self.q_dim = int(q_dim)
        if self.use_words:
            self.word_joint = _mlp([3 * self.atom_dim + self.bond_dim, 96, 48])
            self.word_in_dim = (self.atom_dim + 3 * self.atom_dim + 3 * self.bond_dim
                                + 2 * 48 + 25)
            self.word_mlp = _mlp([self.word_in_dim, word_hidden, self.word_dim])
            self.word_norm = nn.LayerNorm(self.word_dim)
            self.score_mlp = _mlp([self.word_dim + self.h_dim, score_hidden, 1])
            self.q_mlp = _mlp([3 * self.h_dim + 3 * self.word_dim + self.bond_dim + 3,
                               q_hidden, self.q_dim])
        else:
            self.word_in_dim = 0

        public_a = 3 * self.atom_dim
        public_b = 3 * self.bond_dim
        word_readout = (3 * self.h_dim + 2 * self.word_dim + 2 * self.q_dim + 8)
        base_readout = (3 * self.h_dim + 2 * self.atom_dim + 2 * self.bond_dim + 8)
        self.readout_in_dim = int(word_readout if self.use_words else base_readout)
        self.readout_mlp = nn.Sequential(
            nn.Linear(self.readout_in_dim, readout_hidden), nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(readout_hidden, readout_mid), nn.SiLU(),
            nn.Linear(readout_mid, 1),
        )
        self.beta = nn.Parameter(torch.zeros(()))
        self.public_a = int(public_a)
        self.public_b = int(public_b)

    # -- deterministic initialisation ---------------------------------------
    def init_parameters(self, seed: int, last_std: float = 1e-3) -> dict[str, str]:
        gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1000003 + 29)
        with torch.no_grad():
            for emb in list(self.atom_emb) + list(self.bond_raw_emb):
                emb.weight.normal_(0.0, 0.1, generator=gen)
            for emb in (self.nbr_emb, self.path_u, self.path_v, self.path_emb,
                        self.end_emb, self.close_emb):
                emb.weight.normal_(0.0, 0.1, generator=gen)
            self.path_gate.weight.fill_(1.0)
            modules = [self.bond_encoder, self.path_joint, self.backbone_mlp, self.readout_mlp]
            if self.use_words:
                modules += [self.word_joint, self.word_mlp, self.score_mlp, self.q_mlp]
            for module in modules:
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        bound = 1.0 / math.sqrt(layer.in_features)
                        layer.weight.uniform_(-bound, bound, generator=gen)
                        if layer.bias is not None:
                            layer.bias.uniform_(-bound, bound, generator=gen)
            last = [m for m in self.readout_mlp.modules()
                    if isinstance(m, nn.Linear) and m.out_features == 1][-1]
            last.weight.normal_(0.0, float(last_std), generator=gen)
            last.bias.zero_()
            self.beta.zero_()
        return self.shared_hashes()

    def shared_hashes(self) -> dict[str, str]:
        mods: dict[str, nn.Module] = {
            "atom_emb": self.atom_emb,
            "bond_raw_emb": self.bond_raw_emb,
            "atom_norm": self.atom_norm,
            "bond_encoder": self.bond_encoder,
            "nbr_emb": self.nbr_emb,
            "path_u": self.path_u,
            "path_v": self.path_v,
            "path_joint": self.path_joint,
            "path_emb": self.path_emb,
            "end_emb": self.end_emb,
            "close_emb": self.close_emb,
            "path_gate": self.path_gate,
            "backbone_mlp": self.backbone_mlp,
            "backbone_norm": self.backbone_norm,
            "readout_mlp": self.readout_mlp,
            "beta": self.beta,
        }
        if self.use_words:
            mods.update({"word_joint": self.word_joint, "word_mlp": self.word_mlp,
                         "word_norm": self.word_norm, "score_mlp": self.score_mlp,
                         "q_mlp": self.q_mlp})
        return C.module_hashes(mods)

    def parameter_ledger(self) -> dict[str, int]:
        def n(module: nn.Module) -> int:
            if isinstance(module, nn.Parameter):
                return int(module.numel())
            return int(sum(p.numel() for p in module.parameters()))

        out = {
            "atom_emb": n(self.atom_emb),
            "bond_raw_emb": n(self.bond_raw_emb),
            "atom_norm": n(self.atom_norm),
            "bond_encoder": n(self.bond_encoder),
            "nbr_emb": n(self.nbr_emb),
            "path_u": n(self.path_u),
            "path_v": n(self.path_v),
            "path_joint": n(self.path_joint),
            "path_emb": n(self.path_emb),
            "end_emb": n(self.end_emb),
            "close_emb": n(self.close_emb),
            "path_gate": n(self.path_gate),
            "backbone_mlp": n(self.backbone_mlp),
            "backbone_norm": n(self.backbone_norm),
            "readout_mlp": n(self.readout_mlp),
            "beta": n(self.beta),
        }
        if self.use_words:
            out.update({"word_joint": n(self.word_joint), "word_mlp": n(self.word_mlp),
                        "word_norm": n(self.word_norm), "score_mlp": n(self.score_mlp),
                        "q_mlp": n(self.q_mlp)})
        out["total"] = int(sum(out.values()))
        return out

    # -- fit-only category support ------------------------------------------
    def zero_unseen(self, atom_ok: Sequence[np.ndarray], bond_ok: Sequence[np.ndarray],
                    atom_dims: Sequence[int], bond_dims: Sequence[int]) -> dict[str, int]:
        zeroed: dict[str, int] = {}
        atom_mask = torch.as_tensor(np.asarray(atom_ok[0], bool), device=self.path_gate.weight.device)
        bond_masks = [torch.as_tensor(np.asarray(x, bool), device=self.path_gate.weight.device)
                      for x in bond_ok]

        def zero_emb(emb: nn.Embedding, mask: torch.Tensor, tag: str) -> None:
            if mask.numel() != emb.weight.shape[0]:
                raise ValueError(f"{tag} support width mismatch")
            zeroed[tag] = int((~mask).sum().item())
            with torch.no_grad():
                emb.weight[~mask] = 0.0

        for i, emb in enumerate(self.atom_emb):
            m = torch.as_tensor(np.asarray(atom_ok[i], bool), device=emb.weight.device)
            zeroed[f"atom_field_{i}"] = int((~m).sum().item())
            with torch.no_grad():
                emb.weight[~m] = 0.0
        for i, emb in enumerate(self.bond_raw_emb):
            m = torch.as_tensor(np.asarray(bond_ok[i], bool), device=emb.weight.device)
            zeroed[f"bond_field_{i}"] = int((~m).sum().item())
            with torch.no_grad():
                emb.weight[~m] = 0.0

        # composite vocabularies: a code is supported iff all its components are
        mid = atom_mask
        b1 = bond_masks[0]
        b2 = bond_masks[0]
        code_ok = torch.zeros(self.n_path_codes, dtype=torch.bool, device=mid.device)
        idx = torch.arange(self.n_path_codes, device=mid.device)
        cc_mid = idx // (self.n_bond_types * self.n_bond_types)
        cc_b1 = (idx // self.n_bond_types) % self.n_bond_types
        cc_b2 = idx % self.n_bond_types
        code_ok = mid[cc_mid] & b1[cc_b1] & b2[cc_b2]
        for name, emb in (("path_u", self.path_u), ("path_emb", self.path_emb),
                          ("path_gate", self.path_gate)):
            zeroed[name] = int((~code_ok).sum().item())
            with torch.no_grad():
                emb.weight[~code_ok] = 0.0
        nbr_ok = torch.zeros(self.n_atom_types * self.n_bond_types, dtype=torch.bool,
                             device=mid.device)
        nidx = torch.arange(self.n_atom_types * self.n_bond_types, device=mid.device)
        nbr_ok = mid[nidx // self.n_bond_types] & b1[nidx % self.n_bond_types]
        zeroed["nbr_emb"] = int((~nbr_ok).sum().item())
        with torch.no_grad():
            self.nbr_emb.weight[~nbr_ok] = 0.0
        zeroed["end_emb"] = int((~mid).sum().item())
        zeroed["path_v"] = int((~mid).sum().item())
        with torch.no_grad():
            self.end_emb.weight[~mid] = 0.0
            self.path_v.weight[~mid] = 0.0
        return zeroed

    # -- encoders -----------------------------------------------------------
    def embed_atoms(self, atom_feat: torch.Tensor) -> torch.Tensor:
        out = sum(emb(atom_feat[:, i]) for i, emb in enumerate(self.atom_emb))
        return self.atom_norm(out / math.sqrt(len(self.atom_emb)))

    def embed_bonds_raw(self, bond_feat: torch.Tensor) -> torch.Tensor:
        return sum(emb(bond_feat[:, i]) for i, emb in enumerate(self.bond_raw_emb)) / math.sqrt(len(self.bond_raw_emb))

    def encode_bonds(self, a: torch.Tensor, bond_index: torch.Tensor,
                     bond_raw: torch.Tensor) -> torch.Tensor:
        if bond_index.numel() == 0:
            return a.new_zeros((0, self.bond_dim))
        au, av = a[bond_index[:, 0]], a[bond_index[:, 1]]
        return self.bond_encoder(torch.cat([au + av, (au - av).abs(), au * av, bond_raw], dim=1))

    def encode_h(self, batch: Mapping[str, torch.Tensor], a: torch.Tensor,
                 b: torch.Tensor) -> torch.Tensor:
        n_atoms = int(a.shape[0])
        idx = batch["bond_index"]
        if idx.numel():
            u, v = idx[:, 0], idx[:, 1]
            deg = a.new_zeros(n_atoms)
            deg = deg.index_add_(0, u, torch.ones_like(u, dtype=a.dtype))
            deg = deg.index_add_(0, v, torch.ones_like(v, dtype=a.dtype))
            bsum = a.new_zeros((n_atoms, self.bond_dim))
            bsum.index_add_(0, u, b)
            bsum.index_add_(0, v, b)
        else:
            deg = a.new_zeros(n_atoms)
            bsum = a.new_zeros((n_atoms, self.bond_dim))
        bmean = bsum / deg.clamp(min=1.0)[:, None]

        # S1 joint neighbour code: (neighbour atom type, bond type)
        if idx.numel():
            atom_type = batch["atom_feat"][:, 0]
            btype = batch["bond_feat"][:, 0]
            u, v = idx[:, 0], idx[:, 1]
            code_u = atom_type[v] * self.n_bond_types + btype
            code_v = atom_type[u] * self.n_bond_types + btype
            nbr_code = torch.cat([code_u, code_v], dim=0)
            nbr_seg = torch.cat([u, v], dim=0)
            nbr_sum = _seg_sum(self.nbr_emb(nbr_code), nbr_seg, n_atoms, self.nbr_dim)
        else:
            nbr_sum = a.new_zeros((n_atoms, self.nbr_dim))

        # S2 typed rooted 2-paths
        n_paths = int(batch["path_cat"].numel())
        if n_paths:
            cat = batch["path_cat"]
            end_type = batch["path_end_type"]
            joint = self.path_joint(self.path_u(cat) * self.path_v(end_type))
            path_feat = torch.cat([
                self.path_emb(cat), self.end_emb(end_type), joint,
                self.close_emb(batch["path_closure"]),
            ], dim=1)
            path_sum = _seg_sum(path_feat, batch["path_seg"], n_atoms, path_feat.shape[1])
            content = self.path_gate(cat) * a[batch["path_end_atom"]]
            content_sum = _seg_sum(content, batch["path_seg"], n_atoms, self.atom_dim)
        else:
            path_sum = a.new_zeros((n_atoms, 2 * self.rank + self.end_dim + 8))
            content_sum = a.new_zeros((n_atoms, self.atom_dim))

        # complete R2(v) induced bond summaries, including non-root-incident bonds
        if batch["r2_bond"].numel():
            r2b = b[batch["r2_bond"]]
            r2_sum = _seg_sum(r2b, batch["r2_seg"], n_atoms, self.bond_dim)
            r2_count = a.new_zeros(n_atoms)
            r2_count.index_add_(0, batch["r2_seg"], torch.ones_like(batch["r2_seg"], dtype=a.dtype))
            r2_mean = r2_sum / r2_count.clamp(min=1.0)[:, None]
            bucket = _seg_sum(r2b, batch["r2_bucket_seg"], n_atoms * 3, self.bond_dim)
        else:
            r2_sum = a.new_zeros((n_atoms, self.bond_dim))
            r2_mean = a.new_zeros((n_atoms, self.bond_dim))
            r2_count = a.new_zeros(n_atoms)
            bucket = a.new_zeros((n_atoms * 3, self.bond_dim))
        bucket = bucket.view(n_atoms, 3 * self.bond_dim)

        # complete R2(v) atom summaries at exact distance 1 and 2
        d1_sum = _seg_sum(a[batch["dist1_atoms"]], batch["dist1_seg"], n_atoms, self.atom_dim) \
            if batch["dist1_atoms"].numel() else a.new_zeros((n_atoms, self.atom_dim))
        d2_sum = _seg_sum(a[batch["dist2_atoms"]], batch["dist2_seg"], n_atoms, self.atom_dim) \
            if batch["dist2_atoms"].numel() else a.new_zeros((n_atoms, self.atom_dim))
        d1_count = a.new_zeros(n_atoms)
        d2_count = a.new_zeros(n_atoms)
        if batch["dist1_atoms"].numel():
            d1_count.index_add_(0, batch["dist1_seg"], torch.ones_like(batch["dist1_seg"], dtype=a.dtype))
        if batch["dist2_atoms"].numel():
            d2_count.index_add_(0, batch["dist2_seg"], torch.ones_like(batch["dist2_seg"], dtype=a.dtype))
        d1_mean = d1_sum / d1_count.clamp(min=1.0)[:, None]
        d2_mean = d2_sum / d2_count.clamp(min=1.0)[:, None]

        scal = (batch["atom_scalars"] - self.atom_scalar_mean) / self.atom_scalar_scale
        n_r2_atoms = 1.0 + d1_count + d2_count
        r2_scal = torch.stack([
            torch.log1p(n_r2_atoms), torch.log1p(r2_count),
            (r2_count - n_r2_atoms + 1.0),
        ], dim=1)

        x = torch.cat([a, bsum, bmean, nbr_sum, path_sum, content_sum,
                       r2_sum, r2_mean, bucket, d1_sum, d1_mean, d2_sum, d2_mean,
                       scal, r2_scal], dim=1)
        if x.shape[1] != self.backbone_in_dim:
            raise ValueError(f"backbone input width {x.shape[1]} != {self.backbone_in_dim}")
        return self.backbone_norm(self.backbone_mlp(x))

    def encode_words(self, batch: Mapping[str, torch.Tensor], a: torch.Tensor,
                     b: torch.Tensor, h: torch.Tensor,
                     bond_raw: torch.Tensor,
                     geometry_override: torch.Tensor | None = None) -> dict[str, torch.Tensor]:
        support = batch["word_support"]
        internal = batch["word_internal"]
        n_words = int(support.shape[0])
        if n_words == 0:
            zero_w = a.new_zeros((0, self.word_dim))
            return {"w": zero_w, "alpha": a.new_zeros(0), "z": a.new_zeros((a.shape[0], self.word_dim)),
                    "coverage": a.new_zeros((a.shape[0], a.shape[0])),
                    "geom": a.new_zeros((0, 3)), "scores": a.new_zeros(0)}
        valid_a = support >= 0
        safe_a = support.clamp(min=0)
        slots_a = a[safe_a] * valid_a.unsqueeze(-1).to(a.dtype)
        size = valid_a.sum(dim=1).clamp(min=1).to(a.dtype)
        a_sum = slots_a.sum(dim=1)
        a_mean = a_sum / size[:, None]
        a_sq = (slots_a * slots_a).sum(dim=1) / size[:, None]
        a_std = torch.sqrt((a_sq - a_mean * a_mean).clamp(min=0.0) + 1e-8)

        valid_b = internal >= 0
        safe_b = internal.clamp(min=0)
        slots_b = b[safe_b] * valid_b.unsqueeze(-1).to(b.dtype)
        bsize = valid_b.sum(dim=1).clamp(min=1).to(b.dtype)
        b_sum = slots_b.sum(dim=1)
        b_mean = b_sum / bsize[:, None]
        b_sq = (slots_b * slots_b).sum(dim=1) / bsize[:, None]
        b_std = torch.sqrt((b_sq - b_mean * b_mean).clamp(min=0.0) + 1e-8)

        # shared joint representation per real bond, then aggregate per word
        idx = batch["bond_index"]
        if idx.numel():
            au, av = a[idx[:, 0]], a[idx[:, 1]]
            joint = self.word_joint(torch.cat([au + av, (au - av).abs(), au * av, bond_raw], dim=1))
            slots_j = joint[safe_b] * valid_b.unsqueeze(-1).to(joint.dtype)
            j_sum = slots_j.sum(dim=1)
            j_mean = j_sum / bsize[:, None]
        else:
            j_sum = a.new_zeros((n_words, 48))
            j_mean = a.new_zeros((n_words, 48))

        root_h = h[batch["word_root"]]
        root_a = a[batch["word_root"]]
        w_in = torch.cat([root_a, a_sum, a_mean, a_std, b_sum, b_mean, b_std,
                          j_sum, j_mean, batch["word_scalars"]], dim=1)
        if w_in.shape[1] != self.word_in_dim:
            raise ValueError(f"word input width {w_in.shape[1]} != {self.word_in_dim}")
        w = self.word_norm(self.word_mlp(w_in))

        scores = self.score_mlp(torch.cat([w, root_h], dim=1)).view(-1)
        n_atoms = int(a.shape[0])
        seg_max = a.new_full((n_atoms,), -1e30)
        seg_max.scatter_reduce_(0, batch["word_root"], scores, reduce="amax", include_self=True)
        e = torch.exp(scores - seg_max[batch["word_root"]])
        denom = a.new_zeros(n_atoms)
        denom.index_add_(0, batch["word_root"], e)
        alpha = e / denom[batch["word_root"]].clamp(min=1.0)
        z = a.new_zeros((n_atoms, self.word_dim))
        z.index_add_(0, batch["word_root"], alpha[:, None] * w)

        # soft coverage c_(v,t) = sum_{S: t in S} alpha_(v,S)
        root_rep = batch["word_root"][:, None].expand_as(safe_a)[valid_a]
        atom_rep = safe_a[valid_a]
        alpha_rep = alpha[:, None].expand_as(safe_a)[valid_a]
        flat = root_rep * n_atoms + atom_rep
        coverage = a.new_zeros(n_atoms * n_atoms)
        coverage.index_add_(0, flat, alpha_rep)
        coverage = coverage.view(n_atoms, n_atoms)

        if idx.numel():
            u, v = idx[:, 0], idx[:, 1]
            geom = torch.stack([
                (coverage[u] * coverage[v]).sum(dim=1),
                coverage[u, v] + coverage[v, u],
                (coverage[u, v] - coverage[v, u]).abs(),
            ], dim=1)
        else:
            geom = a.new_zeros((0, 3))
        if geometry_override is not None:
            geom = geometry_override
        return {"w": w, "alpha": alpha, "z": z, "coverage": coverage,
                "geom": geom, "scores": scores}

    # -- forward ------------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor], *, arm: str,
                use_geometry: bool | None = None,
                geometry_override: torch.Tensor | None = None,
                return_aux: bool = False) -> dict[str, Any]:
        if arm not in ARMS:
            raise KeyError(f"unknown arm {arm!r}")
        use_words = arm != "SENTENCE_BASE"
        if not use_words and self.use_words:
            raise ValueError("checkpoint was built with words but arm is SENTENCE_BASE")
        if use_words and not self.use_words:
            raise ValueError("checkpoint was built without words")
        if use_geometry is None:
            use_geometry = arm == "WORD_REAL"
        if not use_words and use_geometry:
            raise ValueError("SENTENCE_BASE has no word geometry")

        a = self.embed_atoms(batch["atom_feat"])
        bond_raw = self.embed_bonds_raw(batch["bond_feat"])
        b = self.encode_bonds(a, batch["bond_index"], bond_raw)
        h = self.encode_h(batch, a, b)

        n_atoms = int(a.shape[0])
        n_mol = int(batch["n_molecules"])
        # graph scalar standardisation uses fit-only statistics
        g = (batch["graph_scalars"] - self.graph_mean) / self.graph_scale
        h_sum, h_mean, h_std, atom_count = _seg_stats(h, batch["atom_mol"], n_mol)

        aux: dict[str, Any] = {"a": a, "b": b, "h": h, "g": g}
        if use_words:
            word = self.encode_words(batch, a, b, h, bond_raw,
                                     geometry_override=geometry_override)
            w, alpha, z, geom = word["w"], word["alpha"], word["z"], word["geom"]
            if not use_geometry and geometry_override is None:
                geom = geom.new_zeros(geom.shape)
            idx = batch["bond_index"]
            if idx.numel():
                u, v = idx[:, 0], idx[:, 1]
                hu, hv = h[u], h[v]
                zu, zv = z[u], z[v]
                q_in = torch.cat([
                    hu + hv, (hu - hv).abs(), hu * hv,
                    zu + zv, (zu - zv).abs(), zu * zv,
                    b, geom,
                ], dim=1)
                q = self.q_mlp(q_in)
            else:
                q = a.new_zeros((0, self.q_dim))
            z_sum, z_mean, z_std, _ = _seg_stats(z, batch["atom_mol"], n_mol)
            q_sum, q_mean, q_std, _ = _seg_stats(q, batch["bond_mol"], n_mol)
            readout_in = torch.cat([h_sum, h_mean, h_std, z_mean, z_std,
                                    q_mean, q_std, g], dim=1)
            aux.update({"word": word, "z": z, "q": q, "alpha": alpha, "w": w,
                        "geom": geom})
        else:
            a_sum, a_mean, a_std, _ = _seg_stats(a, batch["atom_mol"], n_mol)
            b_sum, b_mean, b_std, _ = _seg_stats(b, batch["bond_mol"], n_mol)
            readout_in = torch.cat([h_sum, h_mean, h_std, a_mean, a_std,
                                    b_mean, b_std, g], dim=1)
        if readout_in.shape[1] != self.readout_in_dim:
            raise ValueError(f"readout width {readout_in.shape[1]} != {self.readout_in_dim}")
        yhat = self.readout_mlp(readout_in).view(-1) + self.beta
        if return_aux:
            aux.update({"yhat_std": yhat, "readout_in": readout_in,
                        "n_atoms": n_atoms, "n_molecules": n_mol})
            return aux
        return {"yhat_std": yhat}

    # buffers are registered after construction in build_model
    atom_scalar_mean: torch.Tensor
    atom_scalar_scale: torch.Tensor
    graph_mean: torch.Tensor
    graph_scale: torch.Tensor


def build_model(atom_dims: Sequence[int], bond_dims: Sequence[int], seed: int,
                norm: Mapping[str, np.ndarray], use_words: bool,
                config: Mapping[str, Any] | None = None) -> tuple[GwicModel, dict[str, Any]]:
    cfg = dict(config or {})
    dims = dict(cfg.get("dims", {}))
    model = GwicModel(atom_dims, bond_dims, use_words=use_words,
                      atom_dim=int(dims.get("atom", 48)),
                      bond_dim=int(dims.get("bond", 24)),
                      h_dim=int(dims.get("sentence", 96)),
                      word_dim=int(dims.get("word", 64)),
                      q_dim=int(dims.get("q", 64)),
                      rank=int(dims.get("path_rank", 24)),
                      end_dim=int(dims.get("path_end", 16)),
                      close_dim=int(dims.get("path_close", 8)),
                      nbr_dim=int(dims.get("nbr", 16)),
                      backbone_hidden=int(dims.get("backbone_hidden", 192)),
                      word_hidden=int(dims.get("word_hidden", 160)),
                      score_hidden=int(dims.get("score_hidden", 96)),
                      q_hidden=int(dims.get("q_hidden", 128)),
                      readout_hidden=int(dims.get("readout_hidden", 192)),
                      readout_mid=int(dims.get("readout_mid", 96)),
                      dropout=float(dims.get("dropout", 0.1)))
    model.register_buffer("atom_scalar_mean", torch.as_tensor(norm["atom_scalar_mean"], dtype=torch.float32))
    model.register_buffer("atom_scalar_scale", torch.as_tensor(norm["atom_scalar_scale"], dtype=torch.float32))
    model.register_buffer("graph_mean", torch.as_tensor(norm["graph_mean"], dtype=torch.float32))
    model.register_buffer("graph_scale", torch.as_tensor(norm["graph_scale"], dtype=torch.float32))
    init_hashes = model.init_parameters(int(seed), last_std=1e-3)
    info = {
        "seed": int(seed), "use_words": bool(use_words),
        "config": dims, "parameters": model.parameter_ledger(),
        "init_hashes": init_hashes,
        "backbone_in_dim": int(model.backbone_in_dim),
        "word_in_dim": int(model.word_in_dim),
        "readout_in_dim": int(model.readout_in_dim),
    }
    return model, info


def load_model_from_state(atom_dims: Sequence[int], bond_dims: Sequence[int],
                          state: Mapping[str, torch.Tensor], norm: Mapping[str, np.ndarray],
                          use_words: bool, config: Mapping[str, Any] | None = None,
                          device: Any = "cpu") -> GwicModel:
    model, _ = build_model(atom_dims, bond_dims, seed=0, norm=norm,
                           use_words=use_words, config=config)
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    return model.to(device)

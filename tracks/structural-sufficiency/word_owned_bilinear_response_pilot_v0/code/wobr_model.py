"""WOBR-v0 model.

A non-iterative LCRC-style R2 sentence encoder produces ``h_v`` (used only as
the query for candidate-word selection), a shared word encoder produces
``w_(v,S)``, and root-wise sparsemax produces ``alpha`` and

    z_v = sum_S alpha_(v,S) w_(v,S).

The whole-molecule prediction is exactly

    yhat_std = beta + c_n*log1p(n_atoms) + c_m*log1p(n_physical_bonds)
               + sum_v dot(a, z_v)
               + sum_{(u,v)} dot(lambda_e, U(z_u)*U(z_v)) / sqrt(16)

with ``lambda_e = tanh(BondInteraction(b_e))`` and ``b_e`` the GWIC 24-d raw
bond embedding.  There is no other MLP, residual branch or pooled hidden state
in the prediction function.  ``h_v`` never enters the prediction directly.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import wobr_common as C
import wobr_sparsemax as SP


def _mlp(dims: Sequence[int], final_silu: bool = True,
         dropout: float = 0.0) -> nn.Sequential:
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


class WobrModel(nn.Module):
    def __init__(self, atom_dims: Sequence[int], bond_dims: Sequence[int],
                 atom_dim: int = 48, bond_dim: int = 24, h_dim: int = 96,
                 word_dim: int = 64, response_dim: int = 16,
                 rank: int = 24, end_dim: int = 16, close_dim: int = 8,
                 nbr_dim: int = 16, backbone_hidden: int = 192,
                 word_hidden: int = 160, score_hidden: int = 96,
                 interaction_hidden: int = 32, dropout: float = 0.1) -> None:
        super().__init__()
        self.atom_dim = int(atom_dim)
        self.bond_dim = int(bond_dim)
        self.h_dim = int(h_dim)
        self.word_dim = int(word_dim)
        self.response_dim = int(response_dim)
        self.rank = int(rank)
        self.end_dim = int(end_dim)
        self.close_dim = int(close_dim)
        self.nbr_dim = int(nbr_dim)
        self.n_atom_types = int(atom_dims[0])
        self.n_bond_types = int(bond_dims[0])
        self.n_path_codes = self.n_atom_types * self.n_bond_types * self.n_bond_types

        # OGB categorical embeddings: full official vocabulary.  Rows not
        # observed in official Train are zeroed after construction.
        self.atom_emb = nn.ModuleList([nn.Embedding(int(d), self.atom_dim) for d in atom_dims])
        self.bond_raw_emb = nn.ModuleList([nn.Embedding(int(d), self.bond_dim) for d in bond_dims])
        self.atom_norm = nn.LayerNorm(self.atom_dim)
        self.bond_encoder = _mlp([3 * self.atom_dim + self.bond_dim, 96, self.bond_dim])

        # LCRC-style rooted local content: S1, typed S2 paths, R2 summaries.
        self.nbr_emb = nn.Embedding(self.n_atom_types * self.n_bond_types, self.nbr_dim)
        self.path_u = nn.Embedding(self.n_path_codes, self.rank)
        self.path_v = nn.Embedding(self.n_atom_types, self.rank)
        self.path_joint = nn.Linear(self.rank, self.rank)
        self.path_emb = nn.Embedding(self.n_path_codes, self.rank)
        self.end_emb = nn.Embedding(self.n_atom_types, self.end_dim)
        self.close_emb = nn.Embedding(2, self.close_dim)
        self.path_gate = nn.Embedding(self.n_path_codes, 1)
        # Frozen GWIC sentence gate initialisation (verified R2 trunk).
        nn.init.ones_(self.path_gate.weight)

        self.backbone_in_dim = (self.atom_dim + 2 * self.bond_dim + self.nbr_dim
                                + (2 * self.rank + self.end_dim + self.close_dim)
                                + self.atom_dim
                                + 2 * self.bond_dim + 3 * self.bond_dim
                                + 4 * self.atom_dim + 5 + 3)
        self.backbone_mlp = _mlp([self.backbone_in_dim, backbone_hidden, self.h_dim])
        self.backbone_norm = nn.LayerNorm(self.h_dim)

        # Shared candidate-word encoder and score query.
        self.word_joint = _mlp([3 * self.atom_dim + self.bond_dim, 96, 48])
        self.word_in_dim = (self.atom_dim + 3 * self.atom_dim + 3 * self.bond_dim
                            + 2 * 48 + 25)
        self.word_mlp = _mlp([self.word_in_dim, word_hidden, self.word_dim],
                             dropout=dropout)
        self.word_norm = nn.LayerNorm(self.word_dim)
        self.score_mlp = _mlp([self.word_dim + self.h_dim, score_hidden, 1],
                              final_silu=False, dropout=dropout)

        # Response parameters.  These are the only parameters in the final
        # whole-molecule function besides the word/score encoders above.
        self.a = nn.Parameter(torch.zeros(self.word_dim))
        self.U = nn.Linear(self.word_dim, self.response_dim, bias=False)
        self.bond_interaction = nn.Sequential(
            nn.Linear(self.bond_dim, interaction_hidden),
            nn.SiLU(),
            nn.Linear(interaction_hidden, self.response_dim),
        )
        self.beta = nn.Parameter(torch.zeros(()))
        self.c_n = nn.Parameter(torch.zeros(()))
        self.c_m = nn.Parameter(torch.zeros(()))

    # -- deterministic initialisation ---------------------------------------
    def init_parameters(self, seed: int) -> dict[str, Any]:
        gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1000003 + 29)
        with torch.no_grad():
            # Conventional embeddings/hidden layers keep PyTorch defaults.
            # Explicit overrides from the frozen WOBR protocol:
            self.a.normal_(0.0, 0.02, generator=gen)
            bound = math.sqrt(6.0 / (self.word_dim + self.response_dim))
            self.U.weight.uniform_(-bound, bound, generator=gen)
            last = self.bond_interaction[-1]
            if not isinstance(last, nn.Linear):
                raise TypeError("BondInteraction must end in nn.Linear")
            last.weight.normal_(0.0, 0.02, generator=gen)
            if last.bias is not None:
                last.bias.zero_()
            self.beta.zero_()
            self.c_n.zero_()
            self.c_m.zero_()
        return {
            "parameter_hashes": C.module_hashes(self._hash_modules()),
            "state_fingerprint": C.state_fingerprint(self.state_dict()),
            "a_mean": float(self.a.detach().mean()),
            "a_std": float(self.a.detach().std(unbiased=False)),
            "U_xavier_bound": float(bound),
            "U_weight_absmax": float(self.U.weight.detach().abs().max()),
            "bond_interaction_last_weight_std": float(
                last.weight.detach().std(unbiased=False)),
            "beta_c_n_c_m": [float(self.beta), float(self.c_n), float(self.c_m)],
        }

    def _hash_modules(self) -> dict[str, nn.Module]:
        return {
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
            "word_joint": self.word_joint,
            "word_mlp": self.word_mlp,
            "word_norm": self.word_norm,
            "score_mlp": self.score_mlp,
            "a": self.a,
            "U": self.U,
            "bond_interaction": self.bond_interaction,
            "beta": self.beta,
            "c_n": self.c_n,
            "c_m": self.c_m,
        }

    def parameter_ledger(self) -> dict[str, int]:
        def n(module: nn.Module | nn.Parameter) -> int:
            if isinstance(module, nn.Parameter):
                return int(module.numel())
            return int(sum(p.numel() for p in module.parameters()))

        out = {name: n(module) for name, module in self._hash_modules().items()}
        out["total"] = int(sum(out.values()))
        return out

    # -- fit-only category support ------------------------------------------
    def zero_unseen(self, atom_ok: Sequence[np.ndarray], bond_ok: Sequence[np.ndarray],
                    atom_dims: Sequence[int], bond_dims: Sequence[int]) -> dict[str, int]:
        zeroed: dict[str, int] = {}
        atom_mask = torch.as_tensor(np.asarray(atom_ok[0], bool),
                                    device=self.path_gate.weight.device)
        bond_masks = [torch.as_tensor(np.asarray(x, bool),
                                      device=self.path_gate.weight.device)
                      for x in bond_ok]

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

        b1 = bond_masks[0]
        code_ok = torch.zeros(self.n_path_codes, dtype=torch.bool, device=atom_mask.device)
        idx = torch.arange(self.n_path_codes, device=atom_mask.device)
        cc_mid = idx // (self.n_bond_types * self.n_bond_types)
        cc_b1 = (idx // self.n_bond_types) % self.n_bond_types
        cc_b2 = idx % self.n_bond_types
        code_ok = atom_mask[cc_mid] & b1[cc_b1] & b1[cc_b2]
        for name, emb in (("path_u", self.path_u), ("path_emb", self.path_emb),
                          ("path_gate", self.path_gate)):
            zeroed[name] = int((~code_ok).sum().item())
            with torch.no_grad():
                emb.weight[~code_ok] = 0.0
        nbr_ok = torch.zeros(self.n_atom_types * self.n_bond_types, dtype=torch.bool,
                             device=atom_mask.device)
        nidx = torch.arange(self.n_atom_types * self.n_bond_types, device=atom_mask.device)
        nbr_ok = atom_mask[nidx // self.n_bond_types] & b1[nidx % self.n_bond_types]
        zeroed["nbr_emb"] = int((~nbr_ok).sum().item())
        with torch.no_grad():
            self.nbr_emb.weight[~nbr_ok] = 0.0
        zeroed["end_emb"] = int((~atom_mask).sum().item())
        zeroed["path_v"] = int((~atom_mask).sum().item())
        with torch.no_grad():
            self.end_emb.weight[~atom_mask] = 0.0
            self.path_v.weight[~atom_mask] = 0.0
        return zeroed

    # -- encoders -----------------------------------------------------------
    def embed_atoms(self, atom_feat: torch.Tensor) -> torch.Tensor:
        out = sum(emb(atom_feat[:, i]) for i, emb in enumerate(self.atom_emb))
        return self.atom_norm(out / math.sqrt(len(self.atom_emb)))

    def embed_bonds_raw(self, bond_feat: torch.Tensor) -> torch.Tensor:
        out = sum(emb(bond_feat[:, i]) for i, emb in enumerate(self.bond_raw_emb))
        return out / math.sqrt(len(self.bond_raw_emb))

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
            path_sum = a.new_zeros((n_atoms, 2 * self.rank + self.end_dim + self.close_dim))
            content_sum = a.new_zeros((n_atoms, self.atom_dim))

        if batch["r2_bond"].numel():
            r2b = b[batch["r2_bond"]]
            r2_sum = _seg_sum(r2b, batch["r2_seg"], n_atoms, self.bond_dim)
            r2_count = a.new_zeros(n_atoms)
            r2_count.index_add_(0, batch["r2_seg"],
                                torch.ones_like(batch["r2_seg"], dtype=a.dtype))
            r2_mean = r2_sum / r2_count.clamp(min=1.0)[:, None]
            bucket = _seg_sum(r2b, batch["r2_bucket_seg"], n_atoms * 3, self.bond_dim)
        else:
            r2_sum = a.new_zeros((n_atoms, self.bond_dim))
            r2_mean = a.new_zeros((n_atoms, self.bond_dim))
            r2_count = a.new_zeros(n_atoms)
            bucket = a.new_zeros((n_atoms * 3, self.bond_dim))
        bucket = bucket.view(n_atoms, 3 * self.bond_dim)

        d1_sum = _seg_sum(a[batch["dist1_atoms"]], batch["dist1_seg"], n_atoms, self.atom_dim) \
            if batch["dist1_atoms"].numel() else a.new_zeros((n_atoms, self.atom_dim))
        d2_sum = _seg_sum(a[batch["dist2_atoms"]], batch["dist2_seg"], n_atoms, self.atom_dim) \
            if batch["dist2_atoms"].numel() else a.new_zeros((n_atoms, self.atom_dim))
        d1_count = a.new_zeros(n_atoms)
        d2_count = a.new_zeros(n_atoms)
        if batch["dist1_atoms"].numel():
            d1_count.index_add_(0, batch["dist1_seg"],
                                torch.ones_like(batch["dist1_seg"], dtype=a.dtype))
        if batch["dist2_atoms"].numel():
            d2_count.index_add_(0, batch["dist2_seg"],
                                torch.ones_like(batch["dist2_seg"], dtype=a.dtype))
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
                     bond_raw: torch.Tensor) -> dict[str, torch.Tensor]:
        support = batch["word_support"]
        internal = batch["word_internal"]
        n_words = int(support.shape[0])
        if n_words == 0:
            return {"w": a.new_zeros((0, self.word_dim)),
                    "alpha": a.new_zeros(0),
                    "z": a.new_zeros((a.shape[0], self.word_dim)),
                    "scores": a.new_zeros(0)}
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

        idx = batch["bond_index"]
        if idx.numel():
            au, av = a[idx[:, 0]], a[idx[:, 1]]
            joint = self.word_joint(torch.cat([au + av, (au - av).abs(), au * av,
                                               bond_raw], dim=1))
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
        alpha = SP.sparsemax_packed(scores, batch["root_word_ptr"], int(a.shape[0]))
        z = a.new_zeros((int(a.shape[0]), self.word_dim))
        z.index_add_(0, batch["word_root"], alpha[:, None] * w)
        return {"w": w, "alpha": alpha, "z": z, "scores": scores}

    # -- prediction and exact word contributions ----------------------------
    def forward(self, batch: Mapping[str, torch.Tensor], *,
                return_aux: bool = False) -> dict[str, Any]:
        a = self.embed_atoms(batch["atom_feat"])
        bond_raw = self.embed_bonds_raw(batch["bond_feat"])
        b = self.encode_bonds(a, batch["bond_index"], bond_raw)
        h = self.encode_h(batch, a, b)
        word = self.encode_words(batch, a, b, h, bond_raw)
        z = word["z"]
        A = z @ self.a
        Uz = self.U(z)
        lam = torch.tanh(self.bond_interaction(bond_raw))
        idx = batch["bond_index"]
        n_mol = int(batch["n_molecules"])
        if idx.numel():
            u, v = idx[:, 0], idx[:, 1]
            Q = (lam * Uz[u] * Uz[v]).sum(dim=1) / math.sqrt(self.response_dim)
        else:
            Q = a.new_zeros(0)
        self_sum = a.new_zeros(n_mol)
        bond_sum = a.new_zeros(n_mol)
        self_sum.index_add_(0, batch["atom_mol"], A)
        bond_sum.index_add_(0, batch["bond_mol"], Q)
        atom_count = a.new_zeros(n_mol)
        atom_count.index_add_(0, batch["atom_mol"], torch.ones_like(batch["atom_mol"],
                                                                    dtype=a.dtype))
        bond_count = a.new_zeros(n_mol)
        bond_count.index_add_(0, batch["bond_mol"], torch.ones_like(batch["bond_mol"],
                                                                     dtype=a.dtype))
        yhat = (self.beta
                + self.c_n * torch.log1p(atom_count)
                + self.c_m * torch.log1p(bond_count)
                + self_sum + bond_sum)
        out: dict[str, Any] = {"yhat_std": yhat}
        if return_aux:
            out.update({
                "a": a, "b": b, "h": h, "bond_raw": bond_raw,
                "word": word, "w": word["w"], "alpha": word["alpha"],
                "z": z, "scores": word["scores"], "A": A, "Uz": Uz,
                "lam": lam, "Q": Q, "self_sum": self_sum, "bond_sum": bond_sum,
                "atom_count": atom_count, "bond_count": bond_count,
            })
        return out

    def word_contributions(self, batch: Mapping[str, torch.Tensor],
                           out: Mapping[str, torch.Tensor]
                           ) -> dict[str, torch.Tensor]:
        """Exact per-word contribution c_(v,S) in standardized units.

        c = alpha * (dot(a,w) + 0.5 * sum_{u adjacent v} Q_(v,u)(w, z_u)).
        Every physical undirected bond is split equally between its two ends.
        No detach and no auxiliary model is used.
        """
        w = out["w"]
        alpha = out["alpha"]
        Uz = out["Uz"]
        lam = out["lam"]
        n_atoms = int(out["z"].shape[0])
        n_words = int(w.shape[0])
        if n_words == 0:
            return {"c": w.new_zeros(0), "dot_a_w": w.new_zeros(0), "qsum": w.new_zeros(0)}
        dot_a_w = w @ self.a
        Uw = self.U(w)

        ptr = batch["root_word_ptr"]
        ab_ptr = batch["atom_bond_ptr"]
        ab_ids = batch["atom_bond_ids"]
        bond_index = batch["bond_index"]
        word_parts: list[torch.Tensor] = []
        bond_parts: list[torch.Tensor] = []
        owner_parts: list[torch.Tensor] = []
        for v in range(n_atoms):
            ws = torch.arange(int(ptr[v]), int(ptr[v + 1]), device=w.device)
            bs = ab_ids[int(ab_ptr[v]):int(ab_ptr[v + 1])]
            if ws.numel() == 0 or bs.numel() == 0:
                continue
            # rows: (bond, word) with word varying fastest
            ww = ws.unsqueeze(0).expand(bs.numel(), -1).reshape(-1)
            bb = bs.unsqueeze(1).expand(-1, ws.numel()).reshape(-1)
            word_parts.append(ww)
            bond_parts.append(bb)
            owner_parts.append(torch.full_like(ww, v))
        if word_parts:
            widx = torch.cat(word_parts)
            bidx = torch.cat(bond_parts)
            owner = torch.cat(owner_parts)
            endpoints = bond_index[bidx]
            neighbor = endpoints[:, 0] + endpoints[:, 1] - owner
            q_word = (lam[bidx] * Uw[widx] * Uz[neighbor]).sum(dim=1) \
                / math.sqrt(self.response_dim)
            qsum = w.new_zeros(n_words)
            qsum.index_add_(0, widx, q_word)
        else:
            qsum = w.new_zeros(n_words)
        c = alpha * (dot_a_w + 0.5 * qsum)
        return {"c": c, "dot_a_w": dot_a_w, "qsum": qsum}

    # buffers are registered after construction in build_model
    atom_scalar_mean: torch.Tensor
    atom_scalar_scale: torch.Tensor


def build_model(atom_dims: Sequence[int], bond_dims: Sequence[int], seed: int,
                norm: Mapping[str, np.ndarray],
                config: Mapping[str, Any] | None = None) -> tuple[WobrModel, dict[str, Any]]:
    cfg = dict(config or {})
    dims = dict(cfg.get("dims", {}))
    model = WobrModel(
        atom_dims, bond_dims,
        atom_dim=int(dims.get("atom", 48)),
        bond_dim=int(dims.get("bond", 24)),
        h_dim=int(dims.get("sentence", 96)),
        word_dim=int(dims.get("word", 64)),
        response_dim=int(dims.get("response", 16)),
        rank=int(dims.get("path_rank", 24)),
        end_dim=int(dims.get("path_end", 16)),
        close_dim=int(dims.get("path_close", 8)),
        nbr_dim=int(dims.get("nbr", 16)),
        backbone_hidden=int(dims.get("backbone_hidden", 192)),
        word_hidden=int(dims.get("word_hidden", 160)),
        score_hidden=int(dims.get("score_hidden", 96)),
        interaction_hidden=int(dims.get("interaction_hidden", 32)),
        dropout=float(dims.get("dropout", 0.1)),
    )
    model.register_buffer("atom_scalar_mean",
                          torch.as_tensor(norm["atom_scalar_mean"], dtype=torch.float32))
    model.register_buffer("atom_scalar_scale",
                          torch.as_tensor(norm["atom_scalar_scale"], dtype=torch.float32))
    init_info = model.init_parameters(int(seed))
    info = {
        "seed": int(seed),
        "config": dims,
        "parameters": model.parameter_ledger(),
        "init": init_info,
        "backbone_in_dim": int(model.backbone_in_dim),
        "word_in_dim": int(model.word_in_dim),
    }
    return model, info


def load_model_from_state(atom_dims: Sequence[int], bond_dims: Sequence[int],
                          state: Mapping[str, torch.Tensor],
                          norm: Mapping[str, np.ndarray],
                          config: Mapping[str, Any] | None = None,
                          device: Any = "cpu") -> WobrModel:
    model, _ = build_model(atom_dims, bond_dims, seed=0, norm=norm, config=config)
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    return model.to(device)
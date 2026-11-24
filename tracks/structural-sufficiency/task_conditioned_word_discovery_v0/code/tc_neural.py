"""Shared non-iterative root-level neural readout for the vocabulary arms.

No GNN, no message passing beyond the explicit first-order pair encoder, no
pretraining, no teacher, no distillation.  D_TASK and D_FREQ instantiate exactly
the same modules and parameter count; BASE feeds an all-zero word vector of the
same shape.

All second-order structure reaching this model comes from the frozen selected
vocabulary columns in ``batch['word_counts']``.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import tc_common as C

ARMS = ("BASE", "D_TASK", "D_FREQ")


def _mlp(dims: Sequence[int], dropout: float = 0.0) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(int(dims[i]), int(dims[i + 1])))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
    return nn.Sequential(*layers)


class TCWordModel(nn.Module):
    def __init__(self, atom_dims: Sequence[int], bond_dims: Sequence[int], K: int,
                 dims: Mapping[str, Any]) -> None:
        super().__init__()
        self.atom_dim = int(dims["atom"])
        self.bond_dim = int(dims["bond"])
        self.K = int(K)
        self.atom_emb = nn.ModuleList([nn.Embedding(int(d), self.atom_dim)
                                       for d in atom_dims])
        self.bond_emb = nn.ModuleList([nn.Embedding(int(d), self.bond_dim)
                                       for d in bond_dims])
        pair_in = self.atom_dim + self.bond_dim
        self.pair_in_dim = int(pair_in)
        self.pair_encoder = _mlp([pair_in, int(dims["pair_hidden"]),
                                  int(dims["pair_out"])])
        self.word_encoder = _mlp([self.K, int(dims["word_hidden"]),
                                  int(dims["word_out"])])
        sent_in = self.atom_dim + int(dims["pair_out"]) + int(dims["word_out"])
        self.sentence_in_dim = int(sent_in)
        self.sentence_mlp = _mlp([sent_in, int(dims["sentence_hidden"]),
                                  int(dims["sentence_out"])],
                                 dropout=float(dims["dropout"]))
        read_in = 3 * int(dims["sentence_out"]) + 1 + 8
        self.readout_in_dim = int(read_in)
        self.readout_mlp = nn.Sequential(
            nn.Linear(read_in, int(dims["readout_hidden"])), nn.SiLU(),
            nn.Dropout(float(dims["dropout"])),
            nn.Linear(int(dims["readout_hidden"]), int(dims["readout_mid"])), nn.SiLU(),
            nn.Dropout(float(dims["dropout"])),
            nn.Linear(int(dims["readout_mid"]), 1),
        )

    # -- deterministic initialisation ---------------------------------------
    def init_parameters(self, seed: int) -> dict[str, str]:
        gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1000003 + 29)
        with torch.no_grad():
            for emb in list(self.atom_emb) + list(self.bond_emb):
                emb.weight.normal_(0.0, 0.1, generator=gen)
            modules = [self.pair_encoder, self.word_encoder, self.sentence_mlp,
                       self.readout_mlp]
            for module in modules:
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        bound = 1.0 / math.sqrt(layer.in_features)
                        layer.weight.uniform_(-bound, bound, generator=gen)
                        if layer.bias is not None:
                            layer.bias.uniform_(-bound, bound, generator=gen)
            last = [m for m in self.readout_mlp.modules()
                    if isinstance(m, nn.Linear) and m.out_features == 1][-1]
            last.weight.normal_(0.0, 1e-3, generator=gen)
            last.bias.zero_()
        return self.hashes()

    def hashes(self) -> dict[str, str]:
        mods = {"atom_emb": self.atom_emb, "bond_emb": self.bond_emb,
                "pair_encoder": self.pair_encoder, "word_encoder": self.word_encoder,
                "sentence_mlp": self.sentence_mlp, "readout_mlp": self.readout_mlp}
        return C.module_hashes(mods)

    def parameter_ledger(self) -> dict[str, int]:
        def n(module: nn.Module) -> int:
            return int(sum(p.numel() for p in module.parameters()))

        out = {
            "atom_emb": n(self.atom_emb), "bond_emb": n(self.bond_emb),
            "pair_encoder": n(self.pair_encoder), "word_encoder": n(self.word_encoder),
            "sentence_mlp": n(self.sentence_mlp), "readout_mlp": n(self.readout_mlp),
        }
        out["total"] = int(sum(out.values()))
        return out

    def zero_unseen(self, atom_ok: Sequence[np.ndarray],
                    bond_ok: Sequence[np.ndarray]) -> dict[str, int]:
        zeroed: dict[str, int] = {}
        with torch.no_grad():
            for i, emb in enumerate(self.atom_emb):
                mask = torch.as_tensor(np.asarray(atom_ok[i], bool),
                                       device=emb.weight.device)
                zeroed[f"atom_{i}"] = int((~mask).sum().item())
                emb.weight[~mask] = 0.0
            for i, emb in enumerate(self.bond_emb):
                mask = torch.as_tensor(np.asarray(bond_ok[i], bool),
                                       device=emb.weight.device)
                zeroed[f"bond_{i}"] = int((~mask).sum().item())
                emb.weight[~mask] = 0.0
        return zeroed

    # -- forward ------------------------------------------------------------
    def forward(self, batch: Mapping[str, Any], arm: str) -> torch.Tensor:
        atom = sum(emb(batch["atom_feat"][:, i]) for i, emb in enumerate(self.atom_emb))
        bond = sum(emb(batch["bond_feat"][:, i]) for i, emb in enumerate(self.bond_emb))
        pair = self.pair_encoder(torch.cat([
            atom[batch["edge_dst"]], bond[batch["edge_bond"]]], dim=-1))
        nbr = atom.new_zeros((atom.shape[0], pair.shape[1]))
        if pair.numel():
            nbr.index_add_(0, batch["edge_src"], pair)
        if arm == "BASE":
            z = batch["word_counts"].new_zeros((atom.shape[0], self.K))
        elif arm in ("D_TASK", "D_FREQ"):
            z = batch["word_counts"]
            if int(z.shape[1]) != self.K:
                raise ValueError(f"word width {z.shape[1]} != K {self.K}")
        else:
            raise KeyError(arm)
        word = self.word_encoder(z)
        h = self.sentence_mlp(torch.cat([atom, nbr, word], dim=-1))

        mol = batch["atom_mol"]
        n = int(batch["n_molecules"])
        d = int(h.shape[1])
        total = h.new_zeros((n, d)).index_add_(0, mol, h)
        sq = h.new_zeros((n, d)).index_add_(0, mol, h * h)
        cnt = h.new_zeros(n).index_add_(0, mol, torch.ones_like(mol, dtype=h.dtype))
        denom = cnt.clamp(min=1.0)
        mean = total / denom[:, None]
        var = (sq / denom[:, None] - mean * mean).clamp(min=0.0)
        std = torch.sqrt(var + 1e-8)
        std = torch.where((cnt > 1)[:, None], std, torch.zeros_like(std))
        read_in = torch.cat([total, mean, std,
                             torch.log1p(batch["n_atoms"])[:, None],
                             batch["graph_scalars"]], dim=-1)
        return self.readout_mlp(read_in).view(-1)
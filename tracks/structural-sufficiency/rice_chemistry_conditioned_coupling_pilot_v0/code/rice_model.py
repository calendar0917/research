"""RICE-CC-v0 model: the RICE-v0 rooted interaction cumulant encoder with an
atom-conditioned binary pair potential.

Both arms are the **same** model, with **every** parameter shared (identical
module set, identical initialisation, identical recipe).  Both use the
root-conditioned Mobius inversion ``kappa(S, v)`` of size 1 -> 4.  They differ
only in the pair potential that enters ``compute_phi``:

* ``CC_NULL`` uses the original bond-only ``J_base`` from the bond MLP
  (mathematically identical to historical ``RICE_CUMULANT``).
* ``CC_REAL`` uses ``J_real = (J00 + delta, J01 - delta, J11 + delta)`` with
  ``delta = k * h_i * h_j``, ``h = tanh((u_1 - u_0) / 2)``,
  ``k = (J00 + J11 - 2 J01) / 4``.  No parameter is added and nothing is
  detached.

No per-word-type parameters exist: ``u_i`` and ``J_base,e`` are produced by two
shared MLPs from the full OGB atom / bond categorical fields, so a word type has
no private weight anywhere.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import rice_common as C
import rice_energy as E

ARMS = ("CC_NULL", "CC_REAL")
COUPLING = {"CC_NULL": "base", "CC_REAL": "chem"}
SIZES = (1, 2, 3, 4)


def _mlp(dims: Sequence[int], dropout: float = 0.0,
         final_activation: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(int(dims[i]), int(dims[i + 1])))
        if i < len(dims) - 2:
            layers.append(nn.SiLU())
            if dropout > 0:
                layers.append(nn.Dropout(float(dropout)))
    if final_activation:
        layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def _sumsq(values: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    out = values.new_zeros((int(n), int(values.shape[1])))
    if values.numel():
        out.index_add_(0, seg, values * values)
    return out


class RiceModel(nn.Module):
    def __init__(self, atom_dims: Sequence[int], bond_dims: Sequence[int],
                 channels: int = 16, states: int = 2, atom_emb: int = 16,
                 bond_emb: int = 12, atom_hidden: int = 64, bond_hidden: int = 64,
                 potential_bound: float = 2.0, h_dim: int = 96,
                 sentence_hidden: int = 128, readout_hidden: int = 128,
                 readout_mid: int = 64, dropout: float = 0.1,
                 energy_chunk: int = 4096) -> None:
        super().__init__()
        if int(states) != 2:
            raise ValueError("RICE-v0 fixes two internal states per channel")
        self.channels = int(channels)
        self.states = int(states)
        self.kappa_dim = self.channels * self.states
        self.potential_bound = float(potential_bound)
        self.energy_chunk = int(energy_chunk)
        self.atom_emb = nn.ModuleList([nn.Embedding(int(d), int(atom_emb))
                                       for d in atom_dims])
        self.bond_emb = nn.ModuleList([nn.Embedding(int(d), int(bond_emb))
                                       for d in bond_dims])
        self.atom_mlp = _mlp([int(atom_emb) * len(atom_dims), atom_hidden,
                              self.kappa_dim])
        # three independent symmetric binary potentials per channel: 00, 01/10, 11
        self.bond_mlp = _mlp([int(bond_emb) * len(bond_dims), bond_hidden,
                              3 * self.channels])
        self.order_norm = nn.ModuleList([nn.LayerNorm(3 * self.kappa_dim)
                                         for _ in SIZES])
        block = 3 * self.kappa_dim + 1
        self.sentence_in_dim = block * len(SIZES)
        self.sentence_mlp = _mlp([self.sentence_in_dim, sentence_hidden, int(h_dim)],
                                 dropout=dropout)
        self.h_dim = int(h_dim)
        self.readout_in_dim = 3 * self.h_dim + 1 + 8
        self.readout_mlp = nn.Sequential(
            nn.Linear(self.readout_in_dim, int(readout_hidden)), nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(readout_hidden), int(readout_mid)), nn.SiLU(),
            nn.Linear(int(readout_mid), 1),
        )
        self.beta = nn.Parameter(torch.zeros(()))

    # -- deterministic initialisation ---------------------------------------
    def init_parameters(self, seed: int, last_std: float = 1e-3) -> dict[str, str]:
        gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1000003 + 29)
        with torch.no_grad():
            for emb in list(self.atom_emb) + list(self.bond_emb):
                emb.weight.normal_(0.0, 0.1, generator=gen)
            modules = [self.atom_mlp, self.bond_mlp, self.sentence_mlp,
                       self.order_norm, self.readout_mlp]
            for module in modules:
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        bound = 1.0 / math.sqrt(layer.in_features)
                        layer.weight.uniform_(-bound, bound, generator=gen)
                        if layer.bias is not None:
                            layer.bias.uniform_(-bound, bound, generator=gen)
                    elif isinstance(layer, nn.LayerNorm):
                        layer.weight.fill_(1.0)
                        layer.bias.zero_()
            last = [m for m in self.readout_mlp.modules()
                    if isinstance(m, nn.Linear) and m.out_features == 1][-1]
            last.weight.normal_(0.0, float(last_std), generator=gen)
            last.bias.zero_()
            self.beta.zero_()
        return self.shared_hashes()

    def shared_hashes(self) -> dict[str, str]:
        return C.module_hashes({
            "atom_emb": self.atom_emb, "bond_emb": self.bond_emb,
            "atom_mlp": self.atom_mlp, "bond_mlp": self.bond_mlp,
            "order_norm": self.order_norm, "sentence_mlp": self.sentence_mlp,
            "readout_mlp": self.readout_mlp, "beta": self.beta,
        })

    def parameter_ledger(self) -> dict[str, int]:
        def n(module: nn.Module) -> int:
            if isinstance(module, nn.Parameter):
                return int(module.numel())
            return int(sum(p.numel() for p in module.parameters()))

        out = {
            "atom_emb": n(self.atom_emb), "bond_emb": n(self.bond_emb),
            "atom_mlp": n(self.atom_mlp), "bond_mlp": n(self.bond_mlp),
            "order_norm": n(self.order_norm), "sentence_mlp": n(self.sentence_mlp),
            "readout_mlp": n(self.readout_mlp), "beta": n(self.beta),
        }
        out["total"] = int(sum(out.values()))
        return out

    # -- fit-only category support ------------------------------------------
    def zero_unseen(self, atom_ok: Sequence[np.ndarray],
                    bond_ok: Sequence[np.ndarray]) -> dict[str, int]:
        zeroed: dict[str, int] = {}
        with torch.no_grad():
            for i, emb in enumerate(self.atom_emb):
                mask = torch.as_tensor(np.asarray(atom_ok[i], bool), device=emb.weight.device)
                zeroed[f"atom_field_{i}"] = int((~mask).sum().item())
                emb.weight[~mask] = 0.0
            for i, emb in enumerate(self.bond_emb):
                mask = torch.as_tensor(np.asarray(bond_ok[i], bool), device=emb.weight.device)
                zeroed[f"bond_field_{i}"] = int((~mask).sum().item())
                emb.weight[~mask] = 0.0
        return zeroed

    # -- potentials ---------------------------------------------------------
    def potentials(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor]:
        atom_in = torch.cat([emb(batch["atom_feat"][:, i])
                             for i, emb in enumerate(self.atom_emb)], dim=1)
        raw_u = self.atom_mlp(atom_in)
        u = self.potential_bound * torch.tanh(raw_u).view(-1, self.channels, self.states)
        bond_in = torch.cat([emb(batch["bond_feat"][:, i]) for i, emb in enumerate(self.bond_emb)],
                            dim=1)
        raw_j = self.bond_mlp(bond_in)
        j = self.potential_bound * torch.tanh(raw_j).view(-1, self.channels, 3)
        return u, j

    def pair_potential(self, j_base: torch.Tensor, u: torch.Tensor,
                       bond_index: torch.Tensor,
                       arm: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Return the pair potential actually used by ``arm`` and its delta."""
        mode = COUPLING.get(str(arm))
        if mode is None:
            raise KeyError(f"unknown arm {arm!r}")
        if mode == "base":
            return j_base, torch.zeros_like(j_base[..., 0])
        j_real, delta = E.couple_pairs(u, j_base, bond_index)
        return j_real, delta

    # -- forward ------------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor], *, arm: str,
                size_off: Sequence[int] = (), size_off_counts: bool = False,
                word_mask: Mapping[int, torch.Tensor] | None = None,
                block_override: Mapping[int, torch.Tensor] | None = None,
                return_aux: bool = False) -> dict[str, Any]:
        if arm not in ARMS:
            raise KeyError(f"unknown arm {arm!r}")
        u, j_base = self.potentials(batch)
        j, delta = self.pair_potential(j_base, u, batch["bond_index"], arm)
        n_atoms = int(batch["n_atoms"])
        u_flat = u.reshape(n_atoms, self.kappa_dim)
        phi = {size: E.compute_phi(u, j, batch[f"atoms_{size}"], batch[f"bonds_{size}"],
                                   batch[f"bslot_{size}"], chunk=self.energy_chunk)
               for size in (2, 3, 4)}
        kappa = E.mobius(u_flat, phi, {size: batch[f"atoms_{size}"] for size in (2, 3, 4)},
                         {size: batch[f"sub2_{size}"] for size in (3, 4)},
                         batch["sub3_4"])
        values = kappa
        off = set(int(x) for x in size_off)

        blocks: list[torch.Tensor] = []
        aux_words: dict[int, Any] = {}
        for order in SIZES:
            if order == 1:
                vals = u_flat
                seg = torch.arange(n_atoms, device=u.device)
            else:
                vals = values[order]
                seg = batch[f"atoms_{order}"][:, 0]
                if word_mask is not None and order in word_mask:
                    vals = vals * word_mask[order].unsqueeze(-1).to(vals.dtype)
                if order in off:
                    vals = vals * 0.0
            total, mean, std, count = E.seg_stats(vals, seg, n_atoms, self.kappa_dim)
            if order in off and size_off_counts:
                count = count * 0.0
                total = total * 0.0
                mean = mean * 0.0
                std = std * 0.0
            pooled = torch.cat([total, mean, std], dim=1)
            if block_override is not None and order in block_override:
                # analysis-only path: substitute a fixed pooled block (e.g. the Fit
                # mean) so a frozen checkpoint can be re-read without retraining
                pooled = block_override[order].to(pooled.dtype).reshape(1, -1).expand_as(pooled)
            normed = self.order_norm[order - 1](pooled)
            blocks.append(torch.cat([normed, torch.log1p(count).unsqueeze(1)], dim=1))
            if return_aux:
                aux_words[order] = {"values": vals, "sum": total, "mean": mean,
                                    "std": std, "count": count,
                                    "sumsq": _sumsq(vals, seg, n_atoms)}
        sentence = torch.cat(blocks, dim=1)
        h = self.sentence_mlp(sentence)

        h_sum, h_mean, h_std, atom_count = E.seg_stats(h, batch["atom_mol"],
                                                       int(batch["n_molecules"]), self.h_dim)
        g = (batch["graph_scalars"] - self.graph_mean) / self.graph_scale
        readout_in = torch.cat([h_sum, h_mean, h_std,
                                torch.log1p(atom_count).unsqueeze(1), g], dim=1)
        if readout_in.shape[1] != self.readout_in_dim:
            raise ValueError(f"readout width {readout_in.shape[1]} != {self.readout_in_dim}")
        yhat = self.readout_mlp(readout_in).view(-1) + self.beta
        out: dict[str, Any] = {"yhat_std": yhat}
        if return_aux:
            out.update({"u": u, "u_flat": u_flat, "j_base": j_base, "j": j,
                        "delta": delta, "phi": phi, "kappa": kappa,
                        "h": h, "sentence": sentence, "words": aux_words,
                        "readout_in": readout_in, "graph": g,
                        "n_atoms": n_atoms, "n_molecules": int(batch["n_molecules"])})
        return out

    graph_mean: torch.Tensor
    graph_scale: torch.Tensor


def build_model(atom_dims: Sequence[int], bond_dims: Sequence[int], seed: int,
                norm: Mapping[str, np.ndarray],
                config: Mapping[str, Any] | None = None
                ) -> tuple[RiceModel, dict[str, Any]]:
    cfg = dict(config or {})
    dims = dict(cfg.get("dims", {}))
    model = RiceModel(atom_dims, bond_dims,
                      channels=int(dims.get("channels", 16)),
                      states=int(dims.get("states", 2)),
                      atom_emb=int(dims.get("atom_emb", 16)),
                      bond_emb=int(dims.get("bond_emb", 12)),
                      atom_hidden=int(dims.get("atom_hidden", 64)),
                      bond_hidden=int(dims.get("bond_hidden", 64)),
                      potential_bound=float(dims.get("potential_bound", 2.0)),
                      h_dim=int(dims.get("h_dim", 96)),
                      sentence_hidden=int(dims.get("sentence_hidden", 128)),
                      readout_hidden=int(dims.get("readout_hidden", 128)),
                      readout_mid=int(dims.get("readout_mid", 64)),
                      dropout=float(dims.get("dropout", 0.1)),
                      energy_chunk=int(dims.get("energy_chunk", 4096)))
    model.register_buffer("graph_mean", torch.as_tensor(norm["graph_mean"], dtype=torch.float32))
    model.register_buffer("graph_scale", torch.as_tensor(norm["graph_scale"], dtype=torch.float32))
    init_hashes = model.init_parameters(int(seed), last_std=1e-3)
    info = {
        "seed": int(seed), "dims": dims, "parameters": model.parameter_ledger(),
        "init_hashes": init_hashes,
        "sentence_in_dim": int(model.sentence_in_dim),
        "readout_in_dim": int(model.readout_in_dim),
    }
    return model, info


def load_model_from_state(state: Mapping[str, torch.Tensor], atom_dims: Sequence[int],
                          bond_dims: Sequence[int], norm: Mapping[str, np.ndarray],
                          config: Mapping[str, Any] | None = None,
                          device: Any = "cpu") -> RiceModel:
    model, _ = build_model(atom_dims, bond_dims, seed=0, norm=norm, config=config)
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    return model.to(device)
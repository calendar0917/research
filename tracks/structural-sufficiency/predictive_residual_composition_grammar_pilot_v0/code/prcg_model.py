"""PRCG-v0 model: recursive word composition + one-shot word/bond readout.

Structure (identical for all three arms, only the grammar and the port binding
change):

    z(atom word)  = AtomMLP(core-field embeddings)
    z(parent)     = Compose(z_left, z_right, crossing bonds + their real ports)
    q_e           = Q(z_A, z_B, port_A, port_B, bond_e, local context)   [scalar]
    f_w           = F(z_w, context_w)                                    [scalar]
    yhat          = sum_w f_w + sum_e q_e + Readout(global aggregates, graph scalars)

``Compose`` is one shared symmetric MLP over all rules, sizes and molecules.  It
is the only way a multi-atom word can get a state: there is no per-type or
per-rule lookup table and no bypass encoder.  ``Q`` is symmetric under swapping
the two endpoints of an undirected bond.  ``port_mode="MEAN"`` replaces every
real port-role vector by the word's mean port role and changes nothing else.
"""

from __future__ import annotations

import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import prcg_tensor as T

ROLE = T.ROLE_DIM
WSCAL = T.WORD_SCALARS
CTX = 9          # per-word atom context summary (environment-facing)
N_ATOM_CTX = 2 + 2 + 2 * CTX   # local context size of an inter-word bond


def _mlp(dims: Sequence[int], final_act: bool = True) -> nn.Sequential:
    layers: list[nn.Module] = []
    for i in range(len(dims) - 1):
        layers.append(nn.Linear(int(dims[i]), int(dims[i + 1])))
        if i < len(dims) - 2 or final_act:
            layers.append(nn.SiLU())
    return nn.Sequential(*layers)


def _seg_sum(values: torch.Tensor, seg: torch.Tensor, n: int) -> torch.Tensor:
    out = values.new_zeros((int(n), values.shape[1]))
    if values.numel():
        out.index_add_(0, seg, values)
    return out


def _seg_stats_masked(values: torch.Tensor, seg: torch.Tensor, n: int,
                      counts: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    total = _seg_sum(values, seg, n)
    sq = _seg_sum(values * values, seg, n)
    denom = counts.clamp(min=1.0)[:, None]
    mean = total / denom
    var = (sq / denom - mean * mean).clamp(min=0.0)
    std = torch.where((counts > 1)[:, None], torch.sqrt(var + 1e-8), torch.zeros_like(var))
    return total, mean, std


def _seg_stats(values: torch.Tensor, seg: torch.Tensor, n: int
               ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    d = values.shape[1]
    total = _seg_sum(values, seg, n)
    sq = _seg_sum(values * values, seg, n)
    count = values.new_zeros(int(n))
    if values.numel():
        count.index_add_(0, seg, torch.ones(values.shape[0], device=values.device,
                                            dtype=values.dtype))
    denom = count.clamp(min=1.0)
    mean = total / denom[:, None]
    var = (sq / denom[:, None] - mean * mean).clamp(min=0.0)
    std = torch.where((count > 1)[:, None], torch.sqrt(var + 1e-8), torch.zeros_like(var))
    return total, mean, std


class PrcgModel(nn.Module):
    def __init__(self, atom_dims: Sequence[int], bond_dims: Sequence[int],
                 z_dim: int = 48, atom_hidden: int = 48, compose_hidden: int = 96,
                 q_dim: int = 32, q_hidden: int = 128, f_hidden: int = 32,
                 readout_hidden: int = 128, readout_mid: int = 64,
                 bond_dim: int = 10, stereo_dim: int = 4, dropout: float = 0.0) -> None:
        super().__init__()
        self.z_dim = int(z_dim)
        self.q_dim = int(q_dim)
        self.bond_dim = int(bond_dim)
        self.stereo_dim = int(stereo_dim)
        # atom core fields only: atomic number, formal charge, aromaticity
        core_dims = [int(atom_dims[0]), int(atom_dims[3]), int(atom_dims[7])]
        self.atom_emb = nn.ModuleList([nn.Embedding(d, 12) for d in core_dims])
        self.atom_mlp = _mlp([36, atom_hidden, self.z_dim])
        # bond core (type, conjugation) + stereo (context only)
        self.bond_emb = nn.ModuleList([nn.Embedding(int(bond_dims[0]), self.bond_dim),
                                       nn.Embedding(int(bond_dims[2]), 4)])
        self.stereo_emb = nn.Embedding(int(bond_dims[1]), self.stereo_dim)
        bond_feat_dim = self.bond_dim + 4 + self.stereo_dim

        self.compose_in = 3 * self.z_dim + 3 * ROLE + self.bond_dim + 4 + 2 + WSCAL
        self.compose = _mlp([self.compose_in, compose_hidden, self.z_dim])
        self.compose_norm = nn.LayerNorm(self.z_dim)

        self.q_in = 3 * self.z_dim + 3 * ROLE + bond_feat_dim + N_ATOM_CTX
        self.q_mlp = _mlp([self.q_in, q_hidden, self.q_dim])
        self.q_scalar = nn.Linear(self.q_dim, 1)

        self.f_in = self.z_dim + 2 + 1 + CTX
        self.f_mlp = _mlp([self.f_in, f_hidden, 1])

        self.readout_in = 3 * self.z_dim + 3 * self.q_dim + 2 + 8
        self.readout = nn.Sequential(
            nn.Linear(self.readout_in, readout_hidden), nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(readout_hidden, readout_mid), nn.SiLU(),
            nn.Linear(readout_mid, 1),
        )

    # -- deterministic initialisation ---------------------------------------
    def init_parameters(self, seed: int, last_std: float = 1e-3) -> dict[str, str]:
        gen = torch.Generator(device="cpu").manual_seed(int(seed) * 1000003 + 41)
        with torch.no_grad():
            for emb in list(self.atom_emb) + list(self.bond_emb) + [self.stereo_emb]:
                emb.weight.normal_(0.0, 0.1, generator=gen)
            for module in (self.atom_mlp, self.compose, self.q_mlp, self.f_mlp,
                           self.readout):
                for layer in module.modules():
                    if isinstance(layer, nn.Linear):
                        bound = 1.0 / math.sqrt(layer.in_features)
                        layer.weight.uniform_(-bound, bound, generator=gen)
                        if layer.bias is not None:
                            layer.bias.uniform_(-bound, bound, generator=gen)
            self.q_scalar.weight.normal_(0.0, 1.0 / math.sqrt(self.q_dim), generator=gen)
            self.q_scalar.bias.zero_()
            self.compose_norm.reset_parameters()
            last = [m for m in self.readout.modules()
                    if isinstance(m, nn.Linear) and m.out_features == 1][-1]
            last.weight.normal_(0.0, float(last_std), generator=gen)
            last.bias.zero_()
            f_last = [m for m in self.f_mlp.modules()
                      if isinstance(m, nn.Linear) and m.out_features == 1][-1]
            f_last.bias.zero_()
        return self.parameter_hashes()

    def parameter_hashes(self) -> dict[str, str]:
        import prcg_common as C

        mods: dict[str, nn.Module] = {
            "atom_emb": self.atom_emb, "atom_mlp": self.atom_mlp,
            "bond_emb": self.bond_emb, "stereo_emb": self.stereo_emb,
            "compose": self.compose, "compose_norm": self.compose_norm,
            "q_mlp": self.q_mlp, "q_scalar": self.q_scalar, "f_mlp": self.f_mlp,
            "readout": self.readout,
        }
        return C.module_hashes(mods)

    def parameter_ledger(self) -> dict[str, int]:
        def n(module: nn.Module) -> int:
            return int(sum(p.numel() for p in module.parameters()))

        out = {"atom_emb": n(self.atom_emb), "atom_mlp": n(self.atom_mlp),
               "bond_emb": n(self.bond_emb), "stereo_emb": n(self.stereo_emb),
               "compose": n(self.compose), "compose_norm": n(self.compose_norm),
               "q_mlp": n(self.q_mlp), "q_scalar": n(self.q_scalar),
               "f_mlp": n(self.f_mlp), "readout": n(self.readout)}
        out["total"] = int(sum(out.values()))
        return out

    # -- forward pieces -----------------------------------------------------
    def bond_core_features(self, bond_core: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.bond_emb[0](bond_core[:, 0]),
                          self.bond_emb[1](bond_core[:, 1])], dim=1)

    def _bond_features(self, bond_core: torch.Tensor, stereo: torch.Tensor) -> torch.Tensor:
        return torch.cat([self.bond_emb[0](bond_core[:, 0]),
                          self.bond_emb[1](bond_core[:, 1]),
                          self.stereo_emb(stereo)], dim=1)

    def atom_states(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        core = batch["w_atom_core"]
        emb = torch.cat([self.atom_emb[0](core[:, 0]), self.atom_emb[1](core[:, 1]),
                         self.atom_emb[2](core[:, 2])], dim=1)
        return self.atom_mlp(emb)

    def port_roles(self, batch: Mapping[str, torch.Tensor], mode: str) -> dict[str, torch.Tensor]:
        """REAL = the port that this bond actually uses; MEAN = the word's mean
        port role (identical for words with a single port)."""
        if mode == "REAL":
            return {"x_l": batch["x_role_l"], "x_r": batch["x_role_r"],
                    "e_i": batch["e_role_i"], "e_j": batch["e_role_j"]}
        if mode == "MEAN":
            mr = batch["w_mean_role"]
            parent = batch["x_word"]
            return {"x_l": mr[batch["w_child1"][parent]],
                    "x_r": mr[batch["w_child2"][parent]],
                    "e_i": mr[batch["e_i"]], "e_j": mr[batch["e_j"]]}
        raise ValueError(f"unknown port mode {mode}")

    def word_states(self, batch: Mapping[str, torch.Tensor], mode: str = "REAL"
                    ) -> tuple[torch.Tensor, dict[str, Any]]:
        n_words = int(batch["w_rule"].numel())
        z = batch["w_scal"].new_zeros((n_words, self.z_dim))
        is_atom = batch["w_rule"] < 0
        if bool(is_atom.any()):
            z = z.index_copy(0, is_atom.nonzero(as_tuple=True)[0],
                             self.atom_states(batch)[is_atom])
        roles = self.port_roles(batch, mode)
        bond_core = self.bond_core_features(batch["b_core"])
        xfeat = torch.cat([bond_core[batch["x_bond"]],
                           roles["x_l"] + roles["x_r"],
                           (roles["x_l"] - roles["x_r"]).abs(),
                           roles["x_l"] * roles["x_r"]], dim=1)
        x_sum = _seg_sum(xfeat, batch["x_word"], n_words)
        x_count = xfeat.new_zeros(n_words)
        if xfeat.numel():
            x_count.index_add_(0, batch["x_word"], torch.ones(xfeat.shape[0],
                                                             device=xfeat.device,
                                                             dtype=xfeat.dtype))
        x_mean = x_sum / x_count.clamp(min=1.0)[:, None]
        cross_in = torch.cat([x_mean, torch.log1p(x_count)[:, None], x_count[:, None]], dim=1)
        features: dict[str, Any] = {"compose_dispersion": []}
        for size in range(2, 6):
            idx = ((batch["w_size"] == size) & (batch["w_rule"] >= 0)).nonzero(as_tuple=True)[0]
            if idx.numel() == 0:
                continue
            c1 = batch["w_child1"][idx]
            c2 = batch["w_child2"][idx]
            z1, z2 = z[c1], z[c2]
            payload = torch.cat([z1 + z2, (z1 - z2).abs(), z1 * z2,
                                 cross_in[idx], batch["w_scal"][idx]], dim=1)
            if payload.shape[1] != self.compose_in:
                raise ValueError(f"compose input {payload.shape[1]} != {self.compose_in}")
            out = self.compose_norm(self.compose(payload))
            z = z.index_copy(0, idx, out)
            with torch.no_grad():
                features["compose_dispersion"].append(
                    float((z[idx] - z[idx].mean(dim=0, keepdim=True)).norm(dim=1).mean()))
        return z, features

    def forward(self, batch: Mapping[str, torch.Tensor], mode: str = "REAL",
                return_parts: bool = False) -> dict[str, Any]:
        n_mol = int(batch["n_molecules"])
        z, aux = self.word_states(batch, mode)
        final = batch["w_final"]
        word_ctx = torch.cat([batch["w_nports"][:, None],
                              (batch["w_nports"] / batch["w_size"].clamp(min=1))[:, None],
                              (batch["w_size"].to(batch["w_ctx"].dtype) / 5.0)[:, None],
                              batch["w_ctx"]], dim=1)
        if word_ctx.shape[1] != self.f_in - self.z_dim:
            raise ValueError("word context width mismatch")
        f_all = self.f_mlp(torch.cat([z, word_ctx], dim=1)).squeeze(1) * final
        words_per_mol = batch["mol_word_ptr"][1:] - batch["mol_word_ptr"][:-1]
        seg_w = torch.arange(n_mol, device=z.device).repeat_interleave(words_per_mol)
        f_sum = f_all.new_zeros(n_mol)
        f_sum.index_add_(0, seg_w, f_all)

        roles = self.port_roles(batch, mode)
        bond_feat = self._bond_features(batch["b_core"], batch["b_stereo"])
        # every inter-word input is built symmetrically in (i, j): the edge
        # term is a function of the unordered pair only
        qi, qj = batch["e_i"], batch["e_j"]
        npi = batch["w_nports"][qi].to(batch["w_ctx"].dtype)
        npj = batch["w_nports"][qj].to(batch["w_ctx"].dtype)
        szi = batch["w_size"][qi].to(batch["w_ctx"].dtype) / 5.0
        szj = batch["w_size"][qj].to(batch["w_ctx"].dtype) / 5.0
        cxt_i, cxt_j = batch["w_ctx"][qi], batch["w_ctx"][qj]
        local = torch.cat([(npi + npj)[:, None], (npi - npj).abs()[:, None],
                           (szi + szj)[:, None], (szi - szj).abs()[:, None],
                           cxt_i + cxt_j, (cxt_i - cxt_j).abs()], dim=1)
        if local.shape[1] != N_ATOM_CTX:
            raise ValueError(f"local context {local.shape[1]} != {N_ATOM_CTX}")
        q_in = torch.cat([z[qi] + z[qj], (z[qi] - z[qj]).abs(), z[qi] * z[qj],
                          roles["e_i"] + roles["e_j"], (roles["e_i"] - roles["e_j"]).abs(),
                          roles["e_i"] * roles["e_j"], bond_feat[batch["e_bond"]], local], dim=1)
        if q_in.shape[1] != self.q_in:
            raise ValueError(f"q input {q_in.shape[1]} != {self.q_in}")
        g = self.q_mlp(q_in)
        q_scalar = self.q_scalar(g).squeeze(1)

        zf = z * final[:, None]
        n_final = zf.new_zeros(n_mol)
        n_final.index_add_(0, seg_w, final)
        z_tot, z_mean, z_std = _seg_stats_masked(zf, seg_w, n_mol, n_final)
        n_inter = batch["mol_inter_ptr"][1:] - batch["mol_inter_ptr"][:-1]
        e_seg = torch.arange(n_mol, device=z.device).repeat_interleave(n_inter)
        if int(batch["n_inter"]):
            g_tot, g_mean, g_std = _seg_stats(g, e_seg, n_mol)
            q_tot = q_scalar.new_zeros(n_mol)
            q_tot.index_add_(0, e_seg, q_scalar)
            f_tot = f_sum
        else:
            g_tot = g.new_zeros((n_mol, self.q_dim))
            g_mean = g_tot
            g_std = g_tot
            q_tot = q_scalar.new_zeros(n_mol)
            f_tot = f_sum
        readout_in = torch.cat([z_tot, z_mean, z_std, g_tot, g_mean, g_std,
                                (n_final / 10.0)[:, None], (n_inter.to(z.dtype) / 10.0)[:, None],
                                batch["graph"]], dim=1)
        if readout_in.shape[1] != self.readout_in:
            raise ValueError(f"readout input {readout_in.shape[1]} != {self.readout_in}")
        correction = self.readout(readout_in).squeeze(1)
        yhat = f_tot + q_tot + correction
        out = {"yhat_std": yhat, "f_sum": f_tot, "q_sum": q_tot,
               "correction": correction, "z": z, "q": q_scalar, "g": g,
               "f": f_all, "n_final": n_final, "aux": aux}
        return out


def build_model(atom_dims: Sequence[int], bond_dims: Sequence[int], seed: int,
                config: Mapping[str, Any]) -> tuple[PrcgModel, dict[str, Any]]:
    dims = dict(config["dims"])
    model = PrcgModel(atom_dims, bond_dims, **dims)
    hashes = model.init_parameters(int(seed))
    return model, {"parameters": model.parameter_ledger(), "init_hashes": hashes,
                   "dims": dims}
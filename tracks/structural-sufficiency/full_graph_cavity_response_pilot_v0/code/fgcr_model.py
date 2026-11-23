"""FGCR-v0 model: shared two-state energy potentials, exact full-graph
junction-tree responses, root-conditioned environment sentence and readout.

The model is deliberately non-iterative.  Its only graph computation is the
exact sum-product inference in ``fgcr_inference``; no learned message passing,
no hidden state recurrence and no auxiliary objective exist.
"""

from __future__ import annotations

import hashlib
import math
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn

import fgcr_common as C
import fgcr_inference as I


def _mlp(dims: Sequence[int], *, dropout: float = 0.0,
         final_activation: bool = False) -> nn.Sequential:
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


class FGCRModel(nn.Module):
    def __init__(self, atom_dims: Sequence[int], bond_dims: Sequence[int], *,
                 channels: int = 16, states: int = 2,
                 atom_emb: int = 16, bond_emb: int = 12,
                 atom_hidden: int = 64, bond_hidden: int = 64,
                 potential_bound: float = 2.0,
                 sentence_hidden: int = 128, h_dim: int = 96,
                 readout_hidden: int = 128, readout_mid: int = 64,
                 dropout: float = 0.1) -> None:
        super().__init__()
        if int(states) != 2:
            raise ValueError("FGCR-v0 fixes two internal states per channel")
        if len(atom_dims) != C.ATOM_COLUMNS or len(bond_dims) != C.BOND_COLUMNS:
            raise ValueError("FGCR-v0 expects 9 atom and 3 bond categorical fields")
        self.channels = int(channels)
        self.states = int(states)
        self.potential_bound = float(potential_bound)

        self.atom_emb = nn.ModuleList([nn.Embedding(int(d), int(atom_emb))
                                       for d in atom_dims])
        self.bond_emb = nn.ModuleList([nn.Embedding(int(d), int(bond_emb))
                                       for d in bond_dims])
        atom_in_dim = int(atom_emb) * len(atom_dims)
        bond_in_dim = int(bond_emb) * len(bond_dims)
        self.atom_mlp = _mlp([atom_in_dim, int(atom_hidden),
                              self.channels * self.states])
        self.bond_mlp = _mlp([bond_in_dim, int(bond_hidden),
                              3 * self.channels])

        self.eta_norm = nn.LayerNorm(self.channels)
        self.sentence_mlp = _mlp([int(atom_emb) * len(atom_dims) + self.channels,
                                  int(sentence_hidden), int(h_dim)],
                                 dropout=float(dropout), final_activation=True)
        self.readout_in_dim = 3 * int(h_dim) + 2 * self.channels + 2
        self.readout_mlp = nn.Sequential(
            nn.Linear(self.readout_in_dim, int(readout_hidden)),
            nn.SiLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(readout_hidden), int(readout_mid)),
            nn.SiLU(),
            nn.Linear(int(readout_mid), 1),
        )
        self.beta = nn.Parameter(torch.zeros(()))

    # -- deterministic seed-0 initialisation --------------------------------
    def init_parameters(self, seed: int) -> str:
        """PyTorch default initialisation under a local, seed-0 RNG state.

        The global RNG state is restored so that checkpoint/verification code
        can build the same model regardless of prior torch operations.
        """
        rng_state = torch.random.get_rng_state()
        try:
            torch.manual_seed(int(seed))
            for module in self.modules():
                reset = getattr(module, "reset_parameters", None)
                if callable(reset) and module is not self:
                    reset()
            with torch.no_grad():
                self.beta.zero_()
        finally:
            torch.random.set_rng_state(rng_state)
        return self.parameter_hash()

    def parameter_hash(self) -> str:
        digest = hashlib.sha256()
        for name, value in sorted(self.state_dict().items()):
            digest.update(name.encode("utf-8"))
            digest.update(str(tuple(value.shape)).encode("utf-8"))
            digest.update(str(value.dtype).encode("utf-8"))
            digest.update(value.detach().cpu().numpy().tobytes())
        return digest.hexdigest()

    def parameter_ledger(self) -> dict[str, int]:
        def n(module: Any) -> int:
            if isinstance(module, nn.Parameter):
                return int(module.numel())
            return int(sum(p.numel() for p in module.parameters()))

        out = {
            "atom_emb": n(self.atom_emb),
            "bond_emb": n(self.bond_emb),
            "atom_mlp": n(self.atom_mlp),
            "bond_mlp": n(self.bond_mlp),
            "eta_norm": n(self.eta_norm),
            "sentence_mlp": n(self.sentence_mlp),
            "readout_mlp": n(self.readout_mlp),
            "beta": n(self.beta),
        }
        out["total"] = int(sum(out.values()))
        return out

    # -- potentials ---------------------------------------------------------
    def potentials(self, batch: Mapping[str, Any]
                   ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        atom_in = torch.cat([emb(batch["atom_feat"][:, i])
                             for i, emb in enumerate(self.atom_emb)], dim=1)
        raw_u = self.atom_mlp(atom_in)
        u = self.potential_bound * torch.tanh(raw_u).view(
            -1, self.channels, self.states)
        bond_in = torch.cat([emb(batch["bond_feat"][:, i])
                             for i, emb in enumerate(self.bond_emb)], dim=1)
        raw_j = self.bond_mlp(bond_in)
        j = self.potential_bound * torch.tanh(raw_j).view(-1, self.channels, 3)
        log_psi = I.edge_log_potential(j)
        return atom_in, u, j, log_psi

    # -- exact responses and whole-molecule prediction ----------------------
    def forward(self, batch: Mapping[str, Any], *, return_aux: bool = False) -> dict[str, Any]:
        atom_in, u, j, log_psi = self.potentials(batch)
        responses = I.exact_responses(u, log_psi, batch["plan"])
        logz = responses["logz"]
        logw = responses["logw"]

        # H(G) = logZ - sum_v logsumexp(u_v)
        unary_lse = torch.logsumexp(u, dim=-1)                     # [N,C]
        unary_sum = u.new_zeros((int(batch["n_molecules"]), self.channels))
        unary_sum.index_add_(0, batch["atom_mol"], unary_lse)
        H = logz - unary_sum                                       # [G,C]

        # eta(v) = exact posterior log-odds minus the unary log-odds
        posterior_logodds = logw[..., 1] - logw[..., 0]            # [N,C]
        unary_logodds = u[..., 1] - u[..., 0]                      # [N,C]
        eta = posterior_logodds - unary_logodds                    # [N,C]

        sentence_in = torch.cat([atom_in, self.eta_norm(eta)], dim=1)
        h = self.sentence_mlp(sentence_in)                         # [N,96]

        n_atoms = batch["graph_n_atoms"].unsqueeze(1)              # [G,1]
        n_bonds = batch["graph_n_bonds"].unsqueeze(1)
        h_sum = h.new_zeros((int(batch["n_molecules"]), h.shape[1]))
        h_sum.index_add_(0, batch["atom_mol"], h)
        h_sumsq = h.new_zeros((int(batch["n_molecules"]), h.shape[1]))
        h_sumsq.index_add_(0, batch["atom_mol"], h * h)
        h_mean = h_sum / n_atoms
        h_var = (h_sumsq / n_atoms - h_mean * h_mean).clamp_min(0.0)
        h_std = torch.sqrt(h_var)

        readout_in = torch.cat(
            [h_sum, h_mean, h_std,
             H, H / (1.0 + n_bonds),
             torch.log1p(n_atoms), torch.log1p(n_bonds)],
            dim=1,
        )
        if readout_in.shape[1] != self.readout_in_dim:
            raise RuntimeError(
                f"readout width {readout_in.shape[1]} != {self.readout_in_dim}")
        yhat_std = self.readout_mlp(readout_in).view(-1) + self.beta

        out: dict[str, Any] = {"yhat_std": yhat_std}
        if return_aux:
            out.update({
                "u": u, "j": j, "log_psi": log_psi,
                "logz": logz, "logw": logw,
                "H": H, "eta": eta,
                "h": h, "readout_in": readout_in,
                "belief": responses["belief"],
            })
        return out


def build_model(atom_dims: Sequence[int], bond_dims: Sequence[int], seed: int,
                config: Mapping[str, Any] | None = None
                ) -> tuple[FGCRModel, dict[str, Any]]:
    cfg = dict((config or {}).get("model", {}))
    model = FGCRModel(
        atom_dims, bond_dims,
        channels=int(cfg.get("channels", 16)),
        states=int(cfg.get("states", 2)),
        atom_emb=int(cfg.get("atom_emb", 16)),
        bond_emb=int(cfg.get("bond_emb", 12)),
        atom_hidden=int(cfg.get("atom_potential_dims", [144, 64, 32])[1]),
        bond_hidden=int(cfg.get("bond_potential_dims", [36, 64, 48])[1]),
        potential_bound=float(cfg.get("potential_bound", 2.0)),
        sentence_hidden=int(cfg.get("sentence_dims", [160, 128, 96])[1]),
        h_dim=int(cfg.get("sentence_dims", [160, 128, 96])[2]),
        readout_hidden=int(cfg.get("readout_dims", [322, 128, 64, 1])[1]),
        readout_mid=int(cfg.get("readout_dims", [322, 128, 64, 1])[2]),
        dropout=float(cfg.get("sentence_dropout", 0.1)),
    )
    init_hash = model.init_parameters(int(seed))
    info = {
        "seed": int(seed),
        "parameters": model.parameter_ledger(),
        "init_hash": init_hash,
        "readout_in_dim": int(model.readout_in_dim),
        "sentence_in_dim": int(model.sentence_mlp[0].in_features),
    }
    return model, info


def load_model_from_state(state: Mapping[str, torch.Tensor],
                          atom_dims: Sequence[int], bond_dims: Sequence[int],
                          config: Mapping[str, Any] | None = None,
                          device: Any = "cpu") -> FGCRModel:
    cfg = dict((config or {}).get("model", {}))
    model = FGCRModel(
        atom_dims, bond_dims,
        channels=int(cfg.get("channels", 16)),
        states=int(cfg.get("states", 2)),
        atom_emb=int(cfg.get("atom_emb", 16)),
        bond_emb=int(cfg.get("bond_emb", 12)),
        atom_hidden=int(cfg.get("atom_potential_dims", [144, 64, 32])[1]),
        bond_hidden=int(cfg.get("bond_potential_dims", [36, 64, 48])[1]),
        potential_bound=float(cfg.get("potential_bound", 2.0)),
        sentence_hidden=int(cfg.get("sentence_dims", [160, 128, 96])[1]),
        h_dim=int(cfg.get("sentence_dims", [160, 128, 96])[2]),
        readout_hidden=int(cfg.get("readout_dims", [322, 128, 64, 1])[1]),
        readout_mid=int(cfg.get("readout_dims", [322, 128, 64, 1])[2]),
        dropout=float(cfg.get("sentence_dropout", 0.1)),
    )
    model.load_state_dict({k: v.to(device) for k, v in state.items()}, strict=True)
    return model.to(device)
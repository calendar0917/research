"""SPC model: Separator-Port Composition (task §3).

    branch summary S_vj (195) = [sum_a / sqrt(1+n_atoms), sum_b / sqrt(1+n_bonds),
                                 log1p n_atoms, log1p n_bonds, cyclomatic_rank]
    port  = Fport([a_v, S_vj])          291 -> 96 -> 48
    pair  = Fpair([pi+pj, |pi-pj|, pi pj, a_v])   240 -> 96 -> 64
    t_v   = sum_{i<j} Fpair / sqrt(1+n_pairs)
    y     = rho_base(g_base) + sum_v rho_event([a_v, t_v, t_v^2, log1p n_pairs])

``MEAN`` replaces the per-pair composition by one composition on the branch-port
mean, scaled by the true pair count.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import xs_models as SM
from xs_models import MLP, SharedSkeleton

ARMS = ("REAL", "MEAN")
PORT_DIM = 48
PAIR_OUT = 64
S_DIM = 2 * SM.WIDTH + 2 + 1                 # 195
PORT_IN = SM.WIDTH + S_DIM                   # 291
PAIR_IN = 3 * PORT_DIM + SM.WIDTH            # 240


class SPCModel(nn.Module):
    def __init__(self, norm: Mapping[str, object], arm: str = "REAL", dropout: float = 0.0):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown SPC arm {arm!r}")
        self.arm = arm
        self.skeleton = SharedSkeleton(norm, dropout)
        self.Fport = MLP([PORT_IN, 96, PORT_DIM], dropout)
        self.Fpair = MLP([PAIR_IN, 96, PAIR_OUT], dropout)

    # -- branch ports --------------------------------------------------------
    def branch_ports(self, batch, a, b) -> torch.Tensor:
        nb = batch["spc_br_natoms"].shape[0]
        if nb == 0:
            return a.new_zeros((0, PORT_DIM))
        a_ptr = batch["spc_br_atom_ptr"]
        a_cnt = a_ptr[1:] - a_ptr[:-1]
        a_owner = torch.repeat_interleave(torch.arange(nb, device=a.device), a_cnt)
        atom_sum = a.new_zeros((nb, SM.WIDTH)).index_add_(
            0, a_owner, a.index_select(0, batch["spc_br_atom_idx"]) *
            batch["spc_br_atom_w"].unsqueeze(1))
        b_ptr = batch["spc_br_bond_ptr"]
        b_cnt = b_ptr[1:] - b_ptr[:-1]
        b_owner = torch.repeat_interleave(torch.arange(nb, device=a.device), b_cnt)
        bond_sum = b.new_zeros((nb, SM.WIDTH)).index_add_(
            0, b_owner, b.index_select(0, batch["spc_br_bond_idx"]) *
            batch["spc_br_bond_w"].unsqueeze(1))
        s = torch.cat([atom_sum, bond_sum,
                       torch.log1p(batch["spc_br_natoms"]).unsqueeze(1),
                       torch.log1p(batch["spc_br_nbonds"]).unsqueeze(1),
                       batch["spc_br_cyc"].unsqueeze(1)], dim=1)
        if s.shape[1] != S_DIM:
            raise RuntimeError(f"branch summary dim {s.shape[1]} != {S_DIM}")
        a_v = a.index_select(0, batch["spc_cv_atom"]).index_select(0, batch["spc_br_cv"])
        return self.Fport(torch.cat([a_v, s], dim=1))

    # -- forward -------------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor], *, return_stats: bool = False):
        a, b = self.skeleton.encode(batch)
        ports = self.branch_ports(batch, a, b)                   # [NB, 48]
        n_cv = batch["spc_cv_atom"].shape[0]
        a_cv = a.index_select(0, batch["spc_cv_atom"])           # [C, 96]
        n_pairs = batch["spc_cv_npairs"].to(a.dtype)             # [C]
        denom = torch.sqrt(1.0 + n_pairs).unsqueeze(1)

        if self.arm == "REAL":
            pair_br = batch["spc_pair_br"]
            pi = ports.index_select(0, pair_br[:, 0])
            pj = ports.index_select(0, pair_br[:, 1])
            a_pair = a_cv.index_select(0, batch["spc_pair_cv"])
            q = torch.cat([pi + pj, (pi - pj).abs(), pi * pj, a_pair], dim=1)
            fp = self.Fpair(q)                                   # [P, 64]
            t = a.new_zeros((n_cv, PAIR_OUT)).index_add_(
                0, batch["spc_pair_cv"], fp) / denom
        else:
            k = a.new_zeros((n_cv,)).index_add_(
                0, batch["spc_br_cv"], torch.ones_like(batch["spc_br_cv"], dtype=a.dtype))
            pbar = a.new_zeros((n_cv, PORT_DIM)).index_add_(
                0, batch["spc_br_cv"], ports) / k.clamp(min=1.0).unsqueeze(1)
            q = torch.cat([2.0 * pbar, torch.zeros_like(pbar), pbar * pbar, a_cv], dim=1)
            fp = self.Fpair(q)                                   # [C, 64]
            t = (n_pairs / torch.sqrt(1.0 + n_pairs)).unsqueeze(1) * fp

        if n_cv == 0:
            ev = a.new_zeros((0,))
        else:
            ev = self.skeleton.event_readout(torch.cat(
                [a_cv, t, t * t, torch.log1p(n_pairs).unsqueeze(1)], dim=1))
        y = self.skeleton.base_readout(batch, a, b)
        y = y + self.skeleton.scatter_events(ev, batch["spc_cv_batch"], batch["n_graphs"])
        if return_stats:
            return y, {"ports": ports, "t": t, "n_pairs": n_pairs, "a": a, "b": b}
        return y


def build_model(norm, seed: int, arm: str = "REAL", *, device="cpu", dropout: float = 0.0,
                ssl_seed: int = 0):
    import xs_common as C

    model = SM.build_seeded(lambda: SPCModel(norm, arm=arm, dropout=dropout), seed)
    ssl_info = SM.load_ssl(model.skeleton, C.ssl_path(ssl_seed), device="cpu")
    model = model.to(device)
    return model, {"ssl": ssl_info, "seed": int(seed), "arm": arm}

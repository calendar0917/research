"""CMA model: Coupled Motif Algebra (task §2).

Fixed parameter shapes (never swept):

    F2      : 484 -> 96 -> 48   (SiLU)
    F3      : 252 -> 96 -> 64   (SiLU)
    rho_event: 225 -> 128 -> 1  (shared skeleton, zero-init output)
    rho_base : 419 -> 128 -> 64 -> 1

``arm`` is a non-parametric switch: both arms instantiate the *same* modules and
therefore have identical parameter counts and (for one seed) identical initial
tensors.  ``REAL`` composes then averages; ``MEAN`` averages each fragment
statistic first and then composes once.
"""

from __future__ import annotations

from typing import Mapping

import torch
import torch.nn as nn
import torch.nn.functional as F

import xs_models as SM
from xs_models import MLP, SharedSkeleton

ARMS = ("REAL", "MEAN")
N_CHORD = 4
Z_DIM = 48
F3_OUT = 64
F2_IN = 3 * SM.WIDTH + 2 * SM.WIDTH + N_CHORD          # 484
F3_IN = 3 * Z_DIM + SM.WIDTH + 3 * N_CHORD             # 252
R_DIM = 3 * N_CHORD                                    # 12


class CMAModel(nn.Module):
    def __init__(self, norm: Mapping[str, object], arm: str = "REAL", dropout: float = 0.0):
        super().__init__()
        if arm not in ARMS:
            raise ValueError(f"unknown CMA arm {arm!r}")
        self.arm = arm
        self.skeleton = SharedSkeleton(norm, dropout)
        self.F2 = MLP([F2_IN, 96, Z_DIM], dropout)
        self.F3 = MLP([F3_IN, 96, F3_OUT], dropout)

    # -- fragments -----------------------------------------------------------
    def pair_fragments(self, batch, a, b) -> torch.Tensor:
        """``z_ijk = 0.5 [F2(i,j,k) + F2(k,j,i)]`` for every used triple."""
        za = batch["cma_z_atom"]
        zb = batch["cma_z_bond"]
        ch = F.one_hot(batch["cma_z_chord"], N_CHORD).to(a.dtype)
        fwd = torch.cat([a[za[:, 1]], a[za[:, 0]], a[za[:, 2]],
                         b[zb[:, 0]], b[zb[:, 1]], ch], dim=1)
        rev = torch.cat([a[za[:, 2]], a[za[:, 0]], a[za[:, 1]],
                         b[zb[:, 1]], b[zb[:, 0]], ch], dim=1)
        out = self.F2(torch.cat([fwd, rev], dim=0))
        T = za.shape[0]
        out = out.view(2, T, Z_DIM)
        return 0.5 * (out[0] + out[1])

    def path_relation(self, batch, dtype) -> torch.Tensor:
        cd = batch["cma_path_chord"]
        oh_ik = F.one_hot(cd[:, 0], N_CHORD).to(dtype)
        oh_jl = F.one_hot(cd[:, 1], N_CHORD).to(dtype)
        oh_il = F.one_hot(cd[:, 2], N_CHORD).to(dtype)
        return torch.cat([oh_ik + oh_jl, (oh_ik - oh_jl).abs(), oh_il], dim=1)

    # -- forward -------------------------------------------------------------
    def forward(self, batch: Mapping[str, torch.Tensor], *, return_stats: bool = False):
        a, b = self.skeleton.encode(batch)
        z = self.pair_fragments(batch, a, b)                       # [T, 48]
        pz = batch["cma_path_z"]
        zL, zR = z[pz[:, 0]], z[pz[:, 1]]                          # [P, 48]
        rel = self.path_relation(batch, a.dtype)                   # [P, 12]
        centre = batch["cma_path_bond"]                            # [P] bond row
        b_centre = b[centre]                                       # [P, 96]
        n_e = batch["cma_event_n"].to(a.dtype)                     # [E]
        n_bonds = b.shape[0]

        if self.arm == "REAL":
            u = torch.cat([zL + zR, (zL - zR).abs(), zL * zR, b_centre, rel], dim=1)
            f3 = self.F3(u)                                        # [P, 64]
            num = b.new_zeros((n_bonds, F3_OUT)).index_add_(0, centre, f3)
            m = num / (2.0 * torch.sqrt(1.0 + n_e)).unsqueeze(1)
        else:
            count = (2.0 * n_e).clamp(min=1.0).unsqueeze(1)
            mzL = b.new_zeros((n_bonds, Z_DIM)).index_add_(0, centre, zL) / count
            mzR = b.new_zeros((n_bonds, Z_DIM)).index_add_(0, centre, zR) / count
            mr = b.new_zeros((n_bonds, R_DIM)).index_add_(0, centre, rel) / count
            u = torch.cat([mzL + mzR, (mzL - mzR).abs(), mzL * mzR,
                           b, mr], dim=1)
            f3 = self.F3(u)                                        # [E, 64]
            scale = (n_e / torch.sqrt(1.0 + n_e)).unsqueeze(1)
            m = scale * f3
            m = torch.where(n_e.unsqueeze(1) > 0, m, torch.zeros_like(m))

        ev = self.skeleton.event_readout(torch.cat(
            [b, m, m * m, torch.log1p(n_e).unsqueeze(1)], dim=1))   # [E]
        y = self.skeleton.base_readout(batch, a, b)
        y = y + self.skeleton.scatter_events(ev, batch["bond_batch"], batch["n_graphs"])
        if return_stats:
            return y, {"z": z, "m": m, "n_e": n_e, "a": a, "b": b}
        return y


def build_model(norm, seed: int, arm: str = "REAL", *, device="cpu", dropout: float = 0.0,
                ssl_seed: int = 0):
    """Construct → SSL-load → move.  Shared modules are seed-deterministic."""
    model = SM.build_seeded(lambda: CMAModel(norm, arm=arm, dropout=dropout), seed)
    ssl_info = SM.load_ssl(model.skeleton, C_ssl_path(ssl_seed), device="cpu")
    model = model.to(device)
    return model, {"ssl": ssl_info, "seed": int(seed), "arm": arm}


def C_ssl_path(seed: int):
    import xs_common as C

    return C.ssl_path(seed)

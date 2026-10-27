"""DCC-L with a per-bond support-conditioned local-composer input.

Only one line changes relative to the frozen DCC-L forward:

    Wl -> alpha * Wl + (1 - alpha) * Wg
    Vl -> alpha * Vl + (1 - alpha) * Vg

``Wg, Vg`` are the model's existing ``group_to_bonds`` mean-field statistics, so
there is no new module, tensor, feature or parameter.  ``alpha`` is a per-bond
float32 vector supplied by the data collate; it is detached (a label-free
constant, not learned).  When ``alpha is None`` the forward is the exact
original DCC-L forward and the L arm is bit-comparable to the frozen model.
"""

from __future__ import annotations

from typing import Mapping

import torch
from torch import nn

import dcc_models as M


class SCBDualChannel(M.DualChannel):
    """DCC-L with optional per-bond alpha shrinkage of the local statistics."""

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        alpha: torch.Tensor | None = None,
    ) -> torch.Tensor:
        a, b, w = self.encoders(batch)
        Wg, Vg, n_b = self.group_to_bonds(w, batch)
        c_glob = self.group_composer(
            torch.cat([b, Wg, Vg, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
        )
        Wl, Vl, _ = self.local_grouping(w, batch, organized=bool(self.organized_local))
        if alpha is not None:
            # alpha is [n_bonds] float32, detached and label-free.
            al = alpha.to(device=Wl.device, dtype=Wl.dtype).unsqueeze(1)
            keep = 1.0 - al
            Wl = al * Wl + keep * Wg
            Vl = al * Vl + keep * Vg
        c_loc = self.local_composer(
            torch.cat([b, Wl, Vl, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]], dim=1)
        )
        B = int(batch["n_graphs"])
        z = c_glob.new_zeros((B, c_glob.shape[1])).index_add_(0, batch["bond_batch"], c_glob)
        zl = c_loc.new_zeros((B, c_loc.shape[1])).index_add_(0, batch["bond_batch"], c_loc)
        zl2 = c_loc.new_zeros((B, c_loc.shape[1])).index_add_(0, batch["bond_batch"], c_loc * c_loc)
        a_sum = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a)
        a_sq = a.new_zeros((B, a.shape[1])).index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global"] - self.global_mean) / self.global_scale
        return self.reader(torch.cat([z, a_sum, a_sq, g, zl, zl2], dim=1)).view(-1)


def build_scb(seed: int, norm: Mapping[str, object]) -> tuple[SCBDualChannel, dict]:
    """Create an SCB model with exactly the frozen DCC-L initial state.

    The base dual state is built once by ``dcc_models.build_initial_state``
    (identical to the frozen L/C constructor, including the P3 encoder load),
    then copied into the SCB subclass without any new RNG consumption that can
    affect weights.
    """
    base, info = M.build_initial_state(norm, int(seed))
    model = SCBDualChannel(norm)
    missing, unexpected = model.load_state_dict(base.state_dict(), strict=True)
    if missing or unexpected:
        raise RuntimeError(f"state mismatch: {missing} {unexpected}")
    return model, info


def param_count(model: nn.Module) -> int:
    return int(sum(p.numel() for p in model.parameters()))

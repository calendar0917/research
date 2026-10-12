"""CBIC — Cross-Bond Interaction Composer (cscl-bjoc-cross-bond-interaction-v1).

Pre-registration: ``configs/bjoc_cross_bond_interaction_v1.yaml``.

One-shot, non-iterative predictor.  This module keeps the published BJOC
backbone (``structural_relation_composer_v1.BJOC``) exactly as it is and adds
exactly ONE new input channel to the existing bond composer:

For every physical bond ``e = (u, v)`` split the junctions that touch ``e``
by the endpoint they are centred on::

    J_u(e) = {junctions j touching e with centre u}
    J_v(e) = {junctions j touching e with centre v}

    mu_u = sum_{j in J_u(e)} w_j / max(n_u, 1)
    mu_v = sum_{j in J_v(e)} w_j / max(n_v, 1)
    q_e  = sqrt(96) * (mu_u * mu_v)          (element-wise product, 96 dims)
    q_e  = 0                                  if n_u == 0 or n_v == 0

``q_e`` is appended to the existing composer input
``[b_e, W_e, V_e, log1p(n_e), bond_feat_e]``.  The scale ``sqrt(96)`` is
frozen before any training and is never tuned.

Arms (see the protocol):

* ``N`` — null incidence grouping, ``q = 0``
* ``O`` — true (organized) incidence grouping, ``q = 0``
* ``P`` — true grouping, real ``q``

All three arms share **identical initial parameters** (the new composer
columns are zero-initialised) and identical training order per seed.

There is no message passing, attention, Transformer block, recurrent update,
learned bilinear matrix, or rank search.  ``q`` is a single fixed
symmetric function of the two per-side junction means.
"""

from __future__ import annotations

import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import structural_relation_composer_v1 as src  # noqa: E402
import structural_relation_objects_v1 as sro  # noqa: E402

#: frozen pre-training scale; never changed after the protocol freeze
INTERACTION_SCALE = math.sqrt(96.0)

ARM_N = "N"
ARM_O = "O"
ARM_P = "P"
ARMS = (ARM_N, ARM_O, ARM_P)

#: arm -> (organization_mode, use_interaction)
ARM_SPEC = {
    ARM_N: (src.NULL, False),
    ARM_O: (src.ORGANIZED, False),
    ARM_P: (src.ORGANIZED, True),
}


@dataclass(frozen=True)
class CBICConfig:
    """Frozen widths: identical to :class:`src.BJOCConfig` plus the 96-D
    cross-bond interaction channel (protocol §B2)."""

    atom_hidden: int = 128
    atom_out: int = 96
    bond_hidden: int = 160
    bond_out: int = 96
    junc_hidden: int = 160
    junc_out: int = 96
    compose_hidden: int = 160
    compose_out: int = 96
    reader_hidden: int = 256
    reader_layers: int = 2
    dropout: float = 0.0
    interaction_dim: int = 96

    @property
    def atom_in(self) -> int:
        return sro.ATOM_FEAT_DIM

    @property
    def bond_in(self) -> int:
        return 3 * self.atom_out + sro.BOND_FEAT_DIM

    @property
    def junc_in(self) -> int:
        return 4 * self.atom_out + 2 * self.bond_out + sro.JUNC_FEAT_DIM

    @property
    def base_compose_in(self) -> int:
        return self.bond_out + 2 * self.junc_out + 1 + sro.BOND_FEAT_DIM

    @property
    def compose_in(self) -> int:
        return self.base_compose_in + self.interaction_dim

    @property
    def reader_in(self) -> int:
        return self.compose_out + 2 * self.atom_out + sro.GLOBAL_DIM

    def inventory(self) -> dict[str, int]:
        def lin(i: int, o: int) -> int:
            return i * o + o

        atom = lin(self.atom_in, self.atom_hidden) + lin(self.atom_hidden, self.atom_out)
        bond = lin(self.bond_in, self.bond_hidden) + lin(self.bond_hidden, self.bond_out)
        junc = lin(self.junc_in, self.junc_hidden) + lin(self.junc_hidden, self.junc_out)
        compose = lin(self.compose_in, self.compose_hidden) + lin(self.compose_hidden, self.compose_out)
        reader = lin(self.reader_in, self.reader_hidden)
        for _ in range(self.reader_layers - 1):
            reader += lin(self.reader_hidden, self.reader_hidden)
        reader += lin(self.reader_hidden, 1)
        return {
            "atom_encoder": atom,
            "bond_encoder": bond,
            "junction_encoder": junc,
            "group_composer": compose,
            "reader": reader,
            "total": atom + bond + junc + compose + reader,
        }


class CBIC(src.BJOC):
    """Bond-Junction Organized Composer + one cross-bond interaction channel."""

    def __init__(
        self,
        config: CBICConfig,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
        organization_mode: str = src.ORGANIZED,
        use_interaction: bool = True,
    ):
        if config.interaction_dim != config.junc_out:
            raise ValueError("this round fixes interaction_dim == junc_out (96)")
        super().__init__(config, global_mean, global_scale, organization_mode)
        self.use_interaction = bool(use_interaction)

    # -- side decomposition of the junction multiset per physical bond ------
    def bond_side_assignment(self, batch: Mapping[str, torch.Tensor]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Flat (junction_id, bond_id, side) for every junction/bond incidence.

        ``side == 0`` means the junction centre is the bond's ``u`` endpoint
        (``bond_index[:, 0]``), ``side == 1`` means it is ``v``.  Each junction
        contributes exactly two rows (one per touched bond).
        """
        jb = batch["junc_bond"]                       # [n_junc, 2]
        centers = batch["junc_index"][:, 1]           # [n_junc]
        n_j = int(jb.shape[0])
        junc_ids = torch.arange(n_j, device=jb.device).repeat_interleave(2)
        bond_ids = jb.reshape(-1)
        center_per_row = centers.repeat_interleave(2)
        ends = batch["bond_index"][bond_ids]          # [2*n_junc, 2]
        is_u = center_per_row == ends[:, 0]
        is_v = center_per_row == ends[:, 1]
        if not bool((is_u | is_v).all()):
            raise RuntimeError("junction centre is not an endpoint of its touched bond")
        side = torch.where(is_u, torch.zeros_like(bond_ids), torch.ones_like(bond_ids))
        return junc_ids, bond_ids, side

    def side_statistics(
        self, w: torch.Tensor, batch: Mapping[str, torch.Tensor]
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Return (s_u, s_v, n_u, n_v): per-bond sums and counts per side."""
        junc_ids, bond_ids, side = self.bond_side_assignment(batch)
        n_bonds = int(batch["bond_feat"].shape[0])
        dim = int(w.shape[1])
        s_u = w.new_zeros(n_bonds, dim)
        s_v = w.new_zeros(n_bonds, dim)
        n_u = w.new_zeros(n_bonds)
        n_v = w.new_zeros(n_bonds)
        mask_u = side == 0
        mask_v = ~mask_u
        s_u.index_add_(0, bond_ids[mask_u], w[junc_ids[mask_u]])
        s_v.index_add_(0, bond_ids[mask_v], w[junc_ids[mask_v]])
        ones = torch.ones_like(bond_ids, dtype=w.dtype)
        n_u.index_add_(0, bond_ids[mask_u], ones[mask_u])
        n_v.index_add_(0, bond_ids[mask_v], ones[mask_v])
        return s_u, s_v, n_u, n_v

    def cross_bond_q(self, w: torch.Tensor, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """The frozen cross-bond channel: ``sqrt(D) * mu_u * mu_v`` (zeros
        when either side has no junction)."""
        s_u, s_v, n_u, n_v = self.side_statistics(w, batch)
        mu_u = s_u / n_u.clamp(min=1.0).unsqueeze(1)
        mu_v = s_v / n_v.clamp(min=1.0).unsqueeze(1)
        q = INTERACTION_SCALE * (mu_u * mu_v)
        valid = ((n_u > 0) & (n_v > 0)).unsqueeze(1)
        return torch.where(valid, q, torch.zeros_like(q))

    # -- forward ------------------------------------------------------------
    def forward_features(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        mode = self.organization_mode if organization_mode is None else organization_mode
        a = self.encode_atoms(batch)
        b = self.encode_bonds(a, batch)
        w = self.encode_junctions(a, b, batch)
        W, V, n_b = self.group_to_bonds(w, batch, mode)
        if self.use_interaction:
            q = self.cross_bond_q(w, batch)
        else:
            q = w.new_zeros(int(batch["bond_feat"].shape[0]), self.config.interaction_dim)
        compose_in = torch.cat(
            [b, W, V, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"], q],
            dim=1,
        )
        c = self.group_composer(compose_in)
        n_graphs = int(batch["n_graphs"])
        z = torch.zeros((n_graphs, c.shape[1]), device=c.device, dtype=c.dtype)
        z.index_add_(0, batch["bond_batch"], c)
        a_sum = torch.zeros((n_graphs, a.shape[1]), device=a.device, dtype=a.dtype)
        a_sum.index_add_(0, batch["atom_batch"], a)
        a_sq = torch.zeros_like(a_sum)
        a_sq.index_add_(0, batch["atom_batch"], a * a)
        g = (batch["global_feat"].to(a.dtype) - self.global_mean) / self.global_scale
        reader_in = torch.cat([z, a_sum, a_sq, g], dim=1)
        return {
            "a": a,
            "b": b,
            "w": w,
            "W": W,
            "V": V,
            "n_b": n_b,
            "q": q,
            "c": c,
            "z": z,
            "reader_in": reader_in,
        }


# ---------------------------------------------------------------------------
# builders / audits
# ---------------------------------------------------------------------------


def build_cbic(
    config: CBICConfig,
    *,
    seed: int,
    organization_mode: str,
    use_interaction: bool,
    global_mean: np.ndarray,
    global_scale: np.ndarray,
    device: torch.device | str = "cpu",
) -> CBIC:
    """Deterministic builder with a matched initial parameter state.

    The published BJOC backbone is built first with the *published*
    ``BJOCConfig``; every shared tensor is then copied bit-identically into the
    extended model, and the 96 new composer input columns are zero-initialised.
    Result: identical initial parameters for every arm (and therefore identical
    initial predictions for O and P, whose only difference is the q value).
    """
    torch.manual_seed(int(seed))
    base = src.BJOC(src.BJOCConfig(), global_mean, global_scale, organization_mode)
    torch.manual_seed(int(seed))
    model = CBIC(config, global_mean, global_scale, organization_mode, use_interaction)
    base_sd = base.state_dict()
    new_sd: dict[str, torch.Tensor] = {}
    for key, target in model.state_dict().items():
        if key == "group_composer.lins.0.weight":
            source = base_sd[key]
            expanded = torch.zeros_like(target)
            expanded[:, : source.shape[1]] = source.clone()
            new_sd[key] = expanded
        elif key in base_sd:
            source = base_sd[key]
            if source.shape != target.shape:
                raise RuntimeError(f"shape mismatch for {key}: {source.shape} vs {target.shape}")
            new_sd[key] = source.clone()
        else:
            raise KeyError(f"no base parameter for {key}")
    model.load_state_dict(new_sd, strict=True)
    return model.to(device)


def build_arm(arm: str, *, seed: int, global_mean: np.ndarray, global_scale: np.ndarray,
              device: torch.device | str = "cpu") -> CBIC:
    if arm not in ARM_SPEC:
        raise ValueError(arm)
    mode, use_q = ARM_SPEC[arm]
    return build_cbic(
        CBICConfig(),
        seed=seed,
        organization_mode=mode,
        use_interaction=use_q,
        global_mean=global_mean,
        global_scale=global_scale,
        device=device,
    )


def state_dict_hash(model: nn.Module) -> str:
    return src.state_dict_hash(model)


def parameter_inventory(model: nn.Module) -> dict[str, Any]:
    config = model.config
    expected = config.inventory()
    actual = int(sum(p.numel() for p in model.parameters()))
    base_expected = src.BJOCConfig().inventory()
    return {
        "expected": expected,
        "actual_parameters": actual,
        "parameter_exact": actual == expected["total"],
        "base_total": base_expected["total"],
        "interaction_parameters_added": actual - base_expected["total"],
        "interaction_parameters_expected": config.interaction_dim * config.compose_hidden,
    }


def forbidden_module_scan(model: nn.Module) -> list[str]:
    return src.forbidden_module_scan(model)

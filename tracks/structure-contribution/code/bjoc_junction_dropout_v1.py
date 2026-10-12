"""BJOC junction-dropout v1 — the single authorised model change of round
``cscl-bjoc-junction-regularization-v1``.

The round has exactly ONE predictive intervention: an elementwise inverted
dropout applied to the learned junction embedding ``w`` **before** the
static incidence grouping consumes it::

    a = self.encode_atoms(batch)
    b = self.encode_bonds(a, batch)
    w = self.encode_junctions(a, b, batch)

    w_used = F.dropout(w, p=self.junction_dropout_p,
                       training=self.training)          # <-- new

    W, V, n_b = self.group_to_bonds(w_used, batch, mode)

Everything else is the frozen BJOC-Null model
(:mod:`structural_relation_composer_v1`): 108-dim atom input, the original raw
bond/junction features, the original widths, the null grouping
(``W_b = n_b * mean(w_used)``, ``V_b = n_b * mean(w_used**2)``), the same
composer and reader, Adam, raw-y L1, last-5 soup.  The dropout happens before
both moments are formed, so it perturbs ``mean(w)`` and ``mean(w**2)`` — the
intervention is *not* a faithful simulation of deleting whole junction objects.

Inference semantics: ``model.eval()`` makes ``w_used = w`` exactly, so two
models that share a weight state produce identical predictions regardless of
``junction_dropout_p``.  No inference-time capacity is added.

R0 = ``p = 0.00`` (exactly the original model; ``F.dropout(p=0)`` is the
identity and consumes no RNG) and R1 = ``p = 0.10``.  No other probability is
authorised and no other dropout location exists in this module.
"""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

_CODE_DIR = Path(__file__).resolve().parent
REPO_ROOT = _CODE_DIR.parents[2]
for _p in (str(_CODE_DIR), str(REPO_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import structural_relation_composer_v1 as src  # noqa: E402

#: frozen intervention probabilities (one arm each; nothing else may be tried)
R0_P = 0.0
R1_P = 0.10
AUTHORISED_P = (R0_P, R1_P)

#: the published BJOC backbone parameter count; both arms must match exactly
PUBLISHED_PARAMETERS = 413505


@dataclass(frozen=True)
class JunctionDropoutConfig(src.BJOCConfig):
    """Frozen BJOC-Null widths plus the single new scalar (no new parameters)."""

    junction_dropout_p: float = R0_P

    def __post_init__(self) -> None:
        if float(self.junction_dropout_p) not in AUTHORISED_P:
            raise ValueError(
                f"junction_dropout_p={self.junction_dropout_p!r} is not one of the "
                f"frozen values {AUTHORISED_P}; this round forbids a probability search"
            )
        # the base config's generic dropout stays exactly zero: the only
        # regularisation is the junction-specific one above
        if float(self.dropout) != 0.0:
            raise ValueError("the frozen BJOC dropout field must stay 0.0")


class JunctionDropoutBJOC(src.BJOC):
    """BJOC-Null with elementwise dropout on the learned junction embedding ``w``.

    The intervention sits between ``encode_junctions`` and ``group_to_bonds``.
    ``w`` (pre-dropout) is still returned for mechanism diagnostics;
    ``w_used`` is the tensor actually consumed by the grouping.
    """

    def __init__(
        self,
        config: JunctionDropoutConfig,
        global_mean: np.ndarray,
        global_scale: np.ndarray,
        organization_mode: str = src.NULL,
    ):
        super().__init__(config, global_mean, global_scale, organization_mode)
        self.junction_dropout_p = float(config.junction_dropout_p)

    # -- the one changed method --------------------------------------------
    def forward_features(
        self,
        batch: Mapping[str, torch.Tensor],
        organization_mode: str | None = None,
    ) -> dict[str, torch.Tensor]:
        mode = self.organization_mode if organization_mode is None else organization_mode
        a = self.encode_atoms(batch)
        b = self.encode_bonds(a, batch)
        w = self.encode_junctions(a, b, batch)

        # the single authorised intervention: elementwise inverted dropout on
        # the learned junction representation, before mean(w) and mean(w**2)
        w_used = F.dropout(w, p=self.junction_dropout_p, training=self.training)

        W, V, n_b = self.group_to_bonds(w_used, batch, mode)
        compose_in = torch.cat(
            [b, W, V, torch.log1p(n_b).unsqueeze(1), batch["bond_feat"]],
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
            "w_used": w_used,
            "W": W,
            "V": V,
            "n_b": n_b,
            "c": c,
            "z": z,
            "reader_in": reader_in,
        }


# ---------------------------------------------------------------------------
# builders / audits
# ---------------------------------------------------------------------------


def build_model(
    *,
    seed: int,
    junction_dropout_p: float,
    global_mean: np.ndarray,
    global_scale: np.ndarray,
    organization_mode: str = src.NULL,
    device: torch.device | str = "cpu",
) -> JunctionDropoutBJOC:
    """Deterministic builder: identical seed+config -> identical parameters.

    ``junction_dropout_p`` never touches initialisation (verified by the
    R0/R1 init-state hash gate)."""
    p = float(junction_dropout_p)
    if p not in AUTHORISED_P:
        raise ValueError(f"unauthorised junction_dropout_p {p!r}")
    torch.manual_seed(int(seed))
    cfg = JunctionDropoutConfig(junction_dropout_p=p)
    model = JunctionDropoutBJOC(cfg, global_mean, global_scale, organization_mode)
    return model.to(device)


def parameter_inventory() -> dict[str, Any]:
    cfg = JunctionDropoutConfig(junction_dropout_p=0.0)
    inv = dict(cfg.inventory())
    inv["arm"] = "JunctionDropoutBJOC"
    inv["published_bjoc_parameters"] = PUBLISHED_PARAMETERS
    inv["parameter_exact_published"] = int(inv["total"]) == PUBLISHED_PARAMETERS
    return inv


def parameter_audit(model: nn.Module) -> dict[str, Any]:
    actual = int(sum(p.numel() for p in model.parameters()))
    return {
        "expected": parameter_inventory()["total"],
        "actual_parameters": actual,
        "parameter_exact": actual == PUBLISHED_PARAMETERS,
    }


def trainable_parameter_names() -> list[str]:
    return sorted(name for name, _ in JunctionDropoutBJOC(
        JunctionDropoutConfig(junction_dropout_p=0.0),
        np.zeros(src.sro.GLOBAL_DIM, dtype=np.float32),
        np.ones(src.sro.GLOBAL_DIM, dtype=np.float32),
        src.NULL,
    ).named_parameters())


def state_dict_hash(model: nn.Module) -> str:
    """Same audited hash as the frozen backbone (buffers excluded)."""
    return src.state_dict_hash(model)

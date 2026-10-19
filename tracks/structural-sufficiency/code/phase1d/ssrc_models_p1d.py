"""Phase 1d models: an additive node--incidence control and fixed capacity arms.

``R_incidence`` keeps the exact Phase 1b/1c E2 patch trunk, true root edges,
readout modules, size features and training recipe, and changes only the edge
relation function to the strictly additive form

    r_uv = psi(h_u, e_uv) + psi(h_v, e_uv)

on each physical undirected bond.  ``psi`` is a shared two-layer MLP applied
to one endpoint token and the bond feature; it never sees the other endpoint
token.  The edge aggregate is the same mean/sum over true bonds used by
``R1_size``.  There is no ``h_u*h_v``, ``|h_u-h_v|`` or ``MLP(h_u+h_v)`` term.

``R1_size_cap`` and ``R_incidence_cap`` are the single pre-registered
capacity configurations from ``notes/phase1d_protocol.md``.  They scale only
the hidden width and its proportional derived widths; depth, radius, readout
rule, data and schedule are unchanged.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from ssrc_models import ModelConfig, RegressionHead, SharedTrunk
from ssrc_models_p1c import (
    EdgeReadout,
    P1CConfig,
    SizeAwareNodeReadout,
    _node_pools,
    build_r1_size,
)


@dataclass(frozen=True)
class P1DConfig(P1CConfig):
    """P1C readout config plus the incidence relation width."""

    incidence_rel_hidden: int = 251


class AdditiveIncidenceRelation(nn.Module):
    """Shared single-endpoint MLP ``psi(h_v, e_uv)``; no joint endpoint input."""

    def __init__(self, hidden: int, edge_dim: int, rel_hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(hidden + edge_dim, rel_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(rel_hidden, hidden),
        )

    def forward(self, h: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        return self.net(torch.cat([h, edge_attr], dim=-1))


def _edge_pools_unique(
    r: torch.Tensor,
    edge_mol: torch.Tensor,
    n_mol: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if r.numel() == 0:
        z = torch.zeros((n_mol, r.shape[-1]), dtype=r.dtype, device=r.device)
        return z, z.clone(), torch.zeros((n_mol, 1), dtype=r.dtype, device=r.device)
    mean = scatter(r, edge_mol, dim=0, dim_size=n_mol, reduce="mean")
    total = scatter(r, edge_mol, dim=0, dim_size=n_mol, reduce="sum")
    has = scatter(
        torch.ones_like(edge_mol, dtype=r.dtype),
        edge_mol,
        dim=0,
        dim_size=n_mol,
        reduce="amax",
    ).unsqueeze(-1)
    return mean, total, has


class ModelRIncidence(nn.Module):
    """True-incidence additive relation + the R1_size readout rule."""

    def __init__(self, node_dim: int, edge_dim: int, cfg: P1DConfig) -> None:
        super().__init__()
        self.trunk = SharedTrunk(node_dim, edge_dim, cfg)
        self.rel = AdditiveIncidenceRelation(
            cfg.hidden, edge_dim, int(cfg.incidence_rel_hidden), cfg.dropout
        )
        self.readout = SizeAwareNodeReadout(cfg.hidden, cfg.size_dim, cfg.dropout)
        self.edge_readout = EdgeReadout(cfg.hidden, cfg.edge_readout_hidden, cfg.dropout)
        self.head = RegressionHead(cfg.hidden, cfg.hidden, cfg.dropout)

    def edge_aggregate(self, token: torch.Tensor, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """Edge mean/sum over one copy of each true physical bond.

        Returns ``(edge_mean, edge_sum, has_edge, r_unique)``.  The relation
        value is computed once per physical bond as two additive endpoint
        terms; the same shared ``psi`` is used for both slots.
        """
        src, dst = batch["root_edge_index"]
        unique = src < dst
        src_u = src[unique]
        dst_u = dst[unique]
        e_u = batch["root_edge_attr"][unique]
        r = self.rel(token[src_u], e_u) + self.rel(token[dst_u], e_u)
        n_mol = int(batch["n_molecules"])
        edge_mol = batch["root_batch"][src_u]
        edge_mean, edge_sum, has_edge = _edge_pools_unique(r, edge_mol, n_mol)
        return edge_mean, edge_sum, has_edge, r

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token = self.trunk(batch)
        edge_mean, edge_sum, has_edge, _ = self.edge_aggregate(token, batch)
        node_mean, node_sum = _node_pools(token, batch)
        x = self.readout(node_mean, node_sum, batch["mol_size_feat"])
        x = x + self.edge_readout(edge_mean, edge_sum) * has_edge
        return self.head(x)


def r1_size_edge_aggregate(model: nn.Module, token: torch.Tensor, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Frozen R1_size edge aggregate, exposed for the mechanism CPU gate.

    This is the exact Phase 1c computation: apply ``phi`` to both directed
    copies of every bond, then keep one copy per physical bond, then mean/sum.
    """
    r = model.rel(token, batch["root_edge_index"], batch["root_edge_attr"])
    src, dst = batch["root_edge_index"]
    unique = src < dst
    n_mol = int(batch["n_molecules"])
    edge_mol = batch["root_batch"][src[unique]]
    selected = r[unique]
    edge_mean, edge_sum, has_edge = _edge_pools_unique(selected, edge_mol, n_mol)
    return edge_mean, edge_sum, has_edge, selected


# ---------------------------------------------------------------------------
# frozen Phase 1d configurations
# ---------------------------------------------------------------------------


def base_incidence_config() -> P1DConfig:
    return P1DConfig()


def r1_size_capacity_config() -> P1DConfig:
    return P1DConfig(
        hidden=128,
        ffn_hidden=256,
        rel_hidden=256,
        edge_readout_hidden=128,
        graph_ffn_hidden=384,
        incidence_rel_hidden=251,
    )


def incidence_capacity_config() -> P1DConfig:
    return P1DConfig(
        hidden=136,
        ffn_hidden=272,
        rel_hidden=272,
        edge_readout_hidden=136,
        graph_ffn_hidden=408,
        incidence_rel_hidden=272,
    )


def build_r_incidence(atom_dim: int, edge_dim: int, cfg: P1DConfig | None = None) -> nn.Module:
    return ModelRIncidence(atom_dim + 4, edge_dim, cfg or base_incidence_config())


def build_r1_size_cap(atom_dim: int, edge_dim: int, cfg: P1DConfig | None = None) -> nn.Module:
    return build_r1_size(atom_dim, edge_dim, cfg or r1_size_capacity_config())


def build_r_incidence_cap(atom_dim: int, edge_dim: int, cfg: P1DConfig | None = None) -> nn.Module:
    return ModelRIncidence(atom_dim + 4, edge_dim, cfg or incidence_capacity_config())


def register_phase1d_models() -> None:
    from ssrc_models import (
        RELATIONAL_SIZE_KEYS,
        assert_relational_size_batch,
        register_builder,
    )
    from ssrc_train import register_model_keys

    for name, builder in [
        ("R_incidence", build_r_incidence),
        ("R1_size_cap", build_r1_size_cap),
        ("R_incidence_cap", build_r_incidence_cap),
    ]:
        register_builder(name, builder)
        register_model_keys(name, RELATIONAL_SIZE_KEYS, assert_relational_size_batch)

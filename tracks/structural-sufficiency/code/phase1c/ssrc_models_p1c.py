"""Phase 1c readout/correspondence controls.

Three required arms
-------------------
* ``S_cap_size``      -- the Phase 1b S0 local E2 patch encoder, but the graph
  readout uses node mean, node sum and an explicit graph-size feature, plus a
  real graph-level residual FFN.  No root adjacency is used.
* ``R1_size``         -- the Phase 1b true-endpoint relation MLP
  ``phi(h_u, h_v, e_uv)`` on the real root edges, with the same node/size
  readout rule plus a real edge-relation aggregate.
* ``R1_marginal_size``-- exactly the same modules, parameters and readout as
  ``R1_size``, but every edge sees the molecule's global mean node token in
  both endpoint slots.  This keeps the E2 patch representation, the true bond
  type multiset, edge count and the relation-branch computation, while removing
  which two root tokens are attached to the edge.  It is an endpoint-
  correspondence ablation, not a chemical causal experiment.

Optional later diagnostic
-------------------------
``M_keep_edge`` keeps R1's direct edge readout and adds one tied node-update
round.  It is implemented and CPU-tested but is *not* part of the Phase 1c
seed-0 first round; the protocol buys it only under the pre-registered
follow-up gate.

Capacity matching
-----------------
S_cap_size and R1_size have identical trunk, identical node/size readout and
identical head.  R1_size pays for the relation MLP (33,472 params) and a real
edge-readout FFN (12,672).  S_cap_size spends the same budget on two real
graph-level residual FFNs (46,050 at width 177).  The remaining difference is
94 parameters (<0.1%); it is reported, not hidden.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch_geometric.utils import scatter

from ssrc_models import ModelConfig, RegressionHead, RelationMLP, SharedTrunk


@dataclass(frozen=True)
class P1CConfig(ModelConfig):
    """ModelConfig plus the Phase 1c readout widths."""

    size_dim: int = 3
    graph_ffn_hidden: int = 177
    graph_ffn_blocks: int = 2
    edge_readout_hidden: int = 64


def _node_pools(token: torch.Tensor, batch: Mapping[str, Any]) -> tuple[torch.Tensor, torch.Tensor]:
    n_mol = int(batch["n_molecules"])
    mean = scatter(token, batch["patch_to_mol"], dim=0, dim_size=n_mol, reduce="mean")
    total = scatter(token, batch["patch_to_mol"], dim=0, dim_size=n_mol, reduce="sum")
    return mean, total


class SizeAwareNodeReadout(nn.Module):
    """Readout over node mean, node sum and explicit graph-size features."""

    def __init__(self, hidden: int, size_dim: int, dropout: float) -> None:
        super().__init__()
        d_in = 2 * hidden + size_dim
        self.norm = nn.LayerNorm(d_in)
        self.net = nn.Sequential(
            nn.Linear(d_in, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )

    def forward(self, node_mean: torch.Tensor, node_sum: torch.Tensor, size_feat: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(torch.cat([node_mean, node_sum, size_feat], dim=-1)))


class EdgeReadout(nn.Module):
    """Readout over the mean and sum of per-edge relation vectors."""

    def __init__(self, hidden: int, edge_hidden: int, dropout: float) -> None:
        super().__init__()
        d_in = 2 * hidden
        self.norm = nn.LayerNorm(d_in)
        self.net = nn.Sequential(
            nn.Linear(d_in, edge_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(edge_hidden, hidden),
        )

    def forward(self, edge_mean: torch.Tensor, edge_sum: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(torch.cat([edge_mean, edge_sum], dim=-1)))


class GraphResidualFFN(nn.Module):
    """Residual per-graph FFN; real capacity used by S_cap_size."""

    def __init__(self, hidden: int, ffn_hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.net = nn.Sequential(
            nn.Linear(hidden, ffn_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(ffn_hidden, hidden),
        )
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.dropout(self.net(self.norm(x)))


def relation_from_pair(
    rel: RelationMLP,
    h_s: torch.Tensor,
    h_d: torch.Tensor,
    edge_attr: torch.Tensor,
) -> torch.Tensor:
    """Apply the exact same RelationMLP to precomputed endpoint tokens.

    ``RelationMLP.forward`` computes ``x[src], x[dst]`` and then the symmetric
    combination.  Splitting the endpoint lookup out lets the marginal arm pass
    the molecule-level mean token in both slots without changing the module or
    its parameters.
    """
    sym = torch.cat([h_s + h_d, (h_s - h_d).abs(), h_s * h_d, edge_attr], dim=-1)
    return rel.net(sym)


def _edge_pools(
    r: torch.Tensor,
    edge_mol: torch.Tensor,
    unique: torch.Tensor,
    n_mol: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    selected = r[unique]
    selected_mol = edge_mol[unique]
    edge_mean = scatter(selected, selected_mol, dim=0, dim_size=n_mol, reduce="mean")
    edge_sum = scatter(selected, selected_mol, dim=0, dim_size=n_mol, reduce="sum")
    has_edge = scatter(
        torch.ones_like(selected_mol, dtype=torch.float32),
        selected_mol,
        dim=0,
        dim_size=n_mol,
        reduce="amax",
    )
    return edge_mean, edge_sum, has_edge


class ModelSCapSize(nn.Module):
    """Shared local patch model; no root adjacency; size-aware graph readout."""

    def __init__(self, node_dim: int, edge_dim: int, cfg: P1CConfig) -> None:
        super().__init__()
        self.trunk = SharedTrunk(node_dim, edge_dim, cfg)
        self.readout = SizeAwareNodeReadout(cfg.hidden, cfg.size_dim, cfg.dropout)
        self.graph_ffns = nn.ModuleList(
            [GraphResidualFFN(cfg.hidden, cfg.graph_ffn_hidden, cfg.dropout) for _ in range(cfg.graph_ffn_blocks)]
        )
        self.head = RegressionHead(cfg.hidden, cfg.hidden, cfg.dropout)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token = self.trunk(batch)
        node_mean, node_sum = _node_pools(token, batch)
        x = self.readout(node_mean, node_sum, batch["mol_size_feat"])
        for ffn in self.graph_ffns:
            x = ffn(x)
        return self.head(x)


class ModelR1Size(nn.Module):
    """True-endpoint one-shot relation + additively count-aware readout."""

    def __init__(self, node_dim: int, edge_dim: int, cfg: P1CConfig) -> None:
        super().__init__()
        self.trunk = SharedTrunk(node_dim, edge_dim, cfg)
        self.rel = RelationMLP(cfg.hidden, edge_dim, cfg.rel_hidden, cfg.dropout)
        self.readout = SizeAwareNodeReadout(cfg.hidden, cfg.size_dim, cfg.dropout)
        self.edge_readout = EdgeReadout(cfg.hidden, cfg.edge_readout_hidden, cfg.dropout)
        self.head = RegressionHead(cfg.hidden, cfg.hidden, cfg.dropout)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token = self.trunk(batch)
        n_mol = int(batch["n_molecules"])
        r = self.rel(token, batch["root_edge_index"], batch["root_edge_attr"])
        src = batch["root_edge_index"][0]
        dst = batch["root_edge_index"][1]
        unique = src < dst
        edge_mean, edge_sum, has_edge = _edge_pools(r, batch["root_batch"][src], unique, n_mol)
        node_mean, node_sum = _node_pools(token, batch)
        x = self.readout(node_mean, node_sum, batch["mol_size_feat"])
        x = x + self.edge_readout(edge_mean, edge_sum) * has_edge.unsqueeze(-1)
        return self.head(x)


class ModelR1MarginalSize(nn.Module):
    """Same relation/readout as R1_size, but no endpoint correspondence.

    Every directed edge uses the molecule's global mean node token in both
    endpoint slots.  The model never receives endpoint indices or root-batch
    node ids; it sees only the patch tokens, per-edge bond attributes, the
    molecule id of each directed edge, and the unique-edge mask.
    """

    def __init__(self, node_dim: int, edge_dim: int, cfg: P1CConfig) -> None:
        super().__init__()
        self.trunk = SharedTrunk(node_dim, edge_dim, cfg)
        self.rel = RelationMLP(cfg.hidden, edge_dim, cfg.rel_hidden, cfg.dropout)
        self.readout = SizeAwareNodeReadout(cfg.hidden, cfg.size_dim, cfg.dropout)
        self.edge_readout = EdgeReadout(cfg.hidden, cfg.edge_readout_hidden, cfg.dropout)
        self.head = RegressionHead(cfg.hidden, cfg.hidden, cfg.dropout)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token = self.trunk(batch)
        n_mol = int(batch["n_molecules"])
        node_mean, node_sum = _node_pools(token, batch)
        edge_mol = batch["root_edge_mol"]
        h_global = node_mean[edge_mol]
        r = relation_from_pair(self.rel, h_global, h_global, batch["root_edge_attr"])
        edge_mean, edge_sum, has_edge = _edge_pools(r, edge_mol, batch["root_edge_unique"], n_mol)
        x = self.readout(node_mean, node_sum, batch["mol_size_feat"])
        x = x + self.edge_readout(edge_mean, edge_sum) * has_edge.unsqueeze(-1)
        return self.head(x)


class ModelMKeepEdge(nn.Module):
    """R1's direct edge readout plus one tied node-update round (follow-up only)."""

    def __init__(self, node_dim: int, edge_dim: int, cfg: P1CConfig) -> None:
        super().__init__()
        self.trunk = SharedTrunk(node_dim, edge_dim, cfg)
        self.rel = RelationMLP(cfg.hidden, edge_dim, cfg.rel_hidden, cfg.dropout)
        self.readout = SizeAwareNodeReadout(cfg.hidden, cfg.size_dim, cfg.dropout)
        self.edge_readout = EdgeReadout(cfg.hidden, cfg.edge_readout_hidden, cfg.dropout)
        self.update = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.rel_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.rel_hidden, cfg.hidden),
        )
        self.update_norm = nn.LayerNorm(cfg.hidden)
        self.update_dropout = nn.Dropout(cfg.dropout)
        self.head = RegressionHead(cfg.hidden, cfg.hidden, cfg.dropout)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token = self.trunk(batch)
        n_mol = int(batch["n_molecules"])
        r0 = self.rel(token, batch["root_edge_index"], batch["root_edge_attr"])
        src = batch["root_edge_index"][0]
        dst = batch["root_edge_index"][1]
        unique = src < dst
        edge_mean, edge_sum, has_edge = _edge_pools(r0, batch["root_batch"][src], unique, n_mol)

        m = scatter(r0, dst, dim=0, dim_size=token.shape[0], reduce="mean")
        updated = self.update_norm(token + self.update_dropout(self.update(m)))
        node_mean, node_sum = _node_pools(updated, batch)
        x = self.readout(node_mean, node_sum, batch["mol_size_feat"])
        x = x + self.edge_readout(edge_mean, edge_sum) * has_edge.unsqueeze(-1)
        return self.head(x)


def build_scap_size(atom_dim: int, edge_dim: int, cfg: P1CConfig) -> nn.Module:
    return ModelSCapSize(atom_dim + 4, edge_dim, cfg)


def build_r1_size(atom_dim: int, edge_dim: int, cfg: P1CConfig) -> nn.Module:
    return ModelR1Size(atom_dim + 4, edge_dim, cfg)


def build_r1_marginal_size(atom_dim: int, edge_dim: int, cfg: P1CConfig) -> nn.Module:
    return ModelR1MarginalSize(atom_dim + 4, edge_dim, cfg)


def build_m_keep_edge(atom_dim: int, edge_dim: int, cfg: P1CConfig) -> nn.Module:
    return ModelMKeepEdge(atom_dim + 4, edge_dim, cfg)


def register_phase1c_models() -> None:
    """Register builders and batch-key guards with the shared training code."""
    from ssrc_models import (
        MARGINAL_SIZE_KEYS,
        RELATIONAL_SIZE_KEYS,
        SIZE_ONLY_KEYS,
        assert_marginal_size_batch,
        assert_relational_size_batch,
        assert_size_only_batch,
        register_builder,
    )
    from ssrc_train import register_model_keys

    for name, builder in [
        ("S_cap_size", build_scap_size),
        ("R1_size", build_r1_size),
        ("R1_marginal_size", build_r1_marginal_size),
        ("M_keep_edge", build_m_keep_edge),
    ]:
        register_builder(name, builder)

    register_model_keys("S_cap_size", SIZE_ONLY_KEYS, assert_size_only_batch)
    register_model_keys("R1_size", RELATIONAL_SIZE_KEYS, assert_relational_size_batch)
    register_model_keys("R1_marginal_size", MARGINAL_SIZE_KEYS, assert_marginal_size_batch)
    register_model_keys("M_keep_edge", RELATIONAL_SIZE_KEYS, assert_relational_size_batch)

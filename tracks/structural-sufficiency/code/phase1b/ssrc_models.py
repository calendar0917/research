"""Phase 1b models: S0, R1, M1/M2 (tied iteration) and a strong GINE reference.

The four SSRC arms share one local radius-2 patch encoder and one two-block
per-token residual FFN.  They differ only in what is done with the root graph:

* S0    -- no root edges.  Mean-pool patch tokens -> regression head.
* R1    -- one-shot endpoint-swap-invariant edge relation on the *true* root
           edges, then concatenate the mean node token and mean edge relation
           at the readout.  No node-state update.
* M1    -- the same relation function, aggregated into a node-state update
           (one round), then mean-pool.  This adds the update, not iteration.
* M2    -- the exact same (weight-tied) relation/update module applied twice.
           Zero extra parameters versus M1: this isolates iteration.

``G_ref`` is a separate 6-layer residual GINE with FFN blocks on the raw atom
graph and sum+mean readout; it is a strong same-input application reference,
not a paper reproduction.

Endpoint-swap invariance
------------------------
For a root edge (i, j) the relation input is
``[h_i+h_j, |h_i-h_j|, h_i*h_j, e_ij]``.  All three node terms and the bond
feature are invariant under i<->j, so ``r_ij = r_ji``; the undirected edge
readout counts each physical bond once (``src < dst``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv
from torch_geometric.utils import scatter


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(nn.Linear(d_in, d_hidden), nn.ReLU(), nn.Linear(d_hidden, d_out))


class GINELayer(nn.Module):
    """One residual GINE layer with LayerNorm and dropout."""

    def __init__(self, hidden: int, edge_dim: int, mlp_hidden: int, dropout: float) -> None:
        super().__init__()
        self.conv = GINEConv(_mlp(hidden, mlp_hidden, hidden), edge_dim=edge_dim, train_eps=True)
        self.norm = nn.LayerNorm(hidden)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        h = self.conv(x, edge_index, edge_attr)
        return self.norm(x + self.dropout(h))


class LocalPatchEncoder(nn.Module):
    """Shared local GINE inside each rooted patch; token = patch-node mean."""

    def __init__(self, node_dim: int, edge_dim: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(node_dim, hidden)
        self.layers = nn.ModuleList(
            [GINELayer(hidden, edge_dim, hidden, dropout) for _ in range(layers)]
        )

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        x = self.input_proj(batch["patch_node_feat"])
        for layer in self.layers:
            x = layer(x, batch["patch_edge_index"], batch["patch_edge_attr"])
        return scatter(x, batch["patch_batch"], dim=0, dim_size=int(batch["n_patches"]), reduce="mean")


class ResidualTokenFFN(nn.Module):
    """Per-token residual FFN; no token-to-token interaction."""

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


class RegressionHead(nn.Module):
    def __init__(self, hidden_in: int, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden_in)
        self.net = nn.Sequential(
            nn.Linear(hidden_in, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x)).squeeze(-1)


class SharedTrunk(nn.Module):
    """Local patch encoder + two residual token FFN blocks (shared by all arms)."""

    def __init__(self, node_dim: int, edge_dim: int, cfg: "ModelConfig") -> None:
        super().__init__()
        self.local = LocalPatchEncoder(node_dim, edge_dim, cfg.hidden, cfg.local_layers, cfg.dropout)
        self.blocks = nn.ModuleList(
            [ResidualTokenFFN(cfg.hidden, cfg.ffn_hidden, cfg.dropout) for _ in range(cfg.token_ffn_blocks)]
        )

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token = self.local(batch)
        for block in self.blocks:
            token = block(token)
        return token


class ModelS0(nn.Module):
    """Local patches only: no root edges, no cross-patch interaction."""

    def __init__(self, node_dim: int, edge_dim: int, cfg: "ModelConfig") -> None:
        super().__init__()
        self.trunk = SharedTrunk(node_dim, edge_dim, cfg)
        self.head = RegressionHead(cfg.hidden, cfg.hidden, cfg.dropout)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token = self.trunk(batch)
        mol = scatter(token, batch["patch_to_mol"], dim=0, dim_size=int(batch["n_molecules"]), reduce="mean")
        return self.head(mol)


class RelationMLP(nn.Module):
    """Endpoint-swap-invariant relation function on one root edge."""

    def __init__(self, hidden: int, edge_dim: int, rel_hidden: int, dropout: float) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(3 * hidden + edge_dim, rel_hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(rel_hidden, hidden),
        )

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_attr: torch.Tensor) -> torch.Tensor:
        src, dst = edge_index[0], edge_index[1]
        h_s, h_d = x[src], x[dst]
        sym = torch.cat([h_s + h_d, (h_s - h_d).abs(), h_s * h_d, edge_attr], dim=-1)
        return self.net(sym)


class ModelR1(nn.Module):
    """One-shot relation composition; readout uses node and edge representations.

    No node state is updated: the relation representation is pooled directly
    into a graph-level edge embedding and concatenated with the node mean.
    """

    def __init__(self, node_dim: int, edge_dim: int, cfg: "ModelConfig") -> None:
        super().__init__()
        self.trunk = SharedTrunk(node_dim, edge_dim, cfg)
        self.rel = RelationMLP(cfg.hidden, edge_dim, cfg.rel_hidden, cfg.dropout)
        self.head = RegressionHead(2 * cfg.hidden, cfg.hidden, cfg.dropout)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        token = self.trunk(batch)
        r = self.rel(token, batch["root_edge_index"], batch["root_edge_attr"])
        mol_node = scatter(token, batch["patch_to_mol"], dim=0, dim_size=int(batch["n_molecules"]), reduce="mean")
        # count each physical undirected bond once
        src = batch["root_edge_index"][0]
        dst = batch["root_edge_index"][1]
        unique = src < dst
        if bool(unique.any()):
            edge_mol = batch["root_batch"][src[unique]]
            mol_edge = scatter(r[unique], edge_mol, dim=0, dim_size=int(batch["n_molecules"]), reduce="mean")
        else:
            mol_edge = torch.zeros_like(mol_node)
        return self.head(torch.cat([mol_node, mol_edge], dim=-1))


class ModelM(nn.Module):
    """Relation MPNN on the true root graph; ``rounds`` tied applications.

    M1 (rounds=1) adds a node-state update on top of the same relation
    function used by R1.  M2 (rounds=2) reuses exactly the same parameters, so
    R1->M1 measures the update and M1->M2 measures iteration with no capacity
    change.
    """

    def __init__(self, node_dim: int, edge_dim: int, cfg: "ModelConfig") -> None:
        super().__init__()
        self.trunk = SharedTrunk(node_dim, edge_dim, cfg)
        self.rel = RelationMLP(cfg.hidden, edge_dim, cfg.rel_hidden, cfg.dropout)
        self.update = nn.Sequential(
            nn.Linear(cfg.hidden, cfg.rel_hidden),
            nn.SiLU(),
            nn.Dropout(cfg.dropout),
            nn.Linear(cfg.rel_hidden, cfg.hidden),
        )
        self.update_norm = nn.LayerNorm(cfg.hidden)
        self.update_dropout = nn.Dropout(cfg.dropout)
        self.rounds = int(cfg.m_rounds)
        self.head = RegressionHead(cfg.hidden, cfg.hidden, cfg.dropout)

    def _round(self, x: torch.Tensor, batch: Mapping[str, Any]) -> torch.Tensor:
        r = self.rel(x, batch["root_edge_index"], batch["root_edge_attr"])
        dst = batch["root_edge_index"][1]
        m = scatter(r, dst, dim=0, dim_size=x.shape[0], reduce="mean")
        return self.update_norm(x + self.update_dropout(self.update(m)))

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        x = self.trunk(batch)
        for _ in range(self.rounds):
            x = self._round(x, batch)
        mol = scatter(x, batch["patch_to_mol"], dim=0, dim_size=int(batch["n_molecules"]), reduce="mean")
        return self.head(mol)


class ModelGRef(nn.Module):
    """Strong residual GINE reference on the raw atom graph (6 layers).

    Uses only raw atom/bond features, not the E2 patch representation.  It is
    an application reference; it is not a reproduction of a published result.
    """

    def __init__(self, atom_dim: int, edge_dim: int, cfg: "ModelConfig") -> None:
        super().__init__()
        self.input_proj = nn.Linear(atom_dim, cfg.g_hidden)
        self.layers = nn.ModuleList(
            [GINELayer(cfg.g_hidden, edge_dim, cfg.g_hidden, cfg.dropout) for _ in range(cfg.g_layers)]
        )
        self.ffns = nn.ModuleList(
            [
                nn.Sequential(
                    nn.LayerNorm(cfg.g_hidden),
                    nn.Linear(cfg.g_hidden, cfg.g_ffn_hidden),
                    nn.SiLU(),
                    nn.Dropout(cfg.dropout),
                    nn.Linear(cfg.g_ffn_hidden, cfg.g_hidden),
                )
                for _ in range(cfg.g_layers)
            ]
        )
        self.ffn_dropout = nn.Dropout(cfg.dropout)
        self.head = RegressionHead(2 * cfg.g_hidden, cfg.g_hidden, cfg.dropout)

    def forward(self, batch: Mapping[str, Any]) -> torch.Tensor:
        x = self.input_proj(batch["atom_feat"])
        for layer, ffn in zip(self.layers, self.ffns):
            x = layer(x, batch["root_edge_index"], batch["root_edge_attr"])
            x = x + self.ffn_dropout(ffn(x))
        mol_sum = scatter(x, batch["root_batch"], dim=0, dim_size=int(batch["n_molecules"]), reduce="sum")
        mol_mean = scatter(x, batch["root_batch"], dim=0, dim_size=int(batch["n_molecules"]), reduce="mean")
        return self.head(torch.cat([mol_sum, mol_mean], dim=-1))


# ---------------------------------------------------------------------------
# configuration / construction
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelConfig:
    hidden: int = 64
    local_layers: int = 2
    token_ffn_blocks: int = 2
    ffn_hidden: int = 128
    rel_hidden: int = 128
    m_rounds: int = 1
    dropout: float = 0.1
    g_hidden: int = 128
    g_layers: int = 6
    g_ffn_hidden: int = 128

    def as_dict(self) -> dict[str, Any]:
        return dict(self.__dict__)


def build_model(name: str, atom_dim: int, edge_dim: int, cfg: ModelConfig) -> nn.Module:
    node_dim = atom_dim + 4
    if name in _BUILDERS:
        return _BUILDERS[name](atom_dim, edge_dim, cfg)
    if name == "S0":
        return ModelS0(node_dim, edge_dim, cfg)
    if name == "R1":
        return ModelR1(node_dim, edge_dim, cfg)
    if name in {"M1", "M2"}:
        rounds = 1 if name == "M1" else 2
        return ModelM(node_dim, edge_dim, ModelConfig(**{**cfg.as_dict(), "m_rounds": rounds}))
    if name == "G_ref":
        return ModelGRef(atom_dim, edge_dim, cfg)
    raise ValueError(f"unknown model {name!r}")


# Extension point for later phases.  Phase 1b arms keep their original
# construction path unchanged; a later phase can register a builder without
# editing the frozen Phase 1b branches.
_BUILDERS: dict[str, Any] = {}


def register_builder(name: str, builder: Any) -> None:
    _BUILDERS[str(name)] = builder


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ---------------------------------------------------------------------------
# information-isolation guards
# ---------------------------------------------------------------------------

PATCH_ONLY_KEYS = frozenset(
    {
        "patch_node_feat",
        "patch_edge_index",
        "patch_edge_attr",
        "patch_batch",
        "patch_to_mol",
        "n_patches",
        "y",
        "mol_index",
        "n_molecules",
    }
)

RELATIONAL_KEYS = frozenset(
    set(PATCH_ONLY_KEYS)
    | {
        "root_edge_index",
        "root_edge_attr",
        "root_batch",
    }
)

GRAPH_ONLY_KEYS = frozenset(
    {
        "atom_feat",
        "root_edge_index",
        "root_edge_attr",
        "root_batch",
        "y",
        "mol_index",
        "n_molecules",
    }
)

# Phase 1c: explicit graph-size features enter the readout only.
SIZE_ONLY_KEYS = frozenset(set(PATCH_ONLY_KEYS) | {"mol_size_feat"})
RELATIONAL_SIZE_KEYS = frozenset(set(RELATIONAL_KEYS) | {"mol_size_feat"})
# Marginal-correspondence arm: no endpoint indices and no root node ids are
# present at all.  It receives only patch tokens, per-edge attributes, the
# molecule id of each directed edge, and the unique-edge mask.
MARGINAL_SIZE_KEYS = frozenset(
    set(PATCH_ONLY_KEYS)
    | {"root_edge_attr", "root_edge_mol", "root_edge_unique", "mol_size_feat"}
)


def _check_keys(batch: Mapping[str, Any], allowed: frozenset[str], required: frozenset[str], label: str) -> None:
    extra = set(batch) - allowed
    missing = required - set(batch)
    if extra or missing:
        raise AssertionError(f"{label} batch key violation extra={sorted(extra)} missing={sorted(missing)}")


def assert_patch_only_batch(batch: Mapping[str, Any]) -> None:
    _check_keys(batch, PATCH_ONLY_KEYS, PATCH_ONLY_KEYS, "S0")


def assert_relational_batch(batch: Mapping[str, Any]) -> None:
    _check_keys(batch, RELATIONAL_KEYS, RELATIONAL_KEYS, "R1/M")


def assert_graph_only_batch(batch: Mapping[str, Any]) -> None:
    _check_keys(batch, GRAPH_ONLY_KEYS, GRAPH_ONLY_KEYS, "G_ref")


def assert_size_only_batch(batch: Mapping[str, Any]) -> None:
    _check_keys(batch, SIZE_ONLY_KEYS, SIZE_ONLY_KEYS, "S_cap_size")


def assert_relational_size_batch(batch: Mapping[str, Any]) -> None:
    _check_keys(batch, RELATIONAL_SIZE_KEYS, RELATIONAL_SIZE_KEYS, "R1_size")


def assert_marginal_size_batch(batch: Mapping[str, Any]) -> None:
    _check_keys(batch, MARGINAL_SIZE_KEYS, MARGINAL_SIZE_KEYS, "R1_marginal_size")


def filter_batch(batch: Mapping[str, Any], keys: frozenset[str]) -> dict[str, Any]:
    return {k: v for k, v in batch.items() if k in keys}

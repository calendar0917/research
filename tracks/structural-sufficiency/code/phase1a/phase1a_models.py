"""Phase 1a models: S (no cross-patch interaction), X (same-information
cross-patch attention), G (full-graph GINE reference).

Information discipline
----------------------
S and X receive exactly the same patch tensors and mask.  Neither model
receives the original ``edge_index``, patch-to-patch incidence, root identity,
atom numbering or positional encoding.  X's self-attention is permutation
equivariant and its readout is permutation invariant; it is still a function of
the E_2 multiset and cannot break a true E_2 collision.

S is not "no message passing": it uses a shared local GINE inside each patch.
It has no cross-patch interaction.  The per-token residual FFN is a real
non-interactive capacity control, not a parameter-only decoration.
"""

from __future__ import annotations

from typing import Any, Mapping

import torch
import torch.nn as nn
from torch_geometric.nn import GINEConv

from phase1a_data import BOND_FEAT_DIM, PATCH_NODE_DIM, ATOM_BASE_DIM


def _mlp(d_in: int, d_hidden: int, d_out: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(d_in, d_hidden),
        nn.ReLU(),
        nn.Linear(d_hidden, d_out),
    )


class GINEStack(nn.Module):
    """Shared local GINE encoder used by S and X inside each patch."""

    def __init__(self, node_in: int, hidden: int, layers: int, dropout: float) -> None:
        super().__init__()
        self.input_proj = nn.Linear(node_in, hidden)
        self.convs = nn.ModuleList(
            [GINEConv(_mlp(hidden, hidden, hidden), edge_dim=BOND_FEAT_DIM, train_eps=True) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)

    def forward(self, batch: Mapping[str, torch.Tensor], n_patches: int) -> torch.Tensor:
        x = self.input_proj(batch["patch_node_feat"])
        ei = batch["patch_edge_index"]
        ea = batch["patch_edge_attr"]
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, ei, ea)
            x = norm(x + self.dropout(h))
        # Patch token: mean over the patch's nodes.  No root pooling is used, so
        # the token is invariant to the local node order.
        from torch_geometric.utils import scatter

        token = scatter(x, batch["patch_batch"], dim=0, dim_size=n_patches, reduce="mean")
        return token


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
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.norm = nn.LayerNorm(hidden)
        self.net = nn.Sequential(
            nn.Linear(hidden, hidden),
            nn.SiLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(self.norm(x)).squeeze(-1)


def _dense_tokens(batch: Mapping[str, torch.Tensor], token: torch.Tensor) -> torch.Tensor:
    """Place patch tokens into [B, max_patches, d] storage.

    The storage position is not a feature: no positional encoding is added and
    the attention/readout are permutation equivariant/invariant.
    """
    b = int(batch["patch_mask"].shape[0])
    max_patches = int(batch["patch_mask"].shape[1])
    dense = token.new_zeros((b, max_patches, token.shape[-1]))
    dense[batch["patch_to_mol"], batch["patch_pos"]] = token
    return dense


def _masked_mean(dense: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
    weights = mask.unsqueeze(-1).to(dense.dtype)
    return (dense * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1.0)


class ModelS(nn.Module):
    """Shared patch encoder + per-token FFN + mean pooling. No cross-patch interaction."""

    def __init__(
        self,
        hidden: int = 64,
        local_layers: int = 2,
        ffn_layers: int = 4,
        ffn_hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.local = GINEStack(PATCH_NODE_DIM, hidden, local_layers, dropout)
        self.blocks = nn.ModuleList([ResidualTokenFFN(hidden, ffn_hidden, dropout) for _ in range(ffn_layers)])
        self.head = RegressionHead(hidden, dropout)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        token = self.local(batch, int(batch["patch_batch"].max().item()) + 1 if batch["patch_batch"].numel() else 0)
        for block in self.blocks:
            token = block(token)
        from torch_geometric.utils import scatter

        mol = scatter(token, batch["patch_to_mol"], dim=0, dim_size=int(batch["patch_mask"].shape[0]), reduce="mean")
        return self.head(mol)


class ModelX(nn.Module):
    """Same local encoder as S + global self-attention over patch tokens."""

    def __init__(
        self,
        hidden: int = 64,
        local_layers: int = 2,
        attn_layers: int = 2,
        nhead: int = 4,
        ffn_hidden: int = 128,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.local = GINEStack(PATCH_NODE_DIM, hidden, local_layers, dropout)
        layer = nn.TransformerEncoderLayer(
            d_model=hidden,
            nhead=nhead,
            dim_feedforward=ffn_hidden,
            dropout=dropout,
            activation="gelu",
            batch_first=True,
            norm_first=True,
        )
        self.layers = nn.ModuleList([layer for _ in range(attn_layers)])
        self.head = RegressionHead(hidden, dropout)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        token = self.local(batch, int(batch["patch_batch"].max().item()) + 1 if batch["patch_batch"].numel() else 0)
        dense = _dense_tokens(batch, token)
        mask = batch["patch_mask"]
        for layer in self.layers:
            dense = layer(dense, src_key_padding_mask=~mask)
        mol = _masked_mean(dense, mask)
        return self.head(mol)


class ModelG(nn.Module):
    """Full-graph GINE reference; information access differs from S/X."""

    def __init__(
        self,
        hidden: int = 64,
        layers: int = 2,
        dropout: float = 0.1,
    ) -> None:
        super().__init__()
        self.input_proj = nn.Linear(ATOM_BASE_DIM, hidden)
        self.convs = nn.ModuleList(
            [GINEConv(_mlp(hidden, hidden, hidden), edge_dim=BOND_FEAT_DIM, train_eps=True) for _ in range(layers)]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(layers)])
        self.dropout = nn.Dropout(dropout)
        self.head = RegressionHead(hidden, dropout)

    def forward(self, batch: Mapping[str, torch.Tensor]) -> torch.Tensor:
        x = self.input_proj(batch["node_feat"])
        for conv, norm in zip(self.convs, self.norms):
            h = conv(x, batch["edge_index"], batch["edge_attr"])
            x = norm(x + self.dropout(h))
        from torch_geometric.utils import scatter

        mol = scatter(x, batch["batch"], dim=0, dim_size=int(batch["batch"].max().item()) + 1, reduce="mean")
        return self.head(mol)


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
        "patch_pos",
        "patch_mask",
        "mol_n_patches",
        "n_patches",
        "y",
        "mol_index",
    }
)

GRAPH_ONLY_KEYS = frozenset({"node_feat", "edge_index", "edge_attr", "batch", "y", "mol_index"})


def assert_patch_only_batch(batch: Mapping[str, Any]) -> None:
    extra = set(batch) - PATCH_ONLY_KEYS
    missing = PATCH_ONLY_KEYS - set(batch)
    if extra or missing:
        raise AssertionError(f"S/X batch key violation extra={sorted(extra)} missing={sorted(missing)}")


def assert_graph_only_batch(batch: Mapping[str, Any]) -> None:
    extra = set(batch) - GRAPH_ONLY_KEYS
    missing = GRAPH_ONLY_KEYS - set(batch)
    if extra or missing:
        raise AssertionError(f"G batch key violation extra={sorted(extra)} missing={sorted(missing)}")


def parameter_count(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

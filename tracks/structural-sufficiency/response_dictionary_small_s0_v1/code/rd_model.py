"""Frozen conditional-response model and its exact additive control.

Both arms share one encoder ``E_theta`` and one scalar head design.  The only
difference is the fusion operator:

    q(S) = tanh(U z_S),  k(C) = tanh(V z_C)
    T_RESP(C,S) = sum_j w[j] q(S)[j] k(C)[j] / sqrt(16)
    T_ADD(C,S)  = sum_j w[j] (q(S)[j] + k(C)[j]) / sqrt(32)
    s(C,S) = b(C) + a(S) + T(C,S)

``w`` starts at exactly zero, so both arms start at identical predictions.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Batch, Data
from torch_geometric.nn import GINEConv, global_add_pool, global_mean_pool

ATOM_COLUMNS = 9
BOND_COLUMNS = 3
DEGREE_BUCKETS = 17  # 0..15 exact, 16 = ">= 16"


def ogb_category_dims() -> dict[str, int]:
    """Category counts from the official OGB feature definition (never from Val/Test)."""
    from ogb.utils.features import get_atom_feature_dims, get_bond_feature_dims

    atom = [int(x) for x in get_atom_feature_dims()]
    bond = [int(x) for x in get_bond_feature_dims()]
    if len(atom) != ATOM_COLUMNS or len(bond) != BOND_COLUMNS:
        raise ValueError("unexpected OGB feature definition")
    return {
        "atom": atom,
        "bond": bond,
        "port": 2,
        "degree": DEGREE_BUCKETS,
    }


# ---------------------------------------------------------------------------
# fragment tensors
# ---------------------------------------------------------------------------


def fragment_data_list(payload: Mapping[str, Any]) -> list[Data]:
    """One ``Data`` per unique exact fragment (physical bonds in both directions)."""
    n_frag = int(payload["frag_n_atoms"].shape[0])
    atom_ptr = np.asarray(payload["frag_atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(payload["frag_bond_ptr"], dtype=np.int64)
    atom_feat = np.asarray(payload["frag_atom_feat"])
    internal = np.asarray(payload["frag_internal_degree"])
    port_local = np.asarray(payload["frag_port_local"])
    bond_u = np.asarray(payload["frag_bond_u"])
    bond_v = np.asarray(payload["frag_bond_v"])
    bond_feat = np.asarray(payload["frag_bond_feat"])
    out: list[Data] = []
    for i in range(n_frag):
        a0, a1 = int(atom_ptr[i]), int(atom_ptr[i + 1])
        b0, b1 = int(bond_ptr[i]), int(bond_ptr[i + 1])
        n = a1 - a0
        x_cat = torch.as_tensor(np.asarray(atom_feat[a0:a1], dtype=np.int64), dtype=torch.long)
        deg = torch.as_tensor(np.minimum(np.asarray(internal[a0:a1], dtype=np.int64), DEGREE_BUCKETS - 1), dtype=torch.long)
        port = torch.zeros(n, dtype=torch.long)
        port_idx = int(port_local[i])
        if port_idx >= 0:
            port[int(port_idx)] = 1
        bu = np.asarray(bond_u[b0:b1], dtype=np.int64)
        bv = np.asarray(bond_v[b0:b1], dtype=np.int64)
        bfeat = np.asarray(bond_feat[b0:b1], dtype=np.int64)
        if bu.size:
            edge_index = torch.as_tensor(
                np.concatenate([np.stack([bu, bv]), np.stack([bv, bu])], axis=1), dtype=torch.long
            )
            edge_cat = torch.as_tensor(np.concatenate([bfeat, bfeat], axis=0), dtype=torch.long)
        else:
            edge_index = torch.zeros((2, 0), dtype=torch.long)
            edge_cat = torch.zeros((0, BOND_COLUMNS), dtype=torch.long)
        data = Data(
            num_nodes=n,
            x_cat=x_cat,
            port=port,
            deg=deg,
            edge_index=edge_index,
            edge_cat=edge_cat,
            port_local=torch.as_tensor([port_idx], dtype=torch.long),
            n_atoms=torch.as_tensor([n], dtype=torch.long),
        )
        out.append(data)
    return out


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------


class FragmentEncoder(nn.Module):
    """``E_theta``: shared encoder for S, C and the no-cut fallback molecule."""

    def __init__(self, dims: Mapping[str, Any], hidden: int = 96, latent: int = 16,
                 n_layers: int = 3, dropout: float = 0.10) -> None:
        super().__init__()
        self.hidden = hidden
        self.latent = latent
        self.atom_embeddings = nn.ModuleList(
            [nn.Embedding(int(d), hidden) for d in dims["atom"]]
        )
        self.bond_embeddings = nn.ModuleList(
            [nn.Embedding(int(d), hidden) for d in dims["bond"]]
        )
        self.port_embedding = nn.Embedding(int(dims["port"]), hidden)
        self.degree_embedding = nn.Embedding(int(dims["degree"]), hidden)
        self.convs = nn.ModuleList(
            [
                GINEConv(
                    nn.Sequential(nn.Linear(hidden, 2 * hidden), nn.SiLU(), nn.Linear(2 * hidden, hidden)),
                    train_eps=True,
                )
                for _ in range(n_layers)
            ]
        )
        self.norms = nn.ModuleList([nn.LayerNorm(hidden) for _ in range(n_layers)])
        self.drop = nn.Dropout(dropout)
        self.proj = nn.Sequential(
            nn.Linear(hidden * 3 + 1, 128),
            nn.SiLU(),
            nn.LayerNorm(128),
        )

    def encode_nodes(self, batch: Batch) -> torch.Tensor:
        out = self.atom_embeddings[0](batch.x_cat[:, 0])
        for j in range(1, ATOM_COLUMNS):
            out = out + self.atom_embeddings[j](batch.x_cat[:, j])
        out = out + self.port_embedding(batch.port) + self.degree_embedding(batch.deg)
        edge_attr = self.bond_embeddings[0](batch.edge_cat[:, 0])
        for j in range(1, BOND_COLUMNS):
            edge_attr = edge_attr + self.bond_embeddings[j](batch.edge_cat[:, j])
        for conv, norm in zip(self.convs, self.norms):
            residual = out
            out = conv(out, batch.edge_index, edge_attr)
            out = self.drop(F.silu(norm(out)))
            out = out + residual
        return out

    def forward(self, batch: Batch) -> torch.Tensor:
        """Per-graph descriptor projection ``z`` (128-dim)."""
        h = self.encode_nodes(batch)
        batch_index = batch.batch
        summed = global_add_pool(h, batch_index)
        meaned = global_mean_pool(h, batch_index)
        n_graphs = int(summed.shape[0])
        port_local = batch.port_local.view(-1)
        ptr = batch.ptr[:-1]
        port_state = h.new_zeros((n_graphs, self.hidden))
        mask = port_local >= 0
        if bool(mask.any()):
            index = ptr[mask] + port_local[mask]
            port_state = port_state.index_copy(0, torch.nonzero(mask, as_tuple=False).view(-1), h[index])
        n_atoms = batch.n_atoms.view(-1, 1).to(h.dtype)
        pooled = torch.cat([port_state, summed, meaned, torch.log1p(n_atoms)], dim=1)
        return self.proj(pooled)


def _mlp(in_dim: int, hidden: int, out_dim: int) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, hidden),
        nn.SiLU(),
        nn.Linear(hidden, out_dim),
    )


@dataclass
class ModelConfig:
    hidden: int = 96
    latent: int = 16
    n_layers: int = 3
    dropout: float = 0.10
    mode: str = "RESP"


class ResponseModel(nn.Module):
    """Full model: encoder + a(S) + b(C) + T_mode + fallback head."""

    def __init__(self, dims: Mapping[str, Any], config: ModelConfig) -> None:
        super().__init__()
        if config.mode not in ("RESP", "ADD"):
            raise ValueError(config.mode)
        self.config = config
        self.encoder = FragmentEncoder(
            dims,
            hidden=config.hidden,
            latent=config.latent,
            n_layers=config.n_layers,
            dropout=config.dropout,
        )
        self.a_head = _mlp(128, 64, 1)
        self.b_head = _mlp(128, 64, 1)
        self.U = nn.Linear(128, config.latent, bias=False)
        self.V = nn.Linear(128, config.latent, bias=False)
        self.w = nn.Parameter(torch.zeros(config.latent))
        self.fallback_head = _mlp(128, 64, 1)

    # -- pieces ---------------------------------------------------------
    def terms(self, z_s: torch.Tensor, z_c: torch.Tensor, *, detach_w: bool = False):
        a = self.a_head(z_s).view(-1)
        b = self.b_head(z_c).view(-1)
        q = torch.tanh(self.U(z_s))
        k = torch.tanh(self.V(z_c))
        if self.config.mode == "RESP":
            w = self.w
            t = (q * k * w.view(1, -1)).sum(dim=1) / (self.config.latent ** 0.5)
        else:
            w = self.w
            t = ((q + k) * w.view(1, -1)).sum(dim=1) / ((2 * self.config.latent) ** 0.5)
        return a, b, q, k, t

    def view_scores(self, z_s: torch.Tensor, z_c: torch.Tensor) -> torch.Tensor:
        a, b, _, _, t = self.terms(z_s, z_c)
        return b + a + t

    def fallback_scores(self, z: torch.Tensor) -> torch.Tensor:
        return self.fallback_head(z).view(-1)


# ---------------------------------------------------------------------------
# initial state
# ---------------------------------------------------------------------------


def canonical_init_state(dims: Mapping[str, Any], config: ModelConfig, seed: int = 0) -> dict[str, torch.Tensor]:
    """One seeded init, distilled to a state_dict that both arms load exactly."""
    torch.manual_seed(seed)
    model = ResponseModel(dims, config)
    state = {name: tensor.detach().clone() for name, tensor in model.state_dict().items()}
    return state


def state_hash(state: Mapping[str, torch.Tensor]) -> str:
    digest = hashlib.sha256()
    for name in sorted(state):
        tensor = state[name].detach().cpu().contiguous()
        digest.update(name.encode())
        digest.update(str(tuple(tensor.shape)).encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def parameter_layout(model: ResponseModel) -> dict[str, list[int]]:
    return {name: list(p.shape) for name, p in model.state_dict().items()}


def n_parameters(model: ResponseModel) -> int:
    return int(sum(p.numel() for p in model.parameters()))


def build_model(dims: Mapping[str, Any], config: ModelConfig, state: Mapping[str, torch.Tensor]) -> ResponseModel:
    model = ResponseModel(dims, config)
    model.load_state_dict({k: v.clone() for k, v in state.items()}, strict=True)
    return model


def zero_fusion_w(model: ResponseModel, state: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
    """Same state with the fusion weights zeroed (fixed-weight dependence probe)."""
    other = {k: v.clone() for k, v in state.items()}
    if "w" not in other:
        raise KeyError("state has no fusion vector w")
    other["w"] = torch.zeros_like(other["w"])
    return other

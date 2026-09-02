"""BRICS chemical-block graph probe for ogbg-molhiv.

Molecules are partitioned at BRICS bonds.  Exact atom/bond topology is encoded
only inside each block, followed by an optional one-layer block graph.  Cut-bond
endpoint chemistry is carried by block-graph edges.  The set, true graph, and
content-to-topology shuffled controls share initialization and training order.
There is no whole-molecule atom GNN, GINE, KSVD, path vocabulary, or prototype
bank.
"""
from __future__ import annotations

import argparse
import csv
import gzip
import hashlib
import json
import math
import random
import sys
import time
from pathlib import Path
from typing import Any

import networkx as nx
from rdkit import Chem
from rdkit.Chem import BRICS
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import _patch_torch_load_weights_only
from code.run_molhiv_chemical_fragment_graph_probe import (
    FRAGMENT_TYPES,
    MODES,
    Fragment,
    canonical_edge,
    deterministic_fragments,
    fragment_connections,
    fragment_scalars,
    parameter_counts,
    seed_everything,
    segment_mean_max,
    sha256_array,
)

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder

VARIANTS = ("brics",)


class FragmentData(Data):
    """PyG data with a second, atom-occurrence graph inside fragments.

    ``num_nodes`` deliberately remains the number of fragments because the main
    ``edge_index`` is the fragment graph.  Local atom edges therefore need their
    own batching increment.
    """

    def __inc__(self, key: str, value: torch.Tensor, *args: Any, **kwargs: Any) -> Any:
        if key == "local_edge_index":
            return int(self.local_atom_fields.shape[0])
        return super().__inc__(key, value, *args, **kwargs)


def _empty(shape: tuple[int, ...], dtype: np.dtype = np.int64) -> np.ndarray:
    return np.empty(shape, dtype=dtype)


def _stack(rows: list[np.ndarray], width: int, dtype: np.dtype = np.int64) -> np.ndarray:
    return np.stack(rows).astype(dtype, copy=False) if rows else _empty((0, width), dtype)



def brics_fragment_data(
    graph_dict: dict[str, Any], smiles: str, label: float, graph_index: int, shuffle_seed: int
) -> tuple[FragmentData, dict[str, Any]]:
    """Partition a molecule into BRICS blocks and retain exact topology in each block."""
    n_nodes = int(graph_dict["num_nodes"])
    node_feat = np.asarray(graph_dict["node_feat"], dtype=np.int64)
    edge_index_raw = np.asarray(graph_dict["edge_index"], dtype=np.int64)
    edge_feat_raw = np.asarray(graph_dict["edge_feat"], dtype=np.int64)
    edge_features: dict[tuple[int, int], np.ndarray] = {}
    for j in range(edge_index_raw.shape[1]):
        u, v = int(edge_index_raw[0, j]), int(edge_index_raw[1, j])
        if u != v:
            edge_features.setdefault(canonical_edge(u, v), edge_feat_raw[j].copy())
    all_edges = sorted(edge_features)
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(all_edges)

    mol = Chem.MolFromSmiles(smiles)
    if mol is None or mol.GetNumAtoms() != n_nodes:
        cut_edges: set[tuple[int, int]] = set()
        rdkit_ok = False
    else:
        cut_edges = {
            canonical_edge(int(a), int(b))
            for (a, b), _labels in BRICS.FindBRICSBonds(mol)
        }
        cut_edges.intersection_update(edge_features)
        rdkit_ok = True

    internal_graph = graph.copy()
    internal_graph.remove_edges_from(cut_edges)
    components = sorted(
        (tuple(sorted(int(u) for u in comp)) for comp in nx.connected_components(internal_graph)),
        key=lambda atoms: (min(atoms), len(atoms), atoms),
    )
    atom_to_fragment = np.empty(n_nodes, dtype=np.int64)
    for fi, atoms in enumerate(components):
        atom_to_fragment[np.asarray(atoms, dtype=np.int64)] = fi

    frag_types: list[int] = []
    frag_scalars: list[np.ndarray] = []
    atom_fields: list[np.ndarray] = []
    atom_fragment: list[int] = []
    bond_fields: list[np.ndarray] = []
    bond_fragment: list[int] = []
    local_atom_fields: list[np.ndarray] = []
    local_atom_scalar: list[np.ndarray] = []
    local_atom_fragment: list[int] = []
    local_source: list[int] = []
    local_target: list[int] = []
    local_edge_fields: list[np.ndarray] = []

    cut_degree = np.zeros(n_nodes, dtype=np.int64)
    for u, v in cut_edges:
        cut_degree[u] += 1
        cut_degree[v] += 1

    component_sets = [set(atoms) for atoms in components]
    for fi, atoms in enumerate(components):
        atom_set = component_sets[fi]
        internal_edges = [e for e in all_edges if e not in cut_edges and e[0] in atom_set and e[1] in atom_set]
        has_ring = bool(nx.cycle_basis(graph.subgraph(atoms)))
        frag_type = 0 if len(atoms) == 1 else (2 if has_ring else 1)
        frag_types.append(frag_type)
        degrees = np.asarray([graph.degree[a] for a in atoms], dtype=np.float32)
        aromatic = np.asarray([node_feat[a, 7] for a in atoms], dtype=np.float32)
        hetero = np.asarray([node_feat[a, 0] != 5 for a in atoms], dtype=np.float32)  # carbon index is 5
        n_ports = int(sum(cut_degree[a] for a in atoms))
        frag_scalars.append(np.asarray([
            math.log1p(len(atoms)) / 4.0,
            math.log1p(len(internal_edges)) / 4.0,
            math.log1p(n_ports) / 3.0,
            float(has_ring),
            float(len(atoms) == 1),
            float(np.mean(aromatic)) if len(aromatic) else 0.0,
            float(np.mean(hetero)) if len(hetero) else 0.0,
            float(np.mean(degrees)) / 4.0 if len(degrees) else 0.0,
            float(np.max(degrees)) / 4.0 if len(degrees) else 0.0,
            float(n_ports > 1),
            float(rdkit_ok),
        ], dtype=np.float32))
        lookup: dict[int, int] = {}
        for atom in atoms:
            atom_fields.append(node_feat[atom])
            atom_fragment.append(fi)
            lookup[atom] = len(local_atom_fields)
            local_atom_fields.append(node_feat[atom])
            internal_degree = sum(int(v) in atom_set for v in graph.neighbors(atom))
            local_atom_scalar.append(np.asarray([
                internal_degree / 4.0,
                cut_degree[atom] / 4.0,
                graph.degree[atom] / 4.0,
                float(cut_degree[atom] > 0),
                float(internal_degree <= 1),
                float(has_ring),
                math.log1p(len(atoms)) / 4.0,
                node_feat[atom, 7],
                float(node_feat[atom, 0] != 5),
            ], dtype=np.float32))
            local_atom_fragment.append(fi)
        for edge in internal_edges:
            bond_fields.append(edge_features[edge])
            bond_fragment.append(fi)
            u, v = edge
            ui, vi = lookup[u], lookup[v]
            local_source.extend([ui, vi])
            local_target.extend([vi, ui])
            local_edge_fields.extend([edge_features[edge], edge_features[edge]])

    directed_connections: list[tuple[int, int, int, int, tuple[int, int]]] = []
    for u, v in sorted(cut_edges):
        fu, fv = int(atom_to_fragment[u]), int(atom_to_fragment[v])
        directed_connections.extend([(fu, fv, u, v, (u, v)), (fv, fu, v, u, (u, v))])
    if directed_connections:
        connection_index = np.asarray([(a, b) for a, b, *_ in directed_connections], dtype=np.int64).T
    else:
        connection_index = _empty((2, 0), np.int64)
    edge_source_atom: list[np.ndarray] = []
    edge_target_atom: list[np.ndarray] = []
    edge_bond: list[np.ndarray] = []
    edge_role: list[np.ndarray] = []
    for source, target, source_atom, target_atom, edge in directed_connections:
        edge_source_atom.append(node_feat[source_atom])
        edge_target_atom.append(node_feat[target_atom])
        edge_bond.append(edge_features[edge])
        edge_role.append(np.asarray([
            graph.degree[source_atom] / 4.0,
            cut_degree[source_atom] / 4.0,
            graph.degree[target_atom] / 4.0,
            cut_degree[target_atom] / 4.0,
            math.log1p(len(components[source])) / 4.0,
            math.log1p(len(components[target])) / 4.0,
        ], dtype=np.float32))

    rng = np.random.default_rng(np.uint64(shuffle_seed) ^ np.uint64((graph_index + 1) * 0x9E3779B1))
    node_permutation = rng.permutation(len(components)).astype(np.int64)
    if len(components) > 1 and np.array_equal(node_permutation, np.arange(len(components))):
        node_permutation = np.roll(node_permutation, 1)
    edge_permutation = rng.permutation(len(directed_connections)).astype(np.int64)
    local_edge_index = (
        np.asarray([local_source, local_target], dtype=np.int64)
        if local_source else _empty((2, 0), np.int64)
    )
    source_array = _stack(edge_source_atom, 9)
    target_array = _stack(edge_target_atom, 9)
    edge_bond_array = _stack(edge_bond, 3)
    role_array = _stack(edge_role, 6, np.float32)
    data = FragmentData(
        num_nodes=len(components),
        frag_type=torch.tensor(frag_types, dtype=torch.long),
        frag_scalar=torch.tensor(np.stack(frag_scalars), dtype=torch.float32),
        atom_fields=torch.tensor(_stack(atom_fields, 9), dtype=torch.long),
        atom_fragment_index=torch.tensor(atom_fragment, dtype=torch.long),
        bond_fields=torch.tensor(_stack(bond_fields, 3), dtype=torch.long),
        bond_fragment_index=torch.tensor(bond_fragment, dtype=torch.long),
        local_atom_fields=torch.tensor(_stack(local_atom_fields, 9), dtype=torch.long),
        local_atom_scalar=torch.tensor(_stack(local_atom_scalar, 9, np.float32), dtype=torch.float32),
        local_atom_fragment_index=torch.tensor(local_atom_fragment, dtype=torch.long),
        local_edge_index=torch.tensor(local_edge_index, dtype=torch.long),
        local_edge_fields=torch.tensor(_stack(local_edge_fields, 3), dtype=torch.long),
        edge_index=torch.tensor(connection_index, dtype=torch.long),
        edge_source_atom_fields=torch.tensor(source_array, dtype=torch.long),
        edge_target_atom_fields=torch.tensor(target_array, dtype=torch.long),
        edge_bond_fields=torch.tensor(edge_bond_array, dtype=torch.long),
        edge_role_scalar=torch.tensor(role_array, dtype=torch.float32),
        shuffled_edge_source_atom_fields=torch.tensor(source_array[edge_permutation], dtype=torch.long),
        shuffled_edge_target_atom_fields=torch.tensor(target_array[edge_permutation], dtype=torch.long),
        shuffled_edge_bond_fields=torch.tensor(edge_bond_array[edge_permutation], dtype=torch.long),
        shuffled_edge_role_scalar=torch.tensor(role_array[edge_permutation], dtype=torch.float32),
        shuffle_index=torch.tensor(node_permutation, dtype=torch.long),
        y=torch.tensor([label], dtype=torch.float32),
        graph_index=torch.tensor([graph_index], dtype=torch.long),
    )
    return data, {
        "n_fragments": len(components),
        "n_brics_bonds": len(cut_edges),
        "n_local_atom_occurrences": len(local_atom_fields),
        "n_directed_local_edges": len(local_source),
        "n_directed_connections": len(directed_connections),
        "max_fragment_atoms": max(map(len, components), default=0),
        "rdkit_parse_ok": int(rdkit_ok),
    }


class FeaturePool(nn.Module):
    def __init__(self, input_dim: int, hidden: int, dropout: float) -> None:
        super().__init__()
        half = hidden // 2
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, half), nn.SiLU(),
        )
        self.projection = nn.Sequential(nn.Linear(2 * half, hidden), nn.SiLU())

    def forward(self, values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
        if len(values) == 0:
            return values.new_zeros((n, self.projection[0].out_features))
        encoded = self.encoder(values)
        mean, maximum, _ = segment_mean_max(encoded, index, n)
        return self.projection(torch.cat([mean, maximum], dim=1))


class PositionAwareMessageLayer(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.update = nn.Sequential(
            nn.Linear(3 * hidden + 1, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor, edge_h: torch.Tensor) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return x
        source, target = edge_index
        messages = self.message(torch.cat([x[source], edge_h], dim=1))
        mean, maximum, degree = segment_mean_max(messages, target, len(x))
        update = self.update(torch.cat([x, mean, maximum, torch.log1p(degree)[:, None]], dim=1))
        return self.norm(x + update)


class PositionAwareFragmentGraph(nn.Module):
    def __init__(self, variant: str, hidden: int, emb_dim: int, layers: int, local_layers: int, dropout: float) -> None:
        super().__init__()
        if variant not in VARIANTS:
            raise ValueError(variant)
        self.variant = variant
        self.use_chain = False
        self.use_ring = False
        self.use_ports = True
        self.use_local = True
        self.hidden = hidden
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)
        self.type_embedding = nn.Embedding(3, 12)
        self.base_encoder = nn.Sequential(
            nn.Linear(4 * emb_dim + 12 + 11, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        n_parts = 1
        if self.use_chain:
            self.chain_atom_pool = FeaturePool(emb_dim + 6, hidden, dropout)
            # center, symmetric side sum, symmetric side abs-difference, scalars
            self.chain_triple_pool = FeaturePool(3 * emb_dim + 3, hidden, dropout)
            self.endpoint_encoder = nn.Sequential(
                nn.Linear(2 * emb_dim + 4, hidden), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.SiLU(),
            )
            n_parts += 3
        if self.use_ring:
            self.ring_atom_pool = FeaturePool(emb_dim + 11, hidden, dropout)
            self.ring_distance_embedding = nn.Embedding(10, 8)
            self.ring_pair_pool = FeaturePool(2 * emb_dim + 8, hidden, dropout)
            n_parts += 2
        if self.use_local:
            self.local_node_encoder = nn.Sequential(
                nn.Linear(emb_dim + 9, hidden), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.SiLU(),
            )
            self.local_bond_encoder = nn.Sequential(nn.Linear(emb_dim, hidden), nn.SiLU())
            self.local_layers = nn.ModuleList([
                PositionAwareMessageLayer(hidden, dropout) for _ in range(local_layers)
            ])
            self.local_fragment_projection = nn.Sequential(
                nn.Linear(2 * hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.SiLU(),
            )
            n_parts += 1
        self.fragment_fusion = nn.Sequential(
            nn.Linear(n_parts * hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        if self.use_ports:
            self.port_encoder = nn.Sequential(
                nn.Linear(3 * emb_dim + 6, hidden), nn.SiLU(), nn.Dropout(dropout),
                nn.Linear(hidden, hidden), nn.SiLU(),
            )
        self.layers = nn.ModuleList([PositionAwareMessageLayer(hidden, dropout) for _ in range(layers)])
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + 4, hidden), nn.SiLU(), nn.Dropout(0.25),
            nn.Linear(hidden, 1),
        )

    def _atoms(self, fields: torch.Tensor) -> torch.Tensor:
        if len(fields) == 0:
            return self.type_embedding.weight.new_empty((0, self.atom_encoder.atom_embedding_list[0].embedding_dim))
        return self.atom_encoder(fields)

    def _bonds(self, fields: torch.Tensor) -> torch.Tensor:
        if len(fields) == 0:
            return self.type_embedding.weight.new_empty((0, self.bond_encoder.bond_embedding_list[0].embedding_dim))
        return self.bond_encoder(fields)

    def forward(self, batch: Any, mode: str) -> torch.Tensor:
        n = int(batch.num_nodes)
        atom_h = self._atoms(batch.atom_fields)
        atom_mean, atom_max, _ = segment_mean_max(atom_h, batch.atom_fragment_index, n)
        bond_h = self._bonds(batch.bond_fields)
        bond_mean, bond_max, _ = segment_mean_max(bond_h, batch.bond_fragment_index, n)
        parts = [self.base_encoder(torch.cat([
            atom_mean, atom_max, bond_mean, bond_max,
            self.type_embedding(batch.frag_type), batch.frag_scalar,
        ], dim=1))]

        if self.use_local:
            local_x = self.local_node_encoder(torch.cat([
                self._atoms(batch.local_atom_fields), batch.local_atom_scalar,
            ], dim=1))
            local_edge_h = self.local_bond_encoder(self._bonds(batch.local_edge_fields))
            for layer in self.local_layers:
                local_x = layer(local_x, batch.local_edge_index, local_edge_h)
            local_mean, local_max, _ = segment_mean_max(
                local_x, batch.local_atom_fragment_index, n
            )
            parts.append(self.local_fragment_projection(torch.cat([local_mean, local_max], dim=1)))

        if self.use_chain:
            chain_atom = self._atoms(batch.chain_atom_fields)
            chain_atom_h = self.chain_atom_pool(
                torch.cat([chain_atom, batch.chain_atom_scalar], dim=1),
                batch.chain_atom_fragment_index, n,
            )
            center = self._atoms(batch.triple_center_fields)
            left = self._atoms(batch.triple_left_atom_fields) + self._bonds(batch.triple_left_bond_fields)
            right = self._atoms(batch.triple_right_atom_fields) + self._bonds(batch.triple_right_bond_fields)
            triple_h = self.chain_triple_pool(
                torch.cat([center, left + right, torch.abs(left - right), batch.triple_scalar], dim=1),
                batch.triple_fragment_index, n,
            )
            endpoint_h = parts[0].new_zeros((n, self.hidden))
            if len(batch.endpoint_left_fields):
                left_end = self._atoms(batch.endpoint_left_fields)
                right_end = self._atoms(batch.endpoint_right_fields)
                encoded = self.endpoint_encoder(torch.cat([
                    left_end + right_end, torch.abs(left_end - right_end), batch.endpoint_scalar,
                ], dim=1))
                endpoint_h.index_add_(0, batch.endpoint_fragment_index, encoded)
            parts.extend([chain_atom_h, triple_h, endpoint_h])

        if self.use_ring:
            ring_atom = self._atoms(batch.ring_atom_fields)
            ring_atom_h = self.ring_atom_pool(
                torch.cat([ring_atom, batch.ring_atom_scalar], dim=1),
                batch.ring_atom_fragment_index, n,
            )
            left_ring = self._atoms(batch.ring_pair_left_fields)
            right_ring = self._atoms(batch.ring_pair_right_fields)
            distance = self.ring_distance_embedding(batch.ring_pair_distance)
            ring_pair_h = self.ring_pair_pool(
                torch.cat([left_ring + right_ring, torch.abs(left_ring - right_ring), distance], dim=1),
                batch.ring_pair_fragment_index, n,
            )
            parts.extend([ring_atom_h, ring_pair_h])

        x = self.fragment_fusion(torch.cat(parts, dim=1))
        if mode == "shuffled_connection":
            x = x[batch.shuffle_index]
        if mode != "fragment_set":
            if self.use_ports and batch.edge_index.shape[1]:
                prefix = "shuffled_" if mode == "shuffled_connection" else ""
                source_atom = getattr(batch, prefix + "edge_source_atom_fields")
                target_atom = getattr(batch, prefix + "edge_target_atom_fields")
                edge_bond = getattr(batch, prefix + "edge_bond_fields")
                edge_role_scalar = getattr(batch, prefix + "edge_role_scalar")
                edge_h = self.port_encoder(torch.cat([
                    self._atoms(source_atom), self._atoms(target_atom),
                    self._bonds(edge_bond), edge_role_scalar,
                ], dim=1))
            else:
                edge_h = x.new_zeros((batch.edge_index.shape[1], self.hidden))
            for layer in self.layers:
                x = layer(x, batch.edge_index, edge_h)

        graph_mean, graph_max, count = segment_mean_max(x, batch.batch, int(batch.num_graphs))
        type_prop = x.new_zeros((int(batch.num_graphs), 3))
        type_prop.index_add_(0, batch.batch, F.one_hot(batch.frag_type, 3).to(x.dtype))
        type_prop = type_prop / count.clamp_min(1.0)[:, None]
        summary = torch.cat([torch.log1p(count)[:, None] / 4.0, type_prop], dim=1)
        return self.head(torch.cat([graph_mean, graph_max, summary], dim=1)).view(-1)


def make_model(args: argparse.Namespace, variant: str, mode: str, device: torch.device) -> PositionAwareFragmentGraph:
    model = PositionAwareFragmentGraph(variant, args.hidden, args.emb_dim, args.layers, args.local_layers, args.dropout).to(device)
    if mode == "fragment_set":
        for parameter in model.layers.parameters():
            parameter.requires_grad_(False)
        if model.use_ports:
            for parameter in model.port_encoder.parameters():
                parameter.requires_grad_(False)
    return model


def make_loader(records: list[Data], rows: np.ndarray, batch_size: int, shuffle: bool, seed: int) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator().manual_seed(seed)
    return DataLoader([records[int(i)] for i in rows], batch_size=batch_size, shuffle=shuffle, generator=generator, num_workers=0)


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, variant: str, mode: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    ys, logits = [], []
    for batch in loader:
        batch = batch.to(device)
        ys.append(batch.y.view(-1).cpu().numpy())
        logits.append(model(batch, mode).cpu().numpy())
    return np.concatenate(ys), np.concatenate(logits)


def train_epoch(model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer, mode: str, device: torch.device, pos_weight: torch.Tensor) -> float:
    model.train()
    total_loss = 0.0
    total = 0
    for batch in loader:
        batch = batch.to(device)
        target = batch.y.view(-1).float()
        logits = model(batch, mode)
        loss = F.binary_cross_entropy_with_logits(logits, target, pos_weight=pos_weight)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        optimizer.step()
        total_loss += float(loss.item()) * len(target)
        total += len(target)
    return total_loss / max(total, 1)


def fit_select(
    records: list[Data], labels: np.ndarray, train_rows: np.ndarray, valid_rows: np.ndarray,
    variant: str, mode: str, args: argparse.Namespace, device: torch.device,
) -> tuple[int, list[dict[str, float]], dict[str, int]]:
    seed_everything(args.train_seed)
    model = make_model(args, variant, mode, device)
    counts = parameter_counts(model)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(records, train_rows, args.batch_size, True, args.train_seed + 11)
    valid_loader = make_loader(records, valid_rows, 2 * args.batch_size, False, args.train_seed + 12)
    n_pos = max(int(labels[train_rows].sum()), 1)
    pos_weight = torch.tensor((len(train_rows) - n_pos) / n_pos, dtype=torch.float32, device=device)
    history: list[dict[str, float]] = []
    best_auc, best_epoch, stale = -math.inf, 1, 0
    for epoch in range(1, args.max_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, mode, device, pos_weight)
        yv, lv = predict(model, valid_loader, variant, mode, device)
        auc = float(roc_auc_score(yv, lv))
        history.append({"epoch": float(epoch), "train_loss": loss, "valid_auc": auc})
        if auc > best_auc + 1e-7:
            best_auc, best_epoch, stale = auc, epoch, 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or epoch == args.max_epochs:
            print(f"  {variant}/{mode} epoch={epoch:02d} loss={loss:.5f} inner={auc:.6f} best={best_auc:.6f}@{best_epoch}", flush=True)
        if epoch >= args.min_epochs and stale >= args.patience:
            break
    return best_epoch, history, counts


def refit_eval(
    records: list[Data], labels: np.ndarray, fit_rows: np.ndarray, held_rows: np.ndarray,
    variant: str, mode: str, epochs: int, args: argparse.Namespace, device: torch.device,
) -> tuple[dict[str, Any], np.ndarray]:
    seed_everything(args.train_seed)
    model = make_model(args, variant, mode, device)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr, weight_decay=args.weight_decay)
    train_loader = make_loader(records, fit_rows, args.batch_size, True, args.train_seed + 21)
    fit_loader = make_loader(records, fit_rows, 2 * args.batch_size, False, args.train_seed + 22)
    held_loader = make_loader(records, held_rows, 2 * args.batch_size, False, args.train_seed + 23)
    n_pos = max(int(labels[fit_rows].sum()), 1)
    pos_weight = torch.tensor((len(fit_rows) - n_pos) / n_pos, dtype=torch.float32, device=device)
    final_loss = math.nan
    for epoch in range(1, epochs + 1):
        final_loss = train_epoch(model, train_loader, optimizer, mode, device, pos_weight)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  refit {variant}/{mode} {epoch:02d}/{epochs} loss={final_loss:.5f}", flush=True)
    yf, lf = predict(model, fit_loader, variant, mode, device)
    yh, lh = predict(model, held_loader, variant, mode, device)
    return {
        "selected_epoch": int(epochs),
        "final_train_loss": float(final_loss),
        "fit_auc": float(roc_auc_score(yf, lf)),
        "heldout_auc": float(roc_auc_score(yh, lh)),
        "heldout_logit_sha256": sha256_array(lh),
        **parameter_counts(model),
    }, lh


def run_self_test() -> None:
    atom = np.zeros((4, 9), dtype=np.int64)
    atom[:, 0] = 5
    edge_feat = np.zeros((6, 3), dtype=np.int64)
    edge_index = np.asarray([[0,1,1,2,2,3],[1,0,2,1,3,2]], dtype=np.int64)
    g = {"num_nodes": 4, "node_feat": atom, "edge_index": edge_index, "edge_feat": edge_feat}
    d, _ = brics_fragment_data(g, "CCCC", 0.0, 0, 1)
    assert d.local_atom_fields.shape[0] == 4 and d.local_edge_index.shape[1] == 6


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n8000_scaffold_folds3_seed20260726.npz")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--variants", nargs="+", choices=VARIANTS, default=list(VARIANTS))
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--split-seed", type=int, default=20260729)
    ap.add_argument("--train-seed", type=int, default=20260729)
    ap.add_argument("--shuffle-seed", type=int, default=20260729)
    ap.add_argument("--hidden", type=int, default=64)
    ap.add_argument("--emb-dim", type=int, default=24)
    ap.add_argument("--layers", type=int, default=1)
    ap.add_argument("--local-layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.20)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-epochs", type=int, default=30)
    ap.add_argument("--min-epochs", type=int, default=10)
    ap.add_argument("--patience", type=int, default=7)
    ap.add_argument(
        "--fixed-epochs", type=int, default=0,
        help="If positive, skip noisy inner selection and refit every requested model for this many epochs.",
    )
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--output", default="tracks/ksvd/results/molhiv/brics_fragment_n8000_fold1_screen_20260729.json")
    args = ap.parse_args()
    run_self_test()
    t0 = time.time()
    device = torch.device(args.device)
    _patch_torch_load_weights_only()
    from ogb.graphproppred import GraphPropPredDataset
    repo = Path(__file__).resolve().parents[3]
    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=str(repo / "data" / "ogb"))
    smiles_rows: list[str] = []
    with gzip.open(repo / "data" / "ogb" / "ogbg_molhiv" / "mapping" / "mol.csv.gz", "rt") as handle:
        for row in csv.DictReader(handle):
            smiles_rows.append(row["smiles"])
    if len(smiles_rows) != len(dataset):
        raise AssertionError("SMILES mapping length mismatch")

    with np.load(args.fold_cache, allow_pickle=False) as folds:
        original_indices = np.asarray(folds["original_indices"], dtype=np.int64)
        official_train_local = np.asarray(folds["official_train_indices"], dtype=np.int64)
        official_valid_local = np.asarray(folds["official_valid_indices"], dtype=np.int64)
        official_test_local = np.asarray(folds["official_test_indices"], dtype=np.int64)
        groups = np.asarray(folds["train_scaffold_groups"])
        fit_local = np.asarray(folds[f"fold_{args.fold}_train_indices"], dtype=np.int64)
        held_local = np.asarray(folds[f"fold_{args.fold}_valid_indices"], dtype=np.int64)
    local_to_row = np.full(len(original_indices), -1, dtype=np.int64)
    local_to_row[official_train_local] = np.arange(len(official_train_local), dtype=np.int64)
    fit_rows, held_rows = local_to_row[fit_local], local_to_row[held_local]
    if np.any(fit_rows < 0) or np.any(held_rows < 0):
        raise AssertionError("fold contains non-train local indices")
    if set(np.concatenate([fit_local, held_local]).tolist()) != set(official_train_local.tolist()):
        raise AssertionError("outer fold does not partition subset official train")

    records: list[Data] = []
    labels = np.empty(len(official_train_local), dtype=np.float32)
    stat_rows: list[dict[str, Any]] = []
    for row, local_index in enumerate(official_train_local.tolist()):
        raw_index = int(original_indices[local_index])
        graph_dict, raw_label = dataset[raw_index]
        label = float(np.asarray(raw_label).reshape(-1)[0])
        labels[row] = label
        data, stats = brics_fragment_data(
            graph_dict, smiles_rows[raw_index], label, raw_index, args.shuffle_seed
        )
        records.append(data)
        stat_rows.append(stats)
        if (row + 1) % 2000 == 0 or row + 1 == len(official_train_local):
            print(f"encoded {row+1}/{len(official_train_local)}", flush=True)

    # Check both fragment-index and local atom-index batching domains.
    probe = next(iter(DataLoader(records[:8], batch_size=8)))
    for key in ["atom_fragment_index", "bond_fragment_index", "local_atom_fragment_index", "shuffle_index"]:
        value = getattr(probe, key)
        if len(value) and (int(value.min()) < 0 or int(value.max()) >= int(probe.num_nodes)):
            raise AssertionError(f"invalid batched {key}")
    if len(probe.local_edge_index) and (
        int(probe.local_edge_index.min()) < 0
        or int(probe.local_edge_index.max()) >= int(probe.local_atom_fields.shape[0])
    ):
        raise AssertionError("invalid batched local_edge_index")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.split_seed + args.fold)
    inner_train_pos, inner_valid_pos = next(splitter.split(np.zeros(len(fit_rows)), labels[fit_rows], groups[fit_rows]))
    inner_train = fit_rows[np.asarray(inner_train_pos, dtype=np.int64)]
    inner_valid = fit_rows[np.asarray(inner_valid_pos, dtype=np.int64)]
    print(f"n={len(original_indices)} fold={args.fold} inner={len(inner_train)}/{len(inner_valid)} outer={len(fit_rows)}/{len(held_rows)}", flush=True)

    decomposition = {
        key: {"mean": float(np.mean([row[key] for row in stat_rows])), "max": int(max(row[key] for row in stat_rows))}
        for key in stat_rows[0]
    }
    out: dict[str, Any] = {
        "protocol_id": "molhiv-brics-chemical-block-graph-v1",
        "date": "2026-07-29",
        "scope": (
            "subset official-train graph items only; pre-fixed epoch; one outer scaffold fold"
            if args.fixed_epochs > 0 else
            "subset official-train graph items only; inner scaffold epoch selection; one outer scaffold fold"
        ),
        "n_subset": int(len(original_indices)),
        "n_train": int(len(official_train_local)),
        "n_positive": int(labels.sum()),
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "official_valid_graph_items_accessed": 0,
        "official_test_graph_items_accessed": 0,
        "matched_control_policy": "same initialization and minibatch seed within every variant",
        "variant_definitions": {
            "brics": "BRICS blocks + exact within-block atom topology + cut-bond endpoint roles",
        },
        "config": vars(args),
        "decomposition": decomposition,
        "fold": int(args.fold),
        "n_inner_train": int(len(inner_train)),
        "n_inner_valid": int(len(inner_valid)),
        "n_outer_fit": int(len(fit_rows)),
        "n_outer_held": int(len(held_rows)),
        "results": {},
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    for variant in args.variants:
        out["results"][variant] = {}
        for mode in args.modes:
            if args.fixed_epochs > 0:
                epoch = int(args.fixed_epochs)
                history: list[dict[str, float]] = []
                seed_everything(args.train_seed)
                counts = parameter_counts(make_model(args, variant, mode, device))
                print(f"fixed {variant}/{mode} epoch={epoch}", flush=True)
            else:
                print(f"select {variant}/{mode}", flush=True)
                epoch, history, counts = fit_select(
                    records, labels, inner_train, inner_valid, variant, mode, args, device
                )
            print(f"refit {variant}/{mode} epoch={epoch}", flush=True)
            result, held_logits = refit_eval(records, labels, fit_rows, held_rows, variant, mode, epoch, args, device)
            if counts != {k: result[k] for k in counts}:
                raise AssertionError("parameter count mismatch")
            result.update({
                "selection_policy": ("fixed" if args.fixed_epochs > 0 else "single_inner_scaffold"),
                "inner_best_auc": (
                    None if args.fixed_epochs > 0 else float(max(x["valid_auc"] for x in history))
                ),
                "inner_history": history,
                "heldout_logits": held_logits.tolist(),
            })
            out["results"][variant][mode] = result
            output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
            print(json.dumps({"variant": variant, "mode": mode, **{k: result[k] for k in ["selected_epoch","inner_best_auc","fit_auc","heldout_auc","trainable_parameters"]}}, indent=2), flush=True)
        vr = out["results"][variant]
        if "fragment_graph" in vr and "fragment_set" in vr:
            vr["true_minus_set"] = float(vr["fragment_graph"]["heldout_auc"] - vr["fragment_set"]["heldout_auc"])
        if "fragment_graph" in vr and "shuffled_connection" in vr:
            vr["true_minus_shuffled"] = float(vr["fragment_graph"]["heldout_auc"] - vr["shuffled_connection"]["heldout_auc"])
        output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    out["ranking"] = sorted([
        {"variant": variant, "mode": mode, "heldout_auc": result["heldout_auc"], "inner_best_auc": result["inner_best_auc"]}
        for variant, modes in out["results"].items() for mode, result in modes.items() if isinstance(result, dict) and "heldout_auc" in result
    ], key=lambda x: x["heldout_auc"], reverse=True)
    out["elapsed_sec"] = time.time() - t0
    output.write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "top": out["ranking"][:10], "elapsed_sec": out["elapsed_sec"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

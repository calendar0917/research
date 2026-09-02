"""Chemical fragment graph probe for ogbg-molhiv.

This is a deliberately new core representation rather than a GINE side branch.
Each molecule is deterministically decomposed into variable-size fragments:

* every connected system of cyclic (non-bridge) bonds is one ring-system node;
* every maximal path of bridge bonds between ring/branch/end atoms is one
  chain-segment node;
* isolated uncovered atoms become singleton nodes.

Fragments are connected when they share a boundary atom.  Full OGB atom and
bond categorical fields are encoded inside each fragment before two rounds of
fragment-level message passing.  There is no atom--atom neural message passing,
no path vocabulary, KSVD, prototype bank, or GINE backbone.

Three controls are trained under the same split and optimization protocol:

* fragment_set: pool fragment contents without fragment message passing;
* fragment_graph: message passing over the true fragment connections;
* shuffled_connection: preserve both the fragment multiset and graph topology,
  but deterministically permute which fragment content occupies each topology
  position before message passing.

The default run accesses graph items from official train only and evaluates the
most difficult outer scaffold fold 1.  An inner scaffold split selects epochs;
the outer held fold is not used for epoch selection.  Official valid/test are
not encoded or evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import networkx as nx
import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import _patch_torch_load_weights_only

import torch
from torch import nn
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader
from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder


FRAGMENT_TYPES = {"ring_system": 0, "chain_segment": 1, "singleton": 2}
MODES = ("fragment_set", "fragment_graph", "shuffled_connection")


@dataclass(frozen=True)
class Fragment:
    kind: int
    atoms: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]


def sha256_array(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def canonical_edge(u: int, v: int) -> tuple[int, int]:
    return (u, v) if u < v else (v, u)


def deterministic_fragments(n_nodes: int, edges: Iterable[tuple[int, int]]) -> list[Fragment]:
    """Partition bonds into ring systems and maximal non-ring chain segments.

    Atoms at fragment boundaries are intentionally shared: a ring attachment
    atom belongs to its ring system and the outgoing chain segment; a branch
    atom belongs to each incident chain segment.  This overlap induces the
    fragment graph without inventing learned fragment boundaries.
    """
    graph = nx.Graph()
    graph.add_nodes_from(range(n_nodes))
    graph.add_edges_from(canonical_edge(int(u), int(v)) for u, v in edges if u != v)
    all_edges = {canonical_edge(int(u), int(v)) for u, v in graph.edges()}
    bridges = {canonical_edge(int(u), int(v)) for u, v in nx.bridges(graph)}
    cyclic_edges = all_edges - bridges

    fragments: list[Fragment] = []
    ring_atoms: set[int] = set()
    if cyclic_edges:
        cyclic_graph = nx.Graph()
        cyclic_graph.add_edges_from(sorted(cyclic_edges))
        for component in sorted(nx.connected_components(cyclic_graph), key=lambda s: (min(s), len(s))):
            atoms = tuple(sorted(int(u) for u in component))
            atom_set = set(atoms)
            internal = tuple(sorted(e for e in cyclic_edges if e[0] in atom_set and e[1] in atom_set))
            fragments.append(Fragment(FRAGMENT_TYPES["ring_system"], atoms, internal))
            ring_atoms.update(atom_set)

    # Bridge bonds form a forest.  Split it at chemically structural boundaries:
    # ring atoms, branches, and molecular ends.  Degree-2 non-ring atoms remain
    # inside a maximal chain segment.
    visited_bridge_edges: set[tuple[int, int]] = set()
    boundaries = {
        u for u in range(n_nodes)
        if u in ring_atoms or graph.degree[u] != 2
    }

    def walk(start: int, first: int) -> tuple[list[int], list[tuple[int, int]]]:
        atoms = [start, first]
        path_edges = [canonical_edge(start, first)]
        prev, cur = start, first
        while cur not in boundaries:
            candidates = [
                int(v) for v in graph.neighbors(cur)
                if int(v) != prev and canonical_edge(cur, int(v)) in bridges
            ]
            if len(candidates) != 1:
                break
            nxt = candidates[0]
            edge = canonical_edge(cur, nxt)
            if edge in visited_bridge_edges or edge in path_edges:
                break
            atoms.append(nxt)
            path_edges.append(edge)
            prev, cur = cur, nxt
        return atoms, path_edges

    # Starting from every boundary guarantees deterministic maximal paths.
    for start in sorted(boundaries):
        for first in sorted(int(v) for v in graph.neighbors(start)):
            edge = canonical_edge(start, first)
            if edge not in bridges or edge in visited_bridge_edges:
                continue
            atoms, path_edges = walk(start, first)
            visited_bridge_edges.update(path_edges)
            fragments.append(Fragment(
                FRAGMENT_TYPES["chain_segment"], tuple(atoms), tuple(path_edges)
            ))

    # Defensive completion for unusual disconnected inputs or any bridge forest
    # component not reached above.
    for edge in sorted(bridges - visited_bridge_edges):
        atoms, path_edges = walk(edge[0], edge[1])
        visited_bridge_edges.update(path_edges)
        fragments.append(Fragment(
            FRAGMENT_TYPES["chain_segment"], tuple(atoms), tuple(path_edges)
        ))

    covered = {u for fragment in fragments for u in fragment.atoms}
    for atom in range(n_nodes):
        if atom not in covered:
            fragments.append(Fragment(FRAGMENT_TYPES["singleton"], (atom,), ()))

    fragments.sort(key=lambda f: (f.kind, min(f.atoms), len(f.atoms), f.atoms))
    if not fragments and n_nodes:
        raise AssertionError("non-empty molecule produced no fragments")
    if visited_bridge_edges != bridges:
        raise AssertionError("not every bridge bond was assigned to a chain segment")
    assigned_cyclic = {edge for f in fragments if f.kind == FRAGMENT_TYPES["ring_system"] for edge in f.edges}
    if assigned_cyclic != cyclic_edges:
        raise AssertionError("not every cyclic bond was assigned to a ring system")
    return fragments


def fragment_connections(fragments: list[Fragment]) -> list[tuple[int, int, int]]:
    """Return directed (source, target, shared_atom) fragment connections."""
    atom_memberships: dict[int, list[int]] = {}
    for fi, fragment in enumerate(fragments):
        for atom in fragment.atoms:
            atom_memberships.setdefault(atom, []).append(fi)
    directed: list[tuple[int, int, int]] = []
    for atom, memberships in sorted(atom_memberships.items()):
        unique = sorted(set(memberships))
        for source in unique:
            for target in unique:
                if source != target:
                    directed.append((source, target, atom))
    return directed


def fragment_scalars(
    fragment: Fragment,
    graph: nx.Graph,
    ring_atoms: set[int],
) -> np.ndarray:
    atoms = fragment.atoms
    degrees = np.asarray([graph.degree[u] for u in atoms], dtype=np.float32)
    degree_hist = np.asarray([
        np.mean(degrees == 0),
        np.mean(degrees == 1),
        np.mean(degrees == 2),
        np.mean(degrees == 3),
        np.mean(degrees >= 4),
    ], dtype=np.float32)
    n = len(atoms)
    m = len(fragment.edges)
    cycle_rank = max(0, m - n + 1)
    return np.concatenate([
        np.asarray([
            math.log1p(n) / 4.0,
            math.log1p(m) / 4.0,
            float(np.mean([u in ring_atoms for u in atoms])),
            float(degrees.mean() / 4.0) if len(degrees) else 0.0,
            float(degrees.max() / 4.0) if len(degrees) else 0.0,
            math.log1p(cycle_rank) / 2.0,
        ], dtype=np.float32),
        degree_hist,
    ])


def graph_to_fragment_data(
    graph_dict: dict[str, Any], label: float, graph_index: int, shuffle_seed: int
) -> tuple[Data, dict[str, Any]]:
    n_nodes = int(graph_dict["num_nodes"])
    node_feat = np.asarray(graph_dict["node_feat"], dtype=np.int64)
    edge_index_raw = np.asarray(graph_dict["edge_index"], dtype=np.int64)
    edge_feat_raw = np.asarray(graph_dict["edge_feat"], dtype=np.int64)
    if node_feat.shape != (n_nodes, 9):
        raise ValueError(f"unexpected node feature shape {node_feat.shape}")

    edge_features: dict[tuple[int, int], np.ndarray] = {}
    for j in range(edge_index_raw.shape[1]):
        u, v = int(edge_index_raw[0, j]), int(edge_index_raw[1, j])
        if u == v:
            continue
        edge_features.setdefault(canonical_edge(u, v), edge_feat_raw[j].copy())
    undirected_edges = sorted(edge_features)
    fragments = deterministic_fragments(n_nodes, undirected_edges)

    nx_graph = nx.Graph()
    nx_graph.add_nodes_from(range(n_nodes))
    nx_graph.add_edges_from(undirected_edges)
    bridges = {canonical_edge(int(u), int(v)) for u, v in nx.bridges(nx_graph)}
    ring_atoms = {u for e in set(undirected_edges) - bridges for u in e}

    atom_fields: list[np.ndarray] = []
    atom_fragment: list[int] = []
    bond_fields: list[np.ndarray] = []
    bond_fragment: list[int] = []
    scalars: list[np.ndarray] = []
    types: list[int] = []
    for fi, fragment in enumerate(fragments):
        types.append(fragment.kind)
        scalars.append(fragment_scalars(fragment, nx_graph, ring_atoms))
        for atom in fragment.atoms:
            atom_fields.append(node_feat[atom])
            atom_fragment.append(fi)
        for edge in fragment.edges:
            bond_fields.append(edge_features[edge])
            bond_fragment.append(fi)

    connections = fragment_connections(fragments)
    if connections:
        connection_index = np.asarray([(u, v) for u, v, _ in connections], dtype=np.int64).T
    else:
        connection_index = np.empty((2, 0), dtype=np.int64)

    rng = np.random.default_rng(np.uint64(shuffle_seed) ^ np.uint64((graph_index + 1) * 0x9E3779B1))
    permutation = rng.permutation(len(fragments)).astype(np.int64)
    if len(fragments) > 1 and np.array_equal(permutation, np.arange(len(fragments))):
        permutation = np.roll(permutation, 1)

    data = Data(
        num_nodes=len(fragments),
        frag_type=torch.tensor(types, dtype=torch.long),
        frag_scalar=torch.tensor(np.stack(scalars), dtype=torch.float32),
        atom_fields=torch.tensor(np.stack(atom_fields), dtype=torch.long),
        atom_fragment_index=torch.tensor(atom_fragment, dtype=torch.long),
        bond_fields=torch.tensor(
            np.stack(bond_fields) if bond_fields else np.empty((0, 3), dtype=np.int64),
            dtype=torch.long,
        ),
        bond_fragment_index=torch.tensor(bond_fragment, dtype=torch.long),
        edge_index=torch.tensor(connection_index, dtype=torch.long),
        shuffle_index=torch.tensor(permutation, dtype=torch.long),
        y=torch.tensor([float(label)], dtype=torch.float32),
        graph_index=torch.tensor([int(graph_index)], dtype=torch.long),
    )
    stats = {
        "n_atoms": n_nodes,
        "n_bonds": len(undirected_edges),
        "n_fragments": len(fragments),
        "n_ring_systems": sum(f.kind == FRAGMENT_TYPES["ring_system"] for f in fragments),
        "n_chain_segments": sum(f.kind == FRAGMENT_TYPES["chain_segment"] for f in fragments),
        "n_singletons": sum(f.kind == FRAGMENT_TYPES["singleton"] for f in fragments),
        "n_directed_connections": len(connections),
        "max_fragment_atoms": max(len(f.atoms) for f in fragments),
    }
    return data, stats


def segment_mean_max(values: torch.Tensor, index: torch.Tensor, n: int) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if values.ndim != 2:
        raise ValueError("segment values must be a matrix")
    sums = values.new_zeros((n, values.shape[1]))
    counts = values.new_zeros(n)
    if len(index):
        sums.index_add_(0, index, values)
        counts.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    means = sums / counts.clamp_min(1.0)[:, None]
    maxima = values.new_full((n, values.shape[1]), -torch.inf)
    if len(index):
        maxima.scatter_reduce_(
            0, index[:, None].expand_as(values), values, reduce="amax", include_self=True
        )
    maxima = torch.where(torch.isfinite(maxima), maxima, torch.zeros_like(maxima))
    return means, maxima, counts


class FragmentMessageLayer(nn.Module):
    def __init__(self, hidden: int, dropout: float) -> None:
        super().__init__()
        self.message = nn.Sequential(
            nn.Linear(hidden, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.update = nn.Sequential(
            nn.Linear(3 * hidden + 1, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden),
        )
        self.norm = nn.LayerNorm(hidden)

    def forward(self, x: torch.Tensor, edge_index: torch.Tensor) -> torch.Tensor:
        if edge_index.shape[1] == 0:
            return x
        source, target = edge_index[0], edge_index[1]
        messages = self.message(x[source])
        mean, maximum, degree = segment_mean_max(messages, target, len(x))
        update = self.update(torch.cat([x, mean, maximum, torch.log1p(degree)[:, None]], dim=1))
        return self.norm(x + update)


class ChemicalFragmentGraph(nn.Module):
    def __init__(self, hidden: int = 80, emb_dim: int = 24, layers: int = 2, dropout: float = 0.15) -> None:
        super().__init__()
        self.atom_encoder = AtomEncoder(emb_dim)
        self.bond_encoder = BondEncoder(emb_dim)
        self.type_embedding = nn.Embedding(len(FRAGMENT_TYPES), 12)
        scalar_dim = 11
        fragment_input = 4 * emb_dim + 12 + scalar_dim
        self.fragment_encoder = nn.Sequential(
            nn.Linear(fragment_input, hidden), nn.SiLU(), nn.Dropout(dropout),
            nn.Linear(hidden, hidden), nn.SiLU(),
        )
        self.layers = nn.ModuleList([
            FragmentMessageLayer(hidden, dropout) for _ in range(layers)
        ])
        self.head = nn.Sequential(
            nn.Linear(2 * hidden + 4, hidden), nn.SiLU(), nn.Dropout(0.25),
            nn.Linear(hidden, 1),
        )

    def forward(self, batch: Any, mode: str) -> torch.Tensor:
        if mode not in MODES:
            raise ValueError(f"unknown mode {mode}")
        n_fragments = int(batch.num_nodes)
        atom_h = self.atom_encoder(batch.atom_fields)
        atom_mean, atom_max, _ = segment_mean_max(
            atom_h, batch.atom_fragment_index, n_fragments
        )
        if len(batch.bond_fields):
            bond_h = self.bond_encoder(batch.bond_fields)
        else:
            bond_h = atom_h.new_empty((0, atom_h.shape[1]))
        bond_mean, bond_max, _ = segment_mean_max(
            bond_h, batch.bond_fragment_index, n_fragments
        )
        x = self.fragment_encoder(torch.cat([
            atom_mean, atom_max, bond_mean, bond_max,
            self.type_embedding(batch.frag_type), batch.frag_scalar,
        ], dim=1))

        if mode == "shuffled_connection":
            x = x[batch.shuffle_index]
        if mode != "fragment_set":
            for layer in self.layers:
                x = layer(x, batch.edge_index)

        n_graphs = int(batch.num_graphs)
        graph_mean, graph_max, fragment_count = segment_mean_max(x, batch.batch, n_graphs)
        type_prop = x.new_zeros((n_graphs, len(FRAGMENT_TYPES)))
        type_prop.index_add_(0, batch.batch, F.one_hot(
            batch.frag_type, num_classes=len(FRAGMENT_TYPES)
        ).to(x.dtype))
        type_prop = type_prop / fragment_count.clamp_min(1.0)[:, None]
        summary = torch.cat([
            torch.log1p(fragment_count)[:, None] / 4.0, type_prop
        ], dim=1)
        return self.head(torch.cat([graph_mean, graph_max, summary], dim=1)).view(-1)


def make_model(args: argparse.Namespace, mode: str, device: torch.device) -> ChemicalFragmentGraph:
    model = ChemicalFragmentGraph(
        hidden=args.hidden, emb_dim=args.emb_dim, layers=args.layers, dropout=args.dropout
    ).to(device)
    if mode == "fragment_set":
        for parameter in model.layers.parameters():
            parameter.requires_grad_(False)
    return model


def parameter_counts(model: nn.Module) -> dict[str, int]:
    return {
        "total_parameters": int(sum(p.numel() for p in model.parameters())),
        "trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }


def make_loader(
    records: list[Data], rows: np.ndarray, batch_size: int, shuffle: bool, seed: int
) -> DataLoader:
    generator = None
    if shuffle:
        generator = torch.Generator()
        generator.manual_seed(seed)
    return DataLoader(
        [records[int(i)] for i in rows], batch_size=batch_size, shuffle=shuffle,
        generator=generator, num_workers=0,
    )


@torch.no_grad()
def predict(model: nn.Module, loader: DataLoader, mode: str, device: torch.device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    labels: list[np.ndarray] = []
    logits: list[np.ndarray] = []
    for batch in loader:
        batch = batch.to(device)
        labels.append(batch.y.view(-1).cpu().numpy())
        logits.append(model(batch, mode).cpu().numpy())
    return np.concatenate(labels), np.concatenate(logits)


def train_epoch(
    model: nn.Module, loader: DataLoader, optimizer: torch.optim.Optimizer,
    mode: str, device: torch.device, pos_weight: torch.Tensor,
) -> float:
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


def optimizer_for(model: nn.Module, args: argparse.Namespace) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr,
        weight_decay=args.weight_decay,
    )


def select_epoch(
    records: list[Data], labels: np.ndarray, train_rows: np.ndarray, valid_rows: np.ndarray,
    mode: str, args: argparse.Namespace, seed: int, device: torch.device,
) -> tuple[int, list[dict[str, Any]], dict[str, int]]:
    seed_everything(seed)
    model = make_model(args, mode, device)
    counts = parameter_counts(model)
    optimizer = optimizer_for(model, args)
    train_loader = make_loader(records, train_rows, args.batch_size, True, seed + 11)
    valid_loader = make_loader(records, valid_rows, args.batch_size * 2, False, seed + 12)
    n_pos = max(int(labels[train_rows].sum()), 1)
    n_neg = len(train_rows) - n_pos
    pos_weight = torch.tensor(float(n_neg / n_pos), device=device)
    history: list[dict[str, Any]] = []
    best_epoch = 1
    best_auc = -math.inf
    stale = 0
    for epoch in range(1, args.max_epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, mode, device, pos_weight)
        yv, lv = predict(model, valid_loader, mode, device)
        auc = float(roc_auc_score(yv, lv))
        history.append({"epoch": epoch, "train_loss": loss, "valid_auc": auc})
        if auc > best_auc + 1e-7:
            best_auc = auc
            best_epoch = epoch
            stale = 0
        else:
            stale += 1
        if epoch == 1 or epoch % 5 == 0 or epoch == args.max_epochs:
            print(f"  {mode} epoch={epoch:02d} loss={loss:.5f} inner_auc={auc:.6f} best={best_auc:.6f}@{best_epoch}", flush=True)
        if epoch >= args.min_epochs and stale >= args.patience:
            print(f"  {mode} early stop after epoch {epoch}; selected={best_epoch}", flush=True)
            break
    return best_epoch, history, counts


def refit_and_evaluate(
    records: list[Data], labels: np.ndarray, fit_rows: np.ndarray, held_rows: np.ndarray,
    mode: str, epochs: int, args: argparse.Namespace, seed: int, device: torch.device,
) -> tuple[dict[str, Any], np.ndarray, dict[str, int]]:
    seed_everything(seed)
    model = make_model(args, mode, device)
    counts = parameter_counts(model)
    optimizer = optimizer_for(model, args)
    train_loader = make_loader(records, fit_rows, args.batch_size, True, seed + 21)
    fit_eval_loader = make_loader(records, fit_rows, args.batch_size * 2, False, seed + 22)
    held_loader = make_loader(records, held_rows, args.batch_size * 2, False, seed + 23)
    n_pos = max(int(labels[fit_rows].sum()), 1)
    n_neg = len(fit_rows) - n_pos
    pos_weight = torch.tensor(float(n_neg / n_pos), device=device)
    losses = []
    for epoch in range(1, epochs + 1):
        loss = train_epoch(model, train_loader, optimizer, mode, device, pos_weight)
        losses.append(loss)
        if epoch == 1 or epoch % 5 == 0 or epoch == epochs:
            print(f"  refit {mode} epoch={epoch:02d}/{epochs} loss={loss:.5f}", flush=True)
    yf, lf = predict(model, fit_eval_loader, mode, device)
    yh, lh = predict(model, held_loader, mode, device)
    result = {
        "epochs": int(epochs),
        "final_train_loss": float(losses[-1]),
        "fit_auc": float(roc_auc_score(yf, lf)),
        "heldout_auc": float(roc_auc_score(yh, lh)),
        "heldout_logit_sha256": sha256_array(lh),
        **counts,
    }
    return result, lh, counts


def run_self_tests() -> None:
    # Four-cycle -> one ring system.
    fs = deterministic_fragments(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert [(f.kind, f.atoms) for f in fs] == [(FRAGMENT_TYPES["ring_system"], (0, 1, 2, 3))]
    # Linear chain -> one maximal chain segment.
    fs = deterministic_fragments(4, [(0, 1), (1, 2), (2, 3)])
    assert len(fs) == 1 and fs[0].atoms in ((0, 1, 2, 3), (3, 2, 1, 0))
    # Three-arm branch -> three chain segments sharing center; directed clique has 6 edges.
    fs = deterministic_fragments(4, [(0, 1), (0, 2), (0, 3)])
    assert len(fs) == 3 and len(fragment_connections(fs)) == 6
    # Ring with a two-bond substituent -> ring + one chain connected at ring atom.
    fs = deterministic_fragments(6, [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5)])
    assert len(fs) == 2 and len(fragment_connections(fs)) == 2


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold-cache", default="tracks/ksvd/results/molhiv/molhiv_n41127_scaffold_folds3_seed20260726.npz")
    ap.add_argument("--fold", type=int, default=1)
    ap.add_argument("--seed", type=int, default=20260729)
    ap.add_argument("--shuffle-seed", type=int, default=20260729)
    ap.add_argument("--hidden", type=int, default=80)
    ap.add_argument("--emb-dim", type=int, default=24)
    ap.add_argument("--layers", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--batch-size", type=int, default=256)
    ap.add_argument("--max-epochs", type=int, default=40)
    ap.add_argument("--min-epochs", type=int, default=12)
    ap.add_argument("--patience", type=int, default=10)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--modes", nargs="+", choices=MODES, default=list(MODES))
    ap.add_argument("--output", default="tracks/ksvd/results/molhiv/chemical_fragment_graph_fold1_probe_20260729.json")
    args = ap.parse_args()
    if args.fold not in (0, 1, 2):
        raise ValueError("fold must be 0, 1, or 2")
    run_self_tests()
    t0 = time.time()
    device = torch.device(args.device)
    _patch_torch_load_weights_only()
    from ogb.graphproppred import GraphPropPredDataset

    repo = Path(__file__).resolve().parents[3]
    dataset = GraphPropPredDataset(name="ogbg-molhiv", root=str(repo / "data" / "ogb"))
    split = dataset.get_idx_split()
    official_train = np.asarray(split["train"], dtype=np.int64)
    official_valid = np.asarray(split["valid"], dtype=np.int64)
    official_test = np.asarray(split["test"], dtype=np.int64)
    labels_all = np.asarray(dataset.labels).reshape(-1)
    labels = labels_all[official_train].astype(np.float32)
    global_to_row = np.full(len(dataset), -1, dtype=np.int64)
    global_to_row[official_train] = np.arange(len(official_train), dtype=np.int64)

    with np.load(args.fold_cache, allow_pickle=False) as folds:
        if not np.array_equal(np.asarray(folds["official_train_indices"], dtype=np.int64), official_train):
            raise ValueError("fold cache official train mismatch")
        groups = np.asarray(folds["train_scaffold_groups"])
        fit_global = np.asarray(folds[f"fold_{args.fold}_train_indices"], dtype=np.int64)
        held_global = np.asarray(folds[f"fold_{args.fold}_valid_indices"], dtype=np.int64)
    fit_rows = global_to_row[fit_global]
    held_rows = global_to_row[held_global]
    if np.any(fit_rows < 0) or np.any(held_rows < 0):
        raise AssertionError("outer fold contains non-train graphs")
    if set(np.concatenate([fit_global, held_global]).tolist()) != set(official_train.tolist()):
        raise AssertionError("outer fold does not partition official train")
    if np.intersect1d(fit_global, official_valid).size or np.intersect1d(fit_global, official_test).size:
        raise AssertionError("fit leaks official valid/test")

    records: list[Data] = []
    stat_rows: list[dict[str, Any]] = []
    for row, raw_index in enumerate(official_train.tolist()):
        graph_dict, raw_label = dataset[int(raw_index)]
        label = float(np.asarray(raw_label).reshape(-1)[0])
        if label != float(labels[row]):
            raise AssertionError("label mismatch")
        data, stats = graph_to_fragment_data(graph_dict, label, int(raw_index), args.shuffle_seed)
        records.append(data)
        stat_rows.append(stats)
        if (row + 1) % 5000 == 0 or row + 1 == len(official_train):
            print(f"fragmentized {row+1}/{len(official_train)}", flush=True)

    # Validate PyG's batching increments all custom fragment index vectors.
    probe = next(iter(DataLoader(records[: min(8, len(records))], batch_size=min(8, len(records)))))
    if len(probe.atom_fragment_index) and int(probe.atom_fragment_index.max()) >= int(probe.num_nodes):
        raise AssertionError("atom_fragment_index batching increment is invalid")
    if len(probe.bond_fragment_index) and int(probe.bond_fragment_index.max()) >= int(probe.num_nodes):
        raise AssertionError("bond_fragment_index batching increment is invalid")
    if len(probe.shuffle_index) and (int(probe.shuffle_index.min()) < 0 or int(probe.shuffle_index.max()) >= int(probe.num_nodes)):
        raise AssertionError("shuffle_index batching increment is invalid")

    splitter = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=args.seed + args.fold)
    inner_train_pos, inner_valid_pos = next(splitter.split(
        np.zeros(len(fit_rows)), labels[fit_rows], groups[fit_rows]
    ))
    inner_train = fit_rows[np.asarray(inner_train_pos, dtype=np.int64)]
    inner_valid = fit_rows[np.asarray(inner_valid_pos, dtype=np.int64)]
    print(
        f"outer fold {args.fold}: inner_train={len(inner_train)} inner_valid={len(inner_valid)} "
        f"outer_fit={len(fit_rows)} outer_held={len(held_rows)}",
        flush=True,
    )

    numeric_stats = {key: np.asarray([row[key] for row in stat_rows], dtype=np.float64) for key in stat_rows[0]}
    decomposition = {
        key: {
            "mean": float(values.mean()),
            "median": float(np.median(values)),
            "p95": float(np.quantile(values, 0.95)),
            "max": float(values.max()),
        }
        for key, values in numeric_stats.items()
    }
    out: dict[str, Any] = {
        "protocol_id": "molhiv-chemical-fragment-graph-v1",
        "date": "2026-07-29",
        "scope": "official-train graph items only; inner scaffold epoch selection; outer scaffold fold evaluation",
        "representation": {
            "ring_system": "connected component of cyclic/non-bridge bonds",
            "chain_segment": "maximal bridge-bond path split at ring atoms, branches, and molecular ends",
            "connection": "fragments share a boundary atom",
            "atom_message_passing": False,
            "gine_backbone": False,
            "ksvd_or_path_vocabulary": False,
            "uses_all_ogb_atom_fields": True,
            "uses_all_ogb_bond_fields": True,
        },
        "controls": {
            "fragment_set": "same fragment contents, no fragment message passing",
            "fragment_graph": "true fragment content-to-connection assignment",
            "shuffled_connection": "same fragment multiset and topology; content permuted across topology positions",
        },
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "official_valid_graph_items_accessed": 0,
        "official_test_graph_items_accessed": 0,
        "config": vars(args),
        "training_seed_policy": "all structural controls use the same initialization and minibatch seed",
        "n_official_train": int(len(official_train)),
        "n_positive": int(labels.sum()),
        "fold": int(args.fold),
        "n_inner_train": int(len(inner_train)),
        "n_inner_valid": int(len(inner_valid)),
        "n_outer_fit": int(len(fit_rows)),
        "n_outer_held": int(len(held_rows)),
        "decomposition": decomposition,
        "results": {},
    }

    for mode in args.modes:
        print(f"selecting epoch for {mode}", flush=True)
        selected_epoch, history, counts = select_epoch(
            records, labels, inner_train, inner_valid, mode, args,
            args.seed, device,
        )
        print(f"refitting {mode} for {selected_epoch} epochs", flush=True)
        result, held_logits, counts2 = refit_and_evaluate(
            records, labels, fit_rows, held_rows, mode, selected_epoch, args,
            args.seed, device,
        )
        if counts != counts2:
            raise AssertionError("parameter count changed between selection and refit")
        result.update({
            "selected_epoch": int(selected_epoch),
            "inner_best_auc": float(max(row["valid_auc"] for row in history)),
            "inner_history": history,
            "heldout_logits": held_logits.tolist(),
        })
        out["results"][mode] = result
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
        print(json.dumps({"mode": mode, **{k: result[k] for k in [
            "selected_epoch", "inner_best_auc", "fit_auc", "heldout_auc",
            "trainable_parameters", "total_parameters",
        ]}}, indent=2), flush=True)

    if "fragment_graph" in out["results"] and "fragment_set" in out["results"]:
        out["true_graph_minus_set"] = float(
            out["results"]["fragment_graph"]["heldout_auc"] -
            out["results"]["fragment_set"]["heldout_auc"]
        )
    if "fragment_graph" in out["results"] and "shuffled_connection" in out["results"]:
        out["true_graph_minus_shuffled"] = float(
            out["results"]["fragment_graph"]["heldout_auc"] -
            out["results"]["shuffled_connection"]["heldout_auc"]
        )
    out["elapsed_sec"] = time.time() - t0
    Path(args.output).write_text(json.dumps(out, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "aucs": {mode: row["heldout_auc"] for mode, row in out["results"].items()},
        "true_graph_minus_set": out.get("true_graph_minus_set"),
        "true_graph_minus_shuffled": out.get("true_graph_minus_shuffled"),
        "elapsed_sec": out["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

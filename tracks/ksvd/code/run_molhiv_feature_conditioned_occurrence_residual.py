"""Bounded feature-conditioned occurrence correction on a frozen strong base.

The frozen base is the probability ensemble of the farthest and
scaffold-facility broad occurrence predictors.  A low-capacity branch uses
original OGB atom/bond fields inside real-patch occurrences and can exchange
one shallow message between occurrences.  Its output is a zero-gated, bounded
logit correction; the base predictions are never retrained.

Development is restricted to official-train scaffold folds.  Official
valid/test graphs must remain unencoded and are never evaluated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import load_molhiv


@dataclass(frozen=True)
class Control:
    name: str
    atom_layers: int
    occurrence_layers: int
    prediction_level: str
    use_prototype: bool = True


def _sha256(x: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(x).tobytes()).hexdigest()


def _seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def _normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(
        x >= 0.0,
        1.0 / (1.0 + np.exp(-x)),
        np.exp(x) / (1.0 + np.exp(x)),
    )


def _load_broad_pair(
    broad: dict[str, Any], split: str, variant: str
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rows = [
        broad["results"][f"{family}__distance_1_2__real"][split]
        for family in ("farthest", "scaffold_facility")
    ]
    field = "base_scores" if variant == "occurrence_pair" else "scores"
    graph_indices = np.asarray(rows[0]["graph_indices"], dtype=np.int64)
    labels = np.asarray(rows[0]["labels"], dtype=np.float32)
    probabilities = []
    for row in rows:
        if not np.array_equal(
            graph_indices, np.asarray(row["graph_indices"], dtype=np.int64)
        ):
            raise ValueError("broad family graph order mismatch")
        if not np.array_equal(labels, np.asarray(row["labels"], dtype=np.float32)):
            raise ValueError("broad family labels mismatch")
        probabilities.append(_sigmoid(np.asarray(row[field], dtype=np.float32)))
    probability = np.clip(np.mean(probabilities, axis=0), 1e-6, 1.0 - 1e-6)
    logits = np.log(probability / (1.0 - probability)).astype(np.float32)
    return graph_indices, labels, logits


def _select_farthest(candidates: np.ndarray, n_select: int) -> np.ndarray:
    """Deterministic cosine farthest-point selection."""
    if n_select > len(candidates):
        raise ValueError("n_select exceeds candidate count")
    centroid = _normalize_rows(candidates.mean(axis=0, keepdims=True))[0]
    selected = [int(np.argmax(candidates @ centroid))]
    min_distance = 1.0 - candidates @ candidates[selected[0]]
    for _ in range(1, n_select):
        min_distance[np.asarray(selected, dtype=np.int64)] = -np.inf
        choice = int(np.argmax(min_distance))
        selected.append(choice)
        min_distance = np.minimum(
            min_distance, 1.0 - candidates @ candidates[choice]
        )
    return np.asarray(selected, dtype=np.int64)


def _sparse_positive_codes(
    latents: np.ndarray, prototypes: np.ndarray, sparsity: int
) -> np.ndarray:
    cosine = latents @ prototypes.T
    positive = np.maximum(cosine, 0.0)
    if sparsity < positive.shape[1]:
        indices = np.argpartition(positive, -sparsity, axis=1)[:, -sparsity:]
        codes = np.zeros_like(positive)
        rows = np.arange(len(positive))[:, None]
        codes[rows, indices] = positive[rows, indices]
    else:
        codes = positive.copy()
    empty = np.flatnonzero(codes.max(axis=1) <= 0.0)
    if len(empty):
        nearest = cosine[empty].argmax(axis=1)
        codes[empty, nearest] = 1e-6
    return codes.astype(np.float32)


def _bond_one_hot(feature: np.ndarray, dims: list[int]) -> np.ndarray:
    out = np.zeros(sum(dims), dtype=np.float32)
    offset = 0
    for raw_value, dim in zip(np.asarray(feature).reshape(-1), dims):
        value = int(raw_value)
        if value < 0 or value >= dim:
            raise ValueError(f"bond field value {value} outside [0,{dim})")
        out[offset + value] = 1.0
        offset += dim
    return out


def _build_occurrence_graph(
    graph: Any,
    node_fields: np.ndarray,
    edge_fields: dict[tuple[int, int], np.ndarray],
    latents: np.ndarray,
    prototypes: np.ndarray,
    sparsity: int,
    bond_dims: list[int],
    graph_index: int,
) -> dict[str, np.ndarray]:
    """Build connected prototype occurrences with atom/bond-conditioned fields."""
    codes = _sparse_positive_codes(latents, prototypes, sparsity)
    n_nodes, n_prototypes = codes.shape
    bond_hot_dim = sum(bond_dims)

    proto_ids: list[int] = []
    scalar_rows: list[list[float]] = []
    internal_bond_rows: list[np.ndarray] = []
    member_nodes: list[int] = []
    member_occurrences: list[int] = []
    member_weights: list[float] = []
    node_to_occurrences: list[list[int]] = [[] for _ in range(n_nodes)]

    for proto in range(n_prototypes):
        active_nodes = np.flatnonzero(codes[:, proto] > 0.0)
        unseen = set(int(x) for x in active_nodes.tolist())
        while unseen:
            start = unseen.pop()
            stack = [start]
            component = [start]
            while stack:
                u = stack.pop()
                for v in graph.neighbors(u):
                    if v in unseen:
                        unseen.remove(v)
                        stack.append(v)
                        component.append(v)
            component_np = np.asarray(component, dtype=np.int64)
            component_set = set(component)
            coeff = codes[component_np, proto]
            occurrence = len(proto_ids)
            proto_ids.append(proto)
            scalar_rows.append([
                float(np.log1p(len(component))),
                float(coeff.mean()),
                float(coeff.max()),
                float(coeff.sum()),
            ])

            internal_sum = np.zeros(bond_hot_dim, dtype=np.float32)
            internal_count = 0
            for u, v in graph.edges():
                if u in component_set and v in component_set:
                    key = (u, v) if u < v else (v, u)
                    internal_sum += _bond_one_hot(edge_fields[key], bond_dims)
                    internal_count += 1
            if internal_count:
                internal_sum /= float(internal_count)
            internal_bond_rows.append(np.concatenate([
                np.asarray([np.log1p(internal_count)], dtype=np.float32),
                internal_sum,
            ]))

            for node in component:
                member_nodes.append(node)
                member_occurrences.append(occurrence)
                member_weights.append(float(codes[node, proto]))
                node_to_occurrences[node].append(occurrence)

    n_occurrences = len(proto_ids)
    if not n_occurrences:
        raise RuntimeError(f"graph {graph_index} produced no occurrence")

    # Each unordered occurrence relation stores overlap count, connecting-bond
    # count and a normalized histogram of the actual connecting bond fields.
    relation: dict[tuple[int, int], dict[str, Any]] = {}

    def relation_row(a: int, b: int) -> dict[str, Any] | None:
        if a == b:
            return None
        key = (a, b) if a < b else (b, a)
        return relation.setdefault(key, {
            "overlap": 0,
            "bond_count": 0,
            "bond_hot": np.zeros(bond_hot_dim, dtype=np.float32),
        })

    for occurrences in node_to_occurrences:
        for a, b in combinations(sorted(set(occurrences)), 2):
            row = relation_row(a, b)
            assert row is not None
            row["overlap"] += 1

    for u, v in graph.edges():
        key = (u, v) if u < v else (v, u)
        hot = _bond_one_hot(edge_fields[key], bond_dims)
        for a in node_to_occurrences[u]:
            for b in node_to_occurrences[v]:
                row = relation_row(a, b)
                if row is None:
                    continue
                row["bond_count"] += 1
                row["bond_hot"] += hot

    upper_src: list[int] = []
    upper_dst: list[int] = []
    upper_attr: list[np.ndarray] = []
    for (a, b), row in relation.items():
        bond_count = int(row["bond_count"])
        bond_hot = np.asarray(row["bond_hot"], dtype=np.float32)
        if bond_count:
            bond_hot = bond_hot / float(bond_count)
        continuous = np.asarray([
            float(row["overlap"] > 0),
            float(np.log1p(row["overlap"])),
            float(bond_count > 0),
            float(np.log1p(bond_count)),
        ], dtype=np.float32)
        attr = np.concatenate([continuous, bond_hot])
        upper_src.extend([a, b])
        upper_dst.extend([b, a])
        upper_attr.extend([attr, attr])

    atom_src: list[int] = []
    atom_dst: list[int] = []
    atom_edge_fields: list[np.ndarray] = []
    for u, v in graph.edges():
        key = (u, v) if u < v else (v, u)
        field = np.asarray(edge_fields[key], dtype=np.int64)
        atom_src.extend([u, v])
        atom_dst.extend([v, u])
        atom_edge_fields.extend([field, field])

    return {
        "atom_fields": np.asarray(node_fields, dtype=np.int64),
        "atom_edge_index": (
            np.asarray([atom_src, atom_dst], dtype=np.int64)
            if atom_src else np.empty((2, 0), dtype=np.int64)
        ),
        "atom_edge_fields": (
            np.stack(atom_edge_fields).astype(np.int64)
            if atom_edge_fields else np.empty((0, len(bond_dims)), dtype=np.int64)
        ),
        "prototype_id": np.asarray(proto_ids, dtype=np.int64),
        "scalars": np.asarray(scalar_rows, dtype=np.float32),
        "internal_bonds": np.asarray(internal_bond_rows, dtype=np.float32),
        "member_atom": np.asarray(member_nodes, dtype=np.int64),
        "member_occurrence": np.asarray(member_occurrences, dtype=np.int64),
        "member_weight": np.asarray(member_weights, dtype=np.float32),
        "upper_edge_index": (
            np.asarray([upper_src, upper_dst], dtype=np.int64)
            if upper_src else np.empty((2, 0), dtype=np.int64)
        ),
        "upper_edge_attr": (
            np.asarray(upper_attr, dtype=np.float32)
            if upper_attr else np.empty((0, 4 + bond_hot_dim), dtype=np.float32)
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-cache", required=True)
    ap.add_argument("--fold-cache", required=True)
    ap.add_argument("--vocabulary-result", required=True)
    ap.add_argument("--broad-result", required=True)
    ap.add_argument("--prototype-family", default="farthest", choices=("farthest", "scaffold_facility"))
    ap.add_argument("--base-variant", default="occurrence_pair", choices=("occurrence_pair", "relation_pair"))
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--n-prototypes", type=int, default=32)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=32)
    ap.add_argument("--residual-cap", type=float, default=0.25)
    ap.add_argument("--residual-l2", type=float, default=0.10)
    ap.add_argument("--objective", default="pairwise", choices=("pairwise", "bce"))
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--task-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument(
        "--controls",
        default=(
            "atom1_pool_residual,atom1_occ_mil_continuous,"
            "atom1_occ_gnn_continuous,atom1_occ_gnn_no_proto"
        ),
    )
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    known = {
        "atom1_pool_residual": Control("atom1_pool_residual", 1, 0, "atom", False),
        "atom1_occ_mil_continuous": Control(
            "atom1_occ_mil_continuous", 1, 0, "occurrence", True
        ),
        "atom1_occ_gnn_continuous": Control(
            "atom1_occ_gnn_continuous", 1, 1, "occurrence", True
        ),
        "atom1_occ_gnn_no_proto": Control(
            "atom1_occ_gnn_no_proto", 1, 1, "occurrence", False
        ),
    }
    names = [x.strip() for x in args.controls.split(",") if x.strip()]
    if not names or len(set(names)) != len(names) or any(x not in known for x in names):
        raise ValueError(f"invalid controls: {names}")
    if args.residual_cap <= 0 or args.residual_l2 < 0:
        raise ValueError("invalid residual regularization")
    controls = [known[x] for x in names]

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from ogb.graphproppred.mol_encoder import AtomEncoder, BondEncoder
        from ogb.utils.features import get_bond_feature_dims
        from torch.utils.data import DataLoader, Dataset
        from torch_geometric.nn import (
            GINEConv,
            global_max_pool,
            global_mean_pool,
        )
        from torch_geometric.utils import softmax as graph_softmax
    except ImportError as exc:
        raise RuntimeError(f"missing training dependency: {exc}") from exc

    t0 = time.time()
    _seed_everything(args.seed, torch)
    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=True,
    )
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise AssertionError("atom/bond features were not loaded")
    labels = np.asarray(bundle.y, dtype=np.float32)
    original_indices = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    official_train = np.asarray(bundle.split["train"], dtype=np.int64)
    official_valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    official_test = np.asarray(bundle.split["test"], dtype=np.int64)

    with np.load(args.fold_cache, allow_pickle=False) as folds:
        fit_indices = np.asarray(folds[f"fold_{args.fold}_train_indices"], dtype=np.int64)
        heldout_indices = np.asarray(folds[f"fold_{args.fold}_valid_indices"], dtype=np.int64)
        if not np.array_equal(np.asarray(folds["original_indices"]), original_indices):
            raise ValueError("fold original indices mismatch")
        if not np.array_equal(np.asarray(folds["official_train_indices"]), official_train):
            raise ValueError("fold official train mismatch")
    if np.intersect1d(fit_indices, heldout_indices).size:
        raise AssertionError("fit/heldout overlap")
    if set(np.concatenate([fit_indices, heldout_indices]).tolist()) != set(official_train.tolist()):
        raise AssertionError("fold does not partition official train")

    with np.load(args.latent_cache, allow_pickle=False) as source:
        offsets = np.asarray(source["offsets"], dtype=np.int64)
        latents = np.asarray(source["latents"], dtype=np.float32)
        checks = [
            (np.asarray(source["original_indices"]), original_indices, "original"),
            (np.asarray(source["train_indices"]), official_train, "train"),
            (np.asarray(source["valid_indices"]), official_valid, "valid"),
            (np.asarray(source["test_indices"]), official_test, "test"),
        ]
    for actual, expected, name in checks:
        if not np.array_equal(actual, expected):
            raise ValueError(f"latent {name} mismatch")
    official_nontrain_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in np.concatenate([official_valid, official_test])
    ])
    if np.any(latents[official_nontrain_rows] != 0):
        raise AssertionError("official valid/test latents were encoded")

    fit_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in fit_indices
    ])
    fit_latents = _normalize_rows(latents[fit_rows])
    if np.mean(np.linalg.norm(fit_latents, axis=1) > 0.99) < 0.999:
        raise ValueError("missing fit latents")

    vocabulary = json.loads(Path(args.vocabulary_result).read_text())
    broad = json.loads(Path(args.broad_result).read_text())
    expected_fit_sha = _sha256(fit_indices)
    for name, doc in (("vocabulary", vocabulary), ("broad", broad)):
        if int(doc.get("fold", -1)) != args.fold:
            raise ValueError(f"{name} fold mismatch")
        if doc.get("fit_indices_sha256") != expected_fit_sha:
            raise ValueError(f"{name} fit hash mismatch")

    selection_doc = vocabulary["results"][args.prototype_family]["selection"]
    source_rows = np.asarray(selection_doc["source_node_rows"], dtype=np.int64)
    if len(source_rows) != args.n_prototypes:
        raise ValueError("vocabulary prototype count mismatch")
    prototypes = _normalize_rows(latents[source_rows])
    if _sha256(prototypes) != selection_doc["prototype_sha256"]:
        raise ValueError("reconstructed prototype hash mismatch")

    fit_base_graphs, fit_base_y, fit_base_logits = _load_broad_pair(
        broad, "fit", args.base_variant
    )
    heldout_base_graphs, heldout_base_y, heldout_base_logits = _load_broad_pair(
        broad, "heldout", args.base_variant
    )
    if not np.array_equal(fit_base_graphs, fit_indices):
        raise ValueError("base fit graph order mismatch")
    if not np.array_equal(heldout_base_graphs, heldout_indices):
        raise ValueError("base heldout graph order mismatch")
    if not np.array_equal(fit_base_y, labels[fit_indices]):
        raise ValueError("base fit labels mismatch")
    if not np.array_equal(heldout_base_y, labels[heldout_indices]):
        raise ValueError("base heldout labels mismatch")
    base_logits_by_graph = {
        int(i): float(score)
        for i, score in zip(
            np.concatenate([fit_base_graphs, heldout_base_graphs]),
            np.concatenate([fit_base_logits, heldout_base_logits]),
        )
    }
    base_fit_auc = float(roc_auc_score(labels[fit_indices], fit_base_logits))
    base_heldout_auc = float(
        roc_auc_score(labels[heldout_indices], heldout_base_logits)
    )
    print(
        f"frozen {args.base_variant} base fit_auc={base_fit_auc:.6f} "
        f"heldout_auc={base_heldout_auc:.6f}",
        flush=True,
    )

    bond_dims = [int(x) for x in get_bond_feature_dims()]
    all_train = np.concatenate([fit_indices, heldout_indices])
    preprocessing_start = time.time()
    graphs: dict[int, dict[str, np.ndarray]] = {}
    for count, raw_i in enumerate(all_train, start=1):
        i = int(raw_i)
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        graphs[i] = _build_occurrence_graph(
            bundle.graphs[i],
            np.asarray(bundle.node_feats[i], dtype=np.int64),
            bundle.edge_feats[i],
            latents[lo:hi],
            prototypes,
            args.sparsity,
            bond_dims,
            i,
        )
        if count % 1000 == 0 or count == len(all_train):
            print(f"preprocessed {count}/{len(all_train)} graphs", flush=True)
    preprocessing_sec = time.time() - preprocessing_start

    occ_counts = np.asarray([len(graphs[int(i)]["prototype_id"]) for i in all_train])
    upper_edges = np.asarray([graphs[int(i)]["upper_edge_index"].shape[1] for i in all_train])
    memberships = np.asarray([len(graphs[int(i)]["member_atom"]) for i in all_train])
    occurrence_stats = {
        "mean_occurrences_per_graph": float(occ_counts.mean()),
        "max_occurrences_per_graph": int(occ_counts.max()),
        "mean_directed_upper_edges_per_graph": float(upper_edges.mean()),
        "mean_memberships_per_graph": float(memberships.mean()),
    }
    print(f"occurrence audit: {occurrence_stats}", flush=True)

    class GraphIndexDataset(Dataset):
        def __init__(self, indices: np.ndarray):
            self.indices = np.asarray(indices, dtype=np.int64)

        def __len__(self) -> int:
            return int(len(self.indices))

        def __getitem__(self, index: int) -> int:
            return int(self.indices[index])

    def collate(graph_indices: list[int]) -> dict[str, torch.Tensor]:
        atom_fields: list[np.ndarray] = []
        atom_edges: list[np.ndarray] = []
        atom_edge_fields: list[np.ndarray] = []
        atom_batch: list[np.ndarray] = []
        prototype_ids: list[np.ndarray] = []
        scalars: list[np.ndarray] = []
        internal_bonds: list[np.ndarray] = []
        member_atom: list[np.ndarray] = []
        member_occ: list[np.ndarray] = []
        member_weight: list[np.ndarray] = []
        upper_edges: list[np.ndarray] = []
        upper_attrs: list[np.ndarray] = []
        occurrence_batch: list[np.ndarray] = []
        atom_offset = 0
        occ_offset = 0
        for batch_i, raw_i in enumerate(graph_indices):
            row = graphs[int(raw_i)]
            n_atom = len(row["atom_fields"])
            n_occ = len(row["prototype_id"])
            atom_fields.append(row["atom_fields"])
            atom_edges.append(row["atom_edge_index"] + atom_offset)
            atom_edge_fields.append(row["atom_edge_fields"])
            atom_batch.append(np.full(n_atom, batch_i, dtype=np.int64))
            prototype_ids.append(row["prototype_id"])
            scalars.append(row["scalars"])
            internal_bonds.append(row["internal_bonds"])
            member_atom.append(row["member_atom"] + atom_offset)
            member_occ.append(row["member_occurrence"] + occ_offset)
            member_weight.append(row["member_weight"])
            upper_edges.append(row["upper_edge_index"] + occ_offset)
            upper_attrs.append(row["upper_edge_attr"])
            occurrence_batch.append(np.full(n_occ, batch_i, dtype=np.int64))
            atom_offset += n_atom
            occ_offset += n_occ
        return {
            "atom_fields": torch.from_numpy(np.concatenate(atom_fields)),
            "atom_edge_index": torch.from_numpy(np.concatenate(atom_edges, axis=1)),
            "atom_edge_fields": torch.from_numpy(np.concatenate(atom_edge_fields)),
            "atom_batch": torch.from_numpy(np.concatenate(atom_batch)),
            "prototype_id": torch.from_numpy(np.concatenate(prototype_ids)),
            "scalars": torch.from_numpy(np.concatenate(scalars)),
            "internal_bonds": torch.from_numpy(np.concatenate(internal_bonds)),
            "member_atom": torch.from_numpy(np.concatenate(member_atom)),
            "member_occurrence": torch.from_numpy(np.concatenate(member_occ)),
            "member_weight": torch.from_numpy(np.concatenate(member_weight)),
            "upper_edge_index": torch.from_numpy(np.concatenate(upper_edges, axis=1)),
            "upper_edge_attr": torch.from_numpy(np.concatenate(upper_attrs)),
            "occurrence_batch": torch.from_numpy(np.concatenate(occurrence_batch)),
            "labels": torch.from_numpy(labels[np.asarray(graph_indices, dtype=np.int64)]),
            "base_logits": torch.tensor(
                [base_logits_by_graph[int(i)] for i in graph_indices],
                dtype=torch.float32,
            ),
        }

    class Readout(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.attention = nn.Linear(args.hidden, 1)
            self.head = nn.Sequential(
                nn.Linear(3 * args.hidden, args.hidden),
                nn.LayerNorm(args.hidden),
                nn.SiLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden, 1),
            )

        def forward(self, h: torch.Tensor, batch: torch.Tensor) -> torch.Tensor:
            weights = graph_softmax(self.attention(h).view(-1), batch)
            n_graphs = int(batch.max()) + 1
            attended = h.new_zeros((n_graphs, args.hidden))
            attended.index_add_(0, batch, weights[:, None] * h)
            mean = global_mean_pool(h, batch)
            maximum = global_max_pool(h, batch)
            return self.head(torch.cat([attended, mean, maximum], dim=1)).view(-1)

    class FeatureConditionedModel(nn.Module):
        def __init__(self, control: Control) -> None:
            super().__init__()
            self.control = control
            self.atom_encoder = AtomEncoder(args.hidden)
            self.bond_encoder = BondEncoder(args.hidden)
            if control.atom_layers:
                atom_mlp = nn.Sequential(
                    nn.Linear(args.hidden, 2 * args.hidden),
                    nn.LayerNorm(2 * args.hidden),
                    nn.SiLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(2 * args.hidden, args.hidden),
                )
                self.atom_conv = GINEConv(atom_mlp, train_eps=True)
                self.atom_norm = nn.LayerNorm(args.hidden)
            if control.prediction_level == "occurrence":
                self.register_buffer(
                    "prototype_vectors", torch.from_numpy(prototypes.astype(np.float32))
                )
                if control.use_prototype:
                    self.prototype_projection = nn.Sequential(
                        nn.Linear(prototypes.shape[1], args.hidden),
                        nn.LayerNorm(args.hidden),
                        nn.SiLU(),
                    )
                self.scalar_encoder = nn.Sequential(
                    nn.Linear(4, args.hidden), nn.LayerNorm(args.hidden), nn.SiLU()
                )
                self.internal_bond_encoder = nn.Sequential(
                    nn.Linear(1 + sum(bond_dims), args.hidden),
                    nn.LayerNorm(args.hidden),
                    nn.SiLU(),
                )
                self.occurrence_encoder = nn.Sequential(
                    nn.Linear(5 * args.hidden, 2 * args.hidden),
                    nn.LayerNorm(2 * args.hidden),
                    nn.SiLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(2 * args.hidden, args.hidden),
                    nn.SiLU(),
                )
                if control.occurrence_layers:
                    self.upper_edge_encoder = nn.Sequential(
                        nn.Linear(4 + sum(bond_dims), args.hidden),
                        nn.LayerNorm(args.hidden),
                        nn.SiLU(),
                        nn.Linear(args.hidden, args.hidden),
                    )
                    upper_mlp = nn.Sequential(
                        nn.Linear(args.hidden, 2 * args.hidden),
                        nn.LayerNorm(2 * args.hidden),
                        nn.SiLU(),
                        nn.Dropout(args.dropout),
                        nn.Linear(2 * args.hidden, args.hidden),
                    )
                    self.upper_conv = GINEConv(upper_mlp, train_eps=True)
                    self.upper_norm = nn.LayerNorm(args.hidden)
            self.readout = Readout()
            # Exact-zero initial correction without a zero scalar gate.  The
            # final readout learns graph-dependent corrections immediately; a
            # zero gate previously encouraged a constant calibration shift.
            nn.init.zeros_(self.readout.head[-1].weight)
            nn.init.zeros_(self.readout.head[-1].bias)

        def _combine(
            self, raw_correction: torch.Tensor, batch: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, torch.Tensor]:
            correction = args.residual_cap * torch.tanh(raw_correction)
            return batch["base_logits"] + correction, correction

        def forward(
            self, batch: dict[str, torch.Tensor]
        ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
            atom_h = self.atom_encoder(batch["atom_fields"])
            if self.control.atom_layers and batch["atom_edge_index"].shape[1] > 0:
                bond_h = self.bond_encoder(batch["atom_edge_fields"])
                update = self.atom_conv(atom_h, batch["atom_edge_index"], bond_h)
                atom_h = self.atom_norm(atom_h + F.dropout(
                    F.silu(update), p=args.dropout, training=self.training
                ))
            if self.control.prediction_level == "atom":
                raw_correction = self.readout(atom_h, batch["atom_batch"])
                logits, correction = self._combine(raw_correction, batch)
                return logits, atom_h, correction

            member_h = atom_h[batch["member_atom"]]
            weight = batch["member_weight"].clamp_min(1e-6)
            n_occ = int(batch["prototype_id"].shape[0])
            weighted_sum = atom_h.new_zeros((n_occ, args.hidden))
            weighted_sum.index_add_(
                0, batch["member_occurrence"], weight[:, None] * member_h
            )
            mass = atom_h.new_zeros(n_occ)
            mass.index_add_(0, batch["member_occurrence"], weight)
            weighted_mean = weighted_sum / mass.clamp_min(1e-6)[:, None]

            weighted_member = weight[:, None] * member_h
            weighted_max = atom_h.new_full((n_occ, args.hidden), -torch.inf)
            weighted_max.scatter_reduce_(
                0,
                batch["member_occurrence"][:, None].expand_as(weighted_member),
                weighted_member,
                reduce="amax",
                include_self=True,
            )
            weighted_max = torch.nan_to_num(weighted_max, neginf=0.0)
            if self.control.use_prototype:
                proto_h = self.prototype_projection(
                    self.prototype_vectors[batch["prototype_id"]]
                )
            else:
                proto_h = torch.zeros_like(weighted_mean)
            scalar_h = self.scalar_encoder(batch["scalars"])
            internal_h = self.internal_bond_encoder(batch["internal_bonds"])
            occ_h = self.occurrence_encoder(torch.cat([
                weighted_mean, weighted_max, proto_h, scalar_h, internal_h
            ], dim=1))
            if self.control.occurrence_layers and batch["upper_edge_index"].shape[1] > 0:
                upper_edge_h = self.upper_edge_encoder(batch["upper_edge_attr"])
                update = self.upper_conv(
                    occ_h, batch["upper_edge_index"], upper_edge_h
                )
                occ_h = self.upper_norm(occ_h + F.dropout(
                    F.silu(update), p=args.dropout, training=self.training
                ))
            raw_correction = self.readout(occ_h, batch["occurrence_batch"])
            logits, correction = self._combine(raw_correction, batch)
            return logits, occ_h, correction

    device = torch.device(args.device)

    def make_loader(indices: np.ndarray, shuffle: bool, offset: int) -> Any:
        generator = torch.Generator().manual_seed(args.seed + offset)
        return DataLoader(
            GraphIndexDataset(indices),
            batch_size=args.batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            num_workers=args.num_workers,
            collate_fn=collate,
        )

    def move(batch: dict[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        return {k: v.to(device) for k, v in batch.items()}

    @torch.no_grad()
    def evaluate(model: nn.Module, loader: Any) -> dict[str, Any]:
        model.eval()
        scores: list[np.ndarray] = []
        base_scores: list[np.ndarray] = []
        corrections: list[np.ndarray] = []
        ys: list[np.ndarray] = []
        representation_norm = 0.0
        representation_count = 0
        for batch in loader:
            batch = move(batch)
            logits, h, correction = model(batch)
            scores.append(logits.cpu().numpy())
            base_scores.append(batch["base_logits"].cpu().numpy())
            corrections.append(correction.cpu().numpy())
            ys.append(batch["labels"].cpu().numpy())
            representation_norm += float(h.norm(dim=1).sum())
            representation_count += int(h.shape[0])
        score = np.concatenate(scores)
        base_score = np.concatenate(base_scores)
        correction = np.concatenate(corrections)
        y = np.concatenate(ys)
        return {
            "auc": float(roc_auc_score(y, score)),
            "base_auc": float(roc_auc_score(y, base_score)),
            "delta_auc": float(roc_auc_score(y, score) - roc_auc_score(y, base_score)),
            "n_graphs": int(len(y)),
            "n_positive": int(y.sum()),
            "score_sha256": _sha256(score.astype(np.float64)),
            "correction_sha256": _sha256(correction.astype(np.float64)),
            "correction_mean": float(correction.mean()),
            "correction_std": float(correction.std()),
            "correction_abs_max": float(np.abs(correction).max()),
            "scores": score.astype(np.float32).tolist(),
            "base_scores": base_score.astype(np.float32).tolist(),
            "corrections": correction.astype(np.float32).tolist(),
            "mean_local_representation_norm": representation_norm / max(representation_count, 1),
        }

    results: dict[str, Any] = {
        "base": {
            "fit": {"auc": base_fit_auc},
            "heldout": {"auc": base_heldout_auc},
            "trainable_parameters": 0,
            "frozen": True,
        }
    }
    for control in controls:
        control_start = time.time()
        _seed_everything(args.seed, torch)
        model = FeatureConditionedModel(control).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.task_lr, weight_decay=args.weight_decay
        )
        train_loader = make_loader(fit_indices, True, 1000)
        history: list[dict[str, float]] = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            seen = 0
            total_bce = 0.0
            total_pairwise = 0.0
            total_penalty = 0.0
            for batch in train_loader:
                batch = move(batch)
                logits, _, correction = model(batch)
                y = batch["labels"].float()
                bce = F.binary_cross_entropy_with_logits(logits, y)
                positive = logits[y > 0.5]
                negative = logits[y <= 0.5]
                if len(positive) and len(negative):
                    pairwise = F.softplus(
                        -(positive[:, None] - negative[None, :])
                    ).mean()
                else:
                    pairwise = logits.sum() * 0.0
                task_loss = pairwise if args.objective == "pairwise" else bce
                penalty = correction.square().mean()
                loss = task_loss + args.residual_l2 * penalty
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                n = int(batch["labels"].shape[0])
                total_loss += float(loss.detach()) * n
                total_bce += float(bce.detach()) * n
                total_pairwise += float(pairwise.detach()) * n
                total_penalty += float(penalty.detach()) * n
                seen += n
            row = {
                "epoch": epoch,
                "loss": total_loss / max(seen, 1),
                "bce": total_bce / max(seen, 1),
                "pairwise": total_pairwise / max(seen, 1),
                "correction_square": total_penalty / max(seen, 1),
            }
            history.append(row)
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                print(
                    f"{control.name} epoch={epoch:02d} loss={row['loss']:.6f} "
                    f"pairwise={row['pairwise']:.6f}",
                    flush=True,
                )
        fit_metrics = evaluate(model, make_loader(fit_indices, False, 2000))
        heldout_metrics = evaluate(model, make_loader(heldout_indices, False, 3000))
        results[control.name] = {
            "atom_message_passing_layers": control.atom_layers,
            "occurrence_message_passing_layers": control.occurrence_layers,
            "prediction_level": control.prediction_level,
            "use_prototype": control.use_prototype,
            "trainable_parameters": int(sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )),
            "maximum_residual_amplitude": args.residual_cap,
            "fit": fit_metrics,
            "heldout": heldout_metrics,
            "history": history,
            "elapsed_sec": time.time() - control_start,
        }
        print(
            f"{control.name} fit_auc={fit_metrics['auc']:.6f} "
            f"heldout_auc={heldout_metrics['auc']:.6f}",
            flush=True,
        )

    def delta(a: str, b: str) -> float | None:
        if a not in results or b not in results:
            return None
        return float(results[a]["heldout"]["auc"] - results[b]["heldout"]["auc"])

    comparisons = {
        "atom1_pool_residual_minus_base": delta("atom1_pool_residual", "base"),
        "atom1_occ_mil_continuous_minus_base": delta(
            "atom1_occ_mil_continuous", "base"
        ),
        "atom1_occ_gnn_continuous_minus_base": delta(
            "atom1_occ_gnn_continuous", "base"
        ),
        "atom1_occ_gnn_no_proto_minus_base": delta(
            "atom1_occ_gnn_no_proto", "base"
        ),
        "occ_gnn_continuous_minus_occ_mil_continuous": delta(
            "atom1_occ_gnn_continuous", "atom1_occ_mil_continuous"
        ),
        "occ_gnn_continuous_minus_atom_pool": delta(
            "atom1_occ_gnn_continuous", "atom1_pool_residual"
        ),
        "occ_gnn_continuous_minus_no_proto": delta(
            "atom1_occ_gnn_continuous", "atom1_occ_gnn_no_proto"
        ),
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol_id": "molhiv-frozen-broad-feature-conditioned-occurrence-residual-scaffold-v2",
        "date": "2026-07-29",
        "hypothesis": (
            "a bounded atom/bond-conditioned occurrence correction can add stable "
            "information beyond a frozen broad occurrence probability ensemble"
        ),
        "data_policy": {
            "fit": "requested official-train outer scaffold fit fold",
            "heldout": "requested official-train outer scaffold heldout fold",
            "official_valid_encoded_or_evaluated": False,
            "official_test_encoded_or_evaluated": False,
            "epoch_selection": f"none; evaluate once after fixed epoch {args.epochs}",
        },
        "architecture": {
            "base": (
                "fully frozen probability ensemble of farthest and scaffold-facility "
                f"{args.base_variant}"
            ),
            "vocabulary": f"audited broad {args.prototype_family} observed-patch bank",
            "prototype_representation": (
                "fixed PCA64 prototype vector followed by a learned projection; "
                "no free prototype-ID embedding"
            ),
            "occurrence": "connected component of positive top-k prototype support",
            "atom_features": "full OGB AtomEncoder fields with at most one GINE layer",
            "bond_features": "full OGB BondEncoder plus internal/upper-edge bond histograms",
            "occurrence_node": (
                "weighted mean/max atom states + continuous prototype vector + "
                "occurrence scalars + internal bond histogram"
            ),
            "upper_edges": "shared atoms and/or molecular-bond adjacency",
            "correction": (
                "zero-initialized final readout; tanh-bounded graph residual with "
                f"maximum absolute logit correction {args.residual_cap}"
            ),
        },
        "config": vars(args),
        "fold": int(args.fold),
        "n_fit": int(len(fit_indices)),
        "n_fit_positive": int(labels[fit_indices].sum()),
        "n_heldout": int(len(heldout_indices)),
        "n_heldout_positive": int(labels[heldout_indices].sum()),
        "fit_indices_sha256": _sha256(fit_indices),
        "heldout_indices_sha256": _sha256(heldout_indices),
        "prototype_source_rows": source_rows.tolist(),
        "prototype_source_rows_sha256": _sha256(source_rows),
        "prototype_sha256": _sha256(prototypes),
        "frozen_base": {
            "fit_auc": base_fit_auc,
            "heldout_auc": base_heldout_auc,
            "variant": args.base_variant,
        },
        "occurrence_stats": occurrence_stats,
        "preprocessing_sec": preprocessing_sec,
        "results": results,
        "comparisons": comparisons,
        "elapsed_sec": time.time() - t0,
        "output": str(output),
    }
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "fold": args.fold,
        "heldout_auc": {k: v["heldout"]["auc"] for k, v in results.items()},
        "comparisons": comparisons,
        "elapsed_sec": report["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

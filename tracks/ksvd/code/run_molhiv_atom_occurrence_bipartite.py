"""Strict scaffold-fold pilot for a GNN-free atom--occurrence bipartite model.

This runner tests a cleaner division of labor than the historical
"three-layer node GINE + KSVD side channel" architecture:

* frozen local prototypes define sparse node occurrences;
* spatially connected support components become explicit occurrence nodes;
* atoms and occurrences exchange messages over the sparse membership relation;
* no supervised atom--atom or occurrence--occurrence GNN is present;
* the supervised model therefore cannot bypass the prototype construction by
  relearning the original molecular graph with a full GINE backbone.

Farthest observed prototypes, random observed prototypes, PCA directions and
KSVD directions use the same cosine Top-K occurrence construction. KSVD
shuffled-ID and no-ID controls test whether persistent dictionary identity
matters beyond generic sparse transport. Only official-train scaffold folds
are accepted; official-valid/test are never evaluated or encoded by this
runner.
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
class ControlSpec:
    name: str
    family: str
    level: str
    use_gine: bool
    use_identity: bool
    shuffled_identity: bool
    gated_gine: bool


def _parse_controls(raw: str) -> list[ControlSpec]:
    known = {
        "random_node_mil": ControlSpec(
            "random_node_mil", "random", "node", False, True, False, False
        ),
        "ksvd_node_mil": ControlSpec(
            "ksvd_node_mil", "ksvd", "node", False, True, False, False
        ),
        "ksvd_medoid_node_mil": ControlSpec(
            "ksvd_medoid_node_mil", "ksvd_medoid", "node", False, True, False, False
        ),
        "random_occ_mil": ControlSpec(
            "random_occ_mil", "random", "occurrence", False, True, False, False
        ),
        "random_occ_gine": ControlSpec(
            "random_occ_gine", "random", "occurrence", True, True, False, False
        ),
        "ksvd_occ_mil": ControlSpec(
            "ksvd_occ_mil", "ksvd", "occurrence", False, True, False, False
        ),
        "ksvd_occ_gine": ControlSpec(
            "ksvd_occ_gine", "ksvd", "occurrence", True, True, False, False
        ),
        "ksvd_occ_gine_shuffled_id": ControlSpec(
            "ksvd_occ_gine_shuffled_id", "ksvd", "occurrence", True, True, True, False
        ),
        "ksvd_occ_gine_no_id": ControlSpec(
            "ksvd_occ_gine_no_id", "ksvd", "occurrence", True, False, False, False
        ),
        "random_occ_gated_gine": ControlSpec(
            "random_occ_gated_gine", "random", "occurrence", True, True, False, True
        ),
        "ksvd_occ_gated_gine": ControlSpec(
            "ksvd_occ_gated_gine", "ksvd", "occurrence", True, True, False, True
        ),
        "farthest_node_mil": ControlSpec(
            "farthest_node_mil", "farthest", "node", False, True, False, False
        ),
        "farthest_bipartite": ControlSpec(
            "farthest_bipartite", "farthest", "bipartite", False, True, False, False
        ),
        "random_bipartite": ControlSpec(
            "random_bipartite", "random", "bipartite", False, True, False, False
        ),
        "random_direction_bipartite": ControlSpec(
            "random_direction_bipartite", "random_direction", "bipartite", False, True, False, False
        ),
        "pca_bipartite": ControlSpec(
            "pca_bipartite", "pca", "bipartite", False, True, False, False
        ),
        "ksvd_bipartite": ControlSpec(
            "ksvd_bipartite", "ksvd", "bipartite", False, True, False, False
        ),
        "ksvd_medoid_bipartite": ControlSpec(
            "ksvd_medoid_bipartite", "ksvd_medoid", "bipartite", False, True, False, False
        ),
        "ksvd_bipartite_shuffled_id": ControlSpec(
            "ksvd_bipartite_shuffled_id", "ksvd", "bipartite", False, True, True, False
        ),
        "ksvd_bipartite_no_id": ControlSpec(
            "ksvd_bipartite_no_id", "ksvd", "bipartite", False, False, False, False
        ),
    }
    names = [part.strip() for part in raw.split(",") if part.strip()]
    unknown = sorted(set(names).difference(known))
    if not names or unknown or len(set(names)) != len(names):
        raise ValueError(f"invalid controls={names}, unknown={unknown}")
    return [known[name] for name in names]


def _sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


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


def _select_farthest(candidates: np.ndarray, n_select: int) -> np.ndarray:
    """Deterministic cosine farthest-point selection."""
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
) -> tuple[np.ndarray, np.ndarray]:
    cosine = latents @ prototypes.T
    positive = np.maximum(cosine, 0.0)
    if sparsity < positive.shape[1]:
        indices = np.argpartition(positive, -sparsity, axis=1)[:, -sparsity:]
        codes = np.zeros_like(positive)
        rows = np.arange(len(positive))[:, None]
        codes[rows, indices] = positive[rows, indices]
    else:
        codes = positive.copy()
    # Extremely unusual all-negative rows would otherwise disappear from the
    # occurrence graph.  Preserve one nearest-prototype incidence with a tiny
    # positive mass, without altering normal rows.
    empty = np.flatnonzero(codes.max(axis=1) <= 0.0)
    if len(empty):
        nearest = cosine[empty].argmax(axis=1)
        codes[empty, nearest] = 1e-6
    return cosine.astype(np.float32), codes.astype(np.float32)


def _build_occurrence_graph(
    graph: Any,
    center_fields: np.ndarray,
    latents: np.ndarray,
    prototypes: np.ndarray,
    sparsity: int,
    graph_index: int,
    shuffle_seed: int,
) -> dict[str, np.ndarray]:
    """Split each prototype support into radius-1 connected components."""
    _, codes = _sparse_positive_codes(latents, prototypes, sparsity)
    n_nodes, n_prototypes = codes.shape

    proto_ids: list[int] = []
    scalar_rows: list[list[float]] = []
    member_nodes: list[int] = []
    member_occurrences: list[int] = []
    member_weights: list[float] = []
    node_to_occurrences: list[list[int]] = [[] for _ in range(n_nodes)]

    for proto in range(n_prototypes):
        active_nodes = np.flatnonzero(codes[:, proto] > 0.0)
        if len(active_nodes) == 0:
            continue
        active_set = set(int(x) for x in active_nodes.tolist())
        unseen = set(active_set)
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
            occurrence = len(proto_ids)
            coeff = codes[np.asarray(component, dtype=np.int64), proto]
            proto_ids.append(proto)
            scalar_rows.append([
                float(np.log1p(len(component))),
                float(coeff.mean()),
                float(coeff.max()),
                float(coeff.sum()),
            ])
            for node in component:
                member_nodes.append(node)
                member_occurrences.append(occurrence)
                member_weights.append(float(codes[node, proto]))
                node_to_occurrences[node].append(occurrence)

    n_occurrences = len(proto_ids)
    if n_occurrences == 0:
        raise RuntimeError(f"graph {graph_index} produced no prototype occurrences")

    # Relation counts for unordered occurrence pairs.  Overlap means two
    # different prototype occurrences contain the same center atom; adjacency
    # means at least one original molecular bond connects their member atoms.
    relation: dict[tuple[int, int], list[int]] = {}

    def add_relation(a: int, b: int, kind: int) -> None:
        if a == b:
            return
        key = (a, b) if a < b else (b, a)
        counts = relation.setdefault(key, [0, 0])
        counts[kind] += 1

    for occurrences in node_to_occurrences:
        unique_occurrences = sorted(set(occurrences))
        for a, b in combinations(unique_occurrences, 2):
            add_relation(a, b, 0)
    for u, v in graph.edges():
        for a in node_to_occurrences[u]:
            for b in node_to_occurrences[v]:
                add_relation(a, b, 1)

    edge_src: list[int] = []
    edge_dst: list[int] = []
    edge_features: list[list[float]] = []
    for (a, b), (overlap_count, bond_count) in relation.items():
        feature = [
            float(overlap_count > 0),
            float(np.log1p(overlap_count)),
            float(bond_count > 0),
            float(np.log1p(bond_count)),
        ]
        edge_src.extend([a, b])
        edge_dst.extend([b, a])
        edge_features.extend([feature, feature])

    permutation = np.random.default_rng(
        shuffle_seed + 1_000_003 * graph_index
    ).permutation(n_prototypes)
    proto_array = np.asarray(proto_ids, dtype=np.int64)
    edge_index = (
        np.asarray([edge_src, edge_dst], dtype=np.int64)
        if edge_src
        else np.empty((2, 0), dtype=np.int64)
    )
    edge_attr = (
        np.asarray(edge_features, dtype=np.float32)
        if edge_features
        else np.empty((0, 4), dtype=np.float32)
    )
    return {
        "prototype_id": proto_array,
        "shuffled_prototype_id": permutation[proto_array].astype(np.int64),
        "scalars": np.asarray(scalar_rows, dtype=np.float32),
        "member_nodes": np.asarray(member_nodes, dtype=np.int64),
        "member_occurrence": np.asarray(member_occurrences, dtype=np.int64),
        "member_weight": np.asarray(member_weights, dtype=np.float32),
        "member_center": center_fields[np.asarray(member_nodes, dtype=np.int64)].astype(np.int64),
        "atom_center": center_fields.astype(np.int64),
        "edge_index": edge_index,
        "edge_attr": edge_attr,
        "n_original_nodes": np.asarray([n_nodes], dtype=np.int64),
    }


def _summarize_occurrence_graphs(graphs: list[dict[str, np.ndarray]]) -> dict[str, float | int]:
    occurrences = np.asarray([len(g["prototype_id"]) for g in graphs], dtype=np.int64)
    directed_edges = np.asarray([g["edge_index"].shape[1] for g in graphs], dtype=np.int64)
    memberships = np.asarray([len(g["member_nodes"]) for g in graphs], dtype=np.int64)
    sizes = np.concatenate([
        np.bincount(g["member_occurrence"], minlength=len(g["prototype_id"]))
        for g in graphs
    ])
    return {
        "n_graphs": int(len(graphs)),
        "n_occurrences": int(occurrences.sum()),
        "mean_occurrences_per_graph": float(occurrences.mean()),
        "max_occurrences_per_graph": int(occurrences.max()),
        "mean_directed_edges_per_graph": float(directed_edges.mean()),
        "mean_memberships_per_graph": float(memberships.mean()),
        "singleton_fraction": float(np.mean(sizes == 1)),
        "mean_occurrence_size": float(sizes.mean()),
        "max_occurrence_size": int(sizes.max()),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-cache", required=True)
    ap.add_argument("--token-cache", required=True)
    ap.add_argument("--fold-cache", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument(
        "--controls",
        default=(
            "farthest_node_mil,farthest_bipartite,random_bipartite,"
            "random_direction_bipartite,pca_bipartite,ksvd_bipartite,"
            "ksvd_bipartite_shuffled_id,ksvd_bipartite_no_id"
        ),
    )
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--candidate-bank-size", type=int, default=256)
    ap.add_argument("--n-prototypes", type=int, default=32)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--shuffle-seed", type=int, default=314159)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--batch-size", type=int, default=96)
    ap.add_argument("--hidden", type=int, default=48)
    ap.add_argument("--bipartite-rounds", type=int, default=2)
    ap.add_argument("--dropout", type=float, default=0.15)
    ap.add_argument("--temperature", type=float, default=0.25)
    ap.add_argument("--gine-gate-scale", type=float, default=0.25)
    ap.add_argument(
        "--local-metric-description",
        default="fold-specific frozen label-free masked-context latent",
    )
    ap.add_argument(
        "--local-metric-gnn-layers", type=int, default=2,
        help="GNN layers used upstream to construct the frozen local metric.",
    )
    ap.add_argument("--task-lr", type=float, default=1e-3)
    ap.add_argument("--weight-decay", type=float, default=1e-4)
    ap.add_argument("--grad-clip", type=float, default=5.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--num-workers", type=int, default=0)
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    controls = _parse_controls(args.controls)
    if args.fold < 0 or args.epochs <= 0 or args.batch_size <= 0:
        raise ValueError("invalid fold/training configuration")
    if min(args.candidate_bank_size, args.n_prototypes, args.sparsity) <= 0:
        raise ValueError("prototype sizes must be positive")
    if args.n_prototypes > args.candidate_bank_size:
        raise ValueError("n-prototypes exceeds candidate-bank-size")
    if args.sparsity > args.n_prototypes:
        raise ValueError("sparsity exceeds n-prototypes")
    if args.temperature <= 0 or args.task_lr <= 0:
        raise ValueError("temperature/task-lr must be positive")
    if args.local_metric_gnn_layers < 0:
        raise ValueError("local-metric-gnn-layers must be nonnegative")
    if args.bipartite_rounds <= 0:
        raise ValueError("bipartite-rounds must be positive")

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from ogb.utils.features import get_atom_feature_dims
        from torch.utils.data import DataLoader, Dataset
        from torch_geometric.nn import GINEConv, global_add_pool, global_max_pool, global_mean_pool
        from torch_geometric.utils import softmax as graph_softmax
    except ImportError as exc:
        raise RuntimeError(f"missing training dependencies: {exc}") from exc

    t0 = time.time()
    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=True,
    )
    if bundle.node_feats is None:
        raise AssertionError("center atom features were not loaded")
    labels = np.asarray(bundle.y, dtype=np.float32)
    expected_original = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    official_train = np.asarray(bundle.split["train"], dtype=np.int64)
    official_valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    official_test = np.asarray(bundle.split["test"], dtype=np.int64)

    with np.load(args.fold_cache, allow_pickle=False) as folds:
        fit_indices = np.asarray(folds[f"fold_{args.fold}_train_indices"], dtype=np.int64)
        heldout_indices = np.asarray(folds[f"fold_{args.fold}_valid_indices"], dtype=np.int64)
        if "original_indices" in folds.files and not np.array_equal(
            np.asarray(folds["original_indices"], dtype=np.int64), expected_original
        ):
            raise ValueError("fold cache original_indices do not match dataset")
    official_train_set = set(official_train.tolist())
    if not set(fit_indices.tolist()).issubset(official_train_set):
        raise AssertionError("fit contains non-official-train graphs")
    if not set(heldout_indices.tolist()).issubset(official_train_set):
        raise AssertionError("heldout contains non-official-train graphs")
    if np.intersect1d(fit_indices, heldout_indices).size:
        raise AssertionError("fit and heldout overlap")
    if set(np.concatenate([fit_indices, heldout_indices]).tolist()) != official_train_set:
        raise AssertionError("scaffold fold does not partition official train")

    with np.load(args.latent_cache, allow_pickle=False) as source:
        offsets = np.asarray(source["offsets"], dtype=np.int64)
        latent_original = np.asarray(source["original_indices"], dtype=np.int64)
        latent_train = np.asarray(source["train_indices"], dtype=np.int64)
        latent_valid = np.asarray(source["valid_indices"], dtype=np.int64)
        latent_test = np.asarray(source["test_indices"], dtype=np.int64)
        latents_np = np.asarray(source["latents"], dtype=np.float32)
    with np.load(args.token_cache, allow_pickle=False) as source:
        token_arrays = {key: np.asarray(source[key]) for key in source.files}
    checks = [
        (latent_original, expected_original, "latent original_indices"),
        (latent_train, official_train, "latent train_indices"),
        (latent_valid, official_valid, "latent valid_indices"),
        (latent_test, official_test, "latent test_indices"),
        (np.asarray(token_arrays["offsets"], dtype=np.int64), offsets, "token offsets"),
        (np.asarray(token_arrays["original_indices"], dtype=np.int64), expected_original, "token original_indices"),
    ]
    for actual, expected, name in checks:
        if not np.array_equal(actual, expected):
            raise ValueError(f"misaligned {name}")
    if latents_np.shape[0] != int(offsets[-1]):
        raise ValueError("latent row count and offsets disagree")
    official_nontrain_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in np.concatenate([official_valid, official_test])
    ])
    if np.any(latents_np[official_nontrain_rows] != 0):
        raise AssertionError("official-valid/test latent rows are nonzero")

    fit_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in fit_indices
    ])
    heldout_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
        for i in heldout_indices
    ])
    if np.mean(np.linalg.norm(latents_np[np.concatenate([fit_rows, heldout_rows])], axis=1) > 0.99) < 0.999:
        raise ValueError("fit/heldout scaffold latents are missing or unnormalized")

    rng = np.random.default_rng(args.seed + 910_003 * (args.fold + 1))
    candidate_rows = rng.choice(
        fit_rows, size=args.candidate_bank_size, replace=False
    ).astype(np.int64)
    candidate_bank = _normalize_rows(latents_np[candidate_rows])
    random_selection = rng.choice(
        args.candidate_bank_size, size=args.n_prototypes, replace=False
    ).astype(np.int64)
    random_prototypes = candidate_bank[random_selection]
    ksvd_prototypes = _normalize_rows(
        np.asarray(token_arrays["dictionary_ksvd"], dtype=np.float32).T
    )
    if ksvd_prototypes.shape != random_prototypes.shape:
        raise ValueError(
            f"KSVD/random prototype mismatch: {ksvd_prototypes.shape} vs {random_prototypes.shape}"
        )
    # Project each KSVD direction onto a unique real fit patch.  This keeps
    # KSVD's coverage-oriented atom ordering while restoring an actually
    # observed local prototype identity.  Selection is label-free and
    # seed-independent.  Greedy conflict resolution only matters when multiple
    # KSVD atoms have the same nearest patch (one duplicate in folds 0/1 here).
    fit_latents = _normalize_rows(latents_np[fit_rows])
    ksvd_to_fit_cosine = fit_latents @ ksvd_prototypes.T
    best_similarity = ksvd_to_fit_cosine.max(axis=0)
    atom_order = np.argsort(-best_similarity)
    medoid_local_rows = np.full(args.n_prototypes, -1, dtype=np.int64)
    used_local_rows: set[int] = set()
    for atom in atom_order.tolist():
        scores = ksvd_to_fit_cosine[:, atom]
        if used_local_rows:
            scores = scores.copy()
            scores[np.fromiter(used_local_rows, dtype=np.int64)] = -np.inf
        chosen = int(scores.argmax())
        medoid_local_rows[atom] = chosen
        used_local_rows.add(chosen)
    if len(used_local_rows) != args.n_prototypes:
        raise AssertionError("KSVD medoid projection did not produce unique prototypes")
    ksvd_medoid_rows = fit_rows[medoid_local_rows]
    ksvd_medoid_prototypes = fit_latents[medoid_local_rows]
    ksvd_medoid_cosine = ksvd_to_fit_cosine[
        medoid_local_rows, np.arange(args.n_prototypes)
    ]

    # One deterministic observed candidate per fit graph gives the farthest
    # selector broad graph coverage without letting large molecules dominate
    # the candidate reservoir.
    farthest_candidate_rows: list[int] = []
    for graph_i_raw in fit_indices:
        graph_i = int(graph_i_raw)
        lo, hi = int(offsets[graph_i]), int(offsets[graph_i + 1])
        local_rng = np.random.default_rng(17 + 1_000_003 * graph_i)
        farthest_candidate_rows.append(lo + int(local_rng.integers(hi - lo)))
    farthest_candidate_rows_np = np.asarray(farthest_candidate_rows, dtype=np.int64)
    farthest_candidate_bank = _normalize_rows(latents_np[farthest_candidate_rows_np])
    farthest_selection = _select_farthest(
        farthest_candidate_bank, args.n_prototypes
    )
    farthest_source_rows = farthest_candidate_rows_np[farthest_selection]
    farthest_prototypes = farthest_candidate_bank[farthest_selection]

    # PCA and random directions are geometry controls with no observed-patch
    # identity. Compute PCA from the fit-fold covariance only.
    covariance = (fit_latents.T @ fit_latents) / max(len(fit_latents), 1)
    eigenvalues, eigenvectors = np.linalg.eigh(covariance.astype(np.float64))
    pca_prototypes = eigenvectors[:, np.argsort(eigenvalues)[::-1][
        :args.n_prototypes
    ]].T.astype(np.float32)
    largest = np.abs(pca_prototypes).argmax(axis=1)
    signs = np.sign(pca_prototypes[np.arange(args.n_prototypes), largest])
    pca_prototypes *= np.where(signs == 0, 1.0, signs)[:, None]
    pca_prototypes = _normalize_rows(pca_prototypes)
    random_direction_prototypes = _normalize_rows(
        rng.normal(size=(args.n_prototypes, fit_latents.shape[1])).astype(np.float32)
    )
    prototype_sets = {
        "random": random_prototypes,
        "random_direction": random_direction_prototypes,
        "farthest": farthest_prototypes,
        "pca": pca_prototypes,
        "ksvd": ksvd_prototypes,
        "ksvd_medoid": ksvd_medoid_prototypes,
    }

    preprocessing_start = time.time()
    required_families = sorted({
        c.family for c in controls if c.level in {"occurrence", "bipartite"}
    })
    occurrence_graphs: dict[str, dict[int, dict[str, np.ndarray]]] = {}
    occurrence_stats: dict[str, dict[str, float | int]] = {}
    train_all = np.concatenate([fit_indices, heldout_indices])
    for family in required_families:
        family_graphs: dict[int, dict[str, np.ndarray]] = {}
        for graph_i_raw in train_all:
            graph_i = int(graph_i_raw)
            lo, hi = int(offsets[graph_i]), int(offsets[graph_i + 1])
            family_graphs[graph_i] = _build_occurrence_graph(
                bundle.graphs[graph_i],
                np.asarray(bundle.node_feats[graph_i], dtype=np.int64),
                latents_np[lo:hi],
                prototype_sets[family],
                args.sparsity,
                graph_i,
                args.shuffle_seed,
            )
        occurrence_graphs[family] = family_graphs
        occurrence_stats[family] = _summarize_occurrence_graphs(
            [family_graphs[int(i)] for i in train_all]
        )
        print(f"{family} occurrence audit: {occurrence_stats[family]}", flush=True)
    preprocessing_sec = time.time() - preprocessing_start

    atom_dims = [int(x) for x in get_atom_feature_dims()]
    latents = torch.from_numpy(latents_np)

    class GraphIndexDataset(Dataset):
        def __init__(self, indices: np.ndarray):
            self.indices = np.asarray(indices, dtype=np.int64)

        def __len__(self) -> int:
            return int(len(self.indices))

        def __getitem__(self, item: int) -> int:
            return int(self.indices[item])

    def collate_nodes(graph_indices: list[int]) -> dict[str, Any]:
        rows: list[np.ndarray] = []
        center_fields: list[np.ndarray] = []
        graph_ids: list[np.ndarray] = []
        for batch_i, raw_i in enumerate(graph_indices):
            graph_i = int(raw_i)
            lo, hi = int(offsets[graph_i]), int(offsets[graph_i + 1])
            n = hi - lo
            rows.append(np.arange(lo, hi, dtype=np.int64))
            center_fields.append(np.asarray(bundle.node_feats[graph_i], dtype=np.int64))
            graph_ids.append(np.full(n, batch_i, dtype=np.int64))
        return {
            "rows": torch.from_numpy(np.concatenate(rows)),
            "center": torch.from_numpy(np.concatenate(center_fields)),
            "graph_batch": torch.from_numpy(np.concatenate(graph_ids)),
            "labels": torch.from_numpy(labels[np.asarray(graph_indices, dtype=np.int64)]),
        }

    def make_occurrence_collate(family: str, shuffled_identity: bool):
        def collate(graph_indices: list[int]) -> dict[str, Any]:
            prototype_ids: list[np.ndarray] = []
            scalars: list[np.ndarray] = []
            member_center: list[np.ndarray] = []
            atom_center: list[np.ndarray] = []
            member_atom: list[np.ndarray] = []
            member_occurrence: list[np.ndarray] = []
            member_weight: list[np.ndarray] = []
            edge_indices: list[np.ndarray] = []
            edge_attrs: list[np.ndarray] = []
            graph_batch: list[np.ndarray] = []
            atom_graph_batch: list[np.ndarray] = []
            occurrence_offset = 0
            atom_offset = 0
            for batch_i, raw_i in enumerate(graph_indices):
                graph_i = int(raw_i)
                row = occurrence_graphs[family][graph_i]
                ids_key = "shuffled_prototype_id" if shuffled_identity else "prototype_id"
                n_occ = len(row["prototype_id"])
                n_atom = int(row["n_original_nodes"][0])
                prototype_ids.append(row[ids_key])
                scalars.append(row["scalars"])
                member_center.append(row["member_center"])
                atom_center.append(row["atom_center"])
                member_atom.append(row["member_nodes"] + atom_offset)
                member_occurrence.append(row["member_occurrence"] + occurrence_offset)
                member_weight.append(row["member_weight"])
                if row["edge_index"].shape[1]:
                    edge_indices.append(row["edge_index"] + occurrence_offset)
                    edge_attrs.append(row["edge_attr"])
                graph_batch.append(np.full(n_occ, batch_i, dtype=np.int64))
                atom_graph_batch.append(np.full(n_atom, batch_i, dtype=np.int64))
                occurrence_offset += n_occ
                atom_offset += n_atom
            edge_index = (
                np.concatenate(edge_indices, axis=1)
                if edge_indices
                else np.empty((2, 0), dtype=np.int64)
            )
            edge_attr = (
                np.concatenate(edge_attrs, axis=0)
                if edge_attrs
                else np.empty((0, 4), dtype=np.float32)
            )
            return {
                "prototype_id": torch.from_numpy(np.concatenate(prototype_ids)),
                "scalars": torch.from_numpy(np.concatenate(scalars)),
                "member_center": torch.from_numpy(np.concatenate(member_center)),
                "atom_center": torch.from_numpy(np.concatenate(atom_center)),
                "member_atom": torch.from_numpy(np.concatenate(member_atom)),
                "member_occurrence": torch.from_numpy(np.concatenate(member_occurrence)),
                "member_weight": torch.from_numpy(np.concatenate(member_weight)),
                "edge_index": torch.from_numpy(edge_index),
                "edge_attr": torch.from_numpy(edge_attr),
                "graph_batch": torch.from_numpy(np.concatenate(graph_batch)),
                "atom_graph_batch": torch.from_numpy(np.concatenate(atom_graph_batch)),
                "labels": torch.from_numpy(labels[np.asarray(graph_indices, dtype=np.int64)]),
            }
        return collate

    class Readout(nn.Module):
        def __init__(self):
            super().__init__()
            self.attention = nn.Sequential(
                nn.Linear(args.hidden, args.hidden // 2),
                nn.Tanh(),
                nn.Linear(args.hidden // 2, 1),
            )
            self.head = nn.Sequential(
                nn.Linear(3 * args.hidden, args.hidden),
                nn.LayerNorm(args.hidden),
                nn.SiLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden, 1),
            )

        def forward(self, h, graph_batch):
            attention = graph_softmax(self.attention(h).view(-1), graph_batch)
            attention_pool = global_add_pool(h * attention[:, None], graph_batch)
            mean_pool = global_mean_pool(h, graph_batch)
            max_pool = global_max_pool(h, graph_batch)
            return self.head(torch.cat([
                attention_pool, mean_pool, max_pool
            ], dim=1)).view(-1)

    class NodePrototypeMIL(nn.Module):
        def __init__(self, prototypes: np.ndarray):
            super().__init__()
            p = torch.tensor(prototypes, dtype=torch.float32)
            p = p / p.norm(dim=1, keepdim=True).clamp_min(1e-12)
            self.register_buffer("prototypes", p)
            self.center_embeddings = nn.ModuleList([
                nn.Embedding(dim, args.hidden) for dim in atom_dims
            ])
            self.identity_embeddings = nn.Parameter(
                torch.empty(args.n_prototypes, args.hidden)
            )
            self.affinity_embeddings = nn.Parameter(
                torch.empty(args.n_prototypes, args.hidden)
            )
            self.node_mlp = nn.Sequential(
                nn.Linear(3 * args.hidden + 2, args.hidden),
                nn.LayerNorm(args.hidden),
                nn.SiLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden, args.hidden),
                nn.SiLU(),
            )
            self.readout = Readout()
            self.reset_parameters()

        def reset_parameters(self):
            for emb in self.center_embeddings:
                nn.init.xavier_uniform_(emb.weight)
            nn.init.xavier_uniform_(self.identity_embeddings)
            nn.init.xavier_uniform_(self.affinity_embeddings)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(self, z, center, graph_batch):
            cosine = z @ self.prototypes.T
            positive = torch.relu(cosine)
            if args.sparsity < args.n_prototypes:
                values, indices = positive.topk(args.sparsity, dim=1)
                codes = torch.zeros_like(positive).scatter_(1, indices, values)
            else:
                codes = positive
            mass = codes.sum(dim=1, keepdim=True)
            normalized = codes / mass.clamp_min(1e-6)
            sharpened = torch.softmax(codes / args.temperature, dim=1)
            sharpened = sharpened * (codes > 0).float()
            sharpened = sharpened / sharpened.sum(dim=1, keepdim=True).clamp_min(1e-6)
            center_h = 0.0
            for field, emb in enumerate(self.center_embeddings):
                center_h = center_h + emb(center[:, field])
            identity_h = normalized @ self.identity_embeddings
            affinity_h = sharpened @ self.affinity_embeddings
            top_similarity = cosine.max(dim=1, keepdim=True).values
            h = self.node_mlp(torch.cat([
                center_h, identity_h, affinity_h,
                torch.log1p(mass), top_similarity,
            ], dim=1))
            logits = self.readout(h, graph_batch)
            return logits, h

    class OccurrenceModel(nn.Module):
        def __init__(self, use_gine: bool, use_identity: bool, gated_gine: bool):
            super().__init__()
            self.use_gine = use_gine
            self.use_identity = use_identity
            self.gated_gine = gated_gine
            self.center_embeddings = nn.ModuleList([
                nn.Embedding(dim, args.hidden) for dim in atom_dims
            ])
            self.prototype_embedding = nn.Embedding(args.n_prototypes, args.hidden)
            self.scalar_encoder = nn.Sequential(
                nn.Linear(4, args.hidden),
                nn.LayerNorm(args.hidden),
                nn.SiLU(),
                nn.Linear(args.hidden, args.hidden),
            )
            self.occurrence_encoder = nn.Sequential(
                nn.Linear(3 * args.hidden, args.hidden),
                nn.LayerNorm(args.hidden),
                nn.SiLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden, args.hidden),
                nn.SiLU(),
            )
            if use_gine:
                self.edge_encoder = nn.Sequential(
                    nn.Linear(4, args.hidden),
                    nn.LayerNorm(args.hidden),
                    nn.SiLU(),
                    nn.Linear(args.hidden, args.hidden),
                )
                mlp = nn.Sequential(
                    nn.Linear(args.hidden, 2 * args.hidden),
                    nn.LayerNorm(2 * args.hidden),
                    nn.SiLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(2 * args.hidden, args.hidden),
                )
                self.conv = GINEConv(mlp, train_eps=True)
                self.post_norm = nn.LayerNorm(args.hidden)
                self.gine_gate = nn.Parameter(
                    torch.zeros(()), requires_grad=gated_gine
                )
            self.readout = Readout()
            self.reset_parameters()

        def reset_parameters(self):
            for emb in self.center_embeddings:
                nn.init.xavier_uniform_(emb.weight)
            nn.init.xavier_uniform_(self.prototype_embedding.weight)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        def forward(
            self,
            prototype_id,
            scalars,
            member_center,
            member_occurrence,
            edge_index,
            edge_attr,
            graph_batch,
        ):
            n_occurrences = int(prototype_id.shape[0])
            member_h = 0.0
            for field, emb in enumerate(self.center_embeddings):
                member_h = member_h + emb(member_center[:, field])
            center_sum = member_h.new_zeros((n_occurrences, args.hidden))
            center_sum.index_add_(0, member_occurrence, member_h)
            member_count = member_h.new_zeros(n_occurrences)
            member_count.index_add_(
                0, member_occurrence, member_h.new_ones(member_occurrence.shape[0])
            )
            center_h = center_sum / member_count.clamp_min(1.0).unsqueeze(-1)
            if self.use_identity:
                identity_h = self.prototype_embedding(prototype_id)
            else:
                identity_h = torch.zeros_like(center_h)
            scalar_h = self.scalar_encoder(scalars)
            h = self.occurrence_encoder(torch.cat([
                center_h, identity_h, scalar_h
            ], dim=1))
            if self.use_gine and edge_index.shape[1] > 0:
                edge_h = self.edge_encoder(edge_attr)
                update = self.conv(h, edge_index, edge_h)
                if self.gated_gine:
                    update = self.post_norm(F.silu(update))
                    h = h + (
                        args.gine_gate_scale
                        * torch.tanh(self.gine_gate)
                        * F.dropout(update, p=args.dropout, training=self.training)
                    )
                else:
                    h = self.post_norm(h + F.dropout(
                        F.silu(update), p=args.dropout, training=self.training
                    ))
            logits = self.readout(h, graph_batch)
            return logits, h

    class AtomOccurrenceBipartite(nn.Module):
        """Alternating atom->occurrence->atom transport without a graph GNN."""

        def __init__(self, use_identity: bool):
            super().__init__()
            self.use_identity = use_identity
            self.center_embeddings = nn.ModuleList([
                nn.Embedding(dim, args.hidden) for dim in atom_dims
            ])
            self.atom_init = nn.Sequential(
                nn.LayerNorm(args.hidden),
                nn.Linear(args.hidden, args.hidden),
                nn.SiLU(),
            )
            self.prototype_embedding = nn.Embedding(args.n_prototypes, args.hidden)
            self.scalar_encoder = nn.Sequential(
                nn.Linear(4, args.hidden),
                nn.LayerNorm(args.hidden),
                nn.SiLU(),
                nn.Linear(args.hidden, args.hidden),
            )
            self.occurrence_init = nn.Sequential(
                nn.Linear(2 * args.hidden, args.hidden),
                nn.LayerNorm(args.hidden),
                nn.SiLU(),
            )
            self.occurrence_updates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * args.hidden, args.hidden),
                    nn.SiLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(args.hidden, args.hidden),
                )
                for _ in range(args.bipartite_rounds)
            ])
            self.atom_updates = nn.ModuleList([
                nn.Sequential(
                    nn.Linear(2 * args.hidden, args.hidden),
                    nn.SiLU(),
                    nn.Dropout(args.dropout),
                    nn.Linear(args.hidden, args.hidden),
                )
                for _ in range(args.bipartite_rounds)
            ])
            self.occurrence_norms = nn.ModuleList([
                nn.LayerNorm(args.hidden) for _ in range(args.bipartite_rounds)
            ])
            self.atom_norms = nn.ModuleList([
                nn.LayerNorm(args.hidden) for _ in range(args.bipartite_rounds)
            ])
            self.atom_attention = nn.Sequential(
                nn.Linear(args.hidden, args.hidden // 2),
                nn.Tanh(),
                nn.Linear(args.hidden // 2, 1),
            )
            self.occurrence_attention = nn.Sequential(
                nn.Linear(args.hidden, args.hidden // 2),
                nn.Tanh(),
                nn.Linear(args.hidden // 2, 1),
            )
            self.head = nn.Sequential(
                nn.Linear(6 * args.hidden, args.hidden),
                nn.LayerNorm(args.hidden),
                nn.SiLU(),
                nn.Dropout(args.dropout),
                nn.Linear(args.hidden, 1),
            )
            self.reset_parameters()

        def reset_parameters(self):
            for emb in self.center_embeddings:
                nn.init.xavier_uniform_(emb.weight)
            nn.init.xavier_uniform_(self.prototype_embedding.weight)
            for module in self.modules():
                if isinstance(module, nn.Linear):
                    nn.init.xavier_uniform_(module.weight)
                    if module.bias is not None:
                        nn.init.zeros_(module.bias)

        @staticmethod
        def _pool(h, graph_batch, attention_module):
            attention = graph_softmax(
                attention_module(h).view(-1), graph_batch
            )
            return torch.cat([
                global_add_pool(h * attention[:, None], graph_batch),
                global_mean_pool(h, graph_batch),
                global_max_pool(h, graph_batch),
            ], dim=1)

        def forward(
            self,
            prototype_id,
            scalars,
            atom_center,
            member_atom,
            member_occurrence,
            member_weight,
            occurrence_graph_batch,
            atom_graph_batch,
        ):
            atom_h = 0.0
            for field, emb in enumerate(self.center_embeddings):
                atom_h = atom_h + emb(atom_center[:, field])
            atom_h = self.atom_init(atom_h)

            if self.use_identity:
                identity_h = self.prototype_embedding(prototype_id)
            else:
                identity_h = atom_h.new_zeros(
                    (prototype_id.shape[0], args.hidden)
                )
            scalar_h = self.scalar_encoder(scalars)
            occurrence_h = self.occurrence_init(torch.cat([
                identity_h, scalar_h
            ], dim=1))

            weight = member_weight.clamp_min(1e-6)
            n_occurrences = int(prototype_id.shape[0])
            n_atoms = int(atom_center.shape[0])
            for occurrence_update, atom_update, occurrence_norm, atom_norm in zip(
                self.occurrence_updates,
                self.atom_updates,
                self.occurrence_norms,
                self.atom_norms,
            ):
                occurrence_sum = atom_h.new_zeros(
                    (n_occurrences, args.hidden)
                )
                occurrence_sum.index_add_(
                    0,
                    member_occurrence,
                    weight[:, None] * atom_h[member_atom],
                )
                occurrence_mass = atom_h.new_zeros(n_occurrences)
                occurrence_mass.index_add_(0, member_occurrence, weight)
                atom_to_occurrence = occurrence_sum / occurrence_mass.clamp_min(
                    1e-6
                )[:, None]
                occurrence_h = occurrence_norm(
                    occurrence_h
                    + occurrence_update(torch.cat([
                        occurrence_h, atom_to_occurrence
                    ], dim=1))
                )

                atom_sum = atom_h.new_zeros((n_atoms, args.hidden))
                atom_sum.index_add_(
                    0,
                    member_atom,
                    weight[:, None] * occurrence_h[member_occurrence],
                )
                atom_mass = atom_h.new_zeros(n_atoms)
                atom_mass.index_add_(0, member_atom, weight)
                occurrence_to_atom = atom_sum / atom_mass.clamp_min(1e-6)[:, None]
                atom_h = atom_norm(
                    atom_h
                    + atom_update(torch.cat([
                        atom_h, occurrence_to_atom
                    ], dim=1))
                )

            atom_pool = self._pool(
                atom_h, atom_graph_batch, self.atom_attention
            )
            occurrence_pool = self._pool(
                occurrence_h, occurrence_graph_batch, self.occurrence_attention
            )
            logits = self.head(torch.cat([
                atom_pool, occurrence_pool
            ], dim=1)).view(-1)
            return logits, torch.cat([atom_h, occurrence_h], dim=0)

    device = torch.device(args.device)

    def make_loader(
        indices: np.ndarray,
        shuffle: bool,
        seed_offset: int,
        collate_fn: Any,
    ):
        generator = torch.Generator().manual_seed(args.seed + seed_offset)
        return DataLoader(
            GraphIndexDataset(indices),
            batch_size=args.batch_size,
            shuffle=shuffle,
            generator=generator if shuffle else None,
            num_workers=args.num_workers,
            collate_fn=collate_fn,
        )

    def move_batch(batch: dict[str, Any]) -> dict[str, Any]:
        return {key: value.to(device) for key, value in batch.items()}

    def forward_control(model, control: ControlSpec, batch: dict[str, Any]):
        if control.level == "node":
            z = latents[batch["rows"].cpu()].to(device)
            return model(z, batch["center"], batch["graph_batch"])
        if control.level == "bipartite":
            return model(
                batch["prototype_id"],
                batch["scalars"],
                batch["atom_center"],
                batch["member_atom"],
                batch["member_occurrence"],
                batch["member_weight"],
                batch["graph_batch"],
                batch["atom_graph_batch"],
            )
        return model(
            batch["prototype_id"],
            batch["scalars"],
            batch["member_center"],
            batch["member_occurrence"],
            batch["edge_index"],
            batch["edge_attr"],
            batch["graph_batch"],
        )

    @torch.no_grad()
    def evaluate(model, control: ControlSpec, loader):
        model.eval()
        all_scores: list[np.ndarray] = []
        all_labels: list[np.ndarray] = []
        representation_norm = 0.0
        representation_count = 0
        for batch in loader:
            batch = move_batch(batch)
            logits, h = forward_control(model, control, batch)
            all_scores.append(logits.cpu().numpy())
            all_labels.append(batch["labels"].cpu().numpy())
            representation_norm += float(h.norm(dim=1).sum())
            representation_count += int(h.shape[0])
        scores = np.concatenate(all_scores)
        y = np.concatenate(all_labels)
        return {
            "auc": float(roc_auc_score(y, scores)),
            "n_graphs": int(len(y)),
            "n_positive": int(y.sum()),
            "mean_local_representation_norm": representation_norm / max(representation_count, 1),
            "score_sha256": _sha256(scores.astype(np.float64)),
        }

    results: dict[str, Any] = {}
    for control in controls:
        control_start = time.time()
        _seed_everything(args.seed, torch)
        if control.level == "node":
            collate_fn = collate_nodes
            model = NodePrototypeMIL(prototype_sets[control.family]).to(device)
        else:
            collate_fn = make_occurrence_collate(
                control.family, control.shuffled_identity
            )
            if control.level == "bipartite":
                model = AtomOccurrenceBipartite(control.use_identity).to(device)
            else:
                model = OccurrenceModel(
                    control.use_gine, control.use_identity, control.gated_gine
                ).to(device)
        optimizer = torch.optim.AdamW(
            model.parameters(), lr=args.task_lr, weight_decay=args.weight_decay
        )
        train_loader = make_loader(fit_indices, True, 1000, collate_fn)
        history: list[dict[str, float]] = []
        for epoch in range(1, args.epochs + 1):
            model.train()
            total_loss = 0.0
            seen = 0
            for batch in train_loader:
                batch = move_batch(batch)
                logits, _ = forward_control(model, control, batch)
                loss = F.binary_cross_entropy_with_logits(
                    logits, batch["labels"].float()
                )
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                if args.grad_clip > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), args.grad_clip)
                optimizer.step()
                n = int(batch["labels"].shape[0])
                total_loss += float(loss.detach()) * n
                seen += n
            row = {"epoch": float(epoch), "task": total_loss / max(seen, 1)}
            history.append(row)
            if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
                print(f"{control.name} epoch={epoch:02d} task={row['task']:.5f}", flush=True)

        fit_metrics = evaluate(
            model, control, make_loader(fit_indices, False, 2000, collate_fn)
        )
        heldout_metrics = evaluate(
            model, control, make_loader(heldout_indices, False, 3000, collate_fn)
        )
        results[control.name] = {
            "family": control.family,
            "level": control.level,
            "use_gine": control.use_gine,
            "supervised_original_atom_graph_message_passing_layers": 0,
            "supervised_occurrence_graph_message_passing_layers": int(control.use_gine),
            "supervised_atom_occurrence_rounds": (
                int(args.bipartite_rounds) if control.level == "bipartite" else 0
            ),
            "use_identity": control.use_identity,
            "shuffled_identity": control.shuffled_identity,
            "gated_gine": control.gated_gine,
            "final_gine_gate": (
                float(torch.tanh(model.gine_gate).detach().cpu())
                if control.level == "occurrence" and control.gated_gine else None
            ),
            "trainable_parameters": int(sum(
                p.numel() for p in model.parameters() if p.requires_grad
            )),
            "fit": fit_metrics,
            "heldout": heldout_metrics,
            "history": history,
            "elapsed_sec": time.time() - control_start,
        }
        print(
            f"{control.name} fit_auc={fit_metrics['auc']:.4f} "
            f"heldout_auc={heldout_metrics['auc']:.4f}",
            flush=True,
        )

    def delta(a: str, b: str) -> float | None:
        if a not in results or b not in results:
            return None
        return float(results[a]["heldout"]["auc"] - results[b]["heldout"]["auc"])

    comparisons = {
        "random_occ_mil_minus_random_node_mil_auc": delta(
            "random_occ_mil", "random_node_mil"
        ),
        "random_occ_gine_minus_random_occ_mil_auc": delta(
            "random_occ_gine", "random_occ_mil"
        ),
        "ksvd_occ_gine_minus_ksvd_occ_mil_auc": delta(
            "ksvd_occ_gine", "ksvd_occ_mil"
        ),
        "ksvd_occ_gine_minus_random_occ_gine_auc": delta(
            "ksvd_occ_gine", "random_occ_gine"
        ),
        "ksvd_occ_gine_minus_shuffled_id_auc": delta(
            "ksvd_occ_gine", "ksvd_occ_gine_shuffled_id"
        ),
        "ksvd_occ_gine_minus_no_id_auc": delta(
            "ksvd_occ_gine", "ksvd_occ_gine_no_id"
        ),
        "farthest_bipartite_minus_farthest_node_mil_auc": delta(
            "farthest_bipartite", "farthest_node_mil"
        ),
        "ksvd_bipartite_minus_ksvd_node_mil_auc": delta(
            "ksvd_bipartite", "ksvd_node_mil"
        ),
        "ksvd_bipartite_minus_farthest_bipartite_auc": delta(
            "ksvd_bipartite", "farthest_bipartite"
        ),
        "ksvd_bipartite_minus_pca_bipartite_auc": delta(
            "ksvd_bipartite", "pca_bipartite"
        ),
        "ksvd_bipartite_minus_random_direction_bipartite_auc": delta(
            "ksvd_bipartite", "random_direction_bipartite"
        ),
        "ksvd_bipartite_minus_shuffled_id_auc": delta(
            "ksvd_bipartite", "ksvd_bipartite_shuffled_id"
        ),
        "ksvd_bipartite_minus_no_id_auc": delta(
            "ksvd_bipartite", "ksvd_bipartite_no_id"
        ),
    }

    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    report = {
        "protocol_id": "molhiv-atom-occurrence-bipartite-scaffold-v1",
        "date": "2026-07-28",
        "hypothesis": (
            "a parameter-controlled atom-occurrence bipartite network can retain "
            "prototype location without either discarding spatial incidence or "
            "adding a full original-node GINE backbone"
        ),
        "architecture": {
            "local_metric": args.local_metric_description,
            "local_metric_gnn_layers": int(args.local_metric_gnn_layers),
            "occurrence": "radius-1 connected component of positive top-k prototype support",
            "transport": "coefficient-weighted atom-to-occurrence-to-atom membership messages",
            "supervised_original_atom_graph_message_passing_layers": 0,
            "supervised_occurrence_graph_message_passing_layers": 0,
            "supervised_atom_occurrence_rounds": int(args.bipartite_rounds),
            "graph_readout": "separate atom/occurrence attention plus mean plus max",
            "graph_labels_assigned_to_individual_nodes_or_occurrences": False,
        },
        "selection_policy": {
            "data": "official-train only",
            "outer_validation": "one held-out Bemis-Murcko scaffold fold",
            "epoch_policy": f"single evaluation after fixed epoch {args.epochs}",
            "official_valid_evaluations": 0,
            "official_test_evaluations": 0,
            "seed0_promotion_rule_across_3_folds": {
                "bipartite_mean_gain_over_node_mil_at_least": 0.005,
                "minimum_bipartite_fold_wins": 2,
                "ksvd_identity_must_beat_shuffled_mean": True,
                "ksvd_must_beat_pca_random_and_farthest_mean": True,
            },
        },
        "config": vars(args),
        "fold": int(args.fold),
        "n_fit": int(len(fit_indices)),
        "n_fit_positive": int(labels[fit_indices].sum()),
        "n_heldout": int(len(heldout_indices)),
        "n_heldout_positive": int(labels[heldout_indices].sum()),
        "fit_indices_sha256": _sha256(fit_indices),
        "heldout_indices_sha256": _sha256(heldout_indices),
        "candidate_rows_sha256": _sha256(candidate_rows),
        "random_selection": random_selection.tolist(),
        "random_source_node_rows": candidate_rows[random_selection].tolist(),
        "farthest_source_node_rows": farthest_source_rows.tolist(),
        "farthest_source_node_rows_sha256": _sha256(farthest_source_rows),
        "prototype_sha256": {
            family: _sha256(prototypes) for family, prototypes in prototype_sets.items()
        },
        "ksvd_medoid_projection": {
            "selection_data": "fit-fold node rows only; labels unused",
            "unique_real_patch_constraint": True,
            "source_node_rows": ksvd_medoid_rows.tolist(),
            "source_node_rows_sha256": _sha256(ksvd_medoid_rows),
            "atomwise_cosine_to_ksvd_direction": ksvd_medoid_cosine.tolist(),
            "mean_cosine_to_ksvd_direction": float(ksvd_medoid_cosine.mean()),
            "min_cosine_to_ksvd_direction": float(ksvd_medoid_cosine.min()),
            "max_cosine_to_ksvd_direction": float(ksvd_medoid_cosine.max()),
        },
        "occurrence_stats": occurrence_stats,
        "preprocessing_sec": preprocessing_sec,
        "results": results,
        "comparisons": comparisons,
        "elapsed_sec": time.time() - t0,
        "output": str(output),
    }
    output.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": str(output),
        "fold": args.fold,
        "heldout_auc": {name: row["heldout"]["auc"] for name, row in results.items()},
        "comparisons": comparisons,
        "elapsed_sec": report["elapsed_sec"],
    }, indent=2), flush=True)


if __name__ == "__main__":
    main()

"""Nested task-aware relation gates on a fixed broad real-patch vocabulary.

The broad deterministic farthest/scaffold-facility prototypes are never
reselected.  Outer-fit labels are used only to derive smooth prototype gates
through source-clean inner-scaffold OOF ablation ranking.  The gated
assignments feed compact exact-distance 1+2 relation features, while the
frozen broad occurrence ensemble supplies graph-level base logits.

Matched controls are uniform gates, shuffled-label gates, and node-assignment
shuffling.  Official valid/test latents must remain zero and are never scored.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import load_molhiv
from code.molhiv_evaluation_protocol import (
    evaluation_description, forbidden_encoded_indices, resolve_evaluation_protocol,
    validate_latent_sidecar, validate_upstream_result,
)
from code.run_molhiv_stable_taskmatched_prototypes import (
    _graph_candidate_features,
    _nested_candidate_importance,
)

FAMILIES = ("farthest", "scaffold_facility")
GATE_TYPES = ("uniform", "task", "shuffled_label")
RELATION_MODES = ("real", "assignment_shuffled")
DISTANCE_BINS = ("distance_1", "distance_2")


def sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def normalize_rows(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float32)
    return x / np.maximum(np.linalg.norm(x, axis=1, keepdims=True), 1e-12)


def seed_everything(seed: int, torch: Any) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    try:
        torch.use_deterministic_algorithms(True)
    except Exception:
        pass


def all_pairs_distances(graph) -> np.ndarray:
    n = graph.n
    distances = np.full((n, n), -1, dtype=np.int16)
    for source in range(n):
        distances[source, source] = 0
        queue: deque[int] = deque([source])
        while queue:
            u = queue.popleft()
            for v in graph.neighbors(u):
                if distances[source, v] < 0:
                    distances[source, v] = distances[source, u] + 1
                    queue.append(v)
    return distances


def sparse_assignment(z: np.ndarray, prototypes: np.ndarray, sparsity: int) -> tuple[np.ndarray, np.ndarray]:
    cosine = np.asarray(z @ prototypes.T, dtype=np.float32)
    positive = np.maximum(cosine, 0.0)
    if sparsity < positive.shape[1]:
        indices = np.argpartition(positive, -sparsity, axis=1)[:, -sparsity:]
        codes = np.zeros_like(positive)
        rows = np.arange(len(positive))[:, None]
        codes[rows, indices] = positive[rows, indices]
    else:
        codes = positive
    mass = codes.sum(axis=1, keepdims=True)
    assignment = np.divide(codes, mass, out=np.zeros_like(codes), where=mass > 1e-8)
    return assignment.astype(np.float32), cosine


def apply_gate(assignment: np.ndarray, gate: np.ndarray) -> np.ndarray:
    gated = np.asarray(assignment * gate[None, :], dtype=np.float32)
    mass = gated.sum(axis=1, keepdims=True)
    return np.divide(gated, mass, out=np.zeros_like(gated), where=mass > 1e-8).astype(np.float32)


def compact_distance_feature(
    assignment: np.ndarray,
    distances: np.ndarray,
    prototype_similarity: np.ndarray,
) -> np.ndarray:
    """Concatenate the established 70-d compact statistic for d=1 and d=2."""
    n = len(assignment)
    k = assignment.shape[1]
    features: list[np.ndarray] = []
    for distance in (1, 2):
        mask = np.asarray(distances == distance, dtype=np.float32)
        count = float(mask.sum())
        if count > 0:
            matrix = np.asarray(assignment.T @ mask @ assignment / count, dtype=np.float32)
        else:
            matrix = np.zeros((k, k), dtype=np.float32)
        diag = np.diag(matrix).astype(np.float32)
        row_mass = matrix.sum(axis=1, dtype=np.float32)
        trace = float(diag.sum())
        total = float(matrix.sum())
        semantic = float(np.sum(matrix * prototype_similarity))
        diagonal_semantic = float(np.sum(diag * np.diag(prototype_similarity)))
        scalars = np.asarray(
            [
                trace,
                total - trace,
                semantic,
                semantic - diagonal_semantic,
                count / max(float(n * n), 1.0),
                total,
            ],
            dtype=np.float32,
        )
        features.append(np.concatenate([diag, row_mass, scalars]).astype(np.float32))
    return np.concatenate(features).astype(np.float32)


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def load_frozen_occurrence_pair(broad: dict[str, Any], split: str) -> dict[str, np.ndarray]:
    # The compact-ablation runner merely copied the frozen occurrence logits
    # into ``base_scores``.  Full-fold regression checks established that the
    # vocabulary runner's saved logits are bit-identical, so the official-valid
    # path can consume those predictions directly and avoid evaluating unused
    # compact ablations on the held-out split.
    if all(family in broad.get("results", {}) for family in FAMILIES):
        rows = [broad["results"][family][split] for family in FAMILIES]
        score_key = "scores"
    else:
        rows = [
            broad["results"][f"{family}__distance_1_2__real"][split]
            for family in FAMILIES
        ]
        score_key = "base_scores"
    required = ("graph_indices", "labels", score_key)
    if any(any(key not in raw for key in required) for raw in rows):
        raise ValueError("broad result lacks saved frozen occurrence predictions")
    graph_indices = np.asarray(rows[0]["graph_indices"], dtype=np.int64)
    labels = np.asarray(rows[0]["labels"], dtype=np.float32)
    for raw in rows[1:]:
        if not np.array_equal(graph_indices, np.asarray(raw["graph_indices"], dtype=np.int64)):
            raise ValueError("broad family graph order mismatch")
        if not np.array_equal(labels, np.asarray(raw["labels"], dtype=np.float32)):
            raise ValueError("broad family labels mismatch")
    probabilities = np.mean(
        [sigmoid(np.asarray(raw[score_key], dtype=np.float32)) for raw in rows],
        axis=0,
    )
    probabilities = np.clip(probabilities, 1e-6, 1.0 - 1e-6)
    scores = np.log(probabilities / (1.0 - probabilities)).astype(np.float32)
    return {"graph_indices": graph_indices, "labels": labels, "scores": scores}


def gate_from_importance(importance: np.ndarray) -> np.ndarray:
    gate = 0.5 + np.asarray(importance, dtype=np.float32)
    gate /= max(float(gate.mean()), 1e-12)
    return gate.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-cache", required=True)
    ap.add_argument("--fold-cache", required=True)
    ap.add_argument("--vocabulary-result", required=True)
    ap.add_argument("--broad-result", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--evaluation-split", choices=("scaffold_fold", "official_valid", "official_test"), default="scaffold_fold")
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--selector-seed", type=int, default=20260728)
    ap.add_argument("--relation-seed", type=int, default=20260728)
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--gate-types", default=",".join(GATE_TYPES))
    ap.add_argument("--selector-top-nodes", type=int, default=3)
    ap.add_argument("--selector-c", type=float, default=0.1)
    ap.add_argument("--n-prototypes", type=int, default=32)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--relation-epochs", type=int, default=30)
    ap.add_argument("--relation-batch-size", type=int, default=256)
    ap.add_argument("--relation-lr", type=float, default=1e-3)
    ap.add_argument("--relation-weight-decay", type=float, default=1.0)
    ap.add_argument("--max-relation-residual", type=float, default=0.3125)
    ap.add_argument("--standardization-clip", type=float, default=5.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.fold < 0 or args.inner_splits < 2:
        raise ValueError("invalid fold/inner split configuration")
    gate_types = tuple(x.strip() for x in args.gate_types.split(",") if x.strip())
    if not gate_types or len(set(gate_types)) != len(gate_types) or not set(gate_types).issubset(GATE_TYPES):
        raise ValueError(f"invalid gate types: {gate_types}")
    if args.evaluation_split in ("official_valid", "official_test") and gate_types != ("uniform",):
        raise ValueError("frozen official-split protocol permits uniform gate only")
    if args.n_prototypes <= 1 or not (0 < args.sparsity <= args.n_prototypes):
        raise ValueError("invalid prototype/sparsity configuration")
    if args.selector_top_nodes <= 0 or args.selector_c <= 0:
        raise ValueError("invalid selector configuration")
    if min(args.relation_epochs, args.relation_batch_size) <= 0:
        raise ValueError("invalid relation training configuration")
    if args.relation_lr <= 0 or args.relation_weight_decay < 0:
        raise ValueError("invalid relation optimizer configuration")
    if args.max_relation_residual <= 0 or args.standardization_clip <= 0:
        raise ValueError("invalid residual/standardization configuration")

    try:
        import torch
        import torch.nn as nn
        import torch.nn.functional as F
        from torch.utils.data import DataLoader, TensorDataset
    except ImportError as exc:
        raise RuntimeError(f"missing training dependencies: {exc}") from exc

    seed_everything(args.seed, torch)
    started = time.time()
    bundle = load_molhiv(
        max_graphs=None if args.max_graphs <= 0 else args.max_graphs,
        seed=args.data_seed,
        with_features=False,
    )
    labels = np.asarray(bundle.y, dtype=np.float32)
    expected_original = np.asarray(bundle.meta["original_indices"], dtype=np.int64)
    official_train = np.asarray(bundle.split["train"], dtype=np.int64)
    official_valid = np.asarray(bundle.split["valid"], dtype=np.int64)
    official_test = np.asarray(bundle.split["test"], dtype=np.int64)

    protocol = resolve_evaluation_protocol(
        evaluation_split=args.evaluation_split, fold_cache=args.fold_cache, fold=args.fold,
        expected_original=expected_original, official_train=official_train, official_valid=official_valid, official_test=official_test,
    )
    fit_indices, heldout_indices = protocol.fit_indices, protocol.heldout_indices
    scaffold_groups = protocol.scaffold_groups

    with np.load(args.latent_cache, allow_pickle=False) as source:
        offsets = np.asarray(source["offsets"], dtype=np.int64)
        latent_original = np.asarray(source["original_indices"], dtype=np.int64)
        latent_train = np.asarray(source["train_indices"], dtype=np.int64)
        latent_valid = np.asarray(source["valid_indices"], dtype=np.int64)
        latent_test = np.asarray(source["test_indices"], dtype=np.int64)
        latents = np.asarray(source["latents"], dtype=np.float32)
    for actual, expected, name in (
        (latent_original, expected_original, "original"),
        (latent_train, official_train, "train"),
        (latent_valid, official_valid, "valid"),
        (latent_test, official_test, "test"),
    ):
        if not np.array_equal(actual, expected):
            raise ValueError(f"latent {name} indices mismatch")
    if latents.shape[0] != int(offsets[-1]):
        raise ValueError("latent row count and offsets disagree")
    forbidden = forbidden_encoded_indices(args.evaluation_split, official_valid, official_test)
    forbidden_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64) for i in forbidden
    ])
    if np.any(latents[forbidden_rows] != 0):
        raise AssertionError("forbidden official split latent rows are nonzero")

    fit_sha = sha256(fit_indices)
    validate_latent_sidecar(args.latent_cache, args.evaluation_split, fit_sha)
    vocabulary = json.loads(Path(args.vocabulary_result).read_text(encoding="utf-8"))
    broad = json.loads(Path(args.broad_result).read_text(encoding="utf-8"))
    validate_upstream_result(vocabulary, name="vocabulary", evaluation_split=args.evaluation_split, fold=args.fold, fit_sha=fit_sha)
    validate_upstream_result(broad, name="broad result", evaluation_split=args.evaluation_split, fold=args.fold, fit_sha=fit_sha)

    fit_row_mask = np.zeros(len(latents), dtype=bool)
    for graph_index in fit_indices:
        fit_row_mask[int(offsets[int(graph_index)]) : int(offsets[int(graph_index) + 1])] = True
    fit_position_by_graph = {int(graph): pos for pos, graph in enumerate(fit_indices)}
    prototypes: dict[str, np.ndarray] = {}
    prototype_rows: dict[str, np.ndarray] = {}
    source_graphs: dict[str, np.ndarray] = {}
    source_positions: dict[str, np.ndarray] = {}
    for family in FAMILIES:
        selection = vocabulary["results"][family]["selection"]
        rows = np.asarray(selection["source_node_rows"], dtype=np.int64)
        graphs = np.asarray(selection["source_graph_indices"], dtype=np.int64)
        if rows.shape != (args.n_prototypes,) or graphs.shape != (args.n_prototypes,):
            raise ValueError(f"invalid fixed vocabulary shape for {family}")
        if not np.all(fit_row_mask[rows]):
            raise ValueError(f"{family} prototype outside outer-fit rows")
        if any(int(graph) not in fit_position_by_graph for graph in graphs):
            raise ValueError(f"{family} source graph outside outer-fit graphs")
        if not all(
            int(offsets[int(graph)]) <= int(row) < int(offsets[int(graph) + 1])
            for row, graph in zip(rows, graphs)
        ):
            raise ValueError(f"{family} source row/source graph mismatch")
        p = normalize_rows(latents[rows])
        expected_hash = selection.get("prototype_sha256")
        if expected_hash is not None and sha256(p) != expected_hash:
            raise ValueError(f"{family} prototype hash mismatch")
        prototypes[family] = p
        prototype_rows[family] = rows
        source_graphs[family] = graphs
        source_positions[family] = np.asarray(
            [fit_position_by_graph[int(graph)] for graph in graphs], dtype=np.int64
        )

    selector_requested = any(gate_type != "uniform" for gate_type in gate_types)
    gate_values: dict[str, dict[str, np.ndarray]] = {}
    selector_audit: dict[str, Any] = {}
    inner_audit: list[dict[str, Any]] = []
    shuffled_labels: np.ndarray | None = None

    if selector_requested:
        group_by_graph = {int(i): str(g) for i, g in zip(official_train, scaffold_groups)}
        fit_groups = np.asarray([group_by_graph[int(i)] for i in fit_indices])
        inner_splitter = StratifiedGroupKFold(
            n_splits=args.inner_splits,
            shuffle=True,
            random_state=args.selector_seed + 104729 * (args.fold + 1),
        )
        inner_splits = [
            (np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
            for a, b in inner_splitter.split(
                np.zeros(len(fit_indices)), labels[fit_indices].astype(np.int64), fit_groups
            )
        ]
        inner_valid_counts = np.zeros(len(fit_indices), dtype=np.int64)
        for inner_fold, (inner_train, inner_valid) in enumerate(inner_splits):
            if np.intersect1d(inner_train, inner_valid).size:
                raise AssertionError(f"inner fold {inner_fold} index overlap")
            if set(fit_groups[inner_train].tolist()) & set(fit_groups[inner_valid].tolist()):
                raise AssertionError(f"inner fold {inner_fold} scaffold leakage")
            y_train = labels[fit_indices[inner_train]].astype(np.int64)
            y_valid = labels[fit_indices[inner_valid]].astype(np.int64)
            if np.unique(y_train).size != 2 or np.unique(y_valid).size != 2:
                raise ValueError(f"inner fold {inner_fold} lacks one class")
            inner_valid_counts[inner_valid] += 1
            inner_audit.append(
                {
                    "inner_fold": inner_fold,
                    "n_train": int(len(inner_train)),
                    "n_valid": int(len(inner_valid)),
                    "n_train_positive": int(y_train.sum()),
                    "n_valid_positive": int(y_valid.sum()),
                    "scaffold_overlap": 0,
                }
            )
        if not np.all(inner_valid_counts == 1):
            raise AssertionError("inner scaffold folds do not partition outer-fit graphs")

        shuffled_labels = labels[fit_indices].copy()
        shuffled_rng = np.random.default_rng(args.selector_seed + 71_017 * (args.fold + 1))
        shuffled_rng.shuffle(shuffled_labels)
        for family_index, family in enumerate(FAMILIES):
            features = _graph_candidate_features(
                latents,
                offsets,
                fit_indices,
                prototypes[family],
                args.selector_top_nodes,
            )
            task_importance, task_audit = _nested_candidate_importance(
                features,
                labels[fit_indices],
                inner_splits,
                source_positions[family],
                args.selector_seed + 1009 * family_index,
                args.selector_c,
            )
            shuffled_importance, shuffled_audit = _nested_candidate_importance(
                features,
                shuffled_labels,
                inner_splits,
                source_positions[family],
                args.selector_seed + 1 + 1009 * family_index,
                args.selector_c,
            )
            gates = {
                "uniform": np.ones(args.n_prototypes, dtype=np.float32),
                "task": gate_from_importance(task_importance),
                "shuffled_label": gate_from_importance(shuffled_importance),
            }
            gate_values[family] = gates
            selector_audit[family] = {
                "source_node_rows": prototype_rows[family].tolist(),
                "source_graph_indices": source_graphs[family].tolist(),
                "source_positions_sha256": sha256(source_positions[family]),
                "feature_sha256": sha256(features),
                "task": task_audit,
                "shuffled_label": shuffled_audit,
                "task_importance": task_importance.tolist(),
                "shuffled_label_importance": shuffled_importance.tolist(),
                "task_importance_sha256": sha256(task_importance),
                "shuffled_label_importance_sha256": sha256(shuffled_importance),
                "gates": {
                    gate_type: {
                        "values": gate.tolist(),
                        "sha256": sha256(gate),
                        "mean": float(gate.mean()),
                        "std": float(gate.std()),
                        "minimum": float(gate.min()),
                        "maximum": float(gate.max()),
                    }
                    for gate_type, gate in gates.items()
                },
            }
    else:
        for family in FAMILIES:
            uniform = np.ones(args.n_prototypes, dtype=np.float32)
            gate_values[family] = {"uniform": uniform}
            selector_audit[family] = {
                "selector_skipped": True,
                "reason": "only the frozen uniform gate was requested",
                "source_node_rows": prototype_rows[family].tolist(),
                "source_graph_indices": source_graphs[family].tolist(),
                "source_positions_sha256": sha256(source_positions[family]),
                "gates": {
                    "uniform": {
                        "values": uniform.tolist(),
                        "sha256": sha256(uniform),
                        "mean": float(uniform.mean()),
                        "std": float(uniform.std()),
                        "minimum": float(uniform.min()),
                        "maximum": float(uniform.max()),
                    }
                },
            }

    relation_graphs = np.concatenate([fit_indices, heldout_indices])
    per_family_dim = 2 * (2 * args.n_prototypes + 6)
    joint_dim = len(FAMILIES) * per_family_dim
    compact: dict[str, dict[str, np.ndarray]] = {
        gate_type: {
            mode: np.zeros((len(bundle.graphs), joint_dim), dtype=np.float32)
            for mode in RELATION_MODES
        }
        for gate_type in gate_types
    }
    assignment_audit: dict[str, dict[str, Any]] = {
        family: {
            "nodes": 0,
            "zero_mass": 0,
            "best_cosine": [],
            "gate_assignment_l1": {gate_type: [] for gate_type in gate_types if gate_type != "uniform"},
            "real_shuffled_l1": {gate_type: [] for gate_type in gate_types},
        }
        for family in FAMILIES
    }
    precompute_started = time.time()
    permutation_rng = np.random.default_rng(args.relation_seed + 13007 * (args.fold + 1))
    for graph_count, raw_index in enumerate(relation_graphs, start=1):
        i = int(raw_index)
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        distances = all_pairs_distances(bundle.graphs[i])
        permutation = permutation_rng.permutation(hi - lo)
        family_real: dict[str, list[np.ndarray]] = {gate_type: [] for gate_type in gate_types}
        family_shuffled: dict[str, list[np.ndarray]] = {gate_type: [] for gate_type in gate_types}
        for family in FAMILIES:
            assignment, cosine = sparse_assignment(latents[lo:hi], prototypes[family], args.sparsity)
            similarity = np.asarray(prototypes[family] @ prototypes[family].T, dtype=np.float32)
            assignment_audit[family]["nodes"] += hi - lo
            assignment_audit[family]["zero_mass"] += int(np.sum(assignment.sum(axis=1) == 0))
            assignment_audit[family]["best_cosine"].append(float(cosine.max(axis=1).mean()))
            for gate_type in gate_types:
                gated = apply_gate(assignment, gate_values[family][gate_type])
                real = compact_distance_feature(gated, distances, similarity)
                shuffled = compact_distance_feature(gated[permutation], distances, similarity)
                family_real[gate_type].append(real)
                family_shuffled[gate_type].append(shuffled)
                assignment_audit[family]["real_shuffled_l1"][gate_type].append(
                    float(np.mean(np.abs(real - shuffled)))
                )
                if gate_type != "uniform":
                    assignment_audit[family]["gate_assignment_l1"][gate_type].append(
                        float(np.mean(np.abs(gated - assignment)))
                    )
        for gate_type in gate_types:
            compact[gate_type]["real"][i] = np.concatenate(family_real[gate_type])
            compact[gate_type]["assignment_shuffled"][i] = np.concatenate(family_shuffled[gate_type])
        if graph_count % 1000 == 0:
            print(f"fixed-gate relation precompute graphs={graph_count}/{len(relation_graphs)}", flush=True)
    precompute_sec = time.time() - precompute_started

    base = {
        split: load_frozen_occurrence_pair(broad, split)
        for split in ("fit", "heldout")
    }
    if not np.array_equal(base["fit"]["graph_indices"], fit_indices):
        raise AssertionError("frozen base fit prediction order mismatch")
    if not np.array_equal(base["heldout"]["graph_indices"], heldout_indices):
        raise AssertionError("frozen base heldout prediction order mismatch")
    if not np.array_equal(base["fit"]["labels"], labels[fit_indices]):
        raise AssertionError("frozen base fit labels mismatch")
    if not np.array_equal(base["heldout"]["labels"], labels[heldout_indices]):
        raise AssertionError("frozen base heldout labels mismatch")

    device = torch.device(args.device)

    class BoundedLinearResidual(nn.Module):
        def __init__(self, dim: int):
            super().__init__()
            self.linear = nn.Linear(dim, 1, bias=False)
            nn.init.zeros_(self.linear.weight)

        def forward(self, x):
            return args.max_relation_residual * torch.tanh(self.linear(x).view(-1))

    def standardize(source: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
        fit_x = np.asarray(source[fit_indices], dtype=np.float32)
        heldout_x = np.asarray(source[heldout_indices], dtype=np.float32)
        mean = fit_x.mean(axis=0, dtype=np.float64).astype(np.float32)
        std = fit_x.std(axis=0, dtype=np.float64).astype(np.float32)
        active = std > 1e-7
        safe_std = np.where(active, std, 1.0).astype(np.float32)
        fit_z = np.clip(
            (fit_x - mean) / safe_std,
            -args.standardization_clip,
            args.standardization_clip,
        ).astype(np.float32)
        heldout_z = np.clip(
            (heldout_x - mean) / safe_std,
            -args.standardization_clip,
            args.standardization_clip,
        ).astype(np.float32)
        return fit_z, heldout_z, {
            "mean_sha256": sha256(mean),
            "std_sha256": sha256(std),
            "active_dimensions": int(active.sum()),
            "inactive_dimensions": int((~active).sum()),
            "fit_standardized_abs_mean": float(np.mean(np.abs(fit_z))),
        }

    @torch.no_grad()
    def evaluate(head: Any, z: np.ndarray, base_raw: dict[str, np.ndarray]) -> dict[str, Any]:
        head.eval()
        residual = head(torch.from_numpy(z).to(device)).cpu().numpy().astype(np.float32)
        scores = (base_raw["scores"] + residual).astype(np.float32)
        metrics: dict[str, Any] = {
            "auc": float(roc_auc_score(base_raw["labels"], scores)),
            "base_auc": float(roc_auc_score(base_raw["labels"], base_raw["scores"])),
            "n_graphs": int(len(scores)),
            "n_positive": int(base_raw["labels"].sum()),
            "mean_abs_relation_residual": float(np.mean(np.abs(residual))),
            "max_abs_relation_residual": float(np.max(np.abs(residual))),
            "residual_std": float(np.std(residual)),
            "score_sha256": sha256(scores.astype(np.float64)),
            "residual_sha256": sha256(residual.astype(np.float64)),
        }
        if args.save_predictions:
            metrics.update(
                {
                    "graph_indices": base_raw["graph_indices"].tolist(),
                    "labels": base_raw["labels"].tolist(),
                    "scores": scores.tolist(),
                    "base_scores": base_raw["scores"].tolist(),
                    "relation_residuals": residual.tolist(),
                }
            )
        return metrics

    def train_head(gate_type: str, mode: str, source: np.ndarray) -> dict[str, Any]:
        fit_z, heldout_z, normalization = standardize(source)
        seed_everything(args.seed, torch)
        head = BoundedLinearResidual(fit_z.shape[1]).to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(),
            lr=args.relation_lr,
            weight_decay=args.relation_weight_decay,
        )
        dataset = TensorDataset(
            torch.from_numpy(fit_z),
            torch.from_numpy(base["fit"]["scores"]),
            torch.from_numpy(base["fit"]["labels"]),
        )
        generator = torch.Generator().manual_seed(args.seed + 4001)
        loader = DataLoader(
            dataset,
            batch_size=args.relation_batch_size,
            shuffle=True,
            generator=generator,
            num_workers=0,
        )
        history: list[dict[str, float]] = []
        train_started = time.time()
        for epoch in range(1, args.relation_epochs + 1):
            head.train()
            total = 0.0
            seen = 0
            for xb, bb, yb in loader:
                xb, bb, yb = xb.to(device), bb.to(device), yb.to(device)
                residual = head(xb)
                loss = F.binary_cross_entropy_with_logits(bb + residual, yb.float())
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                optimizer.step()
                n = int(len(yb))
                total += float(loss.detach()) * n
                seen += n
            history.append(
                {
                    "epoch": float(epoch),
                    "task": total / max(seen, 1),
                    "weight_l2": float(head.linear.weight.detach().norm().cpu()),
                }
            )
        print(
            f"{gate_type}__{mode} dim={fit_z.shape[1]} "
            f"task={history[-1]['task']:.5f} w={history[-1]['weight_l2']:.4f}",
            flush=True,
        )
        return {
            "gate_type": gate_type,
            "relation_mode": mode,
            "distance_bins": list(DISTANCE_BINS),
            "families": list(FAMILIES),
            "relation_dim": int(fit_z.shape[1]),
            "fit": evaluate(head, fit_z, base["fit"]),
            "heldout": evaluate(head, heldout_z, base["heldout"]),
            "normalization": normalization,
            "history": history,
            "relation_trainable_parameters": int(
                sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
            ),
            "relation_weight_l2": float(head.linear.weight.detach().norm().cpu()),
            "relation_weight_sha256": sha256(head.linear.weight.detach().cpu().numpy()),
            "elapsed_sec": time.time() - train_started,
        }

    results: dict[str, Any] = {}
    for gate_type in gate_types:
        for mode in RELATION_MODES:
            key = f"{gate_type}__{mode}"
            results[key] = train_head(gate_type, mode, compact[gate_type][mode])

    assignment_summary: dict[str, Any] = {}
    for family, audit in assignment_audit.items():
        assignment_summary[family] = {
            "n_nodes": int(audit["nodes"]),
            "zero_mass_node_fraction": float(audit["zero_mass"] / max(audit["nodes"], 1)),
            "mean_graph_best_signed_cosine": float(np.mean(audit["best_cosine"])),
            "mean_gate_assignment_l1": {
                gate_type: float(np.mean(values))
                for gate_type, values in audit["gate_assignment_l1"].items()
            },
            "mean_real_assignment_shuffled_feature_l1": {
                gate_type: float(np.mean(values))
                for gate_type, values in audit["real_shuffled_l1"].items()
            },
            "feature_sha256": {
                f"{gate_type}__{mode}": sha256(compact[gate_type][mode][relation_graphs])
                for gate_type in gate_types
                for mode in RELATION_MODES
            },
        }

    base_heldout_auc = float(roc_auc_score(base["heldout"]["labels"], base["heldout"]["scores"]))
    comparisons = {"frozen_base_heldout_auc": base_heldout_auc}
    for gate_type in gate_types:
        comparisons[f"{gate_type}_real_gain_over_base"] = results[f"{gate_type}__real"]["heldout"]["auc"] - base_heldout_auc
        comparisons[f"{gate_type}_real_minus_assignment_shuffled"] = (
            results[f"{gate_type}__real"]["heldout"]["auc"]
            - results[f"{gate_type}__assignment_shuffled"]["heldout"]["auc"]
        )

    doc = {
        "protocol_id": "molhiv_fixed_broad_vocabulary_nested_relation_gates_v1",
        "date": "2026-07-28",
        "hypothesis": (
            "Task matching is more stable as a smooth relation gate over a fixed broad deterministic "
            "real-patch vocabulary than as reservoir-dependent hard prototype selection."
        ),
        "selection_policy": {
            "development_data": evaluation_description(args.evaluation_split, args.max_graphs),
            "fixed_vocabulary": "outer-fit-only deterministic farthest plus scaffold-facility prototypes",
            "task_gate": "source-clean nested inner-scaffold OOF balanced-log-loss ablation rank",
            "gate_formula": "gate=(0.5+importance)/mean(0.5+importance)",
            "relation": "top-3 positive-cosine assignments; exact distance 1+2 compact statistics",
            "frozen_base": "probability average of broad farthest/scaffold occurrence logits",
            "residual_cap": args.max_relation_residual,
            "official_valid_evaluations": protocol.official_valid_evaluations,
            "official_test_evaluations": protocol.official_test_evaluations,
            "status": (
                "frozen terminal official-test evaluation; no post-test tuning"
                if args.evaluation_split == "official_test"
                else "development robustness diagnostic; no official split promotion"
            ),
        },
        "config": vars(args),
        "evaluation_split": args.evaluation_split,
        "fold": args.fold,
        "n_fit": int(len(fit_indices)),
        "n_heldout": int(len(heldout_indices)),
        "n_fit_positive": int(labels[fit_indices].sum()),
        "n_heldout_positive": int(labels[heldout_indices].sum()),
        "fit_indices_sha256": fit_sha,
        "heldout_indices_sha256": sha256(heldout_indices),
        "relation_graph_indices_sha256": sha256(relation_graphs),
        "shuffled_labels_sha256": None if shuffled_labels is None else sha256(shuffled_labels),
        "inner_splits": inner_audit,
        "families": list(FAMILIES),
        "gate_types": list(gate_types),
        "relation_modes": list(RELATION_MODES),
        "distance_bins": list(DISTANCE_BINS),
        "per_family_relation_dim": per_family_dim,
        "joint_relation_dim": joint_dim,
        "selector_audit": selector_audit,
        "assignment_audit": assignment_summary,
        "relation_precompute_sec": precompute_sec,
        "base_occurrence": {
            "fit_auc": float(roc_auc_score(base["fit"]["labels"], base["fit"]["scores"])),
            "heldout_auc": base_heldout_auc,
            "fit_score_sha256": sha256(base["fit"]["scores"].astype(np.float64)),
            "heldout_score_sha256": sha256(base["heldout"]["scores"].astype(np.float64)),
        },
        "results": results,
        "comparisons": comparisons,
        "elapsed_sec": time.time() - started,
        "output": args.output,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "comparisons": comparisons}, indent=2), flush=True)


if __name__ == "__main__":
    main()

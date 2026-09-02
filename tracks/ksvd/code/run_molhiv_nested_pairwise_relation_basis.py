"""Nested low-rank pairwise relation bases on fixed real-patch vocabularies.

This experiment keeps the broad farthest/scaffold-facility prototypes and the
already-trained compact exact-distance relation predictor frozen.  It asks a
narrower question: does preserving *which prototype pair* occurs at distance 1
or 2 add stable task signal beyond compact row/diagonal summaries?

For each outer fold, inner-scaffold logistic models learn full symmetric pair
coefficients from inner-train graphs only.  Their normalized coefficient
matrices are averaged and compressed to rank-r eigen-directions.  A small
bounded sidecar is then trained on the resulting pair projections.  Matched
shuffled-label, random-basis, and node-assignment-shuffled controls are run.
Official valid/test remain unencoded and unevaluated.
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
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedGroupKFold
from sklearn.preprocessing import StandardScaler

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import load_molhiv

FAMILIES = ("farthest", "scaffold_facility")
DISTANCES = (1, 2)
BASIS_TYPES = ("task", "shuffled_label", "random")
RELATION_MODES = ("real", "assignment_shuffled")


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


def symmetric_vector(matrix: np.ndarray, tri_i: np.ndarray, tri_j: np.ndarray) -> np.ndarray:
    """Isometric upper-triangle vectorization for symmetric matrices."""
    values = np.asarray(matrix[tri_i, tri_j], dtype=np.float32).copy()
    values[tri_i != tri_j] *= np.sqrt(2.0)
    return values


def vector_to_symmetric(vector: np.ndarray, tri_i: np.ndarray, tri_j: np.ndarray, size: int) -> np.ndarray:
    matrix = np.zeros((size, size), dtype=np.float64)
    values = np.asarray(vector, dtype=np.float64).copy()
    offdiag = tri_i != tri_j
    values[offdiag] /= np.sqrt(2.0)
    matrix[tri_i, tri_j] = values
    matrix[tri_j, tri_i] = values
    return matrix


def quadratic_vector(direction: np.ndarray, tri_i: np.ndarray, tri_j: np.ndarray) -> np.ndarray:
    """Vector q such that symmetric_vector(M) @ q == u.T @ M @ u."""
    q = np.asarray(direction[tri_i] * direction[tri_j], dtype=np.float64)
    q[tri_i != tri_j] *= np.sqrt(2.0)
    return q.astype(np.float32)


def load_frozen_compact_base(result: dict[str, Any], split: str) -> dict[str, np.ndarray]:
    raw = result["results"]["uniform__real"][split]
    required = ("graph_indices", "labels", "scores")
    if any(key not in raw for key in required):
        raise ValueError("fixed-gate result lacks saved uniform real predictions")
    return {
        "graph_indices": np.asarray(raw["graph_indices"], dtype=np.int64),
        "labels": np.asarray(raw["labels"], dtype=np.float32),
        "scores": np.asarray(raw["scores"], dtype=np.float32),
    }


def balanced_log_loss(logits: np.ndarray, labels: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.float64)
    z = np.asarray(logits, dtype=np.float64)
    n_pos = max(float(y.sum()), 1.0)
    n_neg = max(float((1.0 - y).sum()), 1.0)
    weights = np.where(y > 0.5, 0.5 / n_pos, 0.5 / n_neg)
    return float(np.sum(weights * (np.logaddexp(0.0, z) - y * z)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-cache", required=True)
    ap.add_argument("--fold-cache", required=True)
    ap.add_argument("--vocabulary-result", required=True)
    ap.add_argument("--compact-base-result", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--basis-seed", type=int, default=20260728)
    ap.add_argument("--projection-seed", type=int, default=20260728)
    ap.add_argument("--relation-seed", type=int, default=20260728)
    ap.add_argument("--inner-splits", type=int, default=3)
    ap.add_argument("--n-prototypes", type=int, default=32)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--rank", type=int, default=4)
    ap.add_argument("--semantic-rank", type=int, default=8)
    ap.add_argument("--selector-c", type=float, default=0.01)
    ap.add_argument("--relation-epochs", type=int, default=30)
    ap.add_argument("--relation-batch-size", type=int, default=256)
    ap.add_argument("--relation-lr", type=float, default=1e-3)
    ap.add_argument("--relation-weight-decay", type=float, default=1.0)
    ap.add_argument("--max-relation-residual", type=float, default=0.25)
    ap.add_argument("--standardization-clip", type=float, default=5.0)
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--save-predictions", action="store_true")
    ap.add_argument("--output", required=True)
    args = ap.parse_args()

    if args.fold < 0 or args.inner_splits < 2:
        raise ValueError("invalid fold/inner split configuration")
    if args.n_prototypes <= 1 or not (0 < args.sparsity <= args.n_prototypes):
        raise ValueError("invalid prototype/sparsity configuration")
    if not (0 < args.rank <= args.n_prototypes):
        raise ValueError("invalid low-rank basis size")
    if not (0 < args.semantic_rank <= args.n_prototypes):
        raise ValueError("invalid semantic basis size")
    if args.selector_c <= 0:
        raise ValueError("selector C must be positive")
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

    with np.load(args.fold_cache, allow_pickle=False) as folds:
        fit_indices = np.asarray(folds[f"fold_{args.fold}_train_indices"], dtype=np.int64)
        heldout_indices = np.asarray(folds[f"fold_{args.fold}_valid_indices"], dtype=np.int64)
        fold_original = np.asarray(folds["original_indices"], dtype=np.int64)
        fold_official_train = np.asarray(folds["official_train_indices"], dtype=np.int64)
        scaffold_groups = np.asarray(folds["train_scaffold_groups"]).astype(str)
    if not np.array_equal(fold_original, expected_original):
        raise ValueError("fold-cache original_indices mismatch")
    if not np.array_equal(fold_official_train, official_train):
        raise ValueError("fold-cache official train mismatch")
    if np.intersect1d(fit_indices, heldout_indices).size:
        raise AssertionError("outer fit/heldout overlap")
    if set(np.concatenate([fit_indices, heldout_indices]).tolist()) != set(official_train.tolist()):
        raise AssertionError("outer scaffold fold does not partition official train")

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
    official_nontrain_rows = np.concatenate(
        [
            np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64)
            for i in np.concatenate([official_valid, official_test])
        ]
    )
    if np.any(latents[official_nontrain_rows] != 0):
        raise AssertionError("official valid/test latent rows are nonzero")
    latent_meta_path = Path(args.latent_cache).with_suffix(".json")
    if not latent_meta_path.exists():
        raise FileNotFoundError(f"missing latent audit sidecar: {latent_meta_path}")
    latent_meta = json.loads(latent_meta_path.read_text(encoding="utf-8"))
    if bool(latent_meta.get("official_valid_encoded", True)) or bool(
        latent_meta.get("official_test_encoded", True)
    ):
        raise ValueError("latent cache encoded official valid/test")

    fit_sha = sha256(fit_indices)
    vocabulary = json.loads(Path(args.vocabulary_result).read_text(encoding="utf-8"))
    compact_base_doc = json.loads(Path(args.compact_base_result).read_text(encoding="utf-8"))
    for name, doc in (("vocabulary", vocabulary), ("compact base", compact_base_doc)):
        if int(doc.get("fold", -1)) != args.fold or doc.get("fit_indices_sha256") != fit_sha:
            raise ValueError(f"{name} fold/split mismatch")
        policy = doc.get("selection_policy", {})
        if int(policy.get("official_valid_evaluations", -1)) != 0 or int(
            policy.get("official_test_evaluations", -1)
        ) != 0:
            raise ValueError(f"{name} was not development-only")

    fit_row_mask = np.zeros(len(latents), dtype=bool)
    for graph_index in fit_indices:
        fit_row_mask[int(offsets[int(graph_index)]) : int(offsets[int(graph_index) + 1])] = True
    prototypes: dict[str, np.ndarray] = {}
    prototype_audit: dict[str, Any] = {}
    for family in FAMILIES:
        selection = vocabulary["results"][family]["selection"]
        rows = np.asarray(selection["source_node_rows"], dtype=np.int64)
        graphs = np.asarray(selection["source_graph_indices"], dtype=np.int64)
        if rows.shape != (args.n_prototypes,) or graphs.shape != (args.n_prototypes,):
            raise ValueError(f"invalid vocabulary shape for {family}")
        if not np.all(fit_row_mask[rows]):
            raise ValueError(f"{family} prototype outside outer-fit rows")
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
        prototype_audit[family] = {
            "source_node_rows": rows.tolist(),
            "source_graph_indices": graphs.tolist(),
            "prototype_sha256": sha256(p),
        }

    base = {
        split: load_frozen_compact_base(compact_base_doc, split)
        for split in ("fit", "heldout")
    }
    if not np.array_equal(base["fit"]["graph_indices"], fit_indices):
        raise AssertionError("compact base fit prediction order mismatch")
    if not np.array_equal(base["heldout"]["graph_indices"], heldout_indices):
        raise AssertionError("compact base heldout prediction order mismatch")
    if not np.array_equal(base["fit"]["labels"], labels[fit_indices]):
        raise AssertionError("compact base fit labels mismatch")
    if not np.array_equal(base["heldout"]["labels"], labels[heldout_indices]):
        raise AssertionError("compact base heldout labels mismatch")

    tri_i, tri_j = np.triu_indices(args.n_prototypes)
    pair_dim = len(tri_i)
    blocks = [(family, distance) for family in FAMILIES for distance in DISTANCES]
    block_count = len(blocks)
    full_pair_dim = block_count * pair_dim
    relation_graphs = np.concatenate([fit_indices, heldout_indices])
    pair_real = np.zeros((len(bundle.graphs), full_pair_dim), dtype=np.float32)
    pair_shuffled = np.zeros_like(pair_real)
    assignment_audit: dict[str, Any] = {
        family: {"nodes": 0, "zero_mass": 0, "best_cosine": [], "real_shuffled_l1": []}
        for family in FAMILIES
    }
    pair_fraction_sum = {distance: 0.0 for distance in DISTANCES}
    precompute_started = time.time()
    permutation_rng = np.random.default_rng(args.relation_seed + 13007 * (args.fold + 1))
    for graph_count, raw_index in enumerate(relation_graphs, start=1):
        i = int(raw_index)
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        distances = all_pairs_distances(bundle.graphs[i])
        permutation = permutation_rng.permutation(hi - lo)
        block_real: list[np.ndarray] = []
        block_shuffled: list[np.ndarray] = []
        for family_index, family in enumerate(FAMILIES):
            assignment, cosine = sparse_assignment(latents[lo:hi], prototypes[family], args.sparsity)
            assignment_audit[family]["nodes"] += hi - lo
            assignment_audit[family]["zero_mass"] += int(np.sum(assignment.sum(axis=1) == 0))
            assignment_audit[family]["best_cosine"].append(float(cosine.max(axis=1).mean()))
            for distance in DISTANCES:
                mask = np.asarray(distances == distance, dtype=np.float32)
                count = float(mask.sum())
                if count > 0:
                    real_matrix = np.asarray(assignment.T @ mask @ assignment / count, dtype=np.float32)
                    shuffled_assignment = assignment[permutation]
                    shuffled_matrix = np.asarray(
                        shuffled_assignment.T @ mask @ shuffled_assignment / count,
                        dtype=np.float32,
                    )
                else:
                    real_matrix = np.zeros((args.n_prototypes, args.n_prototypes), dtype=np.float32)
                    shuffled_matrix = np.zeros_like(real_matrix)
                real_vector = symmetric_vector(real_matrix, tri_i, tri_j)
                shuffled_vector = symmetric_vector(shuffled_matrix, tri_i, tri_j)
                block_real.append(real_vector)
                block_shuffled.append(shuffled_vector)
                assignment_audit[family]["real_shuffled_l1"].append(
                    float(np.mean(np.abs(real_vector - shuffled_vector)))
                )
                if family_index == 0:
                    pair_fraction_sum[distance] += count / max(float((hi - lo) ** 2), 1.0)
        pair_real[i] = np.concatenate(block_real)
        pair_shuffled[i] = np.concatenate(block_shuffled)
        if graph_count % 1000 == 0:
            print(f"pairwise relation precompute graphs={graph_count}/{len(relation_graphs)}", flush=True)
    precompute_sec = time.time() - precompute_started

    group_by_graph = {int(i): str(g) for i, g in zip(official_train, scaffold_groups)}
    fit_groups = np.asarray([group_by_graph[int(i)] for i in fit_indices])
    splitter = StratifiedGroupKFold(
        n_splits=args.inner_splits,
        shuffle=True,
        random_state=args.basis_seed + 104729 * (args.fold + 1),
    )
    inner_splits = [
        (np.asarray(a, dtype=np.int64), np.asarray(b, dtype=np.int64))
        for a, b in splitter.split(
            np.zeros(len(fit_indices)), labels[fit_indices].astype(np.int64), fit_groups
        )
    ]
    inner_valid_counts = np.zeros(len(fit_indices), dtype=np.int64)
    for inner_fold, (inner_train, inner_valid) in enumerate(inner_splits):
        if np.intersect1d(inner_train, inner_valid).size:
            raise AssertionError(f"inner fold {inner_fold} overlap")
        if set(fit_groups[inner_train].tolist()) & set(fit_groups[inner_valid].tolist()):
            raise AssertionError(f"inner fold {inner_fold} scaffold leakage")
        inner_valid_counts[inner_valid] += 1
    if not np.all(inner_valid_counts == 1):
        raise AssertionError("inner folds do not partition outer-fit graphs")

    fit_pair = np.asarray(pair_real[fit_indices], dtype=np.float32)
    true_labels = labels[fit_indices].astype(np.int64)
    shuffled_labels = labels[fit_indices].copy()
    shuffled_rng = np.random.default_rng(args.basis_seed + 71017 * (args.fold + 1))
    shuffled_rng.shuffle(shuffled_labels)
    shuffled_labels = shuffled_labels.astype(np.int64)

    def learn_basis(targets: np.ndarray, basis_name: str, seed_offset: int) -> tuple[list[dict[str, Any]], dict[str, Any]]:
        coefficient_by_block: list[list[np.ndarray]] = [[] for _ in blocks]
        fold_audit: list[dict[str, Any]] = []
        for inner_fold, (train_pos, valid_pos) in enumerate(inner_splits):
            scaler = StandardScaler()
            x_train = scaler.fit_transform(fit_pair[train_pos])
            x_valid = scaler.transform(fit_pair[valid_pos])
            y_train = targets[train_pos]
            y_valid = targets[valid_pos]
            if np.unique(y_train).size != 2 or np.unique(y_valid).size != 2:
                raise ValueError(f"{basis_name} inner fold {inner_fold} lacks one class")
            model = LogisticRegression(
                penalty="l2",
                C=args.selector_c,
                class_weight="balanced",
                solver="liblinear",
                random_state=args.basis_seed + seed_offset + inner_fold,
                max_iter=3000,
            )
            model.fit(x_train, y_train)
            logits = model.decision_function(x_valid).astype(np.float64)
            raw_coef = model.coef_.reshape(-1).astype(np.float64) / np.maximum(
                scaler.scale_.astype(np.float64), 1e-12
            )
            for block_index in range(block_count):
                lo = block_index * pair_dim
                hi = lo + pair_dim
                matrix = vector_to_symmetric(raw_coef[lo:hi], tri_i, tri_j, args.n_prototypes)
                norm = float(np.linalg.norm(matrix))
                if norm > 1e-12:
                    matrix /= norm
                coefficient_by_block[block_index].append(matrix)
            fold_audit.append(
                {
                    "inner_fold": inner_fold,
                    "n_train": int(len(train_pos)),
                    "n_valid": int(len(valid_pos)),
                    "n_train_positive": int(y_train.sum()),
                    "n_valid_positive": int(y_valid.sum()),
                    "valid_auc": float(roc_auc_score(y_valid, logits)),
                    "valid_balanced_log_loss": balanced_log_loss(logits, y_valid),
                    "coefficient_l2_standardized": float(np.linalg.norm(model.coef_)),
                    "n_iter": int(model.n_iter_[0]),
                    "valid_score_sha256": sha256(logits.astype(np.float64)),
                }
            )
        basis_blocks: list[dict[str, Any]] = []
        for block_index, (family, distance) in enumerate(blocks):
            mean_matrix = np.mean(np.stack(coefficient_by_block[block_index], axis=0), axis=0)
            mean_matrix = 0.5 * (mean_matrix + mean_matrix.T)
            eigenvalues, eigenvectors = np.linalg.eigh(mean_matrix)
            order = np.argsort(np.abs(eigenvalues))[::-1][: args.rank]
            values = eigenvalues[order]
            vectors = eigenvectors[:, order]
            signs = np.where(values >= 0.0, 1.0, -1.0).astype(np.float32)
            projection_vectors = np.stack(
                [quadratic_vector(vectors[:, j], tri_i, tri_j) for j in range(args.rank)],
                axis=1,
            )
            basis_blocks.append(
                {
                    "family": family,
                    "distance": distance,
                    "signs": signs,
                    "projection_vectors": projection_vectors,
                    "mean_coefficient_matrix": mean_matrix.astype(np.float32),
                    "eigenvalues": values.astype(np.float32),
                    "eigenvectors": vectors.astype(np.float32),
                }
            )
        audit = {
            "method": "mean of inner-train standardized-logistic raw coefficient matrices; per-fold block Frobenius normalization; top-|eigenvalue| directions",
            "selector_c": args.selector_c,
            "folds": fold_audit,
        }
        return basis_blocks, audit

    task_basis, task_basis_audit = learn_basis(true_labels, "task", 0)
    shuffled_basis, shuffled_basis_audit = learn_basis(shuffled_labels, "shuffled_label", 1000)
    random_rng = np.random.default_rng(args.basis_seed + 900001 * (args.fold + 1))
    random_basis: list[dict[str, Any]] = []
    for family, distance in blocks:
        q, _ = np.linalg.qr(random_rng.normal(size=(args.n_prototypes, args.rank)))
        signs = random_rng.choice(np.asarray([-1.0, 1.0], dtype=np.float32), size=args.rank)
        projection_vectors = np.stack(
            [quadratic_vector(q[:, j], tri_i, tri_j) for j in range(args.rank)], axis=1
        )
        random_basis.append(
            {
                "family": family,
                "distance": distance,
                "signs": signs.astype(np.float32),
                "projection_vectors": projection_vectors,
                "eigenvalues": np.zeros(args.rank, dtype=np.float32),
                "eigenvectors": q.astype(np.float32),
            }
        )
    bases = {
        "task": task_basis,
        "shuffled_label": shuffled_basis,
        "random": random_basis,
    }

    def project(source: np.ndarray, basis: list[dict[str, Any]]) -> np.ndarray:
        output = np.zeros((len(bundle.graphs), block_count * args.rank), dtype=np.float32)
        for block_index, block_basis in enumerate(basis):
            pair_lo = block_index * pair_dim
            pair_hi = pair_lo + pair_dim
            out_lo = block_index * args.rank
            out_hi = out_lo + args.rank
            values = source[relation_graphs, pair_lo:pair_hi] @ block_basis["projection_vectors"]
            values *= block_basis["signs"][None, :]
            output[relation_graphs, out_lo:out_hi] = values.astype(np.float32)
        return output

    projected: dict[str, dict[str, np.ndarray]] = {
        basis_type: {
            "real": project(pair_real, bases[basis_type]),
            "assignment_shuffled": project(pair_shuffled, bases[basis_type]),
        }
        for basis_type in BASIS_TYPES
    }

    def canonicalize_columns(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float64).copy()
        for column in range(vectors.shape[1]):
            pivot = int(np.argmax(np.abs(vectors[:, column])))
            if vectors[pivot, column] < 0.0:
                vectors[:, column] *= -1.0
        return vectors.astype(np.float32)

    def subspace_projection_vectors(vectors: np.ndarray) -> np.ndarray:
        vectors = np.asarray(vectors, dtype=np.float64)
        columns: list[np.ndarray] = []
        for a in range(vectors.shape[1]):
            for b in range(a, vectors.shape[1]):
                if a == b:
                    weight = np.outer(vectors[:, a], vectors[:, a])
                else:
                    weight = (
                        np.outer(vectors[:, a], vectors[:, b])
                        + np.outer(vectors[:, b], vectors[:, a])
                    ) / np.sqrt(2.0)
                columns.append(symmetric_vector(weight, tri_i, tri_j))
        return np.stack(columns, axis=1).astype(np.float32)

    semantic_family_vectors: dict[str, np.ndarray] = {}
    semantic_family_eigenvalues: dict[str, np.ndarray] = {}
    matched_random_family_vectors: dict[str, np.ndarray] = {}
    semantic_random_rng = np.random.default_rng(
        args.projection_seed + 1_700_003 * (args.fold + 1)
    )
    for family in FAMILIES:
        gram = np.asarray(prototypes[family] @ prototypes[family].T, dtype=np.float64)
        eigenvalues, eigenvectors = np.linalg.eigh(gram)
        order = np.argsort(eigenvalues)[::-1][: args.semantic_rank]
        semantic_family_vectors[family] = canonicalize_columns(eigenvectors[:, order])
        semantic_family_eigenvalues[family] = eigenvalues[order].astype(np.float32)
        random_q, _ = np.linalg.qr(
            semantic_random_rng.normal(size=(args.n_prototypes, args.semantic_rank))
        )
        matched_random_family_vectors[family] = canonicalize_columns(random_q)

    semantic_projection_by_block: list[np.ndarray] = []
    random_semantic_projection_by_block: list[np.ndarray] = []
    for family, _distance in blocks:
        semantic_projection_by_block.append(
            subspace_projection_vectors(semantic_family_vectors[family])
        )
        random_semantic_projection_by_block.append(
            subspace_projection_vectors(matched_random_family_vectors[family])
        )
    semantic_block_dim = args.semantic_rank * (args.semantic_rank + 1) // 2

    def project_subspaces(source: np.ndarray, block_projections: list[np.ndarray]) -> np.ndarray:
        output = np.zeros(
            (len(bundle.graphs), block_count * semantic_block_dim), dtype=np.float32
        )
        for block_index, projection_matrix in enumerate(block_projections):
            pair_lo = block_index * pair_dim
            pair_hi = pair_lo + pair_dim
            out_lo = block_index * semantic_block_dim
            out_hi = out_lo + semantic_block_dim
            output[relation_graphs, out_lo:out_hi] = (
                source[relation_graphs, pair_lo:pair_hi] @ projection_matrix
            ).astype(np.float32)
        return output

    semantic_projected = {
        "semantic_pair": {
            "real": project_subspaces(pair_real, semantic_projection_by_block),
            "assignment_shuffled": project_subspaces(
                pair_shuffled, semantic_projection_by_block
            ),
        },
        "random_semantic_rank": {
            "real": project_subspaces(pair_real, random_semantic_projection_by_block),
            "assignment_shuffled": project_subspaces(
                pair_shuffled, random_semantic_projection_by_block
            ),
        },
    }

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

    def train_head(basis_type: str, mode: str, source: np.ndarray) -> dict[str, Any]:
        fit_z, heldout_z, normalization = standardize(source)
        seed_everything(args.seed, torch)
        head = BoundedLinearResidual(fit_z.shape[1]).to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=args.relation_lr, weight_decay=args.relation_weight_decay
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
            f"{basis_type}__{mode} dim={fit_z.shape[1]} "
            f"task={history[-1]['task']:.5f} w={history[-1]['weight_l2']:.4f}",
            flush=True,
        )
        return {
            "basis_type": basis_type,
            "relation_mode": mode,
            "rank": args.rank,
            "projection_dim": int(fit_z.shape[1]),
            "fit": evaluate(head, fit_z, base["fit"]),
            "heldout": evaluate(head, heldout_z, base["heldout"]),
            "normalization": normalization,
            "history": history,
            "trainable_parameters": int(
                sum(parameter.numel() for parameter in head.parameters() if parameter.requires_grad)
            ),
            "weight_l2": float(head.linear.weight.detach().norm().cpu()),
            "weight_sha256": sha256(head.linear.weight.detach().cpu().numpy()),
            "elapsed_sec": time.time() - train_started,
        }

    results: dict[str, Any] = {}
    for basis_type in BASIS_TYPES:
        for mode in RELATION_MODES:
            key = f"{basis_type}__{mode}"
            results[key] = train_head(basis_type, mode, projected[basis_type][mode])

    # Secondary one-shot capacity upper bound.  The compact frozen base already
    # contains prototype diagonals, row masses, and aggregate semantic terms.
    # Keep only exact off-diagonal pair identities here to test whether rank-4
    # compression, rather than pair transferability, is the bottleneck.
    block_offdiag = tri_i != tri_j
    full_offdiag_mask = np.tile(block_offdiag, block_count)
    full_offdiag_sources = {
        "real": pair_real[:, full_offdiag_mask],
        "assignment_shuffled": pair_shuffled[:, full_offdiag_mask],
    }
    for mode in RELATION_MODES:
        key = f"full_offdiag_pair__{mode}"
        results[key] = train_head("full_offdiag_pair", mode, full_offdiag_sources[mode])
    for basis_type in ("semantic_pair", "random_semantic_rank"):
        for mode in RELATION_MODES:
            key = f"{basis_type}__{mode}"
            results[key] = train_head(basis_type, mode, semantic_projected[basis_type][mode])

    def serialize_basis(basis: list[dict[str, Any]]) -> list[dict[str, Any]]:
        rows: list[dict[str, Any]] = []
        for block in basis:
            rows.append(
                {
                    "family": block["family"],
                    "distance": int(block["distance"]),
                    "signs": block["signs"].tolist(),
                    "eigenvalues": block["eigenvalues"].tolist(),
                    "eigenvectors_sha256": sha256(block["eigenvectors"]),
                    "projection_vectors_sha256": sha256(block["projection_vectors"]),
                    **(
                        {"mean_coefficient_matrix_sha256": sha256(block["mean_coefficient_matrix"])}
                        if "mean_coefficient_matrix" in block
                        else {}
                    ),
                }
            )
        return rows

    assignment_summary = {
        family: {
            "n_nodes": int(audit["nodes"]),
            "zero_mass_node_fraction": float(audit["zero_mass"] / max(audit["nodes"], 1)),
            "mean_graph_best_signed_cosine": float(np.mean(audit["best_cosine"])),
            "mean_real_assignment_shuffled_pair_l1": float(np.mean(audit["real_shuffled_l1"])),
        }
        for family, audit in assignment_audit.items()
    }
    base_heldout_auc = float(roc_auc_score(base["heldout"]["labels"], base["heldout"]["scores"]))
    comparisons = {
        "frozen_compact_base_heldout_auc": base_heldout_auc,
        "task_real_gain_over_base": results["task__real"]["heldout"]["auc"] - base_heldout_auc,
        "shuffled_label_real_gain_over_base": results["shuffled_label__real"]["heldout"]["auc"] - base_heldout_auc,
        "random_real_gain_over_base": results["random__real"]["heldout"]["auc"] - base_heldout_auc,
        "task_real_minus_shuffled_label_real": results["task__real"]["heldout"]["auc"] - results["shuffled_label__real"]["heldout"]["auc"],
        "task_real_minus_random_real": results["task__real"]["heldout"]["auc"] - results["random__real"]["heldout"]["auc"],
        "task_real_minus_task_assignment_shuffled": results["task__real"]["heldout"]["auc"] - results["task__assignment_shuffled"]["heldout"]["auc"],
    }

    doc = {
        "protocol_id": "molhiv_nested_lowrank_pairwise_relation_basis_v1",
        "date": "2026-07-28",
        "hypothesis": (
            "Task signal may live in specific prototype-pair relations rather than individual prototype "
            "weights; a nested low-rank pair basis may add stable signal beyond frozen compact relations."
        ),
        "selection_policy": {
            "development_data": "official-train only; three outer scaffold folds on the 8k subset",
            "fixed_vocabulary": "outer-fit-only deterministic farthest plus scaffold-facility prototypes",
            "frozen_base": "uniform-gate real exact-distance 1+2 compact relation predictor",
            "task_basis": "average of three inner-scaffold inner-train coefficient matrices, rank-4 eigen compression",
            "controls": "shuffled-label basis, random orthogonal basis, node-assignment shuffling",
            "secondary_capacity_upper_bound": (
                "all exact off-diagonal pair identities with the same bounded linear sidecar; "
                "diagnostic only and excluded from the promotion gate"
            ),
            "label_free_semantic_pair_screen": (
                "top-8 eigenvectors of the fixed prototype cosine Gram matrix; full symmetric "
                "8x8 projected relation tensor, compared with matched random orthonormal coordinates"
            ),
            "random_projection_robustness": (
                "projection_seed changes only the matched random rank-8 coordinate system; "
                "relation assignments, outer folds, compact base, and task-basis split remain fixed"
            ),
            "fixed_selector_c": args.selector_c,
            "fixed_rank": args.rank,
            "fixed_residual_cap": args.max_relation_residual,
            "official_valid_evaluations": 0,
            "official_test_evaluations": 0,
            "promotion_gate_across_three_outer_folds": {
                "mean_task_gain_over_frozen_compact_base_at_least": 0.003,
                "minimum_task_base_fold_wins": 2,
                "task_mean_must_beat_shuffled_label_mean": True,
                "task_mean_must_beat_random_basis_mean": True,
                "minimum_task_assignment_shuffled_fold_wins": 2,
            },
            "status": "one-shot development diagnostic; no hyperparameter sweep or official split promotion",
        },
        "config": vars(args),
        "fold": args.fold,
        "n_fit": int(len(fit_indices)),
        "n_heldout": int(len(heldout_indices)),
        "n_fit_positive": int(labels[fit_indices].sum()),
        "n_heldout_positive": int(labels[heldout_indices].sum()),
        "fit_indices_sha256": fit_sha,
        "heldout_indices_sha256": sha256(heldout_indices),
        "relation_graph_indices_sha256": sha256(relation_graphs),
        "shuffled_labels_sha256": sha256(shuffled_labels),
        "families": list(FAMILIES),
        "distances": list(DISTANCES),
        "blocks": [{"family": family, "distance": distance} for family, distance in blocks],
        "pair_vectorization": "upper triangle with sqrt(2) off-diagonal scaling",
        "pair_dim_per_block": pair_dim,
        "full_pair_dim": full_pair_dim,
        "projection_dim": block_count * args.rank,
        "full_offdiag_pair_dim": int(full_offdiag_mask.sum()),
        "semantic_rank": args.semantic_rank,
        "semantic_pair_dim_per_block": semantic_block_dim,
        "semantic_pair_projection_dim": block_count * semantic_block_dim,
        "semantic_basis_audit": {
            family: {
                "gram_eigenvalues": semantic_family_eigenvalues[family].tolist(),
                "semantic_vectors_sha256": sha256(semantic_family_vectors[family]),
                "matched_random_vectors_sha256": sha256(
                    matched_random_family_vectors[family]
                ),
            }
            for family in FAMILIES
        },
        "prototype_audit": prototype_audit,
        "assignment_audit": assignment_summary,
        "mean_pair_fraction": {
            str(distance): float(pair_fraction_sum[distance] / len(relation_graphs))
            for distance in DISTANCES
        },
        "pair_real_sha256": sha256(pair_real[relation_graphs]),
        "pair_assignment_shuffled_sha256": sha256(pair_shuffled[relation_graphs]),
        "relation_precompute_sec": precompute_sec,
        "basis_audit": {
            "task": task_basis_audit,
            "shuffled_label": shuffled_basis_audit,
            "random": {"method": "seeded random orthonormal prototype directions with random signs"},
        },
        "bases": {
            "task": serialize_basis(task_basis),
            "shuffled_label": serialize_basis(shuffled_basis),
            "random": serialize_basis(random_basis),
        },
        "projection_sha256": {
            f"{basis_type}__{mode}": sha256(projected[basis_type][mode][relation_graphs])
            for basis_type in BASIS_TYPES
            for mode in RELATION_MODES
        },
        "base_compact": {
            "fit_auc": float(roc_auc_score(base["fit"]["labels"], base["fit"]["scores"])),
            "heldout_auc": base_heldout_auc,
            "fit_score_sha256": sha256(base["fit"]["scores"].astype(np.float64)),
            "heldout_score_sha256": sha256(base["heldout"]["scores"].astype(np.float64)),
        },
        "results": results,
        "comparisons": {
            **comparisons,
            "full_offdiag_real_gain_over_base": (
                results["full_offdiag_pair__real"]["heldout"]["auc"] - base_heldout_auc
            ),
            "full_offdiag_real_minus_assignment_shuffled": (
                results["full_offdiag_pair__real"]["heldout"]["auc"]
                - results["full_offdiag_pair__assignment_shuffled"]["heldout"]["auc"]
            ),
            "semantic_pair_real_gain_over_base": (
                results["semantic_pair__real"]["heldout"]["auc"] - base_heldout_auc
            ),
            "semantic_pair_real_minus_random_semantic_real": (
                results["semantic_pair__real"]["heldout"]["auc"]
                - results["random_semantic_rank__real"]["heldout"]["auc"]
            ),
            "semantic_pair_real_minus_assignment_shuffled": (
                results["semantic_pair__real"]["heldout"]["auc"]
                - results["semantic_pair__assignment_shuffled"]["heldout"]["auc"]
            ),
        },
        "elapsed_sec": time.time() - started,
        "output": args.output,
    }
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(doc, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"output": str(output), "comparisons": doc["comparisons"]}, indent=2), flush=True)


if __name__ == "__main__":
    main()

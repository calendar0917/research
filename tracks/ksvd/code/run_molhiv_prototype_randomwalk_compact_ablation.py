"""Frozen-base compact local-random-walk relation ablation on MolHIV.

The exact-distance compact pilot localized assignment-specific relation signal
to one and two hops.  This follow-up keeps the occurrence logits frozen and
replaces the distance-2 mask by reversible stationary random-walk flows:

    Q_t = diag(pi) @ P**t

For t=2 we compare full flow, return-removed flow, and flow restricted to node
pairs whose exact shortest-path distance is two.  Each operator is compressed
into the same 70 fixed statistics used by the exact-distance pilot.  Real and
deterministically node-assignment-shuffled controls share an identical bounded
linear residual protocol.  Official valid/test remain unencoded and unevaluated.
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

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import load_molhiv

DISTANCE_BINS = ("self", "distance_1", "distance_2", "distance_3plus", "disconnected")
OPERATOR_NAMES = (
    "rw1_stationary",
    "rw2_full",
    "rw2_nonreturn",
    "rw2_exact_distance2",
    "rw1_rw2_nonreturn_mix",
)
ABLATIONS: dict[str, tuple[int, ...]] = {
    "rw2_full": (1,),
    "rw2_nonreturn": (2,),
    "rw2_exact_distance2": (3,),
    "rw1_rw2_full": (0, 1),
    "rw1_rw2_nonreturn": (0, 2),
    "rw1_rw2_exact_distance2": (0, 3),
    "rw1_rw2_nonreturn_mix": (4,),
}
STAT_NAMES = (
    "diag_by_prototype",
    "row_mass_by_prototype",
    "trace",
    "offdiag_mass",
    "semantic_similarity",
    "offdiag_semantic_similarity",
    "pair_fraction",
    "active_assignment_mass",
)


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


def all_pairs_distance_bins(graph) -> np.ndarray:
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
    bins = np.full((n, n), 4, dtype=np.int8)
    bins[distances == 0] = 0
    bins[distances == 1] = 1
    bins[distances == 2] = 2
    bins[distances >= 3] = 3
    return bins


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


def normalize_operator(weight: np.ndarray) -> np.ndarray:
    weight = np.asarray(weight, dtype=np.float64)
    total = float(weight.sum())
    if total <= 1e-15:
        return np.zeros_like(weight, dtype=np.float32)
    weight = weight / total
    weight = 0.5 * (weight + weight.T)
    weight = weight / max(float(weight.sum()), 1e-15)
    return weight.astype(np.float32)


def local_random_walk_operators(graph, distance_bins: np.ndarray) -> list[np.ndarray]:
    """Return reversible local walk flows; every nonzero operator sums to one."""
    n = graph.n
    adjacency = np.zeros((n, n), dtype=np.float64)
    for u in range(n):
        for v in graph.neighbors(u):
            adjacency[u, int(v)] = 1.0
    degree = adjacency.sum(axis=1)
    volume = float(degree.sum())
    if volume <= 0:
        zero = np.zeros((n, n), dtype=np.float32)
        return [zero.copy() for _ in OPERATOR_NAMES]
    transition = np.divide(
        adjacency,
        degree[:, None],
        out=np.zeros_like(adjacency),
        where=degree[:, None] > 0,
    )
    stationary = degree / volume
    rw1 = normalize_operator(stationary[:, None] * transition)
    rw2 = normalize_operator(stationary[:, None] * (transition @ transition))
    rw2_nonreturn_raw = np.asarray(rw2, dtype=np.float64).copy()
    np.fill_diagonal(rw2_nonreturn_raw, 0.0)
    rw2_nonreturn = normalize_operator(rw2_nonreturn_raw)
    rw2_distance2 = normalize_operator(
        np.asarray(rw2, dtype=np.float64) * np.asarray(distance_bins == 2, dtype=np.float64)
    )
    rw_mix = normalize_operator(0.5 * np.asarray(rw1) + 0.5 * np.asarray(rw2_nonreturn))
    return [rw1, rw2, rw2_nonreturn, rw2_distance2, rw_mix]


def compact_operator_feature(
    assignment: np.ndarray,
    operator: np.ndarray,
    prototype_similarity: np.ndarray,
) -> tuple[np.ndarray, float, float]:
    n = len(assignment)
    operator = np.asarray(operator, dtype=np.float32)
    k = assignment.shape[1]
    if float(operator.sum()) > 0:
        matrix = np.asarray(assignment.T @ operator @ assignment, dtype=np.float32)
    else:
        matrix = np.zeros((k, k), dtype=np.float32)
    diag = np.diag(matrix).astype(np.float32)
    row_mass = matrix.sum(axis=1, dtype=np.float32)
    trace = float(diag.sum())
    total = float(matrix.sum())
    semantic = float(np.sum(matrix * prototype_similarity))
    diagonal_semantic = float(np.sum(diag * np.diag(prototype_similarity)))
    support_fraction = float(np.count_nonzero(operator) / max(float(n * n), 1.0))
    scalars = np.asarray([
        trace,
        total - trace,
        semantic,
        semantic - diagonal_semantic,
        support_fraction,
        total,
    ], dtype=np.float32)
    symmetry_error = float(np.max(np.abs(operator - operator.T))) if operator.size else 0.0
    return np.concatenate([diag, row_mass, scalars]).astype(np.float32), float(operator.sum()), symmetry_error


def load_base_raw(prior: dict[str, Any], family: str, split: str) -> dict[str, np.ndarray]:
    prefix = "farthest" if family == "farthest" else "scaffold"
    raw = prior["results"][f"{prefix}_occurrence"][split]
    required = ("graph_indices", "labels", "scores")
    if any(key not in raw for key in required):
        raise ValueError("prior frozen result must contain saved occurrence predictions")
    return {
        "graph_indices": np.asarray(raw["graph_indices"], dtype=np.int64),
        "labels": np.asarray(raw["labels"], dtype=np.float32),
        "scores": np.asarray(raw["scores"], dtype=np.float32),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-cache", required=True)
    ap.add_argument("--fold-cache", required=True)
    ap.add_argument("--vocabulary-result", required=True)
    ap.add_argument("--base-result", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relation-seed", type=int, default=20260728)
    ap.add_argument("--n-prototypes", type=int, default=32)
    ap.add_argument("--sparsity", type=int, default=3)
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
    if args.fold < 0 or args.n_prototypes <= 1 or not (0 < args.sparsity <= args.n_prototypes):
        raise ValueError("invalid fold/prototype configuration")
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

    t0 = time.time()
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
        if "original_indices" in folds.files and not np.array_equal(
            np.asarray(folds["original_indices"], dtype=np.int64), expected_original
        ):
            raise ValueError("fold cache original_indices mismatch")
    if set(np.concatenate([fit_indices, heldout_indices]).tolist()) != set(official_train.tolist()):
        raise AssertionError("outer scaffold fold does not partition official train")
    if np.intersect1d(fit_indices, heldout_indices).size:
        raise AssertionError("outer fit/heldout overlap")

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
    official_nontrain_rows = np.concatenate([
        np.arange(int(offsets[i]), int(offsets[i + 1]), dtype=np.int64)
        for i in np.concatenate([official_valid, official_test])
    ])
    if np.any(latents[official_nontrain_rows] != 0):
        raise AssertionError("official valid/test latent rows are nonzero")

    latent_meta = json.loads(Path(args.latent_cache).with_suffix(".json").read_text())
    if bool(latent_meta.get("official_valid_encoded", True)) or bool(latent_meta.get("official_test_encoded", True)):
        raise ValueError("latent cache encoded official valid/test")

    vocabulary = json.loads(Path(args.vocabulary_result).read_text())
    prior = json.loads(Path(args.base_result).read_text())
    fit_sha = sha256(fit_indices)
    if int(vocabulary.get("fold", -1)) != args.fold or vocabulary.get("fit_indices_sha256") != fit_sha:
        raise ValueError("vocabulary fold/split mismatch")
    if int(prior.get("fold", -1)) != args.fold or prior.get("fit_indices_sha256") != fit_sha:
        raise ValueError("base result fold/split mismatch")
    prior_cfg = prior.get("config", {})
    if int(prior_cfg.get("seed", -1)) != args.seed:
        raise ValueError("base-result seed mismatch")

    fit_row_mask = np.zeros(len(latents), dtype=bool)
    for i in fit_indices:
        fit_row_mask[int(offsets[i]):int(offsets[i + 1])] = True
    families = ("farthest", "scaffold_facility")
    prototypes: dict[str, np.ndarray] = {}
    prototype_rows: dict[str, np.ndarray] = {}
    for family in families:
        result = vocabulary["results"][family]
        rows = np.asarray(result["selection"]["source_node_rows"], dtype=np.int64)
        if rows.shape != (args.n_prototypes,) or not np.all(fit_row_mask[rows]):
            raise ValueError(f"invalid prototype rows for {family}")
        p = normalize_rows(latents[rows])
        expected_hash = result["selection"].get("prototype_sha256")
        if expected_hash is not None and sha256(p) != expected_hash:
            raise ValueError(f"prototype hash mismatch for {family}")
        prototypes[family] = p
        prototype_rows[family] = rows

    per_operator_dim = 2 * args.n_prototypes + 6
    relation_graphs = np.sort(np.concatenate([fit_indices, heldout_indices]))
    compact_real = {
        family: np.zeros((len(bundle.graphs), len(OPERATOR_NAMES), per_operator_dim), dtype=np.float32)
        for family in families
    }
    compact_shuffled = {
        family: np.zeros((len(bundle.graphs), len(OPERATOR_NAMES), per_operator_dim), dtype=np.float32)
        for family in families
    }
    operator_support_sum = np.zeros(len(OPERATOR_NAMES), dtype=np.float64)
    operator_mass_sum = np.zeros(len(OPERATOR_NAMES), dtype=np.float64)
    operator_symmetry_max = np.zeros(len(OPERATOR_NAMES), dtype=np.float64)
    disconnected_graphs = 0
    audit: dict[str, dict[str, Any]] = {
        f: {
            "nodes": 0,
            "zero_mass": 0,
            "best_cosine": [],
            "real_shuffled_l1_by_operator": [[] for _ in OPERATOR_NAMES],
        }
        for f in families
    }
    precompute_start = time.time()
    for graph_count, raw_i in enumerate(relation_graphs, start=1):
        i = int(raw_i)
        lo, hi = int(offsets[i]), int(offsets[i + 1])
        bins = all_pairs_distance_bins(bundle.graphs[i])
        if np.any(bins == 4):
            disconnected_graphs += 1
        operators = local_random_walk_operators(bundle.graphs[i], bins)
        permutation_rng = np.random.default_rng(
            args.relation_seed + 1_000_003 * i + 104729 * (args.fold + 1)
        )
        permutation = permutation_rng.permutation(hi - lo)
        for family_index, family in enumerate(families):
            assignment, cosine = sparse_assignment(latents[lo:hi], prototypes[family], args.sparsity)
            similarity = np.asarray(prototypes[family] @ prototypes[family].T, dtype=np.float32)
            for operator_id, operator in enumerate(operators):
                real, mass, symmetry_error = compact_operator_feature(assignment, operator, similarity)
                shuffled, _, _ = compact_operator_feature(assignment[permutation], operator, similarity)
                compact_real[family][i, operator_id] = real
                compact_shuffled[family][i, operator_id] = shuffled
                audit[family]["real_shuffled_l1_by_operator"][operator_id].append(
                    float(np.mean(np.abs(real - shuffled)))
                )
                if family_index == 0:
                    operator_support_sum[operator_id] += float(np.count_nonzero(operator) / max(float((hi - lo) ** 2), 1.0))
                    operator_mass_sum[operator_id] += mass
                    operator_symmetry_max[operator_id] = max(operator_symmetry_max[operator_id], symmetry_error)
            audit[family]["nodes"] += hi - lo
            audit[family]["zero_mass"] += int(np.sum(assignment.sum(axis=1) == 0))
            audit[family]["best_cosine"].append(float(cosine.max(axis=1).mean()))
        if graph_count % 1000 == 0:
            print(f"compact random-walk precompute graphs={graph_count}/{len(relation_graphs)}", flush=True)
    precompute_sec = time.time() - precompute_start

    base: dict[str, dict[str, dict[str, np.ndarray]]] = {}
    for family in families:
        base[family] = {split: load_base_raw(prior, family, split) for split in ("fit", "heldout")}
        if not np.array_equal(base[family]["fit"]["graph_indices"], fit_indices):
            raise AssertionError(f"{family} fit prediction order mismatch")
        if not np.array_equal(base[family]["heldout"]["graph_indices"], heldout_indices):
            raise AssertionError(f"{family} heldout prediction order mismatch")
        if not np.array_equal(base[family]["fit"]["labels"], labels[fit_indices]):
            raise AssertionError(f"{family} fit labels mismatch")
        if not np.array_equal(base[family]["heldout"]["labels"], labels[heldout_indices]):
            raise AssertionError(f"{family} heldout labels mismatch")

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
        fit_z = np.clip((fit_x - mean) / safe_std, -args.standardization_clip, args.standardization_clip)
        heldout_z = np.clip((heldout_x - mean) / safe_std, -args.standardization_clip, args.standardization_clip)
        return fit_z.astype(np.float32), heldout_z.astype(np.float32), {
            "mean_sha256": sha256(mean),
            "std_sha256": sha256(std),
            "active_dimensions": int(active.sum()),
            "inactive_dimensions": int((~active).sum()),
            "fit_standardized_abs_mean": float(np.mean(np.abs(fit_z))),
        }

    @torch.no_grad()
    def combine(head, z: np.ndarray, base_raw: dict[str, np.ndarray]) -> dict[str, Any]:
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
            metrics.update({
                "graph_indices": base_raw["graph_indices"].tolist(),
                "labels": base_raw["labels"].tolist(),
                "scores": scores.tolist(),
                "base_scores": base_raw["scores"].tolist(),
                "relation_residuals": residual.tolist(),
            })
        return metrics

    def train_head(
        family: str,
        ablation: str,
        mode: str,
        source: np.ndarray,
    ) -> dict[str, Any]:
        fit_z, heldout_z, normalization = standardize(source)
        family_base = base[family]
        seed_everything(args.seed + 41, torch)
        head = BoundedLinearResidual(fit_z.shape[1]).to(device)
        optimizer = torch.optim.AdamW(
            head.parameters(), lr=args.relation_lr, weight_decay=args.relation_weight_decay
        )
        dataset = TensorDataset(
            torch.from_numpy(fit_z),
            torch.from_numpy(family_base["fit"]["scores"]),
            torch.from_numpy(family_base["fit"]["labels"]),
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
        start = time.time()
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
            history.append({
                "epoch": float(epoch),
                "task": total / max(seen, 1),
                "weight_l2": float(head.linear.weight.detach().norm().cpu()),
            })
        print(
            f"{family}_{ablation}_{mode} dim={fit_z.shape[1]} "
            f"task={history[-1]['task']:.5f} w={history[-1]['weight_l2']:.4f}",
            flush=True,
        )
        return {
            "family": family,
            "ablation": ablation,
            "relation_mode": mode,
            "operator_ids": list(ABLATIONS[ablation]),
            "operators": [OPERATOR_NAMES[x] for x in ABLATIONS[ablation]],
            "relation_dim": int(fit_z.shape[1]),
            "fit": combine(head, fit_z, family_base["fit"]),
            "heldout": combine(head, heldout_z, family_base["heldout"]),
            "normalization": normalization,
            "history": history,
            "relation_trainable_parameters": int(sum(p.numel() for p in head.parameters() if p.requires_grad)),
            "relation_weight_l2": float(head.linear.weight.detach().norm().cpu()),
            "relation_weight_sha256": sha256(head.linear.weight.detach().cpu().numpy()),
            "elapsed_sec": time.time() - start,
        }

    results: dict[str, Any] = {}
    for family in families:
        for ablation, operator_ids in ABLATIONS.items():
            for mode, store in (("real", compact_real), ("shuffled", compact_shuffled)):
                selected = store[family][:, operator_ids, :].reshape(len(bundle.graphs), -1)
                key = f"{family}__{ablation}__{mode}"
                results[key] = train_head(family, ablation, mode, selected)

    assignment_audit = {}
    for family, values in audit.items():
        assignment_audit[family] = {
            "n_nodes": int(values["nodes"]),
            "zero_mass_node_fraction": float(values["zero_mass"] / max(values["nodes"], 1)),
            "mean_graph_best_signed_cosine": float(np.mean(values["best_cosine"])),
            "mean_real_shuffled_feature_l1_by_operator": {
                OPERATOR_NAMES[i]: float(np.mean(values["real_shuffled_l1_by_operator"][i]))
                for i in range(len(OPERATOR_NAMES))
            },
            "real_compact_sha256": sha256(compact_real[family][relation_graphs]),
            "shuffled_compact_sha256": sha256(compact_shuffled[family][relation_graphs]),
        }

    doc = {
        "protocol_id": "molhiv_prototype_randomwalk_compact_ablation_v1",
        "date": "2026-07-28",
        "hypothesis": (
            "A reversible local one/two-step random-walk composition may improve on uniform exact-distance "
            "composition while retaining assignment-specific gains over shuffled controls."
        ),
        "selection_policy": {
            "development_data": "official-train only; three outer scaffold folds",
            "official_valid_evaluations": 0,
            "official_test_evaluations": 0,
            "status": "exploratory diagnostic; no official split promotion",
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
        "operator_names": list(OPERATOR_NAMES),
        "operator_definitions": {
            "rw1_stationary": "diag(pi) @ P; identical edge support to exact distance 1",
            "rw2_full": "diag(pi) @ P^2 including returns",
            "rw2_nonreturn": "rw2_full with node-space diagonal removed and renormalized",
            "rw2_exact_distance2": "rw2_full restricted to exact shortest-path distance 2 and renormalized",
            "rw1_rw2_nonreturn_mix": "equal-mass mixture of rw1_stationary and rw2_nonreturn",
        },
        "ablations": {k: {"operator_ids": list(v), "operators": [OPERATOR_NAMES[x] for x in v]} for k, v in ABLATIONS.items()},
        "statistics": list(STAT_NAMES),
        "per_operator_dim": per_operator_dim,
        "operator_support_fraction_mean": (operator_support_sum / len(relation_graphs)).tolist(),
        "operator_mass_mean": (operator_mass_sum / len(relation_graphs)).tolist(),
        "operator_symmetry_max_error": operator_symmetry_max.tolist(),
        "n_disconnected_graphs": disconnected_graphs,
        "relation_precompute_sec": precompute_sec,
        "assignment_audit": assignment_audit,
        "base_occurrence": {
            family: {
                "fit_auc": float(roc_auc_score(base[family]["fit"]["labels"], base[family]["fit"]["scores"])),
                "heldout_auc": float(roc_auc_score(base[family]["heldout"]["labels"], base[family]["heldout"]["scores"])),
                "fit_score_sha256": sha256(base[family]["fit"]["scores"].astype(np.float64)),
                "heldout_score_sha256": sha256(base[family]["heldout"]["scores"].astype(np.float64)),
                "prototype_rows": prototype_rows[family].tolist(),
                "prototype_sha256": sha256(prototypes[family]),
            }
            for family in families
        },
        "results": results,
        "elapsed_sec": time.time() - t0,
        "output": args.output,
    }
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.output).write_text(json.dumps(doc, indent=2), encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "fold": args.fold,
        "n_heads": len(results),
        "operators": list(OPERATOR_NAMES),
        "elapsed_sec": doc["elapsed_sec"],
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
    }, indent=2))


if __name__ == "__main__":
    main()

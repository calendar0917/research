"""Deterministic label-free compression of exact prototype-pair relations.

The fixed broad real-patch vocabularies and frozen compact distance-relation
predictor are reused.  Complete off-diagonal pair identities are compressed
from outer-fit graphs only, without labels, using two deterministic bases:

1. covariance PCA on centered pair frequencies;
2. correlation PCA after feature-wise variance balancing.

The same fit-only preprocessing and basis are applied to real and matched
node-assignment-shuffled relations.  Official valid/test stay unencoded and
unevaluated.
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from collections import deque
from pathlib import Path
from typing import Any

import numpy as np
from scipy.linalg import eigh
from sklearn.metrics import roc_auc_score

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.data_molhiv import load_molhiv
from code.molhiv_evaluation_protocol import (
    evaluation_description, forbidden_encoded_indices, resolve_evaluation_protocol,
    validate_latent_sidecar, validate_upstream_result,
)
from code.run_molhiv_nested_pairwise_relation_basis import (
    load_frozen_compact_base,
    normalize_rows,
    sha256,
    sparse_assignment,
    symmetric_vector,
)

FAMILIES = ("farthest", "scaffold_facility")
DISTANCES = (1, 2)
PCA_TYPES = ("covariance_pca", "correlation_pca")
RELATION_MODES = ("real", "assignment_shuffled")


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


def all_pairs_distances(graph: Any) -> np.ndarray:
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


def canonicalize_columns(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float64).copy()
    for column in range(vectors.shape[1]):
        pivot = int(np.argmax(np.abs(vectors[:, column])))
        if vectors[pivot, column] < 0.0:
            vectors[:, column] *= -1.0
    return vectors.astype(np.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--latent-cache", required=True)
    ap.add_argument("--fold-cache", required=True)
    ap.add_argument("--vocabulary-result", required=True)
    ap.add_argument("--compact-base-result", required=True)
    ap.add_argument("--fold", type=int, required=True)
    ap.add_argument("--evaluation-split", choices=("scaffold_fold", "official_valid"), default="scaffold_fold")
    ap.add_argument("--max-graphs", type=int, default=8000)
    ap.add_argument("--data-seed", type=int, default=0)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--relation-seed", type=int, default=20260728)
    ap.add_argument("--n-prototypes", type=int, default=32)
    ap.add_argument("--sparsity", type=int, default=3)
    ap.add_argument("--pca-rank", type=int, default=144)
    ap.add_argument("--pca-types", default=",".join(PCA_TYPES))
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

    if args.fold < 0:
        raise ValueError("invalid fold")
    pca_types = tuple(x.strip() for x in args.pca_types.split(",") if x.strip())
    if not pca_types or len(set(pca_types)) != len(pca_types) or not set(pca_types).issubset(PCA_TYPES):
        raise ValueError(f"invalid PCA types: {pca_types}")
    if args.evaluation_split == "official_valid" and pca_types != ("covariance_pca",):
        raise ValueError("frozen official-valid protocol permits covariance_pca only")
    if args.n_prototypes <= 1 or not (0 < args.sparsity <= args.n_prototypes):
        raise ValueError("invalid prototype/sparsity configuration")
    full_offdiag_dim = len(FAMILIES) * len(DISTANCES) * (
        args.n_prototypes * (args.n_prototypes - 1) // 2
    )
    if not (0 < args.pca_rank <= full_offdiag_dim):
        raise ValueError("invalid PCA rank")
    if min(args.relation_epochs, args.relation_batch_size) <= 0:
        raise ValueError("invalid relation training configuration")
    if args.relation_lr <= 0 or args.relation_weight_decay < 0:
        raise ValueError("invalid optimizer configuration")
    if args.max_relation_residual <= 0 or args.standardization_clip <= 0:
        raise ValueError("invalid residual/clip configuration")

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
    forbidden = forbidden_encoded_indices(args.evaluation_split, official_valid, official_test)
    forbidden_rows = np.concatenate([
        np.arange(int(offsets[int(i)]), int(offsets[int(i) + 1]), dtype=np.int64) for i in forbidden
    ])
    if np.any(latents[forbidden_rows] != 0):
        raise AssertionError("forbidden official split latent rows are nonzero")

    fit_sha = sha256(fit_indices)
    validate_latent_sidecar(args.latent_cache, args.evaluation_split, fit_sha)
    vocabulary = json.loads(Path(args.vocabulary_result).read_text(encoding="utf-8"))
    compact_base_doc = json.loads(Path(args.compact_base_result).read_text(encoding="utf-8"))
    for name, doc in (("vocabulary", vocabulary), ("compact base", compact_base_doc)):
        validate_upstream_result(doc, name=name, evaluation_split=args.evaluation_split, fold=args.fold, fit_sha=fit_sha)

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
            raise ValueError(f"{family} prototype outside outer-fit")
        p = normalize_rows(latents[rows])
        if selection.get("prototype_sha256") and sha256(p) != selection["prototype_sha256"]:
            raise ValueError(f"{family} prototype hash mismatch")
        prototypes[family] = p
        prototype_audit[family] = {
            "source_node_rows": rows.tolist(),
            "source_graph_indices": graphs.tolist(),
            "prototype_sha256": sha256(p),
        }

    base = {split: load_frozen_compact_base(compact_base_doc, split) for split in ("fit", "heldout")}
    if not np.array_equal(base["fit"]["graph_indices"], fit_indices):
        raise AssertionError("base fit order mismatch")
    if not np.array_equal(base["heldout"]["graph_indices"], heldout_indices):
        raise AssertionError("base heldout order mismatch")

    tri_i, tri_j = np.triu_indices(args.n_prototypes)
    pair_dim = len(tri_i)
    blocks = [(family, distance) for family in FAMILIES for distance in DISTANCES]
    relation_graphs = np.concatenate([fit_indices, heldout_indices])
    pair_real = np.zeros((len(bundle.graphs), len(blocks) * pair_dim), dtype=np.float32)
    pair_shuffled = np.zeros_like(pair_real)
    assignment_audit: dict[str, Any] = {
        family: {"nodes": 0, "zero_mass": 0, "best_cosine": [], "real_shuffled_l1": []}
        for family in FAMILIES
    }
    pair_fraction_sum = {distance: 0.0 for distance in DISTANCES}
    permutation_rng = np.random.default_rng(args.relation_seed + 13007 * (args.fold + 1))
    precompute_started = time.time()
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
            shuffled_assignment = assignment[permutation]
            for distance in DISTANCES:
                mask = np.asarray(distances == distance, dtype=np.float32)
                count = float(mask.sum())
                if count > 0:
                    real_matrix = np.asarray(assignment.T @ mask @ assignment / count, dtype=np.float32)
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

    block_offdiag = tri_i != tri_j
    full_offdiag_mask = np.tile(block_offdiag, len(blocks))
    sources = {
        "real": pair_real[:, full_offdiag_mask],
        "assignment_shuffled": pair_shuffled[:, full_offdiag_mask],
    }

    basis_audit: dict[str, Any] = {}
    projected: dict[str, dict[str, np.ndarray]] = {}
    for pca_type in pca_types:
        fit_x = np.asarray(sources["real"][fit_indices], dtype=np.float64)
        mean = fit_x.mean(axis=0)
        centered = fit_x - mean
        raw_std = centered.std(axis=0)
        active = raw_std > 1e-10
        if pca_type == "covariance_pca":
            scale = np.ones_like(raw_std)
            fit_z = centered
            preprocessing = "center only"
        else:
            scale = np.where(active, raw_std, 1.0)
            fit_z = np.clip(centered / scale, -args.standardization_clip, args.standardization_clip)
            preprocessing = "center and divide by outer-fit real feature std, then clip"
        fit_z[:, ~active] = 0.0
        covariance = np.asarray(fit_z.T @ fit_z / max(len(fit_z) - 1, 1), dtype=np.float64)
        rank = min(args.pca_rank, int(active.sum()))
        eigenvalues, eigenvectors = eigh(
            covariance,
            subset_by_index=(covariance.shape[0] - rank, covariance.shape[0] - 1),
            driver="evr",
            check_finite=False,
        )
        order = np.argsort(eigenvalues)[::-1]
        eigenvalues = np.maximum(eigenvalues[order], 0.0)
        eigenvectors = canonicalize_columns(eigenvectors[:, order])
        total_variance = float(np.trace(covariance))
        basis_audit[pca_type] = {
            "preprocessing": preprocessing,
            "fit_only": True,
            "used_labels": False,
            "rank": int(rank),
            "input_dim": int(covariance.shape[0]),
            "active_input_dimensions": int(active.sum()),
            "top_eigenvalues": eigenvalues.astype(np.float32).tolist(),
            "explained_variance_ratio_sum": float(eigenvalues.sum() / max(total_variance, 1e-20)),
            "mean_sha256": sha256(mean.astype(np.float32)),
            "scale_sha256": sha256(scale.astype(np.float32)),
            "basis_sha256": sha256(eigenvectors),
        }
        projected[pca_type] = {}
        for mode in RELATION_MODES:
            values = np.asarray(sources[mode][relation_graphs], dtype=np.float64)
            transformed = values - mean
            if pca_type == "correlation_pca":
                transformed = np.clip(
                    transformed / scale,
                    -args.standardization_clip,
                    args.standardization_clip,
                )
            transformed[:, ~active] = 0.0
            output = np.zeros((len(bundle.graphs), rank), dtype=np.float32)
            output[relation_graphs] = (transformed @ eigenvectors).astype(np.float32)
            projected[pca_type][mode] = output

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

    def train_head(pca_type: str, mode: str, source: np.ndarray) -> dict[str, Any]:
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
            f"{pca_type}__{mode} dim={fit_z.shape[1]} "
            f"task={history[-1]['task']:.5f} w={history[-1]['weight_l2']:.4f}",
            flush=True,
        )
        return {
            "pca_type": pca_type,
            "relation_mode": mode,
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
    for pca_type in pca_types:
        for mode in RELATION_MODES:
            key = f"{pca_type}__{mode}"
            results[key] = train_head(pca_type, mode, projected[pca_type][mode])

    base_heldout_auc = float(roc_auc_score(base["heldout"]["labels"], base["heldout"]["scores"]))
    comparisons = {
        f"{pca_type}_real_gain_over_base": (
            results[f"{pca_type}__real"]["heldout"]["auc"] - base_heldout_auc
        )
        for pca_type in pca_types
    }
    comparisons.update(
        {
            f"{pca_type}_real_minus_assignment_shuffled": (
                results[f"{pca_type}__real"]["heldout"]["auc"]
                - results[f"{pca_type}__assignment_shuffled"]["heldout"]["auc"]
            )
            for pca_type in pca_types
        }
    )

    assignment_summary = {
        family: {
            "n_nodes": int(audit["nodes"]),
            "zero_mass_node_fraction": float(audit["zero_mass"] / max(audit["nodes"], 1)),
            "mean_graph_best_signed_cosine": float(np.mean(audit["best_cosine"])),
            "mean_real_assignment_shuffled_pair_l1": float(np.mean(audit["real_shuffled_l1"])),
        }
        for family, audit in assignment_audit.items()
    }

    doc = {
        "protocol_id": "molhiv_label_free_pair_pca_v1",
        "date": "2026-07-28",
        "hypothesis": (
            "The weak full-pair capacity signal may survive a deterministic, label-free compression "
            "learned from how concrete prototype-pair frequencies vary across outer-fit molecules."
        ),
        "selection_policy": {
            "development_data": evaluation_description(args.evaluation_split, args.max_graphs),
            "fixed_vocabulary": "outer-fit-only deterministic farthest plus scaffold-facility prototypes",
            "frozen_base": "uniform-gate real exact-distance 1+2 compact relation predictor",
            "compression": f"fit-only real pair data; selected PCA types={list(pca_types)}, rank={args.pca_rank}",
            "controls": "same fit-only PCA transform applied to node-assignment-shuffled pair relations",
            "official_valid_evaluations": protocol.official_valid_evaluations,
            "official_test_evaluations": protocol.official_test_evaluations,
            "promotion_gate_across_three_outer_folds": {
                "mean_real_gain_over_frozen_compact_base_at_least": 0.002,
                "minimum_real_base_fold_wins": 2,
                "real_mean_must_beat_assignment_shuffled_mean": True,
                "minimum_real_assignment_shuffled_fold_wins": 2,
            },
            "status": "one-shot deterministic development diagnostic; no rank or optimizer sweep",
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
        "families": list(FAMILIES),
        "distances": list(DISTANCES),
        "blocks": [{"family": family, "distance": distance} for family, distance in blocks],
        "full_offdiag_pair_dim": int(full_offdiag_mask.sum()),
        "prototype_audit": prototype_audit,
        "assignment_audit": assignment_summary,
        "mean_pair_fraction": {
            str(distance): float(pair_fraction_sum[distance] / len(relation_graphs))
            for distance in DISTANCES
        },
        "pair_real_sha256": sha256(sources["real"][relation_graphs]),
        "pair_assignment_shuffled_sha256": sha256(
            sources["assignment_shuffled"][relation_graphs]
        ),
        "relation_precompute_sec": precompute_sec,
        "basis_audit": basis_audit,
        "projection_sha256": {
            f"{pca_type}__{mode}": sha256(projected[pca_type][mode][relation_graphs])
            for pca_type in pca_types
            for mode in RELATION_MODES
        },
        "base_compact": {
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

"""Held-out masked-patch audit for bag and relation-bound patch representations."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .from_scratch_unplanted_dictionary import deterministic_maximin_initialization
from .imdb_walk_dictionary import encode_with_minimum_sparsity
from .ksvd import ksvd
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import PatchCover, patch_budget
from .overlap_stitching import CoverExample, make_cover_example, stack_cover_examples
from .run_overlap_cover_audit import FAMILIES, generate_graph
from .transition_decoder import fit_ridge_decoder, predict_ridge_decoder


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/from_scratch/patch_relation_representation_audit_20260802.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/from_scratch/PATCH_RELATION_REPRESENTATION_AUDIT_20260802.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_PATCH_RELATION_REPRESENTATION_PROTOCOL_20260802.md"
BRANCHES = ("RAW_BAG", "KSVD_BAG", "KSVD_TRUE_RELATION", "KSVD_SHUFFLED_RELATION")
EPS = 1e-12


def patch_invariant_descriptor(adjacency: np.ndarray) -> np.ndarray:
    """Permutation-invariant 22D target for a fixed-size simple patch."""
    values = np.asarray(adjacency, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] != values.shape[1]:
        raise ValueError("patch adjacency must be square")
    n_nodes = values.shape[0]
    if n_nodes < 3 or not np.array_equal(values, values.T):
        raise ValueError("expected an undirected patch with at least three nodes")
    degrees = np.sort(values.sum(axis=1)) / max(n_nodes - 1, 1)
    eigenvalues = np.sort(np.linalg.eigvalsh(values)) / max(n_nodes - 1, 1)
    edge_count = float(values.sum() / 2.0)
    density = edge_count / max(n_nodes * (n_nodes - 1) / 2.0, 1.0)
    triangles = float(np.trace(values @ values @ values) / 6.0)
    triangle_density = triangles / max(n_nodes * (n_nodes - 1) * (n_nodes - 2) / 6.0, 1.0)
    return np.concatenate([degrees, eigenvalues, [density, triangle_density]])


def all_pairs_shortest_paths(adjacency: np.ndarray) -> np.ndarray:
    values = np.asarray(adjacency)
    n_nodes = values.shape[0]
    distances = np.full((n_nodes, n_nodes), n_nodes + 1, dtype=np.float64)
    for source in range(n_nodes):
        distances[source, source] = 0.0
        queue = [source]
        cursor = 0
        while cursor < len(queue):
            node = queue[cursor]
            cursor += 1
            for raw_neighbor in np.flatnonzero(values[node]):
                neighbor = int(raw_neighbor)
                if distances[source, neighbor] > distances[source, node] + 1.0:
                    distances[source, neighbor] = distances[source, node] + 1.0
                    queue.append(neighbor)
    return distances


def pool_tokens(tokens: np.ndarray) -> np.ndarray:
    values = np.asarray(tokens, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 1:
        raise ValueError("tokens must be a non-empty row matrix")
    return np.concatenate(
        [values.mean(axis=0), values.std(axis=0, ddof=0), values.max(axis=0)]
    )


def _relation_arrays(
    cover: PatchCover,
    distances: np.ndarray,
    target_index: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    target = cover.patches[target_index]
    target_nodes = set(target.node_ids)
    context_indices = np.asarray(
        [index for index in range(len(cover.patches)) if index != target_index],
        dtype=np.int64,
    )
    overlap_weights = []
    center_distances = []
    for index in context_indices:
        context = cover.patches[int(index)]
        overlap_weights.append(
            len(target_nodes & set(context.node_ids)) / len(target.node_ids)
        )
        center_distances.append(float(distances[target.center, context.center]))
    return (
        context_indices,
        np.asarray(overlap_weights, dtype=np.float64),
        np.asarray(center_distances, dtype=np.float64),
    )


def relation_features(
    tokens: np.ndarray,
    cover: PatchCover,
    distances: np.ndarray,
    target_index: int,
    *,
    shuffled: bool,
) -> np.ndarray:
    """Bag plus overlap/distance-weighted token context for one masked patch."""
    values = np.asarray(tokens, dtype=np.float64)
    context_indices, overlap, center_distance = _relation_arrays(
        cover, distances, target_index
    )
    context = values[context_indices].copy()
    if shuffled and context.shape[0] > 1:
        context = np.roll(context, shift=1 + target_index % (context.shape[0] - 1), axis=0)
    distance_weight = np.exp(-center_distance)

    def weighted_mean(weights: np.ndarray) -> np.ndarray:
        if float(weights.sum()) <= EPS:
            return context.mean(axis=0)
        return np.sum(context * weights[:, None], axis=0) / weights.sum()

    relation_summary = np.asarray(
        [
            float(overlap.mean()),
            float(overlap.max()),
            float(center_distance.min()),
            float(center_distance.mean()),
        ],
        dtype=np.float64,
    )
    return np.concatenate(
        [
            pool_tokens(context),
            weighted_mean(overlap),
            weighted_mean(distance_weight),
            relation_summary,
        ]
    )


def bag_features(tokens: np.ndarray, target_index: int) -> np.ndarray:
    values = np.asarray(tokens, dtype=np.float64)
    context = np.delete(values, target_index, axis=0)
    return pool_tokens(context)


def graph_embedding(
    branch: str,
    raw_tokens: np.ndarray,
    code_tokens: np.ndarray,
    cover: PatchCover,
    adjacency: np.ndarray,
) -> np.ndarray:
    if branch == "RAW_BAG":
        return pool_tokens(raw_tokens)
    if branch == "KSVD_BAG":
        return pool_tokens(code_tokens)
    if branch not in ("KSVD_TRUE_RELATION", "KSVD_SHUFFLED_RELATION"):
        raise ValueError(f"unknown branch: {branch}")
    distances = all_pairs_shortest_paths(adjacency)
    rows = []
    shuffled = branch == "KSVD_SHUFFLED_RELATION"
    for target_index, token in enumerate(code_tokens):
        context = relation_features(
            code_tokens,
            cover,
            distances,
            target_index,
            shuffled=shuffled,
        )
        rows.append(np.concatenate([token, context]))
    return pool_tokens(np.stack(rows, axis=0))


def build_masked_matrices(
    examples: Sequence[CoverExample],
    codes_by_graph: dict[int, np.ndarray],
) -> tuple[dict[str, np.ndarray], np.ndarray, np.ndarray]:
    feature_rows = {branch: [] for branch in BRANCHES}
    targets = []
    graph_indices = []
    for example in examples:
        raw_tokens = np.asarray(example.patch_vectors, dtype=np.float64)
        code_tokens = np.asarray(codes_by_graph[example.graph_index], dtype=np.float64)
        if code_tokens.shape[0] != raw_tokens.shape[0]:
            raise ValueError("code and raw patch counts disagree")
        distances = all_pairs_shortest_paths(example.adjacency)
        for target_index, patch in enumerate(example.cover.patches):
            feature_rows["RAW_BAG"].append(bag_features(raw_tokens, target_index))
            feature_rows["KSVD_BAG"].append(bag_features(code_tokens, target_index))
            feature_rows["KSVD_TRUE_RELATION"].append(
                relation_features(
                    code_tokens,
                    example.cover,
                    distances,
                    target_index,
                    shuffled=False,
                )
            )
            feature_rows["KSVD_SHUFFLED_RELATION"].append(
                relation_features(
                    code_tokens,
                    example.cover,
                    distances,
                    target_index,
                    shuffled=True,
                )
            )
            targets.append(patch_invariant_descriptor(patch.adjacency))
            graph_indices.append(example.graph_index)
    return (
        {branch: np.stack(rows, axis=0) for branch, rows in feature_rows.items()},
        np.stack(targets, axis=0),
        np.asarray(graph_indices, dtype=np.int64),
    )


def _metric_blocks(targets: np.ndarray, predictions: np.ndarray) -> dict[str, float]:
    residual = np.asarray(predictions) - np.asarray(targets)
    if residual.shape != targets.shape or targets.shape[1] != 22:
        raise ValueError("masked target/prediction shape mismatch")

    def rmse(block: np.ndarray) -> float:
        return float(np.sqrt(np.mean(block**2)))

    return {
        "overall_rmse": rmse(residual),
        "degree_rmse": rmse(residual[:, :10]),
        "spectrum_rmse": rmse(residual[:, 10:20]),
        "density_triangle_rmse": rmse(residual[:, 20:]),
    }


def graph_balanced_metrics(
    targets: np.ndarray,
    predictions: np.ndarray,
    graph_indices: np.ndarray,
) -> tuple[list[dict[str, Any]], dict[str, float]]:
    rows = []
    for graph_index in sorted(set(int(value) for value in graph_indices)):
        mask = graph_indices == graph_index
        rows.append(
            {
                "graph_index": graph_index,
                **_metric_blocks(targets[mask], predictions[mask]),
            }
        )
    keys = ("overall_rmse", "degree_rmse", "spectrum_rmse", "density_triangle_rmse")
    return rows, {
        key: float(np.mean([float(row[key]) for row in rows])) for key in keys
    }


def _cosine(left: np.ndarray, right: np.ndarray) -> float:
    denominator = float(np.linalg.norm(left) * np.linalg.norm(right))
    if denominator <= EPS:
        return float(np.linalg.norm(left - right) <= EPS)
    return float(np.dot(left, right) / denominator)


def _relative_l2(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.linalg.norm(left - right) / max(np.linalg.norm(left), EPS))


def classify(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    folds = tuple(folds)
    if len(folds) != 3:
        raise ValueError("expected three folds")
    mean_branches = {}
    for branch in BRANCHES:
        metric_keys = folds[0]["branches"][branch]["summary"].keys()
        mean_branches[branch] = {
            key: float(
                np.mean([fold["branches"][branch]["summary"][key] for fold in folds])
            )
            for key in metric_keys
        }
    bag = mean_branches["KSVD_BAG"]["overall_rmse"]
    true = mean_branches["KSVD_TRUE_RELATION"]["overall_rmse"]
    shuffled = mean_branches["KSVD_SHUFFLED_RELATION"]["overall_rmse"]
    bag_gain = float((bag - true) / max(bag, EPS))
    shuffled_gain = float((shuffled - true) / max(shuffled, EPS))
    true_better_folds = [
        fold["branches"]["KSVD_TRUE_RELATION"]["summary"]["overall_rmse"]
        < fold["branches"]["KSVD_BAG"]["summary"]["overall_rmse"]
        and fold["branches"]["KSVD_TRUE_RELATION"]["summary"]["overall_rmse"]
        < fold["branches"]["KSVD_SHUFFLED_RELATION"]["summary"]["overall_rmse"]
        for fold in folds
    ]
    invariant_gate = all(bool(fold["invariants"]["passed"]) for fold in folds)
    checks = {
        "true_vs_bag_gain_at_least_002": bag_gain >= 0.02,
        "true_vs_shuffled_gain_at_least_002": shuffled_gain >= 0.02,
        "true_better_both_all_folds": all(true_better_folds),
        "invariants": invariant_gate,
    }
    if all(checks.values()):
        label = "PATCH_RELATIONS_ADD_MASKED_VALUE"
    elif bag_gain > 0.0 and shuffled_gain > 0.0 and invariant_gate:
        label = "PATCH_RELATION_SIGNAL_BELOW_GATE"
    else:
        label = "REJECT_CURRENT_RELATION_FEATURES"
    stability = {}
    for branch in BRANCHES:
        stability[branch] = {
            "cosine_similarity": float(
                np.mean([fold["stability"][branch]["cosine_similarity"] for fold in folds])
            ),
            "relative_l2_difference": float(
                np.mean(
                    [fold["stability"][branch]["relative_l2_difference"] for fold in folds]
                )
            ),
        }
        stability[branch]["cosine_at_least_090"] = (
            stability[branch]["cosine_similarity"] >= 0.90
        )
    return {
        "classification": label,
        "checks": checks,
        "true_vs_bag_rmse_reduction": bag_gain,
        "true_vs_shuffled_rmse_reduction": shuffled_gain,
        "true_better_both_by_fold": true_better_folds,
        "mean_branches": mean_branches,
        "stability": stability,
    }


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Patch relations：masked structural representation 审计",
        "",
        "> 日期：2026-08-02  ",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{decision['classification']}`",
        "",
        "## 1. Held-out masked-patch prediction",
        "",
        "| branch | overall RMSE | degree | spectrum | density/triangle |",
        "|---|---:|---:|---:|---:|",
    ]
    for branch in BRANCHES:
        summary = decision["mean_branches"][branch]
        lines.append(
            f"| {branch} | {summary['overall_rmse']:.5f} | "
            f"{summary['degree_rmse']:.5f} | {summary['spectrum_rmse']:.5f} | "
            f"{summary['density_triangle_rmse']:.5f} |"
        )
    lines.extend(
        [
            "",
            f"- TRUE vs KSVD_BAG RMSE reduction：`{decision['true_vs_bag_rmse_reduction']:.4f}`；",
            f"- TRUE vs SHUFFLED RMSE reduction：`{decision['true_vs_shuffled_rmse_reduction']:.4f}`；",
            f"- TRUE simultaneously better by fold：`{decision['true_better_both_by_fold']}`；",
            f"- registered checks：`{decision['checks']}`。",
            "",
            "## 2. Node-relabel + resampling stability",
            "",
            "| branch | cosine | relative L2 | cosine >= 0.90 |",
            "|---|---:|---:|---:|",
        ]
    )
    for branch in BRANCHES:
        values = decision["stability"][branch]
        lines.append(
            f"| {branch} | {values['cosine_similarity']:.4f} | "
            f"{values['relative_l2_difference']:.4f} | "
            f"{values['cosine_at_least_090']} |"
        )
    lines.extend(
        [
            "",
            "## 3. 解释",
            "",
            "- RAW_BAG vs KSVD_BAG：回答 sparse token 相对 raw context 的信息损失；",
            "- TRUE vs BAG：回答 relation-weighted context 是否增加信息；",
            "- TRUE vs SHUFFLED：排除只靠 feature dimension、relation marginals 或 graph-level statistics；",
            "- stability 同时包含 sampler、slot ordering 和 token 的变化，不要求 exact chain replay。",
            "",
            "本轮是低容量机制审计，不是最终 Transformer，也不使用 graph labels。",
            "",
        ]
    )
    return "\n".join(lines)


def _encode_examples(
    examples: Sequence[CoverExample],
    dictionary: np.ndarray,
    train_mean: np.ndarray,
    *,
    sparsity: int,
) -> dict[int, np.ndarray]:
    result = {}
    for example in examples:
        centered = example.patch_vectors.T - train_mean
        codes = encode_with_minimum_sparsity(
            centered,
            dictionary,
            sparsity=sparsity,
            minimum_sparsity=1,
        )
        result[example.graph_index] = codes.T
    return result


def _stability_summary(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    return {
        branch: {
            "cosine_similarity": float(
                np.mean([row[branch]["cosine_similarity"] for row in rows])
            ),
            "relative_l2_difference": float(
                np.mean([row[branch]["relative_l2_difference"] for row in rows])
            ),
        }
        for branch in BRANCHES
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bank-seed", type=int, default=810001)
    parser.add_argument("--cover-seed", type=int, default=930101)
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--degrees", type=int, nargs="+", default=[15, 20, 25])
    parser.add_argument("--graphs-per-cell", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--target-overlap", type=int, default=3)
    parser.add_argument("--multiplier", type=float, default=1.5)
    parser.add_argument("--retained-beam", type=int, default=8)
    parser.add_argument("--candidate-restarts", type=int, default=1)
    parser.add_argument("--n-atoms", type=int, default=24)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--ridge-alpha", type=float, default=0.01)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    graph_count = len(FAMILIES) * len(args.degrees) * args.graphs_per_cell
    sequences = np.random.SeedSequence(args.graph_bank_seed).spawn(graph_count)
    examples = []
    sampler_seeds = {}
    graph_index = 0
    for family in FAMILIES:
        for degree in args.degrees:
            for _replicate in range(args.graphs_per_cell):
                graph_seed = int(sequences[graph_index].generate_state(1, dtype=np.uint32)[0])
                adjacency = generate_graph(family, args.n_nodes, degree, graph_seed)
                budget = patch_budget(
                    adjacency,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                    edge_capacity_multiplier=args.multiplier,
                )
                sampler_seed = int(
                    np.random.SeedSequence([args.cover_seed, graph_index]).generate_state(
                        1, dtype=np.uint32
                    )[0]
                )
                sampler_seeds[graph_index] = sampler_seed
                cover = sample_marginal_candidate_cover(
                    adjacency,
                    np.random.default_rng(sampler_seed),
                    n_patches=budget,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                    retained_beam=args.retained_beam,
                    candidate_restarts=args.candidate_restarts,
                )
                examples.append(
                    make_cover_example(graph_index, family, degree, adjacency, cover)
                )
                graph_index += 1

    folds = []
    for fold_index in range(3):
        test_examples = tuple(
            example for example in examples if example.graph_index % 8 % 3 == fold_index
        )
        test_ids = {example.graph_index for example in test_examples}
        train_examples = tuple(
            example for example in examples if example.graph_index not in test_ids
        )
        raw_train = stack_cover_examples(train_examples)
        train_mean = np.mean(raw_train, axis=1, keepdims=True)
        centered_train = raw_train - train_mean
        initial_dictionary, initialization = deterministic_maximin_initialization(
            centered_train, args.n_atoms
        )
        dictionary, _training_codes, training_info = ksvd(
            centered_train,
            n_atoms=args.n_atoms,
            T=args.sparsity,
            T_min=1,
            n_iter=args.iterations,
            seed=0,
            initial_dictionary=initial_dictionary,
        )
        train_codes = _encode_examples(
            train_examples, dictionary, train_mean, sparsity=args.sparsity
        )
        test_codes = _encode_examples(
            test_examples, dictionary, train_mean, sparsity=args.sparsity
        )
        train_features, train_targets, train_graph_rows = build_masked_matrices(
            train_examples, train_codes
        )
        test_features, test_targets, test_graph_rows = build_masked_matrices(
            test_examples, test_codes
        )
        branch_payload = {}
        for branch in BRANCHES:
            model = fit_ridge_decoder(
                train_features[branch], train_targets, alpha=args.ridge_alpha
            )
            predictions = predict_ridge_decoder(model, test_features[branch])
            graph_rows, summary = graph_balanced_metrics(
                test_targets, predictions, test_graph_rows
            )
            branch_payload[branch] = {"summary": summary, "graphs": graph_rows}

        stability_rows = []
        for example in test_examples:
            permutation = np.random.default_rng(
                np.random.SeedSequence(
                    [args.graph_bank_seed, example.graph_index, 8801]
                ).generate_state(1, dtype=np.uint32)[0]
            ).permutation(args.n_nodes)
            relabeled_adjacency = example.adjacency[np.ix_(permutation, permutation)]
            relabeled_cover = sample_marginal_candidate_cover(
                relabeled_adjacency,
                np.random.default_rng(sampler_seeds[example.graph_index]),
                n_patches=len(example.cover.patches),
                patch_size=args.patch_size,
                target_overlap=args.target_overlap,
                retained_beam=args.retained_beam,
                candidate_restarts=args.candidate_restarts,
            )
            relabeled_example = make_cover_example(
                example.graph_index,
                example.family,
                example.target_degree,
                relabeled_adjacency,
                relabeled_cover,
            )
            relabeled_codes = _encode_examples(
                [relabeled_example], dictionary, train_mean, sparsity=args.sparsity
            )[example.graph_index]
            graph_row = {"graph_index": example.graph_index}
            for branch in BRANCHES:
                original_embedding = graph_embedding(
                    branch,
                    example.patch_vectors,
                    test_codes[example.graph_index],
                    example.cover,
                    example.adjacency,
                )
                relabeled_embedding = graph_embedding(
                    branch,
                    relabeled_example.patch_vectors,
                    relabeled_codes,
                    relabeled_cover,
                    relabeled_adjacency,
                )
                graph_row[branch] = {
                    "cosine_similarity": _cosine(
                        original_embedding, relabeled_embedding
                    ),
                    "relative_l2_difference": _relative_l2(
                        original_embedding, relabeled_embedding
                    ),
                }
            stability_rows.append(graph_row)
        stability = _stability_summary(stability_rows)
        expected_train_rows = sum(len(example.cover.patches) for example in train_examples)
        expected_test_rows = sum(len(example.cover.patches) for example in test_examples)
        finite = all(
            np.all(np.isfinite(matrix)) for matrix in train_features.values()
        ) and all(np.all(np.isfinite(matrix)) for matrix in test_features.values())
        invariants = {
            "train_test_graph_isolation": not (
                {example.graph_index for example in train_examples} & test_ids
            ),
            "train_row_count": int(train_targets.shape[0]) == expected_train_rows,
            "test_row_count": int(test_targets.shape[0]) == expected_test_rows,
            "finite": bool(finite),
            "target_dimension_22": train_targets.shape[1] == 22
            and test_targets.shape[1] == 22,
        }
        invariants["passed"] = all(invariants.values())
        folds.append(
            {
                "fold_index": fold_index,
                "train_graph_count": len(train_examples),
                "test_graph_count": len(test_examples),
                "train_masked_row_count": int(train_targets.shape[0]),
                "test_masked_row_count": int(test_targets.shape[0]),
                "initialization": initialization,
                "training_reconstruction_curve": [
                    float(value) for value in training_info["recon_curve"]
                ],
                "branches": branch_payload,
                "stability": stability,
                "stability_graphs": stability_rows,
                "invariants": invariants,
            }
        )
        print(
            f"fold={fold_index} bag/true/shuffled="
            f"{branch_payload['KSVD_BAG']['summary']['overall_rmse']:.5f}/"
            f"{branch_payload['KSVD_TRUE_RELATION']['summary']['overall_rmse']:.5f}/"
            f"{branch_payload['KSVD_SHUFFLED_RELATION']['summary']['overall_rmse']:.5f}",
            flush=True,
        )

    decision = classify(folds)
    payload = {
        "protocol": PROTOCOL,
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seed": args.cover_seed,
            "families": list(FAMILIES),
            "degrees": args.degrees,
            "graphs_per_cell": args.graphs_per_cell,
            "graph_count": graph_count,
            "n_nodes": args.n_nodes,
            "patch_size": args.patch_size,
            "target_overlap": args.target_overlap,
            "multiplier": args.multiplier,
            "retained_beam": args.retained_beam,
            "candidate_restarts": args.candidate_restarts,
            "n_atoms": args.n_atoms,
            "sparsity": args.sparsity,
            "iterations": args.iterations,
            "ridge_alpha": args.ridge_alpha,
            "labels_used": False,
        },
        "folds": folds,
        "decision": decision,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(f"decision={decision['classification']}")
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

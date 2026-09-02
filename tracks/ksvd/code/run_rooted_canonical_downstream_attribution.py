"""Compare direct graph statistics with rooted-canonical patch/KSVD readouts."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .canonical_slots import reorder_cover_structurally
from .from_scratch_unplanted_dictionary import deterministic_maximin_initialization
from .imdb_walk_dictionary import encode_with_minimum_sparsity
from .ksvd import ksvd
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import patch_budget
from .overlap_stitching import CoverExample, make_cover_example, stack_cover_examples
from .run_overlap_cover_audit import FAMILIES, generate_graph
from .run_patch_relation_representation_audit import (
    EPS,
    _cosine,
    _relative_l2,
    all_pairs_shortest_paths,
    pool_tokens,
    relation_features,
)
from .transition_decoder import fit_ridge_decoder, predict_ridge_decoder


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/from_scratch/rooted_canonical_downstream_attribution_20260802.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/from_scratch/ROOTED_CANONICAL_DOWNSTREAM_ATTRIBUTION_20260802.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_ROOTED_CANONICAL_DOWNSTREAM_ATTRIBUTION_PROTOCOL_20260802.md"
BRANCHES = (
    "GLOBAL_STATS",
    "ROOTED_RAW_BAG",
    "ROOTED_RAW_TRUE_RELATION",
    "ROOTED_KSVD_BAG",
    "ROOTED_KSVD_TRUE_RELATION",
    "ROOTED_KSVD_SHUFFLED_RELATION",
)


def global_statistics(adjacency: np.ndarray) -> np.ndarray:
    values = np.asarray(adjacency, dtype=np.float64)
    n_nodes = values.shape[0]
    degrees = np.sort(values.sum(axis=1)) / max(n_nodes - 1, 1)
    spectrum = np.sort(np.linalg.eigvalsh(values)) / max(n_nodes - 1, 1)
    density = float(values.sum() / (n_nodes * max(n_nodes - 1, 1)))
    triangles = float(np.trace(values @ values @ values) / 6.0)
    triangle_density = triangles / max(
        n_nodes * (n_nodes - 1) * (n_nodes - 2) / 6.0,
        1.0,
    )
    return np.concatenate([degrees, spectrum, [density, triangle_density]])


def relation_graph_embedding(
    tokens: np.ndarray,
    example: CoverExample,
    *,
    shuffled: bool,
) -> np.ndarray:
    distances = all_pairs_shortest_paths(example.adjacency)
    rows = []
    for target_index, token in enumerate(tokens):
        context = relation_features(
            tokens,
            example.cover,
            distances,
            target_index,
            shuffled=shuffled,
        )
        rows.append(np.concatenate([token, context]))
    return pool_tokens(np.stack(rows, axis=0))


def graph_features(
    branch: str,
    example: CoverExample,
    codes: np.ndarray,
) -> np.ndarray:
    raw = np.asarray(example.patch_vectors, dtype=np.float64)
    if branch == "GLOBAL_STATS":
        return global_statistics(example.adjacency)
    if branch == "ROOTED_RAW_BAG":
        return pool_tokens(raw)
    if branch == "ROOTED_RAW_TRUE_RELATION":
        return relation_graph_embedding(raw, example, shuffled=False)
    if branch == "ROOTED_KSVD_BAG":
        return pool_tokens(codes)
    if branch == "ROOTED_KSVD_TRUE_RELATION":
        return relation_graph_embedding(codes, example, shuffled=False)
    if branch == "ROOTED_KSVD_SHUFFLED_RELATION":
        return relation_graph_embedding(codes, example, shuffled=True)
    raise ValueError(f"unknown branch: {branch}")


def _fit_projection(train: np.ndarray, rank: int) -> dict[str, np.ndarray]:
    mean = np.mean(train, axis=0, keepdims=True)
    scale = np.std(train, axis=0, ddof=0, keepdims=True)
    scale[scale < 1e-10] = 1.0
    standardized = (train - mean) / scale
    _left, _singular, right = np.linalg.svd(standardized, full_matrices=False)
    actual_rank = min(rank, right.shape[0], right.shape[1])
    return {"mean": mean, "scale": scale, "basis": right[:actual_rank].T}


def _project(values: np.ndarray, projection: dict[str, np.ndarray]) -> np.ndarray:
    return ((values - projection["mean"]) / projection["scale"]) @ projection["basis"]


def _balanced_accuracy(targets: np.ndarray, predictions: np.ndarray) -> float:
    classes = sorted(set(int(value) for value in targets))
    return float(
        np.mean(
            [
                np.mean(predictions[targets == label] == label)
                for label in classes
            ]
        )
    )


def _one_hot(targets: np.ndarray, n_classes: int) -> np.ndarray:
    values = np.zeros((targets.shape[0], n_classes), dtype=np.float64)
    values[np.arange(targets.shape[0]), targets] = 1.0
    return values


def _classify(
    train_features: np.ndarray,
    test_features: np.ndarray,
    train_targets: np.ndarray,
    test_targets: np.ndarray,
    *,
    n_classes: int,
    alpha: float,
) -> tuple[float, list[int]]:
    model = fit_ridge_decoder(
        train_features,
        _one_hot(train_targets, n_classes),
        alpha=alpha,
    )
    scores = predict_ridge_decoder(model, test_features)
    predictions = np.argmax(scores, axis=1).astype(np.int64)
    return _balanced_accuracy(test_targets, predictions), predictions.tolist()


def classify_folds(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    folds = tuple(folds)
    means = {
        branch: {
            task: float(
                np.mean([fold["branches"][branch][task] for fold in folds])
            )
            for task in ("joint_balanced_accuracy", "family_balanced_accuracy", "degree_balanced_accuracy")
        }
        for branch in BRANCHES
    }
    joint = {branch: means[branch]["joint_balanced_accuracy"] for branch in BRANCHES}
    true = joint["ROOTED_KSVD_TRUE_RELATION"]
    stats = joint["GLOBAL_STATS"]
    bag = joint["ROOTED_KSVD_BAG"]
    shuffled = joint["ROOTED_KSVD_SHUFFLED_RELATION"]
    raw_true = joint["ROOTED_RAW_TRUE_RELATION"]
    relation_fold_wins = [
        fold["branches"]["ROOTED_KSVD_TRUE_RELATION"]["joint_balanced_accuracy"]
        > fold["branches"]["ROOTED_KSVD_BAG"]["joint_balanced_accuracy"]
        and fold["branches"]["ROOTED_KSVD_TRUE_RELATION"]["joint_balanced_accuracy"]
        > fold["branches"]["ROOTED_KSVD_SHUFFLED_RELATION"]["joint_balanced_accuracy"]
        for fold in folds
    ]
    checks = {
        "ksvd_true_vs_stats_at_least_002": true >= stats + 0.02,
        "ksvd_true_vs_bag_at_least_002": true >= bag + 0.02,
        "ksvd_true_vs_shuffled_at_least_002": true >= shuffled + 0.02,
        "ksvd_true_better_bag_shuffled_all_folds": all(relation_fold_wins),
        "all_invariants": all(bool(fold["invariants"]["passed"]) for fold in folds),
    }
    patch_best = max(joint[branch] for branch in BRANCHES if branch != "GLOBAL_STATS")
    if all(checks.values()):
        label = "ROOTED_KSVD_RELATIONS_ADD_DOWNSTREAM_VALUE"
    elif stats >= patch_best - 0.02:
        label = "DIRECT_STATS_SUFFICIENT_FOR_SYNTHETIC_FACTORS"
    elif true >= bag + 0.02 and true >= shuffled + 0.02 and raw_true >= true - 0.02:
        label = "RELATIONS_USEFUL_KSVD_NOT_NEEDED_FOR_READOUT"
    else:
        label = "NO_CLEAR_DOWNSTREAM_ATTRIBUTION"
    stability = {
        branch: {
            "cosine_similarity": float(
                np.mean([fold["stability"][branch]["cosine_similarity"] for fold in folds])
            ),
            "relative_l2_difference": float(
                np.mean([fold["stability"][branch]["relative_l2_difference"] for fold in folds])
            ),
        }
        for branch in BRANCHES
    }
    return {
        "classification": label,
        "checks": checks,
        "means": means,
        "relation_fold_wins": relation_fold_wins,
        "joint_deltas": {
            "ksvd_true_minus_global_stats": true - stats,
            "ksvd_true_minus_ksvd_bag": true - bag,
            "ksvd_true_minus_ksvd_shuffled": true - shuffled,
            "ksvd_true_minus_raw_true": true - raw_true,
        },
        "stability": stability,
    }


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Rooted-canonical KSVD：直接统计 vs patch readout",
        "",
        "> 日期：2026-08-02  ",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{decision['classification']}`",
        "",
        "所有 branches 均使用 train-only standardization + 12D PCA + fixed ridge。",
        "",
        "| branch | joint-9 balanced acc | family acc | degree acc | relabel-resample cosine |",
        "|---|---:|---:|---:|---:|",
    ]
    for branch in BRANCHES:
        values = decision["means"][branch]
        stability = decision["stability"][branch]["cosine_similarity"]
        lines.append(
            f"| {branch} | {values['joint_balanced_accuracy']:.4f} | "
            f"{values['family_balanced_accuracy']:.4f} | "
            f"{values['degree_balanced_accuracy']:.4f} | {stability:.4f} |"
        )
    lines.extend(
        [
            "",
            f"Joint deltas：`{decision['joint_deltas']}`。  ",
            f"TRUE relation fold wins：`{decision['relation_fold_wins']}`。  ",
            f"Registered checks：`{decision['checks']}`。",
            "",
            "解释：GLOBAL_STATS 回答这些 synthetic generation factors 是否已由宏观统计充分决定；RAW/KSVD BAG 比较局部token统计；TRUE/SHUFFLED 比较正确 patch binding；RAW TRUE/KSVD TRUE 比较 dictionary 是否在同一关系机制下增加可线性读取信号。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bank-seed", type=int, default=810001)
    parser.add_argument("--cover-seed", type=int, default=940101)
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
    parser.add_argument("--projection-rank", type=int, default=12)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    graph_count = len(FAMILIES) * len(args.degrees) * args.graphs_per_cell
    sequences = np.random.SeedSequence(args.graph_bank_seed).spawn(graph_count)
    examples = []
    relabeled_examples = {}
    family_targets = []
    degree_targets = []
    sampler_seeds = {}
    graph_index = 0
    degree_to_index = {degree: index for index, degree in enumerate(args.degrees)}
    for family_index, family in enumerate(FAMILIES):
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
                rooted_cover, _stats = reorder_cover_structurally(
                    adjacency, cover, "rooted_canonical"
                )
                examples.append(
                    make_cover_example(graph_index, family, degree, adjacency, rooted_cover)
                )
                family_targets.append(family_index)
                degree_targets.append(degree_to_index[degree])

                permutation = np.random.default_rng(
                    np.random.SeedSequence(
                        [args.graph_bank_seed, graph_index, 8801]
                    ).generate_state(1, dtype=np.uint32)[0]
                ).permutation(args.n_nodes)
                relabeled_adjacency = adjacency[np.ix_(permutation, permutation)]
                relabeled_cover = sample_marginal_candidate_cover(
                    relabeled_adjacency,
                    np.random.default_rng(sampler_seed),
                    n_patches=budget,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                    retained_beam=args.retained_beam,
                    candidate_restarts=args.candidate_restarts,
                )
                rooted_relabeled, _ = reorder_cover_structurally(
                    relabeled_adjacency, relabeled_cover, "rooted_canonical"
                )
                relabeled_examples[graph_index] = make_cover_example(
                    graph_index,
                    family,
                    degree,
                    relabeled_adjacency,
                    rooted_relabeled,
                )
                graph_index += 1

    family_targets_array = np.asarray(family_targets, dtype=np.int64)
    degree_targets_array = np.asarray(degree_targets, dtype=np.int64)
    joint_targets_array = family_targets_array * len(args.degrees) + degree_targets_array
    folds = []
    for fold_index in range(3):
        test_positions = np.asarray(
            [index for index, example in enumerate(examples) if example.graph_index % 8 % 3 == fold_index],
            dtype=np.int64,
        )
        train_positions = np.asarray(
            [index for index in range(len(examples)) if index not in set(test_positions.tolist())],
            dtype=np.int64,
        )
        train_examples = tuple(examples[index] for index in train_positions)
        test_examples = tuple(examples[index] for index in test_positions)
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

        def encode(example: CoverExample) -> np.ndarray:
            return encode_with_minimum_sparsity(
                example.patch_vectors.T - train_mean,
                dictionary,
                sparsity=args.sparsity,
                minimum_sparsity=1,
            ).T

        all_original_features = {branch: [] for branch in BRANCHES}
        all_relabeled_features = {branch: [] for branch in BRANCHES}
        for example in examples:
            codes = encode(example)
            relabeled = relabeled_examples[example.graph_index]
            relabeled_codes = encode(relabeled)
            for branch in BRANCHES:
                all_original_features[branch].append(
                    graph_features(branch, example, codes)
                )
                all_relabeled_features[branch].append(
                    graph_features(branch, relabeled, relabeled_codes)
                )

        branch_payload = {}
        stability = {}
        for branch in BRANCHES:
            original_matrix = np.stack(all_original_features[branch], axis=0)
            relabeled_matrix = np.stack(all_relabeled_features[branch], axis=0)
            projection = _fit_projection(
                original_matrix[train_positions], args.projection_rank
            )
            projected_original = _project(original_matrix, projection)
            projected_relabeled = _project(relabeled_matrix, projection)
            train_features = projected_original[train_positions]
            test_features = projected_original[test_positions]
            metrics = {}
            predictions = {}
            for task, targets, n_classes in (
                ("joint", joint_targets_array, len(FAMILIES) * len(args.degrees)),
                ("family", family_targets_array, len(FAMILIES)),
                ("degree", degree_targets_array, len(args.degrees)),
            ):
                accuracy, values = _classify(
                    train_features,
                    test_features,
                    targets[train_positions],
                    targets[test_positions],
                    n_classes=n_classes,
                    alpha=args.ridge_alpha,
                )
                metrics[f"{task}_balanced_accuracy"] = accuracy
                predictions[task] = values
            branch_payload[branch] = {**metrics, "predictions": predictions}
            stability[branch] = {
                "cosine_similarity": float(
                    np.mean(
                        [
                            _cosine(projected_original[index], projected_relabeled[index])
                            for index in test_positions
                        ]
                    )
                ),
                "relative_l2_difference": float(
                    np.mean(
                        [
                            _relative_l2(projected_original[index], projected_relabeled[index])
                            for index in test_positions
                        ]
                    )
                ),
            }
        invariants = {
            "train_test_graph_isolation": not bool(set(train_positions) & set(test_positions)),
            "all_joint_classes_train": len(
                set(joint_targets_array[train_positions].tolist())
            )
            == len(FAMILIES) * len(args.degrees),
            "all_joint_classes_test": len(
                set(joint_targets_array[test_positions].tolist())
            )
            == len(FAMILIES) * len(args.degrees),
            "projection_rank": all(
                len(all_original_features[branch]) == graph_count for branch in BRANCHES
            ),
            "finite": all(
                np.isfinite(branch_payload[branch]["joint_balanced_accuracy"])
                for branch in BRANCHES
            ),
        }
        invariants["passed"] = all(invariants.values())
        folds.append(
            {
                "fold_index": fold_index,
                "train_graph_count": len(train_positions),
                "test_graph_count": len(test_positions),
                "initialization": initialization,
                "training_reconstruction_curve": [
                    float(value) for value in training_info["recon_curve"]
                ],
                "branches": branch_payload,
                "stability": stability,
                "invariants": invariants,
            }
        )
        print(
            f"fold={fold_index} stats/rawrel/ksvdbag/ksvdtrue/shuffled="
            f"{branch_payload['GLOBAL_STATS']['joint_balanced_accuracy']:.4f}/"
            f"{branch_payload['ROOTED_RAW_TRUE_RELATION']['joint_balanced_accuracy']:.4f}/"
            f"{branch_payload['ROOTED_KSVD_BAG']['joint_balanced_accuracy']:.4f}/"
            f"{branch_payload['ROOTED_KSVD_TRUE_RELATION']['joint_balanced_accuracy']:.4f}/"
            f"{branch_payload['ROOTED_KSVD_SHUFFLED_RELATION']['joint_balanced_accuracy']:.4f}",
            flush=True,
        )

    decision = classify_folds(folds)
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
            "projection_rank": args.projection_rank,
            "ridge_alpha": args.ridge_alpha,
            "labels_used_only_in_final_readout": True,
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

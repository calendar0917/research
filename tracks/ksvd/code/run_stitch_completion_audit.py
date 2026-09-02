"""Audit train-only stitch reliability and unseen-pair graph completion."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .from_scratch_unplanted_dictionary import deterministic_maximin_initialization
from .from_scratch_unplanted_representation import upper_triangle_edges
from .imdb_walk_dictionary import encode_with_minimum_sparsity
from .ksvd import ksvd
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import patch_budget, sample_edge_target_bridge_cover
from .overlap_stitching import (
    CoverExample,
    make_cover_example,
    stack_cover_examples,
    stitch_patch_predictions,
)
from .run_overlap_cover_audit import FAMILIES, generate_graph
from .transition_decoder import fit_ridge_decoder, predict_ridge_decoder


DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/stitch_completion_audit_20260802.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/STITCH_COMPLETION_AUDIT_20260802.md"
)
EPS = 1e-12


def _split_examples(
    examples: Sequence[CoverExample], fold_index: int
) -> tuple[tuple[CoverExample, ...], tuple[CoverExample, ...]]:
    test = tuple(
        example for example in examples if example.graph_index % 8 % 3 == fold_index
    )
    test_ids = {example.graph_index for example in test}
    train = tuple(example for example in examples if example.graph_index not in test_ids)
    return train, test


def _encode_predictions(
    examples: Sequence[CoverExample],
    dictionary: np.ndarray,
    train_mean: np.ndarray,
    *,
    sparsity: int,
) -> list[np.ndarray]:
    values = stack_cover_examples(examples)
    centered = values - train_mean
    codes = encode_with_minimum_sparsity(
        centered,
        dictionary,
        sparsity=sparsity,
        minimum_sparsity=1,
    )
    reconstructed = dictionary @ codes + train_mean
    predictions = []
    cursor = 0
    for example in examples:
        count = example.patch_vectors.shape[0]
        predictions.append(reconstructed[:, cursor : cursor + count].T)
        cursor += count
    if cursor != reconstructed.shape[1]:
        raise RuntimeError("prediction slices do not consume reconstruction")
    return predictions


def _fit_slot_weights(
    examples: Sequence[CoverExample], predictions: Sequence[np.ndarray]
) -> tuple[np.ndarray, dict[str, float]]:
    residuals = np.concatenate(
        [prediction - example.patch_vectors for example, prediction in zip(examples, predictions)],
        axis=0,
    )
    slot_mse = np.mean(residuals**2, axis=0)
    global_mse = float(np.mean(residuals**2))
    weights = 1.0 / (slot_mse + 0.1 * global_mse + EPS)
    weights /= np.mean(weights)
    return weights, {
        "global_patch_entry_mse": global_mse,
        "minimum_slot_mse": float(np.min(slot_mse)),
        "maximum_slot_mse": float(np.max(slot_mse)),
        "minimum_weight": float(np.min(weights)),
        "maximum_weight": float(np.max(weights)),
    }


def _partial_graph(example: CoverExample) -> tuple[np.ndarray, np.ndarray]:
    n_nodes = example.adjacency.shape[0]
    observed = np.zeros((n_nodes, n_nodes), dtype=bool)
    partial = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    local_edges = upper_triangle_edges(len(example.cover.patches[0].node_ids))
    for patch in example.cover.patches:
        for left_slot, right_slot in local_edges:
            left = patch.node_ids[left_slot]
            right = patch.node_ids[right_slot]
            observed[left, right] = observed[right, left] = True
            if example.adjacency[left, right] != 0:
                partial[left, right] = partial[right, left] = 1.0
    return observed, partial


def _completion_features(
    example: CoverExample,
) -> tuple[list[tuple[int, int]], np.ndarray, np.ndarray]:
    observed, partial = _partial_graph(example)
    n_nodes = partial.shape[0]
    degrees = np.sum(partial, axis=1) / max(n_nodes - 1, 1)
    pairs = []
    features = []
    targets = []
    for left, right in combinations(range(n_nodes), 2):
        if observed[left, right]:
            continue
        left_degree = float(degrees[left])
        right_degree = float(degrees[right])
        low = min(left_degree, right_degree)
        high = max(left_degree, right_degree)
        common_count = float(partial[left] @ partial[right])
        common = common_count / max(n_nodes - 2, 1)
        union_count = float(
            np.count_nonzero((partial[left] != 0.0) | (partial[right] != 0.0))
        )
        jaccard = common_count / max(union_count, 1.0)
        features.append(
            [
                low,
                high,
                left_degree + right_degree,
                high - low,
                left_degree * right_degree,
                common,
                jaccard,
            ]
        )
        targets.append(float(example.adjacency[left, right]))
        pairs.append((left, right))
    return (
        pairs,
        np.asarray(features, dtype=np.float64),
        np.asarray(targets, dtype=np.float64),
    )


def _stitch_values(
    example: CoverExample,
    predictions: np.ndarray,
    weights: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_nodes = example.adjacency.shape[0]
    all_pairs = list(combinations(range(n_nodes), 2))
    pair_positions = {pair: index for index, pair in enumerate(all_pairs)}
    truth = np.asarray(
        [example.adjacency[left, right] for left, right in all_pairs],
        dtype=np.float64,
    )
    local_edges = upper_triangle_edges(len(example.cover.patches[0].node_ids))
    slot_weights = (
        np.ones(len(local_edges), dtype=np.float64)
        if weights is None
        else np.asarray(weights, dtype=np.float64)
    )
    values: dict[tuple[int, int], list[float]] = {}
    value_weights: dict[tuple[int, int], list[float]] = {}
    for patch, vector in zip(example.cover.patches, predictions):
        for index, (left_slot, right_slot) in enumerate(local_edges):
            pair = tuple(sorted((patch.node_ids[left_slot], patch.node_ids[right_slot])))
            values.setdefault(pair, []).append(float(vector[index]))
            value_weights.setdefault(pair, []).append(float(slot_weights[index]))
    output = np.zeros(len(all_pairs), dtype=np.float64)
    observed_mask = np.zeros(len(all_pairs), dtype=bool)
    for pair, occurrences in values.items():
        position = pair_positions[pair]
        output[position] = float(
            np.average(occurrences, weights=value_weights[pair])
        )
        observed_mask[position] = True
    return truth, output, observed_mask


def _full_metrics(truth: np.ndarray, prediction: np.ndarray) -> dict[str, float]:
    binary_truth = truth.astype(np.int8)
    binary_prediction = (prediction >= 0.5).astype(np.int8)
    true_positive = int(
        np.count_nonzero((binary_truth == 1) & (binary_prediction == 1))
    )
    false_positive = int(
        np.count_nonzero((binary_truth == 0) & (binary_prediction == 1))
    )
    false_negative = int(
        np.count_nonzero((binary_truth == 1) & (binary_prediction == 0))
    )
    precision = true_positive / max(true_positive + false_positive, 1)
    recall = true_positive / max(true_positive + false_negative, 1)
    f1 = 2.0 * precision * recall / max(precision + recall, EPS)
    return {
        "full_adjacency_rmse": float(np.sqrt(np.mean((prediction - truth) ** 2))),
        "full_edge_precision": float(precision),
        "full_edge_recall": float(recall),
        "full_edge_f1": float(f1),
        "prediction_minimum": float(np.min(prediction)),
        "prediction_maximum": float(np.max(prediction)),
    }


def _mean_metric_rows(rows: Sequence[dict[str, Any]]) -> dict[str, float]:
    keys = (
        "full_adjacency_rmse",
        "full_edge_precision",
        "full_edge_recall",
        "full_edge_f1",
    )
    return {
        key: float(np.mean([float(row[key]) for row in rows])) for key in keys
    }


def _run_fold(
    examples: Sequence[CoverExample],
    fold_index: int,
    *,
    n_atoms: int,
    sparsity: int,
    iterations: int,
    ridge_alpha: float,
) -> dict[str, Any]:
    train_examples, test_examples = _split_examples(examples, fold_index)
    raw_train = stack_cover_examples(train_examples)
    train_mean = np.mean(raw_train, axis=1, keepdims=True)
    centered_train = raw_train - train_mean
    initial_dictionary, _initialization = deterministic_maximin_initialization(
        centered_train, n_atoms
    )
    final_dictionary, _codes, _info = ksvd(
        centered_train,
        n_atoms=n_atoms,
        T=sparsity,
        T_min=1,
        n_iter=iterations,
        seed=0,
        initial_dictionary=initial_dictionary,
    )
    train_predictions = _encode_predictions(
        train_examples,
        final_dictionary,
        train_mean,
        sparsity=sparsity,
    )
    test_predictions = _encode_predictions(
        test_examples,
        final_dictionary,
        train_mean,
        sparsity=sparsity,
    )
    slot_weights, slot_weight_audit = _fit_slot_weights(
        train_examples, train_predictions
    )

    train_feature_rows = []
    train_targets = []
    for example in train_examples:
        _pairs, features, targets = _completion_features(example)
        train_feature_rows.append(features)
        train_targets.append(targets)
    completion_features = np.concatenate(train_feature_rows, axis=0)
    completion_targets = np.concatenate(train_targets, axis=0)
    density_prediction = float(np.mean(completion_targets))
    completion_model = fit_ridge_decoder(
        completion_features,
        completion_targets[:, None],
        alpha=ridge_alpha,
    )

    base_rows = {"raw_uniform": [], "ksvd_uniform": [], "ksvd_weighted": []}
    stage_rows: dict[str, list[dict[str, Any]]] = {}
    completion_methods = ("zero", "density", "structural", "shuffled")
    for base in base_rows:
        for method in completion_methods:
            stage_rows[f"{base}_{method}"] = []

    for example, ksvd_prediction in zip(test_examples, test_predictions):
        base_predictions = {
            "raw_uniform": (example.patch_vectors, None),
            "ksvd_uniform": (ksvd_prediction, None),
            "ksvd_weighted": (ksvd_prediction, slot_weights),
        }
        pairs, features, _targets = _completion_features(example)
        structural = np.clip(
            predict_ridge_decoder(completion_model, features).ravel(), 0.0, 1.0
        )
        shuffled = np.roll(structural, 1) if structural.size > 1 else structural.copy()
        completion_values = {
            "zero": np.zeros(len(pairs), dtype=np.float64),
            "density": np.full(len(pairs), density_prediction, dtype=np.float64),
            "structural": structural,
            "shuffled": shuffled,
        }
        all_pairs = list(combinations(range(example.adjacency.shape[0]), 2))
        pair_positions = {pair: index for index, pair in enumerate(all_pairs)}
        unseen_positions = np.asarray(
            [pair_positions[pair] for pair in pairs], dtype=np.int64
        )
        for base, (patch_predictions, weights) in base_predictions.items():
            base_metrics = stitch_patch_predictions(
                example,
                patch_predictions,
                local_pair_weights=weights,
            )
            base_rows[base].append(
                {
                    "graph_index": example.graph_index,
                    "observed_pair_rmse": base_metrics["observed_pair_rmse"],
                    "observed_edge_f1": base_metrics["observed_edge_f1"],
                    "full_edge_recall_zero": base_metrics["full_edge_recall"],
                }
            )
            truth, stitched, observed_mask = _stitch_values(
                example, patch_predictions, weights
            )
            if np.any(observed_mask[unseen_positions]):
                raise RuntimeError("completion pair unexpectedly marked observed")
            for method, values in completion_values.items():
                completed = stitched.copy()
                completed[unseen_positions] = values
                stage_rows[f"{base}_{method}"].append(
                    {
                        "graph_index": example.graph_index,
                        **_full_metrics(truth, completed),
                    }
                )

    base_summaries = {
        base: {
            key: float(np.mean([row[key] for row in rows]))
            for key in (
                "observed_pair_rmse",
                "observed_edge_f1",
                "full_edge_recall_zero",
            )
        }
        for base, rows in base_rows.items()
    }
    stage_summaries = {
        stage: _mean_metric_rows(rows) for stage, rows in stage_rows.items()
    }
    return {
        "fold_index": fold_index,
        "train_graph_count": len(train_examples),
        "test_graph_count": len(test_examples),
        "train_test_isolation": set(example.graph_index for example in train_examples).isdisjoint(
            example.graph_index for example in test_examples
        ),
        "slot_weights": [float(value) for value in slot_weights],
        "slot_weight_audit": slot_weight_audit,
        "completion_train_row_count": int(completion_features.shape[0]),
        "completion_feature_dimension": int(completion_features.shape[1]),
        "completion_train_density": density_prediction,
        "base_summaries": base_summaries,
        "stage_summaries": stage_summaries,
    }


def _relative_reduction(left: float, right: float) -> float:
    return float((left - right) / max(left, EPS))


def classify(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    base_means = {}
    for base in folds[0]["base_summaries"]:
        base_means[base] = {
            key: float(np.mean([fold["base_summaries"][base][key] for fold in folds]))
            for key in folds[0]["base_summaries"][base]
        }
    stage_means = {}
    for stage in folds[0]["stage_summaries"]:
        stage_means[stage] = {
            key: float(np.mean([fold["stage_summaries"][stage][key] for fold in folds]))
            for key in folds[0]["stage_summaries"][stage]
        }
    uniform = base_means["ksvd_uniform"]
    weighted = base_means["ksvd_weighted"]
    stitch_gain = _relative_reduction(
        uniform["observed_pair_rmse"], weighted["observed_pair_rmse"]
    )
    stitch_checks = {
        "observed_rmse_reduction_at_least_001": stitch_gain >= 0.01,
        "observed_f1_not_worse": weighted["observed_edge_f1"]
        >= uniform["observed_edge_f1"] - 1e-12,
        "full_recall_preserved": weighted["full_edge_recall_zero"]
        >= uniform["full_edge_recall_zero"] - 0.01,
    }
    density = stage_means["raw_uniform_density"]
    structural = stage_means["raw_uniform_structural"]
    shuffled = stage_means["raw_uniform_shuffled"]
    structural_density_gain = _relative_reduction(
        density["full_adjacency_rmse"], structural["full_adjacency_rmse"]
    )
    structural_shuffled_gain = _relative_reduction(
        shuffled["full_adjacency_rmse"], structural["full_adjacency_rmse"]
    )
    completion_checks = {
        "rmse_vs_density_at_least_001": structural_density_gain >= 0.01,
        "rmse_vs_shuffled_at_least_001": structural_shuffled_gain >= 0.01,
    }
    stitch_gate = all(stitch_checks.values())
    completion_gate = all(completion_checks.values())
    invariants = all(fold["train_test_isolation"] for fold in folds) and all(
        fold["completion_feature_dimension"] == 7 for fold in folds
    )
    if not invariants:
        label = "FAIL_STITCH_COMPLETION_INVARIANTS"
    elif stitch_gate or completion_gate:
        label = "TRAIN_ONLY_POSTPROCESSING_REDUCES_ERROR"
    else:
        label = "NO_REGISTERED_POSTPROCESSING_GAIN"
    raw_zero = stage_means["raw_uniform_zero"]["full_adjacency_rmse"]
    ksvd_uniform_zero = stage_means["ksvd_uniform_zero"]["full_adjacency_rmse"]
    ksvd_weighted_zero = stage_means["ksvd_weighted_zero"]["full_adjacency_rmse"]
    ksvd_weighted_structural = stage_means["ksvd_weighted_structural"][
        "full_adjacency_rmse"
    ]
    return {
        "classification": label,
        "invariants": invariants,
        "stitch_gate": stitch_gate,
        "stitch_checks": stitch_checks,
        "completion_gate": completion_gate,
        "completion_checks": completion_checks,
        "slot_weighted_observed_rmse_reduction": stitch_gain,
        "structural_vs_density_full_rmse_reduction": structural_density_gain,
        "structural_vs_shuffled_full_rmse_reduction": structural_shuffled_gain,
        "base_means": base_means,
        "stage_means": stage_means,
        "error_budget": {
            "raw_zero_full_rmse": raw_zero,
            "ksvd_uniform_zero_full_rmse": ksvd_uniform_zero,
            "ksvd_weighted_zero_full_rmse": ksvd_weighted_zero,
            "ksvd_weighted_structural_full_rmse": ksvd_weighted_structural,
            "compression_rmse_increment": ksvd_uniform_zero - raw_zero,
            "slot_weighting_rmse_change": ksvd_weighted_zero - ksvd_uniform_zero,
            "completion_rmse_change": ksvd_weighted_structural
            - ksvd_weighted_zero,
        },
    }


def _fmt(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Train-only stitching reliability 与 unseen completion 审计",
        "",
        "> 日期：2026-08-02  ",
        "> K24/T3/u25；3-fold graph isolation；test truth 不参与权重或 completion 拟合。",
        "",
        "## 1. 判定",
        "",
        f"**{decision['classification']}**",
        "",
        "## 2. Observed stitching",
        "",
        "| base | observed RMSE | observed F1 | zero-fill full recall |",
        "|---|---:|---:|---:|",
    ]
    for base in ("raw_uniform", "ksvd_uniform", "ksvd_weighted"):
        mean = decision["base_means"][base]
        lines.append(
            f"| {base} | {_fmt(mean['observed_pair_rmse'])} | "
            f"{_fmt(mean['observed_edge_f1'])} | "
            f"{_fmt(mean['full_edge_recall_zero'])} |"
        )
    lines.extend(
        [
            "",
            f"Slot weighting RMSE reduction：`{_fmt(decision['slot_weighted_observed_rmse_reduction'])}`；checks：`{decision['stitch_checks']}`。",
            "",
            "## 3. Full graph completion",
            "",
            "| stage | full RMSE | precision | recall | F1 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for stage, mean in decision["stage_means"].items():
        lines.append(
            f"| {stage} | {_fmt(mean['full_adjacency_rmse'])} | "
            f"{_fmt(mean['full_edge_precision'])} | "
            f"{_fmt(mean['full_edge_recall'])} | "
            f"{_fmt(mean['full_edge_f1'])} |"
        )
    lines.extend(
        [
            "",
            f"Structural vs density RMSE reduction：`{_fmt(decision['structural_vs_density_full_rmse_reduction'])}`。",
            "",
            f"Structural vs shuffled RMSE reduction：`{_fmt(decision['structural_vs_shuffled_full_rmse_reduction'])}`；checks：`{decision['completion_checks']}`。",
            "",
            "## 4. Error budget",
            "",
            f"`{decision['error_budget']}`",
            "",
            "## 5. 边界",
            "",
            "- slot reliability 只改变 repeated-pair aggregation，不改变 patch reconstruction 本身。",
            "- completion targets 来自 train graphs 的 unseen pairs；它是 graph completion，不是 patch compressor 的内生能力。",
            "- shuffled structural predictions 保留每图预测分布，只破坏 pair binding。",
            "",
        ]
    )
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--graph-bank-seed", type=int, default=810001)
    parser.add_argument("--cover-seed", type=int, default=830101)
    parser.add_argument("--n-nodes", type=int, default=50)
    parser.add_argument("--degrees", type=int, nargs="+", default=[15, 20, 25])
    parser.add_argument("--graphs-per-cell", type=int, default=8)
    parser.add_argument("--patch-size", type=int, default=10)
    parser.add_argument("--target-overlap", type=int, default=5)
    parser.add_argument("--edge-capacity-multiplier", type=float, default=1.5)
    parser.add_argument("--n-atoms", type=int, default=24)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--ridge-alpha", type=float, default=1e-2)
    parser.add_argument("--sampler", choices=("target", "beam"), default="target")
    parser.add_argument("--retained-beam", type=int, default=8)
    parser.add_argument("--candidate-restarts", type=int, default=1)
    parser.add_argument("--sampler-seed-tag", type=int, default=5501)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    examples = []
    sequences = np.random.SeedSequence(args.graph_bank_seed).spawn(
        len(FAMILIES) * len(args.degrees) * args.graphs_per_cell
    )
    graph_index = 0
    for family in FAMILIES:
        for degree in args.degrees:
            for _replicate in range(args.graphs_per_cell):
                graph_seed = int(
                    sequences[graph_index].generate_state(1, dtype=np.uint32)[0]
                )
                adjacency = generate_graph(family, args.n_nodes, degree, graph_seed)
                budget = patch_budget(
                    adjacency,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                    edge_capacity_multiplier=args.edge_capacity_multiplier,
                )
                sampler_seed = int(
                    np.random.SeedSequence(
                        [args.cover_seed, graph_index, args.sampler_seed_tag]
                    ).generate_state(1, dtype=np.uint32)[0]
                )
                if args.sampler == "target":
                    cover = sample_edge_target_bridge_cover(
                        adjacency,
                        np.random.default_rng(sampler_seed),
                        n_patches=budget,
                        patch_size=args.patch_size,
                        target_overlap=args.target_overlap,
                    )
                else:
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
        fold = _run_fold(
            examples,
            fold_index,
            n_atoms=args.n_atoms,
            sparsity=args.sparsity,
            iterations=args.iterations,
            ridge_alpha=args.ridge_alpha,
        )
        folds.append(fold)
        print(
            f"fold={fold_index} uniform/weighted_observed_rmse="
            f"{fold['base_summaries']['ksvd_uniform']['observed_pair_rmse']:.4f}/"
            f"{fold['base_summaries']['ksvd_weighted']['observed_pair_rmse']:.4f} "
            f"raw_zero/density/structural_full_rmse="
            f"{fold['stage_summaries']['raw_uniform_zero']['full_adjacency_rmse']:.4f}/"
            f"{fold['stage_summaries']['raw_uniform_density']['full_adjacency_rmse']:.4f}/"
            f"{fold['stage_summaries']['raw_uniform_structural']['full_adjacency_rmse']:.4f}",
            flush=True,
        )
    decision = classify(folds)
    payload = {
        "protocol": "stitch-completion-audit-v0-20260802",
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seed": args.cover_seed,
            "families": list(FAMILIES),
            "n_nodes": args.n_nodes,
            "target_degrees": args.degrees,
            "graphs_per_family_degree": args.graphs_per_cell,
            "patch_size": args.patch_size,
            "target_overlap": args.target_overlap,
            "edge_capacity_multiplier": args.edge_capacity_multiplier,
            "n_atoms": args.n_atoms,
            "sparsity": args.sparsity,
            "iterations": args.iterations,
            "ridge_alpha": args.ridge_alpha,
            "sampler": args.sampler,
            "retained_beam": args.retained_beam,
            "candidate_restarts": args.candidate_restarts,
            "sampler_seed_tag": args.sampler_seed_tag,
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

"""Capacity grid for sparse KSVD tokens learned in invariant patch space."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from .from_scratch_unplanted_dictionary import (
    deterministic_maximin_initialization,
    dictionary_metrics,
)
from .ksvd import ksvd
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import patch_budget
from .overlap_stitching import make_cover_example
from .run_invariant_space_ksvd_token_audit import (
    BRANCHES,
    build_masked_matrices,
    classify,
    encode_examples,
    graph_embedding,
    stack_invariant_examples,
)
from .run_overlap_cover_audit import FAMILIES, generate_graph
from .run_patch_relation_representation_audit import _cosine, _relative_l2, graph_balanced_metrics
from .transition_decoder import fit_ridge_decoder, predict_ridge_decoder


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/from_scratch/invariant_space_ksvd_capacity_audit_20260802.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/from_scratch/INVARIANT_SPACE_KSVD_CAPACITY_AUDIT_20260802.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_INVARIANT_SPACE_CAPACITY_PROTOCOL_20260802.md"
CELLS = ((16, 3), (16, 4), (24, 3), (24, 4), (32, 3), (32, 4))


def classify_cells(cells: list[dict[str, Any]]) -> dict[str, Any]:
    passing = [
        cell
        for cell in cells
        if cell["decision"]["classification"] == "ADOPT_INVARIANT_SPACE_KSVD_TOKEN"
    ]
    selected = None
    if passing:
        selected = min(
            passing,
            key=lambda cell: (
                int(cell["sparsity"]),
                int(cell["n_atoms"]),
                float(
                    cell["decision"]["mean_branches"]["INV_KSVD_TRUE_RELATION"][
                        "overall_rmse"
                    ]
                ),
            ),
        )
    return {
        "classification": (
            "ADOPT_INVARIANT_KSVD_CAPACITY"
            if selected is not None
            else "INVARIANT_KSVD_CAPACITY_GRID_FAILS_ERROR_GATE"
        ),
        "passing_cell_ids": [cell["cell_id"] for cell in passing],
        "selected_cell_id": None if selected is None else selected["cell_id"],
    }


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Invariant-space KSVD capacity 审计",
        "",
        "> 日期：2026-08-02  ",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{decision['classification']}`",
        "",
        "| cell | dict scalars | code scalars/graph | TRUE RMSE | vs raw invariant | vs BAG | vs SHUFFLED | cosine | cell decision |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for cell in payload["cells"]:
        cell_decision = cell["decision"]
        true = cell_decision["mean_branches"]["INV_KSVD_TRUE_RELATION"]
        cosine = cell_decision["stability"]["INV_KSVD_TRUE_RELATION"][
            "cosine_similarity"
        ]
        lines.append(
            f"| {cell['cell_id']} | {cell['dictionary_scalars']} | "
            f"{cell['mean_code_scalars_per_graph']:.2f} | {true['overall_rmse']:.5f} | "
            f"{cell_decision['true_vs_raw_invariant_rmse_change']:.4f} | "
            f"{cell_decision['true_vs_bag_rmse_reduction']:.4f} | "
            f"{cell_decision['true_vs_shuffled_rmse_reduction']:.4f} | "
            f"{cosine:.4f} | {cell_decision['classification']} |"
        )
    lines.extend(
        [
            "",
            f"Passing cells：`{decision['passing_cell_ids']}`。",
            f"Selected cell：`{decision['selected_cell_id']}`。",
            "",
        ]
    )
    return "\n".join(lines)


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
    parser.add_argument("--iterations", type=int, default=25)
    parser.add_argument("--ridge-alpha", type=float, default=0.01)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    graph_count = len(FAMILIES) * len(args.degrees) * args.graphs_per_cell
    sequences = np.random.SeedSequence(args.graph_bank_seed).spawn(graph_count)
    examples = []
    relabeled_examples = {}
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
                cover = sample_marginal_candidate_cover(
                    adjacency,
                    np.random.default_rng(sampler_seed),
                    n_patches=budget,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                    retained_beam=args.retained_beam,
                    candidate_restarts=args.candidate_restarts,
                )
                example = make_cover_example(graph_index, family, degree, adjacency, cover)
                examples.append(example)
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
                relabeled_examples[graph_index] = make_cover_example(
                    graph_index,
                    family,
                    degree,
                    relabeled_adjacency,
                    relabeled_cover,
                )
                graph_index += 1

    cell_payload = []
    mean_patch_count = float(np.mean([len(example.cover.patches) for example in examples]))
    for n_atoms, sparsity in CELLS:
        folds = []
        for fold_index in range(3):
            test_examples = tuple(
                example for example in examples if example.graph_index % 8 % 3 == fold_index
            )
            test_ids = {example.graph_index for example in test_examples}
            train_examples = tuple(
                example for example in examples if example.graph_index not in test_ids
            )
            raw_train = stack_invariant_examples(train_examples)
            train_mean = np.mean(raw_train, axis=1, keepdims=True)
            centered_train = raw_train - train_mean
            initial_dictionary, initialization = deterministic_maximin_initialization(
                centered_train, n_atoms
            )
            dictionary, training_codes_matrix, training_info = ksvd(
                centered_train,
                n_atoms=n_atoms,
                T=sparsity,
                T_min=1,
                n_iter=args.iterations,
                seed=0,
                initial_dictionary=initial_dictionary,
            )
            health = dictionary_metrics(
                centered_train, dictionary, training_codes_matrix
            )
            train_codes = encode_examples(
                train_examples, dictionary, train_mean, sparsity=sparsity
            )
            test_codes = encode_examples(
                test_examples, dictionary, train_mean, sparsity=sparsity
            )
            train_features, train_targets, _train_rows = build_masked_matrices(
                train_examples, train_codes
            )
            test_features, test_targets, test_graph_rows = build_masked_matrices(
                test_examples, test_codes
            )
            branches = {}
            for branch in BRANCHES:
                model = fit_ridge_decoder(
                    train_features[branch], train_targets, alpha=args.ridge_alpha
                )
                predictions = predict_ridge_decoder(model, test_features[branch])
                graph_rows, summary = graph_balanced_metrics(
                    test_targets, predictions, test_graph_rows
                )
                branches[branch] = {"summary": summary, "graphs": graph_rows}
            stability_rows = []
            for example in test_examples:
                relabeled = relabeled_examples[example.graph_index]
                relabeled_codes = encode_examples(
                    [relabeled], dictionary, train_mean, sparsity=sparsity
                )[example.graph_index]
                row = {"graph_index": example.graph_index}
                for branch in BRANCHES:
                    original_embedding = graph_embedding(
                        branch, example, test_codes[example.graph_index]
                    )
                    relabeled_embedding = graph_embedding(
                        branch, relabeled, relabeled_codes
                    )
                    row[branch] = {
                        "cosine_similarity": _cosine(
                            original_embedding, relabeled_embedding
                        ),
                        "relative_l2_difference": _relative_l2(
                            original_embedding, relabeled_embedding
                        ),
                    }
                stability_rows.append(row)
            stability = {
                branch: {
                    "cosine_similarity": float(
                        np.mean(
                            [row[branch]["cosine_similarity"] for row in stability_rows]
                        )
                    ),
                    "relative_l2_difference": float(
                        np.mean(
                            [
                                row[branch]["relative_l2_difference"]
                                for row in stability_rows
                            ]
                        )
                    ),
                }
                for branch in BRANCHES
            }
            expected_train = sum(len(example.cover.patches) for example in train_examples)
            expected_test = sum(len(example.cover.patches) for example in test_examples)
            invariants = {
                "train_test_graph_isolation": not (
                    {example.graph_index for example in train_examples} & test_ids
                ),
                "train_row_count": train_targets.shape[0] == expected_train,
                "test_row_count": test_targets.shape[0] == expected_test,
                "finite": all(
                    np.all(np.isfinite(matrix)) for matrix in train_features.values()
                )
                and all(np.all(np.isfinite(matrix)) for matrix in test_features.values()),
                "nondead_atoms": health["nondead_atom_count"] >= int(np.ceil(0.8 * n_atoms)),
            }
            invariants["passed"] = all(invariants.values())
            folds.append(
                {
                    "fold_index": fold_index,
                    "initialization": initialization,
                    "training_reconstruction_curve": [
                        float(value) for value in training_info["recon_curve"]
                    ],
                    "dictionary_health": health,
                    "branches": branches,
                    "stability": stability,
                    "invariants": invariants,
                }
            )
        cell_decision = classify(folds)
        cell_id = f"K{n_atoms}_T{sparsity}"
        cell_payload.append(
            {
                "cell_id": cell_id,
                "n_atoms": n_atoms,
                "sparsity": sparsity,
                "dictionary_scalars": 22 * n_atoms,
                "mean_code_scalars_per_graph": mean_patch_count * sparsity,
                "folds": folds,
                "decision": cell_decision,
            }
        )
        print(
            f"cell={cell_id} true_rmse="
            f"{cell_decision['mean_branches']['INV_KSVD_TRUE_RELATION']['overall_rmse']:.5f} "
            f"cos={cell_decision['stability']['INV_KSVD_TRUE_RELATION']['cosine_similarity']:.4f} "
            f"decision={cell_decision['classification']}",
            flush=True,
        )

    decision = classify_cells(cell_payload)
    payload = {
        "protocol": PROTOCOL,
        "config": {
            "graph_bank_seed": args.graph_bank_seed,
            "cover_seed": args.cover_seed,
            "families": list(FAMILIES),
            "degrees": args.degrees,
            "graphs_per_cell": args.graphs_per_cell,
            "graph_count": graph_count,
            "patch_size": args.patch_size,
            "target_overlap": args.target_overlap,
            "multiplier": args.multiplier,
            "retained_beam": args.retained_beam,
            "candidate_restarts": args.candidate_restarts,
            "feature_dimension": 22,
            "iterations": args.iterations,
            "ridge_alpha": args.ridge_alpha,
            "grid": [list(cell) for cell in CELLS],
            "labels_used": False,
        },
        "cells": cell_payload,
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

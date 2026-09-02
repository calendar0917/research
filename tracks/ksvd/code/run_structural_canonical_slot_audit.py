"""Audit full-adjacency KSVD under ID-free structural patch slot orders."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from .canonical_slots import reorder_cover_structurally
from .from_scratch_unplanted_dictionary import (
    deterministic_maximin_initialization,
    dictionary_metrics,
)
from .imdb_walk_dictionary import encode_with_minimum_sparsity
from .ksvd import ksvd
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import PatchCover, patch_budget, remap_cover
from .overlap_stitching import CoverExample, make_cover_example, stack_cover_examples
from .run_ksvd_id_slot_decomposition_audit import (
    _matched_code_cosine,
    _support_jaccard,
    _vector_row_match,
)
from .run_ksvd_stitched_reconstruction_audit import _evaluate_reconstruction
from .run_overlap_cover_audit import FAMILIES, generate_graph
from .run_patch_relation_representation_audit import _cosine, _relative_l2, pool_tokens


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/from_scratch/structural_canonical_slot_audit_20260802.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/from_scratch/STRUCTURAL_CANONICAL_SLOT_AUDIT_20260802.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_STRUCTURAL_CANONICAL_SLOT_PROTOCOL_20260802.md"
BRANCHES = (
    "CONSTRUCTION",
    "ID_SORT",
    "SIGNATURE",
    "CANONICAL",
    "ROOTED_CANONICAL",
    "OVERLAP_CANONICAL",
)
MODE_BY_BRANCH = {
    "ID_SORT": "id",
    "SIGNATURE": "signature",
    "CANONICAL": "canonical",
    "ROOTED_CANONICAL": "rooted_canonical",
    "OVERLAP_CANONICAL": "overlap_canonical",
}
CANDIDATES = ("CANONICAL", "ROOTED_CANONICAL", "OVERLAP_CANONICAL")


def _ordered_cover(
    adjacency: np.ndarray,
    cover: PatchCover,
    branch: str,
) -> tuple[PatchCover, dict[str, float | int]]:
    if branch == "CONSTRUCTION":
        return cover, {
            "patch_count": len(cover.patches),
            "signature_tie_rate": 0.0,
            "canonical_ambiguity_rate": 0.0,
            "mean_search_leaves": 0.0,
            "max_search_leaves": 0,
        }
    return reorder_cover_structurally(adjacency, cover, MODE_BY_BRANCH[branch])


def _encode(
    example: CoverExample,
    dictionary: np.ndarray,
    train_mean: np.ndarray,
    sparsity: int,
) -> np.ndarray:
    return encode_with_minimum_sparsity(
        example.patch_vectors.T - train_mean,
        dictionary,
        sparsity=sparsity,
        minimum_sparsity=1,
    ).T


def _transition_match(left: PatchCover, right: PatchCover) -> float:
    if len(left.transitions) != len(right.transitions):
        return 0.0
    if not left.transitions:
        return 1.0
    return float(
        np.mean(
            [
                a.left_to_right_slots == b.left_to_right_slots
                for a, b in zip(left.transitions, right.transitions)
            ]
        )
    )


def _node_order_equivariance(
    original: PatchCover,
    mapped: PatchCover,
    new_to_old: np.ndarray,
) -> float:
    if len(original.patches) != len(mapped.patches):
        return 0.0
    matches = []
    for left, right in zip(original.patches, mapped.patches):
        restored = tuple(int(new_to_old[node]) for node in right.node_ids)
        matches.append(restored == left.node_ids)
    return float(np.mean(matches))


def _mean_dicts(rows: Sequence[dict[str, float]]) -> dict[str, float]:
    keys = rows[0].keys()
    return {key: float(np.mean([row[key] for row in rows])) for key in keys}


def classify_branches(branches: dict[str, dict[str, Any]]) -> dict[str, Any]:
    construction = branches["CONSTRUCTION"]["mean_reconstruction"]
    decisions = {}
    passing = []
    for branch in BRANCHES:
        values = branches[branch]
        stability = values["mean_mapped_stability"]
        reconstruction = values["mean_reconstruction"]
        checks = {
            "vector_match_at_least_0999": stability["patch_vector_row_match"] >= 0.999,
            "code_cosine_at_least_099": stability["matched_patch_code_cosine"] >= 0.99,
            "pooled_cosine_at_least_099": stability["graph_embedding_cosine"] >= 0.99,
            "invariants": bool(values["all_invariants_passed"]),
            "observed_rmse_within_002": reconstruction["observed_pair_rmse"]
            <= 1.02 * construction["observed_pair_rmse"],
            "full_rmse_within_002": reconstruction["full_adjacency_rmse"]
            <= 1.02 * construction["full_adjacency_rmse"],
        }
        stable = all(
            checks[key]
            for key in (
                "vector_match_at_least_0999",
                "code_cosine_at_least_099",
                "pooled_cosine_at_least_099",
                "invariants",
            )
        )
        reconstruction_ok = checks["observed_rmse_within_002"] and checks["full_rmse_within_002"]
        if branch in CANDIDATES and stable and reconstruction_ok:
            label = "STRUCTURAL_CANONICAL_SLOT_SUPPORTED"
            passing.append(branch)
        elif branch in CANDIDATES and stable:
            label = "CANONICAL_STABLE_BUT_RECONSTRUCTION_WORSE"
        elif branch in CANDIDATES:
            label = "CANONICAL_SLOT_STABILITY_GATE_FAIL"
        else:
            label = "CONTROL"
        decisions[branch] = {"classification": label, "checks": checks}
    selected = None
    if passing:
        selected = min(
            passing,
            key=lambda branch: (
                branches[branch]["mean_reconstruction"]["full_adjacency_rmse"],
                branches[branch]["mean_reconstruction"]["observed_pair_rmse"],
            ),
        )
    return {
        "classification": (
            "ADOPT_STRUCTURAL_CANONICAL_SLOTS"
            if selected is not None
            else "STRUCTURAL_CANONICAL_SLOT_ROUTE_FAILS_JOINT_GATE"
        ),
        "passing_branches": passing,
        "selected_branch": selected,
        "branches": decisions,
    }


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# Structural canonical slots：完整邻接 KSVD 审计",
        "",
        "> 日期：2026-08-02  ",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{decision['classification']}`",
        "",
        "## 1. Frozen patch-set relabel stability",
        "",
        "| branch | vector match | code cosine | support Jaccard | pooled cosine | transition match | node-order equivariance |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for branch in BRANCHES:
        values = payload["branches"][branch]["mean_mapped_stability"]
        lines.append(
            f"| {branch} | {values['patch_vector_row_match']:.4f} | "
            f"{values['matched_patch_code_cosine']:.4f} | "
            f"{values['matched_patch_support_jaccard']:.4f} | "
            f"{values['graph_embedding_cosine']:.4f} | "
            f"{values['transition_slot_map_match']:.4f} | "
            f"{values['node_order_equivariance']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 2. Held-out KSVD + stitched reconstruction",
            "",
            "| branch | patch error | observed RMSE | observed F1 | full RMSE | relabel-resample full RMSE | abs delta | full recall | full F1 | resample cosine |",
            "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for branch in BRANCHES:
        values = payload["branches"][branch]
        reconstruction = values["mean_reconstruction"]
        resampled_reconstruction = values["mean_resampled_reconstruction"]
        full_delta = abs(
            resampled_reconstruction["full_adjacency_rmse"]
            - reconstruction["full_adjacency_rmse"]
        )
        lines.append(
            f"| {branch} | {reconstruction['patch_relative_error']:.4f} | "
            f"{reconstruction['observed_pair_rmse']:.4f} | "
            f"{reconstruction['observed_edge_f1']:.4f} | "
            f"{reconstruction['full_adjacency_rmse']:.4f} | "
            f"{resampled_reconstruction['full_adjacency_rmse']:.4f} | "
            f"{full_delta:.4f} | "
            f"{reconstruction['full_edge_recall']:.4f} | "
            f"{reconstruction['full_edge_f1']:.4f} | "
            f"{values['mean_resample_stability']['graph_embedding_cosine']:.4f} |"
        )
    lines.extend(
        [
            "",
            "## 3. Ordering diagnostics",
            "",
            "| branch | signature tie rate | canonical ambiguity | mean/max search leaves | decision |",
            "|---|---:|---:|---:|---|",
        ]
    )
    for branch in BRANCHES:
        values = payload["branches"][branch]["ordering_diagnostics"]
        branch_decision = decision["branches"][branch]["classification"]
        lines.append(
            f"| {branch} | {values['signature_tie_rate']:.4f} | "
            f"{values['canonical_ambiguity_rate']:.4f} | "
            f"{values['mean_search_leaves']:.2f}/{values['max_search_leaves']} | "
            f"{branch_decision} |"
        )
    lines.extend(
        [
            "",
            f"Passing branches：`{decision['passing_branches']}`。  ",
            f"Selected branch：`{decision['selected_branch']}`。",
            "",
            "Exact canonical vector 的稳定性与 canonical node-map 的唯一性是两件事。若 vector/code 为1而 node-order/transition低于1，差异来自 patch automorphism 中结构等价节点无法由纯结构唯一命名。",
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
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    graph_count = len(FAMILIES) * len(args.degrees) * args.graphs_per_cell
    sequences = np.random.SeedSequence(args.graph_bank_seed).spawn(graph_count)
    originals: dict[str, list[CoverExample]] = {branch: [] for branch in BRANCHES}
    mapped: dict[str, dict[int, CoverExample]] = {branch: {} for branch in BRANCHES}
    resampled: dict[str, dict[int, CoverExample]] = {branch: {} for branch in BRANCHES}
    permutations: dict[int, np.ndarray] = {}
    diagnostics: dict[str, list[dict[str, float | int]]] = {branch: [] for branch in BRANCHES}
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
                permutation = np.random.default_rng(
                    np.random.SeedSequence(
                        [args.graph_bank_seed, graph_index, 6701]
                    ).generate_state(1, dtype=np.uint32)[0]
                ).permutation(args.n_nodes)
                inverse = np.empty(args.n_nodes, dtype=np.int64)
                inverse[permutation] = np.arange(args.n_nodes)
                permutations[graph_index] = permutation
                relabeled_adjacency = adjacency[np.ix_(permutation, permutation)]
                mapped_cover = remap_cover(cover, inverse, relabeled_adjacency)
                resampled_cover = sample_marginal_candidate_cover(
                    relabeled_adjacency,
                    np.random.default_rng(sampler_seed),
                    n_patches=budget,
                    patch_size=args.patch_size,
                    target_overlap=args.target_overlap,
                    retained_beam=args.retained_beam,
                    candidate_restarts=args.candidate_restarts,
                )
                for branch in BRANCHES:
                    original_ordered, stats = _ordered_cover(adjacency, cover, branch)
                    mapped_ordered, _ = _ordered_cover(
                        relabeled_adjacency, mapped_cover, branch
                    )
                    resampled_ordered, _ = _ordered_cover(
                        relabeled_adjacency, resampled_cover, branch
                    )
                    diagnostics[branch].append(stats)
                    originals[branch].append(
                        make_cover_example(
                            graph_index, family, degree, adjacency, original_ordered
                        )
                    )
                    mapped[branch][graph_index] = make_cover_example(
                        graph_index,
                        family,
                        degree,
                        relabeled_adjacency,
                        mapped_ordered,
                    )
                    resampled[branch][graph_index] = make_cover_example(
                        graph_index,
                        family,
                        degree,
                        relabeled_adjacency,
                        resampled_ordered,
                    )
                graph_index += 1
                if graph_index % 12 == 0:
                    print(f"prepared_graphs={graph_index}/{graph_count}", flush=True)

    branch_payload: dict[str, Any] = {}
    for branch in BRANCHES:
        folds = []
        examples = originals[branch]
        for fold_index in range(3):
            test_examples = tuple(
                example for example in examples if example.graph_index % 8 % 3 == fold_index
            )
            test_ids = {example.graph_index for example in test_examples}
            train_examples = tuple(
                example for example in examples if example.graph_index not in test_ids
            )
            raw_train = stack_cover_examples(train_examples)
            raw_test = stack_cover_examples(test_examples)
            train_mean = np.mean(raw_train, axis=1, keepdims=True)
            centered_train = raw_train - train_mean
            initial_dictionary, initialization = deterministic_maximin_initialization(
                centered_train, args.n_atoms
            )
            dictionary, training_codes, training_info = ksvd(
                centered_train,
                n_atoms=args.n_atoms,
                T=args.sparsity,
                T_min=1,
                n_iter=args.iterations,
                seed=0,
                initial_dictionary=initial_dictionary,
            )
            test_codes_matrix = encode_with_minimum_sparsity(
                raw_test - train_mean,
                dictionary,
                sparsity=args.sparsity,
                minimum_sparsity=1,
            )
            reconstructed = dictionary @ test_codes_matrix + train_mean
            _raw_rows, raw_summary = _evaluate_reconstruction(test_examples, None)
            reconstruction_rows, reconstruction_summary = _evaluate_reconstruction(
                test_examples, reconstructed
            )
            resampled_test_examples = tuple(
                resampled[branch][example.graph_index] for example in test_examples
            )
            raw_resampled_test = stack_cover_examples(resampled_test_examples)
            resampled_codes_matrix = encode_with_minimum_sparsity(
                raw_resampled_test - train_mean,
                dictionary,
                sparsity=args.sparsity,
                minimum_sparsity=1,
            )
            resampled_reconstructed = dictionary @ resampled_codes_matrix + train_mean
            (
                resampled_reconstruction_rows,
                resampled_reconstruction_summary,
            ) = _evaluate_reconstruction(
                resampled_test_examples,
                resampled_reconstructed,
            )
            mapped_rows = []
            resample_rows = []
            for example in test_examples:
                base_codes = _encode(example, dictionary, train_mean, args.sparsity)
                mapped_example = mapped[branch][example.graph_index]
                mapped_codes = _encode(
                    mapped_example, dictionary, train_mean, args.sparsity
                )
                resampled_example = resampled[branch][example.graph_index]
                resampled_codes = _encode(
                    resampled_example, dictionary, train_mean, args.sparsity
                )
                mapped_rows.append(
                    {
                        "patch_vector_row_match": _vector_row_match(
                            example, mapped_example
                        ),
                        "matched_patch_code_cosine": _matched_code_cosine(
                            base_codes, mapped_codes
                        ),
                        "matched_patch_support_jaccard": _support_jaccard(
                            base_codes, mapped_codes
                        ),
                        "graph_embedding_cosine": _cosine(
                            pool_tokens(base_codes), pool_tokens(mapped_codes)
                        ),
                        "graph_embedding_relative_l2": _relative_l2(
                            pool_tokens(base_codes), pool_tokens(mapped_codes)
                        ),
                        "transition_slot_map_match": _transition_match(
                            example.cover, mapped_example.cover
                        ),
                        "node_order_equivariance": _node_order_equivariance(
                            example.cover,
                            mapped_example.cover,
                            permutations[example.graph_index],
                        ),
                    }
                )
                resample_rows.append(
                    {
                        "graph_embedding_cosine": _cosine(
                            pool_tokens(base_codes), pool_tokens(resampled_codes)
                        ),
                        "graph_embedding_relative_l2": _relative_l2(
                            pool_tokens(base_codes), pool_tokens(resampled_codes)
                        ),
                    }
                )
            health = dictionary_metrics(centered_train, dictionary, training_codes)
            invariants = {
                "train_test_graph_isolation": not (
                    {example.graph_index for example in train_examples} & test_ids
                ),
                "raw_patch_exact": raw_summary["patch_relative_error"] == 0.0,
                "raw_observed_exact": raw_summary["observed_pair_rmse"] == 0.0,
                "finite": bool(np.all(np.isfinite(reconstructed))),
                "nondead_atoms": health["nondead_atom_count"] >= 20,
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
                    "raw_reconstruction": raw_summary,
                    "reconstruction": reconstruction_summary,
                    "reconstruction_graphs": reconstruction_rows,
                    "resampled_reconstruction": resampled_reconstruction_summary,
                    "resampled_reconstruction_graphs": resampled_reconstruction_rows,
                    "mapped_stability": _mean_dicts(mapped_rows),
                    "resample_stability": _mean_dicts(resample_rows),
                    "invariants": invariants,
                }
            )
        reconstruction_keys = folds[0]["reconstruction"].keys()
        mean_reconstruction = {
            key: float(np.mean([fold["reconstruction"][key] for fold in folds]))
            for key in reconstruction_keys
        }
        resampled_reconstruction_keys = folds[0]["resampled_reconstruction"].keys()
        mean_resampled_reconstruction = {
            key: float(
                np.mean([fold["resampled_reconstruction"][key] for fold in folds])
            )
            for key in resampled_reconstruction_keys
        }
        mean_mapped = _mean_dicts([fold["mapped_stability"] for fold in folds])
        mean_resample = _mean_dicts([fold["resample_stability"] for fold in folds])
        stats = diagnostics[branch]
        ordering_diagnostics = {
            "signature_tie_rate": float(
                np.average(
                    [row["signature_tie_rate"] for row in stats],
                    weights=[row["patch_count"] for row in stats],
                )
            ),
            "canonical_ambiguity_rate": float(
                np.average(
                    [row["canonical_ambiguity_rate"] for row in stats],
                    weights=[row["patch_count"] for row in stats],
                )
            ),
            "mean_search_leaves": float(
                np.average(
                    [row["mean_search_leaves"] for row in stats],
                    weights=[row["patch_count"] for row in stats],
                )
            ),
            "max_search_leaves": int(max(row["max_search_leaves"] for row in stats)),
        }
        branch_payload[branch] = {
            "folds": folds,
            "mean_reconstruction": mean_reconstruction,
            "mean_resampled_reconstruction": mean_resampled_reconstruction,
            "mean_mapped_stability": mean_mapped,
            "mean_resample_stability": mean_resample,
            "ordering_diagnostics": ordering_diagnostics,
            "all_invariants_passed": all(fold["invariants"]["passed"] for fold in folds),
        }
        print(
            f"branch={branch} vector/code/pooled="
            f"{mean_mapped['patch_vector_row_match']:.4f}/"
            f"{mean_mapped['matched_patch_code_cosine']:.4f}/"
            f"{mean_mapped['graph_embedding_cosine']:.4f} "
            f"observed/full={mean_reconstruction['observed_pair_rmse']:.4f}/"
            f"{mean_reconstruction['full_adjacency_rmse']:.4f} "
            f"resample={mean_resample['graph_embedding_cosine']:.4f}",
            flush=True,
        )

    decision = classify_branches(branch_payload)
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
            "feature_dimension": args.patch_size * (args.patch_size - 1) // 2,
            "labels_used": False,
        },
        "branches": branch_payload,
        "decision": decision,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(f"decision={decision['classification']}")
    print(f"selected={decision['selected_branch']}")
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

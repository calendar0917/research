"""Decompose KSVD instability into global-ID, local-slot, and sampler effects."""
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
from .overlap_cover import PatchCover, _make_cover, _make_patch, patch_budget, remap_cover
from .overlap_stitching import CoverExample, make_cover_example, stack_cover_examples
from .run_overlap_cover_audit import FAMILIES, generate_graph
from .run_patch_relation_representation_audit import EPS, _cosine, _relative_l2, pool_tokens


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/from_scratch/ksvd_id_slot_decomposition_audit_20260802.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/from_scratch/KSVD_ID_SLOT_DECOMPOSITION_AUDIT_20260802.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_ID_SLOT_INSTABILITY_DECOMPOSITION_PROTOCOL_20260802.md"
CONDITIONS = (
    "MAPPED_GLOBAL_RELABEL",
    "FROZEN_SET_SLOT_SHUFFLE",
    "FROZEN_SET_ID_SORT",
    "PATCH_SEQUENCE_SHUFFLE",
    "RELABEL_RESAMPLE",
)


def reorder_cover(
    adjacency: np.ndarray,
    cover: PatchCover,
    orders: Sequence[Sequence[int]],
    *,
    method: str,
) -> PatchCover:
    if len(orders) != len(cover.patches):
        raise ValueError("orders must match patch count")
    patches = []
    for patch, order in zip(cover.patches, orders):
        nodes = tuple(int(node) for node in order)
        if set(nodes) != set(patch.node_ids):
            raise ValueError("reordered patch must preserve its node set")
        patches.append(_make_patch(adjacency, nodes, patch.center))
    return _make_cover(
        method,
        patches,
        cover.segment_ids,
        cover.target_edges,
        cover.bridge_lengths,
    )


def random_slot_cover(
    adjacency: np.ndarray,
    cover: PatchCover,
    seed: int,
) -> PatchCover:
    rng = np.random.default_rng(seed)
    orders = [
        tuple(np.asarray(patch.node_ids)[rng.permutation(len(patch.node_ids))].tolist())
        for patch in cover.patches
    ]
    return reorder_cover(adjacency, cover, orders, method="frozen_set_slot_shuffle")


def sorted_id_cover(adjacency: np.ndarray, cover: PatchCover) -> PatchCover:
    return reorder_cover(
        adjacency,
        cover,
        [tuple(sorted(patch.node_ids)) for patch in cover.patches],
        method="frozen_set_id_sort",
    )


def encode_example(
    example: CoverExample,
    dictionary: np.ndarray,
    train_mean: np.ndarray,
    *,
    sparsity: int,
) -> np.ndarray:
    centered = example.patch_vectors.T - train_mean
    return encode_with_minimum_sparsity(
        centered,
        dictionary,
        sparsity=sparsity,
        minimum_sparsity=1,
    ).T


def _support_jaccard(left: np.ndarray, right: np.ndarray) -> float:
    values = []
    for left_row, right_row in zip(left, right):
        left_support = set(np.flatnonzero(np.abs(left_row) > 1e-10).tolist())
        right_support = set(np.flatnonzero(np.abs(right_row) > 1e-10).tolist())
        union = left_support | right_support
        values.append(len(left_support & right_support) / max(len(union), 1))
    return float(np.mean(values))


def _matched_code_cosine(left: np.ndarray, right: np.ndarray) -> float:
    return float(np.mean([_cosine(a, b) for a, b in zip(left, right)]))


def _vector_row_match(left: CoverExample, right: CoverExample) -> float:
    if left.patch_vectors.shape != right.patch_vectors.shape:
        return 0.0
    return float(np.mean(np.all(left.patch_vectors == right.patch_vectors, axis=1)))


def _condition_metrics(
    base_example: CoverExample,
    changed_example: CoverExample,
    base_codes: np.ndarray,
    changed_codes: np.ndarray,
    *,
    matched_patches: bool,
) -> dict[str, float | None]:
    base_embedding = pool_tokens(base_codes)
    changed_embedding = pool_tokens(changed_codes)
    return {
        "graph_embedding_cosine": _cosine(base_embedding, changed_embedding),
        "graph_embedding_relative_l2": _relative_l2(base_embedding, changed_embedding),
        "patch_vector_row_match": _vector_row_match(base_example, changed_example)
        if matched_patches
        else None,
        "matched_patch_code_cosine": _matched_code_cosine(base_codes, changed_codes)
        if matched_patches
        else None,
        "matched_patch_support_jaccard": _support_jaccard(base_codes, changed_codes)
        if matched_patches
        else None,
    }


def classify(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    folds = tuple(folds)
    means = {}
    for condition in CONDITIONS:
        keys = folds[0]["conditions"][condition].keys()
        means[condition] = {}
        for key in keys:
            values = [fold["conditions"][condition][key] for fold in folds]
            finite_values = [float(value) for value in values if value is not None]
            means[condition][key] = (
                float(np.mean(finite_values)) if finite_values else None
            )
    mapped = float(means["MAPPED_GLOBAL_RELABEL"]["graph_embedding_cosine"])
    slot = float(means["FROZEN_SET_SLOT_SHUFFLE"]["graph_embedding_cosine"])
    resample = float(means["RELABEL_RESAMPLE"]["graph_embedding_cosine"])
    sequence = float(means["PATCH_SEQUENCE_SHUFFLE"]["graph_embedding_cosine"])
    invariant_gate = all(bool(fold["invariants"]["passed"]) for fold in folds)
    replay_gate = mapped >= 0.999 and sequence >= 0.999999
    if not invariant_gate or not replay_gate:
        label = "FAIL_KSVD_INSTABILITY_DECOMPOSITION_INVARIANTS"
    elif slot < 0.90 and resample < 0.90:
        label = "MIXED_SAMPLER_AND_SLOT_INSTABILITY"
    elif slot < 0.90 and resample >= 0.90:
        label = "LOCAL_SLOT_INSTABILITY"
    elif slot >= 0.90 and resample < 0.90:
        label = "SAMPLER_SELECTION_INSTABILITY"
    else:
        label = "KSVD_TOKEN_STABILITY_SUPPORTED"
    return {
        "classification": label,
        "invariant_gate": invariant_gate,
        "replay_gate": replay_gate,
        "local_slot_instability_confirmed": slot < 0.90,
        "sampler_resample_instability_confirmed": resample < 0.90,
        "means": means,
    }


def render_report(payload: dict[str, Any]) -> str:
    decision = payload["decision"]
    lines = [
        "# KSVD ID/slot 不稳定性分解审计",
        "",
        "> 日期：2026-08-02  ",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{decision['classification']}`",
        "",
        "## 1. Matched decomposition",
        "",
        "| condition | graph cosine | relative L2 | vector row match | patch-code cosine | support Jaccard |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition in CONDITIONS:
        values = decision["means"][condition]

        def fmt(key: str) -> str:
            value = values[key]
            return "-" if value is None else f"{float(value):.4f}"

        lines.append(
            f"| {condition} | {fmt('graph_embedding_cosine')} | "
            f"{fmt('graph_embedding_relative_l2')} | {fmt('patch_vector_row_match')} | "
            f"{fmt('matched_patch_code_cosine')} | "
            f"{fmt('matched_patch_support_jaccard')} |"
        )
    lines.extend(
        [
            "",
            "## 2. 结论",
            "",
            f"- mapped/sequence replay gate：`{decision['replay_gate']}`；",
            f"- local slot instability confirmed：`{decision['local_slot_instability_confirmed']}`；",
            f"- relabel-resample instability confirmed：`{decision['sampler_resample_instability_confirmed']}`。",
            "",
            "MAPPED_GLOBAL_RELABEL 只改变 numeric IDs；若它保持 1.0，说明 ID 作为节点身份记账本身不会改变 KSVD。FROZEN_SET_SLOT_SHUFFLE 在完全相同 node sets 上直接测量45D local coordinates 的敏感性。RELABEL_RESAMPLE 再加入 sampler tie-breaking 与 patch-set 变化。",
            "",
        ]
    )
    return "\n".join(lines)


def _mean_conditions(rows: list[dict[str, Any]]) -> dict[str, dict[str, float | None]]:
    result = {}
    for condition in CONDITIONS:
        keys = rows[0][condition].keys()
        result[condition] = {}
        for key in keys:
            values = [row[condition][key] for row in rows if row[condition][key] is not None]
            result[condition][key] = float(np.mean(values)) if values else None
    return result


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
        graph_rows = []
        for example in test_examples:
            base_codes = encode_example(
                example, dictionary, train_mean, sparsity=args.sparsity
            )
            permutation = np.random.default_rng(
                np.random.SeedSequence(
                    [args.graph_bank_seed, example.graph_index, 9901]
                ).generate_state(1, dtype=np.uint32)[0]
            ).permutation(args.n_nodes)
            inverse = np.empty(args.n_nodes, dtype=np.int64)
            inverse[permutation] = np.arange(args.n_nodes)
            relabeled_adjacency = example.adjacency[np.ix_(permutation, permutation)]

            mapped_cover = remap_cover(example.cover, inverse, relabeled_adjacency)
            mapped_example = make_cover_example(
                example.graph_index,
                example.family,
                example.target_degree,
                relabeled_adjacency,
                mapped_cover,
            )
            mapped_codes = encode_example(
                mapped_example, dictionary, train_mean, sparsity=args.sparsity
            )

            slot_cover = random_slot_cover(
                example.adjacency,
                example.cover,
                int(
                    np.random.SeedSequence(
                        [args.graph_bank_seed, example.graph_index, 9902]
                    ).generate_state(1, dtype=np.uint32)[0]
                ),
            )
            slot_example = make_cover_example(
                example.graph_index,
                example.family,
                example.target_degree,
                example.adjacency,
                slot_cover,
            )
            slot_codes = encode_example(
                slot_example, dictionary, train_mean, sparsity=args.sparsity
            )

            sorted_original_cover = sorted_id_cover(example.adjacency, example.cover)
            sorted_original_example = make_cover_example(
                example.graph_index,
                example.family,
                example.target_degree,
                example.adjacency,
                sorted_original_cover,
            )
            sorted_mapped_cover = sorted_id_cover(relabeled_adjacency, mapped_cover)
            sorted_mapped_example = make_cover_example(
                example.graph_index,
                example.family,
                example.target_degree,
                relabeled_adjacency,
                sorted_mapped_cover,
            )
            sorted_original_codes = encode_example(
                sorted_original_example, dictionary, train_mean, sparsity=args.sparsity
            )
            sorted_mapped_codes = encode_example(
                sorted_mapped_example, dictionary, train_mean, sparsity=args.sparsity
            )

            resampled_cover = sample_marginal_candidate_cover(
                relabeled_adjacency,
                np.random.default_rng(sampler_seeds[example.graph_index]),
                n_patches=len(example.cover.patches),
                patch_size=args.patch_size,
                target_overlap=args.target_overlap,
                retained_beam=args.retained_beam,
                candidate_restarts=args.candidate_restarts,
            )
            resampled_example = make_cover_example(
                example.graph_index,
                example.family,
                example.target_degree,
                relabeled_adjacency,
                resampled_cover,
            )
            resampled_codes = encode_example(
                resampled_example, dictionary, train_mean, sparsity=args.sparsity
            )

            sequence_permutation = np.random.default_rng(
                np.random.SeedSequence(
                    [args.graph_bank_seed, example.graph_index, 9903]
                ).generate_state(1, dtype=np.uint32)[0]
            ).permutation(base_codes.shape[0])
            sequence_codes = base_codes[sequence_permutation]
            row = {
                "graph_index": example.graph_index,
                "MAPPED_GLOBAL_RELABEL": _condition_metrics(
                    example,
                    mapped_example,
                    base_codes,
                    mapped_codes,
                    matched_patches=True,
                ),
                "FROZEN_SET_SLOT_SHUFFLE": _condition_metrics(
                    example,
                    slot_example,
                    base_codes,
                    slot_codes,
                    matched_patches=True,
                ),
                "FROZEN_SET_ID_SORT": _condition_metrics(
                    sorted_original_example,
                    sorted_mapped_example,
                    sorted_original_codes,
                    sorted_mapped_codes,
                    matched_patches=True,
                ),
                "PATCH_SEQUENCE_SHUFFLE": {
                    "graph_embedding_cosine": _cosine(
                        pool_tokens(base_codes), pool_tokens(sequence_codes)
                    ),
                    "graph_embedding_relative_l2": _relative_l2(
                        pool_tokens(base_codes), pool_tokens(sequence_codes)
                    ),
                    "patch_vector_row_match": None,
                    "matched_patch_code_cosine": None,
                    "matched_patch_support_jaccard": None,
                },
                "RELABEL_RESAMPLE": _condition_metrics(
                    example,
                    resampled_example,
                    base_codes,
                    resampled_codes,
                    matched_patches=False,
                ),
            }
            graph_rows.append(row)
        conditions = _mean_conditions(graph_rows)
        invariants = {
            "train_test_graph_isolation": not (
                {example.graph_index for example in train_examples} & test_ids
            ),
            "finite": all(
                np.isfinite(float(value))
                for condition in conditions.values()
                for value in condition.values()
                if value is not None
            ),
            "mapped_vector_replay": conditions["MAPPED_GLOBAL_RELABEL"][
                "patch_vector_row_match"
            ]
            == 1.0,
            "sequence_replay": conditions["PATCH_SEQUENCE_SHUFFLE"][
                "graph_embedding_cosine"
            ]
            >= 0.999999,
        }
        invariants["passed"] = all(invariants.values())
        folds.append(
            {
                "fold_index": fold_index,
                "train_graph_count": len(train_examples),
                "test_graph_count": len(test_examples),
                "initialization": initialization,
                "training_reconstruction_curve": [
                    float(value) for value in training_info["recon_curve"]
                ],
                "conditions": conditions,
                "graphs": graph_rows,
                "invariants": invariants,
            }
        )
        print(
            f"fold={fold_index} mapped/slot/id-sort/resample="
            f"{conditions['MAPPED_GLOBAL_RELABEL']['graph_embedding_cosine']:.4f}/"
            f"{conditions['FROZEN_SET_SLOT_SHUFFLE']['graph_embedding_cosine']:.4f}/"
            f"{conditions['FROZEN_SET_ID_SORT']['graph_embedding_cosine']:.4f}/"
            f"{conditions['RELABEL_RESAMPLE']['graph_embedding_cosine']:.4f}",
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

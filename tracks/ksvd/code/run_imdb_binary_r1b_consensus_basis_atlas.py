#!/usr/bin/env python3
"""Build a descriptive grouped-fold consensus atlas from registered R0-D atoms."""
from __future__ import annotations

import argparse
import json
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np

from .from_scratch_unplanted_dictionary import exact_maximum_cosine_assignment
from .imdb_walk_dictionary import encode_with_minimum_sparsity, grouped_isomorphism_folds, stack_patch_graphs
from .imdb_walk_substrate import exact_isomorphism_groups, extract_walk_patch_graphs, load_tu_structure_text
from .from_scratch_unplanted_representation import upper_triangle_edges


DEFAULT_DATASET = Path("data/TUD/IMDB-BINARY")
DEFAULT_R0D = Path("tracks/ksvd/results/from_scratch/imdb_binary_r0d_dictionary_audit_20260731.json")
DEFAULT_JSON = Path("tracks/ksvd/results/from_scratch/imdb_binary_r1b_consensus_basis_atlas_20260801.json")
DEFAULT_REPORT = Path("tracks/ksvd/results/from_scratch/IMDB_BINARY_R1B_CONSENSUS_BASIS_ATLAS_20260801.md")


def _key(vector: np.ndarray) -> tuple[int, ...]:
    return tuple(int(value) for value in np.rint(vector).astype(np.int8))


def _effective_count(counts: list[int]) -> float:
    values = np.asarray(counts, dtype=np.float64)
    probabilities = values / np.sum(values)
    return float(np.exp(-np.sum(probabilities * np.log(probabilities))))


def _edge_list(vector: tuple[int, ...]) -> list[list[int]]:
    return [
        [int(left), int(right)]
        for value, (left, right) in zip(vector, upper_triangle_edges(7))
        if value
    ]


def _fold_patch_metadata(examples, split):
    by_index = {int(example.graph_index): example for example in examples}
    items = tuple(by_index[index] for index in split.test_indices)
    raw, slices = stack_patch_graphs(items)
    walk = np.concatenate([item.walk_vectors for item in items], axis=0)
    canonical = np.concatenate([item.canonical_vectors for item in items], axis=0)
    edges = np.concatenate([item.edge_counts for item in items]).astype(np.int64)
    graph_indices = np.empty(raw.shape[1], dtype=np.int64)
    labels = np.empty(raw.shape[1], dtype=np.int64)
    for graph_index, label, start, stop in slices:
        graph_indices[start:stop] = int(graph_index)
        labels[start:stop] = int(label)
    return raw, walk, canonical, edges, graph_indices, labels


def build_atlas(r0d: dict[str, Any], graphs, examples) -> dict[str, Any]:
    groups = exact_isomorphism_groups(graphs)
    splits = grouped_isomorphism_folds(graphs, groups, n_splits=5, seed=731301)
    source_folds = {
        int(item["fold_index"]): item
        for item in r0d["views"]["exact_isomorphism_grouped"]["folds"]
    }
    dictionaries = {
        index: np.asarray(source_folds[index]["dictionaries"]["final"], dtype=np.float64)
        for index in range(5)
    }
    reference = dictionaries[0]
    alignments = {}
    aligned_atoms = np.empty((5, reference.shape[0], reference.shape[1]), dtype=np.float64)
    aligned_atoms[0] = reference
    alignments[0] = {
        "assignment": list(range(12)),
        "cosines": [1.0] * 12,
        "signs": [1.0] * 12,
    }
    for fold_index in range(1, 5):
        current = dictionaries[fold_index]
        cosine = np.abs(reference.T @ current)
        _score, assignment = exact_maximum_cosine_assignment(cosine)
        values = []
        signs = []
        for ref_atom, current_atom in enumerate(assignment):
            dot = float(reference[:, ref_atom] @ current[:, current_atom])
            sign = 1.0 if dot >= 0.0 else -1.0
            aligned_atoms[fold_index, :, ref_atom] = sign * current[:, current_atom]
            values.append(float(abs(dot)))
            signs.append(sign)
        alignments[fold_index] = {
            "assignment": list(assignment),
            "cosines": values,
            "signs": signs,
        }

    consensus = np.mean(aligned_atoms, axis=0)
    consensus /= np.maximum(np.linalg.norm(consensus, axis=0, keepdims=True), 1e-12)
    exemplars: list[list[dict[str, Any]]] = [[] for _ in range(12)]
    for split in splits:
        fold_index = split.fold_index
        source = source_folds[fold_index]
        dictionary = dictionaries[fold_index]
        raw, walk, canonical, edges, graph_indices, labels = _fold_patch_metadata(examples, split)
        mean = np.asarray(source["train_coordinate_mean"], dtype=np.float64).reshape(-1, 1)
        centered = raw - mean
        codes = encode_with_minimum_sparsity(
            centered, dictionary, sparsity=2, minimum_sparsity=1
        )
        assignment = alignments[fold_index]["assignment"]
        for reference_atom, current_atom in enumerate(assignment):
            order = np.argsort(-np.abs(codes[current_atom]), kind="stable")[:5]
            for patch_index in order:
                exemplars[reference_atom].append(
                    {
                        "fold_index": int(fold_index),
                        "source_atom_index": int(current_atom),
                        "graph_index": int(graph_indices[patch_index]),
                        "graph_label_descriptive_only": int(labels[patch_index]),
                        "absolute_coefficient": float(abs(codes[current_atom, patch_index])),
                        "edge_count": int(edges[patch_index]),
                        "walk_vector": list(_key(walk[patch_index])),
                        "canonical_vector": list(_key(canonical[patch_index])),
                    }
                )

    atoms = []
    for atom_index, rows in enumerate(exemplars):
        canonical_counter = Counter(tuple(row["canonical_vector"]) for row in rows)
        walk_counter = Counter(tuple(row["walk_vector"]) for row in rows)
        edge_counter = Counter(int(row["edge_count"]) for row in rows)
        label_counter = Counter(int(row["graph_label_descriptive_only"]) for row in rows)
        graph_count = len({int(row["graph_index"]) for row in rows})
        canonical_ranked = sorted(
            canonical_counter.items(), key=lambda item: (-item[1], item[0])
        )
        other_fold_cosines = [
            alignments[fold_index]["cosines"][atom_index] for fold_index in range(1, 5)
        ]
        atom = consensus[:, atom_index]
        l1 = float(np.sum(np.abs(atom)))
        atoms.append(
            {
                "consensus_atom_index": int(atom_index),
                "reference_fold": 0,
                "mean_reference_matched_cosine_other_folds": float(np.mean(other_fold_cosines)),
                "minimum_reference_matched_cosine_other_folds": float(np.min(other_fold_cosines)),
                "matched_cosines_other_folds": other_fold_cosines,
                "exemplar_count": int(len(rows)),
                "exemplar_unique_graph_count": int(graph_count),
                "edge_count_mean": float(np.mean([row["edge_count"] for row in rows])),
                "edge_count_std": float(np.std([row["edge_count"] for row in rows], ddof=0)),
                "edge_count_histogram": {str(key): int(value) for key, value in sorted(edge_counter.items())},
                "unique_walk_vector_count": int(len(walk_counter)),
                "unique_canonical_signature_count": int(len(canonical_counter)),
                "canonical_effective_count": _effective_count(list(canonical_counter.values())),
                "dominant_canonical_mass": float(canonical_ranked[0][1] / len(rows)),
                "descriptive_label_counts": {str(key): int(value) for key, value in sorted(label_counter.items())},
                "mean_absolute_coefficient": float(np.mean([row["absolute_coefficient"] for row in rows])),
                "continuous_consensus_atom": {
                    "coordinates": atom.tolist(),
                    "l1_mass": l1,
                    "positive_mass_fraction": float(np.sum(np.clip(atom, 0.0, None)) / max(l1, 1e-12)),
                    "maximum_absolute_coordinate": float(np.max(np.abs(atom))),
                },
                "top_canonical_representatives": [
                    {
                        "count": int(count),
                        "mass": float(count / len(rows)),
                        "edge_count": int(sum(vector)),
                        "edge_list": _edge_list(vector),
                        "canonical_vector": list(vector),
                    }
                    for vector, count in canonical_ranked[:3]
                ],
                "exemplars": rows,
            }
        )
    return {
        "reference_fold": 0,
        "alignment_method": "exact_maximum_absolute_cosine_to_fixed_grouped_fold_0",
        "top_k_per_fold_atom": 5,
        "alignment": {str(key): value for key, value in alignments.items()},
        "atoms": atoms,
        "summary": {
            "atom_count": 12,
            "mean_reference_matched_cosine": float(
                np.mean([atom["mean_reference_matched_cosine_other_folds"] for atom in atoms])
            ),
            "minimum_atom_reference_matched_cosine": float(
                np.min([atom["minimum_reference_matched_cosine_other_folds"] for atom in atoms])
            ),
            "mean_unique_canonical_signatures": float(
                np.mean([atom["unique_canonical_signature_count"] for atom in atoms])
            ),
            "mean_canonical_effective_count": float(
                np.mean([atom["canonical_effective_count"] for atom in atoms])
            ),
            "mean_dominant_canonical_mass": float(
                np.mean([atom["dominant_canonical_mass"] for atom in atoms])
            ),
            "edge_count_mean_range": [
                float(np.min([atom["edge_count_mean"] for atom in atoms])),
                float(np.max([atom["edge_count_mean"] for atom in atoms])),
            ],
        },
    }


def _fmt(value: float) -> str:
    return f"{value:.3f}"


def render_report(payload: dict[str, Any]) -> str:
    atlas = payload["atlas"]
    summary = atlas["summary"]
    lines = [
        "# IMDB-BINARY R1-B grouped consensus basis atlas",
        "",
        "> 日期：2026-08-01  ",
        "> 性质：label-free descriptive atlas；无 PASS/FAIL gate  ",
        "> 协议：`tracks/ksvd/docs/KSVD_IMDB_BINARY_R1B_CONSENSUS_BASIS_ATLAS_PROTOCOL_20260801.md`",
        "",
        "## 1. Construction",
        "",
        "- 固定 grouped fold 0 为 reference，不按结果选择。",
        "- 其余 folds 用 exact maximum absolute cosine assignment 对齐，并按 dot-product 对齐 sign。",
        "- 每 fold 每 atom 只取 grouped held-out patches 的 top-5 absolute activations；每个 consensus atom 共 25 exemplars。",
        "- labels 只作 descriptive metadata，不参与匹配、排序或命名。",
        "",
        "## 2. Atlas summary",
        "",
        f"- atom count：`{summary['atom_count']}`；",
        f"- mean reference-matched cosine：`{_fmt(summary['mean_reference_matched_cosine'])}`；",
        f"- worst atom/fold matched cosine：`{_fmt(summary['minimum_atom_reference_matched_cosine'])}`；",
        f"- mean unique canonical signatures / 25 exemplars：`{_fmt(summary['mean_unique_canonical_signatures'])}`；",
        f"- mean canonical effective count：`{_fmt(summary['mean_canonical_effective_count'])}`；",
        f"- mean dominant canonical mass：`{_fmt(summary['mean_dominant_canonical_mass'])}`；",
        f"- atom exemplar edge-count mean range：`{_fmt(summary['edge_count_mean_range'][0])}..{_fmt(summary['edge_count_mean_range'][1])}`。",
        "",
        "| atom | match mean/min | graphs | edge mean±std | unique WALK/canonical | effective canonical | dominant mass | labels descriptive |",
        "|---:|---:|---:|---:|---:|---:|---:|---|",
    ]
    for atom in atlas["atoms"]:
        lines.append(
            f"| {atom['consensus_atom_index']} | "
            f"{_fmt(atom['mean_reference_matched_cosine_other_folds'])}/{_fmt(atom['minimum_reference_matched_cosine_other_folds'])} | "
            f"{atom['exemplar_unique_graph_count']} | {_fmt(atom['edge_count_mean'])}±{_fmt(atom['edge_count_std'])} | "
            f"{atom['unique_walk_vector_count']}/{atom['unique_canonical_signature_count']} | "
            f"{_fmt(atom['canonical_effective_count'])} | {_fmt(atom['dominant_canonical_mass'])} | "
            f"{atom['descriptive_label_counts']} |"
        )
    lines.extend(["", "## 3. Canonical representatives", ""])
    for atom in atlas["atoms"]:
        lines.extend([
            f"### Atom {atom['consensus_atom_index']}",
            "",
            f"edge histogram：`{atom['edge_count_histogram']}`；continuous atom max coordinate：`{_fmt(atom['continuous_consensus_atom']['maximum_absolute_coordinate'])}`。",
            "",
        ])
        for rank, representative in enumerate(atom["top_canonical_representatives"], start=1):
            lines.append(
                f"- representative {rank}：mass `{_fmt(representative['mass'])}`，edges `{representative['edge_count']}`，edge list `{representative['edge_list']}`"
            )
        lines.append("")
    lines.extend([
        "## 4. Interpretation boundary",
        "",
        "这些 atoms 是 continuous latent basis components。代表 patch 是 post-hoc exemplars，不表示 atom 本身等于某一张合法离散图；label composition 也不能作为 task utility 证据。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--r0d-json", type=Path, default=DEFAULT_R0D)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    r0d = json.loads(args.r0d_json.read_text(encoding="utf-8"))
    if r0d["decision"]["classification"] != "PASS_R0D_REAL_DICTIONARY_OPTIMIZATION":
        raise ValueError("R1-B requires the registered passing R0-D artifact")
    graphs = load_tu_structure_text(args.dataset_root, cleaned=False)
    examples = extract_walk_patch_graphs(
        graphs, sampling_seed=20260731, patch_size=7, max_patches_per_graph=24
    )
    payload = {
        "experiment": "IMDB_BINARY_R1B_GROUPED_CONSENSUS_BASIS_ATLAS",
        "date": "2026-08-01",
        "dataset": {"name": "IMDB-BINARY", "variant": "raw", "graph_count": len(graphs)},
        "source_r0d_json": str(args.r0d_json),
        "labels_used_for_selection_or_gate": False,
        "classification": "DESCRIPTIVE_NO_GATE",
        "atlas": build_atlas(r0d, graphs, examples),
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload["atlas"]["summary"], indent=2, ensure_ascii=False))
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

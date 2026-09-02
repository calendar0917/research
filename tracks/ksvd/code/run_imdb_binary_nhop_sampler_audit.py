#!/usr/bin/env python3
"""Run the registered pre-KSVD direct n-hop feasibility audit on raw IMDB."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np

from .from_scratch_unplanted_signal import evaluate_feature_matrices
from .imdb_nhop_sampler import (
    SELECTOR_MODES,
    direct_ego_size_feasibility,
    extract_nhop_patch_graphs,
    nhop_feature_matrices,
    nhop_substrate_summary,
    relabel_selector_audit,
)
from .imdb_walk_substrate import (
    extract_walk_patch_graphs,
    load_tu_structure_text,
    patch_substrate_summary,
)
from .run_imdb_binary_r0p_audit import _stratified_folds, _train_validation_split


DEFAULT_DATASET = Path("data/TUD/IMDB-BINARY")
DEFAULT_JSON = Path(
    "tracks/ksvd/results/from_scratch/imdb_binary_nhop_sampler_audit_20260801.json"
)
DEFAULT_REPORT = Path(
    "tracks/ksvd/results/from_scratch/IMDB_BINARY_NHOP_SAMPLER_AUDIT_20260801.md"
)
SPLIT_SEEDS = (731201, 731202, 731203)


def evaluate_signal(
    matrices: dict[str, np.ndarray],
    labels: np.ndarray,
    *,
    split_seeds: tuple[int, ...] = SPLIT_SEEDS,
) -> dict[str, Any]:
    seed_results = []
    for split_seed in split_seeds:
        folds = _stratified_folds(labels, 5, split_seed)
        fold_results = []
        for fold_index, test_indices in enumerate(folds):
            outer_train = np.setdiff1d(
                np.arange(labels.size), test_indices, assume_unique=True
            )
            train_indices, validation_indices = _train_validation_split(
                outer_train, labels, split_seed * 101 + fold_index
            )
            evaluations = {}
            for key, values in matrices.items():
                evaluations[key] = evaluate_feature_matrices(
                    {
                        "train": values[train_indices],
                        "validation": values[validation_indices],
                        "test": values[test_indices],
                    },
                    {
                        "train": labels[train_indices],
                        "validation": labels[validation_indices],
                        "test": labels[test_indices],
                    },
                    feature_key=key,
                )
            fold_results.append(
                {"fold_index": int(fold_index), "evaluations": evaluations}
            )
        seed_results.append(
            {"split_seed": int(split_seed), "folds": fold_results}
        )
    summary = {}
    for key in matrices:
        per_seed = []
        all_scores = []
        for seed_result in seed_results:
            scores = [
                fold["evaluations"][key]["test_balanced_accuracy"]
                for fold in seed_result["folds"]
            ]
            per_seed.append(float(np.mean(scores)))
            all_scores.extend(scores)
        summary[key] = {
            "per_split_seed_mean_test_balanced_accuracy": per_seed,
            "mean_test_balanced_accuracy": float(np.mean(per_seed)),
            "std_across_split_seed_means": float(np.std(per_seed, ddof=0)),
            "minimum_fold_test_balanced_accuracy": float(np.min(all_scores)),
            "maximum_fold_test_balanced_accuracy": float(np.max(all_scores)),
        }
    return {"summary": summary, "seed_results": seed_results}


def classify(payload: dict[str, Any]) -> dict[str, Any]:
    walk = payload["walk_reference"]
    signal = payload["signal"]["summary"]
    stats_score = signal["stats"]["mean_test_balanced_accuracy"]
    walk_score = signal["walk_canonical_mean_std"]["mean_test_balanced_accuracy"]
    mode_results = {}
    for mode in SELECTOR_MODES:
        substrate = payload["nhop_substrate"][mode]
        invariance = payload["relabel_audit"]["modes"][mode]
        score = signal[f"{mode}_canonical_mean_std"]["mean_test_balanced_accuracy"]
        conditional = signal[
            f"stats_plus_{mode}_canonical_mean_std"
        ]["mean_test_balanced_accuracy"]
        checks = {
            "canonical_relabel_match_ge_0_99": bool(
                invariance["rooted_canonical_vector_exact_match_rate"] >= 0.99
            ),
            "dominant_mass_no_worse_than_walk": bool(
                substrate["dominant_canonical_signature_fraction"]
                <= walk["dominant_canonical_signature_fraction"]
            ),
            "effective_count_no_worse_than_walk": bool(
                substrate["canonical_effective_signature_count"]
                >= walk["canonical_effective_signature_count"]
            ),
            "standalone_ba_exceeds_walk_by_0_02": bool(score - walk_score >= 0.02),
            "conditional_not_below_stats": bool(conditional >= stats_score),
        }
        mode_results[mode] = {
            "checks": checks,
            "passes_worth_dictionary_gate": bool(all(checks.values())),
            "standalone_gain_over_walk": float(score - walk_score),
            "conditional_gain_over_stats": float(conditional - stats_score),
        }
    passed = [
        mode for mode, result in mode_results.items()
        if result["passes_worth_dictionary_gate"]
    ]
    if passed:
        classification = "PASS_NHOP_SAMPLER_WORTH_DICTIONARY_AUDIT"
        next_step = (
            "Freeze one new formal protocol without choosing among modes by test score; "
            "resolve any selector ambiguity before training KSVD."
        )
    else:
        classification = "FAIL_DIRECT_CAPPED_NHOP_SAMPLER"
        next_step = (
            "Do not train n-hop KSVD. Direct capped ego selection did not jointly solve "
            "invariance, substrate diversity, and conditional signal."
        )
    return {
        "classification": classification,
        "passes_any_mode": bool(passed),
        "passing_modes": passed,
        "modes": mode_results,
        "next_step": next_step,
        "note": "Exploratory matched comparison using already-seen R0-P split seeds.",
    }


def _f(value: float) -> str:
    return f"{value:.4f}"


def render_report(payload: dict[str, Any]) -> str:
    feasibility = payload["direct_ego_size_feasibility"]
    signal = payload["signal"]["summary"]
    decision = payload["decision"]
    lines = [
        "# IMDB-BINARY direct n-hop sampler feasibility audit",
        "",
        "> 日期：2026-08-01  ",
        "> 性质：pre-KSVD sampler/ordering diagnostic  ",
        "> 协议：`tracks/ksvd/docs/KSVD_IMDB_NHOP_SAMPLER_AUDIT_PROTOCOL_20260801.md`",
        "",
        "## 1. Direct ego size feasibility",
        "",
        "| radius | min/median/mean/p90/p95/max | <7 | =7 | >7 | equals whole graph |",
        "|---:|---|---:|---:|---:|---:|",
    ]
    for radius in ("1", "2"):
        row = feasibility["radii"][radius]
        size = row["size"]
        lines.append(
            f"| {radius} | {size['minimum']}/{size['median']:.1f}/{size['mean']:.2f}/"
            f"{size['p90']:.1f}/{size['p95']:.1f}/{size['maximum']} | "
            f"{_f(row['fraction_smaller_than_patch_size'])} | "
            f"{_f(row['fraction_equal_to_patch_size'])} | "
            f"{_f(row['fraction_larger_than_patch_size'])} | "
            f"{_f(row['fraction_equal_to_whole_graph'])} |"
        )
    lines.extend([
        "",
        "`radius=1` 若不截断就不是固定 7-node signal；`radius=2` 若接近整图，就不再是 local patch。",
        "",
        "## 2. Selector substrate and relabel audit",
        "",
        "`set match` 检查抽象节点集合；`ranked match` 检查排序邻接；`canonical match` 检查 exact rooted canonical 输出。多排序不自动保证这些指标。",
        "",
        "| selector | cutoff tie | set match | ranked match | canonical match | clique mass | dominant mass | effective count | within-graph unique median |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ])
    for mode in SELECTOR_MODES:
        substrate = payload["nhop_substrate"][mode]
        audit = payload["relabel_audit"]["modes"][mode]
        lines.append(
            f"| {mode} | {_f(substrate['cutoff_tie_fraction'])} | "
            f"{_f(audit['selected_abstract_set_exact_match_rate'])} | "
            f"{_f(audit['ranked_order_vector_exact_match_rate'])} | "
            f"{_f(audit['rooted_canonical_vector_exact_match_rate'])} | "
            f"{_f(substrate['clique_fraction'])} | "
            f"{_f(substrate['dominant_canonical_signature_fraction'])} | "
            f"{_f(substrate['canonical_effective_signature_count'])} | "
            f"{_f(substrate['within_graph_canonical_unique_fraction']['median'])} |"
        )
    walk = payload["walk_reference"]
    lines.extend([
        "",
        "WALK s=7 reference："
        f"clique/dominant mass `{_f(walk['clique_fraction'])}`，"
        f"effective count `{_f(walk['canonical_effective_signature_count'])}`，"
        f"within-graph unique median `{_f(walk['within_graph_canonical_unique_fraction']['median'])}`。",
        "",
        "## 3. Matched raw-stratified signal exposure",
        "",
        "使用与 R0-P 相同且已经看过的 split seeds，因此只用于 matched diagnosis。",
        "",
        "| feature | mean BA | split-seed std | per-seed means |",
        "|---|---:|---:|---|",
    ])
    order = ["stats", "walk_canonical_mean_std", "stats_plus_walk_canonical_mean_std"]
    for mode in SELECTOR_MODES:
        order.extend(
            [f"{mode}_canonical_mean_std", f"stats_plus_{mode}_canonical_mean_std"]
        )
    for key in order:
        row = signal[key]
        per_seed = ", ".join(
            _f(value) for value in row["per_split_seed_mean_test_balanced_accuracy"]
        )
        lines.append(
            f"| {key} | {_f(row['mean_test_balanced_accuracy'])} | "
            f"{_f(row['std_across_split_seed_means'])} | [{per_seed}] |"
        )
    lines.extend([
        "",
        "## 4. Frozen worth-dictionary checks",
        "",
    ])
    for mode in SELECTOR_MODES:
        result = decision["modes"][mode]
        lines.append(f"### {mode}")
        lines.append("")
        for key, value in result["checks"].items():
            lines.append(f"- [{'x' if value else ' '}] `{key}`")
        lines.extend([
            f"- standalone gain over WALK：`{_f(result['standalone_gain_over_walk'])}`；",
            f"- conditional gain over STATS：`{_f(result['conditional_gain_over_stats'])}`；",
            f"- worth dictionary gate：`{'PASS' if result['passes_worth_dictionary_gate'] else 'FAIL'}`。",
            "",
        ])
    lines.extend([
        "## 5. Decision",
        "",
        f"> **{decision['classification']}**",
        "",
        decision["next_step"],
        "",
        "本结果不说明所有 n-hop 表示都失败；它只判断最直接的 fixed-7 capped ego route 是否比当前 WALK 更适合继续。",
    ])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    started = time.perf_counter()
    graphs = load_tu_structure_text(args.dataset_root, cleaned=False)
    print(f"loaded raw graphs={len(graphs)}", flush=True)
    feasibility = direct_ego_size_feasibility(graphs, patch_size=7)
    print("computed direct ego size feasibility", flush=True)
    nhop_examples = extract_nhop_patch_graphs(
        graphs,
        sampling_seed=20260731,
        patch_size=7,
        max_patches_per_graph=24,
    )
    nhop_substrate = nhop_substrate_summary(nhop_examples, patch_size=7)
    relabel_audit = relabel_selector_audit(
        graphs,
        nhop_examples,
        seed=20260801,
        graph_limit=100,
        permutations_per_graph=3,
        patch_size=7,
    )
    print("computed n-hop patches and relabel audit", flush=True)

    walk_examples = extract_walk_patch_graphs(
        graphs,
        sampling_seed=20260731,
        patch_size=7,
        max_patches_per_graph=24,
    )
    walk_reference = patch_substrate_summary(walk_examples)
    matrices, labels = nhop_feature_matrices(nhop_examples)
    walk_canonical = np.stack(
        [example.features["canonical_mean_std"] for example in walk_examples], axis=0
    )
    matrices["walk_canonical_mean_std"] = walk_canonical
    matrices["stats_plus_walk_canonical_mean_std"] = np.column_stack(
        [matrices["stats"], walk_canonical]
    )
    signal = evaluate_signal(matrices, labels)
    print("computed matched signal exposure", flush=True)

    payload = {
        "experiment": "IMDB_BINARY_DIRECT_NHOP_SAMPLER_AUDIT",
        "date": "2026-08-01",
        "dataset": {"name": "IMDB-BINARY", "variant": "raw", "graph_count": len(graphs)},
        "config": {
            "patch_size": 7,
            "patch_dimension": 21,
            "max_patches_per_graph": 24,
            "root_sampling_seed": 20260731,
            "relabel_seed": 20260801,
            "relabel_graph_limit": 100,
            "permutations_per_graph": 3,
            "split_seeds": list(SPLIT_SEEDS),
            "selector_modes": list(SELECTOR_MODES),
            "ksvd_trained": False,
        },
        "direct_ego_size_feasibility": feasibility,
        "walk_reference": walk_reference,
        "nhop_substrate": nhop_substrate,
        "relabel_audit": relabel_audit,
        "signal": signal,
        "runtime_seconds": float(time.perf_counter() - started),
    }
    payload["decision"] = classify(payload)
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(json.dumps(payload["decision"], ensure_ascii=False, indent=2))
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

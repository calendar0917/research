"""Probe whether graph-observable or externally supplied roots repair G0B-R."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .from_scratch_representation_gate import (
    evaluate_representation_condition,
    make_add_one_noncore_variants,
    make_edge_flip_variants,
    transform_variants_oracle_root,
    transform_variants_structural_max_degree_root,
)


DEFAULT_JSON = Path("tracks/ksvd/results/from_scratch/g0b_root_repair_probe_20260731.json")
DEFAULT_REPORT = Path("tracks/ksvd/results/from_scratch/G0B_ROOT_REPAIR_PROBE_20260731.md")


def compact(result: dict[str, Any]) -> dict[str, Any]:
    return {
        "condition": result["condition"],
        "variant_count": result["variant_count"],
        "motif_variant_counts": result["motif_variant_counts"],
        "label_identifiability": result["label_identifiability"],
        "core_coordinate_consistency": result["core_coordinate_consistency"],
        "prototype_separability": result["prototype_separability"],
        "gate_checks": result["gate_checks"],
        "gate_passed": result["gate_passed"],
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# G0B-R Root Repair Probe",
        "",
        "> 日期：2026-07-31  ",
        "> 目的：区分“纯 canonical adjacency 的问题能否由结构性 root 修复”与“必须额外提供稳定 anchor 信息”。不运行 KSVD。",
        "",
        "## 1. 汇总",
        "",
        "| nuisance | representation | collision mass | Bayes accuracy | robust core F1 | medoid accuracy | gate |",
        "|---|---|---:|---:|---:|---:|---:|",
    ]
    for family in payload["families"]:
        for representation, result in family["representations"].items():
            lines.append(
                f"| {family['nuisance']} | {representation} | "
                f"{result['label_identifiability']['cross_family_collision_mass']:.4f} | "
                f"{result['label_identifiability']['equal_prior_bayes_accuracy']:.4f} | "
                f"{result['core_coordinate_consistency']['minimum_best_robust_fixed_support_mean_f1']:.4f} | "
                f"{result['prototype_separability']['nearest_medoid_macro_accuracy']:.4f} | "
                f"{'PASS' if result['gate_passed'] else 'FAIL'} |"
            )
    lines.extend([
        "",
        "表示条件：",
        "",
        "- `unrooted`：原始 exact canonical adjacency；",
        "- `structural_max_degree_root`：只用 observed patch 的最大度节点作为 root；最大度 tie 仍做置换不变最小化，不增加外部信息；",
        "- `oracle_root0`：生成器 node 0 作为可见 root，模拟 patch extractor 提供稳定 anchor。它不是 motif label，但属于额外 side information。",
        "",
        "## 2. 每种 motif 的 robust fixed-core F1",
        "",
        "| nuisance | representation | triangle | four_cycle | three_star | five_path |",
        "|---|---|---:|---:|---:|---:|",
    ])
    for family in payload["families"]:
        for representation, result in family["representations"].items():
            values = [item["best_robust_fixed_support_mean_f1"] for item in result["core_coordinate_consistency"]["families"]]
            lines.append(
                f"| {family['nuisance']} | {representation} | "
                + " | ".join(f"{value:.4f}" for value in values)
                + " |"
            )
    lines.extend([
        "",
        "## 3. 判定逻辑",
        "",
        "1. 如果 `structural_max_degree_root` 修复，则可以继续研究无需额外语义的 rooted structural patch；",
        "2. 如果只有 `oracle_root0` 修复，则 route 需要一个由采样器、节点类型或领域属性提供的稳定 anchor；",
        "3. 如果 oracle root 仍失败，则单个 root 不足以定义跨 variant 的固定 adjacency motif atom；应改用 typed representation、invariant statistics 或弱化 atom 解释。",
        "",
        "## 4. 当前结论",
        "",
        payload["decision"]["interpretation"],
        "",
    ])
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flip-seed", type=int, default=20260731)
    parser.add_argument("--flip-samples-per-motif", type=int, default=2000)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    base_families = {
        "R1_add_one_noncore_exhaustive": make_add_one_noncore_variants(),
        "R2_edge_flip_p005": make_edge_flip_variants(seed=args.flip_seed, samples_per_motif=args.flip_samples_per_motif),
    }
    families = []
    for nuisance, variants in base_families.items():
        representations = {
            "unrooted": compact(evaluate_representation_condition(variants)),
            "structural_max_degree_root": compact(evaluate_representation_condition(transform_variants_structural_max_degree_root(variants))),
            "oracle_root0": compact(evaluate_representation_condition(transform_variants_oracle_root(variants))),
        }
        families.append({"nuisance": nuisance, "representations": representations})
        for name, result in representations.items():
            print(
                f"{nuisance}/{name}: bayes={result['label_identifiability']['equal_prior_bayes_accuracy']:.4f} "
                f"core={result['core_coordinate_consistency']['minimum_best_robust_fixed_support_mean_f1']:.4f} "
                f"medoid={result['prototype_separability']['nearest_medoid_macro_accuracy']:.4f} "
                f"gate={'PASS' if result['gate_passed'] else 'FAIL'}",
                flush=True,
            )

    structural_pass = all(family["representations"]["structural_max_degree_root"]["gate_passed"] for family in families)
    oracle_pass = all(family["representations"]["oracle_root0"]["gate_passed"] for family in families)
    if structural_pass:
        label = "PASS_STRUCTURAL_ROOT_REPAIR"
        interpretation = "A graph-observable max-degree root repairs all registered conditions; rooted structural canonicalization can proceed without extra side information."
    elif oracle_pass:
        label = "PASS_ONLY_WITH_EXTERNAL_ANCHOR"
        interpretation = "A purely structural max-degree root is insufficient, but a stable externally supplied root repairs the registered conditions. Any next KSVD claim must explicitly include anchor availability as an assumption."
    else:
        label = "FAIL_SINGLE_ROOT_REPAIR"
        interpretation = "Neither a graph-observable max-degree root nor the generator-supplied root makes all noisy motif families identifiable with stable fixed core coordinates. Do not proceed to noisy KSVD with a fixed adjacency-mask atom interpretation."
    payload = {
        "protocol": "ksvd-g0b-root-repair-probe-v0-20260731",
        "config": {
            "flip_seed": args.flip_seed,
            "flip_samples_per_motif": args.flip_samples_per_motif,
            "ksvd_run": False,
        },
        "families": families,
        "decision": {
            "classification": label,
            "structural_root_all_conditions_passed": structural_pass,
            "oracle_root_all_conditions_passed": oracle_pass,
            "interpretation": interpretation,
        },
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report.write_text(render(payload), encoding="utf-8")
    print(f"decision={label}")
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

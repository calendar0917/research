"""Run the G0B-R representation gate before any noisy KSVD experiment."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from .from_scratch_representation_gate import (
    evaluate_representation_condition,
    make_add_one_noncore_variants,
    make_clean_variants,
    make_edge_flip_variants,
)


DEFAULT_JSON = Path("tracks/ksvd/results/from_scratch/g0b_representation_gate_20260731.json")
DEFAULT_REPORT = Path("tracks/ksvd/results/from_scratch/G0B_REPRESENTATION_GATE_20260731.md")


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# G0B-R Canonical Representation Gate 结果",
        "",
        "> 日期：2026-07-31  ",
        "> 本实验不运行 KSVD；它只检查加入 motif variation 后，canonical adjacency 是否仍允许定义稳定、可辨识的 motif atom。",
        "",
        "## 1. 总判定",
        "",
        f"**{payload['decision']['classification']}**",
        "",
        payload["decision"]["interpretation"],
        "",
        "## 2. 条件汇总",
        "",
        "| condition | variants | collision mass | Bayes accuracy | robust fixed-core F1 (worst motif) | medoid macro accuracy | gate |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["conditions"]:
        label = result["label_identifiability"]
        core = result["core_coordinate_consistency"]
        prototype = result["prototype_separability"]
        lines.append(
            f"| {result['condition']} | {result['variant_count']} | "
            f"{label['cross_family_collision_mass']:.4f} | "
            f"{label['equal_prior_bayes_accuracy']:.4f} | "
            f"{core['minimum_best_robust_fixed_support_mean_f1']:.4f} | "
            f"{prototype['nearest_medoid_macro_accuracy']:.4f} | "
            f"{'PASS' if result['gate_passed'] else 'FAIL'} |"
        )
    lines.extend([
        "",
        "## 3. Hidden core 在 canonical coordinates 中是否稳定",
        "",
        "| condition | motif | variants | role-identifiable rate | deterministic fixed F1 | optimistic fixed F1 | robust fixed F1 |",
        "|---|---|---:|---:|---:|---:|---:|",
    ])
    for result in payload["conditions"]:
        for family in result["core_coordinate_consistency"]["families"]:
            lines.append(
                f"| {result['condition']} | {family['motif_name']} | {family['variant_count']} | "
                f"{family['role_identifiable_variant_rate']:.4f} | "
                f"{family['best_deterministic_fixed_support_mean_f1']:.4f} | "
                f"{family['best_optimistic_fixed_support_mean_f1']:.4f} | "
                f"{family['best_robust_fixed_support_mean_f1']:.4f} |"
            )
    lines.extend([
        "",
        "其中：",
        "",
        "- `role-identifiable rate`：canonical graph 的所有最小排列是否都把 hidden core 映射到同一个 support；",
        "- `deterministic`：使用实现选出的第一个最小排列；",
        "- `optimistic`：对每个 graph automorphism 选择最有利的 hidden-core mapping；",
        "- `robust`：要求固定 support 对所有等价 mapping 都成立，是进入显式 motif atom 学习最严格、最可信的指标。",
        "",
        "## 4. Cross-family collisions",
        "",
    ])
    for result in payload["conditions"]:
        collisions = result["label_identifiability"]["collisions"]
        lines.append(
            f"### {result['condition']}: {len(collisions)} 个 collision vectors，"
            f"equal-prior mass={result['label_identifiability']['cross_family_collision_mass']:.4f}"
        )
        lines.append("")
        if not collisions:
            lines.append("无 cross-family collision。")
        else:
            for collision in collisions[:8]:
                lines.append(
                    f"- motifs={collision['motif_names']}，mass={collision['equal_prior_probability_mass']:.6f}，"
                    f"vector={collision['canonical_vector']}"
                )
            if len(collisions) > 8:
                lines.append(f"- 其余 {len(collisions) - 8} 个见 JSON。")
        lines.append("")
    lines.extend([
        "## 5. 科学解释",
        "",
        "G0 的 clean patches 只有四种 exact canonical vectors，因此 maximin initialization 可以直接枚举 vocabulary。G0B-R 检查的是：一旦同一 motif 允许结构变化，生成器定义的 hidden motif family 是否仍由无类型邻接图唯一决定，以及 hidden core 是否仍位于一致的线性坐标。",
        "",
        "如果 noisy condition 未通过 representation gate，后续直接运行 KSVD 将无法区分：",
        "",
        "1. KSVD 没有学到 motif；",
        "2. 相同观测图对应多个 hidden motif 解释；",
        "3. exact canonicalization 虽然对单图置换不变，但不同 variants 的 core 坐标不一致。",
        "",
        "此时正确动作不是增加 restart，而是重新定义表示或研究命题，例如使用 rooted/typed canonicalization、显式节点/边属性，或放弃“atom 是固定 adjacency edge mask”的强解释。",
        "",
    ])
    return "\n".join(lines)


def classify(results: list[dict[str, Any]]) -> dict[str, Any]:
    clean = next(result for result in results if result["condition"] == "R0_clean")
    noisy = [result for result in results if result["condition"] != "R0_clean"]
    if not clean["gate_passed"]:
        label = "FAIL_CLEAN_CONTROL"
        interpretation = "Even the clean representation control failed; the evaluator or canonical-role mapping is invalid."
    elif all(result["gate_passed"] for result in noisy):
        label = "PASS_ENTER_G0B_KSVD"
        interpretation = "All registered noisy representations retain identifiable labels, stable core coordinates, and separable prototypes; a noisy KSVD experiment is interpretable."
    else:
        label = "FAIL_NOISY_CANONICAL_REPRESENTATION"
        interpretation = "The clean control passes, but at least one noisy motif condition loses label identifiability or fixed core-coordinate semantics. Do not interpret a subsequent KSVD failure as an optimization failure."
    return {
        "classification": label,
        "clean_control_passed": bool(clean["gate_passed"]),
        "noisy_condition_pass_count": int(sum(result["gate_passed"] for result in noisy)),
        "noisy_condition_count": len(noisy),
        "interpretation": interpretation,
    }


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--flip-seed", type=int, default=20260731)
    parser.add_argument("--flip-probability", type=float, default=0.05)
    parser.add_argument("--flip-samples-per-motif", type=int, default=2000)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()

    conditions = [
        make_clean_variants(),
        make_add_one_noncore_variants(),
        make_edge_flip_variants(
            seed=args.flip_seed,
            flip_probability=args.flip_probability,
            samples_per_motif=args.flip_samples_per_motif,
        ),
    ]
    results = []
    for variants in conditions:
        result = evaluate_representation_condition(variants)
        results.append(result)
        print(
            f"{result['condition']}: variants={result['variant_count']} "
            f"collision={result['label_identifiability']['cross_family_collision_mass']:.4f} "
            f"bayes={result['label_identifiability']['equal_prior_bayes_accuracy']:.4f} "
            f"core={result['core_coordinate_consistency']['minimum_best_robust_fixed_support_mean_f1']:.4f} "
            f"medoid={result['prototype_separability']['nearest_medoid_macro_accuracy']:.4f} "
            f"gate={'PASS' if result['gate_passed'] else 'FAIL'}",
            flush=True,
        )
    decision = classify(results)
    payload = {
        "protocol": "ksvd-g0b-representation-gate-v0-20260731",
        "config": {
            "patch_size": 6,
            "vector_dimension": 15,
            "canonicalization": "exact_lexicographic_minimum_over_720_permutations",
            "motif_role_visible_to_representation": False,
            "flip_seed": args.flip_seed,
            "flip_probability": args.flip_probability,
            "flip_samples_per_motif": args.flip_samples_per_motif,
            "ksvd_run": False,
        },
        "conditions": results,
        "decision": decision,
    }
    args.json.parent.mkdir(parents=True, exist_ok=True)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    args.json.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    args.report.write_text(render_report(payload), encoding="utf-8")
    print(f"decision={decision['classification']}")
    print(f"wrote {args.json}")
    print(f"wrote {args.report}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

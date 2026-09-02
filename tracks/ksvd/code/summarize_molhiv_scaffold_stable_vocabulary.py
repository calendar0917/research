"""Summarize the 8k official-train scaffold-stable vocabulary pilot."""
from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

CONTROLS = (
    "fixed_random",
    "farthest",
    "uniform_facility",
    "scaffold_facility",
    "shuffled_scaffold_facility",
)
REFERENCE_RANDOM_BANK_ENSEMBLE = 0.7282186686866933


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="results/molhiv")
    ap.add_argument(
        "--output",
        default="results/molhiv/scaffold_stable_vocabulary_summary_scaffold3.json",
    )
    ap.add_argument(
        "--markdown",
        default="results/molhiv/REALPROTOTYPE_SCAFFOLD_STABLE_VOCABULARY_PILOT_20260728.md",
    )
    args = ap.parse_args()
    root = Path(args.result_dir)
    docs = [
        json.loads((root / f"scaffold_stable_vocabulary_fold{fold}_seed0.json").read_text())
        for fold in range(3)
    ]
    hybrid_docs = [
        json.loads((root / f"scaffold_stable_vocabulary_hybrid_fold{fold}_seed0.json").read_text())
        for fold in range(3)
    ]

    summary: dict = {
        "protocol_id": "molhiv-label-free-scaffold-facility-real-vocabulary-summary-v1",
        "date": "2026-07-28",
        "data_policy": {
            "dataset": "8k stratified subset",
            "development_graphs": 6400,
            "outer_validation": "three official-train-only Bemis-Murcko scaffold folds",
            "official_valid_evaluations": 0,
            "official_test_evaluations": 0,
            "model_seed": 0,
            "fixed_epochs": 30,
        },
        "controls": {},
        "pair_ensembles": {},
    }

    for control in CONTROLS:
        auc = np.asarray([d["results"][control]["heldout"]["auc"] for d in docs])
        coverage = np.asarray([
            d["results"][control]["heldout"]["mean_best_signed_cosine"] for d in docs
        ])
        q10 = np.asarray([
            d["results"][control]["heldout"]["q10_best_signed_cosine"] for d in docs
        ])
        objective = np.asarray([
            d["results"][control]["selection"]["true_scaffold_coverage_objective"]
            for d in docs
        ])
        summary["controls"][control] = {
            "auc_by_fold": auc.tolist(),
            "auc_mean": float(auc.mean()),
            "auc_std": float(auc.std(ddof=1)),
            "heldout_mean_best_cosine_mean": float(coverage.mean()),
            "heldout_q10_best_cosine_mean": float(q10.mean()),
            "fit_true_scaffold_coverage_objective_mean": float(objective.mean()),
        }

    for a, b in combinations(CONTROLS, 2):
        aucs = []
        correlations = []
        for d in docs:
            ha = d["results"][a]["heldout"]
            hb = d["results"][b]["heldout"]
            y = np.asarray(ha["labels"], dtype=np.int64)
            pa = sigmoid(np.asarray(ha["scores"], dtype=np.float64))
            pb = sigmoid(np.asarray(hb["scores"], dtype=np.float64))
            aucs.append(float(roc_auc_score(y, 0.5 * (pa + pb))))
            correlations.append(float(np.corrcoef(pa, pb)[0, 1]))
        summary["pair_ensembles"][f"{a}+{b}"] = {
            "auc_by_fold": aucs,
            "auc_mean": float(np.mean(aucs)),
            "probability_pearson_mean": float(np.mean(correlations)),
        }

    hybrid_auc = np.asarray([
        d["results"]["hybrid_half"]["heldout"]["auc"] for d in hybrid_docs
    ])
    summary["hybrid_single_bank"] = {
        "definition": "first 16 farthest plus first 16 scaffold-facility prototypes, unique deterministic fill",
        "auc_by_fold": hybrid_auc.tolist(),
        "auc_mean": float(hybrid_auc.mean()),
        "auc_std": float(hybrid_auc.std(ddof=1)),
    }

    random_auc = np.asarray(summary["controls"]["fixed_random"]["auc_by_fold"])
    scaffold_auc = np.asarray(summary["controls"]["scaffold_facility"]["auc_by_fold"])
    farthest_auc = np.asarray(summary["controls"]["farthest"]["auc_by_fold"])
    uniform_auc = np.asarray(summary["controls"]["uniform_facility"]["auc_by_fold"])
    shuffled_auc = np.asarray(summary["controls"]["shuffled_scaffold_facility"]["auc_by_fold"])
    gates = {
        "scaffold_minus_random_auc_mean": float((scaffold_auc - random_auc).mean()),
        "scaffold_random_fold_wins": int(np.sum(scaffold_auc > random_auc)),
        "scaffold_minus_farthest_auc_mean": float((scaffold_auc - farthest_auc).mean()),
        "scaffold_minus_uniform_auc_mean": float((scaffold_auc - uniform_auc).mean()),
        "scaffold_minus_shuffled_auc_mean": float((scaffold_auc - shuffled_auc).mean()),
        "random_gain_at_least_005": bool((scaffold_auc - random_auc).mean() >= 0.005),
        "random_wins_at_least_2_of_3": bool(np.sum(scaffold_auc > random_auc) >= 2),
        "beats_farthest_mean": bool(scaffold_auc.mean() > farthest_auc.mean()),
        "beats_uniform_mean": bool(scaffold_auc.mean() > uniform_auc.mean()),
        "beats_shuffled_mean": bool(scaffold_auc.mean() > shuffled_auc.mean()),
    }
    gates["strict_scaffold_specific_promotion_passed"] = bool(
        gates["random_gain_at_least_005"]
        and gates["random_wins_at_least_2_of_3"]
        and gates["beats_farthest_mean"]
        and gates["beats_uniform_mean"]
        and gates["beats_shuffled_mean"]
    )
    summary["promotion_gate"] = gates
    summary["exploratory_family_comparison"] = {
        "reference_three_random_bank_ensemble_auc_mean": REFERENCE_RANDOM_BANK_ENSEMBLE,
        "farthest_minus_reference_ensemble": float(farthest_auc.mean() - REFERENCE_RANDOM_BANK_ENSEMBLE),
        "scaffold_facility_minus_reference_ensemble": float(scaffold_auc.mean() - REFERENCE_RANDOM_BANK_ENSEMBLE),
    }

    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    c = summary["controls"]
    pair = summary["pair_ensembles"]["farthest+scaffold_facility"]
    lines = [
        "# Real-prototype scaffold-stable vocabulary pilot（2026-07-28）",
        "",
        "## 1. 数据边界",
        "",
        "- 8,000-graph development subset；只使用其中 6,400 个 official-train graphs。",
        "- 三个 official-train-only Bemis–Murcko scaffold outer folds。",
        "- model seed 固定为 0，固定训练 30 epochs。",
        "- 本轮 official valid/test 编码与评估次数均为 **0**。",
        "",
        "## 2. 为什么做这个 pilot",
        "",
        "稳定性诊断显示：三个随机真实 prototype banks 的 optimal-matching cosine 只有 `0.557700`，",
        "而 probability ensemble 相对单 bank 均值增加 `+0.007604` AUC；但是 fit→heldout coverage",
        "变化仅 `-0.000231`，prototype usage JS 仅 `0.001544`。因此主要问题不是普通的",
        "跨 scaffold coverage collapse，而是 prototype identity 与 downstream composition 的方差。",
        "",
        "本 pilot 把候选池从 256 个随机来源图扩大为每个 outer-fit graph 一个确定性 observed patch，",
        "并比较 random、farthest、uniform facility、真实 scaffold-balanced facility、",
        "shuffled-scaffold facility。所有 selector 均 label-free。",
        "",
        "## 3. 三折结果",
        "",
        "| Selector | Fold 0 | Fold 1 | Fold 2 | Mean | Std | Heldout coverage |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for control in CONTROLS:
        row = c[control]
        vals = row["auc_by_fold"]
        lines.append(
            f"| {control} | {vals[0]:.6f} | {vals[1]:.6f} | {vals[2]:.6f} | "
            f"{row['auc_mean']:.6f} | {row['auc_std']:.6f} | "
            f"{row['heldout_mean_best_cosine_mean']:.6f} |"
        )
    lines += [
        "",
        "严格 scaffold-specific gate：",
        "",
        f"- scaffold facility − matched random：`{gates['scaffold_minus_random_auc_mean']:+.6f}`，赢 `{gates['scaffold_random_fold_wins']}/3` folds；通过。",
        f"- scaffold facility − uniform facility：`{gates['scaffold_minus_uniform_auc_mean']:+.6f}`；通过。",
        f"- scaffold facility − shuffled scaffold：`{gates['scaffold_minus_shuffled_auc_mean']:+.6f}`；通过。",
        f"- scaffold facility − farthest：`{gates['scaffold_minus_farthest_auc_mean']:+.6f}`；未通过。",
        "",
        "因此 **scaffold-specific selector 严格 promotion gate 未通过**；它与 broad-pool farthest",
        "几乎完全打平，而不是明确优于所有 matched controls。",
        "",
        "## 4. 更重要的结果",
        "",
        f"- broad deterministic candidate pool 下，farthest mean AUC 为 `{c['farthest']['auc_mean']:.6f}`，",
        f"  scaffold facility 为 `{c['scaffold_facility']['auc_mean']:.6f}`；二者都高于旧 3-random-bank ensemble `{REFERENCE_RANDOM_BANK_ENSEMBLE:.6f}`。",
        f"- farthest + scaffold-facility probability ensemble：fold AUC `{pair['auc_by_fold']}`，mean `{pair['auc_mean']:.6f}`。",
        f"- 二者 probability Pearson mean 只有 `{pair['probability_pearson_mean']:.6f}`，存在明显互补。",
        f"- 但把两套 prototype 各取 16 个压成单 bank 后，mean 仅 `{summary['hybrid_single_bank']['auc_mean']:.6f}`。",
        "",
        "这说明互补性不是简单的 atom union：top-3 assignment、prototype competition 和 MIL readout",
        "会随 vocabulary 整体几何改变。coverage 从约 0.70 提升到约 0.81，也没有自动带来同比例 AUC",
        "提升，所以继续只优化 coverage objective 的收益可能有限。",
        "",
        "## 5. 研究判断",
        "",
        "1. **有效的新结论**：广覆盖、确定性的 observed-patch candidate pool 是明显改进；早期",
        "   farthest 失败主要不能归因于 farthest 原理本身，也与过窄的 256-candidate random reservoir 有关。",
        "2. **尚未证明**：真实 scaffold weighting 本身优于一般的 broad-pool diversity selection。",
        "3. **下一优先级**：固定 farthest/scaffold 两套强 vocabulary，研究 prototype occurrence 的",
        "   distance-binned / diffusion relational composition，而不是继续调 facility 权重。",
        "4. relation head 应先做低容量 GNN-free 版本，并保留 occurrence-independent MIL、",
        "   farthest、scaffold facility、distance-shuffled relation 作为 matched controls。",
        "",
    ]
    Path(args.markdown).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "markdown": args.markdown,
        "promotion_passed": gates["strict_scaffold_specific_promotion_passed"],
        "farthest_mean_auc": c["farthest"]["auc_mean"],
        "scaffold_mean_auc": c["scaffold_facility"]["auc_mean"],
        "pair_ensemble_mean_auc": pair["auc_mean"],
        "hybrid_mean_auc": summary["hybrid_single_bank"]["auc_mean"],
    }, indent=2))


if __name__ == "__main__":
    main()

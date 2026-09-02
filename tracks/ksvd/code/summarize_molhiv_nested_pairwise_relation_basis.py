"""Summarize nested pairwise-relation basis experiments and projection robustness."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

PROJECTION_SEEDS = (20260728, 1729, 2718)
MODES = ("real", "assignment_shuffled")
RANDOM_KEY = "random_semantic_rank"


def sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def fmt(values: list[float] | np.ndarray) -> str:
    return " / ".join(f"{float(value):+.6f}" for value in values)


def sigmoid(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(np.asarray(scores, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-scores))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs=9)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-report", required=True)
    args = ap.parse_args()

    docs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    by_seed_fold: dict[tuple[int, int], dict[str, Any]] = {}
    for doc in docs:
        if doc.get("protocol_id") != "molhiv_nested_lowrank_pairwise_relation_basis_v1":
            raise ValueError("unexpected protocol")
        policy = doc.get("selection_policy", {})
        if int(policy.get("official_valid_evaluations", -1)) != 0:
            raise ValueError("official valid was evaluated")
        if int(policy.get("official_test_evaluations", -1)) != 0:
            raise ValueError("official test was evaluated")
        seed = int(doc.get("config", {}).get("projection_seed", -1))
        fold = int(doc.get("fold", -1))
        key = (seed, fold)
        if key in by_seed_fold:
            raise ValueError(f"duplicate result {key}")
        by_seed_fold[key] = doc
    expected = {(seed, fold) for seed in PROJECTION_SEEDS for fold in range(3)}
    if set(by_seed_fold) != expected:
        raise ValueError(f"expected seed/fold grid {sorted(expected)}, got {sorted(by_seed_fold)}")

    # Check that changing projection seed changed only the matched random rank-8 coordinates.
    invariant_keys = (
        "task__real",
        "task__assignment_shuffled",
        "shuffled_label__real",
        "shuffled_label__assignment_shuffled",
        "random__real",
        "random__assignment_shuffled",
        "full_offdiag_pair__real",
        "full_offdiag_pair__assignment_shuffled",
        "semantic_pair__real",
        "semantic_pair__assignment_shuffled",
    )
    for fold in range(3):
        reference = by_seed_fold[(PROJECTION_SEEDS[0], fold)]
        for seed in PROJECTION_SEEDS[1:]:
            candidate = by_seed_fold[(seed, fold)]
            if candidate["fit_indices_sha256"] != reference["fit_indices_sha256"]:
                raise ValueError("fit split changed across projection seeds")
            if candidate["heldout_indices_sha256"] != reference["heldout_indices_sha256"]:
                raise ValueError("heldout split changed across projection seeds")
            if candidate["pair_real_sha256"] != reference["pair_real_sha256"]:
                raise ValueError("real pair tensor changed across projection seeds")
            if candidate["pair_assignment_shuffled_sha256"] != reference["pair_assignment_shuffled_sha256"]:
                raise ValueError("shuffled pair tensor changed across projection seeds")
            for result_key in invariant_keys:
                a = reference["results"][result_key]["heldout"]
                b = candidate["results"][result_key]["heldout"]
                if a["score_sha256"] != b["score_sha256"]:
                    raise ValueError(f"invariant result changed across projection seeds: {result_key}")

    canonical_docs = [by_seed_fold[(PROJECTION_SEEDS[0], fold)] for fold in range(3)]
    heldout_indices = [
        np.asarray(doc["results"]["task__real"]["heldout"]["graph_indices"], dtype=np.int64)
        for doc in canonical_docs
    ]
    combined_indices = np.concatenate(heldout_indices)
    if len(np.unique(combined_indices)) != len(combined_indices):
        raise ValueError("outer heldout folds overlap")

    base_fold = np.asarray(
        [doc["base_compact"]["heldout_auc"] for doc in canonical_docs], dtype=np.float64
    )
    invariant_summary: dict[str, Any] = {}
    for key in invariant_keys:
        values = np.asarray(
            [doc["results"][key]["heldout"]["auc"] for doc in canonical_docs],
            dtype=np.float64,
        )
        gains = values - base_fold
        invariant_summary[key] = {
            "fold_auc": values.tolist(),
            "mean_auc": float(values.mean()),
            "fold_gain_over_base": gains.tolist(),
            "mean_gain_over_base": float(gains.mean()),
            "base_fold_wins": int(np.sum(gains > 0.0)),
        }

    projection_summary: dict[str, Any] = {}
    all_real_gains: list[float] = []
    all_real_minus_shuffled: list[float] = []
    for seed in PROJECTION_SEEDS:
        real = np.asarray(
            [
                by_seed_fold[(seed, fold)]["results"][f"{RANDOM_KEY}__real"]["heldout"]["auc"]
                for fold in range(3)
            ],
            dtype=np.float64,
        )
        shuffled = np.asarray(
            [
                by_seed_fold[(seed, fold)]["results"][
                    f"{RANDOM_KEY}__assignment_shuffled"
                ]["heldout"]["auc"]
                for fold in range(3)
            ],
            dtype=np.float64,
        )
        gains = real - base_fold
        matched = real - shuffled
        all_real_gains.extend(gains.tolist())
        all_real_minus_shuffled.extend(matched.tolist())
        projection_summary[str(seed)] = {
            "real_fold_auc": real.tolist(),
            "real_mean_auc": float(real.mean()),
            "real_fold_gain_over_base": gains.tolist(),
            "real_mean_gain_over_base": float(gains.mean()),
            "real_base_fold_wins": int(np.sum(gains > 0.0)),
            "assignment_shuffled_fold_auc": shuffled.tolist(),
            "assignment_shuffled_mean_auc": float(shuffled.mean()),
            "real_minus_assignment_shuffled_fold": matched.tolist(),
            "real_minus_assignment_shuffled_mean": float(matched.mean()),
            "real_assignment_shuffled_fold_wins": int(np.sum(matched > 0.0)),
        }

    ensemble_fold: dict[str, list[float]] = {mode: [] for mode in MODES}
    ensemble_logit_fold: dict[str, list[float]] = {mode: [] for mode in MODES}
    residual_correlations: dict[str, list[list[float]]] = {mode: [] for mode in MODES}
    for fold in range(3):
        for mode in MODES:
            scores_by_seed: list[np.ndarray] = []
            residuals_by_seed: list[np.ndarray] = []
            reference_indices = None
            reference_labels = None
            for seed in PROJECTION_SEEDS:
                heldout = by_seed_fold[(seed, fold)]["results"][f"{RANDOM_KEY}__{mode}"][
                    "heldout"
                ]
                indices = np.asarray(heldout["graph_indices"], dtype=np.int64)
                labels = np.asarray(heldout["labels"], dtype=np.float64)
                scores = np.asarray(heldout["scores"], dtype=np.float64)
                residuals = np.asarray(heldout["relation_residuals"], dtype=np.float64)
                if reference_indices is None:
                    reference_indices = indices
                    reference_labels = labels
                elif not np.array_equal(indices, reference_indices) or not np.array_equal(
                    labels, reference_labels
                ):
                    raise ValueError("prediction alignment changed across projection seeds")
                scores_by_seed.append(scores)
                residuals_by_seed.append(residuals)
            probability_average = np.mean(
                np.stack([sigmoid(scores) for scores in scores_by_seed], axis=0), axis=0
            )
            logit_average = np.mean(np.stack(scores_by_seed, axis=0), axis=0)
            ensemble_fold[mode].append(
                float(roc_auc_score(reference_labels, probability_average))
            )
            ensemble_logit_fold[mode].append(float(roc_auc_score(reference_labels, logit_average)))
            corr = np.corrcoef(np.stack(residuals_by_seed, axis=0))
            residual_correlations[mode].append(corr.tolist())

    ensemble_real = np.asarray(ensemble_fold["real"], dtype=np.float64)
    ensemble_shuffled = np.asarray(ensemble_fold["assignment_shuffled"], dtype=np.float64)
    ensemble_gain = ensemble_real - base_fold
    ensemble_matched = ensemble_real - ensemble_shuffled
    ensemble_summary = {
        "method": "fixed equal average of probabilities from projection seeds 20260728, 1729, and 2718",
        "real_fold_auc": ensemble_real.tolist(),
        "real_mean_auc": float(ensemble_real.mean()),
        "real_fold_gain_over_base": ensemble_gain.tolist(),
        "real_mean_gain_over_base": float(ensemble_gain.mean()),
        "real_base_fold_wins": int(np.sum(ensemble_gain > 0.0)),
        "assignment_shuffled_fold_auc": ensemble_shuffled.tolist(),
        "assignment_shuffled_mean_auc": float(ensemble_shuffled.mean()),
        "real_minus_assignment_shuffled_fold": ensemble_matched.tolist(),
        "real_minus_assignment_shuffled_mean": float(ensemble_matched.mean()),
        "real_assignment_shuffled_fold_wins": int(np.sum(ensemble_matched > 0.0)),
        "logit_average_sensitivity": {
            mode: {
                "fold_auc": ensemble_logit_fold[mode],
                "mean_auc": float(np.mean(ensemble_logit_fold[mode])),
            }
            for mode in MODES
        },
        "residual_correlation_by_fold": residual_correlations,
    }

    thresholds = {
        "average_projection_seed_mean_gain_at_least": 0.002,
        "every_projection_seed_mean_gain_positive": True,
        "every_projection_seed_real_mean_beats_assignment_shuffled_mean": True,
        "minimum_positive_fold_gains_out_of_9": 7,
        "minimum_real_beats_assignment_shuffled_out_of_9": 7,
    }
    observed = {
        "average_projection_seed_mean_gain": float(np.mean(all_real_gains)),
        "projection_seeds_with_positive_mean_gain": int(
            sum(projection_summary[str(seed)]["real_mean_gain_over_base"] > 0 for seed in PROJECTION_SEEDS)
        ),
        "projection_seeds_real_mean_beats_assignment_shuffled": int(
            sum(
                projection_summary[str(seed)]["real_minus_assignment_shuffled_mean"] > 0
                for seed in PROJECTION_SEEDS
            )
        ),
        "positive_fold_gains_out_of_9": int(np.sum(np.asarray(all_real_gains) > 0.0)),
        "real_beats_assignment_shuffled_out_of_9": int(
            np.sum(np.asarray(all_real_minus_shuffled) > 0.0)
        ),
    }
    conditions = {
        "average_gain": observed["average_projection_seed_mean_gain"] >= 0.002,
        "every_seed_positive": observed["projection_seeds_with_positive_mean_gain"] == 3,
        "every_seed_matched_control": observed[
            "projection_seeds_real_mean_beats_assignment_shuffled"
        ]
        == 3,
        "fold_gain_wins": observed["positive_fold_gains_out_of_9"] >= 7,
        "matched_control_wins": observed["real_beats_assignment_shuffled_out_of_9"] >= 7,
    }
    robustness_gate = {
        "thresholds": thresholds,
        "observed": observed,
        "conditions": conditions,
        "passed": bool(all(conditions.values())),
    }

    summary = {
        "protocol_id": "molhiv_nested_pairwise_relation_basis_projection_robustness_summary_v1",
        "date": "2026-07-28",
        "inputs": args.inputs,
        "projection_seeds": list(PROJECTION_SEEDS),
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "base_compact": {
            "fold_auc": base_fold.tolist(),
            "mean_auc": float(base_fold.mean()),
        },
        "invariant_results": invariant_summary,
        "projection_seed_results": projection_summary,
        "three_projection_probability_ensemble": ensemble_summary,
        "robustness_gate": robustness_gate,
        "decision": (
            "STOP_RANDOM_PAIR_SKETCH_RETAIN_COMPACT_RELATION"
            if not robustness_gate["passed"]
            else "PROMOTE_TO_ADDITIONAL_DEVELOPMENT_ROBUSTNESS"
        ),
        "interpretation": (
            "Specific off-diagonal pair identity has a weak high-capacity upper-bound signal, but "
            "supervised pair directions do not transfer across scaffolds, semantic top directions fail, "
            "and random rank-8 pair sketches are projection-seed dependent. Their fixed prediction "
            "average leaves almost no advantage over assignment shuffling."
        ),
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    task = invariant_summary["task__real"]
    shuffled_label = invariant_summary["shuffled_label__real"]
    random_rank4 = invariant_summary["random__real"]
    full_pair = invariant_summary["full_offdiag_pair__real"]
    full_pair_shuffled = invariant_summary["full_offdiag_pair__assignment_shuffled"]
    semantic = invariant_summary["semantic_pair__real"]
    semantic_shuffled = invariant_summary["semantic_pair__assignment_shuffled"]

    seed_rows = "\n".join(
        "| `{seed}` | {auc:.6f} | {gain:+.6f} | {wins}/3 | {matched:+.6f} | {mwins}/3 |".format(
            seed=seed,
            auc=projection_summary[str(seed)]["real_mean_auc"],
            gain=projection_summary[str(seed)]["real_mean_gain_over_base"],
            wins=projection_summary[str(seed)]["real_base_fold_wins"],
            matched=projection_summary[str(seed)]["real_minus_assignment_shuffled_mean"],
            mwins=projection_summary[str(seed)]["real_assignment_shuffled_fold_wins"],
        )
        for seed in PROJECTION_SEEDS
    )

    report = f"""# 真实原型 pair 关系路线：完整诊断

> 日期：2026-07-28  
> 数据边界：8k 子集中的 6,400 个 official-train 分子，三折 scaffold 外层划分。  
> official-valid 编码/评估次数：**0**。  
> official-test 编码/评估次数：**0**。

## 1. 这一步在问什么

前一阶段已经证明：固定一组覆盖面较广的真实局部结构，再统计它们在距离 1、2 上如何共同出现，能够稳定优于只统计“每种结构出现多少次”。

这里进一步追问：

> 是否需要保留“具体是哪两种局部结构发生了组合”，而不仅是较粗的汇总统计？

每个结构词汇有 32 个原型。完整记录原型对会产生 1,984 个额外数值，容易过拟合，因此我们比较了多种压缩办法。

## 2. 主要结果

基础方案三折 AUC 为 `{fmt(base_fold)}`，平均 **{base_fold.mean():.6f}**。

| 方法 | 平均 AUC | 相对基础方案 | 解释 |
|---|---:|---:|---|
| 标签学习的 rank-4 pair 方向 | {task['mean_auc']:.6f} | {task['mean_gain_over_base']:+.6f} | 内层能学到，但不能跨 scaffold 迁移 |
| 打乱标签后学习的 rank-4 方向 | {shuffled_label['mean_auc']:.6f} | {shuffled_label['mean_gain_over_base']:+.6f} | 与真实标签方向几乎相同 |
| 同维随机 rank-4 方向 | {random_rank4['mean_auc']:.6f} | {random_rank4['mean_gain_over_base']:+.6f} | 反而略高于真实标签方向 |
| 完整具体 pair 身份（1,984 维） | {full_pair['mean_auc']:.6f} | **{full_pair['mean_gain_over_base']:+.6f}** | 仅作为容量上界，训练拟合很高 |
| 原型语义主方向（144 维） | {semantic['mean_auc']:.6f} | {semantic['mean_gain_over_base']:+.6f} | 高频/主变化方向不是预测方向 |

完整 pair 的真实关系平均比打乱关系高 **{full_pair['mean_auc'] - full_pair_shuffled['mean_auc']:+.6f}**，且 2/3 folds 胜出。这是一个很弱但存在的容量上界：具体 pair 身份可能有少量额外信息。不过，该模型维数高、fit AUC 约 0.98，不能直接晋级。

语义主方向的真实关系平均比其打乱对照高 **{semantic['mean_auc'] - semantic_shuffled['mean_auc']:+.6f}**，但低于同维随机方向，因此不能支持“原型最主要的语义变化就是任务所需变化”。

## 3. 随机 rank-8 压缩的稳健性

这个方法先把 32 个原型投影到 8 个随机坐标，再完整保留 8×8 的组合信息，总维数为 144。改变 projection seed 只会改变这 8 个坐标，不会改变数据划分、原型、节点分配、基础预测器或训练参数。

| Projection seed | 真实关系平均 AUC | 相对基础方案 | 胜基础 folds | 真实 − 打乱关系 | 胜打乱 folds |
|---:|---:|---:|---:|---:|---:|
{seed_rows}

三次投影合计：

- 三个 seed 的平均增益：**{observed['average_projection_seed_mean_gain']:+.6f}**；
- 正增益的 projection seeds：{observed['projection_seeds_with_positive_mean_gain']}/3；
- 真实关系平均胜过打乱关系的 projection seeds：{observed['projection_seeds_real_mean_beats_assignment_shuffled']}/3；
- 胜基础方案的 folds：{observed['positive_fold_gains_out_of_9']}/9；
- 真实关系胜打乱关系的 folds：{observed['real_beats_assignment_shuffled_out_of_9']}/9。

预先固定的稳健性门槛结论：**{'通过' if robustness_gate['passed'] else '不通过'}**。

## 4. 三个投影做固定平均

为了判断“单次投影噪声很大，但共同部分可能稳定”，我们将三个 projection seed 的预测概率按固定的 1/3、1/3、1/3 平均，没有根据 heldout 结果调权重。

- 三折 AUC：`{fmt(ensemble_real)}`；
- 平均 AUC：**{ensemble_real.mean():.6f}**；
- 相对基础方案：**{ensemble_gain.mean():+.6f}**，2/3 folds 胜出；
- 相对打乱关系的固定平均：**{ensemble_matched.mean():+.6f}**，2/3 folds 胜出。

平均确实减少了随机波动，但真实关系相对打乱关系只剩 **{ensemble_matched.mean():+.6f}**。这个差值太小，无法说明提升来自正确的局部结构组合，而不是高维 sidecar 的普通扰动或训练噪声。

## 5. 清晰结论

1. **监督学习具体 pair 方向失败。** 真实标签能提高内层分数，但学到的方向不能跨 scaffold 迁移。
2. **完整 pair 身份存在弱容量上界。** 它说明粗汇总可能遗漏了一点信息，但高维结果过拟合，不能作为最终方法。
3. **原型语义主方向失败。** 最常见、方差最大的结构变化不是最有预测价值的变化。
4. **随机低维 pair sketch 不稳定。** 单次结果依赖投影 seed；固定平均后，真实关系相对打乱关系的优势几乎为零。
5. **停止随机投影分支，不运行 full graph、official-valid 或 official-test。** 当前应保留稳定的 compact exact-distance relation 方案。

## 6. 下一步应该改变什么

下一步仍可研究“具体结构组合”，但不能再靠：

- 增加随机投影 seed；
- 调 rank、selector C 或 residual cap；
- 用标签直接挑 pair 方向。

更合理的一次性诊断是：只利用 outer-fit 分子中**实际 pair 统计如何共同变化**，学习一个确定性的无标签压缩；同时用相同压缩处理打乱关系。它与当前失败方法的区别是：不看标签、不依赖随机坐标，也不是根据原型向量本身的主方向，而是根据真实分子中的组合数据建立坐标。
"""
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(report, encoding="utf-8")
    print(
        json.dumps(
            {
                "output_json": str(output_json),
                "output_report": str(output_report),
                "robustness_gate": robustness_gate,
                "ensemble": ensemble_summary,
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()

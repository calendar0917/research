"""Summarize deterministic label-free pair-PCA development and full-train robustness runs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

METHODS = ("covariance_pca", "correlation_pca")
MODES = ("real", "assignment_shuffled")


def sigmoid(scores: np.ndarray) -> np.ndarray:
    scores = np.clip(np.asarray(scores, dtype=np.float64), -60.0, 60.0)
    return 1.0 / (1.0 + np.exp(-scores))


def fmt(values: list[float] | np.ndarray, signed: bool = False) -> str:
    spec = "+.6f" if signed else ".6f"
    return " / ".join(format(float(v), spec) for v in values)


def load_grid(paths: list[str], expected_seeds: tuple[int, ...], name: str) -> dict[tuple[int, int], dict[str, Any]]:
    grid: dict[tuple[int, int], dict[str, Any]] = {}
    for path in paths:
        doc = json.loads(Path(path).read_text(encoding="utf-8"))
        if doc.get("protocol_id") != "molhiv_label_free_pair_pca_v1":
            raise ValueError(f"unexpected protocol in {path}")
        policy = doc.get("selection_policy", {})
        if int(policy.get("official_valid_evaluations", -1)) != 0:
            raise ValueError(f"official valid was evaluated in {path}")
        if int(policy.get("official_test_evaluations", -1)) != 0:
            raise ValueError(f"official test was evaluated in {path}")
        seed = int(doc.get("config", {}).get("seed", -1))
        fold = int(doc.get("fold", -1))
        key = (seed, fold)
        if key in grid:
            raise ValueError(f"duplicate {name} seed/fold {key}")
        grid[key] = doc
    expected = {(seed, fold) for seed in expected_seeds for fold in range(3)}
    if set(grid) != expected:
        raise ValueError(f"{name}: expected {sorted(expected)}, got {sorted(grid)}")
    return grid


def audit_seed_invariants(grid: dict[tuple[int, int], dict[str, Any]], seeds: tuple[int, ...]) -> None:
    for fold in range(3):
        ref = grid[(seeds[0], fold)]
        for seed in seeds[1:]:
            doc = grid[(seed, fold)]
            for key in ("fit_indices_sha256", "heldout_indices_sha256"):
                if doc.get(key) != ref.get(key):
                    raise ValueError(f"{key} changed across model seeds on fold {fold}")
            if doc["base_compact"]["heldout_score_sha256"] != ref["base_compact"]["heldout_score_sha256"]:
                raise ValueError(f"frozen compact base changed across model seeds on fold {fold}")
            for family in ("farthest", "scaffold_facility"):
                if doc["prototype_audit"][family]["prototype_sha256"] != ref["prototype_audit"][family]["prototype_sha256"]:
                    raise ValueError(f"fixed vocabulary changed across model seeds: fold={fold}, family={family}")
            for basis in METHODS:
                if int(doc["basis_audit"][basis]["rank"]) != int(ref["basis_audit"][basis]["rank"]):
                    raise ValueError(f"{basis} rank changed across model seeds on fold {fold}")


def summarize_seed(grid: dict[tuple[int, int], dict[str, Any]], seed: int) -> dict[str, Any]:
    docs = [grid[(seed, fold)] for fold in range(3)]
    base = np.asarray([d["base_compact"]["heldout_auc"] for d in docs], dtype=np.float64)
    result: dict[str, Any] = {
        "base_compact_fold_auc": base.tolist(),
        "base_compact_mean_auc": float(base.mean()),
        "methods": {},
    }
    for method in METHODS:
        real = np.asarray([d["results"][f"{method}__real"]["heldout"]["auc"] for d in docs])
        shuffled = np.asarray([
            d["results"][f"{method}__assignment_shuffled"]["heldout"]["auc"] for d in docs
        ])
        gain = real - base
        matched = real - shuffled
        result["methods"][method] = {
            "real_fold_auc": real.tolist(),
            "real_mean_auc": float(real.mean()),
            "real_fold_gain_over_base": gain.tolist(),
            "real_mean_gain_over_base": float(gain.mean()),
            "real_base_fold_wins": int(np.sum(gain > 0)),
            "assignment_shuffled_fold_auc": shuffled.tolist(),
            "assignment_shuffled_mean_auc": float(shuffled.mean()),
            "real_minus_assignment_shuffled_fold": matched.tolist(),
            "real_minus_assignment_shuffled_mean": float(matched.mean()),
            "real_assignment_shuffled_fold_wins": int(np.sum(matched > 0)),
        }
    return result


def probability_ensemble(grid: dict[tuple[int, int], dict[str, Any]], seeds: tuple[int, ...]) -> dict[str, Any]:
    base_fold: list[float] = []
    output: dict[str, Any] = {method: {} for method in METHODS}
    for fold in range(3):
        base_fold.append(float(grid[(seeds[0], fold)]["base_compact"]["heldout_auc"]))
    base = np.asarray(base_fold)
    for method in METHODS:
        fold_auc: dict[str, list[float]] = {mode: [] for mode in MODES}
        for fold in range(3):
            for mode in MODES:
                records = [grid[(seed, fold)]["results"][f"{method}__{mode}"]["heldout"] for seed in seeds]
                indices = [np.asarray(r["graph_indices"], dtype=np.int64) for r in records]
                labels = [np.asarray(r["labels"], dtype=np.float64) for r in records]
                if any(not np.array_equal(indices[0], x) for x in indices[1:]) or any(
                    not np.array_equal(labels[0], x) for x in labels[1:]
                ):
                    raise ValueError("prediction alignment changed across model seeds")
                probs = np.mean(np.stack([sigmoid(np.asarray(r["scores"])) for r in records]), axis=0)
                fold_auc[mode].append(float(roc_auc_score(labels[0], probs)))
        real = np.asarray(fold_auc["real"])
        shuffled = np.asarray(fold_auc["assignment_shuffled"])
        output[method] = {
            "method": f"fixed equal probability average over model seeds {list(seeds)}",
            "real_fold_auc": real.tolist(),
            "real_mean_auc": float(real.mean()),
            "real_fold_gain_over_base": (real - base).tolist(),
            "real_mean_gain_over_base": float((real - base).mean()),
            "real_base_fold_wins": int(np.sum(real > base)),
            "assignment_shuffled_fold_auc": shuffled.tolist(),
            "assignment_shuffled_mean_auc": float(shuffled.mean()),
            "real_minus_assignment_shuffled_fold": (real - shuffled).tolist(),
            "real_minus_assignment_shuffled_mean": float((real - shuffled).mean()),
            "real_assignment_shuffled_fold_wins": int(np.sum(real > shuffled)),
        }
    return output


def gate(method: dict[str, Any]) -> dict[str, Any]:
    conditions = {
        "mean_gain_at_least_0.002": method["real_mean_gain_over_base"] >= 0.002,
        "at_least_2_of_3_base_wins": method["real_base_fold_wins"] >= 2,
        "real_mean_beats_assignment_shuffled": method["real_minus_assignment_shuffled_mean"] > 0,
        "at_least_2_of_3_matched_control_wins": method["real_assignment_shuffled_fold_wins"] >= 2,
    }
    return {
        "thresholds": {
            "mean_real_gain_over_frozen_compact_base_at_least": 0.002,
            "minimum_real_base_fold_wins": 2,
            "real_mean_must_beat_assignment_shuffled_mean": True,
            "minimum_real_assignment_shuffled_fold_wins": 2,
        },
        "conditions": conditions,
        "passed": bool(all(conditions.values())),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--development-inputs", nargs=9, required=True)
    ap.add_argument("--full-inputs", nargs=3)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-report", required=True)
    args = ap.parse_args()

    dev_seeds = (0, 1, 2)
    dev = load_grid(args.development_inputs, dev_seeds, "8k development")
    audit_seed_invariants(dev, dev_seeds)
    dev_by_seed = {str(seed): summarize_seed(dev, seed) for seed in dev_seeds}
    dev_ensemble = probability_ensemble(dev, dev_seeds)

    full_summary = None
    full_gate = None
    if args.full_inputs:
        full = load_grid(args.full_inputs, (0,), "full official-train development")
        full_summary = summarize_seed(full, 0)
        full_gate = gate(full_summary["methods"]["covariance_pca"])

    summary = {
        "protocol_id": "molhiv_label_free_pair_pca_robustness_summary_v1",
        "date": "2026-07-28",
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "development_8k": {
            "inputs": args.development_inputs,
            "model_seeds": list(dev_seeds),
            "by_seed": dev_by_seed,
            "probability_ensemble": dev_ensemble,
        },
        "full_official_train_three_fold_development": None if full_summary is None else {
            "inputs": args.full_inputs,
            "seed_0": full_summary,
            "covariance_pca_promotion_gate": full_gate,
        },
        "decision": (
            "awaiting_full_official_train_three_fold_development"
            if full_gate is None
            else "freeze_before_one_shot_official_valid" if full_gate["passed"]
            else "stop_pair_pca_and_keep_compact_exact_distance_relation"
        ),
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Real-prototype 无标签 pair-PCA 路线总结（2026-07-28）",
        "",
        "## 一句话结论",
        "",
        (
            "> 8k 开发结果显示：按训练分子中具体结构组合的实际变化来压缩，明显比随机压缩稳定；"
            "当前最终判断取决于 full official-train 三折开发验证。"
            if full_gate is None
            else (
                "> full official-train 三折达到预先固定的全部条件；下一步应先冻结配置，再只评估一次 official valid。"
                if full_gate["passed"]
                else "> full official-train 三折没有达到预先固定的全部条件；不应触碰 official valid/test，应停止 pair-PCA 分支并保留 compact exact-distance 关系模型。"
            )
        ),
        "",
        "## 8k：三个训练随机种子",
        "",
        "这里改变的只是模型训练中的随机性；结构词汇、PCA 压缩方向和数据划分保持不变。",
        "",
        "| seed | compact base | covariance PCA | 增益 | real−打乱结构归属 | base 胜折数 | 对照胜折数 |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for seed in dev_seeds:
        item = dev_by_seed[str(seed)]
        cov = item["methods"]["covariance_pca"]
        lines.append(
            f"| {seed} | {item['base_compact_mean_auc']:.6f} | {cov['real_mean_auc']:.6f} | "
            f"{cov['real_mean_gain_over_base']:+.6f} | {cov['real_minus_assignment_shuffled_mean']:+.6f} | "
            f"{cov['real_base_fold_wins']}/3 | {cov['real_assignment_shuffled_fold_wins']}/3 |"
        )
    lines += [
        "",
        "固定三 seed 概率平均：",
        "",
    ]
    ens = dev_ensemble["covariance_pca"]
    lines += [
        f"- real fold AUC：`{fmt(ens['real_fold_auc'])}`",
        f"- mean gain over compact base：`{ens['real_mean_gain_over_base']:+.6f}`",
        f"- real − assignment-shuffled：`{ens['real_minus_assignment_shuffled_mean']:+.6f}`",
        "",
        "### 为什么 covariance PCA 更可信",
        "",
        "它不看分子标签，只观察 outer-fit 分子里哪些具体结构组合经常一起变化。"
        "因此，保留下来的方向来自数据中反复出现的结构规律，而不是任意随机方向。",
        "",
        "相反，correlation PCA 会先把每个组合都缩放到近似同等重要。"
        "这会把非常少见、估计不稳定的组合放大；8k 结果表明这种处理平均没有收益。",
    ]

    if full_summary is not None and full_gate is not None:
        base = full_summary["base_compact_fold_auc"]
        cov = full_summary["methods"]["covariance_pca"]
        cor = full_summary["methods"]["correlation_pca"]
        lines += [
            "",
            "## Full official-train 三折开发验证",
            "",
            "这仍然只是在 official train 内部做开发验证。official valid/test 的编码和评估次数都保持为 0。",
            "",
            f"- compact base：`{fmt(base)}`，mean `{full_summary['base_compact_mean_auc']:.6f}`",
            f"- covariance PCA：`{fmt(cov['real_fold_auc'])}`，mean `{cov['real_mean_auc']:.6f}`",
            f"- covariance 增益：`{fmt(cov['real_fold_gain_over_base'], signed=True)}`，mean `{cov['real_mean_gain_over_base']:+.6f}`",
            f"- covariance real − assignment-shuffled：`{fmt(cov['real_minus_assignment_shuffled_fold'], signed=True)}`，mean `{cov['real_minus_assignment_shuffled_mean']:+.6f}`",
            f"- correlation PCA mean gain：`{cor['real_mean_gain_over_base']:+.6f}`",
            "",
            "### 预先固定的晋级条件",
            "",
        ]
        for key, passed in full_gate["conditions"].items():
            lines.append(f"- {'PASS' if passed else 'FAIL'} — `{key}`")
        lines += [
            "",
            f"**总判断：{'PASS' if full_gate['passed'] else 'FAIL'}。**",
        ]

    lines += [
        "",
        "## 当前研究解释",
        "",
        "1. 单个局部结构是否出现还不够；距离 1/2 内的具体结构组合确实可能补充弱信息。",
        "2. 但组合空间很大，不能任意压缩。随机投影的结果不稳定，说明信号不是“随便保留一些组合”就能得到。",
        "3. covariance PCA 的稳定性说明，更有希望的是保留在许多训练分子中反复共同变化、变化幅度较大的组合。",
        "4. 这条路线仍然不是把字典完全按标签训练；结构词汇和压缩都不看标签，只有最后的性质预测器看标签。这样能减少过拟合和数据泄漏风险。",
        "5. official valid/test 当前均未用于编码、选参或评估。",
        "",
        "## 后续纪律",
        "",
        "- 若 full 门槛通过：先写冻结清单，再进行一次 official valid；不能看到 valid 后再改 rank、残差上限或重新启用失败分支。",
        "- 若 full 门槛失败：保留 compact exact-distance 关系模型，停止 pair-PCA，不评估 official valid/test。",
    ]
    Path(args.output_report).write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"output_json": args.output_json, "output_report": args.output_report, "decision": summary["decision"]}, indent=2))


if __name__ == "__main__":
    main()

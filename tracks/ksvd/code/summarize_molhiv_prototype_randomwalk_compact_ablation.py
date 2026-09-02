"""Summarize the three-fold compact local-random-walk ablation."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

FAMILIES = ("farthest", "scaffold_facility")
FAMILY_LABEL = {"farthest": "Farthest", "scaffold_facility": "Scaffold"}


def sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def mean_metrics(real: np.ndarray, shuffled: np.ndarray, base: np.ndarray) -> dict[str, Any]:
    return {
        "base_auc_by_fold": base.tolist(),
        "base_auc_mean": float(base.mean()),
        "real_auc_by_fold": real.tolist(),
        "real_auc_mean": float(real.mean()),
        "shuffled_auc_by_fold": shuffled.tolist(),
        "shuffled_auc_mean": float(shuffled.mean()),
        "real_minus_base_by_fold": (real - base).tolist(),
        "real_minus_base_mean": float((real - base).mean()),
        "real_minus_shuffled_by_fold": (real - shuffled).tolist(),
        "real_minus_shuffled_mean": float((real - shuffled).mean()),
        "real_beats_base_folds": int(np.sum(real > base)),
        "real_beats_shuffled_folds": int(np.sum(real > shuffled)),
        "strict_promotion_gate": bool(
            (real - base).mean() >= 0.005
            and np.sum(real > base) >= 2
            and real.mean() > shuffled.mean()
        ),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", nargs="+", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--markdown", required=True)
    ap.add_argument("--exact-distance-summary", required=True)
    args = ap.parse_args()
    docs = [json.loads(Path(p).read_text()) for p in args.inputs]
    docs.sort(key=lambda d: int(d["fold"]))
    folds = [int(d["fold"]) for d in docs]
    if folds != [0, 1, 2]:
        raise ValueError(f"expected folds [0,1,2], got {folds}")
    if any(d.get("protocol_id") != "molhiv_prototype_randomwalk_compact_ablation_v1" for d in docs):
        raise ValueError("protocol mismatch")
    ablations = list(docs[0]["ablations"])
    if any(list(d["ablations"]) != ablations for d in docs[1:]):
        raise ValueError("ablation mismatch")

    summary: dict[str, Any] = {
        "protocol_id": "molhiv_prototype_randomwalk_compact_ablation_summary_v1",
        "date": "2026-07-28",
        "inputs": args.inputs,
        "folds": folds,
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "per_operator_dim": int(docs[0]["per_operator_dim"]),
        "statistics": docs[0]["statistics"],
        "ablations": docs[0]["ablations"],
        "families": {},
        "pair_ensembles": {},
    }

    for family in FAMILIES:
        family_summary: dict[str, Any] = {}
        base = np.asarray([d["base_occurrence"][family]["heldout_auc"] for d in docs], dtype=float)
        for ablation in ablations:
            real = np.asarray([
                d["results"][f"{family}__{ablation}__real"]["heldout"]["auc"] for d in docs
            ], dtype=float)
            shuffled = np.asarray([
                d["results"][f"{family}__{ablation}__shuffled"]["heldout"]["auc"] for d in docs
            ], dtype=float)
            row = mean_metrics(real, shuffled, base)
            row["relation_dim"] = int(docs[0]["results"][f"{family}__{ablation}__real"]["relation_dim"])
            row["heldout_mean_abs_residual_by_fold"] = [
                float(d["results"][f"{family}__{ablation}__real"]["heldout"]["mean_abs_relation_residual"])
                for d in docs
            ]
            family_summary[ablation] = row
        summary["families"][family] = family_summary

    for ablation in ablations:
        base_auc: list[float] = []
        real_auc: list[float] = []
        shuffled_auc: list[float] = []
        for d in docs:
            heldout_by_family = {}
            for family in FAMILIES:
                real_raw = d["results"][f"{family}__{ablation}__real"]["heldout"]
                shuffled_raw = d["results"][f"{family}__{ablation}__shuffled"]["heldout"]
                for raw_name, raw in (("real", real_raw), ("shuffled", shuffled_raw)):
                    if not all(k in raw for k in ("graph_indices", "labels", "scores", "base_scores")):
                        raise ValueError("saved predictions are required for pair ensembles")
                heldout_by_family[family] = (real_raw, shuffled_raw)
            f_real, f_shuf = heldout_by_family["farthest"]
            s_real, s_shuf = heldout_by_family["scaffold_facility"]
            indices = np.asarray(f_real["graph_indices"], dtype=np.int64)
            labels = np.asarray(f_real["labels"], dtype=np.float64)
            checks = (
                np.asarray(s_real["graph_indices"], dtype=np.int64),
                np.asarray(f_shuf["graph_indices"], dtype=np.int64),
                np.asarray(s_shuf["graph_indices"], dtype=np.int64),
            )
            if any(not np.array_equal(indices, x) for x in checks):
                raise AssertionError("pair ensemble graph order mismatch")
            if any(not np.array_equal(labels, np.asarray(x["labels"], dtype=np.float64)) for x in (s_real, f_shuf, s_shuf)):
                raise AssertionError("pair ensemble labels mismatch")
            base_prob = 0.5 * (
                sigmoid(np.asarray(f_real["base_scores"])) + sigmoid(np.asarray(s_real["base_scores"]))
            )
            real_prob = 0.5 * (
                sigmoid(np.asarray(f_real["scores"])) + sigmoid(np.asarray(s_real["scores"]))
            )
            shuf_prob = 0.5 * (
                sigmoid(np.asarray(f_shuf["scores"])) + sigmoid(np.asarray(s_shuf["scores"]))
            )
            base_auc.append(float(roc_auc_score(labels, base_prob)))
            real_auc.append(float(roc_auc_score(labels, real_prob)))
            shuffled_auc.append(float(roc_auc_score(labels, shuf_prob)))
        row = mean_metrics(np.asarray(real_auc), np.asarray(shuffled_auc), np.asarray(base_auc))
        row["relation_dim_per_family"] = int(
            docs[0]["results"][f"farthest__{ablation}__real"]["relation_dim"]
        )
        summary["pair_ensembles"][ablation] = row

    candidates: list[dict[str, Any]] = []
    for scope, rows in [(f"family:{f}", summary["families"][f]) for f in FAMILIES] + [("pair", summary["pair_ensembles"])]:
        for ablation, row in rows.items():
            candidates.append({
                "scope": scope,
                "ablation": ablation,
                "real_auc_mean": row["real_auc_mean"],
                "real_minus_base_mean": row["real_minus_base_mean"],
                "real_minus_shuffled_mean": row["real_minus_shuffled_mean"],
                "real_beats_base_folds": row["real_beats_base_folds"],
                "real_beats_shuffled_folds": row["real_beats_shuffled_folds"],
                "strict_promotion_gate": row["strict_promotion_gate"],
            })
    candidates.sort(
        key=lambda x: (x["real_minus_shuffled_mean"], x["real_minus_base_mean"], x["real_auc_mean"]),
        reverse=True,
    )
    summary["diagnostic_ranking"] = candidates
    summary["strict_promotion_passed"] = bool(any(x["strict_promotion_gate"] for x in candidates))
    exact = json.loads(Path(args.exact_distance_summary).read_text())
    comparison_map = {
        "rw2_exact_distance2": "distance_2",
        "rw1_rw2_exact_distance2": "distance_1_2",
        "rw1_rw2_nonreturn": "distance_1_2",
        "rw1_rw2_nonreturn_mix": "distance_1_2",
    }
    exact_comparisons: dict[str, Any] = {"source": args.exact_distance_summary, "families": {}, "pair": {}}
    for family in FAMILIES:
        exact_comparisons["families"][family] = {}
        for rw_name, exact_name in comparison_map.items():
            rw_row = summary["families"][family][rw_name]
            exact_row = exact["families"][family][exact_name]
            delta = np.asarray(rw_row["real_auc_by_fold"]) - np.asarray(exact_row["real_auc_by_fold"])
            exact_comparisons["families"][family][rw_name] = {
                "exact_reference": exact_name,
                "rw_minus_exact_by_fold": delta.tolist(),
                "rw_minus_exact_mean": float(delta.mean()),
                "rw_beats_exact_folds": int(np.sum(delta > 0)),
            }
    for rw_name, exact_name in comparison_map.items():
        rw_row = summary["pair_ensembles"][rw_name]
        exact_row = exact["pair_ensembles"][exact_name]
        delta = np.asarray(rw_row["real_auc_by_fold"]) - np.asarray(exact_row["real_auc_by_fold"])
        exact_comparisons["pair"][rw_name] = {
            "exact_reference": exact_name,
            "rw_minus_exact_by_fold": delta.tolist(),
            "rw_minus_exact_mean": float(delta.mean()),
            "rw_beats_exact_folds": int(np.sum(delta > 0)),
        }
    summary["comparison_to_exact_distance"] = exact_comparisons
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Compact local random-walk relation ablation（2026-07-28）",
        "",
        "## 1. Protocol",
        "",
        "- 8,000-graph subset；只使用 6,400 个 official-train graphs 的三个 outer scaffold folds。",
        "- 直接复用上一轮保存的 occurrence logits；base 不重训、不解冻，因此 baseline 完全相同。",
        "- official valid/test 编码与评估均为 **0**。",
        f"- 每个 walk operator 固定压缩为 `{summary['per_operator_dim']}` 维：`diag(C)`、`row_sum(C)`、trace/off-diagonal mass、prototype semantic similarity、support fraction 与 active mass。",
        "- 使用 reversible stationary flow `Q_t = diag(pi)P^t`；每个 ablation 独立训练 zero-init bounded linear residual，real/shuffled 协议完全相同。",
        "- 这是 exact distance 1–2 之后的 operator-value 实验；沿用 `mean gain >= +0.005`、赢 base 至少 2/3、且 real mean > shuffled mean 的严格 promotion gate。",
        "",
        "## 2. Single-vocabulary random-walk means",
        "",
        "| Vocabulary | Walk operator | Dim | Base | Real | Shuffled | Real−base | Real−shuffled | Wins base | Wins shuffled |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for family in FAMILIES:
        for ablation in ablations:
            row = summary["families"][family][ablation]
            lines.append(
                f"| {FAMILY_LABEL[family]} | {ablation} | {row['relation_dim']} | "
                f"{row['base_auc_mean']:.6f} | {row['real_auc_mean']:.6f} | {row['shuffled_auc_mean']:.6f} | "
                f"{row['real_minus_base_mean']:+.6f} | {row['real_minus_shuffled_mean']:+.6f} | "
                f"{row['real_beats_base_folds']}/3 | {row['real_beats_shuffled_folds']}/3 |"
            )

    lines += [
        "",
        "## 3. Farthest + scaffold random-walk probability ensembles",
        "",
        "| Walk operator | Dim/family | Fold 0 real | Fold 1 real | Fold 2 real | Base mean | Real mean | Shuffled mean | Real−base | Real−shuffled |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for ablation in ablations:
        row = summary["pair_ensembles"][ablation]
        folds_real = row["real_auc_by_fold"]
        lines.append(
            f"| {ablation} | {row['relation_dim_per_family']} | {folds_real[0]:.6f} | {folds_real[1]:.6f} | {folds_real[2]:.6f} | "
            f"{row['base_auc_mean']:.6f} | {row['real_auc_mean']:.6f} | {row['shuffled_auc_mean']:.6f} | "
            f"{row['real_minus_base_mean']:+.6f} | {row['real_minus_shuffled_mean']:+.6f} |"
        )

    top = candidates[:6]
    lines += [
        "",
        "## 4. Assignment-specific diagnostic ranking",
        "",
        "按 `real − shuffled` 优先、再按 `real − base` 排序：",
        "",
        "| Rank | Scope | Walk operator | Real | Real−base | Real−shuffled | Wins shuffled |",
        "|---:|---|---|---:|---:|---:|---:|",
    ]
    for rank, row in enumerate(top, start=1):
        lines.append(
            f"| {rank} | {row['scope']} | {row['ablation']} | {row['real_auc_mean']:.6f} | "
            f"{row['real_minus_base_mean']:+.6f} | {row['real_minus_shuffled_mean']:+.6f} | "
            f"{row['real_beats_shuffled_folds']}/3 |"
        )

    pair_best = max(
        ((name, row) for name, row in summary["pair_ensembles"].items()),
        key=lambda item: (item[1]["real_minus_shuffled_mean"], item[1]["real_minus_base_mean"]),
    )
    pair_exact_best = max(
        summary["comparison_to_exact_distance"]["pair"].items(),
        key=lambda item: item[1]["rw_minus_exact_mean"],
    )
    lines += [
        "",
        "## 5. Decision",
        "",
        f"- 严格 promotion：**{summary['strict_promotion_passed']}**。",
        f"- Pair 中 assignment-specific 最强 RW：`{pair_best[0]}`；real−base `{pair_best[1]['real_minus_base_mean']:+.6f}`，real−shuffled `{pair_best[1]['real_minus_shuffled_mean']:+.6f}`。",
        f"- 相对对应 exact-distance baseline 最好的 pair RW：`{pair_exact_best[0]}`，mean delta `{pair_exact_best[1]['rw_minus_exact_mean']:+.6f}`，wins `{pair_exact_best[1]['rw_beats_exact_folds']}/3`。",
        "",
        "## 6. Direct comparison with exact distance operators",
        "",
        "| Scope | RW operator | Exact reference | RW−exact mean | Wins |",
        "|---|---|---|---:|---:|",
    ]
    for family in FAMILIES:
        for rw_name, row in summary["comparison_to_exact_distance"]["families"][family].items():
            lines.append(
                f"| {FAMILY_LABEL[family]} | {rw_name} | {row['exact_reference']} | "
                f"{row['rw_minus_exact_mean']:+.6f} | {row['rw_beats_exact_folds']}/3 |"
            )
    for rw_name, row in summary["comparison_to_exact_distance"]["pair"].items():
        lines.append(
            f"| Pair | {rw_name} | {row['exact_reference']} | "
            f"{row['rw_minus_exact_mean']:+.6f} | {row['rw_beats_exact_folds']}/3 |"
        )
    exact_pair_12 = exact["pair_ensembles"]["distance_1_2"]
    all_pair_rw_lose = all(
        row["rw_beats_exact_folds"] == 0
        for row in summary["comparison_to_exact_distance"]["pair"].values()
    )
    lines += [
        "",
        "判定原则：只有 RW 在相同 frozen base、相同 compact statistics 下稳定超过对应 exact-distance operator，才能宣称 random walk 本身提供了额外价值。",
        "",
        "## 7. Research conclusion",
        "",
        f"- 所有映射到 exact baseline 的 pair RW 是否均为 0/3 wins：**{all_pair_rw_lose}**。",
        f"- Exact distance 1+2 pair：`{exact_pair_12['real_auc_mean']:.6f}`；最佳对应 RW pair：`{summary['pair_ensembles']['rw1_rw2_nonreturn']['real_auc_mean']:.6f}`，差值 `{summary['comparison_to_exact_distance']['pair']['rw1_rw2_nonreturn']['rw_minus_exact_mean']:+.6f}`。",
        "- `rw1_stationary = diag(pi)P` 在无向图上等价于 uniformly normalized directed-edge mask；真正新增的只有二步 path/degree weighting。",
        "- 二步 weighting 没有增加 operator value；return removal 有益于 RW 内部比较，但仍稳定弱于 uniform exact distance 1+2。",
        "- 因而不晋级该 random-walk family，也不触碰 full official valid/test。下一步应把精力从 relation operator 转向 task-matched vocabulary / prototype selection。",
    ]
    Path(args.markdown).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "markdown": args.markdown,
        "strict_promotion_passed": summary["strict_promotion_passed"],
        "best_pair_ablation": pair_best[0],
        "best_pair": pair_best[1],
        "best_pair_vs_exact": {"operator": pair_exact_best[0], **pair_exact_best[1]},
    }, indent=2))


if __name__ == "__main__":
    main()

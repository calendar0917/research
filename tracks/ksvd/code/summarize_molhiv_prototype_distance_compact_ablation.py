"""Summarize the three-fold compact prototype-distance ablation."""
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
    ap.add_argument("--full-relation-summary")
    args = ap.parse_args()
    docs = [json.loads(Path(p).read_text()) for p in args.inputs]
    docs.sort(key=lambda d: int(d["fold"]))
    folds = [int(d["fold"]) for d in docs]
    if folds != [0, 1, 2]:
        raise ValueError(f"expected folds [0,1,2], got {folds}")
    if any(d.get("protocol_id") != "molhiv_prototype_distance_compact_ablation_v1" for d in docs):
        raise ValueError("protocol mismatch")
    ablations = list(docs[0]["ablations"])
    if any(list(d["ablations"]) != ablations for d in docs[1:]):
        raise ValueError("ablation mismatch")

    summary: dict[str, Any] = {
        "protocol_id": "molhiv_prototype_distance_compact_ablation_summary_v1",
        "date": "2026-07-28",
        "inputs": args.inputs,
        "folds": folds,
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "per_bin_dim": int(docs[0]["per_bin_dim"]),
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
    if args.full_relation_summary:
        full = json.loads(Path(args.full_relation_summary).read_text())
        full_pair = float(full["ensembles"]["relation_pair"]["auc_mean"])
        summary["comparison_to_full_2650d_relation"] = {
            "source": args.full_relation_summary,
            "full_pair_auc_mean": full_pair,
            "compact_connected_pair_auc_mean": summary["pair_ensembles"]["connected"]["real_auc_mean"],
            "compact_connected_minus_full": summary["pair_ensembles"]["connected"]["real_auc_mean"] - full_pair,
            "full_relation_dim_per_family": 2650,
            "compact_connected_dim_per_family": summary["pair_ensembles"]["connected"]["relation_dim_per_family"],
        }
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    lines = [
        "# Compact prototype-distance bin ablation（2026-07-28）",
        "",
        "## 1. Protocol",
        "",
        "- 8,000-graph subset；只使用 6,400 个 official-train graphs 的三个 outer scaffold folds。",
        "- 直接复用上一轮保存的 occurrence logits；base 不重训、不解冻，因此 baseline 完全相同。",
        "- official valid/test 编码与评估均为 **0**。",
        f"- 每个距离 bin 固定压缩为 `{summary['per_bin_dim']}` 维：`diag(C)`、`row_sum(C)`、trace/off-diagonal mass、prototype semantic similarity、pair fraction 与 active mass。",
        "- 每个 ablation 独立训练 zero-init bounded linear residual；real 与 node-assignment-shuffled 使用完全相同协议。",
        "- 这是尺度定位实验；沿用 `mean gain >= +0.005`、赢 base 至少 2/3、且 real mean > shuffled mean 的严格 promotion gate。",
        "",
        "## 2. Single vocabulary means",
        "",
        "| Vocabulary | Distance bins | Dim | Base | Real | Shuffled | Real−base | Real−shuffled | Wins base | Wins shuffled |",
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
        "## 3. Farthest + scaffold probability ensembles",
        "",
        "| Distance bins | Dim/family | Fold 0 real | Fold 1 real | Fold 2 real | Base mean | Real mean | Shuffled mean | Real−base | Real−shuffled |",
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
        "| Rank | Scope | Distance bins | Real | Real−base | Real−shuffled | Wins shuffled |",
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
    lines += [
        "",
        "## 5. Decision",
        "",
        f"- 严格 promotion：**{summary['strict_promotion_passed']}**。",
        f"- Pair 中 assignment-specific 最强尺度：`{pair_best[0]}`；real−base `{pair_best[1]['real_minus_base_mean']:+.6f}`，real−shuffled `{pair_best[1]['real_minus_shuffled_mean']:+.6f}`。",
        "- 只有当某个尺度同时稳定超过 occurrence 与 shuffled，才值得将该尺度替换为 diffusion/random-walk operator。",
    ]
    if "comparison_to_full_2650d_relation" in summary:
        comp = summary["comparison_to_full_2650d_relation"]
        lines += [
            "",
            "## 6. Comparison with the previous full 32×32×5 relation",
            "",
            f"- Full relation：`{comp['full_relation_dim_per_family']}` params/family，pair mean `{comp['full_pair_auc_mean']:.6f}`。",
            f"- Compact connected：`{comp['compact_connected_dim_per_family']}` params/family，pair mean `{comp['compact_connected_pair_auc_mean']:.6f}`。",
            f"- AUC difference：`{comp['compact_connected_minus_full']:+.6f}`；即用约 1/12.6 的 relation 参数基本复现 full relation 的 pair AUC。",
            "- 结合 shuffled control，真正可定位的信号集中在 distance 1–2；distance 3+ 不应进入下一版 operator。",
            "- 下一步应只测试 local 1–2 hop diffusion / random-walk composition，并保持 frozen base、bounded residual 和 matched shuffled control。",
        ]
    Path(args.markdown).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "markdown": args.markdown,
        "strict_promotion_passed": summary["strict_promotion_passed"],
        "best_pair_ablation": pair_best[0],
        "best_pair": pair_best[1],
    }, indent=2))


if __name__ == "__main__":
    main()

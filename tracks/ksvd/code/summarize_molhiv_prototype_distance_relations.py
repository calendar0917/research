"""Summarize exact shortest-path prototype-relation experiments on 8k scaffold folds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.metrics import roc_auc_score

FAMILIES = {
    "farthest": {
        "occurrence": "farthest_occurrence",
        "relation": "farthest_relation",
        "shuffled": "farthest_relation_shuffled",
    },
    "scaffold": {
        "occurrence": "scaffold_occurrence",
        "relation": "scaffold_relation",
        "shuffled": "scaffold_relation_shuffled",
    },
}
REFERENCE_OCCURRENCE_AUC = {
    "farthest": np.asarray([0.7507836990595611, 0.7037985428308693, 0.7766044229175960]),
    "scaffold": np.asarray([0.7505774624649398, 0.6870128032821674, 0.7932509081639156]),
}
REFERENCE_OCCURRENCE_PAIR = np.asarray(
    [0.7668856211846231, 0.6993987408926929, 0.8009240966158945]
)


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def load_prediction(doc: dict[str, Any], control: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    heldout = doc["results"][control]["heldout"]
    indices = np.asarray(heldout["graph_indices"], dtype=np.int64)
    labels = np.asarray(heldout["labels"], dtype=np.int64)
    probability = sigmoid(np.asarray(heldout["scores"], dtype=np.float64))
    order = np.argsort(indices)
    return indices[order], labels[order], probability[order]


def ensemble_by_fold(
    docs: list[dict[str, Any]], controls: tuple[str, ...]
) -> tuple[list[float], list[float]]:
    aucs: list[float] = []
    correlations: list[float] = []
    for doc in docs:
        predictions = [load_prediction(doc, control) for control in controls]
        base_indices, labels, _ = predictions[0]
        probabilities = []
        for indices_i, labels_i, probability_i in predictions:
            if not np.array_equal(indices_i, base_indices) or not np.array_equal(labels_i, labels):
                raise ValueError(f"prediction alignment mismatch for controls={controls}")
            probabilities.append(probability_i)
        mean_probability = np.mean(np.stack(probabilities), axis=0)
        aucs.append(float(roc_auc_score(labels, mean_probability)))
        if len(probabilities) == 2:
            correlations.append(float(np.corrcoef(probabilities[0], probabilities[1])[0, 1]))
    return aucs, correlations


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="results/molhiv")
    ap.add_argument(
        "--output",
        default="results/molhiv/prototype_distance_relations_summary_scaffold3.json",
    )
    ap.add_argument(
        "--markdown",
        default="results/molhiv/REALPROTOTYPE_DISTANCE_RELATIONS_PILOT_20260728.md",
    )
    args = ap.parse_args()
    root = Path(args.result_dir)
    docs = [
        json.loads((root / f"prototype_distance_relations_fold{fold}_seed0.json").read_text())
        for fold in range(3)
    ]
    if [int(doc["fold"]) for doc in docs] != [0, 1, 2]:
        raise ValueError("unexpected fold ids")
    if any(doc["selection_policy"]["official_valid_evaluations"] != 0 for doc in docs):
        raise ValueError("official valid was evaluated")
    if any(doc["selection_policy"]["official_test_evaluations"] != 0 for doc in docs):
        raise ValueError("official test was evaluated")

    summary: dict[str, Any] = {
        "protocol_id": "molhiv-prototype-exact-distance-relations-scaffold-summary-v1",
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
        "families": {},
        "ensembles": {},
        "diagnostics": {
            "relation_dim": [int(doc["relation_dim"]) for doc in docs],
            "relation_precompute_sec": [float(doc["relation_precompute_sec"]) for doc in docs],
            "n_disconnected_graphs": [int(doc["n_disconnected_graphs"]) for doc in docs],
            "distance_pair_fraction_mean_by_fold": [doc["distance_pair_fraction_mean"] for doc in docs],
        },
    }

    family_gate_passes = []
    for family, controls in FAMILIES.items():
        occurrence = np.asarray([
            doc["results"][controls["occurrence"]]["heldout"]["auc"] for doc in docs
        ])
        relation = np.asarray([
            doc["results"][controls["relation"]]["heldout"]["auc"] for doc in docs
        ])
        shuffled = np.asarray([
            doc["results"][controls["shuffled"]]["heldout"]["auc"] for doc in docs
        ])
        relation_base = np.asarray([
            doc["results"][controls["relation"]]["heldout"]["base_auc"] for doc in docs
        ])
        gates = np.asarray([
            doc["results"][controls["relation"]]["relation_gate_tanh"] for doc in docs
        ])
        shuffled_gates = np.asarray([
            doc["results"][controls["shuffled"]]["relation_gate_tanh"] for doc in docs
        ])
        residuals = np.asarray([
            doc["results"][controls["relation"]]["heldout"]["mean_abs_relation_residual"]
            for doc in docs
        ])
        real_minus_occurrence = relation - occurrence
        real_minus_shuffled = relation - shuffled
        gate = {
            "relation_minus_occurrence_auc_by_fold": real_minus_occurrence.tolist(),
            "relation_minus_occurrence_auc_mean": float(real_minus_occurrence.mean()),
            "relation_occurrence_fold_wins": int(np.sum(relation > occurrence)),
            "relation_minus_shuffled_auc_by_fold": real_minus_shuffled.tolist(),
            "relation_minus_shuffled_auc_mean": float(real_minus_shuffled.mean()),
            "gain_at_least_005": bool(real_minus_occurrence.mean() >= 0.005),
            "wins_at_least_2_of_3": bool(np.sum(relation > occurrence) >= 2),
            "beats_shuffled_mean": bool(relation.mean() > shuffled.mean()),
        }
        gate["promotion_passed"] = bool(
            gate["gain_at_least_005"]
            and gate["wins_at_least_2_of_3"]
            and gate["beats_shuffled_mean"]
        )
        family_gate_passes.append(gate["promotion_passed"])
        reference_delta = occurrence - REFERENCE_OCCURRENCE_AUC[family]
        summary["families"][family] = {
            "occurrence_auc_by_fold": occurrence.tolist(),
            "occurrence_auc_mean": float(occurrence.mean()),
            "relation_auc_by_fold": relation.tolist(),
            "relation_auc_mean": float(relation.mean()),
            "relation_auc_std": float(relation.std(ddof=1)),
            "shuffled_auc_by_fold": shuffled.tolist(),
            "shuffled_auc_mean": float(shuffled.mean()),
            "relation_base_auc_by_fold": relation_base.tolist(),
            "relation_gate_tanh_by_fold": gates.tolist(),
            "shuffled_gate_tanh_by_fold": shuffled_gates.tolist(),
            "heldout_mean_abs_relation_residual_by_fold": residuals.tolist(),
            "occurrence_reference_delta_by_fold": reference_delta.tolist(),
            "occurrence_reference_max_abs_delta": float(np.max(np.abs(reference_delta))),
            "promotion_gate": gate,
        }

    ensembles = {
        "occurrence_pair": (
            FAMILIES["farthest"]["occurrence"],
            FAMILIES["scaffold"]["occurrence"],
        ),
        "relation_pair": (
            FAMILIES["farthest"]["relation"],
            FAMILIES["scaffold"]["relation"],
        ),
        "shuffled_pair": (
            FAMILIES["farthest"]["shuffled"],
            FAMILIES["scaffold"]["shuffled"],
        ),
    }
    for name, controls in ensembles.items():
        aucs, correlations = ensemble_by_fold(docs, controls)
        summary["ensembles"][name] = {
            "controls": list(controls),
            "auc_by_fold": aucs,
            "auc_mean": float(np.mean(aucs)),
            "probability_pearson_by_fold": correlations,
            "probability_pearson_mean": float(np.mean(correlations)),
        }
    occurrence_pair = np.asarray(summary["ensembles"]["occurrence_pair"]["auc_by_fold"])
    relation_pair = np.asarray(summary["ensembles"]["relation_pair"]["auc_by_fold"])
    shuffled_pair = np.asarray(summary["ensembles"]["shuffled_pair"]["auc_by_fold"])
    pair_delta = relation_pair - occurrence_pair
    pair_gate = {
        "relation_minus_occurrence_auc_by_fold": pair_delta.tolist(),
        "relation_minus_occurrence_auc_mean": float(pair_delta.mean()),
        "relation_occurrence_fold_wins": int(np.sum(relation_pair > occurrence_pair)),
        "relation_minus_shuffled_auc_mean": float((relation_pair - shuffled_pair).mean()),
        "gain_at_least_005": bool(pair_delta.mean() >= 0.005),
        "wins_at_least_2_of_3": bool(np.sum(relation_pair > occurrence_pair) >= 2),
        "beats_shuffled_mean": bool(relation_pair.mean() > shuffled_pair.mean()),
    }
    pair_gate["promotion_passed"] = bool(
        pair_gate["gain_at_least_005"]
        and pair_gate["wins_at_least_2_of_3"]
        and pair_gate["beats_shuffled_mean"]
    )
    summary["ensembles"]["relation_pair"]["promotion_gate"] = pair_gate
    summary["ensembles"]["occurrence_pair"]["reference_delta_by_fold"] = (
        occurrence_pair - REFERENCE_OCCURRENCE_PAIR
    ).tolist()
    summary["ensembles"]["occurrence_pair"]["reference_max_abs_delta"] = float(
        np.max(np.abs(occurrence_pair - REFERENCE_OCCURRENCE_PAIR))
    )
    summary["overall_promotion"] = {
        "any_single_family_passed": bool(any(family_gate_passes)),
        "pair_ensemble_passed": bool(pair_gate["promotion_passed"]),
        "promote_exact_distance_relations": bool(any(family_gate_passes) or pair_gate["promotion_passed"]),
    }

    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    f = summary["families"]["farthest"]
    s = summary["families"]["scaffold"]
    eo = summary["ensembles"]["occurrence_pair"]
    er = summary["ensembles"]["relation_pair"]
    es = summary["ensembles"]["shuffled_pair"]
    lines = [
        "# Real-prototype exact-distance relations pilot（2026-07-28）",
        "",
        "## 1. 数据与边界",
        "",
        "- 使用 8,000-graph development subset 中的 6,400 个 official-train graphs。",
        "- 三个 official-train-only Bemis–Murcko scaffold outer folds；固定 seed 0、30 epochs。",
        "- official valid/test 编码与评估次数均为 **0**。",
        "- 冻结 broad-pool `farthest` 与 `scaffold_facility` vocabulary，不重新选择 prototype。",
        "",
        "## 2. 方法",
        "",
        "- 对 node→prototype positive-cosine top-3 assignment 构造 `C[d] = Z^T M[d] Z`。",
        "- 距离 bins：self、1、2、3+、disconnected；关系向量维度 2,650。",
        "- 关系支路只作为 zero-initialized gated scalar residual，不含 supervised message passing。",
        "- matched shuffled control 固定分子距离矩阵并置乱节点 assignment，保持 prototype marginals。",
        "",
        "## 3. 三折结果",
        "",
        "| Vocabulary | Model | Fold 0 | Fold 1 | Fold 2 | Mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for family_name, row in (("farthest", f), ("scaffold", s)):
        for model_name, key in (
            ("occurrence", "occurrence_auc_by_fold"),
            ("exact relation", "relation_auc_by_fold"),
            ("distance-shuffled", "shuffled_auc_by_fold"),
        ):
            vals = row[key]
            lines.append(
                f"| {family_name} | {model_name} | {vals[0]:.6f} | {vals[1]:.6f} | "
                f"{vals[2]:.6f} | {np.mean(vals):.6f} |"
            )
    lines += [
        "",
        "单 vocabulary promotion gate：",
        "",
        f"- farthest relation − occurrence：`{f['promotion_gate']['relation_minus_occurrence_auc_mean']:+.6f}`，"
        f"赢 `{f['promotion_gate']['relation_occurrence_fold_wins']}/3`；relation − shuffled "
        f"`{f['promotion_gate']['relation_minus_shuffled_auc_mean']:+.6f}`；"
        f"promotion = **{f['promotion_gate']['promotion_passed']}**。",
        f"- scaffold relation − occurrence：`{s['promotion_gate']['relation_minus_occurrence_auc_mean']:+.6f}`，"
        f"赢 `{s['promotion_gate']['relation_occurrence_fold_wins']}/3`；relation − shuffled "
        f"`{s['promotion_gate']['relation_minus_shuffled_auc_mean']:+.6f}`；"
        f"promotion = **{s['promotion_gate']['promotion_passed']}**。",
        "",
        "## 4. Vocabulary-pair ensemble",
        "",
        "| Ensemble | Fold 0 | Fold 1 | Fold 2 | Mean |",
        "|---|---:|---:|---:|---:|",
        f"| occurrence pair | {eo['auc_by_fold'][0]:.6f} | {eo['auc_by_fold'][1]:.6f} | {eo['auc_by_fold'][2]:.6f} | {eo['auc_mean']:.6f} |",
        f"| exact-relation pair | {er['auc_by_fold'][0]:.6f} | {er['auc_by_fold'][1]:.6f} | {er['auc_by_fold'][2]:.6f} | {er['auc_mean']:.6f} |",
        f"| shuffled-relation pair | {es['auc_by_fold'][0]:.6f} | {es['auc_by_fold'][1]:.6f} | {es['auc_by_fold'][2]:.6f} | {es['auc_mean']:.6f} |",
        "",
        f"Pair relation − occurrence：`{pair_gate['relation_minus_occurrence_auc_mean']:+.6f}`，"
        f"赢 `{pair_gate['relation_occurrence_fold_wins']}/3`；relation − shuffled "
        f"`{pair_gate['relation_minus_shuffled_auc_mean']:+.6f}`；promotion = **{pair_gate['promotion_passed']}**。",
        "",
        "## 5. 审计与判断",
        "",
        f"- occurrence baseline 最大复现误差：farthest `{f['occurrence_reference_max_abs_delta']:.3e}`，"
        f"scaffold `{s['occurrence_reference_max_abs_delta']:.3e}`，pair `{eo['reference_max_abs_delta']:.3e}`。",
        f"- disconnected graphs by fold：`{summary['diagnostics']['n_disconnected_graphs']}`。",
        f"- exact shortest-path relation 总 promotion：**{summary['overall_promotion']['promote_exact_distance_relations']}**。",
        "",
    ]
    if summary["overall_promotion"]["promote_exact_distance_relations"]:
        lines += [
            "当前结果支持保留 occurrence relational composition 方向。下一步应先验证其是否来自特定距离尺度，",
            "再比较参数匹配的 diffusion/random-walk operator，而不是立刻增加 GNN 深度。",
        ]
    else:
        lines += [
            "当前 exact shortest-path summary residual 未通过预注册 gate。它不能证明 relational composition",
            "整体无效，但说明当前全量 upper-triangle + 单标量 residual 的参数化不是可靠增益。下一步应优先做",
            "更强的低秩/分尺度 relation readout 或 diffusion operator，并保留相同 shuffled control。",
        ]
    Path(args.markdown).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "markdown": args.markdown,
        "farthest_relation_mean": f["relation_auc_mean"],
        "scaffold_relation_mean": s["relation_auc_mean"],
        "relation_pair_mean": er["auc_mean"],
        "promotion": summary["overall_promotion"],
    }, indent=2))


if __name__ == "__main__":
    main()

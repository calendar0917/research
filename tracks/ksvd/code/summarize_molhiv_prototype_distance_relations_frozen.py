"""Summarize frozen-base bounded-linear exact-distance relation experiments."""
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
REFERENCE = {
    "farthest": np.asarray([0.7507836990595611, 0.7037985428308693, 0.7766044229175960]),
    "scaffold": np.asarray([0.7505774624649398, 0.6870128032821674, 0.7932509081639156]),
    "pair": np.asarray([0.7668856211846231, 0.6993987408926929, 0.8009240966158945]),
}


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -60.0, 60.0)))


def prediction(doc: dict[str, Any], control: str):
    h = doc["results"][control]["heldout"]
    idx = np.asarray(h["graph_indices"], dtype=np.int64)
    y = np.asarray(h["labels"], dtype=np.int64)
    p = sigmoid(np.asarray(h["scores"], dtype=np.float64))
    order = np.argsort(idx)
    return idx[order], y[order], p[order]


def ensemble(docs: list[dict[str, Any]], controls: tuple[str, str]):
    aucs, correlations = [], []
    for doc in docs:
        ia, y, pa = prediction(doc, controls[0])
        ib, yb, pb = prediction(doc, controls[1])
        if not np.array_equal(ia, ib) or not np.array_equal(y, yb):
            raise ValueError(f"prediction mismatch: {controls}")
        aucs.append(float(roc_auc_score(y, 0.5 * (pa + pb))))
        correlations.append(float(np.corrcoef(pa, pb)[0, 1]))
    return np.asarray(aucs), np.asarray(correlations)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--result-dir", default="results/molhiv")
    ap.add_argument(
        "--output",
        default="results/molhiv/prototype_distance_relations_frozen_summary_scaffold3.json",
    )
    ap.add_argument(
        "--markdown",
        default="results/molhiv/REALPROTOTYPE_FROZEN_LINEAR_DISTANCE_RELATIONS_20260728.md",
    )
    args = ap.parse_args()
    root = Path(args.result_dir)
    docs = [
        json.loads((root / f"prototype_distance_relations_frozen_fold{fold}_seed0.json").read_text())
        for fold in range(3)
    ]
    if [int(d["fold"]) for d in docs] != [0, 1, 2]:
        raise ValueError("fold ids mismatch")
    if any(d["selection_policy"]["official_valid_evaluations"] != 0 for d in docs):
        raise ValueError("official valid was evaluated")
    if any(d["selection_policy"]["official_test_evaluations"] != 0 for d in docs):
        raise ValueError("official test was evaluated")

    summary: dict[str, Any] = {
        "protocol_id": "molhiv-prototype-frozen-bounded-linear-distance-relations-summary-v1",
        "date": "2026-07-28",
        "data_policy": {
            "dataset": "8k stratified subset; 6400 official-train development graphs",
            "outer_validation": "three official-train-only Bemis-Murcko scaffold folds",
            "official_valid_evaluations": 0,
            "official_test_evaluations": 0,
            "base_epochs": int(docs[0]["config"]["epochs"]),
            "relation_epochs": int(docs[0]["config"]["relation_epochs"]),
        },
        "relation_config": {
            "parameters": int(docs[0]["relation_dim"]),
            "absolute_residual_bound": float(docs[0]["config"]["max_relation_residual"]),
            "lr": float(docs[0]["config"]["relation_lr"]),
            "weight_decay": float(docs[0]["config"]["relation_weight_decay"]),
        },
        "families": {},
        "ensembles": {},
    }

    family_pass = []
    for family, c in FAMILIES.items():
        occ = np.asarray([d["results"][c["occurrence"]]["heldout"]["auc"] for d in docs])
        real = np.asarray([d["results"][c["relation"]]["heldout"]["auc"] for d in docs])
        shuffled = np.asarray([d["results"][c["shuffled"]]["heldout"]["auc"] for d in docs])
        residual = np.asarray([
            d["results"][c["relation"]]["heldout"]["mean_abs_relation_residual"] for d in docs
        ])
        max_residual = np.asarray([
            d["results"][c["relation"]]["heldout"]["max_abs_relation_residual"] for d in docs
        ])
        d_occ = real - occ
        d_shuf = real - shuffled
        gate = {
            "relation_minus_occurrence_by_fold": d_occ.tolist(),
            "relation_minus_occurrence_mean": float(d_occ.mean()),
            "occurrence_fold_wins": int(np.sum(real > occ)),
            "relation_minus_shuffled_by_fold": d_shuf.tolist(),
            "relation_minus_shuffled_mean": float(d_shuf.mean()),
            "gain_at_least_005": bool(d_occ.mean() >= 0.005),
            "wins_at_least_2_of_3": bool(np.sum(real > occ) >= 2),
            "beats_shuffled_mean": bool(real.mean() > shuffled.mean()),
        }
        gate["promotion_passed"] = bool(
            gate["gain_at_least_005"]
            and gate["wins_at_least_2_of_3"]
            and gate["beats_shuffled_mean"]
        )
        family_pass.append(gate["promotion_passed"])
        summary["families"][family] = {
            "occurrence_auc_by_fold": occ.tolist(),
            "occurrence_auc_mean": float(occ.mean()),
            "relation_auc_by_fold": real.tolist(),
            "relation_auc_mean": float(real.mean()),
            "shuffled_auc_by_fold": shuffled.tolist(),
            "shuffled_auc_mean": float(shuffled.mean()),
            "mean_abs_residual_by_fold": residual.tolist(),
            "max_abs_residual_by_fold": max_residual.tolist(),
            "baseline_reference_max_abs_delta": float(np.max(np.abs(occ - REFERENCE[family]))),
            "promotion_gate": gate,
        }

    ensemble_controls = {
        "occurrence_pair": ("farthest_occurrence", "scaffold_occurrence"),
        "relation_pair": ("farthest_relation", "scaffold_relation"),
        "shuffled_pair": ("farthest_relation_shuffled", "scaffold_relation_shuffled"),
    }
    for name, controls in ensemble_controls.items():
        auc, corr = ensemble(docs, controls)
        summary["ensembles"][name] = {
            "controls": list(controls),
            "auc_by_fold": auc.tolist(),
            "auc_mean": float(auc.mean()),
            "probability_pearson_by_fold": corr.tolist(),
            "probability_pearson_mean": float(corr.mean()),
        }
    occ_pair = np.asarray(summary["ensembles"]["occurrence_pair"]["auc_by_fold"])
    real_pair = np.asarray(summary["ensembles"]["relation_pair"]["auc_by_fold"])
    shuf_pair = np.asarray(summary["ensembles"]["shuffled_pair"]["auc_by_fold"])
    delta = real_pair - occ_pair
    pair_gate = {
        "relation_minus_occurrence_by_fold": delta.tolist(),
        "relation_minus_occurrence_mean": float(delta.mean()),
        "occurrence_fold_wins": int(np.sum(real_pair > occ_pair)),
        "relation_minus_shuffled_mean": float((real_pair - shuf_pair).mean()),
        "gain_at_least_005": bool(delta.mean() >= 0.005),
        "wins_at_least_2_of_3": bool(np.sum(real_pair > occ_pair) >= 2),
        "beats_shuffled_mean": bool(real_pair.mean() > shuf_pair.mean()),
    }
    pair_gate["promotion_passed"] = bool(
        pair_gate["gain_at_least_005"]
        and pair_gate["wins_at_least_2_of_3"]
        and pair_gate["beats_shuffled_mean"]
    )
    summary["ensembles"]["relation_pair"]["promotion_gate"] = pair_gate
    summary["ensembles"]["occurrence_pair"]["reference_max_abs_delta"] = float(
        np.max(np.abs(occ_pair - REFERENCE["pair"]))
    )
    summary["overall_promotion"] = {
        "any_single_family_passed": bool(any(family_pass)),
        "pair_ensemble_passed": bool(pair_gate["promotion_passed"]),
        "promote_frozen_linear_relation": bool(any(family_pass) or pair_gate["promotion_passed"]),
    }
    Path(args.output).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    f, s = summary["families"]["farthest"], summary["families"]["scaffold"]
    eo, er, es = (summary["ensembles"][k] for k in ("occurrence_pair", "relation_pair", "shuffled_pair"))
    lines = [
        "# Frozen-base bounded-linear distance relations（2026-07-28）",
        "",
        "## 1. Protocol",
        "",
        "- 8,000-graph subset；只使用 6,400 个 official-train graphs 和三个 scaffold folds。",
        "- official valid/test 编码与评估均为 **0**。",
        "- 先训练并冻结 46,658-parameter occurrence PrototypeMIL。",
        f"- Relation residual 只有 `{summary['relation_config']['parameters']}` 个参数，且 `|residual| <= {summary['relation_config']['absolute_residual_bound']}`。",
        "- feature standardization 只使用 outer-fit graphs；real 与 distance-shuffled 使用相同训练协议。",
        "",
        "## 2. Results",
        "",
        "| Vocabulary | Model | Fold 0 | Fold 1 | Fold 2 | Mean |",
        "|---|---|---:|---:|---:|---:|",
    ]
    for family, row in (("farthest", f), ("scaffold", s)):
        for label, key in (("occurrence", "occurrence_auc_by_fold"), ("frozen relation", "relation_auc_by_fold"), ("shuffled", "shuffled_auc_by_fold")):
            v = row[key]
            lines.append(f"| {family} | {label} | {v[0]:.6f} | {v[1]:.6f} | {v[2]:.6f} | {np.mean(v):.6f} |")
    lines += [
        "",
        f"- Farthest relation − occurrence：`{f['promotion_gate']['relation_minus_occurrence_mean']:+.6f}`，赢 `{f['promotion_gate']['occurrence_fold_wins']}/3`；real − shuffled `{f['promotion_gate']['relation_minus_shuffled_mean']:+.6f}`。",
        f"- Scaffold relation − occurrence：`{s['promotion_gate']['relation_minus_occurrence_mean']:+.6f}`，赢 `{s['promotion_gate']['occurrence_fold_wins']}/3`；real − shuffled `{s['promotion_gate']['relation_minus_shuffled_mean']:+.6f}`。",
        "",
        "## 3. Pair ensemble",
        "",
        "| Ensemble | Fold 0 | Fold 1 | Fold 2 | Mean |",
        "|---|---:|---:|---:|---:|",
        f"| occurrence | {eo['auc_by_fold'][0]:.6f} | {eo['auc_by_fold'][1]:.6f} | {eo['auc_by_fold'][2]:.6f} | {eo['auc_mean']:.6f} |",
        f"| frozen relation | {er['auc_by_fold'][0]:.6f} | {er['auc_by_fold'][1]:.6f} | {er['auc_by_fold'][2]:.6f} | {er['auc_mean']:.6f} |",
        f"| shuffled | {es['auc_by_fold'][0]:.6f} | {es['auc_by_fold'][1]:.6f} | {es['auc_by_fold'][2]:.6f} | {es['auc_mean']:.6f} |",
        "",
        f"Pair relation − occurrence：`{pair_gate['relation_minus_occurrence_mean']:+.6f}`，赢 `{pair_gate['occurrence_fold_wins']}/3`；real − shuffled `{pair_gate['relation_minus_shuffled_mean']:+.6f}`。",
        "",
        "## 4. Decision",
        "",
        f"- Baseline reproduction max error：farthest `{f['baseline_reference_max_abs_delta']:.3e}`，scaffold `{s['baseline_reference_max_abs_delta']:.3e}`，pair `{eo['reference_max_abs_delta']:.3e}`。",
        f"- Promotion：**{summary['overall_promotion']['promote_frozen_linear_relation']}**。",
        "",
    ]
    if summary["overall_promotion"]["promote_frozen_linear_relation"]:
        lines.append("结果支持在冻结 occurrence 基础上继续研究分尺度 relation 与 diffusion operator。")
    else:
        lines.append("即使冻结 base 并严格限制 residual，当前 full-matrix linear relation 仍未通过 gate；下一步应转向距离尺度消融或固定低秩统计，而不是继续扩大 relation head。")
    Path(args.markdown).write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({
        "output": args.output,
        "markdown": args.markdown,
        "farthest_relation_mean": f["relation_auc_mean"],
        "scaffold_relation_mean": s["relation_auc_mean"],
        "pair_relation_mean": er["auc_mean"],
        "promotion": summary["overall_promotion"],
    }, indent=2))


if __name__ == "__main__":
    main()

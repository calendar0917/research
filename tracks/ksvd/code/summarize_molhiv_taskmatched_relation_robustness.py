"""Summarize task-matched relation-sidecar robustness across prototype-bank seeds."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
from scipy.optimize import linear_sum_assignment
from sklearn.metrics import roc_auc_score


def _sigmoid(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return np.where(x >= 0, 1.0 / (1.0 + np.exp(-x)), np.exp(x) / (1.0 + np.exp(x)))


def _mean(x: list[float]) -> float:
    return float(np.mean(np.asarray(x, dtype=np.float64)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/molhiv")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()
    root = Path(args.results_dir)
    seeds = {
        20260728: "taskmatched_relation_sidecar_cap03125_fold{fold}_seed0.json",
        20260729: "taskmatched_relation_sidecar_protoseed20260729_fold{fold}_seed0.json",
    }
    selectors = {
        20260728: "stable_taskmatched_predictions_fold{fold}_seed0.json",
        20260729: "stable_taskmatched_protoseed20260729_fold{fold}_seed0.json",
    }

    rows: dict[str, Any] = {}
    saved: dict[tuple[int, int, str, str], dict[str, Any]] = {}
    for prototype_seed, pattern in seeds.items():
        for base_variant in ("broad_occurrence_pair", "broad_relation_pair"):
            for family in ("nested_taskaware", "nested_shuffled"):
                base, real, shuffled = [], [], []
                for fold in range(3):
                    doc = json.loads((root / pattern.format(fold=fold)).read_text())
                    key = f"{base_variant}__{family}__distance_1_2"
                    rr = doc["results"][f"{key}__real"]["heldout"]
                    ss = doc["results"][f"{key}__shuffled"]["heldout"]
                    saved[(prototype_seed, fold, base_variant, family)] = rr
                    base.append(float(rr["base_auc"]))
                    real.append(float(rr["auc"]))
                    shuffled.append(float(ss["auc"]))
                rows[f"seed={prototype_seed}__{base_variant}__{family}"] = {
                    "base_by_fold": base,
                    "real_by_fold": real,
                    "shuffled_by_fold": shuffled,
                    "base_mean": _mean(base),
                    "real_mean": _mean(real),
                    "shuffled_mean": _mean(shuffled),
                    "real_minus_base": _mean(real) - _mean(base),
                    "real_minus_shuffled": _mean(real) - _mean(shuffled),
                    "base_wins": int(sum(x > y for x, y in zip(real, base))),
                    "shuffled_wins": int(sum(x > y for x, y in zip(real, shuffled))),
                }

    bank_stability: dict[str, Any] = {}
    for family in ("nested_taskaware", "nested_shuffled"):
        overlap, matched = [], []
        for fold in range(3):
            docs = [
                json.loads((root / selectors[s].format(fold=fold)).read_text())
                for s in seeds
            ]
            selected_rows = [
                np.asarray(d["results"][family]["selection"]["source_node_rows"], dtype=np.int64)
                for d in docs
            ]
            with np.load(root / f"rawpatch_pca64_latents_n8000_scaffoldfit_fold{fold}.npz", allow_pickle=False) as source:
                latents = np.asarray(source["latents"], dtype=np.float32)
            prototypes = []
            for rr in selected_rows:
                p = latents[rr]
                p = p / np.maximum(np.linalg.norm(p, axis=1, keepdims=True), 1e-12)
                prototypes.append(p)
            similarity = prototypes[0] @ prototypes[1].T
            a, b = linear_sum_assignment(-similarity)
            overlap.append(int(len(set(selected_rows[0]).intersection(selected_rows[1]))))
            matched.append(float(similarity[a, b].mean()))
        bank_stability[family] = {
            "exact_row_overlap_by_fold": overlap,
            "optimal_matching_cosine_by_fold": matched,
            "optimal_matching_cosine_mean": _mean(matched),
        }

    ensembles: dict[str, Any] = {}
    for base_variant in ("broad_occurrence_pair", "broad_relation_pair"):
        for family in ("nested_taskaware", "nested_shuffled"):
            aucs, bases = [], []
            for fold in range(3):
                rr = [saved[(s, fold, base_variant, family)] for s in seeds]
                labels = np.asarray(rr[0]["labels"], dtype=np.float32)
                base_scores = np.asarray(rr[0]["base_scores"], dtype=np.float32)
                scores = np.mean([np.asarray(x["scores"], dtype=np.float32) for x in rr], axis=0)
                aucs.append(float(roc_auc_score(labels, scores)))
                bases.append(float(roc_auc_score(labels, base_scores)))
            ensembles[f"{base_variant}__{family}"] = {
                "auc_by_fold": aucs,
                "auc_mean": _mean(aucs),
                "base_mean": _mean(bases),
                "gain": _mean(aucs) - _mean(bases),
            }

    target_seed0 = rows["seed=20260728__broad_occurrence_pair__nested_taskaware"]
    target_seed1 = rows["seed=20260729__broad_occurrence_pair__nested_taskaware"]
    robustness_gate = {
        "required_gain_each_prototype_seed": 0.005,
        "seed_20260728_pass": bool(target_seed0["real_minus_base"] >= 0.005),
        "seed_20260729_pass": bool(target_seed1["real_minus_base"] >= 0.005),
        "all_prototype_seeds_pass": bool(
            target_seed0["real_minus_base"] >= 0.005 and target_seed1["real_minus_base"] >= 0.005
        ),
    }
    summary = {
        "protocol_id": "molhiv_taskmatched_relation_prototype_seed_robustness_v1",
        "date": "2026-07-28",
        "prototype_seeds": list(seeds),
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "rows": rows,
        "bank_stability": bank_stability,
        "two_seed_fixed_logit_ensembles": ensembles,
        "robustness_gate": robustness_gate,
        "decision": "do_not_promote_to_official_splits",
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    def fmt(row: dict[str, Any]) -> str:
        return (
            f"{row['base_mean']:.6f} | {row['real_mean']:.6f} | {row['shuffled_mean']:.6f} | "
            f"{row['real_minus_base']:+.6f} | {row['real_minus_shuffled']:+.6f} | "
            f"{row['base_wins']}/3 | {row['shuffled_wins']}/3"
        )

    md = """# Task-matched relation sidecar: prototype-seed robustness（2026-07-28）

## 1. Why this check was required

cap `0.3125` 在 prototype seed `20260728` 上通过开发 gate，但该 cap 是 near-gate 后的一步优化。为避免把单个随机 candidate reservoir 当作稳定结论，本轮固定所有 downstream 配置，只把 task-matched candidate-bank/prototype seed 改为 `20260729`。official valid/test 编码与评估仍为 **0**。

## 2. Robustness results

| Prototype seed | Frozen base | Relation selector | Base | Real | Assignment shuffled | Real−base | Real−shuffled | Wins base | Wins shuffled |
|---:|---|---|---:|---:|---:|---:|---:|---:|---:|
"""
    for prototype_seed in seeds:
        for base_variant, base_label in (
            ("broad_occurrence_pair", "Broad occurrence pair"),
            ("broad_relation_pair", "Broad relation pair"),
        ):
            for family, family_label in (
                ("nested_taskaware", "Nested task-aware"),
                ("nested_shuffled", "Nested shuffled-label"),
            ):
                row = rows[f"seed={prototype_seed}__{base_variant}__{family}"]
                md += f"| {prototype_seed} | {base_label} | {family_label} | {fmt(row)} |\n"

    ta = bank_stability["nested_taskaware"]
    ns = bank_stability["nested_shuffled"]
    ens_occ = ensembles["broad_occurrence_pair__nested_taskaware"]
    ens_rel = ensembles["broad_relation_pair__nested_taskaware"]
    md += f"""
## 3. Stability diagnosis

- task-aware selected rows 两个 prototype seeds 在每个 fold 的 exact overlap 都是 `{ta['exact_row_overlap_by_fold']}`；optimal-matching cosine 为 `{ta['optimal_matching_cosine_by_fold']}`，mean `{ta['optimal_matching_cosine_mean']:.6f}`。
- shuffled-label bank 的 matching cosine mean 为 `{ns['optimal_matching_cosine_mean']:.6f}`。因此变化不是简单的 atom permutation，而是 candidate reservoir 改变后选出了不同的 vocabulary geometry。
- prototype seed `20260728` 上，broad occurrence + task-aware sidecar gain 为 `{target_seed0['real_minus_base']:+.6f}`；seed `20260729` 仅为 `{target_seed1['real_minus_base']:+.6f}`，assignment-specific margin 也降至 `{target_seed1['real_minus_shuffled']:+.6f}`。
- 更关键的是，seed `20260729` 上 shuffled-label selector 反而比 task-aware selector 更强。这说明当前 256-candidate nested selector 的 label-matched advantage 不稳定。
- 固定平均两个 task-aware sidecar logits 后：broad occurrence base mean `{ens_occ['base_mean']:.6f}` → `{ens_occ['auc_mean']:.6f}`（gain `{ens_occ['gain']:+.6f}`）；broad relation base mean `{ens_rel['base_mean']:.6f}` → `{ens_rel['auc_mean']:.6f}`（gain `{ens_rel['gain']:+.6f}`）。ensemble 降低了 seed 方差，但仍未达到 +0.005 gate。

## 4. Decision

- prototype-seed robustness gate：**{robustness_gate['all_prototype_seeds_pass']}**。
- **不晋级 official valid/test。** 单 seed 的 `0.762375` 仍是有价值的 mechanism signal，但不能当作冻结配置的可靠预期。
- 需要保留的结论不是“当前 supervised selector 已成功”，而是：**task-aware relation bank 可能有用，但当前 256-candidate hard selection 太依赖 reservoir identity。**

## 5. Next optimization direction

下一轮应停止直接从随机 256-candidate pool hard-select 32 个 atoms，改成更稳定的 task matching：

1. 使用 broad deterministic outer-fit bank（每个 fit graph 一个 observed patch）作为冻结候选宇宙；
2. 不替换 broad occurrence vocabulary，只学习 nested outer-fit-only prototype scalar weights / relation gates；
3. 或对多个 candidate reservoirs 做 selector-score consensus，再从 union 中选 prototype；
4. 保持 broad occurrence base、exact distance 1+2、assignment-shuffled 和 shuffled-label controls 不变。

这比继续调 residual cap、增加 random-walk 长度或扩大 relation head 更符合当前证据。
"""
    Path(args.output_md).write_text(md, encoding="utf-8")
    print(json.dumps({
        "output_json": args.output_json,
        "output_md": args.output_md,
        "robustness_gate": robustness_gate,
        "taskaware_bank_stability": ta,
    }, indent=2))


if __name__ == "__main__":
    main()

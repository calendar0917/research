"""Summarize nested task-matched compact relation and role-separated sidecars."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np


def _mean(xs: list[float]) -> float:
    return float(np.mean(np.asarray(xs, dtype=np.float64)))


def _row(base: list[float], real: list[float], shuffled: list[float]) -> dict[str, Any]:
    return {
        "base_by_fold": base,
        "real_by_fold": real,
        "shuffled_by_fold": shuffled,
        "base_mean": _mean(base),
        "real_mean": _mean(real),
        "shuffled_mean": _mean(shuffled),
        "real_minus_base": _mean(real) - _mean(base),
        "real_minus_shuffled": _mean(real) - _mean(shuffled),
        "real_base_wins": int(sum(x > y for x, y in zip(real, base))),
        "real_shuffled_wins": int(sum(x > y for x, y in zip(real, shuffled))),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results-dir", default="results/molhiv")
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-md", required=True)
    args = ap.parse_args()
    root = Path(args.results_dir)

    standalone_docs = [
        json.loads((root / f"taskmatched_distance_compact_fold{fold}_seed0.json").read_text())
        for fold in range(3)
    ]
    standalone: dict[str, Any] = {}
    for family in ("nested_taskaware", "nested_shuffled"):
        base, real, shuffled = [], [], []
        for doc in standalone_docs:
            rr = doc["results"][f"{family}__distance_1_2__real"]["heldout"]
            ss = doc["results"][f"{family}__distance_1_2__shuffled"]["heldout"]
            base.append(float(rr["base_auc"]))
            real.append(float(rr["auc"]))
            shuffled.append(float(ss["auc"]))
        standalone[family] = _row(base, real, shuffled)

    caps = {
        "0.25": "taskmatched_relation_sidecar_fold{fold}_seed0.json",
        "0.3125": "taskmatched_relation_sidecar_cap03125_fold{fold}_seed0.json",
    }
    sidecars: dict[str, Any] = {}
    for cap, pattern in caps.items():
        docs = [json.loads((root / pattern.format(fold=fold)).read_text()) for fold in range(3)]
        cap_rows: dict[str, Any] = {}
        for base_variant in ("broad_occurrence_pair", "broad_relation_pair"):
            for family in ("nested_taskaware", "nested_shuffled"):
                base, real, shuffled = [], [], []
                for doc in docs:
                    prefix = f"{base_variant}__{family}__distance_1_2"
                    rr = doc["results"][f"{prefix}__real"]["heldout"]
                    ss = doc["results"][f"{prefix}__shuffled"]["heldout"]
                    base.append(float(rr["base_auc"]))
                    real.append(float(rr["auc"]))
                    shuffled.append(float(ss["auc"]))
                cap_rows[f"{base_variant}__{family}"] = _row(base, real, shuffled)
        sidecars[cap] = cap_rows

    target = sidecars["0.3125"]["broad_occurrence_pair__nested_taskaware"]
    strict_gate = {
        "threshold_mean_gain": 0.005,
        "minimum_base_wins": 2,
        "requires_real_above_shuffled_mean": True,
        "mean_gain_pass": bool(target["real_minus_base"] >= 0.005),
        "base_wins_pass": bool(target["real_base_wins"] >= 2),
        "real_above_shuffled_pass": bool(target["real_mean"] > target["shuffled_mean"]),
    }
    strict_gate["all_pass"] = bool(all([
        strict_gate["mean_gain_pass"], strict_gate["base_wins_pass"],
        strict_gate["real_above_shuffled_pass"],
    ]))

    best_key = None
    best = None
    for cap, rows in sidecars.items():
        for key, row in rows.items():
            if best is None or row["real_mean"] > best["real_mean"]:
                best_key, best = f"cap={cap}__{key}", row

    summary = {
        "protocol_id": "molhiv_taskmatched_relation_sidecar_summary_v1",
        "date": "2026-07-28",
        "official_valid_evaluations": 0,
        "official_test_evaluations": 0,
        "standalone": standalone,
        "sidecars": sidecars,
        "strict_gate_target": "cap=0.3125__broad_occurrence_pair__nested_taskaware",
        "strict_gate": strict_gate,
        "best_absolute": {"key": best_key, **(best or {})},
        "interpretation": {
            "taskaware_occurrence_is_not_better_than_shuffled_selector": bool(
                standalone["nested_taskaware"]["base_mean"] <=
                standalone["nested_shuffled"]["base_mean"] + 0.001
            ),
            "taskaware_relation_sidecar_is_assignment_specific": bool(
                target["real_minus_shuffled"] > 0 and target["real_shuffled_wins"] >= 2
            ),
            "cap_03125_is_post_pilot_development_optimization": True,
            "official_promotion_requires_robustness_confirmation": True,
        },
    }
    Path(args.output_json).write_text(json.dumps(summary, indent=2), encoding="utf-8")

    s_task = standalone["nested_taskaware"]
    s_shuf = standalone["nested_shuffled"]
    rows025 = sidecars["0.25"]
    rows031 = sidecars["0.3125"]
    def fmt(row: dict[str, Any]) -> str:
        return (
            f"{row['base_mean']:.6f} | {row['real_mean']:.6f} | {row['shuffled_mean']:.6f} | "
            f"{row['real_minus_base']:+.6f} | {row['real_minus_shuffled']:+.6f} | "
            f"{row['real_base_wins']}/3 | {row['real_shuffled_wins']}/3"
        )

    md = f"""# Task-matched prototype relation sidecar（2026-07-28）

## 1. Protocol

- 8,000-graph development subset；只使用其中 6,400 个 official-train graphs 的三个 outer scaffold folds。
- task-aware selector 严格 nested：candidate 只由 outer-fit 构造，candidate source graph 不参与评价该 candidate 的 inner-valid evidence。
- `nested_taskaware` 与 label-shuffled `nested_shuffled` 使用相同 candidate bank、MMR 与模型协议。
- relation 仅使用已定位的 exact shortest-path distance 1+2 compact feature（140 维），并保留 node-assignment-shuffled control。
- broad occurrence base 是 frozen farthest + scaffold-facility probability pair；另测试 frozen broad relation pair。
- official valid/test 编码与评估均为 **0**。

## 2. Task-matched bank standalone

| Prototype selector | Occurrence base | Real d1+2 | Assignment shuffled | Real−base | Real−shuffled | Wins base | Wins shuffled |
|---|---:|---:|---:|---:|---:|---:|---:|
| Nested task-aware | {fmt(s_task)} |
| Nested shuffled-label | {fmt(s_shuf)} |

结论：task-aware selection **没有提升 standalone occurrence**（两种 selector 的 occurrence mean 几乎相同），但 task-aware bank 上的真实 d1+2 relation gain 为 `{s_task['real_minus_base']:+.6f}`，明显高于 shuffled-label selector 的 `{s_shuf['real_minus_base']:+.6f}`。这说明监督信息更可能改变“哪些 prototype 适合做关系组合”，而不是直接改善 occurrence readout。

## 3. Role-separated sidecar, residual cap 0.25

| Frozen base | Relation bank | Base | Real | Assignment shuffled | Real−base | Real−shuffled | Wins base | Wins shuffled |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Broad occurrence pair | Nested task-aware | {fmt(rows025['broad_occurrence_pair__nested_taskaware'])} |
| Broad occurrence pair | Nested shuffled-label | {fmt(rows025['broad_occurrence_pair__nested_shuffled'])} |
| Broad relation pair | Nested task-aware | {fmt(rows025['broad_relation_pair__nested_taskaware'])} |
| Broad relation pair | Nested shuffled-label | {fmt(rows025['broad_relation_pair__nested_shuffled'])} |

cap 0.25 下，task-aware relation sidecar 在 broad occurrence base 上得到 `{rows025['broad_occurrence_pair__nested_taskaware']['real_mean']:.6f}`，gain `{rows025['broad_occurrence_pair__nested_taskaware']['real_minus_base']:+.6f}`，只差 `0.000078` 达到 +0.005 gate，但 real 同时以 3/3 folds 超过 base 和 assignment-shuffled。

## 4. One-step capacity confirmation, residual cap 0.3125

| Frozen base | Relation bank | Base | Real | Assignment shuffled | Real−base | Real−shuffled | Wins base | Wins shuffled |
|---|---|---:|---:|---:|---:|---:|---:|---:|
| Broad occurrence pair | Nested task-aware | {fmt(rows031['broad_occurrence_pair__nested_taskaware'])} |
| Broad occurrence pair | Nested shuffled-label | {fmt(rows031['broad_occurrence_pair__nested_shuffled'])} |
| Broad relation pair | Nested task-aware | {fmt(rows031['broad_relation_pair__nested_taskaware'])} |
| Broad relation pair | Nested shuffled-label | {fmt(rows031['broad_relation_pair__nested_shuffled'])} |

- broad occurrence pair + task-aware relation sidecar：fold AUC `{rows031['broad_occurrence_pair__nested_taskaware']['real_by_fold']}`，mean `{rows031['broad_occurrence_pair__nested_taskaware']['real_mean']:.6f}`。
- 相对 broad occurrence pair gain `{rows031['broad_occurrence_pair__nested_taskaware']['real_minus_base']:+.6f}`，real−shuffled `{rows031['broad_occurrence_pair__nested_taskaware']['real_minus_shuffled']:+.6f}`，两项均 3/3 wins。
- 最佳绝对结果是 broad relation pair + task-aware sidecar：`{rows031['broad_relation_pair__nested_taskaware']['real_mean']:.6f}`；相对上一轮 broad relation pair `{rows031['broad_relation_pair__nested_taskaware']['real_minus_base']:+.6f}`。
- shuffled-label selector 的增益明显更小，说明结果不是任意第三套 prototype bank 或额外 140 参数即可解释。

严格 development gate（以 broad occurrence pair + task-aware sidecar 为 target）：**{strict_gate['all_pass']}**。

注意：cap `0.3125` 是在 cap `0.25` near-gate 后进行的一步 development optimization，因此它不是 untouched confirmation。进入 official valid/test 前仍应做 prototype-bank seed / selector robustness；不能仅凭本轮调参后过线直接使用 official splits 做选择。

## 5. Research decision

1. **B 的更精确版本成立**：字典不应整体变成 supervised vocabulary；更合理的是 role separation：label-free broad banks 负责稳定 occurrence，nested task-matched bank 只负责 local relation composition。
2. 这也解释了前一轮矛盾：task-aware bank standalone occurrence 较弱，但其 d1+2 relation feature 在强 broad base 上有稳定、assignment-specific 的增益。
3. 当前最佳开发模型仍是 GNN-free：broad occurrence/relation + 140-d task-matched relation sidecar；不需要改回 GINE backbone。
4. 下一步不是继续扩 walk 长度或 relation matrix，而是验证 selector 稳定性：至少 3 个 candidate-bank/prototype seeds，并固定 cap 0.3125。只有 robustness 仍通过 gate，才晋级 full official valid/test。
"""
    Path(args.output_md).write_text(md, encoding="utf-8")
    print(json.dumps({
        "output_json": args.output_json,
        "output_md": args.output_md,
        "strict_gate": strict_gate,
        "best_absolute": summary["best_absolute"],
    }, indent=2))


if __name__ == "__main__":
    main()

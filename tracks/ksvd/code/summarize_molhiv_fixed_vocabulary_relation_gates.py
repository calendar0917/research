"""Summarize three outer folds of fixed-vocabulary relation gating."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np

KEYS = (
    "uniform__real",
    "uniform__assignment_shuffled",
    "task__real",
    "task__assignment_shuffled",
    "shuffled_label__real",
    "shuffled_label__assignment_shuffled",
)


def sha256(array: np.ndarray) -> str:
    return hashlib.sha256(np.asarray(array).tobytes()).hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("inputs", nargs=3)
    ap.add_argument("--output-json", required=True)
    ap.add_argument("--output-report", required=True)
    args = ap.parse_args()

    docs = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.inputs]
    docs.sort(key=lambda doc: int(doc["fold"]))
    if [int(doc["fold"]) for doc in docs] != [0, 1, 2]:
        raise ValueError("expected exactly folds 0,1,2")
    for doc in docs:
        if doc.get("protocol_id") != "molhiv_fixed_broad_vocabulary_nested_relation_gates_v1":
            raise ValueError("unexpected protocol")
        policy = doc.get("selection_policy", {})
        if int(policy.get("official_valid_evaluations", -1)) != 0:
            raise ValueError("official valid was evaluated")
        if int(policy.get("official_test_evaluations", -1)) != 0:
            raise ValueError("official test was evaluated")
        if set(KEYS).difference(doc["results"]):
            raise ValueError("missing matched control")

    heldout_indices = [
        np.asarray(doc["results"]["uniform__real"]["heldout"]["graph_indices"], dtype=np.int64)
        for doc in docs
    ]
    for fold, (doc, indices) in enumerate(zip(docs, heldout_indices)):
        if sha256(indices) != doc["heldout_indices_sha256"]:
            raise ValueError(f"fold {fold} heldout prediction indices mismatch")
    combined = np.concatenate(heldout_indices)
    if len(np.unique(combined)) != len(combined):
        raise ValueError("outer heldout predictions overlap")

    base_fold = np.asarray([doc["base_occurrence"]["heldout_auc"] for doc in docs], dtype=np.float64)
    auc_fold = {
        key: np.asarray([doc["results"][key]["heldout"]["auc"] for doc in docs], dtype=np.float64)
        for key in KEYS
    }
    rows: dict[str, Any] = {}
    for key, values in auc_fold.items():
        gains = values - base_fold
        rows[key] = {
            "fold_auc": values.tolist(),
            "mean_auc": float(values.mean()),
            "fold_gain_over_base": gains.tolist(),
            "mean_gain_over_base": float(gains.mean()),
            "base_wins": int(np.sum(gains > 0)),
        }

    task_minus_uniform = auc_fold["task__real"] - auc_fold["uniform__real"]
    task_minus_shuffled_label = auc_fold["task__real"] - auc_fold["shuffled_label__real"]
    task_minus_assignment_shuffled = (
        auc_fold["task__real"] - auc_fold["task__assignment_shuffled"]
    )
    gate = {
        "thresholds": {
            "mean_task_minus_uniform_at_least": 0.005,
            "minimum_task_uniform_fold_wins": 2,
            "task_mean_must_beat_shuffled_label_mean": True,
            "task_real_must_beat_task_assignment_shuffled_in_all_folds": True,
        },
        "observed": {
            "fold_task_minus_uniform": task_minus_uniform.tolist(),
            "mean_task_minus_uniform": float(task_minus_uniform.mean()),
            "task_uniform_fold_wins": int(np.sum(task_minus_uniform > 0)),
            "fold_task_minus_shuffled_label": task_minus_shuffled_label.tolist(),
            "mean_task_minus_shuffled_label": float(task_minus_shuffled_label.mean()),
            "task_shuffled_label_fold_wins": int(np.sum(task_minus_shuffled_label > 0)),
            "fold_task_minus_task_assignment_shuffled": task_minus_assignment_shuffled.tolist(),
            "mean_task_minus_task_assignment_shuffled": float(task_minus_assignment_shuffled.mean()),
            "task_assignment_shuffled_fold_wins": int(np.sum(task_minus_assignment_shuffled > 0)),
        },
    }
    conditions = {
        "mean_gain_over_uniform": bool(task_minus_uniform.mean() >= 0.005),
        "uniform_fold_wins": bool(np.sum(task_minus_uniform > 0) >= 2),
        "beats_shuffled_label_mean": bool(task_minus_shuffled_label.mean() > 0),
        "beats_assignment_shuffled_all_folds": bool(np.all(task_minus_assignment_shuffled > 0)),
    }
    gate["conditions"] = conditions
    gate["passed"] = bool(all(conditions.values()))

    gate_diagnostics: dict[str, Any] = {}
    for family in ("farthest", "scaffold_facility"):
        gate_diagnostics[family] = {}
        for gate_type in ("task", "shuffled_label"):
            std = np.asarray(
                [doc["selector_audit"][family]["gates"][gate_type]["std"] for doc in docs]
            )
            assignment_l1 = np.asarray(
                [
                    doc["assignment_audit"][family]["mean_gate_assignment_l1"][gate_type]
                    for doc in docs
                ]
            )
            gate_diagnostics[family][gate_type] = {
                "fold_gate_std": std.tolist(),
                "mean_gate_std": float(std.mean()),
                "fold_assignment_l1_from_uniform": assignment_l1.tolist(),
                "mean_assignment_l1_from_uniform": float(assignment_l1.mean()),
            }

    summary = {
        "protocol_id": "molhiv_fixed_broad_vocabulary_nested_relation_gates_summary_v1",
        "date": "2026-07-28",
        "inputs": args.inputs,
        "selection_policy": {
            "official_valid_evaluations": 0,
            "official_test_evaluations": 0,
            "decision": "do not promote to official valid/test" if not gate["passed"] else "eligible for next robustness gate",
        },
        "n_outer_heldout_predictions": int(len(combined)),
        "outer_heldout_indices_sha256_sorted": sha256(np.sort(combined)),
        "base": {
            "fold_auc": base_fold.tolist(),
            "mean_auc": float(base_fold.mean()),
        },
        "results": rows,
        "promotion_gate": gate,
        "gate_diagnostics": gate_diagnostics,
    }
    output_json = Path(args.output_json)
    output_json.parent.mkdir(parents=True, exist_ok=True)
    output_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")

    def fmt(values: list[float]) -> str:
        return " / ".join(f"{value:.6f}" for value in values)

    report = f"""# Fixed-vocabulary task-aware relation gates (8k development)

Date: 2026-07-28

## Question

Can task matching be stabilized by keeping the broad deterministic real-patch vocabulary fixed and using nested outer-fit-only supervision only as a smooth prototype gate for exact-distance 1+2 relation composition?

## Protocol

- Data: 8,000-graph subset; 6,400 official-train graphs only.
- Validation: three outer Bemis-Murcko scaffold folds.
- Vocabulary: fixed 32-prototype `farthest` and 32-prototype `scaffold_facility` banks per outer fold.
- Gate: source-clean nested inner-scaffold OOF balanced-log-loss ablation rank, transformed as `(0.5 + importance) / mean(0.5 + importance)`.
- Base: frozen probability average of the two broad occurrence predictors.
- Relation: top-3 positive-cosine assignment, compact exact distance 1+2, both families concatenated (280 dimensions).
- Residual: zero-initialized bounded linear head, cap fixed at 0.3125.
- Controls: uniform gate, shuffled-label gate, and matched node-assignment shuffling.
- Official valid evaluations: **0**.
- Official test evaluations: **0**.

## Outer-fold AUC

| Gate / relation | Fold 0 | Fold 1 | Fold 2 | Mean | Gain over frozen base |
|---|---:|---:|---:|---:|---:|
| Frozen broad occurrence base | {base_fold[0]:.6f} | {base_fold[1]:.6f} | {base_fold[2]:.6f} | **{base_fold.mean():.6f}** | — |
| Uniform / real | {auc_fold['uniform__real'][0]:.6f} | {auc_fold['uniform__real'][1]:.6f} | {auc_fold['uniform__real'][2]:.6f} | **{auc_fold['uniform__real'].mean():.6f}** | {rows['uniform__real']['mean_gain_over_base']:+.6f} |
| Uniform / assignment-shuffled | {auc_fold['uniform__assignment_shuffled'][0]:.6f} | {auc_fold['uniform__assignment_shuffled'][1]:.6f} | {auc_fold['uniform__assignment_shuffled'][2]:.6f} | {auc_fold['uniform__assignment_shuffled'].mean():.6f} | {rows['uniform__assignment_shuffled']['mean_gain_over_base']:+.6f} |
| Task gate / real | {auc_fold['task__real'][0]:.6f} | {auc_fold['task__real'][1]:.6f} | {auc_fold['task__real'][2]:.6f} | **{auc_fold['task__real'].mean():.6f}** | {rows['task__real']['mean_gain_over_base']:+.6f} |
| Task gate / assignment-shuffled | {auc_fold['task__assignment_shuffled'][0]:.6f} | {auc_fold['task__assignment_shuffled'][1]:.6f} | {auc_fold['task__assignment_shuffled'][2]:.6f} | {auc_fold['task__assignment_shuffled'].mean():.6f} | {rows['task__assignment_shuffled']['mean_gain_over_base']:+.6f} |
| Shuffled-label gate / real | {auc_fold['shuffled_label__real'][0]:.6f} | {auc_fold['shuffled_label__real'][1]:.6f} | {auc_fold['shuffled_label__real'][2]:.6f} | **{auc_fold['shuffled_label__real'].mean():.6f}** | {rows['shuffled_label__real']['mean_gain_over_base']:+.6f} |
| Shuffled-label gate / assignment-shuffled | {auc_fold['shuffled_label__assignment_shuffled'][0]:.6f} | {auc_fold['shuffled_label__assignment_shuffled'][1]:.6f} | {auc_fold['shuffled_label__assignment_shuffled'][2]:.6f} | {auc_fold['shuffled_label__assignment_shuffled'].mean():.6f} | {rows['shuffled_label__assignment_shuffled']['mean_gain_over_base']:+.6f} |

## Matched comparisons

| Comparison | Fold values | Mean | Wins |
|---|---|---:|---:|
| Task real − uniform real | {fmt(task_minus_uniform.tolist())} | **{task_minus_uniform.mean():+.6f}** | {int(np.sum(task_minus_uniform > 0))}/3 |
| Task real − shuffled-label real | {fmt(task_minus_shuffled_label.tolist())} | **{task_minus_shuffled_label.mean():+.6f}** | {int(np.sum(task_minus_shuffled_label > 0))}/3 |
| Task real − task assignment-shuffled | {fmt(task_minus_assignment_shuffled.tolist())} | **{task_minus_assignment_shuffled.mean():+.6f}** | {int(np.sum(task_minus_assignment_shuffled > 0))}/3 |

## Decision

**Promotion gate: {'PASS' if gate['passed'] else 'FAIL'}.**

The smooth task-aware gate is below the matched uniform gate in all three folds and below the shuffled-label gate in all three folds. It does preserve assignment-specific local signal—the real task-gated relation beats its own assignment-shuffled control in all three folds—but that signal is not specifically improved by the true-label gate.

Therefore:

1. Do **not** run official valid/test.
2. Stop treating prototype-wise scalar reweighting as the main stabilization mechanism.
3. Retain the stronger mechanism result: fixed broad prototypes plus exact local relation composition consistently helps the occurrence base; the supervised scalar gate does not explain that gain.
4. The next attempt should change the supervised object rather than gate strength or residual cap—for example, learn a low-rank **pairwise relation metric** under strict nested/OOF constraints, or move supervision to graph-level mixture calibration while keeping the local vocabulary and relation tensors label-free.
"""
    output_report = Path(args.output_report)
    output_report.parent.mkdir(parents=True, exist_ok=True)
    output_report.write_text(report, encoding="utf-8")
    print(json.dumps({"output_json": str(output_json), "output_report": str(output_report), "gate": gate}, indent=2))


if __name__ == "__main__":
    main()

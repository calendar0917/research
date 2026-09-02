"""Unseen-split conditional Beam8 patch-graph controls on BZR."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .run_beam8_nci1_chain_classification import ROOT, _atomic_json, _atomic_text
from .run_beam8_small_tu_patch_graph_prescreen import (
    DEFAULT_ROOT,
    _fit_score,
    _matrices,
    _prepare,
)


PROTOCOL = "tracks/ksvd/docs/KSVD_BZR_BEAM8_PATCH_GRAPH_CONDITIONAL_PROTOCOL_20260814.md"
RESULT_DIR = ROOT / "tracks/ksvd/results/luyin14"
PRESCREEN_JSON = RESULT_DIR / "beam8_small_tu_patch_graph_prescreen_20260814.json"
DEFAULT_JSON = RESULT_DIR / "bzr_beam8_patch_graph_conditional_20260814.json"
DEFAULT_REPORT = RESULT_DIR / "BZR_BEAM8_PATCH_GRAPH_CONDITIONAL_20260814.md"
SPLIT_SEEDS = (3, 4)
VARIANTS = (
    "GLOBAL_ONLY",
    "GLOBAL_PLUS_BAG",
    "GLOBAL_PLUS_PATCH_GRAPH_TRUE",
    "GLOBAL_PLUS_PATCH_GRAPH_TOKEN_SHUFFLED",
)


def _conditional_matrices(raw: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    global_values = raw["GLOBAL_STATS"]
    bag = raw["RAW_BAG"]
    true = raw["RAW_PATCH_GRAPH_TRUE"]
    shuffled = raw["RAW_PATCH_GRAPH_TOKEN_SHUFFLED"]
    if true.shape != shuffled.shape or bag.shape[0] != true.shape[0]:
        raise ValueError("incompatible BZR prescreen matrices")
    if bag.shape[1] > true.shape[1]:
        raise ValueError("BAG exceeds TRUE relation feature dimension")
    bag_padded = np.pad(bag, ((0, 0), (0, true.shape[1] - bag.shape[1])))
    zeros = np.zeros_like(true)
    return {
        "GLOBAL_ONLY": np.concatenate([global_values, zeros], axis=1),
        "GLOBAL_PLUS_BAG": np.concatenate([global_values, bag_padded], axis=1),
        "GLOBAL_PLUS_PATCH_GRAPH_TRUE": np.concatenate([global_values, true], axis=1),
        "GLOBAL_PLUS_PATCH_GRAPH_TOKEN_SHUFFLED": np.concatenate(
            [global_values, shuffled], axis=1
        ),
    }


def _delta(
    units: Sequence[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    values = np.asarray(
        [unit["scores"][left]["balanced_accuracy"] - unit["scores"][right]["balanced_accuracy"] for unit in units]
    )
    return {
        "mean": float(values.mean()),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
        "split_means": {
            str(seed): float(
                np.mean(
                    [value for value, unit in zip(values, units) if unit["split_seed"] == seed]
                )
            )
            for seed in SPLIT_SEEDS
        },
        "values": values.tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    reference = json.loads(args.prescreen_json.read_text(encoding="utf-8"))
    audit = reference["datasets"]["BZR"]["audit"]
    required = (
        "token_row_match",
        "relation_matrix_match",
        "true_feature_match",
        "shuffled_feature_match",
    )
    invariance = bool(audit["passed"]) and all(
        audit["rates"][key] == 1.0 for key in required
    )
    _graphs, items, labels, _features, metadata = _prepare("BZR", args.dataset_root)
    units = []
    for split_seed in SPLIT_SEEDS:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=split_seed)
        for fold, (train, test) in enumerate(
            splitter.split(np.zeros(len(labels)), labels)
        ):
            raw = _matrices(
                items,
                train,
                shuffle_seed=731421 + split_seed * 10000 + fold,
            )
            matrices = _conditional_matrices(raw)
            scores = {
                variant: _fit_score(
                    values[train],
                    labels[train],
                    values[test],
                    labels[test],
                    seed=split_seed * 1000 + fold,
                )
                for variant, values in matrices.items()
            }
            units.append(
                {"split_seed": split_seed, "fold_index": fold, "scores": scores}
            )
            print(
                f"split={split_seed} fold={fold} "
                + " ".join(
                    f"{variant}={scores[variant]['balanced_accuracy']:.4f}"
                    for variant in VARIANTS
                ),
                flush=True,
            )
    variants = {
        variant: {
            "balanced_accuracy_mean": float(
                np.mean([unit["scores"][variant]["balanced_accuracy"] for unit in units])
            ),
            "balanced_accuracy_std": float(
                np.std([unit["scores"][variant]["balanced_accuracy"] for unit in units])
            ),
            "accuracy_mean": float(
                np.mean([unit["scores"][variant]["accuracy"] for unit in units])
            ),
        }
        for variant in VARIANTS
    }
    paired = {
        "true_vs_global": _delta(
            units, "GLOBAL_PLUS_PATCH_GRAPH_TRUE", "GLOBAL_ONLY"
        ),
        "true_vs_bag": _delta(
            units, "GLOBAL_PLUS_PATCH_GRAPH_TRUE", "GLOBAL_PLUS_BAG"
        ),
        "true_vs_token_shuffled": _delta(
            units,
            "GLOBAL_PLUS_PATCH_GRAPH_TRUE",
            "GLOBAL_PLUS_PATCH_GRAPH_TOKEN_SHUFFLED",
        ),
    }
    increment = paired["true_vs_global"]
    relation = paired["true_vs_bag"]
    binding = paired["true_vs_token_shuffled"]
    checks = {
        "conditional_increment": increment["mean"] >= 0.005
        and increment["wins"] >= 4,
        "relation_increment": relation["mean"] >= 0.005 and relation["wins"] >= 4,
        "binding": binding["mean"] >= 0.01 and binding["wins"] >= 4,
        "both_splits_increment": all(
            value > 0 for value in increment["split_means"].values()
        ),
        "both_splits_relation": all(
            value > 0 for value in relation["split_means"].values()
        ),
        "both_splits_binding": all(
            value > 0 for value in binding["split_means"].values()
        ),
        "invariance": invariance,
    }
    decision = (
        "BZR_BEAM8_PATCH_GRAPH_CONDITIONAL_INCREMENT_SUPPORTED"
        if all(checks.values())
        else "BZR_BEAM8_PATCH_GRAPH_CONDITIONAL_INCREMENT_NOT_ESTABLISHED"
    )
    return {
        "protocol": PROTOCOL,
        "prescreen": str(args.prescreen_json),
        "dataset": metadata,
        "config": {"split_seeds": list(SPLIT_SEEDS), "variants": list(VARIANTS)},
        "units": units,
        "summary": {
            "variants": variants,
            "paired": paired,
            "checks": checks,
            "decision": decision,
        },
        "seconds": time.time() - started,
    }


def render(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# BZR Beam8 patch-graph conditional increment",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{summary['decision']}`",
        "",
        "| variant | balanced accuracy over 6 unseen units |",
        "|---|---:|",
    ]
    for variant in VARIANTS:
        row = summary["variants"][variant]
        lines.append(
            f"| {variant} | {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} |"
        )
    lines.extend(
        [
            "",
            "| comparison | mean | W/T/L | split3/4 |",
            "|---|---:|---:|---:|",
        ]
    )
    for comparison, row in summary["paired"].items():
        split_text = " / ".join(
            f"{row['split_means'][str(seed)]:+.4f}" for seed in SPLIT_SEEDS
        )
        lines.append(
            f"| {comparison} | {row['mean']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} | {split_text} |"
        )
    lines.extend(["", "## Frozen checks", ""])
    for check, value in summary["checks"].items():
        lines.append(f"- {check}：`{value}`；")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- 原 BZR prescreen 失败判定保持不变；本轮只检查 GLOBAL_STATS 条件后的独立增量。",
            "- 四个 variants 输入维度一致，BAG 与 GLOBAL_ONLY 使用末尾零 padding。",
            "- split3/4 在协议冻结前未用于选择表示、阈值或分类器。",
            "- 失败时不训练 patch-GNN；通过时也只能在新 split5/6 做一层 matched-control 验证。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--prescreen-json", type=Path, default=PRESCREEN_JSON)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run(args)
    _atomic_json(args.json, payload)
    _atomic_text(args.report, render(payload))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Matched label-only control for the ENZYMES full-attribute/global fusion gain."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data_tud import load_tud, node_feature_readout
from .run_attributed_beam8_frozen_binding_shuffle_audit import (
    PARITY_TOLERANCE,
    _attach_cached,
    _cached_score,
    _select_cached_head_epoch,
    _train_cached_head,
)
from .run_attributed_beam8_node_incidence_classification import _normalize_nodes
from .run_beam8_nci1_chain_classification import ROOT, _atomic_json, _atomic_text
from .run_enzymes_full_attribute_prescreen import (
    DEFAULT_ROOT,
    _cache,
    _load_raw,
    _normalize_raw,
    _score,
    _train,
)
from .run_real_structure_ksvd import graph_basic_features


PROTOCOL = "tracks/ksvd/docs/KSVD_ENZYMES_ATTRIBUTE_ORGANIZATION_FOLLOWUP_PROTOCOL_20260814.md"
RESULT_DIR = ROOT / "tracks/ksvd/results/luyin14"
REFERENCE = RESULT_DIR / "enzymes_full_attribute_prescreen_20260814.json"
DEFAULT_JSON = RESULT_DIR / "enzymes_attribute_organization_followup_20260814.json"
DEFAULT_REPORT = RESULT_DIR / "ENZYMES_ATTRIBUTE_ORGANIZATION_FOLLOWUP_20260814.md"


def _delta(units: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    values = np.asarray([unit["scores"][left] - unit["scores"][right] for unit in units])
    return {
        "mean": float(values.mean()),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
        "split_means": {
            str(seed): float(np.mean([value for value, unit in zip(values, units) if unit["split_seed"] == seed]))
            for seed in (0, 1, 2)
        },
        "model_means": {
            str(seed): float(np.mean([value for value, unit in zip(values, units) if unit["model_seed"] == seed]))
            for seed in (0, 1, 2)
        },
        "values": values.tolist(),
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    device = torch.device(args.device)
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    reference_units = {
        (int(unit["split_seed"]), int(unit["model_seed"]), int(unit["fold_index"])): unit
        for unit in reference["units"]
    }
    graphs, labels, features, metadata = load_tud(
        "ENZYMES", args.dataset_root, use_node_attr=True
    )
    raw, raw_labels, attribute_dim, label_dim = _load_raw(args.dataset_root)
    if not np.array_equal(labels, raw_labels):
        raise RuntimeError("load_tud and PyG labels disagree")
    summaries = np.stack(
        [
            np.concatenate([node_feature_readout(node_features), graph_basic_features(graph)])
            for graph, node_features in zip(graphs, features)
        ]
    )
    global_raw_rows = [
        np.repeat(summary[None, :], graph.n, axis=0)
        for summary, graph in zip(summaries, graphs)
    ]
    classes = int(len(np.unique(labels)))
    units = []
    for split_seed in (0, 1, 2):
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=split_seed)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
            inner_train, validation = train_test_split(
                train,
                test_size=0.2,
                stratify=labels[train],
                random_state=1729 + split_seed * 1000 + fold,
            )
            label_inner = _normalize_raw(
                raw, inner_train, attribute_dim=attribute_dim, label_only=True
            )
            label_outer = _normalize_raw(
                raw, train, attribute_dim=attribute_dim, label_only=True
            )
            inner_global_rows = _normalize_nodes(global_raw_rows, inner_train)
            outer_global_rows = _normalize_nodes(global_raw_rows, train)
            for model_seed in (0, 1, 2):
                key = (split_seed, model_seed, fold)
                reference_unit = reference_units[key]
                seed = model_seed * 1000 + fold
                label_epoch = int(reference_unit["selected_epochs"]["label_gin"])
                inner_model = _train(
                    label_inner,
                    inner_train,
                    epochs=label_epoch,
                    seed=seed,
                    classes=classes,
                    device=device,
                )
                outer_model = _train(
                    label_outer,
                    train,
                    epochs=label_epoch,
                    seed=seed,
                    classes=classes,
                    device=device,
                )
                label_score = _score(outer_model, label_outer, test, device)[
                    "balanced_accuracy"
                ]
                parity_error = abs(
                    label_score - float(reference_unit["scores"]["GIN_LABEL_ONLY"])
                )
                if parity_error > PARITY_TOLERANCE:
                    raise RuntimeError(f"label-only parity failed for {key}: {parity_error}")
                inner_cache = _cache(inner_model, label_inner, device)
                outer_cache = _cache(outer_model, label_outer, device)
                inner_data = _attach_cached(inner_cache, inner_global_rows)
                outer_data = _attach_cached(outer_cache, outer_global_rows)
                struct_dim = int(outer_data[0].s.shape[1])
                residual_epoch = _select_cached_head_epoch(
                    inner_data,
                    inner_train,
                    validation,
                    seed=seed,
                    struct_dim=struct_dim,
                    classes=classes,
                    device=device,
                )
                head = _train_cached_head(
                    outer_data,
                    train,
                    epochs=residual_epoch,
                    seed=seed,
                    struct_dim=struct_dim,
                    classes=classes,
                    device=device,
                )
                label_plus_global = _cached_score(head, outer_data, test, device)[
                    "balanced_accuracy"
                ]
                scores = {
                    "GLOBAL_STATS_LINEAR": float(
                        reference_unit["scores"]["GLOBAL_STATS_LINEAR"]
                    ),
                    "GIN_LABEL_ONLY": label_score,
                    "GIN_LABEL_ONLY_PLUS_GLOBAL": label_plus_global,
                    "GIN_FULL_PLUS_GLOBAL": float(
                        reference_unit["scores"]["GIN_FULL_PLUS_GLOBAL"]
                    ),
                }
                units.append(
                    {
                        "split_seed": split_seed,
                        "model_seed": model_seed,
                        "fold_index": fold,
                        "scores": scores,
                        "selected_epochs": {
                            "label_gin": label_epoch,
                            "label_global_residual": residual_epoch,
                        },
                        "parity_error": parity_error,
                    }
                )
                print(
                    f"split={split_seed} model={model_seed} fold={fold} "
                    + " ".join(f"{name}={value:.4f}" for name, value in scores.items()),
                    flush=True,
                )

    variants = {
        variant: {
            "mean": float(np.mean([unit["scores"][variant] for unit in units])),
            "std": float(np.std([unit["scores"][variant] for unit in units])),
        }
        for variant in (
            "GLOBAL_STATS_LINEAR",
            "GIN_LABEL_ONLY",
            "GIN_LABEL_ONLY_PLUS_GLOBAL",
            "GIN_FULL_PLUS_GLOBAL",
        )
    }
    paired = {
        "full_global_vs_label_global": _delta(
            units, "GIN_FULL_PLUS_GLOBAL", "GIN_LABEL_ONLY_PLUS_GLOBAL"
        ),
        "label_global_vs_global": _delta(
            units, "GIN_LABEL_ONLY_PLUS_GLOBAL", "GLOBAL_STATS_LINEAR"
        ),
        "full_global_vs_global": _delta(
            units, "GIN_FULL_PLUS_GLOBAL", "GLOBAL_STATS_LINEAR"
        ),
    }
    target = paired["full_global_vs_label_global"]
    checks = {
        "mean": target["mean"] >= 0.03,
        "wins": target["wins"] >= 18,
        "all_splits": all(value > 0 for value in target["split_means"].values()),
        "model_majority": int(
            np.sum(np.asarray(list(target["model_means"].values())) > 0)
        )
        >= 2,
        "parity": all(unit["parity_error"] <= PARITY_TOLERANCE for unit in units),
    }
    decision = (
        "CONTINUOUS_ATTRIBUTE_ORGANIZATION_ADDS_BEYOND_GLOBAL_STATS"
        if all(checks.values())
        else "FULL_GLOBAL_GAIN_NOT_ATTRIBUTED_TO_CONTINUOUS_ATTRIBUTE_ORGANIZATION"
    )
    return {
        "protocol": PROTOCOL,
        "dataset": metadata
        | {"continuous_attribute_dim": attribute_dim, "discrete_label_dim": label_dim},
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
        "# ENZYMES continuous-attribute organization follow-up",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{summary['decision']}`",
        "",
        "| variant | balanced accuracy over 27 units |",
        "|---|---:|",
    ]
    for variant, row in summary["variants"].items():
        lines.append(f"| {variant} | {row['mean']:.4f} ± {row['std']:.4f} |")
    lines.extend(
        [
            "",
            "## Paired attribution",
            "",
            "| comparison | mean | W/T/L | split0/1/2 | model0/1/2 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in summary["paired"].items():
        split_text = " / ".join(
            f"{row['split_means'][str(seed)]:+.4f}" for seed in (0, 1, 2)
        )
        model_text = " / ".join(
            f"{row['model_means'][str(seed)]:+.4f}" for seed in (0, 1, 2)
        )
        lines.append(
            f"| {name} | {row['mean']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} | {split_text} | {model_text} |"
        )
    lines.extend(["", "## Checks", ""])
    for name, value in summary["checks"].items():
        lines.append(f"- {name}：`{value}`；")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- label-only 与 full-attribute 分支使用相同统计 residual 容量和训练协议。",
            "- 本轮不事后修改原 ENZYMES Beam8 晋级 gate。",
            "- 通过只说明连续属性的图内组织是独立融合信号，不证明 Beam8 能读取该信号。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--device", default="cpu")
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

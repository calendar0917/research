"""Decompose the global-statistics residual on TU Mutagenicity."""
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
from .run_attributed_beam8_bag_specificity_controls import _bag, _delta, _pad_rows
from .run_attributed_beam8_frozen_binding_shuffle_audit import (
    PARITY_TOLERANCE,
    _attach_cached,
    _cache_base,
    _cached_score,
    _select_cached_head_epoch,
    _train_cached_head,
)
from .run_attributed_beam8_frozen_gine_residual import _train_base
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_attributed_beam8_node_incidence_classification import _normalize_nodes
from .run_beam8_mutagenicity_typed_feasibility import _typed_adjacency
from .run_beam8_nci1_chain_classification import DEFAULT_ROOT, ROOT, _atomic_json, _atomic_text
from .run_luyin14_edge_aware_joint import _attach_edge, _load_edge_pyg
from .run_real_structure_ksvd import graph_basic_features


PROTOCOL = "tracks/ksvd/docs/KSVD_MUTAGENICITY_GLOBAL_STATS_DECOMPOSITION_PROTOCOL_20260814.md"
RESULT_DIR = ROOT / "tracks/ksvd/results/luyin14"
REFERENCE = RESULT_DIR / "attributed_beam8_bag_specificity_controls_20260814.json"
DEFAULT_JSON = RESULT_DIR / "mutagenicity_global_stats_decomposition_20260814.json"
DEFAULT_REPORT = RESULT_DIR / "MUTAGENICITY_GLOBAL_STATS_DECOMPOSITION_20260814.md"
STAT_VARIANTS = (
    "ATTR_MEAN",
    "ATTR_MAX",
    "ATTR_SUM",
    "ATTR_ALL",
    "STRUCT_ONLY",
    "GLOBAL_FULL",
)
VARIANTS = ("GINE_FROZEN", *STAT_VARIANTS)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    device = torch.device(args.device)
    reference = json.loads(REFERENCE.read_text(encoding="utf-8"))
    reference_units = {
        (int(unit["split_seed"]), int(unit["model_seed"]), int(unit["fold_index"])): unit
        for unit in reference["units"]
    }
    graphs, labels, features, metadata = load_tud("Mutagenicity", args.dataset_root)
    raw, raw_labels, classes, edge_dim = _load_edge_pyg("Mutagenicity", args.dataset_root)
    if not np.array_equal(labels, raw_labels):
        raise RuntimeError("the graph and PyG loaders disagree on labels")

    first_typed = _typed_adjacency(raw[0], edge_dim)
    first_item = prepare_attributed_beam_graph(
        0, graphs[0], int(labels[0]), features[0], first_typed, edge_dim=edge_dim
    )
    full_dim = (
        2 * (24 + int(features[0].shape[1]))
        + 8
        + 1
        + int(first_item.position_features.shape[1])
        + 4
        + 2
    )
    feature_sets: dict[str, list[np.ndarray]] = {variant: [] for variant in STAT_VARIANTS}
    for graph, node_features in zip(graphs, features):
        readout = node_feature_readout(node_features)
        width = int(node_features.shape[1])
        mean = readout[:width]
        maximum = readout[width : 2 * width]
        total = readout[2 * width :]
        structure = graph_basic_features(graph)
        summaries = {
            "ATTR_MEAN": mean,
            "ATTR_MAX": maximum,
            "ATTR_SUM": total,
            "ATTR_ALL": readout,
            "STRUCT_ONLY": structure,
            "GLOBAL_FULL": np.concatenate([readout, structure]),
        }
        for variant, summary in summaries.items():
            feature_sets[variant].append(np.repeat(summary[None, :], graph.n, axis=0))

    base_data = _attach_edge(raw, None)
    input_dim = int(raw[0].x.shape[1])
    units = []
    for split_seed in (3, 4):
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=split_seed)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
            split_fold_seed = split_seed * 1000 + fold
            inner_train, validation = train_test_split(
                train,
                test_size=0.2,
                stratify=labels[train],
                random_state=1729 + split_fold_seed,
            )
            stat_rows = {
                variant: _bag(
                    _normalize_nodes(_pad_rows(feature_sets[variant], full_dim), train)
                )
                for variant in STAT_VARIANTS
            }
            for model_seed in (0, 1, 2):
                key = (split_seed, model_seed, fold)
                reference_unit = reference_units[key]
                base_epoch = int(reference_unit["selected_epochs"]["base"])
                seed = model_seed * 1000 + fold
                inner_base = _train_base(
                    base_data,
                    inner_train,
                    epochs=base_epoch,
                    seed=seed,
                    input_dim=input_dim,
                    edge_dim=edge_dim,
                    classes=classes,
                    device=device,
                )
                full_base = _train_base(
                    base_data,
                    train,
                    epochs=base_epoch,
                    seed=seed,
                    input_dim=input_dim,
                    edge_dim=edge_dim,
                    classes=classes,
                    device=device,
                )
                inner_cache = _cache_base(inner_base, base_data, device)
                full_cache = _cache_base(full_base, base_data, device)
                zero_rows = [np.zeros((graph.n, 1), dtype=np.float32) for graph in graphs]
                gine_score = _cached_score(
                    None, _attach_cached(full_cache, zero_rows), test, device
                )["balanced_accuracy"]
                gine_error = abs(
                    gine_score - float(reference_unit["scores"]["GINE_FROZEN"])
                )
                if gine_error > PARITY_TOLERANCE:
                    raise RuntimeError(f"GINE parity failed for {key}: {gine_error}")

                scores = {"GINE_FROZEN": gine_score}
                selected_epochs = {"base": base_epoch}
                for variant in STAT_VARIANTS:
                    inner_data = _attach_cached(inner_cache, stat_rows[variant])
                    full_data = _attach_cached(full_cache, stat_rows[variant])
                    struct_dim = int(full_data[0].s.shape[1])
                    epoch = _select_cached_head_epoch(
                        inner_data,
                        inner_train,
                        validation,
                        seed=seed,
                        struct_dim=struct_dim,
                        classes=classes,
                        device=device,
                    )
                    head = _train_cached_head(
                        full_data,
                        train,
                        epochs=epoch,
                        seed=seed,
                        struct_dim=struct_dim,
                        classes=classes,
                        device=device,
                    )
                    scores[variant] = _cached_score(head, full_data, test, device)[
                        "balanced_accuracy"
                    ]
                    selected_epochs[variant] = epoch
                global_error = abs(
                    scores["GLOBAL_FULL"]
                    - float(reference_unit["scores"]["GLOBAL_STATS"])
                )
                if global_error > PARITY_TOLERANCE:
                    raise RuntimeError(f"GLOBAL parity failed for {key}: {global_error}")
                units.append(
                    {
                        "split_seed": split_seed,
                        "model_seed": model_seed,
                        "fold_index": fold,
                        "scores": scores,
                        "selected_epochs": selected_epochs,
                        "parity": {"gine_error": gine_error, "global_error": global_error},
                    }
                )
                print(
                    f"split={split_seed} model={model_seed} fold={fold} "
                    + " ".join(f"{name}={scores[name]:.4f}" for name in VARIANTS),
                    flush=True,
                )

    variants = {
        variant: {
            "mean": float(np.mean([unit["scores"][variant] for unit in units])),
            "std": float(np.std([unit["scores"][variant] for unit in units])),
        }
        for variant in VARIANTS
    }
    paired = {
        f"{variant.lower()}_vs_gine": _delta(units, variant, "GINE_FROZEN")
        for variant in STAT_VARIANTS
    }
    global_minus = {
        variant: _delta(units, "GLOBAL_FULL", variant) for variant in STAT_VARIANTS[:-1]
    }
    near_sufficient = {
        variant: row["mean"] <= 0.005 for variant, row in global_minus.items()
    }
    checks = {
        "parity": all(
            unit["parity"]["gine_error"] <= PARITY_TOLERANCE
            and unit["parity"]["global_error"] <= PARITY_TOLERANCE
            for unit in units
        )
    }
    return {
        "protocol": PROTOCOL,
        "dataset": metadata | {"edge_dim": int(edge_dim)},
        "config": {
            "split_seeds": [3, 4],
            "model_seeds": [0, 1, 2],
            "padded_dimension": full_dim,
        },
        "units": units,
        "summary": {
            "variants": variants,
            "paired_vs_gine": paired,
            "global_minus_component": global_minus,
            "near_sufficient": near_sufficient,
            "checks": checks,
        },
        "seconds": time.time() - started,
    }


def render(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Mutagenicity global-statistics residual decomposition",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> parity：`{summary['checks']['parity']}`",
        "",
        "| variant | balanced accuracy over 18 units | delta vs GINE |",
        "|---|---:|---:|",
    ]
    for variant in VARIANTS:
        row = summary["variants"][variant]
        delta = 0.0 if variant == "GINE_FROZEN" else summary["paired_vs_gine"][
            f"{variant.lower()}_vs_gine"
        ]["mean"]
        lines.append(f"| {variant} | {row['mean']:.4f} ± {row['std']:.4f} | {delta:+.4f} |")
    lines.extend(
        [
            "",
            "## Sufficiency relative to GLOBAL_FULL",
            "",
            "| component | GLOBAL−component | W/T/L | near-sufficient (<=0.5pt) |",
            "|---|---:|---:|---:|",
        ]
    )
    for variant, row in summary["global_minus_component"].items():
        lines.append(
            f"| {variant} | {row['mean']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} | {summary['near_sufficient'][variant]} |"
        )
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- 所有统计只在 outer-train 上归一化，GINE/GLOBAL_FULL 与原实验精确 parity。",
            "- near-sufficient 只表示该统计族足以解释大部分校准，不表示因果机制。",
            "- 本结果用于筛选后续真实 TUD 数据集，不授权继续扩大 Mutagenicity 上的 Beam8 模型。",
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

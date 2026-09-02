"""Unseen-split confirmation of the ENZYMES full-attribute/global baseline."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_tud import load_tud, node_feature_readout
from .run_attributed_beam8_frozen_binding_shuffle_audit import (
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
    _select_epoch,
    _train,
)
from .run_real_structure_ksvd import graph_basic_features


PROTOCOL = "tracks/ksvd/docs/KSVD_ENZYMES_MULTIMODAL_UNSEEN_CONFIRMATION_PROTOCOL_20260814.md"
RESULT_DIR = ROOT / "tracks/ksvd/results/luyin14"
DEFAULT_JSON = RESULT_DIR / "enzymes_multimodal_unseen_confirmation_20260814.json"
DEFAULT_REPORT = RESULT_DIR / "ENZYMES_MULTIMODAL_UNSEEN_CONFIRMATION_20260814.md"
VARIANTS = (
    "GLOBAL_STATS_LINEAR",
    "GIN_FULL_ATTRIBUTES",
    "GIN_LABEL_ONLY_PLUS_GLOBAL",
    "GIN_FULL_PLUS_GLOBAL",
)


def _delta(units: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    values = np.asarray([unit["scores"][left] - unit["scores"][right] for unit in units])
    return {
        "mean": float(values.mean()),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
        "split_means": {
            str(seed): float(np.mean([value for value, unit in zip(values, units) if unit["split_seed"] == seed]))
            for seed in (3, 4)
        },
        "model_means": {
            str(seed): float(np.mean([value for value, unit in zip(values, units) if unit["model_seed"] == seed]))
            for seed in (0, 1, 2)
        },
        "values": values.tolist(),
    }


def _fit_residual(
    inner_model,
    outer_model,
    inner_data,
    outer_data,
    inner_train,
    validation,
    train,
    test,
    global_inner,
    global_outer,
    *,
    seed: int,
    classes: int,
    device: torch.device,
) -> tuple[float, int]:
    inner_cache = _cache(inner_model, inner_data, device)
    outer_cache = _cache(outer_model, outer_data, device)
    cached_inner = _attach_cached(inner_cache, global_inner)
    cached_outer = _attach_cached(outer_cache, global_outer)
    struct_dim = int(cached_outer[0].s.shape[1])
    epoch = _select_cached_head_epoch(
        cached_inner,
        inner_train,
        validation,
        seed=seed,
        struct_dim=struct_dim,
        classes=classes,
        device=device,
    )
    head = _train_cached_head(
        cached_outer,
        train,
        epochs=epoch,
        seed=seed,
        struct_dim=struct_dim,
        classes=classes,
        device=device,
    )
    score = _cached_score(head, cached_outer, test, device)["balanced_accuracy"]
    return float(score), int(epoch)


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    device = torch.device(args.device)
    graphs, labels, features, metadata = load_tud(
        "ENZYMES", args.dataset_root, use_node_attr=True
    )
    raw, raw_labels, attribute_dim, label_dim = _load_raw(args.dataset_root)
    if not np.array_equal(labels, raw_labels):
        raise RuntimeError("load_tud and PyG labels disagree")
    if attribute_dim != 18 or label_dim != 3:
        raise RuntimeError(f"unexpected dimensions: {attribute_dim}+{label_dim}")
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
    for split_seed in (3, 4):
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=split_seed)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
            inner_train, validation = train_test_split(
                train,
                test_size=0.2,
                stratify=labels[train],
                random_state=1729 + split_seed * 1000 + fold,
            )
            linear = make_pipeline(
                StandardScaler(),
                LogisticRegression(
                    max_iter=3000,
                    class_weight="balanced",
                    random_state=split_seed * 1000 + fold,
                ),
            )
            linear.fit(summaries[train], labels[train])
            global_score = float(
                balanced_accuracy_score(labels[test], linear.predict(summaries[test]))
            )
            label_inner = _normalize_raw(
                raw, inner_train, attribute_dim=attribute_dim, label_only=True
            )
            label_outer = _normalize_raw(raw, train, attribute_dim=attribute_dim, label_only=True)
            full_inner = _normalize_raw(
                raw, inner_train, attribute_dim=attribute_dim, label_only=False
            )
            full_outer = _normalize_raw(raw, train, attribute_dim=attribute_dim, label_only=False)
            global_inner = _normalize_nodes(global_raw_rows, inner_train)
            global_outer = _normalize_nodes(global_raw_rows, train)

            for model_seed in (0, 1, 2):
                seed = model_seed * 1000 + fold
                label_epoch = _select_epoch(
                    label_inner,
                    inner_train,
                    validation,
                    seed=seed,
                    classes=classes,
                    device=device,
                )
                label_inner_model = _train(
                    label_inner,
                    inner_train,
                    epochs=label_epoch,
                    seed=seed,
                    classes=classes,
                    device=device,
                )
                label_outer_model = _train(
                    label_outer,
                    train,
                    epochs=label_epoch,
                    seed=seed,
                    classes=classes,
                    device=device,
                )
                label_combined, label_residual_epoch = _fit_residual(
                    label_inner_model,
                    label_outer_model,
                    label_inner,
                    label_outer,
                    inner_train,
                    validation,
                    train,
                    test,
                    global_inner,
                    global_outer,
                    seed=seed,
                    classes=classes,
                    device=device,
                )

                full_epoch = _select_epoch(
                    full_inner,
                    inner_train,
                    validation,
                    seed=seed,
                    classes=classes,
                    device=device,
                )
                full_inner_model = _train(
                    full_inner,
                    inner_train,
                    epochs=full_epoch,
                    seed=seed,
                    classes=classes,
                    device=device,
                )
                full_outer_model = _train(
                    full_outer,
                    train,
                    epochs=full_epoch,
                    seed=seed,
                    classes=classes,
                    device=device,
                )
                full_score = _score(full_outer_model, full_outer, test, device)[
                    "balanced_accuracy"
                ]
                full_combined, full_residual_epoch = _fit_residual(
                    full_inner_model,
                    full_outer_model,
                    full_inner,
                    full_outer,
                    inner_train,
                    validation,
                    train,
                    test,
                    global_inner,
                    global_outer,
                    seed=seed,
                    classes=classes,
                    device=device,
                )
                scores = {
                    "GLOBAL_STATS_LINEAR": global_score,
                    "GIN_FULL_ATTRIBUTES": full_score,
                    "GIN_LABEL_ONLY_PLUS_GLOBAL": label_combined,
                    "GIN_FULL_PLUS_GLOBAL": full_combined,
                }
                units.append(
                    {
                        "split_seed": split_seed,
                        "model_seed": model_seed,
                        "fold_index": fold,
                        "scores": scores,
                        "selected_epochs": {
                            "label_gin": label_epoch,
                            "label_residual": label_residual_epoch,
                            "full_gin": full_epoch,
                            "full_residual": full_residual_epoch,
                        },
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
        "full_global_vs_global": _delta(
            units, "GIN_FULL_PLUS_GLOBAL", "GLOBAL_STATS_LINEAR"
        ),
        "full_global_vs_label_global": _delta(
            units, "GIN_FULL_PLUS_GLOBAL", "GIN_LABEL_ONLY_PLUS_GLOBAL"
        ),
        "full_vs_global": _delta(units, "GIN_FULL_ATTRIBUTES", "GLOBAL_STATS_LINEAR"),
    }
    first = paired["full_global_vs_global"]
    second = paired["full_global_vs_label_global"]
    checks = {
        "combined_increment": first["mean"] >= 0.05 and first["wins"] >= 12,
        "attribute_organization": second["mean"] >= 0.05 and second["wins"] >= 12,
        "both_splits_increment": all(value > 0 for value in first["split_means"].values()),
        "both_splits_organization": all(value > 0 for value in second["split_means"].values()),
        "model_majority_increment": int(
            np.sum(np.asarray(list(first["model_means"].values())) > 0)
        )
        >= 2,
        "model_majority_organization": int(
            np.sum(np.asarray(list(second["model_means"].values())) > 0)
        )
        >= 2,
        "dimensions": attribute_dim == 18 and label_dim == 3,
    }
    decision = (
        "ENZYMES_MULTIMODAL_BASE_CONFIRMED_ON_UNSEEN_SPLITS"
        if all(checks.values())
        else "ENZYMES_MULTIMODAL_BASE_NOT_CONFIRMED"
    )
    return {
        "protocol": PROTOCOL,
        "dataset": metadata
        | {"continuous_attribute_dim": attribute_dim, "discrete_label_dim": label_dim},
        "config": {"split_seeds": [3, 4], "model_seeds": [0, 1, 2]},
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
        "# ENZYMES multimodal baseline unseen-split confirmation",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{summary['decision']}`",
        "",
        "| variant | balanced accuracy over 18 units |",
        "|---|---:|",
    ]
    for variant in VARIANTS:
        row = summary["variants"][variant]
        lines.append(f"| {variant} | {row['mean']:.4f} ± {row['std']:.4f} |")
    lines.extend(
        [
            "",
            "## Paired confirmation",
            "",
            "| comparison | mean | W/T/L | split3/4 | model0/1/2 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in summary["paired"].items():
        split_text = " / ".join(
            f"{row['split_means'][str(seed)]:+.4f}" for seed in (3, 4)
        )
        model_text = " / ".join(
            f"{row['model_means'][str(seed)]:+.4f}" for seed in (0, 1, 2)
        )
        lines.append(
            f"| {name} | {row['mean']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} | {split_text} | {model_text} |"
        )
    lines.extend(["", "## Frozen checks", ""])
    for name, value in summary["checks"].items():
        lines.append(f"- {name}：`{value}`；")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- split3/4 在协议冻结前未用于 ENZYMES 模型选择或阈值调整。",
            "- 通过只确认强多模态基线；Beam8 必须在新的 split5/6 上做条件增量 matched controls。",
            "- 失败时停止 ENZYMES Beam8 分类路线。",
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

"""Canonical-slot continuous-attribute Beam8 controls on unseen ENZYMES splits 9/10."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data_tud import load_tud, node_feature_readout
from .run_attributed_beam8_bag_specificity_controls import (
    _bag,
    _pad_rows,
    _random_dictionary,
)
from .run_attributed_beam8_conditional_increment import _offset_cache
from .run_attributed_beam8_frozen_binding_shuffle_audit import (
    _attach_cached,
    _cached_score,
    _select_cached_head_epoch,
    _train_cached_head,
)
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_attributed_beam8_node_incidence_classification import (
    _normalize_nodes,
    _orbit_shuffle,
)
from .run_beam8_nci1_chain_classification import (
    ROOT,
    _atomic_json,
    _atomic_text,
    _encode,
    _fit_dictionary,
)
from .run_enzymes_beam8_conditional_controls import (
    DEFAULT_ROOT,
    _binary_typed,
    _continuous_feature_colors,
    _incidence_rows,
)
from .run_enzymes_full_attribute_prescreen import (
    _cache,
    _load_raw,
    _normalize_raw,
    _select_epoch,
    _train,
)
from .run_real_structure_ksvd import graph_basic_features


PROTOCOL = "tracks/ksvd/docs/KSVD_ENZYMES_BEAM8_SLOT_ATTRIBUTE_PROTOCOL_20260814.md"
RESULT_DIR = ROOT / "tracks/ksvd/results/luyin14"
DEFAULT_JSON = RESULT_DIR / "enzymes_beam8_slot_attribute_controls_20260814.json"
DEFAULT_REPORT = RESULT_DIR / "ENZYMES_BEAM8_SLOT_ATTRIBUTE_CONTROLS_20260814.md"
SPLIT_SEEDS = (9, 10)
SECOND_VARIANTS = (
    "SLOT_FULL_TRUE",
    "SLOT_FULL_BAG",
    "SLOT_FULL_INCIDENCE_SHUFFLED",
    "SLOT_FULL_WITHIN_PATCH_SHUFFLED",
    "SLOT_CODE_ONLY_TRUE",
    "MEAN_FULL_TRUE",
    "SLOT_RANDOM_DICTIONARY_TRUE",
)
VARIANTS = ("BASE_MULTIMODAL", *SECOND_VARIANTS)


def slot_attribute_tokens(
    item: Any,
    content: np.ndarray,
    *,
    attribute_dim: int = 18,
    patch_size: int = 8,
) -> np.ndarray:
    """Flatten continuous attributes in canonical patch slots and append a mask."""
    values = np.asarray(content, dtype=np.float64)[:, :attribute_dim]
    output = []
    for nodes in item.slot_nodes:
        if len(nodes) > patch_size:
            raise ValueError("Beam8 patch exceeds fixed slot capacity")
        slots = np.zeros((patch_size, attribute_dim), dtype=np.float64)
        mask = np.zeros(patch_size, dtype=np.float64)
        count = len(nodes)
        if count:
            slots[:count] = values[np.asarray(nodes, dtype=np.int64)]
            mask[:count] = 1.0
        output.append(np.concatenate([slots.reshape(-1), mask]))
    return np.stack(output)


def shuffled_slot_attribute_tokens(
    item: Any,
    content: np.ndarray,
    *,
    seed: int,
    attribute_dim: int = 18,
    patch_size: int = 8,
) -> np.ndarray:
    """Preserve each patch's attribute multiset while breaking structural-slot binding."""
    values = np.asarray(content, dtype=np.float64)[:, :attribute_dim]
    rng = np.random.default_rng(seed)
    output = []
    for nodes in item.slot_nodes:
        if len(nodes) > patch_size:
            raise ValueError("Beam8 patch exceeds fixed slot capacity")
        slots = np.zeros((patch_size, attribute_dim), dtype=np.float64)
        mask = np.zeros(patch_size, dtype=np.float64)
        count = len(nodes)
        if count:
            permutation = rng.permutation(count)
            if count > 1 and np.array_equal(permutation, np.arange(count)):
                permutation = np.roll(permutation, 1)
            source = np.asarray(nodes, dtype=np.int64)[permutation]
            slots[:count] = values[source]
            mask[:count] = 1.0
        output.append(np.concatenate([slots.reshape(-1), mask]))
    return np.stack(output)


def _delta(units: list[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    values = np.asarray([unit["scores"][left] - unit["scores"][right] for unit in units])
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
        "model_means": {
            str(seed): float(
                np.mean(
                    [value for value, unit in zip(values, units) if unit["model_seed"] == seed]
                )
            )
            for seed in (0, 1, 2)
        },
        "values": values.tolist(),
    }


def _positive_axis(row: dict[str, Any], axis: str) -> bool:
    return all(value > 0 for value in row[axis].values())


def _model_majority(row: dict[str, Any]) -> bool:
    return int(np.sum(np.asarray(list(row["model_means"].values())) > 0)) >= 2


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    device = torch.device(args.device)
    graphs, labels, content_features, metadata = load_tud(
        "ENZYMES", args.dataset_root, use_node_attr=True
    )
    _graphs_discrete, labels_discrete, discrete_features, discrete_meta = load_tud(
        "ENZYMES", args.dataset_root, use_node_attr=False
    )
    raw, raw_labels, attribute_dim, label_dim = _load_raw(args.dataset_root)
    if not np.array_equal(labels, raw_labels) or not np.array_equal(labels, labels_discrete):
        raise RuntimeError("ENZYMES loaders disagree on labels")
    if attribute_dim != 18 or label_dim != 3:
        raise RuntimeError(f"unexpected dimensions: {attribute_dim}+{label_dim}")

    typed_graphs = [_binary_typed(graph) for graph in graphs]
    canonical_types = [
        _continuous_feature_colors(np.asarray(features)) for features in content_features
    ]
    items = []
    for index, (graph, label, content, discrete, colors, typed) in enumerate(
        zip(
            graphs,
            labels,
            content_features,
            discrete_features,
            canonical_types,
            typed_graphs,
        )
    ):
        items.append(
            prepare_attributed_beam_graph(
                index,
                graph,
                int(label),
                np.asarray(content),
                typed,
                edge_dim=1,
                canonical_node_features=np.asarray(discrete),
                canonical_node_types=np.asarray(colors),
            )
        )
        if (index + 1) % 100 == 0 or index + 1 == len(graphs):
            print(f"prepare {index + 1}/{len(graphs)}", flush=True)

    true_slots = [
        slot_attribute_tokens(item, np.asarray(content), attribute_dim=attribute_dim)
        for item, content in zip(items, content_features)
    ]
    continuous_means = [
        np.stack(
            [
                np.asarray(content)[np.asarray(nodes, dtype=np.int64), :attribute_dim].mean(axis=0)
                for nodes in item.slot_nodes
            ]
        )
        for item, content in zip(items, content_features)
    ]
    summaries = np.stack(
        [
            np.concatenate([node_feature_readout(features), graph_basic_features(graph)])
            for graph, features in zip(graphs, content_features)
        ]
    )
    global_raw_rows = [
        np.repeat(summary[None, :], graph.n, axis=0)
        for summary, graph in zip(summaries, graphs)
    ]
    classes = int(len(np.unique(labels)))
    units = []

    for split_seed in SPLIT_SEEDS:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=split_seed)
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
            split_fold_seed = split_seed * 1000 + fold
            inner_train, validation = train_test_split(
                train,
                test_size=0.2,
                stratify=labels[train],
                random_state=1729 + split_fold_seed,
            )
            dictionary = _fit_dictionary(
                items,
                train,
                n_atoms=24,
                sparsity=3,
                iterations=5,
                max_train_patches=3000,
                seed=split_fold_seed,
            )
            random_dictionary, random_mean = _random_dictionary(
                items,
                train,
                n_atoms=24,
                max_train_patches=3000,
                seed=split_fold_seed,
            )
            if not np.allclose(random_mean, dictionary["mean"]):
                raise RuntimeError("random and deterministic dictionaries use different pools")
            deterministic_codes = [
                _encode(item, dictionary["initial"], dictionary["mean"], 3) for item in items
            ]
            random_codes = [
                _encode(item, random_dictionary, random_mean, 3) for item in items
            ]
            shuffled_slots = [
                shuffled_slot_attribute_tokens(
                    item,
                    np.asarray(content),
                    seed=271828 + split_fold_seed * 100000 + index * 1009,
                    attribute_dim=attribute_dim,
                )
                for index, (item, content) in enumerate(zip(items, content_features))
            ]
            token_families = {
                "SLOT_FULL_TRUE": [
                    np.concatenate([code, slot], axis=1)
                    for code, slot in zip(deterministic_codes, true_slots)
                ],
                "SLOT_FULL_WITHIN_PATCH_SHUFFLED": [
                    np.concatenate([code, slot], axis=1)
                    for code, slot in zip(deterministic_codes, shuffled_slots)
                ],
                "SLOT_CODE_ONLY_TRUE": deterministic_codes,
                "MEAN_FULL_TRUE": [
                    np.concatenate([code, mean], axis=1)
                    for code, mean in zip(deterministic_codes, continuous_means)
                ],
                "SLOT_RANDOM_DICTIONARY_TRUE": [
                    np.concatenate([code, slot], axis=1)
                    for code, slot in zip(random_codes, true_slots)
                ],
            }
            raw_rows = {
                variant: _incidence_rows(
                    items,
                    typed_graphs,
                    content_features,
                    discrete_features,
                    canonical_types,
                    graphs,
                    tokens,
                )
                for variant, tokens in token_families.items()
            }
            full_dim = int(raw_rows["SLOT_FULL_TRUE"][0].shape[1])
            normalized = {
                variant: _normalize_nodes(_pad_rows(rows, full_dim), train)
                for variant, rows in raw_rows.items()
            }
            normalized["SLOT_FULL_BAG"] = _bag(normalized["SLOT_FULL_TRUE"])
            normalized["SLOT_FULL_INCIDENCE_SHUFFLED"] = [
                _orbit_shuffle(
                    values,
                    typed,
                    np.asarray(content),
                    seed=161803 + split_fold_seed * 100000 + index * 1009,
                    canonical_node_features=np.asarray(discrete),
                    canonical_node_types=np.asarray(colors),
                )
                for index, (values, typed, content, discrete, colors) in enumerate(
                    zip(
                        normalized["SLOT_FULL_TRUE"],
                        typed_graphs,
                        content_features,
                        discrete_features,
                        canonical_types,
                    )
                )
            ]

            full_inner = _normalize_raw(
                raw, inner_train, attribute_dim=attribute_dim, label_only=False
            )
            full_outer = _normalize_raw(
                raw, train, attribute_dim=attribute_dim, label_only=False
            )
            global_inner = _normalize_nodes(global_raw_rows, inner_train)
            global_outer = _normalize_nodes(global_raw_rows, train)

            for model_seed in (0, 1, 2):
                seed = model_seed * 1000 + fold
                base_epoch = _select_epoch(
                    full_inner,
                    inner_train,
                    validation,
                    seed=seed,
                    classes=classes,
                    device=device,
                )
                inner_base = _train(
                    full_inner,
                    inner_train,
                    epochs=base_epoch,
                    seed=seed,
                    classes=classes,
                    device=device,
                )
                outer_base = _train(
                    full_outer,
                    train,
                    epochs=base_epoch,
                    seed=seed,
                    classes=classes,
                    device=device,
                )
                inner_cache = _cache(inner_base, full_inner, device)
                outer_cache = _cache(outer_base, full_outer, device)
                inner_global_data = _attach_cached(inner_cache, global_inner)
                outer_global_data = _attach_cached(outer_cache, global_outer)
                global_dim = int(outer_global_data[0].s.shape[1])
                global_epoch = _select_cached_head_epoch(
                    inner_global_data,
                    inner_train,
                    validation,
                    seed=seed,
                    struct_dim=global_dim,
                    classes=classes,
                    device=device,
                )
                inner_global_head = _train_cached_head(
                    inner_global_data,
                    inner_train,
                    epochs=global_epoch,
                    seed=seed,
                    struct_dim=global_dim,
                    classes=classes,
                    device=device,
                )
                outer_global_head = _train_cached_head(
                    outer_global_data,
                    train,
                    epochs=global_epoch,
                    seed=seed,
                    struct_dim=global_dim,
                    classes=classes,
                    device=device,
                )
                inner_offset = _offset_cache(
                    inner_cache, global_inner, inner_global_head, device
                )
                outer_offset = _offset_cache(
                    outer_cache, global_outer, outer_global_head, device
                )
                zero_rows = [
                    np.zeros((graph.n, 1), dtype=np.float32) for graph in graphs
                ]
                scores = {
                    "BASE_MULTIMODAL": _cached_score(
                        None, _attach_cached(outer_offset, zero_rows), test, device
                    )["balanced_accuracy"]
                }
                selected_epochs = {"gin": base_epoch, "global_residual": global_epoch}
                for variant in SECOND_VARIANTS:
                    inner_data = _attach_cached(inner_offset, normalized[variant])
                    outer_data = _attach_cached(outer_offset, normalized[variant])
                    second_epoch = _select_cached_head_epoch(
                        inner_data,
                        inner_train,
                        validation,
                        seed=seed,
                        struct_dim=full_dim,
                        classes=classes,
                        device=device,
                    )
                    head = _train_cached_head(
                        outer_data,
                        train,
                        epochs=second_epoch,
                        seed=seed,
                        struct_dim=full_dim,
                        classes=classes,
                        device=device,
                    )
                    scores[variant] = _cached_score(head, outer_data, test, device)[
                        "balanced_accuracy"
                    ]
                    selected_epochs[variant] = second_epoch
                units.append(
                    {
                        "split_seed": split_seed,
                        "model_seed": model_seed,
                        "fold_index": fold,
                        "scores": scores,
                        "selected_epochs": selected_epochs,
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
        "full_vs_base": _delta(units, "SLOT_FULL_TRUE", "BASE_MULTIMODAL"),
        "full_vs_bag": _delta(units, "SLOT_FULL_TRUE", "SLOT_FULL_BAG"),
        "full_vs_incidence_shuffled": _delta(
            units, "SLOT_FULL_TRUE", "SLOT_FULL_INCIDENCE_SHUFFLED"
        ),
        "full_vs_within_patch_shuffled": _delta(
            units, "SLOT_FULL_TRUE", "SLOT_FULL_WITHIN_PATCH_SHUFFLED"
        ),
        "full_vs_code_only": _delta(units, "SLOT_FULL_TRUE", "SLOT_CODE_ONLY_TRUE"),
        "full_vs_mean": _delta(units, "SLOT_FULL_TRUE", "MEAN_FULL_TRUE"),
        "full_vs_random_dictionary": _delta(
            units, "SLOT_FULL_TRUE", "SLOT_RANDOM_DICTIONARY_TRUE"
        ),
    }
    increment = paired["full_vs_base"]
    slot_mean = paired["full_vs_mean"]
    within_binding = paired["full_vs_within_patch_shuffled"]
    checks = {
        "increment": increment["mean"] >= 0.01 and increment["wins"] >= 12,
        "localization": paired["full_vs_bag"]["mean"] >= 0.005
        and paired["full_vs_bag"]["wins"] >= 12,
        "incidence_binding": paired["full_vs_incidence_shuffled"]["mean"] >= 0.005
        and paired["full_vs_incidence_shuffled"]["wins"] >= 12,
        "slot_beats_mean": slot_mean["mean"] >= 0.005 and slot_mean["wins"] >= 12,
        "within_patch_binding": within_binding["mean"] >= 0.005
        and within_binding["wins"] >= 12,
        "beats_random_dictionary": paired["full_vs_random_dictionary"]["mean"] >= 0.0025
        and paired["full_vs_random_dictionary"]["wins"] >= 11,
        "both_splits_increment": _positive_axis(increment, "split_means"),
        "both_splits_slot_mean": _positive_axis(slot_mean, "split_means"),
        "both_splits_within_binding": _positive_axis(within_binding, "split_means"),
        "model_majority_increment": _model_majority(increment),
        "model_majority_slot_mean": _model_majority(slot_mean),
        "model_majority_within_binding": _model_majority(within_binding),
    }
    decision = (
        "ENZYMES_CANONICAL_SLOT_ATTRIBUTE_BEAM8_INCREMENT_SUPPORTED"
        if all(checks.values())
        else "ENZYMES_CANONICAL_SLOT_ATTRIBUTE_BEAM8_INCREMENT_NOT_ESTABLISHED"
    )
    return {
        "protocol": PROTOCOL,
        "dataset": metadata
        | {
            "continuous_attribute_dim": attribute_dim,
            "discrete_label_dim": label_dim,
            "canonical_feature_dim": discrete_meta["node_feat_dim"],
        },
        "config": {
            "split_seeds": list(SPLIT_SEEDS),
            "model_seeds": [0, 1, 2],
            "patch_size": 8,
            "slot_attribute_dim": 8 * attribute_dim + 8,
            "full_patch_token_dim": 24 + 8 * attribute_dim + 8,
        },
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
        "# ENZYMES canonical-slot attribute Beam8 controls",
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
            "## Paired attribution",
            "",
            "| comparison | mean | W/T/L | split9/10 | model0/1/2 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in summary["paired"].items():
        split_text = " / ".join(
            f"{row['split_means'][str(seed)]:+.4f}" for seed in SPLIT_SEEDS
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
            "- FULL patch token = 24-d train-only INIT code + canonical 8×18 continuous attributes + 8-d mask。",
            "- BASE、dictionary、normalization、rank-16 frozen residual 与上一轮一致；只改变属性表示。",
            "- split9/10 在协议冻结前未用于表示、阈值或超参数选择。",
            "- 分类结果只有在独立 slot-token invariance audit 100%通过后才有效。",
            "- 失败时停止 ENZYMES Beam8 分类，不扩大模型补救。",
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

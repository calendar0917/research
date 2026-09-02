"""Matched controls for Beam8-specific graph-conditioned frozen residual gains."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data_tud import load_tud
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
from .run_attributed_beam8_node_incidence_feasibility import node_incidence_features, orbit_safe_features
from .run_beam8_mutagenicity_typed_feasibility import _typed_adjacency
from .run_beam8_nci1_chain_classification import (
    DEFAULT_ROOT,
    ROOT,
    _atomic_json,
    _atomic_text,
    _encode,
    _fit_dictionary,
)
from .run_luyin14_edge_aware_joint import _attach_edge, _load_edge_pyg


PROTOCOL = "tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_BAG_SPECIFICITY_CONTROLS_PROTOCOL_20260814.md"
RESULT_DIR = ROOT / "tracks/ksvd/results/luyin14"
DEFAULT_JSON = RESULT_DIR / "attributed_beam8_bag_specificity_controls_20260814.json"
DEFAULT_REPORT = RESULT_DIR / "ATTRIBUTED_BEAM8_BAG_SPECIFICITY_CONTROLS_20260814.md"
VARIANTS = (
    "GINE_FROZEN",
    "BEAM8_FULL",
    "BEAM8_CODE_ONLY",
    "BEAM8_HIST_ONLY",
    "BEAM8_RANDOM_DICTIONARY",
    "GLOBAL_STATS",
)


def _reference_path(split_seed: int, model_seed: int) -> Path:
    if model_seed == 0:
        return RESULT_DIR / f"attributed_beam8_frozen_gine_residual_split_seed{split_seed}_20260814.json"
    return RESULT_DIR / f"attributed_beam8_frozen_bag_residual_split_seed{split_seed}_model_seed{model_seed}_20260814.json"


def _random_dictionary(
    items: Sequence[Any],
    train: Sequence[int],
    *,
    n_atoms: int,
    max_train_patches: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray]:
    raw = np.concatenate([items[int(index)].vectors for index in train], axis=0).T
    if raw.shape[1] > max_train_patches:
        selected = np.sort(
            np.random.default_rng(seed).choice(raw.shape[1], size=max_train_patches, replace=False)
        )
        raw = raw[:, selected]
    mean = raw.mean(axis=1, keepdims=True)
    centered = raw - mean
    nonzero = np.flatnonzero(np.linalg.norm(centered, axis=0) > 1e-12)
    atoms = min(n_atoms, len(nonzero), centered.shape[0])
    selected = np.random.default_rng(seed + 104729).choice(nonzero, size=atoms, replace=False)
    dictionary = centered[:, selected].copy()
    dictionary /= np.maximum(np.linalg.norm(dictionary, axis=0, keepdims=True), 1e-12)
    return dictionary, mean


def _incidence_rows(
    items: Sequence[Any],
    typed_graphs: Sequence[np.ndarray],
    node_features: Sequence[np.ndarray],
    graphs: Sequence[Any],
    tokens: Sequence[np.ndarray],
) -> list[np.ndarray]:
    rows = []
    for item, typed, features, graph, patch_tokens in zip(
        items, typed_graphs, node_features, graphs, tokens
    ):
        direct, _diagnostic = node_incidence_features(
            item,
            graph.n,
            tokens=patch_tokens,
            include_relation=False,
        )
        safe, _orbits = orbit_safe_features(direct, typed, np.asarray(features))
        rows.append(safe)
    return rows


def _pad_rows(rows: Sequence[np.ndarray], dimension: int) -> list[np.ndarray]:
    output = []
    for row in rows:
        if row.shape[1] > dimension:
            raise ValueError(f"control dimension {row.shape[1]} exceeds FULL dimension {dimension}")
        output.append(np.pad(row, ((0, 0), (0, dimension - row.shape[1]))))
    return output


def _bag(rows: Sequence[np.ndarray]) -> list[np.ndarray]:
    return [np.repeat(row.mean(0, keepdims=True), len(row), axis=0) for row in rows]


def _delta(units: Sequence[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    values = np.asarray([unit["scores"][left] - unit["scores"][right] for unit in units])
    split_means = {
        str(seed): float(np.mean([value for value, unit in zip(values, units) if unit["split_seed"] == seed]))
        for seed in (3, 4)
    }
    model_means = {
        str(seed): float(np.mean([value for value, unit in zip(values, units) if unit["model_seed"] == seed]))
        for seed in (0, 1, 2)
    }
    return {
        "mean": float(values.mean()),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
        "values": values.tolist(),
        "split_means": split_means,
        "model_means": model_means,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    device = torch.device(args.device)
    graphs, labels, features, metadata = load_tud("Mutagenicity", args.dataset_root)
    raw, raw_labels, classes, edge_dim = _load_edge_pyg("Mutagenicity", args.dataset_root)
    if not np.array_equal(labels, raw_labels):
        raise RuntimeError("the graph and PyG loaders disagree on labels")

    items, typed_graphs = [], []
    for index, (graph, label, node_features, data) in enumerate(zip(graphs, labels, features, raw)):
        typed = _typed_adjacency(data, edge_dim)
        typed_graphs.append(typed)
        items.append(
            prepare_attributed_beam_graph(
                index,
                graph,
                int(label),
                node_features,
                typed,
                edge_dim=edge_dim,
            )
        )
        if (index + 1) % 500 == 0 or index + 1 == len(graphs):
            print(f"prepare {index + 1}/{len(graphs)}", flush=True)

    base_data = _attach_edge(raw, None)
    input_dim = int(raw[0].x.shape[1])
    units = []
    for split_seed in (3, 4):
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=split_seed)
        references = {
            model_seed: json.loads(_reference_path(split_seed, model_seed).read_text(encoding="utf-8"))
            for model_seed in (0, 1, 2)
        }
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
                raise RuntimeError("random and deterministic dictionaries use different train pools")

            deterministic_codes = [
                _encode(item, dictionary["initial"], dictionary["mean"], 3) for item in items
            ]
            random_codes = [_encode(item, random_dictionary, random_mean, 3) for item in items]
            token_families = {
                "BEAM8_FULL": [
                    np.concatenate([code, item.node_histograms], axis=1)
                    for code, item in zip(deterministic_codes, items)
                ],
                "BEAM8_CODE_ONLY": deterministic_codes,
                "BEAM8_HIST_ONLY": [item.node_histograms for item in items],
                "BEAM8_RANDOM_DICTIONARY": [
                    np.concatenate([code, item.node_histograms], axis=1)
                    for code, item in zip(random_codes, items)
                ],
            }
            raw_rows = {
                variant: _incidence_rows(items, typed_graphs, features, graphs, tokens)
                for variant, tokens in token_families.items()
            }
            full_dim = raw_rows["BEAM8_FULL"][0].shape[1]
            global_rows = []
            for item, graph in zip(items, graphs):
                summary = np.concatenate([item.graph_features, item.stats])
                global_rows.append(np.repeat(summary[None, :], graph.n, axis=0))
            raw_rows["GLOBAL_STATS"] = global_rows
            bag_rows = {
                variant: _bag(_normalize_nodes(_pad_rows(rows, full_dim), train))
                for variant, rows in raw_rows.items()
            }

            for model_seed in (0, 1, 2):
                reference_fold = references[model_seed]["folds"][fold]
                base_epoch = int(reference_fold["selected_epochs"]["base"])
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
                cached_base = _attach_cached(full_cache, zero_rows)
                gine_score = _cached_score(None, cached_base, test, device)["balanced_accuracy"]
                reference_gine = float(
                    reference_fold["scores"]["GINE_FROZEN"]["balanced_accuracy"]
                )
                if abs(gine_score - reference_gine) > PARITY_TOLERANCE:
                    raise RuntimeError(
                        f"GINE parity failed split={split_seed} model={model_seed} fold={fold}"
                    )

                scores = {"GINE_FROZEN": gine_score}
                selected_epochs = {"base": base_epoch}
                for variant in VARIANTS[1:]:
                    inner_data = _attach_cached(inner_cache, bag_rows[variant])
                    full_data = _attach_cached(full_cache, bag_rows[variant])
                    struct_dim = int(full_data[0].s.shape[1])
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
                        full_data,
                        train,
                        epochs=residual_epoch,
                        seed=seed,
                        struct_dim=struct_dim,
                        classes=classes,
                        device=device,
                    )
                    scores[variant] = _cached_score(head, full_data, test, device)[
                        "balanced_accuracy"
                    ]
                    selected_epochs[variant] = residual_epoch
                reference_full = float(
                    reference_fold["scores"]["BAG_RESIDUAL"]["balanced_accuracy"]
                )
                full_error = abs(scores["BEAM8_FULL"] - reference_full)
                if full_error > PARITY_TOLERANCE:
                    raise RuntimeError(
                        f"FULL parity failed split={split_seed} model={model_seed} fold={fold}: {full_error}"
                    )
                units.append(
                    {
                        "split_seed": split_seed,
                        "model_seed": model_seed,
                        "fold_index": fold,
                        "scores": scores,
                        "selected_epochs": selected_epochs,
                        "parity": {"gine_error": 0.0, "full_error": full_error},
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
        "full_vs_gine": _delta(units, "BEAM8_FULL", "GINE_FROZEN"),
        "full_vs_global_stats": _delta(units, "BEAM8_FULL", "GLOBAL_STATS"),
        "full_vs_hist_only": _delta(units, "BEAM8_FULL", "BEAM8_HIST_ONLY"),
        "full_vs_random_dictionary": _delta(
            units, "BEAM8_FULL", "BEAM8_RANDOM_DICTIONARY"
        ),
        "code_only_vs_gine": _delta(units, "BEAM8_CODE_ONLY", "GINE_FROZEN"),
    }
    checks = {
        "full_gain": paired["full_vs_gine"]["mean"] >= 0.005
        and paired["full_vs_gine"]["wins"] >= 12,
        "beats_global_stats": paired["full_vs_global_stats"]["mean"] >= 0.005
        and paired["full_vs_global_stats"]["wins"] >= 12,
        "beats_hist_only": paired["full_vs_hist_only"]["mean"] >= 0.0025
        and paired["full_vs_hist_only"]["wins"] >= 11,
        "beats_random_dictionary": paired["full_vs_random_dictionary"]["mean"] >= 0.0025
        and paired["full_vs_random_dictionary"]["wins"] >= 11,
        "global_both_splits": all(
            value > 0 for value in paired["full_vs_global_stats"]["split_means"].values()
        ),
        "global_model_majority": int(
            np.sum(np.asarray(list(paired["full_vs_global_stats"]["model_means"].values())) > 0)
        )
        >= 2,
        "parity": all(
            unit["parity"]["gine_error"] <= PARITY_TOLERANCE
            and unit["parity"]["full_error"] <= PARITY_TOLERANCE
            for unit in units
        ),
    }
    decision = (
        "BEAM8_SPECIFIC_BAG_GAIN_SUPPORTED"
        if all(checks.values())
        else "BAG_GAIN_NOT_ESTABLISHED_AS_BEAM8_SPECIFIC"
    )
    return {
        "protocol": PROTOCOL,
        "dataset": metadata | {"edge_dim": int(edge_dim)},
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
        "# Attributed Beam8 frozen BAG specificity matched controls",
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
            "| comparison | mean | W/T/L | split3/4 | model0/1/2 |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for name, row in summary["paired"].items():
        split_text = " / ".join(f"{row['split_means'][str(seed)]:+.4f}" for seed in (3, 4))
        model_text = " / ".join(f"{row['model_means'][str(seed)]:+.4f}" for seed in (0, 1, 2))
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
            "- All controls use identical frozen GINE states, residual rank and padded input dimension.",
            "- Failure means calibration may be useful but cannot be attributed specifically to Beam8.",
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

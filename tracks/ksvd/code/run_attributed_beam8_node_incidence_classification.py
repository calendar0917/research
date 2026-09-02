"""Strict orbit-safe Beam8 patch-to-node incidence fusion on Mutagenicity."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from .attributed_beam8 import (
    attributed_automorphism_orbits,
    attributed_rooted_signature,
)
from .data_tud import load_tud
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_attributed_beam8_node_incidence_feasibility import (
    node_incidence_features,
    orbit_safe_features,
)
from .run_beam8_mutagenicity_typed_feasibility import _typed_adjacency
from .run_beam8_nci1_chain_classification import (
    DEFAULT_ROOT,
    ROOT,
    _atomic_json,
    _atomic_text,
    _encode,
    _fit_dictionary,
)
from .run_luyin14_edge_aware_joint import (
    _attach_edge,
    _load_edge_pyg,
    _train_selected,
)


PROTOCOL = "tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_NODE_INCIDENCE_CLASSIFICATION_PROTOCOL_20260814.md"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/attributed_beam8_node_incidence_classification_20260814.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/ATTRIBUTED_BEAM8_NODE_INCIDENCE_CLASSIFICATION_20260814.md"
VARIANTS = (
    "GINE_ONLY",
    "RAW_FULL_TRUE",
    "INIT_FULL_TRUE",
    "FINAL_FULL_TRUE",
    "FINAL_FULL_SHUFFLED",
    "FINAL_BAG_BROADCAST",
    "FINAL_NO_RELATION",
)


def _normalize_nodes(
    values: Sequence[np.ndarray], train_indices: Sequence[int]
) -> list[np.ndarray]:
    train = np.concatenate([values[int(index)] for index in train_indices], axis=0)
    mean = train.mean(axis=0, keepdims=True)
    scale = train.std(axis=0, keepdims=True)
    scale = np.where(scale > 1e-8, scale, 1.0)
    return [(row - mean) / scale for row in values]


def _orbit_shuffle(
    values: np.ndarray,
    typed: np.ndarray,
    node_features: np.ndarray,
    *,
    seed: int,
    canonical_node_features: np.ndarray | None = None,
    canonical_node_types: np.ndarray | None = None,
) -> np.ndarray:
    canonical = np.asarray(
        node_features if canonical_node_features is None else canonical_node_features
    )
    node_types = np.asarray(
        np.argmax(canonical, axis=1)
        if canonical_node_types is None
        else canonical_node_types,
        dtype=np.int64,
    )
    orbits = attributed_automorphism_orbits(typed, node_types)
    output = values.copy()
    groups: dict[int, list[tuple[int, ...]]] = {}
    for orbit in orbits:
        groups.setdefault(len(orbit), []).append(orbit)
    rng = np.random.default_rng(seed)
    for size in sorted(groups):
        group = groups[size]
        ordered = sorted(
            group,
            key=lambda orbit: attributed_rooted_signature(
                typed, node_types, int(orbit[0])
            ),
        )
        if len(ordered) <= 1:
            continue
        permutation = rng.permutation(len(ordered))
        if np.array_equal(permutation, np.arange(len(ordered))):
            permutation = np.roll(permutation, 1)
        source_values = [values[np.asarray(orbit)].mean(axis=0) for orbit in ordered]
        for target_index, source_index in enumerate(permutation):
            output[np.asarray(ordered[target_index], dtype=np.int64)] = source_values[int(source_index)]
    return output


def _paired(folds: Sequence[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    values = np.asarray(
        [
            fold["scores"][left]["balanced_accuracy"]
            - fold["scores"][right]["balanced_accuracy"]
            for fold in folds
        ],
        dtype=np.float64,
    )
    return {
        "mean": float(values.mean()),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
        "values": values.tolist(),
    }


def _summarize(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    variants = {
        variant: {
            "balanced_accuracy_mean": float(
                np.mean([fold["scores"][variant]["balanced_accuracy"] for fold in folds])
            ),
            "balanced_accuracy_std": float(
                np.std([fold["scores"][variant]["balanced_accuracy"] for fold in folds])
            ),
            "accuracy_mean": float(
                np.mean([fold["scores"][variant]["accuracy"] for fold in folds])
            ),
            "selected_epoch_mean": float(
                np.mean([fold["scores"][variant]["selected_epoch"] for fold in folds])
            ),
        }
        for variant in VARIANTS
    }
    paired = {
        "classification_increment": _paired(folds, "FINAL_FULL_TRUE", "GINE_ONLY"),
        "binding": _paired(folds, "FINAL_FULL_TRUE", "FINAL_FULL_SHUFFLED"),
        "localization": _paired(folds, "FINAL_FULL_TRUE", "FINAL_BAG_BROADCAST"),
        "relation_metadata": _paired(folds, "FINAL_FULL_TRUE", "FINAL_NO_RELATION"),
        "ksvd_update": _paired(folds, "FINAL_FULL_TRUE", "INIT_FULL_TRUE"),
        "compression_vs_raw": _paired(folds, "FINAL_FULL_TRUE", "RAW_FULL_TRUE"),
    }
    checks = {
        "classification_increment": paired["classification_increment"]["mean"] >= 0.01
        and paired["classification_increment"]["wins"] >= 2,
        "binding": paired["binding"]["mean"] >= 0.005 and paired["binding"]["wins"] >= 2,
        "localization": paired["localization"]["mean"] >= 0.005
        and paired["localization"]["wins"] >= 2,
        "relation_metadata": paired["relation_metadata"]["mean"] >= 0.005
        and paired["relation_metadata"]["wins"] >= 2,
        "ksvd_update": paired["ksvd_update"]["mean"] > 0
        and paired["ksvd_update"]["wins"] >= 2,
    }
    if all(checks.values()):
        decision = "BEAM8_NODE_INCIDENCE_CLASSIFICATION_AND_KSVD_PASS"
    elif checks["classification_increment"] and checks["binding"] and checks["localization"]:
        decision = "BEAM8_LOCALIZED_CLASSIFICATION_PASS_PARTIAL_ATTRIBUTION"
    elif checks["classification_increment"]:
        decision = "GENERAL_BEAM8_FUSION_INCREMENT_ONLY"
    else:
        decision = "BEAM8_NODE_INCIDENCE_CLASSIFICATION_BELOW_GATE"
    return {"variants": variants, "paired": paired, "checks": checks, "decision": decision}


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    graphs, labels, features, metadata = load_tud("Mutagenicity", args.dataset_root)
    raw, raw_labels, classes, edge_dim = _load_edge_pyg("Mutagenicity", args.dataset_root)
    if not np.array_equal(labels, raw_labels):
        raise RuntimeError("the graph and PyG loaders disagree on labels")
    items = []
    typed_graphs = []
    for index, (graph, label, node_features, data) in enumerate(
        zip(graphs, labels, features, raw)
    ):
        typed = _typed_adjacency(data, edge_dim)
        typed_graphs.append(typed)
        items.append(
            prepare_attributed_beam_graph(
                index, graph, int(label), node_features, typed, edge_dim=edge_dim
            )
        )
        if (index + 1) % 500 == 0 or index + 1 == len(graphs):
            print(f"prepare {index + 1}/{len(graphs)}", flush=True)
    folds = []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        dictionary = _fit_dictionary(
            items,
            train,
            n_atoms=24,
            sparsity=3,
            iterations=5,
            max_train_patches=3000,
            seed=fold,
        )
        raw_tokens = [
            np.concatenate([item.vectors, item.node_histograms], axis=1) for item in items
        ]
        init_tokens = [
            np.concatenate(
                [_encode(item, dictionary["initial"], dictionary["mean"], 3), item.node_histograms],
                axis=1,
            )
            for item in items
        ]
        final_tokens = [
            np.concatenate(
                [_encode(item, dictionary["final"], dictionary["mean"], 3), item.node_histograms],
                axis=1,
            )
            for item in items
        ]

        def family(tokens: Sequence[np.ndarray], include_relation: bool = True) -> list[np.ndarray]:
            output = []
            for item, typed, node_features, patch_tokens, graph in zip(
                items, typed_graphs, features, tokens, graphs
            ):
                direct, _diagnostic = node_incidence_features(
                    item,
                    graph.n,
                    tokens=patch_tokens,
                    include_relation=include_relation,
                )
                safe, _orbits = orbit_safe_features(
                    direct, typed, np.asarray(node_features)
                )
                output.append(safe)
            return _normalize_nodes(output, train)

        raw_true = family(raw_tokens)
        init_true = family(init_tokens)
        final_true = family(final_tokens)
        final_no_relation = family(final_tokens, include_relation=False)
        final_shuffled = [
            _orbit_shuffle(
                values,
                typed,
                np.asarray(node_features),
                seed=314159 + fold * 100000 + index * 1009,
            )
            for index, (values, typed, node_features) in enumerate(
                zip(final_true, typed_graphs, features)
            )
        ]
        final_bag = [
            np.repeat(values.mean(axis=0, keepdims=True), len(values), axis=0)
            for values in final_true
        ]
        datasets = {
            "GINE_ONLY": _attach_edge(raw, None),
            "RAW_FULL_TRUE": _attach_edge(raw, raw_true),
            "INIT_FULL_TRUE": _attach_edge(raw, init_true),
            "FINAL_FULL_TRUE": _attach_edge(raw, final_true),
            "FINAL_FULL_SHUFFLED": _attach_edge(raw, final_shuffled),
            "FINAL_BAG_BROADCAST": _attach_edge(raw, final_bag),
            "FINAL_NO_RELATION": _attach_edge(raw, final_no_relation),
        }
        scores = {}
        for variant in VARIANTS:
            data = datasets[variant]
            scores[variant] = _train_selected(
                data,
                labels,
                train,
                test,
                input_dim=int(data[0].x.shape[1]),
                edge_dim=edge_dim,
                struct_dim=0 if variant == "GINE_ONLY" else int(data[0].s.shape[1]),
                classes=classes,
                hidden=args.hidden,
                layers=args.layers,
                dropout=args.dropout,
                lr=args.lr,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                seed=args.model_seed * 1000 + fold,
                device=torch.device(args.device),
            )
            print(
                f"fold={fold} {variant} bacc={scores[variant]['balanced_accuracy']:.4f} "
                f"epoch={scores[variant]['selected_epoch']}",
                flush=True,
            )
        folds.append(
            {
                "fold_index": fold,
                "scores": scores,
                "dictionary_reconstruction": dictionary["training"].get("recon_rel"),
            }
        )
    return {
        "protocol": PROTOCOL,
        "dataset": metadata | {"edge_dim": int(edge_dim)},
        "config": {
            "hidden": args.hidden,
            "layers": args.layers,
            "dropout": args.dropout,
            "lr": args.lr,
            "epochs": args.epochs,
            "patience": args.patience,
            "batch_size": args.batch_size,
            "model_seed": args.model_seed,
        },
        "folds": folds,
        "summary": _summarize(folds),
        "seconds": time.time() - started,
    }


def render(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Attributed Beam8 node-incidence classification",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{summary['decision']}`",
        "",
        "| variant | balanced accuracy | accuracy | epoch |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = summary["variants"][variant]
        lines.append(
            f"| {variant} | {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} | "
            f"{row['accuracy_mean']:.4f} | {row['selected_epoch_mean']:.1f} |"
        )
    lines.extend(["", "## Paired attribution", "", "| comparison | mean | W/T/L |", "|---|---:|---:|"])
    for name, row in summary["paired"].items():
        lines.append(f"| {name} | {row['mean']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} |")
    lines.extend(["", "## Frozen checks", ""])
    for name, value in summary["checks"].items():
        lines.append(f"- {name}：`{value}`；")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- 所有结构通道均为 attributed-orbit-safe node-equivariant features。",
            "- outer-test 不参与字典、normalization、epoch 或模型选择。",
            "- SHUFFLED 保留同图 node incidence row multiset；BAG 保留图级均值但删除 localization。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=80)
    parser.add_argument("--patience", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model-seed", type=int, default=0)
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

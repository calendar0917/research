"""Strict frozen-GINE residual using only graph-conditioned Beam8 BAG features."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold, train_test_split

from .data_tud import load_tud
from .run_attributed_beam8_frozen_gine_residual import (
    _base_score,
    _residual_score,
    _select_base_epoch,
    _select_head_epoch,
    _train_base,
    _train_head,
)
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


PROTOCOL = "tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_BAG_DUAL_AXIS_CONFIRMATION_PROTOCOL_20260814.md"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/attributed_beam8_frozen_bag_residual_20260814.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/ATTRIBUTED_BEAM8_FROZEN_BAG_RESIDUAL_20260814.md"
VARIANTS = ("GINE_FROZEN", "BAG_RESIDUAL")


def _paired(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    values = np.asarray(
        [
            fold["scores"]["BAG_RESIDUAL"]["balanced_accuracy"]
            - fold["scores"]["GINE_FROZEN"]["balanced_accuracy"]
            for fold in folds
        ]
    )
    return {
        "mean": float(values.mean()),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
        "values": values.tolist(),
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
    folds = []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=args.split_seed)
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        split_fold_seed = args.split_seed * 1000 + fold
        model_seed = args.model_seed * 1000 + fold
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
        patch_tokens = [
            np.concatenate(
                [_encode(item, dictionary["initial"], dictionary["mean"], 3), item.node_histograms],
                axis=1,
            )
            for item in items
        ]
        true_rows = []
        for item, typed, node_features, tokens, graph in zip(
            items, typed_graphs, features, patch_tokens, graphs
        ):
            direct, _diagnostic = node_incidence_features(
                item,
                graph.n,
                tokens=tokens,
                include_relation=False,
            )
            safe, _orbits = orbit_safe_features(direct, typed, np.asarray(node_features))
            true_rows.append(safe)
        true_rows = _normalize_nodes(true_rows, train)
        bag_rows = [
            np.repeat(values.mean(0, keepdims=True), len(values), axis=0) for values in true_rows
        ]
        bag_data = _attach_edge(raw, bag_rows)

        base_epoch = _select_base_epoch(
            base_data,
            inner_train,
            validation,
            seed=model_seed,
            input_dim=input_dim,
            edge_dim=edge_dim,
            classes=classes,
            device=device,
        )
        inner_base = _train_base(
            base_data,
            inner_train,
            epochs=base_epoch,
            seed=model_seed,
            input_dim=input_dim,
            edge_dim=edge_dim,
            classes=classes,
            device=device,
        )
        full_base = _train_base(
            base_data,
            train,
            epochs=base_epoch,
            seed=model_seed,
            input_dim=input_dim,
            edge_dim=edge_dim,
            classes=classes,
            device=device,
        )
        scores = {"GINE_FROZEN": _base_score(full_base, base_data, test, device)}
        struct_dim = int(bag_data[0].s.shape[1])
        residual_epoch = _select_head_epoch(
            inner_base,
            bag_data,
            inner_train,
            validation,
            seed=model_seed,
            struct_dim=struct_dim,
            classes=classes,
            device=device,
        )
        head = _train_head(
            full_base,
            bag_data,
            train,
            epochs=residual_epoch,
            seed=model_seed,
            struct_dim=struct_dim,
            classes=classes,
            device=device,
        )
        scores["BAG_RESIDUAL"] = _residual_score(full_base, head, bag_data, test, device)
        folds.append(
            {
                "fold_index": fold,
                "scores": scores,
                "selected_epochs": {"base": base_epoch, "BAG_RESIDUAL": residual_epoch},
            }
        )
        print(
            f"split={args.split_seed} model={args.model_seed} fold={fold} "
            f"gine={scores['GINE_FROZEN']['balanced_accuracy']:.4f} "
            f"bag={scores['BAG_RESIDUAL']['balanced_accuracy']:.4f} "
            f"delta={scores['BAG_RESIDUAL']['balanced_accuracy'] - scores['GINE_FROZEN']['balanced_accuracy']:+.4f}",
            flush=True,
        )

    variants = {
        variant: {
            "balanced_accuracy_mean": float(
                np.mean([fold["scores"][variant]["balanced_accuracy"] for fold in folds])
            ),
            "balanced_accuracy_std": float(
                np.std([fold["scores"][variant]["balanced_accuracy"] for fold in folds])
            ),
            "accuracy_mean": float(np.mean([fold["scores"][variant]["accuracy"] for fold in folds])),
        }
        for variant in VARIANTS
    }
    paired = _paired(folds)
    return {
        "protocol": PROTOCOL,
        "dataset": metadata | {"edge_dim": int(edge_dim)},
        "config": {"split_seed": args.split_seed, "model_seed": args.model_seed},
        "folds": folds,
        "summary": {"variants": variants, "paired": paired},
        "seconds": time.time() - started,
    }


def render(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    config = payload["config"]
    lines = [
        "# Attributed Beam8 graph-conditioned frozen BAG residual",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> split/model seed：`{config['split_seed']}/{config['model_seed']}`",
        "",
        "| variant | balanced accuracy | accuracy |",
        "|---|---:|---:|",
    ]
    for variant in VARIANTS:
        row = summary["variants"][variant]
        lines.append(
            f"| {variant} | {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} | {row['accuracy_mean']:.4f} |"
        )
    paired = summary["paired"]
    lines.extend(
        [
            "",
            "## Paired BAG−GINE",
            "",
            f"- mean：`{paired['mean']:+.4f}`；",
            f"- W/T/L：`{paired['wins']}/{paired['ties']}/{paired['losses']}`；",
            "",
            "## Boundary",
            "",
            "- GINE parameters and BatchNorm state remain frozen during residual training.",
            "- No localized TRUE/SHUFFLED field, relation message or KSVD update is used.",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--split-seed", type=int, required=True)
    parser.add_argument("--model-seed", type=int, required=True)
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

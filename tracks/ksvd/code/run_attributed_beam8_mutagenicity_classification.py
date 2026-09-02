"""Invariant attributed Beam8 BASE/ANCHOR classification on Mutagenicity."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .data_tud import load_tud
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_attributed_beam8_mutagenicity_anchor import attributed_anchor_item
from .run_beam8_mutagenicity_typed_classification import (
    VARIANTS,
    _features,
    _fit_score,
    _summary,
)
from .run_beam8_mutagenicity_typed_feasibility import _typed_adjacency
from .run_beam8_nci1_chain_classification import (
    DEFAULT_ROOT,
    ROOT,
    _atomic_json,
    _atomic_text,
    _fit_dictionary,
)
from .run_luyin14_edge_aware_joint import _load_edge_pyg


PROTOCOL = "tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_MUTAGENICITY_CLASSIFICATION_PROTOCOL_20260813.md"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/attributed_beam8_mutagenicity_classification_20260813.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/ATTRIBUTED_BEAM8_MUTAGENICITY_CLASSIFICATION_20260813.md"


def _prepare(args: argparse.Namespace) -> tuple[list[Any], list[Any], np.ndarray, dict[str, Any], int]:
    graphs, labels, features, metadata = load_tud("Mutagenicity", args.dataset_root)
    raw, raw_labels, _classes, edge_dim = _load_edge_pyg("Mutagenicity", args.dataset_root)
    if len(raw) != len(graphs) or not np.array_equal(np.asarray(raw_labels), labels):
        raise RuntimeError("the structural and attributed Mutagenicity loaders disagree")
    base = []
    anchor = []
    anchor_counts = []
    for index, (graph, label, node_features, data) in enumerate(
        zip(graphs, labels, features, raw)
    ):
        typed = _typed_adjacency(data, edge_dim)
        base_item = prepare_attributed_beam_graph(
            index,
            graph,
            int(label),
            node_features,
            typed,
            patch_size=8,
            overlap=2,
            retained_beam=8,
            edge_capacity_multiplier=1.5,
            seed=20260813,
            edge_dim=edge_dim,
        )
        anchor_item, count = attributed_anchor_item(
            base_item,
            graph,
            node_features,
            typed,
            edge_dim=edge_dim,
            patch_size=8,
        )
        base.append(base_item)
        anchor.append(anchor_item)
        anchor_counts.append(count)
        if (index + 1) % 500 == 0 or index + 1 == len(graphs):
            print(f"prepare {index + 1}/{len(graphs)}", flush=True)
    metadata = metadata | {
        "edge_dim": int(edge_dim),
        "mean_anchors": float(np.mean(anchor_counts)),
        "base_vector_dim": int(base[0].vectors.shape[1]),
    }
    return base, anchor, labels, metadata, edge_dim


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    base, anchor, labels, metadata, _edge_dim = _prepare(args)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    folds = []
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        dictionary = _fit_dictionary(
            anchor,
            train,
            n_atoms=24,
            sparsity=3,
            iterations=5,
            max_train_patches=3000,
            seed=fold,
        )
        matrices = _features(
            base,
            anchor,
            train,
            dictionary,
            731421 + fold,
        )
        scores = {
            variant: _fit_score(
                values[train], labels[train], values[test], labels[test]
            )
            for variant, values in matrices.items()
        }
        folds.append(
            {
                "fold_index": fold,
                "scores": scores,
                "dictionary": {
                    "atoms": dictionary["atoms"],
                    "raw_train_patches": dictionary["raw_train_patches"],
                    "used_train_patches": dictionary["used_train_patches"],
                    "recon_rel": dictionary["training"].get("recon_rel"),
                },
            }
        )
        print(
            f"fold {fold}: BASE_RAW_TRUE={scores['BASE_RAW_TRUE']['balanced_accuracy']:.4f} "
            f"ANCHOR_INIT_TRUE={scores['ANCHOR_INIT_TRUE']['balanced_accuracy']:.4f} "
            f"ANCHOR_FINAL_TRUE={scores['ANCHOR_FINAL_TRUE']['balanced_accuracy']:.4f}",
            flush=True,
        )
    return {
        "protocol": PROTOCOL,
        "dataset": metadata,
        "folds": folds,
        "summary": _summary(folds),
        "seconds": time.time() - started,
    }


def render(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Attributed Beam8/Mutagenicity classification",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{summary['decision']}`",
        "",
        "| variant | balanced accuracy | accuracy |",
        "|---|---:|---:|",
    ]
    for variant in VARIANTS:
        row = summary["variants"][variant]
        lines.append(
            f"| {variant} | {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} | {row['accuracy_mean']:.4f} |"
        )
    lines.extend(
        ["", "## Paired attribution", "", "| comparison | mean | W/T/L |", "|---|---:|---:|"]
    )
    for name, row in summary["paired"].items():
        lines.append(
            f"| {name} | {row['mean']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} |"
        )
    lines.extend(["", "## Frozen checks", ""])
    for name, value in summary["checks"].items():
        lines.append(f"- {name}：`{value}`；")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- 本结果只使用通过 relabel-invariance gate 的 atom-colored/bond-typed canonical representation。",
            "- BASE/ANCHOR 共用 outer-train ANCHOR dictionary 与 normalization。",
            "- anchor 为 relation-isolated 独立 segment；其收益与连续 Beam8 relation、KSVD updates 分开解释。",
            "- 旧 binary-order typed classification 是 invalidated diagnostic，不参与证据汇总。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
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

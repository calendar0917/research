"""Post-Stage-A attribution controls for the stronger Beam8 INIT incidence."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from .data_tud import load_tud
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_attributed_beam8_node_incidence_classification import (
    DEFAULT_JSON as STAGE_A_JSON,
    _normalize_nodes,
    _orbit_shuffle,
)
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
from .run_luyin14_edge_aware_joint import _attach_edge, _load_edge_pyg, _train_selected


PROTOCOL = "tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_INIT_INCIDENCE_FOLLOWUP_PROTOCOL_20260814.md"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/attributed_beam8_init_incidence_followup_20260814.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/ATTRIBUTED_BEAM8_INIT_INCIDENCE_FOLLOWUP_20260814.md"
VARIANTS = (
    "GINE_ONLY",
    "INIT_FULL_TRUE",
    "INIT_FULL_SHUFFLED",
    "INIT_BAG_BROADCAST",
    "INIT_NO_RELATION",
)


def _paired(folds: Sequence[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    values = np.asarray(
        [fold["scores"][left]["balanced_accuracy"] - fold["scores"][right]["balanced_accuracy"] for fold in folds]
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
    frozen = json.loads(args.stage_a_json.read_text(encoding="utf-8"))
    graphs, labels, features, metadata = load_tud("Mutagenicity", args.dataset_root)
    raw, raw_labels, classes, edge_dim = _load_edge_pyg("Mutagenicity", args.dataset_root)
    if not np.array_equal(labels, raw_labels):
        raise RuntimeError("the graph and PyG loaders disagree on labels")
    items, typed_graphs = [], []
    for index, (graph, label, node_features, data) in enumerate(zip(graphs, labels, features, raw)):
        typed = _typed_adjacency(data, edge_dim)
        typed_graphs.append(typed)
        items.append(prepare_attributed_beam_graph(index, graph, int(label), node_features, typed, edge_dim=edge_dim))
        if (index + 1) % 500 == 0 or index + 1 == len(graphs):
            print(f"prepare {index + 1}/{len(graphs)}", flush=True)
    folds = []
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
        dictionary = _fit_dictionary(items, train, n_atoms=24, sparsity=3, iterations=5, max_train_patches=3000, seed=fold)
        init_patch_tokens = [
            np.concatenate([_encode(item, dictionary["initial"], dictionary["mean"], 3), item.node_histograms], axis=1)
            for item in items
        ]

        def family(include_relation: bool) -> list[np.ndarray]:
            rows = []
            for item, typed, node_features, patch_tokens, graph in zip(items, typed_graphs, features, init_patch_tokens, graphs):
                direct, _diagnostic = node_incidence_features(item, graph.n, tokens=patch_tokens, include_relation=include_relation)
                safe, _orbits = orbit_safe_features(direct, typed, np.asarray(node_features))
                rows.append(safe)
            return _normalize_nodes(rows, train)

        init_true = family(True)
        init_no_relation = family(False)
        init_shuffled = [
            _orbit_shuffle(values, typed, np.asarray(node_features), seed=271828 + fold * 100000 + index * 1009)
            for index, (values, typed, node_features) in enumerate(zip(init_true, typed_graphs, features))
        ]
        init_bag = [np.repeat(values.mean(0, keepdims=True), len(values), axis=0) for values in init_true]
        datasets = {
            "INIT_FULL_SHUFFLED": _attach_edge(raw, init_shuffled),
            "INIT_BAG_BROADCAST": _attach_edge(raw, init_bag),
            "INIT_NO_RELATION": _attach_edge(raw, init_no_relation),
        }
        frozen_fold = frozen["folds"][fold]["scores"]
        scores = {
            "GINE_ONLY": frozen_fold["GINE_ONLY"],
            "INIT_FULL_TRUE": frozen_fold["INIT_FULL_TRUE"],
        }
        for variant, data in datasets.items():
            scores[variant] = _train_selected(
                data, labels, train, test,
                input_dim=int(data[0].x.shape[1]), edge_dim=edge_dim,
                struct_dim=int(data[0].s.shape[1]), classes=classes,
                hidden=64, layers=3, dropout=0.5, lr=0.01,
                epochs=80, patience=20, batch_size=64,
                seed=fold, device=torch.device(args.device),
            )
            print(f"fold={fold} {variant} bacc={scores[variant]['balanced_accuracy']:.4f} epoch={scores[variant]['selected_epoch']}", flush=True)
        folds.append({"fold_index": fold, "scores": scores})
    variants = {
        variant: {
            "balanced_accuracy_mean": float(np.mean([fold["scores"][variant]["balanced_accuracy"] for fold in folds])),
            "balanced_accuracy_std": float(np.std([fold["scores"][variant]["balanced_accuracy"] for fold in folds])),
        }
        for variant in VARIANTS
    }
    paired = {
        "classification_increment": _paired(folds, "INIT_FULL_TRUE", "GINE_ONLY"),
        "binding": _paired(folds, "INIT_FULL_TRUE", "INIT_FULL_SHUFFLED"),
        "localization": _paired(folds, "INIT_FULL_TRUE", "INIT_BAG_BROADCAST"),
        "relation_metadata": _paired(folds, "INIT_FULL_TRUE", "INIT_NO_RELATION"),
    }
    checks = {
        key: row["mean"] >= 0.005 and row["wins"] >= 2 for key, row in paired.items()
    }
    if checks["classification_increment"] and checks["binding"] and checks["localization"]:
        decision = "BEAM8_INIT_LOCALIZED_CLASSIFICATION_SIGNAL"
    elif checks["binding"]:
        decision = "BEAM8_INIT_BINDING_ONLY"
    else:
        decision = "BEAM8_INIT_INCIDENCE_BELOW_GATE"
    return {
        "protocol": PROTOCOL,
        "stage_a_json": str(args.stage_a_json),
        "dataset": metadata | {"edge_dim": int(edge_dim)},
        "folds": folds,
        "summary": {"variants": variants, "paired": paired, "checks": checks, "decision": decision},
        "seconds": time.time() - started,
    }


def render(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Attributed Beam8 INIT-incidence follow-up", "",
        f"> 协议：`{payload['protocol']}`  ", f"> 判定：`{summary['decision']}`", "",
        "| variant | balanced accuracy |", "|---|---:|",
    ]
    for variant in VARIANTS:
        row = summary["variants"][variant]
        lines.append(f"| {variant} | {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} |")
    lines.extend(["", "## Paired attribution", "", "| comparison | mean | W/T/L |", "|---|---:|---:|"])
    for name, row in summary["paired"].items():
        lines.append(f"| {name} | {row['mean']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} |")
    lines.extend(["", "## Diagnostic checks", ""])
    for name, value in summary["checks"].items():
        lines.append(f"- {name}：`{value}`；")
    lines.extend(["", "## Boundary", "", "- 本轮是看到 FINAL Stage A 后的机制诊断，不是独立确认实验。", "- GINE_ONLY 与 INIT_FULL_TRUE 读取冻结 Stage-A folds；新增控制按相同 split/checkpoint/seeds 训练。"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage-a-json", type=Path, default=STAGE_A_JSON)
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

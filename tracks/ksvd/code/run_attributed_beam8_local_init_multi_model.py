"""Multi-model-seed validation of localized Beam8 INIT node incidence."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from .data_tud import load_tud
from .run_attributed_beam8_content_relation_stage_b1 import DEFAULT_JSON as STAGE_B1_JSON
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_attributed_beam8_node_incidence_classification import _normalize_nodes
from .run_attributed_beam8_node_incidence_feasibility import node_incidence_features, orbit_safe_features
from .run_beam8_mutagenicity_typed_feasibility import _typed_adjacency
from .run_beam8_nci1_chain_classification import DEFAULT_ROOT, ROOT, _atomic_json, _atomic_text, _encode, _fit_dictionary
from .run_luyin14_edge_aware_joint import _attach_edge, _load_edge_pyg, _train_selected


PROTOCOL = "tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_LOCAL_INIT_MULTI_MODEL_PROTOCOL_20260814.md"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/attributed_beam8_local_init_multi_model_20260814.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/ATTRIBUTED_BEAM8_LOCAL_INIT_MULTI_MODEL_20260814.md"


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    frozen = json.loads(args.stage_b1_json.read_text(encoding="utf-8"))
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
        patch_tokens = [
            np.concatenate([_encode(item, dictionary["initial"], dictionary["mean"], 3), item.node_histograms], axis=1)
            for item in items
        ]
        local_rows = []
        for item, typed, node_features, tokens, graph in zip(items, typed_graphs, features, patch_tokens, graphs):
            direct, _diagnostic = node_incidence_features(item, graph.n, tokens=tokens, include_relation=False)
            safe, _orbits = orbit_safe_features(direct, typed, np.asarray(node_features))
            local_rows.append(safe)
        local_rows = _normalize_nodes(local_rows, train)
        datasets = {"GINE_ONLY": _attach_edge(raw, None), "INIT_LOCAL_CONTENT": _attach_edge(raw, local_rows)}
        for model_seed in (0, 1, 2):
            if model_seed == 0:
                scores = {
                    variant: frozen["folds"][fold]["scores"][variant]
                    for variant in ("GINE_ONLY", "INIT_LOCAL_CONTENT")
                }
            else:
                scores = {}
                for variant, data in datasets.items():
                    scores[variant] = _train_selected(
                        data, labels, train, test,
                        input_dim=int(data[0].x.shape[1]), edge_dim=edge_dim,
                        struct_dim=0 if variant == "GINE_ONLY" else int(data[0].s.shape[1]),
                        classes=classes, hidden=64, layers=3, dropout=0.5, lr=0.01,
                        epochs=80, patience=20, batch_size=64,
                        seed=model_seed * 1000 + fold, device=torch.device(args.device),
                    )
                    print(f"model_seed={model_seed} fold={fold} {variant} bacc={scores[variant]['balanced_accuracy']:.4f} epoch={scores[variant]['selected_epoch']}", flush=True)
            folds.append({"model_seed": model_seed, "fold_index": fold, "scores": scores})
    deltas = np.asarray([
        row["scores"]["INIT_LOCAL_CONTENT"]["balanced_accuracy"] - row["scores"]["GINE_ONLY"]["balanced_accuracy"]
        for row in folds
    ])
    seed_deltas = {
        seed: float(np.mean([
            row["scores"]["INIT_LOCAL_CONTENT"]["balanced_accuracy"] - row["scores"]["GINE_ONLY"]["balanced_accuracy"]
            for row in folds if row["model_seed"] == seed
        ]))
        for seed in (0, 1, 2)
    }
    variants = {
        variant: {
            "balanced_accuracy_mean": float(np.mean([row["scores"][variant]["balanced_accuracy"] for row in folds])),
            "balanced_accuracy_std": float(np.std([row["scores"][variant]["balanced_accuracy"] for row in folds])),
        }
        for variant in ("GINE_ONLY", "INIT_LOCAL_CONTENT")
    }
    paired = {
        "mean": float(deltas.mean()),
        "wins": int(np.sum(deltas > 1e-12)),
        "ties": int(np.sum(np.abs(deltas) <= 1e-12)),
        "losses": int(np.sum(deltas < -1e-12)),
        "values": deltas.tolist(),
        "seed_means": {str(seed): value for seed, value in seed_deltas.items()},
    }
    checks = {
        "mean_increment": paired["mean"] >= 0.01,
        "unit_majority": paired["wins"] >= 6,
        "seed_majority": int(np.sum(np.asarray(list(seed_deltas.values())) > 0)) >= 2,
        "worst_seed": min(seed_deltas.values()) >= -0.005,
    }
    advance = all(checks.values())
    return {
        "protocol": PROTOCOL,
        "stage_b1_json": str(args.stage_b1_json),
        "dataset": metadata | {"edge_dim": int(edge_dim)},
        "fold_model_units": folds,
        "summary": {
            "variants": variants,
            "paired": paired,
            "checks": checks,
            "decision": "LOCALIZED_INIT_ADVANCES_TO_MULTI_SPLIT" if advance else "LOCALIZED_INIT_NOT_STABLE_ACROSS_MODEL_SEEDS",
        },
        "seconds": time.time() - started,
    }


def render(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Attributed Beam8 localized INIT multi-model-seed validation", "",
        f"> 协议：`{payload['protocol']}`  ", f"> 判定：`{summary['decision']}`", "",
        "| variant | balanced accuracy over 9 units |", "|---|---:|",
    ]
    for variant, row in summary["variants"].items():
        lines.append(f"| {variant} | {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} |")
    pair = summary["paired"]
    lines.extend(["", f"- LOCAL−GINE：`{pair['mean']:+.4f}`，W/T/L `{pair['wins']}/{pair['ties']}/{pair['losses']}`；", "", "## Model-seed means", ""])
    for seed, value in pair["seed_means"].items():
        lines.append(f"- seed {seed}：`{value:+.4f}`；")
    lines.extend(["", "## Frozen checks", ""])
    for name, value in summary["checks"].items():
        lines.append(f"- {name}：`{value}`；")
    lines.extend(["", "## Boundary", "", "- split seed 固定为 0；本轮只验证 neural initialization，不是跨数据划分确认。", "- seed0 读取冻结 Stage-B1 folds；seed1/2 使用完全相同的数据、dictionary 和 checkpoint protocol。"])
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--stage-b1-json", type=Path, default=STAGE_B1_JSON)
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

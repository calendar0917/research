"""Content-aware Beam8 INIT relation messages fused into an edge-aware GINE."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from .data_tud import load_tud
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_attributed_beam8_node_incidence_classification import _normalize_nodes
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
    _relation_matrices,
    _shuffle_permutation,
)
from .run_luyin14_edge_aware_joint import _attach_edge, _load_edge_pyg, _train_selected


PROTOCOL = "tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_CONTENT_RELATION_STAGE_B1_PROTOCOL_20260814.md"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/attributed_beam8_content_relation_stage_b1_20260814.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/ATTRIBUTED_BEAM8_CONTENT_RELATION_STAGE_B1_20260814.md"
VARIANTS = (
    "GINE_ONLY",
    "INIT_LOCAL_CONTENT",
    "INIT_RELATION_TRUE",
    "INIT_RELATION_SHUFFLED",
)


def _message(weights: np.ndarray, values: np.ndarray) -> np.ndarray:
    denominator = weights.sum(axis=1, keepdims=True)
    normalized = np.divide(weights, denominator, out=np.zeros_like(weights), where=denominator > 1e-12)
    return normalized @ values


def relation_patch_tokens(
    item: Any,
    codes: np.ndarray,
    *,
    message_codes: np.ndarray | None = None,
) -> np.ndarray:
    local_codes = np.asarray(codes, dtype=np.float64)
    sources = local_codes if message_codes is None else np.asarray(message_codes, dtype=np.float64)
    previous, following, overlap, _slot = _relation_matrices(item)
    chain = previous + following
    nonchain_overlap = overlap.copy()
    nonchain_overlap[chain > 1e-12] = 0.0
    channels = []
    for weights in (chain, nonchain_overlap):
        message = _message(weights, sources)
        active = (weights.sum(axis=1, keepdims=True) > 1e-12).astype(np.float64)
        channels.extend(
            [
                message,
                local_codes * message,
                np.abs(local_codes - message) * active,
            ]
        )
    return np.concatenate([local_codes, item.node_histograms, *channels], axis=1)


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
        codes = [_encode(item, dictionary["initial"], dictionary["mean"], 3) for item in items]
        local_patch_tokens = [np.concatenate([code, item.node_histograms], axis=1) for item, code in zip(items, codes)]
        true_patch_tokens = [relation_patch_tokens(item, code) for item, code in zip(items, codes)]
        shuffled_patch_tokens = []
        for item, code in zip(items, codes):
            permutation = _shuffle_permutation(len(code), seed=731421 + fold * 100000 + item.index * 1009)
            shuffled_patch_tokens.append(relation_patch_tokens(item, code, message_codes=code[permutation]))

        def incidence(patch_tokens: Sequence[np.ndarray]) -> list[np.ndarray]:
            rows = []
            for item, typed, node_features, tokens, graph in zip(items, typed_graphs, features, patch_tokens, graphs):
                direct, _diagnostic = node_incidence_features(item, graph.n, tokens=tokens, include_relation=False)
                safe, _orbits = orbit_safe_features(direct, typed, np.asarray(node_features))
                rows.append(safe)
            return _normalize_nodes(rows, train)

        local = incidence(local_patch_tokens)
        relation_true = incidence(true_patch_tokens)
        relation_shuffled = incidence(shuffled_patch_tokens)
        datasets = {
            "GINE_ONLY": _attach_edge(raw, None),
            "INIT_LOCAL_CONTENT": _attach_edge(raw, local),
            "INIT_RELATION_TRUE": _attach_edge(raw, relation_true),
            "INIT_RELATION_SHUFFLED": _attach_edge(raw, relation_shuffled),
        }
        scores = {}
        for variant in VARIANTS:
            data = datasets[variant]
            scores[variant] = _train_selected(
                data, labels, train, test,
                input_dim=int(data[0].x.shape[1]), edge_dim=edge_dim,
                struct_dim=0 if variant == "GINE_ONLY" else int(data[0].s.shape[1]),
                classes=classes, hidden=64, layers=3, dropout=0.5, lr=0.01,
                epochs=80, patience=20, batch_size=64,
                seed=fold, device=torch.device(args.device),
            )
            print(f"fold={fold} {variant} bacc={scores[variant]['balanced_accuracy']:.4f} epoch={scores[variant]['selected_epoch']}", flush=True)
        folds.append({"fold_index": fold, "scores": scores})
    variants = {
        variant: {
            "balanced_accuracy_mean": float(np.mean([fold["scores"][variant]["balanced_accuracy"] for fold in folds])),
            "balanced_accuracy_std": float(np.std([fold["scores"][variant]["balanced_accuracy"] for fold in folds])),
            "accuracy_mean": float(np.mean([fold["scores"][variant]["accuracy"] for fold in folds])),
        }
        for variant in VARIANTS
    }
    paired = {
        "classification": _paired(folds, "INIT_RELATION_TRUE", "GINE_ONLY"),
        "relation_increment": _paired(folds, "INIT_RELATION_TRUE", "INIT_LOCAL_CONTENT"),
        "relation_binding": _paired(folds, "INIT_RELATION_TRUE", "INIT_RELATION_SHUFFLED"),
        "local_classification": _paired(folds, "INIT_LOCAL_CONTENT", "GINE_ONLY"),
    }
    checks = {
        "classification": paired["classification"]["mean"] >= 0.01 and paired["classification"]["wins"] >= 2,
        "relation_increment": paired["relation_increment"]["mean"] >= 0.005 and paired["relation_increment"]["wins"] >= 2,
        "relation_binding": paired["relation_binding"]["mean"] >= 0.005 and paired["relation_binding"]["wins"] >= 2,
    }
    if all(checks.values()):
        decision = "CONTENT_RELATION_ADVANCES_TO_MULTI_MODEL_SEED"
    elif checks["relation_binding"]:
        decision = "CONTENT_RELATION_BINDING_ONLY"
    elif paired["local_classification"]["mean"] >= 0.01 and paired["local_classification"]["wins"] >= 2:
        decision = "LOCALIZED_INIT_ONLY_STOP_PATCH_RELATION"
    else:
        decision = "CONTENT_RELATION_STAGE_B1_BELOW_GATE"
    return {
        "protocol": PROTOCOL,
        "dataset": metadata | {"edge_dim": int(edge_dim)},
        "folds": folds,
        "summary": {"variants": variants, "paired": paired, "checks": checks, "decision": decision},
        "seconds": time.time() - started,
    }


def render(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Attributed Beam8 content-aware relation Stage B1", "",
        f"> 协议：`{payload['protocol']}`  ", f"> 判定：`{summary['decision']}`", "",
        "| variant | balanced accuracy | accuracy |", "|---|---:|---:|",
    ]
    for variant in VARIANTS:
        row = summary["variants"][variant]
        lines.append(f"| {variant} | {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} | {row['accuracy_mean']:.4f} |")
    lines.extend(["", "## Paired attribution", "", "| comparison | mean | W/T/L |", "|---|---:|---:|"])
    for name, row in summary["paired"].items():
        lines.append(f"| {name} | {row['mean']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} |")
    lines.extend(["", "## Frozen checks", ""])
    for name, value in summary["checks"].items():
        lines.append(f"- {name}：`{value}`；")
    lines.extend(["", "## Boundary", "", "- 本轮只使用 INIT vocabulary；没有普通 KSVD updates。", "- SHUFFLED 只打乱邻居 message codes，local patch content、patch graph 与 node incidence 保持不变。", "- 通过 Stage B1 才允许扩展 model seeds 或学习 atom gate。"])
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

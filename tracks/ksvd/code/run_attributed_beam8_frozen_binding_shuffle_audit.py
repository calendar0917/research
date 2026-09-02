"""Repeated-SHUFFLED audit for frozen-GINE Beam8 residual binding."""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from .data_tud import load_tud
from .run_attributed_beam8_frozen_gine_residual import ResidualHead, _forward_base, _train_base
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_attributed_beam8_node_incidence_classification import _normalize_nodes, _orbit_shuffle
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


PROTOCOL = "tracks/ksvd/docs/KSVD_ATTRIBUTED_BEAM8_FROZEN_BINDING_SHUFFLE_AUDIT_PROTOCOL_20260814.md"
RESULT_DIR = ROOT / "tracks/ksvd/results/luyin14"
REFERENCE_PATHS = (
    RESULT_DIR / "attributed_beam8_frozen_gine_residual_20260814.json",
    RESULT_DIR / "attributed_beam8_frozen_gine_residual_split_seed1_20260814.json",
    RESULT_DIR / "attributed_beam8_frozen_gine_residual_split_seed2_20260814.json",
)
DEFAULT_JSON = RESULT_DIR / "attributed_beam8_frozen_binding_shuffle_audit_20260814.json"
DEFAULT_REPORT = RESULT_DIR / "ATTRIBUTED_BEAM8_FROZEN_BINDING_SHUFFLE_AUDIT_20260814.md"
SHUFFLE_STRIDE = 32452843
PARITY_TOLERANCE = 1e-12


@torch.no_grad()
def _cache_base(model: torch.nn.Module, data: Sequence[Data], device: torch.device) -> list[dict[str, torch.Tensor]]:
    model.eval()
    cached: list[dict[str, torch.Tensor]] = []
    for batch in DataLoader(list(data), batch_size=128, shuffle=False):
        batch = batch.to(device)
        logits, states = _forward_base(model, batch)
        ptr = batch.ptr.detach().cpu().numpy()
        states = states.detach().cpu()
        logits = logits.detach().cpu()
        labels = batch.y.view(-1).detach().cpu()
        for index in range(int(batch.num_graphs)):
            cached.append(
                {
                    "states": states[int(ptr[index]) : int(ptr[index + 1])].clone(),
                    "logits": logits[index : index + 1].clone(),
                    "label": labels[index : index + 1].clone(),
                }
            )
    if len(cached) != len(data):
        raise RuntimeError(f"cached {len(cached)} graphs, expected {len(data)}")
    return cached


def _attach_cached(cached: Sequence[dict[str, torch.Tensor]], rows: Sequence[np.ndarray]) -> list[Data]:
    output = []
    for base, struct in zip(cached, rows):
        if len(base["states"]) != len(struct):
            raise ValueError("cached node states and structural rows disagree")
        output.append(
            Data(
                x=base["states"].clone(),
                s=torch.as_tensor(np.asarray(struct), dtype=torch.float32),
                base_logits=base["logits"].clone(),
                y=base["label"].clone(),
            )
        )
    return output


@torch.no_grad()
def _cached_score(
    head: ResidualHead | None,
    data: Sequence[Data],
    indices: Sequence[int],
    device: torch.device,
) -> dict[str, float]:
    if head is not None:
        head.eval()
    prediction, labels = [], []
    for batch in DataLoader([data[int(i)] for i in indices], batch_size=128, shuffle=False):
        batch = batch.to(device)
        logits = batch.base_logits
        if head is not None:
            logits = logits + head(batch.x, batch.s, batch.batch)
        prediction.extend(logits.argmax(1).cpu().numpy().tolist())
        labels.extend(batch.y.view(-1).cpu().numpy().tolist())
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "accuracy": float(accuracy_score(labels, prediction)),
    }


def _train_cached_head(
    data: Sequence[Data],
    indices: Sequence[int],
    *,
    epochs: int,
    seed: int,
    struct_dim: int,
    classes: int,
    device: torch.device,
) -> ResidualHead:
    torch.manual_seed(seed + 70000)
    head = ResidualHead(64, struct_dim, classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.003, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 23)
    loader = DataLoader(
        [data[int(i)] for i in indices],
        batch_size=64,
        shuffle=True,
        generator=generator,
    )
    for _epoch in range(epochs):
        head.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = batch.base_logits + head(batch.x, batch.s, batch.batch)
            loss = F.cross_entropy(output, batch.y.view(-1))
            loss.backward()
            optimizer.step()
    return head


def _select_cached_head_epoch(
    data: Sequence[Data],
    train: np.ndarray,
    validation: np.ndarray,
    *,
    seed: int,
    struct_dim: int,
    classes: int,
    device: torch.device,
) -> int:
    torch.manual_seed(seed + 70000)
    head = ResidualHead(64, struct_dim, classes).to(device)
    optimizer = torch.optim.Adam(head.parameters(), lr=0.003, weight_decay=1e-4)
    generator = torch.Generator().manual_seed(seed + 23)
    loader = DataLoader(
        [data[int(i)] for i in train],
        batch_size=64,
        shuffle=True,
        generator=generator,
    )
    best_epoch, best_score, stale = 1, -1.0, 0
    for epoch in range(1, 61):
        head.train()
        for batch in loader:
            batch = batch.to(device)
            optimizer.zero_grad()
            output = batch.base_logits + head(batch.x, batch.s, batch.batch)
            loss = F.cross_entropy(output, batch.y.view(-1))
            loss.backward()
            optimizer.step()
        score = _cached_score(head, data, validation, device)["balanced_accuracy"]
        if score > best_score + 1e-12:
            best_epoch, best_score, stale = epoch, score, 0
        else:
            stale += 1
        if stale >= 15:
            break
    return best_epoch


def _reference_payloads(
    paths: Sequence[Path], expected_split_seeds: Sequence[int]
) -> dict[int, dict[str, Any]]:
    payloads: dict[int, dict[str, Any]] = {}
    for fallback_seed, path in enumerate(paths):
        payload = json.loads(path.read_text(encoding="utf-8"))
        config = payload.get("config") or {}
        split_seed = int(config.get("split_seed", fallback_seed))
        model_seed = int(config.get("model_seed", 0))
        if model_seed != 0:
            raise ValueError(f"expected model_seed=0 in {path}, got {model_seed}")
        payloads[split_seed] = payload
    expected = sorted(int(seed) for seed in expected_split_seeds)
    if sorted(payloads) != expected:
        raise ValueError(f"expected reference split seeds {expected}, got {sorted(payloads)}")
    return payloads


def _summarize(units: Sequence[dict[str, Any]], repeats: int) -> dict[str, Any]:
    comparisons = [repeat for unit in units for repeat in unit["repeats"]]
    deltas = np.asarray([row["delta_true_minus_shuffled"] for row in comparisons])
    split_seeds = sorted({int(unit["split_seed"]) for unit in units})
    fold_means = {
        f"{unit['split_seed']}:{unit['fold_index']}": float(
            np.mean([row["delta_true_minus_shuffled"] for row in unit["repeats"]])
        )
        for unit in units
    }
    split_means = {
        str(split_seed): float(
            np.mean(
                [
                    row["delta_true_minus_shuffled"]
                    for unit in units
                    if unit["split_seed"] == split_seed
                    for row in unit["repeats"]
                ]
            )
        )
        for split_seed in split_seeds
    }
    required_wins = math.ceil(0.60 * len(comparisons))
    required_positive_folds = math.ceil((2.0 / 3.0) * len(units))
    required_positive_splits = math.ceil((2.0 / 3.0) * len(split_seeds))
    checks = {
        "repeat0_parity": all(unit["parity"]["passed"] for unit in units),
        "mean_binding": float(deltas.mean()) >= 0.005,
        "comparison_win_rate": int(np.sum(deltas > 1e-12)) >= required_wins,
        "fold_majority": int(np.sum(np.asarray(list(fold_means.values())) > 0))
        >= required_positive_folds,
        "split_majority": int(np.sum(np.asarray(list(split_means.values())) > 0))
        >= required_positive_splits,
        "worst_split": min(split_means.values()) >= -0.005,
    }
    decision = (
        "EXACT_BINDING_SUPPORTED_AGAINST_SHUFFLE_DISTRIBUTION"
        if all(checks.values())
        else "EXACT_BINDING_NOT_SUPPORTED_AGAINST_SHUFFLE_DISTRIBUTION"
    )
    return {
        "comparisons": len(comparisons),
        "repeats_per_fold": repeats,
        "mean_delta": float(deltas.mean()),
        "std_delta": float(deltas.std()),
        "wins": int(np.sum(deltas > 1e-12)),
        "ties": int(np.sum(np.abs(deltas) <= 1e-12)),
        "losses": int(np.sum(deltas < -1e-12)),
        "required_wins": required_wins,
        "required_positive_folds": required_positive_folds,
        "required_positive_splits": required_positive_splits,
        "fold_means": fold_means,
        "split_means": split_means,
        "checks": checks,
        "decision": decision,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    device = torch.device(args.device)
    references = _reference_payloads(args.references, args.split_seeds)
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
    for split_seed in args.split_seeds:
        reference = references[int(split_seed)]
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=int(split_seed))
        for fold, (train, test) in enumerate(splitter.split(np.zeros(len(labels)), labels)):
            split_fold_seed = int(split_seed) * 1000 + fold
            model_seed = fold
            inner_train, validation = train_test_split(
                train,
                test_size=0.2,
                stratify=labels[train],
                random_state=1729 + split_fold_seed,
            )
            reference_fold = reference["folds"][fold]
            base_epoch = int(reference_fold["selected_epochs"]["base"])
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
            inner_cache = _cache_base(inner_base, base_data, device)
            full_cache = _cache_base(full_base, base_data, device)

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

            true_score = float(reference_fold["scores"]["TRUE_RESIDUAL"]["balanced_accuracy"])
            reference_gine = float(reference_fold["scores"]["GINE_FROZEN"]["balanced_accuracy"])
            reference_shuffled = float(
                reference_fold["scores"]["SHUFFLED_RESIDUAL"]["balanced_accuracy"]
            )
            cached_base_data = _attach_cached(full_cache, [np.zeros((len(row), 1), dtype=np.float32) for row in true_rows])
            cached_gine = _cached_score(None, cached_base_data, test, device)["balanced_accuracy"]
            parity = {
                "gine_absolute_error": abs(cached_gine - reference_gine),
                "repeat0_shuffled_absolute_error": None,
                "passed": False,
            }
            if parity["gine_absolute_error"] > PARITY_TOLERANCE:
                raise RuntimeError(
                    f"GINE cache parity failed for split={split_seed}, fold={fold}: {parity}"
                )
            repeats = []
            for repeat in range(args.repeats):
                shuffled_rows = [
                    _orbit_shuffle(
                        values,
                        typed,
                        np.asarray(node_features),
                        seed=(
                            161803
                            + split_fold_seed * 100000
                            + index * 1009
                            + repeat * SHUFFLE_STRIDE
                        ),
                    )
                    for index, (values, typed, node_features) in enumerate(
                        zip(true_rows, typed_graphs, features)
                    )
                ]
                inner_data = _attach_cached(inner_cache, shuffled_rows)
                full_data = _attach_cached(full_cache, shuffled_rows)
                struct_dim = int(full_data[0].s.shape[1])
                residual_epoch = _select_cached_head_epoch(
                    inner_data,
                    inner_train,
                    validation,
                    seed=model_seed,
                    struct_dim=struct_dim,
                    classes=classes,
                    device=device,
                )
                head = _train_cached_head(
                    full_data,
                    train,
                    epochs=residual_epoch,
                    seed=model_seed,
                    struct_dim=struct_dim,
                    classes=classes,
                    device=device,
                )
                score = _cached_score(head, full_data, test, device)
                repeats.append(
                    {
                        "repeat": repeat,
                        "selected_epoch": residual_epoch,
                        "score": score,
                        "delta_true_minus_shuffled": true_score - score["balanced_accuracy"],
                    }
                )
                if repeat == 0:
                    parity["repeat0_shuffled_absolute_error"] = abs(
                        score["balanced_accuracy"] - reference_shuffled
                    )
                    parity["passed"] = bool(
                        parity["repeat0_shuffled_absolute_error"] <= PARITY_TOLERANCE
                    )
                    if not parity["passed"]:
                        raise RuntimeError(
                            f"repeat0 parity failed for split={split_seed}, fold={fold}: {parity}"
                        )
                print(
                    f"split={split_seed} fold={fold} repeat={repeat} "
                    f"shuffled={score['balanced_accuracy']:.4f} "
                    f"true_minus_shuffled={true_score - score['balanced_accuracy']:+.4f}",
                    flush=True,
                )
            units.append(
                {
                    "split_seed": int(split_seed),
                    "fold_index": fold,
                    "base_epoch": base_epoch,
                    "true_balanced_accuracy": true_score,
                    "reference_gine_balanced_accuracy": reference_gine,
                    "reference_shuffled_balanced_accuracy": reference_shuffled,
                    "parity": parity,
                    "repeats": repeats,
                }
            )
    summary = _summarize(units, args.repeats)
    return {
        "protocol": PROTOCOL,
        "dataset": metadata | {"edge_dim": int(edge_dim)},
        "config": {
            "model_seed": 0,
            "split_seeds": [int(seed) for seed in args.split_seeds],
            "repeats": args.repeats,
            "shuffle_stride": SHUFFLE_STRIDE,
        },
        "references": [str(path) for path in args.references],
        "units": units,
        "summary": summary,
        "seconds": time.time() - started,
    }


def render(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Attributed Beam8 frozen binding repeated-SHUFFLED audit",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{summary['decision']}`",
        "",
        "## Aggregate",
        "",
        "| metric | value |",
        "|---|---:|",
        f"| comparisons | {summary['comparisons']} |",
        f"| TRUE−SHUFFLED mean | {summary['mean_delta']:+.4f} |",
        f"| delta std | {summary['std_delta']:.4f} |",
        f"| W/T/L | {summary['wins']}/{summary['ties']}/{summary['losses']} |",
        f"| split means | "
        + " / ".join(
            f"s{seed}:{summary['split_means'][str(seed)]:+.4f}"
            for seed in sorted(int(value) for value in summary["split_means"])
        )
        + " |",
        "",
        "## Fold means over shuffle realizations",
        "",
        "| split:fold | TRUE−mean(SHUFFLED) |",
        "|---|---:|",
    ]
    for key, value in summary["fold_means"].items():
        lines.append(f"| {key} | {value:+.4f} |")
    lines.extend(["", "## Frozen checks", ""])
    for name, value in summary["checks"].items():
        lines.append(f"- {name}：`{value}`；")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- repeat 0 与原 frozen-GINE SHUFFLED 结果逐 fold 一致。",
            "- 该审计在 single-shuffle 结果可见后设计，只用于机制诊断。",
            "- TRUE score、base epoch、model seed、dictionary recipe 与 residual capacity 均未重选。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--references", nargs="+", type=Path, default=list(REFERENCE_PATHS))
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--repeats", type=int, default=8)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    if args.repeats < 1:
        parser.error("--repeats must be positive")
    if len(args.references) != len(args.split_seeds):
        parser.error("--references and --split-seeds must have the same length")
    payload = run(args)
    _atomic_json(args.json, payload)
    _atomic_text(args.report, render(payload))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Strict joint structure-attribute dictionary screen on real TUD data."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch
from sklearn.model_selection import StratifiedKFold

from .data_tud import load_tud
from .from_scratch_unplanted_dictionary import deterministic_maximin_initialization
from .imdb_walk_dictionary import encode_with_minimum_sparsity
from .ksvd import ksvd
from .run_luyin14_node_level_fusion import (
    DEFAULT_ROOT,
    _attach,
    _load_pyg,
    _node_patch_vectors,
    _paired,
    _train_selected,
)
from .run_luyin14_route import _atomic_json, _atomic_text


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/joint_multiview_dictionary_stage_a_20260812.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/JOINT_MULTIVIEW_DICTIONARY_STAGE_A_20260812.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_LUYIN14_JOINT_MULTIVIEW_DICTIONARY_PROTOCOL_20260812.md"
VARIANTS = (
    "GIN_ONLY",
    "JOINT_FINAL_TRUE",
    "JOINT_FINAL_SHUFFLED",
    "JOINT_INIT_TRUE",
)


def _attribute_context(graph, features: np.ndarray) -> np.ndarray:
    neighbor_mean = np.zeros_like(features, dtype=np.float64)
    for node in range(graph.n):
        neighbors = list(graph.adj[node])
        if neighbors:
            neighbor_mean[node] = features[neighbors].mean(axis=0)
    return np.concatenate([features, neighbor_mean], axis=1).astype(np.float64)


def _shuffle_attribute_context(
    attributes: Sequence[np.ndarray], *, seed: int
) -> list[np.ndarray]:
    output = []
    for graph_index, values in enumerate(attributes):
        rng = np.random.default_rng(seed + graph_index * 1009)
        output.append(values[rng.permutation(len(values))])
    return output


def _fit_scaler(
    structures: Sequence[np.ndarray],
    attributes: Sequence[np.ndarray],
    train_indices: Sequence[int],
) -> dict[str, np.ndarray | float]:
    structure_train = np.concatenate(
        [structures[int(index)] for index in train_indices], axis=0
    )
    attribute_train = np.concatenate(
        [attributes[int(index)] for index in train_indices], axis=0
    )
    structure_mean = structure_train.mean(axis=0, keepdims=True)
    attribute_mean = attribute_train.mean(axis=0, keepdims=True)
    structure_rms = float(
        np.sqrt(np.mean(np.sum((structure_train - structure_mean) ** 2, axis=1)))
    )
    attribute_rms = float(
        np.sqrt(np.mean(np.sum((attribute_train - attribute_mean) ** 2, axis=1)))
    )
    return {
        "structure_mean": structure_mean,
        "attribute_mean": attribute_mean,
        "structure_rms": max(structure_rms, 1e-12),
        "attribute_rms": max(attribute_rms, 1e-12),
    }


def _joint_vectors(
    structures: Sequence[np.ndarray],
    attributes: Sequence[np.ndarray],
    scaler: dict[str, np.ndarray | float],
) -> list[np.ndarray]:
    output = []
    for structure, attribute in zip(structures, attributes):
        left = (structure - scaler["structure_mean"]) / scaler["structure_rms"]
        right = (attribute - scaler["attribute_mean"]) / scaler["attribute_rms"]
        output.append(np.concatenate([left, right], axis=1))
    return output


def _encode(
    vectors: np.ndarray, dictionary: np.ndarray, sparsity: int
) -> np.ndarray:
    return np.abs(
        encode_with_minimum_sparsity(
            vectors.T,
            dictionary,
            sparsity=min(sparsity, dictionary.shape[1]),
            minimum_sparsity=1,
        ).T
    )


def _fit_dictionary(
    vectors: Sequence[np.ndarray],
    train_indices: Sequence[int],
    *,
    n_atoms: int,
    sparsity: int,
    iterations: int,
    max_train_patches: int,
    seed: int,
) -> dict[str, Any]:
    train = np.concatenate([vectors[int(index)] for index in train_indices], axis=0)
    raw_count = len(train)
    if raw_count > max_train_patches:
        rng = np.random.default_rng(seed)
        selected = np.sort(rng.choice(raw_count, size=max_train_patches, replace=False))
        train = train[selected]
    nonzero = np.flatnonzero(np.linalg.norm(train, axis=1) > 1e-12)
    atoms = min(n_atoms, len(nonzero), train.shape[1])
    if atoms < 2:
        raise RuntimeError("too few nonzero joint node patches")
    initial, initialization = deterministic_maximin_initialization(
        train[nonzero].T, atoms
    )
    initial_codes = encode_with_minimum_sparsity(
        train.T, initial, sparsity=min(sparsity, atoms), minimum_sparsity=1
    )
    initial_recon = float(
        np.linalg.norm(train.T - initial @ initial_codes, "fro")
        / max(np.linalg.norm(train.T, "fro"), 1e-12)
    )
    final, _codes, training = ksvd(
        train.T,
        n_atoms=atoms,
        T=min(sparsity, atoms),
        T_min=1,
        n_iter=iterations,
        seed=0,
        initial_dictionary=initial,
    )
    return {
        "initial": initial,
        "final": final,
        "initial_reconstruction": initial_recon,
        "final_reconstruction": float(training["recon_rel"]),
        "initialization": initialization,
        "raw_train_patches": int(raw_count),
        "used_train_patches": int(len(train)),
    }


def _normalize_tokens(
    tokens: Sequence[np.ndarray], train_indices: Sequence[int]
) -> list[np.ndarray]:
    train = np.concatenate([tokens[int(index)] for index in train_indices], axis=0)
    mean = train.mean(axis=0, keepdims=True)
    scale = np.where(train.std(axis=0, keepdims=True) > 1e-8,
                     train.std(axis=0, keepdims=True), 1.0)
    return [(values - mean) / scale for values in tokens]


def run_dataset(name: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    graphs, labels, node_features, metadata = load_tud(name, args.dataset_root)
    raw, raw_labels, classes = _load_pyg(name, args.dataset_root)
    if not np.array_equal(labels, raw_labels):
        raise RuntimeError("Graph and PyG loaders disagree on labels")
    if any(features is None for features in node_features):
        raise RuntimeError(f"{name} does not have node attributes for every graph")
    structures = [
        _node_patch_vectors(graph, patch_size=args.patch_size, radius=args.radius)
        for graph in graphs
    ]
    attributes = [
        _attribute_context(graph, features)
        for graph, features in zip(graphs, node_features)
    ]
    shuffled_attributes = _shuffle_attribute_context(attributes, seed=271828)
    splitter = StratifiedKFold(
        n_splits=args.n_splits, shuffle=True, random_state=args.split_seed
    )
    folds = []
    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels)
    ):
        scaler = _fit_scaler(structures, attributes, train_indices)
        true_vectors = _joint_vectors(structures, attributes, scaler)
        shuffled_vectors = _joint_vectors(structures, shuffled_attributes, scaler)
        true_dictionary = _fit_dictionary(
            true_vectors,
            train_indices,
            n_atoms=args.n_atoms,
            sparsity=args.sparsity,
            iterations=args.iterations,
            max_train_patches=args.max_train_patches,
            seed=args.split_seed * 100 + fold_index,
        )
        shuffled_dictionary = _fit_dictionary(
            shuffled_vectors,
            train_indices,
            n_atoms=args.n_atoms,
            sparsity=args.sparsity,
            iterations=args.iterations,
            max_train_patches=args.max_train_patches,
            seed=args.split_seed * 100 + fold_index,
        )
        true_init = _normalize_tokens(
            [_encode(v, true_dictionary["initial"], args.sparsity) for v in true_vectors],
            train_indices,
        )
        true_final = _normalize_tokens(
            [_encode(v, true_dictionary["final"], args.sparsity) for v in true_vectors],
            train_indices,
        )
        shuffled_final = _normalize_tokens(
            [
                _encode(v, shuffled_dictionary["final"], args.sparsity)
                for v in shuffled_vectors
            ],
            train_indices,
        )
        datasets = {
            "GIN_ONLY": _attach(raw, None, mode="only"),
            "JOINT_FINAL_TRUE": _attach(raw, true_final, mode="film"),
            "JOINT_FINAL_SHUFFLED": _attach(raw, shuffled_final, mode="film"),
            "JOINT_INIT_TRUE": _attach(raw, true_init, mode="film"),
        }
        scores = {}
        for variant in VARIANTS:
            data = datasets[variant]
            mode = "only" if variant == "GIN_ONLY" else "film"
            scores[variant] = _train_selected(
                data,
                labels,
                train_indices,
                test_indices,
                input_dim=int(data[0].x.shape[1]),
                struct_dim=0 if mode == "only" else int(data[0].s.shape[1]),
                classes=classes,
                mode=mode,
                hidden=args.hidden,
                layers=args.layers,
                dropout=args.dropout,
                lr=args.lr,
                epochs=args.epochs,
                patience=args.patience,
                batch_size=args.batch_size,
                seed=args.model_seed * 1000 + fold_index,
                device=torch.device(args.device),
            )
            print(
                f"[{name}] fold={fold_index} {variant} "
                f"bacc={scores[variant]['balanced_accuracy']:.3f} "
                f"epoch={scores[variant]['selected_epoch']}",
                flush=True,
            )
        folds.append(
            {
                "fold_index": fold_index,
                "scores": scores,
                "true_initial_reconstruction": true_dictionary["initial_reconstruction"],
                "true_final_reconstruction": true_dictionary["final_reconstruction"],
                "shuffled_final_reconstruction": shuffled_dictionary["final_reconstruction"],
            }
        )
    paired = {
        "joint_final_minus_gin": _paired(folds, "JOINT_FINAL_TRUE", "GIN_ONLY"),
        "joint_true_minus_shuffled": _paired(
            folds, "JOINT_FINAL_TRUE", "JOINT_FINAL_SHUFFLED"
        ),
        "joint_final_minus_init": _paired(
            folds, "JOINT_FINAL_TRUE", "JOINT_INIT_TRUE"
        ),
    }
    return {
        "dataset": name,
        "metadata": metadata,
        "unique_structure_patches": int(
            len(np.unique(np.concatenate(structures), axis=0))
        ),
        "unique_joint_patches": int(
            len(np.unique(np.concatenate([
                np.concatenate([s, a], axis=1)
                for s, a in zip(structures, attributes)
            ]), axis=0))
        ),
        "folds": folds,
        "summary": {
            variant: {
                "balanced_accuracy_mean": float(np.mean([
                    fold["scores"][variant]["balanced_accuracy"] for fold in folds
                ])),
                "selected_epoch_mean": float(np.mean([
                    fold["scores"][variant]["selected_epoch"] for fold in folds
                ])),
            }
            for variant in VARIANTS
        },
        "paired": paired,
        "reconstruction": {
            "initial_mean": float(np.mean([
                fold["true_initial_reconstruction"] for fold in folds
            ])),
            "final_mean": float(np.mean([
                fold["true_final_reconstruction"] for fold in folds
            ])),
        },
        "seconds": time.time() - started,
    }


def classify(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    datasets = {}
    for result in results:
        pairs = result["paired"]
        reconstruction_improved = (
            result["reconstruction"]["final_mean"]
            < result["reconstruction"]["initial_mean"] - 1e-8
        )
        passed = (
            pairs["joint_final_minus_gin"]["mean"] >= 0.01
            and pairs["joint_final_minus_gin"]["wins"] >= 2
            and pairs["joint_true_minus_shuffled"]["mean"] >= 0.005
            and pairs["joint_true_minus_shuffled"]["wins"] >= 2
            and pairs["joint_final_minus_init"]["mean"] > 0.0
            and pairs["joint_final_minus_init"]["wins"] >= 2
            and reconstruction_improved
        )
        datasets[result["dataset"]] = {
            "pass": passed,
            "reconstruction_improved": reconstruction_improved,
            **pairs,
        }
    deltas = [row["joint_final_minus_gin"]["mean"] for row in datasets.values()]
    advance = any(row["pass"] for row in datasets.values()) and min(deltas) >= -0.01
    return {
        "classification": (
            "JOINT_MULTIVIEW_DICTIONARY_ADVANCES"
            if advance else "JOINT_MULTIVIEW_DICTIONARY_STAGE_A_NO_GO"
        ),
        "advance": advance,
        "datasets": datasets,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# luyin14 结构—属性共享稀疏码 strict Stage A",
        "",
        "> 日期：2026-08-12  ",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{payload['decision']['classification']}`",
        "",
        "## 1. 数据与字典诊断",
        "",
        "| dataset | unique structure | unique joint | INIT recon | FINAL recon |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in payload["datasets"]:
        lines.append(
            f"| {result['dataset']} | {result['unique_structure_patches']} | "
            f"{result['unique_joint_patches']} | "
            f"{result['reconstruction']['initial_mean']:.4f} | "
            f"{result['reconstruction']['final_mean']:.4f} |"
        )
    lines += [
        "",
        "## 2. Balanced accuracy",
        "",
        "| dataset | GIN | JOINT FINAL TRUE | JOINT SHUFFLED | JOINT INIT |",
        "|---|---:|---:|---:|---:|",
    ]
    for result in payload["datasets"]:
        get = lambda variant: result["summary"][variant]["balanced_accuracy_mean"]
        lines.append(
            f"| {result['dataset']} | {get('GIN_ONLY'):.3f} | "
            f"{get('JOINT_FINAL_TRUE'):.3f} | {get('JOINT_FINAL_SHUFFLED'):.3f} | "
            f"{get('JOINT_INIT_TRUE'):.3f} |"
        )
    lines += [
        "",
        "## 3. Paired deltas",
        "",
        "| dataset | FINAL-GIN | W/T/L | TRUE-SHUFFLED | W/T/L | FINAL-INIT | W/T/L |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["datasets"]:
        pairs = result["paired"]
        cells = []
        for key in (
            "joint_final_minus_gin",
            "joint_true_minus_shuffled",
            "joint_final_minus_init",
        ):
            pair = pairs[key]
            cells += [
                f"{pair['mean']:+.3f}",
                f"{pair['wins']}/{pair['ties']}/{pair['losses']}",
            ]
        lines.append(f"| {result['dataset']} | " + " | ".join(cells) + " |")
    lines += ["", "## 4. 结论", ""]
    if payload["decision"]["advance"]:
        lines.append(
            "共享稀疏码通过 Stage A；扩展 split seeds 1/2 后再决定是否增加 relation-aware encoder。"
        )
    else:
        lines.append(
            "共享稀疏码未通过 Stage A；不进入 cross-attention。需要改变 patch 语义或训练目标，"
            "不能继续把问题归因于融合头过于简单。"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    # Small TUD splits amplify tiny parallel reduction differences into
    # different early-stopping epochs.  Keep the strict screen reproducible
    # across fresh processes before any model or DataLoader is constructed.
    torch.set_num_threads(1)
    torch.use_deterministic_algorithms(True)
    parser = argparse.ArgumentParser()
    parser.add_argument("--protocol", default=PROTOCOL)
    parser.add_argument("--datasets", nargs="+", default=["MUTAG", "PTC_MR"])
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--radius", type=int, default=2)
    parser.add_argument("--n-atoms", type=int, default=24)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-train-patches", type=int, default=3000)
    parser.add_argument("--split-seed", type=int, default=0)
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--model-seed", type=int, default=0)
    parser.add_argument("--hidden", type=int, default=64)
    parser.add_argument("--layers", type=int, default=3)
    parser.add_argument("--dropout", type=float, default=0.5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=100)
    parser.add_argument("--patience", type=int, default=25)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    results = [run_dataset(dataset, args) for dataset in args.datasets]
    payload = {
        "protocol": args.protocol,
        "config": vars(args) | {
            "dataset_root": str(args.dataset_root),
            "json": str(args.json),
            "report": str(args.report),
        },
        "datasets": results,
    }
    payload["decision"] = classify(results)
    _atomic_json(args.json, payload)
    _atomic_text(args.report, render_report(payload))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

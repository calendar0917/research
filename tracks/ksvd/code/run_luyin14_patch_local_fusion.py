"""Patch-local compact bilinear fusion of KSVD activations and node features."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.decomposition import PCA
from sklearn.model_selection import StratifiedKFold

from .data_tud import load_tud
from .global_stable_ids import compute_global_wl_ids
from .run_luyin14_rich_readout import _feature_matrix as _rich_feature_matrix
from .run_luyin14_route import (
    DEFAULT_ROOT,
    FEATURE_DATASETS,
    PatchSet,
    PreparedGraph,
    _adjacency,
    _atomic_json,
    _atomic_text,
    _encode_patch_set,
    _fit_dictionary,
    _fit_score,
    _sampling_summary,
    prepare_graph,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/patch_local_multimodal_20260812.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/PATCH_LOCAL_MULTIMODAL_20260812.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_LUYIN14_PATCH_LOCAL_MULTIMODAL_PROTOCOL_20260812.md"
DEFAULT_DATASETS = ("MUTAG", "PTC_MR")


def _reordered_node_features(graph, features: np.ndarray) -> np.ndarray:
    stable = compute_global_wl_ids(_adjacency(graph))
    return np.asarray(features, dtype=np.float64)[np.asarray(stable.order, dtype=np.int64)]


def _patch_feature_means(
    patch_set: PatchSet, node_features: np.ndarray
) -> np.ndarray:
    return np.stack(
        [
            node_features[np.asarray(sorted(nodes), dtype=np.int64)].mean(axis=0)
            for nodes in patch_set.node_sets
        ]
    )


def _cross_correlation(
    codes: np.ndarray,
    patch_features: np.ndarray,
    *,
    permutation: np.ndarray | None = None,
) -> np.ndarray:
    activation = np.abs(codes).T
    features = np.asarray(patch_features, dtype=np.float64)
    if permutation is not None:
        features = features[np.asarray(permutation, dtype=np.int64)]
    activation = activation - activation.mean(axis=0, keepdims=True)
    features = features - features.mean(axis=0, keepdims=True)
    activation /= np.maximum(
        np.sqrt(np.mean(np.square(activation), axis=0, keepdims=True)), 1e-12
    )
    features /= np.maximum(
        np.sqrt(np.mean(np.square(features), axis=0, keepdims=True)), 1e-12
    )
    return (activation.T @ features / max(len(features), 1)).reshape(-1)


def _cross_matrices(
    prepared: Sequence[PreparedGraph],
    local_features: Sequence[np.ndarray],
    indices: Sequence[int],
    dictionary: dict[str, Any],
    *,
    sparsity: int,
    shuffle_seed: int,
) -> dict[str, np.ndarray]:
    rows = {
        "INIT_TRUE": [],
        "INIT_SHUFFLED": [],
        "FINAL_TRUE": [],
        "FINAL_SHUFFLED": [],
    }
    for index in indices:
        item = prepared[int(index)]
        patch_features = _patch_feature_means(
            item.fair, local_features[int(index)]
        )
        init_codes = _encode_patch_set(
            item.fair, dictionary["initial"], dictionary["mean"], sparsity
        )
        final_codes = _encode_patch_set(
            item.fair, dictionary["final"], dictionary["mean"], sparsity
        )
        rng = np.random.default_rng(shuffle_seed + item.index * 1009)
        permutation = rng.permutation(len(patch_features))
        rows["INIT_TRUE"].append(
            _cross_correlation(init_codes, patch_features)
        )
        rows["INIT_SHUFFLED"].append(
            _cross_correlation(init_codes, patch_features, permutation=permutation)
        )
        rows["FINAL_TRUE"].append(
            _cross_correlation(final_codes, patch_features)
        )
        rows["FINAL_SHUFFLED"].append(
            _cross_correlation(final_codes, patch_features, permutation=permutation)
        )
    return {name: np.stack(values) for name, values in rows.items()}


def _fit_transform_pca(
    train_true: np.ndarray,
    train_shuffled: np.ndarray,
    test_true: np.ndarray,
    test_shuffled: np.ndarray,
    *,
    rank: int,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    active = np.std(train_true, axis=0) > 1e-12
    if not np.any(active):
        zeros_train = np.zeros((len(train_true), 1), dtype=np.float64)
        zeros_test = np.zeros((len(test_true), 1), dtype=np.float64)
        return (
            {
                "train_true": zeros_train,
                "train_shuffled": zeros_train.copy(),
                "test_true": zeros_test,
                "test_shuffled": zeros_test.copy(),
            },
            {"rank": 1, "active_dimensions": 0, "explained_variance": 0.0},
        )
    values = train_true[:, active]
    components = min(rank, values.shape[0] - 1, values.shape[1])
    pca = PCA(n_components=max(components, 1), random_state=seed).fit(values)
    return (
        {
            "train_true": pca.transform(train_true[:, active]),
            "train_shuffled": pca.transform(train_shuffled[:, active]),
            "test_true": pca.transform(test_true[:, active]),
            "test_shuffled": pca.transform(test_shuffled[:, active]),
        },
        {
            "rank": int(pca.n_components_),
            "active_dimensions": int(np.sum(active)),
            "explained_variance": float(np.sum(pca.explained_variance_ratio_)),
        },
    )


def _paired(folds: Sequence[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    values = np.asarray(
        [
            fold["scores"][left]["balanced_accuracy"]
            - fold["scores"][right]["balanced_accuracy"]
            for fold in folds
        ]
    )
    return {
        "mean": float(values.mean()),
        "std": float(values.std()),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
        "values": values.tolist(),
    }


def _aggregate(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    return {
        name: {
            "balanced_accuracy_mean": float(
                np.mean([fold["scores"][name]["balanced_accuracy"] for fold in folds])
            ),
            "balanced_accuracy_std": float(
                np.std([fold["scores"][name]["balanced_accuracy"] for fold in folds])
            ),
        }
        for name in sorted(folds[0]["scores"])
    }


def run_dataset(name: str, args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    graphs, labels, node_features, metadata = load_tud(name, args.dataset_root)
    if name not in FEATURE_DATASETS or any(feature is None for feature in node_features):
        raise ValueError(f"{name} does not provide the required node features")
    if args.limit is not None and args.limit < len(graphs):
        keep = []
        rng = np.random.default_rng(20260812)
        per_class = max(1, args.limit // len(np.unique(labels)))
        for label in np.unique(labels):
            candidates = np.flatnonzero(labels == label)
            keep.extend(
                int(value)
                for value in rng.choice(
                    candidates, size=min(per_class, len(candidates)), replace=False
                )
            )
        keep = sorted(keep)[: args.limit]
        graphs = [graphs[index] for index in keep]
        node_features = [node_features[index] for index in keep]
        labels = labels[np.asarray(keep)]
        metadata = {**metadata, "limited_graph_count": len(graphs)}
    prepared = [
        prepare_graph(
            index,
            graph,
            int(label),
            feature,
            patch_size=args.patch_size,
            overlap=args.overlap,
            maximum_patches=args.maximum_patches,
            retained_beam=args.retained_beam,
            seed=20260812 + index * 1009,
        )
        for index, (graph, label, feature) in enumerate(zip(graphs, labels, node_features))
    ]
    local_features = [
        _reordered_node_features(graph, feature)
        for graph, feature in zip(graphs, node_features)
    ]
    folds = []
    for split_seed in args.split_seeds:
        splitter = StratifiedKFold(
            n_splits=args.n_splits, shuffle=True, random_state=split_seed
        )
        for fold_index, (train_indices, test_indices) in enumerate(
            splitter.split(np.zeros(len(labels)), labels)
        ):
            seed = split_seed * 100 + fold_index
            dictionary = _fit_dictionary(
                prepared,
                train_indices,
                branch="fair",
                n_atoms=args.n_atoms,
                sparsity=args.sparsity,
                iterations=args.iterations,
                max_train_patches=args.max_train_patches,
                seed=seed,
            )
            rich_train = _rich_feature_matrix(
                prepared, train_indices, dictionary, sparsity=args.sparsity
            )
            rich_test = _rich_feature_matrix(
                prepared, test_indices, dictionary, sparsity=args.sparsity
            )
            cross_train = _cross_matrices(
                prepared,
                local_features,
                train_indices,
                dictionary,
                sparsity=args.sparsity,
                shuffle_seed=314159 + seed,
            )
            cross_test = _cross_matrices(
                prepared,
                local_features,
                test_indices,
                dictionary,
                sparsity=args.sparsity,
                shuffle_seed=314159 + seed,
            )
            blocks = {}
            pca_metadata = {}
            for family in ("INIT", "FINAL"):
                transformed, pca_info = _fit_transform_pca(
                    cross_train[f"{family}_TRUE"],
                    cross_train[f"{family}_SHUFFLED"],
                    cross_test[f"{family}_TRUE"],
                    cross_test[f"{family}_SHUFFLED"],
                    rank=args.pca_rank,
                    seed=seed,
                )
                blocks[f"{family}_TRUE_CROSS"] = (
                    np.concatenate(
                        [rich_train["FEATURE_STATS"], transformed["train_true"]], axis=1
                    ),
                    np.concatenate(
                        [rich_test["FEATURE_STATS"], transformed["test_true"]], axis=1
                    ),
                )
                blocks[f"{family}_SHUFFLED_CROSS"] = (
                    np.concatenate(
                        [rich_train["FEATURE_STATS"], transformed["train_shuffled"]], axis=1
                    ),
                    np.concatenate(
                        [rich_test["FEATURE_STATS"], transformed["test_shuffled"]], axis=1
                    ),
                )
                pca_metadata[family] = pca_info
            blocks["FEATURE_STATS"] = (
                rich_train["FEATURE_STATS"], rich_test["FEATURE_STATS"]
            )
            blocks["NAIVE_FINAL_RICH"] = (
                rich_train["FEATURE_STATS_FINAL_RICH"],
                rich_test["FEATURE_STATS_FINAL_RICH"],
            )
            scores = {
                feature_name: _fit_score(
                    train,
                    labels[train_indices],
                    test,
                    labels[test_indices],
                    seed=seed,
                )
                for feature_name, (train, test) in blocks.items()
            }
            folds.append(
                {
                    "split_seed": split_seed,
                    "fold_index": fold_index,
                    "scores": scores,
                    "pca": pca_metadata,
                }
            )
    paired = {
        "final_true_minus_feature_stats": _paired(
            folds, "FINAL_TRUE_CROSS", "FEATURE_STATS"
        ),
        "final_true_minus_final_shuffled": _paired(
            folds, "FINAL_TRUE_CROSS", "FINAL_SHUFFLED_CROSS"
        ),
        "final_true_minus_init_true": _paired(
            folds, "FINAL_TRUE_CROSS", "INIT_TRUE_CROSS"
        ),
        "final_true_minus_naive_rich": _paired(
            folds, "FINAL_TRUE_CROSS", "NAIVE_FINAL_RICH"
        ),
    }
    return {
        "dataset": name,
        "metadata": metadata,
        "sampling": _sampling_summary(prepared),
        "summary": _aggregate(folds),
        "paired": paired,
        "folds": folds,
        "seconds": time.time() - started,
    }


def classify(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    by_name = {result["dataset"]: result for result in results}
    base = {
        name: result["paired"]["final_true_minus_feature_stats"]
        for name, result in by_name.items()
    }
    shuffled = {
        name: result["paired"]["final_true_minus_final_shuffled"]
        for name, result in by_name.items()
    }

    def cross_gate(rows: dict[str, dict[str, Any]]) -> bool:
        values = [row["mean"] for row in rows.values()]
        stable = [row["mean"] >= 0.01 and row["wins"] >= 6 for row in rows.values()]
        return any(stable) and min(values) >= -0.01

    learning_values = [
        result["paired"]["final_true_minus_init_true"]["mean"]
        for result in results
    ]
    naive_values = [
        result["paired"]["final_true_minus_naive_rich"]["mean"]
        for result in results
    ]
    gates = {
        "added_value": cross_gate(base),
        "binding": cross_gate(shuffled),
        "ksvd_learning": float(np.mean(learning_values)) > 0.0
        and any(value > 0.0 for value in learning_values),
        "beats_naive": float(np.mean(naive_values)) > 0.0,
    }
    label = (
        "PATCH_LOCAL_MULTIMODAL_ALIGNMENT_ADVANCES"
        if all(gates.values())
        else "PATCH_LOCAL_MULTIMODAL_ALIGNMENT_NO_GO"
    )
    return {"classification": label, "gates": gates}


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# luyin14 patch-local KSVD × node-feature 多模态融合",
        "",
        "> 日期：2026-08-12  ",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{payload['decision']['classification']}`",
        "",
        "## 1. Gate",
        "",
    ]
    for name, value in payload["decision"]["gates"].items():
        lines.append(f"- {name}: `{value}`；")
    lines.extend(
        [
            "",
            "## 2. Balanced accuracy",
            "",
            "| dataset | feature+stats | naïve +FINAL rich | INIT true cross | FINAL true cross | FINAL shuffled cross |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in payload["datasets"]:
        summary = result["summary"]
        get = lambda name: summary[name]["balanced_accuracy_mean"]
        lines.append(
            f"| {result['dataset']} | {get('FEATURE_STATS'):.3f} | "
            f"{get('NAIVE_FINAL_RICH'):.3f} | {get('INIT_TRUE_CROSS'):.3f} | "
            f"{get('FINAL_TRUE_CROSS'):.3f} | {get('FINAL_SHUFFLED_CROSS'):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 3. Paired deltas",
            "",
            "| dataset | TRUE-base | W/T/L | TRUE-SHUFFLED | W/T/L | FINAL-INIT | W/T/L | TRUE-naïve |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in payload["datasets"]:
        pairs = [
            result["paired"]["final_true_minus_feature_stats"],
            result["paired"]["final_true_minus_final_shuffled"],
            result["paired"]["final_true_minus_init_true"],
        ]
        cells = []
        for pair in pairs:
            cells.extend(
                [f"{pair['mean']:+.3f}", f"{pair['wins']}/{pair['ties']}/{pair['losses']}"]
            )
        naive = result["paired"]["final_true_minus_naive_rich"]
        lines.append(
            f"| {result['dataset']} | " + " | ".join(cells) + f" | {naive['mean']:+.3f} |"
        )
    lines.extend(["", "## 4. 结论", ""])
    if payload["decision"]["classification"] == "PATCH_LOCAL_MULTIMODAL_ALIGNMENT_ADVANCES":
        lines.append(
            "真实 patch-local atom×feature binding 通过。下一步可把相同对齐对象升级为低容量 FiLM/cross-attention，并保留 shuffled binding、INIT 与 feature+stats controls。"
        )
    else:
        lines.append(
            "compact bilinear 局部对齐未形成稳定增益。现有失败不能仅归因于图级 concat 太粗；在增加 cross-attention 容量前，需要先改变结构专家或 patch 语义。"
        )
    lines.append("")
    return "\n".join(lines)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=list(DEFAULT_DATASETS))
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--patch-size", type=int, default=8)
    parser.add_argument("--overlap", type=int, default=2)
    parser.add_argument("--maximum-patches", type=int, default=48)
    parser.add_argument("--retained-beam", type=int, default=4)
    parser.add_argument("--n-atoms", type=int, default=24)
    parser.add_argument("--sparsity", type=int, default=3)
    parser.add_argument("--iterations", type=int, default=5)
    parser.add_argument("--max-train-patches", type=int, default=3000)
    parser.add_argument("--pca-rank", type=int, default=16)
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    results = []
    for dataset in args.datasets:
        print(f"[{dataset}] start", flush=True)
        result = run_dataset(dataset, args)
        print(f"[{dataset}] done in {result['seconds']:.1f}s", flush=True)
        results.append(result)
    payload = {
        "protocol": PROTOCOL,
        "config": vars(args)
        | {
            "dataset_root": str(args.dataset_root),
            "json": str(args.json),
            "report": str(args.report),
        },
        "datasets": results,
        "decision": classify(results),
    }
    _atomic_json(args.json, payload)
    _atomic_text(args.report, render_report(payload))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


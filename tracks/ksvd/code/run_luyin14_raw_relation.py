"""Terminal uncompressed RAW-patch relation screen for luyin14."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .data_tud import load_tud
from .run_luyin14_route import (
    DEFAULT_DATASETS,
    DEFAULT_ROOT,
    FEATURE_DATASETS,
    PreparedGraph,
    _aligned_relation_readout,
    _atomic_json,
    _atomic_text,
    _fit_score,
    _raw_patch_readout,
    _relation_graph_readout,
    _sampling_summary,
    prepare_graph,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/raw_relation_20260812.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/RAW_RELATION_20260812.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_LUYIN14_RAW_RELATION_PROTOCOL_20260812.md"


def _feature_matrix(
    prepared: Sequence[PreparedGraph],
    indices: Sequence[int],
    *,
    shuffle_seed: int,
) -> dict[str, np.ndarray]:
    names = (
        "STATS",
        "RAW_BAG",
        "RAW_RELATION_GRAPH",
        "RAW_TRUE_RELATION",
        "RAW_SHUFFLED_RELATION",
        "STATS_RAW_TRUE_RELATION",
        "FEATURE_ONLY",
        "FEATURE_STATS",
        "FEATURE_STATS_RAW_TRUE_RELATION",
    )
    rows = {name: [] for name in names}
    for index in indices:
        item = prepared[int(index)]
        bag = _raw_patch_readout(item.fair)
        relation_graph = _relation_graph_readout(item.fair)
        tokens = item.fair.vectors.T
        true_relation = _aligned_relation_readout(tokens, item.fair)
        rng = np.random.default_rng(shuffle_seed + item.index * 1009)
        shuffled_relation = _aligned_relation_readout(
            tokens, item.fair, permutation=rng.permutation(tokens.shape[1])
        )
        true = np.concatenate([bag, relation_graph, true_relation])
        shuffled = np.concatenate([bag, relation_graph, shuffled_relation])
        rows["STATS"].append(item.stats)
        rows["RAW_BAG"].append(bag)
        rows["RAW_RELATION_GRAPH"].append(np.concatenate([bag, relation_graph]))
        rows["RAW_TRUE_RELATION"].append(true)
        rows["RAW_SHUFFLED_RELATION"].append(shuffled)
        rows["STATS_RAW_TRUE_RELATION"].append(np.concatenate([item.stats, true]))
        if item.node_features is not None:
            feature_stats = np.concatenate([item.node_features, item.stats])
            rows["FEATURE_ONLY"].append(item.node_features)
            rows["FEATURE_STATS"].append(feature_stats)
            rows["FEATURE_STATS_RAW_TRUE_RELATION"].append(
                np.concatenate([feature_stats, true])
            )
    return {name: np.stack(values) for name, values in rows.items() if values}


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
    folds = []
    for split_seed in args.split_seeds:
        splitter = StratifiedKFold(
            n_splits=args.n_splits, shuffle=True, random_state=split_seed
        )
        for fold_index, (train_indices, test_indices) in enumerate(
            splitter.split(np.zeros(len(labels)), labels)
        ):
            seed = split_seed * 100 + fold_index
            train_features = _feature_matrix(
                prepared, train_indices, shuffle_seed=731421 + seed
            )
            test_features = _feature_matrix(
                prepared, test_indices, shuffle_seed=731421 + seed
            )
            folds.append(
                {
                    "split_seed": split_seed,
                    "fold_index": fold_index,
                    "scores": {
                        feature_name: _fit_score(
                            train_features[feature_name],
                            labels[train_indices],
                            test_features[feature_name],
                            labels[test_indices],
                            seed=seed,
                        )
                        for feature_name in train_features
                    },
                }
            )
    paired = {
        "true_minus_shuffled": _paired(
            folds, "RAW_TRUE_RELATION", "RAW_SHUFFLED_RELATION"
        ),
        "true_minus_bag": _paired(folds, "RAW_TRUE_RELATION", "RAW_BAG"),
        "stats_true_minus_stats": _paired(
            folds, "STATS_RAW_TRUE_RELATION", "STATS"
        ),
    }
    if name in FEATURE_DATASETS:
        paired["feature_stats_true_minus_feature_stats"] = _paired(
            folds, "FEATURE_STATS_RAW_TRUE_RELATION", "FEATURE_STATS"
        )
    return {
        "dataset": name,
        "metadata": metadata,
        "sampling": _sampling_summary(prepared),
        "summary": _aggregate(folds),
        "paired": paired,
        "folds": folds,
        "seconds": time.time() - started,
    }


def _pass(pair: dict[str, Any]) -> bool:
    return pair["mean"] >= 0.01 and pair["wins"] >= 6


def classify(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = {
        "binding": "true_minus_shuffled",
        "relation_over_bag": "true_minus_bag",
        "added_value": "stats_true_minus_stats",
    }
    axes = {
        axis: {
            result["dataset"]: {
                **result["paired"][key],
                "pass": _pass(result["paired"][key]),
            }
            for result in results
        }
        for axis, key in keys.items()
    }
    gates = {
        axis: sum(row["pass"] for row in datasets.values()) >= 2
        for axis, datasets in axes.items()
    }
    if all(gates.values()):
        label = "RAW_RELATION_ADVANCES_TO_PATCH_GRAPH_ENCODER"
    elif gates["binding"] and not gates["added_value"]:
        label = "RAW_RELATION_REPEATS_GRAPH_STATS"
    else:
        label = "RAW_RELATION_TERMINAL_NO_GO"
    return {"classification": label, "gates": gates, "axes": axes}


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# luyin14 uncompressed RAW patch relation terminal screen",
        "",
        "> 日期：2026-08-12  ",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{payload['decision']['classification']}`",
        "",
        "## 1. Gate",
        "",
        f"- TRUE > SHUFFLED binding：`{payload['decision']['gates']['binding']}`；",
        f"- TRUE > RAW bag：`{payload['decision']['gates']['relation_over_bag']}`；",
        f"- STATS+TRUE > STATS：`{payload['decision']['gates']['added_value']}`。",
        "",
        "## 2. Balanced accuracy 与 paired delta",
        "",
        "| dataset | STATS | RAW bag | TRUE | SHUFFLED | TRUE-SHUFFLED | W/T/L | TRUE-BAG | W/T/L | STATS+TRUE-STATS | W/T/L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["datasets"]:
        summary = result["summary"]
        get = lambda name: summary[name]["balanced_accuracy_mean"]
        pairs = [
            result["paired"]["true_minus_shuffled"],
            result["paired"]["true_minus_bag"],
            result["paired"]["stats_true_minus_stats"],
        ]
        cells = []
        for pair in pairs:
            cells.extend(
                [f"{pair['mean']:+.3f}", f"{pair['wins']}/{pair['ties']}/{pair['losses']}"]
            )
        lines.append(
            f"| {result['dataset']} | {get('STATS'):.3f} | {get('RAW_BAG'):.3f} | "
            f"{get('RAW_TRUE_RELATION'):.3f} | {get('RAW_SHUFFLED_RELATION'):.3f} | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(["", "## 3. 结论", ""])
    label = payload["decision"]["classification"]
    if label == "RAW_RELATION_ADVANCES_TO_PATCH_GRAPH_ENCODER":
        lines.append(
            "未压缩 patch relation 同时通过 binding、bag 增量和 graph-stats 外增量；可立项低容量 relation-aware patch encoder，但 KSVD 只能作为可选辅助对照。"
        )
    elif label == "RAW_RELATION_REPEATS_GRAPH_STATS":
        lines.append(
            "真实绑定存在，但没有越过简单 graph statistics；不应进入 Transformer，关系只保留为解释/诊断。"
        )
    else:
        lines.append(
            "未压缩 RAW token 也没有形成跨数据集、stats 外的 relation 增益。停止 patch-graph/Transformer 下游路线。"
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


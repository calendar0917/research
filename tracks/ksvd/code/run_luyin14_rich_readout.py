"""Audit whether the MolHIV rich sparse-code readout transfers to luyin14.

This is intentionally a narrow follow-up. It reuses the frozen FAIR95 patches,
dictionary settings, folds, and linear classifier from ``run_luyin14_route``.
"""
from __future__ import annotations

import argparse
import json
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
    _atomic_json,
    _atomic_text,
    _code_readout,
    _encode_patch_set,
    _fit_dictionary,
    _fit_score,
    _sampling_summary,
    prepare_graph,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/rich_readout_20260812.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/RICH_READOUT_20260812.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_LUYIN14_RICH_READOUT_PROTOCOL_20260812.md"

BASE_FEATURES = (
    "STATS",
    "INIT_COARSE",
    "FINAL_COARSE",
    "INIT_RICH_NO_RECON",
    "FINAL_RICH_NO_RECON",
    "INIT_RICH",
    "FINAL_RICH",
    "STATS_INIT_RICH",
    "STATS_FINAL_RICH",
)
NODE_FEATURES = (
    "FEATURE_ONLY",
    "FEATURE_STATS",
    "FEATURE_INIT_RICH",
    "FEATURE_FINAL_RICH",
    "FEATURE_STATS_INIT_RICH",
    "FEATURE_STATS_FINAL_RICH",
)


def _rich_code_readouts(
    codes: np.ndarray,
    centered_patches: np.ndarray,
    dictionary: np.ndarray,
) -> dict[str, np.ndarray]:
    """Return the frozen MolHIV-style rich summaries for one graph."""
    absolute = np.abs(codes)
    atoms, count = absolute.shape
    zero = np.zeros(atoms, dtype=np.float64)
    if count:
        mean = absolute.mean(axis=1)
        maximum = absolute.max(axis=1)
        top3 = np.sort(absolute, axis=1)[:, -min(3, count) :].mean(axis=1)
        std = absolute.std(axis=1)
        usage = (absolute > 1e-10).mean(axis=1)
        q75, q90 = np.quantile(absolute, [0.75, 0.90], axis=1)
        energy = np.square(codes).mean(axis=1)
        signed_mean = codes.mean(axis=1)
        winner = (
            np.bincount(np.argmax(absolute, axis=0), minlength=atoms).astype(np.float64)
            / count
        )
    else:
        mean = maximum = top3 = std = usage = q75 = q90 = energy = signed_mean = winner = zero
    rich_no_recon = np.concatenate(
        [mean, maximum, top3, std, usage, q75, q90, energy, signed_mean, winner]
    )
    residual = centered_patches - dictionary @ codes
    denominators = np.maximum(np.linalg.norm(centered_patches, axis=0), 1e-12)
    errors = np.linalg.norm(residual, axis=0) / denominators
    if errors.size:
        recon = np.asarray(
            [
                errors.mean(),
                errors.std(),
                *np.quantile(errors, [0.50, 0.75, 0.90]),
                errors.max(),
            ],
            dtype=np.float64,
        )
    else:
        recon = np.zeros(6, dtype=np.float64)
    count_features = np.asarray([float(count), np.log1p(count)], dtype=np.float64)
    return {
        "rich_no_recon": rich_no_recon,
        "rich": np.concatenate([rich_no_recon, recon, count_features]),
    }


def _feature_matrix(
    prepared: Sequence[PreparedGraph],
    indices: Sequence[int],
    dictionary: dict[str, Any],
    *,
    sparsity: int,
) -> dict[str, np.ndarray]:
    rows = {name: [] for name in BASE_FEATURES + NODE_FEATURES}
    for index in indices:
        item = prepared[int(index)]
        centered = item.fair.vectors.T - dictionary["mean"]
        init_codes = _encode_patch_set(
            item.fair, dictionary["initial"], dictionary["mean"], sparsity
        )
        final_codes = _encode_patch_set(
            item.fair, dictionary["final"], dictionary["mean"], sparsity
        )
        init_rich = _rich_code_readouts(
            init_codes, centered, dictionary["initial"]
        )
        final_rich = _rich_code_readouts(
            final_codes, centered, dictionary["final"]
        )
        rows["STATS"].append(item.stats)
        rows["INIT_COARSE"].append(_code_readout(init_codes))
        rows["FINAL_COARSE"].append(_code_readout(final_codes))
        rows["INIT_RICH_NO_RECON"].append(init_rich["rich_no_recon"])
        rows["FINAL_RICH_NO_RECON"].append(final_rich["rich_no_recon"])
        rows["INIT_RICH"].append(init_rich["rich"])
        rows["FINAL_RICH"].append(final_rich["rich"])
        rows["STATS_INIT_RICH"].append(np.concatenate([item.stats, init_rich["rich"]]))
        rows["STATS_FINAL_RICH"].append(np.concatenate([item.stats, final_rich["rich"]]))
        if item.node_features is not None:
            feature = item.node_features
            feature_stats = np.concatenate([feature, item.stats])
            rows["FEATURE_ONLY"].append(feature)
            rows["FEATURE_STATS"].append(feature_stats)
            rows["FEATURE_INIT_RICH"].append(np.concatenate([feature, init_rich["rich"]]))
            rows["FEATURE_FINAL_RICH"].append(np.concatenate([feature, final_rich["rich"]]))
            rows["FEATURE_STATS_INIT_RICH"].append(
                np.concatenate([feature_stats, init_rich["rich"]])
            )
            rows["FEATURE_STATS_FINAL_RICH"].append(
                np.concatenate([feature_stats, final_rich["rich"]])
            )
    return {name: np.stack(values) for name, values in rows.items() if values}


def _aggregate(folds: Sequence[dict[str, Any]]) -> dict[str, Any]:
    names = sorted({name for fold in folds for name in fold["scores"]})
    return {
        name: {
            "balanced_accuracy_mean": float(
                np.mean([fold["scores"][name]["balanced_accuracy"] for fold in folds])
            ),
            "balanced_accuracy_std": float(
                np.std([fold["scores"][name]["balanced_accuracy"] for fold in folds])
            ),
            "accuracy_mean": float(
                np.mean([fold["scores"][name]["accuracy"] for fold in folds])
            ),
            "fold_count": len(folds),
            "dimension": int(folds[0]["dimensions"][name]),
        }
        for name in names
    }


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
        "left": left,
        "right": right,
        "mean": float(values.mean()),
        "std": float(values.std()),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
        "values": values.tolist(),
    }


def run_dataset(
    name: str,
    *,
    dataset_root: Path,
    patch_size: int,
    overlap: int,
    maximum_patches: int,
    retained_beam: int,
    n_atoms: int,
    sparsity: int,
    iterations: int,
    max_train_patches: int,
    split_seeds: Sequence[int],
    n_splits: int,
    limit: int | None,
) -> dict[str, Any]:
    started = time.time()
    graphs, labels, node_features, metadata = load_tud(name, dataset_root)
    if limit is not None and limit < len(graphs):
        keep = []
        rng = np.random.default_rng(20260812)
        per_class = max(1, limit // len(np.unique(labels)))
        for label in np.unique(labels):
            candidates = np.flatnonzero(labels == label)
            keep.extend(
                int(value)
                for value in rng.choice(
                    candidates, size=min(per_class, len(candidates)), replace=False
                )
            )
        keep = sorted(keep)[:limit]
        graphs = [graphs[index] for index in keep]
        node_features = [node_features[index] for index in keep]
        labels = labels[np.asarray(keep, dtype=np.int64)]
        metadata = {**metadata, "limited_graph_count": len(graphs)}
    prepared = [
        prepare_graph(
            index,
            graph,
            int(label),
            feature,
            patch_size=patch_size,
            overlap=overlap,
            maximum_patches=maximum_patches,
            retained_beam=retained_beam,
            seed=20260812 + index * 1009,
        )
        for index, (graph, label, feature) in enumerate(zip(graphs, labels, node_features))
    ]
    folds = []
    for split_seed in split_seeds:
        splitter = StratifiedKFold(
            n_splits=n_splits, shuffle=True, random_state=int(split_seed)
        )
        for fold_index, (train_indices, test_indices) in enumerate(
            splitter.split(np.zeros(len(labels)), labels)
        ):
            dictionary_seed = int(split_seed) * 100 + fold_index
            dictionary = _fit_dictionary(
                prepared,
                train_indices,
                branch="fair",
                n_atoms=n_atoms,
                sparsity=sparsity,
                iterations=iterations,
                max_train_patches=max_train_patches,
                seed=dictionary_seed,
            )
            train_features = _feature_matrix(
                prepared, train_indices, dictionary, sparsity=sparsity
            )
            test_features = _feature_matrix(
                prepared, test_indices, dictionary, sparsity=sparsity
            )
            scores = {
                feature_name: _fit_score(
                    train_features[feature_name],
                    labels[train_indices],
                    test_features[feature_name],
                    labels[test_indices],
                    seed=dictionary_seed,
                )
                for feature_name in train_features
            }
            folds.append(
                {
                    "split_seed": int(split_seed),
                    "fold_index": int(fold_index),
                    "train_count": int(len(train_indices)),
                    "test_count": int(len(test_indices)),
                    "dictionary": {
                        "atoms": int(dictionary["atoms"]),
                        "raw_train_patches": int(dictionary["raw_train_patches"]),
                        "used_train_patches": int(dictionary["used_train_patches"]),
                        "final_reconstruction": dictionary["training"].get("recon_rel"),
                    },
                    "dimensions": {
                        feature_name: int(values.shape[1])
                        for feature_name, values in train_features.items()
                    },
                    "scores": scores,
                }
            )
    comparisons = {
        "readout_final_rich_minus_final_coarse": _paired(
            folds, "FINAL_RICH", "FINAL_COARSE"
        ),
        "learning_final_rich_minus_init_rich": _paired(
            folds, "FINAL_RICH", "INIT_RICH"
        ),
        "added_stats_final_rich_minus_stats": _paired(
            folds, "STATS_FINAL_RICH", "STATS"
        ),
        "reconstruction_features_final_rich_minus_no_recon": _paired(
            folds, "FINAL_RICH", "FINAL_RICH_NO_RECON"
        ),
    }
    if name in FEATURE_DATASETS:
        comparisons.update(
            {
                "feature_final_rich_minus_feature": _paired(
                    folds, "FEATURE_FINAL_RICH", "FEATURE_ONLY"
                ),
                "feature_stats_final_rich_minus_feature_stats": _paired(
                    folds, "FEATURE_STATS_FINAL_RICH", "FEATURE_STATS"
                ),
            }
        )
    return {
        "dataset": name,
        "metadata": metadata,
        "sampling": _sampling_summary(prepared),
        "folds": folds,
        "summary": _aggregate(folds),
        "paired": comparisons,
        "seconds": time.time() - started,
    }


def _passes(pair: dict[str, Any]) -> bool:
    return pair["mean"] >= 0.01 and pair["wins"] >= 6


def classify(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    axes = {
        "readout": "readout_final_rich_minus_final_coarse",
        "learning": "learning_final_rich_minus_init_rich",
        "added_value": "added_stats_final_rich_minus_stats",
    }
    details = {
        axis: {
            result["dataset"]: {
                **result["paired"][key],
                "pass": _passes(result["paired"][key]),
            }
            for result in results
        }
        for axis, key in axes.items()
    }
    gates = {
        axis: sum(row["pass"] for row in datasets.values()) >= 2
        for axis, datasets in details.items()
    }
    if all(gates.values()):
        label = "RICH_READOUT_TRANSFERS_AS_KSVD_ROUTE"
    elif gates["readout"] and not gates["learning"]:
        label = "RICH_STATISTICS_HELP_WITHOUT_KSVD_UPDATE_ATTRIBUTION"
    elif gates["readout"] and not gates["added_value"]:
        label = "RICH_STATISTICS_REPEAT_GRAPH_STATS"
    else:
        label = "MOLHIV_RICH_READOUT_DOES_NOT_TRANSFER"
    return {"classification": label, "gates": gates, "axes": details}


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# luyin14 rich sparse-code readout 迁移审计",
        "",
        "> 日期：2026-08-12  ",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{payload['decision']['classification']}`",
        "",
        "## 1. 三个主 gate",
        "",
        "| gate | pass | 预注册要求 |",
        "|---|---:|---|",
    ]
    labels = {
        "readout": "FINAL rich > FINAL coarse",
        "learning": "FINAL rich > INIT rich",
        "added_value": "STATS+FINAL rich > STATS",
    }
    for key, passed in payload["decision"]["gates"].items():
        lines.append(
            f"| {labels[key]} | `{passed}` | 至少 2/4 数据集 mean >= +.01 且 wins >= 6/9 |"
        )
    lines.extend(
        [
            "",
            "## 2. Balanced accuracy 主表",
            "",
            "| dataset | STATS | INIT coarse | FINAL coarse | INIT rich | FINAL rich | STATS+INIT rich | STATS+FINAL rich |",
            "|---|---:|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for result in payload["datasets"]:
        summary = result["summary"]
        get = lambda name: summary[name]["balanced_accuracy_mean"]
        lines.append(
            f"| {result['dataset']} | {get('STATS'):.3f} | {get('INIT_COARSE'):.3f} | "
            f"{get('FINAL_COARSE'):.3f} | {get('INIT_RICH'):.3f} | {get('FINAL_RICH'):.3f} | "
            f"{get('STATS_INIT_RICH'):.3f} | {get('STATS_FINAL_RICH'):.3f} |"
        )
    lines.extend(
        [
            "",
            "## 3. 预注册 paired deltas",
            "",
            "| dataset | rich-coarse | W/T/L | FINAL-INIT rich | W/T/L | STATS+rich-STATS | W/T/L |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    keys = (
        "readout_final_rich_minus_final_coarse",
        "learning_final_rich_minus_init_rich",
        "added_stats_final_rich_minus_stats",
    )
    for result in payload["datasets"]:
        pairs = [result["paired"][key] for key in keys]
        cells = []
        for pair in pairs:
            cells.extend(
                [
                    f"{pair['mean']:+.3f}",
                    f"{pair['wins']}/{pair['ties']}/{pair['losses']}",
                ]
            )
        lines.append(f"| {result['dataset']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## 4. 节点特征数据",
            "",
            "| dataset | feature | feature+stats | feature+FINAL rich | feature+stats+FINAL rich | rich over feature+stats |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in payload["datasets"]:
        if result["dataset"] not in FEATURE_DATASETS:
            continue
        summary = result["summary"]
        get = lambda name: summary[name]["balanced_accuracy_mean"]
        delta = result["paired"]["feature_stats_final_rich_minus_feature_stats"]
        lines.append(
            f"| {result['dataset']} | {get('FEATURE_ONLY'):.3f} | {get('FEATURE_STATS'):.3f} | "
            f"{get('FEATURE_FINAL_RICH'):.3f} | {get('FEATURE_STATS_FINAL_RICH'):.3f} | "
            f"{delta['mean']:+.3f} ({delta['wins']}/{delta['ties']}/{delta['losses']}) |"
        )
    lines.extend(["", "## 5. 结论", ""])
    label = payload["decision"]["classification"]
    if label == "RICH_READOUT_TRANSFERS_AS_KSVD_ROUTE":
        lines.append(
            "rich readout 同时通过读出、K-SVD update 归因和 graph-stats 外增量三个 gate；可以把它作为普通 KSVD 的下一条冻结路线。"
        )
    elif label == "RICH_STATISTICS_HELP_WITHOUT_KSVD_UPDATE_ATTRIBUTION":
        lines.append(
            "高阶 sparse-code 统计有用，但 FINAL 没有稳定优于同一 INIT；不能把增益归因于 K-SVD updates。后续若继续，应研究合法原型/patch metric，而不是增加 K-SVD 轮次。"
        )
    elif label == "RICH_STATISTICS_REPEAT_GRAPH_STATS":
        lines.append(
            "rich readout 优于 coarse，但没有越过简单 graph statistics；它主要是高维重述，不形成独立下游路线。"
        )
    else:
        lines.append(
            "MolHIV 的 rich-readout 正例没有迁移到这四个数据集。普通 adjacency-patch K-SVD 的 readout/fusion 路线到此停止；下一研究问题应转向学习 patch metric 或更稳定的 occurrence vocabulary。"
        )
    lines.extend(
        [
            "",
            "解释边界：本实验冻结现有 patch/K/T/folds；没有扫描 classifier、读出变体或网络深度。",
            "",
        ]
    )
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
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    results = []
    for dataset in args.datasets:
        print(f"[{dataset}] start", flush=True)
        result = run_dataset(
            dataset,
            dataset_root=args.dataset_root,
            patch_size=args.patch_size,
            overlap=args.overlap,
            maximum_patches=args.maximum_patches,
            retained_beam=args.retained_beam,
            n_atoms=args.n_atoms,
            sparsity=args.sparsity,
            iterations=args.iterations,
            max_train_patches=args.max_train_patches,
            split_seeds=args.split_seeds,
            n_splits=args.n_splits,
            limit=args.limit,
        )
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

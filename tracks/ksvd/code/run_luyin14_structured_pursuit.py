"""Minimal relation-conditioned structured sparse-pursuit screen for luyin14."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.model_selection import StratifiedKFold

from .data_tud import load_tud
from .run_luyin14_rich_readout import _rich_code_readouts
from .run_luyin14_route import (
    DEFAULT_DATASETS,
    DEFAULT_ROOT,
    FEATURE_DATASETS,
    PatchSet,
    PreparedGraph,
    _atomic_json,
    _atomic_text,
    _encode_patch_set,
    _fit_dictionary,
    _fit_score,
    _relation_channels,
    _sampling_summary,
    prepare_graph,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/structured_pursuit_20260812.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/STRUCTURED_PURSUIT_20260812.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_LUYIN14_STRUCTURED_PURSUIT_PROTOCOL_20260812.md"


def _relation_weights(patch_set: PatchSet) -> np.ndarray:
    chain, overlap = _relation_channels(patch_set)
    return np.maximum(chain, overlap)


def _top_indices(values: np.ndarray, count: int) -> np.ndarray:
    order = np.lexsort((np.arange(len(values), dtype=np.int64), -values))
    return np.sort(order[:count])


def _structured_pursuit(
    centered: np.ndarray,
    dictionary: np.ndarray,
    independent_codes: np.ndarray,
    weights: np.ndarray,
    *,
    sparsity: int,
    rho: float,
    rounds: int,
) -> np.ndarray:
    """Refine supports using a frozen neighbor-support prior, then refit LS."""
    atoms, patches = independent_codes.shape
    if patches <= 1 or not np.any(weights > 0):
        return independent_codes.copy()
    degree = weights.sum(axis=1)
    current = independent_codes.copy()
    correlations = np.abs(dictionary.T @ centered)
    correlations /= np.maximum(correlations.max(axis=0, keepdims=True), 1e-12)
    for _ in range(rounds):
        support = (np.abs(current) > 1e-10).astype(np.float64).T
        neighbor_support = weights @ support
        neighbor_support /= np.maximum(degree[:, None], 1e-12)
        updated = np.zeros_like(current)
        for patch in range(patches):
            if degree[patch] <= 1e-12:
                selected = np.flatnonzero(np.abs(current[:, patch]) > 1e-10)
                if selected.size != min(sparsity, atoms):
                    selected = _top_indices(correlations[:, patch], min(sparsity, atoms))
            else:
                score = (1.0 - rho) * correlations[:, patch] + rho * neighbor_support[patch]
                selected = _top_indices(score, min(sparsity, atoms))
            coefficients, *_ = np.linalg.lstsq(
                dictionary[:, selected], centered[:, patch], rcond=None
            )
            updated[selected, patch] = coefficients
        current = updated
    return current


def _support_diagnostics(
    independent: np.ndarray,
    structured: np.ndarray,
    weights: np.ndarray,
    centered: np.ndarray,
    dictionary: np.ndarray,
) -> dict[str, float]:
    def agreement(codes: np.ndarray) -> float:
        support = np.abs(codes) > 1e-10
        left, right = np.nonzero(np.triu(weights, k=1) > 0)
        if left.size == 0:
            return 0.0
        edge_weights = weights[left, right]
        similarities = []
        for a, b in zip(left, right):
            union = np.sum(support[:, a] | support[:, b])
            similarities.append(
                np.sum(support[:, a] & support[:, b]) / max(int(union), 1)
            )
        return float(np.average(similarities, weights=edge_weights))

    independent_support = np.abs(independent) > 1e-10
    structured_support = np.abs(structured) > 1e-10
    independent_error = np.linalg.norm(centered - dictionary @ independent)
    structured_error = np.linalg.norm(centered - dictionary @ structured)
    return {
        "independent_agreement": agreement(independent),
        "structured_agreement": agreement(structured),
        "support_change_fraction": float(
            np.mean(np.any(independent_support != structured_support, axis=0))
        ),
        "reconstruction_relative_change": float(
            structured_error / max(independent_error, 1e-12) - 1.0
        ),
    }


def _feature_matrix(
    prepared: Sequence[PreparedGraph],
    indices: Sequence[int],
    dictionary: dict[str, Any],
    *,
    sparsity: int,
    rho: float,
    rounds: int,
    shuffle_seed: int,
) -> tuple[dict[str, np.ndarray], list[dict[str, float]]]:
    names = [
        "STATS",
        "INIT_INDEPENDENT_RICH",
        "INIT_TRUE_STRUCTURED_RICH",
        "FINAL_INDEPENDENT_RICH",
        "FINAL_TRUE_STRUCTURED_RICH",
        "FINAL_SHUFFLED_STRUCTURED_RICH",
        "STATS_FINAL_TRUE_STRUCTURED_RICH",
        "FEATURE_ONLY",
        "FEATURE_STATS",
        "FEATURE_STATS_FINAL_TRUE_STRUCTURED_RICH",
    ]
    rows = {name: [] for name in names}
    diagnostics = []
    for index in indices:
        item = prepared[int(index)]
        centered = item.fair.vectors.T - dictionary["mean"]
        weights = _relation_weights(item.fair)
        rng = np.random.default_rng(shuffle_seed + item.index * 1009)
        permutation = rng.permutation(weights.shape[0])
        shuffled_weights = weights[np.ix_(permutation, permutation)]
        init_independent = _encode_patch_set(
            item.fair, dictionary["initial"], dictionary["mean"], sparsity
        )
        final_independent = _encode_patch_set(
            item.fair, dictionary["final"], dictionary["mean"], sparsity
        )
        init_true = _structured_pursuit(
            centered,
            dictionary["initial"],
            init_independent,
            weights,
            sparsity=sparsity,
            rho=rho,
            rounds=rounds,
        )
        final_true = _structured_pursuit(
            centered,
            dictionary["final"],
            final_independent,
            weights,
            sparsity=sparsity,
            rho=rho,
            rounds=rounds,
        )
        final_shuffled = _structured_pursuit(
            centered,
            dictionary["final"],
            final_independent,
            shuffled_weights,
            sparsity=sparsity,
            rho=rho,
            rounds=rounds,
        )
        init_independent_rich = _rich_code_readouts(
            init_independent, centered, dictionary["initial"]
        )["rich"]
        init_true_rich = _rich_code_readouts(
            init_true, centered, dictionary["initial"]
        )["rich"]
        final_independent_rich = _rich_code_readouts(
            final_independent, centered, dictionary["final"]
        )["rich"]
        final_true_rich = _rich_code_readouts(
            final_true, centered, dictionary["final"]
        )["rich"]
        final_shuffled_rich = _rich_code_readouts(
            final_shuffled, centered, dictionary["final"]
        )["rich"]
        rows["STATS"].append(item.stats)
        rows["INIT_INDEPENDENT_RICH"].append(init_independent_rich)
        rows["INIT_TRUE_STRUCTURED_RICH"].append(init_true_rich)
        rows["FINAL_INDEPENDENT_RICH"].append(final_independent_rich)
        rows["FINAL_TRUE_STRUCTURED_RICH"].append(final_true_rich)
        rows["FINAL_SHUFFLED_STRUCTURED_RICH"].append(final_shuffled_rich)
        rows["STATS_FINAL_TRUE_STRUCTURED_RICH"].append(
            np.concatenate([item.stats, final_true_rich])
        )
        if item.node_features is not None:
            feature_stats = np.concatenate([item.node_features, item.stats])
            rows["FEATURE_ONLY"].append(item.node_features)
            rows["FEATURE_STATS"].append(feature_stats)
            rows["FEATURE_STATS_FINAL_TRUE_STRUCTURED_RICH"].append(
                np.concatenate([feature_stats, final_true_rich])
            )
        diagnostics.append(
            _support_diagnostics(
                final_independent, final_true, weights, centered, dictionary["final"]
            )
        )
    return (
        {name: np.stack(values) for name, values in rows.items() if values},
        diagnostics,
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
    names = sorted(folds[0]["scores"])
    return {
        name: {
            "balanced_accuracy_mean": float(
                np.mean([fold["scores"][name]["balanced_accuracy"] for fold in folds])
            ),
            "balanced_accuracy_std": float(
                np.std([fold["scores"][name]["balanced_accuracy"] for fold in folds])
            ),
        }
        for name in names
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
            dictionary_seed = split_seed * 100 + fold_index
            dictionary = _fit_dictionary(
                prepared,
                train_indices,
                branch="fair",
                n_atoms=args.n_atoms,
                sparsity=args.sparsity,
                iterations=args.iterations,
                max_train_patches=args.max_train_patches,
                seed=dictionary_seed,
            )
            train_features, train_diagnostics = _feature_matrix(
                prepared,
                train_indices,
                dictionary,
                sparsity=args.sparsity,
                rho=args.rho,
                rounds=args.structured_rounds,
                shuffle_seed=731421 + dictionary_seed,
            )
            test_features, test_diagnostics = _feature_matrix(
                prepared,
                test_indices,
                dictionary,
                sparsity=args.sparsity,
                rho=args.rho,
                rounds=args.structured_rounds,
                shuffle_seed=731421 + dictionary_seed,
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
                            seed=dictionary_seed,
                        )
                        for feature_name in train_features
                    },
                    "test_diagnostics": {
                        key: float(np.mean([row[key] for row in test_diagnostics]))
                        for key in test_diagnostics[0]
                    },
                    "train_diagnostics": {
                        key: float(np.mean([row[key] for row in train_diagnostics]))
                        for key in train_diagnostics[0]
                    },
                }
            )
    paired = {
        "true_minus_independent": _paired(
            folds, "FINAL_TRUE_STRUCTURED_RICH", "FINAL_INDEPENDENT_RICH"
        ),
        "true_minus_shuffled": _paired(
            folds, "FINAL_TRUE_STRUCTURED_RICH", "FINAL_SHUFFLED_STRUCTURED_RICH"
        ),
        "final_true_minus_init_true": _paired(
            folds, "FINAL_TRUE_STRUCTURED_RICH", "INIT_TRUE_STRUCTURED_RICH"
        ),
        "stats_true_minus_stats": _paired(
            folds, "STATS_FINAL_TRUE_STRUCTURED_RICH", "STATS"
        ),
    }
    if name in FEATURE_DATASETS:
        paired["feature_stats_true_minus_feature_stats"] = _paired(
            folds,
            "FEATURE_STATS_FINAL_TRUE_STRUCTURED_RICH",
            "FEATURE_STATS",
        )
    diagnostics = {
        key: float(np.mean([fold["test_diagnostics"][key] for fold in folds]))
        for key in folds[0]["test_diagnostics"]
    }
    return {
        "dataset": name,
        "metadata": metadata,
        "sampling": _sampling_summary(prepared),
        "summary": _aggregate(folds),
        "paired": paired,
        "diagnostics": diagnostics,
        "folds": folds,
        "seconds": time.time() - started,
    }


def _pass(pair: dict[str, Any]) -> bool:
    return pair["mean"] >= 0.01 and pair["wins"] >= 6


def classify(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    keys = {
        "vs_independent": "true_minus_independent",
        "vs_shuffled": "true_minus_shuffled",
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
    diagnostic_gate = all(
        result["diagnostics"]["structured_agreement"]
        > result["diagnostics"]["independent_agreement"]
        and result["diagnostics"]["reconstruction_relative_change"] <= 0.05
        for result in results
    )
    if all(gates.values()) and diagnostic_gate:
        label = "STRUCTURED_PURSUIT_ADVANCES_TO_MATCHED_CONTROLS"
    elif gates["vs_independent"] and not gates["vs_shuffled"]:
        label = "GENERIC_SUPPORT_REGULARIZATION_ONLY"
    elif gates["vs_shuffled"] and not gates["vs_independent"]:
        label = "RELATION_SIGNAL_PRESENT_BUT_PURSUIT_BIAS_HARMFUL"
    else:
        label = "STRUCTURED_PURSUIT_NO_GO"
    return {
        "classification": label,
        "gates": gates,
        "diagnostic_gate": diagnostic_gate,
        "axes": axes,
    }


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# luyin14 relation-conditioned structured pursuit screen",
        "",
        "> 日期：2026-08-12  ",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{payload['decision']['classification']}`",
        "",
        "## 1. Gate",
        "",
        f"- TRUE structured > independent：`{payload['decision']['gates']['vs_independent']}`；",
        f"- TRUE structured > SHUFFLED structured：`{payload['decision']['gates']['vs_shuffled']}`；",
        f"- STATS+TRUE structured > STATS：`{payload['decision']['gates']['added_value']}`；",
        f"- support/reconstruction diagnostic：`{payload['decision']['diagnostic_gate']}`。",
        "",
        "## 2. Balanced accuracy 与 paired delta",
        "",
        "| dataset | independent | TRUE | SHUFFLED | TRUE-independent | W/T/L | TRUE-SHUFFLED | W/T/L | STATS+TRUE-STATS | W/T/L |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["datasets"]:
        summary = result["summary"]
        get = lambda name: summary[name]["balanced_accuracy_mean"]
        pairs = [
            result["paired"]["true_minus_independent"],
            result["paired"]["true_minus_shuffled"],
            result["paired"]["stats_true_minus_stats"],
        ]
        cells = []
        for pair in pairs:
            cells.extend(
                [f"{pair['mean']:+.3f}", f"{pair['wins']}/{pair['ties']}/{pair['losses']}"]
            )
        lines.append(
            f"| {result['dataset']} | {get('FINAL_INDEPENDENT_RICH'):.3f} | "
            f"{get('FINAL_TRUE_STRUCTURED_RICH'):.3f} | "
            f"{get('FINAL_SHUFFLED_STRUCTURED_RICH'):.3f} | "
            + " | ".join(cells)
            + " |"
        )
    lines.extend(
        [
            "",
            "## 3. 编码机制诊断",
            "",
            "| dataset | support agreement independent→TRUE | changed patches | recon relative change | FINAL TRUE-INIT TRUE |",
            "|---|---:|---:|---:|---:|",
        ]
    )
    for result in payload["datasets"]:
        diag = result["diagnostics"]
        learning = result["paired"]["final_true_minus_init_true"]
        lines.append(
            f"| {result['dataset']} | {diag['independent_agreement']:.3f}→{diag['structured_agreement']:.3f} | "
            f"{diag['support_change_fraction']:.3f} | {diag['reconstruction_relative_change']:+.3f} | "
            f"{learning['mean']:+.3f} ({learning['wins']}/{learning['ties']}/{learning['losses']}) |"
        )
    lines.extend(["", "## 4. 结论", ""])
    label = payload["decision"]["classification"]
    if label == "STRUCTURED_PURSUIT_ADVANCES_TO_MATCHED_CONTROLS":
        lines.append(
            "真实关系在 pursuit 阶段产生稳定、超出 shuffled 和 graph stats 的增益；下一步才运行 matched PCA/random 与更正式的 structured solver。"
        )
    elif label == "GENERIC_SUPPORT_REGULARIZATION_ONLY":
        lines.append(
            "support 平滑可能有益，但 TRUE 不胜 SHUFFLED，不能归因于真实 patch 关系。"
        )
    elif label == "RELATION_SIGNAL_PRESENT_BUT_PURSUIT_BIAS_HARMFUL":
        lines.append(
            "真实绑定优于 shuffled，但当前 pursuit 相对独立 OMP 有害；不扫描 rho，停止此实现。"
        )
    else:
        lines.append(
            "固定 structured pursuit 未建立可迁移增益。结合 rich readout 与 occurrence 证据，普通 adjacency-patch KSVD 的下游路线应停止。"
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
    parser.add_argument("--split-seeds", nargs="+", type=int, default=[0, 1, 2])
    parser.add_argument("--n-splits", type=int, default=3)
    parser.add_argument("--rho", type=float, default=0.25)
    parser.add_argument("--structured-rounds", type=int, default=2)
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


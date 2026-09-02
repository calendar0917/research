"""Compact Beam8 relation diagnostic after the frozen s8/o2 Stage A."""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_tud import load_tud
from .run_beam8_nci1_chain_classification import (
    DEFAULT_ROOT,
    ROOT,
    _atomic_json,
    _atomic_text,
    _encode,
    _fit_dictionary,
    _graph_token_feature,
    _normalize_patch_tokens,
    _relation_graph_summary,
    _relation_matrices,
    _shuffle_permutation,
    _moments,
    prepare_graph,
)


PROTOCOL = "tracks/ksvd/docs/KSVD_BEAM8_NCI1_COMPACT_RELATION_FOLLOWUP_20260813.md"
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/beam8_nci1_compact_relation_followup_20260813.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/BEAM8_NCI1_COMPACT_RELATION_FOLLOWUP_20260813.md"
S8O2_JSON = ROOT / "tracks/ksvd/results/luyin14/beam8_nci1_s8o2_chain_stage_a_20260813.json"
VARIANTS = (
    "RAW_BAG",
    "RAW_COMPACT_TRUE",
    "RAW_COMPACT_SHUFFLED",
    "INIT_BAG",
    "INIT_COMPACT_TRUE",
    "FINAL_BAG",
    "FINAL_COMPACT_TRUE",
    "FINAL_COMPACT_SHUFFLED",
)


def _weighted_mean_std(values: np.ndarray, weights: np.ndarray) -> tuple[float, float]:
    normalizer = max(float(weights.sum()), 1e-12)
    mean = float(weights @ values / normalizer)
    variance = float(weights @ ((values - mean) ** 2) / normalizer)
    return mean, float(np.sqrt(max(variance, 0.0)))


def _compact_channel(values: np.ndarray, weights: np.ndarray) -> np.ndarray:
    target, source = np.nonzero(weights > 0)
    if target.size == 0:
        return np.zeros(10, dtype=np.float64)
    pair_weights = weights[target, source]
    left = values[target]
    right = values[source]
    dot = np.sum(left * right, axis=1)
    left_norm = np.linalg.norm(left, axis=1)
    right_norm = np.linalg.norm(right, axis=1)
    cosine = dot / np.maximum(left_norm * right_norm, 1e-12)
    l1 = np.mean(np.abs(left - right), axis=1)
    normalized_dot = dot / max(values.shape[1], 1)
    norm_ratio = np.log((left_norm + 1e-6) / (right_norm + 1e-6))
    winner = (np.argmax(left, axis=1) == np.argmax(right, axis=1)).astype(np.float64)
    cosine_mean, cosine_std = _weighted_mean_std(cosine, pair_weights)
    l1_mean, l1_std = _weighted_mean_std(l1, pair_weights)
    dot_mean, _dot_std = _weighted_mean_std(normalized_dot, pair_weights)
    ratio_mean, ratio_std = _weighted_mean_std(norm_ratio, pair_weights)
    winner_mean, _winner_std = _weighted_mean_std(winner, pair_weights)
    return np.asarray(
        [
            cosine_mean,
            cosine_std,
            float(np.max(cosine)),
            l1_mean,
            l1_std,
            float(np.max(l1)),
            dot_mean,
            ratio_mean,
            ratio_std,
            winner_mean,
        ],
        dtype=np.float64,
    )


def _compact_feature(item: Any, tokens: np.ndarray, permutation: np.ndarray | None = None) -> np.ndarray:
    values = tokens if permutation is None else tokens[permutation]
    common = np.concatenate(
        [
            _moments(values),
            _moments(item.position_features),
            _relation_graph_summary(item),
            item.residual_features,
        ]
    )
    previous, _following, overlap, slot = _relation_matrices(item)
    compact = np.concatenate(
        [_compact_channel(values, weights) for weights in (previous, overlap, slot)]
    )
    return np.concatenate([common, compact])


def _fit_score(
    train: np.ndarray,
    train_labels: np.ndarray,
    test: np.ndarray,
    test_labels: np.ndarray,
    *,
    seed: int,
) -> dict[str, float]:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=1.0, random_state=seed),
    )
    model.fit(train, train_labels)

    def score(values: np.ndarray, labels: np.ndarray) -> dict[str, float]:
        prediction = model.predict(values)
        return {
            "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
            "accuracy": float(accuracy_score(labels, prediction)),
        }

    return {"train": score(train, train_labels), "test": score(test, test_labels)}


def _matrices(
    prepared: Sequence[Any],
    train_indices: np.ndarray,
    dictionary: dict[str, Any],
    *,
    sparsity: int,
    shuffle_seed: int,
) -> dict[str, np.ndarray]:
    raw = [
        np.concatenate([item.vectors, item.node_histograms], axis=1)
        for item in prepared
    ]
    init = [
        np.concatenate(
            [_encode(item, dictionary["initial"], dictionary["mean"], sparsity), item.node_histograms],
            axis=1,
        )
        for item in prepared
    ]
    final = [
        np.concatenate(
            [_encode(item, dictionary["final"], dictionary["mean"], sparsity), item.node_histograms],
            axis=1,
        )
        for item in prepared
    ]
    families = {
        "RAW": _normalize_patch_tokens(raw, train_indices),
        "INIT": _normalize_patch_tokens(init, train_indices),
        "FINAL": _normalize_patch_tokens(final, train_indices),
    }
    rows = {variant: [] for variant in VARIANTS}
    for graph_index, item in enumerate(prepared):
        for family, tokens in families.items():
            values = tokens[graph_index]
            rows[f"{family}_BAG"].append(
                _graph_token_feature(item, values, chain=False)
            )
            rows[f"{family}_COMPACT_TRUE"].append(_compact_feature(item, values))
            shuffled_name = f"{family}_COMPACT_SHUFFLED"
            if shuffled_name in rows:
                permutation = _shuffle_permutation(
                    len(values), seed=shuffle_seed + item.index * 1009
                )
                rows[shuffled_name].append(
                    _compact_feature(item, values, permutation=permutation)
                )
    return {name: np.stack(values) for name, values in rows.items()}


def _paired(folds: Sequence[dict[str, Any]], left: str, right: str) -> dict[str, Any]:
    values = np.asarray(
        [
            fold["scores"][left]["test"]["balanced_accuracy"]
            - fold["scores"][right]["test"]["balanced_accuracy"]
            for fold in folds
        ]
    )
    return {
        "mean": float(values.mean()),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
        "values": values.tolist(),
    }


def _summarize(folds: Sequence[dict[str, Any]], wide_path: Path) -> dict[str, Any]:
    variants = {}
    for variant in VARIANTS:
        test = [fold["scores"][variant]["test"]["balanced_accuracy"] for fold in folds]
        train = [fold["scores"][variant]["train"]["balanced_accuracy"] for fold in folds]
        variants[variant] = {
            "test_balanced_accuracy_mean": float(np.mean(test)),
            "test_balanced_accuracy_std": float(np.std(test)),
            "train_balanced_accuracy_mean": float(np.mean(train)),
            "generalization_gap": float(np.mean(train) - np.mean(test)),
        }
    paired = {
        "raw_true_vs_shuffled": _paired(folds, "RAW_COMPACT_TRUE", "RAW_COMPACT_SHUFFLED"),
        "raw_true_vs_bag": _paired(folds, "RAW_COMPACT_TRUE", "RAW_BAG"),
        "final_true_vs_shuffled": _paired(folds, "FINAL_COMPACT_TRUE", "FINAL_COMPACT_SHUFFLED"),
        "final_true_vs_bag": _paired(folds, "FINAL_COMPACT_TRUE", "FINAL_BAG"),
        "final_vs_init_true": _paired(folds, "FINAL_COMPACT_TRUE", "INIT_COMPACT_TRUE"),
    }
    wide = json.loads(wide_path.read_text(encoding="utf-8"))
    wide_true_vs_bag = float(wide["summary"]["paired"]["final_true_vs_bag"]["mean"])
    compact_true_vs_bag = paired["final_true_vs_bag"]
    checks = {
        "compact_binding": bool(
            paired["final_true_vs_shuffled"]["mean"] >= 0.01
            and paired["final_true_vs_shuffled"]["wins"] >= 2
        ),
        "compact_incremental": bool(
            compact_true_vs_bag["mean"] >= -1e-12
            and compact_true_vs_bag["losses"] <= 1
        ),
        "capacity_diagnosis": bool(
            compact_true_vs_bag["mean"] - wide_true_vs_bag >= 0.005
        ),
    }
    if all(checks.values()):
        decision = "COMPACT_RELATION_PASS_ONE_LAYER_PATCH_GNN_ALLOWED"
    elif checks["compact_binding"]:
        decision = "RELATION_DETECTABLE_NO_LINEAR_INCREMENT"
    else:
        decision = "COMPACT_RELATION_BELOW_GATE"
    return {
        "variants": variants,
        "paired": paired,
        "wide_final_true_vs_bag": wide_true_vs_bag,
        "compact_minus_wide_true_vs_bag": compact_true_vs_bag["mean"] - wide_true_vs_bag,
        "checks": checks,
        "decision": decision,
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    graphs, labels, features, metadata = load_tud(args.dataset, args.dataset_root)
    prepared = []
    for index, (graph, label, node_features) in enumerate(zip(graphs, labels, features)):
        prepared.append(
            prepare_graph(
                index,
                graph,
                int(label),
                node_features,
                patch_size=8,
                overlap=2,
                retained_beam=8,
                edge_capacity_multiplier=1.5,
                seed=20260813,
            )
        )
        if (index + 1) % 500 == 0 or index + 1 == len(graphs):
            print(f"cover {index + 1}/{len(graphs)}", flush=True)
    splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=0)
    folds = []
    for fold_index, (train_indices, test_indices) in enumerate(
        splitter.split(np.zeros(len(labels)), labels)
    ):
        dictionary = _fit_dictionary(
            prepared,
            train_indices,
            n_atoms=24,
            sparsity=3,
            iterations=5,
            max_train_patches=3000,
            seed=fold_index,
        )
        matrices = _matrices(
            prepared,
            train_indices,
            dictionary,
            sparsity=3,
            shuffle_seed=731421 + fold_index,
        )
        scores = {
            variant: _fit_score(
                values[train_indices], labels[train_indices],
                values[test_indices], labels[test_indices], seed=fold_index,
            )
            for variant, values in matrices.items()
        }
        folds.append({"fold_index": fold_index, "scores": scores})
        print(
            f"fold {fold_index}: FINAL_COMPACT={scores['FINAL_COMPACT_TRUE']['test']['balanced_accuracy']:.4f} "
            f"SHUFFLED={scores['FINAL_COMPACT_SHUFFLED']['test']['balanced_accuracy']:.4f} "
            f"BAG={scores['FINAL_BAG']['test']['balanced_accuracy']:.4f}",
            flush=True,
        )
    return {
        "protocol": PROTOCOL,
        "dataset": metadata,
        "folds": folds,
        "summary": _summarize(folds, args.wide_json),
        "seconds": time.time() - started,
    }


def render_report(payload: dict[str, Any]) -> str:
    summary = payload["summary"]
    lines = [
        "# Beam8/NCI1 compact-relation follow-up",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{summary['decision']}`",
        "",
        "| variant | train bacc | test bacc | gap |",
        "|---|---:|---:|---:|",
    ]
    for variant in VARIANTS:
        row = summary["variants"][variant]
        lines.append(
            f"| {variant} | {row['train_balanced_accuracy_mean']:.4f} | {row['test_balanced_accuracy_mean']:.4f} ± {row['test_balanced_accuracy_std']:.4f} | {row['generalization_gap']:.4f} |"
        )
    lines.extend(["", "## Paired attribution", "", "| comparison | mean | W/T/L |", "|---|---:|---:|"])
    for name, row in summary["paired"].items():
        lines.append(f"| {name} | {row['mean']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} |")
    lines.extend(
        [
            "",
            f"- wide FINAL TRUE−BAG：`{summary['wide_final_true_vs_bag']:+.4f}`；",
            f"- compact 相对 wide 的 TRUE−BAG 改善：`{summary['compact_minus_wide_true_vs_bag']:+.4f}`；",
            "",
            "## Frozen checks",
            "",
        ]
    )
    for name, value in summary["checks"].items():
        lines.append(f"- {name}：`{value}`；")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- 本轮是在 s8/o2 结果可见后注册的容量诊断，不是新的独立确认实验。",
            "- compact 仍不能超过 BAG 时，不继续扫描手工关系统计量。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default="NCI1")
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--wide-json", type=Path, default=S8O2_JSON)
    parser.add_argument("--json", type=Path, default=DEFAULT_JSON)
    parser.add_argument("--report", type=Path, default=DEFAULT_REPORT)
    args = parser.parse_args()
    payload = run(args)
    _atomic_json(args.json, payload)
    _atomic_text(args.report, render_report(payload))
    print(args.report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


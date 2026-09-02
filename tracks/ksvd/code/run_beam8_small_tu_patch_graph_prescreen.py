"""Low-capacity Beam8 patch-graph prescreen on BZR, COX2, and DHFR."""
from __future__ import annotations

import argparse
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
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_beam8_nci1_chain_classification import (
    ROOT,
    _atomic_json,
    _atomic_text,
    _graph_token_feature,
    _normalize_patch_tokens,
    _relation_matrices,
    _shuffle_permutation,
)
from .run_beam8_nci1_compact_relation_followup import _compact_feature
from .run_enzymes_beam8_conditional_controls import DEFAULT_ROOT, _binary_typed
from .run_enzymes_beam8_invariance import _graph


PROTOCOL = "tracks/ksvd/docs/KSVD_BEAM8_SMALL_TU_PATCH_GRAPH_PRESCREEN_PROTOCOL_20260814.md"
RESULT_DIR = ROOT / "tracks/ksvd/results/luyin14"
DEFAULT_JSON = RESULT_DIR / "beam8_small_tu_patch_graph_prescreen_20260814.json"
DEFAULT_REPORT = RESULT_DIR / "BEAM8_SMALL_TU_PATCH_GRAPH_PRESCREEN_20260814.md"
DATASETS = ("BZR", "COX2", "DHFR")
SPLIT_SEEDS = (0, 1, 2)
VARIANTS = (
    "GLOBAL_STATS",
    "RAW_BAG",
    "RAW_PATCH_GRAPH_TRUE",
    "RAW_PATCH_GRAPH_TOKEN_SHUFFLED",
)


def _compact_discrete_features(
    features: Sequence[np.ndarray],
) -> tuple[list[np.ndarray], dict[int, int]]:
    raw_types = [np.argmax(np.asarray(values), axis=1).astype(np.int64) for values in features]
    active = sorted(set(int(value) for row in raw_types for value in row))
    mapping = {value: index for index, value in enumerate(active)}
    compact = []
    for row in raw_types:
        mapped = np.asarray([mapping[int(value)] for value in row], dtype=np.int64)
        one_hot = np.zeros((len(mapped), len(active)), dtype=np.float64)
        one_hot[np.arange(len(mapped)), mapped] = 1.0
        compact.append(one_hot)
    return compact, mapping


def _prepare(
    name: str, dataset_root: Path
) -> tuple[list[Any], list[Any], np.ndarray, list[np.ndarray], dict[str, Any]]:
    graphs, labels, raw_features, metadata = load_tud(
        name, dataset_root, use_node_attr=False
    )
    if any(values is None for values in raw_features):
        raise RuntimeError(f"{name} is missing discrete node labels")
    features, mapping = _compact_discrete_features(
        [np.asarray(values) for values in raw_features]
    )
    items = []
    typed_graphs = []
    for index, (graph, label, values) in enumerate(zip(graphs, labels, features)):
        typed = _binary_typed(graph)
        typed_graphs.append(typed)
        node_types = np.argmax(values, axis=1).astype(np.int64)
        items.append(
            prepare_attributed_beam_graph(
                index,
                graph,
                int(label),
                values,
                typed,
                patch_size=8,
                overlap=2,
                retained_beam=8,
                edge_capacity_multiplier=1.5,
                seed=20260814,
                edge_dim=1,
                canonical_node_features=values,
                canonical_node_types=node_types,
            )
        )
        if (index + 1) % 200 == 0 or index + 1 == len(graphs):
            print(f"{name} prepare {index + 1}/{len(graphs)}", flush=True)
    metadata = metadata | {
        "active_discrete_node_labels": len(mapping),
        "raw_to_compact_node_label": {str(key): value for key, value in mapping.items()},
    }
    return graphs, items, labels, features, metadata


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
        LogisticRegression(
            max_iter=5000,
            C=1.0,
            class_weight="balanced",
            random_state=seed,
        ),
    )
    model.fit(train, train_labels)
    prediction = model.predict(test)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(test_labels, prediction)),
        "accuracy": float(accuracy_score(test_labels, prediction)),
    }


def _matrices(
    items: Sequence[Any],
    train: np.ndarray,
    *,
    shuffle_seed: int,
) -> dict[str, np.ndarray]:
    normalized = _normalize_patch_tokens(
        [np.asarray(item.vectors, dtype=np.float64) for item in items], train
    )
    rows = {variant: [] for variant in VARIANTS}
    for item, tokens in zip(items, normalized):
        rows["GLOBAL_STATS"].append(
            np.concatenate([item.graph_features, item.stats])
        )
        rows["RAW_BAG"].append(_graph_token_feature(item, tokens, chain=False))
        rows["RAW_PATCH_GRAPH_TRUE"].append(_compact_feature(item, tokens))
        permutation = _shuffle_permutation(
            len(tokens), seed=shuffle_seed + item.index * 1009
        )
        rows["RAW_PATCH_GRAPH_TOKEN_SHUFFLED"].append(
            _compact_feature(item, tokens, permutation=permutation)
        )
    return {name: np.stack(values) for name, values in rows.items()}


def _audit(
    graphs: Sequence[Any],
    items: Sequence[Any],
    labels: np.ndarray,
    features: Sequence[np.ndarray],
    *,
    graph_count: int = 32,
    permutations: int = 2,
) -> dict[str, Any]:
    trials = []
    count = min(graph_count, len(graphs))
    for index in range(count):
        graph = graphs[index]
        base = items[index]
        values = np.asarray(features[index])
        typed = _binary_typed(graph)
        node_types = np.argmax(values, axis=1).astype(np.int64)
        base_tokens = np.asarray(base.vectors, dtype=np.float64)
        base_relations = _relation_matrices(base)
        base_true = _compact_feature(base, base_tokens)
        shuffle_seed = 8675309 + index * 1009
        base_shuffled = _compact_feature(
            base,
            base_tokens,
            permutation=_shuffle_permutation(len(base_tokens), seed=shuffle_seed),
        )
        for trial in range(permutations):
            permutation = np.random.default_rng(
                20260814 + index * 1009 + trial
            ).permutation(graph.n)
            rel_typed = typed[np.ix_(permutation, permutation)]
            rel_values = values[permutation]
            rel_types = node_types[permutation]
            rel = prepare_attributed_beam_graph(
                index,
                _graph(rel_typed),
                int(labels[index]),
                rel_values,
                rel_typed,
                patch_size=8,
                overlap=2,
                retained_beam=8,
                edge_capacity_multiplier=1.5,
                seed=20260814,
                edge_dim=1,
                canonical_node_features=rel_values,
                canonical_node_types=rel_types,
            )
            rel_tokens = np.asarray(rel.vectors, dtype=np.float64)
            rel_relations = _relation_matrices(rel)
            rel_true = _compact_feature(rel, rel_tokens)
            rel_shuffled = _compact_feature(
                rel,
                rel_tokens,
                permutation=_shuffle_permutation(len(rel_tokens), seed=shuffle_seed),
            )
            mapped_sets = tuple(
                frozenset(int(permutation[node]) for node in nodes)
                for nodes in rel.node_sets
            )
            trials.append(
                {
                    "graph_index": index,
                    "permutation": trial,
                    "chain_exact_match": mapped_sets == base.node_sets,
                    "token_row_match": base_tokens.shape == rel_tokens.shape
                    and np.allclose(base_tokens, rel_tokens, atol=1e-12, rtol=0),
                    "relation_matrix_match": all(
                        left.shape == right.shape
                        and np.allclose(left, right, atol=1e-12, rtol=0)
                        for left, right in zip(base_relations, rel_relations)
                    ),
                    "true_feature_match": base_true.shape == rel_true.shape
                    and np.allclose(base_true, rel_true, atol=1e-12, rtol=0),
                    "shuffled_feature_match": base_shuffled.shape == rel_shuffled.shape
                    and np.allclose(
                        base_shuffled, rel_shuffled, atol=1e-12, rtol=0
                    ),
                }
            )
    keys = (
        "chain_exact_match",
        "token_row_match",
        "relation_matrix_match",
        "true_feature_match",
        "shuffled_feature_match",
    )
    rates = {key: float(np.mean([row[key] for row in trials])) for key in keys}
    required = keys[1:]
    return {
        "graphs": count,
        "permutations": permutations,
        "rates": rates,
        "passed": all(rates[key] == 1.0 for key in required),
        "failure_examples": {
            key: [row for row in trials if not row[key]][:10] for key in keys
        },
    }


def _delta(
    units: Sequence[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
    values = np.asarray(
        [unit["scores"][left]["balanced_accuracy"] - unit["scores"][right]["balanced_accuracy"] for unit in units]
    )
    return {
        "mean": float(values.mean()),
        "wins": int(np.sum(values > 1e-12)),
        "ties": int(np.sum(np.abs(values) <= 1e-12)),
        "losses": int(np.sum(values < -1e-12)),
        "split_means": {
            str(seed): float(
                np.mean(
                    [value for value, unit in zip(values, units) if unit["split_seed"] == seed]
                )
            )
            for seed in SPLIT_SEEDS
        },
        "values": values.tolist(),
    }


def _run_dataset(name: str, dataset_root: Path) -> dict[str, Any]:
    graphs, items, labels, features, metadata = _prepare(name, dataset_root)
    audit = _audit(graphs, items, labels, features)
    units = []
    for split_seed in SPLIT_SEEDS:
        splitter = StratifiedKFold(n_splits=3, shuffle=True, random_state=split_seed)
        for fold, (train, test) in enumerate(
            splitter.split(np.zeros(len(labels)), labels)
        ):
            matrices = _matrices(
                items,
                train,
                shuffle_seed=731421 + split_seed * 10000 + fold,
            )
            scores = {
                variant: _fit_score(
                    values[train],
                    labels[train],
                    values[test],
                    labels[test],
                    seed=split_seed * 1000 + fold,
                )
                for variant, values in matrices.items()
            }
            units.append(
                {
                    "split_seed": split_seed,
                    "fold_index": fold,
                    "scores": scores,
                }
            )
            print(
                f"{name} split={split_seed} fold={fold} "
                + " ".join(
                    f"{variant}={scores[variant]['balanced_accuracy']:.4f}"
                    for variant in VARIANTS
                ),
                flush=True,
            )
    variants = {
        variant: {
            "balanced_accuracy_mean": float(
                np.mean([unit["scores"][variant]["balanced_accuracy"] for unit in units])
            ),
            "balanced_accuracy_std": float(
                np.std([unit["scores"][variant]["balanced_accuracy"] for unit in units])
            ),
            "accuracy_mean": float(
                np.mean([unit["scores"][variant]["accuracy"] for unit in units])
            ),
        }
        for variant in VARIANTS
    }
    paired = {
        "true_vs_token_shuffled": _delta(
            units, "RAW_PATCH_GRAPH_TRUE", "RAW_PATCH_GRAPH_TOKEN_SHUFFLED"
        ),
        "true_vs_bag": _delta(units, "RAW_PATCH_GRAPH_TRUE", "RAW_BAG"),
        "true_vs_global_stats": _delta(
            units, "RAW_PATCH_GRAPH_TRUE", "GLOBAL_STATS"
        ),
    }
    binding = paired["true_vs_token_shuffled"]
    increment = paired["true_vs_bag"]
    global_delta = paired["true_vs_global_stats"]
    checks = {
        "binding": binding["mean"] >= 0.01 and binding["wins"] >= 6,
        "relation_increment": increment["mean"] >= 0.005
        and increment["wins"] >= 6,
        "beats_global_stats": global_delta["mean"] >= 0
        and global_delta["wins"] >= 5,
        "all_split_binding_positive": all(
            value > 0 for value in binding["split_means"].values()
        ),
        "all_split_increment_positive": all(
            value > 0 for value in increment["split_means"].values()
        ),
        "invariance": bool(audit["passed"]),
    }
    decision = (
        f"{name}_BEAM8_PATCH_GRAPH_ADVANCES_TO_ONE_LAYER_GNN"
        if all(checks.values())
        else f"{name}_BEAM8_PATCH_GRAPH_PRESCREEN_NOT_ESTABLISHED"
    )
    return {
        "dataset": metadata,
        "audit": audit,
        "units": units,
        "summary": {
            "variants": variants,
            "paired": paired,
            "checks": checks,
            "decision": decision,
        },
    }


def run(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    datasets = {
        name: _run_dataset(name, args.dataset_root) for name in DATASETS
    }
    return {
        "protocol": PROTOCOL,
        "config": {
            "datasets": list(DATASETS),
            "split_seeds": list(SPLIT_SEEDS),
            "patch_size": 8,
            "overlap": 2,
            "retained_beam": 8,
        },
        "datasets": datasets,
        "advanced": [
            name
            for name, payload in datasets.items()
            if payload["summary"]["decision"].endswith("ADVANCES_TO_ONE_LAYER_GNN")
        ],
        "seconds": time.time() - started,
    }


def render(payload: dict[str, Any]) -> str:
    lines = [
        "# Beam8 small attributed-TU patch-graph prescreen",
        "",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 晋级数据集：`{payload['advanced']}`",
    ]
    for name in DATASETS:
        dataset = payload["datasets"][name]
        summary = dataset["summary"]
        lines.extend(
            [
                "",
                f"## {name}",
                "",
                f"> 判定：`{summary['decision']}`",
                "",
                "| variant | balanced accuracy over 9 units |",
                "|---|---:|",
            ]
        )
        for variant in VARIANTS:
            row = summary["variants"][variant]
            lines.append(
                f"| {variant} | {row['balanced_accuracy_mean']:.4f} ± {row['balanced_accuracy_std']:.4f} |"
            )
        lines.extend(
            [
                "",
                "| comparison | mean | W/T/L | split0/1/2 |",
                "|---|---:|---:|---:|",
            ]
        )
        for comparison, row in summary["paired"].items():
            split_text = " / ".join(
                f"{row['split_means'][str(seed)]:+.4f}" for seed in SPLIT_SEEDS
            )
            lines.append(
                f"| {comparison} | {row['mean']:+.4f} | {row['wins']}/{row['ties']}/{row['losses']} | {split_text} |"
            )
        lines.extend(["", "### Frozen checks", ""])
        for check, value in summary["checks"].items():
            lines.append(f"- {check}：`{value}`；")
        lines.extend(["", "### Relabel audit", ""])
        for key, value in dataset["audit"]["rates"].items():
            role = "diagnostic" if key == "chain_exact_match" else "required"
            lines.append(f"- {key}：`{value:.6f}`（{role}）；")
    lines.extend(
        [
            "",
            "## Boundary",
            "",
            "- 本轮只使用官方离散 node labels；3D coordinates 不进入模型。",
            "- RAW token 不经过 KSVD，直接检查 Beam8 patch graph substrate。",
            "- 未晋级的数据集不运行可学习 patch-GNN。",
            "- 晋级者只能在新的 split3/4 上运行一层 patch-GNN，并保留 matched controls。",
        ]
    )
    return "\n".join(lines) + "\n"


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_ROOT)
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

"""Strict OOF prediction-level fusion for the four real luyin14 datasets."""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.optimize import minimize, minimize_scalar
from scipy.special import expit, softmax
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, log_loss
from sklearn.model_selection import StratifiedKFold
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from .data_tud import load_tud
from .run_luyin14_rich_readout import _feature_matrix
from .run_luyin14_route import (
    DEFAULT_DATASETS,
    DEFAULT_ROOT,
    FEATURE_DATASETS,
    PreparedGraph,
    _atomic_json,
    _atomic_text,
    _fit_dictionary,
    _sampling_summary,
    prepare_graph,
)


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_JSON = ROOT / "tracks/ksvd/results/luyin14/oof_multiview_fusion_20260812.json"
DEFAULT_REPORT = ROOT / "tracks/ksvd/results/luyin14/OOF_MULTIVIEW_FUSION_20260812.md"
PROTOCOL = "tracks/ksvd/docs/KSVD_LUYIN14_OOF_MULTIVIEW_FUSION_PROTOCOL_20260812.md"


def _centered_log_probabilities(probabilities: np.ndarray) -> np.ndarray:
    values = np.log(np.clip(probabilities, 1e-9, 1.0))
    return values - values.mean(axis=1, keepdims=True)


def _normalize_probabilities(values: np.ndarray) -> np.ndarray:
    values = np.maximum(np.asarray(values, dtype=np.float64), 1e-12)
    return values / values.sum(axis=1, keepdims=True)


def _fit_expert(
    train: np.ndarray,
    train_labels: np.ndarray,
    test: np.ndarray,
    *,
    seed: int,
    n_classes: int,
) -> np.ndarray:
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=1.0, random_state=seed),
    )
    model.fit(train, train_labels)
    raw = model.predict_proba(test)
    output = np.full((len(test), n_classes), 1e-12, dtype=np.float64)
    classes = np.asarray(model[-1].classes_, dtype=np.int64)
    output[:, classes] = raw
    return _normalize_probabilities(output)


def _score(probabilities: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    prediction = np.argmax(probabilities, axis=1)
    return {
        "balanced_accuracy": float(balanced_accuracy_score(labels, prediction)),
        "accuracy": float(accuracy_score(labels, prediction)),
        "log_loss": float(log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1]))),
    }


def _fit_scalar_fusion(
    base_oof: np.ndarray,
    structure_oof: np.ndarray,
    labels: np.ndarray,
) -> float:
    base_logits = _centered_log_probabilities(base_oof)
    structure_logits = _centered_log_probabilities(structure_oof)

    def objective(alpha: float) -> float:
        probabilities = softmax(base_logits + alpha * structure_logits, axis=1)
        return float(
            log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1]))
            + 0.01 * alpha * alpha
        )

    result = minimize_scalar(objective, bounds=(-1.0, 1.0), method="bounded")
    return float(result.x)


def _apply_scalar_fusion(
    base: np.ndarray, structure: np.ndarray, alpha: float
) -> np.ndarray:
    return softmax(
        _centered_log_probabilities(base)
        + alpha * _centered_log_probabilities(structure),
        axis=1,
    )


def _confidence(probabilities: np.ndarray) -> np.ndarray:
    classes = probabilities.shape[1]
    entropy = -np.sum(
        probabilities * np.log(np.clip(probabilities, 1e-12, 1.0)), axis=1
    )
    return 1.0 - entropy / max(np.log(classes), 1e-12)


def _js_divergence(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    middle = 0.5 * (left + right)
    return 0.5 * np.sum(
        left * np.log(np.clip(left / middle, 1e-12, None))
        + right * np.log(np.clip(right / middle, 1e-12, None)),
        axis=1,
    )


def _gate_design(base: np.ndarray, structure: np.ndarray) -> np.ndarray:
    return np.column_stack(
        [
            np.ones(len(base), dtype=np.float64),
            _confidence(structure) - _confidence(base),
            _js_divergence(base, structure),
        ]
    )


def _fit_gate_fusion(
    base_oof: np.ndarray,
    structure_oof: np.ndarray,
    labels: np.ndarray,
) -> np.ndarray:
    design = _gate_design(base_oof, structure_oof)

    def objective(parameters: np.ndarray) -> float:
        weights = expit(design @ parameters)
        probabilities = _normalize_probabilities(
            (1.0 - weights[:, None]) * base_oof
            + weights[:, None] * structure_oof
        )
        return float(
            log_loss(labels, probabilities, labels=np.arange(probabilities.shape[1]))
            + 0.01 * np.sum(np.square(parameters))
        )

    result = minimize(
        objective,
        x0=np.asarray([-2.0, 0.0, 0.0]),
        method="L-BFGS-B",
        bounds=[(-6.0, 6.0)] * 3,
    )
    return np.asarray(result.x, dtype=np.float64)


def _apply_gate_fusion(
    base: np.ndarray, structure: np.ndarray, parameters: np.ndarray
) -> tuple[np.ndarray, np.ndarray]:
    weights = expit(_gate_design(base, structure) @ parameters)
    probabilities = _normalize_probabilities(
        (1.0 - weights[:, None]) * base + weights[:, None] * structure
    )
    return probabilities, weights


def _fit_stack_fusion(
    base_oof: np.ndarray,
    structure_oof: np.ndarray,
    labels: np.ndarray,
    *,
    seed: int,
) -> Any:
    features = np.concatenate(
        [
            _centered_log_probabilities(base_oof),
            _centered_log_probabilities(structure_oof),
        ],
        axis=1,
    )
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=5000, C=0.1, random_state=seed),
    )
    model.fit(features, labels)
    return model


def _apply_stack_fusion(
    model: Any,
    base: np.ndarray,
    structure: np.ndarray,
    *,
    n_classes: int,
) -> np.ndarray:
    features = np.concatenate(
        [
            _centered_log_probabilities(base),
            _centered_log_probabilities(structure),
        ],
        axis=1,
    )
    raw = model.predict_proba(features)
    output = np.full((len(base), n_classes), 1e-12, dtype=np.float64)
    output[:, np.asarray(model[-1].classes_, dtype=np.int64)] = raw
    return _normalize_probabilities(output)


def _fusion_outputs(
    base_oof: np.ndarray,
    structure_oof: np.ndarray,
    labels: np.ndarray,
    base_test: np.ndarray,
    structure_test: np.ndarray,
    *,
    seed: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    fixed_average = 0.5 * (base_test + structure_test)
    logit_add = softmax(
        _centered_log_probabilities(base_test)
        + _centered_log_probabilities(structure_test),
        axis=1,
    )
    alpha = _fit_scalar_fusion(base_oof, structure_oof, labels)
    scalar = _apply_scalar_fusion(base_test, structure_test, alpha)
    gate_parameters = _fit_gate_fusion(base_oof, structure_oof, labels)
    gated, gate_weights = _apply_gate_fusion(
        base_test, structure_test, gate_parameters
    )
    stack = _fit_stack_fusion(
        base_oof, structure_oof, labels, seed=seed
    )
    stacked = _apply_stack_fusion(
        stack, base_test, structure_test, n_classes=base_test.shape[1]
    )
    return (
        {
            "FIXED_AVG": fixed_average,
            "LOGIT_ADD": logit_add,
            "OOF_SCALAR": scalar,
            "OOF_GATE": gated,
            "OOF_STACK": stacked,
        },
        {
            "scalar_alpha": alpha,
            "gate_parameters": gate_parameters.tolist(),
            "test_gate_weight_mean": float(np.mean(gate_weights)),
            "test_gate_weight_std": float(np.std(gate_weights)),
        },
    )


def _disagreement_summary(
    base: np.ndarray, structure: np.ndarray, labels: np.ndarray
) -> dict[str, float]:
    base_prediction = np.argmax(base, axis=1)
    structure_prediction = np.argmax(structure, axis=1)
    base_correct = base_prediction == labels
    structure_correct = structure_prediction == labels
    return {
        "prediction_disagreement_fraction": float(
            np.mean(base_prediction != structure_prediction)
        ),
        "structure_only_correct_fraction": float(
            np.mean((~base_correct) & structure_correct)
        ),
        "base_only_correct_fraction": float(
            np.mean(base_correct & (~structure_correct))
        ),
        "both_wrong_fraction": float(np.mean((~base_correct) & (~structure_correct))),
        "mean_js_divergence": float(np.mean(_js_divergence(base, structure))),
    }


def _cross_fitted_predictions(
    prepared: Sequence[PreparedGraph],
    labels: np.ndarray,
    outer_train: np.ndarray,
    *,
    base_feature: str,
    n_atoms: int,
    sparsity: int,
    iterations: int,
    max_train_patches: int,
    inner_splits: int,
    seed: int,
) -> dict[str, np.ndarray]:
    n_classes = len(np.unique(labels))
    output = {
        "BASE": np.zeros((len(outer_train), n_classes), dtype=np.float64),
        "INIT": np.zeros((len(outer_train), n_classes), dtype=np.float64),
        "FINAL": np.zeros((len(outer_train), n_classes), dtype=np.float64),
    }
    splitter = StratifiedKFold(
        n_splits=inner_splits, shuffle=True, random_state=1729 + seed
    )
    outer_labels = labels[outer_train]
    for inner_fold, (fit_local, heldout_local) in enumerate(
        splitter.split(np.zeros(len(outer_train)), outer_labels)
    ):
        fit_indices = outer_train[fit_local]
        heldout_indices = outer_train[heldout_local]
        dictionary_seed = seed * 1000 + inner_fold
        dictionary = _fit_dictionary(
            prepared,
            fit_indices,
            branch="fair",
            n_atoms=n_atoms,
            sparsity=sparsity,
            iterations=iterations,
            max_train_patches=max_train_patches,
            seed=dictionary_seed,
        )
        fit_features = _feature_matrix(
            prepared, fit_indices, dictionary, sparsity=sparsity
        )
        heldout_features = _feature_matrix(
            prepared, heldout_indices, dictionary, sparsity=sparsity
        )
        output["BASE"][heldout_local] = _fit_expert(
            fit_features[base_feature],
            labels[fit_indices],
            heldout_features[base_feature],
            seed=dictionary_seed,
            n_classes=n_classes,
        )
        output["INIT"][heldout_local] = _fit_expert(
            fit_features["INIT_RICH"],
            labels[fit_indices],
            heldout_features["INIT_RICH"],
            seed=dictionary_seed,
            n_classes=n_classes,
        )
        output["FINAL"][heldout_local] = _fit_expert(
            fit_features["FINAL_RICH"],
            labels[fit_indices],
            heldout_features["FINAL_RICH"],
            seed=dictionary_seed,
            n_classes=n_classes,
        )
    if any(not np.allclose(values.sum(axis=1), 1.0) for values in output.values()):
        raise RuntimeError("cross-fitted probabilities are incomplete")
    return output


def _paired(
    folds: Sequence[dict[str, Any]], left: str, right: str
) -> dict[str, Any]:
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
            "accuracy_mean": float(
                np.mean([fold["scores"][name]["accuracy"] for fold in folds])
            ),
            "log_loss_mean": float(
                np.mean([fold["scores"][name]["log_loss"] for fold in folds])
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
    base_feature = "FEATURE_STATS" if name in FEATURE_DATASETS else "STATS"
    n_classes = len(np.unique(labels))
    folds = []
    for split_seed in args.split_seeds:
        splitter = StratifiedKFold(
            n_splits=args.n_splits, shuffle=True, random_state=split_seed
        )
        for fold_index, (train_indices, test_indices) in enumerate(
            splitter.split(np.zeros(len(labels)), labels)
        ):
            outer_seed = split_seed * 100 + fold_index
            oof = _cross_fitted_predictions(
                prepared,
                labels,
                train_indices,
                base_feature=base_feature,
                n_atoms=args.n_atoms,
                sparsity=args.sparsity,
                iterations=args.iterations,
                max_train_patches=args.max_train_patches,
                inner_splits=args.inner_splits,
                seed=outer_seed,
            )
            dictionary = _fit_dictionary(
                prepared,
                train_indices,
                branch="fair",
                n_atoms=args.n_atoms,
                sparsity=args.sparsity,
                iterations=args.iterations,
                max_train_patches=args.max_train_patches,
                seed=outer_seed,
            )
            train_features = _feature_matrix(
                prepared, train_indices, dictionary, sparsity=args.sparsity
            )
            test_features = _feature_matrix(
                prepared, test_indices, dictionary, sparsity=args.sparsity
            )
            test_predictions = {
                "BASE": _fit_expert(
                    train_features[base_feature],
                    labels[train_indices],
                    test_features[base_feature],
                    seed=outer_seed,
                    n_classes=n_classes,
                ),
                "INIT": _fit_expert(
                    train_features["INIT_RICH"],
                    labels[train_indices],
                    test_features["INIT_RICH"],
                    seed=outer_seed,
                    n_classes=n_classes,
                ),
                "FINAL": _fit_expert(
                    train_features["FINAL_RICH"],
                    labels[train_indices],
                    test_features["FINAL_RICH"],
                    seed=outer_seed,
                    n_classes=n_classes,
                ),
            }
            scores = {
                name_: _score(probabilities, labels[test_indices])
                for name_, probabilities in test_predictions.items()
            }
            fusion_metadata = {}
            for family in ("INIT", "FINAL"):
                outputs, metadata_ = _fusion_outputs(
                    oof["BASE"],
                    oof[family],
                    labels[train_indices],
                    test_predictions["BASE"],
                    test_predictions[family],
                    seed=outer_seed,
                )
                for method, probabilities in outputs.items():
                    scores[f"{family}_{method}"] = _score(
                        probabilities, labels[test_indices]
                    )
                fusion_metadata[family] = metadata_
            folds.append(
                {
                    "split_seed": split_seed,
                    "fold_index": fold_index,
                    "base_feature": base_feature,
                    "scores": scores,
                    "fusion": fusion_metadata,
                    "oof_disagreement": {
                        family: _disagreement_summary(
                            oof["BASE"], oof[family], labels[train_indices]
                        )
                        for family in ("INIT", "FINAL")
                    },
                }
            )
    methods = ("FIXED_AVG", "LOGIT_ADD", "OOF_SCALAR", "OOF_GATE", "OOF_STACK")
    paired = {}
    for method in methods:
        paired[f"final_{method.lower()}_minus_base"] = _paired(
            folds, f"FINAL_{method}", "BASE"
        )
        paired[f"final_minus_init_{method.lower()}"] = _paired(
            folds, f"FINAL_{method}", f"INIT_{method}"
        )
    paired["final_expert_minus_base"] = _paired(folds, "FINAL", "BASE")
    paired["final_expert_minus_init_expert"] = _paired(folds, "FINAL", "INIT")
    return {
        "dataset": name,
        "metadata": metadata,
        "base_feature": base_feature,
        "sampling": _sampling_summary(prepared),
        "summary": _aggregate(folds),
        "paired": paired,
        "folds": folds,
        "seconds": time.time() - started,
    }


def _pass(pair: dict[str, Any]) -> bool:
    return pair["mean"] >= 0.01 and pair["wins"] >= 6


def classify(results: Sequence[dict[str, Any]]) -> dict[str, Any]:
    methods = ("oof_scalar", "oof_gate", "oof_stack")
    method_rows = {}
    for method in methods:
        gains = {
            result["dataset"]: {
                **result["paired"][f"final_{method}_minus_base"],
                "pass": _pass(result["paired"][f"final_{method}_minus_base"]),
            }
            for result in results
        }
        feature_deltas = {
            name: gains[name]["mean"]
            for name in FEATURE_DATASETS
            if name in gains
        }
        fusion_pass = (
            sum(row["pass"] for row in gains.values()) >= 2
            and bool(feature_deltas)
            and max(feature_deltas.values()) >= 0.01
            and min(feature_deltas.values()) >= -0.01
        )
        attribution = {
            result["dataset"]: result["paired"][f"final_minus_init_{method}"]
            for result in results
        }
        attribution_values = [row["mean"] for row in attribution.values()]
        attribution_pass = (
            sum(value > 0.0 for value in attribution_values) >= 2
            and float(np.mean(attribution_values)) > 0.0
        )
        method_rows[method] = {
            "fusion_pass": fusion_pass,
            "attribution_pass": attribution_pass,
            "gains": gains,
            "final_minus_init": attribution,
        }
    passed = [
        method
        for method, row in method_rows.items()
        if row["fusion_pass"] and row["attribution_pass"]
    ]
    if passed:
        label = "OOF_MULTIVIEW_FUSION_ADVANCES"
    elif any(row["fusion_pass"] for row in method_rows.values()):
        label = "MULTIVIEW_ENSEMBLE_WITHOUT_KSVD_ATTRIBUTION"
    else:
        label = "CURRENT_KSVD_EXPERT_LACKS_STABLE_COMPLEMENTARITY"
    return {"classification": label, "passed_methods": passed, "methods": method_rows}


def render_report(payload: dict[str, Any]) -> str:
    lines = [
        "# luyin14 OOF 多视图融合：节点/统计专家 + KSVD 专家",
        "",
        "> 日期：2026-08-12  ",
        f"> 协议：`{payload['protocol']}`  ",
        f"> 判定：`{payload['decision']['classification']}`",
        "",
        "## 1. 专家与融合主表（balanced accuracy）",
        "",
        "| dataset | base view | BASE | INIT | FINAL | FINAL avg | FINAL scalar | FINAL gate | FINAL stack | best OOF Δ |",
        "|---|---|---:|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for result in payload["datasets"]:
        summary = result["summary"]
        get = lambda name: summary[name]["balanced_accuracy_mean"]
        oof_names = ["FINAL_OOF_SCALAR", "FINAL_OOF_GATE", "FINAL_OOF_STACK"]
        best = max(get(name) for name in oof_names) - get("BASE")
        lines.append(
            f"| {result['dataset']} | {result['base_feature']} | {get('BASE'):.3f} | "
            f"{get('INIT'):.3f} | {get('FINAL'):.3f} | {get('FINAL_FIXED_AVG'):.3f} | "
            f"{get('FINAL_OOF_SCALAR'):.3f} | {get('FINAL_OOF_GATE'):.3f} | "
            f"{get('FINAL_OOF_STACK'):.3f} | {best:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## 2. OOF 融合相对 BASE",
            "",
            "| dataset | scalar Δ/W | gate Δ/W | stack Δ/W |",
            "|---|---:|---:|---:|",
        ]
    )
    for result in payload["datasets"]:
        cells = []
        for method in ("oof_scalar", "oof_gate", "oof_stack"):
            pair = result["paired"][f"final_{method}_minus_base"]
            cells.append(
                f"{pair['mean']:+.3f} ({pair['wins']}/{pair['ties']}/{pair['losses']})"
            )
        lines.append(f"| {result['dataset']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## 3. K-SVD update 归因：FINAL fusion − INIT fusion",
            "",
            "| dataset | scalar | gate | stack |",
            "|---|---:|---:|---:|",
        ]
    )
    for result in payload["datasets"]:
        cells = []
        for method in ("oof_scalar", "oof_gate", "oof_stack"):
            pair = result["paired"][f"final_minus_init_{method}"]
            cells.append(
                f"{pair['mean']:+.3f} ({pair['wins']}/{pair['ties']}/{pair['losses']})"
            )
        lines.append(f"| {result['dataset']} | " + " | ".join(cells) + " |")
    lines.extend(
        [
            "",
            "## 4. 融合器行为",
            "",
            "| dataset | FINAL scalar alpha | FINAL gate weight | OOF disagreement | struct-only correct | base-only correct |",
            "|---|---:|---:|---:|---:|---:|",
        ]
    )
    for result in payload["datasets"]:
        alphas = [fold["fusion"]["FINAL"]["scalar_alpha"] for fold in result["folds"]]
        weights = [
            fold["fusion"]["FINAL"]["test_gate_weight_mean"]
            for fold in result["folds"]
        ]
        disagreement = [
            fold["oof_disagreement"]["FINAL"]["prediction_disagreement_fraction"]
            for fold in result["folds"]
        ]
        structure_only = [
            fold["oof_disagreement"]["FINAL"]["structure_only_correct_fraction"]
            for fold in result["folds"]
        ]
        base_only = [
            fold["oof_disagreement"]["FINAL"]["base_only_correct_fraction"]
            for fold in result["folds"]
        ]
        lines.append(
            f"| {result['dataset']} | {np.mean(alphas):+.3f} | {np.mean(weights):.3f} | "
            f"{np.mean(disagreement):.3f} | {np.mean(structure_only):.3f} | "
            f"{np.mean(base_only):.3f} |"
        )
    lines.extend(["", "## 5. 结论", ""])
    label = payload["decision"]["classification"]
    if label == "OOF_MULTIVIEW_FUSION_ADVANCES":
        methods = ", ".join(payload["decision"]["passed_methods"])
        lines.append(
            f"OOF prediction-level fusion 与 K-SVD update 归因同时通过（{methods}）。下一步可以只沿通过机制研究 token-level interaction/cross-attention，并保留 OOF late fusion 作为强低容量基线。"
        )
    elif label == "MULTIVIEW_ENSEMBLE_WITHOUT_KSVD_ATTRIBUTION":
        lines.append(
            "低维多视图融合有用，但 FINAL 没有稳定优于 INIT；收益属于普通多视图集成/校准，不能归因为 K-SVD updates。"
        )
    else:
        lines.append(
            "严格 OOF 融合也未建立稳定互补性。现有 concat 的失败不只是融合器过于简单，而是当前 KSVD expert 在真实数据上不能稳定修正 base expert 的错误；不应直接增加 cross-attention 容量。"
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
    parser.add_argument("--inner-splits", type=int, default=3)
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


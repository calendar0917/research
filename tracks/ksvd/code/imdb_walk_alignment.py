"""Objective-alignment diagnostics for frozen IMDB-BINARY R0-D dictionaries."""
from __future__ import annotations

from typing import Any, Sequence

import numpy as np

from .imdb_walk_dictionary import FoldSplit, encode_with_minimum_sparsity, stack_patch_graphs
from .imdb_walk_downstream import variable_graph_code_readout
from .imdb_walk_substrate import IMDBPatchGraph


EPS = 1e-12


def _fit_standardization(values: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    values = np.asarray(values, dtype=np.float64)
    mean = np.mean(values, axis=0)
    scale = np.std(values, axis=0, ddof=0)
    active = scale > EPS
    safe_scale = scale.copy()
    safe_scale[~active] = 1.0
    return mean, safe_scale, active


def _standardize(
    values: np.ndarray, mean: np.ndarray, scale: np.ndarray, active: np.ndarray
) -> np.ndarray:
    return (np.asarray(values, dtype=np.float64)[:, active] - mean[active]) / scale[active]


def fit_linear_residuals(
    train_x: np.ndarray,
    test_x: np.ndarray,
    train_y: np.ndarray,
    test_y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit train-only least squares and return residuals plus held-out R2.

    Inputs and outputs are standardized with train statistics.  Standardizing
    every active output dimension prevents coefficient-scale differences from
    making activation frequency dominate or disappear in the aggregate R2.
    """
    x_mean, x_scale, x_active = _fit_standardization(train_x)
    y_mean, y_scale, y_active = _fit_standardization(train_y)
    if not np.any(x_active) or not np.any(y_active):
        raise ValueError("linear residualization requires active inputs and outputs")
    x_train = _standardize(train_x, x_mean, x_scale, x_active)
    x_test = _standardize(test_x, x_mean, x_scale, x_active)
    y_train = _standardize(train_y, y_mean, y_scale, y_active)
    y_test = _standardize(test_y, y_mean, y_scale, y_active)
    design_train = np.column_stack([x_train, np.ones(x_train.shape[0])])
    design_test = np.column_stack([x_test, np.ones(x_test.shape[0])])
    coefficients = np.linalg.lstsq(design_train, y_train, rcond=None)[0]
    train_prediction = design_train @ coefficients
    test_prediction = design_test @ coefficients
    train_residual = y_train - train_prediction
    test_residual = y_test - test_prediction
    train_denominator = float(np.sum(y_train**2))
    test_denominator = float(np.sum(y_test**2))
    return train_residual, test_residual, {
        "input_active_dimension": int(np.count_nonzero(x_active)),
        "output_active_dimension": int(np.count_nonzero(y_active)),
        "train_explained_fraction": float(
            1.0 - np.sum(train_residual**2) / max(train_denominator, EPS)
        ),
        "test_explained_fraction": float(
            1.0 - np.sum(test_residual**2) / max(test_denominator, EPS)
        ),
    }


def label_effect_alignment(
    train_values: np.ndarray,
    test_values: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
    *,
    already_standardized: bool = False,
) -> dict[str, Any]:
    """Compare train and held-out class-effect vectors without a classifier."""
    train_values = np.asarray(train_values, dtype=np.float64)
    test_values = np.asarray(test_values, dtype=np.float64)
    train_labels = np.asarray(train_labels, dtype=np.int64)
    test_labels = np.asarray(test_labels, dtype=np.int64)
    if already_standardized:
        active = np.std(train_values, axis=0, ddof=0) > EPS
        train = train_values[:, active]
        test = test_values[:, active]
    else:
        mean, scale, active = _fit_standardization(train_values)
        train = _standardize(train_values, mean, scale, active)
        test = _standardize(test_values, mean, scale, active)
    if train.shape[1] == 0:
        raise ValueError("label effect requires at least one active dimension")
    train_effect = np.mean(train[train_labels == 1], axis=0) - np.mean(
        train[train_labels == 0], axis=0
    )
    test_effect = np.mean(test[test_labels == 1], axis=0) - np.mean(
        test[test_labels == 0], axis=0
    )
    train_norm = float(np.linalg.norm(train_effect))
    test_norm = float(np.linalg.norm(test_effect))
    denominator = max(train_norm * test_norm, EPS)
    projection = float(np.dot(test_effect, train_effect) / max(train_norm, EPS))
    nonzero = (np.abs(train_effect) > EPS) | (np.abs(test_effect) > EPS)
    same_sign = float(np.mean(train_effect[nonzero] * test_effect[nonzero] >= 0.0))
    return {
        "active_dimension": int(train.shape[1]),
        "train_effect_norm": train_norm,
        "test_effect_norm": test_norm,
        "train_test_effect_cosine": float(np.dot(train_effect, test_effect) / denominator),
        "test_effect_projection_on_train_direction": projection,
        "same_sign_dimension_fraction": same_sign,
    }


def scalar_label_effect(
    train_values: np.ndarray,
    test_values: np.ndarray,
    train_labels: np.ndarray,
    test_labels: np.ndarray,
) -> dict[str, Any]:
    """Report class means and pooled Cohen d for a scalar graph quantity."""
    result: dict[str, Any] = {}
    for split, values, labels in (
        ("train", train_values, train_labels),
        ("test", test_values, test_labels),
    ):
        values = np.asarray(values, dtype=np.float64)
        labels = np.asarray(labels, dtype=np.int64)
        class0 = values[labels == 0]
        class1 = values[labels == 1]
        pooled_variance = (
            (class0.size - 1) * np.var(class0, ddof=1)
            + (class1.size - 1) * np.var(class1, ddof=1)
        ) / max(class0.size + class1.size - 2, 1)
        difference = float(np.mean(class1) - np.mean(class0))
        result[split] = {
            "class_0_mean": float(np.mean(class0)),
            "class_1_mean": float(np.mean(class1)),
            "mean_difference_class1_minus_class0": difference,
            "cohen_d": float(difference / max(np.sqrt(max(pooled_variance, 0.0)), EPS)),
        }
    result["effect_sign_consistent"] = bool(
        result["train"]["mean_difference_class1_minus_class0"]
        * result["test"]["mean_difference_class1_minus_class0"]
        >= 0.0
    )
    return result


def _graph_reconstruction_errors(
    values: np.ndarray,
    reconstruction: np.ndarray,
    graph_slices: Sequence[tuple[int, int, int, int]],
) -> np.ndarray:
    errors = []
    for _graph_index, _label, start, stop in graph_slices:
        signal = values[:, start:stop]
        residual = signal - reconstruction[:, start:stop]
        errors.append(
            float(np.linalg.norm(residual, "fro") / max(np.linalg.norm(signal, "fro"), EPS))
        )
    return np.asarray(errors, dtype=np.float64)


def patch_frequency_diagnosis(
    edge_counts: np.ndarray,
    init_squared_errors: np.ndarray,
    final_squared_errors: np.ndarray,
) -> dict[str, Any]:
    """Describe which edge-count bins receive the reconstruction improvement."""
    edge_counts = np.asarray(edge_counts, dtype=np.int64)
    init_squared_errors = np.asarray(init_squared_errors, dtype=np.float64)
    final_squared_errors = np.asarray(final_squared_errors, dtype=np.float64)
    if not (
        edge_counts.shape == init_squared_errors.shape == final_squared_errors.shape
        and edge_counts.ndim == 1
    ):
        raise ValueError("patch frequency inputs must be equal-length vectors")
    total_count = int(edge_counts.size)
    rows = []
    for edge_count in sorted(set(edge_counts.tolist())):
        mask = edge_counts == edge_count
        reduction = init_squared_errors[mask] - final_squared_errors[mask]
        rows.append(
            {
                "edge_count": int(edge_count),
                "patch_count": int(np.count_nonzero(mask)),
                "patch_mass": float(np.mean(mask)),
                "init_mean_squared_error": float(np.mean(init_squared_errors[mask])),
                "final_mean_squared_error": float(np.mean(final_squared_errors[mask])),
                "mean_squared_error_reduction_per_patch": float(np.mean(reduction)),
                "total_squared_error_reduction": float(np.sum(reduction)),
                "positive_total_squared_error_reduction": float(max(np.sum(reduction), 0.0)),
            }
        )
    positive_total = sum(row["positive_total_squared_error_reduction"] for row in rows)
    for row in rows:
        row["positive_gain_contribution"] = float(
            row["positive_total_squared_error_reduction"] / max(positive_total, EPS)
        )
    top = sorted(rows, key=lambda row: (-row["patch_count"], row["edge_count"]))[:3]
    masses = np.asarray([row["patch_mass"] for row in rows], dtype=np.float64)
    gains = np.asarray(
        [row["positive_total_squared_error_reduction"] for row in rows], dtype=np.float64
    )
    correlation = 0.0
    if np.std(masses) > EPS and np.std(gains) > EPS:
        correlation = float(np.corrcoef(masses, gains)[0, 1])
    return {
        "patch_count": total_count,
        "edge_count_bin_count": int(len(rows)),
        "bins": rows,
        "top3_frequency_edge_counts": [row["edge_count"] for row in top],
        "top3_frequency_patch_mass": float(sum(row["patch_mass"] for row in top)),
        "top3_frequency_positive_gain_contribution": float(
            sum(row["positive_gain_contribution"] for row in top)
        ),
        "bin_mass_positive_total_gain_correlation": correlation,
    }


def run_alignment_fold(
    examples: Sequence[IMDBPatchGraph],
    split: FoldSplit,
    *,
    train_mean: np.ndarray,
    initial_dictionary: np.ndarray,
    final_dictionary: np.ndarray,
    sparsity: int = 2,
    minimum_sparsity: int = 1,
) -> dict[str, Any]:
    """Diagnose one frozen R0-D fold without fitting a new dictionary."""
    example_by_index = {int(example.graph_index): example for example in examples}
    split_examples = {
        "train": tuple(example_by_index[index] for index in split.train_indices),
        "test": tuple(example_by_index[index] for index in split.test_indices),
    }
    train_mean = np.asarray(train_mean, dtype=np.float64).reshape(-1, 1)
    dictionaries = {
        "init": np.asarray(initial_dictionary, dtype=np.float64),
        "final": np.asarray(final_dictionary, dtype=np.float64),
    }
    raw: dict[str, np.ndarray] = {}
    centered: dict[str, np.ndarray] = {}
    slices: dict[str, tuple[tuple[int, int, int, int], ...]] = {}
    labels: dict[str, np.ndarray] = {}
    stats: dict[str, np.ndarray] = {}
    raw_walk: dict[str, np.ndarray] = {}
    edge_counts: dict[str, np.ndarray] = {}
    for split_name, items in split_examples.items():
        raw[split_name], slices[split_name] = stack_patch_graphs(items)
        centered[split_name] = raw[split_name] - train_mean
        labels[split_name] = np.asarray([item.label for item in items], dtype=np.int64)
        stats[split_name] = np.stack([item.graph_statistics for item in items], axis=0)
        raw_walk[split_name] = np.stack(
            [item.features["walk_mean_std"] for item in items], axis=0
        )
        edge_counts[split_name] = np.concatenate([item.edge_counts for item in items])

    graph_codes: dict[str, dict[str, np.ndarray]] = {stage: {} for stage in dictionaries}
    graph_errors: dict[str, dict[str, np.ndarray]] = {stage: {} for stage in dictionaries}
    patch_squared_errors: dict[str, dict[str, np.ndarray]] = {stage: {} for stage in dictionaries}
    for stage, dictionary in dictionaries.items():
        for split_name in ("train", "test"):
            codes = encode_with_minimum_sparsity(
                centered[split_name],
                dictionary,
                sparsity=sparsity,
                minimum_sparsity=minimum_sparsity,
            )
            reconstruction = dictionary @ codes
            graph_codes[stage][split_name] = variable_graph_code_readout(
                codes, slices[split_name]
            )
            graph_errors[stage][split_name] = _graph_reconstruction_errors(
                centered[split_name], reconstruction, slices[split_name]
            )
            patch_squared_errors[stage][split_name] = np.sum(
                (centered[split_name] - reconstruction) ** 2, axis=0
            )

    gains = {
        split_name: graph_errors["init"][split_name] - graph_errors["final"][split_name]
        for split_name in ("train", "test")
    }
    redundancy: dict[str, Any] = {}
    residual_codes: dict[str, dict[str, np.ndarray]] = {}
    for key, values in (
        ("init_code", graph_codes["init"]),
        ("final_code", graph_codes["final"]),
        ("raw_walk", raw_walk),
        (
            "reconstruction_gain",
            {split_name: gains[split_name][:, None] for split_name in ("train", "test")},
        ),
    ):
        train_residual, test_residual, metrics = fit_linear_residuals(
            stats["train"], stats["test"], values["train"], values["test"]
        )
        redundancy[key] = metrics
        residual_codes[key] = {"train": train_residual, "test": test_residual}

    label_alignment: dict[str, Any] = {}
    for key, values in (
        ("init_code", graph_codes["init"]),
        ("final_code", graph_codes["final"]),
        ("raw_walk", raw_walk),
    ):
        label_alignment[key] = {
            "raw": label_effect_alignment(
                values["train"], values["test"], labels["train"], labels["test"]
            ),
            "after_stats_residualization": label_effect_alignment(
                residual_codes[key]["train"],
                residual_codes[key]["test"],
                labels["train"],
                labels["test"],
                already_standardized=True,
            ),
        }

    patch_frequency = {
        split_name: patch_frequency_diagnosis(
            edge_counts[split_name],
            patch_squared_errors["init"][split_name],
            patch_squared_errors["final"][split_name],
        )
        for split_name in ("train", "test")
    }
    return {
        "fold_index": int(split.fold_index),
        "train_graph_count": int(len(split.train_indices)),
        "test_graph_count": int(len(split.test_indices)),
        "train_patch_count": int(raw["train"].shape[1]),
        "test_patch_count": int(raw["test"].shape[1]),
        "redundancy": redundancy,
        "label_alignment": label_alignment,
        "reconstruction_gain_label_effect": scalar_label_effect(
            gains["train"], gains["test"], labels["train"], labels["test"]
        ),
        "mean_graph_reconstruction_gain": {
            split_name: float(np.mean(gains[split_name])) for split_name in ("train", "test")
        },
        "patch_frequency": patch_frequency,
    }

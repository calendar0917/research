"""Train-only audit of tuning and task-aligned structure--attribute interactions.

The clean structural-role experiment established that topology-only rooted-WL
roles expose genuine role--attribute dependence, but a dense joint block did
not improve the frozen official validation result over the T+A marginals.
This protocol asks two narrower questions without touching official validation
or test:

1. Does an equal-budget, nested scaffold Optuna search reverse the comparison
   between T+A and the dense raw/centered interaction views?
2. Can a small interaction correction selected against cross-fitted T+A
   residuals transfer to a held-out scaffold group?

The outer folds are the three official-train-only scaffold folds.  Within an
outer training fold, the remaining two scaffold groups form two directional
inner folds.  The residual stack uses only out-of-fold baseline predictions.
Matched within-patch shuffles receive the same sparse-stack search budget.
"""

from __future__ import annotations

import argparse
import json
import platform
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler
from xgboost import XGBClassifier
import yaml

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.role_attribute_binding_screen import (
    REPO_ROOT,
    _fold_indices,
    _resolve,
    _sha256,
    _summary,
    _write_json,
)
from tracks.ksvd.experiments.luyin16.structural_role_fusion_screen import (
    _build_dataset_blocks,
)


DEFAULT_CONFIG = (
    REPO_ROOT
    / "tracks/ksvd/configs/luyin16/task_aligned_interaction_screen.yaml"
)
XGB_VIEWS = ("t_a", "f_raw_factorized", "f_centered")
INTERACTION_SOURCES = (
    "raw",
    "raw_shuffled",
    "centered",
    "centered_shuffled",
)
XGB_SEARCH_KEYS = (
    "n_estimators",
    "max_depth",
    "learning_rate",
    "min_child_weight",
    "subsample",
    "colsample_bytree",
    "reg_lambda",
    "reg_alpha",
    "gamma",
)


def _join(*parts: np.ndarray) -> np.ndarray:
    if not parts:
        raise ValueError("at least one feature block is required")
    rows = {int(np.asarray(part).shape[0]) for part in parts}
    if len(rows) != 1 or any(np.asarray(part).ndim != 2 for part in parts):
        raise ValueError("feature blocks must be aligned two-dimensional arrays")
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def assemble_task_views(
    blocks: Mapping[str, np.ndarray],
    *,
    shuffle_repeat: int = 0,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    """Build dense XGBoost views and interaction-only blocks.

    Marginals stay outside the interaction blocks so the sparse residual stage
    can test incremental information over an explicit T+A baseline.
    """
    marginals = _join(
        blocks["node_role"],
        blocks["edge_role"],
        blocks["node_attribute"],
        blocks["edge_attribute"],
        blocks["context"],
    )
    raw = _join(blocks["node_raw"], blocks["edge_raw"])
    centered = _join(blocks["node_binding"], blocks["edge_binding"])
    suffix = f"_shuffled_{int(shuffle_repeat)}"
    raw_shuffled = _join(
        blocks[f"node_raw{suffix}"], blocks[f"edge_raw{suffix}"]
    )
    centered_shuffled = _join(
        blocks[f"node_binding{suffix}"], blocks[f"edge_binding{suffix}"]
    )
    xgb_views = {
        "t_a": marginals,
        "f_raw_factorized": _join(marginals, raw),
        "f_centered": _join(marginals, centered),
    }
    interactions = {
        "raw": raw,
        "raw_shuffled": raw_shuffled,
        "centered": centered,
        "centered_shuffled": centered_shuffled,
    }
    return xgb_views, interactions


def scaffold_group_mapping(
    archive: Mapping[str, np.ndarray],
) -> tuple[dict[int, int], list[int]]:
    """Map every official-train graph to its unique scaffold validation group."""
    original = np.asarray(archive["original_indices"], dtype=np.int64)
    fold_ids = sorted(
        int(name.removeprefix("fold_").removesuffix("_valid_indices"))
        for name in archive
        if name.startswith("fold_") and name.endswith("_valid_indices")
    )
    mapping: dict[int, int] = {}
    for fold in fold_ids:
        members = original[
            np.asarray(archive[f"fold_{fold}_valid_indices"], dtype=np.int64)
        ]
        for raw_index in members:
            index = int(raw_index)
            if index in mapping:
                raise ValueError(f"dataset index {index} belongs to multiple scaffold groups")
            mapping[index] = int(fold)
    official_train = original[
        np.asarray(archive["official_train_indices"], dtype=np.int64)
    ]
    missing = [int(index) for index in official_train if int(index) not in mapping]
    if missing:
        raise ValueError(f"{len(missing)} official-train graphs lack scaffold groups")
    return mapping, fold_ids


def nested_scaffold_splits(
    dataset_indices: np.ndarray,
    group_by_index: Mapping[int, int],
    *,
    held_out_group: int,
    labels: np.ndarray | None = None,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], np.ndarray]:
    """Return leave-one-group-out splits inside an outer training fold."""
    indices = np.asarray(dataset_indices, dtype=np.int64)
    groups = np.asarray([group_by_index[int(index)] for index in indices], dtype=np.int64)
    present = sorted(int(value) for value in np.unique(groups))
    if int(held_out_group) in present:
        raise ValueError("outer held-out scaffold group leaked into its training rows")
    if len(present) < 2:
        raise ValueError("nested scaffold tuning requires at least two training groups")
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    for group in present:
        valid = np.flatnonzero(groups == group).astype(np.int64)
        train = np.flatnonzero(groups != group).astype(np.int64)
        if np.intersect1d(train, valid).size:
            raise RuntimeError("nested train and validation rows overlap")
        if labels is not None:
            y = np.asarray(labels, dtype=np.int64)
            if np.unique(y[train]).size != 2 or np.unique(y[valid]).size != 2:
                raise ValueError(f"nested scaffold group {group} lacks both classes")
        splits.append((train, valid))
    coverage = np.zeros(indices.size, dtype=np.int64)
    for _, valid in splits:
        coverage[valid] += 1
    if not np.all(coverage == 1):
        raise RuntimeError("nested validation folds must cover every training row once")
    return splits, groups


def _base_xgb_params(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    params = {
        key: config[key]
        for key in XGB_SEARCH_KEYS
        if key in config
    }
    params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": int(seed),
            "tree_method": "hist",
            "n_jobs": int(config.get("n_jobs", -1)),
        }
    )
    return params


def _positive_weight(labels: np.ndarray) -> float:
    y = np.asarray(labels, dtype=np.int64)
    return float(np.sum(y == 0) / max(1, np.sum(y == 1)))


def _fit_xgb_predict(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    params: Mapping[str, Any],
) -> np.ndarray:
    model_params = dict(params)
    model_params["scale_pos_weight"] = _positive_weight(y_train)
    model = XGBClassifier(**model_params)
    model.fit(x_train, y_train)
    return np.asarray(model.predict_proba(x_valid)[:, 1], dtype=np.float64)


def _score_xgb_params(
    matrix: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    params: Mapping[str, Any],
) -> list[float]:
    y = np.asarray(labels, dtype=np.int64)
    scores = []
    for train, valid in splits:
        prediction = _fit_xgb_predict(
            matrix[train], y[train], matrix[valid], params
        )
        scores.append(float(roc_auc_score(y[valid], prediction)))
    return scores


def _suggest_xgb_params(
    trial: optuna.Trial,
    tuning: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    ranges = tuning["ranges"]
    params = {
        "n_estimators": trial.suggest_int(
            "n_estimators", *[int(value) for value in ranges["n_estimators"]]
        ),
        "max_depth": trial.suggest_int(
            "max_depth", *[int(value) for value in ranges["max_depth"]]
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            *[float(value) for value in ranges["learning_rate"]],
            log=True,
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            *[float(value) for value in ranges["min_child_weight"]],
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample", *[float(value) for value in ranges["subsample"]]
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            *[float(value) for value in ranges["colsample_bytree"]],
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda", *[float(value) for value in ranges["reg_lambda"]], log=True
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha", *[float(value) for value in ranges["reg_alpha"]], log=True
        ),
        "gamma": trial.suggest_float(
            "gamma", *[float(value) for value in ranges["gamma"]]
        ),
        "objective": "binary:logistic",
        "eval_metric": "auc",
        "random_state": int(seed),
        "tree_method": "hist",
        "n_jobs": int(tuning.get("n_jobs", -1)),
    }
    return params


def tune_xgb_view(
    matrix: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    tuning: Mapping[str, Any],
    fixed_classifier: Mapping[str, Any],
    *,
    seed: int,
) -> dict[str, Any]:
    """Run equal-budget TPE search, always including the old fixed probe."""
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    fixed_trial = {
        key: fixed_classifier[key]
        for key in XGB_SEARCH_KEYS
        if key in fixed_classifier
    }
    ranges = tuning["ranges"]
    in_range = all(
        key in fixed_trial
        and float(ranges[key][0]) <= float(fixed_trial[key]) <= float(ranges[key][1])
        for key in XGB_SEARCH_KEYS
    )
    if in_range:
        study.enqueue_trial(fixed_trial)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgb_params(trial, tuning, seed=seed)
        return float(np.mean(_score_xgb_params(matrix, labels, splits, params)))

    study.optimize(
        objective,
        n_trials=int(tuning["n_trials"]),
        show_progress_bar=False,
    )
    best_params = dict(study.best_trial.params)
    best_params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "random_state": int(seed),
            "tree_method": "hist",
            "n_jobs": int(tuning.get("n_jobs", -1)),
        }
    )
    best_folds = _score_xgb_params(matrix, labels, splits, best_params)
    return {
        "backend": "optuna_tpe",
        "n_trials": int(tuning["n_trials"]),
        "best_trial": int(study.best_trial.number),
        "best_inner_auc": float(np.mean(best_folds)),
        "best_inner_fold_auc": best_folds,
        "best_params": best_params,
        "trials": [
            {
                "trial": int(trial.number),
                "value": None if trial.value is None else float(trial.value),
                "params": dict(trial.params),
                "state": str(trial.state),
            }
            for trial in study.trials
        ],
    }


def evaluate_xgb_outer(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    params: Mapping[str, Any],
    seeds: Sequence[int],
) -> dict[str, Any]:
    rows = []
    for seed in seeds:
        current = dict(params)
        current["random_state"] = int(seed)
        prediction = _fit_xgb_predict(x_train, y_train, x_valid, current)
        rows.append(
            {
                "seed": int(seed),
                "valid_auc": float(roc_auc_score(y_valid, prediction)),
            }
        )
    values = [row["valid_auc"] for row in rows]
    return {"rows": rows, "summary": _summary(values)}


def cross_fitted_xgb_predictions(
    matrix: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    params: Mapping[str, Any],
) -> np.ndarray:
    prediction = np.full(labels.shape[0], np.nan, dtype=np.float64)
    coverage = np.zeros(labels.shape[0], dtype=np.int64)
    for train, valid in splits:
        prediction[valid] = _fit_xgb_predict(
            matrix[train], labels[train], matrix[valid], params
        )
        coverage[valid] += 1
    if not np.all(coverage == 1) or not np.all(np.isfinite(prediction)):
        raise RuntimeError("cross-fitted predictions do not cover each row exactly once")
    return prediction


def stratified_oof_xgb_predictions(
    matrix: np.ndarray,
    labels: np.ndarray,
    params: Mapping[str, Any],
    *,
    n_splits: int,
    seed: int,
) -> np.ndarray:
    y = np.asarray(labels, dtype=np.int64)
    splitter = StratifiedKFold(
        n_splits=int(n_splits), shuffle=True, random_state=int(seed)
    )
    splits = [
        (np.asarray(train, dtype=np.int64), np.asarray(valid, dtype=np.int64))
        for train, valid in splitter.split(np.zeros(y.shape[0]), y)
    ]
    return cross_fitted_xgb_predictions(matrix, y, splits, params)


def _clip_logit(probability: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(probability, dtype=np.float64), 1.0e-6, 1.0 - 1.0e-6)
    return np.log(values / (1.0 - values))


def interaction_correlations(
    matrix: np.ndarray,
    residual: np.ndarray,
    *,
    support_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return column-wise Pearson correlations and nonzero support counts."""
    x = np.asarray(matrix, dtype=np.float64)
    r = np.asarray(residual, dtype=np.float64).reshape(-1)
    if x.ndim != 2 or x.shape[0] != r.shape[0]:
        raise ValueError("interaction matrix and residuals are not aligned")
    centered_r = r - float(np.mean(r))
    residual_ss = float(np.dot(centered_r, centered_r))
    sums = np.sum(x, axis=0, dtype=np.float64)
    sum_squares = np.sum(np.square(x), axis=0, dtype=np.float64)
    feature_ss = np.maximum(sum_squares - np.square(sums) / max(1, x.shape[0]), 0.0)
    numerator = np.asarray(x.T @ centered_r, dtype=np.float64)
    denominator = np.sqrt(feature_ss * residual_ss)
    correlations = np.divide(
        numerator,
        denominator,
        out=np.zeros_like(numerator),
        where=denominator > 0.0,
    )
    support = np.sum(np.abs(x) > float(support_tolerance), axis=0, dtype=np.int64)
    return correlations, support


def rank_interactions(
    matrix: np.ndarray,
    residual: np.ndarray,
    *,
    minimum_support: int,
    support_tolerance: float,
) -> tuple[np.ndarray, np.ndarray]:
    correlations, support = interaction_correlations(
        matrix, residual, support_tolerance=support_tolerance
    )
    eligible = np.flatnonzero(support >= int(minimum_support)).astype(np.int64)
    if eligible.size == 0:
        return eligible, np.zeros(0, dtype=np.float64)
    order = np.argsort(-np.abs(correlations[eligible]), kind="stable")
    ranked = eligible[order]
    return ranked, np.abs(correlations[ranked])


def rank_stable_interactions(
    matrix: np.ndarray,
    residual: np.ndarray,
    groups: np.ndarray,
    *,
    minimum_support: int,
    support_tolerance: float,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Rank interactions whose residual association has one sign in all groups."""
    x = np.asarray(matrix)
    y_residual = np.asarray(residual)
    group_values = sorted(int(value) for value in np.unique(groups))
    if len(group_values) < 2:
        raise ValueError("stable interaction selection needs at least two scaffold groups")
    correlations = []
    supports = []
    for group in group_values:
        rows = np.flatnonzero(np.asarray(groups) == group)
        corr, support = interaction_correlations(
            x[rows], y_residual[rows], support_tolerance=support_tolerance
        )
        correlations.append(corr)
        supports.append(support)
    corr_matrix = np.stack(correlations, axis=0)
    support_matrix = np.stack(supports, axis=0)
    same_positive = np.all(corr_matrix > 0.0, axis=0)
    same_negative = np.all(corr_matrix < 0.0, axis=0)
    enough_support = np.all(support_matrix >= int(minimum_support), axis=0)
    eligible_mask = (same_positive | same_negative) & enough_support
    eligible = np.flatnonzero(eligible_mask).astype(np.int64)
    if eligible.size:
        scores = np.min(np.abs(corr_matrix[:, eligible]), axis=0)
        order = np.argsort(-scores, kind="stable")
        ranked = eligible[order]
        ranked_scores = scores[order]
    else:
        ranked = eligible
        ranked_scores = np.zeros(0, dtype=np.float64)
    metadata = {
        "groups": group_values,
        "n_features": int(x.shape[1]),
        "n_supported_all_groups": int(np.sum(enough_support)),
        "n_sign_stable": int(np.sum(eligible_mask)),
    }
    return ranked, ranked_scores, metadata


def _fit_sparse_stack(
    train_probability: np.ndarray,
    train_interactions: np.ndarray,
    train_labels: np.ndarray,
    valid_probability: np.ndarray,
    valid_interactions: np.ndarray,
    selected: np.ndarray,
    *,
    c_value: float,
    max_iter: int,
    seed: int,
) -> np.ndarray:
    train_parts = [_clip_logit(train_probability)[:, None]]
    valid_parts = [_clip_logit(valid_probability)[:, None]]
    chosen = np.asarray(selected, dtype=np.int64)
    if chosen.size:
        train_parts.append(np.asarray(train_interactions[:, chosen], dtype=np.float64))
        valid_parts.append(np.asarray(valid_interactions[:, chosen], dtype=np.float64))
    x_train = np.concatenate(train_parts, axis=1)
    x_valid = np.concatenate(valid_parts, axis=1)
    scaler = StandardScaler()
    x_train = scaler.fit_transform(x_train)
    x_valid = scaler.transform(x_valid)
    model = LogisticRegression(
        penalty="l1",
        solver="liblinear",
        C=float(c_value),
        class_weight="balanced",
        max_iter=int(max_iter),
        random_state=int(seed),
    )
    model.fit(x_train, train_labels)
    return np.asarray(model.predict_proba(x_valid)[:, 1], dtype=np.float64)


def tune_sparse_sources(
    base_matrix: np.ndarray,
    interactions: Mapping[str, np.ndarray],
    labels: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    base_params: Mapping[str, Any],
    sparse_config: Mapping[str, Any],
    *,
    seed: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Tune top-k and L1 strength on directional scaffold transfer."""
    y = np.asarray(labels, dtype=np.int64)
    split_predictions = []
    baseline_scores = []
    for split_id, (train, valid) in enumerate(inner_splits):
        local_params = dict(base_params)
        local_params["random_state"] = int(seed) + 1009 * split_id
        train_oof = stratified_oof_xgb_predictions(
            base_matrix[train],
            y[train],
            local_params,
            n_splits=int(sparse_config["oof_folds"]),
            seed=int(seed) + 7919 * split_id,
        )
        valid_probability = _fit_xgb_predict(
            base_matrix[train], y[train], base_matrix[valid], local_params
        )
        split_predictions.append((train_oof, valid_probability))
        baseline_scores.append(float(roc_auc_score(y[valid], valid_probability)))

    source_results: dict[str, Any] = {}
    for source, matrix in interactions.items():
        rankings = []
        for split_id, (train, _) in enumerate(inner_splits):
            train_oof, _ = split_predictions[split_id]
            residual = y[train].astype(np.float64) - train_oof
            ranked, scores = rank_interactions(
                matrix[train],
                residual,
                minimum_support=int(sparse_config["minimum_support"]),
                support_tolerance=float(sparse_config["support_tolerance"]),
            )
            rankings.append((ranked, scores))
        trials = []
        for top_k in [int(value) for value in sparse_config["top_k"]]:
            for c_value in [float(value) for value in sparse_config["c_values"]]:
                fold_scores = []
                selected_counts = []
                for split_id, (train, valid) in enumerate(inner_splits):
                    train_oof, valid_probability = split_predictions[split_id]
                    selected = rankings[split_id][0][:top_k]
                    prediction = _fit_sparse_stack(
                        train_oof,
                        matrix[train],
                        y[train],
                        valid_probability,
                        matrix[valid],
                        selected,
                        c_value=c_value,
                        max_iter=int(sparse_config["max_iter"]),
                        seed=int(seed) + split_id,
                    )
                    fold_scores.append(float(roc_auc_score(y[valid], prediction)))
                    selected_counts.append(int(selected.size))
                trials.append(
                    {
                        "top_k": int(top_k),
                        "c": float(c_value),
                        "fold_auc": fold_scores,
                        "mean_auc": float(np.mean(fold_scores)),
                        "selected_counts": selected_counts,
                    }
                )
        best = max(
            trials,
            key=lambda row: (
                float(row["mean_auc"]),
                -int(row["top_k"]),
                -float(row["c"]),
            ),
        )
        source_results[source] = {
            "best": best,
            "trials": trials,
            "eligible_counts": [int(row[0].size) for row in rankings],
        }
    return source_results, {
        "fold_auc": baseline_scores,
        "mean_auc": float(np.mean(baseline_scores)),
    }


def evaluate_sparse_sources_outer(
    base_train: np.ndarray,
    interactions_train: Mapping[str, np.ndarray],
    y_train: np.ndarray,
    base_valid: np.ndarray,
    interactions_valid: Mapping[str, np.ndarray],
    y_valid: np.ndarray,
    inner_splits: Sequence[tuple[np.ndarray, np.ndarray]],
    scaffold_groups: np.ndarray,
    base_params: Mapping[str, Any],
    tuned_sources: Mapping[str, Any],
    sparse_config: Mapping[str, Any],
    seeds: Sequence[int],
) -> dict[str, Any]:
    by_source: dict[str, list[dict[str, Any]]] = {
        source: [] for source in interactions_train
    }
    base_rows = []
    for seed in seeds:
        params = dict(base_params)
        params["random_state"] = int(seed)
        oof_probability = cross_fitted_xgb_predictions(
            base_train, y_train, inner_splits, params
        )
        valid_probability = _fit_xgb_predict(
            base_train, y_train, base_valid, params
        )
        base_auc = float(roc_auc_score(y_valid, valid_probability))
        base_rows.append({"seed": int(seed), "valid_auc": base_auc})
        residual = y_train.astype(np.float64) - oof_probability
        for source, train_matrix in interactions_train.items():
            selected_config = tuned_sources[source]["best"]
            ranked, ranked_scores, selection_meta = rank_stable_interactions(
                train_matrix,
                residual,
                scaffold_groups,
                minimum_support=int(sparse_config["minimum_support"]),
                support_tolerance=float(sparse_config["support_tolerance"]),
            )
            selected = ranked[: int(selected_config["top_k"])]
            prediction = _fit_sparse_stack(
                oof_probability,
                train_matrix,
                y_train,
                valid_probability,
                interactions_valid[source],
                selected,
                c_value=float(selected_config["c"]),
                max_iter=int(sparse_config["max_iter"]),
                seed=int(seed),
            )
            by_source[source].append(
                {
                    "seed": int(seed),
                    "valid_auc": float(roc_auc_score(y_valid, prediction)),
                    "base_auc": base_auc,
                    "delta_vs_base": float(
                        roc_auc_score(y_valid, prediction) - base_auc
                    ),
                    "n_selected": int(selected.size),
                    "selected_indices": selected.tolist(),
                    "selected_scores": ranked_scores[: selected.size].tolist(),
                    "selection": selection_meta,
                }
            )
    result = {
        "baseline": {
            "rows": base_rows,
            "summary": _summary([row["valid_auc"] for row in base_rows]),
        },
        "sources": {},
    }
    for source, rows in by_source.items():
        result["sources"][source] = {
            "rows": rows,
            "summary": _summary([row["valid_auc"] for row in rows]),
            "delta_vs_base": _summary([row["delta_vs_base"] for row in rows]),
            "tuned": tuned_sources[source],
        }
    return result


def _cache_signature(
    representation: Mapping[str, Any], selected_indices: np.ndarray, shuffle_repeats: int
) -> str:
    payload = {
        "representation": dict(representation),
        "selected_indices": np.asarray(selected_indices, dtype=np.int64).tolist(),
        "shuffle_repeats": int(shuffle_repeats),
        "schema": "rooted_wl",
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def load_or_build_blocks(
    bundle,
    selected_indices: np.ndarray,
    representation: Mapping[str, Any],
    *,
    shuffle_repeats: int,
    seed: int,
    cache_path: Path,
) -> tuple[np.ndarray, dict[str, np.ndarray], bool]:
    signature = _cache_signature(representation, selected_indices, shuffle_repeats)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            cached_signature = str(np.asarray(archive["signature"]).reshape(-1)[0])
            if cached_signature != signature:
                raise ValueError(f"feature cache signature mismatch: {cache_path}")
            indices = np.asarray(archive["dataset_indices"], dtype=np.int64)
            blocks = {
                name.removeprefix("block__"): np.asarray(archive[name], dtype=np.float32)
                for name in archive.files
                if name.startswith("block__")
            }
        return indices, blocks, True

    indices, built = _build_dataset_blocks(
        bundle,
        selected_indices,
        "rooted_wl",
        representation,
        int(shuffle_repeats),
        int(seed),
    )
    keep = {
        "node_role",
        "edge_role",
        "node_attribute",
        "edge_attribute",
        "context",
        "node_raw",
        "edge_raw",
        "node_binding",
        "edge_binding",
    }
    for repeat in range(int(shuffle_repeats)):
        for name in ("node_raw", "edge_raw", "node_binding", "edge_binding"):
            keep.add(f"{name}_shuffled_{repeat}")
    blocks = {name: built[name] for name in sorted(keep)}
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            signature=np.asarray([signature]),
            dataset_indices=indices,
            **{f"block__{name}": value for name, value in blocks.items()},
        )
    temporary.replace(cache_path)
    return indices, blocks, False


def _fold_gate(
    folds: Sequence[Mapping[str, Any]],
    getter_candidate,
    getter_baseline,
    *,
    candidate: str,
    baseline: str,
    minimum_delta: float,
    minimum_wins: int,
) -> dict[str, Any]:
    deltas = [
        float(getter_candidate(fold) - getter_baseline(fold)) for fold in folds
    ]
    return {
        "candidate": candidate,
        "baseline": baseline,
        "fold_deltas": deltas,
        "mean_delta": float(np.mean(deltas)),
        "fold_wins": int(np.sum(np.asarray(deltas) > 0.0)),
        "minimum_mean_delta": float(minimum_delta),
        "minimum_fold_wins": int(minimum_wins),
        "passed": bool(
            np.mean(deltas) >= float(minimum_delta)
            and np.sum(np.asarray(deltas) > 0.0) >= int(minimum_wins)
        ),
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# MolHIV task-aligned interaction screen",
        "",
        f"Protocol: `{result['protocol_id']}`",
        "",
        "Official-train-only nested scaffold audit. Official validation/test are not encoded or evaluated.",
        "",
        "## Equal-budget XGBoost tuning",
        "",
        "| view | fixed outer AUC | tuned outer AUC | tuned - fixed |",
        "|---|---:|---:|---:|",
    ]
    for view, row in result["aggregate"]["xgb"].items():
        lines.append(
            f"| `{view}` | {row['fixed']['mean']:.6f} | {row['tuned']['mean']:.6f} | {row['tuning_gain']['mean']:+.6f} |"
        )
    lines.extend(["", "XGBoost gates:", ""])
    for name, gate in result["gates"]["xgb"].items():
        lines.append(
            f"- `{name}`: {gate['mean_delta']:+.6f}, wins {gate['fold_wins']}/{result['n_outer_folds']} — **{'PASS' if gate['passed'] else 'FAIL'}**"
        )
    lines.extend(
        [
            "",
            "## Cross-fitted sparse interaction correction",
            "",
            "| source | outer AUC | delta vs tuned T+A |",
            "|---|---:|---:|",
        ]
    )
    baseline = result["aggregate"]["sparse"]["baseline"]
    lines.append(f"| `tuned_t_a` | {baseline['mean']:.6f} | +0.000000 |")
    for source, row in result["aggregate"]["sparse"]["sources"].items():
        lines.append(
            f"| `{source}` | {row['auc']['mean']:.6f} | {row['delta_vs_base']['mean']:+.6f} |"
        )
    lines.extend(["", "Sparse gates:", ""])
    for name, gate in result["gates"]["sparse"].items():
        lines.append(
            f"- `{name}`: {gate['mean_delta']:+.6f}, wins {gate['fold_wins']}/{result['n_outer_folds']} — **{'PASS' if gate['passed'] else 'FAIL'}**"
        )
    lines.extend(["", f"Decision: **{result['decision']}**", ""])
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    representation = config["representation"]
    screen = config["screen"]
    tuning = config["tuning"]
    sparse_config = config["sparse"]
    fixed_classifier = config["fixed_classifier"]
    folds_path = _resolve(data_config["scaffold_folds"])
    cache_path = _resolve(data_config["feature_cache"])
    result_json = _resolve(config["output_json"])
    result_markdown = _resolve(config["output_markdown"])
    with np.load(folds_path, allow_pickle=False) as archive:
        fold_archive = {name: np.asarray(archive[name]) for name in archive.files}
    bundle = load_molhiv(root=_resolve(data_config["root"]), with_features=True)
    labels = bundle.y.astype(np.int64)
    start = time.perf_counter()

    group_by_index, fold_ids = scaffold_group_mapping(fold_archive)
    selections = [
        _fold_indices(fold_archive, fold, labels, screen) for fold in fold_ids
    ]
    selected_union = np.unique(
        np.concatenate([np.concatenate(pair) for pair in selections])
    ).astype(np.int64)
    print(
        f"selected graphs={selected_union.size}; outer folds={fold_ids}; "
        f"Optuna trials/view/fold={int(tuning['n_trials'])}",
        flush=True,
    )
    unique_indices, blocks, cache_hit = load_or_build_blocks(
        bundle,
        selected_union,
        representation,
        shuffle_repeats=int(screen["shuffle_repeats"]),
        seed=int(screen["seed"]),
        cache_path=cache_path,
    )
    print(f"feature cache: {'hit' if cache_hit else 'built'} at {cache_path}", flush=True)
    index_to_row = {
        int(index): position for position, index in enumerate(unique_indices)
    }
    model_seeds = [int(value) for value in screen["model_seeds"]]
    outer_results = []

    for outer_fold, (train_indices, valid_indices) in zip(
        fold_ids, selections, strict=True
    ):
        fold_indices = np.concatenate([train_indices, valid_indices]).astype(np.int64)
        rows = np.asarray(
            [index_to_row[int(index)] for index in fold_indices], dtype=np.int64
        )
        fold_blocks = {name: matrix[rows] for name, matrix in blocks.items()}
        xgb_views, interaction_views = assemble_task_views(fold_blocks)
        n_train = int(train_indices.size)
        y_train = labels[train_indices]
        y_valid = labels[valid_indices]
        inner_splits, train_groups = nested_scaffold_splits(
            train_indices,
            group_by_index,
            held_out_group=int(outer_fold),
            labels=y_train,
        )
        xgb_result: dict[str, Any] = {}
        for view_id, view in enumerate(XGB_VIEWS):
            matrix = xgb_views[view]
            search = tune_xgb_view(
                matrix[:n_train],
                y_train,
                inner_splits,
                tuning,
                fixed_classifier,
                seed=int(tuning["seed"]) + 1009 * int(outer_fold) + 97 * view_id,
            )
            fixed_params = _base_xgb_params(fixed_classifier, seed=0)
            fixed_outer = evaluate_xgb_outer(
                matrix[:n_train],
                y_train,
                matrix[n_train:],
                y_valid,
                fixed_params,
                model_seeds,
            )
            tuned_outer = evaluate_xgb_outer(
                matrix[:n_train],
                y_train,
                matrix[n_train:],
                y_valid,
                search["best_params"],
                model_seeds,
            )
            xgb_result[view] = {
                "dimension": int(matrix.shape[1]),
                "search": search,
                "fixed_outer": fixed_outer,
                "tuned_outer": tuned_outer,
            }
            print(
                f"[outer {outer_fold}] {view}: fixed="
                f"{fixed_outer['summary']['mean']:.6f}; tuned="
                f"{tuned_outer['summary']['mean']:.6f}; inner="
                f"{search['best_inner_auc']:.6f}",
                flush=True,
            )

        base_params = xgb_result["t_a"]["search"]["best_params"]
        train_interactions = {
            name: matrix[:n_train] for name, matrix in interaction_views.items()
        }
        valid_interactions = {
            name: matrix[n_train:] for name, matrix in interaction_views.items()
        }
        tuned_sparse, sparse_inner_baseline = tune_sparse_sources(
            xgb_views["t_a"][:n_train],
            train_interactions,
            y_train,
            inner_splits,
            base_params,
            sparse_config,
            seed=int(sparse_config["seed"]) + 1009 * int(outer_fold),
        )
        sparse_outer = evaluate_sparse_sources_outer(
            xgb_views["t_a"][:n_train],
            train_interactions,
            y_train,
            xgb_views["t_a"][n_train:],
            valid_interactions,
            y_valid,
            inner_splits,
            train_groups,
            base_params,
            tuned_sparse,
            sparse_config,
            model_seeds,
        )
        outer_results.append(
            {
                "outer_fold": int(outer_fold),
                "n_train": n_train,
                "n_valid": int(valid_indices.size),
                "n_train_positive": int(y_train.sum()),
                "n_valid_positive": int(y_valid.sum()),
                "inner_scaffold_groups": [
                    {
                        "train": int(train.size),
                        "valid": int(valid.size),
                    }
                    for train, valid in inner_splits
                ],
                "xgb": xgb_result,
                "sparse_inner_baseline": sparse_inner_baseline,
                "sparse": sparse_outer,
            }
        )
        print(
            f"[outer {outer_fold}] sparse raw="
            f"{sparse_outer['sources']['raw']['summary']['mean']:.6f}; centered="
            f"{sparse_outer['sources']['centered']['summary']['mean']:.6f}; base="
            f"{sparse_outer['baseline']['summary']['mean']:.6f}",
            flush=True,
        )

    aggregate_xgb = {}
    for view in XGB_VIEWS:
        fixed_values = [
            fold["xgb"][view]["fixed_outer"]["summary"]["mean"]
            for fold in outer_results
        ]
        tuned_values = [
            fold["xgb"][view]["tuned_outer"]["summary"]["mean"]
            for fold in outer_results
        ]
        aggregate_xgb[view] = {
            "fixed": _summary(fixed_values),
            "tuned": _summary(tuned_values),
            "tuning_gain": _summary(
                [tuned - fixed for tuned, fixed in zip(tuned_values, fixed_values, strict=True)]
            ),
        }
    sparse_base_values = [
        fold["sparse"]["baseline"]["summary"]["mean"] for fold in outer_results
    ]
    aggregate_sparse_sources = {}
    for source in INTERACTION_SOURCES:
        auc_values = [
            fold["sparse"]["sources"][source]["summary"]["mean"]
            for fold in outer_results
        ]
        aggregate_sparse_sources[source] = {
            "auc": _summary(auc_values),
            "delta_vs_base": _summary(
                [
                    auc - base
                    for auc, base in zip(auc_values, sparse_base_values, strict=True)
                ]
            ),
        }

    minimum_delta = float(screen["minimum_delta"])
    minimum_wins = int(screen["minimum_fold_wins"])
    xgb_gates = {
        "fixed_raw_vs_fixed_marginals": _fold_gate(
            outer_results,
            lambda fold: fold["xgb"]["f_raw_factorized"]["fixed_outer"]["summary"]["mean"],
            lambda fold: fold["xgb"]["t_a"]["fixed_outer"]["summary"]["mean"],
            candidate="fixed:f_raw_factorized",
            baseline="fixed:t_a",
            minimum_delta=minimum_delta,
            minimum_wins=minimum_wins,
        ),
        "fixed_centered_vs_fixed_marginals": _fold_gate(
            outer_results,
            lambda fold: fold["xgb"]["f_centered"]["fixed_outer"]["summary"]["mean"],
            lambda fold: fold["xgb"]["t_a"]["fixed_outer"]["summary"]["mean"],
            candidate="fixed:f_centered",
            baseline="fixed:t_a",
            minimum_delta=minimum_delta,
            minimum_wins=minimum_wins,
        ),
        "tuned_raw_vs_tuned_marginals": _fold_gate(
            outer_results,
            lambda fold: fold["xgb"]["f_raw_factorized"]["tuned_outer"]["summary"]["mean"],
            lambda fold: fold["xgb"]["t_a"]["tuned_outer"]["summary"]["mean"],
            candidate="tuned:f_raw_factorized",
            baseline="tuned:t_a",
            minimum_delta=minimum_delta,
            minimum_wins=minimum_wins,
        ),
        "tuned_centered_vs_tuned_marginals": _fold_gate(
            outer_results,
            lambda fold: fold["xgb"]["f_centered"]["tuned_outer"]["summary"]["mean"],
            lambda fold: fold["xgb"]["t_a"]["tuned_outer"]["summary"]["mean"],
            candidate="tuned:f_centered",
            baseline="tuned:t_a",
            minimum_delta=minimum_delta,
            minimum_wins=minimum_wins,
        ),
        "raw_relative_gap_change_after_tuning": _fold_gate(
            outer_results,
            lambda fold: (
                fold["xgb"]["f_raw_factorized"]["tuned_outer"]["summary"]["mean"]
                - fold["xgb"]["f_raw_factorized"]["fixed_outer"]["summary"]["mean"]
            ),
            lambda fold: (
                fold["xgb"]["t_a"]["tuned_outer"]["summary"]["mean"]
                - fold["xgb"]["t_a"]["fixed_outer"]["summary"]["mean"]
            ),
            candidate="tuning_gain:f_raw_factorized",
            baseline="tuning_gain:t_a",
            minimum_delta=minimum_delta,
            minimum_wins=minimum_wins,
        ),
        "centered_relative_gap_change_after_tuning": _fold_gate(
            outer_results,
            lambda fold: (
                fold["xgb"]["f_centered"]["tuned_outer"]["summary"]["mean"]
                - fold["xgb"]["f_centered"]["fixed_outer"]["summary"]["mean"]
            ),
            lambda fold: (
                fold["xgb"]["t_a"]["tuned_outer"]["summary"]["mean"]
                - fold["xgb"]["t_a"]["fixed_outer"]["summary"]["mean"]
            ),
            candidate="tuning_gain:f_centered",
            baseline="tuning_gain:t_a",
            minimum_delta=minimum_delta,
            minimum_wins=minimum_wins,
        ),
    }
    sparse_gates = {}
    for source in ("raw", "centered"):
        shuffled = f"{source}_shuffled"
        sparse_gates[f"{source}_vs_base"] = _fold_gate(
            outer_results,
            lambda fold, name=source: fold["sparse"]["sources"][name]["summary"]["mean"],
            lambda fold: fold["sparse"]["baseline"]["summary"]["mean"],
            candidate=f"sparse:{source}",
            baseline="tuned:t_a",
            minimum_delta=minimum_delta,
            minimum_wins=minimum_wins,
        )
        sparse_gates[f"{source}_vs_shuffle"] = _fold_gate(
            outer_results,
            lambda fold, name=source: fold["sparse"]["sources"][name]["summary"]["mean"],
            lambda fold, name=shuffled: fold["sparse"]["sources"][name]["summary"]["mean"],
            candidate=f"sparse:{source}",
            baseline=f"sparse:{shuffled}",
            minimum_delta=minimum_delta,
            minimum_wins=minimum_wins,
        )

    tuning_rescue = any(
        not xgb_gates[f"fixed_{source}_vs_fixed_marginals"]["passed"]
        and xgb_gates[f"tuned_{source}_vs_tuned_marginals"]["passed"]
        and xgb_gates[f"{source}_relative_gap_change_after_tuning"]["passed"]
        for source in ("raw", "centered")
    )
    dense_signal_persists = any(
        xgb_gates[f"fixed_{source}_vs_fixed_marginals"]["passed"]
        and xgb_gates[f"tuned_{source}_vs_tuned_marginals"]["passed"]
        for source in ("raw", "centered")
    )
    residual_pass = any(
        sparse_gates[f"{source}_vs_base"]["passed"]
        and sparse_gates[f"{source}_vs_shuffle"]["passed"]
        for source in ("raw", "centered")
    )
    if tuning_rescue and residual_pass:
        decision = "HYPERPARAMETER_AND_TASK_ALIGNED_INTERACTION_PASS"
    elif tuning_rescue:
        decision = "DENSE_INTERACTION_HYPERPARAMETER_RESCUE_RESIDUAL_FAIL"
    elif residual_pass:
        decision = "TASK_ALIGNED_SPARSE_INTERACTION_PASS_DENSE_TUNING_FAIL"
    elif dense_signal_persists:
        decision = "DENSE_SIGNAL_PERSISTS_BUT_TUNING_IS_NOT_THE_RESCUE"
    else:
        decision = "NO_STABLE_INTERACTION_INCREMENT_AFTER_TUNING"

    result = {
        "protocol_id": str(config["protocol_id"]),
        "audit_boundary": {
            "official_train_used": True,
            "official_validation_encoded": False,
            "official_validation_evaluated": False,
            "official_test_encoded": False,
            "official_test_evaluated": False,
            "selection": "nested official-train-only scaffold folds",
        },
        "data": {
            "dataset": str(data_config["dataset"]),
            "scaffold_folds": str(folds_path),
            "scaffold_folds_sha256": _sha256(folds_path),
            "feature_cache": str(cache_path),
            "feature_cache_hit": bool(cache_hit),
            "selected_graphs": int(selected_union.size),
        },
        "representation": dict(representation),
        "n_outer_folds": len(outer_results),
        "outer_folds": outer_results,
        "aggregate": {
            "xgb": aggregate_xgb,
            "sparse": {
                "baseline": _summary(sparse_base_values),
                "sources": aggregate_sparse_sources,
            },
        },
        "gates": {"xgb": xgb_gates, "sparse": sparse_gates},
        "decision": decision,
        "runtime": {
            "seconds": float(time.perf_counter() - start),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
        "config": config,
    }
    _write_json(result_json, result)
    result_markdown.parent.mkdir(parents=True, exist_ok=True)
    result_markdown.write_text(_render_markdown(result), encoding="utf-8")
    print(f"decision={decision}; wrote {result_json}", flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    run(args.config.expanduser().resolve())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

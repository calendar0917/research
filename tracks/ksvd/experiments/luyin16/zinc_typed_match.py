"""ZINC typed-match candidate with a single XGBoost downstream model.

This is an explicit, auditable candidate for the mentor's ``typed match``
idea; it is not claimed to be an exact reproduction of an unavailable mentor
implementation.  Every atom centre has four exact rooted typed-WL colours
(levels 0..3).  A prototype bank is selected from training-centre signatures
only.  A centre is matched to a prototype softly by prefix agreement:

    level-0 match < level-1 match < level-2 match < level-3 match

The graph readout contains the prototype response mean/max and compact
response-distribution statistics.  Thus all structure/attribute information
is still sent through one graph-level XGBoost regressor; there is no second
model and no K-SVD.

The representation is deliberately train-only at every learned boundary:
inner-fold prototype banks are used for Optuna, the official-valid bank is
fit on official train, and the official-test bank is refit on official
train+valid.  The test labels are never used for representation or model
selection.
"""

from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import importlib.metadata
import json
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import yaml

from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    REPO_ROOT,
    _load_zinc,
    _resolve,
    global_feature_views,
    source_audit,
)
from tracks.ksvd.experiments.luyin16.zinc_motif_count import _cache_to_rows


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_typed_match.yaml"
N_LEVELS = 4
SEARCH_KEYS = (
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


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _sha256_path(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _label_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "median": float(np.median(array)),
    }


def _validate_rows(rows: Sequence[tuple[tuple[object, ...], ...]]) -> None:
    for graph_index, graph_rows in enumerate(rows):
        if len(graph_rows) != N_LEVELS:
            raise ValueError(
                f"graph {graph_index} has {len(graph_rows)} WL levels, expected {N_LEVELS}"
            )
        n_centres = len(graph_rows[0])
        if n_centres <= 0:
            raise ValueError(f"graph {graph_index} has no atom centres")
        if any(len(graph_rows[level]) != n_centres for level in range(N_LEVELS)):
            raise ValueError(f"graph {graph_index} has unaligned centre rows")


def _fit_prototype_bank(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    indices: Sequence[int],
    *,
    n_prototypes: int,
) -> tuple[tuple[tuple[object, ...], ...], dict[str, Any]]:
    """Fit a label-free bank of frequent complete multi-level signatures."""
    if n_prototypes <= 0:
        raise ValueError("n_prototypes must be positive")
    counts: Counter[tuple[object, ...]] = Counter()
    total_centres = 0
    for graph_index in indices:
        graph_rows = rows[int(graph_index)]
        n_centres = len(graph_rows[0])
        total_centres += n_centres
        for centre in range(n_centres):
            signature = tuple(graph_rows[level][centre] for level in range(N_LEVELS))
            counts[signature] += 1
    ranked = sorted(
        counts.items(),
        key=lambda item: (-int(item[1]), repr(item[0])),
    )[: int(n_prototypes)]
    prototypes = tuple(signature for signature, _count in ranked)
    selected_occurrences = int(sum(count for _signature, count in ranked))
    metadata = {
        "fit_graphs": int(len(indices)),
        "fit_centres": int(total_centres),
        "unique_complete_signatures": int(len(counts)),
        "requested_prototypes": int(n_prototypes),
        "selected_prototypes": int(len(prototypes)),
        "selected_occurrence_fraction": float(
            selected_occurrences / max(total_centres, 1)
        ),
        "selected_occurrences": int(selected_occurrences),
    }
    return prototypes, metadata


def _normalise_weights(values: Sequence[float]) -> np.ndarray:
    weights = np.asarray([float(value) for value in values], dtype=np.float64)
    if weights.shape != (N_LEVELS,) or not np.isfinite(weights).all():
        raise ValueError(f"level_weights must contain {N_LEVELS} finite values")
    if np.any(weights < 0.0) or float(weights.sum()) <= 0.0:
        raise ValueError("level_weights must be non-negative and non-zero")
    weights /= float(weights.sum())
    return weights.astype(np.float32)


def _summary(values: np.ndarray) -> np.ndarray:
    array = np.asarray(values, dtype=np.float32)
    if array.size == 0:
        return np.zeros(7, dtype=np.float32)
    return np.asarray(
        [
            float(array.mean()),
            float(array.std()),
            float(np.quantile(array, 0.25)),
            float(np.quantile(array, 0.50)),
            float(np.quantile(array, 0.75)),
            float(np.quantile(array, 0.90)),
            float(array.max()),
        ],
        dtype=np.float32,
    )


def _encode_typed_match(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    prototypes: Sequence[tuple[object, ...]],
    weights: np.ndarray,
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode graph-level soft *same-centre* prefix matches.

    A centre is compared with a prototype using its complete level-0..3
    signature.  We search for the deepest prefix that occurs in the prototype
    bank.  This is important: a level-2 token at one centre cannot be paired
    with a level-0 token from another centre.  If several prototypes share the
    matched prefix, the centre's response mass is divided between them.  The
    resulting vector is therefore a soft, train-only prototype quantisation
    of aligned centre signatures rather than four independent token bags.
    """
    if not prototypes:
        raise ValueError("cannot encode against an empty prototype bank")
    if weights.shape != (N_LEVELS,):
        raise ValueError("unexpected level-weight shape")
    prefix_weights = np.cumsum(weights, dtype=np.float32)
    prefix_lookup: list[dict[tuple[object, ...], tuple[int, ...]]] = []
    for level in range(N_LEVELS):
        buckets: dict[tuple[object, ...], list[int]] = {}
        for prototype_index, prototype in enumerate(prototypes):
            buckets.setdefault(tuple(prototype[: level + 1]), []).append(
                int(prototype_index)
            )
        prefix_lookup.append(
            {key: tuple(value) for key, value in buckets.items()}
        )
    n_prototypes = len(prototypes)
    # Two per-prototype channels plus four seven-number response summaries and
    # five deepest-match buckets (unmatched, levels 0..3).
    stats_width = 4 * 7 + (N_LEVELS + 1)
    output = np.zeros((len(rows), 2 * n_prototypes + stats_width), dtype=np.float32)
    depth_totals = np.zeros(N_LEVELS + 1, dtype=np.float64)
    centre_total = 0.0

    for graph_index, graph_rows in enumerate(rows):
        n_centres = len(graph_rows[0])
        mean_response = np.zeros(n_prototypes, dtype=np.float32)
        max_response = np.zeros(n_prototypes, dtype=np.float32)
        best_response = np.zeros(n_centres, dtype=np.float32)
        depth_counts = np.zeros(N_LEVELS + 1, dtype=np.float64)
        for centre in range(n_centres):
            signature = tuple(graph_rows[level][centre] for level in range(N_LEVELS))
            matched_depth = -1
            candidates: tuple[int, ...] = ()
            for level in range(N_LEVELS - 1, -1, -1):
                candidates = prefix_lookup[level].get(
                    tuple(signature[: level + 1]), ()
                )
                if candidates:
                    matched_depth = level
                    break
            bucket = matched_depth + 1  # 0 = unmatched, 1..4 = level 0..3
            depth_counts[bucket] += 1.0
            if matched_depth < 0:
                continue
            score = float(prefix_weights[matched_depth])
            best_response[centre] = score
            mass = score / float(len(candidates)) / max(float(n_centres), 1.0)
            for prototype_index in candidates:
                mean_response[int(prototype_index)] += float(mass)
                max_response[int(prototype_index)] = max(
                    float(max_response[int(prototype_index)]), score
                )

        depth_totals += depth_counts
        centre_total += float(n_centres)

        offset = 0
        output[graph_index, offset : offset + n_prototypes] = mean_response
        offset += n_prototypes
        output[graph_index, offset : offset + n_prototypes] = max_response
        offset += n_prototypes
        output[graph_index, offset : offset + 7] = _summary(mean_response)
        offset += 7
        output[graph_index, offset : offset + 7] = _summary(max_response)
        offset += 7
        output[graph_index, offset : offset + 7] = _summary(best_response)
        offset += 7
        output[graph_index, offset : offset + 7] = _summary(1.0 - best_response)
        offset += 7
        output[graph_index, offset : offset + N_LEVELS + 1] = (
            depth_counts / max(float(n_centres), 1.0)
        ).astype(np.float32)

    if not np.isfinite(output).all():
        raise FloatingPointError("non-finite typed-match features")
    summary = {
        "dimension": int(output.shape[1]),
        "prototype_count": int(n_prototypes),
        "deepest_match_fraction": [
            float(depth_totals[bucket] / max(centre_total, 1.0))
            for bucket in range(N_LEVELS + 1)
        ],
        "full_level_match_fraction": float(
            depth_totals[N_LEVELS] / max(centre_total, 1.0)
        ),
        "level_weights": [float(value) for value in weights],
        "readout": (
            "per-prototype weighted prefix-match mean/max; response summaries; "
            "best-centre match and residual summaries"
        ),
    }
    return output, summary


def _params_from_trial(
    trial: optuna.Trial,
    ranges: Mapping[str, Sequence[float]],
) -> dict[str, Any]:
    return {
        "n_estimators": trial.suggest_int(
            "n_estimators", int(ranges["n_estimators"][0]), int(ranges["n_estimators"][1])
        ),
        "max_depth": trial.suggest_int(
            "max_depth", int(ranges["max_depth"][0]), int(ranges["max_depth"][1])
        ),
        "learning_rate": trial.suggest_float(
            "learning_rate",
            float(ranges["learning_rate"][0]),
            float(ranges["learning_rate"][1]),
            log=True,
        ),
        "min_child_weight": trial.suggest_float(
            "min_child_weight",
            float(ranges["min_child_weight"][0]),
            float(ranges["min_child_weight"][1]),
            log=True,
        ),
        "subsample": trial.suggest_float(
            "subsample", float(ranges["subsample"][0]), float(ranges["subsample"][1])
        ),
        "colsample_bytree": trial.suggest_float(
            "colsample_bytree",
            float(ranges["colsample_bytree"][0]),
            float(ranges["colsample_bytree"][1]),
        ),
        "reg_lambda": trial.suggest_float(
            "reg_lambda",
            float(ranges["reg_lambda"][0]),
            float(ranges["reg_lambda"][1]),
            log=True,
        ),
        "reg_alpha": trial.suggest_float(
            "reg_alpha",
            float(ranges["reg_alpha"][0]),
            float(ranges["reg_alpha"][1]),
            log=True,
        ),
        "gamma": trial.suggest_float(
            "gamma", float(ranges["gamma"][0]), float(ranges["gamma"][1])
        ),
    }


def _base_params(config: Mapping[str, Any]) -> dict[str, Any]:
    xgb = config["xgboost"]
    return {
        "objective": str(xgb.get("objective", "reg:absoluteerror")),
        "eval_metric": "mae",
        "tree_method": "hist",
        "max_bin": int(xgb.get("max_bin", 256)),
        "random_state": int(xgb.get("model_seed", 0)),
        "n_jobs": int(xgb.get("n_jobs", 4)),
    }


def _fit_mae(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_valid: np.ndarray,
    y_valid: np.ndarray,
    params: Mapping[str, Any],
    config: Mapping[str, Any],
) -> float:
    model_params = _base_params(config)
    model_params.update(dict(params))
    model = XGBRegressor(**model_params)
    model.fit(x_train, y_train)
    return float(mean_absolute_error(y_valid, model.predict(x_valid)))


def _tune(
    fold_features: Sequence[np.ndarray],
    y_train: np.ndarray,
    folds: Sequence[tuple[np.ndarray, np.ndarray]],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tuning = config["tuning"]
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(tuning["seed"])),
    )
    ranges = tuning["ranges"]

    def objective(trial: optuna.Trial) -> float:
        params = _params_from_trial(trial, ranges)
        scores = [
            _fit_mae(
                fold_features[fold_id][fold_train],
                y_train[fold_train],
                fold_features[fold_id][fold_valid],
                y_train[fold_valid],
                params,
                config,
            )
            for fold_id, (fold_train, fold_valid) in enumerate(folds)
        ]
        return float(np.mean(scores))

    study.optimize(
        objective,
        n_trials=int(tuning["n_trials"]),
        show_progress_bar=False,
    )
    best_params = dict(study.best_trial.params)
    best_fold_scores = [
        _fit_mae(
            fold_features[fold_id][fold_train],
            y_train[fold_train],
            fold_features[fold_id][fold_valid],
            y_train[fold_valid],
            best_params,
            config,
        )
        for fold_id, (fold_train, fold_valid) in enumerate(folds)
    ]
    return {
        "view": "s_typed_match",
        "objective": str(config["xgboost"].get("objective", "reg:absoluteerror")),
        "eval_metric": "mae",
        "selection_scope": "official-train shuffled KFold only; fold-local prototype banks",
        "n_trials": int(tuning["n_trials"]),
        "n_folds": int(len(folds)),
        "best_trial": int(study.best_trial.number),
        "best_cv_mae": float(np.mean(best_fold_scores)),
        "best_cv_fold_mae": best_fold_scores,
        "best_params": best_params,
        "trials": [
            {
                "trial": int(trial.number),
                "state": str(trial.state),
                "value": None if trial.value is None else float(trial.value),
                "params": dict(trial.params),
            }
            for trial in study.trials
        ],
    }


def _evaluate(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_eval: np.ndarray,
    y_eval: np.ndarray,
    params: Mapping[str, Any],
    config: Mapping[str, Any],
) -> dict[str, Any]:
    model_params = _base_params(config)
    model_params.update(dict(params))
    model = XGBRegressor(**model_params)
    model.fit(x_train, y_train)
    prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
    return {
        "seed": int(config["xgboost"].get("model_seed", 0)),
        "mae": float(mean_absolute_error(y_eval, prediction)),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
    }


def _concat(global_values: np.ndarray, match_values: np.ndarray) -> np.ndarray:
    left = np.asarray(global_values, dtype=np.float32)
    right = np.asarray(match_values, dtype=np.float32)
    if left.shape[0] != right.shape[0]:
        raise ValueError(f"global/match row mismatch: {left.shape} vs {right.shape}")
    return np.concatenate([left, right], axis=1).astype(np.float32, copy=False)


def _coverage_summary(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    prototypes: Sequence[tuple[object, ...]],
) -> dict[str, Any]:
    sets = [{prototype[level] for prototype in prototypes} for level in range(N_LEVELS)]
    occurrences = np.zeros(N_LEVELS, dtype=np.float64)
    known = np.zeros(N_LEVELS, dtype=np.float64)
    for graph_rows in rows:
        for level in range(N_LEVELS):
            occurrences[level] += float(len(graph_rows[level]))
            known[level] += float(sum(token in sets[level] for token in graph_rows[level]))
    return {
        "graphs": int(len(rows)),
        "centres": int(occurrences[0]),
        "level_known_fraction": [
            float(known[level] / max(occurrences[level], 1.0))
            for level in range(N_LEVELS)
        ],
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    evaluation = result["evaluation"]
    tuning = result["tuning"]
    lines = [
        f"# {result['protocol_id']}",
        "",
        "Auditable candidate for typed matching: train-only frequent multi-level typed-WL prototypes, soft prefix response readout, and one XGBoost regressor.",
        "",
        "> This is a candidate implementation, not an exact claim about the mentor's unavailable internal typed-match code.",
        "",
        "## Protocol",
        "",
        f"- split: `{result['data']['split']}`; sizes: `{result['data']['sizes']}`",
        f"- objective: `{result['protocol']['xgboost_objective']}`; metric: `MAE`; model seed: `{result['protocol']['model_seed']}`",
        f"- Optuna: `{tuning['n_trials']}` trials, `{tuning['n_folds']}` shuffled folds inside official train",
        "- prototype bank: fold-local for CV; official train only for valid; official train+valid only for test",
        "- K-SVD: disabled",
        "",
        "## Representation",
        "",
        f"- typed-WL levels: `0/1/2/3`; prototypes: `{result['representation']['n_prototypes']}` requested / `{result['representation']['selected_prototypes']}` selected",
        f"- level weights: `{result['representation']['level_weights']}`",
        f"- feature dimension: `{result['representation']['dimension']}` (`S`=`{result['representation']['S_dimension']}`, typed-match=`{result['representation']['match_dimension']}`)",
        "- match readout: per-prototype weighted prefix-match mean/max, response summaries, best-centre score and residual summaries",
        "",
        "## Results",
        "",
        "| view | dimension | train CV MAE | valid MAE | test MAE after train+valid refit |",
        "|---|---:|---:|---:|---:|",
        f"| `S + typed_match` | {evaluation['s_typed_match']['dimension']} | {tuning['best_cv_mae']:.6f} | {evaluation['s_typed_match']['valid']['mae']:.6f} | {evaluation['s_typed_match']['test_after_train_valid_refit']['mae']:.6f} |",
        f"| `S` capacity-matched reference | {evaluation['s_same_params']['dimension']} | — | {evaluation['s_same_params']['valid']['mae']:.6f} | {evaluation['s_same_params']['test_after_train_valid_refit']['mae']:.6f} |",
        "",
        "## Coverage audit",
        "",
        "```json",
        json.dumps(result["coverage"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        "## Selected parameters",
        "",
        "```json",
        json.dumps(tuning["best_params"], ensure_ascii=False, indent=2, sort_keys=True),
        "```",
        "",
        f"Runtime: `{result['runtime']['seconds']:.1f}s`; token cache hit: `{result['runtime']['token_cache_hit']}`; script SHA-256: `{result['runtime']['script_sha256']}`.",
        "",
    ]
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    start = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = _resolve(config["data"]["root"])
    representation = config["representation"]
    token_cache_path = _resolve(representation["token_cache"])
    if not token_cache_path.exists():
        raise FileNotFoundError(f"typed-WL token cache is missing: {token_cache_path}")
    weights = _normalise_weights(representation["level_weights"])
    n_prototypes = int(representation["n_prototypes"])
    if n_prototypes <= 0:
        raise ValueError("n_prototypes must be positive")

    with np.load(token_cache_path, allow_pickle=True) as archive:
        all_rows, _unused_morgan, cache_meta = _cache_to_rows(archive)
    datasets = tuple(_load_zinc(data_root, split) for split in ("train", "val", "test"))
    sizes = tuple(len(dataset) for dataset in datasets)
    if len(all_rows) != sum(sizes):
        raise RuntimeError(f"token cache rows={len(all_rows)} but dataset rows={sum(sizes)}")
    _validate_rows(all_rows)
    labels = tuple(
        np.asarray([float(data.y.view(-1)[0]) for data in dataset], dtype=np.float32)
        for dataset in datasets
    )
    global_values = tuple(
        np.asarray(global_feature_views(dataset)["global_all"], dtype=np.float32)
        for dataset in datasets
    )
    rows_train = all_rows[: sizes[0]]
    rows_valid = all_rows[sizes[0] : sizes[0] + sizes[1]]
    rows_test = all_rows[sizes[0] + sizes[1] :]
    y_train, y_valid, y_test = labels
    n_train = len(rows_train)

    kfold = KFold(
        n_splits=int(config["tuning"]["n_splits"]),
        shuffle=True,
        random_state=int(config["tuning"]["split_seed"]),
    )
    folds = [
        (train.astype(np.int64), valid.astype(np.int64))
        for train, valid in kfold.split(np.arange(n_train, dtype=np.int64))
    ]

    fold_features: list[np.ndarray] = []
    fold_bank_metadata: list[dict[str, Any]] = []
    for fold_id, (fold_train, _fold_valid) in enumerate(folds):
        prototypes, bank_meta = _fit_prototype_bank(
            rows_train,
            fold_train,
            n_prototypes=n_prototypes,
        )
        match, match_summary = _encode_typed_match(rows_train, prototypes, weights)
        fold_features.append(_concat(global_values[0], match))
        fold_bank_metadata.append(
            {
                "fold": int(fold_id),
                "bank": bank_meta,
                "all_train_rows_encoding": match_summary,
                "heldout_coverage": _coverage_summary(
                    [rows_train[int(index)] for index in _fold_valid], prototypes
                ),
            }
        )
        print(
            f"fold {fold_id}: prototypes={len(prototypes)} match_dim={match.shape[1]}",
            flush=True,
        )

    tuning = _tune(fold_features, y_train, folds, config)
    print(
        f"typed_match: CV MAE={tuning['best_cv_mae']:.6f}; params={tuning['best_params']}",
        flush=True,
    )
    best_params = tuning["best_params"]

    train_indices = np.arange(n_train, dtype=np.int64)
    train_prototypes, train_bank_meta = _fit_prototype_bank(
        rows_train,
        train_indices,
        n_prototypes=n_prototypes,
    )
    match_train, train_match_summary = _encode_typed_match(
        rows_train, train_prototypes, weights
    )
    match_valid, valid_match_summary = _encode_typed_match(
        rows_valid, train_prototypes, weights
    )
    match_test_train_bank, test_train_bank_summary = _encode_typed_match(
        rows_test, train_prototypes, weights
    )
    x_train = _concat(global_values[0], match_train)
    x_valid = _concat(global_values[1], match_valid)
    x_test_train_bank = _concat(global_values[2], match_test_train_bank)
    valid_result = _evaluate(x_train, y_train, x_valid, y_valid, best_params, config)

    combined_rows = rows_train + rows_valid
    combined_global = np.concatenate(global_values[:2], axis=0)
    combined_labels = np.concatenate([y_train, y_valid], axis=0)
    combined_indices = np.arange(len(combined_rows), dtype=np.int64)
    combined_prototypes, combined_bank_meta = _fit_prototype_bank(
        combined_rows,
        combined_indices,
        n_prototypes=n_prototypes,
    )
    match_combined, combined_match_summary = _encode_typed_match(
        combined_rows, combined_prototypes, weights
    )
    match_test, test_match_summary = _encode_typed_match(
        rows_test, combined_prototypes, weights
    )
    x_combined = _concat(combined_global, match_combined)
    x_test = _concat(global_values[2], match_test)
    test_result = _evaluate(
        x_combined,
        combined_labels,
        x_test,
        y_test,
        best_params,
        config,
    )

    # A capacity-matched S-only reference uses the selected model parameters
    # but is explicitly not presented as an independently tuned S baseline.
    s_valid_result = _evaluate(
        global_values[0], y_train, global_values[1], y_valid, best_params, config
    )
    s_test_result = _evaluate(
        np.concatenate(global_values[:2], axis=0),
        combined_labels,
        global_values[2],
        y_test,
        best_params,
        config,
    )

    output = config["output"]
    result = {
        "protocol_id": config["protocol_id"],
        "status": "completed",
        "data": {
            "root": str(data_root),
            "split": "PyG ZINC subset=True official train/val/test",
            "sizes": {
                "train": int(sizes[0]),
                "valid": int(sizes[1]),
                "test": int(sizes[2]),
            },
            "source": source_audit(data_root),
            "target": {
                name: _label_summary(values)
                for name, values in zip(("train", "valid", "test"), labels, strict=True)
            },
        },
        "representation": {
            "kind": "train-only frequent complete typed-WL prototype bank with soft prefix match",
            "token_cache": str(token_cache_path),
            "token_cache_metadata": cache_meta,
            "n_levels": N_LEVELS,
            "n_prototypes": n_prototypes,
            "selected_prototypes": int(len(train_prototypes)),
            "level_weights": [float(value) for value in weights],
            "S_dimension": int(global_values[0].shape[1]),
            "match_dimension": int(match_train.shape[1]),
            "dimension": int(x_train.shape[1]),
            "centres": "every atom",
            "typed": True,
            "ksvd": False,
            "labels_used_for_representation": False,
            "prototype_selection": "frequency-ranked complete level-0..3 centre signatures, ties by deterministic repr",
            "match_definition": "weighted sum of exact token agreement at each WL level; deeper level is a soft refinement of lower-level match",
            "readout": "one graph-level feature vector sent to one XGBoost regressor",
        },
        "protocol": {
            "xgboost_objective": str(config["xgboost"].get("objective", "reg:absoluteerror")),
            "xgboost_eval_metric": "mae",
            "model_seed": int(config["xgboost"].get("model_seed", 0)),
            "valid_scope": "official train labels and official-train-only prototype bank",
            "test_scope": "official train+valid labels and train+valid-only prototype bank",
            "test_labels_used_for_selection": False,
        },
        "tuning": {
            **tuning,
            "folds": [
                {"train": int(train.size), "valid": int(valid.size)}
                for train, valid in folds
            ],
            "fold_bank_metadata": fold_bank_metadata,
        },
        "coverage": {
            "train_bank_train": {
                "bank": train_bank_meta,
                "train": train_match_summary,
                "valid": valid_match_summary,
                "test": test_train_bank_summary,
            },
            "combined_bank_train_valid": {
                "bank": combined_bank_meta,
                "combined_train_valid": combined_match_summary,
                "test": test_match_summary,
            },
        },
        "evaluation": {
            "metric": "MAE (lower is better)",
            "s_typed_match": {
                "dimension": int(x_train.shape[1]),
                "valid": valid_result,
                "test_after_train_valid_refit": test_result,
            },
            "s_same_params": {
                "dimension": int(global_values[0].shape[1]),
                "note": "capacity-matched reference with typed-match-selected parameters; not independently tuned",
                "valid": s_valid_result,
                "test_after_train_valid_refit": s_test_result,
            },
        },
        "runtime": {
            "seconds": float(time.perf_counter() - start),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "xgboost": importlib.metadata.version("xgboost"),
            "optuna": importlib.metadata.version("optuna"),
            "token_cache_hit": True,
            "script_sha256": _sha256_path(Path(__file__).resolve()),
        },
    }
    result_json = _resolve(output["json"])
    result_markdown = _resolve(output["markdown"])
    _write_json_atomic(result_json, result)
    result_markdown.parent.mkdir(parents=True, exist_ok=True)
    result_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = run(_resolve(args.config))
    print(
        json.dumps(
            {
                "valid_mae": result["evaluation"]["s_typed_match"]["valid"]["mae"],
                "test_mae": result["evaluation"]["s_typed_match"]["test_after_train_valid_refit"]["mae"],
                "cv_mae": result["tuning"]["best_cv_mae"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

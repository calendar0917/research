"""ZINC hierarchical motif counts with cross-level and cross-centre composition.

This is the maximal follow-up to the hierarchical backoff experiment.  It
keeps the exact typed-WL count/backoff view and adds four deliberately
separable feature families:

* same-centre cross-level exact-token combinations (r1+r2, r2+r3 and all
  four WL levels);
* within-molecule unordered co-occurrences of canonical radius-2 parent
  motifs;
* train-only rarity, entropy, concentration and high-order backoff-rate
  statistics;
* train-only top-K molecule-composition tokens for canonical r2 and r3
  parent bags, with an explicit unknown-composition bucket.

All vocabularies and frequency references are fit inside the training portion
of each inner fold.  The official validation split is evaluated once using an
official-train-only vocabulary.  The official test split is evaluated only
after refitting every vocabulary on official train+validation.

The experiment intentionally reports the previous hierarchical-backoff view
with its already selected parameters alongside the new combined view.  Only
the combined view is sent through the new Optuna search.
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
from scipy import sparse
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import KFold
from xgboost import XGBRegressor
import yaml

from tracks.ksvd.experiments.luyin16.zinc_hierarchical_backoff import (
    N_ROUNDS,
    _base_params,
    _cache_to_rows,
    _encode_hierarchical,
    _fit_mae,
    _fit_vocabularies,
    _hstack_dense_sparse,
    _label_summary,
    _params_from_trial,
)
from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (
    REPO_ROOT,
    _load_zinc,
    _resolve,
    global_feature_views,
    source_audit,
)


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_hierarchical_composition.yaml"
COMBINATION_SCHEMAS: tuple[tuple[str, tuple[int, ...]], ...] = (
    ("r1_r2", (1, 2)),
    ("r2_r3", (2, 3)),
    ("r0_r1_r2_r3", (0, 1, 2, 3)),
)
COMPOSITION_LEVELS: tuple[tuple[str, int], ...] = (
    ("r2_parent_bag", 2),
    ("r3_parent_bag", 3),
)
RARE_THRESHOLDS = (1, 5, 20, 100)


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


def _fit_combo_vocabularies(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    indices: Sequence[int],
    *,
    top_k: int,
) -> dict[str, dict[object, int]]:
    vocabularies: dict[str, dict[object, int]] = {}
    for name, levels in COMBINATION_SCHEMAS:
        counts: Counter[object] = Counter()
        for graph_index in indices:
            graph_rows = rows[int(graph_index)]
            n_centres = len(graph_rows[0])
            if any(len(graph_rows[level]) != n_centres for level in levels):
                raise ValueError("cross-level centre rows are not aligned")
            counts.update(
                tuple(graph_rows[level][centre] for level in levels)
                for centre in range(n_centres)
            )
        ranked = sorted(
            counts.items(),
            key=lambda item: (-int(item[1]), repr(item[0])),
        )[: int(top_k)]
        vocabularies[name] = {token: column for column, (token, _count) in enumerate(ranked)}
    return vocabularies


def _encode_combo_features(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    vocabularies: Mapping[str, Mapping[object, int]],
    *,
    top_k: int,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    width_per_schema = 2 * (int(top_k) + 1)
    offsets = {
        name: position * width_per_schema
        for position, (name, _levels) in enumerate(COMBINATION_SCHEMAS)
    }
    total_width = len(COMBINATION_SCHEMAS) * width_per_schema
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    schema_stats: dict[str, dict[str, float]] = {
        name: {"occurrences": 0.0, "known": 0.0, "unknown": 0.0}
        for name, _levels in COMBINATION_SCHEMAS
    }

    for graph_index, graph_rows in enumerate(rows):
        n_centres = len(graph_rows[0])
        graph_counts: dict[int, float] = {}
        for name, levels in COMBINATION_SCHEMAS:
            vocabulary = vocabularies[name]
            stats = schema_stats[name]
            counts: Counter[object] = Counter(
                tuple(graph_rows[level][centre] for level in levels)
                for centre in range(n_centres)
            )
            stats["occurrences"] += float(n_centres)
            unknown = 0
            for token, count in counts.items():
                column = vocabulary.get(token)
                if column is None:
                    unknown += int(count)
                    continue
                base = offsets[name]
                graph_counts[base + int(column)] = graph_counts.get(base + int(column), 0.0) + float(count)
                graph_counts[base + int(top_k) + 1 + int(column)] = graph_counts.get(
                    base + int(top_k) + 1 + int(column), 0.0
                ) + float(count) / max(float(n_centres), 1.0)
            if unknown:
                base = offsets[name]
                graph_counts[base + int(top_k)] = graph_counts.get(base + int(top_k), 0.0) + float(unknown)
                graph_counts[base + int(top_k) + 1 + int(top_k)] = graph_counts.get(
                    base + int(top_k) + 1 + int(top_k), 0.0
                ) + float(unknown) / max(float(n_centres), 1.0)
            stats["known"] += float(n_centres - unknown)
            stats["unknown"] += float(unknown)
        for column, value in graph_counts.items():
            row_indices.append(int(graph_index))
            column_indices.append(int(column))
            values.append(float(value))

    matrix = sparse.coo_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (np.asarray(row_indices, dtype=np.int64), np.asarray(column_indices, dtype=np.int64)),
        ),
        shape=(len(rows), total_width),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    for stats in schema_stats.values():
        stats["known_fraction"] = stats["known"] / max(stats["occurrences"], 1.0)
        stats["unknown_fraction"] = stats["unknown"] / max(stats["occurrences"], 1.0)
    return matrix, {
        "width": int(total_width),
        "width_per_schema": int(width_per_schema),
        "schemas": schema_stats,
    }


def _canonical_parent_key(
    graph_rows: tuple[tuple[object, ...], ...],
    centre: int,
    source_round: int,
    vocabularies: Sequence[Mapping[object, int]],
) -> tuple[int, int]:
    """Return a reusable ID, or ``(-1, -1)`` when no parent is known.

    A fold-local vocabulary can legitimately miss a rare round-0 token in a
    held-out graph.  The explicit sentinel keeps that occurrence in the
    composition/co-occurrence features without turning it into a fold-specific
    accidental integer.
    """
    for target_round in range(int(source_round), -1, -1):
        column = vocabularies[target_round].get(graph_rows[target_round][int(centre)])
        if column is not None:
            return int(target_round), int(column)
    return (-1, -1)


def _graph_parent_keys(
    graph_rows: tuple[tuple[object, ...], ...],
    source_round: int,
    vocabularies: Sequence[Mapping[object, int]],
) -> list[tuple[int, int]]:
    n_centres = len(graph_rows[0])
    if any(len(graph_rows[round_id]) != n_centres for round_id in range(N_ROUNDS)):
        raise ValueError("graph rows are not aligned across WL rounds")
    return [
        _canonical_parent_key(graph_rows, centre, source_round, vocabularies)
        for centre in range(n_centres)
    ]


def _unordered_pair(left: object, right: object) -> tuple[object, object]:
    return (left, right) if repr(left) <= repr(right) else (right, left)


def _fit_cooccurrence_vocab(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    indices: Sequence[int],
    vocabularies: Sequence[Mapping[object, int]],
    *,
    top_k: int,
) -> dict[object, int]:
    counts: Counter[object] = Counter()
    for graph_index in indices:
        keys = _graph_parent_keys(rows[int(graph_index)], 2, vocabularies)
        counts.update(
            _unordered_pair(left, right)
            for position, left in enumerate(keys)
            for right in keys[position + 1 :]
        )
    ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), repr(item[0])))[: int(top_k)]
    return {token: column for column, (token, _count) in enumerate(ranked)}


def _encode_cooccurrence(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    vocabularies: Sequence[Mapping[object, int]],
    pair_vocabulary: Mapping[object, int],
    *,
    top_k: int,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    width = 2 * (int(top_k) + 1)
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    total_pairs = 0.0
    known_pairs = 0.0
    unknown_pairs = 0.0
    for graph_index, graph_rows in enumerate(rows):
        keys = _graph_parent_keys(graph_rows, 2, vocabularies)
        n_pairs = len(keys) * max(len(keys) - 1, 0) // 2
        total_pairs += float(n_pairs)
        counts: Counter[object] = Counter(
            _unordered_pair(left, right)
            for position, left in enumerate(keys)
            for right in keys[position + 1 :]
        )
        row_counts: dict[int, float] = {}
        unknown = 0
        for token, count in counts.items():
            column = pair_vocabulary.get(token)
            if column is None:
                unknown += int(count)
                continue
            row_counts[int(column)] = row_counts.get(int(column), 0.0) + float(count)
            row_counts[int(top_k) + 1 + int(column)] = row_counts.get(
                int(top_k) + 1 + int(column), 0.0
            ) + float(count) / max(float(n_pairs), 1.0)
        if unknown:
            row_counts[int(top_k)] = row_counts.get(int(top_k), 0.0) + float(unknown)
            row_counts[2 * int(top_k) + 1] = row_counts.get(2 * int(top_k) + 1, 0.0) + float(
                unknown
            ) / max(float(n_pairs), 1.0)
        known_pairs += float(n_pairs - unknown)
        unknown_pairs += float(unknown)
        for column, value in row_counts.items():
            row_indices.append(int(graph_index))
            column_indices.append(int(column))
            values.append(float(value))
    matrix = sparse.coo_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (np.asarray(row_indices, dtype=np.int64), np.asarray(column_indices, dtype=np.int64)),
        ),
        shape=(len(rows), width),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    return matrix, {
        "width": int(width),
        "total_pairs": float(total_pairs),
        "known_pairs": float(known_pairs),
        "unknown_pairs": float(unknown_pairs),
        "known_fraction": float(known_pairs / max(total_pairs, 1.0)),
        "unknown_fraction": float(unknown_pairs / max(total_pairs, 1.0)),
    }


def _composition_token(
    graph_rows: tuple[tuple[object, ...], ...],
    source_round: int,
    vocabularies: Sequence[Mapping[object, int]],
) -> tuple[tuple[int, int, int], ...]:
    counts = Counter(_graph_parent_keys(graph_rows, source_round, vocabularies))
    return tuple(
        (int(level), int(column), int(count))
        for (level, column), count in sorted(counts.items())
    )


def _fit_composition_vocabularies(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    indices: Sequence[int],
    vocabularies: Sequence[Mapping[object, int]],
    *,
    top_k: int,
) -> dict[str, dict[object, int]]:
    output: dict[str, dict[object, int]] = {}
    for name, source_round in COMPOSITION_LEVELS:
        counts: Counter[object] = Counter(
            _composition_token(rows[int(graph_index)], source_round, vocabularies)
            for graph_index in indices
        )
        ranked = sorted(counts.items(), key=lambda item: (-int(item[1]), repr(item[0])))[: int(top_k)]
        output[name] = {token: column for column, (token, _count) in enumerate(ranked)}
    return output


def _encode_compositions(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    vocabularies: Sequence[Mapping[object, int]],
    composition_vocabularies: Mapping[str, Mapping[object, int]],
    *,
    top_k: int,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    width_per_level = int(top_k) + 1
    total_width = len(COMPOSITION_LEVELS) * width_per_level
    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    stats: dict[str, dict[str, float]] = {}
    for graph_index, graph_rows in enumerate(rows):
        row_counts: dict[int, float] = {}
        for position, (name, source_round) in enumerate(COMPOSITION_LEVELS):
            token = _composition_token(graph_rows, source_round, vocabularies)
            column = composition_vocabularies[name].get(token)
            if column is None:
                column = int(top_k)
                known = 0.0
            else:
                known = 1.0
            row_counts[position * width_per_level + int(column)] = 1.0
            current = stats.setdefault(name, {"graphs": 0.0, "known": 0.0, "unknown": 0.0})
            current["graphs"] += 1.0
            current["known"] += known
            current["unknown"] += 1.0 - known
        for column, value in row_counts.items():
            row_indices.append(int(graph_index))
            column_indices.append(int(column))
            values.append(float(value))
    matrix = sparse.coo_matrix(
        (
            np.asarray(values, dtype=np.float32),
            (np.asarray(row_indices, dtype=np.int64), np.asarray(column_indices, dtype=np.int64)),
        ),
        shape=(len(rows), total_width),
        dtype=np.float32,
    ).tocsr()
    matrix.sum_duplicates()
    for current in stats.values():
        current["known_fraction"] = current["known"] / max(current["graphs"], 1.0)
        current["unknown_fraction"] = current["unknown"] / max(current["graphs"], 1.0)
    return matrix, {"width": int(total_width), "levels": stats}


def _fit_frequency_reference(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    indices: Sequence[int],
) -> tuple[Counter[object], ...]:
    references = [Counter() for _ in range(N_ROUNDS)]
    for graph_index in indices:
        graph_rows = rows[int(graph_index)]
        for round_id in range(N_ROUNDS):
            references[round_id].update(graph_rows[round_id])
    return tuple(references)


def _normalized_entropy(counts: Counter[object], n: int) -> float:
    if n <= 1:
        return 0.0
    probabilities = np.asarray(list(counts.values()), dtype=np.float64) / float(n)
    entropy = -float(np.sum(probabilities * np.log(np.maximum(probabilities, 1.0e-12))))
    return float(entropy / np.log(float(n)))


def _encode_rare_statistics(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    vocabularies: Sequence[Mapping[object, int]],
    references: Sequence[Counter[object]],
) -> tuple[np.ndarray, dict[str, Any]]:
    """Encode only train-reference rarity and OOD/backoff diagnostics."""
    feature_rows: list[list[float]] = []
    split_stats: dict[str, dict[str, float]] = {
        str(round_id): {
            "occurrences": 0.0,
            "exact_known": 0.0,
            "exact_unknown": 0.0,
            "backoff": 0.0,
            "backoff_previous": 0.0,
            "unresolved": 0.0,
        }
        for round_id in range(N_ROUNDS)
    }
    for graph_rows in rows:
        n_centres = len(graph_rows[0])
        row: list[float] = [float(n_centres), float(np.log1p(n_centres))]
        for round_id in range(N_ROUNDS):
            tokens = graph_rows[round_id]
            counts = Counter(tokens)
            frequencies = np.asarray(
                [int(references[round_id].get(token, 0)) for token in tokens],
                dtype=np.float64,
            )
            token_count = max(len(tokens), 1)
            exact_known = sum(token in vocabularies[round_id] for token in tokens)
            exact_unknown = len(tokens) - exact_known
            backoff = 0
            backoff_previous = 0
            unresolved = 0
            if round_id >= 1:
                for centre in range(len(tokens)):
                    if tokens[centre] in vocabularies[round_id]:
                        continue
                    resolved = False
                    for target_round in range(round_id - 1, -1, -1):
                        if graph_rows[target_round][centre] in vocabularies[target_round]:
                            backoff += 1
                            backoff_previous += int(target_round == round_id - 1)
                            resolved = True
                            break
                    if not resolved:
                        unresolved += 1
            rarity_features = [
                float(len(counts)),
                float(len(counts) / token_count),
                _normalized_entropy(counts, len(tokens)),
                float(max(counts.values(), default=0) / token_count),
                float(sum(sorted(counts.values(), reverse=True)[:5]) / token_count),
            ]
            rarity_features.extend(
                float(np.mean(frequencies <= threshold))
                for threshold in RARE_THRESHOLDS
            )
            rarity_features.extend(
                [
                    float(exact_unknown / token_count),
                    float(backoff / token_count),
                    float(backoff_previous / token_count),
                    float(unresolved / token_count),
                ]
            )
            row.extend(rarity_features)
            current = split_stats[str(round_id)]
            current["occurrences"] += float(len(tokens))
            current["exact_known"] += float(exact_known)
            current["exact_unknown"] += float(exact_unknown)
            current["backoff"] += float(backoff)
            current["backoff_previous"] += float(backoff_previous)
            current["unresolved"] += float(unresolved)
        feature_rows.append(row)
    matrix = np.asarray(feature_rows, dtype=np.float32)
    if matrix.ndim != 2:
        raise RuntimeError(f"rare-stat feature matrix is not 2-D: {matrix.shape}")
    for current in split_stats.values():
        denominator = max(current["occurrences"], 1.0)
        for key in ("exact_known", "exact_unknown", "backoff", "backoff_previous", "unresolved"):
            current[f"{key}_fraction"] = current[key] / denominator
    return matrix, {"width": int(matrix.shape[1]), "rounds": split_stats}


def _build_combined_features(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    global_values: np.ndarray,
    vocabularies: Sequence[Mapping[object, int]],
    *,
    top_k: int,
    fit_rows: Sequence[tuple[tuple[object, ...], ...]] | None = None,
    frequency_indices: Sequence[int],
    vocabulary_indices: Sequence[int],
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    # ``rows`` is the population being encoded; ``fit_rows`` is the population
    # allowed to define combination vocabularies and frequency references.  In
    # particular, official-valid/test rows must never be used to fit their own
    # composition vocabulary.
    fitting_rows = rows if fit_rows is None else fit_rows
    artifacts = _fit_composition_artifacts(
        fitting_rows,
        vocabularies,
        top_k=top_k,
        fit_indices=vocabulary_indices,
        frequency_indices=frequency_indices,
    )
    return _encode_combined_features(
        rows,
        global_values,
        vocabularies,
        artifacts,
        top_k=top_k,
    )


def _fit_composition_artifacts(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    vocabularies: Sequence[Mapping[object, int]],
    *,
    top_k: int,
    fit_indices: Sequence[int],
    frequency_indices: Sequence[int],
) -> dict[str, Any]:
    """Fit all non-WL artifacts on an explicitly supplied graph subset."""
    combo_vocabularies = _fit_combo_vocabularies(rows, fit_indices, top_k=top_k)
    pair_vocabulary = _fit_cooccurrence_vocab(
        rows, fit_indices, vocabularies, top_k=top_k
    )
    composition_vocabularies = _fit_composition_vocabularies(
        rows,
        fit_indices,
        vocabularies,
        top_k=top_k,
    )
    references = _fit_frequency_reference(rows, frequency_indices)
    return {
        "combo_vocabularies": combo_vocabularies,
        "pair_vocabulary": pair_vocabulary,
        "composition_vocabularies": composition_vocabularies,
        "frequency_references": references,
        "frequency_fit_graphs": int(len(frequency_indices)),
        "vocabulary_fit_graphs": int(len(fit_indices)),
    }


def _encode_combined_features(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    global_values: np.ndarray,
    vocabularies: Sequence[Mapping[object, int]],
    artifacts: Mapping[str, Any],
    *,
    top_k: int,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    hierarchical, hierarchical_summary = _encode_hierarchical(
        rows, vocabularies, top_k=top_k
    )
    combo, combo_summary = _encode_combo_features(
        rows,
        artifacts["combo_vocabularies"],
        top_k=top_k,
    )
    pair, pair_summary = _encode_cooccurrence(
        rows,
        vocabularies,
        artifacts["pair_vocabulary"],
        top_k=top_k,
    )
    compositions, composition_summary = _encode_compositions(
        rows,
        vocabularies,
        artifacts["composition_vocabularies"],
        top_k=top_k,
    )
    rare, rare_summary = _encode_rare_statistics(
        rows,
        vocabularies,
        artifacts["frequency_references"],
    )
    feature_matrix = sparse.hstack(
        [
            sparse.csr_matrix(np.asarray(global_values, dtype=np.float32)),
            hierarchical,
            combo,
            pair,
            compositions,
            sparse.csr_matrix(rare),
        ],
        format="csr",
        dtype=np.float32,
    )
    return feature_matrix, {
        "hierarchical": hierarchical_summary,
        "cross_level": combo_summary,
        "cross_centre": pair_summary,
        "molecule_composition": composition_summary,
        "rarity": rare_summary,
        "dimension": int(feature_matrix.shape[1]),
        "frequency_fit_graphs": int(artifacts["frequency_fit_graphs"]),
        "vocabulary_fit_graphs": int(artifacts["vocabulary_fit_graphs"]),
    }


def _tune_combined(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    global_train: np.ndarray,
    y_train: np.ndarray,
    *,
    top_k: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    tuning = config["tuning"]
    n_train = len(rows)
    splitter = KFold(
        n_splits=int(tuning["n_splits"]),
        shuffle=True,
        random_state=int(tuning["split_seed"]),
    )
    folds = [
        (train.astype(np.int64), valid.astype(np.int64))
        for train, valid in splitter.split(np.arange(n_train, dtype=np.int64))
    ]
    fold_features: list[sparse.csr_matrix] = []
    fold_summaries: list[dict[str, Any]] = []
    for fold_id, (fold_train, _fold_valid) in enumerate(folds):
        fold_vocabularies = _fit_vocabularies(rows, fold_train, top_k=top_k)
        fold_matrix, summary = _build_combined_features(
            rows,
            global_train,
            fold_vocabularies,
            top_k=top_k,
            frequency_indices=fold_train,
            vocabulary_indices=fold_train,
        )
        fold_features.append(fold_matrix)
        fold_summaries.append({"fold": int(fold_id), **summary})

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=int(tuning["seed"])),
    )
    ranges = tuning["ranges"]
    model_seed = int(config["xgboost"]["model_seed"])

    def objective(trial: optuna.Trial) -> float:
        params = _params_from_trial(trial, ranges)
        scores = []
        for fold_id, (fold_train, fold_valid) in enumerate(folds):
            matrix = fold_features[fold_id]
            scores.append(
                _fit_mae(
                    matrix[fold_train],
                    y_train[fold_train],
                    matrix[fold_valid],
                    y_train[fold_valid],
                    params,
                    config,
                    model_seed,
                )
            )
        return float(np.mean(scores))

    study.optimize(
        objective,
        n_trials=int(tuning["n_trials"]),
        show_progress_bar=False,
    )
    best_params = dict(study.best_trial.params)
    best_fold_scores = []
    for fold_id, (fold_train, fold_valid) in enumerate(folds):
        matrix = fold_features[fold_id]
        best_fold_scores.append(
            _fit_mae(
                matrix[fold_train],
                y_train[fold_train],
                matrix[fold_valid],
                y_train[fold_valid],
                best_params,
                config,
                model_seed,
            )
        )
    return {
        "view": "s_hierarchical_composition",
        "objective": str(config["xgboost"].get("objective", "reg:absoluteerror")),
        "eval_metric": "mae",
        "selection_scope": "official-train inner shuffled KFold only; fold-local vocabularies and frequency references",
        "n_trials": int(tuning["n_trials"]),
        "n_folds": int(len(folds)),
        "best_trial": int(study.best_trial.number),
        "best_cv_mae": float(np.mean(best_fold_scores)),
        "best_cv_fold_mae": best_fold_scores,
        "best_params": best_params,
        "folds": [
            {"train": int(train.size), "valid": int(valid.size)}
            for train, valid in folds
        ],
        "fold_feature_summaries": fold_summaries,
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
    x_train: sparse.spmatrix | np.ndarray,
    y_train: np.ndarray,
    x_eval: sparse.spmatrix | np.ndarray,
    y_eval: np.ndarray,
    config: Mapping[str, Any],
    params: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    model_params = _base_params(config, int(config["xgboost"]["model_seed"]))
    if params is not None:
        model_params.update(dict(params))
    model = XGBRegressor(**model_params)
    model.fit(x_train, y_train)
    prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
    return {
        "seed": int(config["xgboost"]["model_seed"]),
        "mae": float(mean_absolute_error(y_eval, prediction)),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    evaluation = result["evaluation"]
    lines = [
        "# ZINC hierarchical motif composition",
        "",
        "Hierarchical exact-token backoff plus cross-level, cross-centre, rarity and molecule-composition features.",
        "",
        "## Protocol",
        "",
        f"- split: `{result['data']['split']}`; sizes: `{result['data']['sizes']}`",
        f"- objective: `{result['protocol']['xgboost_objective']}`; model seed: `{result['protocol']['model_seed']}`",
        f"- top-K per feature vocabulary: `{result['representation']['top_k']}`",
        f"- Optuna: `{result['tuning']['n_trials']}` trials, `{result['tuning']['n_folds']}` train-only folds",
        "- all vocabulary/frequency fitting is train-only; test uses train+valid refit",
        "",
        "## Results",
        "",
        "| view | dimension | valid MAE | test MAE after train+valid refit |",
        "|---|---:|---:|---:|",
    ]
    for name in ("s_hierarchical_backoff", "s_hierarchical_composition"):
        row = evaluation[name]
        lines.append(
            f"| `{name}` | {row['dimension']} | {row['valid']['mae']:.6f} | "
            f"{row['test_after_train_valid_refit']['mae']:.6f} |"
        )
    base_valid = float(evaluation["s_hierarchical_backoff"]["valid"]["mae"])
    combined_valid = float(evaluation["s_hierarchical_composition"]["valid"]["mae"])
    base_test = float(evaluation["s_hierarchical_backoff"]["test_after_train_valid_refit"]["mae"])
    combined_test = float(evaluation["s_hierarchical_composition"]["test_after_train_valid_refit"]["mae"])
    lines.extend(
        [
            "",
            "## Increment over hierarchical backoff",
            "",
            f"- valid MAE reduction: `{base_valid - combined_valid:+.6f}`",
            f"- test MAE reduction: `{base_test - combined_test:+.6f}`",
            "",
            "## Feature blocks",
            "",
            "```json",
            json.dumps(result["feature_blocks"], ensure_ascii=False, indent=2, sort_keys=True),
            "```",
            "",
            f"Runtime: `{result['runtime']['seconds']:.1f}s`; script SHA-256: `{result['runtime']['script_sha256']}`.",
            "",
        ]
    )
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    start = time.perf_counter()
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_root = _resolve(config["data"]["root"])
    representation = config["representation"]
    top_k = int(representation["top_k"])
    if top_k <= 0:
        raise ValueError("top_k must be positive")
    token_cache_path = _resolve(representation["token_cache"])
    if not token_cache_path.exists():
        raise FileNotFoundError(f"typed-WL token cache is missing: {token_cache_path}")
    with np.load(token_cache_path, allow_pickle=True) as archive:
        all_rows, _unused_morgan, split_meta = _cache_to_rows(archive)
    datasets = tuple(_load_zinc(data_root, split) for split in ("train", "val", "test"))
    sizes = tuple(len(dataset) for dataset in datasets)
    if len(all_rows) != sum(sizes):
        raise RuntimeError(f"token cache rows={len(all_rows)} but loaded data rows={sum(sizes)}")
    labels = tuple(
        np.asarray([float(data.y.view(-1)[0]) for data in dataset], dtype=np.float32)
        for dataset in datasets
    )
    global_views = tuple(
        np.asarray(global_feature_views(dataset)["global_all"], dtype=np.float32)
        for dataset in datasets
    )
    rows_train = all_rows[: sizes[0]]
    rows_valid = all_rows[sizes[0] : sizes[0] + sizes[1]]
    rows_test = all_rows[sizes[0] + sizes[1] :]
    train_indices = np.arange(sizes[0], dtype=np.int64)
    train_vocabularies = _fit_vocabularies(rows_train, train_indices, top_k=top_k)
    tuning = _tune_combined(
        rows_train,
        global_views[0],
        labels[0],
        top_k=top_k,
        config=config,
    )
    tuned_params = tuning["best_params"]

    train_features, train_summary = _build_combined_features(
        rows_train,
        global_views[0],
        train_vocabularies,
        top_k=top_k,
        frequency_indices=train_indices,
        vocabulary_indices=train_indices,
    )
    valid_features, valid_summary = _build_combined_features(
        rows_valid,
        global_views[1],
        train_vocabularies,
        top_k=top_k,
        fit_rows=rows_train,
        frequency_indices=train_indices,
        vocabulary_indices=train_indices,
    )

    combined_rows = rows_train + rows_valid
    combined_indices = np.arange(len(combined_rows), dtype=np.int64)
    combined_vocabularies = _fit_vocabularies(
        combined_rows,
        combined_indices,
        top_k=top_k,
    )
    combined_global = np.concatenate(global_views[:2], axis=0)
    combined_features, combined_summary = _build_combined_features(
        combined_rows,
        combined_global,
        combined_vocabularies,
        top_k=top_k,
        frequency_indices=combined_indices,
        vocabulary_indices=combined_indices,
    )
    test_features, test_summary = _build_combined_features(
        rows_test,
        global_views[2],
        combined_vocabularies,
        top_k=top_k,
        fit_rows=combined_rows,
        frequency_indices=combined_indices,
        vocabulary_indices=combined_indices,
    )

    # Reuse the previous tuned hierarchical-backoff parameters as a frozen
    # reference.  The new combined representation gets the Optuna parameters.
    baseline_train, baseline_train_summary = _encode_hierarchical(
        rows_train, train_vocabularies, top_k=top_k
    )
    baseline_valid, baseline_valid_summary = _encode_hierarchical(
        rows_valid, train_vocabularies, top_k=top_k
    )
    baseline_combined, baseline_combined_summary = _encode_hierarchical(
        combined_rows, combined_vocabularies, top_k=top_k
    )
    baseline_test, baseline_test_summary = _encode_hierarchical(
        rows_test, combined_vocabularies, top_k=top_k
    )
    baseline_x_train = _hstack_dense_sparse(global_views[0], baseline_train)
    baseline_x_valid = _hstack_dense_sparse(global_views[1], baseline_valid)
    baseline_x_combined = _hstack_dense_sparse(combined_global, baseline_combined)
    baseline_x_test = _hstack_dense_sparse(global_views[2], baseline_test)
    combined_y = np.concatenate(labels[:2], axis=0)
    evaluation = {
        "s_hierarchical_backoff": {
            "dimension": int(baseline_x_train.shape[1]),
            "valid": _evaluate(
                baseline_x_train, labels[0], baseline_x_valid, labels[1], config
            ),
            "test_after_train_valid_refit": _evaluate(
                baseline_x_combined,
                combined_y,
                baseline_x_test,
                labels[2],
                config,
            ),
        },
        "s_hierarchical_composition": {
            "dimension": int(train_features.shape[1]),
            "valid": _evaluate(
                train_features,
                labels[0],
                valid_features,
                labels[1],
                config,
                tuned_params,
            ),
            "test_after_train_valid_refit": _evaluate(
                combined_features,
                combined_y,
                test_features,
                labels[2],
                config,
                tuned_params,
            ),
        },
    }
    result = {
        "protocol_id": config["protocol_id"],
        "status": "completed",
        "data": {
            "root": str(data_root),
            "split": "PyG ZINC subset=True official train/val/test",
            "sizes": {"train": sizes[0], "valid": sizes[1], "test": sizes[2]},
            "source": source_audit(data_root),
            "target": {
                name: _label_summary(values)
                for name, values in zip(("train", "valid", "test"), labels, strict=True)
            },
        },
        "representation": {
            "radius": int(representation.get("radius", 3)),
            "wl_rounds": list(range(N_ROUNDS)),
            "centres": "every atom",
            "top_k": top_k,
            "token_cache": str(token_cache_path),
            "token_cache_split_meta": split_meta,
            "base": "exact typed-WL counts plus nearest known same-centre lower-order backoff",
            "cross_level": [
                {"name": name, "rounds": list(levels)}
                for name, levels in COMBINATION_SCHEMAS
            ],
            "cross_centre": "unordered within-molecule radius-2 canonical-parent motif pairs",
            "rarity": "train-reference token frequency thresholds, within-graph entropy/concentration, OOV and backoff fractions",
            "molecule_composition": [
                {"name": name, "source_round": int(round_id)}
                for name, round_id in COMPOSITION_LEVELS
            ],
        },
        "protocol": {
            "xgboost_objective": str(config["xgboost"].get("objective", "reg:absoluteerror")),
            "xgboost_eval_metric": "mae",
            "model_seed": int(config["xgboost"]["model_seed"]),
            "valid_vocab_scope": "official train only",
            "test_vocab_scope": "official train+valid only",
            "frequency_scope": "official train only for valid; official train+valid only for test refit",
        },
        "tuning": tuning,
        "feature_blocks": {
            "official_train": train_summary,
            "official_valid": valid_summary,
            "combined_train_valid": combined_summary,
            "official_test": test_summary,
            "baseline_hierarchical_train": baseline_train_summary,
            "baseline_hierarchical_valid": baseline_valid_summary,
            "baseline_hierarchical_combined": baseline_combined_summary,
            "baseline_hierarchical_test": baseline_test_summary,
        },
        "evaluation": evaluation,
        "runtime": {
            "seconds": float(time.perf_counter() - start),
            "python": sys.version,
            "platform": platform.platform(),
            "numpy": importlib.metadata.version("numpy"),
            "scipy": importlib.metadata.version("scipy"),
            "scikit_learn": importlib.metadata.version("scikit-learn"),
            "xgboost": importlib.metadata.version("xgboost"),
            "optuna": importlib.metadata.version("optuna"),
            "script_sha256": _sha256_path(Path(__file__).resolve()),
        },
    }
    output = config["output"]
    result_json = _resolve(output["json"])
    result_markdown = _resolve(output["markdown"])
    _write_json_atomic(result_json, result)
    result_markdown.parent.mkdir(parents=True, exist_ok=True)
    result_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args()
    result = run(args.config)
    for name in ("s_hierarchical_backoff", "s_hierarchical_composition"):
        print(
            f"{name}: valid_mae={result['evaluation'][name]['valid']['mae']:.6f} "
            f"test_mae={result['evaluation'][name]['test_after_train_valid_refit']['mae']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

"""ZINC hierarchical exact-token counts with train-fold backoff.

The existing typed-WL count route keeps the most frequent exact token IDs at
each refinement round and sends every other occurrence to one OOV bucket.  On
ZINC this is especially lossy at rounds 2 and 3: a high-order environment may
be new while its lower-order parent is already common and reusable.

This experiment keeps the exact count view and adds a structured fallback:
for an unseen token at round ``r``, the token at the same centre is looked up
at rounds ``r-1, r-2, ...``.  The first lower-order token present in the
train-only vocabulary receives the occurrence in a source/target-specific
backoff block.  Thus the model can distinguish, for example, a known r=3
motif from an unseen r=3 motif whose r=2 parent is known, without assigning a
meaningless ordinal to a whole molecule.

The official validation split is evaluated after fitting vocabularies on
official train only.  Test is evaluated only after refitting the vocabulary on
train+validation.  The first diagnostic uses the already selected
absolute-error XGBoost parameters from the collision-free typed-WL count
experiment, so the comparison isolates the feature representation.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import optuna
from pathlib import Path
import platform
import sys
import time
from typing import Any, Mapping, Sequence

import numpy as np
from scipy import sparse
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
from tracks.ksvd.experiments.luyin16.zinc_motif_count import (
    _cache_to_rows,
    _encode_wl_counts,
    _fit_topk_vocab,
)


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/zinc_hierarchical_backoff.yaml"
N_ROUNDS = 4


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


def _hstack_dense_sparse(dense: np.ndarray, extra: sparse.spmatrix) -> sparse.csr_matrix:
    return sparse.hstack(
        [sparse.csr_matrix(np.asarray(dense, dtype=np.float32)), extra],
        format="csr",
        dtype=np.float32,
    )


def _backoff_blocks() -> tuple[tuple[int, int], ...]:
    """All possible source-round to lower target-round blocks in fixed order."""
    return tuple(
        (source, target)
        for source in range(1, N_ROUNDS)
        for target in range(source - 1, -1, -1)
    )


def _fit_vocabularies(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    indices: Sequence[int],
    *,
    top_k: int,
) -> tuple[dict[object, int], ...]:
    """Fit one exact top-K vocabulary per WL round on the supplied rows."""
    # Reuse the established deterministic vocabulary implementation.  It
    # counts the ``(round, digest)`` token keys and sorts ties by repr.
    return _fit_topk_vocab(rows, indices, rounds=N_ROUNDS, top_k=int(top_k))


def _add_count(
    counts: dict[int, float],
    column: int,
    value: float = 1.0,
) -> None:
    counts[int(column)] = counts.get(int(column), 0.0) + float(value)


def _encode_hierarchical(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    vocabularies: Sequence[Mapping[object, int]],
    *,
    top_k: int,
) -> tuple[sparse.csr_matrix, dict[str, Any]]:
    """Encode exact counts plus nearest-known lower-order fallback counts.

    Exact blocks have raw and centre-normalized counts for each round.  For an
    unseen source token, only one fallback block is incremented: the nearest
    lower round whose same-centre token is in vocabulary.  This makes the
    blocks mutually interpretable and avoids counting the same occurrence at
    every lower level.  An unresolved scalar is retained as an explicit tail
    rather than silently dropping the occurrence.
    """
    if len(vocabularies) != N_ROUNDS:
        raise ValueError("expected one vocabulary for each of four WL rounds")
    width_per_round = 2 * (int(top_k) + 1)
    exact_width = N_ROUNDS * width_per_round
    blocks = _backoff_blocks()
    block_width = width_per_round
    block_offsets = {
        block: exact_width + position * block_width
        for position, block in enumerate(blocks)
    }
    unresolved_offset = exact_width + len(blocks) * block_width
    total_width = unresolved_offset + 2 * (N_ROUNDS - 1)

    row_indices: list[int] = []
    column_indices: list[int] = []
    values: list[float] = []
    source_stats: dict[str, dict[str, float]] = {
        str(source): {
            "occurrences": 0.0,
            "exact_known": 0.0,
            "exact_unknown": 0.0,
            "unresolved": 0.0,
        }
        for source in range(N_ROUNDS)
    }
    fallback_stats: dict[str, float] = {f"{source}->{target}": 0.0 for source, target in blocks}

    for graph_index, graph_rows in enumerate(rows):
        if len(graph_rows) != N_ROUNDS:
            raise ValueError(f"graph {graph_index} has {len(graph_rows)} rounds, expected {N_ROUNDS}")
        n_centres = len(graph_rows[0])
        if any(len(graph_rows[round_id]) != n_centres for round_id in range(N_ROUNDS)):
            raise ValueError(f"graph {graph_index} has unaligned centre rows across rounds")
        counts: dict[int, float] = {}
        for source in range(N_ROUNDS):
            tokens = graph_rows[source]
            stats = source_stats[str(source)]
            stats["occurrences"] += float(len(tokens))
            for centre_position, token in enumerate(tokens):
                exact_column = vocabularies[source].get(token)
                exact_base = source * width_per_round
                if exact_column is not None:
                    _add_count(counts, exact_base + int(exact_column))
                    _add_count(counts, exact_base + int(top_k) + 1 + int(exact_column), 1.0 / max(float(n_centres), 1.0))
                    stats["exact_known"] += 1.0
                    continue

                oov_column = exact_base + int(top_k)
                _add_count(counts, oov_column)
                _add_count(
                    counts,
                    exact_base + int(top_k) + 1 + int(top_k),
                    1.0 / max(float(n_centres), 1.0),
                )
                stats["exact_unknown"] += 1.0
                resolved = False
                for target in range(source - 1, -1, -1):
                    parent = graph_rows[target][centre_position]
                    parent_column = vocabularies[target].get(parent)
                    if parent_column is None:
                        continue
                    base = block_offsets[(source, target)]
                    _add_count(counts, base + int(parent_column))
                    _add_count(
                        counts,
                        base + int(top_k) + 1 + int(parent_column),
                        1.0 / max(float(n_centres), 1.0),
                    )
                    fallback_stats[f"{source}->{target}"] += 1.0
                    resolved = True
                    break
                if not resolved:
                    source_tail = source - 1
                    _add_count(counts, unresolved_offset + 2 * source_tail)
                    _add_count(
                        counts,
                        unresolved_offset + 2 * source_tail + 1,
                        1.0 / max(float(n_centres), 1.0),
                    )
                    source_stats[str(source)]["unresolved"] += 1.0

        for column, value in counts.items():
            if value == 0.0:
                continue
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
    summary = {
        "exact_width": int(exact_width),
        "backoff_blocks": [
            {
                "source_round": int(source),
                "target_round": int(target),
                "width": int(block_width),
                "resolved_occurrences": float(fallback_stats[f"{source}->{target}"]),
            }
            for source, target in blocks
        ],
        "unresolved_width": int(2 * (N_ROUNDS - 1)),
        "total_width": int(total_width),
        "source_rounds": source_stats,
    }
    return matrix, summary


def _base_params(config: Mapping[str, Any], seed: int) -> dict[str, Any]:
    xgb_config = config["xgboost"]
    params = dict(xgb_config["params"])
    params.update(
        {
            "objective": str(xgb_config.get("objective", "reg:absoluteerror")),
            "eval_metric": "mae",
            "tree_method": "hist",
            "max_bin": int(xgb_config.get("max_bin", 256)),
            "random_state": int(seed),
            "n_jobs": int(xgb_config.get("n_jobs", 4)),
        }
    )
    return params


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


def _fit_mae(
    x_train: sparse.spmatrix | np.ndarray,
    y_train: np.ndarray,
    x_valid: sparse.spmatrix | np.ndarray,
    y_valid: np.ndarray,
    params: Mapping[str, Any],
    config: Mapping[str, Any],
    seed: int,
) -> float:
    model_params = _base_params(config, seed)
    model_params.update(dict(params))
    model = XGBRegressor(**model_params)
    model.fit(x_train, y_train)
    return float(mean_absolute_error(y_valid, model.predict(x_valid)))


def _tune_hierarchical(
    rows: Sequence[tuple[tuple[object, ...], ...]],
    global_train: np.ndarray,
    y_train: np.ndarray,
    *,
    top_k: int,
    config: Mapping[str, Any],
) -> dict[str, Any]:
    """Tune only the hierarchical view with fold-local vocabularies.

    The vocabulary is refit inside each fold before the fold score is
    computed.  This matters here because a high-order exact token being
    present in the full official train set must not make it appear known in
    an inner validation fold.
    """
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
    fold_coverage: list[dict[str, Any]] = []
    for fold_id, (fold_train, _fold_valid) in enumerate(folds):
        fold_vocab = _fit_vocabularies(rows, fold_train, top_k=top_k)
        fold_matrix, coverage = _encode_hierarchical(rows, fold_vocab, top_k=top_k)
        fold_features.append(fold_matrix)
        fold_coverage.append(
            {
                "fold": int(fold_id),
                "fit_graphs": int(fold_train.size),
                "heldout_graphs": int(n_train - fold_train.size),
                "coverage": coverage,
            }
        )

    base_seed = int(tuning["seed"])
    study = optuna.create_study(
        direction="minimize",
        sampler=optuna.samplers.TPESampler(seed=base_seed),
    )
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    ranges = tuning["ranges"]

    def objective(trial: optuna.Trial) -> float:
        params = _params_from_trial(trial, ranges)
        scores = []
        for fold_id, (fold_train, fold_valid) in enumerate(folds):
            scores.append(
                _fit_mae(
                    _hstack_dense_sparse(global_train[fold_train], fold_features[fold_id][fold_train]),
                    y_train[fold_train],
                    _hstack_dense_sparse(global_train[fold_valid], fold_features[fold_id][fold_valid]),
                    y_train[fold_valid],
                    params,
                    config,
                    int(config["xgboost"]["model_seed"]),
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
        best_fold_scores.append(
            _fit_mae(
                _hstack_dense_sparse(global_train[fold_train], fold_features[fold_id][fold_train]),
                y_train[fold_train],
                _hstack_dense_sparse(global_train[fold_valid], fold_features[fold_id][fold_valid]),
                y_train[fold_valid],
                best_params,
                config,
                int(config["xgboost"]["model_seed"]),
            )
        )
    return {
        "view": "s_hierarchical_backoff",
        "objective": str(config["xgboost"].get("objective", "reg:absoluteerror")),
        "eval_metric": "mae",
        "selection_scope": "official-train inner shuffled KFold only; fold-local vocabularies",
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
        "fold_vocabulary_coverage": fold_coverage,
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
    seed = int(config["xgboost"]["model_seed"])
    model_params = _base_params(config, seed)
    if params is not None:
        model_params.update(dict(params))
    model = XGBRegressor(**model_params)
    model.fit(x_train, y_train)
    prediction = np.asarray(model.predict(x_eval), dtype=np.float64)
    return {
        "seed": seed,
        "mae": float(mean_absolute_error(y_eval, prediction)),
        "prediction_mean": float(prediction.mean()),
        "prediction_std": float(prediction.std()),
    }


def _label_summary(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std": float(array.std()),
        "min": float(array.min()),
        "max": float(array.max()),
        "median": float(np.median(array)),
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    evaluation = result["evaluation"]
    lines = [
        "# ZINC hierarchical exact-token backoff",
        "",
        "Train-only exact typed-WL vocabularies with same-centre lower-order fallback.",
        "",
        "## Protocol",
        "",
        f"- split: `{result['data']['split']}`; sizes: `{result['data']['sizes']}`",
        f"- objective: `{result['protocol']['xgboost_objective']}`; model seed: `{result['protocol']['model_seed']}`",
        f"- top-K per WL round: `{result['representation']['top_k']}`",
        "- valid vocabulary: official train only; test vocabulary: train+valid only",
        f"- Optuna: `{result['tuning']['n_trials']}` trials, `{result['tuning']['n_folds']}` train-only folds",
        "",
        "## Results",
        "",
        "| view | dimension | valid MAE | test MAE after train+valid refit |",
        "|---|---:|---:|---:|",
    ]
    for name in ("s_wl_count", "s_hierarchical_backoff"):
        row = evaluation[name]
        lines.append(
            f"| `{name}` | {row['dimension']} | {row['valid']['mae']:.6f} | "
            f"{row['test_after_train_valid_refit']['mae']:.6f} |"
        )
    base_valid = float(evaluation["s_wl_count"]["valid"]["mae"])
    hier_valid = float(evaluation["s_hierarchical_backoff"]["valid"]["mae"])
    base_test = float(evaluation["s_wl_count"]["test_after_train_valid_refit"]["mae"])
    hier_test = float(evaluation["s_hierarchical_backoff"]["test_after_train_valid_refit"]["mae"])
    lines.extend(
        [
            "",
            "## Increment (positive means lower MAE)",
            "",
            f"- valid: `{base_valid - hier_valid:+.6f}`",
            f"- test: `{base_test - hier_test:+.6f}`",
            "",
            "## Backoff coverage",
            "",
        ]
    )
    for split_name, view in result["coverage"].items():
        lines.append(f"### {split_name}")
        lines.append("")
        lines.append("```json")
        lines.append(json.dumps(view, ensure_ascii=False, indent=2, sort_keys=True))
        lines.append("```")
        lines.append("")
    lines.extend(
        [
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
        raise FileNotFoundError(
            f"typed-WL token cache is missing: {token_cache_path}; run zinc_motif_count first"
        )

    with np.load(token_cache_path, allow_pickle=True) as archive:
        all_rows, _unused_morgan, split_meta = _cache_to_rows(archive)
    datasets = tuple(_load_zinc(data_root, split) for split in ("train", "val", "test"))
    sizes = tuple(len(dataset) for dataset in datasets)
    expected_total = sum(sizes)
    if len(all_rows) != expected_total:
        raise RuntimeError(f"token cache rows={len(all_rows)} but loaded data rows={expected_total}")
    labels = tuple(
        np.asarray([float(data.y.view(-1)[0]) for data in dataset], dtype=np.float32)
        for dataset in datasets
    )
    global_blocks = tuple(global_feature_views(dataset) for dataset in datasets)
    global_views = tuple(np.asarray(block["global_all"], dtype=np.float32) for block in global_blocks)
    rows_train = all_rows[: sizes[0]]
    rows_valid = all_rows[sizes[0] : sizes[0] + sizes[1]]
    rows_test = all_rows[sizes[0] + sizes[1] :]
    train_indices = np.arange(len(rows_train), dtype=np.int64)
    train_vocab = _fit_vocabularies(rows_train, train_indices, top_k=top_k)

    tuning = _tune_hierarchical(
        rows_train,
        global_views[0],
        labels[0],
        top_k=top_k,
        config=config,
    )
    tuned_params = tuning["best_params"]

    exact_train = _encode_wl_counts(rows_train, train_vocab, rounds=N_ROUNDS, top_k=top_k)
    exact_valid = _encode_wl_counts(rows_valid, train_vocab, rounds=N_ROUNDS, top_k=top_k)
    hierarchical_train, train_coverage = _encode_hierarchical(rows_train, train_vocab, top_k=top_k)
    hierarchical_valid, valid_coverage = _encode_hierarchical(rows_valid, train_vocab, top_k=top_k)
    x_train = {
        "s_wl_count": _hstack_dense_sparse(global_views[0], exact_train),
        "s_hierarchical_backoff": _hstack_dense_sparse(global_views[0], hierarchical_train),
    }
    x_valid = {
        "s_wl_count": _hstack_dense_sparse(global_views[1], exact_valid),
        "s_hierarchical_backoff": _hstack_dense_sparse(global_views[1], hierarchical_valid),
    }

    combined_rows = rows_train + rows_valid
    combined_indices = np.arange(len(combined_rows), dtype=np.int64)
    combined_vocab = _fit_vocabularies(combined_rows, combined_indices, top_k=top_k)
    exact_combined = _encode_wl_counts(combined_rows, combined_vocab, rounds=N_ROUNDS, top_k=top_k)
    exact_test = _encode_wl_counts(rows_test, combined_vocab, rounds=N_ROUNDS, top_k=top_k)
    hierarchical_combined, combined_coverage = _encode_hierarchical(
        combined_rows, combined_vocab, top_k=top_k
    )
    hierarchical_test, test_coverage = _encode_hierarchical(rows_test, combined_vocab, top_k=top_k)
    x_combined = {
        "s_wl_count": _hstack_dense_sparse(np.concatenate(global_views[:2], axis=0), exact_combined),
        "s_hierarchical_backoff": _hstack_dense_sparse(
            np.concatenate(global_views[:2], axis=0), hierarchical_combined
        ),
    }
    x_test = {
        "s_wl_count": _hstack_dense_sparse(global_views[2], exact_test),
        "s_hierarchical_backoff": _hstack_dense_sparse(global_views[2], hierarchical_test),
    }

    evaluation: dict[str, Any] = {}
    for name in ("s_wl_count", "s_hierarchical_backoff"):
        evaluation[name] = {
            "dimension": int(x_train[name].shape[1]),
            "valid": _evaluate(x_train[name], labels[0], x_valid[name], labels[1], config),
            "test_after_train_valid_refit": _evaluate(
                x_combined[name],
                np.concatenate(labels[:2], axis=0),
                x_test[name],
                labels[2],
                config,
                tuned_params if name == "s_hierarchical_backoff" else None,
            ),
        }
        if name == "s_hierarchical_backoff":
            evaluation[name]["valid"] = _evaluate(
                x_train[name],
                labels[0],
                x_valid[name],
                labels[1],
                config,
                tuned_params,
            )

    output = config["output"]
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
            "exact_definition": "raw count + centre-normalized count per round with OOV",
            "backoff_definition": "unseen source token falls to nearest same-centre lower-order token in train-only vocabulary",
            "backoff_blocks": [list(block) for block in _backoff_blocks()],
        },
        "protocol": {
            "xgboost_objective": str(config["xgboost"].get("objective", "reg:absoluteerror")),
            "xgboost_eval_metric": "mae",
            "xgboost_params": dict(config["xgboost"]["params"]),
            "model_seed": int(config["xgboost"]["model_seed"]),
            "valid_vocab_scope": "official train only",
            "test_vocab_scope": "official train+valid only",
        },
        "tuning": tuning,
        "coverage": {
            "train": train_coverage,
            "valid": valid_coverage,
            "combined_train_valid": combined_coverage,
            "test": test_coverage,
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
            "script_sha256": _sha256_path(Path(__file__).resolve()),
        },
    }
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
    for name in ("s_wl_count", "s_hierarchical_backoff"):
        print(
            f"{name}: valid_mae={result['evaluation'][name]['valid']['mae']:.6f} "
            f"test_mae={result['evaluation'][name]['test_after_train_valid_refit']['mae']:.6f}",
            flush=True,
        )


if __name__ == "__main__":
    main()

"""Freeze and evaluate the centre-interaction route on official validation.

This is the first stage at which official validation is used.  Hyperparameter
search is restricted to the official-train scaffold folds; after the search,
the selected XGBoost parameters are fit on all official-train graphs and
reported once on official-valid.  Official-test is neither encoded nor
evaluated by this entry point.

The representation is deliberately frozen from the mechanism screen:

* invariant all-centre radius-2 rooted-WL patch;
* ``marginal`` topology/attribute distributions;
* ``cross_cov`` centre-level topology--attribute covariance;
* ``binding`` patch role--attribute binding distribution;
* ``S+both`` as the primary joint view.

No gate, bilinear block, GINE, attention, pretraining, or K-SVD update is
searched here.  The only search is the declared XGBoost hyperparameter space.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
import yaml
from sklearn.metrics import roc_auc_score

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.cross_center_interaction_screen import (
    _build_cache,
    _fit_projection_model,
    _frozen_s_rows,
    _resolve,
    _sha256,
    _transform_projection,
)
from tracks.ksvd.experiments.luyin16.task_aligned_interaction_screen import (
    XGB_SEARCH_KEYS,
    _fit_xgb_predict,
    _suggest_xgb_params,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    REPO_ROOT / "tracks/ksvd/configs/luyin16/cross_center_interaction_terminal.yaml"
)
VIEW_NAMES = ("s", "s_marginal", "s_cross_cov", "s_binding", "s_both")


def _jsonable(value: Any) -> Any:
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (tuple, list)):
        return [_jsonable(item) for item in value]
    return value


def _write_json_atomic(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _strip_search_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {key: params[key] for key in XGB_SEARCH_KEYS if key in params}


def _in_range(params: Mapping[str, Any], ranges: Mapping[str, Sequence[float]]) -> bool:
    return all(
        key in params
        and float(ranges[key][0]) <= float(params[key]) <= float(ranges[key][1])
        for key in XGB_SEARCH_KEYS
    )


def _train_valid_indices(
    fold_archive: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray]:
    original = np.asarray(fold_archive["original_indices"], dtype=np.int64)
    train = original[np.asarray(fold_archive["official_train_indices"], dtype=np.int64)]
    valid = original[np.asarray(fold_archive["official_valid_indices"], dtype=np.int64)]
    if np.intersect1d(train, valid).size:
        raise RuntimeError("official train and valid indices overlap")
    return train, valid


def _official_train_scaffold_splits(
    fold_archive: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    """Map the saved scaffold folds to local rows of ``train_indices``."""
    original = np.asarray(fold_archive["original_indices"], dtype=np.int64)
    train_values = np.asarray(train_indices, dtype=np.int64)
    row_by_index = {int(index): position for position, index in enumerate(train_values)}
    fold_ids = sorted(
        int(name.removeprefix("fold_").removesuffix("_train_indices"))
        for name in fold_archive
        if name.startswith("fold_") and name.endswith("_train_indices")
    )
    splits: list[tuple[np.ndarray, np.ndarray]] = []
    metadata: list[dict[str, Any]] = []
    for fold in fold_ids:
        raw_train = original[
            np.asarray(fold_archive[f"fold_{fold}_train_indices"], dtype=np.int64)
        ]
        raw_valid = original[
            np.asarray(fold_archive[f"fold_{fold}_valid_indices"], dtype=np.int64)
        ]
        try:
            local_train = np.asarray([row_by_index[int(index)] for index in raw_train], dtype=np.int64)
            local_valid = np.asarray([row_by_index[int(index)] for index in raw_valid], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(f"scaffold fold index is absent from official train: {exc}") from exc
        if np.intersect1d(local_train, local_valid).size:
            raise RuntimeError(f"scaffold fold {fold} overlaps")
        if np.unique(np.concatenate([local_train, local_valid])).size != train_values.size:
            raise RuntimeError(f"scaffold fold {fold} does not cover official train exactly once")
        splits.append((local_train, local_valid))
        metadata.append(
            {
                "fold": int(fold),
                "n_train": int(local_train.size),
                "n_valid": int(local_valid.size),
            }
        )
    return splits, metadata


def _build_views(
    cache: Mapping[str, np.ndarray],
    frozen: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
    valid_indices: np.ndarray,
    pca_rank: int,
) -> tuple[dict[str, np.ndarray], dict[str, Any], np.ndarray, np.ndarray]:
    dev_indices = np.concatenate([train_indices, valid_indices]).astype(np.int64, copy=False)
    cache_indices = np.asarray(cache["dataset_indices"], dtype=np.int64)
    row_by_index = {int(index): position for position, index in enumerate(cache_indices)}
    try:
        dev_rows = np.asarray([row_by_index[int(index)] for index in dev_indices], dtype=np.int64)
        train_rows = np.asarray([row_by_index[int(index)] for index in train_indices], dtype=np.int64)
        valid_rows = np.asarray([row_by_index[int(index)] for index in valid_indices], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"interaction cache lacks official dev index: {exc}") from exc

    s = _frozen_s_rows(frozen, dev_indices)
    marginal = np.asarray(cache["marginal"][dev_rows], dtype=np.float32)
    context = np.asarray(cache["context"][dev_rows], dtype=np.float32)
    base = np.concatenate([s, marginal, context], axis=1).astype(np.float32, copy=False)

    cross_model, cross_train, cross_meta = _fit_projection_model(
        np.asarray(cache["cross_cov"][train_rows], dtype=np.float32), pca_rank
    )
    binding_model, binding_train, binding_meta = _fit_projection_model(
        np.asarray(cache["binding"][train_rows], dtype=np.float32), pca_rank
    )
    cross_valid = _transform_projection(
        cross_model, np.asarray(cache["cross_cov"][valid_rows], dtype=np.float32)
    )
    binding_valid = _transform_projection(
        binding_model, np.asarray(cache["binding"][valid_rows], dtype=np.float32)
    )
    # Reuse fit_transform for train and transform only the held-out rows.  In
    # float32, calling transform again on train can differ by ~1e-4 because of
    # BLAS accumulation order even though the coordinate system is identical.
    cross = np.concatenate([cross_train, cross_valid], axis=0)
    binding = np.concatenate([binding_train, binding_valid], axis=0)
    views = {
        "s": s,
        "s_marginal": base,
        "s_cross_cov": np.concatenate([base, cross], axis=1),
        "s_binding": np.concatenate([base, binding], axis=1),
        "s_both": np.concatenate([base, cross, binding], axis=1),
    }
    metadata = {
        "pca_rank": int(pca_rank),
        "cross_cov": cross_meta,
        "binding": binding_meta,
        "fit_scope": "official-train only",
        "train_row_count": int(train_rows.size),
        "valid_row_count": int(valid_rows.size),
        "row_order": "official-train then official-valid",
    }
    return views, metadata, train_rows, valid_rows


def _build_cv_fold_blocks(
    cache: Mapping[str, np.ndarray],
    frozen: Mapping[str, np.ndarray],
    train_indices: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    pca_rank: int,
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]], int]:
    """Build fold-specific train/holdout coordinates for honest tuning.

    PCA is refit independently inside every official-train scaffold fold.
    This avoids allowing a fold's label-free holdout distribution to affect
    the projection used by its XGBoost score.
    """
    indices = np.asarray(train_indices, dtype=np.int64)
    cache_indices = np.asarray(cache["dataset_indices"], dtype=np.int64)
    row_by_index = {int(index): position for position, index in enumerate(cache_indices)}
    try:
        cache_rows = np.asarray([row_by_index[int(index)] for index in indices], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"interaction cache lacks official-train index: {exc}") from exc
    s = _frozen_s_rows(frozen, indices)
    s_width = int(s.shape[1])
    marginal = np.asarray(cache["marginal"][cache_rows], dtype=np.float32)
    context = np.asarray(cache["context"][cache_rows], dtype=np.float32)
    base = np.concatenate([s, marginal, context], axis=1).astype(np.float32, copy=False)

    blocks: list[dict[str, np.ndarray]] = []
    metadata: list[dict[str, Any]] = []
    for fold_id, (fold_train, fold_valid) in enumerate(splits):
        train_rows = cache_rows[np.asarray(fold_train, dtype=np.int64)]
        valid_rows = cache_rows[np.asarray(fold_valid, dtype=np.int64)]
        cross_model, cross_train, cross_meta = _fit_projection_model(
            np.asarray(cache["cross_cov"][train_rows], dtype=np.float32), pca_rank
        )
        binding_model, binding_train, binding_meta = _fit_projection_model(
            np.asarray(cache["binding"][train_rows], dtype=np.float32), pca_rank
        )
        cross_valid = _transform_projection(
            cross_model, np.asarray(cache["cross_cov"][valid_rows], dtype=np.float32)
        )
        binding_valid = _transform_projection(
            binding_model, np.asarray(cache["binding"][valid_rows], dtype=np.float32)
        )
        blocks.append(
            {
                "base_train": np.asarray(base[fold_train], dtype=np.float32),
                "base_valid": np.asarray(base[fold_valid], dtype=np.float32),
                "cross_train": cross_train,
                "cross_valid": cross_valid,
                "binding_train": binding_train,
                "binding_valid": binding_valid,
            }
        )
        metadata.append(
            {
                "fold": int(fold_id),
                "n_train": int(len(fold_train)),
                "n_valid": int(len(fold_valid)),
                "cross_cov": cross_meta,
                "binding": binding_meta,
            }
        )
    return blocks, metadata, s_width


def _fold_view_matrices(
    blocks: Sequence[Mapping[str, np.ndarray]], view: str, s_width: int
) -> list[tuple[np.ndarray, np.ndarray]]:
    matrices: list[tuple[np.ndarray, np.ndarray]] = []
    for block in blocks:
        base_train = np.asarray(block["base_train"], dtype=np.float32)
        base_valid = np.asarray(block["base_valid"], dtype=np.float32)
        if view == "s":
            train = base_train[:, : int(s_width)]
            valid = base_valid[:, : int(s_width)]
        elif view == "s_marginal":
            train, valid = base_train, base_valid
        elif view == "s_cross_cov":
            train = np.concatenate([base_train, block["cross_train"]], axis=1)
            valid = np.concatenate([base_valid, block["cross_valid"]], axis=1)
        elif view == "s_binding":
            train = np.concatenate([base_train, block["binding_train"]], axis=1)
            valid = np.concatenate([base_valid, block["binding_valid"]], axis=1)
        elif view == "s_both":
            train = np.concatenate(
                [base_train, block["cross_train"], block["binding_train"]], axis=1
            )
            valid = np.concatenate(
                [base_valid, block["cross_valid"], block["binding_valid"]], axis=1
            )
        else:
            raise ValueError(f"unknown terminal view: {view}")
        matrices.append(
            (
                np.asarray(train, dtype=np.float32),
                np.asarray(valid, dtype=np.float32),
            )
        )
    return matrices


def _fixed_params(
    classifier: Mapping[str, Any], tuning: Mapping[str, Any], view: str
) -> dict[str, Any]:
    params = _strip_search_params(classifier)
    params.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": 0,
            "n_jobs": int(tuning.get("n_jobs", -1)),
        }
    )
    params.update(tuning.get("view_static", {}).get(view, {}))
    return params


def _tune_view(
    fold_matrices: Sequence[tuple[np.ndarray, np.ndarray]],
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    tuning: Mapping[str, Any],
    classifier: Mapping[str, Any],
    view: str,
    seed: int,
) -> dict[str, Any]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    sampler = optuna.samplers.TPESampler(seed=int(seed))
    study = optuna.create_study(direction="maximize", sampler=sampler)
    ranges = tuning["ranges"]
    warm: list[dict[str, Any]] = []
    fixed = _strip_search_params(classifier)
    if _in_range(fixed, ranges):
        warm.append(fixed)
    for candidate in tuning.get("warm_start_params", []):
        candidate_params = _strip_search_params(candidate)
        if _in_range(candidate_params, ranges):
            warm.append(candidate_params)
    unique_warm: list[dict[str, Any]] = []
    seen: set[str] = set()
    for candidate in warm:
        key = json.dumps(candidate, sort_keys=True)
        if key not in seen:
            unique_warm.append(candidate)
            seen.add(key)
            study.enqueue_trial(candidate)

    def score(params: Mapping[str, Any]) -> list[float]:
        if len(fold_matrices) != len(splits):
            raise ValueError("fold-specific matrices and scaffold splits differ")
        scores: list[float] = []
        y = np.asarray(labels, dtype=np.int64)
        for (x_train, x_valid), (train, valid) in zip(
            fold_matrices, splits, strict=True
        ):
            if x_train.shape[0] != len(train) or x_valid.shape[0] != len(valid):
                raise ValueError("fold-specific matrix rows do not match split rows")
            prediction = _fit_xgb_predict(
                x_train,
                y[np.asarray(train, dtype=np.int64)],
                x_valid,
                params,
            )
            scores.append(
                float(
                    roc_auc_score(
                        y[np.asarray(valid, dtype=np.int64)], prediction
                    )
                )
            )
        return scores

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgb_params(trial, tuning, seed=seed)
        params.update(tuning.get("view_static", {}).get(view, {}))
        try:
            return float(np.mean(score(params)))
        finally:
            gc.collect()

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
            "tree_method": "hist",
            "random_state": 0,
            "n_jobs": int(tuning.get("n_jobs", -1)),
        }
    )
    best_params.update(tuning.get("view_static", {}).get(view, {}))
    fold_auc = score(best_params)
    return {
        "backend": "optuna_tpe",
        "n_trials": int(tuning["n_trials"]),
        "n_warm_starts": int(len(unique_warm)),
        "best_trial": int(study.best_trial.number),
        "best_scaffold_auc": float(np.mean(fold_auc)),
        "best_scaffold_fold_auc": fold_auc,
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


def _evaluate_valid(
    matrix: np.ndarray,
    labels: np.ndarray,
    train_rows: np.ndarray,
    valid_rows: np.ndarray,
    params: Mapping[str, Any],
    seeds: Sequence[int],
) -> dict[str, Any]:
    valid_labels = np.asarray(labels[valid_rows], dtype=np.int64)
    records: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    for seed in seeds:
        current = dict(params)
        current["random_state"] = int(seed)
        prediction = _fit_xgb_predict(
            matrix[train_rows],
            np.asarray(labels[train_rows], dtype=np.int64),
            matrix[valid_rows],
            current,
        )
        predictions.append(prediction)
        records.append(
            {
                "seed": int(seed),
                "valid_auc": float(roc_auc_score(valid_labels, prediction)),
            }
        )
    ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
    values = [row["valid_auc"] for row in records]
    return {
        "rows": records,
        "mean_auc": float(np.mean(values)),
        "std_auc": float(np.std(values)),
        "seed_ensemble_auc": float(roc_auc_score(valid_labels, ensemble)),
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Centre-level structure--attribute frozen official-valid evaluation",
        "",
        f"Protocol: `{result['protocol_id']}`",
        "",
        "Hyperparameters were searched only on official-train scaffold folds. Official-test was not encoded or evaluated.",
        "",
        f"Official train/valid: {result['data']['n_train']}/{result['data']['n_valid']} graphs.",
        "",
        "| view | scaffold CV after tuning | fixed valid mean | tuned valid mean | tuned valid ensemble |",
        "|---|---:|---:|---:|---:|",
    ]
    for view in result["views"]:
        row = result["views"][view]
        lines.append(
            f"| `{view}` | {row['search']['best_scaffold_auc']:.6f} | "
            f"{row['fixed_official_valid']['mean_auc']:.6f} | "
            f"{row['tuned_official_valid']['mean_auc']:.6f} | "
            f"{row['tuned_official_valid']['seed_ensemble_auc']:.6f} |"
        )
    selected = result["selection"]["selected_view"]
    lines.extend(
        [
            "",
            f"Selected view by official-train scaffold CV: `{selected}`.",
            "",
            "## Frozen parameters",
            "",
            "```json",
            json.dumps(result["views"][selected]["search"]["best_params"], ensure_ascii=False, indent=2),
            "```",
            "",
            "No test score is reported by this protocol.",
        ]
    )
    return "\n".join(lines) + "\n"


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    representation = config["representation"]
    tuning = config["tuning"]
    classifier = config["classifier"]
    folds_path = _resolve(data["scaffold_folds"])
    frozen_path = _resolve(data["frozen_features"])
    cache_path = _resolve(data["dev_interaction_cache"])
    checkpoint_path = _resolve(config["search_checkpoint_json"])
    output_json = _resolve(config["output_json"])
    output_markdown = _resolve(config["output_markdown"])
    with np.load(folds_path, allow_pickle=False) as archive:
        fold_archive = {name: np.asarray(archive[name]) for name in archive.files}
    train_indices, valid_indices = _train_valid_indices(fold_archive)
    dev_indices = np.concatenate([train_indices, valid_indices]).astype(np.int64, copy=False)
    bundle = load_molhiv(root=_resolve(data["root"]), with_features=True)
    labels = np.asarray(bundle.y, dtype=np.int64)
    if not np.array_equal(np.sort(bundle.split["train"]), np.sort(train_indices)):
        raise RuntimeError("scaffold archive official train differs from OGB split")
    if not np.array_equal(np.sort(bundle.split["valid"]), np.sort(valid_indices)):
        raise RuntimeError("scaffold archive official valid differs from OGB split")
    with np.load(frozen_path, allow_pickle=False) as archive:
        frozen = {name: np.asarray(archive[name]) for name in archive.files}
    start = time.perf_counter()
    cache = _build_cache(
        bundle,
        dev_indices,
        representation,
        shuffle_repeats=0,
        seed=int(config["feature_seed"]),
        cache_path=cache_path,
    )
    views, valid_projection_meta, _, _ = _build_views(
        cache,
        frozen,
        train_indices,
        valid_indices,
        int(config["pca_rank"]),
    )
    train_labels = labels[train_indices]
    train_dev_rows = np.arange(train_indices.size, dtype=np.int64)
    valid_dev_rows = np.arange(train_indices.size, dev_indices.size, dtype=np.int64)
    splits, split_meta = _official_train_scaffold_splits(fold_archive, train_indices)
    cv_blocks, cv_projection_meta, s_width = _build_cv_fold_blocks(
        cache,
        frozen,
        train_indices,
        splits,
        int(config["pca_rank"]),
    )
    # The local scaffold splits are indexed in train-row space, whereas the
    # matrices above are ordered train-then-valid.  Keep this conversion
    # explicit so a cache-order change cannot leak valid rows into tuning.
    local_splits: list[tuple[np.ndarray, np.ndarray]] = []
    for fold_train, fold_valid in splits:
        local_splits.append((fold_train, fold_valid))
    if any(np.max(train_part, initial=-1) >= train_indices.size for train_part, _ in local_splits):
        raise RuntimeError("invalid train scaffold split row")
    if any(np.max(valid_part, initial=-1) >= train_indices.size for _, valid_part in local_splits):
        raise RuntimeError("invalid valid scaffold split row")
    model_seeds = [int(value) for value in config.get("model_seeds", [0, 1, 2, 3, 4])]
    view_results: dict[str, Any] = {}
    checkpoint_identity = {
        "config_sha256": _sha256(config_path),
        "implementation_sha256": _sha256(Path(__file__)),
        "interaction_cache_sha256": _sha256(cache_path),
    }
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        for key, expected in checkpoint_identity.items():
            if checkpoint.get(key) != expected:
                raise RuntimeError(
                    f"terminal search checkpoint {key} mismatch: {checkpoint_path}"
                )
        view_results.update(checkpoint.get("views", {}))
    for view_id, view in enumerate(VIEW_NAMES):
        if view in view_results:
            print(f"{view}: resuming completed search/evaluation", flush=True)
            continue
        matrix = np.asarray(views[view], dtype=np.float32)
        fold_matrices = _fold_view_matrices(cv_blocks, view, s_width)
        search = _tune_view(
            fold_matrices,
            train_labels,
            local_splits,
            tuning,
            classifier,
            view,
            int(tuning["seed"]) + 1009 * view_id,
        )
        del fold_matrices
        gc.collect()
        fixed_valid = _evaluate_valid(
            matrix,
            labels[dev_indices],
            train_dev_rows,
            valid_dev_rows,
            _fixed_params(classifier, tuning, view),
            model_seeds,
        )
        tuned_valid = _evaluate_valid(
            matrix,
            labels[dev_indices],
            train_dev_rows,
            valid_dev_rows,
            search["best_params"],
            model_seeds,
        )
        view_results[view] = {
            "dimension": int(matrix.shape[1]),
            "search": search,
            "fixed_official_valid": fixed_valid,
            "tuned_official_valid": tuned_valid,
        }
        print(
            f"{view}: scaffold={search['best_scaffold_auc']:.6f}; "
            f"fixed-valid={fixed_valid['mean_auc']:.6f}; "
            f"tuned-valid={tuned_valid['mean_auc']:.6f}; "
            f"ensemble={tuned_valid['seed_ensemble_auc']:.6f}",
            flush=True,
        )
        _write_json_atomic(
            checkpoint_path,
            checkpoint_identity
            | {
                "protocol_id": config["protocol_id"],
                "views": view_results,
            },
        )
    selected_view = max(
        VIEW_NAMES,
        key=lambda name: (
            float(view_results[name]["search"]["best_scaffold_auc"]),
            int(name == "s_both"),
        ),
    )
    result = {
        "protocol_id": config["protocol_id"],
        "stage": "frozen_official_valid_after_train_only_search",
        "official_validation_or_test_evaluated": True,
        "official_test_evaluated": False,
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "implementation_sha256": _sha256(Path(__file__)),
        "interaction_encoder_sha256": _sha256(
            Path(__file__).with_name("cross_center_interaction_screen.py")
        ),
        "audit_boundary": {
            "hyperparameters_selected_on": "official-train scaffold folds only",
            "official_validation_used_once_for_frozen_reporting": True,
            "official_test_encoded": False,
            "official_test_evaluated": False,
        },
        "data": {
            "dataset": data["dataset"],
            "scaffold_folds": str(folds_path),
            "scaffold_folds_sha256": _sha256(folds_path),
            "frozen_features": str(frozen_path),
            "dev_interaction_cache": str(cache_path),
            "n_train": int(train_indices.size),
            "n_valid": int(valid_indices.size),
            "train_positive": int(labels[train_indices].sum()),
            "valid_positive": int(labels[valid_indices].sum()),
            "scaffold_folds_metadata": split_meta,
        },
        "representation": dict(representation),
        "pca_rank": int(config["pca_rank"]),
        "projection_meta": {
            "official_train_scaffold_cv": cv_projection_meta,
            "official_train_full_for_valid": valid_projection_meta,
        },
        "model_seeds": model_seeds,
        "views": view_results,
        "selection": {
            "rule": "highest mean official-train scaffold CV; s_both wins ties",
            "candidate_views": list(VIEW_NAMES),
            "selected_view": selected_view,
        },
        "runtime": {
            "seconds": float(time.perf_counter() - start),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    _write_json_atomic(output_json, result)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = run(_resolve(args.config))
    print(
        json.dumps(
            {
                "protocol_id": result["protocol_id"],
                "selected_view": result["selection"]["selected_view"],
                "official_valid": {
                    name: result["views"][name]["tuned_official_valid"]
                    for name in VIEW_NAMES
                },
                "official_test_evaluated": result["official_test_evaluated"],
                "runtime_seconds": result["runtime"]["seconds"],
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

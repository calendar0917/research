"""Persistent Optuna search and frozen test evaluation for mentor artifact features."""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold
from xgboost import XGBClassifier

from tracks.ksvd.experiments.luyin16.mentor_artifact_typed_ksvd import (
    VIEW_BLOCKS,
    _jsonable,
    _write_json,
    assemble_view,
)


InnerSplit = tuple[np.ndarray, np.ndarray]

DEFAULT_ENQUEUED_PARAMS = {
    "learning_rate": 0.028881557202595887,
    "max_depth": 7,
    "min_child_weight": 11.0,
    "subsample": 0.7669478927651876,
    "colsample_bytree": 0.7852959683368484,
    "reg_alpha": 0.09251054395624937,
    "reg_lambda": 29.167447697425548,
    "gamma": 0.0,
    "max_bin": 256,
    "scale_pos_weight_multiplier": 1.0,
}


def _load_payload(path: Path) -> dict[str, np.ndarray]:
    archive = np.load(path, allow_pickle=False)
    required = {
        "dataset_indices",
        "labels",
        "train_indices",
        "valid_indices",
        "test_indices",
        "composition",
        "typed_raw",
        "typed_init",
        "typed_final",
        "context_init",
        "context_final",
        "cross_cov_init",
        "cross_cov_final",
    }
    missing = required.difference(archive.files)
    if missing:
        raise KeyError(f"feature payload is missing {sorted(missing)}")
    return {name: np.asarray(archive[name]) for name in archive.files}


def scaffold_inner_splits(
    folds_file: Path,
    train_dataset_indices: np.ndarray,
    train_labels: np.ndarray,
) -> tuple[list[InnerSplit], dict[str, Any]]:
    archive = np.load(folds_file, allow_pickle=False)
    original_indices = np.asarray(archive["original_indices"], dtype=np.int64)
    dataset_to_row = {
        int(dataset_index): row
        for row, dataset_index in enumerate(np.asarray(train_dataset_indices, dtype=np.int64))
    }
    fold_ids = sorted(
        int(name.removeprefix("fold_").removesuffix("_train_indices"))
        for name in archive.files
        if name.startswith("fold_") and name.endswith("_train_indices")
    )
    splits: list[InnerSplit] = []
    rows: list[dict[str, int]] = []
    for fold in fold_ids:
        train_original = original_indices[np.asarray(archive[f"fold_{fold}_train_indices"], dtype=np.int64)]
        valid_original = original_indices[np.asarray(archive[f"fold_{fold}_valid_indices"], dtype=np.int64)]
        try:
            train = np.asarray([dataset_to_row[int(index)] for index in train_original], dtype=np.int64)
            valid = np.asarray([dataset_to_row[int(index)] for index in valid_original], dtype=np.int64)
        except KeyError as exc:
            raise ValueError(
                f"scaffold fold dataset index {int(exc.args[0])} is absent from features"
            ) from exc
        if np.unique(train_labels[train]).size != 2 or np.unique(train_labels[valid]).size != 2:
            raise ValueError(f"fold {fold} does not contain both classes")
        splits.append((train, valid))
        rows.append(
            {
                "fold": int(fold),
                "n_train": int(train.size),
                "n_valid": int(valid.size),
                "train_positive": int(train_labels[train].sum()),
                "valid_positive": int(train_labels[valid].sum()),
            }
        )
    if len(splits) < 2:
        raise ValueError("at least two scaffold folds are required")
    return splits, {
        "kind": "official-train-only Bemis-Murcko scaffold folds",
        "path": str(folds_file.resolve()),
        "folds": rows,
    }


def stratified_inner_splits(
    labels: np.ndarray,
    folds: int,
    seed: int,
) -> tuple[list[InnerSplit], dict[str, Any]]:
    splitter = StratifiedKFold(n_splits=int(folds), shuffle=True, random_state=int(seed))
    splits = [
        (np.asarray(train, dtype=np.int64), np.asarray(valid, dtype=np.int64))
        for train, valid in splitter.split(np.zeros(labels.size), labels)
    ]
    return splits, {
        "kind": f"{folds}-fold stratified CV over official-train rows",
        "seed": int(seed),
    }


def _trial_params(trial: optuna.Trial) -> dict[str, Any]:
    return {
        "learning_rate": trial.suggest_float("learning_rate", 0.008, 0.15, log=True),
        "max_depth": trial.suggest_int("max_depth", 3, 8),
        "min_child_weight": trial.suggest_float("min_child_weight", 1.0, 40.0, log=True),
        "subsample": trial.suggest_float("subsample", 0.65, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.55, 1.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 30.0, log=True),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 100.0, log=True),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "max_bin": trial.suggest_categorical("max_bin", [128, 256]),
        "scale_pos_weight_multiplier": trial.suggest_float(
            "scale_pos_weight_multiplier", 0.25, 1.75, log=True
        ),
    }


def _model_params(params: Mapping[str, Any]) -> tuple[dict[str, Any], float]:
    model_params = dict(params)
    multiplier = float(model_params.pop("scale_pos_weight_multiplier"))
    return model_params, multiplier


def _fold_scores(
    matrix: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[InnerSplit],
    params: Mapping[str, Any],
    *,
    max_estimators: int,
    early_stopping_rounds: int,
    seed: int,
    n_jobs: int,
) -> tuple[list[float], list[int]]:
    model_params, multiplier = _model_params(params)
    scores: list[float] = []
    iterations: list[int] = []
    for train, valid in splits:
        y_train = labels[train]
        class_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
        model = XGBClassifier(
            **model_params,
            n_estimators=int(max_estimators),
            objective="binary:logistic",
            eval_metric="auc",
            scale_pos_weight=class_weight * multiplier,
            tree_method="hist",
            early_stopping_rounds=int(early_stopping_rounds),
            random_state=int(seed),
            n_jobs=int(n_jobs),
        )
        model.fit(
            matrix[train],
            y_train,
            eval_set=[(matrix[valid], labels[valid])],
            verbose=False,
        )
        prediction = model.predict_proba(matrix[valid])[:, 1]
        scores.append(float(roc_auc_score(labels[valid], prediction)))
        best_iteration = getattr(model, "best_iteration", None)
        iterations.append(int(best_iteration) + 1 if best_iteration is not None else int(max_estimators))
    return scores, iterations


def _study_trials(study: optuna.Study) -> list[dict[str, Any]]:
    return [
        {
            "number": int(trial.number),
            "state": str(trial.state),
            "value": None if trial.value is None else float(trial.value),
            "params": dict(trial.params),
            "user_attrs": dict(trial.user_attrs),
            "duration_seconds": (trial.duration.total_seconds() if trial.duration is not None else None),
        }
        for trial in study.trials
    ]


def tune_view(
    view: str,
    matrix: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[InnerSplit],
    *,
    storage_path: Path,
    study_prefix: str,
    target_trials: int,
    max_estimators: int,
    early_stopping_rounds: int,
    search_seed: int,
    n_jobs: int,
) -> dict[str, Any]:
    storage_path.parent.mkdir(parents=True, exist_ok=True)
    study_name = f"{study_prefix}__{view}"
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(search_seed)),
        study_name=study_name,
        storage=f"sqlite:///{storage_path.resolve()}",
        load_if_exists=True,
    )
    if not study.trials:
        study.enqueue_trial(DEFAULT_ENQUEUED_PARAMS)

    def objective(trial: optuna.Trial) -> float:
        params = _trial_params(trial)
        scores, iterations = _fold_scores(
            matrix,
            labels,
            splits,
            params,
            max_estimators=max_estimators,
            early_stopping_rounds=early_stopping_rounds,
            seed=search_seed,
            n_jobs=n_jobs,
        )
        trial.set_user_attr("fold_auc", [float(value) for value in scores])
        trial.set_user_attr("best_iterations", [int(value) for value in iterations])
        trial.set_user_attr("mean_best_iteration", float(np.mean(iterations)))
        return float(np.mean(scores))

    complete_or_pruned = sum(
        trial.state in (optuna.trial.TrialState.COMPLETE, optuna.trial.TrialState.PRUNED)
        for trial in study.trials
    )
    remaining = max(0, int(target_trials) - int(complete_or_pruned))
    if remaining:
        study.optimize(objective, n_trials=remaining, show_progress_bar=False)
    best = study.best_trial
    fold_auc = [float(value) for value in best.user_attrs["fold_auc"]]
    best_iterations = [int(value) for value in best.user_attrs["best_iterations"]]
    n_estimators = max(1, int(round(float(np.median(best_iterations)))))
    return {
        "study_name": study_name,
        "storage": str(storage_path.resolve()),
        "target_trials": int(target_trials),
        "completed_trials": int(
            sum(trial.state == optuna.trial.TrialState.COMPLETE for trial in study.trials)
        ),
        "best_trial": int(best.number),
        "best_inner_auc": float(best.value),
        "best_inner_fold_auc": fold_auc,
        "best_fold_iterations": best_iterations,
        "refit_n_estimators": n_estimators,
        "best_params": dict(best.params),
        "trials": _study_trials(study),
    }


def _fit_fixed(
    matrix: np.ndarray,
    labels: np.ndarray,
    fit_rows: np.ndarray,
    score_rows: np.ndarray,
    params: Mapping[str, Any],
    n_estimators: int,
    seed: int,
    n_jobs: int,
) -> tuple[dict[str, Any], np.ndarray]:
    model_params, multiplier = _model_params(params)
    y_fit = labels[fit_rows]
    class_weight = float(np.sum(y_fit == 0) / max(1, np.sum(y_fit == 1)))
    model = XGBClassifier(
        **model_params,
        n_estimators=int(n_estimators),
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=class_weight * multiplier,
        tree_method="hist",
        random_state=int(seed),
        n_jobs=int(n_jobs),
    )
    model.fit(matrix[fit_rows], y_fit)
    prediction = model.predict_proba(matrix[score_rows])[:, 1]
    return {
        "seed": int(seed),
        "fit_rows": int(fit_rows.size),
        "score_rows": int(score_rows.size),
        "score_positive": int(labels[score_rows].sum()),
        "roc_auc": float(roc_auc_score(labels[score_rows], prediction)),
    }, prediction.astype(np.float32)


def _summary(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "mean": float(array.mean()),
        "std_sample": float(array.std(ddof=1)) if array.size > 1 else 0.0,
        "minimum": float(array.min()),
        "maximum": float(array.max()),
    }


def run_tune(args: argparse.Namespace) -> dict[str, Any]:
    started = time.time()
    feature_path = args.features.expanduser().resolve()
    result_path = args.result.expanduser().resolve()
    prediction_path = result_path.with_suffix(".predictions.npz")
    payload = _load_payload(feature_path)
    labels = np.asarray(payload["labels"], dtype=np.int64)
    train = np.asarray(payload["train_indices"], dtype=np.int64)
    valid = np.asarray(payload["valid_indices"], dtype=np.int64)
    dataset_indices = np.asarray(payload["dataset_indices"], dtype=np.int64)
    train_dataset_indices = dataset_indices[train]
    train_labels = labels[train]
    if args.folds_file is None:
        splits, split_meta = stratified_inner_splits(train_labels, args.folds, args.search_seed)
    else:
        splits, split_meta = scaffold_inner_splits(
            args.folds_file.expanduser().resolve(), train_dataset_indices, train_labels
        )
    views = [name.strip() for name in args.views.split(",") if name.strip()]
    unknown = sorted(set(views) - set(VIEW_BLOCKS))
    if unknown:
        raise ValueError(f"unknown views {unknown}")
    if result_path.exists():
        result = json.loads(result_path.read_text(encoding="utf-8"))
        if result.get("protocol_id") != str(args.protocol_id):
            raise ValueError("existing result protocol_id does not match --protocol-id")
        if Path(result.get("feature_file", "")).resolve() != feature_path:
            raise ValueError("existing result feature_file does not match --features")
        result["inner_split"] = split_meta
        result["search"] = {
            "sampler": "Optuna TPESampler",
            "seed": int(args.search_seed),
            "latest_target_trials_per_view": int(args.trials),
            "max_estimators": int(args.max_estimators),
            "early_stopping_rounds": int(args.early_stopping_rounds),
            "model_seeds": [int(seed) for seed in args.seeds],
        }
        previous_runtime = float(result.get("runtime_seconds", 0.0))
    else:
        result = {
            "protocol_id": str(args.protocol_id),
            "stage": "official-train-only Optuna then frozen official-valid evaluation",
            "feature_file": str(feature_path),
            "official_test_evaluated": False,
            "inner_split": split_meta,
            "search": {
                "sampler": "Optuna TPESampler",
                "seed": int(args.search_seed),
                "latest_target_trials_per_view": int(args.trials),
                "max_estimators": int(args.max_estimators),
                "early_stopping_rounds": int(args.early_stopping_rounds),
                "model_seeds": [int(seed) for seed in args.seeds],
            },
            "views": {},
        }
        previous_runtime = 0.0
    if prediction_path.exists():
        previous_predictions = np.load(prediction_path, allow_pickle=False)
        prediction_payload = {
            name: np.asarray(previous_predictions[name]) for name in previous_predictions.files
        }
    else:
        prediction_payload = {
            "valid_labels": labels[valid].astype(np.int64),
            "valid_indices": valid,
        }
    for view in views:
        print(f"\nOptuna view={view}", flush=True)
        matrix = assemble_view(payload, view)
        search = tune_view(
            view,
            matrix[train],
            train_labels,
            splits,
            storage_path=args.storage.expanduser().resolve(),
            study_prefix=str(args.study_prefix),
            target_trials=int(args.trials),
            max_estimators=int(args.max_estimators),
            early_stopping_rounds=int(args.early_stopping_rounds),
            search_seed=int(args.search_seed),
            n_jobs=int(args.n_jobs),
        )
        rows: list[dict[str, Any]] = []
        predictions: list[np.ndarray] = []
        for seed in args.seeds:
            row, prediction = _fit_fixed(
                matrix,
                labels,
                train,
                valid,
                search["best_params"],
                search["refit_n_estimators"],
                int(seed),
                int(args.n_jobs),
            )
            rows.append(row)
            predictions.append(prediction)
        stacked = np.stack(predictions, axis=0)
        aucs = [float(row["roc_auc"]) for row in rows]
        search.update(
            {
                "dimension": int(matrix.shape[1]),
                "official_valid_rows": rows,
                "official_valid": _summary(aucs),
                "official_valid_seed_ensemble_auc": float(roc_auc_score(labels[valid], stacked.mean(axis=0))),
            }
        )
        result["views"][view] = search
        prediction_payload[f"{view}_valid_predictions"] = stacked
        _write_json(result_path, result)
        print(
            f"view={view} inner={search['best_inner_auc']:.6f} "
            f"valid={search['official_valid']['mean']:.6f} "
            f"ensemble={search['official_valid_seed_ensemble_auc']:.6f}",
            flush=True,
        )
    best_view = max(
        result["views"],
        key=lambda name: result["views"][name]["official_valid"]["mean"],
    )
    result["selection"] = {
        "best_view_by_official_valid_mean": best_view,
        "best_official_valid_mean": result["views"][best_view]["official_valid"]["mean"],
        "test_used_for_selection": False,
    }
    result["runtime_seconds"] = previous_runtime + float(time.time() - started)
    _write_json(result_path, result)
    np.savez_compressed(prediction_path, **prediction_payload)
    return result


def _evaluate_scope(
    matrix: np.ndarray,
    labels: np.ndarray,
    fit_rows: np.ndarray,
    test_rows: np.ndarray,
    view_result: Mapping[str, Any],
    seeds: Sequence[int],
    n_jobs: int,
) -> tuple[dict[str, Any], np.ndarray]:
    rows: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    for seed in seeds:
        row, prediction = _fit_fixed(
            matrix,
            labels,
            fit_rows,
            test_rows,
            view_result["best_params"],
            int(view_result["refit_n_estimators"]),
            int(seed),
            int(n_jobs),
        )
        rows.append(row)
        predictions.append(prediction)
    stacked = np.stack(predictions, axis=0)
    aucs = [float(row["roc_auc"]) for row in rows]
    return {
        "rows": rows,
        "summary": _summary(aucs),
        "seed_ensemble_auc": float(roc_auc_score(labels[test_rows], stacked.mean(axis=0))),
    }, stacked


def run_test(args: argparse.Namespace) -> dict[str, Any]:
    payload = _load_payload(args.features.expanduser().resolve())
    tuning = json.loads(args.tuning.read_text(encoding="utf-8"))
    labels = np.asarray(payload["labels"], dtype=np.int64)
    train = np.asarray(payload["train_indices"], dtype=np.int64)
    valid = np.asarray(payload["valid_indices"], dtype=np.int64)
    test = np.asarray(payload["test_indices"], dtype=np.int64)
    requested = [name.strip() for name in args.views.split(",") if name.strip()]
    if requested == ["best"]:
        requested = [tuning["selection"]["best_view_by_official_valid_mean"]]
    unknown = [name for name in requested if name not in tuning["views"]]
    if unknown:
        raise ValueError(f"views absent from frozen tuning result: {unknown}")
    result: dict[str, Any] = {
        "protocol_id": str(args.protocol_id),
        "stage": "frozen official-test evaluation",
        "feature_file": str(args.features.resolve()),
        "tuning_result": str(args.tuning.resolve()),
        "official_test_evaluated": True,
        "selection_rule": "view and parameters frozen from train-CV/official-valid before test",
        "historical_disclosure": "MolHIV official test was viewed by older repository routes",
        "views": {},
    }
    prediction_payload: dict[str, np.ndarray] = {
        "test_labels": labels[test].astype(np.int64),
        "test_indices": test,
    }
    train_valid = np.concatenate([train, valid]).astype(np.int64)
    for view in requested:
        matrix = assemble_view(payload, view)
        frozen = tuning["views"][view]
        train_only, train_predictions = _evaluate_scope(
            matrix,
            labels,
            train,
            test,
            frozen,
            args.seeds,
            args.n_jobs,
        )
        refit, refit_predictions = _evaluate_scope(
            matrix,
            labels,
            train_valid,
            test,
            frozen,
            args.seeds,
            args.n_jobs,
        )
        result["views"][view] = {
            "dimension": int(matrix.shape[1]),
            "best_params": frozen["best_params"],
            "refit_n_estimators": int(frozen["refit_n_estimators"]),
            "frozen_inner_auc": float(frozen["best_inner_auc"]),
            "frozen_official_valid": frozen["official_valid"],
            "train_only": train_only,
            "train_valid_refit": refit,
        }
        prediction_payload[f"{view}_train_only_test_predictions"] = train_predictions
        prediction_payload[f"{view}_train_valid_test_predictions"] = refit_predictions
    result_path = args.result.expanduser().resolve()
    _write_json(result_path, result)
    np.savez_compressed(result_path.with_suffix(".predictions.npz"), **prediction_payload)
    print(
        json.dumps(
            {
                view: {
                    "train_only_mean": row["train_only"]["summary"]["mean"],
                    "train_only_ensemble": row["train_only"]["seed_ensemble_auc"],
                    "train_valid_refit_mean": row["train_valid_refit"]["summary"]["mean"],
                    "train_valid_refit_ensemble": row["train_valid_refit"]["seed_ensemble_auc"],
                }
                for view, row in result["views"].items()
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return result


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)

    tune = subparsers.add_parser("tune")
    tune.add_argument("--features", type=Path, required=True)
    tune.add_argument("--result", type=Path, required=True)
    tune.add_argument("--storage", type=Path, required=True)
    tune.add_argument("--folds-file", type=Path)
    tune.add_argument("--folds", type=int, default=3)
    tune.add_argument("--views", default="s,st_raw,st_final,sta_final,sta_cross_cov")
    tune.add_argument("--trials", type=int, default=20)
    tune.add_argument("--max-estimators", type=int, default=1000)
    tune.add_argument("--early-stopping-rounds", type=int, default=60)
    tune.add_argument("--search-seed", type=int, default=20260906)
    tune.add_argument("--seeds", type=int, nargs="+", default=[0])
    tune.add_argument("--n-jobs", type=int, default=-1)
    tune.add_argument("--study-prefix", default="mentor_artifact_typed_ksvd")
    tune.add_argument(
        "--protocol-id",
        default="luyin16-mentor-artifact-typed-ksvd-optuna-v1",
    )
    tune.set_defaults(func=run_tune)

    test = subparsers.add_parser("test")
    test.add_argument("--features", type=Path, required=True)
    test.add_argument("--tuning", type=Path, required=True)
    test.add_argument("--result", type=Path, required=True)
    test.add_argument("--views", default="best")
    test.add_argument("--seeds", type=int, nargs="+", default=[0])
    test.add_argument("--n-jobs", type=int, default=-1)
    test.add_argument(
        "--protocol-id",
        default="luyin16-mentor-artifact-typed-ksvd-frozen-test-v1",
    )
    test.set_defaults(func=run_test)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    result = args.func(args)
    if args.command == "tune":
        print(json.dumps(_jsonable(result["selection"]), ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

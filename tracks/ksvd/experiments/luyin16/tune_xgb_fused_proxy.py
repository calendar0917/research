"""Train-inner XGBoost search on frozen luyin16 feature views.

Optuna TPE is used when available, with a deterministic ParameterSampler
fallback. Feature construction is frozen in a separate NPZ; the official
validation split is touched only after train-inner search is complete.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
from scipy.stats import loguniform, randint, uniform
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import ParameterSampler, StratifiedKFold
from xgboost import XGBClassifier

try:
    import optuna
except ImportError:  # pragma: no cover - exercised only in minimal environments
    optuna = None


DEFAULT_VIEWS = ("s", "s_r_raw", "s_r_final")
InnerSplit = tuple[np.ndarray, np.ndarray]


def _params(seed: int) -> dict[str, Any]:
    return {
        "n_estimators": randint(100, 601),
        "max_depth": randint(2, 8),
        "learning_rate": loguniform(0.01, 0.2),
        "min_child_weight": randint(1, 16),
        "subsample": uniform(0.65, 0.35),
        "colsample_bytree": uniform(0.65, 0.35),
        "reg_alpha": loguniform(1e-4, 3.0),
        "reg_lambda": loguniform(0.1, 30.0),
        "random_state": [seed],
    }


def _fit_auc(
    xtr: np.ndarray,
    ytr: np.ndarray,
    xva: np.ndarray,
    yva: np.ndarray,
    p: dict[str, Any],
    weight: float,
) -> float:
    model = XGBClassifier(
        **p,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=weight,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(xtr, ytr)
    return float(roc_auc_score(yva, model.predict_proba(xva)[:, 1]))


def stratified_inner_splits(y: np.ndarray, folds: int, seed: int) -> list[InnerSplit]:
    splitter = StratifiedKFold(n_splits=folds, shuffle=True, random_state=seed)
    return [
        (np.asarray(train, dtype=np.int64), np.asarray(valid, dtype=np.int64))
        for train, valid in splitter.split(np.zeros(y.shape[0]), y)
    ]


def scaffold_inner_splits(
    folds_file: Path,
    train_dataset_indices: np.ndarray,
    y: np.ndarray,
) -> tuple[list[InnerSplit], dict[str, Any]]:
    """Map a frozen official-train scaffold archive onto feature-row positions."""
    archive = np.load(folds_file, allow_pickle=False)
    original_indices = np.asarray(archive["original_indices"], dtype=np.int64)
    dataset_to_position = {
        int(dataset_index): int(position)
        for position, dataset_index in enumerate(train_dataset_indices)
    }
    fold_ids = sorted(
        int(name.removeprefix("fold_").removesuffix("_train_indices"))
        for name in archive.files
        if name.startswith("fold_") and name.endswith("_train_indices")
    )
    if len(fold_ids) < 2:
        raise ValueError(f"scaffold archive has fewer than two folds: {folds_file}")
    splits: list[InnerSplit] = []
    rows: list[dict[str, Any]] = []
    for fold in fold_ids:
        train_local = np.asarray(archive[f"fold_{fold}_train_indices"], dtype=np.int64)
        valid_local = np.asarray(archive[f"fold_{fold}_valid_indices"], dtype=np.int64)
        train_original = original_indices[train_local]
        valid_original = original_indices[valid_local]
        try:
            train = np.asarray(
                [dataset_to_position[int(index)] for index in train_original], dtype=np.int64
            )
            valid = np.asarray(
                [dataset_to_position[int(index)] for index in valid_original], dtype=np.int64
            )
        except KeyError as exc:
            raise ValueError(
                f"scaffold fold index {int(exc.args[0])} is absent from frozen official train"
            ) from exc
        if np.intersect1d(train, valid).size or np.unique(y[train]).size != 2:
            raise ValueError(f"invalid scaffold train split in fold {fold}")
        if np.unique(y[valid]).size != 2:
            raise ValueError(f"invalid scaffold validation split in fold {fold}")
        splits.append((train, valid))
        rows.append(
            {
                "fold": fold,
                "n_train": int(train.size),
                "n_valid": int(valid.size),
                "n_train_positive": int(y[train].sum()),
                "n_valid_positive": int(y[valid].sum()),
            }
        )
    return splits, {
        "kind": "official-train-only Bemis-Murcko scaffold folds",
        "path": str(folds_file.resolve()),
        "folds": rows,
    }


def _score_params(
    x: np.ndarray,
    y: np.ndarray,
    splits: Sequence[InnerSplit],
    params: dict[str, Any],
) -> list[float]:
    scores: list[float] = []
    for train, valid in splits:
        weight = float(np.sum(y[train] == 0) / max(1, np.sum(y[train] == 1)))
        scores.append(
            _fit_auc(x[train], y[train], x[valid], y[valid], params, weight)
        )
    return scores


def tune_view(
    x: np.ndarray,
    y: np.ndarray,
    n_trials: int,
    seed: int,
    splits: Sequence[InnerSplit],
) -> dict[str, Any]:
    if optuna is not None:

        def objective(trial: Any) -> float:
            params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 600),
                "max_depth": trial.suggest_int("max_depth", 2, 7),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.2, log=True),
                "min_child_weight": trial.suggest_int("min_child_weight", 1, 15),
                "subsample": trial.suggest_float("subsample", 0.65, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.65, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 1e-4, 3.0, log=True),
                "reg_lambda": trial.suggest_float("reg_lambda", 0.1, 30.0, log=True),
                "random_state": seed,
            }
            return float(np.mean(_score_params(x, y, splits, params)))

        study = optuna.create_study(
            direction="maximize",
            sampler=optuna.samplers.TPESampler(seed=seed),
        )
        study.optimize(objective, n_trials=n_trials, show_progress_bar=False)
        best_params = dict(study.best_trial.params) | {"random_state": seed}
        best = {
            "trial": int(study.best_trial.number),
            "params": best_params,
            "inner_auc": float(study.best_value),
            "fold_auc": _score_params(x, y, splits, best_params),
        }
        trials = [
            {
                "trial": int(t.number),
                "params": dict(t.params),
                "inner_auc": None if t.value is None else float(t.value),
                "state": str(t.state),
            }
            for t in study.trials
        ]
        return {"best": best, "trials": trials, "backend": "optuna_tpe"}

    candidates = list(ParameterSampler(_params(seed), n_iter=n_trials, random_state=seed))
    trials: list[dict[str, Any]] = []
    for trial_id, params in enumerate(candidates):
        fold_scores = _score_params(x, y, splits, dict(params))
        trials.append(
            {
                "trial": trial_id,
                "params": params,
                "inner_auc": float(np.mean(fold_scores)),
                "fold_auc": fold_scores,
            }
        )
    best = max(trials, key=lambda row: row["inner_auc"])
    return {"best": best, "trials": trials, "backend": "sklearn_parameter_sampler_fallback"}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--features", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument(
        "--protocol-id",
        default="mentor-typed-slot-proxy-v2-frozen-feature-search-v1",
    )
    parser.add_argument("--trials", type=int, default=12)
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--folds-file", type=Path)
    parser.add_argument(
        "--screen-only",
        action="store_true",
        help="Run train-inner search only; do not evaluate official validation.",
    )
    parser.add_argument("--baseline-view")
    parser.add_argument("--promotion-delta", type=float, default=-0.005)
    parser.add_argument("--minimum-fold-wins", type=int, default=2)
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    parser.add_argument("--views", default=",".join(DEFAULT_VIEWS))
    args = parser.parse_args()

    payload = np.load(args.features, allow_pickle=False)
    n_train = int(payload["train_count"][0])
    y = np.asarray(payload["labels"], dtype=np.int64)
    ytr = y[:n_train]
    train_dataset_indices = np.asarray(payload["dataset_indices"][:n_train], dtype=np.int64)
    if args.folds_file is None:
        inner_splits = stratified_inner_splits(ytr, args.folds, seed=0)
        inner_split_meta: dict[str, Any] = {
            "kind": f"{args.folds}-fold stratified random CV over official train only",
            "seed": 0,
        }
    else:
        inner_splits, inner_split_meta = scaffold_inner_splits(
            args.folds_file.expanduser().resolve(), train_dataset_indices, ytr
        )
    result: dict[str, Any] = {
        "protocol_id": str(args.protocol_id),
        "search_backend": "Optuna TPE sampler (with sklearn fallback if Optuna is unavailable)",
        "feature_file": str(args.features.resolve()),
        "inner_split": inner_split_meta,
        "screen_only": bool(args.screen_only),
        "official_validation_evaluated": not args.screen_only,
        "official_test_evaluated": False,
        "views": {},
    }
    view_names = [name.strip() for name in args.views.split(",") if name.strip()]
    if args.baseline_view is not None and args.baseline_view not in view_names:
        raise ValueError("--baseline-view must be included in --views")
    for view in view_names:
        x = np.asarray(payload[view], dtype=np.float32)
        xtr = x[:n_train]
        search = tune_view(xtr, ytr, args.trials, seed=0, splits=inner_splits)
        best_params = dict(search["best"]["params"])
        rows = []
        if not args.screen_only:
            xva = x[n_train:]
            yva = y[n_train:]
            for seed in args.seeds:
                p = dict(best_params)
                p["random_state"] = int(seed)
                weight = float(np.sum(ytr == 0) / max(1, np.sum(ytr == 1)))
                model = XGBClassifier(
                    **p,
                    objective="binary:logistic",
                    eval_metric="auc",
                    scale_pos_weight=weight,
                    n_jobs=-1,
                    tree_method="hist",
                )
                model.fit(xtr, ytr)
                pred_tr = model.predict_proba(xtr)[:, 1]
                pred_va = model.predict_proba(xva)[:, 1]
                rows.append(
                    {
                        "seed": int(seed),
                        "train_auc": float(roc_auc_score(ytr, pred_tr)),
                        "valid_auc": float(roc_auc_score(yva, pred_va)),
                    }
                )
        result["views"][view] = {
            "dimension": int(x.shape[1]),
            "best_inner_auc": float(search["best"]["inner_auc"]),
            "best_inner_fold_auc": [float(value) for value in search["best"]["fold_auc"]],
            "best_params": best_params,
            "final_rows": rows,
            "final_valid_mean": (
                float(np.mean([row["valid_auc"] for row in rows])) if rows else None
            ),
            "final_valid_std": (
                float(np.std([row["valid_auc"] for row in rows], ddof=1))
                if len(rows) > 1
                else None
            ),
            "n_trials": int(args.trials),
            "search_backend": search.get("backend"),
        }
    if args.baseline_view is not None:
        baseline = result["views"][args.baseline_view]
        baseline_folds = np.asarray(baseline["best_inner_fold_auc"], dtype=np.float64)
        for name, row in result["views"].items():
            fold_scores = np.asarray(row["best_inner_fold_auc"], dtype=np.float64)
            delta = float(row["best_inner_auc"] - baseline["best_inner_auc"])
            wins = int(np.sum(fold_scores > baseline_folds))
            row["promotion"] = {
                "baseline_view": args.baseline_view,
                "inner_delta": delta,
                "fold_wins": wins,
                "minimum_delta": float(args.promotion_delta),
                "minimum_fold_wins": int(args.minimum_fold_wins),
                "promoted": bool(
                    name == args.baseline_view
                    or (
                        delta >= args.promotion_delta
                        and wins >= args.minimum_fold_wins
                    )
                ),
            }
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(
        json.dumps(
            {
                name: {
                    "dim": info["dimension"],
                    "inner": info["best_inner_auc"],
                    "valid": info["final_valid_mean"],
                    "promotion": info.get("promotion"),
                }
                for name, info in result["views"].items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

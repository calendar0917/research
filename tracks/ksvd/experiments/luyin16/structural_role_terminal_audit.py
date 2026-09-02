"""Frozen terminal audit for the strongest clean structural-role candidates.

Stage ``tune`` encodes only official train/validation graphs, performs an
equal-budget official-train scaffold Optuna search for the pre-registered
rooted-WL T+A and centered views, and writes a frozen manifest.  Stage ``test``
requires that manifest, encodes official test once, and evaluates the frozen
candidates without any further selection.

The MolHIV official test was viewed by older routes in this repository.  This
entry point therefore records a controlled terminal evaluation, not an
untouched-test claim.
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import optuna
from sklearn.metrics import roc_auc_score
import yaml

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.role_attribute_binding_screen import (
    REPO_ROOT,
    _resolve,
    _sha256,
    _write_json,
)
from tracks.ksvd.experiments.luyin16.structural_role_fusion_screen import (
    graph_features,
)
from tracks.ksvd.experiments.luyin16.task_aligned_interaction_screen import (
    XGB_SEARCH_KEYS,
    _fit_xgb_predict,
    _score_xgb_params,
    _suggest_xgb_params,
)


DEFAULT_CONFIG = (
    REPO_ROOT
    / "tracks/ksvd/configs/luyin16/structural_role_terminal_audit.yaml"
)
CANDIDATES = ("t_a", "f_centered")
ENCODER_PATH = (
    REPO_ROOT
    / "tracks/ksvd/experiments/luyin16/structural_role_fusion_screen.py"
)


def terminal_graph_views(
    graph,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    representation: Mapping[str, Any],
) -> dict[str, np.ndarray]:
    blocks = graph_features(
        graph,
        node_features,
        edge_features,
        "rooted_wl",
        representation,
        shuffle_repeats=0,
    )
    marginals = np.concatenate(
        [
            blocks["node_role"],
            blocks["edge_role"],
            blocks["node_attribute"],
            blocks["edge_attribute"],
            blocks["context"],
        ]
    ).astype(np.float32, copy=False)
    centered = np.concatenate(
        [marginals, blocks["node_binding"], blocks["edge_binding"]]
    ).astype(np.float32, copy=False)
    return {"t_a": marginals, "f_centered": centered}


def _feature_signature(
    representation: Mapping[str, Any], indices: np.ndarray
) -> str:
    payload = {
        "schema": "rooted_wl",
        "representation": dict(representation),
        "dataset_indices": np.asarray(indices, dtype=np.int64).tolist(),
        "encoder_sha256": _sha256(ENCODER_PATH),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def load_or_build_view_cache(
    bundle,
    indices: np.ndarray,
    representation: Mapping[str, Any],
    cache_path: Path,
) -> tuple[dict[str, np.ndarray], bool]:
    selected = np.asarray(indices, dtype=np.int64)
    signature = _feature_signature(representation, selected)
    if cache_path.exists():
        with np.load(cache_path, allow_pickle=False) as archive:
            cached_signature = str(np.asarray(archive["signature"]).reshape(-1)[0])
            if cached_signature != signature:
                raise ValueError(f"feature cache signature mismatch: {cache_path}")
            payload = {name: np.asarray(archive[name]) for name in archive.files}
        return payload, True

    assert bundle.node_feats is not None and bundle.edge_feats is not None
    first_index = int(selected[0])
    first = terminal_graph_views(
        bundle.graphs[first_index],
        bundle.node_feats[first_index],
        bundle.edge_feats[first_index],
        representation,
    )
    matrices = {
        name: np.zeros((selected.size, values.size), dtype=np.float32)
        for name, values in first.items()
    }
    labels = np.zeros(selected.size, dtype=np.int64)
    for position, raw_index in enumerate(selected):
        index = int(raw_index)
        views = first if position == 0 else terminal_graph_views(
            bundle.graphs[index],
            bundle.node_feats[index],
            bundle.edge_feats[index],
            representation,
        )
        for name in CANDIDATES:
            matrices[name][position] = views[name]
        labels[position] = int(bundle.y[index])
        if position and position % 500 == 0:
            print(f"terminal feature graphs: {position}/{selected.size}", flush=True)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    temporary = cache_path.with_suffix(cache_path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez_compressed(
            handle,
            signature=np.asarray([signature]),
            dataset_indices=selected,
            labels=labels,
            **matrices,
        )
    temporary.replace(cache_path)
    return {
        "signature": np.asarray([signature]),
        "dataset_indices": selected,
        "labels": labels,
        **matrices,
    }, False


def official_train_scaffold_splits(
    fold_archive: Mapping[str, np.ndarray],
    dataset_indices: np.ndarray,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[dict[str, Any]]]:
    original = np.asarray(fold_archive["original_indices"], dtype=np.int64)
    mapping = {
        int(index): position
        for position, index in enumerate(np.asarray(dataset_indices, dtype=np.int64))
    }
    fold_ids = sorted(
        int(name.removeprefix("fold_").removesuffix("_train_indices"))
        for name in fold_archive
        if name.startswith("fold_") and name.endswith("_train_indices")
    )
    splits = []
    metadata = []
    for fold in fold_ids:
        train_indices = original[
            np.asarray(fold_archive[f"fold_{fold}_train_indices"], dtype=np.int64)
        ]
        valid_indices = original[
            np.asarray(fold_archive[f"fold_{fold}_valid_indices"], dtype=np.int64)
        ]
        train = np.asarray([mapping[int(index)] for index in train_indices], dtype=np.int64)
        valid = np.asarray([mapping[int(index)] for index in valid_indices], dtype=np.int64)
        if np.intersect1d(train, valid).size:
            raise RuntimeError(f"scaffold fold {fold} overlaps")
        splits.append((train, valid))
        metadata.append(
            {
                "fold": int(fold),
                "n_train": int(train.size),
                "n_valid": int(valid.size),
            }
        )
    return splits, metadata


def _search_values_in_range(
    params: Mapping[str, Any], ranges: Mapping[str, Sequence[float]]
) -> bool:
    return all(
        key in params
        and float(ranges[key][0]) <= float(params[key]) <= float(ranges[key][1])
        for key in XGB_SEARCH_KEYS
    )


def _strip_search_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {key: params[key] for key in XGB_SEARCH_KEYS if key in params}


def warm_start_params(
    previous_result: Path | None,
    view: str,
    fixed_classifier: Mapping[str, Any],
    ranges: Mapping[str, Sequence[float]],
) -> list[dict[str, Any]]:
    candidates = [_strip_search_params(fixed_classifier)]
    if previous_result is not None and previous_result.exists():
        payload = json.loads(previous_result.read_text(encoding="utf-8"))
        for fold in payload.get("outer_folds", []):
            row = fold.get("xgb", {}).get(view)
            if row is not None:
                candidates.append(_strip_search_params(row["search"]["best_params"]))
    unique = []
    seen = set()
    for params in candidates:
        key = json.dumps(params, sort_keys=True)
        if key not in seen and _search_values_in_range(params, ranges):
            unique.append(params)
            seen.add(key)
    return unique


def tune_terminal_view(
    matrix: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    tuning: Mapping[str, Any],
    fixed_classifier: Mapping[str, Any],
    warm_result: Path | None,
    *,
    view: str,
    seed: int,
) -> dict[str, Any]:
    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(seed)),
    )
    warm = warm_start_params(
        warm_result,
        view,
        fixed_classifier,
        tuning["ranges"],
    )
    for params in warm:
        study.enqueue_trial(params)

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgb_params(trial, tuning, seed=seed)
        params.update(tuning.get("view_static", {}).get(view, {}))
        try:
            return float(np.mean(_score_xgb_params(matrix, labels, splits, params)))
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
            "random_state": int(seed),
            "tree_method": "hist",
            "n_jobs": int(tuning.get("n_jobs", -1)),
        }
    )
    best_params.update(tuning.get("view_static", {}).get(view, {}))
    fold_auc = _score_xgb_params(matrix, labels, splits, best_params)
    return {
        "backend": "optuna_tpe",
        "n_trials": int(tuning["n_trials"]),
        "n_warm_starts": len(warm),
        "best_trial": int(study.best_trial.number),
        "best_params": best_params,
        "best_scaffold_auc": float(np.mean(fold_auc)),
        "best_scaffold_fold_auc": fold_auc,
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


def evaluate_frozen_split(
    matrix: np.ndarray,
    labels: np.ndarray,
    train_rows: np.ndarray,
    valid_rows: np.ndarray,
    params: Mapping[str, Any],
    seeds: Sequence[int],
) -> dict[str, Any]:
    rows = []
    predictions = []
    y_valid = labels[valid_rows]
    for seed in seeds:
        current = dict(params)
        current["random_state"] = int(seed)
        prediction = _fit_xgb_predict(
            matrix[train_rows], labels[train_rows], matrix[valid_rows], current
        )
        predictions.append(prediction)
        rows.append(
            {
                "seed": int(seed),
                "auc": float(roc_auc_score(y_valid, prediction)),
            }
        )
    ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
    return {
        "rows": rows,
        "mean_auc": float(np.mean([row["auc"] for row in rows])),
        "std_auc": float(np.std([row["auc"] for row in rows])),
        "seed_ensemble_auc": float(roc_auc_score(y_valid, ensemble)),
    }


def _tune_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Rooted-WL terminal candidate freeze",
        "",
        f"Protocol: `{result['protocol_id']}`",
        "",
        "Official test was not encoded or evaluated in this stage.",
        "",
        "| view | train scaffold CV | official-valid mean | valid seed ensemble |",
        "|---|---:|---:|---:|",
    ]
    for view in CANDIDATES:
        row = result["views"][view]
        lines.append(
            f"| `{view}` | {row['search']['best_scaffold_auc']:.6f} | "
            f"{row['official_valid']['mean_auc']:.6f} | "
            f"{row['official_valid']['seed_ensemble_auc']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Frozen test candidates: `t_a`, `f_centered`, and their fixed 50/50 prediction ensemble.",
            "",
        ]
    )
    return "\n".join(lines)


def run_tune(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    representation = config["representation"]
    tuning = config["tuning"]
    fixed_classifier = config["fixed_classifier"]
    folds_path = _resolve(data_config["scaffold_folds"])
    dev_cache = _resolve(data_config["dev_feature_cache"])
    output_json = _resolve(config["frozen_manifest_json"])
    output_markdown = _resolve(config["frozen_manifest_markdown"])
    checkpoint_path = _resolve(config["search_checkpoint_json"])
    previous_result_value = data_config.get("warm_start_result")
    previous_result = (
        _resolve(previous_result_value) if previous_result_value is not None else None
    )
    with np.load(folds_path, allow_pickle=False) as archive:
        fold_archive = {name: np.asarray(archive[name]) for name in archive.files}
    original = np.asarray(fold_archive["original_indices"], dtype=np.int64)
    train_indices = original[
        np.asarray(fold_archive["official_train_indices"], dtype=np.int64)
    ]
    valid_indices = original[
        np.asarray(fold_archive["official_valid_indices"], dtype=np.int64)
    ]
    dev_indices = np.concatenate([train_indices, valid_indices]).astype(np.int64)
    bundle = load_molhiv(root=_resolve(data_config["root"]), with_features=True)
    start = time.perf_counter()
    cache, cache_hit = load_or_build_view_cache(
        bundle, dev_indices, representation, dev_cache
    )
    labels = np.asarray(cache["labels"], dtype=np.int64)
    scaffold_splits, split_meta = official_train_scaffold_splits(
        fold_archive, dev_indices
    )
    train_rows = np.arange(train_indices.size, dtype=np.int64)
    valid_rows = np.arange(train_indices.size, dev_indices.size, dtype=np.int64)
    seeds = [int(value) for value in config["model_seeds"]]
    views: dict[str, Any] = {}
    if checkpoint_path.exists():
        checkpoint = json.loads(checkpoint_path.read_text(encoding="utf-8"))
        if checkpoint["config_sha256"] != _sha256(config_path):
            raise RuntimeError("search checkpoint belongs to a different configuration")
        if checkpoint["implementation_sha256"] != _sha256(Path(__file__)):
            raise RuntimeError("search checkpoint belongs to a different implementation")
        views.update(checkpoint.get("views", {}))
    for view_id, view in enumerate(CANDIDATES):
        if view in views:
            print(f"[freeze] resume cached search for {view}", flush=True)
            continue
        matrix = np.asarray(cache[view], dtype=np.float32)
        search = tune_terminal_view(
            matrix,
            labels,
            scaffold_splits,
            tuning,
            fixed_classifier,
            previous_result,
            view=view,
            seed=int(tuning["seed"]) + 1009 * view_id,
        )
        official_valid = evaluate_frozen_split(
            matrix,
            labels,
            train_rows,
            valid_rows,
            search["best_params"],
            seeds,
        )
        views[view] = {
            "dimension": int(matrix.shape[1]),
            "search": search,
            "official_valid": official_valid,
        }
        print(
            f"[freeze] {view}: scaffold={search['best_scaffold_auc']:.6f}; "
            f"official-valid={official_valid['mean_auc']:.6f}; "
            f"ensemble={official_valid['seed_ensemble_auc']:.6f}",
            flush=True,
        )
        _write_json(
            checkpoint_path,
            {
                "config_sha256": _sha256(config_path),
                "implementation_sha256": _sha256(Path(__file__)),
                "encoder_sha256": _sha256(ENCODER_PATH),
                "views": views,
            },
        )
    result = {
        "protocol_id": str(config["protocol_id"]),
        "stage": "candidate_freeze_before_terminal_test",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "implementation_sha256": _sha256(Path(__file__)),
        "encoder_sha256": _sha256(ENCODER_PATH),
        "audit_boundary": {
            "official_train_used_for_search": True,
            "official_validation_used_for_frozen_reporting": True,
            "official_test_encoded": False,
            "official_test_evaluated": False,
            "historical_test_already_seen_by_older_routes": True,
        },
        "selection_rule": {
            "candidates_pre_registered": list(CANDIDATES),
            "test_view_ensemble": "fixed 0.5*t_a + 0.5*f_centered probabilities",
            "test_will_not_select_hyperparameters_or_views": True,
        },
        "data": {
            "dataset": str(data_config["dataset"]),
            "folds": str(folds_path),
            "folds_sha256": _sha256(folds_path),
            "dev_feature_cache": str(dev_cache),
            "dev_feature_cache_hit": bool(cache_hit),
            "n_train": int(train_indices.size),
            "n_valid": int(valid_indices.size),
            "scaffold_folds": split_meta,
        },
        "representation": dict(representation),
        "model_seeds": seeds,
        "views": views,
        "runtime": {
            "seconds": float(time.perf_counter() - start),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    _write_json(output_json, result)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_tune_markdown(result), encoding="utf-8")
    print(f"frozen manifest written: {output_json}", flush=True)
    return result


def _fit_test_predictions(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    test_matrix: np.ndarray,
    params: Mapping[str, Any],
    seeds: Sequence[int],
) -> tuple[list[dict[str, Any]], list[np.ndarray]]:
    rows = []
    predictions = []
    for seed in seeds:
        current = dict(params)
        current["random_state"] = int(seed)
        prediction = _fit_xgb_predict(
            train_matrix, train_labels, test_matrix, current
        )
        predictions.append(prediction)
        rows.append({"seed": int(seed)})
    return rows, predictions


def _score_test_prediction_set(
    rows: list[dict[str, Any]],
    predictions: Sequence[np.ndarray],
    test_labels: np.ndarray,
) -> dict[str, Any]:
    scored_rows = []
    for row, prediction in zip(rows, predictions, strict=True):
        scored_rows.append(
            dict(row)
            | {"test_auc": float(roc_auc_score(test_labels, prediction))}
        )
    ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
    return {
        "rows": scored_rows,
        "mean_auc": float(np.mean([row["test_auc"] for row in scored_rows])),
        "std_auc": float(np.std([row["test_auc"] for row in scored_rows])),
        "seed_ensemble_auc": float(roc_auc_score(test_labels, ensemble)),
    }


def _test_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# Rooted-WL frozen terminal test audit",
        "",
        "Historical disclosure: official test had already been viewed by older routes; this is a controlled frozen terminal evaluation.",
        "",
    ]
    for scope in ("train_only", "train_valid_refit"):
        lines.extend(
            [
                f"## {scope}",
                "",
                "| candidate | five-seed mean AUC | seed-ensemble AUC |",
                "|---|---:|---:|",
            ]
        )
        for view in CANDIDATES:
            row = result["views"][view][scope]
            lines.append(
                f"| `{view}` | {row['mean_auc']:.6f} | {row['seed_ensemble_auc']:.6f} |"
            )
        ensemble = result["fixed_view_ensemble"][scope]
        lines.append(
            f"| `0.5*t_a+0.5*f_centered` | {ensemble['paired_seed_mean_auc']:.6f} | {ensemble['all_model_ensemble_auc']:.6f} |"
        )
        lines.append("")
    return "\n".join(lines)


def run_test(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data_config = config["data"]
    representation = config["representation"]
    frozen_path = _resolve(config["frozen_manifest_json"])
    output_json = _resolve(config["terminal_test_json"])
    output_markdown = _resolve(config["terminal_test_markdown"])
    if output_json.exists():
        raise FileExistsError(
            f"terminal test result already exists; refusing repeated evaluation: {output_json}"
        )
    frozen = json.loads(frozen_path.read_text(encoding="utf-8"))
    if frozen["config_sha256"] != _sha256(config_path):
        raise RuntimeError("configuration changed after candidate freeze")
    if frozen["implementation_sha256"] != _sha256(Path(__file__)):
        raise RuntimeError("terminal implementation changed after candidate freeze")
    if frozen["encoder_sha256"] != _sha256(ENCODER_PATH):
        raise RuntimeError("rooted-WL encoder changed after candidate freeze")

    folds_path = _resolve(data_config["scaffold_folds"])
    dev_cache_path = _resolve(data_config["dev_feature_cache"])
    test_cache_path = _resolve(data_config["test_feature_cache"])
    with np.load(folds_path, allow_pickle=False) as archive:
        fold_archive = {name: np.asarray(archive[name]) for name in archive.files}
    original = np.asarray(fold_archive["original_indices"], dtype=np.int64)
    train_indices = original[
        np.asarray(fold_archive["official_train_indices"], dtype=np.int64)
    ]
    valid_indices = original[
        np.asarray(fold_archive["official_valid_indices"], dtype=np.int64)
    ]
    test_indices = original[
        np.asarray(fold_archive["official_test_indices"], dtype=np.int64)
    ]
    dev_indices = np.concatenate([train_indices, valid_indices]).astype(np.int64)
    bundle = load_molhiv(root=_resolve(data_config["root"]), with_features=True)
    start = time.perf_counter()
    dev_cache, dev_hit = load_or_build_view_cache(
        bundle, dev_indices, representation, dev_cache_path
    )
    test_cache, test_hit = load_or_build_view_cache(
        bundle, test_indices, representation, test_cache_path
    )
    dev_labels = np.asarray(dev_cache["labels"], dtype=np.int64)
    test_labels = np.asarray(test_cache["labels"], dtype=np.int64)
    seeds = [int(value) for value in frozen["model_seeds"]]
    scopes = {
        "train_only": np.arange(train_indices.size, dtype=np.int64),
        "train_valid_refit": np.arange(dev_indices.size, dtype=np.int64),
    }
    view_predictions: dict[str, dict[str, list[np.ndarray]]] = {
        view: {} for view in CANDIDATES
    }
    views = {}
    for view in CANDIDATES:
        dev_matrix = np.asarray(dev_cache[view], dtype=np.float32)
        test_matrix = np.asarray(test_cache[view], dtype=np.float32)
        params = frozen["views"][view]["search"]["best_params"]
        views[view] = {
            "dimension": int(dev_matrix.shape[1]),
            "best_params": params,
        }
        for scope, rows in scopes.items():
            seed_rows, predictions = _fit_test_predictions(
                dev_matrix[rows],
                dev_labels[rows],
                test_matrix,
                params,
                seeds,
            )
            view_predictions[view][scope] = predictions
            views[view][scope] = _score_test_prediction_set(
                seed_rows, predictions, test_labels
            )
            print(
                f"[terminal test] {scope} {view}: mean="
                f"{views[view][scope]['mean_auc']:.6f}; ensemble="
                f"{views[view][scope]['seed_ensemble_auc']:.6f}",
                flush=True,
            )

    fixed_ensemble = {}
    for scope in scopes:
        left = view_predictions["t_a"][scope]
        right = view_predictions["f_centered"][scope]
        paired = [0.5 * a + 0.5 * b for a, b in zip(left, right, strict=True)]
        paired_auc = [float(roc_auc_score(test_labels, pred)) for pred in paired]
        all_model_prediction = np.mean(
            np.stack([*left, *right], axis=0), axis=0
        )
        fixed_ensemble[scope] = {
            "rule": "0.5*t_a + 0.5*f_centered paired by seed",
            "paired_seed_auc": paired_auc,
            "paired_seed_mean_auc": float(np.mean(paired_auc)),
            "paired_seed_std_auc": float(np.std(paired_auc)),
            "all_model_ensemble_auc": float(
                roc_auc_score(test_labels, all_model_prediction)
            ),
        }
        print(
            f"[terminal test] {scope} fixed view ensemble: paired mean="
            f"{fixed_ensemble[scope]['paired_seed_mean_auc']:.6f}; all-model="
            f"{fixed_ensemble[scope]['all_model_ensemble_auc']:.6f}",
            flush=True,
        )

    result = {
        "protocol_id": str(config["protocol_id"]),
        "stage": "controlled_frozen_terminal_test",
        "frozen_manifest": str(frozen_path),
        "frozen_manifest_sha256": _sha256(frozen_path),
        "audit_boundary": {
            "official_test_encoded": True,
            "official_test_evaluated": True,
            "test_used_for_selection_or_tuning": False,
            "historical_test_already_seen_by_older_routes": True,
            "untouched_test_claim": False,
        },
        "split_sizes": {
            "train": int(train_indices.size),
            "valid": int(valid_indices.size),
            "test": int(test_indices.size),
            "test_positive": int(test_labels.sum()),
        },
        "feature_cache": {
            "dev": str(dev_cache_path),
            "dev_cache_hit": bool(dev_hit),
            "test": str(test_cache_path),
            "test_cache_hit": bool(test_hit),
        },
        "model_seeds": seeds,
        "views": views,
        "fixed_view_ensemble": fixed_ensemble,
        "runtime": {
            "seconds": float(time.perf_counter() - start),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    _write_json(output_json, result)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_test_markdown(result), encoding="utf-8")
    print(f"terminal test ledger written: {output_json}", flush=True)
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("tune", "test"), required=True)
    args = parser.parse_args(argv)
    config_path = args.config.expanduser().resolve()
    if args.stage == "tune":
        run_tune(config_path)
    else:
        run_test(config_path)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

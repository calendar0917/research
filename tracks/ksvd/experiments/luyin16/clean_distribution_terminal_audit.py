"""Freeze and terminal-test the clean local-distribution readout.

Stage ``tune`` uses only official-train scaffold folds to tune the pre-
registered S+local candidates and writes a frozen manifest.  Stage ``test``
then builds the test local readout and evaluates every frozen candidate once,
including train-only and train+valid refits.  The official MolHIV test was
seen by older routes, so this is recorded as a controlled terminal evaluation.
"""

from __future__ import annotations

import argparse
import json
import platform
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import yaml
from sklearn.metrics import roc_auc_score

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.clean_patch_distribution_readout import (
    LOCAL_BLOCKS,
    STAT_NAMES,
    build_distribution_features,
)
from tracks.ksvd.experiments.luyin16.role_attribute_binding_screen import (
    REPO_ROOT,
    _resolve,
    _sha256,
    _write_json,
)
from tracks.ksvd.experiments.luyin16.structural_role_terminal_audit import (
    evaluate_frozen_split,
    official_train_scaffold_splits,
    _fit_test_predictions,
    _score_test_prediction_set,
)
from tracks.ksvd.experiments.luyin16.task_aligned_interaction_screen import (
    XGB_SEARCH_KEYS,
    _score_xgb_params,
    _suggest_xgb_params,
)


DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/clean_distribution_terminal_audit.yaml"
DEFAULT_DIR = REPO_ROOT / "tracks/ksvd/results/luyin16/clean_distribution_terminal_audit"
LOCAL_DIMENSION = 149


def _load_config(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"configuration must be a mapping: {path}")
    return payload


def _candidate_views(global_s: np.ndarray, distribution: np.ndarray, context: np.ndarray) -> dict[str, np.ndarray]:
    if distribution.ndim != 2 or distribution.shape[1] != len(STAT_NAMES) * LOCAL_DIMENSION:
        raise ValueError(f"unexpected distribution shape: {distribution.shape}")
    local_mean = np.concatenate([distribution[:, :LOCAL_DIMENSION], context], axis=1)
    local_mean_std = np.concatenate([distribution[:, : 2 * LOCAL_DIMENSION], context], axis=1)
    local_all12 = np.concatenate([distribution, context], axis=1)
    return {
        "s_ta_mean": np.concatenate([global_s, local_mean], axis=1).astype(np.float32, copy=False),
        "s_ta_mean_std": np.concatenate([global_s, local_mean_std], axis=1).astype(np.float32, copy=False),
        "s_ta_all12": np.concatenate([global_s, local_all12], axis=1).astype(np.float32, copy=False),
    }


def _strip_params(params: Mapping[str, Any]) -> dict[str, Any]:
    return {key: params[key] for key in XGB_SEARCH_KEYS if key in params}


def _tune_view(
    matrix: np.ndarray,
    labels: np.ndarray,
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
    tuning: Mapping[str, Any],
    fixed_classifier: Mapping[str, Any],
    view: str,
    seed: int,
) -> dict[str, Any]:
    import optuna

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=int(seed)),
    )
    warm = [_strip_params(fixed_classifier)]

    def objective(trial: optuna.Trial) -> float:
        params = _suggest_xgb_params(trial, tuning, seed=seed)
        params.update(tuning.get("view_static", {}).get(view, {}))
        return float(np.mean(_score_xgb_params(matrix, labels, splits, params)))

    for params in warm:
        study.enqueue_trial(params)
    study.optimize(objective, n_trials=int(tuning["n_trials"]), show_progress_bar=False)
    best = dict(study.best_trial.params)
    best.update(
        {
            "objective": "binary:logistic",
            "eval_metric": "auc",
            "tree_method": "hist",
            "random_state": int(seed),
            "n_jobs": int(tuning.get("n_jobs", 8)),
        }
    )
    best.update(tuning.get("view_static", {}).get(view, {}))
    fold_auc = _score_xgb_params(matrix, labels, splits, best)
    return {
        "backend": "optuna_tpe",
        "n_trials": int(tuning["n_trials"]),
        "best_trial": int(study.best_trial.number),
        "best_params": best,
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


def _load_dev_arrays(config: Mapping[str, Any], result_dir: Path):
    data = config["data"]
    dev_s_path = _resolve(data["dev_global_s"])
    with np.load(dev_s_path, allow_pickle=False) as archive:
        indices = np.asarray(archive["dataset_indices"], dtype=np.int64)
        labels = np.asarray(archive["labels"], dtype=np.int64)
        global_s = np.asarray(archive["s"], dtype=np.float32)
        train_count = int(np.asarray(archive["train_count"]).reshape(-1)[0])
    bundle = load_molhiv(root=_resolve(data["root"]), with_features=True)
    distribution, context, cache_hit = build_distribution_features(
        bundle,
        indices,
        config["representation"],
        _resolve(data["dev_distribution_cache"]),
    )
    if global_s.shape[0] != indices.size or labels.shape[0] != indices.size:
        raise RuntimeError("dev global S and local distribution rows are not aligned")
    return bundle, indices, labels, global_s, distribution, context, train_count, cache_hit


def run_tune(config_path: Path, result_dir: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    data = config["data"]
    folds_path = _resolve(data["scaffold_folds"])
    with np.load(folds_path, allow_pickle=False) as archive:
        fold_archive = {name: np.asarray(archive[name]) for name in archive.files}
    bundle, indices, labels, global_s, distribution, context, train_count, cache_hit = _load_dev_arrays(
        config, result_dir
    )
    del bundle
    views = _candidate_views(global_s, distribution, context)
    scaffold_splits, split_meta = official_train_scaffold_splits(fold_archive, indices)
    train_rows = np.arange(train_count, dtype=np.int64)
    valid_rows = np.arange(train_count, indices.size, dtype=np.int64)
    model_seeds = [int(value) for value in config["model_seeds"]]
    tuning = config["tuning"]
    fixed_classifier = config["fixed_classifier"]
    result_views: dict[str, Any] = {}
    start = time.perf_counter()
    for view_id, view in enumerate(config["candidate_views"]):
        matrix = views[view]
        print(f"[tune] {view}: dim={matrix.shape[1]}", flush=True)
        search = _tune_view(
            matrix[train_rows],
            labels[train_rows],
            scaffold_splits,
            tuning,
            fixed_classifier,
            view,
            int(tuning["seed"]) + 1009 * view_id,
        )
        official_valid = evaluate_frozen_split(
            matrix,
            labels,
            train_rows,
            valid_rows,
            search["best_params"],
            model_seeds,
        )
        result_views[view] = {
            "dimension": int(matrix.shape[1]),
            "search": search,
            "official_valid": official_valid,
        }
        print(
            f"[tune] {view}: scaffold={search['best_scaffold_auc']:.6f}; "
            f"official-valid={official_valid['mean_auc']:.6f}; "
            f"ensemble={official_valid['seed_ensemble_auc']:.6f}",
            flush=True,
        )
    manifest = {
        "protocol_id": str(config["protocol_id"]),
        "stage": "candidate_freeze_before_terminal_test",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "implementation_sha256": _sha256(Path(__file__)),
        "readout_encoder_sha256": _sha256(
            REPO_ROOT / "tracks/ksvd/experiments/luyin16/structural_role_fusion_screen.py"
        ),
        "audit_boundary": {
            "official_train_used_for_search": True,
            "official_validation_used_for_frozen_reporting": True,
            "official_test_encoded": False,
            "official_test_evaluated": False,
            "historical_test_already_seen_by_older_routes": True,
        },
        "selection_rule": {
            "pre_registered_views": list(config["candidate_views"]),
            "pre_registered_fusion": "fixed 0.5*s_ta_mean_std + 0.5*s_ta_all12 probabilities",
            "test_will_not_select_view_or_parameters": True,
        },
        "data": {
            "dataset": str(data["dataset"]),
            "scaffold_folds": str(folds_path),
            "scaffold_folds_sha256": _sha256(folds_path),
            "dev_global_s": str(_resolve(data["dev_global_s"])),
            "dev_distribution_cache": str(_resolve(data["dev_distribution_cache"])),
            "dev_distribution_cache_hit": bool(cache_hit),
            "n_train": int(train_count),
            "n_valid": int(indices.size - train_count),
            "scaffold_split_metadata": split_meta,
        },
        "representation": dict(config["representation"]),
        "local_blocks": list(LOCAL_BLOCKS),
        "statistics": list(STAT_NAMES),
        "model_seeds": model_seeds,
        "views": result_views,
        "runtime": {
            "seconds": float(time.perf_counter() - start),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    result_dir.mkdir(parents=True, exist_ok=True)
    _write_json(result_dir / str(config["output"]["frozen_manifest"]), manifest)
    lines = [
        "# Clean local-distribution terminal candidate freeze",
        "",
        f"Protocol: `{manifest['protocol_id']}`",
        "",
        "Official test was not encoded or evaluated in this stage.",
        "",
        "| view | train scaffold CV | official-valid mean | valid seed ensemble |",
        "|---|---:|---:|---:|",
    ]
    for view in config["candidate_views"]:
        row = result_views[view]
        lines.append(
            f"| `{view}` | {row['search']['best_scaffold_auc']:.6f} | "
            f"{row['official_valid']['mean_auc']:.6f} | "
            f"{row['official_valid']['seed_ensemble_auc']:.6f} |"
        )
    lines.extend(["", "All views and parameters are frozen before the separate test stage."])
    (result_dir / "frozen_manifest.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def _test_views(
    dev_views: Mapping[str, np.ndarray],
    test_views: Mapping[str, np.ndarray],
    dev_labels: np.ndarray,
    test_labels: np.ndarray,
    train_count: int,
    manifest: Mapping[str, Any],
) -> dict[str, Any]:
    seeds = [int(value) for value in manifest["model_seeds"]]
    scopes = {
        "train_only": np.arange(train_count, dtype=np.int64),
        "train_valid_refit": np.arange(dev_labels.size, dtype=np.int64),
    }
    predictions: dict[str, dict[str, list[np.ndarray]]] = {}
    result_views: dict[str, Any] = {}
    for view in manifest["selection_rule"]["pre_registered_views"]:
        params = manifest["views"][view]["search"]["best_params"]
        predictions[view] = {}
        result_views[view] = {
            "dimension": int(dev_views[view].shape[1]),
            "best_params": params,
        }
        for scope, rows in scopes.items():
            seed_rows, pred = _fit_test_predictions(
                dev_views[view][rows],
                dev_labels[rows],
                test_views[view],
                params,
                seeds,
            )
            predictions[view][scope] = pred
            result_views[view][scope] = _score_test_prediction_set(
                seed_rows, pred, test_labels
            )
            print(
                f"[test] {scope} {view}: mean={result_views[view][scope]['mean_auc']:.6f}; "
                f"ensemble={result_views[view][scope]['seed_ensemble_auc']:.6f}",
                flush=True,
            )
    fusion = {}
    for scope in scopes:
        left = predictions["s_ta_mean_std"][scope]
        right = predictions["s_ta_all12"][scope]
        paired = [0.5 * a + 0.5 * b for a, b in zip(left, right, strict=True)]
        fusion[scope] = {
            "rule": "fixed 0.5*s_ta_mean_std + 0.5*s_ta_all12 probabilities",
            "paired_seed_auc": [float(roc_auc_score(test_labels, p)) for p in paired],
            "paired_seed_mean_auc": float(
                np.mean([roc_auc_score(test_labels, p) for p in paired])
            ),
            "all_model_ensemble_auc": float(
                roc_auc_score(test_labels, np.mean(np.stack([*left, *right]), axis=0))
            ),
        }
        print(
            f"[test] {scope} fixed fusion: mean={fusion[scope]['paired_seed_mean_auc']:.6f}; "
            f"ensemble={fusion[scope]['all_model_ensemble_auc']:.6f}",
            flush=True,
        )
    return {"views": result_views, "fixed_fusion": fusion}


def run_test(config_path: Path, result_dir: Path) -> dict[str, Any]:
    config = _load_config(config_path)
    manifest_path = result_dir / str(config["output"]["frozen_manifest"])
    output_path = result_dir / str(config["output"]["terminal_test"])
    if output_path.exists():
        raise FileExistsError(f"refusing repeated terminal test: {output_path}")
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest["config_sha256"] != _sha256(config_path):
        raise RuntimeError("configuration changed after freeze")
    if manifest["implementation_sha256"] != _sha256(Path(__file__)):
        raise RuntimeError("implementation changed after freeze")
    encoder_path = REPO_ROOT / "tracks/ksvd/experiments/luyin16/structural_role_fusion_screen.py"
    if manifest["readout_encoder_sha256"] != _sha256(encoder_path):
        raise RuntimeError("readout encoder changed after freeze")
    data = config["data"]
    dev_s = np.load(_resolve(data["dev_global_s"]), allow_pickle=False)
    dev_indices = np.asarray(dev_s["dataset_indices"], dtype=np.int64)
    dev_labels = np.asarray(dev_s["labels"], dtype=np.int64)
    global_s_dev = np.asarray(dev_s["s"], dtype=np.float32)
    train_count = int(np.asarray(dev_s["train_count"]).reshape(-1)[0])
    full = np.load(_resolve(data["full_feature_file"]), allow_pickle=False)
    test_indices = np.asarray(full["test_indices"], dtype=np.int64)
    test_labels = np.asarray(full["labels"], dtype=np.int64)[test_indices]
    global_s_test = np.asarray(full["composition"], dtype=np.float32)[test_indices]
    bundle = load_molhiv(root=_resolve(data["root"]), with_features=True)
    dev_distribution, dev_context, dev_hit = build_distribution_features(
        bundle,
        dev_indices,
        config["representation"],
        _resolve(data["dev_distribution_cache"]),
    )
    test_cache = result_dir / "test_distribution_features.npz"
    test_distribution, test_context, test_hit = build_distribution_features(
        bundle,
        test_indices,
        config["representation"],
        test_cache,
    )
    dev_views = _candidate_views(global_s_dev, dev_distribution, dev_context)
    test_views = _candidate_views(global_s_test, test_distribution, test_context)
    evaluated = _test_views(
        dev_views, test_views, dev_labels, test_labels, train_count, manifest
    )
    result = {
        "protocol_id": str(config["protocol_id"]),
        "stage": "controlled_frozen_terminal_test",
        "frozen_manifest": str(manifest_path),
        "frozen_manifest_sha256": _sha256(manifest_path),
        "official_test_evaluated": True,
        "untouched_test_claim": False,
        "test_used_for_selection_or_tuning": False,
        "historical_test_already_seen_by_older_routes": True,
        "split_sizes": {
            "train": int(train_count),
            "valid": int(dev_labels.size - train_count),
            "test": int(test_labels.size),
            "test_positive": int(test_labels.sum()),
        },
        "feature_cache": {
            "dev_distribution": str(_resolve(data["dev_distribution_cache"])),
            "dev_cache_hit": bool(dev_hit),
            "test_distribution": str(test_cache),
            "test_cache_hit": bool(test_hit),
        },
        "views": evaluated["views"],
        "fixed_fusion": evaluated["fixed_fusion"],
        "model_seeds": list(manifest["model_seeds"]),
        "representation": dict(config["representation"]),
    }
    _write_json(output_path, result)
    lines = [
        "# Clean local-distribution frozen terminal test",
        "",
        "Historical disclosure: official test was viewed by older routes; this is a controlled frozen evaluation.",
        "",
    ]
    for scope in ("train_only", "train_valid_refit"):
        lines.extend([
            f"## {scope}",
            "",
            "| candidate | five-seed mean AUC | seed/model ensemble AUC |",
            "|---|---:|---:|",
        ])
        for view, row in evaluated["views"].items():
            lines.append(
                f"| `{view}` | {row[scope]['mean_auc']:.6f} | {row[scope]['seed_ensemble_auc']:.6f} |"
            )
        row = evaluated["fixed_fusion"][scope]
        lines.append(
            f"| `fixed 0.5*mean_std+0.5*all12` | {row['paired_seed_mean_auc']:.6f} | {row['all_model_ensemble_auc']:.6f} |"
        )
        lines.append("")
    (result_dir / "terminal_test.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("tune", "test"), required=True)
    parser.add_argument("--result-dir", type=Path, default=DEFAULT_DIR)
    args = parser.parse_args()
    config_path = args.config.expanduser().resolve()
    result_dir = args.result_dir.expanduser().resolve()
    if args.stage == "tune":
        run_tune(config_path, result_dir)
    else:
        run_test(config_path, result_dir)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

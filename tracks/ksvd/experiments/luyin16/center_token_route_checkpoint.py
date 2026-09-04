"""Efficient checkpointed screen for the center-token route.

The original ``center_token_route`` entry point gives every view an equal
Optuna budget.  That is useful as a broad audit, but unnecessarily expensive
once the scientific question has narrowed to conditional binding and one
KSVD challenger.  This runner keeps the feature construction and leakage
boundaries identical while splitting the work into resumable stages:

``screen``
    Build one train-only scaffold checkpoint per fold and evaluate every
    pre-registered view with the same fixed classifier.
``tune``
    Tune only ``s_conditional`` and ``s_ksvd_final`` on the saved fold
    checkpoints, then fit the full train-only schema and report official
    validation once.
``test``
    Delegate to the frozen terminal-test implementation.  No view or
    parameter is selected using test.

The protocol is deliberately separate from the aborted broad search.  Its
reduced KSVD budget and search budget are recorded in a new configuration and
manifest rather than silently changing the earlier protocol.
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
import yaml

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.center_token_route import (
    NULL_NAMES,
    VIEW_NAMES,
    _assemble_views,
    _build_base,
    _build_token_cache,
    _evaluate_frozen,
    _fixed_classifier,
    _fold_blocks,
    _load_indices,
    _resolve,
    _invariance_audit,
    _run_test,
    _score_view,
    _sha256,
    _tune_view,
)
from tracks.ksvd.experiments.luyin16.cross_center_interaction_terminal import (
    _official_train_scaffold_splits,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = REPO_ROOT / "tracks/ksvd/configs/luyin16/center_token_route_checkpoint.yaml"

CHECKPOINT_ARRAYS = (
    "base",
    "distribution",
    "conditional_compact",
    "conditional_pca",
    "ksvd_init",
    "ksvd_final",
    "conditional_shuffled_compact",
    "ksvd_final_shuffled",
    "y_train",
    "y_valid",
    "fold_train",
    "fold_valid",
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


def _config_signature(config_path: Path, config: Mapping[str, Any]) -> str:
    payload = {
        "protocol_id": config["protocol_id"],
        "config_sha256": _sha256(config_path),
        "implementation_sha256": _sha256(
            REPO_ROOT / "tracks/ksvd/experiments/luyin16/center_token_route.py"
        ),
        "representation": dict(config["representation"]),
        "dictionary": dict(config["dictionary"]),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _fold_signature(
    config_path: Path,
    config: Mapping[str, Any],
    fold_id: int,
    fold_train: np.ndarray,
    fold_valid: np.ndarray,
) -> str:
    payload = {
        "config": _config_signature(config_path, config),
        "fold": int(fold_id),
        "fold_train": np.asarray(fold_train, dtype=np.int64).tolist(),
        "fold_valid": np.asarray(fold_valid, dtype=np.int64).tolist(),
    }
    return json.dumps(payload, sort_keys=True, separators=(",", ":"))


def _save_fold_checkpoint(
    path: Path,
    signature: str,
    views: Mapping[str, np.ndarray],
    null_views: Mapping[str, np.ndarray],
    y_train: np.ndarray,
    y_valid: np.ndarray,
    fold_train: np.ndarray,
    fold_valid: np.ndarray,
) -> None:
    payload = {
        "base": np.asarray(views["s_marginal"], dtype=np.float32),
        "distribution": np.asarray(
            views["s_distribution"][:, views["s_marginal"].shape[1] :], dtype=np.float32
        ),
        "conditional_compact": np.asarray(
            views["s_conditional"][:, views["s_marginal"].shape[1] :], dtype=np.float32
        ),
        "conditional_pca": np.asarray(
            views["s_conditional_pca"][:, views["s_marginal"].shape[1] :], dtype=np.float32
        ),
        "ksvd_init": np.asarray(
            views["s_ksvd_init"][:, views["s_marginal"].shape[1] :], dtype=np.float32
        ),
        "ksvd_final": np.asarray(
            views["s_ksvd_final"][:, views["s_marginal"].shape[1] :], dtype=np.float32
        ),
        "conditional_shuffled_compact": np.asarray(
            null_views["conditional_shuffled"][:, views["s_marginal"].shape[1] :], dtype=np.float32
        ),
        "ksvd_final_shuffled": np.asarray(
            null_views["ksvd_final_shuffled"][:, views["s_marginal"].shape[1] :], dtype=np.float32
        ),
        "y_train": np.asarray(y_train, dtype=np.int64),
        "y_valid": np.asarray(y_valid, dtype=np.int64),
        "fold_train": np.asarray(fold_train, dtype=np.int64),
        "fold_valid": np.asarray(fold_valid, dtype=np.int64),
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("wb") as handle:
        np.savez(handle, signature=np.asarray([signature]), **payload)
    temporary.replace(path)


def _load_fold_checkpoint(path: Path, signature: str) -> dict[str, np.ndarray]:
    if not path.exists():
        raise FileNotFoundError(path)
    with np.load(path, allow_pickle=False) as archive:
        cached = str(np.asarray(archive["signature"]).reshape(-1)[0])
        if cached != signature:
            raise ValueError(f"center-token fold checkpoint signature mismatch: {path}")
        missing = [name for name in CHECKPOINT_ARRAYS if name not in archive]
        if missing:
            raise ValueError(f"center-token fold checkpoint is missing {missing}: {path}")
        return {name: np.asarray(archive[name]) for name in CHECKPOINT_ARRAYS}


def _components_to_views(components: Mapping[str, np.ndarray]) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray]]:
    base = np.asarray(components["base"], dtype=np.float32)
    blocks = {
        "distribution": np.asarray(components["distribution"], dtype=np.float32),
        "conditional_compact": np.asarray(components["conditional_compact"], dtype=np.float32),
        "ksvd_init": np.asarray(components["ksvd_init"], dtype=np.float32),
        "ksvd_final": np.asarray(components["ksvd_final"], dtype=np.float32),
    }
    views = _assemble_views(
        base,
        blocks,
        conditional_projection=np.asarray(components["conditional_pca"], dtype=np.float32),
    )
    null_views = {
        "conditional_shuffled": np.concatenate(
            [base, np.asarray(components["conditional_shuffled_compact"], dtype=np.float32)], axis=1
        ).astype(np.float32, copy=False),
        "ksvd_final_shuffled": np.concatenate(
            [base, np.asarray(components["ksvd_final_shuffled"], dtype=np.float32)], axis=1
        ).astype(np.float32, copy=False),
    }
    return views, null_views


def _checkpoint_paths(config: Mapping[str, Any], fold_count: int) -> list[Path]:
    directory = _resolve(config["output"]["checkpoint_dir"])
    return [directory / f"fold_{fold}.npz" for fold in range(int(fold_count))]


def _prepare_checkpoints(
    config_path: Path,
    config: Mapping[str, Any],
    bundle,
    labels: np.ndarray,
    train_indices: np.ndarray,
    frozen: Mapping[str, np.ndarray],
    cache: Mapping[str, np.ndarray],
    splits: Sequence[tuple[np.ndarray, np.ndarray]],
) -> tuple[list[dict[str, np.ndarray]], list[dict[str, Any]]]:
    paths = _checkpoint_paths(config, len(splits))
    components: list[dict[str, np.ndarray]] = []
    metadata: list[dict[str, Any]] = []
    for fold_id, (fold_train, fold_valid) in enumerate(splits):
        fold_train = np.asarray(fold_train, dtype=np.int64)
        fold_valid = np.asarray(fold_valid, dtype=np.int64)
        signature = _fold_signature(config_path, config, fold_id, fold_train, fold_valid)
        path = paths[fold_id]
        if path.exists():
            loaded = _load_fold_checkpoint(path, signature)
            components.append(loaded)
            metadata.append(
                {
                    "fold": int(fold_id),
                    "source": "reused",
                    "n_train": int(fold_train.size),
                    "n_valid": int(fold_valid.size),
                }
            )
            print(f"checkpoint fold {fold_id}: reused", flush=True)
            continue
        all_rows = np.concatenate([fold_train, fold_valid]).astype(np.int64, copy=False)
        blocks, block_meta = _fold_blocks(
            cache,
            fold_train,
            fold_valid,
            dictionary_config=config["dictionary"],
            representation=config["representation"],
            fold_seed=int(config["feature_seed"]) + 1009 * fold_id,
            conditional_rank=int(config["conditional_pca_rank"]),
        )
        raw_indices = train_indices[all_rows]
        base = _build_base(cache, frozen, raw_indices)
        views = _assemble_views(base, blocks, conditional_projection=blocks["conditional_pca"])
        null_views = {
            "conditional_shuffled": np.concatenate(
                [base, blocks["conditional_shuffled_compact"]], axis=1
            ).astype(np.float32, copy=False),
            "ksvd_final_shuffled": np.concatenate(
                [base, blocks["ksvd_final_shuffled"]], axis=1
            ).astype(np.float32, copy=False),
        }
        _save_fold_checkpoint(
            path,
            signature,
            views,
            null_views,
            labels[raw_indices[: fold_train.size]],
            labels[raw_indices[fold_train.size :]],
            fold_train,
            fold_valid,
        )
        with np.load(path, allow_pickle=False) as archive:
            loaded = {name: np.asarray(archive[name]) for name in CHECKPOINT_ARRAYS}
        components.append(loaded)
        metadata.append(
            {
                "fold": int(fold_id),
                "source": "built",
                "n_train": int(fold_train.size),
                "n_valid": int(fold_valid.size),
                **block_meta,
            }
        )
        print(f"checkpoint fold {fold_id}: built", flush=True)
        del blocks, views, null_views
        gc.collect()
    return components, metadata


def _screen_fixed(
    components: Sequence[Mapping[str, np.ndarray]],
    config: Mapping[str, Any],
    output_path: Path,
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    fixed = _fixed_classifier(config["classifier"])
    seeds = [int(value) for value in config["screen"]["seeds"]]
    if output_path.exists():
        existing = json.loads(output_path.read_text(encoding="utf-8"))
    else:
        existing = {"protocol_id": config["protocol_id"], "stage": "fixed_scaffold_screen", "views": {}, "null_views": {}}
    existing["fixed_classifier"] = fixed
    existing["screen_seeds"] = seeds
    existing["fold_metadata"] = list(metadata)
    labels_by_fold = [
        (
            np.asarray(component["y_train"], dtype=np.int64),
            np.asarray(component["y_valid"], dtype=np.int64),
        )
        for component in components
    ]
    for name in VIEW_NAMES:
        if name in existing["views"]:
            continue
        fold_matrices: list[tuple[np.ndarray, np.ndarray]] = []
        for component in components:
            views, _null_views = _components_to_views(component)
            n_train = int(np.asarray(component["y_train"]).size)
            fold_matrices.append((views[name][:n_train], views[name][n_train:]))
            del views
        score = _score_view(fold_matrices, labels_by_fold, fixed, seeds)
        existing["views"][name] = {
            "dimension": int(fold_matrices[0][0].shape[1]),
            "fixed_scaffold": score,
        }
        _write_json_atomic(output_path, existing)
        print(f"fixed screen {name}: {score['mean_auc']:.6f}", flush=True)
        del fold_matrices
        gc.collect()
    for name in NULL_NAMES:
        if name in existing["null_views"]:
            continue
        fold_matrices = []
        for component in components:
            _views, null_views = _components_to_views(component)
            n_train = int(np.asarray(component["y_train"]).size)
            fold_matrices.append((null_views[name][:n_train], null_views[name][n_train:]))
            del _views, null_views
        score = _score_view(fold_matrices, labels_by_fold, fixed, seeds)
        existing["null_views"][name] = {
            "dimension": int(fold_matrices[0][0].shape[1]),
            "fixed_scaffold": score,
        }
        _write_json_atomic(output_path, existing)
        print(f"fixed screen {name}: {score['mean_auc']:.6f}", flush=True)
        del fold_matrices
        gc.collect()
    return existing


def _load_candidate_matrices(
    components: Sequence[Mapping[str, np.ndarray]], name: str,
) -> tuple[list[tuple[np.ndarray, np.ndarray]], list[tuple[np.ndarray, np.ndarray]]]:
    matrices: list[tuple[np.ndarray, np.ndarray]] = []
    labels: list[tuple[np.ndarray, np.ndarray]] = []
    for component in components:
        views, null_views = _components_to_views(component)
        n_train = int(np.asarray(component["y_train"]).size)
        source = views[name] if name in views else null_views[name]
        matrices.append((source[:n_train], source[n_train:]))
        labels.append(
            (
                np.asarray(component["y_train"], dtype=np.int64),
                np.asarray(component["y_valid"], dtype=np.int64),
            )
        )
        del views, null_views
    return matrices, labels


def _run_screen(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    data = config["data"]
    train_indices, _valid_indices, _test_indices, fold_archive = _load_indices(
        _resolve(data["scaffold_folds"])
    )
    bundle = load_molhiv(root=_resolve(data["root"]), with_features=True)
    labels = np.asarray(bundle.y, dtype=np.int64)
    dev_indices = np.concatenate([train_indices, _valid_indices]).astype(np.int64, copy=False)
    with np.load(_resolve(data["frozen_features"]), allow_pickle=False) as archive:
        frozen = {name: np.asarray(archive[name]) for name in archive.files}
    cache = _build_token_cache(
        bundle,
        dev_indices,
        config["representation"],
        _resolve(data["dev_token_cache"]),
        shuffle_seed=int(config["feature_seed"]),
    )
    splits, split_meta = _official_train_scaffold_splits(fold_archive, train_indices)
    audit_n = min(int(config["audit"]["n_graphs"]), int(train_indices.size))
    audit = _invariance_audit(
        bundle,
        train_indices[:audit_n],
        config["representation"],
        int(config["audit"]["seed"]),
        float(config["audit"]["tolerance"]),
    )
    if not audit["pass"]:
        raise RuntimeError(f"center-token invariance audit failed: {audit}")
    components, checkpoint_meta = _prepare_checkpoints(
        config_path,
        config,
        bundle,
        labels,
        train_indices,
        frozen,
        cache,
        splits,
    )
    output_path = _resolve(config["output"]["screen_json"])
    result = _screen_fixed(components, config, output_path, checkpoint_meta)
    result.update(
        {
            "config_sha256": _sha256(config_path),
            "implementation_sha256": _sha256(
                REPO_ROOT / "tracks/ksvd/experiments/luyin16/center_token_route.py"
            ),
            "split_metadata": split_meta,
            "feature_seed": int(config["feature_seed"]),
            "checkpoint_dir": str(_resolve(config["output"]["checkpoint_dir"])),
            "official_test_evaluated": False,
            "invariance_audit": audit,
        }
    )
    _write_json_atomic(output_path, result)
    output_md = _resolve(config["output"]["screen_markdown"])
    lines = [
        "# Center-token checkpoint fixed screen",
        "",
        f"Protocol: `{config['protocol_id']}`",
        "",
        "All rows and transforms are official-train scaffold only.",
        "",
        "| view | dim | fixed scaffold mean | fold std |",
        "|---|---:|---:|---:|",
    ]
    for name in VIEW_NAMES:
        row = result["views"][name]
        score = row["fixed_scaffold"]
        lines.append(f"| `{name}` | {row['dimension']} | {score['mean_auc']:.6f} | {score['fold_std']:.6f} |")
    lines.extend(["", "## Null controls", ""])
    for name in NULL_NAMES:
        score = result["null_views"][name]["fixed_scaffold"]
        lines.append(f"- `{name}`: {score['mean_auc']:.6f} (fold std {score['fold_std']:.6f})")
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return result


def _params_for_view(
    name: str,
    screen: Mapping[str, Any],
    tuned: Mapping[str, Any],
    fixed: Mapping[str, Any],
) -> dict[str, Any]:
    if name in tuned:
        return dict(tuned[name]["best_params"])
    return dict(fixed)


def _run_tune_and_valid(config_path: Path, config: Mapping[str, Any]) -> dict[str, Any]:
    data = config["data"]
    train_indices, valid_indices, _test_indices, fold_archive = _load_indices(
        _resolve(data["scaffold_folds"])
    )
    bundle = load_molhiv(root=_resolve(data["root"]), with_features=True)
    labels = np.asarray(bundle.y, dtype=np.int64)
    dev_indices = np.concatenate([train_indices, valid_indices]).astype(np.int64, copy=False)
    with np.load(_resolve(data["frozen_features"]), allow_pickle=False) as archive:
        frozen = {name: np.asarray(archive[name]) for name in archive.files}
    cache = _build_token_cache(
        bundle,
        dev_indices,
        config["representation"],
        _resolve(data["dev_token_cache"]),
        shuffle_seed=int(config["feature_seed"]),
    )
    splits, split_meta = _official_train_scaffold_splits(fold_archive, train_indices)
    components, checkpoint_meta = _prepare_checkpoints(
        config_path,
        config,
        bundle,
        labels,
        train_indices,
        frozen,
        cache,
        splits,
    )
    screen_path = _resolve(config["output"]["screen_json"])
    if not screen_path.exists():
        raise FileNotFoundError(f"run --stage screen first: {screen_path}")
    screen = json.loads(screen_path.read_text(encoding="utf-8"))
    fixed = _fixed_classifier(config["classifier"])
    tune_path = _resolve(config["output"]["tuning_json"])
    if tune_path.exists():
        tuning_result = json.loads(tune_path.read_text(encoding="utf-8"))
    else:
        tuning_result = {
            "protocol_id": config["protocol_id"],
            "candidate_views": list(config["tuning"]["candidate_views"]),
            "views": {},
        }
    for candidate_id, name in enumerate(config["tuning"]["candidate_views"]):
        if name in tuning_result["views"]:
            print(f"tune {name}: reused", flush=True)
            continue
        matrices, fold_labels = _load_candidate_matrices(components, name)
        result = _tune_view(
            matrices,
            fold_labels,
            config["tuning"],
            config["classifier"],
            seed=int(config["tuning"]["seed"]) + 1009 * candidate_id,
        )
        result["dimension"] = int(matrices[0][0].shape[1])
        tuning_result["views"][name] = result
        _write_json_atomic(tune_path, tuning_result)
        print(f"tune {name}: {result['best_scaffold_auc']:.6f}", flush=True)
        del matrices, fold_labels
        gc.collect()

    # The fold checkpoints are intentionally large (the KSVD view is ~4.3k
    # columns).  They are needed only for the scaffold search above.  Keeping
    # all three decoded checkpoints, the OGB graph objects, and the full
    # train+valid feature blocks alive at once can exceed the worker's memory
    # budget and lead to an uninformative SIGKILL/exit 137.  Release the
    # checkpoint/search-only objects before doing the one-time official-valid
    # fit.  The token cache and frozen S are retained because they are the
    # only inputs needed by `_fold_blocks` and `_build_base` below.
    del components, checkpoint_meta, splits, split_meta, bundle
    gc.collect()

    train_rows = np.arange(train_indices.size, dtype=np.int64)
    valid_rows = np.arange(train_indices.size, dev_indices.size, dtype=np.int64)
    full_blocks, full_meta = _fold_blocks(
        cache,
        train_rows,
        valid_rows,
        dictionary_config=config["dictionary"],
        representation=config["representation"],
        fold_seed=int(config["feature_seed"]),
        conditional_rank=int(config["conditional_pca_rank"]),
    )
    # `conditional_full` and its shuffled counterpart are only intermediate
    # matrices used to fit the train-only PCA.  Keeping them after
    # `_fold_blocks` returns costs roughly another 3 GB for this split and is
    # unnecessary for any reported view.
    full_blocks.pop("conditional_full", None)
    full_blocks.pop("conditional_shuffled", None)
    dev_base = _build_base(cache, frozen, dev_indices)
    model_seeds = [int(value) for value in config["model_seeds"]]
    view_results: dict[str, Any] = {}
    tuned_views = {
        name: tuning_result["views"][name] for name in config["tuning"]["candidate_views"]
    }

    def make_view(name: str) -> np.ndarray:
        """Materialize one view at a time to cap the valid-fit peak."""
        if name == "s_marginal":
            return dev_base
        block_map = {
            "s_distribution": ("distribution",),
            "s_conditional": ("conditional_compact",),
            "s_conditional_pca": ("conditional_pca",),
            "s_ksvd_init": ("ksvd_init",),
            "s_ksvd_final": ("ksvd_final",),
            "s_center_token_all": (
                "distribution",
                "conditional_compact",
                "ksvd_final",
            ),
        }
        try:
            names = block_map[name]
        except KeyError as exc:
            raise KeyError(f"unknown center-token view: {name}") from exc
        return np.concatenate([dev_base, *(full_blocks[key] for key in names)], axis=1).astype(
            np.float32, copy=False
        )

    for name in VIEW_NAMES:
        params = _params_for_view(name, screen, tuned_views, fixed)
        matrix = make_view(name)
        report = _evaluate_frozen(
            matrix,
            labels[dev_indices],
            train_rows,
            valid_rows,
            params,
            model_seeds,
        )
        view_results[name] = {
            "dimension": int(matrix.shape[1]),
            "classifier_source": "tuned" if name in tuned_views else "fixed",
            "params": params,
            "official_valid": report,
        }
        print(f"official-valid {name}: {report['mean_auc']:.6f}; ensemble={report['seed_ensemble_auc']:.6f}", flush=True)
        del matrix
        gc.collect()
    null_results: dict[str, Any] = {}
    null_params = {
        "conditional_shuffled": _params_for_view("s_conditional", screen, tuned_views, fixed),
        "ksvd_final_shuffled": _params_for_view("s_ksvd_final", screen, tuned_views, fixed),
    }
    for name in NULL_NAMES:
        block_name = {
            "conditional_shuffled": "conditional_shuffled_compact",
            "ksvd_final_shuffled": "ksvd_final_shuffled",
        }[name]
        matrix = np.concatenate([dev_base, full_blocks[block_name]], axis=1).astype(
            np.float32, copy=False
        )
        report = _evaluate_frozen(
            matrix,
            labels[dev_indices],
            train_rows,
            valid_rows,
            null_params[name],
            model_seeds,
        )
        null_results[name] = {"params": null_params[name], "official_valid": report}
        print(f"official-valid {name}: {report['mean_auc']:.6f}; ensemble={report['seed_ensemble_auc']:.6f}", flush=True)
        del matrix
        gc.collect()

    # The primary/challenger roles were declared before validation.  A
    # separate fixed-screen winner is recorded only as an exploratory ranking;
    # it does not alter which models are reported on test.
    fixed_winner = max(
        VIEW_NAMES,
        key=lambda name: (
            float(screen["views"][name]["fixed_scaffold"]["mean_auc"]),
            int(name == "s_conditional"),
        ),
    )
    manifest = {
        "protocol_id": config["protocol_id"],
        "stage": "candidate_freeze_before_terminal_test",
        "config_path": str(config_path),
        "config_sha256": _sha256(config_path),
        "implementation_sha256": _sha256(
            REPO_ROOT / "tracks/ksvd/experiments/luyin16/center_token_route.py"
        ),
        "official_test_evaluated": False,
        "test_used_for_selection_or_tuning": False,
        "audit_boundary": {
            "fixed_screen_scope": "official-train scaffold folds only",
            "candidate_tuning_scope": "official-train scaffold folds only",
            "official_validation_used_once_for_frozen_reporting": True,
            "official_test_encoded": False,
        },
        "data": {
            "dataset": data["dataset"],
            "scaffold_folds": str(_resolve(data["scaffold_folds"])),
            "scaffold_folds_sha256": _sha256(_resolve(data["scaffold_folds"])),
            "frozen_features": str(_resolve(data["frozen_features"])),
            "full_features": str(_resolve(data["full_features"])),
            "dev_token_cache": str(_resolve(data["dev_token_cache"])),
            "test_token_cache": str(_resolve(data["test_token_cache"])),
            "n_train": int(train_indices.size),
            "n_valid": int(valid_indices.size),
            "split_metadata": split_meta,
        },
        "representation": dict(config["representation"]),
        "dictionary": dict(config["dictionary"]),
        "conditional_pca_rank": int(config["conditional_pca_rank"]),
        "feature_schema": {
            "token": "[node_role(64), edge_role(32), node_attribute(40), edge_attribute(13)]",
            "conditional": "role-conditioned attribute scalars/top-role strength/covariance; full table projected train-only",
            "distribution": "coordinate quantiles/tails plus centre norm and role entropy/Gini/rare/top-k",
            "ksvd": "train-token reservoir and INIT/FINAL sparse-code histogram/mean/std/covariance",
        },
        "model_seeds": model_seeds,
        "screen": screen,
        "tuning": tuning_result,
        "views": {
            name: {
                **view_results[name],
                "search": {
                    "best_params": view_results[name]["params"],
                    "best_scaffold_auc": (
                        tuned_views[name]["best_scaffold_auc"]
                        if name in tuned_views
                        else screen["views"][name]["fixed_scaffold"]["mean_auc"]
                    ),
                },
            }
            for name in VIEW_NAMES
        },
        "null_views": null_results,
        "checkpoint_metadata": checkpoint_meta,
        "full_fit_metadata": full_meta,
        "selection": {
            "primary_view": "s_conditional",
            "challenger_view": "s_ksvd_final",
            "exploratory_fixed_screen_winner": fixed_winner,
            "test_reports_all_views": True,
        },
        "runtime": {"python": platform.python_version(), "platform": platform.platform()},
    }
    manifest_path = _resolve(config["output"]["frozen_manifest_json"])
    if manifest_path.exists():
        raise FileExistsError(f"refusing to overwrite frozen manifest: {manifest_path}")
    _write_json_atomic(manifest_path, manifest)
    output_md = _resolve(config["output"]["frozen_manifest_markdown"])
    lines = [
        "# Center-token checkpoint candidate freeze",
        "",
        f"Protocol: `{config['protocol_id']}`",
        "",
        "Official test was not encoded or evaluated in this stage.",
        "",
        "| view | dim | classifier | scaffold score | official-valid mean | valid ensemble |",
        "|---|---:|---|---:|---:|---:|",
    ]
    for name in VIEW_NAMES:
        row = manifest["views"][name]
        source = row["classifier_source"]
        scaffold = row["search"]["best_scaffold_auc"]
        valid = row["official_valid"]
        lines.append(
            f"| `{name}` | {row['dimension']} | {source} | {scaffold:.6f} | {valid['mean_auc']:.6f} | {valid['seed_ensemble_auc']:.6f} |"
        )
    lines.extend(
        [
            "",
            "Primary/challenger were pre-registered as `s_conditional` / `s_ksvd_final`; all views remain in the report.",
            f"Exploratory fixed-screen winner: `{fixed_winner}`.",
        ]
    )
    output_md.parent.mkdir(parents=True, exist_ok=True)
    output_md.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    parser.add_argument("--stage", choices=("screen", "tune", "test"), required=True)
    args = parser.parse_args(argv)
    config_path = _resolve(args.config)
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    start = time.perf_counter()
    if args.stage == "screen":
        result = _run_screen(config_path, config)
    elif args.stage == "tune":
        result = _run_tune_and_valid(config_path, config)
    else:
        result = _run_test(config_path, config)
    print(
        json.dumps(
            {
                "protocol_id": result["protocol_id"],
                "stage": result["stage"],
                "elapsed_seconds": time.perf_counter() - start,
                "selected_view": result.get("selection", {}).get("primary_view"),
                "views": {
                    name: {
                        "valid": result["views"][name].get("official_valid", {}).get("mean_auc"),
                        "test_train_only": result["views"][name].get("train_only", {}).get("mean_auc"),
                    }
                    for name in result.get("views", {})
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

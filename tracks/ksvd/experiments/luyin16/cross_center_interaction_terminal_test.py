"""Controlled frozen official-test evaluation for the centre-interaction route.

The validation-search runner is deliberately not modified after its result was
frozen.  This separate entry point verifies the frozen hashes, encodes only
the official-test graphs, fits PCA on either official-train or train+valid,
and evaluates every pre-registered view with its already-frozen XGBoost
parameters.  No test result is used for selection or tuning.
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
from sklearn.metrics import roc_auc_score

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.cross_center_interaction_screen import (
    _build_cache,
    _fit_projection_model,
    _sha256,
    _transform_projection,
)
from tracks.ksvd.experiments.luyin16.cross_center_interaction_terminal import VIEW_NAMES
from tracks.ksvd.experiments.luyin16.mentor_concept_replication import (
    build_composition_features,
)
from tracks.ksvd.experiments.luyin16.task_aligned_interaction_screen import (
    _fit_xgb_predict,
)


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "tracks/ksvd/configs/luyin16/cross_center_interaction_terminal_test.yaml"
)


def _resolve(value: str | Path) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (REPO_ROOT / path).resolve()


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
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.asarray(archive[name]) for name in archive.files}


def _split_indices(
    archive: Mapping[str, np.ndarray],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    original = np.asarray(archive["original_indices"], dtype=np.int64)
    train = original[np.asarray(archive["official_train_indices"], dtype=np.int64)]
    valid = original[np.asarray(archive["official_valid_indices"], dtype=np.int64)]
    test = original[np.asarray(archive["official_test_indices"], dtype=np.int64)]
    parts = [train, valid, test]
    if any(np.unique(part).size != part.size for part in parts):
        raise RuntimeError("an official split contains duplicate dataset indices")
    if (
        np.intersect1d(train, valid).size
        or np.intersect1d(train, test).size
        or np.intersect1d(valid, test).size
    ):
        raise RuntimeError("official train/valid/test splits overlap")
    if np.unique(np.concatenate(parts)).size != original.size:
        raise RuntimeError("official splits do not cover the full dataset exactly once")
    return train, valid, test


def _rows_for_indices(
    available: np.ndarray,
    requested: np.ndarray,
    *,
    name: str,
) -> np.ndarray:
    mapping = {int(index): position for position, index in enumerate(available)}
    try:
        return np.asarray([mapping[int(index)] for index in requested], dtype=np.int64)
    except KeyError as exc:
        raise ValueError(f"{name} lacks dataset index {exc}") from exc


def _score_predictions(
    train_matrix: np.ndarray,
    train_labels: np.ndarray,
    test_matrix: np.ndarray,
    test_labels: np.ndarray,
    params: Mapping[str, Any],
    seeds: Sequence[int],
) -> tuple[dict[str, Any], list[np.ndarray]]:
    rows: list[dict[str, Any]] = []
    predictions: list[np.ndarray] = []
    y_train = np.asarray(train_labels, dtype=np.int64)
    y_test = np.asarray(test_labels, dtype=np.int64)
    for seed in seeds:
        current = dict(params)
        current["random_state"] = int(seed)
        prediction = _fit_xgb_predict(
            np.asarray(train_matrix, dtype=np.float32),
            y_train,
            np.asarray(test_matrix, dtype=np.float32),
            current,
        )
        predictions.append(prediction)
        rows.append(
            {
                "seed": int(seed),
                "fit_rows": int(y_train.size),
                "test_rows": int(y_test.size),
                "test_positive": int(y_test.sum()),
                "test_auc": float(roc_auc_score(y_test, prediction)),
            }
        )
    ensemble = np.mean(np.stack(predictions, axis=0), axis=0)
    values = [row["test_auc"] for row in rows]
    return (
        {
            "rows": rows,
            "mean_auc": float(np.mean(values)),
            "std_auc": float(np.std(values)),
            "seed_ensemble_auc": float(roc_auc_score(y_test, ensemble)),
        },
        predictions,
    )


def _fit_scope_projections(
    dev_raw: Mapping[str, np.ndarray],
    test_raw: Mapping[str, np.ndarray],
    scope_rows: np.ndarray,
    pca_rank: int,
) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    """Fit cross/binding PCA on one allowed scope and project dev/test."""
    dev_size = int(next(iter(dev_raw.values())).shape[0])
    scope = np.asarray(scope_rows, dtype=np.int64)
    if scope.size == 0 or np.any(scope < 0) or np.any(scope >= dev_size):
        raise ValueError("invalid PCA fit scope")
    projected_dev: dict[str, np.ndarray] = {}
    projected_test: dict[str, np.ndarray] = {}
    metadata: dict[str, Any] = {"fit_rows": int(scope.size)}
    for name in ("cross_cov", "binding"):
        model, fit_out, fit_meta = _fit_projection_model(
            np.asarray(dev_raw[name][scope], dtype=np.float32), pca_rank
        )
        all_dev = _transform_projection(model, np.asarray(dev_raw[name], dtype=np.float32))
        # Preserve fit_transform coordinates for the fit rows, matching the
        # validation runner's float32 convention.
        all_dev[scope] = fit_out
        projected_dev[name] = all_dev
        projected_test[name] = _transform_projection(
            model, np.asarray(test_raw[name], dtype=np.float32)
        )
        metadata[name] = fit_meta
    return (
        np.concatenate([projected_dev["cross_cov"], projected_dev["binding"]], axis=1),
        np.concatenate([projected_test["cross_cov"], projected_test["binding"]], axis=1),
        metadata,
    )


def _build_scope_views(
    s_dev: np.ndarray,
    s_test: np.ndarray,
    dev_raw: Mapping[str, np.ndarray],
    test_raw: Mapping[str, np.ndarray],
    scope_rows: np.ndarray,
    pca_rank: int,
) -> tuple[dict[str, np.ndarray], dict[str, np.ndarray], dict[str, Any]]:
    base_dev = np.concatenate(
        [
            np.asarray(s_dev, dtype=np.float32),
            np.asarray(dev_raw["marginal"], dtype=np.float32),
            np.asarray(dev_raw["context"], dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    base_test = np.concatenate(
        [
            np.asarray(s_test, dtype=np.float32),
            np.asarray(test_raw["marginal"], dtype=np.float32),
            np.asarray(test_raw["context"], dtype=np.float32),
        ],
        axis=1,
    ).astype(np.float32, copy=False)
    cross_binding_dev, cross_binding_test, projection_meta = _fit_scope_projections(
        dev_raw, test_raw, scope_rows, pca_rank
    )
    cross_dev = cross_binding_dev[:, :pca_rank]
    binding_dev = cross_binding_dev[:, pca_rank:]
    cross_test = cross_binding_test[:, :pca_rank]
    binding_test = cross_binding_test[:, pca_rank:]
    dev_views = {
        "s": np.asarray(s_dev, dtype=np.float32),
        "s_marginal": base_dev,
        "s_cross_cov": np.concatenate([base_dev, cross_dev], axis=1),
        "s_binding": np.concatenate([base_dev, binding_dev], axis=1),
        "s_both": np.concatenate([base_dev, cross_dev, binding_dev], axis=1),
    }
    test_views = {
        "s": np.asarray(s_test, dtype=np.float32),
        "s_marginal": base_test,
        "s_cross_cov": np.concatenate([base_test, cross_test], axis=1),
        "s_binding": np.concatenate([base_test, binding_test], axis=1),
        "s_both": np.concatenate([base_test, cross_test, binding_test], axis=1),
    }
    return dev_views, test_views, projection_meta


def _render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# MolHIV 中心级结构–属性交互：official-test 冻结评估",
        "",
        f"协议：`{result['protocol_id']}`",
        "",
        "这是在 valid 搜索和视图判断完成后进行的一次 controlled terminal test；"
        "test 结果没有用于选择视图、超参数或融合权重。仓库中更早路线已经查看过"
        "MolHIV test，因此不宣称 untouched-test。",
        "",
        f"数据：train/valid/test = {result['split_sizes']['train']}/"
        f"{result['split_sizes']['valid']}/{result['split_sizes']['test']}，"
        f"test positive = {result['split_sizes']['test_positive']}。",
        "",
        "`train_only` 是严格只用 official-train 拟合 PCA 和模型；"
        "`train_valid_refit` 是冻结后常规的 train+valid 重训，PCA 也只在该拟合范围内学习。",
        "",
    ]
    for scope, title in (
        ("train_only", "## Strict train-only fit"),
        ("train_valid_refit", "## Train+valid refit"),
    ):
        lines.extend(
            [
                title,
                "",
                "| view | valid tuned mean (reference) | test five-seed mean | "
                "test seed ensemble | test − valid |",
                "|---|---:|---:|---:|---:|",
            ]
        )
        for view in VIEW_NAMES:
            info = result["views"][view]
            valid_mean = float(info["valid_reference_mean_auc"])
            test_info = info[scope]
            lines.append(
                f"| `{view}` | {valid_mean:.6f} | {test_info['mean_auc']:.6f} | "
                f"{test_info['seed_ensemble_auc']:.6f} | "
                f"{test_info['mean_auc'] - valid_mean:+.6f} |"
            )
        lines.append("")
    control = result.get("pca_scope_control")
    if control is not None:
        lines.extend(
            [
                "## PCA-scope control",
                "",
                "固定 train-only PCA 坐标、但用 train+valid 标签重训分类器，"
                "用于区分加数据与坐标漂移：",
                "",
                "| view | fixed-train-PCA mean | seed ensemble | vs PCA-refit mean |",
                "|---|---:|---:|---:|",
            ]
        )
        for view in VIEW_NAMES:
            row = control[view]
            refit = result["views"][view]["train_valid_refit"]["mean_auc"]
            lines.append(
                f"| `{view}` | {row['mean_auc']:.6f} | {row['seed_ensemble_auc']:.6f} | "
                f"{row['mean_auc'] - refit:+.6f} |"
            )
        lines.append("")
    lines.extend(
        [
            "## Per-seed test AUC",
            "",
            "| view | train-only seeds 0…4 | train+valid seeds 0…4 |",
            "|---|---|---|",
        ]
    )
    for view in VIEW_NAMES:
        info = result["views"][view]
        left = "/".join(f"{row['test_auc']:.6f}" for row in info["train_only"]["rows"])
        right = "/".join(
            f"{row['test_auc']:.6f}" for row in info["train_valid_refit"]["rows"]
        )
        lines.append(f"| `{view}` | {left} | {right} |")
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "- 这里的主问题是 valid 上的增益能否迁移到 test；不能用 test 排名反向改写路线。",
            "- 五个视图都按各自 valid 搜索得到的参数评估；`S+both` 的 train-CV 选择身份被保留，"
            "但 test 只作冻结后的泛化检查。",
            "- 若不同视图在 test 上排序反转，应视为 scaffold shift / 小阳性集不确定性的证据，"
            "而不是继续用 test 调参。",
            "",
            "原始结果：[test_summary.json](cross_center_interaction_terminal/test_summary.json)",
        ]
    )
    return "\n".join(lines) + "\n"


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    if not isinstance(config, Mapping):
        raise ValueError("test configuration must be a mapping")
    source = config["source"]
    source_config = _resolve(source["config"])
    source_result_path = _resolve(source["result_json"])
    source_runner = _resolve(source["runner"])
    source_encoder = _resolve(source["interaction_encoder"])
    output_json = _resolve(config["output_json"])
    output_markdown = _resolve(config["output_markdown"])
    if output_json.exists():
        raise FileExistsError(
            f"controlled test result already exists; refusing repeated evaluation: {output_json}"
        )
    frozen_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    if frozen_result.get("official_test_evaluated"):
        raise RuntimeError("source validation result already reports test evaluation")
    expected_hashes = {
        "config_sha256": _sha256(source_config),
        "implementation_sha256": _sha256(source_runner),
        "interaction_encoder_sha256": _sha256(source_encoder),
    }
    for key, expected in expected_hashes.items():
        if frozen_result.get(key) != expected:
            raise RuntimeError(
                f"frozen validation result {key} mismatch: expected {expected}, "
                f"got {frozen_result.get(key)}"
            )

    data = config["data"]
    folds_path = _resolve(data["scaffold_folds"])
    frozen_features_path = _resolve(data["frozen_features"])
    dev_cache_path = _resolve(data["dev_interaction_cache"])
    test_cache_path = _resolve(data["test_interaction_cache"])
    folds = _load_npz(folds_path)
    train_indices, valid_indices, test_indices = _split_indices(folds)
    dev_indices = np.concatenate([train_indices, valid_indices]).astype(np.int64, copy=False)
    bundle = load_molhiv(root=_resolve(data["root"]), with_features=True)
    labels = np.asarray(bundle.y, dtype=np.int64)
    for split_name, expected in bundle.split.items():
        actual = {"train": train_indices, "valid": valid_indices, "test": test_indices}[split_name]
        if not np.array_equal(np.sort(np.asarray(expected, dtype=np.int64)), np.sort(actual)):
            raise RuntimeError(f"OGB split mismatch for {split_name}")
    start = time.perf_counter()

    frozen = _load_npz(frozen_features_path)
    frozen_indices = np.asarray(frozen["dataset_indices"], dtype=np.int64)
    if not np.array_equal(frozen_indices, dev_indices):
        raise RuntimeError("frozen S rows are not ordered official-train then official-valid")
    if int(np.asarray(frozen["train_count"]).reshape(-1)[0]) != train_indices.size:
        raise RuntimeError("frozen S train_count mismatch")

    dev_cache = _load_npz(dev_cache_path)
    dev_cache_indices = np.asarray(dev_cache["dataset_indices"], dtype=np.int64)
    dev_rows = _rows_for_indices(dev_cache_indices, dev_indices, name="dev interaction cache")
    test_cache = _build_cache(
        bundle,
        test_indices,
        frozen_result["representation"],
        shuffle_repeats=0,
        seed=int(yaml.safe_load(source_config.read_text(encoding="utf-8"))["feature_seed"]),
        cache_path=test_cache_path,
    )
    test_cache_indices = np.asarray(test_cache["dataset_indices"], dtype=np.int64)
    if not np.array_equal(test_cache_indices, test_indices):
        raise RuntimeError("test interaction cache row order mismatch")

    # Rebuild the label-free S_v1 encoder on all graphs and prove that the
    # train+valid rows exactly match the frozen source before using test rows.
    all_s, _, s_meta = build_composition_features(bundle)
    frozen_s = np.asarray(frozen["s"], dtype=np.float32)
    rebuilt_dev_s = np.asarray(all_s[dev_indices], dtype=np.float32)
    max_s_drift = float(np.max(np.abs(rebuilt_dev_s - frozen_s)))
    if max_s_drift > 1.0e-7:
        raise RuntimeError(f"S_v1 rebuild does not match frozen rows: max drift {max_s_drift}")
    test_s = np.asarray(all_s[test_indices], dtype=np.float32)
    del all_s
    gc.collect()

    dev_raw = {
        name: np.asarray(dev_cache[name][dev_rows], dtype=np.float32)
        for name in ("marginal", "cross_cov", "binding", "context")
    }
    test_raw = {
        name: np.asarray(test_cache[name], dtype=np.float32)
        for name in ("marginal", "cross_cov", "binding", "context")
    }
    dev_s = frozen_s
    dev_labels = labels[dev_indices]
    test_labels = labels[test_indices]
    if int(test_labels.sum()) != 130:
        raise RuntimeError("unexpected MolHIV official-test positive count")
    model_seeds = [int(value) for value in config.get("model_seeds", [0, 1, 2, 3, 4])]
    pca_rank = int(config["pca_rank"])
    tuning_source = yaml.safe_load(source_config.read_text(encoding="utf-8"))
    if pca_rank != int(tuning_source["pca_rank"]):
        raise RuntimeError("test PCA rank differs from frozen validation protocol")

    scope_rows = {
        "train_only": np.arange(train_indices.size, dtype=np.int64),
        "train_valid_refit": np.arange(dev_indices.size, dtype=np.int64),
    }
    view_results: dict[str, Any] = {}
    for scope, fit_rows in scope_rows.items():
        dev_views, test_views, projection_meta = _build_scope_views(
            dev_s,
            test_s,
            dev_raw,
            test_raw,
            fit_rows,
            pca_rank,
        )
        fit_labels = dev_labels[fit_rows]
        for view in VIEW_NAMES:
            if scope == "train_only":
                view_results[view] = {
                    "dimension": int(dev_views[view].shape[1]),
                    "best_params": frozen_result["views"][view]["search"]["best_params"],
                    "valid_reference_mean_auc": float(
                        frozen_result["views"][view]["tuned_official_valid"]["mean_auc"]
                    ),
                    "valid_reference_ensemble_auc": float(
                        frozen_result["views"][view]["tuned_official_valid"]["seed_ensemble_auc"]
                    ),
                    "projection": {},
                }
            params = frozen_result["views"][view]["search"]["best_params"]
            scored, _predictions = _score_predictions(
                dev_views[view][fit_rows],
                fit_labels,
                test_views[view],
                test_labels,
                params,
                model_seeds,
            )
            view_results[view][scope] = scored
            view_results[view]["projection"][scope] = projection_meta
            print(
                f"{scope} {view}: test-mean={scored['mean_auc']:.6f}; "
                f"ensemble={scored['seed_ensemble_auc']:.6f}",
                flush=True,
            )
        del dev_views, test_views
        gc.collect()

    result: dict[str, Any] = {
        "protocol_id": str(config["protocol_id"]),
        "stage": "controlled_frozen_terminal_test",
        "source_validation_result": str(source_result_path),
        "source_validation_result_sha256": _sha256(source_result_path),
        "source_config": str(source_config),
        "source_runner": str(source_runner),
        "source_interaction_encoder": str(source_encoder),
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
        "feature_audit": {
            "frozen_features": str(frozen_features_path),
            "frozen_features_sha256": _sha256(frozen_features_path),
            "dev_interaction_cache": str(dev_cache_path),
            "dev_interaction_cache_sha256": _sha256(dev_cache_path),
            "test_interaction_cache": str(test_cache_path),
            "test_interaction_cache_sha256": _sha256(test_cache_path),
            "s_schema": s_meta,
            "s_rebuild_max_abs_drift_on_dev": max_s_drift,
            "pca_rank": pca_rank,
            "pca_fit_scopes": {
                name: int(rows.size) for name, rows in scope_rows.items()
            },
        },
        "model_seeds": model_seeds,
        "views": view_results,
        "runtime": {
            "seconds": float(time.perf_counter() - start),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    _write_json_atomic(output_json, result)
    output_markdown.parent.mkdir(parents=True, exist_ok=True)
    output_markdown.write_text(_render_markdown(result), encoding="utf-8")
    print(f"controlled test ledger written: {output_json}", flush=True)
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
                "test": {
                    view: {
                        "train_only_mean": result["views"][view]["train_only"]["mean_auc"],
                        "train_only_ensemble": result["views"][view]["train_only"]["seed_ensemble_auc"],
                        "train_valid_refit_mean": result["views"][view]["train_valid_refit"]["mean_auc"],
                        "train_valid_refit_ensemble": result["views"][view]["train_valid_refit"][
                            "seed_ensemble_auc"
                        ],
                    }
                    for view in VIEW_NAMES
                },
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

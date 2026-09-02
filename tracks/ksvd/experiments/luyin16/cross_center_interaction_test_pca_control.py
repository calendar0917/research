"""PCA-scope sensitivity control for the frozen centre-interaction test.

This control keeps the PCA coordinate system fitted on official-train fixed,
then refits XGBoost on official-train+valid labels.  It separates the effect
of adding supervised rows from the effect of changing the low-rank coordinate
system.  No parameter or view is selected from official-test.
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

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.cross_center_interaction_terminal import VIEW_NAMES
from tracks.ksvd.experiments.luyin16.cross_center_interaction_terminal_test import (
    _build_scope_views,
    _load_npz,
    _resolve,
    _rows_for_indices,
    _score_predictions,
    _split_indices,
)
from tracks.ksvd.experiments.luyin16.mentor_concept_replication import (
    build_composition_features,
)
from tracks.ksvd.experiments.luyin16.cross_center_interaction_screen import _sha256


REPO_ROOT = Path(__file__).resolve().parents[4]
DEFAULT_CONFIG = (
    REPO_ROOT
    / "tracks/ksvd/configs/luyin16/cross_center_interaction_terminal_test.yaml"
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
        json.dumps(_jsonable(payload), ensure_ascii=False, indent=2, sort_keys=True)
        + "\n",
        encoding="utf-8",
    )
    temporary.replace(path)


def _render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        "# MolHIV 中心级交互：固定 train-only PCA 的 test 控制",
        "",
        f"协议：`{result['protocol_id']}`",
        "",
        "这是对正式 frozen test 的敏感性控制：PCA 只在 official-train 拟合并固定，"
        "分类器则用 official-train+valid 标签重训。它不改变任何视图或 XGBoost 参数，"
        "也不使用 test 结果做选择。",
        "",
        "| view | fixed-train-PCA + train+valid mean | seed ensemble | "
        "relative to PCA-refit mean |",
        "|---|---:|---:|---:|",
    ]
    formal = result["formal_test_reference"]
    for view in VIEW_NAMES:
        row = result["views"][view]
        ref = formal[view]["train_valid_refit"]["mean_auc"]
        lines.append(
            f"| `{view}` | {row['mean_auc']:.6f} | {row['seed_ensemble_auc']:.6f} | "
            f"{row['mean_auc'] - ref:+.6f} |"
        )
    lines.extend(
        [
            "",
            "解释：若该控制高于 PCA-refit 结果，差异来自交互坐标系改变；若仍低，"
            "才更可能是加入 valid 后的模型拟合/分布问题。这里的结果只用于诊断，"
            "不允许据此回头调 test。",
            "",
            "原始结果：[test_pca_control.json](cross_center_interaction_terminal/test_pca_control.json)",
        ]
    )
    return "\n".join(lines) + "\n"


def run(config_path: Path = DEFAULT_CONFIG) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    source = config["source"]
    formal_path = _resolve(config["output_json"])
    output_json = _resolve(config["pca_control_output_json"])
    output_markdown = _resolve(config["pca_control_output_markdown"])
    if output_json.exists():
        raise FileExistsError(f"PCA control result already exists: {output_json}")
    formal = json.loads(formal_path.read_text(encoding="utf-8"))
    if formal.get("official_test_evaluated", True) is not True:
        raise RuntimeError("formal frozen test result is missing its test audit")
    source_result_path = _resolve(source["result_json"])
    frozen_result = json.loads(source_result_path.read_text(encoding="utf-8"))
    folds = _load_npz(_resolve(config["data"]["scaffold_folds"]))
    train, valid, test = _split_indices(folds)
    dev = np.concatenate([train, valid]).astype(np.int64, copy=False)
    bundle = load_molhiv(root=_resolve(config["data"]["root"]), with_features=True)
    frozen = _load_npz(_resolve(config["data"]["frozen_features"]))
    dev_cache = _load_npz(_resolve(config["data"]["dev_interaction_cache"]))
    test_cache = _load_npz(_resolve(config["data"]["test_interaction_cache"]))
    dev_rows = _rows_for_indices(
        np.asarray(dev_cache["dataset_indices"], dtype=np.int64), dev, name="dev cache"
    )
    all_s, _, s_meta = build_composition_features(bundle)
    dev_s = np.asarray(frozen["s"], dtype=np.float32)
    test_s = np.asarray(all_s[test], dtype=np.float32)
    if float(np.max(np.abs(np.asarray(all_s[dev], dtype=np.float32) - dev_s))) > 1.0e-7:
        raise RuntimeError("rebuilt S does not match frozen train+valid S")
    dev_raw = {
        name: np.asarray(dev_cache[name][dev_rows], dtype=np.float32)
        for name in ("marginal", "cross_cov", "binding", "context")
    }
    test_raw = {
        name: np.asarray(test_cache[name], dtype=np.float32)
        for name in ("marginal", "cross_cov", "binding", "context")
    }
    labels = np.asarray(bundle.y, dtype=np.int64)
    dev_labels = labels[dev]
    test_labels = labels[test]
    train_rows = np.arange(train.size, dtype=np.int64)
    dev_views, test_views, projection_meta = _build_scope_views(
        dev_s,
        test_s,
        dev_raw,
        test_raw,
        train_rows,
        int(config["pca_rank"]),
    )
    result_views: dict[str, Any] = {}
    start = time.perf_counter()
    for view in VIEW_NAMES:
        scored, _ = _score_predictions(
            dev_views[view],
            dev_labels,
            test_views[view],
            test_labels,
            frozen_result["views"][view]["search"]["best_params"],
            [int(seed) for seed in config["model_seeds"]],
        )
        result_views[view] = scored
        print(
            f"fixed-train-PCA train+valid {view}: mean={scored['mean_auc']:.6f}; "
            f"ensemble={scored['seed_ensemble_auc']:.6f}",
            flush=True,
        )
    result: dict[str, Any] = {
        "protocol_id": "luyin16-molhiv-cross-center-interaction-test-pca-scope-control-v1",
        "stage": "controlled_frozen_terminal_test_pca_sensitivity",
        "formal_test_result": str(formal_path),
        "formal_test_result_sha256": _sha256(formal_path),
        "source_validation_result": str(source_result_path),
        "audit_boundary": {
            "official_test_evaluated": True,
            "test_used_for_selection_or_tuning": False,
            "pca_fit_scope": "official-train only",
            "classifier_fit_scope": "official-train plus official-valid",
            "historical_test_already_seen_by_older_routes": True,
            "untouched_test_claim": False,
        },
        "split_sizes": {
            "train": int(train.size),
            "valid": int(valid.size),
            "test": int(test.size),
            "test_positive": int(test_labels.sum()),
        },
        "feature_audit": {
            "pca_rank": int(config["pca_rank"]),
            "pca_fit_rows": int(train.size),
            "s_schema": s_meta,
            "s_rebuild_max_abs_drift_on_dev": float(
                np.max(np.abs(np.asarray(all_s[dev], dtype=np.float32) - dev_s))
            ),
            "projection": projection_meta,
        },
        "model_seeds": [int(seed) for seed in config["model_seeds"]],
        "formal_test_reference": formal["views"],
        "views": result_views,
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
    run(_resolve(args.config))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

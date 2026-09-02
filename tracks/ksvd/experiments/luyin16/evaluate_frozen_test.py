"""Evaluate a frozen luyin16 feature/classifier protocol on official test.

This entry point is intentionally separate from the exploratory tuner.  It
loads the already-frozen Optuna parameters, never searches on test, and reports
both the train-only holdout score and the standard train+valid refit score.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier


def _view_matrix(payload: Mapping[str, np.ndarray], view: str) -> np.ndarray:
    if view == "s":
        return np.asarray(payload["composition"], dtype=np.float32)
    if view == "r_raw":
        return np.asarray(payload["r_raw"], dtype=np.float32)
    if view == "r_final":
        return np.asarray(payload["r_final"], dtype=np.float32)
    if view == "s_r_raw":
        return np.concatenate([payload["composition"], payload["r_raw"]], axis=1).astype(np.float32)
    if view == "s_r_final":
        return np.concatenate([payload["composition"], payload["r_final"]], axis=1).astype(np.float32)
    raise ValueError(f"unsupported frozen test view: {view}")


def _fit_and_score(
    x: np.ndarray,
    y: np.ndarray,
    train: np.ndarray,
    test: np.ndarray,
    params: Mapping[str, Any],
    seed: int,
) -> dict[str, Any]:
    y_train = y[train].astype(np.int64)
    y_test = y[test].astype(np.int64)
    class_weight = float(np.sum(y_train == 0) / max(1, np.sum(y_train == 1)))
    fitted = dict(params)
    fitted["random_state"] = int(seed)
    model = XGBClassifier(
        **fitted,
        objective="binary:logistic",
        eval_metric="auc",
        scale_pos_weight=class_weight,
        n_jobs=-1,
        tree_method="hist",
    )
    model.fit(x[train], y_train)
    prediction = model.predict_proba(x[test])[:, 1]
    return {
        "seed": int(seed),
        "fit_rows": int(train.size),
        "test_rows": int(test.size),
        "test_positive": int(y_test.sum()),
        "test_auc": float(roc_auc_score(y_test, prediction)),
    }


def evaluate(
    full_features: Path,
    search_result: Path,
    views: Sequence[str],
    seeds: Sequence[int],
) -> dict[str, Any]:
    payload = np.load(full_features, allow_pickle=False)
    labels = np.asarray(payload["labels"], dtype=np.int64)
    train = np.asarray(payload["train_indices"], dtype=np.int64)
    valid = np.asarray(payload["valid_indices"], dtype=np.int64)
    test = np.asarray(payload["test_indices"], dtype=np.int64)
    search = json.loads(search_result.read_text(encoding="utf-8"))
    search_views = search["views"]
    result: dict[str, Any] = {
        "protocol_id": "mentor-r2-atom-chem-k64-s8-frozen-test-v1",
        "full_feature_file": str(full_features.resolve()),
        "search_result": str(search_result.resolve()),
        "official_test_evaluated": True,
        "selection_rule": "view and XGBoost parameters were frozen before test evaluation",
        "split_sizes": {
            "train": int(train.size),
            "valid": int(valid.size),
            "test": int(test.size),
        },
        "views": {},
    }
    for view in views:
        if view not in search_views:
            raise ValueError(f"{view!r} is absent from the frozen search result")
        matrix = _view_matrix(payload, view)
        params = dict(search_views[view]["best_params"])
        train_only_rows = [
            _fit_and_score(matrix, labels, train, test, params, seed) for seed in seeds
        ]
        refit_indices = np.concatenate([train, valid]).astype(np.int64)
        train_valid_rows = [
            _fit_and_score(matrix, labels, refit_indices, test, params, seed)
            for seed in seeds
        ]
        result["views"][view] = {
            "dimension": int(matrix.shape[1]),
            "best_params": params,
            "train_only": train_only_rows,
            "train_only_mean": float(np.mean([row["test_auc"] for row in train_only_rows])),
            "train_valid_refit": train_valid_rows,
            "train_valid_refit_mean": float(
                np.mean([row["test_auc"] for row in train_valid_rows])
            ),
        }
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full-features", type=Path, required=True)
    parser.add_argument("--search-result", type=Path, required=True)
    parser.add_argument("--result", type=Path, required=True)
    parser.add_argument("--views", default="s,s_r_raw,s_r_final")
    parser.add_argument("--seeds", type=int, nargs="+", default=[0, 1, 2, 3, 4])
    args = parser.parse_args(argv)
    result = evaluate(
        args.full_features.expanduser().resolve(),
        args.search_result.expanduser().resolve(),
        [name.strip() for name in args.views.split(",") if name.strip()],
        args.seeds,
    )
    args.result.parent.mkdir(parents=True, exist_ok=True)
    args.result.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                view: {
                    "train_only_mean": info["train_only_mean"],
                    "train_valid_refit_mean": info["train_valid_refit_mean"],
                }
                for view, info in result["views"].items()
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Evaluation of frozen checkpoints on Validation420 and Test420.

Part A evaluates the historical seed-0 BASE / CONTEXT checkpoints.  The final
``evaluate-all`` pass evaluates all five frozen checkpoints (seed 0, seed 1 and
ROLEBLIND_s0) after the new runs have finished.

Test access is **EXPOSED_TEST_EXPLORATORY**: Test420 was already used by earlier
rounds on this split and these numbers feed the next research decision.  Test
labels come from the SHA256-verified raw CSV and must reproduce the historical
one-shot Test label vector hash.  Inference runs in ``model.eval()`` under
``torch.no_grad()``; weights are compared bitwise before/after.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import m2_common as C
import m2_data as MD
import m2_models as M

import r42_metrics as MET  # noqa: E402


# ---------------------------------------------------------------------------
# model registry
# ---------------------------------------------------------------------------


def model_registry() -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for key, entry in C.HISTORICAL_CHECKPOINTS.items():
        registry[key] = {
            "key": key,
            "family": entry["family"],
            "seed": int(entry["seed"]),
            "checkpoint": entry["path"],
            "expected_sha256": entry["sha256"],
            "train_json": entry["train_json"],
            "historical": True,
            "seed0_of_pair": True,
        }
    for arm_key, arm in C.ARM_SPECS.items():
        run_dir = C.RUNS / arm["run_key"]
        train_json = run_dir / "train.json"
        if not train_json.exists():
            continue
        payload = C.read_json(train_json)
        registry[arm_key] = {
            "key": arm_key,
            "family": arm["family"],
            "seed": int(arm["seed"]),
            "checkpoint": run_dir / "checkpoint.pt",
            "expected_sha256": payload["checkpoint_sha256"],
            "train_json": train_json,
            "historical": False,
            "seed0_of_pair": int(arm["seed"]) == 0,
        }
    return registry


# ---------------------------------------------------------------------------
# metrics
# ---------------------------------------------------------------------------


def metric_block(pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    pred = np.asarray(pred, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    err = pred - y
    sse = float(np.sum(err * err))
    return {
        "n": int(y.size),
        "rmse": float(np.sqrt(np.mean(err * err))),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(MET.r2(pred, y)),
        "sse": sse,
        "max_abs_error": float(np.abs(err).max()),
        "mean_abs_error": float(np.abs(err).mean()),
    }


# ---------------------------------------------------------------------------
# paired bootstrap
# ---------------------------------------------------------------------------


def paired_bootstrap(
    pred_ref: np.ndarray,
    pred_other: np.ndarray,
    y: np.ndarray,
    *,
    n_resamples: int = C.BOOTSTRAP["n_resamples"],
    seed: int = C.BOOTSTRAP["seed"],
    ci_level: float = C.BOOTSTRAP["ci_level"],
) -> dict[str, Any]:
    """Paired molecule-level percentile bootstrap (descriptive only).

    ``gain_rmse = RMSE(ref) - RMSE(other)``: positive means ``other`` is better.
    ``mae_change = MAE(other) - MAE(ref)``: negative means ``other`` is better.
    ``sse_gain = SSE(ref) - SSE(other)``: positive means ``other`` is better.
    """
    ref = np.asarray(pred_ref, dtype=np.float64)
    other = np.asarray(pred_other, dtype=np.float64)
    target = np.asarray(y, dtype=np.float64)
    n = int(target.size)
    if not (ref.size == other.size == target.size == n):
        raise ValueError("paired bootstrap requires equal-length vectors")
    rng = np.random.default_rng(int(seed))
    alpha = (1.0 - float(ci_level)) / 2.0
    q = [alpha, 0.5, 1.0 - alpha]
    gain_rmse = np.empty(n_resamples, dtype=np.float64)
    mae_change = np.empty(n_resamples, dtype=np.float64)
    sse_gain = np.empty(n_resamples, dtype=np.float64)
    mean_abs_pred_diff = np.empty(n_resamples, dtype=np.float64)
    n_other_better = np.empty(n_resamples, dtype=np.int64)
    for i in range(int(n_resamples)):
        idx = rng.integers(0, n, size=n)
        err_ref = ref[idx] - target[idx]
        err_other = other[idx] - target[idx]
        sse_ref = float(np.sum(err_ref * err_ref))
        sse_other = float(np.sum(err_other * err_other))
        gain_rmse[i] = np.sqrt(sse_ref / n) - np.sqrt(sse_other / n)
        mae_change[i] = float(np.mean(np.abs(err_other)) - np.mean(np.abs(err_ref)))
        sse_gain[i] = sse_ref - sse_other
        mean_abs_pred_diff[i] = float(np.mean(np.abs(other[idx] - ref[idx])))
        n_other_better[i] = int(np.sum(np.abs(err_other) < np.abs(err_ref)))
    err_ref = ref - target
    err_other = other - target
    observed = {
        "rmse_ref": float(np.sqrt(np.mean(err_ref * err_ref))),
        "rmse_other": float(np.sqrt(np.mean(err_other * err_other))),
        "mae_ref": float(np.mean(np.abs(err_ref))),
        "mae_other": float(np.mean(np.abs(err_other))),
        "sse_ref": float(np.sum(err_ref * err_ref)),
        "sse_other": float(np.sum(err_other * err_other)),
        "gain_rmse": float(np.sqrt(np.mean(err_ref * err_ref)) - np.sqrt(np.mean(err_other * err_other))),
        "mae_change": float(np.mean(np.abs(err_other)) - np.mean(np.abs(err_ref))),
        "sse_gain": float(np.sum(err_ref * err_ref) - np.sum(err_other * err_other)),
        "mean_abs_prediction_difference": float(np.mean(np.abs(other - ref))),
        "max_abs_prediction_difference": float(np.max(np.abs(other - ref))),
        "n_other_better_abs": int(np.sum(np.abs(err_other) < np.abs(err_ref))),
        "n_ref_better_abs": int(np.sum(np.abs(err_other) > np.abs(err_ref))),
        "n_tie_abs": int(np.sum(np.abs(err_other) == np.abs(err_ref))),
    }
    interval = {
        "gain_rmse": [float(x) for x in np.quantile(gain_rmse, q)],
        "mae_change": [float(x) for x in np.quantile(mae_change, q)],
        "sse_gain": [float(x) for x in np.quantile(sse_gain, q)],
        "mean_abs_prediction_difference": [float(x) for x in np.quantile(mean_abs_pred_diff, q)],
        "n_other_better_abs": [float(x) for x in np.quantile(n_other_better, q)],
    }
    return {
        "n": n,
        "n_resamples": int(n_resamples),
        "seed": int(seed),
        "ci_level": float(ci_level),
        "method": C.BOOTSTRAP["method"],
        "quantiles": {"lower": float(q[0]), "median": float(q[1]), "upper": float(q[2])},
        "observed": observed,
        "interval": interval,
        "sign_rule": (
            "gain_rmse > 0 and mae_change < 0 and sse_gain > 0 all mean 'other' is better; "
            "interval columns are [lower, median, upper] percentiles"
        ),
    }


# ---------------------------------------------------------------------------
# evaluation
# ---------------------------------------------------------------------------


def _save_predictions(path: Path, rows: np.ndarray, y: np.ndarray, pred: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, rows=np.asarray(rows, dtype=np.int64), y_true=np.asarray(y, dtype=np.float64), pred=np.asarray(pred, dtype=np.float64))


def evaluate_model(
    key: str,
    entry: Mapping[str, Any],
    views: Mapping[str, Any],
    *,
    device: Any = "cpu",
    write_predictions: bool = True,
) -> dict[str, Any]:
    spec = views["spec"]
    norm = views["train_norm"]
    model, _meta = M.build_family(entry["family"], spec, norm, int(entry["seed"]), device)
    payload = M.load_checkpoint(
        model,
        entry["checkpoint"],
        expected_sha256=entry["expected_sha256"],
        expected_family=entry["family"],
        expected_seed=int(entry["seed"]),
    )
    recomputed_soup_hash = C.state_dict_fingerprint(payload["state_dict"])
    if payload.get("soup_state_hash") is not None and recomputed_soup_hash != payload["soup_state_hash"]:
        raise PermissionError(
            f"{key}: loaded state hash {recomputed_soup_hash} != stored soup hash {payload['soup_state_hash']}"
        )

    rows_valid = np.asarray(views["split"]["rows"]["valid"], dtype=np.int64)
    rows_test = np.asarray(views["split"]["rows"]["test"], dtype=np.int64)
    y_valid = np.asarray(views["labels"]["y_valid"], dtype=np.float64)
    y_test = np.asarray(views["test"]["y_test"], dtype=np.float64)
    target_norm = views["target_norm"]
    pred_valid = MD.denormalise(M.predict(model, views["valid_cache"], device), target_norm)
    freeze_valid = M.freeze_weights_audit(model, views["valid_cache"], device)
    pred_test = MD.denormalise(M.predict(model, views["test_cache"], device), target_norm)
    freeze_test = M.freeze_weights_audit(model, views["test_cache"], device)
    if pred_valid.size != rows_valid.size or pred_test.size != rows_test.size:
        raise ValueError(f"{key}: prediction size mismatch")
    if not (np.isfinite(pred_valid).all() and np.isfinite(pred_test).all()):
        raise ValueError(f"{key}: non-finite predictions")

    metrics = {
        "key": key,
        "family": entry["family"],
        "seed": int(entry["seed"]),
        "historical": bool(entry["historical"]),
        "checkpoint": C.track_rel(entry["checkpoint"]),
        "checkpoint_sha256": C.sha256_file(entry["checkpoint"]),
        "soup_state_hash": payload.get("soup_state_hash"),
        "recomputed_state_hash": recomputed_soup_hash,
        "valid": metric_block(pred_valid, y_valid),
        "test": metric_block(pred_test, y_test),
        "freeze_weights_audit": {"valid420": freeze_valid, "test420": freeze_test},
    }
    if write_predictions:
        _save_predictions(C.PREDICTIONS / f"{key}_valid420.npz", rows_valid, y_valid, pred_valid)
        _save_predictions(C.PREDICTIONS / f"{key}_test420.npz", rows_test, y_test, pred_test)
    # Reference cross-checks for the historical seed-0 checkpoints.
    if entry["historical"]:
        ref = C.read_json(entry["train_json"])
        metrics["reference_check"] = {
            "v1_valid_rmse": float(ref["metrics"]["valid"]["rmse"]),
            "recomputed_valid_rmse": float(metrics["valid"]["rmse"]),
            "abs_diff": float(abs(ref["metrics"]["valid"]["rmse"] - metrics["valid"]["rmse"])),
            "v1_checkpoint_sha256": ref["checkpoint_sha256"],
            "checkpoint_sha256": metrics["checkpoint_sha256"],
        }
        if key in C.HISTORICAL_CHECKPOINTS:
            valid_npz = C.V1_RUNS / key / "valid_predictions.npz"
            if valid_npz.exists():
                with np.load(valid_npz) as archive:
                    pred_ref = np.asarray(archive["pred"], dtype=np.float64)
                    rows_ref = np.asarray(archive["rows"], dtype=np.int64)
                if not np.array_equal(rows_ref, rows_valid):
                    raise PermissionError(f"{key}: historical valid rows drifted")
                metrics["reference_check"]["v1_valid_predictions_max_abs_diff"] = float(
                    np.max(np.abs(pred_ref - pred_valid))
                )
                metrics["reference_check"]["v1_valid_predictions_mean_abs_diff"] = float(
                    np.mean(np.abs(pred_ref - pred_valid))
                )
    metrics["predictions_saved"] = bool(write_predictions)
    return {
        "key": key,
        "metrics": metrics,
        "rows_valid": rows_valid,
        "rows_test": rows_test,
        "y_valid": y_valid,
        "y_test": y_test,
        "pred_valid": pred_valid,
        "pred_test": pred_test,
        "entry": dict(entry),
    }


def evaluate_set(
    keys: Sequence[str],
    *,
    out_dir: Path,
    device: Any = "cpu",
    bootstrap_pairs: Sequence[tuple[str, str]],
) -> dict[str, Any]:
    registry = model_registry()
    missing = [key for key in keys if key not in registry]
    if missing:
        raise KeyError(f"unknown evaluation keys: {missing}")
    views = MD.build_views(include_test=True)
    results: dict[str, Any] = {}
    for key in keys:
        C.log(f"[eval] {key} ({registry[key]['family']}, seed {registry[key]['seed']})")
        results[key] = evaluate_model(key, registry[key], views, device=device, write_predictions=True)

    payload: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "device": str(device),
        "scope": "EXPOSED_TEST_EXPLORATORY",
        "test_access": "EXPOSED_TEST_EXPLORATORY",
        "test_labels_source": dict(views["test"]["audit"]),
        "models": {key: results[key]["metrics"] for key in keys},
        "bootstrap": {},
        "pairs": {},
        "environment": C.env_info(),
    }

    for ref_key, other_key in bootstrap_pairs:
        if ref_key not in results or other_key not in results:
            continue
        pair = f"{ref_key}__vs__{other_key}"
        payload["pairs"][pair] = {}
        payload["bootstrap"][pair] = {}
        for split in ("valid", "test"):
            ref_pred = results[ref_key][f"pred_{split}"]
            other_pred = results[other_key][f"pred_{split}"]
            y = results[ref_key][f"y_{split}"]
            if split == "test" and not np.array_equal(y, results[other_key]["y_test"]):
                raise ValueError(f"{pair}: Test labels differ between models")
            boot = paired_bootstrap(ref_pred, other_pred, y)
            payload["bootstrap"][pair][f"{split}420"] = boot
            payload["pairs"][pair][f"{split}420"] = {
                "rmse_ref": boot["observed"]["rmse_ref"],
                "rmse_other": boot["observed"]["rmse_other"],
                "gain_rmse": boot["observed"]["gain_rmse"],
                "mae_ref": boot["observed"]["mae_ref"],
                "mae_other": boot["observed"]["mae_other"],
                "mae_change": boot["observed"]["mae_change"],
                "sse_ref": boot["observed"]["sse_ref"],
                "sse_other": boot["observed"]["sse_other"],
                "sse_gain": boot["observed"]["sse_gain"],
                "mean_abs_prediction_difference": boot["observed"]["mean_abs_prediction_difference"],
                "max_abs_prediction_difference": boot["observed"]["max_abs_prediction_difference"],
                "n_other_better_abs": boot["observed"]["n_other_better_abs"],
                "n_ref_better_abs": boot["observed"]["n_ref_better_abs"],
                "n_tie_abs": boot["observed"]["n_tie_abs"],
            }

    # Long-form metrics table.
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "metrics_by_model_split.csv", "w", newline="", encoding="utf-8") as fh:
        fields = ["key", "family", "seed", "split", "n", "rmse", "mae", "r2", "sse", "max_abs_error", "mean_abs_error"]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key in keys:
            metrics = payload["models"][key]
            for split in ("valid", "test"):
                row = {"key": key, "family": metrics["family"], "seed": metrics["seed"], "split": split}
                row.update({name: metrics[split][name] for name in ("n", "rmse", "mae", "r2", "sse", "max_abs_error", "mean_abs_error")})
                writer.writerow(row)

    # Per-molecule tables with every model side by side.
    for split in ("valid", "test"):
        rows = results[keys[0]][f"rows_{split}"]
        y = results[keys[0]][f"y_{split}"]
        for key in keys:
            if not np.array_equal(results[key][f"rows_{split}"], rows):
                raise ValueError(f"{split}: row order differs between {keys[0]} and {key}")
            if not np.array_equal(results[key][f"y_{split}"], y):
                raise ValueError(f"{split}: labels differ between {keys[0]} and {key}")
        smiles = [str(views["split"]["smiles"][int(row)]) for row in rows]
        columns = ["row_id", "smiles", "y_logD"]
        for key in keys:
            columns += [f"pred_{key}", f"err_{key}", f"abs_err_{key}", f"sq_err_{key}"]
        with open(out_dir / f"per_molecule_{split}420.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for i, row in enumerate(rows):
                record: dict[str, Any] = {
                    "row_id": int(row),
                    "smiles": smiles[i],
                    "y_logD": float(y[i]),
                }
                for key in keys:
                    pred = float(results[key][f"pred_{split}"][i])
                    err = pred - float(y[i])
                    record[f"pred_{key}"] = pred
                    record[f"err_{key}"] = err
                    record[f"abs_err_{key}"] = abs(err)
                    record[f"sq_err_{key}"] = err * err
                writer.writerow(record)

    # Prediction integrity: Test rows exactly the frozen split order.
    for key in keys:
        if not np.array_equal(results[key]["rows_test"], np.asarray(views["split"]["rows"]["test"], dtype=np.int64)):
            raise PermissionError(f"{key}: Test420 rows are not in frozen order")
        if not np.array_equal(results[key]["y_test"], np.asarray(views["test"]["y_test"], dtype=np.float64)):
            raise PermissionError(f"{key}: Test420 labels drifted")
    payload["test_row_identity"] = {
        "rows_sha256": C.rows_sha256_int64(views["split"]["rows"]["test"]),
        "expected": C.SPLIT_SHA256["test"],
        "n": C.SPLIT_SIZES["test"],
        "all_models_aligned": True,
    }
    C.write_json(out_dir / "metrics.json", payload)
    C.write_json(out_dir / "bootstrap.json", {"protocol_id": C.PROTOCOL_ID, "scope": "EXPOSED_TEST_EXPLORATORY", "bootstrap": payload["bootstrap"], "pairs": payload["pairs"]})
    return payload


def evaluate_frozen_s0() -> dict[str, Any]:
    payload = evaluate_set(
        ["base_s0", "context_s0"],
        out_dir=C.EVAL_FROZEN_S0,
        device="cpu",
        bootstrap_pairs=[("base_s0", "context_s0")],
    )
    payload["historical_reference"] = {
        "pna_native_test_rmse": C.PNA_NATIVE_TEST_RMSE,
        "pna_native_test_mae": C.PNA_NATIVE_TEST_MAE,
        "historical_lcrc_a_r3_test_rmse": C.HISTORICAL_LCRC_TEST_RMSE,
        "note": (
            "same frozen random42 split; PNA-native was not retrained in this round; "
            "descriptive reference only"
        ),
    }
    C.write_json(C.EVAL_FROZEN_S0 / "metrics.json", payload)
    C.log(
        "[eval-frozen-s0] base test rmse={:.6f} context test rmse={:.6f} gain={:+.6f}".format(
            payload["models"]["base_s0"]["test"]["rmse"],
            payload["models"]["context_s0"]["test"]["rmse"],
            payload["pairs"]["base_s0__vs__context_s0"]["test420"]["gain_rmse"],
        )
    )
    return payload


def evaluate_all() -> dict[str, Any]:
    keys = ["base_s0", "context_s0", "base_s1", "context_s1", "roleblind_s0"]
    missing = [key for key in keys if not (C.RUNS / C.ARM_SPECS[key]["run_key"] / "train.json").exists()]
    if missing:
        raise FileNotFoundError(f"missing formal run artifacts for {missing}; run the formal training first")
    payload = evaluate_set(
        keys,
        out_dir=C.EVAL_DIR,
        device="cpu",
        bootstrap_pairs=[
            ("base_s0", "context_s0"),
            ("base_s1", "context_s1"),
            ("context_s0", "roleblind_s0"),
        ],
    )
    payload["historical_reference"] = {
        "pna_native_test_rmse": C.PNA_NATIVE_TEST_RMSE,
        "pna_native_test_mae": C.PNA_NATIVE_TEST_MAE,
        "historical_lcrc_a_r3_test_rmse": C.HISTORICAL_LCRC_TEST_RMSE,
        "note": (
            "same frozen random42 split; PNA-native was not retrained in this round; "
            "descriptive reference only"
        ),
    }
    C.write_json(C.EVAL_DIR / "metrics.json", payload)
    return payload


__all__ = [
    "model_registry",
    "metric_block",
    "paired_bootstrap",
    "evaluate_model",
    "evaluate_set",
    "evaluate_frozen_s0",
    "evaluate_all",
]

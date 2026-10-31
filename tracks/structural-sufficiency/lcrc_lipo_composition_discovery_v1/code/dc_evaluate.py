"""Evaluation of frozen baselines and new discovery arms.

Validation420 is the screening split.  Test420 is only read for the candidates
explicitly named on the command line after the Stage-I screen; every Test
number is labelled ``EXPOSED_TEST_EXPLORATORY``.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import dc_common as C
import dc_data as D
import dc_models as M

import m2_evaluate as E  # read-only reused metric/bootstrap functions  # noqa: E402
import r42_metrics as MET  # noqa: E402


def model_registry() -> dict[str, dict[str, Any]]:
    registry: dict[str, dict[str, Any]] = {}
    for key, entry in C.FROZEN_CHECKPOINTS.items():
        registry[key] = {
            "key": key,
            "family": entry["family"],
            "seed": int(entry["seed"]),
            "checkpoint": entry["path"],
            "expected_sha256": entry["sha256"],
            "train_json": entry["train_json"],
            "historical": True,
        }
    for arm_key, arm in C.ARM_SPECS.items():
        run_dir = C.RUNS / arm_key
        train_json = run_dir / "train.json"
        if train_json.exists():
            payload = C.read_json(train_json)
            registry[arm_key] = {
                "key": arm_key,
                "family": arm["family"],
                "seed": int(arm["seed"]),
                "checkpoint": run_dir / "checkpoint.pt",
                "expected_sha256": payload["checkpoint_sha256"],
                "train_json": train_json,
                "historical": False,
                "stage": arm.get("stage", 1),
            }
    return registry


def _save_predictions(path: Path, rows, y, pred) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, rows=np.asarray(rows, dtype=np.int64), y_true=np.asarray(y, dtype=np.float64), pred=np.asarray(pred, dtype=np.float64))


def evaluate_model(key: str, entry: Mapping[str, Any], views: Mapping[str, Any], *, splits: Sequence[str], device="cpu") -> dict[str, Any]:
    spec, norm = views["spec"], views["train_norm"]
    model, meta = M.build_family(entry["family"], spec, norm, int(entry["seed"]), device)
    payload = M.load_checkpoint(
        model,
        entry["checkpoint"],
        expected_sha256=entry["expected_sha256"],
        expected_family=entry["family"],
        expected_seed=int(entry["seed"]),
    )
    recomputed = C.state_dict_fingerprint(payload["state_dict"])
    if payload.get("soup_state_hash") is not None and recomputed != payload["soup_state_hash"]:
        raise PermissionError(f"{key}: loaded state hash != stored soup hash")
    train_json = C.read_json(entry["train_json"]) if Path(entry["train_json"]).exists() else {}
    target_norm = views["target_norm"]
    result: dict[str, Any] = {
        "key": key,
        "family": entry["family"],
        "seed": int(entry["seed"]),
        "historical": bool(entry.get("historical", False)),
        "checkpoint": C.track_rel(entry["checkpoint"]),
        "checkpoint_sha256": C.sha256_file(entry["checkpoint"]),
        "soup_state_hash": payload.get("soup_state_hash"),
        "parameters": int(meta["parameters"]),
        "parameters_base": int(meta.get("parameters_base", meta["parameters"])),
        "parameters_new": int(meta.get("parameters_new", 0)),
        "train_rmse": float(train_json.get("metrics", {}).get("train", {}).get("rmse", float("nan"))),
        "train_mae": float(train_json.get("metrics", {}).get("train", {}).get("mae", float("nan"))),
        "train_wall_seconds": float(train_json.get("timing", {}).get("train_wall_seconds", float("nan"))),
        "soup_members": payload.get("soup_members"),
    }
    for split in splits:
        cache = views[f"{split}_cache"]
        rows = np.asarray(views["split"]["rows"][split], dtype=np.int64)
        if split == "test":
            y = np.asarray(views["test"]["y_test"], dtype=np.float64)
        else:
            y = np.asarray(views["labels"][f"y_{split}"], dtype=np.float64)
        pred = M.predict(model, cache, device)
        pred = pred * float(target_norm["scale"]) + float(target_norm["mean"])
        if pred.size != rows.size or not np.isfinite(pred).all():
            raise ValueError(f"{key}/{split}: prediction size or finiteness failure")
        result[split] = E.metric_block(pred, y)
        result[f"pred_{split}"] = pred
        result[f"rows_{split}"] = rows
        result[f"y_{split}"] = y
        _save_predictions(C.PREDICTIONS / f"{key}_{split}420.npz", rows, y, pred)
    return result


def _bootstrap_pairs(keys: Sequence[str]) -> list[tuple[str, str]]:
    keys = list(keys)
    pairs: list[tuple[str, str]] = []
    for key in keys:
        if key not in ("base_s0", "roleblind_s0"):
            pairs.append(("roleblind_s0", key))
            pairs.append(("base_s0", key))
    for family_a, family_b, _la, _lb in C.MECHANISM_PAIRS.values():
        for a in keys:
            for b in keys:
                if a.endswith(f"{family_a}_s0") or a == family_a:
                    if b.endswith(f"{family_b}_s0") or b == family_b:
                        pairs.append((a, b))
    seen = []
    for p in pairs:
        if p[0] in keys and p[1] in keys and p not in seen:
            seen.append(p)
    return seen


def evaluate_set(keys: Sequence[str], *, splits: Sequence[str], out_dir: Path, device="cpu") -> dict[str, Any]:
    registry = model_registry()
    missing = [key for key in keys if key not in registry]
    if missing:
        raise KeyError(f"unknown evaluation keys: {missing}")
    views = D.load_views(include_test=("test" in splits))
    results = {}
    for key in keys:
        C.log(f"[eval] {key} ({registry[key]['family']}, seed {registry[key]['seed']}) splits={list(splits)}")
        results[key] = evaluate_model(key, registry[key], views, splits=splits, device=device)

    payload: dict[str, Any] = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "device": str(device),
        "splits": list(splits),
        "scope": "VALIDATION_SCREEN" if "test" not in splits else "EXPOSED_TEST_EXPLORATORY",
        "test_access": "EXPOSED_TEST_EXPLORATORY" if "test" in splits else "NOT_READ",
        "test_labels_source": dict(views["test"]["audit"]) if "test" in splits else None,
        "models": {},
        "pairs": {},
        "bootstrap": {},
    }
    for key in keys:
        r = results[key]
        payload["models"][key] = {k: r[k] for k in ("key", "family", "seed", "historical", "checkpoint", "checkpoint_sha256", "soup_state_hash", "parameters", "parameters_base", "parameters_new", "train_rmse", "train_mae", "train_wall_seconds", "soup_members")}
        for split in splits:
            payload["models"][key][split] = r[split]
    for ref_key, other_key in _bootstrap_pairs(keys):
        pair = f"{ref_key}__vs__{other_key}"
        payload["pairs"][pair] = {}
        payload["bootstrap"][pair] = {}
        for split in splits:
            ref_pred = results[ref_key][f"pred_{split}"]
            other_pred = results[other_key][f"pred_{split}"]
            y = results[ref_key][f"y_{split}"]
            if not np.array_equal(y, results[other_key][f"y_{split}"]):
                raise ValueError(f"{pair}/{split}: labels differ")
            boot = E.paired_bootstrap(ref_pred, other_pred, y)
            payload["bootstrap"][pair][f"{split}420"] = boot
            obs = boot["observed"]
            payload["pairs"][pair][f"{split}420"] = {
                "rmse_ref": obs["rmse_ref"],
                "rmse_other": obs["rmse_other"],
                "gain_rmse": obs["gain_rmse"],
                "mae_ref": obs["mae_ref"],
                "mae_other": obs["mae_other"],
                "mae_change": obs["mae_change"],
                "sse_gain": obs["sse_gain"],
                "mean_abs_prediction_difference": obs["mean_abs_prediction_difference"],
                "max_abs_prediction_difference": obs["max_abs_prediction_difference"],
                "n_other_better_abs": obs["n_other_better_abs"],
                "n_ref_better_abs": obs["n_ref_better_abs"],
            }

    out_dir.mkdir(parents=True, exist_ok=True)
    fields = ["key", "family", "seed", "split", "n", "rmse", "mae", "r2", "sse", "max_abs_error", "mean_abs_error"]
    with open(out_dir / "metrics_by_model_split.csv", "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for key in keys:
            for split in splits:
                row = {"key": key, "family": results[key]["family"], "seed": results[key]["seed"], "split": split}
                row.update({name: results[key][split][name] for name in fields[4:]})
                writer.writerow(row)
    for split in splits:
        first = results[keys[0]]
        rows = first[f"rows_{split}"]
        y = first[f"y_{split}"]
        smiles = [str(views["split"]["smiles"][int(row)]) for row in rows]
        columns = ["row_id", "smiles", "y_logD"]
        for key in keys:
            if not np.array_equal(results[key][f"rows_{split}"], rows) or not np.array_equal(results[key][f"y_{split}"], y):
                raise ValueError(f"{split}: row/label alignment differs for {key}")
            columns += [f"pred_{key}", f"err_{key}", f"abs_err_{key}", f"sq_err_{key}"]
        with open(out_dir / f"per_molecule_{split}420.csv", "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for i, row in enumerate(rows):
                rec = {"row_id": int(row), "smiles": smiles[i], "y_logD": float(y[i])}
                for key in keys:
                    pred = float(results[key][f"pred_{split}"][i])
                    err = pred - float(y[i])
                    rec[f"pred_{key}"] = pred
                    rec[f"err_{key}"] = err
                    rec[f"abs_err_{key}"] = abs(err)
                    rec[f"sq_err_{key}"] = err * err
                writer.writerow(rec)
    C.write_json(out_dir / "metrics.json", payload)
    C.write_json(out_dir / "bootstrap.json", {"protocol_id": C.PROTOCOL_ID, "scope": payload["scope"], "bootstrap": payload["bootstrap"], "pairs": payload["pairs"]})
    return payload


__all__ = ["model_registry", "evaluate_model", "evaluate_set"]

"""Training, checkpoint verification and one-shot outer evaluation for the pilot.

The model/training implementation is the frozen engine of
``lcrc_lipo_best_epoch_distance_fix_v1``.  This wrapper only supplies the new
data views, stamps the new round identity, and enforces the order:

    six trainings -> checkpoint verification -> frozen manifest -> outer read

The outer-evaluation cache and labels are not built by the training views.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping

import numpy as np
import torch

import pilot_common as PC
import pilot_data as PD

import be_models as MODELS  # noqa: E402
import be_train as TRAIN  # noqa: E402
import r42_metrics as MET  # noqa: E402
import r42_native_train as NT  # noqa: E402

TOL = 1.0e-5


def arm_parts(arm: str) -> tuple[str, int]:
    if arm not in PC.PILOT_ARMS:
        raise ValueError(f"{arm!r} is not one of {PC.PILOT_ARMS}")
    family = arm[len("pilot_") : arm.rindex("_s")]
    seed = int(arm[arm.rindex("_s") + 2 :])
    return family, seed


def run_dir(arm: str) -> Path:
    return PC.RUNS / str(arm)


def _device(name: str):
    device = torch.device(name)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available")
    return device


def train(arm: str, device_name: str, pool: str) -> dict[str, Any]:
    family, seed = arm_parts(arm)
    PC.require_freeze()
    PD.load_split()
    device = _device(device_name)
    if device.type != "cuda":
        raise PermissionError("formal pilot training must run on CUDA")
    run_path = run_dir(arm)
    if (run_path / "train.json").exists():
        raise PermissionError(f"{arm} already trained (one-shot)")
    cfg = PC.load_config()
    views = PD.build_pilot_views(include_outer=False)
    PC.patch_reused_engine()
    payload = TRAIN.train_arm(
        arm=arm,
        cfg=cfg,
        views=views,
        device=device,
        run_dir=run_path,
        pool=str(pool),
        smoke=False,
        family=family,
        seed=seed,
    )
    payload.update(
        {
            "pilot_protocol_id": PC.PROTOCOL_ID,
            "pilot_round_id": PC.ROUND_ID,
            "pilot_split_manifest": PC.track_rel(PC.SPLIT_MANIFEST),
            "pilot_split_manifest_sha256": PC.sha256_file(PC.SPLIT_MANIFEST),
            "pilot_split_audit_sha256": PC.sha256_file(PC.SPLIT_AUDIT),
            "pilot_config_sha256": PC.config_sha256(),
            "pilot_code_hashes": PC.code_hashes(),
            "outer_labels_read": False,
            "outer_test_cache_built": False,
            "outer_test_rows_used_in_training": False,
        }
    )
    PC.write_json(run_path / "train.json", payload)
    return payload


def smoke() -> dict[str, Any]:
    """Short CPU smoke: both families, two epochs, tiny train/valid slices."""
    PC.SMOKE.mkdir(parents=True, exist_ok=True)
    cfg = PC.load_config()
    views = PD.build_pilot_views(include_outer=False)
    PC.patch_reused_engine()
    out: dict[str, Any] = {}
    for family in PC.PILOT_MODELS:
        arm = f"pilot_{family}_s0"
        run_path = PC.SMOKE / f"{arm}_smoke"
        payload = TRAIN.train_arm(
            arm=arm,
            cfg=cfg,
            views=views,
            device=torch.device("cpu"),
            run_dir=run_path,
            max_epochs=2,
            limit_train=64,
            limit_valid=32,
            pool="local-smoke",
            smoke=True,
            family=family,
            seed=0,
        )
        out[arm] = {
            "run_dir": PC.track_rel(run_path),
            "best_epoch": payload["checkpoint"]["best_epoch"],
            "valid_rmse": payload["metrics"]["valid"]["rmse"],
            "parameters": payload["model_audit"]["parameters"],
            "recheck_abs_diff": payload["checkpoint"]["valid_rmse_recheck_abs_diff"],
        }
    payload = {
        "protocol_id": PC.PROTOCOL_ID,
        "created_utc": PC.utcnow(),
        "status": "COMPLETED",
        "smoke": True,
        "results": out,
        "environment": PC.env_info(),
    }
    PC.write_json(PC.SMOKE / "smoke.json", payload)
    return payload


def _read_curve(path: Path) -> list[dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def verify_arm(arm: str, views: Mapping[str, Any]) -> dict[str, Any]:
    family, seed = arm_parts(arm)
    run_path = run_dir(arm)
    train_json = PC.read_json(run_path / "train.json")
    ckpt_path = run_path / "checkpoint.pt"
    ckpt_sha = PC.sha256_file(ckpt_path)
    if ckpt_sha != train_json["checkpoint"]["sha256"]:
        raise PermissionError(f"{arm}: checkpoint hash differs from train.json")
    if train_json.get("smoke") is not False:
        raise PermissionError(f"{arm}: formal verification refuses smoke runs")
    model, _meta = MODELS.build_family(
        family, views["spec"], views["train_norm"], seed, "cpu"
    )
    checkpoint = MODELS.load_checkpoint(
        model,
        ckpt_path,
        expected_sha256=ckpt_sha,
        expected_family=family,
        expected_seed=seed,
    )
    curve = _read_curve(run_path / "curve.csv")
    curve_valid = np.asarray([float(row["valid_rmse_raw"]) for row in curve], dtype=np.float64)
    argmin_epoch = int(np.argmin(curve_valid)) + 1
    if argmin_epoch != int(checkpoint["best_epoch"]):
        raise RuntimeError(f"{arm}: curve argmin {argmin_epoch} != checkpoint best {checkpoint['best_epoch']}")
    pred_valid = NT.denormalise(
        MODELS.predict(model, views["valid_cache"], "cpu"), views["target_norm"]
    )
    metrics = MET.all_metrics(pred_valid, np.asarray(views["labels"]["y_valid"], dtype=np.float64))
    recheck_diff = abs(float(metrics["rmse"]) - float(checkpoint["best_valid_rmse"]))
    if recheck_diff > TOL:
        raise RuntimeError(f"{arm}: inner Validation RMSE does not reproduce ({recheck_diff})")
    state_hash = PC.OLD.state_dict_fingerprint(checkpoint["state_dict"])
    if state_hash != checkpoint["best_state_hash"]:
        raise RuntimeError(f"{arm}: checkpoint state hash mismatch")
    if train_json.get("outer_labels_read") is not False:
        raise PermissionError(f"{arm}: training record claims outer-label access")
    if train_json["data"].get("test_row_values_used") is not False:
        raise PermissionError(f"{arm}: training data record claims outer-row values were used")
    return {
        "arm": arm,
        "model": PC.MODEL_DISPLAY[family],
        "family": family,
        "seed": seed,
        "checkpoint": PC.track_rel(ckpt_path),
        "checkpoint_sha256": ckpt_sha,
        "best_epoch": int(checkpoint["best_epoch"]),
        "best_inner_valid_rmse": float(checkpoint["best_valid_rmse"]),
        "best_inner_valid_mae": float(checkpoint["best_valid_mae"]),
        "train_rmse": float(train_json["metrics"]["train"]["rmse"]),
        "curve_sha256": PC.sha256_file(run_path / "curve.csv"),
        "curve_argmin_epoch": argmin_epoch,
        "reloaded_inner_valid_rmse": float(metrics["rmse"]),
        "reloaded_inner_valid_mae": float(metrics["mae"]),
        "recheck_abs_diff": float(recheck_diff),
        "state_hash": state_hash,
        "parameters": int(train_json["model_audit"]["parameters"]),
        "smoke": False,
    }


def freeze_manifest() -> dict[str, Any]:
    PC.require_freeze()
    views = PD.build_pilot_views(include_outer=False)
    lanes: dict[str, Any] = {}
    for arm in PC.PILOT_ARMS:
        if not (run_dir(arm) / "train.json").exists():
            raise PermissionError(f"missing pilot run {arm}")
        lanes[arm] = verify_arm(arm, views)
    body = {
        "protocol_id": PC.PROTOCOL_ID,
        "created_utc": PC.utcnow(),
        "git_commit": PC.git_commit(),
        "status": "PILOT_MANIFEST_FROZEN_BEFORE_OUTER_READ",
        "scope": "one new Bemis-Murcko scaffold partition; development-style outer evaluation",
        "split_manifest": PC.track_rel(PC.SPLIT_MANIFEST),
        "split_manifest_sha256": PC.sha256_file(PC.SPLIT_MANIFEST),
        "split_audit_sha256": PC.sha256_file(PC.SPLIT_AUDIT),
        "config_sha256": PC.config_sha256(),
        "n_arms": len(lanes),
        "arms": lanes,
        "outer_labels_read_at_freeze_time": False,
        "outer_test_cache_built_at_freeze_time": False,
    }
    body["manifest_sha256"] = PC.sha256_json(body)
    PC.write_json(PC.MANIFEST, body)
    PC.write_json(
        PC.MANIFEST_FROZEN,
        {
            "manifest_sha256": body["manifest_sha256"],
            "created_utc": body["created_utc"],
            "n_arms": len(lanes),
            "scope": "frozen before the one-shot outer-evaluation read",
        },
    )
    return body


def _frozen_manifest() -> dict[str, Any]:
    if not PC.MANIFEST.exists() or not PC.MANIFEST_FROZEN.exists():
        raise PermissionError("pilot manifest is not frozen; run `manifest` first")
    manifest = PC.read_json(PC.MANIFEST)
    marker = PC.read_json(PC.MANIFEST_FROZEN)
    body = {k: v for k, v in manifest.items() if k != "manifest_sha256"}
    if PC.sha256_json(body) != manifest["manifest_sha256"]:
        raise PermissionError("pilot manifest content changed after the freeze")
    if marker.get("manifest_sha256") != manifest["manifest_sha256"]:
        raise PermissionError("pilot manifest marker mismatch")
    return manifest


def _mean_std(values) -> tuple[float, float]:
    arr = np.asarray(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1)) if arr.size > 1 else 0.0


def _summary(rows: list[dict[str, Any]]) -> dict[str, Any]:
    by_model: dict[str, list[dict[str, Any]]] = {}
    for row in rows:
        by_model.setdefault(row["model"], []).append(row)
    summary: dict[str, Any] = {}
    for model, subset in by_model.items():
        subset = sorted(subset, key=lambda x: int(x["seed"]))
        inner_rmse_mean, inner_rmse_sd = _mean_std([r["inner_valid_rmse"] for r in subset])
        inner_mae_mean, inner_mae_sd = _mean_std([r["inner_valid_mae"] for r in subset])
        outer_rmse_mean, outer_rmse_sd = _mean_std([r["outer_rmse"] for r in subset])
        outer_mae_mean, outer_mae_sd = _mean_std([r["outer_mae"] for r in subset])
        summary[model] = {
            "n_seeds": len(subset),
            "seeds": [int(r["seed"]) for r in subset],
            "best_epochs": [int(r["best_epoch"]) for r in subset],
            "inner_valid_rmse_mean": inner_rmse_mean,
            "inner_valid_rmse_sd": inner_rmse_sd,
            "inner_valid_mae_mean": inner_mae_mean,
            "inner_valid_mae_sd": inner_mae_sd,
            "outer_rmse_mean": outer_rmse_mean,
            "outer_rmse_sd": outer_rmse_sd,
            "outer_mae_mean": outer_mae_mean,
            "outer_mae_sd": outer_mae_sd,
        }
    paired: dict[str, Any] = {}
    if "BASE" in by_model and "G2L-ALL" in by_model:
        base = {int(r["seed"]): r for r in by_model["BASE"]}
        g2l = {int(r["seed"]): r for r in by_model["G2L-ALL"]}
        per_seed = []
        for seed in sorted(set(base) & set(g2l)):
            b, g = base[seed], g2l[seed]
            per_seed.append(
                {
                    "seed": seed,
                    "inner_delta_rmse_base_minus_g2l": float(b["inner_valid_rmse"] - g["inner_valid_rmse"]),
                    "inner_delta_mae_base_minus_g2l": float(b["inner_valid_mae"] - g["inner_valid_mae"]),
                    "outer_delta_rmse_base_minus_g2l": float(b["outer_rmse"] - g["outer_rmse"]),
                    "outer_delta_mae_base_minus_g2l": float(b["outer_mae"] - g["outer_mae"]),
                }
            )
        paired = {
            "definition": "positive delta = G2L-ALL better than BASE (BASE minus G2L-ALL)",
            "per_seed": per_seed,
            "wins": {
                "inner_rmse": sum(1 for r in per_seed if r["inner_delta_rmse_base_minus_g2l"] > 0),
                "inner_mae": sum(1 for r in per_seed if r["inner_delta_mae_base_minus_g2l"] > 0),
                "outer_rmse": sum(1 for r in per_seed if r["outer_delta_rmse_base_minus_g2l"] > 0),
                "outer_mae": sum(1 for r in per_seed if r["outer_delta_mae_base_minus_g2l"] > 0),
            },
            "means": {
                key: float(np.mean([r[key] for r in per_seed]))
                for key in (
                    "inner_delta_rmse_base_minus_g2l",
                    "inner_delta_mae_base_minus_g2l",
                    "outer_delta_rmse_base_minus_g2l",
                    "outer_delta_mae_base_minus_g2l",
                )
            },
        }
    return {"models": summary, "paired_g2l_vs_base": paired}


def evaluate_outer(*, force: bool = False) -> dict[str, Any]:
    manifest = _frozen_manifest()
    if (PC.EVAL / "metrics.json").exists() and not force:
        raise PermissionError("outer evaluation already exists; one-shot policy (pass --force only for a bug fix)")
    views = PD.build_pilot_views(include_outer=True)
    y_valid = np.asarray(views["labels"]["y_valid"], dtype=np.float64)
    y_test = np.asarray(views["y_test"], dtype=np.float64)
    rows_valid = np.asarray(views["split"]["rows"]["valid"], dtype=np.int64)
    rows_test = np.asarray(views["split"]["rows"]["test"], dtype=np.int64)
    PC.EVAL.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for arm in PC.PILOT_ARMS:
        frozen = manifest["arms"][arm]
        ckpt_path = Path(PC.REPO / frozen["checkpoint"])
        if PC.sha256_file(ckpt_path) != frozen["checkpoint_sha256"]:
            raise PermissionError(f"{arm}: checkpoint changed after the manifest freeze")
        family = str(frozen["family"])
        seed = int(frozen["seed"])
        model, _meta = MODELS.build_family(family, views["spec"], views["train_norm"], seed, "cpu")
        MODELS.load_checkpoint(
            model,
            ckpt_path,
            expected_sha256=frozen["checkpoint_sha256"],
            expected_family=family,
            expected_seed=seed,
        )
        pred_valid = NT.denormalise(MODELS.predict(model, views["valid_cache"], "cpu"), views["target_norm"])
        pred_test = NT.denormalise(MODELS.predict(model, views["test_cache"], "cpu"), views["target_norm"])
        valid_metrics = MET.all_metrics(pred_valid, y_valid)
        test_metrics = MET.all_metrics(pred_test, y_test)
        if abs(float(valid_metrics["rmse"]) - float(frozen["best_inner_valid_rmse"])) > TOL:
            raise RuntimeError(f"{arm}: inner Validation RMSE does not reproduce before outer read")
        results[arm] = {
            "arm": arm,
            "model": frozen["model"],
            "family": family,
            "seed": seed,
            "checkpoint": frozen["checkpoint"],
            "checkpoint_sha256": frozen["checkpoint_sha256"],
            "best_epoch": int(frozen["best_epoch"]),
            "inner_valid_rmse": float(valid_metrics["rmse"]),
            "inner_valid_mae": float(valid_metrics["mae"]),
            "inner_valid_r2": float(valid_metrics["r2"]),
            "outer_rmse": float(test_metrics["rmse"]),
            "outer_mae": float(test_metrics["mae"]),
            "outer_r2": float(test_metrics["r2"]),
            "train_rmse": float(frozen["train_rmse"]),
            "parameters": int(frozen["parameters"]),
        }
        arrays[arm] = {"valid": pred_valid, "test": pred_test}
        PC.log(
            f"[outer-eval] {arm} epoch={frozen['best_epoch']} "
            f"inner={valid_metrics['rmse']:.6f} outer={test_metrics['rmse']:.6f}"
        )

    rows_for_table = [
        {
            "model": results[arm]["model"],
            "arm": arm,
            "seed": results[arm]["seed"],
            "best_epoch": results[arm]["best_epoch"],
            "inner_valid_rmse": results[arm]["inner_valid_rmse"],
            "inner_valid_mae": results[arm]["inner_valid_mae"],
            "outer_rmse": results[arm]["outer_rmse"],
            "outer_mae": results[arm]["outer_mae"],
            "outer_r2": results[arm]["outer_r2"],
        }
        for arm in PC.PILOT_ARMS
    ]
    with open(PC.EVAL / "table_runs.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows_for_table[0].keys()))
        writer.writeheader()
        writer.writerows(rows_for_table)

    main_columns = ["Model", "Seed", "Best epoch", "Inner Valid RMSE", "Outer RMSE", "Outer MAE"]
    with open(PC.EVAL / "table_main.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=main_columns)
        writer.writeheader()
        for row in rows_for_table:
            writer.writerow(
                {
                    "Model": row["model"],
                    "Seed": row["seed"],
                    "Best epoch": row["best_epoch"],
                    "Inner Valid RMSE": row["inner_valid_rmse"],
                    "Outer RMSE": row["outer_rmse"],
                    "Outer MAE": row["outer_mae"],
                }
            )

    summary = _summary(rows_for_table)
    PC.write_json(PC.EVAL / "summary.json", summary)
    with open(PC.EVAL / "table_summary.csv", "w", newline="", encoding="utf-8") as handle:
        columns = [
            "model",
            "n_seeds",
            "inner_valid_rmse_mean",
            "inner_valid_rmse_sd",
            "inner_valid_mae_mean",
            "inner_valid_mae_sd",
            "outer_rmse_mean",
            "outer_rmse_sd",
            "outer_mae_mean",
            "outer_mae_sd",
        ]
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for model, rec in summary["models"].items():
            writer.writerow({"model": model, **{key: rec[key] for key in columns if key != "model"}})

    _write_outer_per_molecule(views, arrays, rows_test, y_test)
    payload = {
        "protocol_id": PC.PROTOCOL_ID,
        "created_utc": PC.utcnow(),
        "git_commit": PC.git_commit(),
        "git_dirty_tracked": PC.git_dirty(),
        "scope": "DEVELOPMENT_NEW_SCAFFOLD_PARTITION_NOT_BLIND",
        "split_manifest_sha256": PC.sha256_file(PC.SPLIT_MANIFEST),
        "manifest_sha256": manifest["manifest_sha256"],
        "outer_labels_read_after_manifest_freeze": True,
        "outer_used_for_selection": False,
        "models": results,
        "summary": summary,
        "alignment": {
            "n_train": int(len(views["split"]["rows"]["train"])),
            "n_valid": int(rows_valid.size),
            "n_test": int(rows_test.size),
            "valid_rows_sha256": PC.rows_sha256(rows_valid.tolist()),
            "test_rows_sha256": PC.rows_sha256(rows_test.tolist()),
            "overlap_valid_test": int(len(set(rows_valid.tolist()) & set(rows_test.tolist()))),
        },
    }
    PC.write_json(PC.EVAL / "metrics.json", payload)
    return payload


def _write_outer_per_molecule(
    views: Mapping[str, Any],
    arrays: Mapping[str, Mapping[str, np.ndarray]],
    rows_test: np.ndarray,
    y_test: np.ndarray,
) -> None:
    smiles = views["split"]["smiles"]
    columns = ["row_id", "smiles", "y_logD"]
    for arm in PC.PILOT_ARMS:
        columns += [f"pred_{arm}", f"err_{arm}", f"abs_err_{arm}", f"sq_err_{arm}"]
    with open(PC.EVAL / "per_molecule_outer.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=columns)
        writer.writeheader()
        for index, row in enumerate(rows_test):
            record = {
                "row_id": int(row),
                "smiles": str(smiles[int(row)]),
                "y_logD": float(y_test[index]),
            }
            for arm in PC.PILOT_ARMS:
                pred = float(arrays[arm]["test"][index])
                error = pred - float(y_test[index])
                record[f"pred_{arm}"] = pred
                record[f"err_{arm}"] = error
                record[f"abs_err_{arm}"] = abs(error)
                record[f"sq_err_{arm}"] = error * error
            writer.writerow(record)


__all__ = [
    "arm_parts",
    "run_dir",
    "train",
    "smoke",
    "verify_arm",
    "freeze_manifest",
    "evaluate_outer",
]

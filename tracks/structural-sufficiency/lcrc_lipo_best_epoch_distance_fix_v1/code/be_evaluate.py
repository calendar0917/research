"""Checkpoint verification, manifest freeze and the single Test420 pass.

Order of operations enforced here:

1. ``verify_all``  reload every best-epoch checkpoint, recompute Validation420
   from scratch and check it against the recorded ``best_valid_rmse``, the
   curve argmin, the stored state hash and the saved per-molecule predictions.
2. ``build_manifest``  freeze the 18 checkpoints (hash, best epoch, Validation
   metrics) *before* any Test label is read.
3. ``evaluate_test``  load the frozen manifest and only then read Test420,
   predict every frozen checkpoint with identical weights and write the
   ``EXPOSED_TEST_EXPLORATORY`` numbers.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import be_common as C
import be_data as D
import be_models as M

import r42_data as R42D  # noqa: E402
import r42_metrics as MET  # noqa: E402
import r42_native_train as NT  # noqa: E402

TOL = 1.0e-9


def _entry(arm: str) -> dict[str, Any]:
    spec = C.ARM_SPECS[arm]
    payload = C.load_train_json(arm)
    return {
        "arm": arm,
        "family": spec["family"],
        "seed": int(spec["seed"]),
        "run_dir": C.run_dir(arm),
        "train_json": payload,
        "checkpoint": C.run_dir(arm) / "checkpoint.pt",
        "checkpoint_sha256": payload["checkpoint"]["sha256"],
    }


def _rebuild(entry: Mapping[str, Any], views: Mapping[str, Any], device="cpu"):
    model, meta = M.build_family(
        entry["family"], views["spec"], views["train_norm"], int(entry["seed"]), device
    )
    payload = M.load_checkpoint(
        model,
        entry["checkpoint"],
        expected_sha256=entry["checkpoint_sha256"],
        expected_family=entry["family"],
        expected_seed=int(entry["seed"]),
    )
    return model, meta, payload


def _align_check(name: str, ok: bool, detail: Any, checks: list[dict[str, Any]]) -> bool:
    checks.append({"id": name, "passed": bool(ok), "detail": detail})
    return bool(ok)


def verify_all(device="cpu") -> dict[str, Any]:
    views = D.load_views(include_test=False)
    rows_train = np.asarray(views["split"]["rows"]["train"], dtype=np.int64)
    rows_valid = np.asarray(views["split"]["rows"]["valid"], dtype=np.int64)
    y_valid = np.asarray(views["labels"]["y_valid"], dtype=np.float64)
    n_valid = int(y_valid.size)
    lanes: dict[str, Any] = {}
    all_passed = True
    for arm in C.ARMS:
        entry = _entry(arm)
        checks: list[dict[str, Any]] = []
        payload = entry["train_json"]
        model, meta, ckpt = _rebuild(entry, views, device)
        state_hash = C.state_dict_fingerprint(ckpt["state_dict"])
        curve_path = entry["run_dir"] / "curve.csv"
        curve = _read_curve(curve_path)
        valid_curve = np.asarray([row["valid_rmse_raw"] for row in curve], dtype=np.float64)
        saved = np.load(entry["run_dir"] / "valid_predictions.npz")
        saved_rows = np.asarray(saved["rows"], dtype=np.int64)
        if saved_rows.size == rows_valid.size and np.array_equal(saved_rows, rows_valid):
            eval_cache = views["valid_cache"]
            y_eval = y_valid
        else:
            # smoke / dry-run artifact: re-evaluate exactly the saved rows
            position = {int(row): i for i, row in enumerate(rows_valid.tolist())}
            positions = [position[int(r)] for r in saved_rows.tolist()]
            eval_cache = D.build_distance_buckets(
                R42D.select_cache(views["valid_cache"], positions, split_label="verify_valid")
            )
            y_eval = np.asarray(saved["y_true"], dtype=np.float64)
        recorded_norm = payload["data"]["target_norm"]
        pred = NT.denormalise(M.predict(model, eval_cache, device), recorded_norm)
        metrics = MET.all_metrics(pred, y_eval)

        ok = True
        ok &= _align_check(
            "checkpoint_sha256_matches_train_json",
            C.sha256_file(entry["checkpoint"]) == entry["checkpoint_sha256"],
            {"sha256": C.sha256_file(entry["checkpoint"])},
            checks,
        )
        ok &= _align_check(
            "checkpoint_state_hash_matches_payload",
            state_hash == ckpt["best_state_hash"] == payload["checkpoint"]["best_state_hash"],
            {"recomputed": state_hash, "payload": ckpt["best_state_hash"]},
            checks,
        )
        ok &= _align_check(
            "best_epoch_is_curve_argmin",
            int(np.argmin(valid_curve)) + 1 == int(ckpt["best_epoch"]) == int(payload["checkpoint"]["best_epoch"]),
            {"best_epoch": ckpt["best_epoch"], "curve_argmin": int(np.argmin(valid_curve)) + 1},
            checks,
        )
        ok &= _align_check(
            "curve_min_matches_best_valid_rmse",
            abs(float(valid_curve.min()) - float(ckpt["best_valid_rmse"])) == 0.0,
            {"curve_min": float(valid_curve.min()), "best_valid_rmse": ckpt["best_valid_rmse"]},
            checks,
        )
        ok &= _align_check(
            "reloaded_valid_rmse_matches_recorded",
            abs(float(metrics["rmse"]) - float(ckpt["best_valid_rmse"])) <= TOL,
            {"recomputed": float(metrics["rmse"]), "recorded": float(ckpt["best_valid_rmse"])},
            checks,
        )
        ok &= _align_check(
            "saved_valid_predictions_reproduced",
            float(np.abs(pred - np.asarray(saved["pred"], dtype=np.float64)).max()) == 0.0,
            {"max_abs_diff": float(np.abs(pred - np.asarray(saved["pred"], dtype=np.float64)).max())},
            checks,
        )
        ok &= _align_check(
            "valid_rows_aligned_to_split",
            bool(np.array_equal(saved_rows, rows_valid[: saved_rows.size]))
            and bool(np.array_equal(np.asarray(saved["y_true"], dtype=np.float64), y_eval)),
            {"n": int(saved_rows.size), "full_valid420": int(saved_rows.size) == n_valid},
            checks,
        )
        ok &= _align_check(
            "valid_prediction_length",
            int(pred.size) == int(y_eval.size) and bool(np.isfinite(pred).all()),
            {"n": int(pred.size)},
            checks,
        )
        ok &= _align_check(
            "curve_sha256_matches_train_json",
            C.sha256_file(curve_path) == payload["checkpoint"]["curve_sha256"],
            {"curve_sha256": C.sha256_file(curve_path)},
            checks,
        )
        expected_epochs = int(C.TRAIN_RECIPE["epochs"]) if payload.get("smoke") is not True else len(curve)
        ok &= _align_check(
            "epochs_run_matches_recipe",
            len(curve) == expected_epochs and int(ckpt["epochs_run"]) == len(curve),
            {"epochs_run": len(curve), "expected": expected_epochs, "smoke": bool(payload.get("smoke"))},
            checks,
        )
        ok &= _align_check(
            "no_soup_or_ema",
            ckpt.get("soup_members") is None
            and ckpt.get("soup_state_hash") is None
            and payload["checkpoint"]["soup_used"] is False
            and payload["checkpoint"]["ema_used"] is False
            and payload["checkpoint"]["swa_used"] is False,
            {"soup_members": ckpt.get("soup_members")},
            checks,
        )
        norm_ok = abs(float(recorded_norm["mean"]) - C.TARGET_MEAN) < 1e-15 and abs(
            float(recorded_norm["scale"]) - C.TARGET_SCALE
        ) < 1e-15
        ok &= _align_check(
            "target_norm_is_train3360_only",
            bool(norm_ok or payload.get("smoke") is True),
            {
                "mean": recorded_norm["mean"],
                "scale": recorded_norm["scale"],
                "smoke_override": bool(payload.get("smoke") is True),
            },
            checks,
        )
        ok &= _align_check(
            "no_test_read_during_training",
            payload["test_labels_read"] is False
            and payload["test_predictions_read"] is False
            and payload["test_metrics_read"] is False
            and payload["data"]["test_row_values_used"] is False
            and payload["recipe"]["test_used_for_any_selection"] is False,
            {"label_scope": payload["label_scope"]},
            checks,
        )
        ok &= _align_check(
            "no_early_stopping",
            payload["recipe"]["early_stopping"] is False
            and payload["recipe"]["validation_used_for_early_stopping"] is False,
            {},
            checks,
        )
        ok &= _align_check(
            "train_rows_aligned_to_split",
            np.array_equal(
                np.asarray(np.load(entry["run_dir"] / "train_predictions.npz")["rows"], dtype=np.int64),
                np.asarray(views["split"]["rows"]["train"][: int(payload["data"]["n_train"])], dtype=np.int64),
            ),
            {"n": int(payload["data"]["n_train"])},
            checks,
        )
        all_passed &= bool(ok)
        lanes[arm] = {
            "arm": arm,
            "family": entry["family"],
            "model": C.MODEL_DISPLAY[entry["family"]],
            "seed": int(entry["seed"]),
            "checkpoint": C.track_rel(entry["checkpoint"]),
            "checkpoint_sha256": entry["checkpoint_sha256"],
            "best_epoch": int(ckpt["best_epoch"]),
            "best_valid_rmse": float(ckpt["best_valid_rmse"]),
            "best_valid_mae": float(ckpt["best_valid_mae"]),
            "recomputed_valid": dict(metrics),
            "parameters": int(meta["parameters"]),
            "checks": checks,
            "passed": bool(ok),
        }
    # Cross-run alignment/identity checks.
    rows_train = np.asarray(views["split"]["rows"]["train"], dtype=np.int64)
    test_rows = np.asarray(views["split"]["rows"]["test"], dtype=np.int64)
    global_checks = [
        {
            "id": "split_row_sets_disjoint_and_complete",
            "passed": bool(
                set(rows_train.tolist()).isdisjoint(rows_valid.tolist())
                and set(rows_train.tolist()).isdisjoint(test_rows.tolist())
                and set(rows_valid.tolist()).isdisjoint(test_rows.tolist())
                and len(set(rows_train.tolist()) | set(rows_valid.tolist()) | set(test_rows.tolist())) == C.N_MOLECULES
            ),
            "detail": {
                "n_train": int(rows_train.size),
                "n_valid": int(rows_valid.size),
                "n_test": int(test_rows.size),
            },
        },
        {
            "id": "batch_order_identical_within_seed",
            "passed": all(
                len({lanes[C.arm_key(f, s)]["arm"] for f in C.FAMILIES}) == len(C.FAMILIES)
                and len(
                    {
                        C.load_train_json(C.arm_key(f, s))["recipe"]["batch_order_sha256"]
                        for f in C.FAMILIES
                    }
                )
                == 1
                for s in C.SEEDS
            ),
            "detail": {
                str(s): {
                    f: C.load_train_json(C.arm_key(f, s))["recipe"]["batch_order_sha256"] for f in C.FAMILIES
                }
                for s in C.SEEDS
            },
        },
        {
            "id": "dropout_rng_identical_within_seed",
            "passed": all(
                len(
                    {
                        C.load_train_json(C.arm_key(f, s))["recipe"]["torch_rng_state_sha256_after_training"]
                        for f in C.FAMILIES
                    }
                )
                == 1
                for s in C.SEEDS
            ),
            "detail": {
                str(s): {
                    f: C.load_train_json(C.arm_key(f, s))["recipe"]["torch_rng_state_sha256_after_training"]
                    for f in C.FAMILIES
                }
                for s in C.SEEDS
            },
        },
    ]
    all_passed &= all(bool(chk["passed"]) for chk in global_checks)
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "n_arms": len(lanes),
        "expected_arms": len(C.ARMS),
        "all_passed": bool(all_passed and len(lanes) == len(C.ARMS)),
        "scope": "validation420_only",
        "test_read": False,
        "arms": lanes,
        "global_checks": global_checks,
    }
    C.write_json(C.CHECKPOINT_VERIFY, payload)
    if not payload["all_passed"]:
        failed = [arm for arm, lane in lanes.items() if not lane["passed"]]
        raise SystemExit(f"checkpoint verification failed: {failed}")
    return payload


def _read_curve(path: Path) -> list[dict[str, Any]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return [row for row in csv.DictReader(fh)]


def build_manifest() -> dict[str, Any]:
    if not C.CHECKPOINT_VERIFY.exists():
        raise PermissionError("run `verify` first")
    verify = C.read_json(C.CHECKPOINT_VERIFY)
    if not verify.get("all_passed") or int(verify.get("n_arms", 0)) != len(C.ARMS):
        raise PermissionError("checkpoint verification did not pass for all arms")
    if verify.get("test_read"):
        raise PermissionError("verification record claims Test access; refusing to freeze manifest")
    arms: dict[str, Any] = {}
    for arm in C.ARMS:
        lane = verify["arms"][arm]
        payload = C.load_train_json(arm)
        curve = _read_curve(C.run_dir(arm) / "curve.csv")
        valid_curve = np.asarray([float(row["valid_rmse_raw"]) for row in curve])
        train_curve_std = np.asarray([float(row["train_rmse_standardized"]) for row in curve])
        arms[arm] = {
            "arm": arm,
            "model": lane["model"],
            "family": lane["family"],
            "seed": int(lane["seed"]),
            "checkpoint": lane["checkpoint"],
            "checkpoint_sha256": lane["checkpoint_sha256"],
            "best_epoch": int(lane["best_epoch"]),
            "best_valid_rmse": float(lane["best_valid_rmse"]),
            "best_valid_mae": float(lane["best_valid_mae"]),
            "train_rmse": float(payload["metrics"]["train"]["rmse"]),
            "train_mae": float(payload["metrics"]["train"]["mae"]),
            "valid_r2": float(lane["recomputed_valid"]["r2"]),
            "curve_sha256": payload["checkpoint"]["curve_sha256"],
            "curve_last_epoch": len(curve),
            "curve_min_valid_epoch": int(np.argmin(valid_curve)) + 1,
            "curve_valid_rmse_at_360": float(valid_curve[-1]),
            "curve_train_rmse_standardized_at_360": float(train_curve_std[-1]),
            "checkpoint_is_final_epoch_state": bool(payload["checkpoint"]["checkpoint_is_final_epoch_state"]),
            "parameters": int(payload["model_audit"]["parameters"]),
            "smoke": bool(payload.get("smoke")),
            "git_commit": payload["git_commit"],
        }
    body = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "FROZEN_BEFORE_TEST_EVALUATION",
        "git_commit": C.git_commit(),
        "n_arms": len(arms),
        "any_smoke_runs": bool(any(entry["smoke"] for entry in arms.values())),
        "checkpoint_rule": C.TRAIN_RECIPE["checkpoint"],
        "test_read_at_freeze_time": False,
        "arms": arms,
    }
    body["manifest_sha256"] = C.sha256_json(body)
    C.write_json(C.CHECKPOINT_MANIFEST, body)
    C.write_json(
        C.MANIFEST_FROZEN,
        {
            "manifest_sha256": body["manifest_sha256"],
            "created_utc": body["created_utc"],
            "n_arms": len(arms),
            "scope": "frozen before any Test420 label was read this round",
        },
    )
    return body


def _frozen_manifest() -> dict[str, Any]:
    if not C.CHECKPOINT_MANIFEST.exists() or not C.MANIFEST_FROZEN.exists():
        raise PermissionError("the checkpoint manifest is not frozen; run `manifest` first")
    marker = C.read_json(C.MANIFEST_FROZEN)
    body = C.read_json(C.CHECKPOINT_MANIFEST)
    recomputed = {k: v for k, v in body.items() if k != "manifest_sha256"}
    if C.sha256_json(recomputed) != body["manifest_sha256"]:
        raise PermissionError("checkpoint manifest content changed after the freeze")
    if marker.get("manifest_sha256") != body["manifest_sha256"]:
        raise PermissionError("manifest sha256 does not match the freeze marker")
    if int(body.get("n_arms", 0)) != len(C.ARMS):
        raise PermissionError("frozen manifest does not contain all arms")
    return body


def evaluate_test(device="cpu") -> dict[str, Any]:
    body = _frozen_manifest()
    views = D.load_views(include_test=True)
    if views.get("test") is None:
        raise RuntimeError("Test420 labels were not loaded")
    rows_test = np.asarray(views["split"]["rows"]["test"], dtype=np.int64)
    y_test = np.asarray(views["test"]["y_test"], dtype=np.float64)
    if C.rows_sha256_int64(rows_test.tolist()) != C.SPLIT_SHA256["test"]:
        raise PermissionError("Test420 row order drifted")
    if C.vector_sha256_float64(y_test) != C.TEST_VECTOR_SHA256:
        raise PermissionError("Test420 label vector drifted")
    y_valid = np.asarray(views["labels"]["y_valid"], dtype=np.float64)
    rows_valid = np.asarray(views["split"]["rows"]["valid"], dtype=np.int64)

    C.EVAL_TEST.mkdir(parents=True, exist_ok=True)
    C.PREDICTIONS.mkdir(parents=True, exist_ok=True)
    results: dict[str, Any] = {}
    arrays: dict[str, dict[str, np.ndarray]] = {}
    for arm in C.ARMS:
        frozen = body["arms"][arm]
        entry = {
            "arm": arm,
            "family": frozen["family"],
            "seed": int(frozen["seed"]),
            "run_dir": C.run_dir(arm),
            "checkpoint": C.run_dir(arm) / "checkpoint.pt",
            "checkpoint_sha256": frozen["checkpoint_sha256"],
        }
        if C.sha256_file(entry["checkpoint"]) != frozen["checkpoint_sha256"]:
            raise PermissionError(f"{arm}: checkpoint changed after the manifest freeze")
        model, meta, ckpt = _rebuild(entry, views, device)
        pred_valid = NT.denormalise(M.predict(model, views["valid_cache"], device), views["target_norm"])
        pred_test = NT.denormalise(M.predict(model, views["test_cache"], device), views["target_norm"])
        if pred_test.size != rows_test.size or not np.isfinite(pred_test).all():
            raise ValueError(f"{arm}: Test420 prediction failure")
        valid_metrics = MET.all_metrics(pred_valid, y_valid)
        test_metrics = MET.all_metrics(pred_test, y_test)
        if frozen.get("smoke") is not True and abs(
            float(valid_metrics["rmse"]) - float(frozen["best_valid_rmse"])
        ) > TOL:
            raise RuntimeError(
                f"{arm}: manifest Validation RMSE {frozen['best_valid_rmse']} != recomputed "
                f"{valid_metrics['rmse']}"
            )
        results[arm] = {
            "arm": arm,
            "model": frozen["model"],
            "family": frozen["family"],
            "seed": int(frozen["seed"]),
            "checkpoint": frozen["checkpoint"],
            "checkpoint_sha256": frozen["checkpoint_sha256"],
            "best_epoch": int(frozen["best_epoch"]),
            "parameters": int(meta["parameters"]),
            "train_rmse": float(frozen["train_rmse"]),
            "valid": dict(valid_metrics),
            "test": dict(test_metrics),
            "valid_rows_sha256": C.rows_sha256_int64(rows_valid.tolist()),
            "test_rows_sha256": C.rows_sha256_int64(rows_test.tolist()),
            "test_label_vector_sha256": C.TEST_VECTOR_SHA256,
        }
        arrays[arm] = {"valid_pred": pred_valid, "test_pred": pred_test}
        np.savez(
            C.PREDICTIONS / f"{arm}_valid420.npz",
            rows=rows_valid,
            y_true=y_valid,
            pred=pred_valid,
        )
        np.savez(
            C.PREDICTIONS / f"{arm}_test420.npz",
            rows=rows_test,
            y_true=y_test,
            pred=pred_test,
        )
        C.log(
            f"[test-eval] {arm} epoch={frozen['best_epoch']} valid={valid_metrics['rmse']:.6f} "
            f"test={test_metrics['rmse']:.6f}"
        )

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_dirty_tracked": C.git_dirty(),
        "scope": "EXPOSED_TEST_EXPLORATORY",
        "test_access": "EXPOSED_TEST_EXPLORATORY",
        "test_used_for_selection": False,
        "checkpoint_rule": C.TRAIN_RECIPE["checkpoint"],
        "any_smoke_runs": bool(body.get("any_smoke_runs")),
        "manifest_sha256": body["manifest_sha256"],
        "manifest_frozen_utc": body["created_utc"],
        "device": str(device),
        "test_labels_source": dict(views["test"]["audit"]),
        "target_norm": dict(views["target_norm"]),
        "models": results,
        "alignment": {
            "n_train": int(len(views["split"]["rows"]["train"])),
            "n_valid": int(rows_valid.size),
            "n_test": int(rows_test.size),
            "valid_rows_sha256": C.rows_sha256_int64(rows_valid.tolist()),
            "test_rows_sha256": C.rows_sha256_int64(rows_test.tolist()),
            "overlap_valid_test": int(len(set(rows_valid.tolist()) & set(rows_test.tolist()))),
        },
    }
    C.write_json(C.EVAL_TEST / "metrics.json", payload)
    _write_per_molecule(views, arrays, y_valid, y_test, rows_valid, rows_test)
    return payload


def _write_per_molecule(views, arrays, y_valid, y_test, rows_valid, rows_test) -> None:
    smiles = views["split"]["smiles"]
    for split, rows, y in (("valid420", rows_valid, y_valid), ("test420", rows_test, y_test)):
        columns = ["row_id", "smiles", "y_logD"]
        for arm in C.ARMS:
            columns += [f"pred_{arm}", f"err_{arm}", f"abs_err_{arm}", f"sq_err_{arm}"]
        path = C.EVAL_TEST / f"per_molecule_{split}.csv"
        with open(path, "w", newline="", encoding="utf-8") as fh:
            writer = csv.DictWriter(fh, fieldnames=columns)
            writer.writeheader()
            for i, row in enumerate(rows):
                rec = {"row_id": int(row), "smiles": str(smiles[int(row)]), "y_logD": float(y[i])}
                for arm in C.ARMS:
                    pred = float(arrays[arm][f"{'valid' if split == 'valid420' else 'test'}_pred"][i])
                    err = pred - float(y[i])
                    rec[f"pred_{arm}"] = pred
                    rec[f"err_{arm}"] = err
                    rec[f"abs_err_{arm}"] = abs(err)
                    rec[f"sq_err_{arm}"] = err * err
                writer.writerow(rec)


def write_metrics_csvs() -> None:
    payload = C.read_json(C.EVAL_TEST / "metrics.json")
    fields = ["model", "arm", "seed", "best_epoch", "train_rmse", "valid_rmse", "valid_mae", "valid_r2", "test_rmse", "test_mae", "test_r2"]
    with open(C.TABLE_MAIN, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for arm in C.ARMS:
            row = payload["models"][arm]
            writer.writerow(
                {
                    "model": row["model"],
                    "arm": arm,
                    "seed": row["seed"],
                    "best_epoch": row["best_epoch"],
                    "train_rmse": row["train_rmse"],
                    "valid_rmse": row["valid"]["rmse"],
                    "valid_mae": row["valid"]["mae"],
                    "valid_r2": row["valid"]["r2"],
                    "test_rmse": row["test"]["rmse"],
                    "test_mae": row["test"]["mae"],
                    "test_r2": row["test"]["r2"],
                }
            )


__all__ = ["verify_all", "build_manifest", "evaluate_test", "write_metrics_csvs"]

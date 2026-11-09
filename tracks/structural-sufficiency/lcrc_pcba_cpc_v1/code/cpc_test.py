"""Official Test freeze and the single post-freeze scoring pass for LCRC-CPC.

``freeze`` selects the best-Validation checkpoint with the pre-registered rule,
verifies every hash, replays official Validation AP, predicts Test logits from
the label-free frozen cache, and seals the predictions.  It never opens an
official Test label.

``score`` verifies the seal, reuses the frozen historical W2 / RC-REAL /
W2-DYNAMIC Test logits, then opens the official OGB Test targets through the
inherited ``cc_data.read_official_test_labels`` gate.  All models are scored by
the same code path: official ``ogb.graphproppred.Evaluator("ogbg-molpcba")``
plus an independent per-task sklearn AP recomputation.

The Test split is already exposed.  Role:
``EXPOSED_TEST_ENGINEERING_COMPARISON_SET``.  It is never a model-selection
signal and never a blind generalisation estimate.
"""

from __future__ import annotations

import argparse
import csv
import os
import sys
import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _configure_imports() -> None:
    here = Path(__file__).resolve().parent
    if str(here) not in sys.path:
        sys.path.insert(0, str(here))


_configure_imports()

import torch  # noqa: E402

import cpc_common as C  # noqa: E402
import cc_data as CD  # noqa: E402
import cpc_model as M  # noqa: E402
import cpc_paths as P  # noqa: E402

VALID_REPLAY_TOLERANCE = 1e-4
SELECTION_RULE = "argmax official Validation AP over epochs 1..40 (earliest wins ties)"
PREDICTION_BATCH_SIZE = 256


def _np_sha(array: np.ndarray) -> str:
    return C.sha256_bytes(np.ascontiguousarray(array, dtype=np.float32).tobytes())


def _save_npy_atomic(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, np.ascontiguousarray(array, dtype=np.float32))
    os.replace(tmp, path)
    return C.sha256_file(path)


def _abs_path(value: str | Path) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else C.REPO / path


def run_dir() -> Path:
    return C.RUNS / "LCRC_CPC"


def load_run_json() -> dict[str, Any]:
    path = run_dir() / "run.json"
    if not path.is_file():
        raise RuntimeError(f"run.json missing: {path}")
    return C.read_json(path)


def _load_checkpoint(path: Path, task: Any, device: torch.device) -> tuple[M.CPCModel, dict[str, Any], str]:
    if not path.is_file():
        raise RuntimeError(f"checkpoint missing: {path}")
    sha = C.sha256_file(path)
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("schema") != C.SCHEMA_CHECKPOINT:
        raise RuntimeError(f"checkpoint schema {payload.get('schema')} != {C.SCHEMA_CHECKPOINT}")
    if payload.get("round_id") != C.ROUND_ID:
        raise RuntimeError(f"checkpoint round {payload.get('round_id')} != {C.ROUND_ID}")
    if payload.get("model") != C.MODEL_NAME:
        raise RuntimeError(f"checkpoint model {payload.get('model')} != {C.MODEL_NAME}")
    if payload.get("test_evaluated"):
        raise RuntimeError("checkpoint claims a Test evaluation before the freeze")
    model, audit = M.build_cpc_model(
        task.spec, task.norm, C.N_TASKS, seed=int(payload["seed"]), device=device
    )
    model.load_state_dict(payload["state_dict"], strict=True)
    model.eval()
    return model, payload, sha


def _checkpoint_audit(payload: Mapping[str, Any], task: Any, device: torch.device) -> dict[str, Any]:
    model = M.build_cpc_model(
        task.spec, task.norm, C.N_TASKS, seed=int(payload["seed"]), device="cpu"
    )[0]
    model.load_state_dict(payload["state_dict"], strict=True)
    audit = M.parameter_audit(model)
    return {
        "checkpoint_epoch": int(payload["epoch"]),
        "checkpoint_global_step": int(payload["global_step"]),
        "parameters": audit,
        "feature_spec_hash": payload.get("feature_spec_hash"),
        "split_partition_sha256": payload.get("split_partition_sha256"),
        "label_artifact_sha256": payload.get("label_artifact_sha256"),
        "cache_sha256_manifest": payload.get("cache_sha256_manifest"),
        "cpc_cache_identity": payload.get("cpc_cache_identity"),
        "recipe": payload.get("recipe"),
    }


# ---------------------------------------------------------------------------
# selection manifest
# ---------------------------------------------------------------------------


def build_selection(*, out_path: Path) -> dict[str, Any]:
    run = load_run_json()
    if run.get("status") != "COMPLETED":
        raise RuntimeError(f"run is not COMPLETED: {run.get('status')}")
    if run.get("test_evaluated"):
        raise RuntimeError("run claims a Test evaluation before the freeze")
    if int(run.get("epochs_run", -1)) != int(C.TOTAL_EPOCHS):
        raise RuntimeError(f"run epochs {run.get('epochs_run')} != {C.TOTAL_EPOCHS}")
    if not run.get("all_eval_epochs_completed"):
        raise RuntimeError("not every epoch has an official Validation AP")
    best_path = _abs_path(run["best_checkpoint"])
    epoch40_path = _abs_path(run["primary_checkpoint"])
    for path, sha in (
        (best_path, run["best_checkpoint_sha256"]),
        (epoch40_path, run["primary_checkpoint_sha256"]),
    ):
        if not path.is_file():
            raise RuntimeError(f"checkpoint missing: {path}")
        if C.sha256_file(path) != sha:
            raise RuntimeError(f"checkpoint hash changed: {path}")

    curve = _abs_path(run["curve"])
    rows = list(csv.DictReader(open(curve, encoding="utf-8")))
    ap_by_epoch = {int(r["epoch"]): float(r["valid_ap"]) for r in rows if r["valid_ap"] not in ("", None)}
    if len(ap_by_epoch) != int(C.TOTAL_EPOCHS):
        raise RuntimeError(f"curve has {len(ap_by_epoch)} evaluated epochs, expected {C.TOTAL_EPOCHS}")
    argmax_epoch = max(sorted(ap_by_epoch), key=lambda e: ap_by_epoch[e])
    if int(run["best_epoch"]) != int(argmax_epoch):
        raise RuntimeError(f"best_epoch {run['best_epoch']} != curve argmax {argmax_epoch}")
    if abs(float(run["best_valid_ap"]) - ap_by_epoch[argmax_epoch]) > 1e-12:
        raise RuntimeError("best_valid_ap does not match the curve")

    selection = {
        "schema": "lcrc-pcba-cpc-selection-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "selection_rule": SELECTION_RULE,
        "selection_uses_test": False,
        "test_read": False,
        "test_role": C.TEST_ROLE,
        "models": {
            C.MODEL_NAME: {
                "model": C.MODEL_NAME,
                "parameters": int(run["parameters"]),
                "best_epoch": int(run["best_epoch"]),
                "best_valid_ap": float(run["best_valid_ap"]),
                "best_checkpoint": C.track_rel(best_path),
                "best_checkpoint_sha256": run["best_checkpoint_sha256"],
                "valid_ap_at_40": float(run["fixed_epoch_reports"][str(run["final_epoch"])]["ap"]),
                "epoch40_checkpoint": C.track_rel(epoch40_path),
                "epoch40_checkpoint_sha256": run["primary_checkpoint_sha256"],
                "curve": C.track_rel(curve),
                "curve_sha256": run["curve_sha256"],
                "gpu_hours": float(run["gpu_hours"]),
                "train_ap40": None
                if not run.get("train_eval_final")
                else float(run["train_eval_final"]["ap"]),
                "branch_usage": run.get("branch_usage"),
                "branch_usage_sha256": run.get("branch_usage_sha256"),
            }
        },
        "references": {
            "W2": {
                "parameters": int(C.W2_SEED0["parameters"]),
                "best_epoch": int(C.W2_SEED0["best_epoch"]),
                "best_valid_ap": float(C.W2_SEED0["best_valid_ap"]),
                "valid_ap_at_40": float(C.W2_SEED0["valid_ap_at_40"]),
                "best_checkpoint_sha256": C.W2_SEED0["best_checkpoint_sha256"],
                "epoch40_checkpoint_sha256": C.W2_SEED0["epoch40_checkpoint_sha256"],
                "test_ap_best_brief": float(C.W2_SEED0["test_ap_best"]),
                "test_ap_40_brief": float(C.W2_SEED0["test_ap_at_40"]),
                "gpu_hours": float(C.W2_SEED0["gpu_hours"]),
                "source_round": "lcrc-pcba-backbone-scaling-v1",
            },
            "RC-REAL": {
                "parameters": int(C.RC_REAL["parameters"]),
                "best_epoch": int(C.RC_REAL["best_epoch"]),
                "best_valid_ap": float(C.RC_REAL["best_valid_ap"]),
                "valid_ap_at_40": float(C.RC_REAL["valid_ap_at_40"]),
                "best_checkpoint_sha256": C.RC_REAL["best_checkpoint_sha256"],
                "gpu_hours": float(C.RC_REAL["gpu_hours"]),
                "source_round": "lcrc-pcba-rc-v1",
            },
            "W2-DYNAMIC": {
                "parameters": int(C.DYNAMIC["parameters"]),
                "best_epoch": int(C.DYNAMIC["best_epoch"]),
                "best_valid_ap": float(C.DYNAMIC["best_valid_ap"]),
                "valid_ap_at_40": float(C.DYNAMIC["valid_ap_at_40"]),
                "test_ap_best_brief": float(C.DYNAMIC["test_ap_best"]),
                "test_ap_40_brief": float(C.DYNAMIC["test_ap_at_40"]),
                "best_checkpoint_sha256": C.DYNAMIC["best_checkpoint_sha256"],
                "epoch40_checkpoint_sha256": C.DYNAMIC["epoch40_checkpoint_sha256"],
                "gpu_hours": float(C.DYNAMIC["gpu_hours"]),
                "source_round": "lcrc-pcba-dynamic-v1",
            },
            "frozen_test_logits": C.verify_frozen_reference_logits(),
        },
    }
    C.write_json(out_path, selection)
    return selection


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------


def run_freeze(
    *,
    out_path: Path,
    selection_path: Path,
    predictions_dir: Path,
    device: torch.device,
) -> int:
    task = P.load_task_with_cpc()
    selection = build_selection(out_path=selection_path)
    entry = selection["models"][C.MODEL_NAME]
    test_rows = np.asarray(task.test_rows, dtype=np.int64)
    rows_sha = C.test_rows_sha256_check(test_rows)

    if np.isfinite(np.asarray(task.labels[test_rows])).any():
        raise RuntimeError("official Test labels became visible before the freeze")

    records: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    targets = (
        (
            "LCRC_CPC_best_valid",
            Path(_abs_path(entry["best_checkpoint"])),
            entry["best_valid_ap"],
            entry["best_epoch"],
            entry["best_checkpoint_sha256"],
        ),
        (
            "LCRC_CPC_epoch40",
            Path(_abs_path(entry["epoch40_checkpoint"])),
            entry["valid_ap_at_40"],
            int(C.TOTAL_EPOCHS),
            entry["epoch40_checkpoint_sha256"],
        ),
    )
    for label, path, recorded_valid_ap, epoch, expected_sha in targets:
        t0 = time.perf_counter()
        model, payload, sha = _load_checkpoint(path, task, device)
        if sha != expected_sha:
            raise RuntimeError(f"{label}: checkpoint sha {sha} != selection manifest")
        audit = _checkpoint_audit(payload, task, device)
        t_load = time.perf_counter() - t0
        valid = M.predict_logits(model, task, task.valid_rows, device, batch_size=PREDICTION_BATCH_SIZE)
        y_valid = np.asarray(task.labels[task.valid_rows])
        valid_bce = C.MET.masked_bce_numpy(valid, y_valid)
        valid_report = C.MET.validation_report(y_valid, 1.0 / (1.0 + np.exp(-valid.astype(np.float64))))
        valid_report["valid_bce"] = float(valid_bce)
        delta = abs(float(valid_report["ap"]) - float(recorded_valid_ap))
        if delta > VALID_REPLAY_TOLERANCE:
            raise RuntimeError(
                f"{label}: Validation replay {valid_report['ap']} != recorded {recorded_valid_ap} (delta {delta})"
            )
        t1 = time.perf_counter()
        logits = M.predict_logits(model, task, test_rows, device, batch_size=PREDICTION_BATCH_SIZE)
        t_test = time.perf_counter() - t1
        logits_path = predictions_dir / f"{label}.npy"
        logits_sha = _save_npy_atomic(logits_path, logits)
        records[label] = {
            "label": label,
            "model": C.MODEL_NAME,
            "role": "LCRC-CPC best-Validation checkpoint" if label.endswith("best_valid") else "LCRC-CPC epoch-40 checkpoint",
            "epoch": int(epoch),
            "checkpoint": C.track_rel(path),
            "checkpoint_sha256": sha,
            "checkpoint_payload_epoch": int(audit["checkpoint_epoch"]),
            "checkpoint_global_step": int(audit["checkpoint_global_step"]),
            "parameters": int(audit["parameters"]["total"]),
            "valid_ap_recorded": float(recorded_valid_ap),
            "valid_ap_replay": float(valid_report["ap"]),
            "valid_ap_abs_delta": float(delta),
            "valid_bce_replay": float(valid_report["valid_bce"]),
            "valid_per_task_ap": valid_report["per_task_ap"],
            "valid_observed": valid_report["observed"],
            "valid_positive": valid_report["positive"],
            "valid_negative": valid_report["negative"],
            "valid_logits_sha256": _np_sha(valid),
            "n_tasks_scored_valid": int(valid_report["n_tasks_scored"]),
            "test_logits_path": C.track_rel(logits_path),
            "test_logits_sha256": logits_sha,
            "test_logits_shape": list(logits.shape),
            "test_logits_dtype": "float32",
            "test_logits_min": float(logits.min()),
            "test_logits_max": float(logits.max()),
            "test_logits_mean": float(logits.mean()),
            "test_rows": int(logits.shape[0]),
            "test_n_tasks": int(logits.shape[1]),
            "test_row_ids_sha256": rows_sha,
            "load_seconds": float(t_load),
            "valid_seconds": float(time.perf_counter() - t0 - t_load - t_test),
            "test_predict_seconds": float(t_test),
        }
        del model
        C.log(
            f"[freeze] {label} valid {valid_report['ap']:.9f} (recorded {recorded_valid_ap:.9f}) "
            f"test logits {logits_sha[:12]}"
        )

    rows_path = predictions_dir / "test_row_ids.npy"
    tmp = rows_path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, test_rows)
    os.replace(tmp, rows_path)

    freeze: dict[str, Any] = {
        "schema": C.SCHEMA_TEST_FREEZE,
        "round_id": C.ROUND_ID,
        "test_role": C.TEST_ROLE,
        "logical_models": None,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "test_labels_read": False,
        "test_labels_read_before_freeze": False,
        "selection_manifest": C.track_rel(selection_path),
        "selection_manifest_sha256": C.sha256_file(selection_path),
        "arms_requested": [C.MODEL_NAME],
        "missing_arms": [],
        "test_rows_path": C.track_rel(rows_path),
        "test_rows_sha256": C.sha256_file(rows_path),
        "test_rows_npy_sha256": rows_sha,
        "n_test_rows": int(test_rows.size),
        "n_tasks": int(C.N_TASKS),
        "prediction_batch_size": int(PREDICTION_BATCH_SIZE),
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "environment": C.environment(),
        "predictions": records,
        "unique_predictions": sorted(records),
        "aliases": {},
        "references": selection["references"]["frozen_test_logits"],
        "total_seconds": float(time.perf_counter() - started),
        "note": (
            "Test logits are frozen before any official Test label is opened. Only "
            "cpc_test.py score reads raw/graph-label.csv.gz, through the inherited "
            "cc_data.read_official_test_labels gate. W2 / RC-REAL / W2-DYNAMIC results are "
            "recomputed from the already frozen historical logits with the identical metric code."
        ),
    }
    freeze["seal_sha256"] = C.freeze_seal(freeze)
    C.write_json(out_path, freeze)
    check = C.read_json(out_path)
    if check["seal_sha256"] != C.freeze_seal(check):
        raise RuntimeError("written freeze manifest seal does not verify")
    C.log(f"[freeze] wrote {out_path} seal={freeze['seal_sha256'][:16]}")
    return 0


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _load_prediction(record: Mapping[str, Any], predictions_dir: Path) -> np.ndarray:
    path = _abs_path(record["test_logits_path"])
    if Path(path).resolve().parent != Path(predictions_dir).resolve():
        raise RuntimeError(f"prediction outside the prediction directory: {path}")
    if not path.is_file():
        raise RuntimeError(f"frozen logits missing: {path}")
    sha = C.sha256_file(path)
    if sha != record["test_logits_sha256"]:
        raise RuntimeError(f"frozen logits sha {sha} != manifest {record['test_logits_sha256']}")
    arr = np.load(path)
    if list(arr.shape) != list(record["test_logits_shape"]):
        raise RuntimeError(f"{path} shape {arr.shape} != {record['test_logits_shape']}")
    if arr.dtype != np.float32:
        raise RuntimeError(f"{path} dtype {arr.dtype} != float32")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"{path} contains non-finite logits")
    return arr


def _reference_prediction(record: Mapping[str, Any]) -> tuple[np.ndarray, str]:
    path = _abs_path(record["path"])
    sha = C.sha256_file(path)
    if sha != record["sha256"]:
        raise RuntimeError(f"reference logits sha {sha} != recorded {record['sha256']}")
    arr = np.load(path)
    if list(arr.shape) != list(record["shape"]):
        raise RuntimeError(f"reference shape {arr.shape} != recorded {record['shape']}")
    if not np.isfinite(arr).all():
        raise RuntimeError(f"reference logits contain non-finite values: {path}")
    return arr, sha


def _score_logits(y: np.ndarray, logits: np.ndarray, *, label: str, sha: str, parameters: Any = None) -> dict[str, Any]:
    probs = (1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))).astype(np.float32)
    official = C.MET.official_ap(y, probs)
    independent = C.MET.per_task_ap(y, probs)
    return {
        "label": label,
        "official_ap": float(official["value"]),
        "official_source": official["source"],
        "independent_ap": float(independent["mean_over_scored_tasks"]),
        "independent_source": "sklearn average_precision_score per task, NaN masked",
        "abs_delta_official_vs_independent": float(
            abs(float(official["value"]) - float(independent["mean_over_scored_tasks"]))
        ),
        "n_tasks_scored": int(independent["n_tasks_scored"]),
        "n_tasks_with_observed_labels": int(independent["n_tasks_with_observed_labels"]),
        "per_task_ap": independent["per_task"],
        "observed": independent["observed"],
        "positive": independent["positive"],
        "negative": independent["negative"],
        "test_logits_sha256": sha,
        "parameters": int(parameters) if parameters is not None else None,
    }


def _wins(a: Sequence[Any], b: Sequence[Any]) -> tuple[int, int, int]:
    up = down = both = 0
    for x, y in zip(a, b):
        if x is None or y is None:
            continue
        if not (np.isfinite(x) and np.isfinite(y)):
            continue
        both += 1
        if float(x) > float(y):
            up += 1
        elif float(x) < float(y):
            down += 1
    return up, down, both


def _mean(values: Sequence[Any]) -> float | None:
    vals = [float(v) for v in values if v is not None and np.isfinite(v)]
    return float(np.mean(vals)) if vals else None


def run_score(*, freeze_path: Path, out_path: Path, predictions_dir: Path) -> int:
    freeze = C.read_json(freeze_path)
    if freeze.get("schema") != C.SCHEMA_TEST_FREEZE:
        raise RuntimeError("prediction freeze has the wrong schema")
    if freeze.get("test_labels_read") is not False:
        raise RuntimeError("prediction freeze does not declare test_labels_read=false")
    if freeze.get("seal_sha256") != C.freeze_seal(freeze):
        raise RuntimeError("prediction freeze seal does not verify")
    if set(freeze["unique_predictions"]) != set(freeze["predictions"]):
        raise RuntimeError("freeze predictions are not one unique arm")
    selection = C.read_json(_abs_path(freeze["selection_manifest"]))
    if C.sha256_file(_abs_path(freeze["selection_manifest"])) != freeze["selection_manifest_sha256"]:
        raise RuntimeError("selection manifest hash changed after the freeze")
    entry = selection["models"][C.MODEL_NAME]

    task = P.load_task_with_cpc()
    test_rows = np.asarray(task.test_rows, dtype=np.int64)
    if not np.array_equal(test_rows, np.load(_abs_path(freeze["test_rows_path"]))):
        raise RuntimeError("freeze test row ids do not match the official Test split")
    C.test_rows_sha256_check(test_rows)

    # The inherited gate verifies the freeze, then opens raw/graph-label.csv.gz.
    test_reference = CD.read_official_test_labels(task, freeze_path=freeze_path)
    y = np.asarray(test_reference["labels"], dtype=np.float64)
    if y.shape != (int(test_rows.size), int(C.N_TASKS)):
        raise RuntimeError(f"official Test labels {y.shape} have the wrong shape")

    results: dict[str, Any] = {}
    for label, record in sorted(freeze["predictions"].items()):
        logits = _load_prediction(record, predictions_dir)
        results[label] = _score_logits(
            y,
            logits,
            label=label,
            sha=record["test_logits_sha256"],
            parameters=record["parameters"],
        )
        results[label]["role"] = record["role"]
        results[label]["epoch"] = int(record["epoch"])
        results[label]["source_round"] = C.ROUND_ID
        results[label]["test_logits_path"] = record["test_logits_path"]
        results[label]["valid_ap"] = float(record["valid_ap_recorded"])
        results[label]["valid_ap_replay"] = float(record["valid_ap_replay"])

    reference_logits = C.verify_frozen_reference_logits()
    reference_roles = {
        "W2_best_valid": ("W2", "W2 best-Validation checkpoint", 24, "lcrc-pcba-backbone-scaling-v1"),
        "W2_epoch40": ("W2", "W2 epoch-40 checkpoint", 40, "lcrc-pcba-backbone-scaling-v1"),
        "RC_REAL_best_valid": ("RC-REAL", "RC-REAL best-Validation checkpoint", 33, "lcrc-pcba-rc-v1"),
        "RC_REAL_epoch40": ("RC-REAL", "RC-REAL epoch-40 checkpoint", 40, "lcrc-pcba-rc-v1"),
        "W2_DYNAMIC_best_valid": ("W2-DYNAMIC", "W2-DYNAMIC best-Validation checkpoint", 25, "lcrc-pcba-dynamic-v1"),
        "W2_DYNAMIC_epoch40": ("W2-DYNAMIC", "W2-DYNAMIC epoch-40 checkpoint", 40, "lcrc-pcba-dynamic-v1"),
    }
    for label, record in sorted(reference_logits.items()):
        logits, sha = _reference_prediction(record)
        family, role, epoch, source = reference_roles[label]
        scored = _score_logits(y, logits, label=label, sha=sha)
        scored.update(
            {
                "family": family,
                "role": role,
                "epoch": int(epoch),
                "source_round": source,
                "test_logits_path": record["path"],
            }
        )
        results[label] = scored

    cpc_best = results["LCRC_CPC_best_valid"]
    cpc_40 = results["LCRC_CPC_epoch40"]
    w2_best = results["W2_best_valid"]
    dynamic_best = results["W2_DYNAMIC_best_valid"]
    w2_40 = results["W2_epoch40"]
    dynamic_40 = results["W2_DYNAMIC_epoch40"]

    def recovery(value: float, base: float, top: float) -> float:
        return float((float(value) - float(base)) / (float(top) - float(base)))

    valid_best = float(entry["best_valid_ap"])
    valid_40 = float(entry["valid_ap_at_40"])
    recovery_valid_best = recovery(valid_best, C.W2_SEED0["best_valid_ap"], C.DYNAMIC["best_valid_ap"])
    recovery_valid_40 = recovery(valid_40, C.W2_SEED0["valid_ap_at_40"], C.DYNAMIC["valid_ap_at_40"])
    recovery_test_best = recovery(cpc_best["official_ap"], w2_best["official_ap"], dynamic_best["official_ap"])
    recovery_test_40 = recovery(cpc_40["official_ap"], w2_40["official_ap"], dynamic_40["official_ap"])

    high_mask = [
        i
        for i in range(C.N_TASKS)
        if int(cpc_best["positive"][i]) >= 20 and int(dynamic_best["positive"][i]) >= 20
    ]
    if len(high_mask) != int(C.RECOVERY["high_support_tasks"]):
        raise RuntimeError(f"high-support task set has {len(high_mask)} tasks, expected 106")
    high_cpc = _mean([cpc_best["per_task_ap"][i] for i in high_mask])
    high_w2 = _mean([w2_best["per_task_ap"][i] for i in high_mask])
    high_dynamic = _mean([dynamic_best["per_task_ap"][i] for i in high_mask])
    up, down, both = _wins(cpc_best["per_task_ap"], w2_best["per_task_ap"])
    high_up = sum(
        1
        for i in high_mask
        if cpc_best["per_task_ap"][i] is not None
        and w2_best["per_task_ap"][i] is not None
        and float(cpc_best["per_task_ap"][i]) > float(w2_best["per_task_ap"][i])
    )

    partial = {
        "valid_best_ap": valid_best,
        "valid_ap_at_40": valid_40,
        "test_best_ap": float(cpc_best["official_ap"]),
        "test_ap_at_40": float(cpc_40["official_ap"]),
        "recovery_valid_best": recovery_valid_best,
        "recovery_valid_40": recovery_valid_40,
        "recovery_test_best": recovery_test_best,
        "recovery_test_40": recovery_test_40,
        "target_valid_ap": float(C.RECOVERY["target_valid_ap"]),
        "target_test_ap": float(C.RECOVERY["target_test_ap"]),
        "target_fraction": float(C.RECOVERY["target_fraction"]),
        "high_support_tasks": len(high_mask),
        "high_support_mean_ap_cpc": high_cpc,
        "high_support_mean_ap_w2": high_w2,
        "high_support_mean_ap_dynamic": high_dynamic,
        "high_support_delta_vs_w2": None if high_cpc is None or high_w2 is None else float(high_cpc - high_w2),
        "high_support_delta_vs_dynamic": None
        if high_cpc is None or high_dynamic is None
        else float(high_cpc - high_dynamic),
        "improved_tasks_vs_w2": int(up),
        "degraded_tasks_vs_w2": int(down),
        "comparable_tasks_vs_w2": int(both),
        "improved_high_support_tasks_vs_w2": int(high_up),
        "parameter_delta_vs_w2": int(entry["parameters"]) - int(C.W2_SEED0["parameters"]),
        "parameter_delta_vs_dynamic": int(entry["parameters"]) - int(C.DYNAMIC["parameters"]),
        "gpu_hours": float(entry["gpu_hours"]),
        "gpu_hours_vs_w2": float(entry["gpu_hours"] - C.W2_SEED0["gpu_hours"]),
        "gpu_hours_vs_dynamic": float(entry["gpu_hours"] - C.DYNAMIC["gpu_hours"]),
        "best_epoch": int(entry["best_epoch"]),
        "valid_ap_at_40_minus_best": float(valid_40 - valid_best),
        "test_ap_at_40_minus_best": float(cpc_40["official_ap"] - cpc_best["official_ap"]),
        "validation_to_test_drop_best": float(valid_best - cpc_best["official_ap"]),
        "validation_to_test_drop_dynamic": float(C.DYNAMIC["best_valid_ap"] - dynamic_best["official_ap"]),
        "validation_to_test_drop_w2": float(C.W2_SEED0["best_valid_ap"] - w2_best["official_ap"]),
    }

    out = {
        "schema": C.SCHEMA_TEST_RESULTS,
        "round_id": C.ROUND_ID,
        "test_role": C.TEST_ROLE,
        "test_is_blind": False,
        "test_is_model_selection_signal": False,
        "scored_after_freeze_only": True,
        "freeze_manifest": C.track_rel(freeze_path),
        "freeze_seal_sha256": freeze["seal_sha256"],
        "selection_manifest": freeze["selection_manifest"],
        "selection_manifest_sha256": freeze["selection_manifest_sha256"],
        "test_labels": {
            "path": test_reference["path"],
            "sha256": test_reference["sha256"],
            "shape": test_reference["shape"],
            "observed_entries": int(test_reference["observed_entries"]),
            "positive_entries": int(test_reference["positive_entries"]),
            "negative_entries": int(test_reference["negative_entries"]),
            "test_rows_sha256": test_reference["test_rows_sha256"],
            "freeze_seal_sha256": test_reference["freeze_seal_sha256"],
            "read_after_freeze": True,
        },
        "n_test_rows": int(test_rows.size),
        "n_tasks": int(C.N_TASKS),
        "metric": {
            "selection": "official Validation AP only",
            "test": "OGB official Evaluator(ogbg-molpcba) AP plus independent sklearn per-task AP",
            "nan_policy": "missing labels excluded per task; tasks without both classes skipped",
            "official_python": "ogb.graphproppred.Evaluator",
        },
        "results": results,
        "recovery": partial,
        "decision_inputs": {
            "recovery_targets": dict(C.RECOVERY),
            "decision_rules": dict(C.DECISION_RULES),
            "note": "Test is an exposed engineering comparison set and is not used to alter the model or checkpoint.",
        },
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "environment": C.environment(),
        "created_utc": C.utcnow(),
        "total_seconds": None,
    }
    C.write_json(out_path, out)
    C.log(
        f"[score] CPC best Test AP {cpc_best['official_ap']:.9f} | R_test {recovery_test_best:.4f} | "
        f"R_valid {recovery_valid_best:.4f} | high-support {high_up}/{len(high_mask)}"
    )
    return 0


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", choices=("freeze", "score", "all"), required=True)
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--freeze", default=str(C.ANALYSIS / "test_prediction_manifest.json"))
    parser.add_argument("--selection", default=str(C.ANALYSIS / "test_selection.json"))
    parser.add_argument("--scores", default=str(C.ANALYSIS / "test_scores.json"))
    parser.add_argument("--predictions-dir", default=str(C.PREDICTIONS))
    args = parser.parse_args(argv)
    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise SystemExit("CUDA requested but unavailable")
    if args.mode in ("freeze", "all"):
        run_freeze(
            out_path=Path(args.freeze),
            selection_path=Path(args.selection),
            predictions_dir=Path(args.predictions_dir),
            device=device,
        )
    if args.mode in ("score", "all"):
        run_score(
            freeze_path=Path(args.freeze),
            out_path=Path(args.scores),
            predictions_dir=Path(args.predictions_dir),
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

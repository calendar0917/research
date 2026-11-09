"""Official Test freeze and the single post-freeze scoring pass for W2-DYNAMIC.

``freeze`` picks the unique best-Validation checkpoint (fixed rule), verifies the
checkpoint hashes, replays official Validation AP, writes Test logits from the
**label-free** frozen cache and seals every hash.  It never opens an official
Test label.

``score`` verifies the seal and the selection manifest, hash-verifies the frozen
W2 / RC-REAL reference logits reused from the BOC round, then reads the official
OGB Test labels through the inherited
``cc_data.read_official_test_labels`` gate and scores every model with the same
code path: the official ``ogb.graphproppred.Evaluator("ogbg-molpcba")`` plus an
independent per-task sklearn AP recomputation.

The Test split is already exposed. Role:
``EXPOSED_TEST_ENGINEERING_COMPARISON_SET``, never a blind estimate and never a
model-selection signal.
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

import dy_common as C  # noqa: E402
import dy_mechanism as MECH  # noqa: E402

VALID_REPLAY_TOLERANCE = 1e-4
SELECTION_RULE = "argmax official Validation AP over epochs 1..40 per model (earliest wins ties)"


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _np_sha(array: np.ndarray) -> str:
    return C.sha256_bytes(np.ascontiguousarray(array, dtype=np.float32).tobytes())


def _save_npy_atomic(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, np.ascontiguousarray(array, dtype=np.float32))
    os.replace(tmp, path)
    return C.sha256_file(path)


def _abs_path(value: str) -> Path:
    path = Path(str(value))
    return path if path.is_absolute() else C.REPO / path


def run_dir() -> Path:
    return C.RUNS / "W2_DYNAMIC"


def load_run_json() -> dict[str, Any]:
    path = run_dir() / "run.json"
    if not path.is_file():
        raise RuntimeError(f"run.json missing: {path}")
    return C.read_json(path)


# ---------------------------------------------------------------------------
# selection manifest
# ---------------------------------------------------------------------------


def build_selection(*, out_path: Path) -> dict[str, Any]:
    run = load_run_json()
    if run.get("status") != "COMPLETED":
        raise RuntimeError(f"run is not COMPLETED: {run.get('status')}")
    if run.get("test_evaluated"):
        raise RuntimeError("run claims a Test evaluation before the freeze")
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

    w2_run = C.read_json(C.W2_SEED0["run_json"])
    w2_curve = list(csv.DictReader(open(_abs_path(w2_run["curve"]), encoding="utf-8")))
    w2_ap = {int(r["epoch"]): float(r["valid_ap"]) for r in w2_curve if r["valid_ap"] not in ("", None)}
    if int(w2_run["best_epoch"]) != max(sorted(w2_ap), key=lambda e: w2_ap[e]):
        raise RuntimeError("historical W2 selection rule does not reproduce")

    selection = {
        "schema": "lcrc-pcba-dynamic-selection-v1",
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
                "source_round": "lcrc-pcba-backbone-scaling-v1",
            },
            "RC-REAL": {
                "parameters": int(C.RC_REAL["parameters"]),
                "best_epoch": int(C.RC_REAL["best_epoch"]),
                "best_valid_ap": float(C.RC_REAL["best_valid_ap"]),
                "valid_ap_at_40": float(C.RC_REAL["valid_ap_at_40"]),
                "best_checkpoint_sha256": C.RC_REAL["best_checkpoint_sha256"],
                "source_round": "lcrc-pcba-rc-v1",
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
    task = C.SD.load_task()
    selection = build_selection(out_path=selection_path)
    entry = selection["models"][C.MODEL_NAME]
    test_rows = np.asarray(task.test_rows, dtype=np.int64)
    rows_sha = C.test_rows_sha256_check(test_rows)

    if np.isfinite(np.asarray(task.labels[test_rows])).any():
        raise RuntimeError("official Test labels became visible before the freeze")

    records: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    targets = (
        ("W2_DYNAMIC_best_valid", Path(_abs_path(entry["best_checkpoint"])), entry["best_valid_ap"], entry["best_epoch"]),
        (
            "W2_DYNAMIC_epoch40",
            Path(_abs_path(entry["epoch40_checkpoint"])),
            entry["valid_ap_at_40"],
            int(C.TOTAL_EPOCHS),
        ),
    )
    for label, path, recorded_valid_ap, epoch in targets:
        t0 = time.perf_counter()
        model, audit, sha = MECH.load_frozen(path, task, device)
        if sha != (entry["best_checkpoint_sha256"] if label.endswith("best_valid") else entry["epoch40_checkpoint_sha256"]):
            raise RuntimeError(f"{label}: checkpoint sha {sha} != selection manifest")
        t_load = time.perf_counter() - t0
        valid = MECH.predict_with(model, task, task.valid_rows, device, message="real")
        y_valid = np.asarray(task.labels[task.valid_rows])
        valid_report = MECH.report_from_logits(valid, y_valid)
        delta = abs(float(valid_report["ap"]) - float(recorded_valid_ap))
        if delta > VALID_REPLAY_TOLERANCE:
            raise RuntimeError(
                f"{label}: Validation replay {valid_report['ap']} != recorded {recorded_valid_ap} (delta {delta})"
            )
        t1 = time.perf_counter()
        logits = MECH.predict_with(model, task, test_rows, device, message="real")
        t_test = time.perf_counter() - t1
        logits_path = predictions_dir / f"{label}.npy"
        logits_sha = _save_npy_atomic(logits_path, logits)
        records[label] = {
            "label": label,
            "model": C.MODEL_NAME,
            "role": "W2-DYNAMIC best-Validation checkpoint" if label.endswith("best_valid") else "W2-DYNAMIC epoch-40 checkpoint",
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
        "prediction_batch_size": 256,
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
            "dy_test.py score reads raw/graph-label.csv.gz, through the inherited "
            "cc_data.read_official_test_labels gate. W2 / RC-REAL results are recomputed from the "
            "already frozen BOC logits with the identical metric code."
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


def _valid_per_task(run: Mapping[str, Any], epoch: int) -> dict[str, Any]:
    """Official per-task Validation AP of a historical run at ``epoch``."""
    report = run["fixed_epoch_reports"][str(int(epoch))]
    return {
        "per_task_ap": list(report["per_task_ap"]),
        "positive": [int(x) for x in report["positive"]],
        "observed": [int(x) for x in report["observed"]],
        "ap": float(report["ap"]),
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


def _stratified(
    per_task_a: Sequence[Any],
    per_task_b: Sequence[Any],
    support: Sequence[int],
    *,
    min_support: int,
    max_support: int | None = None,
) -> dict[str, Any]:
    deltas = []
    improved = 0
    for a, b, s in zip(per_task_a, per_task_b, support):
        if a is None or b is None or int(s) < int(min_support):
            continue
        if max_support is not None and int(s) >= int(max_support):
            continue
        deltas.append(float(a) - float(b))
        improved += int(float(a) > float(b))
    if not deltas:
        return {"n_tasks": 0, "mean_delta": None, "median_delta": None, "improved": 0}
    arr = np.asarray(deltas, dtype=np.float64)
    return {
        "n_tasks": int(arr.size),
        "mean_delta": float(arr.mean()),
        "median_delta": float(np.median(arr)),
        "improved": int(improved),
        "degraded": int((arr < 0).sum()),
        "worst_delta": float(arr.min()),
        "best_delta": float(arr.max()),
    }


def _score_one(logits: np.ndarray, y_test: np.ndarray) -> dict[str, Any]:
    probs = (1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))).astype(np.float32)
    official = C.MET.official_ap(y_test, probs)
    independent = C.MET.per_task_ap(y_test, probs)
    return {
        "test_ap_official": float(official["value"]),
        "test_ap_independent": float(independent["mean_over_scored_tasks"]),
        "test_ap_abs_delta": float(abs(float(official["value"]) - float(independent["mean_over_scored_tasks"]))),
        "test_bce": float(C.MET.masked_bce_numpy(logits, y_test)),
        "test_per_task_ap": independent["per_task"],
        "test_observed": independent["observed"],
        "test_positive": independent["positive"],
        "test_negative": independent["negative"],
        "n_tasks_scored": int(independent["n_tasks_scored"]),
        "n_tasks_with_observed_labels": int(independent["n_tasks_with_observed_labels"]),
    }


def run_score(
    *,
    freeze_path: Path,
    out_path: Path,
    per_task_path: Path,
    predictions_dir: Path,
) -> int:
    freeze = C.read_json(freeze_path)
    if freeze.get("schema") != C.SCHEMA_TEST_FREEZE:
        raise RuntimeError("wrong freeze schema")
    if freeze.get("test_labels_read") is not False:
        raise RuntimeError("freeze manifest does not declare test_labels_read=false")
    if freeze.get("seal_sha256") != C.freeze_seal(freeze):
        raise RuntimeError("freeze seal does not verify")
    selection_path = _abs_path(freeze["selection_manifest"])
    if C.sha256_file(selection_path) != freeze["selection_manifest_sha256"]:
        raise RuntimeError("selection manifest changed after the prediction freeze")

    task = C.SD.load_task()
    import cc_data as CD  # the inherited, gated official-Test-label reader

    test_reference = CD.read_official_test_labels(task, freeze_path=freeze_path)
    y_test = np.asarray(test_reference["labels"])
    if y_test.shape != (C.N_TEST, C.N_TASKS):
        raise RuntimeError(f"official Test label matrix {y_test.shape}")

    records = freeze["predictions"]
    for record in records.values():
        _load_prediction(record, predictions_dir)

    # reference models use the same metric code on the frozen BOC logits
    refs = C.verify_frozen_reference_logits()
    scored: dict[str, Any] = {}
    for label, record in records.items():
        logits = _load_prediction(record, predictions_dir)
        scored[label] = {
            "model": record["model"],
            "role": record["role"],
            "epoch": int(record["epoch"]),
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "valid_ap": float(record["valid_ap_replay"]),
            "n_tasks_scored_valid": int(record["n_tasks_scored_valid"]),
            "valid_per_task_ap": record["valid_per_task_ap"],
            "valid_positive": record["valid_positive"],
            "valid_observed": record["valid_observed"],
            "test_logits_sha256": record["test_logits_sha256"],
            "test_logits_path": record["test_logits_path"],
            **_score_one(logits, y_test),
        }
        scored[label]["valid_test_ap_delta"] = scored[label]["test_ap_official"] - scored[label]["valid_ap"]
        C.log(f"[score] {label} valid {scored[label]['valid_ap']:.9f} test {scored[label]['test_ap_official']:.9f}")

    reference_meta = {
        "W2_best_valid": (C.W2_SEED0["best_valid_ap"], C.W2_SEED0["best_epoch"], C.W2_SEED0["run_json"], "W2"),
        "W2_epoch40": (C.W2_SEED0["valid_ap_at_40"], C.TOTAL_EPOCHS, C.W2_SEED0["run_json"], "W2"),
        "RC_REAL_best_valid": (C.RC_REAL["best_valid_ap"], C.RC_REAL["best_epoch"], C.RC_ROUND / "runs/REAL/run.json", "RC-REAL"),
        "RC_REAL_epoch40": (C.RC_REAL["valid_ap_at_40"], C.TOTAL_EPOCHS, C.RC_ROUND / "runs/REAL/run.json", "RC-REAL"),
    }
    for label, rec in refs.items():
        logits = np.load(_abs_path(rec["path"]))
        valid_ap, epoch, run_json, model_name = reference_meta[label]
        run = C.read_json(run_json)
        valid_per_task = _valid_per_task(run, int(epoch))
        scored[label] = {
            "model": model_name,
            "role": "frozen BOC-round reference prediction (not retrained here)",
            "epoch": int(epoch),
            "checkpoint": rec["path"],
            "checkpoint_sha256": rec["sha256"],
            "valid_ap": float(valid_ap),
            "n_tasks_scored_valid": 127,
            "valid_per_task_ap": valid_per_task["per_task_ap"],
            "valid_positive": valid_per_task["positive"],
            "valid_observed": valid_per_task["observed"],
            "test_logits_sha256": rec["sha256"],
            "test_logits_path": rec["path"],
            **_score_one(logits, y_test),
        }
        scored[label]["valid_test_ap_delta"] = scored[label]["test_ap_official"] - scored[label]["valid_ap"]
        C.log(f"[score] {label} valid {scored[label]['valid_ap']:.9f} test {scored[label]['test_ap_official']:.9f}")

    comparisons: dict[str, Any] = {}
    for new in ("W2_DYNAMIC_best_valid", "W2_DYNAMIC_epoch40"):
        for old in ("W2_best_valid", "W2_epoch40", "RC_REAL_best_valid", "RC_REAL_epoch40"):
            up, down, both = _wins(scored[new]["test_per_task_ap"], scored[old]["test_per_task_ap"])
            comparisons[f"{new}_minus_{old}"] = {
                "delta": scored[new]["test_ap_official"] - scored[old]["test_ap_official"],
                "delta_independent": scored[new]["test_ap_independent"]
                - scored[old]["test_ap_independent"],
                "valid_delta": scored[new]["valid_ap"] - scored[old]["valid_ap"],
                "tasks_improved": up,
                "tasks_degraded": down,
                "tasks_scored_both": both,
            }

    support_test = np.asarray(scored["W2_best_valid"]["test_positive"], dtype=np.int64)
    strata: dict[str, Any] = {}
    for arm in ("W2_DYNAMIC_best_valid", "W2_DYNAMIC_epoch40"):
        for ref_label in ("W2_best_valid", "RC_REAL_best_valid"):
            for name, lo, hi in (("ge20", 20, None), ("5_19", 5, 20), ("lt5", 0, 5)):
                strata[f"{arm}_vs_{ref_label}_{name}"] = _stratified(
                    scored[arm]["test_per_task_ap"],
                    scored[ref_label]["test_per_task_ap"],
                    support_test,
                    min_support=lo,
                    max_support=hi,
                )

    test_obs = np.isfinite(y_test)
    results: dict[str, Any] = {
        "schema": C.SCHEMA_TEST_RESULTS,
        "round_id": C.ROUND_ID,
        "generated_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "test_role": C.TEST_ROLE,
        "freeze_manifest": C.track_rel(freeze_path),
        "freeze_seal_sha256": freeze["seal_sha256"],
        "selection_manifest": freeze["selection_manifest"],
        "selection_manifest_sha256": freeze["selection_manifest_sha256"],
        "official_evaluator": "ogb.graphproppred.Evaluator('ogbg-molpcba')",
        "independent_evaluator": "sklearn average_precision_score per task with NaN masking",
        "test_label_artifact": {k: v for k, v in test_reference.items() if k != "labels"},
        "models": scored,
        "comparisons": comparisons,
        "support_strata": strata,
        "label_stats": {
            "test": {
                "rows": int(task.test_rows.size),
                "observed_entries": int(test_obs.sum()),
                "positive_entries": int((y_test[test_obs] == 1.0).sum()),
                "positive_rate_observed": float((y_test[test_obs] == 1.0).sum() / test_obs.sum()),
            }
        },
        "n_test_rows": int(C.N_TEST),
        "n_tasks": int(C.N_TASKS),
        "n_tasks_scored_test": int(scored["W2_best_valid"]["n_tasks_scored"]),
        "test_label_masking": "NaN entries excluded per task; a task is scored only with both classes present",
        "standardisation": "Train-only normalisation inherited from the frozen cache; nothing refitted",
        "checkpoint_selection": SELECTION_RULE,
        "test_exposure_history": {
            "exposed_by": "lcrc_pcba_continuation_recipe_v1 (2026-11-06) read the official Test labels after its own freeze",
            "this_round_role": C.TEST_ROLE,
            "blind_generalisation_estimate": False,
            "no_test_based_selection": True,
            "no_retraining_after_seeing_test": True,
        },
        "descriptive_only": True,
    }
    C.write_json(out_path, results)
    _write_per_task_csv(per_task_path, scored, y_test, task)
    C.log(f"[score] wrote {out_path} and {per_task_path}")
    return 0


def _write_per_task_csv(
    path: Path, scored: Mapping[str, Any], y_test: np.ndarray, task: Any
) -> None:
    labels = [
        "W2_DYNAMIC_best_valid",
        "W2_DYNAMIC_epoch40",
        "W2_best_valid",
        "W2_epoch40",
        "RC_REAL_best_valid",
        "RC_REAL_epoch40",
    ]
    fields = ["task", "test_observed", "test_positive", "support_stratum", "valid_positive"]
    fields += [f"test_ap_{label}" for label in labels]
    fields += [
        "test_delta_dynamic_best_minus_W2_best",
        "test_delta_dynamic_best_minus_RC_REAL_best",
        "validation_ap_dynamic_best",
        "validation_ap_W2_best",
        "validation_ap_RC_REAL_best",
        "validation_positive",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    obs = np.isfinite(y_test)
    valid_pos = np.asarray(scored["W2_DYNAMIC_best_valid"]["valid_positive"], dtype=np.int64)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for t in range(C.N_TASKS):
            pos = int(((y_test[:, t] == 1.0) & obs[:, t]).sum())
            row: dict[str, Any] = {
                "task": t,
                "test_observed": int(obs[:, t].sum()),
                "test_positive": pos,
                "support_stratum": "test_ge20" if pos >= 20 else "test_5_19" if pos >= 5 else "test_lt5",
                "valid_positive": int(valid_pos[t]),
            }
            for label in labels:
                ap = scored[label]["test_per_task_ap"][t]
                row[f"test_ap_{label}"] = "" if ap is None else float(ap)
            a = scored["W2_DYNAMIC_best_valid"]["test_per_task_ap"][t]
            b = scored["W2_best_valid"]["test_per_task_ap"][t]
            c = scored["RC_REAL_best_valid"]["test_per_task_ap"][t]
            row["test_delta_dynamic_best_minus_W2_best"] = "" if a is None or b is None else float(a - b)
            row["test_delta_dynamic_best_minus_RC_REAL_best"] = "" if a is None or c is None else float(a - c)
            for key, label in (
                ("validation_ap_dynamic_best", "W2_DYNAMIC_best_valid"),
                ("validation_ap_W2_best", "W2_best_valid"),
                ("validation_ap_RC_REAL_best", "RC_REAL_best_valid"),
            ):
                value = scored[label]["valid_per_task_ap"][t]
                row[key] = "" if value is None else float(value)
            row["validation_positive"] = int(valid_pos[t])
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    fz = sub.add_parser("freeze")
    fz.add_argument("--out", default=str(C.ANALYSIS / "test_prediction_manifest.json"))
    fz.add_argument("--selection", default=str(C.ANALYSIS / "selected_checkpoint.json"))
    fz.add_argument("--predictions-dir", default=str(C.PREDICTIONS))
    fz.add_argument("--device", default="cuda")
    sc = sub.add_parser("score")
    sc.add_argument("--freeze", default=str(C.ANALYSIS / "test_prediction_manifest.json"))
    sc.add_argument("--out", default=str(C.ANALYSIS / "test_scores.json"))
    sc.add_argument("--per-task", default=str(C.ANALYSIS / "per_task_valid_test.csv"))
    sc.add_argument("--predictions-dir", default=str(C.PREDICTIONS))
    args = parser.parse_args(argv)
    if args.mode == "freeze":
        return run_freeze(
            out_path=Path(args.out),
            selection_path=Path(args.selection),
            predictions_dir=Path(args.predictions_dir),
            device=torch.device(args.device),
        )
    return run_score(
        freeze_path=Path(args.freeze),
        out_path=Path(args.out),
        per_task_path=Path(args.per_task),
        predictions_dir=Path(args.predictions_dir),
    )


if __name__ == "__main__":
    raise SystemExit(main())

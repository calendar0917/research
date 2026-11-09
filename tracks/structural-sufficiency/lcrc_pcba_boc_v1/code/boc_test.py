"""Official Test freeze and the single post-freeze scoring pass for BOC v1.

``freeze`` replays official Validation AP for the two BOC arms plus the
historical W2 and RC-REAL checkpoints, writes Test logits from the **label-free**
structural cache and seals every hash.  It never opens an official Test label.

``score`` verifies the seal, the selection manifest and every prediction hash,
then reads the official OGB Test labels through the inherited
``cc_data.read_official_test_labels`` gate and scores with the official
``ogb.graphproppred.Evaluator("ogbg-molpcba")`` plus an independent per-task AP.

The Test split is already exposed.  Role:
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

CODE = Path(__file__).resolve().parent
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import torch  # noqa: E402

import boc_common as C  # noqa: E402
import boc_model as M  # noqa: E402
import boc_structures as S  # noqa: E402
import boc_train as T  # noqa: E402

CC_CODE = C.CC_ROUND / "code"
RC_CODE = C.RC_ROUND / "code"
for _p in (str(CC_CODE), str(RC_CODE)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import cc_common as CC  # noqa: E402
import cc_data as CD  # noqa: E402

PREDICTIONS = C.ANALYSIS / "test_predictions"
FREEZE_PATH = C.ANALYSIS / "test_prediction_manifest.json"
SCORES_PATH = C.ANALYSIS / "test_scores.json"
PER_TASK_PATH = C.ANALYSIS / "per_task_valid_test.csv"
SELECTION_PATH = C.ANALYSIS / "selected_checkpoint.json"
VALID_REPLAY_TOLERANCE = 1e-4
BOC_ARMS = ("FULL", "BOND")


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


def run_dir(arm: str) -> Path:
    return C.RUNS / arm


def load_run_json(arm: str) -> dict[str, Any]:
    path = run_dir(arm) / "run.json"
    if not path.is_file():
        raise RuntimeError(f"run.json missing for {arm}: {path}")
    return C.read_json(path)


# ---------------------------------------------------------------------------
# selection manifest
# ---------------------------------------------------------------------------


def build_selection(*, out_path: Path = SELECTION_PATH, arms: Sequence[str] = BOC_ARMS) -> dict[str, Any]:
    arms = tuple(arms)
    if not arms or any(a not in BOC_ARMS for a in arms):
        raise KeyError(arms)
    models: dict[str, Any] = {}
    for key in arms:
        run = load_run_json(key)
        if run.get("status") != "COMPLETED":
            raise RuntimeError(f"{key} run is not COMPLETED: {run.get('status')}")
        if run.get("test_evaluated"):
            raise RuntimeError(f"{key} claims a Test evaluation before the freeze")
        best_path = _abs_path(run["best_checkpoint"])
        epoch40_path = _abs_path(run["primary_checkpoint"])
        for path, sha in (
            (best_path, run["best_checkpoint_sha256"]),
            (epoch40_path, run["primary_checkpoint_sha256"]),
        ):
            if not path.is_file():
                raise RuntimeError(f"{key} checkpoint missing: {path}")
            if C.sha256_file(path) != sha:
                raise RuntimeError(f"{key} checkpoint hash changed: {path}")
        curve = run.get("train_curve") or run.get("curve")
        models[key] = {
            "model": key,
            "parameters": int(run["parameters"]),
            "best_epoch": int(run["best_epoch"]),
            "best_valid_ap": float(run["best_valid_ap"]),
            "valid_ap_at_40": float(run["fixed_epoch_reports"][str(C.TOTAL_EPOCHS)]["ap"]),
            "best_checkpoint": run["best_checkpoint"],
            "best_checkpoint_sha256": run["best_checkpoint_sha256"],
            "epoch40_checkpoint": run["primary_checkpoint"],
            "epoch40_checkpoint_sha256": run["primary_checkpoint_sha256"],
            "gpu_hours": float(run["gpu_hours"]),
            "train_ap40": float((run.get("train_eval_final") or {}).get("ap", float("nan"))),
            "gpu_peak_allocated_bytes": int(run.get("gpu_peak_allocated_bytes", 0)),
            "gpu_peak_reserved_bytes": int(run.get("gpu_peak_reserved_bytes", 0)),
            "epoch_seconds_mean": float(run.get("epoch_seconds_mean", 0.0)),
            "curve": curve,
            "curve_sha256": run.get("curve_sha256"),
        }
    selection = {
        "schema": "lcrc-pcba-boc-selection-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "test_read": False,
        "selection_uses_test": False,
        "selection_rule": "argmax official Validation AP over epochs 1..40 per arm (earliest wins ties)",
        "arms_requested": list(arms),
        "missing_arms": [k for k in BOC_ARMS if k not in arms],
        "models": models,
        "references": {
            "W2": {
                "parameters": int(C.W2["parameters"]),
                "best_epoch": int(C.W2["best_epoch"]),
                "best_valid_ap": float(C.W2["best_valid_ap"]),
                "valid_ap_at_40": float(C.W2["ap40"]),
                "best_checkpoint": C.track_rel(C.W2_CHECKPOINT_BEST),
                "best_checkpoint_sha256": C.W2["best_checkpoint_sha256"],
                "epoch40_checkpoint": C.track_rel(C.W2_CHECKPOINT_EPOCH40),
                "epoch40_checkpoint_sha256": C.W2["epoch40_checkpoint_sha256"],
                "gpu_hours": float(C.W2["gpu_hours"]),
                "test_ap_best_brief": float(C.W2_TEST_AP_BRIEF),
                "test_ap_40_brief": float(C.W2_TEST_AP_40_BRIEF),
            },
            "RC-REAL": {
                "parameters": int(C.RC["parameters"]),
                "best_epoch": int(C.RC["best_epoch"]),
                "best_valid_ap": float(C.RC["best_valid_ap"]),
                "valid_ap_at_40": float(C.RC["ap40"]),
                "best_checkpoint": C.track_rel(C.RC["best_checkpoint"]),
                "best_checkpoint_sha256": C.RC["best_checkpoint_sha256"],
                "epoch40_checkpoint": C.track_rel(C.RC["epoch40_checkpoint"]),
                "epoch40_checkpoint_sha256": C.RC["epoch40_checkpoint_sha256"],
                "gpu_hours": float(C.RC["gpu_hours"]),
                "test_ap_best": float(C.RC["test_ap_best"]),
                "test_ap_40": float(C.RC["test_ap_40"]),
            },
        },
        "test_role": C.TEST_ROLE,
    }
    C.write_json(out_path, selection)
    return selection


# ---------------------------------------------------------------------------
# reference loaders / predictors
# ---------------------------------------------------------------------------


def _load_rc_modules():
    import rc_dist as RD
    import rc_model as RM
    import rc_train as RT

    return RD, RM, RT


def restore_model(entry: Mapping[str, Any], task: Any, struct: Mapping[str, Any], device: torch.device):
    path = _abs_path(entry["path"])
    if not path.is_file():
        raise RuntimeError(f"checkpoint missing: {path}")
    sha = C.sha256_file(path)
    if sha != entry["sha256"]:
        raise RuntimeError(f"{entry['label']} checkpoint sha {sha} != recorded {entry['sha256']}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    key = str(entry["model"])
    if key == "W2":
        model, audit = C.BSM.build_model(key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    elif key == "RC-REAL":
        RD, RM, _RT = _load_rc_modules()
        model, audit = RM.build_model("REAL", task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    elif key in BOC_ARMS:
        model, audit = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=C.SEED, device="cpu")
    else:
        raise KeyError(key)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{entry['label']} state mismatch: missing={missing} unexpected={unexpected}")
    if int(payload.get("epoch", -1)) != int(entry["epoch"]):
        raise RuntimeError(f"{entry['label']} epoch {payload.get('epoch')} != {entry['epoch']}")
    model.to(device)
    model.eval()
    return model, payload, audit, sha


def predict_rows(
    model: torch.nn.Module,
    model_key: str,
    task: Any,
    struct: Mapping[str, Any],
    rows: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 256,
    rc_dist: Mapping[str, Any] | None = None,
) -> np.ndarray:
    if model_key in BOC_ARMS:
        return T.predict_logits(model, task, struct, rows, device, batch_size=batch_size)
    if model_key == "W2":
        return C.BC.PC.predict_logits(model, task.cache, rows, device, batch_size=batch_size)
    if model_key == "RC-REAL":
        RD, _RM, RT = _load_rc_modules()
        if rc_dist is None:
            rc_dist = RD.load_cache()
        return RT.predict_logits(model, task, rc_dist, rows, device, batch_size=batch_size)
    raise KeyError(model_key)


def validation_replay(
    model: torch.nn.Module,
    model_key: str,
    task: Any,
    struct: Mapping[str, Any],
    device: torch.device,
    *,
    rc_dist: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    rows = np.asarray(task.valid_rows, dtype=np.int64)
    logits = predict_rows(model, model_key, task, struct, rows, device, rc_dist=rc_dist)
    y = np.asarray(task.labels[rows])
    probs = C.BC.PC.sigmoid(logits)
    report = C.MET.validation_report(y, probs)
    report.update(
        {
            "valid_logits_sha256": _np_sha(logits),
            "valid_bce": float(C.MET.masked_bce_numpy(logits, y)),
            "evaluated_rows": int(rows.size),
        }
    )
    return report


# ---------------------------------------------------------------------------
# candidate checkpoints
# ---------------------------------------------------------------------------


def candidate_models(selection: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if selection.get("test_read") is not False:
        raise RuntimeError("selection manifest must declare test_read=false")
    if bool(selection.get("selection_uses_test")):
        raise RuntimeError("selection manifest claims test-based selection")
    out: dict[str, dict[str, Any]] = {}
    for key in selection.get("arms_requested", BOC_ARMS):
        entry = selection["models"][key]
        out[f"BOC_{key}_best_valid"] = {
            "label": f"BOC_{key}_best_valid",
            "model": key,
            "epoch": int(entry["best_epoch"]),
            "path": entry["best_checkpoint"],
            "sha256": entry["best_checkpoint_sha256"],
            "valid_ap": float(entry["best_valid_ap"]),
            "role": f"{C.MODELS[key]['name']} best-Validation checkpoint",
        }
        out[f"BOC_{key}_epoch40"] = {
            "label": f"BOC_{key}_epoch40",
            "model": key,
            "epoch": int(C.TOTAL_EPOCHS),
            "path": entry["epoch40_checkpoint"],
            "sha256": entry["epoch40_checkpoint_sha256"],
            "valid_ap": float(entry["valid_ap_at_40"]),
            "role": f"{C.MODELS[key]['name']} fixed epoch-40 endpoint",
        }
    refs = selection["references"]
    w2 = refs["W2"]
    out["W2_best_valid"] = {
        "label": "W2_best_valid",
        "model": "W2",
        "epoch": int(w2["best_epoch"]),
        "path": w2["best_checkpoint"],
        "sha256": w2["best_checkpoint_sha256"],
        "valid_ap": float(w2["best_valid_ap"]),
        "role": "historical W2 best-Validation checkpoint (same-pipeline replay)",
    }
    out["W2_epoch40"] = {
        "label": "W2_epoch40",
        "model": "W2",
        "epoch": int(C.TOTAL_EPOCHS),
        "path": w2["epoch40_checkpoint"],
        "sha256": w2["epoch40_checkpoint_sha256"],
        "valid_ap": float(w2["valid_ap_at_40"]),
        "role": "historical W2 epoch-40 checkpoint (same-pipeline replay)",
    }
    rc = refs["RC-REAL"]
    out["RC_REAL_best_valid"] = {
        "label": "RC_REAL_best_valid",
        "model": "RC-REAL",
        "epoch": int(rc["best_epoch"]),
        "path": rc["best_checkpoint"],
        "sha256": rc["best_checkpoint_sha256"],
        "valid_ap": float(rc["best_valid_ap"]),
        "role": "historical RC-REAL best-Validation checkpoint (same-pipeline replay)",
    }
    out["RC_REAL_epoch40"] = {
        "label": "RC_REAL_epoch40",
        "model": "RC-REAL",
        "epoch": int(C.TOTAL_EPOCHS),
        "path": rc["epoch40_checkpoint"],
        "sha256": rc["epoch40_checkpoint_sha256"],
        "valid_ap": float(rc["valid_ap_at_40"]),
        "role": "historical RC-REAL epoch-40 checkpoint (same-pipeline replay)",
    }
    return out


# ---------------------------------------------------------------------------
# freeze
# ---------------------------------------------------------------------------


def run_freeze(
    out_path: Path,
    predictions_dir: Path,
    device: torch.device,
    *,
    struct_cache: str = str(C.STRUCT_CACHE_DIR),
    arms: Sequence[str] = BOC_ARMS,
    selection_path: Path = SELECTION_PATH,
) -> int:
    selection = build_selection(arms=arms, out_path=selection_path)
    task = C.BC.SD.load_task()
    struct = S.load_cache(struct_cache)
    if int(struct["meta"]["n_rows"]) != C.N_MOLECULES:
        raise RuntimeError("freeze requires the full label-free structure cache")
    RD, _RM, _RT = _load_rc_modules()
    rc_dist = RD.load_cache()
    test_rows = np.asarray(task.test_rows, dtype=np.int64)
    if np.intersect1d(test_rows, np.asarray(task.train_rows, np.int64)).size or np.intersect1d(
        test_rows, np.asarray(task.valid_rows, np.int64)
    ).size:
        raise RuntimeError("Test rows overlap Train/Valid")
    if np.isfinite(task.labels[test_rows]).any():
        raise RuntimeError("the training label artifact exposes Test targets")
    for key in arms:
        run = load_run_json(key)
        if str(run.get("status")) != "COMPLETED":
            raise RuntimeError(f"{key} did not complete; refusing to freeze")

    models = candidate_models(selection)
    records: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for label, entry in models.items():
        t0 = time.perf_counter()
        model, payload, audit, sha = restore_model(entry, task, struct, device)
        t_load = time.perf_counter() - t0
        t1 = time.perf_counter()
        valid = validation_replay(model, entry["model"], task, struct, device, rc_dist=rc_dist)
        t_valid = time.perf_counter() - t1
        delta = abs(float(valid["ap"]) - float(entry["valid_ap"]))
        if delta > VALID_REPLAY_TOLERANCE:
            raise RuntimeError(
                f"{label} Validation replay {valid['ap']} != recorded {entry['valid_ap']} (delta {delta})"
            )
        t2 = time.perf_counter()
        logits = predict_rows(
            model, entry["model"], task, struct, test_rows, device, rc_dist=rc_dist
        )
        t_test = time.perf_counter() - t2
        logits_path = predictions_dir / f"{label}.npy"
        logits_sha = _save_npy_atomic(logits_path, logits)
        records[label] = {
            "label": label,
            "model": entry["model"],
            "role": entry["role"],
            "epoch": int(entry["epoch"]),
            "checkpoint": entry["path"],
            "checkpoint_sha256": sha,
            "checkpoint_payload_epoch": int(payload.get("epoch", -1)),
            "checkpoint_global_step": int(payload.get("global_step", -1)),
            "valid_ap_recorded": float(entry["valid_ap"]),
            "valid_ap_replay": float(valid["ap"]),
            "valid_ap_abs_delta": float(delta),
            "valid_bce_replay": float(valid["valid_bce"]),
            "valid_per_task_ap": valid["per_task_ap"],
            "valid_observed": valid["observed"],
            "valid_positive": valid["positive"],
            "valid_negative": valid["negative"],
            "valid_logits_sha256": valid["valid_logits_sha256"],
            "n_tasks_scored_valid": int(valid["n_tasks_scored"]),
            "test_logits_path": C.track_rel(logits_path),
            "test_logits_sha256": logits_sha,
            "test_logits_shape": list(logits.shape),
            "test_logits_dtype": "float32",
            "test_logits_min": float(logits.min()),
            "test_logits_max": float(logits.max()),
            "test_logits_mean": float(logits.mean()),
            "test_rows": int(logits.shape[0]),
            "test_n_tasks": int(logits.shape[1]),
            "replay_audit": {
                "parameters": int(audit.get("total_parameters", audit.get("parameters"))),
                "base_state_sha256": audit.get("base_state_sha256"),
                "model_state_sha256": audit.get("model_state_sha256"),
            },
            "load_seconds": float(t_load),
            "valid_seconds": float(t_valid),
            "test_predict_seconds": float(t_test),
        }
        del model
        C.log(
            f"[freeze] {label} valid {valid['ap']:.9f} (recorded {entry['valid_ap']:.9f}) "
            f"test logits {logits_sha[:12]}"
        )

    rows_path = predictions_dir / "test_row_ids.npy"
    tmp = rows_path.with_suffix(".npy.tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, test_rows)
    os.replace(tmp, rows_path)
    rows_sha = C.sha256_file(rows_path)
    for record in records.values():
        record["test_row_ids_sha256"] = CC.sha256_int64(test_rows)

    freeze: dict[str, Any] = {
        "schema": CC.SCHEMA_TEST_FREEZE,
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
        "arms_requested": list(arms),
        "missing_arms": [k for k in BOC_ARMS if k not in arms],
        "structure_cache": S.cache_identity(struct),
        "rc_distance_cache": RD.cache_identity(rc_dist),
        "test_rows_path": C.track_rel(rows_path),
        "test_rows_sha256": rows_sha,
        "test_rows_npy_sha256": CC.sha256_int64(test_rows),
        "n_test_rows": int(test_rows.size),
        "n_tasks": int(C.N_TASKS),
        "prediction_batch_size": 256,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "environment": C.environment(),
        "predictions": records,
        "unique_predictions": sorted(records),
        "aliases": {},
        "total_seconds": float(time.perf_counter() - started),
        "note": (
            "Test logits are frozen before any official Test label is opened. Only "
            "boc_test.py score reads raw/graph-label.csv.gz, through the inherited "
            "cc_data.read_official_test_labels gate."
        ),
    }
    freeze["seal_sha256"] = CD.freeze_seal(freeze)
    C.write_json(out_path, freeze)
    check = C.read_json(out_path)
    if check["seal_sha256"] != CD.freeze_seal(check):
        raise RuntimeError("written freeze manifest seal does not verify")
    C.log(f"[freeze] wrote {out_path} seal={freeze['seal_sha256'][:16]}")
    return 0


# ---------------------------------------------------------------------------
# scoring
# ---------------------------------------------------------------------------


def _load_prediction(record: Mapping[str, Any], predictions_dir: Path) -> np.ndarray:
    path = _abs_path(record["test_logits_path"])
    if path.resolve().parent != Path(predictions_dir).resolve():
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


def run_score(freeze_path: Path, out_path: Path, per_task_path: Path, predictions_dir: Path) -> int:
    freeze = C.read_json(freeze_path)
    if freeze.get("schema") != CC.SCHEMA_TEST_FREEZE:
        raise RuntimeError("wrong freeze schema")
    if freeze.get("test_labels_read") is not False:
        raise RuntimeError("freeze manifest does not declare test_labels_read=false")
    if freeze.get("seal_sha256") != CD.freeze_seal(freeze):
        raise RuntimeError("freeze seal does not verify")
    selection_path = _abs_path(freeze["selection_manifest"])
    if C.sha256_file(selection_path) != freeze["selection_manifest_sha256"]:
        raise RuntimeError("selection manifest changed after the prediction freeze")

    task = C.BC.SD.load_task()
    test_reference = CD.read_official_test_labels(task, freeze_path=freeze_path)
    y_test = np.asarray(test_reference["labels"])
    if y_test.shape != (C.N_TEST, C.N_TASKS):
        raise RuntimeError(f"official Test label matrix {y_test.shape}")

    records = freeze["predictions"]
    for record in records.values():
        _load_prediction(record, predictions_dir)

    scored: dict[str, Any] = {}
    for label, record in records.items():
        logits = _load_prediction(record, predictions_dir)
        probs = (1.0 / (1.0 + np.exp(-np.asarray(logits, dtype=np.float64)))).astype(np.float32)
        official = C.MET.official_ap(y_test, probs)
        independent = C.MET.per_task_ap(y_test, probs)
        scored[label] = {
            "model": record["model"],
            "epoch": int(record["epoch"]),
            "role": record["role"],
            "checkpoint": record["checkpoint"],
            "checkpoint_sha256": record["checkpoint_sha256"],
            "valid_ap": float(record["valid_ap_replay"]),
            "n_tasks_scored_valid": int(record["n_tasks_scored_valid"]),
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
            "valid_per_task_ap": record["valid_per_task_ap"],
            "valid_observed": record["valid_observed"],
            "valid_positive": record["valid_positive"],
            "valid_negative": record["valid_negative"],
            "valid_test_ap_delta": float(official["value"]) - float(record["valid_ap_replay"]),
            "test_logits_sha256": record["test_logits_sha256"],
        }
        C.log(f"[score] {label} valid {record['valid_ap_replay']:.9f} test {official['value']:.9f}")

    def find(model: str, suffix: str) -> str:
        for label, record in scored.items():
            if record["model"] == model and label.endswith(suffix):
                return label
        raise KeyError((model, suffix))

    comparisons: dict[str, Any] = {}
    for new in ("BOC_FULL_best_valid", "BOC_FULL_epoch40", "BOC_BOND_best_valid", "BOC_BOND_epoch40"):
        if new not in scored:
            continue
        for old in ("W2_best_valid", "RC_REAL_best_valid"):
            if old not in scored:
                continue
            up, down, both = _wins(scored[new]["test_per_task_ap"], scored[old]["test_per_task_ap"])
            comparisons[f"{new}_minus_{old}"] = {
                "delta": scored[new]["test_ap_official"] - scored[old]["test_ap_official"],
                "valid_delta": scored[new]["valid_ap"] - scored[old]["valid_ap"],
                "tasks_improved": up,
                "tasks_degraded": down,
                "tasks_scored_both": both,
            }

    test_obs = np.isfinite(y_test)
    label_stats = {
        "test": {
            "rows": int(task.test_rows.size),
            "observed_entries": int(test_obs.sum()),
            "positive_entries": int((y_test[test_obs] == 1.0).sum()),
            "positive_rate_observed": float((y_test[test_obs] == 1.0).sum() / test_obs.sum()),
        }
    }
    # Support strata use the official Test positive count of the same task set.
    support_test = np.asarray(scored[find("FULL", "best_valid")]["test_positive"], dtype=np.int64)
    strata: dict[str, Any] = {}
    w2_label = find("W2", "best_valid")
    rc_label = find("RC-REAL", "best_valid")
    for arm in ("FULL", "BOND"):
        label = find(arm, "best_valid")
        for ref_label, ref_name in ((w2_label, "W2"), (rc_label, "RC-REAL")):
            for name, lo, hi in (
                ("ge20", 20, None),
                ("5_19", 5, 20),
                ("lt5", 0, 5),
            ):
                strata[f"{arm}_vs_{ref_name}_{name}"] = _stratified(
                    scored[label]["test_per_task_ap"],
                    scored[ref_label]["test_per_task_ap"],
                    support_test,
                    min_support=lo,
                    max_support=hi,
                )
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
        "label_stats": label_stats,
        "n_test_rows": int(C.N_TEST),
        "n_tasks": int(C.N_TASKS),
        "n_tasks_scored_test": int(scored[find("FULL", "best_valid")]["n_tasks_scored"]),
        "test_label_masking": "NaN entries excluded per task; a task is scored only with both classes present",
        "standardisation": "Train-only normalisation inherited from the frozen cache; nothing refitted",
        "checkpoint_selection": "argmax official Validation AP over epochs 1..40, earliest wins ties; no Test score looked at",
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
    _write_per_task_csv(per_task_path, scored, freeze, y_test)
    C.log(f"[score] wrote {out_path} and {per_task_path}")
    return 0


def _write_per_task_csv(path: Path, scored: Mapping[str, Any], freeze: Mapping[str, Any], y_test: np.ndarray) -> None:
    labels = list(freeze["predictions"])
    fields = ["task", "test_observed", "test_positive", "support_stratum"]
    fields += [f"test_ap_{label}" for label in labels]
    fields += ["test_delta_FULL_best_minus_W2", "test_delta_FULL_best_minus_RC_REAL", "test_delta_BOND_best_minus_W2"]
    path.parent.mkdir(parents=True, exist_ok=True)
    obs = np.isfinite(y_test)
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
            }
            for label in labels:
                ap = scored[label]["test_per_task_ap"][t]
                row[f"test_ap_{label}"] = "" if ap is None else float(ap)
            for arm in ("FULL", "BOND"):
                for ref in ("W2", "RC_REAL"):
                    if arm != "FULL" and ref == "RC_REAL":
                        continue
                    a = scored[f"BOC_{arm}_best_valid"]["test_per_task_ap"][t]
                    b = scored[f"{ref}_best_valid"]["test_per_task_ap"][t]
                    key = f"test_delta_{arm}_best_minus_{ref}"
                    if key in fields:
                        row[key] = "" if a is None or b is None else float(a - b)
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    fz = sub.add_parser("freeze")
    fz.add_argument("--selection", default=str(SELECTION_PATH))
    fz.add_argument("--out", default=str(FREEZE_PATH))
    fz.add_argument("--predictions-dir", default=str(PREDICTIONS))
    fz.add_argument("--struct-cache", default=str(C.STRUCT_CACHE_DIR))
    fz.add_argument("--device", default="cuda")
    sc = sub.add_parser("score")
    sc.add_argument("--freeze", default=str(FREEZE_PATH))
    sc.add_argument("--out", default=str(SCORES_PATH))
    sc.add_argument("--per-task", default=str(PER_TASK_PATH))
    sc.add_argument("--predictions-dir", default=str(PREDICTIONS))
    args = parser.parse_args(argv)
    if args.mode == "freeze":
        device = torch.device(args.device)
        if device.type == "cuda" and not torch.cuda.is_available():
            raise RuntimeError("CUDA requested but not available")
        return run_freeze(
            Path(args.out),
            Path(args.predictions_dir),
            device,
            struct_cache=args.struct_cache,
            selection_path=Path(args.selection),
        )
    if args.mode == "score":
        return run_score(Path(args.freeze), Path(args.out), Path(args.per_task), Path(args.predictions_dir))
    raise KeyError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

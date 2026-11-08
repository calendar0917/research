"""Official Test prediction freeze and the single post-freeze scoring pass.

Two subcommands enforce the ordering (identical discipline to ``sp_test.py``):

* ``freeze`` loads the six pre-registered checkpoints (RC-REAL best-Valid and
  epoch-40, RC-BLIND best-Valid and epoch-40, and the two historical W2
  checkpoints used as the *same-pipeline* reference), replays their official
  Validation AP, writes Test logits from the **label-free** distance cache and
  seals every hash.  It never opens an official Test label.
* ``score`` verifies the freeze seal and every prediction hash, and only then
  reads the official OGB Test labels through the inherited
  ``cc_data.read_official_test_labels`` gate, scoring with the official
  ``ogb.graphproppred.Evaluator("ogbg-molpcba")`` plus an independent per-task
  AP recomputation.

The Test split has been exposed before (``lcrc_pcba_continuation_recipe_v1``).
This round's role for Test is ``EXPOSED_TEST_PERFORMANCE_BENCHMARK``: it is the
primary *engineering* comparison metric, never a model-selection signal (the
selected checkpoint is always the highest official Validation AP).
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

import rc_common as C  # noqa: E402
import rc_dist as D  # noqa: E402
import rc_model as M  # noqa: E402
import rc_train as T  # noqa: E402

CC_CODE = C.CC_ROUND / "code"
if str(CC_CODE) not in sys.path:
    sys.path.insert(0, str(CC_CODE))

import cc_common as CC  # noqa: E402  (inherited freeze schema / relativiser)
import cc_data as CD  # noqa: E402  (inherited official Test label gate)

PREDICTIONS = C.ANALYSIS / "test_predictions"
FREEZE_PATH = C.ANALYSIS / "test_prediction_manifest.json"
SCORES_PATH = C.ANALYSIS / "test_scores.json"
PER_TASK_PATH = C.ANALYSIS / "per_task_valid_test.csv"
SELECTION_PATH = C.ANALYSIS / "selected_checkpoint.json"
VALID_REPLAY_TOLERANCE = 1e-4

MODEL_KEYS = ("REAL", "BLIND")


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


def run_dir(model_key: str) -> Path:
    return C.RUNS / model_key


def load_run_json(model_key: str) -> dict[str, Any]:
    path = run_dir(model_key) / "run.json"
    if not path.is_file():
        raise RuntimeError(f"run.json missing for {model_key}: {path}")
    return C.read_json(path)


# ---------------------------------------------------------------------------
# selection manifest (Validation only)
# ---------------------------------------------------------------------------


def build_selection(
    arms: Sequence[str] = MODEL_KEYS,
    *,
    out_path: Path = SELECTION_PATH,
    interim: bool = False,
) -> dict[str, Any]:
    """Freeze the best-Validation choice of each available arm from its run.json.

    ``arms`` only ever shrinks for the *interim* peek that scores the finished
    arm before the other one exists; the recorded round result always uses the
    default (both arms).
    """
    arms = tuple(arms)
    if not arms or any(a not in MODEL_KEYS for a in arms):
        raise KeyError(arms)
    models: dict[str, Any] = {}
    for key in arms:
        run = load_run_json(key)
        if run.get("status") != "COMPLETED":
            raise RuntimeError(f"{key} run is not COMPLETED: {run.get('status')}")
        if run.get("test_evaluated"):
            raise RuntimeError(f"{key} run claims a Test evaluation before the freeze")
        best_epoch = int(run["best_epoch"])
        best_path = _abs_path(run["best_checkpoint"])
        epoch40_path = _abs_path(run["primary_checkpoint"])
        for path, sha in ((best_path, run["best_checkpoint_sha256"]), (epoch40_path, run["primary_checkpoint_sha256"])):
            if not path.is_file():
                raise RuntimeError(f"{key} checkpoint missing: {path}")
            if C.sha256_file(path) != sha:
                raise RuntimeError(f"{key} checkpoint hash changed: {path}")
        models[key] = {
            "model": key,
            "parameters": int(run["parameters"]),
            "best_epoch": best_epoch,
            "best_valid_ap": float(run["best_valid_ap"]),
            "valid_ap_at_40": float(run["fixed_epoch_reports"][str(C.TOTAL_EPOCHS)]["ap"]),
            "best_checkpoint": C.track_rel(best_path),
            "best_checkpoint_sha256": run["best_checkpoint_sha256"],
            "epoch40_checkpoint": C.track_rel(epoch40_path),
            "epoch40_checkpoint_sha256": run["primary_checkpoint_sha256"],
            "gpu_hours": float(run["gpu_hours"]),
        }
    selection = {
        "schema": "lcrc-pcba-rc-selection-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "test_read": False,
        "selection_uses_test": False,
        "selection_rule": "argmax official Validation AP over epochs 1..40 per arm (earliest wins ties)",
        "arms_requested": list(arms),
        "missing_arms": [k for k in MODEL_KEYS if k not in arms],
        "interim_peek": bool(interim),
        "interim_peek_reason": (
            "interim Test read requested by the operator while the remaining arm was still "
            "training; it selects nothing, changes no hyperparameter and no arm is retrained"
        )
        if interim
        else None,
        "models": models,
        "w2_reference": {
            "best_epoch": int(C.W2["best_epoch"]),
            "best_valid_ap": float(C.W2["best_valid_ap"]),
            "best_checkpoint": C.track_rel(C.W2_CHECKPOINT_BEST),
            "best_checkpoint_sha256": C.W2["best_checkpoint_sha256"],
            "epoch40_checkpoint": C.track_rel(C.W2_CHECKPOINT_EPOCH40),
            "epoch40_checkpoint_sha256": C.W2["epoch40_checkpoint_sha256"],
            "valid_ap_at_40": float(C.W2["ap40"]),
            "test_ap_official_best": float(C.W2_TEST_AP_OFFICIAL),
            "parameters": int(C.W2["parameters"]),
            "gpu_hours": float(C.W2["gpu_hours"]),
        },
    }
    C.write_json(out_path, selection)
    return selection


# ---------------------------------------------------------------------------
# model restore / prediction
# ---------------------------------------------------------------------------


def restore_model(entry: Mapping[str, Any], task: Any, dist: Mapping[str, Any], device: torch.device):
    path = _abs_path(entry["path"])
    if not path.is_file():
        raise RuntimeError(f"checkpoint missing: {path}")
    sha = C.sha256_file(path)
    if sha != entry["sha256"]:
        raise RuntimeError(f"{entry['label']} checkpoint sha {sha} != recorded {entry['sha256']}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model") != entry["model"]:
        raise RuntimeError(f"{entry['label']} checkpoint model {payload.get('model')} != {entry['model']}")
    if int(payload.get("epoch", -1)) != int(entry["epoch"]):
        raise RuntimeError(f"{entry['label']} checkpoint epoch {payload.get('epoch')} != {entry['epoch']}")
    key = str(entry["model"])
    if key == "W2":
        model, audit = C.BSM.build_model(key, task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device="cpu")
    elif key in C.MODELS:
        model, audit = M.build_model(key, task.spec, task.norm, C.N_TASKS, seed=int(C.SEED), device="cpu")
    else:  # pragma: no cover
        raise KeyError(key)
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{entry['label']} state mismatch: missing={missing} unexpected={unexpected}")
    model.to(device)
    model.eval()
    return model, payload, audit, sha


def predict_rows(
    model: torch.nn.Module,
    task: Any,
    dist: Mapping[str, Any],
    rows: np.ndarray,
    device: torch.device,
    *,
    batch_size: int = 256,
) -> np.ndarray:
    return T.predict_logits(model, task, dist, rows, device, batch_size=batch_size)


def validation_replay(model: torch.nn.Module, task: Any, dist: Mapping[str, Any], device: torch.device) -> dict[str, Any]:
    import bs_common as BC

    rows = np.asarray(task.valid_rows, dtype=np.int64)
    logits = predict_rows(model, task, dist, rows, device)
    probs = BC.PC.sigmoid(logits)
    report = C.MET.validation_report(np.asarray(task.labels[rows]), probs)
    report.update(
        {
            "valid_logits_sha256": _np_sha(logits),
            "valid_bce": float(C.MET.masked_bce_numpy(logits, np.asarray(task.labels[rows]))),
            "evaluated_rows": int(rows.size),
        }
    )
    return report


# ---------------------------------------------------------------------------
# the pre-registered checkpoint list
# ---------------------------------------------------------------------------


def candidate_models(selection: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    if selection.get("test_read") is not False:
        raise RuntimeError("selection manifest must declare test_read=false")
    if bool(selection.get("selection_uses_test")):
        raise RuntimeError("selection manifest claims test-based selection")
    out: dict[str, dict[str, Any]] = {}
    for key in selection.get("arms_requested", MODEL_KEYS):
        entry = selection["models"][key]
        out[f"{key}_best_valid"] = {
            "label": f"{key}_best_valid",
            "model": key,
            "epoch": int(entry["best_epoch"]),
            "path": entry["best_checkpoint"],
            "sha256": entry["best_checkpoint_sha256"],
            "valid_ap": float(entry["best_valid_ap"]),
            "role": f"{key} selected checkpoint (highest official Validation AP)",
        }
        out[f"{key}_epoch40"] = {
            "label": f"{key}_epoch40",
            "model": key,
            "epoch": int(C.TOTAL_EPOCHS),
            "path": entry["epoch40_checkpoint"],
            "sha256": entry["epoch40_checkpoint_sha256"],
            "valid_ap": float(entry["valid_ap_at_40"]),
            "role": f"{key} fixed epoch-40 endpoint (auxiliary)",
        }
    w2 = selection["w2_reference"]
    out["W2_best_valid_epoch24"] = {
        "label": "W2_best_valid_epoch24",
        "model": "W2",
        "epoch": int(w2["best_epoch"]),
        "path": w2["best_checkpoint"],
        "sha256": w2["best_checkpoint_sha256"],
        "valid_ap": float(w2["best_valid_ap"]),
        "role": "historical W2 best-Validation checkpoint (same-pipeline reference)",
    }
    out["W2_epoch40"] = {
        "label": "W2_epoch40",
        "model": "W2",
        "epoch": int(C.TOTAL_EPOCHS),
        "path": w2["epoch40_checkpoint"],
        "sha256": w2["epoch40_checkpoint_sha256"],
        "valid_ap": float(w2["valid_ap_at_40"]),
        "role": "historical W2 fixed epoch-40 endpoint (same-pipeline reference)",
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
    dist_cache: str = str(C.DIST_CACHE_DIR),
    arms: Sequence[str] = MODEL_KEYS,
    selection_path: Path = SELECTION_PATH,
    interim: bool = False,
) -> int:
    selection = build_selection(arms, out_path=selection_path, interim=interim)
    task = D.load_task_gated()
    dist = D.load_cache(dist_cache)
    test_rows = np.asarray(task.test_rows, dtype=np.int64)
    train_rows = np.asarray(task.train_rows, dtype=np.int64)
    valid_rows = np.asarray(task.valid_rows, dtype=np.int64)
    if np.intersect1d(test_rows, train_rows).size or np.intersect1d(test_rows, valid_rows).size:
        raise RuntimeError("Test rows overlap Train/Valid")
    if np.isfinite(task.labels[test_rows]).any():
        raise RuntimeError("the training label artifact exposes Test targets")
    for key in arms:
        run = load_run_json(key)
        if str(run.get("status")) != "COMPLETED":
            raise RuntimeError(f"{key} did not complete; refusing to freeze a Test prediction")

    models = candidate_models(selection)
    records: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for label, entry in models.items():
        t0 = time.perf_counter()
        model, payload, audit, sha = restore_model(entry, task, dist, device)
        t_load = time.perf_counter() - t0
        t1 = time.perf_counter()
        valid = validation_replay(model, task, dist, device)
        t_valid = time.perf_counter() - t1
        delta = abs(float(valid["ap"]) - float(entry["valid_ap"]))
        if delta > VALID_REPLAY_TOLERANCE:
            raise RuntimeError(
                f"{label} Validation replay {valid['ap']} != recorded {entry['valid_ap']} (delta {delta})"
            )
        t2 = time.perf_counter()
        logits = predict_rows(model, task, dist, test_rows, device)
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
                "parameters": int(audit["total_parameters"]),
                "parameter_check": audit.get("parameter_check"),
                "base_state_sha256": audit.get("base_state_sha256"),
                "model_state_sha256": audit.get("model_state_sha256"),
            },
            "load_seconds": float(t_load),
            "valid_seconds": float(t_valid),
            "test_predict_seconds": float(t_test),
            "test_seconds_per_row": float(t_test / max(1, int(logits.shape[0]))),
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
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "test_labels_read": False,
        "test_labels_read_before_freeze": False,
        "selection_manifest": C.track_rel(selection_path),
        "selection_manifest_sha256": C.sha256_file(selection_path),
        "arms_requested": list(arms),
        "missing_arms": [k for k in MODEL_KEYS if k not in arms],
        "interim_peek": bool(interim),
        "distance_cache": D.cache_identity(dist),
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
            "rc_test.py score reads raw/graph-label.csv.gz, through the inherited "
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
    per_task_a: Sequence[Any], per_task_b: Sequence[Any], support: Sequence[int], *, min_support: int
) -> dict[str, Any]:
    deltas = []
    improved = 0
    for a, b, s in zip(per_task_a, per_task_b, support):
        if a is None or b is None or int(s) < int(min_support):
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

    task = D.load_task_gated()
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
            "test_ap_abs_delta": float(
                abs(float(official["value"]) - float(independent["mean_over_scored_tasks"]))
            ),
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
            "test_seconds_per_row": float(record["test_seconds_per_row"]),
        }
        C.log(
            f"[score] {label} valid {record['valid_ap_replay']:.9f} test {official['value']:.9f} "
            f"(independent {independent['mean_over_scored_tasks']:.9f})"
        )

    pairs = [
        ("REAL_best_valid", "W2_best_valid_epoch24"),
        ("BLIND_best_valid", "W2_best_valid_epoch24"),
        ("REAL_best_valid", "BLIND_best_valid"),
        ("REAL_epoch40", "W2_epoch40"),
        ("BLIND_epoch40", "W2_epoch40"),
        ("REAL_epoch40", "BLIND_epoch40"),
        ("REAL_epoch40", "REAL_best_valid"),
        ("W2_epoch40", "W2_best_valid_epoch24"),
    ]
    comparisons: dict[str, Any] = {}
    for new, old in pairs:
        if new not in scored or old not in scored:
            continue
        up, down, both = _wins(scored[new]["test_per_task_ap"], scored[old]["test_per_task_ap"])
        comparisons[f"test_{new}_minus_{old}"] = {
            "delta_test_ap": scored[new]["test_ap_official"] - scored[old]["test_ap_official"],
            "delta_valid_ap": scored[new]["valid_ap"] - scored[old]["valid_ap"],
            "tasks_improved": up,
            "tasks_degraded": down,
            "tasks_unchanged": both - up - down,
            "tasks_scored_both": both,
        }

    reference = "REAL_best_valid" if "REAL_best_valid" in scored else "W2_best_valid_epoch24"
    support_test = np.asarray(scored[reference]["test_positive"], dtype=np.int64)
    support_valid = np.asarray(scored[reference]["valid_positive"], dtype=np.int64)
    support_stats: dict[str, Any] = {}
    for tag, new, old in (
        ("REAL_minus_W2", "REAL_best_valid", "W2_best_valid_epoch24"),
        ("BLIND_minus_W2", "BLIND_best_valid", "W2_best_valid_epoch24"),
        ("REAL_minus_BLIND", "REAL_best_valid", "BLIND_best_valid"),
    ):
        if new not in scored or old not in scored:
            continue
        support_stats[f"test_ge20_{tag}"] = _stratified(
            scored[new]["test_per_task_ap"], scored[old]["test_per_task_ap"], support_test, min_support=20
        )
        support_stats[f"test_ge5_lt20_{tag}"] = _stratified(
            scored[new]["test_per_task_ap"], scored[old]["test_per_task_ap"], support_test, min_support=5
        )
        support_stats[f"test_lt5_{tag}"] = _stratified(
            scored[new]["test_per_task_ap"], scored[old]["test_per_task_ap"], support_test, min_support=0
        )
        support_stats[f"valid_ge20_{tag}"] = _stratified(
            scored[new]["valid_per_task_ap"], scored[old]["valid_per_task_ap"], support_valid, min_support=20
        )

    valid_obs = np.isfinite(task.labels[task.valid_rows])
    test_obs = np.isfinite(y_test)
    label_stats = {
        "valid": {
            "rows": int(task.valid_rows.size),
            "observed_entries": int(valid_obs.sum()),
            "positive_entries": int((task.labels[task.valid_rows][valid_obs] == 1.0).sum()),
        },
        "test": {
            "rows": int(task.test_rows.size),
            "observed_entries": int(test_obs.sum()),
            "positive_entries": int((y_test[test_obs] == 1.0).sum()),
        },
    }
    for split in label_stats:
        obs = label_stats[split]["observed_entries"]
        label_stats[split]["positive_rate_observed"] = (
            float(label_stats[split]["positive_entries"] / obs) if obs else None
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
        "distance_cache": freeze["distance_cache"],
        "official_evaluator": "ogb.graphproppred.Evaluator('ogbg-molpcba')",
        "independent_evaluator": "sklearn average_precision_score per task with NaN masking",
        "test_label_artifact": {k: v for k, v in test_reference.items() if k != "labels"},
        "models": scored,
        "comparisons": comparisons,
        "support_strata": support_stats,
        "label_stats": label_stats,
        "n_test_rows": int(C.N_TEST),
        "n_tasks": int(C.N_TASKS),
        "n_tasks_scored_test": int(scored[reference]["n_tasks_scored"]),
        "n_tasks_scored_valid": int(scored[reference]["n_tasks_scored_valid"]),
        "reference_model": reference,
        "arms_scored": sorted({record["model"] for record in records.values()}),
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
    _write_per_task_csv(per_task_path, scored, y_test, label_stats)
    C.log(f"[score] wrote {out_path} and {per_task_path}")
    return 0


def _write_per_task_csv(
    path: Path,
    scored: Mapping[str, Any],
    y_test: np.ndarray,
    label_stats: Mapping[str, Any],
) -> None:
    order = [
        "REAL_best_valid",
        "REAL_epoch40",
        "BLIND_best_valid",
        "BLIND_epoch40",
        "W2_best_valid_epoch24",
        "W2_epoch40",
    ]
    order = [x for x in order if x in scored]
    fields = ["task", "valid_observed", "valid_positive", "test_observed", "test_positive", "support_stratum"]
    fields += [f"valid_ap_{label}" for label in order]
    fields += [f"test_ap_{label}" for label in order]
    fields += [
        "test_delta_REAL_minus_W2",
        "test_delta_BLIND_minus_W2",
        "test_delta_REAL_minus_BLIND",
        "valid_delta_REAL_minus_W2",
        "valid_delta_REAL_minus_BLIND",
    ]
    ref = scored["REAL_best_valid"] if "REAL_best_valid" in scored else scored["W2_best_valid_epoch24"]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for t in range(C.N_TASKS):
            support = int(ref["test_positive"][t])
            row: dict[str, Any] = {
                "task": t,
                "valid_observed": int(ref["valid_observed"][t]),
                "valid_positive": int(ref["valid_positive"][t]),
                "test_observed": int(ref["test_observed"][t]),
                "test_positive": support,
                "support_stratum": "test_ge20" if support >= 20 else "test_5_19" if support >= 5 else "test_lt5",
            }
            for label in order:
                ap = scored[label]["valid_per_task_ap"][t]
                row[f"valid_ap_{label}"] = "" if ap is None else float(ap)
            for label in order:
                ap = scored[label]["test_per_task_ap"][t]
                row[f"test_ap_{label}"] = "" if ap is None else float(ap)
            def _ap(label: str, split: str, task: int) -> Any:
                if label not in scored:
                    return None
                return scored[label][f"{split}_per_task_ap"][task]

            r = _ap("REAL_best_valid", "test", t)
            b = _ap("BLIND_best_valid", "test", t)
            w = _ap("W2_best_valid_epoch24", "test", t)
            row["test_delta_REAL_minus_W2"] = "" if r is None or w is None else float(r - w)
            row["test_delta_BLIND_minus_W2"] = "" if b is None or w is None else float(b - w)
            row["test_delta_REAL_minus_BLIND"] = "" if r is None or b is None else float(r - b)
            vr = _ap("REAL_best_valid", "valid", t)
            vw = _ap("W2_best_valid_epoch24", "valid", t)
            vb = _ap("BLIND_best_valid", "valid", t)
            row["valid_delta_REAL_minus_W2"] = "" if vr is None or vw is None else float(vr - vw)
            row["valid_delta_REAL_minus_BLIND"] = "" if vr is None or vb is None else float(vr - vb)
            writer.writerow(row)


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    fz = sub.add_parser("freeze")
    fz.add_argument("--out", default=str(FREEZE_PATH))
    fz.add_argument("--predictions-dir", default=str(PREDICTIONS))
    fz.add_argument("--device", default="cuda")
    fz.add_argument("--dist-cache", default=str(C.DIST_CACHE_DIR))
    fz.add_argument(
        "--arms",
        default=",".join(MODEL_KEYS),
        help="comma separated arms to freeze; a proper subset is an interim peek",
    )
    fz.add_argument("--selection", default=str(SELECTION_PATH))
    fz.add_argument(
        "--interim",
        action="store_true",
        help="record that this freeze is an interim peek, not the round result",
    )
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
        arms = tuple(a.strip() for a in str(args.arms).split(",") if a.strip())
        return run_freeze(
            Path(args.out),
            Path(args.predictions_dir),
            device,
            dist_cache=args.dist_cache,
            arms=arms,
            selection_path=Path(args.selection),
            interim=bool(args.interim) or arms != tuple(MODEL_KEYS),
        )
    if args.mode == "score":
        return run_score(Path(args.freeze), Path(args.out), Path(args.per_task), Path(args.predictions_dir))
    raise KeyError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

"""Official Test prediction freeze and the single post-freeze scoring pass.

Two subcommands enforce the ordering:

* ``freeze`` reads the Validation-only selection manifest, restores the
  selected TopRank checkpoint (and, for a path audit, the historical W2
  checkpoint), replays their official Validation AP, writes the Test logits
  for the official Test row IDs and seals every hash.  It **never opens an
  official Test label**.
* ``score`` verifies the freeze seal and every prediction SHA256, and only then
  reads the official OGB Test labels through the inherited
  ``cc_data.read_official_test_labels`` gate.  It scores with the official
  ``Evaluator("ogbg-molpcba")`` and with an independent per-task sklearn AP
  recomputation.

The MolPCBA Test split has been exposed before.  This round's role is
``EXPOSED_TEST_PERFORMANCE_BENCHMARK``: a descriptive benchmark of a
checkpoint selected on Validation only, never a blind generalisation estimate
and never a selection signal.
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

import tr_common as C  # noqa: E402

CC_CODE = C.REPO / "tracks/structural-sufficiency/lcrc_pcba_continuation_recipe_v1/code"
if str(CC_CODE) not in sys.path:
    sys.path.insert(0, str(CC_CODE))

import cc_common as CC  # noqa: E402
import cc_data as CD  # noqa: E402

ANALYSIS = C.ANALYSIS
PREDICTIONS = ANALYSIS / "test_predictions"
FREEZE_PATH = ANALYSIS / "test_prediction_manifest.json"
SELECTION_PATH = ANALYSIS / "selected_checkpoint.json"
SCORES_PATH = ANALYSIS / "test_scores.json"
PER_TASK_PATH = ANALYSIS / "per_task_valid_test.csv"
VALID_REPLAY_TOLERANCE = 1e-4
W2_LOGITS_TOLERANCE = 1e-5


def _np_sha(array: np.ndarray) -> str:
    return C.sha256_bytes(np.ascontiguousarray(array, dtype=np.float32).tobytes())


def _save_npy_atomic(path: Path, array: np.ndarray) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    with open(tmp, "wb") as fh:
        np.save(fh, np.ascontiguousarray(array, dtype=np.float32))
    os.replace(tmp, path)
    return C.sha256_file(path)


def _sigmoid(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    return (1.0 / (1.0 + np.exp(-x))).astype(np.float32)


def _abs_path(value: str | Path) -> Path:
    return C.relative_or_absolute(value)


def _load_checkpoint(entry: Mapping[str, Any]) -> tuple[dict[str, Any], str]:
    path = _abs_path(entry["path"])
    if not path.is_file():
        raise RuntimeError(f"checkpoint missing: {path}")
    sha = C.sha256_file(path)
    if sha != entry["sha256"]:
        raise RuntimeError(f"{entry['label']} checkpoint sha {sha} != {entry['sha256']}")
    payload = torch.load(path, map_location="cpu", weights_only=False)
    if payload.get("model") != C.MODEL_KEY:
        raise RuntimeError(f"{entry['label']} checkpoint model {payload.get('model')} != W2")
    if int(payload.get("epoch", -1)) != int(entry["epoch"]):
        raise RuntimeError(f"{entry['label']} checkpoint epoch {payload.get('epoch')} != {entry['epoch']}")
    return payload, sha


def _restore_w2(entry: Mapping[str, Any], task: Any, device: torch.device):
    payload, sha = _load_checkpoint(entry)
    model, audit = C.build_w2(task, device="cpu")
    missing, unexpected = model.load_state_dict(payload["state_dict"], strict=False)
    if missing or unexpected:
        raise RuntimeError(f"{entry['label']} state mismatch: missing={missing} unexpected={unexpected}")
    model.to(device)
    model.eval()
    if int(audit["total_parameters"]) != C.MODEL_PARAMETERS:
        raise RuntimeError("W2 parameter count drifted")
    return model, payload, audit, sha


def _predict(model: torch.nn.Module, task: Any, rows: np.ndarray, device: torch.device) -> np.ndarray:
    logits = C.BC.SD.predict_rows(model, task, rows, device, batch_size=256)
    if logits.shape != (int(rows.size), C.N_TASKS):
        raise RuntimeError(f"prediction shape {logits.shape} != {(rows.size, C.N_TASKS)}")
    if not np.isfinite(logits).all():
        raise RuntimeError("non-finite Test logits")
    return logits


def _validation_replay(model: torch.nn.Module, task: Any, device: torch.device) -> dict[str, Any]:
    return C.BC.SD.evaluate_validation(
        model, task, device, reason="post_training_validation_replay"
    )


def selection_entry(selection_path: Path = SELECTION_PATH) -> dict[str, Any]:
    selection = C.read_json(selection_path)
    if selection.get("schema") != C.SCHEMA_SELECTION:
        raise RuntimeError("selection manifest has the wrong schema")
    if selection.get("round_id") != C.ROUND_ID:
        raise RuntimeError("selection manifest belongs to another round")
    if selection.get("test_read") is not False or selection.get("selection_uses_test") is not False:
        raise RuntimeError("selection manifest must declare test_read=false / selection_uses_test=false")
    if bool(selection.get("best_epoch") is None):
        raise RuntimeError("selection manifest has no best epoch")
    return {
        "label": "TopRank_best_valid",
        "path": selection["checkpoint"],
        "sha256": selection["checkpoint_sha256"],
        "epoch": int(selection["best_epoch"]),
        "valid_ap": float(selection["best_valid_ap"]),
        "role": "this round's best-Validation checkpoint (selected on Valid only)",
    }


def candidate_models(selection_path: Path = SELECTION_PATH) -> dict[str, dict[str, Any]]:
    return {
        "TopRank_best_valid": selection_entry(selection_path),
        "W2_best_valid_epoch24_reference": {
            "label": "W2_best_valid_epoch24_reference",
            "path": C.track_rel(C.W2["checkpoint_best"]),
            "sha256": C.W2["checkpoint_best_sha256"],
            "epoch": int(C.W2["best_epoch"]),
            "valid_ap": float(C.W2["best_valid_ap"]),
            "role": "historical W2 best-Validation checkpoint (reference only)",
        },
    }


def run_freeze(
    selection_path: Path,
    out_path: Path,
    predictions_dir: Path,
    device: torch.device,
) -> int:
    task = C.load_task_gated()
    test_rows = np.asarray(task.test_rows, dtype=np.int64)
    if not np.array_equal(test_rows, np.sort(test_rows)):
        raise RuntimeError("official Test rows are not sorted; the frozen row order changed")
    models = candidate_models(selection_path)
    records: dict[str, dict[str, Any]] = {}
    started = time.perf_counter()
    for label, entry in models.items():
        t0 = time.perf_counter()
        model, payload, audit, sha = _restore_w2(entry, task, device)
        t_load = time.perf_counter() - t0
        t1 = time.perf_counter()
        valid = _validation_replay(model, task, device)
        t_valid = time.perf_counter() - t1
        delta = abs(float(valid["ap"]) - float(entry["valid_ap"]))
        if delta > VALID_REPLAY_TOLERANCE:
            raise RuntimeError(
                f"{label} Validation replay {valid['ap']} != recorded {entry['valid_ap']} (delta {delta})"
            )
        t2 = time.perf_counter()
        logits = _predict(model, task, test_rows, device)
        t_test = time.perf_counter() - t2
        logits_path = predictions_dir / f"{label}.npy"
        logits_sha = _save_npy_atomic(logits_path, logits)
        records[label] = {
            "label": label,
            "model": C.MODEL_KEY,
            "role": entry["role"],
            "epoch": int(entry["epoch"]),
            "checkpoint": entry["path"],
            "checkpoint_sha256": sha,
            "checkpoint_payload_epoch": int(payload.get("epoch", -1)),
            "checkpoint_global_step": int(payload.get("global_step", -1)),
            "checkpoint_valid_ap_recorded_in_payload": payload.get("valid_ap"),
            "valid_ap_recorded": float(entry["valid_ap"]),
            "valid_ap_replay": float(valid["ap"]),
            "valid_ap_abs_delta": float(delta),
            "valid_bce_replay": float(valid["valid_bce"]),
            "valid_per_task_ap": valid["per_task_ap"],
            "valid_observed": valid["observed"],
            "valid_positive": valid["positive"],
            "valid_negative": valid["negative"],
            "valid_logits_sha256": valid.get("valid_logits_sha256"),
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
            "parameters": int(audit["total_parameters"]),
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
        record["test_row_ids_sha256"] = C.sha256_bytes(
            np.ascontiguousarray(test_rows, dtype=np.int64).tobytes()
        )

    # Path audit: does the same inherited prediction path reproduce the frozen
    # historical W2 Test logits?  This is a check of the code path, not a new
    # performance claim.  It never touches a Test label.
    w2_frozen_path = C.REPO / (
        "tracks/structural-sufficiency/lcrc_pcba_s3_selective_v1/analysis/test_predictions/"
        "W2_best_valid_epoch24.npy"
    )
    w2_reproduction: dict[str, Any] = {"frozen_predictions_present": w2_frozen_path.is_file()}
    current_w2 = records["W2_best_valid_epoch24_reference"]
    if w2_frozen_path.is_file():
        frozen = np.load(w2_frozen_path)
        if list(frozen.shape) != [int(test_rows.size), C.N_TASKS]:
            raise RuntimeError(f"historical W2 Test logits shape {frozen.shape}")
        diff = float(np.max(np.abs(frozen.astype(np.float64) - _load_prediction(current_w2, predictions_dir).astype(np.float64))))
        w2_reproduction.update(
            {
                "frozen_npy_sha256": C.sha256_file(w2_frozen_path),
                # The recorded round constant is the .npy file SHA, not the raw
                # float32 buffer SHA; keep both for diagnostics.
                "frozen_array_sha256": C.sha256_bytes(
                    np.ascontiguousarray(frozen, dtype=np.float32).tobytes()
                ),
                "rewritten_logits_sha256": current_w2["test_logits_sha256"],
                "frozen_logits_sha256": C.W2["test_logits_sha256"],
                "max_abs_diff": diff,
                "bitwise_sha_match": current_w2["test_logits_sha256"] == C.W2["test_logits_sha256"],
                "within_tolerance": bool(diff <= W2_LOGITS_TOLERANCE),
                "within_tolerance_1e_4": bool(diff <= 1e-4),
            }
        )
    else:
        w2_reproduction["within_tolerance"] = None

    freeze: dict[str, Any] = {
        "schema": CC.SCHEMA_TEST_FREEZE,
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "test_role": C.TEST_ROLE,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "git_state": C.git_state(),
        "test_labels_read": False,
        "test_labels_read_before_freeze": False,
        "selection_manifest": C.track_rel(selection_path),
        "selection_manifest_sha256": C.sha256_file(selection_path),
        "selection_role": "argmax official Validation AP over epochs 1..40, earliest wins ties",
        "test_rows_path": C.track_rel(rows_path),
        "test_rows_sha256": rows_sha,
        "test_rows_int64_sha256": C.sha256_bytes(
            np.ascontiguousarray(test_rows, dtype=np.int64).tobytes()
        ),
        "n_test_rows": int(test_rows.size),
        "n_tasks": int(C.N_TASKS),
        "prediction_batch_size": 256,
        "device": str(device),
        "device_name": torch.cuda.get_device_name(device) if device.type == "cuda" else "cpu",
        "environment": C.environment(),
        "feature_spec_hash": C.SOURCE["feature_spec_hash"],
        "norm_fingerprint": C.SOURCE["norm_fingerprint"],
        "split_partition_sha256": C.SOURCE["split_partition_sha256"],
        "logical_models": {
            label: {
                "prediction": label,
                "checkpoint_sha256": record["checkpoint_sha256"],
                "valid_ap_replay": record["valid_ap_replay"],
            }
            for label, record in records.items()
        },
        "predictions": records,
        "unique_predictions": sorted(records),
        "w2_test_prediction_reproduction": w2_reproduction,
        "total_seconds": float(time.perf_counter() - started),
        "note": (
            "Test logits are frozen before any official Test label is opened. "
            "Only tr_test.py score reads raw/graph-label.csv.gz, through the inherited "
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


def _stratum(rows: Sequence[Mapping[str, Any]], name: str) -> dict[str, Any]:
    deltas = np.asarray([r["delta"] for r in rows if r["delta"] is not None], dtype=np.float64)
    if not deltas.size:
        return {"name": name, "n_tasks": 0}
    return {
        "name": name,
        "n_tasks": int(deltas.size),
        "mean_delta": float(deltas.mean()),
        "median_delta": float(np.median(deltas)),
        "sum_delta": float(deltas.sum()),
        "contribution_to_macro_delta": float(deltas.sum() / 127.0),
        "improved": int((deltas > 0).sum()),
        "degraded": int((deltas < 0).sum()),
        "tied": int((deltas == 0).sum()),
        "worst_delta": float(deltas.min()),
        "best_delta": float(deltas.max()),
    }


def run_score(
    freeze_path: Path,
    out_path: Path,
    per_task_path: Path,
    predictions_dir: Path,
) -> int:
    freeze = C.read_json(freeze_path)
    if freeze.get("schema") != CC.SCHEMA_TEST_FREEZE:
        raise RuntimeError("wrong freeze schema")
    if freeze.get("round_id") != C.ROUND_ID:
        raise RuntimeError("freeze belongs to another round")
    if freeze.get("test_labels_read") is not False:
        raise RuntimeError("freeze manifest does not declare test_labels_read=false")
    if freeze.get("seal_sha256") != CD.freeze_seal(freeze):
        raise RuntimeError("freeze seal does not verify")
    selection_path = _abs_path(freeze["selection_manifest"])
    if C.sha256_file(selection_path) != freeze["selection_manifest_sha256"]:
        raise RuntimeError("selection manifest changed after the prediction freeze")

    task = C.load_task_gated()
    test_reference = CD.read_official_test_labels(task, freeze_path=freeze_path)
    y_test = np.asarray(test_reference["labels"])
    if y_test.shape != (C.N_TEST, C.N_TASKS):
        raise RuntimeError(f"official Test label matrix {y_test.shape}")
    test_rows = np.asarray(task.test_rows, dtype=np.int64)
    if C.sha256_bytes(np.ascontiguousarray(test_rows, dtype=np.int64).tobytes()) != freeze[
        "test_rows_int64_sha256"
    ]:
        raise RuntimeError("official Test rows changed after the prediction freeze")

    records = freeze["predictions"]
    for record in records.values():
        _load_prediction(record, predictions_dir)

    scored: dict[str, Any] = {}
    for label, logical in freeze["logical_models"].items():
        record = records[logical["prediction"]]
        logits = _load_prediction(record, predictions_dir)
        probs = _sigmoid(logits)
        official = C.MET.official_ap(y_test, probs)
        independent = C.MET.per_task_ap(y_test, probs)
        scored[label] = {
            "model": C.MODEL_KEY,
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
        }
        C.log(f"[score] {label} valid {record['valid_ap_replay']:.9f} test {official['value']:.9f}")

    # Historical W2 reference values are read from the frozen S3-SP round, and
    # cross-checked against the frozen W2 predictions in this freeze.
    w2_reference = C.read_json(_abs_path(C.W2["test_scores_source"]))
    w2_model = w2_reference["models"]["W2_best_valid_epoch24"]
    if w2_model["checkpoint_sha256"] != C.W2["checkpoint_best_sha256"]:
        raise RuntimeError("historical W2 checkpoint identity drifted in test_scores.json")
    if int(w2_model["epoch"]) != int(C.W2["best_epoch"]):
        raise RuntimeError("historical W2 best epoch drifted")
    w2_frozen = np.load(
        C.REPO
        / "tracks/structural-sufficiency/lcrc_pcba_s3_selective_v1/analysis/test_predictions/"
        "W2_best_valid_epoch24.npy"
    )
    # ``C.W2["test_logits_sha256"]`` is the SHA256 of the frozen ``.npy`` file
    # (the convention used by the S3-SP freeze), not of the raw float32 buffer.
    if C.sha256_file(
        C.REPO
        / "tracks/structural-sufficiency/lcrc_pcba_s3_selective_v1/analysis/test_predictions/"
        "W2_best_valid_epoch24.npy"
    ) != C.W2["test_logits_sha256"]:
        raise RuntimeError("frozen W2 Test prediction file SHA does not match the recorded value")
    w2_official_recomputed = C.MET.official_ap(y_test, _sigmoid(w2_frozen))
    if abs(float(w2_official_recomputed["value"]) - float(w2_model["test_ap_official"])) > 1e-9:
        raise RuntimeError("frozen W2 Test AP does not reproduce")
    w2_independent = C.MET.per_task_ap(y_test, _sigmoid(w2_frozen))
    if abs(float(w2_independent["mean_over_scored_tasks"]) - float(w2_model["test_ap_independent"])) > 1e-9:
        raise RuntimeError("frozen W2 independent per-task AP does not reproduce")

    top = scored["TopRank_best_valid"]
    comparisons = {
        "test_TopRank_minus_W2": {
            "delta": top["test_ap_official"] - float(w2_model["test_ap_official"]),
            "valid_delta": top["valid_ap"] - float(w2_model["valid_ap"]),
        }
    }
    up, down, both = _wins(top["test_per_task_ap"], w2_model["test_per_task_ap"])
    comparisons["test_TopRank_minus_W2"].update(
        {"tasks_improved": up, "tasks_degraded": down, "tasks_scored_both": both}
    )
    vup, vdown, vboth = _wins(top["valid_per_task_ap"], w2_model["valid_per_task_ap"])
    comparisons["valid_TopRank_minus_W2"] = {
        "delta": top["valid_ap"] - float(w2_model["valid_ap"]),
        "tasks_improved": vup,
        "tasks_degraded": vdown,
        "tasks_scored_both": vboth,
    }

    per_task: list[dict[str, Any]] = []
    for t in range(C.N_TASKS):
        tr_ap = top["test_per_task_ap"][t]
        w2_ap = w2_model["test_per_task_ap"][t]
        tr_valid = top["valid_per_task_ap"][t]
        w2_valid = w2_model["valid_per_task_ap"][t]
        support = int(top["test_positive"][t])
        stratum = (
            "test_ge20" if support >= 20 else "test_5_19" if support >= 5 else "test_lt5"
        )
        per_task.append(
            {
                "task": int(t),
                "test_observed": int(top["test_observed"][t]),
                "test_positive": support,
                "test_negative": int(top["test_negative"][t]),
                "support_stratum": stratum,
                "test_ap_toprank": None if tr_ap is None else float(tr_ap),
                "test_ap_w2": None if w2_ap is None else float(w2_ap),
                "delta": None
                if tr_ap is None or w2_ap is None
                else float(tr_ap) - float(w2_ap),
                "valid_ap_toprank": None if tr_valid is None else float(tr_valid),
                "valid_ap_w2": None if w2_valid is None else float(w2_valid),
                "valid_delta": None
                if tr_valid is None or w2_valid is None
                else float(tr_valid) - float(w2_valid),
                "valid_positive": int(top["valid_positive"][t]),
            }
        )

    scored_tasks = [r for r in per_task if r["delta"] is not None]
    macro_delta = float(np.mean([r["delta"] for r in scored_tasks]))
    strata = {
        name: _stratum([r for r in scored_tasks if r["support_stratum"] == name], name)
        for name in ("test_ge20", "test_5_19", "test_lt5")
    }
    high = strata["test_ge20"]
    low = [r for r in scored_tasks if r["support_stratum"] != "test_ge20"]
    low_deltas = np.asarray([r["delta"] for r in low], dtype=np.float64)
    dominant = sorted(scored_tasks, key=lambda r: abs(r["delta"]), reverse=True)[:5]
    low_dominant = sorted(low, key=lambda r: abs(r["delta"]), reverse=True)[:5]

    def _leave_one_out(rows: Sequence[Mapping[str, Any]], task_id: int) -> float:
        rest = [r["delta"] for r in scored_tasks if r["task"] != int(task_id)]
        return float(np.mean(rest)) if rest else float("nan")

    dominant_report = []
    for r in dominant:
        dominant_report.append(
            {
                **{k: r[k] for k in ("task", "test_positive", "support_stratum", "delta")},
                "share_of_sum_deltas": float(
                    r["delta"] / float(np.sum([x["delta"] for x in scored_tasks]))
                )
                if float(np.sum([x["delta"] for x in scored_tasks])) != 0
                else None,
                "macro_delta_without_task": _leave_one_out(scored_tasks, r["task"]),
            }
        )
    low_report = []
    for r in low_dominant:
        low_report.append(
            {
                **{k: r[k] for k in ("task", "test_positive", "support_stratum", "delta")},
                "macro_delta_without_task": _leave_one_out(scored_tasks, r["task"]),
            }
        )

    valid_deltas = [r["valid_delta"] for r in scored_tasks if r["valid_delta"] is not None]
    test_deltas = [r["delta"] for r in scored_tasks]
    same_sign = int(
        sum(
            1
            for r in scored_tasks
            if r["valid_delta"] is not None
            and r["delta"] != 0
            and r["valid_delta"] != 0
            and (r["delta"] > 0) == (r["valid_delta"] > 0)
        )
    )
    try:
        from scipy.stats import spearmanr, pearsonr

        rho = spearmanr(valid_deltas, test_deltas)
        pr = pearsonr(valid_deltas, test_deltas)
        correlation = {
            "spearman_rho": float(rho.statistic),
            "spearman_p": float(rho.pvalue),
            "pearson_r": float(pr[0]),
            "pearson_p": float(pr[1]),
        }
    except Exception:  # pragma: no cover
        correlation = {}

    n_test_pos = int(np.sum([r["test_positive"] for r in per_task]))
    label_stats = {
        "test": {
            "rows": int(C.N_TEST),
            "observed_entries": int(np.isfinite(y_test).sum()),
            "positive_entries": int(np.sum(y_test[np.isfinite(y_test)] == 1.0)),
        },
        "valid": {
            "rows": int(C.N_VALID),
            "observed_entries": int(np.isfinite(task.labels[task.valid_rows]).sum()),
            "positive_entries": int(
                np.sum(task.labels[task.valid_rows][np.isfinite(task.labels[task.valid_rows])] == 1.0)
            ),
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
        "official_evaluator": "ogb.graphproppred.Evaluator('ogbg-molpcba')",
        "independent_evaluator": "sklearn average_precision_score per task with NaN masking",
        "test_label_artifact": {k: v for k, v in test_reference.items() if k != "labels"},
        "standardisation": "Train-only normalisation inherited from the frozen cache; nothing refitted",
        "checkpoint_selection": "argmax official Validation AP over epochs 1..40, earliest wins ties; no Test score looked at",
        "models": {
            "TopRank_best_valid": top,
            "W2_best_valid_epoch24": {
                key: value for key, value in w2_model.items()
            },
        },
        "comparisons": comparisons,
        "per_task": per_task,
        "scored_tasks": int(len(scored_tasks)),
        "macro_test_delta": macro_delta,
        "support_strata": strata,
        "high_support": high,
        "low_support": {
            "n_tasks": int(len(low)),
            "mean_delta": float(low_deltas.mean()) if low_deltas.size else None,
            "sum_delta": float(low_deltas.sum()) if low_deltas.size else None,
            "contribution_to_macro_delta": float(low_deltas.sum() / 127.0)
            if low_deltas.size
            else None,
            "improved": int((low_deltas > 0).sum()) if low_deltas.size else 0,
            "degraded": int((low_deltas < 0).sum()) if low_deltas.size else 0,
            "dominant_tasks": low_report,
        },
        "dominant_tasks": dominant_report,
        "valid_test_direction": {
            "valid_delta_mean": float(np.mean(valid_deltas)) if valid_deltas else None,
            "test_delta_mean": float(np.mean(test_deltas)) if test_deltas else None,
            "same_sign_tasks": same_sign,
            "compared_tasks": len(valid_deltas),
            "correlation": correlation,
        },
        "label_stats": label_stats,
        "n_test_rows": int(C.N_TEST),
        "n_tasks": int(C.N_TASKS),
        "n_tasks_scored_test": int(top["n_tasks_scored"]),
        "n_tasks_scored_valid": int(top["n_tasks_scored_valid"]),
        "n_test_positive_entries": n_test_pos,
        "test_label_masking": "NaN entries excluded per task; a task is scored only with both classes present",
        "w2_reference_reproduction": {
            "official_ap_recomputed": float(w2_official_recomputed["value"]),
            "official_ap_recorded": float(w2_model["test_ap_official"]),
            "independent_ap_recomputed": float(w2_independent["mean_over_scored_tasks"]),
            "independent_ap_recorded": float(w2_model["test_ap_independent"]),
            "checkpoint_sha256": w2_model["checkpoint_sha256"],
            "epoch": int(w2_model["epoch"]),
            "logits_sha256": C.W2["test_logits_sha256"],
            "reproduced_from_frozen_npy": True,
        },
        "test_exposure_history": {
            "exposed_by": "lcrc_pcba_continuation_recipe_v1 (2026-11-06) read the official Test labels after its own freeze",
            "this_round_role": C.TEST_ROLE,
            "blind_generalisation_estimate": False,
            "reason": (
                "the split has been read before and the result may influence the next architecture decision, "
                "so this is a benchmark of an already-exposed split, not an independent blind test"
            ),
            "no_test_based_selection": True,
            "no_retraining_after_seeing_test": True,
        },
        "descriptive_only": True,
        "cannot_change_recipe_or_selection": True,
    }
    C.write_json(out_path, results)
    _write_per_task_csv(per_task_path, per_task)
    C.log(f"[score] wrote {out_path} and {per_task_path}")
    return 0


def _write_per_task_csv(path: Path, per_task: Sequence[Mapping[str, Any]]) -> None:
    fields = [
        "task",
        "test_positive",
        "test_observed",
        "test_negative",
        "support_stratum",
        "test_ap_toprank",
        "test_ap_w2",
        "delta",
        "valid_positive",
        "valid_ap_toprank",
        "valid_ap_w2",
        "valid_delta",
    ]
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in per_task:
            writer.writerow({k: ("" if row.get(k) is None else row.get(k)) for k in fields})


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="mode", required=True)
    fz = sub.add_parser("freeze")
    fz.add_argument("--selection", default=str(SELECTION_PATH))
    fz.add_argument("--out", default=str(FREEZE_PATH))
    fz.add_argument("--predictions-dir", default=str(PREDICTIONS))
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
            Path(args.selection), Path(args.out), Path(args.predictions_dir), device
        )
    if args.mode == "score":
        return run_score(
            Path(args.freeze), Path(args.out), Path(args.per_task), Path(args.predictions_dir)
        )
    raise KeyError(args.mode)


if __name__ == "__main__":
    raise SystemExit(main())

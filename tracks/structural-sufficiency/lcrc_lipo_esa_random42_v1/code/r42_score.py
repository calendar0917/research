"""Terminal, one-shot Test420 evaluation.

Guards (all hard failures):

* ``analysis/pre_test_manifest.json`` must exist and match the caller-provided
  SHA256 - the models, checkpoints and Test predictions are frozen at that point;
* the frozen prediction files must still hash to the manifest values;
* a one-time lock file must not already exist;
* the Test labels are read *here and only here*, from the hash-verified CSV.

The scorer never imports a training module, never touches Validation, and never
writes anything that a later step could use to re-select a model.
"""

from __future__ import annotations

from typing import Any

import numpy as np

import r42_common as C
import r42_metrics as MET

LOCK = C.ANALYSIS / "test_read.lock"
METRICS = C.ANALYSIS / "metrics_test.json"


def score_test(*, expected_manifest_sha256: str) -> dict[str, Any]:
    import r42_labels as L

    manifest_path = C.ANALYSIS / "pre_test_manifest.json"
    if not manifest_path.exists():
        raise PermissionError("pre-test manifest is missing; refusing to score Test")
    actual_manifest_sha = C.sha256_file(manifest_path)
    if actual_manifest_sha != expected_manifest_sha256:
        raise PermissionError(
            f"pre-test manifest sha256 {actual_manifest_sha} != expected {expected_manifest_sha256}"
        )
    if LOCK.exists():
        raise PermissionError(f"Test has already been scored once ({C.track_rel(LOCK)} exists)")
    if METRICS.exists():
        raise PermissionError(f"{C.track_rel(METRICS)} already exists; Test cannot be scored twice")

    manifest = C.read_json(manifest_path)
    if not manifest.get("test_scoring_allowed"):
        raise PermissionError("manifest does not authorise Test scoring")

    # 1. re-verify frozen predictions
    for row in manifest["predictions"]:
        path = C.REPO / row["path"]
        if not path.exists():
            raise FileNotFoundError(path)
        if C.sha256_file(path) != row["sha256"]:
            raise PermissionError(f"{row['model']}: frozen Test prediction hash drifted")

    # 2. read Test labels once
    vault = L.TestLabelVault(manifest_path, LOCK)
    y_test = vault.read(expected_manifest_sha256=expected_manifest_sha256)
    split = C.read_json(C.SPLIT_FILE)
    rows = np.asarray(split["rows"]["test"], dtype=np.int64)
    if y_test.shape != (len(rows),):
        raise ValueError("Test label vector has the wrong length")

    # 3. score
    results = []
    for row in manifest["predictions"]:
        npz = np.load(C.REPO / row["path"])
        pred = np.asarray(npz["pred"], dtype=np.float64)
        pred_rows = np.asarray(npz["rows"], dtype=np.int64)
        if not np.array_equal(pred_rows, rows):
            raise PermissionError(f"{row['model']}: prediction rows do not match the frozen Test rows")
        metrics = MET.all_metrics(pred, y_test)
        results.append(
            {
                "model": row["model"],
                "prediction_path": row["path"],
                "prediction_sha256": row["sha256"],
                "checkpoint_sha256": next(m["checkpoint_sha256"] for m in manifest["models"] if m["model"] == row["model"]),
                "n": int(pred.size),
                **metrics,
                "units": "raw_logD",
            }
        )

    from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

    for row in results:
        npz = np.load(C.REPO / row["prediction_path"])
        pred = np.asarray(npz["pred"], dtype=np.float64)
        row["sklearn_check"] = {
            "rmse": float(np.sqrt(mean_squared_error(y_test, pred))),
            "mae": float(mean_absolute_error(y_test, pred)),
            "r2": float(r2_score(y_test, pred)),
        }
        row["sklearn_agrees"] = all(abs(row["sklearn_check"][k] - row[k]) < 1e-10 for k in ("rmse", "mae", "r2"))

    baseline_rmse = float(np.sqrt(np.mean((y_test - np.mean(y_test)) ** 2)))
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "status": "SCORED_ONCE",
        "scored_utc": C.utcnow(),
        "pre_test_manifest_sha256": actual_manifest_sha,
        "split_file_sha256": C.sha256_file(C.SPLIT_FILE),
        "test_rows_sha256": C.rows_sha256_int64(rows),
        "test_label_sha256": C.sha256_bytes(np.ascontiguousarray(y_test, dtype="<f8").tobytes()),
        "n_test": int(y_test.size),
        "test_label_mean": float(y_test.mean()),
        "test_label_std": float(y_test.std()),
        "constant_baseline_rmse": baseline_rmse,
        "results": results,
        "models_scored": [r["model"] for r in results],
        "models_missing": [m["model"] for m in manifest["missing_models"]],
        "complete_four_model_comparison": len(results) == 4 and not manifest["missing_models"],
        "single_evaluation_evidence": {
            "lock_file": C.track_rel(LOCK),
            "lock_written_before_labels_read": True,
            "metrics_written_once": True,
            "no_retraining_after_test": True,
        },
        "gpu_hours_charged": manifest.get("gpu_hours_charged"),
    }
    C.write_json(METRICS, payload)
    vault.write_lock(
        {
            "protocol_id": C.PROTOCOL_ID,
            "scored_utc": payload["scored_utc"],
            "pre_test_manifest_sha256": actual_manifest_sha,
            "metrics_test_sha256": C.sha256_file(METRICS),
            "test_label_sha256": payload["test_label_sha256"],
        }
    )
    C.log(f"[score] one-shot Test evaluation complete for {payload['models_scored']}")
    return payload


__all__ = ["score_test", "LOCK", "METRICS"]

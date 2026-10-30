"""Monitor-only label access, gated on a frozen prediction manifest.

This module is the **only** path that can construct Monitor label values.  It
refuses to open the label artifact unless:

* ``analysis/predictions/MANIFEST.json`` and its ``.sha256`` lock exist;
* the manifest hash matches the lock;
* the manifest declares ``monitor_labels_read=false`` and ``status=FROZEN``;
* the protocol id / config hash match the frozen protocol;
* every registered arm x seed prediction file exists and its SHA-256 matches;
* the predicted rows equal the frozen Monitor row list exactly.

Consequently the evaluation order is mechanically enforced: predictions first,
labels second.  There is no code path that lets training call this module.
"""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any

import numpy as np

import access_guard
import fit_labels
import screen_common as C


class FreezeError(RuntimeError):
    """Raised when predictions are not frozen or do not match the manifest."""


def _expected_prediction_keys() -> list[str]:
    keys = [f"{arm}_s{seed}" for arm in C.NN_ARMS for seed in C.SEEDS]
    keys += [f"{arm}_r{r}" for arm in C.BASELINE_ARMS for r in C.BASELINE_REPLICATES]
    return keys


def verify_prediction_freeze(
    manifest_path: Path | str = C.PREDICTION_MANIFEST,
    lock_path: Path | str = C.PREDICTION_LOCK,
) -> dict[str, Any]:
    manifest_path = Path(manifest_path)
    lock_path = Path(lock_path)
    if not manifest_path.exists():
        raise FreezeError(f"prediction manifest missing: {C.track_rel(manifest_path)}")
    if not lock_path.exists():
        raise FreezeError(f"prediction manifest lock missing: {C.track_rel(lock_path)}")
    actual = C.sha256_file(manifest_path)
    lock = C.read_json(lock_path)
    locked = lock.get("sha256")
    if not locked or actual != locked:
        raise FreezeError("prediction manifest changed after the freeze lock was written")
    manifest = C.read_json(manifest_path)
    if manifest.get("status") != "FROZEN":
        raise FreezeError("prediction manifest status is not FROZEN")
    if manifest.get("monitor_labels_read") is not False:
        raise FreezeError("prediction manifest does not certify label-blind prediction")
    if manifest.get("protocol_id") != C.PROTOCOL_ID:
        raise FreezeError("prediction manifest protocol id mismatch")
    if manifest.get("config_hash") != C.config_hash(C.load_config()):
        raise FreezeError("prediction manifest config hash mismatch")
    if manifest.get("rows") != [int(r) for r in C.load_frozen_split()["split"]["monitor"]]:
        raise FreezeError("prediction manifest Monitor row order mismatch")

    entries = manifest.get("predictions") or {}
    expected = set(_expected_prediction_keys())
    if set(entries) != expected:
        missing = sorted(expected - set(entries))
        extra = sorted(set(entries) - expected)
        raise FreezeError(f"prediction registry mismatch missing={missing} extra={extra}")
    for key, entry in sorted(entries.items()):
        path = C.REPO / entry["file"]
        if not path.exists():
            raise FreezeError(f"{key}: prediction file missing")
        if C.sha256_file(path) != entry["sha256"]:
            raise FreezeError(f"{key}: prediction file sha256 mismatch")
        if entry.get("rows_sha256") != manifest["rows_sha256"]:
            raise FreezeError(f"{key}: prediction rows do not match the frozen Monitor rows")
        if not np.isfinite(np.load(path)["pred"]).all():
            raise FreezeError(f"{key}: non-finite frozen prediction")
    manifest["manifest_sha256"] = actual
    return manifest


def load_monitor_labels(
    *,
    manifest_path: Path | str = C.PREDICTION_MANIFEST,
    lock_path: Path | str = C.PREDICTION_LOCK,
    arm_guard: bool = True,
) -> dict[str, Any]:
    """Return Monitor labels in frozen row order, only after the prediction freeze."""
    manifest = verify_prediction_freeze(manifest_path, lock_path)
    split_info = C.load_frozen_split()
    fit_rows = split_info["split"]["fit"]
    monitor_rows = split_info["split"]["monitor"]
    reserved = set(split_info["split"]["reserved"])
    if (set(fit_rows) | set(monitor_rows)) & reserved:
        raise PermissionError("reserved rows overlap the Fit/Monitor supervision sets")

    if arm_guard:
        access_guard.arm()
    raw = access_guard.read_bytes_with_grant(C.LIPO_LABELS)
    actual_sha = hashlib.sha256(raw).hexdigest()
    if actual_sha != C.EXPECTED_SHA256["labels_fit_monitor"]:
        raise PermissionError(f"label artifact sha256 {actual_sha} != frozen")
    values, meta, audit = fit_labels._scan(
        raw.decode("utf-8"),
        requested_label_keys=(str(r) for r in monitor_rows),
        path=C.LIPO_LABELS,
    )
    fit_labels.check_artifact_metadata(meta, set(fit_rows) | set(monitor_rows))
    y = np.asarray([values[str(r)] for r in monitor_rows], dtype=np.float64)
    if y.shape != (C.SPLIT_SIZES["monitor"],):
        raise fit_labels.LabelArtifactError(f"Monitor label vector has shape {y.shape}")
    if not np.isfinite(y).all():
        raise fit_labels.LabelArtifactError("non-finite Monitor label")
    audit.update(
        {
            "labels_sha256": actual_sha,
            "rows_in_label_artifact": len(meta["rows"]),
            "reserved_rows_in_artifact": 0,
            "official_valid_test_rows_in_artifact": 0,
            "monitor_label_vector_sha256": hashlib.sha256(
                np.ascontiguousarray(y, dtype="<f8").tobytes()
            ).hexdigest(),
            "split_row_hash_monitor": split_info["hashes"]["monitor"],
            "prediction_manifest_sha256": manifest["manifest_sha256"],
            "prediction_keys_verified": sorted(manifest["predictions"]),
            "guard": access_guard.guard_status(),
        }
    )
    return {"y": y, "rows": list(monitor_rows), "labels_sha256": actual_sha, "audit": audit, "manifest": manifest}


__all__ = ["FreezeError", "verify_prediction_freeze", "load_monitor_labels"]

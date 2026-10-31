"""Monitor504 label access, gated by the v2 prediction manifest.

This module is imported only by the v2 evaluation entry point.  It refuses to
open the label artifact until the v2 prediction manifest and lock verify every
frozen prediction.  It constructs exactly the 504 Monitor values; the 2352 Fit
values are skipped without being converted.
"""

from __future__ import annotations

import hashlib
from typing import Any

import numpy as np

import v2_common as C
import v2_predict as P


def load_monitor_labels() -> dict[str, Any]:
    manifest = P.verify_manifest()
    split = C.load_frozen_split()
    monitor_rows = [int(r) for r in split["split"]["monitor"]]
    fit_rows = [int(r) for r in split["split"]["fit"]]
    guard = C.v1_access_guard
    guard.arm(gated=(C.v1c.LIPO_LABELS,))
    try:
        # The gated artifact is opened exactly once.  Hash the same bytes in
        # memory and scan those bytes; a separate sha256_file() would be a
        # second open and would correctly be denied by the one-shot guard.
        raw = guard.read_bytes_with_grant(C.v1c.LIPO_LABELS)
        labels_sha = C.v1_fit_labels._verify_label_bytes(raw)
        scanned = C.v1_fit_labels._scan(
            raw.decode("utf-8"),
            requested_label_keys=(str(r) for r in monitor_rows),
            path=C.v1c.LIPO_LABELS,
        )
    finally:
        guard.disarm()
    values = scanned["values"]
    meta = scanned["meta"]
    audit = scanned["audit"]
    C.v1_fit_labels.check_artifact_metadata(meta, set(fit_rows) | set(monitor_rows))
    y = np.asarray([values[str(r)] for r in monitor_rows], dtype=np.float64)
    if y.shape != (C.SPLIT_SIZES["monitor"],) or not np.isfinite(y).all():
        raise ValueError("Monitor label vector is invalid")
    if int(audit["label_values_constructed"]) != C.SPLIT_SIZES["monitor"]:
        raise PermissionError("Monitor label construction count is wrong")
    if int(audit["label_values_skipped"]) != C.SPLIT_SIZES["fit"]:
        raise PermissionError("Monitor label skip count is wrong")
    if int(audit["unrequested_values_constructed"]) != 0:
        raise PermissionError("unrequested label values were constructed")
    audit.update(
        {
            "labels_sha256": labels_sha,
            "manifest_sha256": C.sha256_file(C.PREDICTIONS / "MANIFEST.json"),
            "manifest_lock_sha256": (C.PREDICTIONS / "MANIFEST.sha256").read_text().strip(),
            "monitor_row_sha256": C.row_sha256_bytes(monitor_rows),
            "monitor_label_vector_sha256": hashlib.sha256(np.ascontiguousarray(y, dtype="<f8").tobytes()).hexdigest(),
            "reserved_rows_in_artifact": 0,
            "official_valid_test_rows_in_artifact": 0,
        }
    )
    return {
        "y": y,
        "rows": monitor_rows,
        "audit": audit,
        "meta": {k: meta[k] for k in ("protocol_id", "status", "row_sha256", "split_sha256", "scope")},
        "manifest": manifest,
    }


__all__ = ["load_monitor_labels"]

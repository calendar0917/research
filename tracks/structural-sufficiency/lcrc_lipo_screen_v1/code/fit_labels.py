"""Fit-only label access with a selective JSON scanner.

``labels_fit_monitor.json`` stores Fit and Monitor labels in one JSON object.
The training entry point is allowed to obtain **only** the 2352 Fit values.
This module therefore does not call ``json.load`` on the artifact: it scans the
raw text, constructs the Fit numeric tokens, and *skips* the Monitor numeric
tokens without converting them to Python numbers.

The audit record returned with the labels documents this property:

* ``label_values_constructed == 2352`` (Fit only)
* ``unrequested_values_constructed == 0``
* every requested Fit row key was present exactly once
* the artifact's own metadata (scope, row hash, split hash) matches the frozen
  Phase 1a values

Nothing here is reused by :mod:`monitor_labels`, which has its own gate and its
own request set and refuses to run before the prediction freeze.
"""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

import access_guard
import screen_common as C

METADATA_FIELDS = ("protocol_id", "status", "scope", "rows", "row_sha256", "split_sha256")


class LabelArtifactError(RuntimeError):
    """Raised when the label artifact does not match the frozen metadata."""


class _Scanner:
    """Small index-based JSON scanner that can *skip* label values."""

    def __init__(self, text: str):
        self.text = text
        self.i = 0

    def ws(self) -> None:
        t = self.text
        n = len(t)
        while self.i < n and t[self.i] in " \t\r\n":
            self.i += 1

    def peek(self) -> str:
        return "" if self.i >= len(self.text) else self.text[self.i]

    def expect(self, ch: str) -> None:
        self.ws()
        if self.peek() != ch:
            raise LabelArtifactError(f"expected {ch!r} at offset {self.i}, got {self.peek()!r}")
        self.i += 1

    def string(self) -> str:
        self.ws()
        if self.peek() != '"':
            raise LabelArtifactError(f"expected a JSON string at offset {self.i}")
        start = self.i
        self.i += 1
        n = len(self.text)
        while self.i < n:
            ch = self.text[self.i]
            if ch == "\\":
                self.i += 2
                continue
            if ch == '"':
                self.i += 1
                return json.loads(self.text[start : self.i])
            self.i += 1
        raise LabelArtifactError("unterminated JSON string")

    def number_token(self) -> str:
        """Consume a JSON number token and return its raw text (no float())."""
        self.ws()
        start = self.i
        n = len(self.text)
        allowed = set("0123456789+-.eE")
        while self.i < n and self.text[self.i] in allowed:
            self.i += 1
        token = self.text[start : self.i]
        if not token:
            raise LabelArtifactError(f"expected a JSON number at offset {start}")
        try:
            json.loads(token)
        except json.JSONDecodeError as exc:  # pragma: no cover - malformed artifact
            raise LabelArtifactError(f"invalid number token {token!r}") from exc
        return token

    def skip_value(self) -> None:
        """Skip any JSON value without constructing it."""
        self.ws()
        ch = self.peek()
        if ch == "{":
            self.i += 1
            self.ws()
            if self.peek() == "}":
                self.i += 1
                return
            while True:
                self.string()
                self.expect(":")
                self.skip_value()
                self.ws()
                if self.peek() == ",":
                    self.i += 1
                    continue
                self.expect("}")
                return
        if ch == "[":
            self.i += 1
            self.ws()
            if self.peek() == "]":
                self.i += 1
                return
            while True:
                self.skip_value()
                self.ws()
                if self.peek() == ",":
                    self.i += 1
                    continue
                self.expect("]")
                return
        if ch == '"':
            self.string()
            return
        for literal in ("true", "false", "null"):
            if self.text.startswith(literal, self.i):
                self.i += len(literal)
                return
        self.number_token()

    def parse_value(self) -> Any:
        """Construct a label-free metadata value (rows/scope/strings)."""
        self.ws()
        if self.peek() in "{[":
            start = self.i
            self.skip_value()
            return json.loads(self.text[start : self.i])
        if self.peek() == '"':
            return self.string()
        for literal, value in (("true", True), ("false", False), ("null", None)):
            if self.text.startswith(literal, self.i):
                self.i += len(literal)
                return value
        return json.loads(self.number_token())


def _scan(
    text: str,
    *,
    requested_label_keys: Iterable[str],
    path: Path | str,
) -> tuple[dict[str, float], dict[str, Any], dict[str, Any]]:
    wanted = {str(k) for k in requested_label_keys}
    if not wanted:
        raise ValueError("no label keys requested")
    meta: dict[str, Any] = {}
    values: dict[str, float] = {}
    skipped = 0
    total = 0
    duplicates = 0
    want_meta = set(METADATA_FIELDS)

    scan = _Scanner(text)
    scan.ws()
    scan.expect("{")
    scan.ws()
    if scan.peek() != "}":
        while True:
            key = scan.string()
            scan.expect(":")
            if key == "labels":
                scan.ws()
                scan.expect("{")
                scan.ws()
                if scan.peek() != "}":
                    while True:
                        label_key = scan.string()
                        scan.expect(":")
                        total += 1
                        if label_key in values:
                            duplicates += 1
                        if label_key in wanted:
                            value = float(scan.number_token())
                            if not np.isfinite(value):
                                raise LabelArtifactError("non-finite label value")
                            values[label_key] = value
                        else:
                            if scan.peek() not in " \t\r\n-0123456789":
                                raise LabelArtifactError(
                                    "unexpected non-numeric label value; refusing to construct it"
                                )
                            scan.skip_value()
                            skipped += 1
                        scan.ws()
                        if scan.peek() == ",":
                            scan.i += 1
                            continue
                        scan.expect("}")
                        break
                else:
                    scan.i += 1
            elif key in want_meta:
                meta[key] = scan.parse_value()
            else:
                scan.skip_value()
            scan.ws()
            if scan.peek() == ",":
                scan.i += 1
                continue
            scan.expect("}")
            break
    scan.ws()
    if text[scan.i :].strip():
        raise LabelArtifactError("trailing content after the JSON object")
    missing = sorted(wanted - set(values))
    if missing:
        raise LabelArtifactError(f"{len(missing)} requested label keys are absent")
    if duplicates:
        raise LabelArtifactError(f"{duplicates} duplicate label keys")
    audit = {
        "path": C.track_rel(path),
        "artifact_bytes": len(text.encode("utf-8")),
        "requested_label_keys": len(wanted),
        "label_values_constructed": len(values),
        "label_values_skipped": skipped,
        "total_label_keys_in_artifact": total,
        "unrequested_values_constructed": 0,
        "metadata_fields_constructed": sorted(meta),
        "label_values_used_for_features": 0,
    }
    return values, meta, audit


def scan_label_artifact(path: Path | str, requested_label_keys: Iterable[str]) -> dict[str, Any]:
    """Read a gated label artifact and construct exactly the requested values."""
    text = access_guard.read_text_with_grant(path)
    values, meta, audit = _scan(text, requested_label_keys=requested_label_keys, path=path)
    return {"values": values, "meta": meta, "audit": audit}


# ---------------------------------------------------------------------------
# artifact metadata checks
# ---------------------------------------------------------------------------


def check_artifact_metadata(meta: Mapping[str, Any], label_rows_expected: set[int]) -> None:
    if meta.get("protocol_id") != "ssrc-phase1a-lipo-v1":
        raise LabelArtifactError(f"unexpected label protocol {meta.get('protocol_id')!r}")
    if meta.get("status") != "FROZEN_FIT_MONITOR_ONLY":
        raise LabelArtifactError(f"unexpected label status {meta.get('status')!r}")
    scope = meta.get("scope") or {}
    if int(scope.get("fit_rows", -1)) != C.SPLIT_SIZES["fit"]:
        raise LabelArtifactError("label scope fit_rows mismatch")
    if int(scope.get("monitor_rows", -1)) != C.SPLIT_SIZES["monitor"]:
        raise LabelArtifactError("label scope monitor_rows mismatch")
    if int(scope.get("reserved_rows_excluded", -1)) != C.SPLIT_SIZES["reserved"]:
        raise LabelArtifactError("label scope reserved_rows_excluded mismatch")
    if scope.get("reserved_labels_present") is not False:
        raise PermissionError("label artifact declares reserved labels present")
    if scope.get("official_valid_test_labels_present") is not False:
        raise PermissionError("label artifact declares official valid/test labels present")
    rows = [int(r) for r in meta.get("rows", [])]
    if len(rows) != len(set(rows)):
        raise LabelArtifactError("duplicate rows in label artifact")
    if set(rows) != label_rows_expected:
        raise LabelArtifactError("label artifact row set != frozen Fit U Monitor")
    if meta.get("row_sha256") != C.EXPECTED_LABELS_ROW_HASH:
        raise LabelArtifactError("label artifact row hash mismatch")
    if meta.get("split_sha256") != C.EXPECTED_SHA256["split_phase1a"]:
        raise LabelArtifactError("label artifact split hash mismatch")


def _verify_label_bytes(raw: bytes) -> str:
    actual = hashlib.sha256(raw).hexdigest()
    if actual != C.EXPECTED_SHA256["labels_fit_monitor"]:
        raise PermissionError(f"label artifact sha256 {actual} != frozen")
    return actual


# ---------------------------------------------------------------------------
# Fit labels
# ---------------------------------------------------------------------------


def load_fit_labels(*, arm_guard: bool = True) -> dict[str, Any]:
    """Return Fit labels in frozen split-row order.  Never returns Monitor values."""
    split_info = C.load_frozen_split()
    fit_rows = split_info["split"]["fit"]
    monitor_rows = split_info["split"]["monitor"]
    reserved = set(split_info["split"]["reserved"])
    if (set(fit_rows) | set(monitor_rows)) & reserved:
        raise PermissionError("reserved rows overlap the Fit/Monitor supervision sets")

    if arm_guard:
        access_guard.arm()
    raw = access_guard.read_bytes_with_grant(C.LIPO_LABELS)
    actual_sha = _verify_label_bytes(raw)
    text = raw.decode("utf-8")
    values, meta, audit = _scan(text, requested_label_keys=(str(r) for r in fit_rows), path=C.LIPO_LABELS)
    check_artifact_metadata(meta, set(fit_rows) | set(monitor_rows))

    y = np.asarray([values[str(r)] for r in fit_rows], dtype=np.float64)
    if y.shape != (C.SPLIT_SIZES["fit"],):
        raise LabelArtifactError(f"Fit label vector has shape {y.shape}")
    if not np.isfinite(y).all():
        raise LabelArtifactError("non-finite Fit label")
    audit.update(
        {
            "labels_sha256": actual_sha,
            "rows_in_label_artifact": len(meta["rows"]),
            "reserved_rows_in_artifact": 0,
            "official_valid_test_rows_in_artifact": 0,
            "fit_label_vector_sha256": hashlib.sha256(np.ascontiguousarray(y, dtype="<f8").tobytes()).hexdigest(),
            "split_row_hash_fit": split_info["hashes"]["fit"],
            "split_row_hash_monitor": split_info["hashes"]["monitor"],
            "guard": access_guard.guard_status(),
        }
    )
    return {
        "y": y,
        "rows": list(fit_rows),
        "labels_sha256": actual_sha,
        "audit": audit,
        "meta": {k: meta[k] for k in ("protocol_id", "status", "row_sha256", "split_sha256", "scope")},
    }


def fit_label_stats(y: np.ndarray) -> dict[str, Any]:
    """Fit-only target standardisation parameters (the only allowed label statistic)."""
    y = np.asarray(y, dtype=np.float64).reshape(-1)
    mean = float(y.mean())
    scale = float(y.std())
    if not np.isfinite(mean) or not np.isfinite(scale) or scale <= 0:
        raise ValueError("invalid Fit target standardisation")
    return {
        "source": "fit_labels_only",
        "n": int(y.shape[0]),
        "mean": mean,
        "scale": scale,
        "scale_definition": "population_std_float64",
        "monitor_or_reserved_used": False,
        "fit_label_vector_sha256": hashlib.sha256(np.ascontiguousarray(y, dtype="<f8").tobytes()).hexdigest(),
    }


__all__ = [
    "LabelArtifactError",
    "METADATA_FIELDS",
    "scan_label_artifact",
    "load_fit_labels",
    "fit_label_stats",
    "check_artifact_metadata",
]

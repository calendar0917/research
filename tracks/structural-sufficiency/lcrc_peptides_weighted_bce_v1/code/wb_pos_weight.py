"""Train-only positive-class weights for Peptides-func.

The intervention is exactly ``w_j = N_negative,j / N_positive,j`` on the
official Train split.  There is no clipping, smoothing, square root, rescaling
or upper bound.  ``antimicrobial`` therefore has ``w < 1`` by construction.
"""

from __future__ import annotations

import hashlib
import json
from typing import Any, Sequence

import numpy as np

import wb_common as C


class PosWeightError(RuntimeError):
    pass


def label_counts(y_train: np.ndarray, labels: Sequence[str] = C.LABELS) -> dict[str, Any]:
    """Count positives/negatives per label and derive the exact ratio weight."""
    y = np.asarray(y_train)
    labels = tuple(labels)
    if y.ndim != 2:
        raise PosWeightError(f"train labels must be 2-D, got {y.shape}")
    if y.shape[1] != len(labels):
        raise PosWeightError(f"train labels {y.shape} do not match {len(labels)} labels")
    if y.shape[0] != int(C.SPLIT_SIZES["train"]):
        raise PosWeightError(f"train rows {y.shape[0]} != {C.SPLIT_SIZES['train']}")
    if not np.isfinite(y.astype(np.float64)).all():
        raise PosWeightError("train labels contain non-finite entries")
    unique = np.unique(y)
    if not np.isin(unique, [0, 1]).all():
        raise PosWeightError(f"train labels are not binary: unique={unique.tolist()}")

    y64 = y.astype(np.float64)
    n = int(y64.shape[0])
    rows: list[dict[str, Any]] = []
    weights: list[float] = []
    zero_positive: list[str] = []
    for j, name in enumerate(labels):
        pos = int((y64[:, j] == 1.0).sum())
        neg = int((y64[:, j] == 0.0).sum())
        if pos + neg != n:
            raise PosWeightError(f"label {name}: pos+neg={pos + neg} != n={n}")
        if pos == 0:
            zero_positive.append(name)
            continue
        weight = float(neg) / float(pos)
        if not np.isfinite(weight) or weight <= 0.0:
            raise PosWeightError(f"label {name}: invalid pos_weight {weight}")
        rows.append(
            {
                "label_index": int(j),
                "label": name,
                "train_n": n,
                "train_pos": pos,
                "train_neg": neg,
                "train_pos_rate": float(pos) / float(n),
                "pos_weight": weight,
            }
        )
        weights.append(weight)
    if zero_positive:
        raise PosWeightError(f"labels with zero Train positives: {zero_positive}")
    return {"rows": rows, "pos_weight": weights, "n_train": n}


def build_payload(y_train: np.ndarray, labels: Sequence[str] = C.LABELS) -> dict[str, Any]:
    labels = tuple(labels)
    counts = label_counts(y_train, labels)
    y = np.asarray(y_train)
    return {
        "round_id": C.ROUND_ID,
        "dataset": C.DATASET,
        "source_split": "official_train_only",
        "source_path": C.track_rel(C.TRAIN_LABELS),
        "source_file_sha256": C.sha256_file(C.TRAIN_LABELS),
        "train_labels_array_sha256": C.data_sha256(y),
        "n_train": int(counts["n_train"]),
        "n_labels": int(len(labels)),
        "label_order": list(labels),
        "labels": counts["rows"],
        "pos_weight": counts["pos_weight"],
        "formula": "pos_weight_j = train_negative_j / train_positive_j",
        "transformations": "none (no clipping, smoothing, square root, normalisation or upper bound)",
        "zero_positive_labels": [],
        "valid_used": False,
        "test_used": False,
        "generated_utc": C.utcnow(),
        "generator_code_sha256": C.sha256_file(C.CODE_DIR / "wb_pos_weight.py"),
    }


def verify_payload(
    payload: dict[str, Any],
    y_train: np.ndarray,
    *,
    labels: Sequence[str] = C.LABELS,
) -> dict[str, Any]:
    """Re-derive every weight from the labels actually passed to training."""
    labels = tuple(labels)
    required = {
        "round_id",
        "source_split",
        "source_file_sha256",
        "train_labels_array_sha256",
        "n_train",
        "n_labels",
        "label_order",
        "labels",
        "pos_weight",
        "valid_used",
        "test_used",
    }
    missing = sorted(required.difference(payload))
    if missing:
        raise PosWeightError(f"pos_weight payload missing keys: {missing}")
    if payload["round_id"] != C.ROUND_ID:
        raise PosWeightError(f"round_id {payload['round_id']!r} != {C.ROUND_ID!r}")
    if payload["source_split"] != "official_train_only":
        raise PosWeightError(f"pos_weight source_split is not Train-only: {payload['source_split']!r}")
    if payload["valid_used"] is not False or payload["test_used"] is not False:
        raise PosWeightError("pos_weight payload claims Validation/Test usage")
    if tuple(payload["label_order"]) != labels:
        raise PosWeightError(f"label order mismatch: {payload['label_order']} != {list(labels)}")
    if int(payload["n_labels"]) != len(labels):
        raise PosWeightError("n_labels mismatch")
    if int(payload["n_train"]) != int(np.asarray(y_train).shape[0]):
        raise PosWeightError("n_train mismatch")

    counts = label_counts(y_train, labels)
    expected = np.asarray(counts["pos_weight"], dtype=np.float64)
    stored = np.asarray(payload["pos_weight"], dtype=np.float64)
    if stored.shape != expected.shape:
        raise PosWeightError(f"pos_weight shape {stored.shape} != {expected.shape}")
    if not np.array_equal(stored, expected):
        raise PosWeightError(
            f"stored pos_weight does not exactly equal re-derived Train ratio: "
            f"max_abs_delta={float(np.max(np.abs(stored - expected)))}"
        )
    rows = payload["labels"]
    if len(rows) != len(labels):
        raise PosWeightError("label row count mismatch")
    for j, (row, expected_row) in enumerate(zip(rows, counts["rows"], strict=True)):
        for key in ("label_index", "label", "train_n", "train_pos", "train_neg"):
            if row[key] != expected_row[key]:
                raise PosWeightError(f"row {j} {key}: {row[key]!r} != {expected_row[key]!r}")
        if not np.isclose(
            float(row["pos_weight"]), float(expected_row["pos_weight"]), rtol=0.0, atol=0.0
        ):
            raise PosWeightError(f"row {j} pos_weight mismatch")
    if C.data_sha256(y_train) != payload["train_labels_array_sha256"]:
        raise PosWeightError("train label array SHA-256 mismatch")
    if C.sha256_file(C.TRAIN_LABELS) != payload["source_file_sha256"]:
        raise PosWeightError("train label file SHA-256 mismatch")
    return {
        "status": "PASS",
        "n_train": int(counts["n_train"]),
        "n_labels": int(len(labels)),
        "weights_exact_train_ratio": True,
        "zero_positive_labels": [],
        "valid_used": False,
        "test_used": False,
        "train_labels_array_sha256": C.data_sha256(y_train),
        "source_file_sha256": payload["source_file_sha256"],
        "pos_weight": [float(x) for x in counts["pos_weight"]],
    }


def payload_sha256(payload: dict[str, Any]) -> str:
    canonical = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return canonical

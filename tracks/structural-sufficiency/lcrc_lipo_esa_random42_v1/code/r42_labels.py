"""Independent, split-aware label access.

Only this module ever extracts the ``exp`` column from the raw CSV.  The
training entry points consume :class:`TrainValidLabels`, which structurally
cannot return Test labels: it stores exactly the Train3360 and Validation420
vectors and raises :class:`TestLabelAccessError` for anything else.

Test labels are never written into the repository.  The one-shot scorer reads
them locally from the hash-verified CSV after a pre-test manifest has been
frozen and writes a one-time lock file.
"""

from __future__ import annotations

import csv
import hashlib
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import r42_common as C


class TestLabelAccessError(RuntimeError):
    """Raised when a training/validation path asks for Test labels."""


def _read_exp_column(path: Path | str = C.CSV_PATH) -> tuple[list[str], list[str], np.ndarray]:
    path = Path(path)
    C.verify_sha256(path, C.CSV_SHA256, "Lipophilicity.csv")
    ids: list[str] = []
    smiles: list[str] = []
    labels: list[float] = []
    with open(path, newline="", encoding="utf-8") as fh:
        reader = csv.reader(fh)
        header = next(reader)
        if header != ["CMPD_CHEMBLID", "exp", "smiles"]:
            raise ValueError(f"unexpected CSV header {header!r}")
        for i, row in enumerate(reader):
            if len(row) != 3:
                raise ValueError(f"row {i}: expected 3 fields, got {len(row)}")
            compound, exp, smi = row
            try:
                value = float(exp)
            except ValueError as exc:
                raise ValueError(f"row {i}: non-numeric exp {exp!r}") from exc
            if not np.isfinite(value):
                raise ValueError(f"row {i}: non-finite exp {value!r}")
            ids.append(compound)
            smiles.append(smi)
            labels.append(value)
    if len(ids) != C.N_MOLECULES:
        raise ValueError(f"expected {C.N_MOLECULES} rows, parsed {len(ids)}")
    return ids, smiles, np.asarray(labels, dtype=np.float64)


def _vector_sha(values: Sequence[float] | np.ndarray) -> str:
    return C.sha256_bytes(np.ascontiguousarray(np.asarray(values, dtype="<f8")).tobytes())


def freeze_train_valid_labels(split_path: Path | str = C.SPLIT_FILE, out_path: Path | str = C.LABELS_TRAIN_VALID) -> dict[str, Any]:
    """Write the Train+Validation label artifact (Test values never included)."""
    split = C.read_json(split_path)
    ids, smiles, y = _read_exp_column()
    rows = split["rows"]
    for name in ("train", "valid"):
        if len(rows[name]) != C.SPLIT_SIZES[name]:
            raise ValueError(f"{name} split size mismatch")
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "source_csv_sha256": C.CSV_SHA256,
        "split_file_sha256": C.sha256_file(split_path),
        "columns": ["CMPD_CHEMBLID", "exp"],
        "scope": "train_and_valid_only",
        "test_labels_included": False,
        "rows": {name: [int(r) for r in rows[name]] for name in ("train", "valid")},
        "values": {name: [float(y[int(r)]) for r in rows[name]] for name in ("train", "valid")},
        "vector_sha256": {name: _vector_sha(y[np.asarray(rows[name], dtype=np.int64)]) for name in ("train", "valid")},
        "compound_ids_sha256": {
            name: hashlib.sha256("\n".join(ids[int(r)] for r in rows[name]).encode("utf-8")).hexdigest()
            for name in ("train", "valid")
        },
        "row_sha256_int64": {name: C.rows_sha256_int64(rows[name]) for name in ("train", "valid")},
        "smiles_sha256": {name: hashlib.sha256("\n".join(smiles[int(r)] for r in rows[name]).encode("utf-8")).hexdigest() for name in ("train", "valid")},
    }
    C.write_json(out_path, payload)
    return payload


@dataclass(frozen=True)
class TrainValidLabels:
    """Train/Valid labels with a hard-coded scope guard."""

    y_train: np.ndarray
    y_valid: np.ndarray
    rows_train: tuple[int, ...]
    rows_valid: tuple[int, ...]
    audit: Mapping[str, Any]

    @property
    def split_names(self) -> tuple[str, ...]:
        return ("train", "valid")

    def get(self, split: str) -> np.ndarray:
        if split == "train":
            return self.y_train
        if split == "valid":
            return self.y_valid
        if split == "test":
            raise TestLabelAccessError("Test labels are unavailable to training/validation code")
        raise KeyError(split)

    def __getitem__(self, split: str) -> np.ndarray:
        return self.get(split)


def load_train_valid_labels(path: Path | str = C.LABELS_TRAIN_VALID, split_path: Path | str = C.SPLIT_FILE) -> TrainValidLabels:
    payload = C.read_json(path)
    if payload.get("protocol_id") != C.PROTOCOL_ID:
        raise PermissionError("label artifact protocol mismatch")
    if payload.get("test_labels_included") is not False:
        raise PermissionError("label artifact is not certified Test-free")
    split = C.read_json(split_path)
    if C.sha256_file(split_path) != payload["split_file_sha256"]:
        raise PermissionError("label artifact was built for a different split file")
    out: dict[str, np.ndarray] = {}
    rows: dict[str, tuple[int, ...]] = {}
    for name in ("train", "valid"):
        values = np.asarray(payload["values"][name], dtype=np.float64).reshape(-1)
        row_list = tuple(int(x) for x in payload["rows"][name])
        if row_list != tuple(int(x) for x in split["rows"][name]):
            raise PermissionError(f"{name} label rows do not match the frozen split")
        if values.shape != (C.SPLIT_SIZES[name],):
            raise ValueError(f"{name} label vector has the wrong length")
        if not np.isfinite(values).all():
            raise ValueError(f"{name} labels are not finite")
        if _vector_sha(values) != payload["vector_sha256"][name]:
            raise PermissionError(f"{name} label vector hash mismatch")
        if C.rows_sha256_int64(row_list) != payload["row_sha256_int64"][name]:
            raise PermissionError(f"{name} label row hash mismatch")
        out[name] = values
        rows[name] = row_list
    audit = {
        "path": C.track_rel(path),
        "sha256": C.sha256_file(path),
        "scope": payload["scope"],
        "test_labels_included": False,
        "n_train": int(out["train"].size),
        "n_valid": int(out["valid"].size),
        "vector_sha256": dict(payload["vector_sha256"]),
    }
    return TrainValidLabels(out["train"], out["valid"], rows["train"], rows["valid"], audit)


def assert_train_valid_only(labels: Any) -> TrainValidLabels:
    """Reject anything that is not the frozen Train/Valid label object."""
    if not isinstance(labels, TrainValidLabels):
        if isinstance(labels, Mapping) and "test" in labels:
            raise TestLabelAccessError("a mapping containing Test labels may not enter a training path")
        raise TestLabelAccessError("training requires a TrainValidLabels object")
    names = set(labels.split_names)
    if names != {"train", "valid"}:
        raise TestLabelAccessError(f"unexpected label scopes {sorted(names)}")
    try:
        labels.get("test")
    except TestLabelAccessError:
        return labels
    raise TestLabelAccessError("TrainValidLabels unexpectedly returned Test labels")


class TestLabelVault:
    """One-shot access to the local Test labels, bound to a frozen manifest."""

    def __init__(self, manifest_path: Path | str, lock_path: Path | str):
        self.manifest_path = Path(manifest_path)
        self.lock_path = Path(lock_path)

    def read(self, *, expected_manifest_sha256: str) -> np.ndarray:
        if self.lock_path.exists():
            raise PermissionError(f"Test labels were already scored once ({C.track_rel(self.lock_path)} exists)")
        if not self.manifest_path.exists():
            raise PermissionError("pre-test manifest is missing; Test labels stay locked")
        actual = C.sha256_file(self.manifest_path)
        if actual != expected_manifest_sha256:
            raise PermissionError("pre-test manifest hash mismatch")
        _, _, y = _read_exp_column()
        split = C.read_json(C.SPLIT_FILE)
        rows = np.asarray(split["rows"]["test"], dtype=np.int64)
        if rows.shape != (C.SPLIT_SIZES["test"],):
            raise ValueError("Test split has the wrong size")
        return np.asarray(y[rows], dtype=np.float64)

    def write_lock(self, payload: Mapping[str, Any]) -> None:
        if self.lock_path.exists():
            raise PermissionError("Test scoring lock already exists")
        C.write_json(self.lock_path, dict(payload))


__all__ = [
    "TestLabelAccessError",
    "TrainValidLabels",
    "freeze_train_valid_labels",
    "load_train_valid_labels",
    "assert_train_valid_only",
    "TestLabelVault",
]

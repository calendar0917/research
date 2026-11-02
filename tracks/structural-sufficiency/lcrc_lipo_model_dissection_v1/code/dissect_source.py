"""Repeat-aware loader for the frozen ChemBench Lipophilicity data.

The published split files are the only source of repeat identity.  The loader
verifies the CSV bytes, the split-file bytes and the int64 row-index bytes
against the hashes frozen in :mod:`dissect_common` before returning anything.
No split is generated, repaired, filtered or reordered.

``labels`` scope:
    ``"none"``          structure + SMILES only (no target byte decoded)
    ``"train_valid"``   train/validation targets, used by training/selection
    ``"all"``           every target; only the frozen Test scoring step uses it
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

import dissect_common as C


def _read_split_file(path):
    series = pd.read_pickle(path)
    parts = [list(part) for part in series.to_list()]
    if len(parts) != 3:
        raise ValueError(f"{path.name}: expected 3 split parts, got {len(parts)}")
    return [[int(x) for x in part] for part in parts]


def load_lipop(repeat: int, *, labels: str = "train_valid", verify: bool = True) -> dict[str, Any]:
    repeat = int(repeat)
    if repeat not in C.REPEATS:
        raise ValueError(f"repeat must be one of {C.REPEATS}, got {repeat!r}")
    if labels not in ("none", "train_valid", "all"):
        raise ValueError(f"labels must be none/train_valid/all, got {labels!r}")

    split_path = C.CHEMBENCH_LIPOP / C.SPLIT_FILES[repeat]
    if verify:
        got = C.sha256_file(split_path)
        if got != C.SPLIT_FILE_SHA256[repeat]:
            raise ValueError(f"{split_path.name}: sha256 {got} != frozen {C.SPLIT_FILE_SHA256[repeat]}")
        got = C.sha256_file(C.LIPOP_CSV)
        if got != C.LIPOP_CSV_SHA256:
            raise ValueError(f"{C.LIPOP_CSV.name}: sha256 {got} != frozen {C.LIPOP_CSV_SHA256}")

    train, valid, test = _read_split_file(split_path)
    sizes = (len(train), len(valid), len(test))
    if sizes != C.SPLIT_SIZES:
        raise ValueError(f"repeat {repeat}: split sizes {sizes} != {C.SPLIT_SIZES}")
    for name, part, expected in zip(("train", "valid", "test"), (train, valid, test), C.SPLIT_ROWS_SHA256[repeat]):
        if verify:
            got = C.row_sha256(part)
            if got != expected:
                raise ValueError(f"repeat {repeat}/{name}: row-index sha256 {got} != frozen {expected}")

    if labels == "none":
        df = pd.read_csv(C.LIPOP_CSV, usecols=["smiles"])
    else:
        df = pd.read_csv(C.LIPOP_CSV)
    if len(df) != C.N_MOLECULES:
        raise ValueError(f"{C.LIPOP_CSV.name}: {len(df)} rows != {C.N_MOLECULES}")
    if "smiles" not in df.columns:
        raise ValueError("no 'smiles' column")

    n = len(df)
    union = set(train) | set(valid) | set(test)
    if len(union) != n:
        raise ValueError(f"repeat {repeat}: split union {len(union)} != {n}")
    if set(train) & set(valid) or set(train) & set(test) or set(valid) & set(test):
        raise ValueError(f"repeat {repeat}: split partitions overlap")
    for part in (train, valid, test):
        if len(set(part)) != len(part):
            raise ValueError(f"repeat {repeat}: duplicate row id inside a split part")
        if min(part) < 0 or max(part) >= n:
            raise ValueError(f"repeat {repeat}: row id outside [0, {n})")

    smiles = [str(s) for s in df["smiles"].tolist()]
    y: dict[str, np.ndarray] | None = None
    if labels != "none":
        if C.LABEL_COLUMN not in df.columns:
            raise ValueError(f"label column {C.LABEL_COLUMN!r} not found")
        raw = np.asarray(df[C.LABEL_COLUMN].to_numpy(), dtype=np.float64)
        if not np.isfinite(raw).all():
            raise ValueError("non-finite target values")
        y = {
            "train": raw[np.asarray(train, dtype=np.int64)],
            "valid": raw[np.asarray(valid, dtype=np.int64)],
        }
        if labels == "all":
            y["test"] = raw[np.asarray(test, dtype=np.int64)]

    return {
        "repeat": repeat,
        "split_file": C.SPLIT_FILES[repeat],
        "split_file_sha256": C.SPLIT_FILE_SHA256[repeat],
        "split_rows_sha256": list(C.SPLIT_ROWS_SHA256[repeat]),
        "sizes": list(sizes),
        "smiles": smiles,
        "rows": {"train": train, "valid": valid, "test": test},
        "y": y,
        "labels_scope": labels,
        "csv_sha256": C.LIPOP_CSV_SHA256,
        "source_commit": C.CHEMBENCH_COMMIT,
        "label_column": C.LABEL_COLUMN,
        "units": "logD (log10 octanol/water distribution coefficient, experimental)",
    }


__all__ = ["load_lipop"]

"""Published-split and label access for ``tsb-lipo-feature-ladder-v1``.

The three Lipophilicity repeats are the *published* ChemBench random splits,
already vendored (byte-identical) inside this repository by
``lcrc_molnet_benchmark_pilot_v1``.  Nothing is regenerated here:

* split rows come from the pinned ``Lipop.random{122,123,124}.ind.pkl``;
* SMILES and labels come from the pinned ``Lipop.csv.gz``.

Every loader verifies the recorded SHA-256 first and raises on drift, so a
silent change of split or label bytes is impossible.  Labels are **only**
returned by :func:`split_labels` / :func:`all_labels`; the feature builders in
:mod:`tsb_features` call :func:`load_smiles` and never see a target.
"""

from __future__ import annotations

import pickle
from functools import lru_cache
from typing import Any

import numpy as np

import tsb_common as C


@lru_cache(maxsize=1)
def _frame() -> tuple[list[str], np.ndarray]:
    import pandas as pd

    actual = C.sha256_file(C.CHEMBENCH_CSV)
    if actual != C.CHEMBENCH_CSV_SHA256:
        raise ValueError(f"Lipop.csv.gz sha256 drift: {actual} != {C.CHEMBENCH_CSV_SHA256}")
    frame = pd.read_csv(C.CHEMBENCH_CSV)
    if list(frame.columns) != ["smiles", "exp"]:
        raise ValueError(f"unexpected Lipop.csv.gz columns: {list(frame.columns)}")
    smiles = [str(s) for s in frame["smiles"].tolist()]
    labels = np.asarray(frame["exp"], dtype=np.float64)
    if labels.shape != (4200,):
        raise ValueError(f"unexpected label shape {labels.shape}")
    if not np.all(np.isfinite(labels)):
        raise ValueError("non-finite Lipophilicity labels in the pinned source file")
    return smiles, labels


def load_smiles() -> list[str]:
    """The 4,200 SMILES in source row order.  No label leaves this function."""
    return list(_frame()[0])


def all_labels() -> np.ndarray:
    return _frame()[1].copy()


@lru_cache(maxsize=1)
def load_splits() -> dict[int, dict[str, Any]]:
    """Load and hash-verify the three published splits.

    The pickle is a ``pandas.Series`` of three index lists.  Its positional
    order is *asserted* rather than trusted: each list is hashed and matched
    against the recorded Train/Validation/Test row hashes.
    """
    splits: dict[int, dict[str, Any]] = {}
    for suffix in C.SPLIT_SUFFIXES:
        path = C.CHEMBENCH_DIR / C.SPLIT_FILES[suffix]
        actual = C.sha256_file(path)
        if actual != C.SPLIT_SHA256[suffix]:
            raise ValueError(f"{path.name} sha256 drift: {actual} != {C.SPLIT_SHA256[suffix]}")
        with path.open("rb") as handle:
            series = pickle.load(handle)
        parts = [np.asarray(list(part), dtype=np.int64) for part in series]
        if len(parts) != 3:
            raise ValueError(f"{path.name}: expected 3 index parts, got {len(parts)}")
        by_hash = {C.row_sha256(part): part for part in parts}
        expected = C.SPLIT_ROWS_SHA256[suffix]
        missing = {name: h for name, h in expected.items() if h not in by_hash}
        if missing:
            raise ValueError(f"{path.name}: row hashes not found for {sorted(missing)}")
        roles = {name: by_hash[h] for name, h in expected.items()}
        for name, rows in roles.items():
            if int(rows.size) != C.SPLIT_SIZES[name]:
                raise ValueError(f"{path.name} {name}: size {rows.size} != {C.SPLIT_SIZES[name]}")
            if np.unique(rows).size != rows.size:
                raise ValueError(f"{path.name} {name}: duplicate row ids")
        union = np.unique(np.concatenate([roles["train"], roles["valid"], roles["test"]]))
        if union.size != 4200:
            raise ValueError(f"{path.name}: split union has {union.size} rows, expected 4200")
        splits[suffix] = {
            "suffix": int(suffix),
            "file": C.SPLIT_FILES[suffix],
            "file_sha256": actual,
            "rows": roles,
            "rows_sha256": {name: C.row_sha256(rows) for name, rows in roles.items()},
        }
    return splits


def split(suffix: int) -> dict[str, Any]:
    return load_splits()[int(suffix)]


def split_labels(suffix: int, include_test: bool = False) -> dict[str, np.ndarray]:
    """Targets for one split.  ``test`` is only returned when explicitly asked."""
    labels = all_labels()
    rows = split(suffix)["rows"]
    out = {
        "train": labels[rows["train"]],
        "valid": labels[rows["valid"]],
    }
    if include_test:
        out["test"] = labels[rows["test"]]
    return out


def sample_order(train_rows: np.ndarray) -> np.ndarray:
    """Deterministic nested order of the official Train rows.

    Order key = ``sha256("20261104:<original row id>")`` ascending.  It depends
    only on the frozen seed and the *original source row id*, so it is
    independent of split-file ordering, of the label values, and of the arm.
    The three training sizes are prefixes of this single order.
    """
    rows = np.asarray(train_rows, dtype=np.int64)
    keys = np.asarray(
        [C.sha256_bytes(f"{C.SEED}:{int(r)}".encode()) for r in rows], dtype="<U64"
    )
    order = np.argsort(keys, kind="stable")
    return rows[order]


def nested_prefix(train_rows: np.ndarray, n: int) -> np.ndarray:
    ordered = sample_order(train_rows)
    if n > ordered.size:
        raise ValueError(f"requested {n} rows but Train has only {ordered.size}")
    return ordered[: int(n)]


def data_manifest() -> dict[str, Any]:
    splits = load_splits()
    smiles = load_smiles()
    labels = all_labels()
    return {
        "schema": "tsb-data-manifest-v1",
        "protocol_id": C.PROTOCOL_ID,
        "dataset": C.DATASET,
        "units": C.LABEL_UNITS,
        "n_rows_total": len(smiles),
        "source": {
            "csv": C.track_rel(C.CHEMBENCH_CSV),
            "csv_sha256": C.CHEMBENCH_CSV_SHA256,
            "label_column": "exp",
            "label_sha256": C.sha256_bytes(np.ascontiguousarray(labels).tobytes()),
            "smiles_sha256": C.sha256_bytes("\n".join(smiles).encode()),
            "upstream_repository": "https://github.com/shenwanxiang/ChemBench",
            "upstream_commit": "0b10d4c1ac72b6929503694a5f60446824736391",
        },
        "splits": {
            str(suffix): {
                "file": meta["file"],
                "file_sha256": meta["file_sha256"],
                "sizes": {name: int(rows.size) for name, rows in meta["rows"].items()},
                "rows_sha256": meta["rows_sha256"],
            }
            for suffix, meta in splits.items()
        },
        "split_generation": "published ChemBench random splits; no new split is generated",
        "test_exposure": C.TEST_EXPOSURE_LABEL,
    }

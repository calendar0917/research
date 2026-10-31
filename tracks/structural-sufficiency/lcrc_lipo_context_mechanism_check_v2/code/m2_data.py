"""Frozen data views for the context mechanism check v2.

The split, labels and LCRC structure cache are consumed read-only and
hash-verified.  Unlike the historical v1 training/validation path, this round is
allowed to read the Test labels, but only in the explicitly exposed exploratory
sense: Test labels are read from the raw CSV whose SHA256 is frozen, and the
resulting vector must reproduce the historical one-shot Test label hash.
Training never receives a Test cache or Test labels.
"""

from __future__ import annotations

import csv
from typing import Any, Mapping, Sequence

import numpy as np

import m2_common as C

import lcrc_common as prep_common  # noqa: E402
import lcrc_data as prep_data  # noqa: E402
import feature_spec as prep_feature_spec  # noqa: E402
import r42_data as D  # noqa: E402


# ---------------------------------------------------------------------------
# raw CSV labels
# ---------------------------------------------------------------------------


def read_exp_column() -> tuple[list[str], list[str], np.ndarray]:
    """Read the hash-verified CSV; only this module ever extracts ``exp``."""
    C.verify_sha256(C.CSV_PATH, C.CSV_SHA256, "Lipophilicity.csv")
    ids: list[str] = []
    smiles: list[str] = []
    labels: list[float] = []
    with open(C.CSV_PATH, newline="", encoding="utf-8") as fh:
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


# ---------------------------------------------------------------------------
# frozen split / labels
# ---------------------------------------------------------------------------


def load_frozen_split() -> dict[str, Any]:
    C.verify_sha256(C.V1_SPLIT_FILE, C.V1_SPLIT_SHA256, "v1 split file")
    split = C.read_json(C.V1_SPLIT_FILE)
    if split.get("protocol_id") != "lcrc-lipo-esa-random42-v1":
        raise PermissionError("split file protocol id mismatch")
    if split.get("source_csv_sha256") != C.CSV_SHA256:
        raise PermissionError("split file was built from a different CSV")
    if int(split.get("n_molecules")) != C.N_MOLECULES:
        raise PermissionError("split file molecule count mismatch")
    if int(split.get("seed")) != C.SPLIT_SEED:
        raise PermissionError("split seed drifted")
    if {k: int(v) for k, v in split["sizes"].items()} != C.SPLIT_SIZES:
        raise PermissionError("split sizes drifted")
    rows: dict[str, list[int]] = {}
    for name in ("train", "valid", "test"):
        values = [int(x) for x in split["rows"][name]]
        if len(values) != C.SPLIT_SIZES[name]:
            raise ValueError(f"{name} split has {len(values)} rows, expected {C.SPLIT_SIZES[name]}")
        if len(set(values)) != len(values):
            raise ValueError(f"{name} split contains duplicates")
        if min(values) < 0 or max(values) >= C.N_MOLECULES:
            raise ValueError(f"{name} split row outside 0..4199")
        digest = C.rows_sha256_int64(values)
        if digest != C.SPLIT_SHA256[name]:
            raise PermissionError(f"{name} split sha256 {digest} != frozen {C.SPLIT_SHA256[name]}")
        if values[:5] != C.SPLIT_FIRST_ROWS[name]:
            raise PermissionError(f"{name} split first five rows drifted")
        rows[name] = values
    if set(rows["train"]) & set(rows["valid"]):
        raise ValueError("Train/Validation overlap")
    if set(rows["train"]) & set(rows["test"]):
        raise ValueError("Train/Test overlap")
    if set(rows["valid"]) & set(rows["test"]):
        raise ValueError("Validation/Test overlap")
    if set(rows["train"]) | set(rows["valid"]) | set(rows["test"]) != set(range(C.N_MOLECULES)):
        raise ValueError("the three splits do not cover exactly 0..4199")
    return split


def load_train_valid_labels(split: Mapping[str, Any]) -> dict[str, Any]:
    C.verify_sha256(C.V1_LABELS_FILE, C.V1_LABELS_SHA256, "v1 labels file")
    payload = C.read_json(C.V1_LABELS_FILE)
    if payload.get("test_labels_included") is not False:
        raise PermissionError("the Train/Valid label artifact is not certified Test-free")
    y: dict[str, np.ndarray] = {}
    for name in ("train", "valid"):
        if [int(x) for x in payload["rows"][name]] != [int(x) for x in split["rows"][name]]:
            raise PermissionError(f"{name} label rows do not match the frozen split")
        values = np.asarray(payload["values"][name], dtype=np.float64).reshape(-1)
        if values.shape != (C.SPLIT_SIZES[name],):
            raise ValueError(f"{name} label vector has the wrong length")
        digest = C.vector_sha256_float64(values)
        expected = C.TARGET_VECTOR_SHA256 if name == "train" else C.VALID_VECTOR_SHA256
        if digest != expected:
            raise PermissionError(f"{name} label vector sha256 {digest} != frozen {expected}")
        y[name] = values
    return {
        "y_train": y["train"],
        "y_valid": y["valid"],
        "audit": {
            "path": C.track_rel(C.V1_LABELS_FILE),
            "sha256": C.V1_LABELS_SHA256,
            "n_train": int(y["train"].size),
            "n_valid": int(y["valid"].size),
            "test_labels_included": False,
            "train_vector_sha256": C.TARGET_VECTOR_SHA256,
            "valid_vector_sha256": C.VALID_VECTOR_SHA256,
        },
    }


def load_test_labels(split: Mapping[str, Any]) -> dict[str, Any]:
    """Read Test420 labels from the hash-verified raw CSV, in frozen row order."""
    ids, smiles, all_y = read_exp_column()
    rows = [int(r) for r in split["rows"]["test"]]
    if C.rows_sha256_int64(rows) != C.SPLIT_SHA256["test"]:
        raise PermissionError("Test420 row hash drifted")
    y = np.asarray(all_y[np.asarray(rows, dtype=np.int64)], dtype=np.float64)
    digest = C.vector_sha256_float64(y)
    if digest != C.TEST_VECTOR_SHA256:
        raise PermissionError(f"Test420 label vector sha256 {digest} != frozen one-shot {C.TEST_VECTOR_SHA256}")
    if not np.isfinite(y).all():
        raise ValueError("Test420 labels are not finite")
    return {
        "y_test": y,
        "rows": rows,
        "audit": {
            "csv_path": C.track_rel(C.CSV_PATH),
            "csv_sha256": C.CSV_SHA256,
            "rows_sha256": C.SPLIT_SHA256["test"],
            "label_vector_sha256": digest,
            "n_test": int(y.size),
            "first_five_rows": rows[:5],
            "compound_ids_sha256": __import__("hashlib").sha256(
                "\n".join(ids[r] for r in rows).encode("utf-8")
            ).hexdigest(),
            "smiles_sha256": __import__("hashlib").sha256(
                "\n".join(smiles[r] for r in rows).encode("utf-8")
            ).hexdigest(),
            "scope": "EXPOSED_TEST_EXPLORATORY",
        },
    }


# ---------------------------------------------------------------------------
# frozen LCRC structure cache
# ---------------------------------------------------------------------------


def _cache_candidates() -> list[Any]:
    return [
        C.EXP_DIR / "cache/lcrc_all4200.pt",
        C.V1_CONTEXT_CACHE,
        C.V1_ESA_CACHE,
    ]


def load_lcrc_cache() -> dict[str, Any]:
    path = next((p for p in _cache_candidates() if p.exists()), None)
    if path is None:
        raise FileNotFoundError("no frozen lcrc_all4200.pt cache found")
    cache = prep_data.load_cache(path)
    fingerprint = prep_common.content_fingerprint(cache)
    if fingerprint != C.LCRC_CACHE_CONTENT_FINGERPRINT:
        raise PermissionError(
            f"LCRC cache content fingerprint {fingerprint} != frozen {C.LCRC_CACHE_CONTENT_FINGERPRINT}"
        )
    if cache["meta"].get("labels_present") is not False:
        raise PermissionError("LCRC cache is not certified label-free")
    if int(cache["n_molecules"]) != C.N_MOLECULES:
        raise PermissionError("LCRC cache molecule count mismatch")
    return {
        "cache": cache,
        "path": path,
        "file_sha256": C.sha256_file(path),
        "content_fingerprint": fingerprint,
        "labels_present": False,
    }


def _select(cache: Mapping[str, Any], rows: Sequence[int], label: str) -> dict[str, Any]:
    selected = D.select_cache(cache, [int(r) for r in rows], split_label=label)
    if int(selected["n_molecules"]) != len(rows):
        raise ValueError(f"{label}: selected cache has the wrong size")
    if [str(x) for x in selected["molecule_ids"]] != [str(int(r)) for r in rows]:
        raise ValueError(f"{label}: cache row identity drifted")
    return selected


def build_views(*, include_test: bool = True) -> dict[str, Any]:
    split = load_frozen_split()
    labels = load_train_valid_labels(split)
    test = load_test_labels(split) if include_test else None
    built = load_lcrc_cache()
    cache_all = built["cache"]
    spec = prep_feature_spec.lipo_v1_spec()
    rows_train = [int(r) for r in split["rows"]["train"]]
    rows_valid = [int(r) for r in split["rows"]["valid"]]
    rows_test = [int(r) for r in split["rows"]["test"]]
    train_cache = _select(cache_all, rows_train, "train3360")
    valid_cache = _select(cache_all, rows_valid, "valid420")
    test_cache = _select(cache_all, rows_test, "test420") if include_test else None
    for name, cache, rows in (
        ("train3360", train_cache, rows_train),
        ("valid420", valid_cache, rows_valid),
    ):
        if [str(x) for x in cache["smiles"]] != [str(split["smiles"][r]) for r in rows]:
            raise ValueError(f"{name}: cache SMILES identity drifted")
    if test_cache is not None and [str(x) for x in test_cache["smiles"]] != [
        str(split["smiles"][r]) for r in rows_test
    ]:
        raise ValueError("test420: cache SMILES identity drifted")
    train_norm = prep_data.fit_norm(train_cache, spec, source="train3360_structures")
    norm_fingerprint = prep_data.norm_fingerprint(train_norm)
    if norm_fingerprint != C.LCRC_TRAIN_NORM_FINGERPRINT:
        raise PermissionError("LCRC Train3360 structure normalisation drifted from v1")
    target_norm = {
        "mean": C.TARGET_MEAN,
        "scale": C.TARGET_SCALE,
        "source": "frozen_train3360_labels_only",
        "fit_on": "Train3360 only",
        "n": int(labels["y_train"].size),
        "label_vector_sha256": C.TARGET_VECTOR_SHA256,
        "valid_or_test_used": False,
    }
    return {
        "split": split,
        "labels": labels,
        "test": test,
        "spec": spec,
        "cache_built": built,
        "train_cache": train_cache,
        "valid_cache": valid_cache,
        "test_cache": test_cache,
        "train_norm": train_norm,
        "train_norm_fingerprint": norm_fingerprint,
        "target_norm": target_norm,
        "rows_train": rows_train,
        "rows_valid": rows_valid,
        "rows_test": rows_test,
    }


def target_tensor(y: np.ndarray, norm: Mapping[str, Any]):
    import torch

    return torch.as_tensor(
        (np.asarray(y, dtype=np.float64) - float(norm["mean"])) / float(norm["scale"]),
        dtype=torch.float32,
    )


def denormalise(pred: np.ndarray, norm: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(pred, dtype=np.float64) * float(norm["scale"]) + float(norm["mean"])


__all__ = [
    "read_exp_column",
    "load_frozen_split",
    "load_train_valid_labels",
    "load_test_labels",
    "load_lcrc_cache",
    "build_views",
    "target_tensor",
    "denormalise",
]

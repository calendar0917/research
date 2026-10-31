"""Data views that reuse the frozen random42 artifacts byte-for-byte.

Only Train3360 and Validation420 are materialised.  The frozen split file is
verified without dereferencing its ``test`` row list; the label artifact
structurally cannot return Test labels.  A runtime ``open`` audit proves that no
Test label / Test prediction artifact is touched by the data, gate, training or
analysis entry points.
"""

from __future__ import annotations

import contextlib
import os
import shutil
import sys
from pathlib import Path
from typing import Any, Iterator, Mapping, Sequence

import numpy as np

import x1_common as C

import r42_data as D  # noqa: E402  (v1 code is on sys.path via x1_common)
import r42_labels as L  # noqa: E402

import feature_spec as prep_feature_spec  # noqa: E402
import lcrc_common as prep_common  # noqa: E402
import lcrc_data as prep_data  # noqa: E402

#: Frozen content fingerprint of the label-free 4,200-molecule lipo_v1 cache.
LCRC_CACHE_CONTENT_FINGERPRINT = "23ec00be4fb7e3cf93967cd62911bd932159b1d7e0b43226b23cf7c6a3718e03"

#: Path fragments that must never appear in a file opened by this round.
FORBIDDEN_READ_TOKENS = (
    "test420",
    "metrics_test",
    "labels_test",
    "test_read",
    "/predictions/",
)

# ---------------------------------------------------------------------------
# runtime read audit
# ---------------------------------------------------------------------------

_OPEN_LOG: list[str] = []
_AUDIT_ACTIVE = False
_AUDIT_INSTALLED = False


def _audit_hook(event: str, args: tuple[Any, ...]) -> None:
    if not _AUDIT_ACTIVE or event != "open":
        return
    try:
        path = args[0]
        text = os.fspath(path) if isinstance(path, (str, bytes, os.PathLike)) else str(path)
    except Exception:
        return
    if isinstance(text, bytes):
        text = text.decode("utf-8", "replace")
    _OPEN_LOG.append(str(text))


def install_read_audit() -> None:
    global _AUDIT_INSTALLED
    if _AUDIT_INSTALLED:
        return
    sys.addaudithook(_audit_hook)
    _AUDIT_INSTALLED = True


@contextlib.contextmanager
def read_audit() -> Iterator[list[str]]:
    global _AUDIT_ACTIVE
    install_read_audit()
    _OPEN_LOG.clear()
    _AUDIT_ACTIVE = True
    try:
        yield _OPEN_LOG
    finally:
        _AUDIT_ACTIVE = False


def forbidden_reads(paths: Sequence[str]) -> list[str]:
    return [p for p in paths if any(token in p for token in FORBIDDEN_READ_TOKENS)]


# ---------------------------------------------------------------------------
# frozen split / labels
# ---------------------------------------------------------------------------


def _verify_frozen_split(payload: Mapping[str, Any]) -> dict[str, Any]:
    if payload.get("protocol_id") != "lcrc-lipo-esa-random42-v1":
        raise PermissionError("split file protocol id mismatch")
    if payload.get("source_csv_sha256") != C.CSV_SHA256:
        raise PermissionError("split file was built from a different CSV")
    if int(payload.get("n_molecules")) != C.N_MOLECULES:
        raise PermissionError("split file molecule count mismatch")
    if int(payload.get("seed")) != C.SPLIT_SEED:
        raise PermissionError("split seed drifted")
    if {k: int(v) for k, v in payload["sizes"].items()} != {
        "train": C.SPLIT_SIZES["train"],
        "valid": C.SPLIT_SIZES["valid"],
        "test": C.N_MOLECULES - C.SPLIT_SIZES["train"] - C.SPLIT_SIZES["valid"],
    }:
        raise PermissionError("split sizes drifted")
    rows = payload["rows"]
    train = [int(x) for x in rows["train"]]
    valid = [int(x) for x in rows["valid"]]
    for name, values in (("train", train), ("valid", valid)):
        expected_size = C.SPLIT_SIZES[name]
        if len(values) != expected_size:
            raise ValueError(f"{name} split has {len(values)} rows, expected {expected_size}")
        if len(set(values)) != expected_size:
            raise ValueError(f"{name} split contains duplicates")
        if min(values) < 0 or max(values) >= C.N_MOLECULES:
            raise ValueError(f"{name} split row outside 0..4199")
        digest = C.rows_sha256_int64(values)
        if digest != C.SPLIT_SHA256[name]:
            raise PermissionError(f"{name} split sha256 {digest} != frozen {C.SPLIT_SHA256[name]}")
        if values[:5] != C.SPLIT_FIRST_ROWS[name]:
            raise PermissionError(f"{name} split first five rows drifted")
    if set(train) & set(valid):
        raise ValueError("Train/Validation overlap")
    complement = set(range(C.N_MOLECULES)) - set(train) - set(valid)
    if len(complement) != C.N_MOLECULES - C.SPLIT_SIZES["train"] - C.SPLIT_SIZES["valid"]:
        raise ValueError("Train+Validation sizes do not leave the declared Test size")
    # No Test row value is used.  Only the declared Test size is checked; the
    # Test label / prediction / per-molecule artifacts are never opened.
    if len(rows["test"]) != C.N_MOLECULES - C.SPLIT_SIZES["train"] - C.SPLIT_SIZES["valid"]:
        raise PermissionError("declared Test size drifted")
    return {
        "file": C.track_rel(C.V1_SPLIT_FILE),
        "sha256": C.V1_SPLIT_SHA256,
        "train_rows_sha256": C.SPLIT_SHA256["train"],
        "valid_rows_sha256": C.SPLIT_SHA256["valid"],
        "n_train": len(train),
        "n_valid": len(valid),
        "n_test_declared": len(complement),
        "test_row_values_used": False,
        "first_five": dict(C.SPLIT_FIRST_ROWS),
        "multi_component_row": C.MULTI_COMPONENT_ROW,
        "multi_component_split": (
            "train" if C.MULTI_COMPONENT_ROW in train else "valid" if C.MULTI_COMPONENT_ROW in valid else "test(not inspected)"
        ),
    }


def load_frozen_split_and_labels() -> tuple[dict[str, Any], L.TrainValidLabels]:
    C.verify_sha256(C.V1_SPLIT_FILE, C.V1_SPLIT_SHA256, "v1 split file")
    C.verify_sha256(C.V1_LABELS_FILE, C.V1_LABELS_SHA256, "v1 labels file")
    split = C.read_json(C.V1_SPLIT_FILE)
    split_audit = _verify_frozen_split(split)

    payload = C.read_json(C.V1_LABELS_FILE)
    if payload.get("protocol_id") != "lcrc-lipo-esa-random42-v1":
        raise PermissionError("label artifact protocol mismatch")
    if payload.get("test_labels_included") is not False or payload.get("scope") != "train_and_valid_only":
        raise PermissionError("label artifact is not certified Test-free")
    if payload.get("split_file_sha256") != C.V1_SPLIT_SHA256:
        raise PermissionError("label artifact was built for a different split file")
    for name in ("train", "valid"):
        if [int(x) for x in payload["rows"][name]] != [int(x) for x in split["rows"][name]]:
            raise PermissionError(f"{name} label rows do not match the frozen split")
    if C.vector_sha256_float64(payload["values"]["valid"]) != C.VALID_VECTOR_SHA256:
        raise PermissionError("Validation420 label vector hash drifted")
    if C.vector_sha256_float64(payload["values"]["train"]) != C.TARGET_VECTOR_SHA256:
        raise PermissionError("Train3360 label vector hash drifted")

    labels = L.load_train_valid_labels(path=C.V1_LABELS_FILE, split_path=C.V1_SPLIT_FILE)
    labels = L.assert_train_valid_only(labels)
    try:
        labels.get("test")
    except L.TestLabelAccessError:
        pass
    else:
        raise PermissionError("TrainValidLabels unexpectedly returned Test labels")
    split["_audit"] = split_audit
    return split, labels


def load_target_norm(labels: L.TrainValidLabels) -> dict[str, Any]:
    norm = D.fit_target_scaler(labels.y_train)
    if norm["mean"] != C.TARGET_MEAN or norm["scale"] != C.TARGET_SCALE:
        raise PermissionError("Train3360 target scaler drifted from the frozen values")
    if norm["label_vector_sha256"] != C.TARGET_VECTOR_SHA256:
        raise PermissionError("Train3360 label vector hash drifted")
    return norm


# ---------------------------------------------------------------------------
# LCRC structure cache
# ---------------------------------------------------------------------------


def _materialise_cache() -> dict[str, Any]:
    C.CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = C.CACHE_DIR / "lcrc_all4200.pt"
    source = "rebuilt_from_raw_graph_artifact"
    v1_cache = C.V1 / "cache/lcrc_all4200.pt"
    if not cache_path.exists() and v1_cache.exists():
        shutil.copy2(v1_cache, cache_path)
        source = "byte_copy_of_v1_cache"
    elif cache_path.exists():
        source = "existing_round_cache"
    built = D.build_lcrc_cache(path=cache_path)
    cache = built["cache"]
    if cache["meta"].get("labels_present") is not False:
        raise PermissionError("LCRC cache is not certified label-free")
    if int(cache["n_molecules"]) != C.N_MOLECULES:
        raise PermissionError("LCRC cache molecule count mismatch")
    if built["content_fingerprint"] != LCRC_CACHE_CONTENT_FINGERPRINT:
        raise PermissionError(
            f"LCRC cache content fingerprint {built['content_fingerprint']} != frozen {LCRC_CACHE_CONTENT_FINGERPRINT}"
        )
    v1_fingerprint = None
    if v1_cache.exists():
        v1_obj = prep_data.load_cache(v1_cache)
        v1_fingerprint = prep_common.content_fingerprint(v1_obj)
        if v1_fingerprint != built["content_fingerprint"]:
            raise PermissionError("LCRC cache content differs from the frozen v1 cache")
    return {
        "cache": cache,
        "path": built["path"],
        "file_sha256": built["file_sha256"],
        "content_fingerprint": built["content_fingerprint"],
        "rebuilt": bool(built["rebuilt"]),
        "source": source,
        "v1_cache_path": C.track_rel(v1_cache) if v1_cache.exists() else None,
        "v1_content_fingerprint": v1_fingerprint,
        "labels_present": False,
    }


def build_lcrc_views(split: Mapping[str, Any], labels: L.TrainValidLabels) -> dict[str, Any]:
    labels = L.assert_train_valid_only(labels)
    built = _materialise_cache()
    cache_all = built["cache"]
    rows_train = [int(r) for r in split["rows"]["train"]]
    rows_valid = [int(r) for r in split["rows"]["valid"]]
    if len(rows_train) != C.SPLIT_SIZES["train"] or len(rows_valid) != C.SPLIT_SIZES["valid"]:
        raise ValueError("split sizes drifted")
    if C.rows_sha256_int64(rows_train) != C.SPLIT_SHA256["train"]:
        raise PermissionError("Train3360 row hash drifted")
    if C.rows_sha256_int64(rows_valid) != C.SPLIT_SHA256["valid"]:
        raise PermissionError("Validation420 row hash drifted")
    train_cache = D.select_cache(cache_all, rows_train, split_label="train3360")
    valid_cache = D.select_cache(cache_all, rows_valid, split_label="valid420")
    if [str(x) for x in train_cache["molecule_ids"]] != [str(r) for r in rows_train]:
        raise ValueError("Train3360 cache row identity drifted")
    if [str(x) for x in valid_cache["molecule_ids"]] != [str(r) for r in rows_valid]:
        raise ValueError("Validation420 cache row identity drifted")
    if [str(x) for x in train_cache["smiles"]] != [str(split["smiles"][r]) for r in rows_train]:
        raise ValueError("Train3360 cache SMILES identity drifted")
    if [str(x) for x in valid_cache["smiles"]] != [str(split["smiles"][r]) for r in rows_valid]:
        raise ValueError("Validation420 cache SMILES identity drifted")
    spec = prep_feature_spec.lipo_v1_spec()
    train_norm = prep_data.fit_norm(train_cache, spec, source="train3360_structures")
    norm_fingerprint = prep_data.norm_fingerprint(train_norm)
    if norm_fingerprint != C.LCRC_TRAIN_NORM_FINGERPRINT:
        raise PermissionError("LCRC Train3360 structure normalisation drifted from v1")
    target_norm = load_target_norm(labels)
    return {
        "built": built,
        "spec": spec,
        "train_cache": train_cache,
        "valid_cache": valid_cache,
        "train_norm": train_norm,
        "train_norm_fingerprint": norm_fingerprint,
        "target_norm": target_norm,
        "rows_train": rows_train,
        "rows_valid": rows_valid,
        "labels": labels,
        "split_audit": split.get("_audit"),
    }


# ---------------------------------------------------------------------------
# element-class role tables
# ---------------------------------------------------------------------------


def element_role_ids(spec: Mapping[str, Any] | Any) -> dict[str, list[int]]:
    """Map ``spec.atom_type_metadata`` elements onto the two fixed role sets."""
    metadata = list(spec.atom_type_metadata)
    hetero: list[int] = []
    carbon_halogen: list[int] = []
    for index, meta in enumerate(metadata):
        element = meta.get("element")
        if element is None:
            continue
        if str(element) in C.HETERO_ELEMENTS:
            hetero.append(int(index))
        if str(element) in C.CARBON_HALOGEN_ELEMENTS:
            carbon_halogen.append(int(index))
    if not hetero or not carbon_halogen:
        raise ValueError("role element sets are empty for this spec")
    return {"hetero": hetero, "carbon_halogen": carbon_halogen}


def role_tables(spec: Mapping[str, Any] | Any) -> dict[str, Any]:
    ids = element_role_ids(spec)
    metadata = list(spec.atom_type_metadata)
    return {
        "hetero_elements": list(C.HETERO_ELEMENTS),
        "carbon_halogen_elements": list(C.CARBON_HALOGEN_ELEMENTS),
        "hetero_type_ids": ids["hetero"],
        "carbon_halogen_type_ids": ids["carbon_halogen"],
        "hetero_type_names": [metadata[i]["name"] for i in ids["hetero"]],
        "carbon_halogen_type_names": [metadata[i]["name"] for i in ids["carbon_halogen"]],
        "unassigned_type_names": [
            meta["name"]
            for index, meta in enumerate(metadata)
            if index not in ids["hetero"] and index not in ids["carbon_halogen"]
        ],
    }


def role_coverage(cache: Mapping[str, Any], spec: Mapping[str, Any] | Any) -> dict[str, Any]:
    """Per-molecule coverage of the two fixed element classes."""
    atom_type = np.asarray(cache["atom_type"], dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    tables = role_tables(spec)
    hetero = np.isin(atom_type, np.asarray(tables["hetero_type_ids"], dtype=np.int64))
    ch = np.isin(atom_type, np.asarray(tables["carbon_halogen_type_ids"], dtype=np.int64))
    n_mol = int(cache["n_molecules"])
    counts = {"both": 0, "hetero_only": 0, "carbon_halogen_only": 0, "neither": 0}
    n_hetero_atoms = 0
    n_ch_atoms = 0
    for m in range(n_mol):
        lo, hi = int(atom_ptr[m]), int(atom_ptr[m + 1])
        h = bool(hetero[lo:hi].any())
        c = bool(ch[lo:hi].any())
        n_hetero_atoms += int(hetero[lo:hi].sum())
        n_ch_atoms += int(ch[lo:hi].sum())
        if h and c:
            counts["both"] += 1
        elif h:
            counts["hetero_only"] += 1
        elif c:
            counts["carbon_halogen_only"] += 1
        else:
            counts["neither"] += 1
    total_atoms = int(atom_type.size)
    return {
        "n_molecules": n_mol,
        "n_atoms": total_atoms,
        "n_hetero_atoms": n_hetero_atoms,
        "n_carbon_halogen_atoms": n_ch_atoms,
        "hetero_atom_fraction": n_hetero_atoms / max(total_atoms, 1),
        "carbon_halogen_atom_fraction": n_ch_atoms / max(total_atoms, 1),
        "molecule_counts": counts,
        "molecule_fractions": {k: v / max(n_mol, 1) for k, v in counts.items()},
        "molecules_with_hetero_fraction": (counts["both"] + counts["hetero_only"]) / max(n_mol, 1),
        "molecules_with_carbon_halogen_fraction": (counts["both"] + counts["carbon_halogen_only"]) / max(n_mol, 1),
        "role_tables": tables,
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
    "LCRC_CACHE_CONTENT_FINGERPRINT",
    "FORBIDDEN_READ_TOKENS",
    "read_audit",
    "forbidden_reads",
    "load_frozen_split_and_labels",
    "load_target_norm",
    "build_lcrc_views",
    "element_role_ids",
    "role_tables",
    "role_coverage",
    "target_tensor",
    "denormalise",
]

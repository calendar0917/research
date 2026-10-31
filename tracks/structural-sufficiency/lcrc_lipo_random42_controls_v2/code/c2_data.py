"""Data views that reuse the frozen random42 v1 artifacts byte-for-byte.

Nothing here writes into the v1 directory.  The LCRC structure cache is
materialised under this round's ``cache/`` (copied from v1 when present, else
rebuilt from the committed label-free raw-graph artifact with the read-only
preparation code).  Only Train3360 and Validation420 views are selected.
"""

from __future__ import annotations

import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import c2_common as C

import r42_data as D  # noqa: E402  (v1 code is on sys.path via c2_common)
import r42_labels as L  # noqa: E402

import lcrc_data as prep_data  # noqa: E402
import feature_spec as prep_feature_spec  # noqa: E402


def load_frozen_split_and_labels() -> tuple[dict[str, Any], L.TrainValidLabels]:
    C.verify_sha256(C.V1_SPLIT_FILE, C.V1_SPLIT_SHA256, "v1 split file")
    C.verify_sha256(C.V1_LABELS_FILE, C.V1_LABELS_SHA256, "v1 labels file")
    split = D.load_split()
    labels = L.load_train_valid_labels()
    L.assert_train_valid_only(labels)
    return split, labels


def load_target_norm(labels: L.TrainValidLabels) -> dict[str, Any]:
    norm = D.fit_target_scaler(labels.y_train)
    if norm["mean"] != C.TARGET_MEAN or norm["scale"] != C.TARGET_SCALE:
        raise PermissionError("Train3360 target scaler drifted from the frozen values")
    if norm["label_vector_sha256"] != C.TARGET_VECTOR_SHA256:
        raise PermissionError("Train3360 label vector hash drifted")
    return norm


def _materialise_cache() -> dict[str, Any]:
    CACHE_DIR = C.CACHE_DIR
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    cache_path = CACHE_DIR / "lcrc_all4200.pt"
    source = "rebuilt_from_raw_graph_artifact"
    v1_cache = C.V1 / "cache/lcrc_all4200.pt"
    if not cache_path.exists() and v1_cache.exists():
        shutil.copy2(v1_cache, cache_path)
        source = "byte_copy_of_v1_cache"
    built = D.build_lcrc_cache(path=cache_path)
    cache = built["cache"]
    if cache["meta"].get("labels_present") is not False:
        raise PermissionError("LCRC cache is not certified label-free")
    if int(cache["n_molecules"]) != C.N_MOLECULES:
        raise PermissionError("LCRC cache molecule count mismatch")
    v1_fingerprint = None
    if v1_cache.exists():
        import lcrc_common as prep_common

        v1_cache_obj = prep_data.load_cache(v1_cache)
        v1_fingerprint = prep_common.content_fingerprint(v1_cache_obj)
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
    }


def load_structures_for_rows(rows: Sequence[int]) -> list[Any]:
    """Reconstruct only the requested molecules (never a Test row here)."""
    return D.load_structure_data(rows=[int(r) for r in rows])


def structure_dimensions() -> dict[str, int]:
    arrays = D.load_structure_arrays()
    return {
        "num_features": int(arrays["atom_feat"].shape[1]),
        "edge_dim": int(arrays["edge_attr"].shape[1]),
    }


def target_tensor(y: np.ndarray, norm: Mapping[str, Any]) -> torch.Tensor:
    return torch.as_tensor(
        (np.asarray(y, dtype=np.float64) - float(norm["mean"])) / float(norm["scale"]),
        dtype=torch.float32,
    )


def denormalise(pred: np.ndarray, norm: Mapping[str, Any]) -> np.ndarray:
    return np.asarray(pred, dtype=np.float64) * float(norm["scale"]) + float(norm["mean"])


__all__ = [
    "load_frozen_split_and_labels",
    "load_target_norm",
    "build_lcrc_views",
    "load_structures_for_rows",
    "structure_dimensions",
    "target_tensor",
    "denormalise",
]

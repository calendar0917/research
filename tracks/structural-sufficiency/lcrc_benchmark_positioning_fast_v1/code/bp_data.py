"""Cache IO, Train-only normalisation and batching helpers."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import bp_common as C
import lcrc_data as PREP_DATA
from feature_spec import FeatureSpec


def cache_rows_atoms(cache: Mapping[str, Any], rows: Sequence[int]) -> np.ndarray:
    ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([np.arange(ptr[int(r)], ptr[int(r) + 1], dtype=np.int64) for r in rows])


def cache_rows_bonds(cache: Mapping[str, Any], rows: Sequence[int]) -> np.ndarray:
    ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    rows = np.asarray(rows, dtype=np.int64)
    if not len(rows):
        return np.zeros(0, dtype=np.int64)
    return np.concatenate([np.arange(ptr[int(r)], ptr[int(r) + 1], dtype=np.int64) for r in rows])


def _scale_from_std(std: np.ndarray) -> tuple[np.ndarray, list[int]]:
    std = np.asarray(std, dtype=np.float64)
    scale = std.astype(np.float32)
    constant = np.flatnonzero(~np.isfinite(scale) | (scale == 0.0))
    scale[constant] = 1.0
    return scale, [int(x) for x in constant]


def fit_norm_rows(cache: Mapping[str, Any], spec: FeatureSpec, rows: Sequence[int], *, source: str) -> dict[str, Any]:
    """Fit scalar/global standardisation on exactly the supplied cache rows."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1 or not rows.size:
        raise ValueError("fit_norm_rows needs a non-empty 1-D row set")
    if np.any(rows < 0) or np.any(rows >= int(cache["n_molecules"])):
        raise ValueError("fit row outside cache")
    atom_rows = cache_rows_atoms(cache, rows)
    scalars = np.asarray(cache["atom_scalar"], dtype=np.float64)[atom_rows]
    globals_ = np.asarray(cache["global"], dtype=np.float64)[rows]
    if scalars.shape[1] != spec.n_scalars or globals_.shape[1] != spec.global_dim:
        raise ValueError("cache dimensions do not match feature spec")
    scalar_mean = scalars.mean(0).astype(np.float32)
    scalar_scale, scalar_const = _scale_from_std(scalars.std(0))
    global_mean = globals_.mean(0).astype(np.float32)
    global_scale, global_const = _scale_from_std(globals_.std(0))
    norm = {
        "source": str(source),
        "fit_split": "train_only",
        "fit_rows": int(rows.size),
        "fit_rows_sha256": C.index_hash(rows),
        "valid_or_test_used": False,
        "zero_variance_policy": "constant_fit_features_use_scale_1.0",
        "scalar_constant_indices": scalar_const,
        "global_constant_indices": global_const,
        "scalar_mean": scalar_mean,
        "scalar_scale": scalar_scale,
        "global_mean": global_mean,
        "global_scale": global_scale,
    }
    norm["norm_fingerprint"] = PREP_DATA.norm_fingerprint(norm)
    return norm


def fit_continuous_atom_norm(cache: Mapping[str, Any], rows: Sequence[int], *, start: int, dim: int) -> dict[str, Any]:
    """Train-only mean/scale for the retained continuous node-attribute block."""
    if dim <= 0:
        return {"dim": 0, "mean": [], "scale": [], "constant_indices": [], "fit_rows": int(len(rows))}
    atom_rows = cache_rows_atoms(cache, rows)
    values = np.asarray(cache["atom_feat"], dtype=np.float64)[atom_rows, start : start + dim]
    mean = values.mean(0).astype(np.float32)
    scale, constant = _scale_from_std(values.std(0))
    out = {
        "dim": int(dim),
        "mean": mean,
        "scale": scale,
        "constant_indices": constant,
        "fit_rows": int(len(rows)),
        "fit_atoms": int(values.shape[0]),
        "fit_rows_sha256": C.index_hash(np.asarray(rows, dtype=np.int64)),
        "source": "tu_train_only_continuous_atom_attributes",
    }
    out["fingerprint"] = C.sha256_bytes(
        np.asarray(mean, dtype=np.float32).tobytes() + np.asarray(scale, dtype=np.float32).tobytes()
    )
    return out


def save_cache(cache: Mapping[str, Any], path: Path | str) -> str:
    return PREP_DATA.save_cache(cache, path)


def load_cache(path: Path | str) -> dict[str, Any]:
    return PREP_DATA.load_cache(path)


def collate(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    return PREP_DATA.collate(cache, rows)


def subset_stats(cache: Mapping[str, Any], spec: FeatureSpec, rows: Sequence[int]) -> dict[str, Any]:
    rows = np.asarray(rows, dtype=np.int64)
    atom_rows = cache_rows_atoms(cache, rows)
    bond_rows = cache_rows_bonds(cache, rows)
    return {
        "rows": int(rows.size),
        "atoms": int(atom_rows.size),
        "bonds": int(bond_rows.size),
        "s1_codes": int(np.sum(np.diff(np.asarray(cache["nbr_ptr"])[atom_rows]).astype(np.int64)))
        if atom_rows.size
        else 0,
        "s2_paths": int(np.sum(np.diff(np.asarray(cache["path_ptr"])[atom_rows]).astype(np.int64)))
        if atom_rows.size
        else 0,
        "feature_spec_hash": spec.hash(),
    }


def cache_meta(cache: Mapping[str, Any]) -> dict[str, Any]:
    m = dict(cache.get("meta", {}))
    return {
        "n_molecules": int(cache["n_molecules"]),
        "n_atoms_total": int(m.get("n_atoms_total", 0)),
        "n_bonds_total": int(m.get("n_bonds_total", 0)),
        "n_s1_codes": int(m.get("n_s1_codes", 0)),
        "n_s2_paths": int(m.get("n_s2_paths", 0)),
        "build_seconds": float(m.get("build_seconds", 0.0)),
        "feature_spec_hash": str(m.get("feature_spec_hash", "")),
    }

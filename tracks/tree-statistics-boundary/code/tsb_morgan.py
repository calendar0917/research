"""``MORGAN_COUNT``: RDKit Morgan count fingerprint block (radius 2, 2048).

This is the standard chemistry-fingerprint reference *inside* the controlled
ladder.  It is deliberately not the same thing as the repository's external
``ECFP4 binary + 217 descriptors + XGBoost`` baseline: here the fingerprint is
count-valued, there are no physicochemical descriptors, and the rest of the
input is only ``G0``.

Hashed-bucket collisions are measured (not repaired): the unhashed Morgan
identifier set is compared with the 2048-bucket set, and the difference is
reported as ``collision_buckets`` in the feature manifest.
"""

from __future__ import annotations

from typing import Any, Sequence

import numpy as np
import scipy.sparse as sp

import tsb_common as C


def column_names() -> list[str]:
    return [f"morgan_r{C.MORGAN_RADIUS}_b{index:04d}" for index in range(C.MORGAN_NBITS)]


def vector(mol) -> tuple[np.ndarray, dict[str, int]]:
    """Dense 2048 count vector plus hash-collision diagnostics."""
    from rdkit.Chem import rdMolDescriptors

    dense = np.zeros(C.MORGAN_NBITS, dtype=np.float32)
    hashed = rdMolDescriptors.GetHashedMorganFingerprint(
        mol, C.MORGAN_RADIUS, nBits=C.MORGAN_NBITS
    )
    for index, value in hashed.GetNonzeroElements().items():
        dense[int(index)] = float(value)
    raw = rdMolDescriptors.GetMorganFingerprint(mol, C.MORGAN_RADIUS, useCounts=True)
    raw_size = len(raw.GetNonzeroElements())
    bucket_size = len(hashed.GetNonzeroElements())
    return dense, {"raw_environments": raw_size, "buckets_used": bucket_size,
                   "collision_buckets": max(0, raw_size - bucket_size)}


def build(mols: Sequence[Any]) -> tuple[sp.csr_matrix, dict[str, Any]]:
    import time

    rows, cols, data = [], [], []
    diagnostics = {"raw_environments": 0, "buckets_used": 0, "collision_buckets": 0}
    per_row_collisions: list[int] = []
    t0 = time.perf_counter()
    for i, mol in enumerate(mols):
        dense, info = vector(mol)
        nz = np.flatnonzero(dense)
        rows.extend([i] * nz.size)
        cols.extend(int(j) for j in nz)
        data.extend(float(dense[j]) for j in nz)
        for key in diagnostics:
            diagnostics[key] += int(info[key])
        per_row_collisions.append(int(info["collision_buckets"]))
    seconds = time.perf_counter() - t0
    matrix = sp.csr_matrix(
        (np.asarray(data, dtype=np.float32),
         (np.asarray(rows, dtype=np.int64), np.asarray(cols, dtype=np.int32))),
        shape=(len(mols), C.MORGAN_NBITS),
        dtype=np.float32,
    )
    matrix.sort_indices()
    meta = {
        "radius": C.MORGAN_RADIUS,
        "n_bits": C.MORGAN_NBITS,
        "use_counts": True,
        "rdkit_version": _rdkit_version(),
        "n_rows": int(len(mols)),
        "mean_raw_environments": diagnostics["raw_environments"] / max(1, len(mols)),
        "mean_buckets_used": diagnostics["buckets_used"] / max(1, len(mols)),
        "mean_collision_buckets": diagnostics["collision_buckets"] / max(1, len(mols)),
        "rows_with_collision": int(sum(1 for c in per_row_collisions if c > 0)),
        "density": float(matrix.nnz / (matrix.shape[0] * matrix.shape[1])),
        "seconds": float(seconds),
    }
    return matrix, meta


def _rdkit_version() -> str:
    try:
        import rdkit

        return str(rdkit.__version__)
    except Exception:  # pragma: no cover
        return "unknown"


def spec() -> dict[str, Any]:
    return {
        "arm_block": "MORGAN_COUNT",
        "kind": "RDKit Morgan count fingerprint",
        "radius": C.MORGAN_RADIUS,
        "n_bits": C.MORGAN_NBITS,
        "use_counts": True,
        "dim": C.MORGAN_NBITS,
        "column_names": column_names(),
        "rdkit_function": "rdMolDescriptors.GetHashedMorganFingerprint(mol, 2, nBits=2048) with useCounts semantics",
        "label_used": False,
    }

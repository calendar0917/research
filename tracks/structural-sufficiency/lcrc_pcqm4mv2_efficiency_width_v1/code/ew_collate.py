"""E1 -- equivalent vectorised batch collation.

``pq_data.collate`` (= ``lcrc_data.collate``) builds four per-index arrays with
Python-level per-molecule / per-atom loops::

    idx = np.concatenate([np.arange(s, e) for s, e in zip(starts, ends)])
    nbr_idx = np.concatenate([np.arange(p, p + c) for p, c in zip(...)])

At batch 128 that is ~4,100 ``np.arange`` calls per batch (128 molecules x 2
range sets + ~1,810 atoms x 2 range sets).  ``collate_v2`` replaces every loop
with the algebraically identical ``repeat``/``arange`` form

    local = arange(total) - repeat(exclusive_cumsum(counts), counts)
    idx   = repeat(starts, counts) + local

and builds the per-molecule offsets with one ``cumsum``.  **Every output field
is unchanged**: same keys, same dtype, same shape, same values.  The equality is
asserted per field against the frozen collation in
``tests/test_ew_round.py`` and in the GPU preflight ``ew_verify``.

Nothing else about the batch changes -- the model sees exactly the tensors the
historical code produced.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch

import ew_common as C  # noqa: F401  (puts the inherited code directories on sys.path)


def _segment_indices(starts: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Vectorised ``concatenate([arange(s, s+c) ...])`` with local ids.

    Returns ``(idx, local, total)`` where ``idx`` is the concatenation of the
    per-segment ranges and ``local`` is ``0..c-1`` inside each segment.  The
    frozen loop produces exactly these two arrays.
    """
    counts = np.asarray(counts, dtype=np.int64)
    starts = np.asarray(starts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), 0
    offsets = np.cumsum(counts) - counts
    positions = np.arange(total, dtype=np.int64)
    local = positions - np.repeat(offsets, counts)
    idx = np.repeat(starts, counts) + local
    return idx, local, total


def _ranges_v2(ptr: np.ndarray, rows: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    starts = ptr[rows].astype(np.int64, copy=False)
    ends = ptr[rows + 1].astype(np.int64, copy=False)
    counts = ends - starts
    if rows.size == 0:
        z = np.zeros(0, dtype=np.int64)
        return z, z, z, counts
    idx, local, _total = _segment_indices(starts, counts)
    batch = np.repeat(np.arange(rows.size, dtype=np.int64), counts)
    return idx, batch, local, counts


def collate_v2(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    """Identical output to ``pq_data.collate`` with loop-free index assembly."""
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError("rows must be 1-D")
    n_molecules = int(cache["n_molecules"])
    if np.any(rows < 0) or np.any(rows >= n_molecules):
        raise ValueError("row index outside cache")
    if len(rows) != len(set(int(x) for x in rows.tolist())):
        raise ValueError("duplicate row in batch")
    B = int(rows.size)
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    atom_idx, atom_batch, atom_local, atom_counts = _ranges_v2(atom_ptr, rows)
    bond_idx, bond_batch, _bond_local, bond_counts = _ranges_v2(bond_ptr, rows)
    if B:
        atom_off = np.concatenate(([0], np.cumsum(atom_counts)[:-1])).astype(np.int64, copy=False)
    else:  # pragma: no cover - defensive; collate is never called with B=0
        atom_off = np.zeros(0, dtype=np.int64)

    if bond_idx.size:
        bi = np.asarray(cache["bond_index"])[bond_idx].astype(np.int64, copy=False)
        bi = bi + atom_off[bond_batch].reshape(-1, 1)
    else:  # pragma: no cover - a batch without bonds is not part of this dataset
        bi = np.zeros((0, 2), dtype=np.int64)

    nbr_ptr = np.asarray(cache["nbr_ptr"])
    if atom_idx.size:
        nbr_counts = nbr_ptr[atom_idx + 1] - nbr_ptr[atom_idx]
        nbr_idx, _nbr_local, _ = _segment_indices(nbr_ptr[atom_idx], nbr_counts)
        nbr_atom = np.repeat(np.arange(atom_idx.size, dtype=np.int64), nbr_counts)
    else:  # pragma: no cover - defensive
        nbr_idx = np.zeros(0, dtype=np.int64)
        nbr_atom = np.zeros(0, dtype=np.int64)

    path_ptr = np.asarray(cache["path_ptr"])
    if atom_idx.size:
        path_counts = path_ptr[atom_idx + 1] - path_ptr[atom_idx]
        path_idx, _path_local, _ = _segment_indices(path_ptr[atom_idx], path_counts)
        path_graph = np.repeat(atom_batch, path_counts)
        path_atom = np.repeat(np.arange(atom_idx.size, dtype=np.int64), path_counts)
        path_end = np.asarray(cache["path_end"])[path_idx].astype(np.int64, copy=False) + atom_off[path_graph]
    else:  # pragma: no cover - defensive
        path_idx = np.zeros(0, dtype=np.int64)
        path_graph = np.zeros(0, dtype=np.int64)
        path_atom = np.zeros(0, dtype=np.int64)
        path_end = np.zeros(0, dtype=np.int64)

    def _t(arr: Any, dtype: Any) -> torch.Tensor:
        return torch.from_numpy(np.asarray(arr, dtype=dtype))

    #: Same key order as the frozen dict, for readable diffs.
    return {
        "n_graphs": B,
        "atom_feat": _t(np.asarray(cache["atom_feat"])[atom_idx], np.float32),
        "atom_type": _t(np.asarray(cache["atom_type"])[atom_idx], np.int64),
        "atom_scalar": _t(np.asarray(cache["atom_scalar"])[atom_idx], np.float32),
        "atom_batch": _t(atom_batch, np.int64),
        "atom_off": _t(atom_off, np.int64),
        "bond_feat": _t(np.asarray(cache["bond_feat"])[bond_idx], np.float32),
        "bond_type": _t(np.asarray(cache["bond_type"])[bond_idx], np.int64),
        "bond_index": _t(bi, np.int64),
        "bond_batch": _t(bond_batch, np.int64),
        "nbr_atom": _t(nbr_atom, np.int64),
        "nbr_code": _t(np.asarray(cache["nbr_code"])[nbr_idx], np.int64),
        "path_atom": _t(path_atom, np.int64),
        "path_cat": _t(np.asarray(cache["path_cat"])[path_idx], np.int64),
        "path_end63": _t(np.asarray(cache["path_end63"])[path_idx], np.int64),
        "path_end": _t(path_end, np.int64),
        "path_closure": _t(np.asarray(cache["path_closure"])[path_idx], np.int64),
        "path_graph": _t(path_graph, np.int64),
        "global": _t(np.asarray(cache["global"])[rows], np.float32),
        "atom_counts": _t(atom_counts, np.int64),
        "bond_counts": _t(bond_counts, np.int64),
    }


def compare_batches(
    a: Mapping[str, Any], b: Mapping[str, Any]
) -> dict[str, Any]:
    """Field-by-field equality of two collated batches (value + dtype + shape)."""
    keys_a, keys_b = set(a), set(b)
    missing = sorted(keys_a - keys_b)
    extra = sorted(keys_b - keys_a)
    per_field: dict[str, Any] = {}
    ok = not missing and not extra
    for key in sorted(keys_a & keys_b):
        va, vb = a[key], b[key]
        if isinstance(va, torch.Tensor) or isinstance(vb, torch.Tensor):
            if not (isinstance(va, torch.Tensor) and isinstance(vb, torch.Tensor)):
                per_field[key] = {"equal": False, "reason": "tensor vs non-tensor"}
                ok = False
                continue
            same_shape = tuple(va.shape) == tuple(vb.shape)
            same_dtype = va.dtype == vb.dtype
            same_value = bool(same_shape and same_dtype and torch.equal(va, vb))
            per_field[key] = {
                "equal": bool(same_shape and same_dtype and same_value),
                "shape": list(va.shape),
                "shape_b": list(vb.shape),
                "dtype": str(va.dtype),
                "dtype_b": str(vb.dtype),
                "same_value": same_value,
            }
            ok = ok and per_field[key]["equal"]
        else:
            same = va == vb
            per_field[key] = {"equal": bool(same), "a": va, "b": vb}
            ok = ok and bool(same)
    return {"all_equal": bool(ok), "missing": missing, "extra": extra, "fields": per_field}


def collate_for(cache: Mapping[str, Any], rows: Sequence[int], *, implementation: str) -> dict[str, Any]:
    """Dispatch between the frozen and the vectorised collation."""
    if implementation in ("frozen", "pq", "original", "baseline"):
        return C.PQ_DATA.collate(cache, rows)
    if implementation in ("fast", "v2", "optimized", "optimised", "E1"):
        return collate_v2(cache, rows)
    raise KeyError(implementation)


__all__ = ["collate_for", "collate_v2", "compare_batches"]

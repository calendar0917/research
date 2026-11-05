"""Batch construction: frozen reference, timer-instrumented mirror, fast path.

Three implementations of the same function live here:

``reference``
    ``lcrc_data.collate`` (the frozen preparation round), imported unchanged.
    This is what ``pc_train.py`` calls today and what defines "the original".

``instrumented``
    A line-for-line mirror of ``reference`` with ``time.perf_counter``
    checkpoints inserted, used only to attribute the CPU cost of one batch to
    the individual construction steps (pointer reads, per-molecule
    ``np.arange`` loops, fancy-index gathers from the memory map, tensor wrap).
    ``tests/test_pf_collate.py`` asserts its output is field-for-field equal to
    ``reference`` for every measured batch.

``fast``
    The optimised path (``collate_fast``).  It must be numerically identical to
    ``reference``: same keys, same dtypes, same shapes, same values, same order.
"""

from __future__ import annotations

import time
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import lcrc_data as FROZEN

#: names, in the frozen constructor order, of every collated field
FIELD_ORDER = (
    "atom_feat",
    "atom_type",
    "atom_scalar",
    "atom_batch",
    "atom_off",
    "bond_feat",
    "bond_type",
    "bond_index",
    "bond_batch",
    "nbr_atom",
    "nbr_code",
    "path_atom",
    "path_cat",
    "path_end63",
    "path_end",
    "path_closure",
    "path_graph",
    "global",
    "atom_counts",
    "bond_counts",
)


# ---------------------------------------------------------------------------
# reference
# ---------------------------------------------------------------------------


def collate_reference(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    return FROZEN.collate(cache, rows)


def ranges_reference(ptr: np.ndarray, rows: np.ndarray):
    return FROZEN._ranges(ptr, rows)


# ---------------------------------------------------------------------------
# instrumented mirror (timing only, no behaviour change)
# ---------------------------------------------------------------------------


class _Marks:
    def __init__(self) -> None:
        self.t0 = time.perf_counter()
        self.rows: list[tuple[str, float]] = []

    def mark(self, name: str) -> None:
        self.rows.append((name, time.perf_counter() - self.t0))


def collate_instrumented(
    cache: Mapping[str, Any], rows: Sequence[int], marks: _Marks | None = None
) -> tuple[dict[str, Any], _Marks]:
    """``lcrc_data.collate`` with checkpoints.  Output must be identical."""
    m = marks or _Marks()
    m.t0 = time.perf_counter()
    rows = np.asarray(rows, dtype=np.int64)
    if rows.ndim != 1:
        raise ValueError("rows must be 1-D")
    n_molecules = int(cache["n_molecules"])
    if np.any(rows < 0) or np.any(rows >= n_molecules):
        raise ValueError("row index outside cache")
    if len(rows) != len(set(int(x) for x in rows.tolist())):
        raise ValueError("duplicate row in batch")
    B = int(len(rows))
    atom_ptr = np.asarray(cache["atom_ptr"])
    bond_ptr = np.asarray(cache["bond_ptr"])
    m.mark("ptr_read")

    atom_idx, atom_batch, atom_local, atom_counts = _ranges_timed(atom_ptr, rows, m, "atom")
    bond_idx, bond_batch, _bond_local, bond_counts = _ranges_timed(bond_ptr, rows, m, "bond")
    atom_off = np.concatenate([[0], np.cumsum(atom_counts)[:-1]]).astype(np.int64) if B else np.zeros(0, np.int64)

    bi = np.asarray(cache["bond_index"])[bond_idx].astype(np.int64) if len(bond_idx) else np.zeros((0, 2), np.int64)
    if len(bi):
        bi = bi + atom_off[bond_batch].reshape(-1, 1)
    m.mark("bond_index_gather")

    nbr_ptr = np.asarray(cache["nbr_ptr"])
    nbr_counts = nbr_ptr[atom_idx + 1] - nbr_ptr[atom_idx] if len(atom_idx) else np.zeros(0, np.int64)
    atom_local_idx = np.arange(len(atom_idx), dtype=np.int64)
    if len(atom_idx):
        nbr_idx = np.concatenate(
            [np.arange(nbr_ptr[a], nbr_ptr[a] + c, dtype=np.int64) for a, c in zip(atom_idx, nbr_counts)]
        )
    else:
        nbr_idx = np.zeros(0, dtype=np.int64)
    m.mark("nbr_index_build")
    nbr_atom = np.repeat(atom_local_idx, nbr_counts) if len(atom_idx) else np.zeros(0, np.int64)
    nbr_code = _t(np.asarray(cache["nbr_code"])[nbr_idx], np.int64)
    m.mark("nbr_gather")

    path_ptr = np.asarray(cache["path_ptr"])
    path_counts = path_ptr[atom_idx + 1] - path_ptr[atom_idx] if len(atom_idx) else np.zeros(0, np.int64)
    if len(atom_idx):
        path_idx = np.concatenate(
            [np.arange(path_ptr[a], path_ptr[a] + c, dtype=np.int64) for a, c in zip(atom_idx, path_counts)]
        )
    else:
        path_idx = np.zeros(0, dtype=np.int64)
    m.mark("path_index_build")
    path_graph = np.repeat(atom_batch, path_counts) if len(atom_idx) else np.zeros(0, np.int64)
    path_atom = np.repeat(atom_local_idx, path_counts) if len(atom_idx) else np.zeros(0, np.int64)
    if len(path_idx):
        path_end = np.asarray(cache["path_end"])[path_idx].astype(np.int64) + atom_off[path_graph]
    else:
        path_end = np.zeros(0, np.int64)
    m.mark("path_index_gather")

    out = {
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
    m.mark("field_gather_and_wrap")
    return out, m


def _ranges_timed(ptr: np.ndarray, rows: np.ndarray, m: _Marks, prefix: str):
    rows = np.asarray(rows, dtype=np.int64)
    starts = ptr[rows].astype(np.int64)
    ends = ptr[rows + 1].astype(np.int64)
    counts = ends - starts
    if len(rows) == 0:
        return (
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
            np.zeros(0, dtype=np.int64),
        )
    idx = np.concatenate([np.arange(s, e, dtype=np.int64) for s, e in zip(starts, ends)])
    m.mark(f"{prefix}_idx_concat")
    batch = np.repeat(np.arange(len(rows), dtype=np.int64), counts)
    local = np.concatenate([np.arange(c, dtype=np.int64) for c in counts])
    m.mark(f"{prefix}_local_concat")
    return idx, batch, local, counts


def _t(arr, dtype):
    return torch.from_numpy(np.asarray(arr, dtype=dtype))


__all__ = [
    "FIELD_ORDER",
    "collate_instrumented",
    "collate_reference",
    "ranges_reference",
]

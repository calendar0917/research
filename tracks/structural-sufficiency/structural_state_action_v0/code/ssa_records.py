"""Exact, numbering-independent root records of the frozen W3 local encoding.

What the local encoder actually reads (audited against
``lcrc_tu_quick_screen_v1/code/tu_model.py::TULocalContent`` for the ``C1``
variant that W3 uses):

``x_v = cat[ a_v,                      sum_{e incident to v} b_e,
             sum_{n in S1(v)} nbr_emb(nbr_code_n),
             sum_{p in S2(v)} path_emb(path_cat_p),
             sum_{p in S2(v)} end_emb(path_end63_p),
             sum_{p in S2(v)} joint_proj(path_u(cat_p) * path_v(end63_p)),
             sum_{p in S2(v)} close_emb(path_closure_p),
             sum_{p in S2(v)} a_{path_end_p} * gate(cat_p),
             scal_v ]``

with ``a_v = atom_encoder(atom_feat_v)``, ``b_e = bond_encoder([a_u+a_v, |a_u-a_v|,
a_u*a_v, bond_feat_e])`` and ``scal_v = (atom_scalar_v - mean)/scale``.  ``C1``
has no ``b_mean`` branch.  The one-shot ``local`` call is the *only* consumer of
``bond_*`` / ``nbr_*`` / ``path_*`` (proved by the call-count audit), and
everything after it reads ``h``, ``atom_batch``, the molecule count and
``global`` only.

Therefore a root record ``R_v`` that contains

* the root's raw ``atom_feat``, ``atom_scalar`` and ``atom_type``,
* the multiset of incident bonds (``bond_feat`` + the *other* endpoint's raw
  ``atom_feat``),
* the multiset of S1 neighbour codes,
* the multiset of S2 paths (``path_cat``, ``path_end63``, ``path_closure`` and
  the *end* atom's raw ``atom_feat``),

determines ``x_v`` exactly.  Unordered occurrences are kept as a **sorted
multiset with multiplicities**, never a set.  Records are serialised to fixed
width little-endian bytes and identified by a BLAKE2b-128 digest; two records
are equal iff their digests are equal, never by approximate embedding
similarity.
"""

from __future__ import annotations

import hashlib
from typing import Any, Mapping

import numpy as np

_DIGEST_SIZE = 16
_RECORD_VERSION = b"R\x01"
_INT64 = np.dtype("<i8")
_INT32 = np.dtype("<i4")
_UINT32 = np.dtype("<u4")


def _u32(n: int) -> bytes:
    return np.asarray([n], dtype=_UINT32).tobytes()


def _digest(data: bytes) -> tuple[int, int]:
    raw = hashlib.blake2b(data, digest_size=_DIGEST_SIZE).digest()
    return int.from_bytes(raw[:8], "big"), int.from_bytes(raw[8:], "big")


def extract_batch_records(
    batch: Mapping[str, Any],
    *,
    n_graphs: int | None = None,
) -> dict[str, Any]:
    """Root records of every atom of a collated batch, as 128-bit digests.

    The batch is the exact tensor dictionary the frozen model consumes
    (``ew_collate.collate_v2`` / ``pq_data.collate`` are field-identical), so no
    index bookkeeping is re-derived from the cache.
    """
    import torch

    def _np(key: str, dtype=None) -> np.ndarray:
        value = batch[key]
        arr = value.detach().cpu().numpy() if torch.is_tensor(value) else np.asarray(value)
        return arr if dtype is None else arr.astype(dtype, copy=False)

    G = int(batch["n_graphs"] if n_graphs is None else n_graphs)
    atom_feat = _np("atom_feat")                     # (N, 51) float32
    atom_scalar = _np("atom_scalar")                 # (N, 9)
    atom_type = _np("atom_type")                     # (N,)
    atom_off = _np("atom_off", np.int64)             # (G,)
    atom_batch = _np("atom_batch", np.int64)
    bond_index = _np("bond_index", np.int64)         # (M, 2) global atom ids
    bond_feat = _np("bond_feat")
    bond_batch = _np("bond_batch", np.int64)
    nbr_atom = _np("nbr_atom", np.int64)             # global atom ids, sorted
    nbr_code = _np("nbr_code", np.int64)
    path_atom = _np("path_atom", np.int64)           # global atom ids, sorted
    path_cat = _np("path_cat", np.int64)
    path_end63 = _np("path_end63", np.int64)
    path_closure = _np("path_closure", np.int64)
    path_end = _np("path_end", np.int64)             # global atom ids

    feat_u8 = np.ascontiguousarray(atom_feat).view(np.uint8).reshape(atom_feat.shape[0], -1)
    scal_u8 = np.ascontiguousarray(atom_scalar).view(np.uint8).reshape(atom_scalar.shape[0], -1)
    bf_u8 = np.ascontiguousarray(bond_feat).view(np.uint8).reshape(bond_feat.shape[0], -1)
    atype_i8 = np.ascontiguousarray(atom_type).astype(_INT64)

    n_atoms = int(atom_feat.shape[0])
    hi = np.zeros(n_atoms, np.uint64)
    lo = np.zeros(n_atoms, np.uint64)
    sizes = np.zeros(n_atoms, np.int32)

    # bond segments per molecule (bond_batch is non-decreasing)
    bond_start = np.searchsorted(bond_batch, np.arange(G), side="left")
    bond_stop = np.searchsorted(bond_batch, np.arange(G), side="right")
    # neighbour / path segments per molecule (both atom index arrays are sorted)
    atom_stop = np.r_[atom_off[1:], n_atoms].astype(np.int64)
    nbr_start = np.searchsorted(nbr_atom, atom_off, side="left")
    nbr_stop = np.searchsorted(nbr_atom, atom_stop, side="left")
    path_start = np.searchsorted(path_atom, atom_off, side="left")
    path_stop = np.searchsorted(path_atom, atom_stop, side="left")

    for g in range(G):
        a0 = int(atom_off[g])
        a1 = int(atom_stop[g])
        n = a1 - a0
        bs, be = int(bond_start[g]), int(bond_stop[g])
        entries: list[list[tuple[bytes, bytes]]] = [[] for _ in range(n)]
        for t in range(bs, be):
            u = int(bond_index[t, 0]) - a0
            v = int(bond_index[t, 1]) - a0
            if u < 0 or v < 0 or u >= n or v >= n:
                raise RuntimeError("bond endpoint outside its molecule")
            bf = bf_u8[t].tobytes()
            entries[u].append((bf, feat_u8[a0 + v].tobytes()))
            entries[v].append((bf, feat_u8[a0 + u].tobytes()))

        ns, ne = int(nbr_start[g]), int(nbr_stop[g])
        nbr_by_atom: list[list[int]] = [[] for _ in range(n)]
        for t in range(ns, ne):
            nbr_by_atom[int(nbr_atom[t]) - a0].append(int(nbr_code[t]))

        ps, pe = int(path_start[g]), int(path_stop[g])
        path_by_atom: list[list[tuple[int, int, int, bytes]]] = [[] for _ in range(n)]
        for t in range(ps, pe):
            v = int(path_atom[t]) - a0
            e = int(path_end[t]) - a0
            if v < 0 or v >= n or e < 0 or e >= n:
                raise RuntimeError("path endpoint outside its molecule")
            path_by_atom[v].append(
                (int(path_cat[t]), int(path_end63[t]), int(path_closure[t]),
                 feat_u8[a0 + e].tobytes())
            )

        for v in range(n):
            parts = [
                _RECORD_VERSION,
                feat_u8[a0 + v].tobytes(),
                scal_u8[a0 + v].tobytes(),
                atype_i8[a0 + v].tobytes(),
            ]
            inc = sorted(entries[v])
            parts.append(_u32(len(inc)))
            parts.extend(bf + nf for bf, nf in inc)
            codes = sorted(nbr_by_atom[v])
            parts.append(_u32(len(codes)))
            parts.append(np.asarray(codes, dtype=_INT32).tobytes() if codes else b"")
            paths = sorted(path_by_atom[v])
            parts.append(_u32(len(paths)))
            for cat, end63, closure, endfeat in paths:
                parts.append(np.asarray([cat], dtype=_INT32).tobytes())
                parts.append(np.asarray([end63], dtype=_INT32).tobytes())
                parts.append(bytes([closure]))
                parts.append(endfeat)
            data = b"".join(parts)
            h, l = _digest(data)
            hi[a0 + v] = np.uint64(h)
            lo[a0 + v] = np.uint64(l)
            sizes[a0 + v] = len(data)

    return {
        "hi": hi,
        "lo": lo,
        "record_bytes": sizes,
        "n_atoms": int(n_atoms),
        "n_graphs": G,
        "atom_start": atom_off.astype(np.int64),
        "atom_stop": atom_stop.astype(np.int64),
        "atom_off": np.r_[atom_off, n_atoms].astype(np.int64),
    }


# ---------------------------------------------------------------------------
# multisets
# ---------------------------------------------------------------------------


def multiset_counts(hi: np.ndarray, lo: np.ndarray) -> dict[str, np.ndarray]:
    """Sorted unique digests with multiplicities (never a set)."""
    hi = np.asarray(hi, np.uint64)
    lo = np.asarray(lo, np.uint64)
    if hi.size == 0:
        return {"hi": hi, "lo": lo, "count": np.zeros(0, np.int64)}
    order = np.lexsort((lo, hi))
    h = hi[order]
    l = lo[order]
    new = np.r_[True, (h[1:] != h[:-1]) | (l[1:] != l[:-1])]
    first = np.flatnonzero(new)
    count = np.diff(np.r_[first, h.size])
    return {"hi": h[first], "lo": l[first], "count": count.astype(np.int64)}


def multiset_equal(a: Mapping[str, np.ndarray], b: Mapping[str, np.ndarray]) -> bool:
    if a["hi"].size != b["hi"].size:
        return False
    if not np.array_equal(a["hi"], b["hi"]) or not np.array_equal(a["lo"], b["lo"]):
        return False
    return bool(np.array_equal(a["count"], b["count"]))


def multiset_signature(m: Mapping[str, np.ndarray]) -> str:
    h = hashlib.blake2b(digest_size=16)
    h.update(np.ascontiguousarray(m["hi"], np.uint64).tobytes())
    h.update(np.ascontiguousarray(m["lo"], np.uint64).tobytes())
    h.update(np.ascontiguousarray(m["count"], np.int64).tobytes())
    return h.hexdigest()


def multiset_edit(m_new: Mapping[str, np.ndarray], m_old: Mapping[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Signed multiset ``m_new - m_old`` (zero counts dropped)."""
    ah, al, ac = m_new["hi"], m_new["lo"], m_new["count"].astype(np.int64)
    bh, bl, bc = m_old["hi"], m_old["lo"], m_old["count"].astype(np.int64)
    keys_h = np.concatenate([ah, bh])
    keys_l = np.concatenate([al, bl])
    weights = np.concatenate([ac, -bc])
    if keys_h.size == 0:
        return {"hi": np.zeros(0, np.uint64), "lo": np.zeros(0, np.uint64),
                "count": np.zeros(0, np.int64)}
    order = np.lexsort((keys_l, keys_h))
    h = keys_h[order]
    l = keys_l[order]
    w = weights[order]
    new = np.r_[True, (h[1:] != h[:-1]) | (l[1:] != l[:-1])]
    starts = np.flatnonzero(new)
    totals = np.add.reduceat(w, starts)
    keep = totals != 0
    return {"hi": h[starts][keep], "lo": l[starts][keep], "count": totals[keep].astype(np.int64)}


def edit_equal(e1: Mapping[str, np.ndarray], e2: Mapping[str, np.ndarray]) -> bool:
    if e1["hi"].size != e2["hi"].size:
        return False
    if not np.array_equal(e1["hi"], e2["hi"]) or not np.array_equal(e1["lo"], e2["lo"]):
        return False
    return bool(np.array_equal(e1["count"], e2["count"]))


def edit_signature(e: Mapping[str, np.ndarray]) -> str:
    return multiset_signature(e)


__all__ = [
    "edit_equal",
    "edit_signature",
    "extract_batch_records",
    "multiset_counts",
    "multiset_edit",
    "multiset_equal",
    "multiset_signature",
]

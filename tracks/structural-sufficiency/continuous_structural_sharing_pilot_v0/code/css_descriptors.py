"""Fixed sparse root descriptors for CSS-v0.

For every real root atom ``v`` the descriptor is built from the complete
radius-2 induced ego graph ``R2(v)``.  Five blocks are produced:

``ROOT``       one-hot of the root's 9 official OGB categorical fields
``S1``         joint (neighbour atomic number, bond type) counts
``S2``         joint (middle type, first bond, second bond, end type, closure)
               counts over real non-backtracking 2-paths v-w-u, u != v
``EGO_ATOM``   per root distance 0/1/2 category counts of every atom field
``EGO_BOND``   per unordered endpoint-distance pair category counts of every
               bond field over the real induced R2 bonds

Per block: ``sqrt(count)`` then L2 normalisation; empty blocks stay zero.  The
five blocks are concatenated and divided by ``sqrt(5)``.  Real multiplicities
are kept.  No atom/molecule identifier, label or task response is used and no
edge or path is invented.

Keys are packed as ``(block << 48) | local_key`` so one int64 array carries the
whole sparse row structure.  Keys are canonical within a root: they do not
depend on atom numbering or bond order.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import scipy.sparse as sp

import css_common as C
import gwic_data as GD

ATOM_DIMS = (119, 5, 12, 12, 10, 6, 6, 2, 2)
BOND_DIMS = (5, 6, 2)
ATOM_OFFSETS = np.concatenate([[0], np.cumsum(ATOM_DIMS)]).astype(np.int64)

BLOCK_NAMES = ("ROOT", "S1", "S2", "EGO_ATOM", "EGO_BOND")
N_BLOCKS = len(BLOCK_NAMES)
BLOCK_SHIFT = np.int64(48)
ROOT, S1, S2, EGO_ATOM, EGO_BOND = range(N_BLOCKS)

N_ATOM_TYPES = int(ATOM_DIMS[0])
N_BOND_TYPES = int(BOND_DIMS[0])
EGO_BOND_WIDTH = max(BOND_DIMS)


def _block_keys(block: int, local: np.ndarray) -> np.ndarray:
    return (np.int64(block) << BLOCK_SHIFT) | np.asarray(local, np.int64)


def _bfs(adj: Sequence[Sequence[int]], source: int, n: int) -> np.ndarray:
    dist = np.full(n, -1, np.int16)
    dist[source] = 0
    frontier = [int(source)]
    d = 0
    while frontier:
        d += 1
        nxt: list[int] = []
        for x in frontier:
            for y in adj[x]:
                if dist[y] < 0:
                    dist[y] = d
                    nxt.append(int(y))
        frontier = nxt
    return dist


def molecule_descriptors(atom_feat: np.ndarray, bond_feat: np.ndarray,
                         bond_u: np.ndarray, bond_v: np.ndarray
                         ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """All root descriptors of one molecule.

    Returns ``(keys, counts, root_ptr)`` where ``keys``/``counts`` are flat and
    ``root_ptr`` indexes them per root (molecule-local atom order).
    """
    atom_feat = np.asarray(atom_feat, np.int64)
    bond_feat = np.asarray(bond_feat, np.int64)
    bu = np.asarray(bond_u, np.int64)
    bv = np.asarray(bond_v, np.int64)
    n = int(atom_feat.shape[0])
    m = int(bu.shape[0])
    if atom_feat.shape != (n, C.ATOM_COLUMNS) or bond_feat.shape != (m, C.BOND_COLUMNS):
        raise ValueError("raw feature width mismatch")
    if m and not np.all(bu < bv):
        raise ValueError("bonds must be given with u < v")

    adj: list[list[int]] = [[] for _ in range(n)]
    bond_adj: list[list[int]] = [[] for _ in range(n)]
    for e in range(m):
        u, v = int(bu[e]), int(bv[e])
        adj[u].append(v)
        adj[v].append(u)
        bond_adj[u].append(e)
        bond_adj[v].append(e)

    atom_type = atom_feat[:, 0]
    bond_type = bond_feat[:, 0]
    root_atom_offsets = ATOM_OFFSETS[:C.ATOM_COLUMNS]
    bond_field_ids = np.arange(C.BOND_COLUMNS, dtype=np.int64)

    key_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    root_ptr = np.zeros(n + 1, np.int64)
    for v in range(n):
        dist = _bfs(adj, v, n)
        r2 = np.flatnonzero((dist >= 0) & (dist <= 2))
        in_r2 = np.zeros(n, bool)
        in_r2[r2] = True
        parts: list[np.ndarray] = []

        # ROOT: one-hot of all 9 official categorical fields
        parts.append(_block_keys(ROOT, root_atom_offsets + atom_type[v]))

        # S1: real incident bonds
        if bond_adj[v]:
            e1 = np.asarray(bond_adj[v], np.int64)
            w = np.where(bu[e1] == v, bv[e1], bu[e1])
            parts.append(_block_keys(S1, atom_type[w] * N_BOND_TYPES + bond_type[e1]))

        # S2: real non-backtracking rooted 2-paths v-w-u, u != v
        nbr_v = set(int(x) for x in adj[v])
        s2_codes: list[int] = []
        for e1 in bond_adj[v]:
            w = int(bv[e1]) if int(bu[e1]) == v else int(bu[e1])
            for e2 in bond_adj[w]:
                u = int(bv[e2]) if int(bu[e2]) == w else int(bu[e2])
                if u == v:
                    continue
                closure = 1 if u in nbr_v else 0
                code = (((int(atom_type[w]) * N_BOND_TYPES + int(bond_type[e1]))
                         * N_BOND_TYPES + int(bond_type[e2])) * N_ATOM_TYPES
                        + int(atom_type[u])) * 2 + closure
                s2_codes.append(code)
        if s2_codes:
            parts.append(_block_keys(S2, np.asarray(s2_codes, np.int64)))

        # EGO_ATOM: per root distance category counts of every atom field
        d = dist[r2].astype(np.int64)
        ego_atom = ((d[:, None] * C.ATOM_COLUMNS + np.arange(C.ATOM_COLUMNS)[None, :])
                    * N_ATOM_TYPES + atom_feat[r2])
        parts.append(_block_keys(EGO_ATOM, ego_atom.ravel()))

        # EGO_BOND: real induced R2 bonds, unordered endpoint-distance pair
        if m:
            induced = np.flatnonzero(in_r2[bu] & in_r2[bv])
            if induced.size:
                du = dist[bu[induced]].astype(np.int64)
                dv = dist[bv[induced]].astype(np.int64)
                dmin = np.minimum(du, dv)
                dmax = np.maximum(du, dv)
                ego_bond = (((dmin[:, None] * 3 + dmax[:, None]) * C.BOND_COLUMNS
                             + bond_field_ids[None, :]) * EGO_BOND_WIDTH
                            + bond_feat[induced])
                parts.append(_block_keys(EGO_BOND, ego_bond.ravel()))

        flat = np.concatenate(parts)
        keys, counts = np.unique(flat, return_counts=True)
        key_parts.append(keys)
        count_parts.append(counts.astype(np.int32))
        root_ptr[v + 1] = root_ptr[v] + keys.size

    keys = np.concatenate(key_parts) if key_parts else np.zeros(0, np.int64)
    counts = np.concatenate(count_parts) if count_parts else np.zeros(0, np.int32)
    return keys, counts, root_ptr


def build_all(raw: Mapping[str, Any], rows: Sequence[int]) -> dict[str, np.ndarray]:
    """Descriptors for every root of the given molecules, in row order."""
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    atom_feat = np.asarray(raw["atom_feat"])
    bond_feat = np.asarray(raw["bond_feat"])
    bond_u = np.asarray(raw["bond_u"], np.int64)
    bond_v = np.asarray(raw["bond_v"], np.int64)
    keys_parts: list[np.ndarray] = []
    count_parts: list[np.ndarray] = []
    root_ptr = [0]
    root_row: list[int] = []
    root_atom: list[int] = []
    for row in np.asarray(rows, np.int64).tolist():
        a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        n = a1 - a0
        # cache bond endpoints are molecule-local (0 .. n-1)
        if bond_u[b0:b1].size and (bond_u[b0:b1].min() < 0 or bond_v[b0:b1].max() >= n):
            raise ValueError(f"row {row}: local bond endpoint outside molecule")
        keys, counts, ptr = molecule_descriptors(atom_feat[a0:a1], bond_feat[b0:b1],
                                                 bond_u[b0:b1], bond_v[b0:b1])
        keys_parts.append(keys)
        count_parts.append(counts)
        root_ptr.extend((np.asarray(ptr[1:]) + root_ptr[-1]).tolist())
        root_row.extend([row] * n)
        root_atom.extend(range(n))
    return {
        "keys": np.concatenate(keys_parts) if keys_parts else np.zeros(0, np.int64),
        "counts": np.concatenate(count_parts) if count_parts else np.zeros(0, np.int32),
        "root_ptr": np.asarray(root_ptr, np.int64),
        "root_row": np.asarray(root_row, np.int32),
        "root_atom": np.asarray(root_atom, np.int32),
    }


def block_of(keys: np.ndarray) -> np.ndarray:
    return (np.asarray(keys, np.int64) >> BLOCK_SHIFT).astype(np.int64)


def oov_report(keys: np.ndarray, root_ptr: np.ndarray, root_row: np.ndarray,
               vocab: np.ndarray, groups: Mapping[str, np.ndarray]) -> dict[str, Any]:
    """Unseen-key fraction of the stored descriptors for row groups."""
    keys = np.asarray(keys, np.int64)
    vocab = np.asarray(vocab, np.int64)
    n_roots = int(root_row.size)
    idx = root_index(root_ptr, n_roots)
    block = block_of(keys)
    cols = np.searchsorted(vocab, keys)
    clipped = np.minimum(cols, max(0, vocab.size - 1))
    seen = ((cols < vocab.size) & (vocab[clipped] == keys)) if vocab.size else np.zeros(keys.size, bool)
    out: dict[str, Any] = {}
    for name, rows in groups.items():
        rows = np.asarray(rows, np.int64)
        entry = np.isin(root_row[idx], rows)
        n = int(entry.sum())
        n_oov = int((entry & ~seen).sum())
        out[name] = {
            "n_roots": int(np.isin(root_row, rows).sum()),
            "n_entries": n,
            "oov_entries": n_oov,
            "oov_fraction": float(n_oov / n) if n else 0.0,
            "oov_by_block": {bname: int((entry & ~seen & (block == b)).sum())
                             for b, bname in enumerate(BLOCK_NAMES)},
        }
    return out


def root_index(root_ptr: np.ndarray, n_roots: int) -> np.ndarray:
    return np.repeat(np.arange(int(n_roots), dtype=np.int64), np.diff(root_ptr))


def block_totals(keys: np.ndarray, counts: np.ndarray, root_ptr: np.ndarray,
                 n_roots: int) -> np.ndarray:
    """Per-root real occurrence count of each block (L2 norm before normalisation)."""
    idx = root_index(root_ptr, n_roots)
    block = block_of(keys)
    totals = np.zeros((int(n_roots), N_BLOCKS), np.float64)
    for b in range(N_BLOCKS):
        mask = block == b
        totals[:, b] = np.bincount(idx[mask], weights=counts[mask].astype(np.float64),
                                   minlength=int(n_roots))
    return totals


def build_vocabulary(keys: np.ndarray, root_ptr: np.ndarray, root_row: np.ndarray,
                     fit_rows: Sequence[int]) -> np.ndarray:
    rows = np.asarray(fit_rows, np.int64)
    root_mask = np.isin(root_row, rows)
    entry_mask = np.repeat(root_mask, np.diff(root_ptr))
    return np.unique(np.asarray(keys, np.int64)[entry_mask])


def normalised_values(keys: np.ndarray, counts: np.ndarray, root_ptr: np.ndarray,
                      n_roots: int, totals: np.ndarray | None = None) -> np.ndarray:
    """sqrt(count) / sqrt(block total) / sqrt(5); block-empty entries never occur."""
    if totals is None:
        totals = block_totals(keys, counts, root_ptr, n_roots)
    idx = root_index(root_ptr, n_roots)
    block = block_of(keys)
    denom = np.sqrt(totals[idx, block]) * np.sqrt(float(N_BLOCKS))
    if np.any(denom <= 0):
        raise ValueError("zero block norm with a non-zero count")
    return (np.sqrt(counts.astype(np.float64)) / denom).astype(np.float64)


def sparse_matrix(keys: np.ndarray, counts: np.ndarray, root_ptr: np.ndarray,
                  n_roots: int, vocab: np.ndarray
                  ) -> tuple[sp.csr_matrix, dict[str, Any]]:
    """Project the descriptors onto a frozen vocabulary, ignoring unseen keys."""
    keys = np.asarray(keys, np.int64)
    counts = np.asarray(counts, np.int64)
    vocab = np.asarray(vocab, np.int64)
    totals = block_totals(keys, counts, root_ptr, n_roots)
    values = normalised_values(keys, counts, root_ptr, n_roots, totals)
    idx = root_index(root_ptr, n_roots)
    block = block_of(keys)
    cols = np.searchsorted(vocab, keys)
    clipped = np.minimum(cols, max(0, vocab.size - 1))
    seen = (cols < vocab.size) & (vocab[clipped] == keys) if vocab.size else np.zeros(keys.size, bool)
    oov_by_block = {name: int((~seen & (block == b)).sum()) for b, name in enumerate(BLOCK_NAMES)}
    keep = np.flatnonzero(seen)
    matrix = sp.csr_matrix(
        (values[keep].astype(np.float32), (idx[keep], cols[keep])),
        shape=(int(n_roots), int(vocab.size)), dtype=np.float32)
    stats = {
        "n_roots": int(n_roots),
        "n_features": int(vocab.size),
        "nnz_in_vocab": int(keep.size),
        "nnz_total": int(keys.size),
        "oov_entries": int((~seen).sum()),
        "oov_fraction": float((~seen).mean()) if keys.size else 0.0,
        "oov_by_block": oov_by_block,
        "nnz_by_block": {name: int((block == b).sum()) for b, name in enumerate(BLOCK_NAMES)},
        "block_totals_sha256": C.array_sha256(totals),
    }
    return matrix, stats


def summary_stats(keys: np.ndarray, counts: np.ndarray, root_ptr: np.ndarray,
                  n_roots: int) -> dict[str, Any]:
    block = block_of(keys)
    totals = block_totals(keys, counts, root_ptr, n_roots)
    nnz_per_root = np.diff(root_ptr)
    return {
        "n_roots": int(n_roots),
        "n_entries": int(keys.size),
        "entries_per_root_mean": float(nnz_per_root.mean()),
        "entries_per_root_median": float(np.median(nnz_per_root)),
        "entries_per_root_max": int(nnz_per_root.max()),
        "counts_by_block": {name: int(counts[block == b].sum()) for b, name in enumerate(BLOCK_NAMES)},
        "block_totals": {
            name: {
                "root_total_mean": float(totals[:, b].mean()),
                "root_total_median": float(np.median(totals[:, b])),
                "empty_roots": int((totals[:, b] == 0).sum()),
            } for b, name in enumerate(BLOCK_NAMES)
        },
    }

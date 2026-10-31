"""Data views plus the label-free distance-bucket cache.

The frozen split, labels and LCRC structure cache are loaded through the
read-only v2 loader (``m2_data.build_views``), so all hash checks are identical
to the previous round.  This module adds one purely structural cache for the B
family:

    D34_i = {j : 3 <= dist(i,j) <= 4}
    D5_i  = {j : dist(i,j) >= 5 and dist is finite}

Self, distance 1/2 atoms and unreachable atoms are excluded.  The bucket cache
contains no labels and no learned quantity.  ``collate`` extends the frozen
LCRC collation with batch-local bucket index tensors.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import dc_common as C

import lcrc_data  # noqa: E402
import m2_data as MD  # noqa: E402


# ---------------------------------------------------------------------------
# structural distance buckets
# ---------------------------------------------------------------------------


def _bfs(adj: list[list[int]], source: int) -> np.ndarray:
    n = len(adj)
    dist = np.full(n, -1, dtype=np.int64)
    dist[source] = 0
    queue: deque[int] = deque([source])
    while queue:
        u = queue.popleft()
        du = int(dist[u]) + 1
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = du
                queue.append(int(v))
    return dist


def build_distance_buckets(cache: Mapping[str, Any]) -> dict[str, Any]:
    """Return a shallow copy of ``cache`` with D34/D5 flat index arrays.

    ``d34_idx`` / ``d5_idx`` contain cache-global target atom positions.  The
    per-atom pointers are cache-global and are remapped to batch-local indices
    by :func:`collate`.
    """
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    n_atoms = int(atom_ptr[-1])
    n_molecules = int(cache["n_molecules"])

    d34_ptr = np.zeros(n_atoms + 1, dtype=np.int64)
    d5_ptr = np.zeros(n_atoms + 1, dtype=np.int64)
    d34: list[int] = []
    d5: list[int] = []
    n34_total = 0
    n5_total = 0

    for g in range(n_molecules):
        a0, a1 = int(atom_ptr[g]), int(atom_ptr[g + 1])
        b0, b1 = int(bond_ptr[g]), int(bond_ptr[g + 1])
        n = a1 - a0
        adj: list[list[int]] = [[] for _ in range(n)]
        for u, v in bond_index[b0:b1].tolist():
            adj[int(u)].append(int(v))
            adj[int(v)].append(int(u))
        for i in range(n):
            dist = _bfs(adj, i)
            for j in range(n):
                d = int(dist[j])
                if d >= 3 and d <= 4:
                    d34.append(a0 + j)
                elif d >= 5:
                    d5.append(a0 + j)
            d34_ptr[a0 + i + 1] = len(d34)
            d5_ptr[a0 + i + 1] = len(d5)
        n34_total += int(d34_ptr[a1] - d34_ptr[a0])
        n5_total += int(d5_ptr[a1] - d5_ptr[a0])

    out = dict(cache)
    out["d34_ptr"] = d34_ptr
    out["d34_idx"] = np.asarray(d34, dtype=np.int64)
    out["d5_ptr"] = d5_ptr
    out["d5_idx"] = np.asarray(d5, dtype=np.int64)
    out["distance_bucket_meta"] = {
        "definition": "D34 = 3<=dist<=4; D5 = dist>=5 finite; self and dist<=2 excluded; unreachable excluded",
        "n_atoms": n_atoms,
        "n_d34_entries": int(len(d34)),
        "n_d5_entries": int(len(d5)),
        "n34_total": int(n34_total),
        "n5_total": int(n5_total),
        "labels_present": False,
    }
    return out


def _entry_ranges(ptr: np.ndarray, atom_idx: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    if atom_idx.size == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64)
    counts = ptr[atom_idx + 1] - ptr[atom_idx]
    parts = [np.arange(ptr[a], ptr[a] + c, dtype=np.int64) for a, c in zip(atom_idx, counts)]
    idx = np.concatenate(parts) if parts else np.zeros(0, dtype=np.int64)
    return idx, counts


def collate(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    """Frozen LCRC collation plus batch-local D34/D5 indices."""
    out = lcrc_data.collate(cache, rows)
    if "d34_ptr" not in cache:
        return out
    rows_np = np.asarray(rows, dtype=np.int64)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    atom_idx = np.concatenate(
        [np.arange(int(atom_ptr[r]), int(atom_ptr[r + 1]), dtype=np.int64) for r in rows_np]
    ) if rows_np.size else np.zeros(0, dtype=np.int64)
    atom_batch = out["atom_batch"]
    atom_off = out["atom_off"]

    for prefix in ("d34", "d5"):
        ptr = np.asarray(cache[f"{prefix}_ptr"], dtype=np.int64)
        idx = np.asarray(cache[f"{prefix}_idx"], dtype=np.int64)
        entry_idx, counts = _entry_ranges(ptr, atom_idx)
        targets = idx[entry_idx] if entry_idx.size else np.zeros(0, dtype=np.int64)
        entry_atom_local = np.repeat(np.arange(atom_idx.size, dtype=np.int64), counts)
        target_batch = torch.zeros(0, dtype=torch.long)
        if entry_atom_local.size:
            entry_batch = atom_batch[entry_atom_local]
            row_g = torch.as_tensor(rows_np, dtype=torch.long)[entry_batch]
            offset = atom_off[entry_batch] - torch.as_tensor(atom_ptr, dtype=torch.long)[row_g]
            target_batch = torch.as_tensor(targets, dtype=torch.long) + offset
        batch_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
        out[f"{prefix}_ptr"] = torch.as_tensor(batch_ptr, dtype=torch.long)
        out[f"{prefix}_idx"] = target_batch
        out[f"{prefix}_counts"] = torch.as_tensor(counts, dtype=torch.long)
    return out


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------


def load_views(*, include_test: bool = True, with_distance: bool = True) -> dict[str, Any]:
    views = MD.build_views(include_test=include_test)
    if with_distance:
        views["train_cache"] = build_distance_buckets(views["train_cache"])
        views["valid_cache"] = build_distance_buckets(views["valid_cache"])
        if views.get("test_cache") is not None:
            views["test_cache"] = build_distance_buckets(views["test_cache"])
    return views


def distance_bucket_coverage(cache: Mapping[str, Any]) -> dict[str, Any]:
    """Descriptive coverage of the two buckets over a cache."""
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    n_atoms = int(atom_ptr[-1])
    d34 = np.asarray(cache["d34_ptr"], dtype=np.int64)
    d5 = np.asarray(cache["d5_ptr"], dtype=np.int64)
    n34 = np.diff(d34)
    n5 = np.diff(d5)
    both_empty = (n34 == 0) & (n5 == 0)
    return {
        "n_atoms": n_atoms,
        "n_d34_nonempty_atoms": int(np.sum(n34 > 0)),
        "n_d5_nonempty_atoms": int(np.sum(n5 > 0)),
        "n_both_empty_atoms": int(np.sum(both_empty)),
        "fraction_d34_nonempty": float(np.mean(n34 > 0)) if n_atoms else 0.0,
        "fraction_d5_nonempty": float(np.mean(n5 > 0)) if n_atoms else 0.0,
        "mean_d34_bucket_size": float(n34.mean()) if n_atoms else 0.0,
        "mean_d5_bucket_size": float(n5.mean()) if n_atoms else 0.0,
        "max_d34_bucket_size": int(n34.max()) if n_atoms else 0,
        "max_d5_bucket_size": int(n5.max()) if n_atoms else 0,
        "n_d34_entries": int(np.asarray(cache["d34_idx"]).size),
        "n_d5_entries": int(np.asarray(cache["d5_idx"]).size),
    }


def permute_batch_atoms(batch: Mapping[str, Any], generator: torch.Generator) -> dict[str, Any]:
    """Atom renumbering that also remaps D34/D5 target indices."""
    import x1_models as X1M

    out = X1M.permute_batch_atoms(batch, generator)
    # Reconstruct the old->new map exactly as X1M does.
    atom_batch = batch["atom_batch"]
    n_atoms = int(atom_batch.shape[0])
    perm = torch.arange(n_atoms, dtype=torch.long)
    n_graphs = int(batch["n_graphs"])
    for g in range(n_graphs):
        idx = torch.nonzero(atom_batch == g, as_tuple=False).squeeze(1)
        local = torch.randperm(int(idx.numel()), generator=generator)
        perm[idx] = idx[local]
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(n_atoms, dtype=torch.long)
    for prefix in ("d34", "d5"):
        idx_key, ptr_key, cnt_key = f"{prefix}_idx", f"{prefix}_ptr", f"{prefix}_counts"
        if idx_key not in batch:
            continue
        ptr = batch[ptr_key]
        idx = batch[idx_key]
        counts = (ptr[1:] - ptr[:-1])
        new_counts = counts.index_select(0, perm)
        new_ptr = torch.cat([torch.zeros(1, dtype=ptr.dtype), torch.cumsum(new_counts, dim=0)])
        pieces = []
        for new_slot in range(n_atoms):
            old_root = int(perm[new_slot])
            lo, hi = int(ptr[old_root]), int(ptr[old_root + 1])
            pieces.append(inverse[idx[lo:hi]])
        out[idx_key] = torch.cat(pieces) if pieces else idx[:0]
        out[ptr_key] = new_ptr
        if cnt_key in batch:
            out[cnt_key] = new_counts
    return out


def permute_paths_within_roots(batch: Mapping[str, Any], generator: torch.Generator) -> dict[str, Any]:
    """Permute S2 path instances inside every root atom (path order only)."""
    path_atom = batch["path_atom"]
    n_paths = int(path_atom.shape[0])
    if n_paths == 0:
        return dict(batch)
    perm = torch.arange(n_paths, dtype=torch.long)
    # path_atom is grouped by root after collate; still derive groups robustly.
    for root in torch.unique(path_atom).tolist():
        idx = torch.nonzero(path_atom == int(root), as_tuple=False).squeeze(1)
        local = torch.randperm(int(idx.numel()), generator=generator)
        perm[idx] = idx[local]
    out = dict(batch)
    for key in ("path_cat", "path_end63", "path_end", "path_closure", "path_graph", "path_atom"):
        out[key] = batch[key].index_select(0, perm)
    return out


__all__ = [
    "build_distance_buckets",
    "collate",
    "load_views",
    "distance_bucket_coverage",
    "permute_batch_atoms",
    "permute_paths_within_roots",
]

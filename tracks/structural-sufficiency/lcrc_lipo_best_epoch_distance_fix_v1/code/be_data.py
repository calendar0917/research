"""Data views for the best-epoch / distance-fix round.

The frozen split, labels, LCRC structure cache and D34/D5 bucket cache are
imported unchanged from the read-only historical modules:

    dc_data.load_views            split + labels + cache + distance buckets
    dc_data.collate               batch-local D34/D5 target indices
    dc_data.permute_batch_atoms   atom renumbering that remaps bucket indices

What this module adds is (a) the *correct* bucket-mean aggregation and (b) an
independent, loop-based reference implementation used by the gates, so the
PyTorch path is checked against a hand-written computation instead of itself.

Definitions (per root atom ``i`` of a molecule):

    D34_i = {j : 3 <= dist(i, j) <= 4}
    D5_i  = {j : dist(i, j) >= 5, finite}

    c34_i = mean_{j in D34_i} h_j      (zero if D34_i is empty)
    c5_i  = mean_{j in D5_i}  h_j      (zero if D5_i  is empty)

``d34_idx``/``d5_idx`` are *target* atom indices; the aggregation output row is
the *root* atom index.  ``d34_counts``/``d5_counts`` are per-root bucket sizes,
so the root index of every flat entry is ``repeat_interleave(arange(N), counts)``.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Mapping, Sequence

import numpy as np
import torch

import be_common as C
import dc_data as DD  # read-only frozen bucket cache / collation


# ---------------------------------------------------------------------------
# views
# ---------------------------------------------------------------------------


def load_views(*, include_test: bool, with_distance: bool = True) -> dict[str, Any]:
    return DD.load_views(include_test=include_test, with_distance=with_distance)


def collate(cache: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    return DD.collate(cache, rows)


def build_distance_buckets(cache: Mapping[str, Any]) -> dict[str, Any]:
    return DD.build_distance_buckets(cache)


def distance_bucket_coverage(cache: Mapping[str, Any]) -> dict[str, Any]:
    return DD.distance_bucket_coverage(cache)


def permute_batch_atoms(batch: Mapping[str, Any], generator: torch.Generator) -> dict[str, Any]:
    """Renumber atoms inside every molecule, remapping D34/D5 target indices.

    The frozen ``dc_data.permute_batch_atoms`` draws the atom permutation twice
    (once inside ``x1_models.permute_batch_atoms`` and once when rebuilding the
    bucket index), so it silently uses two different permutations while the
    bucket contents stay aligned to neither.  This version draws one permutation
    and applies it to the atom tensors and to the bucket index structure alike.
    """
    atom_batch = batch["atom_batch"]
    n_atoms = int(atom_batch.numel())
    n_graphs = int(batch["n_graphs"])
    perm = torch.arange(n_atoms, dtype=torch.long)  # perm[new] = old
    for g in range(n_graphs):
        idx = torch.nonzero(atom_batch == g, as_tuple=False).squeeze(1)
        local = torch.randperm(int(idx.numel()), generator=generator)
        perm[idx] = idx[local]
    inverse = torch.empty_like(perm)
    inverse[perm] = torch.arange(n_atoms, dtype=torch.long)
    out = dict(batch)
    for key in ("atom_feat", "atom_type", "atom_scalar", "atom_batch"):
        out[key] = batch[key].index_select(0, perm)
    out["bond_index"] = inverse[batch["bond_index"]]
    if "nbr_atom" in batch and batch["nbr_atom"].numel():
        out["nbr_atom"] = inverse[batch["nbr_atom"]]
    if "path_atom" in batch and batch["path_atom"].numel():
        out["path_atom"] = inverse[batch["path_atom"]]
        out["path_end"] = inverse[batch["path_end"]]
    for prefix in ("d34", "d5"):
        idx_key, ptr_key, cnt_key = f"{prefix}_idx", f"{prefix}_ptr", f"{prefix}_counts"
        if idx_key not in batch:
            continue
        ptr = batch[ptr_key]
        idx = batch[idx_key]
        counts = ptr[1:] - ptr[:-1]
        new_counts = counts.index_select(0, perm)
        new_ptr = torch.cat([torch.zeros(1, dtype=ptr.dtype), torch.cumsum(new_counts, dim=0)])
        pieces = [
            inverse[idx[int(ptr[old_root]) : int(ptr[old_root + 1])]] for old_root in perm.tolist()
        ]
        out[idx_key] = torch.cat(pieces) if pieces else idx[:0]
        out[ptr_key] = new_ptr
        if cnt_key in batch:
            out[cnt_key] = new_counts
    return out


# ---------------------------------------------------------------------------
# the corrected aggregation
# ---------------------------------------------------------------------------


def bucket_roots(d34_counts: torch.Tensor, d5_counts: torch.Tensor, n_atoms: int):
    """Root index of every flat bucket entry (``repeat_interleave`` of arange)."""
    base = torch.arange(int(n_atoms), dtype=torch.long, device=d34_counts.device)
    root34 = torch.repeat_interleave(base, d34_counts.to(torch.long))
    root5 = torch.repeat_interleave(base, d5_counts.to(torch.long))
    return root34, root5


def bucket_means(
    h: torch.Tensor,
    d34_idx: torch.Tensor,
    d5_idx: torch.Tensor,
    d34_counts: torch.Tensor,
    d5_counts: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Correct per-root bucket means ``(c34, c5)``; empty bucket -> exact zero."""
    n_atoms = int(h.shape[0])
    width = int(h.shape[1])
    root34, root5 = bucket_roots(d34_counts, d5_counts, n_atoms)
    if int(root34.numel()) != int(d34_idx.numel()) or int(root5.numel()) != int(d5_idx.numel()):
        raise ValueError(
            "bucket index/count mismatch: "
            f"sum(counts)={int(root34.numel())}/{int(root5.numel())} "
            f"len(idx)={int(d34_idx.numel())}/{int(d5_idx.numel())}"
        )
    sum34 = h.new_zeros((n_atoms, width)).index_add_(0, root34, h.index_select(0, d34_idx.to(torch.long)))
    sum5 = h.new_zeros((n_atoms, width)).index_add_(0, root5, h.index_select(0, d5_idx.to(torch.long)))
    denom34 = d34_counts.to(h.dtype).clamp(min=1.0).unsqueeze(1)
    denom5 = d5_counts.to(h.dtype).clamp(min=1.0).unsqueeze(1)
    return sum34 / denom34, sum5 / denom5


# ---------------------------------------------------------------------------
# independent reference implementation (no PyTorch indexing tricks)
# ---------------------------------------------------------------------------


def _adjacency(batch: Mapping[str, Any]) -> list[list[int]]:
    n_atoms = int(batch["atom_batch"].numel())
    adj: list[list[int]] = [[] for _ in range(n_atoms)]
    for u, v in batch["bond_index"].detach().cpu().tolist():
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    return adj


def _bfs_distances(adj: list[list[int]], source: int) -> list[int]:
    dist = [-1] * len(adj)
    dist[source] = 0
    queue: deque[int] = deque([source])
    while queue:
        u = queue.popleft()
        for v in adj[u]:
            if dist[v] < 0:
                dist[v] = dist[u] + 1
                queue.append(v)
    return dist


def reference_buckets(batch: Mapping[str, Any]) -> dict[str, Any]:
    """Python-loop D34/D5 membership for a collated batch.

    Returns ``{"d34": [list_of_targets_per_root], "d5": [...], "d34_roots":
    tensor, "d5_roots": tensor}``.  Distance is computed *only inside a
    molecule* by restricting BFS to atoms of the same graph.
    """
    atom_batch = batch["atom_batch"].detach().cpu()
    n_atoms = int(atom_batch.numel())
    adjacency = _adjacency(batch)
    per_graph: dict[int, list[int]] = {}
    for atom in range(n_atoms):
        per_graph.setdefault(int(atom_batch[atom]), []).append(atom)
    d34: list[list[int]] = []
    d5: list[list[int]] = []
    for i in range(n_atoms):
        members = set(per_graph[int(atom_batch[i])])
        dist = _bfs_distances(adjacency, i)
        d34.append(sorted(j for j in members if 3 <= dist[j] <= 4))
        d5.append(sorted(j for j in members if dist[j] >= 5))
    roots34 = [i for i in range(n_atoms) for _ in d34[i]]
    roots5 = [i for i in range(n_atoms) for _ in d5[i]]
    return {
        "d34": d34,
        "d5": d5,
        "d34_roots": torch.as_tensor(roots34, dtype=torch.long),
        "d5_roots": torch.as_tensor(roots5, dtype=torch.long),
        "d34_targets": torch.as_tensor([j for bucket in d34 for j in bucket], dtype=torch.long),
        "d5_targets": torch.as_tensor([j for bucket in d5 for j in bucket], dtype=torch.long),
        "d34_counts": torch.as_tensor([len(b) for b in d34], dtype=torch.long),
        "d5_counts": torch.as_tensor([len(b) for b in d5], dtype=torch.long),
    }


def reference_bucket_means(batch: Mapping[str, Any], h: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Independent ``(c34, c5)`` for a collated batch and an ``h`` array."""
    ref = reference_buckets(batch)
    h = np.asarray(h, dtype=np.float64)
    c34 = np.zeros_like(h)
    c5 = np.zeros_like(h)
    for i, bucket in enumerate(ref["d34"]):
        if bucket:
            c34[i] = h[bucket].mean(axis=0)
    for i, bucket in enumerate(ref["d5"]):
        if bucket:
            c5[i] = h[bucket].mean(axis=0)
    return c34, c5


def independent_bucket_from_adjacency(adjacency: Sequence[Sequence[int]]) -> dict[str, Any]:
    """Bucket membership for one molecule given only its adjacency list."""
    n = len(adjacency)
    d34: list[list[int]] = []
    d5: list[list[int]] = []
    for i in range(n):
        dist = _bfs_distances([list(map(int, row)) for row in adjacency], i)
        d34.append(sorted(j for j in range(n) if 3 <= dist[j] <= 4))
        d5.append(sorted(j for j in range(n) if dist[j] >= 5))
    return {"d34": d34, "d5": d5}


__all__ = [
    "load_views",
    "collate",
    "build_distance_buckets",
    "distance_bucket_coverage",
    "permute_batch_atoms",
    "bucket_roots",
    "bucket_means",
    "reference_buckets",
    "reference_bucket_means",
    "independent_bucket_from_adjacency",
]

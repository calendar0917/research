"""ZINC structural-token attention v1 — token batch assembly and relation tables.

The frozen ``MolObjects`` already contain the atom graph, physical bonds and
junction objects.  This module adds a label-blind, deterministic token view:

* token types A / B / J in that order, with supports
  ``A={i}``, ``B={u,v}``, ``J={i,j,k}``;
* ``dist_bin`` = min atom-graph shortest path between the two support sets,
  bins ``{0,1,2,3,>=4,disconnected}``;
* ``overlap_bin`` = support-set intersection size, bins ``{0,1,2,3}``.

No new molecule, label, feature or pretrained model is introduced.  These are
deterministic functions of the existing frozen object indices.
"""

from __future__ import annotations

import hashlib
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

_CODE_DIR = Path(__file__).resolve().parent
REPO = _CODE_DIR.parents[3]
for _p in (_CODE_DIR, REPO / "tracks/structure-contribution/code"):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import structural_relation_objects_v1 as sro  # noqa: E402

N_DIST_BINS = 6
N_OVERLAP_BINS = 4
DIST_DISCONNECTED = 5
DIST_GE4 = 4
INF = np.int32(1 << 20)


# ---------------------------------------------------------------------------
# relation tables
# ---------------------------------------------------------------------------


def atom_distance_matrix(obj: sro.MolObjects) -> np.ndarray:
    """Unweighted all-pairs shortest paths on the original undirected atom graph."""
    n = int(obj.n_atoms)
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in np.asarray(obj.bond_index, dtype=np.int64):
        u, v = int(u), int(v)
        adj[u].append(v)
        adj[v].append(u)
    dist = np.full((n, n), INF, dtype=np.int32)
    for s in range(n):
        dist[s, s] = 0
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt: list[int] = []
            for u in frontier:
                for v in adj[u]:
                    if dist[s, v] == INF:
                        dist[s, v] = d
                        nxt.append(v)
            frontier = nxt
    return dist


def token_supports(obj: sro.MolObjects) -> np.ndarray:
    """Supports padded to 3 columns with -1; rows are A, B, J in token order."""
    n_a = int(obj.atom_feat.shape[0])
    n_b = int(obj.bond_index.shape[0])
    n_j = int(obj.junc_index.shape[0])
    out = np.full((n_a + n_b + n_j, 3), -1, dtype=np.int64)
    for i in range(n_a):
        out[i, 0] = i
    for e in range(n_b):
        out[n_a + e, :2] = np.asarray(obj.bond_index[e], dtype=np.int64)
    for j in range(n_j):
        out[n_a + n_b + j, :3] = np.asarray(obj.junc_index[j], dtype=np.int64)
    return out


def relation_tables(obj: sro.MolObjects) -> dict[str, np.ndarray]:
    """Return type ids, distance bins and overlap bins for one molecule."""
    n_a = int(obj.atom_feat.shape[0])
    n_b = int(obj.bond_index.shape[0])
    n_j = int(obj.junc_index.shape[0])
    n = n_a + n_b + n_j
    supports = token_supports(obj)
    d_atom = atom_distance_matrix(obj)

    min_d = np.full((n, n), INF, dtype=np.int32)
    overlap = np.zeros((n, n), dtype=np.uint8)
    for a in range(3):
        va = supports[:, a]
        valid_a = va >= 0
        for b in range(3):
            vb = supports[:, b]
            valid_b = vb >= 0
            both = valid_a[:, None] & valid_b[None, :]
            if not bool(both.any()):
                continue
            va0 = np.where(valid_a, va, 0)
            vb0 = np.where(valid_b, vb, 0)
            d = d_atom[np.ix_(va0, vb0)]
            min_d = np.where(both & (d < min_d), d, min_d)
            eq = (va0[:, None] == vb0[None, :]) & both
            overlap += eq.astype(np.uint8)

    dist_bin = np.full((n, n), DIST_DISCONNECTED, dtype=np.uint8)
    finite = min_d < INF
    dist_bin[finite & (min_d == 0)] = 0
    dist_bin[finite & (min_d == 1)] = 1
    dist_bin[finite & (min_d == 2)] = 2
    dist_bin[finite & (min_d == 3)] = 3
    dist_bin[finite & (min_d >= 4)] = DIST_GE4

    overlap_bin = np.minimum(overlap, N_OVERLAP_BINS - 1).astype(np.uint8)
    type_ids = np.concatenate(
        [
            np.zeros(n_a, dtype=np.uint8),
            np.ones(n_b, dtype=np.uint8),
            np.full(n_j, 2, dtype=np.uint8),
        ]
    )
    return {
        "type_ids": type_ids,
        "dist_bin": dist_bin,
        "overlap_bin": overlap_bin,
        "n_tokens": np.asarray([n], dtype=np.int64),
        "n_a": np.asarray([n_a], dtype=np.int64),
        "n_b": np.asarray([n_b], dtype=np.int64),
        "n_j": np.asarray([n_j], dtype=np.int64),
    }


def build_relation_cache(objects: Sequence[sro.MolObjects]) -> list[dict[str, np.ndarray]]:
    cache: list[dict[str, np.ndarray]] = []
    for obj in objects:
        cache.append(relation_tables(obj))
    return cache


def relation_cache_sha256(cache: Sequence[Mapping[str, np.ndarray]]) -> str:
    h = hashlib.sha256()
    for rec in cache:
        for key in ("type_ids", "dist_bin", "overlap_bin"):
            h.update(np.ascontiguousarray(rec[key], dtype=np.uint8).tobytes())
    return h.hexdigest()


def relation_bin_counts(
    cache: Sequence[Mapping[str, np.ndarray]], rows: Sequence[int] | None = None
) -> dict[str, Any]:
    """Descriptive counts over all token pairs (label-free)."""
    if rows is None:
        rows = range(len(cache))
    dist = np.zeros(N_DIST_BINS, dtype=np.int64)
    over = np.zeros(N_OVERLAP_BINS, dtype=np.int64)
    pairs = 0
    for r in rows:
        rec = cache[int(r)]
        d = rec["dist_bin"]
        o = rec["overlap_bin"]
        dist += np.bincount(d.reshape(-1), minlength=N_DIST_BINS)[:N_DIST_BINS]
        over += np.bincount(o.reshape(-1), minlength=N_OVERLAP_BINS)[:N_OVERLAP_BINS]
        pairs += int(d.size)
    return {
        "pairs": pairs,
        "dist_bin_counts": [int(x) for x in dist],
        "overlap_bin_counts": [int(x) for x in over],
    }


# ---------------------------------------------------------------------------
# token batch assembly
# ---------------------------------------------------------------------------


def _type_slots(
    token_batch: torch.Tensor,
    token_type: torch.Tensor,
    n_graphs: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-molecule slot in the A|B|J order, plus per-type counts."""
    counts = torch.zeros((n_graphs, 3), dtype=torch.long)
    for t in range(3):
        idx = token_type == t
        if bool(idx.any()):
            counts[:, t] = torch.bincount(token_batch[idx], minlength=n_graphs)
    # Offset of each type's block inside one molecule: A=0, B=n_A, J=n_A+n_B.
    type_offsets = torch.cumsum(counts, dim=1) - counts
    slot = torch.zeros_like(token_batch)
    for t in range(3):
        idx = token_type == t
        if not bool(idx.any()):
            continue
        tb = token_batch[idx]
        ct = counts[:, t]
        off = torch.cumsum(ct, dim=0) - ct
        slot[idx] = torch.arange(int(idx.sum()), dtype=torch.long) - off[tb]
    slot = slot + type_offsets[token_batch, token_type]
    return slot, counts, type_offsets


def batch_objects_sta(
    objects: Sequence[sro.MolObjects],
    rows: Sequence[int],
    cache: Sequence[Mapping[str, np.ndarray]] | None = None,
) -> dict[str, Any]:
    """Concatenate frozen objects and add dense padded token/relation tensors."""
    selected = [objects[int(i)] for i in rows]
    base = sro.batch_objects(selected)
    n_graphs = len(selected)
    n_a = int(base["atom_feat"].shape[0])
    n_b = int(base["bond_feat"].shape[0])
    n_j = int(base["junc_feat"].shape[0])
    token_type = torch.cat(
        [
            torch.zeros(n_a, dtype=torch.long),
            torch.ones(n_b, dtype=torch.long),
            torch.full((n_j,), 2, dtype=torch.long),
        ]
    )
    token_batch = torch.cat([base["atom_batch"], base["bond_batch"], base["junc_batch"]], dim=0)
    token_slot, token_counts, _ = _type_slots(token_batch, token_type, n_graphs)
    tmax = int(token_counts.sum(dim=1).max().item()) if n_graphs else 0

    token_valid = torch.zeros((n_graphs, tmax), dtype=torch.bool)
    token_type_padded = torch.zeros((n_graphs, tmax), dtype=torch.long)
    if token_batch.numel() > 0:
        token_valid[token_batch, token_slot] = True
        token_type_padded[token_batch, token_slot] = token_type

    rel_dist = torch.full((n_graphs, tmax, tmax), DIST_DISCONNECTED, dtype=torch.uint8)
    rel_overlap = torch.zeros((n_graphs, tmax, tmax), dtype=torch.uint8)
    if cache is not None:
        for g, row in enumerate(rows):
            rec = cache[int(row)]
            n = int(rec["n_tokens"][0])
            if n != int(token_counts[g].sum()):
                raise RuntimeError(f"relation cache size mismatch at row {int(row)}")
            rel_dist[g, :n, :n] = torch.from_numpy(np.ascontiguousarray(rec["dist_bin"]))
            rel_overlap[g, :n, :n] = torch.from_numpy(np.ascontiguousarray(rec["overlap_bin"]))

    out = dict(base)
    out.update(
        {
            "token_type": token_type,
            "token_batch": token_batch,
            "token_slot": token_slot,
            "token_counts": token_counts,
            "token_valid": token_valid,
            "token_type_padded": token_type_padded,
            "rel_dist": rel_dist,
            "rel_overlap": rel_overlap,
            "n_tokens": int(token_batch.numel()),
            "tmax": int(tmax),
        }
    )
    return out


def move_batch(batch: Mapping[str, Any], device: Any) -> dict[str, Any]:
    return {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in batch.items()}

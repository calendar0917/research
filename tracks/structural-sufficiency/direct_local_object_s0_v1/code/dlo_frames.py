"""Exact rooted enumeration of the *complete* local object.

An object is ``(molecule, root, atom_set)`` and its frame is the full induced
subgraph on ``atom_set`` with the root pinned at slot 0.  We never pick one
canonical permutation: every permutation reaching the lexicographically
smallest adjacency bit pattern is kept, because the remaining permutations move
*attributes* between slots and are therefore different inputs to the object
function.

The canonicalisation is a label-free topology table (a small LUT over adjacency
bit patterns), not a per-object graph-isomorphism call.
"""

from __future__ import annotations

import itertools
import math
import time
from typing import Any, Iterable, Sequence

import numpy as np

import dlo_common as C

GLOBAL_PAIR_INDEX = {p: i for i, p in enumerate(C.EDGE_PAIRS)}


def _pairs(k: int) -> list[tuple[int, int]]:
    return list(itertools.combinations(range(k), 2))


def _permuted_bits(masks: np.ndarray, perm: Sequence[int], pairs: Sequence[tuple[int, int]],
                   pair_index: dict[tuple[int, int], int]) -> np.ndarray:
    out = np.zeros(masks.shape, dtype=np.uint32)
    for e_slot, (i, j) in enumerate(pairs):
        a, b = sorted((int(perm[i]), int(perm[j])))
        e_orig = pair_index[(a, b)]
        out |= ((masks >> e_orig) & 1) << e_slot
    return out


def build_lut(k: int) -> dict[str, np.ndarray]:
    """Canonical adjacency bit + all winning root-fixed permutations for ``k``."""
    pairs = _pairs(k)
    n_pairs = len(pairs)
    n_masks = 1 << n_pairs
    masks = np.arange(n_masks, dtype=np.uint32)
    perms = [(0,) + p for p in itertools.permutations(range(1, k))]
    pair_index = {p: i for i, p in enumerate(pairs)}
    cand = np.empty((n_masks, len(perms)), dtype=np.uint16)
    for c, perm in enumerate(perms):
        cand[:, c] = _permuted_bits(masks, perm, pairs, pair_index).astype(np.uint16)
    canon = cand.min(axis=1).astype(np.uint16)
    hit = cand == canon[:, None]
    rows, cols = np.nonzero(hit)
    if rows.size and not np.all(np.diff(rows) >= 0):
        raise AssertionError("winner rows are not ordered")
    ptr = np.searchsorted(rows, np.arange(n_masks + 1, dtype=np.int64)).astype(np.int64)
    perm_table = np.asarray(perms, dtype=np.uint8)
    winner_perm = perm_table[cols] if cols.size else np.zeros((0, k), np.uint8)
    return {
        "k": np.asarray(k, np.uint8),
        "n_pairs": np.asarray(n_pairs, np.uint8),
        "canon": canon,
        "count": np.diff(ptr).astype(np.int32),
        "ptr": ptr,
        "winner_perm": winner_perm,
    }


def build_luts(max_k: int = C.MAX_ATOMS) -> tuple[dict[int, dict[str, np.ndarray]], dict[str, Any]]:
    t0 = time.perf_counter()
    luts: dict[int, dict[str, np.ndarray]] = {}
    per_k: dict[str, Any] = {}
    for k in range(1, max_k + 1):
        t = time.perf_counter()
        luts[k] = build_lut(k)
        per_k[str(k)] = {
            "n_masks": int(1 << (k * (k - 1) // 2)),
            "n_perms": int(max(1, math.factorial(k - 1)) if k > 1 else 1),
            "n_winner_frames": int(luts[k]["winner_perm"].shape[0]),
            "seconds": float(time.perf_counter() - t),
        }
    info = {
        "max_k": int(max_k),
        "total_seconds": float(time.perf_counter() - t0),
        "per_k": per_k,
    }
    return luts, info


def brute_force_lut(k: int) -> tuple[np.ndarray, list[list[tuple[int, ...]]]]:
    """Reference implementation used only by the checks (all permutations)."""
    pairs = _pairs(k)
    n_masks = 1 << len(pairs)
    perms = [(0,) + p for p in itertools.permutations(range(1, k))]
    pair_index = {p: i for i, p in enumerate(pairs)}
    canon = np.zeros(n_masks, np.uint16)
    winners: list[list[tuple[int, ...]]] = []
    for mask in range(n_masks):
        arr = np.asarray([mask], np.uint32)
        vals = [_permuted_bits(arr, p, pairs, pair_index)[0] for p in perms]
        best = int(min(vals))
        canon[mask] = best
        winners.append([p for p, v in zip(perms, vals) if int(v) == best])
    return canon, winners


def connected_components(adj: Sequence[Sequence[int]]) -> list[list[int]]:
    n = len(adj)
    seen = [False] * n
    comps: list[list[int]] = []
    for start in range(n):
        if seen[start]:
            continue
        stack = [start]
        seen[start] = True
        comp: list[int] = []
        while stack:
            u = stack.pop()
            comp.append(int(u))
            for w in adj[u]:
                w = int(w)
                if not seen[w]:
                    seen[w] = True
                    stack.append(w)
        comps.append(sorted(comp))
    return comps


def all_connected_subsets(adj: Sequence[Sequence[int]], sizes: Iterable[int]) -> list[tuple[int, tuple[int, ...]]]:
    """Every connected induced ``(root, set)`` with ``|set|`` in ``sizes``."""
    n = len(adj)
    wanted = set(int(s) for s in sizes)
    out: list[tuple[int, tuple[int, ...]]] = []
    for root in range(n):
        frontier = [frozenset((root,))]
        if 1 in wanted:
            out.append((root, (root,)))
        for size in range(2, max(wanted, default=1) + 1):
            nxt: set[frozenset[int]] = set()
            for s in frontier:
                for u in s:
                    for w in adj[u]:
                        if w not in s:
                            nxt.add(s | {int(w)})
            frontier = list(nxt)
            if size in wanted:
                for s in frontier:
                    out.append((root, tuple(sorted(s))))
    return out


def _expand_winners(lut: dict[str, np.ndarray], masks: np.ndarray, canon: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Expand the per-mask winner lists (NOT the canonical mask's own list)."""
    counts = lut["count"][masks]
    total = int(counts.sum())
    obj_pos = np.repeat(np.arange(masks.size, dtype=np.int64), counts)
    starts = np.cumsum(counts, dtype=np.int64) - counts
    local = np.arange(total, dtype=np.int64) - np.repeat(starts, counts)
    offsets = np.repeat(lut["ptr"][masks], counts) + local
    perm = lut["winner_perm"][offsets]          # [total, k] uint8
    return obj_pos, perm, np.repeat(canon, counts)


def frames_for_objects(
    atom_row: np.ndarray,
    bond_row: np.ndarray,
    objects: Sequence[tuple[int, tuple[int, ...]]],
    luts: dict[int, dict[str, np.ndarray]],
    *,
    check: bool = False,
) -> dict[str, np.ndarray]:
    """All rooted frames of ``objects`` (object order preserved).

    ``atom_row[a]`` is the attribute row index of local atom ``a``;
    ``bond_row[a, b]`` is the bond attribute row index or ``-1``.
    Objects must be non-decreasing in size, which the cache builder guarantees.
    """
    n_obj = len(objects)
    sizes = np.fromiter((len(s) for _, s in objects), np.int64, count=n_obj)
    if n_obj and np.any(np.diff(sizes) < 0):
        raise ValueError("objects must be grouped by non-decreasing size")
    frame_counts = np.zeros(n_obj, dtype=np.int64)
    atom_attr_parts: list[np.ndarray] = []
    bond_attr_parts: list[np.ndarray] = []
    k_parts: list[np.ndarray] = []

    start = 0
    while start < n_obj:
        k = int(sizes[start])
        stop = start
        while stop < n_obj and int(sizes[stop]) == k:
            stop += 1
        group = objects[start:stop]
        n = len(group)
        # local enumeration order: root, then sorted remaining atoms
        L = np.empty((n, k), dtype=np.int64)
        for i, (root, atoms) in enumerate(group):
            others = [a for a in atoms if int(a) != int(root)]
            if int(root) not in atoms:
                raise ValueError("root not in object")
            if len(others) != k - 1:
                raise ValueError("object set does not match its size")
            L[i, 0] = int(root)
            L[i, 1:] = others
        pairs = _pairs(k)
        adj = bond_row >= 0
        bits = np.zeros(n, dtype=np.uint32)
        for e, (i, j) in enumerate(pairs):
            bits |= adj[L[:, i], L[:, j]].astype(np.uint32) << e
        if np.any(bits >= np.uint32(1 << len(pairs))):
            raise AssertionError("adjacency bits out of range")
        lut = luts[k]
        canon = lut["canon"][bits]
        obj_pos, perm, canon_exp = _expand_winners(lut, bits, canon)
        slots = L[obj_pos[:, None], perm.astype(np.int64)]      # [F, k]
        n_frames = slots.shape[0]
        atoms_attr = np.zeros((n_frames, C.MAX_ATOMS), dtype=np.uint16)
        atoms_attr[:, :k] = atom_row[slots]
        bonds_attr = np.full((n_frames, len(C.EDGE_PAIRS)), -1, dtype=np.int32)
        for e, (i, j) in enumerate(pairs):
            present = ((canon_exp >> e) & 1).astype(bool)
            vals = bond_row[slots[:, i], slots[:, j]]
            g = GLOBAL_PAIR_INDEX[(i, j)]
            bonds_attr[:, g] = np.where(present, vals, -1)
        if check:
            # every frame must reproduce the canonical adjacency exactly in the
            # *global* 15-slot layout
            got = (bonds_attr >= 0)
            want = np.zeros_like(got)
            for e, (i, j) in enumerate(pairs):
                want[:, GLOBAL_PAIR_INDEX[(i, j)]] = ((canon_exp >> e) & 1).astype(bool)
            if not np.array_equal(got, want):
                raise AssertionError("frame adjacency does not match the canonical pattern")
        atom_attr_parts.append(atoms_attr)
        bond_attr_parts.append(bonds_attr)
        k_parts.append(np.full(n_frames, k, dtype=np.uint8))
        counts = np.bincount(obj_pos, minlength=n).astype(np.int64)
        frame_counts[start:stop] = counts
        if int(counts.sum()) != n_frames:
            raise AssertionError("frame expansion lost frames")
        start = stop

    if atom_attr_parts:
        frame_atom_attr = np.concatenate(atom_attr_parts, axis=0)
        frame_bond_attr = np.concatenate(bond_attr_parts, axis=0)
        frame_k = np.concatenate(k_parts, axis=0)
    else:
        frame_atom_attr = np.zeros((0, C.MAX_ATOMS), np.uint16)
        frame_bond_attr = np.zeros((0, len(C.EDGE_PAIRS)), np.int32)
        frame_k = np.zeros(0, np.uint8)
    return {
        "obj_frame_count": frame_counts,
        "frame_atom_attr": frame_atom_attr,
        "frame_bond_attr": frame_bond_attr,
        "frame_k": frame_k,
    }

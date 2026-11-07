"""Frozen relation mathematics: distances, motif pairs, REAL / SHUFFLE features.

Definitions (all frozen in ``rie_common``):

* ``d(M_i, M_j) = min shortest-path distance(u, v)`` over ``u in M_i``,
  ``v in M_j`` on the **physical bond graph**; only finite distances inside one
  connected component are eligible,
* pairs with ``d < 3`` are dropped; ``d in {3,4}`` -> bin 0, ``{5,6}`` -> bin 1,
  ``>= 7`` -> bin 2,
* ``pair_feature(i,j) = concat(z_i * z_j, |z_i - z_j|, onehot36(family_i, family_j), 1)``
  with ``z_i = mean(h_v for v in M_i)``,
* ``S_b(G) = sum_{pair in bin b} pair_feature / sqrt(1 + n_pairs_in_b)``,
  all-zero when the bin has no pair,
* SHUFFLE keeps the motif set, the motif pairs, every pair feature and the
  per-bin pair counts; it permutes only **which pair carries which distance
  bin**, with a per-molecule RNG derived from ``SHA256("rie-distance-shuffle-v1|" + row_id)``.

Everything in this module is a pure function of tensors/arrays so the invariance
audits can exercise it on CPU without the frozen encoder.
"""

from __future__ import annotations

from collections import deque
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import rie_common as C


# ---------------------------------------------------------------------------
# physical graph distances
# ---------------------------------------------------------------------------


def adjacency(n_atoms: int, edges: Iterable[Sequence[int]]) -> list[list[int]]:
    """Undirected adjacency list of the physical bond graph."""
    adj: list[list[int]] = [[] for _ in range(int(n_atoms))]
    for u, v in edges:
        u, v = int(u), int(v)
        if u == v:
            raise ValueError(f"self-loop on atom {u}")
        if not (0 <= u < n_atoms and 0 <= v < n_atoms):
            raise ValueError(f"edge ({u},{v}) outside {n_atoms} atoms")
        adj[u].append(v)
        adj[v].append(u)
    return adj


def shortest_paths(n_atoms: int, edges: Iterable[Sequence[int]]) -> np.ndarray:
    """All-pairs unweighted BFS distances; ``-1`` marks 'different component'."""
    n = int(n_atoms)
    adj = adjacency(n, edges)
    dist = -np.ones((n, n), dtype=np.int32)
    for source in range(n):
        row = dist[source]
        row[source] = 0
        queue: deque[int] = deque([source])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if row[v] < 0:
                    row[v] = row[u] + 1
                    queue.append(v)
    return dist


def reference_shortest_paths(n_atoms: int, edges: Iterable[Sequence[int]]) -> np.ndarray:
    """Independent reference implementation (``scipy.sparse.csgraph``), for audits."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import shortest_path

    n = int(n_atoms)
    rows: list[int] = []
    cols: list[int] = []
    for u, v in edges:
        rows.extend([int(u), int(v)])
        cols.extend([int(v), int(u)])
    if not rows:
        graph = coo_matrix((n, n))
    else:
        graph = coo_matrix((np.ones(len(rows), dtype=np.int8), (rows, cols)), shape=(n, n))
    d = shortest_path(graph, method="D", directed=False, unweighted=True)
    out = np.where(np.isfinite(d), np.rint(d), -1.0).astype(np.int32)
    return out


def connected_components(n_atoms: int, edges: Iterable[Sequence[int]]) -> np.ndarray:
    """Component label per atom (BFS), used by the cross-component audit."""
    adj = adjacency(n_atoms, edges)
    labels = -np.ones(int(n_atoms), dtype=np.int32)
    current = 0
    for start in range(int(n_atoms)):
        if labels[start] >= 0:
            continue
        labels[start] = current
        queue = deque([start])
        while queue:
            u = queue.popleft()
            for v in adj[u]:
                if labels[v] < 0:
                    labels[v] = current
                    queue.append(v)
        current += 1
    return labels


# ---------------------------------------------------------------------------
# motif pairs and distance bins
# ---------------------------------------------------------------------------


def motif_pair_distances(
    motifs: Sequence[Sequence[int]], dist: np.ndarray
) -> list[tuple[int, int, int]]:
    """``(i, j, min_distance)`` for every motif pair with a finite distance."""
    out: list[tuple[int, int, int]] = []
    n = len(motifs)
    for i in range(n):
        atoms_i = motifs[i]
        for j in range(i + 1, n):
            best = -1
            for u in atoms_i:
                row = dist[int(u)]
                for v in motifs[j]:
                    d = int(row[int(v)])
                    if d >= 0 and (best < 0 or d < best):
                        best = d
            if best >= 0:
                out.append((i, j, best))
    return out


def eligible_pairs(
    motifs: Sequence[Sequence[int]], dist: np.ndarray
) -> list[tuple[int, int, int, int]]:
    """``(i, j, distance, bin)`` for every eligible pair (``d >= 3``, same component)."""
    out: list[tuple[int, int, int, int]] = []
    for i, j, d in motif_pair_distances(motifs, dist):
        b = C.bin_for_distance(d)
        if b >= 0:
            out.append((i, j, int(d), int(b)))
    return out


def shuffle_bins(bins: Sequence[int], official_row_id: int) -> np.ndarray:
    """Permute the distance-bin list of one molecule (fixed per-row RNG)."""
    arr = np.asarray(list(bins), dtype=np.int8)
    rng = np.random.Generator(np.random.PCG64(C.shuffle_seed_sequence(int(official_row_id))))
    if arr.size <= 1:
        return arr.copy()
    return arr[rng.permutation(arr.size)].astype(np.int8, copy=True)


# ---------------------------------------------------------------------------
# per-molecule structure (the CPU-only, label-blind artifact)
# ---------------------------------------------------------------------------


def molecule_structure(
    *,
    n_atoms: int,
    edges: Sequence[Sequence[int]],
    motifs: Sequence[tuple[int, Sequence[int]]],
    official_row_id: int,
    parse_failed: bool = False,
) -> dict[str, Any]:
    """Everything about one molecule that does not depend on the encoder.

    Returned arrays are already in the canonical (deduped, sorted) motif order so
    the aggregation order is independent of how the motifs were supplied.
    """
    if parse_failed:
        motifs = []
    unique = sorted(
        {(int(fam), tuple(sorted({int(a) for a in atoms}))) for fam, atoms in motifs},
        key=lambda item: (int(item[0]), tuple(int(a) for a in item[1])),
    )
    families = np.asarray([f for f, _ in unique], dtype=np.int8)
    atom_lists = [np.asarray(atoms, dtype=np.int64) for _, atoms in unique]
    motif_atoms = np.concatenate(atom_lists) if atom_lists else np.zeros(0, dtype=np.int64)
    motif_len = np.asarray([len(a) for a in atom_lists], dtype=np.int64)
    motif_atom_ptr = np.concatenate([[0], np.cumsum(motif_len)]).astype(np.int64)

    if parse_failed or not unique:
        pairs = np.zeros((0, 4), dtype=np.int64)
    else:
        dist = shortest_paths(int(n_atoms), edges)
        pairs = np.asarray(eligible_pairs(atom_lists, dist), dtype=np.int64).reshape(-1, 4)

    pair_a = pairs[:, 0].astype(np.int32) if len(pairs) else np.zeros(0, np.int32)
    pair_b = pairs[:, 1].astype(np.int32) if len(pairs) else np.zeros(0, np.int32)
    pair_bin = pairs[:, 3].astype(np.int8) if len(pairs) else np.zeros(0, np.int8)
    pair_famidx = (
        np.asarray(
            [C.family_pair_index(int(families[i]), int(families[j])) for i, j in zip(pair_a, pair_b)],
            dtype=np.int8,
        )
        if len(pairs)
        else np.zeros(0, np.int8)
    )
    pair_bin_shuffle = shuffle_bins(pair_bin, int(official_row_id)) if len(pair_bin) else pair_bin.copy()

    for atoms in atom_lists:
        if atoms.size and int(atoms[-1]) >= int(n_atoms):
            raise ValueError(f"motif atom {int(atoms[-1])} outside {n_atoms} atoms")

    return {
        "n_atoms": int(n_atoms),
        "parse_failed": bool(parse_failed),
        "motif_family": families,
        "motif_len": motif_len,
        "motif_atom_ptr": motif_atom_ptr,
        "motif_atoms": motif_atoms,
        "pair_a": pair_a,
        "pair_b": pair_b,
        "pair_bin": pair_bin,
        "pair_bin_shuffle": pair_bin_shuffle,
        "pair_famidx": pair_famidx,
        "pair_distance": pairs[:, 2].astype(np.int32) if len(pairs) else np.zeros(0, np.int32),
        "n_motifs": int(len(unique)),
        "n_pairs": int(len(pairs)),
        "bin_counts": np.bincount(pair_bin, minlength=C.N_BINS).astype(np.int64)
        if len(pair_bin)
        else np.zeros(C.N_BINS, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# concatenated split-level structures
# ---------------------------------------------------------------------------

#: Per-molecule arrays that concatenate without offset arithmetic.
#: ``motif_atom_ptr`` is deliberately excluded: its entries are absolute offsets
#: into the concatenated ``motif_atoms`` array and are written separately.
STRUCT_ARRAY_KEYS = (
    "motif_family",
    "motif_len",
    "motif_atoms",
    "pair_a",
    "pair_b",
    "pair_bin",
    "pair_bin_shuffle",
    "pair_famidx",
    "pair_distance",
)


def concat_structures(items: Sequence[Mapping[str, Any]], *, rows: np.ndarray) -> dict[str, Any]:
    """Concatenate per-molecule structures in split row order (offsets applied)."""
    n = len(items)
    if n != int(np.asarray(rows).size):
        raise ValueError(f"{n} structures for {int(np.asarray(rows).size)} rows")
    motif_ptr = np.zeros(n + 1, dtype=np.int64)
    pair_ptr = np.zeros(n + 1, dtype=np.int64)
    motif_atom_ptr = np.zeros(int(sum(len(it["motif_len"]) for it in items)) + 1, dtype=np.int64)
    parts: dict[str, list[np.ndarray]] = {k: [] for k in STRUCT_ARRAY_KEYS}
    motif_off = 0
    pair_off = 0
    atom_off = 0
    for idx, item in enumerate(items):
        nm = int(item["n_motifs"])
        npa = int(item["n_pairs"])
        motif_ptr[idx + 1] = motif_ptr[idx] + nm
        pair_ptr[idx + 1] = pair_ptr[idx] + npa
        for key in STRUCT_ARRAY_KEYS:
            parts[key].append(np.asarray(item[key]))
        motif_atom_ptr[motif_off : motif_off + nm + 1] = np.asarray(item["motif_atom_ptr"]) + atom_off
        # pair motif indices are stored split-global (molecule offset applied)
        parts["pair_a"][-1] = parts["pair_a"][-1].astype(np.int64) + motif_off
        parts["pair_b"][-1] = parts["pair_b"][-1].astype(np.int64) + motif_off
        motif_off += nm
        pair_off += npa
        atom_off += int(np.asarray(item["motif_atoms"]).size)
    out: dict[str, Any] = {
        "n_molecules": n,
        "rows": np.asarray(rows, dtype=np.int64),
        "motif_ptr": motif_ptr,
        "pair_ptr": pair_ptr,
        "motif_atom_ptr": motif_atom_ptr,
        "parse_failed": np.asarray([bool(it["parse_failed"]) for it in items], dtype=bool),
        "n_motifs": np.asarray([int(it["n_motifs"]) for it in items], dtype=np.int32),
        "n_pairs": np.asarray([int(it["n_pairs"]) for it in items], dtype=np.int32),
        "bin_counts": np.stack([np.asarray(it["bin_counts"], dtype=np.int64) for it in items])
        if n
        else np.zeros((0, C.N_BINS), dtype=np.int64),
        "n_atoms": np.asarray([int(it["n_atoms"]) for it in items], dtype=np.int64),
    }
    for key in STRUCT_ARRAY_KEYS:
        blocks = parts[key]
        out[key] = np.concatenate(blocks) if blocks else np.zeros(0)
        if key in ("motif_family", "pair_bin", "pair_bin_shuffle", "pair_famidx"):
            out[key] = out[key].astype(np.int8)
        elif key in ("pair_a", "pair_b"):
            out[key] = out[key].astype(np.int64)
    return out


def merge_structures(chunks: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Merge consecutive chunk-level structures into one split-level structure."""
    chunks = [c for c in chunks if int(c["n_molecules"]) > 0]
    if not chunks:
        raise ValueError("nothing to merge")
    n = int(sum(int(c["n_molecules"]) for c in chunks))
    motif_ptr = np.zeros(n + 1, dtype=np.int64)
    pair_ptr = np.zeros(n + 1, dtype=np.int64)
    motif_atom_ptr_parts: list[np.ndarray] = []
    motif_atom_off = 0
    motif_off = 0
    pair_off = 0
    molecule_off = 0
    parts: dict[str, list[np.ndarray]] = {k: [] for k in STRUCT_ARRAY_KEYS}
    parse_failed: list[np.ndarray] = []
    n_motifs: list[np.ndarray] = []
    n_pairs: list[np.ndarray] = []
    bin_counts: list[np.ndarray] = []
    n_atoms: list[np.ndarray] = []
    rows: list[np.ndarray] = []
    for chunk in chunks:
        cm = int(chunk["n_molecules"])
        local_motif_ptr = np.asarray(chunk["motif_ptr"], dtype=np.int64)
        local_pair_ptr = np.asarray(chunk["pair_ptr"], dtype=np.int64)
        motif_ptr[molecule_off : molecule_off + cm + 1] = local_motif_ptr + motif_off
        pair_ptr[molecule_off : molecule_off + cm + 1] = local_pair_ptr + pair_off
        local_atom_ptr = np.asarray(chunk["motif_atom_ptr"], dtype=np.int64)
        # drop each chunk's trailing boundary: it is re-added once at the end
        motif_atom_ptr_parts.append(local_atom_ptr[:-1] + motif_atom_off)
        for key in STRUCT_ARRAY_KEYS:
            block = np.asarray(chunk[key])
            if key in ("pair_a", "pair_b"):
                block = block.astype(np.int64) + motif_off
            parts[key].append(block)
        parse_failed.append(np.asarray(chunk["parse_failed"], dtype=bool))
        n_motifs.append(np.asarray(chunk["n_motifs"], dtype=np.int32))
        n_pairs.append(np.asarray(chunk["n_pairs"], dtype=np.int32))
        bin_counts.append(np.asarray(chunk["bin_counts"], dtype=np.int64).reshape(cm, C.N_BINS))
        n_atoms.append(np.asarray(chunk["n_atoms"], dtype=np.int64))
        rows.append(np.asarray(chunk["rows"], dtype=np.int64))
        motif_atom_off += int(local_atom_ptr[-1])
        motif_off += int(local_motif_ptr[-1])
        pair_off += int(local_pair_ptr[-1])
        molecule_off += cm
    out: dict[str, Any] = {
        "n_molecules": n,
        "rows": np.concatenate(rows),
        "motif_ptr": motif_ptr,
        "pair_ptr": pair_ptr,
        "motif_atom_ptr": np.concatenate(
            motif_atom_ptr_parts + [np.asarray([motif_atom_off], dtype=np.int64)]
        ),
        "parse_failed": np.concatenate(parse_failed),
        "n_motifs": np.concatenate(n_motifs),
        "n_pairs": np.concatenate(n_pairs),
        "bin_counts": np.concatenate(bin_counts, axis=0),
        "n_atoms": np.concatenate(n_atoms),
    }
    for key in STRUCT_ARRAY_KEYS:
        out[key] = np.concatenate(parts[key]) if parts[key] else np.zeros(0)
        if key in ("motif_family", "pair_bin", "pair_bin_shuffle", "pair_famidx"):
            out[key] = out[key].astype(np.int8)
    return out


def save_structures(struct: Mapping[str, Any], path: Any) -> dict[str, Any]:
    """Save one split's structure arrays to ``.npz`` with a hash manifest."""
    arrays = {k: np.ascontiguousarray(v) for k, v in struct.items() if isinstance(v, np.ndarray)}
    np.savez(path, **arrays)
    record = {k: {"shape": list(v.shape), "dtype": str(v.dtype), "sha256": C.sha256_array(v)} for k, v in arrays.items()}
    return record


def load_structures(path: Any) -> dict[str, Any]:
    with np.load(path, allow_pickle=False) as payload:
        struct = {k: payload[k] for k in payload.files}
    # ``n_molecules`` is a plain int in memory; .npz only stores ndarrays
    struct["n_molecules"] = int(np.asarray(struct["rows"]).size)
    return struct


# ---------------------------------------------------------------------------
# batch gathering (split-level structure -> per-batch tensors)
# ---------------------------------------------------------------------------


def _ranges(starts: np.ndarray, counts: np.ndarray) -> np.ndarray:
    """Concatenated ``arange(start, start + count)`` for every entry."""
    counts = np.asarray(counts, dtype=np.int64)
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64)
    offsets = np.repeat(np.asarray(starts, dtype=np.int64) - np.concatenate([[0], np.cumsum(counts)[:-1]]), counts)
    return offsets + np.arange(total, dtype=np.int64)


def gather_batch(struct: Mapping[str, Any], positions: np.ndarray, atom_counts: np.ndarray) -> dict[str, Any]:
    """Slice a split-level structure to one batch of split positions."""
    pos = np.asarray(positions, dtype=np.int64)
    atom_counts = np.asarray(atom_counts, dtype=np.int64)
    B = int(pos.size)
    if atom_counts.size != B:
        raise ValueError(f"{atom_counts.size} atom counts for {B} molecules")
    motif_ptr = np.asarray(struct["motif_ptr"])[pos]
    pair_ptr = np.asarray(struct["pair_ptr"])[pos]
    motif_counts = np.asarray(struct["motif_ptr"])[pos + 1] - motif_ptr
    pair_counts = np.asarray(struct["pair_ptr"])[pos + 1] - pair_ptr

    motif_idx = _ranges(motif_ptr, motif_counts)
    pair_idx = _ranges(pair_ptr, pair_counts)
    motif_mol = np.repeat(np.arange(B, dtype=np.int64), motif_counts)
    pair_mol = np.repeat(np.arange(B, dtype=np.int64), pair_counts)

    motif_len = np.asarray(struct["motif_len"])[motif_idx].astype(np.int64)
    motif_atom_ptr_split = np.asarray(struct["motif_atom_ptr"])[motif_idx]
    motif_atoms_local = np.asarray(struct["motif_atoms"])[
        _ranges(motif_atom_ptr_split, motif_len)
    ].astype(np.int64)

    atom_off = np.concatenate([[0], np.cumsum(atom_counts)[:-1]]).astype(np.int64)
    motif_atom_owner = np.repeat(np.arange(motif_idx.size, dtype=np.int64), motif_len)
    motif_atoms_batch = motif_atoms_local + atom_off[motif_mol][motif_atom_owner]

    motif_off = np.concatenate([[0], np.cumsum(motif_counts)[:-1]]).astype(np.int64)
    pair_a = np.asarray(struct["pair_a"])[pair_idx].astype(np.int64) - motif_ptr[pair_mol] + motif_off[pair_mol]
    pair_b = np.asarray(struct["pair_b"])[pair_idx].astype(np.int64) - motif_ptr[pair_mol] + motif_off[pair_mol]

    return {
        "n_molecules": B,
        "motif_mol": motif_mol,
        "motif_len": motif_len,
        "motif_atom_ptr": np.concatenate([[0], np.cumsum(motif_len)]).astype(np.int64),
        "motif_atoms": motif_atoms_batch,
        "motif_family": np.asarray(struct["motif_family"])[motif_idx].astype(np.int8),
        "pair_mol": pair_mol,
        "pair_a": pair_a,
        "pair_b": pair_b,
        "pair_famidx": np.asarray(struct["pair_famidx"])[pair_idx].astype(np.int64),
        "pair_bin": np.asarray(struct["pair_bin"])[pair_idx].astype(np.int64),
        "pair_bin_shuffle": np.asarray(struct["pair_bin_shuffle"])[pair_idx].astype(np.int64),
        "atom_counts": atom_counts,
        "parse_failed": np.asarray(struct["parse_failed"])[pos].astype(bool),
    }


# ---------------------------------------------------------------------------
# relation features
# ---------------------------------------------------------------------------


def motif_states(h: np.ndarray, gathered: Mapping[str, Any]) -> np.ndarray:
    """``z_i = mean(h_v for v in M_i)``; returns ``(n_motifs_batch, width)``."""
    h = np.asarray(h, dtype=np.float32)
    n_motifs = int(gathered["motif_len"].size)
    if n_motifs == 0:
        return np.zeros((0, h.shape[1]), dtype=np.float32)
    starts = np.asarray(gathered["motif_atom_ptr"][:-1], dtype=np.int64)
    atoms = np.asarray(gathered["motif_atoms"], dtype=np.int64)
    sums = np.add.reduceat(h[atoms], starts, axis=0)
    return (sums / np.asarray(gathered["motif_len"], dtype=np.float32)[:, None]).astype(np.float32)


def pair_features(z: np.ndarray, gathered: Mapping[str, Any]) -> np.ndarray:
    """Dense ``(n_pairs, 165)`` pair features for one batch."""
    pair_a = np.asarray(gathered["pair_a"], dtype=np.int64)
    n_pairs = int(pair_a.size)
    out = np.zeros((n_pairs, C.PAIR_FEATURE_DIM), dtype=np.float32)
    if n_pairs == 0:
        return out
    pair_b = np.asarray(gathered["pair_b"], dtype=np.int64)
    za = z[pair_a]
    zb = z[pair_b]
    width = int(z.shape[1])
    out[:, 0:width] = za * zb
    out[:, width : 2 * width] = np.abs(za - zb)
    pair_famidx = np.asarray(gathered["pair_famidx"], dtype=np.int64)
    out[np.arange(n_pairs), 2 * width + pair_famidx] = 1.0
    out[:, -1] = 1.0
    return out


def aggregate_features(
    features: np.ndarray, keys: np.ndarray, n_keys: int
) -> tuple[np.ndarray, np.ndarray]:
    """Sum ``features`` per key, then divide by ``sqrt(1 + count)``."""
    n_keys = int(n_keys)
    dim = int(features.shape[1])
    sums = np.zeros((n_keys, dim), dtype=np.float64)
    counts = np.zeros(n_keys, dtype=np.int64)
    if features.shape[0]:
        order = np.argsort(keys, kind="stable")
        keys_sorted = keys[order]
        uniq, starts, counts_u = np.unique(keys_sorted, return_index=True, return_counts=True)
        blocked = np.add.reduceat(features[order].astype(np.float64), starts, axis=0)
        sums[uniq] = blocked
        counts[uniq] = counts_u
    scale = 1.0 / np.sqrt(1.0 + counts.astype(np.float64))
    return (sums * scale[:, None]).astype(np.float32), counts


def batch_relation_features(
    h: np.ndarray, gathered: Mapping[str, Any], *, which: str = "real"
) -> tuple[np.ndarray, np.ndarray]:
    """Relation tensors ``(B, 3, 165)`` and per-bin pair counts ``(B, 3)``."""
    if which not in ("real", "shuffle"):
        raise ValueError(f"unknown relation variant {which!r}")
    B = int(gathered["n_molecules"])
    z = motif_states(h, gathered)
    feats = pair_features(z, gathered)
    bins = np.asarray(gathered["pair_bin" if which == "real" else "pair_bin_shuffle"], dtype=np.int64)
    keys = np.asarray(gathered["pair_mol"], dtype=np.int64) * C.N_BINS + bins
    out, counts = aggregate_features(feats, keys, B * C.N_BINS)
    return out.reshape(B, C.N_BINS, C.PAIR_FEATURE_DIM), counts.reshape(B, C.N_BINS)


def molecule_relation_features(
    h: np.ndarray,
    motifs: Sequence[tuple[int, Sequence[int]]],
    *,
    n_atoms: int,
    edges: Sequence[Sequence[int]],
    official_row_id: int,
) -> dict[str, Any]:
    """Single-molecule REAL/SHUFFLE features (CPU reference used by the audit)."""
    struct = molecule_structure(
        n_atoms=n_atoms, edges=edges, motifs=motifs, official_row_id=official_row_id
    )
    items = [struct]
    split = concat_structures(items, rows=np.asarray([official_row_id], dtype=np.int64))
    gathered = gather_batch(split, np.asarray([0], dtype=np.int64), np.asarray([n_atoms], dtype=np.int64))
    real, counts_real = batch_relation_features(h, gathered, which="real")
    shuffle, counts_shuffle = batch_relation_features(h, gathered, which="shuffle")
    return {
        "real": real[0],
        "shuffle": shuffle[0],
        "bin_counts_real": counts_real[0],
        "bin_counts_shuffle": counts_shuffle[0],
        "structure": struct,
    }


# ---------------------------------------------------------------------------
# RMS normalisation (REAL Train only)
# ---------------------------------------------------------------------------


def accumulate_sumsq(acc: np.ndarray, block: np.ndarray) -> np.ndarray:
    """Accumulate ``sum(S**2)`` per flattened column in float64."""
    flat = np.asarray(block, dtype=np.float64).reshape(-1, C.RELATION_DIM_FLAT)
    acc += (flat * flat).sum(axis=0)
    return acc


def rms_scale_from_sumsq(sumsq: np.ndarray, n_rows: int) -> tuple[np.ndarray, dict[str, Any]]:
    """``rms_k = sqrt(mean_train(S_k ** 2))``; floor to 1.0 below ``RMS_EPS``."""
    sumsq = np.asarray(sumsq, dtype=np.float64)
    n = int(n_rows)
    if n <= 0:
        raise ValueError("RMS scale needs at least one Train row")
    rms = np.sqrt(sumsq / float(n))
    applied = np.where(rms < C.RMS_EPS, C.RMS_FLOOR_SCALE, rms)
    audit = {
        "n_rows_fit": n,
        "dim": int(rms.size),
        "definition": "rms_k = sqrt(mean over REAL Train rows of S_k**2), per flattened "
        "(bin, feature) column; no mean centering",
        "floor_epsilon": float(C.RMS_EPS),
        "floor_scale": float(C.RMS_FLOOR_SCALE),
        "n_floored_columns": int(np.sum(rms < C.RMS_EPS)),
        "floored_columns": np.flatnonzero(rms < C.RMS_EPS).astype(int).tolist(),
        "rms_min": float(rms.min()),
        "rms_max": float(rms.max()),
        "rms_median": float(np.median(rms)),
        "fit_split": "real_train_only",
        "valid_used": False,
    }
    return applied.astype(np.float32), audit


def apply_scale(block: np.ndarray, scale: np.ndarray) -> np.ndarray:
    out = np.asarray(block, dtype=np.float32).reshape(-1, C.RELATION_DIM_FLAT) / np.asarray(
        scale, dtype=np.float32
    )[None, :]
    return out.reshape((-1,) + C.RELATION_SHAPE).astype(np.float32)


def feature_health(block: np.ndarray, *, threshold: float = 1e-12) -> dict[str, Any]:
    """Zero/non-zero columns, NaN/Inf, abs-max summary for an ``(N,3,165)`` block."""
    flat = np.asarray(block, dtype=np.float64).reshape(-1, C.RELATION_DIM_FLAT)
    if flat.size == 0:
        raise ValueError("empty feature block")
    nonzero = np.abs(flat).max(axis=0) > threshold
    return {
        "rows": int(flat.shape[0]),
        "dim": int(flat.shape[1]),
        "n_zero_columns": int((~nonzero).sum()),
        "zero_columns": np.flatnonzero(~nonzero).astype(int).tolist(),
        "n_nonzero_columns": int(nonzero.sum()),
        "nan_count": int(np.isnan(flat).sum()),
        "inf_count": int(np.isinf(flat).sum()),
        "abs_max": float(np.abs(flat).max()),
        "abs_mean": float(np.abs(flat).mean()),
        "zero_row_fraction": float((np.abs(flat).max(axis=1) == 0.0).mean()),
        "nonzero_threshold": float(threshold),
    }


__all__ = [
    "STRUCT_ARRAY_KEYS",
    "accumulate_sumsq",
    "adjacency",
    "aggregate_features",
    "apply_scale",
    "batch_relation_features",
    "concat_structures",
    "connected_components",
    "eligible_pairs",
    "feature_health",
    "gather_batch",
    "load_structures",
    "molecule_relation_features",
    "molecule_structure",
    "merge_structures",
    "motif_pair_distances",
    "motif_states",
    "pair_features",
    "reference_shortest_paths",
    "rms_scale_from_sumsq",
    "save_structures",
    "shortest_paths",
    "shuffle_bins",
]

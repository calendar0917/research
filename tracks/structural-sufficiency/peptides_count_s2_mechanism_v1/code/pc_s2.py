"""Exact directed non-backtracking 2-hop S2 enumeration and content hashing.

Definitions frozen in ``PROTOCOL.md``:

* a path is an ordered triple ``(u, v, w)`` of atoms with ``u != w`` and
  ``(u, v)``, ``(v, w)`` stored undirected bonds;
* TYPE-S2 key = (root atom column 0, first bond column 0, middle atom column 0,
  second bond column 0, endpoint atom column 0);
* FULL-S2 key = all 9 raw atom columns of root / middle / endpoint and all 3
  raw bond columns of the first / second bond, in that order;
* every key is a fixed byte string hashed with ``blake2b(digest_size=8,
  person=b"pcs2v1")``; the bucket is ``digest % 2048``;
* counts are additive non-negative integer counts; FULL-S2-BINARY is
  ``counts > 0``.

No Python built-in randomised ``hash()`` is used.
"""

from __future__ import annotations

import hashlib
from typing import Any, Iterable

import numpy as np

import pc_common as C


# ---------------------------------------------------------------------------
# Enumeration
# ---------------------------------------------------------------------------


def enumerate_paths(
    atom_raw: np.ndarray,
    bond_raw: np.ndarray,
    bond_index: np.ndarray,
) -> dict[str, Any]:
    """Enumerate every directed non-backtracking two-hop path exactly once."""
    atom_raw = np.asarray(atom_raw, dtype=np.int64)
    bond_raw = np.asarray(bond_raw, dtype=np.int64)
    bond_index = np.asarray(bond_index, dtype=np.int64)
    n = int(atom_raw.shape[0])
    m = int(bond_index.shape[0])
    if atom_raw.ndim != 2 or atom_raw.shape[1] != len(C.ATOM_VOCABS):
        raise ValueError(f"atom_raw shape {atom_raw.shape}")
    if bond_raw.shape != (m, len(C.BOND_VOCABS)):
        raise ValueError(f"bond_raw shape {bond_raw.shape} != {(m, len(C.BOND_VOCABS))}")
    if bond_index.shape != (m, 2):
        raise ValueError(f"bond_index shape {bond_index.shape}")
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for k in range(m):
        u, v = int(bond_index[k, 0]), int(bond_index[k, 1])
        if u == v:
            raise ValueError("self-loop in bond_index")
        if u >= v:
            raise ValueError("bond_index not canonical (u<v)")
        adj[u].append((v, k))
        adj[v].append((u, k))
    for row in adj:
        row.sort()

    roots: list[int] = []
    mids: list[int] = []
    ends: list[int] = []
    k1s: list[int] = []
    k2s: list[int] = []
    per_root = np.zeros(n, dtype=np.int64)
    for u in range(n):
        start = len(roots)
        for v, k1 in adj[u]:
            for w, k2 in adj[v]:
                if w == u:
                    continue
                roots.append(u)
                mids.append(v)
                ends.append(w)
                k1s.append(k1)
                k2s.append(k2)
        per_root[u] = len(roots) - start

    out = {
        "root": np.asarray(roots, dtype=np.int64),
        "mid": np.asarray(mids, dtype=np.int64),
        "end": np.asarray(ends, dtype=np.int64),
        "k1": np.asarray(k1s, dtype=np.int64),
        "k2": np.asarray(k2s, dtype=np.int64),
        "n_paths": int(len(roots)),
        "per_root": per_root,
        "adj": adj,
    }
    return out


# ---------------------------------------------------------------------------
# Keys
# ---------------------------------------------------------------------------


def type_keys(atom_raw: np.ndarray, bond_raw: np.ndarray, paths: dict[str, Any]) -> np.ndarray:
    atom_raw = np.asarray(atom_raw, dtype=np.int64)
    bond_raw = np.asarray(bond_raw, dtype=np.int64)
    root, mid, end = paths["root"], paths["mid"], paths["end"]
    k1, k2 = paths["k1"], paths["k2"]
    if root.size == 0:
        return np.zeros((0, 5), dtype=np.uint8)
    out = np.stack(
        [
            atom_raw[root, 0],
            bond_raw[k1, 0],
            atom_raw[mid, 0],
            bond_raw[k2, 0],
            atom_raw[end, 0],
        ],
        axis=1,
    )
    return out.astype(np.uint8)


def full_keys(atom_raw: np.ndarray, bond_raw: np.ndarray, paths: dict[str, Any]) -> np.ndarray:
    atom_raw = np.asarray(atom_raw, dtype=np.int64)
    bond_raw = np.asarray(bond_raw, dtype=np.int64)
    root, mid, end = paths["root"], paths["mid"], paths["end"]
    k1, k2 = paths["k1"], paths["k2"]
    if root.size == 0:
        return np.zeros((0, 33), dtype=np.uint8)
    out = np.concatenate(
        [
            atom_raw[root],  # 9
            bond_raw[k1],  # 3
            atom_raw[mid],  # 9
            bond_raw[k2],  # 3
            atom_raw[end],  # 9
        ],
        axis=1,
    )
    return out.astype(np.uint8)


def split_halves(full: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    full = np.asarray(full, dtype=np.uint8)
    if full.ndim != 2 or full.shape[1] != 33:
        raise ValueError(f"FULL key width {full.shape}")
    return full[:, :21], full[:, 21:]


def graph_content_seed(atom_raw: np.ndarray, bond_raw: np.ndarray, bond_index: np.ndarray) -> int:
    """Label-free, process-independent graph seed derived from raw content."""
    h = hashlib.sha256()
    h.update(b"pc-graph-v1")
    h.update(np.ascontiguousarray(atom_raw, dtype=np.uint8).tobytes())
    h.update(np.ascontiguousarray(bond_raw, dtype=np.uint8).tobytes())
    h.update(np.ascontiguousarray(bond_index, dtype=np.int32).tobytes())
    return int.from_bytes(h.digest()[:8], "little")


def canonical_path_order(full: np.ndarray) -> np.ndarray:
    """Lexicographic order of the 33-byte FULL keys, independent of node ids."""
    full = np.asarray(full, dtype=np.uint8)
    if full.shape[0] <= 1:
        return np.arange(full.shape[0], dtype=np.int64)
    return np.lexsort(full[:, ::-1].T).astype(np.int64)


def canonical_full_seed(sorted_full: np.ndarray) -> int:
    """Seed from the canonical sorted FULL-key multiset (relabelling invariant)."""
    h = hashlib.sha256()
    h.update(b"pc-canonical-full-v1")
    h.update(np.ascontiguousarray(sorted_full, dtype=np.uint8).tobytes())
    return int.from_bytes(h.digest()[:8], "little")


def null_permutation(n_paths: int, seed: int) -> np.ndarray:
    """Fixed pseudo-random derangement over paths (identity only for n<=1)."""
    n = int(n_paths)
    base = np.arange(n, dtype=np.int64)
    if n <= 1:
        return base
    rng = np.random.Generator(np.random.PCG64(int(seed) & ((1 << 63) - 1)))
    for _ in range(1000):
        perm = rng.permutation(n).astype(np.int64)
        if np.all(perm != base):
            return perm
    shift = 1 + (int(seed) % (n - 1))
    return (base + shift) % n


def null_keys(first: np.ndarray, second: np.ndarray, seed: int) -> tuple[np.ndarray, np.ndarray]:
    first = np.asarray(first, dtype=np.uint8)
    second = np.asarray(second, dtype=np.uint8)
    if first.shape[0] != second.shape[0]:
        raise ValueError("half mismatch")
    perm = null_permutation(first.shape[0], seed)
    if first.shape[0] == 0:
        return np.zeros((0, 33), dtype=np.uint8), perm
    return np.concatenate([first, second[perm]], axis=1).astype(np.uint8), perm


# ---------------------------------------------------------------------------
# Hashing / counting
# ---------------------------------------------------------------------------


def count_unique_keys(keys: np.ndarray, *, digest_audit: bool = True) -> tuple[np.ndarray, dict[str, Any], np.ndarray]:
    """Return (2048 counts, stats, unique rows) for one graph's keys."""
    keys = np.asarray(keys, dtype=np.uint8)
    vec = np.zeros(C.N_BUCKETS, dtype=np.float32)
    if keys.ndim != 2 or keys.shape[0] == 0:
        return vec, {
            "n_paths": 0,
            "n_unique_patterns": 0,
            "n_nonzero_buckets": 0,
            "n_unique_64bit_digests": 0,
            "n_64bit_digest_collisions": 0,
            "bucket_collision_rate": 0.0,
            "max_bucket_count": 0,
        }, np.zeros((0, keys.shape[1] if keys.ndim == 2 else 0), dtype=np.uint8)
    unique, counts = np.unique(keys, axis=0, return_counts=True)
    digests = np.fromiter(
        (C.stable_digest_int(row.tobytes()) for row in unique),
        dtype=np.uint64,
        count=int(unique.shape[0]),
    )
    buckets = (digests % C.N_BUCKETS).astype(np.int64)
    vec = np.bincount(buckets, weights=counts, minlength=C.N_BUCKETS).astype(np.float32)
    n_unique = int(unique.shape[0])
    n_nonzero = int(np.count_nonzero(vec))
    n_unique_digests = int(np.unique(digests).size) if digest_audit else n_unique
    stats = {
        "n_paths": int(keys.shape[0]),
        "n_unique_patterns": n_unique,
        "n_nonzero_buckets": n_nonzero,
        "n_unique_64bit_digests": n_unique_digests,
        "n_64bit_digest_collisions": int(n_unique - n_unique_digests),
        "bucket_collision_rate": float(max(n_unique - n_nonzero, 0) / max(n_unique, 1)),
        "max_bucket_count": float(vec.max()) if vec.size else 0.0,
    }
    return vec, stats, unique


def global_key_audit(unique_parts: Iterable[np.ndarray]) -> dict[str, Any]:
    parts = [np.asarray(p, dtype=np.uint8) for p in unique_parts if np.asarray(p).size]
    total_rows = int(sum(int(p.shape[0]) for p in parts))
    if not parts:
        return {
            "n_unique_patterns": 0,
            "n_nonzero_buckets": 0,
            "n_unique_64bit_digests": 0,
            "n_64bit_digest_collisions": 0,
            "bucket_collision_rate": 0.0,
            "max_bucket_count": 0,
            "n_unique_rows_before_merge": 0,
        }
    all_unique = np.unique(np.concatenate(parts, axis=0), axis=0)
    digests = np.fromiter(
        (C.stable_digest_int(row.tobytes()) for row in all_unique),
        dtype=np.uint64,
        count=int(all_unique.shape[0]),
    )
    buckets = (digests % C.N_BUCKETS).astype(np.int64)
    bucket_counts = np.bincount(buckets, minlength=C.N_BUCKETS)
    n_unique = int(all_unique.shape[0])
    n_nonzero = int(np.count_nonzero(bucket_counts))
    n_unique_digests = int(np.unique(digests).size)
    return {
        "n_unique_patterns": n_unique,
        "n_nonzero_buckets": n_nonzero,
        "n_unique_64bit_digests": n_unique_digests,
        "n_64bit_digest_collisions": int(n_unique - n_unique_digests),
        "bucket_collision_rate": float(max(n_unique - n_nonzero, 0) / max(n_unique, 1)),
        "max_bucket_count": int(bucket_counts.max()) if bucket_counts.size else 0,
        "n_unique_rows_before_merge": total_rows,
    }


# ---------------------------------------------------------------------------
# One-graph feature construction
# ---------------------------------------------------------------------------


def graph_features(graph: dict, *, digests: bool = True) -> dict[str, Any]:
    atom_raw = np.asarray(graph["atom_raw"], dtype=np.int64)
    bond_raw = np.asarray(graph["bond_raw"], dtype=np.int64)
    bond_index = np.asarray(graph["bond_index"], dtype=np.int64)
    paths = enumerate_paths(atom_raw, bond_raw, bond_index)
    n_paths = int(paths["n_paths"])
    if n_paths != int(graph.get("n_paths_cache", n_paths)):
        raise ValueError(f"graph {graph.get('gid')}: path count {n_paths} != cache {graph.get('n_paths_cache')}")

    tkeys = type_keys(atom_raw, bond_raw, paths)
    fkeys_raw = full_keys(atom_raw, bond_raw, paths)
    order = canonical_path_order(fkeys_raw)
    fkeys = fkeys_raw[order] if n_paths else fkeys_raw
    first, second = split_halves(fkeys)
    seed = canonical_full_seed(fkeys)
    nkeys, perm = null_keys(first, second, seed)

    vec_type, st_type, uniq_type = count_unique_keys(tkeys, digest_audit=digests)
    vec_full, st_full, uniq_full = count_unique_keys(fkeys, digest_audit=digests)
    vec_null, st_null, uniq_null = count_unique_keys(nkeys, digest_audit=digests)

    marginal_first_equal = bool(np.array_equal(first, nkeys[:, :21]))
    marginal_second_equal = bool(np.array_equal(np.sort(second, axis=0), np.sort(second[perm], axis=0)))
    joint_equal_rows = int(np.all(fkeys == nkeys, axis=1).sum()) if n_paths else 0
    diff_buckets = int(np.count_nonzero(vec_full != vec_null))
    return {
        "gid": int(graph.get("gid", -1)),
        "n_atoms": int(atom_raw.shape[0]),
        "n_bonds": int(bond_index.shape[0]),
        "n_paths": n_paths,
        "per_root_sha256": C.sha256_array(paths["per_root"]),
        "counts_type": vec_type,
        "counts_full": vec_full,
        "counts_null": vec_null,
        "stats": {"type": st_type, "full": st_full, "null": st_null},
        "unique": {"type": uniq_type, "full": uniq_full, "null": uniq_null},
        "null_audit": {
            "graph_seed": int(seed),
            "canonical_path_order": "lexicographic order of the 33-byte FULL key",
            "permutation_derangement": bool(n_paths <= 1 or np.all(perm != np.arange(n_paths))),
            "marginal_second_equal": marginal_second_equal,
            "marginal_first_equal": marginal_first_equal,
            "joint_equal_rows": joint_equal_rows,
            "n_diff_buckets_full_vs_null": diff_buckets,
            "full_vs_null_bucket_vectors_equal": bool(diff_buckets == 0),
        },
    }


# ---------------------------------------------------------------------------
# Split-level construction
# ---------------------------------------------------------------------------


def build_split_features(split: str, *, limit: int | None = None, verbose: bool = True) -> dict[str, Any]:
    import time

    from pc_data import cache_graphs, load_cache_split

    if split not in C.ALLOWED_SPLITS:
        raise ValueError(f"split {split!r} is not allowed")
    t0 = time.perf_counter()
    cache = load_cache_split(split)
    graphs = cache_graphs(cache)
    n_graphs = len(graphs) if limit is None else min(int(limit), len(graphs))
    counts_type = np.zeros((n_graphs, C.N_BUCKETS), dtype=np.float32)
    counts_full = np.zeros((n_graphs, C.N_BUCKETS), dtype=np.float32)
    counts_null = np.zeros((n_graphs, C.N_BUCKETS), dtype=np.float32)
    n_paths = np.zeros(n_graphs, dtype=np.int64)
    n_atoms = np.zeros(n_graphs, dtype=np.int64)
    n_bonds = np.zeros(n_graphs, dtype=np.int64)
    gids = np.zeros(n_graphs, dtype=np.int64)
    uniq = {"type": [], "full": [], "null": []}
    null_rows_equal = 0
    null_vectors_equal = 0
    null_marginal_fail = 0
    null_derangement_fail = 0
    stats_rows: list[dict[str, Any]] = []
    for i in range(n_graphs):
        gf = graph_features(graphs[i])
        counts_type[i] = gf["counts_type"]
        counts_full[i] = gf["counts_full"]
        counts_null[i] = gf["counts_null"]
        n_paths[i] = gf["n_paths"]
        n_atoms[i] = gf["n_atoms"]
        n_bonds[i] = gf["n_bonds"]
        gids[i] = gf["gid"]
        for kind in uniq:
            uniq[kind].append(gf["unique"][kind])
        na = gf["null_audit"]
        null_rows_equal += int(na["joint_equal_rows"])
        null_vectors_equal += int(na["full_vs_null_bucket_vectors_equal"])
        null_marginal_fail += int(not na["marginal_second_equal"])
        null_derangement_fail += int(not na["permutation_derangement"])
        stats_rows.append(
            {
                "gid": gf["gid"],
                "n_atoms": gf["n_atoms"],
                "n_bonds": gf["n_bonds"],
                "n_paths": gf["n_paths"],
                **{f"{kind}_unique": gf["stats"][kind]["n_unique_patterns"] for kind in stats_rows_kinds()},
                **{f"{kind}_buckets": gf["stats"][kind]["n_nonzero_buckets"] for kind in stats_rows_kinds()},
                "full_vs_null_diff_buckets": na["n_diff_buckets_full_vs_null"],
            }
        )
        if verbose and (i + 1) % 1000 == 0:
            C.log(f"[features {split}] {i + 1}/{n_graphs} graphs {time.perf_counter() - t0:.1f}s")

    global_stats = {kind: global_key_audit(uniq[kind]) for kind in uniq}
    binary_nonzero = np.count_nonzero(counts_full > 0, axis=1)
    stats = {
        "round_id": C.ROUND_ID,
        "split": split,
        "n_graphs": int(n_graphs),
        "n_atoms_total": int(n_atoms.sum()),
        "n_bonds_total": int(n_bonds.sum()),
        "n_paths_total": int(n_paths.sum()),
        "n_paths_per_graph": C.distribution(n_paths.tolist()),
        "n_paths_zero_graphs": int(np.count_nonzero(n_paths == 0)),
        "features": {
            kind: {
                "n_buckets": C.N_BUCKETS,
                "global": global_stats[kind],
                "per_graph_nonzero_buckets": C.distribution(
                    np.count_nonzero(counts_type if kind == "type" else counts_full if kind == "full" else counts_null, axis=1).tolist()
                ),
                "per_graph_unique_patterns": C.distribution(
                    [row[f"{kind}_unique"] for row in stats_rows]
                ),
            }
            for kind in ("type", "full", "null")
        },
        "full_binary": {
            "definition": "FULL-S2 counts > 0",
            "per_graph_nonzero_buckets": C.distribution(binary_nonzero.tolist()),
            "global_nonzero_buckets": int(np.count_nonzero((counts_full > 0).any(axis=0))),
        },
        "null_audit": {
            "n_paths_total_null": int(n_paths.sum()),
            "n_graphs_full_vs_null_bucket_vectors_equal": int(null_vectors_equal),
            "n_graphs_with_second_half_marginal_failure": int(null_marginal_fail),
            "n_graphs_with_derangement_failure": int(null_derangement_fail),
            "joint_equal_rows_total": int(null_rows_equal),
            "derangement_rule": "fixed per-graph PCG64 permutation of second halves; resample until no fixed point; identity only when n_paths<=1; fallback cyclic shift",
            "marginal_rule": "first halves unchanged; second halves are a bijective permutation of the same multiset",
        },
        "gids_sha256": C.sha256_array(gids),
        "counts_type_sha256": C.sha256_array(counts_type),
        "counts_full_sha256": C.sha256_array(counts_full),
        "counts_null_sha256": C.sha256_array(counts_null),
        "n_paths_sha256": C.sha256_array(n_paths),
        "labels_used": False,
        "seconds": float(time.perf_counter() - t0),
        "stats_rows": stats_rows,
        "arrays": {
            "counts_type": counts_type,
            "counts_full": counts_full,
            "counts_null": counts_null,
            "n_paths": n_paths,
            "n_atoms": n_atoms,
            "n_bonds": n_bonds,
            "gids": gids,
            "full_binary": (counts_full > 0).astype(np.float32),
        },
    }
    return stats


def stats_rows_kinds() -> tuple[str, ...]:
    return ("type", "full", "null")


def save_split_features(split: str, stats: dict[str, Any], npz_path, json_path) -> dict[str, Any]:
    import os

    arrays = stats.pop("arrays")
    os.makedirs(os.path.dirname(str(npz_path)), exist_ok=True)
    np.savez_compressed(str(npz_path), **arrays)
    stats["npz_path"] = C.track_rel(npz_path)
    stats["npz_sha256"] = C.sha256_file(npz_path)
    stats["npz_bytes"] = int(os.path.getsize(npz_path))
    C.write_json(json_path, stats)
    stats["json_path"] = C.track_rel(json_path)
    stats["json_sha256"] = C.sha256_file(json_path)
    return stats


__all__ = [
    "build_split_features",
    "count_unique_keys",
    "enumerate_paths",
    "full_keys",
    "global_key_audit",
    "graph_content_seed",
    "graph_features",
    "null_keys",
    "null_permutation",
    "save_split_features",
    "split_halves",
    "type_keys",
]

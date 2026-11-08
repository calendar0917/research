"""Label-free shortest-bond-path distance cache and the RC collate function.

The six relation buckets of this round are defined by the *shortest path
distance in the original chemical bond graph*, not by any learned quantity, so
they can be precomputed once for every molecule of the official split.  The
build reads **structure only** (``atom_ptr`` / ``bond_ptr`` / ``bond_index``);
it never opens the label artifact and therefore cannot leak an official Test
target.  Per-molecule distances come from one BFS all-pairs call
(``scipy.sparse.csgraph.shortest_path(..., unweighted=True, directed=False)``),
never from a per-batch RDKit / NetworkX / SciPy call inside the training loop.

Storage: a single packed ``uint8`` array of the strict upper triangle of each
molecule's distance-bucket matrix, concatenated in official row order, plus a
``int64`` row pointer.  152.7 MB for all 437,929 OGB MolPCBA graphs: every
unordered atom pair occupies exactly one byte, so the representation is exact
(no sampling, no truncation) and cheap to slice per batch.

Bucket code 0..5 = the six buckets, 255 = different connected component (no
finite distance); self pairs are not stored at all (the diagonal is excluded by
construction).
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import rc_common as C


# ---------------------------------------------------------------------------
# bucket coding
# ---------------------------------------------------------------------------


def bucket_edges() -> list[tuple[str, int, int | None]]:
    return [tuple(b) for b in C.BUCKETS]  # type: ignore[misc]


def distance_to_code(distance: np.ndarray) -> np.ndarray:
    """Map a finite/inf distance matrix entry to a bucket code (255 = inf)."""
    d = np.asarray(distance, dtype=np.float64)
    out = np.full(d.shape, C.UNREACHABLE, dtype=np.uint8)
    finite = np.isfinite(d)
    if not finite.any():
        return out
    v = d[finite]
    if np.any(v < 1.0):
        raise RuntimeError("a pair of distinct atoms has distance < 1")
    if np.any(v != np.round(v)):
        raise RuntimeError("unweighted shortest path produced a non-integer distance")
    vi = np.round(v).astype(np.int64)
    code = np.zeros(vi.shape, dtype=np.uint8)
    covered = np.zeros(vi.shape, dtype=bool)
    for index, (_label, lo, hi) in enumerate(C.BUCKETS):
        if hi is None:
            sel = vi >= int(lo)
        else:
            sel = (vi >= int(lo)) & (vi <= int(hi))
        if np.any(sel & covered):
            raise RuntimeError("distance buckets overlap")
        code[sel] = np.uint8(index)
        covered |= sel
    if not bool(covered.all()):
        raise RuntimeError("distance buckets do not cover every finite distance")
    out[finite] = code
    return out


def code_to_distance_bounds() -> list[dict[str, Any]]:
    return [
        {"index": i, "label": label, "lo": int(lo), "hi": None if hi is None else int(hi)}
        for i, (label, lo, hi) in enumerate(C.BUCKETS)
    ]


# ---------------------------------------------------------------------------
# brute-force reference (used by the tests and the cache self-check)
# ---------------------------------------------------------------------------


def brute_force_distances(n_atoms: int, edges: np.ndarray) -> np.ndarray:
    """Plain BFS from every atom; ``inf`` when unreachable.  Structure only."""
    n = int(n_atoms)
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        adj[int(u)].append(int(v))
        adj[int(v)].append(int(u))
    dist = np.full((n, n), np.inf, dtype=np.float64)
    for src in range(n):
        dist[src, src] = 0.0
        frontier = [src]
        d = 0
        while frontier:
            d += 1
            nxt: list[int] = []
            for u in frontier:
                for w in adj[u]:
                    if not np.isfinite(dist[src, w]):
                        dist[src, w] = d
                        nxt.append(w)
            frontier = nxt
    return dist


def molecule_bucket_codes(n_atoms: int, edges: np.ndarray) -> np.ndarray:
    """Upper-triangle bucket codes of one molecule in row-major ``i<j`` order."""
    n = int(n_atoms)
    if n < 2:
        return np.zeros(0, dtype=np.uint8)
    import scipy.sparse as sp
    from scipy.sparse.csgraph import shortest_path

    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    if e.size:
        rows = np.concatenate([e[:, 0], e[:, 1]])
        cols = np.concatenate([e[:, 1], e[:, 0]])
        graph = sp.csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n, n))
    else:  # pragma: no cover - a molecule with one atom cannot reach this branch
        graph = sp.csr_matrix((n, n), dtype=np.int8)
    dist = shortest_path(graph, method="D", unweighted=True, directed=False)
    iu = np.triu_indices(n, 1)
    return distance_to_code(dist[iu])


def pair_counts_from_atoms(n_atoms: np.ndarray) -> np.ndarray:
    """Unordered distinct-pair count ``n(n-1)/2`` per molecule."""
    n = np.asarray(n_atoms, dtype=np.int64)
    return (n * (n - 1)) // 2


# ---------------------------------------------------------------------------
# cache build (structure only)
# ---------------------------------------------------------------------------


def build_cache(
    out_dir: Path | str = C.DIST_CACHE_DIR,
    *,
    cache_path: Path | str | None = None,
    rows: Sequence[int] | None = None,
    verify_fraction: float = 0.002,
    verify_seed: int = 20261109,
) -> dict[str, Any]:
    """Build the packed distance-bucket cache for the requested rows.

    Only ``atom_ptr``, ``bond_ptr`` and ``bond_index`` are read; the label
    artifact is never opened in this function.
    """
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_path is None:
        import lr_common as LC  # inherited frozen path holder

        cache_path = Path(LC.CACHE_PATH)
    cache_path = Path(cache_path)
    atom_ptr = np.load(cache_path / "atom_ptr.npy")
    bond_ptr = np.load(cache_path / "bond_ptr.npy")
    bond_index = np.load(cache_path / "bond_index.npy", mmap_mode="r")
    n_molecules = int(atom_ptr.size - 1)
    n_atoms = np.diff(atom_ptr).astype(np.int64)
    if n_atoms.size != n_molecules:
        raise RuntimeError("atom_ptr does not describe the molecule count")

    if rows is None:
        row_ids = np.arange(n_molecules, dtype=np.int64)
    else:
        row_ids = np.asarray(rows, dtype=np.int64)
    counts = pair_counts_from_atoms(n_atoms[row_ids])
    ptr = np.zeros(row_ids.size + 1, dtype=np.int64)
    np.cumsum(counts, out=ptr[1:])
    packed = np.zeros(int(ptr[-1]), dtype=np.uint8)

    bucket_pair_totals = np.zeros(C.N_BUCKETS, dtype=np.int64)
    unreachable_pairs = 0
    for k, row in enumerate(row_ids):
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        edges = np.asarray(bond_index[b0:b1], dtype=np.int64)
        if edges.size:
            if int(edges.min()) < 0 or int(edges.max()) >= a1 - a0:
                raise RuntimeError(f"row {row}: bond_index outside the molecule")
            if np.any(edges[:, 0] == edges[:, 1]):
                raise RuntimeError(f"row {row}: self loop in bond_index")
        codes = molecule_bucket_codes(a1 - a0, edges)
        if codes.size != int(counts[k]):
            raise RuntimeError(f"row {row}: {codes.size} pair codes != {int(counts[k])}")
        packed[ptr[k] : ptr[k + 1]] = codes
        finite = (codes != C.UNREACHABLE).sum()
        unreachable_pairs += int(codes.size - finite)
        for b in range(C.N_BUCKETS):
            bucket_pair_totals[b] += int((codes == b).sum())

    # self-check on a deterministic sample: brute-force BFS must agree exactly
    gen = np.random.default_rng(int(verify_seed))
    n_verify = max(1, int(round(float(verify_fraction) * row_ids.size)))
    sample = np.sort(gen.choice(row_ids.size, size=min(n_verify, row_ids.size), replace=False))
    mismatches = 0
    for k in sample:
        row = int(row_ids[k])
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        edges = np.asarray(bond_index[b0:b1], dtype=np.int64)
        n = a1 - a0
        ref = brute_force_distances(n, edges)
        iu = np.triu_indices(n, 1)
        ref_codes = distance_to_code(ref[iu])
        mismatch = int((ref_codes != packed[ptr[k] : ptr[k + 1]]).sum())
        mismatches += mismatch
        # connectivity must agree with the sentinel code
        reachable_ref = np.isfinite(ref[iu])
        reachable_cache = packed[ptr[k] : ptr[k + 1]] != C.UNREACHABLE
        if not np.array_equal(reachable_ref, reachable_cache):
            mismatches += 1
    if mismatches:
        raise RuntimeError(f"distance cache disagrees with brute force on {mismatches} entries")

    np.save(out_dir / "pair_bucket.npy", packed)
    np.save(out_dir / "pair_ptr.npy", ptr)
    meta = {
        "schema": C.SCHEMA_DIST_CACHE,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "purpose": "shortest-bond-path distance buckets for the RC relation branch",
        "labels_read": False,
        "label_artifact_opened": False,
        "training_scope": False,
        "rows": "all official rows" if rows is None else f"{row_ids.size} requested rows",
        "n_rows": int(row_ids.size),
        "row_ids_sha256": C.sha256_bytes(np.ascontiguousarray(row_ids, dtype=np.int64).tobytes()),
        "n_atoms_total": int(n_atoms[row_ids].sum()),
        "n_unordered_pairs": int(ptr[-1]),
        "n_ordered_pairs": int(2 * ptr[-1]),
        "bucket_pair_totals": [int(x) for x in bucket_pair_totals],
        "bucket_ordered_pair_totals": [int(2 * x) for x in bucket_pair_totals],
        "unreachable_unordered_pairs": int(unreachable_pairs),
        "bucket_edges": code_to_distance_bounds(),
        "distance_kind": "unweighted shortest path over the original bond graph (scipy BFS)",
        "storage": {
            "pair_bucket": "uint8 packed strict upper triangle, row-major i<j",
            "pair_ptr": "int64 offsets (n_rows + 1)",
        },
        "pair_bucket_sha256": C.sha256_file(out_dir / "pair_bucket.npy"),
        "pair_ptr_sha256": C.sha256_file(out_dir / "pair_ptr.npy"),
        "build_seconds": float(time.perf_counter() - t0),
        "verification": {
            "fraction": float(verify_fraction),
            "n_rows": int(sample.size),
            "seed": int(verify_seed),
            "method": "independent pure-python BFS + identical bucket mapping",
            "mismatches": int(mismatches),
            "passed": True,
        },
    }
    C.write_json(out_dir / "meta.json", meta)
    C.log(
        f"[dist-cache] rows={meta['n_rows']} pairs={meta['n_unordered_pairs']} "
        f"seconds={meta['build_seconds']:.1f} verified={sample.size} mismatches=0"
    )
    return meta


def load_cache(path: Path | str = C.DIST_CACHE_DIR) -> dict[str, Any]:
    """Load the packed bucket cache and re-verify its recorded hashes."""
    path = Path(path)
    meta_path = path / "meta.json"
    if not meta_path.is_file():
        raise RuntimeError(f"distance cache missing: {meta_path}")
    meta = C.read_json(meta_path)
    if meta.get("schema") != C.SCHEMA_DIST_CACHE:
        raise RuntimeError(f"distance cache schema {meta.get('schema')}")
    if meta.get("labels_read") is not False:
        raise RuntimeError("distance cache claims labels were read")
    bucket = np.load(path / "pair_bucket.npy")
    ptr = np.load(path / "pair_ptr.npy")
    for name, arr in (("pair_bucket", bucket), ("pair_ptr", ptr)):
        if C.sha256_file(path / f"{name}.npy") != meta[f"{name}_sha256"]:
            raise RuntimeError(f"distance cache {name} hash mismatch")
    if bucket.dtype != np.uint8 or ptr.dtype != np.int64:
        raise RuntimeError("distance cache dtypes drifted")
    if int(ptr[-1]) != int(bucket.size):
        raise RuntimeError("distance cache pointer does not span the packed array")
    if int(meta["n_unordered_pairs"]) != int(bucket.size):
        raise RuntimeError("distance cache size does not match its manifest")
    return {
        "bucket": bucket,
        "ptr": ptr,
        "pair_counts": np.diff(ptr),
        "meta": meta,
        "dir": C.track_rel(path),
    }


def cache_identity(dist: Mapping[str, Any]) -> dict[str, Any]:
    meta = dist["meta"]
    return {
        "dir": dist.get("dir"),
        "pair_bucket_sha256": meta["pair_bucket_sha256"],
        "pair_ptr_sha256": meta["pair_ptr_sha256"],
        "n_rows": int(meta["n_rows"]),
        "n_unordered_pairs": int(meta["n_unordered_pairs"]),
        "bucket_pair_totals": list(meta["bucket_pair_totals"]),
        "labels_read": bool(meta["labels_read"]),
        "row_ids_sha256": meta["row_ids_sha256"],
        "build_seconds": float(meta["build_seconds"]),
    }


# ---------------------------------------------------------------------------
# batch assembly
# ---------------------------------------------------------------------------


def gather_pairs(
    dist: Mapping[str, Any], rows: Sequence[int]
) -> tuple[np.ndarray, np.ndarray]:
    """Packed bucket codes and pair counts for a batch of molecule rows."""
    ptr = dist["ptr"]
    counts = dist["pair_counts"]
    rows = np.asarray(rows, dtype=np.int64)
    parts = [dist["bucket"][int(ptr[r]) : int(ptr[r + 1])] for r in rows]
    packed = np.concatenate(parts) if parts else np.zeros(0, dtype=np.uint8)
    return packed, np.asarray(counts[rows], dtype=np.int64)


def collate_rel(task: Any, dist: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    """Inherited collate + the packed distance buckets of the same rows."""
    import torch

    import bs_common as _BC

    batch = _BC.PC.collate(task.cache, rows)
    packed, counts = gather_pairs(dist, rows)
    if int(counts.sum()) != int(packed.size):
        raise RuntimeError(f"pair counts {int(counts.sum())} != packed pairs {int(packed.size)}")
    batch["pair_bucket"] = torch.from_numpy(np.ascontiguousarray(packed, dtype=np.uint8))
    batch["pair_counts"] = torch.from_numpy(np.ascontiguousarray(counts, dtype=np.int64))
    return batch


# ---------------------------------------------------------------------------
# gated task loading (identical gates to the inherited rounds)
# ---------------------------------------------------------------------------


def load_task_gated() -> Any:
    task = C.BC.SD.load_task()
    if task.labels.shape != (C.N_MOLECULES, C.N_TASKS):
        raise RuntimeError(f"label artifact shape {task.labels.shape}")
    if np.isfinite(task.labels[task.test_rows]).any():
        raise RuntimeError("official Test labels are exposed in the training artifact")
    checks = {
        "feature_spec_hash": task.spec.hash() == C.SOURCE["feature_spec_hash"],
        "norm_fingerprint": task.norm["norm_fingerprint"] == C.SOURCE["norm_fingerprint"],
        "split_partition_sha256": task.split_partition_sha256 == C.SOURCE["split_partition_sha256"],
        "label_artifact_sha256": task.label_artifact_sha256 == C.SOURCE["label_artifact_sha256"],
    }
    failed = [k for k, v in checks.items() if not v]
    if failed:
        raise RuntimeError(f"frozen data identity drifted: {failed}")
    return task


def task_counts(task: Any) -> dict[str, Any]:
    return C.BC.SD.task_counts(task)


__all__ = [
    "brute_force_distances",
    "build_cache",
    "cache_identity",
    "code_to_distance_bounds",
    "collate_rel",
    "distance_to_code",
    "gather_pairs",
    "load_cache",
    "load_task_gated",
    "molecule_bucket_codes",
    "pair_counts_from_atoms",
    "task_counts",
]

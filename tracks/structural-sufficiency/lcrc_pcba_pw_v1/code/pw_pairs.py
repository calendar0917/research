"""Offline pair index: every legal unordered atom pair of every molecule.

The PW branch needs, for each molecule, the list of unordered pairs
``i < j`` that lie in the *same connected component* of the original chemical
bond graph, together with the shortest-bond-path distance bucket of that pair.
That relation is a property of the structure alone, so it is computed **once**,
offline, over the official row order and stored as three packed arrays:

    pair_i.npy       int16   local atom index i (0-based, inside the molecule)
    pair_j.npy       int16   local atom index j (> i)
    pair_bucket.npy  uint8   bucket code 0..5  (see ``pw_common.BUCKETS``)
    pair_ptr.npy     int64   offsets into the three arrays (n_rows + 1)

Pairs of different connected components are *not* stored (they have no finite
shortest-path distance and are not legal PW pairs); comparing
``n_unordered_pairs_stored`` with ``n(n-1)/2`` therefore makes the dropped set
explicit.  The build reads ``atom_ptr`` / ``bond_ptr`` / ``bond_index`` only --
the label artifact is never opened, so no official Test target can leak into
the geometry.  Distances come from one all-pairs BFS per molecule
(``scipy.sparse.csgraph.shortest_path(unweighted=True, directed=False)``); the
training loop never calls BFS, RDKit, NetworkX or SciPy.

The build also verifies itself twice:

* an independent pure-Python BFS re-derives the bucket of a deterministic
  row sample (not the same code path as the scipy build), and
* when the RC round's distance cache is present, its packed bucket array
  (hash-verified against its own manifest) is compared pair-by-pair with the
  PW index over a second sample: same pairs, same buckets.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import pw_common as C

I_DTYPE = np.int16  # MolPCBA max atoms per molecule = 332 (checked at build)
B_DTYPE = np.uint8


# ---------------------------------------------------------------------------
# distances and buckets
# ---------------------------------------------------------------------------


def bucket_of_distance(distance: float) -> int | None:
    """Bucket index of a finite distance; ``None`` for infinity."""
    d = float(distance)
    if not np.isfinite(d):
        return None
    if d < 1.0:
        raise RuntimeError("a pair of distinct atoms has distance < 1")
    if d != round(d):
        raise RuntimeError("unweighted shortest path produced a non-integer distance")
    di = int(round(d))
    for index, (_label, lo, hi) in enumerate(C.BUCKETS):
        if hi is None:
            if di >= int(lo):
                return index
        elif int(lo) <= di <= int(hi):
            return index
    raise RuntimeError(f"distance {d} is not covered by the bucket definitions")


def code_array(dist: np.ndarray) -> np.ndarray:
    """Bucket codes of a full distance matrix cell set; 255 = unreachable."""
    values = np.asarray(dist, dtype=np.float64)
    out = np.full(values.shape, C.UNREACHABLE, dtype=np.uint8)
    finite = np.isfinite(values)
    if not finite.any():
        return out
    v = values[finite]
    if np.any(v < 1.0):
        raise RuntimeError("a pair of distinct atoms has distance < 1")
    if np.any(v != np.round(v)):
        raise RuntimeError("unweighted shortest path produced a non-integer distance")
    vi = np.round(v).astype(np.int64)
    codes = np.zeros(vi.shape, dtype=np.uint8)
    covered = np.zeros(vi.shape, dtype=bool)
    for index, (_label, lo, hi) in enumerate(C.BUCKETS):
        sel = (vi >= int(lo)) & ((vi <= int(hi)) if hi is not None else True)
        if np.any(sel & covered):
            raise RuntimeError("distance buckets overlap")
        codes[sel] = np.uint8(index)
        covered |= sel
    if not bool(covered.all()):
        raise RuntimeError("distance buckets do not cover every finite distance")
    out[finite] = codes
    return out


def brute_force_distances(n_atoms: int, edges: np.ndarray) -> np.ndarray:
    """Plain BFS from every atom; ``inf`` when unreachable.  Structure only."""
    n = int(n_atoms)
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in np.asarray(edges, dtype=np.int64).reshape(-1, 2):
        if int(u) == int(v):
            raise RuntimeError("self loop in the bond graph")
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


def molecule_pairs(n_atoms: int, edges: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``(i, j, bucket)`` of the legal same-component pairs of one molecule."""
    n = int(n_atoms)
    if n < 2:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), np.zeros(0, dtype=np.uint8)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    import scipy.sparse as sp
    from scipy.sparse.csgraph import shortest_path

    if e.size:
        rows = np.concatenate([e[:, 0], e[:, 1]])
        cols = np.concatenate([e[:, 1], e[:, 0]])
        graph = sp.csr_matrix((np.ones(rows.size, dtype=np.int8), (rows, cols)), shape=(n, n))
    else:
        graph = sp.csr_matrix((n, n), dtype=np.int8)
    dist = shortest_path(graph, method="D", unweighted=True, directed=False)
    iu = np.triu_indices(n, 1)
    codes = code_array(dist[iu])
    keep = codes != C.UNREACHABLE
    return iu[0][keep], iu[1][keep], codes[keep]


# ---------------------------------------------------------------------------
# build
# ---------------------------------------------------------------------------


def _structure_paths() -> tuple[Path, Path, Path]:
    import lr_common as LC  # inherited frozen path holder

    base = Path(LC.CACHE_PATH)
    return base / "atom_ptr.npy", base / "bond_ptr.npy", base / "bond_index.npy"


def build_cache(
    out_dir: Path | str = C.PAIR_CACHE_DIR,
    *,
    rows: Sequence[int] | None = None,
    verify_fraction: float = 0.002,
    verify_seed: int = 20261111,
    rc_crosscheck_rows: int = 2000,
) -> dict[str, Any]:
    """Build the packed legal-pair index (structure only).

    ``rows=None`` builds the production cache for every official row.  A
    subset build keeps the *full-length* row pointer (unselected rows simply
    carry zero pairs) so the cache can still be indexed by official row id;
    it exists for tests and for a CPU-only smoke run, and the trainer refuses
    to use it.
    """
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    atom_path, bond_path, index_path = _structure_paths()
    atom_ptr = np.load(atom_path)
    bond_ptr = np.load(bond_path)
    bond_index = np.load(index_path, mmap_mode="r")
    n_atoms = np.diff(atom_ptr).astype(np.int64)
    n_rows = int(n_atoms.size)
    max_atoms = int(n_atoms.max())
    if max_atoms > np.iinfo(I_DTYPE).max:
        raise RuntimeError(f"{max_atoms} atoms per molecule does not fit {I_DTYPE}")

    if rows is None:
        row_ids = np.arange(n_rows, dtype=np.int64)
        scope = "all official rows"
    else:
        row_ids = np.unique(np.asarray(rows, dtype=np.int64))
        if row_ids.size == 0 or int(row_ids.min()) < 0 or int(row_ids.max()) >= n_rows:
            raise RuntimeError("requested rows outside the official row order")
        scope = f"{row_ids.size} selected rows (test/smoke cache)"
    selected = np.zeros(n_rows, dtype=bool)
    selected[row_ids] = True

    pair_counts = np.zeros(n_rows, dtype=np.int64)
    parts_i: list[np.ndarray] = []
    parts_j: list[np.ndarray] = []
    parts_b: list[np.ndarray] = []
    bucket_totals = np.zeros(C.N_BUCKETS, dtype=np.int64)
    dropped_cross_component = 0
    for row in row_ids:
        row = int(row)
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        edges = np.asarray(bond_index[b0:b1], dtype=np.int64)
        n = a1 - a0
        if edges.size:
            if int(edges.min()) < 0 or int(edges.max()) >= n:
                raise RuntimeError(f"row {row}: bond_index outside the molecule")
            if np.any(edges[:, 0] == edges[:, 1]):
                raise RuntimeError(f"row {row}: self loop in bond_index")
        i, j, b = molecule_pairs(n, edges)
        if i.size != b.size:
            raise RuntimeError(f"row {row}: pair index/bucket length mismatch")
        if i.size and (int(i.min()) < 0 or int(j.max()) >= n or np.any(i >= j)):
            raise RuntimeError(f"row {row}: pair index outside the strict upper triangle")
        pair_counts[row] = int(i.size)
        dropped_cross_component += int(n * (n - 1) // 2) - int(i.size)
        parts_i.append(i.astype(I_DTYPE))
        parts_j.append(j.astype(I_DTYPE))
        parts_b.append(b.astype(B_DTYPE))
        for index in range(C.N_BUCKETS):
            bucket_totals[index] += int((b == index).sum())

    pair_i = np.concatenate(parts_i) if parts_i else np.zeros(0, dtype=I_DTYPE)
    pair_j = np.concatenate(parts_j) if parts_j else np.zeros(0, dtype=I_DTYPE)
    pair_b = np.concatenate(parts_b) if parts_b else np.zeros(0, dtype=B_DTYPE)
    ptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(pair_counts, out=ptr[1:])
    if int(ptr[-1]) != int(pair_i.size):
        raise RuntimeError("pair pointer does not span the packed arrays")

    np.save(out_dir / "pair_i.npy", pair_i)
    np.save(out_dir / "pair_j.npy", pair_j)
    np.save(out_dir / "pair_bucket.npy", pair_b)
    np.save(out_dir / "pair_ptr.npy", ptr)

    # ---- verification 1: independent pure-python BFS on a row sample -------
    gen = np.random.default_rng(int(verify_seed))
    n_verify = max(1, int(round(float(verify_fraction) * row_ids.size)))
    sample = np.sort(gen.choice(row_ids, size=min(n_verify, row_ids.size), replace=False))
    mismatched_rows = 0
    checked_pairs = 0
    for row in sample:
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        edges = np.asarray(bond_index[b0:b1], dtype=np.int64)
        n = a1 - a0
        ref = brute_force_distances(n, edges)
        lo, hi = int(ptr[row]), int(ptr[row + 1])
        got = set(
            zip(
                pair_i[lo:hi].astype(np.int64).tolist(),
                pair_j[lo:hi].astype(np.int64).tolist(),
                pair_b[lo:hi].astype(np.int64).tolist(),
            )
        )
        expected = set()
        for u in range(n):
            for v in range(u + 1, n):
                code = bucket_of_distance(ref[u, v])
                if code is not None:
                    expected.add((u, v, code))
        checked_pairs += len(expected)
        if got != expected:
            mismatched_rows += 1
    if mismatched_rows:
        raise RuntimeError(f"pair index disagrees with BFS on {mismatched_rows} rows")

    # ---- verification 2: cross-check the RC round's independent cache ------
    crosscheck: dict[str, Any] = {"available": False}
    rc_meta_path = C.RCC.DIST_CACHE_META
    if rc_meta_path.is_file():
        rc_meta = C.read_json(rc_meta_path)
        rc_bucket = np.load(C.RCC.DIST_CACHE_DIR / "pair_bucket.npy", mmap_mode="r")
        rc_ptr = np.load(C.RCC.DIST_CACHE_DIR / "pair_ptr.npy")
        hashes_ok = (
            C.sha256_file(C.RCC.DIST_CACHE_DIR / "pair_bucket.npy") == rc_meta["pair_bucket_sha256"]
            and C.sha256_file(C.RCC.DIST_CACHE_DIR / "pair_ptr.npy") == rc_meta["pair_ptr_sha256"]
        )
        if not hashes_ok:
            raise RuntimeError("RC distance cache hash mismatch")
        gen2 = np.random.default_rng(int(verify_seed) + 1)
        n_cross = min(int(rc_crosscheck_rows), row_ids.size)
        rows2 = np.sort(gen2.choice(row_ids, size=n_cross, replace=False))
        bad = 0
        for row in rows2:
            r0, r1 = int(rc_ptr[row]), int(rc_ptr[row + 1])
            rc_codes = np.asarray(rc_bucket[r0:r1])
            keep = rc_codes != C.UNREACHABLE
            lo, hi = int(ptr[row]), int(ptr[row + 1])
            if int(keep.sum()) != hi - lo:
                bad += 1
                continue
            if not np.array_equal(rc_codes[keep], np.asarray(pair_b[lo:hi])):
                bad += 1
            # the surviving pairs must be exactly the triangular positions of
            # the RC cache that are not the 255 sentinel
            n = int(n_atoms[row])
            iu = np.triu_indices(n, 1)
            if not (
                np.array_equal(iu[0][keep].astype(np.int64), pair_i[lo:hi].astype(np.int64))
                and np.array_equal(iu[1][keep].astype(np.int64), pair_j[lo:hi].astype(np.int64))
            ):
                bad += 1
        if bad:
            raise RuntimeError(f"PW pair index disagrees with the RC distance cache on {bad} rows")
        crosscheck = {
            "available": True,
            "rows": int(rows2.size),
            "mismatched_rows": 0,
            "rc_pair_bucket_sha256": rc_meta["pair_bucket_sha256"],
            "rc_pair_ptr_sha256": rc_meta["pair_ptr_sha256"],
            "note": "independent build (PW: scipy BFS -> (i,j,bucket); RC: scipy BFS -> triangular codes)",
        }

    meta: dict[str, Any] = {
        "schema": C.SCHEMA_PAIR_INDEX,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "purpose": "legal (same connected component) unordered atom pairs + shortest-path distance bucket",
        "labels_read": False,
        "label_artifact_opened": False,
        "training_scope": False,
        "n_rows": n_rows,
        "rows": scope,
        "rows_subset": bool(rows is not None),
        "row_ids_sha256": C.sha256_bytes(np.ascontiguousarray(row_ids, dtype=np.int64).tobytes()),
        "n_rows_with_pairs": int((pair_counts > 0).sum()),
        "max_atoms_per_molecule": max_atoms,
        "index_dtype": np.dtype(I_DTYPE).name,
        "bucket_dtype": np.dtype(B_DTYPE).name,
        "n_pairs_stored": int(pair_i.size),
        "n_unordered_pairs_all": int(((n_atoms * (n_atoms - 1)) // 2).sum()),
        "n_pairs_dropped_cross_component": int(dropped_cross_component),
        "n_molecules_without_pairs": int((pair_counts == 0).sum()),
        "bucket_pair_totals": [int(x) for x in bucket_totals],
        "bucket_edges": [
            {"index": i, "label": label, "lo": int(lo), "hi": None if hi is None else int(hi)}
            for i, (label, lo, hi) in enumerate(C.BUCKETS)
        ],
        "distance_kind": "unweighted shortest path over the original bond graph (scipy BFS, once)",
        "storage": {
            "pair_i": "int16 local atom index i (strict upper triangle)",
            "pair_j": "int16 local atom index j > i",
            "pair_bucket": "uint8 bucket code 0..5",
            "pair_ptr": "int64 offsets (n_rows + 1)",
        },
        "hashes": {},
        "build_seconds": float(time.perf_counter() - t0),
        "verification": {
            "fraction": float(verify_fraction),
            "n_rows": int(sample.size),
            "seed": int(verify_seed),
            "checked_pairs": int(checked_pairs),
            "method": "independent pure-python BFS over every atom pair of the sampled rows",
            "mismatched_rows": 0,
            "passed": True,
        },
        "rc_crosscheck": crosscheck,
    }
    for name in ("pair_i", "pair_j", "pair_bucket", "pair_ptr"):
        meta["hashes"][name] = C.sha256_file(out_dir / f"{name}.npy")
    C.write_json(out_dir / "meta.json", meta)
    C.log(
        f"[pair-index] rows={n_rows} pairs={meta['n_pairs_stored']} "
        f"dropped_cross_component={meta['n_pairs_dropped_cross_component']} "
        f"seconds={meta['build_seconds']:.1f} verified={sample.size} mismatches=0"
    )
    return meta


# ---------------------------------------------------------------------------
# load
# ---------------------------------------------------------------------------


def load_cache(path: Path | str = C.PAIR_CACHE_DIR, *, require_full: bool = False) -> dict[str, Any]:
    path = Path(path)
    meta_path = path / "meta.json"
    if not meta_path.is_file():
        raise RuntimeError(f"pair index missing: {meta_path}")
    meta = C.read_json(meta_path)
    if meta.get("schema") != C.SCHEMA_PAIR_INDEX:
        raise RuntimeError(f"pair index schema {meta.get('schema')}")
    if meta.get("labels_read") is not False:
        raise RuntimeError("pair index claims labels were read")
    if require_full and bool(meta.get("rows_subset")):
        raise RuntimeError(f"pair index is a test/smoke subset ({meta.get('rows')})")
    arrays: dict[str, np.ndarray] = {}
    for name, dtype in (
        ("pair_i", I_DTYPE),
        ("pair_j", I_DTYPE),
        ("pair_bucket", B_DTYPE),
        ("pair_ptr", np.int64),
    ):
        arr = np.load(path / f"{name}.npy")
        if arr.dtype != dtype:
            raise RuntimeError(f"pair index {name} dtype {arr.dtype} != {dtype}")
        if C.sha256_file(path / f"{name}.npy") != meta["hashes"][name]:
            raise RuntimeError(f"pair index {name} hash mismatch")
        arrays[name] = arr
    ptr = arrays["pair_ptr"]
    if int(ptr[-1]) != int(arrays["pair_i"].size):
        raise RuntimeError("pair index pointer does not span the arrays")
    if int(meta["n_pairs_stored"]) != int(arrays["pair_i"].size):
        raise RuntimeError("pair index size does not match its manifest")
    return {
        "pair_i": arrays["pair_i"],
        "pair_j": arrays["pair_j"],
        "pair_bucket": arrays["pair_bucket"],
        "ptr": ptr,
        "pair_counts": np.diff(ptr),
        "meta": meta,
        "dir": C.track_rel(path),
    }


def cache_identity(dist: Mapping[str, Any]) -> dict[str, Any]:
    meta = dist["meta"]
    return {
        "dir": dist.get("dir"),
        "hashes": dict(meta["hashes"]),
        "n_rows": int(meta["n_rows"]),
        "rows": meta.get("rows"),
        "rows_subset": bool(meta.get("rows_subset")),
        "row_ids_sha256": meta.get("row_ids_sha256"),
        "n_pairs_stored": int(meta["n_pairs_stored"]),
        "n_pairs_dropped_cross_component": int(meta["n_pairs_dropped_cross_component"]),
        "bucket_pair_totals": list(meta["bucket_pair_totals"]),
        "labels_read": bool(meta["labels_read"]),
        "build_seconds": float(meta["build_seconds"]),
        "rc_crosscheck": dict(meta.get("rc_crosscheck") or {}),
    }


# ---------------------------------------------------------------------------
# batch assembly
# ---------------------------------------------------------------------------


def batch_pair_slices(
    dist: Mapping[str, Any], rows: Sequence[int]
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Gather the legal pairs of a batch of molecules.

    Returns ``(pair_i, pair_j, pair_bucket, pair_graph, pair_counts)`` where
    ``pair_i/pair_j`` are *local* atom indices inside their molecule and
    ``pair_graph`` is the position of the molecule inside the batch.  The
    gather is fully vectorised (no Python loop over pairs).
    """
    ptr = dist["ptr"]
    rows = np.asarray(rows, dtype=np.int64)
    counts = np.asarray(dist["pair_counts"][rows], dtype=np.int64)
    total = int(counts.sum())
    n_graphs = int(rows.size)
    if total == 0:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty.copy(), empty.copy(), empty.copy(), counts
    starts = np.repeat(ptr[rows], counts)
    # intra-row position: 0..counts-1 for every pair, restarting at each row
    intra = np.arange(total, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts)
    idx = starts + intra
    pair_graph = np.repeat(np.arange(n_graphs, dtype=np.int64), counts)
    return (
        np.asarray(dist["pair_i"][idx], dtype=np.int64),
        np.asarray(dist["pair_j"][idx], dtype=np.int64),
        np.asarray(dist["pair_bucket"][idx], dtype=np.int64),
        pair_graph,
        counts,
    )


def collate_pw(task: Any, dist: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    """Inherited collate + the precomputed legal-pair payload of the same rows."""
    import torch

    batch = C.BC.PC.collate(task.cache, rows)
    i, j, b, graph, counts = batch_pair_slices(dist, rows)
    if int(counts.sum()) != int(i.size):
        raise RuntimeError(f"pair counts {int(counts.sum())} != gathered pairs {int(i.size)}")
    if int(batch["n_graphs"]) != int(counts.size):
        raise RuntimeError("pair counts do not describe the batch")
    batch["pair_i"] = torch.from_numpy(np.ascontiguousarray(i))
    batch["pair_j"] = torch.from_numpy(np.ascontiguousarray(j))
    batch["pair_bucket"] = torch.from_numpy(np.ascontiguousarray(b))
    batch["pair_graph"] = torch.from_numpy(np.ascontiguousarray(graph))
    batch["pair_counts"] = torch.from_numpy(np.ascontiguousarray(counts))
    return batch


def load_task_gated() -> Any:
    """Inherited Train-only label gate (identical function object as W2/RC)."""
    return C.RD.load_task_gated()


def task_counts(task: Any) -> dict[str, Any]:
    return C.BC.SD.task_counts(task)


__all__ = [
    "B_DTYPE",
    "I_DTYPE",
    "batch_pair_slices",
    "brute_force_distances",
    "bucket_of_distance",
    "build_cache",
    "cache_identity",
    "code_array",
    "collate_pw",
    "load_cache",
    "load_task_gated",
    "molecule_pairs",
    "task_counts",
]

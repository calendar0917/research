"""Exact rooted two-bond path enumeration and the label-free CPC structure cache.

Definition (frozen, ``PROTOCOL.md`` §2.2)
----------------------------------------

For every root atom ``i`` in an undirected molecular graph enumerate every
**simple** two-bond path ``p = (i, j, k)`` with

    i-j and j-k real chemical bonds,
    i, j, k pairwise distinct,
    (i, j, k) rooted at i.

The pair ``(i, j, k)`` and its reversal ``(k, j, i)`` are different instances
because the updated root differs.  Every legal path is kept: no sampling, no
truncation, no endpoint deduplication.  Rings and triangles are kept.  Only real
chemical bonds are used; there are no virtual nodes or artificial long edges.

The cache stores static indices only and is memory-mapped.  It is built from the
official frozen PCBA feature cache: the per-atom pointer and endpoint arrays are
the frozen S2 arrays, and the middle atom and the two physical bond ids are
recovered from the original sorted physical-bond table.  The two endpoint
sequences must agree exactly (or, if the historical order ever differs, their
per-root multisets must agree).

Nothing here reads a label.  The structural cache covers all official rows,
including Test structure, but it contains no target value.
"""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import cpc_common as C

CPC_CACHE_DIR = C.RUNS / "cpc_cache"
CPC_CACHE_FIELDS = (
    "cpc_ptr",
    "cpc_middle",
    "cpc_end",
    "cpc_bond_ij",
    "cpc_bond_jk",
    "rows",
    "row_atom_ptr",
    "row_bond_ptr",
    "row_n_atoms",
    "row_n_bonds",
)


# ---------------------------------------------------------------------------
# graph primitives
# ---------------------------------------------------------------------------


def molecule_adjacency(
    n_atoms: int,
    bond_u: Sequence[int] | np.ndarray,
    bond_v: Sequence[int] | np.ndarray,
    bond_type: Sequence[int] | np.ndarray,
) -> list[list[tuple[int, int, int]]]:
    """Sorted adjacency ``(neighbour, one-based bond type, physical bond id)``.

    ``bond_u < bond_v`` and the list is sorted by ``(u, v)`` exactly as the
    frozen cache stores it, so the position in this list *is* the physical bond
    id used by ``bond_index``.
    """
    n = int(n_atoms)
    u = np.asarray(bond_u, dtype=np.int64).reshape(-1)
    v = np.asarray(bond_v, dtype=np.int64).reshape(-1)
    t = np.asarray(bond_type, dtype=np.int64).reshape(-1)
    if not (u.shape == v.shape == t.shape):
        raise ValueError("bond arrays must have the same length")
    if np.any(u < 0) or np.any(v < 0) or np.any(u >= n) or np.any(v >= n):
        raise ValueError("bond endpoint outside atom range")
    if np.any(u >= v):
        raise ValueError("physical bonds must satisfy u < v")
    if u.size > 1 and np.any(np.diff(u * (n + 1) + v) <= 0):
        raise ValueError("physical bonds must be sorted by (u, v)")
    adj: list[list[tuple[int, int, int]]] = [[] for _ in range(n)]
    for e, (a, b, bt) in enumerate(zip(u.tolist(), v.tolist(), t.tolist())):
        adj[a].append((int(b), int(bt), int(e)))
        adj[b].append((int(a), int(bt), int(e)))
    for row in adj:
        row.sort()
    for row in adj:
        if len({nb for nb, _bt, _e in row}) != len(row):
            raise ValueError("duplicate physical bond in neighbour list")
    return adj


def bond_id_lookup(adj: Sequence[Sequence[tuple[int, int, int]]], a: int, b: int) -> int:
    for nb, _bt, e in adj[int(a)]:
        if nb == int(b):
            return int(e)
    raise KeyError((a, b))


# ---------------------------------------------------------------------------
# enumeration
# ---------------------------------------------------------------------------


def enumerate_cpc_molecule(
    n_atoms: int,
    adj: Sequence[Sequence[tuple[int, int, int]]],
) -> dict[str, np.ndarray]:
    """Every rooted simple two-bond path, root-major, in frozen S2 order."""
    n = int(n_atoms)
    counts = np.zeros(n, dtype=np.int64)
    for root in range(n):
        counts[root] = sum(len(adj[int(mid)]) - 1 for mid, _bt, _e in adj[root])
    total = int(counts.sum())
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    middle = np.empty(total, dtype=np.int32)
    end = np.empty(total, dtype=np.int32)
    bond_ij = np.empty(total, dtype=np.int32)
    bond_jk = np.empty(total, dtype=np.int32)
    pos = 0
    for root in range(n):
        for mid, _bt1, e1 in adj[root]:
            for nxt, _bt2, e2 in adj[int(mid)]:
                if nxt == root:
                    continue
                middle[pos] = int(mid)
                end[pos] = int(nxt)
                bond_ij[pos] = int(e1)
                bond_jk[pos] = int(e2)
                pos += 1
    if pos != total:
        raise RuntimeError(f"CPC enumeration wrote {pos} paths, expected {total}")
    if total:
        rows = np.repeat(np.arange(n, dtype=np.int64), counts)
        if not np.all(middle != rows) or not np.all(end != rows) or not np.all(middle != end):
            raise RuntimeError("CPC enumeration produced a non-simple path")
    return {
        "ptr": ptr,
        "middle": middle,
        "end": end,
        "bond_ij": bond_ij,
        "bond_jk": bond_jk,
        "counts": counts,
    }


def count_cpc_closed_form(n_atoms: int, adj: Sequence[Sequence[tuple[int, int, int]]]) -> np.ndarray:
    """Exact per-root count ``sum_{j in N(i)} (deg(j) - 1)`` (independent check)."""
    n = int(n_atoms)
    deg = np.asarray([len(adj[i]) for i in range(n)], dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)
    for root in range(n):
        counts[root] = sum(int(deg[int(mid)]) - 1 for mid, _bt, _e in adj[root])
    return counts


def reference_enumerate_cpc(
    n_atoms: int,
    adj: Sequence[Sequence[tuple[int, int, int]]],
) -> list[list[dict[str, int]]]:
    """Independent brute-force reference (audit only).

    Enumerates ordered candidate triples by explicit neighbour-set membership,
    with no shared code path with :func:`enumerate_cpc_molecule`.
    """
    n = int(n_atoms)
    nbr = [{int(nb) for nb, _bt, _e in adj[i]} for i in range(n)]
    out: list[list[dict[str, int]]] = [[] for _ in range(n)]
    for root in range(n):
        for mid in range(n):
            if mid == root or mid not in nbr[root]:
                continue
            for end in range(n):
                if end in (root, mid) or end not in nbr[mid]:
                    continue
                out[root].append(
                    {
                        "root_atom": int(root),
                        "middle_atom": int(mid),
                        "end_atom": int(end),
                        "bond_ij": int(bond_id_lookup(adj, root, mid)),
                        "bond_jk": int(bond_id_lookup(adj, mid, end)),
                    }
                )
    return out


# ---------------------------------------------------------------------------
# build / load the structure cache
# ---------------------------------------------------------------------------


def _scope_rows(task: Any) -> np.ndarray:
    return np.arange(int(C.N_MOLECULES), dtype=np.int64)


def build_cpc_cache(
    task: Any,
    out_dir: Path | str = CPC_CACHE_DIR,
    *,
    rows: Sequence[int] | np.ndarray | None = None,
    log_every: int = 25_000,
) -> dict[str, Any]:
    """Build the exact, label-free CPC index cache for the requested rows.

    Default scope is every official row, because the frozen protocol allows the
    *structure* of the exposed Test set to be prepared before Test labels are
    opened.  No label array is indexed anywhere in this function.
    """
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = task.cache
    scope = np.sort(np.asarray(rows if rows is not None else _scope_rows(task), dtype=np.int64))
    if scope.size == 0 or scope[0] < 0 or scope[-1] >= int(C.N_MOLECULES):
        raise ValueError("CPC cache rows outside the official cache")
    if np.unique(scope).size != scope.size:
        raise ValueError("duplicate row in CPC cache scope")

    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    bond_ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    path_ptr = np.asarray(cache["path_ptr"], dtype=np.int64)
    path_end = np.asarray(cache["path_end"], dtype=np.int64)
    bond_index = np.asarray(cache["bond_index"], dtype=np.int64)
    bond_type = np.asarray(cache["bond_type"], dtype=np.int64)
    if int(atom_ptr[-1]) != int(path_ptr.size) - 1:
        raise RuntimeError("atom_ptr/path_ptr atom bookkeeping mismatch")

    row_n_atoms = (atom_ptr[scope + 1] - atom_ptr[scope]).astype(np.int64)
    row_n_bonds = (bond_ptr[scope + 1] - bond_ptr[scope]).astype(np.int64)
    n_scope_atoms = int(row_n_atoms.sum())
    total_paths = int(path_ptr[-1]) if scope.size == int(C.N_MOLECULES) else int(
        sum(path_ptr[atom_ptr[row + 1]] - path_ptr[atom_ptr[row]] for row in scope.tolist())
    )

    cpc_ptr = np.empty(n_scope_atoms + 1, dtype=np.int64)
    cpc_ptr[0] = 0
    cpc_middle = np.empty(total_paths, dtype=np.int32)
    cpc_end = np.empty(total_paths, dtype=np.int32)
    cpc_bond_ij = np.empty(total_paths, dtype=np.int32)
    cpc_bond_jk = np.empty(total_paths, dtype=np.int32)

    pos = 0
    ptr_pos = 0
    order_mismatch_rows: list[int] = []
    multiset_mismatch_rows: list[int] = []
    bond_mismatch = 0
    cross_molecule = 0
    for i, row in enumerate(scope.tolist()):
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        n = a1 - a0
        bi = bond_index[b0:b1]
        bt = bond_type[b0:b1]
        adj = molecule_adjacency(n, bi[:, 0], bi[:, 1], bt)
        paths = enumerate_cpc_molecule(n, adj)
        counts = paths["counts"]
        expected_counts = np.diff(path_ptr[a0 : a1 + 1])
        if not np.array_equal(counts, expected_counts):
            raise RuntimeError(
                f"row {row}: CPC per-root counts differ from frozen S2 counts "
                f"({int(counts.sum())} vs {int(expected_counts.sum())})"
            )
        p0 = int(path_ptr[a0])
        expected_ends = path_end[p0 : p0 + int(counts.sum())].astype(np.int32)
        got_ends = paths["end"]
        if not np.array_equal(got_ends, expected_ends):
            order_mismatch_rows.append(int(row))
            expected_sorted = []
            got_sorted = []
            off = 0
            for c in counts.tolist():
                expected_sorted.append(np.sort(expected_ends[off : off + c]))
                got_sorted.append(np.sort(got_ends[off : off + c]))
                off += c
            if not all(np.array_equal(a, b) for a, b in zip(expected_sorted, got_sorted)):
                multiset_mismatch_rows.append(int(row))
        c = int(paths["ptr"][-1])
        cpc_ptr[ptr_pos : ptr_pos + n + 1] = paths["ptr"] + pos
        if pos + c > total_paths:
            raise RuntimeError("CPC cache allocation overflow")
        cpc_middle[pos : pos + c] = paths["middle"]
        cpc_end[pos : pos + c] = got_ends
        cpc_bond_ij[pos : pos + c] = paths["bond_ij"]
        cpc_bond_jk[pos : pos + c] = paths["bond_jk"]
        # independent bond-endpoint verification on every path of this molecule
        if c:
            root = np.repeat(np.arange(n, dtype=np.int64), counts)
            mid = paths["middle"].astype(np.int64)
            end = paths["end"].astype(np.int64)
            e1 = paths["bond_ij"].astype(np.int64)
            e2 = paths["bond_jk"].astype(np.int64)
            u1, v1 = bi[e1, 0], bi[e1, 1]
            u2, v2 = bi[e2, 0], bi[e2, 1]
            ok1 = ((u1 == root) & (v1 == mid)) | ((u1 == mid) & (v1 == root))
            ok2 = ((u2 == mid) & (v2 == end)) | ((u2 == end) & (v2 == mid))
            bond_mismatch += int((~ok1).sum() + (~ok2).sum())
            cross_molecule += int(
                ((root < 0) | (root >= n) | (mid < 0) | (mid >= n) | (end < 0) | (end >= n)).sum()
            )
        ptr_pos += n
        pos += c
        if log_every and (i + 1) % int(log_every) == 0:
            C.log(
                f"[cpc-cache] {i + 1}/{scope.size} rows, {pos} paths, "
                f"{time.perf_counter() - t0:.1f}s"
            )

    if ptr_pos != n_scope_atoms:
        raise RuntimeError("CPC cache atom pointer length mismatch")
    if pos != total_paths or cpc_ptr[-1] != total_paths:
        raise RuntimeError("CPC cache total path count mismatch")
    if multiset_mismatch_rows:
        raise RuntimeError(
            f"CPC endpoint multiset mismatch on rows {multiset_mismatch_rows[:10]}"
        )
    if bond_mismatch or cross_molecule:
        raise RuntimeError(
            f"CPC bond/molecule audit failed: bond_mismatch={bond_mismatch} cross={cross_molecule}"
        )

    row_atom_ptr = np.concatenate([[0], np.cumsum(row_n_atoms)]).astype(np.int64)
    row_bond_ptr = np.concatenate([[0], np.cumsum(row_n_bonds)]).astype(np.int64)
    arrays: dict[str, np.ndarray] = {
        "cpc_ptr": cpc_ptr,
        "cpc_middle": cpc_middle,
        "cpc_end": cpc_end,
        "cpc_bond_ij": cpc_bond_ij,
        "cpc_bond_jk": cpc_bond_jk,
        "rows": scope.astype(np.int64),
        "row_atom_ptr": row_atom_ptr,
        "row_bond_ptr": row_bond_ptr,
        "row_n_atoms": row_n_atoms,
        "row_n_bonds": row_n_bonds,
    }
    field_sha256: dict[str, str] = {}
    for name, arr in arrays.items():
        path = out_dir / f"{name}.npy"
        np.save(path, arr)
        field_sha256[name] = C.sha256_file(path)

    n_paths_total = int(cpc_ptr[-1])
    nbytes = int(sum((out_dir / f"{name}.npy").stat().st_size for name in arrays))
    seconds = float(time.perf_counter() - t0)
    meta = {
        "schema": C.SCHEMA_CACHE,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "scope": "official rows sorted (default: all 437,929 rows; Test structure only, no labels)",
        "enumeration_schema_version": "rooted-two-bond-simple-path-v1",
        "n_rows": int(scope.size),
        "n_atoms": int(n_scope_atoms),
        "n_bonds": int(row_n_bonds.sum()),
        "n_paths": n_paths_total,
        "paths_per_atom_mean": float(n_paths_total / n_scope_atoms) if n_scope_atoms else 0.0,
        "paths_per_row_mean": float(n_paths_total / scope.size) if scope.size else 0.0,
        "max_paths_per_atom": int(np.diff(cpc_ptr).max()) if n_paths_total else 0,
        "bytes": nbytes,
        "build_seconds": seconds,
        "paths_per_second": float(n_paths_total / seconds) if seconds > 0 else 0.0,
        "rows_sha256": C.sha256_bytes(np.ascontiguousarray(scope, dtype=np.int64).tobytes()),
        "field_sha256": field_sha256,
        "feature_spec_hash": C.SOURCE["feature_spec_hash"],
        "split_partition_sha256": C.SOURCE["split_partition_sha256"],
        "cache_sha256": C.SOURCE["cache_sha256"],
        "label_artifact_sha256": C.SOURCE["label_artifact_sha256"],
        "official_row_ids": "arange(437929) for the all-rows scope",
        "endpoint_order_matches_frozen_s2": len(order_mismatch_rows) == 0,
        "endpoint_order_mismatch_rows": int(len(order_mismatch_rows)),
        "endpoint_multiset_matches_frozen_s2": len(multiset_mismatch_rows) == 0,
        "frozen_s2_path_count": int(path_ptr[-1]),
        "labels_present": False,
        "test_labels_present": False,
        "test_structure_present": True,
        "definition": dict(C.CPC),
    }
    C.write_json(out_dir / "meta.json", meta)
    C.log(
        f"[cpc-cache] done rows={meta['n_rows']} atoms={meta['n_atoms']} "
        f"paths={meta['n_paths']} bytes={nbytes / 1e9:.3f}GB seconds={seconds:.1f}"
    )
    return meta


def load_cpc_cache(path: Path | str = CPC_CACHE_DIR) -> dict[str, Any]:
    path = Path(path)
    meta = C.read_json(path / "meta.json")
    if meta.get("schema") != C.SCHEMA_CACHE:
        raise RuntimeError(f"CPC cache schema {meta.get('schema')} != {C.SCHEMA_CACHE}")
    if bool(meta.get("labels_present")) or bool(meta.get("test_labels_present")):
        raise RuntimeError("CPC cache unexpectedly contains labels")
    out: dict[str, Any] = {"meta": meta, "path": path}
    for name in CPC_CACHE_FIELDS:
        out[name] = np.load(path / f"{name}.npy", mmap_mode="r")
    rows = np.asarray(out["rows"], dtype=np.int64)
    row_index = np.full(int(C.N_MOLECULES), -1, dtype=np.int64)
    row_index[rows] = np.arange(rows.size, dtype=np.int64)
    out["row_index"] = row_index
    if int(out["row_atom_ptr"][-1]) != int(out["cpc_ptr"].shape[0]) - 1:
        raise RuntimeError("CPC cache atom bookkeeping mismatch")
    if int(out["row_bond_ptr"][-1]) != int(meta["n_bonds"]):
        raise RuntimeError("CPC cache bond bookkeeping mismatch")
    if int(out["cpc_ptr"][-1]) != int(meta["n_paths"]):
        raise RuntimeError("CPC cache path bookkeeping mismatch")
    return out


def cpc_cache_identity(path: Path | str | None = None) -> dict[str, Any]:
    base = Path(path) if path is not None else CPC_CACHE_DIR
    meta_path = base / "meta.json"
    if not meta_path.is_file():
        return {"present": False, "path": C.track_rel(base)}
    meta = C.read_json(meta_path)
    return {
        "present": True,
        "path": C.track_rel(base),
        "meta_sha256": C.sha256_file(meta_path),
        "n_rows": int(meta["n_rows"]),
        "n_paths": int(meta["n_paths"]),
        "bytes": int(meta["bytes"]),
        "build_seconds": float(meta["build_seconds"]),
        "rows_sha256": meta["rows_sha256"],
        "field_sha256": meta["field_sha256"],
        "enumeration_schema_version": meta["enumeration_schema_version"],
        "labels_present": bool(meta.get("labels_present", True)),
        "test_labels_present": bool(meta.get("test_labels_present", True)),
        "definition": meta.get("definition"),
    }


def load_task_with_cpc(cache_path: Path | str | None = None) -> Any:
    task = C.SD.load_task()
    cpc = load_cpc_cache(Path(cache_path) if cache_path else CPC_CACHE_DIR)
    rows = np.asarray(cpc["rows"], dtype=np.int64)
    missing_train = np.setdiff1d(np.asarray(task.train_rows, np.int64), rows)
    missing_valid = np.setdiff1d(np.asarray(task.valid_rows, np.int64), rows)
    missing_test = np.setdiff1d(np.asarray(task.test_rows, np.int64), rows)
    if missing_train.size or missing_valid.size or missing_test.size:
        raise RuntimeError(
            f"CPC cache is missing rows: train {missing_train.size}, valid {missing_valid.size}, "
            f"test {missing_test.size}"
        )
    task.cpc = cpc
    return task


# ---------------------------------------------------------------------------
# batch collation (vectorised; no per-path Python loop)
# ---------------------------------------------------------------------------


def collate_cpc(cpc: Mapping[str, Any], rows: Sequence[int]) -> dict[str, np.ndarray]:
    rows = np.asarray(rows, dtype=np.int64).reshape(-1)
    idx = np.asarray(cpc["row_index"])[rows]
    if rows.size and np.any(idx < 0):
        raise ValueError("row outside the CPC structure cache")
    row_atom_ptr = np.asarray(cpc["row_atom_ptr"], dtype=np.int64)
    row_bond_ptr = np.asarray(cpc["row_bond_ptr"], dtype=np.int64)
    a0 = row_atom_ptr[idx] if rows.size else np.zeros(0, dtype=np.int64)
    a1 = row_atom_ptr[idx + 1] if rows.size else np.zeros(0, dtype=np.int64)
    b0 = row_bond_ptr[idx] if rows.size else np.zeros(0, dtype=np.int64)
    b1 = row_bond_ptr[idx + 1] if rows.size else np.zeros(0, dtype=np.int64)
    atom_counts = (a1 - a0).astype(np.int64)
    bond_counts = (b1 - b0).astype(np.int64)
    n_atoms = int(atom_counts.sum())
    n_bonds = int(bond_counts.sum())
    atom_off = np.concatenate([[0], np.cumsum(atom_counts)[:-1]]).astype(np.int64) if rows.size else np.zeros(0, np.int64)
    bond_off = np.concatenate([[0], np.cumsum(bond_counts)[:-1]]).astype(np.int64) if rows.size else np.zeros(0, np.int64)

    # vectorised row ranges: repeat the start and add the within-row offset
    atom_idx = (
        np.repeat(a0, atom_counts)
        + (np.arange(n_atoms, dtype=np.int64) - np.repeat(np.cumsum(atom_counts) - atom_counts, atom_counts))
        if n_atoms
        else np.zeros(0, dtype=np.int64)
    )
    atom_graph = np.repeat(np.arange(rows.size, dtype=np.int64), atom_counts) if rows.size else np.zeros(0, np.int64)
    cpc_ptr = np.asarray(cpc["cpc_ptr"], dtype=np.int64)
    counts = (cpc_ptr[atom_idx + 1] - cpc_ptr[atom_idx]).astype(np.int64) if n_atoms else np.zeros(0, dtype=np.int64)
    n_paths = int(counts.sum())
    path_idx = (
        np.repeat(cpc_ptr[atom_idx], counts)
        + (np.arange(n_paths, dtype=np.int64) - np.repeat(np.cumsum(counts) - counts, counts))
        if n_paths
        else np.zeros(0, dtype=np.int64)
    )
    path_atom = np.repeat(np.arange(n_atoms, dtype=np.int64), counts) if n_atoms else np.zeros(0, np.int64)
    path_graph = np.repeat(atom_graph, counts) if n_atoms else np.zeros(0, np.int64)
    middle = np.asarray(cpc["cpc_middle"])[path_idx].astype(np.int64) + atom_off[path_graph] if n_paths else np.zeros(0, np.int64)
    end = np.asarray(cpc["cpc_end"])[path_idx].astype(np.int64) + atom_off[path_graph] if n_paths else np.zeros(0, np.int64)
    bond_ij = np.asarray(cpc["cpc_bond_ij"])[path_idx].astype(np.int64) + bond_off[path_graph] if n_paths else np.zeros(0, np.int64)
    bond_jk = np.asarray(cpc["cpc_bond_jk"])[path_idx].astype(np.int64) + bond_off[path_graph] if n_paths else np.zeros(0, np.int64)
    return {
        "cpc_path_atom": path_atom,
        "cpc_middle": middle,
        "cpc_end": end,
        "cpc_bond_ij": bond_ij,
        "cpc_bond_jk": bond_jk,
        "atom_counts": atom_counts,
        "bond_counts": bond_counts,
        "n_paths": np.asarray([n_paths], dtype=np.int64),
        "n_atoms": np.asarray([n_atoms], dtype=np.int64),
        "n_bonds": np.asarray([n_bonds], dtype=np.int64),
    }


def cpc_torch_batch(cpc_np: Mapping[str, np.ndarray]) -> dict[str, Any]:
    import torch

    return {
        "cpc_path_atom": torch.from_numpy(np.ascontiguousarray(cpc_np["cpc_path_atom"], dtype=np.int64)),
        "cpc_middle": torch.from_numpy(np.ascontiguousarray(cpc_np["cpc_middle"], dtype=np.int64)),
        "cpc_end": torch.from_numpy(np.ascontiguousarray(cpc_np["cpc_end"], dtype=np.int64)),
        "cpc_bond_ij": torch.from_numpy(np.ascontiguousarray(cpc_np["cpc_bond_ij"], dtype=np.int64)),
        "cpc_bond_jk": torch.from_numpy(np.ascontiguousarray(cpc_np["cpc_bond_jk"], dtype=np.int64)),
    }


def collate_with_cpc(task: Any, cpc: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    batch = dict(C.PC.collate(task.cache, list(rows)))
    cpc_np = collate_cpc(cpc, rows)
    if not np.array_equal(batch["atom_counts"], cpc_np["atom_counts"]):
        raise RuntimeError("CPC atom counts disagree with the frozen collate")
    if not np.array_equal(batch["bond_counts"], cpc_np["bond_counts"]):
        raise RuntimeError("CPC bond counts disagree with the frozen collate")
    batch.update(cpc_torch_batch(cpc_np))
    return batch


# ---------------------------------------------------------------------------
# isolation and the post-training pairing intervention
# ---------------------------------------------------------------------------


def check_batch_isolation(batch: Mapping[str, Any]) -> dict[str, Any]:
    import torch

    path_atom = batch["cpc_path_atom"]
    middle = batch["cpc_middle"]
    end = batch["cpc_end"]
    atom_batch = batch["atom_batch"]
    if path_atom.numel() == 0:
        return {"n_paths": 0, "cross_molecule": 0, "isolated": True, "bonds_match": True}
    same = (
        (atom_batch.index_select(0, path_atom) == atom_batch.index_select(0, middle))
        & (atom_batch.index_select(0, middle) == atom_batch.index_select(0, end))
    )
    cross = int((~same).sum().item())
    bij = batch["cpc_bond_ij"]
    bjk = batch["cpc_bond_jk"]
    bi = batch["bond_index"]
    bond_batch = batch["bond_batch"]
    ok_bonds = bool(
        torch.equal(bond_batch.index_select(0, bij), atom_batch.index_select(0, path_atom))
    ) and bool(torch.equal(bond_batch.index_select(0, bjk), atom_batch.index_select(0, middle)))
    if bij.numel():
        u1 = bi.index_select(0, bij)[:, 0]
        v1 = bi.index_select(0, bij)[:, 1]
        u2 = bi.index_select(0, bjk)[:, 0]
        v2 = bi.index_select(0, bjk)[:, 1]
        ok1 = ((u1 == path_atom) & (v1 == middle)) | ((u1 == middle) & (v1 == path_atom))
        ok2 = ((u2 == middle) & (v2 == end)) | ((u2 == end) & (v2 == middle))
        endpoints_match = bool(ok1.all().item() and ok2.all().item())
    else:
        endpoints_match = True
    return {
        "n_paths": int(path_atom.numel()),
        "cross_molecule": cross,
        "isolated": cross == 0,
        "bonds_same_molecule": ok_bonds,
        "bond_endpoints_match_path": endpoints_match,
        "max_path_atom": int(path_atom.max().item()),
        "max_middle": int(middle.max().item()),
        "max_end": int(end.max().item()),
        "n_atoms": int(atom_batch.numel()),
    }


def shuffle_pairing_within_root(
    path_atom: Any,
    middle: Any,
    end: Any,
    bond_ij: Any,
    bond_jk: Any,
    *,
    seed: int,
) -> tuple[Any, Any, Any, Any, Any, Any]:
    """Permute ``end``/``bond_jk`` slots inside each root (diagnostic only).

    ``path_atom``, ``middle`` and ``bond_ij`` stay at their original positions;
    the endpoint and second-bond slots are randomly permuted among the paths of
    the same root.  Root membership and path counts are unchanged, every value
    still belongs to the same molecule, and no cross-molecule index can be
    created (all paths of a root are in the same molecule by construction).
    """
    import numpy as np
    import torch

    device = path_atom.device
    atom = path_atom.detach().cpu().numpy().astype(np.int64)
    n = int(atom.size)
    if n == 0:
        return path_atom, middle, end, bond_ij, bond_jk, torch.zeros(0, dtype=torch.long)
    rng = np.random.default_rng(int(seed))
    keys = rng.random(n)
    if n > 1 and np.any(np.diff(atom) < 0):
        raise RuntimeError("within-root shuffle expects root-major path order")
    order = np.lexsort((keys, atom))
    new_end = np.empty_like(end.detach().cpu().numpy())
    new_jk = np.empty_like(bond_jk.detach().cpu().numpy())
    new_end[order] = end.detach().cpu().numpy()
    new_jk[order] = bond_jk.detach().cpu().numpy()
    perm = np.arange(n, dtype=np.int64)
    perm[order] = np.arange(n, dtype=np.int64)
    return (
        path_atom,
        middle,
        torch.as_tensor(new_end, device=device, dtype=end.dtype),
        bond_ij,
        torch.as_tensor(new_jk, device=device, dtype=bond_jk.dtype),
        torch.as_tensor(perm, device=device, dtype=torch.long),
    )


__all__ = [name for name in dir() if not name.startswith("_")]

"""Static, label-free structural cache for the LCRC-BOC round.

Everything here is derived from the original chemical bond graph only:

* every physical bond ``{u,v}`` becomes exactly one bond-centred object;
* the object carries the induced subgraph ``P_uv = G[{u,v} u N(u) u N(v)]``:
  the two endpoint neighbour lists, the same-side forks and the cross-side
  branches, with every real induced edge and its true ``bond_feat`` row;
* bridge edges are found on the original graph; the non-bridge subgraph is
  split into connected components, one ring-system object each; ring atoms
  carry their external substituents and every intra-system atom pair with the
  true shortest path distance inside the non-bridge subgraph.

The build reads only ``atom_ptr`` / ``bond_ptr`` / ``bond_index``.  It never
opens the label artifact, never uses RDKit 3-D information and never uses an
SSSR ring basis.  The cache is a pure function of the official row order.

Storage is packed: one row of flat arrays per physical bond / ring system /
ring atom, sliced with vectorised pointer expansion at collate time.  No atom
embedding, bond embedding or network output is cached.
"""

from __future__ import annotations

import json
import time
from array import array
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import boc_common as C


# ---------------------------------------------------------------------------
# primitives
# ---------------------------------------------------------------------------


def distance_bucket(distance: int) -> int:
    """Map a positive ring-subgraph distance to bucket 0..5."""
    d = int(distance)
    if d < 1:
        raise ValueError("ring-system pair distance must be >= 1")
    for index, (lo, hi) in enumerate(C.RING_BUCKETS):
        if hi is None:
            if d >= lo:
                return int(index)
        elif lo <= d <= hi:
            return int(index)
    raise AssertionError("distance bucket table does not cover a positive distance")


def find_bridges(n_atoms: int, edges: np.ndarray) -> np.ndarray:
    """Iterative Tarjan bridge detection on the original undirected graph."""
    n = int(n_atoms)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    m = int(e.shape[0])
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for eid in range(m):
        u, v = int(e[eid, 0]), int(e[eid, 1])
        adj[u].append((v, eid))
        adj[v].append((u, eid))
    disc = [-1] * n
    low = [0] * n
    bridge = np.zeros(m, dtype=bool)
    timer = 0
    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = timer
        timer += 1
        stack: list[tuple[int, int, int]] = [(root, -1, 0)]
        while stack:
            u, parent_edge, cursor = stack[-1]
            if cursor < len(adj[u]):
                v, eid = adj[u][cursor]
                stack[-1] = (u, parent_edge, cursor + 1)
                if eid == parent_edge:
                    continue
                if disc[v] == -1:
                    disc[v] = low[v] = timer
                    timer += 1
                    stack.append((v, eid, 0))
                else:
                    if disc[v] < low[u]:
                        low[u] = disc[v]
            else:
                stack.pop()
                if stack:
                    parent = stack[-1][0]
                    if low[u] < low[parent]:
                        low[parent] = low[u]
                    if low[u] > disc[parent]:
                        bridge[parent_edge] = True
    return bridge


def reference_bridges_by_removal(n_atoms: int, edges: np.ndarray) -> np.ndarray:
    """Independent reference: an edge is a bridge iff removing it adds a component."""
    n = int(n_atoms)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    m = int(e.shape[0])

    def components(skip: int) -> int:
        adj: list[list[int]] = [[] for _ in range(n)]
        for eid in range(m):
            if eid == skip:
                continue
            u, v = int(e[eid, 0]), int(e[eid, 1])
            adj[u].append(v)
            adj[v].append(u)
        seen = [False] * n
        count = 0
        for root in range(n):
            if seen[root]:
                continue
            count += 1
            seen[root] = True
            stack = [root]
            while stack:
                u = stack.pop()
                for v in adj[u]:
                    if not seen[v]:
                        seen[v] = True
                        stack.append(v)
        return count

    base = components(-1)
    out = np.zeros(m, dtype=bool)
    for skip in range(m):
        out[skip] = components(skip) > base
    return out


def _adjacency(n_atoms: int, edges: np.ndarray):
    n = int(n_atoms)
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    adj_list: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    adj_map: list[dict[int, int]] = [{} for _ in range(n)]
    for eid in range(int(e.shape[0])):
        u, v = int(e[eid, 0]), int(e[eid, 1])
        if u == v:
            raise RuntimeError("self loop in bond_index")
        if u in adj_map[v] or v in adj_map[u]:
            raise RuntimeError("duplicate physical bond")
        adj_list[u].append((v, eid))
        adj_list[v].append((u, eid))
        adj_map[u][v] = eid
        adj_map[v][u] = eid
    for u in range(n):
        adj_list[u].sort(key=lambda pair: pair[0])
    return e, adj_list, adj_map


def _ring_system_records(
    n_atoms: int,
    e: np.ndarray,
    adj_list: Sequence[Sequence[tuple[int, int]]],
    adj_map: Sequence[Mapping[int, int]],
) -> list[dict[str, Any]]:
    """Components of the non-bridge subgraph, with exact internal distances."""
    n = int(n_atoms)
    m = int(e.shape[0])
    bridge = find_bridges(n, e)
    nonbridge = ~bridge
    nonbridge_adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for eid in range(m):
        if not bool(nonbridge[eid]):
            continue
        u, v = int(e[eid, 0]), int(e[eid, 1])
        nonbridge_adj[u].append((v, eid))
        nonbridge_adj[v].append((u, eid))
    for u in range(n):
        nonbridge_adj[u].sort(key=lambda pair: pair[0])

    comp_id = np.full(n, -1, dtype=np.int64)
    comps: list[list[int]] = []
    for root in range(n):
        if comp_id[root] != -1 or not nonbridge_adj[root]:
            continue
        gid = len(comps)
        comp_id[root] = gid
        stack = [root]
        members: list[int] = []
        while stack:
            u = stack.pop()
            members.append(int(u))
            for v, _eid in nonbridge_adj[u]:
                if comp_id[v] == -1:
                    comp_id[v] = gid
                    stack.append(int(v))
        comps.append(sorted(members))

    records: list[dict[str, Any]] = []
    for gid, atoms in enumerate(comps):
        atom_set = set(atoms)
        index_of = {int(a): int(pos) for pos, a in enumerate(atoms)}
        internal_edges = [
            int(eid)
            for eid in range(m)
            if bool(nonbridge[eid]) and int(e[eid, 0]) in atom_set and int(e[eid, 1]) in atom_set
        ]
        # external substituents, grouped by ring atom in sorted atom order
        ext_entries: list[tuple[int, int, int]] = []
        for atom in atoms:
            for other, eid in adj_list[atom]:
                if int(other) not in atom_set:
                    ext_entries.append((int(atom), int(other), int(eid)))
        ext_entries.sort(key=lambda row: (index_of[row[0]], row[1]))
        internal_adj: list[list[int]] = [[] for _ in atoms]
        for eid in internal_edges:
            u, v = int(e[eid, 0]), int(e[eid, 1])
            iu, iv = index_of[u], index_of[v]
            internal_adj[iu].append(iv)
            internal_adj[iv].append(iu)
        for row in internal_adj:
            row.sort()
        # exact all-pairs BFS inside this ring system only
        n_sys = len(atoms)
        dist = np.full((n_sys, n_sys), -1, dtype=np.int64)
        for src in range(n_sys):
            dist[src, src] = 0
            frontier = [src]
            d = 0
            while frontier:
                d += 1
                nxt: list[int] = []
                for u in frontier:
                    for v in internal_adj[u]:
                        if dist[src, v] == -1:
                            dist[src, v] = d
                            nxt.append(v)
                frontier = nxt
        pairs: list[tuple[int, int, int, int]] = []
        for i in range(n_sys):
            for j in range(i + 1, n_sys):
                d = int(dist[i, j])
                if d < 1:
                    raise RuntimeError(f"ring system pair ({i},{j}) is not connected")
                bond = int(adj_map[atoms[i]].get(atoms[j], -1))
                if bond >= 0:
                    if not bool(nonbridge[bond]):
                        raise RuntimeError("an internal ring-system bond was marked as a bridge")
                    if d != 1:
                        raise RuntimeError("a directly bonded ring pair has distance != 1")
                pairs.append((i, j, distance_bucket(d), bond))
        records.append(
            {
                "atoms": atoms,
                "index_of": index_of,
                "internal_edges": internal_edges,
                "external_entries": ext_entries,
                "pairs": pairs,
                "n_edges": len(internal_edges),
                "n_external": len(ext_entries),
            }
        )
    return records


def enumerate_molecule(n_atoms: int, edges: np.ndarray) -> dict[str, Any]:
    """Full, independent enumeration of one molecule's BOC objects (Python)."""
    n = int(n_atoms)
    e, adj_list, adj_map = _adjacency(n, edges)
    m = int(e.shape[0])

    counts = np.zeros((m, 5), dtype=np.int64)  # nL,nR,nLL,nRR,nLR
    neighbours: list[dict[str, Any]] = []
    pairs: list[dict[str, Any]] = []
    for eid in range(m):
        u, v = int(e[eid, 0]), int(e[eid, 1])
        left = sorted(int(w) for w, _ in adj_list[u] if int(w) != v)
        right = sorted(int(w) for w, _ in adj_list[v] if int(w) != u)
        n_l, n_r = len(left), len(right)
        tri = len(set(left) & set(right))
        n_ll = n_l * (n_l - 1) // 2
        n_rr = n_r * (n_r - 1) // 2
        n_lr = n_l * n_r - tri
        counts[eid] = (n_l, n_r, n_ll, n_rr, n_lr)

        nb: list[tuple[int, int, int, int]] = []  # atom, side, bond_side, bond_other
        for w in left:
            nb.append((w, 0, int(adj_map[u][w]), int(adj_map[v].get(w, -1))))
        for w in right:
            nb.append((w, 1, int(adj_map[v][w]), int(adj_map[u].get(w, -1))))
        neighbours.append({"nL": n_l, "nR": n_r, "entries": nb})

        pr: list[tuple[int, int, int, int]] = []  # nb1, nb2, kind, bond
        for a in range(n_l):
            for b in range(a + 1, n_l):
                w1, w2 = left[a], left[b]
                pr.append((a, b, 0, int(adj_map[w1].get(w2, -1))))
        for a in range(n_r):
            for b in range(a + 1, n_r):
                w1, w2 = right[a], right[b]
                pr.append((n_l + a, n_l + b, 1, int(adj_map[w1].get(w2, -1))))
        for a, w in enumerate(left):
            for b, z in enumerate(right):
                if w == z:
                    continue
                pr.append((a, n_l + b, 2, int(adj_map[w].get(z, -1))))
        if len(pr) != int(n_ll + n_rr + n_lr):
            raise RuntimeError("pair enumeration count mismatch")
        pairs.append(pr)

    rings = _ring_system_records(n, e, adj_list, adj_map)
    return {
        "n_atoms": n,
        "n_bonds": m,
        "counts": counts,
        "neighbours": neighbours,
        "pairs": pairs,
        "rings": rings,
        "adjacency": adj_list,
        "adjacency_map": adj_map,
        "edges": e,
        "bridges": find_bridges(n, e),
    }


# ---------------------------------------------------------------------------
# two-pass packed cache build
# ---------------------------------------------------------------------------


ARRAY_NAMES = (
    "bond_nb_ptr",
    "bond_pair_ptr",
    "bond_nL",
    "bond_nR",
    "bond_nLL",
    "bond_nRR",
    "bond_nLR",
    "nb_atom",
    "nb_side",
    "nb_bond_side",
    "nb_bond_other",
    "pair_a",
    "pair_b",
    "pair_kind",
    "pair_bond",
    "ring_ptr",
    "ring_atom_ptr",
    "ring_pair_ptr",
    "ring_ext_ptr",
    "ring_n_edges",
    "ring_atom",
    "ring_pair_i",
    "ring_pair_j",
    "ring_pair_d",
    "ring_pair_bond",
    "ring_ext_atom",
    "ring_ext_bond",
    "row_bond_ptr",
    "bond_map",
)


def _source_arrays(cache_path: Path | str):
    cache_path = Path(cache_path)
    atom_ptr = np.load(cache_path / "atom_ptr.npy")
    bond_ptr = np.load(cache_path / "bond_ptr.npy")
    bond_index = np.load(cache_path / "bond_index.npy", mmap_mode="r")
    return atom_ptr, bond_ptr, bond_index


def build_cache(
    out_dir: Path | str = C.STRUCT_CACHE_DIR,
    *,
    cache_path: Path | str | None = None,
    rows: Sequence[int] | None = None,
    reference_fraction: float = 0.001,
    reference_seed: int = 20261109,
    log_every: int = 50_000,
) -> dict[str, Any]:
    """Build the packed BOC structural cache.  Labels are not opened.

    ``rows=None`` builds the full official row order (the training cache).
    A subset build is supported for tests and for a future label-free Test-only
    cache: bond objects are compacted into the selected row order and
    ``bond_map`` maps a global physical-bond id to its compact slot.
    """
    t0 = time.perf_counter()
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    if cache_path is None:
        cache_path = C.BC.SD.LD.C.CACHE_PATH
    cache_path = Path(cache_path)
    atom_ptr, bond_ptr, bond_index = _source_arrays(cache_path)
    n_molecules = int(atom_ptr.size - 1)
    if rows is None:
        row_ids = np.arange(n_molecules, dtype=np.int64)
    else:
        row_ids = np.asarray(rows, dtype=np.int64)
        if row_ids.ndim != 1 or np.unique(row_ids).size != row_ids.size:
            raise ValueError("rows must be a 1-D unique row list")
        if row_ids.size and (row_ids.min() < 0 or row_ids.max() >= n_molecules):
            raise ValueError("row id outside the cache")
    row_ids = np.asarray(row_ids, dtype=np.int64)
    n_rows = int(row_ids.size)
    atom_counts = np.diff(atom_ptr).astype(np.int64)
    bond_counts = np.diff(bond_ptr).astype(np.int64)
    selected_bond_counts = bond_counts[row_ids].astype(np.int64)
    row_bond_ptr = np.zeros(n_rows + 1, dtype=np.int64)
    np.cumsum(selected_bond_counts, out=row_bond_ptr[1:])
    n_bonds_total = int(row_bond_ptr[-1])
    if n_bonds_total >= 2**31:
        raise RuntimeError("bond count does not fit in int32")
    bond_map = np.full(int(bond_counts.sum()), -1, dtype=np.int32)
    for k, row in enumerate(row_ids):
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        bond_map[b0:b1] = (int(row_bond_ptr[k]) + np.arange(b1 - b0, dtype=np.int64)).astype(np.int32)
    # ---- pass 1: per-bond counts and per-system / per-ring-atom extents ----
    bond_n = np.zeros((n_bonds_total, 5), dtype=np.uint8)
    ring_ptr = np.zeros(n_rows + 1, dtype=np.int64)
    system_atoms: array = array("i")
    system_pairs: array = array("i")
    system_ext: array = array("i")
    system_edges: array = array("i")
    ring_atom_ext: array = array("i")
    for k, row in enumerate(row_ids):
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        rec = enumerate_molecule(a1 - a0, np.asarray(bond_index[b0:b1], dtype=np.int64))
        lb0, lb1 = int(row_bond_ptr[k]), int(row_bond_ptr[k + 1])
        if lb1 - lb0 != rec["n_bonds"]:
            raise RuntimeError("selected bond block size mismatch")
        bond_n[lb0:lb1] = rec["counts"].astype(np.uint8)
        rings = rec["rings"]
        ring_ptr[k + 1] = ring_ptr[k] + len(rings)
        for system in rings:
            system_atoms.append(len(system["atoms"]))
            system_pairs.append(len(system["pairs"]))
            system_ext.append(len(system["external_entries"]))
            system_edges.append(int(system["n_edges"]))
            per_atom = np.zeros(len(system["atoms"]), dtype=np.int64)
            for atom, _other, _eid in system["external_entries"]:
                per_atom[system["index_of"][atom]] += 1
            for count in per_atom.tolist():
                ring_atom_ext.append(int(count))
        if log_every and (k + 1) % int(log_every) == 0:
            C.log(f"[struct-cache] counts {k + 1}/{n_rows} rows ({time.perf_counter() - t0:.1f}s)")

    n_nb_total = int(bond_n[:, 0].astype(np.int64).sum() + bond_n[:, 1].astype(np.int64).sum())
    n_pair_total = int(
        bond_n[:, 2].astype(np.int64).sum()
        + bond_n[:, 3].astype(np.int64).sum()
        + bond_n[:, 4].astype(np.int64).sum()
    )
    if n_nb_total >= 2**31 or n_pair_total >= 2**31:
        raise RuntimeError("packed neighbour/pair offsets do not fit in int32")
    n_systems = int(ring_ptr[-1])
    n_ring_atoms = int(sum(system_atoms))
    n_ring_pairs = int(sum(system_pairs))
    n_ext = int(sum(system_ext))
    if int(sum(ring_atom_ext)) != n_ext:
        raise RuntimeError("ring external counts do not match the flat entry count")

    arrays: dict[str, np.ndarray] = {}
    arrays["row_bond_ptr"] = row_bond_ptr.astype(np.int64)
    arrays["bond_map"] = bond_map
    arrays["bond_nb_ptr"] = np.zeros(n_bonds_total + 1, dtype=np.int64)
    arrays["bond_pair_ptr"] = np.zeros(n_bonds_total + 1, dtype=np.int64)
    np.cumsum(
        bond_n[:, 0].astype(np.int64) + bond_n[:, 1].astype(np.int64),
        out=arrays["bond_nb_ptr"][1:],
    )
    np.cumsum(
        bond_n[:, 2].astype(np.int64)
        + bond_n[:, 3].astype(np.int64)
        + bond_n[:, 4].astype(np.int64),
        out=arrays["bond_pair_ptr"][1:],
    )
    for index, name in enumerate(("bond_nL", "bond_nR", "bond_nLL", "bond_nRR", "bond_nLR")):
        arrays[name] = bond_n[:, index].copy()
    arrays["nb_atom"] = np.zeros(n_nb_total, dtype=np.int32)
    arrays["nb_side"] = np.zeros(n_nb_total, dtype=np.uint8)
    arrays["nb_bond_side"] = np.zeros(n_nb_total, dtype=np.int32)
    arrays["nb_bond_other"] = np.full(n_nb_total, -1, dtype=np.int32)
    arrays["pair_a"] = np.zeros(n_pair_total, dtype=np.int32)
    arrays["pair_b"] = np.zeros(n_pair_total, dtype=np.int32)
    arrays["pair_kind"] = np.zeros(n_pair_total, dtype=np.uint8)
    arrays["pair_bond"] = np.full(n_pair_total, -1, dtype=np.int32)
    arrays["ring_ptr"] = ring_ptr.astype(np.int64)
    arrays["ring_atom_ptr"] = np.zeros(n_systems + 1, dtype=np.int64)
    arrays["ring_pair_ptr"] = np.zeros(n_systems + 1, dtype=np.int64)
    arrays["ring_ext_ptr"] = np.zeros(n_ring_atoms + 1, dtype=np.int64)
    arrays["ring_n_edges"] = np.zeros(n_systems, dtype=np.int32)
    arrays["ring_atom"] = np.zeros(n_ring_atoms, dtype=np.int32)
    arrays["ring_pair_i"] = np.zeros(n_ring_pairs, dtype=np.int32)
    arrays["ring_pair_j"] = np.zeros(n_ring_pairs, dtype=np.int32)
    arrays["ring_pair_d"] = np.zeros(n_ring_pairs, dtype=np.uint8)
    arrays["ring_pair_bond"] = np.full(n_ring_pairs, -1, dtype=np.int32)
    arrays["ring_ext_atom"] = np.zeros(n_ext, dtype=np.int32)
    arrays["ring_ext_bond"] = np.zeros(n_ext, dtype=np.int32)
    np.cumsum(system_atoms, out=arrays["ring_atom_ptr"][1:])
    np.cumsum(system_pairs, out=arrays["ring_pair_ptr"][1:])
    np.cumsum(ring_atom_ext, out=arrays["ring_ext_ptr"][1:])
    arrays["ring_n_edges"][:] = np.asarray(system_edges, dtype=np.int32)

    # ---- pass 2: fill every packed object ----
    for k, row in enumerate(row_ids):
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        rec = enumerate_molecule(a1 - a0, np.asarray(bond_index[b0:b1], dtype=np.int64))
        nb_off = int(arrays["bond_nb_ptr"][int(row_bond_ptr[k])])
        pair_off = int(arrays["bond_pair_ptr"][int(row_bond_ptr[k])])
        for eid_local in range(b1 - b0):
            gid_local = int(row_bond_ptr[k]) + eid_local
            info = rec["neighbours"][eid_local]
            cursor = nb_off
            for atom, side, bond_side, bond_other in info["entries"]:
                arrays["nb_atom"][cursor] = int(atom)
                arrays["nb_side"][cursor] = int(side)
                arrays["nb_bond_side"][cursor] = int(bond_side)
                arrays["nb_bond_other"][cursor] = int(bond_other)
                cursor += 1
            if cursor != int(arrays["bond_nb_ptr"][gid_local + 1]):
                raise RuntimeError("neighbour pointer mismatch during fill")
            cursor = pair_off
            for nb1, nb2, kind, bond in rec["pairs"][eid_local]:
                arrays["pair_a"][cursor] = int(nb1)
                arrays["pair_b"][cursor] = int(nb2)
                arrays["pair_kind"][cursor] = int(kind)
                arrays["pair_bond"][cursor] = int(bond)
                cursor += 1
            if cursor != int(arrays["bond_pair_ptr"][gid_local + 1]):
                raise RuntimeError("pair pointer mismatch during fill")
            nb_off = int(arrays["bond_nb_ptr"][gid_local + 1])
            pair_off = int(arrays["bond_pair_ptr"][gid_local + 1])
        sys_off = int(arrays["ring_ptr"][k])
        atom_cursor = int(arrays["ring_atom_ptr"][sys_off])
        pair_cursor = int(arrays["ring_pair_ptr"][sys_off])
        ext_cursor = int(arrays["ring_ext_ptr"][atom_cursor])
        for system in rec["rings"]:
            atoms = system["atoms"]
            for pos, atom in enumerate(atoms):
                arrays["ring_atom"][atom_cursor + pos] = int(atom)
            for i, j, bucket, bond in system["pairs"]:
                arrays["ring_pair_i"][pair_cursor] = int(i)
                arrays["ring_pair_j"][pair_cursor] = int(j)
                arrays["ring_pair_d"][pair_cursor] = int(bucket)
                arrays["ring_pair_bond"][pair_cursor] = int(bond)
                pair_cursor += 1
            for atom, other, eid in system["external_entries"]:
                arrays["ring_ext_atom"][ext_cursor] = int(other)
                arrays["ring_ext_bond"][ext_cursor] = int(eid)
                ext_cursor += 1
            atom_cursor += len(atoms)
        if atom_cursor != int(arrays["ring_atom_ptr"][sys_off + len(rec["rings"])]):
            raise RuntimeError("ring atom pointer mismatch during fill")
        if pair_cursor != int(arrays["ring_pair_ptr"][sys_off + len(rec["rings"])]):
            raise RuntimeError("ring pair pointer mismatch during fill")
        if ext_cursor != int(arrays["ring_ext_ptr"][atom_cursor]):
            raise RuntimeError("ring external pointer mismatch during fill")
        if log_every and (k + 1) % int(log_every) == 0:
            C.log(f"[struct-cache] fill {k + 1}/{n_rows} rows ({time.perf_counter() - t0:.1f}s)")

    # ---- pointer / range invariants ----
    max_atoms = int(atom_counts[row_ids].max()) if n_rows else 0
    max_bonds = int(selected_bond_counts.max()) if n_rows else 0
    checks = {
        "bond_nb_ptr_monotone": bool(np.all(np.diff(arrays["bond_nb_ptr"]) >= 0)),
        "bond_pair_ptr_monotone": bool(np.all(np.diff(arrays["bond_pair_ptr"]) >= 0)),
        "ring_ptr_monotone": bool(np.all(np.diff(arrays["ring_ptr"]) >= 0)),
        "ring_atom_ptr_final": bool(int(arrays["ring_atom_ptr"][-1]) == n_ring_atoms),
        "ring_pair_ptr_final": bool(int(arrays["ring_pair_ptr"][-1]) == n_ring_pairs),
        "ring_ext_ptr_final": bool(int(arrays["ring_ext_ptr"][-1]) == n_ext),
        "bond_nb_total": bool(int(arrays["bond_nb_ptr"][-1]) == n_nb_total),
        "bond_pair_total": bool(int(arrays["bond_pair_ptr"][-1]) == n_pair_total),
        "nb_atom_range": bool(
            arrays["nb_atom"].size == 0
            or (arrays["nb_atom"].min() >= 0 and arrays["nb_atom"].max() < max_atoms)
        ),
        "nb_bond_range": bool(
            arrays["nb_bond_side"].size == 0
            or (arrays["nb_bond_side"].min() >= 0 and arrays["nb_bond_side"].max() < max_bonds)
        ),
        "pair_kind_range": bool(arrays["pair_kind"].max(initial=0) <= 2),
        "ring_pair_distance_range": bool(arrays["ring_pair_d"].max(initial=0) < C.N_RING_BUCKETS),
        "bond_map_shape": bool(bond_map.dtype == np.int32 and bond_map.size == int(bond_counts.sum())),
    }
    # exact bond-map check per selected row (cheap, deterministic)
    for k, row in enumerate(row_ids):
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        if b1 > b0 and not np.array_equal(
            bond_map[b0:b1],
            np.arange(int(row_bond_ptr[k]), int(row_bond_ptr[k + 1]), dtype=np.int32),
        ):
            checks["bond_map_selected"] = False
            break
    failed = [name for name, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"structure cache invariants failed: {failed}")

    # ---- deterministic independent reference on a sample ----
    gen = np.random.default_rng(int(reference_seed))
    n_ref = max(1, int(round(float(reference_fraction) * n_rows)))
    sample = np.sort(gen.choice(n_rows, size=min(n_ref, n_rows), replace=False))
    reference_mismatches = _verify_rows_against_reference(
        arrays, atom_ptr, bond_ptr, bond_index, row_ids, sample
    )
    if reference_mismatches:
        raise RuntimeError(f"structure cache reference audit found {reference_mismatches} mismatches")

    # ---- write arrays + metadata ----
    for name in ARRAY_NAMES:
        np.save(out_dir / f"{name}.npy", arrays[name])
    meta: dict[str, Any] = {
        "schema": C.SCHEMA_STRUCT_CACHE,
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "git_commit": C.git_commit(),
        "purpose": "bond-centred local objects + ring-system instances for LCRC-BOC",
        "labels_read": False,
        "label_artifact_opened": False,
        "training_scope": False,
        "rows": "all official rows" if rows is None else f"{n_rows} requested rows",
        "row_order": "official row order" if rows is None else "requested row order (compacted bonds)",
        "n_rows": n_rows,
        "row_ids_sha256": C.sha256_bytes(np.ascontiguousarray(row_ids, dtype=np.int64).tobytes()),
        "n_atoms": int(atom_counts[row_ids].sum()),
        "n_bonds": int(n_bonds_total),
        "n_bonds_source_total": int(bond_counts.sum()),
        "n_bond_neighbour_entries": int(n_nb_total),
        "n_bond_pair_entries": int(n_pair_total),
        "n_ring_systems": int(n_systems),
        "n_ring_atoms": int(n_ring_atoms),
        "n_ring_pairs": int(n_ring_pairs),
        "n_external_bonds": int(n_ext),
        "counts": {
            "nL": _histogram(arrays["bond_nL"]),
            "nR": _histogram(arrays["bond_nR"]),
            "nLL": _histogram(arrays["bond_nLL"]),
            "nRR": _histogram(arrays["bond_nRR"]),
            "nLR": _histogram(arrays["bond_nLR"]),
            "ring_system_size": _histogram(np.diff(arrays["ring_atom_ptr"])),
            "ring_pair_count": _histogram(np.diff(arrays["ring_pair_ptr"])),
            "ring_distance_bucket": _histogram(arrays["ring_pair_d"]),
            "external_per_ring_atom": _histogram(np.diff(arrays["ring_ext_ptr"])),
        },
        "maxima": {
            "nL": int(arrays["bond_nL"].max(initial=0)),
            "nR": int(arrays["bond_nR"].max(initial=0)),
            "nLL": int(arrays["bond_nLL"].max(initial=0)),
            "nRR": int(arrays["bond_nRR"].max(initial=0)),
            "nLR": int(arrays["bond_nLR"].max(initial=0)),
            "ring_system_size": int(np.diff(arrays["ring_atom_ptr"]).max(initial=0)),
            "ring_pairs": int(np.diff(arrays["ring_pair_ptr"]).max(initial=0)),
            "external_per_ring_atom": int(np.diff(arrays["ring_ext_ptr"]).max(initial=0)),
        },
        "storage": {
            "bond_nb_ptr": "int64 per compact physical bond (row-major selected bond order)",
            "bond_pair_ptr": "int64 per compact physical bond",
            "bond_map": "int32 global physical bond id -> compact slot (-1 for unselected)",
            "nb_*": "int32/uint8 flat neighbour entries, grouped per bond",
            "pair_*": "int32/uint8 flat pair entries, grouped per bond",
            "ring_*": "int64 pointers, int32/uint8 flat ring-system entries",
        },
        "invariants": checks,
        "reference_audit": {
            "fraction": float(reference_fraction),
            "n_rows": int(sample.size),
            "seed": int(reference_seed),
            "method": "independent Python enumeration + independent bridge-by-removal",
            "mismatches": int(reference_mismatches),
        },
        "build_seconds": float(time.perf_counter() - t0),
        "source": {
            "cache_path": C.track_rel(cache_path),
            "atom_ptr_sha256": C.sha256_file(cache_path / "atom_ptr.npy"),
            "bond_ptr_sha256": C.sha256_file(cache_path / "bond_ptr.npy"),
            "bond_index_sha256": C.sha256_file(cache_path / "bond_index.npy"),
            "feature_spec_hash": C.SOURCE["feature_spec_hash"],
            "split_partition_sha256": C.SOURCE["split_partition_sha256"],
        },
    }
    meta["array_sha256"] = {name: C.sha256_file(out_dir / f"{name}.npy") for name in ARRAY_NAMES}
    meta["cache_sha256"] = C.sha256_bytes(
        "|".join(f"{name}:{meta['array_sha256'][name]}" for name in ARRAY_NAMES).encode()
    )
    meta["cache_bytes"] = int(sum((out_dir / f"{name}.npy").stat().st_size for name in ARRAY_NAMES))
    C.write_json(out_dir / "meta.json", meta)
    C.log(
        f"[struct-cache] rows={n_rows} bonds={n_bonds_total} nb={n_nb_total} pairs={n_pair_total} "
        f"rings={n_systems} ring_atoms={n_ring_atoms} ring_pairs={n_ring_pairs} "
        f"seconds={meta['build_seconds']:.1f} bytes={meta['cache_bytes']}"
    )
    return meta


def _histogram(values: np.ndarray) -> dict[str, int]:
    v = np.asarray(values, dtype=np.int64)
    if v.size == 0:
        return {}
    uniq, counts = np.unique(v, return_counts=True)
    return {str(int(k)): int(c) for k, c in zip(uniq.tolist(), counts.tolist())}


# ---------------------------------------------------------------------------
# independent row verification (used by the build and the audit CLI)
# ---------------------------------------------------------------------------


def _verify_rows_against_reference(
    arrays: Mapping[str, np.ndarray],
    atom_ptr: np.ndarray,
    bond_ptr: np.ndarray,
    bond_index: np.ndarray,
    row_ids: np.ndarray,
    sample: Sequence[int],
) -> int:
    mismatches = 0
    row_bond_ptr = arrays["row_bond_ptr"]
    for pos in sample:
        k = int(pos)
        row = int(row_ids[k])
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        gb0, gb1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        lb0, lb1 = int(row_bond_ptr[k]), int(row_bond_ptr[k + 1])
        edges = np.asarray(bond_index[gb0:gb1], dtype=np.int64)
        if lb1 - lb0 != gb1 - gb0:
            mismatches += 1
            continue
        if lb1 > lb0 and not np.array_equal(
            arrays["bond_map"][gb0:gb1], np.arange(lb0, lb1, dtype=np.int32)
        ):
            mismatches += 1
        ref = enumerate_molecule(a1 - a0, edges)
        ref_bridge = reference_bridges_by_removal(a1 - a0, edges)
        if not np.array_equal(ref["bridges"], ref_bridge):
            mismatches += 1
        nb0, nb1 = int(arrays["bond_nb_ptr"][lb0]), int(arrays["bond_nb_ptr"][lb1])
        pa0, pa1 = int(arrays["bond_pair_ptr"][lb0]), int(arrays["bond_pair_ptr"][lb1])
        r0, r1 = int(arrays["ring_ptr"][k]), int(arrays["ring_ptr"][k + 1])
        if nb1 - nb0 != sum(len(info["entries"]) for info in ref["neighbours"]):
            mismatches += 1
        if pa1 - pa0 != sum(len(rows) for rows in ref["pairs"]):
            mismatches += 1
        expected_nb = [
            (int(atom), int(side), int(bond_side), int(bond_other))
            for info in ref["neighbours"]
            for atom, side, bond_side, bond_other in info["entries"]
        ]
        got_nb = [
            (
                int(arrays["nb_atom"][i]),
                int(arrays["nb_side"][i]),
                int(arrays["nb_bond_side"][i]),
                int(arrays["nb_bond_other"][i]),
            )
            for i in range(nb0, nb1)
        ]
        if expected_nb != got_nb:
            mismatches += 1
        expected_pairs = [
            (int(a), int(b), int(kind), int(bond))
            for rows in ref["pairs"]
            for a, b, kind, bond in rows
        ]
        got_pairs = [
            (
                int(arrays["pair_a"][i]),
                int(arrays["pair_b"][i]),
                int(arrays["pair_kind"][i]),
                int(arrays["pair_bond"][i]),
            )
            for i in range(pa0, pa1)
        ]
        if expected_pairs != got_pairs:
            mismatches += 1
        got_rings = []
        for g in range(r0, r1):
            aa0, aa1 = int(arrays["ring_atom_ptr"][g]), int(arrays["ring_atom_ptr"][g + 1])
            pp0, pp1 = int(arrays["ring_pair_ptr"][g]), int(arrays["ring_pair_ptr"][g + 1])
            got_rings.append(
                {
                    "atoms": [int(x) for x in arrays["ring_atom"][aa0:aa1]],
                    "pairs": [
                        (
                            int(arrays["ring_pair_i"][i]),
                            int(arrays["ring_pair_j"][i]),
                            int(arrays["ring_pair_d"][i]),
                            int(arrays["ring_pair_bond"][i]),
                        )
                        for i in range(pp0, pp1)
                    ],
                    "n_edges": int(arrays["ring_n_edges"][g]),
                }
            )
            ext0, ext1 = int(arrays["ring_ext_ptr"][aa0]), int(arrays["ring_ext_ptr"][aa1])
            got_ext = [
                (int(arrays["ring_ext_atom"][i]), int(arrays["ring_ext_bond"][i]))
                for i in range(ext0, ext1)
            ]
            ring_atoms = {int(x) for x in arrays["ring_atom"][aa0:aa1]}
            expected_ext: list[tuple[int, int]] = []
            for atom in sorted(ring_atoms):
                for other, eid in ref["adjacency"][atom]:
                    if int(other) not in ring_atoms:
                        expected_ext.append((int(other), int(eid)))
            if got_ext != expected_ext:
                mismatches += 1
        expected_rings = [
            {
                "atoms": [int(x) for x in system["atoms"]],
                "pairs": [
                    (int(i), int(j), int(bucket), int(bond))
                    for i, j, bucket, bond in system["pairs"]
                ],
                "n_edges": int(system["n_edges"]),
            }
            for system in ref["rings"]
        ]
        if got_rings != expected_rings:
            mismatches += 1
    return int(mismatches)


def load_cache(path: Path | str = C.STRUCT_CACHE_DIR, *, verify: bool = True) -> dict[str, Any]:
    path = Path(path)
    meta_path = path / "meta.json"
    if not meta_path.is_file():
        raise RuntimeError(f"structure cache missing: {meta_path}")
    meta = C.read_json(meta_path)
    if meta.get("schema") != C.SCHEMA_STRUCT_CACHE:
        raise RuntimeError(f"structure cache schema {meta.get('schema')}")
    if meta.get("labels_read") is not False:
        raise RuntimeError("structure cache claims labels were read")
    arrays: dict[str, np.ndarray] = {}
    for name in ARRAY_NAMES:
        p = path / f"{name}.npy"
        if not p.is_file():
            raise RuntimeError(f"structure cache array missing: {p}")
        if verify and C.sha256_file(p) != meta["array_sha256"][name]:
            raise RuntimeError(f"structure cache array hash mismatch: {name}")
        arrays[name] = np.load(p)
    if verify:
        digest = C.sha256_bytes(
            "|".join(f"{name}:{meta['array_sha256'][name]}" for name in ARRAY_NAMES).encode()
        )
        if digest != meta["cache_sha256"]:
            raise RuntimeError("structure cache aggregate hash mismatch")
    arrays["meta"] = meta
    arrays["dir"] = C.track_rel(path)
    return arrays


def cache_identity(cache: Mapping[str, Any]) -> dict[str, Any]:
    meta = cache["meta"]
    return {
        "dir": cache.get("dir"),
        "cache_sha256": meta["cache_sha256"],
        "n_rows": int(meta["n_rows"]),
        "row_ids_sha256": meta["row_ids_sha256"],
        "n_bonds": int(meta["n_bonds"]),
        "n_bond_neighbour_entries": int(meta["n_bond_neighbour_entries"]),
        "n_bond_pair_entries": int(meta["n_bond_pair_entries"]),
        "n_ring_systems": int(meta["n_ring_systems"]),
        "n_ring_atoms": int(meta["n_ring_atoms"]),
        "n_ring_pairs": int(meta["n_ring_pairs"]),
        "n_external_bonds": int(meta["n_external_bonds"]),
        "labels_read": bool(meta["labels_read"]),
        "build_seconds": float(meta.get("build_seconds", 0.0)),
        "cache_bytes": int(meta.get("cache_bytes", 0)),
    }


# ---------------------------------------------------------------------------
# vectorised batch assembly
# ---------------------------------------------------------------------------


def expand_ranges(starts: np.ndarray, counts: np.ndarray) -> tuple[np.ndarray, np.ndarray, int]:
    """Concatenate ``[start, start+count)`` ranges without a Python loop."""
    starts = np.asarray(starts, dtype=np.int64)
    counts = np.asarray(counts, dtype=np.int64)
    if starts.shape != counts.shape:
        raise ValueError("starts/counts shape mismatch")
    total = int(counts.sum())
    if total == 0:
        return np.zeros(0, dtype=np.int64), np.zeros(0, dtype=np.int64), 0
    owner = np.repeat(np.arange(starts.size, dtype=np.int64), counts)
    within_base = np.repeat(np.cumsum(counts) - counts, counts)
    idx = np.repeat(starts, counts) + (np.arange(total, dtype=np.int64) - within_base)
    return idx, owner, total


def collate_boc(task: Any, struct: Mapping[str, Any], rows: Sequence[int]) -> dict[str, Any]:
    """Inherited W2 collate + packed BOC structure for the same rows."""
    import torch

    rows_np = np.asarray(rows, dtype=np.int64)
    batch = C.BC.PC.collate(task.cache, rows_np.tolist())
    # Local offsets from the inherited batch itself (already validated).
    atom_off = np.asarray(batch["atom_off"], dtype=np.int64)
    atom_counts = np.asarray(batch["atom_counts"], dtype=np.int64)
    bond_counts = np.asarray(batch["bond_counts"], dtype=np.int64)
    bond_batch = np.asarray(batch["bond_batch"], dtype=np.int64)
    n_graphs = int(batch["n_graphs"])
    bond_off = np.cumsum(bond_counts) - bond_counts
    # The inherited collate exposes no global bond index, so rebuild the exact
    # row-major global order from the frozen pointers.
    cache_path = Path(str(task.cache_path))
    bond_ptr = np.load(cache_path / "bond_ptr.npy")
    bond_index_cache = np.load(cache_path / "bond_index.npy", mmap_mode="r")
    bond_start = bond_ptr[rows_np].astype(np.int64)
    bond_idx = expand_ranges(bond_start, bond_counts)[0]
    if bond_idx.size != int(bond_counts.sum()):
        raise RuntimeError("global bond index reconstruction failed")
    bond_map = struct.get("bond_map")
    if bond_map is None:
        local_bond = bond_idx
    else:
        local_bond = np.asarray(bond_map[bond_idx], dtype=np.int64)
        if local_bond.size and int(local_bond.min()) < 0:
            raise RuntimeError("a batch bond is absent from the compact structure cache")
    nb_ptr = struct["bond_nb_ptr"]
    pair_ptr = struct["bond_pair_ptr"]
    nb_starts = nb_ptr[local_bond].astype(np.int64)
    nb_counts = (nb_ptr[local_bond + 1] - nb_ptr[local_bond]).astype(np.int64)
    pair_starts = pair_ptr[local_bond].astype(np.int64)
    pair_counts = (pair_ptr[local_bond + 1] - pair_ptr[local_bond]).astype(np.int64)
    nb_index, nb_owner_bond, nb_total = expand_ranges(nb_starts, nb_counts)
    pair_index, pair_owner_bond, pair_total = expand_ranges(pair_starts, pair_counts)
    nb_mol = bond_batch[nb_owner_bond]
    pair_mol = bond_batch[pair_owner_bond]
    nb_ptr_batch = np.zeros(int(nb_counts.size) + 1, dtype=np.int64)
    np.cumsum(nb_counts, out=nb_ptr_batch[1:])

    nb_atom = struct["nb_atom"][nb_index].astype(np.int64) + atom_off[nb_mol]
    nb_side = struct["nb_side"][nb_index].astype(np.int64)
    nb_bond_side = struct["nb_bond_side"][nb_index].astype(np.int64) + bond_off[nb_mol]
    nb_bond_other_local = struct["nb_bond_other"][nb_index].astype(np.int64)
    nb_bond_other = np.where(nb_bond_other_local >= 0, nb_bond_other_local + bond_off[nb_mol], -1)

    pair_a = struct["pair_a"][pair_index].astype(np.int64) + nb_ptr_batch[pair_owner_bond]
    pair_b = struct["pair_b"][pair_index].astype(np.int64) + nb_ptr_batch[pair_owner_bond]
    pair_kind = struct["pair_kind"][pair_index].astype(np.int64)
    pair_bond_local = struct["pair_bond"][pair_index].astype(np.int64)
    pair_bond = np.where(pair_bond_local >= 0, pair_bond_local + bond_off[pair_mol], -1)

    ring_ptr = struct["ring_ptr"]
    sys_start = ring_ptr[rows_np].astype(np.int64)
    sys_counts = (ring_ptr[rows_np + 1] - ring_ptr[rows_np]).astype(np.int64)
    sys_index, sys_mol, n_systems = expand_ranges(sys_start, sys_counts)
    ring_ptr_batch = np.zeros(n_graphs + 1, dtype=np.int64)
    np.cumsum(sys_counts, out=ring_ptr_batch[1:])

    ring_atom_ptr = struct["ring_atom_ptr"]
    ra_starts = ring_atom_ptr[sys_index].astype(np.int64)
    ra_counts = (ring_atom_ptr[sys_index + 1] - ring_atom_ptr[sys_index]).astype(np.int64)
    ra_index, ra_system, n_ring_atoms = expand_ranges(ra_starts, ra_counts)
    if n_systems:
        ring_atom_off_system = np.concatenate([[0], np.cumsum(ra_counts)[:-1]]).astype(np.int64)
    else:
        ring_atom_off_system = np.zeros(0, dtype=np.int64)
    ring_atom_batch = struct["ring_atom"][ra_index].astype(np.int64) + atom_off[sys_mol[ra_system]]

    ring_pair_ptr = struct["ring_pair_ptr"]
    rp_starts = ring_pair_ptr[sys_index].astype(np.int64)
    rp_counts = (ring_pair_ptr[sys_index + 1] - ring_pair_ptr[sys_index]).astype(np.int64)
    rp_index, rp_system, n_ring_pairs = expand_ranges(rp_starts, rp_counts)
    ring_pair_i = struct["ring_pair_i"][rp_index].astype(np.int64) + ring_atom_off_system[rp_system]
    ring_pair_j = struct["ring_pair_j"][rp_index].astype(np.int64) + ring_atom_off_system[rp_system]
    ring_pair_d = struct["ring_pair_d"][rp_index].astype(np.int64)
    ring_pair_bond_local = struct["ring_pair_bond"][rp_index].astype(np.int64)
    ring_pair_bond = np.where(
        ring_pair_bond_local >= 0, ring_pair_bond_local + bond_off[sys_mol[rp_system]], -1
    )

    ring_ext_ptr = struct["ring_ext_ptr"]
    ext_starts = ring_ext_ptr[ra_index].astype(np.int64)
    ext_counts = (ring_ext_ptr[ra_index + 1] - ring_ext_ptr[ra_index]).astype(np.int64)
    ext_index, ext_owner_ra, n_ext = expand_ranges(ext_starts, ext_counts)
    ext_mol = sys_mol[ra_system[ext_owner_ra]]
    ring_ext_atom = struct["ring_ext_atom"][ext_index].astype(np.int64) + atom_off[ext_mol]
    ring_ext_bond = struct["ring_ext_bond"][ext_index].astype(np.int64) + bond_off[ext_mol]

    ring_ext_counts_per_atom = ext_counts
    ring_n_ext = np.bincount(ra_system, weights=ring_ext_counts_per_atom, minlength=n_systems).astype(np.int64)
    ring_n_edges = struct["ring_n_edges"][sys_index].astype(np.int64)
    ring_pair_counts = rp_counts.astype(np.int64)
    ring_atom_counts = ra_counts.astype(np.int64)
    ring_atom_system = ra_system.astype(np.int64)
    ring_pair_system = rp_system.astype(np.int64)

    # range checks before the tensors leave the CPU
    checks = {
        "nb_owner_in_range": bool(nb_total == 0 or (nb_owner_bond.size == nb_total and nb_index.max() < struct["nb_atom"].size)),
        "pair_owner_in_range": bool(pair_total == 0 or pair_index.max() < struct["pair_a"].size),
        "pair_nb_in_range": bool(pair_total == 0 or (pair_a.min() >= 0 and pair_a.max() < max(nb_total, 1) and pair_b.max() < max(nb_total, 1))),
        "ring_pair_index_in_range": bool(n_ring_pairs == 0 or (ring_pair_i.min() >= 0 and ring_pair_i.max() < max(n_ring_atoms, 1) and ring_pair_j.max() < max(n_ring_atoms, 1))),
        "ring_ext_owner_in_range": bool(n_ext == 0 or (ext_owner_ra.min() >= 0 and ext_owner_ra.max() < max(n_ring_atoms, 1))),
    }
    failed = [k for k, ok in checks.items() if not ok]
    if failed:
        raise RuntimeError(f"BOC batch range checks failed: {failed}")

    def t(arr: np.ndarray, dtype=np.int64) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr, dtype=dtype))

    batch.update(
        {
            "boc_nb_ptr": t(nb_ptr_batch, np.int64),
            "boc_nb_atom": t(nb_atom, np.int64),
            "boc_nb_side": t(nb_side, np.int64),
            "boc_nb_bond_side": t(nb_bond_side, np.int64),
            "boc_nb_bond_other": t(nb_bond_other, np.int64),
            "boc_pair_ptr": t(np.concatenate([[0], np.cumsum(pair_counts)]), np.int64),
            "boc_pair_a": t(pair_a, np.int64),
            "boc_pair_b": t(pair_b, np.int64),
            "boc_pair_kind": t(pair_kind, np.int64),
            "boc_pair_bond": t(pair_bond, np.int64),
            "boc_nL": t(struct["bond_nL"][local_bond], np.int64),
            "boc_nR": t(struct["bond_nR"][local_bond], np.int64),
            "boc_nLL": t(struct["bond_nLL"][local_bond], np.int64),
            "boc_nRR": t(struct["bond_nRR"][local_bond], np.int64),
            "boc_nLR": t(struct["bond_nLR"][local_bond], np.int64),
            "boc_n_rings": int(n_systems),
            "boc_ring_ptr": t(ring_ptr_batch, np.int64),
            "boc_ring_atom": t(ring_atom_batch, np.int64),
            "boc_ring_atom_counts": t(ring_atom_counts, np.int64),
            "boc_ring_atom_system": t(ring_atom_system, np.int64),
            "boc_ring_pair_ptr": t(
                np.concatenate([[0], np.cumsum(ring_pair_counts)]) if n_systems else np.zeros(1, dtype=np.int64),
                np.int64,
            ),
            "boc_ring_pair_i": t(ring_pair_i, np.int64),
            "boc_ring_pair_j": t(ring_pair_j, np.int64),
            "boc_ring_pair_d": t(ring_pair_d, np.int64),
            "boc_ring_pair_bond": t(ring_pair_bond, np.int64),
            "boc_ring_pair_system": t(ring_pair_system, np.int64),
            "boc_ring_ext_ptr": t(np.concatenate([[0], np.cumsum(ext_counts)]) if n_ring_atoms else np.zeros(1, dtype=np.int64), np.int64),
            "boc_ring_ext_atom": t(ring_ext_atom, np.int64),
            "boc_ring_ext_bond": t(ring_ext_bond, np.int64),
            "boc_ring_n_edges": t(ring_n_edges, np.int64),
            "boc_ring_n_ext": t(ring_n_ext, np.int64),
            "boc_n_bonds": int(bond_counts.sum()),
            "boc_n_atoms": int(atom_counts.sum()),
            "boc_n_nb": int(nb_total),
            "boc_n_pairs": int(pair_total),
            "boc_n_ring_atoms": int(n_ring_atoms),
            "boc_n_ring_pairs": int(n_ring_pairs),
            "boc_n_ext": int(n_ext),
            "boc_bond_global_index": t(bond_idx, np.int64),
        }
    )
    return batch


def main(argv: Sequence[str] | None = None) -> int:
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.STRUCT_CACHE_DIR))
    parser.add_argument("--reference-fraction", type=float, default=0.001)
    parser.add_argument("--log-every", type=int, default=50_000)
    args = parser.parse_args(argv)
    meta = build_cache(args.out, reference_fraction=args.reference_fraction, log_every=args.log_every)
    print(json.dumps({k: v for k, v in meta.items() if k not in {"array_sha256", "counts", "maxima"}}, indent=1, default=str))
    return 0


__all__ = [
    "ARRAY_NAMES",
    "build_cache",
    "cache_identity",
    "collate_boc",
    "distance_bucket",
    "enumerate_molecule",
    "expand_ranges",
    "find_bridges",
    "load_cache",
    "reference_bridges_by_removal",
]


if __name__ == "__main__":
    raise SystemExit(main())

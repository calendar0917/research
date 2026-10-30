"""Canonical molecular graph and deterministic graph algorithms.

This module is feature-spec agnostic.  It turns a directed/duplicated bond list
into one canonical physical-bond list (``u < v``, sorted) and computes the
structural primitives used by both schemas:

* adjacency, degree, connected components;
* all-pairs unweighted shortest paths with an explicit unreachable sentinel;
* biconnected blocks, ring bonds, ring atoms, ring systems;
* the smallest cycle through each atom;
* non-backtracking 2-path enumeration (the S2 universe);
* the legacy distance/block relation pairs (used only by the ZINC replay
  check; LCRC-A does not need them).

No labels, no learned quantities and no dataset statistics enter this module.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np

UNREACHABLE = -1


# ---------------------------------------------------------------------------
# canonical physical bonds
# ---------------------------------------------------------------------------


def canonicalize_physical_bonds(
    n_atoms: int,
    bond_index: Sequence[Sequence[int]] | np.ndarray,
    bond_type: Sequence[int] | np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Deduplicate directed bond records into sorted physical bonds.

    ``bond_index`` may contain each physical bond once or twice (both
    directions).  Duplicate records must agree on the bond type.  The returned
    arrays have ``u < v`` and are sorted by ``(u, v)``.
    """
    bi = np.asarray(bond_index, dtype=np.int64)
    if bi.ndim == 1:
        if bi.size == 0:
            bi = bi.reshape(0, 2)
        else:
            raise ValueError("bond_index must be [M, 2] or [2, M]")
    if bi.ndim != 2:
        raise ValueError("bond_index must be 2-D")
    if bi.shape[1] == 2:
        pass
    elif bi.shape[0] == 2:
        bi = bi.T
    else:
        raise ValueError(f"bond_index has shape {bi.shape}; expected [M, 2] or [2, M]")
    bt = np.asarray(bond_type, dtype=np.int64).reshape(-1)
    if bt.shape[0] != bi.shape[0]:
        raise ValueError(f"bond_type length {bt.shape[0]} != bond_index length {bi.shape[0]}")
    if np.any(bi < 0) or np.any(bi >= int(n_atoms)):
        raise ValueError("bond endpoint outside atom range")
    if np.any(bi[:, 0] == bi[:, 1]):
        raise ValueError("self-loop in molecular graph")

    seen: dict[tuple[int, int], int] = {}
    for (u, v), t in zip(bi.tolist(), bt.tolist()):
        key = (min(int(u), int(v)), max(int(u), int(v)))
        if key in seen and seen[key] != int(t):
            raise ValueError(f"inconsistent directed bond records for {key}: {seen[key]} vs {int(t)}")
        seen[key] = int(t)
    keys = sorted(seen)
    u = np.asarray([k[0] for k in keys], dtype=np.int64)
    v = np.asarray([k[1] for k in keys], dtype=np.int64)
    t = np.asarray([seen[k] for k in keys], dtype=np.int64)
    return u, v, t


def canonicalize_bonds_with_features(
    n_atoms: int,
    bond_index: Sequence[Sequence[int]] | np.ndarray,
    bond_type: Sequence[int] | np.ndarray,
    bond_feat: np.ndarray,
    bond_stereo: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray | None]:
    """Canonicalise directed/physical bonds while keeping row features aligned.

    ``bond_feat`` (and optional ``bond_stereo``) must have one row per input
    bond record.  Duplicate directed records are allowed only when their bond
    type and feature rows agree.  The returned physical bonds are sorted by
    ``(u, v)`` and the feature rows follow that order.
    """
    bi = np.asarray(bond_index, dtype=np.int64)
    if bi.size == 0:
        bi = bi.reshape(0, 2)
    if bi.ndim != 2:
        raise ValueError("bond_index must be 2-D")
    if bi.shape[1] != 2:
        if bi.shape[0] == 2:
            bi = bi.T
        else:
            raise ValueError(f"bond_index has shape {bi.shape}; expected [M, 2] or [2, M]")
    bt = np.asarray(bond_type, dtype=np.int64).reshape(-1)
    bf = np.asarray(bond_feat, dtype=np.float32)
    if bf.ndim != 2 or bf.shape[0] != bi.shape[0]:
        raise ValueError(f"bond_feat shape {bf.shape} does not match {bi.shape[0]} bond records")
    if bt.shape[0] != bi.shape[0]:
        raise ValueError(f"bond_type length {bt.shape[0]} != {bi.shape[0]}")
    if bond_stereo is None:
        bs_in = None
    else:
        bs_in = np.asarray(bond_stereo, dtype=np.int64).reshape(-1)
        if bs_in.shape[0] != bi.shape[0]:
            raise ValueError(f"bond_stereo length {bs_in.shape[0]} != {bi.shape[0]}")
    if np.any(bi < 0) or np.any(bi >= int(n_atoms)):
        raise ValueError("bond endpoint outside atom range")
    if np.any(bi[:, 0] == bi[:, 1]):
        raise ValueError("self-loop in molecular graph")
    seen: dict[tuple[int, int], tuple[int, np.ndarray, int | None]] = {}
    for k, ((u, v), t) in enumerate(zip(bi.tolist(), bt.tolist())):
        key = (min(int(u), int(v)), max(int(u), int(v)))
        st = None if bs_in is None else int(bs_in[k])
        if key in seen:
            old_t, old_row, old_st = seen[key]
            if old_t != int(t) or not np.array_equal(old_row, bf[k]) or old_st != st:
                raise ValueError(f"inconsistent duplicate bond records for {key}")
        else:
            seen[key] = (int(t), bf[k].copy(), st)
    keys = sorted(seen)
    u = np.asarray([k[0] for k in keys], dtype=np.int64)
    v = np.asarray([k[1] for k in keys], dtype=np.int64)
    t_out = np.asarray([seen[k][0] for k in keys], dtype=np.int64)
    f_out = (
        np.stack([seen[k][1] for k in keys], axis=0).astype(np.float32)
        if keys
        else np.zeros((0, bf.shape[1]), dtype=np.float32)
    )
    if bs_in is None:
        st_out = None
    else:
        st_out = np.asarray([seen[k][2] for k in keys], dtype=np.int64)
    return u, v, t_out, f_out, st_out


def adjacency(
    n_atoms: int,
    bond_u: np.ndarray,
    bond_v: np.ndarray,
    bond_type: np.ndarray,
) -> tuple[tuple[tuple[int, int, int], ...], ...]:
    """Sorted per-atom adjacency: ``(neighbour, bond_type, physical_bond_id)``."""
    adj: list[list[tuple[int, int, int]]] = [[] for _ in range(int(n_atoms))]
    for e, (u, v, t) in enumerate(zip(bond_u.tolist(), bond_v.tolist(), bond_type.tolist())):
        adj[int(u)].append((int(v), int(t), int(e)))
        adj[int(v)].append((int(u), int(t), int(e)))
    for i in range(int(n_atoms)):
        adj[i].sort()
    return tuple(tuple(row) for row in adj)


# ---------------------------------------------------------------------------
# graph primitives
# ---------------------------------------------------------------------------


def degrees(n_atoms: int, bond_u: np.ndarray, bond_v: np.ndarray) -> np.ndarray:
    deg = np.zeros(int(n_atoms), dtype=np.int64)
    if len(bond_u):
        np.add.at(deg, bond_u, 1)
        np.add.at(deg, bond_v, 1)
    return deg


def connected_components(
    n_atoms: int,
    bond_u: np.ndarray,
    bond_v: np.ndarray,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Return component id per atom, number of components, component size per atom."""
    n = int(n_atoms)
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for u, v in zip(bond_u.tolist(), bond_v.tolist()):
        ru, rv = find(int(u)), find(int(v))
        if ru != rv:
            parent[ru] = rv
    roots = [find(i) for i in range(n)]
    unique = {r: i for i, r in enumerate(sorted(set(roots)))}
    comp = np.asarray([unique[r] for r in roots], dtype=np.int64)
    counts = np.bincount(comp, minlength=len(unique)).astype(np.int64) if n else np.zeros(0, np.int64)
    comp_size = counts[comp] if n else np.zeros(0, np.int64)
    return comp, int(len(unique)), comp_size


def all_pairs_shortest_paths(n_atoms: int, adj: Sequence[Sequence[tuple[int, int, int]]]) -> np.ndarray:
    """Unweighted all-pairs distances; unreachable pairs are ``-1``.

    The explicit ``-1`` sentinel prevents disconnected components from being
    mistaken for a large finite distance.
    """
    n = int(n_atoms)
    dist = np.full((n, n), UNREACHABLE, dtype=np.int64)
    for s in range(n):
        dist[s, s] = 0
        frontier = [s]
        d = 0
        while frontier:
            d += 1
            nxt: list[int] = []
            for u in frontier:
                for v, _bt, _e in adj[u]:
                    if dist[s, v] == UNREACHABLE:
                        dist[s, v] = d
                        nxt.append(int(v))
            frontier = nxt
    return dist


def biconnected_ring_analysis(
    n_atoms: int,
    bond_u: np.ndarray,
    bond_v: np.ndarray,
) -> dict[str, Any]:
    """Biconnected-block ring analysis (same definition as the frozen P3 code).

    A bond is a ring bond iff its biconnected block has at least three edges.
    An atom is a ring atom iff it has cycle degree >= 2 in such blocks.  A ring
    system is one such block (fused/spiro systems are counted as one block, the
    historical P3 convention).
    """
    import networkx as nx

    n = int(n_atoms)
    graph = nx.Graph()
    graph.add_nodes_from(range(n))
    graph.add_edges_from((int(u), int(v)) for u, v in zip(bond_u.tolist(), bond_v.tolist()))
    bond_key = {(int(u), int(v)): i for i, (u, v) in enumerate(zip(bond_u.tolist(), bond_v.tolist()))}
    is_ring_bond = np.zeros(len(bond_u), dtype=bool)
    cycle_degree = np.zeros(n, dtype=np.int64)
    ring_bond_count = np.zeros(n, dtype=np.int64)
    ring_systems = 0
    ring_block_membership: list[set[int]] = [set() for _ in range(n)]
    block_records: list[dict[str, Any]] = []
    for block_id, nodes in enumerate(nx.biconnected_components(graph)):
        nodes_sorted = sorted(int(x) for x in nodes)
        node_set = set(nodes_sorted)
        edges = [
            (int(u), int(v))
            for (u, v) in bond_key
            if int(u) in node_set and int(v) in node_set
        ]
        is_ring_block = len(edges) >= 3
        if is_ring_block:
            ring_systems += 1
            for u, v in edges:
                is_ring_bond[bond_key[(u, v)]] = True
                cycle_degree[u] += 1
                cycle_degree[v] += 1
                ring_bond_count[u] += 1
                ring_bond_count[v] += 1
            for atom in nodes_sorted:
                ring_block_membership[atom].add(int(block_id))
        block_records.append(
            {
                "block_id": int(block_id),
                "n_atoms": len(nodes_sorted),
                "n_edges": len(edges),
                "is_ring_block": bool(is_ring_block),
                "atoms": nodes_sorted,
            }
        )
    ring_atom = cycle_degree >= 2
    return {
        "is_ring_bond": is_ring_bond,
        "ring_atom": ring_atom,
        "cycle_degree": cycle_degree,
        "ring_bond_count": ring_bond_count,
        "n_ring_systems": int(ring_systems),
        "ring_block_membership": ring_block_membership,
        "block_records": block_records,
    }


def min_ring_size_per_atom(
    n_atoms: int,
    adj: Sequence[Sequence[tuple[int, int, int]]],
) -> np.ndarray:
    """Smallest cycle through each atom (0 if acyclic).

    For atom ``a``, consider each pair of distinct neighbours; the shortest path
    between them that avoids ``a`` plus the two incident bonds is a cycle.
    """
    n = int(n_atoms)
    out = np.zeros(n, dtype=np.int64)
    for a in range(n):
        nbrs = [int(v) for v, _bt, _e in adj[a]]
        if len(nbrs) < 2:
            continue
        best = 0
        for x in range(len(nbrs)):
            for y in range(x + 1, len(nbrs)):
                s, t = nbrs[x], nbrs[y]
                seen = np.full(n, -1, dtype=np.int64)
                seen[a] = -2
                seen[s] = 0
                frontier = [s]
                found = False
                while frontier and not found:
                    nxt: list[int] = []
                    for u in frontier:
                        for v, _bt, _e in adj[u]:
                            if seen[v] != -1:
                                continue
                            seen[v] = seen[u] + 1
                            if v == t:
                                found = True
                                break
                            nxt.append(int(v))
                        if found:
                            break
                    frontier = nxt
                if found:
                    size = int(seen[t]) + 2
                    if best == 0 or size < best:
                        best = size
        out[a] = best
    return out


def junction_count(degree: np.ndarray) -> int:
    d = np.asarray(degree, dtype=np.int64)
    return int(np.sum(d * (d - 1) // 2))


def dist_bucket(distance: int) -> int:
    if distance == 3:
        return 0
    if 4 <= distance <= 5:
        return 1
    return 2


def relation_pairs(
    n_atoms: int,
    dist: np.ndarray,
    ring_block_membership: Sequence[set[int]],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Legacy R/N distance/block pair categories (distance >= 3 only).

    Unreachable pairs are rejected: this definition is used only for the
    connected-only legacy replay.  LCRC-A does not consume these arrays.
    """
    n = int(n_atoms)
    pi: list[int] = []
    pj: list[int] = []
    pc: list[int] = []
    for i in range(n):
        for j in range(i + 1, n):
            d = int(dist[i, j])
            if d == UNREACHABLE:
                raise ValueError("relation pairs are undefined for unreachable pairs")
            if d < 3:
                continue
            ring = 1 if (ring_block_membership[i] & ring_block_membership[j]) else 0
            pi.append(i)
            pj.append(j)
            pc.append(dist_bucket(d) * 2 + ring)
    return (
        np.asarray(pi, dtype=np.int64),
        np.asarray(pj, dtype=np.int64),
        np.asarray(pc, dtype=np.int8),
    )


# ---------------------------------------------------------------------------
# container
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class MolecularGraph:
    """Canonical structure plus encoded atom/bond feature matrices.

    ``global_feat`` is set only for the legacy schema, where the frozen P3
    global33 vector is an input.  ``raw_atom_charge`` / ``raw_atom_h`` /
    ``bond_stereo`` are used by the lipo_v1 global/feature builder and are
    ``None`` for legacy.
    """

    atom_type: np.ndarray
    bond_u: np.ndarray
    bond_v: np.ndarray
    bond_type: np.ndarray
    atom_feat: np.ndarray
    bond_feat: np.ndarray
    global_feat: np.ndarray | None = None
    n_components: int = 0
    molecule_id: str = ""
    smiles: str = ""
    raw_atom_charge: np.ndarray | None = None
    raw_atom_h: np.ndarray | None = None
    bond_stereo: np.ndarray | None = None
    raw_metadata: dict[str, Any] = field(default_factory=dict)

    @property
    def n_atoms(self) -> int:
        return int(self.atom_type.shape[0])

    @property
    def n_bonds(self) -> int:
        return int(self.bond_u.shape[0])


@dataclass(frozen=True)
class GraphAnalysis:
    n_atoms: int
    n_bonds: int
    bond_u: np.ndarray
    bond_v: np.ndarray
    degree: np.ndarray
    adj: tuple[tuple[tuple[int, int, int], ...], ...]
    dist: np.ndarray
    component_id: np.ndarray
    component_size: np.ndarray
    n_components: int
    n_junctions: int
    is_ring_bond: np.ndarray
    ring_atom: np.ndarray
    ring_bond_count: np.ndarray
    min_ring_size: np.ndarray
    n_ring_systems: int
    ring_block_membership: list[set[int]]
    block_records: list[dict[str, Any]]


def analyse_graph(graph: MolecularGraph) -> GraphAnalysis:
    n = graph.n_atoms
    m = graph.n_bonds
    if n < 1:
        raise ValueError("empty molecule is not supported")
    if graph.atom_type.shape != (n,):
        raise ValueError("atom_type shape mismatch")
    if graph.bond_type.shape != (m,):
        raise ValueError("bond_type shape mismatch")
    if graph.atom_feat.shape[0] != n:
        raise ValueError("atom_feat row count mismatch")
    if graph.bond_feat.shape[0] != m:
        raise ValueError("bond_feat row count mismatch")
    ug = np.asarray(graph.bond_u, dtype=np.int64)
    vg = np.asarray(graph.bond_v, dtype=np.int64)
    if np.any(ug >= vg):
        raise ValueError("physical bonds must satisfy u < v")
    if len(ug) > 1 and np.any(np.diff(ug * (n + 1) + vg) <= 0):
        raise ValueError("physical bonds must be sorted by (u, v)")
    adj = adjacency(n, ug, vg, graph.bond_type)
    deg = degrees(n, ug, vg)
    comp, n_comp, comp_size = connected_components(n, ug, vg)
    dist = all_pairs_shortest_paths(n, adj)
    ring = biconnected_ring_analysis(n, ug, vg)
    min_ring = min_ring_size_per_atom(n, adj)
    return GraphAnalysis(
        n_atoms=n,
        n_bonds=m,
        bond_u=ug.copy(),
        bond_v=vg.copy(),
        degree=deg,
        adj=adj,
        dist=dist,
        component_id=comp,
        component_size=comp_size,
        n_components=n_comp,
        n_junctions=junction_count(deg),
        is_ring_bond=ring["is_ring_bond"],
        ring_atom=ring["ring_atom"],
        ring_bond_count=ring["ring_bond_count"],
        min_ring_size=min_ring,
        n_ring_systems=ring["n_ring_systems"],
        ring_block_membership=ring["ring_block_membership"],
        block_records=ring["block_records"],
    )


def build_molecular_graph(
    *,
    atom_type: np.ndarray,
    bond_index: Sequence[Sequence[int]] | np.ndarray,
    bond_type: Sequence[int] | np.ndarray,
    atom_feat: np.ndarray,
    bond_feat: np.ndarray,
    global_feat: np.ndarray | None = None,
    molecule_id: str = "",
    smiles: str = "",
    raw_atom_charge: np.ndarray | None = None,
    raw_atom_h: np.ndarray | None = None,
    bond_stereo: np.ndarray | None = None,
    raw_metadata: dict[str, Any] | None = None,
) -> MolecularGraph:
    """Canonicalise physical bonds, keeping feature rows aligned, and validate."""
    at = np.asarray(atom_type, dtype=np.int64).reshape(-1)
    n = int(at.shape[0])
    if n < 1:
        raise ValueError("empty molecule is not supported")
    af = np.asarray(atom_feat, dtype=np.float32)
    if af.shape[0] != n:
        raise ValueError(f"atom_feat rows {af.shape[0]} != atoms {n}")
    u, v, t, bf, bs = canonicalize_bonds_with_features(
        n, bond_index, bond_type, bond_feat, bond_stereo
    )
    if raw_atom_charge is not None and np.asarray(raw_atom_charge).shape != (n,):
        raise ValueError("raw_atom_charge shape mismatch")
    if raw_atom_h is not None and np.asarray(raw_atom_h).shape != (n,):
        raise ValueError("raw_atom_h shape mismatch")
    _comp, n_comp, _size = connected_components(n, u, v)
    return MolecularGraph(
        atom_type=at,
        bond_u=u,
        bond_v=v,
        bond_type=np.asarray(t, dtype=np.int64),
        atom_feat=af,
        bond_feat=bf,
        global_feat=None if global_feat is None else np.asarray(global_feat, dtype=np.float32).reshape(-1),
        n_components=int(n_comp),
        molecule_id=str(molecule_id),
        smiles=str(smiles),
        raw_atom_charge=None if raw_atom_charge is None else np.asarray(raw_atom_charge, dtype=np.int64),
        raw_atom_h=None if raw_atom_h is None else np.asarray(raw_atom_h, dtype=np.int64),
        bond_stereo=bs,
        raw_metadata=dict(raw_metadata or {}),
    )


# ---------------------------------------------------------------------------
# independent path enumeration used by data assembly and the reference gate
# ---------------------------------------------------------------------------


def enumerate_s1(
    atom_type: np.ndarray,
    adj: Sequence[Sequence[tuple[int, int, int]]],
    n_bond_types: int,
) -> list[list[int]]:
    """Per-atom joint ``(neighbour type, bond type)`` codes."""
    out: list[list[int]] = []
    for row in adj:
        out.append(
            [
                int(atom_type[int(v)]) * int(n_bond_types) + (int(bt) - 1)
                for v, bt, _e in row
            ]
        )
    return out


def enumerate_s2(
    atom_type: np.ndarray,
    adj: Sequence[Sequence[tuple[int, int, int]]],
    n_bond_types: int,
    dist: np.ndarray,
) -> list[list[dict[str, int]]]:
    """Per-root ordered non-backtracking 2-paths.

    A record contains ``middle``, ``end``, ``bond1``, ``bond2``, ``path_cat``,
    ``end63`` and ``closure``.  Immediate backtracks (``end == root``) are
    excluded; triangles are retained with ``closure == 1``.
    """
    nb = int(n_bond_types)
    out: list[list[dict[str, int]]] = []
    for root, row in enumerate(adj):
        records: list[dict[str, int]] = []
        for middle, bt1, _e1 in row:
            type_middle = int(atom_type[int(middle)])
            for end, bt2, _e2 in adj[int(middle)]:
                end = int(end)
                if end == root:
                    continue
                records.append(
                    {
                        "middle": int(middle),
                        "end": end,
                        "bond1": int(bt1),
                        "bond2": int(bt2),
                        "path_cat": type_middle * nb * nb + (int(bt1) - 1) * nb + (int(bt2) - 1),
                        "end63": int(atom_type[end]) * nb + (int(bt2) - 1),
                        "closure": 1 if int(dist[root, end]) == 1 else 0,
                    }
                )
        out.append(records)
    return out


__all__ = [
    "UNREACHABLE",
    "canonicalize_physical_bonds",
    "canonicalize_bonds_with_features",
    "adjacency",
    "degrees",
    "connected_components",
    "all_pairs_shortest_paths",
    "biconnected_ring_analysis",
    "min_ring_size_per_atom",
    "junction_count",
    "dist_bucket",
    "relation_pairs",
    "MolecularGraph",
    "GraphAnalysis",
    "analyse_graph",
    "build_molecular_graph",
    "enumerate_s1",
    "enumerate_s2",
]

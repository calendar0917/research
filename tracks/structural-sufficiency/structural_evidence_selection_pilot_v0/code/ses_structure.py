"""SES-v0 lean induced-subgraph structures and canonical BFS candidates.

Reused scientific definitions (copied from the frozen GWIC-v0
``gwic_structure.py``; no words are enumerated here):

* S1 incident-bond / S2 typed rooted two-path / complete radius-2 summaries are
  assembled exactly as in the frozen model's ``encode_h`` inputs;
* atom and graph scalars are derived *from the subgraph itself*.

Two adaptations are necessary for this round:

1. no word catalogue (size 2--4) is built any more;
2. an induced subgraph owns its own degree / ring / graph scalars, so the same
   builder can represent a full molecule or a candidate evidence subgraph.

BFS candidate generation uses a canonical atom order from a pynauty canonical
labelling of the coloured atom/bond incidence graph.  No original atom id is
ever used as a model input; original ids are stored only for explanation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

import numpy as np
import pynauty

ATOM_COLUMNS = 9
BOND_COLUMNS = 3
ATOM_SCALARS = 5
GRAPH_SCALARS = 8
BOND_TYPE_WIDTH = 5
NODE_BUDGET_FRACTION = 0.65
MAX_CANDIDATES = 4


@dataclass
class Struct:
    """One self-contained attributed induced graph (full molecule or candidate)."""

    row: int
    n_atoms: int
    n_bonds: int
    atom_feat: np.ndarray            # [n, 9], in this struct's local order
    bond_feat: np.ndarray            # [m, 3]
    bond_u: np.ndarray               # [m], local, bond_u < bond_v
    bond_v: np.ndarray
    degree: np.ndarray               # [n], local induced degree
    min_ring: np.ndarray             # [n], shortest cycle through atom in subgraph, 0 = none
    in_cycle: np.ndarray             # [n], incident to a non-bridge edge
    atom_scalars: np.ndarray         # [n, 5]
    graph_scalars: np.ndarray        # [8]
    dist1_atoms: np.ndarray
    dist1_ptr: np.ndarray
    dist2_atoms: np.ndarray
    dist2_ptr: np.ndarray
    path_cat: np.ndarray
    path_end_type: np.ndarray
    path_closure: np.ndarray
    path_end_atom: np.ndarray
    path_ptr: np.ndarray
    r2_bond: np.ndarray
    r2_bond_bucket: np.ndarray
    r2_bond_ptr: np.ndarray
    support_atoms: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    support_bonds: np.ndarray = field(default_factory=lambda: np.zeros(0, np.int64))
    source_rank: int = -1
    target_k: int = -1
    component_size: int = -1

    def local_edges(self) -> set[tuple[int, int]]:
        return {(int(u), int(v)) for u, v in zip(self.bond_u, self.bond_v)}


# ---------------------------------------------------------------------------
# canonical order
# ---------------------------------------------------------------------------


def canonical_atom_order(atom_feat: np.ndarray, bond_u: np.ndarray,
                         bond_v: np.ndarray, bond_feat: np.ndarray) -> np.ndarray:
    """Canonical rank -> original local atom id.

    pynauty is applied to the coloured incidence graph: atom vertices carry all
    nine raw atom columns, auxiliary bond vertices carry the three raw bond
    columns.  Isomorphic coloured graphs therefore produce the same atom order.
    """
    n = int(atom_feat.shape[0])
    m = int(bond_u.shape[0])
    total = n + m
    adj: dict[int, set[int]] = {i: set() for i in range(total)}
    for k in range(m):
        a, b = int(bond_u[k]), int(bond_v[k])
        bn = n + k
        adj[a].add(bn)
        adj[b].add(bn)
        adj[bn].add(a)
        adj[bn].add(b)
    colours: dict[int, tuple[int, ...]] = {}
    for i in range(n):
        colours[i] = (0, *[int(x) for x in atom_feat[i]])
    for k in range(m):
        colours[n + k] = (1, *[int(x) for x in bond_feat[k]])
    classes: dict[tuple[int, ...], set[int]] = {}
    for v, c in colours.items():
        classes.setdefault(c, set()).add(v)
    coloring = [classes[c] for c in sorted(classes)]
    graph = pynauty.Graph(
        number_of_vertices=total,
        directed=False,
        adjacency_dict={v: sorted(nb) for v, nb in adj.items()},
        vertex_coloring=coloring,
    )
    order = pynauty.canon_label(graph)  # canonical position -> original vertex
    if order is None:
        raise RuntimeError("pynauty returned no canonical labelling")
    atoms = [int(v) for v in order if int(v) < n]
    if len(atoms) != n:
        raise RuntimeError("canonical labelling lost atom vertices")
    return np.asarray(atoms, np.int64)


def certificate(atom_feat: np.ndarray, bond_u: np.ndarray, bond_v: np.ndarray,
                bond_feat: np.ndarray) -> str:
    """Exact relabelling-invariant key of one coloured induced subgraph."""
    order = canonical_atom_order(atom_feat, bond_u, bond_v, bond_feat)
    pos = np.empty(int(atom_feat.shape[0]), np.int64)
    pos[order] = np.arange(order.size, dtype=np.int64)
    node_part = ";".join(",".join(str(int(x)) for x in atom_feat[v]) for v in order)
    edges = sorted({tuple(sorted((int(pos[int(u)]), int(pos[int(v)]))))
                    for u, v in zip(bond_u, bond_v)})
    feat_by_edge = {}
    for k in range(int(bond_u.shape[0])):
        key = tuple(sorted((int(pos[int(bond_u[k])]), int(pos[int(bond_v[k])]))))
        feat_by_edge[key] = tuple(int(x) for x in bond_feat[k])
    edge_part = ";".join(f"{a}-{b}:{','.join(str(x) for x in feat_by_edge[(a, b)])}"
                         for a, b in edges)
    return "N|" + node_part + "|E|" + edge_part


# ---------------------------------------------------------------------------
# local graph primitives
# ---------------------------------------------------------------------------


def _adjacency(n: int, bond_u: np.ndarray, bond_v: np.ndarray
               ) -> tuple[list[list[int]], list[list[int]]]:
    adj = [[] for _ in range(n)]
    bond_adj = [[] for _ in range(n)]
    for e, (u, v) in enumerate(zip(bond_u.tolist(), bond_v.tolist())):
        adj[u].append(v)
        adj[v].append(u)
        bond_adj[u].append(e)
        bond_adj[v].append(e)
    return adj, bond_adj


def _adjacency_edges(n: int, bond_u: np.ndarray, bond_v: np.ndarray
                     ) -> list[list[tuple[int, int]]]:
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for e, (u, v) in enumerate(zip(bond_u.tolist(), bond_v.tolist())):
        adj[u].append((v, e))
        adj[v].append((u, e))
    return adj


def _bfs(n: int, adj: Sequence[Sequence[int]], source: int) -> np.ndarray:
    dist = np.full(n, -1, np.int16)
    dist[source] = 0
    frontier = [int(source)]
    d = 0
    while frontier:
        d += 1
        nxt: list[int] = []
        for x in frontier:
            for y in adj[x]:
                if dist[y] < 0:
                    dist[y] = d
                    nxt.append(int(y))
        frontier = nxt
    return dist


def find_bridges(n: int, bond_u: np.ndarray, bond_v: np.ndarray) -> np.ndarray:
    adj = _adjacency_edges(n, bond_u, bond_v)
    tin = np.full(n, -1, np.int64)
    low = np.zeros(n, np.int64)
    bridges: list[int] = []
    timer = 0
    for root in range(n):
        if tin[root] != -1:
            continue
        tin[root] = low[root] = timer
        timer += 1
        stack: list[list[int]] = [[root, -1, 0]]
        while stack:
            node, parent_edge, at = stack[-1]
            if at < len(adj[node]):
                nxt, edge_id = adj[node][at]
                stack[-1][2] += 1
                if edge_id == parent_edge:
                    continue
                if tin[nxt] == -1:
                    tin[nxt] = low[nxt] = timer
                    timer += 1
                    stack.append([nxt, edge_id, 0])
                else:
                    low[node] = min(low[node], tin[nxt])
            else:
                stack.pop()
                if stack:
                    parent = stack[-1][0]
                    low[parent] = min(low[parent], low[node])
                    if low[node] > tin[parent]:
                        bridges.append(int(parent_edge))
    return np.asarray(sorted(set(bridges)), np.int64)


def min_ring_sizes(n: int, bond_u: np.ndarray, bond_v: np.ndarray) -> np.ndarray:
    """Exact shortest cycle through each atom inside this induced graph.

    BFS from each neighbour of ``v`` with ``v`` blocked, stopping at the first
    other neighbour; the cycle length is ``dist + 2``.  Results are 0 when the
    atom is not on a cycle.
    """
    masks = [0] * n
    for u, v in zip(bond_u.tolist(), bond_v.tolist()):
        masks[u] |= 1 << v
        masks[v] |= 1 << u
    out = np.zeros(n, np.int64)
    for v in range(n):
        neighbours = [i for i in range(n) if (masks[v] >> i) & 1]
        if len(neighbours) < 2:
            continue
        best = 0
        for u in neighbours:
            target = masks[v] & ~(1 << u)
            visited = (1 << u) | (1 << v)
            frontier = 1 << u
            dist = 0
            found = 0
            while frontier:
                dist += 1
                nxt = 0
                f = frontier
                while f:
                    b = f & -f
                    x = b.bit_length() - 1
                    f ^= b
                    nxt |= masks[x]
                nxt &= ~visited
                if not nxt:
                    break
                if nxt & target:
                    found = dist + 2
                    break
                visited |= nxt
                frontier = nxt
            if found and (best == 0 or found < best):
                best = found
        out[v] = best
    return out


def component_labels(n: int, bond_u: np.ndarray, bond_v: np.ndarray) -> np.ndarray:
    parent = list(range(n))

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in zip(bond_u.tolist(), bond_v.tolist()):
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[ra] = rb
    roots = [find(i) for i in range(n)]
    remap = {r: i for i, r in enumerate(sorted(set(roots)))}
    return np.asarray([remap[r] for r in roots], np.int64)


def graph_scalars_from_graph(n: int, m: int, degree: np.ndarray,
                             in_cycle: np.ndarray, aromatic: np.ndarray,
                             n_components: int) -> np.ndarray:
    cycle_rank = m - n + int(n_components)
    return np.asarray([
        np.log1p(n), np.log1p(m), float(cycle_rank), float(n_components),
        float(degree.mean()) if n else 0.0,
        float(degree.max()) if n else 0.0,
        float(in_cycle.mean()) if n else 0.0,
        float(aromatic.mean()) if n else 0.0,
    ], np.float64)


# ---------------------------------------------------------------------------
# struct builder (S1/S2/R2 summaries, no words)
# ---------------------------------------------------------------------------


def build_struct(atom_feat: np.ndarray, bond_feat: np.ndarray,
                 bond_u: np.ndarray, bond_v: np.ndarray,
                 degree: np.ndarray, min_ring: np.ndarray, in_cycle: np.ndarray,
                 graph_scalars: np.ndarray, *, row: int = -1,
                 support_atoms: np.ndarray | None = None,
                 support_bonds: np.ndarray | None = None,
                 source_rank: int = -1, target_k: int = -1,
                 component_size: int = -1) -> Struct:
    atom_feat = np.asarray(atom_feat, np.int64)
    bond_feat = np.asarray(bond_feat, np.int64)
    bond_u = np.asarray(bond_u, np.int64)
    bond_v = np.asarray(bond_v, np.int64)
    degree = np.asarray(degree, np.int64)
    min_ring = np.asarray(min_ring, np.int64)
    in_cycle = np.asarray(in_cycle, np.int64)
    n = int(atom_feat.shape[0])
    m = int(bond_u.shape[0])
    if atom_feat.shape != (n, ATOM_COLUMNS) or bond_feat.shape != (m, BOND_COLUMNS):
        raise ValueError("feature width mismatch")
    if not np.all(bond_u < bond_v):
        raise ValueError("bonds must be given with local u < v")
    adj, bond_adj = _adjacency(n, bond_u, bond_v)
    atom_type = atom_feat[:, 0]
    bond_type = bond_feat[:, 0]
    n_bond_types = BOND_TYPE_WIDTH

    dist1_parts: list[np.ndarray] = []
    dist2_parts: list[np.ndarray] = []
    dist1_ptr = np.zeros(n + 1, np.int64)
    dist2_ptr = np.zeros(n + 1, np.int64)
    path_cat_parts: list[np.ndarray] = []
    path_end_type_parts: list[np.ndarray] = []
    path_closure_parts: list[np.ndarray] = []
    path_end_atom_parts: list[np.ndarray] = []
    path_ptr = np.zeros(n + 1, np.int64)
    r2_bond_parts: list[np.ndarray] = []
    r2_bucket_parts: list[np.ndarray] = []
    r2_bond_ptr = np.zeros(n + 1, np.int64)

    for v in range(n):
        dist = _bfs(n, adj, v)
        r2_atoms = np.flatnonzero((dist >= 0) & (dist <= 2))
        r2_set = set(int(x) for x in r2_atoms.tolist())
        dist1 = np.flatnonzero(dist == 1)
        dist2 = np.flatnonzero(dist == 2)
        dist1_parts.append(dist1.astype(np.int64))
        dist2_parts.append(dist2.astype(np.int64))
        dist1_ptr[v + 1] = dist1_ptr[v] + dist1.size
        dist2_ptr[v + 1] = dist2_ptr[v] + dist2.size

        cats: list[int] = []
        ends: list[int] = []
        closures: list[int] = []
        end_atoms: list[int] = []
        nbr_v = {int(x) for x in adj[v]}
        for b1 in bond_adj[v]:
            w = int(bond_v[b1]) if int(bond_u[b1]) == v else int(bond_u[b1])
            for b2 in bond_adj[w]:
                u = int(bond_v[b2]) if int(bond_u[b2]) == w else int(bond_u[b2])
                if int(u) == int(v):
                    continue
                bt1 = int(bond_type[b1])
                bt2 = int(bond_type[b2])
                cats.append(int(atom_type[w]) * (n_bond_types * n_bond_types)
                            + bt1 * n_bond_types + bt2)
                ends.append(int(atom_type[u]))
                closures.append(1 if int(u) in nbr_v else 0)
                end_atoms.append(int(u))
        path_cat_parts.append(np.asarray(cats, np.int64))
        path_end_type_parts.append(np.asarray(ends, np.int64))
        path_closure_parts.append(np.asarray(closures, np.int64))
        path_end_atom_parts.append(np.asarray(end_atoms, np.int64))
        path_ptr[v + 1] = path_ptr[v] + len(cats)

        r2_bonds = [e for e in range(m)
                    if int(bond_u[e]) in r2_set and int(bond_v[e]) in r2_set]
        r2_bonds.sort()
        r2_bond_parts.append(np.asarray(r2_bonds, np.int64))
        r2_bucket_parts.append(np.asarray(
            [min(int(dist[bond_u[e]]), int(dist[bond_v[e]])) for e in r2_bonds],
            np.int64))
        r2_bond_ptr[v + 1] = r2_bond_ptr[v] + len(r2_bonds)

    def cat(parts: list[np.ndarray]) -> np.ndarray:
        return np.concatenate(parts) if parts else np.zeros(0, np.int64)

    atom_scalars = np.stack([
        degree.astype(np.float64),
        min_ring.astype(np.float64),
        in_cycle.astype(np.float64),
        np.log1p(degree.astype(np.float64)),
        (degree == 0).astype(np.float64),
    ], axis=1)
    return Struct(
        row=int(row), n_atoms=n, n_bonds=m,
        atom_feat=atom_feat, bond_feat=bond_feat, bond_u=bond_u, bond_v=bond_v,
        degree=degree, min_ring=min_ring, in_cycle=in_cycle,
        atom_scalars=atom_scalars,
        graph_scalars=np.asarray(graph_scalars, np.float64),
        dist1_atoms=cat(dist1_parts), dist1_ptr=dist1_ptr,
        dist2_atoms=cat(dist2_parts), dist2_ptr=dist2_ptr,
        path_cat=cat(path_cat_parts), path_end_type=cat(path_end_type_parts),
        path_closure=cat(path_closure_parts), path_end_atom=cat(path_end_atom_parts),
        path_ptr=path_ptr,
        r2_bond=cat(r2_bond_parts), r2_bond_bucket=cat(r2_bucket_parts),
        r2_bond_ptr=r2_bond_ptr,
        support_atoms=(np.zeros(0, np.int64) if support_atoms is None
                       else np.asarray(support_atoms, np.int64)),
        support_bonds=(np.zeros(0, np.int64) if support_bonds is None
                       else np.asarray(support_bonds, np.int64)),
        source_rank=int(source_rank), target_k=int(target_k),
        component_size=int(component_size),
    )


# ---------------------------------------------------------------------------
# full molecule and candidate views
# ---------------------------------------------------------------------------


def canonical_full_struct(raw: Mapping[str, Any], row: int) -> Struct:
    """Full original graph in canonical atom/bond order (DENSE input)."""
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
    b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
    atom_feat = np.asarray(raw["atom_feat"][a0:a1], np.int64)
    bond_feat = np.asarray(raw["bond_feat"][b0:b1], np.int64)
    bu = np.asarray(raw["bond_u"][b0:b1], np.int64)
    bv = np.asarray(raw["bond_v"][b0:b1], np.int64)
    degree0 = np.asarray(raw["degree"][a0:a1], np.int64)
    min_ring0 = np.asarray(raw["min_ring"][a0:a1], np.int64)
    in_cycle0 = np.asarray(raw["in_cycle"][a0:a1], np.int64)
    n = int(atom_feat.shape[0])
    m = int(bu.shape[0])
    order = canonical_atom_order(atom_feat, bu, bv, bond_feat)  # rank -> old id
    pos = np.empty(n, np.int64)
    pos[order] = np.arange(n, dtype=np.int64)
    # sorted induced bonds in canonical endpoint order
    pair = np.stack([np.minimum(pos[bu], pos[bv]), np.maximum(pos[bu], pos[bv])], axis=1)
    bond_order = np.lexsort((pair[:, 1], pair[:, 0]))
    new_u = pair[bond_order, 0]
    new_v = pair[bond_order, 1]
    graph = np.asarray([
        np.log1p(n), np.log1p(m),
        float(np.asarray(raw["graph_cycle_rank"])[row]),
        float(np.asarray(raw["graph_n_components"])[row]),
        float(degree0.mean()) if n else 0.0,
        float(degree0.max()) if n else 0.0,
        float(in_cycle0.mean()) if n else 0.0,
        float(atom_feat[:, 7].mean()) if n else 0.0,
    ], np.float64)
    return build_struct(
        atom_feat=atom_feat[order], bond_feat=bond_feat[bond_order],
        bond_u=new_u, bond_v=new_v,
        degree=degree0[order], min_ring=min_ring0[order], in_cycle=in_cycle0[order],
        graph_scalars=graph, row=int(row),
        support_atoms=order.copy(), support_bonds=(b0 + bond_order),
        target_k=int(max(1, min(n - 1, int(np.floor(NODE_BUDGET_FRACTION * n))))),
        component_size=n,
    )


@dataclass
class CandidateSpec:
    source_rank: int
    ranks: tuple[int, ...]
    original_atoms: np.ndarray
    original_bonds: np.ndarray
    target_k: int
    component_size: int


def _bfs_support_from_rank(adj_rank: Sequence[Sequence[int]], source_rank: int,
                           k: int) -> tuple[int, ...]:
    seen = {int(source_rank)}
    queue = [int(source_rank)]
    out: list[int] = []
    head = 0
    while head < len(queue) and len(out) < k:
        x = queue[head]
        head += 1
        out.append(x)
        for y in adj_rank[x]:
            if y not in seen:
                seen.add(y)
                queue.append(y)
    return tuple(sorted(out))


def generate_candidates(raw: Mapping[str, Any], row: int
                         ) -> tuple[list[CandidateSpec], dict[str, Any]]:
    """Canonical BFS supports for one molecule; at most four diverse supports."""
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    bond_ptr = np.asarray(raw["bond_ptr"], np.int64)
    a0, a1 = int(node_ptr[row]), int(node_ptr[row + 1])
    b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
    atom_feat = np.asarray(raw["atom_feat"][a0:a1], np.int64)
    bond_feat = np.asarray(raw["bond_feat"][b0:b1], np.int64)
    bu = np.asarray(raw["bond_u"][b0:b1], np.int64)
    bv = np.asarray(raw["bond_v"][b0:b1], np.int64)
    n = int(atom_feat.shape[0])
    m = int(bu.shape[0])
    order = canonical_atom_order(atom_feat, bu, bv, bond_feat)  # rank -> old
    rank_of = np.empty(n, np.int64)
    rank_of[order] = np.arange(n, dtype=np.int64)
    adj_rank: list[list[int]] = [[] for _ in range(n)]
    for u, v in zip(bu.tolist(), bv.tolist()):
        ru, rv = int(rank_of[u]), int(rank_of[v])
        adj_rank[ru].append(rv)
        adj_rank[rv].append(ru)
    for r in range(n):
        adj_rank[r] = sorted(adj_rank[r])
    if n == 1:
        k = 1
    else:
        k = int(max(1, min(n - 1, np.floor(NODE_BUDGET_FRACTION * n))))
    comp = component_labels(n, bu, bv)
    comp_size = {int(c): int((comp == c).sum()) for c in np.unique(comp)}

    # unique BFS supports over every atom, keyed by canonical rank tuple
    by_key: dict[tuple[int, ...], int] = {}
    for source in range(n):
        ranks = _bfs_support_from_rank(adj_rank, source, k)
        if ranks not in by_key:
            by_key[ranks] = source
    keys = sorted(by_key, key=lambda rk: (min(rk), rk))
    # fixed canonical first candidate: BFS from canonical atom rank 0
    first_key = _bfs_support_from_rank(adj_rank, 0, k)
    if first_key not in by_key:
        raise RuntimeError("canonical first BFS candidate missing")

    def jaccard_distance(a: set[int], b: set[int]) -> float:
        union = a | b
        return 1.0 - (len(a & b) / len(union) if union else 0.0)

    selected: list[tuple[int, ...]] = [first_key]
    remaining = [rk for rk in keys if rk != first_key]
    while remaining and len(selected) < MAX_CANDIDATES:
        chosen = None
        chosen_key = None
        for rk in remaining:
            s = set(rk)
            score = min(jaccard_distance(s, set(x)) for x in selected)
            tie = (by_key[rk], rk)
            if chosen is None or score > chosen[0] + 1e-12 or (
                    abs(score - chosen[0]) <= 1e-12 and tie < chosen[1]):
                chosen = (score, tie)
                chosen_key = rk
        if chosen_key is None:
            break
        selected.append(chosen_key)
        remaining.remove(chosen_key)

    specs: list[CandidateSpec] = []
    for ranks in selected:
        atoms = order[np.asarray(ranks, np.int64)]
        atom_set = set(int(x) for x in atoms.tolist())
        bonds = [e for e in range(m)
                 if int(bu[e]) in atom_set and int(bv[e]) in atom_set]
        # canonical internal bond order: by canonical rank of endpoints
        bonds.sort(key=lambda e: (min(int(rank_of[bu[e]]), int(rank_of[bv[e]])),
                                  max(int(rank_of[bu[e]]), int(rank_of[bv[e]]))))
        source = int(by_key[ranks])
        specs.append(CandidateSpec(
            source_rank=source,
            ranks=tuple(int(x) for x in ranks),
            original_atoms=np.asarray(atoms, np.int64),
            original_bonds=np.asarray([b0 + e for e in bonds], np.int64),
            target_k=int(k),
            component_size=int(comp_size[int(comp[atoms[0]])]),
        ))
    # deterministic listing order for <=4: canonical source/support order
    if len(specs) <= MAX_CANDIDATES:
        specs.sort(key=lambda s: (s.source_rank, s.ranks))
    meta = {
        "row": int(row), "n_atoms": int(n), "n_bonds": int(m),
        "target_k": int(k), "n_unique_candidates": int(len(by_key)),
        "n_candidates": int(len(specs)),
        "n_components": int(len(comp_size)),
        "component_sizes": sorted(comp_size.values()),
        "any_component_smaller_than_k": bool(any(sz < k for sz in comp_size.values())),
        "n_sources_below_k": int(sum(1 for sz in comp_size.values() if sz < k)),
        "n_atoms_special": bool(n == 1),
    }
    return specs, meta


def build_candidate_struct(raw: Mapping[str, Any], row: int,
                           spec: CandidateSpec) -> Struct:
    """Build one candidate from its real induced atoms/bonds only."""
    node_ptr = np.asarray(raw["node_ptr"], np.int64)
    a0 = int(node_ptr[row])
    atoms = np.asarray(spec.original_atoms, np.int64)
    s = int(atoms.size)
    remap = {int(g): i for i, g in enumerate(atoms.tolist())}
    atom_feat = np.asarray(raw["atom_feat"][a0 + atoms], np.int64)
    bonds = np.asarray(spec.original_bonds, np.int64)
    bond_feat = np.asarray(raw["bond_feat"][bonds], np.int64)
    bu = np.asarray([remap[int(raw["bond_u"][e])] for e in bonds], np.int64)
    bv = np.asarray([remap[int(raw["bond_v"][e])] for e in bonds], np.int64)
    # orient by local canonical order; atoms are already sorted by canonical rank
    flip = bu > bv
    if flip.any():
        bu2 = bu.copy()
        bu[flip] = bv[flip]
        bv[flip] = bu2[flip]
    degree = np.bincount(np.concatenate([bu, bv]), minlength=s).astype(np.int64) \
        if bu.size else np.zeros(s, np.int64)
    bridges = set(int(x) for x in find_bridges(s, bu, bv).tolist())
    in_cycle = np.zeros(s, np.int64)
    for e in range(int(bu.size)):
        if e not in bridges:
            in_cycle[int(bu[e])] = 1
            in_cycle[int(bv[e])] = 1
    min_ring = min_ring_sizes(s, bu, bv)
    comp = component_labels(s, bu, bv) if s else np.zeros(0, np.int64)
    n_comp = int(comp.max() + 1) if comp.size else 0
    graph = graph_scalars_from_graph(s, int(bu.size), degree, in_cycle,
                                     atom_feat[:, 7].astype(np.float64), n_comp)
    struct = build_struct(
        atom_feat=atom_feat, bond_feat=bond_feat, bond_u=bu, bond_v=bv,
        degree=degree, min_ring=min_ring, in_cycle=in_cycle,
        graph_scalars=graph, row=int(row),
        support_atoms=atoms, support_bonds=bonds,
        source_rank=int(spec.source_rank), target_k=int(spec.target_k),
        component_size=int(spec.component_size),
    )
    # exact induced-edge check against the original physical graph
    orig_edges = set()
    for e in range(int(np.asarray(raw["bond_ptr"])[row]),
                   int(np.asarray(raw["bond_ptr"])[row + 1])):
        u = int(np.asarray(raw["bond_u"])[e])
        v = int(np.asarray(raw["bond_v"])[e])
        if u in remap and v in remap:
            orig_edges.add(tuple(sorted((u, v))))
    if orig_edges != {tuple(sorted((int(atoms[u]), int(atoms[v])))) for u, v in zip(bu, bv)}:
        raise RuntimeError("candidate edge set is not the exact induced edge set")
    return struct

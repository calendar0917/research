"""Discrete GWIC-v0 structure: R2 sentences and connected induced word supports.

For every atom v of a molecule the *sentence* is the full radius-2 rooted ego
graph

    R2(v) = {t : dist(v,t) <= 2} with every real induced bond.

A *candidate word* is a distinct atom support S with v in S, 2 <= |S| <= 4,
whose induced subgraph is connected.  Every support is enumerated once and
keeps all internal induced bonds and the real boundary bonds to R2(v) \\ S.
Atom ids are molecule-local indices only and never learnable features.

The enumeration is deterministic and uses no labels.  Supports for one root are
sorted lexicographically, so candidate order is canonical.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

MAX_WORD_ATOMS = 4
MAX_WORD_BONDS = 6
WORD_SCALARS = 25
BOUNDARY_FIELDS = (5, 6, 2)  # official OGB bond categorical widths
N_BOUNDARY_TYPES = sum(BOUNDARY_FIELDS)


@dataclass
class MoleculeStruct:
    row: int
    n_atoms: int
    n_bonds: int
    atom_feat: np.ndarray            # [n, 9]
    bond_feat: np.ndarray            # [m, 3]
    bond_u: np.ndarray               # [m]
    bond_v: np.ndarray
    degree: np.ndarray               # [n]
    min_ring: np.ndarray
    in_cycle: np.ndarray
    atom_scalars: np.ndarray         # [n, 5]
    graph_scalars: np.ndarray        # [8]
    dist1_atoms: np.ndarray          # flat local ids, distance exactly 1
    dist1_ptr: np.ndarray            # [n+1]
    dist2_atoms: np.ndarray
    dist2_ptr: np.ndarray
    path_cat: np.ndarray             # flat, one per rooted non-backtracking 2-path
    path_end_type: np.ndarray
    path_closure: np.ndarray
    path_end_atom: np.ndarray
    path_ptr: np.ndarray             # [n+1]
    r2_bond: np.ndarray              # flat bond ids in R2(v)
    r2_bond_bucket: np.ndarray       # 0,1,2 = min distance to root
    r2_bond_ptr: np.ndarray          # [n+1]
    word_root: np.ndarray            # [w], nondecreasing
    word_support: np.ndarray         # [w, 4], -1 pad
    word_internal: np.ndarray        # [w, 6], -1 pad
    word_scalars: np.ndarray         # [w, 25]
    word_root_ptr: np.ndarray        # [n+1]
    n_words: int
    n_internal_bonds: int
    n_boundary_bonds: int
    n_roots_with_words: int
    max_words_per_root: int


def _bfs(n_atoms: int, adj: Sequence[Sequence[int]], source: int) -> np.ndarray:
    dist = np.full(n_atoms, -1, np.int16)
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


def _bfs_restricted(n_atoms: int, adj: Sequence[Sequence[int]],
                    source: int, allowed: set[int]) -> np.ndarray:
    dist = np.full(n_atoms, -1, np.int16)
    dist[source] = 0
    frontier = [int(source)]
    d = 0
    while frontier:
        d += 1
        nxt: list[int] = []
        for x in frontier:
            for y in adj[x]:
                if y in allowed and dist[y] < 0:
                    dist[y] = d
                    nxt.append(int(y))
        frontier = nxt
    return dist


def _enumerate_supports(root: int, induced_adj: Mapping[int, Sequence[int]]) -> list[tuple[int, ...]]:
    """All connected supports of size 2..4 containing ``root``, once each."""
    found: set[tuple[int, ...]] = set()
    stack: list[tuple[int, ...]] = [(int(root),)]
    while stack:
        cur = stack.pop()
        if len(cur) >= 2:
            found.add(tuple(sorted(cur)))
        if len(cur) >= MAX_WORD_ATOMS:
            continue
        present = set(cur)
        for x in cur:
            for y in induced_adj[x]:
                if y not in present:
                    stack.append(tuple(sorted(cur + (int(y),))))
    return sorted(found)


def _word_scalars(n_atoms: int, n_bonds: int, support: tuple[int, ...],
                  dist: np.ndarray, boundary_bonds: Sequence[int],
                  bond_feat: np.ndarray, degree: np.ndarray) -> np.ndarray:
    S = set(int(x) for x in support)
    # distances inside the induced support (BFS already restricted to S)
    pair_d = [0, 0, 0, 0]
    root_d = [0, 0, 0, 0]
    for i, a in enumerate(support):
        da = int(dist[a])
        if 0 <= da <= 3:
            root_d[da] += 1
        for b in support[i + 1:]:
            dd = int(dist[b])
            if 0 <= dd <= 3:
                pair_d[dd] += 1
    n_pairs = max(1, len(support) * (len(support) - 1) // 2)
    n_support = max(1, len(support))
    cycle_rank = n_bonds - n_atoms + 1
    boundary = list(boundary_bonds)
    bcount = len(boundary)
    total_deg = float(np.sum(degree[list(S)])) if S else 0.0
    type_counts = np.zeros(N_BOUNDARY_TYPES, np.float64)
    for bidx in boundary:
        off = 0
        for col, width in enumerate(BOUNDARY_FIELDS):
            cat = int(bond_feat[int(bidx), col])
            if 0 <= cat < width:
                type_counts[off + cat] += 1.0
            off += width
    frac = type_counts / bcount if bcount else np.zeros_like(type_counts)
    out = np.zeros(WORD_SCALARS, np.float64)
    out[0] = len(support) / 4.0
    out[1] = n_bonds / 6.0
    out[2] = cycle_rank / 3.0
    out[3] = 0.0  # root degree, filled by builder
    out[4:7] = np.asarray(pair_d[1:4], np.float64) / n_pairs
    out[7:10] = np.asarray(root_d[1:4], np.float64) / n_support
    out[10] = bcount / max(1.0, total_deg)
    out[11] = np.log1p(bcount)
    out[12:25] = frac
    return out


def build_molecule_struct(view: Mapping[str, Any]) -> MoleculeStruct:
    row = int(view["row"])
    n = int(view["n_atoms"])
    m = int(view["n_bonds"])
    atom_feat = np.asarray(view["atom_feat"], np.int64)
    bond_feat = np.asarray(view["bond_feat"], np.int64)
    bu = np.asarray(view["bond_u"], np.int64)
    bv = np.asarray(view["bond_v"], np.int64)
    degree = np.asarray(view["degree"], np.int64)
    min_ring = np.asarray(view["min_ring"], np.int64)
    in_cycle = np.asarray(view["in_cycle"], np.int64)

    adj: list[list[int]] = [[] for _ in range(n)]
    bond_adj: list[list[int]] = [[] for _ in range(n)]
    for e in range(m):
        u, v = int(bu[e]), int(bv[e])
        adj[u].append(v)
        adj[v].append(u)
        bond_adj[u].append(e)
        bond_adj[v].append(e)

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
    word_roots: list[int] = []
    word_supports: list[np.ndarray] = []
    word_internals: list[np.ndarray] = []
    word_scalar_rows: list[np.ndarray] = []
    word_root_ptr = np.zeros(n + 1, np.int64)
    n_internal_total = 0
    n_boundary_total = 0
    max_words_per_root = 0
    n_roots_with_words = 0

    atom_type = atom_feat[:, 0]
    bond_type = bond_feat[:, 0]
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

        # rooted non-backtracking 2-paths v -> w -> u, u != v.  Iterating the
        # real bond ids avoids any dependence on neighbour-list order.
        cats: list[int] = []
        ends: list[int] = []
        closures: list[int] = []
        end_atoms: list[int] = []
        nbr_v = set(int(x) for x in adj[v])
        for b1 in bond_adj[v]:
            w = int(bv[b1]) if int(bu[b1]) == v else int(bu[b1])
            for b2 in bond_adj[w]:
                u = int(bv[b2]) if int(bu[b2]) == w else int(bu[b2])
                if int(u) == int(v):
                    continue
                bt1 = int(bond_type[b1])
                bt2 = int(bond_type[b2])
                cats.append(int(atom_type[w]) * 25 + bt1 * 5 + bt2)
                ends.append(int(atom_type[u]))
                closures.append(1 if int(u) in nbr_v else 0)
                end_atoms.append(int(u))
        path_cat_parts.append(np.asarray(cats, np.int64))
        path_end_type_parts.append(np.asarray(ends, np.int64))
        path_closure_parts.append(np.asarray(closures, np.int64))
        path_end_atom_parts.append(np.asarray(end_atoms, np.int64))
        path_ptr[v + 1] = path_ptr[v] + len(cats)

        # all real induced bonds of R2(v)
        r2_bonds = [e for e in range(m)
                    if int(bu[e]) in r2_set and int(bv[e]) in r2_set]
        r2_bonds.sort()
        r2_bond_parts.append(np.asarray(r2_bonds, np.int64))
        buckets = np.asarray([min(int(dist[bu[e]]), int(dist[bv[e]])) for e in r2_bonds],
                             np.int64)
        r2_bucket_parts.append(buckets)
        r2_bond_ptr[v + 1] = r2_bond_ptr[v] + len(r2_bonds)

        induced_adj = {int(x): [int(y) for y in adj[int(x)] if int(y) in r2_set]
                       for x in r2_atoms.tolist()}
        supports = _enumerate_supports(v, induced_adj)
        if supports:
            n_roots_with_words += 1
            max_words_per_root = max(max_words_per_root, len(supports))
        for support in supports:
            S = set(int(x) for x in support)
            internal = sorted(e for e in range(m)
                              if int(bu[e]) in S and int(bv[e]) in S)
            boundary = sorted(e for e in range(m)
                              if (int(bu[e]) in S) != (int(bv[e]) in S)
                              and (int(bu[e]) in r2_set) and (int(bv[e]) in r2_set))
            # exact induced distances inside S
            wdist = _bfs_restricted(n, adj, v, S)
            internal_arr = np.full(MAX_WORD_BONDS, -1, np.int64)
            internal_arr[:len(internal)] = internal
            support_arr = np.full(MAX_WORD_ATOMS, -1, np.int64)
            support_arr[:len(support)] = support
            scal = _word_scalars(len(support), len(internal), support, wdist,
                                 boundary, bond_feat, degree)
            root_deg = 0
            for e in internal:
                if int(bu[e]) == v or int(bv[e]) == v:
                    root_deg += 1
            scal[3] = root_deg / 3.0
            word_roots.append(v)
            word_supports.append(support_arr)
            word_internals.append(internal_arr)
            word_scalar_rows.append(scal)
            n_internal_total += len(internal)
            n_boundary_total += len(boundary)
        word_root_ptr[v + 1] = word_root_ptr[v] + len(supports)

    graph = np.asarray([
        np.log1p(n), np.log1p(m), float(view["graph_cycle_rank"]),
        float(view["graph_n_components"]),
        float(degree.mean()) if n else 0.0,
        float(degree.max()) if n else 0.0,
        float(in_cycle.mean()) if n else 0.0,
        float(atom_feat[:, 7].mean()) if n else 0.0,
    ], np.float64)

    def _cat(parts: list[np.ndarray], width: int) -> np.ndarray:
        if not parts:
            return np.zeros((0, width), np.int64)
        return np.stack(parts, axis=0).astype(np.int64, copy=False)

    if word_supports:
        word_support = np.stack(word_supports, axis=0).astype(np.int64)
        word_internal = np.stack(word_internals, axis=0).astype(np.int64)
        word_scalars = np.stack(word_scalar_rows, axis=0).astype(np.float32)
    else:
        word_support = np.zeros((0, MAX_WORD_ATOMS), np.int64)
        word_internal = np.zeros((0, MAX_WORD_BONDS), np.int64)
        word_scalars = np.zeros((0, WORD_SCALARS), np.float32)
    return MoleculeStruct(
        row=row, n_atoms=n, n_bonds=m,
        atom_feat=atom_feat, bond_feat=bond_feat, bond_u=bu, bond_v=bv,
        degree=degree, min_ring=min_ring, in_cycle=in_cycle,
        atom_scalars=np.asarray(view["atom_scalars"], np.float64),
        graph_scalars=graph,
        dist1_atoms=np.concatenate(dist1_parts) if dist1_parts else np.zeros(0, np.int64),
        dist1_ptr=dist1_ptr,
        dist2_atoms=np.concatenate(dist2_parts) if dist2_parts else np.zeros(0, np.int64),
        dist2_ptr=dist2_ptr,
        path_cat=np.concatenate(path_cat_parts) if path_cat_parts else np.zeros(0, np.int64),
        path_end_type=np.concatenate(path_end_type_parts) if path_end_type_parts else np.zeros(0, np.int64),
        path_closure=np.concatenate(path_closure_parts) if path_closure_parts else np.zeros(0, np.int64),
        path_end_atom=np.concatenate(path_end_atom_parts) if path_end_atom_parts else np.zeros(0, np.int64),
        path_ptr=path_ptr,
        r2_bond=np.concatenate(r2_bond_parts) if r2_bond_parts else np.zeros(0, np.int64),
        r2_bond_bucket=np.concatenate(r2_bucket_parts) if r2_bucket_parts else np.zeros(0, np.int64),
        r2_bond_ptr=r2_bond_ptr,
        word_root=np.asarray(word_roots, np.int64),
        word_support=word_support,
        word_internal=word_internal,
        word_scalars=word_scalars,
        word_root_ptr=word_root_ptr,
        n_words=int(len(word_roots)),
        n_internal_bonds=int(n_internal_total),
        n_boundary_bonds=int(n_boundary_total),
        n_roots_with_words=int(n_roots_with_words),
        max_words_per_root=int(max_words_per_root),
    )


def build_all(raw: Mapping[str, Any], rows: Sequence[int]) -> dict[int, MoleculeStruct]:
    from gwic_data import atom_scalars, molecule_view

    out: dict[int, MoleculeStruct] = {}
    for row in np.asarray(rows, np.int64).tolist():
        view = molecule_view(raw, int(row))
        view["atom_scalars"] = atom_scalars(view)
        out[int(row)] = build_molecule_struct(view)
    return out


def word_reference(struct: MoleculeStruct, word: int) -> dict[str, Any]:
    """Explicit reference record for one candidate word (tests / diagnostics)."""
    support = [int(x) for x in struct.word_support[word].tolist() if int(x) >= 0]
    internal = [int(x) for x in struct.word_internal[word].tolist() if int(x) >= 0]
    root = int(struct.word_root[word])
    S = set(support)
    r2 = set()
    # explicit R2 membership from the stored sentence: all endpoints of R2 bonds
    for e in struct.r2_bond[struct.r2_bond_ptr[root]:struct.r2_bond_ptr[root + 1]].tolist():
        r2.add(int(struct.bond_u[e])); r2.add(int(struct.bond_v[e]))
    r2.add(root)
    boundary = sorted(e for e in range(struct.n_bonds)
                      if (int(struct.bond_u[e]) in S) != (int(struct.bond_v[e]) in S)
                      and (int(struct.bond_u[e]) in r2) and (int(struct.bond_v[e]) in r2))
    return {"root": root, "support": support, "internal_bonds": internal,
            "boundary_bonds": boundary, "r2_atoms": sorted(r2)}


def structure_stats(structs: Mapping[int, MoleculeStruct]) -> dict[str, Any]:
    n_words = np.asarray([s.n_words for s in structs.values()], np.float64)
    n_atoms = np.asarray([s.n_atoms for s in structs.values()], np.float64)
    per_root = np.concatenate([
        np.diff(s.word_root_ptr) for s in structs.values()
    ]) if structs else np.zeros(0, np.float64)
    return {
        "n_molecules": int(len(structs)),
        "n_atoms": int(n_atoms.sum()),
        "n_words": int(n_words.sum()),
        "n_internal_bond_occurrences": int(sum(s.n_internal_bonds for s in structs.values())),
        "n_boundary_bond_occurrences": int(sum(s.n_boundary_bonds for s in structs.values())),
        "words_per_molecule_mean": float(n_words.mean()) if n_words.size else 0.0,
        "words_per_molecule_median": float(np.median(n_words)) if n_words.size else 0.0,
        "words_per_molecule_max": int(n_words.max()) if n_words.size else 0,
        "words_per_root_mean": float(per_root.mean()) if per_root.size else 0.0,
        "words_per_root_median": float(np.median(per_root)) if per_root.size else 0.0,
        "words_per_root_p90": float(np.percentile(per_root, 90)) if per_root.size else 0.0,
        "words_per_root_p99": float(np.percentile(per_root, 99)) if per_root.size else 0.0,
        "words_per_root_max": int(per_root.max()) if per_root.size else 0,
        "roots_without_words": int((per_root == 0).sum()) if per_root.size else 0,
    }

"""Exact one-shot S3 enumeration: simple bond paths of length exactly three.

Definition (frozen, see ``PROTOCOL.md`` §1)

    for every root atom ``r`` in an undirected molecular graph, enumerate every
    **simple** path ``p = (r, u, v, x)`` with

        r-u, u-v, v-x bonds   and   {r, u, v, x} pairwise distinct

    recording ``root=r, mid1=u, mid2=v, end=x``, the three bond types in path
    order, the two mid atom types, and ``shortest_distance = dist(r, x)``.

Properties that are *requirements*, not conveniences:

* exactly three real bonds; four distinct atoms; no repeated-atom walk ever
  enters the enumeration (no backtracking, no cycles shorter than the path);
* a path is kept once per (root, u, v, x) tuple — path multiplicity to the same
  endpoint is preserved, never deduplicated, never sampled, never truncated;
* the direction is rooted: the enumeration always starts at ``r`` and walks
  outward, so the three bond slots are ordered;
* only the raw graph (bond list, bond types, atom types) is used; no learned or
  trained quantity can influence which paths exist.

The module is deliberately torch-free and dependency-light (numpy only) so the
structural audit runs on CPU and so the enumerator can be cross-checked against
an independent brute-force reference (``reference_enumerate_s3``).

Also here: the label-free local distinguishability machinery of §3.2 of the
round brief.

* ``rooted_ball`` extracts the exact rooted induced subgraph of radius ``k``;
* ``wl_signature`` is a 1-dimensional Weisfeiler-Lehman colour refinement of
  that ball.  It is a **sound but incomplete** invariant: equal signature does
  not prove isomorphism and unequal signature does prove non-isomorphism, so a
  witness pair is always re-verified exactly with ``ball_isomorphic``
  (networkx VF2 on the labelled induced subgraphs), which is complete for the
  ball sizes seen here.  Nothing in this module claims a *representation
  capacity* theorem; it only reports local structural distinguishability.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

UNREACHABLE = -1
S3_N_BONDS = 3
S3_N_ATOMS = 4
S3_DIST_CLASSES = (1, 2, 3)


# ---------------------------------------------------------------------------
# graph primitives
# ---------------------------------------------------------------------------


def molecule_adjacency(
    n_atoms: int,
    bond_u: Sequence[int] | np.ndarray,
    bond_v: Sequence[int] | np.ndarray,
    bond_type: Sequence[int] | np.ndarray,
) -> list[list[tuple[int, int]]]:
    """Symmetric, sorted neighbour lists ``(neighbour, one_based_bond_type)``.

    ``bond_u`` / ``bond_v`` must be the canonical physical bond list of the
    frozen cache (``u < v``, sorted).  The adjacency is rebuilt from it, never
    from a directed/duplicated edge list, so S3 cannot depend on edge order.
    """
    n = int(n_atoms)
    u = np.asarray(bond_u, dtype=np.int64).reshape(-1)
    v = np.asarray(bond_v, dtype=np.int64).reshape(-1)
    t = np.asarray(bond_type, dtype=np.int64).reshape(-1)
    if not (u.shape == v.shape == t.shape):
        raise ValueError("bond arrays must have the same length")
    if np.any(u < 0) or np.any(v < 0) or np.any(u >= n) or np.any(v >= n):
        raise ValueError("bond endpoint outside atom range")
    if np.any(u == v):
        raise ValueError("self-loop in molecular graph")
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for a, b, bt in zip(u.tolist(), v.tolist(), t.tolist()):
        adj[a].append((int(b), int(bt)))
        adj[b].append((int(a), int(bt)))
    for row in adj:
        row.sort()
    for row in adj:
        if len({nb for nb, _ in row}) != len(row):
            raise ValueError("duplicate physical bond in neighbour list")
    return adj


def bond_type_lookup(adj: Sequence[Sequence[tuple[int, int]]], a: int, b: int) -> int:
    """One-based bond type of ``(a, b)``; raises if the bond does not exist."""
    for nb, bt in adj[int(a)]:
        if nb == int(b):
            return int(bt)
    raise KeyError((a, b))


def all_pairs_shortest_paths(n_atoms: int, adj: Sequence[Sequence[tuple[int, int]]]) -> np.ndarray:
    """Unweighted BFS all-pairs distances, ``-1`` for unreachable atoms."""
    n = int(n_atoms)
    dist = np.full((n, n), UNREACHABLE, dtype=np.int16)
    for src in range(n):
        d = dist[src]
        d[src] = 0
        frontier = [src]
        step = 0
        while frontier:
            step += 1
            nxt: list[int] = []
            for node in frontier:
                for nb, _bt in adj[node]:
                    if d[nb] == UNREACHABLE:
                        d[nb] = step
                        nxt.append(nb)
            frontier = nxt
    return dist


# ---------------------------------------------------------------------------
# S3 enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class S3Paths:
    """Flat, root-major S3 path arrays for one molecule.

    ``ptr[i]..ptr[i+1]`` slices the paths whose root is atom ``i``.  Endpoint and
    mid-atom indices are *local atom ids in the same molecule*; the bond types
    are one-based (as in the frozen cache); ``atom_type_mid*`` are the raw
    ``atom_type`` vocabulary ids; ``dist`` is ``dist(root, end) in {1, 2, 3}``.
    """

    ptr: np.ndarray
    end: np.ndarray
    atom_type_mid1: np.ndarray
    atom_type_mid2: np.ndarray
    bond_type_1: np.ndarray
    bond_type_2: np.ndarray
    bond_type_3: np.ndarray
    dist: np.ndarray

    @property
    def n_paths(self) -> int:
        return int(self.ptr[-1])

    def per_root_counts(self) -> np.ndarray:
        return np.diff(self.ptr).astype(np.int64)

    def records(self, root: int) -> list[dict[str, int]]:
        lo, hi = int(self.ptr[root]), int(self.ptr[root + 1])
        return [
            {
                "root_atom": int(root),
                "mid_atom_1": int(self.mid1[i]),
                "mid_atom_2": int(self.mid2[i]),
                "end_atom": int(self.end[i]),
                "bond_type_1": int(self.bond_type_1[i]),
                "bond_type_2": int(self.bond_type_2[i]),
                "bond_type_3": int(self.bond_type_3[i]),
                "mid_type_1": int(self.atom_type_mid1[i]),
                "mid_type_2": int(self.atom_type_mid2[i]),
                "shortest_distance": int(self.dist[i]),
            }
            for i in range(lo, hi)
        ]

    # the mid atom *ids* live only inside :func:`enumerate_s3`'s working list;
    # they are kept as attributes for the audit and are not part of the cache.
    mid1: np.ndarray | None = None
    mid2: np.ndarray | None = None


def enumerate_s3(
    n_atoms: int,
    adj: Sequence[Sequence[tuple[int, int]]],
    atom_type: Sequence[int] | np.ndarray,
    dist: np.ndarray | None = None,
    *,
    keep_mid_atoms: bool = True,
) -> S3Paths:
    """Every simple length-3 bond path, one record per (root, u, v, x)."""
    n = int(n_atoms)
    at = np.asarray(atom_type, dtype=np.int64).reshape(-1)
    if at.shape != (n,):
        raise ValueError("atom_type length != n_atoms")
    if dist is None:
        dist = all_pairs_shortest_paths(n, adj)

    roots: list[int] = []
    ends: list[int] = []
    mids1: list[int] = []
    mids2: list[int] = []
    b1s: list[int] = []
    b2s: list[int] = []
    b3s: list[int] = []
    ds: list[int] = []

    for r in range(n):
        for u, bt1 in adj[r]:
            if u == r:  # pragma: no cover - guarded by construction
                raise RuntimeError("self-loop")
            for v, bt2 in adj[u]:
                if v == r or v == u:
                    continue
                for x, bt3 in adj[v]:
                    if x == r or x == u or x == v:
                        continue
                    d = int(dist[r, x])
                    if not (1 <= d <= 3):
                        raise RuntimeError(
                            f"length-3 path with shortest distance {d}; the four atoms are "
                            "distinct so dist(r, x) must be in {1,2,3}"
                        )
                    roots.append(r)
                    ends.append(x)
                    if keep_mid_atoms:
                        mids1.append(u)
                        mids2.append(v)
                    b1s.append(bt1)
                    b2s.append(bt2)
                    b3s.append(bt3)
                    ds.append(d)

    counts = np.bincount(np.asarray(roots, dtype=np.int64), minlength=n).astype(np.int64)
    ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    mid1_idx = np.asarray(mids1, dtype=np.int32)
    mid2_idx = np.asarray(mids2, dtype=np.int32)
    return S3Paths(
        ptr=ptr,
        end=np.asarray(ends, dtype=np.int32),
        atom_type_mid1=at[mid1_idx.astype(np.int64)].astype(np.int8)
        if keep_mid_atoms
        else np.zeros(len(ends), dtype=np.int8),
        atom_type_mid2=at[mid2_idx.astype(np.int64)].astype(np.int8)
        if keep_mid_atoms
        else np.zeros(len(ends), dtype=np.int8),
        bond_type_1=np.asarray(b1s, dtype=np.int8),
        bond_type_2=np.asarray(b2s, dtype=np.int8),
        bond_type_3=np.asarray(b3s, dtype=np.int8),
        dist=np.asarray(ds, dtype=np.int8),
        mid1=mid1_idx if keep_mid_atoms else None,
        mid2=mid2_idx if keep_mid_atoms else None,
    )


def count_s3_closed_form(n_atoms: int, adj: Sequence[Sequence[tuple[int, int]]]) -> np.ndarray:
    """Exact per-root path count without materialising the paths.

    For root ``r``, mid1 ``u`` and mid2 ``v`` the number of valid endpoints is
    ``deg(v) - 1 - [r adjacent v]``: ``u`` is always a neighbour of ``v`` and is
    excluded, ``r`` is excluded when the triangle ``r-u-v`` closes.  Summing over
    ``u in N(r)`` and ``v in N(u)\\{r}`` gives the count.  Used as an
    independent identity check of :func:`enumerate_s3` and for the full-cache
    size projection (it needs no path arrays).
    """
    n = int(n_atoms)
    nbr_sets = [set(nb for nb, _ in adj[i]) for i in range(n)]
    deg = np.asarray([len(nbr_sets[i]) for i in range(n)], dtype=np.int64)
    counts = np.zeros(n, dtype=np.int64)
    for r in range(n):
        total = 0
        for u in nbr_sets[r]:
            for v in nbr_sets[u]:
                if v == r:
                    continue
                total += int(deg[v]) - 1 - (1 if r in nbr_sets[v] else 0)
        counts[r] = total
    return counts


def reference_enumerate_s3(
    n_atoms: int,
    adj: Sequence[Sequence[tuple[int, int]]],
    atom_type: Sequence[int] | np.ndarray,
) -> list[list[dict[str, int]]]:
    """Independent, deliberately naive reference enumeration (audit only).

    Enumerates ordered 4-tuples with explicit distinctness and explicit bond
    lookups, so it shares no code path with :func:`enumerate_s3`.
    """
    n = int(n_atoms)
    at = np.asarray(atom_type, dtype=np.int64).reshape(-1)
    dist = all_pairs_shortest_paths(n, adj)
    out: list[list[dict[str, int]]] = [[] for _ in range(n)]
    for r in range(n):
        for u in range(n):
            if u == r or u not in {nb for nb, _ in adj[r]}:
                continue
            for v in range(n):
                if v in (r, u) or v not in {nb for nb, _ in adj[u]}:
                    continue
                for x in range(n):
                    if x in (r, u, v) or x not in {nb for nb, _ in adj[v]}:
                        continue
                    out[r].append(
                        {
                            "root_atom": r,
                            "mid_atom_1": u,
                            "mid_atom_2": v,
                            "end_atom": x,
                            "bond_type_1": bond_type_lookup(adj, r, u),
                            "bond_type_2": bond_type_lookup(adj, u, v),
                            "bond_type_3": bond_type_lookup(adj, v, x),
                            "mid_type_1": int(at[u]),
                            "mid_type_2": int(at[v]),
                            "shortest_distance": int(dist[r, x]),
                        }
                    )
    return out


# ---------------------------------------------------------------------------
# rooted balls + label-free local distinguishability
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RootedBall:
    """The induced subgraph of radius ``radius`` around ``root``.

    ``nodes`` is BFS order (deterministic), ``labels[i]`` is the atom type of
    ``nodes[i]``, ``is_root[i]`` flags the root, and ``edges`` are ``(i, j,
    bond_type)`` with ``i < j`` and both endpoints inside the ball.
    """

    radius: int
    root: int
    nodes: tuple[int, ...]
    labels: tuple[int, ...]
    is_root: tuple[bool, ...]
    edges: tuple[tuple[int, int, int], ...]
    dist_from_root: tuple[int, ...]

    def to_networkx(self):  # pragma: no cover - imported lazily
        import networkx as nx

        g = nx.Graph()
        for i, lab in enumerate(self.labels):
            g.add_node(i, atom_type=int(lab), is_root=bool(self.is_root[i]), dist=int(self.dist_from_root[i]))
        for i, j, bt in self.edges:
            g.add_edge(int(i), int(j), bond_type=int(bt))
        return g


def rooted_ball(
    n_atoms: int,
    adj: Sequence[Sequence[tuple[int, int]]],
    atom_type: Sequence[int] | np.ndarray,
    root: int,
    radius: int,
    dist: np.ndarray | None = None,
) -> RootedBall:
    """Exact rooted induced subgraph of the given radius (no truncation)."""
    n = int(n_atoms)
    r = int(root)
    if dist is None:
        dist = all_pairs_shortest_paths(n, adj)
    at = np.asarray(atom_type, dtype=np.int64).reshape(-1)
    nodes = [r]
    seen = {r}
    # BFS by distance, then by atom id -> deterministic node order
    by_dist: dict[int, list[int]] = {0: [r]}
    for d in range(1, int(radius) + 1):
        layer: list[int] = []
        for node in by_dist[d - 1]:
            for nb, _bt in adj[node]:
                if nb not in seen:
                    seen.add(nb)
                    layer.append(nb)
        layer.sort()
        by_dist[d] = layer
        nodes.extend(layer)
    order = {node: i for i, node in enumerate(nodes)}
    edges: list[tuple[int, int, int]] = []
    for node in nodes:
        i = order[node]
        for nb, bt in adj[node]:
            j = order.get(nb)
            if j is not None and i < j:
                edges.append((i, j, int(bt)))
    edges.sort()
    return RootedBall(
        radius=int(radius),
        root=r,
        nodes=tuple(int(x) for x in nodes),
        labels=tuple(int(at[x]) for x in nodes),
        is_root=tuple(bool(x == r) for x in nodes),
        edges=tuple(edges),
        dist_from_root=tuple(int(dist[r, x]) for x in nodes),
    )


def wl_signature(ball: RootedBall, *, max_iter: int | None = None) -> tuple[Any, ...]:
    """Rooted 1-WL colour-refinement signature of a ball (deterministic).

    Colours start from ``(atom_type, dist_from_root, is_root)`` and are refined
    by ``(colour, sorted multiset of (bond_type, neighbour colour))``.  Colour
    ids are assigned by sorting the unique refined objects, so the numbering is
    isomorphism-invariant without any hashing.  The signature is
    ``(n_nodes, n_edges, root_colour, sorted colour histogram, sorted edge-label
    histogram)``.

    Incompleteness: 1-WL cannot separate all non-isomorphic graphs (e.g. some
    regular graphs).  Equal signatures are therefore treated as "not proven
    different", and every witness pair in the audit is re-verified with the
    exact :func:`ball_isomorphic` test.
    """
    n = len(ball.nodes)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for i, j, bt in ball.edges:
        adj[i].append((j, bt))
        adj[j].append((i, bt))
    init = [(int(ball.labels[i]), int(ball.dist_from_root[i]), bool(ball.is_root[i])) for i in range(n)]
    uniq = {v: k for k, v in enumerate(sorted(set(init)))}
    colours = [uniq[v] for v in init]
    rounds = int(ball.radius) + 2 if max_iter is None else int(max_iter)
    for _ in range(rounds):
        sigs = [
            (colours[i], tuple(sorted((int(bt), int(colours[j])) for j, bt in adj[i])))
            for i in range(n)
        ]
        uniq2 = {v: k for k, v in enumerate(sorted(set(sigs)))}
        new = [uniq2[s] for s in sigs]
        if new == colours:
            break
        colours = new
    hist = tuple(sorted(np.bincount(np.asarray(colours, dtype=np.int64), minlength=max(colours) + 1).tolist()))
    edge_hist = tuple(sorted(np.bincount(np.asarray([bt for _i, _j, bt in ball.edges], dtype=np.int64)).tolist()))
    return (
        int(n),
        int(len(ball.edges)),
        int(colours[0]),
        hist,
        edge_hist,
        tuple(sorted(int(ball.labels[i]) for i in range(n) if ball.is_root[i])),
    )


def ball_isomorphic(a: RootedBall, b: RootedBall) -> bool:
    """Exact rooted isomorphism of two typed induced subgraphs (VF2).

    Node labels are the raw atom types plus the root flag (so the roots must map
    to each other), edge labels are the raw one-based bond types.  This is a
    complete test for the ball sizes involved here (the ball is extracted
    exhaustively, so no truncation enters the comparison).
    """
    from networkx.algorithms import isomorphism as iso

    def node_match(x: Mapping[str, Any], y: Mapping[str, Any]) -> bool:
        return (
            int(x["atom_type"]) == int(y["atom_type"])
            and bool(x["is_root"]) == bool(y["is_root"])
        )

    def edge_match(x: Mapping[str, Any], y: Mapping[str, Any]) -> bool:
        return int(x["bond_type"]) == int(y["bond_type"])

    return bool(
        iso.GraphMatcher(
            a.to_networkx(), b.to_networkx(), node_match=node_match, edge_match=edge_match
        ).is_isomorphic()
    )


def ball_signature(ball: RootedBall) -> str:
    """Strong content signature of a typed rooted ball (SHA-256, *screen only*).

    Every vertex starts with the label ``(atom_type, dist_from_root, is_root)``
    and is refined by ``(own content, sorted[(bond_type, neighbour content)])``
    for ``radius + 2`` rounds; the signature is the SHA-256 of the sorted vertex
    contents plus the root content.  Unlike a partition-only WL signature this
    keeps the **absolute** atom/bond labels, so it does not conflate different
    label sets that happen to have the same colour-class shape.

    It is still a hash: equal signatures do **not** prove isomorphism (a hash
    collision or a tree-unfolding ambiguity would both be silent).  The audit
    therefore uses this only to bucket candidate atoms and then assigns exact
    isomorphism classes with :func:`ball_isomorphic` inside each bucket.
    """
    import hashlib

    n = len(ball.nodes)
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for i, j, bt in ball.edges:
        adj[i].append((j, bt))
        adj[j].append((i, bt))
    content = [repr((int(ball.labels[i]), int(ball.dist_from_root[i]), bool(ball.is_root[i]))) for i in range(n)]
    for _ in range(int(ball.radius) + 2):
        content = [
            repr((content[i], sorted((int(bt), content[j]) for j, bt in adj[i]))) for i in range(n)
        ]
    digest = hashlib.sha256()
    for item in sorted(content):
        digest.update(item.encode("utf-8"))
        digest.update(b"|")
    digest.update(b"root:")
    digest.update(content[0].encode("utf-8"))
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# path-type bookkeeping (audit side)
# ---------------------------------------------------------------------------


def path_type_keys(paths: S3Paths) -> np.ndarray:
    """Factorised, order-preserving path type id per path.

    ``(bond_type_1, bond_type_2, bond_type_3, mid_type_1, mid_type_2, dist)``
    packed into one int64.  This is *only* an audit-side counter; the model never
    builds a joint vocabulary (it keeps the six factors separate, see
    ``s3_model``).
    """
    nb = 8  # 1..7 covers every one-based bond type with headroom
    na = 32  # 0..31 covers every atom type id with headroom
    nd = 4
    key = (
        ((paths.bond_type_1.astype(np.int64) - 1) * nb**2 + (paths.bond_type_2.astype(np.int64) - 1) * nb
         + (paths.bond_type_3.astype(np.int64) - 1)) * na**2
        + paths.atom_type_mid1.astype(np.int64) * na
        + paths.atom_type_mid2.astype(np.int64)
    ) * nd + paths.dist.astype(np.int64)
    return key


def decode_path_type(key: int) -> dict[str, int]:
    """Inverse of :func:`path_type_keys` (audit-side only)."""
    k = int(key)
    nb, na, nd = 8, 32, 4
    d = k % nd
    k //= nd
    m2 = k % na
    k //= na
    m1 = k % na
    k //= na
    b3 = k % nb + 1
    k //= nb
    b2 = k % nb + 1
    k //= nb
    b1 = k % nb + 1
    return {
        "bond_type_1": int(b1),
        "bond_type_2": int(b2),
        "bond_type_3": int(b3),
        "mid_type_1": int(m1),
        "mid_type_2": int(m2),
        "shortest_distance": int(d),
    }


__all__ = [
    "RootedBall",
    "S3Paths",
    "S3_DIST_CLASSES",
    "S3_N_ATOMS",
    "S3_N_BONDS",
    "UNREACHABLE",
    "all_pairs_shortest_paths",
    "ball_signature",
    "ball_isomorphic",
    "bond_type_lookup",
    "count_s3_closed_form",
    "decode_path_type",
    "enumerate_s3",
    "molecule_adjacency",
    "path_type_keys",
    "reference_enumerate_s3",
    "rooted_ball",
    "wl_signature",
]

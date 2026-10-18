"""SSRC Phase 0: exact CPU calibration for bounded rooted structural encodings.

This module is deliberately small and transparent.  It implements:

* two attributed graph families (a 24-cycle and a caterpillar / spine+leaves);
* exact rooted-radius-r induced subgraph extraction;
* an exact rooted-attributed-graph canonical key (pynauty), with an
  exhaustive permutation canonicalisation used as a cross-check on small
  patches;
* the multiset encoding ``E_r`` and collision / label-conflict statistics;
* two *oracles* that make the information-access difference explicit:
  - a depth-L dynamic oracle that propagates the two mark indicators on the
    actual graph (not a learned model);
  - a one-shot non-iterative oracle that reads the A--B distance (not a
    learned model).

The code never receives labels, sample ids, or generation seeds through the
encoder.  Node numbering is only used to *construct* a graph; every encoding is
a permutation-invariant multiset of exact rooted-patch keys.

Nothing here is a claim about molecular learning.  See
``results/phase0/REPORT.md``.
"""

from __future__ import annotations

import itertools
import random
from collections import Counter, defaultdict
from dataclasses import dataclass
from typing import Any, Iterable, Sequence

import networkx as nx

# ---------------------------------------------------------------------------
# attributed graph
# ---------------------------------------------------------------------------

TYPE_C = 0
TYPE_A = 1
TYPE_B = 2

EDGE_RING = 1
EDGE_SPINE = 2
EDGE_LEAF = 3
EDGE_GRID = 4

TYPE_NAME = {TYPE_C: "C", TYPE_A: "A", TYPE_B: "B"}
EDGE_NAME = {EDGE_RING: "ring", EDGE_SPINE: "spine", EDGE_LEAF: "leaf", EDGE_GRID: "grid"}


@dataclass(frozen=True)
class AttributedGraph:
    """Small undirected node/edge-attributed graph.

    ``edges`` are stored as ``(u, v, edge_type)`` with ``u < v``.  The graph is
    simple; self-loops and parallel edges are not used in this phase.
    """

    n: int
    node_types: tuple[int, ...]
    edges: tuple[tuple[int, int, int], ...]

    def __post_init__(self) -> None:
        if len(self.node_types) != self.n:
            raise ValueError("node_types length must equal n")
        for u, v, _ in self.edges:
            if not (0 <= u < self.n and 0 <= v < self.n):
                raise ValueError("edge endpoint out of range")
            if u >= v:
                raise ValueError("edges must be stored with u < v")

    def adjacency(self) -> dict[int, list[tuple[int, int]]]:
        adj: dict[int, list[tuple[int, int]]] = {v: [] for v in range(self.n)}
        for u, v, t in self.edges:
            adj[u].append((v, t))
            adj[v].append((u, t))
        for v in adj:
            adj[v].sort()
        return adj


def cycle_graph(n: int = 24, distance: int = 6) -> AttributedGraph:
    """C_n with A at 0 and B at graph distance ``distance`` (1 <= d <= n//2)."""
    if n < 3:
        raise ValueError("n must be >= 3")
    if not 1 <= distance <= n // 2:
        raise ValueError("distance must satisfy 1 <= distance <= n//2")
    types = [TYPE_C] * n
    types[0] = TYPE_A
    types[distance % n] = TYPE_B
    edges = [(i, (i + 1) % n, EDGE_RING) for i in range(n)]
    edges = [(min(u, v), max(u, v), t) for u, v, t in edges]
    return AttributedGraph(n=n, node_types=tuple(types), edges=tuple(sorted(edges)))


def caterpillar_graph(n_spine: int = 24, distance: int = 6, margin: int = 4) -> AttributedGraph:
    """A spine path with one leaf attached to every spine node.

    A is at spine position ``margin`` and B at ``margin + distance``.  Leaves
    never create a shortcut, so d(A, B) = ``distance``.  This is a second,
    non-cycle structural family: local balls are small trees with branches.
    """
    if margin < 0 or distance < 1:
        raise ValueError("margin >= 0 and distance >= 1")
    if margin + distance >= n_spine - margin:
        raise ValueError("marks must lie away from the spine ends")
    total = 2 * n_spine
    types = [TYPE_C] * total
    a = margin
    b = margin + distance
    types[a] = TYPE_A
    types[b] = TYPE_B
    edges: list[tuple[int, int, int]] = []
    for i in range(n_spine - 1):
        edges.append((i, i + 1, EDGE_SPINE))
    for i in range(n_spine):
        edges.append((i, n_spine + i, EDGE_LEAF))
    edges = [(min(u, v), max(u, v), t) for u, v, t in edges]
    return AttributedGraph(n=total, node_types=tuple(types), edges=tuple(sorted(edges)))


def grid_graph(
    width: int = 20,
    height: int = 9,
    distance: int = 6,
    a_x: int = 4,
    a_y: int = 4,
    margin: int = 3,
) -> AttributedGraph:
    """A 2-D rectangular grid with A and B on the same row.

    This is a genuinely non-1-D structural family (4-cycles everywhere).  The
    marks are kept away from the boundary by ``margin`` so that their radius-r
    balls are translation-invariant for r <= margin.
    """
    if width < 2 or height < 2:
        raise ValueError("grid must be at least 2x2")
    if not (0 <= a_y < height):
        raise ValueError("a_y out of range")
    b_x = a_x + distance
    if not (margin <= a_x and b_x < width - margin):
        raise ValueError("marks must lie away from the grid boundary")
    if not (margin <= a_y < height - margin):
        raise ValueError("a_y must lie away from the grid boundary")
    n = width * height

    def idx(x: int, y: int) -> int:
        return y * width + x

    types = [TYPE_C] * n
    types[idx(a_x, a_y)] = TYPE_A
    types[idx(b_x, a_y)] = TYPE_B
    edges: list[tuple[int, int, int]] = []
    for y in range(height):
        for x in range(width):
            if x + 1 < width:
                edges.append((idx(x, y), idx(x + 1, y), EDGE_GRID))
            if y + 1 < height:
                edges.append((idx(x, y), idx(x, y + 1), EDGE_GRID))
    return AttributedGraph(n=n, node_types=tuple(types), edges=tuple(sorted(edges)))


def markers(graph: AttributedGraph) -> tuple[int, int]:
    """Return (A_index, B_index)."""
    a = [i for i, t in enumerate(graph.node_types) if t == TYPE_A]
    b = [i for i, t in enumerate(graph.node_types) if t == TYPE_B]
    if len(a) != 1 or len(b) != 1:
        raise ValueError("graph must have exactly one A and one B")
    return a[0], b[0]


def bfs_distances(graph: AttributedGraph, root: int) -> dict[int, int]:
    adj = graph.adjacency()
    dist = {root: 0}
    queue = [root]
    for v in queue:
        for u, _ in adj[v]:
            if u not in dist:
                dist[u] = dist[v] + 1
                queue.append(u)
    return dist


def graph_distance(graph: AttributedGraph, source: int, target: int) -> int:
    dist = bfs_distances(graph, source)
    if target not in dist:
        raise ValueError("graph is disconnected")
    return dist[target]


# ---------------------------------------------------------------------------
# rooted patches and exact canonical keys
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class RootedPatch:
    """A rooted induced ball, with local node order (root is local index 0)."""

    node_types: tuple[int, ...]
    dists: tuple[int, ...]
    edges: tuple[tuple[int, int, int], ...]
    original_nodes: tuple[int, ...]

    @property
    def n(self) -> int:
        return len(self.node_types)

    def has_both_marks(self) -> bool:
        c = Counter(self.node_types)
        return c[TYPE_A] > 0 and c[TYPE_B] > 0

    def summary(self) -> dict[str, Any]:
        c = Counter(self.node_types)
        ec = Counter(t for _, _, t in self.edges)
        return {
            "n_nodes": self.n,
            "n_A": int(c[TYPE_A]),
            "n_B": int(c[TYPE_B]),
            "n_C": int(c[TYPE_C]),
            "max_dist": int(max(self.dists)) if self.dists else 0,
            "edge_type_counts": {EDGE_NAME[t]: int(ec[t]) for t in sorted(ec)},
            "has_both_marks": self.has_both_marks(),
        }


def rooted_ball(graph: AttributedGraph, root: int, radius: int) -> RootedPatch:
    """Exact induced subgraph on ``{u : d(u, root) <= radius}``.

    The root is placed at local index 0.  Distances from the root are carried
    as node colours; they are determined by the patch, but keeping them makes
    the rooted isomorphism explicit and eases auditing.
    """
    if radius < 0:
        raise ValueError("radius must be >= 0")
    dist = bfs_distances(graph, root)
    nodes = [root] + sorted(v for v, d in dist.items() if d <= radius and v != root)
    local = {v: i for i, v in enumerate(nodes)}
    node_types = tuple(graph.node_types[v] for v in nodes)
    dists = tuple(dist[v] for v in nodes)
    edges: list[tuple[int, int, int]] = []
    for u, v, t in graph.edges:
        if u in local and v in local:
            iu, iv = local[u], local[v]
            edges.append((min(iu, iv), max(iu, iv), t))
    edges.sort()
    return RootedPatch(
        node_types=node_types,
        dists=dists,
        edges=tuple(edges),
        original_nodes=tuple(nodes),
    )


def all_rooted_balls(graph: AttributedGraph, radius: int) -> list[RootedPatch]:
    return [rooted_ball(graph, v, radius) for v in range(graph.n)]


def exhaustive_canonical(patch: RootedPatch) -> tuple[Any, ...]:
    """Exact canonical form by enumerating all rooted permutations.

    Only use for small patches (n <= 8 here).  The key is a complete invariant
    for rooted node/edge-attributed graphs: any rooted isomorphism is a
    permutation of the non-root vertices, and the key retains node types,
    root distances and all edge types.
    """
    n = patch.n
    if n > 8:
        raise ValueError("exhaustive canonicalisation is intended for n <= 8")
    best: tuple[Any, ...] | None = None
    for perm in itertools.permutations(range(1, n)):
        order = (0,) + perm
        types = tuple(patch.node_types[i] for i in order)
        dists = tuple(patch.dists[i] for i in order)
        adj = [[0] * n for _ in range(n)]
        for u, v, t in patch.edges:
            iu, iv = order.index(u), order.index(v)
            if iu > iv:
                iu, iv = iv, iu
            adj[iu][iv] = t + 1
        flat = tuple(adj[i][j] for i in range(n) for j in range(i + 1, n))
        key = (types, dists, flat)
        if best is None or key < best:
            best = key
    assert best is not None
    return best


def pynauty_key_from_colors(
    node_colors: Sequence[tuple[int, ...]],
    edges: Sequence[tuple[int, int, tuple[int, ...]]],
) -> bytes:
    """Exact canonical key of a rooted node/edge-coloured simple graph.

    ``node_colors[i]`` and the edge colour tuple are arbitrary hashable tuples
    of ints.  Edge colours are encoded by subdividing every edge with a
    colour-bearing edge vertex.  pynauty then canonicalises the resulting
    simple vertex-coloured graph.  We append the canonical colour sequence (the
    same convention as the repo's ``colored_graph_key``) because the
    certificate alone is an adjacency certificate.
    """
    import pynauty

    n = len(node_colors)
    m = len(edges)
    nv = n + m
    colors: list[tuple[int, ...]] = []
    for i in range(n):
        colors.append((0,) + tuple(int(x) for x in node_colors[i]))
    adjacency: dict[int, list[int]] = {v: [] for v in range(nv)}
    for e, (u, v, edge_color) in enumerate(edges):
        edge_vertex = n + e
        colors.append((1,) + tuple(int(x) for x in edge_color))
        for endpoint in (u, v):
            adjacency[edge_vertex].append(int(endpoint))
            adjacency[int(endpoint)].append(edge_vertex)
    groups: dict[tuple[int, ...], list[int]] = defaultdict(list)
    for v, color in enumerate(colors):
        groups[color].append(v)
    cell_keys = sorted(groups, key=repr)
    coloring = [set(groups[k]) for k in cell_keys]
    pg = pynauty.Graph(
        number_of_vertices=nv,
        directed=False,
        adjacency_dict={v: sorted(adj) for v, adj in adjacency.items()},
        vertex_coloring=coloring,
    )
    certificate = bytes(pynauty.certificate(pg))
    label = [int(v) for v in pynauty.canon_label(pg)]
    if sorted(label) != list(range(nv)):
        raise RuntimeError("pynauty.canon_label is not a permutation")
    color_sequence = tuple(colors[v] for v in label)
    return certificate + b"|" + repr(color_sequence).encode()


def pynauty_key(patch: RootedPatch) -> bytes:
    """Exact canonical key of a rooted patch (toy integer attributes)."""
    node_colors = [
        (int(patch.node_types[i]), 1 if i == 0 else 0, int(patch.dists[i])) for i in range(patch.n)
    ]
    edges = [(int(u), int(v), (int(t),)) for u, v, t in patch.edges]
    return pynauty_key_from_colors(node_colors, edges)


def nx_isomorphic_rooted(p1: RootedPatch, p2: RootedPatch) -> bool:
    """Independent exact rooted-isomorphism check via networkx GraphMatcher."""
    if p1.n != p2.n or len(p1.edges) != len(p2.edges):
        return False

    def build(p: RootedPatch) -> nx.Graph:
        g = nx.Graph()
        for i in range(p.n):
            g.add_node(i, color=(int(p.node_types[i]), int(p.dists[i]), i == 0))
        for u, v, t in p.edges:
            g.add_edge(u, v, color=int(t))
        return g

    g1, g2 = build(p1), build(p2)
    matcher = nx.algorithms.isomorphism.GraphMatcher(
        g1,
        g2,
        node_match=lambda a, b: a["color"] == b["color"],
        edge_match=lambda a, b: a["color"] == b["color"],
    )
    return bool(matcher.is_isomorphic())


def encoding(graph: AttributedGraph, radius: int) -> tuple[bytes, ...]:
    """The multiset ``E_r`` as a sorted tuple of exact rooted-patch keys."""
    return tuple(sorted(pynauty_key(p) for p in all_rooted_balls(graph, radius)))


def encoding_stats(graph: AttributedGraph, radius: int) -> dict[str, Any]:
    patches = all_rooted_balls(graph, radius)
    return {
        "n_roots": len(patches),
        "n_unique_patch_keys": len(set(pynauty_key(p) for p in patches)),
        "n_patches_with_both_marks": sum(p.has_both_marks() for p in patches),
        "max_patch_nodes": max(p.n for p in patches),
        "mean_patch_nodes": sum(p.n for p in patches) / len(patches),
    }


# ---------------------------------------------------------------------------
# relabelling invariance
# ---------------------------------------------------------------------------


def relabel_graph(graph: AttributedGraph, rng: random.Random) -> AttributedGraph:
    """Apply a random node permutation; node/edge attributes are carried along."""
    new_of_old = list(range(graph.n))
    rng.shuffle(new_of_old)
    types = [TYPE_C] * graph.n
    for old, new in enumerate(new_of_old):
        types[new] = graph.node_types[old]
    edges = [(min(new_of_old[u], new_of_old[v]), max(new_of_old[u], new_of_old[v]), t) for u, v, t in graph.edges]
    return AttributedGraph(n=graph.n, node_types=tuple(types), edges=tuple(sorted(edges)))


def check_relabel_invariance(
    graph: AttributedGraph, radii: Iterable[int], n_permutations: int = 50, seed: int = 20261018
) -> dict[str, Any]:
    rng = random.Random(seed)
    reference = {r: encoding(graph, r) for r in radii}
    failures: list[str] = []
    for trial in range(n_permutations):
        permuted = relabel_graph(graph, rng)
        for r in radii:
            if encoding(permuted, r) != reference[r]:
                failures.append(f"trial={trial} r={r}")
    return {
        "n_permutations": n_permutations,
        "seed": seed,
        "radii": list(radii),
        "failures": failures,
        "passed": not failures,
    }


# ---------------------------------------------------------------------------
# collision / label-conflict statistics
# ---------------------------------------------------------------------------


def collision_stats(encodings: dict[int, tuple[bytes, ...]], label_threshold: int = 4) -> dict[str, Any]:
    """Group graphs by ``E_r`` and report collisions and label conflicts.

    ``encodings`` maps a scalar structural parameter (here the A--B distance)
    to its multiset key.  The label is ``d <= label_threshold``.
    """
    groups: dict[tuple[bytes, ...], list[int]] = defaultdict(list)
    for d, key in encodings.items():
        groups[key].append(d)

    d_values = sorted(encodings)
    pairs = 0
    collision_pairs: list[tuple[int, int]] = []
    for key, ds in groups.items():
        k = len(ds)
        if k > 1:
            pairs += k * (k - 1) // 2
            for i in range(k):
                for j in range(i + 1, k):
                    collision_pairs.append((ds[i], ds[j]))
    label_mixed_groups = 0
    label_conflict_graphs = 0
    for key, ds in groups.items():
        labs = {int(d <= label_threshold) for d in ds}
        if len(labs) > 1:
            label_mixed_groups += 1
            label_conflict_graphs += len(ds)
    ranges = [max(ds) - min(ds) for ds in groups.values()]
    return {
        "n_graphs": len(d_values),
        "n_distinct_encodings": len(groups),
        "n_collision_pairs": pairs,
        "collision_pairs": sorted(collision_pairs),
        "max_group_size": max(len(ds) for ds in groups.values()),
        "label_threshold": label_threshold,
        "label_mixed_groups": label_mixed_groups,
        "label_conflict_graphs": label_conflict_graphs,
        "label_conflict_graph_fraction": label_conflict_graphs / len(d_values),
        "mean_within_group_distance_range": sum(ranges) / len(ranges),
        "max_within_group_distance_range": max(ranges),
        "same_6_9": (encodings[6] == encodings[9]) if (6 in encodings and 9 in encodings) else None,
        "groups": [
            {"d_values": sorted(ds), "n": len(ds)}
            for ds in sorted(groups.values(), key=lambda x: (min(x), len(x)))
        ],
    }


def same_encoding_pair(
    graph_factory,
    distance_a: int,
    distance_b: int,
    radius: int,
    **kwargs: Any,
) -> bool:
    ga = graph_factory(distance=distance_a, **kwargs)
    gb = graph_factory(distance=distance_b, **kwargs)
    return encoding(ga, radius) == encoding(gb, radius)


# ---------------------------------------------------------------------------
# oracles (not learned models)
# ---------------------------------------------------------------------------


def dynamic_oracle_detect(graph: AttributedGraph, depth: int) -> bool:
    """Depth-L propagation of the two mark indicators on the real graph.

    This is a hand-written nonlinear detector: after L rounds of sum
    aggregation, a node has both marks in its L-hop neighbourhood iff both
    coordinates are positive.  It demonstrates that the *arrangement* of
    patches is information the dynamic model can access.  It is NOT a trained
    model and is NOT evidence of learnability.
    """
    if depth < 0:
        raise ValueError("depth must be >= 0")
    a, b = markers(graph)
    adj = graph.adjacency()
    h = [[1 if v == a else 0, 1 if v == b else 0] for v in range(graph.n)]
    for _ in range(depth):
        new = [row[:] for row in h]
        for v in range(graph.n):
            for u, _ in adj[v]:
                new[v][0] += h[u][0]
                new[v][1] += h[u][1]
        h = new
    return any(row[0] > 0 and row[1] > 0 for row in h)


def noniter_distance_oracle(graph: AttributedGraph) -> int:
    """One-shot non-iterative relational readout: the A--B graph distance."""
    a, b = markers(graph)
    return graph_distance(graph, a, b)


def both_marks_count(graph: AttributedGraph, radius: int) -> int:
    return sum(p.has_both_marks() for p in all_rooted_balls(graph, radius))


def oracle_positive_control(graph: AttributedGraph, radius: int, threshold: int = 4) -> tuple[int, int]:
    """Deterministic static readout: label 1 iff some r-ball contains both marks.

    Returns ``(prediction, truth)``.  On the cycle/caterpillar families this is
    exact for ``label = [d <= 2r]``.  It is an oracle positive control for the
    *encoding*, not a learned model.
    """
    a, b = markers(graph)
    d = graph_distance(graph, a, b)
    pred = int(both_marks_count(graph, radius) > 0)
    truth = int(d <= threshold)
    return pred, truth

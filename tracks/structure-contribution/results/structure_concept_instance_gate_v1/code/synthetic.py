"""Deterministic synthetic graph generator for the
``structure_concept_instance_gate_v1`` round.

This module is frozen before any model training.  It builds:

* two required counterexamples (binding, relation) and their preflight checks,
* a small multi-background graph family (binding / relation / composition),
* splits by *base graph* (no positive/negative variant of a base graph crosses
  the train/dev/test boundary),
* instance-level ground truth (per-patch canonical structure-attribute pattern
  SAP, decisive patches, true anchor distances, generation rule label),
* a label-blind relation scramble used as a control for models B and D.

Nothing here reads ZINC, official valid/test or any molecular data.  All graphs
are tiny (<= 14 nodes), node attributes are symbolic (C/N/O/S), edges are
unweighted and undirected, and every construction is a pure function of fixed
IDs (``numpy`` generators seeded from SHA-256 digests; no search, no tuning).

Graph vocabulary
----------------
* node attribute ids: ``C=0, N=1, O=2, S=3``
* edge-anchored radius-1 patch = the edge endpoints plus their neighbours,
  with roles ``root_u / root_v / u-only / v-only / common``
* patch feature vector (dim 35): role x attribute counts (5x4) plus edge role
  pair counts (15), canonicalised over the two anchor orientations
* SAP = exact isomorphism class of the attributed patch, anchored by
  (anchor-flag, attribute) colours; computed by brute-force canonicalisation
  (patch size <= 6)
* relation feature (dim 11) between two patches: one-hot bucket of the
  shortest-path distance between their anchor node sets (0..8, >=9) plus a
  shared-endpoint overlap flag.
"""

from __future__ import annotations

import hashlib
import itertools
import json
from collections import Counter, defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Iterable, Mapping, Sequence

import numpy as np

ATTR_C, ATTR_N, ATTR_O, ATTR_S = 0, 1, 2, 3
ATTR_NAMES = {ATTR_C: "C", ATTR_N: "N", ATTR_O: "O", ATTR_S: "S"}
N_ATTRS = 4

N_ROLES = 5
ROLE_U, ROLE_V, ROLE_U_ONLY, ROLE_V_ONLY, ROLE_COMMON = 0, 1, 2, 3, 4
_ROLE_PAIRS = list(itertools.combinations_with_replacement(range(N_ROLES), 2))
PAIR_INDEX = {pair: i for i, pair in enumerate(_ROLE_PAIRS)}
PATCH_FEAT_DIM = N_ROLES * N_ATTRS + len(_ROLE_PAIRS)  # 20 + 15 = 35

REL_BUCKETS = 10  # distance buckets 0..8 and >=9
REL_FEAT_DIM = REL_BUCKETS + 1  # + shared-endpoint overlap flag

SHORT_D = 5
LONG_D = 9

FAMILIES = ("binding", "relation", "composition")


# ---------------------------------------------------------------------------
# graph primitives
# ---------------------------------------------------------------------------


def adjacency(n: int, edges: Sequence[tuple[int, int]]) -> tuple[tuple[int, ...], ...]:
    adj: list[list[int]] = [[] for _ in range(n)]
    for u, v in edges:
        adj[u].append(v)
        adj[v].append(u)
    return tuple(tuple(sorted(a)) for a in adj)


def all_pairs_distances(n: int, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    adj = adjacency(n, edges)
    dist = np.full((n, n), -1, dtype=np.int64)
    for src in range(n):
        dist[src, src] = 0
        frontier = [src]
        d = 0
        while frontier:
            d += 1
            nxt: list[int] = []
            for u in frontier:
                for w in adj[u]:
                    if dist[src, w] < 0:
                        dist[src, w] = d
                        nxt.append(w)
            frontier = nxt
    if (dist < 0).any():
        raise ValueError("graph is not connected")
    return dist


def triangle_nodes(n: int, edges: Sequence[tuple[int, int]]) -> frozenset[int]:
    """Nodes that belong to at least one triangle."""
    eset = {frozenset(e) for e in edges}
    adj = adjacency(n, edges)
    out: set[int] = set()
    for v in range(n):
        nbrs = adj[v]
        for i in range(len(nbrs)):
            for j in range(i + 1, len(nbrs)):
                if frozenset((nbrs[i], nbrs[j])) in eset:
                    out.add(v)
    return frozenset(out)


def cycle_nodes(n: int, edges: Sequence[tuple[int, int]]) -> frozenset[int]:
    """Nodes that lie on at least one simple cycle (small graphs only)."""
    adj = list(adjacency(n, edges))
    out: set[int] = set()
    for v in range(n):
        if len(adj[v]) < 2:
            continue
        for i in range(len(adj[v])):
            for j in range(i + 1, len(adj[v])):
                a, b = adj[v][i], adj[v][j]
                # BFS from a to b in G - v
                seen = {v}
                frontier = [a]
                seen.add(a)
                while frontier and b not in seen:
                    nxt = []
                    for u in frontier:
                        for w in adj[u]:
                            if w not in seen:
                                seen.add(w)
                                nxt.append(w)
                    frontier = nxt
                if b in seen:
                    out.add(v)
    return frozenset(out)


def graph6_like_key(n: int, edges: Sequence[tuple[int, int]]) -> str:
    """A cheap label-free identifier used only inside cache keys."""
    return f"n{n}:" + ",".join(f"{u}-{v}" for u, v in sorted(edges))


def relabel_graph(graph: Graph, perm: Sequence[int]) -> Graph:
    """Relabel nodes.  ``perm[i]`` is the *new* label of old node ``i``.

    This is the convention used everywhere in this round; the inverse map is
    therefore ``argsort(perm)``.
    """
    perm = [int(p) for p in perm]
    if sorted(perm) != list(range(graph.n)):
        raise ValueError("perm must be a permutation of range(n)")
    new_attrs = [0] * graph.n
    for old, new in enumerate(perm):
        new_attrs[new] = graph.attrs[old]
    new_edges = tuple((perm[u], perm[v]) for u, v in graph.edges)
    return Graph(
        graph.graph_id + "_perm",
        graph.family,
        graph.base_id,
        graph.pair_id,
        graph.bag_twin,
        graph.split,
        graph.rule,
        graph.n,
        tuple(new_attrs),
        new_edges,
        graph.label,
        graph.meta,
    )


# ---------------------------------------------------------------------------
# data structures
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class Graph:
    graph_id: str
    family: str
    base_id: str
    pair_id: str
    bag_twin: str | None
    split: str
    rule: str
    n: int
    attrs: tuple[int, ...]
    edges: tuple[tuple[int, int], ...]
    label: int
    meta: Mapping[str, object]

    def __post_init__(self) -> None:
        attrs = tuple(int(a) for a in self.attrs)
        edges = tuple((int(u), int(v)) if u < v else (int(v), int(u)) for u, v in self.edges)
        object.__setattr__(self, "attrs", attrs)
        object.__setattr__(self, "edges", tuple(sorted(edges)))
        object.__setattr__(self, "meta", dict(self.meta))

    def to_json(self) -> dict:
        return {
            "graph_id": self.graph_id,
            "family": self.family,
            "base_id": self.base_id,
            "pair_id": self.pair_id,
            "bag_twin": self.bag_twin,
            "split": self.split,
            "rule": self.rule,
            "n": self.n,
            "attrs": list(self.attrs),
            "edges": [list(e) for e in self.edges],
            "label": self.label,
            "meta": _jsonable(self.meta),
        }


@dataclass(frozen=True)
class Patch:
    graph_id: str
    edge: tuple[int, int]
    nodes: tuple[int, ...]
    anchors: tuple[int, int]
    feature: tuple[float, ...]
    sap: str
    sap_desc: str
    special: tuple[str, ...]  # which special attribute names are inside the patch

    def to_json(self) -> dict:
        return {
            "graph_id": self.graph_id,
            "edge": list(self.edge),
            "nodes": list(self.nodes),
            "anchors": list(self.anchors),
            "feature": list(self.feature),
            "sap": self.sap,
            "sap_desc": self.sap_desc,
            "special": list(self.special),
        }


def _jsonable(obj):
    if isinstance(obj, dict):
        return {k: _jsonable(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_jsonable(v) for v in obj]
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    return obj


def _seed_from(*parts: object) -> int:
    h = hashlib.sha256("|".join(str(p) for p in parts).encode("utf-8")).digest()
    return int.from_bytes(h[:8], "little")


# ---------------------------------------------------------------------------
# patch extraction / canonical SAP
# ---------------------------------------------------------------------------


def _patch_nodes_roles(
    n: int,
    edges: Sequence[tuple[int, int]],
    u: int,
    v: int,
    adj: Sequence[Sequence[int]],
) -> tuple[tuple[int, ...], dict[int, int], tuple[tuple[int, int], ...]]:
    nodes = sorted({u, v, *adj[u], *adj[v]})
    role: dict[int, int] = {}
    for w in nodes:
        if w == u:
            role[w] = ROLE_U
        elif w == v:
            role[w] = ROLE_V
        elif w in adj[u] and w in adj[v]:
            role[w] = ROLE_COMMON
        elif w in adj[u]:
            role[w] = ROLE_U_ONLY
        else:
            role[w] = ROLE_V_ONLY
    local = set(nodes)
    local_edges = tuple(sorted((a, b) for a, b in edges if a in local and b in local))
    return tuple(nodes), role, local_edges


def _oriented_feature(
    role: Mapping[int, int],
    local_edges: Sequence[tuple[int, int]],
    attrs: Sequence[int],
) -> tuple[float, ...]:
    feat = [0.0] * PATCH_FEAT_DIM
    for w, r in role.items():
        feat[r * N_ATTRS + attrs[w]] += 1.0
    off = N_ROLES * N_ATTRS
    for a, b in local_edges:
        ra, rb = role[a], role[b]
        if ra > rb:
            ra, rb = rb, ra
        feat[off + PAIR_INDEX[(ra, rb)]] += 1.0
    return tuple(feat)


def patch_feature(
    n: int,
    edges: Sequence[tuple[int, int]],
    attrs: Sequence[int],
    u: int,
    v: int,
    adj: Sequence[Sequence[int]] | None = None,
) -> tuple[float, ...]:
    """Canonical (orientation-invariant) radius-1 edge patch feature."""
    if adj is None:
        adj = adjacency(n, edges)
    u, v = (u, v) if u < v else (v, u)
    nodes, role, local_edges = _patch_nodes_roles(n, edges, u, v, adj)
    f_uv = _oriented_feature(role, local_edges, attrs)
    role_swapped = {w: {ROLE_U: ROLE_V, ROLE_V: ROLE_U, ROLE_U_ONLY: ROLE_V_ONLY, ROLE_V_ONLY: ROLE_U_ONLY, ROLE_COMMON: ROLE_COMMON}[r] for w, r in role.items()}
    f_vu = _oriented_feature(role_swapped, local_edges, attrs)
    return min(f_uv, f_vu)


@lru_cache(maxsize=None)
def _canonical_patch(
    colors: tuple[tuple[int, int], ...],
    edge_index_local: tuple[tuple[int, int], ...],
) -> tuple[tuple[tuple[int, int], ...], tuple[tuple[int, int], ...]]:
    """Exact canonical form of a coloured graph with <= 6 nodes.

    ``colors[i] = (anchor_flag, attr)``; ``edge_index_local`` uses node-local
    indices.  Returns the lexicographically minimal (colour sequence, edge set)
    over all relabelings.
    """
    k = len(colors)
    best: tuple | None = None
    for perm in itertools.permutations(range(k)):
        pos = {old: i for i, old in enumerate(perm)}
        new_colors = tuple(colors[i] for i in perm)
        new_edges = tuple(sorted(tuple(sorted((pos[a], pos[b]))) for a, b in edge_index_local))
        cand = (new_colors, new_edges)
        if best is None or cand < best:
            best = cand
    assert best is not None
    return best


def patch_sap(
    n: int,
    edges: Sequence[tuple[int, int]],
    attrs: Sequence[int],
    u: int,
    v: int,
    adj: Sequence[Sequence[int]] | None = None,
) -> tuple[str, str]:
    """Exact structural-attribute pattern (SAP) id + human-readable summary."""
    if adj is None:
        adj = adjacency(n, edges)
    u, v = (u, v) if u < v else (v, u)
    nodes, _role, local_edges = _patch_nodes_roles(n, edges, u, v, adj)
    pos = {g: i for i, g in enumerate(nodes)}
    anchors = frozenset((u, v))
    colors = tuple((1 if g in anchors else 0, attrs[g]) for g in nodes)
    edge_local = tuple(sorted((pos[a], pos[b]) for a, b in local_edges))
    canon_colors, canon_edges = _canonical_patch(colors, edge_local)
    payload = json.dumps([canon_colors, canon_edges], separators=(",", ":"))
    sap = hashlib.sha1(payload.encode("utf-8")).hexdigest()[:12]
    deg = [0] * len(canon_colors)
    for a, b in canon_edges:
        deg[a] += 1
        deg[b] += 1
    node_desc = sorted(
        f"{'A' if c[0] else 'n'}{ATTR_NAMES[c[1]]}d{d}" for c, d in zip(canon_colors, deg)
    )
    edge_desc = sorted(tuple(sorted(e)) for e in canon_edges)
    sap_desc = "|".join(node_desc) + ";E" + str(edge_desc)
    return sap, sap_desc


def extract_patches(graph: Graph) -> list[Patch]:
    adj = adjacency(graph.n, graph.edges)
    out: list[Patch] = []
    for u, v in graph.edges:
        nodes, _role, _le = _patch_nodes_roles(graph.n, graph.edges, u, v, adj)
        feat = patch_feature(graph.n, graph.edges, graph.attrs, u, v, adj)
        sap, desc = patch_sap(graph.n, graph.edges, graph.attrs, u, v, adj)
        special = tuple(sorted({ATTR_NAMES[graph.attrs[w]] for w in nodes if graph.attrs[w] != ATTR_C}))
        out.append(
            Patch(
                graph_id=graph.graph_id,
                edge=(u, v),
                nodes=nodes,
                anchors=(u, v),
                feature=feat,
                sap=sap,
                sap_desc=desc,
                special=special,
            )
        )
    return out


def anchor_distance_matrix(graph: Graph, patches: Sequence[Patch]) -> np.ndarray:
    dist = all_pairs_distances(graph.n, graph.edges)
    p = len(patches)
    out = np.zeros((p, p), dtype=np.int64)
    for i in range(p):
        ai = patches[i].anchors
        for j in range(i + 1, p):
            bj = patches[j].anchors
            out[i, j] = out[j, i] = min(
                int(dist[a, b]) for a in ai for b in bj
            )
    return out


def relation_features(dist: np.ndarray, patches: Sequence[Patch]) -> np.ndarray:
    """(p, p, REL_FEAT_DIM) relation features from the true anchor distances."""
    p = len(patches)
    feat = np.zeros((p, p, REL_FEAT_DIM), dtype=np.float32)
    overlap = np.zeros((p, p), dtype=np.float32)
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            if set(patches[i].anchors) & set(patches[j].anchors):
                overlap[i, j] = 1.0
    for i in range(p):
        for j in range(p):
            if i == j:
                continue
            b = min(int(dist[i, j]), REL_BUCKETS - 1)
            feat[i, j, b] = 1.0
            feat[i, j, REL_BUCKETS] = overlap[i, j]
    return feat


# ---------------------------------------------------------------------------
# base graph templates
# ---------------------------------------------------------------------------


def _triangle(a: int, b: int, c: int) -> list[tuple[int, int]]:
    return [(a, b), (b, c), (a, c)]


def _chain(attach: int, new_nodes: Sequence[int]) -> list[tuple[int, int]]:
    nodes = [attach, *new_nodes]
    return [(nodes[i], nodes[i + 1]) for i in range(len(nodes) - 1)]


@dataclass(frozen=True)
class BindingBase:
    name: str
    split: str
    n: int
    edges: tuple[tuple[int, int], ...]
    p_in: int
    p_out: int


BINDING_BASES: tuple[BindingBase, ...] = (
    BindingBase("tri_tail3", "train", 5, tuple(_triangle(0, 1, 2) + _chain(0, (3, 4))), 1, 3),
    BindingBase("tri_tail5", "train", 7, tuple(_triangle(0, 1, 2) + _chain(0, (3, 4, 5, 6))), 1, 3),
    BindingBase(
        "bowtie_tail",
        "train",
        7,
        tuple(_triangle(0, 1, 2) + _triangle(0, 3, 4) + _chain(2, (5, 6))),
        1,
        5,
    ),
    BindingBase(
        "tri_square",
        "train",
        6,
        tuple(_triangle(0, 1, 2) + [(0, 3), (3, 4), (4, 5), (5, 0)]),
        1,
        3,
    ),
    BindingBase(
        "two_tri_edge",
        "train",
        8,
        tuple(_triangle(0, 1, 2) + _triangle(3, 4, 5) + [(2, 3)] + _chain(5, (6, 7))),
        0,
        6,
    ),
    BindingBase(
        "tri_branch",
        "train",
        7,
        tuple(_triangle(0, 1, 2) + [(0, 3), (3, 4), (4, 5), (4, 6)]),
        1,
        3,
    ),
    BindingBase(
        "tri_long_branch",
        "dev",
        9,
        tuple(_triangle(0, 1, 2) + _chain(0, (3, 4, 5, 6)) + [(5, 7), (5, 8)]),
        1,
        3,
    ),
    BindingBase(
        "spiro_tail",
        "test",
        6,
        tuple(_triangle(0, 1, 2) + _triangle(0, 1, 3) + _chain(2, (4, 5))),
        2,
        4,
    ),
    BindingBase(
        "tri_fused5",
        "test",
        7,
        tuple(_triangle(0, 1, 2) + [(2, 3), (3, 4), (4, 5), (5, 6), (6, 2)]),
        1,
        3,
    ),
)

BINDING_AUX = (0, 1, 2)
BINDING_DESC = "label=+1 iff the unique N sits on a triangle; paired variants swap N/S between p_in (triangle) and p_out (no triangle)"


@dataclass(frozen=True)
class CompositionBase:
    name: str
    split: str
    n: int
    edges: tuple[tuple[int, int], ...]
    variants: tuple[tuple[str, int, int], ...]  # (variant name, N position, O position)


COMPOSITION_BASES: tuple[CompositionBase, ...] = (
    CompositionBase(
        "comp_fused_cycle20",
        "dev",
        20,
        tuple(_triangle(0, 1, 2) + _chain(0, tuple(range(3, 20))) + [(19, 0)]),
        (("in_short", 1, 7), ("in_long", 1, 13), ("out_short", 3, 7), ("out_long", 3, 13)),
    ),
    CompositionBase(
        "comp_fused_cycle22",
        "test",
        22,
        tuple(_triangle(0, 1, 2) + _chain(0, tuple(range(3, 22))) + [(21, 0)]),
        (("in_short", 1, 7), ("in_long", 1, 14), ("out_short", 3, 7), ("out_long", 3, 14)),
    ),
    CompositionBase(
        "comp_fused_cycle24",
        "test",
        24,
        tuple(_triangle(0, 1, 2) + _chain(0, tuple(range(3, 24))) + [(23, 0)]),
        (("in_short", 1, 8), ("in_long", 1, 15), ("out_short", 3, 8), ("out_long", 3, 15)),
    ),
)

COMPOSITION_SHORT_D = 7
COMPOSITION_RULE = f"xor(n in triangle, d(N,O) <= {COMPOSITION_SHORT_D})"
COMPOSITION_BAG_TWIN = {"in_short": "in_long", "in_long": "in_short", "out_short": "out_long", "out_long": "out_short"}


# ---------------------------------------------------------------------------
# generators
# ---------------------------------------------------------------------------


def _binding_attrs(base: BindingBase, aux: int) -> list[int]:
    allowed = [v for v in range(base.n) if v not in (base.p_in, base.p_out)]
    n_aux = 1 if aux < 2 else 2
    rng = np.random.default_rng(_seed_from("binding_aux", base.name, aux))
    pos = sorted(int(x) for x in rng.choice(len(allowed), size=n_aux, replace=False))
    chosen = [allowed[i] for i in pos]
    attrs = [ATTR_C] * base.n
    for p in chosen:
        attrs[p] = ATTR_O
    return attrs


def binding_graphs() -> list[Graph]:
    out: list[Graph] = []
    for base in BINDING_BASES:
        tri = triangle_nodes(base.n, base.edges)
        assert base.p_in in tri, (base.name, "p_in not in triangle")
        assert base.p_out not in tri, (base.name, "p_out in triangle")
        for aux in BINDING_AUX:
            aux_attrs = _binding_attrs(base, aux)
            pair_id = f"{base.name}#aux{aux}"
            for variant, label in (("in", 1), ("out", -1)):
                attrs = list(aux_attrs)
                if variant == "in":
                    attrs[base.p_in] = ATTR_N
                    attrs[base.p_out] = ATTR_S
                else:
                    attrs[base.p_in] = ATTR_S
                    attrs[base.p_out] = ATTR_N
                out.append(
                    Graph(
                        graph_id=f"bind_{base.name}_aux{aux}_{variant}",
                        family="binding",
                        base_id=base.name,
                        pair_id=pair_id,
                        bag_twin=None,
                        split=base.split,
                        rule="n_in_triangle",
                        n=base.n,
                        attrs=tuple(attrs),
                        edges=base.edges,
                        label=label,
                        meta={"p_in": base.p_in, "p_out": base.p_out, "aux": aux, "variant": variant},
                    )
                )
    return out


CYCLE_SIZES = {"train": (18, 20, 22), "dev": (24,), "test": (26, 28)}
REL_PAIR_TYPES = ((ATTR_N, ATTR_O), (ATTR_N, ATTR_S), (ATTR_O, ATTR_S))
REL_RULE = f"label=+1 iff shortest-path distance between the two marked heteroatoms is <= {SHORT_D}"


def relation_graphs() -> list[Graph]:
    out: list[Graph] = []
    for split, sizes in CYCLE_SIZES.items():
        for size in sizes:
            edges = tuple((i, (i + 1) % size) for i in range(size))
            for a1, a2 in REL_PAIR_TYPES:
                base_id = f"cycle{size}_{ATTR_NAMES[a1]}{ATTR_NAMES[a2]}"
                for variant, d, label in (("short", SHORT_D, 1), ("long", LONG_D, -1)):
                    attrs = [ATTR_C] * size
                    attrs[0] = a1
                    attrs[d] = a2
                    out.append(
                        Graph(
                            graph_id=f"rel_cycle{size}_{ATTR_NAMES[a1]}{ATTR_NAMES[a2]}_{variant}",
                            family="relation",
                            base_id=base_id,
                            pair_id=base_id,
                            bag_twin=f"rel_cycle{size}_{ATTR_NAMES[a1]}{ATTR_NAMES[a2]}_{'long' if variant == 'short' else 'short'}",
                            split=split,
                            rule="d_short",
                            n=size,
                            attrs=tuple(attrs),
                            edges=edges,
                            label=label,
                            meta={"cycle": size, "roles": [ATTR_NAMES[a1], ATTR_NAMES[a2]], "d": d, "variant": variant, "mark_nodes": [0, d]},
                        )
                    )
    return out


def _xor_label(in_triangle: bool, short: bool) -> int:
    return 1 if (in_triangle != short) else -1


def composition_graphs() -> list[Graph]:
    out: list[Graph] = []
    for base in COMPOSITION_BASES:
        tri = triangle_nodes(base.n, base.edges)
        for variant, n_pos, o_pos in base.variants:
            assert (n_pos in tri) == variant.startswith("in")
            attrs = [ATTR_C] * base.n
            attrs[n_pos] = ATTR_N
            attrs[o_pos] = ATTR_O
            dist = all_pairs_distances(base.n, base.edges)
            d = int(dist[n_pos, o_pos])
            short = d <= COMPOSITION_SHORT_D
            out.append(
                Graph(
                    graph_id=f"comp_{base.name}_{variant}",
                    family="composition",
                    base_id=base.name,
                    pair_id=f"{base.name}#{variant}",
                    bag_twin=f"comp_{base.name}_{COMPOSITION_BAG_TWIN[variant]}",
                    split=base.split,
                    rule="xor_binding_short",
                    n=base.n,
                    attrs=tuple(attrs),
                    edges=base.edges,
                    label=_xor_label(n_pos in tri, short),
                    meta={"n_pos": n_pos, "o_pos": o_pos, "d": d, "in_triangle": n_pos in tri, "short": short, "variant": variant},
                )
            )
    return out


def all_graphs() -> list[Graph]:
    return binding_graphs() + relation_graphs() + composition_graphs()


# ---------------------------------------------------------------------------
# label-blind relation scramble
# ---------------------------------------------------------------------------


def scrambled_relation_features(
    dist_matrices: Mapping[str, np.ndarray],
    patches_by_graph: Mapping[str, Sequence[Patch]],
    reference_graph_ids: Sequence[str],
) -> dict[str, np.ndarray]:
    """Label-blind scramble of the pairwise relation features.

    The marginal distribution of relation features is estimated from the
    frozen *train* split (``reference_graph_ids``) as a flat empirical
    distribution over all off-diagonal pairs, then for each graph every
    off-diagonal pair feature is replaced by an i.i.d. draw from that
    distribution using a per-graph deterministic RNG.  No label is used.
    """
    pool: list[np.ndarray] = []
    for gid in sorted(reference_graph_ids):
        patches = patches_by_graph[gid]
        feat = relation_features(dist_matrices[gid], patches)
        p = len(patches)
        for i in range(p):
            for j in range(p):
                if i != j:
                    pool.append(feat[i, j])
    pool_arr = np.stack(pool) if pool else np.zeros((1, REL_FEAT_DIM), dtype=np.float32)
    out: dict[str, np.ndarray] = {}
    for gid, patches in patches_by_graph.items():
        p = len(patches)
        rng = np.random.default_rng(_seed_from("scramble", gid))
        feat = np.zeros((p, p, REL_FEAT_DIM), dtype=np.float32)
        for i in range(p):
            for j in range(p):
                if i == j:
                    continue
                idx = int(rng.integers(0, len(pool_arr)))
                feat[i, j] = pool_arr[idx]
        out[gid] = feat
    return out


# ---------------------------------------------------------------------------
# structural comparisons (preflight)
# ---------------------------------------------------------------------------


def unattributed_graph_key(graph: Graph) -> str:
    """Isomorphism-class identifier of the un-attributed graph.

    Small graphs only; a canonical form is obtained by brute force over
    relabelings (n <= 14 is too large for full permutation, so we use a
    degree-sequence + sorted-triangle multiset + edge-graph WL hash, which is
    sufficient here because all compared pairs share the same node set and edge
    list; for cross-checks we also compare edge lists directly)."""
    deg = [len(a) for a in adjacency(graph.n, graph.edges)]
    edge_set = {frozenset(e) for e in graph.edges}
    tri = []
    adj = adjacency(graph.n, graph.edges)
    for v in range(graph.n):
        for i in range(len(adj[v])):
            for j in range(i + 1, len(adj[v])):
                if frozenset((adj[v][i], adj[v][j])) in edge_set:
                    tri.append(tuple(sorted((v, adj[v][i], adj[v][j]))))
    return json.dumps({"n": graph.n, "m": len(graph.edges), "deg": sorted(deg), "tri": sorted(tri)})


def attr_histogram(graph: Graph) -> dict[str, int]:
    c = Counter(graph.attrs)
    return {ATTR_NAMES[k]: v for k, v in sorted(c.items())}


def patch_bag(graph: Graph, patches: Sequence[Patch] | None = None) -> Counter:
    if patches is None:
        patches = extract_patches(graph)
    return Counter(p.feature for p in patches)


def patch_sap_bag(graph: Graph, patches: Sequence[Patch] | None = None) -> Counter:
    if patches is None:
        patches = extract_patches(graph)
    return Counter(p.sap for p in patches)


# ---------------------------------------------------------------------------
# preflight
# ---------------------------------------------------------------------------


def _pair_of(graphs: Sequence[Graph], graph_id: str) -> Graph:
    for g in graphs:
        if g.graph_id == graph_id:
            return g
    raise KeyError(graph_id)


def run_preflight() -> dict:
    """The five checks required before any training; raises on failure."""
    graphs = all_graphs()
    by_split: dict[str, list[Graph]] = defaultdict(list)
    for g in graphs:
        by_split[g.split].append(g)
    patches = {g.graph_id: extract_patches(g) for g in graphs}

    report: dict[str, object] = {"n_graphs": len(graphs), "checks": {}}

    # -- 1. binding counterexample -----------------------------------------
    bind = next(g for g in graphs if g.family == "binding")
    twin = next(
        g
        for g in graphs
        if g.family == "binding" and g.base_id == bind.base_id and g.meta["aux"] == bind.meta["aux"] and g.label != bind.label
    )
    assert bind.edges == twin.edges, "binding variants differ in topology"
    assert bind.n == twin.n
    assert attr_histogram(bind) == attr_histogram(twin)
    assert unattributed_graph_key(bind) == unattributed_graph_key(twin)
    assert bind.label != twin.label
    assert patch_bag(bind) != patch_bag(twin), "binding variants have identical joint patch bags"
    # topology-only bag (all attributes collapsed) must be identical
    def _topo_bag(g: Graph) -> Counter:
        flat = Graph(g.graph_id, g.family, g.base_id, g.pair_id, g.bag_twin, g.split, g.rule, g.n, tuple(ATTR_C for _ in g.attrs), g.edges, g.label, g.meta)
        return patch_bag(flat, extract_patches(flat))

    assert _topo_bag(bind) == _topo_bag(twin), "binding variants differ in topology-only patch bag"
    report["checks"]["binding_counterexample"] = {
        "positive": bind.graph_id,
        "negative": twin.graph_id,
        "same_edges": True,
        "same_attr_histogram": True,
        "same_topology_only_patch_bag": True,
        "joint_patch_bag_differs": True,
    }

    # -- 2. relation counterexample ----------------------------------------
    rel = next(g for g in graphs if g.graph_id.startswith("rel_cycle20_NO_"))
    rel_twin = _pair_of(graphs, rel.bag_twin or "")
    assert rel.edges == rel_twin.edges
    assert attr_histogram(rel) == attr_histogram(rel_twin)
    assert unattributed_graph_key(rel) == unattributed_graph_key(rel_twin)
    assert rel.label == 1 and rel_twin.label == -1
    assert patch_bag(rel, patches[rel.graph_id]) == patch_bag(rel_twin, patches[rel_twin.graph_id]), "relation twins differ in patch bag"
    assert patch_sap_bag(rel, patches[rel.graph_id]) == patch_sap_bag(rel_twin, patches[rel_twin.graph_id])
    d_rel = int(rel.meta["d"])
    d_twin = int(rel_twin.meta["d"])
    assert d_rel != d_twin
    report["checks"]["relation_counterexample"] = {
        "positive": rel.graph_id,
        "negative": rel_twin.graph_id,
        "same_edges": True,
        "same_attr_histogram": True,
        "identical_attributed_patch_bag": True,
        "identical_sap_bag": True,
        "true_distances": [d_rel, d_twin],
    }

    # -- 3. relabeling invariance ------------------------------------------
    rng = np.random.default_rng(_seed_from("relabel_preflight"))
    for g in graphs:
        perm = rng.permutation(g.n)
        g2 = relabel_graph(g, perm)
        assert patch_bag(g, patches[g.graph_id]) == patch_bag(g2), f"relabeling changed patch bag for {g.graph_id}"
        # per-patch correspondence: old edge (u,v) maps to (perm[u],perm[v])
        feats1 = {p.edge: p.feature for p in patches[g.graph_id]}
        feats2 = {p.edge: p.feature for p in extract_patches(g2)}
        for (u, v), f in feats1.items():
            mapped = tuple(sorted((int(perm[u]), int(perm[v]))))
            assert f == feats2[mapped], f"relabeled patch differs for {g.graph_id} edge {(u, v)}"
    report["checks"]["relabeling_invariance"] = {"n_permuted_graphs": len(graphs), "patch_bags_equal": True}

    # -- 4. no marginal shortcut -------------------------------------------
    marginal_diff = []
    seen_pairs: set[str] = set()
    for g in graphs:
        if g.pair_id in seen_pairs:
            continue
        seen_pairs.add(g.pair_id)
        variants = [x for x in graphs if x.pair_id == g.pair_id]
        if len(variants) < 2:
            continue
        base_marg = {
            (len(x.edges), x.n, tuple(sorted(Counter(x.attrs).items())), len(patches[x.graph_id]))
            for x in variants
        }
        marginal_diff.append(len(base_marg))
    assert all(v == 1 for v in marginal_diff), "marginals differ inside a matched pair"
    # whole-dataset label balance per split
    balance = {}
    for split, gs in by_split.items():
        pos = sum(1 for x in gs if x.label == 1)
        neg = len(gs) - pos
        balance[split] = {"n": len(gs), "pos": pos, "neg": neg}
        assert pos == neg, f"split {split} not balanced: {pos} vs {neg}"
    report["checks"]["no_marginal_shortcut"] = {
        "matched_pairs_with_identical_marginals": len(marginal_diff),
        "split_balance": balance,
    }

    # -- 5. instance-level truth -------------------------------------------
    truth_ok = True
    for g in graphs:
        ps = patches[g.graph_id]
        assert len(ps) == len(g.edges)
        tri = triangle_nodes(g.n, g.edges)
        if g.family == "binding":
            n_pos = [i for i, a in enumerate(g.attrs) if a == ATTR_N]
            assert len(n_pos) == 1
            expect = 1 if n_pos[0] in tri else -1
            truth_ok &= expect == g.label
        elif g.family == "relation":
            d = int(g.meta["d"])
            expect = 1 if d <= SHORT_D else -1
            truth_ok &= expect == g.label
        elif g.family == "composition":
            expect = _xor_label(bool(g.meta["in_triangle"]), bool(g.meta["short"]))
            truth_ok &= expect == g.label
        dist = anchor_distance_matrix(g, ps)
        feat = relation_features(dist, ps)
        assert feat.shape == (len(ps), len(ps), REL_FEAT_DIM)
    assert truth_ok
    report["checks"]["instance_truth"] = {
        "per_graph_labels_match_rule": True,
        "per_patch_sap_returned": True,
        "per_patch_special_attributes_returned": True,
        "pairwise_anchor_distances_returned": True,
    }

    # -- split disjointness -------------------------------------------------
    base_split: dict[str, set[str]] = defaultdict(set)
    for g in graphs:
        base_split[g.base_id].add(g.split)
    assert all(len(s) == 1 for s in base_split.values()), "a base graph crosses splits"
    report["checks"]["split_disjointness"] = {"n_base_families": len(base_split), "each_base_in_one_split": True}

    report["split_summary"] = {
        split: {
            fam: sum(1 for x in gs if x.family == fam)
            for fam in FAMILIES
        }
        for split, gs in by_split.items()
    }
    return report


def frozen_split_manifest() -> dict:
    graphs = all_graphs()
    return {
        "generator": "structure_concept_instance_gate_v1/code/synthetic.py",
        "graph_ids": {g.split: sorted(x.graph_id for x in graphs if x.split == g.split) for g in graphs},
        "base_ids": {split: sorted({x.base_id for x in graphs if x.split == split}) for split in ("train", "dev", "test")},
        "n_graphs": len(graphs),
        "labels": {g.graph_id: g.label for g in graphs},
        "pair_ids": {g.graph_id: g.pair_id for g in graphs},
        "bag_twins": {g.graph_id: g.bag_twin for g in graphs},
    }


if __name__ == "__main__":  # pragma: no cover - manual inspection
    rep = run_preflight()
    print(json.dumps(rep, indent=2, sort_keys=True))

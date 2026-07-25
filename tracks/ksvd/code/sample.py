from __future__ import annotations

import random
from dataclasses import dataclass, field

from .graph import Graph


@dataclass
class SampleConfig:
    p: float = 1.0
    q: float = 1.0
    walk_length: int = 8
    max_nodes: int = 12
    num_walks: int = 30
    edge_decay: float = 0.7
    no_backtrack: bool = True
    seed: int = 0


@dataclass
class SampleBundle:
    """One full sampling run on a graph: list of induced subgraphs + edge hit counts."""

    method: str
    subgraphs: list[Graph] = field(default_factory=list)
    node_sets: list[set[int]] = field(default_factory=list)
    edge_counts: dict[tuple[int, int], int] = field(default_factory=dict)
    trajectory_edges: list[list[tuple[int, int]]] = field(default_factory=list)


def _alpha(p: float, q: float, t: int, x: int, g: Graph) -> float:
    if x == t:
        return 1.0 / p
    if g.has_edge(t, x):
        return 1.0
    return 1.0 / q


def _weighted_choice(
    rng: random.Random,
    candidates: list[int],
    weights: list[float],
) -> int:
    total = sum(weights)
    if total <= 0 or not candidates:
        raise ValueError("empty weighted choice")
    r = rng.random() * total
    acc = 0.0
    for c, w in zip(candidates, weights):
        acc += w
        if r <= acc:
            return c
    return candidates[-1]


def node2vec_walk(
    g: Graph,
    start: int,
    cfg: SampleConfig,
    rng: random.Random,
    edge_weight: dict[tuple[int, int], float],
) -> tuple[list[int], list[tuple[int, int]]]:
    """Return node sequence and undirected edges traversed (as sorted pairs)."""
    walk = [start]
    traj_edges: list[tuple[int, int]] = []
    if cfg.walk_length <= 0:
        return walk, traj_edges

    # first step: uniform among neighbors, weighted by edge_weight
    nbrs = list(g.neighbors(start))
    if not nbrs:
        return walk, traj_edges
    w0 = [edge_weight.get(g.edge_key(start, x), 1.0) for x in nbrs]
    cur = _weighted_choice(rng, nbrs, w0)
    traj_edges.append(g.edge_key(start, cur))
    walk.append(cur)
    prev = start

    for _ in range(1, cfg.walk_length):
        nbrs = list(g.neighbors(cur))
        if cfg.no_backtrack and len(nbrs) > 1:
            nbrs = [x for x in nbrs if x != prev] or nbrs
        if not nbrs:
            break
        weights = []
        for x in nbrs:
            a = _alpha(cfg.p, cfg.q, prev, x, g)
            ew = edge_weight.get(g.edge_key(cur, x), 1.0)
            weights.append(a * ew)
        nxt = _weighted_choice(rng, nbrs, weights)
        traj_edges.append(g.edge_key(cur, nxt))
        walk.append(nxt)
        prev, cur = cur, nxt
        if len(set(walk)) >= cfg.max_nodes:
            break

    return walk, traj_edges


def _cap_nodes(nodes: list[int], max_nodes: int) -> set[int]:
    """Preserve walk order uniqueness then cap."""
    seen: list[int] = []
    for u in nodes:
        if u not in seen:
            seen.append(u)
        if len(seen) >= max_nodes:
            break
    return set(seen)


def sample_B0(g: Graph, cfg: SampleConfig) -> SampleBundle:
    """One-hop star per node: S = {v} ∪ N(v), induced (capped)."""
    bundle = SampleBundle(method="B0")
    for v in g.nodes:
        S = {v} | set(g.neighbors(v))
        if len(S) > cfg.max_nodes:
            # keep v + highest-degree neighbors
            others = sorted(S - {v}, key=lambda x: (-len(g.neighbors(x)), x))
            S = {v} | set(others[: cfg.max_nodes - 1])
        sub = g.induced(S)
        bundle.node_sets.append(S)
        bundle.subgraphs.append(sub)
        for e in sub.edges():
            bundle.edge_counts[e] = bundle.edge_counts.get(e, 0) + 1
    return bundle


def sample_rw(
    g: Graph,
    cfg: SampleConfig,
    method: str,
    *,
    force_pq: tuple[float, float] | None = None,
    force_decay: float | None = None,
) -> SampleBundle:
    rng = random.Random(cfg.seed)
    p, q = force_pq if force_pq is not None else (cfg.p, cfg.q)
    decay = force_decay if force_decay is not None else cfg.edge_decay
    local = SampleConfig(
        p=p,
        q=q,
        walk_length=cfg.walk_length,
        max_nodes=cfg.max_nodes,
        num_walks=cfg.num_walks,
        edge_decay=decay,
        no_backtrack=cfg.no_backtrack,
        seed=cfg.seed,
    )
    edge_weight = {e: 1.0 for e in g.edges()}
    bundle = SampleBundle(method=method)
    nodes = g.nodes
    if not nodes:
        return bundle

    for i in range(local.num_walks):
        start = nodes[i % len(nodes)] if local.num_walks >= len(nodes) else rng.choice(nodes)
        # diversify starts
        start = nodes[rng.randrange(len(nodes))]
        walk, traj_e = node2vec_walk(g, start, local, rng, edge_weight)
        S = _cap_nodes(walk, local.max_nodes)
        sub = g.induced(S)
        bundle.node_sets.append(S)
        bundle.subgraphs.append(sub)
        bundle.trajectory_edges.append(traj_e)
        # decay on trajectory (sampling dynamics); metrics count induced edges (KSVD input)
        for e in traj_e:
            if e in edge_weight:
                edge_weight[e] *= decay
        for e in sub.edges():
            bundle.edge_counts[e] = bundle.edge_counts.get(e, 0) + 1
    return bundle


def run_method(g: Graph, method: str, cfg: SampleConfig) -> SampleBundle:
    m = method.upper()
    if m == "B0":
        return sample_B0(g, cfg)
    if m == "B1":
        return sample_rw(g, cfg, "B1", force_pq=(1.0, 1.0), force_decay=1.0)
    if m == "M0":
        return sample_rw(g, cfg, "M0")
    raise ValueError(f"unknown method {method}")

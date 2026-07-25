"""
Coverage-driven RW sampling (luyin10 R4/R5/R6).

Rules:
  - Soft edge downweight after visit (NOT permanent deletion)
  - Seed starts: sparse (degree-stratified / uncovered), NOT necessarily every node
  - Optional stop when edge_cover >= target or budget exhausted
  - Process metrics: cover, repeat, |S|
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Any, Literal

from .graph import Graph
from .sample import SampleBundle, SampleConfig, _cap_nodes, node2vec_walk

SeedPolicy = Literal["random", "all_nodes", "degree_stratified", "uncovered"]


@dataclass
class CoverageConfig:
    p: float = 1.0
    q: float = 1.0
    walk_length: int = 8
    max_nodes: int = 8
    max_walks: int = 20
    edge_decay: float = 0.7
    no_backtrack: bool = True
    seed: int = 0
    seed_policy: SeedPolicy = "degree_stratified"
    cover_target: float | None = 0.95
    hard_delete: bool = False  # must stay False (R6)
    # ring bias: init edge weight = 1 + ring_boost * (#common neighbors)
    ring_boost: float = 0.0


def _pick_seeds(g: Graph, cfg: CoverageConfig, rng: random.Random) -> list[int]:
    nodes = g.nodes
    if not nodes:
        return []
    if cfg.seed_policy == "all_nodes":
        return list(nodes)
    if cfg.seed_policy == "random":
        return [rng.choice(nodes) for _ in range(cfg.max_walks)]
    if cfg.seed_policy == "degree_stratified":
        degs = sorted(nodes, key=lambda u: (len(g.neighbors(u)), u))
        n = len(degs)
        buckets = [
            degs[: max(1, n // 3)],
            degs[n // 3 : 2 * n // 3] or degs,
            degs[2 * n // 3 :] or degs,
        ]
        seeds: list[int] = []
        i = 0
        while len(seeds) < cfg.max_walks:
            b = buckets[i % 3]
            seeds.append(rng.choice(b))
            i += 1
        return seeds
    return [rng.choice(nodes) for _ in range(cfg.max_walks)]


def _uncovered_start(
    g: Graph,
    edge_weight: dict[tuple[int, int], float],
    rng: random.Random,
) -> int:
    scored = []
    for u in g.nodes:
        ws = [edge_weight.get(g.edge_key(u, v), 1.0) for v in g.neighbors(u)]
        scored.append((sum(ws) / max(len(ws), 1), u))
    scored.sort(key=lambda t: (-t[0], t[1]))
    top = scored[: max(1, len(scored) // 4)]
    return rng.choice([u for _, u in top])


def sample_coverage(g: Graph, cfg: CoverageConfig) -> SampleBundle:
    if cfg.hard_delete:
        raise ValueError("hard_delete must be False (luyin10 R6)")
    rng = random.Random(cfg.seed)
    sc = SampleConfig(
        p=cfg.p,
        q=cfg.q,
        walk_length=cfg.walk_length,
        max_nodes=cfg.max_nodes,
        num_walks=1,
        edge_decay=cfg.edge_decay,
        no_backtrack=cfg.no_backtrack,
        seed=cfg.seed,
    )
    if cfg.ring_boost and cfg.ring_boost > 0:
        edge_weight = {}
        for e in g.edges():
            u, v = e
            common = len(g.neighbors(u) & g.neighbors(v))
            edge_weight[e] = 1.0 + float(cfg.ring_boost) * float(common)
    else:
        edge_weight = {e: 1.0 for e in g.edges()}
    all_edges = g.edges()
    m_e = max(len(all_edges), 1)
    traj_hit: set[tuple[int, int]] = set()

    bundle = SampleBundle(method=f"COV_{cfg.seed_policy}")
    seeds = _pick_seeds(g, cfg, rng)

    for i in range(cfg.max_walks):
        if cfg.seed_policy == "uncovered":
            start = _uncovered_start(g, edge_weight, rng)
        else:
            start = seeds[i % len(seeds)] if seeds else rng.choice(g.nodes)

        walk, traj_e = node2vec_walk(g, start, sc, rng, edge_weight)
        S = _cap_nodes(walk, cfg.max_nodes)
        sub = g.induced(S)
        bundle.node_sets.append(S)
        bundle.subgraphs.append(sub)
        bundle.trajectory_edges.append(traj_e)

        for e in traj_e:
            traj_hit.add(e)
            if e in edge_weight:
                edge_weight[e] *= cfg.edge_decay
        for e in sub.edges():
            bundle.edge_counts[e] = bundle.edge_counts.get(e, 0) + 1

        cover = len(traj_hit) / m_e
        if cfg.cover_target is not None and cover >= cfg.cover_target:
            break

    return bundle


def trajectory_cover_metrics(g: Graph, bundle: SampleBundle) -> dict[str, Any]:
    all_e = g.edges()
    m_e = max(len(all_e), 1)
    counts: dict[tuple[int, int], int] = {}
    for traj in bundle.trajectory_edges:
        for e in traj:
            counts[e] = counts.get(e, 0) + 1
    hit = sum(1 for e in all_e if counts.get(e, 0) > 0)
    reps = [counts.get(e, 0) for e in all_e]
    repeat_mass = sum(max(c - 1, 0) for c in reps)
    sizes = [len(s) for s in bundle.node_sets]
    return {
        "n_walks": len(bundle.trajectory_edges),
        "traj_edge_cover": hit / m_e,
        "traj_edge_repeat": repeat_mass / m_e,
        "traj_mean_count": sum(reps) / m_e,
        "mean_|S|": sum(sizes) / max(len(sizes), 1) if sizes else 0.0,
        "max_|S|": max(sizes) if sizes else 0,
        "n_patches": len(bundle.node_sets),
    }

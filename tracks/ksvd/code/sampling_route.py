"""Matched-budget patch sampling and KSVD relation probes.

This module is intentionally separate from the historical MolHIV pipeline.  It
implements the mechanism tests requested after luyin10/luyin11:

* same patch-count / max-size budget across samplers;
* explicit process metrics (coverage, overlap, repetition, radius);
* shared train-fold KSVD;
* bag-of-patches readout versus patch-relation readout;
* shuffled code-to-patch assignment as a negative control.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import random
from typing import Any

import numpy as np

from .graph import Graph
from .ksvd import _omp
from .vectorize import canonical_adjacency_features, labeled_wl_patch_features, wl_patch_features


@dataclass(frozen=True)
class MatchedSamplingConfig:
    n_patches: int = 8
    max_nodes: int = 8
    walk_length: int = 16
    edge_decay: float = 0.7
    ppr_alpha: float = 0.85
    ppr_steps: int = 60
    seed: int = 0


@dataclass
class PatchCollection:
    method: str
    node_sets: list[set[int]]
    centers: list[int]
    trajectories: list[list[tuple[int, int]]]


def choose_centers(g: Graph, cfg: MatchedSamplingConfig) -> list[int]:
    """Uniform seed schedule shared by all methods for a graph/config."""
    if not g.nodes:
        return []
    rng = np.random.default_rng(cfg.seed)
    replace = g.n < cfg.n_patches
    return [int(x) for x in rng.choice(g.nodes, size=cfg.n_patches, replace=replace)]


def _edge(g: Graph, u: int, v: int) -> tuple[int, int]:
    return g.edge_key(u, v)


def _cap_candidates(
    g: Graph,
    center: int,
    candidates: list[tuple[int, int]],
    max_nodes: int,
    rng: random.Random,
) -> set[int]:
    """Cap nodes by distance, then degree, with randomized ties."""
    tie = {u: rng.random() for _, u in candidates}
    ordered = sorted(candidates, key=lambda z: (z[0], -len(g.neighbors(z[1])), tie[z[1]]))
    kept = [u for _, u in ordered[:max_nodes]]
    if center not in kept:
        kept[-1] = center
    return set(kept)


def _bfs_distances(g: Graph, center: int, max_depth: int | None = None) -> dict[int, int]:
    dist = {center: 0}
    q = deque([center])
    while q:
        u = q.popleft()
        if max_depth is not None and dist[u] >= max_depth:
            continue
        for v in g.neighbors(u):
            if v not in dist:
                dist[v] = dist[u] + 1
                q.append(v)
    return dist


def _sample_b0(g: Graph, centers: list[int], cfg: MatchedSamplingConfig) -> PatchCollection:
    rng = random.Random(cfg.seed + 101)
    sets = []
    for c in centers:
        cand = [(0, c)] + [(1, v) for v in g.neighbors(c)]
        sets.append(_cap_candidates(g, c, cand, cfg.max_nodes, rng))
    return PatchCollection("B0", sets, centers, [[] for _ in sets])


def _sample_r2(g: Graph, centers: list[int], cfg: MatchedSamplingConfig) -> PatchCollection:
    rng = random.Random(cfg.seed + 211)
    sets = []
    for c in centers:
        d = _bfs_distances(g, c, max_depth=2)
        sets.append(_cap_candidates(g, c, [(r, u) for u, r in d.items()], cfg.max_nodes, rng))
    return PatchCollection("R2", sets, centers, [[] for _ in sets])


def _walk_one(
    g: Graph,
    center: int,
    cfg: MatchedSamplingConfig,
    rng: random.Random,
    edge_weight: dict[tuple[int, int], float],
    self_avoiding: bool,
) -> tuple[set[int], list[tuple[int, int]]]:
    visited = [center]
    cur = center
    prev: int | None = None
    traj: list[tuple[int, int]] = []
    for _ in range(cfg.walk_length):
        nbrs = list(g.neighbors(cur))
        if self_avoiding:
            fresh = [v for v in nbrs if v not in visited]
            if fresh:
                nbrs = fresh
            else:
                break
        elif prev is not None and len(nbrs) > 1:
            nbrs = [v for v in nbrs if v != prev] or nbrs
        if not nbrs:
            break
        weights = [max(edge_weight.get(_edge(g, cur, v), 1.0), 1e-12) for v in nbrs]
        total = sum(weights)
        z = rng.random() * total
        acc = 0.0
        nxt = nbrs[-1]
        for v, w in zip(nbrs, weights):
            acc += w
            if z <= acc:
                nxt = v
                break
        traj.append(_edge(g, cur, nxt))
        prev, cur = cur, nxt
        if nxt not in visited:
            visited.append(nxt)
        if len(visited) >= cfg.max_nodes:
            break
    return set(visited), traj


def _sample_walk(
    g: Graph,
    centers: list[int],
    cfg: MatchedSamplingConfig,
    coverage: bool,
    self_avoiding: bool = False,
) -> PatchCollection:
    offset = 307 if coverage else (401 if self_avoiding else 503)
    rng = random.Random(cfg.seed + offset)
    weights = {e: 1.0 for e in g.edges()}
    sets, trajectories = [], []
    for c in centers:
        S, traj = _walk_one(g, c, cfg, rng, weights, self_avoiding=self_avoiding)
        sets.append(S)
        trajectories.append(traj)
        if coverage:
            for e in traj:
                weights[e] *= cfg.edge_decay
    name = "CoverageRW" if coverage else ("Path" if self_avoiding else "UniformRW")
    return PatchCollection(name, sets, centers, trajectories)


def _ppr_scores(g: Graph, center: int, alpha: float, steps: int) -> dict[int, float]:
    nodes = g.nodes
    if not nodes:
        return {}
    idx = {u: i for i, u in enumerate(nodes)}
    p = np.zeros(len(nodes), dtype=np.float64)
    p[idx[center]] = 1.0
    restart = p.copy()
    for _ in range(steps):
        nxt = np.zeros_like(p)
        for u in nodes:
            ui = idx[u]
            nbrs = g.neighbors(u)
            if nbrs:
                share = p[ui] / len(nbrs)
                for v in nbrs:
                    nxt[idx[v]] += share
            else:
                nxt[ui] += p[ui]
        p = alpha * nxt + (1.0 - alpha) * restart
    return {u: float(p[idx[u]]) for u in nodes}


def _sample_ppr(g: Graph, centers: list[int], cfg: MatchedSamplingConfig) -> PatchCollection:
    rng = random.Random(cfg.seed + 601)
    sets = []
    for c in centers:
        score = _ppr_scores(g, c, cfg.ppr_alpha, cfg.ppr_steps)
        tie = {u: rng.random() for u in g.nodes}
        order = sorted(g.nodes, key=lambda u: (-score.get(u, 0.0), -len(g.neighbors(u)), tie[u]))
        kept = order[: min(cfg.max_nodes, len(order))]
        # Keep the seed semantics explicit even in unusual high-degree graphs
        # where a finite-step approximation could rank the center below the cap.
        if c not in kept:
            kept[-1] = c
        sets.append(set(kept))
    return PatchCollection("PPR", sets, centers, [[] for _ in sets])


def sample_matched(g: Graph, method: str, cfg: MatchedSamplingConfig) -> PatchCollection:
    centers = choose_centers(g, cfg)
    if method == "B0":
        return _sample_b0(g, centers, cfg)
    if method == "R2":
        return _sample_r2(g, centers, cfg)
    if method == "UniformRW":
        return _sample_walk(g, centers, cfg, coverage=False)
    if method == "CoverageRW":
        return _sample_walk(g, centers, cfg, coverage=True)
    if method == "Path":
        return _sample_walk(g, centers, cfg, coverage=False, self_avoiding=True)
    if method == "PPR":
        return _sample_ppr(g, centers, cfg)
    raise ValueError(f"unknown sampler {method!r}")


def collection_metrics(g: Graph, pc: PatchCollection) -> dict[str, float]:
    edge_counts = {e: 0 for e in g.edges()}
    node_union: set[int] = set()
    sizes: list[int] = []
    radii: list[int] = []
    for c, S in zip(pc.centers, pc.node_sets):
        node_union |= S
        sizes.append(len(S))
        d = _bfs_distances(g, c)
        radii.append(max((d.get(u, 0) for u in S), default=0))
        for u, v in g.edges():
            if u in S and v in S:
                edge_counts[(u, v)] += 1
    E = max(g.num_edges(), 1)
    induced_cover = sum(v > 0 for v in edge_counts.values()) / E
    induced_repeat = sum(max(v - 1, 0) for v in edge_counts.values()) / E

    traj_counts = {e: 0 for e in g.edges()}
    for traj in pc.trajectories:
        for e in traj:
            if e in traj_counts:
                traj_counts[e] += 1
    traj_cover = sum(v > 0 for v in traj_counts.values()) / E
    traj_repeat = sum(max(v - 1, 0) for v in traj_counts.values()) / E

    jac: list[float] = []
    for i in range(len(pc.node_sets)):
        for j in range(i + 1, len(pc.node_sets)):
            a, b = pc.node_sets[i], pc.node_sets[j]
            jac.append(len(a & b) / max(len(a | b), 1))
    return {
        "n_patches": float(len(pc.node_sets)),
        "mean_patch_nodes": float(np.mean(sizes)) if sizes else 0.0,
        "std_patch_nodes": float(np.std(sizes)) if sizes else 0.0,
        "mean_radius": float(np.mean(radii)) if radii else 0.0,
        "node_cover": len(node_union) / max(g.n, 1),
        "induced_edge_cover": float(induced_cover),
        "induced_edge_repeat": float(induced_repeat),
        "trajectory_edge_cover": float(traj_cover),
        "trajectory_edge_repeat": float(traj_repeat),
        "mean_pair_jaccard": float(np.mean(jac)) if jac else 0.0,
    }


def vectorize_collection(
    g: Graph,
    pc: PatchCollection,
    max_nodes: int,
    feature_mode: str,
    node_feat: np.ndarray | None = None,
) -> np.ndarray:
    cols: list[np.ndarray] = []
    for S in pc.node_sets:
        if feature_mode == "wl":
            y = np.asarray(wl_patch_features(g, S, max_nodes=max_nodes), dtype=np.float64)
        elif feature_mode == "canonical":
            y = np.asarray(
                canonical_adjacency_features(g, S, max_nodes=max_nodes),
                dtype=np.float64,
            )
        elif feature_mode == "rooted_canonical":
            y = np.asarray(
                canonical_adjacency_features(
                    g, S, max_nodes=max_nodes, root=pc.centers[len(cols)]
                ),
                dtype=np.float64,
            )
        elif feature_mode == "labeled_wl":
            y = np.asarray(
                labeled_wl_patch_features(
                    g,
                    S,
                    max_nodes=max_nodes,
                    node_feat=node_feat,
                    edge_feat=None,
                    atom_bins=32,
                    bond_bins=4,
                    wl_bins=64,
                    n_iter=3,
                ),
                dtype=np.float64,
            )
        else:
            raise ValueError(feature_mode)
        n = float(np.linalg.norm(y))
        cols.append(y / max(n, 1e-12))
    if not cols:
        raise RuntimeError("empty patch collection")
    return np.stack(cols, axis=1)


def relation_channels(g: Graph, pc: PatchCollection) -> dict[str, np.ndarray]:
    p = len(pc.node_sets)
    center = np.zeros((p, p), dtype=np.float64)
    overlap = np.zeros((p, p), dtype=np.float64)
    for i in range(p):
        for j in range(i + 1, p):
            ci, cj = pc.centers[i], pc.centers[j]
            if ci != cj and g.has_edge(ci, cj):
                center[i, j] = center[j, i] = 1.0
            a, b = pc.node_sets[i], pc.node_sets[j]
            ov = len(a & b) / max(len(a | b), 1)
            overlap[i, j] = overlap[j, i] = ov
    return {"center": center, "overlap": overlap}


def sparse_code_matrix(D: np.ndarray, Y: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
    X = np.zeros((D.shape[1], Y.shape[1]), dtype=np.float64)
    for j in range(Y.shape[1]):
        X[:, j] = _omp(D, Y[:, j], min(T, D.shape[1]))
    R = Y - D @ X
    denom = np.maximum(np.linalg.norm(Y, axis=0), 1e-12)
    errors = np.linalg.norm(R, axis=0) / denom
    return X, errors


def content_readout(X: np.ndarray, errors: np.ndarray) -> np.ndarray:
    A = np.abs(X)
    k, n = A.shape
    mean = A.mean(axis=1)
    maxv = A.max(axis=1)
    std = A.std(axis=1)
    usage = (A > 1e-10).mean(axis=1)
    q75 = np.quantile(A, 0.75, axis=1)
    energy = (X * X).mean(axis=1)
    signed = X.mean(axis=1)
    winner = np.bincount(np.argmax(A, axis=0), minlength=k).astype(np.float64) / max(n, 1)
    recon = np.asarray([errors.mean(), errors.std(), np.quantile(errors, 0.9)], dtype=np.float64)
    return np.concatenate([mean, maxv, std, usage, q75, energy, signed, winner, recon])


def _upper_triangle(M: np.ndarray) -> np.ndarray:
    idx = np.triu_indices(M.shape[0])
    return M[idx]


def relation_readout(
    X: np.ndarray,
    channels: dict[str, np.ndarray],
    assignment_permutation: np.ndarray | None = None,
    include_topology: bool = True,
) -> np.ndarray:
    """Atom-pair co-occurrence over true patch relations.

    assignment_permutation shuffles sparse codes across fixed patch positions.
    It preserves the content multiset and relation graph, destroying only their
    correct alignment; this is the negative control.
    """
    A = np.abs(X).T  # patches x atoms
    if assignment_permutation is not None:
        A = A[np.asarray(assignment_permutation, dtype=np.int64)]
    out: list[np.ndarray] = []
    for name in ["center", "overlap"]:
        W = np.asarray(channels[name], dtype=np.float64)
        mass = float(W.sum())
        if mass <= 1e-12:
            C = np.zeros((A.shape[1], A.shape[1]), dtype=np.float64)
            contrast = np.zeros(A.shape[1], dtype=np.float64)
            topo = np.zeros(3, dtype=np.float64)
        else:
            C = (A.T @ W @ A) / mass
            diff = np.abs(A[:, None, :] - A[None, :, :])
            contrast = (diff * W[:, :, None]).sum(axis=(0, 1)) / mass
            deg = W.sum(axis=1)
            topo = np.asarray([mass / 2.0, deg.mean(), deg.std()], dtype=np.float64)
        out.extend([_upper_triangle(C), contrast])
        if include_topology:
            out.append(topo)
    return np.concatenate(out)


def relation_graph_readout(channels: dict[str, np.ndarray]) -> np.ndarray:
    """Relation-graph statistics that do not use patch content.

    Keeping this separate is essential for the shuffle control: a fixed relation
    graph may itself classify a graph even when patch contents are permuted.
    """
    out: list[float] = []
    for name in ["center", "overlap"]:
        W = np.asarray(channels[name], dtype=np.float64)
        mass = float(W.sum())
        if mass <= 1e-12:
            out.extend([0.0, 0.0, 0.0])
        else:
            deg = W.sum(axis=1)
            out.extend([mass / 2.0, float(deg.mean()), float(deg.std())])
    return np.asarray(out, dtype=np.float64)


def raw_mean_embedding(Y: np.ndarray) -> np.ndarray:
    z = Y.mean(axis=1)
    return z / max(float(np.linalg.norm(z)), 1e-12)


def aggregate_process(rows: list[dict[str, float]]) -> dict[str, float]:
    if not rows:
        return {}
    return {k: float(np.mean([r[k] for r in rows])) for k in rows[0]}

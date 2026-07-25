"""
Export full graph-level pipeline traces for interactive HTML viz.

Usage:
  python -m code.viz_pipeline_data
  → results/viz_pipeline/data.json
"""

from __future__ import annotations

import json
import random
import sys
from pathlib import Path
from typing import Any

import numpy as np

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.coverage_sample import (  # noqa: E402
    CoverageConfig,
    _pick_seeds,
    _uncovered_start,
)
from code.graph import Graph, from_edges  # noqa: E402
from code.graph_level import GraphLevelConfig, bundle_to_Y  # noqa: E402
from code.ksvd import _omp, ksvd, readout_X  # noqa: E402
from code.sample import SampleConfig, _alpha, _cap_nodes, _weighted_choice  # noqa: E402
from code.synthetic_task import _count_c4  # noqa: E402
from code.vectorize import adjacency_padded, flatten_upper  # noqa: E402


def make_demo_c4() -> Graph:
    """C4 on 0-1-2-3 plus a small tree for layout clarity."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (4, 5), (1, 6)]
    return from_edges(7, edges)


def make_demo_c8() -> Graph:
    edges = [(i, (i + 1) % 8) for i in range(8)]
    edges += [(0, 8), (8, 9)]
    return from_edges(10, edges)


def spring_layout(g: Graph, seed: int = 0, iters: int = 100) -> dict[str, list[float]]:
    rng = np.random.default_rng(seed)
    nodes = g.nodes
    pos = {u: rng.normal(size=2) for u in nodes}
    for _ in range(iters):
        disp = {u: np.zeros(2) for u in nodes}
        for i, u in enumerate(nodes):
            for v in nodes[i + 1 :]:
                d = pos[u] - pos[v]
                dist = float(np.linalg.norm(d) + 1e-3)
                f = (d / dist) * (0.6 / dist)
                disp[u] += f
                disp[v] -= f
        for u, v in g.edges():
            d = pos[v] - pos[u]
            dist = float(np.linalg.norm(d) + 1e-3)
            f = d * 0.08 * dist
            disp[u] += f
            disp[v] -= f
        for u in nodes:
            pos[u] = pos[u] + 0.12 * disp[u]
    xs = np.array([pos[u][0] for u in nodes])
    ys = np.array([pos[u][1] for u in nodes])
    xs = (xs - xs.min()) / (np.ptp(xs) + 1e-9)
    ys = (ys - ys.min()) / (np.ptp(ys) + 1e-9)
    # scale to pixel-ish coords
    return {str(u): [float(xs[i] * 420 + 40), float(ys[i] * 320 + 40)] for i, u in enumerate(nodes)}


def traced_walk(
    g: Graph,
    start: int,
    cfg: SampleConfig,
    rng: random.Random,
    edge_weight: dict[tuple[int, int], float],
) -> dict[str, Any]:
    """Walk with per-step log for viz."""
    walk = [start]
    traj_edges: list[tuple[int, int]] = []
    steps: list[dict[str, Any]] = []
    if cfg.walk_length <= 0:
        return {"walk": walk, "traj_edges": traj_edges, "steps": steps}

    nbrs = list(g.neighbors(start))
    if not nbrs:
        return {"walk": walk, "traj_edges": traj_edges, "steps": steps}

    # first step
    cand = nbrs
    weights = [edge_weight.get(g.edge_key(start, x), 1.0) for x in cand]
    cur = _weighted_choice(rng, cand, weights)
    e0 = g.edge_key(start, cur)
    steps.append(
        {
            "t": 0,
            "from": start,
            "to": cur,
            "edge": list(e0),
            "candidates": [
                {"node": x, "weight": float(edge_weight.get(g.edge_key(start, x), 1.0)), "alpha": 1.0}
                for x in cand
            ],
            "edge_weight_before": float(edge_weight.get(e0, 1.0)),
        }
    )
    traj_edges.append(e0)
    walk.append(cur)
    prev = start

    for t in range(1, cfg.walk_length):
        nbrs = list(g.neighbors(cur))
        if cfg.no_backtrack and len(nbrs) > 1:
            nbrs = [x for x in nbrs if x != prev] or nbrs
        if not nbrs:
            break
        cands = []
        weights = []
        for x in nbrs:
            a = _alpha(cfg.p, cfg.q, prev, x, g)
            ew = edge_weight.get(g.edge_key(cur, x), 1.0)
            cands.append({"node": x, "weight": float(ew), "alpha": float(a), "score": float(a * ew)})
            weights.append(a * ew)
        nxt = _weighted_choice(rng, [c["node"] for c in cands], weights)
        e = g.edge_key(cur, nxt)
        steps.append(
            {
                "t": t,
                "from": cur,
                "to": nxt,
                "edge": list(e),
                "candidates": cands,
                "edge_weight_before": float(edge_weight.get(e, 1.0)),
            }
        )
        traj_edges.append(e)
        walk.append(nxt)
        prev, cur = cur, nxt
        if len(set(walk)) >= cfg.max_nodes:
            break

    return {"walk": walk, "traj_edges": [list(e) for e in traj_edges], "steps": steps}


def sample_with_trace(g: Graph, cfg: CoverageConfig) -> dict[str, Any]:
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
    edge_weight = {e: 1.0 for e in g.edges()}
    all_e = g.edges()
    m_e = max(len(all_e), 1)
    traj_hit: set[tuple[int, int]] = set()
    seeds_pre = _pick_seeds(g, cfg, rng)
    walks_out = []

    for i in range(cfg.max_walks):
        if cfg.seed_policy == "uncovered":
            start = _uncovered_start(g, edge_weight, rng)
        else:
            start = seeds_pre[i % len(seeds_pre)] if seeds_pre else rng.choice(g.nodes)

        # snapshot weights before walk
        w_snap = {f"{a}-{b}": float(edge_weight[(a, b)]) for a, b in all_e}
        tw = traced_walk(g, start, sc, rng, edge_weight)
        S = _cap_nodes(tw["walk"], cfg.max_nodes)
        # decay
        for e in tw["traj_edges"]:
            ee = (int(e[0]), int(e[1]))
            traj_hit.add(ee)
            if ee in edge_weight:
                edge_weight[ee] *= cfg.edge_decay

        A = adjacency_padded(g, S, cfg.max_nodes, order_mode="bfs")
        y = flatten_upper(A)
        has_c4 = _count_c4(g.induced(S)) > 0
        walks_out.append(
            {
                "walk_id": i,
                "start": start,
                "walk": tw["walk"],
                "steps": tw["steps"],
                "traj_edges": tw["traj_edges"],
                "S": sorted(S),
                "has_c4": has_c4,
                "y_norm": float(np.linalg.norm(y)),
                "weights_before": w_snap,
                "cover_after": len(traj_hit) / m_e,
            }
        )
        if cfg.cover_target is not None and len(traj_hit) / m_e >= cfg.cover_target:
            break

    return {
        "walks": walks_out,
        "final_edge_weights": {f"{a}-{b}": float(edge_weight[(a, b)]) for a, b in all_e},
        "n_walks": len(walks_out),
        "traj_cover": len(traj_hit) / m_e,
    }


def b0_patches(g: Graph, max_nodes: int = 8) -> list[dict[str, Any]]:
    out = []
    for v in g.nodes:
        S = {v} | set(g.neighbors(v))
        if len(S) > max_nodes:
            others = sorted(S - {v}, key=lambda u: (-len(g.neighbors(u)), u))
            S = {v} | set(others[: max_nodes - 1])
        out.append(
            {
                "center": v,
                "S": sorted(S),
                "has_c4": _count_c4(g.induced(S)) > 0,
            }
        )
    return out


def graph_payload(g: Graph, name: str, label: str) -> dict[str, Any]:
    return {
        "name": name,
        "label": label,
        "n": g.n,
        "nodes": g.nodes,
        "edges": [list(e) for e in g.edges()],
        "layout": spring_layout(g, seed=0),
        "has_c4_global": _count_c4(g) > 0,
    }


def run_ksvd_demo(g: Graph, sample: dict, max_nodes: int = 8, n_atoms: int = 8) -> dict[str, Any]:
    cols = []
    for w in sample["walks"]:
        S = set(w["S"])
        y = np.array(flatten_upper(adjacency_padded(g, S, max_nodes)), dtype=np.float64)
        if np.linalg.norm(y) > 1e-12:
            cols.append(y)
            w["y"] = y.tolist()
        else:
            w["y"] = y.tolist()
    if not cols:
        cols = [np.zeros(max_nodes * (max_nodes - 1) // 2)]
    Y = np.stack(cols, axis=1)
    D, X, info = ksvd(Y, n_atoms=min(n_atoms, Y.shape[1], Y.shape[0]), T=3, n_iter=8, seed=0, T_min=2)

    # reshape atoms to m x m for viz
    m = max_nodes
    atoms = []
    for j in range(D.shape[1]):
        mat = np.zeros((m, m))
        k = 0
        for i in range(m):
            for jj in range(i + 1, m):
                mat[i, jj] = D[k, j]
                mat[jj, i] = D[k, j]
                k += 1
        atoms.append(
            {
                "id": j,
                "vec": D[:, j].tolist(),
                "matrix": mat.tolist(),
                "usage": float((np.abs(X[j, :]) > 1e-10).mean()),
                "energy": float((X[j, :] ** 2).sum()),
            }
        )

    patch_coefs = []
    for j in range(X.shape[1]):
        patch_coefs.append(
            {
                "patch_id": j,
                "coef": X[:, j].tolist(),
                "abs_coef": np.abs(X[:, j]).tolist(),
                "recon_err": float(
                    np.linalg.norm(Y[:, j] - D @ X[:, j]) / (np.linalg.norm(Y[:, j]) + 1e-12)
                ),
            }
        )

    emb = readout_X(X, mode="rich")
    energy = (X**2).sum(axis=1)
    usage = (np.abs(X) > 1e-10).mean(axis=1)
    s_G = np.concatenate([emb, energy, usage])

    return {
        "Y_shape": list(Y.shape),
        "D_shape": list(D.shape),
        "X_shape": list(X.shape),
        "recon_rel": info["recon_rel"],
        "atoms": atoms,
        "patch_coefs": patch_coefs,
        "s_G": s_G.tolist(),
        "s_G_parts": {
            "readout_rich_dim": int(emb.shape[0]),
            "energy": energy.tolist(),
            "usage": usage.tolist(),
        },
    }


def main() -> int:
    out_dir = _TRACK / "results" / "viz_pipeline"
    out_dir.mkdir(parents=True, exist_ok=True)

    g4 = make_demo_c4()
    g8 = make_demo_c8()
    cfg = CoverageConfig(
        p=0.5,
        q=2.0,
        walk_length=8,
        max_nodes=6,
        max_walks=8,
        edge_decay=0.65,
        seed=0,
        seed_policy="degree_stratified",
        cover_target=0.99,
    )

    sample4 = sample_with_trace(g4, cfg)
    sample8 = sample_with_trace(g8, cfg)
    ksvd4 = run_ksvd_demo(g4, sample4, max_nodes=6, n_atoms=6)
    # encode c8 with D learned on c4? better learn on both for demo dictionary richness
    # relearn D on c4 only for "train", apply to c8
    cols = []
    for w in sample4["walks"]:
        y = np.array(w["y"], dtype=np.float64)
        if np.linalg.norm(y) > 1e-12:
            cols.append(y)
    Ytr = np.stack(cols, axis=1) if cols else np.eye(6)
    D, _, _ = ksvd(Ytr, n_atoms=6, T=3, n_iter=8, seed=0, T_min=2)
    # encode c8 patches
    c8_coefs = []
    for w in sample8["walks"]:
        S = set(w["S"])
        y = np.array(flatten_upper(adjacency_padded(g8, S, 6)), dtype=np.float64)
        w["y"] = y.tolist()
        if np.linalg.norm(y) < 1e-12:
            x = np.zeros(D.shape[1])
        else:
            x = _omp(D, y, 3)
        c8_coefs.append({"coef": x.tolist(), "abs_coef": np.abs(x).tolist(), "S": w["S"], "has_c4": w["has_c4"]})

    payload = {
        "title": "Graph-level RW → KSVD interactive pipeline",
        "config": {
            "p": cfg.p,
            "q": cfg.q,
            "walk_length": cfg.walk_length,
            "max_nodes": cfg.max_nodes,
            "max_walks": cfg.max_walks,
            "edge_decay": cfg.edge_decay,
            "seed_policy": cfg.seed_policy,
            "cover_target": cfg.cover_target,
            "hard_delete": False,
        },
        "legend": {
            "seed": "Walk start (degree-stratified)",
            "soft_decay": "Traversed edge weight *= decay (never deleted)",
            "S": "Node set of patch after walk",
            "induced": "G[S] edges among S",
            "D": "Shared dictionary atoms (columns)",
            "X": "Sparse coefficients per patch",
            "s_G": "Graph readout from X (energy + stats)",
        },
        "graphs": {
            "c4": {
                **graph_payload(g4, "demo_C4", "class0_has_C4"),
                "sample": sample4,
                "b0": b0_patches(g4, 6),
                "ksvd": ksvd4,
            },
            "c8": {
                **graph_payload(g8, "demo_C8", "class1_no_C4"),
                "sample": sample8,
                "b0": b0_patches(g8, 6),
                "encode_with_c4_D": c8_coefs,
            },
        },
        "steps_ui": [
            {"id": "seeds", "title": "1. Seeds"},
            {"id": "walk", "title": "2. Walk + soft decay"},
            {"id": "patches", "title": "3. Patches G[S]"},
            {"id": "dict", "title": "4. Dictionary D"},
            {"id": "encode", "title": "5. Encode X → s_G"},
            {"id": "compare", "title": "6. B0 vs RW"},
        ],
    }

    path = out_dir / "data.json"
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"wrote {path}")
    print(f"c4 walks={sample4['n_walks']} cover={sample4['traj_cover']:.3f}")
    print(f"c4 recon={ksvd4['recon_rel']:.4f} atoms={len(ksvd4['atoms'])}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Trace Beam8 candidate selection on one demo graph, matching the real sampler.

Reproduces the exact chain produced by ``sample_marginal_candidate_cover`` with
the same RNG seed (asserted) and dumps the per-step candidate beam to
results/viz_beam8_v2/beam8_trace.json for the interactive HTML.

Usage:
  python -m code.viz_beam8_trace
"""
from __future__ import annotations

import json
import sys
from itertools import combinations
from pathlib import Path

import numpy as np

_TRACK = Path(__file__).resolve().parents[1]
if str(_TRACK) not in sys.path:
    sys.path.insert(0, str(_TRACK))

from code.marginal_candidate_cover import (
    _candidate_score,
    _greedy_fill,
    _pairs,
    _retained_potential,
    sample_marginal_candidate_cover,
)
from code.overlap_cover import (
    _make_cover,
    _make_patch,
    is_connected,
    sample_edge_target_bridge_cover,
)


def make_demo_graph(n: int = 40, seed: int = 7) -> np.ndarray:
    rng = np.random.default_rng(seed)
    adj = np.zeros((n, n), dtype=float)
    # two dense blocks + a sparse tail, connected by a bridge band
    blocks = [(0, 12), (12, 26), (26, 40)]
    for lo, hi in blocks:
        for i in range(lo, hi):
            for j in range(i + 1, hi):
                if rng.random() < 0.55:
                    adj[i, j] = adj[j, i] = 1.0
    for i in range(0, n):
        for j in range(i + 1, n):
            if rng.random() < 0.06:
                adj[i, j] = adj[j, i] = 1.0
    # guarantee connectivity ring
    for i in range(n - 1):
        if adj[i, i + 1] == 0:
            adj[i, i + 1] = adj[i + 1, i] = 1.0
    return adj


def spring_layout(adj: np.ndarray, seed: int = 1, iters: int = 400) -> dict:
    n = adj.shape[0]
    rng = np.random.default_rng(seed)
    pos = rng.normal(size=(n, 2))
    disp = np.zeros((n, 2))
    for _ in range(iters):
        disp.fill(0.0)
        for i in range(n):
            for j in range(i + 1, n):
                d = pos[i] - pos[j]
                dist = float(np.linalg.norm(d)) + 1e-6
                f = d / dist * (80.0 / (dist + 1e-3))
                disp[i] += f
                disp[j] -= f
        for i in range(n):
            for j in np.flatnonzero(adj[i]):
                if j <= i:
                    continue
                d = pos[j] - pos[i]
                dist = float(np.linalg.norm(d)) + 1e-6
                f = d * 0.012
                disp[i] += f
                disp[j] -= f
        disp = np.clip(disp, -25.0, 25.0)
        pos = pos + 0.12 * disp
        span = np.ptp(pos, axis=0).max() + 1e-9
        pos = pos / span * 40.0
    xs = pos[:, 0]
    ys = pos[:, 1]
    xs = (xs - xs.min()) / (np.ptp(xs) + 1e-9)
    ys = (ys - ys.min()) / (np.ptp(ys) + 1e-9)
    return {int(i): [float(xs[i] * 840 + 60), float(ys[i] * 560 + 50)] for i in range(n)}


def trace(adj: np.ndarray, rng: np.random.Generator, *, n_patches: int, patch_size: int, target_overlap: int, retained_beam: int, candidate_restarts: int):
    steps = []

    first_cover = sample_edge_target_bridge_cover(
        adj, rng, n_patches=1, patch_size=patch_size, target_overlap=target_overlap
    )
    patches = [first_cover.patches[0]]
    observed_pairs = _pairs(patches[0].node_ids)
    covered_edges = {pair for pair in observed_pairs if adj[pair[0], pair[1]] != 0}
    covered_nodes = set(patches[0].node_ids)

    def register(nodes):
        pairs = _pairs(nodes)
        observed_pairs.update(pairs)
        covered_edges.update(pair for pair in pairs if adj[pair[0], pair[1]] != 0)
        covered_nodes.update(nodes)

    n_nodes = adj.shape[0]
    true_edges = {
        (int(a), int(b)) for a in range(n_nodes) for b in range(a + 1, n_nodes) if adj[a, b] != 0
    }

    def progress():
        return {
            "node_cover": len(covered_nodes) / n_nodes,
            "edge_cover": len(covered_edges) / len(true_edges),
            "pair_cover": len(observed_pairs) / (n_nodes * (n_nodes - 1) / 2),
        }

    steps.append(
        {
            "patch": 0,
            "nodes": list(patches[0].node_ids),
            "center": int(patches[0].center),
            "kind": "target-edge seed",
            "selected_retained": None,
            "beam": None,
            "progress": progress(),
        }
    )

    while len(patches) < n_patches:
        previous = patches[-1]
        previous_set = set(previous.node_ids)
        retained_candidates = []
        for retained in combinations(previous.node_ids, target_overlap):
            induced = adj[np.ix_(retained, retained)]
            if not is_connected(induced):
                continue
            potential = _retained_potential(adj, retained, previous_set, covered_edges, covered_nodes)
            retained_candidates.append((potential, list(retained)))
        retained_candidates.sort(key=lambda item: item[0], reverse=True)
        top = retained_candidates[:retained_beam]

        completed = []
        for _potential, retained in top:
            for _restart in range(candidate_restarts):
                candidate = _greedy_fill(
                    adj,
                    retained,
                    previous_set,
                    observed_pairs,
                    covered_edges,
                    covered_nodes,
                    rng,
                    patch_size=patch_size,
                )
                if candidate is not None:
                    completed.append({"retained": retained, "nodes": list(candidate)})
        scores = [
            _candidate_score(adj, c["nodes"], observed_pairs, covered_edges, covered_nodes)
            for c in completed
        ]
        best = max(scores)
        choices = [c for c, s in zip(completed, scores) if s == best]
        selected = choices[int(rng.integers(len(choices)))]
        new_nodes = [node for node in selected["nodes"] if node not in previous_set]
        center = new_nodes[0] if new_nodes else selected["nodes"][0]

        # score snapshot must be taken BEFORE registering the selection, exactly
        # like the real sampler (scores are marginal gains against the pre-step
        # observed state). Otherwise the selected candidate would read 0.
        completed_json = [
            {
                "retained": c["retained"],
                "nodes": c["nodes"],
                "score": list(sc),
            }
            for c, sc in zip(completed, scores)
        ]
        patches.append(_make_patch(adj, selected["nodes"], center))
        register(selected["nodes"])

        beam_json = [
            {
                "potential": list(pot),
                "retained": list(ret),
            }
            for pot, ret in top
        ]
        steps.append(
            {
                "patch": len(patches) - 1,
                "nodes": list(patches[-1].node_ids),
                "center": int(patches[-1].center),
                "kind": "beam8 selection",
                "retained_total": len(retained_candidates),
                "connected_total": len(retained_candidates),
                "beam": beam_json,
                "completed": completed_json,
                "best_score": list(best),
                "selected_retained": selected["retained"],
                "overlap_nodes": sorted(set(previous.node_ids) & set(selected["nodes"])),
                "progress": progress(),
            }
        )

    cover = _make_cover("marginal_candidate_beam", patches)
    return cover, steps


def main():
    adj = make_demo_graph()
    rng = np.random.default_rng(42)
    n_patches, patch_size, target_overlap, beam, restarts = 8, 10, 3, 8, 1
    cover, steps = trace(
        adj, rng, n_patches=n_patches, patch_size=patch_size,
        target_overlap=target_overlap, retained_beam=beam, candidate_restarts=restarts,
    )

    # verify against the real sampler
    rng2 = np.random.default_rng(42)
    real = sample_marginal_candidate_cover(
        adj, rng2, n_patches=n_patches, patch_size=patch_size,
        target_overlap=target_overlap, retained_beam=beam, candidate_restarts=restarts,
    )
    real_nodes = [list(p.node_ids) for p in real.patches]
    traced_nodes = [list(p.node_ids) for p in cover.patches]
    assert real_nodes == traced_nodes, f"MISMATCH\n{real_nodes}\n{traced_nodes}"

    transitions = [
        [[int(a), int(b)] for a, b in t.left_to_right_slots]
        for t in cover.transitions
    ]
    payload = {
        "n": adj.shape[0],
        "layout": spring_layout(adj),
        "edges": [
            [int(a), int(b)]
            for a in range(adj.shape[0])
            for b in range(a + 1, adj.shape[0])
            if adj[a, b] != 0
        ],
        "patches": [list(p.node_ids) for p in cover.patches],
        "transitions": transitions,
        "steps": steps,
        "edge_total": int(adj.sum() // 2),
    }
    out = _TRACK / "results" / "viz_beam8_v2" / "beam8_trace.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
    print(f"OK  matched real sampler; wrote {out}")
    print("patches:", [list(p.node_ids) for p in cover.patches])


if __name__ == "__main__":
    main()

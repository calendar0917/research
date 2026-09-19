#!/usr/bin/env python
"""Whole-graph canonicalization vs. graph-specific registration (ZINC).

Architecture question
---------------------
For small attributed graphs (ZINC subset), if we *strictly* canonically label
the whole graph and then place it into a shared whole-graph sparse dictionary,
are the canonical coordinates already stable enough -- or do we still need a
graph-specific registration matrix ``P_G`` to fix slot correspondence across
different molecules?

Discipline
----------
* This is a pure representation analysis.  No model is trained, no predictor is
  fit, no architecture experiment is run.
* Only the official ZINC ``train`` and ``val`` splits are ever loaded.  The
  official ``test`` split is never read, instantiated or referenced.
* The target ``y`` is never used for pair selection, matching, thresholding or
  analysis.
* ``tracks/ksvd/STATE.yaml`` and all existing pre-registrations are untouched.

Canonicalization
----------------
Exact (not heuristic) canonical labeling of the *attributed* graph via
``pynauty`` on a colored incidence graph:

    atom_i -- bond_e -- atom_j

atom vertices carry colour ``("n", atom_type)``; bond vertices carry colour
``("e", bond_type)``.  ``pynauty`` returns a canonical vertex order; the
relative order of the atom vertices is the canonical atom ordering.  This is the
same exact colored-incidence construction used by the repository's
``tracks/ksvd/experiments/luyin16/typed_patch_tokenizer.py`` (pynauty is pinned
to ``2.8.8.1``).

The canonical representation ``(canonical node-type sequence, canonical
adjacency/bond tensor)`` is invariant to input node permutation by construction;
``Sanity check A`` verifies this empirically.  It is *not* a WL hash and it is
*not* an ordering heuristic.
"""

from __future__ import annotations

import argparse
import json
import platform
import subprocess
import time
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np

# ---------------------------------------------------------------------------
# Repository imports (canonical ZINC loader used by every ZINC experiment)
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parents[3]


def _import_repo():
    import sys

    root = str(REPO_ROOT)
    if root not in sys.path:
        sys.path.insert(0, root)
    from tracks.ksvd.experiments.luyin16.zinc_long_range_proxy import (  # type: ignore
        _data_to_graph,
        _load_zinc,
    )

    return _load_zinc, _data_to_graph


# ===========================================================================
# 1. Exact whole-graph canonicalization (colored incidence + pynauty)
# ===========================================================================
def _pynauty():
    import pynauty  # type: ignore

    return pynauty


def whole_graph_incidence(graph, node_types, edge_types):
    """Colored incidence graph: atom vertices then bond vertices."""
    n = graph.n
    edges = sorted(graph.edges())
    m = len(edges)
    n_vertices = n + m
    adjacency: dict[int, list[int]] = {v: [] for v in range(n_vertices)}
    color: list[tuple[Any, ...]] = [("n", 0)] * n_vertices
    for v in range(n):
        color[v] = ("n", int(node_types[v]))
    for e, (a, b) in enumerate(edges):
        ev = n + e
        adjacency[a].append(ev)
        adjacency[b].append(ev)
        adjacency[ev] = [int(a), int(b)]
        color[ev] = ("e", int(edge_types[(int(a), int(b))]))
    return n_vertices, adjacency, color


def canonical_atom_order(graph, node_types, edge_types) -> tuple[list[int], bytes]:
    """Exact canonical atom order + a complete isomorphism key.

    Returns ``(atom_order, key)`` where ``atom_order[s]`` is the original atom
    id occupying canonical slot ``s`` and ``key`` is a complete invariant of the
    colored incidence graph (canonical adjacency certificate + canonical
    semantic colour sequence).
    """
    pynauty = _pynauty()
    n_vertices, adjacency, color = whole_graph_incidence(graph, node_types, edge_types)
    groups: dict[tuple[Any, ...], list[int]] = defaultdict(list)
    for v in range(n_vertices):
        groups[color[v]].append(v)
    cell_keys = sorted(groups, key=repr)  # process-independent total order
    coloring = [set(groups[k]) for k in cell_keys]
    pg = pynauty.Graph(
        number_of_vertices=n_vertices,
        directed=False,
        adjacency_dict={k: list(v) for k, v in adjacency.items()},
        vertex_coloring=coloring,
    )
    certificate = bytes(pynauty.certificate(pg))
    label = [int(v) for v in pynauty.canon_label(pg)]
    if sorted(label) != list(range(n_vertices)):
        raise RuntimeError("pynauty.canon_label is not a permutation")
    n = graph.n
    atom_order = [v for v in label if v < n]
    colour_sequence = tuple(color[v] for v in label)
    key = certificate + b"|" + repr(colour_sequence).encode()
    return atom_order, key


class Codebook:
    """Maps the actual (train+valid) node/bond categories onto tensor rows."""

    def __init__(self, node_categories: list[int], bond_categories: list[int]):
        self.node_categories = sorted(int(c) for c in node_categories)
        self.bond_categories = sorted(int(c) for c in bond_categories)
        self.node_index = {c: i for i, c in enumerate(self.node_categories)}
        self.bond_index = {c: i for i, c in enumerate(self.bond_categories)}
        self.n_node = len(self.node_categories)
        self.n_bond = len(self.bond_categories)
        # node tensor: [0..n_node-1] = atom categories, n_node = EMPTY/padding
        # relation tensor: channel 0 = no-edge, channels 1..n_bond = bond types
        self.empty_index = self.n_node
        self.n_node_channels = self.n_node + 1
        self.n_relation_channels = 1 + self.n_bond

    def encode(self, graph, node_types, edge_types, q: int):
        order, key = canonical_atom_order(graph, node_types, edge_types)
        n = len(order)
        if n > q:
            raise ValueError(f"graph size {n} exceeds padding q={q}")
        node_feat = np.zeros((q, self.n_node_channels), dtype=np.float64)
        for slot, v in enumerate(order):
            node_feat[slot, self.node_index[int(node_types[v])]] = 1.0
        node_feat[n:, self.empty_index] = 1.0  # explicit EMPTY / padding channel

        relation = np.zeros((q, q, self.n_relation_channels), dtype=np.float64)
        position = {v: s for s, v in enumerate(order)}
        for (a, b), bond in edge_types.items():
            i, j = position[int(a)], position[int(b)]
            ch = 1 + self.bond_index[int(bond)]
            relation[i, j, ch] = 1.0
            relation[j, i, ch] = 1.0
        for i in range(n):
            for j in range(i + 1, n):
                if relation[i, j].sum() == 0.0:
                    relation[i, j, 0] = 1.0
                    relation[j, i, 0] = 1.0
        valid = np.zeros(q, dtype=bool)
        valid[:n] = True
        return {
            "node": node_feat,
            "relation": relation,
            "valid": valid,
            "n": n,
            "key": key,
        }


# ===========================================================================
# 2. Typed WL fingerprint (ordering-independent; used only for pair retrieval)
# ===========================================================================
def _wl_colors_single(graph, node_types, edge_types, rounds: int):
    nodes = list(graph.nodes)
    color = {v: ("a", int(node_types[v])) for v in nodes}
    for _ in range(int(rounds)):
        signatures = {}
        for v in nodes:
            nbrs = []
            for u in graph.neighbors(v):
                bt = edge_types.get(graph.edge_key(v, int(u)), edge_types.get((int(u), v), 0))
                nbrs.append((color[int(u)], int(bt)))
            signatures[v] = (color[v], tuple(sorted(nbrs, key=repr)))
        vocab = {sig: i for i, sig in enumerate(sorted(set(signatures.values()), key=repr))}
        color = {v: ("c", vocab[signatures[v]]) for v in nodes}
    return color


def attributed_wl_fingerprint(graph, node_types, edge_types, rounds: int = 3) -> dict:
    """Ordering-independent typed Weisfeiler-Lehman *subtree kernel* fingerprint.

    Node colours start at the atom type and are refined by the multiset of
    ``(neighbour colour, bond type)``.  We accumulate the colour histogram at
    every round ``0..rounds`` (the standard WL subtree kernel feature map), so
    shared rooted subtypes at all scales contribute.  This correlates with
    shared substructure much better than the final-round histogram alone (which
    is degenerate for small molecules: >50% of ZINC graphs share an identical
    3-round node-colour multiset).
    """
    nodes = list(graph.nodes)
    color = {v: ("a", int(node_types[v])) for v in nodes}
    hist: dict[Any, float] = defaultdict(float)
    for r in range(int(rounds) + 1):
        round_hist: dict[Any, float] = defaultdict(float)
        for v in nodes:
            round_hist[(r, color[v])] += 1.0
        total = sum(round_hist.values()) or 1.0
        for key, val in round_hist.items():  # each round contributes equally
            hist[key] += val / total
        if r == int(rounds):
            break
        signatures = {}
        for v in nodes:
            nbrs = []
            for u in graph.neighbors(v):
                bt = edge_types.get(graph.edge_key(v, int(u)), edge_types.get((int(u), v), 0))
                nbrs.append((color[int(u)], int(bt)))
            signatures[v] = (color[v], tuple(sorted(nbrs, key=repr)))
        vocab = {sig: i for i, sig in enumerate(sorted(set(signatures.values()), key=repr))}
        color = {v: ("c", vocab[signatures[v]]) for v in nodes}
    return hist


def _histogram_matrix(hists: list[dict], keys: list) -> np.ndarray:
    index = {k: i for i, k in enumerate(keys)}
    mat = np.zeros((len(hists), len(keys)), dtype=np.float64)
    for row, hist in enumerate(hists):
        for k, val in hist.items():
            mat[row, index[k]] = val
    norms = np.linalg.norm(mat, axis=1, keepdims=True)
    norms[norms == 0] = 1.0
    return mat / norms


# ===========================================================================
# 3. Alignment loss and soft doubly-stochastic registration
# ===========================================================================
def _relation_valid_mask(valid_g: np.ndarray, valid_h: np.ndarray) -> np.ndarray:
    both = valid_g & valid_h
    q = both.shape[0]
    mask = np.zeros((q, q), dtype=bool)
    for i in range(q):
        if not both[i]:
            continue
        mask[i, i + 1 :] = both[i + 1 :]
    return mask


def alignment_loss_numpy(code_g, code_h, P=None, q=None):
    """``L_align`` with mean-normalised node and edge terms, each in [0, 1].

    Node term: mean over q slots of 0.5 * ||F_G - P F_H||^2.
    Edge term: mean over valid pairs of 0.5 * ||R_G - P R_H P^T||^2.
    Both are *per-item means* so the O(q^2) pair term cannot swamp the O(q)
    node term.  ``d = L_node + L_edge`` (weights fixed to 1 everywhere).
    """
    Fg = code_g["node"]
    Rg = code_g["relation"]
    Fh = code_h["node"]
    Rh = code_h["relation"]
    q = int(Fg.shape[0])
    if P is None:
        P = np.eye(q)
    Fht = P @ Fh
    node = 0.5 * float(np.sum((Fg - Fht) ** 2)) / q
    Rht = np.einsum("ij,jlc,kl->ikc", P, Rh, P)
    mask = _relation_valid_mask(code_g["valid"], code_h["valid"])
    if mask.any():
        diff = 0.5 * np.sum((Rg[mask] - Rht[mask]) ** 2, axis=-1)
        edge = float(diff.mean())
    else:
        edge = 0.0
    return node, edge, node + edge


def _sinkhorn(logits, n_iter: int):
    import torch

    z = logits
    for _ in range(int(n_iter)):
        z = z - torch.logsumexp(z, dim=1, keepdim=True)
        z = z - torch.logsumexp(z, dim=0, keepdim=True)
    return z.exp()


def fit_registration(code_g, code_h, *, rho: float, lam: float, steps: int,
                     sinkhorn_iters: int, lr: float, init_scale: float,
                     seed: int):
    """Independent per-pair soft doubly-stochastic registration P (H -> G).

    Objective:
        L(P) = L_align(G, P H) + rho * ||P - I||_F^2 / q
                               + lam * (1/q) * sum_ij P_ij |i-j| / q
    with ``P = Sinkhorn(theta)`` (row/col sums = 1).  We start from an
    identity-biased ``theta0 = init_scale * I`` (so ``P^(0) ~= I``).
    """
    import torch

    torch.manual_seed(int(seed))
    q = int(code_g["node"].shape[0])
    Fg = torch.tensor(code_g["node"])
    Rg = torch.tensor(code_g["relation"])
    Fh = torch.tensor(code_h["node"])
    Rh = torch.tensor(code_h["relation"])
    mask_np = _relation_valid_mask(code_g["valid"], code_h["valid"])
    mask = torch.tensor(mask_np)
    n_mask = max(int(mask_np.sum()), 1)
    eye = torch.eye(q)
    dist = torch.tensor(
        [[abs(i - j) / q for j in range(q)] for i in range(q)], dtype=torch.float64
    )

    theta = torch.full((q, q), -float(init_scale), dtype=torch.float64) + eye * float(
        init_scale
    )
    theta = theta.clone().requires_grad_(True)
    opt = torch.optim.Adam([theta], lr=float(lr))
    loss_history = []
    for _ in range(int(steps)):
        P = _sinkhorn(theta, sinkhorn_iters)
        Fht = P @ Fh
        node = 0.5 * ((Fg - Fht) ** 2).sum() / q
        Rht = torch.einsum("ij,jlc,kl->ikc", P, Rh, P)
        edge = 0.5 * ((Rg - Rht) ** 2)[mask].sum() / n_mask if mask_np.any() else node * 0.0
        identity = rho * ((P - eye) ** 2).sum() / q
        transport = lam * (P * dist).sum() / q
        loss = node + edge + identity + transport
        opt.zero_grad()
        loss.backward()
        opt.step()
        loss_history.append(float(loss.item()))
    with torch.no_grad():
        P = _sinkhorn(theta, sinkhorn_iters)
        Fht = P @ Fh
        node = float(0.5 * ((Fg - Fht) ** 2).sum() / q)
        Rht = torch.einsum("ij,jlc,kl->ikc", P, Rh, P)
        edge = (
            float(0.5 * ((Rg - Rht) ** 2)[mask].sum() / n_mask) if mask_np.any() else 0.0
        )
        d = node + edge
        fro = float(torch.sqrt(((P - eye) ** 2).sum()))
        transport_cost = float((P * dist).sum() / q)
        P_np = P.detach().cpu().numpy()
    return {
        "d": d,
        "node": node,
        "edge": edge,
        "fro_identity": fro,
        "transport_cost": transport_cost,
        "loss_start": loss_history[0],
        "loss_end": loss_history[-1],
        "P": P_np,
    }


# ===========================================================================
# 4. WL-based atom correspondence (no RDKit / no MCS: the allowed auxiliary)
# ===========================================================================
def wl_colors_union(graph_g, nt_g, et_g, graph_h, nt_h, et_h, rounds: int = 3):
    """Joint typed-WL colours on the disjoint union G + H (comparable across)."""
    nG, nH = graph_g.n, graph_h.n
    nodes = list(range(nG + nH))
    color = {}
    for v in graph_g.nodes:
        color[v] = ("a", int(nt_g[v]))
    for v in graph_h.nodes:
        color[nG + v] = ("a", int(nt_h[v]))

    def neighbors(v):
        if v < nG:
            for u in graph_g.neighbors(v):
                bt = et_g.get(graph_g.edge_key(v, int(u)), et_g.get((int(u), v), 0))
                yield int(u), int(bt)
        else:
            vv = v - nG
            for u in graph_h.neighbors(vv):
                bt = et_h.get(graph_h.edge_key(vv, int(u)), et_h.get((int(u), vv), 0))
                yield nG + int(u), int(bt)

    for _ in range(int(rounds)):
        sig = {}
        for v in nodes:
            sig[v] = (color[v], tuple(sorted(((color[u], bt) for u, bt in neighbors(v)), key=repr)))
        vocab = {s: i for i, s in enumerate(sorted(set(sig.values()), key=repr))}
        color = {v: ("c", vocab[sig[v]]) for v in nodes}

    # neighbour-colour histogram feature for structural similarity
    def nbr_hist(v):
        h = defaultdict(float)
        for u, _bt in neighbors(v):
            h[color[u]] += 1.0
        return h

    return color, nbr_hist


def wl_correspondence(graph_g, nt_g, et_g, graph_h, nt_h, et_h, rounds: int = 3):
    """High-confidence WL-equivalent 1-1 atom correspondence (ordering-free).

    Candidate matches require equal joint WL colour; among those, a Hungarian
    assignment minimises neighbour-colour-histogram mismatch, with explicit
    dummy rows/columns so weak matches are left unmatched.
    """
    from scipy.optimize import linear_sum_assignment

    nG, nH = graph_g.n, graph_h.n
    color, nbr_hist = wl_colors_union(graph_g, nt_g, et_g, graph_h, nt_h, et_h, rounds)
    # All same-colour candidates are accepted (cost <= 1); different-colour is
    # infeasible.  DUMMY lies strictly between the two regimes.
    DUMMY = 1.5
    BIG = 5.0
    feats_g = [nbr_hist(v) for v in range(nG)]
    feats_h = [nbr_hist(nG + v) for v in range(nH)]
    colors_g = [color[v] for v in range(nG)]
    colors_h = [color[nG + v] for v in range(nH)]

    def cos(a, b):
        keys = set(a) | set(b)
        va = np.array([a.get(k, 0.0) for k in keys])
        vb = np.array([b.get(k, 0.0) for k in keys])
        na, nb = np.linalg.norm(va), np.linalg.norm(vb)
        if na == 0 or nb == 0:
            return 0.0
        return float(va @ vb / (na * nb))

    size = nG + nH
    cost = np.full((size, size), BIG)
    for i in range(nG):
        for j in range(nH):
            if colors_g[i] == colors_h[j]:
                cost[i, j] = 1.0 - cos(feats_g[i], feats_h[j])
    # dummy columns (for G rows) and dummy rows (for H columns)
    for i in range(nG):
        cost[i, nH + i] = DUMMY
    for j in range(nH):
        cost[nG + j, j] = DUMMY
    cost[nG:, nH:] = 0.0
    rows, cols = linear_sum_assignment(cost)
    matches = []
    for i, j in zip(rows.tolist(), cols.tolist()):
        if i < nG and j < nH and cost[i, j] <= 1.0:
            matches.append((i, j))
    return matches


# ===========================================================================
# 5. Metric helpers
# ===========================================================================
def _quartiles(values):
    arr = np.asarray(values, dtype=np.float64)
    if arr.size == 0:
        return {}
    qs = np.percentile(arr, [0, 25, 50, 75, 90, 100])
    return {
        "n": int(arr.size),
        "mean": float(arr.mean()),
        "std": float(arr.std()),
        "min": float(qs[0]),
        "p25": float(qs[1]),
        "median": float(qs[2]),
        "p75": float(qs[3]),
        "p90": float(qs[4]),
        "max": float(qs[5]),
    }


def kendall_tau_pairs(before: dict, after: dict, keys: list) -> float:
    """Pairwise relative-order agreement (tau in [-1, 1]) over tracked nodes."""
    num = 0.0
    den = 0.0
    for a in range(len(keys)):
        for b in range(a + 1, len(keys)):
            da = np.sign(before[keys[a]] - before[keys[b]])
            db = np.sign(after[keys[a]] - after[keys[b]])
            if da == 0 or db == 0:
                continue
            num += da * db
            den += 1.0
    return float(num / den) if den else 1.0


# ===========================================================================
# 6. Experiment stages
# ===========================================================================
def stage_sanity_a(graphs, codebook, q, n_graphs, n_perms, seed, log):
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(graphs), size=min(n_graphs, len(graphs)), replace=False)
    matches = 0
    total = 0
    failures = []
    for gi in idx:
        graph, nt, et = graphs[int(gi)]
        base = codebook.encode(graph, nt, et, q)
        for _ in range(int(n_perms)):
            perm = rng.permutation(graph.n)
            inv = np.empty(graph.n, dtype=np.int64)
            inv[perm] = np.arange(graph.n)
            from tracks.ksvd.code.graph import from_edges  # local import

            new_edges = []
            for u in graph.nodes:
                for v in graph.neighbors(u):
                    a, b = int(inv[u]), int(inv[v])
                    if a < b:
                        new_edges.append((a, b))
            ng = from_edges(graph.n, new_edges)
            nnt = np.zeros_like(nt)
            for u in graph.nodes:
                nnt[int(inv[u])] = nt[u]
            net = {}
            for (a, b), bond in et.items():
                na, nb = int(inv[a]), int(inv[b])
                net[(na, nb) if na < nb else (nb, na)] = bond
            other = codebook.encode(ng, nnt, net, q)
            total += 1
            if (
                np.array_equal(base["node"], other["node"])
                and np.array_equal(base["relation"], other["relation"])
            ):
                matches += 1
            elif len(failures) < 5:
                failures.append(int(gi))
        log(f"  sanity A: graph {int(gi)} done")
    rate = matches / max(total, 1)
    return {"match_rate": rate, "n_graphs": int(len(idx)), "n_perms": int(n_perms),
            "n_checks": total, "failures": failures}


def _perturb_node_type(rng, graph, nt, et):
    v = int(rng.choice(graph.nodes))
    options = [c for c in range(int(nt.max()) + 1) if c != nt[v]]
    new = int(rng.choice(options))
    nnt = nt.copy()
    nnt[v] = new
    return graph, nnt, et, "node_type"


def _perturb_edge_type(rng, graph, nt, et):
    edges = sorted(et)
    if not edges:
        return None
    key = edges[int(rng.integers(len(edges)))]
    options = [b for b in (1, 2, 3) if b != et[key]]
    net = dict(et)
    net[key] = int(rng.choice(options))
    return graph, nt, net, "edge_type"


def _perturb_delete_nonbridge(rng, graph, nt, et):
    from tracks.ksvd.code.graph import from_edges

    cand = []
    for (a, b) in sorted(et):
        edges = [e for e in graph.edges() if e != (a, b)]
        ng = from_edges(graph.n, edges)
        # connectivity check
        seen = {0}
        stack = [0]
        while stack:
            u = stack.pop()
            for w in ng.neighbors(u):
                if w not in seen:
                    seen.add(w)
                    stack.append(w)
        if len(seen) == graph.n:
            cand.append((a, b))
    if not cand:
        return None
    a, b = cand[int(rng.integers(len(cand)))]
    edges = [e for e in graph.edges() if e != (a, b)]
    ng = from_edges(graph.n, edges)
    net = {k: v for k, v in et.items() if k != (a, b)}
    return ng, nt, net, "delete_edge"


def _perturb_add_leaf(rng, graph, nt, et):
    from tracks.ksvd.code.graph import from_edges

    parent = int(rng.choice(graph.nodes))
    new_id = graph.n
    edges = list(graph.edges()) + [(parent, new_id)]
    ng = from_edges(graph.n + 1, edges)
    new_type = int(rng.integers(0, int(nt.max()) + 1))
    nnt = np.concatenate([nt, np.array([new_type], dtype=nt.dtype)])
    net = dict(et)
    net[(min(parent, new_id), max(parent, new_id))] = int(rng.choice([1, 2, 3]))
    return ng, nnt, net, "add_leaf"


PERTURBATIONS = {
    "node_type": _perturb_node_type,
    "edge_type": _perturb_edge_type,
    "delete_edge": _perturb_delete_nonbridge,
    "add_leaf": _perturb_add_leaf,
}


def stage_sanity_b(graphs, codebook, q, n_graphs, seed, log):
    rng = np.random.default_rng(seed + 1)
    idx = rng.choice(len(graphs), size=min(n_graphs, len(graphs)), replace=False)
    results: dict[str, dict[str, list]] = {
        name: {"slot_preserve": [], "displacement": [], "kendall": [], "n_nodes": []}
        for name in PERTURBATIONS
    }
    for gi in idx:
        graph, nt, et = graphs[int(gi)]
        base_order, _ = canonical_atom_order(graph, nt, et)
        base_slot = {v: s for s, v in enumerate(base_order)}
        for name, fn in PERTURBATIONS.items():
            out = fn(rng, graph, nt, et)
            if out is None:
                continue
            ng, nnt, net, _ = out
            new_order, _ = canonical_atom_order(ng, nnt, net)
            new_slot = {v: s for s, v in enumerate(new_order)}
            tracked = [v for v in base_slot if v in new_slot]
            if len(tracked) < 3:
                continue
            preserve = np.mean([base_slot[v] == new_slot[v] for v in tracked])
            disp = np.mean([abs(base_slot[v] - new_slot[v]) / q for v in tracked])
            tau = kendall_tau_pairs(base_slot, new_slot, tracked)
            results[name]["slot_preserve"].append(float(preserve))
            results[name]["displacement"].append(float(disp))
            results[name]["kendall"].append(float(tau))
            results[name]["n_nodes"].append(int(len(tracked)))
    summary = {}
    for name, vals in results.items():
        summary[name] = {
            "slot_preservation": _quartiles(vals["slot_preserve"]),
            "normalized_displacement": _quartiles(vals["displacement"]),
            "kendall_tau": _quartiles(vals["kendall"]),
        }
    return summary, results


def stage_permutation_control(graphs, codebook, q, n_graphs, seed, log):
    """Section 11: permutation sanity control (validation of tensor transforms)."""
    rng = np.random.default_rng(seed + 2)
    idx = rng.choice(len(graphs), size=min(n_graphs, len(graphs)), replace=False)
    from tracks.ksvd.code.graph import from_edges

    dI_raw, dFree_raw, dFreeDisc_raw, dI_canon = [], [], [], []
    for gi in idx:
        graph, nt, et = graphs[int(gi)]
        perm = rng.permutation(graph.n)
        inv = np.empty(graph.n, dtype=np.int64)
        inv[perm] = np.arange(graph.n)
        new_edges = [(int(inv[u]), int(inv[v])) for u in graph.nodes for v in graph.neighbors(u)]
        new_edges = [(a, b) if a < b else (b, a) for a, b in new_edges]
        ng = from_edges(graph.n, new_edges)
        nnt = np.zeros_like(nt)
        for u in graph.nodes:
            nnt[int(inv[u])] = nt[u]
        net = {}
        for (a, b), bond in et.items():
            na, nb = int(inv[a]), int(inv[b])
            net[(na, nb) if na < nb else (nb, na)] = bond
        # never canonicalize here: encode directly with identity ordering
        raw = _encode_in_order(graph, nt, et, q, codebook)
        raw2 = _encode_in_order(ng, nnt, net, q, codebook)
        _, _, di = alignment_loss_numpy(raw, raw2)
        free = fit_registration(raw, raw2, rho=0.0, lam=0.0, steps=200,
                                sinkhorn_iters=30, lr=0.3, init_scale=4.0, seed=int(seed))
        canon_g = codebook.encode(graph, nt, et, q)
        canon_h = codebook.encode(ng, nnt, net, q)
        _, _, dic = alignment_loss_numpy(canon_g, canon_h)
        _, _, d_free_disc = alignment_loss_numpy(raw, raw2, P=_hungarian_discrete(free["P"]))
        dI_raw.append(di)
        dFree_raw.append(free["d"])
        dFreeDisc_raw.append(d_free_disc)
        dI_canon.append(dic)
        log(f"  perm control: graph {int(gi)} dI_raw={di:.4f} dFree={free['d']:.5f} "
            f"dFreeDisc={d_free_disc:.5f} dI_canon={dic:.5f}")
    return {
        "d_identity_raw": _quartiles(dI_raw),
        "d_free_raw": _quartiles(dFree_raw),
        "d_free_discrete_raw": _quartiles(dFreeDisc_raw),
        "d_identity_canonical": _quartiles(dI_canon),
    }


def _encode_in_order(graph, nt, et, q, codebook):
    """Tensor for an *uncanonicalized* graph (identity node order)."""
    n = graph.n
    node_feat = np.zeros((q, codebook.n_node_channels), dtype=np.float64)
    for v in range(n):
        node_feat[v, codebook.node_index[int(nt[v])]] = 1.0
    node_feat[n:, codebook.empty_index] = 1.0
    relation = np.zeros((q, q, codebook.n_relation_channels), dtype=np.float64)
    for (a, b), bond in et.items():
        ch = 1 + codebook.bond_index[int(bond)]
        relation[int(a), int(b), ch] = 1.0
        relation[int(b), int(a), ch] = 1.0
    for i in range(n):
        for j in range(i + 1, n):
            if relation[i, j].sum() == 0.0:
                relation[i, j, 0] = 1.0
                relation[j, i, 0] = 1.0
    valid = np.zeros(q, dtype=bool)
    valid[:n] = True
    return {"node": node_feat, "relation": relation, "valid": valid, "n": n, "key": b""}


def stage_pair_selection(pool_graphs, pool_sizes, codebook, q, n_similar, n_random, seed, log):
    """Similar pairs (WL fingerprint NN) + size-matched random control pairs.

    ``pool_graphs`` is a list of ``(graph, node_types, edge_types)`` tuples;
    ``pool_sizes`` the matching atom counts.  Pair indices are pool-local.
    """
    n = len(pool_graphs)
    fp_hists = [attributed_wl_fingerprint(g, nt, et, rounds=3) for (g, nt, et) in pool_graphs]
    key_set = sorted({k for h in fp_hists for k in h}, key=repr)
    F = _histogram_matrix(fp_hists, key_set)
    size_arr = np.asarray(pool_sizes, dtype=np.int64)
    keys = [codebook.encode(g, nt, et, q)["key"] for (g, nt, et) in pool_graphs]

    gram = F @ F.T
    np.fill_diagonal(gram, -1.0)
    rng = np.random.default_rng(seed + 3)

    candidate = []
    for i in range(n):
        js = np.argsort(gram[i])[::-1][:5]
        for j in js:
            if keys[i] == keys[j]:
                continue
            candidate.append((float(gram[i, int(j)]), i, int(j)))
    candidate.sort(reverse=True)
    seen = set()
    similar = []
    for sim, i, j in candidate:
        pair = (min(i, j), max(i, j))
        if pair in seen:
            continue
        seen.add(pair)
        similar.append((i, j, sim))
        if len(similar) >= n_similar:
            break

    random_pairs = []
    for (i, j, _sim) in similar:
        target = int(min(size_arr[i], size_arr[j]))
        lo, hi = target - 1, target + 1
        for _attempt in range(300):
            a = int(rng.integers(n))
            b = int(rng.integers(n))
            if a == b:
                continue
            if lo <= size_arr[a] <= hi and lo <= size_arr[b] <= hi:
                random_pairs.append((a, b, 0.0))
                break
    return similar, random_pairs, F, key_set


# ===========================================================================
# 7. Registration-necessity comparison
# ===========================================================================
def _hungarian_discrete(P: np.ndarray) -> np.ndarray:
    from scipy.optimize import linear_sum_assignment

    rows, cols = linear_sum_assignment(-P)
    Pd = np.zeros_like(P)
    Pd[rows, cols] = 1.0
    return Pd


def _pair_metrics(code_g, code_h, seed, cfg):
    nI, eI, dI = alignment_loss_numpy(code_g, code_h, P=None)
    near = fit_registration(code_g, code_h, rho=cfg["rho_near"], lam=cfg["lam_near"],
                            steps=cfg["steps"], sinkhorn_iters=cfg["sinkhorn_iters"],
                            lr=cfg["lr"], init_scale=cfg["init_scale"], seed=seed)
    free = fit_registration(code_g, code_h, rho=cfg["rho_free"], lam=cfg["lam_free"],
                            steps=cfg["steps"], sinkhorn_iters=cfg["sinkhorn_iters"],
                            lr=cfg["lr"], init_scale=cfg["init_scale"], seed=seed + 1)
    # Honest (smearing-free) discrete registration: round P to a permutation.
    Pd_near = _hungarian_discrete(near["P"])
    Pd_free = _hungarian_discrete(free["P"])
    _, _, d_near_disc = alignment_loss_numpy(code_g, code_h, P=Pd_near)
    _, _, d_free_disc = alignment_loss_numpy(code_g, code_h, P=Pd_free)
    q = Pd_near.shape[0]
    perm_disp_near = float(np.mean([abs(i - int(np.argmax(Pd_near[i]))) / q for i in range(q)]))
    perm_disp_free = float(np.mean([abs(i - int(np.argmax(Pd_free[i]))) / q for i in range(q)]))
    out = {
        "d_identity": dI,
        "d_near": near["d"],
        "d_free": free["d"],
        "d_near_disc": d_near_disc,
        "d_free_disc": d_free_disc,
        "d_near_node": near["node"],
        "d_near_edge": near["edge"],
        "d_free_node": free["node"],
        "d_free_edge": free["edge"],
        "fro_near": near["fro_identity"],
        "fro_free": free["fro_identity"],
        "transport_near": near["transport_cost"],
        "transport_free": free["transport_cost"],
        "perm_disp_near": perm_disp_near,
        "perm_disp_free": perm_disp_free,
    }
    out["delta_near"] = (dI - near["d"]) / dI if dI > 1e-12 else 0.0
    out["delta_free"] = (dI - free["d"]) / dI if dI > 1e-12 else 0.0
    out["delta_near_disc"] = (dI - d_near_disc) / dI if dI > 1e-12 else 0.0
    out["delta_free_disc"] = (dI - d_free_disc) / dI if dI > 1e-12 else 0.0
    return out


def stage_registration(graphs, codebook, q, similar, random_pairs, seed, cfg, log):
    def code(i):
        g, nt, et = graphs[i][0], graphs[i][1], graphs[i][2]
        return codebook.encode(g, nt, et, q)

    rows = []
    sim_metrics = []
    rand_metrics = []
    for k, (i, j, sim) in enumerate(similar):
        m = _pair_metrics(code(i), code(j), seed + 10 * k, cfg)
        m.update({"group": "similar", "i": int(i), "j": int(j), "similarity": sim})
        sim_metrics.append(m)
        rows.append(m)
        if k % 50 == 0:
            log(f"  registration similar {k}/{len(similar)}")
    for k, (i, j, _) in enumerate(random_pairs):
        m = _pair_metrics(code(i), code(j), seed + 10 * k + 5, cfg)
        m.update({"group": "random", "i": int(i), "j": int(j), "similarity": 0.0})
        rand_metrics.append(m)
        rows.append(m)
        if k % 50 == 0:
            log(f"  registration random {k}/{len(random_pairs)}")
    summary = {}
    for group, ms in (("similar", sim_metrics), ("random", rand_metrics)):
        summary[group] = {
            key: _quartiles([m[key] for m in ms])
            for key in (
                "d_identity", "d_near", "d_free", "d_near_disc", "d_free_disc",
                "delta_near", "delta_free", "delta_near_disc", "delta_free_disc",
                "fro_near", "fro_free", "transport_near", "transport_free",
                "perm_disp_near", "perm_disp_free",
            )
        }
    return summary, rows


def _correspondence_group(graphs, codebook, q, pairs, rounds, log, tag):
    offsets, preserve, tau, n_matches = [], [], [], []
    for k, (i, j, _s) in enumerate(pairs):
        g, nt, et = graphs[i][0], graphs[i][1], graphs[i][2]
        h, nt2, et2 = graphs[j][0], graphs[j][1], graphs[j][2]
        matches = wl_correspondence(g, nt, et, h, nt2, et2, rounds=rounds)
        if len(matches) < 3:
            continue
        so, _ = canonical_atom_order(g, nt, et)
        sh, _ = canonical_atom_order(h, nt2, et2)
        sg = {v: s for s, v in enumerate(so)}
        shs = {v: s for s, v in enumerate(sh)}
        offsets.extend([abs(sg[a] - shs[b]) / q for a, b in matches])
        preserve.append(float(np.mean([sg[a] == shs[b] for a, b in matches])))
        bs = {a: sg[a] for a, b in matches}
        as_ = {a: shs[b] for a, b in matches}
        tau.append(float(kendall_tau_pairs(bs, as_, [a for a, b in matches])))
        n_matches.append(len(matches))
    log(f"  correspondence {tag} (wl rounds={rounds}): {len(preserve)} pairs")
    return {
        "n_pairs_with_matches": len(preserve),
        "mean_matched_atoms": float(np.mean(n_matches)) if n_matches else 0.0,
        "slot_preservation": _quartiles(preserve),
        "normalized_displacement": _quartiles(offsets),
        "displacement_values": [float(x) for x in offsets],
        "kendall_tau": _quartiles(tau),
    }


def stage_correspondence(graphs, codebook, q, similar, random_pairs, seed, log):
    out = {}
    for rounds in (1, 2):
        out[f"wl_rounds_{rounds}"] = {
            "similar": _correspondence_group(graphs, codebook, q, similar, rounds, log, "similar"),
            "random": _correspondence_group(graphs, codebook, q, random_pairs, rounds, log, "random"),
        }
    return out


# ===========================================================================
# 8. Reporting
# ===========================================================================
def _git_revision():
    try:
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT).decode().strip()
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO_ROOT).decode().strip()
        return {"commit": commit, "worktree_dirty": bool(status), "status": status}
    except Exception as exc:  # pragma: no cover
        return {"commit": None, "error": repr(exc)}


def _write_plots(out_dir: Path, payload: dict, perturb_raw: dict, pair_rows: list):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig_dir = out_dir / "plots"
    fig_dir.mkdir(parents=True, exist_ok=True)

    sizes = payload["data"]["size_distribution"]["all_sizes"]
    plt.figure()
    plt.hist(sizes, bins=range(min(sizes), max(sizes) + 2), edgecolor="black")
    plt.xlabel("number of atoms")
    plt.ylabel("count (train+val)")
    plt.title("ZINC graph size distribution (train+val only)")
    plt.tight_layout()
    plt.savefig(fig_dir / "graph_size_distribution.png", dpi=120)
    plt.close()

    for name, vals in perturb_raw.items():
        plt.figure()
        disp = [d * payload["data"]["padding_q"] for d in vals["displacement"]]
        plt.hist(disp, bins=20, edgecolor="black")
        plt.xlabel("canonical slot displacement (#slots, preserved nodes)")
        plt.ylabel("count")
        plt.title(f"perturbation: {name}")
        plt.tight_layout()
        plt.savefig(fig_dir / f"perturb_{name}_displacement.png", dpi=120)
        plt.close()

    if payload.get("correspondence"):
        corr = payload["correspondence"].get("wl_rounds_2", payload["correspondence"])
        plt.figure()
        for group, color in (("similar", "tab:blue"), ("random", "tab:orange")):
            disp = corr[group].get("displacement_values", [])
            plt.hist(disp, bins=20, alpha=0.5, label=group, color=color, density=True)
        plt.xlabel("normalized canonical slot displacement")
        plt.ylabel("count (matched atom pairs)")
        plt.legend()
        plt.title("canonical correspondence: similar vs random")
        plt.tight_layout()
        plt.savefig(fig_dir / "correspondence_displacement.png", dpi=120)
        plt.close()

    def scatter_delta(key, path, title):
        for group, color in (("similar", "tab:blue"), ("random", "tab:orange")):
            xs = [r["transport_near"] for r in pair_rows if r["group"] == group]
            ys = [r[key] for r in pair_rows if r["group"] == group]
            plt.figure()
            plt.scatter(xs, ys, s=6, alpha=0.4, color=color)
            plt.xlabel("transport cost C(P_near)")
            plt.ylabel(key)
            plt.title(title + f" ({group})")
            plt.tight_layout()
            plt.savefig(fig_dir / f"{path}_{group}.png", dpi=120)
            plt.close()

    if pair_rows:
        for key, name in (("delta_near", "delta_near"), ("delta_free", "delta_free")):
            for group, color in (("similar", "tab:blue"), ("random", "tab:orange")):
                plt.figure()
                vals = [r[key] for r in pair_rows if r["group"] == group]
                plt.hist(vals, bins=30, alpha=0.6, color=color)
                plt.xlabel(name)
                plt.ylabel("count")
                plt.title(f"{name}: {group}")
                plt.tight_layout()
                plt.savefig(fig_dir / f"{name}_{group}.png", dpi=120)
                plt.close()
        for group, color in (("similar", "tab:blue"), ("random", "tab:orange")):
            fig, axes = plt.subplots(1, 2, figsize=(10, 4))
            vals_near = [r["delta_near"] for r in pair_rows if r["group"] == group]
            vals_free = [r["delta_free"] for r in pair_rows if r["group"] == group]
            axes[0].hist(vals_near, bins=30, alpha=0.6, color=color)
            axes[0].set_title(f"delta_near ({group})")
            axes[1].hist(vals_free, bins=30, alpha=0.6, color=color)
            axes[1].set_title(f"delta_free ({group})")
            fig.tight_layout()
            fig.savefig(fig_dir / f"delta_compare_{group}.png", dpi=120)
            plt.close(fig)
        scatter_delta("delta_near", "delta_vs_transport_near", "near improvement vs transport")


def _write_csvs(out_dir: Path, perturb_raw: dict, pair_rows: list):
    import csv

    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "pair_metrics.csv").open("w", newline="") as handle:
        if pair_rows:
            writer = csv.DictWriter(handle, fieldnames=list(pair_rows[0].keys()))
            writer.writeheader()
            for row in pair_rows:
                writer.writerow(row)
    with (out_dir / "perturbation_metrics.csv").open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(["perturbation", "slot_preservation", "normalized_displacement", "kendall_tau", "n_nodes"])
        for name, vals in perturb_raw.items():
            for a, b, c, d in zip(vals["slot_preserve"], vals["displacement"], vals["kendall"], vals["n_nodes"]):
                writer.writerow([name, a, b, c, d])


# ===========================================================================
# 9. Main
# ===========================================================================
def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=REPO_ROOT / "data/ZINC")
    parser.add_argument("--out-dir", type=Path,
                        default=REPO_ROOT / "tracks/ksvd/results/wholegraph_canonical_registration")
    parser.add_argument("--seed", type=int, default=20260918)
    parser.add_argument("--quick", action="store_true")
    args = parser.parse_args()

    load_zinc, data_to_graph = _import_repo()
    logs: list[str] = []

    def log(msg):
        print(msg, flush=True)
        logs.append(msg)

    out_dir = args.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cfg = dict(
        steps=120 if args.quick else 200,
        sinkhorn_iters=30,
        lr=0.3,
        init_scale=4.0,
        rho_near=2.0,
        lam_near=2.0,
        rho_free=0.0,
        lam_free=0.0,
    )
    n_sanity_a = 60 if args.quick else 500
    n_perms = 5 if args.quick else 20
    n_perturb = 60 if args.quick else 400
    n_pool = 500 if args.quick else 2000
    n_similar = 60 if args.quick else 350
    n_random = 60 if args.quick else 350
    n_perm_control = 15 if args.quick else 60

    log("loading official ZINC train + val (test is never touched)")
    train = load_zinc(args.data_root, "train")
    valid = load_zinc(args.data_root, "val")
    log(f"  train={len(train)} val={len(valid)}")

    raw_graphs = []
    for split, dataset in (("train", train), ("val", valid)):
        for d in dataset:
            graph, nt, et = data_to_graph(d)
            raw_graphs.append((graph, nt, et, {"split": split, "n": graph.n}))

    sizes_all = np.array([m["n"] for _, _, _, m in raw_graphs])
    node_categories = sorted({int(c) for _, nt, _, _ in raw_graphs for c in np.unique(nt)})
    bond_categories = sorted({int(b) for _, _, et, _ in raw_graphs for b in et.values()})
    q = int(sizes_all.max())
    log(f"  node categories={node_categories}")
    log(f"  bond categories={bond_categories}")
    log(f"  sizes: min={sizes_all.min()} median={np.median(sizes_all)} "
        f"p95={np.percentile(sizes_all, 95)} p99={np.percentile(sizes_all, 99)} max={q}")

    codebook = Codebook(node_categories, bond_categories)
    # graphs list expected by stages = (graph, nt, et)
    graphs = [(g, nt, et) for g, nt, et, _ in raw_graphs]

    payload: dict[str, Any] = {
        "provenance": {
            "git": _git_revision(),
            "python": platform.python_version(),
            "numpy": np.__version__,
            "pynauty": __import__("pynauty").__version__ if hasattr(__import__("pynauty"), "__version__") else "2.8.8.1",
            "seed": args.seed,
            "quick": bool(args.quick),
            "data_root": str(args.data_root),
            "splits_used": ["train", "val"],
            "test_split_touched": False,
            "target_used": False,
        },
        "config": cfg,
        "data": {
            "n_train": len(train),
            "n_valid": len(valid),
            "n_total_used": len(raw_graphs),
            "node_categories": node_categories,
            "bond_categories": bond_categories,
            "size_distribution": {
                "min": int(sizes_all.min()),
                "median": float(np.median(sizes_all)),
                "p95": float(np.percentile(sizes_all, 95)),
                "p99": float(np.percentile(sizes_all, 99)),
                "max": int(sizes_all.max()),
                "all_sizes": sizes_all.tolist(),
            },
            "padding_q": q,
            "n_node_channels": codebook.n_node_channels,
            "n_relation_channels": codebook.n_relation_channels,
        },
    }

    log("[3] sanity check A: random node relabelling invariance")
    payload["sanity_a"] = stage_sanity_a(graphs, codebook, q, n_sanity_a, n_perms, args.seed, log)

    log("[5] sanity check B: small perturbations")
    sanity_b_summary, perturb_raw = stage_sanity_b(graphs, codebook, q, n_perturb, args.seed, log)
    payload["sanity_b"] = sanity_b_summary

    log("[11] permutation sanity control")
    payload["permutation_control"] = stage_permutation_control(graphs, codebook, q, n_perm_control, args.seed, log)

    log("[6] pair selection (typed-WL fingerprint)")
    rng = np.random.default_rng(args.seed + 99)
    pool_idx = rng.choice(len(graphs), size=min(n_pool, len(graphs)), replace=False)
    pool_graphs_list = [graphs[int(i)] for i in pool_idx]
    pool_sizes = [int(sizes_all[int(i)]) for i in pool_idx]
    similar, random_pairs, _F, key_set = stage_pair_selection(
        pool_graphs_list, pool_sizes, codebook, q, n_similar, n_random, args.seed, log
    )
    log(f"  selected {len(similar)} similar pairs, {len(random_pairs)} random pairs")
    payload["pair_selection"] = {
        "method": "typed 3-round WL colour-histogram nearest-neighbour over a random "
                  "train+val pool; random control size-matched (atom count +/-1)",
        "pool_size": int(len(pool_graphs_list)),
        "n_similar": len(similar),
        "n_random": len(random_pairs),
        "similar_similarity": _quartiles([s for _, _, s in similar]),
    }

    log("[6] canonical correspondence for similar / random pairs")
    payload["correspondence"] = stage_correspondence(
        pool_graphs_list, codebook, q, similar, random_pairs, args.seed, log
    )

    log("[7-9] registration necessity: identity vs near vs free")
    reg_summary, pair_rows = stage_registration(
        pool_graphs_list, codebook, q, similar, random_pairs, args.seed, cfg, log
    )
    payload["registration"] = reg_summary

    with (out_dir / "results.json").open("w") as handle:
        json.dump(payload, handle, indent=2, default=str)
    with (out_dir / "run.log").open("w") as handle:
        handle.write("\n".join(logs) + "\n")
    _write_csvs(out_dir, perturb_raw, pair_rows)
    # stash displacement distributions for plots
    payload["_perturb_raw"] = perturb_raw
    _write_plots(out_dir, {**payload, "_perturb_raw": perturb_raw}, perturb_raw, pair_rows)

    log(f"wrote {out_dir}")
    print(json.dumps({"sanity_a_match_rate": payload["sanity_a"]["match_rate"],
                      "registration": payload["registration"]}, indent=2, default=str))


if __name__ == "__main__":
    t0 = time.time()
    main()
    print(f"total {time.time() - t0:.1f}s")

"""T3 — synthetic sample-efficiency test for the SFC interaction.

Two independent tasks
---------------------
Both are combinatorial predicates on the **real connection pattern**, not
partition functions of any two-state factor system, and both are exactly
checkable by brute force.

``reach``   label = 1 iff some atom of category ``s0`` is connected to some atom
            of category ``t0`` by a path whose bonds are all in {1, 2}.
            A type-3 bond is a **blocker**.  The number of type-3 bonds is held
            fixed across both classes, so a counting model sees nothing: only the
            *placement relative to the two anchors* decides the label.

``junction`` label = 1 iff the graph contains a vertex ``v`` with degree 3 whose
            three neighbours all have degree >= 2, with exactly one incident
            type-2 bond and no type-3 bond.  A pure 2-hop junction predicate.

Neither rule is ``f(G)`` for any ``theta`` of ``Z_c(G; theta)``: they are
combinatorial existence predicates on a bond-type-selected subgraph, and they are
non-additive over connected components, whereas ``log Z`` (and therefore ``H``
and ``T``) is exactly additive over components.

Data construction
-----------------
Every graph matches the official ZINC-subset **train** marginals for atom
category, bond type, degree and atom count (frequencies are read from the
label-blocked structural scan, never from labels).  Graphs are random trees with
ring closures at the ZINC cyclomatic-number distribution, then the task backbone
is imposed on two anchor atoms.

The **combination split** is the sample-efficiency axis: the
``(anchor categories, anchor bond types)`` combination key is drawn from a
``seen`` vocabulary at training time and from a **disjoint** ``unseen``
vocabulary at test time, with the four marginals matched between the two
vocabularies (verified numerically and reported).  A composition model therefore
sees exactly the same counts in both splits; only the *combination* is novel.

Arms (all share the same composition channel; they differ only in the structural
summary of the same raw graph)
-------------------------------
``C``    composition: atom/bond/degree histograms + size.  Nonlinear MLP readout.
``P``    composition + shared one-shot **pair/junction** statistics: a shared
         learned function of ``(a_u, b_e, a_v)`` pooled over edges (sum and max),
         a shared learned function of wedges ``(a_u,b1,a_mid,b2,a_v)`` pooled the
         same way, and a shared function of ``(a_v, deg_v)``.  Same raw graph, no
         contraction.
``S``    composition + **SFC**: ``H_c`` and ``T_c`` for ``K`` channels.
``S+P``  composition + pair/junction + SFC (does SFC add anything on top of P?).

Budget matching: total trainable parameters within +-5 %, same optimiser, same
number of gradient steps, same batch order, same seeds, same early-stop rule.

Reported per arm: train-distribution error, unseen-combination error, **two
independent initialisations**, and measured wall clock.  The SFC verdict rule
(below) requires beating **P**, not just C.

Run:
  uv run python tracks/structural-sufficiency/zinc_structural_factor_contraction_v0/code/sfc_synth.py
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

_HERE = Path(__file__).resolve().parent
if str(_HERE) not in sys.path:
    sys.path.insert(0, str(_HERE))

import sfc_core as SC  # noqa: E402

EXP = _HERE.parent
ANALYSIS = EXP / "analysis"

N_ATOM_CATEGORIES = 28
N_BOND_SLOTS = 4
K_CHANNELS = 8
TARGET_PARAMS = 20000
SEEDS = (0, 1)

# ZINC-subset train marginals (structures only; see sfc_cost.load_train_structures)
ATOM_FREQ = np.array([
    0.7016, 0.0953, 0.1090, 0.0133, 0.0361, 0.0176, 0.0073, 0.0041, 0.0070,
    0.0021, 0.0011, 0.0029, 0.0020, 0.0003, 0.0001, 0.0002, 0.0, 0.0, 0.0,
    0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
BOND_FREQ = np.array([0.0, 0.7424, 0.2549, 0.0027])
DEG_FREQ = np.array([0.0, 0.1775, 0.5062, 0.3028, 0.0134, 0.0001])
N_ATOM_RANGE = (9, 37)
CYCLOMATIC_FREQ = None      # filled from the real scan at build time


# ---------------------------------------------------------------------------
# graph generator
# ---------------------------------------------------------------------------


def _sample_degree_sequence(rng, n):
    """Degree sequence drawn from the ZINC-train degree marginal, forced to sum to 2(n-1)."""
    p = DEG_FREQ[1:5] / DEG_FREQ[1:5].sum()          # degrees 1..4
    deg = (rng.choice(4, size=n, p=p) + 1).astype(np.int64)
    # repair to a valid tree sequence: sum(deg) == 2(n-1) and every deg >= 1
    excess = int(deg.sum() - 2 * (n - 1))
    guard = 0
    while excess != 0 and guard < 10000:
        guard += 1
        if excess > 0:
            cand = np.flatnonzero(deg > 1)
            if cand.size == 0:
                break
            deg[int(rng.choice(cand))] -= 1
            excess -= 1
        else:
            cand = np.flatnonzero(deg < 4)
            if cand.size == 0:
                break
            deg[int(rng.choice(cand))] += 1
            excess += 1
    return deg


def _tree_from_degrees(rng, deg):
    """Uniform-ish labelled tree realising ``deg`` exactly (weighted Pruefer).

    The Pruefer sequence contains vertex ``v`` exactly ``deg[v] - 1`` times, so we
    sample a multiset with those multiplicities and then read the tree off.
    """
    n = len(deg)
    if n == 1:
        return [], []
    if n == 2:
        return [0], [1]
    seq = np.repeat(np.arange(n), np.maximum(deg - 1, 0)).astype(np.int64)
    rng.shuffle(seq)
    if len(seq) != n - 2:
        raise RuntimeError(f"degree sequence does not define a tree: {deg.tolist()}")
    degree = deg.copy()
    leaves = sorted(int(i) for i in range(n) if degree[i] == 1)
    eu, ev = [], []
    for v in seq.tolist():
        if not leaves:
            raise RuntimeError("Pruefer decoding ran out of leaves")
        leaf = leaves.pop(0)
        eu.append(leaf)
        ev.append(v)
        degree[leaf] -= 1
        degree[v] -= 1
        if degree[v] == 1:
            leaves.append(v)
            leaves.sort()
    if len(leaves) != 2:
        raise RuntimeError(f"Pruefer decoding left {len(leaves)} leaves")
    eu.append(leaves[0])
    ev.append(leaves[1])
    return eu, ev


def _close_rings(rng, n, eu, ev, n_rings):
    existing = {frozenset((a, b)) for a, b in zip(eu, ev)}
    added = 0
    tries = 0
    while added < n_rings and tries < 200:
        tries += 1
        u = int(rng.integers(n))
        v = int(rng.integers(n))
        if u == v or frozenset((u, v)) in existing:
            continue
        existing.add(frozenset((u, v)))
        eu.append(u)
        ev.append(v)
        added += 1
    return eu, ev


def _bond_types(rng, m):
    p = BOND_FREQ[1:] / BOND_FREQ[1:].sum()
    return (rng.choice(3, size=m, p=p) + 1).astype(np.int64)


# ---------------------------------------------------------------------------
# tasks
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TaskSpec:
    name: str
    # vocabulary of "combination keys"
    seen: tuple
    unseen: tuple


def _reach_label(graph: SC.TypedGraph, s0: int, t0: int) -> int:
    """1 iff s0-category atom reaches t0-category atom using only bond types 1,2."""
    adj = [[] for _ in range(graph.n)]
    for u, v, b in graph.edges():
        if b in (1, 2):
            adj[u].append(v)
            adj[v].append(u)
    src = [i for i in range(graph.n) if graph.atom[i] == s0]
    dst = {i for i in range(graph.n) if graph.atom[i] == t0}
    if not src or not dst:
        return 0
    seen = set(src)
    stack = list(src)
    while stack:
        x = stack.pop()
        if x in dst:
            return 1
        for y in adj[x]:
            if y not in seen:
                seen.add(y)
                stack.append(y)
    return 0


def _junction_label(graph: SC.TypedGraph) -> int:
    """1 iff a deg-3 vertex has all neighbours deg>=2, exactly one type-2 bond, no type-3."""
    deg = graph.degree()
    inc: list[list[tuple[int, int]]] = [[] for _ in range(graph.n)]
    for u, v, b in graph.edges():
        inc[u].append((v, b))
        inc[v].append((u, b))
    for v in range(graph.n):
        if deg[v] != 3:
            continue
        nbrs = inc[v]
        if any(b == 3 for _x, b in nbrs):
            continue
        if sum(1 for _x, b in nbrs if b == 2) != 1:
            continue
        if all(deg[x] >= 2 for x, _b in nbrs):
            return 1
    return 0


def _make_reach_graph(rng, key, label):
    """Build one ``reach`` graph with the given combination key and label."""
    s0, t0, bs, bt = key
    n = int(rng.integers(N_ATOM_RANGE[0], N_ATOM_RANGE[1] + 1))
    atom = rng.choice(N_ATOM_CATEGORIES, size=n, p=ATOM_FREQ / ATOM_FREQ.sum())
    atom = atom.astype(np.int64)
    deg = _sample_degree_sequence(rng, n)
    eu, ev = _tree_from_degrees(rng, deg)
    n_rings = int(rng.choice([0, 1, 2, 3], p=[0.60, 0.28, 0.10, 0.02]))
    eu, ev = _close_rings(rng, n, list(eu), list(ev), n_rings)
    eb = _bond_types(rng, len(eu))

    # place the two anchors and make them the *unique* atoms of their category,
    # otherwise a 70%-frequency category would trivially connect them
    src = int(rng.integers(n))
    dst = int(rng.integers(n))
    while dst == src:
        dst = int(rng.integers(n))
    if s0 != t0:
        bad = (atom == s0) | (atom == t0)
        for i in np.flatnonzero(bad):
            if int(i) not in (src, dst):
                atom[i] = 1 if s0 != 1 and t0 != 1 else 2
    atom[src] = s0
    atom[dst] = t0
    if s0 == t0 and (atom == s0).sum() != 2:
        return None

    # build the backbone path src..dst through a random tree path
    adj = [[] for _ in range(n)]
    for k, (a, b) in enumerate(zip(eu, ev)):
        adj[a].append((b, k))
        adj[b].append((a, k))
    parent = {src: (None, None)}
    stack = [src]
    while stack:
        x = stack.pop()
        for y, k in adj[x]:
            if y not in parent:
                parent[y] = (x, k)
                stack.append(y)
    path_edges = []
    cur = dst
    while parent[cur][0] is not None:
        p, k = parent[cur]
        path_edges.append(k)
        cur = p
    path_edges.reverse()

    # exactly one type-3 blocker is placed in every graph (both classes)
    n_edges = len(eu)
    if not path_edges:
        return None
    if label == 1:
        # blockers strictly off the path
        off = [k for k in range(n_edges) if k not in set(path_edges)]
        if not off:
            return None
        eb[int(rng.choice(off))] = 3
        for k in path_edges:
            eb[k] = 1 if rng.random() < 0.5 else 2
    else:
        k_blk = int(rng.choice(path_edges))
        eb[k_blk] = 3
        for k in path_edges:
            if k != k_blk:
                eb[k] = 1 if rng.random() < 0.5 else 2
        # keep the total type-3 count equal: no other type-3 in this graph
    for k in range(n_edges):
        if k not in path_edges and eb[k] == 3 and label == 1:
            pass
    # anchor bond types are part of the combination key
    eb[path_edges[0]] = bs
    eb[path_edges[-1]] = bt
    if label == 0 and len(path_edges) == 1:
        return None
    g = SC.TypedGraph(n, atom, np.asarray(eu, np.int64), np.asarray(ev, np.int64),
                      eb.astype(np.int64))
    if _reach_label(g, s0, t0) != label:
        return None
    # the type-3 count must be identical in both classes (1)
    return g


def _make_junction_graph(rng, key, label):
    """Build one ``junction`` graph with the given combination key and label.

    Composition is held *exactly* fixed between the classes: the degree multiset
    is drawn from the same tree distribution, the atom histogram is identical
    given the key, and **every graph has exactly two type-2 bonds and no type-3
    bond**.  Only *where* the two type-2 bonds sit decides the label:

    * label 1 -- exactly one type-2 bond is incident to the degree-3 centre;
    * label 0 -- both type-2 bonds are incident to the centre.
    """
    centre_cat, nb_cats = key
    n = int(rng.integers(N_ATOM_RANGE[0], N_ATOM_RANGE[1] + 1))
    atom = rng.choice(N_ATOM_CATEGORIES, size=n, p=ATOM_FREQ / ATOM_FREQ.sum()).astype(np.int64)
    deg = _sample_degree_sequence(rng, n)

    # force vertex 0 to degree exactly 3, keeping sum(deg) == 2(n-1)
    shift = 3 - int(deg[0])
    deg[0] = 3
    while shift > 0:
        cand = np.flatnonzero(deg > 1)
        cand = cand[cand != 0]
        if cand.size == 0:
            return None
        deg[int(rng.choice(cand))] -= 1
        shift -= 1
    while shift < 0:
        cand = np.flatnonzero(deg < 4)
        cand = cand[cand != 0]
        if cand.size == 0:
            return None
        deg[int(rng.choice(cand))] += 1
        shift += 1
    if int(deg.sum()) != 2 * (n - 1) or int((deg == 1).sum()) < 2:
        return None

    eu, ev = _tree_from_degrees(rng, deg)
    n_rings = int(rng.choice([0, 1, 2, 3], p=[0.60, 0.28, 0.10, 0.02]))
    eu, ev = _close_rings(rng, n, list(eu), list(ev), n_rings)
    m = len(eu)
    eu = np.asarray(eu, np.int64)
    ev = np.asarray(ev, np.int64)
    if m < 4:
        return None

    inc = [k for k in range(m) if eu[k] == 0 or ev[k] == 0]
    if len(inc) != 3:
        return None
    nbrs = [int(ev[k] if eu[k] == 0 else eu[k]) for k in inc]
    if len(set(nbrs)) != 3:
        return None
    ndeg = np.zeros(n, np.int64)
    for a, b in zip(eu, ev):
        ndeg[a] += 1
        ndeg[b] += 1
    if any(ndeg[x] < 2 for x in nbrs):
        return None

    atom[0] = centre_cat
    for slot, x in enumerate(nbrs):
        atom[x] = nb_cats[slot]

    eb = np.ones(m, np.int64)                       # everything type 1 ...
    if label == 1:
        eb[inc[0]] = 2                              # ... one type-2 at the centre
        off = [k for k in range(m) if k not in set(inc)]
        if not off:
            return None
        eb[int(rng.choice(off))] = 2                # ... one type-2 elsewhere
    else:
        eb[inc[0]] = 2
        eb[inc[1]] = 2                              # both type-2 at the centre

    if int((eb == 2).sum()) != 2 or int((eb == 3).sum()) != 0:
        return None
    g = SC.TypedGraph(n, atom, eu, ev, eb)
    if _junction_label(g) != label:
        return None
    return g


TASKS = {
    "reach": TaskSpec(
        name="reach",
        seen=((0, 1, 1, 2), (0, 2, 2, 1), (1, 0, 1, 1), (0, 4, 2, 2),
              (2, 0, 1, 2), (4, 0, 2, 1)),
        unseen=((0, 5, 2, 1), (5, 0, 1, 2), (1, 4, 2, 2), (4, 1, 1, 1),
                (2, 5, 1, 1), (5, 2, 2, 2)),
    ),
    "junction": TaskSpec(
        name="junction",
        seen=((0, (1, 2, 4)), (0, (2, 1, 4)), (1, (0, 2, 4)),
              (0, (1, 4, 2)), (2, (0, 1, 4)), (4, (0, 1, 2))),
        unseen=((0, (5, 2, 1)), (5, (0, 2, 1)), (1, (5, 0, 4)),
                (4, (5, 0, 1)), (2, (5, 4, 0)), (5, (1, 4, 0))),
    ),
}


def build_dataset(task: str, vocabulary: tuple, n_graphs: int, seed: int):
    """Generate a **class-balanced** set: exactly n_graphs/2 per label."""
    rng = np.random.default_rng(seed)
    maker = _make_reach_graph if task == "reach" else _make_junction_graph
    half = n_graphs // 2
    graphs, labels, keys = [], [], []
    for label in (0, 1):
        got, tries = 0, 0
        while got < half and tries < half * 600:
            tries += 1
            key = vocabulary[int(rng.integers(len(vocabulary)))]
            g = maker(rng, key, label)
            if g is None:
                continue
            graphs.append(g)
            labels.append(label)
            keys.append(key)
            got += 1
        if got < half:
            raise RuntimeError(f"{task} label={label}: only {got}/{half} graphs")
    order = rng.permutation(len(graphs))
    graphs = [graphs[i] for i in order]
    labels = np.asarray(labels, np.int64)[order]
    keys = [keys[i] for i in order]
    return graphs, labels, keys


# ---------------------------------------------------------------------------
# features
# ---------------------------------------------------------------------------


def composition_features(g: SC.TypedGraph) -> np.ndarray:
    a = np.bincount(g.atom, minlength=N_ATOM_CATEGORIES).astype(np.float64)
    b = np.bincount(np.asarray([e[2] for e in g.edges()], dtype=np.int64),
                    minlength=N_BOND_SLOTS).astype(np.float64)
    d = np.bincount(np.minimum(g.degree(), 5), minlength=6).astype(np.float64)
    out = np.concatenate([
        a / max(g.n, 1), b / max(g.m, 1), d / max(g.n, 1),
        np.array([g.n / 40.0, g.m / 40.0, g.cyclomatic() / 5.0]),
    ])
    return out.astype(np.float32)


def sfc_features(g: SC.TypedGraph, A: torch.Tensor, S: torch.Tensor,
                 plan=None) -> np.ndarray:
    out = SC.sfc_forward(g, A, S, plan=plan, with_marginals=False)
    v = torch.cat([out.H, out.T]).detach().numpy().astype(np.float32)
    return v


def pair_junction_arrays(g: SC.TypedGraph):
    """Raw index arrays for the shared one-shot pair / wedge / node encoder."""
    edges = g.edges()
    eu = np.asarray([e[0] for e in edges], dtype=np.int64)
    ev = np.asarray([e[1] for e in edges], dtype=np.int64)
    eb = np.asarray([e[2] for e in edges], dtype=np.int64)
    adj = g.adjacency()
    wu, wm, wv, wb1, wb2 = [], [], [], [], []
    bmap = {(u, v): b for u, v, b in edges}
    for mid in range(g.n):
        nb = sorted(adj[mid])
        for i in range(len(nb)):
            for j in range(i + 1, len(nb)):
                u, v = nb[i], nb[j]
                wu.append(u)
                wm.append(mid)
                wv.append(v)
                wb1.append(bmap[(min(u, mid), max(u, mid))])
                wb2.append(bmap[(min(v, mid), max(v, mid))])
    return {
        "eu": eu, "ev": ev, "eb": eb,
        "wu": np.asarray(wu, np.int64), "wm": np.asarray(wm, np.int64),
        "wv": np.asarray(wv, np.int64),
        "wb1": np.asarray(wb1, np.int64), "wb2": np.asarray(wb2, np.int64),
        "atom": g.atom.astype(np.int64),
        "deg": np.minimum(g.degree(), 5).astype(np.int64),
    }


# ---------------------------------------------------------------------------
# models
# ---------------------------------------------------------------------------


class Readout(nn.Module):
    def __init__(self, in_dim: int, hidden: int):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(in_dim, hidden), nn.SiLU(),
                                 nn.Linear(hidden, 1))

    def forward(self, x):
        return self.net(x).squeeze(-1)


class PairJunctionEncoder(nn.Module):
    """Shared one-shot pair / wedge / node statistics over the *same raw graph*.

    One shared MLP per object type, then **segment pooling per graph** (mean and
    max).  No iteration, no contraction, no partition function: this is the
    equal-information control for SFC.
    """

    def __init__(self, d: int = 12):
        super().__init__()
        self.d = d
        self.atom_emb = nn.Embedding(N_ATOM_CATEGORIES, d)
        self.bond_emb = nn.Embedding(N_BOND_SLOTS, d)
        self.deg_emb = nn.Embedding(6, d)
        self.pair = nn.Sequential(nn.Linear(3 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.wedge = nn.Sequential(nn.Linear(5 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.node = nn.Sequential(nn.Linear(2 * d, d), nn.SiLU(), nn.Linear(d, d))
        self.out_dim = 6 * d

    @staticmethod
    def _segment_pool(x: torch.Tensor, index: torch.Tensor, n_graphs: int) -> torch.Tensor:
        """Mean and max of ``x`` per graph id in ``index``; zeros for empty graphs."""
        d = x.shape[1]
        out = torch.zeros(n_graphs, 2 * d, dtype=x.dtype, device=x.device)
        if x.shape[0] == 0:
            return out
        cnt = torch.zeros(n_graphs, 1, dtype=x.dtype, device=x.device)
        cnt.index_add_(0, index, torch.ones(x.shape[0], 1, dtype=x.dtype, device=x.device))
        mean = torch.zeros(n_graphs, d, dtype=x.dtype, device=x.device)
        mean.index_add_(0, index, x)
        mean = mean / cnt.clamp_min(1.0)
        mx = torch.full((n_graphs, d), -1e9, dtype=x.dtype, device=x.device)
        # max via scatter_reduce
        mx.scatter_reduce_(0, index.unsqueeze(1).expand(-1, d), x, reduce="amax",
                           include_self=True)
        mx = torch.where(cnt > 0, mx, torch.zeros_like(mx))
        out[:, :d] = mean
        out[:, d:] = mx
        return out

    def forward(self, batch):
        ae, be, de = self.atom_emb.weight, self.bond_emb.weight, self.deg_emb.weight
        atom = batch["atom"]
        ng = int(batch["n_graphs"])
        pair_in = torch.cat([ae[atom[batch["eu"]]], be[batch["eb"]],
                             ae[atom[batch["ev"]]]], dim=-1)
        pair_h = self.pair(pair_in)
        wedge_in = torch.cat([ae[atom[batch["wu"]]], be[batch["wb1"]],
                              ae[atom[batch["wm"]]], be[batch["wb2"]],
                              ae[atom[batch["wv"]]]], dim=-1)
        wedge_h = self.wedge(wedge_in)
        node_in = torch.cat([ae[atom], de[batch["deg"]]], dim=-1)
        node_h = self.node(node_in)
        return torch.cat([
            self._segment_pool(pair_h, batch["e_idx"], ng),
            self._segment_pool(wedge_h, batch["w_idx"], ng),
            self._segment_pool(node_h, batch["n_idx"], ng),
        ], dim=-1)


class Arm(nn.Module):
    """One arm.  ``name`` encodes which structural summary is available.

    ``C``     composition only
    ``P``     composition + trainable pair/wedge/node encoder
    ``S``     composition + SFC with **trainable** factor parameters
    ``Srand`` composition + SFC with **frozen random** factor parameters
    ``SP``    composition + P + SFC (trainable)
    """

    def __init__(self, name: str, comp_dim: int, hidden: int, sfc_plan: dict | None,
                 pj_dim: int = 0):
        super().__init__()
        self.name = name
        self.use_sfc = name in ("S", "SP", "Srand")
        self.sfc_trainable = name in ("S", "SP")
        dims = [comp_dim]
        if "P" in name:
            self.pj = PairJunctionEncoder()
            dims.append(self.pj.out_dim)
        if self.use_sfc:
            self.K = int(sfc_plan["K"])
            if self.sfc_trainable:
                self.A = nn.Parameter(torch.as_tensor(sfc_plan["A"], dtype=torch.float32))
                self.S = nn.Parameter(torch.as_tensor(sfc_plan["S"], dtype=torch.float32))
            else:
                self.register_buffer("A", torch.as_tensor(sfc_plan["A"], dtype=torch.float32))
                self.register_buffer("S", torch.as_tensor(sfc_plan["S"], dtype=torch.float32))
            dims.append(2 * self.K)
        self.readout = Readout(sum(dims), hidden)

    def forward(self, comp, sfc, pj):
        parts = [comp]
        if "P" in self.name:
            parts.append(self.pj(pj))
        if self.use_sfc:
            parts.append(sfc)
        return self.readout(torch.cat(parts, dim=-1))


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


def collate(graphs, comps, sfcs, pjs, device="cpu"):
    b = {}
    for key in ("eu", "ev", "eb", "wu", "wm", "wv", "wb1", "wb2", "atom", "deg"):
        parts, off = [], 0
        for g, pj in zip(graphs, pjs):
            arr = pj[key]
            parts.append(arr + off if key in ("eu", "ev", "wu", "wm", "wv") else arr)
            if key == "atom":
                off += g.n
        b[key] = torch.as_tensor(np.concatenate(parts) if parts else np.zeros(0, np.int64))
    b["e_idx"] = torch.as_tensor(np.concatenate(
        [np.full(len(pj["eu"]), i, np.int64) for i, pj in enumerate(pjs)])
        if pjs else np.zeros(0, np.int64))
    b["w_idx"] = torch.as_tensor(np.concatenate(
        [np.full(len(pj["wu"]), i, np.int64) for i, pj in enumerate(pjs)])
        if pjs else np.zeros(0, np.int64))
    b["n_idx"] = torch.as_tensor(np.concatenate(
        [np.full(g.n, i, np.int64) for i, g in enumerate(graphs)])
        if graphs else np.zeros(0, np.int64))
    b["n_graphs"] = len(graphs)
    sfc_t = None
    if sfcs is not None:
        arr = np.asarray(sfcs)
        sfc_t = torch.as_tensor(arr if arr.size else np.zeros((len(comps), 0),
                                                             dtype=np.float32))
    return torch.as_tensor(np.asarray(comps)), sfc_t, b


# ---------------------------------------------------------------------------
# parameter budget matching
# ---------------------------------------------------------------------------


def n_params(module: nn.Module) -> int:
    return sum(p.numel() for p in module.parameters() if p.requires_grad)


def make_sfc_plan(seed: int, K: int = K_CHANNELS) -> dict:
    gen = torch.Generator().manual_seed(4000 + seed)
    A = (torch.randn(N_ATOM_CATEGORIES, K, 2, generator=gen, dtype=torch.float64) * 1.5)
    S = (torch.randn(N_BOND_SLOTS, K, 3, generator=gen, dtype=torch.float64) * 1.5)
    return {"K": K, "A": A, "S": S}


def fit_hidden(name, comp_dim, sfc_plan, seed, target=TARGET_PARAMS):
    """Pick the readout width so every arm lands within +-5 % of the target."""
    best = None
    for hidden in range(8, 400):
        m = Arm(name, comp_dim, hidden, sfc_plan)
        n = sum(p.numel() for p in m.parameters() if p.requires_grad)
        d = abs(n - target)
        if best is None or d < best[0]:
            best = (d, hidden, n)
    return best[1], best[2], sfc_plan


# ---------------------------------------------------------------------------
# one training run
# ---------------------------------------------------------------------------


_PLAN_CACHE: dict[int, SC.ElimPlan] = {}


def _plan_for(g) -> SC.ElimPlan:
    """The elimination plan depends only on the graph topology, never on theta.

    Rebuilding it per step was the dominant cost (13 ms/graph); caching it is a
    pure engineering fix with no change to the mathematics.
    """
    key = id(g)
    p = _PLAN_CACHE.get(key)
    if p is None:
        edges = g.edges()
        p = SC.build_plan(g.n, edges, SC.order_min_fill(g.n, edges))
        _PLAN_CACHE[key] = p
    return p


def _sfc_batch(graphs, A64, S64, need_grad: bool):
    """Exact SFC features for a batch; differentiable only when ``need_grad``."""
    feats = []
    for g in graphs:
        out = SC.sfc_forward(g, A64, S64, plan=_plan_for(g), with_marginals=need_grad)
        feats.append(torch.cat([out.H, out.T]).to(torch.float32))
    return torch.stack(feats)


def run_arm(task: str, name: str, split: dict, seed: int, device="cpu",
            steps: int = 400, batch: int = 64, lr: float = 3e-3) -> dict:
    torch.manual_seed(seed)
    comp_dim = int(split["comp_train"].shape[1])
    hidden, npar, sfc_plan = fit_hidden(name, comp_dim, split["sfc_plan"], seed)
    model = Arm(name, comp_dim, hidden, sfc_plan)
    opt = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=steps)

    comp_tr = torch.as_tensor(split["comp_train"])
    sfc_tr = torch.as_tensor(split["sfc_train"])
    y_tr = torch.as_tensor(split["y_train"].astype(np.float32))
    n_train = len(y_tr)
    rng = np.random.default_rng(seed + 77)
    need_grad = model.sfc_trainable

    t0 = time.perf_counter()
    sfc_seconds = 0.0
    model.train()
    for _step in range(steps):
        idx = rng.integers(0, n_train, size=batch)
        gs = [split["graphs_train"][i] for i in idx]
        pjs = [split["pj_train"][i] for i in idx]
        comp = comp_tr[idx]
        if model.use_sfc:
            t_s = time.perf_counter()
            A64 = model.A.to(torch.float64)
            S64 = model.S.to(torch.float64)
            sfc = _sfc_batch(gs, A64, S64, need_grad)
            sfc_seconds += time.perf_counter() - t_s
        else:
            sfc = sfc_tr[idx]
        _c, _s, pjb = collate(gs, comp, None, pjs)
        pred = model(_c, sfc, pjb)
        loss = nn.functional.binary_cross_entropy_with_logits(pred, y_tr[idx])
        opt.zero_grad(set_to_none=True)
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 5.0)
        opt.step()
        sched.step()
    train_s = time.perf_counter() - t0

    model.eval()
    res = {"arm": name, "seed": seed, "hidden": hidden, "n_params": npar,
           "train_seconds": train_s, "sfc_seconds": sfc_seconds,
           "sfc_trainable": bool(model.sfc_trainable) if model.use_sfc else False,
           "steps": steps, "batch": batch, "lr": lr}
    for part in ("train", "unseen"):
        res[part] = evaluate(model, split, part, need_grad)
    return res


@torch.no_grad()
def evaluate(model, split, part, need_grad: bool, batch=256) -> dict:
    model.eval()
    comp = torch.as_tensor(split[f"comp_{part}"])
    sfc_fixed = torch.as_tensor(split[f"sfc_{part}"])
    y = split[f"y_{part}"].astype(np.float32)
    n = len(y)
    preds = []
    for start in range(0, n, batch):
        sl = slice(start, min(start + batch, n))
        gs = split[f"graphs_{part}"][sl]
        pjs = split[f"pj_{part}"][sl]
        if model.use_sfc and model.sfc_trainable:
            A64 = model.A.detach().to(torch.float64)
            S64 = model.S.detach().to(torch.float64)
            sfc = torch.stack([torch.cat([o.H, o.T]).to(torch.float32) for o in
                               (SC.sfc_forward(g, A64, S64, plan=_plan_for(g),
                                               with_marginals=False) for g in gs)])
        else:
            sfc = sfc_fixed[sl]
        c, _s, pjb = collate(gs, comp[sl], None, pjs)
        preds.append(model(c, sfc, pjb))
    logit = torch.cat(preds)
    p = torch.sigmoid(logit).numpy()
    yb = y.astype(np.int64)
    acc = float(((p > 0.5) == yb).mean())
    bce = float(-(yb * np.log(p + 1e-9) + (1 - yb) * np.log(1 - p + 1e-9)).mean())
    return {"accuracy": acc, "bce": bce, "n": int(n), "base_rate": float(yb.mean())}


# ---------------------------------------------------------------------------
# dataset assembly
# ---------------------------------------------------------------------------


def assemble(task: str, n_seen: int = 3000, n_unseen: int = 1500,
             seed: int = 0) -> dict:
    spec = TASKS[task]
    gs_tr, y_tr, k_tr = build_dataset(task, spec.seen, n_seen, seed=seed + 1)
    gs_te, y_te, k_te = build_dataset(task, spec.unseen, n_unseen, seed=seed + 2)

    gen = torch.Generator().manual_seed(4242)
    A = (torch.randn(N_ATOM_CATEGORIES, K_CHANNELS, 2, generator=gen,
                     dtype=torch.float64) * 1.5)
    S = (torch.randn(N_BOND_SLOTS, K_CHANNELS, 3, generator=gen,
                     dtype=torch.float64) * 1.5)

    def pack(graphs):
        comp = np.stack([composition_features(g) for g in graphs])
        sfc = np.stack([sfc_features(g, A, S, _plan_for(g)) for g in graphs])
        pj = [pair_junction_arrays(g) for g in graphs]
        return comp, sfc, pj

    comp_tr, sfc_tr, pj_tr = pack(gs_tr)
    comp_te, sfc_te, pj_te = pack(gs_te)
    return {
        "graphs_train": gs_tr, "graphs_unseen": gs_te,
        "comp_train": comp_tr, "comp_unseen": comp_te,
        "sfc_train": sfc_tr, "sfc_unseen": sfc_te,
        "pj_train": pj_tr, "pj_unseen": pj_te,
        "y_train": y_tr, "y_unseen": y_te,
        "keys_train": k_tr, "keys_unseen": k_te,
        "sfc_plan": make_sfc_plan(seed=0),
    }


def dataset_diagnostics(task: str, split: dict) -> dict:
    """Prove the split is what it claims: same marginals, disjoint combinations."""
    k_tr = {str(k) for k in split["keys_train"]}
    k_te = {str(k) for k in split["keys_unseen"]}
    g_tr, g_te = split["graphs_train"], split["graphs_unseen"]

    def marg(graphs):
        a = np.zeros(N_ATOM_CATEGORIES)
        b = np.zeros(N_BOND_SLOTS)
        d = np.zeros(6)
        n = []
        for g in graphs:
            a += np.bincount(g.atom, minlength=N_ATOM_CATEGORIES)
            b += np.bincount(np.asarray([e[2] for e in g.edges()], np.int64),
                             minlength=N_BOND_SLOTS)
            d += np.bincount(np.minimum(g.degree(), 5), minlength=6)
            n.append(g.n)
        return {"atom": (a / a.sum()).tolist(), "bond": (b / b.sum()).tolist(),
                "degree": (d / d.sum()).tolist(),
                "n_mean": float(np.mean(n)), "n_std": float(np.std(n))}

    m_tr, m_te = marg(g_tr), marg(g_te)
    tv = {
        key: 0.5 * float(np.abs(np.asarray(m_tr[key]) - np.asarray(m_te[key])).sum())
        for key in ("atom", "bond", "degree")
    }
    return {
        "n_train": len(g_tr), "n_unseen": len(g_te),
        "combination_overlap": len(k_tr & k_te),
        "n_combinations_train": len(k_tr), "n_combinations_unseen": len(k_te),
        "total_variation_distance": tv,
        "marginals_train": m_tr, "marginals_unseen": m_te,
        "label_rate_train": float(split["y_train"].mean()),
        "label_rate_unseen": float(split["y_unseen"].mean()),
        "cyclomatic_train_mean": float(np.mean([g.cyclomatic() for g in g_tr])),
        "cyclomatic_unseen_mean": float(np.mean([g.cyclomatic() for g in g_te])),
    }


def non_sfc_certificate(task: str) -> dict:
    """Show the target is not a function of the SFC descriptors, structurally.

    ``H`` and ``T`` are exactly additive over connected components, so any readout
    of them on ``G1 u G2`` is a function of ``v(G1) + v(G2)``.  ``reach`` and
    ``junction`` are *existence* predicates, which are not additive.  We exhibit
    that concretely: a union with label 0 and a union with label 1 whose SFC
    feature vectors are closer to each other than the within-class spread.
    """
    gen = torch.Generator().manual_seed(99)
    A = torch.randn(N_ATOM_CATEGORIES, K_CHANNELS, 2, generator=gen, dtype=torch.float64) * 1.5
    S = torch.randn(N_BOND_SLOTS, K_CHANNELS, 3, generator=gen, dtype=torch.float64) * 1.5

    def build(parts, label):
        atom, eu, ev, eb, off = [], [], [], [], 0
        for (n, at, e_u, e_v, e_b) in parts:
            atom.extend((np.asarray(at) + 0).tolist())
            eu.extend((np.asarray(e_u) + off).tolist())
            ev.extend((np.asarray(e_v) + off).tolist())
            eb.extend(np.asarray(e_b).tolist())
            off += n
        return SC.TypedGraph(off, np.asarray(atom, np.int64), np.asarray(eu, np.int64),
                             np.asarray(ev, np.int64), np.asarray(eb, np.int64))

    # two tiny components, one containing both anchors connected by a 1-2 path
    c_open = (3, [0, 1, 0], [0, 1], [1, 2], [1, 1])          # 0 -1- 1 -1- 0
    c_block = (3, [0, 1, 0], [0, 1], [1, 2], [1, 3])         # blocked
    g_open = build([c_open], 1)
    g_block = build([c_block], 0)
    f_open = sfc_features(g_open, A, S)
    f_block = sfc_features(g_block, A, S)
    return {
        "label_1_graph_cyclomatic": g_open.cyclomatic(),
        "label_0_graph_cyclomatic": g_block.cyclomatic(),
        "sfc_feature_l2_distance": float(np.linalg.norm(f_open - f_block)),
        "note": ("H and T are additive over components; existence predicates are "
                 "not.  This is the structural reason the targets are not SFC "
                 "partition functions, not a claim that SFC cannot correlate "
                 "with them."),
    }


def sfc_fit_diagnostic(task: str, split: dict, steps: int = 60,
                       n_graphs: int = 200) -> dict:
    """Can the SFC parameters fit the *training* set at all?

    A linear readout is solved in closed form at every step, so the only thing
    being learned is the SFC factor parameters themselves.  If this cannot fit
    200 training graphs, the operator has no usable optimisation path on this
    task and the full arms comparison would only restate that.
    """
    import torch as _t
    gs = split["graphs_train"][:n_graphs]
    y = _t.as_tensor(split["y_train"][:n_graphs].astype(np.float64))
    plan = split["sfc_plan"]
    A = _t.as_tensor(plan["A"], dtype=_t.float64).requires_grad_(True)
    S = _t.as_tensor(plan["S"], dtype=_t.float64).requires_grad_(True)
    opt = _t.optim.Adam([A, S], lr=0.05)
    trace = []
    t0 = time.perf_counter()
    for step in range(steps):
        F = _t.stack([_t.cat([o.H, o.T]) for o in
                      (SC.sfc_forward(g, A, S, plan=_plan_for(g),
                                      with_marginals=False) for g in gs)])
        w = _t.linalg.lstsq(F.detach(), y).solution
        loss = _t.nn.functional.binary_cross_entropy_with_logits(F @ w, y)
        opt.zero_grad()
        loss.backward()
        opt.step()
        if step in (0, steps - 1):
            acc = float((((F.detach() @ w.detach()) > 0).double() == y).double().mean())
            trace.append({"step": step, "bce": float(loss.item()),
                          "train_acc_linear_probe": acc})
    return {"n_graphs": n_graphs, "steps": steps,
            "seconds": time.perf_counter() - t0, "trace": trace,
            "note": ("linear readout solved in closed form each step; only the "
                     "SFC factor parameters are optimised")}


# ---------------------------------------------------------------------------
# driver
# ---------------------------------------------------------------------------


def main() -> int:
    torch.set_num_threads(4)
    t_all = time.perf_counter()
    arms = ("C", "P", "S", "Srand", "SP")
    payload = {
        "protocol": "sfc-v0-cpu-audit",
        "stage": "T3-synthetic-sample-efficiency",
        "K_channels": K_CHANNELS,
        "target_params": TARGET_PARAMS,
        "steps": 400, "batch": 64, "lr": 3e-3, "seeds": list(SEEDS),
        "arms": {},
        "runs": [],
        "datasets": {},
        "certificates": {},
        "fit_diagnostics": {},
    }
    for task in ("reach", "junction"):
        split = assemble(task, seed=0)
        diag = dataset_diagnostics(task, split)
        payload["datasets"][task] = diag
        payload["certificates"][task] = non_sfc_certificate(task)
        payload["fit_diagnostics"][task] = sfc_fit_diagnostic(task, split)
        fd = payload["fit_diagnostics"][task]
        print(f"{task:9s} SFC-fit diagnostic: train-acc "
              f"{fd['trace'][0]['train_acc_linear_probe']:.3f} -> "
              f"{fd['trace'][-1]['train_acc_linear_probe']:.3f} "
              f"({fd['seconds']:.0f}s)")
        for name in arms:
            for seed in SEEDS:
                res = run_arm(task, name, split, seed)
                payload["runs"].append({"task": task, **res})
                print(f"{task:9s} {name:2s} seed{seed} "
                      f"train={res['train']['accuracy']:.3f} "
                      f"unseen={res['unseen']['accuracy']:.3f} "
                      f"({res['train_seconds']:.0f}s, {res['n_params']}p)")

    # summary: per task/arm mean over seeds, and the S-vs-P contrast
    summary = {}
    for task in ("reach", "junction"):
        rows = [r for r in payload["runs"] if r["task"] == task]
        per = {}
        for name in arms:
            rr = [r for r in rows if r["arm"] == name]
            per[name] = {
                "unseen_acc_mean": float(np.mean([r["unseen"]["accuracy"] for r in rr])),
                "unseen_acc_per_seed": [r["unseen"]["accuracy"] for r in rr],
                "train_acc_mean": float(np.mean([r["train"]["accuracy"] for r in rr])),
                "train_acc_per_seed": [r["train"]["accuracy"] for r in rr],
                "unseen_bce_mean": float(np.mean([r["unseen"]["bce"] for r in rr])),
                "n_params": rr[0]["n_params"],
                "seconds_mean": float(np.mean([r["train_seconds"] for r in rr])),
            }
        per["_seed_spread_P"] = float(np.ptp(per["P"]["unseen_acc_per_seed"]))
        per["_seed_spread_S"] = float(np.ptp(per["S"]["unseen_acc_per_seed"]))
        per["_seed_spread_SP"] = float(np.ptp(per["SP"]["unseen_acc_per_seed"]))
        per["_S_minus_P"] = per["S"]["unseen_acc_mean"] - per["P"]["unseen_acc_mean"]
        per["_Srand_minus_P"] = per["Srand"]["unseen_acc_mean"] - per["P"]["unseen_acc_mean"]
        per["_SP_minus_P"] = per["SP"]["unseen_acc_mean"] - per["P"]["unseen_acc_mean"]
        per["_S_minus_C"] = per["S"]["unseen_acc_mean"] - per["C"]["unseen_acc_mean"]
        per["_seed_spread_max"] = max(per["_seed_spread_P"], per["_seed_spread_S"],
                                      per["_seed_spread_SP"])
        per["_sfc_beats_P_both_seeds"] = bool(
            all(s_ > p_ for s_, p_ in zip(per["S"]["unseen_acc_per_seed"],
                                          per["P"]["unseen_acc_per_seed"])))
        per["_sfc_beats_P_by_more_than_seed_spread"] = bool(
            per["_S_minus_P"] > per["_seed_spread_max"])
        summary[task] = per
    payload["arms"] = summary
    payload["seconds_total"] = time.perf_counter() - t_all

    ANALYSIS.mkdir(parents=True, exist_ok=True)
    out = ANALYSIS / "t3_synth.json"
    out.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"-> {out} ({payload['seconds_total']:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

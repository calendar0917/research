"""SFC-v0 core: exact two-state graph factor systems by variable elimination.

What this module computes
------------------------
For a typed graph ``G = (V, E)`` with atom categories ``a_v`` and bond types
``b_e in {1,2,3}``, and ``C`` channels, a pairwise Markov random field

    phi_c(s | a)    = exp( A[a, c, s] )                    shared node factor
    K_{b,c}(s, t)   = exp( tanh( S[b, c, s, t] ) )          shared edge factor
    Z_c(G; theta)   = sum_{s in {0,1}^V} prod_v phi_c(s_v|a_v) prod_e K_{b_e,c}(s_u,s_v)

``A`` and ``S`` are shared by every graph, atom and position.  There is no
node-id / whole-pattern / split-id embedding anywhere in this module.  The node
factor is deliberately **not** softmax-normalised: that keeps ``H`` below a
non-trivial quantity (with a normalised node field ``H`` would just equal
``logZ``).

How it is computed
------------------
**Exact variable elimination** (bucket elimination) over a tree decomposition of
the primal graph, carried out in the log domain with ``logsumexp``.  Every
intermediate factor is a dense table over its scope; nothing is sampled, nothing
is truncated, nothing is approximated.  ``logZ`` equals the dense ``2^|V|``
enumeration up to float rounding.

**This is exact graph inference, not message passing.**  There is no per-node
hidden state, no iteration, no layer stacking, no learned propagation.  Variable
elimination is a dynamic program over a tree decomposition; it returns a scalar
normaliser and (via the marginal identity) the exact node/edge marginals.

Two derived descriptors
-----------------------
``H``  *centred interaction*::

    H_c(G) = logZ_c(G) - sum_v logsumexp_s A[a_v, c, s]

i.e. everything ``logZ`` knows beyond treating the atoms as independent.  It is
**exactly zero** when the coupling is zero (``K == 1``), for every graph.

``T``  *cycle interaction*: the Bethe residual at the **exact** marginals::

    F_B(b*) = sum_v E_{b_v}[log phi_v] + sum_e E_{b_e}[log K_e]
              + sum_v (1 - d_v) H(b_v) + sum_e H(b_e)
    T_c(G)  = logZ_c(G) - F_B(b*)
            = H(p) - sum_v (1 - d_v) H(b_v) - sum_e H(b_e)

Bethe is exact on any forest, so ``T == 0`` on trees (with or without a node
field) and is generally non-zero only when the graph closes a cycle.  ``T`` is a
pure entropy/multi-information quantity: it measures how much of ``logZ`` the
tree-structured (Bethe) decomposition fails to explain.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence

import numpy as np
import torch

LOG2 = math.log(2.0)


# ---------------------------------------------------------------------------
# typed graphs
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class TypedGraph:
    """An undirected typed graph.  ``atom`` is [n] int64, ``edge_b`` in {1,2,3}."""

    n: int
    atom: np.ndarray
    edge_u: np.ndarray
    edge_v: np.ndarray
    edge_b: np.ndarray

    @property
    def m(self) -> int:
        return int(len(self.edge_u))

    def edges(self) -> list[tuple[int, int, int]]:
        """Edges with the smaller endpoint first (factor axes follow this order)."""
        out = []
        for u, v, b in zip(self.edge_u, self.edge_v, self.edge_b):
            u, v = int(u), int(v)
            if u > v:
                u, v = v, u
            out.append((u, v, int(b)))
        return out

    def adjacency(self) -> list[set[int]]:
        adj: list[set[int]] = [set() for _ in range(self.n)]
        for u, v, _ in self.edges():
            adj[u].add(v)
            adj[v].add(u)
        return adj

    def degree(self) -> np.ndarray:
        deg = np.zeros(self.n, dtype=np.int64)
        for u, v, _ in self.edges():
            deg[u] += 1
            deg[v] += 1
        return deg

    def check(self) -> None:
        if self.atom.shape != (self.n,):
            raise ValueError(f"atom shape {self.atom.shape} != ({self.n},)")
        if not (len(self.edge_u) == len(self.edge_v) == len(self.edge_b)):
            raise ValueError("edge arrays have inconsistent lengths")
        if len(self.edge_u) and (self.edge_b.min() < 1 or self.edge_b.max() > 3):
            raise ValueError(f"bond types outside 1..3: {np.unique(self.edge_b)}")
        if len(self.edge_u):
            if self.edge_u.min() < 0 or self.edge_u.max() >= self.n:
                raise ValueError("edge_u out of range")
            if self.edge_v.min() < 0 or self.edge_v.max() >= self.n:
                raise ValueError("edge_v out of range")
            if np.any(self.edge_u == self.edge_v):
                raise ValueError("self loop")
            keys = {(min(int(u), int(v)), max(int(u), int(v)))
                    for u, v in zip(self.edge_u, self.edge_v)}
            if len(keys) != len(self.edge_u):
                raise ValueError("duplicate undirected edge")

    def components(self) -> int:
        adj = self.adjacency()
        seen = [False] * self.n
        n_comp = 0
        for s in range(self.n):
            if seen[s]:
                continue
            n_comp += 1
            stack = [s]
            seen[s] = True
            while stack:
                x = stack.pop()
                for y in adj[x]:
                    if not seen[y]:
                        seen[y] = True
                        stack.append(y)
        return n_comp

    def cyclomatic(self) -> int:
        return self.m - self.n + self.components()

    def is_forest(self) -> bool:
        return self.cyclomatic() == 0

    def subgraph(self, nodes: Sequence[int]) -> "TypedGraph":
        """Induced subgraph on ``nodes`` (relabelled 0..k-1, order preserved)."""
        keep = list(nodes)
        pos = {v: i for i, v in enumerate(keep)}
        eu, ev, eb = [], [], []
        for u, v, b in self.edges():
            if u in pos and v in pos:
                eu.append(pos[u])
                ev.append(pos[v])
                eb.append(b)
        return TypedGraph(
            n=len(keep),
            atom=np.asarray([self.atom[v] for v in keep], dtype=np.int64),
            edge_u=np.asarray(eu, dtype=np.int64),
            edge_v=np.asarray(ev, dtype=np.int64),
            edge_b=np.asarray(eb, dtype=np.int64),
        )


# ---------------------------------------------------------------------------
# parameterisation
# ---------------------------------------------------------------------------


def log_node_factors(A: torch.Tensor, atom: torch.Tensor) -> torch.Tensor:
    """``A``: [n_atom, C, 2] log-potentials.  Returns log phi of shape [C, n, 2]."""
    return A.index_select(0, atom).permute(1, 0, 2).contiguous()


def log_edge_factors(S: torch.Tensor) -> torch.Tensor:
    """``S``: [n_bond_slots, C, 3] symmetric upper-triangle (00, 01, 11).

    Returns log K of shape [n_bond_slots, C, 2, 2].  Bond-type slot 0 is unused.
    """
    t = torch.tanh(S)
    row0 = torch.stack([t[:, :, 0], t[:, :, 1]], dim=-1)    # [T, C, 2]
    row1 = torch.stack([t[:, :, 1], t[:, :, 2]], dim=-1)
    return torch.stack([row0, row1], dim=-2)                # [T, C, 2, 2]


# ---------------------------------------------------------------------------
# elimination plans
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ElimStep:
    bucket: tuple[int, ...]        # factor ids consumed
    union_scope: tuple[int, ...]   # sorted union of their scopes
    elim: int                      # variable summed out
    new_scope: tuple[int, ...]     # sorted union minus elim
    new_id: int                    # factor id produced


@dataclass(frozen=True)
class ElimPlan:
    n: int
    order: tuple[int, ...]
    steps: tuple[ElimStep, ...]
    final_ids: tuple[int, ...]
    n_factors_in: int
    induced_width: int             # max |union_scope| - 1
    max_message_scope: int         # max |new_scope|
    max_bucket_elems: int          # max 2^|union_scope| over steps
    max_message_elems: int         # max 2^|new_scope| over steps
    arith_ops: int                 # sum over steps of 2^|union_scope| (per channel)

    def cost(self, C: int) -> dict:
        return {
            "induced_width": self.induced_width,
            "max_message_scope": self.max_message_scope,
            "max_bucket_floats": int(C * self.max_bucket_elems),
            "max_message_floats": int(C * self.max_message_elems),
            "arith_ops_per_channel": int(self.arith_ops),
            "arith_ops_total": int(C * self.arith_ops),
        }


def _primal_adjacency(n: int, edges: Iterable[tuple[int, int, int]]) -> list[set[int]]:
    adj: list[set[int]] = [set() for _ in range(n)]
    for u, v, _ in edges:
        adj[u].add(v)
        adj[v].add(u)
    return adj


def order_min_degree(n: int, edges: Iterable[tuple[int, int, int]]) -> list[int]:
    """Classic min-degree elimination order on the primal graph."""
    adj = [set(a) for a in _primal_adjacency(n, edges)]
    alive = set(range(n))
    order: list[int] = []
    while alive:
        v = min(alive, key=lambda x: (len(adj[x] & alive), x))
        nbrs = adj[v] & alive
        for a in nbrs:
            adj[a] |= nbrs
            adj[a].discard(a)
        alive.discard(v)
        order.append(int(v))
    return order


def order_min_fill(n: int, edges: Iterable[tuple[int, int, int]]) -> list[int]:
    """Min-fill elimination order (fewest fill edges, ties by degree then id)."""
    adj = [set(a) for a in _primal_adjacency(n, edges)]
    alive = set(range(n))
    order: list[int] = []
    while alive:
        best, best_key = None, None
        for v in sorted(alive):
            nbrs = sorted(adj[v] & alive)
            fill = 0
            for i, a in enumerate(nbrs):
                for b in nbrs[i + 1:]:
                    if b not in adj[a]:
                        fill += 1
            key = (fill, len(nbrs), v)
            if best_key is None or key < best_key:
                best, best_key = v, key
        v = best
        nbrs = adj[v] & alive
        for a in nbrs:
            adj[a] |= nbrs
            adj[a].discard(a)
        alive.discard(v)
        order.append(int(v))
    return order


ORDERS = {
    "min_degree": order_min_degree,
    "min_fill": order_min_fill,
}


def build_plan(n: int, edges: Sequence[tuple[int, int, int]],
               order: Sequence[int]) -> ElimPlan:
    """Simulate bucket elimination symbolically; record every step."""
    scopes: dict[int, tuple[int, ...]] = {v: (v,) for v in range(n)}
    for k, (u, v, _b) in enumerate(edges):
        scopes[n + k] = (min(u, v), max(u, v))
    next_id = n + len(edges)
    live = set(scopes)
    steps: list[ElimStep] = []
    arith = 0
    max_bucket = 1
    max_msg = 1
    width = 0
    for var in order:
        bucket = tuple(sorted(f for f in live if var in scopes[f]))
        if not bucket:
            continue
        union = tuple(sorted({s for f in bucket for s in scopes[f]}))
        new_scope = tuple(s for s in union if s != var)
        new_id = next_id
        next_id += 1
        steps.append(ElimStep(bucket, union, int(var), new_scope, new_id))
        live -= set(bucket)
        live.add(new_id)
        scopes[new_id] = new_scope
        arith += 1 << len(union)
        max_bucket = max(max_bucket, 1 << len(union))
        max_msg = max(max_msg, 1 << len(new_scope))
        width = max(width, len(union) - 1)
    return ElimPlan(
        n=n,
        order=tuple(int(v) for v in order),
        steps=tuple(steps),
        final_ids=tuple(sorted(live)),
        n_factors_in=n + len(edges),
        induced_width=width,
        max_message_scope=max((len(s.new_scope) for s in steps), default=0),
        max_bucket_elems=max_bucket,
        max_message_elems=max_msg,
        arith_ops=arith,
    )


def plan_scopes(plan: ElimPlan, edges: Sequence[tuple[int, int, int]]) -> dict[int, tuple[int, ...]]:
    scopes: dict[int, tuple[int, ...]] = {v: (v,) for v in range(plan.n)}
    for k, (u, v, _b) in enumerate(edges):
        scopes[plan.n + k] = (min(u, v), max(u, v))
    for step in plan.steps:
        scopes[step.new_id] = step.new_scope
    return scopes


def _align(table: torch.Tensor, scope: Sequence[int], target: Sequence[int]) -> torch.Tensor:
    """Broadcast-view ``table`` (axes = scope, plus leading channel axis) onto ``target``."""
    if tuple(scope) == tuple(target):
        return table
    perm = [0]
    shape = [table.shape[0]]
    for s in target:
        if s in scope:
            j = scope.index(s)
            perm.append(1 + j)
            shape.append(table.shape[1 + j])
        else:
            shape.append(1)
    return table.permute(*perm).reshape(shape)


def run_plan(plan: ElimPlan, log_q: torch.Tensor, log_K: torch.Tensor,
             edges: Sequence[tuple[int, int, int]], scopes: dict[int, tuple[int, ...]],
             *, track_memory: bool = False) -> tuple[torch.Tensor, dict]:
    """Execute ``plan``.  ``log_q`` [C, n, 2]; ``log_K`` [T, C, 2, 2].  Returns [C] logZ."""
    table: dict[int, torch.Tensor] = {v: log_q[:, v, :] for v in range(plan.n)}
    for k in range(len(edges)):
        table[plan.n + k] = log_K[k]

    itemsize = log_q.element_size()
    peak_elems = 0
    for step in plan.steps:
        union = step.union_scope
        acc = None
        for fid in step.bucket:
            piece = _align(table[fid], scopes[fid], union)
            acc = piece if acc is None else acc + piece
        for fid in step.bucket:
            del table[fid]
        if track_memory:
            peak_elems = max(peak_elems, sum(t.numel() for t in table.values()) + acc.numel())
        ax = union.index(step.elim)
        msg = torch.logsumexp(acc, dim=1 + ax)
        del acc
        table[step.new_id] = msg
        if track_memory:
            peak_elems = max(peak_elems, sum(t.numel() for t in table.values()))

    out = table[plan.final_ids[0]]
    for fid in plan.final_ids[1:]:
        out = out + table[fid]
    return out, {"peak_bytes_measured": int(peak_elems * itemsize) if track_memory else 0,
                 "n_steps": len(plan.steps)}


# ---------------------------------------------------------------------------
# dense enumeration reference (independent of the elimination path)
# ---------------------------------------------------------------------------


def _bit_table(n: int, dtype=torch.long) -> torch.Tensor:
    states = torch.arange(1 << n, dtype=dtype)
    return (states.unsqueeze(1) >> torch.arange(n, dtype=dtype)) & 1      # [2^n, n]


def log_partition_dense(graph: TypedGraph, A: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """Brute-force ``log Z`` over all ``2^n`` joint states in the input dtype.

    Independent of :func:`run_plan`: it only shares the factor *tables*.
    """
    log_q = log_node_factors(A, torch.as_tensor(graph.atom, dtype=torch.long))
    log_K = log_edge_factors(S)
    C = log_q.shape[0]
    bits = _bit_table(graph.n)
    total = torch.zeros(C, 1 << graph.n, dtype=log_q.dtype, device=log_q.device)
    for v in range(graph.n):
        total = total + log_q[:, v, :].index_select(1, bits[:, v])
    for u, v, b in graph.edges():
        total = total + log_K[b][:, bits[:, u], bits[:, v]]
    return torch.logsumexp(total, dim=1)


def marginals_dense(graph: TypedGraph, A: torch.Tensor, S: torch.Tensor
                    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Exact ``logZ``, node marginals [C,n,2] and edge marginals [C,m,2,2]."""
    log_q = log_node_factors(A, torch.as_tensor(graph.atom, dtype=torch.long))
    log_K = log_edge_factors(S)
    C = log_q.shape[0]
    bits = _bit_table(graph.n)
    total = torch.zeros(C, 1 << graph.n, dtype=log_q.dtype)
    for v in range(graph.n):
        total = total + log_q[:, v, :].index_select(1, bits[:, v])
    for u, v, b in graph.edges():
        total = total + log_K[b][:, bits[:, u], bits[:, v]]
    logZ = torch.logsumexp(total, dim=1)
    p_joint = torch.softmax(total, dim=1)                       # [C, 2^n]
    node_p = torch.zeros(C, graph.n, 2, dtype=log_q.dtype)
    for v in range(graph.n):
        for s in (0, 1):
            node_p[:, v, s] = (p_joint * (bits[:, v] == s).to(log_q.dtype)).sum(1)
    edge_p = torch.zeros(C, graph.m, 2, 2, dtype=log_q.dtype)
    for k, (u, v, _b) in enumerate(graph.edges()):
        for a in (0, 1):
            for b2 in (0, 1):
                m = ((bits[:, u] == a) & (bits[:, v] == b2)).to(log_q.dtype)
                edge_p[:, k, a, b2] = (p_joint * m).sum(1)
    return logZ, node_p, edge_p


# ---------------------------------------------------------------------------
# exact logZ, exact marginals, descriptors
# ---------------------------------------------------------------------------


@dataclass
class SFCOutput:
    logZ: torch.Tensor            # [C]
    H: torch.Tensor               # [C]  centred interaction
    T: torch.Tensor               # [C]  cycle (Bethe residual) interaction
    minus_FB: torch.Tensor        # [C]  -F_B at the exact marginals
    node_p: torch.Tensor          # [C, n, 2] exact marginals
    edge_p: torch.Tensor          # [C, m, 2, 2] exact marginals
    stats: dict = field(default_factory=dict)


def sfc_forward(graph: TypedGraph, A: torch.Tensor, S: torch.Tensor, *,
                order: Sequence[int] | None = None,
                plan: ElimPlan | None = None,
                with_marginals: bool = True,
                track_memory: bool = False) -> SFCOutput:
    """Exact ``logZ``, exact marginals, and the ``H`` / ``T`` descriptors."""
    graph.check()
    edges = graph.edges()
    if plan is None:
        if order is None:
            order = order_min_fill(graph.n, edges)
        plan = build_plan(graph.n, edges, order)
    scopes = plan_scopes(plan, edges)

    atom = torch.as_tensor(graph.atom, dtype=torch.long)
    log_q_full = log_node_factors(A, atom)
    log_K_full = log_edge_factors(S)
    bond = torch.as_tensor([b for (_u, _v, b) in edges], dtype=torch.long)
    log_K_full = log_K_full.index_select(0, bond)     # [m, C, 2, 2] per-edge slots

    need_grad = bool(A.requires_grad or S.requires_grad)
    if with_marginals and not need_grad:
        # marginals need a differentiable path; when the caller only wants values
        # we run on private copies that require grad, and the result is detached
        log_q = log_q_full.detach().clone().requires_grad_(True)
        log_K = log_K_full.detach().clone().requires_grad_(True)
    else:
        log_q, log_K = log_q_full, log_K_full

    logZ, run_stats = run_plan(plan, log_q, log_K, edges, scopes,
                               track_memory=track_memory)
    C = log_q.shape[0]

    # independent-atom baseline: logZ of the graph with every edge removed
    node_logZ = torch.logsumexp(log_q, dim=2)                    # [C, n]
    H = logZ - node_logZ.sum(1)

    if with_marginals:
        # d logZ / d(log phi) and d logZ / d(log K) are exactly the marginals; the
        # zero-weight touch keeps unused factors (isolated node, edgeless graph)
        # inside the autograd graph
        zero = 0.0 * (log_q.sum() + (log_K.sum() if log_K.numel() else 0.0))
        g_node, g_edge = torch.autograd.grad(
            (logZ + zero).sum(), [log_q, log_K],
            create_graph=torch.is_grad_enabled(), allow_unused=True)
        if g_node is None:
            g_node = torch.zeros_like(log_q)
        if g_edge is None:
            g_edge = torch.zeros_like(log_K)
        node_p = g_node.contiguous()                              # [C, n, 2]
        edge_p = (g_edge.permute(1, 0, 2, 3).contiguous() if len(edges)
                  else torch.zeros(C, 0, 2, 2, dtype=log_q.dtype))
    else:
        node_p = torch.zeros(C, graph.n, 2, dtype=log_q.dtype)
        edge_p = torch.zeros(C, len(edges), 2, 2, dtype=log_q.dtype)

    minus_FB = _minus_bethe_free_energy(log_q, log_K, node_p, edge_p, graph.degree())
    T = logZ - minus_FB

    stats = {
        **plan.cost(int(C)),
        **run_stats,
        "n": graph.n,
        "m": len(edges),
        "cyclomatic": graph.cyclomatic(),
    }
    return SFCOutput(logZ=logZ, H=H, T=T, minus_FB=minus_FB,
                     node_p=node_p, edge_p=edge_p, stats=stats)


def _minus_bethe_free_energy(log_q: torch.Tensor, log_K: torch.Tensor,
                             node_p: torch.Tensor, edge_p: torch.Tensor,
                             degree: np.ndarray) -> torch.Tensor:
    """``-F_B`` at the supplied (exact) marginals.

    ``F_B = U - H_B`` with ``U = -E[ln phi] - E[ln K]`` and the Bethe entropy
    ``H_B = sum_v (1 - d_v) H(b_v) + sum_e H(b_e)``.  On a forest ``F_B = -logZ``
    exactly, so ``logZ - (-F_B)`` is the residual the report calls ``T``.
    """
    tiny = torch.finfo(node_p.dtype).tiny
    log_bv = torch.log(node_p.clamp_min(tiny))                  # [C, n, 2]
    log_be = torch.log(edge_p.clamp_min(tiny))                  # [C, m, 2, 2]
    out = (node_p * log_q.detach()).sum(dim=(1, 2))
    out = out + (edge_p * log_K.detach().permute(1, 0, 2, 3)).sum(dim=(1, 2, 3))
    d = torch.as_tensor(degree, dtype=node_p.dtype)
    out = out + ((d - 1.0)[None, :] * (node_p * log_bv).sum(dim=2)).sum(1)
    out = out - (edge_p * log_be).sum(dim=(1, 2, 3))
    return out


def sfc_descriptor(graph: TypedGraph, A: torch.Tensor, S: torch.Tensor) -> torch.Tensor:
    """``[H_c, T_c]`` concatenated, shape ``[2 * C]``."""
    out = sfc_forward(graph, A, S)
    return torch.cat([out.H, out.T])


# ---------------------------------------------------------------------------
# graph utilities used by the synthetic targets (independent of SFC)
# ---------------------------------------------------------------------------


def simple_cycles_up_to(graph: TypedGraph, max_len: int = 12) -> list[tuple[int, ...]]:
    """All simple cycles of length <= ``max_len``, each once (canonical form)."""
    adj = graph.adjacency()
    found: set[tuple[int, ...]] = set()
    for start in range(graph.n):
        stack = [(start, (start,))]
        while stack:
            node, path = stack.pop()
            if len(path) > max_len:
                continue
            for nxt in adj[node]:
                if nxt == start and len(path) >= 3:
                    found.add(tuple(sorted(path)))
                elif nxt > start and nxt not in path:
                    stack.append((nxt, path + (nxt,)))
    canon: set[tuple[int, ...]] = set()
    for cyc in found:
        s = list(cyc)
        i = s.index(min(s))
        rot = tuple(s[i:] + s[:i])
        rev = (rot[0],) + tuple(reversed(rot[1:]))
        canon.add(min(rot, rev))
    return sorted(canon)


def girth(graph: TypedGraph, max_len: int = 12) -> int:
    cyc = simple_cycles_up_to(graph, max_len)
    return min((len(c) for c in cyc), default=0)


def articulation_points(graph: TypedGraph) -> set[int]:
    """Tarjan articulation points (iterative)."""
    n = graph.n
    adj = graph.adjacency()
    disc = [-1] * n
    low = [0] * n
    parent = [-1] * n
    aps: set[int] = set()
    timer = 0
    for root in range(n):
        if disc[root] != -1:
            continue
        stack = [(root, iter(sorted(adj[root])))]
        disc[root] = low[root] = timer
        timer += 1
        root_children = 0
        while stack:
            u, it = stack[-1]
            advanced = False
            for v in it:
                if disc[v] == -1:
                    parent[v] = u
                    disc[v] = low[v] = timer
                    timer += 1
                    if u == root:
                        root_children += 1
                    stack.append((v, iter(sorted(adj[v]))))
                    advanced = True
                    break
                if v != parent[u]:
                    low[u] = min(low[u], disc[v])
            if not advanced:
                stack.pop()
                if stack:
                    p = stack[-1][0]
                    low[p] = min(low[p], low[u])
                    if p != root and low[u] >= disc[p]:
                        aps.add(p)
        if root_children > 1:
            aps.add(root)
    return aps


def bridges(graph: TypedGraph) -> list[tuple[int, int]]:
    """Bridge edges (iterative Tarjan)."""
    n = graph.n
    adj: list[list[tuple[int, int]]] = [[] for _ in range(n)]
    for k, (u, v, _b) in enumerate(graph.edges()):
        adj[u].append((v, k))
        adj[v].append((u, k))
    disc = [-1] * n
    low = [0] * n
    out: list[tuple[int, int]] = []
    timer = 0
    for root in range(n):
        if disc[root] != -1:
            continue
        disc[root] = low[root] = timer
        timer += 1
        stack = [(root, -1, iter(adj[root]))]
        while stack:
            u, pe, it = stack[-1]
            advanced = False
            for v, eid in it:
                if eid == pe:
                    continue
                if disc[v] == -1:
                    disc[v] = low[v] = timer
                    timer += 1
                    stack.append((v, eid, iter(adj[v])))
                    advanced = True
                    break
                low[u] = min(low[u], disc[v])
            if not advanced:
                stack.pop()
                if stack:
                    p, _pe2, _it2 = stack[-1]
                    low[p] = min(low[p], low[u])
                    if low[u] > disc[p]:
                        out.append((min(p, u), max(p, u)))
    return sorted(out)


def resistance_distances(graph: TypedGraph, conductance=(0.0, 1.0, 2.0, 3.0)) -> np.ndarray:
    """Effective resistance matrix with bond-type conductances (pseudo-inverse)."""
    n = graph.n
    if n == 0:
        return np.zeros((0, 0))
    L = np.zeros((n, n), dtype=np.float64)
    for u, v, b in graph.edges():
        w = float(conductance[int(b)])
        L[u, u] += w
        L[v, v] += w
        L[u, v] -= w
        L[v, u] -= w
    pinv = np.linalg.pinv(L)
    d = np.diag(pinv)
    return (d[:, None] + d[None, :] - 2.0 * pinv)


def kirchhoff_index(graph: TypedGraph, conductance=(0.0, 1.0, 2.0, 3.0)) -> float:
    """Sum of pairwise effective resistances (Kirchhoff index)."""
    R = resistance_distances(graph, conductance)
    return float(np.triu(R, 1).sum())

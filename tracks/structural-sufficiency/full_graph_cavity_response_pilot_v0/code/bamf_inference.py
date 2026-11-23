"""Exact global marginals for the BAMF-v1 bond-anchored binary field.

This module implements *exact* variable elimination on the physical-bond
primal graph via a junction tree (clique tree) built from a min-fill
elimination order.  The message passing below is exact sum-product inference on
that tree; it is **not** an approximate belief-propagation loop on the original
molecular graph and it is not a learned message-passing layer.

The model has ``C`` independent binary channels per atom.  For each channel and
molecule the joint distribution is

    p(s | G) ∝ exp( Σ_v logφ_v(s_v) + Σ_{(u,v)∈E} logψ_uv(s_u,s_v) )

with symmetric 2x2 edge potentials.  The junction tree returns every node
marginal and every physical-edge 2x2 marginal exactly (up to float rounding).
The same code also computes the independent-edge (I-arm) marginals, which use
only the two endpoint node factors and the one edge factor.

No use of official valid/test data occurs here.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Sequence

import numpy as np
import torch

__all__ = [
    "GraphPlan",
    "GlobalPlan",
    "BatchPlan",
    "build_graph_plan",
    "build_global_plan",
    "min_fill_width",
    "edge_log_potential",
    "exact_marginals",
    "independent_marginals",
    "marginal_features",
]


# ---------------------------------------------------------------------------
# small helpers
# ---------------------------------------------------------------------------


def _bits(x: int) -> Iterable[int]:
    while x:
        lb = x & -x
        yield lb.bit_length() - 1
        x ^= lb


def _scope_state(scope: Sequence[int], state: int) -> dict[int, int]:
    return {int(v): int((state >> j) & 1) for j, v in enumerate(scope)}


def _index_in_scope(scope: Sequence[int], v: int) -> int:
    for j, x in enumerate(scope):
        if int(x) == int(v):
            return j
    raise KeyError(v)


def _sep_index(scope: Sequence[int], sep: Sequence[int], state: int) -> int:
    out = 0
    for j, v in enumerate(sep):
        out |= ((state >> _index_in_scope(scope, v)) & 1) << j
    return out


def _parent_state_for_child(
    scope_c: Sequence[int], scope_p: Sequence[int], sep: Sequence[int], state_c: int
) -> int:
    out = 0
    for j, v in enumerate(sep):
        bit = (state_c >> _index_in_scope(scope_c, v)) & 1
        out |= bit << _index_in_scope(scope_p, v)
    return out


# ---------------------------------------------------------------------------
# min-fill order and clique tree
# ---------------------------------------------------------------------------


def min_fill_width(n: int, edges: Sequence[tuple[int, int]]) -> int:
    """Induced width of the min-fill order (the clique-size-1)."""
    adj = [0] * int(n)
    for u, v in edges:
        u, v = int(u), int(v)
        if u == v:
            raise ValueError("self loop")
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    remaining = (1 << int(n)) - 1
    width = 0
    while remaining:
        best = None
        rem = remaining
        while rem:
            lb = rem & -rem
            v = lb.bit_length() - 1
            rem ^= lb
            nb = adj[v] & remaining & ~lb
            fill = 0
            x = nb
            while x:
                lb2 = x & -x
                u = lb2.bit_length() - 1
                x ^= lb2
                fill += (nb & ~adj[u] & ~(1 << u)).bit_count()
            fill //= 2
            key = (fill, nb.bit_count(), v)
            if best is None or key < best[0]:
                best = (key, v, nb)
        _, v, nb = best
        width = max(width, nb.bit_count())
        # fill the neighbourhood into a clique
        x = nb
        while x:
            lb = x & -x
            u = lb.bit_length() - 1
            x ^= lb
            adj[u] |= nb & ~(1 << u)
        remaining &= ~(1 << v)
    return int(width)


@dataclass(frozen=True)
class GraphPlan:
    """Topology-only plan for one graph.  All arrays are numpy, CPU."""

    n: int
    m: int
    scope: np.ndarray          # [C, Smax] local atom ids, sorted, padding 0
    scope_size: np.ndarray     # [C]
    parent: np.ndarray         # [C] local parent or -1
    level: np.ndarray          # [C] root = 0
    node_clique: np.ndarray    # [n] local clique id
    node_pos: np.ndarray       # [n] position in that clique
    edge_clique: np.ndarray    # [m] local clique id
    edge_pos_u: np.ndarray     # [m]
    edge_pos_v: np.ndarray     # [m]
    up_sep_child: np.ndarray   # [C, S] state -> separator index
    up_sep_parent: np.ndarray  # [C, S] parent state -> separator index
    down_gather_parent: np.ndarray  # [C, S] child state -> parent state
    sep_size: np.ndarray       # [C]
    width: int


def _build_one_graph_plan(n: int, edges: Sequence[tuple[int, int]], smax: int) -> GraphPlan:
    n = int(n)
    edges = [(int(u), int(v)) for u, v in edges]
    for u, v in edges:
        if u == v:
            raise ValueError("self loop")
    # deterministic canonical edge order, but keep the caller's edge order for
    # the edge arrays (the potential is symmetric and the features are too).
    adj = [0] * n
    for u, v in edges:
        adj[u] |= 1 << v
        adj[v] |= 1 << u
    remaining = (1 << n) - 1 if n else 0

    # factor id -> (scope tuple, origin clique or None)
    factors: dict[int, tuple[tuple[int, ...], int | None]] = {}
    active: set[int] = set()
    var_factors: list[set[int]] = [set() for _ in range(n)]
    next_fid = 0
    for v in range(n):
        factors[next_fid] = ((v,), None)
        active.add(next_fid)
        var_factors[v].add(next_fid)
        next_fid += 1
    for u, v in edges:
        factors[next_fid] = (tuple(sorted((u, v))), None)
        active.add(next_fid)
        var_factors[u].add(next_fid)
        var_factors[v].add(next_fid)
        next_fid += 1

    parent = [-1] * n
    scopes: list[tuple[int, ...]] = []
    step_of = [-1] * n
    order: list[int] = []
    width = 0

    for step in range(n):
        best = None
        rem = remaining
        while rem:
            lb = rem & -rem
            v = lb.bit_length() - 1
            rem ^= lb
            nb = adj[v] & remaining & ~lb
            fill = 0
            x = nb
            while x:
                lb2 = x & -x
                u = lb2.bit_length() - 1
                x ^= lb2
                fill += (nb & ~adj[u] & ~(1 << u)).bit_count()
            fill //= 2
            key = (fill, nb.bit_count(), v)
            if best is None or key < best[0]:
                best = (key, v, nb)
        _, v, nb = best
        bucket = sorted(var_factors[v])
        if not bucket:
            raise RuntimeError(f"no factor contains variable {v}")
        U: set[int] = set()
        for f in bucket:
            U.update(factors[f][0])
        if v not in U:
            raise RuntimeError("bucket does not contain the eliminated variable")
        scope = tuple(sorted(U))
        width = max(width, len(scope) - 1)
        if len(scope) > smax:
            raise RuntimeError(f"clique size {len(scope)} exceeds declared Smax={smax}")
        c = step
        scopes.append(scope)
        step_of[v] = c
        order.append(v)
        for f in bucket:
            fscope, origin = factors[f]
            if origin is not None:
                if parent[origin] != -1:
                    raise RuntimeError("factor tree gives a clique two parents")
                parent[origin] = c
            del factors[f]
            active.discard(f)
            for x in fscope:
                var_factors[x].discard(f)
        new_scope = tuple(x for x in scope if x != v)
        factors[next_fid] = (new_scope, c)
        active.add(next_fid)
        for x in new_scope:
            var_factors[x].add(next_fid)
        next_fid += 1
        # fill neighbourhood and remove v
        x = nb
        while x:
            lb = x & -x
            u = lb.bit_length() - 1
            x ^= lb
            adj[u] |= nb & ~(1 << u)
        remaining &= ~(1 << v)

    # every remaining factor must be a scalar (all variables eliminated); its
    # origin clique is a root.
    for f in sorted(active):
        scope, origin = factors[f]
        if scope:
            raise RuntimeError(f"active factor with non-empty scope {scope}")
        if origin is not None and parent[origin] != -1:
            raise RuntimeError("root origin already has a parent")
    for c, p in enumerate(parent):
        if p != -1 and p <= c:
            raise RuntimeError(f"parent {p} not after child {c}")

    # levels: depth from the root (root = 0).  Parents are later than
    # children, so a reverse pass sees every parent before its children.
    level = [0] * n
    for c in range(n - 1, -1, -1):
        p = parent[c]
        if p != -1:
            level[c] = level[p] + 1

    # node/edge assignment
    node_clique = np.asarray(step_of, dtype=np.int32)
    node_pos = np.zeros(n, dtype=np.int32)
    for v in range(n):
        node_pos[v] = _index_in_scope(scopes[step_of[v]], v)
    edge_clique = np.zeros(len(edges), dtype=np.int32)
    edge_pos_u = np.zeros(len(edges), dtype=np.int32)
    edge_pos_v = np.zeros(len(edges), dtype=np.int32)
    pos_in_order = {v: i for i, v in enumerate(order)}
    for e, (u, v) in enumerate(edges):
        w = u if pos_in_order[u] < pos_in_order[v] else v
        c = step_of[w]
        edge_clique[e] = c
        edge_pos_u[e] = _index_in_scope(scopes[c], u)
        edge_pos_v[e] = _index_in_scope(scopes[c], v)

    S = 1 << int(smax)
    up_sep_child = np.zeros((n, S), dtype=np.int16)
    up_sep_parent = np.zeros((n, S), dtype=np.int16)
    down_gather_parent = np.zeros((n, S), dtype=np.int16)
    sep_size = np.zeros(n, dtype=np.int8)
    for c in range(n):
        p = parent[c]
        if p == -1:
            continue
        sc, sp = scopes[c], scopes[p]
        sep = tuple(sorted(set(sc) & set(sp)))
        sep_size[c] = len(sep)
        for s in range(1 << len(sc)):
            up_sep_child[c, s] = _sep_index(sc, sep, s)
        for s in range(1 << len(sp)):
            up_sep_parent[c, s] = _sep_index(sp, sep, s)
        for s in range(1 << len(sc)):
            down_gather_parent[c, s] = _parent_state_for_child(sc, sp, sep, s)

    scope_arr = np.zeros((n, smax), dtype=np.int16)
    for c, sc in enumerate(scopes):
        scope_arr[c, : len(sc)] = np.asarray(sc, dtype=np.int16)

    return GraphPlan(
        n=n,
        m=len(edges),
        scope=scope_arr,
        scope_size=np.asarray([len(s) for s in scopes], dtype=np.int8),
        parent=np.asarray(parent, dtype=np.int32),
        level=np.asarray(level, dtype=np.int32),
        node_clique=node_clique,
        node_pos=node_pos,
        edge_clique=edge_clique,
        edge_pos_u=edge_pos_u,
        edge_pos_v=edge_pos_v,
        up_sep_child=up_sep_child,
        up_sep_parent=up_sep_parent,
        down_gather_parent=down_gather_parent,
        sep_size=sep_size,
        width=int(width),
    )


def build_graph_plan(n: int, edges: Sequence[tuple[int, int]], *, smax: int = 4) -> GraphPlan:
    return _build_one_graph_plan(n, edges, smax)


# ---------------------------------------------------------------------------
# global plan and batch selection
# ---------------------------------------------------------------------------


@dataclass
class GlobalPlan:
    """Concatenated topology plans for a fixed corpus, with offsets."""

    smax: int
    n_graphs: int
    clique_scope: torch.Tensor       # [Ctot, Smax] local atom ids
    clique_scope_size: torch.Tensor  # [Ctot]
    clique_parent: torch.Tensor      # [Ctot] global parent or -1
    clique_level: torch.Tensor       # [Ctot]
    node_clique: torch.Tensor        # [Ntot] global clique
    node_pos: torch.Tensor           # [Ntot]
    edge_clique: torch.Tensor        # [Etot] global clique
    edge_pos_u: torch.Tensor         # [Etot]
    edge_pos_v: torch.Tensor         # [Etot]
    up_sep_child: torch.Tensor       # [Ctot, S]
    up_sep_parent: torch.Tensor      # [Ctot, S]
    down_gather_parent: torch.Tensor # [Ctot, S]
    sep_size: torch.Tensor           # [Ctot]
    graph_clique_ptr: torch.Tensor   # [G+1]
    graph_node_ptr: torch.Tensor     # [G+1]
    graph_edge_ptr: torch.Tensor     # [G+1]
    widths: torch.Tensor             # [G]
    max_clique_size: int
    max_width: int

    def to(self, device) -> "GlobalPlan":
        for name in (
            "clique_scope", "clique_scope_size", "clique_parent", "clique_level",
            "node_clique", "node_pos", "edge_clique", "edge_pos_u", "edge_pos_v",
            "up_sep_child", "up_sep_parent", "down_gather_parent", "sep_size",
            "graph_clique_ptr", "graph_node_ptr", "graph_edge_ptr", "widths",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self

    def select(self, rows: Sequence[int], atom_counts: Sequence[int],
               bond_counts: Sequence[int]) -> "BatchPlan":
        return _select_batch(self, rows, atom_counts, bond_counts)


def build_global_plan(
    graphs: Sequence[tuple[int, Sequence[tuple[int, int]]]],
    *,
    smax: int | None = None,
    verbose: bool = False,
) -> GlobalPlan:
    """Build one junction plan for every graph in ``graphs``.

    ``graphs`` is a sequence of ``(n, edges)``.  The graphs are independent;
    the returned offsets let a training batch select its own sub-forest.
    """
    if smax is None:
        # min-fill has width <= 3 on the official ZINC train split; a padded
        # clique size of 4 is enough.  Derive a safe bound first.
        widths = [min_fill_width(n, e) for n, e in graphs]
        smax = max(widths) + 1
    smax = int(smax)
    plans: list[GraphPlan] = []
    for i, (n, edges) in enumerate(graphs):
        plans.append(_build_one_graph_plan(n, edges, smax))
        if verbose and (i + 1) % 2000 == 0:
            print(f"[bamf-plan] {i+1}/{len(graphs)}", flush=True)

    clique_ptr = np.concatenate([[0], np.cumsum([p.n for p in plans])]).astype(np.int64)
    node_ptr = np.concatenate([[0], np.cumsum([p.n for p in plans])]).astype(np.int64)
    edge_ptr = np.concatenate([[0], np.cumsum([p.m for p in plans])]).astype(np.int64)
    widths = np.asarray([p.width for p in plans], dtype=np.int32)
    max_clique_size = int(max((p.scope_size.max() if p.n else 0) for p in plans))

    # Clique ids in GraphPlan are graph-local.  The concatenated plan uses
    # global ids, so parent/node_clique/edge_clique must be shifted by each
    # graph's clique start.  Scope variables stay graph-local; batch selection
    # adds the corresponding atom offset later.  Roots keep parent == -1.
    clique_start = clique_ptr[:-1]

    def cat(name, dtype, *, offset: bool = False):
        arrs = []
        for i, p in enumerate(plans):
            a = np.asarray(getattr(p, name))
            if offset and a.size:
                if name == "parent":
                    a = np.where(a >= 0, a + clique_start[i], -1)
                else:
                    a = a + clique_start[i]
            arrs.append(a)
        return torch.as_tensor(np.concatenate(arrs, axis=0), dtype=dtype)

    return GlobalPlan(
        smax=smax,
        n_graphs=len(plans),
        clique_scope=cat("scope", torch.int16),
        clique_scope_size=cat("scope_size", torch.int16),
        clique_parent=cat("parent", torch.int32, offset=True),
        clique_level=cat("level", torch.int32),
        node_clique=cat("node_clique", torch.int32, offset=True),
        node_pos=cat("node_pos", torch.int32),
        edge_clique=cat("edge_clique", torch.int32, offset=True),
        edge_pos_u=cat("edge_pos_u", torch.int32),
        edge_pos_v=cat("edge_pos_v", torch.int32),
        up_sep_child=cat("up_sep_child", torch.int16),
        up_sep_parent=cat("up_sep_parent", torch.int16),
        down_gather_parent=cat("down_gather_parent", torch.int16),
        sep_size=cat("sep_size", torch.int16),
        graph_clique_ptr=torch.as_tensor(clique_ptr, dtype=torch.int64),
        graph_node_ptr=torch.as_tensor(node_ptr, dtype=torch.int64),
        graph_edge_ptr=torch.as_tensor(edge_ptr, dtype=torch.int64),
        widths=torch.as_tensor(widths, dtype=torch.int32),
        max_clique_size=max_clique_size,
        max_width=int(widths.max()) if len(widths) else 0,
    )


@dataclass
class BatchPlan:
    smax: int
    clique_scope: torch.Tensor
    clique_scope_size: torch.Tensor
    parent: torch.Tensor
    level: torch.Tensor
    node_clique: torch.Tensor
    node_pos: torch.Tensor
    edge_clique: torch.Tensor
    edge_pos_u: torch.Tensor
    edge_pos_v: torch.Tensor
    up_sep_child: torch.Tensor
    up_sep_parent: torch.Tensor
    down_gather_parent: torch.Tensor
    sep_size: torch.Tensor
    n_cliques: int
    n_atoms: int
    n_bonds: int
    n_graphs: int
    clique_graph: torch.Tensor

    def to(self, device) -> "BatchPlan":
        for name in (
            "clique_scope", "clique_scope_size", "parent", "level",
            "node_clique", "node_pos", "edge_clique", "edge_pos_u",
            "edge_pos_v", "up_sep_child", "up_sep_parent",
            "down_gather_parent", "sep_size", "clique_graph",
        ):
            setattr(self, name, getattr(self, name).to(device))
        return self


def _select_batch(plan: GlobalPlan, rows: Sequence[int], atom_counts: Sequence[int],
                 bond_counts: Sequence[int]) -> BatchPlan:
    rows = [int(r) for r in rows]
    clique_ids = []
    for r in rows:
        s = int(plan.graph_clique_ptr[r])
        e = int(plan.graph_clique_ptr[r + 1])
        clique_ids.append(torch.arange(s, e, dtype=torch.int64))
    if clique_ids:
        clique_idx = torch.cat(clique_ids)
    else:
        clique_idx = torch.zeros(0, dtype=torch.int64)
    k = int(clique_idx.numel())
    # global -> batch local clique map.  The global plan has ~2.3e5 cliques for
    # ZINC-10k; a dense map is small and fast.
    cmap = torch.full((int(plan.graph_clique_ptr[-1]),), -1, dtype=torch.int32)
    cmap[clique_idx.to(torch.int64)] = torch.arange(k, dtype=torch.int32)

    scope = plan.clique_scope.index_select(0, clique_idx).to(torch.int32)
    sc_size = plan.clique_scope_size.index_select(0, clique_idx).to(torch.int32)
    parent_global = plan.clique_parent.index_select(0, clique_idx)
    parent = torch.where(
        parent_global >= 0,
        cmap[parent_global.clamp_min(0).to(torch.int64)],
        torch.full_like(parent_global, -1, dtype=torch.int32),
    )
    level = plan.clique_level.index_select(0, clique_idx).to(torch.int32)

    # atom/bond global offsets for the selected rows
    atom_ptr = plan.graph_node_ptr
    bond_ptr = plan.graph_edge_ptr
    atom_start = atom_ptr[torch.as_tensor(rows, dtype=torch.int64)]
    bond_start = bond_ptr[torch.as_tensor(rows, dtype=torch.int64)]
    atom_counts_t = torch.as_tensor(atom_counts, dtype=torch.int64)
    bond_counts_t = torch.as_tensor(bond_counts, dtype=torch.int64)
    atom_off_in_batch = torch.cat(
        [torch.zeros(1, dtype=torch.int64), torch.cumsum(atom_counts_t, 0)[:-1]]
    )
    # map each selected clique to its graph's offset
    clique_graph = torch.zeros(k, dtype=torch.int64)
    pos = 0
    for i, r in enumerate(rows):
        m = int(plan.graph_clique_ptr[r + 1] - plan.graph_clique_ptr[r])
        clique_graph[pos:pos + m] = i
        pos += m
    scope = scope + atom_off_in_batch[clique_graph].to(torch.int32).unsqueeze(1)

    # nodes of the selected rows
    node_ids = []
    for r, c in zip(rows, atom_counts):
        s = int(plan.graph_node_ptr[r])
        node_ids.append(torch.arange(s, s + int(c), dtype=torch.int64))
    node_idx = torch.cat(node_ids) if node_ids else torch.zeros(0, dtype=torch.int64)
    node_clique_global = plan.node_clique.index_select(0, node_idx)
    node_clique = cmap[node_clique_global.to(torch.int64)]
    node_pos = plan.node_pos.index_select(0, node_idx).to(torch.int32)

    # bonds of the selected rows
    edge_ids = []
    for r, c in zip(rows, bond_counts):
        s = int(plan.graph_edge_ptr[r])
        edge_ids.append(torch.arange(s, s + int(c), dtype=torch.int64))
    edge_idx = torch.cat(edge_ids) if edge_ids else torch.zeros(0, dtype=torch.int64)
    edge_clique_global = plan.edge_clique.index_select(0, edge_idx)
    edge_clique = cmap[edge_clique_global.to(torch.int64)]
    edge_pos_u = plan.edge_pos_u.index_select(0, edge_idx).to(torch.int32)
    edge_pos_v = plan.edge_pos_v.index_select(0, edge_idx).to(torch.int32)

    return BatchPlan(
        smax=plan.smax,
        clique_scope=scope,
        clique_scope_size=sc_size,
        parent=parent,
        level=level,
        node_clique=node_clique,
        node_pos=node_pos,
        edge_clique=edge_clique,
        edge_pos_u=edge_pos_u,
        edge_pos_v=edge_pos_v,
        up_sep_child=plan.up_sep_child.index_select(0, clique_idx).to(torch.int32),
        up_sep_parent=plan.up_sep_parent.index_select(0, clique_idx).to(torch.int32),
        down_gather_parent=plan.down_gather_parent.index_select(0, clique_idx).to(torch.int32),
        sep_size=plan.sep_size.index_select(0, clique_idx).to(torch.int32),
        n_cliques=k,
        n_atoms=int(node_idx.numel()),
        n_bonds=int(edge_idx.numel()),
        n_graphs=len(rows),
        clique_graph=clique_graph,
    )


# ---------------------------------------------------------------------------
# potentials and exact inference
# ---------------------------------------------------------------------------


def edge_log_potential(raw: torch.Tensor) -> torch.Tensor:
    """[M, C, 3] -> [M, C, 2, 2] symmetric, state order (00, 01, 10, 11)."""
    if raw.dim() != 3 or raw.shape[-1] != 3:
        raise ValueError(f"expected [M,C,3], got {tuple(raw.shape)}")
    t = raw
    row0 = torch.stack([t[..., 0], t[..., 1]], dim=-1)
    row1 = torch.stack([t[..., 1], t[..., 2]], dim=-1)
    return torch.stack([row0, row1], dim=-2)


def _marginalise_up(
    agg: torch.Tensor,
    valid: torch.Tensor,
    sep_child: torch.Tensor,
    sep_parent: torch.Tensor,
) -> torch.Tensor:
    """Exact log-sum-exp message over child-only variables.

    ``agg`` [K,C,S]; ``valid`` [K,S]; ``sep_child`` maps a child state to a
    separator index; ``sep_parent`` maps a parent state to a separator index.
    The returned table is over the parent state space and is constant along
    parent-only variables.
    """
    neg = torch.finfo(agg.dtype).min
    vmask = valid.unsqueeze(1)
    maxv = agg.masked_fill(~vmask, neg).max(dim=-1, keepdim=True).values
    shifted = (agg - maxv).masked_fill(~vmask, neg)
    exp_agg = torch.exp(shifted)
    out = agg.new_zeros(agg.shape)
    out.scatter_add_(2, sep_child.to(torch.long).unsqueeze(1).expand(-1, agg.shape[1], -1), exp_agg)
    out = torch.log(out.clamp_min(torch.finfo(agg.dtype).tiny)) + maxv
    return out.gather(2, sep_parent.to(torch.long).unsqueeze(1).expand(-1, agg.shape[1], -1))


def _exact_beliefs(
    log_phi: torch.Tensor,
    log_psi: torch.Tensor,
    bp: BatchPlan,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Calibrated clique beliefs for the exact junction tree.

    Returns ``belief`` [K,C,S], ``valid`` [K,S], ``states`` [S].  The caller
    reads node/edge marginals or the root partition function from the same
    beliefs; no quantity is detached.
    """
    C = log_phi.shape[1]
    S = 1 << int(bp.smax)
    states = torch.arange(S, dtype=torch.long, device=log_phi.device)
    valid = states.unsqueeze(0) < (1 << bp.clique_scope_size).unsqueeze(1)

    pot = log_phi.new_zeros((bp.n_cliques, C, S))
    if bp.n_atoms:
        bit = (states.unsqueeze(0) >> bp.node_pos.unsqueeze(1)) & 1
        idx = bit.unsqueeze(1).expand(-1, C, -1)
        node_gather = log_phi.gather(2, idx)
        pot.index_add_(0, bp.node_clique.to(torch.long), node_gather)
    if bp.n_bonds:
        bu = (states.unsqueeze(0) >> bp.edge_pos_u.unsqueeze(1)) & 1
        bv = (states.unsqueeze(0) >> bp.edge_pos_v.unsqueeze(1)) & 1
        eidx = (2 * bu + bv).unsqueeze(1).expand(-1, C, -1)
        edge_gather = log_psi.reshape(log_psi.shape[0], C, 4).gather(2, eidx)
        pot.index_add_(0, bp.edge_clique.to(torch.long), edge_gather)

    inbox = torch.zeros_like(pot)
    msg_up = torch.zeros_like(pot)

    # upward pass: children (higher level) -> parent (lower level)
    max_level = int(bp.level.max().item()) if bp.n_cliques else 0
    for lev in range(max_level, 0, -1):
        ci = (bp.level == lev).nonzero(as_tuple=False).reshape(-1)
        if ci.numel() == 0:
            continue
        pi = bp.parent.index_select(0, ci).to(torch.long)
        agg = pot.index_select(0, ci) + inbox.index_select(0, ci)
        msg = _marginalise_up(
            agg,
            valid.index_select(0, ci),
            bp.up_sep_child.index_select(0, ci),
            bp.up_sep_parent.index_select(0, ci),
        )
        msg_up.index_copy_(0, ci, msg)
        inbox.index_add_(0, pi, msg)

    # downward pass: parent -> child, subtracting the child's own upward message
    child_idx = (bp.parent >= 0).nonzero(as_tuple=False).reshape(-1)
    parent_of_child = bp.parent.index_select(0, child_idx).to(torch.long)
    parent_level = bp.level.index_select(0, parent_of_child)
    for lev in range(0, max_level):
        sel = (parent_level == lev).nonzero(as_tuple=False).reshape(-1)
        if sel.numel() == 0:
            continue
        c = child_idx.index_select(0, sel)
        p = parent_of_child.index_select(0, sel)
        agg = (
            pot.index_select(0, p)
            + inbox.index_select(0, p)
            - msg_up.index_select(0, c)
        )
        # parent -> child is the same exact marginalisation as child -> parent,
        # with the separator maps swapped: input states are parent states,
        # output states are child states.  (A gather would silently pick one
        # assignment of the parent-only variables instead of summing them.)
        msg = _marginalise_up(
            agg,
            valid.index_select(0, p),
            bp.up_sep_parent.index_select(0, c),
            bp.up_sep_child.index_select(0, c),
        )
        inbox.index_add_(0, c, msg)

    return pot + inbox, valid, states


def exact_marginals(
    log_phi: torch.Tensor,
    log_psi: torch.Tensor,
    bp: BatchPlan,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Exact node and physical-edge marginals for a batch.

    Parameters
    ----------
    log_phi : [N, C, 2]
    log_psi : [M, C, 2, 2] symmetric
    bp : batch junction plan

    Returns
    -------
    node_marg : [N, C, 2]
    edge_marg : [M, C, 2, 2]
    """
    if bp.n_cliques == 0:
        c = log_phi.shape[1]
        return (
            torch.softmax(log_phi, dim=-1),
            torch.zeros((0, c, 2, 2), dtype=log_phi.dtype, device=log_phi.device),
        )
    C = log_phi.shape[1]
    belief, valid, states = _exact_beliefs(log_phi, log_psi, bp)

    # node marginals from the clique that owns each atom
    if bp.n_atoms:
        b = belief.index_select(0, bp.node_clique.to(torch.long))
        v = valid.index_select(0, bp.node_clique.to(torch.long))
        bit1 = ((states.unsqueeze(0) >> bp.node_pos.unsqueeze(1)) & 1).bool()
        m1 = bit1 & v
        m0 = (~bit1) & v
        neg = torch.finfo(b.dtype).min
        p1 = torch.logsumexp(b.masked_fill(~m1.unsqueeze(1), neg), dim=-1)
        p0 = torch.logsumexp(b.masked_fill(~m0.unsqueeze(1), neg), dim=-1)
        node_marg = torch.softmax(torch.stack([p0, p1], dim=-1), dim=-1)
    else:
        node_marg = torch.zeros((0, C, 2), dtype=log_phi.dtype, device=log_phi.device)

    # edge marginals from the clique that owns the earlier endpoint
    if bp.n_bonds:
        b = belief.index_select(0, bp.edge_clique.to(torch.long))
        v = valid.index_select(0, bp.edge_clique.to(torch.long))
        bu = ((states.unsqueeze(0) >> bp.edge_pos_u.unsqueeze(1)) & 1).bool()
        bv = ((states.unsqueeze(0) >> bp.edge_pos_v.unsqueeze(1)) & 1).bool()
        neg = torch.finfo(b.dtype).min
        logs = []
        for su in (0, 1):
            for sv in (0, 1):
                m = (bu == bool(su)) & (bv == bool(sv)) & v
                logs.append(torch.logsumexp(b.masked_fill(~m.unsqueeze(1), neg), dim=-1))
        edge_marg = torch.softmax(torch.stack(logs, dim=-1), dim=-1).reshape(
            bp.n_bonds, C, 2, 2
        )
    else:
        edge_marg = torch.zeros((0, C, 2, 2), dtype=log_phi.dtype, device=log_phi.device)
    return node_marg, edge_marg


def exact_logz(
    log_phi: torch.Tensor,
    log_psi: torch.Tensor,
    bp: BatchPlan,
) -> torch.Tensor:
    """Exact log partition function per graph, [G,C].

    For each connected component the calibrated root belief is the log joint
    factor over that root clique; its log-sum-exp over valid assignments is the
    component partition function.  Components of a graph are combined in log
    space.
    """
    C = log_phi.shape[1]
    out = log_phi.new_zeros((bp.n_graphs, C))
    if bp.n_cliques == 0:
        return out
    belief, valid, _ = _exact_beliefs(log_phi, log_psi, bp)
    root = (bp.parent < 0).nonzero(as_tuple=False).reshape(-1)
    b = belief.index_select(0, root)
    v = valid.index_select(0, root)
    neg = torch.finfo(b.dtype).min
    root_logz = torch.logsumexp(b.masked_fill(~v.unsqueeze(1), neg), dim=-1)
    cg = bp.clique_graph.index_select(0, root)
    for g in range(bp.n_graphs):
        sel = (cg == g).nonzero(as_tuple=False).reshape(-1)
        if sel.numel():
            # independent components multiply, so their log partition
            # functions add.
            out[g] = root_logz.index_select(0, sel).sum(dim=0)
    return out


def independent_marginals(
    log_phi: torch.Tensor,
    log_psi: torch.Tensor,
    bond_index: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-edge two-node marginals (I arm), no cross-edge consistency.

    ``bond_index`` [M,2] contains batch-local atom indices.  The returned edge
    table has state order (00, 01, 10, 11) with axes (s_u, s_v).
    """
    u = bond_index[:, 0].to(torch.long)
    v = bond_index[:, 1].to(torch.long)
    C = log_phi.shape[1]
    logits = (
        log_phi.index_select(0, u).unsqueeze(-1)
        + log_phi.index_select(0, v).unsqueeze(-2)
        + log_psi
    )  # [M,C,2,2]
    p = torch.softmax(logits.reshape(logits.shape[0], C, 4), dim=-1)
    edge_marg = p.reshape(-1, C, 2, 2)
    # local node marginals from the same two-node model
    node_marg = None
    return node_marg, edge_marg


def marginal_features(
    edge_marg: torch.Tensor,
    *,
    eps: float = 1e-8,
) -> torch.Tensor:
    """[M,C,2,2] -> [M,4*3] features in the fixed channel/feature order.

    m_e = [mu_u+mu_v-1, |mu_u-mu_v|, 4*(mu_uv - mu_u*mu_v)]

    The subtraction of mu_u*mu_v is a covariance; the factor 4 makes the
    feature equal to the centered product for Bernoulli variables.
    """
    p = edge_marg.clamp_min(eps)
    p = p / p.sum(dim=(-1, -2), keepdim=True)
    mu_u = p[..., 1, 0] + p[..., 1, 1]  # s_u = 1
    mu_v = p[..., 0, 1] + p[..., 1, 1]  # s_v = 1
    mu_uv = p[..., 1, 1]
    f0 = mu_u + mu_v - 1.0
    f1 = (mu_u - mu_v).abs()
    f2 = 4.0 * (mu_uv - mu_u * mu_v)
    return torch.stack([f0, f1, f2], dim=-1).reshape(edge_marg.shape[0], -1)


def topological_checks(plan: GlobalPlan) -> dict:
    """Cheap structural assertions on the global plan (CPU, no labels)."""
    out = {
        "n_graphs": int(plan.n_graphs),
        "n_cliques": int(plan.graph_clique_ptr[-1]),
        "n_atoms": int(plan.graph_node_ptr[-1]),
        "n_bonds": int(plan.graph_edge_ptr[-1]),
        "max_clique_size": int(plan.max_clique_size),
        "max_width": int(plan.max_width),
        "smax": int(plan.smax),
    }
    # parents are after children; root count equals number of connected
    # components (a forest, not necessarily one tree per molecule).
    parent = plan.clique_parent
    child = torch.arange(parent.numel())
    has_parent = parent >= 0
    out["cliques_with_parent"] = int(has_parent.sum())
    out["roots"] = int((~has_parent).sum())
    out["all_valid"] = bool(
        (parent[has_parent] > child[has_parent]).all().item()
        and (plan.clique_scope_size <= plan.smax).all().item()
        and (plan.clique_scope_size >= 1).all().item()
    )
    return out

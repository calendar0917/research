"""WG-ICSC-v0 — whole-graph incidence-coupled sparse coding (model half).

This module contains the scientific object only: the graph encoding, the
shared relational dictionaries, the correctness-first per-graph reference
solver, the batched scatter solver, the unrolled proximal-gradient inference,
the graph code, the prediction head and the diagnostics needed by the
pre-registration (``tracks/ksvd/notes/wg_icsc_v0_preregistration.md``).

Everything here is deliberately free of training / run-control plumbing so the
algebraic properties can be tested in isolation.

Design constraints (frozen, see the pre-registration):

* raw categorical one-hot inputs ``X_V in R^{d_V x n}``, ``X_E in R^{d_E x m}``;
  no learnable encoder, no GNN, no hand-crafted descriptor;
* one undirected chemical bond == one edge object (PyG's two directed entries
  are deduplicated and their ``edge_attr`` copies must agree);
* column-normalised incidence ``Q_ve = B_ve / sum_u B_ue`` (a normal bond gives
  ``Q_ue = Q_ve = 1/2``) used only through endpoint indexing / scatter;
* shared trainable ``D_V``, ``D_E``, ``W``; per-graph ``Z_V, Z_E >= 0`` inferred
  from zero on every forward with ``T`` shared block proximal-gradient steps.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

# ---------------------------------------------------------------------------
# frozen schema / dimensions
# ---------------------------------------------------------------------------

D_V = 21  # atom-type categories {0..20}
D_E = 3  # bond-type categories {1,2,3}
BOND_OFFSET = 1  # bond types are stored as {1,2,3}
K_V = 128
K_E = 64
T_STEPS = 8
GAMMA = 1.0
RHO = 1.0e-3
MU = 0.1
ALPHA_DIM = K_V + K_E  # 192

SPARSITY_EPS = 1.0e-8
EPS = 1.0e-8
ALPHA_EPS = 1.0e-12  # keeps d(sqrt)/d(0) finite for dead dictionary rows
NORM_EPS = 1.0e-24  # keeps d(sqrt)/d(0) finite inside group norms


# ---------------------------------------------------------------------------
# data container
# ---------------------------------------------------------------------------


@dataclass
class GraphSample:
    """One molecule as node / bond objects (bond objects are undirected)."""

    xv: torch.Tensor  # [D_V, n] float32 one-hot
    xe: torch.Tensor  # [D_E, m] float32 one-hot
    src: torch.Tensor  # [m] int64 local node id of endpoint 0
    dst: torch.Tensor  # [m] int64 local node id of endpoint 1
    y: float

    @property
    def n(self) -> int:
        return int(self.xv.shape[1])

    @property
    def m(self) -> int:
        return int(self.xe.shape[1])


@dataclass
class Batch:
    """Batched block object.  Global endpoint ids are offset per graph."""

    xv: torch.Tensor  # [D_V, N]
    xe: torch.Tensor  # [D_E, M]
    src: torch.Tensor  # [M] int64 global node id
    dst: torch.Tensor  # [M] int64 global node id
    node_graph: torch.Tensor  # [N] int64 graph id of each node
    edge_graph: torch.Tensor  # [M] int64 graph id of each bond
    n_nodes: torch.Tensor  # [B] int64
    m_edges: torch.Tensor  # [B] int64
    y: torch.Tensor  # [B] float32
    n_graphs: int

    @property
    def N(self) -> int:
        return int(self.xv.shape[1])

    @property
    def M(self) -> int:
        return int(self.xe.shape[1])

    def to(self, device: torch.device | str) -> "Batch":
        device = torch.device(device)

        def move(t: torch.Tensor) -> torch.Tensor:
            return t.to(device)

        return Batch(
            xv=move(self.xv),
            xe=move(self.xe),
            src=move(self.src),
            dst=move(self.dst),
            node_graph=move(self.node_graph),
            edge_graph=move(self.edge_graph),
            n_nodes=move(self.n_nodes),
            m_edges=move(self.m_edges),
            y=move(self.y),
            n_graphs=int(self.n_graphs),
        )


def graph_sample_from_arrays(
    atom_types: Sequence[int],
    bonds: Sequence[tuple[int, int, int]],
    y: float = 0.0,
) -> GraphSample:
    """Build a :class:`GraphSample` from raw atom types and bond triples.

    ``bonds`` are ``(u, v, bond_type)`` with ``bond_type in {1,2,3}`` and
    ``u < v`` (each bond listed once).  This is the exact object the model is
    defined on; the PyG loader is a thin adapter on top.
    """
    atom = torch.as_tensor(list(atom_types), dtype=torch.long)
    n = int(atom.numel())
    if n == 0:
        raise ValueError("empty molecule")
    if int(atom.min()) < 0 or int(atom.max()) >= D_V:
        raise ValueError(f"atom type out of range: {atom.tolist()}")
    xv = F.one_hot(atom, D_V).to(torch.float32).t().contiguous()

    src: list[int] = []
    dst: list[int] = []
    types: list[int] = []
    for u, v, b in bonds:
        if u == v:
            raise ValueError("self-loop bond is not a ZINC chemical bond")
        lo, hi = (u, v) if u < v else (v, u)
        if not (0 <= lo < hi < n):
            raise ValueError(f"bond endpoint out of range: {(u, v)}")
        if int(b) < 1 or int(b) > D_E:
            raise ValueError(f"bond type out of range: {b}")
        src.append(int(lo))
        dst.append(int(hi))
        types.append(int(b) - BOND_OFFSET)
    if src:
        order = np.lexsort((np.asarray(dst), np.asarray(src)))
        src = [src[i] for i in order]
        dst = [dst[i] for i in order]
        types = [types[i] for i in order]
    xe = F.one_hot(torch.as_tensor(types, dtype=torch.long), D_E).to(torch.float32).t().contiguous()
    if xe.numel() == 0:
        xe = torch.zeros(D_E, 0, dtype=torch.float32)
    return GraphSample(
        xv=xv,
        xe=xe,
        src=torch.as_tensor(src, dtype=torch.long),
        dst=torch.as_tensor(dst, dtype=torch.long),
        y=float(y),
    )


def graph_sample_from_pyg(data: Any) -> GraphSample:
    """Adapter from a PyG ``Data`` object (raw ZINC semantics).

    The two directed entries of every undirected bond are deduplicated; both
    ``edge_attr`` copies must agree or a :class:`RuntimeError` is raised (never
    guessed).
    """
    atom = data.x.reshape(-1).to(torch.long)
    edge_index = data.edge_index.to(torch.long)
    edge_attr = data.edge_attr.reshape(-1).to(torch.long)
    seen: dict[tuple[int, int], int] = {}
    for k in range(int(edge_index.shape[1])):
        u = int(edge_index[0, k])
        v = int(edge_index[1, k])
        key = (u, v) if u <= v else (v, u)
        val = int(edge_attr[k])
        prev = seen.get(key)
        if prev is None:
            seen[key] = val
        elif prev != val:
            raise RuntimeError(
                f"inconsistent directed bond attributes for {key}: {prev} vs {val}"
            )
    bonds = sorted((u, v, b) for (u, v), b in seen.items())
    y = float(data.y.reshape(-1)[0]) if hasattr(data, "y") else 0.0
    return graph_sample_from_arrays(atom.tolist(), bonds, y)


def collate(samples: Sequence[GraphSample]) -> Batch:
    if not samples:
        raise ValueError("cannot collate an empty sequence")
    xs_v: list[torch.Tensor] = []
    xs_e: list[torch.Tensor] = []
    srcs: list[torch.Tensor] = []
    dsts: list[torch.Tensor] = []
    node_graphs: list[torch.Tensor] = []
    edge_graphs: list[torch.Tensor] = []
    n_nodes: list[int] = []
    m_edges: list[int] = []
    ys: list[float] = []
    node_off = 0
    for g, sample in enumerate(samples):
        n = sample.n
        m = sample.m
        xs_v.append(sample.xv)
        xs_e.append(sample.xe)
        srcs.append(sample.src + node_off)
        dsts.append(sample.dst + node_off)
        node_graphs.append(torch.full((n,), g, dtype=torch.long))
        edge_graphs.append(torch.full((m,), g, dtype=torch.long))
        n_nodes.append(n)
        m_edges.append(m)
        ys.append(sample.y)
        node_off += n
    return Batch(
        xv=torch.cat(xs_v, dim=1).to(torch.float32),
        xe=torch.cat(xs_e, dim=1).to(torch.float32),
        src=torch.cat(srcs).to(torch.long) if srcs else torch.zeros(0, dtype=torch.long),
        dst=torch.cat(dsts).to(torch.long) if dsts else torch.zeros(0, dtype=torch.long),
        node_graph=torch.cat(node_graphs).to(torch.long),
        edge_graph=torch.cat(edge_graphs).to(torch.long),
        n_nodes=torch.as_tensor(n_nodes, dtype=torch.long),
        m_edges=torch.as_tensor(m_edges, dtype=torch.long),
        y=torch.as_tensor(ys, dtype=torch.float32),
        n_graphs=len(samples),
    )


# ---------------------------------------------------------------------------
# small differentiable scatter helpers
# ---------------------------------------------------------------------------


def _index_add_rows(target: torch.Tensor, index: torch.Tensor, values: torch.Tensor) -> torch.Tensor:
    """``target[:, index[i]] += values[:, i]`` (differentiable in values)."""
    return target.index_add(1, index, values)


def _graph_sum_from_nodes(values: torch.Tensor, node_graph: torch.Tensor, n_graphs: int) -> torch.Tensor:
    """Sum ``values[k, j]`` over nodes ``j`` belonging to each graph -> [K, B]."""
    out = torch.zeros(values.shape[0], n_graphs, dtype=values.dtype, device=values.device)
    return out.index_add(1, node_graph, values)


def _graph_sum_scalar(values: torch.Tensor, group: torch.Tensor, n_graphs: int) -> torch.Tensor:
    """Sum 1-D ``values[j]`` over items belonging to each group -> [B]."""
    out = torch.zeros(n_graphs, dtype=values.dtype, device=values.device)
    return out.index_add(0, group, values)


def _endpoint_gather(values: torch.Tensor, src: torch.Tensor, dst: torch.Tensor) -> torch.Tensor:
    """Mean of the two endpoint columns of every bond: ``(z_u + z_v)/2``."""
    return 0.5 * (values[:, src] + values[:, dst])


# ---------------------------------------------------------------------------
# lower-level objective / diagnostics
# ---------------------------------------------------------------------------


def _per_graph_scales(batch: Batch) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Per-node ``n``, per-edge ``m`` and per-graph ``q_bound``."""
    device = batch.xv.device
    n_of_node = batch.n_nodes.to(device)[batch.node_graph].to(batch.xv.dtype)
    m_of_edge = batch.m_edges.to(device)[batch.edge_graph].to(batch.xv.dtype)
    ones = torch.ones(batch.M, dtype=batch.xv.dtype, device=device)
    deg = torch.zeros(batch.N, dtype=batch.xv.dtype, device=device)
    deg = deg.index_add(0, batch.src, ones).index_add(0, batch.dst, ones)
    q_bound = torch.zeros(batch.n_graphs, dtype=batch.xv.dtype, device=device)
    q_bound = q_bound.scatter_reduce(0, batch.node_graph, deg / 2.0, reduce="amax")
    return n_of_node, m_of_edge, q_bound


def lower_energy(
    batch: Batch,
    d_v: torch.Tensor,
    d_e: torch.Tensor,
    w: torch.Tensor,
    z_v: torch.Tensor,
    z_e: torch.Tensor,
    *,
    gamma: float = GAMMA,
    rho: float = RHO,
    lambda1: float = 0.0,
    lambda_g: float = 0.0,
) -> torch.Tensor:
    """Per-graph lower-level objective ``E_G`` (vectorised over the batch)."""
    device = batch.xv.device
    b = batch.n_graphs
    n_of_node = batch.n_nodes.to(device)[batch.node_graph].to(torch.float32)
    m_of_edge = batch.m_edges.to(device)[batch.edge_graph].to(torch.float32)
    n_b = batch.n_nodes.to(device).to(torch.float32)
    m_b = batch.m_edges.to(device).to(torch.float32)

    rec_v = d_v @ z_v - batch.xv
    v_err = _graph_sum_scalar(rec_v.pow(2).sum(dim=0), batch.node_graph, b) / (2.0 * n_b)
    rec_e = d_e @ z_e - batch.xe
    e_err = _graph_sum_scalar(rec_e.pow(2).sum(dim=0), batch.edge_graph, b) / (2.0 * m_b)

    total = v_err + e_err
    if gamma > 0.0:
        z_avg = _endpoint_gather(z_v, batch.src, batch.dst)
        diff = w @ z_avg - z_e
        c_err = _graph_sum_scalar(diff.pow(2).sum(dim=0), batch.edge_graph, b) / (2.0 * m_b)
        total = total + gamma * c_err

    l2_v = _graph_sum_scalar(z_v.pow(2).sum(dim=0), batch.node_graph, b)
    l2_e = _graph_sum_scalar(z_e.pow(2).sum(dim=0), batch.edge_graph, b)
    total = total + 0.5 * rho * (l2_v / n_b + l2_e / m_b)

    l1_v = _graph_sum_scalar(z_v.abs().sum(dim=0), batch.node_graph, b)
    l1_e = _graph_sum_scalar(z_e.abs().sum(dim=0), batch.edge_graph, b)
    total = total + lambda1 * (l1_v / n_b + l1_e / m_b)

    row_v = torch.sqrt(_graph_sum_from_nodes(z_v.pow(2), batch.node_graph, b) + NORM_EPS)
    row_e = torch.sqrt(_graph_sum_from_nodes(z_e.pow(2), batch.edge_graph, b) + NORM_EPS)
    total = total + lambda_g * (
        row_v.sum(dim=0) / torch.sqrt(n_b) + row_e.sum(dim=0) / torch.sqrt(m_b)
    )
    del n_of_node, m_of_edge
    return total


# ---------------------------------------------------------------------------
# solver
# ---------------------------------------------------------------------------


@dataclass
class SolveOutput:
    z_v: torch.Tensor
    z_e: torch.Tensor
    alpha: torch.Tensor  # [B, ALPHA_DIM]
    per_graph: dict[str, torch.Tensor] = field(default_factory=dict)
    step_stats: list[dict[str, float]] = field(default_factory=list)
    grad_ratio_stats: list[dict[str, float]] = field(default_factory=list)


def normalize_columns(dictionary: torch.Tensor) -> torch.Tensor:
    return dictionary / (dictionary.norm(dim=0, keepdim=True) + EPS)


def _step_sizes(
    batch: Batch,
    d_v: torch.Tensor,
    d_e: torch.Tensor,
    w: torch.Tensor,
    *,
    gamma: float,
    rho: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    device = batch.xv.device
    n_of_node, m_of_edge, q_bound = _per_graph_scales(batch)
    n_b = batch.n_nodes.to(device).to(torch.float32)
    m_b = batch.m_edges.to(device).to(torch.float32)
    s_v = torch.linalg.matrix_norm(d_v, ord=2).detach()
    s_e = torch.linalg.matrix_norm(d_e, ord=2).detach()
    s_w = torch.linalg.matrix_norm(w, ord=2).detach()
    l_v = s_v.pow(2) / n_b + gamma * s_w.pow(2) * q_bound / m_b + rho / n_b
    l_e = s_e.pow(2) / m_b + gamma / m_b + rho / m_b
    eta_v = 0.9 / (l_v + EPS)
    eta_e = 0.9 / (l_e + EPS)
    eta_v_node = eta_v[batch.node_graph]
    eta_e_edge = eta_e[batch.edge_graph]
    return eta_v_node, eta_e_edge, n_of_node, m_of_edge


def _group_shrink(
    u: torch.Tensor,
    groups: torch.Tensor,
    n_graphs: int,
    *,
    eta: torch.Tensor,
    size_per: torch.Tensor,
    lambda_g: float,
) -> torch.Tensor:
    """Exact row-group shrink, with the group norm taken **within each graph**."""
    if lambda_g <= 0.0:
        return u
    per_graph_sq = _graph_sum_from_nodes(u.pow(2), groups, n_graphs)  # [K, B]
    row_norm = torch.sqrt(per_graph_sq + NORM_EPS)  # [K, B]
    row_norm_nodes = row_norm[:, groups]  # [K, N]
    threshold = (eta * (lambda_g / torch.sqrt(size_per))).unsqueeze(0)  # [1, N]
    return F.relu(1.0 - threshold / (row_norm_nodes + EPS)) * u


def solve_batched(
    d_v: torch.Tensor,
    d_e: torch.Tensor,
    w: torch.Tensor,
    batch: Batch,
    *,
    gamma: float = GAMMA,
    rho: float = RHO,
    lambda1: float = 0.0,
    lambda_g: float = 0.0,
    t_steps: int = T_STEPS,
    record: bool = False,
) -> SolveOutput:
    """Batched unrolled block proximal-gradient solver.

    ``d_v`` / ``d_e`` must already be column-normalised; ``w`` is used raw.
    Every graph starts from ``Z_V = Z_E = 0``.
    """
    device = batch.xv.device
    b = batch.n_graphs
    dtype = batch.xv.dtype
    n_of_node, m_of_edge, _ = _per_graph_scales(batch)
    eta_v_node, eta_e_edge, _, _ = _step_sizes(
        batch, d_v, d_e, w, gamma=gamma, rho=rho
    )

    z_v = torch.zeros(K_V, batch.N, dtype=dtype, device=device)
    z_e = torch.zeros(K_E, batch.M, dtype=dtype, device=device)

    step_stats: list[dict[str, float]] = []
    grad_ratio_stats: list[dict[str, float]] = []
    if record:
        step_stats.append(_step_record(batch, d_v, d_e, w, z_v, z_e, gamma, rho, lambda1, lambda_g, -1))

    for t in range(int(t_steps)):
        # ---- node block ----
        g_attr = (d_v.t() @ (d_v @ z_v - batch.xv)) / n_of_node.unsqueeze(0)
        g_v = g_attr + (rho / n_of_node).unsqueeze(0) * z_v
        g_comp_norm = 0.0
        if gamma > 0.0:
            z_avg = _endpoint_gather(z_v, batch.src, batch.dst)
            diff = w @ z_avg - z_e
            wdiff = w.t() @ diff
            weight = 0.5 * gamma / m_of_edge
            g_comp = torch.zeros_like(z_v)
            g_comp = g_comp.index_add(1, batch.src, wdiff * weight.unsqueeze(0))
            g_comp = g_comp.index_add(1, batch.dst, wdiff * weight.unsqueeze(0))
            g_v = g_v + g_comp
            g_comp_norm = float(g_comp.norm())
        u = z_v - eta_v_node.unsqueeze(0) * g_v
        u = F.relu(u - (eta_v_node * lambda1 / n_of_node).unsqueeze(0))
        z_v = _group_shrink(
            u, batch.node_graph, b, eta=eta_v_node, size_per=n_of_node, lambda_g=lambda_g
        )
        if record:
            grad_ratio_stats.append(
                {
                    "step": float(t),
                    "side": 0.0,
                    "g_attr": float(g_attr.norm()),
                    "g_comp": g_comp_norm,
                }
            )

        # ---- edge block (uses the freshly updated Z_V) ----
        g_attr_e = (d_e.t() @ (d_e @ z_e - batch.xe)) / m_of_edge.unsqueeze(0)
        g_e = g_attr_e + (rho / m_of_edge).unsqueeze(0) * z_e
        g_comp_e_norm = 0.0
        if gamma > 0.0:
            z_avg = _endpoint_gather(z_v, batch.src, batch.dst)
            diff_e = (z_e - w @ z_avg) * (gamma / m_of_edge).unsqueeze(0)
            g_e = g_e + diff_e
            g_comp_e_norm = float(diff_e.norm())
        u = z_e - eta_e_edge.unsqueeze(0) * g_e
        u = F.relu(u - (eta_e_edge * lambda1 / m_of_edge).unsqueeze(0))
        z_e = _group_shrink(
            u, batch.edge_graph, b, eta=eta_e_edge, size_per=m_of_edge, lambda_g=lambda_g
        )
        if record:
            step_stats.append(
                _step_record(batch, d_v, d_e, w, z_v, z_e, gamma, rho, lambda1, lambda_g, t)
            )
            grad_ratio_stats.append(
                {
                    "step": float(t),
                    "side": 1.0,
                    "g_attr": float(g_attr_e.norm()),
                    "g_comp": g_comp_e_norm,
                }
            )

    alpha, per_graph = graph_code(z_v, z_e, batch)
    return SolveOutput(
        z_v=z_v,
        z_e=z_e,
        alpha=alpha,
        per_graph=per_graph,
        step_stats=step_stats,
        grad_ratio_stats=grad_ratio_stats,
    )


def graph_code(
    z_v: torch.Tensor, z_e: torch.Tensor, batch: Batch
) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
    """Per-graph row-L2 amplitudes ``alpha_G = [alpha^V ; alpha^E] in R^192``."""
    b = batch.n_graphs
    node_sq = _graph_sum_from_nodes(z_v.pow(2), batch.node_graph, b)  # [K_V, B]
    edge_sq = _graph_sum_from_nodes(z_e.pow(2), batch.edge_graph, b)  # [K_E, B]
    alpha = torch.cat(
        [torch.sqrt(node_sq + ALPHA_EPS).t(), torch.sqrt(edge_sq + ALPHA_EPS).t()], dim=1
    )
    return alpha, {"node_sq": node_sq, "edge_sq": edge_sq}


def _step_record(
    batch: Batch,
    d_v: torch.Tensor,
    d_e: torch.Tensor,
    w: torch.Tensor,
    z_v: torch.Tensor,
    z_e: torch.Tensor,
    gamma: float,
    rho: float,
    lambda1: float,
    lambda_g: float,
    step: int,
) -> dict[str, float]:
    n_b = batch.n_nodes.to(batch.xv.device).to(torch.float32)
    m_b = batch.m_edges.to(batch.xv.device).to(torch.float32)
    b = batch.n_graphs
    rec_v = (d_v @ z_v - batch.xv).pow(2).sum(dim=0)
    node_recon = _graph_sum_scalar(rec_v, batch.node_graph, b) / (2.0 * n_b)
    rec_e = (d_e @ z_e - batch.xe).pow(2).sum(dim=0)
    edge_recon = _graph_sum_scalar(rec_e, batch.edge_graph, b) / (2.0 * m_b)
    comp = torch.zeros(b, device=z_v.device)
    if gamma > 0.0:
        z_avg = _endpoint_gather(z_v, batch.src, batch.dst)
        diff = (w @ z_avg - z_e).pow(2).sum(dim=0)
        comp = _graph_sum_scalar(diff, batch.edge_graph, b) / (2.0 * m_b)
    energy = lower_energy(
        batch, d_v, d_e, w, z_v, z_e, gamma=gamma, rho=rho, lambda1=lambda1, lambda_g=lambda_g
    )
    node_active = (torch.sqrt(_graph_sum_from_nodes(z_v.pow(2), batch.node_graph, b)) > SPARSITY_EPS).to(
        torch.float32
    ).mean()
    edge_active = (torch.sqrt(_graph_sum_from_nodes(z_e.pow(2), batch.edge_graph, b)) > SPARSITY_EPS).to(
        torch.float32
    ).mean()
    return {
        "step": float(step),
        "node_recon": float(node_recon.mean()),
        "edge_recon": float(edge_recon.mean()),
        "composition": float(comp.mean()),
        "energy": float(energy.mean()),
        "node_active_frac": float(node_active),
        "edge_active_frac": float(edge_active),
    }


# ---------------------------------------------------------------------------
# correctness-first per-graph reference solver
# ---------------------------------------------------------------------------


def dense_incidence(sample: GraphSample) -> torch.Tensor:
    """Dense column-normalised incidence ``Q in R^{n x m}``."""
    q = torch.zeros(sample.n, sample.m, dtype=torch.float32)
    for e in range(sample.m):
        q[int(sample.src[e]), e] += 0.5
        q[int(sample.dst[e]), e] += 0.5
    return q


def solve_reference(
    d_v: torch.Tensor,
    d_e: torch.Tensor,
    w: torch.Tensor,
    sample: GraphSample,
    *,
    gamma: float = GAMMA,
    rho: float = RHO,
    lambda1: float = 0.0,
    lambda_g: float = 0.0,
    t_steps: int = T_STEPS,
    record: bool = False,
) -> SolveOutput:
    """Single-graph dense reference solver (exactly the mathematical model)."""
    n = sample.n
    m = sample.m
    q = dense_incidence(sample)
    s_v = torch.linalg.matrix_norm(d_v, ord=2).detach()
    s_e = torch.linalg.matrix_norm(d_e, ord=2).detach()
    s_w = torch.linalg.matrix_norm(w, ord=2).detach()
    q_bound = float(q.sum(dim=1).max()) if m > 0 else 0.0
    l_v = float(s_v) ** 2 / n + gamma * float(s_w) ** 2 * q_bound / max(m, 1) + rho / n
    l_e = float(s_e) ** 2 / max(m, 1) + gamma / max(m, 1) + rho / max(m, 1)
    eta_v = 0.9 / (l_v + EPS)
    eta_e = 0.9 / (l_e + EPS)

    z_v = torch.zeros(K_V, n, dtype=torch.float32)
    z_e = torch.zeros(K_E, m, dtype=torch.float32)
    step_stats: list[dict[str, float]] = []
    if record:
        step_stats.append(
            _ref_step_record(sample, d_v, d_e, w, z_v, z_e, gamma, rho, lambda1, lambda_g, -1)
        )

    for t in range(int(t_steps)):
        g_v = (d_v.t() @ (d_v @ z_v - sample.xv)) / n
        if gamma > 0.0:
            g_v = g_v + (gamma / max(m, 1)) * (w.t() @ ((w @ z_v @ q - z_e) @ q.t()))
        g_v = g_v + (rho / n) * z_v
        u = z_v - eta_v * g_v
        u = F.relu(u - eta_v * lambda1 / n)
        z_v = _ref_group_shrink(u, eta_v, lambda_g, math.sqrt(n))

        g_e = (d_e.t() @ (d_e @ z_e - sample.xe)) / max(m, 1)
        if gamma > 0.0:
            g_e = g_e + (gamma / max(m, 1)) * (z_e - w @ z_v @ q)
        g_e = g_e + (rho / max(m, 1)) * z_e
        u = z_e - eta_e * g_e
        u = F.relu(u - eta_e * lambda1 / max(m, 1))
        z_e = _ref_group_shrink(u, eta_e, lambda_g, math.sqrt(max(m, 1)))
        if record:
            step_stats.append(
                _ref_step_record(sample, d_v, d_e, w, z_v, z_e, gamma, rho, lambda1, lambda_g, t)
            )

    alpha_v = z_v.norm(dim=1)
    alpha_e = z_e.norm(dim=1)
    alpha = torch.cat([alpha_v, alpha_e]).unsqueeze(0)
    return SolveOutput(z_v=z_v, z_e=z_e, alpha=alpha, step_stats=step_stats)


def _ref_group_shrink(u: torch.Tensor, eta: float, lambda_g: float, scale: float) -> torch.Tensor:
    if lambda_g <= 0.0:
        return u
    row_norm = u.norm(dim=1, keepdim=True)
    return F.relu(1.0 - (eta * lambda_g / scale) / (row_norm + EPS)) * u


def _ref_step_record(
    sample: GraphSample,
    d_v: torch.Tensor,
    d_e: torch.Tensor,
    w: torch.Tensor,
    z_v: torch.Tensor,
    z_e: torch.Tensor,
    gamma: float,
    rho: float,
    lambda1: float,
    lambda_g: float,
    step: int,
) -> dict[str, float]:
    n = sample.n
    m = sample.m
    node_recon = float((d_v @ z_v - sample.xv).pow(2).sum()) / (2.0 * n)
    edge_recon = float((d_e @ z_e - sample.xe).pow(2).sum()) / (2.0 * max(m, 1))
    comp = 0.0
    if gamma > 0.0:
        q = dense_incidence(sample)
        comp = gamma * float((w @ z_v @ q - z_e).pow(2).sum()) / (2.0 * max(m, 1))
    energy = (
        node_recon
        + edge_recon
        + comp
        + 0.5 * rho * (float(z_v.pow(2).sum()) / n + float(z_e.pow(2).sum()) / max(m, 1))
        + lambda1 * (float(z_v.abs().sum()) / n + float(z_e.abs().sum()) / max(m, 1))
        + lambda_g
        * (
            float(z_v.norm(dim=1).sum()) / math.sqrt(n)
            + float(z_e.norm(dim=1).sum()) / math.sqrt(max(m, 1))
        )
    )
    return {
        "step": float(step),
        "node_recon": node_recon,
        "edge_recon": edge_recon,
        "composition": comp,
        "energy": energy,
        "node_active_frac": float((z_v.norm(dim=1) > SPARSITY_EPS).to(torch.float32).mean()),
        "edge_active_frac": float((z_e.norm(dim=1) > SPARSITY_EPS).to(torch.float32).mean()),
    }


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class WGICSC(nn.Module):
    """Whole-graph incidence-coupled sparse coder + prediction head."""

    def __init__(
        self,
        *,
        seed: int = 0,
        gamma: float = GAMMA,
        rho: float = RHO,
        lambda1: float = 0.0,
        lambda_g: float = 0.0,
        t_steps: int = T_STEPS,
        init_std: float = 0.5,
    ) -> None:
        super().__init__()
        generator = torch.Generator().manual_seed(int(seed))
        self.d_v = nn.Parameter(torch.randn(D_V, K_V, generator=generator) * init_std)
        self.d_e = nn.Parameter(torch.randn(D_E, K_E, generator=generator) * init_std)
        self.w = nn.Parameter(torch.randn(K_E, K_V, generator=generator) * (1.0 / math.sqrt(K_V)))
        self.head = nn.Sequential(
            nn.Linear(ALPHA_DIM, 64),
            nn.SiLU(),
            nn.Linear(64, 1),
        )
        self.gamma = float(gamma)
        self.rho = float(rho)
        self.lambda1 = float(lambda1)
        self.lambda_g = float(lambda_g)
        self.t_steps = int(t_steps)

    def normalized_dictionaries(self) -> tuple[torch.Tensor, torch.Tensor]:
        return normalize_columns(self.d_v), normalize_columns(self.d_e)

    def solve(self, batch: Batch, *, t_steps: int | None = None, record: bool = False) -> SolveOutput:
        d_v, d_e = self.normalized_dictionaries()
        return solve_batched(
            d_v,
            d_e,
            self.w,
            batch,
            gamma=self.gamma,
            rho=self.rho,
            lambda1=self.lambda1,
            lambda_g=self.lambda_g,
            t_steps=self.t_steps if t_steps is None else t_steps,
            record=record,
        )

    def fit_loss(self, batch: Batch, output: SolveOutput) -> torch.Tensor:
        d_v, d_e = self.normalized_dictionaries()
        b = batch.n_graphs
        n_b = batch.n_nodes.to(batch.xv.device).to(torch.float32)
        m_b = batch.m_edges.to(batch.xv.device).to(torch.float32)
        rec_v = (d_v @ output.z_v - batch.xv).pow(2).sum(dim=0)
        per = _graph_sum_scalar(rec_v, batch.node_graph, b) / (2.0 * n_b)
        rec_e = (d_e @ output.z_e - batch.xe).pow(2).sum(dim=0)
        per = per + _graph_sum_scalar(rec_e, batch.edge_graph, b) / (2.0 * m_b)
        if self.gamma > 0.0:
            z_avg = _endpoint_gather(output.z_v, batch.src, batch.dst)
            diff = (self.w @ z_avg - output.z_e).pow(2).sum(dim=0)
            per = per + self.gamma * _graph_sum_scalar(diff, batch.edge_graph, b) / (2.0 * m_b)
        return per.mean()

    def forward(self, batch: Batch, *, record: bool = False) -> tuple[torch.Tensor, SolveOutput]:
        output = self.solve(batch, record=record)
        prediction = self.head(output.alpha).view(-1)
        return prediction, output

    def parameter_groups(self) -> dict[str, int]:
        return {
            "D_V": int(self.d_v.numel()),
            "D_E": int(self.d_e.numel()),
            "W": int(self.w.numel()),
            "head": int(sum(p.numel() for p in self.head.parameters())),
            "total": int(sum(p.numel() for p in self.parameters())),
        }


# ---------------------------------------------------------------------------
# support / calibration metrics
# ---------------------------------------------------------------------------


def support_metrics(output: SolveOutput, batch: Batch) -> dict[str, float]:
    """Support / element-sparsity metrics for one solved batch.

    A dictionary row is *active* in a graph when its within-graph row-L2 norm
    exceeds ``SPARSITY_EPS`` (tested on the squared norm so no ``+EPS`` can
    inflate a dead row above the threshold).
    """
    node_sq = output.per_graph["node_sq"]  # [K_V, B]
    edge_sq = output.per_graph["edge_sq"]  # [K_E, B]
    b = batch.n_graphs
    node_active = node_sq > SPARSITY_EPS**2  # [K_V, B] bool
    edge_active = edge_sq > SPARSITY_EPS**2  # [K_E, B] bool
    node_active_frac = node_active.to(torch.float32).mean(dim=0)  # [B]
    edge_active_frac = edge_active.to(torch.float32).mean(dim=0)
    whole_active = (node_active.sum(0).to(torch.float32) + edge_active.sum(0).to(torch.float32)) / float(
        K_V + K_E
    )

    m_b = batch.m_edges.to(batch.xv.device).to(torch.float32)
    n_b = batch.n_nodes.to(batch.xv.device).to(torch.float32)
    node_nz = (
        _graph_sum_from_nodes((output.z_v > SPARSITY_EPS).to(torch.float32), batch.node_graph, b) / n_b
    )  # [K_V, B] fraction of the graph's nodes that are nonzero in each row
    edge_nz = (
        _graph_sum_from_nodes((output.z_e > SPARSITY_EPS).to(torch.float32), batch.edge_graph, b) / m_b
    )
    node_selected = node_nz[node_active]
    edge_selected = edge_nz[edge_active]
    parts = [x for x in (node_selected, edge_selected) if x.numel() > 0]
    element_active = float(torch.cat(parts).mean()) if parts else float("nan")

    nz_total = float((output.z_v > SPARSITY_EPS).sum()) + float((output.z_e > SPARSITY_EPS).sum())
    element_overall = nz_total / float(output.z_v.numel() + output.z_e.numel() + EPS)

    def _q(x: torch.Tensor) -> tuple[float, float, float]:
        if x.numel() == 0:
            return (float("nan"),) * 3
        return (float(x.mean()), float(x.median()), float(x.max()))

    wm, wmd, wmx = _q(whole_active)
    return {
        "node_active_row_frac_mean": float(node_active_frac.mean()),
        "node_active_row_frac_median": float(node_active_frac.median()),
        "node_active_row_frac_p90": float(node_active_frac.quantile(0.90)),
        "edge_active_row_frac_mean": float(edge_active_frac.mean()),
        "edge_active_row_frac_median": float(edge_active_frac.median()),
        "edge_active_row_frac_p90": float(edge_active_frac.quantile(0.90)),
        "whole_active_row_frac_mean": wm,
        "whole_active_row_frac_median": wmd,
        "whole_active_row_frac_max": wmx,
        "element_nonzero_frac": element_overall,
        "element_nonzero_frac_in_active_rows": element_active,
        "dead_node_atoms": float((node_active.sum(dim=1) == 0).sum()),
        "dead_edge_atoms": float((edge_active.sum(dim=1) == 0).sum()),
    }


def dictionary_stats(*, num_graphs: int, d_v: torch.Tensor, d_e: torch.Tensor, w: torch.Tensor) -> dict[str, float]:
    """Coherence / rank / norm diagnostics for the shared dictionaries."""
    del num_graphs
    out: dict[str, float] = {}
    for name, mat in (("D_V", d_v), ("D_E", d_e), ("W", w)):
        m = mat.detach()
        if m.shape[0] > 1 and m.shape[1] > 1:
            gram = m.t() @ m
            norm = torch.linalg.matrix_norm(m, ord=2)
            out[f"{name}_spectral_norm"] = float(norm)
            out[f"{name}_frobenius_norm"] = float(m.norm())
            sv = torch.linalg.svdvals(m)
            out[f"{name}_effective_rank"] = float((sv.sum() ** 2) / (sv.pow(2).sum() + EPS))
        if name in ("D_V", "D_E"):
            cols = m / (m.norm(dim=0, keepdim=True) + EPS)
            gram = cols.t() @ cols
            k = gram.shape[0]
            off = gram - torch.eye(k, dtype=gram.dtype, device=gram.device)
            out[f"{name}_coherence_mean"] = float(off.abs().sum() / max(k * (k - 1), 1))
            out[f"{name}_coherence_max"] = float(off.abs().max()) if k > 1 else 0.0
    return out

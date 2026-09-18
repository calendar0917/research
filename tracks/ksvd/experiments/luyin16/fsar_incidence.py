"""FSAR-C1: persistent node/edge S-A-B objects + a real incidence processor.

This is the model / feature half of the **parameter-matched incidence
processor** round.  It contains no training infrastructure (that lives in
``zinc_fsar_incidence.py``) so that the algebraic properties demanded by the
pre-registration (``notes/fsar_incidence_processor_preregistration.md``) can be
tested in isolation.

It reuses, without modification:

* the frozen radius-2 pure-topology node coordinate ``phi_v in R^65``
  (:mod:`fsar_r2_ar0`),
* the symmetric, chemistry-free edge role
  ``psi_e = [phi_u + phi_v, |phi_u - phi_v|] in R^130``
  (:mod:`fsar_r2_ar0_edge`),
* the attribute one-hots ``q_v = onehot(atom_type) in R^28`` and
  ``r_e = onehot(bond_type) in R^4``,
* the whole-graph attribute marginal ``A(G) in R^64``.

Two variants share the **exact same modules and parameter count**:

* ``C1`` (incidence) -- the edge update reads the true endpoints and the node
  update aggregates over the true incident edges:
  ``h_u + h_v`` / ``|h_u - h_v|`` for edges, degree-normalised mean of
  ``G(g_e, h_w)`` for nodes.
* ``C0`` (bag) -- the same modules, but real incidence is destroyed: every edge
  sees only its graph's global node mean, and every node in a graph receives the
  same global edge-message mean.  This is the parameter-exact incidence-free
  control.

No new structural feature family is used: no RRWP, no LapPE, no C4/C5 or
homomorphism counts, no attention positional encoding, no radius > 2, no
learned motif vocabulary, no typed-token identity lookup, no pair state and no
triangles.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0_edge as edge

# ---------------------------------------------------------------------------
# frozen schema / dimensions
# ---------------------------------------------------------------------------

PHI_DIM = r2.PHI_DIM  # 65
EDGE_ROLE_DIM = edge.EDGE_ROLE_DIM  # 130
ATOM_CATEGORIES = r2.ATOM_CATEGORIES  # 28
BOND_CATEGORIES = r2.BOND_CATEGORIES  # 4
A_DIM = r2.A_DIM  # 64

HIDDEN = 48
PROCESSOR_HIDDEN = 2 * HIDDEN  # 96
ROUNDS = 4
READOUT_HIDDEN = 64
RESIDUAL_INIT = 0.05
READOUT_INPUT_DIM = 4 * HIDDEN + A_DIM + 2  # 258

MODELS = ("C1", "C0")
INCIDENCE_MODELS = ("C1",)
BAG_MODELS = ("C0",)

_EPS = 1.0e-6


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


@dataclass
class IncidenceBatch:
    """Batched persistent-object input.

    ``edge_index`` holds *global* node ids (offset per graph) for the two
    endpoints of every undirected edge, in the same order as ``psi`` / ``r``.
    """

    phi: torch.Tensor  # [N, 65]
    q: torch.Tensor  # [N, 28]
    node_graph: torch.Tensor  # [N]
    psi: torch.Tensor  # [M, 130]
    r: torch.Tensor  # [M, 4]
    edge_index: torch.Tensor  # [2, M] int64, global node ids
    edge_graph: torch.Tensor  # [M]
    A: torch.Tensor  # [B, 64]
    n_nodes: torch.Tensor  # [B]
    n_edges: torch.Tensor  # [B]
    y: torch.Tensor  # [B]
    n_graphs: int

    def to(self, device: torch.device | str) -> "IncidenceBatch":
        device = torch.device(device)
        return IncidenceBatch(
            phi=self.phi.to(device),
            q=self.q.to(device),
            node_graph=self.node_graph.to(device),
            psi=self.psi.to(device),
            r=self.r.to(device),
            edge_index=self.edge_index.to(device),
            edge_graph=self.edge_graph.to(device),
            A=self.A.to(device),
            n_nodes=self.n_nodes.to(device),
            n_edges=self.n_edges.to(device),
            y=self.y.to(device),
            n_graphs=int(self.n_graphs),
        )

    def with_r(self, r: torch.Tensor) -> "IncidenceBatch":
        if r.shape != self.r.shape:
            raise RuntimeError("replacement bond one-hot shape mismatch")
        return IncidenceBatch(
            phi=self.phi,
            q=self.q,
            node_graph=self.node_graph,
            psi=self.psi,
            r=r,
            edge_index=self.edge_index,
            edge_graph=self.edge_graph,
            A=self.A,
            n_nodes=self.n_nodes,
            n_edges=self.n_edges,
            y=self.y,
            n_graphs=int(self.n_graphs),
        )

    def with_q(self, q: torch.Tensor) -> "IncidenceBatch":
        if q.shape != self.q.shape:
            raise RuntimeError("replacement atom one-hot shape mismatch")
        return IncidenceBatch(
            phi=self.phi,
            q=q,
            node_graph=self.node_graph,
            psi=self.psi,
            r=self.r,
            edge_index=self.edge_index,
            edge_graph=self.edge_graph,
            A=self.A,
            n_nodes=self.n_nodes,
            n_edges=self.n_edges,
            y=self.y,
            n_graphs=int(self.n_graphs),
        )


def collate_incidence_molecules(
    molecules: Sequence[edge.EdgeMoleculeFeatures],
    atom_categories: int = ATOM_CATEGORIES,
) -> IncidenceBatch:
    phi_parts: list[torch.Tensor] = []
    q_parts: list[torch.Tensor] = []
    node_index: list[torch.Tensor] = []
    psi_parts: list[torch.Tensor] = []
    r_parts: list[torch.Tensor] = []
    edge_index_parts: list[torch.Tensor] = []
    edge_index_global: list[torch.Tensor] = []
    a_rows: list[torch.Tensor] = []
    n_nodes: list[int] = []
    n_edges: list[int] = []
    targets: list[float] = []
    node_offset = 0
    for graph_id, molecule in enumerate(molecules):
        phi = np.asarray(molecule.phi, dtype=np.float32)
        q = r2.one_hot_q(molecule.atom_idx, atom_categories).astype(np.float32)
        phi_parts.append(torch.as_tensor(phi))
        q_parts.append(torch.as_tensor(q))
        node_index.append(torch.full((phi.shape[0],), int(graph_id), dtype=torch.long))
        edge_u = np.asarray(molecule.edge_u, dtype=np.int64).reshape(-1)
        edge_v = np.asarray(molecule.edge_v, dtype=np.int64).reshape(-1)
        edges_local = list(zip(edge_u.tolist(), edge_v.tolist()))
        psi = edge.build_edge_roles(phi, edges_local)
        r = edge.one_hot_bond(molecule.bond_type, BOND_CATEGORIES)
        if psi.shape[0] != r.shape[0] or psi.shape[0] != edge_u.shape[0]:
            raise RuntimeError("edge role / bond / endpoint count mismatch in molecule")
        psi_parts.append(torch.as_tensor(psi.astype(np.float32)))
        r_parts.append(torch.as_tensor(r.astype(np.float32)))
        edge_index_parts.append(
            torch.as_tensor(
                np.stack([edge_u + node_offset, edge_v + node_offset], axis=0),
                dtype=torch.long,
            )
        )
        edge_index_global.append(
            torch.full((psi.shape[0],), int(graph_id), dtype=torch.long)
        )
        a_rows.append(torch.as_tensor(np.asarray(molecule.A, dtype=np.float32)))
        n_nodes.append(int(molecule.n_nodes))
        n_edges.append(int(molecule.n_edges))
        targets.append(float(molecule.y))
        node_offset += int(molecule.n_nodes)
    return IncidenceBatch(
        phi=torch.cat(phi_parts, dim=0),
        q=torch.cat(q_parts, dim=0),
        node_graph=torch.cat(node_index, dim=0),
        psi=torch.cat(psi_parts, dim=0) if psi_parts else torch.zeros((0, EDGE_ROLE_DIM)),
        r=torch.cat(r_parts, dim=0) if r_parts else torch.zeros((0, BOND_CATEGORIES)),
        edge_index=(
            torch.cat(edge_index_parts, dim=1)
            if edge_index_parts
            else torch.zeros((2, 0), dtype=torch.long)
        ),
        edge_graph=(
            torch.cat(edge_index_global, dim=0)
            if edge_index_global
            else torch.zeros((0,), dtype=torch.long)
        ),
        A=torch.stack(a_rows, dim=0),
        n_nodes=torch.as_tensor(n_nodes, dtype=torch.float32),
        n_edges=torch.as_tensor(n_edges, dtype=torch.float32),
        y=torch.as_tensor(targets, dtype=torch.float32),
        n_graphs=int(len(molecules)),
    )


def make_incidence_loader(
    molecules: Sequence[edge.EdgeMoleculeFeatures],
    batch_size: int,
    shuffle: bool,
    seed: int,
):
    from torch.utils.data import DataLoader as TorchDataLoader

    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return TorchDataLoader(
        list(molecules),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        collate_fn=collate_incidence_molecules,
    )


# ---------------------------------------------------------------------------
# dense, deterministic incidence / segment helpers
# ---------------------------------------------------------------------------


def incidence_matrices(
    edge_index: torch.Tensor, n_nodes: int, dtype: torch.dtype
) -> tuple[torch.Tensor, torch.Tensor]:
    """Dense ``[N, M]`` first/second endpoint indicators (no atomic ``index_add_``)."""
    n_edges = int(edge_index.shape[1])
    first = torch.zeros((int(n_nodes), n_edges), dtype=dtype, device=edge_index.device)
    second = torch.zeros_like(first)
    if n_edges > 0:
        columns = torch.arange(n_edges, device=edge_index.device)
        first[edge_index[0], columns] = 1.0
        second[edge_index[1], columns] = 1.0
    return first, second


def segment_mean(
    values: torch.Tensor, indicator: torch.Tensor, counts: torch.Tensor
) -> torch.Tensor:
    return (indicator.t() @ values) / counts.view(-1, 1).clamp_min(1.0)


def _safe_std(value: torch.Tensor) -> torch.Tensor:
    if value.numel() <= 1:
        return value.new_zeros(())
    return value.float().std(unbiased=False)


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class FSARIncidenceModel(nn.Module):
    """Persistent S-A-B objects + shared recurrent incidence processor."""

    def __init__(
        self,
        variant: str = "C1",
        *,
        phi_dim: int = PHI_DIM,
        edge_role_dim: int = EDGE_ROLE_DIM,
        atom_categories: int = ATOM_CATEGORIES,
        bond_categories: int = BOND_CATEGORIES,
        a_dim: int = A_DIM,
        hidden: int = HIDDEN,
        processor_hidden: int = PROCESSOR_HIDDEN,
        rounds: int = ROUNDS,
        readout_hidden: int = READOUT_HIDDEN,
        residual_init: float = RESIDUAL_INIT,
    ) -> None:
        super().__init__()
        variant = str(variant)
        if variant not in MODELS:
            raise ValueError(f"unknown variant {variant!r}; expected {MODELS}")
        self.variant = variant
        self.hidden = int(hidden)
        self.processor_hidden = int(processor_hidden)
        self.rounds = int(rounds)
        self.a_dim = int(a_dim)
        self.readout_input_dim = 4 * int(hidden) + int(a_dim) + 2

        d = int(hidden)
        # -- persistent object initialisation --------------------------------
        self.node_s = nn.Linear(int(phi_dim), d)
        self.node_a = nn.Linear(int(atom_categories), d)
        self.node_us = nn.Linear(d, d, bias=False)
        self.node_ua = nn.Linear(d, d, bias=False)
        self.node_b = nn.Linear(d, d)
        self.edge_s = nn.Linear(int(edge_role_dim), d)
        self.edge_a = nn.Linear(int(bond_categories), d)
        self.edge_us = nn.Linear(d, d, bias=False)
        self.edge_ua = nn.Linear(d, d, bias=False)
        self.edge_b = nn.Linear(d, d)

        # -- shared recurrent processor (built identically for both variants)
        h_proc = int(processor_hidden)
        self.edge_ln = nn.LayerNorm(3 * d)
        self.edge_in = nn.Linear(3 * d, h_proc)
        self.edge_out = nn.Linear(h_proc, d)
        self.edge_scale = nn.Parameter(torch.full((1,), float(residual_init)))
        self.node_ln = nn.LayerNorm(2 * d)
        self.node_msg_in = nn.Linear(2 * d, h_proc)
        self.node_msg_out = nn.Linear(h_proc, d)
        self.node_update_in = nn.Linear(2 * d, h_proc)
        self.node_update_out = nn.Linear(h_proc, d)
        self.node_scale = nn.Parameter(torch.full((1,), float(residual_init)))

        # -- readout ---------------------------------------------------------
        self.readout = nn.Sequential(
            nn.Linear(self.readout_input_dim, int(readout_hidden)),
            nn.SiLU(),
            nn.Linear(int(readout_hidden), 1),
        )

        self.capture_diagnostics = False
        self.last_stats: dict[str, float] = {}

    # -- initialisation ------------------------------------------------------

    def initial_objects(
        self, batch: IncidenceBatch
    ) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
        s_v = self.node_s(batch.phi)
        a_v = self.node_a(batch.q)
        b_v = self.node_b(self.node_us(s_v) * self.node_ua(a_v))
        h = s_v + a_v + b_v

        s_e = self.edge_s(batch.psi)
        a_e = self.edge_a(batch.r)
        b_e = self.edge_b(self.edge_us(s_e) * self.edge_ua(a_e))
        g = s_e + a_e + b_e

        binding = {"node_binding": b_v, "edge_binding": b_e}
        return h, g, binding

    # -- processor -----------------------------------------------------------

    def process(
        self, batch: IncidenceBatch, h: torch.Tensor, g: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        node_indicator = r2.graph_indicator(batch.node_graph, int(batch.n_graphs))
        edge_indicator = r2.graph_indicator(batch.edge_graph, int(batch.n_graphs))
        node_counts = node_indicator.sum(dim=0).clamp_min(1.0)
        edge_counts = edge_indicator.sum(dim=0).clamp_min(1.0)
        incidence_first, incidence_second = incidence_matrices(
            batch.edge_index, h.shape[0], h.dtype
        )
        degree = (incidence_first + incidence_second).sum(dim=1).clamp_min(1.0)

        edge_residuals: list[torch.Tensor] = []
        node_residuals: list[torch.Tensor] = []
        for _round in range(self.rounds):
            if self.variant == "C1":
                left = h[batch.edge_index[0]]
                right = h[batch.edge_index[1]]
                edge_input = torch.cat(
                    [left + right, (left - right).abs(), g], dim=1
                )
            else:  # C0: global node context only, no incidence
                node_context = segment_mean(h, node_indicator, node_counts)
                per_edge_context = node_context[batch.edge_graph]
                edge_input = torch.cat(
                    [per_edge_context, per_edge_context, g], dim=1
                )
            edge_update = self.edge_out(
                F.silu(self.edge_in(self.edge_ln(edge_input)))
            )
            residual_e = self.edge_scale * edge_update
            g = g + residual_e
            edge_residuals.append(residual_e)

            if self.variant == "C1":
                message_to_first = self.node_msg_out(
                    F.silu(self.node_msg_in(torch.cat([g, right], dim=1)))
                )
                message_to_second = self.node_msg_out(
                    F.silu(self.node_msg_in(torch.cat([g, left], dim=1)))
                )
                message_sum = (
                    incidence_first @ message_to_first
                    + incidence_second @ message_to_second
                )
                node_message = message_sum / degree.view(-1, 1)
            else:  # C0: one global edge-message mean per graph
                edge_context = segment_mean(g, edge_indicator, edge_counts)
                per_edge_context = edge_context[batch.edge_graph]
                message = self.node_msg_out(
                    F.silu(self.node_msg_in(torch.cat([g, per_edge_context], dim=1)))
                )
                graph_message = segment_mean(message, edge_indicator, edge_counts)
                node_message = graph_message[batch.node_graph]
            node_update = self.node_update_out(
                F.silu(self.node_update_in(self.node_ln(torch.cat([h, node_message], dim=1))))
            )
            residual_n = self.node_scale * node_update
            h = h + residual_n
            node_residuals.append(residual_n)

        diagnostics = {
            "edge_residual_std": float(
                torch.stack([_safe_std(value) for value in edge_residuals]).mean()
            ),
            "node_residual_std": float(
                torch.stack([_safe_std(value) for value in node_residuals]).mean()
            ),
        }
        return h, g, diagnostics

    # -- readout -------------------------------------------------------------

    def pool(
        self, batch: IncidenceBatch, h: torch.Tensor, g: torch.Tensor
    ) -> torch.Tensor:
        node_indicator = r2.graph_indicator(batch.node_graph, int(batch.n_graphs))
        edge_indicator = r2.graph_indicator(batch.edge_graph, int(batch.n_graphs))
        node_counts = node_indicator.sum(dim=0).clamp_min(1.0)
        edge_counts = edge_indicator.sum(dim=0).clamp_min(1.0)
        node_mean = segment_mean(h, node_indicator, node_counts)
        edge_mean = segment_mean(g, edge_indicator, edge_counts)
        node_second = segment_mean(h * h, node_indicator, node_counts)
        edge_second = segment_mean(g * g, edge_indicator, edge_counts)
        node_std = (node_second - node_mean * node_mean).clamp_min(0.0).add(_EPS).sqrt()
        edge_std = (edge_second - edge_mean * edge_mean).clamp_min(0.0).add(_EPS).sqrt()
        size = torch.cat(
            [
                torch.log1p(batch.n_nodes).view(-1, 1),
                torch.log1p(batch.n_edges).view(-1, 1),
            ],
            dim=1,
        )
        return torch.cat([node_mean, node_std, edge_mean, edge_std, batch.A, size], dim=1)

    # -- forward -------------------------------------------------------------

    def forward(
        self, batch: IncidenceBatch, capture: bool = False
    ) -> torch.Tensor:
        h, g, binding = self.initial_objects(batch)
        h, g, process_stats = self.process(batch, h, g)
        read = self.pool(batch, h, g)
        prediction = self.readout(read).view(-1)
        if capture or self.capture_diagnostics:
            self.last_stats = {
                "node_binding_std": float(binding["node_binding"].std()),
                "edge_binding_std": float(binding["edge_binding"].std()),
                "node_object_std": float(h.std()),
                "edge_object_std": float(g.std()),
                **process_stats,
            }
        return prediction

    def forward_with_states(
        self, batch: IncidenceBatch
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, dict[str, torch.Tensor], dict[str, float]]:
        """Return ``(prediction, h, g, binding, process_stats)`` for diagnostics."""
        h, g, binding = self.initial_objects(batch)
        h, g, process_stats = self.process(batch, h, g)
        prediction = self.readout(self.pool(batch, h, g)).view(-1)
        return prediction, h, g, binding, process_stats

    # -- diagnostics ---------------------------------------------------------

    def residual_scales(self) -> tuple[float, float]:
        return float(self.edge_scale.detach()), float(self.node_scale.detach())

    def node_binding_norm(self) -> float:
        total = 0.0
        for module in (self.node_us, self.node_ua, self.node_b, self.node_s, self.node_a):
            for parameter in module.parameters():
                total += float(parameter.detach().norm() ** 2)
        return float(total**0.5)

    def edge_binding_norm(self) -> float:
        total = 0.0
        for module in (self.edge_us, self.edge_ua, self.edge_b, self.edge_s, self.edge_a):
            for parameter in module.parameters():
                total += float(parameter.detach().norm() ** 2)
        return float(total**0.5)

    def binding_weight_norm(self) -> float:
        total = 0.0
        for module in (self.node_us, self.node_ua, self.node_b, self.edge_us, self.edge_ua, self.edge_b):
            for parameter in module.parameters():
                total += float(parameter.detach().norm() ** 2)
        return float(total**0.5)


def build_incidence_model(
    variant: str = "C1", seed: int | None = None, **kwargs: Any
) -> FSARIncidenceModel:
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
    return FSARIncidenceModel(variant=str(variant), **kwargs)


# ---------------------------------------------------------------------------
# parameter accounting / optimizer groups
# ---------------------------------------------------------------------------

_BINDING_MODULES = ("node_us", "node_ua", "node_b", "edge_us", "edge_ua", "edge_b")
_NO_DECAY_MODULES = _BINDING_MODULES + ("edge_ln", "node_ln")


def parameter_breakdown_incidence(model: FSARIncidenceModel) -> dict[str, Any]:
    def _count(module: nn.Module) -> int:
        return int(sum(parameter.numel() for parameter in module.parameters()))

    node_initializer = _count(model.node_s) + _count(model.node_a)
    edge_initializer = _count(model.edge_s) + _count(model.edge_a)
    binding = sum(_count(getattr(model, name)) for name in _BINDING_MODULES)
    processor = (
        _count(model.edge_ln)
        + _count(model.edge_in)
        + _count(model.edge_out)
        + model.edge_scale.numel()
        + _count(model.node_ln)
        + _count(model.node_msg_in)
        + _count(model.node_msg_out)
        + _count(model.node_update_in)
        + _count(model.node_update_out)
        + model.node_scale.numel()
    )
    readout = _count(model.readout)
    total = _count(model)
    return {
        "variant": str(model.variant),
        "node_initializer": int(node_initializer),
        "edge_initializer": int(edge_initializer),
        "binding": int(binding),
        "incidence_processor": int(processor),
        "readout": int(readout),
        "total": int(total),
        "hidden": int(model.hidden),
        "processor_hidden": int(model.processor_hidden),
        "rounds": int(model.rounds),
        "dataset_dependent_vocabulary_params": 0,
    }


def optimizer_parameter_groups(
    model: FSARIncidenceModel, weight_decay: float
) -> list[dict[str, Any]]:
    """Canonical wd for everything except binding / LayerNorm / residual scales."""
    no_decay: list[nn.Parameter] = []
    decay: list[nn.Parameter] = []
    for name, parameter in model.named_parameters():
        root = name.split(".", 1)[0]
        if root in _NO_DECAY_MODULES or name.endswith("_scale"):
            no_decay.append(parameter)
        else:
            decay.append(parameter)
    return [
        {"params": decay, "weight_decay": float(weight_decay)},
        {"params": no_decay, "weight_decay": 0.0},
    ]


# ---------------------------------------------------------------------------
# evaluation-only attribute permutations (topology and object identity fixed)
# ---------------------------------------------------------------------------


def permute_q_within_graphs(batch: IncidenceBatch, seed: int) -> torch.Tensor:
    """Permute atom attributes among the original nodes of each graph."""
    generator = torch.Generator().manual_seed(int(seed))
    q = batch.q.clone()
    node_graph = batch.node_graph.detach().cpu()
    for graph_id in range(int(batch.n_graphs)):
        indices = torch.nonzero(node_graph == int(graph_id), as_tuple=False).view(-1)
        if indices.numel() <= 1:
            continue
        order = torch.randperm(indices.numel(), generator=generator)
        q[indices] = batch.q[indices[order]]
    return q


def permute_bond_types_within_graphs(batch: IncidenceBatch, seed: int) -> torch.Tensor:
    """Permute bond types among the undirected edges of each graph."""
    generator = torch.Generator().manual_seed(int(seed))
    r = batch.r.clone()
    edge_graph = batch.edge_graph.detach().cpu()
    for graph_id in range(int(batch.n_graphs)):
        indices = torch.nonzero(edge_graph == int(graph_id), as_tuple=False).view(-1)
        if indices.numel() <= 1:
            continue
        order = torch.randperm(indices.numel(), generator=generator)
        r[indices] = batch.r[indices[order]]
    return r


__all__ = [
    "PHI_DIM",
    "EDGE_ROLE_DIM",
    "ATOM_CATEGORIES",
    "BOND_CATEGORIES",
    "A_DIM",
    "HIDDEN",
    "PROCESSOR_HIDDEN",
    "ROUNDS",
    "READOUT_INPUT_DIM",
    "MODELS",
    "INCIDENCE_MODELS",
    "BAG_MODELS",
    "IncidenceBatch",
    "FSARIncidenceModel",
    "collate_incidence_molecules",
    "make_incidence_loader",
    "incidence_matrices",
    "segment_mean",
    "build_incidence_model",
    "parameter_breakdown_incidence",
    "optimizer_parameter_groups",
    "permute_q_within_graphs",
    "permute_bond_types_within_graphs",
]

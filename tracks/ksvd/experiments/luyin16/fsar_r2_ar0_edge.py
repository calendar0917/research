"""FSAR-R2-AR0-EDGE: fixed radius-2 explicit coordinates + edge assignment residual.

This is the model / feature half of the **edge structural-role ↔ bond-type
assignment** round.  It does not touch the frozen AR0 node model or protocol.

Definitions (frozen, no sweep)
------------------------------

A molecule is ``G = (T, X, E_attr)``.  For every original **undirected** edge
``e = (u, v)`` we already have the two endpoint pure-topology coordinates
``phi_u, phi_v in R^65`` (the frozen radius-2 explicit coordinate of AR0).
The symmetric pure-topology edge role is

    psi_e = [ phi_u + phi_v ,  |phi_u - phi_v| ]   in R^130,

which is exactly invariant under endpoint swap.  No chemistry (atom type /
bond type), no typed path, no learned message passing and no new motif enters
``psi_e``.

With ``r_e = onehot(bond_type_e) in R^4`` (4 padded ZINC bond categories;
observed values 1..3 -- verified from ``data/ZINC/raw/bond_dict.pickle``)::

    J_E = sum_e psi_e r_e^T
    P_E = (1/m) (sum_e psi_e) (sum_e r_e)^T
    C_E = J_E - P_E = sum_e (psi_e - psibar)(r_e - rbar)^T   # sum-centred

For fixed topology and fixed bond-type multiset, ``E_pi[C_E] = 0`` under a
uniform random permutation of the bond types across the undirected edges.

Train-only per-coordinate RMS scaling (no dataset-mean subtraction)::

    C~_E = C_E / D_E     (coordinates with raw train RMS <= 1e-9 masked to 0)
    P~_E = P_E / D_PE    (same safe handling)

Three models share one base and the AR0 node assignment term:

* ``BV``   -- ``yhat = F0([z_S, a]) + <W_B, C~_V>``
* ``BVE``  -- ``yhat = F0([z_S, a]) + <W_B, C~_V> + <W_E, C~_E>``
* ``BVEM`` -- ``yhat = F0([z_S, a]) + <W_B, C~_V> + <W_ME, P~_E>``

``W_B``, ``W_E`` and ``W_ME`` are bias-free plain parameters and are the only
routes for their statistic into the prediction (no MLP, activation or bypass).
``W_E``/``W_ME`` are zero-initialised, so all three predict exactly the base at
step 0.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2

# ---------------------------------------------------------------------------
# frozen schema / dimensions
# ---------------------------------------------------------------------------

PHI_DIM = r2.PHI_DIM  # 65
EDGE_ROLE_DIM = 2 * PHI_DIM  # 130
BOND_CATEGORIES = r2.BOND_CATEGORIES  # 4 (NONE, SINGLE, DOUBLE, TRIPLE)
ATOM_CATEGORIES = r2.ATOM_CATEGORIES  # 28
A_DIM = r2.A_DIM  # 64

S_HIDDEN = r2.S_HIDDEN
S_OUT = r2.S_OUT
Z_S_DIM = r2.Z_S_DIM
BASE_HIDDEN = r2.BASE_HIDDEN
BASE_INPUT_DIM = r2.BASE_INPUT_DIM

MODELS = ("BV", "BVE", "BVEM")
EDGE_MODELS = ("BVE", "BVEM")
NODE_ONLY = ("BV",)

SCALER_FLOOR = r2.SCALER_FLOOR
SCALER_EPS = r2.SCALER_EPS


# ---------------------------------------------------------------------------
# pure-topology edge roles
# ---------------------------------------------------------------------------


def build_edge_roles(phi: np.ndarray, edges: Sequence[tuple[int, int]]) -> np.ndarray:
    """``[m, 130]`` symmetric pure-topology edge roles (one row per undirected edge)."""
    phi = np.asarray(phi, dtype=np.float64)
    if len(edges) == 0:
        return np.zeros((0, EDGE_ROLE_DIM), dtype=np.float64)
    rows = []
    for left, right in edges:
        u = phi[int(left)]
        v = phi[int(right)]
        rows.append(np.concatenate([u + v, np.abs(u - v)]))
    out = np.stack(rows, axis=0)
    if out.shape[1] != EDGE_ROLE_DIM:
        raise RuntimeError(f"edge role width {out.shape[1]} != {EDGE_ROLE_DIM}")
    return out


def one_hot_bond(bond_types: np.ndarray, bond_categories: int = BOND_CATEGORIES) -> np.ndarray:
    idx = np.asarray(bond_types, dtype=np.int64).reshape(-1)
    out = np.zeros((idx.shape[0], int(bond_categories)), dtype=np.float64)
    valid = (idx >= 0) & (idx < int(bond_categories))
    out[np.arange(idx.shape[0])[valid], idx[valid]] = 1.0
    return out


# ---------------------------------------------------------------------------
# assignment statistics (numpy reference + torch batched implementation)
# ---------------------------------------------------------------------------


def edge_center_stats(psi: np.ndarray, r: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference ``(C_E, P_E)`` for one molecule (sum-centred, no ``/(m-1)``)."""
    psi = np.asarray(psi, dtype=np.float64)
    r = np.asarray(r, dtype=np.float64)
    if psi.shape[0] != r.shape[0]:
        raise RuntimeError("edge role / bond one-hot count mismatch")
    if psi.shape[0] == 0:
        width, categories = int(psi.shape[1]), int(r.shape[1])
        zero = np.zeros((width, categories), dtype=np.float64)
        return zero, zero.copy()
    m = float(psi.shape[0])
    psi_bar = psi.mean(axis=0)
    r_bar = r.mean(axis=0)
    j_matrix = psi.T @ r
    p_matrix = m * np.outer(psi_bar, r_bar)
    return j_matrix - p_matrix, p_matrix


def segment_edge_stats(
    psi: torch.Tensor,
    r: torch.Tensor,
    edge_graph: torch.Tensor,
    n_graphs: int,
    indicator: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched torch ``(C_E, P_E)`` with shape ``[n_graphs, 130, 4]``."""
    if indicator is None:
        indicator = r2.graph_indicator(edge_graph, n_graphs)
    indicator = indicator.to(psi.dtype)
    width, categories = int(psi.shape[1]), int(r.shape[1])
    outer = (psi.unsqueeze(2) * r.unsqueeze(1)).reshape(psi.shape[0], width * categories)
    j_matrix = (indicator.t() @ outer).reshape(int(n_graphs), width, categories)
    sum_psi = indicator.t() @ psi
    sum_r = indicator.t() @ r
    counts = indicator.sum(dim=0).clamp_min(1.0)
    p_matrix = (sum_psi.unsqueeze(2) * sum_r.unsqueeze(1)) / counts.view(-1, 1, 1)
    return j_matrix - p_matrix, p_matrix


def fit_edge_scalers(
    c_matrices: Sequence[np.ndarray],
    p_matrices: Sequence[np.ndarray],
    floor: float = SCALER_FLOOR,
    eps: float = SCALER_EPS,
) -> dict[str, np.ndarray]:
    """Train-only per-coordinate RMS scalers for ``C_E`` / ``P_E``."""
    if not c_matrices:
        raise RuntimeError("no training edge statistics provided")
    c_stack = np.stack([np.asarray(item, dtype=np.float64) for item in c_matrices])
    p_stack = np.stack([np.asarray(item, dtype=np.float64) for item in p_matrices])
    c_rms_raw = np.sqrt((c_stack**2).mean(axis=0))
    p_rms_raw = np.sqrt((p_stack**2).mean(axis=0))
    c_mask = (c_rms_raw > float(floor)).astype(np.float64)
    p_mask = (p_rms_raw > float(floor)).astype(np.float64)
    c_scale = np.where(c_mask > 0.0, np.sqrt((c_stack**2).mean(axis=0) + float(eps)), 1.0)
    p_scale = np.where(p_mask > 0.0, np.sqrt((p_stack**2).mean(axis=0) + float(eps)), 1.0)
    return {
        "CE_rms": c_scale.astype(np.float32),
        "CE_mask": c_mask.astype(np.float32),
        "PE_rms": p_scale.astype(np.float32),
        "PE_mask": p_mask.astype(np.float32),
        "CE_rms_raw": c_rms_raw.astype(np.float64),
        "PE_rms_raw": p_rms_raw.astype(np.float64),
    }


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


@dataclass
class EdgeMoleculeFeatures:
    """Cached per-molecule features for the edge round."""

    phi: np.ndarray  # [n, 65] float32 pure-topology
    atom_idx: np.ndarray  # [n] int64
    edge_u: np.ndarray  # [m] int64 canonical undirected endpoints
    edge_v: np.ndarray  # [m] int64
    bond_type: np.ndarray  # [m] int64
    A: np.ndarray  # [64] float32
    n_nodes: int
    n_edges: int
    y: float


@dataclass
class FSAREdgeBatch:
    phi: torch.Tensor  # [N, 65]
    q: torch.Tensor  # [N, 28]
    node_graph: torch.Tensor  # [N]
    psi: torch.Tensor  # [M, 130]
    r: torch.Tensor  # [M, 4]
    edge_graph: torch.Tensor  # [M]
    A: torch.Tensor  # [B, 64]
    n_nodes: torch.Tensor
    n_edges: torch.Tensor
    y: torch.Tensor
    n_graphs: int

    def to(self, device: torch.device | str) -> "FSAREdgeBatch":
        device = torch.device(device)
        return FSAREdgeBatch(
            phi=self.phi.to(device),
            q=self.q.to(device),
            node_graph=self.node_graph.to(device),
            psi=self.psi.to(device),
            r=self.r.to(device),
            edge_graph=self.edge_graph.to(device),
            A=self.A.to(device),
            n_nodes=self.n_nodes.to(device),
            n_edges=self.n_edges.to(device),
            y=self.y.to(device),
            n_graphs=int(self.n_graphs),
        )

    def with_r(self, r: torch.Tensor) -> "FSAREdgeBatch":
        if r.shape != self.r.shape:
            raise RuntimeError("replacement bond one-hot shape mismatch")
        return FSAREdgeBatch(
            phi=self.phi,
            q=self.q,
            node_graph=self.node_graph,
            psi=self.psi,
            r=r,
            edge_graph=self.edge_graph,
            A=self.A,
            n_nodes=self.n_nodes,
            n_edges=self.n_edges,
            y=self.y,
            n_graphs=int(self.n_graphs),
        )


def collate_edge_molecules(
    molecules: Sequence[EdgeMoleculeFeatures], atom_categories: int = ATOM_CATEGORIES
) -> FSAREdgeBatch:
    phi_parts: list[torch.Tensor] = []
    q_parts: list[torch.Tensor] = []
    node_index: list[torch.Tensor] = []
    psi_parts: list[torch.Tensor] = []
    r_parts: list[torch.Tensor] = []
    edge_index: list[torch.Tensor] = []
    a_rows: list[torch.Tensor] = []
    n_nodes: list[int] = []
    n_edges: list[int] = []
    targets: list[float] = []
    for graph_id, molecule in enumerate(molecules):
        phi = np.asarray(molecule.phi, dtype=np.float32)
        q = r2.one_hot_q(molecule.atom_idx, atom_categories).astype(np.float32)
        phi_parts.append(torch.as_tensor(phi))
        q_parts.append(torch.as_tensor(q))
        node_index.append(torch.full((phi.shape[0],), int(graph_id), dtype=torch.long))
        edges = list(zip(molecule.edge_u.tolist(), molecule.edge_v.tolist()))
        psi = build_edge_roles(phi, edges)
        r = one_hot_bond(molecule.bond_type, BOND_CATEGORIES)
        if psi.shape[0] != r.shape[0]:
            raise RuntimeError("edge role / bond count mismatch in molecule")
        psi_parts.append(torch.as_tensor(psi.astype(np.float32)))
        r_parts.append(torch.as_tensor(r.astype(np.float32)))
        edge_index.append(torch.full((psi.shape[0],), int(graph_id), dtype=torch.long))
        a_rows.append(torch.as_tensor(np.asarray(molecule.A, dtype=np.float32)))
        n_nodes.append(int(molecule.n_nodes))
        n_edges.append(int(molecule.n_edges))
        targets.append(float(molecule.y))
    return FSAREdgeBatch(
        phi=torch.cat(phi_parts, dim=0),
        q=torch.cat(q_parts, dim=0),
        node_graph=torch.cat(node_index, dim=0),
        psi=torch.cat(psi_parts, dim=0) if psi_parts else torch.zeros((0, EDGE_ROLE_DIM)),
        r=torch.cat(r_parts, dim=0) if r_parts else torch.zeros((0, BOND_CATEGORIES)),
        edge_graph=torch.cat(edge_index, dim=0) if edge_index else torch.zeros((0,), dtype=torch.long),
        A=torch.stack(a_rows, dim=0),
        n_nodes=torch.as_tensor(n_nodes, dtype=torch.float32),
        n_edges=torch.as_tensor(n_edges, dtype=torch.float32),
        y=torch.as_tensor(targets, dtype=torch.float32),
        n_graphs=int(len(molecules)),
    )


def make_edge_loader(
    molecules: Sequence[EdgeMoleculeFeatures],
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
        collate_fn=collate_edge_molecules,
    )


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class FSARR2AR0EdgeModel(nn.Module):
    """``F0`` + node assignment (``W_B``) + optional edge branch."""

    def __init__(
        self,
        variant: str,
        *,
        phi_dim: int = PHI_DIM,
        atom_categories: int = ATOM_CATEGORIES,
        bond_categories: int = BOND_CATEGORIES,
        a_dim: int = A_DIM,
        s_hidden: int = S_HIDDEN,
        s_out: int = S_OUT,
        base_hidden: Sequence[int] = BASE_HIDDEN,
        activation: str = "silu",
        scaler_C: np.ndarray | None = None,
        mask_C: np.ndarray | None = None,
        scaler_P: np.ndarray | None = None,
        mask_P: np.ndarray | None = None,
        scaler_CE: np.ndarray | None = None,
        mask_CE: np.ndarray | None = None,
        scaler_PE: np.ndarray | None = None,
        mask_PE: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        variant = str(variant)
        if variant not in MODELS:
            raise ValueError(f"unknown variant {variant!r}; expected {MODELS}")
        self.variant = variant
        self.phi_dim = int(phi_dim)
        self.atom_categories = int(atom_categories)
        self.bond_categories = int(bond_categories)
        self.edge_role_dim = 2 * int(phi_dim)
        self.a_dim = int(a_dim)
        self.s_out = int(s_out)
        self.z_s_dim = 2 * int(s_out) + 2
        self.base_input_dim = self.z_s_dim + self.a_dim

        self.s_encoder = nn.Sequential(
            nn.Linear(self.phi_dim, int(s_hidden)),
            r2._act(activation),
            nn.Linear(int(s_hidden), int(s_out)),
        )
        base_layers: list[nn.Module] = []
        widths = [self.base_input_dim, *[int(width) for width in base_hidden], 1]
        for index in range(len(widths) - 1):
            base_layers.append(nn.Linear(widths[index], widths[index + 1]))
            if index < len(widths) - 2:
                base_layers.append(r2._act(activation))
        self.base = nn.Sequential(*base_layers)

        # Created in a fixed order so base + W_B are bit-identical across
        # variants at the same seed; the edge parameter is always appended last.
        self.W_B = nn.Parameter(torch.zeros(self.phi_dim, self.atom_categories))
        self.W_E: nn.Parameter | None = None
        self.W_ME: nn.Parameter | None = None
        if variant == "BVE":
            self.W_E = nn.Parameter(torch.zeros(self.edge_role_dim, self.bond_categories))
        elif variant == "BVEM":
            self.W_ME = nn.Parameter(torch.zeros(self.edge_role_dim, self.bond_categories))

        def _buffer(value: np.ndarray | None, shape: tuple[int, int], name: str) -> None:
            array = (
                np.zeros(shape, dtype=np.float32)
                if value is None
                else np.asarray(value, dtype=np.float32)
            )
            if array.shape != shape:
                raise RuntimeError(f"{name} shape {array.shape} mismatch")
            self.register_buffer(name, torch.from_numpy(array.copy()))

        node_shape = (self.phi_dim, self.atom_categories)
        edge_shape = (self.edge_role_dim, self.bond_categories)
        _buffer(scaler_C, node_shape, "scaler_C")
        _buffer(mask_C, node_shape, "mask_C")
        _buffer(scaler_P, node_shape, "scaler_P")
        _buffer(mask_P, node_shape, "mask_P")
        _buffer(scaler_CE, edge_shape, "scaler_CE")
        _buffer(mask_CE, edge_shape, "mask_CE")
        _buffer(scaler_PE, edge_shape, "scaler_PE")
        _buffer(mask_PE, edge_shape, "mask_PE")

    # -- geometry -----------------------------------------------------------

    def encode_s(self, batch: FSAREdgeBatch, indicator: torch.Tensor | None = None) -> torch.Tensor:
        u = self.s_encoder(batch.phi)
        if indicator is None:
            indicator = r2.graph_indicator(batch.node_graph, int(batch.n_graphs))
        indicator = indicator.to(u.dtype)
        sum_u = indicator.t() @ u
        counts = indicator.sum(dim=0).clamp_min(1.0)
        mean_u = sum_u / counts.view(-1, 1)
        n_log = torch.log1p(batch.n_nodes).view(-1, 1)
        m_log = torch.log1p(batch.n_edges).view(-1, 1)
        return torch.cat([sum_u, mean_u, n_log, m_log], dim=1)

    def base_prediction(self, batch: FSAREdgeBatch, indicator: torch.Tensor | None = None) -> torch.Tensor:
        z_s = self.encode_s(batch, indicator=indicator)
        return self.base(torch.cat([z_s, batch.A], dim=1)).view(-1)

    def compute_node_statistics(
        self, batch: FSAREdgeBatch, indicator: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return r2.segment_center_stats(
            batch.phi, batch.q, batch.node_graph, int(batch.n_graphs), indicator=indicator
        )

    def compute_edge_statistics(
        self, batch: FSAREdgeBatch, indicator: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return segment_edge_stats(
            batch.psi, batch.r, batch.edge_graph, int(batch.n_graphs), indicator=indicator
        )

    def node_assignment_term(self, c_matrix: torch.Tensor) -> torch.Tensor:
        statistic = r2.normalized_stat(c_matrix, self.scaler_C, self.mask_C)
        return (statistic * self.W_B.unsqueeze(0)).sum(dim=(1, 2))

    def edge_assignment_term(self, c_matrix: torch.Tensor, p_matrix: torch.Tensor) -> torch.Tensor:
        if self.variant == "BVE":
            assert self.W_E is not None
            statistic = r2.normalized_stat(c_matrix, self.scaler_CE, self.mask_CE)
            weight = self.W_E
        elif self.variant == "BVEM":
            assert self.W_ME is not None
            statistic = r2.normalized_stat(p_matrix, self.scaler_PE, self.mask_PE)
            weight = self.W_ME
        else:
            return c_matrix.new_zeros(c_matrix.shape[0])
        return (statistic * weight.unsqueeze(0)).sum(dim=(1, 2))

    def forward(self, batch: FSAREdgeBatch) -> torch.Tensor:
        indicator = r2.graph_indicator(batch.node_graph, int(batch.n_graphs))
        base = self.base_prediction(batch, indicator=indicator)
        c_node, _p_node = self.compute_node_statistics(batch, indicator=indicator)
        prediction = base + self.node_assignment_term(c_node)
        if self.variant in EDGE_MODELS:
            c_edge, p_edge = self.compute_edge_statistics(batch)
            prediction = prediction + self.edge_assignment_term(c_edge, p_edge)
        return prediction

    # -- diagnostics --------------------------------------------------------

    def branch_weight_norm(self) -> float:
        total = 0.0
        if self.W_B is not None:
            total += float(self.W_B.detach().norm() ** 2)
        if self.W_E is not None:
            total += float(self.W_E.detach().norm() ** 2)
        if self.W_ME is not None:
            total += float(self.W_ME.detach().norm() ** 2)
        return float(np.sqrt(total))

    def node_weight_norm(self) -> float:
        return float(self.W_B.detach().norm())

    def edge_weight_norm(self) -> float:
        if self.W_E is not None:
            return float(self.W_E.detach().norm())
        if self.W_ME is not None:
            return float(self.W_ME.detach().norm())
        return 0.0

    def branch_weight_abs_sum(self) -> float:
        total = float(self.W_B.detach().abs().sum())
        if self.W_E is not None:
            total += float(self.W_E.detach().abs().sum())
        if self.W_ME is not None:
            total += float(self.W_ME.detach().abs().sum())
        return total

    def branch_weight_sparsity(self, threshold: float = 1.0e-8) -> float:
        values = [self.W_B.detach()]
        if self.W_E is not None:
            values.append(self.W_E.detach())
        if self.W_ME is not None:
            values.append(self.W_ME.detach())
        weight = torch.cat([value.reshape(-1) for value in values])
        return float((weight.abs() <= float(threshold)).to(torch.float32).mean())


def build_edge_model(variant: str, scalers: dict[str, np.ndarray] | None = None, seed: int | None = None) -> FSARR2AR0EdgeModel:
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
    scalers = scalers or {}
    return FSARR2AR0EdgeModel(
        variant=str(variant),
        scaler_C=scalers.get("C_rms"),
        mask_C=scalers.get("C_mask"),
        scaler_P=scalers.get("P_rms"),
        mask_P=scalers.get("P_mask"),
        scaler_CE=scalers.get("CE_rms"),
        mask_CE=scalers.get("CE_mask"),
        scaler_PE=scalers.get("PE_rms"),
        mask_PE=scalers.get("PE_mask"),
    )


def parameter_breakdown_edge(model: FSARR2AR0EdgeModel) -> dict[str, Any]:
    def _count(module: nn.Module | None) -> int:
        if module is None:
            return 0
        return int(sum(parameter.numel() for parameter in module.parameters()))

    node_binding = int(model.W_B.numel())
    edge_binding = 0
    if model.W_E is not None:
        edge_binding += int(model.W_E.numel())
    if model.W_ME is not None:
        edge_binding += int(model.W_ME.numel())
    return {
        "variant": str(model.variant),
        "S_encoder": _count(model.s_encoder),
        "base_F0": _count(model.base),
        "node_binding": int(node_binding),
        "edge_binding": int(edge_binding),
        "total": _count(model),
        "dataset_dependent_vocabulary_params": 0,
    }


# ---------------------------------------------------------------------------
# evaluation-only bond-type permutation (undirected edges, within graph)
# ---------------------------------------------------------------------------


def permute_bond_types_within_graphs(batch: FSAREdgeBatch, seed: int) -> torch.Tensor:
    """Permute the bond-type rows among the **undirected** edges of each graph.

    Topology, atom attributes, atom assignment and the bond-type multiset are
    all held fixed; only which edge carries which bond type changes.
    """
    generator = torch.Generator().manual_seed(int(seed))
    r = batch.r.clone()
    edge_graph = batch.edge_graph.detach().cpu()
    for graph_id in range(int(batch.n_graphs)):
        mask = edge_graph == int(graph_id)
        indices = torch.nonzero(mask, as_tuple=False).view(-1)
        if indices.numel() <= 1:
            continue
        order = torch.randperm(indices.numel(), generator=generator)
        r[indices] = batch.r[indices[order]]
    return r


__all__ = [
    "PHI_DIM",
    "EDGE_ROLE_DIM",
    "BOND_CATEGORIES",
    "ATOM_CATEGORIES",
    "MODELS",
    "EDGE_MODELS",
    "NODE_ONLY",
    "EdgeMoleculeFeatures",
    "FSAREdgeBatch",
    "FSARR2AR0EdgeModel",
    "build_edge_roles",
    "one_hot_bond",
    "edge_center_stats",
    "segment_edge_stats",
    "fit_edge_scalers",
    "collate_edge_molecules",
    "make_edge_loader",
    "build_edge_model",
    "parameter_breakdown_edge",
    "permute_bond_types_within_graphs",
]

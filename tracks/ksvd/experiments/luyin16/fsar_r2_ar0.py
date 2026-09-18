"""FSAR-R2-AR0: fixed radius-2 explicit pure-topology coordinates + assignment residual.

This module is the *model / feature* half of the round named ``FSAR-R2-AR0``.
It deliberately contains no training infrastructure (that lives in
``zinc_fsar_r2_ar0.py``) so that every algebraic property demanded by the
pre-registration can be tested in isolation.

Definitions (frozen, no sweep)
------------------------------

A molecule is ``G = (T, X, E_attr)``:

* ``T``        -- the unlabelled topology (adjacency only);
* ``X``        -- node attributes (ZINC atom type, 28 padded categories);
* ``E_attr``   -- undirected edge attributes (ZINC bond type, 4 padded
  categories).

Node coordinate ``phi_v in R^65`` (pure topology, chemistry-free)::

    phi_v = [ root_basis(11),
              mean_node_basis(11), std_node_basis(11),
              mean_edge_basis(15), std_edge_basis(15),
              log1p(|V_{B2(v)}|), log1p(|E_{B2(v)}|) ]

The 11-D node basis and 15-D edge basis are the frozen explicit rooted
operator basis of :mod:`fsar_v2` (root indicator, restrained shell indicators,
induced degree, shell-resolved neighbour counts, rooted walks ``A^k``;
endpoint shell pairs, degree statistics, common neighbours, shell-neighbour
statistics).  ``|E_{B2(v)}|`` counts every induced **undirected** edge exactly
once.  No atom type / bond type ever enters ``phi_v``.

Whole-graph attribute marginal ``a(G) in R^64`` (whole-graph multiset only)::

    a = [ log1p(atom_counts)(28), atom_freq(28),
          log1p(bond_counts)(4),  bond_freq(4) ]

``a`` cannot see where an attribute sits.  ``q_v = onehot(x_v) in R^28``.

Three models, one shared base ``F0``:

* ``M0`` -- ``yhat = F0([z_S, a])``.
* ``MB`` -- ``yhat = F0([z_S, a]) + <W_B, C~>`` (aligned assignment residual).
* ``MM`` -- ``yhat = F0([z_S, a]) + <W_M, P~>`` (matched marginal capacity).

with ``z_S = [sum_v u_v, mean_v u_v, log1p(n), log1p(m)]``, ``u_v = MLP_S(phi_v)``
(``65 -> 64 -> 64``, SiLU),::

    J = sum_v phi_v q_v^T
    P = (1/n) (sum_v phi_v) (sum_v q_v)^T
    C = J - P = sum_v (phi_v - phibar)(q_v - qbar)^T

``C`` and ``P`` are normalised by *train-only* per-coordinate RMS
(``C~ = C / D_C``, ``P~ = P / D_P``); the dataset mean is **never** subtracted.
``W_B`` and ``W_M`` are bias-free ``[65, 28]`` matrices and are the *only* way
``C`` / ``P`` reach the prediction (no MLP, no activation, no bypass).

For any fixed parameters and a uniform random permutation ``pi`` of the node
attributes, ``E_pi[C(T, pi X)] = 0`` exactly, hence
``E_pi[<W_B, C(T, pi X)>] = 0``.  The convention ``C`` (sum centered) -- not
``C/(n-1)`` -- is what makes this identity hold without a residual factor.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Sequence

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import fsar_v2 as v2

# ---------------------------------------------------------------------------
# frozen schema / dimensions
# ---------------------------------------------------------------------------

ATOM_CATEGORIES = v2.ATOM_CATEGORIES  # 28 (ZINC pads 21 observed types)
BOND_CATEGORIES = v2.BOND_CATEGORIES  # 4  (ZINC uses 1..3)
PATCH_RADIUS = 2
NODE_BASIS_DIM = v2.NODE_BASIS_DIM  # 11
EDGE_BASIS_DIM = v2.EDGE_BASIS_DIM  # 15
PHI_DIM = 3 * NODE_BASIS_DIM + 2 * EDGE_BASIS_DIM + 2  # 65
A_DIM = 2 * ATOM_CATEGORIES + 2 * BOND_CATEGORIES  # 64

S_HIDDEN = 64
S_OUT = 64
Z_S_DIM = 2 * S_OUT + 2  # 130
BASE_HIDDEN = (64, 32)
BASE_INPUT_DIM = Z_S_DIM + A_DIM  # 194

MODELS = ("M0", "MB", "MM")
SINGLE_MODELS = ("M0",)
BINDING_MODELS = ("MB", "MM")

#: RMS values at or below this are treated as dead coordinates and masked out
#: (safe handling of zero / near-zero ``C`` / ``P`` coordinates).
SCALER_FLOOR = 1.0e-9
SCALER_EPS = 1.0e-12


def _act(name: str) -> nn.Module:
    key = str(name).lower()
    if key in {"silu", "swish"}:
        return nn.SiLU()
    if key in {"relu"}:
        return nn.ReLU()
    raise ValueError(f"unsupported activation {name!r}")


# ---------------------------------------------------------------------------
# pure-topology explicit coordinate
# ---------------------------------------------------------------------------


def _std_of_rows(rows: np.ndarray) -> np.ndarray:
    if rows.shape[0] == 0:
        return np.zeros(rows.shape[1], dtype=np.float64)
    return rows.std(axis=0)


def phi_for_center(graph: Any, center: int, radius: int = PATCH_RADIUS) -> np.ndarray:
    """Pure-topology 65-D coordinate for one original centre node."""
    nodes, index, node_basis, edge_basis, edges = v2._explicit_basis_for_patch(
        graph, int(center), int(radius)
    )
    root_position = index[int(center)]
    root_basis = node_basis[root_position]
    node_mean = (
        node_basis.mean(axis=0) if node_basis.shape[0] else np.zeros(NODE_BASIS_DIM)
    )
    node_std = _std_of_rows(node_basis)
    edge_mean = (
        edge_basis.mean(axis=0) if edge_basis.shape[0] else np.zeros(EDGE_BASIS_DIM)
    )
    edge_std = _std_of_rows(edge_basis)
    scales = np.asarray(
        [np.log1p(len(nodes)), np.log1p(len(edges))], dtype=np.float64
    )
    phi = np.concatenate(
        [root_basis, node_mean, node_std, edge_mean, edge_std, scales]
    ).astype(np.float64)
    if phi.shape[0] != PHI_DIM:
        raise RuntimeError(f"phi width {phi.shape[0]} != {PHI_DIM}")
    return phi


def build_phi(graph: Any, radius: int = PATCH_RADIUS) -> np.ndarray:
    """``[n_nodes, 65]`` pure-topology coordinates, one row per original node."""
    return np.stack(
        [phi_for_center(graph, int(center), int(radius)) for center in graph.nodes],
        axis=0,
    )


# ---------------------------------------------------------------------------
# whole-graph attribute marginal A
# ---------------------------------------------------------------------------


def build_A(
    node_types: np.ndarray,
    edge_types: Any,
    n_nodes: int,
    atom_categories: int = ATOM_CATEGORIES,
    bond_categories: int = BOND_CATEGORIES,
) -> np.ndarray:
    """Whole-graph marginal ``a(G) in R^64``.

    ``edge_types`` may be a mapping ``(u, v) -> bond type`` keyed by the sorted
    node pair (each undirected edge once) or a sequence of undirected bond
    types.  Undirected edges are counted exactly once.
    """
    atom = np.asarray(node_types, dtype=np.int64).reshape(-1)
    if atom.shape[0] != int(n_nodes):
        raise RuntimeError("node_types / n_nodes mismatch")
    atom_counts = np.bincount(atom, minlength=int(atom_categories)).astype(np.float64)
    atom_counts = atom_counts[: int(atom_categories)]
    if isinstance(edge_types, dict):
        bond_values = list(edge_types.values())
    else:
        bond_values = list(edge_types)
    if bond_values:
        bond = np.asarray(bond_values, dtype=np.int64).reshape(-1)
        bond_counts = np.bincount(bond, minlength=int(bond_categories)).astype(
            np.float64
        )
    else:
        bond_counts = np.zeros(int(bond_categories), dtype=np.float64)
    bond_counts = bond_counts[: int(bond_categories)]
    n = float(max(int(n_nodes), 1))
    m = float(max(int(bond_counts.sum()), 1))
    return np.concatenate(
        [
            np.log1p(atom_counts),
            atom_counts / n,
            np.log1p(bond_counts),
            bond_counts / m,
        ]
    ).astype(np.float64)


# ---------------------------------------------------------------------------
# assignment statistics (numpy reference + torch batched implementation)
# ---------------------------------------------------------------------------


def one_hot_q(
    atom_idx: np.ndarray, atom_categories: int = ATOM_CATEGORIES
) -> np.ndarray:
    idx = np.asarray(atom_idx, dtype=np.int64).reshape(-1)
    out = np.zeros((idx.shape[0], int(atom_categories)), dtype=np.float64)
    valid = (idx >= 0) & (idx < int(atom_categories))
    out[np.arange(idx.shape[0])[valid], idx[valid]] = 1.0
    return out


def center_stats(phi: np.ndarray, q: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Reference ``(C, P)`` for one molecule.

    ``C = J - P``, ``J = phi^T q``, ``P = (1/n)(sum phi)(sum q)^T``.
    Uses the **sum-centred** statistic (no division by ``n-1``).
    """
    phi = np.asarray(phi, dtype=np.float64)
    q = np.asarray(q, dtype=np.float64)
    if phi.shape[0] != q.shape[0]:
        raise RuntimeError("phi / q node count mismatch")
    n = float(phi.shape[0])
    if n <= 0.0:
        raise RuntimeError("empty molecule")
    phibar = phi.mean(axis=0)
    qbar = q.mean(axis=0)
    j_matrix = phi.T @ q
    p_matrix = n * np.outer(phibar, qbar)
    return j_matrix - p_matrix, p_matrix


def graph_indicator(node_graph: torch.Tensor, n_graphs: int) -> torch.Tensor:
    """Dense ``[N, B]`` segment indicator, built and consumed deterministically."""
    node_graph = node_graph.long().view(-1, 1)
    indicator = torch.zeros(
        node_graph.shape[0], int(n_graphs), device=node_graph.device, dtype=torch.float32
    )
    indicator.scatter_(1, node_graph, 1.0)
    return indicator


def segment_center_stats(
    phi: torch.Tensor,
    q: torch.Tensor,
    node_graph: torch.Tensor,
    n_graphs: int,
    indicator: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Batched torch ``(C, P)`` with shape ``[n_graphs, 65, 28]``.

    Uses only dense matmuls (no atomic ``index_add_``) so the result is
    bit-reproducible under ``torch.use_deterministic_algorithms(True)``.
    """
    if indicator is None:
        indicator = graph_indicator(node_graph, n_graphs)
    indicator = indicator.to(phi.dtype)
    width, categories = int(phi.shape[1]), int(q.shape[1])
    outer = (phi.unsqueeze(2) * q.unsqueeze(1)).reshape(phi.shape[0], width * categories)
    j_matrix = (indicator.t() @ outer).reshape(int(n_graphs), width, categories)
    sum_phi = indicator.t() @ phi
    sum_q = indicator.t() @ q
    counts = indicator.sum(dim=0).clamp_min(1.0)
    p_matrix = (sum_phi.unsqueeze(2) * sum_q.unsqueeze(1)) / counts.view(-1, 1, 1)
    return j_matrix - p_matrix, p_matrix


def fit_scalers(
    c_matrices: Sequence[np.ndarray],
    p_matrices: Sequence[np.ndarray],
    floor: float = SCALER_FLOOR,
    eps: float = SCALER_EPS,
) -> dict[str, np.ndarray]:
    """Train-only per-coordinate RMS scalers and effective-coordinate masks.

    A coordinate is *effective* when its raw train RMS is above ``floor``.
    Ineffective coordinates are masked to zero (safe handling); effective ones
    are divided by ``sqrt(mean(C^2) + eps)``.  The dataset mean is never
    subtracted.
    """
    if not c_matrices:
        raise RuntimeError("no training statistics provided")
    c_stack = np.stack([np.asarray(item, dtype=np.float64) for item in c_matrices])
    p_stack = np.stack([np.asarray(item, dtype=np.float64) for item in p_matrices])
    c_rms_raw = np.sqrt((c_stack**2).mean(axis=0))
    p_rms_raw = np.sqrt((p_stack**2).mean(axis=0))
    c_mask = (c_rms_raw > float(floor)).astype(np.float64)
    p_mask = (p_rms_raw > float(floor)).astype(np.float64)
    c_scale = np.where(c_mask > 0.0, np.sqrt((c_stack**2).mean(axis=0) + float(eps)), 1.0)
    p_scale = np.where(p_mask > 0.0, np.sqrt((p_stack**2).mean(axis=0) + float(eps)), 1.0)
    return {
        "C_rms": c_scale.astype(np.float32),
        "C_mask": c_mask.astype(np.float32),
        "P_rms": p_scale.astype(np.float32),
        "P_mask": p_mask.astype(np.float32),
        "C_rms_raw": c_rms_raw.astype(np.float64),
        "P_rms_raw": p_rms_raw.astype(np.float64),
    }


def normalized_stat(
    value: torch.Tensor, scale: torch.Tensor, mask: torch.Tensor
) -> torch.Tensor:
    return (value / scale) * mask


# ---------------------------------------------------------------------------
# batching
# ---------------------------------------------------------------------------


@dataclass
class MoleculeFeatures:
    """Cached per-molecule features (all pure / assignment-audited upstream)."""

    phi: np.ndarray  # [n, 65] float32 pure-topology
    atom_idx: np.ndarray  # [n] int64 atom category index
    A: np.ndarray  # [64] float32 whole-graph marginal
    n_nodes: int
    n_edges: int
    y: float


@dataclass
class FSARBatch:
    phi: torch.Tensor  # [N, 65]
    q: torch.Tensor  # [N, 28]
    node_graph: torch.Tensor  # [N]
    A: torch.Tensor  # [B, 64]
    n_nodes: torch.Tensor  # [B]
    n_edges: torch.Tensor  # [B]
    y: torch.Tensor  # [B]
    n_graphs: int

    def to(self, device: torch.device | str) -> "FSARBatch":
        device = torch.device(device)
        return FSARBatch(
            phi=self.phi.to(device),
            q=self.q.to(device),
            node_graph=self.node_graph.to(device),
            A=self.A.to(device),
            n_nodes=self.n_nodes.to(device),
            n_edges=self.n_edges.to(device),
            y=self.y.to(device),
            n_graphs=int(self.n_graphs),
        )

    def with_q(self, q: torch.Tensor) -> "FSARBatch":
        if q.shape != self.q.shape:
            raise RuntimeError("replacement q shape mismatch")
        return FSARBatch(
            phi=self.phi,
            q=q,
            node_graph=self.node_graph,
            A=self.A,
            n_nodes=self.n_nodes,
            n_edges=self.n_edges,
            y=self.y,
            n_graphs=int(self.n_graphs),
        )


def collate_molecules(
    molecules: Sequence[MoleculeFeatures], atom_categories: int = ATOM_CATEGORIES
) -> FSARBatch:
    phi_parts: list[torch.Tensor] = []
    q_parts: list[torch.Tensor] = []
    graph_index: list[torch.Tensor] = []
    a_rows: list[torch.Tensor] = []
    n_nodes: list[int] = []
    n_edges: list[int] = []
    targets: list[float] = []
    for graph_id, molecule in enumerate(molecules):
        phi = torch.as_tensor(np.asarray(molecule.phi, dtype=np.float32))
        q = torch.as_tensor(
            one_hot_q(molecule.atom_idx, atom_categories).astype(np.float32)
        )
        phi_parts.append(phi)
        q_parts.append(q)
        graph_index.append(
            torch.full((phi.shape[0],), int(graph_id), dtype=torch.long)
        )
        a_rows.append(torch.as_tensor(np.asarray(molecule.A, dtype=np.float32)))
        n_nodes.append(int(molecule.n_nodes))
        n_edges.append(int(molecule.n_edges))
        targets.append(float(molecule.y))
    return FSARBatch(
        phi=torch.cat(phi_parts, dim=0),
        q=torch.cat(q_parts, dim=0),
        node_graph=torch.cat(graph_index, dim=0),
        A=torch.stack(a_rows, dim=0),
        n_nodes=torch.as_tensor(n_nodes, dtype=torch.float32),
        n_edges=torch.as_tensor(n_edges, dtype=torch.float32),
        y=torch.as_tensor(targets, dtype=torch.float32),
        n_graphs=int(len(molecules)),
    )


def make_loader(
    molecules: Sequence[MoleculeFeatures],
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
        collate_fn=collate_molecules,
    )


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


class FSARR2AR0Model(nn.Module):
    """Shared ``F0`` base plus an optional linear assignment / marginal term."""

    def __init__(
        self,
        variant: str = "M0",
        *,
        phi_dim: int = PHI_DIM,
        atom_categories: int = ATOM_CATEGORIES,
        a_dim: int = A_DIM,
        s_hidden: int = S_HIDDEN,
        s_out: int = S_OUT,
        base_hidden: Sequence[int] = BASE_HIDDEN,
        activation: str = "silu",
        scaler_C: np.ndarray | None = None,
        mask_C: np.ndarray | None = None,
        scaler_P: np.ndarray | None = None,
        mask_P: np.ndarray | None = None,
    ) -> None:
        super().__init__()
        variant = str(variant)
        if variant not in MODELS:
            raise ValueError(f"unknown variant {variant!r}; expected {MODELS}")
        self.variant = variant
        self.phi_dim = int(phi_dim)
        self.atom_categories = int(atom_categories)
        self.a_dim = int(a_dim)
        self.s_out = int(s_out)
        self.z_s_dim = 2 * int(s_out) + 2
        self.base_input_dim = self.z_s_dim + self.a_dim

        self.s_encoder = nn.Sequential(
            nn.Linear(self.phi_dim, int(s_hidden)),
            _act(activation),
            nn.Linear(int(s_hidden), int(s_out)),
        )
        base_layers: list[nn.Module] = []
        widths = [self.base_input_dim, *[int(width) for width in base_hidden], 1]
        for index in range(len(widths) - 1):
            base_layers.append(nn.Linear(widths[index], widths[index + 1]))
            if index < len(widths) - 2:
                base_layers.append(_act(activation))
        self.base = nn.Sequential(*base_layers)

        self.W_B: nn.Parameter | None = None
        self.W_M: nn.Parameter | None = None
        if variant == "MB":
            self.W_B = nn.Parameter(
                torch.zeros(self.phi_dim, self.atom_categories)
            )
        elif variant == "MM":
            self.W_M = nn.Parameter(
                torch.zeros(self.phi_dim, self.atom_categories)
            )

        def _buffer(value: np.ndarray | None, name: str) -> None:
            array = (
                np.zeros((self.phi_dim, self.atom_categories), dtype=np.float32)
                if value is None
                else np.asarray(value, dtype=np.float32)
            )
            if array.shape != (self.phi_dim, self.atom_categories):
                raise RuntimeError(f"{name} shape {array.shape} mismatch")
            self.register_buffer(name, torch.from_numpy(array.copy()))

        _buffer(scaler_C, "scaler_C")
        _buffer(mask_C, "mask_C")
        _buffer(scaler_P, "scaler_P")
        _buffer(mask_P, "mask_P")

        self.capture_diagnostics = False
        self.last_stats: dict[str, float] = {}

    # -- geometry -----------------------------------------------------------

    def compute_statistics(
        self, batch: FSARBatch, indicator: torch.Tensor | None = None
    ) -> tuple[torch.Tensor, torch.Tensor]:
        return segment_center_stats(
            batch.phi,
            batch.q,
            batch.node_graph,
            int(batch.n_graphs),
            indicator=indicator,
        )

    def encode_s(
        self, batch: FSARBatch, indicator: torch.Tensor | None = None
    ) -> torch.Tensor:
        u = self.s_encoder(batch.phi)
        if indicator is None:
            indicator = graph_indicator(batch.node_graph, int(batch.n_graphs))
        indicator = indicator.to(u.dtype)
        sum_u = indicator.t() @ u
        counts = indicator.sum(dim=0).clamp_min(1.0)
        mean_u = sum_u / counts.view(-1, 1)
        n_log = torch.log1p(batch.n_nodes).view(-1, 1)
        m_log = torch.log1p(batch.n_edges).view(-1, 1)
        return torch.cat([sum_u, mean_u, n_log, m_log], dim=1)

    def base_prediction(
        self, batch: FSARBatch, indicator: torch.Tensor | None = None
    ) -> torch.Tensor:
        z_s = self.encode_s(batch, indicator=indicator)
        return self.base(torch.cat([z_s, batch.A], dim=1)).view(-1)

    def assignment_term(
        self, c_matrix: torch.Tensor, p_matrix: torch.Tensor
    ) -> torch.Tensor:
        if self.variant == "MB":
            assert self.W_B is not None
            statistic = normalized_stat(c_matrix, self.scaler_C, self.mask_C)
            weight = self.W_B
        elif self.variant == "MM":
            assert self.W_M is not None
            statistic = normalized_stat(p_matrix, self.scaler_P, self.mask_P)
            weight = self.W_M
        else:
            return c_matrix.new_zeros(c_matrix.shape[0])
        return (statistic * weight.unsqueeze(0)).sum(dim=(1, 2))

    def forward(self, batch: FSARBatch) -> torch.Tensor:
        indicator = graph_indicator(batch.node_graph, int(batch.n_graphs))
        base = self.base_prediction(batch, indicator=indicator)
        if self.variant == "M0":
            return base
        c_matrix, p_matrix = self.compute_statistics(batch, indicator=indicator)
        return base + self.assignment_term(c_matrix, p_matrix)

    # -- diagnostics --------------------------------------------------------

    def branch_weight_norm(self) -> float:
        if self.variant == "MB" and self.W_B is not None:
            return float(self.W_B.detach().norm())
        if self.variant == "MM" and self.W_M is not None:
            return float(self.W_M.detach().norm())
        return 0.0

    def branch_weight_abs_sum(self) -> float:
        if self.variant == "MB" and self.W_B is not None:
            return float(self.W_B.detach().abs().sum())
        if self.variant == "MM" and self.W_M is not None:
            return float(self.W_M.detach().abs().sum())
        return 0.0

    def branch_weight_sparsity(self, threshold: float = 1.0e-8) -> float:
        if self.variant == "MB" and self.W_B is not None:
            weight = self.W_B.detach()
        elif self.variant == "MM" and self.W_M is not None:
            weight = self.W_M.detach()
        else:
            return 0.0
        return float((weight.abs() <= float(threshold)).to(torch.float32).mean())


def build_model(
    variant: str,
    scalers: dict[str, np.ndarray] | None = None,
    seed: int | None = None,
) -> FSARR2AR0Model:
    if seed is not None:
        torch.manual_seed(int(seed))
        np.random.seed(int(seed))
    scalers = scalers or {}
    return FSARR2AR0Model(
        variant=str(variant),
        scaler_C=scalers.get("C_rms"),
        mask_C=scalers.get("C_mask"),
        scaler_P=scalers.get("P_rms"),
        mask_P=scalers.get("P_mask"),
    )


def parameter_breakdown(model: FSARR2AR0Model) -> dict[str, Any]:
    def _count(module: nn.Module | None) -> int:
        if module is None:
            return 0
        return int(sum(parameter.numel() for parameter in module.parameters()))

    binding = 0
    if model.W_B is not None:
        binding += int(model.W_B.numel())
    if model.W_M is not None:
        binding += int(model.W_M.numel())
    return {
        "variant": str(model.variant),
        "S_encoder": _count(model.s_encoder),
        "base_F0": _count(model.base),
        "binding_linear": int(binding),
        "total": _count(model),
        "dataset_dependent_vocabulary_params": 0,
    }


# ---------------------------------------------------------------------------
# evaluation-only assignment permutation (original nodes, propagated once)
# ---------------------------------------------------------------------------


def permute_q_within_graphs(batch: FSARBatch, seed: int) -> torch.Tensor:
    """Permute atom-attribute rows among the original nodes of each graph.

    The permutation acts on the **original molecule nodes**; there are no patch
    copies in this representation, so nothing further has to be propagated.
    """
    generator = torch.Generator().manual_seed(int(seed))
    q = batch.q.clone()
    node_graph = batch.node_graph.detach().cpu()
    for graph_id in range(int(batch.n_graphs)):
        mask = node_graph == int(graph_id)
        indices = torch.nonzero(mask, as_tuple=False).view(-1)
        if indices.numel() <= 1:
            continue
        order = torch.randperm(indices.numel(), generator=generator)
        q[indices] = batch.q[indices[order]]
    return q


__all__ = [
    "ATOM_CATEGORIES",
    "BOND_CATEGORIES",
    "PATCH_RADIUS",
    "PHI_DIM",
    "A_DIM",
    "Z_S_DIM",
    "BASE_INPUT_DIM",
    "MODELS",
    "MoleculeFeatures",
    "FSARBatch",
    "FSARR2AR0Model",
    "build_phi",
    "phi_for_center",
    "build_A",
    "one_hot_q",
    "center_stats",
    "segment_center_stats",
    "fit_scalers",
    "normalized_stat",
    "collate_molecules",
    "make_loader",
    "build_model",
    "parameter_breakdown",
    "permute_q_within_graphs",
]

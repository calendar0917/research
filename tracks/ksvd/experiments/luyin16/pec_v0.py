"""PEC-v0 core — Pure Environment Composition (no MP / no recurrence / no bypass).

Pre-registration: ``tracks/ksvd/notes/pec_v0_preregistration.md``.
Prior-artifact audit: ``tracks/ksvd/notes/pec_v0_prior_artifact_audit.md``.

Object (frozen)
---------------
For every root atom ``i`` of a molecule take the radius-2 induced patch ``P_i``.
Every *occurrence* ``(i, v)`` of a node ``v`` inside ``P_i`` has the audited
pure-topology FSAR explicit rooted node basis row ``b^V_{iv} in R^11`` (root
indicator, shell one-hot, induced degree, shell-resolved neighbour counts,
rooted walks).  Every occurrence ``(i, e)`` of an undirected induced bond has
the audited edge basis row ``b^E_{ie} in R^15`` (endpoint shell pair,
degree statistics, common neighbours, shell-resolved neighbour stats).

    alpha^V_{iv} = SparseDict_V(b^V_{iv})              # R^16, exact l0 <= 4
    r^V_{iv}     = [ one_hot(shell(i,v), 3) ; alpha^V_{iv} ]      # 19
    B^V_i        = sum_{v in P_i} r^V_{iv} (x) a(q_v)             # 19 x 28
    alpha^E_{ie} = SparseDict_E(b^E_{ie})
    r^E_{ie}     = [ one_hot(shellpair(i,e), 5) ; alpha^E_{ie} ]  # 21
    B^E_i        = sum_{e in Q_i} r^E_{ie} (x) c(b_e)             # 21 x  4
    E_i          = H([ a(q_i) ; vec(B^V_i) ; vec(B^E_i) ; S_i ]) -> R^48

``alpha`` is the *only* learned structural coordinate; the basis rows are the
audited chemistry-free FSAR bases.  ``E_i`` is frozen before any pair
information exists.  The cross-root computation is exactly one read-only static
pair pass ``c_ij = F(E_i, E_j, rho_ij)`` followed by graph-level pooling.

Forbidden anywhere in this module: message passing, pair->centre / centre
update, recurrence, relation refresh, attention, any raw typed readout, any
global typed histogram, ``path_bond_mean``, adjacent-bond chemistry.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.experiments.luyin16 import fsar_r2_ar0 as r2
from tracks.ksvd.experiments.luyin16 import fsar_v2 as v2

# ---------------------------------------------------------------------------
# frozen constants
# ---------------------------------------------------------------------------

NODE_ROLE_DIM = int(v2.NODE_BASIS_DIM)  # 11 explicit rooted node basis
EDGE_ROLE_DIM = int(v2.EDGE_BASIS_DIM)  # 15 explicit rooted edge basis
PHI_DIM = int(r2.PHI_DIM)  # 65, retained only for provenance checks
ATOM_CATEGORIES = int(r2.ATOM_CATEGORIES)  # 28
BOND_CATEGORIES = int(r2.BOND_CATEGORIES)  # 4

K_V = 16
S_V = 4
K_E = 16
S_E = 4
IHT_STEPS = 10

SHELL_CLASSES = 3  # 0,1,2
SHELLPAIR_CLASSES = 5  # (0,1),(0,2),(1,1),(1,2),(2,2)
SHELLPAIRS: tuple[tuple[int, int], ...] = ((0, 1), (0, 2), (1, 1), (1, 2), (2, 2))

TOPO_ROOT_DIM = 6
TOPO_PAIR_DIM = 18  # 8 distance one-hot + 1 log dist + 5 overlap + 3 boundary + 1 log path
TOPO_GLOBAL_DIM = 8

ENV_DIM = 48
PAIR_DIM = 48
ENV_HIDDEN = 96
PAIR_HIDDEN = 64
READER_HIDDEN = 64

PATCH_RADIUS = 2
DISTANCE_BUCKETS = 8

ROLE_MODES = ("coarse", "dense", "sparse")


def shellpair_index(shell_a: int, shell_b: int) -> int:
    """Fixed rooted structural role of an undirected bond inside a patch."""
    pair = (int(min(shell_a, shell_b)), int(max(shell_a, shell_b)))
    try:
        return SHELLPAIRS.index(pair)  # type: ignore[arg-type]
    except ValueError as exc:  # pragma: no cover - defensive
        raise ValueError(f"illegal shell pair {pair!r} for radius {PATCH_RADIUS}") from exc


# ---------------------------------------------------------------------------
# per-molecule sample (numpy; topology roles and chemistry kept apart)
# ---------------------------------------------------------------------------


@dataclass
class MoleculeSample:
    """One molecule with occurrence-level rooted roles and chemistry kept apart."""

    node_basis: np.ndarray  # [T, 11] float32 pure topology, occurrence (i, v)
    occ_root: np.ndarray  # [T] int64 root index i
    occ_node: np.ndarray  # [T] int64 node index v
    occ_shell: np.ndarray  # [T] int64 shell(i, v)  (pure topology)
    edge_basis: np.ndarray  # [U, 15] float32 pure topology, occurrence (i, e)
    eocc_root: np.ndarray  # [U] int64 root index i
    eocc_edge: np.ndarray  # [U] int64 undirected edge index e
    eocc_shellpair: np.ndarray  # [U] int64 shell pair (pure topology)
    atom_idx: np.ndarray  # [n] int64 chemistry only
    bond_idx: np.ndarray  # [m] int64 chemistry only
    root_scalars: np.ndarray  # [n, 6] float64 topology only
    pair_i: np.ndarray  # [P] int64
    pair_j: np.ndarray  # [P] int64
    pair_rho: np.ndarray  # [P, 18] float32 topology only
    global_topo: np.ndarray  # [8] float64 topology only
    y: float = 0.0
    n_nodes: int = 0
    n_edges: int = 0


def _boundary_features(left_boundary: set[int], right_boundary: set[int]) -> np.ndarray:
    boundary_intersection = len(left_boundary & right_boundary)
    boundary_union = len(left_boundary | right_boundary)
    boundary_min = min(len(left_boundary), len(right_boundary))
    return np.asarray(
        [
            float(boundary_intersection) / max(float(boundary_union), 1.0),
            float(boundary_intersection) / max(float(boundary_min), 1.0),
            float(boundary_intersection > 0),
        ],
        dtype=np.float64,
    )


def _patch_pair_relation(
    left_nodes: set[int],
    right_nodes: set[int],
    left_boundary: set[int],
    right_boundary: set[int],
    distance: int,
    path_count: float,
) -> np.ndarray:
    """Topology-only subset of the strict-static S0 pair relation (18-D).

    The two chemistry-bearing S0 blocks -- ``path_bond_mean`` (4) and the
    adjacent-bond one-hot (4) -- are deleted by construction.
    """
    bucket = min(max(int(distance), 1), DISTANCE_BUCKETS) - 1
    distance_one_hot = np.zeros(DISTANCE_BUCKETS, dtype=np.float64)
    distance_one_hot[bucket] = 1.0

    intersection = len(left_nodes & right_nodes)
    union = len(left_nodes | right_nodes)
    left_size = len(left_nodes)
    right_size = len(right_nodes)
    overlap = np.asarray(
        [
            float(intersection) / max(float(PATCH_RADIUS * PATCH_RADIUS + 10), 1.0),
            float(intersection) / max(float(union), 1.0),
            float(intersection) / max(float(min(left_size, right_size)), 1.0),
            float(intersection) / max(float(max(left_size, right_size)), 1.0),
            float(abs(left_size - right_size))
            / max(float(PATCH_RADIUS * PATCH_RADIUS + 10), 1.0),
        ],
        dtype=np.float64,
    )
    relation = np.concatenate(
        [
            distance_one_hot,
            np.asarray([np.log1p(float(distance))], dtype=np.float64),
            overlap,
            _boundary_features(left_boundary, right_boundary),
            np.asarray([np.log1p(float(path_count))], dtype=np.float64),
        ]
    )
    if relation.shape[0] != TOPO_PAIR_DIM:
        raise RuntimeError(f"pair relation width {relation.shape[0]} != {TOPO_PAIR_DIM}")
    return relation.astype(np.float32)


def build_sample(
    graph: Any,
    node_types: Sequence[int],
    edge_types: Mapping[tuple[int, int], int],
    *,
    y: float = 0.0,
) -> MoleculeSample:
    """Build one ``MoleculeSample`` from an untyped graph + chemistry mappings.

    ``graph`` contributes topology only; ``node_types`` / ``edge_types``
    contribute chemistry only.  The two are consumed by disjoint code paths and
    only meet inside the binding tensor at model time.
    """
    from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

    nodes = list(graph.nodes)
    n = len(nodes)
    if n == 0:
        raise ValueError("empty molecule")

    undirected = sorted(
        {graph.edge_key(int(u), int(v)) for u, v in graph.edges()}
    )
    edges_undirected: list[tuple[int, int]] = [(int(u), int(v)) for u, v in undirected]
    bond_idx = np.asarray(
        [int(edge_types.get((int(u), int(v)), 0)) for u, v in edges_undirected],
        dtype=np.int64,
    )
    edge_lookup = {key: index for index, key in enumerate(edges_undirected)}

    node_basis_parts: list[np.ndarray] = []
    occ_root: list[int] = []
    occ_node: list[int] = []
    occ_shell: list[int] = []
    edge_basis_parts: list[np.ndarray] = []
    eocc_root: list[int] = []
    eocc_edge: list[int] = []
    eocc_shellpair: list[int] = []
    patch_sets: list[set[int]] = []
    patch_boundaries: list[set[int]] = []
    root_scalars = np.zeros((n, TOPO_ROOT_DIM), dtype=np.float64)
    degree = np.asarray(
        [len(graph.neighbors(int(node))) for node in nodes], dtype=np.float64
    )
    mean_degree = float(degree.mean()) if n else 0.0
    cycle_rank = float(max(len(edges_undirected) - n + 1, 0))

    for root in nodes:
        root = int(root)
        nodes_i, _index_i, node_basis, edge_basis, edges_i = v2._explicit_basis_for_patch(
            graph, root, PATCH_RADIUS
        )
        shell_of = {
            int(node): int(np.argmax(node_basis[position, 1 : 1 + SHELL_CLASSES]))
            for position, node in enumerate(nodes_i)
        }
        node_basis_parts.append(np.asarray(node_basis, dtype=np.float32))
        for node in nodes_i:
            occ_root.append(root)
            occ_node.append(int(node))
            occ_shell.append(int(shell_of[int(node)]))
        edge_basis_parts.append(np.asarray(edge_basis, dtype=np.float32))
        for left, right in edges_i:
            key = graph.edge_key(int(left), int(right))
            eocc_root.append(root)
            eocc_edge.append(int(edge_lookup[key]))
            eocc_shellpair.append(
                shellpair_index(shell_of[int(left)], shell_of[int(right)])
            )
        patch_sets.append({int(node) for node in nodes_i})
        patch_boundaries.append(
            {int(node) for node in nodes_i if shell_of[int(node)] == PATCH_RADIUS}
        )
        position = nodes.index(root)
        root_scalars[position] = np.asarray(
            [
                np.log1p(float(len(nodes_i))),
                np.log1p(float(len(edges_i))),
                float(degree[position]),
                mean_degree,
                float(len(patch_boundaries[-1])) / max(float(len(nodes_i)), 1.0),
                cycle_rank,
            ],
            dtype=np.float64,
        )

    # topology-only pair relations
    pair_i: list[int] = []
    pair_j: list[int] = []
    pair_rho: list[np.ndarray] = []
    for left_index, left_center in enumerate(nodes):
        summaries = zpp._shortest_path_summary(graph, int(left_center), edge_types)
        for right_index in range(left_index + 1, len(nodes)):
            right_center = int(nodes[right_index])
            distance, path_count, _bond_mean = summaries[right_center]
            pair_i.append(int(left_index))
            pair_j.append(int(right_index))
            pair_rho.append(
                _patch_pair_relation(
                    patch_sets[left_index],
                    patch_sets[right_index],
                    patch_boundaries[left_index],
                    patch_boundaries[right_index],
                    int(distance),
                    float(path_count),
                )
            )

    global_topo = np.asarray(
        [
            np.log1p(float(n)),
            np.log1p(float(len(edges_undirected))),
            cycle_rank,
            mean_degree,
            float(degree.std()) if n else 0.0,
            float(degree.max()) if n else 0.0,
            float(degree.min()) if n else 0.0,
            float(n),
        ],
        dtype=np.float64,
    )

    return MoleculeSample(
        node_basis=np.concatenate(node_basis_parts, axis=0),
        occ_root=np.asarray(occ_root, dtype=np.int64),
        occ_node=np.asarray(occ_node, dtype=np.int64),
        occ_shell=np.asarray(occ_shell, dtype=np.int64),
        edge_basis=np.concatenate(edge_basis_parts, axis=0),
        eocc_root=np.asarray(eocc_root, dtype=np.int64),
        eocc_edge=np.asarray(eocc_edge, dtype=np.int64),
        eocc_shellpair=np.asarray(eocc_shellpair, dtype=np.int64),
        atom_idx=np.asarray(node_types, dtype=np.int64).reshape(-1),
        bond_idx=bond_idx,
        root_scalars=root_scalars,
        pair_i=np.asarray(pair_i, dtype=np.int64),
        pair_j=np.asarray(pair_j, dtype=np.int64),
        pair_rho=(
            np.stack(pair_rho, axis=0).astype(np.float32)
            if pair_rho
            else np.zeros((0, TOPO_PAIR_DIM), dtype=np.float32)
        ),
        global_topo=global_topo,
        y=float(y),
        n_nodes=int(n),
        n_edges=int(len(edges_undirected)),
    )


# ---------------------------------------------------------------------------
# collation
# ---------------------------------------------------------------------------


def _cat_or_empty(parts: Sequence[torch.Tensor], width: int) -> torch.Tensor:
    if parts:
        return torch.cat(list(parts), dim=0)
    return torch.zeros((0, width), dtype=torch.float32)


def collate(samples: Sequence[MoleculeSample]) -> dict[str, torch.Tensor]:
    """Flatten molecules into one batch with explicit graph/offset bookkeeping."""
    node_basis: list[torch.Tensor] = []
    occ_root: list[torch.Tensor] = []
    occ_node: list[torch.Tensor] = []
    occ_shell: list[torch.Tensor] = []
    occ_graph: list[torch.Tensor] = []
    edge_basis: list[torch.Tensor] = []
    eocc_root: list[torch.Tensor] = []
    eocc_edge: list[torch.Tensor] = []
    eocc_shellpair: list[torch.Tensor] = []
    eocc_graph: list[torch.Tensor] = []
    atom_idx: list[torch.Tensor] = []
    bond_idx: list[torch.Tensor] = []
    node_graph: list[torch.Tensor] = []
    edge_graph: list[torch.Tensor] = []
    root_scalars: list[torch.Tensor] = []
    pair_left: list[torch.Tensor] = []
    pair_right: list[torch.Tensor] = []
    pair_rho: list[torch.Tensor] = []
    pair_graph: list[torch.Tensor] = []
    global_topo: list[torch.Tensor] = []
    targets: list[float] = []
    node_offset = 0
    edge_offset = 0
    for graph_id, sample in enumerate(samples):
        n = int(sample.n_nodes)
        m = int(sample.bond_idx.shape[0])
        node_basis.append(torch.as_tensor(sample.node_basis, dtype=torch.float32))
        occ_root.append(torch.as_tensor(sample.occ_root, dtype=torch.long) + node_offset)
        occ_node.append(torch.as_tensor(sample.occ_node, dtype=torch.long) + node_offset)
        occ_shell.append(torch.as_tensor(sample.occ_shell, dtype=torch.long))
        occ_graph.append(torch.full((sample.occ_root.shape[0],), graph_id, dtype=torch.long))
        edge_basis.append(torch.as_tensor(sample.edge_basis, dtype=torch.float32))
        eocc_root.append(torch.as_tensor(sample.eocc_root, dtype=torch.long) + node_offset)
        eocc_edge.append(torch.as_tensor(sample.eocc_edge, dtype=torch.long) + edge_offset)
        eocc_shellpair.append(torch.as_tensor(sample.eocc_shellpair, dtype=torch.long))
        eocc_graph.append(torch.full((sample.eocc_root.shape[0],), graph_id, dtype=torch.long))
        atom_idx.append(torch.as_tensor(sample.atom_idx, dtype=torch.long))
        bond_idx.append(torch.as_tensor(sample.bond_idx, dtype=torch.long))
        node_graph.append(torch.full((n,), graph_id, dtype=torch.long))
        edge_graph.append(torch.full((m,), graph_id, dtype=torch.long))
        root_scalars.append(torch.as_tensor(sample.root_scalars, dtype=torch.float32))
        pair_left.append(torch.as_tensor(sample.pair_i, dtype=torch.long) + node_offset)
        pair_right.append(torch.as_tensor(sample.pair_j, dtype=torch.long) + node_offset)
        pair_rho.append(torch.as_tensor(sample.pair_rho, dtype=torch.float32))
        pair_graph.append(torch.full((sample.pair_i.shape[0],), graph_id, dtype=torch.long))
        global_topo.append(torch.as_tensor(sample.global_topo, dtype=torch.float32))
        targets.append(float(sample.y))
        node_offset += n
        edge_offset += m

    return {
        "node_basis": _cat_or_empty(node_basis, NODE_ROLE_DIM),
        "occ_root": torch.cat(occ_root, dim=0) if occ_root else torch.zeros(0, dtype=torch.long),
        "occ_node": torch.cat(occ_node, dim=0) if occ_node else torch.zeros(0, dtype=torch.long),
        "occ_shell": torch.cat(occ_shell, dim=0) if occ_shell else torch.zeros(0, dtype=torch.long),
        "occ_graph": torch.cat(occ_graph, dim=0) if occ_graph else torch.zeros(0, dtype=torch.long),
        "edge_basis": _cat_or_empty(edge_basis, EDGE_ROLE_DIM),
        "eocc_root": torch.cat(eocc_root, dim=0) if eocc_root else torch.zeros(0, dtype=torch.long),
        "eocc_edge": torch.cat(eocc_edge, dim=0) if eocc_edge else torch.zeros(0, dtype=torch.long),
        "eocc_shellpair": torch.cat(eocc_shellpair, dim=0) if eocc_shellpair else torch.zeros(0, dtype=torch.long),
        "eocc_graph": torch.cat(eocc_graph, dim=0) if eocc_graph else torch.zeros(0, dtype=torch.long),
        "atom_idx": torch.cat(atom_idx, dim=0) if atom_idx else torch.zeros(0, dtype=torch.long),
        "bond_idx": torch.cat(bond_idx, dim=0) if bond_idx else torch.zeros(0, dtype=torch.long),
        "node_graph": torch.cat(node_graph, dim=0) if node_graph else torch.zeros(0, dtype=torch.long),
        "edge_graph": torch.cat(edge_graph, dim=0) if edge_graph else torch.zeros(0, dtype=torch.long),
        "root_scalars": _cat_or_empty(root_scalars, TOPO_ROOT_DIM),
        "pair_left": torch.cat(pair_left, dim=0) if pair_left else torch.zeros(0, dtype=torch.long),
        "pair_right": torch.cat(pair_right, dim=0) if pair_right else torch.zeros(0, dtype=torch.long),
        "pair_rho": _cat_or_empty(pair_rho, TOPO_PAIR_DIM),
        "pair_graph": torch.cat(pair_graph, dim=0) if pair_graph else torch.zeros(0, dtype=torch.long),
        "global_topo": (
            torch.stack(global_topo, dim=0)
            if global_topo
            else torch.zeros((0, TOPO_GLOBAL_DIM), dtype=torch.float32)
        ),
        "y": torch.as_tensor(targets, dtype=torch.float32),
        "n_graphs": torch.as_tensor(len(samples), dtype=torch.long),
    }


def to_device(batch: Mapping[str, torch.Tensor], device: Any) -> dict[str, torch.Tensor]:
    return {key: value.to(device) for key, value in batch.items()}


# ---------------------------------------------------------------------------
# scatter helpers (no message passing: only pool-into-root / pool-into-graph)
# ---------------------------------------------------------------------------


def _segment_mean(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    if values.numel() == 0 or n <= 0:
        width = values.shape[1] if values.dim() > 1 else 1
        return values.new_zeros((max(n, 0), width))
    out = values.new_zeros((n, values.shape[1]))
    out.index_add_(0, index, values)
    counts = values.new_zeros(n)
    counts.index_add_(0, index, torch.ones_like(index, dtype=values.dtype))
    return out / counts.clamp_min(1.0).unsqueeze(1)


def _segment_max(values: torch.Tensor, index: torch.Tensor, n: int) -> torch.Tensor:
    if values.numel() == 0 or n <= 0:
        width = values.shape[1] if values.dim() > 1 else 1
        return values.new_zeros((max(n, 0), width))
    out = values.new_full((n, values.shape[1]), float("-inf"))
    out = out.index_reduce_(0, index, values, "amax", include_self=True)
    return torch.where(torch.isfinite(out), out, torch.zeros_like(out))


def _binding(
    root_index: torch.Tensor,
    role: torch.Tensor,
    chemistry_indices: torch.Tensor,
    n_roots: int,
    n_classes: int,
) -> torch.Tensor:
    """Explicit one-hot binding ``sum_t role_t (x) one_hot(chem_t)``."""
    role_dim = int(role.shape[1])
    if role.numel() == 0 or n_roots <= 0:
        return role.new_zeros((max(n_roots, 0), role_dim * n_classes))
    chem = chemistry_indices.long().clamp(0, n_classes - 1)
    offsets = chem.unsqueeze(1) * role_dim + torch.arange(role_dim, device=role.device)
    flat_index = root_index.unsqueeze(1) * (role_dim * n_classes) + offsets
    out = role.new_zeros((n_roots, role_dim * n_classes))
    out.reshape(-1).index_add_(0, flat_index.reshape(-1), role.reshape(-1))
    return out


# ---------------------------------------------------------------------------
# model
# ---------------------------------------------------------------------------


def _role_dims(role_mode: str) -> tuple[int, int]:
    if role_mode == "coarse":
        return SHELL_CLASSES, SHELLPAIR_CLASSES
    if role_mode in ("dense", "sparse"):
        return SHELL_CLASSES + K_V, SHELLPAIR_CLASSES + K_E
    raise ValueError(f"unknown role_mode {role_mode!r}")


def environment_input_dim(role_mode: str) -> int:
    role_v, role_e = _role_dims(role_mode)
    return ATOM_CATEGORIES + role_v * ATOM_CATEGORIES + role_e * BOND_CATEGORIES + TOPO_ROOT_DIM


class PECModel(nn.Module):
    """Pure environment composition model (frozen contract, no MP / recurrence)."""

    def __init__(
        self,
        *,
        role_mode: str = "sparse",
        d_node: np.ndarray | None = None,
        d_edge: np.ndarray | None = None,
        env_hidden: int = ENV_HIDDEN,
        env_dim: int = ENV_DIM,
        pair_hidden: int = PAIR_HIDDEN,
        pair_dim: int = PAIR_DIM,
        reader_hidden: int = READER_HIDDEN,
        ablation: str = "true",
    ) -> None:
        super().__init__()
        role_mode = str(role_mode)
        if role_mode not in ROLE_MODES:
            raise ValueError(f"unknown role_mode {role_mode!r}")
        if ablation not in ("true", "bag", "shuffle"):
            raise ValueError(f"unknown ablation {ablation!r}")
        self.role_mode = role_mode
        self.ablation = ablation
        self.env_dim = int(env_dim)
        self.pair_dim = int(pair_dim)

        role_v, role_e = _role_dims(role_mode)
        input_dim = environment_input_dim(role_mode)
        self.environment_input_dim = int(input_dim)

        if role_mode == "sparse":
            if d_node is None or d_edge is None:
                raise ValueError("sparse mode requires fitted dictionaries")
            self.d_node = nn.Parameter(torch.as_tensor(np.asarray(d_node), dtype=torch.float32))
            self.d_edge = nn.Parameter(torch.as_tensor(np.asarray(d_edge), dtype=torch.float32))
            if tuple(self.d_node.shape) != (NODE_ROLE_DIM, K_V):
                raise ValueError("node dictionary shape mismatch")
            if tuple(self.d_edge.shape) != (EDGE_ROLE_DIM, K_E):
                raise ValueError("edge dictionary shape mismatch")
        elif role_mode == "dense":
            self.m_node = nn.Linear(NODE_ROLE_DIM, K_V, bias=False)
            self.m_edge = nn.Linear(EDGE_ROLE_DIM, K_E, bias=False)
            if d_node is not None:
                with torch.no_grad():
                    self.m_node.weight.copy_(torch.as_tensor(np.asarray(d_node).T, dtype=torch.float32))
            if d_edge is not None:
                with torch.no_grad():
                    self.m_edge.weight.copy_(torch.as_tensor(np.asarray(d_edge).T, dtype=torch.float32))

        self.environment = nn.Sequential(
            nn.Linear(input_dim, int(env_hidden)),
            nn.SiLU(),
            nn.Linear(int(env_hidden), self.env_dim),
        )
        self.pair = nn.Sequential(
            nn.Linear(3 * self.env_dim + TOPO_PAIR_DIM, int(pair_hidden)),
            nn.SiLU(),
            nn.Linear(int(pair_hidden), self.pair_dim),
        )
        reader_input = 2 * self.env_dim + 2 * self.pair_dim + TOPO_GLOBAL_DIM
        self.reader = nn.Sequential(
            nn.Linear(reader_input, int(reader_hidden)),
            nn.SiLU(),
            nn.Linear(int(reader_hidden), 1),
        )

    # -- role coordinates ---------------------------------------------------
    @staticmethod
    def _sparse_codes(dictionary: torch.Tensor, features: torch.Tensor, sparsity: int) -> torch.Tensor:
        atoms = F.normalize(dictionary, dim=0, eps=1.0e-8)
        return T.iht_codes(atoms, features, s=int(sparsity), steps=IHT_STEPS)

    def _node_coord(self, node_basis: torch.Tensor) -> torch.Tensor | None:
        if self.role_mode == "coarse":
            return None
        if self.role_mode == "dense":
            return self.m_node(node_basis)
        return self._sparse_codes(self.d_node, node_basis, S_V)

    def _edge_coord(self, edge_basis: torch.Tensor) -> torch.Tensor | None:
        if self.role_mode == "coarse":
            return None
        if self.role_mode == "dense":
            return self.m_edge(edge_basis)
        return self._sparse_codes(self.d_edge, edge_basis, S_E)

    def _occurrence_node_roles(
        self, node_basis: torch.Tensor, occ_shell: torch.Tensor
    ) -> torch.Tensor:
        shell = F.one_hot(occ_shell, num_classes=SHELL_CLASSES).to(node_basis.dtype)
        coord = self._node_coord(node_basis)
        if coord is None:
            return shell
        return torch.cat([shell, coord], dim=1)

    def _occurrence_edge_roles(
        self, edge_basis: torch.Tensor, eocc_shellpair: torch.Tensor
    ) -> torch.Tensor:
        shellpair = F.one_hot(eocc_shellpair, num_classes=SHELLPAIR_CLASSES).to(edge_basis.dtype)
        coord = self._edge_coord(edge_basis)
        if coord is None:
            return shellpair
        return torch.cat([shellpair, coord], dim=1)

    # -- environment formation (frozen before any pair information exists) --
    def encode_environments(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        atom_idx: torch.Tensor | None = None,
        bond_idx: torch.Tensor | None = None,
    ) -> torch.Tensor:
        node_chem = batch["atom_idx"] if atom_idx is None else atom_idx
        edge_chem = batch["bond_idx"] if bond_idx is None else bond_idx

        node_roles = self._occurrence_node_roles(batch["node_basis"], batch["occ_shell"])
        edge_roles = self._occurrence_edge_roles(batch["edge_basis"], batch["eocc_shellpair"])

        n_nodes = int(batch["node_graph"].numel())
        binding_v = _binding(
            batch["occ_root"],
            node_roles,
            node_chem[batch["occ_node"]],
            n_nodes,
            ATOM_CATEGORIES,
        )
        binding_e = _binding(
            batch["eocc_root"],
            edge_roles,
            edge_chem[batch["eocc_edge"]],
            n_nodes,
            BOND_CATEGORIES,
        )
        root_chem = F.one_hot(
            node_chem.clamp(0, ATOM_CATEGORIES - 1), num_classes=ATOM_CATEGORIES
        ).to(binding_v.dtype)
        features = torch.cat(
            [root_chem, binding_v, binding_e, batch["root_scalars"]], dim=1
        )
        if features.shape[1] != self.environment_input_dim:
            raise RuntimeError(
                f"environment input width {features.shape[1]} != {self.environment_input_dim}"
            )
        return self.environment(features)

    # -- read-only static composition --------------------------------------
    def compose_pairs(
        self, environments: torch.Tensor, batch: Mapping[str, torch.Tensor]
    ) -> torch.Tensor:
        e_left = environments[batch["pair_left"]]
        e_right = environments[batch["pair_right"]]
        pair_input = torch.cat(
            [e_left + e_right, torch.abs(e_left - e_right), e_left * e_right, batch["pair_rho"]],
            dim=1,
        )
        return self.pair(pair_input)

    def forward(
        self,
        batch: Mapping[str, torch.Tensor],
        *,
        shuffle_generator: torch.Generator | None = None,
    ) -> dict[str, torch.Tensor]:
        n_graphs = int(batch["n_graphs"].item())
        environments = self.encode_environments(batch)

        if self.ablation == "shuffle":
            environments = _shuffle_within_graphs(
                environments, batch["node_graph"], n_graphs, shuffle_generator
            )

        unary_mean = _segment_mean(environments, batch["node_graph"], n_graphs)
        unary_max = _segment_max(environments, batch["node_graph"], n_graphs)

        pairs = self.compose_pairs(environments, batch)
        pair_mean = _segment_mean(pairs, batch["pair_graph"], n_graphs)
        pair_max = _segment_max(pairs, batch["pair_graph"], n_graphs)
        if self.ablation == "bag":
            pair_mean = torch.zeros_like(pair_mean)
            pair_max = torch.zeros_like(pair_max)

        reader_input = torch.cat(
            [unary_mean, unary_max, pair_mean, pair_max, batch["global_topo"]], dim=1
        )
        prediction = self.reader(reader_input).squeeze(-1)
        return {
            "prediction": prediction,
            "environments": environments,
            "pairs": pairs,
        }


def _shuffle_within_graphs(
    values: torch.Tensor,
    graph_index: torch.Tensor,
    n_graphs: int,
    generator: torch.Generator | None,
) -> torch.Tensor:
    """Permute rows within each graph; keeps the multiset, breaks correspondence."""
    out = values.clone()
    for graph_id in range(n_graphs):
        positions = torch.nonzero(graph_index == graph_id, as_tuple=False).reshape(-1)
        if positions.numel() <= 1:
            continue
        order = torch.randperm(positions.numel(), generator=generator, device=values.device)
        out[positions] = values[positions[order]]
    return out


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def n_params(module: nn.Module) -> int:
    return int(sum(parameter.numel() for parameter in module.parameters()))


def role_parameters(role_mode: str) -> int:
    if role_mode == "coarse":
        return 0
    return NODE_ROLE_DIM * K_V + EDGE_ROLE_DIM * K_E


def matched_hidden(role_mode: str, target_total: int, probe: "PECModel") -> int:
    """Solve for the environment-MLP hidden width matching a total parameter count."""
    input_dim = environment_input_dim(role_mode)
    fixed = n_params(probe.pair) + n_params(probe.reader)
    numerator = int(target_total) - fixed - ENV_DIM
    denominator = input_dim + 1 + ENV_DIM
    return max(1, int(round(numerator / denominator)))


def _reference_total_params(sparse_hidden: int = ENV_HIDDEN) -> int:
    model = PECModel(
        role_mode="sparse",
        d_node=np.zeros((NODE_ROLE_DIM, K_V), dtype=np.float32),
        d_edge=np.zeros((EDGE_ROLE_DIM, K_E), dtype=np.float32),
        env_hidden=int(sparse_hidden),
    )
    return n_params(model)


def build_model(
    role_mode: str,
    *,
    d_node: np.ndarray | None = None,
    d_edge: np.ndarray | None = None,
    env_hidden: int | None = None,
    ablation: str = "true",
    seed: int = 0,
) -> PECModel:
    """Build one arm; C0's hidden width is matched to CK's total parameter count."""
    if env_hidden is None:
        if role_mode == "coarse":
            target = _reference_total_params(sparse_hidden=ENV_HIDDEN)
            probe = PECModel(role_mode="coarse", env_hidden=1)
            env_hidden = matched_hidden("coarse", target, probe)
        else:
            env_hidden = ENV_HIDDEN
    torch.manual_seed(int(seed))
    return PECModel(
        role_mode=role_mode,
        d_node=d_node,
        d_edge=d_edge,
        env_hidden=int(env_hidden),
        ablation=ablation,
    )

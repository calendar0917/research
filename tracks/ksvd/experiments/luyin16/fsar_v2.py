"""FSAR-v2: Exact Attribute Marginals + Explicit Structural Basis (ZINC).

FSAR-v1 (``fsar.py``) established that the strictly factorized backbone is
trainable, that the relational core / pooling are functional, that ``B`` is
genuinely used, and that ``SAB`` beats a matched marginal-only capacity control
``SAM`` by ~0.00327.  It also exposed two limits:

1. the ``A`` channel pooled mean/std of learned node embeddings, which loses
   the exact categorical attribute multiset ("attribute-marginal information
   starvation");
2. ``S`` was a *pure-topology but implicit* structural representation (the
   hidden state of a topology-only message-passing GNN).

FSAR-v2 answers exactly those two questions and changes nothing else:

* ``A_exact`` -- a **lossless** representation of the discrete categorical
  attribute marginals of the rooted radius-2 reference frame ``W_v``.  The raw
  vector is ``[root_one_hot, context_atom_counts, bond_counts]`` (exact integer
  counts over the *encoder schema* categories), so the multiset of categorical
  attributes that ``B`` sees is exactly recoverable from ``A_exact``.  No
  distance / degree / shell / role / adjacency-assignment information enters.
* ``S_explicit`` -- a fixed, readable rooted structural operator basis
  (root indicator, distance shell, induced degree, shell-resolved neighbour
  counts, rooted walk coordinates ``(A^k)_{vu}``, plus an analogous rooted edge
  basis).  A learned MLP may *combine* these explicit coordinates but never
  invents them from adjacency; the explicit-S path contains no message passing.
* ``B_explicit`` -- the same centred aligned interaction as ``B_implicit`` but
  the structural side is the explicit basis, so ``B`` reads
  ``attribute aligned to explicit rooted structural coordinate``.

Both phases share one implementation, one relational core, one head, one
optimizer and one schedule.  The pre-registered mode grid is:

phase 1 (implicit S):  ``A``  ``SA``  ``SAB``  ``SAM``
phase 2 (explicit S):  ``SAE`` ``SABE`` ``SAME``

Official ZINC test is never read.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import fsar as v1
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp

# ---------------------------------------------------------------------------
# frozen schema / geometry (one architecture; no sweep)
# ---------------------------------------------------------------------------

#: Encoder schema sizes.  Confirmed against the actual PyG ZINC subset: ``x``
#: is a single categorical field with observed values 0..20 (train) / 0..15
#: (valid) and ``edge_attr`` a single categorical field with observed values
#: 1..3.  The inherited backbone pads these to 28 / 4; FSAR-v2 keeps the
#: inherited schema so the channel dimension is identical to FSAR-v1.
ATOM_CATEGORIES = v1.ATOM_CATEGORIES  # 28
BOND_CATEGORIES = v1.BOND_CATEGORIES  # 4
PATCH_RADIUS = v1.PATCH_RADIUS  # 2
N_SHELLS = PATCH_RADIUS + 1  # 3
N_WALK_STEPS = 3  # fixed; A, A^2, A^3

# --- A_exact ---------------------------------------------------------------
A_RAW_DIM = ATOM_CATEGORIES + ATOM_CATEGORIES + BOND_CATEGORIES  # 60
A_DIM = 64
A_HIDDEN = 64

# --- S (implicit uses the v1 geometry; explicit is fixed) ------------------
S_DIM = 32
S_HIDDEN = 32
ROLE_DIM = v1.ROLE_DIM
ROUNDS = v1.ROUNDS

# explicit rooted node / edge operator basis
SHELL_PAIRS = ((0, 0), (0, 1), (0, 2), (1, 1), (1, 2), (2, 2))
NODE_BASIS_DIM = 1 + N_SHELLS + 1 + N_SHELLS + N_WALK_STEPS  # 11
EDGE_BASIS_DIM = len(SHELL_PAIRS) + 1 + 1 + 1 + 2 * N_SHELLS  # 15
S_EXPLICIT_INPUT_DIM = 3 * NODE_BASIS_DIM + 2 * EDGE_BASIS_DIM  # 63

# --- B ---------------------------------------------------------------------
ATTR_DIM = v1.ATTR_DIM  # 16 shared atom/bond semantic embedding width
B_DIM = v1.B_DIM  # 32
B_HIDDEN = v1.B_HIDDEN  # 32

# --- capacity control M (marginal-only) ------------------------------------
#: Hidden widths are chosen (pre-registered, no sweep) so that the *total*
#: parameter count of the marginal-only control matches the aligned organ mode
#: to < 0.1 %.  Matching the total is the right control: the extra node-init
#: capacity from the wider [A,S,M] input is itself part of the released budget.
SAM_MARGINAL_DIM = 64
SAM_HIDDEN_IMPLICIT = 58  # capacity_mlp 9,402; SAM total == SAB - 6
SAM_HIDDEN_EXPLICIT = 44  # capacity_mlp 7,148; SAME total == SABE - 20

MODE_STRUCTURE_KIND = {
    "A": None,
    "SA": "implicit",
    "SAB": "implicit",
    "SAM": "implicit",
    "SAE": "explicit",
    "SABE": "explicit",
    "SAME": "explicit",
}
MODE_INCLUDE_BINDING = {"SAB": True, "SABE": True}
MODES = ("A", "SA", "SAB", "SAM")
EXPLICIT_MODES = ("SAE", "SABE", "SAME")
ALL_MODES = MODES + EXPLICIT_MODES

#: Modes whose ``S`` channel is the implicit topology-GNN state.
IMPLICIT_MODES = MODES
#: Modes whose ``S`` channel is the explicit structural basis.
EXPLICIT_ONLY_MODES = EXPLICIT_MODES


def mode_input_dim(mode: str) -> int:
    kind = MODE_STRUCTURE_KIND[mode]
    dim = A_DIM
    if kind is not None:
        dim += S_DIM
    if MODE_INCLUDE_BINDING.get(mode, False):
        dim += B_DIM
    if mode in {"SAM", "SAME"}:
        dim += SAM_MARGINAL_DIM
    return int(dim)


# ---------------------------------------------------------------------------
# rooted explicit structural basis (chemistry-free, no message passing)
# ---------------------------------------------------------------------------


def _shell_pair_index(distance_u: int, distance_w: int) -> int:
    pair = tuple(sorted((int(distance_u), int(distance_w))))
    return int(SHELL_PAIRS.index(pair))


def _explicit_basis_for_patch(
    graph: Any, center: int, radius: int = PATCH_RADIUS
) -> tuple[list[int], dict[int, int], np.ndarray, np.ndarray, np.ndarray]:
    """Compute the fixed explicit basis for one rooted induced frame.

    Returns ``(nodes, local_index, node_basis [n, NODE_BASIS_DIM],
    edge_basis [m, EDGE_BASIS_DIM], undirected_edges)`` where ``nodes`` is the
    sorted node list (same order as the FSAR patch graph) and the edge basis is
    provided once per **undirected** induced edge in ``sorted(induced.edges())``
    order.  Only adjacency and rooted distances are read: no chemistry.
    """
    distances = zpp._ego_distances(graph, int(center), int(radius))
    nodes = sorted(int(node) for node in distances)
    index = {node: position for position, node in enumerate(nodes)}
    n_nodes = len(nodes)
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.float64)
    induced = graph.induced(set(nodes))
    edges = sorted((int(left), int(right)) for left, right in induced.edges())
    for left, right in edges:
        i, j = index[left], index[right]
        adjacency[i, j] = 1.0
        adjacency[j, i] = 1.0
    rooted_distance = np.asarray([int(distances[node]) for node in nodes], dtype=np.int64)
    degree = adjacency.sum(axis=1)
    shell_members = [np.where(rooted_distance == shell)[0] for shell in range(N_SHELLS)]
    neighbour_by_shell = np.zeros((n_nodes, N_SHELLS), dtype=np.float64)
    for shell in range(N_SHELLS):
        neighbour_by_shell[:, shell] = adjacency[:, shell_members[shell]].sum(axis=1)

    root_position = index[int(center)]
    walks: list[np.ndarray] = []
    walk_vector = np.zeros(n_nodes, dtype=np.float64)
    walk_vector[root_position] = 1.0
    for _ in range(N_WALK_STEPS):
        walk_vector = adjacency @ walk_vector
        walks.append(walk_vector.copy())

    root_indicator = np.zeros((n_nodes, 1), dtype=np.float64)
    root_indicator[root_position, 0] = 1.0
    shell_one_hot = np.stack(
        [(rooted_distance == shell).astype(np.float64) for shell in range(N_SHELLS)],
        axis=1,
    )
    node_basis = np.concatenate(
        [
            root_indicator,
            shell_one_hot,
            np.log1p(degree)[:, None],
            np.log1p(neighbour_by_shell),
            np.log1p(np.stack(walks, axis=1)),
        ],
        axis=1,
    )
    if node_basis.shape[1] != NODE_BASIS_DIM:
        raise RuntimeError(
            f"node basis width {node_basis.shape[1]} != {NODE_BASIS_DIM}"
        )

    edge_rows: list[np.ndarray] = []
    for left, right in edges:
        i, j = index[left], index[right]
        pair_one_hot = np.zeros(len(SHELL_PAIRS), dtype=np.float64)
        pair_one_hot[_shell_pair_index(rooted_distance[i], rooted_distance[j])] = 1.0
        common = float((adjacency[i] * adjacency[j]).sum())
        edge_rows.append(
            np.concatenate(
                [
                    pair_one_hot,
                    np.asarray(
                        [
                            np.log1p(degree[i] + degree[j]),
                            np.log1p(abs(degree[i] - degree[j])),
                            np.log1p(common),
                        ],
                        dtype=np.float64,
                    ),
                    np.log1p(neighbour_by_shell[i] + neighbour_by_shell[j]),
                    np.log1p(np.abs(neighbour_by_shell[i] - neighbour_by_shell[j])),
                ]
            )
        )
    edge_basis = (
        np.stack(edge_rows, axis=0) if edge_rows else np.zeros((0, EDGE_BASIS_DIM))
    )
    if edge_basis.shape[1] != EDGE_BASIS_DIM:
        raise RuntimeError(
            f"edge basis width {edge_basis.shape[1]} != {EDGE_BASIS_DIM}"
        )
    return nodes, index, node_basis.astype(np.float32), edge_basis.astype(np.float32), edges


def explicit_basis_arrays(
    data_row: Any, patch_graph: Any, radius: int = PATCH_RADIUS
) -> tuple[np.ndarray, np.ndarray]:
    """Per-molecule ``(node_basis [T, 11], edge_basis [E, 15])`` arrays.

    ``edge_basis`` repeats each undirected edge basis once per directed entry,
    exactly matching the ``struct_src`` / ``struct_dst`` / ``struct_bond``
    ordering produced by ``_patch_graphs_from_dataset``.
    """
    graph, _node_types, _edge_types = zpp._data_to_graph(data_row)
    centers = list(graph.nodes)
    if len(centers) != int(patch_graph.n_patches):
        raise RuntimeError("centre / patch count mismatch while building basis")
    node_parts: list[np.ndarray] = []
    edge_parts: list[np.ndarray] = []
    for center in centers:
        _nodes, _index, node_basis, edge_basis, edges = _explicit_basis_for_patch(
            graph, int(center), radius
        )
        node_parts.append(node_basis)
        # Mirror the directed (left,right), (right,left) expansion.
        for edge_index, (left, right) in enumerate(edges):
            edge_parts.append(edge_basis[edge_index])
            edge_parts.append(edge_basis[edge_index])
    node_array = (
        np.concatenate(node_parts, axis=0)
        if node_parts
        else np.zeros((0, NODE_BASIS_DIM), dtype=np.float32)
    )
    edge_array = (
        np.stack(edge_parts, axis=0)
        if edge_parts
        else np.zeros((0, EDGE_BASIS_DIM), dtype=np.float32)
    )
    return node_array.astype(np.float32), edge_array.astype(np.float32)


# ---------------------------------------------------------------------------
# dataset construction
# ---------------------------------------------------------------------------


def build_fsar_v2_dataset(
    dataset: Sequence[Any],
    patch_graphs: Sequence[Any],
    topology_matrix: np.ndarray,
    topology_mean: np.ndarray | None = None,
    topology_scale: np.ndarray | None = None,
) -> tuple[list[Data], dict[str, np.ndarray]]:
    """FSAR-v1 data plus the explicit rooted structural basis tensors."""
    if len(dataset) != len(patch_graphs):
        raise RuntimeError("dataset / patch graph molecule count mismatch")
    base, standardizer = v1.build_fsar_dataset(
        dataset,
        patch_graphs,
        topology_matrix,
        topology_mean=topology_mean,
        topology_scale=topology_scale,
    )
    output: list[Data] = []
    for row, data_row, patch_graph in zip(base, dataset, patch_graphs):
        node_basis, edge_basis = explicit_basis_arrays(data_row, patch_graph)
        if node_basis.shape[0] != int(patch_graph.atom.shape[0]):
            raise RuntimeError("node basis / patch node count mismatch")
        if edge_basis.shape[0] != int(patch_graph.bond.shape[0]):
            raise RuntimeError("edge basis / patch edge count mismatch")
        row.node_basis = torch.from_numpy(node_basis)
        row.edge_basis = torch.from_numpy(edge_basis)
        output.append(row)
    return output, standardizer


# ---------------------------------------------------------------------------
# raw exact marginal vector (audit / losslessness)
# ---------------------------------------------------------------------------


def raw_marginal_vector(
    struct_atom: torch.Tensor,
    struct_root: torch.Tensor,
    struct_patch: torch.Tensor,
    struct_bond: torch.Tensor,
    struct_edge_patch: torch.Tensor,
    n_patches: int,
    atom_categories: int = ATOM_CATEGORIES,
    bond_categories: int = BOND_CATEGORIES,
) -> torch.Tensor:
    """The exact raw categorical marginal ``[root, atom counts, bond counts]``.

    The vector is built from integers only, so it is *lossless* for the defined
    categorical marginal: two rooted frames collide iff their root category and
    their context atom / bond category multisets are identical.
    """
    dtype = torch.float32
    device = struct_atom.device
    root_one_hot = torch.zeros((n_patches, atom_categories), dtype=dtype, device=device)
    context_counts = torch.zeros(
        (n_patches, atom_categories), dtype=dtype, device=device
    )
    bond_counts = torch.zeros((n_patches, bond_categories), dtype=dtype, device=device)
    root_mask = struct_root.long() > 0
    if bool(root_mask.any()):
        one_hot = F.one_hot(
            struct_atom.long()[root_mask], atom_categories
        ).to(dtype)
        root_one_hot.index_add_(0, struct_patch.long()[root_mask], one_hot)
    context_mask = ~root_mask
    if bool(context_mask.any()):
        one_hot = F.one_hot(
            struct_atom.long()[context_mask], atom_categories
        ).to(dtype)
        context_counts.index_add_(
            0, struct_patch.long()[context_mask], one_hot
        )
    if struct_bond.numel():
        one_hot = F.one_hot(struct_bond.long(), bond_categories).to(dtype)
        bond_counts.index_add_(0, struct_edge_patch.long(), one_hot)
    return torch.cat([root_one_hot, context_counts, bond_counts], dim=1)


def marginal_key(
    struct_atom: np.ndarray,
    struct_root: np.ndarray,
    struct_patch: np.ndarray,
    struct_bond: np.ndarray,
    struct_edge_patch: np.ndarray,
    n_patches: int,
) -> list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]]:
    """Pure-python exact marginal key per patch (for the collision audit)."""
    atom = np.asarray(struct_atom, dtype=np.int64)
    root = np.asarray(struct_root, dtype=np.int64)
    patch = np.asarray(struct_patch, dtype=np.int64)
    bond = np.asarray(struct_bond, dtype=np.int64)
    edge_patch = np.asarray(struct_edge_patch, dtype=np.int64)
    keys: list[tuple[tuple[int, ...], tuple[int, ...], tuple[int, ...]]] = []
    for patch_id in range(int(n_patches)):
        root_mask = (patch == patch_id) & (root > 0)
        ctx_mask = (patch == patch_id) & (root == 0)
        root_category = int(atom[root_mask][0]) if bool(root_mask.any()) else -1
        ctx_counts = tuple(
            int((atom[ctx_mask] == category).sum())
            for category in range(ATOM_CATEGORIES)
        )
        bond_counts = tuple(
            int((bond[edge_patch == patch_id] == category).sum())
            for category in range(BOND_CATEGORIES)
        )
        keys.append((root_category, ctx_counts, bond_counts))
    return keys


# ---------------------------------------------------------------------------
# channel encoder
# ---------------------------------------------------------------------------


class FSARV2ChannelEncoder(nn.Module):
    """FSAR-v2 strict factorized encoder with exact-A and selectable S/B.

    ``structure_kind``:

    * ``None``        -- mode ``A`` (no S, no B);
    * ``"implicit"``  -- FSAR-v1 topology-only message-passing role state;
    * ``"explicit"``  -- fixed rooted structural operator basis (no message
      passing anywhere in the explicit-S path).

    ``include_binding`` adds ``B`` aligned to whichever structural side exists.
    """

    def __init__(
        self,
        *,
        atom_categories: int = ATOM_CATEGORIES,
        bond_categories: int = BOND_CATEGORIES,
        structure_kind: str | None = None,
        include_binding: bool = False,
        a_dim: int = A_DIM,
        a_hidden: int = A_HIDDEN,
        s_dim: int = S_DIM,
        s_hidden: int = S_HIDDEN,
        role_dim: int = ROLE_DIM,
        rounds: int = ROUNDS,
        attr_dim: int = ATTR_DIM,
        b_dim: int = B_DIM,
        b_hidden: int = B_HIDDEN,
        activation: str = v1.ACTIVATION,
    ) -> None:
        super().__init__()
        if structure_kind not in (None, "implicit", "explicit"):
            raise ValueError(f"unknown structure_kind={structure_kind!r}")
        if include_binding and structure_kind is None:
            raise ValueError("binding requires a structural side (B is S x A)")
        self.atom_categories = int(atom_categories)
        self.bond_categories = int(bond_categories)
        self.structure_kind = structure_kind
        self.include_structure = structure_kind is not None
        self.include_binding = bool(include_binding)
        self.a_dim = int(a_dim)
        self.s_dim = int(s_dim)
        self.attr_dim = int(attr_dim)
        self.b_dim = int(b_dim)

        # ---- shared categorical attribute semantics (A_exact and B) --------
        self.atom_embedding = nn.Embedding(self.atom_categories, self.attr_dim)
        self.bond_embedding = nn.Embedding(self.bond_categories, self.attr_dim)
        self.atom_mlp = nn.Sequential(
            nn.Linear(self.attr_dim, int(a_hidden)),
            v1._act(activation),
            nn.Linear(int(a_hidden), self.attr_dim),
        )
        self.bond_mlp = nn.Sequential(
            nn.Linear(self.attr_dim, int(a_hidden)),
            v1._act(activation),
            nn.Linear(int(a_hidden), self.attr_dim),
        )
        # A_exact: raw exact marginals -> A_v (fixed width, no sweep).
        self.a_exact_mlp = nn.Sequential(
            nn.Linear(A_RAW_DIM, int(a_hidden)),
            v1._act(activation),
            nn.Linear(int(a_hidden), self.a_dim),
        )

        # ---- implicit structure stream (FSAR-v1) ---------------------------
        if structure_kind == "implicit":
            self.root_embedding = nn.Embedding(2, int(role_dim))
            self.distance_embedding = nn.Embedding(N_SHELLS, int(role_dim))
            self.topology_base = nn.Parameter(torch.zeros(int(role_dim)))
            self.message = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(2 * int(role_dim), int(s_hidden)),
                    v1._act(activation),
                    nn.Linear(int(s_hidden), int(role_dim)),
                )
                for _ in range(int(rounds))
            )
            self.update = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(2 * int(role_dim), int(s_hidden)),
                    v1._act(activation),
                    nn.Linear(int(s_hidden), int(role_dim)),
                )
                for _ in range(int(rounds))
            )
            s_pool_width = 3 * int(role_dim)
            self.structure_pool = nn.Sequential(
                nn.Linear(s_pool_width, int(s_hidden)),
                v1._act(activation),
                nn.Linear(int(s_hidden), self.s_dim),
            )
            node_basis_dim = int(role_dim)
            edge_basis_dim = 2 * int(role_dim)
        elif structure_kind == "explicit":
            # No learned structural vocabulary, no adjacency message passing.
            self.root_embedding = None
            self.distance_embedding = None
            self.topology_base = None
            self.message = None
            self.update = None
            self.structure_pool = nn.Sequential(
                nn.Linear(S_EXPLICIT_INPUT_DIM, int(s_hidden)),
                v1._act(activation),
                nn.Linear(int(s_hidden), self.s_dim),
            )
            node_basis_dim = NODE_BASIS_DIM
            edge_basis_dim = EDGE_BASIS_DIM
        else:
            self.root_embedding = None
            self.distance_embedding = None
            self.topology_base = None
            self.message = None
            self.update = None
            self.structure_pool = None
            node_basis_dim = 0
            edge_basis_dim = 0

        # ---- binding stream B (centred aligned interaction) ----------------
        if self.include_binding:
            self.node_role_projection = nn.Linear(
                node_basis_dim, self.b_dim, bias=False
            )
            self.node_attribute_projection = nn.Linear(
                self.attr_dim, self.b_dim, bias=False
            )
            self.edge_role_mlp = nn.Sequential(
                nn.Linear(edge_basis_dim, int(b_hidden)),
                v1._act(activation),
                nn.Linear(int(b_hidden), self.b_dim),
            )
            self.edge_attribute_mlp = nn.Sequential(
                nn.Linear(self.attr_dim, int(b_hidden)),
                v1._act(activation),
                nn.Linear(int(b_hidden), self.b_dim),
            )
            self.edge_role_projection = nn.Linear(self.b_dim, self.b_dim, bias=False)
            self.edge_attribute_projection = nn.Linear(
                self.b_dim, self.b_dim, bias=False
            )
            self.binding_fuse = nn.Sequential(
                nn.Linear(4 * self.b_dim, int(b_hidden)),
                v1._act(activation),
                nn.Linear(int(b_hidden), self.b_dim),
            )
        else:
            self.node_role_projection = None
            self.node_attribute_projection = None
            self.edge_role_mlp = None
            self.edge_attribute_mlp = None
            self.edge_role_projection = None
            self.edge_attribute_projection = None
            self.binding_fuse = None

        self.capture_diagnostics = False
        self.last_stats: dict[str, Any] = {}
        self.intervention_disable: frozenset[str] = frozenset()

    # -- interventions -------------------------------------------------------

    def set_intervention(self, disable: Sequence[str] | None) -> None:
        if disable is None:
            self.intervention_disable = frozenset()
            return
        valid = {"A", "S", "B"}
        unknown = set(disable) - valid
        if unknown:
            raise ValueError(
                f"unknown FSAR-v2 channel intervention {sorted(unknown)}; valid {sorted(valid)}"
            )
        self.intervention_disable = frozenset(str(item) for item in disable)

    # -- geometry helpers ----------------------------------------------------

    @staticmethod
    def _patch_geometry(data: object) -> tuple[torch.Tensor, torch.Tensor, int]:
        unique_patch, node_patch = torch.unique(
            data.struct_patch.long(), return_inverse=True
        )
        return unique_patch, node_patch, int(unique_patch.numel())

    def _edge_group(
        self, data: object, unique_patch: torch.Tensor, n_patches: int
    ) -> torch.Tensor:
        edge_patch_ids = data.struct_edge_patch.long()
        if edge_patch_ids.numel():
            return torch.searchsorted(unique_patch, edge_patch_ids)
        return edge_patch_ids

    # -- implicit structure stream ------------------------------------------

    def _implicit_role(self, data: object) -> torch.Tensor:
        root = data.struct_root.long()
        distance = data.struct_dist.long()
        n_nodes = int(root.shape[0])
        role = (
            self.root_embedding(root)
            + self.distance_embedding(distance)
            + self.topology_base
        )
        src = data.struct_src.long()
        dst = data.struct_dst.long()
        for round_index in range(len(self.message)):
            if src.numel():
                messages = self.message[round_index](
                    torch.cat([role[src], role[dst]], dim=1)
                )
                aggregate = v1.scatter_mean(messages, dst, n_nodes)
            else:
                aggregate = torch.zeros_like(role)
            role = role + self.update[round_index](
                torch.cat([role, aggregate], dim=1)
            )
        return role

    def _implicit_structure(
        self,
        role: torch.Tensor,
        data: object,
        node_patch: torch.Tensor,
        n_patches: int,
    ) -> torch.Tensor:
        role_mean = v1.scatter_mean(role, node_patch, n_patches)
        role_std = v1.masked_std(role, node_patch, n_patches, role_mean)
        root_mask = data.struct_root.long() > 0
        root_state = role.new_zeros((n_patches, int(role.shape[1])))
        if bool(root_mask.any()):
            root_state.index_add_(0, node_patch[root_mask], role[root_mask])
        return self.structure_pool(torch.cat([root_state, role_mean, role_std], dim=1))

    # -- explicit structure stream ------------------------------------------

    def _explicit_structure(
        self,
        data: object,
        node_patch: torch.Tensor,
        edge_group: torch.Tensor,
        n_patches: int,
    ) -> torch.Tensor:
        node_basis = data.node_basis.float()
        edge_basis = data.edge_basis.float()
        node_mean = v1.scatter_mean(node_basis, node_patch, n_patches)
        node_std = v1.masked_std(node_basis, node_patch, n_patches, node_mean)
        root_mask = data.struct_root.long() > 0
        root_basis = node_basis.new_zeros((n_patches, int(node_basis.shape[1])))
        if bool(root_mask.any()):
            root_basis.index_add_(0, node_patch[root_mask], node_basis[root_mask])
        if edge_basis.numel():
            edge_mean = v1.scatter_mean(edge_basis, edge_group, n_patches)
            edge_std = v1.masked_std(edge_basis, edge_group, n_patches, edge_mean)
        else:
            edge_mean = node_basis.new_zeros((n_patches, int(edge_basis.shape[1])))
            edge_std = torch.zeros_like(edge_mean)
        combined = torch.cat(
            [root_basis, node_mean, node_std, edge_mean, edge_std], dim=1
        )
        if int(combined.shape[1]) != S_EXPLICIT_INPUT_DIM:
            raise RuntimeError(
                f"explicit S input width {combined.shape[1]} != {S_EXPLICIT_INPUT_DIM}"
            )
        return self.structure_pool(combined)

    # -- exact attribute stream ---------------------------------------------

    def _exact_attribute_impl(
        self,
        data: object,
        node_patch: torch.Tensor,
        edge_group: torch.Tensor,
        n_patches: int,
    ) -> torch.Tensor:
        raw = raw_marginal_vector(
            data.struct_atom,
            data.struct_root,
            data.struct_patch,
            data.struct_bond,
            data.struct_edge_patch,
            n_patches,
            self.atom_categories,
            self.bond_categories,
        )
        return self.a_exact_mlp(raw)

    # -- binding stream ------------------------------------------------------

    def _binding_vectors(
        self,
        role: torch.Tensor,
        node_basis_used: torch.Tensor | None,
        edge_basis_used: torch.Tensor | None,
        node_attribute: torch.Tensor,
        edge_attribute: torch.Tensor,
        node_patch: torch.Tensor,
        edge_group: torch.Tensor,
        n_patches: int,
    ) -> torch.Tensor:
        if node_basis_used is None:
            # implicit: role is per node; reconstruct edge role from endpoint
            # roles exactly like FSAR-v1.
            node_role_input = role
            src = None
            dst = None
        else:
            node_role_input = node_basis_used
        role_mean = v1.scatter_mean(node_role_input, node_patch, n_patches)
        attribute_mean = v1.scatter_mean(node_attribute, node_patch, n_patches)
        centered_role = node_role_input - role_mean[node_patch]
        centered_attribute = node_attribute - attribute_mean[node_patch]
        node_binding = self.node_role_projection(
            centered_role
        ) * self.node_attribute_projection(centered_attribute)
        node_mean = v1.scatter_mean(node_binding, node_patch, n_patches)
        node_std = v1.masked_std(node_binding, node_patch, n_patches, node_mean)
        blocks = [node_mean, node_std]

        if edge_attribute.numel():
            if node_basis_used is None:
                src = self._last_src
                dst = self._last_dst
                edge_role = self.edge_role_mlp(
                    torch.cat(
                        [
                            role[src] + role[dst],
                            torch.abs(role[src] - role[dst]),
                        ],
                        dim=1,
                    )
                )
            else:
                edge_role = self.edge_role_mlp(edge_basis_used)
            edge_attribute_role = self.edge_attribute_mlp(edge_attribute)
            edge_role_mean = v1.scatter_mean(edge_role, edge_group, n_patches)
            edge_attribute_mean = v1.scatter_mean(
                edge_attribute_role, edge_group, n_patches
            )
            edge_binding = self.edge_role_projection(
                edge_role - edge_role_mean[edge_group]
            ) * self.edge_attribute_projection(
                edge_attribute_role - edge_attribute_mean[edge_group]
            )
            edge_mean = v1.scatter_mean(edge_binding, edge_group, n_patches)
            edge_std = v1.masked_std(edge_binding, edge_group, n_patches, edge_mean)
            blocks.extend([edge_mean, edge_std])
        else:
            blocks.append(node_binding.new_zeros((n_patches, self.b_dim)))
            blocks.append(node_binding.new_zeros((n_patches, self.b_dim)))
        return self.binding_fuse(torch.cat(blocks, dim=1))

    # -- forward -------------------------------------------------------------

    def forward_channels(self, data: object) -> dict[str, torch.Tensor]:
        unique_patch, node_patch, n_patches = self._patch_geometry(data)
        edge_group = self._edge_group(data, unique_patch, n_patches)
        atom = data.struct_atom.long()
        node_attribute = self.atom_mlp(self.atom_embedding(atom))
        bond = data.struct_bond.long()
        edge_attribute = (
            self.bond_mlp(self.bond_embedding(bond))
            if bond.numel()
            else node_attribute.new_zeros((0, self.attr_dim))
        )

        channels: dict[str, torch.Tensor] = {
            "attributes": self._exact_attribute_impl(
                data, node_patch, edge_group, n_patches
            ),
            "raw_marginal": raw_marginal_vector(
                data.struct_atom,
                data.struct_root,
                data.struct_patch,
                data.struct_bond,
                data.struct_edge_patch,
                n_patches,
                self.atom_categories,
                self.bond_categories,
            ),
            "node_attribute": node_attribute,
            "edge_attribute": edge_attribute,
        }
        if self.include_structure:
            if self.structure_kind == "implicit":
                role = self._implicit_role(data)
                structure = self._implicit_structure(
                    role, data, node_patch, n_patches
                )
                channels["role"] = role
                channels["structure"] = structure
                if self.include_binding:
                    self._last_src = data.struct_src.long()
                    self._last_dst = data.struct_dst.long()
                    channels["binding"] = self._binding_vectors(
                        role,
                        None,
                        None,
                        node_attribute,
                        edge_attribute,
                        node_patch,
                        edge_group,
                        n_patches,
                    )
            else:
                structure = self._explicit_structure(
                    data, node_patch, edge_group, n_patches
                )
                channels["structure"] = structure
                channels["node_basis"] = data.node_basis.float()
                channels["edge_basis"] = data.edge_basis.float()
                if self.include_binding:
                    channels["binding"] = self._binding_vectors(
                        None,
                        data.node_basis.float(),
                        data.edge_basis.float(),
                        node_attribute,
                        edge_attribute,
                        node_patch,
                        edge_group,
                        n_patches,
                    )
        if self.capture_diagnostics:
            self.last_stats = self._stats_from_channels(channels, n_patches)
        return channels

    def forward(self, data: object) -> torch.Tensor:
        channels = self.forward_channels(data)
        attributes = channels["attributes"]
        disabled = self.intervention_disable
        if disabled:
            attributes = attributes.clone()
            if "A" in disabled:
                attributes = torch.zeros_like(attributes)
        blocks = [attributes]
        if self.include_structure:
            structure = channels["structure"]
            if "S" in disabled:
                structure = torch.zeros_like(structure)
            blocks.append(structure)
            if self.include_binding:
                binding = channels["binding"]
                if "B" in disabled:
                    binding = torch.zeros_like(binding)
                blocks.append(binding)
        return torch.cat(blocks, dim=1)

    # -- diagnostics ---------------------------------------------------------

    def _stats_from_channels(
        self, channels: Mapping[str, torch.Tensor], n_patches: int
    ) -> dict[str, Any]:
        def _stats(value: torch.Tensor, name: str) -> dict[str, float]:
            if value.numel() == 0:
                return {
                    f"{name}_norm_mean": 0.0,
                    f"{name}_norm_std": 0.0,
                    f"{name}_cross_mol_std": 0.0,
                }
            norms = value.detach().norm(dim=1)
            return {
                f"{name}_norm_mean": float(norms.mean()),
                f"{name}_norm_std": float(norms.std(unbiased=False)),
                f"{name}_cross_mol_std": float(
                    value.detach().std(dim=0, unbiased=False).mean()
                ),
            }

        stats: dict[str, Any] = {
            "n_patches": int(n_patches),
            "structure_kind": str(self.structure_kind),
        }
        stats.update(_stats(channels["attributes"], "A"))
        if "structure" in channels:
            stats.update(_stats(channels["structure"], "S"))
        if "binding" in channels:
            stats.update(_stats(channels["binding"], "B"))
        return stats


# ---------------------------------------------------------------------------
# full backbone
# ---------------------------------------------------------------------------


class PatchPathFSARV2Model(nn.Module):
    """FSAR-v2 backbone: exact-A init + selectable implicit/explicit S/B."""

    def __init__(
        self,
        *,
        mode: str = "SAB",
        hidden: int = v1.H_DIM,
        q_dim: int = v1.Q_DIM,
        recurrence_rounds: int = v1.T_ROUNDS,
        center_hidden: int = v1.CENTER_HIDDEN,
        relation_hidden: int = v1.RELATION_HIDDEN,
        pair_hidden: int = v1.PAIR_HIDDEN,
        head_hidden: Sequence[int] = v1.HEAD_HIDDEN,
        dropout: float = v1.DROPOUT,
        use_topology_channel: bool = True,
        a_dim: int = A_DIM,
        a_hidden: int = A_HIDDEN,
        s_dim: int = S_DIM,
        s_hidden: int = S_HIDDEN,
        role_dim: int = ROLE_DIM,
        rounds: int = ROUNDS,
        attr_dim: int = ATTR_DIM,
        b_dim: int = B_DIM,
        b_hidden: int = B_HIDDEN,
        activation: str = v1.ACTIVATION,
    ) -> None:
        super().__init__()
        mode = str(mode)
        if mode not in ALL_MODES:
            raise ValueError(f"unknown FSAR-v2 mode={mode!r}; expected {ALL_MODES}")
        self.mode = mode
        self.hidden = int(hidden)
        self.q_dim = int(q_dim)
        self.recurrence_rounds = int(recurrence_rounds)
        self.use_topology_channel = bool(use_topology_channel)
        structure_kind = MODE_STRUCTURE_KIND[mode]
        include_binding = bool(MODE_INCLUDE_BINDING.get(mode, False))
        self.encoder = FSARV2ChannelEncoder(
            structure_kind=structure_kind,
            include_binding=include_binding,
            a_dim=int(a_dim),
            a_hidden=int(a_hidden),
            s_dim=int(s_dim),
            s_hidden=int(s_hidden),
            role_dim=int(role_dim),
            rounds=int(rounds),
            attr_dim=int(attr_dim),
            b_dim=int(b_dim),
            b_hidden=int(b_hidden),
            activation=str(activation),
        )
        input_dim = mode_input_dim(mode)
        self.capacity_mlp = None
        if mode in {"SAM", "SAME"}:
            hidden_m = (
                SAM_HIDDEN_IMPLICIT if mode == "SAM" else SAM_HIDDEN_EXPLICIT
            )
            self.capacity_mlp = nn.Sequential(
                nn.Linear(A_DIM + S_DIM, int(hidden_m)),
                nn.ReLU(),
                nn.Linear(int(hidden_m), SAM_MARGINAL_DIM),
            )
        self.node_init = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Linear(128, self.hidden),
            nn.ReLU(),
        )
        self.pair_projection = nn.Linear(self.hidden, self.q_dim, bias=False)
        self.relation_encoder = nn.Sequential(
            nn.Linear(v1.RELATION_WIDTH, int(relation_hidden)),
            nn.ReLU(),
            nn.Linear(int(relation_hidden), self.q_dim),
        )
        self.distance_gate = nn.Embedding(v1.DISTANCE_BUCKETS, self.q_dim)
        self.pair_encoder = nn.Sequential(
            nn.Linear(4 * self.q_dim, int(pair_hidden)),
            nn.ReLU(),
            nn.Linear(int(pair_hidden), self.q_dim),
        )
        self.center_context_width = v1.DISTANCE_BUCKETS * (2 * self.q_dim + 1)
        self.center_update = nn.Sequential(
            nn.Linear(self.hidden + self.center_context_width, int(center_hidden)),
            nn.LayerNorm(int(center_hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(center_hidden), self.hidden),
        )
        nn.init.zeros_(self.center_update[-1].weight)
        nn.init.zeros_(self.center_update[-1].bias)
        self.topology_encoder = (
            nn.Sequential(
                nn.Linear(v1.TOPOLOGY_WIDTH, 16),
                nn.ReLU(),
                nn.Linear(16, 8),
            )
            if self.use_topology_channel
            else None
        )
        pooled_unary_width = 2 * self.hidden + 1
        self.pooled_unary_width = int(pooled_unary_width)
        self.unified_graph_width = int(
            pooled_unary_width + (8 if self.topology_encoder is not None else 0)
        )
        head_layers: list[nn.Module] = []
        widths = [self.unified_graph_width, *[int(w) for w in head_hidden]]
        for index in range(len(widths) - 1):
            head_layers.append(nn.Linear(widths[index], widths[index + 1]))
            head_layers.append(nn.ReLU())
            if dropout and index == 0:
                head_layers.append(nn.Dropout(float(dropout)))
        head_layers.append(nn.Linear(widths[-1], 1))
        self.head = nn.Sequential(*head_layers)
        self.capture_diagnostics = False
        self.last_h0: torch.Tensor | None = None
        self.last_h1: torch.Tensor | None = None
        self.last_h2: torch.Tensor | None = None
        self.last_q: torch.Tensor | None = None

    # -- pooling (identical geometry to FSAR-v1) -----------------------------

    def _pool_nodes(
        self, value: torch.Tensor, batch: torch.Tensor, n_graphs: int
    ) -> torch.Tensor:
        total = value.new_zeros((int(n_graphs), int(value.shape[1])))
        squared = value.new_zeros((int(n_graphs), int(value.shape[1])))
        total.index_add_(0, batch, value)
        squared.index_add_(0, batch, value * value)
        counts = torch.bincount(batch, minlength=int(n_graphs)).to(value.dtype)
        mean = total / counts.clamp_min(1.0).unsqueeze(1)
        variance = (
            squared / counts.clamp_min(1.0).unsqueeze(1) - mean * mean
        ).clamp_min(0.0)
        std = torch.sqrt(variance + 1.0e-8)
        return torch.cat([mean, std, torch.log1p(counts).unsqueeze(1)], dim=1)

    def _pool_pairs_to_centres(
        self,
        value: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        pair_bucket: torch.Tensor,
        n_centres: int,
    ) -> torch.Tensor:
        blocks: list[torch.Tensor] = []
        width = int(value.shape[1])
        for bucket in range(v1.DISTANCE_BUCKETS):
            mask = pair_bucket == int(bucket)
            current = value[mask]
            current_source = source[mask]
            current_target = target[mask]
            total = value.new_zeros((int(n_centres), width))
            squared = value.new_zeros((int(n_centres), width))
            counts = value.new_zeros((int(n_centres), 1))
            if current.numel():
                endpoints = torch.cat([current_source, current_target], dim=0)
                duplicated = torch.cat([current, current], dim=0)
                total.index_add_(0, endpoints, duplicated)
                squared.index_add_(0, endpoints, duplicated * duplicated)
                counts.index_add_(
                    0,
                    endpoints,
                    torch.ones(
                        (endpoints.shape[0], 1),
                        device=value.device,
                        dtype=value.dtype,
                    ),
                )
            denominator = counts.clamp_min(1.0)
            mean = total / denominator
            variance = (squared / denominator - mean * mean).clamp_min(0.0)
            std = torch.sqrt(variance + 1.0e-8)
            occupied = (counts > 0).to(value.dtype)
            blocks.append(
                torch.cat([mean, std * occupied, torch.log1p(counts)], dim=1)
            )
        return torch.cat(blocks, dim=1)

    def _pair_value(
        self,
        patch: torch.Tensor,
        source: torch.Tensor,
        target: torch.Tensor,
        relation: torch.Tensor,
        gate: torch.Tensor,
    ) -> torch.Tensor:
        left = self.pair_projection(patch[source])
        right = self.pair_projection(patch[target])
        product = left * right
        pair_input = torch.cat(
            [
                left + right,
                torch.abs(left - right),
                product * gate,
                relation,
            ],
            dim=1,
        )
        return self.pair_encoder(pair_input)

    def encode(self, data: Data) -> torch.Tensor:
        z = self.encoder(data)
        if self.capacity_mlp is not None:
            z = torch.cat([z, self.capacity_mlp(z)], dim=1)
        h = self.node_init(z)
        h0 = h
        source = data.pair_index[0].long()
        target = data.pair_index[1].long()
        relation = self.relation_encoder(data.pair_relation)
        gate = 1.0 + torch.tanh(self.distance_gate(data.pair_bucket.long()))
        q = self._pair_value(h, source, target, relation, gate)
        centre_states: list[torch.Tensor] = []
        q_final = q
        for round_index in range(self.recurrence_rounds):
            center_context = self._pool_pairs_to_centres(
                q, source, target, data.pair_bucket.long(), int(h.shape[0])
            )
            delta = self.center_update(torch.cat([h, center_context], dim=1))
            h = h + delta
            centre_states.append(h)
            if round_index + 1 < self.recurrence_rounds:
                q = self._pair_value(h, source, target, relation, gate)
                q_final = q
        n_graphs = int(data.batch.max().item()) + 1 if data.batch.numel() else 1
        unary = self._pool_nodes(h, data.batch, n_graphs)
        blocks = [unary]
        if self.topology_encoder is not None:
            topology = data.topology_features
            if topology.ndim == 1:
                topology = topology.unsqueeze(0)
            blocks.append(self.topology_encoder(topology))
        if self.capture_diagnostics or not torch.is_grad_enabled():
            self.last_h0 = h0
            self.last_h1 = centre_states[0] if centre_states else None
            self.last_h2 = centre_states[1] if len(centre_states) > 1 else None
            self.last_q = q_final
        return torch.cat(blocks, dim=1)

    def forward(self, data: Data) -> torch.Tensor:
        return self.head(self.encode(data)).view(-1)


# ---------------------------------------------------------------------------
# batch collation (extends the FSAR-v1 collate with basis tensors)
# ---------------------------------------------------------------------------


def fsar_v2_collate(data_list: Sequence[Data]) -> Any:
    batch = v1.fsar_collate(data_list)
    batch.node_basis = torch.cat(
        [row.node_basis.float() for row in data_list], dim=0
    )
    batch.edge_basis = torch.cat(
        [row.edge_basis.float() for row in data_list], dim=0
    )
    return batch


def make_fsar_v2_loader(
    graphs: Sequence[Data], batch_size: int, shuffle: bool, seed: int
):
    from torch.utils.data import DataLoader as TorchDataLoader

    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return TorchDataLoader(
        list(graphs),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        collate_fn=fsar_v2_collate,
    )


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def _n_params(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return int(sum(parameter.numel() for parameter in module.parameters()))


def parameter_breakdown(model: PatchPathFSARV2Model) -> dict[str, Any]:
    encoder = model.encoder
    breakdown = {
        "A_exact_encoder": int(
            _n_params(encoder.atom_embedding)
            + _n_params(encoder.bond_embedding)
            + _n_params(encoder.atom_mlp)
            + _n_params(encoder.bond_mlp)
            + _n_params(encoder.a_exact_mlp)
        ),
        "S_encoder": int(
            _n_params(encoder.root_embedding)
            + _n_params(encoder.distance_embedding)
            + (
                int(encoder.topology_base.numel())
                if encoder.topology_base is not None
                else 0
            )
            + _n_params(encoder.message)
            + _n_params(encoder.update)
            + _n_params(encoder.structure_pool)
        ),
        "B_encoder": int(
            _n_params(encoder.node_role_projection)
            + _n_params(encoder.node_attribute_projection)
            + _n_params(encoder.edge_role_mlp)
            + _n_params(encoder.edge_attribute_mlp)
            + _n_params(encoder.edge_role_projection)
            + _n_params(encoder.edge_attribute_projection)
            + _n_params(encoder.binding_fuse)
        ),
        "node_init": _n_params(model.node_init),
        "capacity_mlp": _n_params(model.capacity_mlp),
        "relation_core": int(
            _n_params(model.pair_projection)
            + _n_params(model.relation_encoder)
            + _n_params(model.distance_gate)
            + _n_params(model.pair_encoder)
            + _n_params(model.center_update)
        ),
        "topology_channel": _n_params(model.topology_encoder),
        "graph_head": _n_params(model.head),
    }
    breakdown["total"] = _n_params(model)
    breakdown["dataset_dependent_vocabulary_params"] = 0
    breakdown["mode"] = model.mode
    breakdown["structure_kind"] = model.encoder.structure_kind
    return breakdown

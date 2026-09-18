"""FSAR: Factorized Structure--Attribute Relational Network (ZINC).

This module is the *whole-backbone* redesign requested after the FSAB
"optional-token collapse" (branch ``exp/factorized-structure-attribute-binding-zinc``,
verdict Case E).  FSAB only replaced the 16D local patch token while every other
predictive input (``patch_cont``, ``parent_token``, the typed-pair relation,
``global_context``) stayed a mixed topology x chemistry descriptor, so the
optimizer could fit the target from those paths alone and annihilate the token.

FSAR instead removes every mixed bypass and makes the strictly factorized
channels the actual information substrate:

    raw molecular graph
      |-- chemistry semantics ................ A_v
      |-- untyped local topology ............. S_v
      +-- aligned correspondence (S x A) ..... B_v

      z_v = [A_v ; S_v ; B_v]  ->  h_v^0 = F_init(z_v)
      topology-only relational core  h -> q -> h -> q -> h   (T = 2)
      g = Pool(h^T) [+ topology-only hinge]  ->  shared head

Information access is enforced by construction, not by naming:

* ``A``: centre-atom semantics ``E_atom(x_v)`` (explicitly allowed to know it is
  the centre) plus permutation-invariant marginals over ``W_v \\ {v}`` atom
  attributes and over patch bond attributes.  ``A`` never reads the root flag
  of any *context* node, the BFS distance, the degree, the adjacency
  (``src``/``dst``) or any structural role.
* ``S``: root flag + root-relative BFS distance + **untyped** adjacency only.
  ``S`` never reads ``atom`` or ``bond``.
* ``B``: the only channel that pairs a structural role with the attribute of
  the *same* node / edge, centred inside the patch (the finite-patch analogue of
  ``P(S,A) - P(S)P(A)``).  No attention, no orthogonality / HSIC / MI loss.

The radius-2 patch is used only as a **rooted local reference frame**; it is not
a motif, a learned object or a discovered structure.

Nested modes share one implementation and differ only in what initialises
``h_v^0``:

* ``A``   : ``h^0 = F_A(A_v)``            -- chemistry-initialized relational model
* ``SA``  : ``h^0 = F_SA([A_v, S_v])``    -- adds explicit local topology
* ``SAB`` : ``h^0 = F_SAB([A_v,S_v,B_v])``-- adds explicit alignment (binding)

All modes use the same topology-only relation core, the same graph pooling, the
same head, the same optimizer and the same schedule.

Deleted relative to the current strong model (see
``notes/zinc_current_backbone_information_flow.md``): ``patch_cont`` (typed
shell histogram), ``patch_context`` (radius-3 typed histogram), typed
``e_patch`` lookup / shared structural / bag encoders, ``parent_token``,
``global_context`` (global typed histogram), direct token readout, the v6
``attribute_encoder``, the ring-context conditioning, and the mixed columns of
``pair_relation`` (path bond mean + adjacent bond type).  The pair moments are
not read out directly: pair information reaches the readout only through the
``T=2`` centre updates, so ``g`` is a function of the final node states.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo

# ---------------------------------------------------------------------------
# frozen geometry constants (one architecture; no sweep)
# ---------------------------------------------------------------------------

ATOM_CATEGORIES = 28
BOND_CATEGORIES = 4
PATCH_RADIUS = 2
N_DISTANCE_BINS = PATCH_RADIUS + 1  # 3
DISTANCE_BUCKETS = 5  # 1,2,3,4,5+
TOPOLOGY_MODE = "hinge"
TOPOLOGY_WIDTH = ztopo.raw_width(TOPOLOGY_MODE)  # 25

#: Topology-only relation descriptor.  The historical mixed relation had 23
#: columns; the two chemistry blocks (``path_bond_mean`` [4] and the
#: ``adjacent`` bond one-hot [4]) are removed.  Kept columns:
#:   distance bucket one-hot (5) + log distance (1)
#:   + patch overlap / size relation (5) + boundary overlap (3)
#:   + log shortest-path count (1)
RELATION_WIDTH = DISTANCE_BUCKETS + 1 + 5 + 3 + 1  # 15

# --- FSAR channel widths (brief sections 7-14) -----------------------------
ROLE_DIM = 32
S_DIM = 32
A_SELF_DIM = 16
A_CTX_DIM = 16
A_DIM = A_SELF_DIM + A_CTX_DIM  # 32
B_DIM = 32
ATTR_DIM = A_SELF_DIM  # shared atom/bond semantic embedding width
B_HIDDEN = 32
S_HIDDEN = 32
A_HIDDEN = 32
ROUNDS = 2
ACTIVATION = "silu"
INCLUDE_STD_POOL = True

# --- relational core / head (reuses the proven cell-A geometry) -------------
H_DIM = 64
Q_DIM = 16
T_ROUNDS = 2
CENTER_HIDDEN = 64
RELATION_HIDDEN = 64
PAIR_HIDDEN = 64
HEAD_HIDDEN = (64, 32)
DROPOUT = 0.05

MODE_INPUT_DIM = {"A": A_DIM, "SA": A_DIM + S_DIM, "SAB": A_DIM + S_DIM + B_DIM}
MODES = ("A", "SA", "SAB")
#: Conditional capacity control (brief section 25): SA plus a marginal-only MLP
#: ``M_v = F_M([A_v, S_v])`` with approximately the B encoder's parameter count.
#: ``M`` never reads an aligned per-node / per-edge pair, so it cannot encode
#: binding; it can only add capacity on the same marginals.
SAM_MARGINAL_DIM = 64
SAM_MARGINAL_HIDDEN = 72
MODE_INPUT_DIM["SAM"] = A_DIM + S_DIM + SAM_MARGINAL_DIM
ALL_MODES = MODES + ("SAM",)


def scatter_mean(
    values: torch.Tensor, index: torch.Tensor, n_groups: int
) -> torch.Tensor:
    if values.numel() == 0:
        return values.new_zeros((int(n_groups), int(values.shape[1])))
    total = values.new_zeros((int(n_groups), int(values.shape[1])))
    total.index_add_(0, index, values)
    counts = torch.bincount(index, minlength=int(n_groups)).clamp_min(1).to(values.dtype)
    return total / counts.unsqueeze(1)


def masked_std(
    values: torch.Tensor,
    index: torch.Tensor,
    n_groups: int,
    mean: torch.Tensor,
) -> torch.Tensor:
    dim = int(values.shape[1]) if values.dim() == 2 else 0
    if values.numel() == 0:
        return values.new_zeros((int(n_groups), dim))
    total = values.new_zeros((int(n_groups), dim))
    total.index_add_(0, index, values * values)
    counts = torch.bincount(index, minlength=int(n_groups)).to(values.dtype)
    safe = counts.clamp_min(1.0).unsqueeze(1)
    variance = (total / safe - mean * mean).clamp_min(0.0)
    std = torch.sqrt(variance + 1.0e-8)
    return std * (counts > 0).to(values.dtype).unsqueeze(1)


def _act(name: str) -> nn.Module:
    if name == "silu":
        return nn.SiLU()
    if name == "gelu":
        return nn.GELU()
    if name == "relu":
        return nn.ReLU()
    raise ValueError(f"unknown activation {name!r}")


class FSARChannelEncoder(nn.Module):
    """Strict A / S / B patch encoder over one rooted local reference frame.

    ``include_structure`` / ``include_binding`` select which channels exist, so
    the three nested modes share exactly one implementation and one downstream:

    * mode ``A``   : ``include_structure=False, include_binding=False``
    * mode ``SA``  : ``include_structure=True,  include_binding=False``
    * mode ``SAB`` : ``include_structure=True,  include_binding=True``

    ``A`` always exists.  When ``S`` is absent its role tensor is not computed
    and ``B`` cannot exist (``B`` is *defined* as a structure x attribute
    alignment).

    Input contract (attached by :func:`build_fsar_dataset`):

    ``struct_atom``       int64 ``[T]``  atom category per patch node
    ``struct_root``       int64 ``[T]``  1 for the patch root, else 0
    ``struct_dist``       int64 ``[T]``  BFS distance from the patch root
    ``struct_patch``      int64 ``[T]``  patch index owning each node
    ``struct_src``        int64 ``[E]``  directed edge source (batch-local)
    ``struct_dst``        int64 ``[E]``  directed edge target (batch-local)
    ``struct_bond``       int64 ``[E]``  bond category (attribute channel only)
    ``struct_edge_patch`` int64 ``[E]``  bond -> patch grouping index
    """

    def __init__(
        self,
        *,
        atom_categories: int = ATOM_CATEGORIES,
        bond_categories: int = BOND_CATEGORIES,
        n_distance_bins: int = N_DISTANCE_BINS,
        role_dim: int = ROLE_DIM,
        s_dim: int = S_DIM,
        a_self_dim: int = A_SELF_DIM,
        a_ctx_dim: int = A_CTX_DIM,
        b_dim: int = B_DIM,
        attr_dim: int = ATTR_DIM,
        s_hidden: int = S_HIDDEN,
        a_hidden: int = A_HIDDEN,
        b_hidden: int = B_HIDDEN,
        rounds: int = ROUNDS,
        activation: str = ACTIVATION,
        include_structure: bool = True,
        include_binding: bool = True,
        include_std_pool: bool = INCLUDE_STD_POOL,
    ) -> None:
        super().__init__()
        if int(a_self_dim) != int(attr_dim):
            raise ValueError(
                "A_self is the raw atom embedding; a_self_dim must equal attr_dim"
            )
        if bool(include_binding) and not bool(include_structure):
            raise ValueError("binding requires the structure channel (B is S x A)")
        self.atom_categories = int(atom_categories)
        self.bond_categories = int(bond_categories)
        self.n_distance_bins = int(n_distance_bins)
        self.role_dim = int(role_dim)
        self.s_dim = int(s_dim)
        self.a_self_dim = int(a_self_dim)
        self.a_ctx_dim = int(a_ctx_dim)
        self.a_dim = int(a_self_dim) + int(a_ctx_dim)
        self.b_dim = int(b_dim)
        self.attr_dim = int(attr_dim)
        self.s_hidden = int(s_hidden)
        self.a_hidden = int(a_hidden)
        self.b_hidden = int(b_hidden)
        self.rounds = int(rounds)
        self.activation_name = str(activation)
        self.include_structure = bool(include_structure)
        self.include_binding = bool(include_binding)
        self.include_std_pool = bool(include_std_pool)

        # ---- attribute semantics shared by A and B -------------------------
        self.atom_embedding = nn.Embedding(self.atom_categories, self.attr_dim)
        self.bond_embedding = nn.Embedding(self.bond_categories, self.attr_dim)
        self.atom_mlp = nn.Sequential(
            nn.Linear(self.attr_dim, self.a_hidden),
            _act(self.activation_name),
            nn.Linear(self.a_hidden, self.attr_dim),
        )
        self.bond_mlp = nn.Sequential(
            nn.Linear(self.attr_dim, self.a_hidden),
            _act(self.activation_name),
            nn.Linear(self.a_hidden, self.attr_dim),
        )
        self.attribute_fuse = nn.Sequential(
            nn.Linear(4 * self.attr_dim, self.a_hidden),
            _act(self.activation_name),
            nn.Linear(self.a_hidden, self.a_ctx_dim),
        )

        # ---- structure stream S (topology only) ----------------------------
        if self.include_structure:
            self.root_embedding = nn.Embedding(2, self.role_dim)
            self.distance_embedding = nn.Embedding(
                self.n_distance_bins, self.role_dim
            )
            self.topology_base = nn.Parameter(torch.zeros(self.role_dim))
            self.message = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(2 * self.role_dim, self.s_hidden),
                    _act(self.activation_name),
                    nn.Linear(self.s_hidden, self.role_dim),
                )
                for _ in range(self.rounds)
            )
            self.update = nn.ModuleList(
                nn.Sequential(
                    nn.Linear(2 * self.role_dim, self.s_hidden),
                    _act(self.activation_name),
                    nn.Linear(self.s_hidden, self.role_dim),
                )
                for _ in range(self.rounds)
            )
            s_pool_width = (
                3 * self.role_dim if self.include_std_pool else 2 * self.role_dim
            )
            self.structure_pool = nn.Sequential(
                nn.Linear(s_pool_width, self.s_hidden),
                _act(self.activation_name),
                nn.Linear(self.s_hidden, self.s_dim),
            )
        else:
            self.root_embedding = None
            self.distance_embedding = None
            self.topology_base = None
            self.message = None
            self.update = None
            self.structure_pool = None

        # ---- binding stream B (centered low-rank interaction) --------------
        if self.include_binding:
            self.node_role_projection = nn.Linear(
                self.role_dim, self.b_dim, bias=False
            )
            self.node_attribute_projection = nn.Linear(
                self.attr_dim, self.b_dim, bias=False
            )
            self.edge_role_mlp = nn.Sequential(
                nn.Linear(2 * self.role_dim, self.b_hidden),
                _act(self.activation_name),
                nn.Linear(self.b_hidden, self.b_dim),
            )
            self.edge_attribute_mlp = nn.Sequential(
                nn.Linear(self.attr_dim, self.b_hidden),
                _act(self.activation_name),
                nn.Linear(self.b_hidden, self.b_dim),
            )
            self.edge_role_projection = nn.Linear(
                self.b_dim, self.b_dim, bias=False
            )
            self.edge_attribute_projection = nn.Linear(
                self.b_dim, self.b_dim, bias=False
            )
            b_fuse_width = (
                4 * self.b_dim if self.include_std_pool else 2 * self.b_dim
            )
            self.binding_fuse = nn.Sequential(
                nn.Linear(b_fuse_width, self.b_hidden),
                _act(self.activation_name),
                nn.Linear(self.b_hidden, self.b_dim),
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
        # Evaluation-only channel interventions (never trained through).
        self.intervention_disable: frozenset[str] = frozenset()

    def set_intervention(self, disable: Sequence[str] | None) -> None:
        """Set evaluation-only channel interventions.

        Supported keys: ``A``, ``A_self``, ``A_ctx``, ``S``, ``B``.
        """
        if disable is None:
            self.intervention_disable = frozenset()
            return
        valid = {"A", "A_self", "A_ctx", "S", "B"}
        unknown = set(disable) - valid
        if unknown:
            raise ValueError(
                f"unknown FSAR channel intervention {sorted(unknown)}; valid {sorted(valid)}"
            )
        self.intervention_disable = frozenset(str(item) for item in disable)

    # -- helpers -------------------------------------------------------------

    @staticmethod
    def _patch_geometry(
        data: object,
    ) -> tuple[torch.Tensor, torch.Tensor, int]:
        unique_patch, node_patch = torch.unique(
            data.struct_patch.long(), return_inverse=True
        )
        return unique_patch, node_patch, int(unique_patch.numel())

    # -- structure stream ----------------------------------------------------

    def _structure_role(self, data: object) -> torch.Tensor:
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
        for round_index in range(self.rounds):
            if src.numel():
                messages = self.message[round_index](
                    torch.cat([role[src], role[dst]], dim=1)
                )
                aggregate = scatter_mean(messages, dst, n_nodes)
            else:
                aggregate = torch.zeros_like(role)
            role = role + self.update[round_index](
                torch.cat([role, aggregate], dim=1)
            )
        return role

    def _structure_vector(
        self, role: torch.Tensor, data: object, node_patch: torch.Tensor, n_patches: int
    ) -> torch.Tensor:
        role_mean = scatter_mean(role, node_patch, n_patches)
        blocks = [role_mean]
        if self.include_std_pool:
            blocks.append(masked_std(role, node_patch, n_patches, role_mean))
        root_mask = data.struct_root.long() > 0
        root_state = role.new_zeros((n_patches, self.role_dim))
        if bool(root_mask.any()):
            root_state.index_add_(0, node_patch[root_mask], role[root_mask])
        blocks.insert(0, root_state)
        return self.structure_pool(torch.cat(blocks, dim=1))

    # -- attribute stream ----------------------------------------------------

    def _attribute_vector(
        self,
        node_attribute: torch.Tensor,
        edge_attribute: torch.Tensor,
        data: object,
        node_patch: torch.Tensor,
        n_patches: int,
        unique_patch: torch.Tensor,
    ) -> torch.Tensor:
        root_mask = data.struct_root.long() > 0

        # A_self: the centre atom's own semantics (explicitly rooted).
        self_state = node_attribute.new_zeros((n_patches, self.attr_dim))
        if bool(root_mask.any()):
            self_state.index_add_(0, node_patch[root_mask], node_attribute[root_mask])

        # A_ctx: permutation-invariant marginals over W_v \ {v} atoms and all
        # patch bonds.  No distance / degree / role / adjacency is read.
        context_mask = ~root_mask
        if bool(context_mask.any()):
            context_patch = node_patch[context_mask]
            context_attribute = node_attribute[context_mask]
            atom_mean = scatter_mean(context_attribute, context_patch, n_patches)
            atom_std = masked_std(
                context_attribute, context_patch, n_patches, atom_mean
            )
        else:
            atom_mean = node_attribute.new_zeros((n_patches, self.attr_dim))
            atom_std = node_attribute.new_zeros((n_patches, self.attr_dim))

        edge_patch_ids = data.struct_edge_patch.long()
        edge_group = (
            torch.searchsorted(unique_patch, edge_patch_ids)
            if edge_patch_ids.numel()
            else edge_patch_ids
        )
        if edge_attribute.numel():
            bond_mean = scatter_mean(edge_attribute, edge_group, n_patches)
            bond_std = masked_std(edge_attribute, edge_group, n_patches, bond_mean)
        else:
            bond_mean = node_attribute.new_zeros((n_patches, self.attr_dim))
            bond_std = node_attribute.new_zeros((n_patches, self.attr_dim))

        context = self.attribute_fuse(
            torch.cat([atom_mean, atom_std, bond_mean, bond_std], dim=1)
        )
        return torch.cat([self_state, context], dim=1)

    # -- binding stream ------------------------------------------------------

    def _binding_vector(
        self,
        role: torch.Tensor,
        node_attribute: torch.Tensor,
        edge_attribute: torch.Tensor,
        node_patch: torch.Tensor,
        edge_group: torch.Tensor,
        n_patches: int,
        data: object,
    ) -> torch.Tensor:
        role_mean = scatter_mean(role, node_patch, n_patches)
        attribute_mean = scatter_mean(node_attribute, node_patch, n_patches)
        centered_role = role - role_mean[node_patch]
        centered_attribute = node_attribute - attribute_mean[node_patch]
        node_binding = self.node_role_projection(
            centered_role
        ) * self.node_attribute_projection(centered_attribute)
        node_mean = scatter_mean(node_binding, node_patch, n_patches)
        blocks = [node_mean]
        if self.include_std_pool:
            blocks.append(masked_std(node_binding, node_patch, n_patches, node_mean))

        src = data.struct_src.long()
        dst = data.struct_dst.long()
        if edge_attribute.numel():
            edge_role = self.edge_role_mlp(
                torch.cat(
                    [role[src] + role[dst], torch.abs(role[src] - role[dst])],
                    dim=1,
                )
            )
            edge_attribute_role = self.edge_attribute_mlp(edge_attribute)
            edge_role_mean = scatter_mean(edge_role, edge_group, n_patches)
            edge_attribute_mean = scatter_mean(
                edge_attribute_role, edge_group, n_patches
            )
            edge_binding = self.edge_role_projection(
                edge_role - edge_role_mean[edge_group]
            ) * self.edge_attribute_projection(
                edge_attribute_role - edge_attribute_mean[edge_group]
            )
            edge_mean = scatter_mean(edge_binding, edge_group, n_patches)
            blocks.append(edge_mean)
            if self.include_std_pool:
                blocks.append(
                    masked_std(edge_binding, edge_group, n_patches, edge_mean)
                )
        else:
            edge_mean = node_binding.new_zeros((n_patches, self.b_dim))
            if self.include_std_pool:
                blocks.append(torch.zeros_like(edge_mean))
        return self.binding_fuse(torch.cat(blocks, dim=1))

    # -- forward -------------------------------------------------------------

    def forward_channels(self, data: object) -> dict[str, torch.Tensor]:
        """Return the factorized channel values (audit / diagnostic use)."""
        unique_patch, node_patch, n_patches = self._patch_geometry(data)
        atom = data.struct_atom.long()
        node_attribute = self.atom_mlp(self.atom_embedding(atom))

        bond = data.struct_bond.long()
        edge_patch_ids = data.struct_edge_patch.long()
        edge_group = (
            torch.searchsorted(unique_patch, edge_patch_ids)
            if edge_patch_ids.numel()
            else edge_patch_ids
        )
        edge_attribute = (
            self.bond_mlp(self.bond_embedding(bond))
            if bond.numel()
            else node_attribute.new_zeros((0, self.attr_dim))
        )

        attributes = self._attribute_vector(
            node_attribute, edge_attribute, data, node_patch, n_patches, unique_patch
        )
        channels: dict[str, torch.Tensor] = {
            "attributes": attributes,
            "node_attribute": node_attribute,
            "edge_attribute": edge_attribute,
        }
        if self.include_structure:
            role = self._structure_role(data)
            structure = self._structure_vector(role, data, node_patch, n_patches)
            channels["role"] = role
            channels["structure"] = structure
            if self.include_binding:
                channels["binding"] = self._binding_vector(
                    role,
                    node_attribute,
                    edge_attribute,
                    node_patch,
                    edge_group,
                    n_patches,
                    data,
                )
        if self.capture_diagnostics:
            self.last_stats = self._stats_from_channels(channels, n_patches)
        return channels

    def forward(self, data: object) -> torch.Tensor:
        """Return ``z_v = [A_v ; S_v ; B_v]`` (channels present in this mode)."""
        channels = self.forward_channels(data)
        attributes = channels["attributes"]
        disabled = self.intervention_disable
        if disabled:
            attributes = attributes.clone()
            if "A" in disabled:
                attributes = torch.zeros_like(attributes)
            else:
                if "A_self" in disabled:
                    attributes[:, : self.a_self_dim] = 0.0
                if "A_ctx" in disabled:
                    attributes[:, self.a_self_dim :] = 0.0
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

        stats: dict[str, Any] = {"n_patches": int(n_patches)}
        stats.update(_stats(channels["attributes"], "A"))
        if "structure" in channels:
            stats.update(_stats(channels["structure"], "S"))
        if "binding" in channels:
            stats.update(_stats(channels["binding"], "B"))
        return stats


# ---------------------------------------------------------------------------
# topology-only pair relation / feature extraction
# ---------------------------------------------------------------------------


def _untyped_bfs(graph: Any, source: int) -> tuple[dict[int, int], dict[int, int]]:
    """Untyped BFS from ``source``: ``(distance, shortest_path_count)`` maps.

    Reads only the physical adjacency; no atom / bond attribute is touched.
    """
    from collections import deque

    source = int(source)
    distances = {source: 0}
    counts = {source: 1}
    queue: deque[int] = deque([source])
    while queue:
        node = queue.popleft()
        for neighbor in sorted(graph.neighbors(node)):
            neighbor = int(neighbor)
            if neighbor not in distances:
                distances[neighbor] = distances[node] + 1
                counts[neighbor] = 0
                queue.append(neighbor)
            if distances[neighbor] == distances[node] + 1:
                counts[neighbor] += counts[node]
    return distances, counts


def topology_pair_relation(
    graph: Any,
    centers: Sequence[int],
    patches: Sequence[Any],
    patch_radius: int = PATCH_RADIUS,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Build the topology-only relation for every unordered centre pair.

    ``centers`` are the graph node ids of the patch roots and ``patches`` the
    per-centre node sets (``W_v``).  Only graph structure is read: no atom or
    bond attribute is touched, which is what makes the relation ``S``-pure.

    Returns ``(pair_index [2, P], pair_relation [P, 15], pair_bucket [P])``.
    """
    n_patches = len(patches)
    # One untyped BFS per centre (not per pair): the full distance / path-count
    # maps are reused for every right centre.
    bfs = [_untyped_bfs(graph, int(center)) for center in centers]
    sources: list[int] = []
    targets: list[int] = []
    relations: list[np.ndarray] = []
    buckets: list[int] = []
    for left_index in range(n_patches):
        left_nodes = patches[left_index]
        left_boundary = left_nodes["boundary"]
        left_size = len(left_nodes["nodes"])
        left_distance, left_count = bfs[left_index]
        for right_index in range(left_index + 1, n_patches):
            right_nodes = patches[right_index]
            right_boundary = right_nodes["boundary"]
            right_size = len(right_nodes["nodes"])
            right_center = int(centers[right_index])
            distance = int(left_distance.get(right_center, 0))
            path_count = int(left_count.get(right_center, 0))
            bucket = min(max(int(distance), 1), DISTANCE_BUCKETS) - 1
            distance_one_hot = np.zeros(DISTANCE_BUCKETS, dtype=np.float32)
            distance_one_hot[bucket] = 1.0

            intersection = len(left_nodes["nodes"] & right_nodes["nodes"])
            union = len(left_nodes["nodes"] | right_nodes["nodes"])
            overlap = np.asarray(
                [
                    float(intersection)
                    / max(float(int(patch_radius) * int(patch_radius) + 10), 1.0),
                    float(intersection) / max(float(union), 1.0),
                    float(intersection)
                    / max(float(min(left_size, right_size)), 1.0),
                    float(intersection)
                    / max(float(max(left_size, right_size)), 1.0),
                    float(abs(left_size - right_size))
                    / max(float(int(patch_radius) * int(patch_radius) + 10), 1.0),
                ],
                dtype=np.float32,
            )
            boundary_intersection = len(left_boundary & right_boundary)
            boundary_union = len(left_boundary | right_boundary)
            boundary_min = min(len(left_boundary), len(right_boundary))
            boundary_features = np.asarray(
                [
                    float(boundary_intersection) / max(float(boundary_union), 1.0),
                    float(boundary_intersection) / max(float(boundary_min), 1.0),
                    float(boundary_intersection > 0),
                ],
                dtype=np.float32,
            )
            relation = np.concatenate(
                [
                    distance_one_hot,
                    np.asarray([np.log1p(float(distance))], dtype=np.float32),
                    overlap,
                    boundary_features,
                    np.asarray([np.log1p(float(path_count))], dtype=np.float32),
                ]
            ).astype(np.float32, copy=False)
            if relation.shape != (RELATION_WIDTH,):
                raise RuntimeError(
                    f"FSAR relation width changed: {relation.shape}; "
                    f"expected {(RELATION_WIDTH,)}"
                )
            sources.append(int(left_index))
            targets.append(int(right_index))
            relations.append(relation)
            buckets.append(int(bucket))
    pair_index = np.asarray([sources, targets], dtype=np.int64)
    if relations:
        pair_relation = np.stack(relations, axis=0).astype(np.float32, copy=False)
    else:
        pair_relation = np.zeros((0, RELATION_WIDTH), dtype=np.float32)
    return pair_index, pair_relation, np.asarray(buckets, dtype=np.int64)


def _patch_node_sets(
    graph: Any, center: int, radius: int = PATCH_RADIUS
) -> dict[str, Any]:
    """Return ``{nodes, boundary}`` for one rooted reference frame."""
    distances = zpp._ego_distances(graph, int(center), int(radius))
    nodes = frozenset(int(node) for node in distances)
    boundary = frozenset(
        int(node) for node, distance in distances.items() if int(distance) == radius
    )
    return {"nodes": nodes, "boundary": boundary}


def build_fsar_dataset(
    dataset: Sequence[Any],
    patch_graphs: Sequence[Any],
    topology_matrix: np.ndarray,
    topology_mean: np.ndarray | None = None,
    topology_scale: np.ndarray | None = None,
) -> tuple[list[Data], dict[str, np.ndarray]]:
    """Build FSAR ``Data`` objects for one split.

    ``patch_graphs`` are the ``PatchGraph`` typed tensors from
    ``zinc_shared_structural_patch_encoder._patch_graphs_from_dataset``;
    ``dataset`` supplies the untyped graph used for the topology-only relation.
    """
    if len(dataset) != len(patch_graphs):
        raise RuntimeError("dataset / patch graph molecule count mismatch")
    if topology_mean is None:
        topology_mean = topology_matrix.mean(axis=0, dtype=np.float64).astype(
            np.float32
        )
    if topology_scale is None:
        topology_scale = topology_matrix.std(axis=0, dtype=np.float64).astype(
            np.float32
        )
        topology_scale[~np.isfinite(topology_scale) | (topology_scale < 1.0e-6)] = 1.0
    output: list[Data] = []
    for data_row, patch_graph in zip(dataset, patch_graphs):
        graph, _node_types, _edge_types = zpp._data_to_graph(data_row)
        centers = list(graph.nodes)
        if len(centers) != int(patch_graph.n_patches):
            raise RuntimeError("centre / patch count mismatch")
        patch_sets = [_patch_node_sets(graph, int(center)) for center in centers]
        pair_index, pair_relation, pair_bucket = topology_pair_relation(
            graph, centers, patch_sets
        )
        node_counts = np.bincount(
            patch_graph.patch, minlength=int(patch_graph.n_patches)
        ).astype(np.int64)
        node_start = np.concatenate([[0], np.cumsum(node_counts)[:-1]]).astype(
            np.int64
        )
        topology = (
            (topology_matrix[len(output)][None, :] - topology_mean) / topology_scale
        ).astype(np.float32)
        output.append(
            Data(
                struct_atom=torch.from_numpy(np.asarray(patch_graph.atom, dtype=np.int64)),
                struct_root=torch.from_numpy(np.asarray(patch_graph.root, dtype=np.int64)),
                struct_dist=torch.from_numpy(np.asarray(patch_graph.dist, dtype=np.int64)),
                struct_patch=torch.from_numpy(np.asarray(patch_graph.patch, dtype=np.int64)),
                struct_src=torch.from_numpy(
                    np.asarray(patch_graph.src, dtype=np.int64)
                    + node_start[np.asarray(patch_graph.edge_patch, dtype=np.int64)]
                ),
                struct_dst=torch.from_numpy(
                    np.asarray(patch_graph.dst, dtype=np.int64)
                    + node_start[np.asarray(patch_graph.edge_patch, dtype=np.int64)]
                ),
                struct_bond=torch.from_numpy(np.asarray(patch_graph.bond, dtype=np.int64)),
                struct_edge_patch=torch.from_numpy(
                    np.asarray(patch_graph.edge_patch, dtype=np.int64)
                ),
                pair_index=torch.from_numpy(pair_index),
                pair_relation=torch.from_numpy(pair_relation),
                pair_bucket=torch.from_numpy(pair_bucket),
                topology_features=torch.from_numpy(topology),
                y=torch.tensor([float(data_row.y.view(-1)[0])], dtype=torch.float32),
                num_nodes=int(patch_graph.n_patches),
            )
        )
    standardizer = {"mean": topology_mean, "scale": topology_scale}
    return output, standardizer


# ---------------------------------------------------------------------------
# full FSAR backbone
# ---------------------------------------------------------------------------


class PatchPathFSARModel(nn.Module):
    """FSAR backbone: A/S/B node init + topology-only T=2 pair--centre core."""

    def __init__(
        self,
        *,
        mode: str = "SAB",
        hidden: int = H_DIM,
        q_dim: int = Q_DIM,
        recurrence_rounds: int = T_ROUNDS,
        center_hidden: int = CENTER_HIDDEN,
        relation_hidden: int = RELATION_HIDDEN,
        pair_hidden: int = PAIR_HIDDEN,
        head_hidden: Sequence[int] = HEAD_HIDDEN,
        dropout: float = DROPOUT,
        use_topology_channel: bool = True,
        role_dim: int = ROLE_DIM,
        s_dim: int = S_DIM,
        a_self_dim: int = A_SELF_DIM,
        a_ctx_dim: int = A_CTX_DIM,
        b_dim: int = B_DIM,
        attr_dim: int = ATTR_DIM,
        s_hidden: int = S_HIDDEN,
        a_hidden: int = A_HIDDEN,
        b_hidden: int = B_HIDDEN,
        rounds: int = ROUNDS,
        activation: str = ACTIVATION,
    ) -> None:
        super().__init__()
        mode = str(mode)
        if mode not in ALL_MODES:
            raise ValueError(f"unknown FSAR mode={mode!r}; expected one of {ALL_MODES}")
        self.mode = mode
        self.hidden = int(hidden)
        self.q_dim = int(q_dim)
        self.recurrence_rounds = int(recurrence_rounds)
        self.use_topology_channel = bool(use_topology_channel)
        include_structure = mode in {"SA", "SAB", "SAM"}
        include_binding = mode == "SAB"
        self.encoder = FSARChannelEncoder(
            role_dim=int(role_dim),
            s_dim=int(s_dim),
            a_self_dim=int(a_self_dim),
            a_ctx_dim=int(a_ctx_dim),
            b_dim=int(b_dim),
            attr_dim=int(attr_dim),
            s_hidden=int(s_hidden),
            a_hidden=int(a_hidden),
            b_hidden=int(b_hidden),
            rounds=int(rounds),
            activation=str(activation),
            include_structure=include_structure,
            include_binding=include_binding,
        )
        input_dim = MODE_INPUT_DIM[mode]
        self.capacity_mlp = None
        if mode == "SAM":
            # Marginal-only capacity control: reads [A, S] and nothing else.
            self.capacity_mlp = nn.Sequential(
                nn.Linear(A_DIM + S_DIM, SAM_MARGINAL_HIDDEN),
                _act("relu"),
                nn.Linear(SAM_MARGINAL_HIDDEN, SAM_MARGINAL_DIM),
            )
        self.node_init = nn.Sequential(
            nn.Linear(input_dim, 128),
            _act("relu"),
            nn.Linear(128, self.hidden),
            nn.ReLU(),
        )
        self.pair_projection = nn.Linear(self.hidden, self.q_dim, bias=False)
        self.relation_encoder = nn.Sequential(
            nn.Linear(RELATION_WIDTH, int(relation_hidden)),
            nn.ReLU(),
            nn.Linear(int(relation_hidden), self.q_dim),
        )
        self.distance_gate = nn.Embedding(DISTANCE_BUCKETS, self.q_dim)
        self.pair_encoder = nn.Sequential(
            nn.Linear(4 * self.q_dim, int(pair_hidden)),
            nn.ReLU(),
            nn.Linear(int(pair_hidden), self.q_dim),
        )
        self.center_context_width = DISTANCE_BUCKETS * (2 * self.q_dim + 1)
        self.center_update = nn.Sequential(
            nn.Linear(self.hidden + self.center_context_width, int(center_hidden)),
            nn.LayerNorm(int(center_hidden)),
            nn.ReLU(),
            nn.Dropout(float(dropout)),
            nn.Linear(int(center_hidden), self.hidden),
        )
        # Mirror the proven cell-A center update: zero-init the final layer so
        # the recurrence starts as identity and the node init carries step-0.
        nn.init.zeros_(self.center_update[-1].weight)
        nn.init.zeros_(self.center_update[-1].bias)
        self.topology_encoder = (
            nn.Sequential(
                nn.Linear(TOPOLOGY_WIDTH, 16),
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

    # -- pooling helpers -----------------------------------------------------

    def _pool_nodes(
        self, value: torch.Tensor, batch: torch.Tensor, n_graphs: int
    ) -> torch.Tensor:
        total = value.new_zeros((int(n_graphs), int(value.shape[1])))
        squared = value.new_zeros((int(n_graphs), int(value.shape[1])))
        total.index_add_(0, batch, value)
        squared.index_add_(0, batch, value * value)
        counts = torch.bincount(batch, minlength=int(n_graphs)).to(value.dtype)
        mean = total / counts.clamp_min(1.0).unsqueeze(1)
        variance = (squared / counts.clamp_min(1.0).unsqueeze(1) - mean * mean).clamp_min(0.0)
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
        for bucket in range(DISTANCE_BUCKETS):
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
                        (endpoints.shape[0], 1), device=value.device, dtype=value.dtype
                    ),
                )
            denominator = counts.clamp_min(1.0)
            mean = total / denominator
            variance = (squared / denominator - mean * mean).clamp_min(0.0)
            std = torch.sqrt(variance + 1.0e-8)
            occupied = (counts > 0).to(value.dtype)
            blocks.append(torch.cat([mean, std * occupied, torch.log1p(counts)], dim=1))
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

    # -- forward -------------------------------------------------------------

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
# batching
# ---------------------------------------------------------------------------


def fsar_collate(data_list: Sequence[Data]) -> Any:
    """Batch FSAR ``Data`` objects with explicit structural offsets."""
    from torch_geometric.data import Batch

    batch = Batch.from_data_list(list(data_list))
    node_offset = 0
    patch_offset = 0
    patch_parts: list[torch.Tensor] = []
    edge_patch_parts: list[torch.Tensor] = []
    src_parts: list[torch.Tensor] = []
    dst_parts: list[torch.Tensor] = []
    pair_parts: list[torch.Tensor] = []
    for data in data_list:
        n_patches = int(data.num_nodes)
        patch_parts.append(data.struct_patch + patch_offset)
        edge_patch_parts.append(data.struct_edge_patch + patch_offset)
        src_parts.append(data.struct_src + node_offset)
        dst_parts.append(data.struct_dst + node_offset)
        pair_parts.append(data.pair_index + patch_offset)
        node_offset += int(data.struct_atom.numel())
        patch_offset += n_patches
    batch.struct_patch = (
        torch.cat(patch_parts) if patch_parts else torch.zeros(0, dtype=torch.long)
    )
    batch.struct_edge_patch = (
        torch.cat(edge_patch_parts)
        if edge_patch_parts
        else torch.zeros(0, dtype=torch.long)
    )
    batch.struct_src = (
        torch.cat(src_parts) if src_parts else torch.zeros(0, dtype=torch.long)
    )
    batch.struct_dst = (
        torch.cat(dst_parts) if dst_parts else torch.zeros(0, dtype=torch.long)
    )
    batch.pair_index = (
        torch.cat(pair_parts, dim=1)
        if pair_parts
        else torch.zeros((2, 0), dtype=torch.long)
    )
    batch.struct_n_patches = int(patch_offset)
    return batch


def make_fsar_loader(
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
        collate_fn=fsar_collate,
    )


# ---------------------------------------------------------------------------
# parameter accounting
# ---------------------------------------------------------------------------


def _n_params(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return int(sum(parameter.numel() for parameter in module.parameters()))


def parameter_breakdown(model: PatchPathFSARModel) -> dict[str, Any]:
    encoder = model.encoder
    breakdown = {
        "A_encoder": int(
            _n_params(encoder.atom_embedding)
            + _n_params(encoder.bond_embedding)
            + _n_params(encoder.atom_mlp)
            + _n_params(encoder.bond_mlp)
            + _n_params(encoder.attribute_fuse)
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
    return breakdown

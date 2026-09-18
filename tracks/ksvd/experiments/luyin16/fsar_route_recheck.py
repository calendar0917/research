"""FSAR route recheck: clean binding null + explicit local structure (ZINC).

This module implements exactly the two corrections pre-registered in
``notes/fsar_route_recheck_preregistration.md`` and nothing else:

1. an **operator-matched, active-capacity-matched assignment-independent null**
   ``B_indep`` that reuses the *same module objects* as the aligned binding
   ``B_align`` but replaces the aligned product moments with the analytical
   independent-assignment moments (``m = 0``,
   ``sigma = sqrt(mean(p^2) * mean(q^2) + eps)``);
2. an **explicit local structural basis** ``S_explicit`` that replaces the
   latent topology-GNN state ``S_latent`` while keeping FSAR-v1's ``A``, the
   topology-only pair relation ``R_S`` and the global topology hinge ``G_S``
   untouched.

Nothing about FSAR-v1 ``A`` changes.  ``A_exact`` is deliberately absent.

Mode grid::

    A      local S none      B none
    SA     local S latent    B none
    SAB    local S latent    B aligned          (bit-for-bit FSAR-v1 SAB)
    SAM    local S latent    B marginal MLP     (bit-for-bit FSAR-v1 SAM)
    SABI   local S latent    B independent null
    SAE    local S explicit  B none
    SABE   local S explicit  B aligned explicit
    SABEI  local S explicit  B independent null explicit

Official ZINC test is never read.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16 import fsar as v1
from tracks.ksvd.experiments.luyin16 import fsar_v2 as v2

# ---------------------------------------------------------------------------
# frozen widths (reused from FSAR-v1 / FSAR-v2; no sweep)
# ---------------------------------------------------------------------------

ROLE_DIM = v1.ROLE_DIM  # 32
S_DIM = v1.S_DIM  # 32
S_HIDDEN = v1.S_HIDDEN  # 32
A_DIM = v1.A_DIM  # 32
A_SELF_DIM = v1.A_SELF_DIM  # 16
A_CTX_DIM = v1.A_CTX_DIM  # 16
A_HIDDEN = v1.A_HIDDEN  # 32
ATTR_DIM = v1.ATTR_DIM  # 16
B_DIM = v1.B_DIM  # 32
B_HIDDEN = v1.B_HIDDEN  # 32
ROUNDS = v1.ROUNDS  # 2
ACTIVATION = v1.ACTIVATION  # silu
INCLUDE_STD_POOL = v1.INCLUDE_STD_POOL  # True

H_DIM = v1.H_DIM
Q_DIM = v1.Q_DIM
T_ROUNDS = v1.T_ROUNDS
CENTER_HIDDEN = v1.CENTER_HIDDEN
RELATION_HIDDEN = v1.RELATION_HIDDEN
PAIR_HIDDEN = v1.PAIR_HIDDEN
HEAD_HIDDEN = v1.HEAD_HIDDEN
DROPOUT = v1.DROPOUT

# explicit basis (from FSAR-v2; reused verbatim)
NODE_BASIS_DIM = v2.NODE_BASIS_DIM  # 11
EDGE_BASIS_DIM = v2.EDGE_BASIS_DIM  # 15
S_EXPLICIT_INPUT_DIM = v2.S_EXPLICIT_INPUT_DIM  # 63

#: same marginal-only capacity control as FSAR-v1 ``SAM``
SAM_MARGINAL_DIM = v1.SAM_MARGINAL_DIM  # 64
SAM_MARGINAL_HIDDEN = v1.SAM_MARGINAL_HIDDEN  # 72

#: numerical floor for the independent-assignment second moment
INDEP_EPS = 1.0e-8

#: binding kinds
BINDING_KINDS = (None, "align", "indep")
#: local structure kinds
LOCAL_KINDS = (None, "latent", "explicit")

MODE_LOCAL_S: dict[str, str | None] = {
    "A": None,
    "SA": "latent",
    "SAB": "latent",
    "SAM": "latent",
    "SABI": "latent",
    "SAE": "explicit",
    "SABE": "explicit",
    "SABEI": "explicit",
}
MODE_BINDING: dict[str, str | None] = {
    "A": None,
    "SA": None,
    "SAB": "align",
    "SAM": None,
    "SABI": "indep",
    "SAE": None,
    "SABE": "align",
    "SABEI": "indep",
}
MODE_CAPACITY = {"SAM": True}
MODES = ("A", "SA", "SAB", "SAM", "SABI", "SAE", "SABE", "SABEI")
ALL_MODES = MODES


def mode_input_dim(mode: str) -> int:
    if mode not in MODE_LOCAL_S:
        raise ValueError(f"unknown FSAR route-recheck mode={mode!r}")
    dim = A_DIM
    if MODE_LOCAL_S[mode] is not None:
        dim += S_DIM
    if MODE_BINDING[mode] is not None:
        dim += B_DIM
    if MODE_CAPACITY.get(mode, False):
        dim += SAM_MARGINAL_DIM
    return int(dim)


# ---------------------------------------------------------------------------
# independent-assignment moments
# ---------------------------------------------------------------------------


def independent_moments(
    p: torch.Tensor, q: torch.Tensor, group: torch.Tensor, n_groups: int
) -> tuple[torch.Tensor, torch.Tensor]:
    """Analytical assignment-independent moments of ``p_i * q_{pi(i)}``.

    For centered projections ``p`` / ``q`` (``mean_i p_i = mean_i q_i = 0``)
    and a uniform random reassignment ``pi``::

        E_pi[ mean_i (p_i * q_{pi(i)}) ]     = 0
        E_pi[ mean_i (p_i^2 * q_{pi(i)}^2) ] = mean_i(p_i^2) * mean_i(q_i^2)

    Returns ``(m_indep, sigma_indep)`` with ``m_indep = 0`` and
    ``sigma_indep = sqrt(mean(p^2) * mean(q^2) + eps)``.  Reads no pairing.
    """
    p_squared_mean = v1.scatter_mean(p * p, group, n_groups)
    q_squared_mean = v1.scatter_mean(q * q, group, n_groups)
    sigma = torch.sqrt(p_squared_mean * q_squared_mean + INDEP_EPS)
    return torch.zeros_like(sigma), sigma


# ---------------------------------------------------------------------------
# latent local-S encoder (FSAR-v1 encoder + selectable binding kind)
# ---------------------------------------------------------------------------


class FSARLatentRouteEncoder(v1.FSARChannelEncoder):
    """FSAR-v1 encoder with ``binding_kind`` in ``{"align", "indep"}``.

    Module construction is inherited *unchanged*, so with ``binding_kind="align"``
    this class is bit-for-bit :class:`fsar.FSARChannelEncoder` with
    ``include_structure=True, include_binding=True`` (verified by
    ``test_latent_align_matches_fsar_v1``).
    """

    def __init__(
        self,
        *,
        binding_kind: str | None,
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
        include_std_pool: bool = INCLUDE_STD_POOL,
    ) -> None:
        if binding_kind not in (None, "align", "indep"):
            raise ValueError(
                "latent route encoder needs binding_kind in "
                f"(None, 'align', 'indep'); got {binding_kind!r}"
            )
        super().__init__(
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
            include_structure=True,
            include_binding=binding_kind is not None,
            include_std_pool=bool(include_std_pool),
        )
        self.binding_kind = binding_kind

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
        if self.binding_kind == "align":
            return super()._binding_vector(
                role,
                node_attribute,
                edge_attribute,
                node_patch,
                edge_group,
                n_patches,
                data,
            )

        # ---- independent-assignment null (same modules, no pairing) --------
        role_mean = v1.scatter_mean(role, node_patch, n_patches)
        attribute_mean = v1.scatter_mean(node_attribute, node_patch, n_patches)
        centered_role = role - role_mean[node_patch]
        centered_attribute = node_attribute - attribute_mean[node_patch]
        # p_i and q_i are the *same* projections as the aligned organ.
        p_node = self.node_role_projection(centered_role)
        q_node = self.node_attribute_projection(centered_attribute)
        m_node, sigma_node = independent_moments(
            p_node, q_node, node_patch, n_patches
        )

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
            edge_role_mean = v1.scatter_mean(edge_role, edge_group, n_patches)
            edge_attribute_mean = v1.scatter_mean(
                edge_attribute_role, edge_group, n_patches
            )
            p_edge = self.edge_role_projection(
                edge_role - edge_role_mean[edge_group]
            )
            q_edge = self.edge_attribute_projection(
                edge_attribute_role - edge_attribute_mean[edge_group]
            )
            m_edge, sigma_edge = independent_moments(
                p_edge, q_edge, edge_group, n_patches
            )
        else:
            m_edge = torch.zeros_like(m_node)
            sigma_edge = torch.zeros_like(sigma_node)
        return self.binding_fuse(
            torch.cat([m_node, sigma_node, m_edge, sigma_edge], dim=1)
        )


# ---------------------------------------------------------------------------
# explicit local-S encoder (FSAR-v1 A + explicit rooted structural basis)
# ---------------------------------------------------------------------------


class FSARExplicitRouteEncoder(v1.FSARChannelEncoder):
    """FSAR-v1 ``A`` + explicit local structural basis ``S`` (+ aligned/null B).

    The explicit path contains **no adjacency message passing** and no learned
    structural vocabulary: it only combines the fixed readable rooted
    coordinates (node 11-D, edge 15-D) with the shared MLP.  The attribute side
    of ``B`` is the *same* FSAR-v1 learned entity semantics used by the latent
    route.
    """

    def __init__(
        self,
        *,
        binding_kind: str | None,
        s_dim: int = S_DIM,
        s_hidden: int = S_HIDDEN,
        a_self_dim: int = A_SELF_DIM,
        a_ctx_dim: int = A_CTX_DIM,
        b_dim: int = B_DIM,
        attr_dim: int = ATTR_DIM,
        a_hidden: int = A_HIDDEN,
        b_hidden: int = B_HIDDEN,
        activation: str = ACTIVATION,
        include_std_pool: bool = INCLUDE_STD_POOL,
    ) -> None:
        if binding_kind not in (None, "align", "indep"):
            raise ValueError(f"unknown binding_kind={binding_kind!r}")
        # Build the shared FSAR-v1 attribute stream only; the explicit structure
        # and binding modules are attached below.
        super().__init__(
            s_dim=int(s_dim),
            a_self_dim=int(a_self_dim),
            a_ctx_dim=int(a_ctx_dim),
            b_dim=int(b_dim),
            attr_dim=int(attr_dim),
            s_hidden=int(s_hidden),
            a_hidden=int(a_hidden),
            b_hidden=int(b_hidden),
            activation=str(activation),
            include_structure=False,
            include_binding=False,
            include_std_pool=bool(include_std_pool),
        )
        self.binding_kind = binding_kind
        self.include_structure = True

        # ---- explicit structure readout (no adjacency message passing) -----
        self.structure_pool = nn.Sequential(
            nn.Linear(S_EXPLICIT_INPUT_DIM, int(s_hidden)),
            v1._act(str(activation)),
            nn.Linear(int(s_hidden), int(s_dim)),
        )

        # ---- explicit binding modules (same capacity class as latent B) ----
        if binding_kind is not None:
            self.include_binding = True
            self.node_role_projection = nn.Linear(
                NODE_BASIS_DIM, int(b_dim), bias=False
            )
            self.node_attribute_projection = nn.Linear(
                int(attr_dim), int(b_dim), bias=False
            )
            self.edge_role_mlp = nn.Sequential(
                nn.Linear(EDGE_BASIS_DIM, int(b_hidden)),
                v1._act(str(activation)),
                nn.Linear(int(b_hidden), int(b_dim)),
            )
            self.edge_attribute_mlp = nn.Sequential(
                nn.Linear(int(attr_dim), int(b_hidden)),
                v1._act(str(activation)),
                nn.Linear(int(b_hidden), int(b_dim)),
            )
            self.edge_role_projection = nn.Linear(
                int(b_dim), int(b_dim), bias=False
            )
            self.edge_attribute_projection = nn.Linear(
                int(b_dim), int(b_dim), bias=False
            )
            self.binding_fuse = nn.Sequential(
                nn.Linear(4 * int(b_dim), int(b_hidden)),
                v1._act(str(activation)),
                nn.Linear(int(b_hidden), int(b_dim)),
            )
        else:
            self.include_binding = False

    # -- explicit structure ------------------------------------------------

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
            edge_mean = node_basis.new_zeros(
                (n_patches, int(edge_basis.shape[1]))
            )
            edge_std = torch.zeros_like(edge_mean)
        combined = torch.cat(
            [root_basis, node_mean, node_std, edge_mean, edge_std], dim=1
        )
        if int(combined.shape[1]) != S_EXPLICIT_INPUT_DIM:
            raise RuntimeError(
                f"explicit S input width {combined.shape[1]} != {S_EXPLICIT_INPUT_DIM}"
            )
        return self.structure_pool(combined)

    # -- explicit binding --------------------------------------------------

    def _explicit_binding(
        self,
        node_basis: torch.Tensor,
        edge_basis: torch.Tensor,
        node_attribute: torch.Tensor,
        edge_attribute: torch.Tensor,
        node_patch: torch.Tensor,
        edge_group: torch.Tensor,
        n_patches: int,
    ) -> torch.Tensor:
        role_mean = v1.scatter_mean(node_basis, node_patch, n_patches)
        attribute_mean = v1.scatter_mean(node_attribute, node_patch, n_patches)
        centered_role = node_basis - role_mean[node_patch]
        centered_attribute = node_attribute - attribute_mean[node_patch]
        p_node = self.node_role_projection(centered_role)
        q_node = self.node_attribute_projection(centered_attribute)

        if edge_attribute.numel():
            edge_role = self.edge_role_mlp(edge_basis)
            edge_attribute_role = self.edge_attribute_mlp(edge_attribute)
            edge_role_mean = v1.scatter_mean(edge_role, edge_group, n_patches)
            edge_attribute_mean = v1.scatter_mean(
                edge_attribute_role, edge_group, n_patches
            )
            p_edge = self.edge_role_projection(
                edge_role - edge_role_mean[edge_group]
            )
            q_edge = self.edge_attribute_projection(
                edge_attribute_role - edge_attribute_mean[edge_group]
            )
            has_edges = True
        else:
            has_edges = False

        if self.binding_kind == "align":
            z_node = p_node * q_node
            m_node = v1.scatter_mean(z_node, node_patch, n_patches)
            sigma_node = v1.masked_std(z_node, node_patch, n_patches, m_node)
            if has_edges:
                z_edge = p_edge * q_edge
                m_edge = v1.scatter_mean(z_edge, edge_group, n_patches)
                sigma_edge = v1.masked_std(
                    z_edge, edge_group, n_patches, m_edge
                )
            else:
                m_edge = torch.zeros_like(m_node)
                sigma_edge = torch.zeros_like(sigma_node)
        else:
            m_node, sigma_node = independent_moments(
                p_node, q_node, node_patch, n_patches
            )
            if has_edges:
                m_edge, sigma_edge = independent_moments(
                    p_edge, q_edge, edge_group, n_patches
                )
            else:
                m_edge = torch.zeros_like(m_node)
                sigma_edge = torch.zeros_like(sigma_node)
        return self.binding_fuse(
            torch.cat([m_node, sigma_node, m_edge, sigma_edge], dim=1)
        )

    # -- forward -----------------------------------------------------------

    def forward_channels(self, data: object) -> dict[str, torch.Tensor]:
        unique_patch, node_patch, n_patches = self._patch_geometry(data)
        edge_patch_ids = data.struct_edge_patch.long()
        edge_group = (
            torch.searchsorted(unique_patch, edge_patch_ids)
            if edge_patch_ids.numel()
            else edge_patch_ids
        )
        atom = data.struct_atom.long()
        node_attribute = self.atom_mlp(self.atom_embedding(atom))
        bond = data.struct_bond.long()
        edge_attribute = (
            self.bond_mlp(self.bond_embedding(bond))
            if bond.numel()
            else node_attribute.new_zeros((0, self.attr_dim))
        )
        attributes = self._attribute_vector(
            node_attribute,
            edge_attribute,
            data,
            node_patch,
            n_patches,
            unique_patch,
        )
        structure = self._explicit_structure(
            data, node_patch, edge_group, n_patches
        )
        channels: dict[str, torch.Tensor] = {
            "attributes": attributes,
            "structure": structure,
            "node_attribute": node_attribute,
            "edge_attribute": edge_attribute,
            "node_basis": data.node_basis.float(),
            "edge_basis": data.edge_basis.float(),
        }
        if self.include_binding:
            channels["binding"] = self._explicit_binding(
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


# ---------------------------------------------------------------------------
# full backbone (identical core to FSAR-v1)
# ---------------------------------------------------------------------------


class PatchPathFSARRouteModel(nn.Module):
    """FSAR route-recheck backbone: FSAR-v1 A + selectable local S / B.

    ``node_init``, the topology-only pair relation core, the global topology
    hinge, graph pooling and the head are copied unchanged from
    :class:`fsar.PatchPathFSARModel`, in the same construction order, so the
    latent aligned/null modes consume an identical RNG stream and share an
    identical core initialisation for a fixed seed.
    """

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
        s_dim: int = S_DIM,
        s_hidden: int = S_HIDDEN,
        role_dim: int = ROLE_DIM,
        rounds: int = ROUNDS,
        a_self_dim: int = A_SELF_DIM,
        a_ctx_dim: int = A_CTX_DIM,
        b_dim: int = B_DIM,
        attr_dim: int = ATTR_DIM,
        a_hidden: int = A_HIDDEN,
        b_hidden: int = B_HIDDEN,
        activation: str = ACTIVATION,
    ) -> None:
        super().__init__()
        mode = str(mode)
        if mode not in ALL_MODES:
            raise ValueError(
                f"unknown FSAR route-recheck mode={mode!r}; expected {ALL_MODES}"
            )
        self.mode = mode
        self.hidden = int(hidden)
        self.q_dim = int(q_dim)
        self.recurrence_rounds = int(recurrence_rounds)
        self.use_topology_channel = bool(use_topology_channel)
        local_kind = MODE_LOCAL_S[mode]
        binding_kind = MODE_BINDING[mode]
        if local_kind == "latent":
            self.encoder: nn.Module = FSARLatentRouteEncoder(
                binding_kind=binding_kind,
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
            )
        elif local_kind == "explicit":
            self.encoder = FSARExplicitRouteEncoder(
                binding_kind=binding_kind,
                s_dim=int(s_dim),
                s_hidden=int(s_hidden),
                a_self_dim=int(a_self_dim),
                a_ctx_dim=int(a_ctx_dim),
                b_dim=int(b_dim),
                attr_dim=int(attr_dim),
                a_hidden=int(a_hidden),
                b_hidden=int(b_hidden),
                activation=str(activation),
            )
        else:
            self.encoder = v1.FSARChannelEncoder(
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
                include_structure=False,
                include_binding=False,
            )
        input_dim = mode_input_dim(mode)
        self.capacity_mlp = None
        if MODE_CAPACITY.get(mode, False):
            self.capacity_mlp = nn.Sequential(
                nn.Linear(A_DIM + S_DIM, SAM_MARGINAL_HIDDEN),
                v1._act("relu"),
                nn.Linear(SAM_MARGINAL_HIDDEN, SAM_MARGINAL_DIM),
            )
        self.node_init = nn.Sequential(
            nn.Linear(input_dim, 128),
            v1._act("relu"),
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

    # -- pooling helpers (identical to FSAR-v1) -----------------------------

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

    # -- forward -------------------------------------------------------------

    def encode(self, data: Any) -> torch.Tensor:
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

    def forward(self, data: Any) -> torch.Tensor:
        return self.head(self.encode(data)).view(-1)


# ---------------------------------------------------------------------------
# batch collation
# ---------------------------------------------------------------------------


def fsar_route_collate(data_list: Sequence[Any]) -> Any:
    """FSAR-v1 collate plus the explicit-basis tensors (always present)."""
    return v2.fsar_v2_collate(data_list)


def make_fsar_route_loader(
    graphs: Sequence[Any], batch_size: int, shuffle: bool, seed: int
):
    from torch.utils.data import DataLoader as TorchDataLoader

    generator = torch.Generator().manual_seed(int(seed)) if shuffle else None
    return TorchDataLoader(
        list(graphs),
        batch_size=int(batch_size),
        shuffle=bool(shuffle),
        generator=generator,
        num_workers=0,
        collate_fn=fsar_route_collate,
    )


# ---------------------------------------------------------------------------
# parameter accounting / module groups
# ---------------------------------------------------------------------------


def _n_params(module: nn.Module | None) -> int:
    if module is None:
        return 0
    return int(sum(parameter.numel() for parameter in module.parameters()))


def module_groups(
    model: PatchPathFSARRouteModel,
) -> dict[str, list[tuple[str, nn.Module]]]:
    """Named module groups used for parameter / active-path accounting.

    Names are unique inside a group so the module-granularity active-path audit
    cannot be corrupted by two modules of the same class in one group.
    """
    encoder = model.encoder
    raw: dict[str, list[tuple[str, nn.Module | None]]] = {
        "A": [
            ("atom_embedding", encoder.atom_embedding),
            ("bond_embedding", encoder.bond_embedding),
            ("atom_mlp", encoder.atom_mlp),
            ("bond_mlp", encoder.bond_mlp),
            ("attribute_fuse", encoder.attribute_fuse),
        ],
        "local_S": [
            ("root_embedding", getattr(encoder, "root_embedding", None)),
            ("distance_embedding", getattr(encoder, "distance_embedding", None)),
            ("topology_base", None),
            ("message", getattr(encoder, "message", None)),
            ("update", getattr(encoder, "update", None)),
            ("structure_pool", encoder.structure_pool),
        ],
        "B": [
            ("node_role_projection", encoder.node_role_projection),
            ("node_attribute_projection", encoder.node_attribute_projection),
            ("edge_role_mlp", encoder.edge_role_mlp),
            ("edge_attribute_mlp", encoder.edge_attribute_mlp),
            ("edge_role_projection", encoder.edge_role_projection),
            ("edge_attribute_projection", encoder.edge_attribute_projection),
            ("binding_fuse", encoder.binding_fuse),
        ],
        "node_init": [("node_init", model.node_init)],
        "capacity_mlp": [("capacity_mlp", model.capacity_mlp)],
        "relation_core": [
            ("pair_projection", model.pair_projection),
            ("relation_encoder", model.relation_encoder),
            ("distance_gate", model.distance_gate),
            ("pair_encoder", model.pair_encoder),
            ("center_update", model.center_update),
        ],
        "topology_channel": [("topology_encoder", model.topology_encoder)],
        "graph_head": [("head", model.head)],
    }
    return {
        group: [
            (name, module) for name, module in modules if module is not None
        ]
        for group, modules in raw.items()
    }


def parameter_breakdown(model: PatchPathFSARRouteModel) -> dict[str, Any]:
    groups = module_groups(model)
    breakdown: dict[str, Any] = {
        f"{name}_params": int(
            sum(_n_params(module) for _n, module in modules)
        )
        for name, modules in groups.items()
    }
    breakdown["topology_base_params"] = (
        int(model.encoder.topology_base.numel())
        if getattr(model.encoder, "topology_base", None) is not None
        else 0
    )
    breakdown["total"] = _n_params(model)
    breakdown["trainable"] = int(
        sum(
            parameter.numel()
            for parameter in model.parameters()
            if parameter.requires_grad
        )
    )
    encoder = model.encoder
    breakdown["mode"] = model.mode
    breakdown["local_s_kind"] = MODE_LOCAL_S[model.mode]
    breakdown["binding_kind"] = MODE_BINDING[model.mode]
    breakdown["structure_kind"] = getattr(encoder, "structure_kind", None) or (
        "explicit" if MODE_LOCAL_S[model.mode] == "explicit" else (
            "implicit" if MODE_LOCAL_S[model.mode] == "latent" else None
        )
    )
    breakdown["dataset_dependent_vocabulary_params"] = 0
    return breakdown


def active_path_from_grads(model: PatchPathFSARRouteModel) -> dict[str, Any]:
    """Module-granularity active-path accounting from the current ``.grad`` values.

    Counts parameters whose *owning module* receives a nonzero task gradient.
    This is the definition pre-registered in section 5.2 of the route-recheck
    note: it directly addresses the FSAR-v2 failure mode where the total
    parameter count matched but whole semantic modules were outside the
    prediction path.

    Also reports the exact count of individual parameter elements with zero
    gradient, which for the assignment-independent null is exactly the
    ``binding_fuse`` first-moment input columns (section 5.2).
    """
    groups = module_groups(model)
    payload: dict[str, Any] = {
        "active_modules": {},
        "active_prediction_path_params": 0,
        "group_active_params": {},
        "group_total_params": {},
        "zero_grad_parameter_elements": 0,
        "zero_grad_parameter_names": [],
    }
    active_total = 0
    module_level_active_params = 0
    seen: set[int] = set()
    for name, modules in groups.items():
        group_active = 0
        group_total = 0
        for module_name, module in modules:
            module_active = 0
            for parameter in module.parameters():
                group_total += int(parameter.numel())
                if id(parameter) in seen:
                    continue
                seen.add(id(parameter))
                grad = parameter.grad
                n_zero = (
                    int(parameter.numel())
                    if grad is None
                    else int((grad == 0).sum())
                )
                module_active += int(parameter.numel()) - n_zero
                if n_zero > 0:
                    payload["zero_grad_parameter_elements"] += n_zero
                    payload["zero_grad_parameter_names"].append(
                        {
                            "group": f"{name}",
                            "module": f"{name}.{module_name}",
                            "n_zero": int(n_zero),
                            "n_total": int(parameter.numel()),
                        }
                    )
            is_active = bool(module_active > 0)
            payload["active_modules"][f"{name}.{module_name}"] = is_active
            if is_active:
                module_level_active_params += _n_params(module)
            group_active += module_active
        payload["group_active_params"][name] = int(group_active)
        payload["group_total_params"][name] = int(group_total)
        active_total += group_active
    # ``topology_base`` is a bare parameter of the latent structure stream.
    base = getattr(model.encoder, "topology_base", None)
    if base is not None and base.grad is not None and bool((base.grad != 0).any()):
        module_level_active_params += int(base.numel())
    payload["active_prediction_path_element_params"] = int(active_total)
    # Module-granularity definition (pre-registered section 5.2): a module
    # counts with its full parameter budget when any element is active.  This is
    # the quantity that must match exactly between aligned and null binding.
    payload["active_prediction_path_params"] = int(module_level_active_params)
    payload["all_modules_active"] = bool(all(payload["active_modules"].values()))
    return payload


def active_path_accounting(
    model: PatchPathFSARRouteModel, batch: Any
) -> dict[str, Any]:
    """Run one training step on ``batch`` and return the active-path summary."""
    import torch.nn.functional as F

    was_training = model.training
    model.train()
    model.zero_grad()
    loss = F.l1_loss(model(batch).view(-1), batch.y.view(-1))
    loss.backward()
    payload = active_path_from_grads(model)
    payload["loss"] = float(loss.detach())
    model.zero_grad()
    model.train(was_training)
    return payload

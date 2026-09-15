"""Shared connectivity-aware encoder for rooted typed radius-2 patches.

Motivation
----------
The historical compact-v2/v4/v6 patch representation turns each rooted typed
patch into a categorical certificate, maps it to a vocabulary id and looks up a
learned row (``typed_embedding``).  That representation is an opaque,
vocabulary-sized memory: it cannot encode an unseen certificate without a row
and its parameter count grows with the certificate vocabulary.

This module is the *representation-forming* alternative: a single small encoder
whose parameters are shared by every patch of every molecule.  It reads the
real rooted typed patch graph -- atom type, root flag, distance from the root
and bond type -- and produces ``e_struct in R^output_dim`` through two rounds of
edge-aware message passing followed by a permutation-invariant pooling and a
fusion MLP.  No exact patch identity is ever looked up.

Form (exactly one pre-registered architecture; no attention, no Transformer, no
learned token dictionary)::

    z_v^0 = atom_embed(atom_v) + root_embed(root_v) + dist_embed(dist_v)
    for r in {0, 1}:
        m_v = mean_{u in N(v)} G_r([z_u ; bond_embed(bond_uv)])
        z_v = z_v + F_r([z_v ; m_v])
    e_struct = H([z_root ; mean_v z_v ; std_v z_v])

The two rounds use independent parameters (``G_0``/``F_0`` and ``G_1``/``F_1``)
so the depth is a genuine two-hop computation over the patch connectivity.  The
aggregations are means over multiset inputs, so ``e_struct`` is invariant to a
permutation of the patch's node labels and to the arbitrary direction in which
each undirected bond is listed.

Connection to the historical v6 attribute branch
------------------------------------------------
The compact-v6 ``_AttributeEncoder`` pooled atom/bond (type, topological role)
primitives *independently* (atom MLP and bond MLP, then a fusion) and kept the
coarse typed lookup.  This encoder is different in kind: it propagates messages
along the actual patch edges, its root state is a function of its neighbours,
and it *replaces* the typed lookup entirely (see
``notes/shared_structural_patch_encoder.md``).

Preprocessing contract
----------------------
The encoder consumes per-patch flat tensors attached to the PyG ``Data`` object
by the experiment module (``zinc_shared_structural_patch_encoder``):

``struct_atom``   int64 ``[T]``  atom category per patch-node
``struct_root``   int64 ``[T]``  1 for the patch root, else 0
``struct_dist``   int64 ``[T]``  BFS distance from the patch root (0..radius)
``struct_patch``  int64 ``[T]``  global patch index owning the node
``struct_src``    int64 ``[E]``  global patch-node index (directed edge source)
``struct_dst``    int64 ``[E]``  global patch-node index (directed edge target)
``struct_bond``   int64 ``[E]``  bond category
``struct_n_patches`` int         total number of patches in the batch

Every undirected bond is stored twice (``u -> v`` and ``v -> u``).
"""

from __future__ import annotations

from typing import Any

import torch
import torch.nn as nn


def scatter_mean(
    source: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    """Permutation-invariant mean of ``source`` rows into ``dim_size`` groups."""
    dim_size = int(dim_size)
    if source.numel() == 0 or dim_size <= 0:
        return source.new_zeros((max(dim_size, 0), int(source.shape[1])))
    total = source.new_zeros((dim_size, int(source.shape[1])))
    total.index_add_(0, index.long(), source)
    counts = (
        torch.bincount(index.long(), minlength=dim_size)
        .clamp_min(1)
        .to(source.dtype)
        .unsqueeze(1)
    )
    return total / counts


class SharedStructuralPatchEncoder(nn.Module):
    """Two-round edge-aware message-passing encoder over rooted typed patches.

    All parameters are shared across every patch and every molecule; the module
    is vocabulary-free and can encode a certificate never seen during training.
    """

    def __init__(
        self,
        *,
        atom_categories: int = 28,
        bond_categories: int = 4,
        n_distance_bins: int = 3,
        node_dim: int = 48,
        edge_dim: int = 24,
        hidden_dim: int = 48,
        output_dim: int = 16,
        rounds: int = 2,
        include_std_pool: bool = True,
    ) -> None:
        super().__init__()
        self.atom_categories = int(atom_categories)
        self.bond_categories = int(bond_categories)
        self.n_distance_bins = int(n_distance_bins)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.hidden_dim = int(hidden_dim)
        self.output_dim = int(output_dim)
        self.rounds = int(rounds)
        self.include_std_pool = bool(include_std_pool)
        if min(
            self.atom_categories,
            self.bond_categories,
            self.n_distance_bins,
            self.node_dim,
            self.edge_dim,
            self.hidden_dim,
            self.output_dim,
        ) < 1:
            raise ValueError("structural encoder dimensions must be positive")
        if self.rounds < 1:
            raise ValueError("structural encoder needs at least one round")

        self.atom_embedding = nn.Embedding(self.atom_categories, self.node_dim)
        self.root_embedding = nn.Embedding(2, self.node_dim)
        self.distance_embedding = nn.Embedding(
            self.n_distance_bins, self.node_dim
        )
        self.bond_embedding = nn.Embedding(self.bond_categories, self.edge_dim)

        # One message MLP G_r and one update MLP F_r per round.  The rounds are
        # independent parameters so depth is a real two-hop computation.
        self.message = nn.ModuleList(
            nn.Sequential(
                nn.Linear(self.node_dim + self.edge_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.node_dim),
            )
            for _ in range(self.rounds)
        )
        self.update = nn.ModuleList(
            nn.Sequential(
                nn.Linear(2 * self.node_dim, self.hidden_dim),
                nn.ReLU(),
                nn.Linear(self.hidden_dim, self.node_dim),
            )
            for _ in range(self.rounds)
        )
        fusion_input = self.node_dim * (3 if self.include_std_pool else 2)
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, self.hidden_dim),
            nn.ReLU(),
            nn.Linear(self.hidden_dim, self.output_dim),
        )

    def forward(self, data: object) -> torch.Tensor:
        atom = data.struct_atom.long()
        root = data.struct_root.long()
        distance = data.struct_dist.long()
        # Patch ids are globally unique within a split; remap them to contiguous
        # batch-local group indices (deterministic, sorting-based).
        _unique_patch, node_patch = torch.unique(
            data.struct_patch.long(), return_inverse=True
        )
        n_patches = int(_unique_patch.numel())
        n_nodes = int(atom.shape[0])

        state = (
            self.atom_embedding(atom)
            + self.root_embedding(root)
            + self.distance_embedding(distance)
        )

        src = data.struct_src.long()
        dst = data.struct_dst.long()
        bond = data.struct_bond.long()
        if bond.numel():
            edge = self.bond_embedding(bond)
        else:
            edge = state.new_zeros((0, self.edge_dim))

        for round_index in range(self.rounds):
            if src.numel():
                messages = self.message[round_index](
                    torch.cat([state[src], edge], dim=1)
                )
                aggregate = scatter_mean(messages, dst, n_nodes)
            else:
                aggregate = torch.zeros_like(state)
            state = state + self.update[round_index](
                torch.cat([state, aggregate], dim=1)
            )

        mean = scatter_mean(state, node_patch, n_patches)
        blocks = [mean]
        if self.include_std_pool:
            dim_size = n_patches
            total = state.new_zeros((dim_size, self.node_dim))
            total.index_add_(0, node_patch, state * state)
            counts = (
                torch.bincount(node_patch, minlength=dim_size)
                .clamp_min(1)
                .to(state.dtype)
                .unsqueeze(1)
            )
            second_moment = total / counts
            variance = (second_moment - mean * mean).clamp_min(0.0)
            blocks.append(torch.sqrt(variance + 1.0e-8))

        root_mask = root > 0
        root_state = state.new_zeros((n_patches, self.node_dim))
        if bool(root_mask.any()):
            root_state.index_add_(
                0, node_patch[root_mask], state[root_mask]
            )
        blocks.insert(0, root_state)
        return self.fusion(torch.cat(blocks, dim=1))


class SharedBagPatchEncoder(nn.Module):
    """Permutation-invariant DeepSets encoder over rooted-patch primitives.

    This is the *connectivity-free* ablation of
    :class:`SharedStructuralPatchEncoder`.  It reads exactly the same input
    primitives -- atom type, root flag, distance from the root and bond type --
    but it **never** consumes ``struct_src`` / ``struct_dst`` and never
    propagates a message along a real patch edge.  A patch is treated as a bag
    of node primitives plus a bag of bond-type primitives, grouped only by the
    patch they belong to (``struct_patch`` / ``struct_edge_patch`` are grouping
    indices, not adjacency).

    Form (one pre-registered architecture; no attention, no message passing)::

        h_v = node_MLP(atom_embed(atom_v) + root_embed(root_v) + dist_embed(d_v))
        g_e = bond_MLP(bond_embed(bond_e))
        e_bag = fusion([h_root ; mean_v h_v ; std_v h_v ;
                        mean_e g_e ; std_e g_e])

    Because every aggregation is a mean / second-moment over a multiset and the
    root term is the shared node map evaluated at the unique root, ``e_bag`` is
    invariant to a permutation of the patch's node labels and to the arbitrary
    direction / order in which the undirected bonds are listed.  Two patches
    with identical atom / root / distance / bond-type multisets but **different
    actual connectivity** therefore receive exactly the same ``e_bag`` -- which
    is precisely the variable this encoder is meant to isolate.

    Attributes ``hidden_dim``, ``rounds`` and ``include_std_pool`` mirror the
    interface of :class:`SharedStructuralPatchEncoder` so the shared parameter
    audit keeps working; ``rounds == 0`` records that no message passing is
    performed.

    Preprocessing contract (attached by ``zinc_shared_bag_patch_encoder``):

    ``struct_atom``        int64 ``[T]``  atom category per patch-node
    ``struct_root``        int64 ``[T]``  1 for the patch root, else 0
    ``struct_dist``        int64 ``[T]``  BFS distance from the patch root
    ``struct_patch``       int64 ``[T]``  patch index owning each node
    ``struct_bond``        int64 ``[E]``  bond category per directed edge
    ``struct_edge_patch``  int64 ``[E]``  patch index owning each directed edge

    ``struct_src`` / ``struct_dst`` are deliberately **not** read.
    """

    def __init__(
        self,
        *,
        atom_categories: int = 28,
        bond_categories: int = 4,
        n_distance_bins: int = 3,
        node_dim: int = 48,
        edge_dim: int = 24,
        node_hidden: int = 96,
        bond_hidden: int = 48,
        fusion_hidden: int = 104,
        output_dim: int = 16,
    ) -> None:
        super().__init__()
        self.atom_categories = int(atom_categories)
        self.bond_categories = int(bond_categories)
        self.n_distance_bins = int(n_distance_bins)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.node_hidden = int(node_hidden)
        self.bond_hidden = int(bond_hidden)
        self.fusion_hidden = int(fusion_hidden)
        self.output_dim = int(output_dim)
        # Interface-compatible audit metadata (see class docstring).
        self.hidden_dim = int(fusion_hidden)
        self.rounds = 0
        self.include_std_pool = True
        self.kind = "shared_bag"
        if min(
            self.atom_categories,
            self.bond_categories,
            self.n_distance_bins,
            self.node_dim,
            self.edge_dim,
            self.node_hidden,
            self.bond_hidden,
            self.fusion_hidden,
            self.output_dim,
        ) < 1:
            raise ValueError("bag encoder dimensions must be positive")

        self.atom_embedding = nn.Embedding(self.atom_categories, self.node_dim)
        self.root_embedding = nn.Embedding(2, self.node_dim)
        self.distance_embedding = nn.Embedding(
            self.n_distance_bins, self.node_dim
        )
        self.bond_embedding = nn.Embedding(self.bond_categories, self.edge_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.node_hidden),
            nn.ReLU(),
            nn.Linear(self.node_hidden, self.node_dim),
        )
        self.bond_mlp = nn.Sequential(
            nn.Linear(self.edge_dim, self.bond_hidden),
            nn.ReLU(),
            nn.Linear(self.bond_hidden, self.edge_dim),
        )
        fusion_input = 3 * self.node_dim + 2 * self.edge_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, self.fusion_hidden),
            nn.ReLU(),
            nn.Linear(self.fusion_hidden, self.output_dim),
        )

    def _std_pool(
        self, values: torch.Tensor, group: torch.Tensor, n_groups: int, mean: torch.Tensor
    ) -> torch.Tensor:
        total = values.new_zeros((n_groups, int(values.shape[1])))
        if values.numel():
            total.index_add_(0, group, values * values)
        counts = (
            torch.bincount(group, minlength=n_groups)
            .clamp_min(1)
            .to(values.dtype)
            .unsqueeze(1)
        )
        second_moment = total / counts
        return torch.sqrt((second_moment - mean * mean).clamp_min(0.0) + 1.0e-8)

    def forward(self, data: object) -> torch.Tensor:
        atom = data.struct_atom.long()
        root = data.struct_root.long()
        distance = data.struct_dist.long()
        # Patch ids are grouping indices, not adjacency.  ``struct_edge_patch``
        # is required so that the bond bag is built without ever touching an
        # edge endpoint (``struct_src`` / ``struct_dst``).
        if not hasattr(data, "struct_edge_patch"):
            raise AttributeError(
                "SharedBagPatchEncoder requires struct_edge_patch "
                "(bond -- patch grouping); it never reads struct_src/struct_dst"
            )
        unique_patch, node_patch = torch.unique(
            data.struct_patch.long(), return_inverse=True
        )
        n_patches = int(unique_patch.numel())

        node_state = self.node_mlp(
            self.atom_embedding(atom)
            + self.root_embedding(root)
            + self.distance_embedding(distance)
        )
        node_mean = scatter_mean(node_state, node_patch, n_patches)
        node_std = self._std_pool(node_state, node_patch, n_patches, node_mean)

        root_mask = root > 0
        root_state = node_state.new_zeros((n_patches, self.node_dim))
        if bool(root_mask.any()):
            root_state.index_add_(
                0, node_patch[root_mask], node_state[root_mask]
            )

        edge_patch_ids = data.struct_edge_patch.long()
        if edge_patch_ids.numel():
            # ``unique_patch`` is sorted, so this maps an edge's patch id to
            # its batch-local group index without using any edge endpoint.
            edge_group = torch.searchsorted(unique_patch, edge_patch_ids)
            bond_state = self.bond_mlp(self.bond_embedding(data.struct_bond.long()))
            bond_mean = scatter_mean(bond_state, edge_group, n_patches)
            bond_std = self._std_pool(bond_state, edge_group, n_patches, bond_mean)
        else:
            bond_mean = node_state.new_zeros((n_patches, self.edge_dim))
            bond_std = node_state.new_zeros((n_patches, self.edge_dim))

        return self.fusion(
            torch.cat(
                [root_state, node_mean, node_std, bond_mean, bond_std], dim=1
            )
        )


def _straight_through_binary(soft: torch.Tensor) -> torch.Tensor:
    """Hard 0/1 forward membership with a straight-through gradient.

    Forward: ``round(soft > 0.5)``.  Backward: identity (the gradient of the
    hard mask with respect to ``soft`` is treated as 1), which is the standard
    straight-through estimator for a discrete membership variable.
    """
    hard = (soft > 0.5).to(soft.dtype)
    return soft + (hard - soft).detach()


class AdaptiveStructureBindingEncoder(nn.Module):
    """Adaptive Structure-Binding (ASB) cell for rooted typed radius-2 patches.

    This encoder realises Z1 / *Local ASB*: inside each existing radius-2 patch
    the model derives an explicit, connected support from the activation of
    structure--attribute *bindings*, and then performs the patch computation on
    that learned structure.  The radius-2 patch is the *search region*; the
    learned support is the *actual structure*.

    Unified computation (one cell, no parallel branch)::

        z_v   = node_mlp(atom_embed + root_embed + dist_embed)
        b_uv  = bond_mlp(bond_embed) + bind_delta([z_u+z_v, |z_u-z_v|, bond_embed])
        gate  = sigmoid(gate_mlp(b_uv))
        a_root = 1
        a_v    = gate(root--v)                       for dist(v)=1
        a_w    = 1 - prod_v (1 - a_v * gate(v--w))   for dist(w)=2, v in N(w)
        a_hard = StraightThroughBinary(a_soft)
        m_v    = mean_{u in N(v) and selected} message_mlp(b_uv)
        z'_v   = z_v + update_mlp([z_v, m_v])
        b'_uv  = b_uv + bind_update([z'_u+z'_v, |z'_u-z'_v|, bond_embed])
        e      = fusion([z'_root; mean_S z'; std_S z'; mean_E b'; std_E b'])

    The same ``b_uv`` objects decide the support (through ``gate_mlp``) and are
    the messages inside the support (through ``message_mlp``), so the binding
    states both *form* and *compute within* the learned structure.

    Degeneration to the connectivity-free bag (B-bag)
    -------------------------------------------------
    ``bond_mlp(bond_embed)`` is exactly the B-bag bond primitive map and the
    node / fusion maps are the B-bag maps.  The three perturbation paths
    (``bind_delta``, ``update_mlp``, ``bind_update``) are initialised with tiny
    final-layer weights and the gate is initialised open
    (``sigmoid(gate_bias) ~ 0.95``); with all gates on and the perturbations
    near zero the cell reduces to the B-bag computation.  The perturbations are
    deliberately *not* exactly zero so that gradients reach the gate, binding,
    message and update modules from the first step.

    Permutation invariance
    ----------------------
    Every aggregation is a mean / second moment over a multiset (nodes in the
    support, real undirected bonds in the induced support) and the root term is
    the shared node map at the unique root.  The support is built only from
    root-relative roles (root distance) and exchangeable binding states, so a
    relabelling of the patch nodes yields the same ``e_asb`` and the same
    support.  The induced support keeps **all** real bonds among selected
    nodes, including bonds that were not used to grow the support (e.g. rings).

    ``struct_src`` / ``struct_dst`` are the only adjacency used; there is no
    vocabulary, no certificate, no motif enumeration and no per-patch table.
    """

    def __init__(
        self,
        *,
        atom_categories: int = 28,
        bond_categories: int = 4,
        n_distance_bins: int = 3,
        node_dim: int = 48,
        edge_dim: int = 24,
        bind_dim: int | None = None,
        node_hidden: int = 96,
        bond_hidden: int = 48,
        bind_hidden: int = 32,
        gate_hidden: int = 24,
        message_hidden: int = 32,
        update_hidden: int = 32,
        bind_update_hidden: int = 32,
        fusion_hidden: int = 104,
        output_dim: int = 16,
        perturb_init: float = 1.0e-3,
        gate_bias_init: float = 3.0,
        gate_weight_init_std: float = 1.0e-2,
    ) -> None:
        super().__init__()
        self.atom_categories = int(atom_categories)
        self.bond_categories = int(bond_categories)
        self.n_distance_bins = int(n_distance_bins)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.bind_dim = int(edge_dim if bind_dim is None else bind_dim)
        self.node_hidden = int(node_hidden)
        self.bond_hidden = int(bond_hidden)
        self.bind_hidden = int(bind_hidden)
        self.gate_hidden = int(gate_hidden)
        self.message_hidden = int(message_hidden)
        self.update_hidden = int(update_hidden)
        self.bind_update_hidden = int(bind_update_hidden)
        self.fusion_hidden = int(fusion_hidden)
        self.output_dim = int(output_dim)
        self.perturb_init = float(perturb_init)
        self.gate_bias_init = float(gate_bias_init)
        self.gate_weight_init_std = float(gate_weight_init_std)
        # Interface-compatible audit metadata (see SharedBagPatchEncoder).
        self.hidden_dim = int(fusion_hidden)
        self.rounds = 1
        self.include_std_pool = True
        self.kind = "adaptive_structure_binding"
        if min(
            self.atom_categories,
            self.bond_categories,
            self.n_distance_bins,
            self.node_dim,
            self.edge_dim,
            self.bind_dim,
            self.node_hidden,
            self.fusion_hidden,
            self.output_dim,
        ) < 1:
            raise ValueError("ASB encoder dimensions must be positive")

        # --- B-bag-compatible attribute primitive maps ----------------------
        self.atom_embedding = nn.Embedding(self.atom_categories, self.node_dim)
        self.root_embedding = nn.Embedding(2, self.node_dim)
        self.distance_embedding = nn.Embedding(
            self.n_distance_bins, self.node_dim
        )
        self.bond_embedding = nn.Embedding(self.bond_categories, self.edge_dim)
        self.node_mlp = nn.Sequential(
            nn.Linear(self.node_dim, self.node_hidden),
            nn.ReLU(),
            nn.Linear(self.node_hidden, self.node_dim),
        )
        # The bond primitive map is shape-compatible with the B-bag bond map so
        # the attribute-pretrained initialisation is an exact tensor copy.
        self.bond_mlp = nn.Sequential(
            nn.Linear(self.edge_dim, self.bond_hidden),
            nn.ReLU(),
            nn.Linear(self.bond_hidden, self.bind_dim),
        )

        # --- binding state (structure + computation) ------------------------
        # b = bond_mlp(bond_embed) + bind_delta([z_u+z_v, |z_u-z_v|, bond_embed])
        self.bind_delta_mlp = nn.Sequential(
            nn.Linear(2 * self.node_dim + self.edge_dim, self.bind_hidden),
            nn.ReLU(),
            nn.Linear(self.bind_hidden, self.bind_dim),
        )
        # one shared gate scores every real bond; the same b_uv is the message.
        self.gate_mlp = nn.Sequential(
            nn.Linear(self.bind_dim, self.gate_hidden),
            nn.ReLU(),
            nn.Linear(self.gate_hidden, 1),
        )
        self.message_mlp = nn.Sequential(
            nn.Linear(self.bind_dim, self.message_hidden),
            nn.ReLU(),
            nn.Linear(self.message_hidden, self.node_dim),
        )
        self.update_mlp = nn.Sequential(
            nn.Linear(2 * self.node_dim, self.update_hidden),
            nn.ReLU(),
            nn.Linear(self.update_hidden, self.node_dim),
        )
        self.bind_update = nn.Sequential(
            nn.Linear(2 * self.node_dim + self.edge_dim, self.bind_update_hidden),
            nn.ReLU(),
            nn.Linear(self.bind_update_hidden, self.bind_dim),
        )
        fusion_input = 3 * self.node_dim + 2 * self.bind_dim
        self.fusion = nn.Sequential(
            nn.Linear(fusion_input, self.fusion_hidden),
            nn.ReLU(),
            nn.Linear(self.fusion_hidden, self.output_dim),
        )

        # --- near-function-preserving initialisation ------------------------
        for module in (self.bind_delta_mlp, self.update_mlp, self.bind_update):
            nn.init.normal_(module[-1].weight, mean=0.0, std=self.perturb_init)
            nn.init.zeros_(module[-1].bias)
        nn.init.normal_(
            self.gate_mlp[-1].weight, mean=0.0, std=self.gate_weight_init_std
        )
        nn.init.constant_(self.gate_mlp[-1].bias, self.gate_bias_init)

        # Diagnostics (never parameters / buffers).  Sufficient statistics are
        # refreshed by every forward so support / gate / binding health can be
        # audited without a second pass.  ``record_support`` additionally keeps
        # the explicit per-patch selected support for correctness tests; it is
        # off by default so training runs never accumulate it.
        self.last_stats: dict[str, Any] = {}
        self.record_support = False
        self.last_support: dict[str, Any] = {}

    # -- near-function-preserving initialisation from B-bag ------------------
    def load_attribute_pretrained(self, bbag: "SharedBagPatchEncoder") -> None:
        """Copy the B-bag attribute primitive maps into this cell.

        Copies ``atom / root / distance / bond`` embeddings, the node map, the
        bond primitive map and the fusion.  After this call the ASB cell reads
        the same attribute primitives as B-bag and, with open gates and tiny
        perturbation paths, reproduces the B-bag patch token to first order.
        """
        for name in (
            "atom_embedding",
            "root_embedding",
            "distance_embedding",
            "bond_embedding",
            "node_mlp",
            "bond_mlp",
            "fusion",
        ):
            target = getattr(self, name)
            source = getattr(bbag, name)
            target.load_state_dict(source.state_dict())

    # -- helpers -------------------------------------------------------------
    @staticmethod
    def _group_mean(
        values: torch.Tensor,
        index: torch.Tensor,
        weight: torch.Tensor,
        n_groups: int,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Weighted group mean and population std (weights in ``{0, 1}``)."""
        dim = int(values.shape[1])
        total = values.new_zeros((n_groups, dim))
        if values.numel():
            total.index_add_(0, index, values * weight.unsqueeze(1))
        count = values.new_zeros(n_groups)
        if weight.numel():
            count.index_add_(0, index, weight)
        denom = count.clamp_min(1.0).unsqueeze(1)
        mean = total / denom
        second = values.new_zeros((n_groups, dim))
        if values.numel():
            second.index_add_(0, index, values * values * weight.unsqueeze(1))
        variance = (second / denom - mean * mean).clamp_min(0.0)
        return mean, torch.sqrt(variance + 1.0e-8)

    def forward(self, data: object) -> torch.Tensor:
        atom = data.struct_atom.long()
        root = data.struct_root.long()
        distance = data.struct_dist.long()
        unique_patch, node_patch = torch.unique(
            data.struct_patch.long(), return_inverse=True
        )
        n_patches = int(unique_patch.numel())
        n_nodes = int(atom.shape[0])

        # --- node states ----------------------------------------------------
        z = self.node_mlp(
            self.atom_embedding(atom)
            + self.root_embedding(root)
            + self.distance_embedding(distance)
        )

        # --- real undirected bonds (each bond is stored twice) --------------
        src = data.struct_src.long()
        dst = data.struct_dst.long()
        bond = data.struct_bond.long()
        keep = src < dst
        eu = src[keep]
        ev = dst[keep]
        ebond = bond[keep]
        edge_group = node_patch[eu]
        n_edges = int(eu.numel())
        if n_edges:
            bond_emb = self.bond_embedding(ebond)
            base_b = self.bond_mlp(bond_emb)
            delta_in = torch.cat(
                [z[eu] + z[ev], (z[eu] - z[ev]).abs(), bond_emb], dim=1
            )
            b = base_b + self.bind_delta_mlp(delta_in)
            gate = torch.sigmoid(self.gate_mlp(b)).view(-1)
        else:
            b = z.new_zeros((0, self.bind_dim))
            gate = z.new_zeros(0)

        # --- adaptive support (root-relative, connectivity-guaranteed) ------
        level1_soft = z.new_zeros(n_nodes)
        level2_soft = z.new_zeros(n_nodes)
        if n_edges:
            dist_u = distance[eu]
            dist_v = distance[ev]
            # root -> distance-1 bonds: the non-root endpoint is the child.
            root_edge = (dist_u == 0) | (dist_v == 0)
            re_idx = torch.nonzero(root_edge, as_tuple=False).view(-1)
            if re_idx.numel():
                children = torch.where(
                    dist_u[re_idx] == 0, ev[re_idx], eu[re_idx]
                )
                level1_soft = level1_soft.index_put(
                    (children,), gate[re_idx], accumulate=False
                )
            # distance-1 -> distance-2 bonds: activation requires an active
            # parent, so the support can never touch a disconnected node.
            level1_hard = _straight_through_binary(level1_soft)
            cross = ((dist_u == 1) & (dist_v == 2)) | (
                (dist_u == 2) & (dist_v == 1)
            )
            cr_idx = torch.nonzero(cross, as_tuple=False).view(-1)
            if cr_idx.numel():
                parent_is_u = dist_u[cr_idx] == 1
                parents = torch.where(parent_is_u, eu[cr_idx], ev[cr_idx])
                child2 = torch.where(parent_is_u, ev[cr_idx], eu[cr_idx])
                path = level1_hard[parents] * gate[cr_idx]
                logs = torch.log1p(-path.clamp(max=1.0 - 1.0e-6))
                acc = z.new_zeros(n_nodes).index_add(0, child2, logs)
                level2_soft = 1.0 - torch.exp(acc)

        ones = torch.ones_like(level1_soft)
        a_soft = torch.where(
            distance == 0,
            ones,
            torch.where(distance == 1, level1_soft, level2_soft),
        )
        selected = _straight_through_binary(a_soft)

        # --- binding-aware computation on the learned support ---------------
        if n_edges:
            selected_edge = selected[eu] * selected[ev]
            message = self.message_mlp(b)
            message = message * selected_edge.unsqueeze(1)
            incident_sum = z.new_zeros((n_nodes, self.node_dim))
            incident_sum = incident_sum.index_add(0, eu, message)
            incident_sum = incident_sum.index_add(0, ev, message)
            incident_count = z.new_zeros(n_nodes)
            incident_count = incident_count.index_add(0, eu, selected_edge)
            incident_count = incident_count.index_add(0, ev, selected_edge)
            m = incident_sum / incident_count.clamp_min(1.0).unsqueeze(1)
            z2 = z + self.update_mlp(torch.cat([z, m], dim=1))
            update_in = torch.cat(
                [z2[eu] + z2[ev], (z2[eu] - z2[ev]).abs(), bond_emb], dim=1
            )
            b2 = b + self.bind_update(update_in)
        else:
            selected_edge = z.new_zeros(0)
            m = z.new_zeros((n_nodes, self.node_dim))
            z2 = z + self.update_mlp(torch.cat([z, m], dim=1))
            b2 = b

        # --- permutation-invariant pooling over the learned structure -------
        is_root = root > 0
        root_state = z2.new_zeros((n_patches, self.node_dim))
        if bool(is_root.any()):
            root_state = root_state.index_put(
                (node_patch[is_root],), z2[is_root], accumulate=False
            )
        node_mean, node_std = self._group_mean(
            z2, node_patch, selected, n_patches
        )
        if n_edges:
            bind_mean, bind_std = self._group_mean(
                b2, edge_group, selected_edge, n_patches
            )
        else:
            bind_mean = z.new_zeros((n_patches, self.bind_dim))
            bind_std = z.new_zeros((n_patches, self.bind_dim))

        output = self.fusion(
            torch.cat(
                [root_state, node_mean, node_std, bind_mean, bind_std], dim=1
            )
        )

        self._record_stats(
            n_patches=n_patches,
            distance=distance,
            selected=selected,
            node_patch=node_patch,
            selected_edge=selected_edge if n_edges else None,
            n_edges=n_edges,
            gate=gate if n_edges else None,
            node_state=z2,
            binding_state=b2,
            message=m,
            update=z2 - z,
        )
        if self.record_support:
            self.last_support = {
                "selected": selected.detach().cpu(),
                "distance": distance.detach().cpu(),
                "root": root.detach().cpu(),
                "node_patch": node_patch.detach().cpu(),
                "src": eu.detach().cpu(),
                "dst": ev.detach().cpu(),
                "selected_edge": (
                    selected_edge.detach().cpu()
                    if n_edges
                    else torch.zeros(0)
                ),
                "n_patches": int(n_patches),
                "output": output.detach().cpu(),
            }
        return output

    # -- diagnostics ---------------------------------------------------------
    def _record_stats(self, **kw: Any) -> None:
        n_patches = int(kw["n_patches"])
        distance = kw["distance"]
        selected = kw["selected"]
        node_patch = kw["node_patch"]
        selected_edge = kw["selected_edge"]
        n_edges = int(kw["n_edges"])
        gate = kw["gate"]

        dist1 = distance == 1
        dist2 = distance == 2
        patch_nodes = torch.bincount(
            node_patch, minlength=n_patches
        ).to(torch.float64)
        support_sizes = node_patch.new_zeros(n_patches, dtype=torch.float64)
        if selected.numel():
            support_sizes.index_add_(0, node_patch, selected.to(torch.float64))
        gate_f = gate.double() if gate is not None and gate.numel() else None
        stats: dict[str, Any] = {
            "n_patches": int(n_patches),
            "n_nodes": int(selected.numel()),
            "n_selected_nodes": float(selected.sum()) if selected.numel() else 0.0,
            "n_dist1": int(dist1.sum()),
            "n_dist1_selected": float(selected[dist1].sum()) if dist1.any() else 0.0,
            "n_dist2": int(dist2.sum()),
            "n_dist2_selected": float(selected[dist2].sum()) if dist2.any() else 0.0,
            "n_edges": int(n_edges),
            "n_selected_edges": (
                float(selected_edge.sum()) if selected_edge is not None else 0.0
            ),
            "gate_sum": float(gate_f.sum()) if gate_f is not None else 0.0,
            "gate_sq_sum": (
                float((gate_f * gate_f).sum()) if gate_f is not None else 0.0
            ),
            "gate_count": int(gate.numel()) if gate is not None else 0,
            "gate_entropy_sum": (
                float(
                    (
                        -gate_f * torch.log(gate_f.clamp_min(1.0e-12))
                        - (1.0 - gate_f)
                        * torch.log((1.0 - gate_f).clamp_min(1.0e-12))
                    ).sum()
                )
                if gate_f is not None
                else 0.0
            ),
            "node_state_norm_sum": float(kw["node_state"].norm(dim=1).sum()),
            "binding_state_norm_sum": (
                float(kw["binding_state"].norm(dim=1).sum())
                if kw["binding_state"].numel()
                else 0.0
            ),
            "message_norm_sum": float(kw["message"].norm(dim=1).sum()),
            "update_norm_sum": float(kw["update"].norm(dim=1).sum()),
            "support_sizes": support_sizes.detach().cpu(),
            "patch_node_counts": patch_nodes.detach().cpu(),
        }
        self.last_stats = stats

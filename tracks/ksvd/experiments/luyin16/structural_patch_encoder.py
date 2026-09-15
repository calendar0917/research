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

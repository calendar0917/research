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

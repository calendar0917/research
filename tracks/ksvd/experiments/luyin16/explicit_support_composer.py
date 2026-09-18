"""Explicit-support structural composer for rooted typed radius-2 patches.

Motivation
----------
The completed B-full encoder (``structural_patch_encoder.SharedStructuralPatchEncoder``)
reads the real rooted typed patch graph and produces ``e_struct in R^16`` through
two rounds of edge-aware message passing.  A zero-training audit showed that its
16-D output is effectively rank-1 (effective rank ~1.1-1.2, rank-1
reconstruction of the 2-seed soup changes MAE by ~1.6e-5), so the gain is a
small shared near-bias function, not a rich structural manifold.  The B-bag
control then showed that removing *all* adjacency costs ~0.0013 soup MAE.

This module tests a different representation hypothesis:

    Do not encode the whole patch implicitly through GNN hidden-state
    propagation.  Instead maintain **explicit structural objects** with exact
    atom / bond supports and learn only *which legal local objects are worth
    composing*.

Form (exactly one pre-registered architecture; no attention, no Transformer, no
learned token dictionary, no soft clustering)::

    round 0 : atom primitives            support = {atom},  bonds = {}
    round 1 : legal atom+atom compositions (real bonds)
                                         support = 2 atoms,   bonds = {bond}
    round 2 : legal object+object compositions
              (overlap or a real connecting bond, union <= 4 atoms)
                                         support = exact parent union
    pooling : activity-weighted mean / std over the surviving objects
              + log object count -> fusion -> e_struct in R^16

Core principle
--------------
**Topology defines which compositions are legal; neural parameters learn which
legal compositions are useful.**  Supports are discrete, exact and
deterministic; the only learned quantities are the scalar activities
``a_k = sigmoid(score_k)`` and the composition content
``z_new = a_k * f(z_a, z_b, explicit_relation)``.  No node hidden state is
propagated along edges, so the "no message passing" claim is structural: the
composer never reads ``struct_src`` / ``struct_dst``.

Preprocessing contract
----------------------
The candidate topology is target-free and precomputed once per split
(``build_explicit_candidate_graph``).  The module consumes per-molecule flat
tensors attached to the PyG ``Data`` object:

``struct_atom``    int64 ``[N]``  atom category per patch-node
``struct_root``    int64 ``[N]``  1 for the root, else 0
``struct_dist``    int64 ``[N]``  BFS distance from the root
``struct_degree``  int64 ``[N]``  degree within the patch
``struct_patch``   int64 ``[N]``  patch index owning each node
``exo_l1_u``       int64 ``[B]``  level-0 (atom) id of the first endpoint
``exo_l1_v``       int64 ``[B]``  level-0 (atom) id of the second endpoint
``exo_l1_bond``    int64 ``[B]``  bond category
``exo_l1_patch``   int64 ``[B]``  patch index owning each bond object
``exo_l2_a``       int64 ``[C]``  level-1 object id of parent a
``exo_l2_b``       int64 ``[C]``  level-1 object id of parent b
``exo_l2_patch``   int64 ``[C]``  patch index owning each level-2 object
``exo_l2_overlap`` int64 ``[C]``  ``|S_a^V inter S_b^V|``
``exo_l2_conn``    int64 ``[C,K]`` level-1 ids of the NEW connecting bonds
                                   (parents' own bonds excluded), ``-1`` padding
``struct_n_patches`` int          total patches in the batch

The candidate tables never contain a target and never depend on the model.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn


# ---------------------------------------------------------------------------
# target-free topological candidate enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplicitCandidateGraph:
    """Exact, deterministic composition search space of one molecule."""

    patch: np.ndarray  # [N] patch id of every atom (level-0 object)
    degree: np.ndarray  # [N] degree within the owning patch
    l1_u: np.ndarray  # [B] level-0 ids (global patch-node ids)
    l1_v: np.ndarray  # [B]
    l1_bond: np.ndarray  # [B]
    l1_patch: np.ndarray  # [B]
    l2_a: np.ndarray  # [C] level-1 local ids
    l2_b: np.ndarray  # [C]
    l2_patch: np.ndarray  # [C]
    l2_overlap: np.ndarray  # [C]
    l2_conn: np.ndarray  # [C, K] level-1 local ids of new connecting bonds, -1 pad
    n_atoms: int
    n_level1: int
    n_level2: int
    n_patches: int


def build_explicit_candidate_graph(
    patch: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    bond: np.ndarray,
    edge_patch: np.ndarray,
    n_patches: int,
    *,
    max_union_size: int = 4,
    max_conn: int = 4,
) -> ExplicitCandidateGraph:
    """Enumerate the legal composition candidates of one molecule.

    ``patch``/``src``/``dst``/``bond``/``edge_patch`` follow the
    ``sspe.PatchGraph`` convention: ``patch[i]`` is the patch owning patch-node
    ``i``; ``src[e]`` / ``dst[e]`` are **patch-local** node indices (relative to
    the owning patch) and each undirected bond appears once per direction.

    Supports are explicit sets of atoms; bonds are explicit.  The returned
    tables are exact and can be replayed to recover every atom set, every bond
    set and the full provenance DAG.
    """
    patch = np.asarray(patch, dtype=np.int64)
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    bond = np.asarray(bond, dtype=np.int64)
    edge_patch = np.asarray(edge_patch, dtype=np.int64)
    n_patches = int(n_patches)
    n_atoms = int(patch.shape[0])
    if src.shape[0] != dst.shape[0] or src.shape[0] != bond.shape[0]:
        raise ValueError("edge arrays must have identical length")
    if edge_patch.shape[0] != src.shape[0]:
        raise ValueError("edge_patch must match the edge arrays")

    node_counts = np.bincount(patch, minlength=n_patches).astype(np.int64)
    node_start = np.concatenate([[0], np.cumsum(node_counts)[:-1]]).astype(np.int64)

    # -- undirected bonds per patch (deterministic, sorted by (min, max)) -----
    patch_bonds: list[dict[tuple[int, int], int]] = [
        {} for _ in range(n_patches)
    ]
    for edge_index in range(int(src.shape[0])):
        p = int(edge_patch[edge_index])
        left = int(src[edge_index])
        right = int(dst[edge_index])
        key = (min(left, right), max(left, right))
        value = int(bond[edge_index])
        previous = patch_bonds[p].get(key)
        if previous is not None and previous != value:
            raise ValueError("conflicting bond types for one undirected bond")
        patch_bonds[p][key] = value

    degree = np.zeros(n_atoms, dtype=np.int64)
    l1_entries: list[tuple[int, int, int, int]] = []  # (patch, u_local, v_local, bond)
    for p in range(n_patches):
        for key in sorted(patch_bonds[p]):
            u, v = key
            degree[node_start[p] + u] += 1
            degree[node_start[p] + v] += 1
            l1_entries.append((p, int(u), int(v), int(patch_bonds[p][key])))

    n_level1 = len(l1_entries)
    l1_u = np.asarray(
        [node_start[p] + u for (p, u, _v, _t) in l1_entries], dtype=np.int64
    )
    l1_v = np.asarray(
        [node_start[p] + v for (p, _u, v, _t) in l1_entries], dtype=np.int64
    )
    l1_bond = np.asarray([t for (_p, _u, _v, t) in l1_entries], dtype=np.int64)
    l1_patch = np.asarray([p for (p, _u, _v, _t) in l1_entries], dtype=np.int64)

    # local level-1 index of each (patch, min, max)
    bond_index: dict[tuple[int, int, int], int] = {}
    for index, (p, u, v, _t) in enumerate(l1_entries):
        bond_index[(p, u, v)] = index

    # -- level-2 candidates: legal pairs of level-1 objects -------------------
    l2_a: list[int] = []
    l2_b: list[int] = []
    l2_patch: list[int] = []
    l2_overlap: list[int] = []
    l2_conn: list[list[int]] = []
    seen_union: dict[tuple[int, frozenset[int]], None] = {}
    for p in range(n_patches):
        entries = [
            (index, u, v)
            for index, (q, u, v, _t) in enumerate(l1_entries)
            if q == p
        ]
        for i in range(len(entries)):
            index_i, u_i, v_i = entries[i]
            support_i = (node_start[p] + u_i, node_start[p] + v_i)
            set_i = {support_i[0], support_i[1]}
            own_i = (p, min(u_i, v_i), max(u_i, v_i))
            for j in range(i + 1, len(entries)):
                index_j, u_j, v_j = entries[j]
                support_j = (node_start[p] + u_j, node_start[p] + v_j)
                set_j = {support_j[0], support_j[1]}
                union = set_i | set_j
                if len(union) > int(max_union_size):
                    continue
                overlap = len(set_i & set_j)
                own_j = (p, min(u_j, v_j), max(u_j, v_j))
                connecting: list[int] = []
                for key in sorted(patch_bonds[p]):
                    if key == own_i or key == own_j:
                        continue
                    left = node_start[p] + key[0]
                    right = node_start[p] + key[1]
                    if (left in set_i and right in set_j) or (
                        right in set_i and left in set_j
                    ):
                        connecting.append(int(bond_index[(p, key[0], key[1])]))
                if overlap == 0 and not connecting:
                    continue
                union_key = (p, frozenset(union))
                if union_key in seen_union:
                    # deterministic de-duplication: first (smallest ordered
                    # pair) candidate with this exact support wins.
                    continue
                seen_union[union_key] = None
                l2_a.append(int(index_i))
                l2_b.append(int(index_j))
                l2_patch.append(int(p))
                l2_overlap.append(int(overlap))
                if len(connecting) > int(max_conn):
                    connecting = sorted(connecting)[: int(max_conn)]
                l2_conn.append(sorted(connecting))

    n_level2 = len(l2_a)
    conn = np.full((n_level2, int(max_conn)), -1, dtype=np.int64)
    for index, values in enumerate(l2_conn):
        for slot, value in enumerate(values):
            conn[index, slot] = int(value)

    return ExplicitCandidateGraph(
        patch=patch.astype(np.int64),
        degree=degree,
        l1_u=l1_u,
        l1_v=l1_v,
        l1_bond=l1_bond,
        l1_patch=l1_patch,
        l2_a=np.asarray(l2_a, dtype=np.int64),
        l2_b=np.asarray(l2_b, dtype=np.int64),
        l2_patch=np.asarray(l2_patch, dtype=np.int64),
        l2_overlap=np.asarray(l2_overlap, dtype=np.int64),
        l2_conn=conn,
        n_atoms=n_atoms,
        n_level1=n_level1,
        n_level2=n_level2,
        n_patches=n_patches,
    )


def expand_atom_support(
    graph: ExplicitCandidateGraph, level: int, index: int
) -> frozenset[int]:
    """Exact atom support of one object (recursive provenance replay)."""
    if level == 0:
        return frozenset({int(index)})
    if level == 1:
        return frozenset({int(graph.l1_u[index]), int(graph.l1_v[index])})
    if level == 2:
        left = expand_atom_support(graph, 1, int(graph.l2_a[index]))
        right = expand_atom_support(graph, 1, int(graph.l2_b[index]))
        return left | right
    raise ValueError(f"unknown object level {level}")


def expand_bond_support(
    graph: ExplicitCandidateGraph, level: int, index: int
) -> frozenset[int]:
    """Exact bond support (level-1 object ids) of one object."""
    if level == 0:
        return frozenset()
    if level == 1:
        return frozenset({int(index)})
    if level == 2:
        left = expand_bond_support(graph, 1, int(graph.l2_a[index]))
        right = expand_bond_support(graph, 1, int(graph.l2_b[index]))
        connecting = {
            int(value) for value in graph.l2_conn[index].tolist() if value >= 0
        }
        return left | right | connecting
    raise ValueError(f"unknown object level {level}")


def parents(graph: ExplicitCandidateGraph, level: int, index: int):
    """Provenance parents ``(level, index)`` of one object."""
    if level == 0:
        return ()
    if level == 1:
        return ((0, int(graph.l1_u[index])), (0, int(graph.l1_v[index])))
    if level == 2:
        return ((1, int(graph.l2_a[index])), (1, int(graph.l2_b[index])))
    raise ValueError(f"unknown object level {level}")


# ---------------------------------------------------------------------------
# module
# ---------------------------------------------------------------------------


def scatter_sum(
    source: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    """Permutation-invariant sum of ``source`` rows into ``dim_size`` groups."""
    dim_size = int(dim_size)
    if source.numel() == 0 or dim_size <= 0:
        return source.new_zeros((max(dim_size, 0), int(source.shape[1])))
    total = source.new_zeros((dim_size, int(source.shape[1])))
    total.index_add_(0, index.long(), source)
    return total


def scatter_mean(
    source: torch.Tensor, index: torch.Tensor, dim_size: int
) -> torch.Tensor:
    dim_size = int(dim_size)
    if source.numel() == 0 or dim_size <= 0:
        return source.new_zeros((max(dim_size, 0), int(source.shape[1])))
    total = scatter_sum(source, index, dim_size)
    counts = (
        torch.bincount(index.long(), minlength=dim_size)
        .clamp_min(1)
        .to(source.dtype)
        .unsqueeze(1)
    )
    return total / counts


class ExplicitSupportComposer(nn.Module):
    """Two-round explicit-support object composer (one pre-registered form).

    Object latent width ``latent_dim`` is deliberately small (4-8): the earlier
    rank audit showed the structural manifold is near rank-1, so the parameter
    budget lives in the composition scorer / content MLPs and the object fusion,
    not in a wide hidden state.  Supports are *not* latents: they live in the
    precomputed candidate tables and are untouched by the network.
    """

    def __init__(
        self,
        *,
        atom_categories: int = 28,
        bond_categories: int = 4,
        n_distance_bins: int = 3,
        n_degree_bins: int = 6,
        latent_dim: int = 8,
        edge_dim: int = 8,
        candidate_hidden: int = 192,
        fusion_hidden: int = 208,
        output_dim: int = 16,
        max_conn: int = 4,
    ) -> None:
        super().__init__()
        self.atom_categories = int(atom_categories)
        self.bond_categories = int(bond_categories)
        self.n_distance_bins = int(n_distance_bins)
        self.n_degree_bins = int(n_degree_bins)
        self.latent_dim = int(latent_dim)
        self.edge_dim = int(edge_dim)
        self.candidate_hidden = int(candidate_hidden)
        self.fusion_hidden = int(fusion_hidden)
        self.output_dim = int(output_dim)
        self.max_conn = int(max_conn)
        # Interface metadata consumed by the shared parameter audit; the
        # composer performs exactly two composition rounds and uses a
        # mean/std object pooling.
        self.node_dim = self.latent_dim
        self.hidden_dim = self.candidate_hidden
        self.rounds = 2
        self.include_std_pool = True
        self.kind = "explicit_support_composer"
        if min(
            self.atom_categories,
            self.bond_categories,
            self.n_distance_bins,
            self.n_degree_bins,
            self.latent_dim,
            self.edge_dim,
            self.candidate_hidden,
            self.fusion_hidden,
            self.output_dim,
            self.max_conn,
        ) < 1:
            raise ValueError("composer dimensions must be positive")

        self.atom_embedding = nn.Embedding(self.atom_categories, self.latent_dim)
        self.root_embedding = nn.Embedding(2, self.latent_dim)
        self.distance_embedding = nn.Embedding(
            self.n_distance_bins, self.latent_dim
        )
        self.degree_embedding = nn.Embedding(self.n_degree_bins, self.latent_dim)
        self.bond_embedding = nn.Embedding(self.bond_categories, self.edge_dim)

        # Level-1 relation feature (symmetric in the two endpoints):
        #   [z_u + z_v ; |z_u - z_v| ; bond ; size_u + size_v ;
        #    |size_u - size_v| ; root_u + root_v ; |root_u - root_v|]
        l1_input = 2 * self.latent_dim + self.edge_dim + 4
        self.level1_input = int(l1_input)
        self.level1_score = nn.Sequential(
            nn.Linear(l1_input, self.candidate_hidden),
            nn.ReLU(),
            nn.Linear(self.candidate_hidden, 1),
        )
        self.level1_content = nn.Sequential(
            nn.Linear(l1_input, self.candidate_hidden),
            nn.ReLU(),
            nn.Linear(self.candidate_hidden, self.latent_dim),
        )

        # Level-2 relation feature (symmetric in the two parents):
        #   [z_a + z_b ; |z_a - z_b| ; overlap ; n_new_conn ;
        #    sum_new_conn_bond_emb ; union_size ; root_a + root_b ;
        #    |root_a - root_b| ; a_a + a_b ; |a_a - a_b|]
        l2_input = 2 * self.latent_dim + self.edge_dim + 7
        self.level2_input = int(l2_input)
        self.level2_score = nn.Sequential(
            nn.Linear(l2_input, self.candidate_hidden),
            nn.ReLU(),
            nn.Linear(self.candidate_hidden, 1),
        )
        self.level2_content = nn.Sequential(
            nn.Linear(l2_input, self.candidate_hidden),
            nn.ReLU(),
            nn.Linear(self.candidate_hidden, self.latent_dim),
        )

        self.fusion = nn.Sequential(
            nn.Linear(2 * self.latent_dim + 1, self.fusion_hidden),
            nn.ReLU(),
            nn.Linear(self.fusion_hidden, self.output_dim),
        )

    # -- forward ---------------------------------------------------------------
    def _n_patches(self, data: Any, all_patch: torch.Tensor) -> int:
        value = getattr(data, "struct_n_patches", None)
        if value is not None:
            return int(value)
        if all_patch.numel() == 0:
            return 0
        return int(all_patch.max().item()) + 1

    def forward(self, data: Any, return_details: bool = False):
        atom = data.struct_atom.long()
        root = data.struct_root.to(torch.float32)
        distance = data.struct_dist.long()
        degree = data.struct_degree.long().clamp_max(self.n_degree_bins - 1)
        n_atoms = int(atom.numel())

        z0 = (
            self.atom_embedding(atom)
            + self.root_embedding((root > 0).long())
            + self.distance_embedding(distance)
            + self.degree_embedding(degree)
        )
        activity0 = z0.new_ones(n_atoms)
        size0 = z0.new_ones(n_atoms)

        # -- round 1: atom + atom, legal only along a real bond ----------------
        l1_u = data.exo_l1_u.long()
        l1_v = data.exo_l1_v.long()
        l1_bond = data.exo_l1_bond.long()
        n_level1 = int(l1_u.numel())
        if n_level1:
            bond_edge = self.bond_embedding(l1_bond)
            feature1 = torch.cat(
                [
                    z0[l1_u] + z0[l1_v],
                    torch.abs(z0[l1_u] - z0[l1_v]),
                    bond_edge,
                    (size0[l1_u] + size0[l1_v]).unsqueeze(1),
                    torch.abs(size0[l1_u] - size0[l1_v]).unsqueeze(1),
                    (root[l1_u] + root[l1_v]).unsqueeze(1),
                    torch.abs(root[l1_u] - root[l1_v]).unsqueeze(1),
                ],
                dim=1,
            )
            activity1 = torch.sigmoid(self.level1_score(feature1).squeeze(-1))
            z1 = activity1.unsqueeze(1) * self.level1_content(feature1)
            root1 = torch.clamp(root[l1_u] + root[l1_v], max=1.0)
            size1 = z0.new_full((n_level1,), 2.0)
        else:
            z1 = z0.new_zeros((0, self.latent_dim))
            activity1 = z0.new_zeros((0,))
            root1 = z0.new_zeros((0,))
            size1 = z0.new_zeros((0,))

        # -- round 2: level-1 + level-1, legal by overlap or a real new bond ---
        l2_a = data.exo_l2_a.long()
        l2_b = data.exo_l2_b.long()
        l2_overlap = data.exo_l2_overlap.to(torch.float32)
        l2_conn = data.exo_l2_conn.long()
        n_level2 = int(l2_a.numel())
        if n_level2:
            conn_mask = l2_conn >= 0
            conn_index = l2_conn.clamp_min(0)
            conn_edge = self.bond_embedding(
                l1_bond[conn_index] if n_level1 else conn_index
            )
            conn_edge = conn_edge * conn_mask.unsqueeze(-1).to(conn_edge.dtype)
            conn_sum = conn_edge.sum(dim=1)
            n_conn = conn_mask.sum(dim=1).to(z0.dtype)
            union_size = size1[l2_a] + size1[l2_b] - l2_overlap
            feature2 = torch.cat(
                [
                    z1[l2_a] + z1[l2_b],
                    torch.abs(z1[l2_a] - z1[l2_b]),
                    l2_overlap.unsqueeze(1),
                    n_conn.unsqueeze(1),
                    conn_sum,
                    union_size.unsqueeze(1),
                    (root1[l2_a] + root1[l2_b]).unsqueeze(1),
                    torch.abs(root1[l2_a] - root1[l2_b]).unsqueeze(1),
                    (activity1[l2_a] + activity1[l2_b]).unsqueeze(1),
                    torch.abs(activity1[l2_a] - activity1[l2_b]).unsqueeze(1),
                ],
                dim=1,
            )
            activity2 = torch.sigmoid(self.level2_score(feature2).squeeze(-1))
            z2 = activity2.unsqueeze(1) * self.level2_content(feature2)
        else:
            z2 = z0.new_zeros((0, self.latent_dim))
            activity2 = z0.new_zeros((0,))

        # -- per-patch object selection: prefer round-2, else round-1, else round-0
        patch0 = data.struct_patch.long()
        patch1 = data.exo_l1_patch.long()
        patch2 = data.exo_l2_patch.long()
        all_patch = torch.cat([patch0, patch1, patch2], dim=0)
        all_level = torch.cat(
            [
                torch.zeros(n_atoms, dtype=torch.long, device=z0.device),
                torch.ones(n_level1, dtype=torch.long, device=z0.device),
                torch.full((n_level2,), 2, dtype=torch.long, device=z0.device),
            ],
            dim=0,
        )
        n_patches = self._n_patches(data, all_patch)
        if n_patches == 0:
            return z0.new_zeros((0, self.output_dim))

        ones = z0.new_ones((n_atoms + n_level1 + n_level2, 1))
        count_by_patch = scatter_sum(ones, all_patch, n_patches).squeeze(1)
        # per-patch counts at each level
        level_counts = []
        for level in (0, 1, 2):
            mask = (all_level == level).to(z0.dtype).unsqueeze(1)
            level_counts.append(
                scatter_sum(mask, all_patch, n_patches).squeeze(1)
            )
        selected_level = torch.where(
            level_counts[2] > 0,
            torch.full_like(level_counts[0], 2),
            torch.where(
                level_counts[1] > 0,
                torch.ones_like(level_counts[0]),
                torch.zeros_like(level_counts[0]),
            ),
        )
        selected = (all_level == selected_level[all_patch]).to(z0.dtype)

        all_z = torch.cat([z0, z1, z2], dim=0)
        all_activity = torch.cat([activity0, activity1, activity2], dim=0)
        weights = all_activity * selected

        weight_sum = scatter_sum(weights.unsqueeze(1), all_patch, n_patches)
        numerator = scatter_sum(
            weights.unsqueeze(1) * all_z, all_patch, n_patches
        )
        denominator = weight_sum.clamp_min(1.0e-6)
        mean = numerator / denominator
        uniform_mean = scatter_mean(
            all_z * selected.unsqueeze(1), all_patch, n_patches
        )
        mean = torch.where(weight_sum > 1.0e-6, mean, uniform_mean)

        diff = all_z - mean[all_patch]
        variance = scatter_sum(
            weights.unsqueeze(1) * diff * diff, all_patch, n_patches
        ) / denominator
        uniform_variance = scatter_mean(
            (diff * diff) * selected.unsqueeze(1), all_patch, n_patches
        )
        variance = torch.where(
            weight_sum > 1.0e-6, variance, uniform_variance
        )
        std = torch.sqrt(variance.clamp_min(0.0) + 1.0e-8)
        object_count = scatter_sum(
            selected.unsqueeze(1), all_patch, n_patches
        )
        summary = torch.cat(
            [mean, std, torch.log1p(object_count)], dim=1
        )
        output = self.fusion(summary)
        if not return_details:
            return output
        return {
            "e_struct": output,
            "z0": z0,
            "z1": z1,
            "z2": z2,
            "activity0": activity0,
            "activity1": activity1,
            "activity2": activity2,
            "all_patch": all_patch,
            "all_level": all_level,
            "selected_level": selected_level,
            "selected": selected,
            "object_count": object_count.squeeze(1),
            "level_counts": torch.stack(level_counts, dim=1),
        }

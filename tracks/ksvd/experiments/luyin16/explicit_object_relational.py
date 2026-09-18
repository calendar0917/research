"""Explicit structural objects with explicit relations and recurrent reasoning.

Motivation
----------
The previous experiment (``explicit_structural_basis.py``) enumerated an exact
deterministic B0/B1/B2 basis and scored each object by a shared signed scalar
valuation, then pooled immediately.  It matched the *connectivity-free* B-bag
but not B-full's hidden patch GNN: the order ablation showed B1 bond objects
dominate while the information they carry is local (bond endpoint primitives),
i.e. support-local valuation alone cannot propagate connectivity.  B-full's
16-D structural channel is effectively rank-1, so the missing ingredient is
not output dimensionality but *reasoning over relations between explicit
objects*.

This module tests exactly one pre-registered hypothesis:

    The objects are explicit.  Their relations are explicit.  Neural
    computation only refines states over those explicit relations.

It is deliberately **not** a message-passing GNN over hidden atoms: no hidden
atom state exists, and no parameter can create, delete or re-weight an object
or a relation.  Object existence and the relation graph are fixed by the
molecular graph before any parameter is evaluated.  The honest classification
is *neural relational reasoning over explicitly enumerated, support-traceable
structural objects and relations* -- not "structure discovery".

Object family (unchanged from the previous experiment)
-----------------------------------------------------
* B0 atom objects:             support ``{v}``
* B1 bond objects:             support ``{u, v}``,  bond ``{(u, v)}``
* B2 centred 3-atom objects:   centre ``v``, support ``{u, v, w}``,
  bonds ``{(v, u), (v, w)}`` (+ closure ``(u, w)`` when it is a real bond)

Relation graph
--------------
Two distinct objects inside the *same patch* get a relation edge iff one of
the following deterministic, support-derived conditions holds:

* ``bond_overlap``  their bond supports intersect;
* ``overlap``       their atom supports intersect (share an atom, incl.
  containment);
* ``adjacent``      their supports are disjoint but a real molecular bond
  connects an atom of one to an atom of the other.

All other pairs have **no** edge.  The graph is therefore support-local and
sparse (measured edge/object ratio ~7, far from complete).

Encoder (exactly ``T_obj = 2`` rounds, one architecture, no sweep)::

    h_i^0        = E_{order(i)}(explicit object i)               in R^8
    q_ij^t       = Q(h_i^t, h_j^t, r_ij)                         in R^8
    A_i^t        = fixed invariant stats of q over incident rels
    h_i^{t+1}    = h_i^t + U([h_i^t, A_i^t])     (U tied across rounds)
    s(P)         = fusion(late per-order pooling of h^2)
    e_struct(P)  = b + s(P) * v                                  (rank-1)

No attention, no softmax, no learned support / existence gate, no GRU/LSTM,
no certificate or token lookup.

Preprocessing contract
----------------------
The driver attaches two groups of flat tensors to each PyG ``Data``:

* the explicit basis tensors (``esb_*``) built by ``build_explicit_basis_graph``
  exactly as in the previous experiment;
* the explicit relation tensors (``eor_*``) built by
  ``build_object_relation_graph``: ``eor_edge_i``/``eor_edge_j`` object ids,
  ``eor_family`` relation family ids and ``eor_feat`` symmetric descriptors.

Object ids are ``[0, n_atoms)`` for B0, ``[n_atoms, n_atoms+n_bonds)`` for B1
and ``[n_atoms+n_bonds, n_obj)`` for B2.  Relation edges are unique undirected
pairs (``i < j``) listed once.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

from tracks.ksvd.experiments.luyin16.explicit_structural_basis import (
    ExplicitBasisGraph,
    build_explicit_basis_graph,
    expected_triple_count,
    object_count,
    triple_object_bond_support,
    triple_object_center,
    triple_object_support,
    atom_object_support,
    bond_object_support,
)

# ---------------------------------------------------------------------------
# frozen architecture constants (one architecture; no sweep)
# ---------------------------------------------------------------------------

OBJ_DIM = 8
REL_HIDDEN = 64
REL_LATENT_DIM = 16
Q_OBJ_DIM = 8
Q_HIDDEN = 64
N_FAMILIES = 3
T_OBJ = 2
PRIMITIVE_DIM = OBJ_DIM

# relation family ids
FAMILY_OVERLAP = 0
FAMILY_BOND_OVERLAP = 1
FAMILY_ADJACENT = 2
FAMILY_NAMES = ("overlap", "bond_overlap", "adjacent")

# relation descriptor layout (symmetric under endpoint swap):
#   order pair one-hot (6)
#   [atom_overlap, bond_overlap, containment, share_centre, share_root, adjacent] (6)
#   connecting bond type one-hot (4) + has_connecting (1)
#   min support distance bucket (1)
#   [support_size_min, support_size_max] (2)
#   relation family one-hot (3)
REL_ORDER_PAIR_DIM = 6
REL_EXTRA_DIM = 17
REL_FEATURE_DIM = REL_ORDER_PAIR_DIM + REL_EXTRA_DIM  # == 23
_MAX_DIST_BUCKET = 3
assert REL_FEATURE_DIM == 23

_ORDER_PAIR_INDEX = {}
_next = 0
for _a in range(3):
    for _b in range(_a, 3):
        _ORDER_PAIR_INDEX[(_a, _b)] = _next
        _next += 1


# ---------------------------------------------------------------------------
# explicit relation graph
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ObjectRelationGraph:
    """Explicit, deterministic relation graph over the B0/B1/B2 objects."""

    edge_i: np.ndarray  # [R] object id (lower)
    edge_j: np.ndarray  # [R] object id (higher)
    family: np.ndarray  # [R] relation family id
    feature: np.ndarray  # [R, REL_FEATURE_DIM] symmetric descriptor
    n_atoms: int
    n_bonds: int
    n_triples: int
    n_objects: int


def _patch_local_adjacency(
    graph: ExplicitBasisGraph,
) -> tuple[np.ndarray, list[dict[int, list[tuple[int, int]]]], np.ndarray]:
    """Rebuild the real molecular adjacency inside every patch from B1 objects."""
    n_patches = int(graph.n_patches)
    node_counts = np.bincount(graph.patch, minlength=n_patches).astype(np.int64)
    node_start = np.concatenate([[0], np.cumsum(node_counts)[:-1]]).astype(np.int64)
    adjacency: list[dict[int, list[tuple[int, int]]]] = [{} for _ in range(n_patches)]
    for k in range(int(graph.n_bonds)):
        p = int(graph.l1_patch[k])
        u = int(graph.l1_u[k]) - int(node_start[p])
        v = int(graph.l1_v[k]) - int(node_start[p])
        adjacency[p].setdefault(u, []).append((v, k))
        adjacency[p].setdefault(v, []).append((u, k))
    return node_start, adjacency, node_counts


def _patch_distance_bounds(
    adjacency: list[dict[int, list[tuple[int, int]]]],
    n_patches: int,
) -> list[dict[int, dict[int, int]]]:
    """All-pairs shortest path (BFS) inside every patch."""
    distances: list[dict[int, dict[int, int]]] = []
    for p in range(n_patches):
        patch_distances: dict[int, dict[int, int]] = {}
        for source in sorted(adjacency[p]):
            dist = {source: 0}
            frontier = [source]
            while frontier:
                nxt: list[int] = []
                for node in frontier:
                    base = dist[node]
                    for neighbour, _bond in adjacency[p][node]:
                        if neighbour not in dist:
                            dist[neighbour] = base + 1
                            nxt.append(neighbour)
                frontier = nxt
            patch_distances[source] = dist
        distances.append(patch_distances)
    return distances


def _object_supports(
    graph: ExplicitBasisGraph,
) -> tuple[list[frozenset[int]], list[frozenset[int]], np.ndarray]:
    """Atom supports, bond supports and object order for every object."""
    n_atoms = int(graph.n_atoms)
    n_bonds = int(graph.n_bonds)
    n_triples = int(graph.n_triples)
    n_objects = n_atoms + n_bonds + n_triples
    atom_support: list[frozenset[int]] = [frozenset()] * n_objects
    bond_support: list[frozenset[int]] = [frozenset()] * n_objects
    order = np.zeros(n_objects, dtype=np.int64)
    for index in range(n_atoms):
        atom_support[index] = atom_object_support(graph, index)
    for index in range(n_bonds):
        object_id = n_atoms + index
        atom_support[object_id] = bond_object_support(graph, index)
        bond_support[object_id] = frozenset({index})
        order[object_id] = 1
    for index in range(n_triples):
        object_id = n_atoms + n_bonds + index
        atom_support[object_id] = triple_object_support(graph, index)
        bond_support[object_id] = triple_object_bond_support(graph, index)
        order[object_id] = 2
    return atom_support, bond_support, order


def _min_support_distance(
    support_i: frozenset[int],
    support_j: frozenset[int],
    patch_distances: dict[int, dict[int, int]],
    node_start_p: int,
) -> int:
    if support_i & support_j:
        return 0
    best = None
    for a in support_i:
        local_a = int(a) - node_start_p
        reachable = patch_distances.get(local_a, {})
        for b in support_j:
            dist = reachable.get(int(b) - node_start_p)
            if dist is None:
                continue
            if best is None or dist < best:
                best = dist
    if best is None:
        return _MAX_DIST_BUCKET
    return min(int(best), _MAX_DIST_BUCKET)


def build_object_relation_graph(
    graph: ExplicitBasisGraph,
    root_atoms: np.ndarray | None = None,
) -> ObjectRelationGraph:
    """Enumerate the deterministic, support-local relation graph.

    ``root_atoms`` is an optional ``[N]`` 0/1 array (aligned with the B0 patch
    nodes) marking the patch root; when given, a symmetric ``share_root``
    descriptor is computed.  It is target-free graph information only.
    """
    n_atoms = int(graph.n_atoms)
    n_bonds = int(graph.n_bonds)
    n_triples = int(graph.n_triples)
    n_objects = n_atoms + n_bonds + n_triples
    n_patches = int(graph.n_patches)
    if n_objects == 0:
        return ObjectRelationGraph(
            edge_i=np.zeros(0, dtype=np.int64),
            edge_j=np.zeros(0, dtype=np.int64),
            family=np.zeros(0, dtype=np.int64),
            feature=np.zeros((0, REL_FEATURE_DIM), dtype=np.float32),
            n_atoms=n_atoms,
            n_bonds=n_bonds,
            n_triples=n_triples,
            n_objects=0,
        )

    node_start, adjacency, _counts = _patch_local_adjacency(graph)
    patch_distances = _patch_distance_bounds(adjacency, n_patches)
    atom_support, bond_support, order = _object_supports(graph)

    obj_patch = np.concatenate(
        [graph.patch, graph.l1_patch, graph.b2_patch]
    ).astype(np.int64)
    by_patch: list[list[int]] = [[] for _ in range(n_patches)]
    for object_id in range(n_objects):
        by_patch[int(obj_patch[object_id])].append(object_id)

    # global atom adjacency for the ``adjacent`` family (deterministic)
    global_adjacent: dict[int, list[tuple[int, int]]] = {}
    for k in range(n_bonds):
        u = int(graph.l1_u[k])
        v = int(graph.l1_v[k])
        global_adjacent.setdefault(u, []).append((v, int(graph.l1_bond[k])))
        global_adjacent.setdefault(v, []).append((u, int(graph.l1_bond[k])))

    root_global: set[int] = set()
    if root_atoms is not None:
        root_atoms = np.asarray(root_atoms)
        if int(root_atoms.shape[0]) == n_atoms:
            root_global = {int(i) for i in np.nonzero(root_atoms)[0]}

    edge_i: list[int] = []
    edge_j: list[int] = []
    family: list[int] = []
    features: list[np.ndarray] = []

    for p in range(n_patches):
        objects = by_patch[p]
        p_root = sorted({a for a in root_global if int(graph.patch[a]) == p})
        root_in_patch = p_root[0] if p_root else -1
        for a_index in range(len(objects)):
            i = objects[a_index]
            support_i = atom_support[i]
            bond_i = bond_support[i]
            order_i = int(order[i])
            centre_i = (
                int(graph.b2_center[i - n_atoms - n_bonds])
                if order_i == 2
                else -1
            )
            for b_index in range(a_index + 1, len(objects)):
                j = objects[b_index]
                support_j = atom_support[j]
                bond_j = bond_support[j]
                order_j = int(order[j])
                centre_j = (
                    int(graph.b2_center[j - n_atoms - n_bonds])
                    if order_j == 2
                    else -1
                )
                atom_overlap = len(support_i & support_j)
                bond_overlap = len(bond_i & bond_j)
                adjacent = False
                connecting: list[int] = []
                if bond_overlap > 0:
                    fam = FAMILY_BOND_OVERLAP
                elif atom_overlap > 0:
                    fam = FAMILY_OVERLAP
                else:
                    for atom in support_i:
                        for neighbour, bond_type in global_adjacent.get(atom, ()):
                            if neighbour in support_j:
                                adjacent = True
                                connecting.append(int(bond_type))
                    if not adjacent:
                        continue
                    fam = FAMILY_ADJACENT

                lower_order = min(order_i, order_j)
                upper_order = max(order_i, order_j)
                pair_one_hot = np.zeros(REL_ORDER_PAIR_DIM, dtype=np.float32)
                pair_one_hot[_ORDER_PAIR_INDEX[(lower_order, upper_order)]] = 1.0

                containment = 0.0
                if support_i and support_j:
                    if support_i < support_j or support_j < support_i:
                        containment = 1.0
                share_centre = 0.0
                if centre_i >= 0 and centre_i in support_j:
                    share_centre = 1.0
                if centre_j >= 0 and centre_j in support_i:
                    share_centre = 1.0
                share_root = 0.0
                if root_in_patch >= 0:
                    if root_in_patch in support_i and root_in_patch in support_j:
                        share_root = 1.0

                connecting_one_hot = np.zeros(4, dtype=np.float32)
                if connecting:
                    for bond_type in connecting:
                        if 0 <= bond_type < 4:
                            connecting_one_hot[bond_type] += 1.0
                    connecting_one_hot /= float(len(connecting))
                has_connecting = 1.0 if connecting else 0.0

                min_dist = _min_support_distance(
                    support_i, support_j, patch_distances[p], int(node_start[p])
                )
                size_min = float(min(len(support_i), len(support_j)))
                size_max = float(max(len(support_i), len(support_j)))
                family_one_hot = np.zeros(N_FAMILIES, dtype=np.float32)
                family_one_hot[fam] = 1.0

                descriptor = np.concatenate(
                    [
                        pair_one_hot,
                        np.asarray(
                            [
                                float(atom_overlap),
                                float(bond_overlap),
                                containment,
                                share_centre,
                                share_root,
                                1.0 if adjacent else 0.0,
                            ],
                            dtype=np.float32,
                        ),
                        connecting_one_hot,
                        np.asarray([has_connecting], dtype=np.float32),
                        np.asarray([float(min_dist)], dtype=np.float32),
                        np.asarray([size_min, size_max], dtype=np.float32),
                        family_one_hot,
                    ]
                ).astype(np.float32)
                if descriptor.shape[0] != REL_FEATURE_DIM:
                    raise RuntimeError("relation descriptor width mismatch")
                edge_i.append(i)
                edge_j.append(j)
                family.append(int(fam))
                features.append(descriptor)

    edge_i_arr = np.asarray(edge_i, dtype=np.int64)
    edge_j_arr = np.asarray(edge_j, dtype=np.int64)
    family_arr = np.asarray(family, dtype=np.int64)
    feature_arr = (
        np.stack(features).astype(np.float32)
        if features
        else np.zeros((0, REL_FEATURE_DIM), dtype=np.float32)
    )
    return ObjectRelationGraph(
        edge_i=edge_i_arr,
        edge_j=edge_j_arr,
        family=family_arr,
        feature=feature_arr,
        n_atoms=n_atoms,
        n_bonds=n_bonds,
        n_triples=n_triples,
        n_objects=n_objects,
    )


def relation_counts(graph: ExplicitBasisGraph) -> dict[str, float]:
    """Sparsity summary used by the zero-training relation audit."""
    relations = build_object_relation_graph(graph)
    n_objects = relations.n_objects
    n_rel = int(relations.edge_i.shape[0])
    return {
        "objects": float(n_objects),
        "relations": float(n_rel),
        "edge_object_ratio": (float(n_rel / n_objects) if n_objects else 0.0),
        "family_overlap": float((relations.family == FAMILY_OVERLAP).sum()),
        "family_bond_overlap": float(
            (relations.family == FAMILY_BOND_OVERLAP).sum()
        ),
        "family_adjacent": float((relations.family == FAMILY_ADJACENT).sum()),
    }


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------


def _scatter_stats(
    values: torch.Tensor, group: torch.Tensor, n_groups: int
) -> torch.Tensor:
    """Per-group ``[mean, std, log1p(count)]`` for ``values`` ``[E, D]``."""
    out_width = 3 * int(values.shape[1])
    if n_groups <= 0:
        return values.new_zeros((0, out_width))
    if values.shape[0] == 0:
        return values.new_zeros((n_groups, out_width))
    dtype = values.dtype
    counts = torch.bincount(group.long(), minlength=n_groups).to(dtype)
    safe = counts.clamp_min(1.0).unsqueeze(1)
    sums = values.new_zeros((n_groups, values.shape[1]))
    sums = sums.index_add(0, group.long(), values)
    mean = sums / safe
    squares = values.new_zeros((n_groups, values.shape[1]))
    squares = squares.index_add(0, group.long(), values * values)
    variance = (squares / safe - mean * mean).clamp_min(0.0)
    std = torch.sqrt(variance + 1.0e-8)
    log_count = torch.log1p(counts).unsqueeze(1).expand(-1, values.shape[1])
    return torch.cat([mean, std, log_count], dim=1)


class ExplicitObjectRelationalEncoder(nn.Module):
    """Rank-1 structural channel from explicit objects and explicit relations."""

    def __init__(
        self,
        *,
        atom_categories: int = 28,
        bond_categories: int = 4,
        n_distance_bins: int = 3,
        n_degree_bins: int = 8,
        obj_dim: int = OBJ_DIM,
        rel_hidden: int = REL_HIDDEN,
        rel_latent_dim: int = REL_LATENT_DIM,
        q_hidden: int = Q_HIDDEN,
        q_obj_dim: int = Q_OBJ_DIM,
        object_hidden: int = 100,
        update_hidden: int = 79,
        fusion_hidden: int = 136,
        output_dim: int = 16,
        t_obj: int = T_OBJ,
    ) -> None:
        super().__init__()
        self.atom_categories = int(atom_categories)
        self.bond_categories = int(bond_categories)
        self.n_distance_bins = int(n_distance_bins)
        self.n_degree_bins = int(n_degree_bins)
        self.obj_dim = int(obj_dim)
        self.rel_hidden = int(rel_hidden)
        self.rel_latent_dim = int(rel_latent_dim)
        self.q_hidden = int(q_hidden)
        self.q_obj_dim = int(q_obj_dim)
        self.object_hidden = int(object_hidden)
        self.update_hidden = int(update_hidden)
        self.fusion_hidden = int(fusion_hidden)
        self.output_dim = int(output_dim)
        self.t_obj = int(t_obj)
        if self.t_obj != T_OBJ:
            raise ValueError("t_obj is frozen at 2 (no sweep)")
        # interface metadata for the shared parameter audit
        self.node_dim = self.obj_dim
        self.edge_dim = self.obj_dim
        self.hidden_dim = self.object_hidden
        self.rounds = self.t_obj
        self.include_std_pool = True
        self.kind = "explicit_object_relational"

        d = self.obj_dim
        # -- explicit primitive embeddings ---------------------------------
        self.atom_embedding = nn.Embedding(self.atom_categories, d)
        self.root_embedding = nn.Embedding(2, d)
        self.distance_embedding = nn.Embedding(self.n_distance_bins, d)
        self.degree_embedding = nn.Embedding(self.n_degree_bins, d)
        self.boundary_embedding = nn.Embedding(2, d)
        self.bond_embedding = nn.Embedding(self.bond_categories, d)

        # -- per-order initial object state encoders E0/E1/E2 --------------
        self.order0_encoder = nn.Sequential(
            nn.Linear(d, self.object_hidden),
            nn.ReLU(),
            nn.Linear(self.object_hidden, d),
        )
        self.order1_encoder = nn.Sequential(
            nn.Linear(2 * d + d, self.object_hidden),
            nn.ReLU(),
            nn.Linear(self.object_hidden, d),
        )
        triple_input = 5 * d + 1
        self.triple_input = int(triple_input)
        self.order2_encoder = nn.Sequential(
            nn.Linear(triple_input, self.object_hidden),
            nn.ReLU(),
            nn.Linear(self.object_hidden, d),
        )

        # -- explicit relation encoder + relation latent q -----------------
        self.relation_encoder = nn.Sequential(
            nn.Linear(REL_FEATURE_DIM, self.rel_hidden),
            nn.ReLU(),
            nn.Linear(self.rel_hidden, self.rel_latent_dim),
        )
        q_input = 3 * d + self.rel_latent_dim
        self.q_encoder = nn.Sequential(
            nn.Linear(q_input, self.q_hidden),
            nn.ReLU(),
            nn.Linear(self.q_hidden, self.q_obj_dim),
        )

        # -- shared object update U (weight-tied across rounds and orders) --
        self.aggregate_dim = int((N_FAMILIES + 1) * 3 * self.q_obj_dim)
        self.update = nn.Sequential(
            nn.Linear(d + self.aggregate_dim, self.update_hidden),
            nn.ReLU(),
            nn.Linear(self.update_hidden, d),
        )

        # -- late pooling + scalar + rank-1 channel -------------------------
        self.fusion = nn.Sequential(
            nn.Linear(3 * 3 * d, self.fusion_hidden),
            nn.ReLU(),
            nn.Linear(self.fusion_hidden, 1),
        )
        self.rank1_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.rank1_direction = nn.Parameter(torch.zeros(self.output_dim))
        nn.init.normal_(self.rank1_direction, std=0.1)

        # frozen diagnostic hooks (never active during training)
        self._ablate_relations = False
        self._rounds = self.t_obj
        self._family_mask: tuple[bool, bool, bool] = (True, True, True)
        self._diag: dict[str, Any] | None = None

    # -- explicit object features ------------------------------------------
    def _atom_primitives(self, data) -> torch.Tensor:
        atom = data.struct_atom.long()
        root = data.struct_root.long()
        distance = data.struct_dist.long().clamp_max(self.n_distance_bins - 1)
        degree = data.esb_degree.long().clamp_max(self.n_degree_bins - 1)
        boundary = (data.struct_dist.long() >= (self.n_distance_bins - 1)).long()
        return (
            self.atom_embedding(atom)
            + self.root_embedding(root)
            + self.distance_embedding(distance)
            + self.degree_embedding(degree)
            + self.boundary_embedding(boundary)
        )

    def _initial_states(self, data, z: torch.Tensor) -> torch.Tensor:
        n_atoms = int(data.struct_atom.numel())
        h0 = self.order0_encoder(z)

        l1_u = data.esb_l1_u.long()
        l1_v = data.esb_l1_v.long()
        if l1_u.numel():
            edge = self.bond_embedding(data.esb_l1_bond.long())
            feature1 = torch.cat(
                [z[l1_u] + z[l1_v], torch.abs(z[l1_u] - z[l1_v]), edge], dim=1
            )
            h1 = self.order1_encoder(feature1)
        else:
            h1 = z.new_zeros((0, self.obj_dim))

        center = data.esb_b2_center.long()
        if center.numel():
            endpoint_u = data.esb_b2_u.long()
            endpoint_w = data.esb_b2_w.long()
            bond_cu = self.bond_embedding(data.esb_b2_bond_cu.long())
            bond_cw = self.bond_embedding(data.esb_b2_bond_cw.long())
            closure = data.esb_b2_closure.long()
            closure_flag = (closure >= 0).to(z.dtype).unsqueeze(1)
            closure_emb = self.bond_embedding(closure.clamp_min(0).long())
            closure_emb = closure_emb * closure_flag
            feature2 = torch.cat(
                [
                    z[center],
                    z[endpoint_u] + z[endpoint_w],
                    torch.abs(z[endpoint_u] - z[endpoint_w]),
                    bond_cu + bond_cw,
                    closure_flag,
                    closure_emb,
                ],
                dim=1,
            )
            h2 = self.order2_encoder(feature2)
        else:
            h2 = z.new_zeros((0, self.obj_dim))
        return torch.cat([h0, h1, h2], dim=0)

    def _relation_latent(
        self, h: torch.Tensor, edge_i: torch.Tensor, edge_j: torch.Tensor,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        hi = h[edge_i]
        hj = h[edge_j]
        rel = self.relation_encoder(feature)
        q_input = torch.cat([hi + hj, torch.abs(hi - hj), hi * hj, rel], dim=1)
        return self.q_encoder(q_input)

    def _aggregate(
        self,
        q: torch.Tensor,
        edge_i: torch.Tensor,
        edge_j: torch.Tensor,
        family: torch.Tensor,
        n_objects: int,
    ) -> torch.Tensor:
        """Fixed per-family invariant statistics around every object."""
        if q.shape[0] == 0:
            return q.new_zeros((n_objects, self.aggregate_dim))
        # every undirected relation contributes to both endpoints
        src = torch.cat([edge_i, edge_j])
        fam = torch.cat([family, family])
        q_both = torch.cat([q, q], dim=0)
        family_key = src * N_FAMILIES + fam
        per_family = _scatter_stats(q_both, family_key, n_objects * N_FAMILIES)
        per_family = per_family.view(n_objects, N_FAMILIES * 3 * self.q_obj_dim)
        total = _scatter_stats(q_both, src, n_objects)
        return torch.cat([per_family, total], dim=1)

    # -- forward -----------------------------------------------------------
    def forward(self, data, return_details: bool = False, **kwargs):
        z = self._atom_primitives(data)
        n_atoms = int(data.struct_atom.numel())
        n_bonds = int(data.esb_l1_u.numel())
        n_triples = int(data.esb_b2_center.numel())
        n_objects = n_atoms + n_bonds + n_triples

        unique_patch, atom_group = torch.unique(
            data.struct_patch.long(), return_inverse=True
        )
        n_patches = int(unique_patch.numel())
        if n_objects == 0 or n_patches == 0:
            empty = z.new_zeros((0, self.output_dim))
            if not return_details:
                return empty
            return {"e_struct": empty, "s": z.new_zeros((0,))}

        h = self._initial_states(data, z)
        edge_i = data.eor_edge_i.long()
        edge_j = data.eor_edge_j.long()
        family = data.eor_family.long()
        feature = data.eor_feat.to(z.dtype)

        family_mask = torch.tensor(
            self._family_mask, dtype=torch.bool, device=z.device
        )
        rounds = int(self._rounds)
        h_rounds = [h]
        q_rounds = []
        update_magnitudes = []
        for _round in range(rounds):
            if edge_i.numel():
                q = self._relation_latent(h, edge_i, edge_j, feature)
                if self._ablate_relations:
                    q = torch.zeros_like(q)
                else:
                    allowed = family_mask[family]
                    q = q * allowed.to(q.dtype).unsqueeze(1)
                aggregation = self._aggregate(q, edge_i, edge_j, family, n_objects)
            else:
                q = h.new_zeros((0, self.q_obj_dim))
                aggregation = h.new_zeros((n_objects, self.aggregate_dim))
            delta = self.update(torch.cat([h, aggregation], dim=1))
            h = h + delta
            h_rounds.append(h)
            q_rounds.append(q)
            update_magnitudes.append(float(delta.detach().norm().item()))

        h_final = h
        stats0 = _scatter_stats(
            h_final[:n_atoms], atom_group, n_patches
        )
        if n_bonds:
            bond_group = torch.searchsorted(unique_patch, data.esb_l1_patch.long())
            stats1 = _scatter_stats(
                h_final[n_atoms : n_atoms + n_bonds], bond_group, n_patches
            )
        else:
            stats1 = z.new_zeros((n_patches, 3 * self.obj_dim))
        if n_triples:
            triple_group = torch.searchsorted(
                unique_patch, data.esb_b2_patch.long()
            )
            stats2 = _scatter_stats(
                h_final[n_atoms + n_bonds :], triple_group, n_patches
            )
        else:
            stats2 = z.new_zeros((n_patches, 3 * self.obj_dim))
        pooled = torch.cat([stats0, stats1, stats2], dim=1)
        s = self.fusion(pooled).squeeze(-1)
        e_struct = self.rank1_bias.unsqueeze(0) + s.unsqueeze(1) * (
            self.rank1_direction.unsqueeze(0)
        )

        if return_details:
            with torch.no_grad():
                detail = {
                    "e_struct": e_struct,
                    "s": s,
                    "h_rounds": [t.detach().clone() for t in h_rounds],
                    "q_rounds": [t.detach().clone() for t in q_rounds],
                    "update_magnitudes": update_magnitudes,
                    "obj_order": torch.cat(
                        [
                            torch.zeros(n_atoms, dtype=torch.long, device=z.device),
                            torch.ones(n_bonds, dtype=torch.long, device=z.device),
                            torch.full(
                                (n_triples,), 2, dtype=torch.long, device=z.device
                            ),
                        ]
                    ),
                    "n_objects": n_objects,
                    "n_patches": n_patches,
                }
            return detail
        return e_struct

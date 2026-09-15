"""Explicit structural basis with learned scalar valuation (rank-1 channel).

Motivation
----------
The completed B-full encoder (``structural_patch_encoder.SharedStructuralPatchEncoder``)
propagates hidden node states over the real rooted typed radius-2 patch graph
and emits ``e_struct in R^16``.  A zero-training mechanism audit showed that
this 16-D structural channel is *effectively rank-1* (effective rank ~1.1-1.2;
a frozen rank-1 reconstruction changes the 2-seed soup by ~1.6e-5), i.e. in
practice it is a patch-specific scalar modulation.  The previous explicit
*composition* route (learned object existence / activity gates) failed because
the learned gate saturated near its initialisation.

This module tests a different, deliberately narrow hypothesis:

    The graph defines **what structural objects exist**; the network only
    learns **what each explicitly present object is worth for the task**.

Form (exactly one pre-registered architecture; no attention, no Transformer,
no learned dictionary, no soft assignment, no message passing)::

    B0  atom objects              support = {v}
    B1  bond objects              support = {u, v},        bonds = {(u, v)}
    B2  centred 3-atom objects    support = {u, v, w},     centre = v,
                                  bonds = {(v, u), (v, w)} (+ closure (u, w))
    valuation   t_r = f_r(object) in R          (shared per order, signed)
    statistics  A(P) = [mean, std, log1p(count)] for each order  in R^9
    scalar      s(P) = fusion(A(P)) in R
    channel     e_struct(P) = b + s(P) * v      with b, v trainable in R^16

``e_struct`` is therefore rank-1 **by construction** (it is an affine image of
a single scalar), which is exactly the functional form the frozen B-full audit
identified.  Every object exists deterministically before any parameter is
evaluated; no parameter can create, delete or re-weight object existence, and
there is no sigmoid/gate anywhere.

Core principle
--------------
**We do not learn what the structure is.  We learn the value of structures that
are explicitly present.**

Preprocessing contract
----------------------
Candidate objects are enumerated once per split by
``build_explicit_basis_graph`` from the same patch graphs used by B-full.  The
encoder consumes per-molecule flat tensors attached to the PyG ``Data`` object:

``struct_atom``        int64 ``[N]``  atom category per patch-node
``struct_root``        int64 ``[N]``  1 for the patch root, else 0
``struct_dist``        int64 ``[N]``  BFS distance from the patch root
``struct_patch``       int64 ``[N]``  patch id owning each node
``esb_degree``         int64 ``[N]``  degree *within the owning patch*
``esb_l1_u``/``esb_l1_v``     int64 ``[M]``  atom ids of every real bond object
``esb_l1_bond``/``esb_l1_patch`` int64 ``[M]`` bond type / owning patch
``esb_l1_index``       int64 ``[M]``  identity index (== object id)
``esb_b2_center``/``esb_b2_u``/``esb_b2_w`` int64 ``[C]`` centred triple
``esb_b2_l1_cu``/``esb_b2_l1_cw``/``esb_b2_l1_closure`` int64 ``[C]`` bonds
``esb_b2_bond_cu``/``esb_b2_bond_cw``/``esb_b2_closure`` int64 ``[C]`` types
``esb_b2_patch``       int64 ``[C]``  owning patch
``struct_n_patches``   int            total patches in the batch

``struct_src`` / ``struct_dst`` are **never** read by the encoder: adjacency is
used only when enumerating the basis (``build_explicit_basis_graph``).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
import torch.nn as nn

BASIS_ORDERS = (0, 1, 2)


# ---------------------------------------------------------------------------
# target-free deterministic object enumeration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ExplicitBasisGraph:
    """Explicit, deterministic B0/B1/B2 object basis of one molecule."""

    patch: np.ndarray  # [N] patch id of every atom (a B0 object)
    degree: np.ndarray  # [N] degree within the owning patch
    # B1: one object per real molecular bond
    l1_u: np.ndarray  # [M] atom id (global within molecule)
    l1_v: np.ndarray  # [M]
    l1_bond: np.ndarray  # [M] bond category
    l1_patch: np.ndarray  # [M]
    # B2: one object per centred connected wedge/closure (centre, u, w)
    b2_center: np.ndarray  # [C]
    b2_u: np.ndarray  # [C]
    b2_w: np.ndarray  # [C]
    b2_l1_cu: np.ndarray  # [C] B1 object id of bond (centre, u)
    b2_l1_cw: np.ndarray  # [C] B1 object id of bond (centre, w)
    b2_l1_closure: np.ndarray  # [C] B1 object id of (u, w), -1 if absent
    b2_bond_cu: np.ndarray  # [C] bond category of (centre, u)
    b2_bond_cw: np.ndarray  # [C] bond category of (centre, w)
    b2_closure: np.ndarray  # [C] bond category of (u, w), -1 if absent
    b2_patch: np.ndarray  # [C]
    n_atoms: int
    n_bonds: int
    n_triples: int
    n_patches: int


def build_explicit_basis_graph(
    patch: np.ndarray,
    src: np.ndarray,
    dst: np.ndarray,
    bond: np.ndarray,
    edge_patch: np.ndarray,
    n_patches: int,
) -> ExplicitBasisGraph:
    """Enumerate B0/B1/B2 explicitly and deterministically.

    ``patch``/``src``/``dst``/``bond``/``edge_patch`` follow the
    ``sspe.PatchGraph`` convention: ``patch[i]`` is the patch owning patch-node
    ``i``; ``src[e]``/``dst[e]`` are **patch-local** node indices and each
    undirected bond appears once per direction.  Adjacency is read here *only*
    to decide which explicit objects exist; no hidden state is propagated.

    B0: every patch atom.  B1: every real molecular bond (one undirected
    object).  B2: for every patch atom ``v`` and every unordered neighbour pair
    ``{u, w}`` of ``v`` inside the patch, a *centred* object with distinguished
    centre ``v``; its bond support is ``{(v,u), (v,w)}`` plus the closure bond
    ``(u,w)`` when that is a real bond.  ``(u-v-w)`` and ``(v-u-w)`` are
    distinct objects even when their atom supports coincide: the centre is part
    of the object's semantic role.
    """
    patch = np.asarray(patch, dtype=np.int64)
    src = np.asarray(src, dtype=np.int64)
    dst = np.asarray(dst, dtype=np.int64)
    bond = np.asarray(bond, dtype=np.int64)
    edge_patch = np.asarray(edge_patch, dtype=np.int64)
    n_patches = int(n_patches)
    n_atoms = int(patch.shape[0])
    if not (src.shape == dst.shape == bond.shape == edge_patch.shape):
        raise ValueError("edge arrays must have identical length")

    node_counts = np.bincount(patch, minlength=n_patches).astype(np.int64)
    node_start = np.concatenate([[0], np.cumsum(node_counts)[:-1]]).astype(np.int64)

    # -- undirected bonds per patch (deterministic, keyed by sorted endpoints)
    patch_bonds: list[dict[tuple[int, int], int]] = [{} for _ in range(n_patches)]
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
    adjacency: list[dict[int, list[tuple[int, int]]]] = [
        {} for _ in range(n_patches)
    ]
    l1_entries: list[tuple[int, int, int, int]] = []  # (patch, u, v, bond)
    for p in range(n_patches):
        for key in sorted(patch_bonds[p]):
            u, v = key
            bond_type = int(patch_bonds[p][key])
            degree[node_start[p] + u] += 1
            degree[node_start[p] + v] += 1
            adjacency[p].setdefault(u, []).append((v, bond_type))
            adjacency[p].setdefault(v, []).append((u, bond_type))
            l1_entries.append((p, int(u), int(v), bond_type))

    n_bonds = len(l1_entries)
    l1_u = np.asarray(
        [node_start[p] + u for (p, u, _v, _t) in l1_entries], dtype=np.int64
    )
    l1_v = np.asarray(
        [node_start[p] + v for (p, _u, v, _t) in l1_entries], dtype=np.int64
    )
    l1_bond = np.asarray([t for (_p, _u, _v, t) in l1_entries], dtype=np.int64)
    l1_patch = np.asarray([p for (p, _u, _v, _t) in l1_entries], dtype=np.int64)

    # local B1 id of every (patch, sorted endpoints)
    bond_index: dict[tuple[int, int, int], int] = {}
    for index, (p, u, v, _t) in enumerate(l1_entries):
        bond_index[(p, u, v)] = index

    # -- B2: centred wedges / closed triangles --------------------------------
    b2_center: list[int] = []
    b2_u: list[int] = []
    b2_w: list[int] = []
    b2_l1_cu: list[int] = []
    b2_l1_cw: list[int] = []
    b2_l1_closure: list[int] = []
    b2_bond_cu: list[int] = []
    b2_bond_cw: list[int] = []
    b2_closure: list[int] = []
    b2_patch: list[int] = []
    for p in range(n_patches):
        for center in sorted(adjacency[p]):
            neighbours = sorted(adjacency[p][center])
            for i in range(len(neighbours)):
                u, bond_cu = neighbours[i]
                for j in range(i + 1, len(neighbours)):
                    w, bond_cw = neighbours[j]
                    if u == w:
                        continue
                    key_uw = (min(u, w), max(u, w))
                    closure_type = patch_bonds[p].get(key_uw)
                    closure_index = (
                        bond_index[(p, key_uw[0], key_uw[1])]
                        if closure_type is not None
                        else -1
                    )
                    b2_center.append(node_start[p] + center)
                    b2_u.append(node_start[p] + u)
                    b2_w.append(node_start[p] + w)
                    b2_l1_cu.append(bond_index[(p, min(center, u), max(center, u))])
                    b2_l1_cw.append(bond_index[(p, min(center, w), max(center, w))])
                    b2_l1_closure.append(int(closure_index))
                    b2_bond_cu.append(int(bond_cu))
                    b2_bond_cw.append(int(bond_cw))
                    b2_closure.append(
                        int(closure_type) if closure_type is not None else -1
                    )
                    b2_patch.append(int(p))

    n_triples = len(b2_center)
    return ExplicitBasisGraph(
        patch=patch,
        degree=degree,
        l1_u=l1_u,
        l1_v=l1_v,
        l1_bond=l1_bond,
        l1_patch=l1_patch,
        b2_center=np.asarray(b2_center, dtype=np.int64),
        b2_u=np.asarray(b2_u, dtype=np.int64),
        b2_w=np.asarray(b2_w, dtype=np.int64),
        b2_l1_cu=np.asarray(b2_l1_cu, dtype=np.int64),
        b2_l1_cw=np.asarray(b2_l1_cw, dtype=np.int64),
        b2_l1_closure=np.asarray(b2_l1_closure, dtype=np.int64),
        b2_bond_cu=np.asarray(b2_bond_cu, dtype=np.int64),
        b2_bond_cw=np.asarray(b2_bond_cw, dtype=np.int64),
        b2_closure=np.asarray(b2_closure, dtype=np.int64),
        b2_patch=np.asarray(b2_patch, dtype=np.int64),
        n_atoms=n_atoms,
        n_bonds=n_bonds,
        n_triples=n_triples,
        n_patches=n_patches,
    )


# ---------------------------------------------------------------------------
# exact support / provenance helpers (auditable, no latent membership)
# ---------------------------------------------------------------------------


def atom_object_support(graph: ExplicitBasisGraph, index: int) -> frozenset[int]:
    """B0 support: a single atom."""
    return frozenset({int(index)})


def bond_object_support(graph: ExplicitBasisGraph, index: int) -> frozenset[int]:
    """B1 atom support: the two bond endpoints."""
    return frozenset({int(graph.l1_u[index]), int(graph.l1_v[index])})


def bond_object_bond_support(
    graph: ExplicitBasisGraph, index: int
) -> frozenset[int]:
    """B1 bond support: the single B1 object id (itself)."""
    return frozenset({int(index)})


def triple_object_center(graph: ExplicitBasisGraph, index: int) -> int:
    """B2 distinguished centre atom."""
    return int(graph.b2_center[index])


def triple_object_support(graph: ExplicitBasisGraph, index: int) -> frozenset[int]:
    """B2 atom support: ``{centre, endpoint_u, endpoint_w}``."""
    return frozenset(
        {
            int(graph.b2_center[index]),
            int(graph.b2_u[index]),
            int(graph.b2_w[index]),
        }
    )


def triple_object_bond_support(
    graph: ExplicitBasisGraph, index: int
) -> frozenset[int]:
    """B2 bond support: the two incident bonds (+ closure when present)."""
    support = {int(graph.b2_l1_cu[index]), int(graph.b2_l1_cw[index])}
    closure = int(graph.b2_l1_closure[index])
    if closure >= 0:
        support.add(closure)
    return frozenset(support)


def object_count(graph: ExplicitBasisGraph, order: int) -> int:
    """Number of explicit objects of one order."""
    if order == 0:
        return int(graph.n_atoms)
    if order == 1:
        return int(graph.n_bonds)
    if order == 2:
        return int(graph.n_triples)
    raise ValueError(f"unknown basis order {order}")


def expected_triple_count(graph: ExplicitBasisGraph) -> int:
    """``sum_v C(deg_within_patch(v), 2)`` -- the B2 enumeration invariant."""
    total = 0
    for value in np.asarray(graph.degree, dtype=np.int64).tolist():
        total += value * (value - 1) // 2
    return int(total)


# ---------------------------------------------------------------------------
# encoder
# ---------------------------------------------------------------------------


def _scatter_statistics(
    values: torch.Tensor, group: torch.Tensor, n_groups: int
) -> torch.Tensor:
    """``[mean, std, log1p(count)]`` per group for a flat scalar tensor."""
    if n_groups <= 0:
        return values.new_zeros((0, 3))
    if values.numel() == 0:
        return values.new_zeros((n_groups, 3))
    counts = torch.bincount(group.long(), minlength=n_groups).to(values.dtype)
    safe = counts.clamp_min(1.0)
    total = values.new_zeros(n_groups)
    total.index_add_(0, group.long(), values)
    mean = total / safe
    second = values.new_zeros(n_groups)
    second.index_add_(0, group.long(), values * values)
    variance = (second / safe - mean * mean).clamp_min(0.0)
    std = torch.sqrt(variance + 1.0e-8)
    log_count = torch.log1p(counts)
    return torch.stack([mean, std, log_count], dim=1)


class ExplicitStructuralBasisEncoder(nn.Module):
    """Scalar valuation of an explicit B0/B1/B2 basis with a rank-1 output.

    Every explicit object is scored by a shared, signed per-order valuation
    MLP (no sigmoid, no gate, no selection).  Fixed invariant statistics
    (mean / std / log1p count) are computed per order and fused into one patch
    scalar ``s(P)``.  The structural channel is ``b + s(P) * v`` with trainable
    ``b`` and ``v``, so it is rank-1 by construction.
    """

    def __init__(
        self,
        *,
        atom_categories: int = 28,
        bond_categories: int = 4,
        n_distance_bins: int = 3,
        n_degree_bins: int = 8,
        node_dim: int = 20,
        edge_dim: int = 16,
        valuation_hidden: int = 184,
        fusion_hidden: int = 184,
        output_dim: int = 16,
    ) -> None:
        super().__init__()
        self.atom_categories = int(atom_categories)
        self.bond_categories = int(bond_categories)
        self.n_distance_bins = int(n_distance_bins)
        self.n_degree_bins = int(n_degree_bins)
        self.node_dim = int(node_dim)
        self.edge_dim = int(edge_dim)
        self.valuation_hidden = int(valuation_hidden)
        self.fusion_hidden = int(fusion_hidden)
        self.output_dim = int(output_dim)
        # Interface metadata for the shared parameter audit.  ``rounds = 0``
        # records that no message passing happens anywhere in this module.
        self.hidden_dim = int(valuation_hidden)
        self.rounds = 0
        self.include_std_pool = True
        self.kind = "explicit_basis_rank1"
        if min(
            self.atom_categories,
            self.bond_categories,
            self.n_distance_bins,
            self.n_degree_bins,
            self.node_dim,
            self.edge_dim,
            self.valuation_hidden,
            self.fusion_hidden,
            self.output_dim,
        ) < 1:
            raise ValueError("encoder dimensions must be positive")

        # -- explicit primitives (allowed fields only) -----------------------
        self.atom_embedding = nn.Embedding(self.atom_categories, self.node_dim)
        self.root_embedding = nn.Embedding(2, self.node_dim)
        self.distance_embedding = nn.Embedding(
            self.n_distance_bins, self.node_dim
        )
        self.degree_embedding = nn.Embedding(self.n_degree_bins, self.node_dim)
        self.boundary_embedding = nn.Embedding(2, self.node_dim)
        self.bond_embedding = nn.Embedding(self.bond_categories, self.edge_dim)

        # -- per-order shared valuation networks (signed scalars) ------------
        self.atom_valuation = nn.Sequential(
            nn.Linear(self.node_dim, self.valuation_hidden),
            nn.ReLU(),
            nn.Linear(self.valuation_hidden, 1),
        )
        self.bond_valuation = nn.Sequential(
            nn.Linear(2 * self.node_dim + self.edge_dim, self.valuation_hidden),
            nn.ReLU(),
            nn.Linear(self.valuation_hidden, 1),
        )
        triple_input = 3 * self.node_dim + 2 * self.edge_dim + 1
        self.triple_input = int(triple_input)
        self.triple_valuation = nn.Sequential(
            nn.Linear(triple_input, self.valuation_hidden),
            nn.ReLU(),
            nn.Linear(self.valuation_hidden, 1),
        )
        # -- fixed invariant aggregation -> one patch scalar -----------------
        self.fusion = nn.Sequential(
            nn.Linear(3 * len(BASIS_ORDERS), self.fusion_hidden),
            nn.ReLU(),
            nn.Linear(self.fusion_hidden, 1),
        )
        # -- rank-1 structural channel --------------------------------------
        self.rank1_bias = nn.Parameter(torch.zeros(self.output_dim))
        self.rank1_direction = nn.Parameter(torch.zeros(self.output_dim))
        nn.init.normal_(self.rank1_direction, std=0.1)

    # -- explicit object features ------------------------------------------
    def _atom_primitives(self, data: Any) -> torch.Tensor:
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

    def _group_index(self, data: Any, patch_ids: torch.Tensor, unique: torch.Tensor):
        # ``unique`` is the sorted unique patch ids of the atom objects; every
        # object's patch id is one of them, so searchsorted yields the dense
        # batch-local group index without relying on adjacency.
        if patch_ids.numel() == 0:
            return patch_ids.new_zeros((0,))
        return torch.searchsorted(unique, patch_ids.long())

    # -- forward -----------------------------------------------------------
    def forward(
        self,
        data: Any,
        return_details: bool = False,
        ablate_orders: Any = None,
    ):
        """Encode a batch of patches.

        ``ablate_orders`` is a frozen *diagnostic* hook: the listed basis orders
        have their three aggregated statistics (mean / std / log1p count)
        zeroed before the fusion MLP.  It is empty by default (or read from the
        transient ``_ablate_orders`` attribute when not given), so the training
        and evaluation forward pass is unaffected.
        """
        if ablate_orders is None:
            ablate_orders = getattr(self, "_ablate_orders", ())
        ablate = {int(order) for order in ablate_orders}
        z = self._atom_primitives(data)
        unique_patch, atom_group = torch.unique(
            data.struct_patch.long(), return_inverse=True
        )
        n_patches = int(unique_patch.numel())
        if n_patches == 0:
            empty = z.new_zeros((0, self.output_dim))
            if not return_details:
                return empty
            return {"e_struct": empty, "s": z.new_zeros((0,))}

        # B0 --------------------------------------------------------------
        t0 = self.atom_valuation(z).squeeze(-1)
        stats0 = _scatter_statistics(t0, atom_group, n_patches)

        # B1 --------------------------------------------------------------
        l1_u = data.esb_l1_u.long()
        l1_v = data.esb_l1_v.long()
        l1_bond = data.esb_l1_bond.long()
        if l1_u.numel():
            edge = self.bond_embedding(l1_bond)
            feature1 = torch.cat(
                [z[l1_u] + z[l1_v], torch.abs(z[l1_u] - z[l1_v]), edge], dim=1
            )
            t1 = self.bond_valuation(feature1).squeeze(-1)
            l1_group = self._group_index(data, data.esb_l1_patch, unique_patch)
            stats1 = _scatter_statistics(t1, l1_group, n_patches)
        else:
            t1 = z.new_zeros((0,))
            stats1 = z.new_zeros((n_patches, 3))

        # B2 --------------------------------------------------------------
        center = data.esb_b2_center.long()
        endpoint_u = data.esb_b2_u.long()
        endpoint_w = data.esb_b2_w.long()
        if center.numel():
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
            t2 = self.triple_valuation(feature2).squeeze(-1)
            b2_group = self._group_index(data, data.esb_b2_patch, unique_patch)
            stats2 = _scatter_statistics(t2, b2_group, n_patches)
        else:
            t2 = z.new_zeros((0,))
            stats2 = z.new_zeros((n_patches, 3))

        aggregated = torch.cat([stats0, stats1, stats2], dim=1)
        if ablate:
            aggregated = aggregated.clone()
            for order in ablate:
                if order in BASIS_ORDERS:
                    aggregated[:, 3 * int(order) : 3 * int(order) + 3] = 0.0
        s = self.fusion(aggregated).squeeze(-1)
        e_struct = self.rank1_bias.unsqueeze(0) + s.unsqueeze(1) * (
            self.rank1_direction.unsqueeze(0)
        )
        if not return_details:
            return e_struct
        return {
            "e_struct": e_struct,
            "s": s,
            "aggregated": aggregated,
            "t0": t0,
            "t1": t1,
            "t2": t2,
            "atom_group": atom_group,
            "unique_patch": unique_patch,
            "n_patches": n_patches,
        }

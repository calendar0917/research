"""Compact-v6: coarse rooted topological roles + attribute primitives.

Motivation
----------
The historical compact-v2/v4 exact patch token is (in effect) a *rooted
coarse topology* token: the chemistry was carried by the 146D continuous
shell descriptor.  The corrected exact typed token makes the local identity
provably typed, but that identity is an exact-token lookup whose vocabulary
fragments massively and does not improve, and slightly degrades, the compact
family (see ``notes/typed_patch_tokenizer_correctness_repair.md`` and
``notes/corrected_token_fragmentation_rarity_audit.md``).

Compact-v6 does not change the token at all.  It adds a *factorized*
attribute channel on top of the shared coarse topology:

* topology sharing is unchanged (historical coarse token embedding);
* chemistry enters through a small, shared, permutation-invariant,
  compositional attribute encoder that is **conditioned on coarse rooted
  topological roles** instead of an exact typed-token vocabulary.

This module computes the *role primitives* for one radius-r patch.  It never
looks at the exact typed token, the target, or any audit quantity.  The only
inputs are: the coarse rooted patch topology (root designation, incidence
structure, root distance; atom/bond *types* deliberately dropped) and the
atom/bond categorical types themselves.

Topological role
----------------
On the coarse rooted topology (root marked; every other atom colored only by
its distance from the root; every bond-incidence vertex an untyped cell) we
take the **automorphism orbits** of the vertex-colored incidence graph via
``pynauty.autgrp``.  Orbits are re-indexed canonically through
``pynauty.canon_label`` so the role numbering does not depend on node IDs.
Two atoms therefore share a topological role iff they are interchangeable by a
root-preserving, distance-preserving automorphism of the coarse patch.

Each atom is described by a *small structural role descriptor* (<= 8 raw
features); we deliberately do **not** build a global orbit-id embedding
(a global orbit vocabulary is exactly the failure mode we are trying to
avoid).  The descriptor is: root flag, normalized root distance, normalized
degree, normalized orbit size, boundary flag, and the fraction of neighbours
that are closer / equidistant / farther from the root.  It distinguishes the
semantic roles required by the design (root / distance-1 / distance-2 /
branch / ring-closure) and is permutation-invariant.

Attribute primitives
--------------------
* atom  : ``(atom_type, role_descriptor)``
* bond  : ``(bond_type, role_descriptor_u, role_descriptor_v)`` where the two
  endpoints are ordered by their canonical position (a permutation-invariant
  tie break), so the primitive has no node-ID directionality.

Everything here is a pure function of the patch; no train-only fitting is
required (role features are structural and bounded), so there is no leakage.
"""

from __future__ import annotations

from collections import defaultdict, deque
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np

# Version of the topological-role definition.  Bump this whenever the orbit
# construction or the structural role descriptor changes; it is recorded in
# every run so a stale role cache can never be silently reused.
ROLE_DEFINITION_VERSION = "coarse_rooted_automorphism_orbit_v1"
ATOM_SEMANTIC_VERSION = "zinc_atom_type_v1"
BOND_SEMANTIC_VERSION = "zinc_bond_type_v1"

# Structural role descriptor width (design target: <= 8 raw features).
ROLE_DIM = 8

# ZINC PyG subset categorical schema (kept identical to the patch module; a
# test asserts the two constants agree).
ATOM_CATEGORIES = 28
BOND_CATEGORIES = 4


@dataclass(frozen=True)
class PatchRolePrimitives:
    """Per-patch attribute primitives (one row per atom / per induced bond)."""

    atom_types: np.ndarray  # [n_atoms] int64
    atom_roles: np.ndarray  # [n_atoms, ROLE_DIM] float32
    bond_types: np.ndarray  # [n_bonds] int64
    bond_role_left: np.ndarray  # [n_bonds, ROLE_DIM] float32
    bond_role_right: np.ndarray  # [n_bonds, ROLE_DIM] float32
    n_orbits: int


def attribute_role_fingerprint(radius: int) -> str:
    """Stable fingerprint of the role construction (version + semantics)."""
    import hashlib

    payload = (
        f"{ROLE_DEFINITION_VERSION};radius={int(radius)};"
        f"{ATOM_SEMANTIC_VERSION};{BOND_SEMANTIC_VERSION};role_dim={ROLE_DIM}"
    )
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _ego_distances(graph: Any, center: int, radius: int) -> dict[int, int]:
    distances = {int(center): 0}
    queue: deque[int] = deque([int(center)])
    while queue:
        node = queue.popleft()
        if distances[node] >= int(radius):
            continue
        for neighbor in sorted(graph.neighbors(node)):
            if neighbor not in distances:
                distances[int(neighbor)] = distances[node] + 1
                queue.append(int(neighbor))
    return distances


def _canonical_orbit_ids(
    vertices: Sequence[int],
    orbit_ids: Sequence[int],
    canonical_position: Mapping[int, int],
) -> tuple[np.ndarray, int]:
    """Re-index raw orbits deterministically (independent of node IDs).

    Orbits are ordered by the sorted canonical positions of their members, so
    two isomorphic patches receive the same orbit numbering.
    """
    groups: dict[int, list[int]] = defaultdict(list)
    for vertex in vertices:
        groups[int(orbit_ids[int(vertex)])].append(int(vertex))
    ordered = sorted(
        groups.values(),
        key=lambda group: tuple(sorted(canonical_position[v] for v in group)),
    )
    mapping = {
        vertex: orbit
        for orbit, group in enumerate(ordered)
        for vertex in group
    }
    return (
        np.asarray([mapping[int(vertex)] for vertex in vertices], dtype=np.int64),
        len(ordered),
    )


def coarse_rooted_role_descriptors(
    graph: Any,
    center: int,
    radius: int,
) -> tuple[dict[int, np.ndarray], dict[int, int], int]:
    """Topological roles of a radius-r patch on its coarse rooted topology.

    Returns ``(role_descriptor, canonical_position, n_orbits)`` keyed by the
    *original* node labels.
    """
    try:
        import pynauty  # type: ignore
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise ImportError(
            "compact-v6 attribute roles require pynauty==2.8.8.1"
        ) from exc

    radius = int(radius)
    distances = _ego_distances(graph, int(center), radius)
    original_nodes = tuple(sorted(distances))
    node_to_local = {node: index for index, node in enumerate(original_nodes)}
    induced = graph.induced(set(original_nodes))
    local_edges = tuple(
        (node_to_local[int(left)], node_to_local[int(right)])
        for left, right in sorted(induced.edges())
    )
    n_nodes = len(original_nodes)
    n_edges = len(local_edges)
    root_local = node_to_local[int(center)]

    adjacency: dict[int, list[int]] = {vertex: [] for vertex in range(n_nodes + n_edges)}
    for edge_index, (left, right) in enumerate(local_edges):
        edge_vertex = n_nodes + edge_index
        adjacency[left].append(edge_vertex)
        adjacency[right].append(edge_vertex)
        adjacency[edge_vertex] = [left, right]

    # Coarse rooted coloring: {root}, one cell per root-distance, then one
    # cell for all (untyped) bond-incidence vertices.
    by_distance: dict[int, set[int]] = defaultdict(set)
    for local, node in enumerate(original_nodes):
        if local == root_local:
            continue
        by_distance[int(distances[node])].add(local)
    coloring: list[set[int]] = [{root_local}]
    for distance in sorted(by_distance):
        coloring.append(by_distance[distance])
    if n_edges:
        coloring.append(set(range(n_nodes, n_nodes + n_edges)))

    nauty_graph = pynauty.Graph(
        number_of_vertices=n_nodes + n_edges,
        directed=False,
        adjacency_dict={k: list(v) for k, v in adjacency.items()},
        vertex_coloring=coloring,
    )
    canonical = tuple(int(value) for value in pynauty.canon_label(nauty_graph))
    canonical_position = {vertex: position for position, vertex in enumerate(canonical)}
    _generators, _size1, _size2, raw_orbits, _n_orbits = pynauty.autgrp(nauty_graph)

    node_vertices = tuple(range(n_nodes))
    node_orbits, n_orbits = _canonical_orbit_ids(
        node_vertices, raw_orbits, canonical_position
    )
    orbit_sizes = np.bincount(node_orbits, minlength=max(n_orbits, 1)).astype(np.float64)

    descriptors: dict[int, np.ndarray] = {}
    positions: dict[int, int] = {}
    for local, node in enumerate(original_nodes):
        neighbors = sorted(int(neighbor) for neighbor in induced.neighbors(node))
        distance = int(distances[node])
        degree = len(neighbors)
        if degree:
            n_closer = sum(int(distances[nb]) == distance - 1 for nb in neighbors)
            n_same = sum(int(distances[nb]) == distance for nb in neighbors)
            n_farther = sum(int(distances[nb]) == distance + 1 for nb in neighbors)
        else:
            n_closer = n_same = n_farther = 0
        safe_degree = max(degree, 1)
        descriptor = np.asarray(
            [
                1.0 if local == root_local else 0.0,
                float(distance) / float(radius),
                float(degree) / 5.0,
                float(orbit_sizes[int(node_orbits[local])]) / float(n_nodes),
                1.0 if distance == radius else 0.0,
                float(n_farther) / safe_degree,
                float(n_same) / safe_degree,
                float(n_closer) / safe_degree,
            ],
            dtype=np.float32,
        )
        descriptors[int(node)] = descriptor
        positions[int(node)] = int(canonical_position[local])
    return descriptors, positions, int(n_orbits)


def patch_role_primitives(
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    radius: int,
) -> PatchRolePrimitives:
    """Attribute primitives ``(type, role)`` for one patch."""
    radius = int(radius)
    distances = _ego_distances(graph, int(center), radius)
    original_nodes = tuple(sorted(distances))
    induced = graph.induced(set(original_nodes))
    descriptors, positions, n_orbits = coarse_rooted_role_descriptors(
        graph, int(center), radius
    )

    atom_types = np.asarray(
        [int(node_types[node]) for node in original_nodes], dtype=np.int64
    )
    if atom_types.size and (atom_types.min() < 0 or atom_types.max() >= ATOM_CATEGORIES):
        raise ValueError("atom category outside the ZINC schema")
    atom_roles = np.stack(
        [descriptors[int(node)] for node in original_nodes], axis=0
    ).astype(np.float32, copy=False)

    edge_list = sorted(induced.edges())
    bond_types: list[int] = []
    bond_left: list[np.ndarray] = []
    bond_right: list[np.ndarray] = []
    for left, right in edge_list:
        bond = int(edge_types[graph.edge_key(int(left), int(right))])
        if bond < 0 or bond >= BOND_CATEGORIES:
            raise ValueError("bond category outside the ZINC schema")
        # Canonical endpoint order: smaller canonical position first.  This is
        # a permutation-invariant tie break (canonical labeling fixes the
        # order up to automorphism, and automorphic endpoints share a role).
        if positions[int(left)] <= positions[int(right)]:
            first, second = int(left), int(right)
        else:
            first, second = int(right), int(left)
        bond_types.append(bond)
        bond_left.append(descriptors[first])
        bond_right.append(descriptors[second])

    if bond_types:
        bond_type_array = np.asarray(bond_types, dtype=np.int64)
        bond_left_array = np.stack(bond_left, axis=0).astype(np.float32, copy=False)
        bond_right_array = np.stack(bond_right, axis=0).astype(np.float32, copy=False)
    else:
        bond_type_array = np.zeros(0, dtype=np.int64)
        bond_left_array = np.zeros((0, ROLE_DIM), dtype=np.float32)
        bond_right_array = np.zeros((0, ROLE_DIM), dtype=np.float32)

    return PatchRolePrimitives(
        atom_types=atom_types,
        atom_roles=atom_roles,
        bond_types=bond_type_array,
        bond_role_left=bond_left_array,
        bond_role_right=bond_right_array,
        n_orbits=int(n_orbits),
    )

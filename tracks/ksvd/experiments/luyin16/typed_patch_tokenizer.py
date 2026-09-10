"""Typed patch tokenizer: versioned canonical keys for rooted typed patches.

Motivation
----------
The historical compact-v2 / compact-v4 tokenizer used
``pynauty.certificate(colored incidence graph)`` as the exact patch token.
That byte string is the *canonical adjacency matrix* of the graph, but it does
**not** encode the vertex coloring.  ``pynauty.isomorphic`` compensates with an
extra ordered color-cell-size check, but the bare certificate does not.  As a
result the historical token was a strictly coarser invariant than the intended
rooted typed / distance-aware patch identity: ~10.8% of official-train radius-2
tokens mix different root atoms.

This module makes the equivalence relation explicit and offers two versions:

``typed_tokenizer_v1_historical``
    The exact historical behaviour: ``bytes(pynauty.certificate(incidence))``.
    Kept verbatim so every historical run stays reproducible.

``typed_tokenizer_v2_corrected``
    A complete invariant of the colored incidence graph: the canonical
    adjacency certificate **plus** the canonical semantic color sequence
    (the semantic color label attached to each canonically labelled vertex).
    Two patches get the same key iff there is an isomorphism of the colored
    incidence graphs preserving atom type, bond type, root designation,
    root-distance class and the incidence-type colors.

Ground-truth oracle
-------------------
``typed_isomorphic(A, B)`` is an *independent* reference implemented with
``pynauty.isomorphic``: the incidence colorings are built with cells sorted by
their deterministic semantic key, and equality of the ordered semantic-label
sequence is checked explicitly (this is the part bare ``isomorphic`` misses
for color-label identity).  It is never used on the training path -- only to
validate tokenizer keys (soundness / completeness gates).

Determinism
-----------
No ``hash()``, no unordered iteration.  Semantic color keys are sorted with
``repr`` (a total, process-independent order), cell membership is stored as a
sorted tuple, and the corrected key is an explicit ``bytes`` serialization.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

TYPED_TOKENIZER_V1_HISTORICAL = "typed_tokenizer_v1_historical"
TYPED_TOKENIZER_V2_CORRECTED = "typed_tokenizer_v2_corrected"
DEFAULT_TYPED_TOKENIZER_VERSION = TYPED_TOKENIZER_V1_HISTORICAL

# Order matters for determinism of any persisted fingerprint: it is a fixed
# tuple, never a set.
SUPPORTED_TYPED_TOKENIZER_VERSIONS = (
    TYPED_TOKENIZER_V1_HISTORICAL,
    TYPED_TOKENIZER_V2_CORRECTED,
)

# Corrected-key wire format.  Bumping this invalidates every corrected cache.
CORRECTED_KEY_PREFIX = b"typedpatchkey/v2;"
CORRECTED_KEY_FIELD_SEP = b"|"


def resolve_typed_tokenizer_version(value: Any) -> str:
    """Normalize and validate a configured tokenizer version string."""
    if value is None:
        return DEFAULT_TYPED_TOKENIZER_VERSION
    version = str(value)
    if version not in SUPPORTED_TYPED_TOKENIZER_VERSIONS:
        raise ValueError(
            f"unknown typed_tokenizer_version={version!r}; expected one of "
            f"{SUPPORTED_TYPED_TOKENIZER_VERSIONS}"
        )
    return version


def typed_tokenizer_fingerprint(version: str, radius: int) -> str:
    """Stable fingerprint identifying a (version, construction) tokenizer.

    Included in cache metadata so a corrected tokenizer can never silently
    reuse a historical cache (or vice versa).
    """
    import hashlib

    version = resolve_typed_tokenizer_version(version)
    payload = f"{version};radius={int(radius)};format={CORRECTED_KEY_PREFIX.decode()}"
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def assert_cache_tokenizer_version(
    metadata: Mapping[str, Any] | None, expected_version: str
) -> None:
    """Refuse to load a cache whose recorded tokenizer version is different.

    ``metadata`` may be ``None`` for caches written before tokenizer
    versioning existed; those must be treated as historical v1 and rejected
    for any corrected run.
    """
    recorded = (metadata or {}).get("typed_tokenizer_version")
    if recorded is None:
        recorded = TYPED_TOKENIZER_V1_HISTORICAL
    if str(recorded) != str(expected_version):
        raise RuntimeError(
            "cache tokenizer version mismatch: cache was written with "
            f"typed_tokenizer_version={recorded!r}, requested {expected_version!r}; "
            "refusing to reuse it (delete/regenerate the cache)"
        )


@dataclass(frozen=True)
class ColoredIncidence:
    """A rooted typed patch as a vertex-colored incidence graph.

    Vertices ``0..n_nodes-1`` are atoms, vertices ``n_nodes..`` are bond
    incidence vertices.  ``color_keys[v]`` is the deterministic semantic color
    of vertex ``v`` (a tuple, never a bare ``int``) -- this is the complete
    typed identity of that vertex.
    """

    number_of_vertices: int
    adjacency: dict[int, list[int]]
    cell_keys: tuple[tuple[Any, ...], ...]
    cell_vertices: tuple[tuple[int, ...], ...]
    color_keys: tuple[tuple[Any, ...], ...]
    n_nodes: int
    n_edges: int
    root_local: int
    original_nodes: tuple[int, ...]


def ego_distances(graph: Any, center: int, radius: int) -> dict[int, int]:
    """BFS distances from ``center`` up to ``radius`` (inclusive)."""
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


def build_colored_incidence(
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    radius: int,
) -> ColoredIncidence:
    """Build the rooted typed patch as a colored incidence graph.

    This is the *single* construction shared by the historical certificate,
    the corrected key and the ground-truth oracle.  Node color:

        ("node", is_root, distance_from_root, atom_type)

    Edge (incidence) color:

        ("edge", bond_type)

    All semantic information therefore lives in the vertex coloring, matching
    the historical ``_typed_certificate`` construction exactly.
    """
    radius = int(radius)
    distances = ego_distances(graph, int(center), radius)
    original_nodes = tuple(sorted(distances))
    node_to_local = {node: index for index, node in enumerate(original_nodes)}
    induced = graph.induced(set(original_nodes))
    local_edges = tuple(
        (node_to_local[int(left)], node_to_local[int(right)])
        for left, right in sorted(induced.edges())
    )
    n_nodes = len(original_nodes)
    n_edges = len(local_edges)
    adjacency: dict[int, list[int]] = {vertex: [] for vertex in range(n_nodes + n_edges)}
    color_of_vertex: dict[int, tuple[Any, ...]] = {}
    root_local = node_to_local[int(center)]
    for local, node in enumerate(original_nodes):
        key = (
            "node",
            int(local == root_local),
            int(distances[node]),
            int(node_types[int(node)]),
        )
        color_of_vertex[local] = key
    for edge_local, (left, right) in enumerate(local_edges):
        edge_vertex = n_nodes + edge_local
        adjacency[left].append(edge_vertex)
        adjacency[right].append(edge_vertex)
        adjacency[edge_vertex] = [left, right]
        bond_type = int(
            edge_types[
                graph.edge_key(int(original_nodes[left]), int(original_nodes[right]))
            ]
        )
        color_of_vertex[edge_vertex] = ("edge", bond_type)

    # Deterministic cell order: sort by repr of the semantic key.  ``repr`` is
    # a total order on the (str/int) tuples and is independent of process hash
    # randomization.
    key_to_vertices: dict[tuple[Any, ...], list[int]] = {}
    for vertex in range(n_nodes + n_edges):
        key_to_vertices.setdefault(color_of_vertex[vertex], []).append(vertex)
    cell_keys = tuple(sorted(key_to_vertices, key=repr))
    cell_vertices = tuple(
        tuple(sorted(key_to_vertices[key])) for key in cell_keys
    )
    color_keys = tuple(color_of_vertex[vertex] for vertex in range(n_nodes + n_edges))
    return ColoredIncidence(
        number_of_vertices=n_nodes + n_edges,
        adjacency=adjacency,
        cell_keys=cell_keys,
        cell_vertices=cell_vertices,
        color_keys=color_keys,
        n_nodes=n_nodes,
        n_edges=n_edges,
        root_local=root_local,
        original_nodes=original_nodes,
    )


def _import_pynauty():
    try:
        import pynauty  # type: ignore
    except ImportError as exc:  # pragma: no cover - dependency diagnostic
        raise RuntimeError("this experiment requires pynauty==2.8.8.1") from exc
    return pynauty


def _pynauty_graph(pynauty, incidence: ColoredIncidence):
    coloring = [set(cell) for cell in incidence.cell_vertices]
    return pynauty.Graph(
        number_of_vertices=int(incidence.number_of_vertices),
        directed=False,
        adjacency_dict={k: list(v) for k, v in incidence.adjacency.items()},
        vertex_coloring=coloring,
    )


def _serialize_color_sequence(sequence: tuple[tuple[Any, ...], ...]) -> bytes:
    return repr(tuple(sequence)).encode("utf-8")


def historical_certificate(incidence: ColoredIncidence) -> bytes:
    """The exact historical token: ``pynauty.certificate`` bytes only."""
    pynauty = _import_pynauty()
    return bytes(pynauty.certificate(_pynauty_graph(pynauty, incidence)))


def corrected_canonical_key(incidence: ColoredIncidence) -> bytes:
    """Complete invariant of the colored incidence graph.

    ``canonical adjacency`` (the historical certificate) **plus** the semantic
    color of each canonically labelled vertex.  The semantic color sequence
    carries color-label identity, so two patches with the same topology but
    different atom/bond/root colors cannot collide.
    """
    pynauty = _import_pynauty()
    graph = _pynauty_graph(pynauty, incidence)
    certificate = bytes(pynauty.certificate(graph))
    canonical = tuple(int(value) for value in pynauty.canon_label(graph))
    if sorted(canonical) != list(range(int(incidence.number_of_vertices))):
        raise RuntimeError(
            f"pynauty.canon_label is not a permutation: {canonical!r}"
        )
    color_sequence = tuple(incidence.color_keys[vertex] for vertex in canonical)
    return (
        CORRECTED_KEY_PREFIX
        + certificate
        + CORRECTED_KEY_FIELD_SEP
        + _serialize_color_sequence(color_sequence)
    )


def typed_patch_key(
    version: str,
    graph: Any,
    center: int,
    node_types: np.ndarray,
    edge_types: Mapping[tuple[int, int], int],
    radius: int,
    cache: dict[bytes, bytes] | None = None,
    cache_key: bytes | None = None,
) -> bytes:
    """Versioned typed patch key.

    ``cache``/``cache_key`` are optional and let callers (the legacy
    experiment) reuse its existing per-(raw subgraph) memo without changing
    compute structure.
    """
    version = resolve_typed_tokenizer_version(version)
    if cache is not None and cache_key is not None:
        cached = cache.get(cache_key)
        if cached is not None:
            return cached
    incidence = build_colored_incidence(
        graph, int(center), node_types, edge_types, int(radius)
    )
    if version == TYPED_TOKENIZER_V1_HISTORICAL:
        key = historical_certificate(incidence)
    elif version == TYPED_TOKENIZER_V2_CORRECTED:
        key = corrected_canonical_key(incidence)
    else:  # pragma: no cover - resolve() guarantees membership
        raise AssertionError(version)
    if cache is not None and cache_key is not None:
        cache[cache_key] = key
    return key


# --------------------------------------------------------------------------
# Ground-truth oracle (reference only; never on the training path)
# --------------------------------------------------------------------------
def oracle_partition_signature(incidence: ColoredIncidence) -> tuple[tuple[Any, ...], ...]:
    """Ordered semantic color labels aligned with the ordered color cells."""
    return incidence.cell_keys


def typed_isomorphic(
    left: ColoredIncidence, right: ColoredIncidence
) -> bool:
    """Reference colored-incidence isomorphism oracle.

    Two patches are equivalent iff there is a graph isomorphism of the
    incidence graphs preserving every semantic color label (atom type, bond
    type, root designation, root-distance class, node-vs-edge incidence type).

    Implementation: cells are ordered by their semantic label; equality of the
    ordered label sequence is checked explicitly (color-label identity, which
    bare ``pynauty.isomorphic`` does not test), then ``pynauty.isomorphic``
    checks the permutation-invariant structure.  Validated against brute-force
    labeled color-preserving isomorphism on random small colored graphs.
    """
    if left.number_of_vertices != right.number_of_vertices:
        return False
    if int(left.n_nodes) != int(right.n_nodes) or int(left.n_edges) != int(right.n_edges):
        return False
    if oracle_partition_signature(left) != oracle_partition_signature(right):
        return False
    pynauty = _import_pynauty()
    return bool(
        pynauty.isomorphic(
            _pynauty_graph(pynauty, left), _pynauty_graph(pynauty, right)
        )
    )


def incidence_to_networkx(incidence: ColoredIncidence):
    """Convert a colored incidence graph to a ``networkx.Graph``.

    Vertex attribute ``color`` holds the semantic color tuple.  Graph edges are
    untyped (all incidence relations); bond type lives on the edge vertex.
    """
    import networkx as nx

    graph = nx.Graph()
    for vertex in range(int(incidence.number_of_vertices)):
        graph.add_node(vertex, color=incidence.color_keys[vertex])
    for left, neighbors in incidence.adjacency.items():
        for right in neighbors:
            if int(left) < int(right):
                graph.add_edge(int(left), int(right))
    return graph


def typed_isomorphic_vf2(left: ColoredIncidence, right: ColoredIncidence) -> bool:
    """Independent ground-truth oracle (VF2, networkx).

    Uses a completely different implementation from ``pynauty``: exact
    color-preserving subgraph isomorphism via networkx's VF2 with a node
    matcher on the full semantic color tuple.  Validated against brute-force
    labeled color-preserving isomorphism on random small colored graphs (see
    ``tests/test_typed_patch_tokenizer.py``).
    """
    import networkx as nx

    if int(left.number_of_vertices) != int(right.number_of_vertices):
        return False
    if int(left.n_nodes) != int(right.n_nodes) or int(left.n_edges) != int(right.n_edges):
        return False
    return bool(
        nx.is_isomorphic(
            incidence_to_networkx(left),
            incidence_to_networkx(right),
            node_match=lambda a, b: a["color"] == b["color"],
        )
    )


def typed_isomorphic_patches(
    left_graph: Any,
    left_center: int,
    left_node_types: np.ndarray,
    left_edge_types: Mapping[tuple[int, int], int],
    right_graph: Any,
    right_center: int,
    right_node_types: np.ndarray,
    right_edge_types: Mapping[tuple[int, int], int],
    radius: int,
) -> bool:
    """Convenience wrapper around :func:`typed_isomorphic` for two patches."""
    radius = int(radius)
    left = build_colored_incidence(
        left_graph, int(left_center), left_node_types, left_edge_types, radius
    )
    right = build_colored_incidence(
        right_graph, int(right_center), right_node_types, right_edge_types, radius
    )
    return typed_isomorphic(left, right)

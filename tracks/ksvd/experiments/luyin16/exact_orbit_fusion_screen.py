"""Train-only MolHIV screen for exact topology--attribute binding.

The experiment treats structure and chemistry as two parts of one rooted
radius-2 object instead of two already-pooled modalities:

* structure is the exact rooted untyped topology, certified by nauty;
* node and edge positions are rooted automorphism orbits of that topology;
* chemistry is averaged only inside the corresponding node/edge orbit;
* a matched within-patch shuffle preserves topology and attribute marginals
  while breaking which chemistry occupies which structural orbit.

Two candidates are evaluated.  ``orbit_raw`` is a deliberately generous
high-dimensional upper bound.  ``prototype`` is a compact deterministic bank:
each frequent topology owns a few farthest-point empirical chemistry templates,
and a graph reads out occurrence-weighted and max similarities to those
topology-conditioned templates.  No K-SVD update and no random dictionary
initialisation are used.

Only official-train scaffold folds are used for supervised scores.  Exact
topology vocabularies and prototype banks are independently fit on each outer
training fold.  Official validation/test examples are neither encoded nor
evaluated by this protocol.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict, deque
from dataclasses import dataclass
import hashlib
import platform
from pathlib import Path
import time
from typing import Any, Mapping, Sequence

import numpy as np
import yaml

from ksvd_research.data import load_molhiv
from tracks.ksvd.experiments.luyin16.patch_object_audit import (
    relabel_graph_features,
)
from tracks.ksvd.experiments.luyin16.role_attribute_binding_screen import (
    REPO_ROOT,
    _aggregate_folds,
    _delta,
    _delta_against_controls,
    _fit_auc_views,
    _fold_indices,
    _frozen_s_rows,
    _resolve,
    _sha256,
    _write_json,
    compact_bond_semantics,
)
from tracks.ksvd.experiments.luyin16.structural_role_fusion_screen import (
    BOND_DIM,
    STRICT_ATOM_DIM,
    patch_roles,
    strict_atom_semantics,
)


DEFAULT_CONFIG = (
    REPO_ROOT / "tracks/ksvd/configs/luyin16/exact_orbit_fusion_screen.yaml"
)
ATOM_FIELD_MASS = 7.0
BOND_FIELD_MASS = 3.0


@dataclass(frozen=True)
class ExactOrbitPatch:
    topology_key: bytes
    n_node_orbits: int
    n_edge_orbits: int
    template: np.ndarray
    shuffled_templates: tuple[np.ndarray, ...]
    node_attribute: np.ndarray
    edge_attribute: np.ndarray
    wl_key: int
    wl_binned_key: bytes
    n_nodes: int
    n_edges: int


@dataclass(frozen=True)
class GraphPatchData:
    patches: tuple[ExactOrbitPatch, ...]
    wl_structure: np.ndarray
    context: np.ndarray


@dataclass(frozen=True)
class TopologySpec:
    index: int
    key: bytes
    count: int
    n_node_orbits: int
    n_edge_orbits: int
    raw_start: int
    raw_stop: int


@dataclass(frozen=True)
class TopologyVocabulary:
    specs: tuple[TopologySpec, ...]
    lookup: Mapping[bytes, TopologySpec]
    raw_width: int
    total_train_patches: int


@dataclass(frozen=True)
class PrototypeBank:
    values: Mapping[bytes, np.ndarray]
    offsets: Mapping[bytes, int]
    total_prototypes: int


def _ego_distances(graph, center: int, radius: int) -> dict[int, int]:
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
    groups: dict[int, list[int]] = defaultdict(list)
    for vertex in vertices:
        groups[int(orbit_ids[int(vertex)])].append(int(vertex))
    ordered = sorted(
        groups.values(),
        key=lambda group: tuple(sorted(canonical_position[vertex] for vertex in group)),
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


def _lexicographic_order(rows: np.ndarray) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2:
        raise ValueError("lexicographic order expects a matrix")
    if values.shape[0] <= 1:
        return np.arange(values.shape[0], dtype=np.int64)
    keys = tuple(
        values[:, column] for column in range(values.shape[1] - 1, -1, -1)
    )
    return np.asarray(np.lexsort(keys), dtype=np.int64)


def _shuffle_entity_attributes(
    attributes: np.ndarray,
    canonical_entity_order: np.ndarray,
    topology_key: bytes,
    *,
    base_seed: int,
    repeat: int,
    modality: str,
) -> np.ndarray:
    """Invariant matched shuffle of one patch's node or edge attributes."""
    values = np.asarray(attributes, dtype=np.float32)
    order = np.asarray(canonical_entity_order, dtype=np.int64)
    if values.ndim != 2 or order.shape != (values.shape[0],):
        raise ValueError("attribute rows and canonical order are incompatible")
    if values.shape[0] <= 1:
        return values.copy()
    sorted_values = values[_lexicographic_order(values)]
    digest = hashlib.blake2b(digest_size=8)
    digest.update(b"exact-orbit-matched-shuffle-v1")
    digest.update(topology_key)
    digest.update(str((int(base_seed), int(repeat), str(modality))).encode("utf-8"))
    digest.update(np.asarray(sorted_values.shape, dtype=np.int64).tobytes())
    digest.update(sorted_values.tobytes())
    rng = np.random.default_rng(int.from_bytes(digest.digest(), "little"))
    assigned = sorted_values[rng.permutation(sorted_values.shape[0])]
    output = np.empty_like(values)
    output[order] = assigned
    if not np.array_equal(values.sum(axis=0), output.sum(axis=0)):
        raise RuntimeError("matched shuffle changed an attribute marginal")
    return output


def _orbit_means(
    attributes: np.ndarray,
    orbit_ids: np.ndarray,
    n_orbits: int,
) -> np.ndarray:
    values = np.asarray(attributes, dtype=np.float32)
    ids = np.asarray(orbit_ids, dtype=np.int64)
    if values.ndim != 2 or ids.shape != (values.shape[0],):
        raise ValueError("orbit ids and attributes are not aligned")
    output = np.zeros((int(n_orbits), values.shape[1]), dtype=np.float32)
    if values.shape[0] == 0:
        return output
    np.add.at(output, ids, values)
    counts = np.bincount(ids, minlength=int(n_orbits)).astype(np.float32)
    output /= np.maximum(counts[:, None], 1.0)
    return output


def _template(
    node_attributes: np.ndarray,
    node_orbits: np.ndarray,
    n_node_orbits: int,
    edge_attributes: np.ndarray,
    edge_orbits: np.ndarray,
    n_edge_orbits: int,
) -> np.ndarray:
    node = _orbit_means(node_attributes, node_orbits, n_node_orbits)
    edge = _orbit_means(edge_attributes, edge_orbits, n_edge_orbits)
    return np.concatenate(
        [node.reshape(-1) / ATOM_FIELD_MASS, edge.reshape(-1) / BOND_FIELD_MASS]
    ).astype(np.float32, copy=False)


def _exact_orbit_patch(
    graph,
    center: int,
    atom_semantics: np.ndarray,
    bond_semantics: Mapping[tuple[int, int], np.ndarray],
    representation: Mapping[str, Any],
    *,
    shuffle_repeats: int,
    shuffle_seed: int,
) -> ExactOrbitPatch:
    try:
        import pynauty
    except ImportError as exc:  # pragma: no cover - environment diagnostic
        raise ImportError("exact orbit fusion requires pynauty==2.8.8.1") from exc

    radius = int(representation["radius"])
    distances = _ego_distances(graph, int(center), radius)
    original_nodes = tuple(sorted(distances))
    node_to_local = {node: index for index, node in enumerate(original_nodes)}
    induced = graph.induced(set(original_nodes))
    original_edges = tuple(sorted(induced.edges()))
    local_edges = tuple(
        (node_to_local[int(left)], node_to_local[int(right)])
        for left, right in original_edges
    )
    n_nodes = len(original_nodes)
    n_edges = len(local_edges)
    adjacency: dict[int, list[int]] = {
        vertex: [] for vertex in range(n_nodes + n_edges)
    }
    for edge_index, (left, right) in enumerate(local_edges):
        edge_vertex = n_nodes + edge_index
        adjacency[left].append(edge_vertex)
        adjacency[right].append(edge_vertex)
        adjacency[edge_vertex] = [left, right]
    root = node_to_local[int(center)]
    coloring: list[set[int]] = [{root}]
    other_nodes = set(range(n_nodes)) - {root}
    if other_nodes:
        coloring.append(other_nodes)
    if n_edges:
        coloring.append(set(range(n_nodes, n_nodes + n_edges)))
    nauty_graph = pynauty.Graph(
        number_of_vertices=n_nodes + n_edges,
        directed=False,
        adjacency_dict=adjacency,
        vertex_coloring=coloring,
    )
    topology_key = bytes(pynauty.certificate(nauty_graph))
    canonical = tuple(int(value) for value in pynauty.canon_label(nauty_graph))
    canonical_position = {vertex: position for position, vertex in enumerate(canonical)}
    _generators, _size1, _size2, raw_orbits, _n_orbits = pynauty.autgrp(
        nauty_graph
    )
    node_vertices = tuple(range(n_nodes))
    edge_vertices = tuple(range(n_nodes, n_nodes + n_edges))
    node_orbits, n_node_orbits = _canonical_orbit_ids(
        node_vertices, raw_orbits, canonical_position
    )
    edge_orbits, n_edge_orbits = _canonical_orbit_ids(
        edge_vertices, raw_orbits, canonical_position
    )
    node_canonical_order = np.asarray(
        sorted(node_vertices, key=canonical_position.__getitem__), dtype=np.int64
    )
    edge_canonical_order = np.asarray(
        [
            vertex - n_nodes
            for vertex in sorted(edge_vertices, key=canonical_position.__getitem__)
        ],
        dtype=np.int64,
    )
    node_attributes = np.stack(
        [atom_semantics[int(node)] for node in original_nodes], axis=0
    ).astype(np.float32, copy=False)
    if original_edges:
        edge_attributes = np.stack(
            [
                bond_semantics[graph.edge_key(int(left), int(right))]
                for left, right in original_edges
            ],
            axis=0,
        ).astype(np.float32, copy=False)
    else:
        edge_attributes = np.zeros((0, BOND_DIM), dtype=np.float32)
    true_template = _template(
        node_attributes,
        node_orbits,
        n_node_orbits,
        edge_attributes,
        edge_orbits,
        n_edge_orbits,
    )
    shuffled_templates: list[np.ndarray] = []
    for repeat in range(int(shuffle_repeats)):
        shuffled_nodes = _shuffle_entity_attributes(
            node_attributes,
            node_canonical_order,
            topology_key,
            base_seed=int(shuffle_seed),
            repeat=repeat,
            modality="node",
        )
        shuffled_edges = _shuffle_entity_attributes(
            edge_attributes,
            edge_canonical_order,
            topology_key,
            base_seed=int(shuffle_seed),
            repeat=repeat,
            modality="edge",
        )
        shuffled_templates.append(
            _template(
                shuffled_nodes,
                node_orbits,
                n_node_orbits,
                shuffled_edges,
                edge_orbits,
                n_edge_orbits,
            )
        )

    roles = patch_roles(graph, int(center), "rooted_wl", representation)
    node_wl = np.bincount(
        roles.node_ids, minlength=int(representation["node_role_bins"])
    ).astype(np.float32)
    node_wl /= max(float(roles.node_ids.size), 1.0)
    edge_wl = np.bincount(
        roles.edge_ids, minlength=int(representation["edge_role_bins"])
    ).astype(np.float32)
    edge_wl /= max(float(roles.edge_ids.size), 1.0)
    wl_vector = np.concatenate([node_wl, edge_wl]).astype(np.float32, copy=False)
    wl_binned_key = hashlib.sha256(wl_vector.tobytes()).digest()
    return ExactOrbitPatch(
        topology_key=topology_key,
        n_node_orbits=int(n_node_orbits),
        n_edge_orbits=int(n_edge_orbits),
        template=true_template,
        shuffled_templates=tuple(shuffled_templates),
        node_attribute=node_attributes.mean(axis=0, dtype=np.float32),
        edge_attribute=(
            edge_attributes.mean(axis=0, dtype=np.float32)
            if edge_attributes.shape[0]
            else np.zeros(BOND_DIM, dtype=np.float32)
        ),
        wl_key=int(roles.patch_key),
        wl_binned_key=wl_binned_key,
        n_nodes=n_nodes,
        n_edges=n_edges,
    )


def _graph_context(graph, patches: Sequence[ExactOrbitPatch]) -> np.ndarray:
    return np.asarray(
        [
            np.log1p(float(graph.n)),
            np.log1p(float(graph.num_edges())),
            np.log1p(float(len(patches))),
            float(np.mean([patch.n_nodes for patch in patches])) if patches else 0.0,
            float(np.mean([patch.n_edges for patch in patches])) if patches else 0.0,
        ],
        dtype=np.float32,
    )


def _extract_graph_data(
    graph,
    node_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    representation: Mapping[str, Any],
    *,
    shuffle_repeats: int,
    shuffle_seed: int,
) -> GraphPatchData:
    if representation.get("max_centers_per_graph") is not None:
        raise ValueError("exact orbit screen requires all graph nodes as centres")
    atom_semantics, _atom_groups = strict_atom_semantics(node_features)
    bond_semantics = {
        graph.edge_key(int(left), int(right)): compact_bond_semantics(values)
        for (left, right), values in edge_features.items()
    }
    patches = tuple(
        _exact_orbit_patch(
            graph,
            int(center),
            atom_semantics,
            bond_semantics,
            representation,
            shuffle_repeats=int(shuffle_repeats),
            shuffle_seed=int(shuffle_seed),
        )
        for center in graph.nodes
    )
    wl_width = int(representation["node_role_bins"]) + int(
        representation["edge_role_bins"]
    )
    if patches:
        wl_rows = []
        # Recompute the two topology-only role histograms once for the graph
        # readout.  The per-patch digest is retained only for collision audits.
        for center in graph.nodes:
            roles = patch_roles(graph, int(center), "rooted_wl", representation)
            node = np.bincount(
                roles.node_ids, minlength=int(representation["node_role_bins"])
            ).astype(np.float32)
            node /= max(float(roles.node_ids.size), 1.0)
            edge = np.bincount(
                roles.edge_ids, minlength=int(representation["edge_role_bins"])
            ).astype(np.float32)
            edge /= max(float(roles.edge_ids.size), 1.0)
            wl_rows.append(np.concatenate([node, edge]))
        wl_structure = np.mean(
            np.stack(wl_rows, axis=0), axis=0, dtype=np.float32
        )
    else:
        wl_structure = np.zeros(wl_width, dtype=np.float32)
    return GraphPatchData(
        patches=patches,
        wl_structure=np.asarray(wl_structure, dtype=np.float32),
        context=_graph_context(graph, patches),
    )


def _record_token(record: ExactOrbitPatch) -> bytes:
    digest = hashlib.sha256()
    digest.update(record.topology_key)
    digest.update(
        np.asarray(
            [record.n_node_orbits, record.n_edge_orbits, record.n_nodes, record.n_edges],
            dtype=np.int64,
        ).tobytes()
    )
    digest.update(record.template.tobytes())
    for shuffled in record.shuffled_templates:
        digest.update(shuffled.tobytes())
    digest.update(record.node_attribute.tobytes())
    digest.update(record.edge_attribute.tobytes())
    digest.update(int(record.wl_key).to_bytes(8, "little", signed=False))
    digest.update(record.wl_binned_key)
    return digest.digest()


def audit_invariance(
    bundle,
    indices: np.ndarray,
    representation: Mapping[str, Any],
    audit: Mapping[str, Any],
    *,
    shuffle_repeats: int,
) -> dict[str, Any]:
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise ValueError("exact orbit audit requires node and edge features")
    rng = np.random.default_rng(int(audit["seed"]))
    candidates = np.asarray(indices, dtype=np.int64)
    if candidates.size > int(audit["n_graphs"]):
        candidates = np.sort(
            rng.choice(candidates, size=int(audit["n_graphs"]), replace=False)
        )
    multiset_matches: list[float] = []
    wl_drifts: list[float] = []
    context_drifts: list[float] = []
    for raw_index in candidates:
        index = int(raw_index)
        base = _extract_graph_data(
            bundle.graphs[index],
            bundle.node_feats[index],
            bundle.edge_feats[index],
            representation,
            shuffle_repeats=int(shuffle_repeats),
            shuffle_seed=int(audit["shuffle_seed"]),
        )
        base_tokens = Counter(_record_token(record) for record in base.patches)
        for _ in range(int(audit["permutations_per_graph"])):
            permutation = rng.permutation(bundle.graphs[index].n)
            changed_graph, changed_nodes, changed_edges = relabel_graph_features(
                bundle.graphs[index],
                bundle.node_feats[index],
                bundle.edge_feats[index],
                permutation,
            )
            changed = _extract_graph_data(
                changed_graph,
                changed_nodes,
                changed_edges,
                representation,
                shuffle_repeats=int(shuffle_repeats),
                shuffle_seed=int(audit["shuffle_seed"]),
            )
            changed_tokens = Counter(_record_token(record) for record in changed.patches)
            multiset_matches.append(float(base_tokens == changed_tokens))
            wl_drifts.append(
                float(
                    np.max(
                        np.abs(base.wl_structure - changed.wl_structure), initial=0.0
                    )
                )
            )
            context_drifts.append(
                float(np.max(np.abs(base.context - changed.context), initial=0.0))
            )
    tolerance = float(audit["tolerance"])
    return {
        "n_graphs": int(candidates.size),
        "permutations_per_graph": int(audit["permutations_per_graph"]),
        "patch_multiset_match_rate": float(np.mean(multiset_matches))
        if multiset_matches
        else 1.0,
        "maximum_wl_drift": max(wl_drifts, default=0.0),
        "maximum_context_drift": max(context_drifts, default=0.0),
        "tolerance": tolerance,
        "pass": bool(
            all(value == 1.0 for value in multiset_matches)
            and max(wl_drifts, default=0.0) <= tolerance
            and max(context_drifts, default=0.0) <= tolerance
        ),
    }


def _fit_vocabulary(
    train_graphs: Sequence[GraphPatchData], maximum_types: int
) -> TopologyVocabulary:
    counts: Counter[bytes] = Counter()
    metadata: dict[bytes, tuple[int, int, int]] = {}
    for graph in train_graphs:
        for patch in graph.patches:
            counts[patch.topology_key] += 1
            current = (
                int(patch.n_node_orbits),
                int(patch.n_edge_orbits),
                int(patch.template.shape[0]),
            )
            previous = metadata.setdefault(patch.topology_key, current)
            if previous != current:
                raise RuntimeError("one exact topology produced incompatible orbit layouts")
    ordered = sorted(counts, key=lambda key: (-counts[key], key))[: int(maximum_types)]
    specs: list[TopologySpec] = []
    start = 0
    for index, key in enumerate(ordered):
        n_node_orbits, n_edge_orbits, width = metadata[key]
        expected = n_node_orbits * STRICT_ATOM_DIM + n_edge_orbits * BOND_DIM
        if width != expected:
            raise RuntimeError(f"orbit template width changed: {width} != {expected}")
        specs.append(
            TopologySpec(
                index=index,
                key=key,
                count=int(counts[key]),
                n_node_orbits=n_node_orbits,
                n_edge_orbits=n_edge_orbits,
                raw_start=start,
                raw_stop=start + width,
            )
        )
        start += width
    return TopologyVocabulary(
        specs=tuple(specs),
        lookup={spec.key: spec for spec in specs},
        raw_width=start,
        total_train_patches=int(sum(counts.values())),
    )


def _normalise_template(row: np.ndarray) -> np.ndarray:
    values = np.asarray(row, dtype=np.float32)
    norm = float(np.linalg.norm(values))
    return values / max(norm, 1.0e-12)


def _deterministic_farthest_prototypes(
    rows: np.ndarray, maximum: int
) -> np.ndarray:
    values = np.asarray(rows, dtype=np.float32)
    if values.ndim != 2 or values.shape[0] == 0:
        raise ValueError("prototype selection needs a non-empty matrix")
    unique = np.unique(values, axis=0)
    normalised = np.stack([_normalise_template(row) for row in unique], axis=0)
    centre = normalised.mean(axis=0, dtype=np.float32)
    first = int(np.argmin(np.sum((normalised - centre) ** 2, axis=1)))
    selected = [first]
    minimum_distance = 1.0 - np.clip(normalised @ normalised[first], -1.0, 1.0)
    while len(selected) < min(int(maximum), normalised.shape[0]):
        candidate = int(np.argmax(minimum_distance))
        if float(minimum_distance[candidate]) <= 1.0e-8:
            break
        selected.append(candidate)
        distance = 1.0 - np.clip(
            normalised @ normalised[candidate], -1.0, 1.0
        )
        minimum_distance = np.minimum(minimum_distance, distance)
    return normalised[np.asarray(selected, dtype=np.int64)].astype(
        np.float32, copy=False
    )


def _fit_prototype_bank(
    train_graphs: Sequence[GraphPatchData],
    vocabulary: TopologyVocabulary,
    maximum_per_topology: int,
) -> PrototypeBank:
    rows: dict[bytes, list[np.ndarray]] = {
        spec.key: [] for spec in vocabulary.specs
    }
    for graph in train_graphs:
        for patch in graph.patches:
            if patch.topology_key in rows:
                rows[patch.topology_key].append(patch.template)
    values: dict[bytes, np.ndarray] = {}
    offsets: dict[bytes, int] = {}
    offset = 0
    for spec in vocabulary.specs:
        matrix = np.stack(rows[spec.key], axis=0).astype(np.float32, copy=False)
        selected = _deterministic_farthest_prototypes(
            matrix, int(maximum_per_topology)
        )
        values[spec.key] = selected
        offsets[spec.key] = offset
        offset += int(selected.shape[0])
    return PrototypeBank(values=values, offsets=offsets, total_prototypes=offset)


def _patch_attribute_row(patch: ExactOrbitPatch) -> np.ndarray:
    """Fixed chemistry multiset descriptor, independent of orbit placement."""
    return np.concatenate(
        [
            patch.node_attribute / ATOM_FIELD_MASS,
            patch.edge_attribute / BOND_FIELD_MASS,
        ]
    ).astype(np.float32, copy=False)


def _fit_conditional_prototype_bank(
    train_graphs: Sequence[GraphPatchData],
    vocabulary: TopologyVocabulary,
    maximum_per_topology: int,
) -> PrototypeBank:
    """Fit chemistry-multiset prototypes separately inside each topology."""
    rows: dict[bytes, list[np.ndarray]] = {
        spec.key: [] for spec in vocabulary.specs
    }
    for graph in train_graphs:
        for patch in graph.patches:
            if patch.topology_key in rows:
                rows[patch.topology_key].append(_patch_attribute_row(patch))
    values: dict[bytes, np.ndarray] = {}
    offsets: dict[bytes, int] = {}
    offset = 0
    for spec in vocabulary.specs:
        matrix = np.stack(rows[spec.key], axis=0).astype(np.float32, copy=False)
        selected = _deterministic_farthest_prototypes(
            matrix, int(maximum_per_topology)
        )
        values[spec.key] = selected
        offsets[spec.key] = offset
        offset += int(selected.shape[0])
    return PrototypeBank(values=values, offsets=offsets, total_prototypes=offset)


def _cross_patch_attribute_pairs(
    graph: GraphPatchData,
    *,
    base_seed: int,
    repeat: int,
) -> tuple[tuple[ExactOrbitPatch, np.ndarray], ...]:
    """Break topology--chemistry patch pairing while preserving both bags.

    Topologies and chemistry rows are independently put in invariant sorted
    orders before applying a content-derived fixed permutation.  Ties between
    identical topology keys are harmless because they use the same conditional
    prototype bank and are aggregated as a bag.
    """
    if not graph.patches:
        return ()
    ordered_patches = tuple(
        sorted(graph.patches, key=lambda patch: patch.topology_key)
    )
    attributes = np.stack(
        [_patch_attribute_row(patch) for patch in graph.patches], axis=0
    ).astype(np.float32, copy=False)
    sorted_attributes = attributes[_lexicographic_order(attributes)]
    digest = hashlib.blake2b(digest_size=8)
    digest.update(b"exact-topology-attribute-patch-pair-shuffle-v1")
    digest.update(str((int(base_seed), int(repeat))).encode("utf-8"))
    for patch in ordered_patches:
        digest.update(len(patch.topology_key).to_bytes(4, "big"))
        digest.update(patch.topology_key)
    digest.update(sorted_attributes.tobytes())
    rng = np.random.default_rng(int.from_bytes(digest.digest(), "little"))
    shuffled_attributes = sorted_attributes[
        rng.permutation(sorted_attributes.shape[0])
    ]
    if not np.array_equal(
        np.sort(attributes, axis=0), np.sort(shuffled_attributes, axis=0)
    ):
        raise RuntimeError("cross-patch shuffle changed the attribute-row bag")
    return tuple(zip(ordered_patches, shuffled_attributes, strict=True))


def _prototype_response(
    pairs: Sequence[tuple[ExactOrbitPatch, np.ndarray]],
    bank: PrototypeBank,
    n_patches: int,
    *,
    normalization: str = "graph",
) -> np.ndarray:
    if normalization not in {"graph", "topology"}:
        raise ValueError(f"unknown prototype response normalization: {normalization}")
    output = np.zeros(2 * bank.total_prototypes, dtype=np.float32)
    topology_counts = Counter(
        patch.topology_key
        for patch, _row in pairs
        if patch.topology_key in bank.values
    )
    for patch, row in pairs:
        prototypes = bank.values.get(patch.topology_key)
        if prototypes is None:
            continue
        denominator = (
            max(int(n_patches), 1)
            if normalization == "graph"
            else max(int(topology_counts[patch.topology_key]), 1)
        )
        offset = int(bank.offsets[patch.topology_key])
        similarity = np.clip(prototypes @ _normalise_template(row), 0.0, 1.0)
        output[offset : offset + prototypes.shape[0]] += similarity / float(
            denominator
        )
        maximum = bank.total_prototypes + offset
        output[maximum : maximum + prototypes.shape[0]] = np.maximum(
            output[maximum : maximum + prototypes.shape[0]], similarity
        )
    return output


def _graph_feature_blocks(
    graph: GraphPatchData,
    vocabulary: TopologyVocabulary,
    prototypes: PrototypeBank,
    conditional_prototypes: PrototypeBank,
    *,
    shuffle_repeats: int,
    patch_pair_shuffle_seed: int,
) -> dict[str, np.ndarray]:
    n_patches = max(len(graph.patches), 1)
    exact = np.zeros(len(vocabulary.specs) + 1, dtype=np.float32)
    attributes = np.zeros(STRICT_ATOM_DIM + BOND_DIM, dtype=np.float32)
    orbit_raw = np.zeros(vocabulary.raw_width, dtype=np.float32)
    shuffled_raw = [
        np.zeros(vocabulary.raw_width, dtype=np.float32)
        for _ in range(int(shuffle_repeats))
    ]
    prototype_width = 2 * prototypes.total_prototypes
    prototype = np.zeros(prototype_width, dtype=np.float32)
    shuffled_prototypes = [
        np.zeros(prototype_width, dtype=np.float32)
        for _ in range(int(shuffle_repeats))
    ]
    conditional_pairs = tuple(
        (patch, _patch_attribute_row(patch)) for patch in graph.patches
    )
    conditional = _prototype_response(
        conditional_pairs, conditional_prototypes, n_patches
    )
    conditional_mean = _prototype_response(
        conditional_pairs,
        conditional_prototypes,
        n_patches,
        normalization="topology",
    )
    conditional_shuffled = [
        _prototype_response(
            _cross_patch_attribute_pairs(
                graph,
                base_seed=int(patch_pair_shuffle_seed),
                repeat=repeat,
            ),
            conditional_prototypes,
            n_patches,
        )
        for repeat in range(int(shuffle_repeats))
    ]
    conditional_mean_shuffled = [
        _prototype_response(
            _cross_patch_attribute_pairs(
                graph,
                base_seed=int(patch_pair_shuffle_seed),
                repeat=repeat,
            ),
            conditional_prototypes,
            n_patches,
            normalization="topology",
        )
        for repeat in range(int(shuffle_repeats))
    ]
    for patch in graph.patches:
        attributes[:STRICT_ATOM_DIM] += patch.node_attribute / float(n_patches)
        attributes[STRICT_ATOM_DIM:] += patch.edge_attribute / float(n_patches)
        spec = vocabulary.lookup.get(patch.topology_key)
        if spec is None:
            exact[-1] += 1.0 / float(n_patches)
            continue
        exact[spec.index] += 1.0 / float(n_patches)
        orbit_raw[spec.raw_start : spec.raw_stop] += patch.template / float(
            n_patches
        )
        bank = prototypes.values[patch.topology_key]
        offset = int(prototypes.offsets[patch.topology_key])
        normalised = _normalise_template(patch.template)
        similarity = np.clip(bank @ normalised, 0.0, 1.0)
        prototype[offset : offset + bank.shape[0]] += similarity / float(n_patches)
        maximum = prototypes.total_prototypes + offset
        prototype[maximum : maximum + bank.shape[0]] = np.maximum(
            prototype[maximum : maximum + bank.shape[0]], similarity
        )
        for repeat in range(int(shuffle_repeats)):
            shuffled = patch.shuffled_templates[repeat]
            shuffled_raw[repeat][spec.raw_start : spec.raw_stop] += shuffled / float(
                n_patches
            )
            shuffled_normalised = _normalise_template(shuffled)
            shuffled_similarity = np.clip(bank @ shuffled_normalised, 0.0, 1.0)
            shuffled_prototypes[repeat][
                offset : offset + bank.shape[0]
            ] += shuffled_similarity / float(n_patches)
            shuffled_prototypes[repeat][
                maximum : maximum + bank.shape[0]
            ] = np.maximum(
                shuffled_prototypes[repeat][maximum : maximum + bank.shape[0]],
                shuffled_similarity,
            )
    output = {
        "wl_structure": graph.wl_structure,
        "exact_structure": exact,
        "attribute": attributes,
        "orbit_raw": orbit_raw,
        "prototype": prototype,
        "conditional": conditional,
        "conditional_mean": conditional_mean,
        "context": graph.context,
    }
    for repeat in range(int(shuffle_repeats)):
        output[f"orbit_raw_shuffled_{repeat}"] = shuffled_raw[repeat]
        output[f"prototype_shuffled_{repeat}"] = shuffled_prototypes[repeat]
        output[f"conditional_shuffled_{repeat}"] = conditional_shuffled[repeat]
        output[f"conditional_mean_shuffled_{repeat}"] = (
            conditional_mean_shuffled[repeat]
        )
    return output


def _stack_blocks(rows: Sequence[Mapping[str, np.ndarray]]) -> dict[str, np.ndarray]:
    names = tuple(rows[0])
    return {
        name: np.stack([np.asarray(row[name], dtype=np.float32) for row in rows], axis=0)
        for name in names
    }


def _join(*parts: np.ndarray) -> np.ndarray:
    return np.concatenate(parts, axis=1).astype(np.float32, copy=False)


def _assemble_views(
    frozen_s: np.ndarray,
    blocks: Mapping[str, np.ndarray],
    *,
    shuffle_repeats: int,
) -> dict[str, np.ndarray]:
    s = np.asarray(frozen_s, dtype=np.float32)
    wl = blocks["wl_structure"]
    exact = blocks["exact_structure"]
    attribute = blocks["attribute"]
    context = blocks["context"]
    views = {
        "s": s,
        "s_wl_structure": _join(s, wl, context),
        "s_exact_structure": _join(s, exact, context),
        "s_attribute": _join(s, attribute, context),
        "s_wl_unbound": _join(s, wl, attribute, context),
        "s_wl_conditional": _join(
            s, wl, attribute, blocks["conditional"], context
        ),
        "s_wl_conditional_mean": _join(
            s, wl, attribute, blocks["conditional_mean"], context
        ),
        "s_exact_unbound": _join(s, exact, attribute, context),
        "s_exact_orbit_raw": _join(
            s, exact, attribute, blocks["orbit_raw"], context
        ),
        "s_exact_prototype": _join(
            s, exact, attribute, blocks["prototype"], context
        ),
        "s_exact_conditional": _join(
            s, exact, attribute, blocks["conditional"], context
        ),
        "s_hybrid_unbound": _join(s, wl, exact, attribute, context),
        "s_hybrid_prototype": _join(
            s, wl, exact, attribute, blocks["prototype"], context
        ),
        "s_hybrid_conditional": _join(
            s, wl, exact, attribute, blocks["conditional"], context
        ),
        "s_hybrid_conditional_mean": _join(
            s, wl, exact, attribute, blocks["conditional_mean"], context
        ),
    }
    for repeat in range(int(shuffle_repeats)):
        views[f"s_exact_orbit_raw_shuffled_{repeat}"] = _join(
            s,
            exact,
            attribute,
            blocks[f"orbit_raw_shuffled_{repeat}"],
            context,
        )
        views[f"s_exact_prototype_shuffled_{repeat}"] = _join(
            s,
            exact,
            attribute,
            blocks[f"prototype_shuffled_{repeat}"],
            context,
        )
        views[f"s_hybrid_prototype_shuffled_{repeat}"] = _join(
            s,
            wl,
            exact,
            attribute,
            blocks[f"prototype_shuffled_{repeat}"],
            context,
        )
        views[f"s_exact_conditional_shuffled_{repeat}"] = _join(
            s,
            exact,
            attribute,
            blocks[f"conditional_shuffled_{repeat}"],
            context,
        )
        views[f"s_wl_conditional_shuffled_{repeat}"] = _join(
            s,
            wl,
            attribute,
            blocks[f"conditional_shuffled_{repeat}"],
            context,
        )
        views[f"s_wl_conditional_mean_shuffled_{repeat}"] = _join(
            s,
            wl,
            attribute,
            blocks[f"conditional_mean_shuffled_{repeat}"],
            context,
        )
        views[f"s_hybrid_conditional_shuffled_{repeat}"] = _join(
            s,
            wl,
            exact,
            attribute,
            blocks[f"conditional_shuffled_{repeat}"],
            context,
        )
        views[f"s_hybrid_conditional_mean_shuffled_{repeat}"] = _join(
            s,
            wl,
            exact,
            attribute,
            blocks[f"conditional_mean_shuffled_{repeat}"],
            context,
        )
    return views


def _coverage(
    train_graphs: Sequence[GraphPatchData],
    valid_graphs: Sequence[GraphPatchData],
    vocabulary: TopologyVocabulary,
) -> dict[str, Any]:
    train_keys = {
        patch.topology_key for graph in train_graphs for patch in graph.patches
    }
    valid_patches = [patch for graph in valid_graphs for patch in graph.patches]
    valid_keys = {patch.topology_key for patch in valid_patches}
    selected = set(vocabulary.lookup)
    train_total = sum(len(graph.patches) for graph in train_graphs)
    selected_train = sum(spec.count for spec in vocabulary.specs)
    return {
        "train_patches": int(train_total),
        "valid_patches": int(len(valid_patches)),
        "train_exact_types": int(len(train_keys)),
        "valid_exact_types": int(len(valid_keys)),
        "selected_types": int(len(selected)),
        "selected_train_patch_coverage": float(selected_train / max(train_total, 1)),
        "selected_valid_patch_coverage": float(
            np.mean([patch.topology_key in selected for patch in valid_patches])
        )
        if valid_patches
        else 0.0,
        "valid_patch_seen_in_train_rate": float(
            np.mean([patch.topology_key in train_keys for patch in valid_patches])
        )
        if valid_patches
        else 0.0,
        "valid_type_seen_in_train_rate": float(
            np.mean([key in train_keys for key in valid_keys])
        )
        if valid_keys
        else 0.0,
    }


def _collision_summary(
    graphs: Sequence[GraphPatchData], key_name: str
) -> dict[str, Any]:
    mapping: dict[Any, set[bytes]] = defaultdict(set)
    for graph in graphs:
        for patch in graph.patches:
            mapping[getattr(patch, key_name)].add(patch.topology_key)
    widths = np.asarray([len(values) for values in mapping.values()], dtype=np.int64)
    return {
        "n_wl_keys": int(len(mapping)),
        "ambiguous_key_fraction": float(np.mean(widths > 1)) if widths.size else 0.0,
        "maximum_exact_types_per_key": int(widths.max()) if widths.size else 0,
        "mean_exact_types_per_key": float(widths.mean()) if widths.size else 0.0,
    }


def _render_markdown(result: Mapping[str, Any]) -> str:
    lines = [
        f"# {result['protocol_id']}",
        "",
        "Official-train scaffold folds only; official validation/test were not encoded.",
        "",
        f"Relabel invariance: **{'PASS' if result['object_audit']['pass'] else 'FAIL'}**",
        "",
        "| view | mean validation ROC-AUC | fold std |",
        "|---|---:|---:|",
    ]
    for name, row in result["aggregate"].items():
        lines.append(f"| `{name}` | {row['mean_auc']:.6f} | {row['std_auc']:.6f} |")
    lines.extend(["", "## Gates", ""])
    for name, gate in result["gates"].items():
        lines.append(
            f"- `{name}`: {gate['mean_delta']:+.6f}, wins "
            f"{gate['fold_wins']}/{result['n_folds']} — "
            f"**{'PASS' if gate['passed'] else 'FAIL'}**"
        )
    lines.extend(["", "## Fold representation diagnostics", ""])
    for row in result["fold_diagnostics"]:
        coverage = row["coverage"]
        lines.append(
            f"- fold {row['fold']}: top-{coverage['selected_types']} train/valid mass "
            f"{coverage['selected_train_patch_coverage']:.4f}/"
            f"{coverage['selected_valid_patch_coverage']:.4f}; raw orbit "
            f"{row['raw_orbit_dimension']}D; prototype {row['prototype_dimension']}D."
        )
    lines.extend(["", f"Decision: **{result['decision']}**", ""])
    return "\n".join(lines)


def run(config_path: Path) -> dict[str, Any]:
    config = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    data = config["data"]
    representation = config["representation"]
    screen = config["screen"]
    classifier = config["classifier"]
    result_json = _resolve(config["output_json"])
    result_markdown = _resolve(config["output_markdown"])
    frozen_path = _resolve(data["frozen_features"])
    folds_path = _resolve(data["scaffold_folds"])
    with np.load(frozen_path, allow_pickle=False) as archive:
        frozen = {name: np.asarray(archive[name]) for name in archive.files}
    with np.load(folds_path, allow_pickle=False) as archive:
        fold_archive = {name: np.asarray(archive[name]) for name in archive.files}
    bundle = load_molhiv(root=_resolve(data["root"]), with_features=True)
    if bundle.node_feats is None or bundle.edge_feats is None:
        raise ValueError("exact orbit fusion requires OGB node/edge features")
    labels = np.asarray(bundle.y, dtype=np.int64)
    start = time.perf_counter()
    fold_ids = sorted(
        int(name.removeprefix("fold_").removesuffix("_train_indices"))
        for name in fold_archive
        if name.startswith("fold_") and name.endswith("_train_indices")
    )
    selections = [
        _fold_indices(fold_archive, fold, labels, screen) for fold in fold_ids
    ]
    original = np.asarray(fold_archive["original_indices"], dtype=np.int64)
    audit_candidates = original[
        np.asarray(fold_archive["official_train_indices"], dtype=np.int64)
    ]
    shuffle_repeats = int(representation["shuffle_repeats"])
    object_audit = audit_invariance(
        bundle,
        audit_candidates,
        representation,
        config["audit"],
        shuffle_repeats=shuffle_repeats,
    )
    if not object_audit["pass"]:
        raise RuntimeError("exact orbit patch object failed relabel invariance")

    folds: list[dict[str, Any]] = []
    diagnostics: list[dict[str, Any]] = []
    model_seeds = [int(value) for value in classifier.get("model_seeds", [0])]
    for fold, (train_indices, valid_indices) in zip(
        fold_ids, selections, strict=True
    ):
        fold_indices = np.concatenate([train_indices, valid_indices]).astype(np.int64)
        graph_rows: list[GraphPatchData] = []
        for position, raw_index in enumerate(fold_indices):
            index = int(raw_index)
            graph_rows.append(
                _extract_graph_data(
                    bundle.graphs[index],
                    bundle.node_feats[index],
                    bundle.edge_feats[index],
                    representation,
                    shuffle_repeats=shuffle_repeats,
                    shuffle_seed=int(representation["shuffle_seed"]),
                )
            )
            if position and position % 500 == 0:
                print(
                    f"fold {fold} exact graphs: {position}/{fold_indices.size}",
                    flush=True,
                )
        train_graphs = graph_rows[: train_indices.size]
        valid_graphs = graph_rows[train_indices.size :]
        vocabulary = _fit_vocabulary(
            train_graphs, int(representation["topology_vocab_size"])
        )
        prototype_bank = _fit_prototype_bank(
            train_graphs,
            vocabulary,
            int(representation["prototypes_per_topology"]),
        )
        conditional_bank = _fit_conditional_prototype_bank(
            train_graphs,
            vocabulary,
            int(representation["prototypes_per_topology"]),
        )
        primitive = _stack_blocks(
            [
                _graph_feature_blocks(
                    graph,
                    vocabulary,
                    prototype_bank,
                    conditional_bank,
                    shuffle_repeats=shuffle_repeats,
                    patch_pair_shuffle_seed=int(
                        representation.get(
                            "patch_pair_shuffle_seed",
                            int(representation["shuffle_seed"]) + 1,
                        )
                    ),
                )
                for graph in graph_rows
            ]
        )
        views = _assemble_views(
            _frozen_s_rows(frozen, fold_indices),
            primitive,
            shuffle_repeats=shuffle_repeats,
        )
        scores, by_seed = _fit_auc_views(
            views,
            labels[fold_indices],
            int(train_indices.size),
            classifier,
            model_seeds,
        )
        folds.append(
            {
                "fold": int(fold),
                "n_train": int(train_indices.size),
                "n_valid": int(valid_indices.size),
                "scores": scores,
                "scores_by_model_seed": by_seed,
            }
        )
        fold_diagnostic = {
            "fold": int(fold),
            "coverage": _coverage(train_graphs, valid_graphs, vocabulary),
            "raw_orbit_dimension": int(vocabulary.raw_width),
            "prototype_count": int(prototype_bank.total_prototypes),
            "prototype_dimension": int(2 * prototype_bank.total_prototypes),
            "conditional_prototype_count": int(
                conditional_bank.total_prototypes
            ),
            "conditional_dimension": int(
                2 * conditional_bank.total_prototypes
            ),
            "wl_full_key_collision": _collision_summary(graph_rows, "wl_key"),
            "wl_binned_vector_collision": _collision_summary(
                graph_rows, "wl_binned_key"
            ),
            "topologies": [
                {
                    "rank": int(spec.index),
                    "certificate_sha256": hashlib.sha256(spec.key).hexdigest(),
                    "train_count": int(spec.count),
                    "node_orbits": int(spec.n_node_orbits),
                    "edge_orbits": int(spec.n_edge_orbits),
                    "prototypes": int(prototype_bank.values[spec.key].shape[0]),
                    "conditional_prototypes": int(
                        conditional_bank.values[spec.key].shape[0]
                    ),
                }
                for spec in vocabulary.specs
            ],
        }
        diagnostics.append(fold_diagnostic)
        print(
            f"fold {fold}: wl={scores['s_wl_structure']['valid_auc']:.6f}; "
            f"exact={scores['s_exact_structure']['valid_auc']:.6f}; "
            f"unbound={scores['s_exact_unbound']['valid_auc']:.6f}; "
            f"orbit={scores['s_exact_orbit_raw']['valid_auc']:.6f}; "
            f"prototype={scores['s_exact_prototype']['valid_auc']:.6f}; "
            f"hybrid-prototype={scores['s_hybrid_prototype']['valid_auc']:.6f}; "
            f"conditional={scores['s_wl_conditional_mean']['valid_auc']:.6f}",
            flush=True,
        )

    aggregate = {
        name: _aggregate_folds(folds, name) for name in folds[0]["scores"]
    }
    gates = {
        "exact_structure_vs_wl": _delta(
            folds, "s_exact_structure", "s_wl_structure"
        ),
        "exact_unbound_vs_wl_unbound": _delta(
            folds, "s_exact_unbound", "s_wl_unbound"
        ),
        "orbit_raw_vs_exact_unbound": _delta(
            folds, "s_exact_orbit_raw", "s_exact_unbound"
        ),
        "prototype_vs_exact_unbound": _delta(
            folds, "s_exact_prototype", "s_exact_unbound"
        ),
        "hybrid_prototype_vs_hybrid_unbound": _delta(
            folds, "s_hybrid_prototype", "s_hybrid_unbound"
        ),
        "conditional_vs_exact_unbound": _delta(
            folds, "s_exact_conditional", "s_exact_unbound"
        ),
        "wl_conditional_vs_wl_unbound": _delta(
            folds, "s_wl_conditional", "s_wl_unbound"
        ),
        "wl_conditional_mean_vs_wl_unbound": _delta(
            folds, "s_wl_conditional_mean", "s_wl_unbound"
        ),
        "hybrid_conditional_vs_hybrid_unbound": _delta(
            folds, "s_hybrid_conditional", "s_hybrid_unbound"
        ),
        "hybrid_conditional_mean_vs_hybrid_unbound": _delta(
            folds, "s_hybrid_conditional_mean", "s_hybrid_unbound"
        ),
    }
    for prefix, true_view in (
        ("orbit_raw", "s_exact_orbit_raw"),
        ("prototype", "s_exact_prototype"),
        ("hybrid_prototype", "s_hybrid_prototype"),
    ):
        gates[f"{prefix}_true_vs_shuffle"] = _delta_against_controls(
            folds,
            true_view,
            [f"{true_view}_shuffled_{repeat}" for repeat in range(shuffle_repeats)],
        )
    for prefix, true_view in (
        ("conditional_pairing", "s_exact_conditional"),
        ("wl_conditional_pairing", "s_wl_conditional"),
        ("wl_conditional_mean_pairing", "s_wl_conditional_mean"),
        ("hybrid_conditional_pairing", "s_hybrid_conditional"),
        (
            "hybrid_conditional_mean_pairing",
            "s_hybrid_conditional_mean",
        ),
    ):
        gates[f"{prefix}_true_vs_shuffle"] = _delta_against_controls(
            folds,
            true_view,
            [f"{true_view}_shuffled_{repeat}" for repeat in range(shuffle_repeats)],
        )
    minimum_delta = float(screen["minimum_delta"])
    minimum_wins = int(screen["minimum_fold_wins"])
    for gate in gates.values():
        gate["minimum_mean_delta"] = minimum_delta
        gate["minimum_fold_wins"] = minimum_wins
        gate["passed"] = bool(
            gate["mean_delta"] >= minimum_delta
            and gate["fold_wins"] >= minimum_wins
        )

    raw_pass = bool(
        gates["orbit_raw_vs_exact_unbound"]["passed"]
        and gates["orbit_raw_true_vs_shuffle"]["passed"]
    )
    compact_pass = bool(
        gates["prototype_vs_exact_unbound"]["passed"]
        and gates["prototype_true_vs_shuffle"]["passed"]
    )
    hybrid_pass = bool(
        gates["hybrid_prototype_vs_hybrid_unbound"]["passed"]
        and gates["hybrid_prototype_true_vs_shuffle"]["passed"]
    )
    conditional_pass = bool(
        gates["conditional_vs_exact_unbound"]["passed"]
        and gates["conditional_pairing_true_vs_shuffle"]["passed"]
    )
    wl_conditional_pass = bool(
        gates["wl_conditional_vs_wl_unbound"]["passed"]
        and gates["wl_conditional_pairing_true_vs_shuffle"]["passed"]
    )
    wl_conditional_mean_pass = bool(
        gates["wl_conditional_mean_vs_wl_unbound"]["passed"]
        and gates["wl_conditional_mean_pairing_true_vs_shuffle"]["passed"]
    )
    hybrid_conditional_pass = bool(
        gates["hybrid_conditional_vs_hybrid_unbound"]["passed"]
        and gates["hybrid_conditional_pairing_true_vs_shuffle"]["passed"]
    )
    hybrid_conditional_mean_pass = bool(
        gates["hybrid_conditional_mean_vs_hybrid_unbound"]["passed"]
        and gates["hybrid_conditional_mean_pairing_true_vs_shuffle"]["passed"]
    )
    if (
        wl_conditional_mean_pass
        or hybrid_conditional_mean_pass
        or wl_conditional_pass
        or conditional_pass
        or hybrid_conditional_pass
    ):
        decision = "PATCH_LEVEL_CONDITIONAL_FUSION_PROMISING"
    elif compact_pass or hybrid_pass:
        decision = "COMPACT_EXACT_ORBIT_FUSION_PROMISING"
    elif raw_pass:
        decision = "ORBIT_SIGNAL_ONLY_COMPRESSION_UNRESOLVED"
    else:
        decision = "EXACT_ORBIT_BINDING_NO_GO"

    result = {
        "protocol_id": str(config["protocol_id"]),
        "config": config,
        "data": {
            "dataset": str(data["dataset"]),
            "split": "official-train-only scaffold folds",
            "official_validation_encoded_or_evaluated": False,
            "official_test_encoded_or_evaluated": False,
        },
        "audit_boundary": {
            "config_sha256": _sha256(config_path),
            "frozen_s_sha256": _sha256(frozen_path),
            "scaffold_folds_sha256": _sha256(folds_path),
            "topology": "exact rooted colored-incidence nauty certificate",
            "alignment": "rooted structural automorphism node/edge orbits",
            "attributes": "strict atom semantics plus compact bond semantics",
            "vocabulary_fit": "independently on each outer-train fold",
            "prototype_fit": "deterministic farthest empirical templates on each outer-train fold",
            "labels_used_for_representation": False,
            "shuffle": "within-patch canonical-order matched permutation preserving attribute marginals",
            "patch_pair_shuffle": "within-graph permutation preserving topology and patch-attribute bags",
        },
        "object_audit": object_audit,
        "n_folds": len(folds),
        "folds": folds,
        "aggregate": aggregate,
        "fold_diagnostics": diagnostics,
        "gates": gates,
        "decision": decision,
        "runtime": {
            "seconds": float(time.perf_counter() - start),
            "python": platform.python_version(),
            "platform": platform.platform(),
        },
    }
    _write_json(result_json, result)
    result_markdown.parent.mkdir(parents=True, exist_ok=True)
    result_markdown.write_text(_render_markdown(result), encoding="utf-8")
    return result


def main(argv: Sequence[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    args = parser.parse_args(argv)
    result = run(args.config.resolve())
    print(_render_markdown(result))
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())

"""Typed Beam8 atom--patch incidence data for OGB MolHIV.

The historical Beam8 node interface averages all patches incident to an atom
before the neural model sees them.  This module keeps patch instances and the
atom--patch / patch--patch relations explicit.  It is deliberately dictionary
free: the first pilot tests the Beam8 topology before mixing in KSVD.
"""

from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Any, Mapping, Sequence

import numpy as np

from .attributed_beam8 import (
    attributed_component_signature,
    attributed_stable_order,
    exact_typed_canonical_order,
)
from .graph import Graph
from .overlap_cover import _make_cover, _make_patch
from .run_attributed_beam8_mutagenicity import prepare_attributed_beam_graph
from .run_beam8_nci1_chain_classification import (
    _adjacency,
    _components,
    _position_and_residual_features,
)
from .run_luyin14_route import (
    _complete_edges,
    _fair_prefix_from_completed,
    _prefix_metrics,
)


COVERAGE_CHECKPOINTS = ("base", "fair95", "edge100")


@dataclass(frozen=True)
class MolhivBeam8Incidence:
    """One molecule represented as atoms, Beam8 patches, and typed relations."""

    graph_index: int
    label: float
    atom_features: np.ndarray
    atom_edge_index: np.ndarray
    atom_edge_features: np.ndarray
    patch_slot_features: np.ndarray
    patch_slot_mask: np.ndarray
    patch_position_features: np.ndarray
    patch_internal_index: np.ndarray
    patch_internal_pair: np.ndarray
    patch_internal_features: np.ndarray
    bond_endpoint_index: np.ndarray
    bond_endpoint_pair: np.ndarray
    bond_endpoint_slot: np.ndarray
    bond_endpoint_features: np.ndarray
    incidence_index: np.ndarray
    incidence_slot: np.ndarray
    incidence_flags: np.ndarray
    chain_index: np.ndarray
    chain_features: np.ndarray
    shuffled_chain_index: np.ndarray
    mapping_shuffled_chain_features: np.ndarray
    overlap_index: np.ndarray
    overlap_features: np.ndarray
    mapping_shuffled_overlap_features: np.ndarray
    base_overlap_index: np.ndarray
    base_overlap_features: np.ndarray
    mapping_shuffled_base_overlap_features: np.ndarray
    token_shuffle_index: np.ndarray
    patch_is_completion: np.ndarray
    slot_nodes: tuple[tuple[int, ...], ...]
    segment_ids: tuple[int, ...]
    component_ids: tuple[int, ...]
    sampling: dict[str, Any]

    @property
    def n_atoms(self) -> int:
        return int(self.atom_features.shape[0])

    @property
    def n_patches(self) -> int:
        return int(self.patch_slot_features.shape[0])


def _feature_row_colors(values: np.ndarray) -> np.ndarray:
    """Exact graph-local colors for full categorical atom feature rows."""

    rows = [tuple(int(value) for value in row) for row in np.asarray(values)]
    ordered = sorted(set(rows))
    lookup = {row: index for index, row in enumerate(ordered)}
    return np.asarray([lookup[row] for row in rows], dtype=np.int64)


def _atomic_number_one_hot(values: np.ndarray, width: int) -> np.ndarray:
    values = np.asarray(values, dtype=np.int64)
    if values.ndim != 2 or values.shape[1] == 0:
        raise ValueError("atom features must be a non-empty 2-D categorical array")
    atomic = values[:, 0]
    if np.any(atomic < 0) or np.any(atomic >= width):
        raise ValueError("atomic-number category is outside the declared feature width")
    output = np.zeros((len(values), width), dtype=np.float64)
    output[np.arange(len(values)), atomic] = 1.0
    return output


def typed_bond_adjacency(
    graph: Graph,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    bond_feature_dims: Sequence[int],
) -> np.ndarray:
    """Encode each complete OGB bond-feature tuple as one stable positive type."""

    dims = tuple(int(value) for value in bond_feature_dims)
    if not dims or any(value <= 0 for value in dims):
        raise ValueError("bond feature dimensions must be positive")
    typed = np.zeros((graph.n, graph.n), dtype=np.int16)
    for left, right in graph.edges():
        key = (left, right) if left < right else (right, left)
        if key not in edge_features:
            raise ValueError(f"missing bond features for edge {key}")
        row = np.asarray(edge_features[key], dtype=np.int64).reshape(-1)
        if len(row) != len(dims):
            raise ValueError("bond feature row has the wrong number of channels")
        if any(value < 0 or value >= width for value, width in zip(row, dims)):
            raise ValueError("bond feature category is outside its declared width")
        code = 1 + int(np.ravel_multi_index(tuple(row.tolist()), dims))
        typed[left, right] = typed[right, left] = code
    return typed


def _atom_edges(
    graph: Graph,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    edge_feature_dim: int,
) -> tuple[np.ndarray, np.ndarray]:
    sources: list[int] = []
    targets: list[int] = []
    rows: list[np.ndarray] = []
    for left, right in sorted(graph.edges()):
        key = (left, right) if left < right else (right, left)
        row = np.asarray(edge_features[key], dtype=np.int64).reshape(-1)
        if len(row) != edge_feature_dim:
            raise ValueError("unexpected bond feature width")
        sources.extend([left, right])
        targets.extend([right, left])
        rows.extend([row, row])
    if not sources:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, edge_feature_dim), dtype=np.int64),
        )
    return np.asarray([sources, targets], dtype=np.int64), np.stack(rows)


def _segment_patch_indices(segment_ids: Sequence[int]) -> list[list[int]]:
    segments: dict[int, list[int]] = {}
    for patch, segment in enumerate(segment_ids):
        segments.setdefault(int(segment), []).append(int(patch))
    return [segments[key] for key in sorted(segments)]


def _nonidentity_permutation(size: int, rng: np.random.Generator) -> np.ndarray:
    permutation = rng.permutation(size)
    if size > 1 and np.array_equal(permutation, np.arange(size)):
        permutation = np.roll(permutation, 1)
    return permutation.astype(np.int64)


def _chain_relations(
    slot_nodes: Sequence[Sequence[int]],
    segment_ids: Sequence[int],
    patch_size: int,
) -> tuple[np.ndarray, np.ndarray]:
    sources: list[int] = []
    targets: list[int] = []
    features: list[np.ndarray] = []
    for patches in _segment_patch_indices(segment_ids):
        for left_patch, right_patch in zip(patches, patches[1:]):
            left_slots = {int(node): slot for slot, node in enumerate(slot_nodes[left_patch])}
            right_slots = {int(node): slot for slot, node in enumerate(slot_nodes[right_patch])}
            shared = sorted(set(left_slots) & set(right_slots))
            mapping = np.zeros((patch_size, patch_size), dtype=np.float64)
            for node in shared:
                mapping[left_slots[node], right_slots[node]] = 1.0
            forward = np.concatenate(
                [np.asarray([1.0, len(shared) / patch_size]), mapping.reshape(-1)]
            )
            reverse = np.concatenate(
                [np.asarray([-1.0, len(shared) / patch_size]), mapping.T.reshape(-1)]
            )
            sources.extend([left_patch, right_patch])
            targets.extend([right_patch, left_patch])
            features.extend([forward, reverse])
    if not sources:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, 2 + patch_size * patch_size), dtype=np.float64),
        )
    return np.asarray([sources, targets], dtype=np.int64), np.stack(features)


def _overlap_relations(
    slot_nodes: Sequence[Sequence[int]],
    component_ids: Sequence[int],
    patch_size: int,
    *,
    allowed: np.ndarray | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Connect every patch pair in a component that shares at least one atom."""

    sources: list[int] = []
    targets: list[int] = []
    features: list[np.ndarray] = []
    allowed_values = (
        np.ones(len(slot_nodes), dtype=np.bool_)
        if allowed is None
        else np.asarray(allowed, dtype=np.bool_)
    )
    if allowed_values.shape != (len(slot_nodes),):
        raise ValueError("allowed overlap mask must match the patch count")
    for patches in _segment_patch_indices(component_ids):
        for left_patch, right_patch in combinations(patches, 2):
            if not (allowed_values[left_patch] and allowed_values[right_patch]):
                continue
            left_slots = {
                int(node): slot for slot, node in enumerate(slot_nodes[left_patch])
            }
            right_slots = {
                int(node): slot for slot, node in enumerate(slot_nodes[right_patch])
            }
            shared = sorted(set(left_slots) & set(right_slots))
            if not shared:
                continue
            mapping = np.zeros((patch_size, patch_size), dtype=np.float64)
            for node in shared:
                mapping[left_slots[node], right_slots[node]] = 1.0
            forward = np.concatenate(
                [np.asarray([0.0, len(shared) / patch_size]), mapping.reshape(-1)]
            )
            reverse = np.concatenate(
                [np.asarray([0.0, len(shared) / patch_size]), mapping.T.reshape(-1)]
            )
            sources.extend([left_patch, right_patch])
            targets.extend([right_patch, left_patch])
            features.extend([forward, reverse])
    if not sources:
        return (
            np.empty((2, 0), dtype=np.int64),
            np.empty((0, 2 + patch_size * patch_size), dtype=np.float64),
        )
    return np.asarray([sources, targets], dtype=np.int64), np.stack(features)


def _shuffled_relations(
    chain_index: np.ndarray,
    segment_ids: Sequence[int],
    rng: np.random.Generator,
) -> np.ndarray:
    if chain_index.shape[1] == 0:
        return chain_index.copy()
    mapping = np.arange(len(segment_ids), dtype=np.int64)
    for patches in _segment_patch_indices(segment_ids):
        local = _nonidentity_permutation(len(patches), rng)
        for target_position, source_position in enumerate(local):
            mapping[patches[target_position]] = patches[int(source_position)]
    return mapping[chain_index]


def _mapping_shuffled_features(
    chain_features: np.ndarray,
    rng: np.random.Generator,
    patch_size: int,
) -> tuple[np.ndarray, float]:
    """Shuffle only shared-slot bindings while preserving chain endpoints."""

    output = np.asarray(chain_features, dtype=np.float64).copy()
    if output.shape[0] == 0:
        return output, 1.0
    if output.shape[0] % 2 != 0:
        raise ValueError("bidirectional chain features must occur in pairs")
    unchanged = 0
    pairs = output.shape[0] // 2
    for forward_index in range(0, output.shape[0], 2):
        reverse_index = forward_index + 1
        mapping = output[forward_index, 2:].reshape(patch_size, patch_size)
        source_slots = np.flatnonzero(mapping.sum(axis=1) > 0)
        target_slots = np.flatnonzero(mapping.sum(axis=0) > 0)
        if len(source_slots) != len(target_slots):
            raise ValueError("chain mapping must be one-to-one")
        shuffled = np.zeros_like(mapping)
        if len(source_slots) > 1:
            permutation = _nonidentity_permutation(len(target_slots), rng)
            shuffled[source_slots, target_slots[permutation]] = 1.0
        else:
            shuffled[source_slots, target_slots] = 1.0
            unchanged += 1
        output[forward_index, 2:] = shuffled.reshape(-1)
        output[reverse_index, 2:] = shuffled.T.reshape(-1)
    return output, unchanged / max(pairs, 1)


def _token_shuffle_relation(
    component_ids: Sequence[int], rng: np.random.Generator
) -> np.ndarray:
    """Return source->target edges that permute patch content within a component."""

    sources = np.arange(len(component_ids), dtype=np.int64)
    targets = np.arange(len(component_ids), dtype=np.int64)
    for patches in _segment_patch_indices(component_ids):
        local = _nonidentity_permutation(len(patches), rng)
        for target_position, source_position in enumerate(local):
            sources[patches[target_position]] = patches[int(source_position)]
    return np.stack([sources, targets])


def _canonical_global_coordinates(
    adjacency: np.ndarray,
    typed: np.ndarray,
    canonical_types: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    """Return invariant whole-graph coordinates and component IDs per node."""

    components = []
    for component in _components(adjacency):
        nodes = np.asarray(component, dtype=np.int64)
        local_typed = typed[np.ix_(nodes, nodes)]
        local_types = canonical_types[nodes]
        signature = attributed_component_signature(local_typed, local_types)
        stable = attributed_stable_order(local_typed, local_types)
        ordered = nodes[np.asarray(stable.order, dtype=np.int64)]
        components.append((signature, ordered))
    components.sort(key=lambda item: item[0])
    order = np.concatenate([nodes for _signature, nodes in components]).astype(np.int64)
    component_ids = np.empty(adjacency.shape[0], dtype=np.int64)
    for component_id, (_signature, nodes) in enumerate(components):
        component_ids[nodes] = component_id
    return order, component_ids


def _coverage_slots(
    item: Any,
    adjacency: np.ndarray,
    typed: np.ndarray,
    canonical_types: np.ndarray,
    *,
    patch_size: int,
    coverage_checkpoint: str,
) -> tuple[
    tuple[tuple[int, ...], ...],
    tuple[int, ...],
    tuple[int, ...],
    tuple[int, ...],
    np.ndarray,
    np.ndarray,
    dict[str, Any],
]:
    """Select BASE/FAIR95/EDGE100 in invariant coordinates and type slots."""

    if coverage_checkpoint not in COVERAGE_CHECKPOINTS:
        raise ValueError(
            f"coverage_checkpoint must be one of {COVERAGE_CHECKPOINTS}, "
            f"got {coverage_checkpoint!r}"
        )
    global_order, node_component_ids = _canonical_global_coordinates(
        adjacency, typed, canonical_types
    )
    inverse = np.empty(len(global_order), dtype=np.int64)
    inverse[global_order] = np.arange(len(global_order), dtype=np.int64)
    canonical_adjacency = adjacency[np.ix_(global_order, global_order)]
    base_patches = [
        _make_patch(
            canonical_adjacency,
            tuple(int(inverse[node]) for node in nodes),
            int(inverse[center]),
        )
        for nodes, center in zip(item.slot_nodes, item.centers)
    ]
    base = _make_cover(
        "molhiv_beam8_base",
        base_patches,
        item.segment_ids,
        [None] * len(base_patches),
        [0] * len(base_patches),
    )
    fair_reached = None
    if coverage_checkpoint == "base":
        selected = base
    else:
        completed = _complete_edges(
            canonical_adjacency, base, patch_size=patch_size
        )
        if coverage_checkpoint == "fair95":
            selected, fair_reached = _fair_prefix_from_completed(
                canonical_adjacency, completed
            )
        else:
            selected = completed

    selected_base_count = min(len(selected.patches), len(base.patches))
    slot_nodes: list[tuple[int, ...]] = []
    centers: list[int] = []
    component_ids: list[int] = []
    for patch in selected.patches:
        raw_nodes = tuple(int(global_order[node]) for node in patch.node_ids)
        center = int(global_order[patch.center])
        local_typed = typed[np.ix_(raw_nodes, raw_nodes)]
        result = exact_typed_canonical_order(
            local_typed,
            raw_nodes,
            canonical_types[np.asarray(raw_nodes, dtype=np.int64)],
            root=center,
        )
        slot_nodes.append(tuple(int(node) for node in result.node_ids))
        centers.append(center)
        component_ids.append(int(node_component_ids[raw_nodes[0]]))

    segments = tuple(int(value) for value in selected.segment_ids)
    node_sets = tuple(frozenset(nodes) for nodes in slot_nodes)
    position, _residual, coverage = _position_and_residual_features(
        adjacency, node_sets, segments
    )
    metrics = _prefix_metrics(canonical_adjacency, selected.patches)
    segment_sizes = {
        segment: segments.count(segment) for segment in set(segments)
    }
    chain_eligible = sum(
        1
        for patch, segment in enumerate(segments)
        if patch < selected_base_count and segment_sizes[segment] > 1
    )
    sampling = dict(item.sampling)
    sampling.update(
        {
            "coverage_checkpoint": coverage_checkpoint,
            "base_patches_total": len(base.patches),
            "selected_base_patches": selected_base_count,
            "completion_patches": len(selected.patches) - selected_base_count,
            "chain_eligible_patches": chain_eligible,
            "fair95_reached": fair_reached,
            "edge_coverage": float(coverage["edge_coverage"]),
            "node_coverage": float(coverage["node_coverage"]),
            "residual_edges": float(coverage["residual_edges"]),
            "pair_coverage": float(metrics["pair_coverage"]),
            "incident_p10": float(metrics["incident_p10"]),
        }
    )
    completion = np.arange(len(selected.patches)) >= selected_base_count
    return (
        tuple(slot_nodes),
        tuple(centers),
        segments,
        tuple(component_ids),
        position,
        completion,
        sampling,
    )


def build_molhiv_beam8_incidence(
    graph_index: int,
    graph: Graph,
    label: float,
    atom_features: np.ndarray,
    edge_features: Mapping[tuple[int, int], np.ndarray],
    *,
    atom_feature_dims: Sequence[int],
    bond_feature_dims: Sequence[int],
    patch_size: int = 8,
    overlap: int = 2,
    retained_beam: int = 8,
    seed: int = 20260815,
    coverage_checkpoint: str = "edge100",
) -> MolhivBeam8Incidence:
    """Build a dictionary-free, full-feature-canonical Beam8 molecule."""

    atoms = np.asarray(atom_features, dtype=np.int64)
    if atoms.shape[0] != graph.n:
        raise ValueError("atom feature rows and graph nodes disagree")
    atom_dims = tuple(int(value) for value in atom_feature_dims)
    bond_dims = tuple(int(value) for value in bond_feature_dims)
    if atoms.shape[1] != len(atom_dims):
        raise ValueError("atom feature row has the wrong number of channels")
    typed = typed_bond_adjacency(graph, edge_features, bond_dims)
    atomic_one_hot = _atomic_number_one_hot(atoms, atom_dims[0])
    canonical_types = _feature_row_colors(atoms)
    item = prepare_attributed_beam_graph(
        int(graph_index),
        graph,
        int(label),
        atomic_one_hot,
        typed,
        patch_size=patch_size,
        overlap=overlap,
        retained_beam=retained_beam,
        seed=seed,
        edge_dim=int(np.prod(bond_dims)),
        canonical_node_features=atomic_one_hot,
        canonical_node_types=canonical_types,
    )

    adjacency = _adjacency(graph)
    (
        slot_nodes,
        centers,
        segment_ids,
        component_ids,
        position_features,
        patch_is_completion,
        sampling,
    ) = _coverage_slots(
        item,
        adjacency,
        typed,
        canonical_types,
        patch_size=patch_size,
        coverage_checkpoint=coverage_checkpoint,
    )

    n_patches = len(slot_nodes)
    slots = np.zeros((n_patches, patch_size, atoms.shape[1]), dtype=np.int64)
    slot_mask = np.zeros((n_patches, patch_size), dtype=np.float64)
    internal_patch: list[int] = []
    internal_pair: list[int] = []
    internal_features: list[np.ndarray] = []
    endpoint_patch: list[int] = []
    endpoint_atom: list[int] = []
    endpoint_pair: list[int] = []
    endpoint_slot: list[int] = []
    endpoint_features: list[np.ndarray] = []
    pair_lookup = {pair: index for index, pair in enumerate(combinations(range(patch_size), 2))}
    for patch, nodes in enumerate(slot_nodes):
        count = len(nodes)
        slots[patch, :count] = atoms[np.asarray(nodes, dtype=np.int64)]
        slot_mask[patch, :count] = 1.0
        for left_slot, right_slot in combinations(range(count), 2):
            left, right = int(nodes[left_slot]), int(nodes[right_slot])
            key = (left, right) if left < right else (right, left)
            if key not in edge_features:
                continue
            internal_patch.append(patch)
            pair_index = pair_lookup[(left_slot, right_slot)]
            feature_row = np.asarray(edge_features[key], dtype=np.int64)
            internal_pair.append(pair_index)
            internal_features.append(feature_row)
            endpoint_patch.extend([patch, patch])
            endpoint_atom.extend([left, right])
            endpoint_pair.extend([pair_index, pair_index])
            endpoint_slot.extend([left_slot, right_slot])
            endpoint_features.extend([feature_row, feature_row])

    incidence_atom: list[int] = []
    incidence_patch: list[int] = []
    incidence_slot: list[int] = []
    incidence_flags: list[tuple[float, float, float]] = []
    segments = _segment_patch_indices(segment_ids)
    previous_patch: dict[int, int] = {}
    next_patch: dict[int, int] = {}
    for patches in segments:
        for left, right in zip(patches, patches[1:]):
            next_patch[left] = right
            previous_patch[right] = left
    for patch, nodes in enumerate(slot_nodes):
        previous_nodes = (
            set(slot_nodes[previous_patch[patch]]) if patch in previous_patch else set()
        )
        next_nodes = set(slot_nodes[next_patch[patch]]) if patch in next_patch else set()
        for slot, node in enumerate(nodes):
            incidence_atom.append(int(node))
            incidence_patch.append(int(patch))
            incidence_slot.append(int(slot))
            incidence_flags.append(
                (
                    float(int(node) == int(centers[patch])),
                    float(int(node) in previous_nodes),
                    float(int(node) in next_nodes),
                )
            )

    atom_edge_index, atom_edge_rows = _atom_edges(graph, edge_features, len(bond_dims))
    chain_index, chain_features = _chain_relations(
        slot_nodes, segment_ids, patch_size
    )
    overlap_index, overlap_features = _overlap_relations(
        slot_nodes, component_ids, patch_size
    )
    base_overlap_index, base_overlap_features = _overlap_relations(
        slot_nodes,
        component_ids,
        patch_size,
        allowed=~np.asarray(patch_is_completion, dtype=np.bool_),
    )
    rng = np.random.default_rng(seed + 1000003 * int(graph_index))
    shuffled_chain = _shuffled_relations(chain_index, segment_ids, rng)
    mapping_shuffled_features, mapping_unchanged = _mapping_shuffled_features(
        chain_features, rng, patch_size
    )
    shuffled_overlap_features, overlap_mapping_unchanged = _mapping_shuffled_features(
        overlap_features, rng, patch_size
    )
    (
        shuffled_base_overlap_features,
        base_overlap_mapping_unchanged,
    ) = _mapping_shuffled_features(base_overlap_features, rng, patch_size)
    token_shuffle = _token_shuffle_relation(component_ids, rng)
    sampling["mapping_shuffle_unchanged_fraction"] = float(mapping_unchanged)
    sampling["overlap_edges_undirected"] = int(overlap_index.shape[1] // 2)
    sampling["base_overlap_edges_undirected"] = int(
        base_overlap_index.shape[1] // 2
    )
    sampling["overlap_mapping_shuffle_unchanged_fraction"] = float(
        overlap_mapping_unchanged
    )
    sampling["base_overlap_mapping_shuffle_unchanged_fraction"] = float(
        base_overlap_mapping_unchanged
    )

    if internal_features:
        internal_index = np.asarray(internal_patch, dtype=np.int64)
        internal_pairs = np.asarray(internal_pair, dtype=np.int64)
        internal_rows = np.stack(internal_features)
    else:
        internal_index = np.empty(0, dtype=np.int64)
        internal_pairs = np.empty(0, dtype=np.int64)
        internal_rows = np.empty((0, len(bond_dims)), dtype=np.int64)
    if endpoint_features:
        endpoint_index = np.asarray(
            [endpoint_patch, endpoint_atom], dtype=np.int64
        )
        endpoint_pairs = np.asarray(endpoint_pair, dtype=np.int64)
        endpoint_slots = np.asarray(endpoint_slot, dtype=np.int64)
        endpoint_rows = np.stack(endpoint_features)
    else:
        endpoint_index = np.empty((2, 0), dtype=np.int64)
        endpoint_pairs = np.empty(0, dtype=np.int64)
        endpoint_slots = np.empty(0, dtype=np.int64)
        endpoint_rows = np.empty((0, len(bond_dims)), dtype=np.int64)

    return MolhivBeam8Incidence(
        graph_index=int(graph_index),
        label=float(label),
        atom_features=atoms,
        atom_edge_index=atom_edge_index,
        atom_edge_features=atom_edge_rows,
        patch_slot_features=slots,
        patch_slot_mask=slot_mask,
        patch_position_features=np.asarray(position_features, dtype=np.float64),
        patch_internal_index=internal_index,
        patch_internal_pair=internal_pairs,
        patch_internal_features=internal_rows,
        bond_endpoint_index=endpoint_index,
        bond_endpoint_pair=endpoint_pairs,
        bond_endpoint_slot=endpoint_slots,
        bond_endpoint_features=endpoint_rows,
        incidence_index=np.asarray([incidence_atom, incidence_patch], dtype=np.int64),
        incidence_slot=np.asarray(incidence_slot, dtype=np.int64),
        incidence_flags=np.asarray(incidence_flags, dtype=np.float64),
        chain_index=chain_index,
        chain_features=chain_features,
        shuffled_chain_index=shuffled_chain,
        mapping_shuffled_chain_features=mapping_shuffled_features,
        overlap_index=overlap_index,
        overlap_features=overlap_features,
        mapping_shuffled_overlap_features=shuffled_overlap_features,
        base_overlap_index=base_overlap_index,
        base_overlap_features=base_overlap_features,
        mapping_shuffled_base_overlap_features=shuffled_base_overlap_features,
        token_shuffle_index=token_shuffle,
        patch_is_completion=np.asarray(patch_is_completion, dtype=np.bool_),
        slot_nodes=slot_nodes,
        segment_ids=segment_ids,
        component_ids=component_ids,
        sampling=sampling,
    )


def to_heterodata(item: MolhivBeam8Incidence) -> Any:
    """Convert one incidence item to a PyG ``HeteroData`` object."""

    try:
        import torch
        from torch_geometric.data import HeteroData
    except ImportError as exc:  # pragma: no cover - exercised in the runner environment
        raise RuntimeError(f"PyTorch and torch-geometric are required: {exc}") from exc

    data = HeteroData()
    data["atom"].x = torch.tensor(item.atom_features, dtype=torch.long)
    data["atom"].num_nodes = item.n_atoms
    data["patch"].slot_x = torch.tensor(item.patch_slot_features, dtype=torch.long)
    data["patch"].slot_mask = torch.tensor(item.patch_slot_mask, dtype=torch.float32)
    data["patch"].position = torch.tensor(
        item.patch_position_features, dtype=torch.float32
    )
    data["patch"].is_completion = torch.tensor(
        item.patch_is_completion, dtype=torch.bool
    )
    data["patch"].num_nodes = item.n_patches

    data["atom", "bond", "atom"].edge_index = torch.tensor(
        item.atom_edge_index, dtype=torch.long
    )
    data["atom", "bond", "atom"].edge_attr = torch.tensor(
        item.atom_edge_features, dtype=torch.long
    )
    # Store patch->atom orientation for direct scatter-to-atom operations.
    data["patch", "contains", "atom"].edge_index = torch.tensor(
        item.incidence_index[[1, 0]], dtype=torch.long
    )
    data["patch", "contains", "atom"].slot = torch.tensor(
        item.incidence_slot, dtype=torch.long
    )
    data["patch", "contains", "atom"].flags = torch.tensor(
        item.incidence_flags, dtype=torch.float32
    )
    internal_patch = torch.tensor(item.patch_internal_index, dtype=torch.long)
    data["patch", "internal_bond", "patch"].edge_index = torch.stack(
        [internal_patch, internal_patch], dim=0
    )
    data["patch", "internal_bond", "patch"].pair = torch.tensor(
        item.patch_internal_pair, dtype=torch.long
    )
    data["patch", "internal_bond", "patch"].edge_attr = torch.tensor(
        item.patch_internal_features, dtype=torch.long
    )
    data["patch", "bond_endpoint", "atom"].edge_index = torch.tensor(
        item.bond_endpoint_index, dtype=torch.long
    )
    data["patch", "bond_endpoint", "atom"].pair = torch.tensor(
        item.bond_endpoint_pair, dtype=torch.long
    )
    data["patch", "bond_endpoint", "atom"].slot = torch.tensor(
        item.bond_endpoint_slot, dtype=torch.long
    )
    data["patch", "bond_endpoint", "atom"].edge_attr = torch.tensor(
        item.bond_endpoint_features, dtype=torch.long
    )
    data["patch", "chain", "patch"].edge_index = torch.tensor(
        item.chain_index, dtype=torch.long
    )
    data["patch", "chain", "patch"].edge_attr = torch.tensor(
        item.chain_features, dtype=torch.float32
    )
    data["patch", "chain_shuffled", "patch"].edge_index = torch.tensor(
        item.shuffled_chain_index, dtype=torch.long
    )
    data["patch", "chain_shuffled", "patch"].edge_attr = torch.tensor(
        item.chain_features, dtype=torch.float32
    )
    data["patch", "mapping_shuffled", "patch"].edge_index = torch.tensor(
        item.chain_index, dtype=torch.long
    )
    data["patch", "mapping_shuffled", "patch"].edge_attr = torch.tensor(
        item.mapping_shuffled_chain_features, dtype=torch.float32
    )
    data["patch", "overlap", "patch"].edge_index = torch.tensor(
        item.overlap_index, dtype=torch.long
    )
    data["patch", "overlap", "patch"].edge_attr = torch.tensor(
        item.overlap_features, dtype=torch.float32
    )
    data["patch", "overlap_mapping_shuffled", "patch"].edge_index = torch.tensor(
        item.overlap_index, dtype=torch.long
    )
    data["patch", "overlap_mapping_shuffled", "patch"].edge_attr = torch.tensor(
        item.mapping_shuffled_overlap_features, dtype=torch.float32
    )
    data["patch", "base_overlap", "patch"].edge_index = torch.tensor(
        item.base_overlap_index, dtype=torch.long
    )
    data["patch", "base_overlap", "patch"].edge_attr = torch.tensor(
        item.base_overlap_features, dtype=torch.float32
    )
    data["patch", "base_overlap_mapping_shuffled", "patch"].edge_index = torch.tensor(
        item.base_overlap_index, dtype=torch.long
    )
    data["patch", "base_overlap_mapping_shuffled", "patch"].edge_attr = torch.tensor(
        item.mapping_shuffled_base_overlap_features, dtype=torch.float32
    )
    data["patch", "token_shuffle", "patch"].edge_index = torch.tensor(
        item.token_shuffle_index, dtype=torch.long
    )
    data.y = torch.tensor([item.label], dtype=torch.float32)
    data.graph_index = torch.tensor([item.graph_index], dtype=torch.long)
    return data


def incidence_summary(items: Sequence[MolhivBeam8Incidence]) -> dict[str, float | int]:
    if not items:
        raise ValueError("cannot summarize an empty incidence collection")
    patches = np.asarray([item.n_patches for item in items], dtype=np.float64)
    atoms = np.asarray([item.n_atoms for item in items], dtype=np.float64)
    incidences = np.asarray(
        [item.incidence_index.shape[1] for item in items], dtype=np.float64
    )
    chains = np.asarray([item.chain_index.shape[1] // 2 for item in items], dtype=np.float64)
    overlaps = np.asarray(
        [item.overlap_index.shape[1] // 2 for item in items], dtype=np.float64
    )
    base_overlaps = np.asarray(
        [item.base_overlap_index.shape[1] // 2 for item in items], dtype=np.float64
    )
    completion = np.asarray(
        [float(np.sum(item.patch_is_completion)) for item in items], dtype=np.float64
    )
    chain_eligible = np.asarray(
        [float(item.sampling.get("chain_eligible_patches", 0.0)) for item in items],
        dtype=np.float64,
    )
    edge_coverage = np.asarray(
        [float(item.sampling.get("edge_coverage", 0.0)) for item in items],
        dtype=np.float64,
    )
    node_coverage = np.asarray(
        [float(item.sampling.get("node_coverage", 0.0)) for item in items],
        dtype=np.float64,
    )
    incident_p10 = np.asarray(
        [float(item.sampling.get("incident_p10", 0.0)) for item in items],
        dtype=np.float64,
    )
    return {
        "graphs": int(len(items)),
        "mean_atoms": float(atoms.mean()),
        "mean_patches": float(patches.mean()),
        "mean_incidence_edges": float(incidences.mean()),
        "mean_chain_edges_undirected": float(chains.mean()),
        "mean_overlap_edges_undirected": float(overlaps.mean()),
        "mean_base_overlap_edges_undirected": float(base_overlaps.mean()),
        "mean_patches_per_atom": float(np.mean(incidences / np.maximum(atoms, 1.0))),
        "mean_completion_patches": float(completion.mean()),
        "mean_completion_patch_fraction": float(
            np.mean(completion / np.maximum(patches, 1.0))
        ),
        "mean_chain_eligible_patch_fraction": float(
            np.mean(chain_eligible / np.maximum(patches, 1.0))
        ),
        "mean_edge_coverage": float(edge_coverage.mean()),
        "mean_node_coverage": float(node_coverage.mean()),
        "mean_incident_p10": float(incident_p10.mean()),
        "fair95_gate_fraction": float(
            np.mean(
                (edge_coverage >= 0.95 - 1e-12)
                & (node_coverage >= 1.0 - 1e-12)
                & (incident_p10 >= 0.90 - 1e-12)
            )
        ),
        "full_edge_coverage_fraction": float(np.mean(edge_coverage >= 1.0 - 1e-12)),
    }

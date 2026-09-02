"""Construct invariant attributed Beam8 items for Mutagenicity."""
from __future__ import annotations

from typing import Any

import numpy as np

from .attributed_beam8 import (
    attributed_component_signature,
    attributed_stable_order,
    exact_typed_canonical_order,
    typed_slot_vector,
)
from .data_tud import node_feature_readout
from .marginal_candidate_cover import sample_marginal_candidate_cover
from .overlap_cover import _make_patch, patch_budget
from .run_beam8_nci1_chain_classification import (
    BeamGraph,
    _adjacency,
    _component_seed,
    _components,
    _position_and_residual_features,
)
from .run_real_structure_ksvd import graph_basic_features


def prepare_attributed_beam_graph(
    index: int,
    graph: Any,
    label: int,
    node_features: np.ndarray,
    typed_adjacency: np.ndarray,
    *,
    patch_size: int = 8,
    overlap: int = 2,
    retained_beam: int = 8,
    edge_capacity_multiplier: float = 1.5,
    seed: int = 20260813,
    edge_dim: int | None = None,
    canonical_node_features: np.ndarray | None = None,
    canonical_node_types: np.ndarray | None = None,
) -> BeamGraph:
    adjacency = _adjacency(graph)
    typed = np.asarray(typed_adjacency, dtype=np.int16)
    features = np.asarray(node_features, dtype=np.float64)
    slot_features = np.asarray(
        features if canonical_node_features is None else canonical_node_features,
        dtype=np.float64,
    )
    if slot_features.shape[0] != features.shape[0]:
        raise ValueError("canonical and content node features disagree on node count")
    slot_node_types = np.argmax(slot_features, axis=1).astype(np.int64)
    canonical_types = np.asarray(
        slot_node_types if canonical_node_types is None else canonical_node_types,
        dtype=np.int64,
    )
    if canonical_types.shape != (features.shape[0],):
        raise ValueError("canonical node types must contain one color per node")
    vector_edge_dim = (
        int(typed.max(initial=0)) if edge_dim is None else int(edge_dim)
    )
    if vector_edge_dim < int(typed.max(initial=0)):
        raise ValueError("edge_dim is smaller than an observed bond type")
    components = []
    for component in _components(adjacency):
        nodes = np.asarray(component, dtype=np.int64)
        local_typed = typed[np.ix_(nodes, nodes)]
        signature = attributed_component_signature(local_typed, canonical_types[nodes])
        components.append((signature, nodes))
    components.sort(key=lambda item: item[0])
    node_sets = []
    slot_nodes = []
    centers = []
    segment_ids = []
    vectors = []
    histograms = []
    component_sizes = []
    partial_components = 0
    for segment, (_signature, component_nodes) in enumerate(components):
        local_binary = adjacency[np.ix_(component_nodes, component_nodes)]
        local_typed_source = typed[np.ix_(component_nodes, component_nodes)]
        local_types_source = canonical_types[component_nodes]
        stable = attributed_stable_order(local_typed_source, local_types_source)
        order = np.asarray(stable.order, dtype=np.int64)
        binary = local_binary[np.ix_(order, order)]
        typed_local = local_typed_source[np.ix_(order, order)]
        types_local = local_types_source[order]
        ordered_global = component_nodes[order]
        component_sizes.append(len(component_nodes))
        if len(component_nodes) <= patch_size:
            local_patch_nodes = [tuple(range(len(component_nodes)))]
            local_centers = [0]
            expected = 1
        else:
            expected = patch_budget(
                binary,
                patch_size=patch_size,
                target_overlap=overlap,
                edge_capacity_multiplier=edge_capacity_multiplier,
            )
            component_seed = _component_seed(binary, seed)
            cover = sample_marginal_candidate_cover(
                binary,
                np.random.default_rng(component_seed),
                n_patches=expected,
                patch_size=patch_size,
                target_overlap=overlap,
                retained_beam=retained_beam,
                candidate_restarts=1,
                allow_partial=True,
            )
            local_patch_nodes = [patch.node_ids for patch in cover.patches]
            local_centers = [patch.center for patch in cover.patches]
        partial_components += int(len(local_patch_nodes) < expected)
        for nodes, center in zip(local_patch_nodes, local_centers):
            local_indices = tuple(int(node) for node in nodes)
            patch_typed = typed_local[np.ix_(local_indices, local_indices)]
            patch_types = types_local[np.asarray(local_indices, dtype=np.int64)]
            result = exact_typed_canonical_order(
                patch_typed,
                local_indices,
                patch_types,
                root=int(center),
            )
            global_nodes = tuple(int(ordered_global[node]) for node in result.node_ids)
            global_center = int(ordered_global[int(center)])
            node_sets.append(frozenset(global_nodes))
            slot_nodes.append(global_nodes)
            centers.append(global_center)
            segment_ids.append(segment)
            vectors.append(
                typed_slot_vector(
                    global_nodes,
                    typed,
                    slot_node_types,
                    patch_size=patch_size,
                    node_dim=slot_features.shape[1],
                    edge_dim=vector_edge_dim,
                )
            )
            histograms.append(features[np.asarray(global_nodes)].mean(axis=0))
    position, residual, coverage = _position_and_residual_features(
        adjacency, node_sets, segment_ids
    )
    return BeamGraph(
        index=int(index),
        label=int(label),
        vectors=np.stack(vectors),
        node_histograms=np.stack(histograms),
        node_sets=tuple(node_sets),
        slot_nodes=tuple(slot_nodes),
        centers=tuple(centers),
        segment_ids=tuple(segment_ids),
        position_features=position,
        residual_features=residual,
        graph_features=node_feature_readout(features),
        stats=graph_basic_features(graph),
        sampling={
            "components": len(components),
            "component_sizes": component_sizes,
            "patches": len(vectors),
            "partial_components": partial_components,
            **coverage,
        },
    )

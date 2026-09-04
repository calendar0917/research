from __future__ import annotations

import numpy as np
import torch

from ksvd_research.core import from_edges
from tracks.ksvd.experiments.luyin16.center_relation_network import (
    ATTRIBUTE_WIDTH,
    RELATION_WIDTH,
    TOPOLOGY_WIDTH,
    CenterRelationNetwork,
    _shuffle_center_attributes,
    _centre_arrays,
)


def _typed_graph_inputs(types: np.ndarray, bonds: dict[tuple[int, int], int]):
    graph = from_edges(5, [(0, 1), (1, 2), (1, 3), (3, 4)])
    return graph, types, bonds


def test_topology_block_is_independent_of_chemistry() -> None:
    graph = from_edges(5, [(0, 1), (1, 2), (1, 3), (3, 4)])
    topology_a, attributes_a, edge_index_a, relations_a, _ = _centre_arrays(
        graph,
        np.asarray([0, 1, 2, 3, 4], dtype=np.int64),
        {(0, 1): 0, (1, 2): 1, (1, 3): 2, (3, 4): 3},
        radius=3,
        wl_rounds=2,
        node_role_bins=64,
        edge_role_bins=32,
        relation_radius=3,
    )
    topology_b, attributes_b, edge_index_b, relations_b, _ = _centre_arrays(
        graph,
        np.asarray([4, 3, 2, 1, 0], dtype=np.int64),
        {(0, 1): 3, (1, 2): 2, (1, 3): 1, (3, 4): 0},
        radius=3,
        wl_rounds=2,
        node_role_bins=64,
        edge_role_bins=32,
        relation_radius=3,
    )
    np.testing.assert_allclose(topology_a, topology_b)
    np.testing.assert_array_equal(edge_index_a, edge_index_b)
    np.testing.assert_allclose(relations_a[:, :5], relations_b[:, :5])
    assert not np.allclose(attributes_a, attributes_b)
    assert topology_a.shape[1] == TOPOLOGY_WIDTH
    assert attributes_a.shape[1] == ATTRIBUTE_WIDTH
    assert relations_a.shape[1] == RELATION_WIDTH


def test_center_relation_network_produces_one_prediction_per_graph() -> None:
    graph = from_edges(4, [(0, 1), (1, 2), (2, 3)])
    topology, attributes, edge_index, relations, _ = _centre_arrays(
        graph,
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        {(0, 1): 0, (1, 2): 1, (2, 3): 2},
        radius=2,
        wl_rounds=1,
        node_role_bins=64,
        edge_role_bins=32,
        relation_radius=3,
    )
    from torch_geometric.data import Batch, Data

    item = Data(
        node_topology=torch.from_numpy(topology),
        node_attributes=torch.from_numpy(attributes),
        center_edge_index=torch.from_numpy(edge_index),
        center_edge_attr=torch.from_numpy(relations),
        y=torch.tensor([0.0]),
    )
    item.num_nodes = graph.n
    batch = Batch.from_data_list([item, item])
    model = CenterRelationNetwork("conditional_relation", TOPOLOGY_WIDTH, ATTRIBUTE_WIDTH, RELATION_WIDTH, 8, 1)
    prediction = model(batch)
    assert prediction.shape == (2,)
    assert torch.isfinite(prediction).all()


def test_shuffle_control_preserves_attributes_but_breaks_center_alignment() -> None:
    graph = from_edges(4, [(0, 1), (1, 2), (2, 3)])
    topology, attributes, edge_index, relations, _ = _centre_arrays(
        graph,
        np.asarray([0, 1, 2, 3], dtype=np.int64),
        {(0, 1): 0, (1, 2): 1, (2, 3): 2},
        radius=2,
        wl_rounds=1,
        node_role_bins=64,
        edge_role_bins=32,
        relation_radius=3,
    )
    from torch_geometric.data import Data

    item = Data(
        node_topology=torch.from_numpy(topology),
        node_attributes=torch.from_numpy(attributes),
        center_edge_index=torch.from_numpy(edge_index),
        center_edge_attr=torch.from_numpy(relations),
        y=torch.tensor([0.0]),
    )
    shuffled = _shuffle_center_attributes(item, seed=19, key=7)
    assert torch.equal(shuffled.node_topology, item.node_topology)
    assert torch.equal(shuffled.center_edge_index, item.center_edge_index)
    assert torch.equal(shuffled.center_edge_attr, item.center_edge_attr)
    original_rows = sorted(tuple(row) for row in item.node_attributes.numpy().tolist())
    shuffled_rows = sorted(tuple(row) for row in shuffled.node_attributes.numpy().tolist())
    assert shuffled_rows == original_rows
    assert not torch.equal(shuffled.node_attributes, item.node_attributes)

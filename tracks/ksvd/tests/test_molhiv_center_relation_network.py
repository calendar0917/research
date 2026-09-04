from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from ksvd_research.core import from_edges
from tracks.ksvd.experiments.luyin16.molhiv_center_relation_network import (
    ATTRIBUTE_WIDTH,
    TOPOLOGY_WIDTH,
    _center_features,
    _new_model,
)


def test_molhiv_center_features_and_conditional_model() -> None:
    graph = from_edges(4, [(0, 1), (1, 2), (2, 3)])
    node_features = np.asarray(
        [
            [5, 0, 1, 0, 2, 0, 0, 0, 0],
            [5, 0, 2, 0, 1, 0, 0, 0, 0],
            [7, 1, 2, 0, 0, 0, 0, 0, 1],
            [5, 0, 1, 0, 3, 0, 0, 0, 0],
        ],
        dtype=np.int64,
    )
    edge_features = {
        (0, 1): np.asarray([0, 0, 0], dtype=np.int64),
        (1, 2): np.asarray([1, 0, 0], dtype=np.int64),
        (2, 3): np.asarray([0, 0, 0], dtype=np.int64),
    }
    representation = {
        "radius": 3,
        "wl_rounds": 3,
        "node_role_bins": 64,
        "edge_role_bins": 32,
        "degree_scale": 4.0,
    }
    topology, attributes = _center_features(
        graph, node_features, edge_features, representation
    )
    assert topology.shape == (4, TOPOLOGY_WIDTH)
    assert attributes.shape == (4, ATTRIBUTE_WIDTH)
    item = Data(
        node_topology=torch.from_numpy(topology),
        node_attributes=torch.from_numpy(attributes),
        y=torch.tensor([1.0]),
    )
    item.num_nodes = 4
    prediction = _new_model(8)(Batch.from_data_list([item, item]))
    assert prediction.shape == (2,)
    assert torch.isfinite(prediction).all()

from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from .graph import from_edges
from .run_luyin14_edge_aware_joint import EdgeFusionGINE, _typed_node_patch_vectors


def test_typed_patch_distinguishes_edge_types() -> None:
    graph = from_edges(3, [(0, 1), (1, 2)])
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    first = Data(
        edge_index=edge_index,
        edge_attr=torch.tensor([[1, 0], [1, 0], [0, 1], [0, 1]], dtype=torch.float32),
    )
    second = Data(
        edge_index=edge_index,
        edge_attr=torch.tensor([[0, 1], [0, 1], [1, 0], [1, 0]], dtype=torch.float32),
    )
    left = _typed_node_patch_vectors(graph, first, patch_size=3, edge_dim=2)
    right = _typed_node_patch_vectors(graph, second, patch_size=3, edge_dim=2)
    assert left.shape == (3, 6)
    assert not np.array_equal(left, right)


def test_zero_init_edge_film_replays_gine() -> None:
    data = Data(
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        s=torch.tensor([[0.3, -0.2], [0.5, 0.1]]),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_attr=torch.tensor([[1.0, 0.0], [1.0, 0.0]]),
        y=torch.tensor([0]),
    )
    batch = Batch.from_data_list([data])
    torch.manual_seed(5)
    base = EdgeFusionGINE(2, 2, 8, 2, layers=2, dropout=0.0, struct_dim=0)
    torch.manual_seed(5)
    film = EdgeFusionGINE(2, 2, 8, 2, layers=2, dropout=0.0, struct_dim=2)
    base.eval()
    film.eval()
    with torch.no_grad():
        assert torch.allclose(base(batch), film(batch), atol=1e-6)

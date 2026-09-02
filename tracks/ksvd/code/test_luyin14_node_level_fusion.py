from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from .graph import from_edges
from .run_luyin14_node_level_fusion import (
    NodeFusionGIN,
    _node_patch_vectors,
    _shuffle_tokens,
)


def test_node_patch_vectors_are_rooted_and_fixed_width() -> None:
    graph = from_edges(5, [(0, 1), (0, 2), (0, 3), (3, 4)])
    vectors = _node_patch_vectors(graph, patch_size=4)
    assert vectors.shape == (5, 6)
    assert np.all(np.isfinite(vectors))


def test_node_patch_vectors_accept_disconnected_low_degree_graph() -> None:
    graph = from_edges(5, [(0, 1), (2, 3)])
    vectors = _node_patch_vectors(graph, patch_size=4)
    assert vectors.shape == (5, 6)
    assert np.all(np.isfinite(vectors))


def test_radius_two_patch_changes_local_context() -> None:
    graph = from_edges(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)])
    radius_one = _node_patch_vectors(graph, patch_size=4, radius=1)
    radius_two = _node_patch_vectors(graph, patch_size=4, radius=2)
    assert radius_one.shape == radius_two.shape == (6, 6)
    assert np.all(np.isfinite(radius_two))
    assert np.any(radius_one != radius_two)


def test_shuffle_preserves_graph_token_multisets() -> None:
    tokens = [np.arange(12, dtype=np.float64).reshape(4, 3)]
    shuffled = _shuffle_tokens(tokens, seed=7)
    assert np.array_equal(np.sort(tokens[0], axis=0), np.sort(shuffled[0], axis=0))
    assert not np.array_equal(tokens[0], shuffled[0])


def test_zero_init_film_replays_base_path() -> None:
    data = Data(
        x=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        s=torch.tensor([[0.3, -0.2], [0.5, 0.1]]),
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        y=torch.tensor([0]),
    )
    batch = Batch.from_data_list([data])
    torch.manual_seed(5)
    base = NodeFusionGIN(2, 8, 2, layers=2, dropout=0.0, mode="only", struct_dim=0)
    torch.manual_seed(5)
    film = NodeFusionGIN(2, 8, 2, layers=2, dropout=0.0, mode="film", struct_dim=2)
    base.eval()
    film.eval()
    with torch.no_grad():
        assert torch.allclose(base(batch), film(batch), atol=1e-6)

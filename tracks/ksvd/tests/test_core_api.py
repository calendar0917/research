"""Small deterministic smoke tests for the maintained package boundary."""

from __future__ import annotations

import numpy as np

from ksvd_research.core import Graph, from_edges, ksvd, readout_X
from ksvd_research.evaluation import GraphLevelConfig


def test_core_graph_and_ksvd_api() -> None:
    graph = from_edges(3, [(0, 1), (1, 2)])
    assert isinstance(graph, Graph)
    assert graph.n == 3
    assert graph.num_edges() == 2

    y = np.array(
        [[1.0, 0.9, 0.0, 0.1], [0.0, 0.1, 1.0, 0.9], [0.2, 0.0, 0.8, 1.0]],
        dtype=np.float64,
    )
    dictionary, codes, info = ksvd(y, n_atoms=2, T=1, T_min=1, n_iter=2, seed=7)
    assert dictionary.shape == (3, 2)
    assert codes.shape == (2, 4)
    assert info["n_atoms"] == 2
    assert np.isfinite(info["recon_rel"])
    assert readout_X(codes, mode="basic").shape == (6,)


def test_graph_level_config_is_available_from_new_namespace() -> None:
    config = GraphLevelConfig(max_nodes=5, n_atoms=3)
    assert config.max_nodes == 5
    assert config.n_atoms == 3

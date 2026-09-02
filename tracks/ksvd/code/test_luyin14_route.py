from __future__ import annotations

import numpy as np

from .graph import from_edges
from .run_luyin14_route import prepare_graph


def test_prepare_graph_reaches_fair_and_edge100() -> None:
    graph = from_edges(
        12,
        [(index, (index + 1) % 12) for index in range(12)]
        + [(0, 6), (2, 8), (4, 10)],
    )
    prepared = prepare_graph(
        0,
        graph,
        1,
        np.eye(12, 3),
        patch_size=8,
        overlap=2,
        maximum_patches=12,
        retained_beam=4,
        seed=20260812,
    )
    assert prepared.sampling["fair_edge_coverage"] >= 0.95
    assert prepared.sampling["edge100_edge_coverage"] == 1.0
    assert prepared.sampling["edge100_patch_count"] >= prepared.sampling["fair_patch_count"]
    assert prepared.fair.vectors.shape[1] == 28
    assert prepared.node_features is not None


def test_small_graph_uses_single_padded_patch() -> None:
    graph = from_edges(4, [(0, 1), (1, 2), (2, 3)])
    prepared = prepare_graph(
        0,
        graph,
        0,
        None,
        patch_size=8,
        overlap=2,
        maximum_patches=12,
        retained_beam=4,
        seed=20260812,
    )
    assert prepared.fair.vectors.shape == (1, 28)
    assert prepared.edge100.vectors.shape == (1, 28)
    assert prepared.sampling["fair_edge_coverage"] == 1.0

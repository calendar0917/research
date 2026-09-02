from __future__ import annotations

import numpy as np

from .graph import from_edges
from .run_luyin14_raw_relation import _feature_matrix
from .run_luyin14_route import prepare_graph


def test_raw_relation_feature_shapes_and_shuffle() -> None:
    graph = from_edges(
        12,
        [(index, (index + 1) % 12) for index in range(12)]
        + [(0, 6), (2, 8), (4, 10)],
    )
    prepared = prepare_graph(
        0,
        graph,
        1,
        None,
        patch_size=8,
        overlap=2,
        maximum_patches=12,
        retained_beam=4,
        seed=20260812,
    )
    features = _feature_matrix([prepared], [0], shuffle_seed=17)
    assert features["RAW_BAG"].shape == (1, 84)
    assert features["RAW_RELATION_GRAPH"].shape == (1, 92)
    assert features["RAW_TRUE_RELATION"].shape == (1, 102)
    assert features["RAW_SHUFFLED_RELATION"].shape == (1, 102)
    assert np.allclose(
        features["RAW_TRUE_RELATION"][0, :92],
        features["RAW_SHUFFLED_RELATION"][0, :92],
    )


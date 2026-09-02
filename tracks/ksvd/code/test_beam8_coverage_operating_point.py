"""Self-tests for Beam8 coverage operating-point utilities."""
from __future__ import annotations

import numpy as np

from .beam8_coverage_operating_point import (
    edge_categories,
    nondominated_candidates,
    prefix_coverage_trajectory,
    select_operating_checkpoints,
)
from .overlap_cover import _make_cover, _make_patch


def _path(n_nodes: int) -> np.ndarray:
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for node in range(n_nodes - 1):
        adjacency[node, node + 1] = adjacency[node + 1, node] = 1
    return adjacency


def _test_trajectory() -> None:
    adjacency = _path(6)
    cover = _make_cover(
        "test",
        [
            _make_patch(adjacency, (0, 1, 2), 0),
            _make_patch(adjacency, (2, 3, 4), 3),
        ],
    )
    rows = prefix_coverage_trajectory(
        adjacency,
        cover,
        patch_size=3,
        overlap=1,
        maximum_patches=7,
    )
    assert len(rows) == 3
    assert rows[0]["edge_coverage"] == 0.0
    assert rows[1]["covered_edge_count"] == 2
    assert rows[2]["covered_edge_count"] == 4
    assert rows[2]["residual_edge_count"] == 1
    assert rows[2]["last_marginal_new_edges"] == 2
    assert rows[2]["node_coverage"] == 5 / 6
    assert rows[2]["bridge_edge_recall"] == 4 / 5
    assert rows[2]["maximum_residual_incident_count"] == 1
    categories = edge_categories(adjacency)
    assert categories["bridge"] == {
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 5),
    }


def _test_checkpoint_and_pareto() -> None:
    trajectory = [
        {
            "patch_count": 1,
            "edge_coverage": 0.90,
            "node_incident_recall_p10": 0.70,
            "node_coverage": 1.0,
        },
        {
            "patch_count": 2,
            "edge_coverage": 0.95,
            "node_incident_recall_p10": 0.90,
            "node_coverage": 1.0,
        },
        {
            "patch_count": 3,
            "edge_coverage": 1.00,
            "node_incident_recall_p10": 1.00,
            "node_coverage": 1.0,
        },
    ]
    selected = select_operating_checkpoints(trajectory, base_patch_count=1)
    assert selected["BASE"]["patch_count"] == 1
    assert selected["EDGE95"]["patch_count"] == 2
    assert selected["FAIR95"]["patch_count"] == 2
    assert selected["EDGE100"]["patch_count"] == 3

    common = {
        "unique_observed_pair_count": 20,
        "code_scalars": 6,
        "dictionary_scalars": 100,
        "node_incident_recall_p10": 0.9,
        "low_degree_incident_edge_recall": 0.9,
    }
    candidates = [
        {"candidate": "A", "patch_count": 2, "edge_coverage": 0.95, **common},
        {"candidate": "B", "patch_count": 3, "edge_coverage": 0.95, **common},
        {
            "candidate": "C",
            "patch_count": 3,
            "edge_coverage": 0.99,
            **common,
        },
    ]
    selected_labels = nondominated_candidates(candidates)
    assert "B" not in selected_labels
    assert set(selected_labels) == {"A", "C"}


def main() -> int:
    _test_trajectory()
    _test_checkpoint_and_pareto()
    print("beam8_coverage_operating_point self-tests: PASS")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

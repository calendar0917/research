from __future__ import annotations

import unittest

import numpy as np

from .run_molhiv_beam8_cover_diversity_audit import (
    _cover_signature,
    _patch_jaccard,
    _random_bfs_patch,
    classify,
    summarize_cover_family,
)


class MolhivBeam8CoverDiversityAuditTests(unittest.TestCase):
    def test_random_bfs_patch_is_connected_and_full(self) -> None:
        adjacency = np.zeros((10, 10), dtype=np.int8)
        for node in range(9):
            adjacency[node, node + 1] = adjacency[node + 1, node] = 1
        patch = _random_bfs_patch(
            adjacency, root=5, rng=np.random.default_rng(7), patch_size=8
        )
        self.assertEqual(len(patch), 8)
        self.assertEqual(len(set(patch)), 8)
        selected = set(patch)
        self.assertTrue(
            all(
                any(adjacency[node, other] for other in selected if other != node)
                for node in selected
            )
        )

    def test_signature_ignores_patch_and_slot_order(self) -> None:
        left = ((0, 1, 2), (2, 3, 4))
        right = ((4, 3, 2), (2, 1, 0))
        self.assertEqual(_cover_signature(left), _cover_signature(right))
        self.assertEqual(_patch_jaccard(left, right), 1.0)

    def test_summary_detects_occurrence_changes(self) -> None:
        covers = [((0, 1, 2), (2, 3, 4)), ((0, 1, 3), (2, 3, 4))]
        summary = summarize_cover_family(
            covers, n_nodes=5, edges=((0, 1), (1, 2), (2, 3), (3, 4))
        )
        self.assertEqual(summary["unique_covers"], 2)
        self.assertLess(summary["pair_exact_rate"], 1.0)
        self.assertGreater(summary["variable_node_fraction"], 0.0)
        self.assertGreater(summary["variable_edge_fraction"], 0.0)

    def test_classification_thresholds(self) -> None:
        self.assertEqual(
            classify(
                {
                    "changed_graph_fraction_eligible": 0.05,
                    "pair_patch_jaccard_eligible": 0.80,
                }
            ),
            "BEAM_SEED_DIVERSITY_INSUFFICIENT",
        )
        self.assertEqual(
            classify(
                {
                    "changed_graph_fraction_eligible": 0.75,
                    "pair_patch_jaccard_eligible": 0.85,
                }
            ),
            "BEAM_SEED_DIVERSITY_MEANINGFUL",
        )


if __name__ == "__main__":
    unittest.main()

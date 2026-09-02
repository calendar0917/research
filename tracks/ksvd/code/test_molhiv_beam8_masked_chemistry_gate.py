from __future__ import annotations

import unittest

import numpy as np

from .run_molhiv_beam8_masked_chemistry_gate import (
    _limit,
    balanced_bfs_cover_slots,
    classify,
    random_bfs_cover_slots,
)


class MolhivBeam8MaskedChemistryGateTests(unittest.TestCase):
    def test_limit_is_label_free_and_deterministic(self) -> None:
        values = np.arange(100, dtype=np.int64)
        self.assertTrue(np.array_equal(_limit(values, 10, 7), _limit(values, 10, 7)))
        self.assertEqual(len(_limit(values, 10, 7)), 10)

    def test_random_bfs_cover_produces_full_patches(self) -> None:
        adjacency = np.zeros((12, 12), dtype=np.int8)
        for node in range(11):
            adjacency[node, node + 1] = adjacency[node + 1, node] = 1
        typed = adjacency.astype(np.int16)
        types = np.arange(12, dtype=np.int64)
        slots = random_bfs_cover_slots(
            adjacency,
            typed,
            types,
            seed=1,
            patch_size=8,
            overlap=2,
            edge_capacity_multiplier=1.5,
        )
        self.assertTrue(slots)
        self.assertTrue(all(len(patch) == 8 for patch in slots))

    def test_balanced_bfs_cover_produces_full_patches(self) -> None:
        adjacency = np.zeros((12, 12), dtype=np.int8)
        for node in range(11):
            adjacency[node, node + 1] = adjacency[node + 1, node] = 1
        slots = balanced_bfs_cover_slots(
            adjacency,
            adjacency.astype(np.int16),
            np.zeros(12, dtype=np.int64),
            seed=3,
            patch_size=8,
            overlap=2,
            edge_capacity_multiplier=1.5,
            candidate_multiplier=4,
        )
        self.assertTrue(slots)
        self.assertTrue(all(len(patch) == 8 for patch in slots))

    def test_decision_requires_all_three_effects(self) -> None:
        def result(loss: float, accuracy: float):
            return {"heldout": {"mean_cross_entropy": loss, "mean_field_accuracy": accuracy}}

        results = {
            "single_beam": result(1.02, 0.70),
            "multi_beam": result(1.00, 0.72),
            "multi_beam_shuffled": result(1.02, 0.70),
            "multi_random_bfs": result(1.02, 0.70),
        }
        self.assertEqual(
            classify(results)["classification"], "MULTIBEAM_MASKED_CHEMISTRY_PASS"
        )


if __name__ == "__main__":
    unittest.main()

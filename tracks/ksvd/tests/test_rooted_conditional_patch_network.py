from __future__ import annotations

import unittest

import numpy as np

from tracks.ksvd.experiments.luyin16.rooted_conditional_patch_network import (
    _collate,
    _patch_record,
    make_model,
)
from ksvd_research.core.graph import from_edges


class RootedConditionalPatchNetworkTests(unittest.TestCase):
    def _toy(self):
        graph = from_edges(4, [(0, 1), (1, 2), (1, 3)])
        atoms = np.zeros((4, 9), dtype=np.int64)
        atoms[:, 0] = 5  # carbon bucket after the OGB +1 convention
        atoms[0, 0] = 6
        edges = {(0, 1): np.zeros(3, dtype=np.int64),
                 (1, 2): np.ones(3, dtype=np.int64),
                 (1, 3): np.zeros(3, dtype=np.int64)}
        return graph, atoms, edges

    def test_patch_record_has_complete_radius_two_collection(self):
        graph, atoms, edges = self._toy()
        record = _patch_record(graph, atoms, edges, index=7, label=1.0)
        self.assertEqual(record.patch_count, 4)
        self.assertEqual(record.attr_x.shape[1], 40)
        self.assertEqual(record.edge_attr.shape[1], 13)
        self.assertEqual(int(record.root_mask.sum()), record.patch_count)
        self.assertEqual(record.node_patch.max(), record.patch_count - 1)

    def test_collate_offsets_patch_and_node_segments(self):
        graph, atoms, edges = self._toy()
        first = _patch_record(graph, atoms, edges, index=1, label=0.0)
        second = _patch_record(graph, atoms, edges, index=2, label=1.0)
        batch = _collate([first, second])
        self.assertEqual(batch["n_graphs"], 2)
        self.assertEqual(batch["n_patches"], 8)
        self.assertEqual(int(batch["graph_patch"].max()), 1)
        self.assertEqual(int(batch["node_patch"].max()), 7)
        self.assertTrue(np.array_equal(batch["patch_shuffle"].numpy()[:4], first.patch_shuffle))
        self.assertTrue(np.array_equal(batch["patch_shuffle"].numpy()[4:], second.patch_shuffle + 4))

    def test_size_matched_shuffle_preserves_patch_sizes(self):
        graph, atoms, edges = self._toy()
        record = _patch_record(graph, atoms, edges, index=5, label=0.0)
        node_counts = np.bincount(record.node_patch, minlength=record.patch_count)
        edge_counts = np.bincount(record.edge_patch, minlength=record.patch_count)
        for destination, source in enumerate(record.patch_shuffle_matched.tolist()):
            self.assertEqual(int(node_counts[destination]), int(node_counts[source]))
            self.assertEqual(int(edge_counts[destination]), int(edge_counts[source]))

    def test_model_variants_forward(self):
        import torch

        graph, atoms, edges = self._toy()
        record = _patch_record(graph, atoms, edges, index=3, label=1.0)
        batch = _collate([record])
        for variant in ("structure", "attribute", "concat", "conditional", "conditional_film", "cross_attention"):
            model = make_model(variant, hidden=16, struct_layers=2, experts=3, dropout=0.0)
            logits = model(batch)
            self.assertEqual(tuple(logits.shape), (1,))
            self.assertTrue(torch.isfinite(logits).all())
        model = make_model("conditional", hidden=16, struct_layers=2, experts=3, dropout=0.0)
        logits = model(batch, pair_shuffle=True)
        self.assertEqual(tuple(logits.shape), (1,))

        model = make_model("cross_attention", hidden=16, struct_layers=2, experts=3, dropout=0.0)
        logits = model(batch, pair_shuffle=True)
        self.assertEqual(tuple(logits.shape), (1,))

    def test_film_is_initialized_as_attribute_path(self):
        import torch

        graph, atoms, edges = self._toy()
        record = _patch_record(graph, atoms, edges, index=4, label=1.0)
        batch = _collate([record])
        model = make_model("conditional_film", hidden=16, struct_layers=2, experts=3, dropout=0.0)
        with torch.no_grad():
            attribute = model.encode_attribute(batch)
            fused = model.fuse(model.encode_structure(batch), attribute, batch, False)
        torch.testing.assert_close(fused, attribute)


if __name__ == "__main__":
    unittest.main()

"""Self-tests for typed MolHIV Beam8 atom--patch incidence construction."""

from __future__ import annotations

import unittest

import numpy as np

from .graph import from_edges
from .molhiv_beam8_incidence import (
    build_molhiv_beam8_incidence,
    incidence_summary,
    typed_bond_adjacency,
)


ATOM_DIMS = (20, 4)
BOND_DIMS = (3, 2)


def _molecule(n: int = 11):
    edges = [(index, index + 1) for index in range(n - 1)]
    if n >= 9:
        edges.extend([(1, 5), (5, 8)])
    graph = from_edges(n, edges)
    atoms = np.asarray([[index, index % 4] for index in range(n)], dtype=np.int64)
    bonds = {
        (left, right): np.asarray([(left + right) % 3, (right - left) % 2], dtype=np.int64)
        for left, right in graph.edges()
    }
    return graph, atoms, bonds


def _coverage_molecule(n: int = 20):
    edges = {(index, index + 1) for index in range(n - 1)}
    edges.update(
        tuple(sorted((index, (index + 5) % n)))
        for index in range(n)
        if index != (index + 5) % n
    )
    graph = from_edges(n, sorted(edges))
    atoms = np.asarray([[index, index % 4] for index in range(n)], dtype=np.int64)
    bonds = {
        (left, right): np.asarray([(left + right) % 3, (right - left) % 2], dtype=np.int64)
        for left, right in graph.edges()
    }
    return graph, atoms, bonds


def _relabel(graph, atoms, bonds, permutation):
    inverse = np.empty(len(permutation), dtype=np.int64)
    inverse[permutation] = np.arange(len(permutation))
    edges = [(int(inverse[left]), int(inverse[right])) for left, right in graph.edges()]
    relabeled = from_edges(graph.n, edges)
    relabeled_bonds = {}
    for left, right in graph.edges():
        new_left, new_right = int(inverse[left]), int(inverse[right])
        key = (new_left, new_right) if new_left < new_right else (new_right, new_left)
        relabeled_bonds[key] = bonds[(left, right)].copy()
    return relabeled, atoms[permutation], relabeled_bonds


class MolhivBeam8IncidenceTest(unittest.TestCase):
    def test_complete_typed_bond_codes(self) -> None:
        graph, _atoms, bonds = _molecule()
        typed = typed_bond_adjacency(graph, bonds, BOND_DIMS)
        self.assertTrue(np.array_equal(typed, typed.T))
        self.assertTrue(np.all(np.diag(typed) == 0))
        self.assertEqual(int(np.count_nonzero(typed) // 2), len(graph.edges()))
        self.assertGreaterEqual(int(typed[0, 1]), 1)
        self.assertLessEqual(int(typed.max()), int(np.prod(BOND_DIMS)))

    def test_relabel_equivariance_with_unique_full_atom_rows(self) -> None:
        graph, atoms, bonds = _molecule()
        permutation = np.asarray([7, 2, 10, 0, 4, 8, 1, 9, 5, 3, 6], dtype=np.int64)
        rel_graph, rel_atoms, rel_bonds = _relabel(graph, atoms, bonds, permutation)
        for checkpoint in ("base", "fair95", "edge100"):
            with self.subTest(checkpoint=checkpoint):
                base = build_molhiv_beam8_incidence(
                    7,
                    graph,
                    1.0,
                    atoms,
                    bonds,
                    atom_feature_dims=ATOM_DIMS,
                    bond_feature_dims=BOND_DIMS,
                    coverage_checkpoint=checkpoint,
                )
                relabeled = build_molhiv_beam8_incidence(
                    7,
                    rel_graph,
                    1.0,
                    rel_atoms,
                    rel_bonds,
                    atom_feature_dims=ATOM_DIMS,
                    bond_feature_dims=BOND_DIMS,
                    coverage_checkpoint=checkpoint,
                )

                mapped_slots = tuple(
                    tuple(int(permutation[node]) for node in nodes)
                    for nodes in relabeled.slot_nodes
                )
                self.assertEqual(mapped_slots, base.slot_nodes)
                self.assertEqual(relabeled.segment_ids, base.segment_ids)
                self.assertEqual(relabeled.component_ids, base.component_ids)
                self.assertTrue(
                    np.array_equal(
                        relabeled.patch_is_completion, base.patch_is_completion
                    )
                )
                self.assertTrue(
                    np.array_equal(relabeled.patch_slot_features, base.patch_slot_features)
                )
                self.assertTrue(
                    np.array_equal(relabeled.patch_slot_mask, base.patch_slot_mask)
                )
                self.assertTrue(np.array_equal(relabeled.chain_index, base.chain_index))
                self.assertTrue(np.allclose(relabeled.chain_features, base.chain_features))
                self.assertTrue(
                    np.allclose(
                        relabeled.mapping_shuffled_chain_features,
                        base.mapping_shuffled_chain_features,
                    )
                )
                self.assertTrue(
                    np.array_equal(relabeled.overlap_index, base.overlap_index)
                )
                self.assertTrue(
                    np.allclose(relabeled.overlap_features, base.overlap_features)
                )
                self.assertTrue(
                    np.allclose(
                        relabeled.mapping_shuffled_overlap_features,
                        base.mapping_shuffled_overlap_features,
                    )
                )
                self.assertTrue(
                    np.array_equal(
                        relabeled.base_overlap_index, base.base_overlap_index
                    )
                )
                mapped_incidence = relabeled.incidence_index.copy()
                mapped_incidence[0] = permutation[mapped_incidence[0]]
                self.assertTrue(np.array_equal(mapped_incidence, base.incidence_index))
                self.assertTrue(
                    np.array_equal(relabeled.incidence_slot, base.incidence_slot)
                )
                self.assertTrue(
                    np.array_equal(relabeled.incidence_flags, base.incidence_flags)
                )
                mapped_endpoints = relabeled.bond_endpoint_index.copy()
                mapped_endpoints[1] = permutation[mapped_endpoints[1]]
                self.assertTrue(
                    np.array_equal(mapped_endpoints, base.bond_endpoint_index)
                )
                self.assertTrue(
                    np.array_equal(
                        relabeled.bond_endpoint_pair, base.bond_endpoint_pair
                    )
                )
                self.assertTrue(
                    np.array_equal(
                        relabeled.bond_endpoint_slot, base.bond_endpoint_slot
                    )
                )
                self.assertTrue(
                    np.array_equal(
                        relabeled.bond_endpoint_features,
                        base.bond_endpoint_features,
                    )
                )

    def test_coverage_checkpoints_and_completion_segments(self) -> None:
        graph, atoms, bonds = _coverage_molecule()
        values = {
            checkpoint: build_molhiv_beam8_incidence(
                13,
                graph,
                0.0,
                atoms,
                bonds,
                atom_feature_dims=ATOM_DIMS,
                bond_feature_dims=BOND_DIMS,
                coverage_checkpoint=checkpoint,
            )
            for checkpoint in ("base", "fair95", "edge100")
        }
        self.assertLess(values["base"].sampling["edge_coverage"], 0.95)
        self.assertGreaterEqual(values["fair95"].sampling["edge_coverage"], 0.95)
        self.assertGreaterEqual(values["fair95"].sampling["incident_p10"], 0.90)
        self.assertEqual(values["fair95"].sampling["node_coverage"], 1.0)
        self.assertEqual(values["edge100"].sampling["edge_coverage"], 1.0)
        self.assertGreater(int(values["fair95"].patch_is_completion.sum()), 0)

        for checkpoint in ("fair95", "edge100"):
            item = values[checkpoint]
            completion_indices = set(np.flatnonzero(item.patch_is_completion).tolist())
            self.assertTrue(completion_indices)
            segment_counts = {
                segment: item.segment_ids.count(segment)
                for segment in set(item.segment_ids)
            }
            self.assertTrue(
                all(segment_counts[item.segment_ids[index]] == 1 for index in completion_indices)
            )
            self.assertTrue(
                completion_indices.isdisjoint(set(item.chain_index.reshape(-1).tolist()))
            )

    def test_controls_preserve_required_marginals(self) -> None:
        graph, atoms, bonds = _molecule()
        item = build_molhiv_beam8_incidence(
            11,
            graph,
            0.0,
            atoms,
            bonds,
            atom_feature_dims=ATOM_DIMS,
            bond_feature_dims=BOND_DIMS,
        )
        source, target = item.token_shuffle_index
        self.assertEqual(sorted(source.tolist()), list(range(item.n_patches)))
        self.assertEqual(sorted(target.tolist()), list(range(item.n_patches)))
        if item.chain_index.shape[1]:
            true_degree = np.bincount(
                item.chain_index.reshape(-1), minlength=item.n_patches
            )
            shuffled_degree = np.bincount(
                item.shuffled_chain_index.reshape(-1), minlength=item.n_patches
            )
            self.assertEqual(sorted(true_degree.tolist()), sorted(shuffled_degree.tolist()))
            true_mapping = item.chain_features[:, 2:].reshape(-1, 8, 8)
            shuffled_mapping = item.mapping_shuffled_chain_features[:, 2:].reshape(
                -1, 8, 8
            )
            self.assertTrue(
                np.array_equal(item.chain_features[:, :2], item.mapping_shuffled_chain_features[:, :2])
            )
            self.assertTrue(
                np.array_equal(true_mapping.sum(axis=1), shuffled_mapping.sum(axis=1))
            )
            self.assertTrue(
                np.array_equal(true_mapping.sum(axis=2), shuffled_mapping.sum(axis=2))
            )
        if item.overlap_index.shape[1]:
            true_overlap = item.overlap_features[:, 2:].reshape(-1, 8, 8)
            shuffled_overlap = item.mapping_shuffled_overlap_features[:, 2:].reshape(
                -1, 8, 8
            )
            self.assertTrue(
                np.array_equal(
                    item.overlap_features[:, :2],
                    item.mapping_shuffled_overlap_features[:, :2],
                )
            )
            self.assertTrue(
                np.array_equal(
                    true_overlap.sum(axis=1), shuffled_overlap.sum(axis=1)
                )
            )
            self.assertTrue(
                np.array_equal(
                    true_overlap.sum(axis=2), shuffled_overlap.sum(axis=2)
                )
            )

    def test_small_graph_is_one_patch(self) -> None:
        graph, atoms, bonds = _molecule(5)
        item = build_molhiv_beam8_incidence(
            3,
            graph,
            0.0,
            atoms,
            bonds,
            atom_feature_dims=ATOM_DIMS,
            bond_feature_dims=BOND_DIMS,
        )
        self.assertEqual(item.n_patches, 1)
        self.assertEqual(item.incidence_index.shape[1], graph.n)
        self.assertEqual(item.chain_index.shape, (2, 0))
        summary = incidence_summary([item])
        self.assertEqual(summary["graphs"], 1)
        self.assertEqual(summary["mean_patches"], 1.0)


if __name__ == "__main__":
    unittest.main()

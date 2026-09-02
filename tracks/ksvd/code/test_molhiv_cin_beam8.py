from __future__ import annotations

import unittest

import numpy as np
import torch
from torch_geometric.data import HeteroData

from .graph import from_edges
from .molhiv_cin_beam8 import add_ring_complex


def _edge_features(edges: list[tuple[int, int]]) -> dict[tuple[int, int], np.ndarray]:
    return {tuple(sorted(edge)): np.asarray([0, 0, 0], dtype=np.int64) for edge in edges}


class MolhivCINBeam8Test(unittest.TestCase):
    def test_six_cycle_lift(self) -> None:
        edges = [(i, (i + 1) % 6) for i in range(6)]
        graph = from_edges(6, edges)
        data = add_ring_complex(
            HeteroData(), graph, _edge_features(edges), graph_index=0
        )
        self.assertEqual(data["bond_cell"].num_nodes, 6)
        self.assertEqual(data["ring"].num_nodes, 1)
        self.assertEqual(
            data["atom", "cell_boundary", "bond_cell"].edge_index.shape[1], 12
        )
        self.assertEqual(
            data["bond_cell", "cell_boundary", "ring"].edge_index.shape[1], 6
        )
        self.assertEqual(
            data["bond_cell", "cell_upper", "bond_cell"].edge_index.shape[1], 30
        )

    def test_chord_removes_noninduced_cycle(self) -> None:
        edges = [(0, 1), (1, 2), (2, 3), (3, 0), (0, 2)]
        graph = from_edges(4, edges)
        data = add_ring_complex(
            HeteroData(), graph, _edge_features(edges), graph_index=1
        )
        self.assertEqual(data["ring"].num_nodes, 2)
        self.assertEqual(data["ring"].size.tolist(), [3, 3])

    def test_ring_shuffle_preserves_marginals(self) -> None:
        edges = [
            (0, 1), (1, 2), (2, 0),
            (2, 3), (3, 4), (4, 2),
            (4, 5), (5, 6), (6, 4),
        ]
        graph = from_edges(7, edges)
        data = add_ring_complex(
            HeteroData(), graph, _edge_features(edges), graph_index=2
        )
        source, target = data["ring", "token_shuffle", "ring"].edge_index
        self.assertEqual(sorted(source.tolist()), list(range(3)))
        self.assertEqual(sorted(target.tolist()), list(range(3)))
        self.assertFalse(np.array_equal(source.numpy(), target.numpy()))

    def test_relabeling_preserves_cell_counts(self) -> None:
        edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0), (1, 5)]
        permutation = {0: 3, 1: 5, 2: 1, 3: 4, 4: 0, 5: 2}
        relabeled = [(permutation[u], permutation[v]) for u, v in edges]
        first = add_ring_complex(
            HeteroData(), from_edges(6, edges), _edge_features(edges), graph_index=3
        )
        second = add_ring_complex(
            HeteroData(),
            from_edges(6, relabeled),
            _edge_features(relabeled),
            graph_index=3,
        )
        for node_type in ("bond_cell", "ring"):
            self.assertEqual(first[node_type].num_nodes, second[node_type].num_nodes)
        self.assertEqual(
            first["bond_cell", "cell_boundary", "ring"].edge_index.shape[1],
            second["bond_cell", "cell_boundary", "ring"].edge_index.shape[1],
        )

    def test_patch_endpoints_map_to_bond_cells(self) -> None:
        edges = [(0, 1), (1, 2), (2, 0), (2, 3)]
        data = HeteroData()
        data["patch"].num_nodes = 2
        data["patch", "bond_endpoint", "atom"].edge_index = torch.tensor(
            [[0, 0, 1, 1], [0, 1, 2, 3]], dtype=torch.long
        )
        lifted = add_ring_complex(
            data, from_edges(4, edges), _edge_features(edges), graph_index=4
        )
        bonds = tuple(sorted(from_edges(4, edges).edges()))
        lookup = {edge: index for index, edge in enumerate(bonds)}
        expected = [lookup[(0, 1)], lookup[(0, 1)], lookup[(2, 3)], lookup[(2, 3)]]
        self.assertEqual(
            lifted["patch", "bond_endpoint", "atom"].bond_cell.tolist(), expected
        )


if __name__ == "__main__":
    unittest.main()

"""Focused CPU tests for PEC-v0 (``pec_v0``).

The Gate-0 correctness checks live in
``tracks/ksvd/experiments/luyin16/pec_v0_gate0.py`` so that the runner's
``gate0`` stage and this pytest module exercise exactly the same code.
Data-free: no ZINC data, no checkpoints, no official test.
"""

from __future__ import annotations

import numpy as np
import pytest

from tracks.ksvd.experiments.luyin16 import pec_v0 as pec
from tracks.ksvd.experiments.luyin16 import pec_v0_gate0 as gate0
from tracks.ksvd.experiments.luyin16.pec_v0_gate0 import (
    _batch,
    _chain,
    _chem,
    _ring,
    _sample,
)

GATE0_CHECKS = gate0.GATE0_CHECKS


@pytest.mark.parametrize("name", sorted(GATE0_CHECKS))
def test_gate0(name: str) -> None:
    GATE0_CHECKS[name]()


def test_parameter_accounting() -> None:
    counts = gate0.check_dense_matched_params()
    assert counts["sparse"] == counts["dense"]
    mismatch = abs(counts["coarse"] - counts["sparse"]) / counts["sparse"]
    assert mismatch <= 0.02, counts


def test_role_basis_is_audited_fsar_basis() -> None:
    """Occurrence roles must be exactly the FSAR explicit rooted basis rows."""
    from tracks.ksvd.experiments.luyin16 import fsar_v2 as v2

    graph = _ring(7)
    atom_types, edge_types = _chem(graph, (0, 1, 2, 3), 1)
    sample = pec.build_sample(
        graph, [atom_types[node] for node in graph.nodes], edge_types
    )
    node_parts = []
    edge_parts = []
    for root in graph.nodes:
        _nodes, _index, node_basis, edge_basis, edges = v2._explicit_basis_for_patch(
            graph, int(root), pec.PATCH_RADIUS
        )
        node_parts.append(np.asarray(node_basis, dtype=np.float32))
        edge_parts.append(np.asarray(edge_basis, dtype=np.float32))
        for left, right in edges:
            shell_a = int(np.argmax(node_basis[_nodes.index(int(left)), 1:4]))
            shell_b = int(np.argmax(node_basis[_nodes.index(int(right)), 1:4]))
            assert pec.shellpair_index(shell_a, shell_b) in range(pec.SHELLPAIR_CLASSES)
    expected_node = np.concatenate(node_parts, axis=0)
    expected_edge = np.concatenate(edge_parts, axis=0)
    assert np.array_equal(sample.node_basis, expected_node)
    assert np.array_equal(sample.edge_basis, expected_edge)
    assert sample.node_basis.shape[1] == pec.NODE_ROLE_DIM
    assert sample.edge_basis.shape[1] == pec.EDGE_ROLE_DIM


def test_pair_relation_is_topology_only() -> None:
    """Changing only chemistry must not change any pair-relation column."""
    graph = _ring(6)
    base = _sample(graph, atom_categories=(0, 1, 2, 3), bond_category=1)
    changed = _sample(graph, atom_categories=(4, 5, 6, 7), bond_category=3)
    assert np.array_equal(base.pair_rho, changed.pair_rho)
    assert np.array_equal(base.root_scalars, changed.root_scalars)
    assert np.array_equal(base.global_topo, changed.global_topo)
    assert base.pair_rho.shape[1] == pec.TOPO_PAIR_DIM


def test_collate_shapes() -> None:
    batch = _batch([_sample(_ring(6)), _sample(_chain(4))])
    assert batch["node_basis"].shape[1] == pec.NODE_ROLE_DIM
    assert batch["edge_basis"].shape[1] == pec.EDGE_ROLE_DIM
    assert batch["pair_rho"].shape[1] == pec.TOPO_PAIR_DIM
    assert batch["n_graphs"].item() == 2

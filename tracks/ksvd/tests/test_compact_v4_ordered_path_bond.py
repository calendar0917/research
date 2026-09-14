"""Tests for the compact-v4 ordered shortest-path bond encoder experiment.

Static / unit tests for the Phase-A audit helpers and the Phase-B/C model.
They never train to completion and never load official test.  Synthetic
molecules are used for structural checks; the large path-count cache is only
touched if it already exists.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges, ring_chords
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_ordered_path_bond as opb,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre as rec,
)
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _path_counts(n_pairs: int) -> torch.Tensor:
    return torch.zeros(n_pairs, opb.PATH_MAX_LEN, opb.BOND_CATEGORIES)


def _synthetic_batch():
    topo_width = int(ztopo.raw_width("hinge"))
    datasets: list[Data] = []
    for graph in (
        ring_chords(6, []),
        ring_chords(5, []),
        from_edges(6, [(0, 1), (1, 2), (2, 0)]),
    ):
        n = len(graph.nodes)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        pair_index = (
            torch.tensor(pairs, dtype=torch.long).t().contiguous()
            if pairs
            else torch.zeros(2, 0, dtype=torch.long)
        )
        buckets = torch.tensor(
            [min(max(i % 5, 0), 4) for i in range(len(pairs))], dtype=torch.long
        )
        counts = _path_counts(len(pairs))
        if len(pairs):
            counts[:, 0, 1] = 1.0
        datasets.append(
            Data(
                patch_cont=torch.zeros(n, int(zpp.SHELL_WIDTH)),
                patch_context=torch.zeros(n, 0),
                typed_token=torch.zeros(n, dtype=torch.long),
                parent_token=torch.zeros(n, dtype=torch.long),
                structural_token=torch.zeros(n, dtype=torch.long),
                structural_coarse=torch.zeros(n, 0),
                pair_index=pair_index,
                pair_relation=torch.zeros(len(pairs), int(zpp.RELATION_WIDTH)),
                pair_bucket=buckets,
                global_context=torch.zeros(1, int(zpp.GLOBAL_WIDTH)),
                topology_features=torch.zeros(1, topo_width),
                y=torch.zeros(1),
                num_nodes=n,
                pair_path_counts=counts,
                pair_path_counts_sorted=counts.clone(),
            )
        )
    return next(iter(zpp._make_loader(datasets, 128, False, 0)))


def _raw_triangle():
    """A 3-node path 0-1-2 with bond types 1, 2 (plus reverse edges)."""
    return Data(
        x=torch.tensor([[0], [1], [2]], dtype=torch.float32),
        edge_index=torch.tensor(
            [[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long
        ),
        edge_attr=torch.tensor([1, 1, 2, 2], dtype=torch.long),
        y=torch.zeros(1),
        num_nodes=3,
    )


@pytest.fixture(scope="module")
def batch():
    return _synthetic_batch()


@pytest.fixture(scope="module")
def ordered():
    model = opb.build_ordered(0)
    model.eval()
    return model


@pytest.fixture(scope="module")
def orderless():
    model = opb.build_orderless(0)
    model.eval()
    return model


@pytest.fixture(scope="module")
def recurrent():
    model = rec.build_recurrent(0)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Phase A helpers
# ---------------------------------------------------------------------------


def test_canonical_bond_is_reversal_invariant():
    for sequence in [(1, 2, 3), (3, 2, 1), (1, 1, 2), (2, 1, 1), (1, 2, 1)]:
        assert opb._canonical_bond(sequence) == opb._canonical_bond(sequence[::-1])
    assert opb._canonical_bond((1, 2, 3)) == (1, 2, 3)
    assert opb._canonical_bond((3, 2, 1)) == (1, 2, 3)
    assert opb._canonical_bond((2, 1, 3)) == (2, 1, 3)  # reverse is (3, 1, 2)


def test_process_graph_counts_for_a_path():
    counts, counts_sorted = opb._process_graph(_raw_triangle(), opb.PATH_MAX_LEN, opb.ENUM_CAP)
    # pairs in order: (0,1), (0,2), (1,2)
    # (0,1): bond 1 at position 0
    assert counts[0, 0, 1] == pytest.approx(1.0)
    # (0,2): unique path with bonds (1, 2); canonical (1,2) <= (2,1)
    assert counts[1, 0, 1] == pytest.approx(1.0)
    assert counts[1, 1, 2] == pytest.approx(1.0)
    assert counts[1].sum() == pytest.approx(2.0)
    # orderless control sorts the same path -> identical here
    assert np.allclose(counts, counts_sorted)


def test_orderless_differs_from_ordered_on_asymmetric_path():
    data = Data(
        x=torch.zeros(4, 1),
        edge_index=torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        ),
        edge_attr=torch.tensor([1, 1, 2, 2, 1, 1], dtype=torch.long),
        y=torch.zeros(1),
        num_nodes=4,
    )
    counts, counts_sorted = opb._process_graph(data, opb.PATH_MAX_LEN, opb.ENUM_CAP)
    # pair (0,3): path bonds (1, 2, 1); canonical (1,2,1)
    assert counts[2, 0, 1] == pytest.approx(1.0)
    assert counts[2, 1, 2] == pytest.approx(1.0)
    assert counts[2, 2, 1] == pytest.approx(1.0)
    # orderless: sorted (1,1,2)
    assert counts_sorted[2, 0, 1] == pytest.approx(1.0)
    assert counts_sorted[2, 1, 1] == pytest.approx(1.0)
    assert counts_sorted[2, 2, 2] == pytest.approx(1.0)
    assert not np.allclose(counts[2], counts_sorted[2])


def test_shortest_path_enumeration_counts_multiplicity():
    # Square with a bond-type difference: 0-1-2 and 0-3-2 are two shortest paths.
    graph = from_edges(4, [(0, 1), (1, 2), (0, 3), (3, 2)])
    edge_bonds = {
        (0, 1): 1,
        (1, 2): 2,
        (0, 3): 2,
        (2, 3): 1,
    }
    adj_bond = {u: [] for u in graph.nodes}
    for (u, v), bond in edge_bonds.items():
        adj_bond[u].append((v, bond))
        adj_bond[v].append((u, bond))
    for key in adj_bond:
        adj_bond[key] = sorted(adj_bond[key])
    distances, parents = opb._bfs_parents(adj_bond, 0)
    paths = opb._enumerate_shortest_paths(0, 2, parents, opb.ENUM_CAP)
    assert len(paths) == 2
    bond_sequences = sorted(opb._bond_sequence(p, graph, edge_bonds) for p in paths)
    assert bond_sequences == [(1, 2), (2, 1)]


# ---------------------------------------------------------------------------
# Phase B model
# ---------------------------------------------------------------------------


def test_parameter_match(ordered, orderless, recurrent):
    n_ordered = opb._n_params(ordered)
    n_orderless = opb._n_params(orderless)
    n_recurrent = opb._n_params(recurrent)
    assert n_ordered == n_orderless
    assert n_ordered > n_recurrent
    assert n_ordered - n_recurrent < 5000


def test_path_off_equals_recurrent(ordered, recurrent, batch):
    ordered.force_zero_path = True
    try:
        with torch.no_grad():
            r_ordered = ordered.encode(batch)
            r_recurrent = recurrent.encode(batch)
    finally:
        ordered.force_zero_path = False
    assert float((r_ordered - r_recurrent).abs().max()) == 0.0
    assert r_ordered.shape[1] == 302


def test_baseline_pair_encoder_columns_preserved(ordered, recurrent):
    width = int(recurrent.pair_encoder.layers[0].weight.shape[1])
    restored = ordered.pair_encoder.layers[0].weight[:, :width]
    baseline = recurrent.pair_encoder.layers[0].weight
    assert float((restored - baseline).abs().max()) == 0.0


def test_path_branch_is_alive_and_finite(ordered, batch):
    ordered.train()
    out = ordered(batch).view(-1)
    assert torch.isfinite(out).all()
    out.mean().backward()
    path_grad = ordered.path_encoder[0].weight.grad
    assert path_grad is not None
    assert float(path_grad.abs().max()) > 0.0
    for parameter in ordered.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_ordered_changes_readout(ordered, batch):
    ordered.force_zero_path = True
    try:
        with torch.no_grad():
            off = ordered.encode(batch)
    finally:
        ordered.force_zero_path = False
    with torch.no_grad():
        on = ordered.encode(batch)
    assert float((on - off).abs().max()) > 0.0


def test_aggregation_is_permutation_invariant(ordered):
    counts_a = _path_counts(1)
    counts_a[0, 0, 1] = 0.5
    counts_a[0, 0, 2] = 0.5
    counts_b = _path_counts(1)
    counts_b[0, 0, 2] = 0.5
    counts_b[0, 0, 1] = 0.5
    data_a, data_b = Data(), Data()
    data_a.pair_path_counts = counts_a
    data_b.pair_path_counts = counts_b
    with torch.no_grad():
        encoding_a = ordered._compute_path_encoding(data_a)
        encoding_b = ordered._compute_path_encoding(data_b)
    assert float((encoding_a - encoding_b).abs().max()) == 0.0


def test_no_cross_graph_leakage(ordered):
    # The path encoder is strictly row-wise: a permutation of rows permutes
    # the encodings identically (no row mixing).
    counts = torch.rand(7, opb.PATH_MAX_LEN, opb.BOND_CATEGORIES)
    data = Data()
    data.pair_path_counts = counts
    with torch.no_grad():
        encoding = ordered._compute_path_encoding(data)
    permutation = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    data_perm = Data()
    data_perm.pair_path_counts = counts[permutation]
    with torch.no_grad():
        encoding_perm = ordered._compute_path_encoding(data_perm)
    # Row-wise application: permuting rows permutes encodings exactly up to
    # floating-point reduction order (no row mixing).
    assert float((encoding_perm - encoding[permutation]).abs().max()) < 1e-5


def test_orderless_uses_sorted_field(orderless):
    counts = _path_counts(2)
    counts_sorted = _path_counts(2)
    counts[0, 0, 1] = 1.0
    counts_sorted[0, 0, 2] = 1.0
    data = Data()
    data.pair_path_counts = counts
    data.pair_path_counts_sorted = counts_sorted
    with torch.no_grad():
        from_ordered = orderless._compute_path_encoding(data)
        data.pair_path_counts_sorted = counts.clone()
        from_ordered_again = orderless._compute_path_encoding(data)
    # Changing only the sorted field must change the orderless encoding.
    assert float((from_ordered - from_ordered_again).abs().max()) > 0.0

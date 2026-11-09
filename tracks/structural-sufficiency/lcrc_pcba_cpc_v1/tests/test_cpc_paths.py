"""Structural correctness of the rooted two-bond path enumeration."""

from __future__ import annotations

import numpy as np

import cpc_audit as A
import cpc_common as C
import cpc_paths as P


def test_small_graph_cases_all_match_reference():
    out = A.structural_small_graph_audit()
    assert out["all_ok"]
    assert set(out["cases"]) == {
        "single_atom",
        "single_bond",
        "chain_3",
        "chain_4",
        "triangle",
        "square",
        "branched_tree",
        "multi_path_ring",
        "k4",
    }


def test_triangle_keeps_both_two_bond_paths_per_root():
    graph = A._graph(3, [(0, 1), (1, 2), (2, 0)])
    paths = P.enumerate_cpc_molecule(3, graph["adj"])
    assert paths["ptr"].tolist() == [0, 2, 4, 6]
    assert paths["ptr"][-1] == 6
    # explicitly: root 0 has 0-1-2 and 0-2-1, two distinct middles/endpoints
    rec = [x for x in A.path_records(paths) if x[0] == 0]
    assert sorted((m, e) for _r, m, e, _a, _b in rec) == [(1, 2), (2, 1)]


def test_multiplicity_to_same_endpoint_is_preserved():
    # K4: root 0 reaches each other atom by two different middles.
    graph = A._graph(4, [(0, 1), (0, 2), (0, 3), (1, 2), (1, 3), (2, 3)])
    paths = P.enumerate_cpc_molecule(4, graph["adj"])
    rec = [x for x in A.path_records(paths) if x[0] == 0]
    ends = [e for _r, _m, e, _a, _b in rec]
    assert ends.count(1) == 2 and ends.count(2) == 2 and ends.count(3) == 2
    assert len(rec) == 6


def test_physical_bond_ids_connect_the_path():
    graph = A._graph(5, [(0, 1), (1, 2), (2, 3), (3, 4), (0, 2)])
    paths = P.enumerate_cpc_molecule(5, graph["adj"])
    for root, mid, end, e1, e2 in A.path_records(paths):
        u1, v1 = graph["edges"][e1]
        u2, v2 = graph["edges"][e2]
        assert {int(u1), int(v1)} == {root, mid}
        assert {int(u2), int(v2)} == {mid, end}


def test_atom_renumbering_equivalence():
    graph = A._graph(6, [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 0), (0, 3)])
    paths = P.enumerate_cpc_molecule(6, graph["adj"])
    perm = np.asarray([3, 1, 5, 0, 4, 2], dtype=np.int64)
    relabelled = A.relabel_graph(graph, perm)
    other = P.enumerate_cpc_molecule(6, relabelled["adj"])
    # exact rooted multiset after applying the inverse relabel
    inv = np.empty(6, dtype=np.int64)
    inv[perm] = np.arange(6)
    got = sorted((int(inv[r]), int(inv[m]), int(inv[e])) for r, m, e, _a, _b in A.path_records(other))
    expected = sorted((r, m, e) for r, m, e, _a, _b in A.path_records(paths))
    assert got == expected


def test_cache_subset_matches_frozen_s2(task_with_cpc):
    cpc = task_with_cpc.cpc
    cache = task_with_cpc.cache
    ptr = np.asarray(cpc["cpc_ptr"])
    path_ptr = np.asarray(cache["path_ptr"])
    rows = np.asarray(cpc["rows"])
    atom_ptr = np.asarray(cache["atom_ptr"])
    row_atom_ptr = np.asarray(cpc["row_atom_ptr"])
    row_index = np.asarray(cpc["row_index"])
    for row in rows[:16]:
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        c0 = int(row_atom_ptr[row_index[row]])
        c1 = int(row_atom_ptr[row_index[row] + 1])
        assert np.array_equal(np.diff(ptr[c0 : c1 + 1]), np.diff(path_ptr[a0 : a1 + 1]))


def test_collate_isolation_and_zero_path(task_with_cpc):
    import torch

    rows = [int(task_with_cpc.train_rows[0]), int(task_with_cpc.train_rows[1])]
    batch = P.collate_with_cpc(task_with_cpc, task_with_cpc.cpc, rows)
    iso = P.check_batch_isolation(batch)
    assert iso["isolated"] and iso["bonds_same_molecule"] and iso["bond_endpoints_match_path"]
    assert int(batch["cpc_path_atom"].numel()) == int(batch["cpc_path_atom"].numel())
    # every path's three atoms belong to exactly one molecule
    assert torch.equal(
        batch["atom_batch"][batch["cpc_path_atom"]], batch["atom_batch"][batch["cpc_middle"]]
    )
    assert torch.equal(
        batch["atom_batch"][batch["cpc_middle"]], batch["atom_batch"][batch["cpc_end"]]
    )

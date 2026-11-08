"""Pair index: bucket definition, legality, packing and the batch gather."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import pw_common as C
import pw_pairs as D
from conftest import TINY_GRAPHS, synthetic_batch


def test_bucket_edges_match_protocol():
    assert [tuple(b) for b in C.BUCKETS] == [
        ("d1", 1, 1),
        ("d2", 2, 2),
        ("d3", 3, 3),
        ("d4", 4, 4),
        ("d5_6", 5, 6),
        ("d7_plus", 7, None),
    ]


@pytest.mark.parametrize("distance,code", [(1, 0), (2, 1), (3, 2), (4, 3), (5, 4), (6, 4), (7, 5), (19, 5)])
def test_bucket_of_distance(distance, code):
    assert D.bucket_of_distance(distance) == code


def test_bucket_of_distance_rejects_invalid():
    assert D.bucket_of_distance(np.inf) is None
    with pytest.raises(RuntimeError):
        D.bucket_of_distance(0.5)
    with pytest.raises(RuntimeError):
        D.bucket_of_distance(2.5)


def test_bucket_codes_are_exclusive_and_total():
    d = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 40.0, np.inf])
    codes = D.code_array(d)
    assert list(codes[:8]) == [0, 1, 2, 3, 4, 4, 5, 5]
    assert int(codes[8]) == C.UNREACHABLE


@pytest.mark.parametrize("name", sorted(TINY_GRAPHS))
def test_molecule_pairs_against_brute_force(name):
    n, edges = TINY_GRAPHS[name]
    e = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    i, j, b = D.molecule_pairs(n, e)
    ref = D.brute_force_distances(n, e)
    expected = {
        (u, v, D.bucket_of_distance(ref[u, v]))
        for u in range(n)
        for v in range(u + 1, n)
        if D.bucket_of_distance(ref[u, v]) is not None
    }
    assert set(zip(i.tolist(), j.tolist(), b.tolist())) == expected
    assert np.all(i < j)
    assert i.size == len(expected)


def test_two_components_drops_cross_component_pairs():
    # components {0,1,2}, {3,4}, {5}
    n, edges = TINY_GRAPHS["two_components"]
    i, j, b = D.molecule_pairs(n, np.asarray(edges, dtype=np.int64).reshape(-1, 2))
    component = {0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 2}
    stored = set(zip(i.tolist(), j.tolist()))
    assert all(component[u] == component[v] for u, v in stored)
    assert len(stored) == 3 + 1 + 0
    assert sorted(set(b.tolist())) == [0, 1]  # chain bonds (d1) and the 3-4 bond (d2)
    # every cross-component pair is absent
    for u in range(n):
        for v in range(u + 1, n):
            if component[u] != component[v]:
                assert (u, v) not in stored


def test_single_atom_has_no_pairs():
    i, j, b = D.molecule_pairs(1, np.zeros((0, 2), dtype=np.int64))
    assert i.size == 0 and j.size == 0 and b.size == 0


def test_subset_cache_round_trip(tmp_path):
    rows = [0, 1, 2, 300, 4000]
    meta = D.build_cache(tmp_path, rows=rows, verify_fraction=1.0, rc_crosscheck_rows=5)
    assert meta["rows_subset"] is True
    assert meta["labels_read"] is False
    cache = D.load_cache(tmp_path)
    assert cache["ptr"].size == int(meta["n_rows"]) + 1
    assert int(cache["ptr"][-1]) == int(meta["n_pairs_stored"])
    counts = np.asarray(cache["pair_counts"])
    assert counts[rows].sum() == int(meta["n_pairs_stored"])
    unselected = np.setdiff1d(np.arange(counts.size), np.asarray(rows))
    assert counts[unselected].sum() == 0
    with pytest.raises(RuntimeError):
        D.load_cache(tmp_path, require_full=True)


def test_cache_identity_records_no_labels(pair_cache):
    identity = D.cache_identity(pair_cache)
    assert identity["labels_read"] is False
    assert identity["n_pairs_stored"] > 0
    assert identity["hashes"]["pair_i"] and identity["hashes"]["pair_bucket"]


def test_batch_pair_slices_match_per_row_gather(pair_cache, pair_rows):
    sub = pair_rows[:8]
    i, j, b, graph, counts = D.batch_pair_slices(pair_cache, sub)
    ptr = pair_cache["ptr"]
    assert int(counts.sum()) == i.size
    assert np.array_equal(np.bincount(graph, minlength=len(sub)), counts)
    for g, row in enumerate(sub):
        lo, hi = int(ptr[row]), int(ptr[row + 1])
        sel = graph == g
        assert np.array_equal(i[sel], pair_cache["pair_i"][lo:hi].astype(np.int64))
        assert np.array_equal(j[sel], pair_cache["pair_j"][lo:hi].astype(np.int64))
        assert np.array_equal(b[sel], pair_cache["pair_bucket"][lo:hi].astype(np.int64))


def test_batch_pair_slices_handles_empty_batch(pair_cache):
    i, j, b, graph, counts = D.batch_pair_slices(pair_cache, [])
    assert i.size == j.size == b.size == graph.size == 0
    assert counts.size == 0


def test_collate_adds_only_pair_payload(task, pair_cache, pair_rows, tmp_path):
    batch = D.collate_pw(task, pair_cache, pair_rows[:4])
    base_keys = {
        "atom_feat",
        "atom_scalar",
        "atom_type",
        "bond_index",
        "bond_feat",
        "bond_type",
        "atom_batch",
        "bond_batch",
        "atom_off",
        "atom_counts",
        "global",
        "n_graphs",
        "bond_counts",
        "nbr_atom",
        "nbr_code",
        "path_atom",
        "path_end",
        "path_graph",
        "path_cat",
        "path_end63",
        "path_closure",
    }
    assert set(batch) == base_keys | {"pair_i", "pair_j", "pair_bucket", "pair_graph", "pair_counts"}
    assert int(batch["pair_counts"].sum()) == int(batch["pair_i"].numel())
    assert int(batch["pair_j"].max()) < int(batch["atom_counts"].max())


def test_synthetic_batch_pair_payload_is_consistent():
    batch = synthetic_batch(graphs=[TINY_GRAPHS["chain5"], TINY_GRAPHS["two_components"]])
    assert int(batch["pair_counts"].sum()) == int(batch["pair_i"].numel())
    assert torch.equal(
        torch.bincount(batch["pair_graph"], minlength=int(batch["n_graphs"])), batch["pair_counts"]
    )
    assert bool((batch["pair_j"] > batch["pair_i"]).all())

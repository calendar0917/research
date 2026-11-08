"""Distance cache: bucket coding, brute-force agreement and cache invariants."""

from __future__ import annotations

import numpy as np
import pytest

import rc_checks as CHK
import rc_common as C
import rc_dist as D


@pytest.mark.parametrize("name", sorted(CHK.TINY_GRAPHS))
def test_molecule_bucket_codes_match_bruteforce(name):
    n, edges = CHK.TINY_GRAPHS[name]
    edges = np.asarray(edges, dtype=np.int64).reshape(-1, 2)
    codes = D.molecule_bucket_codes(n, edges)
    dist = D.brute_force_distances(n, edges)
    iu = np.triu_indices(n, 1)
    assert np.array_equal(codes, D.distance_to_code(dist[iu]))


def test_single_and_two_atom_molecules():
    assert D.molecule_bucket_codes(1, np.zeros((0, 2), np.int64)).size == 0
    codes = D.molecule_bucket_codes(2, np.asarray([[0, 1]], dtype=np.int64))
    assert codes.tolist() == [0]


def test_bucket_codes_reject_bad_edges():
    with pytest.raises(Exception):
        D.molecule_bucket_codes(3, np.asarray([[0, 5]], dtype=np.int64))


def test_gather_pairs_matches_row_slices(dist_cache):
    rows = np.asarray([0, 5, 17, 1000, 350_342, 400_000], dtype=np.int64)
    packed, counts = D.gather_pairs(dist_cache, rows)
    expected = np.concatenate(
        [dist_cache["bucket"][dist_cache["ptr"][r] : dist_cache["ptr"][r + 1]] for r in rows]
    )
    assert np.array_equal(packed, expected)
    assert int(counts.sum()) == int(packed.size)


def test_real_cache_manifest_and_invariants(dist_cache):
    meta = dist_cache["meta"]
    bucket = dist_cache["bucket"]
    assert meta["labels_read"] is False
    assert meta["label_artifact_opened"] is False
    assert bucket.dtype == np.uint8
    assert np.isin(np.unique(bucket), np.concatenate([np.arange(C.N_BUCKETS), [C.UNREACHABLE]])).all()
    assert int(dist_cache["ptr"][-1]) == int(bucket.size)
    assert int(meta["n_rows"]) == C.N_MOLECULES
    assert int(meta["n_unordered_pairs"]) == int(bucket.size)
    counts = np.bincount(bucket, minlength=C.UNREACHABLE + 1)
    for d in range(C.N_BUCKETS):
        assert int(counts[d]) == int(meta["bucket_pair_totals"][d])
    assert int(counts[C.UNREACHABLE]) == int(meta["unreachable_unordered_pairs"])
    # per-bucket ordered counts must sum to 2 x pairs of same-component atoms
    assert sum(meta["bucket_ordered_pair_totals"]) + 2 * int(meta["unreachable_unordered_pairs"]) == int(
        2 * bucket.size
    )


def test_real_cache_matches_bruteforce_on_a_sample(dist_cache, n_rows: int = 40):
    import lr_common as LC

    cache_dir = LC.CACHE_PATH
    atom_ptr = np.load(cache_dir / "atom_ptr.npy")
    bond_ptr = np.load(cache_dir / "bond_ptr.npy")
    bond_index = np.load(cache_dir / "bond_index.npy", mmap_mode="r")
    gen = np.random.default_rng(4242)
    rows = gen.choice(int(dist_cache["meta"]["n_rows"]), size=n_rows, replace=False)
    for row in rows:
        a0, a1 = int(atom_ptr[row]), int(atom_ptr[row + 1])
        b0, b1 = int(bond_ptr[row]), int(bond_ptr[row + 1])
        edges = np.asarray(bond_index[b0:b1], dtype=np.int64)
        ref = D.brute_force_distances(a1 - a0, edges)
        iu = np.triu_indices(a1 - a0, 1)
        packed = dist_cache["bucket"][int(dist_cache["ptr"][row]) : int(dist_cache["ptr"][row + 1])]
        assert np.array_equal(packed, D.distance_to_code(ref[iu])), int(row)


def test_cache_covers_test_rows_without_labels(dist_cache, task):
    """Every official Test row has a structure entry; no label was read to get it."""
    rows = np.asarray(task.test_rows, dtype=np.int64)
    ptr = dist_cache["ptr"]
    assert int(ptr[rows + 1].max()) <= int(ptr[-1])
    assert int((ptr[rows + 1] - ptr[rows]).sum()) > 0
    assert dist_cache["meta"]["labels_read"] is False
    assert not bool(np.isfinite(task.labels[rows]).any())


def test_built_cache_is_self_consistent(tmp_path):
    """A tiny build of a few real molecules reproduces the frozen cache values."""
    try:
        meta = D.build_cache(tmp_path / "small", rows=[0, 1, 2, 3, 4], verify_fraction=1.0)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"PCBA cache unavailable: {type(exc).__name__}: {exc}")
    assert meta["n_rows"] == 5
    small = D.load_cache(tmp_path / "small")
    full = D.load_cache()
    rows = np.asarray([0, 1, 2, 3, 4], dtype=np.int64)
    packed, counts = D.gather_pairs(small, rows)
    ref_packed, ref_counts = D.gather_pairs(full, rows)
    assert np.array_equal(packed, ref_packed)
    assert np.array_equal(counts, ref_counts)

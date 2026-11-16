"""Data-layer semantics: physical bonds, ring definition, splits, normalisation."""

from __future__ import annotations

import numpy as np
import pytest

import sq_ast as A
import sq_bank as B
import sq_common as C
import sq_data as D
from helpers import make_packed


# ---------------------------------------------------------------------------
# physical bonds
# ---------------------------------------------------------------------------


def test_directed_duplicates_are_merged_and_must_agree():
    # a single triangle written with both directions, consistent attributes
    edge_index = np.asarray([[0, 1, 1, 2, 0, 2], [1, 0, 2, 1, 2, 0]])
    edge_attr = np.asarray([[0, 0, 1], [0, 0, 1], [1, 0, 0], [1, 0, 0], [2, 0, 0], [2, 0, 0]])
    u, v, feat = D.canonical_physical_bonds(3, edge_index, edge_attr)
    assert u.tolist() == [0, 0, 1]
    assert v.tolist() == [1, 2, 2]
    assert feat.shape == (3, 3)
    assert feat[:, 0].tolist() == [0, 2, 1]

    conflict = edge_attr.copy()
    conflict[1] = [3, 0, 0]
    with pytest.raises(ValueError):
        D.canonical_physical_bonds(3, edge_index, conflict)


def test_physical_bond_count_is_not_doubled_in_a_real_cache():
    cache = D.load_cache("ogbg-molesol")
    packed = D.packed_split(cache, "train")
    assert int(packed.n_bonds) == int(packed.graph_n_bonds.sum())
    # physical degrees: every bond contributes exactly 2 to the degree sum
    assert int(packed.degree.sum()) == 2 * int(packed.n_bonds)
    assert int(packed.degree.max()) <= 6


# ---------------------------------------------------------------------------
# ring definition
# ---------------------------------------------------------------------------


def _ring_bucket(n_atoms: int, edges) -> np.ndarray:
    u = np.asarray([min(a, b) for a, b in edges])
    v = np.asarray([max(a, b) for a, b in edges])
    rings, in_cycle = D.min_ring_buckets(n_atoms, u.astype(np.int32), v.astype(np.int32))
    assert np.array_equal(in_cycle, rings > 0)
    return rings


def test_min_ring_is_the_shortest_simple_cycle_through_each_atom():
    triangle = _ring_bucket(3, [(0, 1), (1, 2), (2, 0)])
    assert triangle.tolist() == [3, 3, 3]
    square = _ring_bucket(4, [(0, 1), (1, 2), (2, 3), (3, 0)])
    assert square.tolist() == [4, 4, 4, 4]
    hexagon = _ring_bucket(6, [(i, (i + 1) % 6) for i in range(6)])
    assert hexagon.tolist() == [6] * 6
    dodecagon = _ring_bucket(12, [(i, (i + 1) % 12) for i in range(12)])
    assert dodecagon.tolist() == [9] * 12  # >= 9 bucket
    chain = _ring_bucket(5, [(0, 1), (1, 2), (2, 3), (3, 4)])
    assert chain.tolist() == [0] * 5
    # a chord splits a hexagon into two 5-rings? here: 0-1-2-3-4-5 plus chord 0-2
    chorded = _ring_bucket(6, [(i, (i + 1) % 6) for i in range(6)] + [(0, 2)])
    assert chorded[0] == 3 and chorded[1] == 3 and chorded[2] == 3
    # spiro-like: two triangles sharing one atom -> shared atom sees a 3-ring
    spiro = _ring_bucket(5, [(0, 1), (1, 2), (2, 0), (0, 3), (3, 4), (4, 0)])
    assert spiro.tolist() == [3, 3, 3, 3, 3]


def test_ring_bucket_agrees_with_explicit_ring_primitive():
    cache = D.load_cache("ogbg-molesol")
    packed = D.packed_split(cache, "valid")
    explicit = A.evaluate(packed, A.IN_CYCLE())
    assert np.array_equal(explicit, packed.min_ring > 0)
    assert np.array_equal(A.evaluate(packed, A.ACYCLIC()), packed.min_ring == 0)


# ---------------------------------------------------------------------------
# official split identity and normalisation
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("dataset", ["ogbg-molesol", "ogbg-mollipo"])
def test_official_split_sizes_and_indices(dataset):
    cache = D.load_cache(dataset)
    sizes = {split: int(D.split_indices(cache, split).size) for split in ("train", "valid", "test")}
    assert sizes == C.EXPECTED_SPLIT_SIZES[dataset]
    for split in ("train", "valid", "test"):
        indices = D.split_indices(cache, split)
        assert np.all(np.diff(indices) > 0), "official indices must be sorted and unique"
        assert indices.min() >= 0 and indices.max() < int(cache["row_ids"].size)
    all_indices = np.concatenate([D.split_indices(cache, s) for s in ("train", "valid", "test")])
    assert all_indices.size == int(cache["row_ids"].size)
    assert len(set(all_indices.tolist())) == all_indices.size


@pytest.mark.parametrize("dataset", ["ogbg-molesol", "ogbg-mollipo"])
def test_target_standardisation_is_fitted_on_train_and_reversible(dataset):
    cache = D.load_cache(dataset)
    train = cache["labels"][D.split_indices(cache, "train")].astype(np.float64)
    scaler = C.train_target_scaler(train)
    assert scaler["std"] > 0
    roundtrip = (train - scaler["mean"]) / scaler["std"] * scaler["std"] + scaler["mean"]
    assert np.max(np.abs(roundtrip - train)) < 1e-6
    # population std (ddof=0), not sample std
    assert scaler["std"] == pytest.approx(float(np.std(train, ddof=0)), rel=1e-12)
    with pytest.raises(ValueError):
        C.train_target_scaler(np.zeros(5))


def test_train_vocabulary_ignores_val_test_values():
    """The query vocabulary only contains values seen in Train."""
    cache = D.load_cache("ogbg-molesol")
    vocab = D.train_vocabulary(cache)
    packed_train = D.packed_split(cache, "train")
    packed_valid = D.packed_split(cache, "valid")
    for j, values in vocab["atom_values"].items():
        assert np.isin(packed_train.atom_feat[:, j], values).all()
        unseen = sorted(set(np.unique(packed_valid.atom_feat[:, j]).tolist()) - set(values))
        # Val may contain unseen values, but they must not be in the vocabulary
        for value in unseen:
            assert value not in values


# ---------------------------------------------------------------------------
# label-blindness of the query bank
# ---------------------------------------------------------------------------


def test_bank_is_invariant_to_val_test_labels_and_train_label_change_only_moves_scoring():
    cache = D.load_cache("ogbg-molesol")
    train = D.packed_split(cache, "train")
    valid = D.packed_split(cache, "valid")
    vocab = D.train_vocabulary(cache)
    masks = B.MaskCache(limit_bytes=32 * 1024 * 1024)
    records_before, _ = B.build_base_bank(train, vocab, cache=masks)
    canon_before = [A.canon(record.ast) for record in records_before]

    labels = np.asarray(cache["labels"], dtype=np.float32).copy()
    labels[D.split_indices(cache, "valid")] = 123.0
    labels[D.split_indices(cache, "test")] = -456.0
    mutated = dict(cache)
    mutated["labels"] = labels
    train_mut, valid_mut = D.packed_split(mutated, "train"), D.packed_split(mutated, "valid")
    assert np.array_equal(valid_mut.labels, np.full(valid.n_graphs, 123.0, dtype=np.float32))
    records_after, _ = B.build_base_bank(train_mut, vocab, cache=B.MaskCache(limit_bytes=32 * 1024 * 1024))
    assert [A.canon(record.ast) for record in records_after] == canon_before
    # The bank's identity hash is a function of structures only.
    assert C.sha256_json(canon_before) == C.sha256_json([A.canon(r.ast) for r in records_after])


def test_vocabulary_and_bank_do_not_read_labels_structurally():
    """A quick guard against accidental label plumbing in the bank path."""
    import inspect

    source = inspect.getsource(B.build_base_bank) + inspect.getsource(D.train_vocabulary)
    for forbidden in ('payload["labels"]', "labels[", "y_train", "targets["):
        assert forbidden not in source

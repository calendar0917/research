"""Published splits, nested sample order and label alignment (no feature cache needed)."""

from __future__ import annotations

import numpy as np
import pytest

import tsb_common as C
import tsb_data as D


def test_three_published_splits_match_recorded_history():
    splits = D.load_splits()
    assert sorted(splits) == [122, 123, 124]
    for suffix, meta in splits.items():
        assert meta["file_sha256"] == C.SPLIT_SHA256[suffix]
        for role in ("train", "valid", "test"):
            rows = meta["rows"][role]
            assert rows.size == C.SPLIT_SIZES[role]
            assert meta["rows_sha256"][role] == C.SPLIT_ROWS_SHA256[suffix][role]
            assert np.unique(rows).size == rows.size
        train, valid, test = (meta["rows"][r] for r in ("train", "valid", "test"))
        assert np.intersect1d(train, valid).size == 0
        assert np.intersect1d(train, test).size == 0
        assert np.intersect1d(valid, test).size == 0
        assert np.unique(np.concatenate([train, valid, test])).size == 4200


def test_nested_sample_prefixes_are_deterministic_and_label_free():
    splits = D.load_splits()
    for suffix, meta in splits.items():
        train = meta["rows"]["train"]
        order_a = D.sample_order(train)
        order_b = D.sample_order(train[::-1].copy())
        assert np.array_equal(order_a, order_b), "order must not depend on row-array order"
        assert np.array_equal(np.sort(order_a), np.sort(train))
        for n in (1000, 2000, 3360):
            prefix = D.nested_prefix(train, n)
            assert prefix.size == n
            assert np.array_equal(prefix, order_a[:n])
        assert set(D.nested_prefix(train, 1000)).issubset(set(D.nested_prefix(train, 2000)))
        assert set(D.nested_prefix(train, 2000)).issubset(set(D.nested_prefix(train, 3360)))


def test_sample_order_uses_seed_and_row_id_only():
    train = D.split(122)["rows"]["train"]
    first = D.sample_order(train)
    # the key is sha256("<seed>:<row id>"), so the same row id in a different
    # split must occupy the same position within that split's own order
    other = D.split(124)["rows"]["train"]
    shared = np.intersect1d(train, other)
    pos_a = {int(r): i for i, r in enumerate(first)}
    pos_b = {int(r): i for i, r in enumerate(D.sample_order(other))}
    if shared.size:
        # relative order of shared rows is identical because the key is per-row
        a_shared = [int(r) for r in first if int(r) in set(shared.tolist())]
        b_shared = [int(r) for r in D.sample_order(other) if int(r) in set(shared.tolist())]
        assert a_shared == b_shared
    assert len(pos_a) == train.size and len(pos_b) == other.size


def test_labels_align_with_predictions_of_a_recorded_fit_contract():
    for suffix in C.SPLIT_SUFFIXES:
        labels = D.split_labels(suffix)
        assert labels["train"].shape == (3360,)
        assert labels["valid"].shape == (420,)
        assert np.all(np.isfinite(labels["train"]))
        assert np.all(np.isfinite(labels["valid"]))
        assert "test" not in labels, "Test labels must not be loaded by default"
        with_test = D.split_labels(suffix, include_test=True)
        assert with_test["test"].shape == (420,)


def test_data_manifest_is_hash_pinned():
    manifest = D.data_manifest()
    assert manifest["n_rows_total"] == 4200
    assert manifest["source"]["csv_sha256"] == C.CHEMBENCH_CSV_SHA256
    assert manifest["test_exposure"] == "EXPOSED_PUBLIC_TEST"
    for suffix in C.SPLIT_SUFFIXES:
        entry = manifest["splits"][str(suffix)]
        assert entry["file_sha256"] == C.SPLIT_SHA256[suffix]
        assert entry["sizes"] == C.SPLIT_SIZES


def test_smiles_are_the_pinned_chembench_rows():
    smiles = D.load_smiles()
    assert len(smiles) == 4200
    assert len(set(smiles)) > 3900
    assert smiles[0].startswith("Cn1c(CN2CCN(CC2)c3ccc(Cl)cc3)nc4ccccc14")

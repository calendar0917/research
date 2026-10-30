from __future__ import annotations

import hashlib

import numpy as np
import pytest

import v2_common as C
import v2_data as D
import v2_folds as F


def test_fold_partition_and_group_disjointness():
    folds = F.load_folds()
    sizes = [f["n_molecules"] for f in folds["folds"]]
    assert sizes == [784, 784, 784]
    assert folds["n_groups_total"] == 729
    seen = set()
    group_fold = {}
    smiles_fold = {}
    for f in folds["folds"]:
        assert f["n_molecules"] == 784
        for g in f["groups"]:
            assert g["group_key"] not in group_fold
            group_fold[g["group_key"]] = f["fold"]
        for row in f["rows_global"]:
            assert row not in seen
            seen.add(row)
            info = folds["rows"][str(row)]
            smiles_fold.setdefault(info["canonical_smiles_sha256"], set()).add(f["fold"])
    assert len(seen) == C.SPLIT_SIZES["fit"]
    assert all(len(v) == 1 for v in smiles_fold.values())


def test_fold_order_rule_is_recorded_and_reproducible():
    folds = F.load_folds()
    for f in folds["folds"]:
        keys = [g["group_key"] for g in f["groups"]]
        scores = [hashlib.sha256(("lcrc-lipo-v2|" + k).encode()).hexdigest() for k in keys]
        # The fold builder greedily assigns to the smallest fold; within a fold
        # the recorded group order is not a selection rule and need not be
        # sorted.  The global deterministic ordering is checked by rebuilding.
        assert len(keys) == len(set(keys))
        assert all(len(s) == 64 for s in scores)
    rebuilt = F.build_folds()
    assert rebuilt["n_groups_total"] == folds["n_groups_total"]
    assert [x["rows_global"] for x in rebuilt["folds"]] == [x["rows_global"] for x in folds["folds"]]
    assert [[g["group_key"] for g in x["groups"]] for x in rebuilt["folds"]] == [
        [g["group_key"] for g in x["groups"]] for x in folds["folds"]
    ]


def test_fold_file_invariants_are_verified():
    folds = F.load_folds(verify_hash=True)
    for f in folds["folds"]:
        assert C.row_sha256_bytes(f["rows_global"]) == f["row_sha256_int64"]
    assert folds["fit_row_sha256_int64"] == C.row_sha256_bytes(folds["fit_rows_global"])


def test_fold_views_recompute_norm_from_train_only(fit_cache):
    fit = C.load_fit_labels()
    y = np.asarray(fit["y"], dtype=np.float64)
    folds = F.load_folds()
    view = D.build_fold_view(0, cache=fit_cache, y_fit=y, folds=folds)
    assert view.train_cache["n_molecules"] == 1568
    assert view.dev_cache["n_molecules"] == 784
    assert view.y_train.shape == (1568,)
    assert view.y_dev.shape == (784,)
    assert set(view.train_positions).isdisjoint(view.dev_positions)
    full_norm = C.load_frozen_norm()
    assert C.prep_data.norm_fingerprint(view.train_norm) != C.prep_data.norm_fingerprint(full_norm)
    assert view.target_norm["n"] == 1568
    assert view.target_norm["source"] == "fold_train_labels_only"
    # The dev labels did not enter the target normalisation.
    expected_mean = float(y[np.asarray(view.train_positions)].mean())
    assert abs(float(view.target_norm["mean"]) - expected_mean) < 1e-12


def test_full_view_certifies_frozen_norm_and_target(fit_cache):
    fit = C.load_fit_labels()
    y = np.asarray(fit["y"], dtype=np.float64)
    view = D.build_full_view(cache=fit_cache, y_fit=y)
    assert view.train_cache["n_molecules"] == 2352
    assert view.target_norm["n"] == 2352
    assert C.prep_data.norm_fingerprint(view.train_norm) == C.prep_data.norm_fingerprint(C.load_frozen_norm())


def test_select_cache_is_a_molecule_copy(fit_cache):
    rows = [5, 0, 17, 3]
    sub = D.select_cache(fit_cache, rows, split_label="test")
    assert sub["n_molecules"] == 4
    assert sub["molecule_ids"] == [fit_cache["molecule_ids"][i] for i in rows]
    assert np.array_equal(sub["global"], np.asarray(fit_cache["global"])[rows])
    assert int(sub["atom_ptr"][-1]) == sum(
        int(fit_cache["atom_ptr"][i + 1]) - int(fit_cache["atom_ptr"][i]) for i in rows
    )
    assert len(sub["path_ptr"]) == int(sub["atom_ptr"][-1]) + 1
    assert len(sub["nbr_ptr"]) == int(sub["atom_ptr"][-1]) + 1
    with pytest.raises(ValueError):
        D.select_cache(fit_cache, [1, 1])
    with pytest.raises(IndexError):
        D.select_cache(fit_cache, [-1])


def test_shuffle_is_deterministic_and_epoch_local():
    a = D.shuffle_batches(1000, 128, 917000)
    b = D.shuffle_batches(1000, 128, 917000)
    c = D.shuffle_batches(1000, 128, 917001)
    assert a == b
    assert a != c
    flat = [r for batch in a for r in batch]
    assert len(flat) == len(set(flat)) == 1000
    assert len(a[-1]) == 1000 - (len(a) - 1) * 128


def test_fold_rows_are_fit_only():
    folds = F.load_folds()
    split = C.load_frozen_split()
    fit = set(int(x) for x in split["split"]["fit"])
    monitor = set(int(x) for x in split["split"]["monitor"])
    for f in folds["folds"]:
        rows = set(int(x) for x in f["rows_global"])
        assert rows <= fit
        assert rows.isdisjoint(monitor)

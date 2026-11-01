"""Structure cache, normalisation and label-isolation tests."""

from __future__ import annotations

import numpy as np
import pytest

import mb_common as C
import mb_data as D

import lcrc_common as PREP_COMMON
import lcrc_data as PREP_DATA


@pytest.fixture(scope="module")
def spec():
    return D.frozen_spec()


@pytest.fixture(scope="module")
def small_cache(spec):
    src = __import__("mb_source").load_dataset("FreeSolv", labels="none")
    rows = src["rows"]["valid"][:6]
    return D.build_split_cache("FreeSolv", "valid", rows, src["smiles"], spec, save=False)


def test_cache_is_label_free(small_cache):
    assert small_cache["meta"]["labels_present"] is False
    assert not any("label" in str(k).lower() for k in small_cache)
    assert not any("target" in str(k).lower() for k in small_cache)


def test_cache_row_ids_and_smiles(small_cache):
    src = __import__("mb_source").load_dataset("FreeSolv", labels="none")
    rows = [int(r) for r in src["rows"]["valid"][:6]]
    assert [int(x) for x in small_cache["rows"].tolist()] == rows
    assert small_cache["molecule_ids"] == [str(r) for r in rows]
    assert small_cache["smiles"] == [src["smiles"][r] for r in rows]


def test_cache_rebuild_is_byte_identical(spec):
    src = __import__("mb_source").load_dataset("ESOL", labels="none")
    rows = src["rows"]["test"][:4]
    a = D.build_split_cache("ESOL", "test", rows, src["smiles"], spec, save=False)
    b = D.build_split_cache("ESOL", "test", rows, src["smiles"], spec, save=False)
    assert PREP_COMMON.content_fingerprint(a) == PREP_COMMON.content_fingerprint(b)
    assert a["meta"]["content_fingerprint"] == b["meta"]["content_fingerprint"]


def test_cache_validation_passes(small_cache, spec):
    report = PREP_DATA.validate_cache(small_cache, spec)
    assert report["n_molecules_checked"] == 6
    assert report["finite"] and report["index_ranges_valid"]
    assert report["n_s1_codes"] == 2 * report["n_bonds"]


def test_collate_batch_offsets_are_local(small_cache):
    batch = PREP_DATA.collate(small_cache, [0, 1, 2])
    n_atoms = int(batch["atom_feat"].shape[0])
    assert int(batch["atom_batch"].max()) == 2
    if batch["bond_index"].numel():
        assert int(batch["bond_index"].max()) < n_atoms
    if batch["path_end"].numel():
        assert int(batch["path_end"].max()) < n_atoms
    assert batch["n_graphs"] == 3


def test_collate_rejects_duplicate_rows(small_cache):
    with pytest.raises(ValueError, match="duplicate row"):
        PREP_DATA.collate(small_cache, [0, 0])


def test_train_only_normalisation_differs_from_valid(small_view, spec):
    train_norm = small_view.train_norm
    valid_norm = PREP_DATA.fit_norm(small_view.valid_cache, spec, source="test_valid_only")
    assert train_norm["valid_or_test_used"] is False
    assert not np.array_equal(np.asarray(train_norm["global_mean"]), np.asarray(valid_norm["global_mean"]))
    recomputed = PREP_DATA.fit_norm(small_view.train_cache, spec, source="test_recompute")
    for key in ("scalar_mean", "scalar_scale", "global_mean", "global_scale"):
        assert np.array_equal(np.asarray(train_norm[key]), np.asarray(recomputed[key])), key


def test_target_normaliser_is_train_only():
    y = np.asarray([1.0, 2.0, 3.0, 4.0])
    norm = D.target_normaliser(y)
    assert norm["mean"] == pytest.approx(2.5)
    assert norm["scale"] == pytest.approx(np.std(y))
    assert norm["source"] == "train_split_only"
    assert norm["valid_or_test_used"] is False


def test_classification_view_has_no_target_normaliser():
    import mb_source as S

    src = S.load_dataset("BACE", labels="train_valid")
    assert src["y"] is not None
    # a classification view must not expose a z-score normaliser
    view = D.build_view("BACE")
    assert view.task == "classification"
    assert view.target_norm is None


def test_build_view_does_not_construct_test_cache(monkeypatch):
    import mb_source as S

    calls = []
    original = D.build_test_structure_cache

    def spy(*args, **kwargs):  # pragma: no cover - only triggers on a violation
        calls.append(args)
        return original(*args, **kwargs)

    monkeypatch.setattr(D, "build_test_structure_cache", spy)
    D.build_view("FreeSolv", spec=D.frozen_spec())
    assert calls == []

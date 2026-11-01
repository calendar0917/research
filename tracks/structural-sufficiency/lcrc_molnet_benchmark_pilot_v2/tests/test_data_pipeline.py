"""Adapter fidelity, cache label-freedom and alignment tests."""

from __future__ import annotations

import gzip
import json

import numpy as np
import pytest

import mb2_common as C
import mb2_data as D
import mb2_raw_artifact as RAW
import mb2_rdkit as R
import mb2_source as S


def test_raw_artifact_verifies():
    report = RAW.check()
    for key, item in report.items():
        assert item["sha256_matches"], key
        assert item["rows_digest_matches"], key
        assert item["n_rows"] == C.DATASETS[key]["expected_rows"], key


def test_adapter_vocabulary_matches_v1_constants():
    import mb_rdkit as V1R  # Sprint-01 module, read-only

    assert R.ATOM_CLASSES == V1R.ATOM_CLASSES
    assert R.BOND_CLASSES == V1R.BOND_CLASSES
    assert R.CHARGE_BUCKETS == V1R.CHARGE_BUCKETS
    assert R.H_BUCKETS == V1R.H_BUCKETS
    assert R.ATOM_UNK == V1R.ATOM_UNK and R.BOND_UNK == V1R.BOND_UNK


def test_adapter_reproduces_the_sprint01_artifact_on_a_sample():
    path = C.V1 / "analysis/source/raw_attributes_v1.json.gz"
    if not path.exists():  # pragma: no cover
        pytest.skip("Sprint-01 raw artifact not available")
    with gzip.open(path, "rt", encoding="utf-8") as fh:
        old = json.load(fh)
    fields = ("atom_class", "atom_charge", "atom_h", "edge_class", "edge_stereo")
    for key in ("ESOL", "FreeSolv"):
        old_rows = old["rows"][key]
        idx = list(range(0, len(old_rows), max(1, len(old_rows) // 50)))
        new_rows = RAW.load_rows(key, idx)
        for a, b in zip([old_rows[i] for i in idx], new_rows):
            for f in fields:
                assert np.array_equal(np.asarray(a[f], dtype=np.int64), np.asarray(b[f], dtype=np.int64)), (key, f)
            assert np.array_equal(
                np.asarray(a["edge_index"], dtype=np.int64), np.asarray(b["edge_index"], dtype=np.int64)
            )
            assert str(a["smiles"]) == str(b["smiles"])
            assert int(a["n_components"]) == int(b["n_components"])


def test_tox21_aluminium_rows_use_the_documented_fallback():
    manifest = C.read_json(C.RAW_ARTIFACT_MANIFEST)
    tox = manifest["datasets"]["Tox21"]
    assert tox["parse_methods"]["strict"] == 7823
    assert tox["parse_methods"]["unsanitized_fallback"] == 8
    rows = {r["row"] for r in tox["unsanitized_fallback_rows"]}
    assert rows == {1322, 2290, 2297, 3558, 4565, 4649, 5538, 6723}
    assert all("Al" in r["smiles"] for r in tox["unsanitized_fallback_rows"])
    # the fallback still produced a real graph for each row
    for r in tox["unsanitized_fallback_rows"]:
        assert r["n_atoms"] > 0


def test_unsanitized_fallback_is_not_silent_for_a_bad_smiles():
    with pytest.raises(ValueError):
        R.smiles_to_raw_row_relaxed("not-a-smiles", 0)


@pytest.mark.parametrize("key", ["FreeSolv", "Tox21", "BBBP"])
def test_caches_are_label_free_and_row_aligned(key):
    spec = D.frozen_spec()
    view = D.build_view(key, spec=spec)
    src = S.load_dataset(key, labels="train_valid")
    assert view.train_cache["meta"]["labels_present"] is False
    assert not any("label" in str(k).lower() for k in view.train_cache.keys())
    for split in ("train", "valid"):
        cache = view.train_cache if split == "train" else view.valid_cache
        expected = [int(r) for r in src["rows"][split]]
        assert [int(x) for x in np.asarray(cache["rows"]).tolist()] == expected
        assert cache["molecule_ids"] == [str(r) for r in expected]
        assert all(str(cache["smiles"][i]) == str(src["smiles"][r]) for i, r in enumerate(expected))


def test_train_only_normalisation(freesolv_view):
    spec = freesolv_view.spec
    import lcrc_data as PREP_DATA

    recomputed = PREP_DATA.fit_norm(freesolv_view.train_cache, spec, source="test")
    assert np.allclose(
        np.asarray(freesolv_view.train_norm["global_mean"]), np.asarray(recomputed["global_mean"])
    )
    valid_only = PREP_DATA.fit_norm(freesolv_view.valid_cache, spec, source="test")
    assert not np.allclose(
        np.asarray(freesolv_view.train_norm["scalar_mean"]), np.asarray(valid_only["scalar_mean"])
    )
    assert freesolv_view.train_norm["valid_or_test_used"] is False
    assert freesolv_view.target_norm["source"] == "train_split_only"


def test_tox21_targets_keep_nan_in_the_view(tox21_view):
    assert tox21_view.task == "multitask"
    assert np.isnan(tox21_view.targets.y_train).any()
    assert not (np.isnan(tox21_view.targets.y_train) & (tox21_view.targets.y_train == 0)).any()
    reference = tox21_view.targets.loss_reference(None)
    assert np.isnan(reference).sum() == np.isnan(tox21_view.targets.y_train).sum()

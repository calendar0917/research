from __future__ import annotations

import numpy as np

import screen_common as C
import screen_data as D


def test_frozen_feature_spec(spec):
    assert spec.name == "lipo_v1"
    assert spec.hash() == C.EXPECTED_FEATURE_SPEC_HASH
    assert (spec.atom_feat_dim, spec.bond_feat_dim, spec.global_dim, spec.n_scalars) == (51, 13, 42, 9)
    assert C.sha256_file(C.PREP_ANALYSIS / "feature_spec_lipo.json") == C.EXPECTED_SHA256["feature_spec_lipo"]


def test_frozen_norm_is_fit_only(norm):
    assert norm["fit_split"] == "fit"
    assert norm["valid_or_test_used"] is False
    assert C.prep_data.norm_fingerprint(norm) == C.EXPECTED_NORM_FINGERPRINT
    assert C.sha256_file(C.PREP_ANALYSIS / "norm_lipo.json") == C.EXPECTED_SHA256["norm_lipo"]


def test_split_rows_and_scope(split):
    assert {k: len(v) for k, v in split["split"].items()} == C.SPLIT_SIZES
    assert split["hashes"] == C.EXPECTED_ROW_HASHES
    assert split["payload"]["scope"]["labels_read"] is False
    assert split["payload"]["scope"]["official_valid_test_labels_read"] is False
    assert not set(split["split"]["fit"]) & set(split["split"]["monitor"])
    assert C.sha256_file(C.LIPO_SPLIT) == C.EXPECTED_SHA256["split_phase1a"]


def test_cache_fingerprints_and_row_alignment(caches):
    for name, payload in caches.items():
        assert payload["content_fingerprint"] == C.EXPECTED_CACHE_FINGERPRINT[name]
        assert payload["cache"]["meta"]["labels_present"] is False
        assert payload["cache"]["meta"]["feature_spec_hash"] == C.EXPECTED_FEATURE_SPEC_HASH
        assert [str(x) for x in payload["cache"]["molecule_ids"]] == [str(int(r)) for r in payload["rows"]]
        assert int(payload["cache"]["n_molecules"]) == C.SPLIT_SIZES[name]


def test_cache_arrays_have_no_label_field(caches):
    for payload in caches.values():
        keys = {str(k).lower() for k in payload["cache"]}
        assert not ({"y", "labels", "target", "exp"} & keys)
        assert payload["cache"]["meta"]["labels_present"] is False


def test_no_reserved_cache_is_loaded(caches):
    assert set(caches) == {"fit", "monitor"}


def test_target_norm_artifact_matches_fit_labels():
    """norm_target.json must be the Fit-only standardisation (written at freeze)."""
    norm = D.load_target_norm(verify=True)
    assert norm["source"] == "fit_labels_only"
    assert norm["n"] == C.SPLIT_SIZES["fit"]
    assert norm["monitor_or_reserved_used"] is False
    assert np.isfinite(norm["mean"]) and np.isfinite(norm["scale"]) and norm["scale"] > 0

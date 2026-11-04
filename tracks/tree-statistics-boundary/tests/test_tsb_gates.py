"""The full P0 gate set, exercised through pytest on the recorded feature cache."""

from __future__ import annotations

import pytest

import tsb_common as C
import tsb_features as F
import tsb_gates as G

pytestmark = [
    pytest.mark.slow,
    pytest.mark.skipif(not F.MANIFEST_PATH.is_file(), reason="feature cache absent"),
]


def test_wl2_bin_and_count_share_column_definitions():
    payload = G.gate_wl2_columns()
    assert payload["passed"], payload["evidence"]


def test_wl3_low_rounds_match_wl2():
    payload = G.gate_wl3_low_rounds()
    assert payload["passed"], payload["evidence"]


def test_vocabulary_comes_from_train_only():
    payload = G.gate_vocab_train_only()
    assert payload["passed"], payload["evidence"]


def test_atom_renumbering_does_not_change_features():
    payload = G.gate_permutation_invariance(sample=60)
    assert payload["passed"], payload["evidence"]


def test_cache_reload_is_elementwise_identical():
    payload = G.gate_cache_reload()
    assert payload["passed"], payload["evidence"]


def test_no_nan_and_oov_policy_is_explicit():
    payload = G.gate_no_nan_and_policy()
    assert payload["passed"], payload["evidence"]


def test_features_do_not_read_labels():
    payload = G.gate_labels_not_used(sample=40)
    assert payload["passed"], payload["evidence"]


def test_arm_dimensions_are_frozen():
    spec = C.read_json(C.CONFIGS / "feature_spec.json")
    for suffix in C.SPLIT_SUFFIXES:
        for arm in C.ARMS:
            _matrix, names = F.arm_matrix(suffix, arm)
            frozen = spec["arms"][arm]["per_split"][str(suffix)]
            assert len(names) == frozen["n_columns"]
            assert C.sha256_json(names) == frozen["column_list_sha256"]
    assert spec["arms"]["WL2_COUNT"]["n_columns"] == spec["arms"]["WL2_BIN"]["n_columns"]
    assert (
        spec["arms"]["WL3_COUNT"]["n_columns"]
        == spec["arms"]["WL2_COUNT"]["n_columns"] + (C.WL_VOCAB_SIZE + 1)
    )
    assert spec["arms"]["MORGAN_COUNT"]["n_columns"] == (
        spec["arms"]["G0"]["n_columns"] + C.MORGAN_NBITS
    )


def test_model_roundtrip_is_bit_identical():
    payload = G.gate_model_roundtrip()
    assert payload["passed"], payload["evidence"]


def test_full_gate_suite_passes():
    gates = G.run_all(include_fit_gates=bool(list((C.ANALYSIS / "fits").glob("*.json"))))
    failed = [gate["name"] for gate in gates if not gate["passed"]]
    assert not failed, failed

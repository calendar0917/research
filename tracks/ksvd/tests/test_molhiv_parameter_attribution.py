"""Tests for the MolHIV current-model parameter attribution audit.

Static / unit tests only: no data loading beyond the small local frozen run
record, no training, and no official-test access.
"""

from __future__ import annotations

import json
from collections import Counter

import torch

from tracks.ksvd.experiments.luyin16 import molhiv_parameter_attribution as mpa
from tracks.ksvd.experiments.luyin16 import molhiv_recurrent_pair_centre as rpc

# Exact vocabulary sizes of the frozen MolHIV run (run_seed0.json).
TYPED_WITH_OOV = 26233
PARENT_WITH_OOV = 103
EXPECTED_TOTAL = 1076589


def _frozen_run() -> dict:
    return json.loads(
        (rpc.RESULTS_DIR / "run_seed0.json").read_text(encoding="utf-8")
    )


def _model() -> torch.nn.Module:
    return rpc.build_model(TYPED_WITH_OOV, PARENT_WITH_OOV, 0)


def test_01_frozen_run_vocabulary_matches_the_audit_constants():
    run = _frozen_run()
    assert run["typed_vocabulary_size_with_oov"] == TYPED_WITH_OOV
    assert run["parent_vocabulary_size_with_oov"] == PARENT_WITH_OOV
    assert run["parameters"] == EXPECTED_TOTAL


def test_02_classification_sums_exactly_to_the_model_total():
    model = _model()
    out = mpa._classify(model)
    assert out["total_params"] == EXPECTED_TOTAL
    assert out["named_parameters_match_parameters"] is True
    assert out["category_sum_matches_total"] is True
    assert out["other_uncategorized_tensors"] == []


def test_03_identity_storage_dominates():
    out = mpa._classify(_model())
    # typed 26233 x 32 + parent 103 x 16
    assert out["identity_storage_params"] == 839456 + 1648
    assert out["fixed_size_params"] == EXPECTED_TOTAL - (839456 + 1648)
    assert out["identity_storage_fraction"] > 0.78
    assert out["fixed_size_fraction"] < 0.22


def test_04_top20_is_sorted_and_typed_embedding_is_first():
    out = mpa._classify(_model())
    top = out["top20_tensors"]
    assert len(top) == 20
    assert top[0]["name"] == "typed_embedding.weight"
    assert top[0]["shape"] == [TYPED_WITH_OOV, 32]
    assert top[0]["numel"] == 839456
    counts = [row["numel"] for row in top]
    assert counts == sorted(counts, reverse=True)


def test_05_feature_block_attribution_matches_the_encoder_totals():
    model = _model()
    attribution = mpa._feature_block_attribution(model)
    assert (
        attribution["patch_encoder_first_layer_block_sum"]
        == attribution["patch_encoder_total"]
    )
    assert (
        attribution["global_encoder_first_layer_block_sum"]
        == attribution["global_encoder_total"]
    )
    patch = attribution["patch_encoder_first_layer_blocks"]
    assert patch["typed_token_columns"] == 96 * int(mpa.TOKEN_WIDTH)
    assert patch["parent_token_columns"] == 96 * int(mpa.PARENT_WIDTH)
    assert patch["atom_node_shell_columns"] > patch["bond_edge_shell_columns"]


def test_06_absent_categories_are_explained():
    assert "atom_node_encoding_table" in mpa.ABSENT_CATEGORIES
    assert "bond_edge_encoding_table" in mpa.ABSENT_CATEGORIES
    assert "topology_encoder" in mpa.ABSENT_CATEGORIES


def test_07_topk_accounting_is_pure_and_monotone():
    total = EXPECTED_TOTAL
    identity = (TYPED_WITH_OOV) * 32 + PARENT_WITH_OOV * 16
    fixed = total - identity
    rows = mpa._topk_accounting(fixed, total, TYPED_WITH_OOV - 1, PARENT_WITH_OOV - 1)
    assert [row["top_k"] for row in rows] == [50, 100, 250, 500, 1000]
    totals = [row["theoretical_total_params"] for row in rows]
    assert totals == sorted(totals)  # more kept -> more params
    assert all(row["theoretical_total_params"] < total for row in rows)
    # K=1000: fixed + 1001*32 + 103*16
    assert rows[-1]["theoretical_total_params"] == fixed + 1001 * 32 + 103 * 16
    assert rows[-1]["saved_vs_current"] == total - rows[-1]["theoretical_total_params"]


def test_08_frequency_distribution_singletons_and_coverage():
    counts = Counter({"a": 10, "b": 5, "c": 1, "d": 1})
    out = mpa._frequency_distribution(counts)
    assert out["unique_types"] == 4
    assert out["occurrences"] == 17
    assert out["singleton_types"] == 2
    assert out["max_frequency"] == 10
    assert abs(out["coverage_top_50"] - 1.0) < 1e-12
    assert abs(out["singleton_occurrence_fraction"] - 2 / 17) < 1e-12


def test_09_summarize_empty_and_nonempty():
    empty = mpa._summarize([])
    assert empty["mean"] == 0.0 and empty["max"] == 0.0
    out = mpa._summarize([0.0, 0.5, 1.0])
    assert abs(out["mean"] - 0.5) < 1e-12
    assert out["max"] == 1.0

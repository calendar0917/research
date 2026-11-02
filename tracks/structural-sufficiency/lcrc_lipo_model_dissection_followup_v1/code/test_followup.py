"""Minimal regression tests for the follow-up analyses.

Run with:
    uv run python -m pytest code/test_followup.py -q
"""

from __future__ import annotations

import copy

import numpy as np
import pytest
import torch

import corrected_tables as T
import error_concentration as E
import followup_common as F
import frozen_jc_interventions as J


def test_main_run_identity_is_unique_and_complete():
    rows = F.load_main_rows()
    check = F.check_main_rows(rows)
    assert check["n_rows"] == 72
    assert check["n_unique_identities"] == 72
    assert check["per_repeat"] == {"122": 54, "123": 9, "124": 9}
    assert check["n_valid_is_420"]
    assert check["variants_by_repeat"]["123"] == ["A0", "B5", "C1"]
    with pytest.raises(ValueError):
        T.identity_index(rows + [rows[0]])


def test_missing_comparison_arm_is_not_borrowed_from_another_repeat():
    rows = [r for r in F.load_main_rows() if int(r["repeat"]) == 122]
    # Drop the repeat-122 A0 seed 1 cell.  A0 exists in 123/124, but must not
    # be borrowed to complete the contrast.
    rows = [r for r in rows if not (r["variant"] == "A0" and int(r["seed"]) == 1)]
    index = T.identity_index(rows)
    item = T.contrast_for_repeat(index, 122, "A2", "A0", "NO_G - FULL", "valid_rmse")
    assert item["status"] == "partial"
    assert item["matched_seeds"] == [0, 2]
    assert any("missing reference cell 122/A0/s1" in x["reasons"] for x in item["missing_cells"])


def test_corrected_anchors_and_legacy_offset():
    rows = F.load_main_rows()
    tables = T.build_tables(rows, "valid")
    anchors = T.anchor_checks(rows, tables)
    assert anchors["all_passed"]
    assert anchors["max_abs_residual"] <= 1e-6
    decomp = anchors["legacy_A2_A0_decomposition"]
    assert decomp["offset_matches_FULL122_minus_FULL124"]
    assert abs(decomp["offset_legacy_minus_correct"] - 0.0692998089) <= 1e-6
    assert abs(decomp["legacy_A2_minus_A0"] - 0.0779990011) <= 1e-6
    assert abs(decomp["correct_repeat122_A2_minus_A0"] - 0.0086991922) <= 1e-6
    # C1/B5 are available in 123/124 only as their own repeat pairs.
    by = {(x["repeat"], x["contrast"], x["metric"]): x for x in tables["entries"]}
    assert by[(123, "C1-A0", "valid_rmse")]["status"] == "complete"
    assert abs(by[(123, "C1-A0", "valid_rmse")]["mean_diff"] - (-0.00190287)) <= 1e-6
    assert by[(123, "A2-A0", "valid_rmse")]["status"] == "unavailable"
    cross_check = T.cross_repeat_check(rows)
    assert cross_check["all_fields_match"]
    boot_check = T.bootstrap_consistency(rows)
    assert boot_check["all_points_agree"]
    assert boot_check["max_abs_residual"] <= 1e-6


def test_test_table_is_scalar_only_and_per_repeat():
    rows = F.load_main_rows()
    test = T.build_tables(rows, "test")
    assert "EXPOSED_TEST_SCALAR_DIFF_ONLY" in test["status_note"]
    by = {(x["repeat"], x["contrast"], x["metric"]): x for x in test["entries"]}
    assert by[(122, "C1-A0", "test_rmse")]["status"] == "complete"
    assert by[(124, "B5-A0", "test_rmse")]["status"] == "complete"
    assert by[(122, "A2-A0", "test_rmse")]["status"] == "complete"
    # No Test prediction file is ever opened by corrected_tables.py; this test
    # checks the table contains only the required exposed-test disclaimer.


def test_error_predictions_align_by_row_id_and_identity():
    ref = F.valid_reference()
    for variant in F.ERROR_VARIANTS:
        for seed in F.SEEDS:
            payload = F.load_valid_predictions(122, variant, seed)
            aligned = F.align_predictions(payload, ref)
            assert aligned["ok"]
            assert np.array_equal(aligned["rows"], ref["rows"])
            assert aligned["y_max_abs_diff"] == 0.0


def test_error_decomposition_identity_and_row2087():
    result = E.analyse()
    items = result["decomposition"]
    assert len(items) == 5 * 3  # A0 reference plus four candidates, per seed
    for item in items:
        assert item["net_equals_loss_minus_gain"]
        assert abs(item["delta_sse"] - (item["gross_loss"] - item["gross_gain"])) <= 1e-9
        assert item["main_metrics_match_tolerance"]
    row = result["row2087"]
    assert row["row_id"] == 2087
    assert row["smiles"] == "CC(N)(COP(=O)(O)O)C(=O)O"
    assert row["y"] == 2.5
    assert row["seed0_full_abs_rank_zero_based"] == 0
    assert abs(row["share_of_full_sse_seed0"] - 90.66088133651519 / 206.24148735334805) <= 1e-12
    # A3/A5 row2087 improves, while the rest of the split worsens on all seeds.
    for variant in ("A3", "A5"):
        outs = [
            x["sets"]["row2087"]["outside"]["delta_rmse"]
            for x in items
            if x["variant"] == variant
        ]
        assert all(v is not None and v > 0 for v in outs), (variant, outs)


def test_jc_algebra_checks():
    checks = J.implementation_checks()
    assert checks["all_passed"]
    assert checks["empty_groups_zero"]
    assert checks["singleton_equals_full"]
    assert checks["constant_gate_content_full_equals_marginal"]
    assert checks["two_molecules_do_not_cross_group"]
    assert checks["joint_bias_counted_n_g_times"]


def test_jc_one_checkpoint_parity_and_independent_switches():
    ref = F.valid_reference()
    view = ref["view"]
    rows = np.asarray(ref["rows"], dtype=np.int64)
    y = np.asarray(ref["y"], dtype=np.float64)
    smiles = list(ref["smiles"])
    batch = F.D.to_device(F.D.PREP_DATA.collate(view.valid_cache, list(range(rows.size))), torch.device("cpu"))
    summary, mol_rows, arrays = J.process_checkpoint("A0", 0, view, batch, rows, y, smiles)
    assert summary["native_parity"]["passed"]
    assert summary["both_parity"]["passed"]
    assert summary["both_parity"]["max_abs_diff_standardised"] == 0.0
    assert summary["invariance"]["state_hash_unchanged"]
    assert summary["invariance"]["batch_hash_unchanged"]
    assert summary["invariance"]["J_ONLY_content_unchanged"]
    assert summary["invariance"]["C_ONLY_joint_unchanged"]
    assert summary["invariance"]["J_ONLY_tek_unchanged"]
    assert summary["invariance"]["C_ONLY_tek_unchanged"]
    assert summary["invariance"]["C_ONLY_content_differs"]
    assert len(summary["conditions"]) == 4
    assert len(mol_rows) == 4 * 420
    for cond in summary["conditions"].values():
        assert cond["decomposition_passed"]
        assert cond["max_abs_decomposition_residual"] <= 1e-5
    # Per-root arrays are recorded, not just vector norms.
    assert arrays["d_abs_J"].shape == (11465,)
    assert arrays["d_abs_C"].shape == (11465,)
    assert np.isfinite(arrays["d_rel_J"]).all() and np.isfinite(arrays["d_rel_C"]).all()


def test_jc_completed_summary_has_all_24_conditions():
    payload = F.read_json(F.TRAINED_JC / "summary.json")
    assert payload["failures"] == []
    assert payload["n_checkpoints_loaded"] == 6
    assert payload["n_condition_evaluations"] == 24
    for item in payload["summaries"]:
        assert item["native_parity"]["passed"]
        assert item["both_parity"]["passed"]
        assert set(item["conditions"]) == {"NATIVE", "J_ONLY", "C_ONLY", "BOTH"}
        for cond in item["conditions"].values():
            assert cond["decomposition_passed"]

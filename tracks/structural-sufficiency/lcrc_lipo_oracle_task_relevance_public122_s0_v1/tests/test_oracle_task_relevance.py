"""Key scientific checks for lcrc-lipo-oracle-task-relevance-public122-s0-v1.

Focused, not a large gate suite: label isolation, target standardisation units,
the independent oracle recomputation, the MolWt invariant, the raw-graph /
RDKit mapping, and the paired / recovery arithmetic.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND / "code"))

import round_common as RC  # noqa: E402
import oracle as O  # noqa: E402
import train_task as TT  # noqa: E402
import evaluate as EV  # noqa: E402


# ---------------------------------------------------------------------------
# label loader: task / row_id explicit, original rows only, no cross-read
# ---------------------------------------------------------------------------


def _fixture_rows() -> list[dict]:
    return [
        {"task": "wc_logp", "source_kind": "original", "kind": "O", "split": "test", "row_id": "5", "value": "1.5"},
        {"task": "wc_logp", "source_kind": "rewire", "kind": "T", "split": "test", "row_id": "5", "value": "9.9"},
        {"task": "molwt", "source_kind": "original", "kind": "O", "split": "test", "row_id": "5", "value": "250.0"},
        {"task": "wc_logp", "source_kind": "original", "kind": "O", "split": "valid", "row_id": "5", "value": "1.5"},
    ]


def test_label_loader_isolation():
    rows = _fixture_rows()
    out = TT.select_original_labels(rows, "wc_logp", "test", [5])
    assert out == {5: 1.5}
    # the molwt row of the same row_id must never be read for the wc_logp task
    assert out[5] == 1.5 and out[5] != 250.0
    # a rewire row for the same row_id must never be selected
    assert 9.9 not in out.values()


def test_label_loader_missing_coverage_raises():
    rows = _fixture_rows()
    with pytest.raises(ValueError):
        TT.select_original_labels(rows, "wc_logp", "test", [5, 6])


def test_label_loader_rejects_rewire_for_training():
    # if only a rewire row existed, coverage would fail (never silently used)
    rows = [{"task": "wc_logp", "source_kind": "rewire", "kind": "T", "split": "train", "row_id": "1", "value": "3"}]
    with pytest.raises(ValueError):
        TT.select_original_labels(rows, "wc_logp", "train", [1])


# ---------------------------------------------------------------------------
# target standardisation units
# ---------------------------------------------------------------------------


def test_target_standardisation_roundtrip_and_units():
    y = np.asarray([0.5, 1.0, 1.5, 2.0], dtype=np.float64)
    mean = float(y.mean())
    scale = float(y.std(ddof=0))
    z = (y - mean) / scale
    m = TT.raw_metrics(z, y, mean, scale)
    assert abs(m["rmse"]) < 1e-12
    assert abs(m["mae"]) < 1e-12
    assert abs(m["r2"] - 1.0) < 1e-12
    # a raw shift of one scale must appear as exactly one unit of raw error
    shifted = z + 1.0
    m2 = TT.raw_metrics(shifted, y, mean, scale)
    assert abs(m2["mae"] - scale) < 1e-12


# ---------------------------------------------------------------------------
# oracle: independent recomputation and the MolWt invariant
# ---------------------------------------------------------------------------


def test_oracle_originals_finite_and_match_both_columns():
    man = RC.read_json(RC.NEW_ANALYSIS / "oracle_manifest.json")
    assert man["checks"]["originals_finite"]
    assert man["checks"]["train_std_positive"]
    assert man["checks"]["wc_logp_rewire_fraction_changed_gt_1e-8"] > 0.0


def test_molwt_invariant_on_frozen_rewire():
    man = RC.read_json(RC.NEW_ANALYSIS / "oracle_manifest.json")
    assert man["checks"]["molwt_invariant_pass"]
    assert man["checks"]["molwt_invariant_max_abs_delta"] <= RC.ORACLE_INVARIANT_TOL


def test_independent_reparse_matches_direct():
    man = RC.read_json(RC.NEW_ANALYSIS / "oracle_manifest.json")
    ind = man["checks"]["independent_reparse"]
    assert ind["pass"]
    assert ind["max_abs_wc_diff"] <= 1e-9
    assert ind["max_abs_mw_diff"] <= 1e-9


def test_edit_identity_pass():
    man = RC.read_json(RC.NEW_ANALYSIS / "oracle_manifest.json")
    assert man["checks"]["edit_identity"]["pass"]
    assert man["checks"]["edit_identity"]["n_failed"] == 0


def test_no_direct_label_leak():
    man = RC.read_json(RC.NEW_ANALYSIS / "oracle_manifest.json")
    for task in RC.TASKS:
        assert not man["checks"]["direct_leak"][task]["direct_leak"]
        assert man["checks"]["direct_leak"][task]["name_token_hits"] == []


# ---------------------------------------------------------------------------
# raw-graph / RDKit mapping agreement on a real frozen rewire
# ---------------------------------------------------------------------------


def test_raw_graph_rdkit_mapping_on_first_rewire():
    rows = RC.read_csv(RC.NEW_ANALYSIS / "oracle_labels.csv")
    rewires = [r for r in rows if r["source_kind"] == "rewire"]
    assert rewires, "no rewire rows"
    graph_rows = {r["graph_id"]: r for r in RC.read_csv(RC.SOURCE_ANALYSIS / "graph_manifest.csv")}
    raw = RC.SC.load_raw_rows()["rows"]["Lipophilicity"]
    checked = 0
    for r in rewires:
        g = graph_rows[r["graph_id"]]
        rid = int(g["row_id"])
        mp = RC.CHEM.smiles_mapping(str(raw[rid]["smiles"]), raw[rid], rid)
        assert mp["method"] == "identity"
        mapping = [int(x) for x in mp["raw_to_rdkit"]]
        mol = O.parse_smiles(str(raw[rid]["smiles"]))
        remove = RC.parse_bonds(g["remove"]); add = RC.parse_bonds(g["add"])
        edited = O.edited_rdkit_mol(mol, mapping, remove, add)
        base = RC.FEAT.row_from_artifact({**dict(raw[rid]), "mol_index": rid})
        view_row, _ = RC.R.edit_row(base, remove, add)
        chk = O.verify_edit_identity(mol, edited, view_row, mapping)
        assert chk["pass"], chk
        checked += 1
        if checked >= 5:
            break
    assert checked >= 1


# ---------------------------------------------------------------------------
# paired sign / recovery arithmetic
# ---------------------------------------------------------------------------


def test_sign_indicator_zero_reference_nan():
    assert np.isnan(EV._sign_indicator(1.0, 0.0))
    assert EV._sign_indicator(1.0, 2.0) == 1.0
    assert EV._sign_indicator(-1.0, 2.0) == 0.0


def test_recovery_arithmetic_identity():
    p = RC.read_json(RC.NEW_ANALYSIS / "partA_clarifications.json")
    a3 = p["A3_conditional_recovery"]
    assert a3["arithmetic_pass"]
    assert a3["arithmetic_residual_max"] <= 1e-5


def test_three_tables_populations_distinct():
    p = RC.read_json(RC.NEW_ANALYSIS / "partA_clarifications.json")
    t = p["A2_same_population_three_tables"]
    # for amide/test the ALL_K0 population is the whole main queue;
    # RETAINED_K0 / RETAINED_C share the same focal set but differ in the contrast
    assert t["amide/test/ALL_K0"]["n_pairs"] >= t["amide/test/RETAINED_K0"]["n_pairs"]
    assert t["amide/test/RETAINED_K0"]["n_pairs"] == t["amide/test/RETAINED_C"]["n_pairs"]


def test_coverage_never_calls_links_molecules():
    p = RC.read_json(RC.NEW_ANALYSIS / "partA_clarifications.json")
    cov = p["A1_coverage_and_claim_matrix"]
    total = cov["_total"]
    assert "c_links_all_families" in total
    assert total["c_links_all_families"] == 539
    assert total["c_unique_views_all_families"] < total["c_links_all_families"]

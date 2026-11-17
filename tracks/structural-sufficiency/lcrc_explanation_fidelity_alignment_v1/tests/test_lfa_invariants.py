"""Invariants that can be checked without a GPU or a full experiment.

Heavy end-to-end gates live in ``code/lfa_gates.py`` and run on the frozen
models; this file pins the cheap structural invariants that the aggregation
relies on, so a regression fails here in seconds rather than in a report.
"""

from __future__ import annotations

import ast
import itertools
import json
import math
from pathlib import Path

import numpy as np
import pytest

import lfa_common as C
import lfa_data as D
import lfa_gates as G
import lfa_search as S


# ---------------------------------------------------------------------------
# frozen protocol
# ---------------------------------------------------------------------------


def test_freeze_hash_matches_recorded_protocol():
    path = C.ANALYSIS_DIR / "protocol_freeze.json"
    if not path.is_file():
        pytest.skip("protocol freeze not written yet")
    recorded = C.read_json(path)
    assert recorded["freeze_hash"] == C.freeze_hash()
    assert recorded["constants"] == C.FROZEN_CONSTANTS


def test_frozen_search_scope():
    assert (C.SEARCH_MIN_K, C.SEARCH_MAX_K, C.BEAM_WIDTH) == (2, 6, 12)
    assert C.DOSE_GATE == 0.5
    assert C.N_RANDOM == 16 and C.MIN_RANDOM == 8
    assert set(C.SELECTORS) == {"A_LCRC_GRAD", "B_LCRC_LOCAL", "C_LCRC_MASK50",
                                "D_TOPOLOGY", "E_GINE_GRAD", "F_GINE_MASK50"}


# ---------------------------------------------------------------------------
# selection and tie-breaks
# ---------------------------------------------------------------------------


def test_rank_key_tie_break_order():
    entries = [((0, 1, 2), 1.0), ((0, 1), 1.0), ((0, 2), 1.0), ((3, 4), 2.0)]
    ordered = sorted(entries, key=lambda kv: S.rank_key(kv[1], kv[0]))
    assert [b for b, _ in ordered] == [(3, 4), (0, 1), (0, 2), (0, 1, 2)]


def test_select_top1_is_order_independent():
    pool = [(0, 1), (0, 2), (1, 2), (0, 1, 2), (2, 3)]
    scores = [1.0, 1.0, 0.5, 1.0, 0.5]
    first = S.select_top1(pool, scores)
    for perm in itertools.islice(itertools.permutations(range(len(pool))), 20):
        idx = S.select_top1([pool[i] for i in perm], [scores[i] for i in perm])
        assert pool[perm[idx]] == pool[first]


def test_scores_are_deterministic_and_scale_free():
    rng = np.random.default_rng([C.SAMPLE_SEED, 1])
    grad = rng.normal(size=9)
    bonds = [0, 3, 4]
    a = S.score_functions.__name__  # keep import honest
    assert a == "score_functions"
    import lfa_common as _C
    AS = _C.source_modules()["as_search"]
    v1 = AS.score_discovery(bonds, grad)
    v2 = AS.score_discovery(list(reversed(bonds)), grad)
    assert v1 == pytest.approx(v2)
    assert v1 == pytest.approx(abs(sum(grad[b] for b in bonds)) / math.sqrt(len(bonds)))


# ---------------------------------------------------------------------------
# oracle capture and missing values
# ---------------------------------------------------------------------------


def test_capture_zero_denominator_is_missing_not_zero():
    zeros = np.zeros(5)
    assert G.capture(zeros, [0, 1]) is None
    contrib = np.asarray([1.0, -3.0, 0.0, 0.0, 0.0])
    assert G.capture(contrib, [1]) == pytest.approx(0.75)
    assert G.capture(contrib, [2, 3]) == pytest.approx(0.0)
    assert G.capture(contrib, []) == pytest.approx(0.0)


def test_capture_is_mass_capture_of_absolute_contributions():
    contrib = np.asarray([2.0, -2.0, 1.0, -1.0])
    assert G.capture(contrib, [0, 1]) == pytest.approx(4.0 / 6.0)
    assert G.capture(contrib, [0]) == pytest.approx(1.0 / 3.0)


def test_oracle_payload_is_the_frozen_asset():
    payload = D.oracle_payload()
    path = C.XR_DIR / "analysis" / "ogbg-mollipo.npz"
    assert C.sha256_file(path) == C.EXPECTED_ASSET_SHA256["ogbg-mollipo.npz"]
    assert np.asarray(payload["WC_LOGP"]).shape == (4200,)
    ptr = np.asarray(payload["atom_ptr"], dtype=np.int64)
    assert ptr.size == 4201
    assert int(ptr[-1]) == np.asarray(payload["WC_LOGP_atom"]).size


def test_support_label_never_hides_missing_units():
    assert C.support_label(0, 0) == "NO_SUPPORT"
    # a non-empty slice with too few independent components is low, not OK;
    # an empty slice (e.g. both non-Train sides absent) is NO_SUPPORT.
    assert C.support_label(10, 0) == "LOW_SUPPORT"
    assert C.support_label(10, 5) == "LOW_SUPPORT"
    assert C.support_label(200, 60) == "OK"


def test_positive_concentration_reports_zero_effects():
    conc = C.positive_concentration([0.0, 0.0, 1.0, 2.0])
    assert conc["n_zero"] == 2
    assert conc["total_positive"] == pytest.approx(3.0)
    assert conc["top_share"] == pytest.approx(2.0 / 3.0)
    assert C.positive_concentration([0.0, 0.0])["top_share"] is None


# ---------------------------------------------------------------------------
# label blindness of the discovery code
# ---------------------------------------------------------------------------

FORBIDDEN = ("label", "oracle", "edit_atoms", "edit_bonds", "pair_id", "y_raw",
             "targets_raw", "target_std", "target_values", "true_value")


def test_discovery_modules_reference_no_labels():
    for name in ("lfa_search.py", "lfa_pool.py"):
        tree = ast.parse((C.CODE_DIR / name).read_text(encoding="utf-8"))
        used = set()
        for node in ast.walk(tree):
            if isinstance(node, ast.Name):
                used.add(node.id.lower())
            elif isinstance(node, ast.Attribute):
                used.add(node.attr.lower())
        hits = [n for n in used if any(tok in n for tok in FORBIDDEN)]
        assert not hits, f"{name} references forbidden identifiers: {hits}"


def test_discovery_callables_take_no_labels():
    import inspect
    import lfa_pool

    for fn in (lfa_pool.build_molecule, S.generate_families, S.score_functions,
               S.matched_controls, S.shared_pool, S.select_top1):
        for arg in inspect.signature(fn).parameters:
            assert not any(tok in arg.lower() for tok in FORBIDDEN), (fn.__name__, arg)


# ---------------------------------------------------------------------------
# dose path
# ---------------------------------------------------------------------------


def test_dose_masks_put_half_on_selected_bonds_only():
    grid = np.linspace(0.0, 1.0, 5)
    masks = G._path_masks(6, [1, 4], grid)
    assert len(masks) == 5
    assert masks[0][[1, 4]].tolist() == [1.0, 1.0]
    assert masks[2][[1, 4]].tolist() == [0.5, 0.5]
    assert masks[-1][[1, 4]].tolist() == [0.0, 0.0]
    assert masks[2][[0, 2, 3, 5]].tolist() == [1.0, 1.0, 1.0, 1.0]


def test_random_control_seeds_are_frozen():
    assert C.CONTROL_SEED == 20261117
    assert C.SAMPLE_SEED == 20261117
    assert C.BOOTSTRAP_SEED == 20261117
    assert C.BOOTSTRAP_REPLICATES == 2000


# ---------------------------------------------------------------------------
# aggregation accounting
# ---------------------------------------------------------------------------


def _toy_rows(n=4, k=2):
    rows = []
    for i in range(n):
        rows.append({
            "split": "valid", "method": "B_LCRC_LOCAL", "fidelity_model": "LCRC", "status": "OK",
            "cache_row": i, "k": k, "n_atoms": 3, "oracle_enrichment": 0.1 * (i - 1),
            "capture_top1": 0.2 + 0.01 * i, "fidelity_advantage_std": 0.05 * (i - 2),
            "abs_delta0_std": 0.3, "half_over_full": 0.5, "sign_consistent": True,
            "controls_missing": False, "fidelity_missing": False, "hit_own_top1": True,
            "n_s2_affected": 3,
        })
    return rows


def test_summary_denominator_is_the_scope_not_the_evaluable_subset():
    import lfa_main

    rows = _toy_rows(4)
    rows[0]["oracle_enrichment"] = None
    rows[1]["fidelity_advantage_std"] = None
    summary = lfa_main.method_summary(rows, split="valid", n_scope=8)["B_LCRC_LOCAL"]
    assert summary["n"] == 4
    assert summary["n_oracle_evaluable"] == 3
    assert summary["n_fidelity_evaluable"] == 3
    assert summary["oracle_evaluable_fraction"] == pytest.approx(3 / 8)
    assert summary["fidelity_evaluable_fraction"] == pytest.approx(3 / 8)
    assert summary["n_controls_missing_oracle"] == 0
    assert summary["oracle_support"] == "INSUFFICIENT_SUPPORT"


def test_summary_counts_zero_and_positive_effects_separately():
    import lfa_main

    rows = _toy_rows(4)
    summary = lfa_main.method_summary(rows, split="valid", n_scope=4)["B_LCRC_LOCAL"]
    assert summary["enrichment_eq0_fraction"] == pytest.approx(0.25)
    assert summary["enrichment_gt0_fraction"] == pytest.approx(0.5)
    assert summary["enrichment_lt0_fraction"] == pytest.approx(0.25)
    assert summary["both_positive"] + summary["only_oracle"] + summary["only_fidelity"] + summary["neither"] == 4


def test_paired_difference_uses_only_common_molecules():
    import lfa_main

    base = _toy_rows(4)
    rows = []
    for row in base:
        rows.append(dict(row, method="A_LCRC_GRAD"))
        rows.append(dict(row, method="B_LCRC_LOCAL", oracle_enrichment=row["oracle_enrichment"] + 0.05))
    rows[6]["oracle_enrichment"] = None          # A on cache_row 3: missing
    paired = lfa_main.paired_differences(rows, split="valid")
    ca = next(r for r in paired if r["method_a"] == "A_LCRC_GRAD" and r["method_b"] == "B_LCRC_LOCAL")
    assert ca["n_pairs"] == 3                    # cache_rows 0..2, not the missing one
    assert ca["mean_diff"] == pytest.approx(-0.05)
    assert ca["same_fidelity_family"] is True


def test_bootstrap_is_deterministic_and_clustered():
    values = np.arange(10, dtype=np.float64)
    mask = np.ones(10, dtype=bool)
    one = C.bootstrap_columns({"x": values}, mask)
    two = C.bootstrap_columns({"x": values}, mask)
    assert one["ci95"]["x"]["mean"] == two["ci95"]["x"]["mean"]
    units = np.asarray([0, 0, 0, 1, 1, 1, 2, 2, 2, 3])
    clustered = C.bootstrap_columns({"x": values}, mask, unit_ids=units)
    assert clustered["ci95"]["x"]["mean"][0] <= clustered["ci95"]["x"]["mean"][1]


# ---------------------------------------------------------------------------
# region composition bookkeeping
# ---------------------------------------------------------------------------


def test_region_fractions_sum_to_one():
    import lfa_aux

    class Mol:
        n_bonds = 4
        n_atoms = 5
        bond_u = np.asarray([0, 1, 2, 3])
        bond_v = np.asarray([1, 2, 3, 4])

        def bond_atoms(self, bonds):
            return sorted({int(self.bond_u[b]) for b in bonds} | {int(self.bond_v[b]) for b in bonds})

    regions = {
        "var_atoms": {0}, "boundary_atoms": {1}, "core_atoms": {2, 3, 4},
        "var_bonds": {(0, 1)}, "attach_bonds": {(1, 2)}, "changed_core_bonds": set(),
        "other_bonds": {(2, 3), (3, 4)},
    }
    out = lfa_aux.composition(Mol(), [0, 1, 2], regions)
    assert out["atom_fraction_var_atoms"] == pytest.approx(1 / 4)
    assert out["atom_fraction_boundary_atoms"] == pytest.approx(1 / 4)
    assert out["atom_fraction_core_atoms"] == pytest.approx(2 / 4)
    total = (out["bond_fraction_var_bonds"] + out["bond_fraction_attach_bonds"]
             + out["bond_fraction_changed_core_bonds"] + out["bond_fraction_other_bonds"])
    assert total == pytest.approx(1.0)

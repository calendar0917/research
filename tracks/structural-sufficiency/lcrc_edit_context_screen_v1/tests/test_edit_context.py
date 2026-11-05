"""Unit tests for the edit-context screen (CPU only, synthetic data)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import ec_common as C  # noqa: E402
import ec_score as S  # noqa: E402
import ec_units as U  # noqa: E402


def _unit(class_id: str, core: int, comp: str, pair: str) -> dict:
    return {
        "unit_id": f"{class_id}|{core}|{comp}",
        "class_id": class_id,
        "core_class": core,
        "component": comp,
        "pair_ids": [pair],
        "n_pairs": 1,
        "partitions": ["diagnose"],
        "rows": [],
    }


def test_build_comparisons_requires_new_component_and_core():
    units = [
        _unit("X=>Y", 1, "c1", "p1"),
        _unit("X=>Y", 1, "c2", "p2"),  # same core -> not a context comparison
        _unit("X=>Y", 2, "c1", "p3"),  # same component -> not a comparison
        _unit("X=>Y", 3, "c3", "p4"),  # valid partner for unit 1
        _unit("A=>B", 9, "c9", "p9"),
    ]
    comps = U.build_comparisons(units)
    pairs = {frozenset((c["unit_p"], c["unit_q"])) for c in comps}
    assert frozenset(("X=>Y|1|c1", "X=>Y|3|c3")) in pairs
    assert frozenset(("X=>Y|1|c2", "X=>Y|2|c1")) in pairs
    assert frozenset(("X=>Y|1|c1", "X=>Y|2|c1")) not in pairs
    assert frozenset(("X=>Y|1|c1", "X=>Y|1|c2")) not in pairs
    assert all(c["class_id"] == "X=>Y" for c in comps)
    # 4 admissible pairings exist; the two same-core/same-component ones do not
    assert len(comps) == 4
    for c in comps:
        assert c["component_p"] != c["component_q"]
        assert c["core_class_p"] != c["core_class_q"]


def test_greedy_non_overlapping_is_deterministic_and_disjoint():
    units = [
        _unit("F", 1, "c1", "p1"),
        _unit("F", 2, "c2", "p2"),
        _unit("F", 3, "c3", "p3"),
        _unit("F", 4, "c4", "p4"),
    ]
    comps = U.build_comparisons(units)
    first = U.greedy_non_overlapping(comps)
    second = U.greedy_non_overlapping(list(reversed(comps)))
    used: set[str] = set()
    for cid in first["comparison_ids"]:
        comp = next(c for c in comps if c["comparison_id"] == cid)
        assert comp["component_p"] not in used and comp["component_q"] not in used
        used.update({comp["component_p"], comp["component_q"]})
    assert first["n_kept"] == 2  # c1-c2 and c3-c4
    assert first["n_kept"] == second["n_kept"]


def test_support_gates():
    ok = U.support_gates(
        {"n_families_with_comparisons": 5, "n_components_in_comparisons": 15, "n_comparisons": 20}
    )
    assert all(g["pass"] for g in ok.values())
    bad = U.support_gates(
        {"n_families_with_comparisons": 4, "n_components_in_comparisons": 15, "n_comparisons": 19}
    )
    assert not bad["min_families"]["pass"] and not bad["min_comparisons"]["pass"]


def _comparison_frame() -> pd.DataFrame:
    # family A: two comparisons, family B: one comparison
    return pd.DataFrame(
        [
            {"class_id": "A", "context_y": 0.2, "cp": 0.2},
            {"class_id": "A", "context_y": -0.4, "cp": -0.2},
            {"class_id": "B", "context_y": 1.0, "cp": 0.0},
        ]
    )


def test_family_equal_gain_hand_computed():
    df = _comparison_frame()
    g = S.family_gains(df, "cp").set_index("class_id")
    # family A: |0.2|,|0.4| -> mae_zero 0.3 ; errors |0.0|,|0.2| -> mae_pred 0.1
    assert g.loc["A", "mae_zero"] == pytest.approx(0.3)
    assert g.loc["A", "mae_pred"] == pytest.approx(0.1)
    assert g.loc["A", "gain"] == pytest.approx(0.2)
    # family B: mae_zero 1.0 ; mae_pred 1.0 -> gain 0
    assert g.loc["B", "gain"] == pytest.approx(0.0)
    # family-equal mean is (0.2 + 0)/2, not the comparison-weighted value
    out = S.primary_gain(df, "cp")
    assert out["family_equal_gain"] == pytest.approx(0.1)
    assert out["n_families"] == 2 and out["n_comparisons"] == 3


def test_zero_predictor_gives_zero_gain():
    df = _comparison_frame()
    df["zero"] = 0.0
    out = S.primary_gain(df, "zero")
    assert out["family_equal_gain"] == pytest.approx(0.0)


def test_direction_accuracy_threshold_and_ties():
    df = pd.DataFrame(
        [
            {"class_id": "A", "context_y": 0.3, "cp": 0.1},
            {"class_id": "A", "context_y": -0.3, "cp": 0.1},
            {"class_id": "B", "context_y": 0.25, "cp": 0.0},
            {"class_id": "B", "context_y": 0.05, "cp": 0.05},  # below threshold
        ]
    )
    df["abs_context_y"] = df["context_y"].abs()
    df["in_effect_subset"] = df["abs_context_y"] >= C.CONTEXT_EFFECT_THRESHOLD
    out = S.direction_accuracy(df, "cp")
    assert out["n_comparisons"] == 3
    assert out["n_correct"] == 1
    assert out["accuracy"] == pytest.approx(1 / 3)
    assert out["family_equal_accuracy"] == pytest.approx((0.5 + 0.0) / 2)


def test_leave_one_family_out_detects_a_dominant_family():
    gains = pd.DataFrame(
        [
            {"class_id": "A", "n_comparisons": 1, "mae_zero": 0.0, "mae_pred": 0.0, "gain": 3.0},
            {"class_id": "B", "n_comparisons": 1, "mae_zero": 0.0, "mae_pred": 0.0, "gain": -0.4},
            {"class_id": "C", "n_comparisons": 1, "mae_zero": 0.0, "mae_pred": 0.0, "gain": -0.4},
        ]
    )
    out = S.leave_one_family_out(gains)
    assert out["any_flip"] is True
    assert out["families_whose_removal_flips_sign"] == ["A"]
    # removing A leaves a negative mean
    row = next(r for r in out["table"] if r["removed_family"] == "A")
    assert row["mean_gain_without"] < 0


def test_permutation_control_preserves_family_distribution():
    units_df = pd.DataFrame(
        [
            {"unit_id": "F|1|c1", "class_id": "F", "dp": 0.5},
            {"unit_id": "F|2|c2", "class_id": "F", "dp": -0.5},
            {"unit_id": "G|3|c3", "class_id": "G", "dp": 0.2},
        ]
    )
    df = pd.DataFrame(
        [
            {
                "class_id": "F",
                "unit_p": "F|1|c1",
                "unit_q": "F|2|c2",
                "context_y": 1.0,
                "context_pred_C1:s0": 1.0,
            },
            {
                "class_id": "G",
                "unit_p": "G|3|c3",
                "unit_q": "F|1|c1",
                "context_y": 0.3,
                "context_pred_C1:s0": 0.2,
            },
        ]
    )
    out = S.permutation_control(
        df, units_df, {"C1:s0": "dp"}, replicates=20, seed=0
    )
    assert out["replicates"] == 20
    assert out["units"]["C1:s0"]["replicates"] == 20
    # a family with a single unit cannot be permuted: its contribution is fixed,
    # so the null for a family-constant predictor is exactly the observed value.
    units_const = units_df.copy()
    units_const["dp"] = 0.0
    out2 = S.permutation_control(df, units_const, {"C1:s0": "dp"}, replicates=5, seed=0)
    assert out2["units"]["C1:s0"]["null_median"] == pytest.approx(0.0)


def test_decide_three_labels():
    counts = U.support_gates(
        {"n_families_with_comparisons": 10, "n_components_in_comparisons": 20, "n_comparisons": 40}
    )
    loo_ok = {"families_whose_removal_flips_sign": [], "min_mean_gain_without": 0.02, "any_flip": False}
    loo_bad = {"families_whose_removal_flips_sign": ["A"], "min_mean_gain_without": -0.01, "any_flip": True}
    boot = {"ci_2.5": -0.1, "ci_97.5": 0.2}
    promising = {
        "C1:s0": 0.10,
        "C1:s1": 0.02,
        "C1:s2": 0.11,
        "A0:s0": 0.1,
        "A0:s1": 0.1,
        "A0:s2": 0.1,
    }
    out = S.decide(
        promising,
        seed_mean_null_median=-0.05,
        seed_mean_loo=loo_ok,
        seed_mean_bootstrap=boot,
        support_ok=True,
        support_counts=counts,
    )
    assert out["status"] == "PILOT_PROMISING"
    small = dict(promising, **{"C1:s0": 0.01, "C1:s1": 0.0, "C1:s2": 0.01})
    out = S.decide(
        small,
        seed_mean_null_median=-0.05,
        seed_mean_loo=loo_ok,
        seed_mean_bootstrap=boot,
        support_ok=True,
        support_counts=counts,
    )
    assert out["status"] == "PILOT_NOT_ESTABLISHED"
    dominated = dict(promising, **{"C1:s0": 3.0, "C1:s1": 0.0, "C1:s2": 0.0})
    out = S.decide(
        dominated,
        seed_mean_null_median=-0.05,
        seed_mean_loo=loo_bad,
        seed_mean_bootstrap=boot,
        support_ok=True,
        support_counts=counts,
    )
    assert out["status"] == "PILOT_NOT_ESTABLISHED"
    bad_support = U.support_gates(
        {"n_families_with_comparisons": 3, "n_components_in_comparisons": 5, "n_comparisons": 6}
    )
    out = S.decide(
        promising,
        seed_mean_null_median=-0.05,
        seed_mean_loo=loo_ok,
        seed_mean_bootstrap=boot,
        support_ok=False,
        support_counts=bad_support,
    )
    assert out["status"] == "INSUFFICIENT_CONTEXT_SUPPORT"


def test_pair_dy_requires_train_rows():
    labels = {"row_to_y": {1: 1.0, 2: -0.5}}
    assert S.pair_dy(labels, 1, 2) == pytest.approx(-1.5)
    with pytest.raises(KeyError):
        S.pair_dy(labels, 1, 3)


def test_unit_effects_average_repeated_edits():
    labels = {"row_to_y": {1: 0.0, 2: 1.0, 3: 0.0, 4: 3.0}}
    units = [
        {
            "unit_id": "F|1|c1",
            "class_id": "F",
            "core_class": 1,
            "component": "c1",
            "pair_ids": ["p1", "p2"],
            "n_pairs": 2,
            "partitions": ["audit"],
            "rows": [],
        }
    ]
    traces = pd.DataFrame(
        [
            {
                "pair_id": "p1",
                "unit": u,
                "row_a": 1,
                "row_b": 2,
                "d_pred_raw": 0.2,
                "model": u.split(":")[0],
                "seed": int(u.split(":s")[1]),
                "partition": "audit",
                "class_id": "F",
                "core_class": 1,
                "pred_a_raw": 0.0,
                "pred_b_raw": 0.2,
            }
            for u in C.MODEL_UNITS
        ]
        + [
            {
                "pair_id": "p2",
                "unit": u,
                "row_a": 3,
                "row_b": 4,
                "d_pred_raw": 1.0,
                "model": u.split(":")[0],
                "seed": int(u.split(":s")[1]),
                "partition": "audit",
                "class_id": "F",
                "core_class": 1,
                "pred_a_raw": 0.0,
                "pred_b_raw": 1.0,
            }
            for u in C.MODEL_UNITS
        ]
    )
    df = S.unit_effects(units, labels, traces)
    assert len(df) == 1
    assert df.loc[0, "dy"] == pytest.approx(((1.0 - 0.0) + (3.0 - 0.0)) / 2)
    assert df.loc[0, "dp_C1:s0"] == pytest.approx((0.2 + 1.0) / 2)
    assert df.loc[0, "n_pairs_in_unit"] == 2

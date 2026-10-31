"""Registry scope, tail diagnostic and decision-rule truth table."""

from __future__ import annotations

import numpy as np
import pytest

import v3_common as C
import v3_evaluate as E
import v3_registry as R


def test_registry_is_exactly_two_plus_one_conditional():
    reg = R.planned_registry()
    assert reg["n_runs"] == 3
    assert reg["n_unconditional"] == 2
    assert reg["n_conditional"] == 1
    assert [r["key"] for r in reg["runs"]] == [
        "F1_A_Chem12_R3_s0",
        "F1_A_Chem13_R3_s0",
        "F1_C_Chem12_R3_s0",
    ]
    assert R.unconditional_keys() == ["F1_A_Chem12_R3_s0", "F1_A_Chem13_R3_s0"]
    assert R.conditional_key() == "F1_C_Chem12_R3_s0"
    for row in reg["runs"]:
        assert row["fold"] == 1
        assert row["seed"] == 0
        assert row["recipe"] == "R3"
        assert row["variant"] == "scratch"


def _cmp(delta, delta_mae, robust):
    return {
        "delta_rmse": float(delta),
        "delta_mae": float(delta_mae),
        "tail": {"not_tail_dependent": bool(robust), "retained_fraction": 1.0 if robust else 0.1},
        "bootstrap_rmse": {"ci95_low": -0.01, "ci95_high": 0.10},
    }


def _results(d12, d13, m12=None, m13=None, robust12=True, robust13=True):
    return {
        "comparisons": {
            "A_Chem12": _cmp(d12, d12 if m12 is None else m12, robust12),
            "A_Chem13": _cmp(d13, d13 if m13 is None else m13, robust13),
        },
        "conditional_C": {},
    }


def test_decision_general_chem_prior():
    out = E.decide(_results(0.05, 0.06))
    assert out["verdict"] == "GENERAL_CHEM_PRIOR_SUPPORTED"
    assert out["mollogp_incremental_delta_rmse"] == pytest.approx(0.01)


def test_decision_direct_logp_only():
    out = E.decide(_results(0.01, 0.05))
    assert out["verdict"] == "DIRECT_LOGP_PRIOR_ONLY"


def test_decision_no_gain():
    out = E.decide(_results(0.01, 0.02))
    assert out["verdict"] == "NO_GLOBAL_DESCRIPTOR_GAIN"


def test_decision_tail_dependent_is_inconclusive():
    out = E.decide(_results(0.05, 0.06, robust12=False))
    assert out["verdict"] == "INCONCLUSIVE_TAIL_DEPENDENT"


def test_decision_mollogp_removing_gain_is_inconclusive():
    out = E.decide(_results(0.05, 0.01))
    assert out["verdict"] == "INCONCLUSIVE_MOLLOGP_REMOVES_GAIN"


def test_decision_rmse_mae_conflict_is_inconclusive():
    out = E.decide(_results(0.05, 0.06, m12=-0.001))
    assert out["verdict"] == "INCONCLUSIVE_RMSE_MAE_CONFLICT"


def test_decision_pilot_not_started_without_runs():
    out = E.decide({"comparisons": {}, "conditional_C": {}})
    assert out["verdict"] == "PILOT_NOT_STARTED"


def test_tail_analysis_flags_a_single_extreme_molecule():
    n = 100
    y = np.zeros(n)
    hist = np.full(n, 0.1)
    hist[7] = 10.0
    new = np.zeros(n)
    out = E.tail_analysis(y, hist, new, fraction=0.02)
    assert out["n_dropped"] == 2
    assert out["direction_stable_after_tail_drop"] is True
    assert out["not_tail_dependent"] is False
    assert out["top1_share_of_net_gain"] > 0.9


def test_tail_analysis_accepts_a_uniform_improvement():
    n = 100
    y = np.zeros(n)
    hist = np.linspace(0.2, 0.4, n)
    new = hist * 0.5
    out = E.tail_analysis(y, hist, new, fraction=0.02)
    assert out["delta_full"] > 0
    assert out["delta_drop_top_tail"] > 0
    assert out["not_tail_dependent"] is True
    assert out["top1_share_of_net_gain"] < 1.0


def test_tail_analysis_uses_historical_residuals_for_exclusion():
    n = 50
    y = np.zeros(n)
    hist = np.full(n, 0.1)
    hist[0] = 5.0
    new = np.zeros(n)
    out = E.tail_analysis(y, hist, new, fraction=0.02)
    assert 0 in out["dropped_rows"]

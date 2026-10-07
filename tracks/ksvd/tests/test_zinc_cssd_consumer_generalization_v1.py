"""Focused wiring tests for zinc_cssd_consumer_generalization_v1.

Only the NEW risks of this round: the frozen routing rule, the estimator
roster per route, the FP32 averaging semantics (integers never float-averaged),
the deep-copy independence of captured members, the schedule prefix property,
the general candidate-vs-control group-paired bootstrap (shared resampling,
seeds averaged first) and the frozen retention gate + tie-break.  No data,
no GPU, no old-artifact dependency.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_consumer_generalization_v1 as g,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_task_dictionary_and_cycle_witness_seed0_v1 as zw,
)

torch.set_num_threads(4)


# ---- routing rule -----------------------------------------------------------

def _route_inputs(f120, f240, d0, d1):
    fit = {0: {"epoch120": f120, "epoch240": f240}, 1: {"epoch120": f120, "epoch240": f240}}
    dev = {0: {"epoch120": 0.10, "epoch240": 0.10 + d0}, 1: {"epoch120": 0.10, "epoch240": 0.10 + d1}}
    return fit, dev


def test_route_overfit():
    fit, dev = _route_inputs(0.06, 0.05, +0.004, +0.003)
    r = g.route_from_diagnosis(fit, dev)
    assert r["route"] == "OVERFIT"
    assert r["F240_minus_F120"] == pytest.approx(-0.01)
    assert r["mean_d"] == pytest.approx(0.0035)


def test_route_still_improving():
    fit, dev = _route_inputs(0.06, 0.05, -0.004, -0.003)
    assert g.route_from_diagnosis(fit, dev)["route"] == "STILL_IMPROVING"


def test_route_flat_when_fit_not_improving():
    fit, dev = _route_inputs(0.05, 0.0495, +0.01, +0.01)
    assert g.route_from_diagnosis(fit, dev)["route"] == "FLAT_OR_MIXED"


def test_route_flat_when_seeds_mixed():
    fit, dev = _route_inputs(0.06, 0.05, +0.004, -0.004)
    assert g.route_from_diagnosis(fit, dev)["route"] == "FLAT_OR_MIXED"


def test_route_boundary_just_inside_and_outside():
    # exactly-at-threshold behaviour is FP-representation dependent; test
    # clearly inside (OVERFIT) and just outside (FLAT_OR_MIXED) instead
    fit, dev = _route_inputs(0.0521, 0.05, +0.0025, +0.0025)
    assert g.route_from_diagnosis(fit, dev)["route"] == "OVERFIT"
    fit, dev = _route_inputs(0.0521, 0.05, +0.0025, +0.0012)
    assert g.route_from_diagnosis(fit, dev)["route"] == "FLAT_OR_MIXED"
    fit, dev = _route_inputs(0.0521, 0.05, -0.0025, -0.0025)
    assert g.route_from_diagnosis(fit, dev)["route"] == "STILL_IMPROVING"
    fit, dev = _route_inputs(0.0521, 0.05, -0.0025, -0.0012)
    assert g.route_from_diagnosis(fit, dev)["route"] == "FLAT_OR_MIXED"


# ---- estimator roster -------------------------------------------------------

def test_estimator_specs_members_frozen():
    over = g.estimator_specs("OVERFIT")
    assert over["epochs"] == 240
    assert over["estimators"]["CTRL240"]["members"] == [236, 237, 238, 239, 240]
    assert over["estimators"]["EARLY120"]["members"] == [116, 117, 118, 119, 120]
    assert over["estimators"]["WIDE5_240"]["members"] == [200, 210, 220, 230, 240]
    assert over["estimators"]["CTRL240"]["candidate"] is False
    assert {k for k, v in over["estimators"].items() if v["candidate"]} == {"EARLY120", "WIDE5_240"}
    still = g.estimator_specs("STILL_IMPROVING")
    assert still["epochs"] == 360
    assert still["estimators"]["LONG360"]["members"] == [356, 357, 358, 359, 360]
    assert still["estimators"]["CTRL240"]["members"] == [236, 237, 238, 239, 240]
    assert {k for k, v in still["estimators"].items() if v["candidate"]} == {"LONG360", "WIDE5_240"}
    flat = g.estimator_specs("FLAT_OR_MIXED")
    assert flat["epochs"] == 240
    assert {k for k, v in flat["estimators"].items() if v["candidate"]} == {"WIDE5_240"}
    for specs in (over, still, flat):
        for name, e in specs["estimators"].items():
            assert len(e["members"]) == 5
            assert name != "CTRL240" or e["candidate"] is False


def test_estimator_specs_rejects_unknown_route():
    with pytest.raises(ValueError):
        g.estimator_specs("MYSTERY")


# ---- averaging semantics ----------------------------------------------------

def _mini_state(seed: int) -> dict[str, torch.Tensor]:
    gen = torch.Generator().manual_seed(seed)
    return {
        "a.weight": torch.randn(4, 3, generator=gen),
        "b.flag": torch.tensor([1, 2], dtype=torch.int64),
    }


def test_average_states_float_mean_and_int_copy():
    states = [_mini_state(s) for s in (1, 2, 3)]
    avg = g.average_states(states)
    assert avg["a.weight"].dtype == torch.float32
    assert torch.allclose(avg["a.weight"], torch.stack([s["a.weight"] for s in states]).mean(0))
    assert torch.equal(avg["b.flag"], states[0]["b.flag"])  # int never float-averaged


def test_average_states_rejects_differing_int_entries():
    states = [_mini_state(1), _mini_state(2)]
    states[1]["b.flag"] = torch.tensor([1, 3], dtype=torch.int64)
    with pytest.raises(RuntimeError):
        g.average_states(states)


def test_average_states_returns_independent_tensors():
    states = [_mini_state(s) for s in (1, 2)]
    avg = g.average_states(states)
    avg["a.weight"].add_(1.0)
    assert not torch.allclose(states[0]["a.weight"], avg["a.weight"])


def test_capture_state_deep_copy_independent():
    model = torch.nn.Linear(3, 2, bias=False)
    captured = g._capture_state(model)
    with torch.no_grad():
        model.weight.add_(1.0)
    assert not torch.allclose(captured["weight"], model.weight)


# ---- schedule prefix property -----------------------------------------------

def test_build_schedule_prefix_property():
    a, _ = zw.build_schedule(50, 5, 7)
    b, _ = zw.build_schedule(50, 10, 7)
    for e in range(5):
        assert np.array_equal(a[e], b[e])


# ---- general candidate-vs-control bootstrap ---------------------------------

def _bootstrap_fixture():
    rng = np.random.default_rng(11)
    groups = np.array(["s1", "s1", "s2", "s2", "s2", "s3"], dtype=object)
    ctrl = {0: rng.normal(0, 1, 6), 1: rng.normal(0, 1, 6)}
    cand = {0: ctrl[0] - 0.5, 1: ctrl[1] - 0.3}
    errs = {
        "CTRL_s0": ctrl[0], "CAND_s0": cand[0],
        "CTRL_s1": ctrl[1], "CAND_s1": cand[1],
    }
    return errs, groups


def test_bootstrap_structure_and_shared_picks():
    errs, groups = _bootstrap_fixture()
    out = g.candidate_paired_bootstrap(errs, groups, "CTRL", ["CAND"], n_boot=200, seed=20261007)
    assert out["n_rows"] == 6 and out["n_groups"] == 3
    assert out["shared_group_resampling"] is True
    avg_ci = out["per_candidate"]["CAND"]["avg_delta_ci95"]
    assert avg_ci[0] <= out["per_candidate"]["CAND"]["avg_delta_mean"] <= avg_ci[1]
    # mean effect ~ mean of the two seed effects (candidate errs are
    # ctrl - 0.5 / ctrl - 0.3, so the paired mean difference is -0.4)
    assert out["per_candidate"]["CAND"]["avg_delta_mean"] == pytest.approx(0.5 * (-0.5 + -0.3), abs=1e-9)


def test_bootstrap_identical_candidate_ci_collapses():
    errs, groups = _bootstrap_fixture()
    errs["SAME_s0"], errs["SAME_s1"] = errs["CTRL_s0"], errs["CTRL_s1"]
    out = g.candidate_paired_bootstrap(errs, groups, "CTRL", ["SAME"], n_boot=200, seed=20261007)
    ci = out["per_candidate"]["SAME"]["avg_delta_ci95"]
    assert ci[0] == pytest.approx(0.0) and ci[1] == pytest.approx(0.0)


def test_bootstrap_shared_picks_deterministic():
    errs, groups = _bootstrap_fixture()
    a = g.candidate_paired_bootstrap(errs, groups, "CTRL", ["CAND"], n_boot=50, seed=20261007)
    b = g.candidate_paired_bootstrap(errs, groups, "CTRL", ["CAND"], n_boot=50, seed=20261007)
    assert a["per_candidate"]["CAND"]["avg_delta_ci95"] == b["per_candidate"]["CAND"]["avg_delta_ci95"]


def test_bootstrap_two_candidates_share_the_same_picks():
    errs, groups = _bootstrap_fixture()
    errs["C2_s0"], errs["C2_s1"] = errs["CTRL_s0"] - 0.1, errs["CTRL_s1"] - 0.1
    one = g.candidate_paired_bootstrap(errs, groups, "CTRL", ["CAND"], n_boot=200, seed=5)
    two = g.candidate_paired_bootstrap(errs, groups, "CTRL", ["CAND", "C2"], n_boot=200, seed=5)
    assert (
        one["per_candidate"]["CAND"]["avg_delta_ci95"]
        == two["per_candidate"]["CAND"]["avg_delta_ci95"]
    )  # adding a candidate never changes another candidate's resampling


# ---- frozen retention gate + tie-break ---------------------------------------

def _gate_args(delta_y, delta_g, mean_y, resp):
    candidates = list(delta_y)
    return (
        candidates,
        {c: {0: delta_y[c][0], 1: delta_y[c][1]} for c in candidates},
        {c: {0: delta_g[c][0], 1: delta_g[c][1]} for c in candidates},
        {c: mean_y[c] for c in candidates},
        {c: resp[c] for c in candidates},
        True,
    )


def test_gate_pass_and_fail_conditions():
    args = _gate_args(
        {"C": (-0.004, -0.004)}, {"C": (-0.001, -0.001)}, {"C": 0.11}, {"C": True},
    )
    out = g.retention_gate(*args)
    assert out["passed_candidates"] == ["C"]
    assert out["winner"] == "C"
    # fails: one seed's y does not improve
    args = _gate_args(
        {"C": (-0.004, +0.001)}, {"C": (-0.001, -0.001)}, {"C": 0.11}, {"C": True},
    )
    assert g.retention_gate(*args)["passed_candidates"] == []
    # fails: mean delta_y above the -0.003 gate
    args = _gate_args(
        {"C": (-0.002, -0.002)}, {"C": (-0.001, -0.001)}, {"C": 0.11}, {"C": True},
    )
    assert g.retention_gate(*args)["passed_candidates"] == []
    # fails: g worsens beyond the 1e-4 band
    args = _gate_args(
        {"C": (-0.004, -0.004)}, {"C": (+0.002, -0.001)}, {"C": 0.11}, {"C": True},
    )
    assert g.retention_gate(*args)["passed_candidates"] == []
    # fails: unresponsive intervention
    args = _gate_args(
        {"C": (-0.004, -0.004)}, {"C": (-0.001, -0.001)}, {"C": 0.11}, {"C": False},
    )
    assert g.retention_gate(*args)["passed_candidates"] == []
    # fails: checks not ok
    args = _gate_args(
        {"C": (-0.004, -0.004)}, {"C": (-0.001, -0.001)}, {"C": 0.11}, {"C": True},
    )
    gate = g.retention_gate(*args[:-1], False)
    assert gate["passed_candidates"] == []


def test_gate_tie_break_prefers_shorter_effective_length():
    # both pass; mean y_raw within 1e-4 -> shorter effective training length wins
    # (EARLY120=120 beats WIDE5_240=240)
    args = _gate_args(
        {"EARLY120": (-0.004, -0.004), "WIDE5_240": (-0.004, -0.004)},
        {"EARLY120": (-0.001, -0.001), "WIDE5_240": (-0.001, -0.001)},
        {"EARLY120": 0.110000, "WIDE5_240": 0.110050},
        {"EARLY120": True, "WIDE5_240": True},
    )
    out = g.retention_gate(*args)
    assert out["winner"] == "EARLY120"
    assert out["tie_break_applied"] == "shorter effective training length, then contiguous last-five mean"


def test_gate_tie_break_prefers_clear_best_mean_y():
    # WIDE5_240 clearly better (diff > 1e-4) -> wins on mean y_raw
    args = _gate_args(
        {"EARLY120": (-0.004, -0.004), "WIDE5_240": (-0.004, -0.004)},
        {"EARLY120": (-0.001, -0.001), "WIDE5_240": (-0.001, -0.001)},
        {"EARLY120": 0.110200, "WIDE5_240": 0.110050},
        {"EARLY120": True, "WIDE5_240": True},
    )
    assert g.retention_gate(*args)["winner"] == "WIDE5_240"


def test_gate_tie_break_equal_mean_prefers_shorter_length_over_contiguity():
    # LONG360 (contiguous, 360) vs WIDE5_240 (spread, 240), both passing and
    # within 1e-4 on mean y_raw -> the SHORTER effective training length wins
    # (the contiguous-last-five tie-break only applies at equal length)
    args = _gate_args(
        {"LONG360": (-0.004, -0.004), "WIDE5_240": (-0.004, -0.004)},
        {"LONG360": (-0.001, -0.001), "WIDE5_240": (-0.001, -0.001)},
        {"LONG360": 0.110000, "WIDE5_240": 0.110020},
        {"LONG360": True, "WIDE5_240": True},
    )
    out = g.retention_gate(*args)
    assert out["winner"] == "WIDE5_240"


# ---- group tables -----------------------------------------------------------

def test_k_group_table_adds_back():
    rng = np.random.default_rng(3)
    e_y = rng.normal(0, 0.1, 100)
    e_g = rng.normal(0, 0.1, 100)
    k = rng.integers(-3, 1, 100)
    k[rng.random(100) < 0.5] = 0
    out = g._k_group_table(e_y, e_g, k)
    assert sum(v["n"] for v in out.values()) == 100
    assert sum(v["C_y"] for v in out.values()) == pytest.approx(float(np.abs(e_y).mean()), abs=1e-12)
    assert sum(v["C_g"] for v in out.values()) == pytest.approx(float(np.abs(e_g).mean()), abs=1e-12)


def test_k0_subgroup_table_bins_cover():
    rng = np.random.default_rng(4)
    n = 200
    e_y = rng.normal(0, 0.1, n)
    e_g = rng.normal(0, 0.1, n)
    k = np.zeros(n, dtype=np.int64)
    k[:5] = -1
    n_nodes = rng.integers(18, 34, n)
    g_abs = rng.random(n)
    out = g._k0_subgroup_table(e_y, e_g, k, n_nodes, g_abs, 0.5)
    assert out["n_k0"] == n - 5
    assert sum(out[a]["n"] for a in ("n_nodes<=23", "n_nodes=24..27", "n_nodes>=28")) == n - 5
    assert sum(out[a]["n"] for a in ("|g|<=fit_p90", "|g|>fit_p90")) == n - 5
    total_c_g = sum(out[a]["C_g"] for a in ("n_nodes<=23", "n_nodes=24..27", "n_nodes>=28"))
    assert total_c_g == pytest.approx(float(np.abs(e_g[k == 0]).sum() / n), abs=1e-12)

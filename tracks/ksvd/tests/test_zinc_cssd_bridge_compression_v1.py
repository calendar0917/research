"""Focused wiring tests for zinc_cssd_bridge_compression_v1.

Only the NEW risks of this round: the SmallMLPBridge contract (shapes, no
biases, the EXACT historical forward formula, the teacher-free fork_rng
init), the per-arm parameter contracts (never the historical 297,539 contract
for B72), the two-metric group-paired bootstrap (shared picks across
arms/seeds/metrics, seeds averaged first), the frozen decision branches A-E
and the noise marker.  No data, no GPU, no old-artifact dependency.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_bridge_compression_v1 as r,
)
from tracks.ksvd.experiments.luyin16 import e2e_dictenv_latent_bridge_v1 as lb

torch.set_num_threads(4)


# ---- SmallMLPBridge contract -----------------------------------------------

def test_small_bridge_parameter_contract():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(r.B72_INIT_SEED_BASE)
        bridge = r.SmallMLPBridge()
    assert int(sum(p.numel() for p in bridge.parameters())) == r.B72_BRIDGE_PARAMETERS == 20736
    assert bridge.fc1.bias is None and bridge.fc2.bias is None
    assert tuple(bridge.fc1.weight.shape) == (72, 144)
    assert tuple(bridge.fc2.weight.shape) == (144, 72)


def test_small_bridge_forward_formula_exact():
    torch.manual_seed(7)
    h = torch.randn(32, 144) * 2.5
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(r.B72_INIT_SEED_BASE + 1)
        bridge = r.SmallMLPBridge()
    scale = torch.sqrt(h.pow(2).mean(dim=1, keepdim=True) + float(lb.BRIDGE_EPS))
    manual = scale * bridge.fc2(F.silu(bridge.fc1(h / scale)))
    with torch.no_grad():
        got = bridge(h)
    assert torch.equal(got, manual)
    assert got.shape == h.shape


def test_small_bridge_teacher_free_init_is_seed_private_and_rng_neutral():
    torch.manual_seed(999)
    sentinel = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(r.B72_INIT_SEED_BASE + 0)
        b0 = r.SmallMLPBridge()
        torch.manual_seed(r.B72_INIT_SEED_BASE + 1)
        b1 = r.SmallMLPBridge()
    assert torch.equal(torch.get_rng_state(), sentinel)  # fork restored the stream
    assert not torch.equal(b0.fc1.weight, b1.fc1.weight)
    assert not torch.equal(b0.fc2.weight, b1.fc2.weight)


def test_arm_parameter_contracts():
    ctrl = r.arm_expected_audit("CTRL288")
    b72 = r.arm_expected_audit("B72")
    assert ctrl == {
        "total_parameters": 297539, "base_body_parameters": 184707,
        "bridge_parameters": 82944, "local_tuple_parameters": 29888,
        "reader_output_parameters": 80,
    }
    assert b72 == {
        "total_parameters": 235331, "base_body_parameters": 184707,
        "bridge_parameters": 20736, "local_tuple_parameters": 29888,
        "reader_output_parameters": 80,
    }
    assert ctrl["total_parameters"] - b72["total_parameters"] == r.B72_PARAMETER_REDUCTION == 62208
    assert r.B72_PARAMETER_REDUCTION / ctrl["total_parameters"] == pytest.approx(0.209, abs=0.001)
    with pytest.raises(ValueError):
        r.arm_expected_audit("RAW")


# ---- two-metric group-paired bootstrap --------------------------------------

def _errs(keys, group_of_row, offset):
    rng = np.random.default_rng(3)
    return {k: rng.random(len(group_of_row)) + offset for k in keys}


def _groups():
    # 6 groups of sizes 1..6 over 21 rows
    sizes = [1, 2, 3, 4, 5, 6]
    return np.asarray(["g%d" % i for i, s in enumerate(sizes) for _ in range(s)], dtype=object)


def test_bootstrap_shared_picks_and_shapes():
    groups = _groups()
    keys = [f"{a}_s{s}" for a in r.ARMS for s in r.SEEDS]
    errs_y = _errs(keys, groups, 0.0)
    errs_g = _errs(keys, groups, 10.0)
    out = r.paired_bootstrap_two_metrics(errs_y, errs_g, groups, n_boot=500)
    assert out["n_rows"] == 21 and out["n_groups"] == 6
    assert out["n_boot"] == 500 and out["boot_seed"] == r.BOOT_SEED == 20261007
    assert out["shared_group_resampling"] and out["shared_picks_across_metrics"]
    for tag in ("delta_y", "delta_g"):
        for s in ("0", "1"):
            lo, hi = out[tag]["per_seed_ci95"][s]
            assert lo <= out[tag]["per_seed_mean"][s] <= hi
        lo, hi = out[tag]["avg_ci95"]
        assert lo <= out[tag]["avg_mean"] <= hi
        # avg = mean of the two per-seed means (seeds averaged first)
        assert out[tag]["avg_mean"] == pytest.approx(
            np.mean([out[tag]["per_seed_mean"][s] for s in ("0", "1")]))


def test_bootstrap_shared_picks_scale_and_shift_cancellation():
    groups = _groups()
    keys = [f"{a}_s{s}" for a in r.ARMS for s in r.SEEDS]
    errs_y = _errs(keys, groups, 0.0)
    errs_g = {k: 2.0 * v for k, v in errs_y.items()}   # exact power-of-two scaling
    out = r.paired_bootstrap_two_metrics(errs_y, errs_g, groups, n_boot=300)
    # shared picks: within one call the g metric is the exact 2x of the y metric
    for side in (0, 1):
        assert out["delta_g"]["avg_ci95"][side] == pytest.approx(
            2 * out["delta_y"]["avg_ci95"][side], abs=1e-15)
    for s in ("0", "1"):
        assert out["delta_g"]["per_seed_mean"][s] == pytest.approx(
            2 * out["delta_y"]["per_seed_mean"][s], abs=1e-15)
    # a constant additive shift on both arms cancels in the paired difference
    shifted = {k: v + 0.05 for k, v in errs_y.items()}
    out2 = r.paired_bootstrap_two_metrics(shifted, shifted, groups, n_boot=300)
    assert out2["delta_y"]["avg_mean"] == pytest.approx(out["delta_y"]["avg_mean"], abs=1e-10)
    assert out2["delta_y"]["avg_ci95"] == pytest.approx(out["delta_y"]["avg_ci95"], abs=1e-10)


def test_bootstrap_rejects_missing_keys():
    groups = _groups()
    with pytest.raises(RuntimeError):
        r.paired_bootstrap_two_metrics({}, {}, groups)


# ---- frozen decision branches A-E -------------------------------------------

def _ci(lo, hi):
    return {
        "delta_y": {"avg_ci95": [lo, hi]},
        "delta_g": {"avg_ci95": [lo, hi]},
    }


def test_branch_A_keep_performance():
    d = r.decision_branches(
        {0: -0.002, 1: -0.001}, {0: -0.001, 1: -0.0005}, _ci(-0.003, -0.0005),
        mean_fit_delta_y=-0.0002, responsive_b72=True, checks_ok=True,
    )
    assert d["branch"] == "keep-performance-candidate"
    assert all(d["conditions"]["A_keep_performance"].values())
    # fit improving clearly less than dev -> capacity-constraint reading
    assert "capacity-constraint" in d["reading"]


def test_branch_A_fit_dev_together_is_recipe_reading():
    d = r.decision_branches(
        {0: -0.002, 1: -0.001}, {0: -0.001, 1: -0.0005}, _ci(-0.003, -0.0005),
        mean_fit_delta_y=-0.0025, responsive_b72=True, checks_ok=True,
    )
    assert d["branch"] == "keep-performance-candidate"
    assert "recipe performance improvement" in d["reading"]


def test_branch_B_keep_cheaper():
    d = r.decision_branches(
        {0: 0.0005, 1: 0.001}, {0: 0.0002, 1: 0.0009}, _ci(-0.0005, 0.0015),
        mean_fit_delta_y=0.0004, responsive_b72=True, checks_ok=True,
    )
    assert d["branch"] == "keep-cheaper-candidate"
    assert all(d["conditions"]["B_keep_cheaper"].values())
    assert "62,208" in d["reading"]


def test_branch_C_readout_not_supported_even_with_good_performance():
    d = r.decision_branches(
        {0: -0.002, 1: -0.001}, {0: -0.001, 1: -0.0005}, _ci(-0.003, -0.0005),
        mean_fit_delta_y=-0.0002, responsive_b72=False, checks_ok=True,
    )
    assert d["branch"] == "dictionary-readout-not-supported"


def test_branch_D_direction_flip():
    d = r.decision_branches(
        {0: -0.001, 1: 0.0025}, {0: 0.0005, 1: 0.0025}, _ci(0.0, 0.003),
        mean_fit_delta_y=0.0006, responsive_b72=True, checks_ok=True,
    )
    assert d["branch"] == "inconclusive"
    assert d["conditions"]["D_direction_flip"]


def test_branch_D_ci_spans_decision_space():
    d = r.decision_branches(
        {0: 0.0012, 1: 0.0013}, {0: 0.0004, 1: 0.0004}, _ci(-0.0002, 0.0025),
        mean_fit_delta_y=0.0006, responsive_b72=True, checks_ok=True,
    )
    assert d["branch"] == "inconclusive"
    assert d["conditions"]["D_ci_spans_decision_space"]


def test_branch_E_close_with_subreadings():
    d = r.decision_branches(
        {0: 0.0012, 1: 0.0013}, {0: 0.0004, 1: 0.0004}, _ci(0.0005, 0.002),
        mean_fit_delta_y=0.0006, responsive_b72=True, checks_ok=True,
    )
    assert d["branch"] == "close-b72"
    assert "restricted-capacity" in d["reading"]
    d2 = r.decision_branches(
        {0: 0.0012, 1: 0.0013}, {0: 0.0004, 1: 0.0004}, _ci(0.0005, 0.002),
        mean_fit_delta_y=0.00002, responsive_b72=True, checks_ok=True,
    )
    assert d2["branch"] == "close-b72"
    assert "no benefit seen" in d2["reading"]
    d3 = r.decision_branches(
        {0: 0.0012, 1: 0.0013}, {0: 0.0004, 1: 0.0004}, _ci(0.0005, 0.002),
        mean_fit_delta_y=-0.0006, responsive_b72=True, checks_ok=True,
    )
    assert d3["branch"] == "close-b72"
    assert "overfitting" in d3["reading"]


def test_branch_A_requires_ci_upper_below_zero():
    d = r.decision_branches(
        {0: -0.002, 1: -0.001}, {0: -0.001, 1: -0.0005}, _ci(-0.003, 0.00001),
        mean_fit_delta_y=-0.0002, responsive_b72=True, checks_ok=True,
    )
    assert d["branch"] != "keep-performance-candidate"


def test_branch_checks_not_ok_blocks_a_and_b():
    d = r.decision_branches(
        {0: -0.002, 1: -0.001}, {0: -0.001, 1: -0.0005}, _ci(-0.003, -0.0005),
        mean_fit_delta_y=-0.0002, responsive_b72=True, checks_ok=False,
    )
    assert d["branch"] not in ("keep-performance-candidate", "keep-cheaper-candidate")


# ---- noise marker -----------------------------------------------------------

def test_noise_marker_formula():
    assert float(max(r.REPLAY_TOL, 10.0 * 1e-6)) == 1e-4
    assert float(max(r.REPLAY_TOL, 10.0 * 5e-4)) == 5e-3

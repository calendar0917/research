"""Focused wiring tests for zinc_cssd_pair_joint_residual_v1.

Only the NEW risks of this round: the JointEndpointResidual module (shapes,
no biases, parameter counts, W2 exactly zero / W1 live default draw, private
seed determinism, fork_rng RNG neutrality, endpoint-swap symmetry BY
CONSTRUCTION, step-0 delta exactly zero), the 2-D collision demo (the pair
interface's per-coordinate statistics collide where a joint function on
[left,right] does not), the build-arm contract constants, the single-contrast
group-paired bootstrap (shared picks across arms AND metrics) and the frozen
exploratory-reading branches.  No data, no GPU, no old-artifact dependency.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_pair_joint_residual_v1 as r,
)

torch.set_num_threads(4)


# ---- joint residual module contracts ----------------------------------------


def test_joint_residual_contract():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(r.JOINT_INIT_SEED)
        jr = r.JointEndpointResidual()
    assert int(sum(p.numel() for p in jr.parameters())) == r.JOINT_PARAMETERS == 12_288
    assert jr.w1.bias is None and jr.w2.bias is None
    assert tuple(jr.w1.weight.shape) == (64, 144)
    assert tuple(jr.w2.weight.shape) == (48, 64)
    # W2 exactly zero, W1 a live default draw (never both zero: no dead branch)
    assert float(jr.w2.weight.abs().sum()) == 0.0
    assert float(jr.w1.weight.abs().sum()) > 0.0
    # the parameter split: 144*64 + 64*48
    assert jr.w1.weight.numel() == 144 * 64 and jr.w2.weight.numel() == 64 * 48


def test_joint_residual_private_seed_deterministic_and_rng_neutral():
    torch.manual_seed(123)
    rng_before = torch.get_rng_state().clone()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(r.JOINT_INIT_SEED)
        a = r.JointEndpointResidual()
    assert torch.equal(rng_before, torch.get_rng_state())  # fork_rng neutrality
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(r.JOINT_INIT_SEED)
        b = r.JointEndpointResidual()
    assert torch.equal(a.w1.weight, b.w1.weight)
    assert torch.equal(a.w2.weight, b.w2.weight)


def test_joint_residual_step0_delta_exactly_zero_and_symmetric_when_active():
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(r.JOINT_INIT_SEED)
        jr = r.JointEndpointResidual()
    g = torch.Generator().manual_seed(7)
    left = torch.randn(9, 48, generator=g)
    right = torch.randn(9, 48, generator=g)
    rel = torch.randn(9, 48, generator=g)
    d0 = jr(left, right, rel)
    assert float(d0.abs().sum()) == 0.0  # W2 zero => delta exactly zero at step 0
    # activate the branch (artificial W2, fit-only style) => the two orders are
    # evaluated by the SAME psi and the averaged delta is endpoint-swap symmetric
    with torch.no_grad():
        jr.w2.weight.copy_(torch.randn_like(jr.w2.weight) * 0.1)
    d1 = jr(left, right, rel)
    d2 = jr(right, left, rel)
    assert torch.equal(d1, d2)
    assert float(d1.abs().sum()) > 0.0
    # psi is a deterministic shared function (no dropout): repeated calls agree
    assert torch.equal(d1, jr(left, right, rel))


def test_joint_residual_gradient_structure_at_step0():
    """W1's first-backward gradient is zero BY CONSTRUCTION (W2 = 0); W2's
    gradient is reachable — the frozen init contract, never 'rescued'."""
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(r.JOINT_INIT_SEED)
        jr = r.JointEndpointResidual()
    g = torch.Generator().manual_seed(3)
    left, right, rel = (torch.randn(5, 48, generator=g) for _ in range(3))
    delta = jr(left, right, rel)
    delta.sum().backward()
    assert float(jr.w1.weight.grad.abs().sum()) == 0.0
    assert float(jr.w2.weight.grad.norm()) > 0.0


def test_collision_demo_contract():
    """{(1,0),(0,1)} vs {(1,1),(0,0)}: the per-coordinate pair statistics
    collide; the concatenated endpoints do not.  Interface property only."""
    e1, e2 = torch.tensor([[1.0, 0.0]]), torch.tensor([[0.0, 1.0]])
    e3, e4 = torch.tensor([[1.0, 1.0]]), torch.tensor([[0.0, 0.0]])
    stats_a = [e1 + e2, torch.abs(e1 - e2), e1 * e2]
    stats_b = [e3 + e4, torch.abs(e3 - e4), e3 * e4]
    assert all(torch.equal(sa, sb) for sa, sb in zip(stats_a, stats_b))
    assert not torch.equal(torch.cat([e1, e2], dim=1), torch.cat([e3, e4], dim=1))
    # a shared linear function on [left, right] separates the two pairs
    w = torch.tensor([[1.0, 2.0, -1.0, 0.0]])
    assert not torch.equal(
        F.linear(torch.cat([e1, e2], dim=1), w), F.linear(torch.cat([e3, e4], dim=1), w)
    )


# ---- round-level contracts (constants only; no data/factory) ----------------


def test_arm_and_parameter_contracts():
    assert r.ARMS == ("CTRL_C", "JOINT_RESIDUAL")
    assert r.CONTROL == "CTRL_C"
    assert r.CANDIDATES == ("JOINT_RESIDUAL",)
    assert r.SEEDS == (0,)
    assert r.EXPECTED_CTRL_C_AUDIT == {
        "total_parameters": 325_187,
        "base_body_parameters": 212_355,
        "bridge_parameters": 82_944,
        "local_tuple_parameters": 29_888,
        "reader_output_parameters": 80,
    }
    assert r.EXPECTED_JOINT_AUDIT["total_parameters"] == 337_475
    assert r.EXPECTED_JOINT_AUDIT["base_body_parameters"] == 224_643
    assert r.EXPECTED_JOINT_AUDIT["total_parameters"] - r.EXPECTED_CTRL_C_AUDIT["total_parameters"] == r.JOINT_PARAMETERS
    # the recipe constants are inherited untouched
    assert r.EPOCHS == 240 and r.LR == 1e-3 and r.BATCH_SIZE == 128
    assert r.WEIGHT_DECAY == 1e-5 and r.GRAD_CLIP == 5.0
    assert r.SOUP_EPOCHS == (236, 237, 238, 239, 240)
    assert r.TRAIN_SHUFFLE_OFFSET == 101 and r.COMPONENT_LOSS_WEIGHT == 0.5
    assert r.STEPS_EXPECTED == 15_120
    assert r.JOINT_INIT_SEED == 202610081 and r.BOOT_SEED == 20261008 and r.N_BOOT == 2000


def test_model_class_defaults_mask_and_switch():
    assert r.PairJointResidualFullM.DEFAULT_MASK is r.cm.C6_MASK
    assert r.PairJointResidualFullM.joint_enabled is True
    # the round class descends from the previous round's class (shared code,
    # never a monkey patch)
    from tracks.ksvd.experiments.luyin16 import zinc_cssd_relation_projection_v1 as proj

    assert issubclass(r.PairJointResidualFullM, proj.RelationProjectionFullM)


# ---- single-contrast group-paired bootstrap ---------------------------------


def _errs(n: int, shift: float, seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    # identical base draws for both arms: the paired delta is exactly the shift
    return np.abs(rng.normal(0.1, 0.02, n)) + shift


def test_bootstrap_contrast_shared_picks_and_metrics():
    n, n_groups = 60, 12
    group_of_row = np.repeat(np.arange(n_groups), n // n_groups)
    base_y, base_g = _errs(n, 0.0, 1), _errs(n, 0.0, 11)
    errs_y = {
        f"{arm}_s0": base_y + (0.004 if arm == "JOINT_RESIDUAL" else 0.0)
        for arm in r.ARMS
    }
    errs_g = {
        f"{arm}_s0": base_g + (0.003 if arm == "JOINT_RESIDUAL" else 0.0)
        for arm in r.ARMS
    }
    boot = r.paired_bootstrap_contrast(errs_y, errs_g, group_of_row, cand="JOINT_RESIDUAL", ctrl="CTRL_C")
    assert boot["n_rows"] == n and boot["n_groups"] == n_groups
    assert boot["contrast"] == "JOINT_RESIDUAL|CTRL_C"
    assert boot["shared_picks_across_arms_and_metrics"] is True
    assert boot["delta_y_mean"] == pytest.approx(0.004, abs=1e-12)
    assert boot["delta_g_mean"] == pytest.approx(0.003, abs=1e-12)
    for tag in ("y", "g"):
        lo, hi = boot[f"delta_{tag}_ci95"]
        assert lo == pytest.approx(boot[f"delta_{tag}_mean"], abs=1e-12)
        assert hi == pytest.approx(boot[f"delta_{tag}_mean"], abs=1e-12)
    # deterministic under the frozen seed
    boot2 = r.paired_bootstrap_contrast(errs_y, errs_g, group_of_row, cand="JOINT_RESIDUAL", ctrl="CTRL_C")
    assert boot["delta_y_ci95"] == boot2["delta_y_ci95"]
    assert boot["delta_g_ci95"] == boot2["delta_g_ci95"]


def test_bootstrap_row_misalignment_raises():
    n = 12
    group_of_row = np.arange(n) % 3
    errs = {f"{arm}_s0": np.ones(n) for arm in r.ARMS}
    with pytest.raises(RuntimeError, match="row misalignment"):
        r.paired_bootstrap_contrast(errs, errs, group_of_row[:-1], cand="JOINT_RESIDUAL", ctrl="CTRL_C")


# ---- frozen exploratory reading rubric --------------------------------------


def _reading(dy: float, dg: float, *, ci_y=(-0.01, 0.001), ci_g=(-0.01, 0.001),
             alpha_ok: bool = True, toggle_ok: bool = True, marker: float = 1e-4):
    deltas = {"JOINT-CTRL_C_y": dy, "JOINT-CTRL_C_g": dg}
    bootstrap = {
        "delta_y_ci95": list(ci_y), "delta_g_ci95": list(ci_g),
        "delta_y_mean": dy, "delta_g_mean": dg,
    }
    alpha_rows = {
        arm: {"abs_dpred": {"p95": 1.0 if (alpha_ok or arm == "CTRL_C") else marker * 0.5}}
        for arm in r.ARMS
    }
    toggle_row = {
        "abs_dpred": {"p95": 0.5 if toggle_ok else marker * 0.5},
        "disable_improves_y": False, "disable_improves_g": False,
    }
    return r.exploratory_reading(deltas, bootstrap, alpha_rows, toggle_row, marker)


def test_reading_branch_lead():
    out = _reading(-0.003, -0.002, ci_y=(-0.005, -0.001), ci_g=(-0.004, -0.0005))
    assert out["branch"] == "joint-residual-single-seed-lead"
    assert out["single_seed_caveat"]


def test_reading_branch_lead_weak_when_ci_crosses_zero():
    out = _reading(-0.002, -0.001, ci_y=(-0.005, 0.0009), ci_g=(-0.004, -0.0005))
    assert out["branch"] == "joint-residual-single-seed-lead-weak-uncertain"
    assert out["ci_y_contains_zero"] is True


def test_reading_branch_lead_without_branch_response_is_not_upgraded():
    out = _reading(-0.003, -0.002, ci_y=(-0.005, -0.001), ci_g=(-0.004, -0.0005), toggle_ok=False)
    assert out["branch"] == "performance-lead-without-branch-response"


def test_reading_branch_dictionary_readout_lost():
    out = _reading(-0.003, -0.002, alpha_ok=False)
    assert out["branch"] == "dictionary-readout-lost-no-dictionary-mainline-upgrade"


def test_reading_branch_mixed_and_not_retained():
    mixed = _reading(-0.004, 0.002)
    assert mixed["branch"] == "mixed-inconclusive"
    below = _reading(5e-5, 5e-5, ci_y=(-0.001, 0.001), ci_g=(-0.001, 0.001))
    assert below["branch"] == "mixed-inconclusive"
    worse = _reading(0.002, 0.003)
    assert worse["branch"] == "joint-residual-configuration-not-retained"

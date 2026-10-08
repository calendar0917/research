"""Focused wiring tests for zinc_cssd_relation_projection_v1.

Only the NEW risks of this round: the two endpoint-projection modules
(shapes, no biases, parameter counts, the fixed function-preserving C
identity init, R's bitwise P0 copies, fork_rng RNG neutrality), the
three-contrast group-paired bootstrap (shared picks across arms AND metrics),
the frozen exploratory-reading branches and the fixed route-shuffle plan
(per-molecule usage counts preserved).  No data, no GPU, no old-artifact
dependency.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import (
    zinc_cssd_relation_projection_v1 as r,
)

torch.set_num_threads(4)


# ---- projection module contracts -------------------------------------------

def _p0() -> torch.Tensor:
    g = torch.Generator().manual_seed(3)
    return torch.randn(r.PROJECTION_OUT, r.PROJECTION_IN, generator=g)


def test_shared_endpoint_mlp_contract():
    w0 = _p0()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(r.PROJ_INIT_SEED)
        mlp = r.SharedEndpointMLP(w0)
    assert int(sum(p.numel() for p in mlp.parameters())) == r.SHARED_MLP_PROJECTION_PARAMETERS == 34560
    assert mlp.w1.bias is None and mlp.w2.bias is None
    assert tuple(mlp.w1.weight.shape) == (180, 144)
    assert tuple(mlp.w2.weight.shape) == (48, 180)
    assert torch.equal(mlp.w1.weight[:48], w0)
    assert torch.equal(mlp.w1.weight[48:96], -w0)
    ident = torch.eye(48)
    assert torch.equal(mlp.w2.weight[:, :48], ident)
    assert torch.equal(mlp.w2.weight[:, 48:96], -ident)
    assert torch.equal(mlp.w2.weight[:, 96:], torch.zeros(48, 84))
    # the supplementary rows keep a live default draw (they must be trainable)
    assert float(mlp.w1.weight[96:].abs().sum()) > 0.0


def test_shared_endpoint_mlp_initial_function_is_p0():
    w0 = _p0()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(r.PROJ_INIT_SEED)
        mlp = r.SharedEndpointMLP(w0)
    x = torch.randn(512, 144, generator=torch.Generator().manual_seed(11)) * 3.0
    with torch.no_grad():
        got = mlp(x)
        want = F.linear(x, w0)
    assert (got - want).abs().max() < 1e-4


def test_bucket_endpoint_projection_contract():
    w0 = _p0()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(r.PROJ_INIT_SEED)
        proj = r.BucketEndpointProjection(w0)
    assert int(sum(p.numel() for p in proj.parameters())) == r.BUCKET_PROJECTION_PARAMETERS == 34560
    assert proj.n_buckets == 5
    assert all(layer.bias is None for layer in proj.layers)
    assert all(tuple(layer.weight.shape) == (48, 144) for layer in proj.layers)
    assert all(torch.equal(layer.weight, w0) for layer in proj.layers)


def test_projection_build_is_rng_neutral_and_deterministic():
    torch.manual_seed(4242)
    sentinel = torch.get_rng_state().clone()
    w0 = _p0()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(r.PROJ_INIT_SEED)
        a = r.SharedEndpointMLP(w0)
    assert torch.equal(torch.get_rng_state(), sentinel)  # fork restored the stream
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(r.PROJ_INIT_SEED)
        b = r.SharedEndpointMLP(w0)
        c = r.BucketEndpointProjection(w0)
        d = r.BucketEndpointProjection(w0)
    assert all(torch.equal(pa, pb) for pa, pb in zip(a.parameters(), b.parameters()))
    assert all(torch.equal(pa, pb) for pa, pb in zip(c.parameters(), d.parameters()))
    # C and R genuinely differ as modules
    assert r.state_hash({f"w{i}": t for i, t in enumerate(a.state_dict().values())}) != r.state_hash(
        {f"w{i}": t for i, t in enumerate(c.state_dict().values())})


def test_parameter_contracts_are_distinct():
    base = r.arm_expected_audit("BASE")
    cand = r.arm_expected_audit("SHARED_MLP")
    assert base["total_parameters"] == 297_539
    assert cand["total_parameters"] == 325_187 == r.arm_expected_audit("BUCKET_LINEAR")["total_parameters"]
    assert cand["base_body_parameters"] - base["base_body_parameters"] == r.DELTA_PROJECTION == 27_648
    assert base["bridge_parameters"] == cand["bridge_parameters"] == 82_944
    assert base["local_tuple_parameters"] == cand["local_tuple_parameters"] == 29_888


# ---- route plan + override assembly ----------------------------------------

class _FakeRow:
    def __init__(self, buckets: list[int]) -> None:
        self.pair_bucket = torch.tensor(buckets, dtype=torch.long)


def test_route_plan_preserves_per_molecule_bucket_usage():
    rows = [
        _FakeRow([0, 1, 2, 3, 4, 4, 0, 0]),
        _FakeRow([2, 2]),
        _FakeRow([1, 1, 1, 4, 4, 4, 4, 4, 2, 3]),
    ]
    gid = np.asarray([10, 10, 11], np.int64)
    plan = r.build_route_plan(rows, gid)
    assert plan["seed"] == r.ROUTE_SHUFFLE_SEED == 20261008
    assert plan["n_pairs_total"] == 20
    for row, start in zip(rows, (0, 8, 10)):
        n = int(row.pair_bucket.numel())
        natural = plan["natural"][start : start + n]
        permuted = plan["permuted"][start : start + n]
        assert np.array_equal(natural, row.pair_bucket.numpy())
        assert sorted(permuted.tolist()) == sorted(natural.tolist())  # usage counts preserved
    # deterministic regeneration
    plan2 = r.build_route_plan(rows, gid)
    assert np.array_equal(plan["permuted"], plan2["permuted"])


def test_route_override_chunk_matches_flattened_plan():
    rows = [_FakeRow([0, 1, 2]), _FakeRow([4, 4, 4, 3])]
    gid = np.asarray([7, 8], np.int64)
    plan = r.build_route_plan(rows, gid)
    chunk = r._route_override_for_chunk(plan, [0, 1])
    assert chunk.shape[0] == 7
    assert int(chunk.max()) <= 4 and int(chunk.min()) >= 0


# ---- bootstrap -------------------------------------------------------------

def test_bootstrap_three_contrasts_shared_picks_and_metrics():
    rng = np.random.default_rng(5)
    n = 60
    groups = np.asarray(rng.integers(0, 7, n), object)
    base_y = rng.normal(0.1, 0.02, n)
    base_g = rng.normal(0.08, 0.02, n)
    errs_y = {
        "BASE_s0": np.abs(base_y),
        "SHARED_MLP_s0": np.abs(base_y - 0.003),
        "BUCKET_LINEAR_s0": np.abs(base_y - 0.005),
    }
    errs_g = {
        "BASE_s0": np.abs(base_g),
        "SHARED_MLP_s0": np.abs(base_g - 0.002),
        "BUCKET_LINEAR_s0": np.abs(base_g - 0.004),
    }
    out = r.paired_bootstrap_three_contrasts(errs_y, errs_g, groups, n_boot=200, seed=12345)
    assert set(out["contrasts"]) == {"R-BASE", "C-BASE", "R-C"}
    for c in out["contrasts"].values():
        assert set(c) == {"delta_y_ci95", "delta_y_mean", "delta_g_ci95", "delta_g_mean"}
        assert c["delta_y_ci95"][0] <= c["delta_y_mean"] <= c["delta_y_ci95"][1]
    # the point estimates order as the constructed effects, and R-C = R-BASE - C-BASE
    assert out["contrasts"]["R-BASE"]["delta_y_mean"] < out["contrasts"]["C-BASE"]["delta_y_mean"] < 0
    assert abs(
        out["contrasts"]["R-C"]["delta_y_mean"]
        - (out["contrasts"]["R-BASE"]["delta_y_mean"] - out["contrasts"]["C-BASE"]["delta_y_mean"])
    ) < 1e-9
    # misaligned rows are refused
    with pytest.raises(RuntimeError):
        r.paired_bootstrap_three_contrasts(errs_y, errs_g, groups[:-1], n_boot=10)


# ---- exploratory reading branches ------------------------------------------

def _reading(deltas, *, r_alpha_p95=1.0, p_div=0.05, route_p95=1.0, changed=0.5, marker=1e-4, ci_half=0.001):
    bootstrap = {
        "contrasts": {
            label: {
                "delta_y_ci95": [deltas[f"{label}_y"] - ci_half, deltas[f"{label}_y"] + ci_half],
                "delta_y_mean": deltas[f"{label}_y"],
                "delta_g_ci95": [deltas[f"{label}_g"] - ci_half, deltas[f"{label}_g"] + ci_half],
                "delta_g_mean": deltas[f"{label}_g"],
            }
            for label in ("R-BASE", "C-BASE", "R-C")
        }
    }
    alpha_rows = {
        arm: {"abs_dpred": {"p95": r_alpha_p95 if arm == "BUCKET_LINEAR" else 1.0}} for arm in r.ARMS
    }
    route_rows = {
        arm: {
            "abs_dpred": {"p95": route_p95 if arm == "BUCKET_LINEAR" else 0.0},
            "plan": {"fraction_pairs_changed": changed},
            "p_divergence": {"mean_rel_dev_from_mean": p_div if arm == "BUCKET_LINEAR" else None},
        }
        for arm in r.ARMS
    }
    return r.exploratory_reading(deltas, bootstrap, alpha_rows, route_rows, marker)


def test_reading_branch_1_relational_signal():
    deltas = {"R-BASE_y": -0.004, "R-BASE_g": -0.003, "C-BASE_y": -0.001, "C-BASE_g": -0.001,
              "R-C_y": -0.003, "R-C_g": -0.002}
    out = _reading(deltas)
    assert out["branch"] == "relational-conditional-projection-directional-signal"
    assert out["r_ci_crossings"]["R-C_y_ci_contains_zero"] is False
    # the same point estimates with a wide R-C CI must record the crossing
    out_wide = _reading(deltas, ci_half=0.01)
    assert out_wide["branch"] == "relational-conditional-projection-directional-signal"
    assert out_wide["r_ci_crossings"]["R-C_y_ci_contains_zero"] is True
    assert out_wide["r_ci_crossings"]["R-C_g_ci_contains_zero"] is True


def test_reading_branch_2_shared_projection_improves_without_split():
    deltas = {"R-BASE_y": -0.002, "R-BASE_g": -0.001, "C-BASE_y": -0.0025, "C-BASE_g": -0.002,
              "R-C_y": 0.0005, "R-C_g": 0.001}
    out = _reading(deltas)
    assert out["branch"] == "stronger-endpoint-projection-directional-signal"


def test_reading_branch_3_shared_mlp_only():
    deltas = {"R-BASE_y": 0.001, "R-BASE_g": 0.002, "C-BASE_y": -0.002, "C-BASE_g": -0.001,
              "R-C_y": 0.003, "R-C_g": 0.003}
    out = _reading(deltas)
    assert out["branch"] == "shared-nonlinear-projection-directional-signal"


def test_reading_branch_mixed_and_no_signal():
    mixed = {"R-BASE_y": -0.002, "R-BASE_g": 0.002, "C-BASE_y": 0.0, "C-BASE_g": 0.0,
             "R-C_y": -0.002, "R-C_g": -0.002}
    assert _reading(mixed)["branch"] == "mixed-signals"
    none = {"R-BASE_y": 0.001, "R-BASE_g": 0.001, "C-BASE_y": 0.002, "C-BASE_g": -0.002,
            "R-C_y": -0.001, "R-C_g": 0.003}
    out = _reading(none)
    assert out["branch"] in {"no-clear-signal", "mixed-signals"}


def test_reading_route_viewpoints_not_learned_overrides_performance():
    deltas = {"R-BASE_y": -0.004, "R-BASE_g": -0.003, "C-BASE_y": -0.001, "C-BASE_g": -0.001,
              "R-C_y": -0.003, "R-C_g": -0.002}
    # performance looks like branch 1 but the P_b never diverged: no upgrade
    out = _reading(deltas, p_div=0.0)
    assert out["branch"] == "route-viewpoints-not-learned"


def test_reading_dictionary_readout_not_supported():
    deltas = {"R-BASE_y": -0.004, "R-BASE_g": -0.003, "C-BASE_y": -0.001, "C-BASE_g": -0.001,
              "R-C_y": -0.003, "R-C_g": -0.002}
    out = _reading(deltas, r_alpha_p95=1e-5, marker=1e-4)
    assert out["branch"] == "dictionary-readout-not-supported"

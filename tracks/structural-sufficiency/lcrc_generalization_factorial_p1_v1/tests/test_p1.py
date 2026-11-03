"""Correctness tests for the LCRC-P1 metrics, bootstrap and unit registry.

These tests are CPU-only and use no GPU, no network and no historical results:
they check arithmetic that can be verified by hand, the frozen constants, and
the fact that the training driver this round calls is byte-identical to the
frozen one.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

import p1_analysis as A
import p1_common as C
import p1_stats as S
import p1_sources as SRC


# ---------------------------------------------------------------------------
# point metrics
# ---------------------------------------------------------------------------


def test_metrics_match_hand_computation():
    y = np.array([1.0, 2.0, 3.0, 4.0])
    pred = np.array([1.5, 1.0, 3.0, 6.0])
    err = np.array([0.5, -1.0, 0.0, 2.0])
    out = S.metrics(y, pred)
    assert out["n"] == 4
    assert out["sse"] == pytest.approx(0.25 + 1.0 + 0.0 + 4.0)
    assert out["rmse"] == pytest.approx(math.sqrt(5.25 / 4.0))
    assert out["mae"] == pytest.approx(3.5 / 4.0)
    assert out["bias"] == pytest.approx(err.mean())
    assert out["p90_ae"] == pytest.approx(float(np.percentile(np.abs(err), 90.0)))
    # worst ceil(0.1*4) = 1 molecule
    assert out["cvar90_ae"] == pytest.approx(2.0)


def test_absolute_and_squared_error_signs():
    y = np.array([0.0, 0.0])
    pred = np.array([1.0, -2.0])
    assert np.allclose(S.absolute_error(y, pred), [1.0, 2.0])
    assert np.allclose(S.squared_error(y, pred), [1.0, 4.0])


def test_shape_mismatch_is_rejected():
    with pytest.raises(ValueError):
        S.squared_error(np.zeros(3), np.zeros(4))


@pytest.mark.parametrize("n,expected", [(420, 42), (10, 1), (1, 1), (41, 5), (0, 0)])
def test_cvar_tail_size_is_ceil_ten_percent(n, expected):
    assert S.tail_count(n) == expected


def test_cvar_uses_the_models_own_worst_molecules():
    ae = np.array([0.1, 0.2, 0.3, 10.0])
    # ceil(0.1*4) = 1 -> only the 10.0 molecule
    assert S.cvar90_ae(ae) == pytest.approx(10.0)
    ae2 = np.array([0.1, 0.2, 0.3, 10.0] * 25)  # n = 100 -> k = 10
    assert S.cvar90_ae(ae2) == pytest.approx(10.0)


def test_top_abs_indices_are_descending_and_deterministic():
    ae = np.array([1.0, 5.0, 5.0, 0.0])
    assert list(S.top_abs_indices(ae, 2)) == [1, 2]
    assert list(S.top_abs_indices(ae, 2)) == list(S.top_abs_indices(ae, 2))


# ---------------------------------------------------------------------------
# strict alignment
# ---------------------------------------------------------------------------


def test_strict_align_requires_frozen_row_order():
    expected = np.array([3, 1, 2])
    got = SRC.strict_align(expected, np.array([[0.5], [0.25], [0.75]]), expected, label="t")
    assert np.allclose(got, [0.5, 0.25, 0.75])
    with pytest.raises(SRC.AlignmentError):
        SRC.strict_align(np.array([1, 3, 2]), np.zeros((3, 1)), expected, label="t")
    with pytest.raises(SRC.AlignmentError):
        SRC.strict_align(np.array([3, 1, 1]), np.zeros((3, 1)), expected, label="t")


def test_destandardisation_is_the_inverse_of_standardisation():
    y = np.array([-1.0, 0.0, 2.5])
    norm = {"mean": 2.5, "scale": 1.75}
    raw = (y - norm["mean"]) / norm["scale"]
    back = SRC._destandardise(raw, norm)
    assert np.allclose(back, y, atol=1e-12)


# ---------------------------------------------------------------------------
# cluster bootstrap
# ---------------------------------------------------------------------------


def test_cluster_index_resamples_whole_clusters():
    idx = S.ClusterIndex.from_values(["a", "a", "b", "c"])
    profile = idx.profile()
    assert profile["n_clusters"] == 3
    assert profile["max_cluster_size"] == 2
    rng = np.random.default_rng(0)
    positions = idx.sample_positions(rng)
    counts = np.bincount(positions, minlength=4)
    # the two molecules of cluster "a" are always drawn together
    assert counts[0] == counts[1]
    assert len(positions) >= 3  # at least one molecule per drawn cluster


def test_cluster_bootstrap_is_deterministic_for_a_fixed_seed():
    idx = S.ClusterIndex.from_values(list("aabbccdd"))
    se_a = np.array([1.0, 2.0, 3.0, 4.0, 1.0, 1.0, 9.0, 9.0])
    se_b = np.array([1.0, 1.0, 1.0, 1.0, 1.0, 1.0, 9.0, 9.0])
    compute = lambda pos: {"delta_rmse": S.rmse_from_se(se_b[pos]) - S.rmse_from_se(se_a[pos])}
    first = S.cluster_bootstrap(idx, compute, n_boot=64, seed=7)
    second = S.cluster_bootstrap(idx, compute, n_boot=64, seed=7)
    assert np.array_equal(first["delta_rmse"], second["delta_rmse"])
    third = S.cluster_bootstrap(idx, compute, n_boot=64, seed=8)
    assert not np.array_equal(first["delta_rmse"], third["delta_rmse"])


def test_interval_summary_reports_percentiles_and_sign_fractions():
    draws = {"x": np.array([-2.0, -1.0, 1.0, 2.0])}
    summary = S.interval_summary({"x": 0.5}, draws)["x"]
    assert summary["point"] == 0.5
    assert summary["frac_gt_zero"] == pytest.approx(0.5)
    assert summary["frac_lt_zero"] == pytest.approx(0.5)
    assert summary["ci_low"] == pytest.approx(-1.925)
    assert summary["ci_high"] == pytest.approx(1.925)


# ---------------------------------------------------------------------------
# quantile bins (Train-fitted, applied unchanged to Valid/Test)
# ---------------------------------------------------------------------------


def test_quantile_edges_and_binning_are_monotone_and_closed_low():
    values = list(range(1, 101))
    edges = S.quantile_edges(values, 4)
    assert len(edges) == 3
    assert edges == pytest.approx([25.75, 50.5, 75.25])
    idx = S.bin_index([0.0, 25.75, 50.5, 75.25, 1000.0, float("nan")], edges)
    assert list(idx) == [0, 0, 1, 2, 3, -1]


def test_quantile_edges_ignore_non_finite_values():
    assert S.quantile_edges([1.0, 2.0, float("nan"), 3.0, 4.0], 2) == pytest.approx([2.5])


# ---------------------------------------------------------------------------
# interaction and paired arithmetic
# ---------------------------------------------------------------------------


def test_interaction_sign_convention():
    mse = {"base": 10.0, "chem8": 9.0, "multicontrib": 9.0, "combo": 8.0}
    interaction = mse["combo"] - mse["chem8"] - mse["multicontrib"] + mse["base"]
    assert interaction == pytest.approx(0.0)
    # a purely additive world gives exactly zero interaction
    mse_add = {"base": 10.0, "chem8": 9.0, "multicontrib": 9.0, "combo": 8.0}
    assert mse_add["combo"] == mse_add["base"] + (mse_add["chem8"] - mse_add["base"]) + (
        mse_add["multicontrib"] - mse_add["base"]
    )
    # synergistic combination -> negative
    assert (7.0 - 9.0 - 9.0 + 10.0) < 0
    # antagonistic combination -> positive
    assert (10.5 - 9.0 - 9.0 + 10.0) > 0


def test_combo_base_point_negative_delta_means_combo_better():
    y = np.zeros(4)
    pred_base = np.array([1.0, 1.0, 1.0, 1.0])
    pred_combo = np.array([0.5, 0.5, 0.5, 0.5])
    ae_b, se_b = S.absolute_error(y, pred_base), S.squared_error(y, pred_base)
    ae_c, se_c = S.absolute_error(y, pred_combo), S.squared_error(y, pred_combo)
    point = A.combo_base_point(ae_b, se_b, ae_c, se_c)
    assert point["delta_rmse"] < 0
    assert point["delta_mae"] < 0
    assert point["frac_improved"] == 1.0


def test_seed_mean_bootstrap_averages_the_same_resample_over_seeds():
    idx = S.ClusterIndex.from_values(["a", "a", "b", "b"])
    per_seed = {
        0: (np.array([1.0, 1.0, 2.0, 2.0]), np.array([1.0, 1.0, 4.0, 4.0])),
        1: (np.array([2.0, 2.0, 2.0, 2.0]), np.array([4.0, 4.0, 4.0, 4.0])),
    }

    def compute(pos):
        values = [
            S.rmse_from_se(se[pos]) - S.rmse_from_se(np.ones(4)[pos]) for _, se in per_seed.values()
        ]
        return {"delta_rmse": float(np.mean(values))}

    point = compute(np.arange(4))
    draws = S.cluster_bootstrap(idx, compute, n_boot=32, seed=3)
    assert draws["delta_rmse"].shape == (32,)
    assert np.isfinite(point["delta_rmse"])


# ---------------------------------------------------------------------------
# frozen constants and unit registry
# ---------------------------------------------------------------------------


def test_frozen_constants():
    assert C.N_BOOT == 2000
    assert C.BOOTSTRAP_SEED == 20261104
    assert C.TOP_FRACTION == pytest.approx(0.10)
    assert C.TOL == pytest.approx(1e-5)
    assert C.SPLIT_ORDER == ("random122", "random123", "random124")
    assert C.ARMS == ("base", "chem8", "multicontrib", "combo_chem_contrib")
    assert C.Q2_ARMS == ("base", "combo_chem_contrib")
    assert C.SHUFFLE_OFFSET == 917000


def test_seed_design_is_exactly_as_pre_registered():
    assert C.SEEDS == {"random122": (0, 1, 2), "random123": (0, 1), "random124": (0, 1)}


def test_unit_registry_is_complete_and_unique():
    units = C.unit_registry()
    assert len(units) == 20
    assert len({u["key"] for u in units}) == 20
    assert sum(1 for u in units if u["origin"] == "reused") == 10
    assert sum(1 for u in units if u["origin"] == "fresh") == 10
    have = {(u["split"], u["arm"], u["seed"]) for u in units}
    for split, seeds in C.SEEDS.items():
        arms = C.ARMS if split == "random122" else C.Q2_ARMS
        for arm in arms:
            for seed in seeds:
                assert (split, arm, seed) in have, (split, arm, seed)
    # exactly the four missing CHEM8/MULTICONTRIB cells plus the six BASE/COMBO cells
    fresh = sorted((u["split"], u["arm"], u["seed"]) for u in units if u["origin"] == "fresh")
    assert fresh == sorted(
        [
            ("random122", "chem8", 1),
            ("random122", "chem8", 2),
            ("random122", "multicontrib", 1),
            ("random122", "multicontrib", 2),
            ("random123", "base", 0),
            ("random123", "base", 1),
            ("random123", "combo_chem_contrib", 1),
            ("random124", "base", 0),
            ("random124", "base", 1),
            ("random124", "combo_chem_contrib", 1),
        ]
    )


def test_reused_units_point_at_frozen_longrun_directories():
    for unit in C.unit_registry():
        if unit["origin"] == "reused":
            assert unit["run_dir"].is_dir(), unit["run_dir"]
            assert (unit["run_dir"] / "train.json").is_file()
        else:
            assert unit["run_dir"].parent == C.RUNS


# ---------------------------------------------------------------------------
# frozen training driver identity
# ---------------------------------------------------------------------------


def test_expected_driver_hashes_match_the_round_record():
    import run_unit

    actual = __import__("cl_common").code_hashes()
    assert set(run_unit.EXPECTED_CODE_HASHES) <= set(actual)
    for name, digest in run_unit.EXPECTED_CODE_HASHES.items():
        assert actual[name] == digest, name


def test_frozen_driver_still_passes_its_protocol_freeze():
    import run_unit

    payload = run_unit.verify_frozen_driver()
    assert payload["protocol_freeze"]["code_hashes_match"] is True


def test_frozen_recipe_knobs_are_present_in_the_longrun_record():
    recipe = C.read_json(C.LONGRUN_RUNS / "p0_base_lipo_parity" / "train.json")["recipe"]
    for key in C.FROZEN_RECIPE_KEYS:
        assert key in recipe, key
    assert recipe["epochs_planned"] == 360
    assert recipe["peak_lr"] == pytest.approx(3e-4)
    assert recipe["min_lr"] == pytest.approx(1e-5)
    assert recipe["weight_decay"] == pytest.approx(1e-3)
    assert recipe["batch_size"] == 128
    assert recipe["dropout"] == pytest.approx(0.10)
    assert recipe["gradient_clip"] == pytest.approx(5.0)
    assert recipe["optimizer"] == "AdamW"
    assert recipe["checkpoint_rule"] == "lowest_validation_rmse__strict_improvement__tie_keeps_earlier"
    assert recipe["test_used_for_any_selection"] is False

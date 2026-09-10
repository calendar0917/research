"""Unit tests for the fast baseline-difficulty / typed-refinement audit helpers.

The audit is a read-only diagnostic over frozen run artifacts; these tests pin the
pure, deterministic pieces (child-context aggregation, exact centre/spread
decomposition, quintile assignment, movement classification) so the mechanism
tables cannot drift silently.
"""

from __future__ import annotations

import numpy as np

from tracks.ksvd.experiments.luyin16.zinc_baseline_difficulty_typed_refinement_audit import (
    _weighted_std,
    center_spread_decomposition,
    child_statistics,
    difficulty_quintiles,
    mechanism_2x2,
    movement_classes,
    parent_informativeness,
    quintile_assignment,
)


def test_weighted_std_matches_manual():
    values = np.array([0.0, 2.0])
    weights = np.array([1.0, 1.0])
    assert abs(_weighted_std(values, weights) - 1.0) < 1e-12


def test_child_statistics_molecule_level_support():
    occ = {
        "train_mol_ids": np.array([0, 0, 1, 2, 2, 2], dtype=np.int64),
        "train_corr_r2_id": np.array([10, 10, 11, 10, 11, 11], dtype=np.int64),
    }
    ctx = {"target": np.array([1.0, 2.0, 3.0]), "resid": np.array([0.1, 0.2, 0.3])}
    stats = child_statistics(occ, ctx)
    # child 10 occurs in molecules {0, 2}; child 11 in {1, 2}
    assert stats[10]["support"] == 2
    assert stats[11]["support"] == 2
    assert abs(stats[10]["mean_target"] - 2.0) < 1e-12  # mean(1, 3)
    assert abs(stats[11]["mean_target"] - 2.5) < 1e-12  # mean(2, 3)


def test_parent_informativeness_eligibility_gate():
    split = {
        0: {"corrected_child_ids": [0, 1], "split_multiplicity": 2, "historical_count": 10},
        1: {"corrected_child_ids": [2], "split_multiplicity": 1, "historical_count": 5},
    }
    child_stats = {
        0: {"support": 8, "mean_target": -2.0, "mean_resid": -1.0},
        1: {"support": 8, "mean_target": 2.0, "mean_resid": 1.0},
        2: {"support": 20, "mean_target": 0.0, "mean_resid": 0.0},
    }
    parents = parent_informativeness(split, child_stats)
    assert parents[0]["eligible"] is True
    assert abs(parents[0]["target_dispersion"] - 2.0) < 1e-12
    assert abs(parents[0]["resid_dispersion"] - 1.0) < 1e-12
    assert parents[1]["eligible"] is False  # only one child


def test_parent_informativeness_needs_two_supported_children():
    split = {7: {"corrected_child_ids": [0, 1], "split_multiplicity": 2, "historical_count": 9}}
    child_stats = {
        0: {"support": 40, "mean_target": -5.0, "mean_resid": -2.0},
        1: {"support": 2, "mean_target": 5.0, "mean_resid": 2.0},  # below MIN_CHILD_SUPPORT
    }
    parents = parent_informativeness(split, child_stats)
    assert parents[7]["eligible"] is False


def test_center_spread_decomposition_is_exact():
    pred_hist = np.array([[1.0, 3.0], [3.0, 3.0]])
    pred_corr = np.array([[1.0, 3.0], [1.0, 3.0]])
    targets = np.array([2.0, 3.0])
    d = center_spread_decomposition(pred_hist, pred_corr, targets)
    # historical: ens=(2,3): center 0, spread var(1,3)=1
    assert np.allclose(d["center_hist"], [0.0, 0.0])
    assert np.allclose(d["spread_hist"], [1.0, 0.0])
    # corrected: ens=(1,3): center (1)^2=1, spread 0
    assert np.allclose(d["center_corr"], [1.0, 0.0])
    assert np.allclose(d["spread_corr"], [0.0, 0.0])
    # L1: center_mae change = 1 - 0 = 1
    assert np.allclose(d["delta_center_mae"], [1.0, 0.0])


def test_quintile_assignment_and_table():
    metric = np.arange(100, dtype=np.float64)
    bins, edges = quintile_assignment(metric, n_bins=5)
    assert set(bins.tolist()) == {0, 1, 2, 3, 4}
    assert int((bins == 0).sum()) == 20
    err_hist = np.tile(metric, (4, 1))
    err_corr = err_hist - 1.0
    degradation = {
        "err_hist": err_hist, "err_corr": err_corr, "delta": err_corr - err_hist,
        "disagreement_hist": np.zeros(100), "disagreement_corr": np.zeros(100),
        "delta_disagreement": np.zeros(100),
    }
    ens = np.zeros(100)
    rows = difficulty_quintiles(bins, degradation, ens, ens)
    assert [r["n"] for r in rows] == [20, 20, 20, 20, 20]
    assert all(abs(r["degradation"] + 1.0) < 1e-12 for r in rows)


def test_movement_classes_classify_direction():
    # one molecule, one seed: historical err +0.1, corrected crosses to -0.2 (overshoot harmful)
    pred_hist = np.array([[1.1]])
    pred_corr = np.array([[0.8]])
    targets = np.array([1.0])
    rows = movement_classes(np.array([0]), pred_hist, pred_corr, targets, n_bins=1)
    row = rows[0]
    assert row["harmful_pct"] == 100.0
    assert row["overshoot_pct"] == 100.0
    assert row["wrong_direction_pct"] == 0.0
    # same-side harmful move: historical 1.1 (err +0.1), corrected 1.3 (err +0.3)
    rows2 = movement_classes(np.array([0]), np.array([[1.1]]), np.array([[1.3]]), targets, n_bins=1)
    assert rows2[0]["wrong_direction_pct"] == 100.0


def test_mechanism_2x2_shapes_and_interaction():
    bins = np.array([0, 0, 1, 1, 3, 3, 4, 4])
    info = np.array([0.0, 1.0, 0.0, 1.0, 0.0, 1.0, 0.0, 1.0])
    err_hist = np.zeros((4, 8))
    err_corr = np.full((4, 8), 0.01)
    degradation = {
        "err_hist": err_hist, "err_corr": err_corr, "delta": err_corr - err_hist,
        "delta_disagreement": np.zeros(8),
    }
    cells = mechanism_2x2(bins, 0.5, info, degradation)
    assert len(cells) == 4
    assert {c["difficulty"] for c in cells} == {"easy", "hard"}
    assert {c["typed_info"] for c in cells} == {"low", "high"}
    assert all(abs(c["gain"] + 0.01) < 1e-12 for c in cells)

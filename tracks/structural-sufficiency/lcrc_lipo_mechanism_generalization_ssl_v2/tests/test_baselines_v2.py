from __future__ import annotations

import numpy as np
import pytest

import v2_baselines as B
import v2_common as C


def test_feature_groups_are_deterministic_and_frozen(fit_cache):
    split = C.load_frozen_split()
    rows = [int(r) for r in split["split"]["fit"][:8]]
    a = B.compute_features(rows)
    b = B.compute_features(rows)
    assert a.rows == rows
    for name in ("P1", "P2", "P3", "physchem13"):
        assert np.array_equal(getattr(a, name), getattr(b, name))
    assert a.P1.shape == (8, 2048)
    assert a.P2.shape == (8, 217)
    assert a.P3.shape == (8, 2048 + 217)
    assert a.physchem13.shape == (8, 13)
    assert a.p2_names == sorted(a.p2_names)
    assert a.audit["physchem13"]["names"] == list(C.DESCRIPTOR_13)
    idx = [a.p2_names.index(name) for name in C.DESCRIPTOR_13]
    assert np.allclose(a.physchem13, a.P2[:, idx])
    assert a.audit["labels_used"] is False
    assert a.audit["P1"]["radius"] == 2
    assert a.audit["P1"]["bits"] == 2048
    assert a.audit["P2"]["source"] == "rdkit.Chem.Descriptors.CalcMolDescriptors"


def test_candidate_registry_counts_and_grid():
    reg = B.candidate_registry()
    ridge = [x for x in reg if x["family"] == "ridge"]
    xgb = [x for x in reg if x["family"] == "xgboost"]
    assert len(ridge) == 6
    assert len(xgb) == 24
    assert {x["feature_group"] for x in ridge} == {"P2", "P3"}
    assert {x["params"]["alpha"] for x in ridge} == {1.0, 10.0, 100.0}
    combos = {(x["params"]["n_estimators"], x["params"]["max_depth"], x["params"]["min_child_weight"]) for x in xgb}
    assert combos == {(400, 3, 1), (400, 3, 5), (400, 5, 1), (400, 5, 5), (800, 3, 1), (800, 3, 5), (800, 5, 1), (800, 5, 5)}
    assert all(x["feature_group"] in {"P1", "P2", "P3"} for x in xgb)


def test_preprocessor_is_fit_on_train_only():
    rng = np.random.default_rng(0)
    X_train = rng.normal(size=(40, 5))
    X_train[:, 1] = 2.0  # constant, must be dropped
    X_train[0, 2] = np.nan
    X_train[7, 2] = np.nan
    X_dev = rng.normal(size=(10, 5)) * 100 + 50
    prep = B.fit_preprocessor(X_train, kind="ridge")
    Z1 = B.transform(prep, X_dev)
    Z2 = B.transform(prep, X_dev)
    assert np.array_equal(Z1, Z2)
    assert not np.isnan(Z1).any()
    assert 1 not in prep.kept.tolist()
    # A changed dev row cannot change the train-derived median/mean.
    med = prep.median.copy()
    mean = prep.mean.copy()
    X_dev2 = X_dev.copy()
    X_dev2[0, 2] = 1e9
    B.transform(prep, X_dev2)
    assert np.array_equal(med, prep.median)
    assert np.array_equal(mean, prep.mean)


def test_xgboost_is_deterministic_on_small_data():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(64, 8))
    y = rng.normal(size=64)
    p1, info1 = B.fit_predict_xgb(X[:48], y[:48], X[48:], {"n_estimators": 20, "max_depth": 3, "min_child_weight": 1})
    p2, info2 = B.fit_predict_xgb(X[:48], y[:48], X[48:], {"n_estimators": 20, "max_depth": 3, "min_child_weight": 1})
    assert np.array_equal(p1, p2)
    assert C.XGB_FIXED["random_state"] == 0


def test_selection_tie_prefers_simpler():
    # Two candidates within 0.01: Ridge P2 is simpler than XGBoost P3.
    results = [
        {"candidate_id": "xgb_P3_n800_d5_w1", "family": "xgboost", "feature_group": "P3", "params": {"n_estimators": 800, "max_depth": 5, "min_child_weight": 1}, "dev_rmse": 0.7000, "dev_mae": 0.5, "fold": 0},
        {"candidate_id": "xgb_P3_n800_d5_w1", "family": "xgboost", "feature_group": "P3", "params": {"n_estimators": 800, "max_depth": 5, "min_child_weight": 1}, "dev_rmse": 0.7000, "dev_mae": 0.5, "fold": 1},
        {"candidate_id": "ridge_P2_a100", "family": "ridge", "feature_group": "P2", "params": {"alpha": 100.0}, "dev_rmse": 0.7050, "dev_mae": 0.5, "fold": 0},
        {"candidate_id": "ridge_P2_a100", "family": "ridge", "feature_group": "P2", "params": {"alpha": 100.0}, "dev_rmse": 0.7050, "dev_mae": 0.5, "fold": 1},
    ]
    sel = B.select_best(results)
    assert sel["best_by_mean_rmse"] == "xgb_P3_n800_d5_w1"
    assert sel["chosen"] == "ridge_P2_a100"

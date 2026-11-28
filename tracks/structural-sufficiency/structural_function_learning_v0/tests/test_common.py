"""Fast invariant tests for the structural-function-learning-v0 code.

These do not need any fitted model: they exercise the linear parameterisation, the
ADD-TREE singleton-constraint semantics, the additive-neural f_j(0)=0 property and
the bootstrap direction on synthetic data.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import scipy.sparse as sp

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "code"))

import sfl_common as K  # noqa: E402


def test_molhiv_identity_shapes():
    d = K.HIVData()
    assert d.n == 41127
    assert d.C.shape == (41127, 754)
    assert d.F.shape == (41127, 3652)
    assert {k: d.rows[k].size for k in K.SPLITS} == K.SPLIT_N
    assert d.n_bonds.min() > 0 and d.n_bonds.max() < d.n_atoms.max() * 3


def test_zinc_identity_shapes():
    d = K.ZincData()
    assert d.n == 12000
    assert d.Z.shape[1] == 4642 and d.C.shape[1] == 9 and d.F.shape[1] == 1250
    assert d.design_dense(np.arange(3)).shape == (3, 5901)


def test_linear_raw_parameterisation_closes_both_baselines():
    sys.path.insert(0, str(K.ISRD / "code"))
    import isrd1_common as I

    rng = np.random.default_rng(0)
    n, p = 800, 25
    X = sp.random(n, p, density=0.3, random_state=1, format="csr", data_rvs=rng.random)
    X = sp.csr_matrix(X)
    y = (rng.random(n) < 0.3).astype(np.float64)
    train = np.arange(600)
    sw = np.zeros(n)
    sw[train] = I.class_balanced_weights(y[train])
    mu = np.asarray(X[train].mean(axis=0)).ravel()
    mu2 = np.asarray(X[train].multiply(X[train]).mean(axis=0)).ravel()
    sd = np.sqrt(np.maximum(mu2 - mu ** 2, 0.0))
    sd = np.where(sd > 0, sd, 1.0)
    fit = I.fit_logistic_sparse_standardised(X, y, sw, 0.03, mu, sd,
                                             max_iter=50, tol=1e-9)
    coef_std = np.asarray(fit["coef"])
    w_raw = coef_std / sd
    b_raw = float(fit["intercept"] - np.sum(coef_std * mu / sd))
    x = X.toarray()
    zero = b_raw + x @ w_raw
    centred = float(fit["intercept"]) + ((x - mu) / sd) @ coef_std
    assert np.max(np.abs(zero - fit["logit"])) < 1e-10
    assert np.max(np.abs(centred - fit["logit"])) < 1e-10
    # the historical double-standardisation bug is detectably different
    buggy = b_raw + (((x - mu) / sd) * w_raw).sum(axis=1)
    assert np.max(np.abs(buggy - fit["logit"])) > 1e-3


def test_add_tree_singleton_constraints_are_honoured():
    import xgboost as xgb

    from phaseB_add_tree import feats_in_tree, feature_names

    rng = np.random.default_rng(0)
    n, p = 3000, 5
    X = rng.normal(size=(n, p)).astype(np.float32)
    y = (X[:, 0] + np.sin(2 * X[:, 1]) - X[:, 2] ** 2 > 0).astype(np.float32)
    names = feature_names(p)
    params = {"objective": "binary:logistic", "max_depth": 4, "eta": 0.3,
              "tree_method": "hist", "seed": 0, "nthread": 2,
              "interaction_constraints": [[nm] for nm in names]}
    b = xgb.train(params, xgb.DMatrix(X, label=y, feature_names=names), 20)
    used = [len(feats_in_tree(t, names)) for t in b.get_dump()]
    assert max(used) == 1
    assert all(u > 0 for u in used)          # constrained model still splits
    pred = b.predict(xgb.DMatrix(X, feature_names=names), output_margin=True)
    assert pred.shape == (n,)


def test_additive_nn_f0_zero_and_forward_closure():
    import torch

    from phaseC_add_nn import AdditiveNN

    rng = np.random.default_rng(2)
    p = 17
    knots = rng.normal(size=(p, 4)).astype(np.float32)
    active = np.ones(p, dtype=bool)
    model = AdditiveNN(knots, active)
    model.eval()
    with torch.no_grad():
        f0 = model.feature_values(torch.zeros(3, p))
        assert float(f0.abs().max()) == 0.0
        x = torch.from_numpy(rng.normal(size=(9, p)).astype(np.float32))
        pred = model(x)
        closure = model.bias + model.feature_values(x).sum(dim=1)
        assert torch.allclose(pred, closure, atol=1e-6)
    # knots are trainable and excluded from weight decay by the training code's groups
    assert model.t.requires_grad and model.a.requires_grad and model.v.requires_grad


def test_bootstrap_direction_matches_point_estimate():
    rng = np.random.default_rng(0)
    n = 400
    y = (rng.random(n) < 0.4).astype(np.int64)
    a = rng.normal(size=n) + y
    b = rng.normal(size=n) + 0.5 * y
    code = np.repeat(np.arange(40), 10)
    from sklearn.metrics import roc_auc_score

    res = K.scaffold_bootstrap(np.arange(n), y, a, b, code, n_boot=50, seed=0)
    expected = roc_auc_score(y, K.sigmoid(a)) - roc_auc_score(y, K.sigmoid(b))
    assert abs(res["delta_auc"] - expected) < 1e-12
    assert res["delta_auc_ci95"][0] <= res["delta_auc"] <= res["delta_auc_ci95"][1]


def test_zinc_bootstrap_direction():
    rng = np.random.default_rng(1)
    y = rng.normal(size=300)
    a = y + rng.normal(scale=0.1, size=300)
    b = y + rng.normal(scale=0.5, size=300)
    res = K.molecule_bootstrap_mae(y, a, b, np.arange(300), n_boot=50, seed=0)
    expected = float(np.mean(np.abs(y - b)) - np.mean(np.abs(y - a)))
    assert abs(res["delta_mae_gain"] - expected) < 1e-12
    assert res["delta_mae_gain"] > 0

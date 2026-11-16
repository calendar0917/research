"""Aggregation algebra: the four arms must be the formulas of the protocol.

The decisive unit test is the oracle of section 6: with ``mu=7``, ``sigma=3``,
``y_query=10``, ``y_ref=4`` and an oracle ``h`` that returns ``2 / -2``,
``delta = 3/2 * (2 - (-2)) = 6`` and ``C = 4 + 6 = 10``.  Adding ``mu`` to the
delta would give 17 and is therefore detectable.
"""

from __future__ import annotations

import numpy as np
import pytest

import rd_common as C
import rd_eval as E


def test_delta_oracle_restores_the_absolute_value_and_mu_is_not_added():
    mu, sigma = 7.0, 3.0
    y_query, y_ref = 10.0, 4.0
    h_forward, h_reverse = 2.0, -2.0  # standardised margins of the oracle h
    delta = E.delta_from_margins(np.array([[h_forward]]), np.array([[h_reverse]]), sigma)
    assert delta[0, 0] == pytest.approx(6.0)
    weights = np.array([[1.0]])
    reference = np.array([[y_ref]])
    pred = E.aggregate_c(weights, reference, delta)[0]
    assert pred == pytest.approx(y_query)
    wrong = pred + mu
    assert wrong == pytest.approx(17.0)
    # the forward branch, de-standardised with mu/sigma, is the *absolute* model
    absolute = mu + sigma * h_forward
    assert absolute == pytest.approx(13.0)


def test_delta_is_antisymmetric_and_zero_for_a_self_reference():
    rng = np.random.default_rng(0)
    h = rng.normal(size=(9, 64))
    sigma = 1.7
    delta_forward = E.delta_from_margins(h, h[::-1], sigma)
    delta_reverse = E.delta_from_margins(h[::-1], h, sigma)
    assert np.allclose(delta_forward, -delta_reverse, rtol=0.0, atol=1e-12)
    # z(G,G) is one and the same row in both directions -> h is identical
    same = np.full((1, 1), 0.913)
    assert E.delta_from_margins(same, same, sigma)[0, 0] == 0.0


def test_c_reduces_to_the_r_formula_when_h_is_a_difference_of_absolute_models():
    """If h(z(G,R)) = g(G) - g(R), the C aggregation is algebraically R."""
    rng = np.random.default_rng(1)
    n_query, width = 13, C.MAX_REFERENCES
    sigma = 2.5
    g_std_query = rng.normal(size=(n_query, 1))
    g_std_ref = rng.normal(size=(n_query, width))
    h_forward = g_std_query - g_std_ref
    h_reverse = g_std_ref - g_std_query
    delta = 0.5 * sigma * (h_forward - h_reverse)
    assert np.allclose(delta, sigma * (g_std_query - g_std_ref), rtol=0.0, atol=1e-12)
    weights = rng.random((n_query, width))
    weights /= weights.sum(axis=1, keepdims=True)
    y_ref = rng.normal(size=(n_query, width))
    mu = 7.0
    d_query = mu + sigma * g_std_query[:, 0]
    d_ref = mu + sigma * g_std_ref
    c_pred = E.aggregate_c(weights, y_ref, delta)
    r_pred = E.aggregate_r(d_query, d_ref, weights, y_ref)
    assert np.allclose(c_pred, r_pred, rtol=0.0, atol=1e-10)


def test_neighbour_aggregation_and_weights_sum_to_one_check():
    y_ref = np.array([[1.0, 3.0]])
    weights = np.array([[0.25, 0.75]])
    assert E.aggregate_n(weights, y_ref)[0] == pytest.approx(2.5)
    leftover = E.aggregate_c(weights, y_ref, np.zeros((1, 2)))
    assert leftover[0] == pytest.approx(2.5), "C with delta=0 must equal N"


def test_weighted_pair_rmse_uses_the_frozen_weights():
    pred = np.array([0.0, 2.0])
    target = np.array([0.0, 0.0])
    weight = np.array([1.0, 3.0])
    expected = np.sqrt((0.0 + 3.0 * 4.0) / 4.0)
    assert E.weighted_rmse(pred, target, weight) == pytest.approx(expected)


def test_leave_one_out_and_bootstrap_are_paired():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    pred_c = np.array([0.1, 1.1, 2.1, 3.1])
    pred_b = np.array([0.1, 1.1, 2.1, 10.0])
    out = E.leave_one_out_delta(pred_c, pred_b, y)
    assert out["dropped_index"] == 3
    assert out["sign_flip"] is True
    boot = E.paired_bootstrap(pred_c, pred_b, y, n_resamples=50, rng=np.random.default_rng(1703))
    assert boot["n_resamples"] == 50
    assert boot["delta"] == pytest.approx(C.rmse(pred_c, y) - C.rmse(pred_b, y))

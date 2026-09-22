"""Targeted correctness tests for the TCCD-GLOBAL62-v0 diagnostic."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tracks.ksvd.code import global62_residual as V

ARTIFACT = V.RESULTS_DIR / "stageA_seed0.json"


# ---------------------------------------------------------------------------
# geometry / parameters
# ---------------------------------------------------------------------------
def test_global_width_and_parameters():
    assert V.GLOBAL_WIDTH == 62
    assert V.GLOBAL_STRUCTURE_SHORT + V.GLOBAL_STRUCTURE_LONG + V.GLOBAL_ATTRIBUTES == 62
    assert V.GLOBAL_ATOM_BINS == 28 and V.GLOBAL_BOND_BINS == 4
    assert V.parameter_accounting() == {
        "encoder_linear1": 2016,
        "encoder_layernorm": 64,
        "encoder_linear2": 1056,
        "residual_linear": 33,
        "total": 3169,
    }
    model = V.global62_head_class()
    names = [n for n, _ in model.named_parameters()]
    assert all(n.startswith("encoder.") or n.startswith("residual.") for n in names)
    shapes = {n: tuple(p.shape) for n, p in model.named_parameters()}
    assert shapes["encoder.0.weight"] == (32, 62)
    assert shapes["encoder.1.weight"] == (32,)
    assert shapes["encoder.3.weight"] == (32, 32)
    assert shapes["residual.weight"] == (1, 32)
    assert sum(p.numel() for p in model.parameters()) == 3169


def test_head_is_b_full_global_encoder_sequence():
    """The hidden transformation is exactly Linear-LayerNorm-ReLU-Linear-ReLU."""
    model = V.global62_head_class()
    seq = model.encoder
    assert isinstance(seq[0], torch.nn.Linear) and seq[0].in_features == 62 and seq[0].out_features == 32
    assert isinstance(seq[1], torch.nn.LayerNorm) and seq[1].normalized_shape == (32,)
    assert isinstance(seq[2], torch.nn.ReLU)
    assert isinstance(seq[3], torch.nn.Linear) and seq[3].in_features == 32 and seq[3].out_features == 32
    assert isinstance(seq[4], torch.nn.ReLU)
    assert isinstance(model.residual, torch.nn.Linear) and model.residual.out_features == 1


# ---------------------------------------------------------------------------
# derangement
# ---------------------------------------------------------------------------
def test_derangement_no_fixed_points_and_multiset():
    for n in (2, 3, 7, 2000):
        perm = V.derangement(n)
        assert perm.shape == (n,)
        assert not np.any(perm == np.arange(n)), n
        assert np.array_equal(np.sort(perm), np.arange(n)), n
    # deterministic for the registered seed
    assert np.array_equal(V.derangement(500), V.derangement(500))
    assert V.SHUFFLE_SEED == 20260923


# ---------------------------------------------------------------------------
# train-only standardizer
# ---------------------------------------------------------------------------
def test_standardizer_fit_on_train_only():
    rng = np.random.default_rng(0)
    raw = rng.standard_normal((100, V.GLOBAL_WIDTH)).astype(np.float32)
    raw[:, 3] = 5.0  # constant column
    train = list(range(70))
    dev = list(range(70, 100))
    std = V.fit_train_only_standardizer(raw, train)
    assert std.constant_columns == [3]
    manual_mean = raw[train].astype(np.float64).mean(axis=0)
    manual_std = raw[train].astype(np.float64).std(axis=0)
    assert np.allclose(std.mean[[i for i in range(V.GLOBAL_WIDTH) if i != 3]], manual_mean[[i for i in range(V.GLOBAL_WIDTH) if i != 3]], atol=1e-5)
    assert np.allclose(std.std[[i for i in range(V.GLOBAL_WIDTH) if i != 3]], manual_std[[i for i in range(V.GLOBAL_WIDTH) if i != 3]], atol=1e-5)
    assert std.std[3] == 1.0
    # dev statistics are irrelevant: changing dev rows must not change the transform
    raw2 = raw.copy()
    raw2[dev] += 1000.0
    std2 = V.fit_train_only_standardizer(raw2, train)
    assert np.allclose(std.mean, std2.mean)
    assert np.allclose(std.std, std2.std)


# ---------------------------------------------------------------------------
# verdict logic
# ---------------------------------------------------------------------------
def test_preregistered_verdict_cases():
    assert V.preregistered_verdict(0.04, 0.031, 0.021)["case"] == "A"
    assert V.preregistered_verdict(0.03, 0.030, 0.020)["case"] == "A"
    assert V.preregistered_verdict(0.02, 0.020, 0.010)["case"] == "B"
    assert V.preregistered_verdict(0.01, 0.006, 0.012)["case"] == "C"
    assert V.preregistered_verdict(0.004, 0.004, 0.012)["case"] == "D"
    # shuffle precedence: strong-looking global gain with shallow binding is downgraded
    assert V.preregistered_verdict(0.06, 0.05, 0.004)["case"] == "DOWNGRADED_SHALLOW_BINDING"
    assert V.preregistered_verdict(0.05, 0.049, 0.0049)["case"] == "DOWNGRADED_SHALLOW_BINDING"


# ---------------------------------------------------------------------------
# tiny training smoke
# ---------------------------------------------------------------------------
def test_head_train_smoke_and_shuffle_defined():
    torch.manual_seed(0)
    rng = np.random.default_rng(1)
    n_train, n_dev = 128, 32
    raw = rng.standard_normal((n_train + n_dev, V.GLOBAL_WIDTH)).astype(np.float32)
    base = rng.standard_normal(n_train + n_dev).astype(np.float32)
    # residual is a function of one global column, so a good head can fit it
    y = (base + 0.5 * raw[:, 0]).astype(np.float32)
    std = V.fit_train_only_standardizer(raw, list(range(n_train)))
    X = std.transform(raw)
    res = V.train_residual_head(
        X[:n_train],
        y[:n_train] - base[:n_train],
        base[:n_train],
        y[:n_train],
        X[n_train:],
        base[n_train:],
        y[n_train:],
        seed=0,
        max_epochs=5,
        patience=5,
        batch=64,
        log=lambda *_: None,
    )
    assert res.soup_valid is not None and np.isfinite(res.soup_valid)
    assert len(res.soup_members) >= 1
    real = V.head_predictions(res.state_soup, X[n_train:], base[n_train:])
    perm = V.derangement(n_dev)
    shuf = V.head_predictions(res.state_soup, X[n_train:][perm], base[n_train:])
    assert real.shape == shuf.shape == (n_dev,)
    # the head actually learns something on this synthetic separable task
    init_state = {k: v.clone() for k, v in V.global62_head_class().state_dict().items()}
    init_pred = V.head_predictions(init_state, X[n_train:], base[n_train:])
    init_mae = float(np.abs(init_pred - y[n_train:]).mean())
    assert res.soup_valid < init_mae


# ---------------------------------------------------------------------------
# residual correlation helper
# ---------------------------------------------------------------------------
def test_residual_correlations_are_finite_and_bounded():
    rng = np.random.default_rng(3)
    raw = rng.standard_normal((50, 4)).astype(np.float32)
    raw[:, 2] = 1.0  # constant
    residual = (raw[:, 0] * 2.0 + rng.standard_normal(50) * 0.1).astype(np.float64)
    out = V.residual_correlations(raw, residual)
    assert len(out["pearson"]) == 4 and len(out["spearman"]) == 4
    assert all(np.isfinite(v) for v in out["pearson"] + out["spearman"])
    assert all(-1.0 <= v <= 1.0 for v in out["pearson"] + out["spearman"])
    assert out["pearson"][2] == 0.0 and out["spearman"][2] == 0.0
    assert out["pearson"][0] > 0.9


# ---------------------------------------------------------------------------
# saved artifact consistency (skipped on a fresh clone)
# ---------------------------------------------------------------------------
@pytest.mark.skipif(not ARTIFACT.exists(), reason="formal TCCD-GLOBAL62-v0 artifact not present")
def test_saved_artifact_consistency():
    payload = json.loads(ARTIFACT.read_text(encoding="utf-8"))
    assert payload["official_test_loaded"] is False
    assert payload["official_valid_loaded"] is False
    results = payload["results"]
    base = results["BASE"]["soup_valid_mae"]
    bias = results["BIAS_ONLY"]["soup_valid_mae"]
    real = results["REAL_GLOBAL62"]["soup_valid_mae"]
    shuf = results["SHUFFLED_GLOBAL62"]["soup_valid_mae"]
    assert abs(payload["g_total"] - (base - real)) < 1e-12
    assert abs(payload["g_global"] - (bias - real)) < 1e-12
    assert abs(payload["g_bind"] - (shuf - real)) < 1e-12
    assert abs(
        payload["gap_closed_fraction"] - (base - real) / (base - V.B_FULL_SCALE_REFERENCE)
    ) < 1e-12
    assert payload["verdict"]["case"] == "DOWNGRADED_SHALLOW_BINDING"
    assert payload["verdict"]["g_global"] < V.CASE_C_GLOBAL
    assert payload["verdict"]["g_bind"] < V.BIND_PRECEDENCE
    # base reproduction against the frozen v5 reference
    assert abs(base - 0.2514181435108185) <= V.REPRODUCTION_TOL
    assert payload["parameter_accounting"]["total"] == 3169
    # secondary correlation lists are finite
    corr = payload["secondary"]["residual_correlation_train"]
    assert len(corr["pearson"]) == 62 and len(corr["spearman"]) == 62
    assert all(np.isfinite(v) for v in corr["pearson"] + corr["spearman"])

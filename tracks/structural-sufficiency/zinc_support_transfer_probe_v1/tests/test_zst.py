"""Fast CPU tests for zinc-support-transfer-probe-v1 (no data, no checkpoints)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

_CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(_CODE))

import zst_head as H  # noqa: E402


def _fake_features(n=6, u=3):
    f = object.__new__(H.Features)
    rng = np.random.default_rng(0)
    f.ctx = rng.normal(size=(n, 64))
    f.aux = np.abs(rng.normal(size=(n, 2))) + 1.0
    f.atoms = rng.normal(size=(n, u, 110))
    f.mask = np.ones((n, u))
    f.gate = np.array([1.0, 0.0, 1.0, 0.0, 1.0, 1.0])
    f.rows = np.arange(n)
    f.n_unseen = f.mask.sum(1)
    return H.attach_tensors(f)


def test_zero_init_head_reproduces_base_exactly():
    f = _fake_features()
    for model in (H.HeadG(), H.HeadS()):
        h = H.predict_head(model, f)
        assert np.all(np.abs(h) == 0.0)


def test_hard_gate_makes_ungated_molecules_identical():
    f = _fake_features()
    model = H.HeadS()
    model.readout[-1].bias.data.fill_(0.25)
    h = H.predict_head(model, f)
    assert np.all(h[f.gate == 0] == 0.0)
    assert np.all(np.abs(h[f.gate == 1] - 0.25) < 1e-6)


def test_padding_does_not_enter_pooling():
    f = _fake_features()
    model = H.HeadS()
    model.readout[-1].bias.data.fill_(0.0)
    model.readout[-1].weight.data.normal_(0, 0.1)
    h1 = H.predict_head(model, f)
    f2 = object.__new__(H.Features)
    f2.__dict__.update(f.__dict__)
    # append an extra all-padding slot filled with a huge value
    f2.atoms = np.concatenate([f.atoms, np.full((f.atoms.shape[0], 1, f.atoms.shape[2]), 1e6)], axis=1)
    f2.mask = np.concatenate([f.mask, np.zeros((f.mask.shape[0], 1))], axis=1)
    H.attach_tensors(f2)
    assert np.all(np.abs(H.predict_head(model, f2) - h1) < 1e-5)


def test_standardiser_does_not_amplify_constant_columns():
    f = _fake_features()
    f.atoms[:, :, 5] = 7.0
    std = H.Standardiser(f)
    assert std.at_sd[5] == 1.0


def test_lr_schedule_endpoints():
    assert H._lr(0, 1500) == pytest.approx(H.LR)
    assert H._lr(1500, 1500) == pytest.approx(H.MIN_LR, abs=1e-12)


def test_metrics_trimmed_and_sign():
    y = np.array([0.0, 0.0, 0.0, 10.0])
    base = np.array([0.1, -0.1, 0.2, 0.0])
    corr = np.array([0.05, -0.05, 0.1, 0.0])
    m = H.metrics(y, base, corr)
    assert m["mae_corrected"] < m["mae_base"]
    assert m["trimmed_delta"] < 0
    assert 0.0 <= m["frac_improved"] <= 1.0


def test_key_clusters_partition():
    tkeys = np.array([11, 22])
    mol2 = [np.array([11, 99]), np.array([22]), np.array([99]), np.array([11, 22])]
    rows = np.array([0, 1, 2, 3])
    cl = H.key_cluster_ids(rows, tkeys, mol2)
    assert cl[0] == 0 and cl[1] == 1 and cl[2] == -1 and cl[3] == 0

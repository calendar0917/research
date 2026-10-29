"""Fast CPU unit tests for ECLC-v1.

The full correctness gate suite is ``code/eclc_cpu_checks.py``; these tests
cover the same contract at a smaller scale so a regression fails quickly.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import eclc_common as E
import eclc_models as EM
import eclc_train as T

D = E.D
L = E.L


@pytest.fixture(scope="module")
def norm():
    return E.load_norm()


@pytest.fixture(scope="module")
def models(norm):
    out = {}
    for arm in E.ARMS:
        out[arm] = EM.build_model(arm, norm, 0, "cpu")
    return out


def test_parameter_identity_and_zero_base(models):
    real, ri = models["REAL"]
    null, ni = models["NULL"]
    assert ri["parameters"] == ni["parameters"] == 248_368
    assert ri["parameters"] <= 500_000
    assert ri["parameters_breakdown"] == ni["parameters_breakdown"]
    assert ri["initial_state_sha256"] == ni["initial_state_sha256"]
    for a, b in zip(real.state_dict().values(), null.state_dict().values()):
        assert torch.equal(a, b)
    assert torch.count_nonzero(real.base.lins[1].weight).item() == 0
    assert torch.count_nonzero(real.base.lins[1].bias).item() == 0


def test_initial_equality_with_lcrc_a(norm, models):
    legacy, _ = E.M.build_model("A", norm, 0, "cpu")
    rows = list(range(8))
    cache = E.load_train_cache()
    batch = D.collate(cache, rows)
    with torch.no_grad():
        y0 = legacy(batch)
        for arm in E.ARMS:
            m, _ = models[arm]
            assert torch.equal(m(batch), y0)
    for k, v in legacy.state_dict().items():
        assert torch.equal(v, models["REAL"][0].backbone.state_dict()[k])


def test_base_layout_and_real_mask():
    atom_type = np.array([0, 1, 0, 2], dtype=np.int64)
    bond_index = np.array([[0, 1], [1, 2], [1, 3]], dtype=np.int64)
    bond_type = np.array([1, 2, 1], dtype=np.int64)
    atom_feat = np.zeros((4, L.ATOM_FEAT_DIM), dtype=np.float32)
    bond_feat = np.zeros((3, L.BOND_FEAT_DIM), dtype=np.float32)
    mol = D.build_from_arrays(atom_feat, atom_type, bond_index, bond_type,
                              bond_feat, np.zeros(L.GLOBAL_DIM, np.float32))
    cache = D._concat_molecules([mol])
    batch = D.collate(cache, [0])
    full = EM.base_features(batch, zero_r2=False)
    real = EM.base_features(batch, zero_r2=True)
    assert full.shape[1] == E.BASE_INPUT_DIM == 337
    assert torch.equal(full[:, :E.BASE_R1_TOTAL], real[:, :E.BASE_R1_TOTAL])
    assert torch.count_nonzero(real[:, E.BASE_R1_TOTAL:]).item() == 0
    assert torch.count_nonzero(full[:, E.BASE_R1_TOTAL:]).item() > 0


def test_synthetic_radius1_invariance(norm):
    from eclc_cpu_checks import _synthetic_mols, _nonzero_base
    ca, cb = _synthetic_mols()
    ba, bb = D.collate(ca, [0]), D.collate(cb, [0])
    m, _ = EM.build_model("REAL", norm, 0, "cpu")
    n, _ = EM.build_model("NULL", norm, 0, "cpu")
    _nonzero_base(m)
    _nonzero_base(n)
    with torch.no_grad():
        ra = EM.base_features(ba, zero_r2=True)
        rb = EM.base_features(bb, zero_r2=True)
        na = EM.base_features(ba, zero_r2=False)
        nb = EM.base_features(bb, zero_r2=False)
        assert torch.equal(ra[0, :E.BASE_R1_TOTAL], rb[0, :E.BASE_R1_TOTAL])
        assert not torch.equal(na[0, E.BASE_R1_TOTAL:], nb[0, E.BASE_R1_TOTAL:])
        assert torch.equal(m.base(ra)[0], m.base(rb)[0])
        assert not torch.equal(n.base(na)[0], n.base(nb)[0])


def test_mechanism_difference(norm, models):
    from eclc_cpu_checks import _nonzero_base
    real, _ = models["REAL"]
    null, _ = models["NULL"]
    _nonzero_base(real)
    _nonzero_base(null)
    cache = E.load_train_cache()
    batch = D.collate(cache, list(range(16)))
    with torch.no_grad():
        _, ar = real(batch, return_aux=True)
        _, an = null(batch, return_aux=True)
        assert torch.equal(ar["context"], an["context"])
        assert torch.equal(ar["rho"], an["rho"])
        assert not torch.equal(ar["base"], an["base"])
        assert not torch.equal(real(batch), null(batch))


def test_pair_loss_algebra():
    r = torch.tensor([0.3, -0.7, 1.1, 2.0], dtype=torch.float64)
    n = len(r)
    pair_sq = sum(float((r[a] - r[b]) ** 2) for a in range(n) for b in range(a + 1, n))
    quad = float((n - 1) * (r ** 2).sum() - 2 * sum(float(r[a] * r[b])
                                                    for a in range(n) for b in range(a + 1, n)))
    assert abs(pair_sq - quad) < 1e-6
    y = torch.zeros(4)
    p = torch.tensor([0.5, 0.2, -0.1, 0.4])
    assert torch.allclose(F.l1_loss((p[0] + 7.0) - (p[1] + 7.0), y[0] - y[1]),
                          F.l1_loss(p[0] - p[1], y[0] - y[1]))
    assert not torch.allclose(F.l1_loss(p + 7.0, y), F.l1_loss(p, y))


def test_checkpoint_roundtrip(models):
    cache = E.load_train_cache()
    batch = D.collate(cache, [0, 1, 2])
    for arm in E.ARMS:
        m, _ = models[arm]
        state = {k: v.detach().clone() for k, v in m.state_dict().items()}
        m2 = EM.load_model_from_state(arm, E.load_norm(), state, "cpu")
        with torch.no_grad():
            assert torch.equal(m(batch), m2(batch))


def test_pair_support_counts():
    pairs = T.load_pair_arrays()
    assert pairs["n"] == 350
    assert pairs["i"].min() >= 0 and pairs["j"].min() >= 0
    assert pairs["i"].max() < 10000 and pairs["j"].max() < 10000
    assert np.all(pairs["i"] != pairs["j"])
    payload = pairs["payload"]
    assert payload["counts"]["n_accepted_pairs"] == 350
    assert payload["counts"]["n_distinct_bodies"] == 317
    assert payload["counts"]["n_independent_molecules"] == 621


def test_frozen_slice_counts():
    import lc_analysis as LA
    anchor = LA._read_anchor()
    masks, _ = LA._fixed_masks(anchor)
    masks = dict(masks)
    masks["seen_ordinary"] = (~anchor["unseen"]) & (anchor["y"] > -3.0)
    expected = {
        "all": 1000, "ordinary_y_gt_-3": 935, "tail_y_le_-3": 65,
        "r2_unseen_ordinary": 225, "seen_ordinary": 710,
        "all_minus_row172": 999, "all_minus_worst10": 990,
    }
    for k, n in expected.items():
        assert int(masks[k].sum()) == n, k


def test_no_test_reference_in_production_code():
    forbidden = "lc_" + "test"
    for name in ("eclc_common.py", "eclc_match.py", "eclc_models.py",
                 "eclc_phase0.py", "eclc_train.py"):
        assert forbidden not in (E.CODE_DIR / name).read_text(), name

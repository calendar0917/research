"""Fast unit tests for SFC-v0 (no data, no checkpoints, CPU only)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import sfc_core as SC  # noqa: E402

N_ATOM = 5
N_BOND = 4


def _params(C=2, seed=0, scale=1.0, grad=False):
    g = torch.Generator().manual_seed(seed)
    A = (torch.randn(N_ATOM, C, 2, generator=g, dtype=torch.float64) * scale
         ).requires_grad_(grad)
    S = (torch.randn(N_BOND, C, 3, generator=g, dtype=torch.float64) * scale
         ).requires_grad_(grad)
    return A, S


def _path(n):
    return SC.TypedGraph(n, np.zeros(n, np.int64), np.arange(n - 1),
                         np.arange(1, n), np.ones(max(n - 1, 0), np.int64))


def _ring(n):
    return SC.TypedGraph(n, np.zeros(n, np.int64), np.arange(n),
                         (np.arange(n) + 1) % n, np.ones(n, np.int64))


def test_elimination_matches_dense_enumeration():
    for C in (1, 3):
        A, S = _params(C)
        for n in range(1, 8):
            for g in (_path(n), _ring(n) if n >= 3 else _path(n)):
                out = SC.sfc_forward(g, A, S)
                ref = SC.log_partition_dense(g, A, S)
                assert torch.allclose(out.logZ, ref, atol=1e-10)


def test_relabel_invariance():
    A, S = _params(3, seed=1)
    rng = np.random.default_rng(1)
    g = SC.TypedGraph(6, rng.integers(0, N_ATOM, 6).astype(np.int64),
                      np.array([0, 1, 2, 2, 4]), np.array([1, 2, 3, 4, 5]),
                      np.array([1, 2, 1, 3, 2]))
    base = SC.sfc_forward(g, A, S)
    for _ in range(8):
        perm = rng.permutation(6)
        atom = np.zeros_like(g.atom)
        atom[perm] = g.atom
        hp = SC.sfc_forward(SC.TypedGraph(6, atom, perm[g.edge_u], perm[g.edge_v],
                                          g.edge_b.copy()), A, S)
        assert torch.allclose(base.logZ, hp.logZ, atol=1e-12)
        assert torch.allclose(base.T, hp.T, atol=1e-12)


def test_gradient_matches_finite_difference():
    A, S = _params(2, seed=2, grad=True)
    g = _ring(4)
    SC.sfc_forward(g, A, S).logZ.sum().backward()
    ana = A.grad.clone()
    A.grad = None
    eps = 1e-6
    with torch.no_grad():
        num = torch.zeros_like(A)
        for idx in np.ndindex(*A.shape):
            old = float(A[idx])
            A[idx] = old + eps
            fp = float(SC.log_partition_dense(g, A, S).sum())
            A[idx] = old - eps
            fm = float(SC.log_partition_dense(g, A, S).sum())
            A[idx] = old
            num[idx] = (fp - fm) / (2 * eps)
    assert torch.allclose(ana, num, atol=1e-6, rtol=1e-5)


def test_zero_coupling_gives_zero_interaction():
    A, _ = _params(3, seed=3)
    S = torch.zeros(N_BOND, 3, 3, dtype=torch.float64)
    for g in (_path(5), _ring(5)):
        out = SC.sfc_forward(g, A, S)
        assert float(out.H.abs().max()) < 1e-12
        assert float(out.T.abs().max()) < 1e-12


def test_forest_has_zero_cycle_interaction_ring_does_not():
    A, S = _params(3, seed=4, scale=1.5)
    A0 = torch.zeros_like(A)          # no node field
    for AA in (A, A0):
        for n in range(2, 8):
            assert float(SC.sfc_forward(_path(n), AA, S).T.abs().max()) < 1e-10
        for n in range(3, 8):
            assert float(SC.sfc_forward(_ring(n), AA, S).T.abs().max()) > 1e-8


def test_same_composition_different_connections_differ():
    atoms = np.array([0, 1, 2, 0, 1, 2], np.int64)
    bonds = np.ones(6, np.int64)
    two_tri = SC.TypedGraph(6, atoms, np.array([0, 1, 2, 3, 4, 5]),
                            np.array([1, 2, 0, 4, 5, 3]), bonds)
    hexagon = SC.TypedGraph(6, atoms, np.array([0, 1, 2, 3, 4, 5]),
                            np.array([1, 2, 3, 4, 5, 0]), bonds)
    A, S = _params(2, seed=5)
    a = SC.sfc_forward(two_tri, A, S)
    b = SC.sfc_forward(hexagon, A, S)
    assert float((a.T - b.T).abs().max()) > 1e-6


def test_marginals_normalise():
    A, S = _params(3, seed=6)
    g = _ring(5)
    out = SC.sfc_forward(g, A, S)
    assert torch.allclose(out.node_p.sum(-1), torch.ones(3, 5, dtype=torch.float64),
                          atol=1e-10)
    assert torch.allclose(out.edge_p.sum(dim=(-1, -2)),
                          torch.ones(3, g.m, dtype=torch.float64), atol=1e-10)


def test_elimination_order_does_not_change_value():
    A, S = _params(2, seed=7)
    g = SC.TypedGraph(6, np.zeros(6, np.int64),
                      np.array([0, 1, 2, 3, 4, 0]), np.array([1, 2, 3, 4, 5, 3]),
                      np.ones(6, np.int64))
    vals = [SC.sfc_forward(g, A, S, order=fn(g.n, g.edges())).logZ
            for fn in SC.ORDERS.values()]
    assert torch.allclose(vals[0], vals[1], atol=1e-12)


def test_plan_cost_is_reported():
    g = _ring(6)
    plan = SC.build_plan(g.n, g.edges(), SC.order_min_fill(g.n, g.edges()))
    cost = plan.cost(16)
    assert cost["induced_width"] >= 1
    assert cost["max_bucket_floats"] == 16 * plan.max_bucket_elems
    assert plan.arith_ops > 0


def test_label_blocked_reader_refuses_target():
    sys.path.insert(0, str(_CODE))
    import sfc_cost as CO

    d = CO.LabelBlocked({"num_atom": 3, CO.LABEL_KEY: 1.0})
    assert d["num_atom"] == 3
    with pytest.raises(PermissionError):
        _ = d[CO.LABEL_KEY]
    with pytest.raises(PermissionError):
        d.get(CO.LABEL_KEY)

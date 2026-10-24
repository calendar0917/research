"""Regression tests for the SFC-v0 corrigendum (Bug A / Bug B).

These tests encode the two defects found in the pre-fix implementation
(commit ``ed70416``):

* Bug A — evaluation computed ``T`` without the exact marginals, so ``T`` was
  silently ``logZ``; train-time and eval-time features differed.
* Bug B — ``_minus_bethe_free_energy`` detached the explicit potentials, so
  ``dT/dtheta`` did not match the value of ``T`` (forests had ``T == 0`` but a
  non-zero gradient).

CPU only, no data, float64.
"""

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
import sfc_synth as SY  # noqa: E402

N_ATOM = 4
N_BOND = 4


def _params(C=1, seed=0, scale=1.2, grad=False):
    g = torch.Generator().manual_seed(seed)
    A = (torch.randn(N_ATOM, C, 2, generator=g, dtype=torch.float64) * scale
         ).requires_grad_(grad)
    S = (torch.randn(N_BOND, C, 3, generator=g, dtype=torch.float64) * scale
         ).requires_grad_(grad)
    return A, S


def _graph(n, eu, ev, eb, atom=None):
    atom = np.arange(n, dtype=np.int64) % N_ATOM if atom is None else atom
    return SC.TypedGraph(n, np.asarray(atom, np.int64), np.asarray(eu, np.int64),
                         np.asarray(ev, np.int64), np.asarray(eb, np.int64))


def _ring(n, seed=0):
    rng = np.random.default_rng(seed)
    return _graph(n, np.arange(n), (np.arange(n) + 1) % n,
                  rng.integers(1, 4, n), rng.integers(0, N_ATOM, n))


def _path(n):
    return _graph(n, np.arange(n - 1), np.arange(1, n),
                  (np.arange(n - 1) % 3) + 1)


def test_T_without_marginals_is_refused_not_logZ():
    A, S = _params(seed=1)
    g = _ring(4)
    out = SC.sfc_forward(g, A, S, with_marginals=False)
    assert torch.isfinite(out.H).all()
    assert torch.isfinite(out.logZ).all()
    with pytest.raises(RuntimeError, match="requires the exact marginals"):
        _ = out.T
    with pytest.raises(RuntimeError):
        _ = out.minus_FB


def test_eval_under_no_grad_computes_the_same_T_as_train():
    A, S = _params(C=2, seed=2)
    g = _ring(4)
    train = SC.sfc_forward(g, A, S)
    with torch.no_grad():
        ev = SC.sfc_forward(g, A, S)
    assert torch.equal(train.H, ev.H)
    assert torch.equal(train.T, ev.T)
    assert not ev.T.requires_grad
    assert float(train.T.abs().max()) > 1e-6


def test_train_and_eval_feature_batches_are_identical():
    graphs = [_ring(4), _ring(5, seed=1), _path(6)]
    A, S = _params(C=2, seed=3)
    with torch.enable_grad():
        train_feat = SY._sfc_batch(graphs, A.clone().requires_grad_(True),
                                   S.clone().requires_grad_(True), True)
    with torch.no_grad():
        eval_feat = SY._sfc_batch(graphs, A, S, False)
    assert torch.equal(train_feat.detach(), eval_feat)


def _fd(fn, tensors, eps=1e-6):
    out = []
    for t in tensors:
        num = torch.zeros_like(t)
        for idx in np.ndindex(*t.shape):
            old = float(t[idx])
            with torch.no_grad():
                t[idx] = old + eps
            fp = float(fn())
            with torch.no_grad():
                t[idx] = old - eps
            fm = float(fn())
            with torch.no_grad():
                t[idx] = old
            num[idx] = (fp - fm) / (2 * eps)
        out.append(num)
    return out


def test_T_gradient_matches_finite_difference_on_a_cycle():
    g = _ring(4, seed=5)
    A, S = _params(seed=6, grad=True)

    def f():
        with torch.enable_grad():
            return SC.sfc_forward(g, A, S).T.sum()

    val = f()
    gA, gS = torch.autograd.grad(val, [A, S])
    nA, nS = _fd(f, [A, S])
    assert float(val.abs()) > 1e-6
    assert torch.allclose(gA, nA, atol=1e-7, rtol=1e-4)
    assert torch.allclose(gS, nS, atol=1e-7, rtol=1e-4)


def test_forest_T_and_its_gradient_are_both_zero():
    cases = {
        "isolated": _graph(1, [], [], []),
        "single_edge": _graph(2, [0], [1], [2]),
        "path_7": _path(7),
    }
    for i, (name, g) in enumerate(cases.items()):
        assert g.is_forest(), name
        A, S = _params(seed=10 + i, grad=True)
        T = SC.sfc_forward(g, A, S).T.sum()
        gA, gS = torch.autograd.grad(T, [A, S], allow_unused=True)
        gA = torch.zeros_like(A) if gA is None else gA
        gS = torch.zeros_like(S) if gS is None else gS
        assert float(T.abs()) < 1e-10, name
        # checking only the value is exactly what missed the original bug
        assert float(gA.abs().max()) < 1e-9, name
        assert float(gS.abs().max()) < 1e-9, name


def test_srand_cache_is_recomputed_from_the_arm_buffers():
    split = SY.assemble("reach", n_seen=8, n_unseen=4, seed=0)
    plan = split["sfc_plan"]
    model = SY.Arm("Srand", comp_dim=int(split["comp_train"].shape[1]),
                   hidden=8, sfc_plan=plan)
    assert torch.equal(model.A, torch.as_tensor(plan["A"], dtype=torch.float64))
    assert torch.equal(model.S, torch.as_tensor(plan["S"], dtype=torch.float64))
    graphs = split["graphs_train"]
    with torch.no_grad():
        feat = SY._sfc_batch(graphs, model.A, model.S, need_grad=False)
    assert torch.equal(torch.as_tensor(split["sfc_train"]), feat)


def test_fit_diagnostic_does_not_mutate_shared_factors():
    split = SY.assemble("reach", n_seen=32, n_unseen=8, seed=0)
    fp0 = SY.plan_fingerprint(split["sfc_plan"])
    diag = SY.sfc_fit_diagnostic("reach", split, steps=2, n_graphs=8)
    assert SY.plan_fingerprint(split["sfc_plan"]) == fp0
    assert diag["plan_mutated"] is False
    # and the cached Srand features are still exactly reproducible afterwards
    audit = SY.audit_sfc_feature_consistency(split, "train", n=8)
    assert audit["pass"], audit


def test_arm_parameters_do_not_alias_the_shared_plan():
    plan = SY.make_sfc_plan(seed=0)
    fp0 = SY.plan_fingerprint(plan)
    model = SY.Arm("S", comp_dim=4, hidden=8, sfc_plan=plan)
    with torch.no_grad():
        model.A.add_(1.0)
        model.S.add_(1.0)
    assert SY.plan_fingerprint(plan) == fp0
    frozen = SY.Arm("Srand", comp_dim=4, hidden=8, sfc_plan=plan)
    assert frozen.A is not plan["A"] and frozen.S is not plan["S"]


def test_plan_cache_key_ignores_object_identity():
    g1 = _ring(4, seed=7)
    g2 = _ring(5, seed=8)
    assert SY._plan_key(g1) != SY._plan_key(g2)
    p1 = SY._plan_for(g1)
    p2 = SY._plan_for(g2)
    assert p1.induced_width >= 1 and p2.induced_width >= 1
    # a structurally equal copy (different object / id) reuses the same plan
    g1_copy = _graph(g1.n, g1.edge_u, g1.edge_v, g1.edge_b, atom=g1.atom)
    assert SY._plan_key(g1_copy) == SY._plan_key(g1)
    assert SY._plan_for(g1_copy) is p1

"""Fast CPU unit tests for the rooted-moment mathematics (no data, no GPU)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

_CODE = Path(__file__).resolve().parents[1]
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import rmc_common as C  # noqa: E402
import rmc_data as D  # noqa: E402
import rmc_moments as RM  # noqa: E402


def _direct_moments(u, p):
    mu = (p[:, None] * u).sum(0)
    d = u - mu
    c2 = torch.einsum("n,na,nb->ab", p, d, d)
    c3 = torch.einsum("n,na,nb,nc->abc", p, d, d, d)
    pi, pj, _ = RM.pair_arrays()
    ti, tj, tk = RM.triple_arrays()
    return mu, c2[pi, pj], c3[ti, tj, tk]


def test_witness_values():
    w = RM.witness()["distributions"]
    assert abs(w["A"]["F_moment_with_C3"] - 0.3846153846) < 1e-9
    assert abs(w["B"]["F_moment_with_C3"] - 0.4526064213) < 1e-9
    assert abs(w["B"]["F_gaussian_closure"] - 0.3846153846) < 1e-9
    assert abs(w["B"]["third_central_moment"] - 0.08838834765) < 1e-10
    assert abs(w["A"]["F_direct_sum"] - w["A"]["F_moment_with_C3"]) < 1e-12


def test_readout_matches_direct_sum_and_gradient():
    torch.manual_seed(0)
    n, s_max = 9, 3
    u = torch.randn(n, C.PROJ_DIM, dtype=torch.float64)
    prob = torch.rand(n, s_max, n, dtype=torch.float64) + 0.05
    prob = prob / prob.sum(-1, keepdim=True)
    mu = torch.zeros(n, s_max, C.PROJ_DIM, dtype=torch.float64)
    c2 = torch.zeros(n, s_max, C.N_PAIR, dtype=torch.float64)
    c3 = torch.zeros(n, s_max, C.N_TRIPLE, dtype=torch.float64)
    for i in range(n):
        for s in range(s_max):
            m, q2, q3 = _direct_moments(u, prob[i, s])
            mu[i, s], c2[i, s], c3[i, s] = m, q2, q3
    q = torch.randn(n, 2, C.PROJ_DIM, dtype=torch.float64, requires_grad=True)
    F, Z = RM.readout_batched(mu, c2, c3, q)
    worst = 0.0
    for i in range(n):
        for s in range(s_max):
            for h in range(2):
                Fd, Zd = RM.readout_direct(u, torch.log(prob[i, s]), q[i, h].detach().unsqueeze(0))
                worst = max(worst, float((F[i, s, h] - Fd[0]).abs().max()), float(abs(Z[i, s, h] - Zd[0])))
    assert worst < 1e-10
    # gradient of the moment readout equals the gradient of the direct sum
    wt = torch.randn(n, s_max, 2, C.PROJ_DIM, dtype=torch.float64)
    (F * wt).sum().backward()
    g_moment = q.grad.clone()
    qd = q.detach().clone().requires_grad_(True)
    total = 0.0
    for i in range(n):
        for s in range(s_max):
            for h in range(2):
                Fd, _ = RM.readout_direct(u, torch.log(prob[i, s]), qd[i, h].unsqueeze(0))
                total = total + (Fd[0] * wt[i, s, h]).sum()
    total.backward()
    assert float((g_moment - qd.grad).abs().max()) < 1e-10


def test_packed_equals_full_and_q_zero():
    torch.manual_seed(1)
    n = 13
    u = torch.randn(n, C.PROJ_DIM, dtype=torch.float64)
    p = torch.rand(n, dtype=torch.float64) + 0.01
    p = p / p.sum()
    mu, c2p, c3p = _direct_moments(u, p)
    d = u - mu
    c2_full = torch.einsum("n,na,nb->ab", p, d, d)
    c3_full = torch.einsum("n,na,nb,nc->abc", p, d, d, d)
    assert float(torch.linalg.eigvalsh(c2_full).min()) > -1e-12
    q = torch.randn(1, 1, C.PROJ_DIM, dtype=torch.float64)
    F, Z = RM.readout_batched(mu.unsqueeze(0).unsqueeze(0), c2p.unsqueeze(0).unsqueeze(0),
                              c3p.unsqueeze(0).unsqueeze(0), q)
    a = 1.0 + q[0, 0] @ mu
    Zf = a * a + q[0, 0] @ c2_full @ q[0, 0] + C.EPS
    Ff = mu + (2.0 * a * (c2_full @ q[0, 0]) + torch.einsum("abc,b,c->a", c3_full, q[0, 0], q[0, 0])) / Zf
    assert float((F[0, 0, 0] - Ff).abs().max()) < 1e-11
    assert abs(float(Z[0, 0, 0]) - float(Zf)) < 1e-11
    # q = 0 -> F = mu, Z = 1 + eps
    F0, Z0 = RM.readout_batched(mu.unsqueeze(0).unsqueeze(0), c2p.unsqueeze(0).unsqueeze(0),
                                c3p.unsqueeze(0).unsqueeze(0),
                                torch.zeros(1, 1, C.PROJ_DIM, dtype=torch.float64))
    assert float((F0[0, 0, 0] - mu).abs().max()) < 1e-13
    assert abs(float(Z0[0, 0, 0]) - (1.0 + C.EPS)) < 1e-13


def test_walk_powers_and_k0():
    rg = np.random.default_rng(3)
    n = 5
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    keep = rg.random(len(pairs)) < 0.6
    bonds = np.asarray([p for p, k in zip(pairs, keep) if k], dtype=np.int64)
    powers = RM.walk_powers(n, bonds, (0, 1, 2, 3))
    # independent count: enumerate every walk of length k over I + A
    for k, W in zip((0, 1, 2, 3), powers):
        count = np.zeros((n, n), dtype=np.int64)
        adj = [set([i]) for i in range(n)]
        for a, b in bonds:
            adj[int(a)].add(int(b))
            adj[int(b)].add(int(a))

        def rec(start, cur, left):
            if left == 0:
                count[start, cur] += 1
                return
            for nxt in adj[cur]:
                rec(start, nxt, left - 1)

        for s in range(n):
            rec(s, s, k)
        assert np.array_equal(W, count)
    u = np.stack([rg.normal(size=(n, C.PROJ_DIM))] * 2, axis=1)
    out = D.molecule_moments(u, bonds)
    z = C.SCALES.index(0)
    assert np.all(out["c2"][:, z] == 0.0)
    assert np.all(out["c3"][:, z] == 0.0)
    assert np.allclose(out["mu"][:, z], u, atol=1e-15)


def test_single_atom_and_permutation():
    rg = np.random.default_rng(5)
    bonds = np.asarray([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]])
    u = rg.normal(size=(4, C.PROJ_DIM))
    uu = np.stack([u, u], axis=1)
    base = D.molecule_moments(uu, bonds)
    flipped = D.molecule_moments(uu, bonds[:, ::-1])
    assert np.allclose(base["c3"], flipped["c3"], atol=1e-14)
    perm = np.array([2, 0, 3, 1])
    inv = np.argsort(perm)
    permuted = D.molecule_moments(uu[perm], inv[bonds])
    assert np.allclose(base["mu"][perm], permuted["mu"], atol=1e-12)
    assert np.allclose(base["c3"][perm], permuted["c3"], atol=1e-12)
    single = D.molecule_moments(uu[:1], np.zeros((0, 2), dtype=np.int64))
    assert np.allclose(single["mu"][0], uu[0], atol=1e-15)
    assert np.all(single["c3"] == 0.0)

"""CPU correctness tests for the BAMF exact inference operator.

Run: uv run pytest -q tracks/structural-sufficiency/zinc_bond_anchored_marginal_field_v1/tests
"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
import torch

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE.parent / "code"))

import bamf_inference as B  # noqa: E402

torch.set_default_dtype(torch.float64)


def _dense(n, edges, phi, psi):
    c = phi.shape[1]
    zero = torch.zeros(c, dtype=phi.dtype)
    weights = []
    for mask in range(1 << n):
        s = [(mask >> v) & 1 for v in range(n)]
        w = sum((phi[v, :, s[v]] for v in range(n)), zero)
        w = w + sum((psi[e, :, s[u], s[v]] for e, (u, v) in enumerate(edges)), zero)
        weights.append(w)
    w = torch.stack(weights, 0)
    logz = torch.logsumexp(w, 0)
    p = torch.exp(w - logz)
    nm = torch.zeros(n, c, 2)
    em = torch.zeros(len(edges), c, 2, 2)
    for mask in range(1 << n):
        for v in range(n):
            nm[v, :, (mask >> v) & 1] += p[mask]
        for e, (u, v) in enumerate(edges):
            em[e, :, (mask >> u) & 1, (mask >> v) & 1] += p[mask]
    return logz, nm, em


def _random_graph(n, rng):
    edges = []
    for v in range(1, n):
        edges.append((int(rng.integers(0, v)), v))
    for u in range(n):
        for v in range(u + 1, n):
            if (u, v) not in edges and rng.random() < 0.2:
                edges.append((u, v))
    return edges


@pytest.mark.parametrize("n", [2, 3, 4, 5, 6, 7, 8])
def test_exact_marginals_logz_vs_dense(n):
    rng = __import__("numpy").random.default_rng(1000 + n)
    for _ in range(4):
        edges = _random_graph(n, rng)
        c = 4
        phi = torch.randn(n, c, 2) * 0.7
        raw = torch.randn(len(edges), c, 3) * 0.7
        psi = B.edge_log_potential(raw)
        gp = B.build_global_plan([(n, edges)], smax=8)
        bp = gp.select([0], [n], [len(edges)])
        nm, em = B.exact_marginals(phi, psi, bp)
        logz = B.exact_logz(phi, psi, bp)[0]
        lz0, nm0, em0 = _dense(n, edges, phi, psi)
        assert torch.allclose(nm, nm0, atol=1e-10)
        assert torch.allclose(em, em0, atol=1e-10)
        assert torch.allclose(logz, lz0, atol=1e-10)


def test_batch_matches_single_with_empty_and_disconnected():
    graphs = [
        (0, []),
        (1, []),
        (2, [(0, 1)]),
        (6, [(0, 1), (1, 2), (2, 0), (3, 4)]),
        (4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        (5, [(i, j) for i in range(5) for j in range(i + 1, 5)]),
    ]
    torch.manual_seed(7)
    phis = [torch.randn(n, 4, 2) * 0.9 for n, _ in graphs]
    raws = [torch.randn(len(e), 4, 3) * 0.9 for _, e in graphs]
    psis = [B.edge_log_potential(r) for r in raws]
    gp = B.build_global_plan(graphs, smax=8)
    bp = gp.select(list(range(len(graphs))), [n for n, _ in graphs], [len(e) for _, e in graphs])
    phi = torch.cat(phis)
    psi = torch.cat(psis)
    nm, em = B.exact_marginals(phi, psi, bp)
    logz = B.exact_logz(phi, psi, bp)
    noff = eoff = 0
    for i, (n, edges) in enumerate(graphs):
        lz0, nm0, em0 = _dense(n, edges, phis[i], psis[i])
        if n:
            assert torch.allclose(nm[noff:noff + n], nm0, atol=1e-10)
        if edges:
            assert torch.allclose(em[eoff:eoff + len(edges)], em0, atol=1e-10)
        assert torch.allclose(logz[i], lz0, atol=1e-10)
        noff += n
        eoff += len(edges)


def test_invariances():
    n = 7
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0), (1, 4)]
    torch.manual_seed(11)
    phi = torch.randn(n, 4, 2) * 0.8
    psi = B.edge_log_potential(torch.randn(len(edges), 4, 3) * 0.8)
    gp = B.build_global_plan([(n, edges)], smax=8)
    bp = gp.select([0], [n], [len(edges)])
    nm, em = B.exact_marginals(phi, psi, bp)

    # atom renumbering: new index j carries old atom perm[j]
    perm = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    edges_p = [(int(inv[u]), int(inv[v])) for u, v in edges]
    gp_p = B.build_global_plan([(n, edges_p)], smax=8)
    bp_p = gp_p.select([0], [n], [len(edges)])
    nm_p, em_p = B.exact_marginals(phi[perm], psi, bp_p)
    assert torch.allclose(nm_p[inv], nm, atol=1e-10)
    assert torch.allclose(em_p, em, atol=1e-10)

    # endpoint swap
    edges_r = [(v, u) for u, v in edges]
    gp_r = B.build_global_plan([(n, edges_r)], smax=8)
    bp_r = gp_r.select([0], [n], [len(edges)])
    _, em_r = B.exact_marginals(phi, psi.transpose(-1, -2), bp_r)
    assert torch.allclose(em_r.transpose(-1, -2), em, atol=1e-10)

    # bond permutation
    pe = torch.randperm(len(edges))
    edges_e = [edges[i] for i in pe.tolist()]
    gp_e = B.build_global_plan([(n, edges_e)], smax=8)
    bp_e = gp_e.select([0], [n], [len(edges)])
    _, em_e = B.exact_marginals(phi, psi[pe], bp_e)
    inv_e = torch.empty_like(pe)
    inv_e[pe] = torch.arange(len(edges))
    assert torch.allclose(em_e[inv_e], em, atol=1e-10)


@pytest.mark.parametrize("scale", [20.0, 50.0, 100.0])
def test_extreme_potentials_stay_finite_and_normalised(scale):
    n = 7
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5), (5, 6), (6, 0), (1, 4)]
    torch.manual_seed(12)
    phi = torch.randn(n, 4, 2) * scale
    psi = B.edge_log_potential(torch.randn(len(edges), 4, 3) * scale)
    gp = B.build_global_plan([(n, edges)], smax=8)
    bp = gp.select([0], [n], [len(edges)])
    nm, em = B.exact_marginals(phi, psi, bp)
    logz = B.exact_logz(phi, psi, bp)
    assert torch.isfinite(nm).all() and torch.isfinite(em).all() and torch.isfinite(logz).all()
    assert torch.allclose(nm.sum(-1), torch.ones_like(nm.sum(-1)), atol=1e-9)
    assert torch.allclose(em.sum((-1, -2)), torch.ones_like(em.sum((-1, -2))), atol=1e-9)


def test_finite_difference_and_second_order():
    n = 5
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    gp = B.build_global_plan([(n, edges)], smax=8)
    bp = gp.select([0], [n], [len(edges)])
    torch.manual_seed(3)
    phi = torch.randn(n, 2, 2, requires_grad=True)
    raw = torch.randn(len(edges), 2, 3, requires_grad=True)
    nm, em = B.exact_marginals(phi, B.edge_log_potential(raw), bp)
    w = torch.randn_like(em)
    loss = (em * w).sum() + nm.square().sum()

    def loss_of(rv):
        nn, ee = B.exact_marginals(phi, B.edge_log_potential(rv), bp)
        return (ee * w).sum() + nn.square().sum()

    graw, = torch.autograd.grad(loss, raw, create_graph=True)
    eps = 1e-6
    idx = (0, 0, 0)
    with torch.no_grad():
        rp = raw.detach().clone(); rp[idx] += eps
        rm = raw.detach().clone(); rm[idx] -= eps
        fd = (loss_of(rp) - loss_of(rm)) / (2 * eps)
    assert abs(float(fd) - float(graw[idx])) < 2e-6

    w2 = torch.randn_like(raw)
    second, = torch.autograd.grad((graw * w2).sum(), raw)

    def graw_of(rv):
        rv = rv.detach().clone().requires_grad_(True)
        with torch.enable_grad():
            g, = torch.autograd.grad(loss_of(rv), rv)
        return g

    with torch.no_grad():
        rp = raw.detach() + eps * w2
        rm = raw.detach() - eps * w2
        fd2 = ((graw_of(rp) - graw_of(rm)) / (2 * eps) * w2).sum()
    assert abs(float(fd2) - float((second * w2).sum())) < 2e-5


def test_independent_is_two_node_softmax_and_global_differs_on_chain():
    n = 6
    edges = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 5)]
    torch.manual_seed(5)
    phi = torch.randn(n, 2, 2)
    psi = B.edge_log_potential(torch.randn(len(edges), 2, 3))
    bond_index = torch.tensor(edges)
    _, em_i = B.independent_marginals(phi, psi, bond_index)
    # each I edge is exactly the two-node conditional on that edge alone
    for e, (u, v) in enumerate(edges):
        logits = phi[u].unsqueeze(-1) + phi[v].unsqueeze(-2) + psi[e]
        expected = torch.softmax(logits.reshape(2, 4), dim=-1).reshape(2, 2, 2)
        assert torch.allclose(em_i[e], expected, atol=1e-12)
    gp = B.build_global_plan([(n, edges)], smax=8)
    bp = gp.select([0], [n], [len(edges)])
    _, em_g = B.exact_marginals(phi, psi, bp)
    assert float((em_g - em_i).abs().max()) > 1e-3

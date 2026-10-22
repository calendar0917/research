"""Fast, data-free unit tests for the ``zinc_shared_factor_v1`` operator.

These cover only the local operator identity properties (exact contraction,
reordering invariance, the H1/H2 witness, the pure star, padding silence,
type paths).  They are CPU-only, need no ZINC data and no checkpoint.

The full pre-training gate (cache digests, LF-D identity, gradients, schedule)
is ``code/sf_checks.py`` and needs the data caches.
"""

from __future__ import annotations

import itertools
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
for _p in (str(_CODE),):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import sf_models as SF  # noqa: E402

PAIR_K = {p: k for k, p in enumerate(SF.PAIR_INDEX)}
SEED = 20261022


def make_table(atom, pair, length, family):
    """Minimal static table with the attributes ``AtomicBranch`` reads."""
    class T:
        pass

    t = T()
    t.atom_ids = torch.as_tensor(np.asarray(atom, dtype=np.int64)).reshape(1, -1)
    t.pair_ids = torch.as_tensor(np.asarray(pair, dtype=np.int64)).reshape(1, -1)
    t.node_mask = (torch.arange(SF.MAX_NODES) < length).float().reshape(1, -1)
    t.family = torch.as_tensor([family], dtype=torch.long)
    t.n_patterns = 1
    t.split = "__unit"
    return t


def typed(cats, adj, family=0):
    atom = np.zeros(SF.MAX_NODES, dtype=np.int64)
    pair = np.zeros(SF.N_PAIRS, dtype=np.int64)
    for i, c in enumerate(cats):
        atom[i] = c
    for (i, j), b in adj.items():
        pair[PAIR_K[(i, j)]] = b
    return atom, pair


def edges(atom, pair, length):
    return [(i, j, int(pair[PAIR_K[(i, j)]])) for (i, j) in SF.PAIR_INDEX
            if j < length and int(pair[PAIR_K[(i, j)]]) != 0]


def branch(mode, double=True):
    b = SF.AtomicBranch(mode)
    b.reinit(SEED)
    return b.double() if double else b


def content(mode, atom, pair, length):
    b = branch(mode)
    with torch.no_grad():
        return b._group_content(make_table(atom, pair, length, 0),
                                torch.zeros(1, dtype=torch.long), length)[0]


def brute(atom, pair, length, q, kk, mode):
    Cc, R = q.shape[1], q.shape[2]
    out = np.ones((Cc, R)) if mode == "N" else np.zeros((Cc, R))
    es = edges(atom, pair, length)
    for c in range(Cc):
        for s in range(R):
            if mode == "N":
                for (i, j, b) in es:
                    if i == 0:
                        out[c, s] *= sum(q[atom[j], c, t] * kk[b - 1, c, s, t]
                                         for t in range(R))
                    else:
                        out[c, s] *= sum(q[atom[i], c, u] * q[atom[j], c, v]
                                         * kk[b - 1, c, u, v]
                                         for u in range(R) for v in range(R))
            else:
                tot = 0.0
                for assign in itertools.product(range(R), repeat=length - 1):
                    w = 1.0
                    for i in range(1, length):
                        w *= q[atom[i], c, assign[i - 1]]
                    for (i, j, b) in es:
                        si = s if i == 0 else assign[i - 1]
                        sj = s if j == 0 else assign[j - 1]
                        w *= kk[b - 1, c, si, sj]
                    tot += w
                out[c, s] = tot
    return out


@pytest.mark.parametrize("mode", SF.MODES)
@pytest.mark.parametrize("length", [4, 5])
def test_contraction_matches_bruteforce(mode, length):
    if length == 4:
        atom, pair = typed([6, 7, 8, 9], {(0, 1): 1, (0, 2): 2, (0, 3): 1, (1, 2): 3})
    else:
        atom, pair = typed([6, 7, 8, 9, 10],
                           {(0, 1): 1, (0, 2): 2, (0, 3): 1, (0, 4): 3,
                            (1, 2): 3, (2, 3): 1, (3, 4): 2})
    b = branch("T")
    with torch.no_grad():
        q = torch.exp(b.state.q_log()).numpy().astype(np.float64)
        kk = torch.exp(b.state.log_k()).numpy().astype(np.float64)
    vec = content(mode, atom, pair, length)
    m = max(len(edges(atom, pair, length)), 1)
    got = vec[64:128].numpy().reshape(SF.C_CH, SF.R_STATE) * m
    ref = np.log(brute(atom, pair, length, q, kk, mode))
    assert np.abs(got - ref).max() < 1e-10


def test_q_block_is_root_marginal():
    atom, pair = typed([6, 7, 8, 9], {(0, 1): 1, (0, 2): 2, (0, 3): 1})
    b = branch("T")
    with torch.no_grad():
        q = torch.exp(b.state.q_log()).numpy().astype(np.float64)
    vec = content("T", atom, pair, 4)
    assert np.abs(vec[:64].numpy() - q[atom[0]].reshape(-1)).max() < 1e-12


@pytest.mark.parametrize("mode", SF.MODES)
def test_relabelling_invariance(mode):
    atom, pair = typed([6, 7, 8, 9, 6], {(0, 1): 1, (0, 2): 2, (0, 3): 1,
                                         (0, 4): 3, (1, 2): 3, (2, 3): 1, (3, 4): 2})
    base = content(mode, atom, pair, 5)
    for order in [(0, 2, 1, 3, 4), (0, 4, 3, 2, 1), (0, 3, 1, 4, 2)]:
        new_atom = np.zeros(SF.MAX_NODES, dtype=np.int64)
        new_pair = np.zeros(SF.N_PAIRS, dtype=np.int64)
        for t, src in enumerate(order):
            new_atom[t] = atom[src]
        for (i, j), b in {(i, j): int(pair[PAIR_K[(i, j)]]) for (i, j) in SF.PAIR_INDEX
                          if j < 5}.items():
            if b:
                a, c = order.index(i), order.index(j)
                new_pair[PAIR_K[(min(a, c), max(a, c))]] = b
        assert torch.abs(content(mode, new_atom, new_pair, 5) - base).max() < 1e-9


def test_witness_h1_vs_h2():
    h1 = typed([6] * 5, {(0, 1): 1, (0, 2): 1, (0, 3): 1, (0, 4): 1,
                         (1, 2): 1, (2, 3): 1}, family=1)
    h2 = typed([6] * 5, {(0, 1): 1, (0, 2): 1, (0, 3): 1, (0, 4): 1,
                         (1, 2): 1, (3, 4): 1}, family=1)
    assert np.array_equal(h1[0], h2[0])
    assert sorted(v for v in h1[1].tolist() if v) == sorted(v for v in h2[1].tolist() if v)
    v_t1, v_t2 = content("T", *h1, 5), content("T", *h2, 5)
    v_n1, v_n2 = content("N", *h1, 5), content("N", *h2, 5)
    assert float((v_t1[64:128] - v_t2[64:128]).abs().max()) > 1e-3
    assert float((v_n1 - v_n2).abs().max()) == 0.0


def test_star_t_equals_n():
    star = typed([6] * 5, {(0, 1): 1, (0, 2): 2, (0, 3): 1, (0, 4): 3}, family=1)
    vt = content("T", *star, 5)
    vn = content("N", *star, 5)
    assert float((vt[64:128] - vn[64:128]).abs().max()) < 1e-9


@pytest.mark.parametrize("mode", SF.MODES)
def test_padding_values_are_silent(mode):
    base = typed([6, 7, 8, 9], {(0, 1): 1, (0, 2): 2, (0, 3): 1})
    t1 = make_table(base[0], base[1], 4, 0)
    garbage_atom = base[0].copy()
    garbage_pair = base[1].copy()
    garbage_atom[4] = 27
    for k, (_i, j) in enumerate(SF.PAIR_INDEX):
        if j >= 4:
            garbage_pair[k] = 3
    t2 = make_table(garbage_atom, garbage_pair, 4, 0)
    b = branch(mode)
    sel = torch.zeros(1, dtype=torch.long)
    with torch.no_grad():
        a = b._group_content(t1, sel, 4)
        c = b._group_content(t2, sel, 4)
    assert float((a - c).abs().max()) == 0.0


@pytest.mark.parametrize("mode", SF.MODES)
def test_isolated_node_is_unit_sum(mode):
    four = typed([6, 7, 8, 9], {(0, 1): 1, (0, 2): 2, (0, 3): 1})
    five = typed([6, 7, 8, 9, 3], {(0, 1): 1, (0, 2): 2, (0, 3): 1})
    v4 = content(mode, *four, 4)
    v5 = content(mode, *five, 5)
    assert float((v4[64:128] - v5[64:128]).abs().max()) < 1e-9


@pytest.mark.parametrize("mode", SF.MODES)
def test_type_paths_are_live(mode):
    base = typed([6, 7, 8, 9], {(0, 1): 1, (0, 2): 2, (0, 3): 1, (1, 2): 3})
    root_new = typed([11, 7, 8, 9], {(0, 1): 1, (0, 2): 2, (0, 3): 1, (1, 2): 3})
    leaf_new = typed([6, 7, 12, 9], {(0, 1): 1, (0, 2): 2, (0, 3): 1, (1, 2): 3})
    bond_new = typed([6, 7, 8, 9], {(0, 1): 2, (0, 2): 2, (0, 3): 1, (1, 2): 3})
    v0 = content(mode, *base, 4)
    vr = content(mode, *root_new, 4)
    vl = content(mode, *leaf_new, 4)
    vb = content(mode, *bond_new, 4)
    assert float((vr - v0).norm() / v0.norm()) > 1e-6       # q block changes
    assert float((vl[64:128] - v0[64:128]).abs().max()) > 1e-6
    assert float((vb[64:128] - v0[64:128]).abs().max()) > 1e-6


def test_parameter_count_and_isolation():
    b = SF.AtomicBranch("T")
    n = sum(p.numel() for p in b.parameters())
    assert n == SF.PHI_PARAMETERS == 12_928
    n_ = SF.AtomicBranch("N")
    assert sum(p.numel() for p in n_.parameters()) == n
    b.reinit(SEED)
    n_.reinit(SEED)
    assert SF.hash_tensors(b.state_dict()) == SF.hash_tensors(n_.state_dict())


def test_k_is_symmetric_and_bounded():
    b = branch("T")
    with torch.no_grad():
        logk = b.state.log_k()
    assert torch.equal(logk, logk.transpose(-1, -2))
    assert float(logk.abs().max()) <= 1.0     # |tanh| < 1 -> K in (exp(-1), exp(1))

"""Fast CPU tests for the ESR computation invariants (no data needed)."""

from __future__ import annotations

import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROUND = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROUND / "code"))

import esr_common as C  # noqa: E402
import esr_model as M  # noqa: E402
import esr_pairs as P  # noqa: E402


def test_relation_dimension_and_onehot():
    dist = torch.tensor([1, 2, 2, 1])
    common = torch.tensor([0, 1, 3, 2])
    bf = torch.arange(4 * 13, dtype=torch.float32).reshape(4, 13) / 100.0
    r = P.relation_features(dist, common, bf)
    assert r.shape == (4, C.REL_DIM) == (4, 19)
    assert torch.allclose(r[:, :5].sum(1), torch.ones(4))
    assert torch.equal(r[:, 1], torch.tensor([1.0, 0.0, 0.0, 1.0]))
    assert torch.equal(r[:, 2], torch.tensor([0.0, 1.0, 1.0, 0.0]))
    assert torch.allclose(r[:, 5], torch.log1p(common.float()))
    assert torch.allclose(r[:, 6:], bf)


def test_aggregation_identities_and_reference():
    theta = torch.log(torch.tensor(C.PI_INIT)).repeat(1, 1)
    # empty set -> 0
    out = M.evidence_aggregate(torch.zeros(0, 1), torch.zeros(0, dtype=torch.long), torch.zeros(3, dtype=torch.long), theta)
    assert float(out.abs().max()) == 0.0
    # single token -> identity
    out = M.evidence_aggregate(torch.tensor([[1.7]]), torch.zeros(1, dtype=torch.long), torch.ones(1, dtype=torch.long), theta)
    assert float(out[0, 0]) == pytest.approx(1.7, abs=1e-6)
    # constant set -> sqrt(m) * s (for any theta)
    theta0 = torch.zeros(1, 3)
    out = M.evidence_aggregate(torch.full((7, 1), -0.4), torch.zeros(7, dtype=torch.long), torch.tensor([7]), theta0)
    assert float(out[0, 0]) == pytest.approx(math.sqrt(7) * -0.4, abs=1e-5)
    # float64 per-graph reference
    s = torch.tensor([[0.3], [-1.2], [2.5], [0.0]])
    seg = torch.zeros(4, dtype=torch.long)
    counts = torch.tensor([4])
    got = float(M.evidence_aggregate(s, seg, counts, theta)[0, 0])
    vals = s.reshape(-1).tolist()
    mean = sum(vals) / 4
    lme = math.log(sum(math.exp(v) for v in vals) / 4)
    lme_neg = math.log(sum(math.exp(-v) for v in vals) / 4)
    ref = 2.0 * (0.5 * mean + 0.25 * lme - 0.25 * lme_neg)
    assert got == pytest.approx(ref, abs=1e-6)


def test_gaps_nonnegative():
    torch.manual_seed(0)
    s = torch.randn(500, 3, dtype=torch.float64) * 4
    seg = torch.zeros(500, dtype=torch.long)
    counts = torch.tensor([500])
    mean = M.segment_sum(s, seg, 1) / 500
    dpos = M.segment_logmeanexp(s, seg, counts) - mean
    dneg = M.segment_logmeanexp_negative(s, seg, counts) + mean
    assert float(dpos.min()) >= -1e-12
    assert float(dneg.min()) >= -1e-12


def test_molecule_pairs_degenerate_and_components():
    assert len(P.molecule_pairs(1, np.zeros((0, 2), np.int64))["u"]) == 0
    res = P.molecule_pairs(4, np.asarray([[0, 1], [1, 2]], np.int64), max_dist=2)
    pairs = list(zip(res["u"].tolist(), res["v"].tolist()))
    assert pairs == [(0, 1), (0, 2), (1, 2)]
    assert res["dist"].tolist() == [1, 2, 1]
    # cross-component pairs never appear
    assert all(not ({a, b} == {2, 3}) for a, b in pairs)
    # range 4 keeps 1..4 only
    chain = P.molecule_pairs(5, np.asarray([[0, 1], [1, 2], [2, 3], [3, 4]], np.int64), max_dist=4)
    assert int(chain["dist"].max()) == 4


def test_evidence_aggregation_noncommuting_with_projection():
    torch.manual_seed(3)
    K, T, N = 8, 4, 11
    s = torch.randn(N, K)
    w = torch.randn(T, K)
    seg = torch.zeros(N, dtype=torch.long)
    counts = torch.tensor([N])
    theta = torch.randn(K, 3)
    channel_first = M.evidence_aggregate(s, seg, counts, theta) @ w.t()
    task_space = M.evidence_aggregate(s @ w.t(), seg, counts, theta[:1].repeat(T, 1))
    assert float((channel_first - task_space).abs().max()) > 1e-3
    # the additive control commutes with a linear projection
    add_channel = M.evidence_aggregate(s, seg, counts, torch.zeros(1, 3), mode="add") @ w.t()
    add_task = M.evidence_aggregate(s @ w.t(), seg, counts, torch.zeros(1, 3), mode="add")
    assert float((add_channel - add_task).abs().max()) < 1e-5
    # a pure-mean evidence aggregator is also linear
    pure = torch.tensor([[50.0, -50.0, -50.0]] * K)
    pf = M.evidence_aggregate(s, seg, counts, pure) @ w.t()
    tf = M.evidence_aggregate(s @ w.t(), seg, counts, pure[:1].repeat(T, 1))
    assert float((pf - tf).abs().max()) < 1e-4


def test_evidence_decomposition_reconstructs():
    torch.manual_seed(4)
    s = torch.randn(23, 3) * 2
    seg = torch.zeros(23, dtype=torch.long)
    counts = torch.tensor([23])
    theta = torch.randn(3, 3)
    total = M.evidence_aggregate(s, seg, counts, theta)
    dec = M.evidence_decompose(s, seg, counts, theta)
    recon = dec["mean"] + dec["pos"] - dec["neg"]
    assert float((recon - total).abs().max()) < 1e-5

"""§6: the historical LCRC-v1 ``N`` mean field is batch-scoped.

Minimal characterisation with a **non-zero** relation head (the historical gate
used a zero-initialised head, which hides the defect):

* ``R`` aggregates per molecule -> its ``Δ_rel`` is batch-invariant;
* the historical ``N`` aggregator uses ``z.sum(0)/z.shape[0]`` over the whole
  batch -> the same molecule's ``Δ_rel`` moves with the batch it is put in;
* a corrected per-molecule reference aggregator is batch-invariant.

The middle assertion is a *characterisation test of the frozen historical code*:
it must keep failing until a new round implements a per-molecule mean.  It is not
a licence to reinterpret or retrain LCRC-v1.
"""

from __future__ import annotations

import numpy as np
import torch

import pb_common as P
import pb_batch_invariance as BI
import lc_data as LD
import lc_models as M


def _models():
    norm = P.load_norm()
    torch.manual_seed(20261030)
    m = M.LCRC("N", norm)
    m.add_relation()
    BI._nonzero_head(m, 20261030, scale=0.05)  # non-zero head: the defect is visible
    m.eval()
    return m, norm


def _delta(model, cache, rows, rows_ctx, *, corrected):
    return BI._delta_for(model, cache, rows, rows_ctx, mode="N",
                         n_cat=model.rel.n_cat, corrected=corrected)


def test_R_mode_is_batch_invariant():
    cache = P.load_cache("valid")
    model, _ = _models()
    probe = [0, 1]
    alone = BI._delta_for(model, cache, probe, probe, mode="R",
                          n_cat=model.rel.n_cat, corrected=False)
    with_ctx = BI._delta_for(model, cache, probe, list(range(0, 16)), mode="R",
                             n_cat=model.rel.n_cat, corrected=False)
    for k in probe:
        assert abs(alone[k] - with_ctx[k]) < 1e-6


def test_historical_N_is_batch_dependent_with_nonzero_head():
    cache = P.load_cache("valid")
    model, _ = _models()
    assert float(torch.linalg.vector_norm(model.rel.head.weight)) > 0.0
    probe = [0, 1]
    alone = _delta(model, cache, probe, probe, corrected=False)
    bigger = _delta(model, cache, probe, list(range(0, 32)), corrected=False)
    spread = max(abs(alone[k] - bigger[k]) for k in probe)
    assert spread > 1e-5, (
        "expected the frozen historical N aggregator to be batch-scoped; if this "
        "fails the implementation was changed and LCRC-v1's R/N interpretation "
        "must be re-derived from scratch")
    # and the defect is not merely numerical noise: it survives at 1e-3 scale
    assert spread > 1e-4


def test_corrected_per_molecule_N_is_batch_invariant():
    cache = P.load_cache("valid")
    model, _ = _models()
    probe = [0, 1]
    alone = _delta(model, cache, probe, probe, corrected=True)
    bigger = _delta(model, cache, probe, list(range(0, 32)), corrected=True)
    for k in probe:
        assert abs(alone[k] - bigger[k]) < 1e-6


def test_corrected_aggregate_matches_direct_per_molecule_mean():
    """The reference aggregator equals an explicit per-molecule mean field."""
    torch.manual_seed(3)
    n_graphs, n_cat, z_dim = 3, 2, 4
    pair_graph = torch.tensor([0, 0, 1, 2, 2, 2])
    pair_cat = torch.tensor([0, 1, 1, 0, 0, 1])
    z = torch.randn(len(pair_graph), z_dim, dtype=torch.float64)
    got = BI.per_molecule_N_aggregate(z, pair_graph, pair_cat, n_graphs, n_cat)
    got = got.view(n_graphs, n_cat, z_dim)
    for g in range(n_graphs):
        sel = pair_graph == g
        mean = z[sel].mean(0)
        for c in torch.unique(pair_cat[sel]):
            assert torch.allclose(got[g, int(c)], mean)
        empty = set(range(n_cat)) - {int(c) for c in torch.unique(pair_cat[sel])}
        for c in empty:
            assert torch.allclose(got[g, c], torch.zeros(z_dim, dtype=torch.float64))

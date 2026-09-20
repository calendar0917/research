"""Data-free correctness tests for the PSCD-SC-v0 strong port-operator reader.

These lock the reader plumbing the diagnostic depends on and need no ZINC data,
no frozen artifact and no checkpoints:

* the port-operator batch collate is **batch-invariant** (a molecule's prediction
  does not depend on which other molecules share its batch) -- the failure mode
  that contaminated PSCD-R's historical number is never re-introduced;
* the operator is finite for empty / single-port / multi-port decompositions;
* the operator is sensitive to port routing (it does not degenerate to a port
  mean);
* the corpus-level occurrence / port offsets are correct.

They use a small synthetic dictionary, so ``data_free_tests`` (which does load
the frozen ``stageA.pkl``) stays a separate, artifact-dependent gate.
"""

from __future__ import annotations

import numpy as np
import pytest

from tracks.ksvd.code.run_pscd_compositional_dictionary_audit import (
    StructuredCode,
    Vocabulary,
)
from tracks.ksvd.code.run_pscd_reader_diagnostic import make_structural_encoder
from tracks.ksvd.code.run_pscd_strong_reader import (
    collate_portop,
    make_port_operator,
    motif_node_offsets,
    build_tau_index,
    precompute_graph,
)

torch = pytest.importorskip("torch")


def _vocab():
    v = Vocabulary()
    v.add_singleton(0)
    v.add_singleton(1)
    # mid 2: 2-atom chain; mid 3: 3-atom chain (single bond type)
    v.add(b"e00", 2, (0, 0), ((0, 1, 1),), False)
    v.add(b"c000", 3, (0, 0, 0), ((0, 1, 1), (1, 2, 1)), False)
    return v


def _codes(v):
    e, c = 2, 3
    a = StructuredCode(4, 3, [c, 0, 1], [(0, 0, 1, 1, 0), (0, 1, 1, 2, 0)])
    b = StructuredCode(6, 5, [c, c], [(0, 1, 1, 1, 0)])
    cc = StructuredCode(3, 2, [e, 1], [(0, 0, 1, 1, 0)])
    return [a, b, cc]


def _model(v, intra=True):
    atom_index = {0: 0, 1: 1}
    bond_index = {1: 0}
    torch.manual_seed(0)
    enc = make_structural_encoder(v, atom_index, bond_index, d=8, layers=2)
    node_off = motif_node_offsets(v)
    tau = build_tau_index(v, node_off)
    return make_port_operator(enc, n_bond=1, tau_index=tau, d=8, layers=2,
                              intra_transfer=intra).eval(), node_off, tau


def _fwd(model, pre):
    with torch.no_grad():
        return model(collate_portop(pre, "cpu"))


def test_port_operator_batch_invariant_and_finite():
    v = _vocab()
    model, node_off, tau = _model(v)
    pre = [precompute_graph(sc, v, tau, node_off, {1: 0}) for sc in _codes(v)]
    batched = _fwd(model, pre)
    singles = torch.stack([_fwd(model, [it])[0] for it in pre])
    assert torch.allclose(batched, singles, atol=1e-5)
    assert torch.isfinite(batched).all()


def test_no_intra_transfer_batch_invariant():
    v = _vocab()
    model, node_off, tau = _model(v, intra=False)
    pre = [precompute_graph(sc, v, tau, node_off, {1: 0}) for sc in _codes(v)]
    batched = _fwd(model, pre)
    singles = torch.stack([_fwd(model, [it])[0] for it in pre])
    assert torch.allclose(batched, singles, atol=1e-5)


def test_empty_single_multi_port_finite():
    v = _vocab()
    model, node_off, tau = _model(v)
    empty = StructuredCode(3, 2, [0, 1, 2], [])
    one = StructuredCode(2, 1, [2, 1], [(0, 0, 1, 1, 0)])
    multi = StructuredCode(4, 3, [3, 0, 1, 2],
                           [(0, 0, 1, 1, 0), (0, 1, 1, 2, 0), (0, 2, 1, 3, 0)])
    for sc in (empty, one, multi):
        out = _fwd(model, [precompute_graph(sc, v, tau, node_off, {1: 0})])
        assert torch.isfinite(out).all()


def test_port_routing_sensitivity():
    v = _vocab()
    model, node_off, tau = _model(v)
    # same motif ids (chain 3 + singleton); the chain port rides canonical slot
    # 0 vs the (inequivalent) middle slot 1 -> the output must differ.
    g0 = StructuredCode(4, 3, [3, 1], [(0, 0, 1, 1, 0)])
    g2 = StructuredCode(4, 3, [3, 1], [(0, 1, 1, 1, 0)])
    o0 = _fwd(model, [precompute_graph(g0, v, tau, node_off, {1: 0})])
    o2 = _fwd(model, [precompute_graph(g2, v, tau, node_off, {1: 0})])
    assert float((o0 - o2).abs().max()) > 1e-8


def test_collate_occurrence_and_port_offsets():
    v = _vocab()
    _, node_off, tau = _model(v)
    codes = _codes(v)
    pre = [precompute_graph(sc, v, tau, node_off, {1: 0}) for sc in codes]
    batch = collate_portop(pre, "cpu")
    m_tot = sum(it["n_occ"] for it in pre)
    p_tot = sum(it["n_port"] for it in pre)
    assert int(batch["occ_ids"].numel()) == m_tot
    assert int(batch["port_occ"].numel()) == p_tot
    assert int(batch["port_occ"].max()) < m_tot
    assert int(batch["ext_src"].max()) < p_tot
    assert int(batch["intra_q"].max()) < p_tot
    # pooling covers every occurrence exactly once
    assert int(batch["pool_index"].numel()) == m_tot
    assert sorted(batch["pool_index"].tolist()) == sorted(
        int(g) for g, it in enumerate(pre) for _ in range(it["n_occ"]))

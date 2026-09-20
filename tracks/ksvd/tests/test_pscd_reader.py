"""Data-free correctness tests for the PSCD-R-v0 reader diagnostic.

These lock the *reader plumbing* the diagnostic depends on and need no ZINC
data, no checkpoints and no results artifacts:

* the corrected composition collate (``collate_comp_fixed``) makes the PSCD-v0
  opacity reader **batch-invariant** (a molecule's prediction does not depend on
  which other molecules share its batch), while the PSCD-v0 as-shipped collate
  is not;
* the structured-static (R1) and port-resolved (R2) collates and readers are
  batch-invariant and produce finite, correctly-shaped predictions;
* the structured dictionary encoder exposes ``h_k = sum_v z_v`` and ``z_p`` and
  its motif-sum equals the node-sum, so the port-mean ablation is well defined.
"""

from __future__ import annotations

import numpy as np
import pytest

from tracks.ksvd.code.run_pscd_compositional_dictionary_audit import (
    MAX_MOTIF,
    StructuredCode,
    Vocabulary,
    make_module,
)
from tracks.ksvd.code.run_pscd_reader_diagnostic import (
    collate_comp_fixed,
    collate_r1,
    collate_r2,
    make_r1_reader,
    make_r2_reader,
    make_structural_encoder,
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
    e = 2
    c = 3
    # graph A: chain motif + 2 singletons, ports on chain slots 0 and 1
    a = StructuredCode(4, 3, [c, 0, 1], [(0, 0, 1, 1, 0), (0, 1, 1, 2, 0)])
    # graph B: two chain motifs joined
    b = StructuredCode(6, 5, [c, c], [(0, 1, 1, 1, 0)])
    # graph C: edge motif + singleton
    cc = StructuredCode(3, 2, [e, 1], [(0, 0, 1, 1, 0)])
    return [a, b, cc]


def _batch_consistency(forward, codes, builder):
    torch.manual_seed(0)
    with torch.no_grad():
        batch_pred = forward(builder(codes))
        singles = [float(forward(builder([c]))) for c in codes]
    return [float(x) for x in batch_pred], singles


def test_corrected_composition_collate_is_batch_invariant():
    v = _vocab()
    codes = _codes(v)
    bond_index = {1: 0}
    torch.manual_seed(0)
    model = make_module("comp", len(v), 1, MAX_MOTIF).eval()
    batch, singles = _batch_consistency(
        model, codes, lambda cs: collate_comp_fixed(cs, v, bond_index, "cpu"))
    assert np.allclose(batch, singles, atol=1e-5)


def test_shipped_composition_collate_is_not_batch_invariant():
    """Documents the PSCD-v0 defect that motivated R0 (corrected)."""
    from tracks.ksvd.code.run_pscd_compositional_dictionary_audit import collate
    v = _vocab()
    codes = _codes(v)
    bond_index = {1: 0}
    torch.manual_seed(0)
    model = make_module("comp", len(v), 1, MAX_MOTIF).eval()
    batch, singles = _batch_consistency(
        model, codes, lambda cs: collate(cs, v, None, bond_index, "comp", "cpu"))
    assert not np.allclose(batch, singles, atol=1e-4)


def test_r1_r2_readers_batch_invariant_and_finite():
    v = _vocab()
    codes = _codes(v)
    bond_index = {1: 0}
    atom_index = {0: 0, 1: 1}
    for maker, collate_fn in ((make_r1_reader, collate_r1), (make_r2_reader, collate_r2)):
        torch.manual_seed(0)
        enc = make_structural_encoder(v, atom_index, bond_index)
        model = maker(enc, n_bond=1).eval()
        batch, singles = _batch_consistency(
            model, codes, lambda cs: collate_fn(cs, v, bond_index, "cpu"))
        assert np.allclose(batch, singles, atol=1e-5)
        assert np.all(np.isfinite(batch))


def test_encoder_motif_sum_and_port_mean():
    v = _vocab()
    atom_index = {0: 0, 1: 1}
    bond_index = {1: 0}
    torch.manual_seed(0)
    enc = make_structural_encoder(v, atom_index, bond_index).eval()
    with torch.no_grad():
        h_dict, x = enc()
    # h_dict[k] == sum of z over k's canonical vertices
    off = enc.node_off.tolist()
    size = [int(s) for s in enc.motif_size.tolist()]
    for k in range(len(v)):
        z = x[off[k]:off[k] + size[k]]
        assert torch.allclose(h_dict[k], z.sum(0), atol=1e-5)
    # port-mean ablation uses the well-defined mean z
    mean_port = h_dict / enc.motif_size.unsqueeze(-1)
    assert torch.isfinite(mean_port).all()

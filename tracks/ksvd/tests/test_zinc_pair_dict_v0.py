"""Targeted tests for PSD-v0 (pair-structural encoder + function-preserving dictionary).

Fresh-clone-safe: works on tiny synthetic molecules in-process, no ZINC data,
no checkpoints, CPU only.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tracks.ksvd.code.run_aiom_representation_audit import Mol, permute_mol
from tracks.ksvd.code import run_zinc_pair_dict_v0 as psd


def _synth_mol(seed: int, n: int = 8) -> Mol:
    rng = np.random.RandomState(seed)
    node_types = rng.randint(0, 4, size=n).astype(np.int64)
    bonds = []
    for v in range(1, n):
        u = int(rng.randint(0, v))
        bonds.append((min(u, v), max(u, v)))
    bond_types = rng.randint(0, 2, size=len(bonds)).astype(np.int64)
    return Mol(n=n, bonds=bonds, bond_types=bond_types, node_types=node_types)


def _fixtures():
    mols = [_synth_mol(s) for s in range(5)]
    ai, bi = psd.category_index(mols)
    n_atom = max(ai.values()) + 1
    n_bond = max(bi.values()) + 1
    return mols, ai, bi, n_atom, n_bond


def test_permutation_equivariance_and_prediction_invariance():
    mols, ai, bi, n_atom, n_bond = _fixtures()
    torch.manual_seed(0)
    model = psd.PairEncoder.build(n_atom, n_bond)
    model.eval()
    mol = mols[0]
    perm = np.random.RandomState(3).permutation(mol.n)
    eb = np.random.RandomState(4).permutation(mol.m)
    molp = permute_mol(mol, perm, eb)
    b = psd.build_pair_batch([mol], ai, bi, "cpu")
    bp = psd.build_pair_batch([molp], ai, bi, "cpu")
    with torch.no_grad():
        sg = model.forward_layer_states(b)
        sp = model.forward_layer_states(bp)
        pg = float(model(b)[0])
        pp = float(model(bp)[0])
    for hg, hp in zip(sg, sp):
        hgn = hg.numpy()[0]
        hpn = hp.numpy()[0]
        idx = np.ix_(perm, perm)
        assert np.abs(hpn[idx] - hgn).max() < 1e-5
    assert abs(pg - pp) < 1e-5


def test_padding_and_graph_order_invariance():
    mols, ai, bi, n_atom, n_bond = _fixtures()
    torch.manual_seed(0)
    model = psd.PairEncoder.build(n_atom, n_bond)
    model.eval()
    mol = mols[0]
    with torch.no_grad():
        pg = float(model(psd.build_pair_batch([mol], ai, bi, "cpu"))[0])
        outpad = model(psd.build_pair_batch([mol, mols[3], mols[4]], ai, bi, "cpu")).numpy()
        outorder = model(psd.build_pair_batch([mols[1], mol, mols[2]], ai, bi, "cpu")).numpy()
        bsingle = psd.build_pair_batch(mols[:4], ai, bi, "cpu")
        outsingle = model(bsingle).numpy()
    assert abs(outpad[0] - pg) < 1e-6
    assert abs(outorder[1] - pg) < 1e-5
    b0 = psd.build_pair_batch([mols[0]], ai, bi, "cpu")
    with torch.no_grad():
        o0 = float(model(b0)[0])
    assert abs(outsingle[0] - o0) < 1e-5


def test_gradients_reach_all_structural_layers():
    mols, ai, bi, n_atom, n_bond = _fixtures()
    torch.manual_seed(0)
    model = psd.PairEncoder.build(n_atom, n_bond)
    b = psd.build_pair_batch(mols, ai, bi, "cpu")
    y = torch.tensor([0.0, 1.0, 0.5, -0.5, 2.0])
    loss = torch.nn.functional.l1_loss(model(b), y)
    needed = ("E_A", "E_R", "W_init", ".phi", ".psi", ".eta", "W_read", "head")
    for name, p in model.named_parameters():
        if any(k in name for k in needed):
            g = torch.autograd.grad(loss, p, retain_graph=True, allow_unused=True)[0]
            assert g is not None and float(g.norm()) > 0.0, name


def test_no_absolute_position_parameter():
    _, _, _, n_atom, n_bond = _fixtures()
    model = psd.PairEncoder.build(n_atom, n_bond)
    assert not any(("pos" in n or "node_id" in n) for n, _ in model.named_parameters())


def test_pair_params_target_and_raw_match():
    # use the real ZINC category counts for the canonical parameter identity
    n_atom, n_bond = 21, 3
    p_pair = psd._param_breakdown(psd.build_model("pair", n_atom, n_bond))["total_trainable"]
    p_raw = psd._param_breakdown(psd.build_model("raw", n_atom, n_bond))["total_trainable"]
    p_wide = psd._param_breakdown(psd.build_model("raw_wide", n_atom, n_bond))["total_trainable"]
    assert p_raw == 55_681
    # raw_wide is parameter-matched to the pair encoder within 10 %
    assert abs(p_wide - p_pair) / p_pair <= 0.10, (p_pair, p_wide)


def test_stage0_all_passed_cpu():
    res = psd.stage0_tests("cpu", log=lambda *_: None)
    assert res["all_passed"], res["pair"]


def test_synth_mechanism_c6_vs_two_triangles():
    res = psd.synth_mechanism("cpu", log=lambda *_: None)
    assert res["raw_node_state_multiset_gap"] < 1e-4
    assert res["pair_state_projection_gap"] > 1e-3
    assert res["graph_representation_gap"] > 1e-3


@pytest.mark.parametrize("kind", ["dense", "learned_dict", "fixed_identity", "generic_topk"])
def test_insertion_is_function_preserving(kind):
    """At insertion (lambda=0, k=d, D=I) the coding is the identity."""
    mols, ai, bi, n_atom, n_bond = _fixtures()
    torch.manual_seed(0)
    enc = psd.PairEncoder.build(n_atom, n_bond)
    head = enc.head
    b = psd.build_pair_batch(mols, ai, bi, "cpu")
    enc.eval()
    with torch.no_grad():
        base = enc(b)
        model = psd.GraphCodeModel.build(enc, head, psd.D_MODEL, kind, k_active=psd.D_MODEL)
        model.eval()
        after = model(b, lam=0.0)
    assert torch.allclose(base, after, atol=1e-5), (base - after).abs().max().item()


def test_soft_threshold_zero_identity_and_sparsity():
    x = torch.randn(4, 8)
    assert torch.allclose(psd._soft(x, 0.0), x)
    out = psd._soft(x, 0.5)
    assert int((out != 0).sum()) < int((x != 0).sum())

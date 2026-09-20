"""Targeted tests for GSCN-v0 (raw-input sparse dictionary-core graph network).

Fresh-clone-safe: builds tiny synthetic molecules in-process, no ZINC data, no
checkpoints, CPU only.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

from tracks.ksvd.code.run_aiom_representation_audit import Mol, permute_mol
from tracks.ksvd.code import run_gscn_v0 as g


def _synth_mol(seed: int, n: int = 7) -> Mol:
    rng = np.random.RandomState(seed)
    node_types = rng.randint(0, 5, size=n).astype(np.int64)
    bonds = []
    for v in range(1, n):
        u = int(rng.randint(0, v))
        bonds.append((min(u, v), max(u, v)))
    bond_types = rng.randint(0, 3, size=len(bonds)).astype(np.int64)
    return Mol(n=n, bonds=bonds, bond_types=bond_types, node_types=node_types)


def _fixtures():
    mols = [_synth_mol(s) for s in range(4)]
    ai, bi = g.category_index(mols)
    return mols, ai, bi


@pytest.mark.parametrize("arm", ["bnull_raw", "generic", "nothresh", "sparse"])
def test_permutation_equivariance_and_batch_invariance(arm):
    mols, ai, bi = _fixtures()
    n_atom = max(ai.values()) + 1
    n_bond = max(bi.values()) + 1
    torch.manual_seed(0)
    model = g.build_model(arm, n_atom, n_bond)
    if arm == "sparse":
        g.calibrate_lambdas(model, mols, ai, bi, "cpu", log=lambda *_: None)
    model.eval()
    mol = mols[0]
    perm = np.random.RandomState(3).permutation(mol.n)
    eb = np.random.RandomState(4).permutation(mol.m)
    molp = permute_mol(mol, perm, eb)
    b = g.build_raw_batch([mol], ai, bi, "cpu")
    bp = g.build_raw_batch([molp], ai, bi, "cpu")
    with torch.no_grad():
        sg = model.forward_layer_states(b)
        sp = model.forward_layer_states(bp)
        pg = float(model(b)[0])
        pp = float(model(bp)[0])
    for hg, hp in zip(sg, sp):
        diff = hp[perm].numpy() - hg.numpy()
        assert np.abs(diff).max() < 1e-5
    assert abs(pg - pp) < 1e-5

    # batching invariance
    b2 = g.build_raw_batch(mols[:3], ai, bi, "cpu")
    with torch.no_grad():
        out2 = model(b2)
    assert abs(float(out2[0]) - pg) < 1e-5


@pytest.mark.parametrize("arm", ["nothresh", "sparse"])
def test_code_dims_nonneg_and_dictionary_gradient(arm):
    mols, ai, bi = _fixtures()
    n_atom = max(ai.values()) + 1
    n_bond = max(bi.values()) + 1
    torch.manual_seed(0)
    model = g.build_model(arm, n_atom, n_bond)
    if arm == "sparse":
        g.calibrate_lambdas(model, mols, ai, bi, "cpu", log=lambda *_: None)
    assert float(model.lambdas.min()) > 0.0 if arm == "sparse" \
        else float(model.lambdas.abs().max()) == 0.0
    b = g.build_raw_batch(mols, ai, bi, "cpu")
    y = torch.tensor([0.0, 1.0, 0.5, 2.0])
    cap = {"Y": [], "C": [], "A": [], "DN": []}
    pred = model(b, capture=cap)
    loss = torch.nn.functional.l1_loss(pred, y)
    for li in range(model.layers):
        gD = torch.autograd.grad(loss, model.D[li], retain_graph=True)[0]
        assert float(gD.norm()) > 0.0
        A = cap["A"][li]
        assert A.shape[1] == model.K
        assert A.shape[0] == cap["Y"][li].shape[0]
        assert cap["Y"][li].shape[1] == model.d
        assert float(A.min()) >= -1e-7
        assert (A @ cap["DN"][li]).shape == cap["Y"][li].shape


def test_no_hidden_bypass():
    mols, ai, bi = _fixtures()
    n_atom = max(ai.values()) + 1
    n_bond = max(bi.values()) + 1
    torch.manual_seed(0)
    model = g.build_model("sparse", n_atom, n_bond)
    g.calibrate_lambdas(model, mols, ai, bi, "cpu", log=lambda *_: None)
    b = g.build_raw_batch(mols, ai, bi, "cpu")
    model.eval()
    with torch.no_grad():
        c0 = {"Y": [], "C": [], "A": [], "DN": []}
        model(b, capture=c0)
        y0 = c0["A"][0] @ c0["DN"][0]
        for p in model.D:
            p.data.add_(torch.randn_like(p) * 0.1)
        c1 = {"Y": [], "C": [], "A": [], "DN": []}
        model(b, capture=c1)
        y1 = c1["A"][0] @ c1["DN"][0]
    assert float((y1 - y0).abs().mean()) > 1e-3


def test_generic_param_match():
    mols, ai, bi = _fixtures()
    n_atom = max(ai.values()) + 1
    n_bond = max(bi.values()) + 1
    p_gscn = g.param_breakdown(g.build_model("sparse", n_atom, n_bond))["total_trainable"]
    p_gen = g.param_breakdown(g.build_model("generic", n_atom, n_bond))["total_trainable"]
    assert abs(p_gen - p_gscn) / p_gscn <= 0.10
    assert g.generic_width(64) in (63, 64)


def test_ablation_shapes_and_signal():
    mols, ai, bi = _fixtures()
    n_atom = max(ai.values()) + 1
    n_bond = max(bi.values()) + 1
    torch.manual_seed(0)
    model = g.build_model("sparse", n_atom, n_bond)
    g.calibrate_lambdas(model, mols, ai, bi, "cpu", log=lambda *_: None)
    y = np.random.RandomState(0).rand(len(mols))
    res = g.run_ablations(model, mols, y, ai, bi, "cpu", log=lambda *_: None)
    # atom permutation is a sanity invariance
    assert abs(res["delta_atom_permutation"]) < 1e-5
    # destruction / shuffle / mean-code must change the prediction path
    assert res["delta_random_dictionary"] > 1e-6
    assert abs(res["delta_mean_code"]) > 1e-6
    assert abs(res["delta_code_shuffle"]) > 1e-6

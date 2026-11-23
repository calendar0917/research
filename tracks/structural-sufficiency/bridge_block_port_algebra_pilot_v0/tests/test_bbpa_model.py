from __future__ import annotations

import numpy as np
import torch

import bbpa_data as D
import bbpa_model as M
import bbpa_synthetic as Y
import bbpa_tensor as T


def _model(seed: int = 0):
    atom_dims, bond_dims = D.atom_bond_dims()
    model, hashes = M.build_model(atom_dims, bond_dims, seed, {"dims": {}})
    model.eval()
    return model, hashes


def _batch(graphs):
    raw = Y.combine(graphs)
    mols = T.build_dataset(raw, list(range(len(graphs))))
    return raw, mols, T.collate(list(range(len(graphs))), mols, raw, "cpu")


def test_parameter_count_and_identical_init():
    atom_dims, bond_dims = D.atom_bond_dims()
    hashes = []
    ledgers = []
    for _ in M.MODES:
        model, h = M.build_model(atom_dims, bond_dims, 0, {"dims": {}})
        hashes.append(h)
        ledgers.append(M.parameter_ledger(model))
    assert hashes[0] == hashes[1] == hashes[2]
    assert ledgers[0] == ledgers[1] == ledgers[2]
    assert 100_000 <= ledgers[0]["total"] <= 500_000


def test_forward_modes_and_permutation_invariance():
    model, _ = _model()
    graphs = [Y.benzene(), Y.ethane(), Y.chain4(), Y.star4(),
              Y.concatenated_chain(4), Y.disubstituted_benzene(2)]
    raw, mols, batch = _batch(graphs)
    with torch.no_grad():
        full = {mode: model.forward_batch(batch, mode)["yhat_std"].numpy()
                for mode in M.MODES}
        single = []
        for i in range(len(graphs)):
            bi = T.collate([i], mols, raw, "cpu")
            single.append(float(model.forward_batch(bi, "TREE_PORT")["yhat_std"][0]))
    assert np.allclose(full["TREE_PORT"], np.asarray(single), atol=1e-5, rtol=0)
    for mode in M.MODES:
        assert np.isfinite(full[mode]).all()


def test_tree_vs_local_on_chain_and_star():
    model, _ = _model()
    # four-block chain: recursion must change the computation
    raw, mols, batch = _batch([Y.chain4()])
    with torch.no_grad():
        tr = model.forward_batch(batch, "TREE_PORT", return_states=True)
        lo = model.forward_batch(batch, "LOCAL_PORT", return_states=True)
    assert (tr["states"] - lo["states"]).norm(dim=1).max() > 1e-6
    assert (tr["q"] - lo["q"]).abs().max() > 1e-6
    # star: both computations coincide exactly (leaf states are local)
    raw, mols, batch = _batch([Y.star4()])
    with torch.no_grad():
        tr = model.forward_batch(batch, "TREE_PORT", return_states=True)
        lo = model.forward_batch(batch, "LOCAL_PORT", return_states=True)
    assert (tr["states"] - lo["states"]).abs().max() <= 1e-7
    assert abs(float(tr["yhat_std"][0] - lo["yhat_std"][0])) <= 1e-7


def test_isomer_port_distance_and_encoding():
    graphs = [Y.disubstituted_benzene(p) for p in (1, 2, 3)]
    raw, mols, batch = _batch(graphs)
    model, _ = _model()
    with torch.no_grad():
        out = model.forward_batch(batch, "TREE_PORT", return_states=True)
    u = out["u_port"]
    # first four ports belong to ortho, next four to meta, last four to para
    d_ortho_meta = (u[0:4].unsqueeze(1) - u[4:8].unsqueeze(0)).abs().max()
    d_meta_para = (u[4:8].unsqueeze(1) - u[8:12].unsqueeze(0)).abs().max()
    assert float(d_ortho_meta) > 1e-6
    assert float(d_meta_para) > 1e-6
    assert float(out["yhat_std"].max() - out["yhat_std"].min()) > 0.0


def test_finite_backward_and_key_gradients():
    graphs = [Y.benzene(), Y.ethane(), Y.chain4(), Y.star4(),
              Y.concatenated_chain(4), Y.disubstituted_benzene(2)]
    raw, mols, batch = _batch(graphs)
    atom_dims, bond_dims = D.atom_bond_dims()
    for mode in M.MODES:
        model, _ = M.build_model(atom_dims, bond_dims, 0, {"dims": {}})
        model.train()
        out = model.forward_batch(batch, mode)
        loss = ((out["yhat_std"] - torch.ones_like(out["yhat_std"])) ** 2).mean()
        loss.backward()
        groups = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                assert torch.isfinite(p.grad).all()
                groups[name.split(".")[0]] = groups.get(name.split(".")[0], 0.0) + float(p.grad.norm())
        required = ["atom_emb", "bond_emb", "block_mlp", "readout_mlp", "beta"]
        if mode != "BLOCK_BAG":
            required += ["port_mlp", "message_mlp", "update_mlp", "norm", "q_mlp"]
        for key in required:
            assert groups.get(key, 0.0) > 0.0, (mode, key)


def test_no_bridge_molecule_and_multi_component_forward():
    # benzene has no bridge; the two-component forest has one bridge
    graphs = [Y.benzene(), Y.ethane()]
    raw = Y.combine(graphs)
    mols = T.build_dataset(raw, [0, 1])
    batch = T.collate([0, 1], mols, raw, "cpu")
    model, _ = _model()
    with torch.no_grad():
        out = model.forward_batch(batch, "TREE_PORT", return_states=True)
    assert torch.isfinite(out["yhat_std"]).all()
    assert out["q"].shape[0] == 1  # only ethane contributes a bridge


def test_zero_unseen_keeps_finite_outputs():
    model, _ = _model()
    graphs = [Y.chain4()]
    raw, mols, batch = _batch(graphs)
    atom_dims, bond_dims = D.atom_bond_dims()
    atom_ok = [np.ones(d, bool) for d in atom_dims]
    bond_ok = [np.ones(d, bool) for d in bond_dims]
    # zero all but the categories used by the synthetic graph
    for i in range(len(atom_ok)):
        atom_ok[i][:] = False
        atom_ok[i][0] = True
    for i in range(len(bond_ok)):
        bond_ok[i][:] = False
        bond_ok[i][0] = True
    zeroed = model.zero_unseen(atom_ok, bond_ok)
    assert sum(zeroed.values()) > 0
    model.eval()
    with torch.no_grad():
        out = model.forward_batch(batch, "TREE_PORT")
    assert torch.isfinite(out["yhat_std"]).all()
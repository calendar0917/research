from __future__ import annotations

import math

import numpy as np
import pytest
import torch
import torch.nn.functional as F

import common as C
import gates as G
import model as M


def test_parameters_and_init(spec, norm):
    for seed in C.SEEDS:
        r, ri = M.build_model("R", spec, norm, seed, "cpu")
        n, ni = M.build_model("N", spec, norm, seed, "cpu")
        assert ri["parameters"] == ni["parameters"] == C.TOTAL_PARAMS_REFERENCE
        assert ri["parameters_new"] == ni["parameters_new"] == C.NEW_PARAMS
        assert ri["trunk_state_hash"] == C.EXPECTED_TRUNK_INIT_HASH[seed]
        assert ri["state_hash"] == ni["state_hash"]
        assert torch.equal(r.edge_score.weight, torch.zeros_like(r.edge_score.weight))
        assert torch.equal(r.edge_score.bias, torch.zeros_like(r.edge_score.bias))


def test_initial_predictions_equal_trunk(spec, norm, batch):
    r, _ = M.build_model("R", spec, norm, 0, "cpu")
    n, _ = M.build_model("N", spec, norm, 0, "cpu")
    r.eval(); n.eval()
    with torch.no_grad():
        yb = r.trunk(batch)
        yr = r(batch)
        yn = n(batch)
        _, aux = r(batch, return_aux=True)
    assert float((yb - yr).abs().max()) <= 1e-6
    assert float((yb - yn).abs().max()) <= 1e-6
    assert float(aux["delta_bond"].abs().max()) == 0.0


def test_meanfield_bruteforce():
    torch.manual_seed(0)
    z = torch.randn(8, 4)
    edges = [(0, 1), (1, 2), (3, 4), (4, 5), (5, 6)]
    types = [1, 1, 1, 2, 2]
    graphs = [0, 0, 0, 1, 1]
    batch = {
        "bond_index": torch.tensor(edges),
        "bond_type": torch.tensor(types),
        "bond_batch": torch.tensor(graphs),
        "n_graphs": 2,
    }
    got = M.meanfield_pair_products(z, batch, 5)
    for i, ((u, v), t, g) in enumerate(zip(edges, types, graphs)):
        stubs = []
        for (a, b), tt, gg in zip(edges, types, graphs):
            if tt == t and gg == g:
                stubs += [z[a], z[b]]
        expected = torch.stack([stubs[a] * stubs[b] for a in range(len(stubs)) for b in range(a + 1, len(stubs))]).mean(0)
        assert float((got[i] - expected).abs().max()) <= 1e-6


def test_meanfield_M1_and_groups():
    z = torch.randn(6, 3)
    b1 = {"bond_index": torch.tensor([[0, 1]]), "bond_type": torch.tensor([1]), "bond_batch": torch.tensor([0]), "n_graphs": 1}
    assert float((M.meanfield_pair_products(z, b1, 5)[0] - z[0] * z[1]).abs().max()) <= 1e-6
    b2 = {"bond_index": torch.tensor([[0, 1], [2, 3]]), "bond_type": torch.tensor([1, 1]), "bond_batch": torch.tensor([0, 1]), "n_graphs": 2}
    j = M.meanfield_pair_products(z, b2, 5)
    assert float((j[0] - z[0] * z[1]).abs().max()) <= 1e-6
    assert float((j[1] - z[2] * z[3]).abs().max()) <= 1e-6
    b3 = {"bond_index": torch.tensor([[0, 1], [2, 3]]), "bond_type": torch.tensor([1, 2]), "bond_batch": torch.tensor([0, 0]), "n_graphs": 1}
    j3 = M.meanfield_pair_products(z, b3, 5)
    assert float((j3[0] - z[0] * z[1]).abs().max()) <= 1e-6
    assert float((j3[1] - z[2] * z[3]).abs().max()) <= 1e-6


def test_relation_sensitivity():
    z = torch.tensor([[1.0, 0.5], [2.0, -0.5], [3.0, 1.5], [4.0, -1.5]])
    a = {"bond_index": torch.tensor([[0, 1], [2, 3]]), "bond_type": torch.tensor([1, 1]), "bond_batch": torch.tensor([0, 0]), "n_graphs": 1}
    b = {"bond_index": torch.tensor([[0, 2], [1, 3]]), "bond_type": torch.tensor([1, 1]), "bond_batch": torch.tensor([0, 0]), "n_graphs": 1}
    jr_a = z[torch.tensor([0, 2])] * z[torch.tensor([1, 3])]
    jr_b = z[torch.tensor([0, 1])] * z[torch.tensor([2, 3])]
    jn_a = M.meanfield_pair_products(z, a, 5)
    jn_b = M.meanfield_pair_products(z, b, 5)
    assert float((jr_a - jr_b).abs().max()) > 1e-6
    assert float((jn_a - jn_b).abs().max()) <= 1e-6


def test_invariance(spec, caches, batch):
    r, _ = M.build_model("R", spec, C.load_frozen_norm(), 0, "cpu")
    n, _ = M.build_model("N", spec, C.load_frozen_norm(), 0, "cpu")
    r.eval(); n.eval()
    single = D_collate(caches, [0])
    perm = torch.randperm(int(single["atom_feat"].shape[0]), generator=torch.Generator().manual_seed(3))
    permuted = G._permute_atoms(single, perm)
    eperm = torch.randperm(int(batch["bond_index"].shape[0]), generator=torch.Generator().manual_seed(4))
    reordered = G._permute_bonds(batch, eperm)
    which = torch.zeros(int(batch["bond_index"].shape[0]), dtype=torch.bool)
    which[::2] = True
    swapped = G._swap_endpoints(batch, which)
    with torch.no_grad():
        for m in (r, n):
            assert float((m(single) - m(permuted)).abs().max()) <= 1e-5
            assert float((m(batch) - m(reordered)).abs().max()) <= 1e-5
            assert float((m(batch) - m(swapped)).abs().max()) <= 1e-5


def test_batch_split_merge(spec, caches):
    r, _ = M.build_model("R", spec, C.load_frozen_norm(), 0, "cpu")
    r.eval()
    full = D_collate(caches, list(range(8)))
    left = D_collate(caches, list(range(4)))
    right = D_collate(caches, list(range(4, 8)))
    with torch.no_grad():
        assert float((r(full) - torch.cat([r(left), r(right)])).abs().max()) <= 1e-5


def test_optimization_path(spec, norm, batch):
    for arm in C.ARMS:
        m, info = M.build_model(arm, spec, norm, 0, "cpu")
        m.train()
        opt = torch.optim.AdamW(m.parameters(), lr=C.LR, weight_decay=C.WEIGHT_DECAY)
        target = torch.linspace(-1, 1, int(batch["n_graphs"]))
        init = {k: v.detach().clone() for k, v in m.state_dict().items()}
        with torch.no_grad():
            _, aux0 = m(batch, return_aux=True)
        assert float(aux0["delta_bond"].abs().max()) == 0.0
        first = None
        for step in range(3):
            opt.zero_grad(set_to_none=True)
            y = m(batch)
            F.mse_loss(y, target).backward()
            if step == 0:
                first = float(m.edge_score.weight.grad.abs().sum())
            opt.step()
        assert first > 0 and math.isfinite(first)
        assert float(m.atom_proj.weight.grad.abs().sum()) > 0
        assert float(m.edge_lin1.weight.grad.abs().sum()) > 0
        assert float(m.edge_lin2.weight.grad.abs().sum()) > 0
        after = {k: v.detach().clone() for k, v in m.state_dict().items()}
        assert max(float((after[k] - init[k]).abs().max()) for k in after if k.startswith("atom_proj.")) > 0
        assert max(float((after[k] - init[k]).abs().max()) for k in after if k.startswith("trunk.")) > 0


def D_collate(caches, rows):
    import data as D
    return D.collate(caches["fit"]["cache"], rows)

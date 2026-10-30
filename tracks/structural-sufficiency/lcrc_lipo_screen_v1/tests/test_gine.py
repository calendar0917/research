from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

import screen_common as C
import screen_data as D
import screen_models as M


class _NoOpConv(torch.nn.Module):
    def forward(self, x, edge_index, edge_attr):
        return torch.zeros_like(x)


def test_parameter_budget_is_matched_to_lcrc(model_c):
    _, info = model_c
    assert info["parameters"] == C.GINE_PARAMS_REFERENCE
    assert info["hidden"] == C.GINE_HIDDEN and info["layers"] == C.GINE_LAYERS
    assert abs(info["parameters"] - C.LCRC_PARAMS_REFERENCE) / C.LCRC_PARAMS_REFERENCE < 0.05


def test_true_adjacency_and_bond_features_matter(model_c, caches):
    model, _ = model_c
    cache = caches["fit"]["cache"]
    counts = D.atom_counts(cache)
    row = next(r for r in range(int(cache["n_molecules"])) if counts[r] >= 6)
    one = D.collate(cache, [row])
    with torch.no_grad():
        base = model(one)[0].item()
        rewired = dict(one)
        bi = one["bond_index"].clone()
        u, v = int(bi[0, 0]), int(bi[0, 1])
        w = next(x for x in range(int(one["atom_feat"].shape[0])) if x not in (u, v))
        bi[0, 1] = w
        rewired["bond_index"] = bi
        y_rewire = model(rewired)[0].item()
        edge = dict(one)
        bf = one["bond_feat"].clone()
        bf[0] += 0.5
        edge["bond_feat"] = bf
        y_edge = model(edge)[0].item()
        no_mp = copy.deepcopy(model)
        for i in range(len(no_mp.convs)):
            no_mp.convs[i] = _NoOpConv()
        y_no_mp = no_mp(one)[0].item()
    assert abs(y_rewire - base) > 1e-4
    assert abs(y_edge - base) > 1e-4
    assert abs(y_no_mp - base) > 1e-3


def test_permutation_invariance(model_c, caches):
    model, _ = model_c
    cache = caches["fit"]["cache"]
    row = next(r for r in range(int(cache["n_molecules"])) if D.atom_counts(cache)[r] >= 6)
    one = D.collate(cache, [row])
    n = int(one["atom_feat"].shape[0])
    g = torch.Generator().manual_seed(3)
    perm = torch.randperm(n, generator=g)
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(n)
    permuted = dict(one)
    permuted["atom_feat"] = one["atom_feat"][perm]
    permuted["bond_index"] = inv[one["bond_index"]]
    with torch.no_grad():
        base = model(one)[0].item()
        after = model(permuted)[0].item()
    assert abs(after - base) < 1e-4


def test_batch_composition_is_isolated(model_c, caches):
    model, _ = model_c
    cache = caches["fit"]["cache"]
    rows = list(range(8))
    batch = D.collate(cache, rows)
    with torch.no_grad():
        batched = model(batch).numpy()
        singles = np.asarray([model(D.collate(cache, [r]))[0].item() for r in rows])
    assert np.abs(batched - singles).max() < 1e-5


def test_gradients_are_finite_and_reach_every_parameter(model_c, fit_batch):
    model, _ = model_c
    batch, _ = fit_batch
    model.zero_grad(set_to_none=True)
    target = torch.linspace(-1, 1, int(batch["n_graphs"]), dtype=torch.float32)
    F.mse_loss(model(batch).view(-1), target).backward()
    grads = {n: float(p.grad.abs().sum()) if p.grad is not None else 0.0 for n, p in model.named_parameters()}
    assert all(np.isfinite(v) for v in grads.values())
    assert all(v > 0 for v in grads.values())


def test_edge_index_is_bidirectional(caches):
    cache = caches["fit"]["cache"]
    one = D.collate(cache, [0])
    assert one["bond_index"].shape[1] == 2
    n_bonds = int(one["bond_index"].shape[0])
    assert n_bonds > 0
    edge_index = M.bidirectional_edge_index(one["bond_index"])
    assert edge_index.shape == (2, 2 * n_bonds)
    pairs = set(zip(edge_index[0].tolist(), edge_index[1].tolist()))
    assert all((v, u) in pairs for u, v in pairs)
    assert len(pairs) == 2 * n_bonds

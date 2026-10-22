"""Fast CPU unit tests for the local-content encoders (no data, no GPU).

Narrow copies of the ``lf_checks`` gate so that a regression in the moment
formula, the permutation table, the aggregation semantics or the parameter
accounting fails a normal test run.
"""

from __future__ import annotations

import itertools
import math
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
_V1 = _HERE.parents[2] / "zinc_arch_discovery_v1" / "code"
_V2 = _HERE.parents[2] / "zinc_recipe_trunk_v2" / "code"
_BF = _HERE.parents[2] / "zinc_branching_fragment_v1" / "code"
for _p in (str(_CODE), str(_V1), str(_V2), str(_BF)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

import lf_models as LM  # noqa: E402


def _pattern_cache(atoms, pairs, lengths, families):
    return {
        "split": "__test",
        "pat_atom": np.asarray(atoms, dtype=np.uint8),
        "pat_pair": np.asarray(pairs, dtype=np.uint8),
        "pat_len": np.asarray(lengths, dtype=np.uint8),
        "pat_family": np.asarray(families, dtype=np.uint8),
    }


def _chain_pattern(length: int = 4, family: int = 0):
    atoms = np.zeros((1, LM.MAX_NODES), dtype=np.uint8)
    atoms[0, :length] = np.arange(6, 6 + length, dtype=np.uint8)
    pairs = np.zeros((1, LM.N_PAIRS), dtype=np.uint8)
    for k, (i, j) in enumerate(LM.PAIR_INDEX):
        if j < length and (i == 0 or j == i + 1):
            pairs[0, k] = 1
    return _pattern_cache(atoms, pairs, [length], [family])


# ---------------------------------------------------------------------------
# parameters / structure
# ---------------------------------------------------------------------------


def test_phi_parameter_counts():
    dense = LM.DensePhi()
    factor = LM.FactorizedPhi()
    assert sum(p.numel() for p in dense.parameters()) == 10_512
    assert sum(p.numel() for p in factor.parameters()) == 10_368
    # breakdown as declared in the round config
    assert sum(p.numel() for p in factor.a_proj.parameters()) == 288
    assert sum(p.numel() for p in factor.b_proj.parameters()) == 160
    assert factor.role_atom.numel() + factor.role_pair.numel() == 480
    assert sum(p.numel() for p in factor.l1.parameters()) == 7_360
    assert sum(p.numel() for p in factor.l2.parameters()) == 2_080
    assert LM.DensePhi.IN_DIM == 98 and LM.FactorizedPhi.IN_DIM == 114
    assert LM.PHI_OUT == 32 and LM.S_DIM == 198


def test_no_whole_pattern_parameter():
    for arm in ("D", "F"):
        branch = LM.ContentBranch(arm)
        for name, p in branch.named_parameters():
            assert p.shape[0] < 1280, f"{arm}:{name} looks like a pattern table"
        embeds = [(name, tuple(m.weight.shape)) for name, m in branch.named_modules()
                  if isinstance(m, torch.nn.Embedding)]
        assert embeds == [("atom_emb", (28, 8)), ("pair_emb", (4, 4))]


def test_role_tables_are_zero_mean_unit_scale_free_init():
    phi = LM.FactorizedPhi()
    phi.reinit(303)
    phi2 = LM.FactorizedPhi()
    phi2.reinit(303)
    for (n1, p1), (n2, p2) in zip(phi.named_parameters(), phi2.named_parameters()):
        assert n1 == n2 and torch.equal(p1, p2), f"{n1} is not reproducible from its seed"
    assert float(phi.role_atom.std()) < 0.2 and float(phi.role_pair.std()) < 0.2


# ---------------------------------------------------------------------------
# permutation table
# ---------------------------------------------------------------------------


def test_permutation_table_weights_and_counts():
    cache = _pattern_cache(
        atoms=[[6, 7, 8, 9, 0], [6, 7, 8, 9, 10]],
        pairs=[[1, 2, 1, 0, 3, 0, 0, 0, 0, 0], [1, 2, 1, 1, 3, 0, 0, 0, 0, 0]],
        lengths=[4, 5], families=[0, 1])
    table = LM.build_pattern_table(cache)
    assert table.n_perm_by_pattern.tolist() == [6, 24]
    weights = table.weight.sum(1)
    assert torch.allclose(weights, torch.ones(2))
    assert torch.equal(table.weight[0, 6:], torch.zeros(18))
    assert torch.equal(table.weight[1, :24], torch.full((24,), 1.0 / 24))
    # root slot never moves; padding slots stay padding
    assert torch.all(table.idx_node[:, :, 0] == 0)
    assert torch.equal(table.idx_node[0, :6, 4], torch.full((6,), 4))  # padding stays
    # every permutation row is a bijection of the real slots
    for p, length in ((0, 4), (1, 5)):
        for k in range(int(table.n_perm_by_pattern[p])):
            row = table.idx_node[p, k, :length].tolist()
            assert sorted(row) == list(range(length))


def test_masks_are_permutation_invariant():
    cache = _pattern_cache(
        atoms=[[6, 7, 8, 9, 0], [6, 7, 8, 9, 10]],
        pairs=[[1, 2, 1, 0, 3, 0, 0, 0, 0, 0], [1, 2, 1, 1, 3, 0, 0, 0, 0, 0]],
        lengths=[4, 5], families=[0, 1])
    table = LM.build_pattern_table(cache)
    assert table.node_mask[0].tolist() == [1, 1, 1, 1, 0]
    assert table.node_mask[1].tolist() == [1, 1, 1, 1, 1]
    assert float(table.pair_mask[0].sum()) == 6
    assert float(table.pair_mask[1].sum()) == 10


# ---------------------------------------------------------------------------
# moments
# ---------------------------------------------------------------------------


@pytest.mark.parametrize("m", [0, 1, 2, 10, 15])
def test_moments_match_explicit_enumeration(m):
    gen = torch.Generator().manual_seed(7)
    real = torch.randn(m, 5, generator=gen, dtype=torch.float64)
    raw = torch.full((15, 5), 1e6, dtype=torch.float64)   # padding carries huge values
    raw[:m] = real
    mask = torch.zeros(15, dtype=torch.float64)
    mask[:m] = 1.0
    fields = raw * mask.reshape(-1, 1)
    z1, z2, z3 = LM.FactorizedPhi.moments(fields, torch.tensor([float(m)], dtype=torch.float64))
    z1r = real.sum(0) / max(m, 1)
    z2r = (torch.stack([real[a] * real[b]
                        for a, b in itertools.combinations(range(m), 2)]).mean(0)
           if m >= 2 else torch.zeros(5, dtype=torch.float64))
    z3r = (torch.stack([real[a] * real[b] * real[c]
                        for a, b, c in itertools.combinations(range(m), 3)]).mean(0)
           if m >= 3 else torch.zeros(5, dtype=torch.float64))
    assert torch.allclose(z1, z1r, atol=1e-12)
    assert torch.allclose(z2, z2r, atol=1e-12)
    assert torch.allclose(z3, z3r, atol=1e-12)


def test_moment_denominator_is_real_field_count():
    fields = torch.tensor([2.0, 3.0] + [1e6] * 5, dtype=torch.float64).reshape(-1, 1)
    mask = torch.zeros(7, dtype=torch.float64)
    mask[:2] = 1.0
    z1, z2, z3 = LM.FactorizedPhi.moments(fields * mask.reshape(-1, 1),
                                          torch.tensor([2.0], dtype=torch.float64))
    assert float(z1[0, 0]) == pytest.approx(2.5, abs=1e-12)
    assert float(z2[0, 0]) == pytest.approx(6.0, abs=1e-12)
    assert float(z3[0, 0]) == 0.0


# ---------------------------------------------------------------------------
# forward / aggregation
# ---------------------------------------------------------------------------


def _branch(arm: str) -> LM.ContentBranch:
    branch = LM.ContentBranch(arm)
    branch.reinit_shared_embeddings()
    branch.reinit_phi()
    return branch


def _batch(table, pat, root=0, family=0, count=1.0, n_atoms=1):
    return {
        "occ_pat": torch.tensor([pat]),
        "occ_root": torch.tensor([root]),
        "occ_family": torch.tensor([family]),
        "occ_count": torch.tensor([count], dtype=torch.float32),
        "n_atoms_total": int(n_atoms),
        "split": table.split,
    }


@pytest.mark.parametrize("arm", ["D", "F"])
def test_phi_bar_equals_explicit_permutation_average(arm):
    branch = _branch(arm)
    cache = _chain_pattern(4, 0)
    table = LM.build_pattern_table(cache)
    atom = cache["pat_atom"][0].astype(np.int64)
    pair = cache["pat_pair"][0].astype(np.int64)
    node_mask = torch.tensor([1, 1, 1, 1, 0], dtype=torch.float32)
    pair_mask = torch.tensor([LM.PAIR_J[k] < 4 for k in range(LM.N_PAIRS)], dtype=torch.float32)
    fam = torch.tensor([1.0, 0.0, 0.0])
    outs = []
    with torch.no_grad():
        for q in itertools.permutations(range(1, 4)):
            order = (0,) + q                       # target slot -> source node
            atoms = np.zeros(LM.MAX_NODES, dtype=np.int64)
            for t, o in enumerate(order):
                atoms[t] = atom[o]
            pairs = np.zeros(LM.N_PAIRS, dtype=np.int64)
            for k, (i, j) in enumerate(LM.PAIR_INDEX):
                if j < 4:
                    src = LM._PAIR_LOOKUP[order[i], order[j]]
                    pairs[k] = pair[src]
            atom_e = branch.atom_emb(torch.as_tensor(atoms, dtype=torch.long)) * node_mask[:, None]
            pair_e = branch.pair_emb(torch.as_tensor(pairs, dtype=torch.long)) * pair_mask[:, None]
            if arm == "D":
                x = torch.cat([atom_e.reshape(-1), pair_e.reshape(-1), node_mask, pair_mask, fam])
                outs.append(branch.phi(x[None])[0])
            else:
                m = node_mask.sum() + pair_mask.sum()
                outs.append(branch.phi(atom_e[None], pair_e[None], node_mask[None],
                                       pair_mask[None], fam[None], m.reshape(1))[0])
        expected = torch.stack(outs, 0).mean(0)
        got = branch.permuted_phi(table)[0]
    assert torch.allclose(got, expected, atol=2e-6)


@pytest.mark.parametrize("arm", ["D", "F"])
def test_aggregation_semantics(arm):
    branch = _branch(arm)
    content = _chain_pattern(4, 0)
    table = LM.build_pattern_table(content)
    with torch.no_grad():
        phi_bar = branch.permuted_phi(table)[0]
        stats = branch(_batch(table, 0, count=1.0), table)[0]
        stats_occ = branch(_batch(table, 0, count=2.0), table)[0]
    assert torch.allclose(stats[:32], phi_bar, atol=1e-6)
    assert torch.allclose(stats[32:64], phi_bar * phi_bar, atol=1e-6)
    assert float(stats[64]) == pytest.approx(math.log1p(1.0), abs=1e-6)
    assert float(stats[65]) == 1.0
    # empty families are exact zeros
    assert torch.equal(stats[66:], torch.zeros_like(stats[66:]))
    # multiplicity scales the counts but not the weighted mean
    assert torch.allclose(stats_occ[:32], phi_bar, atol=1e-6)
    assert torch.allclose(stats_occ[32:64], phi_bar * phi_bar, atol=1e-6)
    assert float(stats_occ[64]) == pytest.approx(math.log1p(2.0), abs=1e-6)


@pytest.mark.parametrize("arm", ["D", "F"])
def test_permutation_average_keeps_autograd(arm):
    branch = _branch(arm)
    table = LM.build_pattern_table(_chain_pattern(4, 0))
    out = branch.permuted_phi(table)
    assert out.requires_grad
    out.sum().backward()
    assert branch.phi.l1.weight.grad is not None
    assert float(branch.phi.l1.weight.grad.abs().sum()) > 0.0

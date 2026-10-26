"""Fast, data-free unit tests for the additive / dense reader models.

The full pre-training gate (historical production replay, split hashes, cache
access audit) is ``code/ar_checks.py`` and needs the local ZINC cache.  These
tests only exercise the operator itself: parameter matching, shared
initialisation, additive separation, finite forward and gradients.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
if str(_CODE) not in sys.path:
    sys.path.insert(0, str(_CODE))

import ar_models as M  # noqa: E402

SEED = 20261027
_NORM = {
    "global_mean": np.zeros(33, dtype=np.float32),
    "global_scale": np.ones(33, dtype=np.float32),
}


def ring_batch(n_atoms: int = 6, seed: int = 7) -> dict:
    """A synthetic labelled-free ring with one junction per atom."""
    gen = torch.Generator().manual_seed(seed)
    bonds = [(i, (i + 1) % n_atoms) for i in range(n_atoms)]
    bond_index = torch.tensor(bonds, dtype=torch.long)
    junc_index = torch.tensor(
        [((j - 1) % n_atoms, j, (j + 1) % n_atoms) for j in range(n_atoms)],
        dtype=torch.long,
    )
    junc_bond = torch.tensor(
        [((j - 1) % n_atoms, j) for j in range(n_atoms)], dtype=torch.long
    )
    return {
        "n_graphs": 1,
        "atom_feat": torch.randn((n_atoms, 108), generator=gen),
        "atom_batch": torch.zeros(n_atoms, dtype=torch.long),
        "bond_feat": torch.randn((n_atoms, 7), generator=gen),
        "bond_batch": torch.zeros(n_atoms, dtype=torch.long),
        "bond_index": bond_index,
        "junc_feat": torch.randn((n_atoms, 32), generator=gen),
        "junc_batch": torch.zeros(n_atoms, dtype=torch.long),
        "junc_index": junc_index,
        "junc_bond": junc_bond,
        "global": torch.randn((1, 33), generator=gen),
    }


def make_models(seed: int = SEED):
    torch.manual_seed(seed)
    a = M.AdditiveReaderModel(_NORM)
    torch.manual_seed(seed)
    d = M.DenseReaderModel(_NORM)
    return a, d


def test_parameter_counts_and_matching():
    a, d = make_models()
    pa = sum(p.numel() for p in a.parameters())
    pd = sum(p.numel() for p in d.parameters())
    assert pa == 453_361
    assert pd == 461_521
    assert max(pa, pd) <= 500_000
    assert abs(pa - pd) / max(pa, pd) <= 0.02


@pytest.mark.skipif(
    not (Path(M.C.DCC).parent.parent / "zinc_arch_discovery_v1").exists(),
    reason="historical SSL encoders are not available in this checkout",
)
def test_build_pair_shared_initial_state():
    import ar_common as C

    if not C.TRAIN_CACHE.exists():
        pytest.skip("ZINC train cache is not available")
    # build_pair only needs the norm buffers; use the real Fit8k norm when possible.
    norm = C.fit_only_norm(C.load_train_cache(), C.split_rows()["fit8k"])
    a, d, info = M.build_pair(norm, 0)
    keys = [k for k in a.state_dict()
            if k.startswith(("encoders.", "group_composer.", "local_composer.", "phi."))]
    assert len(keys) > 20
    for k in keys:
        assert torch.equal(a.state_dict()[k], d.state_dict()[k])
    assert info["parameters_A"] == 453_361
    assert info["parameters_D"] == 461_521
    assert info["parameters_relative_difference"] <= 0.02
    # readers are not identical / not zero
    assert not torch.allclose(a.w.weight, torch.zeros_like(a.w.weight))
    assert info["A_reader_state_sha256"] != info["D_reader_state_sha256"]


@pytest.mark.parametrize("arm", ["A", "D"])
def test_forward_finite_and_gradients(arm):
    a, d = make_models()
    model = a if arm == "A" else d
    batch = ring_batch()
    y = model(batch)
    assert y.shape == (1,)
    assert torch.isfinite(y).all()
    model.zero_grad(set_to_none=True)
    y.sum().backward()
    for name, p in model.named_parameters():
        assert p.grad is not None, name
        assert torch.isfinite(p.grad).all(), name
    if arm == "A":
        assert float(a.w.weight.grad.norm()) > 0
        assert float(a.phi.lins[0].weight.grad.norm()) > 0


def test_additive_separation_and_dense_interaction():
    a, d = make_models()
    batch = ring_batch()
    e0 = 0
    delta = torch.full((M.PHI_OUT,), 0.2)
    gen = torch.Generator().manual_seed(99)
    with torch.no_grad():
        for model in (a, d):
            model.eval()
        ca = a.local_components(batch)
        h = ca["h"]
        noise = 1.5 * torch.randn(h.shape, generator=gen)
        h1 = h.clone(); h1[e0] += delta
        hn = h + noise
        hn1 = hn.clone(); hn1[e0] += delta
        y0 = a(batch, h_override=h)
        y1 = a(batch, h_override=h1)
        yn = a(batch, h_override=hn)
        yn1 = a(batch, h_override=hn1)
        da = float((y1 - y0)[0]); dna = float((yn1 - yn)[0])
        wdot = float((a.w.weight.view(-1) * delta).sum())
        assert abs(da - dna) <= 1e-5
        assert abs(da - wdot) <= 1e-5

        y0d = d(batch, h_override=h)
        y1d = d(batch, h_override=h1)
        ynd = d(batch, h_override=hn)
        yn1d = d(batch, h_override=hn1)
        dd = float((y1d - y0d)[0]); dnd = float((yn1d - ynd)[0])
        assert abs(dd - dnd) > 1e-5


def test_edge_reversal_and_renumber_invariance():
    a, d = make_models()
    batch = ring_batch()
    rev = dict(batch)
    rev["bond_index"] = batch["bond_index"].flip(1)
    with torch.no_grad():
        for model in (a, d):
            assert torch.allclose(model(batch), model(rev), atol=1e-6)
    # atom relabeling within the ring
    perm = torch.tensor([3, 0, 5, 2, 1, 4])
    inv = torch.empty_like(perm); inv[perm] = torch.arange(len(perm))
    per = dict(batch)
    per["atom_feat"] = batch["atom_feat"][perm]
    per["bond_index"] = inv[batch["bond_index"]]
    per["junc_index"] = inv[batch["junc_index"]]
    with torch.no_grad():
        for model in (a, d):
            assert torch.allclose(model(batch), model(per), atol=1e-5)

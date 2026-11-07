"""Fast CPU tests of the static pretext computation on synthetic graphs.

No cache access: the encoder pair is built at the frozen geometry with a fixed
private generator, and every property is checked on hand-built molecules.
"""

from __future__ import annotations

import pytest
import torch

import ssl_common as C
import ssl_model as M
import tu_model as TU


def _encoders(seed: int = 3):
    gen = torch.Generator().manual_seed(seed)
    atom_encoder = TU.MLP([C.ATOM_FEAT_DIM, C.ATOM_HIDDEN, C.ATOM_OUT])
    bond_encoder = TU.MLP([3 * C.ATOM_OUT + C.BOND_FEAT_DIM, C.BOND_HIDDEN, C.BOND_OUT])
    with torch.no_grad():
        for module in (atom_encoder, bond_encoder):
            for sub in module.modules():
                if isinstance(sub, torch.nn.Linear):
                    bound = 1.0 / float(sub.in_features) ** 0.5
                    sub.weight.uniform_(-bound, bound, generator=gen)
                    sub.bias.uniform_(-bound, bound, generator=gen)
    return atom_encoder, bond_encoder


def _chain(n_atoms: int = 6, seed: int = 11):
    gen = torch.Generator().manual_seed(seed)
    atom_feat = torch.randn(n_atoms, C.ATOM_FEAT_DIM, generator=gen)
    bonds = [(i, i + 1) for i in range(n_atoms - 1)]
    bond_index = torch.tensor(bonds, dtype=torch.long)
    bond_feat = torch.randn(len(bonds), C.BOND_FEAT_DIM, generator=gen)
    return atom_feat, bond_feat, bond_index


def test_context_widths():
    atom_encoder, bond_encoder = _encoders()
    atom_feat, bond_feat, bond_index = _chain()
    a, b = M.encode_static(atom_encoder, bond_encoder, atom_feat, bond_feat, bond_index)
    assert a.shape == (6, C.ATOM_OUT)
    assert b.shape == (5, C.BOND_OUT)
    ids = torch.arange(6)
    ctx = M.atom_context(a, b, bond_index, ids)
    assert ctx.shape == (6, C.ATOM_HEAD_INPUT)
    bctx = M.bond_context(a, b, bond_index, torch.arange(5))
    assert bctx.shape == (5, C.BOND_HEAD_INPUT)


def test_empty_neighbourhood_is_zero():
    atom_encoder, bond_encoder = _encoders()
    atom_feat = torch.randn(1, C.ATOM_FEAT_DIM)
    bond_feat = torch.zeros(0, C.BOND_FEAT_DIM)
    bond_index = torch.zeros((0, 2), dtype=torch.long)
    a, b = M.encode_static(atom_encoder, bond_encoder, atom_feat, bond_feat, bond_index)
    ctx = M.atom_context(a, b, bond_index, torch.arange(1))
    assert torch.allclose(ctx[0, C.ATOM_OUT :], torch.zeros(2 * C.ATOM_OUT))


def test_bond_endpoint_swap_is_symmetric():
    atom_encoder, bond_encoder = _encoders()
    atom_feat, bond_feat, bond_index = _chain()
    a, b = M.encode_static(atom_encoder, bond_encoder, atom_feat, bond_feat, bond_index)
    ids = torch.arange(len(bond_index))
    base = M.bond_context(a, b, bond_index, ids)
    swapped = bond_index[:, [1, 0]].clone()
    swapped_sorted, _ = torch.sort(swapped, dim=1)
    assert torch.equal(swapped_sorted, bond_index)
    b2 = M.bond_context(a, b, swapped, ids)
    assert torch.allclose(base, b2, atol=1e-6)
    # the same holds for the encoded bond vector itself
    a2, b_sw = M.encode_static(atom_encoder, bond_encoder, atom_feat, bond_feat, swapped)
    assert torch.allclose(a, a2) and torch.allclose(b, b_sw, atol=1e-6)


def test_atom_context_ignores_distance_two_atoms():
    atom_encoder, bond_encoder = _encoders()
    atom_feat, bond_feat, bond_index = _chain()
    target = 2  # chain 0-1-2-3-4-5, distance-2 atom is 4 and 0
    rng = torch.Generator().manual_seed(77)
    a, b = M.encode_static(atom_encoder, bond_encoder, atom_feat, bond_feat, bond_index)
    ctx = M.atom_context(a, b, bond_index, torch.tensor([target]))
    perturbed = atom_feat.clone()
    perturbed[4] = perturbed[4] + torch.randn(C.ATOM_FEAT_DIM, generator=rng) * 5.0
    a2, b2 = M.encode_static(atom_encoder, bond_encoder, perturbed, bond_feat, bond_index)
    ctx2 = M.atom_context(a2, b2, bond_index, torch.tensor([target]))
    assert torch.equal(ctx, ctx2)


def test_bond_context_ignores_distance_two_atoms():
    atom_encoder, bond_encoder = _encoders()
    atom_feat, bond_feat, bond_index = _chain()
    bond_id = 2  # bond (2, 3); distance-2 atoms of its endpoints: 0 and 5
    rng = torch.Generator().manual_seed(88)
    a, b = M.encode_static(atom_encoder, bond_encoder, atom_feat, bond_feat, bond_index)
    ctx = M.bond_context(a, b, bond_index, torch.tensor([bond_id]))
    perturbed = atom_feat.clone()
    for atom in (0, 5):
        perturbed[atom] = perturbed[atom] + torch.randn(C.ATOM_FEAT_DIM, generator=rng) * 5.0
    a2, b2 = M.encode_static(atom_encoder, bond_encoder, perturbed, bond_feat, bond_index)
    ctx2 = M.bond_context(a2, b2, bond_index, torch.tensor([bond_id]))
    assert torch.equal(ctx, ctx2)


def test_atom_context_permutation_equivariance():
    atom_encoder, bond_encoder = _encoders()
    atom_feat, bond_feat, bond_index = _chain(n_atoms=7)
    perm = torch.tensor([3, 0, 6, 1, 5, 2, 4])
    inv = torch.empty_like(perm)
    inv[perm] = torch.arange(perm.numel())
    # ``perm[r]`` is the old atom that lands at new position ``r``
    new_feat = atom_feat[perm]
    new_index, _ = torch.sort(inv[bond_index], dim=1)
    a, b = M.encode_static(atom_encoder, bond_encoder, atom_feat, bond_feat, bond_index)
    a2, b2 = M.encode_static(atom_encoder, bond_encoder, new_feat, bond_feat, new_index)
    ids = torch.arange(7)
    ctx = M.atom_context(a, b, bond_index, ids)
    ctx2 = M.atom_context(a2, b2, new_index, inv)
    assert torch.allclose(ctx, ctx2, atol=1e-6)


def test_downstream_model_has_no_pretext_heads():
    atom_encoder, bond_encoder = _encoders()
    net = M.PretextNet(atom_encoder, bond_encoder)
    keys = set(net.state_dict())
    assert any(k.startswith("atom_head.") for k in keys)
    assert any(k.startswith("bond_head.") for k in keys)
    enc = M.encoder_state(net)
    assert all(k.startswith(C.TRANSFERRED_PREFIXES) for k in enc)
    assert not any(k.startswith(("atom_head.", "bond_head.")) for k in enc)
    assert len(enc) == 8


def test_install_encoders_is_strict():
    atom_encoder, bond_encoder = _encoders()
    net = M.PretextNet(atom_encoder, bond_encoder)

    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.atom_encoder = TU.MLP([C.ATOM_FEAT_DIM, C.ATOM_HIDDEN, C.ATOM_OUT])
            self.bond_encoder = TU.MLP(
                [3 * C.ATOM_OUT + C.BOND_FEAT_DIM, C.BOND_HIDDEN, C.BOND_OUT]
            )
            self.other = torch.nn.Linear(3, 2)

    holder = Holder()
    before = {k: v.clone() for k, v in holder.state_dict().items()}
    audit = M.install_encoders(holder, M.encoder_state(net))
    assert audit["transferred_tensors"] == 8
    assert torch.equal(holder.state_dict()["other.weight"], before["other.weight"])
    assert M.encoder_fingerprint(holder) == M.encoder_fingerprint(net)
    with pytest.raises(RuntimeError):
        M.install_encoders(holder, {"atom_encoder.lins.0.weight": torch.zeros(1)})


def test_parameter_change_reports_every_tensor():
    atom_encoder, bond_encoder = _encoders()
    net = M.PretextNet(atom_encoder, bond_encoder)
    before = M.encoder_state(net)
    with torch.no_grad():
        for p in net.atom_encoder.parameters():
            p.add_(0.01)
    change = M.parameter_change(before, M.encoder_state(net))
    assert len(change["per_tensor"]) == 8
    assert change["all_tensors_changed"] is False  # bond encoder untouched
    assert change["l2_delta"] > 0.0


def test_loss_is_finite_and_uses_both_views():
    atom_encoder, bond_encoder = _encoders()
    net = M.PretextNet(atom_encoder, bond_encoder)
    n_atoms, n_bonds = 6, 5
    gen = torch.Generator().manual_seed(5)
    view = {
        "atom_feat": torch.randn(n_atoms, C.ATOM_FEAT_DIM, generator=gen),
        "bond_feat": torch.randn(n_bonds, C.BOND_FEAT_DIM, generator=gen),
        "bond_index": torch.tensor([[0, 1], [1, 2], [2, 3], [3, 4], [4, 5]]),
    }
    bond_view = {
        "atom_feat": view["atom_feat"].clone(),
        "bond_feat": view["bond_feat"].clone(),
        "bond_index": view["bond_index"],
    }
    atom_ids = torch.tensor([1, 3])
    bond_ids = torch.tensor([0, 4])
    loss, stats = net.loss(
        view,
        bond_view,
        atom_ids,
        bond_ids,
        torch.tensor([2, 3]),
        torch.tensor([0, 3]),
    )
    assert torch.isfinite(loss)
    assert "ce_atom" in stats and "ce_bond" in stats
    assert float(stats["loss"]) == pytest.approx(float(stats["ce_atom"]) + float(stats["ce_bond"]))
    assert int(stats["atom_n"]) == 2 and int(stats["bond_n"]) == 2


def test_empty_bond_targets_fall_back_to_atom_loss_only():
    atom_encoder, bond_encoder = _encoders()
    net = M.PretextNet(atom_encoder, bond_encoder)
    gen = torch.Generator().manual_seed(6)
    view = {
        "atom_feat": torch.randn(1, C.ATOM_FEAT_DIM, generator=gen),
        "bond_feat": torch.zeros(0, C.BOND_FEAT_DIM),
        "bond_index": torch.zeros((0, 2), dtype=torch.long),
    }
    loss, stats = net.loss(
        view, view, torch.tensor([0]), torch.zeros(0, dtype=torch.long),
        torch.tensor([7]), torch.zeros(0, dtype=torch.long),
    )
    assert torch.isfinite(loss)
    assert "ce_bond" not in stats
    assert float(loss) == pytest.approx(float(stats["ce_atom"]))


def test_empty_pretext_batch_is_refused():
    atom_encoder, bond_encoder = _encoders()
    net = M.PretextNet(atom_encoder, bond_encoder)
    gen = torch.Generator().manual_seed(9)
    view = {
        "atom_feat": torch.randn(1, C.ATOM_FEAT_DIM, generator=gen),
        "bond_feat": torch.zeros(0, C.BOND_FEAT_DIM),
        "bond_index": torch.zeros((0, 2), dtype=torch.long),
    }
    with pytest.raises(ValueError):
        net.loss(
            view, view, torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
            torch.zeros(0, dtype=torch.long), torch.zeros(0, dtype=torch.long),
        )

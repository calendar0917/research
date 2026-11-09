"""Forward / invariance / gradient checks for the BOC model."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import boc_common as C
import boc_model as M
import boc_structures as S


@pytest.fixture(scope="module")
def fixtures(tmp_path_factory):
    rows = list(range(0, 64))
    out = tmp_path_factory.mktemp("boc_struct")
    meta = S.build_cache(out, rows=rows, reference_fraction=0.1, log_every=0)
    assert meta["reference_audit"]["mismatches"] == 0
    struct = S.load_cache(out)
    task = C.BC.SD.load_task()
    batch = S.collate_boc(task, struct, rows[:12])
    return task, struct, batch


def _flip_bonds(batch):
    out = dict(batch)
    out["bond_index"] = batch["bond_index"][:, [1, 0]].contiguous()
    out["boc_nb_side"] = 1 - batch["boc_nb_side"]
    kind = batch["boc_pair_kind"].clone()
    kind = torch.where(kind == 0, torch.ones_like(kind), torch.where(kind == 1, torch.zeros_like(kind), kind))
    out["boc_pair_kind"] = kind
    cross = batch["boc_pair_kind"] == 2
    a = batch["boc_pair_a"].clone()
    b = batch["boc_pair_b"].clone()
    a[cross], b[cross] = batch["boc_pair_b"][cross], batch["boc_pair_a"][cross]
    out["boc_pair_a"] = a
    out["boc_pair_b"] = b
    out["boc_nL"], out["boc_nR"] = batch["boc_nR"].clone(), batch["boc_nL"].clone()
    out["boc_nLL"], out["boc_nRR"] = batch["boc_nRR"].clone(), batch["boc_nLL"].clone()
    return out


def _shuffle_sides(batch, seed=3):
    """Permute neighbour entries inside each side and remap pair indices."""
    out = dict(batch)
    gen = torch.Generator().manual_seed(int(seed))
    nb_ptr = batch["boc_nb_ptr"]
    nL = batch["boc_nL"].tolist()
    nR = batch["boc_nR"].tolist()
    side = batch["boc_nb_side"].clone()
    pair_a = batch["boc_pair_a"].clone()
    pair_b = batch["boc_pair_b"].clone()
    fields = (
        "boc_nb_atom",
        "boc_nb_side",
        "boc_nb_bond_side",
        "boc_nb_bond_other",
    )
    new = {f: batch[f].clone() for f in fields}
    for bond in range(int(batch["boc_n_bonds"])):
        base = int(nb_ptr[bond])
        nl, nr = int(nL[bond]), int(nR[bond])
        perm = torch.cat(
            [
                torch.randperm(nl, generator=gen),
                nl + torch.randperm(nr, generator=gen),
            ]
        ) if nl + nr else torch.zeros(0, dtype=torch.long)
        for f in fields:
            new[f][base : base + nl + nr] = batch[f][base : base + nl + nr][perm]
        # pair indices are relative to this bond's neighbour block
        inv = torch.empty(nl + nr, dtype=torch.long)
        inv[perm] = torch.arange(nl + nr, dtype=torch.long)
        for field in (pair_a, pair_b):
            lo = int(batch["boc_pair_ptr"][bond])
            hi = int(batch["boc_pair_ptr"][bond + 1])
            old = field[lo:hi] - base
            field[lo:hi] = inv[old] + base
    out.update(new)
    out["boc_pair_a"] = pair_a
    out["boc_pair_b"] = pair_b
    return out


def _renumber_atoms(batch, seed=5):
    out = dict(batch)
    atom_batch = batch["atom_batch"]
    n_graphs = int(batch["n_graphs"])
    gen = torch.Generator().manual_seed(int(seed))
    parts = []
    off = 0
    counts = torch.bincount(atom_batch, minlength=n_graphs).tolist()
    for g in range(n_graphs):
        n = int(counts[g])
        parts.append(torch.arange(off, off + n, dtype=torch.long)[torch.randperm(n, generator=gen)])
        off += n
    p = torch.cat(parts)
    p_inv = torch.empty_like(p)
    p_inv[p] = torch.arange(p.numel(), dtype=torch.long)
    for key in ("atom_feat", "atom_scalar", "atom_type"):
        if key in batch:
            out[key] = batch[key][p]
    for key in ("boc_nb_atom", "boc_ring_atom", "boc_ring_ext_atom", "nbr_atom", "path_atom", "path_end"):
        if key in batch and batch[key].numel():
            out[key] = p_inv[batch[key]]
    if batch["bond_index"].numel():
        out["bond_index"] = p_inv[batch["bond_index"]]
    return out


@pytest.fixture(scope="module")
def models(fixtures):
    task, _struct, _batch = fixtures
    full, full_audit = M.build_model("FULL", task.spec, task.norm, seed=0, device="cpu")
    bond, bond_audit = M.build_model("BOND", task.spec, task.norm, seed=0, device="cpu")
    full.eval()
    bond.eval()
    return full, full_audit, bond, bond_audit


def test_parameter_count_and_historical_init(models):
    full, full_audit, bond, bond_audit = models
    assert full_audit["total_parameters"] == 1_352_596
    assert bond_audit["total_parameters"] == 1_352_596
    assert full_audit["base_parameters"] == 994_324
    assert full_audit["base_state_sha256"] == C.W2["base_state_sha256_at_init"]
    assert full_audit["base_trunk_init_fingerprint"] == C.W2["trunk_init_fingerprint"]
    assert full.state_dict().keys() == bond.state_dict().keys()
    for k in full.state_dict():
        assert full.state_dict()[k].shape == bond.state_dict()[k].shape
    # no zeroed fusion output (explicitly forbidden by the protocol)
    assert full_audit["new_module_init"]["fusion_output_zeroed"] is False


def test_forward_shapes_and_call_counts(fixtures, models):
    _task, _struct, batch = fixtures
    full, _, bond, _ = models
    with torch.no_grad():
        logits, aux = full(batch, return_aux=True)
        assert logits.shape == (int(batch["n_graphs"]), 128)
        assert aux["h0"].shape[1] == 128
        assert aux["m"].shape[1] == 64
        assert aux["r"].shape[1] == 32
        assert aux["fusion_input"].shape[1] == 224
        assert aux["ring_nonzero"] > 0
        calls = M.count_module_calls(full, batch)
        for name in ("atom_encoder", "bond_encoder", "local", "drop", "f", "rho", "readout", "fusion"):
            assert calls[name] == 1, name
        assert calls["q_mlp"] == 2  # forward and reverse centre order
        assert calls["boc_atom_proj"] == 1 and calls["boc_bond_proj"] == 1
        assert calls["ring_ext_proj"] == 1 and calls["ring_pair_mlp"] == 1
        logits_b, aux_b = bond(batch, return_aux=True)
        assert int(aux_b["ring_nonzero"]) == 0
        assert torch.count_nonzero(aux_b["r"]).item() == 0
        assert logits_b.shape == logits.shape


def test_bond_ring_channels_are_identical_at_init(fixtures, models):
    _task, _struct, batch = fixtures
    full, _, bond, _ = models
    with torch.no_grad():
        z_full = full(batch, ring_zero=True)
        z_bond = bond(batch)
    torch.testing.assert_close(z_full, z_bond, rtol=1e-6, atol=1e-6)


def test_bond_zero_removes_m_and_exact_ring_zero_removes_r(fixtures, models):
    _task, _struct, batch = fixtures
    full, _, _, _ = models
    with torch.no_grad():
        _, aux = full(batch, return_aux=True, bond_zero=True, ring_zero=True)
    assert int(torch.count_nonzero(aux["m"]).item()) == 0
    assert int(torch.count_nonzero(aux["r"]).item()) == 0
    assert torch.isfinite(full(batch, bond_zero=True, ring_zero=True)).all()


def test_swap_centre_direction_leaves_z_b_unchanged(fixtures, models):
    _task, _struct, batch = fixtures
    full, _, _, _ = models
    flipped = _flip_bonds(batch)
    with torch.no_grad():
        _, a = full(batch, return_aux=True)
        _, b = full(flipped, return_aux=True)
    torch.testing.assert_close(a["z_b"], b["z_b"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a["m"], b["m"], rtol=1e-5, atol=1e-5)
    torch.testing.assert_close(a["r"], b["r"], rtol=1e-5, atol=1e-5)


def test_same_side_neighbour_order_invariance(fixtures, models):
    _task, _struct, batch = fixtures
    full, _, _, _ = models
    shuffled = _shuffle_sides(batch)
    with torch.no_grad():
        _, a = full(batch, return_aux=True)
        _, b = full(shuffled, return_aux=True)
    torch.testing.assert_close(a["z_b"], b["z_b"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(a["m"], b["m"], rtol=1e-4, atol=1e-5)
    torch.testing.assert_close(a["r"], b["r"], rtol=1e-4, atol=1e-5)


def test_atom_renumbering_leaves_logits_unchanged(fixtures, models):
    _task, _struct, batch = fixtures
    full, _, _, _ = models
    permuted = _renumber_atoms(batch)
    with torch.no_grad():
        a = full(batch)
        b = full(permuted)
    torch.testing.assert_close(a, b, rtol=1e-5, atol=1e-6)


def test_two_molecules_do_not_leak(fixtures, models):
    task, struct, batch = fixtures
    full, _, _, _ = models
    rows = list(range(12))
    one = S.collate_boc(task, struct, rows[:1])
    two = S.collate_boc(task, struct, rows[:2])
    with torch.no_grad():
        _, a = full(one, return_aux=True)
        _, b = full(two, return_aux=True)
    n0 = int(one["boc_n_atoms"])
    n1 = int(two["boc_n_atoms"]) - n0
    torch.testing.assert_close(a["m"], b["m"][:n0], rtol=1e-6, atol=1e-6)
    torch.testing.assert_close(a["r"], b["r"][:n0], rtol=1e-6, atol=1e-6)
    assert int(one["boc_n_nb"]) <= int(two["boc_n_nb"])
    assert int(one["boc_n_rings"]) <= int(two["boc_n_rings"])
    assert n1 > 0


def test_gradients_finite_and_ring_channel_active(fixtures, models):
    task, _struct, batch = fixtures
    full, _, bond, _ = models
    y = torch.as_tensor(task.labels[list(range(12))], dtype=torch.float32)
    for model, expect_ring in ((full, True), (bond, False)):
        model.train()
        model.zero_grad(set_to_none=True)
        loss = C.MET.masked_bce_loss(model(batch), y)
        assert torch.isfinite(loss)
        loss.backward()
        rep = M.gradient_report(model)
        for name, value in rep["grad_l2"].items():
            assert np.isfinite(value)
        assert rep["grad_l2"]["bond_channel"] > 0
        assert rep["grad_l2"]["main_fusion"] > 0
        assert rep["grad_l2"]["w2_trunk"] > 0
        if expect_ring:
            assert rep["grad_l2"]["ring_channel"] > 0
            assert rep["nonzero_fraction"]["ring_channel"] > 0
        else:
            assert rep["grad_l2"]["ring_channel"] == 0.0
            assert rep["nonzero_fraction"]["ring_channel"] == 0.0
        model.eval()
        model.zero_grad(set_to_none=True)


def test_object_encoders_read_atom_bond_codes_not_h(fixtures, models):
    _task, _struct, batch = fixtures
    full, _, _, _ = models
    captured = {}
    handles = []

    def _hook(name):
        def inner(_m, inp, _out):
            captured[name] = inp[0].detach().clone()

        return inner

    for name in ("boc_atom_proj", "boc_bond_proj", "q_mlp", "t_mlp"):
        handles.append(getattr(full, name).register_forward_hook(_hook(name)))
    try:
        full.eval()
        with torch.no_grad():
            a, b, _scal = full._encode(batch)
            full(batch)
    finally:
        for h in handles:
            h.remove()
    assert captured["boc_atom_proj"].shape == a.shape
    torch.testing.assert_close(captured["boc_atom_proj"], a)
    assert captured["boc_bond_proj"].shape == b.shape
    torch.testing.assert_close(captured["boc_bond_proj"], b)
    # q reads three 64-d codes (192 columns), never the 128-d h0
    assert captured["q_mlp"].shape[1] == 192
    assert captured["t_mlp"].shape[1] == 385

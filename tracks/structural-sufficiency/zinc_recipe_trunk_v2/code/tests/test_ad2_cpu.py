"""Focused CPU tests for zinc-recipe-trunk-v2 (no GPU, no long training)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
import torch

_HERE = Path(__file__).resolve().parent
_CODE = _HERE.parent
_V1 = _CODE.parents[1] / "zinc_arch_discovery_v1" / "code"
for _p in (str(_CODE), str(_V1)):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import ad_common as C  # noqa: E402
import ad_data as D  # noqa: E402
import ad_models as M1  # noqa: E402
import ad_train as T1  # noqa: E402
import ad2_models as A2  # noqa: E402

NEW = ("P3N", "M", "S", "Maux", "Saux")


def _norm():
    return D.load_norm()


@pytest.fixture(scope="module")
def valid_batch():
    cache = D.load_cache("valid")
    return cache, T1.collate(cache, np.arange(6, dtype=np.int64), "M")


def test_closes_marks_return_to_start_not_same_type():
    wa = torch.tensor([[6, 6, 6], [6, 6, 6]], dtype=torch.long)
    anon = torch.tensor([[0, 1, 2], [0, 1, 0]], dtype=torch.long)
    L = torch.tensor([2, 2], dtype=torch.long)
    closes = M1.path_token_scalars(wa, anon, L)[..., 3].bool()
    assert closes[0].tolist() == [False, False, False]  # three distinct atoms
    assert closes[1].tolist() == [False, False, True]  # one genuine return


def test_p3n_matches_bjoc_parameter_count_and_loads_ssl():
    model, info = A2.build_model("P3N", _norm(), 0, "cpu")
    assert C.parameter_count(model) == 413505
    assert info["parameters"] == 413505
    # the recipe cross must start from the same seed-matched SSL encoder state
    assert info["ssl"] is not None and info["ssl"]["source"].endswith("pretrain_encoder_s0.pt")


def test_new_trunks_parameter_cap_and_forward(valid_batch):
    cache, batch = valid_batch
    for cand in NEW:
        model, info = A2.build_model(cand, _norm(), 0, "cpu")
        assert C.parameter_count(model) <= 500_000
        assert info["parameters"] == C.parameter_count(model)
        model.train()
        pred = model(batch)
        assert pred.shape == (6,)
        assert torch.isfinite(pred).all()
        pred.pow(2).mean().backward()
        trunk = [(n, p) for n, p in model.named_parameters() if not n.startswith("aux.")]
        grads = [p.grad for _, p in trunk if p.grad is not None]
        # the CE heads are only used by the auxiliary branch, never by L1(clean)
        assert len(grads) == len(trunk)
        assert all(torch.isfinite(g).all() for g in grads)
        assert all(torch.count_nonzero(g).item() > 0 for g in grads)


def test_aux_loss_only_on_joint_variants(valid_batch):
    cache, batch = valid_batch
    sup, _ = A2.build_model("M", _norm(), 0, "cpu")
    assert sup.aux is None
    joint, _ = A2.build_model("Maux", _norm(), 0, "cpu")
    assert joint.aux is not None
    atom_ids = torch.arange(4)
    bond_ids = torch.arange(3)
    total, parts = joint.aux_loss(
        batch, atom_ids, torch.zeros(4, dtype=torch.long),
        bond_ids, torch.zeros(3, dtype=torch.long),
    )
    assert set(parts) == {"ce_atom", "ce_bond"}
    assert torch.isfinite(total)
    total.backward()
    aux_params = dict(joint.named_parameters())
    for name in ("aux.atom_head.weight", "aux.atom_head.bias", "aux.bond_head.weight", "aux.bond_head.bias"):
        g = aux_params[name].grad
        assert g is not None and torch.isfinite(g).all() and torch.count_nonzero(g).item() > 0


def test_slot_trunk_isolates_graphs(valid_batch):
    cache, batch = valid_batch
    model, _ = A2.build_model("S", _norm(), 0, "cpu")
    model.eval()
    with torch.no_grad():
        base = model(batch)
        b2 = dict(batch)
        b2["atom_feat"] = batch["atom_feat"].clone()
        a0 = int(cache["atom_ptr"][0])
        a1 = int(cache["atom_ptr"][1])
        b2["atom_feat"][a0:a1] += 0.5
        pert = model(b2)
    assert torch.equal(base[1:], pert[1:])
    assert torch.isfinite(pert[0])


def test_moment_trunk_batch_vs_single(valid_batch):
    cache, batch = valid_batch
    model, _ = A2.build_model("M", _norm(), 0, "cpu")
    model.eval()
    with torch.no_grad():
        base = model(batch)
        single = model(T1.collate(cache, np.arange(1, dtype=np.int64), "M"))
    assert torch.allclose(base[0], single[0], atol=1e-5)


def test_recipe_variants_are_declared_and_distinct():
    a, b = T1.RECIPE_VARIANTS["A"], T1.RECIPE_VARIANTS["B"]
    assert a["optimizer"] == "adam" and a["epochs"] == 360 and a["warmup_steps"] == 0
    assert b["optimizer"] == "adamw" and b["epochs"] == 600 and b["warmup_steps"] == 500
    assert a["batch_size"] == b["batch_size"] == 128
    assert a["lr"] == b["lr"] == 1.0e-3
    assert a["weight_decay"] == b["weight_decay"] == 1.0e-5

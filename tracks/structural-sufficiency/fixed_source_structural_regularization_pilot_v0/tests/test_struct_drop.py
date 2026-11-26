"""CPU unit tests for the STRUCT_DROP compute path (synthetic tensors only)."""

from __future__ import annotations

import copy
import sys
from pathlib import Path

import torch
import torch.nn.functional as F

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import sr_common  # noqa: E402,F401  (adds the inherited ds_* code path)
import sr_model as SM  # noqa: E402


def make_model(q: float) -> SM.StructuralDropModel:
    model = SM.StructuralDropModel.__new__(SM.StructuralDropModel)
    torch.nn.Module.__init__(model)
    model.cfg = {"width": 4}
    model.arm = "FIXED1"
    model.struct_drop_q = float(q)
    model.mask_seed = 7
    model._mask_gen = None
    model.mask_draws = 0
    model.eval_forward_calls = 0
    model.mask_events = []
    model.last_mask = None
    model.bond_proj = torch.nn.Linear(3, 4)
    model.blocks = torch.nn.ModuleList([SM.DM.MessageFFN(4, 8), SM.DM.MessageFFN(4, 8)])
    return model


def synthetic_batch() -> dict[str, torch.Tensor]:
    return {
        "bond_index": torch.tensor([[0, 1], [1, 2], [2, 3], [3, 0], [0, 2]], dtype=torch.long),
        "bond_feat": torch.randn(5, 3),
    }


def reference_fixed1(model: SM.StructuralDropModel, h0: torch.Tensor, batch: dict) -> tuple[torch.Tensor, torch.Tensor]:
    src, dst, edge_msg = SM.DM.directed_bonds(model, batch)
    blk1, blk2 = model.blocks[0], model.blocks[1]
    z1 = blk1.norm(h0)
    m1 = torch.zeros_like(z1).index_add_(0, dst, F.silu(z1.index_select(0, src) + edge_msg))
    h1 = h0 + SM.DM._apply_ffn(blk1, z1, m1)
    zt, zs = blk2.norm(h1), blk2.norm(h0)
    m2 = torch.zeros_like(zs).index_add_(0, dst, F.silu(zs.index_select(0, src) + edge_msg))
    h2 = h1 + SM.DM._apply_ffn(blk2, zt, m2)
    return h1, h2


def test_q0_forward_and_gradients_match_fixed1() -> None:
    torch.manual_seed(0)
    model = make_model(0.0)
    ref = copy.deepcopy(model)
    h0 = torch.randn(4, 4)
    batch = synthetic_batch()
    model.train()
    ref.train()
    h1, h2, info = model.structural_layers(h0, batch)
    r1, r2 = reference_fixed1(ref, h0, batch)
    assert torch.equal(h1, r1) and torch.equal(h2, r2)
    assert info["mask"]["dropped_fraction_physical"] == 0.0
    loss = h2.pow(2).mean()
    loss.backward()
    ref_loss = r2.pow(2).mean()
    ref_loss.backward()
    for (name, p), (rname, rp) in zip(model.named_parameters(), ref.named_parameters()):
        assert name == rname
        assert p.grad is not None and rp.grad is not None
        assert torch.equal(p.grad, rp.grad), name


def test_mask_shared_by_both_blocks_and_both_directions() -> None:
    torch.manual_seed(0)
    model = make_model(0.10)
    model.train()
    batch = synthetic_batch()
    h0 = torch.randn(4, 4)
    _h1, _h2, info = model.structural_layers(h0, batch)
    mask = info["mask"]
    assert mask["n_physical_bonds"] == 5
    assert mask["n_directed_edges"] == 10
    assert mask["same_mask_tensor_both_blocks"] is True
    assert mask["directions_share_physical_mask"] is True
    assert mask["block_masks"][0]["keep_data_ptr"] == mask["block_masks"][1]["keep_data_ptr"]
    assert mask["block_masks"][0]["keep_numel"] == 10


def test_mask_generator_does_not_advance_global_rng() -> None:
    torch.manual_seed(123)
    model = make_model(0.10)
    model.train()
    batch = synthetic_batch()
    h0 = torch.randn(4, 4)
    before = torch.get_rng_state().clone()
    model.structural_layers(h0, batch)
    after = torch.get_rng_state()
    assert torch.equal(before, after)
    assert model.mask_draws == 1
    state_after_first = model.mask_generator(torch.device("cpu")).get_state()
    model.structural_layers(h0, batch)
    assert model.mask_draws == 2
    assert not torch.equal(state_after_first, model.mask_generator(torch.device("cpu")).get_state())


def test_eval_mode_has_no_mask_and_matches_fixed1() -> None:
    torch.manual_seed(0)
    model = make_model(0.10)
    ref = copy.deepcopy(model)
    model.train()
    batch = synthetic_batch()
    h0 = torch.randn(4, 4)
    model.structural_layers(h0, batch)
    draws = model.mask_draws
    model.eval()
    ref.eval()
    with torch.inference_mode():
        h1, h2, info = model.structural_layers(h0, batch)
        r1, r2 = reference_fixed1(ref, h0, batch)
    assert model.mask_draws == draws
    assert model.eval_forward_calls == 1
    assert "mask" not in info
    assert torch.equal(h1, r1) and torch.equal(h2, r2)


def test_mask_mapping_exact_against_loop() -> None:
    z = torch.randn(6, 4)
    src = torch.tensor([0, 1, 2, 3, 4, 5])
    dst = torch.tensor([1, 2, 3, 4, 5, 0])
    edge = torch.zeros(6, 4)
    keep = torch.tensor([1.0, 0.0, 2.0, 0.0, 1.0, 2.0])
    got = SM.masked_source_message(z, src, dst, edge, keep)
    ref = torch.zeros_like(z)
    for e in range(6):
        ref[dst[e]] += F.silu(z[src[e]]) * keep[e]
    assert torch.equal(got, ref)


def test_realised_drop_fraction_near_q() -> None:
    model = make_model(0.10)
    model.train()
    total = 0
    dropped = 0
    for _ in range(20):
        keep = model.draw_keep(20_000, torch.device("cpu"), torch.float32)
        assert keep.numel() == 20_000
        dropped += int((keep == 0).sum().item())
        total += keep.numel()
    frac = dropped / total
    assert abs(frac - 0.10) < 0.005
    assert torch.allclose(keep[keep > 0], torch.full_like(keep[keep > 0], 1.0 / 0.9))

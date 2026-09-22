"""Targeted correctness tests for the corrected TCCD-v3b bridge."""

from __future__ import annotations

import numpy as np
import torch

from tracks.ksvd.code import tccd_v0 as T
from tracks.ksvd.code import tccd_v3b as V
from tracks.ksvd.code.run_tccd_v0 import synthetic_zinc_like


def _records(n_graphs: int = 4):
    layout = T.PatchLayout(capacity=8, n_atom=5, n_bond=3)
    atom_index = {c: i for i, c in enumerate(range(5))}
    bond_index = {c: i for i, c in enumerate(range(3))}
    out = []
    for seed in range(n_graphs):
        rec = T.build_mol_record(
            synthetic_zinc_like(seed, 10 + seed), layout, atom_index, bond_index
        )
        rng = np.random.default_rng(100 + seed)
        rec["X"] = rng.standard_normal((int(rec["n"]), V.SELECTED_WIDTH)).astype(np.float32)
        rec["y"] = float(seed) / 3.0
        out.append(rec)
    return out


def test_selected_tensor_is_full_pre_pair_patch_encoder_output():
    model = V._source_model("cpu")
    inventory = V._model_inventory(model)
    assert inventory["selected_layer"] == "patch_encoder"
    assert inventory["is_patch_encoder_output"] is True
    assert inventory["is_patch_encoder_input_block"] is False
    assert inventory["patch_encoder_input_width"] == 170
    assert inventory["patch_encoder_output_width"] == 64
    assert inventory["pair_projection_input_width"] == 64
    assert inventory["input_blocks_sum"] == 170


def test_gate0_passes_correct_bridge_invariants_and_freeze():
    checks = V.gate0_checks("cpu", require_cache=False)
    assert checks["official_test_loaded"] is False
    assert checks["is_patch_encoder_output"] is True
    assert checks["is_patch_encoder_input_block"] is False
    assert checks["exterior_invariant"] is True
    assert checks["permutation_invariant"] is True
    assert checks["batch_invariant"] is True
    assert checks["deterministic_reload"] is True
    assert checks["pair_global_independent"] is True
    assert checks["frozen_parameters"] is True
    assert checks["source_parameters_receive_no_gradient"] is True
    assert checks["all_pass"] is True


def test_both_arms_receive_task_gradient_with_matched_64_adapter():
    records = _records()
    batch = V.V3.make_batch(records, [0, 1, 2], "cpu")
    for arm in ("dense", "prototype"):
        model = V.build_model(arm, seed=0)
        assert model.adapter.in_features == 64
        assert model.adapter.out_features == 64
        pred, _, _ = model.forward_padded(batch)
        loss = (pred - batch["y"]).abs().mean()
        loss.backward()
        assert model.adapter.weight.grad is not None
        assert float(model.adapter.weight.grad.abs().sum()) > 0.0
        assert model.head.weight.grad is not None
        assert float(model.head.weight.grad.abs().sum()) > 0.0
    proto = V.build_model("prototype", seed=0)
    proto_pred, _, _ = proto.forward_padded(batch)
    proto_loss = (proto_pred - batch["y"]).abs().mean()
    proto_loss.backward()
    assert proto.P.grad is not None
    assert proto.temp_logit.grad is not None


def test_prototype_reader_uses_assignment_composition_only():
    records = _records(2)
    model = V.build_model("prototype", seed=0)
    batch = V.V3.make_batch(records, [0, 1], "cpu")
    with torch.no_grad():
        pred, assignments, features = model.forward_padded(batch)
        rebuilt = model.head(model.graph_features(assignments, batch)).reshape(-1)
    assert torch.equal(pred, rebuilt)
    assert model.uses_latent_bypass is False
    assert model.graph_feature_kind == "assignment_composition_only"
    assert model.bridge_kind == "correct_fused_prepair_patch_encoder_output"
    assert assignments.shape[-1] == V.K_PROTO
    assert features.shape[-1] == T.h_dim(V.K_PROTO, V.N_REL)


def test_official_test_is_rejected_at_cache_api_boundary():
    try:
        V.load_or_build_prepair_cache("test", "cpu")
    except ValueError as exc:
        assert "official test" in str(exc)
    else:
        raise AssertionError("official test cache request was not rejected")

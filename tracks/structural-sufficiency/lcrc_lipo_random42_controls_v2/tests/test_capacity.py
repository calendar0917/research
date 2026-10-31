from __future__ import annotations

import torch

import c2_common as C
import c2_models as CM

import lcrc_data
import v2_models as V2M


def test_capacity_arithmetic_and_shared_init(lcrc_views):
    spec = lcrc_views["spec"]
    norm = lcrc_views["train_norm"]
    a_model = V2M.build_a(spec, norm, C.TRAIN_SEED)
    e_model, _base, audit = CM.build_s2off(spec, norm, C.TRAIN_SEED)
    assert sum(p.numel() for p in a_model.parameters()) == 247_106
    assert int(audit["capacity"]["nominal_parameters"]) == 285_449
    assert int(audit["capacity"]["effective_parameters"]) == 247_049
    assert int(audit["capacity"]["dead_localmlp_s2_input_columns"]) == 38_400
    assert int(audit["capacity"]["replacement_parameters"]) == 70_535
    assert audit["capacity"]["effective_equals_B_plus_replacement"] is True
    assert audit["shared_init"]["all_shared_bitwise_identical"] is True
    assert int(audit["shared_init"]["n_shared_entries"]) == 27
    assert torch.equal(a_model.local.mlp.lins[0].weight, e_model.local.mlp.lins[0].weight)
    assert not any("path_" in key for key in e_model.state_dict())


def test_typed_s2_invariance(lcrc_views):
    spec = lcrc_views["spec"]
    norm = lcrc_views["train_norm"]
    e_model, _base, _audit = CM.build_s2off(spec, norm, C.TRAIN_SEED)
    batch = lcrc_data.collate(lcrc_views["train_cache"], list(range(16)))
    gate = CM.s2off_invariance(e_model, batch)
    assert gate["identical_restricted_batch"] is True
    assert gate["identical_perturbed_s2"] is True
    assert gate["max_abs_diff_perturbed"] == 0.0
    assert gate["typed_s2_access_raises"] is True


def test_gradient_placement(lcrc_views):
    spec = lcrc_views["spec"]
    norm = lcrc_views["train_norm"]
    e_model, _base, _audit = CM.build_s2off(spec, norm, C.TRAIN_SEED)
    batch = lcrc_data.collate(lcrc_views["train_cache"], list(range(16)))
    gate = CM.s2off_gradient_gate(e_model, batch)
    assert gate["all_replacement_receive_nonzero_grad"] is True
    assert gate["dead_s2_input_columns_zero_grad"] is True
    assert gate["dead_s2_input_columns_max_abs_grad"] == 0.0
    assert gate["live_localmlp_columns_receive_grad"] is True

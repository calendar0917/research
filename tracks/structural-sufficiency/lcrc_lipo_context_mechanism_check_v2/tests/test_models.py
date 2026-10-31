from __future__ import annotations

import torch

import m2_common as C
import m2_models as M


def test_roleblind_init_equals_context(views):
    spec = views["spec"]
    norm = views["train_norm"]
    context, _base, context_meta = M.build_context(spec, norm, 0, "cpu")
    roleblind, _context, roleblind_meta = M.build_roleblind(spec, norm, 0, "cpu")
    parity = M.init_parity(context, roleblind)
    assert parity["all_entries_bitwise_identical"]
    assert parity["all_parameters_bitwise_identical"]
    assert parity["state_dict_keys_identical"]
    assert parity["parameter_names_identical"]
    assert context_meta["parameters"] == roleblind_meta["parameters"] == C.CONTEXT_TOTAL_PARAMS_REFERENCE
    assert context_meta["parameters_new"] == roleblind_meta["parameters_new"] == C.CONTEXT_NEW_PARAMS_REFERENCE


def test_base_and_context_init(views):
    spec = views["spec"]
    norm = views["train_norm"]
    base, base_meta = M.build_base(spec, norm, 1, "cpu")
    context, _base, context_meta = M.build_context(spec, norm, 1, "cpu")
    shared = M.shared_init_audit(base, context)
    assert shared["all_shared_bitwise_identical"]
    assert base_meta["parameters"] == C.LCRC_BASE_PARAMS_REFERENCE
    assert context_meta["parameters"] == C.CONTEXT_TOTAL_PARAMS_REFERENCE


def test_roleblind_all_atom_aggregation(views):
    spec = views["spec"]
    norm = views["train_norm"]
    roleblind, _context, _meta = M.build_roleblind(spec, norm, 0, "cpu")
    audit = M.roleblind_aggregation_audit(roleblind, views["valid_cache"], list(range(8)), "cpu")
    assert audit["passed"]
    assert audit["m_het_equals_m_ch_bitwise"]
    assert audit["non_hetero_atom_delta_max_abs"] == 0.0
    assert audit["all_atom_mean_vs_hetero_only_mean_max_abs_diff"] > 0.0


def test_roleblind_batch_invariance(views):
    spec = views["spec"]
    norm = views["train_norm"]
    roleblind, _context, _meta = M.build_roleblind(spec, norm, 0, "cpu")
    audit = M.roleblind_batch_invariance_audit(roleblind, views["valid_cache"], list(range(16)), "cpu", subset_size=16)
    assert audit["passed"]


def test_test_inference_does_not_update_weights(views):
    spec = views["spec"]
    norm = views["train_norm"]
    roleblind, _context, _meta = M.build_roleblind(spec, norm, 0, "cpu")
    audit = M.freeze_weights_audit(roleblind, views["test_cache"], "cpu")
    assert audit["passed"]
    assert audit["n_changed_state_entries"] == 0

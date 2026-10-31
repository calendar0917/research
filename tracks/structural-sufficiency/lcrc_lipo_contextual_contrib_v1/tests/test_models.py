"""BASE/CONTEXT model-definition and invariance tests."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import x1_common as C
import x1_models as M


ROWS = list(range(8))


def test_parameter_counts(models):
    assert models["base_info"]["parameters"] == C.LCRC_BASE_PARAMS_REFERENCE == 247_106
    assert models["context_info"]["parameters_new"] == C.CONTEXT_NEW_PARAMS_REFERENCE == 8_352
    assert models["context_info"]["parameters"] == C.CONTEXT_TOTAL_PARAMS_REFERENCE == 255_458
    assert set(M.new_parameter_names(models["context"])) == {
        "context.W_atom.weight",
        "context.W_atom.bias",
        "context.W_het.weight",
        "context.W_het.bias",
        "context.W_ch.weight",
        "context.W_ch.bias",
        "context.W_out.weight",
        "context.W_out.bias",
    }


def test_base_matches_original_lcrc(models, train_cache):
    audit = M.base_matches_original_lcrc(models["spec"], models["norm"], C.TRAIN_SEED, train_cache, ROWS)
    assert audit["passed"], audit
    assert audit["max_abs_forward_diff"] == 0.0


def test_shared_initialisation_bitwise(models):
    audit = models["context_info"]["shared_init"]
    assert audit["all_shared_bitwise_identical"], audit["mismatched"]
    assert audit["n_shared_entries"] >= 30
    assert not audit["base_only_keys"]


def test_zero_out_equivalence(models, train_cache):
    audit = M.zero_out_equivalence_audit(models["base_ref"], models["context"], train_cache, ROWS)
    assert audit["passed"], audit
    assert audit["max_abs_diff_base_vs_zeroed_context"] <= 1.0e-5
    assert audit["initial_w_out_is_not_degenerate"]


def test_role_masks(models, train_cache):
    audit = M.role_mask_audit(models["spec"], models["context"], train_cache, ROWS)
    assert audit["passed"], audit
    assert audit["hetero_lut_matches_expected"]
    assert audit["carbon_halogen_lut_matches_expected"]


def test_renumber_and_reorder(models, train_cache):
    renumber = M.renumber_invariance_audit(models["context"], train_cache, ROWS)
    reorder = M.molecule_reorder_audit(models["context"], train_cache, ROWS)
    assert renumber["passed"], renumber
    assert reorder["passed"], reorder


def test_no_context_mixing(models, train_cache):
    audit = M.context_no_mixing_audit(models["context"], train_cache, ROWS)
    assert audit["passed"], audit
    assert audit["max_context_diff_single_vs_batch"] == 0.0
    assert audit["other_molecule_prediction_diff_after_perturbation"] == 0.0


def test_rho_reads_original_h(models, train_cache):
    audit = M.rho_input_audit(models["context"], train_cache, ROWS)
    assert audit["passed"], audit
    assert audit["rho_input_matches_h_original"]
    assert audit["discriminating"]


def test_dropout_hook_is_executed(models, train_cache):
    for arm in ("base", "context"):
        audit = M.dropout_activity_audit(models[arm], train_cache, ROWS)
        assert audit["passed"], (arm, audit)
        assert audit["dropout_p"] == 0.10
        assert audit["train_mode_masked_atoms"]
        assert audit["eval_mode_is_identity"]
        assert audit["head_calls_train"] == 2 and audit["head_calls_eval"] == 2


def test_context_updates_hetero_atoms_only(models, train_cache):
    model = models["context"]
    model.eval()
    import lcrc_data

    batch = lcrc_data.collate(train_cache, ROWS)
    with torch.no_grad():
        _y, aux = model(batch, return_aux=True)
    het = aux["het_mask"]
    h_orig = aux["h_original"]
    h_new = aux["h_new"]
    assert torch.equal(h_new[~het], h_orig[~het])
    assert torch.allclose(h_new[het], h_orig[het] + aux["change"][het], atol=1.0e-6, rtol=0.0)
    assert int(het.sum()) > 0


def test_empty_role_mean_is_exactly_zero(models):
    module = models["context"].context
    h = torch.randn(5, C.WIDTH)
    atom_type = torch.full((5,), 14, dtype=torch.long)  # Si: neither role
    atom_batch = torch.zeros(5, dtype=torch.long)
    h_new, aux = module(h, atom_type, atom_batch, 1)
    assert torch.equal(aux["m_het"], torch.zeros(1, C.WIDTH))
    assert torch.equal(aux["m_ch"], torch.zeros(1, C.WIDTH))
    assert torch.equal(h_new, h)


def test_w_out_initialisation(models):
    w_out = models["context"].context.W_out
    assert w_out.bias.abs().max().item() == 0.0
    std = float(w_out.weight.detach().std())
    assert abs(std - C.W_OUT_INIT_STD) < 5.0e-5, std


def test_new_layer_gradients_after_two_steps(models, train_cache):
    probe, _base, _meta = M.build_context(models["spec"], models["norm"], C.TRAIN_SEED)
    audit = M.gradient_probe(probe, train_cache, list(range(32)), steps=2)
    assert audit["passed"], audit
    assert audit["all_new_tensors_nonzero_gradient_at_last_probe_step"]
    assert audit["all_new_layers_finite_gradient"]
    for name in M.new_parameter_names(probe):
        assert audit["history"][1]["grads"][name]["norm"] > 0.0


def test_context_module_has_no_forbidden_components(models):
    names = [name for name, _ in models["context"].named_modules()]
    assert not any("attention" in n for n in names)
    assert not any("prototype" in n for n in names)
    assert not any("junction" in n for n in names)
    assert not any("message" in n for n in names)


def test_only_allowed_new_state_entries(models):
    base_keys = set(models["base_ref"].state_dict())
    context_keys = set(models["context"].state_dict())
    new_keys = context_keys - base_keys
    assert new_keys == {
        "context.W_atom.weight",
        "context.W_atom.bias",
        "context.W_het.weight",
        "context.W_het.bias",
        "context.W_ch.weight",
        "context.W_ch.bias",
        "context.W_out.weight",
        "context.W_out.bias",
    }

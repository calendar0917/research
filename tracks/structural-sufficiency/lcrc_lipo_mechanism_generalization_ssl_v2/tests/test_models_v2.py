from __future__ import annotations

import numpy as np
import pytest
import torch

import v2_common as C
import v2_models as M


def test_capacity_matched_E_arithmetic(spec, norm):
    a = M.build_a(spec, norm, 0)
    e, base, _shared_state = M.build_e(spec, norm, 0)
    audit = M.e_capacity_audit(e, spec)
    shared = M.shared_init_audit(base, e)
    assert M.parameter_count(a) == 247_106
    assert audit["nominal_parameters"] == 285_449
    assert audit["dead_localmlp_s2_input_columns"] == 38_400
    assert audit["effective_parameters"] == 247_049
    assert audit["delta_effective_vs_A"] == -57
    assert abs(audit["delta_fraction_vs_A"]) < 0.001
    assert audit["replacement_parameters"] == 70_535
    assert audit["effective_equals_B_plus_replacement"] is True
    assert audit["typed_s2_inputs_used"] is False
    assert audit["iterative_hidden_message_passing"] is False
    assert shared["all_shared_bitwise_identical"] is True
    assert shared["n_shared_entries"] == 27


def test_E_has_no_typed_s2_dependence(spec, norm, fit_cache):
    import v2_data as D

    e, base, _ = M.build_e(spec, norm, 0)
    e.eval()
    batch = D.collate(fit_cache, list(range(24)))
    with torch.no_grad():
        y0 = e(batch)
        y_restricted = e(M.restricted_batch(batch))
    assert torch.equal(y0, y_restricted)
    perturbed = dict(batch)
    for key, hi in (
        ("path_cat", int(spec.n_path_codes)),
        ("path_end63", int(spec.n_nbr_codes)),
        ("path_end", int(batch["atom_feat"].shape[0])),
        ("path_closure", 2),
    ):
        perturbed[key] = torch.randint_like(batch[key], 0, hi)
    with torch.no_grad():
        y1 = e(perturbed)
    assert torch.equal(y0, y1)
    with pytest.raises(KeyError):
        M.restricted_batch(batch)["path_cat"]


def test_E_replacement_receives_gradient_and_dead_slot_does_not(spec, norm, fit_cache):
    import v2_data as D

    e, _base, _shared = M.build_e(spec, norm, 0)
    e.train()
    batch = D.collate(fit_cache, list(range(16)))
    loss = e(batch).pow(2).mean()
    loss.backward()
    names = [n for n, _ in e.local.replacement.named_parameters()]
    assert names
    for name, p in e.local.replacement.named_parameters():
        assert p.grad is not None
        assert float(p.grad.abs().max()) > 0.0
    dead = e.local.mlp.lins[0].weight.grad[:, 320:520]
    assert dead is not None
    assert float(dead.abs().max()) == 0.0


def test_shared_init_matches_A_same_seed(spec, norm):
    for seed in (0, 1, 2):
        a = M.build_a(spec, norm, seed)
        e, base, _shared_state = M.build_e(spec, norm, seed)
        shared = M.shared_init_audit(base, e)
        assert shared["all_shared_bitwise_identical"], shared["mismatched"]
        assert shared["n_shared_entries"] == 27
        # The S2 input slot is zeroed at runtime, not by rewriting the shared
        # initial weights; the dead columns therefore stay bit-identical to A.
        assert torch.equal(a.local.mlp.lins[0].weight, e.local.mlp.lins[0].weight)


def test_dropout_R0_equivalence_and_eval_disable(spec, norm, fit_cache):
    import v2_data as D

    batch = D.collate(fit_cache, list(range(16)))
    for arm in ("A", "C"):
        base = M.build_a(spec, norm, 0) if arm == "A" else M.build_c(spec, norm, 0)
        r0, _ = M.build_model(arm, spec, norm, 0, dropout=0.0)
        for mode in ("train", "eval"):
            getattr(base, mode)()
            getattr(r0, mode)()
            with torch.no_grad():
                assert torch.equal(base(batch), r0(batch))
        p20, meta = M.build_model(arm, spec, norm, 0, dropout=0.2)
        p20.train()
        torch.manual_seed(0)
        with torch.no_grad():
            train_out = p20(batch)
        p20.eval()
        with torch.no_grad():
            eval_out = p20(batch)
            base.eval()
            base_out = base(batch)
        assert torch.equal(eval_out, base_out)
        assert meta["dropout_hook"].calls > 0
        assert not any("dropout" in k for k in p20.state_dict())


def test_E_capacity_audit_is_not_literal_529(spec, norm):
    e, _base, _shared = M.build_e(spec, norm, 0)
    audit = M.e_capacity_audit(e, spec)
    # A literal 200-wide replacement loaded into a 529-wide LocalMLP would
    # revive the 38,400-column S2 slot; the implementation explicitly avoids
    # that and documents the effective count.
    assert audit["literal_529_input_live_parameters_if_slot_not_dead"] == 285_449
    assert audit["effective_parameters"] != audit["literal_529_input_live_parameters_if_slot_not_dead"]

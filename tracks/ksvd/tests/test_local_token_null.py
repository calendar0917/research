"""Targeted tests for the local 16-D patch-token channel-necessity experiment.

Fast, self-contained unit tests: no ZINC data, no training, official test never
loaded.  Requirements covered (numbered as in the experiment brief):

1.  parameter accounting is exact (Null 49,343; Constant 49,359; references
    85,763 / 84,511 / 84,495; shared downstream backbone 49,343)
2.  the Null model instantiates no local-token generator at all
3.  the downstream patch-encoder input width is unchanged
4.  Zero-16 intervention on the frozen reference == the Null model forward
5.  Constant-16 intervention == the Constant model forward
6.  the constant is a single trainable 16-vector (16 params)
7.  the permutation intervention preserves the per-molecule token multiset
8.  disabling the intervention restores the frozen forward exactly
9.  all remaining major backbone groups receive a non-zero task gradient
10. no hidden local-token path survives in the Null model
"""

from __future__ import annotations

import torch

from tracks.ksvd.experiments.luyin16 import zinc_local_token_null as ltn
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_identity_capacity_control as ic,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _batch():
    return ic._synthetic_batch()


# ---------------------------------------------------------------------------
# 1. parameter accounting
# ---------------------------------------------------------------------------


def test_parameter_accounting_exact():
    accounting = ltn.parameter_accounting()
    assert accounting["total_candidate"]["Null"] == 49343
    assert accounting["total_candidate"]["Constant"] == 49359
    assert accounting["total_reference"] == {
        "A0": 85763,
        "B-Bag": 84511,
        "B-Full": 84495,
    }
    assert accounting["downstream_backbone_params"] == 49343
    assert accounting["removed_local_token_generator"] == {
        "A0": 36420,
        "B-Bag": 35168,
        "B-Full": 35152,
    }
    assert accounting["null_parameter_delta_vs"] == {
        "A0": -36420,
        "B-Bag": -35168,
        "B-Full": -35152,
    }
    assert accounting["official_test_loaded"] is False


# ---------------------------------------------------------------------------
# 2. no generator survives in the Null model
# ---------------------------------------------------------------------------


def test_null_has_no_local_token_generator():
    model = ltn.build_null(0)
    assert model.typed_embedding is None
    assert model.structural_encoder is None
    assert model.local_token_constant is None
    names = [name for name, _ in model.named_parameters()]
    assert not any(name.startswith("typed_embedding") for name in names)
    assert not any(name.startswith("structural_encoder") for name in names)
    assert not any(name.startswith("local_token_constant") for name in names)


def test_constant_has_single_trainable_vector():
    model = ltn.build_constant(0)
    assert model.typed_embedding is None
    assert model.structural_encoder is None
    assert model.local_token_constant is not None
    assert tuple(model.local_token_constant.shape) == (16,)
    assert model.local_token_constant.requires_grad
    assert ltn._n_params(model) == 49359


# ---------------------------------------------------------------------------
# 3. downstream shape unchanged
# ---------------------------------------------------------------------------


def test_patch_encoder_input_width_unchanged():
    null = ltn.build_null(0)
    typed = ltn.build_reference("A0", 0)
    assert int(null.patch_encoder.layers[0].in_features) == int(
        typed.patch_encoder.layers[0].in_features
    )


# ---------------------------------------------------------------------------
# 4./5. interventions reproduce the null / constant models
# ---------------------------------------------------------------------------


def test_zero_intervention_equals_null_forward():
    batch = _batch()
    null = ltn.build_null(0).eval()
    typed = ltn.build_reference("A0", 0).eval()
    with torch.no_grad():
        null_pred = null(batch)
        typed.set_local_token_intervention("zero")
        zero_pred = typed(batch)
        typed.clear_local_token_intervention()
    assert torch.equal(null_pred, zero_pred)


def test_constant_intervention_equals_constant_model():
    batch = _batch()
    constant = ltn.build_constant(0).eval()
    typed = ltn.build_reference("A0", 0).eval()
    value = torch.arange(16, dtype=torch.float32)
    with torch.no_grad():
        constant.local_token_constant.data.copy_(value)
        constant_pred = constant(batch)
        typed.set_local_token_intervention("constant", constant=value)
        intervention_pred = typed(batch)
        typed.clear_local_token_intervention()
    assert torch.equal(constant_pred, intervention_pred)


def test_null_forward_is_exact_zero_token():
    batch = _batch()
    null = ltn.build_null(0).eval()
    with torch.no_grad():
        token = null._patch_token_value(batch)
    assert token.shape == (int(batch.patch_cont.shape[0]), 16)
    assert torch.equal(token, torch.zeros_like(token))


# ---------------------------------------------------------------------------
# 6. permutation preserves the per-molecule token multiset
# ---------------------------------------------------------------------------


def test_permutation_preserves_multiset():
    batch = _batch()
    model = ltn.build_null(0).eval()
    with torch.no_grad():
        base = model._patch_token_value(batch)
    # Use a constant multiset so every patch token is distinct per patch order.
    model.set_local_token_intervention(
        "permute", permute_seed=1234
    )
    # Directly exercise the intervention on a synthetic value.
    value = torch.arange(base.shape[0] * 16, dtype=torch.float32).view(
        base.shape[0], 16
    )
    with torch.no_grad():
        permuted = model._apply_local_token_intervention(value, batch)
    model.clear_local_token_intervention()
    batch_cpu = batch.batch.detach().cpu()
    for graph_id in torch.unique(batch_cpu).tolist():
        index = torch.nonzero(batch_cpu == int(graph_id), as_tuple=False).view(-1)
        original = value[index].sum(dim=0)
        changed = permuted[index].sum(dim=0)
        assert torch.allclose(original, changed)


# ---------------------------------------------------------------------------
# 7. disabling the intervention restores the frozen forward exactly
# ---------------------------------------------------------------------------


def test_intervention_off_is_frozen_forward():
    batch = _batch()
    typed = ltn.build_reference("A0", 0).eval()
    with torch.no_grad():
        before = typed(batch)
        typed.set_local_token_intervention("zero")
        _ = typed(batch)
        typed.clear_local_token_intervention()
        after = typed(batch)
    assert torch.equal(before, after)
    assert typed.local_token_intervention is None


# ---------------------------------------------------------------------------
# 8. gradient reaches every remaining major backbone group
# ---------------------------------------------------------------------------


def test_null_gradient_reaches_all_groups():
    batch = _batch()
    model = ltn.build_null(0)
    audit = ltn._gradient_audit(model, batch, torch.device("cpu"))
    assert audit["all_groups_nonzero"]
    for name, row in audit["groups"].items():
        if row["params"] > 0:
            assert row["grad_norm"] > 0.0, name


# ---------------------------------------------------------------------------
# 9. no hidden local-token path / no forbidden feature family in the forward
# ---------------------------------------------------------------------------


def test_no_hidden_local_token_path():
    model = ltn.build_null(0)
    modules = [type(module).__name__ for module in model.modules()]
    assert "SharedStructuralPatchEncoder" not in modules
    assert "SharedBagPatchEncoder" not in modules
    assert "IdentityTokenChannel" not in modules
    names = [name for name, _ in model.named_parameters()]
    assert not any("local_token" in name for name in names)

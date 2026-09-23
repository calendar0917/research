"""Targeted tests for FEC-S1 (shared local-environment replacement).

Audit-style tests: they build the FEC-S1 model, verify the two per-key lookups
are gone, the shared adapter reads exactly the factorized standardized
``patch_cont``, token poisoning / vocabulary changes leave the prediction
invariant, the adapter receives task gradient, and the strict-static contract
holds.  No training, no official test.
"""

from __future__ import annotations

import torch

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16.fec_s1_shared_local_env import (
    adapter_param_count,
    build_fec_s1,
    choose_local_env_hidden,
    lookup_param_count,
)
from tracks.ksvd.experiments.luyin16.zinc_static_dictionary_pair import (
    _first_batch,
    load_encoded,
    static_contract_checks,
)

_SETUP: dict | None = None


def _setup() -> dict:
    global _SETUP
    if _SETUP is None:
        _train, valid_data, _audit = load_encoded(valid_subset=64)
        batch = _first_batch(valid_data, torch.device("cpu"), limit=64)
        model = build_fec_s1(seed=0).eval()
        _SETUP = {"batch": batch, "model": model}
    return _SETUP


def test_no_lookup_params_and_parameter_fairness() -> None:
    lookup = lookup_param_count()
    match = choose_local_env_hidden(int(lookup["p_lookup"]))
    model = build_fec_s1(seed=0, hidden=int(match["hidden"]))
    keys = list(model.state_dict().keys())
    assert model.typed_embedding is None
    assert model.parent_embedding is None
    for key in keys:
        assert "typed_embedding" not in key
        assert "parent_embedding" not in key
        assert "token_table" not in key
        assert "certificate_embedding" not in key
    actual = sum(p.numel() for p in model.local_env_adapter.parameters())
    assert actual == adapter_param_count(int(match["hidden"]))
    assert abs(actual - int(lookup["p_lookup"])) / float(lookup["p_lookup"]) <= 0.01


def test_adapter_input_is_factorized_patch_cont_bit_identical() -> None:
    setup = _setup()
    model, batch = setup["model"], setup["batch"]
    captured: dict[str, torch.Tensor] = {}

    def hook(_module, inputs, _output):
        captured["input"] = inputs[0].detach().clone()

    handle = model.local_env_adapter.register_forward_hook(hook)
    try:
        with torch.no_grad():
            model(batch)
    finally:
        handle.remove()
    assert torch.equal(captured["input"], batch.patch_cont)


def test_token_poisoning_leaves_prediction_unchanged() -> None:
    setup = _setup()
    model, batch = setup["model"], setup["batch"]
    with torch.no_grad():
        base = model(batch).view(-1)
    poison = batch.clone()
    poison.typed_token = torch.full_like(poison.typed_token, 10**9)
    poison.parent_token = torch.full_like(poison.parent_token, 10**9)
    with torch.no_grad():
        other = model(poison).view(-1)
    assert torch.equal(base, other)


def test_vocabulary_independence() -> None:
    import torch as _torch

    from tracks.ksvd.experiments.luyin16.zinc_static_dictionary_pair import (
        StrictStaticPairModel,
        base_model_kwargs,
    )

    setup = _setup()
    model, batch = setup["model"], setup["batch"]
    hidden = int(model.local_env_adapter.hidden)
    alt = StrictStaticPairModel(
        9000,
        64,
        residual_mode="none",
        patch_representation="shared_local_env",
        local_env_hidden=hidden,
        **base_model_kwargs(),
    )
    alt.load_state_dict(model.state_dict(), strict=False)
    alt.eval()
    poison = batch.clone()
    poison.typed_token = _torch.full_like(poison.typed_token, 10**9)
    poison.parent_token = _torch.full_like(poison.parent_token, 10**9)
    with _torch.no_grad():
        base = model(batch).view(-1)
        other = alt(poison).view(-1)
    assert _torch.equal(base, other)


def test_adapter_receives_task_gradient() -> None:
    setup = _setup()
    model, batch = setup["model"], setup["batch"]
    model.train()
    model.zero_grad(set_to_none=True)
    loss = torch.nn.functional.l1_loss(model(batch).view(-1), batch.y.view(-1))
    loss.backward()
    linear0 = model.local_env_adapter.net[0]
    linear1 = model.local_env_adapter.net[2]
    for parameter in (linear0.weight, linear0.bias, linear1.weight, linear1.bias):
        assert parameter.grad is not None
        assert torch.isfinite(parameter.grad).all()
    assert float(linear0.weight.grad.norm()) > 0.0
    assert float(linear1.weight.grad.norm()) > 0.0
    model.zero_grad(set_to_none=True)
    model.eval()


def test_strict_static_no_pair_to_centre_and_once_only() -> None:
    setup = _setup()
    model, batch = setup["model"], setup["batch"]
    contract = static_contract_checks(model, batch)
    assert contract["passed"]
    assert contract["pair_encoder_calls_per_forward"] == 1
    assert contract["relation_encoder_calls_per_forward"] == 1
    assert contract["h_identical_under_relation_mutation"]


def test_shared_local_env_rejects_missing_hidden() -> None:
    from tracks.ksvd.experiments.luyin16.zinc_static_dictionary_pair import (
        PARENT_VOCAB_SIZE,
        TYPED_VOCAB_SIZE,
        StrictStaticPairModel,
        base_model_kwargs,
    )

    try:
        StrictStaticPairModel(
            TYPED_VOCAB_SIZE,
            PARENT_VOCAB_SIZE,
            residual_mode="none",
            patch_representation="shared_local_env",
            **base_model_kwargs(),
        )
    except ValueError:
        return
    raise AssertionError("missing local_env_hidden must raise")


def test_s0_typed_lookup_path_still_available() -> None:
    """The new mode must not disturb the historical S0 representation."""
    from tracks.ksvd.experiments.luyin16.zinc_static_dictionary_pair import build_s0

    s0 = build_s0(0)
    assert s0.typed_embedding is not None
    assert s0.parent_embedding is not None
    assert s0.local_env_adapter is None
    total = sum(p.numel() for p in s0.parameters())
    assert total == 66228


def test_adapter_uses_only_patch_cont() -> None:
    """Adapter weights are the only local-channel parameters (24-D output)."""
    setup = _setup()
    model = setup["model"]
    assert model.local_env_adapter.input_width == zpp.SHELL_WIDTH
    assert model.local_env_adapter.output_width == 24
    assert model.local_env_adapter.hidden == 214

from __future__ import annotations

import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import molhiv_local_token_null as ltn
from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as mpp
from tracks.ksvd.experiments.luyin16 import molhiv_recurrent_pair_centre as rpc

TOKEN_WIDTH = 6


def _toy_batch() -> Data:
    generator = torch.Generator().manual_seed(23)
    return Data(
        patch_cont=torch.randn(3, mpp.SHELL_WIDTH, generator=generator),
        typed_token=torch.tensor([1, 2, 3], dtype=torch.long),
        parent_token=torch.tensor([1, 1, 2], dtype=torch.long),
        pair_index=torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
        pair_relation=torch.randn(3, mpp.RELATION_WIDTH, generator=generator),
        pair_bucket=torch.tensor([0, 1, 0], dtype=torch.long),
        global_context=torch.randn(1, mpp.GLOBAL_WIDTH, generator=generator),
        batch=torch.zeros(3, dtype=torch.long),
        y=torch.tensor([0.0]),
        num_nodes=3,
    )


def _model(representation: str = "typed_lookup") -> mpp.PatchPathModel:
    return mpp.PatchPathModel(
        8,
        6,
        patch_hidden=8,
        pair_hidden=4,
        token_width=TOKEN_WIDTH,
        dropout=0.0,
        center_context=True,
        center_context_hidden=16,
        patch_representation=representation,
    )


def test_default_representation_is_typed_lookup() -> None:
    model = _model()
    assert model.patch_representation == "typed_lookup"
    assert model.typed_embedding is not None
    assert model.local_token_constant is None


def test_null_has_no_generator_and_exact_zero_token() -> None:
    model = _model("null")
    assert model.typed_embedding is None
    assert model.local_token_constant is None
    token = model._patch_token_value(_toy_batch())
    assert token.shape == (3, TOKEN_WIDTH)
    assert bool(torch.all(token == 0).item())


def test_constant_is_one_shared_trainable_vector() -> None:
    model = _model("constant")
    assert model.typed_embedding is None
    assert model.local_token_constant is not None
    token = model._patch_token_value(_toy_batch())
    assert token.shape == (3, TOKEN_WIDTH)
    assert bool(torch.all(token == token[0]).item())
    assert model.local_token_constant.requires_grad
    # 16/32-D constant is exactly token_width trainable parameters
    assert model.local_token_constant.numel() == TOKEN_WIDTH


def test_parameter_drop_equals_lookup_size() -> None:
    typed = _model("typed_lookup")
    null = _model("null")
    constant = _model("constant")
    typed_total = sum(p.numel() for p in typed.parameters())
    null_total = sum(p.numel() for p in null.parameters())
    constant_total = sum(p.numel() for p in constant.parameters())
    assert typed_total - null_total == typed.typed_embedding.weight.numel()
    assert constant_total - null_total == TOKEN_WIDTH
    assert null.parameter_breakdown()["total"] == null_total
    assert constant.parameter_breakdown()["total"] == constant_total


def test_patch_encoder_input_width_is_unchanged() -> None:
    widths = {
        representation: int(
            _model(representation).patch_encoder.layers[0].in_features
        )
        for representation in ("typed_lookup", "null", "constant")
    }
    assert len(set(widths.values())) == 1
    assert widths["null"] == mpp.SHELL_WIDTH + TOKEN_WIDTH + max(TOKEN_WIDTH // 2, 1)


def test_null_forward_equals_zeroed_typed_lookup_forward() -> None:
    torch.manual_seed(11)
    baseline = _model("typed_lookup")
    with torch.no_grad():
        baseline.typed_embedding.weight.zero_()
    null = _model("null")
    incompatible = null.load_state_dict(baseline.state_dict(), strict=False)
    assert all(key.startswith("typed_embedding.") for key in incompatible.unexpected_keys)
    assert not incompatible.missing_keys
    baseline.eval()
    null.eval()
    batch = _toy_batch()
    with torch.no_grad():
        torch.testing.assert_close(null(batch), baseline(batch), rtol=0.0, atol=0.0)


def test_unknown_representation_is_rejected() -> None:
    with pytest.raises(ValueError):
        _model("shared_structural")


def test_gate_bands() -> None:
    assert ltn._gate(0.0) == "STRONG_LOCAL_TOKEN_CHANNEL_NOT_REQUIRED"
    assert ltn._gate(0.005) == "STRONG_LOCAL_TOKEN_CHANNEL_NOT_REQUIRED"
    assert ltn._gate(0.0051) == "MILD_LOCAL_TOKEN_CHANNEL_NEARLY_REDUNDANT"
    assert ltn._gate(0.02) == "MILD_LOCAL_TOKEN_CHANNEL_NEARLY_REDUNDANT"
    assert ltn._gate(0.021) == "SUBSTANTIAL_LOCAL_TOKEN_CHANNEL_CONTRIBUTES"
    assert ltn._gate(0.05) == "SUBSTANTIAL_LOCAL_TOKEN_CHANNEL_CONTRIBUTES"
    assert ltn._gate(0.0501) == "STOP_LOCAL_TOKEN_CHANNEL_REQUIRED"


def test_train_refuses_the_frozen_reference() -> None:
    with pytest.raises(ValueError):
        ltn.train_seed("typed_lookup", seed=0, device="cpu")


def test_recurrent_model_passes_representation_through() -> None:
    model = rpc.MolhivRecurrentPairCentreModel(
        8,
        6,
        patch_hidden=8,
        pair_hidden=4,
        token_width=TOKEN_WIDTH,
        dropout=0.0,
        center_context=True,
        center_context_hidden=16,
        recurrence_rounds=2,
        patch_representation="null",
    )
    assert model.patch_representation == "null"
    assert model.typed_embedding is None
    assert model._patch_token_value(_toy_batch()).abs().max() == 0.0


def test_build_model_accepts_representation() -> None:
    model = rpc.build_model(8, 6, 0, patch_representation="constant")
    assert model.typed_embedding is None
    assert model.local_token_constant is not None
    assert model.local_token_constant.numel() == rpc.TOKEN_WIDTH

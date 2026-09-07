from __future__ import annotations

import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import molhiv_patch_path_pooling as module


def _toy_batch() -> Data:
    generator = torch.Generator().manual_seed(23)
    return Data(
        patch_cont=torch.randn(3, module.SHELL_WIDTH, generator=generator),
        typed_token=torch.tensor([1, 2, 3], dtype=torch.long),
        parent_token=torch.tensor([1, 1, 2], dtype=torch.long),
        pair_index=torch.tensor([[0, 0, 1], [1, 2, 2]], dtype=torch.long),
        pair_relation=torch.randn(
            3, module.RELATION_WIDTH, generator=generator
        ),
        pair_bucket=torch.tensor([0, 1, 0], dtype=torch.long),
        global_context=torch.randn(
            1, module.GLOBAL_WIDTH, generator=generator
        ),
        batch=torch.zeros(3, dtype=torch.long),
        y=torch.tensor([0.0]),
        num_nodes=3,
    )


def _model(*, center_context: bool) -> module.PatchPathModel:
    return module.PatchPathModel(
        8,
        6,
        patch_hidden=8,
        pair_hidden=4,
        token_width=6,
        dropout=0.0,
        center_context=center_context,
        center_context_hidden=16,
    )


def test_zero_initialised_center_context_preserves_base_function() -> None:
    torch.manual_seed(7)
    baseline = _model(center_context=False)
    expanded = _model(center_context=True)
    incompatible = expanded.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith("center_update.") for key in incompatible.missing_keys)

    baseline.eval()
    expanded.eval()
    batch = _toy_batch()
    with torch.no_grad():
        expected = baseline(batch)
        actual = expanded(batch)
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_parameter_breakdown_matches_total() -> None:
    model = _model(center_context=True)
    breakdown = model.parameter_breakdown()
    assert breakdown["center_context_update"] > 0
    assert breakdown["total"] == sum(
        parameter.numel() for parameter in model.parameters()
    )

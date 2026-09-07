from __future__ import annotations

import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as module


def _toy_batch(shell_width: int) -> Data:
    generator = torch.Generator().manual_seed(17)
    return Data(
        patch_cont=torch.randn(3, shell_width, generator=generator),
        patch_context=torch.zeros(3, 0),
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
        shell_width=7,
        center_context=center_context,
        center_context_hidden=16,
    )


def test_zero_initialised_center_context_preserves_base_function() -> None:
    torch.manual_seed(5)
    baseline = _model(center_context=False)
    expanded = _model(center_context=True)
    incompatible = expanded.load_state_dict(baseline.state_dict(), strict=False)
    assert not incompatible.unexpected_keys
    assert all(key.startswith("center_update.") for key in incompatible.missing_keys)

    baseline.eval()
    expanded.eval()
    batch = _toy_batch(shell_width=7)

    with torch.no_grad():
        expected = baseline(batch)
        actual = expanded(batch)

    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)


def test_center_context_is_invariant_to_pair_endpoint_order() -> None:
    torch.manual_seed(11)
    model = _model(center_context=True)
    # Make the residual active while keeping the test deterministic.
    torch.nn.init.normal_(model.center_update[-1].weight, std=0.02)
    model.eval()
    original = _toy_batch(shell_width=7)
    swapped = original.clone()
    swapped.pair_index = original.pair_index.flip(0)

    with torch.no_grad():
        left = model(original)
        right = model(swapped)

    torch.testing.assert_close(left, right, rtol=1.0e-6, atol=1.0e-6)

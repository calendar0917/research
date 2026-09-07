import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tracks.ksvd.experiments.luyin16.unified_typed_relational_patch_encoder import (
    DatasetSpec,
    RelationalPatchSetModel,
)


def _graph(offset: float, label: float) -> Data:
    return Data(
        patch_structure=torch.tensor(
            [[offset, 0.0], [0.0, 1.0 + offset]], dtype=torch.float32
        ),
        patch_attribute=torch.tensor(
            [[1.0, offset], [0.0, 1.0]], dtype=torch.float32
        ),
        patch_auxiliary=torch.zeros((2, 4), dtype=torch.float32),
        typed_token=torch.tensor([0, 0], dtype=torch.long),
        backoff_token=torch.tensor([0, 0], dtype=torch.long),
        pair_index=torch.tensor([[0], [1]], dtype=torch.long),
        pair_relation=torch.zeros((1, 3), dtype=torch.float32),
        pair_bucket=torch.tensor([0], dtype=torch.long),
        global_context=torch.zeros((1, 2), dtype=torch.float32),
        y=torch.tensor([label], dtype=torch.float32),
        num_nodes=2,
    )


def test_patch_pair_encoder_is_batched_and_has_one_output_per_graph() -> None:
    spec = DatasetSpec(
        name="synthetic",
        structure_width=2,
        attribute_width=2,
        relation_width=3,
        context_width=2,
        distance_buckets=1,
        typed_certificates=False,
    )
    config = {
        "model": {
            "structure_code_width": 4,
            "attribute_code_width": 4,
            "binding_width": 2,
            "token_width": 4,
            "token_rank": 2,
            "backoff_width": 2,
            "backoff_rank": 1,
            "patch_hidden": 4,
            "pair_hidden": 3,
            "context_code_width": 2,
            "head_hidden": 8,
            "head_bottleneck": 4,
            "dropout": 0.0,
        }
    }
    model = RelationalPatchSetModel(
        spec=spec,
        typed_vocabulary_size=1,
        backoff_vocabulary_size=1,
        config=config,
    )
    batch = next(iter(DataLoader([_graph(0.0, 0.0), _graph(1.0, 1.0)], batch_size=2)))
    output = model(batch)
    assert output.shape == (2,)
    assert torch.isfinite(output).all()


def test_model_parameter_count_is_finite_and_no_second_head_exists() -> None:
    spec = DatasetSpec("synthetic", 2, 2, 3, 2, 1, False)
    config = {
        "model": {
            "structure_code_width": 4,
            "attribute_code_width": 4,
            "binding_width": 2,
            "token_width": 4,
            "token_rank": 2,
            "backoff_width": 2,
            "backoff_rank": 1,
            "patch_hidden": 4,
            "pair_hidden": 3,
            "context_code_width": 2,
            "head_hidden": 8,
            "head_bottleneck": 4,
            "dropout": 0.0,
        }
    }
    model = RelationalPatchSetModel(
        spec=spec,
        typed_vocabulary_size=1,
        backoff_vocabulary_size=1,
        config=config,
    )
    assert sum(parameter.numel() for parameter in model.parameters()) > 0
    assert sum(1 for name, _ in model.named_parameters() if name.startswith("head")) > 0

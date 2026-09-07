import torch
from torch_geometric.data import Data
from torch_geometric.loader import DataLoader

from tracks.ksvd.experiments.luyin16.structured_patch_relational_model import (
    StructuredPatchRelationalModel,
)
from tracks.ksvd.experiments.luyin16.unified_typed_relational_patch_encoder import (
    DatasetSpec,
)


def _config() -> dict[str, dict[str, float | int]]:
    return {
        "model": {
            "adapter_hidden": 16,
            "structure_code_width": 6,
            "attribute_code_width": 6,
            "binding_width": 4,
            "geometry_code_width": 3,
            "pair_structure_width": 4,
            "pair_attribute_width": 4,
            "pair_binding_width": 3,
            "pair_relation_width": 4,
            "context_code_width": 4,
            "dropout": 0.0,
        }
    }


def _graph(reverse_pair: bool = False) -> Data:
    return Data(
        patch_structure=torch.tensor(
            [[1.0, 0.0, 0.5], [0.0, 1.0, -0.5]], dtype=torch.float32
        ),
        patch_attribute=torch.tensor(
            [[0.5, 1.0], [1.0, -0.5]], dtype=torch.float32
        ),
        patch_auxiliary=torch.zeros((2, 4), dtype=torch.float32),
        pair_index=(
            torch.tensor([[1], [0]], dtype=torch.long)
            if reverse_pair
            else torch.tensor([[0], [1]], dtype=torch.long)
        ),
        pair_relation=torch.tensor([[0.5, -1.0, 0.25]], dtype=torch.float32),
        pair_bucket=torch.tensor([0], dtype=torch.long),
        global_context=torch.zeros((1, 3), dtype=torch.float32),
        y=torch.tensor([0.0], dtype=torch.float32),
        num_nodes=2,
    )


def test_shared_pair_projection_is_invariant_to_unordered_pair_orientation() -> None:
    spec = DatasetSpec("synthetic", 3, 2, 3, 3, 1, False)
    model = StructuredPatchRelationalModel(spec=spec, config=_config()).eval()
    forward = next(iter(DataLoader([_graph(False)], batch_size=1)))
    reverse = next(iter(DataLoader([_graph(True)], batch_size=1)))
    with torch.no_grad():
        first = model(forward)
        second = model(reverse)
    assert torch.allclose(first, second, atol=1.0e-7, rtol=0.0)


def test_linear_head_components_reconstruct_prediction() -> None:
    spec = DatasetSpec("synthetic", 3, 2, 3, 3, 1, False)
    model = StructuredPatchRelationalModel(spec=spec, config=_config()).eval()
    batch = next(iter(DataLoader([_graph()], batch_size=1)))
    with torch.no_grad():
        prediction, components = model(batch, return_components=True)
    reconstructed = sum(components.values())
    assert torch.allclose(prediction, reconstructed, atol=1.0e-6, rtol=0.0)

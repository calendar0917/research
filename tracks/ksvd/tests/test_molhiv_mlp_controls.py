from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Batch, Data

from tracks.ksvd.experiments.luyin16.molhiv_mlp_controls import (
    GLOBAL_WIDTH,
    CenterFusionGlobalMLP,
    FeatureStandardizer,
    GlobalMLP,
)


def _item() -> Data:
    item = Data(
        node_topology=torch.randn(3, 168),
        node_attributes=torch.randn(3, 106),
        global_features=torch.randn(1, GLOBAL_WIDTH),
        y=torch.tensor([1.0]),
    )
    item.num_nodes = 3
    return item


def test_mlp_controls_forward_and_standardization() -> None:
    batch = Batch.from_data_list([_item(), _item()])
    global_model = GlobalMLP(GLOBAL_WIDTH, 8)
    combined_model = CenterFusionGlobalMLP(GLOBAL_WIDTH, 8)
    assert global_model(batch).shape == (2,)
    assert combined_model(batch).shape == (2,)
    assert torch.isfinite(global_model(batch)).all()
    assert torch.isfinite(combined_model(batch)).all()

    values = np.asarray([[1.0, 3.0], [3.0, 3.0]], dtype=np.float32)
    scaler = FeatureStandardizer.fit(values)
    transformed = scaler.transform(values)
    assert np.allclose(transformed[:, 0].mean(), 0.0)
    assert np.allclose(transformed[:, 1], 0.0)

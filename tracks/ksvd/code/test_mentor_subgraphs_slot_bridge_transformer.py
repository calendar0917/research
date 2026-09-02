"""Self-tests for slot-level shared-node bridge construction and model."""
from __future__ import annotations

import numpy as np

from .overlap_cover import _make_cover, _make_patch
from .overlap_stitching import make_cover_example
from .run_mentor_subgraphs_slot_bridge_transformer import (
    build_model,
    build_samples,
    exact_bridge_weights,
    make_collate,
    slot_features,
)


def _path(n: int) -> np.ndarray:
    adjacency = np.zeros((n, n), dtype=np.int8)
    for node in range(n - 1):
        adjacency[node, node + 1] = adjacency[node + 1, node] = 1
    return adjacency


def _example() -> object:
    adjacency = _path(8)
    cover = _make_cover(
        "test",
        [
            _make_patch(adjacency, (0, 1, 2, 3), 1),
            _make_patch(adjacency, (2, 3, 4, 5), 3),
            _make_patch(adjacency, (4, 5, 6, 7), 5),
        ],
    )
    return make_cover_example(9, "lt5", 2, adjacency, cover)


def main() -> int:
    example = _example()
    features = slot_features(example, 4)
    assert features.shape == (3, 4, 6)
    context, bridge = exact_bridge_weights(example, 1, 4)
    assert context.tolist() == [0, 2]
    assert bridge.shape == (4, 8)
    assert np.isclose(bridge[0].sum(), 1.0)
    assert np.isclose(bridge[1].sum(), 1.0)
    assert np.isclose(bridge[2].sum(), 1.0)
    assert np.isclose(bridge[3].sum(), 1.0)

    true = build_samples([example], branch="TRUE_BRIDGE", patch_size=4)
    shuffled = build_samples([example], branch="SHUFFLED_BRIDGE", patch_size=4)
    none = build_samples([example], branch="NO_BRIDGE", patch_size=4)
    assert len(true) == len(shuffled) == len(none) == 3
    assert np.count_nonzero(none[1].bridge_weights) == 0
    assert np.allclose(
        np.sort(true[1].bridge_weights, axis=0),
        np.sort(shuffled[1].bridge_weights, axis=0),
    )

    import torch
    from torch import nn

    batch = make_collate(torch, 4)(true[:2])
    model = build_model(
        nn,
        torch,
        patch_size=4,
        feature_dim=6,
        hidden=16,
        heads=4,
        local_layers=1,
        global_layers=1,
        dropout=0.0,
    )
    output = model(batch["context"], batch["valid"], batch["bridge"])
    assert output.shape == batch["targets"].shape == (2, 6)
    assert torch.isfinite(output).all()
    print("slot bridge transformer self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

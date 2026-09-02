"""Small self-tests for the GNN-free patch Transformer pilot."""
from __future__ import annotations

import numpy as np

from .overlap_cover import _make_cover, _make_patch
from .overlap_stitching import make_cover_example
from .run_mentor_subgraphs_patch_transformer import (
    _context_permutation,
    build_model,
    build_samples,
    exact_pair_relations,
    invariant_tokens,
    make_collate,
)


def _path(n_nodes: int) -> np.ndarray:
    adjacency = np.zeros((n_nodes, n_nodes), dtype=np.int8)
    for node in range(n_nodes - 1):
        adjacency[node, node + 1] = 1
        adjacency[node + 1, node] = 1
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
    return make_cover_example(7, "lt5", 2, adjacency, cover)


def _test_relation_tensor() -> None:
    example = _example()
    relation = exact_pair_relations(example, patch_size=4)
    assert relation.shape == (3, 3, 18)
    assert np.isclose(relation[0, 1, -2], 0.5)
    slot_map = relation[0, 1, :16].reshape(4, 4)
    assert slot_map[2, 0] == 1.0
    assert slot_map[3, 1] == 1.0
    assert int(slot_map.sum()) == 2


def _test_shuffle_preserves_target() -> None:
    for target in range(3):
        permutation = _context_permutation(3, target, graph_index=7)
        assert permutation[target] == target
        assert sorted(permutation.tolist()) == [0, 1, 2]


def _test_samples() -> None:
    example = _example()
    true_samples = build_samples([example], branch="TRUE_RELATION", patch_size=4)
    shuffled = build_samples([example], branch="SHUFFLED_RELATION", patch_size=4)
    no_relation = build_samples([example], branch="NO_RELATION", patch_size=4)
    relation_only = build_samples([example], branch="RELATION_ONLY", patch_size=4)
    assert len(true_samples) == len(shuffled) == len(no_relation) == len(relation_only) == 3
    assert np.allclose(true_samples[1].target, invariant_tokens(example)[1])
    assert np.count_nonzero(no_relation[0].relations) == 0
    assert np.allclose(true_samples[0].relations, shuffled[0].relations)
    assert np.count_nonzero(relation_only[0].tokens) == 0
    assert np.count_nonzero(relation_only[0].relations) > 0
    assert np.allclose(relation_only[1].target, invariant_tokens(example)[1])
    assert np.count_nonzero(relation_only[1].target) > 0


def _test_forward() -> None:
    import torch
    from torch import nn

    samples = build_samples([_example()], branch="TRUE_RELATION", patch_size=4)
    batch = make_collate(torch)(samples[:2])
    model = build_model(
        nn,
        torch,
        token_dim=samples[0].tokens.shape[1],
        relation_dim=samples[0].relations.shape[2],
        hidden=16,
        heads=4,
        layers=1,
        dropout=0.0,
    )
    output = model(
        batch["tokens"],
        batch["relations"],
        batch["valid"],
        batch["target_indices"],
    )
    assert output.shape == batch["targets"].shape
    assert torch.isfinite(output).all()


def main() -> int:
    _test_relation_tensor()
    _test_shuffle_preserves_target()
    _test_samples()
    _test_forward()
    print("patch transformer self-tests passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

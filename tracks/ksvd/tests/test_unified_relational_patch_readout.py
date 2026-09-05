import numpy as np
import torch

from tracks.ksvd.experiments.luyin16.unified_relational_patch_readout import (
    FeatureLayout,
    RawGraph,
    UnifiedMLP,
    _fit_transforms,
    _graph_feature,
    _pair_sketch,
    _relation_codes,
    _signed_sketch,
)


def _zinc_relation_row(distance: int, overlap: float, adjacent: int) -> np.ndarray:
    row = np.zeros(23, dtype=np.float32)
    row[distance - 1] = 1.0
    row[5] = np.log1p(float(distance))
    row[6] = overlap
    row[19 + adjacent] = 1.0
    return row


def _synthetic_zinc_graph() -> RawGraph:
    descriptors = np.zeros((3, 840), dtype=np.float32)
    descriptors[0, 0] = 1.0
    descriptors[1, 1] = 1.0
    descriptors[2, 2] = 1.0
    relation = np.stack(
        [_zinc_relation_row(1, 0.25, 0), _zinc_relation_row(2, 0.5, 1)],
        axis=0,
    )
    return RawGraph(
        descriptors=descriptors,
        token_hash=np.asarray([11, 22, 33], dtype=np.uint64),
        parent_hash=np.asarray([1, 1, 2], dtype=np.uint64),
        pair_index=np.asarray([[0, 0], [1, 2]], dtype=np.int64),
        pair_relation=relation,
        pair_bucket=np.asarray([0, 1], dtype=np.int64),
        context=np.zeros(62, dtype=np.float32),
        y=0.0,
    )


def test_signed_sketch_is_order_invariant() -> None:
    keys = np.asarray([5, 7, 11, 13], dtype=np.uint64)
    np.testing.assert_array_equal(
        _signed_sketch(keys, width=32, salt=17),
        _signed_sketch(keys[::-1], width=32, salt=17),
    )


def test_pair_sketch_is_unordered_pair_invariant() -> None:
    graph = _synthetic_zinc_graph()
    reversed_graph = RawGraph(
        descriptors=graph.descriptors,
        token_hash=graph.token_hash,
        parent_hash=graph.parent_hash,
        pair_index=graph.pair_index[:, ::-1],
        pair_relation=graph.pair_relation[::-1],
        pair_bucket=graph.pair_bucket[::-1],
        context=graph.context,
        y=graph.y,
    )
    codes = _relation_codes(
        graph.pair_relation,
        graph.pair_bucket,
        bond_width=4,
        distance_buckets=5,
    )
    reversed_codes = _relation_codes(
        reversed_graph.pair_relation,
        reversed_graph.pair_bucket,
        bond_width=4,
        distance_buckets=5,
    )
    np.testing.assert_allclose(
        _pair_sketch(graph, codes, width=64),
        _pair_sketch(reversed_graph, reversed_codes, width=64),
    )


def test_relation_code_uses_fixed_distance_columns() -> None:
    relation = np.stack([_zinc_relation_row(1, 0.75, 2)], axis=0)
    codes = _relation_codes(
        relation,
        np.asarray([0], dtype=np.int64),
        bond_width=4,
        distance_buckets=5,
    )
    # The first overlap field is 0.75 -> bin 3, and the adjacent bond is 2.
    np.testing.assert_array_equal(codes, np.asarray([3 * 16 + 2], dtype=np.int64))


def test_synthetic_graph_feature_matches_declared_layout() -> None:
    graph = _synthetic_zinc_graph()
    config = {
        "representation": {
            "max_patch_samples": 16,
            "max_relation_samples": 16,
        },
        "model": {
            "prototype_width": 4,
            "prototype_atoms": 4,
            "prototype_sparsity": 2,
            "relation_code_width": 3,
            "context_width": 4,
        },
    }
    fit_graphs = []
    for index in range(4):
        fit_graphs.append(
            RawGraph(
                descriptors=graph.descriptors,
                token_hash=graph.token_hash,
                parent_hash=graph.parent_hash,
                pair_index=graph.pair_index,
                pair_relation=graph.pair_relation,
                pair_bucket=graph.pair_bucket,
                context=np.full(62, float(index), dtype=np.float32),
                y=graph.y,
            )
        )
    transforms, _ = _fit_transforms(fit_graphs, dataset="zinc", config=config, seed=0)
    layout = FeatureLayout(
        structure_width=4,
        attribute_width=4,
        prototype_width=4,
        relation_width=3,
        distance_buckets=5,
        unary_sketch_width=8,
        parent_sketch_width=4,
        pair_sketch_width=8,
        context_width=4,
    )
    feature = _graph_feature(
        graph,
        transforms,
        dataset="zinc",
        layout=layout,
        bond_width=4,
    )
    assert feature.shape == (layout.width,)
    assert np.isfinite(feature).all()


def test_unified_mlp_output_shape() -> None:
    model = UnifiedMLP(input_width=13, hidden=8, bottleneck=4, dropout=0.0)
    output = model(torch.zeros((5, 13), dtype=torch.float32))
    assert output.shape == (5,)

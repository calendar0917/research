from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import tu_patch_path_pooling as tu


# ---------------------------------------------------------------------------
# strict splits
# ---------------------------------------------------------------------------


def test_build_folds_is_strict_and_deterministic() -> None:
    rng = np.random.default_rng(0)
    y = np.asarray([0] * 30 + [1] * 30)
    folds = tu.build_folds(y, 3, seed=123)
    checks = tu.check_folds(folds, y.shape[0])
    assert checks["all_disjoint"] is True
    assert checks["each_graph_tested_once"] is True
    again = tu.build_folds(y, 3, seed=123)
    assert tu.folds_fingerprint(folds) == tu.folds_fingerprint(again)
    other = tu.build_folds(y, 3, seed=124)
    assert tu.folds_fingerprint(folds) != tu.folds_fingerprint(other)
    assert rng is not None


def test_check_folds_detects_overlap() -> None:
    folds = [
        {
            "fold": np.int64(0),
            "train": np.asarray([0, 1], dtype=np.int64),
            "valid": np.asarray([1], dtype=np.int64),
            "test": np.asarray([2], dtype=np.int64),
        }
    ]
    checks = tu.check_folds(folds, 3)
    assert checks["all_disjoint"] is False


# ---------------------------------------------------------------------------
# patch extraction
# ---------------------------------------------------------------------------


def _triangle() -> Data:
    # 3 nodes, full triangle, 3 discrete node types, 2 edge types
    x = torch.eye(3)
    edge_index = torch.tensor([[0, 1, 2, 0, 1, 2], [1, 2, 0, 0, 0, 0]], dtype=torch.long)
    edge_attr = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [1.0, 0.0], [1.0, 0.0], [0.0, 1.0]]
    )
    return Data(x=x, edge_index=edge_index, edge_attr=edge_attr, y=torch.tensor([1]))


def test_extract_graph_shapes() -> None:
    graph = _triangle()
    cache: dict = {}
    record = tu.extract_graph(graph, 3, 2, cache)
    assert len(record.patches) == 3
    assert record.pair_index.shape == (2, 3)  # all unordered pairs of 3 centres
    assert record.pair_relation.shape == (3, tu.relation_width(2))
    assert record.global_context.shape == (tu.GLOBAL_WIDTH,)
    assert record.patches[0].shell_descriptor.shape == (tu.shell_width(3, 2),)
    assert np.isfinite(record.pair_relation).all()


def test_certificate_is_permutation_invariant() -> None:
    graph = _triangle()
    record = tu.extract_graph(graph, 3, 2, {})
    certificates = sorted(p.typed_certificate for p in record.patches)

    generator = torch.Generator().manual_seed(7)
    n = int(graph.num_nodes)
    permutation = torch.randperm(n, generator=generator)
    inverse = torch.empty_like(permutation)
    inverse[permutation] = torch.arange(n)

    x = graph.x[permutation]
    edge_index = inverse[graph.edge_index]
    relabeled = Data(
        x=x, edge_index=edge_index, edge_attr=graph.edge_attr, y=torch.tensor([1])
    )
    record2 = tu.extract_graph(relabeled, 3, 2, {})
    assert certificates == sorted(p.typed_certificate for p in record2.patches)
    assert record.pair_bucket.tolist() == record2.pair_bucket.tolist()


def test_structure_only_uses_degree_types() -> None:
    # no x: IMDB-style graph; node type is the clipped degree
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    graph = Data(edge_index=edge_index, y=torch.tensor([0]), num_nodes=2)
    types = tu.node_types_of(graph, tu.DEGREE_BINS)
    assert types.tolist() == [1, 1]


# ---------------------------------------------------------------------------
# model / local-token channel
# ---------------------------------------------------------------------------


def _batch(representation: str, n_classes: int = 2):
    shell = tu.shell_width(3, 2)
    relation = tu.relation_width(2)
    data = Data(
        patch_cont=torch.randn(4, shell),
        typed_token=torch.tensor([1, 2, 3, 4], dtype=torch.long),
        parent_token=torch.tensor([1, 1, 2, 2], dtype=torch.long),
        pair_index=torch.tensor([[0, 1, 2], [1, 2, 3]], dtype=torch.long),
        pair_relation=torch.randn(3, relation),
        pair_bucket=torch.tensor([0, 1, 0], dtype=torch.long),
        global_context=torch.randn(1, tu.GLOBAL_WIDTH),
        y=torch.tensor([0], dtype=torch.long),
        num_nodes=4,
    )
    loader = tu.make_loader([data], 1, False, 0)
    return next(iter(loader))


def _model(representation: str) -> tu.TuPatchPathModel:
    return tu.TuPatchPathModel(
        8,
        6,
        shell_width=tu.shell_width(3, 2),
        relation_width=tu.relation_width(2),
        n_classes=2,
        patch_representation=representation,
    )


def test_null_has_no_generator_and_zero_token() -> None:
    model = _model("null")
    assert model.typed_embedding is None
    assert model.local_token_constant is None
    token = model._patch_token_value(_batch("null"))
    assert bool(torch.all(token == 0).item())


def test_constant_is_shared_and_trainable() -> None:
    model = _model("constant")
    assert model.typed_embedding is None
    assert model.local_token_constant is not None
    token = model._patch_token_value(_batch("constant"))
    assert bool(torch.all(token == token[0]).item())
    assert model.local_token_constant.numel() == tu.TOKEN_WIDTH


def test_parameter_drop_equals_lookup_size() -> None:
    typed = _model("typed_lookup")
    null = _model("null")
    constant = _model("constant")
    typed_total = typed.n_params()
    null_total = null.n_params()
    constant_total = constant.n_params()
    assert typed_total - null_total == typed.typed_embedding.weight.numel()
    assert constant_total - null_total == tu.TOKEN_WIDTH


def test_null_forward_equals_zeroed_typed_forward() -> None:
    torch.manual_seed(3)
    baseline = _model("typed_lookup")
    with torch.no_grad():
        baseline.typed_embedding.weight.zero_()
    null = _model("null")
    incompatible = null.load_state_dict(baseline.state_dict(), strict=False)
    assert all(key.startswith("typed_embedding.") for key in incompatible.unexpected_keys)
    assert not incompatible.missing_keys
    baseline.eval()
    null.eval()
    batch = _batch("null")
    with torch.no_grad():
        torch.testing.assert_close(null(batch), baseline(batch), rtol=0.0, atol=0.0)


def test_forward_shapes() -> None:
    for representation in tu.TRAINABLE_REPRESENTATIONS:
        model = _model(representation).eval()
        with torch.no_grad():
            logits = model(_batch(representation))
        assert logits.shape == (1, 2)
        assert bool(torch.isfinite(logits).all())


def test_unknown_representation_rejected() -> None:
    with pytest.raises(ValueError):
        _model("shared_structural")

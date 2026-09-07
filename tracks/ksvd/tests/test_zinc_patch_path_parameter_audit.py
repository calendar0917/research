from __future__ import annotations

import numpy as np
import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as module


def _small_model(**overrides) -> module.PatchPathModel:
    kwargs = dict(
        typed_vocabulary_size=10,
        parent_vocabulary_size=4,
        patch_hidden=8,
        pair_hidden=4,
        token_width=6,
        dropout=0.0,
        embedding_mode="hybrid",
        embedding_rank=2,
        hybrid_full_typed_tokens=4,
        hybrid_full_parent_tokens=4,
        shell_width=7,
        center_context=True,
        center_context_hidden=16,
    )
    kwargs.update(overrides)
    return module.PatchPathModel(**kwargs)


def _data(shell_width: int) -> Data:
    generator = torch.Generator().manual_seed(29)
    return Data(
        patch_cont=torch.randn(4, shell_width, generator=generator),
        patch_context=torch.zeros(4, 0),
        typed_token=torch.tensor([0, 1, 4, 9], dtype=torch.long),
        parent_token=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        pair_index=torch.tensor([[0, 0, 1, 2], [1, 2, 3, 3]], dtype=torch.long),
        pair_relation=torch.randn(4, module.RELATION_WIDTH, generator=generator),
        pair_bucket=torch.tensor([0, 1, 2, 3], dtype=torch.long),
        global_context=torch.randn(1, module.GLOBAL_WIDTH, generator=generator),
        batch=torch.zeros(4, dtype=torch.long),
        y=torch.tensor([0.0]),
        num_nodes=4,
    )


def test_parameter_audit_blocks_equal_global_trainable_total() -> None:
    torch.manual_seed(3)
    model = _small_model()
    audit = module.audit_parameters(model)
    expected_global = int(
        sum(p.numel() for p in model.parameters() if p.requires_grad)
    )
    assert audit["total_trainable"] == expected_global
    assert sum(audit["blocks"].values()) == expected_global
    for block in module.PARAMETER_AUDIT_BLOCKS:
        assert block[0] in audit["blocks"]
    # A zero-initialised residual still has trainable parameters.
    assert audit["blocks"]["center_context"] > 0
    assert audit["blocks"]["typed_token_embedding"] > 0
    assert audit["blocks"]["graph_head"] > 0


def test_parameter_audit_disabled_center_context_has_zero_block() -> None:
    model = _small_model(center_context=False)
    audit = module.audit_parameters(model)
    assert audit["blocks"]["center_context"] == 0
    assert audit["total_trainable"] == sum(
        p.numel() for p in model.parameters() if p.requires_grad
    )


def test_hybrid_embedding_full_rare_boundary_and_identity() -> None:
    torch.manual_seed(7)
    embedding = module._HybridEmbedding(
        vocabulary_size=10, output_width=6, rank=2, full_count=4
    )
    tokens = torch.tensor([0, 1, 3, 4, 5, 9], dtype=torch.long)
    values = embedding(tokens)
    assert values.shape == (6, 6)
    with torch.no_grad():
        rare_codes = embedding.rare.embedding(tokens[3:] - 4)
        rare_projected = embedding.rare.projection(rare_codes)
    torch.testing.assert_close(values[3:], rare_projected, rtol=0.0, atol=0.0)
    # Every rare token keeps its own independent row (exact identity).
    assert not torch.allclose(values[3], values[4])
    assert not torch.allclose(values[4], values[5])
    # Full and rare rows are distinct parameterizations.
    assert embedding.full.weight.shape == (4, 6)
    assert embedding.rare.embedding.weight.shape == (6, 2)


def test_hybrid_embedding_full_count_equals_vocabulary_uses_only_full() -> None:
    embedding = module._HybridEmbedding(
        vocabulary_size=5, output_width=6, rank=2, full_count=5
    )
    assert embedding.rare is None
    values = embedding(torch.tensor([0, 4], dtype=torch.long))
    assert values.shape == (2, 6)


def test_fit_vocabulary_orders_by_train_frequency_descending() -> None:
    def record(*keys: bytes) -> module.GraphRecord:
        return module.GraphRecord(
            patches=tuple(
                module.PatchRecord(
                    typed_certificate=key,
                    parent_certificate=b"p",
                    nodes=frozenset({index}),
                    boundary=frozenset(),
                    shell_descriptor=np.zeros(7, dtype=np.float32),
                )
                for index, key in enumerate(keys)
            ),
            pair_index=np.zeros((2, 1), dtype=np.int64),
            pair_relation=np.zeros((1, module.RELATION_WIDTH), dtype=np.float32),
            pair_bucket=np.zeros(1, dtype=np.int64),
            global_context=np.zeros(module.GLOBAL_WIDTH, dtype=np.float32),
            y=0.0,
        )

    records = [record(b"a", b"b"), record(b"b", b"a"), record(b"a")]
    vocabulary = module._fit_vocabulary(
        records, "typed_certificate", maximum=10, minimum_frequency=1
    )
    assert list(vocabulary.keys()) == [b"a", b"b"]
    # ID zero is reserved for OOV; known tokens start at one, highest
    # frequency first.
    assert vocabulary[b"a"] < vocabulary[b"b"]
    counts = {"a": 3, "b": 2}
    ordered_counts = [counts[key.decode()] for key in vocabulary]
    assert ordered_counts == sorted(ordered_counts, reverse=True)


def test_compact_forward_backward_shapes_and_finite() -> None:
    torch.manual_seed(13)
    model = _small_model()
    data = _data(7)
    prediction = model(data)
    assert prediction.shape == (1,)
    loss = torch.nn.functional.l1_loss(prediction, data.y)
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [
        parameter.grad
        for parameter in model.parameters()
        if parameter.grad is not None
    ]
    assert gradients
    for gradient in gradients:
        assert torch.isfinite(gradient).all()

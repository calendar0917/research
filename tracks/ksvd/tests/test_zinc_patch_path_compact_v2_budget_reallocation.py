"""Tests for Compact-Hybrid-v2 (fixed-budget capacity reallocation).

v2 moves capacity from the graph prediction head to the rare exact-token
representation:

* ``embedding_rank`` 2 -> 4 (typed token embedding only; parent vocabulary
  stays fully covered by the full table)
* graph head hidden widths 96/48 -> 64/32 (configurable ``graph_head_hidden_0``
  / ``graph_head_hidden_1``, defaulting to the historical widths so old
  configs reproduce Compact-v1 exactly)

Everything else (representation, training protocol, seeds, budget) is
intentionally unchanged.
"""

from __future__ import annotations

import yaml
import torch

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as module

# Official ZINC vocabulary sizes (train-only fit and train+valid refit fit)
# recorded by the 277053-parameter baseline / Compact-v1 runs.
SELECTION_TYPED_VOCAB = 6785
REFIT_TYPED_VOCAB = 7051
PARENT_VOCAB = 32

# Total trainable parameters of the two phases for Compact-v1 (known, from
# the v1 run) and Compact-v2 (derived analytically from the same blocks:
# only the typed-token rare table and the graph head change).
COMPACT_V1_TOTALS = {
    "validation-selection": 98_579,
    "train-valid-refit": 99_111,
}
COMPACT_V2_TOTALS = {
    "validation-selection": 98_549,
    "train-valid-refit": 99_613,
}


def _compact_v1_config() -> dict:
    path = (
        module.REPO_ROOT
        / "tracks/ksvd/configs/luyin16/zinc_hierarchical_patch_relation_context_compact_hybrid_v1.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _compact_v2_config() -> dict:
    path = (
        module.REPO_ROOT
        / "tracks/ksvd/configs/luyin16/zinc_hierarchical_patch_relation_context_compact_hybrid_v2_budget_reallocation.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _model_from_config(config: dict, typed_vocab: int) -> module.PatchPathModel:
    model = config["model"]
    return module.PatchPathModel(
        typed_vocab,
        PARENT_VOCAB,
        patch_hidden=int(model["patch_hidden"]),
        pair_hidden=int(model["pair_hidden"]),
        token_width=int(model["token_width"]),
        dropout=float(model["dropout"]),
        embedding_mode=str(model["embedding_mode"]),
        embedding_rank=int(model["embedding_rank"]),
        hybrid_full_typed_tokens=int(model["hybrid_full_typed_tokens"]),
        hybrid_full_parent_tokens=int(model["hybrid_full_parent_tokens"]),
        center_context=bool(model["center_context"]),
        center_context_hidden=int(model["center_context_hidden"]),
        graph_head_hidden_0=(
            None
            if model.get("graph_head_hidden_0") is None
            else int(model["graph_head_hidden_0"])
        ),
        graph_head_hidden_1=(
            None
            if model.get("graph_head_hidden_1") is None
            else int(model["graph_head_hidden_1"])
        ),
        shell_width=module._shell_width_for_radius(module.PATCH_RADIUS),
        context_width=0,
    )


def _small_model(*, head_0: int | None = None, head_1: int | None = None) -> module.PatchPathModel:
    return module.PatchPathModel(
        10,
        4,
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
        graph_head_hidden_0=head_0,
        graph_head_hidden_1=head_1,
    )


def _head_shapes(model: module.PatchPathModel) -> tuple[tuple[int, ...], ...]:
    layers = tuple(model.head)
    return (
        tuple(layers[0].weight.shape),  # Linear in
        tuple(layers[1].weight.shape),  # LayerNorm
        tuple(layers[4].weight.shape),  # Linear hidden
        tuple(layers[6].weight.shape),  # Linear out
    )


def test_graph_head_default_widths_preserve_baseline() -> None:
    model = _small_model()
    # Historical default: max(2*patch_hidden, 96) and patch_hidden (8 here).
    assert model.head_hidden_0 == 96
    assert model.head_hidden_1 == 8
    assert _head_shapes(model) == (
        (96, model.readout_width + 32),
        (96,),
        (8, 96),
        (1, 8),
    )


def test_graph_head_configurable_widths() -> None:
    model = _small_model(head_0=64, head_1=32)
    assert model.head_hidden_0 == 64
    assert model.head_hidden_1 == 32
    assert _head_shapes(model) == (
        (64, model.readout_width + 32),
        (64,),
        (32, 64),
        (1, 32),
    )


def test_compact_v1_config_preserves_baseline_parameter_count() -> None:
    config = _compact_v1_config()
    assert int(config["model"]["embedding_rank"]) == 2
    for phase, typed_vocab in (
        ("validation-selection", SELECTION_TYPED_VOCAB),
        ("train-valid-refit", REFIT_TYPED_VOCAB),
    ):
        model = _model_from_config(config, typed_vocab)
        audit = module.audit_parameters(model)
        assert audit["total_trainable"] == COMPACT_V1_TOTALS[phase], phase
        assert audit["blocks"]["graph_head"] == 33_217, phase
        assert audit["blocks"]["typed_token_embedding"] == (
            24_354 if phase == "validation-selection" else 24_886
        ), phase
        assert audit["dimensions"]["graph_head_hidden_dims"] == [96, 48]


def test_compact_v2_parameter_budget() -> None:
    config = _compact_v2_config()
    model_config = config["model"]
    assert int(model_config["embedding_rank"]) == 4
    assert int(model_config["graph_head_hidden_0"]) == 64
    assert int(model_config["graph_head_hidden_1"]) == 32
    budget = int(model_config["expected_max_trainable_params"])
    for phase, typed_vocab in (
        ("validation-selection", SELECTION_TYPED_VOCAB),
        ("train-valid-refit", REFIT_TYPED_VOCAB),
    ):
        model = _model_from_config(config, typed_vocab)
        audit = module.audit_parameters(model)
        total = audit["total_trainable"]
        assert total == sum(
            p.numel() for p in model.parameters() if p.requires_grad
        ), phase
        assert total <= budget, (phase, total, budget)
        assert total == COMPACT_V2_TOTALS[phase], phase
        assert audit["blocks"]["graph_head"] == 21_121, phase
        assert audit["blocks"]["typed_token_embedding"] == (
            36_420 if phase == "validation-selection" else 37_484
        ), phase
        assert audit["dimensions"]["graph_head_hidden_dims"] == [64, 32]
        # The reallocation is a capacity transfer under one fixed budget:
        # selection-phase v2 (98549) is even slightly below v1 (98579); the
        # refit-phase v2 (99613) is above v1 (99111) because the rank-4 rare
        # table grows faster than the shrunken head saves with the larger
        # train+valid vocab — both remain strictly <= 100000.
        assert total <= budget, (phase, total, budget)
        assert audit["dimensions"]["patch_descriptor_dim"] == 146
        assert audit["dimensions"]["relation_descriptor_dim"] == 23
        assert audit["dimensions"]["distance_buckets"] == 5
        assert audit["dimensions"]["typed_token_width"] == 16
        assert audit["dimensions"]["patch_hidden_dim"] == 48
        assert audit["dimensions"]["pair_hidden_dim"] == 16
        assert audit["dimensions"]["center_context_hidden_dim"] == 60
        assert audit["dimensions"]["graph_readout_dim"] == 262


def test_compact_v2_rank4_exact_identity() -> None:
    torch.manual_seed(7)
    embedding = module._HybridEmbedding(
        vocabulary_size=10, output_width=6, rank=4, full_count=4
    )
    tokens = torch.tensor([0, 1, 3, 4, 5, 9], dtype=torch.long)
    values = embedding(tokens)
    assert values.shape == (6, 6)
    with torch.no_grad():
        rare_codes = embedding.rare.embedding(tokens[3:] - 4)
        rare_projected = embedding.rare.projection(rare_codes)
    torch.testing.assert_close(values[3:], rare_projected, rtol=0.0, atol=0.0)
    # Every rare token keeps its own independent rank-4 row (exact identity).
    assert embedding.rare.embedding.weight.shape == (6, 4)
    assert embedding.rare.projection.weight.shape == (6, 4)
    assert not torch.allclose(values[3], values[4])
    assert not torch.allclose(values[4], values[5])
    # Full rows remain full-width; the boundary is unchanged.
    assert embedding.full.weight.shape == (4, 6)


def test_compact_v2_forward_backward_finite() -> None:
    torch.manual_seed(13)
    config = _compact_v2_config()
    model = _model_from_config(config, SELECTION_TYPED_VOCAB)
    assert model.head_hidden_0 == 64
    assert model.head_hidden_1 == 32
    generator = torch.Generator().manual_seed(101)
    n = 6
    data = module.Data(
        patch_cont=torch.randn(n, module._shell_width_for_radius(2), generator=generator),
        patch_context=torch.zeros(n, 0),
        typed_token=torch.tensor(
            [0, 1, 767, 768, 769, SELECTION_TYPED_VOCAB - 1], dtype=torch.long
        ),
        parent_token=torch.tensor([0, 1, 2, 3, 31, 31], dtype=torch.long),
        pair_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long),
        pair_relation=torch.randn(5, module.RELATION_WIDTH, generator=generator),
        pair_bucket=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        global_context=torch.randn(1, module.GLOBAL_WIDTH, generator=generator),
        batch=torch.zeros(n, dtype=torch.long),
        y=torch.tensor([0.0]),
        num_nodes=n,
    )
    model.train()
    prediction = model(data)
    assert prediction.shape == (1,)
    assert torch.isfinite(prediction).all()
    loss = torch.nn.functional.l1_loss(prediction, data.y)
    loss.backward()
    assert torch.isfinite(loss)
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients
    for gradient in gradients:
        assert torch.isfinite(gradient).all()

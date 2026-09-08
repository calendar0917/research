"""Correctness tests for Compact-v3: context-conditioned patch representation.

Covered properties (see ``notes/compact_v3_*.md`` for the design):

1. ``structural_context_mode=none`` degenerates exactly to compact-v2
   (same parameter counts, same forward/backward behaviour).
2. NO_RING patches get a bit-exact zero conditioning delta.
3. Per-node structural context keys are permutation invariant under
   node-ID relabelling.
4. Canonical typed cycle signatures are direction invariant.
5. The context vocabulary is train-only; unseen keys map to UNK_CONTEXT.
6. The conditioning module has normal backward gradients.
"""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch
import yaml

from tracks.ksvd.code.graph import from_edges
from tracks.ksvd.experiments.luyin16 import structural_context as sc
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as module

SELECTION_TYPED_VOCAB = 6785
REFIT_TYPED_VOCAB = 7051
PARENT_VOCAB = 32
COMPACT_V2_TOTALS = {
    "validation-selection": 98_549,
    "train-valid-refit": 99_613,
}
# Measured boundary for the v3 parameter budget: the typed context vocabulary
# at max_cycle_len=10 on the official train split is 9,700 keys (plus the
# NO_RING/UNK reserved tokens) and the refit (train+valid) vocabulary adds a
# small number of val-only keys.  The real audit in the run is authoritative;
# these constants only drive the unit-level formulas below.
V3_BUDGET = 110_000


def _v2_config() -> dict:
    path = (
        module.REPO_ROOT
        / "tracks/ksvd/configs/luyin16/zinc_hierarchical_patch_relation_context_compact_hybrid_v2_budget_reallocation.yaml"
    )
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def _model_from_config(config: dict, typed_vocab: int, **model_overrides: object) -> module.PatchPathModel:
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
        graph_head_hidden_0=int(model["graph_head_hidden_0"]),
        graph_head_hidden_1=int(model["graph_head_hidden_1"]),
        shell_width=module._shell_width_for_radius(module.PATCH_RADIUS),
        context_width=0,
        structural_context_mode=str(model_overrides.pop("structural_context_mode", "none")),
        structural_context_fusion=str(model_overrides.pop("structural_context_fusion", "condition")),
        structural_context_dim=int(model_overrides.pop("structural_context_dim", 4)),
        structural_context_embedding_rank=int(
            model_overrides.pop("structural_context_embedding_rank", 1)
        ),
        structural_context_condition_rank=int(
            model_overrides.pop("structural_context_condition_rank", 8)
        ),
        structural_context_vocabulary_size=int(
            model_overrides.pop("structural_context_vocabulary_size", 2)
        ),
        **model_overrides,
    )


def _data_for_v3(
    *,
    n: int = 6,
    typed_vocab: int,
    tokens: torch.Tensor,
    coarse: torch.Tensor | None = None,
    shell_width: int | None = None,
) -> module.Data:
    generator = torch.Generator().manual_seed(101)
    if coarse is None:
        coarse = torch.zeros((n, 4), dtype=torch.float32)
    if shell_width is None:
        shell_width = module._shell_width_for_radius(module.PATCH_RADIUS)
    return module.Data(
        patch_cont=torch.randn(n, shell_width, generator=generator),
        patch_context=torch.zeros(n, 0),
        typed_token=torch.tensor(
            [0, 1, 767, 768, 769, typed_vocab - 1], dtype=torch.long
        ),
        parent_token=torch.tensor([0, 1, 2, 3, 31, 31], dtype=torch.long),
        structural_token=tokens,
        structural_coarse=coarse,
        pair_index=torch.tensor([[0, 1, 2, 3, 4], [1, 2, 3, 4, 5]], dtype=torch.long),
        pair_relation=torch.randn(5, module.RELATION_WIDTH, generator=generator),
        pair_bucket=torch.tensor([0, 1, 2, 3, 4], dtype=torch.long),
        global_context=torch.randn(1, module.GLOBAL_WIDTH, generator=generator),
        batch=torch.zeros(n, dtype=torch.long),
        y=torch.tensor([0.0]),
        num_nodes=n,
    )


def test_context_mode_none_reproduces_compact_v2_parameters() -> None:
    """Test 1: mode=none is exactly compact-v2 (both phases)."""
    config = _v2_config()
    for phase, typed_vocab in (
        ("validation-selection", SELECTION_TYPED_VOCAB),
        ("train-valid-refit", REFIT_TYPED_VOCAB),
    ):
        model = _model_from_config(config, typed_vocab)
        audit = module.audit_parameters(model)
        assert audit["total_trainable"] == COMPACT_V2_TOTALS[phase], phase
        assert audit["blocks"]["structural_context_embedding"] == 0
        assert audit["blocks"]["context_patch_projection"] == 0
        assert audit["blocks"]["context_condition_projection"] == 0
        assert audit["blocks"]["context_delta_output"] == 0
        assert model.structural_context_mode == "none"
        # The patch-state pipeline is untouched: input width equals v2.
        assert model.patch_encoder.layers[0].in_features == 146 + 16 + 8


def test_context_mode_none_forward_backward_finite() -> None:
    """Test 1b: mode=none forward/backward remains finite (v2 path)."""
    config = _v2_config()
    model = _model_from_config(config, SELECTION_TYPED_VOCAB)
    model.train()
    data = _data_for_v3(
        typed_vocab=SELECTION_TYPED_VOCAB,
        tokens=torch.tensor([0, 1, 767, 768, 769, SELECTION_TYPED_VOCAB - 1], dtype=torch.long),
    )
    prediction = model(data)
    assert prediction.shape == (1,)
    loss = torch.nn.functional.l1_loss(prediction, data.y)
    loss.backward()
    assert torch.isfinite(loss)
    for parameter in model.parameters():
        if parameter.grad is not None:
            assert torch.isfinite(parameter.grad).all()


def test_no_ring_delta_is_bit_exact_zero() -> None:
    """Test 2: NO_RING -> conditioned patch == original patch (bit-exact)."""
    torch.manual_seed(3)
    model = module.PatchPathModel(
        20,
        8,
        patch_hidden=8,
        pair_hidden=4,
        token_width=6,
        dropout=0.0,
        embedding_mode="hybrid",
        embedding_rank=2,
        hybrid_full_typed_tokens=4,
        hybrid_full_parent_tokens=4,
        shell_width=7,
        structural_context_mode="typed_ring",
        structural_context_fusion="condition",
        structural_context_dim=4,
        structural_context_embedding_rank=1,
        structural_context_condition_rank=4,
        structural_context_vocabulary_size=6,
    )
    e_patch = torch.randn(5, 6, generator=torch.Generator().manual_seed(7))
    e_ctx = torch.randn(5, 4, generator=torch.Generator().manual_seed(8))
    # W_out is zero-initialised by design; make it nontrivial so the test
    # actually exercises a nonzero delta.
    with torch.no_grad():
        model.context_delta_output.weight.normal_()
        model.context_delta_output.bias.normal_()
    # Even with random weights + random context, a zero mask means delta == 0.
    masked = model._condition_patch(e_patch, e_ctx, torch.zeros(5, 1))
    torch.testing.assert_close(masked, e_patch, rtol=0.0, atol=0.0)
    assert torch.equal(masked, e_patch)
    # With mask == 1 the module does change the representation.
    active = model._condition_patch(e_patch, e_ctx, torch.ones(5, 1))
    assert not torch.allclose(active, e_patch)


def test_no_ring_token_stays_zero_across_modes() -> None:
    """Test 2b: NO_RING token/coarse row maps to a zero e_ctx in forward."""
    torch.manual_seed(3)
    model = module.PatchPathModel(
        20,
        8,
        patch_hidden=8,
        pair_hidden=4,
        token_width=6,
        dropout=0.0,
        embedding_mode="hybrid",
        embedding_rank=2,
        hybrid_full_typed_tokens=4,
        hybrid_full_parent_tokens=4,
        shell_width=7,
        structural_context_mode="typed_ring",
        structural_context_fusion="condition",
        structural_context_dim=4,
        structural_context_embedding_rank=1,
        structural_context_condition_rank=4,
        structural_context_vocabulary_size=6,
    )
    data = module.Data(
        patch_cont=torch.zeros(3, 7),
        patch_context=torch.zeros(3, 0),
        typed_token=torch.tensor([0, 1, 2], dtype=torch.long),
        parent_token=torch.tensor([0, 1, 2], dtype=torch.long),
        structural_token=torch.tensor([0, 1, 2], dtype=torch.long),
        structural_coarse=torch.zeros(3, 4),
        pair_index=torch.tensor([[0, 1], [1, 2]], dtype=torch.long),
        pair_relation=torch.zeros(2, module.RELATION_WIDTH),
        pair_bucket=torch.tensor([0, 1], dtype=torch.long),
        global_context=torch.zeros(1, module.GLOBAL_WIDTH),
        batch=torch.zeros(3, dtype=torch.long),
        y=torch.tensor([0.0]),
        num_nodes=3,
    )
    e_ctx = model._structural_context_embedding_value(data)
    assert torch.equal(e_ctx[0], torch.zeros(4))
    assert not torch.equal(e_ctx[1], torch.zeros(4))  # UNK keeps its own row
    assert not torch.equal(e_ctx[2], torch.zeros(4))
    mask = model._structural_no_ring_mask(data)
    assert torch.equal(mask, torch.tensor([[0.0], [1.0], [1.0]]))


def _relabel_data(data, permutation: np.ndarray, node_types: np.ndarray, edge_types: dict):
    """Relabel node IDs of a graph by permutation (bijective map)."""
    n = int(data.num_nodes)
    mapping = {old: int(permutation[old]) for old in range(n)}
    permuted_edges = []
    for index in range(data.edge_index.shape[1]):
        left, right = int(data.edge_index[0, index]), int(data.edge_index[1, index])
        if left == right:
            continue
        permuted_edges.append((mapping[left], mapping[right]))
    graph = from_edges(n, permuted_edges)
    # Node types travel with the physical nodes.
    permuted_types = np.zeros_like(node_types)
    for old, new in mapping.items():
        permuted_types[new] = node_types[old]
    # edge_types keyed by the relabelled edge (bond ids are labels, not ids).
    new_edge_types: dict[tuple[int, int], int] = {}
    for (left, right), bond in edge_types.items():
        x, y = mapping[left], mapping[right]
        new_edge_types[(min(x, y), max(x, y))] = bond
    return graph, permuted_types, new_edge_types, mapping


def test_structural_context_permutation_invariance() -> None:
    """Test 3: relabelling node IDs never changes per-node context keys."""
    # Two fused rings (3 shared atoms) + a tail chain: rich ring structure.
    edges = [(0, 1), (1, 2), (2, 0), (1, 3), (3, 4), (4, 0), (2, 5), (5, 6)]
    graph = from_edges(7, edges)
    node_types = np.asarray([0, 0, 1, 0, 2, 0, 0], dtype=np.int64)
    edge_types = {
        (min(a, b), max(a, b)): (1 + ((a * 7 + b) % 3)) for a, b in edges
    }
    contexts, _ = sc.extract_structural_context(graph, node_types, edge_types, 10)
    keys_original = {node: contexts[node].key for node in contexts}
    coarse_original = {node: tuple(contexts[node].coarse) for node in contexts}

    rng = np.random.default_rng(11)
    for _ in range(5):
        permutation = rng.permutation(7)
        relabelled, relabelled_types, relabelled_edges, mapping = _relabel_data(
            SimpleNamespace(
                num_nodes=7, edge_index=np.asarray([[a, b] for a, b in edges]).T,
                edge_attr=np.asarray([edge_types[(min(a, b), max(a, b))] for a, b in edges]),
            ),
            permutation,
            node_types,
            edge_types,
        )
        permuted_contexts, _ = sc.extract_structural_context(
            relabelled, relabelled_types, relabelled_edges, 10
        )
        for old, new in mapping.items():
            assert permuted_contexts[new].key == keys_original[old], (old, new)
            assert tuple(permuted_contexts[new].coarse) == coarse_original[old]


def test_direction_invariance_of_cycle_signature() -> None:
    """Test 4: forward and reverse traversal of a cycle give one signature."""
    edges = [(0, 1), (1, 2), (2, 3), (3, 0)]
    graph = from_edges(4, edges)
    node_types = np.asarray([0, 1, 2, 0], dtype=np.int64)
    edge_types = {(0, 1): 1, (1, 2): 1, (2, 3): 2, (0, 3): 2}
    forward = sc._typed_cycle_signature(
        graph, (0, 1, 2, 3), node_types, edge_types, 0
    )
    reverse = sc._typed_cycle_signature(
        graph, (0, 3, 2, 1), node_types, edge_types, 0
    )
    assert forward == reverse
    # A different centre on the same cycle gives a different signature.
    other = sc._typed_cycle_signature(graph, (1, 2, 3, 0), node_types, edge_types, 1)
    assert other != forward


def test_train_only_vocabulary_maps_unseen_to_unk() -> None:
    """Test 5: val-only context keys map to UNK_CONTEXT, never join the vocab."""
    train_key = b"v1:CYCLE|v1:0;1;0;1;0;1"
    val_key = b"v1:CYCLE|v1:5;2;5;2;5;2"


    def record(*keys: bytes) -> SimpleNamespace:
        patches = [
            SimpleNamespace(
                structural_context=SimpleNamespace(
                    key=key,
                    in_cycle=True,
                    cycle_count=1,
                    min_cycle_len=3,
                    max_cycle_len=3,
                    coarse=np.asarray([1.0, 1.0, 3.0, 3.0], dtype=np.float32),
                )
            )
            for key in keys
        ]
        return SimpleNamespace(patches=patches)

    train_records = [record(train_key, train_key), record(train_key)]
    vocabulary = sc.fit_structural_vocabulary(train_records)
    assert train_key in vocabulary
    assert vocabulary[train_key] >= 2
    # A ring context that only exists on validation:
    assert sc.encode_structural_token(SimpleNamespace(key=val_key, in_cycle=True), vocabulary) == sc.UNK_CONTEXT_TOKEN
    # NO_RING is its own reserved token:
    assert sc.encode_structural_token(SimpleNamespace(key=None, in_cycle=False), vocabulary) == sc.NO_RING_TOKEN
    # Cycle count / coarse statistics are still computed from graph structure:
    value = SimpleNamespace(
        patches=[
            SimpleNamespace(
                structural_context=SimpleNamespace(
                    key=train_key,
                    in_cycle=True,
                    cycle_count=1,
                    min_cycle_len=3,
                    max_cycle_len=3,
                    coarse=np.asarray([1.0, 2.0, 3.0, 5.0], dtype=np.float32),
                )
            )
        ]
    )
    stats = sc.context_statistics([value], vocabulary)
    assert stats["in_ring_fraction"] == 1.0
    assert stats["no_ring_fraction"] == 0.0


def test_conditioning_backward_is_finite() -> None:
    """Test 6: the conditioning path has real, finite gradients."""
    torch.manual_seed(5)
    model = module.PatchPathModel(
        20,
        8,
        patch_hidden=8,
        pair_hidden=4,
        token_width=6,
        dropout=0.0,
        embedding_mode="hybrid",
        embedding_rank=2,
        hybrid_full_typed_tokens=4,
        hybrid_full_parent_tokens=4,
        shell_width=7,
        structural_context_mode="typed_ring",
        structural_context_fusion="condition",
        structural_context_dim=4,
        structural_context_embedding_rank=1,
        structural_context_condition_rank=4,
        structural_context_vocabulary_size=6,
    )
    model.train()
    data = _data_for_v3(
        typed_vocab=20,
        tokens=torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long),
        coarse=torch.tensor(
            [[1.0, 1.0, 3.0, 3.0], [1.0, 2.0, 4.0, 5.0], [0, 0, 0, 0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0], [0.0, 0.0, 0.0, 0.0]],
            dtype=torch.float32,
        ),
        shell_width=7,
    )
    prediction = model(data)
    loss = torch.nn.functional.l1_loss(prediction, data.y)
    loss.backward()
    assert torch.isfinite(loss)
    for name in (
        "context_patch_projection",
        "context_condition_projection",
        "context_delta_output",
        "structural_context_embedding",
    ):
        parameter = getattr(model, name)
        assert parameter is not None
        if name == "structural_context_embedding":
            # The UNK row (index 0) receives no gradient; known rows do.
            assert parameter.embedding.weight.grad is not None
            assert torch.isfinite(parameter.embedding.weight.grad).all()
        else:
            assert parameter.weight.grad is not None
            assert torch.isfinite(parameter.weight.grad).all()


def test_cycle_enumeration_is_exact() -> None:
    """Cycle enumeration on small graphs: counts + lengths are exact."""
    # Triangle 0-1-2 and square 0-1-3-4 sharing edge 0-1 plus a tail 5-6.
    edges = [(0, 1), (1, 2), (0, 2), (1, 3), (3, 4), (4, 0), (2, 5), (5, 6)]
    graph = from_edges(7, edges)
    node_types = np.zeros(7, dtype=np.int64)
    edge_types = {(min(a, b), max(a, b)): 1 for a, b in edges}
    contexts, lengths = sc.extract_structural_context(graph, node_types, edge_types, 10)
    assert sorted(lengths) == [3, 4, 5]
    # node 0 lies on all three cycles (triangle, square, 5-cycle)
    assert contexts[0].cycle_count == 3
    assert contexts[0].min_cycle_len == 3
    assert contexts[0].max_cycle_len == 5
    # node 1 touches all three cycles (triangle, square, 5-cycle)
    assert contexts[1].cycle_count == 3
    assert contexts[1].min_cycle_len == 3
    assert contexts[1].max_cycle_len == 5
    assert contexts[3].cycle_count == 2
    assert contexts[6].in_cycle is False
    assert contexts[6].key is None
    assert np.array_equal(contexts[6].coarse, np.zeros(4, dtype=np.float32))
    # cap at 4 hides the 5-cycle (documented bounded definition)
    _, capped = sc.extract_structural_context(graph, node_types, edge_types, 4)
    assert sorted(capped) == [3, 4]


def test_v3_typed_condition_parameter_budget() -> None:
    """v3 typed conditioning stays within the 110k budget at both phases."""
    vocab = 9_700  # measured train context keys at max_cycle_len=10
    condition = module.PatchPathModel(
        SELECTION_TYPED_VOCAB,
        PARENT_VOCAB,
        patch_hidden=48,
        pair_hidden=16,
        token_width=16,
        dropout=0.05,
        embedding_mode="hybrid",
        embedding_rank=4,
        hybrid_full_typed_tokens=768,
        hybrid_full_parent_tokens=32,
        center_context=True,
        center_context_hidden=60,
        graph_head_hidden_0=64,
        graph_head_hidden_1=32,
        shell_width=module._shell_width_for_radius(module.PATCH_RADIUS),
        context_width=0,
        structural_context_mode="typed_ring",
        structural_context_fusion="condition",
        structural_context_dim=4,
        structural_context_embedding_rank=1,
        structural_context_condition_rank=8,
        structural_context_vocabulary_size=vocab + 2,
    )
    audit = module.audit_parameters(condition)
    assert audit["total_trainable"] <= V3_BUDGET
    assert audit["blocks"]["structural_context_embedding"] == (vocab + 1) * 1 + 1 * 4
    assert audit["blocks"]["context_patch_projection"] == 16 * 8
    assert audit["blocks"]["context_condition_projection"] == 4 * 8
    assert audit["blocks"]["context_delta_output"] == 8 * 16 + 16
    # Formula check against the compact-v2 refit total (selection-phase
    # baseline 98,549 is used by the model built above; the refit baseline is
    # 99,613 and the delta is the context module only).
    assert audit["total_trainable"] == (
        COMPACT_V2_TOTALS["validation-selection"]
        + (vocab + 1) * 1
        + 4
        + 16 * 8
        + 4 * 8
        + (8 * 16 + 16)
    )


def test_v3_typed_concat_parameter_budget() -> None:
    """v3 typed concat ablation stays within the 110k budget."""
    vocab = 9_700
    concat = module.PatchPathModel(
        SELECTION_TYPED_VOCAB,
        PARENT_VOCAB,
        patch_hidden=48,
        pair_hidden=16,
        token_width=16,
        dropout=0.05,
        embedding_mode="hybrid",
        embedding_rank=4,
        hybrid_full_typed_tokens=768,
        hybrid_full_parent_tokens=32,
        center_context=True,
        center_context_hidden=60,
        graph_head_hidden_0=64,
        graph_head_hidden_1=32,
        shell_width=module._shell_width_for_radius(module.PATCH_RADIUS),
        context_width=0,
        structural_context_mode="typed_ring",
        structural_context_fusion="concat",
        structural_context_dim=4,
        structural_context_embedding_rank=1,
        structural_context_condition_rank=8,
        structural_context_vocabulary_size=vocab + 2,
    )
    audit = module.audit_parameters(concat)
    assert audit["total_trainable"] <= V3_BUDGET
    assert audit["blocks"]["context_delta_output"] == 0
    # concat widens the patch encoder input by context_dim.
    assert concat.patch_encoder.layers[0].in_features == 146 + 16 + 8 + 4


def test_context_statistics_cycle_distribution() -> None:
    """Cycle length distribution and NO_RING fractions are reported per split."""
    # Two triangles (0-1-2 and 2-3-4) sharing node 2.
    edges = [(0, 1), (1, 2), (0, 2), (2, 3), (3, 4), (4, 2)]
    graph = from_edges(5, edges)
    edge_types = {(min(a, b), max(a, b)): 1 for a, b in edges}
    contexts, lengths = sc.extract_structural_context(
        graph, np.zeros(5, dtype=np.int64), edge_types, 10
    )
    # two triangles sharing node 2
    assert sorted(lengths) == [3, 3]
    assert contexts[2].cycle_count == 2
    assert sc.cycle_length_distribution([SimpleNamespace(cycle_lengths=lengths)]) == {"3": 2}

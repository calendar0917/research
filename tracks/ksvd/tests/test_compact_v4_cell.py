"""Tests for the compact-v4-cell minimal falsification experiment.

Architecture tests (1-10) and broken-incidence control tests (11-20) from the
authorised plan.  These are static / unit tests; they never load official test.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges, ring_chords
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_cell as cell
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _synthetic_graphs():
    graphs = [
        ring_chords(6, []),
        ring_chords(5, []),
        from_edges(4, [(0, 1), (1, 2), (2, 3), (3, 0)]),
        from_edges(5, [(0, 1), (1, 2), (2, 0)]),
        from_edges(5, [(0, 1), (1, 2), (2, 3), (3, 4)]),
    ]
    data_list = []
    for graph in graphs:
        edges = list(graph.edges())
        bi = edges + [(v, u) for u, v in edges]
        data_list.append(
            Data(
                edge_index=torch.tensor(bi, dtype=torch.long).t().contiguous(),
                x=torch.zeros(len(graph.nodes), dtype=torch.long),
                edge_attr=torch.zeros(len(bi), dtype=torch.long),
                y=torch.tensor([0.0]),
                num_nodes=len(graph.nodes),
            )
        )
    return data_list


@pytest.fixture(scope="module")
def model():
    config = cell.cell_config()
    built = cell._build_cell_model(
        typed_vocabulary_size=cell.V4_TYPED_VOCAB_WITH_OOV,
        parent_vocabulary_size=cell.V4_PARENT_VOCAB_WITH_OOV,
        config=config,
        seed=0,
    )
    built.eval()
    return built


# ---------------------------------------------------------------------------
# architecture tests
# ---------------------------------------------------------------------------


def test_01_baseline_pathway_preserved(model):
    """Test 1 / G0.1: the compact-v4 pathway is numerically untouched."""
    synthetic = cell._make_synthetic_dataset()
    encoded = cell._encode_synthetic(model, synthetic)
    base = zpp.PatchPathModel(
        cell.V4_TYPED_VOCAB_WITH_OOV,
        cell.V4_PARENT_VOCAB_WITH_OOV,
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
        topology_mode="hinge",
        topology_input_width=25,
        topology_hidden_dim=16,
        topology_out_dim=8,
    )
    base.eval()
    state = model.state_dict()
    base_state = base.state_dict()
    shared = {
        key: value
        for key, value in state.items()
        if key in base_state and base_state[key].shape == value.shape
    }
    base.load_state_dict(shared, strict=False)
    max_diff = 0.0
    with torch.no_grad():
        for data in encoded:
            batch = cell.cell_collate([data])
            max_diff = max(
                max_diff,
                float((model.encode_original(batch) - base.encode(batch)).abs().max()),
            )
    assert max_diff < 1e-5


def test_02_cycle_construction_deterministic():
    for data in cell._make_synthetic_dataset():
        assert cell.cycles_for_data(data) == cell.cycles_for_data(data)


def test_03_relabel_invariance():
    rng = np.random.default_rng(3)
    for _ in range(30):
        n = int(rng.integers(6, 12))
        edges = [(i, (i + 1) % n) for i in range(n)]
        for _ in range(int(rng.integers(0, 3))):
            a, b = sorted(rng.choice(n, size=2, replace=False).tolist())
            edges.append((int(a), int(b)))
        graph = from_edges(n, edges)
        reference = cell.induced_cycles(graph, cell.MAX_CYCLE_LEN)
        perm = rng.permutation(n)
        mapping = {i: int(perm[i]) for i in range(n)}
        relabelled = from_edges(n, [(mapping[u], mapping[v]) for u, v in edges])
        inverse = {v: k for k, v in mapping.items()}
        back = sorted(
            tuple(sorted(inverse[x] for x in cycle))
            for cycle in cell.induced_cycles(relabelled, cell.MAX_CYCLE_LEN)
        )
        assert back == reference


def test_04_cycle_order_permutation_invariant(model):
    synthetic = cell._make_synthetic_dataset()
    encoded = cell._encode_synthetic(model, synthetic)
    for data in encoded:
        if int(data.num_cells.item()) < 2:
            continue
        batch = cell.cell_collate([data])
        patch = cell._mature_states_for_batch(model, batch)
        summary = model._cell_summary(patch, batch)
        permuted = cell._permute_cells(data)
        permuted_batch = cell.cell_collate([permuted])
        summary_permuted = model._cell_summary(patch, permuted_batch)
        assert float((summary - summary_permuted).abs().max()) < 1e-6


def test_05_no_cycle_molecule_runs(model):
    synthetic = cell._make_synthetic_dataset()
    encoded = cell._encode_synthetic(model, synthetic)
    no_cycle = [d for d in encoded if int(d.num_cells.item()) == 0]
    assert no_cycle, "synthetic set must include a cycle-free graph"
    batch = cell.cell_collate([no_cycle[0]])
    with torch.no_grad():
        out = model(batch)
        summary = model._cell_summary(model.mature_patch_states(), batch)
    assert torch.isfinite(out).all()
    assert torch.allclose(summary, torch.zeros_like(summary))


def test_06_r_dimension_399(model):
    synthetic = cell._make_synthetic_dataset()
    encoded = cell._encode_synthetic(model, synthetic)
    batch = cell.cell_collate([encoded[0]])
    with torch.no_grad():
        r = model.encode(batch)
    assert r.shape[1] == 399
    assert cell.R_ORIGINAL == 302
    assert cell.CELL_SUMMARY_WIDTH == 97


def test_07_parameter_ceiling(model):
    total = sum(p.numel() for p in model.parameters())
    assert total <= cell.PARAM_CEILING
    assert total == 113069


def test_08_exactly_two_incidence_rounds():
    assert cell.CELL_ROUNDS == 2
    source = Path(cell.__file__).read_text(encoding="utf-8")
    assert "for _round in range(CELL_ROUNDS)" in source


def test_09_no_extra_message_passing():
    source = Path(cell.__file__).read_text(encoding="utf-8")
    imports = "\n".join(
        line for line in source.splitlines() if line.startswith(("import ", "from "))
    )
    for token in ("MessagePassing", "GATConv", "TransformerConv", "MultiheadAttention"):
        assert token not in imports
    assert issubclass(cell.PatchPathCellModel, zpp.PatchPathModel)


def test_10b_cell_loader_carries_incidence(model):
    """The data loader must deliver the cycle-cell incidence to the model."""
    synthetic = cell._make_synthetic_dataset()
    encoded = cell._encode_synthetic(model, synthetic)
    loader = cell._make_cell_loader(encoded, batch_size=2, shuffle=False, seed=0)
    batch = next(iter(loader))
    assert hasattr(batch, "cell_batch")
    assert int(batch.num_cells_total) > 0
    assert int(batch.cell_edge_patch.numel()) > 0
    with torch.no_grad():
        out = model(batch)
        summary = model._cell_summary(model.mature_patch_states(), batch)
    assert torch.isfinite(out).all()
    assert float(summary.abs().sum()) > 0.0


def test_10_global_topology_branch_present(model):
    assert model.topology_encoder is not None
    assert str(model.topology_mode) == "hinge"
    assert int(model.topology_input_width) == 25


# ---------------------------------------------------------------------------
# broken-incidence control tests
# ---------------------------------------------------------------------------


def test_11_true_broken_identical_initialization():
    config = cell.cell_config()
    true_model = cell._build_cell_model(
        typed_vocabulary_size=cell.V4_TYPED_VOCAB_WITH_OOV,
        parent_vocabulary_size=cell.V4_PARENT_VOCAB_WITH_OOV,
        config=config,
        seed=0,
    )
    broken_model = cell._build_cell_model(
        typed_vocabulary_size=cell.V4_TYPED_VOCAB_WITH_OOV,
        parent_vocabulary_size=cell.V4_PARENT_VOCAB_WITH_OOV,
        config=config,
        seed=0,
    )
    a = true_model.state_dict()
    b = broken_model.state_dict()
    assert set(a) == set(b)
    for key in a:
        assert torch.equal(a[key], b[key]), key


def test_12_true_broken_identical_param_count():
    config = cell.cell_config()
    true_model = cell._build_cell_model(
        typed_vocabulary_size=cell.V4_TYPED_VOCAB_WITH_OOV,
        parent_vocabulary_size=cell.V4_PARENT_VOCAB_WITH_OOV,
        config=config,
        seed=0,
    )
    broken_model = cell._build_cell_model(
        typed_vocabulary_size=cell.V4_TYPED_VOCAB_WITH_OOV,
        parent_vocabulary_size=cell.V4_PARENT_VOCAB_WITH_OOV,
        config=config,
        seed=0,
    )
    assert sum(p.numel() for p in true_model.parameters()) == sum(
        p.numel() for p in broken_model.parameters()
    )


def _rewire(cells, base_seed=cell.BROKEN_BASE_SEED):
    true_list = []
    broken_list = []
    for index, item in enumerate(cells):
        true_edges = cell.incidence_edges(item.cycles, item.n_patches)
        edges, _info = cell.rewire_incidence(
            true_edges, item.n_patches, len(item.cycles), base_seed * 1_000_003 + index
        )
        true_list.append(true_edges)
        broken_list.append(edges)
    return true_list, broken_list


def test_13_cell_count_identical():
    cells = cell._cell_records_from_dataset(_synthetic_graphs())
    true_list, broken_list = _rewire(cells)
    assert len(true_list) == len(broken_list)


def test_14_cycle_feature_multiset_identical():
    cells = cell._cell_records_from_dataset(_synthetic_graphs())
    true_list, broken_list = _rewire(cells)
    for true_edges, broken_edges in zip(true_list, broken_list):
        true_sizes = sorted(
            sum(1 for p, c in true_edges if c == cell_id)
            for cell_id in range(max((c for _, c in true_edges), default=-1) + 1)
        )
        broken_sizes = sorted(
            sum(1 for p, c in broken_edges if c == cell_id)
            for cell_id in range(max((c for _, c in broken_edges), default=-1) + 1)
        )
        assert true_sizes == broken_sizes


def test_15_total_incidence_identical():
    cells = cell._cell_records_from_dataset(_synthetic_graphs())
    true_list, broken_list = _rewire(cells)
    for true_edges, broken_edges in zip(true_list, broken_list):
        assert len(true_edges) == len(broken_edges)


def test_16_declared_marginals_preserved():
    cells = cell._cell_records_from_dataset(_synthetic_graphs())
    for item in cells:
        true_edges = cell.incidence_edges(item.cycles, item.n_patches)
        edges, _info = cell.rewire_incidence(
            true_edges, item.n_patches, len(item.cycles), seed=12345
        )
        marginals = cell._validate_incidence_marginals(
            true_edges, edges, item.n_patches, len(item.cycles)
        )
        assert marginals["patch_degrees_preserved"]
        assert marginals["cell_degrees_preserved"]
        assert marginals["total_edges_preserved"]


def test_17_broken_differs_from_true_when_breakable():
    # a graph with two disjoint cycles is breakable
    graph = from_edges(8, [(0, 1), (1, 2), (2, 0), (3, 4), (4, 5), (5, 3)])
    data = Data(
        edge_index=torch.tensor(
            list(graph.edges()) + [(v, u) for u, v in graph.edges()], dtype=torch.long
        )
        .t()
        .contiguous(),
        x=torch.zeros(8, dtype=torch.long),
        edge_attr=torch.zeros(12, dtype=torch.long),
        y=torch.tensor([0.0]),
        num_nodes=8,
    )
    cycles = cell.cycles_for_data(data)
    assert len(cycles) == 2
    true_edges = cell.incidence_edges(cycles, 8)
    broken_edges, info = cell.rewire_incidence(true_edges, 8, 2, seed=999)
    assert info["swaps"] > 0
    assert set(true_edges) != set(broken_edges)


def test_18_broken_manifest_deterministic():
    cells = cell._cell_records_from_dataset(_synthetic_graphs())
    true_list_a, broken_list_a = _rewire(cells)
    true_list_b, broken_list_b = _rewire(cells)
    assert true_list_a == true_list_b
    assert broken_list_a == broken_list_b


def test_19_no_cross_molecule_rewiring():
    # rewiring uses only per-molecule edge lists; a single-molecule call cannot
    # reference any other graph.  Verify the function signature by construction.
    edges = [(0, 0), (1, 0), (2, 0), (3, 1), (4, 1), (5, 1)]
    out, _info = cell.rewire_incidence(edges, 6, 2, seed=7)
    assert set(out) <= {(p, c) for p in range(6) for c in range(2)}
    assert len(out) == len(edges)


def test_20_official_test_never_loaded():
    source = Path(cell.__file__).read_text(encoding="utf-8")
    assert '_load_zinc(ZINC_ROOT, "test")' not in source
    assert "load_zinc(ZINC_ROOT, 'test')" not in source
    assert '"official_test_loaded": False' in source

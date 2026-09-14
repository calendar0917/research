"""Tests for the ZINC 2x2 centre/encoder capacity decomposition.

Static / unit tests only: exact parameter accounting for the four cells, the
initialisation identity of cells A/D with the frozen H64/H96 builders, the
"only these tensors change shape" property of cells B/C, weight tying and a
finite forward/backward.  No training is performed and official test is never
loaded.
"""

from __future__ import annotations

import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges, ring_chords
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity as cap,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity_decomposition as cd,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_hwidth as hw,
)
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo


def _synthetic_batch():
    topo_width = int(ztopo.raw_width("hinge"))
    datasets: list[Data] = []
    for graph in (
        ring_chords(6, []),
        ring_chords(5, []),
        from_edges(6, [(0, 1), (1, 2), (2, 0)]),
    ):
        n = len(graph.nodes)
        pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
        pair_index = (
            torch.tensor(pairs, dtype=torch.long).t().contiguous()
            if pairs
            else torch.zeros(2, 0, dtype=torch.long)
        )
        buckets = torch.tensor(
            [min(max(i % 5, 0), 4) for i in range(len(pairs))], dtype=torch.long
        )
        datasets.append(
            Data(
                patch_cont=torch.zeros(n, int(zpp.SHELL_WIDTH)),
                patch_context=torch.zeros(n, 0),
                typed_token=torch.zeros(n, dtype=torch.long),
                parent_token=torch.zeros(n, dtype=torch.long),
                structural_token=torch.zeros(n, dtype=torch.long),
                structural_coarse=torch.zeros(n, 0),
                pair_index=pair_index,
                pair_relation=torch.zeros(len(pairs), int(zpp.RELATION_WIDTH)),
                pair_bucket=buckets,
                global_context=torch.zeros(1, int(zpp.GLOBAL_WIDTH)),
                topology_features=torch.zeros(1, topo_width),
                y=torch.zeros(1),
                num_nodes=n,
            )
        )
    return next(iter(zpp._make_loader(datasets, 128, False, 0)))


def _state_diff(x, y) -> float:
    xs, ys = x.state_dict(), y.state_dict()
    return max(float((xs[k] - ys[k]).abs().max()) for k in xs)


def test_01_exact_parameter_accounting():
    for cell, expected in cd.EXPECTED_PARAMS.items():
        model = cd.build_cell(cell, 0)
        assert cap._n_params(model) == expected
        assert cap._n_params(model.head) == cd.EXPECTED_HEAD[cell]
        assert int(model.unified_graph_width) == cd.EXPECTED_WIDTH[cell]


def test_02_cells_A_and_D_match_the_frozen_builders():
    assert _state_diff(cd.build_cell("A", 0), hw.build_h64(0)) == 0.0
    assert _state_diff(cd.build_cell("D", 0), cap.build_capacity(0, h=96, q=16)) == 0.0


def test_03_cell_B_changes_only_state_tensors():
    a = cd.build_cell("A", 0).state_dict()
    b = cd.build_cell("B", 0).state_dict()
    changed = {k for k in a if a[k].shape != b[k].shape}
    assert changed == set(hw.H_DEPENDENT_KEYS)


def test_04_cell_C_changes_only_encoder_hidden_tensors():
    a = cd.build_cell("A", 0).state_dict()
    c = cd.build_cell("C", 0).state_dict()
    changed = {k for k in a if a[k].shape != c[k].shape}
    assert changed
    assert all(k.startswith(("patch_encoder.", "global_encoder.")) for k in changed)


def test_05_weight_tying_and_refresh():
    batch = _synthetic_batch()
    model = cd.build_cell("B", 0)
    counts = model.module_call_counts(batch)
    assert counts == {"pair_projection": 4, "pair_encoder": 2, "center_update": 2}
    assert int(model.pair_hidden) == 16
    assert int(model.recurrence_rounds) == 2
    assert bool(model.recurrence_enabled)
    assert str(model.recurrence_mode) == "refresh"


def test_06_forward_backward_finite_nonzero():
    batch = _synthetic_batch()
    for cell in cd.CELL_ORDER:
        model = cd.build_cell(cell, 0)
        model.train()
        out = model(batch)
        assert bool(torch.isfinite(out).all())
        loss = torch.nn.functional.l1_loss(out.view(-1), batch.y.view(-1))
        loss.backward()
        assert any(
            p.grad is not None and float(p.grad.abs().max()) > 0.0
            for p in model.parameters()
        )


def test_07_q_dim_fixed_across_cells():
    for cell in cd.CELL_ORDER:
        model = cd.build_cell(cell, 0)
        assert int(model.pair_hidden) == 16
        assert int(model.pair_encoder.layers[-2].out_features) == 16
        assert int(model.center_context_hidden) == 60

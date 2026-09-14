"""Tests for the compact-v4 T=2 recurrent pair--centre capacity-scaling audit.

Static / unit tests only: exact parameter accounting, regression identity of
the Q16 capacity builder against the frozen recurrent builder, finite
forward/backward and unchanged weight tying for every width variant, and the
two temporarily-relaxed construction guards being correctly restored.  No
training is performed and official test is never loaded.
"""

from __future__ import annotations

import inspect

import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges, ring_chords
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre as rec,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity as cap,
)
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as sh
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


def test_01_exact_parameter_accounting():
    expected = {
        "Q16": (82115, 4135),
        "Q24": (91211, 5175),
        "Q32": (100307, 6215),
        "BALANCED": (104211, 6631),
    }
    for name, (total, head) in expected.items():
        model = cap.BUILDERS[name](0)
        assert cap._n_params(model) == total
        assert cap._n_params(model.head) == head
        assert int(model.recurrence_rounds) == 2
        assert model.recurrence_mode == "refresh"


def test_02_unified_graph_width_formula():
    for name, cfg in cap.CONFIGS.items():
        model = cap.BUILDERS[name](0)
        assert int(model.unified_graph_width) == cap.relation_readout_width(
            cfg["h"], cfg["q"]
        )
        assert int(model.pair_hidden) == cfg["q"]


def test_03_q16_builder_matches_frozen_recurrent():
    frozen = rec.build_recurrent(0)
    capacity = cap.build_q16(0)
    frozen_state = frozen.state_dict()
    capacity_state = capacity.state_dict()
    assert set(frozen_state) == set(capacity_state)
    for key in frozen_state:
        assert frozen_state[key].shape == capacity_state[key].shape
    assert cap._state_hash(frozen_state) == cap._state_hash(capacity_state)


def test_04_construction_guards_restored():
    assert rec.Q_DIM == 16
    assert sh.R_DIM == 302
    cap.build_q32(0)
    cap.build_balanced(0)
    assert rec.Q_DIM == 16
    assert sh.R_DIM == 302


def test_05_head_hidden_definition_unchanged():
    for name in cap.BUILDERS:
        model = cap.BUILDERS[name](0)
        assert tuple(model.small_head_hidden) == tuple(sh.SMALL_HEAD_HIDDEN)
        # head is Linear(R,13)->ReLU->Linear(13,13)->ReLU->Linear(13,1)
        linear_widths = [
            int(module.out_features)
            for module in model.head.modules()
            if isinstance(module, torch.nn.Linear)
        ]
        assert linear_widths == [13, 13, 1]


def test_06_all_variants_forward_backward_finite():
    batch = _synthetic_batch()
    for name, builder in cap.BUILDERS.items():
        model = builder(0)
        model.train()
        out = model(batch)
        loss = torch.nn.functional.l1_loss(out.view(-1), batch.y.view(-1))
        loss.backward()
        assert bool(torch.isfinite(out).all())
        assert all(
            parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
            for parameter in model.parameters()
        )


def test_07_weight_tying_two_rounds_every_variant():
    batch = _synthetic_batch()
    for name, builder in cap.BUILDERS.items():
        model = builder(0)
        model.eval()
        counts = model.module_call_counts(batch)
        assert counts == {
            "pair_projection": 4,
            "pair_encoder": 2,
            "center_update": 2,
        }


def test_08_q_min_widths_scale_with_q_dim():
    narrow = cap.build_q16(0)
    wide = cap.build_q32(0)
    assert int(narrow.pair_projection.out_features) == 16
    assert int(wide.pair_projection.out_features) == 32
    assert int(wide.pooled_pair_width) == 2 * 32 + 1
    assert int(narrow.pooled_unary_width) == int(wide.pooled_unary_width) == 2 * 48 + 1
    assert int(narrow.center_context_width) == 5 * (2 * 16 + 1)
    assert int(wide.center_context_width) == 5 * (2 * 32 + 1)


def test_09_activation_stats_are_correct_on_known_tensor():
    value = torch.tensor(
        [[0.0, 1.0, -1.0], [0.0, 2.0, 0.0], [0.0, 3.0, 1.0]]
    )
    stats = cap._activation_stats(value)
    assert stats["zero_fraction"] == 4.0 / 9.0
    assert stats["q_dim"] == 3
    assert stats["dead_dim_fraction"] == 1.0 / 3.0  # first column is all zero
    assert stats["never_active_dim_fraction"] == 1.0 / 3.0


def test_10_scope_and_protocol_inherited():
    assert cap.PROTOCOL_VERSION.endswith("_v1")
    assert sh.OPTIMIZED_PROTOCOL["max_epochs"] == 240
    assert sh.OPTIMIZED_PROTOCOL["patience"] == 40
    assert sh.OPTIMIZED_PROTOCOL["learning_rate"] == 1.0e-3
    assert sh.OPTIMIZED_PROTOCOL["weight_decay"] == 1.0e-5
    source = inspect.getsource(cap)
    for forbidden in ("nn.MultiheadAttention", "TransformerEncoderLayer", "path_features"):
        assert forbidden not in source
    # training must never request an official-test identity read
    assert "real_batch_identity=False" in source
    assert "official_test_loaded=True" not in source

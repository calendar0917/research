"""Tests for the compact-v4 centre/patch-state width (H48 -> H64) audit.

Static / unit tests only: exact parameter accounting, the *only* state_dict
tensors whose shapes change, backbone initialisation identity with H48, the
unchanged relation width / recurrence flags, finite forward-backward and
weight tying.  No training is performed and official test is never loaded.
"""

from __future__ import annotations

import inspect

import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges, ring_chords
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_capacity as cap,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_hwidth as hw,
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


def test_01_exact_h64_parameter_accounting():
    model = hw.build_h64(0)
    assert cap._n_params(model) == 85763
    assert cap._n_params(model.head) == 4551
    assert int(model.unified_graph_width) == 334
    assert int(model.pooled_unary_width) == 129
    assert int(model.pooled_pair_width) == 33
    assert int(model.center_context_width) == 165


def test_02_only_h_dependent_shapes_change():
    h48 = cap.build_q16(0).state_dict()
    h64 = hw.build_h64(0).state_dict()
    changed = {key for key in h48 if h48[key].shape != h64[key].shape}
    assert changed == set(hw.H_DEPENDENT_KEYS)
    # every same-shape backbone tensor keeps the exact H48 initialisation
    for key in h48:
        if key in changed or key.startswith("head."):
            continue
        assert torch.equal(h48[key], h64[key]), key


def test_03_relation_width_and_recurrence_unchanged():
    model = hw.build_h64(0)
    assert int(model.pair_hidden) == 16
    assert int(model.pair_encoder.layers[-2].out_features) == 16
    assert int(model.recurrence_rounds) == 2
    assert bool(model.recurrence_enabled)
    assert model.recurrence_mode == "refresh"


def test_04_two_round_weight_tying():
    batch = _synthetic_batch()
    model = hw.build_h64(0).eval()
    counts = model.module_call_counts(batch)
    assert counts == {"pair_projection": 4, "pair_encoder": 2, "center_update": 2}


def test_05_forward_backward_finite():
    batch = _synthetic_batch()
    model = hw.build_h64(0)
    model.train()
    out = model(batch)
    loss = torch.nn.functional.l1_loss(out.view(-1), batch.y.view(-1))
    loss.backward()
    assert bool(torch.isfinite(out).all())
    assert all(
        parameter.grad is None or bool(torch.isfinite(parameter.grad).all())
        for parameter in model.parameters()
    )


def test_06_h48_reference_untouched_and_no_test():
    assert hw.EXPECTED_H48_TOTAL == 82115
    assert hw.H48_REF_SOUP_MEAN == 0.13464355785958469
    assert sh.OPTIMIZED_PROTOCOL["max_epochs"] == 240
    assert sh.OPTIMIZED_PROTOCOL["patience"] == 40
    assert sh.OPTIMIZED_PROTOCOL["learning_rate"] == 1.0e-3
    assert sh.OPTIMIZED_PROTOCOL["weight_decay"] == 1.0e-5
    source = inspect.getsource(hw)
    assert "nn.MultiheadAttention" not in source
    assert "TransformerEncoderLayer" not in source
    assert "real_batch_identity=False" in source
    assert "official_test_loaded=True" not in source

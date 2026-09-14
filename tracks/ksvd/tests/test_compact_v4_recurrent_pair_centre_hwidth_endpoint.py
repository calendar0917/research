"""Static tests for the H64 -> H96 centre-width endpoint.

No training and, critically, **no official-test load**: these tests only build
models and inspect the frozen invariants / gating discipline.
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
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre_hwidth_endpoint as ep,
)
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo


def _synthetic_batch():
    topo_width = int(ztopo.raw_width("hinge"))
    datasets: list[Data] = []
    for graph in (ring_chords(6, []), from_edges(6, [(0, 1), (1, 2), (2, 0)])):
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


def test_01_h96_exact_parameter_accounting():
    model = ep.build_h96(0)
    assert cap._n_params(model) == 103219
    assert cap._n_params(model.head) == 5383
    assert int(model.unified_graph_width) == 398
    assert int(model.pooled_unary_width) == 193
    assert int(model.pooled_pair_width) == 33
    assert int(model.center_context_width) == 165
    assert int(model.center_context_hidden) == 60


def test_02_only_h_dependent_shapes_change_vs_h64():
    h64 = hw.build_h64(0).state_dict()
    h96 = ep.build_h96(0).state_dict()
    changed = {key for key in h64 if h64[key].shape != h96[key].shape}
    assert changed == set(ep.H96_DEPENDENT_KEYS)
    for key in h64:
        if key in changed or key.startswith("head."):
            continue
        assert torch.equal(h64[key], h96[key]), key


def test_03_relation_and_recurrence_unchanged():
    model = ep.build_h96(0)
    assert int(model.pair_hidden) == 16
    assert int(model.pair_encoder.layers[-2].out_features) == 16
    assert int(model.recurrence_rounds) == 2
    assert model.recurrence_mode == "refresh"


def test_04_two_round_weight_tying_and_finite():
    batch = _synthetic_batch()
    model = ep.build_h96(0)
    model.train()
    out = model(batch)
    loss = torch.nn.functional.l1_loss(out.view(-1), batch.y.view(-1))
    loss.backward()
    assert bool(torch.isfinite(out).all())
    model.eval()
    counts = model.module_call_counts(batch)
    assert counts == {"pair_projection": 4, "pair_encoder": 2, "center_update": 2}


def test_05_h48_h64_references_and_protocol_unchanged():
    assert ep.H64_TOTAL == 85763
    assert ep.H64_SOUP_MEAN == 0.13082823178218678
    assert ep.EXPECTED_H96_TOTAL == 103219
    assert ep.H48_TEST_RAW_MEAN == 0.11495107560744508


def test_06_test_is_gated_behind_freeze_and_never_in_validation_source():
    source = inspect.getsource(ep)
    # the only test-split loader is inside ``test_eval``
    assert source.count("extract_test_records") == 1
    assert "architecture_freeze.json" in source
    assert "official_test_unlock.json" in source
    assert "test_unlock_authorised" in source
    # validation stages must not mark the test as loaded
    assert "official_test_loaded\": False" in source

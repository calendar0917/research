"""Tests for the corrected frozen-state export + readout-sufficiency audit.

These pin the measurement repairs and the adapter contract used by
``experiments/luyin16/zinc_frozen_readout_sufficiency.py`` without touching
the real dataset or any checkpoint:

1.  pair graph ids come from graph membership, not the source patch index;
2.  exported pair counts equal ``n choose 2`` (unordered all-pairs);
3.  every pair row is consumed exactly once;
4.  the pre-head ``R`` hook captures the *input* to ``model.head[0]`` (302D),
    not its 64D output;
5.  ``R`` reconstructed from ``h'_i`` / ``q_ij`` / bucket ids matches the
    captured true pre-head R;
6.  the reconstructed R reproduces the frozen prediction through the head;
7.  batch size / graph ordering do not change per-molecule summaries;
8.  patch (node) permutation leaves R and the prediction invariant;
9.  a legacy/foreign cache cannot be loaded as a corrected export;
10. the set adapter genuinely depends on its set summary.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_frozen_readout_sufficiency as frs


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------

def _tiny_model() -> zpp.PatchPathModel:
    torch.manual_seed(0)
    return zpp.PatchPathModel(
        8,
        4,
        patch_hidden=frs.PATCH_DIM,
        pair_hidden=frs.PAIR_DIM,
        token_width=8,
        dropout=0.0,
        embedding_mode="full",
        center_context=True,
        center_context_hidden=12,
        graph_head_hidden_0=16,
        graph_head_hidden_1=8,
        shell_width=zpp._shell_width_for_radius(zpp.PATCH_RADIUS),
        context_width=0,
        topology_mode="hinge",
        topology_input_width=25,
        topology_hidden_dim=8,
        topology_out_dim=8,
    )


def _dummy_graph(n: int = 6, seed: int = 0) -> zpp.Data:
    generator = torch.Generator().manual_seed(seed)
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_index = torch.tensor(
        [[i for i, _ in pairs], [j for _, j in pairs]], dtype=torch.long
    )
    return zpp.Data(
        patch_cont=torch.randn(n, zpp._shell_width_for_radius(zpp.PATCH_RADIUS), generator=generator),
        patch_context=torch.zeros(n, 0),
        typed_token=torch.randint(0, 8, (n,), generator=generator),
        parent_token=torch.randint(0, 4, (n,), generator=generator),
        structural_token=torch.zeros(n, dtype=torch.long),
        structural_coarse=torch.zeros(n, 4),
        pair_index=pair_index,
        pair_relation=torch.randn(len(pairs), zpp.RELATION_WIDTH, generator=generator),
        pair_bucket=torch.randint(0, frs.N_BUCKETS, (len(pairs),), generator=generator),
        global_context=torch.randn(1, zpp.GLOBAL_WIDTH, generator=generator),
        batch=torch.zeros(n, dtype=torch.long),
        y=torch.tensor([0.0]),
        num_nodes=n,
        topology_features=torch.randn(1, 25, generator=generator),
    )


def _synthetic_export(n_patches, *, corrupt_counts=False, corrupt_endpoint=False, corrupt_coverage=False):
    n_patches = np.asarray(n_patches, dtype=np.int64)
    n_graphs = len(n_patches)
    n_pairs = n_patches * (n_patches - 1) // 2
    stored_pairs = n_pairs.copy()
    if corrupt_counts:
        stored_pairs = stored_pairs + 1
    source_local, target_local, source_graph, target_graph = [], [], [], []
    for graph, size in enumerate(n_patches):
        combos = [(i, j) for i in range(int(size)) for j in range(i + 1, int(size))]
        if corrupt_coverage and combos:
            combos = combos[:-1]  # drop one pair row
        for i, j in combos:
            source_local.append(i)
            target_local.append(j)
            source_graph.append(graph)
            target_graph.append(graph if not corrupt_endpoint else graph + 1)
    return {
        "subset_index": np.arange(n_graphs, dtype=np.int64),
        "n_patches": n_patches,
        "n_pairs": stored_pairs,
        "pair_source_local": np.asarray(source_local, dtype=np.int64),
        "pair_target_local": np.asarray(target_local, dtype=np.int64),
        "pair_source_graph": np.asarray(source_graph, dtype=np.int64),
        "pair_target_graph": np.asarray(target_graph, dtype=np.int64),
    }


# ---------------------------------------------------------------------------
# Test 1 / 2 -- pair graph grouping and counts
# ---------------------------------------------------------------------------

def test_pair_counts_equal_n_choose_2():
    export = _synthetic_export([1, 3, 6])
    assert frs._gate_pair_grouping(export)["passed"]
    assert not frs._gate_pair_grouping(_synthetic_export([3, 4], corrupt_counts=True))["passed"]


def test_pair_endpoint_graph_identity():
    export = _synthetic_export([3, 4])
    assert frs._gate_pair_endpoint_identity(export)["passed"]
    bad = _synthetic_export([3, 4], corrupt_endpoint=True)
    assert not frs._gate_pair_endpoint_identity(bad)["passed"]


def test_capture_states_pair_grouping_is_graph_membership():
    """The corrected capture groups pairs by graph, not by source index."""
    model = _tiny_model()
    model.eval()
    graphs = [_dummy_graph(n=4, seed=1), _dummy_graph(n=6, seed=2)]
    capture = frs._capture_states(model, graphs, batch_size=2)
    capture["subset_index"] = np.arange(2)
    assert np.array_equal(
        capture["pair_source_graph"][: len(capture["pair_source_graph"])],
        capture["pair_target_graph"],
    )
    assert frs._gate_pair_grouping(capture)["passed"]
    assert frs._gate_pair_coverage(capture)["passed"]
    # the historical bug grouped by source index: with the fix there are
    # exactly sum(choose2) pair rows, not one group per source value
    assert capture["pair_states"].shape[0] == 6 + 15


# ---------------------------------------------------------------------------
# Test 3 -- every pair row consumed exactly once
# ---------------------------------------------------------------------------

def test_pair_coverage_exactly_once():
    export = _synthetic_export([2, 5, 7])
    assert frs._gate_pair_coverage(export)["passed"]
    dropped = _synthetic_export([2, 5, 7], corrupt_coverage=True)
    assert not frs._gate_pair_coverage(dropped)["passed"]


# ---------------------------------------------------------------------------
# Test 4 / 5 / 6 -- pre-head R hook, reconstruction, prediction
# ---------------------------------------------------------------------------

def test_pre_head_R_hook_shape_and_position():
    model = _tiny_model()
    model.eval()
    graph = _dummy_graph()
    capture = frs._capture_states(model, [graph], batch_size=1)
    R = capture["R"]
    # true pre-head R has the unified graph width (302 for the frozen config)
    assert R.shape == (1, model.unified_graph_width)
    assert R.shape[1] == frs.R_DIM
    # it is NOT the 64D output of head[0] (the historical bug)
    assert R.shape[1] != model.head[0].out_features
    # and it really is the input of head[0]
    head0_out = {}
    handle = model.head[0].register_forward_hook(lambda _m, _i, o: head0_out.__setitem__("v", o.detach()))
    with torch.no_grad():
        model(graph)
    handle.remove()
    with torch.no_grad():
        recomputed = model.head[0](torch.tensor(R))
    assert torch.allclose(recomputed, head0_out["v"], atol=1e-5)


def test_reconstruct_R_and_prediction():
    model = _tiny_model()
    model.eval()
    graph = _dummy_graph()
    capture = frs._capture_states(model, [graph], batch_size=1)
    capture["subset_index"] = np.arange(1)
    reconstructed = frs._reconstruct_R(capture)
    assert reconstructed.shape == (1, frs.R_DIM)
    assert np.abs(reconstructed - capture["R"]).max() < 1e-4
    gate = frs._gate_reconstruct_R(capture)
    assert gate["passed"], gate
    assert frs._gate_reconstruct_prediction(capture, model)["passed"]


# ---------------------------------------------------------------------------
# Test 7 -- batch size / graph order invariance
# ---------------------------------------------------------------------------

def test_batch_and_order_invariance():
    model = _tiny_model()
    model.eval()
    graphs = [_dummy_graph(n=5, seed=i) for i in range(4)]
    baseline = frs._capture_states(model, graphs, batch_size=2)

    order = np.asarray([3, 1, 0, 2])
    reordered = frs._capture_states(model, [graphs[i] for i in order], batch_size=1)
    inverse = np.argsort(order)
    assert np.abs(baseline["R"] - reordered["R"][inverse]).max() < 1e-5
    assert np.abs(baseline["yhat_0"] - reordered["yhat_0"][inverse]).max() < 1e-5
    patch_offsets = np.concatenate([[0], np.cumsum(baseline["n_patches"])])
    for position in range(len(graphs)):
        p0, p1 = int(patch_offsets[position]), int(patch_offsets[position + 1])
        assert baseline["patch_states"][p0:p1].shape[0] == baseline["n_patches"][position]


# ---------------------------------------------------------------------------
# Test 8 -- node (patch) permutation invariance
# ---------------------------------------------------------------------------

def test_patch_permutation_invariance():
    model = _tiny_model()
    model.eval()
    graph = _dummy_graph(n=7, seed=3)
    idx = torch.tensor(np.random.RandomState(0).permutation(7))
    # new row p holds old row idx[p]; old vertex a moves to position
    # inv[a] = argsort(idx)[a], and pair endpoints must be remapped the
    # same way (a consistent relabelling, not a different graph).
    inv = torch.argsort(idx)
    permuted = zpp.Data(
        patch_cont=graph.patch_cont[idx],
        patch_context=graph.patch_context,
        typed_token=graph.typed_token[idx],
        parent_token=graph.parent_token[idx],
        structural_token=graph.structural_token[idx],
        structural_coarse=graph.structural_coarse[idx],
        pair_index=inv[graph.pair_index],
        pair_relation=graph.pair_relation,
        pair_bucket=graph.pair_bucket,
        global_context=graph.global_context,
        batch=graph.batch,
        y=graph.y,
        num_nodes=graph.num_nodes,
        topology_features=graph.topology_features,
    )
    base = frs._capture_states(model, [graph], batch_size=1)
    other = frs._capture_states(model, [permuted], batch_size=1)
    assert np.abs(base["R"] - other["R"]).max() < 1e-4
    assert np.abs(base["yhat_0"] - other["yhat_0"]).max() < 1e-5


# ---------------------------------------------------------------------------
# Test 9 -- legacy cache cannot be loaded as a corrected export
# ---------------------------------------------------------------------------

def test_legacy_cache_rejected(tmp_path: Path):
    fingerprint = {
        "export_version": "head_input_v1",
        "tokenizer_version": frs.TOKENIZER_VERSION,
    }
    path = tmp_path / "legacy.npz"
    np.savez_compressed(
        path,
        fingerprint_json=np.asarray(json.dumps(fingerprint)),
        R=np.zeros((1, frs.R_DIM), dtype=np.float32),
    )
    with pytest.raises(RuntimeError):
        frs.load_export(path)

    # and a corrected export round-trips
    corrected = tmp_path / "corrected.npz"
    np.savez_compressed(
        corrected,
        fingerprint_json=np.asarray(json.dumps({"export_version": frs.EXPORT_VERSION})),
        R=np.zeros((1, frs.R_DIM), dtype=np.float32),
    )
    assert frs.load_export(corrected)["fingerprint"]["export_version"] == frs.EXPORT_VERSION


# ---------------------------------------------------------------------------
# Test 10 -- the set adapter really uses the set summary
# ---------------------------------------------------------------------------

def test_set_adapter_depends_on_summary():
    torch.manual_seed(0)
    adapter = frs.SetAdapter()
    fit = frs.FoldStates(_export_like()).tensors(np.arange(3))
    r_mean, r_scale = frs._standardizer(fit["R"].numpy())
    adapter.set_standardizers(
        (r_mean, r_scale),
        frs._standardizer(fit["patch_states"].numpy()),
        frs._standardizer(fit["pair_states"].numpy()),
    )
    with torch.no_grad():
        with_summary = adapter(fit)
        without = adapter(fit, use_summary=False)
    assert (with_summary - without).abs().max() > 0
    # nonzero gradient flows into the shared set encoders
    loss = adapter(fit).sum()
    loss.backward()
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.phi_h.parameters())
    assert any(p.grad is not None and p.grad.abs().sum() > 0 for p in adapter.phi_q.parameters())


def test_adapter_parameter_match():
    ronly = frs._count_parameters(frs._make_ronly(0))
    set_adapter = frs._count_parameters(frs._make_set(0))
    assert abs(ronly - set_adapter) / ronly < 0.01


# ---------------------------------------------------------------------------
# split determinism
# ---------------------------------------------------------------------------

def test_hash_split_is_deterministic_and_disjoint():
    ids = [f"train:{i:04d}" for i in range(2000)]
    roles_a = frs._hash_split(ids)
    roles_b = frs._hash_split(ids)
    assert np.array_equal(roles_a, roles_b)
    counts = {label: int((roles_a == label).sum()) for label in set(roles_a.tolist())}
    assert counts == {"adapter_fit": 1200, "adapter_selection": 400, "adapter_evaluation": 400}


# ---------------------------------------------------------------------------
# synthetic FoldStates fixture
# ---------------------------------------------------------------------------

def _export_like(n: int = 3, patches=(3, 4, 5)):
    total_pairs = sum(p * (p - 1) // 2 for p in patches)
    patch_states = np.random.RandomState(0).randn(sum(patches), frs.PATCH_DIM).astype(np.float32)
    pair_states = np.random.RandomState(1).randn(total_pairs, frs.PAIR_DIM).astype(np.float32)
    pair_bucket = np.random.RandomState(2).randint(0, frs.N_BUCKETS, total_pairs).astype(np.int64)
    return {
        "subset_index": np.arange(n, dtype=np.int64),
        "target": np.zeros(n, dtype=np.float64),
        "yhat_0": np.zeros(n, dtype=np.float64),
        "n_patches": np.asarray(patches, dtype=np.int64),
        "n_pairs": np.asarray([p * (p - 1) // 2 for p in patches], dtype=np.int64),
        "R": np.zeros((n, frs.R_DIM), dtype=np.float32),
        "patch_states": patch_states,
        "pair_states": pair_states,
        "pair_bucket": pair_bucket,
        "pair_source": np.zeros(total_pairs, dtype=np.int64),
        "pair_target": np.zeros(total_pairs, dtype=np.int64),
        "pair_source_local": np.zeros(total_pairs, dtype=np.int64),
        "pair_target_local": np.zeros(total_pairs, dtype=np.int64),
        "pair_source_graph": np.zeros(total_pairs, dtype=np.int64),
        "pair_target_graph": np.zeros(total_pairs, dtype=np.int64),
        "global_hidden": np.zeros((n, 32), dtype=np.float32),
        "topology_hidden": np.zeros((n, frs.R_DIM - 262 - 32), dtype=np.float32),
    }

"""Tests for the compact-v4 small-head end-to-end compression experiment.

Static / unit tests for the pre-registered Stage-0 integrity gates.  They never
train a model, never evaluate the official valid split as a decision, and never
load official test.  Tests 1-14 are the mandatory set; test 15 runs because
seed1 was actually purchased (seed0 was non-inferior).
"""

from __future__ import annotations

import inspect

import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.code.graph import from_edges, ring_chords
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as sh
from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
from tracks.ksvd.experiments.luyin16.zinc_graph_head_function_family import (
    GenericReader,
)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _synthetic_batch():
    topo_width = int(ztopo.raw_width("hinge"))
    datasets: list[Data] = []
    for graph in (ring_chords(6, []), ring_chords(5, []), from_edges(6, [(0, 1), (1, 2), (2, 0)])):
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


@pytest.fixture(scope="module")
def baseline():
    model = sh.build_baseline(0)
    model.eval()
    return model


@pytest.fixture(scope="module")
def smallhead():
    model = sh.build_smallhead(0)
    model.eval()
    return model


# ---------------------------------------------------------------------------
# Test 1-4: parameter accounting
# ---------------------------------------------------------------------------


def test_01_baseline_total_params(baseline):
    """Test 1 / G0.1: baseline total params == 99,613."""
    assert sh._n_params(baseline) == 99613 == sh.EXPECTED_BASELINE_TOTAL


def test_02_baseline_head_params(baseline):
    """Test 2 / G0.2: baseline graph head == 21,633."""
    assert sh._n_params(baseline.head) == 21633 == sh.EXPECTED_BASELINE_HEAD


def test_03_small_head_params(smallhead):
    """Test 3 / G0.3: small head == 4,135."""
    assert sh._n_params(smallhead.head) == 4135 == sh.EXPECTED_SMALL_HEAD


def test_04_smallhead_total_params(smallhead, baseline):
    """Test 4 / G0.4: candidate total == 82,115 and mechanical account holds."""
    total = sh._n_params(smallhead)
    assert total == 82115 == sh.EXPECTED_SMALL_TOTAL
    assert total == sh._n_params(baseline) - sh._n_params(baseline.head) + sh._n_params(
        smallhead.head
    )
    assert sh._n_params(baseline.head) - sh._n_params(smallhead.head) == 17498


# ---------------------------------------------------------------------------
# Test 5-6: architecture identity
# ---------------------------------------------------------------------------


def test_05_non_head_architecture_identical(baseline, smallhead):
    """Test 5 / G0.6: every non-head trainable tensor is identical."""
    base = {
        name: (tuple(p.shape), str(p.dtype))
        for name, p in baseline.named_parameters()
        if not name.startswith("head.")
    }
    small = {
        name: (tuple(p.shape), str(p.dtype))
        for name, p in smallhead.named_parameters()
        if not name.startswith("head.")
    }
    assert base == small
    assert len(base) == 40


def test_06_R_dimension(baseline, smallhead):
    """Test 6 / G0.5: pre-head R is still 302D for both models."""
    assert int(baseline.unified_graph_width) == 302
    assert int(smallhead.unified_graph_width) == 302
    assert int(smallhead.head.net[0].in_features) == 302


# ---------------------------------------------------------------------------
# Test 7-8: initialization matching and representation identity
# ---------------------------------------------------------------------------


def test_07_seed0_shared_init_exact(baseline, smallhead):
    """Test 7 / G0.7: seed0 shared initial tensors are exactly baseline."""
    baseline_state = baseline.state_dict()
    small_state = smallhead.state_dict()
    shared = [
        key
        for key in baseline_state
        if key in small_state and small_state[key].shape == baseline_state[key].shape
    ]
    assert len(shared) == 40
    max_abs = max(
        float((small_state[key] - baseline_state[key]).abs().max()) for key in shared
    )
    assert max_abs == 0.0
    assert (
        sh._state_hash({k: small_state[k] for k in shared})
        == sh._state_hash({k: baseline_state[k] for k in shared})
    )


def test_08_pre_head_R_exact(baseline, smallhead):
    """Test 8 / G0.8: pre-head R is bit-identical at initialization."""
    batch = _synthetic_batch()
    with torch.no_grad():
        r_base = baseline.encode(batch)
        r_small = smallhead.encode(batch)
    assert r_base.shape[1] == r_small.shape[1] == 302
    assert float((r_base - r_small).abs().max()) == 0.0


# ---------------------------------------------------------------------------
# Test 9-10: small-head definition is the frozen historical Sraw
# ---------------------------------------------------------------------------


def test_09_small_head_reuses_locked_sraw(smallhead):
    """Test 9: the small head is the exact historical Sraw / GenericReader."""
    reference = GenericReader(302, (13, 13))
    assert isinstance(smallhead.head, GenericReader)
    assert tuple(smallhead.small_head_hidden) == (13, 13)
    ref_spec = [(type(m).__name__, getattr(m, "in_features", None), getattr(m, "out_features", None)) for m in reference.net]
    got_spec = [(type(m).__name__, getattr(m, "in_features", None), getattr(m, "out_features", None)) for m in smallhead.head.net]
    assert got_spec == ref_spec
    assert sh._n_params(smallhead.head) == sh._n_params(reference) == 4135


def test_10_no_layernorm_or_dropout_in_small_head(smallhead):
    """Test 10: no LayerNorm / dropout was smuggled into the raw small head."""
    kinds = {type(module).__name__ for module in smallhead.head.modules()}
    assert "LayerNorm" not in kinds
    assert "Dropout" not in kinds
    activations = [type(module).__name__ for module in smallhead.head.net if not isinstance(module, torch.nn.Linear)]
    assert activations == ["ReLU", "ReLU"]
    assert sh.SMALL_HEAD_ACTIVATION == "ReLU"
    assert sh.SMALL_HEAD_INPUT.startswith("raw R")


# ---------------------------------------------------------------------------
# Test 11-14: protocol / budget / test-lock
# ---------------------------------------------------------------------------


def test_11_training_protocol_matches_optimized_baseline():
    """Test 11 / G0.9: the frozen optimized protocol is inherited verbatim."""
    assert sh.OPTIMIZED_PROTOCOL["optimizer"] == "Adam"
    assert sh.OPTIMIZED_PROTOCOL["learning_rate"] == 1.0e-3
    assert sh.OPTIMIZED_PROTOCOL["weight_decay"] == 1.0e-5
    assert sh.OPTIMIZED_PROTOCOL["batch_size"] == 128
    assert sh.OPTIMIZED_PROTOCOL["max_epochs"] == 240
    assert sh.OPTIMIZED_PROTOCOL["patience"] == 40
    assert sh.OPTIMIZED_PROTOCOL["scheduler"] == "none"
    assert sh.OPTIMIZED_PROTOCOL["gradient_clip_norm"] == 5.0
    assert sh.OPTIMIZED_PROTOCOL["checkpoint_selection"] == "best official-valid MAE"
    assert sh.OPTIMIZED_PROTOCOL["train_shuffle_seed_offset"] == 91011
    assert sh.OPTIMIZED_PROTOCOL["eval_shuffle_seed_offset"] == 91012


def test_12_no_warm_start_or_trained_backbone_load():
    """Test 12: build_smallhead never loads a trained checkpoint."""
    source = inspect.getsource(sh.build_smallhead)
    assert "torch.load" not in source
    assert "load_state_dict" not in source
    assert "distill" not in source.lower()
    # construction explicitly copies from a freshly initialized baseline
    assert "build_baseline" in source
    assert "copy_" in source


def test_13_valid_used_only_for_selection():
    """Test 13: the training loop selects the best official-valid checkpoint."""
    source = inspect.getsource(sh.train_model)
    assert "valid_mae < best_mae" in source
    assert "copy.deepcopy(model.state_dict())" in source
    assert "best_state" in source
    # official-valid MAE is the checkpoint-selection metric, not a training target
    assert "mean_absolute_error" not in inspect.getsource(sh.train_model)
    assert sh.V4_SEED0_VALID == 0.14642022556537995
    assert sh.V4_SEED1_VALID == 0.1493322635096847


def test_14_official_test_never_loaded():
    """Test 14 / G0.10: the experiment source never loads official test."""
    source = inspect.getsource(sh)
    pattern = "_load_zinc(ZINC_ROOT, " + '"test")'
    assert source.count(pattern) == 0
    assert sh.base_config()["test_policy"] == "no_test"


# ---------------------------------------------------------------------------
# Test 15: seed1 initialization matching (seed1 was purchased)
# ---------------------------------------------------------------------------


def test_15_seed1_shared_init_exact():
    """Test 15: seed1 shared init exact-matches canonical baseline seed1."""
    report = sh.initialization_match_seed1()
    assert report["seed"] == 1
    assert report["n_shared_tensors"] == 40
    assert report["max_abs_diff"] == 0.0
    assert report["exact_equal_to_baseline_init"] is True
    assert report["hash_match"] is True
    # seed1 must not reuse seed0 initialization
    seed0 = sh.initialization_match_seed0()
    assert report["shared_state_sha256"] != seed0["shared_state_sha256"]

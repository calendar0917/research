"""Tests for the optimized-manifold broad frozen-state sufficiency screen.

These tests pin the *protocol discipline* and the *measurement repairs* of the
screen without training any backbone and without loading official test:

1.  the checkpoints come from the frozen optimized 240/40 protocol;
2.  seed0 / seed1 checkpoint fingerprints are computed independently & correctly;
3.  the true pre-head representation R is 302D;
4.  R fed through the frozen original head reconstructs yhat_0;
5.  pair states q_ij are grouped by graph membership (not source index);
6.  updated patch states h'_i are grouped by graph membership;
7.  the set/state summary is invariant to patch/pair row ordering;
8.  the B1 / E reader parameter mismatch is <= 2%;
9.  the 7200/800/2000 adapter split is deterministic and disjoint;
10. official valid is never used to train / select an adapter;
11. official test is never loaded;
12. the state reader genuinely depends on the injected frozen states.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import numpy as np
import pytest
import torch

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_frozen_readout_sufficiency as frs
from tracks.ksvd.experiments.luyin16 import (
    zinc_optimized_manifold_broad_state_screen as m,
)


# ---------------------------------------------------------------------------
# synthetic fixtures (no real dataset / checkpoint needed)
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
        patch_cont=torch.randn(
            n, zpp._shell_width_for_radius(zpp.PATCH_RADIUS), generator=generator
        ),
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


def _synthetic_tensors(n_graphs: int = 3, patches=(3, 4, 5)):
    total_patches = sum(patches)
    total_pairs = sum(p * (p - 1) // 2 for p in patches)
    patch_states = np.random.RandomState(0).randn(total_patches, frs.PATCH_DIM).astype(np.float32)
    pair_states = np.random.RandomState(1).randn(total_pairs, frs.PAIR_DIM).astype(np.float32)
    pair_bucket = np.random.RandomState(2).randint(0, frs.N_BUCKETS, total_pairs).astype(np.int64)
    patch_graph, pair_graph = [], []
    for graph, size in enumerate(patches):
        patch_graph.extend([graph] * size)
        pair_graph.extend([graph] * (size * (size - 1) // 2))
    return {
        "R": torch.zeros(n_graphs, frs.R_DIM),
        "y": torch.zeros(n_graphs),
        "yhat_0": torch.zeros(n_graphs),
        "patch_states": torch.tensor(patch_states),
        "patch_graph": torch.tensor(patch_graph, dtype=torch.long),
        "pair_states": torch.tensor(pair_states),
        "pair_graph": torch.tensor(pair_graph, dtype=torch.long),
        "pair_bucket": torch.tensor(pair_bucket, dtype=torch.long),
        "n": n_graphs,
    }


# ---------------------------------------------------------------------------
# Test 1 -- checkpoints come from the optimized 240/40 protocol
# ---------------------------------------------------------------------------

def test_optimized_protocol_is_240_40():
    protocol = m.OPTIMIZED_PROTOCOL
    assert protocol["max_epochs"] == 240
    assert protocol["patience"] == 40
    assert protocol["batch_size"] == 128
    assert protocol["learning_rate"] == 1.0e-3
    assert protocol["weight_decay"] == 1.0e-5
    assert protocol["scheduler"] == "none"
    assert protocol["loss"].startswith("L1")
    assert protocol["single_stage"] is True
    assert m.BACKBONE_SEEDS == (0, 1)
    assert m.EXPECTED_PARAMS == 99613
    # the module must never call the backbone training phase
    source = inspect.getsource(m)
    assert "_train_phase" not in source


def test_optimized_run_records_match_protocol():
    runs = [
        m.TRACK_ROOT / "results/compact_v4_training_sufficiency/runs/Pstar_A2_long_seed0.json",
        m.TRACK_ROOT / "results/compact_v4_training_sufficiency/runs/Pstar_A2_long_seed1.json",
    ]
    if not all(p.exists() for p in runs):
        pytest.skip("optimized run records not present on this clone")
    for seed, path in enumerate(runs):
        run = json.loads(path.read_text())
        assert run["best_valid_mae"] == pytest.approx(m.EXPECTED_VALID[seed], abs=1e-9)
        assert run["best_epoch"] == m.EXPECTED_BEST_EPOCH[seed]
        assert run["parameters"] == m.EXPECTED_PARAMS


# ---------------------------------------------------------------------------
# Test 2 -- independent seed fingerprints
# ---------------------------------------------------------------------------

def test_checkpoint_fingerprints_are_independent_and_correct():
    fp0 = m._checkpoint_fingerprint(0)
    fp1 = m._checkpoint_fingerprint(1)
    if not Path(fp0["path"]).exists():
        pytest.skip("optimized checkpoint not present on this clone")
    assert fp0["sha256"] != fp1["sha256"]
    # recompute the SHA independently
    import hashlib

    for fp in (fp0, fp1):
        digest = hashlib.sha256(Path(fp["path"]).read_bytes()).hexdigest()
        assert fp["sha256"] == digest
    assert fp0["sha256"] == "60b7d297a44befb7328f4f9da0cb379e308ecec3f5eab17fa7c199896ac3e71b"
    assert fp1["sha256"] == "93bf4ec231469793c20da829551cf1c95693513ddf58be7d92e28501534db8a3"


def test_config_sha_is_pinned():
    assert m.CONFIG_SHA256 == m._sha256_file(m.CONFIG_PATH)


# ---------------------------------------------------------------------------
# Test 3 / 4 -- R dimension and R -> head reconstruction
# ---------------------------------------------------------------------------

def test_true_pre_head_R_is_302d():
    model = _tiny_model()
    model.eval()
    capture = frs._capture_states(model, [_dummy_graph()], batch_size=1)
    assert capture["R"].shape == (1, frs.R_DIM) == (1, 302)
    assert model.unified_graph_width == 302
    # 302 = unary(2*48+1) + pair(5*(2*16+1)) + global(32) + topology(8)
    assert 2 * frs.PATCH_DIM + 1 + frs.N_BUCKETS * (2 * frs.PAIR_DIM + 1) + 32 + 8 == 302


def test_R_through_frozen_head_reconstructs_prediction():
    model = _tiny_model()
    model.eval()
    graph = _dummy_graph()
    capture = frs._capture_states(model, [graph], batch_size=1)
    capture["subset_index"] = np.arange(1)
    assert frs._gate_reconstruct_R(capture)["passed"]
    assert frs._gate_reconstruct_prediction(capture, model)["passed"]


# ---------------------------------------------------------------------------
# Test 5 / 6 -- pair / patch graph grouping
# ---------------------------------------------------------------------------

def test_pair_and_patch_graph_grouping_is_graph_membership():
    model = _tiny_model()
    model.eval()
    graphs = [_dummy_graph(n=4, seed=1), _dummy_graph(n=6, seed=2)]
    capture = frs._capture_states(model, graphs, batch_size=2)
    capture["subset_index"] = np.arange(2)
    # q_ij grouped by graph membership, not by source patch index
    assert np.array_equal(capture["pair_source_graph"], capture["pair_target_graph"])
    assert frs._gate_pair_grouping(capture)["passed"]
    assert frs._gate_pair_coverage(capture)["passed"]
    assert capture["pair_states"].shape[0] == 6 + 15
    # h'_i (patch states) grouped by graph membership
    assert int(capture["n_patches"].sum()) == len(capture["patch_states"]) == 10


# ---------------------------------------------------------------------------
# Test 7 -- set summary ordering invariance
# ---------------------------------------------------------------------------

def test_set_summary_is_order_invariant():
    torch.manual_seed(0)
    adapter = frs.SetAdapter()
    tensors = _synthetic_tensors()
    with torch.no_grad():
        base = adapter.summarize(tensors)
        # permute patch rows within each graph and pair rows
        perm = torch.randperm(tensors["patch_states"].shape[0])
        shuffled = dict(tensors)
        shuffled["patch_states"] = tensors["patch_states"][perm]
        shuffled["patch_graph"] = tensors["patch_graph"][perm]
        pp = torch.randperm(tensors["pair_states"].shape[0])
        shuffled["pair_states"] = tensors["pair_states"][pp]
        shuffled["pair_graph"] = tensors["pair_graph"][pp]
        shuffled["pair_bucket"] = tensors["pair_bucket"][pp]
        other = adapter.summarize(shuffled)
    assert torch.allclose(base, other, atol=1e-5)


# ---------------------------------------------------------------------------
# Test 8 -- parameter match
# ---------------------------------------------------------------------------

def test_reader_parameter_match_within_2pct():
    ronly = frs._count_parameters(frs._make_ronly(m.PRIMARY_ADAPTER_SEED))
    state = frs._count_parameters(frs._make_set(m.PRIMARY_ADAPTER_SEED))
    assert ronly == 4135
    assert state == 4145
    assert abs(ronly - state) / ronly <= 0.02
    lock = m.adapter_protocol_lock()
    assert lock["parameter_mismatch_within_2pct"] is True


# ---------------------------------------------------------------------------
# Test 9 -- deterministic disjoint split
# ---------------------------------------------------------------------------

def test_adapter_split_is_deterministic_and_disjoint():
    roles_a = m._role_assignment()
    roles_b = m._role_assignment()
    assert np.array_equal(roles_a, roles_b)
    counts = {label: int((roles_a == label).sum()) for label in m.SPLIT_ROLES}
    assert counts == {"adapter_fit": 7200, "adapter_selection": 800, "train_probe": 2000}
    # only official train indices appear (0..9999)
    assert len(roles_a) == m.N_TRAIN


# ---------------------------------------------------------------------------
# Test 10 -- official valid is not used for adapter training / selection
# ---------------------------------------------------------------------------

def test_valid_not_used_for_adapter_training_or_selection():
    source = inspect.getsource(m._run_seed)
    assert "train_states.tensors(fit_positions)" in source
    assert "train_states.tensors(sel_positions)" in source
    assert "valid_states.tensors" in source
    # adapters are trained only on fit/selection (never on valid)
    assert "_train_adapter(ronly, fit, selection" in source
    assert "_train_adapter(state, fit, selection" in source
    assert "_train_adapter(state, valid" not in source
    assert "_train_adapter(ronly, valid" not in source


# ---------------------------------------------------------------------------
# Test 11 -- official test is never loaded
# ---------------------------------------------------------------------------

def test_official_test_never_loaded():
    source = inspect.getsource(m)
    assert '"test"' not in source
    assert "'test'" not in source
    assert "extract_test" not in source
    # every direct dataset load uses the train or valid split only
    import re

    for match in re.finditer(r'_load_zinc\(ZINC_ROOT, "(\w+)"\)', source):
        assert match.group(1) in {"train", "val"}
    # the shared record extractor only touches train + valid
    assert "_extract_v4_records" in source


# ---------------------------------------------------------------------------
# Test 12 -- the state reader depends on the injected frozen states
# ---------------------------------------------------------------------------

def test_state_reader_depends_on_frozen_states():
    torch.manual_seed(0)
    adapter = frs.SetAdapter()
    adapter.set_standardizers(
        (np.zeros(frs.R_DIM, dtype=np.float32), np.ones(frs.R_DIM, dtype=np.float32)),
        (np.zeros(frs.PATCH_DIM, dtype=np.float32), np.ones(frs.PATCH_DIM, dtype=np.float32)),
        (np.zeros(frs.PAIR_DIM, dtype=np.float32), np.ones(frs.PAIR_DIM, dtype=np.float32)),
    )
    dependency = m._state_dependency(adapter, _synthetic_tensors(), adapter_seed=0)
    assert dependency["summary_off_max_abs_diff"] > 0.0
    assert dependency["state_zero_max_abs_diff"] > 0.0
    assert dependency["state_permuted_max_abs_diff"] > 0.0

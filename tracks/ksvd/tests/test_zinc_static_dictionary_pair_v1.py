"""Targeted tests for the strict-static dictionary-pair v1 confirmation round.

Static / CPU-only.  These tests never load ZINC data, checkpoints, official
valid, or official test.  They verify:

* the Phase-A common-horizon audit helpers and verdict logic;
* the seed-1 pre-registered interpretation cases B1-B4;
* the strict-static architecture contract for all three arms at seed 1;
* bit-identical shared initialisation and parameter matching at seed 1;
* the forced-full-horizon + shadow early-stopping separation, using a scripted
  validation curve so the two protocol views are checked deterministically.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch
from torch_geometric.data import Data

from tracks.ksvd.experiments.luyin16 import zinc_patch_path_pooling as zpp
from tracks.ksvd.experiments.luyin16 import zinc_topology_features as ztopo
from tracks.ksvd.experiments.luyin16 import zinc_static_dictionary_pair as sdp
from tracks.ksvd.experiments.luyin16 import (
    zinc_static_dictionary_pair_v1_confirmation as v1,
)


# ---------------------------------------------------------------------------
# synthetic structure
# ---------------------------------------------------------------------------


def _graph(n: int, seed: int) -> Data:
    generator = torch.Generator().manual_seed(int(seed))
    pairs = [(i, j) for i in range(n) for j in range(i + 1, n)]
    pair_index = (
        torch.tensor(pairs, dtype=torch.long).t().contiguous()
        if pairs
        else torch.zeros(2, 0, dtype=torch.long)
    )
    return Data(
        patch_cont=torch.randn(n, int(zpp.SHELL_WIDTH), generator=generator),
        patch_context=torch.zeros(n, 0),
        typed_token=torch.randint(0, sdp.TYPED_VOCAB_SIZE, (n,), generator=generator),
        parent_token=torch.randint(0, sdp.PARENT_VOCAB_SIZE, (n,), generator=generator),
        structural_token=torch.zeros(n, dtype=torch.long),
        structural_coarse=torch.zeros(n, 0),
        pair_index=pair_index,
        pair_relation=torch.randn(len(pairs), int(zpp.RELATION_WIDTH), generator=generator),
        pair_bucket=torch.randint(0, zpp.DISTANCE_BUCKETS, (len(pairs),), generator=generator),
        global_context=torch.randn(1, int(zpp.GLOBAL_WIDTH), generator=generator),
        topology_features=torch.randn(1, int(ztopo.raw_width("hinge")), generator=generator),
        y=torch.randn(1, generator=generator),
        num_nodes=n,
    )


@pytest.fixture(scope="module")
def batch():
    graphs = [_graph(n, seed=n) for n in (7, 6, 8)]
    return next(iter(zpp._make_loader(graphs, 8, False, 0)))


# ---------------------------------------------------------------------------
# Phase A helpers
# ---------------------------------------------------------------------------


def _curve(values):
    return [{"epoch": i + 1, "valid_mae": float(v)} for i, v in enumerate(values)]


def test_window_stats_and_running_best():
    rows = _curve([0.5, 0.4, 0.45, 0.3, 0.6])
    stats = v1._window_stats(rows, 3)
    assert stats["best_valid"] == pytest.approx(0.4)
    assert stats["best_epoch"] == 2
    assert stats["mean_best5_valid"] == pytest.approx((0.4 + 0.45 + 0.5) / 3)
    running = v1._running_best(rows)
    assert running == pytest.approx([0.5, 0.4, 0.4, 0.3, 0.3])


def test_common_horizon_verdicts():
    # A1: both gates fire.
    assert (
        v1._classify_common_horizon(0.150, 0.150, 0.140)
        == "COMMON_HORIZON_SIGNAL_SURVIVES"
    )
    # A2: dict still beats both, but below the double gate.
    assert (
        v1._classify_common_horizon(0.145, 0.145, 0.144)
        == "COMMON_HORIZON_DIRECTION_ONLY"
    )
    # A3: dict does not beat S0.
    assert (
        v1._classify_common_horizon(0.140, 0.150, 0.145)
        == "COMMON_HORIZON_SIGNAL_FRAGILE"
    )


def test_restricted_soup_is_reported_unavailable_not_faked():
    saved = v1._saved_epoch_states_available()
    assert saved["restricted_top5_soup_available"] is False
    assert saved["per_epoch_states_saved"] is False


# ---------------------------------------------------------------------------
# seed-1 interpretation cases
# ---------------------------------------------------------------------------


def test_seed1_cases():
    # B1 strong replication: both gates + soup same direction.
    b1 = v1._classify_seed1(0.150, 0.150, 0.140, 0.145, 0.145, 0.138)
    assert b1["case"] == "B1_SEED1_DICT_SPECIFIC_REPLICATION"
    # B2 directional only.
    b2 = v1._classify_seed1(0.145, 0.145, 0.1445, 0.145, 0.145, 0.1447)
    assert b2["case"] == "B2_SEED1_DIRECTIONAL_REPLICATION"
    # B3 dict beats S0 but not Dense.
    b3 = v1._classify_seed1(0.146, 0.140, 0.145, 0.146, 0.140, 0.145)
    assert b3["case"] == "B3_SEED1_GENERIC_CAPACITY_OR_AMBIGUOUS"
    # B4 no replication.
    b4 = v1._classify_seed1(0.140, 0.141, 0.145, 0.140, 0.141, 0.145)
    assert b4["case"] == "B4_SEED1_NO_REPLICATION"


# ---------------------------------------------------------------------------
# strict-static contract at seed 1
# ---------------------------------------------------------------------------


def test_contract_passes_for_every_arm_at_seed1(batch):
    for arm in ("s0", "dense", "dict"):
        model = sdp.ARMS[arm](seed=1)
        report = sdp.static_contract_checks(model, batch)
        assert report["passed"], (arm, report)
        assert report["center_context_false"] and report["center_update_is_none"], arm
        assert report["pair_encoder_calls_per_forward"] == 1, arm
        assert report["relation_encoder_calls_per_forward"] == 1, arm
        assert report["h_identical_under_relation_mutation"], arm
        assert report["max_abs_h_diff"] == 0.0, arm
        assert report["prediction_changes_under_relation_mutation"], arm


def test_seed1_shared_tensors_bit_identical():
    report = v1.initialization_match(seed=1)
    assert report["bit_identical"], report
    assert report["max_abs_diff_dense_vs_dict"] == 0.0
    assert report["max_abs_diff_s0_vs_dict"] == 0.0
    assert report["n_shared_tensors"] > 20


def test_seed1_parameter_match():
    audit = v1.parameter_audit(seed=1)
    assert audit["center_context_false"] and audit["center_update_is_none"]
    assert audit["unified_graph_width"] == 302
    assert audit["parameter_match"]["dict_branch_params"] == 5201
    assert audit["parameter_match"]["dense_branch_params"] == 5189
    assert audit["parameter_match"]["within_one_percent"]


# ---------------------------------------------------------------------------
# forced full horizon + shadow tracker (deterministic scripted valid curve)
# ---------------------------------------------------------------------------


def test_forced_full_horizon_with_shadow_early_stopping(monkeypatch, tmp_path):
    scripted = [0.50, 0.40, 0.41, 0.39, 0.45]
    calls = {"n": 0}

    def _fake_evaluate(_model, _loader, _device):
        index = calls["n"]
        calls["n"] += 1
        value = scripted[index] if index < len(scripted) else float(scripted[-1])
        return float(value), np.zeros(8), np.zeros(8)

    train = [_graph(n, seed=100 + n) for n in (5, 6, 7)]
    valid = [_graph(n, seed=200 + n) for n in (5, 6)]

    monkeypatch.setattr(sdp, "_evaluate_mae", _fake_evaluate)
    monkeypatch.setattr(sdp, "load_encoded", lambda *_a, **_k: (train, valid, {
        "typed_vocabulary_size_with_oov": sdp.TYPED_VOCAB_SIZE,
        "parent_vocabulary_size_with_oov": sdp.PARENT_VOCAB_SIZE,
    }))
    monkeypatch.setattr(v1, "_write_json", lambda *_a, **_k: None)
    monkeypatch.setattr(sdp, "_write_csv", lambda *_a, **_k: None)
    monkeypatch.setattr(v1, "RESULTS_DIR", tmp_path / "results")
    monkeypatch.setattr(v1, "CURVE_DIR", tmp_path / "curves")
    monkeypatch.setattr(v1, "RUNS_DIR", tmp_path / "runs")
    monkeypatch.setattr(v1, "STATE_DIR", tmp_path / "states")

    summary = v1.train_arm_v1(
        "s0", seed=999, device="cpu", max_epochs=5, patience=1, tag="testshadow", save_state=False
    )

    # Optimizer always runs the full horizon, even though the shadow protocol
    # would have stopped at epoch 3.
    assert summary["epochs_run"] == 5
    assert summary["early_stopped"] is False
    assert summary["forced_full_horizon"] is True

    eq = summary["equal_horizon"]
    assert eq["best_epoch"] == 4  # global min over 1..5
    assert eq["best_valid_mae"] == pytest.approx(0.39)
    assert sorted(eq["soup"]["members"]) == [1, 2, 3, 4, 5]

    shadow = summary["shadow"]
    assert shadow["stop_epoch"] == 3  # first epoch with stale >= patience(1)
    assert shadow["best_epoch"] == 2
    assert shadow["best_valid_mae"] == pytest.approx(0.40)
    assert max(shadow["soup"]["members"]) <= 3
    assert shadow["soup"]["available"] is True

    # The shadow prefix window cannot be better than the full-horizon window.
    assert shadow["best_valid_mae"] >= eq["best_valid_mae"]

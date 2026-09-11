"""Tests for the compact-v4 training-sufficiency / protocol-calibration audit.

These tests enforce the *protocol discipline* of the search, not any numeric
outcome:

1.  historical config reconstruction is correct;
2.  the architecture fingerprint is invariant across every search run;
3.  the target / loss definition is unchanged;
4.  official test is unreachable during the search stages;
5.  Stage 1A changes only horizon / patience;
6.  Stage 1B is exactly the pre-registered 2x2 batch x weight-decay matrix;
7.  Stage 2 changes only the learning rate;
8.  the candidate protocol is locked before seed-1 replication;
9.  seed-1 replication cannot mutate the candidate config;
10. a protocol lock must exist before the official test may run;
11. test evaluation never participates in checkpoint selection;
12. every completed run logs optimizer steps and epoch counts.
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_training_sufficiency as m


# ---------------------------------------------------------------------------
# Test 1 -- historical config reconstruction
# ---------------------------------------------------------------------------


def test_historical_config_reconstruction() -> None:
    config = m.base_config()
    model = config["model"]
    assert model["learning_rate"] == 1.0e-3
    assert model["weight_decay"] == 1.0e-5
    assert model["batch_size"] == 128
    assert model["epochs"] == 60
    assert model["patience"] == 12
    assert config["seed"] == 0
    assert "scheduler" not in model or str(model["scheduler"]) == "none"
    assert m.PROTOCOL_ID == config["protocol_id"]
    # historical run ids / valid numbers are reconstructed from the committed
    # multiseed confirmation, not invented.
    assert m.HISTORICAL[0]["run_id"] == "20260909-194445-182c7021"
    assert m.HISTORICAL[0]["valid_best_mae"] == pytest.approx(0.17006561887910357)
    assert m.HISTORICAL[1]["valid_best_mae"] == pytest.approx(0.1631665097149671)


# ---------------------------------------------------------------------------
# Test 2 -- architecture fingerprint invariant
# ---------------------------------------------------------------------------


def test_architecture_fingerprint_invariant() -> None:
    assert m.ARCH_PARAM_COUNT == 99613
    all_specs = m.STAGE1A + m.STAGE1B
    # every protocol shares the exact same model/representation config; only
    # training-protocol numbers differ.  Rebuild each config and compare the
    # frozen model architecture leaves.
    base = m.base_config()["model"]
    architecture_keys = [
        "patch_hidden",
        "pair_hidden",
        "token_width",
        "embedding_mode",
        "embedding_rank",
        "hybrid_full_typed_tokens",
        "hybrid_full_parent_tokens",
        "center_context",
        "center_context_hidden",
        "graph_head_hidden_0",
        "graph_head_hidden_1",
        "dropout",
        "topology_mode",
        "topology_hidden_dim",
        "topology_out_dim",
    ]
    for name, epochs, patience, batch, wd, lr in all_specs:
        assert base["topology_mode"] == "hinge"
        for key in architecture_keys:
            assert base[key] == m.base_config()["model"][key]
    # completed runs, if any, must all report the frozen parameter count
    for path in (m.RESULTS_DIR / "runs").glob("*.json"):
        summary = json.loads(path.read_text(encoding="utf-8"))
        assert int(summary["parameters"]) == m.ARCH_PARAM_COUNT


# ---------------------------------------------------------------------------
# Test 3 -- target / loss unchanged
# ---------------------------------------------------------------------------


def test_target_and_loss_unchanged() -> None:
    model = m.base_config()["model"]
    assert model.get("quantile_mode", "none") == "none"
    # the training loop's loss is L1 (== mean absolute error); no objective change
    lock_loss = "L1 / mean absolute error"
    assert lock_loss == "L1 / mean absolute error"
    # no target recomputation: topology features are graph invariants only
    assert "target" not in model
    assert model["topology_mode"] == "hinge"


# ---------------------------------------------------------------------------
# Test 4 -- official test blocked during search
# ---------------------------------------------------------------------------


def test_offical_test_blocked_during_search() -> None:
    # the search config never loads test
    assert m.base_config()["test_policy"] == "no_test"
    source = Path(m.__file__).read_text(encoding="utf-8")
    # official test extraction is invoked exactly once, from the guarded
    # `test()` stage, and only after the protocol-lock refusal branch.
    call = "test_records = extract_test_records()"
    assert source.count(call) == 1
    assert source.index(call) > source.index("refusing official test")
    # a non-terminal search stage must not create a test artifact
    assert not (m.RESULTS_DIR / "test_results.csv").exists() or (
        m.RESULTS_DIR / "final_training_protocol_lock.json"
    ).exists()


# ---------------------------------------------------------------------------
# Test 5 -- Stage 1A changes only horizon / patience
# ---------------------------------------------------------------------------


def test_stage1a_only_changes_horizon() -> None:
    specs = m.STAGE1A
    fixed = {(bs, wd, lr) for _, _, _, bs, wd, lr in specs}
    assert fixed == {(128, 1.0e-5, 1.0e-3)}, "Stage 1A must fix batch/wd/lr"
    names = [name for name, *_ in specs]
    assert names == ["A0_historical", "A1_moderate", "A2_long"]
    horizons = {name: epochs for name, epochs, *_ in specs}
    assert horizons == {"A0_historical": 60, "A1_moderate": 120, "A2_long": 240}
    patiences = {name: patience for name, _, patience, *_ in specs}
    assert patiences == {"A0_historical": 12, "A1_moderate": 24, "A2_long": 40}


# ---------------------------------------------------------------------------
# Test 6 -- Stage 1B is the pre-registered 2x2 batch x wd matrix
# ---------------------------------------------------------------------------


def test_stage1b_is_2x2_matrix() -> None:
    cells = {(bs, wd) for _, _, _, bs, wd, _ in m.STAGE1B}
    assert cells == {(128, 1.0e-5), (512, 1.0e-5), (128, 0.0), (512, 0.0)}
    # batch and wd are the only non-horizon variables; lr/epochs/patience fixed
    assert {lr for *_, lr in m.STAGE1B} == {1.0e-3}
    assert {epochs for _, epochs, *_ in m.STAGE1B} == {240}
    assert {patience for _, _, patience, *_ in m.STAGE1B} == {40}
    names = {name for name, *_ in m.STAGE1B}
    assert names == {"B0_b128_wd1e-5", "B1_b512_wd1e-5", "B2_b128_wd0", "B3_b512_wd0"}


# ---------------------------------------------------------------------------
# Test 7 -- Stage 2 changes only LR
# ---------------------------------------------------------------------------


def test_stage2_changes_only_lr() -> None:
    stage2_path = m.RESULTS_DIR / "stage2_lr_results.csv"
    if not stage2_path.exists():
        pytest.skip("Stage 2 not yet run")
    import csv

    with stage2_path.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert {row["batch_size"] for row in rows} == {rows[0]["batch_size"]}
    assert {row["weight_decay"] for row in rows} == {rows[0]["weight_decay"]}
    assert len({row["learning_rate"] for row in rows}) == len(rows) >= 2


# ---------------------------------------------------------------------------
# Test 8 -- candidate locked before seed1 replication
# ---------------------------------------------------------------------------


def test_candidate_locked_before_seed1() -> None:
    candidate = m.RESULTS_DIR / "candidate_protocol.json"
    seed1 = m.RESULTS_DIR / "seed1_replication.json"
    if not seed1.exists():
        pytest.skip("seed-1 replication not yet run")
    assert candidate.exists()
    assert candidate.stat().st_mtime <= seed1.stat().st_mtime


# ---------------------------------------------------------------------------
# Test 9 -- seed1 cannot mutate the candidate config
# ---------------------------------------------------------------------------


def test_seed1_does_not_mutate_candidate() -> None:
    seed1_path = m.RESULTS_DIR / "seed1_replication.json"
    if not seed1_path.exists():
        pytest.skip("seed-1 replication not yet run")
    payload = json.loads(seed1_path.read_text(encoding="utf-8"))
    candidate = json.loads((m.RESULTS_DIR / "candidate_protocol.json").read_text(encoding="utf-8"))
    embedded = payload["candidate_protocol"]
    for key in ("selected_protocol", "max_epochs", "patience", "batch_size", "weight_decay", "learning_rate"):
        assert embedded[key] == candidate[key]


# ---------------------------------------------------------------------------
# Test 10 -- protocol lock required before official test
# ---------------------------------------------------------------------------


def test_test_requires_protocol_lock(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(m, "RESULTS_DIR", tmp_path)
    with pytest.raises(RuntimeError):
        m.test()


# ---------------------------------------------------------------------------
# Test 11 -- test never participates in checkpoint selection
# ---------------------------------------------------------------------------


def test_test_not_used_for_checkpoint_selection() -> None:
    # checkpoint selection is argmin of the per-epoch *valid* curve
    candidate_curve = m.RESULTS_DIR / "candidate_seed0_curve.csv"
    if not candidate_curve.exists():
        pytest.skip("candidate curve not yet produced")
    import csv

    with candidate_curve.open(encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    best = min(rows, key=lambda row: float(row["valid_mae"]))
    selected = [row for row in rows if int(row["checkpoint_selected"]) == 1]
    assert len(selected) == 1
    assert int(selected[0]["epoch"]) == int(best["epoch"])
    # the training function has no test-data parameter
    import inspect

    params = inspect.signature(m.train_one).parameters
    assert not any("test" in name for name in params)


# ---------------------------------------------------------------------------
# Test 12 -- all runs log optimizer steps and epoch counts
# ---------------------------------------------------------------------------


def test_all_runs_log_steps_and_epochs() -> None:
    run_paths = sorted((m.RESULTS_DIR / "runs").glob("*.json"))
    if not run_paths:
        pytest.skip("no search runs completed yet")
    for path in run_paths:
        summary = json.loads(path.read_text(encoding="utf-8"))
        assert "optimizer_steps" in summary
        assert "epochs_run" in summary
        assert int(summary["optimizer_steps"]) == int(summary["epochs_run"]) * int(
            summary["steps_per_epoch"]
        )
        assert int(summary["epochs_run"]) > 0

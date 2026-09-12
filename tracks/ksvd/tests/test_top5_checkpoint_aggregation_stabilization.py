"""Tests for the Top-5 checkpoint aggregation stabilization audit.

Static / unit tests plus lightweight local-artifact checks.  They never train a
model, never evaluate official valid as a decision and never load official test.
"""

from __future__ import annotations

import hashlib
import inspect
import json
from pathlib import Path

import pytest
import torch

from tracks.ksvd.experiments.luyin16 import zinc_top5_checkpoint_aggregation_stabilization as T
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as shead

RESULTS = T.RESULTS_DIR


def _has(path: Path) -> bool:
    return path.exists()


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for chunk in iter(lambda: handle.read(1 << 20), b""):
            digest.update(chunk)
    return digest.hexdigest()


# ---------------------------------------------------------------------------
# Test 2/3: K and tie rule frozen
# ---------------------------------------------------------------------------


def test_K_is_fixed_at_five():
    assert T.K == 5
    assert T.LOCKED_PROTOCOL["K"] == 5


def test_registered_protocol_and_gates():
    assert T.LOCKED_PROTOCOL["checkpoint_ranking_metric"] == "800-selection MAE"
    assert T.LOCKED_PROTOCOL["tie_rule"] == "earliest epoch"
    assert T.LOCKED_PROTOCOL["prediction_aggregation"] == "arithmetic mean"
    assert T.LOCKED_PROTOCOL["weight_aggregation"] == "arithmetic mean"
    assert T.LOCKED_PROTOCOL["epoch_spacing"] == "none"
    assert T.LOCKED_PROTOCOL["greedy_selection"] is False
    assert T.LOCKED_PROTOCOL["weighted_soup"] is False
    assert T.LOCKED_PROTOCOL["primary_estimator"] == "weight soup"
    assert T.LOCKED_PROTOCOL["diagnostic_estimator"] == "prediction ensemble"
    assert T.SOUP_MEAN_GATE == 0.0015
    assert T.ENS_MEAN_GATE == 0.0020
    assert T.BOOTSTRAP_B == 2000


# ---------------------------------------------------------------------------
# Test 1/3: Top-5 ranking strictly by 800 MAE with earliest-epoch ties
# ---------------------------------------------------------------------------


def test_top5_selection_uses_only_800_mae():
    snapshots = [
        {"epoch": 10, "select_800": 0.5, "train_loss": 9.9},
        {"epoch": 20, "select_800": 0.1, "train_loss": 0.1},
        {"epoch": 30, "select_800": 0.3, "train_loss": 0.0},
        {"epoch": 40, "select_800": 0.2, "train_loss": 100.0},
        {"epoch": 50, "select_800": 0.4, "train_loss": -1.0},
        {"epoch": 60, "select_800": 0.6, "train_loss": 5.0},
    ]
    selected = T._top5_epochs(snapshots)
    assert [row["epoch"] for row in selected] == [20, 40, 30, 50, 10]
    assert [row["800_mae"] for row in selected] == [0.1, 0.2, 0.3, 0.4, 0.5]


def test_top5_tie_breaks_to_earliest_epoch():
    snapshots = [
        {"epoch": 100, "select_800": 0.2, "train_loss": 0.0},
        {"epoch": 5, "select_800": 0.2, "train_loss": 1.0},
        {"epoch": 50, "select_800": 0.2, "train_loss": 2.0},
    ]
    selected = T._top5_epochs(snapshots, k=3)
    assert [row["epoch"] for row in selected] == [5, 50, 100]


def test_top5_returns_exactly_k():
    snapshots = [{"epoch": e, "select_800": float(e)} for e in range(1, 20)]
    assert len(T._top5_epochs(snapshots)) == 5


# ---------------------------------------------------------------------------
# Test 4/5/6: probe / valid / test never participate in selection
# ---------------------------------------------------------------------------


def test_selection_function_has_no_probe_or_valid_arguments():
    source = inspect.getsource(T._top5_epochs)
    assert "probe" not in source
    assert "valid" not in source
    assert "select_800" in source


def test_selection_ranks_only_on_select_800_field():
    source = inspect.getsource(T.top5_manifest)
    assert "official_valid_used_in_selection" in source
    assert "probe_2000_used_in_selection" in source
    assert "select_800" in inspect.getsource(T._top5_epochs)


def test_official_test_never_loaded():
    source = inspect.getsource(T)
    assert "_load_zinc(ZINC_ROOT, " + '"test")' not in source
    assert "matrices_for_split(" + '"test"' not in source


# ---------------------------------------------------------------------------
# Test 8: prediction ensemble is the equal arithmetic mean
# ---------------------------------------------------------------------------


def test_ensemble_is_arithmetic_prediction_mean():
    for fn in (T.development_reuse_probe, T.official_valid):
        source = inspect.getsource(fn)
        assert "five.mean(axis=0)" in source
        assert "median" not in source


# ---------------------------------------------------------------------------
# Test 9/10/11/12/13: soup is an equal parameter mean, not greedy/weighted
# ---------------------------------------------------------------------------


def test_soup_is_equal_parameter_mean_source():
    source = inspect.getsource(T.build_soup)
    assert "stacked.mean(dim=0)" in source
    assert '"weighted": False' in source
    assert '"greedy": False' in source


def test_registered_no_greedy_no_weighted():
    assert T.LOCKED_PROTOCOL["greedy_selection"] is False
    assert T.LOCKED_PROTOCOL["weighted_soup"] is False


@pytest.mark.skipif(not _has(RESULTS / "soup_construction_I0T0.json"), reason="local soup artifacts absent")
def test_soup_matches_manual_equal_mean_and_params():
    construction = json.loads((RESULTS / "soup_construction_I0T0.json").read_text())
    members = construction["members"]
    states = [torch.load(m["checkpoint_path"], map_location="cpu", weights_only=True) for m in members]
    soup = torch.load(construction["soup_state_path"], map_location="cpu", weights_only=True)
    assert sorted(soup.keys()) == sorted(states[0].keys())
    for key in soup:
        manual = torch.stack([s[key].float() for s in states], dim=0).mean(dim=0)
        assert torch.allclose(soup[key], manual, atol=0.0, rtol=0.0)
    model = shead.build_smallhead(0)
    model.load_state_dict(soup, strict=True)
    assert int(shead._n_params(model)) == 82115


@pytest.mark.skipif(not _has(RESULTS / "top5_manifest_I0T0.json"), reason="local top5 artifacts absent")
def test_soup_state_keys_match_fresh_raw_model():
    construction = json.loads((RESULTS / "soup_construction_I0T0.json").read_text())
    soup = torch.load(construction["soup_state_path"], map_location="cpu", weights_only=True)
    model = shead.build_smallhead(0)
    assert sorted(soup.keys()) == sorted(model.state_dict().keys())


# ---------------------------------------------------------------------------
# Test 7: checkpoint hashes correct
# ---------------------------------------------------------------------------


@pytest.mark.skipif(not _has(T.SOURCE_DIR / "checkpoint_manifest_I0T0.json"), reason="source snapshots absent")
def test_all_top5_checkpoint_hashes_correct():
    for run_id in T.RUNS:
        manifest = json.loads((T.SOURCE_DIR / f"checkpoint_manifest_{run_id}.json").read_text())
        by_epoch = {int(e["epoch"]): e for e in manifest["snapshots"]}
        selected = T._top5_epochs(
            [
                {
                    "epoch": int(e["epoch"]),
                    "select_800": float(e["select_800"]),
                    "checkpoint_path": e["snapshot"],
                    "checkpoint_sha256": e["snapshot_sha256"],
                }
                for e in manifest["snapshots"]
            ]
        )
        for row in selected:
            entry = by_epoch[row["epoch"]]
            assert _sha256(Path(entry["snapshot"])) == entry["snapshot_sha256"]


# ---------------------------------------------------------------------------
# Test 14: buffers follow the locked rule
# ---------------------------------------------------------------------------


def test_smallhead_has_no_buffers():
    model = shead.build_smallhead(0)
    assert list(model.named_buffers()) == []


# ---------------------------------------------------------------------------
# Test 15/16: confirmation lock ordering and no post-valid modification
# ---------------------------------------------------------------------------


def test_confirmation_lock_required_before_valid(tmp_path, monkeypatch):
    monkeypatch.setattr(T, "RESULTS_DIR", tmp_path)
    with pytest.raises(RuntimeError):
        T._require_confirmation_lock()


def test_official_valid_has_no_optimizer_or_training_step():
    source = inspect.getsource(T.official_valid)
    assert "optimizer" not in source
    assert "backward" not in source
    assert "train()" not in source


@pytest.mark.skipif(not _has(RESULTS / "confirmation_lock.json"), reason="local lock artifacts absent")
def test_confirmation_lock_precedes_valid_artifacts_and_hashes_match():
    lock = json.loads((RESULTS / "confirmation_lock.json").read_text())
    assert lock["K"] == 5
    assert lock["official_valid_loaded"] is False
    assert lock["official_test_loaded"] is False
    assert "protocol_hash" in lock
    assert set(lock["top5_manifest_hashes"]) == set(T.RUNS)
    assert set(lock["soup_state_hashes"]) == set(T.RUNS)
    # lock must have been written before the official-valid outputs
    valid_paths = [RESULTS / f"official_valid_{r}.json" for r in T.RUNS]
    if all(p.exists() for p in valid_paths):
        assert (RESULTS / "confirmation_lock.json").stat().st_mtime <= min(p.stat().st_mtime for p in valid_paths) + 1e-6
    # recorded soup hash still matches the on-disk frozen state
    for run_id in T.RUNS:
        construction = json.loads((RESULTS / f"soup_construction_{run_id}.json").read_text())
        assert _sha256(Path(construction["soup_state_path"])) == lock["soup_state_hashes"][run_id]["file_sha256"]


# ---------------------------------------------------------------------------
# Test 17: no new backbone training
# ---------------------------------------------------------------------------


def test_module_does_not_train():
    source = inspect.getsource(T)
    assert "loss.backward" not in source
    assert "optimizer.step" not in source
    assert "F.l1_loss" not in source


def test_source_trajectories_are_reused_only():
    source = inspect.getsource(T)
    assert "internal_generalization_stochasticity_audit" in source or "iga.RESULTS_DIR" in source
    assert "train_run" not in source

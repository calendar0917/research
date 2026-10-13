"""Fast, data-free tests for cscl-bjoc-pretraining-mechanism-terminal-test-v1.

These tests import the driver module and check frozen semantics only.  They do
not load ZINC data, checkpoints or result artifacts, and they never reference
the official test split.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch
import yaml

_CODE_DIR = Path(__file__).resolve().parents[1] / "code"
if str(_CODE_DIR) not in sys.path:
    sys.path.insert(0, str(_CODE_DIR))

import run_bjoc_pretraining_mechanism_terminal_test_v1 as drv  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
CONFIG = ROOT / "configs" / "bjoc_pretraining_mechanism_terminal_test_v1.yaml"


def test_config_matches_driver():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["protocol"] == drv.PROTOCOL_ID
    assert cfg["status"] == "frozen"
    assert cfg["frozen_models"]["parameters"] == 413505
    assert cfg["terminal_test"]["models"] == ["B0", "B1", "P0", "P1"]
    assert cfg["junction_reset"]["seeds"] == [0, 1]
    assert cfg["terminal_test"]["bootstrap"]["n_resamples"] == 5000
    assert cfg["budget"]["gpu_hours_max"] == 0.8


def test_cli_subcommands():
    parser = drv.build_parser()
    actions = [a for a in parser._actions if getattr(a, "choices", None)]
    names = set(actions[0].choices)
    assert {
        "reuse-audit",
        "legality",
        "mechanism",
        "tail-analysis",
        "jreset-gate",
        "train-jreset",
        "predict-jreset",
        "score-jreset",
        "next-hypothesis",
        "test-protocol",
        "test-predict",
        "test-score",
        "valid-test-compare",
        "decision",
    } <= names


def test_label_guard_blocks_labels():
    class Row:
        def __init__(self):
            self.x = torch.zeros(2, 1)
            self.y = torch.ones(1)

    guard = drv.LabelGuard(Row())
    assert guard.x.shape == (2, 1)
    for field in drv.LABEL_FIELDS:
        with pytest.raises(RuntimeError):
            getattr(guard, field)


def test_encoder_state_hash_matches_previous_round():
    state = {
        "atom_encoder": {"w": torch.arange(4, dtype=torch.float32).reshape(2, 2)},
        "bond_encoder": {"w": torch.ones(2, 1)},
        "junction_encoder": {"w": torch.zeros(3)},
    }
    combined = {}
    for prefix, group in (("atom", "atom_encoder"), ("bond", "bond_encoder"), ("junc", "junction_encoder")):
        for k, v in state[group].items():
            combined[f"{prefix}:{k}"] = v
    assert drv._encoder_state_hash(state) == drv.f10k.state_group_sha256(combined)


def test_group_equal_semantics():
    a = {"w": torch.zeros(2)}
    b = {"w": torch.zeros(2)}
    c = {"w": torch.ones(2)}
    assert drv._group_equal(a, b)
    assert not drv._group_equal(a, c)
    assert not drv._group_equal(a, {"x": torch.zeros(2)})


def test_top_k_error_share():
    err = np.asarray([1.0, 2.0, 3.0, 4.0], dtype=np.float64)
    out = drv._top_k_error_share(err, 2)
    assert out["k"] == 2
    assert out["max_error_row"] == 3
    assert abs(out["share_of_total_error"] - 7.0 / 10.0) < 1e-12


def test_paired_bootstrap_definition():
    delta = np.asarray([-0.1] * 100 + [0.1] * 100, dtype=np.float64)
    out = drv.paired_bootstrap(delta, n=200, seed=1)
    assert abs(out["mean"]) < 1e-12
    assert out["ci_lo"] <= out["ci_hi"]
    assert out["unit"] == "molecule"
    assert out["n_resamples"] == 200


def test_decision_vocabulary_is_frozen():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    vocab = set(cfg["decision_vocabulary"])
    for item in ("REPLAY_CONFIRMED", "TERMINAL_SCORE_BLOCKED", "PURCHASE_JRESET", "J_RESET_NOT_PURCHASED",
                 "PRETRAINING_TEST_GAIN_REPLICATES", "PRETRAINING_TEST_GAIN_SPLIT_SENSITIVE", "TERMINAL_TEST_INCONCLUSIVE"):
        assert item in vocab


def test_budget_and_bans_present():
    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    banned = " ".join(cfg["bans"]).lower()
    assert "retraining b/p" in banned
    assert "seed selection on test" in banned
    assert "test-guided" in banned
    assert cfg["budget"]["no_extra_training_after_test_read"] is True
    assert cfg["next_round"]["draft_rule"].startswith("the next-round predictive hypothesis must be drafted")

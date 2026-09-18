"""Targeted tests for the MolHIV small-head official-test closure.

Fast, self-contained: no MolHIV record cache, no training, no test load.
"""

from __future__ import annotations

import json

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import (
    molhiv_bnull_small_head_probe_test_closure as closure,
)
from tracks.ksvd.experiments.luyin16 import molhiv_bnull_small_head_probe as probe


def test_head_state_names_and_param_counts():
    assert closure.HEAD_NAMES == ("H_refit", "H_small32")
    assert probe._n_params(probe.HEADS["H_refit"]()) == 100417
    assert probe._n_params(probe.HEADS["H_small32"]()) == 14177


def test_head_logits_batched_matches_single_forward():
    torch.manual_seed(0)
    head = probe.HEADS["H_small32"]()
    R = torch.randn(300, probe.HEAD_REFIT_INPUT_DIM)
    batched = closure._head_logits(head, R, torch.device("cpu"))
    head.eval()
    with torch.no_grad():
        single = probe._head_forward(head, R).numpy().astype(np.float64)
    assert batched.shape == (300,)
    assert np.allclose(batched, single, atol=1e-9)


def test_freeze_refuses_when_extraction_missing(tmp_path, monkeypatch):
    monkeypatch.setattr(closure, "SOURCE_DIR", tmp_path)
    monkeypatch.setattr(closure, "RESULTS_DIR", tmp_path / "out")
    monkeypatch.setattr(probe, "R_META_PATH", tmp_path / "R_extraction.json")
    try:
        closure.freeze()
    except FileNotFoundError:
        return
    raise AssertionError("freeze should fail when the extraction record is missing")


def test_freeze_refuses_when_already_unlocked(tmp_path, monkeypatch):
    out = tmp_path / "out"
    out.mkdir()
    (out / "official_test_unlock.json").write_text("{}", encoding="utf-8")
    monkeypatch.setattr(closure, "RESULTS_DIR", out)
    try:
        closure.freeze()
    except RuntimeError as error:
        assert "already unlocked" in str(error)
        return
    raise AssertionError("freeze must refuse after the official test was unlocked")


def test_test_eval_requires_freeze(tmp_path, monkeypatch):
    monkeypatch.setattr(closure, "RESULTS_DIR", tmp_path / "out")
    try:
        closure.test_eval(device="cpu")
    except RuntimeError as error:
        assert "architecture_freeze.json missing" in str(error)
        return
    raise AssertionError("test must require a pre-test freeze record")


def test_git_commit_is_nonempty():
    assert isinstance(closure._git_commit(), str)
    assert closure._git_commit()

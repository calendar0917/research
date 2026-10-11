"""zinc-full-y-budget-control-v1 correctness tests (fast, CPU, no data).

These tests pin the frozen protocol constants, the source-level discipline
guards (seed 0 only, no official test, no component-supervision targets in the
new driver), the paired-bootstrap statistic and the pre-registered decision
bands.  Heavy data/checkpoint gates live in the runner's ``audit`` / ``replay``
/ ``predict`` stages and in ``historical_reuse_audit.json`` /
``protocol_replay.json``.
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[3]
_CODE_DIR = REPO_ROOT / "tracks/structure-contribution/code"
for p in (str(_CODE_DIR),):
    if p not in sys.path:
        sys.path.insert(0, p)

CONFIG = REPO_ROOT / "tracks/structure-contribution/configs/zinc_full_y_budget_control_v1.yaml"


def _driver():
    import run_full_y_budget_control_v1 as drv

    return drv


def _driver_source() -> str:
    return Path(_driver().__file__).read_text(encoding="utf-8")


def _strip_docstrings(text: str) -> str:
    class _Strip(ast.NodeTransformer):
        def _drop(self, node):
            self.generic_visit(node)
            if (
                node.body
                and isinstance(node.body[0], ast.Expr)
                and isinstance(node.body[0].value, ast.Constant)
                and isinstance(node.body[0].value.value, str)
            ):
                node.body = node.body[1:] or [ast.Pass()]
            return node

        visit_Module = _drop
        visit_FunctionDef = _drop
        visit_AsyncFunctionDef = _drop
        visit_ClassDef = _drop

    return ast.unparse(_Strip().visit(ast.parse(text)))


# ---------------------------------------------------------------------------
# frozen protocol constants
# ---------------------------------------------------------------------------


def test_frozen_constants():
    drv = _driver()
    assert drv.PROTOCOL_ID == "zinc-full-y-budget-control-v1"
    assert drv.EPOCHS == 240
    assert drv.LR == 1.0e-3
    assert drv.WEIGHT_DECAY == 1.0e-5
    assert drv.GRAD_CLIP == 5.0
    assert drv.BATCH_SIZE == 128
    assert drv.SOUP_LAST == 5
    assert drv.TRAIN_SHUFFLE_OFFSET == 101
    assert drv.FULL_PARAMETERS == 408_651
    assert abs(drv.H1_LAMBDA - 33.95873017865987) < 1e-12
    assert drv.BOOTSTRAP_N == 5000
    assert drv.BOOTSTRAP_SEED == 0
    assert drv.DECISION_BANDS == {"A": -0.010, "B": -0.003}


def test_config_matches_driver():
    import yaml

    cfg = yaml.safe_load(CONFIG.read_text(encoding="utf-8"))
    assert cfg["protocol_id"] == "zinc-full-y-budget-control-v1"
    assert cfg["checkpoints_primary"]["rule"] == "last-5 soup for BOTH arms"
    assert cfg["model"]["parameters_expected"] == 408_651
    assert cfg["training"]["train_shuffle_seed"] == 101
    assert cfg["training"]["steps_expected"] == {"F7200": 13680, "F10000": 18960}
    assert cfg["discipline"]["seeds"] == [0]
    assert cfg["discipline"]["no_official_test"] is True
    assert cfg["endpoint"]["official_test"] == "forbidden"
    assert cfg["decision_rule"]["case_A"].startswith("delta <= -0.010")
    assert cfg["decision_rule"]["case_B"].startswith("-0.010 < delta <= -0.003")
    assert cfg["decision_rule"]["case_C"].startswith("delta > -0.003")
    assert cfg["arms"]["F10000"]["new_training"] is False
    assert cfg["arms"]["F7200"]["new_training"] is True


# ---------------------------------------------------------------------------
# discipline guards (source level)
# ---------------------------------------------------------------------------


def test_seed_zero_only():
    assert "choices=[0]" in _driver_source()


def test_no_official_test_access():
    code = _strip_docstrings(_driver_source())
    for forbidden in ('split="test"', "split='test'", "load_test_data"):
        assert forbidden not in code, f"driver must never touch official test ({forbidden})"
    # the only split literal is the validation read
    assert 'split="val"' in _driver_source()


def test_no_component_supervision_targets():
    """The decomposition may only be *read* inside stage_audit's provenance check;
    no training/prediction/scoring stage may touch it."""
    tree = ast.parse(_driver_source())
    offenders = []
    for fn in (node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)):
        src = ast.unparse(fn)
        if fn.name in ("stage_audit", "audit_runner_source"):
            continue
        if "DECOMP_PATH" in src or "load_decomp" in src or "decomp[" in src:
            offenders.append(fn.name)
    assert offenders == [], f"component targets read outside the audit stage: {offenders}"


def test_train_loss_is_raw_y_plus_label_free_reconstruction():
    tree = ast.parse(_driver_source())
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "train_one_arm"
    )
    src = ast.unparse(fn)
    assert "F.l1_loss" not in src  # torch.nn.functional is used via the module alias
    assert "torch.nn.functional.l1_loss(prediction.view(-1), batch.y.view(-1))" in src
    assert "model.reconstruction_loss(aux['phi'], aux['coord'])" in src
    assert "loss = task + H1_LAMBDA * rec" in src
    assert "mask=cm.C6_MASK" in src


def test_predict_stage_does_not_read_labels_before_freeze():
    tree = ast.parse(_driver_source())
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "stage_predict"
    )
    src = ast.unparse(fn)
    # predictions are written and the manifest is frozen; y is never touched
    assert ".y" not in src
    assert "np.savez_compressed" in src
    assert "write_json(FREEZE_MANIFEST" in src
    assert "extraction_program_sha256" in src


def test_score_locks_the_frozen_evaluation_program():
    tree = ast.parse(_driver_source())
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "stage_score"
    )
    src = ast.unparse(fn)
    assert "extraction_program_sha256" in src
    i_prog = src.index("extraction_program_sha256")
    i_y = src.index("float(ds[i].y")
    assert i_prog < i_y


def test_score_rehashes_predictions_before_reading_y():
    tree = ast.parse(_driver_source())
    fn = next(
        node for node in ast.walk(tree)
        if isinstance(node, ast.FunctionDef) and node.name == "stage_score"
    )
    src = ast.unparse(fn)
    i_hash = src.index("file_sha256(path)")
    i_y = src.index("float(ds[i].y")
    assert i_hash < i_y, "prediction re-hash must precede the official y read"


# ---------------------------------------------------------------------------
# statistics + decision mapping
# ---------------------------------------------------------------------------


def test_bootstrap_is_signed_and_centered():
    drv = _driver()
    rng = np.random.default_rng(0)
    d = rng.normal(loc=-0.01, scale=0.05, size=1000)
    out = drv.bootstrap_paired(d, n=500, seed=0)
    assert out["statistic"].startswith("paired signed per-molecule diff")
    assert out["ci95"][0] <= out["mean_diff"] <= out["ci95"][1]
    assert out["ci95"][0] < out["ci95"][1]
    assert out["n_F10000_better"] + out["n_F7200_better"] + int(out["frac_tied"] * len(d)) == len(d)
    # a purely positive-gap input must not be reported as an improvement
    pos = drv.bootstrap_paired(np.abs(d), n=200, seed=0)
    assert pos["mean_diff"] > 0


def test_decision_bands():
    drv = _driver()
    assert drv.decide_band(-0.02) == "A_clear_budget_gain"
    assert drv.decide_band(-0.0100001) == "A_clear_budget_gain"
    assert drv.decide_band(-0.010) == "A_clear_budget_gain"
    assert drv.decide_band(-0.0099) == "B_moderate_gain"
    assert drv.decide_band(-0.003) == "B_moderate_gain"
    assert drv.decide_band(-0.0029) == "C_no_useful_gain"
    assert drv.decide_band(0.0) == "C_no_useful_gain"
    assert drv.decide_band(+0.01) == "C_no_useful_gain"


def test_state_hash_and_row_hash_are_content_addressed():
    drv = _driver()
    import torch

    a = {"x": torch.zeros(3), "y": torch.ones(2)}
    b = {"y": torch.ones(2), "x": torch.zeros(3)}
    assert drv.state_hash(a) == drv.state_hash(b)  # key order independent
    c = {"x": torch.zeros(3), "y": torch.ones(2) + 1}
    assert drv.state_hash(a) != drv.state_hash(c)
    assert drv.sha256_rows(np.array([0, 1, 2])) == drv.sha256_rows(np.array([0, 1, 2]))
    assert drv.sha256_rows(np.array([0, 1, 2])) != drv.sha256_rows(np.array([0, 1, 3]))

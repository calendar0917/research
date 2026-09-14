"""Static / unit tests for the recurrent variance-diagnosis module.

No training, no model forward, official test never loaded.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_pair_centre as rec,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_compact_v4_recurrent_variance_diagnosis as vd,
)
from tracks.ksvd.experiments.luyin16 import zinc_compact_v4_smallhead_e2e as sh


def test_01_canonical_architecture_is_the_t2_recurrent_q16():
    assert rec.RECURRENCE_ROUNDS == 2
    assert rec.Q_DIM == 16
    assert rec.R_DIM == 302
    assert vd.EXPECTED_TOTAL == 82115
    assert vd.EXPECTED_HEAD == 4135
    assert vd.CANONICAL_WEIGHT_DECAY == 1.0e-5
    assert vd.WD_WEIGHT_DECAY == 1.0e-4
    assert sh.OPTIMIZED_PROTOCOL["weight_decay"] == 1.0e-5
    assert sh.OPTIMIZED_PROTOCOL["max_epochs"] == 240
    assert sh.OPTIMIZED_PROTOCOL["patience"] == 40
    assert sh.OPTIMIZED_PROTOCOL["scheduler"] == "none"


def test_02_soup_rule_is_fixed_top5_ties_earliest():
    manifest = [
        {"epoch": 1, "valid_mae": 0.3},
        {"epoch": 2, "valid_mae": 0.2},
        {"epoch": 3, "valid_mae": 0.2},
        {"epoch": 4, "valid_mae": 0.5},
        {"epoch": 5, "valid_mae": 0.4},
        {"epoch": 6, "valid_mae": 0.9},
    ]
    top = vd._top5_epochs({"snapshot_manifest": manifest})
    assert [row["epoch"] for row in top] == [2, 3, 1, 5, 4]
    assert vd.SOUP_K == 5


def test_03_soup_state_is_exact_arithmetic_mean(tmp_path: Path):
    states = []
    for index, value in enumerate((1.0, 2.0, 3.0, 4.0, 5.0)):
        state = {
            "w": torch.full((3,), value),
            "b": torch.tensor([value, -value]),
        }
        path = tmp_path / f"epoch_{index:03d}.pt"
        torch.save(state, path)
        states.append({"epoch": index + 1, "path": str(path), "valid_mae": value})
    soup = vd.build_soup_state(states)
    assert torch.allclose(soup["w"], torch.full((3,), 3.0))
    assert torch.allclose(soup["b"], torch.tensor([3.0, -3.0]))
    assert soup["w"].dtype == torch.float32


def test_04_equal_weight_ensemble_and_mae_arithmetic():
    targets = np.array([1.0, 2.0, 3.0, 4.0])
    p0 = np.array([1.5, 1.5, 3.5, 3.5])
    p1 = np.array([0.5, 2.5, 2.5, 4.5])
    p2 = targets + 0.3
    # single MAEs
    assert vd._mae(targets, p0) == 0.5
    assert vd._mae(targets, p1) == 0.5
    assert abs(vd._mae(targets, p2) - 0.3) < 1.0e-12
    # pair ensemble
    pair01 = 0.5 * (p0 + p1)
    assert np.allclose(pair01, targets)
    assert vd._mae(targets, pair01) == 0.0
    # three-seed ensemble -> targets + 0.1 exactly
    triple = np.mean(np.stack([p0, p1, p2], axis=0), axis=0)
    assert np.allclose(triple, targets + 0.1)
    assert abs(vd._mae(targets, triple) - 0.1) < 1.0e-12


def test_05_wd_trigger_is_two_thousandths():
    assert vd.WD_TRIGGER == 0.002
    canonical = 0.140609
    assert (canonical - 0.138609) >= vd.WD_TRIGGER
    assert (canonical - 0.138610) < vd.WD_TRIGGER


def test_06_snapshot_saving_is_inert_by_default():
    import inspect

    sig = inspect.signature(sh.train_model)
    assert sig.parameters["snapshot_dir"].default is None
    sig_rec = inspect.signature(rec.train)
    assert sig_rec.parameters["snapshot_dir"].default is None


def test_07_no_forbidden_knobs_in_module():
    import inspect

    # The module *documents* the prohibitions; check that no forbidden
    # operation is actually implemented or imported.
    names = {
        name
        for name, _ in inspect.getmembers(vd, inspect.isfunction)
    } | {name for name, _ in inspect.getmembers(vd, inspect.isclass)}
    assert "build_q24" not in names
    assert "build_pair_to_pair" not in names
    assert not any("distill" in name.lower() for name in names)
    assert not any("teacher" in name.lower() for name in names)
    source = inspect.getsource(vd)
    assert "reduce_on_plateau" not in source
    assert "learning_rate" not in source  # no LR sweep hook

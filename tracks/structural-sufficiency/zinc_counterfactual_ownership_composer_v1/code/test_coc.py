"""Fast, data-free unit tests for the COC model surface."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import coc_model as CM  # noqa: E402


def _norm():
    return {"global_mean": np.zeros(33, dtype=np.float32),
            "global_scale": np.ones(33, dtype=np.float32)}


def _empty_batch():
    return {
        "n_graphs": 1,
        "atom_feat": torch.zeros((1, 108), dtype=torch.float32),
        "atom_type": torch.zeros(1, dtype=torch.long),
        "atom_batch": torch.zeros(1, dtype=torch.long),
        "bond_feat": torch.zeros((0, 7), dtype=torch.float32),
        "bond_type": torch.zeros(0, dtype=torch.long),
        "bond_batch": torch.zeros(0, dtype=torch.long),
        "bond_index": torch.zeros((0, 2), dtype=torch.long),
        "junc_feat": torch.zeros((0, 32), dtype=torch.float32),
        "junc_batch": torch.zeros(0, dtype=torch.long),
        "junc_index": torch.zeros((0, 3), dtype=torch.long),
        "junc_bond": torch.zeros((0, 2), dtype=torch.long),
        "global": torch.zeros((1, 33), dtype=torch.float32),
    }


def test_parameter_count_and_empty_forward():
    model = CM.COCDualChannel(_norm())
    params = int(sum(p.numel() for _, p in model.named_parameters()))
    assert params == 494193
    y = model(_empty_batch())
    assert y.shape == (1,)
    assert torch.isfinite(y).all()


def test_model_has_no_extra_modules_and_no_detach():
    model = CM.COCDualChannel(_norm())
    names = {name for name, _ in model.named_modules()}
    assert "local_composer" in names
    assert "group_composer" in names
    src = (HERE / "coc_model.py").read_text(encoding="utf-8")
    line = [ln for ln in src.splitlines() if "c_loc =" in ln]
    assert line and all("detach" not in ln for ln in line)

"""Test bootstrap: put the round's code, the frozen prep code and v1 on ``sys.path``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROUND = Path(__file__).resolve().parents[1]
CODE_DIR = ROUND / "code"
PREP_CODE = ROUND.parent / "lcrc_cross_task_preparation_v1" / "code"
V1_CODE = ROUND.parent / "lcrc_molnet_benchmark_pilot_v1" / "code"

for path in (str(CODE_DIR), str(PREP_CODE), str(V1_CODE)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(scope="session")
def round_dir() -> Path:
    return ROUND


@pytest.fixture(scope="session")
def freesolv_view():
    """A tiny regression view built from the real FreeSolv train/valid split."""
    import mb2_data as D

    return D.build_view("FreeSolv")


@pytest.fixture(scope="session")
def tox21_view():
    """The real Tox21 train/valid view (multitask, with missing labels)."""
    import mb2_data as D

    return D.build_view("Tox21")

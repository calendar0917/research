"""Test bootstrap: put the round's code (and the frozen upstream code) on ``sys.path``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

ROUND = Path(__file__).resolve().parents[1]
CODE_DIR = ROUND / "code"
PREP_CODE = ROUND.parent / "lcrc_cross_task_preparation_v1" / "code"
V1_CODE = ROUND.parent / "lcrc_molnet_benchmark_pilot_v1" / "code"
V2_CODE = ROUND.parent / "lcrc_molnet_benchmark_pilot_v2" / "code"

for path in (str(CODE_DIR), str(V2_CODE), str(PREP_CODE), str(V1_CODE)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(scope="session")
def round_dir() -> Path:
    return ROUND


@pytest.fixture(scope="session")
def freesolv_rows():
    import mb2_data as D

    view = D.build_view("FreeSolv")
    return view.rows["train"][:8], view.rows["valid"][:4]

"""Test bootstrap: put the round's code and the frozen prep code on ``sys.path``."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

CODE_DIR = Path(__file__).resolve().parents[1] / "code"
PREP_CODE = (
    Path(__file__).resolve().parents[3]
    / "lcrc_cross_task_preparation_v1"
    / "code"
)

for path in (str(CODE_DIR), str(PREP_CODE)):
    if path not in sys.path:
        sys.path.insert(0, path)


@pytest.fixture(scope="session")
def code_dir() -> Path:
    return CODE_DIR


@pytest.fixture(scope="session")
def small_view():
    """A tiny regression view built from the real FreeSolv train/valid split."""
    import mb_common as C
    import mb_data as D

    spec = D.frozen_spec()
    view = D.build_view("FreeSolv", spec=spec)
    return view

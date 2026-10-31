from __future__ import annotations

import sys
from pathlib import Path

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import pytest  # noqa: E402

import m2_data as MD  # noqa: E402


@pytest.fixture(scope="session")
def views():
    return MD.build_views(include_test=True)

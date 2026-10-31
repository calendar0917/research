from __future__ import annotations

import sys
from pathlib import Path

import pytest

EXP_DIR = Path(__file__).resolve().parents[1]
CODE_DIR = EXP_DIR / "code"
if str(CODE_DIR) not in sys.path:
    sys.path.insert(0, str(CODE_DIR))

import c2_common as C  # noqa: E402
import c2_data as CD  # noqa: E402


@pytest.fixture(scope="session")
def frozen_inputs():
    return CD.load_frozen_split_and_labels()


@pytest.fixture(scope="session")
def lcrc_views(frozen_inputs):
    split, labels = frozen_inputs
    return CD.build_lcrc_views(split, labels)

"""Test configuration: put this round's ``code/`` and its read-only reuse roots on ``sys.path``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROUND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = ROUND_DIR.parents[2]
CODE_DIR = ROUND_DIR / "code"
REUSED = (
    REPO_ROOT / "tracks/structural-query-learning/code",
    REPO_ROOT / "tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/code",
    REPO_ROOT / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code",
    REPO_ROOT / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code",
    REPO_ROOT / "tracks/structural-sufficiency/lcrc_relation_interface_routing_pilot_v1/code",
)
for path in (CODE_DIR, *REUSED):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# Local tests must never eat the whole machine.
os.environ.setdefault("SQ_MASK_CACHE_MB", "64")


import pytest  # noqa: E402


@pytest.fixture(scope="session")
def bundle():
    """The verified round bundle (built once per test session; ~10 s)."""
    import rp_data as D

    return D.build_bundle()

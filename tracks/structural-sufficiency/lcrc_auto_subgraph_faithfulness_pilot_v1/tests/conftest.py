"""Test configuration for lcrc-auto-subgraph-faithfulness-pilot-v1.

Puts this round's ``code/`` and its read-only reuse roots on ``sys.path`` and
registers the local ``slow`` marker (these tests load the frozen OGB caches and
the reused arm-A checkpoint; they are not fresh-clone-safe).
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROUND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = ROUND_DIR.parents[2]
CODE_DIR = ROUND_DIR / "code"
REUSED = (
    REPO_ROOT / "tracks/structural-query-learning/code",
    REPO_ROOT / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code",
    REPO_ROOT / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1/code",
    REPO_ROOT / "tracks/structural-sufficiency/lcrc_readout_context_pilot_small_v1/code",
)
for path in (CODE_DIR, *REUSED):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

os.environ.setdefault("OMP_NUM_THREADS", "4")


def pytest_configure(config) -> None:
    config.addinivalue_line(
        "markers",
        "slow: loads the frozen OGB caches and/or the reused arm-A checkpoint (not fresh-clone-safe)",
    )

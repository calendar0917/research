"""Test configuration: put this round's ``code/`` and its read-only reuse roots on ``sys.path``."""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROUND_DIR = Path(__file__).resolve().parent.parent
REPO_ROOT = ROUND_DIR.parents[2]
CODE_DIR = ROUND_DIR / "code"
REUSED = (
    REPO_ROOT / "tracks/structural-sufficiency/lcrc_pcba_structural_gain_sprint_v1/code",
    REPO_ROOT / "tracks/structural-sufficiency/lcrc_pcba_scale_pilot_v1/code",
    REPO_ROOT / "tracks/structural-sufficiency/lcrc_pcba_pipeline_perf_v1/code",
    REPO_ROOT / "tracks/structural-sufficiency/lcrc_pcba_lr_probe_v1/code",
    REPO_ROOT / "tracks/structural-sufficiency/lcrc_pcba_cpc_v1/code",
    REPO_ROOT / "tracks/structural-sufficiency/lcrc_pcba_continuation_recipe_v1/code",
    REPO_ROOT / "tracks/structural-sufficiency/lcrc_pcba_backbone_scaling_v1/code",
    REPO_ROOT / "tracks/structural-sufficiency/lcrc_benchmark_positioning_fast_v1/code",
    REPO_ROOT / "tracks/structural-sufficiency/lcrc_cross_task_preparation_v1/code",
)
for path in (CODE_DIR, *REUSED):
    if str(path) not in sys.path:
        sys.path.insert(0, str(path))

# local tests must never eat the whole machine
os.environ.setdefault("OMP_NUM_THREADS", "4")
os.environ.setdefault("MKL_NUM_THREADS", "4")


import pytest  # noqa: E402


@pytest.fixture(scope="session")
def dev_task():
    """The frozen task + development label view (loaded once; ~4.5 GB mmap/cache)."""
    import css_data as D

    return D.load_task(with_cpc=True)


@pytest.fixture(scope="session")
def plan40(dev_task):
    import css_data as D

    return D.make_plan40(dev_task)

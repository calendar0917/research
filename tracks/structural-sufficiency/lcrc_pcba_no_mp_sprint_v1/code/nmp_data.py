"""Read-only PCBA access for the no-MP sprint.

Everything delegates to the already-audited ``sg_data`` / ``lr_data`` /
``PinnedStager`` path.  No new data protocol is introduced.
"""

from __future__ import annotations

from typing import Any

import nmp_common as C  # noqa: F401  (sets up inherited import paths)
import sg_data as SD

load_task = SD.load_task
make_plan40 = SD.make_plan40
plan_summary = SD.plan_summary
plan_prefix_check = SD.plan_prefix_check
task_counts = SD.task_counts
evaluate_validation = SD.evaluate_validation
evaluate_train = SD.evaluate_train
predict_rows = SD.predict_rows
PinnedStager = SD.PinnedStager


def load_and_gate() -> Any:
    task = SD.load_task()
    return task


__all__ = [
    "PinnedStager",
    "evaluate_train",
    "evaluate_validation",
    "load_and_gate",
    "load_task",
    "make_plan40",
    "plan_prefix_check",
    "plan_summary",
    "predict_rows",
    "task_counts",
]

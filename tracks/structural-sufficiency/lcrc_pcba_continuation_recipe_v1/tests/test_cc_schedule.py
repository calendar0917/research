"""Pure-function tests for the continuation LR schedules."""

from __future__ import annotations

import math

import numpy as np
import pytest

import cc_common as C


def test_recipe_l_is_constant_floor():
    for step in (C.STEP40, C.STEP40 + 1, C.STEP60, C.STEP80 - 1):
        assert C.scheduled_lr("L", step) == C.LR_FLOOR
    with pytest.raises(ValueError):
        C.scheduled_lr("L", C.STEP40 - 1)


def test_recipe_r_warmup_and_endpoints():
    assert C.scheduled_lr("R", C.STEP40) == C.LR_FLOOR
    assert C.scheduled_lr("R", C.RESTART_WARMUP_END - 1) == C.LR_PEAK
    assert C.scheduled_lr("R", C.RESTART_WARMUP_END) == C.LR_PEAK
    assert abs(C.scheduled_lr("R", C.STEP80) - C.LR_FLOOR) < 1e-18
    # First training step of L and R is the same; the schedules diverge after it.
    assert C.scheduled_lr("L", C.STEP40 + 1) == C.LR_FLOOR
    assert C.scheduled_lr("R", C.STEP40 + 1) > C.LR_FLOOR


def test_recipe_r_cosine_decreases_after_warmup():
    steps = [C.RESTART_WARMUP_END, C.STEP50, C.STEP60, C.STEP70, C.STEP80 - 1]
    values = [C.scheduled_lr("R", s) for s in steps]
    assert all(values[i] > values[i + 1] for i in range(len(values) - 1))
    assert all(C.LR_FLOOR <= v <= C.LR_PEAK for v in values)


def test_schedule_is_deterministic_and_hashed():
    assert C.schedule_sha256("L") == C.schedule_sha256("L")
    assert C.schedule_sha256("R") == C.schedule_sha256("R")
    assert C.schedule_sha256("L") != C.schedule_sha256("R")
    assert len(C.schedule_sha256("R")) == 64


def test_schedule_values_length():
    assert len(C.schedule_values("L")) == C.STEP80 - C.STEP40
    assert len(C.schedule_values("R")) == C.STEP80 - C.STEP40


def test_audit_contains_all_epochs_and_definitions():
    audit = C.schedule_audit()
    assert len(audit["per_epoch"]["L"]) == 40
    assert len(audit["per_epoch"]["R"]) == 40
    assert audit["definitions"]["function_of_global_step_only"] is True
    assert audit["per_epoch"]["R"][0]["epoch"] == C.EPOCH40 + 1
    assert audit["per_epoch"]["R"][-1]["epoch"] == C.EPOCH80
    assert audit["endpoints"]["R"]["epoch41_last"] == C.LR_PEAK
    assert abs(audit["endpoints"]["L"]["step40"] - C.LR_FLOOR) < 1e-18

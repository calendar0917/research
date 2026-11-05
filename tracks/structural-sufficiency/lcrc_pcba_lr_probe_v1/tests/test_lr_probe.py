"""Targeted CPU correctness tests for the LR-probe round.

Fast tests (recipe, decision rule, order generator) need no data.  The
cache/checkpoint tests are marked ``slow`` because they use the frozen 4.5 GB
feature cache and the 1 MB historical checkpoint.

The authoritative GPU gates live in ``code/lr_preflight.py``; these tests catch
the same contract errors locally before deployment.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import lr_common as C  # noqa: E402
import lr_data  # noqa: E402


# ---------------------------------------------------------------------------
# fast tests
# ---------------------------------------------------------------------------


def test_frozen_recipe_and_arm_values():
    assert C.PROBE_RECIPE["optimizer"] == "AdamW"
    assert tuple(C.PROBE_RECIPE["adam_betas"]) == (0.9, 0.999)
    assert C.PROBE_RECIPE["adam_eps"] == 1e-8
    assert C.PROBE_RECIPE["weight_decay"] == 1e-3
    assert C.PROBE_RECIPE["dropout"] == 0.10
    assert C.PROBE_RECIPE["gradient_clip"] == 5.0
    assert C.PROBE_RECIPE["batch_size"] == 128
    assert C.PROBE_RECIPE["scheduler"] == "none"
    assert C.PROBE_RECIPE["epochs"] == 12
    assert list(C.PROBE_RECIPE["eval_epochs"]) == [0, 3, 6, 9, 12]
    assert C.GROUPS["H"]["learning_rate"] == 3e-4
    assert C.GROUPS["L"]["learning_rate"] == 5e-5
    assert C.GROUPS["H"]["learning_rate"] != C.GROUPS["L"]["learning_rate"]
    C.assert_inherited_constants()


def test_order_plan_matches_pc_train_batches():
    """The paired plan must be exactly ``pc_train._batches`` for 3 epochs."""
    import torch
    import pc_train

    rows = np.arange(10_000, dtype=np.int64)
    plan = lr_data.make_order_plan(rows, batch_size=128, seed=0, epochs=3)
    generator = torch.Generator().manual_seed(0)
    expected = [pc_train._batches(rows, 128, generator) for _ in range(3)]
    for p, exp in zip(plan, expected):
        assert len(p["batches"]) == len(exp)
        for got, want in zip(p["batches"], exp):
            assert np.array_equal(got, want)
        assert p["row_order_sha256"] == C.sha256_int64(np.concatenate(exp))


def test_order_plan_is_repeatable_and_independent_of_arm():
    rows = np.arange(20_000, dtype=np.int64)
    a = lr_data.make_order_plan(rows, epochs=2)
    b = lr_data.make_order_plan(rows, epochs=2)
    assert [x["row_order_sha256"] for x in a] == [x["row_order_sha256"] for x in b]
    assert [x["batch_plan_sha256"] for x in a] == [x["batch_plan_sha256"] for x in b]
    # No mutable global RNG is consumed by the order plan.
    import torch

    torch.manual_seed(123)
    before = torch.get_rng_state().clone()
    lr_data.make_order_plan(rows, epochs=1)
    assert torch.equal(before, torch.get_rng_state())


def test_nan_aware_label_comparison():
    import torch
    from lr_preflight import _labels_equal

    a = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
    b = torch.tensor([[1.0, float("nan")], [0.0, 1.0]])
    c = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    ok = _labels_equal(a, b)
    assert ok["equal"] is True
    assert ok["nan_mask_equal"] is True
    assert ok["n_nan_a"] == 1
    bad = _labels_equal(a, c)
    assert bad["equal"] is False
    assert bad["nan_mask_equal"] is False


def test_decision_labels_are_verbatim():
    def label(h, low, hs="COMPLETED", ls="COMPLETED"):
        return C.decide_label(h, low, hs, ls)["label"]

    assert label(0.2350, 0.2450) == "LR_SIGNAL"          # D_LR 0.0100, D_old 0.01051
    assert label(0.2400, 0.2440) == "TRAINING_SIGNAL_NOT_LR_SPECIFIC"  # D_LR 0.0040
    assert label(0.2320, 0.2360) == "NO_CLEAR_INCREMENT"  # D_old 0.00151
    assert label(0.2320, 0.2400) == "LR_SIGNAL"           # D_LR 0.0080, D_old 0.00551
    assert label(0.2350, 0.2450, "INCOMPLETE", "COMPLETED") == "INCOMPLETE"
    assert label(0.2350, 0.2450, "COMPLETED", "INCOMPLETE") == "INCOMPLETE"
    # Exact floor: 0.23449 + 0.005 = 0.23949 on both comparisons.
    assert label(0.23000, 0.23949) == "LR_SIGNAL"
    assert label(0.23400, 0.23949) == "LR_SIGNAL"
    assert label(0.23950, 0.24000) == "TRAINING_SIGNAL_NOT_LR_SPECIFIC"
    assert label(0.24000, 0.24499) == "TRAINING_SIGNAL_NOT_LR_SPECIFIC"


# ---------------------------------------------------------------------------
# cache/checkpoint tests (slow)
# ---------------------------------------------------------------------------


@pytest.fixture(scope="module")
def task():
    if not C.CACHE_PATH.is_dir():
        pytest.skip(f"frozen feature cache absent: {C.CACHE_PATH}")
    return lr_data.load_task()


@pytest.fixture(scope="module")
def checkpoint():
    if not C.HIST["checkpoint"].exists():
        pytest.skip(f"historical checkpoint absent: {C.HIST['checkpoint']}")
    import lr_model

    return lr_model.load_checkpoint()


@pytest.mark.slow
def test_no_test_labels_and_frozen_hashes(task):
    assert not np.isfinite(task.labels[task.test_rows]).any()
    assert task.label_artifact_sha256 == C.HIST["label_artifact_sha256"]
    assert task.split_partition_sha256 == C.HIST["split_partition_sha256"]
    assert task.norm["norm_fingerprint"] == C.HIST["norm_fingerprint"]
    assert task.spec.hash() == C.HIST["feature_spec_hash"]
    assert task.data_audit["n_tasks"] == 128
    assert task.data_audit["gids_are_official_row_order"] is True


@pytest.mark.slow
def test_historical_checkpoint_loads_bitwise(task, checkpoint):
    import torch
    import lr_model

    model, info, audit = lr_model.build_from_checkpoint(
        task, checkpoint, device=torch.device("cpu"), group="H"
    )
    assert audit["state_dict_bitwise_equal"] is True
    assert audit["parameter_fingerprints_equal"] is True
    assert audit["model_info_equal_to_historical"] is True
    assert audit["parameter_count"] == 245_184
    assert float(getattr(model, "drop").p) == 0.10
    assert next(model.parameters()).dtype == torch.float32


@pytest.mark.slow
def test_epoch0_official_ap_cpu(task, checkpoint):
    import torch
    import lr_model

    model, _, _ = lr_model.build_from_checkpoint(
        task, checkpoint, device=torch.device("cpu"), group="H"
    )
    model.eval()
    report = lr_model.evaluate_validation(model, task, torch.device("cpu"), reason="test_epoch0")
    assert abs(float(report["ap"]) - C.HIST["best_valid_ap"]) <= 1e-4
    assert report["n_tasks_scored"] == 127


@pytest.mark.slow
def test_epoch1_prefix_matches_perf_round(task):
    """The first 320 batches must be the exact seed-0 prefix measured in perf v1."""
    import pf_common

    plan = lr_data.make_order_plan(task.train_rows, epochs=1)
    batches = plan[0]["batches"][:320]
    fp = pf_common.batch_plan_fingerprint(batches)
    assert fp["rows_sha256"] == "b9266fc0b05485a117295b0d18abff3e84545da1cb60deea14b14db74e5122a9"
    assert fp["first_batch"] == [36946, 392335, 346641, 97288, 248043, 429838, 5670, 411765]
    assert fp["n_batches"] == 320


@pytest.mark.slow
def test_order_isolation_full_plan(task):
    plan = lr_data.make_order_plan(task.train_rows)
    iso = lr_data.assert_order_isolation(plan, task.train_rows, task.valid_rows, task.test_rows)
    assert iso["passed"] is True
    assert all(c["test_rows_in_epoch"] == 0 for c in iso["checks"])
    assert all(c["valid_rows_in_epoch"] == 0 for c in iso["checks"])

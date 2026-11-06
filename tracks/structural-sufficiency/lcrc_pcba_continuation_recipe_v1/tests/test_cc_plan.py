"""Order-plan tests that do not need the 9 GB PCBA cache."""

from __future__ import annotations

from types import SimpleNamespace

import numpy as np
import torch

import cc_common as C
import cc_data as D


def _fake_task():
    train = np.arange(0, 1000, dtype=np.int64)
    valid = np.arange(1000, 1100, dtype=np.int64)
    test = np.arange(1100, 1200, dtype=np.int64)
    return SimpleNamespace(train_rows=train, valid_rows=valid, test_rows=test)


def test_plan_is_shared_permutation_of_train_rows_only():
    task = _fake_task()
    plan = D.make_plan_41_80(task, seed=C.SEED)
    assert len(plan) == 40
    assert [p["epoch"] for p in plan] == list(range(41, 81))
    train_sorted = np.sort(task.train_rows)
    for record in plan:
        rows = np.asarray(record["rows"], dtype=np.int64)
        assert np.array_equal(np.sort(rows), train_sorted)
        assert rows.size == task.train_rows.size
        assert np.intersect1d(rows, task.valid_rows).size == 0
        assert np.intersect1d(rows, task.test_rows).size == 0
        assert record["n_batches"] == 8
        assert len(record["batches"][0]) == C.BATCH_SIZE
        assert record["batch_plan_sha256"] == C.sha256_batches(record["batches"])


def test_plan_does_not_touch_global_torch_rng():
    task = _fake_task()
    torch.manual_seed(1234)
    before = torch.get_rng_state().clone()
    plan_a = D.make_plan_41_80(task, seed=C.SEED)
    after = torch.get_rng_state()
    assert torch.equal(before, after)
    plan_b = D.make_plan_41_80(task, seed=C.SEED)
    assert D.plan_global_sha256(plan_a) == D.plan_global_sha256(plan_b)


def test_plan_summary_and_prefix_hash():
    task = _fake_task()
    plan = D.make_plan_41_80(task, seed=C.SEED)
    summary = D.plan_summary(plan, task, seed=C.SEED)
    assert summary["isolation"]["passed"] is True
    assert summary["global_torch_rng_unchanged"] is True
    assert summary["first_epoch"] == 41 and summary["last_epoch"] == 80
    meta60 = D.plan_for_checkpoint(plan, C.EPOCH60)
    assert meta60["consumed_epochs"] == 20
    assert meta60["consumed_through_epoch"] == 60
    assert meta60["global_plan_sha256"] == D.plan_global_sha256(plan)
    assert meta60["prefix_sha256"] == D.plan_prefix_sha256(plan, 20)
    meta80 = D.plan_for_checkpoint(plan, C.EPOCH80)
    assert meta80["consumed_epochs"] == 40
    assert meta80["prefix_sha256"] == D.plan_prefix_sha256(plan, 40)

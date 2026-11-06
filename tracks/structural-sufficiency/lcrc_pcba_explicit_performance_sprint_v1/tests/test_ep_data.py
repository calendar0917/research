"""Data-path tests: augmented collate alignment and Test isolation."""

from __future__ import annotations

import numpy as np
import pytest
import torch

import ep_common as C
import ep_data as D

_TASK = None


def _task():
    global _TASK
    if _TASK is None:
        _TASK = D.load_task()
    return _TASK


def _branch_feature(task, rows, rng):
    atom_ptr = np.asarray(task.cache["atom_ptr"], dtype=np.int64)
    counts = np.zeros(int(atom_ptr[-1]), dtype=np.int32)
    row_counts = np.zeros(C.N_MOLECULES, dtype=np.int64)
    for r in np.sort(rows):
        a0, a1 = int(atom_ptr[r]), int(atom_ptr[r + 1])
        c = rng.integers(0, 4, size=a1 - a0).astype(np.int32)
        counts[a0:a1] = c
        row_counts[r] = int(c.sum())
    ptr = np.zeros(C.N_MOLECULES + 1, dtype=np.int64)
    ptr[1:] = np.cumsum(row_counts)
    tokens = rng.integers(0, C.N_BRANCH_BUCKETS, size=int(ptr[-1])).astype(np.int16)
    return {"row_token_ptr": ptr, "token_bucket": tokens, "atom_token_count": counts,
            "feature_hash": "test", "meta": {}}


def test_branch_collate_is_atom_and_row_aligned():
    task = _task()
    rows = task.train_rows[10:14]
    rng = np.random.default_rng(5)
    branch = _branch_feature(task, rows, rng)
    feat = D.Features(modules=("A3",), branch=branch)
    batch = D.augmented_collate(task, rows, feat)
    atom_ptr = np.asarray(task.cache["atom_ptr"], dtype=np.int64)
    counts = np.concatenate([branch["atom_token_count"][atom_ptr[r]:atom_ptr[r + 1]] for r in rows])
    expected_ptr = np.concatenate([[0], np.cumsum(counts)]).astype(np.int64)
    assert np.array_equal(batch["bt_ptr"].numpy(), expected_ptr)
    tokens = batch["bt_bucket"].numpy()
    assert tokens.shape[0] == expected_ptr[-1]
    # Per-atom slices reproduce the synthetic per-atom tokens exactly.
    for i, (lo, hi) in enumerate(zip(expected_ptr[:-1], expected_ptr[1:])):
        assert int(counts[i]) == int(hi - lo)
    # Row slices are concatenated in row order.
    parts = [branch["token_bucket"][branch["row_token_ptr"][r]:branch["row_token_ptr"][r + 1]] for r in rows]
    assert np.array_equal(tokens, np.concatenate(parts))
    # The atom count tensor from the frozen collate matches the number of atoms.
    assert int(batch["atom_counts"].sum()) == int(counts.size)


def test_physchem_collate_row_alignment():
    task = _task()
    rows = task.train_rows[20:24]
    matrix = np.full((C.N_MOLECULES, C.PHYSCHEM_DIM), np.nan, dtype=np.float32)
    matrix[rows] = np.arange(len(rows) * C.PHYSCHEM_DIM, dtype=np.float32).reshape(len(rows), C.PHYSCHEM_DIM)
    feat = D.Features(modules=("A2",), physchem={"matrix": matrix, "feature_hash": "t",
                                                "norm_fingerprint": "t", "meta": {}})
    batch = D.augmented_collate(task, rows, feat)
    assert np.array_equal(batch["physchem"].numpy(), matrix[rows])


def test_collate_and_predict_refuse_test_rows():
    task = _task()
    rows = task.test_rows[:2]
    with pytest.raises(PermissionError):
        D.augmented_collate(task, rows, D.Features(modules=()))
    with pytest.raises(PermissionError):
        D.predict_logits(torch.nn.Identity(), task, D.Features(modules=()), rows, torch.device("cpu"))


def test_augmented_collate_adds_only_requested_features():
    task = _task()
    rows = task.train_rows[:3]
    base = D.augmented_collate(task, rows, D.Features(modules=()))
    for key in ("physchem", "bt_ptr", "bt_bucket"):
        assert key not in base
    rng = np.random.default_rng(0)
    a3 = D.augmented_collate(task, rows, D.Features(modules=("A3",), branch=_branch_feature(task, rows, rng)))
    assert "bt_ptr" in a3 and "physchem" not in a3
    # A0/A1 batches are exactly the frozen collate; A3 only appends tokens.
    for key in base:
        if key == "n_graphs":
            assert base[key] == a3[key]
        else:
            assert torch.equal(base[key], a3[key]), key


def test_frozen_plan_and_isolation_gates():
    task = _task()
    plan = D.make_plan40(task, seed=0)
    summary = D.plan_summary(plan, task)
    assert summary["summary"]["global_plan_sha256"] == C.HISTORICAL_PARITY[0]["global_plan_sha256"]
    prefix = D.plan_prefix(plan, 40)
    assert prefix["prefix_sha256"] == C.HISTORICAL_PARITY[0]["consumed40_prefix_sha256"]
    assert summary["isolation"]["passed"]
    assert np.isfinite(task.labels[task.test_rows]).sum() == 0

"""End-to-end builder tests on a small real subset (no full cache build)."""

from __future__ import annotations

import copy

import numpy as np
import pytest

import ep_common as C
import ep_data as D
import ep_features as F

_TASK = None


def _task():
    global _TASK
    if _TASK is None:
        _TASK = D.load_task()
    return _TASK


def test_branch_cache_builder_on_subset(tmp_path):
    task = _task()
    small = copy.copy(task)
    small.train_rows = np.asarray(task.train_rows[:48], dtype=np.int64)
    small.valid_rows = np.asarray(task.valid_rows[:12], dtype=np.int64)
    cache = tmp_path / "branch.npz"
    meta_path = tmp_path / "branch.json"
    meta = F.build_branch_cache(small, workers=2, out_cache=cache, out_meta=meta_path)
    assert meta["n_buckets"] == C.N_BRANCH_BUCKETS
    assert meta["rows_attempted"] == 60
    assert meta["test_rows_built"] == 0
    assert meta["n_tokens"] > 0
    assert meta["collision_stats"]["n_distinct_signatures"] > 0
    assert meta["collision_stats"]["n_buckets_used"] > 0
    feat = F.load_branch(small, cache=cache, meta_path=meta_path)
    assert np.all(
        feat["row_token_ptr"][task.test_rows] == feat["row_token_ptr"][np.asarray(task.test_rows) + 1]
    )
    rows = small.train_rows[:6]
    a3 = D.Features(modules=("A3",), branch=feat)
    batch = D.augmented_collate(small, rows, a3)
    n_atoms = int(batch["atom_counts"].sum())
    assert batch["bt_ptr"].numel() == n_atoms + 1
    assert int(batch["bt_ptr"][-1]) == int(batch["bt_bucket"].numel())
    assert int(batch["bt_bucket"].min()) >= 0
    assert int(batch["bt_bucket"].max()) < C.N_BRANCH_BUCKETS
    # At least one real molecule exercises a non-empty second-hop branch list.
    assert meta["n_tokens"] >= meta["n_atoms_built"]


def test_physchem_cache_builder_on_subset(tmp_path):
    pytest.importorskip("rdkit")
    task = _task()
    small = copy.copy(task)
    small.train_rows = np.asarray(task.train_rows[:40], dtype=np.int64)
    small.valid_rows = np.asarray(task.valid_rows[:10], dtype=np.int64)
    cache = tmp_path / "physchem.npz"
    meta_path = tmp_path / "physchem.json"
    meta = F.build_physchem_cache(small, workers=2, out_cache=cache, out_meta=meta_path)
    assert meta["rows_attempted"] == 50
    assert meta["built_rows"] + meta["failed_rows"] == 50
    assert meta["test_rows_built"] == 0
    assert meta["finite"]
    assert meta["train_only_norm"]["valid_or_test_used"] is False
    assert len(meta["train_only_norm"]["mean"]) == C.N_PHYSCHEM
    feat = F.load_physchem(small, cache=cache, meta_path=meta_path)
    usable = small.train_rows[feat["built"][small.train_rows]]
    assert usable.size > 0
    m = feat["matrix"]
    assert np.isfinite(m[usable]).all()
    assert np.isnan(m[task.test_rows]).all()
    assert np.allclose(m[usable][~feat["failed"][usable], :C.N_PHYSCHEM].mean(0), 0.0, atol=1e-4)

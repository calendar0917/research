"""Full-Fit batch-alignment regression tests (amendment-001).

These tests cover the defect that halted the 2026-10-30 GPU stage: global dataset
row ids were passed to a collator that accepts cache positions.  They use the real
frozen Fit2352 data, all 19 batches of a full epoch, seeds 0/1/2, and an
independent ``global_row_id -> label`` map.
"""

from __future__ import annotations

import numpy as np
import pytest
import torch

import access_guard
import fit_labels
import screen_common as C
import screen_data as D
import screen_training as T


@pytest.fixture(scope="module")
def fit_context():
    caches = D.load_caches()
    cache = caches["fit"]["cache"]
    rows = [int(r) for r in caches["fit"]["rows"]]
    access_guard.arm()
    try:
        fit = fit_labels.load_fit_labels()
        keys = [str(r) for r in sorted(rows, reverse=True)]
        scan = fit_labels.scan_label_artifact(C.LIPO_LABELS, keys)
    finally:
        access_guard.disarm()
    labels = {int(k): float(v) for k, v in scan["values"].items()}
    return cache, rows, np.asarray(fit["y"], dtype=np.float64), labels


def test_cache_order_is_frozen_split_order(fit_context):
    cache, rows, y, labels = fit_context
    assert len(rows) == len(y) == len(labels) == C.SPLIT_SIZES["fit"] == 2352
    assert [int(x) for x in cache["molecule_ids"]] == rows
    assert max(rows) > 2351, "Fit split must contain global row ids above 2351"
    for p, g in enumerate(rows):
        assert float(y[p]) == labels[g], f"y_fit[{p}] is not the label of global row {g}"


@pytest.mark.parametrize("seed", list(C.SEEDS))
def test_full_fit_epoch_is_complete_and_aligned(fit_context, seed):
    cache, rows, y, labels = fit_context
    n_fit = C.SPLIT_SIZES["fit"]
    plans = D.fit_epoch_plans(seed, cache=cache, fit_global_rows=rows)
    assert len(plans) == C.STEPS_PER_EPOCH == 19
    assert len(plans[-1].cache_positions) == 48

    positions: list[int] = []
    ids: list[int] = []
    for index, plan in enumerate(plans):
        assert list(plan.cache_positions) == list(plan.shuffle_positions)
        assert len(plan.cache_positions) == C.BATCH_SIZE or index == len(plans) - 1
        info = D.assemble_fit_batch(cache, y, plan)
        assert int(info["batch"]["n_graphs"]) == len(plan.cache_positions) == int(info["y"].shape[0])
        for p, g, target in zip(plan.cache_positions, plan.global_row_ids, info["y"]):
            assert 0 <= p < n_fit
            assert g == rows[p] == int(cache["molecule_ids"][p])
            assert float(target) == labels[g] == float(y[p])
        positions.extend(plan.cache_positions)
        ids.extend(plan.global_row_ids)

    assert len(positions) == n_fit
    assert sorted(positions) == list(range(n_fit))
    assert len(set(positions)) == n_fit
    assert len(ids) == len(set(ids)) == n_fit
    assert any(g > 2351 for g in ids)
    fingerprint = D.fit_batch_fingerprints(plans, y)
    assert fingerprint["all_positions_in_range"]
    assert fingerprint["all_positions_unique"]
    assert fingerprint["covers_all_subset_positions"]
    assert fingerprint["cache_positions_sha256"] == D.batch_order_hash(seed, n=n_fit, epochs=1)


def test_global_row_ids_cannot_reach_the_collator(fit_context):
    cache, rows, y, _labels = fit_context
    # A dataset row id >= n_molecules is rejected by the collator itself.
    with pytest.raises(ValueError, match="outside cache"):
        D.collate(cache, [0, 1, 3, 2352, 4199])
    # Passing the full global-id vector as shuffle positions is rejected.
    with pytest.raises(IndexError, match="global dataset row ids"):
        D.plan_fit_batch(rows, train_positions=D.fit_training_positions(len(rows)), cache=cache, fit_global_rows=rows)
    # Raw index lists are not accepted by the assembly interface.
    with pytest.raises(TypeError, match="FitBatchPlan"):
        D.assemble_fit_batch(cache, y, rows[:128])
    # Tensors and non-integer positions are rejected at planning time.
    with pytest.raises(TypeError):
        D.plan_fit_batch(torch.arange(8), train_positions=D.fit_training_positions(len(rows)), cache=cache, fit_global_rows=rows)
    with pytest.raises(TypeError):
        D.plan_fit_batch([0.0, 1.0], train_positions=D.fit_training_positions(len(rows)), cache=cache, fit_global_rows=rows)
    # Duplicate positions are rejected instead of silently double-counting a row.
    with pytest.raises(ValueError, match="duplicate"):
        D.plan_fit_batch([0, 0], train_positions=D.fit_training_positions(len(rows)), cache=cache, fit_global_rows=rows)


def test_executed_training_path_matches_gate_path(fit_context):
    cache, rows, y, _labels = fit_context
    cfg = T.TrainConfig(arm="A", seed=0, device="cpu", smoke=True, fit_limit=64, epochs=1)
    payload = T.train_one(cfg)
    record = payload["batch_alignment"]["per_epoch"][0]
    plans = D.fit_epoch_plans(0, n_subset=64, cache=cache, fit_global_rows=rows)
    fingerprint = D.fit_batch_fingerprints(plans, y)
    assert record["cache_positions_sha256"] == fingerprint["cache_positions_sha256"]
    assert record["global_row_ids_sha256"] == fingerprint["global_row_ids_sha256"]
    assert record["targets_sha256"] == fingerprint["targets_sha256"]
    assert payload["batch_order"]["matches"] is True
    assert payload["monitor_labels_read"] is False

"""CPU-only unit tests for the fullscale round (no cache build, no training)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import fs_common as C  # noqa: E402
import fs_data as D  # noqa: E402


def test_horizon_and_recipe_are_frozen():
    assert C.TRAIN_ROWS == 3_378_606
    assert C.STEPS_PER_EPOCH == 26_396
    assert C.TOTAL_STEPS == 1_055_840
    assert int(C.PQ.RECIPE["batch_size"]) == 128
    assert int(C.PQ.RECIPE["epochs"]) == 40
    assert tuple(C.PQ.RECIPE["adam_betas"]) == (0.9, 0.999)
    assert float(C.PQ.RECIPE["learning_rate"]) == 3e-4
    assert float(C.PQ.RECIPE["warmup_start_lr"]) == 3e-5
    assert float(C.PQ.RECIPE["end_lr"]) == 1e-5
    assert float(C.PQ.RECIPE["weight_decay"]) == 1e-3
    assert float(C.PQ.RECIPE["dropout"]) == 0.10
    assert float(C.PQ.RECIPE["gradient_clip"]) == 5.0


def test_lr_schedule_uses_full_horizon():
    s, t = C.STEPS_PER_EPOCH, C.TOTAL_STEPS
    assert C.scheduled_lr(0, steps_epoch=s, steps_total=t) == pytest.approx(
        3e-5 + (3e-4 - 3e-5) * 1 / s
    )
    assert C.scheduled_lr(s - 1, steps_epoch=s, steps_total=t) == pytest.approx(3e-4)
    assert C.scheduled_lr(s, steps_epoch=s, steps_total=t) == pytest.approx(3e-4)
    assert C.scheduled_lr(t - 1, steps_epoch=s, steps_total=t) == pytest.approx(1e-5, abs=1e-7)
    # the 1M horizon must not be accepted as this round's horizon
    assert C.PQ.steps_per_epoch(1_000_000) == 7_813
    assert C.PQ.total_steps(1_000_000) == 312_520


def test_parameter_counts_match_preregistered():
    table = C.param_table()
    assert int(table["2"]["total_parameters"]) == 939_079
    assert int(table["3"]["total_parameters"]) == 2_038_167
    assert int(table["2"]["matches_expected"]) == 1
    assert int(table["3"]["matches_expected"]) == 1


def test_full_row_plan_is_the_official_train_permutation():
    split = D.load_split_dict()["split"]
    plan = D.full_row_plan(split=split)
    rows = np.asarray(plan["_rows"])
    assert rows.size == C.TRAIN_ROWS
    assert np.unique(rows).size == rows.size
    assert np.array_equal(np.sort(rows), np.sort(np.asarray(split["train"])))
    assert np.array_equal(rows[:1_000_000], np.load(C.PARENT_CACHE / "pcqm_subset_rows.npy"))
    assert np.intersect1d(rows, split["valid"]).size == 0
    assert np.intersect1d(rows, split["test-dev"]).size == 0
    assert np.intersect1d(rows, split["test-challenge"]).size == 0
    assert plan["rows_sha256"] == C.read_json(C.PARENT_ANALYSIS / "pcqm_subset_plan.json")[
        "permutation_rows_sha256_full"
    ]


def test_epoch_orders_are_deterministic_and_horizon_sized():
    orders = C.PQ_DATA.epoch_orders(1_000, seed=0, epochs=3)
    again = C.PQ_DATA.epoch_orders(1_000, seed=0, epochs=3)
    assert orders["epoch_permutation_sha256"] == again["epoch_permutation_sha256"]
    assert orders["n_rows"] == 1_000
    assert len(orders["epoch_permutation_sha256"]) == 3
    assert C.steps_per_epoch(1_000) == 8  # ceil(1000/128)


def test_normalisation_constants_are_the_historical_300k_ones():
    label_norm = C.read_json(C.PARENT_ANALYSIS / "pcqm_label_norm.json")
    feature_norm = C.read_json(C.PARENT_ANALYSIS / "pcqm_feature_norm.json")
    assert int(label_norm["n"]) == 300_000
    assert float(label_norm["mean_eV"]) == pytest.approx(5.687756175403825)
    assert float(label_norm["std_eV"]) == pytest.approx(1.1607753112412411)
    assert label_norm["both_runs_shared"] is True
    assert float(feature_norm["fit_rows"]) == 300_000.0
    assert "cache rows 0..299999" in feature_norm["source"]


def test_e1_collation_equals_frozen_on_real_cache_rows():
    import torch

    import ew_collate as EC

    cache = D.load_cache(C.PARENT_CACHE / "pcqm_train1m")
    for rows in ([0], [0, 1], [3, 999_999, 123_456], list(range(0, 128))):
        fast = EC.collate_for(cache, rows, implementation="E1")
        frozen = EC.collate_for(cache, rows, implementation="frozen")
        cmp = EC.compare_batches(fast, frozen)
        assert cmp["all_equal"], (len(rows), cmp)
        assert fast["n_graphs"] == len(rows)
        assert torch.equal(fast["atom_counts"], frozen["atom_counts"])


def test_batch_order_covers_every_row_once_per_epoch():
    rng = np.random.default_rng(0)
    order = rng.permutation(C.TRAIN_ROWS).astype(np.int64)
    assert order.size == C.TRAIN_ROWS
    assert np.unique(order).size == C.TRAIN_ROWS
    batches = [order[i : i + 128] for i in range(0, order.size, 128)]
    assert len(batches) == C.STEPS_PER_EPOCH
    assert int(sum(b.size for b in batches)) == C.TRAIN_ROWS
    assert batches[-1].size == C.TRAIN_ROWS - (C.STEPS_PER_EPOCH - 1) * 128

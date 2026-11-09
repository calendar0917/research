"""Hand-checked examples for the ranking statistics used by P1/P3."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from scipy.stats import spearmanr

CODE = Path(__file__).resolve().parents[1] / "code"
if str(CODE) not in sys.path:
    sys.path.insert(0, str(CODE))

import lc_common as C  # noqa: E402
import lc_metrics as M  # noqa: E402


def test_spearman_monotone_and_reversed():
    x = np.array([1.0, 2.0, 3.0, 4.0])
    assert M.spearman(x, x) == pytest.approx(1.0)
    assert M.spearman(x, -x) == pytest.approx(-1.0)


def test_spearman_matches_scipy_with_ties():
    rng = np.random.default_rng(0)
    x = rng.integers(0, 5, size=200).astype(float)
    y = rng.integers(0, 7, size=200).astype(float)
    assert M.spearman(x, y) == pytest.approx(float(spearmanr(x, y).statistic), abs=1e-12)


def test_spearman_undefined_for_constant_vector():
    assert M.spearman(np.ones(10), np.arange(10.0)) is None
    assert M.spearman(np.arange(2.0), np.arange(2.0)) is None


def test_jaccard_hand_example():
    assert M.jaccard([1, 2, 3], [2, 3, 4]) == pytest.approx(0.5)
    assert M.jaccard([1, 2], [1, 2]) == pytest.approx(1.0)
    assert M.jaccard([1], [2]) == pytest.approx(0.0)
    assert M.jaccard([], []) is None


def test_topk_size_ceil():
    assert M.topk_size(0, 0.01) == 0
    assert M.topk_size(1, 0.01) == 1
    assert M.topk_size(100, 0.01) == 1
    assert M.topk_size(100, 0.05) == 5
    assert M.topk_size(57, 0.05) == 3  # ceil(2.85)
    assert M.topk_size(201, 0.01) == 3  # ceil(2.01)


def test_topk_tie_break_by_row_id():
    logits = np.array([0.5, 0.9, 0.9, 0.1])
    rows = np.array([10, 5, 3, 1])
    idx = M.top_k_indices(logits, rows, 2)
    assert rows[idx].tolist() == [3, 5]  # 0.9 ties resolved by smaller row id
    idx1 = M.top_k_indices(logits, rows, 1)
    assert rows[idx1].tolist() == [3]


def test_normalized_percentile_hand_example():
    pct = M.normalized_percentile(np.array([1.0, 2.0, 3.0]))
    assert pct.tolist() == pytest.approx([0.0, 0.5, 1.0])
    pct_ties = M.normalized_percentile(np.array([1.0, 2.0, 2.0]))
    assert pct_ties.tolist() == pytest.approx([0.0, 0.75, 0.75])
    assert M.normalized_percentile(np.array([7.0])).tolist() == pytest.approx([0.5])


def test_mutual_positive_table_hand_example():
    y = np.array([1.0, 0.0, 1.0, 1.0])
    t = M.mutual_positive_table([0, 1], [1, 3], y)
    assert t == {
        "both": 0,
        "only_a": 1,
        "only_b": 1,
        "neither": 1,
        "positives_total": 3,
        "pos_in_a": 1,
        "pos_in_b": 1,
    }
    t2 = M.mutual_positive_table([0, 2], [0, 2], y)
    assert t2["both"] == 2 and t2["only_a"] == 0 and t2["only_b"] == 0


def test_positive_rank_table_orientation():
    y = np.array([1.0, 0.0, 0.0, 0.0])
    # reference ranks the positive last, new model ranks it first
    ref = np.array([0.0, 1.0, 2.0, 3.0])
    new = np.array([4.0, 3.0, 2.0, 1.0])
    t = M.positive_rank_table(ref, new, y)  # delta = pct_new - pct_reference
    assert t["n_positive"] == 1
    assert t["median_pct_reference"] == pytest.approx(0.0)
    assert t["median_pct_new"] == pytest.approx(1.0)
    assert t["median_delta_pct"] == pytest.approx(1.0)
    t_rev = M.positive_rank_table(new, ref, y)
    assert t_rev["median_delta_pct"] == pytest.approx(-1.0)


def test_tasks_are_not_mixed():
    """Two tasks with disjoint score scales are ranked independently."""
    rng = np.random.default_rng(1)
    task0 = rng.normal(size=50)
    task1 = task0 * 1000.0  # same order, wildly different scale
    # Spearman inside each task is 1.0; a pooled mix would be distorted by the scale
    assert M.spearman(task0, task1) == pytest.approx(1.0)
    other = rng.normal(size=50)
    assert M.spearman(task0, other) == pytest.approx(float(spearmanr(task0, other).statistic), abs=1e-12)


def test_support_strata_boundaries():
    assert C.support_stratum(20, 100) == "ge20"
    assert C.support_stratum(19, 100) == "5_19"
    assert C.support_stratum(5, 100) == "5_19"
    assert C.support_stratum(4, 100) == "1_4"
    assert C.support_stratum(0, 100) == "na"
    assert C.support_stratum(10, 0) == "na"


def test_novelty_group_boundaries_are_shared_constants():
    assert C.novelty_group(0.0) == "SEEN"
    assert C.novelty_group(1e-9) == "PARTIAL_NOVEL"
    assert C.novelty_group(0.25) == "PARTIAL_NOVEL"
    assert C.novelty_group(0.2500001) == "HIGH_NOVEL"
    assert C.novelty_group(float("nan")) == "UNRESOLVED"
    assert C.novelty_group(None) == "UNRESOLVED"

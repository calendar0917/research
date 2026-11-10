"""CPU tests for the pre-registered decision rule and support sets."""

from __future__ import annotations

import numpy as np

import rhic_common as C
import rhic_test as T


def test_support_sets_are_frozen_and_complete():
    assert len(C.HIGH_SUPPORT_TASKS) == 106
    assert len(C.LOW_SUPPORT_TASKS) == 22
    assert set(C.HIGH_SUPPORT_TASKS) | set(C.LOW_SUPPORT_TASKS) == set(range(128))
    assert not set(C.HIGH_SUPPORT_TASKS) & set(C.LOW_SUPPORT_TASKS)
    assert 45 in C.LOW_SUPPORT_SET  # unscorable on Test, never in the macro count


def test_choose_label_boundaries():
    # exact advance thresholds
    assert (
        T.choose_label(valid_best=0.280, test_best=0.270, high_delta_vs_cpc=0.003, improved_high_vs_cpc=70)
        == "PERFORMANCE_ADVANCE"
    )
    # below the advance bar but above the partial bar -> PARTIAL, never ADVANCE
    assert (
        T.choose_label(valid_best=0.2799, test_best=0.270, high_delta_vs_cpc=0.003, improved_high_vs_cpc=70)
        == "PARTIAL_ADVANCE"
    )
    assert (
        T.choose_label(valid_best=0.281, test_best=0.271, high_delta_vs_cpc=0.0029, improved_high_vs_cpc=80)
        == "PARTIAL_ADVANCE"
    )
    assert (
        T.choose_label(valid_best=0.281, test_best=0.271, high_delta_vs_cpc=0.004, improved_high_vs_cpc=69)
        == "PARTIAL_ADVANCE"
    )
    assert (
        T.choose_label(valid_best=0.276, test_best=0.264, high_delta_vs_cpc=0.001, improved_high_vs_cpc=50)
        == "NO_CLEAR_ADVANCE"
    )


def test_choose_label_partial_and_regression():
    assert (
        T.choose_label(valid_best=0.273, test_best=0.265, high_delta_vs_cpc=0.0, improved_high_vs_cpc=60)
        == "PARTIAL_ADVANCE"
    )
    assert (
        T.choose_label(valid_best=0.2689, test_best=0.2648, high_delta_vs_cpc=0.002, improved_high_vs_cpc=60)
        == "PARTIAL_ADVANCE"
    )
    assert (
        T.choose_label(valid_best=0.2629, test_best=0.2593, high_delta_vs_cpc=-0.01, improved_high_vs_cpc=30)
        == "REGRESSION"
    )
    assert (
        T.choose_label(valid_best=0.270, test_best=0.264, high_delta_vs_cpc=-0.004, improved_high_vs_cpc=40)
        == "REGRESSION"
    )
    assert (
        T.choose_label(valid_best=0.266, test_best=0.262, high_delta_vs_cpc=0.000, improved_high_vs_cpc=50)
        == "NO_CLEAR_ADVANCE"
    )


def test_choose_label_mechanism_invalid_takes_priority():
    assert (
        T.choose_label(
            valid_best=0.30,
            test_best=0.30,
            high_delta_vs_cpc=0.02,
            improved_high_vs_cpc=100,
            mechanism_valid=False,
        )
        == "MECHANISM_INVALID"
    )


def test_macro_scope_is_127_tasks():
    rng = np.random.default_rng(0)
    y = rng.integers(0, 2, size=(64, C.N_TASKS)).astype(np.float64)
    y[:, 45] = np.nan  # task 45 has no Test positives
    logits = rng.normal(size=(64, C.N_TASKS))
    rep = C.MET.per_task_ap(y, logits)
    assert rep["n_tasks_scored"] == 127
    assert rep["per_task"][45] is None

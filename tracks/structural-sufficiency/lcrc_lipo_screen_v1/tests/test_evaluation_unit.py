from __future__ import annotations

import numpy as np
import pytest

import evaluate
import monitor_labels
import screen_common as C


def test_error_metrics():
    y = np.array([0.0, 1.0, 2.0, 3.0])
    pred = np.array([1.0, 0.0, 3.0, 2.0])
    assert evaluate.rmse(pred, y) == pytest.approx(1.0)
    assert evaluate.mae(pred, y) == pytest.approx(1.0)
    m = evaluate.metrics(pred, y)
    assert m["bias"] == pytest.approx(0.0)
    assert m["r2"] == pytest.approx(0.2)
    assert m["max_abs_error"] == pytest.approx(1.0)


def test_bootstrap_ci_is_deterministic_and_ordered():
    values = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
    a = evaluate.bootstrap_ci_mean(values, 500, 0)
    b = evaluate.bootstrap_ci_mean(values, 500, 0)
    assert a == b
    assert a["ci95_low"] <= a["estimate"] <= a["ci95_high"]
    assert a["resamples"] == 500


def test_paired_delta_rmse_sign_and_identity():
    y = np.zeros(8)
    good = np.zeros(8)
    bad = np.ones(8)
    delta = evaluate.bootstrap_delta_rmse(bad, good, y, 500, 0)
    assert delta["estimate"] > 0
    assert delta["resamples"] == 500
    zero = evaluate.bootstrap_delta_rmse(good, good, y, 200, 0)
    assert zero["estimate"] == 0.0
    assert zero["ci95_low"] == zero["ci95_high"] == 0.0


def test_classification_rules_are_frozen():
    m = C.MATERIALITY_RMSE
    assert evaluate.classify_comparison(-m - 0.01, -m - 0.02, -0.001, m) == "A_MATERIALLY_BETTER"
    assert evaluate.classify_comparison(-m + 0.01, -0.04, 0.02, m) == "COMPETITIVE"
    assert evaluate.classify_comparison(0.0, -0.02, 0.02, m) == "COMPETITIVE"
    assert evaluate.classify_comparison(m + 0.01, 0.001, m + 0.02, m) == "A_MATERIALLY_WORSE"
    assert evaluate.classify_comparison(0.10, -0.01, 0.20, m) == "INCONCLUSIVE"
    assert evaluate.classify_s2([-0.04, -0.05, -0.03], -0.04, -0.01, m) == "CLEAR_INCREMENT"
    assert evaluate.classify_s2([-0.04, -0.05, -0.03], -0.02, -0.01, m) == "SUGGESTIVE_NOT_MATERIAL"
    assert evaluate.classify_s2([-0.04, 0.01, -0.03], -0.02, -0.01, m) == "NO_CLEAR_INCREMENT"
    assert evaluate.classify_s2([-0.04, -0.05, -0.03], -0.04, 0.01, m) == "NO_CLEAR_INCREMENT"


def test_case_rules():
    assert evaluate.classify_case(True, "CLEAR_INCREMENT", "COMPETITIVE", "COMPETITIVE").startswith("CASE_1")
    assert evaluate.classify_case(True, "SUGGESTIVE_NOT_MATERIAL", "COMPETITIVE", "COMPETITIVE").startswith("CASE_2")
    assert evaluate.classify_case(True, "NO_CLEAR_INCREMENT", "A_MATERIALLY_WORSE", "A_MATERIALLY_WORSE").startswith("CASE_3")
    assert evaluate.classify_case(False, "CLEAR_INCREMENT", "COMPETITIVE", "COMPETITIVE").startswith("CASE_3")
    # CASE_1 also requires not being materially worse than either reference.
    assert evaluate.classify_case(True, "CLEAR_INCREMENT", "A_MATERIALLY_WORSE", "COMPETITIVE").startswith("CASE_3")


def test_evaluate_is_blocked_before_prediction_freeze():
    assert not C.PREDICTION_MANIFEST.exists()
    with pytest.raises(monitor_labels.FreezeError):
        evaluate.evaluate()
    assert not C.PREDICTION_MANIFEST.exists()


def test_decision_code_has_no_seed_selection_path():
    source = (C.EXP_DIR / "code" / "evaluate.py").read_text(encoding="utf-8")
    for forbidden in ("select_seed", "choose_seed", "early_stop", "checkpoint_selection"):
        assert forbidden not in source
    # Best-seed optimism is reported as a diagnostic, never used for selection.
    assert "best_seed_minus_mean_rmse" in source

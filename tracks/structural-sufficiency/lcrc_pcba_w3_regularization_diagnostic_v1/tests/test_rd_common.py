"""The frozen experiment definition must not drift."""

import pytest

import rd_common as C


def test_two_runs_and_their_single_intervention_each():
    assert C.RUN_ORDER == ("W3_WD10", "W3_DO20")
    assert C.RUNS["W3_WD10"]["dropout"] == 0.10
    assert C.RUNS["W3_WD10"]["weight_decay"] == pytest.approx(0.010)
    assert C.RUNS["W3_DO20"]["dropout"] == 0.20
    assert C.RUNS["W3_DO20"]["weight_decay"] == pytest.approx(0.001)
    # only these two values may differ between the runs
    a, b = (C.RUNS[k] for k in C.RUN_ORDER)
    diff = {k for k in a if k not in ("assigned_pool", "role") and a[k] != b[k]}
    assert diff == {"dropout", "weight_decay"}


def test_shared_recipe_is_historical_w3():
    r = C.SHARED_RECIPE
    assert r["seed"] == 0
    assert r["epochs"] == 40
    assert r["batch_size"] == 128
    assert r["learning_rate"] == pytest.approx(3e-4)
    assert r["warmup_start_lr"] == pytest.approx(3e-5)
    assert r["end_lr"] == pytest.approx(1e-5)
    assert r["gradient_clip"] == pytest.approx(5.0)
    assert r["adam_betas"] == (0.9, 0.999)
    assert r["adam_eps"] == pytest.approx(1e-8)
    assert r["early_stopping"] is False
    assert r["batch_plan_sha256"].startswith("06f43729")
    assert r["steps_per_epoch"] == 2738
    assert r["total_steps"] == 109520


def test_lr_schedule_endpoints():
    lr = C.lr_audit()
    assert lr["step0"] == pytest.approx(3.0098612e-05, rel=1e-6)
    assert lr["warmup_last"] == pytest.approx(3e-4, rel=1e-12)
    assert lr["step40e"] == pytest.approx(1e-5, rel=1e-12)


def test_recipe_diff_is_exactly_the_interventions():
    audit = C.assert_shared_recipe()
    assert audit["ok"] is True
    assert audit["differing_keys"] == sorted(C.INTERVENTION_KEYS + ("run", "role", "intervention"))


def test_parameter_targets():
    assert C.EXPECTED_PARAMETERS == 2120844
    assert C.EXPECTED_BASE_PARAMETERS == 1809408
    assert C.EXPECTED_READOUT_PARAMETERS == 311436
    assert C.EXPECTED_BASE_PARAMETERS + C.EXPECTED_READOUT_PARAMETERS == C.EXPECTED_PARAMETERS


def test_historical_reference_values():
    assert C.HISTORICAL["W3"]["parameters"] == 2120844
    assert C.HISTORICAL["W3"]["best_valid_ap"] == pytest.approx(0.26367025868637306)
    assert C.HISTORICAL["W3"]["valid_ap40"] == pytest.approx(0.259003625836404)
    assert C.HISTORICAL["W2"]["best_valid_ap"] == pytest.approx(0.26527331610028143)
    assert C.HISTORICAL["W3"]["best_epoch"] == 25
    assert C.HISTORICAL["W2"]["best_epoch"] == 24
    assert set(C.HISTORICAL["W3"]["valid_curve"]) == set(C.REPORT_EPOCHS)


def test_decision_thresholds_are_the_protocol_ones():
    assert C.DECISION["best_delta_min"] == pytest.approx(0.005)
    assert C.DECISION["ap40_delta_min"] == pytest.approx(0.003)
    assert C.DECISION["support_ge20_min_positive"] == 20
    assert C.DECISION["support_5_19_min_positive"] == 5


def test_split_bookkeeping():
    assert C.SPLITS == ("train", "valid", "test")
    assert C.ROW_COUNTS == {"train": 350343, "valid": 43793, "test": 43793}

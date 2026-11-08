from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

import tr_common as C
import tr_compare as CMP
import tr_test as TT
CC = TT.CC
CD = TT.CD


def _minimal_freeze() -> dict:
    freeze = {
        "schema": CC.SCHEMA_TEST_FREEZE,
        "round_id": C.ROUND_ID,
        "test_labels_read": False,
        "selection_manifest_sha256": "0" * 64,
        "test_rows_sha256": "1" * 64,
        "logical_models": {"m": {"prediction": "m", "checkpoint_sha256": "2" * 64}},
        "predictions": {
            "m": {
                "test_logits_path": "x.npy",
                "test_logits_sha256": "3" * 64,
                "test_logits_shape": [10, 128],
            }
        },
    }
    freeze["seal_sha256"] = CD.freeze_seal(freeze)
    return freeze


def test_freeze_seal_roundtrip_and_tamper_detection():
    freeze = _minimal_freeze()
    assert freeze["seal_sha256"] == CD.freeze_seal(freeze)
    tampered = json.loads(json.dumps(freeze))
    tampered["predictions"]["m"]["test_logits_sha256"] = "4" * 64
    assert tampered["seal_sha256"] != CD.freeze_seal(tampered)


def _fake_run(tmp_path: Path, *, test_evaluated: bool = False, epochs: int = 40) -> Path:
    checkpoint = tmp_path / "checkpoint_best.pt"
    checkpoint.write_bytes(b"fake checkpoint bytes")
    reports = {
        str(e): {"ap": 0.10 + 0.001 * e, "reason": "unit"} for e in range(1, epochs + 1)
    }
    # exact float tie at the best value: earlier epoch must win
    reports["3"]["ap"] = 0.50
    reports["5"]["ap"] = 0.50
    run = {
        "schema": C.SCHEMA_RUN,
        "round_id": C.ROUND_ID,
        "status": "COMPLETED",
        "formal_run": True,
        "epochs_run": epochs,
        "global_steps_run": epochs * C.STEPS_PER_EPOCH,
        "eval_epochs_completed": list(range(1, epochs + 1)),
        "best_epoch": 3,
        "best_valid_ap": 0.50,
        "epoch40_valid_ap": reports[str(epochs)]["ap"],
        "best_checkpoint": str(checkpoint),
        "best_checkpoint_sha256": C.sha256_file(checkpoint),
        "parameters": C.MODEL_PARAMETERS,
        "test_evaluated": test_evaluated,
        "batch_plan": {"global_plan_sha256": C.GLOBAL_PLAN_SHA256},
        "fixed_epoch_reports": reports,
        "calibration": {
            "lambda_rank": 1e-3,
            "sha256": "5" * 64,
            "g_bce_median": 1.0,
            "g_rank_median": 10.0,
            "ratio_raw": 0.025,
            "clipped": False,
        },
        "git_commit": "deadbeef",
    }
    path = tmp_path / "run.json"
    C.write_json(path, run)
    return path


def test_selection_uses_validation_only_and_earliest_tie(tmp_path):
    run_path = _fake_run(tmp_path)
    out = tmp_path / "selected_checkpoint.json"
    assert CMP.run_select(run_path, out) == 0
    selection = C.read_json(out)
    assert selection["schema"] == C.SCHEMA_SELECTION
    assert selection["test_read"] is False
    assert selection["selection_uses_test"] is False
    assert selection["best_epoch"] == 3
    assert selection["best_valid_ap"] == 0.50
    assert selection["checkpoint_sha256"] == C.sha256_file(tmp_path / "checkpoint_best.pt")
    entry = TT.selection_entry(out)
    assert entry["epoch"] == 3


def test_selection_rejects_a_run_that_read_test(tmp_path):
    run_path = _fake_run(tmp_path, test_evaluated=True)
    with pytest.raises(RuntimeError):
        CMP.run_select(run_path, tmp_path / "selected.json")


def test_selection_refuses_a_manifest_that_used_test(tmp_path):
    manifest = {
        "schema": C.SCHEMA_SELECTION,
        "round_id": C.ROUND_ID,
        "test_read": True,
        "selection_uses_test": True,
        "best_epoch": 1,
        "checkpoint": "x",
        "checkpoint_sha256": "0" * 64,
        "best_valid_ap": 0.2,
    }
    path = tmp_path / "selection.json"
    C.write_json(path, manifest)
    with pytest.raises(RuntimeError):
        TT.selection_entry(path)


def test_decision_labels_are_the_four_frozen_cases():
    high_ok = {
        "n_tasks": 108,
        "mean_delta": 0.003,
        "improved": 61,
    }
    high_weak = {"n_tasks": 108, "mean_delta": 0.001, "improved": 61}
    target = float(C.WORK_TARGET["test_ap_min"])
    above_valid = float(C.W2["best_valid_ap"]) + 0.001
    below_valid = float(C.W2["best_valid_ap"]) - 0.001

    d = CMP._decision(
        best_valid=above_valid, test_ap=target, high_support=high_ok, work_target=C.WORK_TARGET
    )
    assert d["label"] == "VALID_AND_TEST_GAIN"
    assert d["engineering_signal"] is True

    d = CMP._decision(
        best_valid=above_valid,
        test_ap=target - 0.001,
        high_support=high_ok,
        work_target=C.WORK_TARGET,
    )
    assert d["label"] == "VALID_ONLY"

    d = CMP._decision(
        best_valid=below_valid,
        test_ap=target,
        high_support=high_weak,
        work_target=C.WORK_TARGET,
    )
    assert d["label"] == "EXPOSED_TEST_ONLY_SIGNAL"
    assert d["engineering_signal"] is False

    d = CMP._decision(
        best_valid=below_valid,
        test_ap=target - 0.001,
        high_support=high_ok,
        work_target=C.WORK_TARGET,
    )
    assert d["label"] == "NO_CLEAR_GAIN"


def test_stratum_contribution_uses_the_127_task_macro_mean():
    rows = [
        {"delta": 0.1, "task": 1},
        {"delta": -0.1, "task": 2},
        {"delta": 0.2, "task": 3},
    ]
    out = TT._stratum(rows, "x")
    assert out["n_tasks"] == 3
    assert out["sum_delta"] == pytest.approx(0.2)
    assert out["contribution_to_macro_delta"] == pytest.approx(0.2 / 127.0)
    assert out["improved"] == 2
    assert out["degraded"] == 1


def test_historical_high_support_count_and_w2_identity():
    path = C.REPO / C.W2["test_scores_source"]
    if not path.is_file():
        pytest.skip("historical W2 test scores unavailable")
    scores = C.read_json(path)
    assert scores["models"]["W2_best_valid_epoch24"]["checkpoint_sha256"] == C.W2["checkpoint_best_sha256"]
    assert scores["models"]["W2_best_valid_epoch24"]["epoch"] == C.W2["best_epoch"]
    assert scores["models"]["W2_best_valid_epoch24"]["test_ap_official"] == C.W2["test_ap_official"]
    assert scores["support_strata"]["test_vs_w2_ge20"]["n_tasks"] == 108
    assert scores["n_tasks_scored_test"] == 127


def test_test_rows_are_the_official_sorted_rows(task):
    rows = np.asarray(task.test_rows, dtype=np.int64)
    assert rows.size == C.N_TEST
    assert np.array_equal(rows, np.sort(rows))
    assert np.intersect1d(rows, task.train_rows).size == 0
    assert np.intersect1d(rows, task.valid_rows).size == 0

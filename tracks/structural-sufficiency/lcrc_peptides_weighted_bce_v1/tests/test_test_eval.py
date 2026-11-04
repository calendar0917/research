"""Integrity tests for the authorized post-hoc Test evaluation.

These tests only read committed JSON/CSV artifacts.  They never open the Test
cache, Test labels or Test predictions.
"""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path

import pytest

ROUND = Path(__file__).resolve().parents[1]
REPO = ROUND.parents[2]
CODE = ROUND / "code"
sys.path.insert(0, str(CODE))

import wb_common as C  # noqa: E402

AUTH = ROUND / "analysis/test_access_authorization.json"
AMENDMENT = ROUND / "configs/test_access_amendment.json"
TEST_EVAL = ROUND / "analysis/test_eval.json"
TEST_GATES = ROUND / "analysis/test_gates.json"
TEST_COMPARISON = ROUND / "results/test_comparison.csv"
TEST_PER_LABEL = ROUND / "results/test_per_label_comparison.csv"

pytestmark = pytest.mark.skipif(not TEST_EVAL.exists(), reason="post-hoc Test evaluation not executed")


def _rows(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_authorization_and_checkpoint_freeze_order() -> None:
    auth = json.loads(AUTH.read_text(encoding="utf-8"))
    amendment = json.loads(AMENDMENT.read_text(encoding="utf-8"))
    data = json.loads(TEST_EVAL.read_text(encoding="utf-8"))
    assert auth["authorized"] is True
    assert auth["authorization_message"] == "我指的是test数据集，我现在授权访问，不然没办法比较"
    assert amendment["status"] == "AUTHORIZED_POSTHOC_TERMINAL_TEST_EVALUATION"
    assert data["authorization"]["authorized"] is True
    assert data["training_performed"] is False
    assert data["checkpoint_selection_used_test"] is False
    assert data["test_used_for_any_decision"] is False
    assert data["formal_training_test_evaluated"] is False
    assert data["test_evaluated"] is True
    assert data["result_label"] == "POSTHOC_AUTHORIZED_TERMINAL_TEST_EVALUATION"


def test_test_gates_all_pass() -> None:
    gates = json.loads(TEST_GATES.read_text(encoding="utf-8"))
    assert gates["all_passed"] is True
    assert len(gates["checks"]) == 21
    assert all(gates["checks"].values())
    assert gates["checks"]["authorization_present"] is True
    assert gates["checks"]["all_checkpoints_frozen_before_authorization"] is True
    assert gates["checks"]["fastv1_test_ap_reproduces_stored_within_1e-3"] is True
    assert gates["checks"]["test_row_identity_pass"] is True
    assert gates["checks"]["no_training_performed"] is True
    assert gates["checks"]["no_checkpoint_selection_used_test"] is True
    assert gates["checks"]["formal_training_run_kept_test_evaluated_false"] is True


def test_test_split_identity_recorded() -> None:
    data = json.loads(TEST_EVAL.read_text(encoding="utf-8"))
    split = data["test_split"]
    assert split["n"] == 2331
    assert split["cache_sha256"] == "ea897c1b51432be4ae8c50fdd6dc7c70fd0a973b9779d711bc3c9737715ffbf6"
    assert split["feature_spec_hash"] == "0eaba915b633642f618bcd7a273436df2305a10838e70be749721c2ea394cdc9"
    assert split["row_identity_status"] == "PASS"
    assert split["labels_array_sha256"] == "5059461eb517cba9ec7c4273ccc243518f63e5a3f3a576114b2ab571b9aac977"
    assert split["labels_file_sha256"] == "5ea5101be6d62ecbfff5f8ab83e201a18a5a5a8c3a9b079d87b4dcb1fe2ea97d"


def test_checkpoint_test_metrics_are_finite_and_reloaded() -> None:
    data = json.loads(TEST_EVAL.read_text(encoding="utf-8"))
    expected = {
        "fastv1_seed0_120ep": "1da869b8735872443c96bc5055d9628c58a63741911e93b106369e6efdba9a46",
        "diagnostic_ordinary_bce_240ep": "22643f093cc96dd3d8a81f916d7a6cc6e4bb38a91faf405eaf222fb7bf57634b",
        "weighted_bce_240ep": "8d9f7f6f78f1aed27c137ab59e3399bc2551f0ba283a62237acbc968b4c7d26b",
    }
    assert {row["id"] for row in data["checkpoints"]} == set(expected)
    for row in data["checkpoints"]:
        assert row["checkpoint_sha256"] == expected[row["id"]]
        assert 0.0 <= row["test_macro_ap"] <= 1.0
        assert math.isfinite(row["test_macro_ap"])
        assert row["test_ap_recheck_delta"] == 0.0
        assert row["test_probs_min"] >= 0.0 and row["test_probs_max"] <= 1.0
        assert len(row["per_label_test_ap"]) == 10
        assert len(row["per_label_valid_ap"]) == 10
        assert set(row["per_label_test_ap"]) == set(C.LABELS)
        assert row["stored_test_delta"] is None or row["stored_test_delta"] < 1e-3


def test_test_comparison_table_matches_json() -> None:
    data = json.loads(TEST_EVAL.read_text(encoding="utf-8"))
    rows = _rows(TEST_COMPARISON)
    assert len(rows) == 3
    by_method = {row["method"]: row for row in rows}
    ordinary = by_method["LCRC-C1 ordinary BCE 240ep"]
    weighted = by_method["LCRC-C1 weighted BCE 240ep"]
    assert float(ordinary["test_macro_ap"]) == pytest.approx(
        data["comparison"]["ordinary_bce_240ep"]["test_macro_ap"], abs=1e-15
    )
    assert float(weighted["test_macro_ap"]) == pytest.approx(
        data["comparison"]["weighted_bce_240ep"]["test_macro_ap"], abs=1e-15
    )
    delta = 100.0 * (float(weighted["test_macro_ap"]) - float(ordinary["test_macro_ap"]))
    assert delta == pytest.approx(data["comparison"]["weighted_minus_ordinary_test_pp"], abs=1e-12)
    assert delta < 0.0


def test_per_label_test_table_matches_json() -> None:
    data = json.loads(TEST_EVAL.read_text(encoding="utf-8"))
    rows = _rows(TEST_PER_LABEL)
    assert [row["label"] for row in rows] == list(C.LABELS)
    by_id = {row["id"]: row for row in data["checkpoints"]}
    ordinary = by_id["diagnostic_ordinary_bce_240ep"]
    weighted = by_id["weighted_bce_240ep"]
    for j, row in enumerate(rows):
        name = C.LABELS[j]
        assert int(row["label_index"]) == j
        assert float(row["ordinary_test_ap"]) == pytest.approx(ordinary["per_label_test_ap"][name], abs=1e-15)
        assert float(row["weighted_test_ap"]) == pytest.approx(weighted["per_label_test_ap"][name], abs=1e-15)
        expected = 100.0 * (
            ordinary["per_label_test_ap"][name] - weighted["per_label_test_ap"][name]
        )
        assert float(row["delta_test_ap_pp"]) == pytest.approx(-expected, abs=1e-12)
    assert all(float(row["delta_test_ap_pp"]) < 0.0 for row in rows)


def test_original_training_is_still_test_isolated() -> None:
    run = json.loads((ROUND / "runs/weighted_bce_seed0/run.json").read_text(encoding="utf-8"))
    assert run["test_evaluated"] is False
    assert run["test_access_attempts"] == []
    assert run["guard"]["attempts"] == []

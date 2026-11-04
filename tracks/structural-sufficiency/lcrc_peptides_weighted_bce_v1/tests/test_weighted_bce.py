"""Correctness and isolation tests for the weighted-BCE round.

CPU-only and fast.  They read the committed Train-only weight file and the
small official label arrays; no cache, checkpoint or Test artifact is opened.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest
import torch

ROUND = Path(__file__).resolve().parents[1]
REPO = ROUND.parents[2]
CODE = ROUND / "code"
sys.path.insert(0, str(CODE))

import wb_checks as CH  # noqa: E402
import wb_common as C  # noqa: E402
import wb_loss as L  # noqa: E402


def test_pos_weight_train_only_and_exact() -> None:
    detail = CH.check_pos_weight()
    assert detail["passed"] is True
    assert detail["counts_match_exact_ratio"] is True
    assert detail["valid_used"] is False
    assert detail["test_used"] is False
    assert len(detail["pos_weight"]) == 10
    assert detail["pos_weight"][8] > 40.0  # antiparasitic
    assert detail["pos_weight"][4] < 1.0  # antimicrobial


def test_pos_weight_label_order_dimension_dtype_device() -> None:
    detail = CH.check_label_order()
    assert detail["passed"] is True
    assert detail["label_order"] == list(C.LABELS)
    payload = C.read_json(C.POS_WEIGHT_JSON)
    weight = L.validate_pos_weight(payload["pos_weight"], C.N_LABELS, torch.device("cpu"))
    assert weight.shape == (C.N_LABELS,)
    assert weight.dtype == torch.float32
    assert weight.device.type == "cpu"
    assert bool(torch.isfinite(weight).all())
    with pytest.raises(L.WeightedBCEError):
        L.validate_pos_weight(payload["pos_weight"][:9], C.N_LABELS, torch.device("cpu"))
    with pytest.raises(L.WeightedBCEError):
        L.weighted_bce_with_logits(
            torch.zeros(2, C.N_LABELS),
            torch.zeros(2, C.N_LABELS),
            payload["pos_weight"][:9],
        )


def test_ones_equals_ordinary_bce_value_and_grad() -> None:
    detail = CH.check_ones_equivalence()
    assert detail["passed"] is True
    assert detail["value_delta_float64"] == 0.0
    assert detail["grad_delta_float64"] == 0.0
    assert detail["value_delta_float32"] <= 1e-6


def test_extreme_imbalance_matches_manual_formula() -> None:
    detail = CH.check_extreme_imbalance()
    assert detail["passed"] is True
    assert detail["manual_delta"] <= 1e-12
    assert detail["single_element_delta"] <= 1e-12


def test_missing_label_mask_preserves_shapes_and_broadcast() -> None:
    detail = CH.check_missing_label_mask()
    assert detail["passed"] is True
    assert detail["masked_manual_delta"] <= 1e-12
    assert detail["masked_gradient_zero_at_missing"] is True
    assert detail["elementwise_shape_preserved"] == [5, C.N_LABELS]
    assert detail["all_missing_loss"] == 0.0


def test_finite_gradient_and_optimizer_update() -> None:
    detail = CH.check_finite_grad_update()
    assert detail["passed"] is True
    assert detail["gradients_finite"] is True
    assert detail["parameters_updated"] is True
    assert np.isfinite(detail["loss"])
    assert np.isfinite(detail["grad_norm"])


def test_test_access_guard_blocks_and_records() -> None:
    detail = CH.check_guard()
    assert detail["passed"] is True
    assert detail["blocked"] is True
    assert detail["recorded_attempts"] == 1
    assert detail["guard_state_after"] == 0


def test_entrypoint_sources_do_not_reference_test_artifacts() -> None:
    detail = CH.check_source_no_test_access()
    assert detail["passed"] is True
    assert detail["hits"] == {}


def test_protocol_is_frozen_to_the_historical_recipe() -> None:
    protocol = C.read_json(C.PROTOCOL_JSON)
    assert protocol["round_id"] == C.ROUND_ID
    assert protocol["result_label"] == C.RESULT_LABEL
    assert protocol["single_intervention"]["changed"] == "training loss"
    training = protocol["training"]
    assert training["init"] == "from scratch with the frozen Fast v1 seed0 initialisation mapping"
    assert training["expected_parameters"] == 464739
    assert training["peak_lr"] == pytest.approx(3e-4)
    assert training["weight_decay"] == pytest.approx(1e-3)
    assert training["dropout"] == pytest.approx(0.1)
    assert training["batch_size"] == 64
    assert training["gradient_clip"] == pytest.approx(5.0)
    assert training["scheduler_horizon_epochs"] == 240
    assert training["max_epochs"] == 240
    assert training["min_epochs"] == 80
    assert training["patience"] == 40
    assert training["test_evaluated"] is False
    history = C.read_json(C.HISTORY_RUN_JSON)
    ref = protocol["historical_reference"]
    assert ref["best_epoch"] == history["best_epoch"] == 231
    assert float(ref["best_valid_macro_ap"]) == float(history["best_valid_macro_ap"])
    assert float(ref["train_macro_ap"]) == float(history["train_macro_ap_per_label_mean"])
    assert ref["checkpoint_sha256"] == history["checkpoint_sha256"]
    for forbidden in ("LCRC-B", "Focal", "Dice", "ECFP+MLP", "Test"):
        assert forbidden not in protocol["single_intervention"].get("to", "")


@pytest.mark.skipif(
    not (C.FORMAL_RUN_DIR / "run.json").exists(), reason="formal weighted-BCE run not executed yet"
)
def test_formal_run_is_consistent_and_test_isolated() -> None:
    run = C.read_json(C.FORMAL_RUN_DIR / "run.json")
    assert run["status"] == "complete_no_test"
    assert run["test_evaluated"] is False
    assert run["test_access_attempts"] == []
    assert run["guard"]["attempts"] == []
    assert run["parameters"] == 464739
    assert run["structural_gate"]["all_match"] is True
    assert run["batch_size"] == 64
    assert run["batch_fallback_used"] is False
    assert run["strict_single_variable_control"] is True
    assert run["recipe"]["max_epochs"] == 240
    assert run["recipe"]["min_epochs"] == 80
    assert run["recipe"]["patience"] == 40
    assert run["recipe"]["scheduler_horizon_epochs"] == 240
    assert run["recipe"]["peak_lr"] == pytest.approx(3e-4)
    assert run["recipe"]["weight_decay"] == pytest.approx(1e-3)
    weights = run["intervention"]["pos_weight"]
    assert len(weights) == 10 and all(np.isfinite(weights))
    assert run["intervention"]["verification"]["status"] == "PASS"
    assert run["intervention"]["verification"]["weights_exact_train_ratio"] is True

    curve = list(csv.DictReader(open(C.FORMAL_RUN_DIR / "curve.csv", encoding="utf-8")))
    assert [int(r["epoch"]) for r in curve] == list(range(1, len(curve) + 1))
    assert len(curve) <= 240
    metrics = [float(r["valid_metric"]) for r in curve]
    assert int(np.argmax(metrics)) + 1 == int(run["best_epoch"])
    assert abs(max(metrics) - float(run["best_valid_macro_ap"])) < 1e-12
    assert all(np.isfinite(float(r["train_loss"])) for r in curve)
    assert all(np.isfinite(float(r["grad_norm_mean"])) for r in curve)
    assert all(np.isfinite(float(r["grad_norm_max"])) for r in curve)
    best = max(curve, key=lambda r: float(r["valid_metric"]))
    assert best["is_best"] == "True"
    for j, name in enumerate(C.LABELS):
        vals = json.loads(best["valid_per_label_ap"])
        assert vals[j] == pytest.approx(run["per_label_valid_ap"][name], abs=1e-12)
    assert C.sha256_file(C.FORMAL_RUN_DIR / "checkpoint.pt") == run["checkpoint_sha256"]


@pytest.mark.skipif(
    not (ROUND / "results/per_label_comparison.csv").exists(),
    reason="analysis outputs not built yet",
)
def test_per_label_comparison_tables_are_consistent() -> None:
    with open(ROUND / "results/per_label_comparison.csv", encoding="utf-8", newline="") as fh:
        rows = list(csv.DictReader(fh))
    assert [r["label"] for r in rows] == list(C.LABELS)
    run = C.read_json(C.FORMAL_RUN_DIR / "run.json")
    for j, row in enumerate(rows):
        name = C.LABELS[j]
        assert int(row["label_index"]) == j
        assert float(row["pos_weight"]) == pytest.approx(run["intervention"]["pos_weight"][j], abs=0.0)
        assert float(row["new_valid_ap"]) == pytest.approx(run["per_label_valid_ap"][name], abs=1e-12)
        delta = 100.0 * (float(row["new_valid_ap"]) - float(row["history_valid_ap"]))
        assert float(row["delta_valid_ap_pp"]) == pytest.approx(delta, abs=1e-9)
    with open(ROUND / "results/validation_comparison.csv", encoding="utf-8", newline="") as fh:
        summary = list(csv.DictReader(fh))
    current = [r for r in summary if r["method"].startswith("LCRC-C1 weighted BCE")]
    assert len(current) == 1
    assert float(current[0]["valid_macro_ap"]) == pytest.approx(run["best_valid_macro_ap"], abs=1e-12)
    history = [r for r in summary if r["method"].startswith("LCRC-C1 ordinary BCE")]
    assert len(history) == 1
    assert float(history[0]["valid_macro_ap"]) == pytest.approx(0.606228818027094, abs=1e-15)

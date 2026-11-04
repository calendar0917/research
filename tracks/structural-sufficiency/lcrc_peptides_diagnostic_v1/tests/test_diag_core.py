"""Correctness gates for the Peptides-func diagnostic round.

These tests only read committed artifacts (plus the historical Fast v1 copies
under ``history/``); they never train and never touch Test data.
"""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import numpy as np
import pytest

ROUND = Path(__file__).resolve().parents[1]
REPO = ROUND.parents[2]
CODE = ROUND / "code"
sys.path.insert(0, str(CODE))

import diag_common as C  # noqa: E402

SKIP_NO_GPU_ARTIFACT = pytest.mark.skipif(
    not (ROUND / "results/ecfp_validation.csv").exists(), reason="P1 artifacts not built"
)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def test_history_curve_audit_reproduces_recorded_best() -> None:
    rows = _read_csv(ROUND / "analysis/learning_curve_summary.csv")
    assert len(rows) == 3
    for row in rows:
        seed = int(row["seed"])
        run = C.history_run(seed)
        assert int(row["best_epoch"]) == int(run["best_epoch"])
        assert abs(float(row["best_valid_mean_ap"]) - float(run["best_metric"])) < 1e-12
        assert int(row["epochs_run"]) == int(run["epochs_run"])
        curve = _read_csv(C.history_curve_path(seed))
        assert len(curve) == int(run["epochs_run"])
        metrics = [float(r["valid_metric"]) for r in curve]
        assert int(curve[int(np.argmax(metrics))]["epoch"]) == int(run["best_epoch"])
        assert C.sha256_file(C.history_curve_path(seed)) == run["curve_sha256"]


def test_snapshot_rows_have_all_requested_epochs() -> None:
    rows = _read_csv(ROUND / "analysis/learning_curves.csv")
    assert len(rows) == 3 * 7
    for seed in C.LEGACY_SEEDS:
        subset = [r for r in rows if int(r["seed"]) == seed]
        assert [int(r["epoch"]) for r in subset] == [1, 20, 40, 60, 80, 100, 120]
        assert all(r["present"] == "True" for r in subset)


def test_label_statistics_match_official_csv_and_fast_v1_arrays() -> None:
    rows = _read_csv(ROUND / "analysis/label_statistics.csv")
    assert [r["label"] for r in rows] == list(C.LABELS)
    with open(C.PEPTIDES_CSV, "r", encoding="utf-8", newline="") as fh:
        raw = list(csv.DictReader(fh))
    splits = C.read_json(C.PEPTIDES_SPLITS)
    y = np.asarray([[int(r[name]) for name in C.LABELS] for r in raw], dtype=np.int64)
    y_train = y[np.asarray(splits["train"], dtype=np.int64)]
    y_valid = y[np.asarray(splits["valid"], dtype=np.int64)]
    for j, row in enumerate(rows):
        assert int(row["train_pos"]) == int(y_train[:, j].sum())
        assert int(row["valid_pos"]) == int(y_valid[:, j].sum())
        assert int(row["train_neg"]) == int(y_train.shape[0]) - int(y_train[:, j].sum())
        assert int(row["valid_neg"]) == int(y_valid.shape[0]) - int(y_valid[:, j].sum())
        assert abs(float(row["train_constant_ap_baseline"]) - float(y_train[:, j].mean())) < 1e-12
        assert abs(float(row["valid_constant_ap_baseline"]) - float(y_valid[:, j].mean())) < 1e-12
    assert np.array_equal(y_train, np.load(C.PREV_ANALYSIS / "peptides_func_y_train.npy"))
    assert np.array_equal(y_valid, np.load(C.PREV_ANALYSIS / "peptides_func_y_valid.npy"))


def test_row_identity_gate_passed() -> None:
    gate = C.read_json(ROUND / "analysis/row_identity.json")
    assert gate["status"] == "PASS"
    assert gate["checks"]["split_sizes"] == {"train": 10873, "valid": 2331, "test": 2331}
    assert gate["checks"]["split_indices_disjoint_and_complete"] is True
    for split in C.SPLITS:
        info = gate["per_split"][split]
        assert info["n_atom_feature_mismatch_rows"] == 0
        assert info["n_bond_feature_mismatch_rows"] == 0
        assert info["n_label_mismatch_rows"] == 0
        assert info["n_canonical_smiles_mismatch_rows"] == 0
        assert info["n_rows"] == C.SPLIT_SIZES[split]


def test_frozen_checkpoint_eval_reproduces_history_and_label_rates() -> None:
    rows = _read_csv(ROUND / "results/p0_frozen_checkpoints_per_label.csv")
    assert len(rows) == 3 * 10
    summary = _read_csv(ROUND / "results/p0_frozen_checkpoints_summary.csv")
    assert len(summary) == 3
    for row in summary:
        seed = int(row["seed"])
        stored = float(row["stored_best_valid_macro_ap"])
        recomputed = float(row["valid_macro_ap"])
        # CPU reload of a GPU-trained checkpoint: AP can move at the 1e-5 level
        # (CUDA index_add_ reductions are not bitwise deterministic).
        assert abs(recomputed - stored) < 1e-3
        assert float(row["valid_macro_ap_helper_delta"]) < 1e-9
        assert row["checkpoint_sha256"] == C.history_run(seed)["checkpoint_sha256"]
        subset = [r for r in rows if int(r["seed"]) == seed]
        assert abs(np.mean([float(r["valid_ap"]) for r in subset]) - recomputed) < 1e-12
        pred_path = ROUND / f"runs/p0_frozen_eval/seed{seed}/train_valid_predictions.npz"
        if not pred_path.exists():
            continue  # per-run npz under runs/ is local-only evidence
        pred = np.load(pred_path)
        assert pred["valid_probs"].shape == (2331, 10)
        assert pred["train_probs"].shape == (10873, 10)
        assert pred["valid_probs"].min() >= 0.0 and pred["valid_probs"].max() <= 1.0
        from sklearn.metrics import average_precision_score

        for j, r in enumerate(subset):
            assert abs(float(r["valid_ap"]) - average_precision_score(pred["y_valid"][:, j], pred["valid_probs"][:, j])) < 1e-12
            assert abs(float(r["train_ap"]) - average_precision_score(pred["y_train"][:, j], pred["train_probs"][:, j])) < 1e-12


@SKIP_NO_GPU_ARTIFACT
def test_ecfp_artifacts_are_consistent() -> None:
    manifest = C.read_json(ROUND / "analysis/ecfp_manifest.json")
    assert manifest["test_rows_read"] == 0
    assert manifest["checks"]["sizes_ok"] is True
    assert manifest["checks"]["train_labels_equal_fast_v1"] is True
    assert manifest["checks"]["valid_labels_equal_fast_v1"] is True
    assert manifest["fingerprint"]["features"]["shape_train"] == [10873, 2048]
    assert manifest["fingerprint"]["features"]["shape_valid"] == [2331, 2048]
    assert manifest["fingerprint"]["kwargs"] == {"count": True, "radius": 2, "fp_size": 2048}
    assert manifest["classifier"]["kwargs"]["n_estimators"] == 500
    assert manifest["classifier"]["kwargs"]["is_unbalance"] is True
    assert manifest["classifier"]["kwargs"]["random_state"] == 0
    assert manifest["classifier"]["kwargs"]["n_jobs"] == 8
    assert all(manifest["fingerprint"]["purity"].values())

    per_label = _read_csv(ROUND / "results/ecfp_per_label.csv")
    assert [r["label"] for r in per_label] == list(C.LABELS)
    pred = np.load(ROUND / "results/ecfp_valid_predictions.npz")
    assert pred["probs"].shape == (2331, 10)
    assert pred["probs"].min() >= 0 and pred["probs"].max() <= 1
    assert np.array_equal(pred["y_true"], np.load(C.PREV_ANALYSIS / "peptides_func_y_valid.npy"))
    from sklearn.metrics import average_precision_score

    aps = []
    for j, row in enumerate(per_label):
        ap = average_precision_score(pred["y_true"][:, j], pred["probs"][:, j])
        assert abs(ap - float(row["valid_ap"])) < 1e-12
        aps.append(ap)
    summary = _read_csv(ROUND / "results/ecfp_validation.csv")[0]
    assert abs(float(summary["valid_macro_ap"]) - float(np.mean(aps))) < 1e-12


def test_p2_source_never_references_test_artifacts() -> None:
    source = (ROUND / "code/diag_p2_train.py").read_text(encoding="utf-8")
    for token in ("peptides_func_test", "peptides_func_y_test", "test_predictions", "y_test", "_peptide_test_loader"):
        assert token not in source, f"P2 training source mentions a Test artifact: {token}"
    assert "install_test_access_guard" in source


@pytest.mark.skipif(
    not (ROUND / "runs/peptide_c1_seed0_240ep/run.json").exists(), reason="P2 run not executed yet"
)
def test_p2_run_is_consistent() -> None:
    run = C.read_json(ROUND / "runs/peptide_c1_seed0_240ep/run.json")
    assert run["status"] == "complete_no_test"
    assert run["test_evaluated"] is False
    assert run["guard"]["attempts"] == []
    assert run["parameters"] == 464739
    assert run["structural_gate"]["init_state_hash_match_history_seed0"] is True
    assert run["recipe"]["max_epochs"] == 240
    assert run["recipe"]["min_epochs"] == 80
    assert run["recipe"]["patience"] == 40
    assert run["recipe"]["scheduler_horizon_epochs"] == 240
    assert run["recipe"]["peak_lr"] == pytest.approx(3e-4)
    assert run["recipe"]["weight_decay"] == pytest.approx(1e-3)
    assert run["batch_size"] == 64
    curve = _read_csv(ROUND / "runs/peptide_c1_seed0_240ep/curve.csv")
    assert [int(r["epoch"]) for r in curve] == list(range(1, len(curve) + 1))
    metrics = [float(r["valid_metric"]) for r in curve]
    assert int(np.argmax(metrics)) + 1 == int(run["best_epoch"])
    assert abs(max(metrics) - float(run["best_valid_macro_ap"])) < 1e-12
    first = [float(r["valid_metric"]) for r in curve if int(r["epoch"]) <= 120]
    assert abs(max(first) - float(run["phase_first_120"]["best_valid_macro_ap"])) < 1e-12
    second = [float(r["valid_metric"]) for r in curve if int(r["epoch"]) > 120]
    if second:
        assert abs(max(second) - float(run["phase_121_240"]["best_valid_macro_ap"])) < 1e-12
    else:
        assert run["phase_121_240"] is None
    assert C.sha256_file(ROUND / "runs/peptide_c1_seed0_240ep/checkpoint.pt") == run["checkpoint_sha256"]
    pred_path = ROUND / "runs/peptide_c1_seed0_240ep/train_valid_predictions.npz"
    if pred_path.exists():
        pred = np.load(pred_path)
        assert pred["valid_probs"].shape == (2331, 10)
        assert pred["train_probs"].shape == (10873, 10)
        assert float(pred["valid_probs"].min()) >= 0 and float(pred["valid_probs"].max()) <= 1

"""Frozen-code integrity plus post-run artifact consistency.

The artifact checks are skipped when ``analysis/`` has not been produced yet; they
run after ``code/nb_run.py`` and verify that the committed numbers agree with each
other (fold hashes, gates, out-of-fold predictions, decision string).
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path

import numpy as np
import pytest

import nb_common as C


def test_reused_frozen_modules_are_unmodified():
    paths = list(C.REUSED_MODULES)
    out = subprocess.run(
        ["git", "status", "--porcelain", "--", *paths],
        cwd=C.REPO,
        capture_output=True,
        text=True,
        check=True,
    )
    assert out.stdout.strip() == "", f"frozen modules modified: {out.stdout}"
    for rel in paths:
        assert (C.REPO / rel).exists(), rel
        assert C.sha256_file(C.REPO / rel)


def test_round_uses_no_gpu_and_no_training():
    assert C.GPU_BUDGET_HOURS == 0.0
    assert C.FORWARD_VARIANTS == ("A0", "C1")
    assert C.FORWARD_SEEDS == (0, 1, 2)
    assert len(C.BRANCHES) == 3
    assert C.BRANCHES == ("BASE", "BASE_NULL", "BASE_REAL")


def test_no_optimizer_or_training_import_in_the_rounds_source():
    for path in sorted(C.CODE_DIR.glob("*.py")):
        text = path.read_text(encoding="utf-8")
        for forbidden in ("torch.optim", "loss.backward(", "optimizer.step(", "AdamW"):
            assert forbidden not in text, f"{path.name} contains {forbidden}"


ANALYSIS = C.ANALYSIS


@pytest.fixture(scope="module")
def artifacts():
    names = (
        "structural_witness.json",
        "feature_margin_checks.json",
        "folds.json",
        "metrics.json",
        "bootstrap.json",
        "provenance.json",
        "cpu_ledger.json",
        "oof_predictions.npz",
    )
    payload = {}
    for name in names:
        path = ANALYSIS / name
        if not path.exists():
            pytest.skip(f"artifact {name} not produced yet")
        payload[name] = path
    return payload


def test_structural_witness_artifact(artifacts):
    w = json.loads(artifacts["structural_witness.json"].read_text())
    assert w["labels_read"] is False
    assert w["training_performed"] is False
    assert w["forward"]["A0_s0"]["h_row_matching"]["identical"] is True
    assert all(w["gates"].values()), [k for k, v in w["gates"].items() if not v]
    assert w["normalisation"]["matches_executed_dissection_run"] is True
    for key, rec in w["forward"].items():
        assert rec["y_abs_diff"] <= C.PREDICTION_TOL, key
        assert rec["renumber_invariance"]["pass"] is True, key


def test_fold_artifact_matches_its_own_hashes(artifacts):
    folds = json.loads(artifacts["folds.json"].read_text())
    assert folds["status"] == "OK"
    assert folds["outer_valid_partition_covers_all_rows"] is True
    seen: list[int] = []
    for f in folds["folds"]:
        assert C.rows_sha256(f["train_rows"]) == f["train"]["index_sha256"]
        assert C.rows_sha256(f["valid_rows"]) == f["valid"]["index_sha256"]
        seen.extend(f["valid_rows"])
    assert sorted(seen) == list(range(folds["n_train"]))
    assert folds["n_positive"] + folds["n_negative"] == folds["n_train"]


def test_metrics_artifact_is_internally_consistent(artifacts):
    m = json.loads(artifacts["metrics.json"].read_text())["metrics"]
    for branch in C.BRANCHES:
        per_fold = m["per_fold"]["roc_auc"][branch]
        assert len(per_fold) == 3
        assert m["mean_roc_auc"][branch] == pytest.approx(float(np.mean(per_fold)))
        assert m["mean_average_precision"][branch] == pytest.approx(
            float(np.mean(m["per_fold"]["average_precision"][branch]))
        )
    per_fold_diff = m["per_fold_roc_auc_diff"]["REAL_minus_NULL"]
    assert len(per_fold_diff) == 3
    assert m["mean_roc_auc_diff"]["REAL_minus_NULL"] == pytest.approx(float(np.mean(per_fold_diff)))
    assert m["estimator"]["solver"] == "liblinear"
    assert m["C_grid"] == list(C.C_GRID)


def test_gates_in_metrics_match_the_bootstrap_and_decision(artifacts):
    payload = json.loads(artifacts["metrics.json"].read_text())
    gates = payload["gates"]
    boot = json.loads(artifacts["bootstrap.json"].read_text())
    m = payload["metrics"]
    assert gates["checks"]["g1_mean_auc_real_minus_null_ge_0.015"]["value"] == pytest.approx(
        m["mean_roc_auc_diff"]["REAL_minus_NULL"]
    )
    assert gates["checks"]["g5_bootstrap_ci95_lower_gt_0"]["value"] == pytest.approx(
        boot["REAL_minus_NULL"]["ci95_lower"]
    )
    assert boot["n_valid_replicates"] == (C.BOOTSTRAP_REPLICATES if boot["status"] == "OK" else boot["n_valid_replicates"])
    assert gates["decision"] in {
        "PROMOTE_INFORMATION_SIGNAL",
        "WEAK_OR_INCONSISTENT_SIGNAL",
        "NO_OBSERVED_INCREMENT",
        "INFEASIBLE",
    }
    per_fold = m["per_fold_roc_auc_diff"]["REAL_minus_NULL"]
    expected_all_positive = all(x > 0 for x in per_fold)
    assert gates["checks"]["g2_all_fold_auc_real_minus_null_gt_0"]["passed"] is expected_all_positive


def test_oof_predictions_cover_every_train_row(artifacts):
    z = np.load(artifacts["oof_predictions.npz"], allow_pickle=True)
    y = z["y"]
    assert y.shape == (C.OGB_TRAIN_ROWS,)
    covered = np.zeros(y.size, dtype=bool)
    for fold in range(C.OUTER_SPLITS):
        rows = z[f"fold{fold}_valid_rows"]
        covered[rows] = True
        assert np.array_equal(z[f"fold{fold}_y"], y[rows])
        for branch in C.BRANCHES:
            pred = z[f"fold{fold}_pred_{branch}"]
            assert pred.shape == rows.shape
            assert np.isfinite(pred).all()
            assert pred.min() >= 0.0 and pred.max() <= 1.0
            assert np.allclose(z[f"oof_{branch}"][rows], pred)
    assert covered.all()
    assert np.isfinite(z["oof_BASE"]).all()


def test_cpu_ledger_is_within_budget(artifacts):
    ledger = json.loads(artifacts["cpu_ledger.json"].read_text())
    assert ledger["gpu_hours_used"] == 0.0
    assert ledger["budget_ok"] is True
    assert ledger["cpu_wall_hours_total"] <= C.CPU_SOFT_CAP_HOURS
    assert ledger["steps"]


def test_provenance_records_everything_required(artifacts):
    prov = json.loads(artifacts["provenance.json"].read_text())
    assert prov["git_head"]
    assert prov["reused_modules"] and prov["code"]
    env = prov["environment"]
    for key in ("rdkit", "torch", "scikit_learn", "numpy", "scipy"):
        assert env[key]
    assert prov["feature_spec_hash"] == C.FEATURE_SPEC_HASH
    assert prov["frozen_design"]["branches"] == list(C.BRANCHES)
    assert prov["frozen_design"]["gates"]["real_minus_null_auc_mean"] == 0.015
    if "data" in prov:
        assert prov["data"]["valid_test_labels_read"] is False


def test_feature_margin_artifact_reports_conservation(artifacts):
    fm = json.loads(artifacts["feature_margin_checks.json"].read_text())
    margins = fm["margin_checks"]
    assert margins["row_margin_conserved"] is True
    assert margins["col_margin_conserved"] is True
    assert margins["total_conserved"] is True
    assert margins["raw_row_margin_matches_counts"] is True
    assert fm["coverage"]["n_molecules"] == C.OGB_TRAIN_ROWS
    assert fm["feature_definition"]["n_features"] == 126
    assert fm["feature_definition"]["distance_bins"] == list(C.DIST_BIN_LABELS)

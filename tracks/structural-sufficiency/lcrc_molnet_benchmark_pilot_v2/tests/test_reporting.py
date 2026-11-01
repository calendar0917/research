"""Tests for the reporting layer: literature grading, probe verdicts, table assembly."""

from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import pytest

CODE = Path(__file__).resolve().parents[1] / "code"
sys.path.insert(0, str(CODE))

import mb2_common as C  # noqa: E402
import mb2_literature as LIT  # noqa: E402
import mb2_probe as PROBE  # noqa: E402


# ---------------------------------------------------------------------------
# literature matrix
# ---------------------------------------------------------------------------
def test_literature_rows_are_well_formed():
    rows = LIT.build()
    assert len(rows) == len(LIT.ROWS)
    assert {r["grade"] for r in rows} <= {"A", "B", "C"}
    for row in rows:
        assert row["source_key"] in LIT.SOURCES
        assert row["value"] is not None
        assert row["locator"]
        assert row["metric"] in {"RMSE", "ROC-AUC", "MAE"}
        # every row carries the hash of the document it came from
        assert len(row["source_sha256"]) == 64


def test_literature_covers_all_seven_datasets():
    rows = LIT.build()
    seen = {r["dataset"] for r in rows}
    assert seen == set(C.DATASET_ORDER)


def test_only_ogbg_molhiv_claims_an_exact_split_match():
    rows = LIT.build()
    exact = {r["dataset"] for r in rows if r["grade"] == "A"}
    assert exact == {"ogbg-molhiv"}
    # and the claim is consistent with the recorded flag
    assert LIT.EXACT_MATCH_FOUND["ogbg-molhiv"].startswith("YES")
    for dataset in C.DATASET_ORDER:
        if dataset != "ogbg-molhiv":
            assert LIT.EXACT_MATCH_FOUND[dataset] == "NO_EXACT_MATCH_FOUND"


def test_literature_matrix_csv_roundtrip(tmp_path):
    rows = LIT.build()
    out = tmp_path / "lit.csv"
    with out.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LIT.COLUMNS)
        writer.writeheader()
        writer.writerows(rows)
    back = list(csv.DictReader(out.open(encoding="utf-8")))
    assert len(back) == len(rows)
    assert list(back[0]) == LIT.COLUMNS
    assert back[0]["dataset"] == rows[0]["dataset"]


# ---------------------------------------------------------------------------
# probes
# ---------------------------------------------------------------------------
def test_probe_allowlist_rejects_unregistered_overrides():
    import mb2_train as TRAIN

    with pytest.raises(ValueError):
        C.recipe_for({"learning_rate": 1e-3})
    with pytest.raises(SystemExit):
        TRAIN.resolve_recipe("ESOL", "fs_dropout00")  # registered for FreeSolv only
    with pytest.raises(SystemExit):
        TRAIN.resolve_recipe("BACE", "nope")
    recipe, record = TRAIN.resolve_recipe("FreeSolv", "fs_dropout00")
    assert recipe["dropout"] == 0.0
    assert record["effective_differences_vs_R3"] == {"dropout": {"R3": 0.1, "probe": 0.0}}
    base, record = TRAIN.resolve_recipe("FreeSolv", None)
    assert base["dropout"] == 0.10 and record["probe"] is False


def test_probe_candidates_are_the_four_registered_ones():
    assert C.PROBE_ORDER == ("fs_dropout00", "fs_dropout20", "bace_dropout20", "bace_wd1e2")
    assert C.PROBE_CANDIDATES["fs_dropout00"]["dataset"] == "FreeSolv"
    assert C.PROBE_CANDIDATES["bace_wd1e2"]["dataset"] == "BACE"
    for name in C.PROBE_ORDER:
        overrides = C.PROBE_CANDIDATES[name]["overrides"]
        assert len(overrides) == 1


def test_probe_table_refuses_a_test_reading_record(tmp_path):
    """A probe record that claims Test access must not enter the probe table.

    The fixture is written under ``tmp_path`` -- never under ``analysis/runs`` --
    so a test can never touch a real run artifact.
    """
    run = tmp_path / "probe_fs_dropout00_s0"
    run.mkdir()
    payload = {
        "dataset": "FreeSolv",
        "task": "regression",
        "seed": 0,
        "data": {"test_labels_read": True, "test_cache_built": True},
        "metrics": {"valid": {"rmse": 1.0, "mae": 0.8}, "train": {"rmse": 0.5, "mae": 0.3}},
        "recipe": {"dropout": 0.0, "weight_decay": 1e-3, "epochs_run": 360, "overrides_vs_R3": {"dropout": 0.0}},
        "model": {"dropout": 0.0},
        "checkpoint": {"best_epoch": 3, "best_metric_name": "valid_rmse", "curve_path": "x.csv", "sha256": "0" * 64},
        "timing": {"train_wall_seconds": 1.0, "process_wall_seconds": 2.0},
        "gpu": {"peak_memory_allocated_bytes": 0},
        "probe": {"probe": True, "name": "fs_dropout00"},
    }
    path = run / "train.json"
    path.write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(SystemExit):
        PROBE._from_train_json(path, dataset="FreeSolv", config="fs_dropout00", kind="validation_only_probe")


def test_probe_verdicts_use_the_task_direction(tmp_path):
    """A positive RMSE delta is worse, a positive ROC-AUC delta is better."""

    def row(task, value, base):
        return {
            "dataset": "X",
            "task": task,
            "config": "p",
            "kind": "validation_only_probe",
            "valid_primary_value": value,
            "valid_primary_metric": "valid_rmse_raw" if task == "regression" else "valid_roc_auc",
            "delta_valid_vs_base": value - base,
            "delta_relative_vs_base": (value - base) / base,
        }

    verdict, _ = PROBE._verdict(row("regression", 1.10, 1.00))
    assert verdict.startswith("no_validation_improvement")
    verdict, _ = PROBE._verdict(row("regression", 0.90, 1.00))
    assert verdict.startswith("large_validation_gain")
    verdict, _ = PROBE._verdict(row("binary", 0.90, 0.80))
    assert verdict.startswith("large_validation_gain")
    verdict, _ = PROBE._verdict(row("binary", 0.79, 0.80))
    assert verdict.startswith("no_validation_improvement")

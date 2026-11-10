"""Decision rules and the comparison pipeline on synthetic run directories."""

import csv
import json
import math
from pathlib import Path

import pytest

import rd_common as C
import rd_compare as R


def test_decide_promising():
    label, _ = R.decide(
        delta_best=0.006,
        delta_40=0.004,
        test_delta_best=-0.001,
        train_drop_40=-0.05,
        strat_valid_best={"support_ge20": {"mean_delta": 0.004, "improved": 70, "n": 107}},
        longest_above=9,
    )
    assert label == "PROMISING_SINGLE_SEED_SIGNAL"


def test_decide_fit_suppression():
    label, _ = R.decide(
        delta_best=0.001,
        delta_40=-0.001,
        test_delta_best=0.0,
        train_drop_40=-0.10,
        strat_valid_best={"support_ge20": {"mean_delta": 0.0, "improved": 50, "n": 107}},
        longest_above=3,
    )
    assert label == "FIT_SUPPRESSION_WITHOUT_GENERALIZATION_GAIN"


def test_decide_peak_only():
    label, _ = R.decide(
        delta_best=0.004,
        delta_40=-0.002,
        test_delta_best=0.0,
        train_drop_40=-0.02,
        strat_valid_best={"support_ge20": {"mean_delta": 0.002, "improved": 60, "n": 107}},
        longest_above=6,
    )
    assert label == "PEAK_ONLY_OR_UNSTABLE_SIGNAL"


def test_decide_narrow_peak_is_peak_only():
    label, _ = R.decide(
        delta_best=0.002,
        delta_40=0.0019,
        test_delta_best=0.0,
        train_drop_40=-0.01,
        strat_valid_best={"support_ge20": {"mean_delta": 0.001, "improved": 60, "n": 107}},
        longest_above=2,
    )
    assert label == "PEAK_ONLY_OR_UNSTABLE_SIGNAL"


def test_decide_no_clear_gain():
    label, _ = R.decide(
        delta_best=0.001,
        delta_40=0.001,
        test_delta_best=0.0,
        train_drop_40=-0.01,
        strat_valid_best={"support_ge20": {"mean_delta": 0.0005, "improved": 55, "n": 107}},
        longest_above=0,
    )
    assert label == "NO_CLEAR_GAIN"


def test_decide_mixed_when_big_but_not_promising():
    label, _ = R.decide(
        delta_best=0.004,
        delta_40=0.0035,
        test_delta_best=-0.01,
        train_drop_40=-0.01,
        strat_valid_best={"support_ge20": {"mean_delta": 0.003, "improved": 70, "n": 107}},
        longest_above=8,
    )
    assert label == "MIXED_SIGNAL"


def _write_fake_run(root: Path, key: str, *, best_epoch: int, valid_best: float, valid_40: float) -> None:
    run_dir = root / key
    run_dir.mkdir(parents=True, exist_ok=True)
    epoch_fields = [
        "epoch", "split", "rows", "ap_official", "ap_sklearn", "ap_abs_delta", "bce",
        "n_tasks_scored", "n_tasks_with_observed_labels", "observed_entries", "positive_entries",
        "infer_seconds", "metric_seconds", "eval_seconds", "global_step", "learning_rate",
        "best_epoch_so_far", "best_valid_ap_so_far", "is_best_epoch", "logits_sha256",
        "probs_sha256", "ap_source",
    ]
    task_fields = [
        "epoch", "split", "task", "ap", "observed", "positive", "negative", "train_positive",
        "valid_positive", "stratum",
    ]
    with open(run_dir / "epoch_metrics.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=epoch_fields)
        w.writeheader()
        for e in range(1, 41):
            for split in C.SPLITS:
                ap = (valid_best if e == best_epoch else valid_40) if split == "valid" else 0.2
                w.writerow(
                    {k: 0 for k in epoch_fields}
                    | {
                        "epoch": e,
                        "split": split,
                        "rows": C.ROW_COUNTS[split],
                        "ap_official": ap,
                        "ap_sklearn": ap,
                        "ap_abs_delta": 0.0,
                        "n_tasks_scored": 127 if split != "train" else 128,
                        "logits_sha256": "x",
                        "probs_sha256": "y",
                        "ap_source": "OGB official Evaluator(ogbg-molpcba)",
                    }
                )
    with open(run_dir / "per_task_ap.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=task_fields)
        w.writeheader()
        for e in range(1, 41):
            for split in C.SPLITS:
                for t in range(C.N_TASKS):
                    ap = "" if (split != "train" and t == 45) else 0.2
                    w.writerow(
                        {
                            "epoch": e,
                            "split": split,
                            "task": t,
                            "ap": ap,
                            "observed": 10,
                            "positive": 3,
                            "negative": 7,
                            "train_positive": 30,
                            "valid_positive": 3,
                            "stratum": "support_lt5",
                        }
                    )
    with open(run_dir / "epoch_log.csv", "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=["epoch", "train_bce"])
        w.writeheader()
        for e in range(1, 41):
            w.writerow({"epoch": e, "train_bce": 0.02})
    run = {
        "status": "COMPLETED",
        "parameters": 2120844,
        "dropout": C.RUNS[key]["dropout"],
        "weight_decay": C.RUNS[key]["weight_decay"],
        "best_epoch": best_epoch,
        "best_valid_ap": valid_best,
        "timing": {
            "gpu_hours": 2.0,
            "train_seconds": 4000.0,
            "eval_seconds_total": 3000.0,
            "per_split": {s: {} for s in C.SPLITS},
            "epoch_seconds_mean": 180.0,
            "steps_per_second_mean": 20.0,
        },
        "resources": {"gpu_peak_allocated_bytes": 1, "peak_rss_kb": 1},
        "eval_state_integrity_all_epochs": True,
        "best_checkpoint": f"runs/{key}/checkpoint_best_valid.pt",
        "best_checkpoint_sha256": "a",
        "final_checkpoint": f"runs/{key}/checkpoint_epoch40.pt",
        "final_checkpoint_sha256": "b",
        "last_checkpoint_sha256": "c",
    }
    (run_dir / "run.json").write_text(json.dumps(run))


def test_compare_pipeline_on_synthetic_runs(tmp_path, monkeypatch):
    runs_root = tmp_path / "runs"
    monkeypatch.setattr(C, "run_dir", lambda key: runs_root / key)
    for key, best_epoch, valid_best, valid_40 in (
        ("W3_WD10", 27, 0.2680, 0.2640),
        ("W3_DO20", 22, 0.2650, 0.2610),
    ):
        _write_fake_run(runs_root, key, best_epoch=best_epoch, valid_best=valid_best, valid_40=valid_40)
    out = tmp_path / "analysis"
    rc = R.main(["--runs", ",".join(C.RUN_ORDER), "--out-dir", str(out)])
    assert rc == 0
    comparison = json.loads((out / "comparison.json").read_text())
    assert set(comparison["runs"]) == set(C.RUN_ORDER)
    a = comparison["runs"]["W3_WD10"]
    assert a["best_epoch"] == 27
    assert a["delta_vs_w3"]["best"] == pytest.approx(0.2680 - C.AP_W3_BEST)
    assert a["delta_vs_w3"]["ap40"] == pytest.approx(0.2640 - C.AP_W3_40)
    assert a["report_epochs_ap"]["40"]["valid"] == pytest.approx(0.2640)
    assert a["peak_persistence"]["longest_consecutive"] >= 1
    assert a["gap"]["common_scorable_at_best"]["n_common_tasks"] == 127
    assert a["metric_cross_check"]["max_abs_delta_official_vs_sklearn"] == 0.0
    decision = json.loads((out / "decision.json").read_text())
    assert set(decision["runs"]) == set(C.RUN_ORDER)
    assert (out / "comparison.csv").is_file()
    assert (out / "curves_epoch.csv").is_file()
    assert (out / "per_task_delta.csv").is_file()
    lines = (out / "comparison.csv").read_text().strip().splitlines()
    assert len(lines) == 5  # header + W2 + W3 + two runs

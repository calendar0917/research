"""FGCR-v0 final artifact audit.

Cross-checks the committed evidence: data identity, topology gate, correctness
checks, preflight budget, best-Valid selection, per-epoch table, complete
predictions, checkpoint metrics and the report's decision.  This does not train
and does not change the model.
"""

from __future__ import annotations

import argparse
import csv
import subprocess
from pathlib import Path
from typing import Any

import numpy as np
import torch

import fgcr_common as C
import fgcr_data as D
import fgcr_model as M


def _read_metrics(path: Path) -> list[dict[str, Any]]:
    rows = []
    with path.open(encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            parsed: dict[str, Any] = {}
            for key, value in row.items():
                if key == "epoch":
                    parsed[key] = int(value)
                else:
                    parsed[key] = float(value)
            rows.append(parsed)
    return rows


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", default=str(C.ANALYSIS_DIR / "final_audit.json"))
    parser.add_argument("--run", default=str(C.RESULTS_DIR / "seed0"))
    args = parser.parse_args()
    run_dir = Path(args.run)
    cfg = C.load_config()
    checks: dict[str, Any] = {}

    bundle = D.load_bundle()
    checks["data"] = {
        "cache_sha256": bundle.cache_sha256,
        "split_sizes": [int(bundle.train_idx.size), int(bundle.valid_idx.size),
                        int(bundle.test_idx.size)],
        "n_graphs": int(bundle.n_graphs),
        "scaler": bundle.scaler,
        "pass": bool(bundle.cache_sha256 == C.CACHE_SHA256
                     and bundle.n_graphs == 4200
                     and [bundle.train_idx.size, bundle.valid_idx.size,
                          bundle.test_idx.size] == [3360, 420, 420]),
    }

    topology = C.read_json(C.ANALYSIS_DIR / "topology.json")
    checks["topology"] = {
        "status": topology.get("status"),
        "width_max": topology.get("width_max"),
        "width_median": topology.get("width_median"),
        "width_p95": topology.get("width_p95"),
        "width_p99": topology.get("width_p99"),
        "max_clique_state_table": topology.get("max_clique_state_table"),
        "n_graphs": topology.get("n_graphs"),
        "all_graphs_covered": topology.get("all_graphs_covered"),
        "pass": bool(topology.get("status") == "PASS"
                     and int(topology.get("width_max", 99)) <= 8
                     and int(topology.get("n_graphs", 0)) == 4200
                     and bool(topology.get("all_graphs_covered"))),
    }

    correctness = C.read_json(C.ANALYSIS_DIR / "correctness.json")
    checks["correctness"] = {
        "status": correctness.get("status"),
        "bruteforce_max_abs_error": correctness["checks"]["bruteforce"]["max_abs_error"],
        "zero_potential_pass": correctness["checks"]["zero_potential"]["pass"],
        "invariance_pass": correctness["checks"]["invariance"]["pass"],
        "real_batch_gradients_pass": correctness["checks"]["real_batch_gradients"]["pass"],
        "single_belief_calls": correctness["checks"]["single_belief_and_nondegeneracy"]
                              ["belief_calls_for_one_forward"],
        "pass": bool(correctness.get("status") == "PASS"),
    }

    preflight = C.read_json(C.ANALYSIS_DIR / "preflight_gpu.json")
    checks["preflight"] = {
        "status": preflight.get("status"),
        "n_epochs_run": preflight.get("n_epochs_run"),
        "projected_gpu_hours": preflight.get("projected_gpu_hours"),
        "budget_gpu_hours": preflight.get("budget_gpu_hours"),
        "peak_vram_gb": preflight.get("peak_vram_gb"),
        "pass": bool(preflight.get("budget_ok") is True
                     and float(preflight.get("projected_gpu_hours", 1e9)) <=
                     float(cfg["resource_budget"]["gpu_hours_total"])),
    }

    metrics = _read_metrics(run_dir / "metrics.csv")
    best_row = min(metrics, key=lambda row: row["valid_rmse"])
    run = C.read_json(run_dir / "run.json")
    checks["selection"] = {
        "n_epochs_in_csv": len(metrics),
        "argmin_valid_epoch": int(best_row["epoch"]),
        "run_best_epoch": int(run["best_epoch"]),
        "argmin_valid_rmse": float(best_row["valid_rmse"]),
        "run_best_valid_rmse": float(run["best_metrics"]["valid_rmse"]),
        "run_verdict": run.get("verdict"),
        "run_decision": run.get("decision"),
        "pass": bool(int(best_row["epoch"]) == int(run["best_epoch"])
                     and abs(float(best_row["valid_rmse"])
                             - float(run["best_metrics"]["valid_rmse"])) <= 1e-12
                     and run.get("verdict") in
                     ("PREDICTIVE_CANDIDATE", "WEAK_PREDICTIVE_SIGNAL",
                      "NO_COMPETITIVE_SIGNAL")),
    }

    predictions = np.load(run_dir / "predictions_best.npz")
    pred_checks: dict[str, Any] = {}
    pred_ok = True
    for split in ("train", "valid", "test"):
        row_ids = predictions[f"{split}_row_ids"]
        y_true = predictions[f"{split}_y_true"]
        y_pred = predictions[f"{split}_y_pred"]
        rmse = float(np.sqrt(np.mean((y_true - y_pred) ** 2)))
        mae = float(np.mean(np.abs(y_true - y_pred)))
        expected_rmse = float(run["best_metrics"][f"{split}_rmse"])
        expected_mae = float(run["best_metrics"][f"{split}_mae"])
        ok = (row_ids.size == (3360 if split == "train" else 420)
              and len(np.unique(row_ids)) == row_ids.size
              and np.isfinite(y_pred).all()
              and abs(rmse - expected_rmse) <= 1e-12
              and abs(mae - expected_mae) <= 1e-12)
        pred_ok = pred_ok and bool(ok)
        pred_checks[split] = {
            "n": int(row_ids.size), "rmse": rmse, "mae": mae,
            "expected_rmse": expected_rmse, "expected_mae": expected_mae,
            "pass": bool(ok),
        }
    all_ids = np.concatenate([predictions[f"{split}_row_ids"]
                              for split in ("train", "valid", "test")])
    pred_checks["union_row_ids"] = int(np.unique(all_ids).size)
    pred_checks["pass"] = bool(pred_ok and int(np.unique(all_ids).size) == 4200)
    checks["predictions"] = pred_checks

    checkpoint = torch.load(run_dir / "best.pt", map_location="cpu",
                            weights_only=False)
    state = checkpoint["state_dict"]
    model, info = M.build_model(bundle.atom_dims, bundle.bond_dims, seed=0,
                                config=cfg)
    model.load_state_dict(state, strict=True)
    checks["checkpoint"] = {
        "epoch": int(checkpoint["epoch"]),
        "parameters": info["parameters"],
        "parameter_total": int(info["parameters"]["total"]),
        "state_dict_matches_model": True,
        "pass": bool(int(checkpoint["epoch"]) == int(run["best_epoch"])
                     and int(info["parameters"]["total"]) ==
                     int(run["parameters"]["total"])),
    }

    reload_info = run.get("checkpoint_reload", {})
    checks["checkpoint_reload"] = {
        "ok": reload_info.get("ok"),
        "max_abs_pred_diff": reload_info.get("max_abs_pred_diff"),
        "pass": bool(reload_info.get("ok") is True),
    }

    report_text = (C.ROUND_DIR / "REPORT.md").read_text(encoding="utf-8")
    checks["report"] = {
        "contains_decision": "STOP_THIS_IMPLEMENTATION" in report_text,
        "contains_verdict": str(run.get("verdict")) in report_text,
        "contains_best_epoch": f"| {run['best_epoch']} |" in report_text,
        "contains_full_epoch_table": "## Appendix: complete per-epoch metrics" in report_text,
        "pass": bool("STOP_THIS_IMPLEMENTATION" in report_text
                     and str(run.get("verdict")) in report_text
                     and "## Appendix: complete per-epoch metrics" in report_text),
    }

    status = "PASS" if all(bool(v.get("pass")) for v in checks.values()) else "FAIL"
    out_rel = C.relative(Path(args.out))
    git_status = subprocess.run(
        ["git", "-C", str(C.REPO_ROOT), "status", "--porcelain"],
        capture_output=True, text=True, check=True,
    )
    dirty_other = []
    for line in git_status.stdout.splitlines():
        if not line.strip():
            continue
        path = line[3:].strip()
        if path == out_rel:
            continue
        dirty_other.append(line)
    payload = {
        "status": status,
        "checks": checks,
        "run": str(run_dir.relative_to(C.ROUND_DIR)),
        "commit": C.git_commit(),
        "dirty": bool(dirty_other),
        "dirty_files_excluding_audit_output": dirty_other,
        "environment": C.environment_snapshot(),
    }
    C.write_json(args.out, payload)
    print(f"STATUS={status}")
    for name, value in checks.items():
        print(f"  {name}: pass={value.get('pass')}")
    return 0 if status == "PASS" else 1


if __name__ == "__main__":
    raise SystemExit(main())
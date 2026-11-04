"""Formal consistency checks over the completed P1a/P1b runs.

Writes ``analysis/formal_consistency.json``.  These are post-run checks of the
artifacts, not a second model-selection step.
"""

from __future__ import annotations

import base64
import csv
import json
import math
from pathlib import Path
from typing import Any

import numpy as np

import pc_common as C

RUN_NAMES = {
    "P1A": "p1a_480_s0",
    "A0": "a0_s0",
    "A1": "a1_s0",
    "A2": "a2_s0",
    "A3": "a3_s0",
}
EXPECTED_EPOCHS = {"P1A": 480, "A0": 240, "A1": 240, "A2": 240, "A3": 240}


def load_run(name: str) -> dict[str, Any]:
    path = C.RUNS / name / "run.json"
    if not path.is_file():
        raise SystemExit(f"missing {path}")
    run = C.read_json(path)
    run["_name"] = name
    with open(C.RUNS / name / "curve.csv", "r", encoding="utf-8") as fh:
        run["_curve"] = list(csv.DictReader(fh))
    return run


def rr_meta(run: dict[str, Any]) -> dict[str, Any]:
    raw = run.get("rr_env", {}).get("RR_META_B64")
    if not raw:
        return {}
    return json.loads(base64.b64decode(raw).decode("utf-8"))


def main() -> int:
    runs = {arm: load_run(name) for arm, name in RUN_NAMES.items()}
    checks: dict[str, bool] = {}
    details: dict[str, Any] = {}

    for arm, run in runs.items():
        checks[f"{arm}_status_complete"] = run["status"] == "complete_no_test"
        checks[f"{arm}_no_test"] = run["test_evaluated"] is False
        checks[f"{arm}_guard_empty"] = not run.get("guard", {}).get("attempts")
        checks[f"{arm}_epochs"] = int(run["epochs_run"]) == EXPECTED_EPOCHS[arm]
        checks[f"{arm}_last_epoch"] = int(run["last_epoch"]) == EXPECTED_EPOCHS[arm]
        checks[f"{arm}_reload_delta"] = float(run["reload_delta"]) <= 1e-4
        curve = run["_curve"]
        checks[f"{arm}_curve_length"] = len(curve) == EXPECTED_EPOCHS[arm]
        checks[f"{arm}_curve_epochs"] = [int(row["epoch"]) for row in curve] == list(
            range(1, EXPECTED_EPOCHS[arm] + 1)
        )
        checks[f"{arm}_curve_finite"] = all(
            math.isfinite(float(row[k])) for row in curve for k in ("train_loss", "valid_macro_ap")
        )
        metrics = [float(row["valid_macro_ap"]) for row in curve]
        checks[f"{arm}_best_epoch_argmax"] = int(np.argmax(metrics)) + 1 == int(run["best_epoch"])
        checks[f"{arm}_curve_sha"] = C.sha256_file(C.RUNS / RUN_NAMES[arm] / "curve.csv") == run["curve_sha256"]
        checks[f"{arm}_checkpoint_sha"] = C.sha256_file(C.RUNS / RUN_NAMES[arm] / "checkpoint.pt") == run["checkpoint_sha256"]
        checks[f"{arm}_predictions_sha"] = C.sha256_file(C.RUNS / RUN_NAMES[arm] / "predictions.npz") == run["predictions_sha256"]
        checks[f"{arm}_base_params"] = int(run["model_info"]["base_parameters"]) == 464739
        checks[f"{arm}_base_init_history"] = bool(run["model_info"]["base_init_matches_history"])
        if arm in ("A1", "A2", "A3"):
            checks[f"{arm}_branch_params"] = int(run["model_info"]["branch_parameters"]) == 20480
            checks[f"{arm}_total_params"] = int(run["model_info"]["total_parameters"]) == 485219
            checks[f"{arm}_init_zero"] = bool(run["init_gate"]["branch_weight_all_zero"])
            checks[f"{arm}_init_delta_exact"] = float(run["init_gate"]["max_abs_branch_delta"]) == 0.0
            checks[f"{arm}_first_step_finite"] = bool(run["first_step_gate"]["branch_grad_finite"])
            checks[f"{arm}_first_step_nonzero"] = bool(run["first_step_gate"]["branch_grad_nonzero"])
            checks[f"{arm}_post_nonzero"] = bool(run["branch_post_gate"]["not_identically_zero"])
            checks[f"{arm}_post_weight_nonzero"] = float(run["branch_post_gate"]["weight_fraction_nonzero"]) > 0.0
            checks[f"{arm}_post_delta_nonzero"] = float(run["branch_post_gate"]["valid_branch_delta_fraction_nonzero"]) > 0.0
        else:
            checks[f"{arm}_total_params"] = int(run["model_info"]["total_parameters"]) == 464739

    arms = ("P1A", "A0", "A1", "A2", "A3")
    checks["same_first_epoch_batch_order"] = len({runs[a]["first_epoch_rows_sha256"] for a in arms}) == 1
    checks["same_labels"] = len({json.dumps(runs[a]["label_hashes"], sort_keys=True) for a in arms}) == 1
    checks["same_feature_spec"] = len({runs[a]["feature_spec_hash"] for a in arms}) == 1
    checks["same_norm"] = len({runs[a]["norm_fingerprint"] for a in arms}) == 1
    checks["same_cache_identity"] = len({json.dumps(runs[a]["cache_identity"], sort_keys=True) for a in arms}) == 1
    checks["same_base_init"] = len({runs[a]["model_info"]["base_init_state_hash"] for a in arms}) == 1

    base_recipe_keys = (
        "optimizer",
        "peak_lr",
        "weight_decay",
        "adam_betas",
        "adam_eps",
        "min_lr",
        "dropout",
        "batch_size",
        "gradient_clip",
        "scheduler",
        "min_epochs",
        "patience",
        "early_stop",
        "selection",
    )
    p1b = ("A0", "A1", "A2", "A3")
    checks["same_p1b_recipe"] = all(
        all(runs[a]["recipe"][k] == runs["A0"]["recipe"][k] for k in base_recipe_keys) for a in p1b
    )
    checks["p1a_horizon_480"] = int(runs["P1A"]["recipe"]["scheduler_horizon_epochs"]) == 480
    checks["p1b_horizon_240"] = all(int(runs[a]["recipe"]["scheduler_horizon_epochs"]) == 240 for a in p1b)
    checks["p1b_min_epochs_240"] = all(int(runs[a]["recipe"]["min_epochs"]) == 240 for a in p1b)
    checks["p1b_no_early_stop"] = all("disabled" in str(runs[a]["recipe"]["early_stop"]) for a in p1b)

    for arm in ("A1", "A2", "A3"):
        align = runs[arm]["structural_gate"]["counts_alignment"]
        checks[f"{arm}_counts_align_train"] = bool(align["train_gids_match_rows"])
        checks[f"{arm}_counts_align_valid"] = bool(align["valid_gids_match_rows"])
        checks[f"{arm}_counts_dim"] = int(align["counts_dim"]) == 2048
        checks[f"{arm}_counts_rows"] = int(align["train_counts_rows"]) == 10873 and int(align["valid_counts_rows"]) == 2331

    manifest = C.read_json(C.ANALYSIS / "p0_features_manifest.json")
    checks["features_train_paths"] = int(manifest["splits"]["train"]["n_paths_total"]) == 4492788
    checks["features_valid_paths"] = int(manifest["splits"]["valid"]["n_paths_total"]) == 949954
    checks["features_null_total_equal"] = (
        manifest["splits"]["train"]["n_paths_total"] == manifest["splits"]["train"]["null_audit"]["n_paths_total_null"]
        and manifest["splits"]["valid"]["n_paths_total"] == manifest["splits"]["valid"]["null_audit"]["n_paths_total_null"]
    )
    checks["features_null_marginals_ok"] = (
        manifest["splits"]["train"]["null_audit"]["n_graphs_with_second_half_marginal_failure"] == 0
        and manifest["splits"]["valid"]["null_audit"]["n_graphs_with_second_half_marginal_failure"] == 0
    )
    checks["features_null_vectors_differ"] = (
        manifest["splits"]["train"]["null_audit"]["n_graphs_full_vs_null_bucket_vectors_equal"] == 0
        and manifest["splits"]["valid"]["null_audit"]["n_graphs_full_vs_null_bucket_vectors_equal"] == 0
    )

    gpu_total = float(sum(float(runs[a].get("gpu_hours", 0.0)) for a in arms))
    smoke_path = C.ANALYSIS / "gpu_smoke.json"
    if smoke_path.is_file():
        gpu_total += float(C.read_json(smoke_path)["gpu_hours"])
    checks["gpu_within_cap"] = gpu_total <= 10.0

    run_meta = {arm: rr_meta(runs[arm]) for arm in arms}
    details["runs"] = {
        arm: {
            "name": RUN_NAMES[arm],
            "experiment": run_meta[arm].get("experiment"),
            "run_id": run_meta[arm].get("run_id"),
            "pool": run_meta[arm].get("pool"),
            "command": run_meta[arm].get("command"),
            "git_commit": run_meta[arm].get("git_commit"),
            "environment": {
                k: runs[arm]["environment"].get(k)
                for k in ("python", "torch", "torch_cuda", "cuda_available", "gpu_names", "hostname")
            },
            "best_epoch": int(runs[arm]["best_epoch"]),
            "valid_macro_ap_pct": float(runs[arm]["valid_macro_ap_pct"]),
            "train_macro_ap_pct": float(runs[arm]["train_macro_ap_pct"]),
            "gpu_hours": float(runs[arm]["gpu_hours"]),
        }
        for arm in arms
    }
    details["gpu_total_hours_including_smoke"] = gpu_total
    details["features"] = {
        split: {
            "npz_sha256": manifest["splits"][split]["features_npz_sha256"],
            "counts_type_sha256": manifest["splits"][split]["counts_type_sha256"],
            "counts_full_sha256": manifest["splits"][split]["counts_full_sha256"],
            "counts_null_sha256": manifest["splits"][split]["counts_null_sha256"],
        }
        for split in ("train", "valid")
    }
    details["checks"] = checks
    failed = [k for k, v in checks.items() if not v]
    details["failed_checks"] = failed
    details["status"] = "PASS" if not failed else "FAIL"
    details["round_id"] = C.ROUND_ID
    details["generated_utc"] = C.utcnow()
    details["git_commit"] = C.git_commit()
    C.write_json(C.ANALYSIS / "formal_consistency.json", details)
    C.log(f"formal consistency: {details['status']} (failed={failed})")
    return 0 if not failed else 2


if __name__ == "__main__":
    raise SystemExit(main())

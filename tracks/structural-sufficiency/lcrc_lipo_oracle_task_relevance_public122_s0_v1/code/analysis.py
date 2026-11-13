"""Stage C: cross-task consistency checks and the combined Part-B summary.

Reads the two committed ``train.json`` / ``partB_*_metrics.json`` outputs and
verifies that the two tasks share the frozen recipe, batch order and initial
state, that Part B used only original Train labels, and that the CF isolation
split is respected.  Writes ``analysis_checks.json``.
"""

from __future__ import annotations

import argparse
from typing import Any, Sequence

import numpy as np

import round_common as RC


def _f(x: Any, d: float = float("nan")) -> float:
    try:
        s = str(x).strip()
        return float(s) if s not in ("", "nan", "None") else d
    except Exception:
        return d


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--analysis-dir", default=str(RC.NEW_ANALYSIS))
    args = ap.parse_args(argv)
    checks: dict[str, Any] = {"round_id": RC.ROUND_ID, "generated_utc": RC.utcnow(), "git_commit": RC.git_commit()}

    trains: dict[str, Any] = {}
    for task in RC.TASKS:
        p = RC.NEW_RUNS / task / "train.json"
        if not p.exists():
            raise SystemExit(f"MISSING_RUN: {RC.track_rel(p)}")
        trains[task] = RC.read_json(p)

    # ---- shared recipe / batch order / initial state --------------------
    shared = {}
    for task, t in trains.items():
        shared[task] = {
            "variant": t["model"]["variant"],
            "parameters": t["model"]["parameters"],
            "init_state_hash": t["model"]["init_state_hash"],
            "full_init_state_hash": t["model"]["full_init_state_hash"],
            "batch_order_sha256": t["batch_order"]["sha256"],
            "epochs_run": t["recipe"]["epochs_run"],
            "steps_per_epoch": t["recipe"]["steps_per_epoch"],
            "batch_size": t["recipe"]["batch_size"],
            "target_norm": t["data"]["target_norm"],
            "train_rows_sha256": t["data"]["train_rows_sha256"],
            "feature_norm_fingerprint": t["data"]["train_norm_fingerprint"],
            "rewire_labels_used_in_training": t["label_source"]["rewire_labels_used_in_training"],
            "other_task_labels_used": t["label_source"]["other_task_labels_used"],
        }
    keys = list(RC.TASKS)
    a, b = shared[keys[0]], shared[keys[1]]
    consistency = {
        "same_parameters": a["parameters"] == b["parameters"] == RC.SC.EXPECTED_PARAMS,
        "same_init_state_hash": a["init_state_hash"] == b["init_state_hash"] == RC.SC.INIT_STATE_HASH,
        "same_full_init_state_hash": a["full_init_state_hash"] == b["full_init_state_hash"] == RC.SC.FULL_INIT_STATE_HASH,
        "same_batch_order": a["batch_order_sha256"] == b["batch_order_sha256"],
        "same_recipe_steps": (a["epochs_run"], a["steps_per_epoch"], a["batch_size"])
        == (b["epochs_run"], b["steps_per_epoch"], b["batch_size"]) == (360, 27, 128),
        "same_train_rows_sha256": a["train_rows_sha256"] == b["train_rows_sha256"],
        "same_feature_norm_fingerprint": a["feature_norm_fingerprint"] == b["feature_norm_fingerprint"],
        "target_norm_differs_as_expected": a["target_norm"] != b["target_norm"],
        "no_rewire_labels_in_training": all(not shared[t]["rewire_labels_used_in_training"] for t in keys),
        "no_cross_task_labels": all(not shared[t]["other_task_labels_used"] for t in keys),
    }
    checks["shared_recipe"] = {"per_task": shared, "consistency": consistency}
    if not all(v for k, v in consistency.items() if isinstance(v, bool)):
        raise SystemExit(f"PARTB_CONSISTENCY_FAILED: {consistency}")

    # ---- CF isolation respected ----------------------------------------
    cf = RC.read_csv(RC.NEW_ANALYSIS / "cf_isolation.csv")
    n_main = sum(1 for r in cf if str(r["main_cf"]) == "1")
    checks["cf_isolation"] = {
        "n_pairs": len(cf), "n_main_cf": n_main,
        "all_main_cf_have_no_train_duplicate": all(
            str(r["o_exact_train_duplicate"]) == "0" and str(r["t_exact_train_duplicate"]) == "0"
            and str(r["k_exact_train_duplicate"]) == "0"
            for r in cf if str(r["main_cf"]) == "1"
        ),
        "note": "main CF = O, T and K0 all absent from the Train original canonical set",
    }

    # ---- cross-task overview -------------------------------------------
    overview: dict[str, Any] = {}
    for task in RC.TASKS:
        m = RC.read_json(RC.NEW_ANALYSIS / f"partB_{task}_metrics.json")
        row: dict[str, Any] = {
            "units": m["units"],
            "test_rmse": m["original_graph_metrics"]["test"]["rmse"],
            "test_r2": m["original_graph_metrics"]["test"]["r2"],
            "train_mean_baseline_test_rmse": m["original_graph_metrics"]["train_mean_constant_baseline_test_rmse"],
            "families": {},
        }
        for fam in RC.SC.FAMILIES:
            b = m["family_stats"].get(f"{task}/test/{fam}")
            if b and "gain" in b:
                row["families"][fam] = {
                    "n_molecules": b["n_molecules"], "gain": b["gain"], "gain_ci95": b["gain_ci95"],
                    "sign_agreement": b["sign_agreement"],
                    "abs_delta_pred_mean": b["abs_delta_pred"]["mean"],
                }
        overview[task] = row
    checks["partB_overview"] = overview

    RC.write_json(RC.NEW_ANALYSIS / "analysis_checks.json", checks)
    RC.log("analysis checks done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

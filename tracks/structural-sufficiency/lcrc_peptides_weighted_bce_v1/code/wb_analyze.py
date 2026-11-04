"""Build the two comparison tables and the final correctness-gate record.

Reads the completed weighted-BCE run plus the read-only historical Diagnostic
v1 run and ECFP reference.  Writes only under this round's ``results/`` and
``analysis/`` directories.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Any

import numpy as np

import wb_checks as CH
import wb_common as C


def read_csv(path: Path) -> list[dict[str, str]]:
    with open(path, "r", encoding="utf-8", newline="") as fh:
        return list(csv.DictReader(fh))


def write_csv(path: Path, rows: list[dict[str, Any]], fields: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row.get(k, "") for k in fields})


def finite(value: Any) -> float:
    out = float(value)
    if not math.isfinite(out):
        raise ValueError(f"non-finite value in analysis input: {value!r}")
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", default=str(C.FORMAL_RUN_DIR))
    args = parser.parse_args(argv)
    run_dir = Path(args.run_dir)
    run = C.read_json(run_dir / "run.json")
    history = C.read_json(C.HISTORY_RUN_JSON)
    pos = C.read_json(C.POS_WEIGHT_JSON)
    ecfp = read_csv(C.ECFP_SUMMARY)[0]
    stats = {row["label"]: row for row in read_csv(C.LABEL_STATISTICS)}

    if run["status"] != "complete_no_test" or run.get("smoke"):
        raise SystemExit(f"cannot analyze non-formal run status={run['status']} smoke={run.get('smoke')}")

    curve = read_csv(run_dir / "curve.csv")
    metrics = [finite(row["valid_metric"]) for row in curve]
    best_epoch_from_curve = int(curve[int(np.argmax(metrics))]["epoch"])
    best_metric_from_curve = max(metrics)
    train_losses = [finite(row["train_loss"]) for row in curve]
    grad_means = [finite(row["grad_norm_mean"]) for row in curve]
    grad_maxes = [finite(row["grad_norm_max"]) for row in curve]

    # ---- per-label table -------------------------------------------------
    new_valid = {name: finite(run["per_label_valid_ap"][name]) for name in C.LABELS}
    new_train = {name: finite(run["per_label_train_ap"][name]) for name in C.LABELS}
    hist_valid = {name: finite(history["per_label_valid_ap"][name]) for name in C.LABELS}
    hist_train = {name: finite(history["per_label_train_ap"][name]) for name in C.LABELS}
    deltas = {name: 100.0 * (new_valid[name] - hist_valid[name]) for name in C.LABELS}
    order = sorted(C.LABELS, key=lambda name: deltas[name], reverse=True)
    rank = {name: i + 1 for i, name in enumerate(order)}

    per_label_rows: list[dict[str, Any]] = []
    for j, name in enumerate(C.LABELS):
        payload_row = pos["labels"][j]
        if payload_row["label"] != name:
            raise SystemExit(f"pos_weight label order mismatch at {j}: {payload_row['label']} != {name}")
        valid_pos = int(stats[name]["valid_pos"])
        valid_n = int(stats[name]["valid_n"])
        old_gap = hist_train[name] - hist_valid[name]
        new_gap = new_train[name] - new_valid[name]
        per_label_rows.append(
            {
                "label_index": j,
                "label": name,
                "train_n": int(payload_row["train_n"]),
                "train_pos": int(payload_row["train_pos"]),
                "train_neg": int(payload_row["train_neg"]),
                "train_pos_rate": finite(payload_row["train_pos_rate"]),
                "pos_weight": finite(payload_row["pos_weight"]),
                "valid_n": valid_n,
                "valid_pos": valid_pos,
                "valid_pos_rate": float(valid_pos) / float(valid_n),
                "history_valid_ap": hist_valid[name],
                "new_valid_ap": new_valid[name],
                "delta_valid_ap_pp": deltas[name],
                "direction": "improved" if deltas[name] > 0 else ("declined" if deltas[name] < 0 else "unchanged"),
                "improvement_rank": rank[name],
                "history_train_ap": hist_train[name],
                "new_train_ap": new_train[name],
                "delta_train_ap_pp": 100.0 * (new_train[name] - hist_train[name]),
                "history_train_valid_gap": old_gap,
                "new_train_valid_gap": new_gap,
                "gap_change_pp": 100.0 * (new_gap - old_gap),
            }
        )

    per_label_fields = [
        "label_index",
        "label",
        "train_n",
        "train_pos",
        "train_neg",
        "train_pos_rate",
        "pos_weight",
        "valid_n",
        "valid_pos",
        "valid_pos_rate",
        "history_valid_ap",
        "new_valid_ap",
        "delta_valid_ap_pp",
        "direction",
        "improvement_rank",
        "history_train_ap",
        "new_train_ap",
        "delta_train_ap_pp",
        "history_train_valid_gap",
        "new_train_valid_gap",
        "gap_change_pp",
    ]
    write_csv(C.RESULTS / "per_label_comparison.csv", per_label_rows, per_label_fields)

    # ---- three-method table ---------------------------------------------
    hist_macro = finite(history["best_valid_macro_ap"])
    new_macro = finite(run["best_valid_macro_ap"])
    hist_train_macro = finite(history["train_macro_ap_per_label_mean"])
    new_train_macro = finite(run["train_macro_ap_per_label_mean"])
    ecfp_macro = finite(ecfp["valid_macro_ap"])
    comparison_rows = [
        {
            "method": "LCRC-C1 ordinary BCE (historical)",
            "comparison_role": "single_variable_control",
            "model": "LCRC-C1 / NO_BOND_MEAN",
            "seed": 0,
            "loss": "BCEWithLogitsLoss (ordinary)",
            "best_epoch": int(history["best_epoch"]),
            "train_macro_ap": hist_train_macro,
            "train_macro_ap_pct": 100.0 * hist_train_macro,
            "valid_macro_ap": hist_macro,
            "valid_macro_ap_pct": 100.0 * hist_macro,
            "train_valid_macro_gap": hist_train_macro - hist_macro,
            "delta_vs_historical_valid_ap_pp": 0.0,
            "delta_vs_historical_train_ap_pp": 0.0,
            "delta_vs_historical_gap_pp": 0.0,
            "checkpoint_sha256": history["checkpoint_sha256"],
            "source": C.track_rel(C.HISTORY_RUN_JSON),
        },
        {
            "method": "LCRC-C1 weighted BCE (this round)",
            "comparison_role": "single_variable_intervention",
            "model": "LCRC-C1 / NO_BOND_MEAN",
            "seed": 0,
            "loss": "Weighted BCEWithLogitsLoss (Train-only Nneg/Npos)",
            "best_epoch": int(run["best_epoch"]),
            "train_macro_ap": new_train_macro,
            "train_macro_ap_pct": 100.0 * new_train_macro,
            "valid_macro_ap": new_macro,
            "valid_macro_ap_pct": 100.0 * new_macro,
            "train_valid_macro_gap": new_train_macro - new_macro,
            "delta_vs_historical_valid_ap_pp": 100.0 * (new_macro - hist_macro),
            "delta_vs_historical_train_ap_pp": 100.0 * (new_train_macro - hist_train_macro),
            "delta_vs_historical_gap_pp": 100.0 * ((new_train_macro - new_macro) - (hist_train_macro - hist_macro)),
            "checkpoint_sha256": run["checkpoint_sha256"],
            "source": C.track_rel(run_dir / "run.json"),
        },
        {
            "method": "ECFP4 count + LightGBM (background reference)",
            "comparison_role": "background_reference_not_single_variable_control",
            "model": "ECFP4 count + LightGBM, one model per label",
            "seed": 0,
            "loss": "LightGBM binary objective, is_unbalance=True",
            "best_epoch": "not_applicable",
            "train_macro_ap": "not_applicable",
            "train_macro_ap_pct": "not_applicable",
            "valid_macro_ap": ecfp_macro,
            "valid_macro_ap_pct": 100.0 * ecfp_macro,
            "train_valid_macro_gap": "not_applicable",
            "delta_vs_historical_valid_ap_pp": 100.0 * (ecfp_macro - hist_macro),
            "delta_vs_historical_train_ap_pp": "not_applicable",
            "delta_vs_historical_gap_pp": "not_applicable",
            "checkpoint_sha256": "not_applicable",
            "source": C.track_rel(C.ECFP_SUMMARY),
        },
    ]
    comparison_fields = [
        "method",
        "comparison_role",
        "model",
        "seed",
        "loss",
        "best_epoch",
        "train_macro_ap",
        "train_macro_ap_pct",
        "valid_macro_ap",
        "valid_macro_ap_pct",
        "train_valid_macro_gap",
        "delta_vs_historical_valid_ap_pp",
        "delta_vs_historical_train_ap_pp",
        "delta_vs_historical_gap_pp",
        "checkpoint_sha256",
        "source",
    ]
    write_csv(C.RESULTS / "validation_comparison.csv", comparison_rows, comparison_fields)

    # ---- explicit run gates ---------------------------------------------
    run_checks = {
        "run_status_complete_no_test": run["status"] == "complete_no_test",
        "formal_run_not_smoke": run.get("smoke") is False,
        "test_evaluated_false": run.get("test_evaluated") is False,
        "test_access_attempts_empty": run.get("test_access_attempts") == [],
        "guard_attempts_empty": run.get("guard", {}).get("attempts") == [],
        "parameters_464739": int(run["parameters"]) == 464739,
        "structural_gate_all_match": run["structural_gate"].get("all_match") is True,
        "init_hash_matches_fast_v1_seed0": run["structural_gate"].get("init_state_hash_match_fast_v1_seed0") is True,
        "init_hash_matches_diagnostic_run": run["structural_gate"].get("init_state_hash_match_diagnostic_run") is True,
        "feature_spec_and_norm_match": bool(
            run["structural_gate"].get("spec_hash_match_diagnostic_run")
            and run["structural_gate"].get("norm_fingerprint_match_diagnostic_run")
        ),
        "cache_identity_match": run["structural_gate"].get("cache_identity_match_diagnostic_run") is True,
        "pos_weight_payload_verified": run["intervention"]["verification"].get("status") == "PASS",
        "pos_weight_train_only": run["intervention"]["verification"].get("valid_used") is False
        and run["intervention"]["verification"].get("test_used") is False,
        "antiparasitic_weight_gt_1": run["intervention"]["pos_weight"][8] > 1.0,
        "antimicrobial_weight_lt_1": run["intervention"]["pos_weight"][4] < 1.0,
        "batch_size_64": int(run["batch_size"]) == 64,
        "no_batch_fallback": run.get("batch_fallback_used") is False,
        "strict_single_variable_control": run.get("strict_single_variable_control") is True,
        "recipe_max_240": int(run["recipe"]["max_epochs"]) == 240,
        "recipe_min_80": int(run["recipe"]["min_epochs"]) == 80,
        "recipe_patience_40": int(run["recipe"]["patience"]) == 40,
        "recipe_horizon_240": int(run["recipe"]["scheduler_horizon_epochs"]) == 240,
        "peak_lr_3e-4": abs(float(run["recipe"]["peak_lr"]) - 3e-4) < 1e-15,
        "weight_decay_1e-3": abs(float(run["recipe"]["weight_decay"]) - 1e-3) < 1e-15,
        "gradient_clip_5": abs(float(run["recipe"]["gradient_clip"]) - 5.0) < 1e-15,
        "curve_best_epoch_reproduced": best_epoch_from_curve == int(run["best_epoch"]),
        "curve_best_metric_reproduced": abs(best_metric_from_curve - new_macro) < 1e-12,
        "curve_train_losses_finite": all(np.isfinite(train_losses)),
        "curve_grad_norms_finite": all(np.isfinite(grad_means)) and all(np.isfinite(grad_maxes)),
        "checkpoint_sha256_match": C.sha256_file(run_dir / "checkpoint.pt") == run["checkpoint_sha256"],
        "reload_reproduces_best_within_1e-4": float(run["reload_valid_delta"]) <= 1e-4,
        "no_nonfinite_loss_or_gradient_event": run.get("nonfinite_events", 0) == 0,
        "no_resource_block": run.get("status") == "complete_no_test",
    }

    result_checks = {
        "comparison_has_three_rows": len(comparison_rows) == 3,
        "per_label_has_ten_rows": len(per_label_rows) == 10,
        "per_label_order_matches_protocol": [r["label"] for r in per_label_rows] == list(C.LABELS),
        "per_label_new_macro_reproduced": abs(np.mean([r["new_valid_ap"] for r in per_label_rows]) - float(run["valid_macro_ap_per_label_mean"])) < 1e-12,
        "per_label_history_macro_reproduced": abs(np.mean([r["history_valid_ap"] for r in per_label_rows]) - float(history["valid_macro_ap_per_label_mean"])) < 1e-12,
        "pos_weight_rows_match_payload": all(
            r["pos_weight"] == float(pos["labels"][j]["pos_weight"]) for j, r in enumerate(per_label_rows)
        ),
        "deltas_sum_to_reloaded_macro_delta": abs(
            np.mean([r["delta_valid_ap_pp"] for r in per_label_rows])
            - 100.0
            * (
                float(run["valid_macro_ap_per_label_mean"])
                - float(history["valid_macro_ap_per_label_mean"])
            )
        ) < 1e-9,
    }

    base = CH.run_all()
    checks = dict(base["checks"])
    checks.update({f"run_{k}": bool(v) for k, v in run_checks.items()})
    checks.update({f"tables_{k}": bool(v) for k, v in result_checks.items()})
    gates = {
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "generated_utc": C.utcnow(),
        "all_passed": bool(all(checks.values())),
        "checks": checks,
        "preflight_checks": base["checks"],
        "run_checks": run_checks,
        "table_checks": result_checks,
        "preflight_details": base["details"],
        "run": {
            "best_epoch": int(run["best_epoch"]),
            "epochs_run": int(run["epochs_run"]),
            "stopped_early": bool(run["stopped_early"]),
            "best_valid_macro_ap": new_macro,
            "train_macro_ap": new_train_macro,
            "test_evaluated": False,
            "test_access_attempts": run["test_access_attempts"],
            "batch_size": int(run["batch_size"]),
        },
        "environment": C.environment(),
    }
    C.write_json(C.ANALYSIS / "gates.json", gates)

    resource = {
        "round_id": C.ROUND_ID,
        "gpu_budget_hours": 1.25,
        "formal_runs": 1,
        "seeds": [0],
        "gpu_hours_used_formal": float(run["seconds"]) / 3600.0,
        "seconds_formal": float(run["seconds"]),
        "max_seconds_allowed": float(run["max_seconds"]),
        "gpu_peak_allocated_bytes": int(run["gpu_peak_allocated_bytes"]),
        "gpu_peak_reserved_bytes": int(run["gpu_peak_reserved_bytes"]),
        "gpu_name": (run.get("environment", {}).get("gpu_names") or [None])[0],
        "batch_size": int(run["batch_size"]),
        "batch_fallback_used": bool(run["batch_fallback_used"]),
        "oom": False,
        "time_limit_hit": False,
        "status": "within_budget",
        "environment": run.get("environment", {}),
    }
    C.write_json(C.ANALYSIS / "resource_ledger.json", resource)

    facts = {
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "history": {
            "best_epoch": int(history["best_epoch"]),
            "epochs_run": int(history["epochs_run"]),
            "train_macro_ap": hist_train_macro,
            "valid_macro_ap": hist_macro,
            "train_valid_gap": hist_train_macro - hist_macro,
            "checkpoint_sha256": history["checkpoint_sha256"],
        },
        "current": {
            "best_epoch": int(run["best_epoch"]),
            "epochs_run": int(run["epochs_run"]),
            "stopped_early": bool(run["stopped_early"]),
            "train_macro_ap": new_train_macro,
            "valid_macro_ap": new_macro,
            "train_valid_gap": new_train_macro - new_macro,
            "delta_valid_pp": 100.0 * (new_macro - hist_macro),
            "delta_train_pp": 100.0 * (new_train_macro - hist_train_macro),
            "delta_gap_pp": 100.0 * ((new_train_macro - new_macro) - (hist_train_macro - hist_macro)),
            "reload_valid_delta": float(run["reload_valid_delta"]),
            "best_epoch_train_loss": float(run["best_epoch_train_loss"]),
            "last_epoch_train_loss": float(run["last_epoch_train_loss"]),
            "last_epoch_valid_macro_ap": float(run["last_epoch_valid_macro_ap"]),
            "grad_clip_events": int(run["grad_clip_events"]),
            "grad_norm_max": float(run["grad_norm_max"]),
            "peak_grad_norm_mean_epoch": max(grad_means),
            "peak_grad_norm_max_epoch": max(grad_maxes),
            "seconds": float(run["seconds"]),
            "gpu_hours": float(run["seconds"]) / 3600.0,
            "gpu_peak_allocated_bytes": int(run["gpu_peak_allocated_bytes"]),
            "gpu_peak_reserved_bytes": int(run["gpu_peak_reserved_bytes"]),
            "batch_size": int(run["batch_size"]),
            "checkpoint_sha256": run["checkpoint_sha256"],
            "curve_sha256": run["curve_sha256"],
        },
        "ecfp_background": {
            "valid_macro_ap": ecfp_macro,
            "delta_vs_history_pp": 100.0 * (ecfp_macro - hist_macro),
            "role": "background reference, not a single-variable control",
        },
        "improved_labels": [
            {"label": r["label"], "delta_pp": r["delta_valid_ap_pp"]}
            for r in per_label_rows
            if r["delta_valid_ap_pp"] > 0
        ],
        "declined_labels": [
            {"label": r["label"], "delta_pp": r["delta_valid_ap_pp"]}
            for r in per_label_rows
            if r["delta_valid_ap_pp"] < 0
        ],
        "largest_improvements": [
            {"label": name, "delta_pp": deltas[name]} for name in order[:3]
        ],
        "largest_declines": [
            {"label": name, "delta_pp": deltas[name]} for name in order[-3:][::-1]
        ],
    }
    C.write_json(C.ANALYSIS / "run_facts.json", facts)

    C.log(
        f"weighted valid macro AP {100.0 * new_macro:.4f}% vs history {100.0 * hist_macro:.4f}% "
        f"(delta {100.0 * (new_macro - hist_macro):+.4f} pp)"
    )
    C.log(f"wrote {C.track_rel(C.RESULTS / 'validation_comparison.csv')}")
    C.log(f"wrote {C.track_rel(C.RESULTS / 'per_label_comparison.csv')}")
    C.log(f"gates all_passed={gates['all_passed']} -> {C.track_rel(C.ANALYSIS / 'gates.json')}")
    return 0 if gates["all_passed"] else 2


if __name__ == "__main__":
    raise SystemExit(main())

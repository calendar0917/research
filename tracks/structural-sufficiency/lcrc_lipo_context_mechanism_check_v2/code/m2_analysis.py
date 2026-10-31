"""Comparison, tail audit and two-dimensional decision for the v2 check.

Inputs are the frozen evaluation artifacts written by ``m2_evaluate.py``.  All
Test numbers are ``EXPOSED_TEST_EXPLORATORY``.  No model is changed here.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import m2_common as C


PAIRS = {
    "base_s0__vs__context_s0": ("base_s0", "context_s0"),
    "base_s1__vs__context_s1": ("base_s1", "context_s1"),
    "context_s0__vs__roleblind_s0": ("context_s0", "roleblind_s0"),
}


def _load_eval() -> dict[str, Any]:
    path = C.EVAL_DIR / "metrics.json"
    if not path.exists():
        raise FileNotFoundError(f"missing {path}; run `run_m2.py eval-all` first")
    payload = C.read_json(path)
    if payload.get("scope") != "EXPOSED_TEST_EXPLORATORY":
        raise PermissionError("evaluation payload is not marked EXPOSED_TEST_EXPLORATORY")
    return payload


def _bootstrap(eval_payload: Mapping[str, Any], pair: str, split: str) -> dict[str, Any]:
    try:
        return eval_payload["bootstrap"][pair][f"{split}420"]
    except KeyError as exc:
        raise KeyError(f"missing bootstrap for {pair} {split}") from exc


def _model_parameters(key: str) -> int | None:
    if key in C.HISTORICAL_CHECKPOINTS:
        train_json = C.HISTORICAL_CHECKPOINTS[key]["train_json"]
        if Path(train_json).exists():
            return int(C.read_json(train_json)["model_audit"]["parameters"])
        return None
    run_dir = C.RUNS / C.ARM_SPECS[key]["run_key"]
    train_json = run_dir / "train.json"
    if train_json.exists():
        return int(C.read_json(train_json)["model_audit"]["parameters"])
    return None


def build_seed01(eval_payload: Mapping[str, Any]) -> dict[str, Any]:
    cells: list[dict[str, Any]] = []
    for seed in (0, 1):
        base_key = f"base_s{seed}"
        context_key = f"context_s{seed}"
        pair = f"{base_key}__vs__{context_key}"
        for split in ("valid", "test"):
            base = eval_payload["models"][base_key][split]
            context = eval_payload["models"][context_key][split]
            boot = _bootstrap(eval_payload, pair, split)
            cells.append(
                {
                    "seed": seed,
                    "split": f"{split}420",
                    "base_rmse": base["rmse"],
                    "base_mae": base["mae"],
                    "base_r2": base["r2"],
                    "base_sse": base["sse"],
                    "context_rmse": context["rmse"],
                    "context_mae": context["mae"],
                    "context_r2": context["r2"],
                    "context_sse": context["sse"],
                    "gain_rmse": base["rmse"] - context["rmse"],
                    "mae_change": context["mae"] - base["mae"],
                    "r2_change": context["r2"] - base["r2"],
                    "sse_gain": base["sse"] - context["sse"],
                    "n_context_better_abs": None,
                    "bootstrap_gain_rmse_interval": boot["interval"]["gain_rmse"],
                    "bootstrap_mae_change_interval": boot["interval"]["mae_change"],
                    "bootstrap_sse_gain_interval": boot["interval"]["sse_gain"],
                    "bootstrap_observed": boot["observed"],
                    "base_parameters": _model_parameters(base_key),
                    "context_parameters": _model_parameters(context_key),
                }
            )
    # Fill per-cell improved-molecule counts from the per-molecule table.
    per_mol = _read_per_molecule("valid")
    per_mol_test = _read_per_molecule("test")
    improved = {"valid": _improved_count(per_mol, "base_s0", "context_s0"), "test": _improved_count(per_mol_test, "base_s0", "context_s0")}
    for cell in cells:
        if cell["seed"] == 0:
            cell["n_context_better_abs"] = improved[cell["split"].replace("420", "")]
    improved1 = {"valid": _improved_count(per_mol, "base_s1", "context_s1"), "test": _improved_count(per_mol_test, "base_s1", "context_s1")}
    for cell in cells:
        if cell["seed"] == 1:
            cell["n_context_better_abs"] = improved1[cell["split"].replace("420", "")]

    gains = [cell["gain_rmse"] for cell in cells]
    reversals = []
    for seed in (0, 1):
        gv = next(c["gain_rmse"] for c in cells if c["seed"] == seed and c["split"] == "valid420")
        gt = next(c["gain_rmse"] for c in cells if c["seed"] == seed and c["split"] == "test420")
        reversals.append(
            {
                "seed": seed,
                "valid_gain": gv,
                "test_gain": gt,
                "difference_valid_minus_test": gv - gt,
                "sign_reversal_between_valid_and_test": bool(gv * gt < 0),
            }
        )
    all_positive = bool(all(g > 0.0 for g in gains))
    performance = {
        "rule": C.PERFORMANCE_RULE,
        "n_positive_cells": int(sum(1 for g in gains if g > 0.0)),
        "n_cells": len(gains),
        "gains": gains,
        "all_four_positive": all_positive,
        "PERFORMANCE_SIGNAL": "PRESENT" if all_positive else "ABSENT",
        "per_seed": reversals,
        "seed_to_seed_gain_change": {
            "valid420": next(c["gain_rmse"] for c in cells if c["seed"] == 1 and c["split"] == "valid420")
            - next(c["gain_rmse"] for c in cells if c["seed"] == 0 and c["split"] == "valid420"),
            "test420": next(c["gain_rmse"] for c in cells if c["seed"] == 1 and c["split"] == "test420")
            - next(c["gain_rmse"] for c in cells if c["seed"] == 0 and c["split"] == "test420"),
        },
        "interpretation_note": (
            "Paired bootstrap intervals describe molecule-sampling variability inside one "
            "trained model; they do not estimate training-seed variance.  The seed-0 vs "
            "seed-1 difference is a second training draw and is reported separately."
        ),
    }
    return {"cells": cells, "performance": performance}


def _read_per_molecule(split: str) -> list[dict[str, str]]:
    path = C.EVAL_DIR / f"per_molecule_{split}420.csv"
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _improved_count(rows: Sequence[Mapping[str, str]], ref_key: str, other_key: str) -> int:
    return int(sum(float(row[f"abs_err_{other_key}"]) < float(row[f"abs_err_{ref_key}"]) for row in rows))


def build_roleblind(eval_payload: Mapping[str, Any]) -> dict[str, Any]:
    pair = "context_s0__vs__roleblind_s0"
    cells: list[dict[str, Any]] = []
    for split in ("valid", "test"):
        context = eval_payload["models"]["context_s0"][split]
        roleblind = eval_payload["models"]["roleblind_s0"][split]
        boot = _bootstrap(eval_payload, pair, split)
        # boot.gain_rmse = RMSE(context) - RMSE(roleblind); negate for "CONTEXT better".
        interval_ref = boot["interval"]["gain_rmse"]
        context_gain_interval = [-float(interval_ref[2]), -float(interval_ref[1]), -float(interval_ref[0])]
        cells.append(
            {
                "split": f"{split}420",
                "context_rmse": context["rmse"],
                "context_mae": context["mae"],
                "context_r2": context["r2"],
                "context_sse": context["sse"],
                "roleblind_rmse": roleblind["rmse"],
                "roleblind_mae": roleblind["mae"],
                "roleblind_r2": roleblind["r2"],
                "roleblind_sse": roleblind["sse"],
                "gain_rmse_context_better": roleblind["rmse"] - context["rmse"],
                "mae_gain_context_better": roleblind["mae"] - context["mae"],
                "sse_gain_context_better": roleblind["sse"] - context["sse"],
                "mean_abs_prediction_difference": boot["observed"]["mean_abs_prediction_difference"],
                "max_abs_prediction_difference": boot["observed"]["max_abs_prediction_difference"],
                "n_context_better_abs": boot["observed"]["n_ref_better_abs"],
                "n_roleblind_better_abs": boot["observed"]["n_other_better_abs"],
                "n_tie_abs": boot["observed"]["n_tie_abs"],
                "bootstrap_gain_rmse_context_better_interval": context_gain_interval,
                "bootstrap_raw_gain_rmse_roleblind_minus_context_interval": interval_ref,
                "bootstrap_observed_raw": boot["observed"],
                "context_parameters": _model_parameters("context_s0"),
                "roleblind_parameters": _model_parameters("roleblind_s0"),
            }
        )
    positive_both = bool(all(c["gain_rmse_context_better"] > 0.0 for c in cells))
    ci_excludes_zero_both = bool(all(c["bootstrap_gain_rmse_context_better_interval"][0] > 0.0 for c in cells))
    present = bool(positive_both and ci_excludes_zero_both)
    rolebl = {
        "rule": C.ROLE_SPECIFICITY_RULE,
        "cells": cells,
        "context_better_on_both_splits": positive_both,
        "both_context_gain_intervals_above_zero": ci_excludes_zero_both,
        "ROLE_SPECIFICITY_SIGNAL": "PRESENT" if present else "ABSENT",
        "interpretation_note": (
            "If ROLEBLIND is close to or better than CONTEXT, chemical-role specificity is "
            "not supported; CONTEXT's gain must not be attributed to a chemical interaction "
            "mechanism."
        ),
    }
    return rolebl


def build_tail(eval_payload: Mapping[str, Any]) -> dict[str, Any]:
    rows_valid = _read_per_molecule("valid")
    by_row = {int(row["row_id"]): row for row in rows_valid}
    missing = [row for row in C.TAIL_ROWS if row not in by_row]
    if missing:
        raise ValueError(f"fixed tail rows missing from Validation420: {missing}")
    comparisons = {
        "context_s0_vs_base_s0": ("base_s0", "context_s0", "CONTEXT", "BASE"),
        "roleblind_s0_vs_context_s0": ("context_s0", "roleblind_s0", "ROLEBLIND", "CONTEXT"),
    }
    records: dict[str, Any] = {}
    for name, (ref_key, other_key, other_label, ref_label) in comparisons.items():
        tail = []
        for row_id in C.TAIL_ROWS:
            row = by_row[row_id]
            y = float(row["y_logD"])
            pred_ref = float(row[f"pred_{ref_key}"])
            pred_other = float(row[f"pred_{other_key}"])
            abs_ref = float(row[f"abs_err_{ref_key}"])
            abs_other = float(row[f"abs_err_{other_key}"])
            sq_ref = float(row[f"sq_err_{ref_key}"])
            sq_other = float(row[f"sq_err_{other_key}"])
            tail.append(
                {
                    "row_id": int(row_id),
                    "y_logD": y,
                    "pred_ref": pred_ref,
                    "pred_other": pred_other,
                    "abs_error_ref": abs_ref,
                    "abs_error_other": abs_other,
                    "sq_error_ref": sq_ref,
                    "sq_error_other": sq_other,
                    "abs_error_improvement_other_better": abs_ref - abs_other,
                    "sse_improvement_other_better": sq_ref - sq_other,
                    "other_better": bool(abs_other < abs_ref),
                }
            )
        tail_rows = [int(row["row_id"]) for row in rows_valid]
        tail_mask = np.asarray([row in set(C.TAIL_ROWS) for row in tail_rows], dtype=bool)
        sq_ref = np.asarray([float(row[f"sq_err_{ref_key}"]) for row in rows_valid], dtype=np.float64)
        sq_other = np.asarray([float(row[f"sq_err_{other_key}"]) for row in rows_valid], dtype=np.float64)
        sse_ref = float(sq_ref.sum())
        sse_other = float(sq_other.sum())
        net = sse_ref - sse_other
        tail_sse = float((sq_ref[tail_mask] - sq_other[tail_mask]).sum())
        keep = ~tail_mask
        rmse_ref_without = float(np.sqrt(np.mean(sq_ref[keep])))
        rmse_other_without = float(np.sqrt(np.mean(sq_other[keep])))
        records[name] = {
            "ref_model": ref_key,
            "other_model": other_key,
            "ref_label": ref_label,
            "other_label": other_label,
            "split": "valid420",
            "tail_rows": tail,
            "n_tail_other_better": int(sum(1 for row in tail if row["other_better"])),
            "n_tail_ref_better": int(sum(1 for row in tail if not row["other_better"])),
            "sse_ref": sse_ref,
            "sse_other": sse_other,
            "net_sse_improvement_other_better": net,
            "tail_sse_improvement_other_better": tail_sse,
            "tail_share_of_net_sse_improvement": (tail_sse / net) if net != 0.0 else None,
            "rmse_ref_without_tail": rmse_ref_without,
            "rmse_other_without_tail": rmse_other_without,
            "gain_without_tail_other_better": rmse_ref_without - rmse_other_without,
            "rmse_ref_all": float(np.sqrt(np.mean(sq_ref))),
            "rmse_other_all": float(np.sqrt(np.mean(sq_other))),
            "gain_all_other_better": float(np.sqrt(np.mean(sq_ref)) - np.sqrt(np.mean(sq_other))),
            "n_non_tail": int(keep.sum()),
            "note": "fixed five rows from lcrc_lipo_random42_gap_audit_v1; descriptive only, no model change",
        }
    return {
        "status": "POST_HOC_DESCRIPTIVE_ONLY",
        "tail_rows_fixed_before_this_round": list(C.TAIL_ROWS),
        "tail_rows_in_validation": True,
        "tail_rows_in_test": [],
        "tail_rows_in_test_note": "the five fixed tail rows belong to Validation420; they are not Test rows",
        "comparisons": records,
    }


def analyze() -> dict[str, Any]:
    eval_payload = _load_eval()
    seed01 = build_seed01(eval_payload)
    roleblind = build_roleblind(eval_payload)
    tail = build_tail(eval_payload)
    comparison = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "scope": "EXPOSED_TEST_EXPLORATORY",
        "seed01": seed01,
        "roleblind": roleblind,
        "historical_reference": eval_payload.get("historical_reference"),
        "model_metrics": eval_payload["models"],
        "test_row_identity": eval_payload["test_row_identity"],
    }
    C.write_json(C.COMPARISON, comparison)
    C.write_json(C.ROLEBLIND_COMPARISON, roleblind)
    C.write_json(C.TAIL_AUDIT, tail)
    decision = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "scope": "EXPOSED_TEST_EXPLORATORY",
        "PERFORMANCE_SIGNAL": seed01["performance"]["PERFORMANCE_SIGNAL"],
        "ROLE_SPECIFICITY_SIGNAL": roleblind["ROLE_SPECIFICITY_SIGNAL"],
        "performance_detail": seed01["performance"],
        "role_specificity_detail": {
            "context_better_on_both_splits": roleblind["context_better_on_both_splits"],
            "both_context_gain_intervals_above_zero": roleblind["both_context_gain_intervals_above_zero"],
            "cells": roleblind["cells"],
            "rule": roleblind["rule"],
        },
        "independent_dimensions": True,
        "test_access": "EXPOSED_TEST_EXPLORATORY",
        "no_model_change_after_decision": True,
    }
    C.write_json(C.DECISION, decision)

    # CSV tables.
    with open(C.ANALYSIS / "table_seed01.csv", "w", newline="", encoding="utf-8") as fh:
        fields = [
            "seed", "split", "base_rmse", "base_mae", "base_r2", "context_rmse", "context_mae",
            "context_r2", "gain_rmse", "mae_change", "r2_change", "sse_gain",
            "n_context_better_abs", "bootstrap_gain_rmse_lower", "bootstrap_gain_rmse_median",
            "bootstrap_gain_rmse_upper",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for cell in seed01["cells"]:
            writer.writerow(
                {
                    "seed": cell["seed"],
                    "split": cell["split"],
                    "base_rmse": cell["base_rmse"],
                    "base_mae": cell["base_mae"],
                    "base_r2": cell["base_r2"],
                    "context_rmse": cell["context_rmse"],
                    "context_mae": cell["context_mae"],
                    "context_r2": cell["context_r2"],
                    "gain_rmse": cell["gain_rmse"],
                    "mae_change": cell["mae_change"],
                    "r2_change": cell["r2_change"],
                    "sse_gain": cell["sse_gain"],
                    "n_context_better_abs": cell["n_context_better_abs"],
                    "bootstrap_gain_rmse_lower": cell["bootstrap_gain_rmse_interval"][0],
                    "bootstrap_gain_rmse_median": cell["bootstrap_gain_rmse_interval"][1],
                    "bootstrap_gain_rmse_upper": cell["bootstrap_gain_rmse_interval"][2],
                }
            )
    with open(C.ANALYSIS / "table_roleblind.csv", "w", newline="", encoding="utf-8") as fh:
        fields = [
            "split", "context_rmse", "context_mae", "context_r2", "roleblind_rmse", "roleblind_mae",
            "roleblind_r2", "gain_rmse_context_better", "mae_gain_context_better",
            "sse_gain_context_better", "mean_abs_prediction_difference",
            "max_abs_prediction_difference", "n_context_better_abs", "n_roleblind_better_abs",
            "context_gain_ci_lower", "context_gain_ci_median", "context_gain_ci_upper",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for cell in roleblind["cells"]:
            writer.writerow(
                {
                    "split": cell["split"],
                    "context_rmse": cell["context_rmse"],
                    "context_mae": cell["context_mae"],
                    "context_r2": cell["context_r2"],
                    "roleblind_rmse": cell["roleblind_rmse"],
                    "roleblind_mae": cell["roleblind_mae"],
                    "roleblind_r2": cell["roleblind_r2"],
                    "gain_rmse_context_better": cell["gain_rmse_context_better"],
                    "mae_gain_context_better": cell["mae_gain_context_better"],
                    "sse_gain_context_better": cell["sse_gain_context_better"],
                    "mean_abs_prediction_difference": cell["mean_abs_prediction_difference"],
                    "max_abs_prediction_difference": cell["max_abs_prediction_difference"],
                    "n_context_better_abs": cell["n_context_better_abs"],
                    "n_roleblind_better_abs": cell["n_roleblind_better_abs"],
                    "context_gain_ci_lower": cell["bootstrap_gain_rmse_context_better_interval"][0],
                    "context_gain_ci_median": cell["bootstrap_gain_rmse_context_better_interval"][1],
                    "context_gain_ci_upper": cell["bootstrap_gain_rmse_context_better_interval"][2],
                }
            )
    with open(C.ANALYSIS / "table_tail.csv", "w", newline="", encoding="utf-8") as fh:
        fields = [
            "comparison", "row_id", "y_logD", "pred_ref", "pred_other", "abs_error_ref",
            "abs_error_other", "sq_error_ref", "sq_error_other", "abs_error_improvement_other_better",
            "sse_improvement_other_better", "other_better",
        ]
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for name, record in tail["comparisons"].items():
            for row in record["tail_rows"]:
                writer.writerow({"comparison": name, **row})
    C.log(f"[analysis] PERFORMANCE_SIGNAL={decision['PERFORMANCE_SIGNAL']} ROLE_SPECIFICITY_SIGNAL={decision['ROLE_SPECIFICITY_SIGNAL']}")
    return decision


__all__ = ["analyze", "build_seed01", "build_roleblind", "build_tail", "paired_bootstrap"]

"""Evaluation: main table, paired component bootstrap, cases, fixed sensitivities.

Reads only committed artefacts plus the pulled run directories:

* ``analysis/alias_audit.json``             (A replication + identical-input groups)
* ``analysis/baseline_reference.json``      (A checkpoint identity / metrics / RNG)
* ``../lcrc_lipo_edit_supervision_public122_s0_v1/analysis/per_molecule.csv`` (A rows, labels, components)
* ``runs/<arm>/predictions_best.npz`` + ``train.json``

Writes ``analysis/results.csv``, ``comparisons.json``, ``case_responses.csv``,
``runtime.json``.
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import common as C
import data as D
import metrics as M


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def load_source_rows() -> tuple[dict[str, list[int]], dict[str, np.ndarray], dict[str, Any], list[str], dict[str, float]]:
    src = C.source_with_labels("all")
    rows = {s: [int(x) for x in src["rows"][s]] for s in C.SPLITS}
    y = {s: np.asarray(src["y"][s], dtype=np.float64) for s in C.SPLITS}
    target_norm = C.DD.target_normaliser(np.asarray(src["y"]["train"], dtype=np.float64))
    smiles = [str(x) for x in src["smiles"]]
    return rows, y, target_norm, smiles, {s: rows[s] for s in C.SPLITS}


def load_baseline_table() -> dict[str, dict[int, dict[str, Any]]]:
    out: dict[int, dict[str, Any]] = {}
    with open(C.BASELINE_PER_MOLECULE, newline="", encoding="utf-8") as fh:
        for row in csv.DictReader(fh):
            if row["arm"] != "A":
                continue
            out[int(row["row_id"])] = {
                "split": row["split"],
                "y_true": float(row["y_true"]),
                "pred": float(row["pred"]),
                "eval_component": int(row["eval_component"]),
            }
    return {"A": out}


def load_run(arm: str) -> dict[str, Any]:
    run = C.run_dir(arm)
    train_json = run / "train.json"
    pred_path = run / "predictions_best.npz"
    last_path = run / "predictions_last.npz"
    if not train_json.exists() or not pred_path.exists():
        raise FileNotFoundError(f"{arm}: missing {train_json} or {pred_path}; pull the run first")
    payload = C.read_json(train_json)
    with np.load(pred_path) as z:
        best = {k: z[k].copy() for k in z.files}
    last = None
    if last_path.exists():
        with np.load(last_path) as z:
            last = {k: z[k].copy() for k in z.files}
    return {"train_json": payload, "best": best, "last": last, "dir": run}


def arm_predictions(loaded: Mapping[str, Mapping[str, Any]], rows: Mapping[str, list[int]]) -> dict[str, dict[str, np.ndarray]]:
    out: dict[str, dict[str, np.ndarray]] = {}
    for arm, rec in loaded.items():
        out[arm] = {}
        if "per_row" in rec:  # baseline table
            per = rec["per_row"]
            for split in C.SPLITS:
                out[arm][split] = np.asarray([per[int(r)]["pred"] for r in rows[split]], dtype=np.float64)
        else:
            best = rec["best"]
            for split in C.SPLITS:
                split_rows = np.asarray(best[f"{split}_rows"], dtype=np.int64)
                if [int(x) for x in split_rows.tolist()] != [int(x) for x in rows[split]]:
                    raise ValueError(f"{arm}/{split}: prediction rows differ from the published split")
                out[arm][split] = np.asarray(best[f"{split}_pred_raw"], dtype=np.float64)
    return out


# ---------------------------------------------------------------------------
# main evaluation
# ---------------------------------------------------------------------------


def evaluate() -> dict[str, Any]:
    rows, y, target_norm, smiles, _ = load_source_rows()
    baseline_table = load_baseline_table()
    audit = C.read_json(C.ANALYSIS / "alias_audit.json")
    baseline_ref_path = C.ANALYSIS / "baseline_reference.json"
    baseline_ref = C.read_json(baseline_ref_path) if baseline_ref_path.exists() else {}

    loaded: dict[str, dict[str, Any]] = {"A": {"per_row": baseline_table["A"]}}
    for arm in C.ARMS:
        if (C.run_dir(arm) / "train.json").exists():
            loaded[arm] = load_run(arm)
    preds = arm_predictions(loaded, rows)

    # ---- label-free evaluation components --------------------------------
    same_input_groups = [g["rows"] for g in audit["groups"]["detail"]["full_groups"]]
    baseline_component = {int(r): int(rec["eval_component"]) for r, rec in baseline_table["A"].items()}
    component = M.build_eval_components(rows["train"] + rows["valid"] + rows["test"], baseline_component, same_input_groups)
    members = M.component_members(component)
    n_merged_by_input = sum(
        1 for g in same_input_groups if len({component[int(r)] for r in g}) == 1 and len(g) > 1
    )

    # ---- point metrics ---------------------------------------------------
    point: dict[str, dict[str, dict[str, float]]] = {}
    for arm, per_split in preds.items():
        point[arm] = {
            split: M.regression_metrics(per_split[split], y[split]) for split in C.SPLITS
        }
    last_point: dict[str, dict[str, dict[str, float]]] = {}
    for arm, rec in loaded.items():
        if "best" in rec and rec.get("last") is not None:
            last_point[arm] = {
                split: M.regression_metrics(np.asarray(rec["last"][f"{split}_pred_raw"], dtype=np.float64), y[split])
                for split in C.SPLITS
            }

    # ---- paired component bootstrap -------------------------------------
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    boot: dict[str, dict[str, Any]] = {}
    for split in C.SPLITS:
        boot[split] = M.bootstrap_split(
            predictions={arm: preds[arm][split] for arm in preds},
            rows=rows[split],
            y=y[split],
            component=component,
            rng=rng,
        )

    comparisons: dict[str, Any] = {}
    pairs = list(C.MAIN_COMPARISONS) + ([C.NULL_COMPARISON] if "NULL3" in preds else [])
    for ref, cand in pairs:
        if ref not in preds or cand not in preds:
            continue
        key = f"{cand}_vs_{ref}"
        entry: dict[str, Any] = {}
        for split in C.SPLITS:
            for metric in ("rmse", "mae"):
                entry[f"{split}_{metric}"] = M.gain_summary(
                    reference=ref,
                    candidate=cand,
                    metric=metric,
                    point_reference=point[ref][split][metric],
                    point_candidate=point[cand][split][metric],
                    boots_reference=boot[split][ref][metric],
                    boots_candidate=boot[split][cand][metric],
                )
        comparisons[key] = entry

    # ---- fixed sensitivity checks ---------------------------------------
    sensitivity: dict[str, Any] = {}
    a_test = preds["A"]["test"]
    a_valid = preds["A"]["valid"]
    test_err2 = (a_test - y["test"]) ** 2
    worst_test = int(np.argmax(test_err2))
    valid_err2 = (a_valid - y["valid"]) ** 2
    worst_valid = int(np.argmax(valid_err2))
    keep_test = np.ones(len(y["test"]), dtype=bool)
    keep_test[worst_test] = False
    keep_valid = np.ones(len(y["valid"]), dtype=bool)
    keep_valid[worst_valid] = False
    sens_test_rows = {arm: M.regression_metrics(preds[arm]["test"][keep_test], y["test"][keep_test]) for arm in preds}
    sens_valid_rows = {arm: M.regression_metrics(preds[arm]["valid"][keep_valid], y["valid"][keep_valid]) for arm in preds}
    sensitivity["remove_A_largest_test_squared_error"] = {
        "removed_row": int(rows["test"][worst_test]),
        "removed_row_squared_error_A": float(test_err2[worst_test]),
        "n_remaining": int(keep_test.sum()),
        "test_rmse": {arm: float(v["rmse"]) for arm, v in sens_test_rows.items()},
        "gains": {
            key: float(sens_test_rows[ref]["rmse"] - sens_test_rows[cand]["rmse"])
            for ref, cand in pairs
            if ref in sens_test_rows and cand in sens_test_rows
            for key in [f"{cand}_vs_{ref}"]
        },
        "all_gains_positive": bool(
            all(
                sens_test_rows[ref]["rmse"] - sens_test_rows[cand]["rmse"] > 0
                for ref, cand in pairs
                if ref in sens_test_rows and cand in sens_test_rows
            )
        ),
    }
    sensitivity["remove_A_largest_valid_squared_error"] = {
        "removed_row": int(rows["valid"][worst_valid]),
        "removed_row_squared_error_A": float(valid_err2[worst_valid]),
        "n_remaining": int(keep_valid.sum()),
        "valid_rmse": {arm: float(v["rmse"]) for arm, v in sens_valid_rows.items()},
        "gains": {
            key: float(sens_valid_rows[ref]["rmse"] - sens_valid_rows[cand]["rmse"])
            for ref, cand in pairs
            if ref in sens_valid_rows and cand in sens_valid_rows
            for key in [f"{cand}_vs_{ref}"]
        },
    }
    last_gains = {}
    for ref, cand in pairs:
        if ref in last_point and cand in last_point:
            last_gains[f"{cand}_vs_{ref}"] = {
                split: float(last_point[ref][split]["rmse"] - last_point[cand][split]["rmse"]) for split in C.SPLITS
            }
    sensitivity["last_epoch_360_revaluation"] = {
        "note": "reported for progress inspection only; never re-selects a checkpoint",
        "test_rmse": {arm: float(last_point[arm]["test"]["rmse"]) for arm in last_point},
        "valid_rmse": {arm: float(last_point[arm]["valid"]["rmse"]) for arm in last_point},
        "train_rmse": {arm: float(last_point[arm]["train"]["rmse"]) for arm in last_point},
        "gains": last_gains,
    }

    # ---- decision labels -------------------------------------------------
    decisions: dict[str, Any] = {}
    alias_summary = audit["alias_pair_summary"]
    decisions["C1_INPUT_ALIAS_VERIFIED"] = bool(
        alias_summary["n_signature_multiset_equal"] == 5
        and alias_summary["n_global42_equal"] == 5
        and alias_summary["n_prediction_equal_within_tol"] == 5
        and alias_summary["n_non_isomorphic_raw_graph"] == 5
    )
    decisions["S3_INPUT_SEPARATION_VERIFIED"] = bool(
        alias_summary["n_u3_separated"] == 4
        and alias_summary["u3_matches_design_expectation"]
        and alias_summary["n_u2_separated"] == 0
    )

    perf = {}
    if "LOCAL3" in comparisons:
        key = "LOCAL3_vs_A"
        test = comparisons[key]["test_rmse"]
        valid = comparisons[key]["valid_rmse"]
        influence_ok = bool(sensitivity["remove_A_largest_test_squared_error"]["gains"].get(key, -1.0) > 0)
        valid_influence_ok = bool(sensitivity["remove_A_largest_valid_squared_error"]["gains"].get(key, -1.0) > 0)
        direction_ok = bool(np.sign(test["gain"]) == np.sign(valid["gain"]) and test["gain"] != 0)
        perf["LOCAL3_vs_A"] = {
            "test_gain": test["gain"],
            "test_ci": [test["gain_ci95_lower"], test["gain_ci95_upper"]],
            "valid_gain": valid["gain"],
            "valid_ci": [valid["gain_ci95_lower"], valid["gain_ci95_upper"]],
            "direction_consistent": direction_ok,
            "test_influence_positive": influence_ok,
            "valid_influence_positive": valid_influence_ok,
        }
        if test["gain"] <= 0.0:
            decisions["PERFORMANCE"] = "NO_PREDICTIVE_GAIN_THIS_CONFIG"
        elif (not test["ci_lower_gt_zero"]) or (not direction_ok) or (not influence_ok) or (not valid_influence_ok):
            decisions["PERFORMANCE"] = "PERFORMANCE_UNRESOLVED_THIS_RUN"
        elif "DUP2" in comparisons:
            key2 = "LOCAL3_vs_DUP2"
            test2 = comparisons[key2]["test_rmse"]
            valid2 = comparisons[key2]["valid_rmse"]
            inf2 = bool(sensitivity["remove_A_largest_test_squared_error"]["gains"].get(key2, -1.0) > 0)
            infv2 = bool(sensitivity["remove_A_largest_valid_squared_error"]["gains"].get(key2, -1.0) > 0)
            perf[key2] = {
                "test_gain": test2["gain"],
                "test_ci": [test2["gain_ci95_lower"], test2["gain_ci95_upper"]],
                "valid_gain": valid2["gain"],
                "valid_ci": [valid2["gain_ci95_lower"], valid2["gain_ci95_upper"]],
                "test_influence_positive": inf2,
                "valid_influence_positive": infv2,
            }
            if test2["gain"] > 0 and test2["ci_lower_gt_zero"] and np.sign(test2["gain"]) == np.sign(valid2["gain"]) and inf2 and infv2:
                decisions["PERFORMANCE"] = "S3_PREDICTIVE_SIGNAL_THIS_RUN"
            else:
                decisions["PERFORMANCE"] = "NEW_INFORMATION_INCREMENT_UNRESOLVED"
        else:
            decisions["PERFORMANCE"] = "PERFORMANCE_UNRESOLVED_THIS_RUN"
    decisions["detailed"] = perf

    if "NULL3" in comparisons:
        key = "LOCAL3_vs_NULL3"
        test = comparisons[key]["test_rmse"]
        valid = comparisons[key]["valid_rmse"]
        inf_t = bool(sensitivity["remove_A_largest_test_squared_error"]["gains"].get(key, -1.0) > 0)
        inf_v = bool(sensitivity["remove_A_largest_valid_squared_error"]["gains"].get(key, -1.0) > 0)
        ok = bool(test["gain"] > 0 and test["ci_lower_gt_zero"] and np.sign(test["gain"]) == np.sign(valid["gain"]) and inf_t and inf_v)
        decisions["ROOT_ALIGNMENT"] = "ROOT_ALIGNMENT_SIGNAL_THIS_RUN" if ok else "ROOT_ALIGNMENT_UNRESOLVED"
        decisions["ROOT_ALIGNMENT_detail"] = {
            "test_gain": test["gain"],
            "test_ci": [test["gain_ci95_lower"], test["gain_ci95_upper"]],
            "valid_gain": valid["gain"],
            "valid_ci": [valid["gain_ci95_lower"], valid["gain_ci95_upper"]],
            "test_influence_positive": inf_t,
            "valid_influence_positive": inf_v,
        }
    else:
        decisions["ROOT_ALIGNMENT"] = "NOT_PURCHASED_BY_FIXED_RULE"

    # ---- case responses --------------------------------------------------
    case_rows: list[dict[str, Any]] = []
    pair_info = {tuple(sorted((r["row_a"], r["row_b"]))): r for r in audit["alias_pairs"]}
    sep_info = {
        (r["row_a"], r["row_b"]): r for r in C.read_json(C.ANALYSIS / "feature_separation.json")["pairs"]
    }
    for ra, rb in C.ALIAS_PAIRS:
        split_a = next(s for s in C.SPLITS if ra in set(rows[s]))
        split_b = next(s for s in C.SPLITS if rb in set(rows[s]))
        pos_a = {s: rows[s].index(ra) for s in C.SPLITS if ra in set(rows[s])}
        pos_b = {s: rows[s].index(rb) for s in C.SPLITS if rb in set(rows[s])}
        ya = float(y[split_a][pos_a[split_a]])
        yb = float(y[split_b][pos_b[split_b]])
        for arm in preds:
            pa = float(preds[arm][split_a][pos_a[split_a]])
            pb = float(preds[arm][split_b][pos_b[split_b]])
            case_rows.append(
                {
                    "arm": arm,
                    "variant": "as_trained",
                    "row_a": ra,
                    "row_b": rb,
                    "split_a": split_a,
                    "split_b": split_b,
                    "y_a": ya,
                    "y_b": yb,
                    "y_diff": yb - ya,
                    "pred_a": pa,
                    "pred_b": pb,
                    "pred_diff": pb - pa,
                    "diff_error": (pb - pa) - (yb - ya),
                    "abs_diff_error": abs((pb - pa) - (yb - ya)),
                    "smiles_a": smiles[ra],
                    "smiles_b": smiles[rb],
                    "isomorphism_class": pair_info[tuple(sorted((ra, rb)))]["isomorphism_class"],
                    "u3_separated": pair_info[tuple(sorted((ra, rb)))]["u3_separated"],
                    "u2_separated": pair_info[tuple(sorted((ra, rb)))]["u2_separated"],
                }
            )

    # ---- runtime / budget ------------------------------------------------
    runtime: dict[str, Any] = {
        "schema": "lcrc-lipo-s3-locality-runtime-v1",
        "round_id": C.ROUND_ID,
        "arms": {},
        "gpu_soft_cap_hours": C.GPU_SOFT_CAP_HOURS,
        "gpu_hard_cap_hours": C.GPU_HARD_CAP_HOURS,
    }
    total_gpu = 0.0
    for arm, rec in loaded.items():
        if "train_json" not in rec:
            continue
        tj = rec["train_json"]
        gpu_h = float(tj["gpu"]["gpu_hours_charged"])
        total_gpu += gpu_h
        runtime["arms"][arm] = {
            "best_epoch": int(tj["checkpoint"]["best_epoch"]),
            "process_wall_seconds": float(tj["timing"]["process_wall_seconds"]),
            "train_wall_seconds": float(tj["timing"]["train_wall_seconds"]),
            "gpu_hours": gpu_h,
            "device": tj["environment"]["device"],
            "device_name": tj["gpu"]["device_name"],
            "hostname": tj["environment"]["hostname"],
            "git_commit": tj["provenance"]["git_commit"],
            "parameters": int(tj["model"]["parameters"]),
            "peak_memory_allocated_bytes": int(tj["gpu"]["peak_memory_allocated_bytes"]),
        }
    smoke_dir = C.RUNS / "smoke"
    smoke_hours = 0.0
    for arm in C.STAGE1_ARMS:
        p = smoke_dir / arm / "train.json"
        if p.exists():
            tj = C.read_json(p)
            smoke_hours += float(tj["gpu"]["gpu_hours_charged"])
            runtime.setdefault("smoke", {})[arm] = {
                "gpu_hours": float(tj["gpu"]["gpu_hours_charged"]),
                "best_epoch": int(tj["checkpoint"]["best_epoch"]),
            }
    runtime["formal_gpu_hours"] = float(total_gpu)
    runtime["smoke_gpu_hours"] = float(smoke_hours)
    runtime["total_gpu_hours_charged"] = float(total_gpu + smoke_hours)
    runtime["within_soft_cap"] = bool(total_gpu + smoke_hours <= C.GPU_SOFT_CAP_HOURS)
    runtime["within_hard_cap"] = bool(total_gpu + smoke_hours <= C.GPU_HARD_CAP_HOURS)
    runtime["budget_note"] = "sum of wall-clock over every CUDA process of this round (smoke included)"

    # ---- write ------------------------------------------------------------
    results_fields = [
        "arm", "best_epoch", "parameters",
        "train_rmse", "train_mae", "valid_rmse", "valid_mae", "test_rmse", "test_mae",
        "train_rmse_last", "valid_rmse_last", "test_rmse_last", "gpu_hours",
    ]
    with open(C.ANALYSIS / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=results_fields)
        w.writeheader()
        for arm in preds:
            rec = loaded[arm]
            best_epoch = int(rec["train_json"]["checkpoint"]["best_epoch"]) if "train_json" in rec else int(
                baseline_ref.get("checkpoint", {}).get("best_epoch", C.BASELINE_EXPECTED["best_epoch"])
            )
            parameters = int(rec["train_json"]["model"]["parameters"]) if "train_json" in rec else C.BASE_PARAMS
            gpu_hours = float(rec["train_json"]["gpu"]["gpu_hours_charged"]) if "train_json" in rec else float(
                baseline_ref.get("gpu_hours", 0.0)
            )
            w.writerow(
                {
                    "arm": arm,
                    "best_epoch": best_epoch,
                    "parameters": parameters,
                    "train_rmse": point[arm]["train"]["rmse"],
                    "train_mae": point[arm]["train"]["mae"],
                    "valid_rmse": point[arm]["valid"]["rmse"],
                    "valid_mae": point[arm]["valid"]["mae"],
                    "test_rmse": point[arm]["test"]["rmse"],
                    "test_mae": point[arm]["test"]["mae"],
                    "train_rmse_last": last_point.get(arm, {}).get("train", {}).get("rmse", ""),
                    "valid_rmse_last": last_point.get(arm, {}).get("valid", {}).get("rmse", ""),
                    "test_rmse_last": last_point.get(arm, {}).get("test", {}).get("rmse", ""),
                    "gpu_hours": gpu_hours,
                }
            )

    with open(C.ANALYSIS / "case_responses.csv", "w", newline="", encoding="utf-8") as fh:
        fields = [
            "arm", "variant", "row_a", "row_b", "split_a", "split_b", "y_a", "y_b", "y_diff",
            "pred_a", "pred_b", "pred_diff", "diff_error", "abs_diff_error", "isomorphism_class",
            "u3_separated", "u2_separated", "smiles_a", "smiles_b",
        ]
        w = csv.DictWriter(fh, fieldnames=fields, extrasaction="ignore")
        w.writeheader()
        for row in case_rows:
            w.writerow(row)
        intervention_path = C.ANALYSIS / "interventions.json"
        if intervention_path.exists():
            for row in C.read_json(intervention_path).get("case_rows", []):
                merged = dict(row)
                merged.setdefault("isomorphism_class", "")
                merged.setdefault("u3_separated", "")
                merged.setdefault("u2_separated", "")
                merged.setdefault("smiles_a", "")
                merged.setdefault("smiles_b", "")
                w.writerow(merged)
        else:
            C.log("[evaluate] interventions.json missing: LOCAL3 intervention rows not in case_responses.csv")

    payload = {
        "schema": "lcrc-lipo-s3-locality-comparisons-v1",
        "round_id": C.ROUND_ID,
        "created_utc": C.utcnow(),
        "point_metrics": point,
        "last_epoch_metrics": last_point,
        "comparisons": comparisons,
        "sensitivity": sensitivity,
        "decisions": decisions,
        "evaluation_components": {
            "n_components_total": len(members),
            "n_components_per_split": {
                split: len({component[r] for r in rows[split]}) for split in C.SPLITS
            },
            "n_full_input_groups": len(same_input_groups),
            "n_full_input_groups_merged_inside_one_component": int(n_merged_by_input),
            "source": "baseline per_molecule eval_component UNION identical complete C1 input groups",
            "label_free": True,
            "bootstrap": {
                "replicates": C.BOOTSTRAP_REPLICATES,
                "seed": C.BOOTSTRAP_SEED,
                "shared_across_models": True,
                "unit": "component (baseline structural component merged with identical-input group)",
            },
        },
        "baseline_reference": baseline_ref,
    }
    C.write_json(C.ANALYSIS / "comparisons.json", payload)
    C.write_json(C.ANALYSIS / "runtime.json", runtime)
    return payload


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="evaluate the S3-locality arms")
    ap.parse_args(argv)
    payload = evaluate()
    print(
        "decisions: "
        + ", ".join(f"{k}={v}" for k, v in payload["decisions"].items() if isinstance(v, str))
    )
    for arm, m in payload["point_metrics"].items():
        print(f"{arm:7s} train={m['train']['rmse']:.6f} valid={m['valid']['rmse']:.6f} test={m['test']['rmse']:.6f}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

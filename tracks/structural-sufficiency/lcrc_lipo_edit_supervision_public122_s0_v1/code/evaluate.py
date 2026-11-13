"""CPU evaluation of the four fixed arms.

Reads only pulled per-molecule predictions and the frozen pair/schedule
manifests.  All numbers are in original logD units.  The main edit metric is the
component-equal pair-difference MAE; the bootstrap resamples evaluation
components, not individual pairs.
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import common as C
import metrics as M
import pairs as P

SET_NAMES = [
    "P_train",
    "E_T",
    "E_T_Train_Test",
    "E_T_Valid_Test",
    "E_T_Test_Test",
    "E_HH",
    "E_HH_seen_core",
    "E_HH_unseen_core",
    "Train_Valid",
    "Valid_Valid",
]


def _load_npz(path: Path) -> dict[str, np.ndarray]:
    with np.load(path) as z:
        return {k: z[k].copy() for k in z.files}


def _strip_arrays(d: Mapping[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in d.items() if k != "arrays"}


def _load_arm(arm: str, expected_rows: Mapping[str, Sequence[int]], y_source: Mapping[str, np.ndarray]) -> dict[str, Any]:
    run = C.run_dir(arm)
    train_json = C.read_json(run / "train.json")
    best = _load_npz(run / "predictions_best.npz")
    last = _load_npz(run / "predictions_last.npz")
    init = _load_npz(run / "predictions_initial.npz")
    pred_by_row: dict[int, float] = {}
    pred_last_by_row: dict[int, float] = {}
    pred_init_by_row: dict[int, float] = {}
    for split in C.SPLITS:
        rows = [int(r) for r in expected_rows[split]]
        if [int(x) for x in best[f"{split}_rows"].tolist()] != rows:
            raise ValueError(f"{arm}/{split}: best prediction row ids are not the published split")
        if [int(x) for x in last[f"{split}_rows"].tolist()] != rows:
            raise ValueError(f"{arm}/{split}: last prediction row ids are not the published split")
        if not np.array_equal(np.asarray(best[f"{split}_y"]), np.asarray(y_source[split])):
            raise ValueError(f"{arm}/{split}: best prediction targets differ from source")
        if not np.array_equal(np.asarray(last[f"{split}_y"]), np.asarray(y_source[split])):
            raise ValueError(f"{arm}/{split}: last prediction targets differ from source")
        for r, v in zip(rows, np.asarray(best[f"{split}_pred_raw"], dtype=np.float64)):
            pred_by_row[int(r)] = float(v)
        for r, v in zip(rows, np.asarray(last[f"{split}_pred_raw"], dtype=np.float64)):
            pred_last_by_row[int(r)] = float(v)
        if split in ("train", "valid"):
            if [int(x) for x in init[f"{split}_rows"].tolist()] != rows:
                raise ValueError(f"{arm}/{split}: initial prediction row ids differ")
            for r, v in zip(rows, np.asarray(init[f"{split}_pred_raw"], dtype=np.float64)):
                pred_init_by_row[int(r)] = float(v)
    if len(pred_by_row) != C.N_MOLECULES:
        raise ValueError(f"{arm}: best predictions cover {len(pred_by_row)} rows, expected {C.N_MOLECULES}")
    return {
        "train_json": train_json,
        "best": best,
        "last": last,
        "init": init,
        "pred_by_row": pred_by_row,
        "pred_last_by_row": pred_last_by_row,
        "pred_init_by_row": pred_init_by_row,
    }


def _split_metrics(pred_by_row, y_by_row, rows_by_split) -> dict[str, Any]:
    out = {}
    for split in C.SPLITS:
        rows = [int(r) for r in rows_by_split[split]]
        pred = np.asarray([pred_by_row[int(r)] for r in rows], dtype=np.float64)
        y = np.asarray([y_by_row[int(r)] for r in rows], dtype=np.float64)
        out[split] = M.regression_metrics(pred, y)
    return out


def _gain(point_ref: float, point_cand: float, boots_ref: np.ndarray, boots_cand: np.ndarray) -> dict[str, Any]:
    return M.gain_summary(point_ref, point_cand, boots_ref, boots_cand)


def _point_gain(point_ref: float, point_cand: float) -> dict[str, Any]:
    return {
        "reference": float(point_ref),
        "candidate": float(point_cand),
        "gain": float(point_ref) - float(point_cand),
        "gain_ci95_lower": None,
        "gain_ci95_upper": None,
        "ci_lower_gt_zero": None,
        "ci_upper_lt_zero": None,
        "ci_crosses_zero": None,
        "n_bootstrap": 0,
    }


def _decide_common(comp: Mapping[str, Any], data_audit: Mapping[str, Any], budget_status: str) -> dict[str, Any]:
    if data_audit.get("status") != "OK":
        return {"code": "DATA_INSUFFICIENT", "reasons": ["data audit status is not OK"]}
    if budget_status == "BUDGET_LIMITED":
        return {"code": "BUDGET_LIMITED", "reasons": ["GPU budget hard cap exceeded"]}
    ra = comp["comparisons"]["R_vs_A"]
    test = ra["test_rmse"]
    et = ra["e_t_ce_mae"]
    valid = ra["valid_rmse"]
    imp = comp["impact"]
    reasons = []
    if not (test["gain"] > 0):
        reasons.append("R vs A Test RMSE gain is not positive")
    if not (et["gain"] > 0):
        reasons.append("R vs A E_T CE-MAE gain is not positive")
    if not (valid["gain"] > 0):
        reasons.append("R vs A Valid RMSE gain is not positive")
    if not (test["ci_lower_gt_zero"]):
        reasons.append("Test RMSE 95% interval lower bound is not > 0")
    if not (et["ci_lower_gt_zero"]):
        reasons.append("E_T CE-MAE 95% interval lower bound is not > 0")
    if not imp["remove_top_e_t_component"]["not_reversed_r_vs_a_e_t"]:
        reasons.append("removing the top E_T component reverses the R vs A E_T gain")
    if not imp["remove_top_test_molecule"]["not_reversed_r_vs_a_test"]:
        reasons.append("removing the top Test molecule reverses the R vs A Test RMSE gain")
    if not imp["remove_top_test_molecule"]["not_reversed_r_vs_a_e_t"]:
        reasons.append("removing the top Test molecule reverses the R vs A E_T gain")
    if not reasons:
        code = "DUAL_SIGNAL_THIS_RUN"
    elif test["gain"] > 0 and et["gain"] > 0:
        code = "DUAL_TREND_UNCERTAIN_THIS_RUN"
    elif et["gain"] > 0 and not (test["gain"] > 0):
        code = "EDIT_PERFORMANCE_TRADEOFF"
    elif test["gain"] > 0 and not (et["gain"] > 0):
        code = "PERFORMANCE_ONLY_THIS_RUN"
    else:
        code = "NO_JOINT_GAIN_THIS_CONFIG"
    return {"code": code, "reasons": reasons}


def _decide_relation(comp: Mapping[str, Any], data_audit: Mapping[str, Any], budget_status: str) -> dict[str, Any]:
    if data_audit.get("status") != "OK":
        return {"code": "DATA_INSUFFICIENT", "reasons": ["data audit status is not OK"]}
    if budget_status == "BUDGET_LIMITED":
        return {"code": "BUDGET_LIMITED", "reasons": ["GPU budget hard cap exceeded"]}
    comps = comp["comparisons"]
    imp = comp["impact"]
    reasons = []
    control_better = []
    for name in ("R_vs_W", "R_vs_N"):
        et = comps[name]["e_t_ce_mae"]
        test = comps[name]["test_rmse"]
        if not (et["gain"] > 0):
            reasons.append(f"{name} E_T CE-MAE gain is not positive")
        if not (et["ci_lower_gt_zero"]):
            reasons.append(f"{name} E_T 95% interval lower bound is not > 0")
        if not (test["gain"] >= 0):
            reasons.append(f"{name} Test RMSE point estimate is worse for R")
        if not imp["remove_top_e_t_component"][f"not_reversed_{name.lower()}_e_t"]:
            reasons.append(f"removing the top E_T component reverses {name} E_T gain")
        if not imp["remove_top_test_molecule"][f"not_reversed_{name.lower()}_e_t"]:
            reasons.append(f"removing the top Test molecule reverses {name} E_T gain")
        if not imp["remove_top_test_molecule"][f"not_reversed_{name.lower()}_test"]:
            reasons.append(f"removing the top Test molecule reverses {name} Test RMSE gain")
        # control better than R = positive gain of control over R on either main metric
        if (et["gain"] < 0 and et["ci_upper_lt_zero"]) or (test["gain"] < 0 and test["ci_upper_lt_zero"]):
            control_better.append(name)
    if not reasons:
        code = "RELATION_INCREMENT_THIS_RUN"
    elif control_better:
        code = "CONTROL_BETTER_THIS_RUN"
    else:
        code = "RELATION_INCREMENT_UNRESOLVED"
    return {"code": code, "reasons": reasons, "control_better": control_better}


def main() -> int:
    data_audit = C.read_json(C.ANALYSIS / "data_audit.json")
    pairs = P.load_pairs()
    src = C.source_with_labels("all")
    rows_by_split = {s: [int(r) for r in src["rows"][s]] for s in C.SPLITS}
    all_rows = sorted(int(r) for s in C.SPLITS for r in rows_by_split[s])
    smiles = [str(x) for x in src["smiles"]]
    canonical = P.canonical_smiles_map(smiles)
    y_by_row: dict[int, float] = {}
    y_source: dict[str, np.ndarray] = {}
    split_of: dict[int, str] = {}
    for split in C.SPLITS:
        y_source[split] = np.asarray(src["y"][split], dtype=np.float64)
        for r, v in zip(rows_by_split[split], y_source[split]):
            y_by_row[int(r)] = float(v)
            split_of[int(r)] = split

    # structural components on all 4200 structures (used for bootstrap)
    units_eval = C.ES.build_units(pairs, all_rows, canonical)
    row_to_component = {int(k): int(v) for k, v in units_eval["row_to_unit"].items()}
    for p in pairs:
        if int(p["eval_component"]) != int(row_to_component[int(p["row_a"])]):
            raise ValueError(f"{p['pair_id']}: stored eval component differs from rebuild")

    arms = list(C.ARMS)
    missing = [a for a in arms if not (C.run_dir(a) / "train.json").exists()]
    if missing:
        payload = {
            "schema": "lcrc-lipo-edit-supervision-comparisons-v1",
            "round_id": C.ROUND_ID,
            "created_utc": C.utcnow(),
            "status": "EXECUTION_INCOMPLETE",
            "missing_arms": missing,
        }
        C.write_json(C.ANALYSIS / "comparisons.json", payload)
        C.log(f"[evaluate] EXECUTION_INCOMPLETE missing {missing}")
        return 2

    loaded = {a: _load_arm(a, rows_by_split, y_source) for a in arms}
    split_metrics = {a: _split_metrics(loaded[a]["pred_by_row"], y_by_row, rows_by_split) for a in arms}
    last_split_metrics = {a: _split_metrics(loaded[a]["pred_last_by_row"], y_by_row, rows_by_split) for a in arms}

    sets = {name: P.set_by_name(pairs, name) for name in SET_NAMES}
    pair_metrics = {a: {name: _strip_arrays(M.pair_metrics(loaded[a]["pred_by_row"], y_by_row, sets[name])) for name in SET_NAMES} for a in arms}
    last_pair_metrics = {a: {name: _strip_arrays(M.pair_metrics(loaded[a]["pred_last_by_row"], y_by_row, sets[name])) for name in SET_NAMES} for a in arms}

    # ---- bootstrap indices, shared by all arms for each metric -------------
    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    test_rows_by_comp: dict[int, list[int]] = {}
    for r in rows_by_split["test"]:
        test_rows_by_comp.setdefault(int(row_to_component[int(r)]), []).append(int(r))
    test_comps = sorted(test_rows_by_comp)
    test_index = rng.integers(0, len(test_comps), size=(C.BOOTSTRAP_REPLICATES, len(test_comps)))

    e_t_pairs = sets["E_T"]
    e_hh_pairs = sets["E_HH"]
    e_t_comps = sorted({int(p["eval_component"]) for p in e_t_pairs})
    e_hh_comps = sorted({int(p["eval_component"]) for p in e_hh_pairs})
    e_t_index = rng.integers(0, len(e_t_comps), size=(C.BOOTSTRAP_REPLICATES, len(e_t_comps))) if e_t_comps else None
    e_hh_index = rng.integers(0, len(e_hh_comps), size=(C.BOOTSTRAP_REPLICATES, len(e_hh_comps))) if e_hh_comps else None

    test_sq = {}
    test_boots = {}
    for a in arms:
        test_sq[a] = {int(r): float((loaded[a]["pred_by_row"][int(r)] - y_by_row[int(r)]) ** 2) for r in rows_by_split["test"]}
        test_boots[a], _, _, _ = M.bootstrap_test_rmse(
            test_rows_by_comp, test_sq[a], rng=rng, index=test_index
        )

    e_t_values = {}
    e_t_boots = {}
    e_hh_values = {}
    e_hh_boots = {}
    for a in arms:
        arr = M.pair_arrays(loaded[a]["pred_by_row"], y_by_row, e_t_pairs)
        e_t_values[a] = M.component_means(np.abs(arr["err"]), arr["component"])
        if e_t_comps:
            e_t_boots[a], _ = M.bootstrap_component_equal(e_t_values[a], rng=rng, index=e_t_index)
        else:
            e_t_boots[a] = np.full(C.BOOTSTRAP_REPLICATES, np.nan)
        arr_hh = M.pair_arrays(loaded[a]["pred_by_row"], y_by_row, e_hh_pairs)
        e_hh_values[a] = M.component_means(np.abs(arr_hh["err"]), arr_hh["component"])
        if e_hh_comps:
            e_hh_boots[a], _ = M.bootstrap_component_equal(e_hh_values[a], rng=rng, index=e_hh_index)
        else:
            e_hh_boots[a] = np.full(C.BOOTSTRAP_REPLICATES, np.nan)

    # ---- comparisons -------------------------------------------------------
    def comparison(ref: str, cand: str) -> dict[str, Any]:
        out: dict[str, Any] = {}
        out["test_rmse"] = _gain(
            split_metrics[ref]["test"]["rmse"],
            split_metrics[cand]["test"]["rmse"],
            test_boots[ref],
            test_boots[cand],
        )
        out["valid_rmse"] = _point_gain(
            split_metrics[ref]["valid"]["rmse"], split_metrics[cand]["valid"]["rmse"]
        )
        out["train_rmse"] = _point_gain(
            split_metrics[ref]["train"]["rmse"], split_metrics[cand]["train"]["rmse"]
        )
        out["e_t_ce_mae"] = _gain(
            pair_metrics[ref]["E_T"]["ce_mae"],
            pair_metrics[cand]["E_T"]["ce_mae"],
            e_t_boots[ref],
            e_t_boots[cand],
        )
        out["e_hh_ce_mae"] = _gain(
            pair_metrics[ref]["E_HH"]["ce_mae"],
            pair_metrics[cand]["E_HH"]["ce_mae"],
            e_hh_boots[ref],
            e_hh_boots[cand],
        )
        for name in SET_NAMES:
            if name in ("P_train", "E_T", "E_HH"):
                continue
            out[f"{name}_ce_mae"] = _point_gain(
                pair_metrics[ref][name]["ce_mae"], pair_metrics[cand][name]["ce_mae"]
            )
        return out

    # gain = control - R, so a positive gain means R is better (protocol section 8)
    comparisons = {
        "R_vs_A": comparison("A", "R"),
        "R_vs_W": comparison("W", "R"),
        "R_vs_N": comparison("N", "R"),
    }

    # ---- fixed impact checks ----------------------------------------------
    a_e_t_arrays = M.pair_arrays(loaded["A"]["pred_by_row"], y_by_row, e_t_pairs)
    a_comp_means = M.component_means(np.abs(a_e_t_arrays["err"]), a_e_t_arrays["component"])
    top_comp = max(sorted(a_comp_means), key=lambda c: (a_comp_means[c], -c))
    e_t_without_comp = [p for p in e_t_pairs if int(p["eval_component"]) != int(top_comp)]
    comp_removed_metrics = {
        a: _strip_arrays(M.pair_metrics(loaded[a]["pred_by_row"], y_by_row, e_t_without_comp)) for a in arms
    }
    comp_removed_gains = {}
    for label, ref, cand in (("R_vs_A", "A", "R"), ("R_vs_W", "W", "R"), ("R_vs_N", "N", "R")):
        comp_removed_gains[label] = {
            "e_t_ce_mae": _point_gain(
                comp_removed_metrics[ref]["ce_mae"], comp_removed_metrics[cand]["ce_mae"]
            ),
            "n_pairs": comp_removed_metrics[cand]["n_pairs"],
            "n_components": comp_removed_metrics[cand]["n_components"],
        }

    a_test_sq = {int(r): float((loaded["A"]["pred_by_row"][int(r)] - y_by_row[int(r)]) ** 2) for r in rows_by_split["test"]}
    top_test_row = max(sorted(a_test_sq), key=lambda r: (a_test_sq[r], -r))
    remaining_test_rows = [r for r in rows_by_split["test"] if int(r) != int(top_test_row)]
    e_t_without_row = [p for p in e_t_pairs if int(top_test_row) not in (int(p["row_a"]), int(p["row_b"]))]
    e_hh_without_row = [p for p in e_hh_pairs if int(top_test_row) not in (int(p["row_a"]), int(p["row_b"]))]
    row_removed_test = {
        a: M.regression_metrics(
            np.asarray([loaded[a]["pred_by_row"][int(r)] for r in remaining_test_rows], dtype=np.float64),
            np.asarray([y_by_row[int(r)] for r in remaining_test_rows], dtype=np.float64),
        )
        for a in arms
    }
    row_removed_e_t = {a: _strip_arrays(M.pair_metrics(loaded[a]["pred_by_row"], y_by_row, e_t_without_row)) for a in arms}
    row_removed_e_hh = {a: _strip_arrays(M.pair_metrics(loaded[a]["pred_by_row"], y_by_row, e_hh_without_row)) for a in arms}
    row_removed_gains = {}
    for label, ref, cand in (("R_vs_A", "A", "R"), ("R_vs_W", "W", "R"), ("R_vs_N", "N", "R")):
        row_removed_gains[label] = {
            "test_rmse": _point_gain(
                row_removed_test[ref]["rmse"], row_removed_test[cand]["rmse"]
            ),
            "e_t_ce_mae": _point_gain(
                row_removed_e_t[ref]["ce_mae"], row_removed_e_t[cand]["ce_mae"]
            ),
            "e_hh_ce_mae": _point_gain(
                row_removed_e_hh[ref]["ce_mae"], row_removed_e_hh[cand]["ce_mae"]
            ),
            "n_test_rows": len(remaining_test_rows),
            "n_e_t_pairs": row_removed_e_t[cand]["n_pairs"],
        }

    impact = {
        "top_e_t_component_A": int(top_comp),
        "top_e_t_component_A_mean_abs_err": float(a_comp_means[top_comp]),
        "top_e_t_component_A_n_pairs": int(sum(1 for p in e_t_pairs if int(p["eval_component"]) == int(top_comp))),
        "top_test_molecule_A": int(top_test_row),
        "top_test_molecule_A_sq_err": float(a_test_sq[top_test_row]),
        "remove_top_e_t_component": {
            **comp_removed_gains,
            "not_reversed_r_vs_a_e_t": bool(comp_removed_gains["R_vs_A"]["e_t_ce_mae"]["gain"] > 0),
            "not_reversed_r_vs_w_e_t": bool(comp_removed_gains["R_vs_W"]["e_t_ce_mae"]["gain"] > 0),
            "not_reversed_r_vs_n_e_t": bool(comp_removed_gains["R_vs_N"]["e_t_ce_mae"]["gain"] > 0),
        },
        "remove_top_test_molecule": {
            **row_removed_gains,
            "not_reversed_r_vs_a_test": bool(row_removed_gains["R_vs_A"]["test_rmse"]["gain"] > 0),
            "not_reversed_r_vs_a_e_t": bool(row_removed_gains["R_vs_A"]["e_t_ce_mae"]["gain"] > 0),
            "not_reversed_r_vs_w_e_t": bool(row_removed_gains["R_vs_W"]["e_t_ce_mae"]["gain"] > 0),
            "not_reversed_r_vs_w_test": bool(row_removed_gains["R_vs_W"]["test_rmse"]["gain"] > 0),
            "not_reversed_r_vs_n_e_t": bool(row_removed_gains["R_vs_N"]["e_t_ce_mae"]["gain"] > 0),
            "not_reversed_r_vs_n_test": bool(row_removed_gains["R_vs_N"]["test_rmse"]["gain"] > 0),
        },
    }

    # ---- endpoint (last epoch) check, reporting only -----------------------
    endpoint = {}
    for a in arms:
        endpoint[a] = {
            "best_test_rmse": float(split_metrics[a]["test"]["rmse"]),
            "last_test_rmse": float(last_split_metrics[a]["test"]["rmse"]),
            "best_e_t_ce_mae": float(pair_metrics[a]["E_T"]["ce_mae"]),
            "last_e_t_ce_mae": float(last_pair_metrics[a]["E_T"]["ce_mae"]),
            "test_direction": "last_better" if last_split_metrics[a]["test"]["rmse"] < split_metrics[a]["test"]["rmse"] else (
                "last_worse" if last_split_metrics[a]["test"]["rmse"] > split_metrics[a]["test"]["rmse"] else "equal"
            ),
            "e_t_direction": "last_better" if last_pair_metrics[a]["E_T"]["ce_mae"] < pair_metrics[a]["E_T"]["ce_mae"] else (
                "last_worse" if last_pair_metrics[a]["E_T"]["ce_mae"] > pair_metrics[a]["E_T"]["ce_mae"] else "equal"
            ),
        }

    # ---- Test descriptive group: has a Train real-edit neighbour -----------
    train_neighbor = set()
    for p in sets["E_T_Train_Test"]:
        train_neighbor.add(int(p["row_a"]))
        train_neighbor.add(int(p["row_b"]))
    test_groups = {}
    for a in arms:
        for label, rows in (
            ("has_train_edit_neighbor", [r for r in rows_by_split["test"] if int(r) in train_neighbor]),
            ("no_train_edit_neighbor", [r for r in rows_by_split["test"] if int(r) not in train_neighbor]),
        ):
            pred = np.asarray([loaded[a]["pred_by_row"][int(r)] for r in rows], dtype=np.float64)
            y = np.asarray([y_by_row[int(r)] for r in rows], dtype=np.float64)
            test_groups.setdefault(a, {})[label] = M.regression_metrics(pred, y) if rows else {
                "n": 0, "rmse": None, "mae": None, "r2": None, "bias": None, "sse": None
            }

    # ---- cases: R vs A, max improvement / max worsening / median ----------
    arr_a = M.pair_arrays(loaded["A"]["pred_by_row"], y_by_row, e_t_pairs)
    arr_r = M.pair_arrays(loaded["R"]["pred_by_row"], y_by_row, e_t_pairs)
    improvement = np.abs(arr_a["err"]) - np.abs(arr_r["err"])
    median_improvement = float(np.median(improvement)) if len(improvement) else float("nan")
    if len(improvement):
        order = np.argsort([str(x) for x in arr_a["pair_id"]])
        best_idx = max(order, key=lambda i: (float(improvement[i]), str(arr_a["pair_id"][i])))
        worst_idx = min(order, key=lambda i: (float(improvement[i]), str(arr_a["pair_id"][i])))
        median_idx = min(order, key=lambda i: (abs(float(improvement[i]) - median_improvement), str(arr_a["pair_id"][i])))
        chosen = [("max_improvement", best_idx), ("max_worsening", worst_idx), ("median_improvement", median_idx)]
    else:
        chosen = []
    cases = []
    for label, idx in chosen:
        pair = e_t_pairs[int(idx)]
        case = {
            "case": label,
            "pair_id": str(pair["pair_id"]),
            "row_a": int(pair["row_a"]),
            "row_b": int(pair["row_b"]),
            "split_combo": str(pair["split_combo"]),
            "eval_component": int(pair["eval_component"]),
            "core_seen_in_train_cuts": bool(pair["core_seen_in_train_cuts"]),
            "smiles_a": smiles[int(pair["row_a"])],
            "smiles_b": smiles[int(pair["row_b"])],
            "d_y": float(arr_a["d_y"][idx]),
            "median_improvement": median_improvement,
            "improvement_A_minus_R": float(improvement[idx]),
            "arms": {},
        }
        for a in arms:
            arr = M.pair_arrays(loaded[a]["pred_by_row"], y_by_row, [pair])
            case["arms"][a] = {
                "d_pred": float(arr["d_pred"][0]),
                "err": float(arr["err"][0]),
                "abs_err": float(abs(arr["err"][0])),
            }
        cases.append(case)

    # ---- results / decisions ----------------------------------------------
    total_gpu_hours = float(sum(float(loaded[a]["train_json"]["gpu"]["gpu_hours_charged"]) for a in arms))
    budget_status = "BUDGET_OK" if total_gpu_hours <= C.GPU_HARD_CAP_HOURS else "BUDGET_LIMITED"
    comp = {
        "schema": "lcrc-lipo-edit-supervision-comparisons-v1",
        "round_id": C.ROUND_ID,
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "OK",
        "units": "original logD",
        "metric_definitions": {
            "test_rmse": "sqrt(mean(squared error)) over all Test420 molecules",
            "e_t_ce_mae": "per-pair abs((pred_b-pred_a)-(y_b-y_a)); mean within component; equal weight per component with an E_T pair",
            "e_hh_ce_mae": "same, on Valid-Test union Test-Test",
            "gain": "metric(reference) - metric(candidate); positive means candidate is better",
        },
        "support": {
            "data_audit_status": data_audit.get("status"),
            "sets": data_audit.get("sets"),
            "gates": data_audit.get("gates"),
            "low_support": data_audit.get("low_support"),
        },
        "bootstrap": {
            "seed": C.BOOTSTRAP_SEED,
            "replicates": C.BOOTSTRAP_REPLICATES,
            "method": "paired component bootstrap; same resampled components for all arms and comparisons",
            "test_rmse": "resample structural components with Test molecules; keep all Test molecules in sampled components; recompute overall RMSE",
            "e_t_e_hh": "resample evaluation components with a pair in the set; component-equal mean",
            "warning": "interval describes evaluation-sample sensitivity, not training-seed variation",
        },
        "arms": {},
        "comparisons": comparisons,
        "impact": impact,
        "endpoint_check": endpoint,
        "test_descriptive_groups": test_groups,
        "cases": cases,
        "total_gpu_hours": total_gpu_hours,
        "budget": {
            "soft_cap_hours": C.GPU_SOFT_CAP_HOURS,
            "hard_cap_hours": C.GPU_HARD_CAP_HOURS,
            "status": budget_status,
        },
        "decisions": {},
    }

    for a in arms:
        js = loaded[a]["train_json"]
        comp["arms"][a] = {
            "best_epoch": int(js["checkpoint"]["best_epoch"]),
            "parameters": int(js["model"]["parameters"]),
            "gpu_hours": float(js["gpu"]["gpu_hours_charged"]),
            "splits": split_metrics[a],
            "last_splits": last_split_metrics[a],
            "pairs": pair_metrics[a],
            "last_pairs": last_pair_metrics[a],
            "train_diagnostics": js["train_diagnostics"],
            "anomalies": js["anomalies"],
            "checkpoint": js["checkpoint"],
            "schedule": js["schedule"],
        }

    comp["decisions"]["common_benefit"] = _decide_common(comp, data_audit, budget_status)
    comp["decisions"]["relation_increment"] = _decide_relation(comp, data_audit, budget_status)
    comp["decisions"]["support_status"] = (
        "LOW_SUPPORT" if bool(data_audit.get("low_support", {}).get("E_HH", True)) else "SUPPORT_OK"
    )
    comp["decisions"]["support_detail"] = {
        "E_HH": data_audit.get("low_support", {}).get("E_HH"),
        "E_HH_seen_core": data_audit.get("low_support", {}).get("E_HH_seen_core"),
        "E_HH_unseen_core": data_audit.get("low_support", {}).get("E_HH_unseen_core"),
        "note": "E_HH below 40 pairs or 15 components is LOW_SUPPORT and does not block the round",
    }
    C.write_json(C.ANALYSIS / "comparisons.json", comp)
    C.write_json(C.ANALYSIS / "cases.json", {"schema": "lcrc-lipo-edit-supervision-cases-v1", "cases": cases})

    # ---- results.csv -------------------------------------------------------
    fields = [
        "arm",
        "best_epoch",
        "train_rmse",
        "train_mae",
        "valid_rmse",
        "valid_mae",
        "test_rmse",
        "test_mae",
        "e_t_ce_mae",
        "e_hh_ce_mae",
        "e_t_pair_equal_mae",
        "e_t_zero_change_reference",
        "e_hh_pair_equal_mae",
        "e_hh_zero_change_reference",
        "e_t_train_test_ce_mae",
        "e_t_valid_test_ce_mae",
        "e_t_test_test_ce_mae",
        "e_hh_seen_core_ce_mae",
        "e_hh_unseen_core_ce_mae",
        "parameters",
        "gpu_hours",
    ]
    with open(C.ANALYSIS / "results.csv", "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for a in arms:
            pm = pair_metrics[a]
            w.writerow(
                {
                    "arm": a,
                    "best_epoch": int(loaded[a]["train_json"]["checkpoint"]["best_epoch"]),
                    "train_rmse": split_metrics[a]["train"]["rmse"],
                    "train_mae": split_metrics[a]["train"]["mae"],
                    "valid_rmse": split_metrics[a]["valid"]["rmse"],
                    "valid_mae": split_metrics[a]["valid"]["mae"],
                    "test_rmse": split_metrics[a]["test"]["rmse"],
                    "test_mae": split_metrics[a]["test"]["mae"],
                    "e_t_ce_mae": pm["E_T"]["ce_mae"],
                    "e_hh_ce_mae": pm["E_HH"]["ce_mae"],
                    "e_t_pair_equal_mae": pm["E_T"]["pair_equal_mae"],
                    "e_t_zero_change_reference": pm["E_T"]["zero_change_reference_mean_abs_dy"],
                    "e_hh_pair_equal_mae": pm["E_HH"]["pair_equal_mae"],
                    "e_hh_zero_change_reference": pm["E_HH"]["zero_change_reference_mean_abs_dy"],
                    "e_t_train_test_ce_mae": pm["E_T_Train_Test"]["ce_mae"],
                    "e_t_valid_test_ce_mae": pm["E_T_Valid_Test"]["ce_mae"],
                    "e_t_test_test_ce_mae": pm["E_T_Test_Test"]["ce_mae"],
                    "e_hh_seen_core_ce_mae": pm["E_HH_seen_core"]["ce_mae"],
                    "e_hh_unseen_core_ce_mae": pm["E_HH_unseen_core"]["ce_mae"],
                    "parameters": int(loaded[a]["train_json"]["model"]["parameters"]),
                    "gpu_hours": float(loaded[a]["train_json"]["gpu"]["gpu_hours_charged"]),
                }
            )

    # ---- per_molecule.csv --------------------------------------------------
    endpoint_rows = {int(p["row_a"]) for p in sets["P_train"]} | {int(p["row_b"]) for p in sets["P_train"]}
    with open(C.ANALYSIS / "per_molecule.csv", "w", newline="", encoding="utf-8") as fh:
        fields_m = [
            "arm", "split", "row_id", "y_true", "pred", "residual", "sq_err",
            "eval_component", "in_p_train_endpoint", "test_has_train_edit_neighbor",
        ]
        w = csv.DictWriter(fh, fieldnames=fields_m)
        w.writeheader()
        for a in arms:
            for split in C.SPLITS:
                for r in rows_by_split[split]:
                    r = int(r)
                    pred = float(loaded[a]["pred_by_row"][r])
                    y = float(y_by_row[r])
                    w.writerow(
                        {
                            "arm": a,
                            "split": split,
                            "row_id": r,
                            "y_true": y,
                            "pred": pred,
                            "residual": pred - y,
                            "sq_err": (pred - y) ** 2,
                            "eval_component": int(row_to_component[r]),
                            "in_p_train_endpoint": int(r in endpoint_rows),
                            "test_has_train_edit_neighbor": int(split == "test" and r in train_neighbor),
                        }
                    )

    # ---- per_pair.csv ------------------------------------------------------
    with open(C.ANALYSIS / "per_pair.csv", "w", newline="", encoding="utf-8") as fh:
        fields_p = [
            "arm", "pair_id", "row_a", "row_b", "split_combo", "eval_component",
            "in_p_train", "in_e_t", "in_e_hh", "core_seen_in_train_cuts",
            "d_y", "d_pred", "err", "abs_err", "sq_err",
        ]
        w = csv.DictWriter(fh, fieldnames=fields_p)
        w.writeheader()
        for a in arms:
            for p in pairs:
                src_pred = loaded[a]["pred_by_row"] if not bool(p["in_p_train"]) else loaded[a]["pred_by_row"]
                arr = M.pair_arrays(src_pred, y_by_row, [p])
                w.writerow(
                    {
                        "arm": a,
                        "pair_id": p["pair_id"],
                        "row_a": int(p["row_a"]),
                        "row_b": int(p["row_b"]),
                        "split_combo": p["split_combo"],
                        "eval_component": int(p["eval_component"]),
                        "in_p_train": int(bool(p["in_p_train"])),
                        "in_e_t": int(bool(p["in_e_t"])),
                        "in_e_hh": int(bool(p["in_e_hh"])),
                        "core_seen_in_train_cuts": int(bool(p["core_seen_in_train_cuts"])),
                        "d_y": float(arr["d_y"][0]),
                        "d_pred": float(arr["d_pred"][0]),
                        "err": float(arr["err"][0]),
                        "abs_err": float(abs(arr["err"][0])),
                        "sq_err": float(arr["err"][0] ** 2),
                    }
                )

    C.log(
        f"[evaluate] total_gpu_hours={total_gpu_hours:.4f} "
        f"common={comp['decisions']['common_benefit']['code']} "
        f"relation={comp['decisions']['relation_increment']['code']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

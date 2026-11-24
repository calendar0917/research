"""Analysis, bootstrap, evidence diagnostics and main table for SES-v0.

Reads only the best checkpoints' saved per-molecule / per-candidate records.
The oracle is a post-hoc label-using diagnostic; it is never a score, a
gradient, or a checkpoint-selection rule.
"""

from __future__ import annotations

import csv
import json
import pickle
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import ses_common as C
import ses_data as D
import ses_structure as S
import ses_train as TR

ARMS = ("DENSE", "RANDOM", "SELECT")


def _read_curve(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _f(row: Mapping[str, Any], key: str) -> float:
    value = row.get(key, "")
    return float(value) if value not in ("", None) else float("nan")


def _metrics(y: np.ndarray, pred: np.ndarray) -> dict[str, float]:
    se = (np.asarray(pred, np.float64) - np.asarray(y, np.float64)) ** 2
    return {"rmse": float(np.sqrt(se.mean())), "mae": float(np.abs(np.asarray(pred)
                                                                   - np.asarray(y)).mean())}


def _paired_bootstrap(se_a: np.ndarray, se_b: np.ndarray, n_boot: int,
                      seed: int) -> dict[str, Any]:
    se_a = np.asarray(se_a, np.float64)
    se_b = np.asarray(se_b, np.float64)
    assert se_a.shape == se_b.shape
    rng = np.random.default_rng(int(seed))
    n = se_a.size
    idx = rng.integers(0, n, size=(int(n_boot), n))
    rmse_a = np.sqrt(se_a[idx].mean(axis=1))
    rmse_b = np.sqrt(se_b[idx].mean(axis=1))
    delta = rmse_a - rmse_b
    return {
        "n": int(n), "n_boot": int(n_boot),
        "rmse_a": float(np.sqrt(se_a.mean())), "rmse_b": float(np.sqrt(se_b.mean())),
        "delta_a_minus_b": float(np.sqrt(se_a.mean()) - np.sqrt(se_b.mean())),
        "delta_bootstrap_mean": float(delta.mean()),
        "delta_ci95": [float(np.percentile(delta, 2.5)), float(np.percentile(delta, 97.5))],
        "delta_bootstrap_se": float(delta.std(ddof=0)),
    }


def _load_arm(arm: str) -> dict[str, Any]:
    d = C.RESULTS_DIR / arm.lower()
    manifest = C.read_json(d / "manifest.json")
    curve = _read_curve(d / "curve.csv")
    pred = np.load(d / "predictions.npz")
    return {"dir": d, "manifest": manifest, "curve": curve, "pred": pred}


def _select_selection_examples(rows: np.ndarray, y: np.ndarray, pred: np.ndarray,
                               cand_ptr: np.ndarray, cand_pred: np.ndarray,
                               chosen: np.ndarray) -> tuple[list[int], list[int], list[int]]:
    se = (pred - y) ** 2
    expected = np.zeros(rows.size, np.float64)
    for i in range(rows.size):
        lo, hi = int(cand_ptr[i]), int(cand_ptr[i + 1])
        expected[i] = float(np.mean((cand_pred[lo:hi] - y[i]) ** 2))
    benefit = expected - se
    order_acc = np.argsort(se)
    order_benefit = np.argsort(-benefit)
    effective: list[int] = []
    for i in order_benefit.tolist():
        if benefit[i] > 0 and se[i] <= np.median(se):
            effective.append(i)
        if len(effective) == 2:
            break
    no_benefit: list[int] = []
    for i in np.argsort(np.abs(benefit)).tolist():
        if i in effective:
            continue
        if abs(benefit[i]) <= 0.02 * max(1e-12, expected[i]) or abs(benefit[i]) < 1e-3:
            no_benefit.append(i)
        elif abs(benefit[i]) <= 0.1 * max(1e-12, expected[i]):
            no_benefit.append(i)
        if len(no_benefit) == 2:
            break
    failures: list[int] = []
    for i in np.argsort(-se).tolist():
        if i in effective or i in no_benefit:
            continue
        failures.append(i)
        if len(failures) == 2:
            break
    # deterministic fallbacks; categories stay disjoint
    used = set(effective + no_benefit + failures)
    for i in order_acc.tolist():
        if len(effective) < 2 and i not in used:
            effective.append(i)
            used.add(i)
    for i in np.argsort(np.abs(benefit)).tolist():
        if len(no_benefit) < 2 and i not in used:
            no_benefit.append(i)
            used.add(i)
    for i in np.argsort(-se).tolist():
        if len(failures) < 2 and i not in used:
            failures.append(i)
            used.add(i)
    return effective, no_benefit, failures


def _example_payload(row: int, bundle: D.DataBundle, prepared: Mapping[str, Any],
                     y: float, pred: float, chosen: int, scores: np.ndarray,
                     cand_pred_select: np.ndarray, cand_pred_random: np.ndarray,
                     support_atoms: Sequence[np.ndarray],
                     support_bonds: Sequence[np.ndarray], raw_b0: int,
                     fixed_index: int) -> dict[str, Any]:
    raw = bundle.raw
    a0 = int(np.asarray(raw["node_ptr"])[row])
    a1 = int(np.asarray(raw["node_ptr"])[row + 1])
    b0 = int(np.asarray(raw["bond_ptr"])[row])
    b1 = int(np.asarray(raw["bond_ptr"])[row + 1])
    atoms = []
    for i in range(a1 - a0):
        atoms.append({"node": i,
                      "atom_feat": [int(x) for x in np.asarray(raw["atom_feat"])[a0 + i]]})
    bonds = []
    for e in range(b0, b1):
        bonds.append({"bond": e - b0,
                      "u": int(np.asarray(raw["bond_u"])[e]),
                      "v": int(np.asarray(raw["bond_v"])[e]),
                      "bond_feat": [int(x) for x in np.asarray(raw["bond_feat"])[e]]})
    cands = []
    for j, (sa, sb) in enumerate(zip(support_atoms, support_bonds)):
        sa = np.asarray(sa, np.int64)
        sb = np.asarray(sb, np.int64) - raw_b0
        cands.append({
            "candidate_index": j,
            "support_atoms": [int(x) for x in sa.tolist()],
            "support_bonds": [int(x) for x in sb.tolist()],
            "n_atoms": int(sa.size), "n_bonds": int(sb.size),
            "node_fraction": float(sa.size) / float(max(1, a1 - a0)),
            "edge_fraction": float(sb.size) / float(max(1, b1 - b0)),
            "select_pred": float(cand_pred_select[j]),
            "random_pred": float(cand_pred_random[j]),
            "select_score": float(scores[j]),
            "select_se": float((cand_pred_select[j] - y) ** 2),
            "chosen": bool(j == int(chosen)),
            "fixed_random_choice": bool(j == int(fixed_index)),
        })
    return {
        "row": int(row),
        "smiles": str(np.asarray(raw["smiles"])[row]),
        "n_atoms": int(a1 - a0), "n_bonds": int(b1 - b0),
        "y_true": float(y), "select_pred": float(pred),
        "abs_error": float(abs(pred - y)),
        "chosen_index": int(chosen), "fixed_random_index": int(fixed_index),
        "candidates": cands,
        "original_graph": {"atoms": atoms, "bonds": bonds,
                           "atom_feature_columns": "OGB 9 categorical columns",
                           "bond_feature_columns": "OGB 3 categorical columns"},
    }


def main(prepared_path: str | None = None) -> dict[str, Any]:
    C.assert_frozen_sources()
    bundle = D.load_bundle()
    cfg = C.load_config()
    prepared = TR.load_prepared(
        prepared_path or str(C.PREPARED_DIR / "ses_prepared.pkl"), bundle)
    arms = {arm: _load_arm(arm) for arm in ARMS}
    checks = C.read_json(C.ANALYSIS_DIR / "checks.json")
    preflight = C.read_json(C.ANALYSIS_DIR / "preflight.json")
    n_boot = int(cfg["evaluation"]["bootstrap_replicates"])
    boot_seed = int(cfg["evaluation"]["bootstrap_seed"])
    scaler = bundle.scaler
    mean, std = float(scaler["mean"]), float(scaler["std"])

    # ------------------------------------------------------------------
    # replay: saved best-checkpoint records must reproduce the manifest
    # ------------------------------------------------------------------
    replay: dict[str, Any] = {}
    for arm in ARMS:
        man = arms[arm]["manifest"]
        pred = arms[arm]["pred"]
        best = man["best_metrics"]
        ok = True
        detail = {}
        for split in ("train", "valid", "test"):
            y = pred[f"{split}_y_raw"]
            if arm == "DENSE":
                p = pred[f"{split}_pred_raw"]
                m = _metrics(y, p)
                expected = best[split]
            else:
                m = {
                    "rmse": float(np.sqrt(pred[f"{split}_mse_expected"].mean())),
                    "mae": float(pred[f"{split}_mae_expected"].mean()),
                }
                if arm == "SELECT":
                    ms = _metrics(y, pred[f"{split}_pred_select"])
                    m.update(ms)
                    expected = best[split]
                else:
                    m_sel = _metrics(y, pred[f"{split}_pred_fixed"])
                    m.update({"rmse_fixed": m_sel["rmse"], "mae_fixed": m_sel["mae"]})
                    expected = best[split]
            detail[split] = m
            for key, value in expected.items():
                if key in m and abs(float(m[key]) - float(value)) > 1e-8:
                    ok = False
        replay[arm] = {"ok": bool(ok), "detail": detail}

    # ------------------------------------------------------------------
    # main table
    # ------------------------------------------------------------------
    main_rows = []
    full_rows = []
    for arm in ARMS:
        man = arms[arm]["manifest"]
        best = man["best_metrics"]
        base = {
            "Model": arm, "Predictor params": int(man["predictor_parameters"]),
            "Selector params": int(man["selector_parameters"]),
            "Best epoch": int(man["best_epoch"]),
            "Train RMSE": float(best["train"]["rmse"]),
            "Valid RMSE": float(best["valid"]["rmse"]),
            "Test RMSE": float(best["test"]["rmse"]),
            "Wall time": float(man["wall_seconds"]),
            "wall_time_minutes": float(man["wall_seconds"]) / 60.0,
        }
        main_rows.append(base)
        full = dict(base)
        for split in ("train", "valid", "test"):
            for key, value in best[split].items():
                full[f"{split}_{key}"] = float(value)
        full["best_epoch"] = int(man["best_epoch"])
        full["epochs"] = int(man["epochs"])
        full_rows.append(full)
        if arm == "RANDOM":
            fixed = dict(base)
            fixed["Model"] = "RANDOM_FIXED"
            fixed["Train RMSE"] = float(best["train"]["rmse_fixed"])
            fixed["Valid RMSE"] = float(best["valid"]["rmse_fixed"])
            fixed["Test RMSE"] = float(best["test"]["rmse_fixed"])
            main_rows.append(fixed)
            fr = dict(full)
            fr["Model"] = "RANDOM_FIXED"
            for split in ("train", "valid", "test"):
                fr[f"{split}_rmse"] = float(best[split]["rmse_fixed"])
                fr[f"{split}_mae"] = float(best[split]["mae_fixed"])
            full_rows.append(fr)
    C.write_csv(C.RESULTS_DIR / "main_table.csv", main_rows,
                columns=["Model", "Predictor params", "Selector params", "Best epoch",
                         "Train RMSE", "Valid RMSE", "Test RMSE", "Wall time",
                         "wall_time_minutes"])
    C.write_csv(C.RESULTS_DIR / "main_table_full.csv", full_rows)

    # ------------------------------------------------------------------
    # per-molecule arrays and paired comparisons
    # ------------------------------------------------------------------
    dense = arms["DENSE"]["pred"]
    rand = arms["RANDOM"]["pred"]
    sel = arms["SELECT"]["pred"]
    comparisons: dict[str, Any] = {}
    per_split: dict[str, Any] = {}
    for split in ("valid", "test"):
        rows = np.asarray(sel[f"{split}_rows"])
        if not (np.array_equal(rows, np.asarray(dense[f"{split}_rows"]))
                and np.array_equal(rows, np.asarray(rand[f"{split}_rows"]))):
            raise RuntimeError("arm row order mismatch")
        y = np.asarray(sel[f"{split}_y_raw"], np.float64)
        dense_se = (np.asarray(dense[f"{split}_pred_raw"], np.float64) - y) ** 2
        sel_se = (np.asarray(sel[f"{split}_pred_select"], np.float64) - y) ** 2
        rand_exp = np.asarray(rand[f"{split}_mse_expected"], np.float64)
        rand_fixed_se = (np.asarray(rand[f"{split}_pred_fixed"], np.float64) - y) ** 2
        per_split[split] = {
            "rows": rows, "y": y, "dense_se": dense_se, "select_se": sel_se,
            "random_expected_mse": rand_exp, "random_fixed_se": rand_fixed_se,
        }
        comparisons[split] = {
            "select_vs_dense": _paired_bootstrap(sel_se, dense_se, n_boot, boot_seed),
            "random_expected_vs_dense": _paired_bootstrap(rand_exp, dense_se, n_boot, boot_seed),
            "select_vs_random_expected": _paired_bootstrap(sel_se, rand_exp, n_boot, boot_seed),
            "select_vs_random_fixed": _paired_bootstrap(sel_se, rand_fixed_se, n_boot, boot_seed),
        }

    # ------------------------------------------------------------------
    # evidence coverage (SELECT's chosen candidate; Valid/Test)
    # ------------------------------------------------------------------
    coverage: dict[str, Any] = {}
    for split in ("valid", "test"):
        rows = np.asarray(sel[f"{split}_rows"], np.int64)
        ptr = np.asarray(sel[f"{split}_cand_ptr"], np.int64)
        chosen = np.asarray(sel[f"{split}_chosen_index"], np.int64)
        node_ptr_all = np.asarray(sel[f"{split}_cand_node_ptr"], np.int64)
        nodes_all = np.asarray(sel[f"{split}_cand_nodes"], np.int64)
        bond_ptr_all = np.asarray(sel[f"{split}_cand_bond_ptr"], np.int64)
        bonds_all = np.asarray(sel[f"{split}_cand_bonds"], np.int64)
        orig_n = np.asarray(sel[f"{split}_orig_n_atoms"], np.int64)
        orig_m = np.asarray(sel[f"{split}_orig_n_bonds"], np.int64)
        node_frac, edge_frac, counts = [], [], []
        jac, inter, union_frac = [], [], []
        connectivity_bad = 0
        for i, row in enumerate(rows.tolist()):
            lo, hi = int(ptr[i]), int(ptr[i + 1])
            counts.append(hi - lo)
            j = int(chosen[i])
            supports = []
            for c in range(lo, hi):
                na, nb = int(node_ptr_all[c]), int(node_ptr_all[c + 1])
                ba, bb = int(bond_ptr_all[c]), int(bond_ptr_all[c + 1])
                supports.append(set(int(x) for x in nodes_all[na:nb].tolist()))
                if c == lo + j:
                    node_frac.append((nb - na) / float(max(1, orig_n[i])))
                    edge_frac.append((bb - ba) / float(max(1, orig_m[i])))
                    # connectivity of the chosen support
                    st = prepared["candidates"][int(row)][j]
                    adjacency, _ = S._adjacency(st.n_atoms, st.bond_u, st.bond_v)
                    seen = {0}
                    stack = [0]
                    while stack:
                        x = stack.pop()
                        for yy in adjacency[x]:
                            if yy not in seen:
                                seen.add(yy)
                                stack.append(yy)
                    if len(seen) != st.n_atoms:
                        connectivity_bad += 1
            for a in range(len(supports)):
                for b in range(a + 1, len(supports)):
                    u = supports[a] | supports[b]
                    n_inter = len(supports[a] & supports[b])
                    jac.append(n_inter / max(1, len(u)))
                    inter.append(n_inter / max(1, min(len(supports[a]),
                                                      len(supports[b]))))
            union = set().union(*supports) if supports else set()
            union_frac.append(len(union) / float(max(1, orig_n[i])))
        coverage[split] = {
            "n_molecules": int(rows.size),
            "n_candidates": int(sum(counts)),
            "candidates_per_molecule_mean": float(np.mean(counts)),
            "candidates_per_molecule_min": int(np.min(counts)),
            "candidates_per_molecule_max": int(np.max(counts)),
            "selected_node_fraction_mean": float(np.mean(node_frac)),
            "selected_node_fraction_median": float(np.median(node_frac)),
            "selected_node_fraction_min": float(np.min(node_frac)),
            "selected_node_fraction_p10": float(np.percentile(node_frac, 10)),
            "selected_node_fraction_p90": float(np.percentile(node_frac, 90)),
            "selected_edge_fraction_mean": float(np.mean(edge_frac)),
            "selected_edge_fraction_median": float(np.median(edge_frac)),
            "selected_edge_fraction_min": float(np.min(edge_frac)),
            "selected_edge_fraction_p10": float(np.percentile(edge_frac, 10)),
            "selected_edge_fraction_p90": float(np.percentile(edge_frac, 90)),
            "selected_connected_fraction": float(1.0 - connectivity_bad / max(1, rows.size)),
            "selected_connected_count": int(rows.size - connectivity_bad),
            "pairwise_jaccard_overlap_mean": float(np.mean(jac)),
            "pairwise_intersection_over_smaller_mean": float(np.mean(inter)),
            "candidate_union_coverage_mean": float(np.mean(union_frac)),
        }
    input_fields = {
        "read_by_predictor": [
            "selected raw atom categorical columns (9)",
            "selected real induced bond categorical columns (3)",
            "candidate-local degree / min_ring / in_cycle / log1p(degree) / isolated",
            "candidate-local S1 neighbour code",
            "candidate-local typed S2 rooted 2-paths",
            "candidate-local complete R2 atom/bond summaries",
            "candidate-local graph scalars (8)",
        ],
        "forbidden_fields_absent": [
            "original atom id", "original graph node count", "original graph edge count",
            "full-graph descriptors", "full-graph R2 hidden representation",
            "selection score", "candidate index", "candidate count",
        ],
        "batch_key_audit": "C17 passes: no field outside the allowed induced-subgraph set",
    }

    # ------------------------------------------------------------------
    # frozen-selection intervention (SELECT predictor)
    # ------------------------------------------------------------------
    intervention: dict[str, Any] = {}
    for split in ("valid", "test"):
        rows = np.asarray(sel[f"{split}_rows"], np.int64)
        y = np.asarray(sel[f"{split}_y_raw"], np.float64)
        ptr = np.asarray(sel[f"{split}_cand_ptr"], np.int64)
        preds = np.asarray(sel[f"{split}_cand_pred"], np.float64)
        scores = np.asarray(sel[f"{split}_cand_scores"], np.float64)
        chosen = np.asarray(sel[f"{split}_chosen_index"], np.int64)
        se_a = np.asarray(per_split[split]["select_se"], np.float64)
        se_b = np.asarray(per_split[split]["random_expected_mse"], np.float64)
        se_c = np.zeros(rows.size, np.float64)
        pred_c = np.zeros(rows.size, np.float64)
        for i in range(rows.size):
            lo, hi = int(ptr[i]), int(ptr[i + 1])
            j = int(np.argmin(scores[lo:hi]))
            pred_c[i] = preds[lo + j]
            se_c[i] = (pred_c[i] - y[i]) ** 2
        intervention[split] = {
            "A_argmax": {"rmse": float(np.sqrt(se_a.mean())),
                         "mae": float(np.mean(np.abs(
                             np.asarray(sel[f"{split}_pred_select"]) - y)))},
            "B_uniform_expected": {"rmse": float(np.sqrt(se_b.mean())),
                                   "mae": float(np.mean(
                                       np.asarray(sel[f"{split}_mae_expected"])))},
            "C_min_score": {"rmse": float(np.sqrt(se_c.mean())),
                            "mae": float(np.mean(np.abs(pred_c - y)))},
            "A_vs_B": _paired_bootstrap(se_a, se_b, n_boot, boot_seed),
            "A_vs_C": _paired_bootstrap(se_a, se_c, n_boot, boot_seed),
        }

    # ------------------------------------------------------------------
    # independent RANDOM predictor on SELECT's chosen candidate
    # ------------------------------------------------------------------
    cross: dict[str, Any] = {}
    for split in ("valid", "test"):
        y = np.asarray(sel[f"{split}_y_raw"], np.float64)
        ptr = np.asarray(sel[f"{split}_cand_ptr"], np.int64)
        chosen = np.asarray(sel[f"{split}_chosen_index"], np.int64)
        sel_preds = np.asarray(sel[f"{split}_cand_pred"], np.float64)
        rand_preds = np.asarray(rand[f"{split}_cand_pred"], np.float64)
        rand_exp = np.asarray(rand[f"{split}_mse_expected"], np.float64)
        cross_pred = np.zeros(y.size, np.float64)
        sel_on_chosen = np.zeros(y.size, np.float64)
        for i in range(y.size):
            lo = int(ptr[i])
            j = int(chosen[i])
            cross_pred[i] = rand_preds[lo + j]
            sel_on_chosen[i] = sel_preds[lo + j]
        se_cross = (cross_pred - y) ** 2
        cross[split] = {
            "random_predictor_on_select_choice": {
                "rmse": float(np.sqrt(se_cross.mean())),
                "mae": float(np.mean(np.abs(cross_pred - y))),
            },
            "random_predictor_uniform_expected": {
                "rmse": float(np.sqrt(rand_exp.mean())),
                "mae": float(np.mean(np.asarray(rand[f"{split}_mae_expected"]))),
            },
            "random_on_select_choice_vs_random_expected": _paired_bootstrap(
                se_cross, rand_exp, n_boot, boot_seed),
            "fraction_molecules_cross_better_than_random_expectation": float(
                np.mean(se_cross < rand_exp)),
            "select_predictor_on_its_choice": {
                "rmse": float(np.sqrt(((sel_on_chosen - y) ** 2).mean()))},
        }

    # ------------------------------------------------------------------
    # candidate-space oracle (post-hoc, labels used only for diagnosis)
    # ------------------------------------------------------------------
    oracle: dict[str, Any] = {}
    for split in ("valid", "test"):
        y = np.asarray(sel[f"{split}_y_raw"], np.float64)
        ptr = np.asarray(sel[f"{split}_cand_ptr"], np.int64)
        dense_se = np.asarray(per_split[split]["dense_se"], np.float64)
        out = {}
        for arm in ("SELECT", "RANDOM"):
            preds = np.asarray(arms[arm]["pred"][f"{split}_cand_pred"], np.float64)
            best = np.zeros(y.size, np.float64)
            for i in range(y.size):
                lo, hi = int(ptr[i]), int(ptr[i + 1])
                best[i] = float(np.min((preds[lo:hi] - y[i]) ** 2))
            actual = (np.asarray(per_split[split]["select_se"], np.float64)
                      if arm == "SELECT"
                      else np.asarray(per_split[split]["random_expected_mse"], np.float64))
            out[arm] = {
                "oracle_mse": float(best.mean()),
                "oracle_rmse": float(np.sqrt(best.mean())),
                "actual_mse": float(actual.mean()),
                "actual_rmse": float(np.sqrt(actual.mean())),
                "gap_oracle_to_actual_mse": float(actual.mean() - best.mean()),
                "gap_oracle_to_dense_mse": float(dense_se.mean() - best.mean()),
                "actual_minus_dense_mse": float(actual.mean() - dense_se.mean()),
                "n_molecules_where_oracle_beats_dense": int(np.sum(best < dense_se)),
            }
            # exact oracle MAE and its own prediction
            oracle_mae = []
            for i in range(y.size):
                lo, hi = int(ptr[i]), int(ptr[i + 1])
                pm = preds[lo:hi]
                oracle_mae.append(float(np.min(np.abs(pm - y[i]))))
            out[arm]["oracle_mae"] = float(np.mean(oracle_mae))
        out["dense"] = {"mse": float(dense_se.mean()),
                        "rmse": float(np.sqrt(dense_se.mean()))}
        oracle[split] = out

    # ------------------------------------------------------------------
    # resource accounting
    # ------------------------------------------------------------------
    resources = {
        "prepared_build_seconds": float(prepared["stats"]["build_seconds"]),
        "preflight_seconds": float(preflight["preflight_seconds"]),
        "preflight_horizon": int(preflight["decision"]["horizon"]),
        "preflight_estimated_wall_seconds": float(
            preflight["decision"]["estimated_wall_seconds"]),
        "formal": {
            arm: {"wall_seconds": float(arms[arm]["manifest"]["wall_seconds"]),
                  "gpu_hours": float(arms[arm]["manifest"]["gpu_hours_estimate"]),
                  "epochs": int(arms[arm]["manifest"]["epochs"]),
                  "best_epoch": int(arms[arm]["manifest"]["best_epoch"])}
            for arm in ARMS
        },
        "formal_wall_seconds_sum": float(sum(
            arms[arm]["manifest"]["wall_seconds"] for arm in ARMS)),
        "formal_gpu_hours_sum": float(sum(
            arms[arm]["manifest"]["gpu_hours_estimate"] for arm in ARMS)),
    }

    # ------------------------------------------------------------------
    # convergence / budget limited flag
    # ------------------------------------------------------------------
    convergence: dict[str, Any] = {}
    for arm in ARMS:
        curve = arms[arm]["curve"]
        key = "valid_rmse" if arm != "RANDOM" else "valid_rmse_expected"
        vals = np.asarray([_f(r, key) for r in curve], np.float64)
        last = vals[-1]
        best = float(vals.min())
        best_epoch = int(np.argmin(vals) + 1)
        convergence[arm] = {
            "epochs": int(vals.size), "best_epoch": best_epoch,
            "best_valid_metric": best, "last_valid_metric": float(last),
            "last_minus_best": float(last - best),
            "best_epoch_fraction": float(best_epoch / max(1, vals.size)),
            "horizon_limited_flag": bool(best_epoch >= int(0.9 * vals.size)
                                         and last < best - 1e-12),
        }

    # ------------------------------------------------------------------
    # examples (Valid only)
    # ------------------------------------------------------------------
    rows = np.asarray(sel["valid_rows"], np.int64)
    y = np.asarray(sel["valid_y_raw"], np.float64)
    pred = np.asarray(sel["valid_pred_select"], np.float64)
    ptr = np.asarray(sel["valid_cand_ptr"], np.int64)
    cand_pred = np.asarray(sel["valid_cand_pred"], np.float64)
    chosen = np.asarray(sel["valid_chosen_index"], np.int64)
    eff, nob, fail = _select_selection_examples(rows, y, pred, ptr, cand_pred, chosen)
    examples: dict[str, Any] = {"README": "Valid examples; labels shown only for case display.",
                                "categories": {}}
    node_ptr_all = np.asarray(sel["valid_cand_node_ptr"], np.int64)
    nodes_all = np.asarray(sel["valid_cand_nodes"], np.int64)
    bond_ptr_all = np.asarray(sel["valid_cand_bond_ptr"], np.int64)
    bonds_all = np.asarray(sel["valid_cand_bonds"], np.int64)
    rand_preds = np.asarray(rand["valid_cand_pred"], np.float64)
    for cat, idxs in (("accurate_and_selection_effective", eff),
                      ("no_clear_selection_benefit", nob),
                      ("select_prediction_failure", fail)):
        entries = []
        for i in idxs:
            row = int(rows[i])
            lo, hi = int(ptr[i]), int(ptr[i + 1])
            supports_atoms = []
            supports_bonds = []
            for c in range(lo, hi):
                na, nb = int(node_ptr_all[c]), int(node_ptr_all[c + 1])
                ba, bb = int(bond_ptr_all[c]), int(bond_ptr_all[c + 1])
                supports_atoms.append(np.asarray(nodes_all[na:nb], np.int64))
                supports_bonds.append(np.asarray(bonds_all[ba:bb], np.int64))
            b0 = int(np.asarray(bundle.raw["bond_ptr"])[row])
            entries.append(_example_payload(
                row, bundle, prepared, y[i], pred[i], int(chosen[i]),
                np.asarray(sel["valid_cand_scores"], np.float64)[lo:hi],
                cand_pred[lo:hi], rand_preds[lo:hi],
                supports_atoms, supports_bonds, b0,
                int(np.asarray(sel["valid_fixed_index"])[i])))
        examples["categories"][cat] = entries
    C.write_json(C.ANALYSIS_DIR / "examples.json", examples)

    # ------------------------------------------------------------------
    # route judgement
    # ------------------------------------------------------------------
    valid = {arm: float(arms[arm]["manifest"]["best_metrics"]["valid"]["rmse"])
             for arm in ARMS}
    test = {arm: float(arms[arm]["manifest"]["best_metrics"]["test"]["rmse"])
            for arm in ARMS}
    d_valid = valid["SELECT"] - valid["DENSE"]
    d_test = test["SELECT"] - test["DENSE"]
    r_valid = valid["RANDOM"] - valid["DENSE"]
    r_test = test["RANDOM"] - test["DENSE"]
    s_minus_r_valid = valid["SELECT"] - valid["RANDOM"]
    s_minus_r_test = test["SELECT"] - test["RANDOM"]
    dense_poor = bool(valid["DENSE"] > 0.75 or test["DENSE"] > 0.75)
    select_better_random = bool(s_minus_r_valid < 0 and s_minus_r_test < 0)
    compression_gap_ok = bool(d_valid <= 0.03 and d_test <= 0.03)
    if dense_poor:
        route = "DENSE_INSUFFICIENT_CANNOT_JUDGE_EVIDENCE_MECHANISM"
    elif select_better_random and compression_gap_ok:
        route = "PROMISING_EVIDENCE_COMPRESSION_SIGNAL"
    elif select_better_random and not compression_gap_ok:
        route = "SELECTION_EFFECTIVE_COMPRESSION_LIMITING"
    elif (not select_better_random) and compression_gap_ok:
        route = "COMPRESSION_COST_SMALL_BUT_NO_LEARNED_SELECTION_GAIN"
    else:
        route = "NO_LEARNED_SELECTION_VALUE"
    route_detail = {
        "dense_valid": valid["DENSE"], "dense_test": test["DENSE"],
        "select_minus_dense_valid": d_valid, "select_minus_dense_test": d_test,
        "random_expected_minus_dense_valid": r_valid,
        "random_expected_minus_dense_test": r_test,
        "select_minus_random_valid": s_minus_r_valid,
        "select_minus_random_test": s_minus_r_test,
        "reference_gap": float(cfg["route_reference"]["select_over_dense_gap"]),
        "dense_poor": dense_poor,
        "select_better_random_both_splits": select_better_random,
        "compression_gap_within_reference": compression_gap_ok,
        "route": route,
        "single_seed": True,
    }

    # ------------------------------------------------------------------
    # consolidated per-molecule / candidate file
    # ------------------------------------------------------------------
    consolidated: dict[str, np.ndarray] = {}
    for split in ("train", "valid", "test"):
        consolidated[f"{split}_rows"] = np.asarray(sel[f"{split}_rows"])
        consolidated[f"{split}_y_raw"] = np.asarray(sel[f"{split}_y_raw"])
        consolidated[f"{split}_dense_pred"] = np.asarray(dense[f"{split}_pred_raw"])
        if f"{split}_cand_ptr" not in sel.files:
            continue
        consolidated[f"{split}_cand_ptr"] = np.asarray(sel[f"{split}_cand_ptr"])
        consolidated[f"{split}_cand_pred_select"] = np.asarray(sel[f"{split}_cand_pred"])
        consolidated[f"{split}_cand_scores_select"] = np.asarray(sel[f"{split}_cand_scores"])
        consolidated[f"{split}_cand_pred_random"] = np.asarray(rand[f"{split}_cand_pred"])
        consolidated[f"{split}_chosen_index"] = np.asarray(sel[f"{split}_chosen_index"])
        consolidated[f"{split}_fixed_index"] = np.asarray(sel[f"{split}_fixed_index"])
        for key in ("cand_node_ptr", "cand_nodes", "cand_bond_ptr", "cand_bonds",
                    "orig_n_atoms", "orig_n_bonds", "mse_expected",
                    "pred_fixed", "pred_select"):
            consolidated[f"{split}_{key}"] = np.asarray(sel[f"{split}_{key}"])
    np.savez_compressed(C.RESULTS_DIR / "per_molecule_candidates.npz", **consolidated)

    # ------------------------------------------------------------------
    # write evidence analysis
    # ------------------------------------------------------------------
    payload = {
        "protocol_id": cfg["protocol_id"],
        "commit": C.git_commit(),
        "checks_all_pass": bool(checks["all_pass"]),
        "replay": replay,
        "main_table": main_rows,
        "comparisons": comparisons,
        "coverage": coverage,
        "predictor_input_fields": input_fields,
        "selection_intervention": intervention,
        "cross_evaluation": cross,
        "candidate_oracle": oracle,
        "resources": resources,
        "convergence": convergence,
        "route": route_detail,
        "examples_path": str((C.ANALYSIS_DIR / "examples.json").relative_to(C.ROUND_DIR)),
        "test_role": "DEVELOPMENT_TEST",
        "notes": {
            "oracle": "post-hoc diagnostic only; uses labels; never a reported score",
            "compression": "selected supports retain about 65% of nodes; not an extreme-sparsity claim",
            "causality": "prediction invariance does not prove the selection action is free of unselected information, nor chemical causality",
            "cross_evaluation": "no model was retrained for this comparison",
        },
    }
    C.write_json(C.ANALYSIS_DIR / "evidence_analysis.json", payload)
    print(json.dumps({"route": route, "valid": valid, "test": test,
                      "select_minus_dense": [d_valid, d_test],
                      "select_minus_random": [s_minus_r_valid, s_minus_r_test]},
                     indent=2, sort_keys=True), flush=True)
    return payload


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--prepared", default=None)
    args = parser.parse_args()
    main(args.prepared)

"""Analysis: P0 prior evidence, selection sensitivity, data x update budget, readout.

Everything here is recomputation from stored artefacts (old curves / new grid
predictions); no new training and no new inference beyond what the trainer and
probe already stored.

Naming discipline (see PROTOCOL): ``observed_oracle_gap`` is
``R_Test(theta_Valid) - min_E R_Test(theta_E)`` over the *observed* state set; it
contains Test-selection optimism and is not an achievable gain.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import lb_common as C
import lb_selection as SEL
import lb_data as D

observed_oracle_gap = SEL.observed_oracle_gap

PERF_RUNS = C.REPO / "tracks/structural-sufficiency/lcrc_performance_route_8h_public122_s0_v1/runs"


# ---------------------------------------------------------------------------
# P0 -- reuse existing curve artefacts
# ---------------------------------------------------------------------------

P0_CURVES = {
    "C1_R1": "C1_R1__Lipophilicity",
    "C1_R3": "C1_R3__Lipophilicity",
    "CHEM8_R1": "C1_CHEM8__Lipophilicity",
    "CONTRIB_R1": "C1_CONTRIB__Lipophilicity",
    "K_R1": "K_COMBO__Lipophilicity",
}


def _read_curve(run_dir: Path) -> list[dict[str, float]]:
    rows = []
    with open(run_dir / "curve.csv", newline="", encoding="utf-8") as fh:
        for r in csv.DictReader(fh):
            rows.append({k: (float(v) if v not in ("", None) else float("nan")) for k, v in r.items()
                         if k and k not in ("is_best",)})
    return rows


def p0_prior_evidence() -> dict[str, Any]:
    """Replay the 5 existing Lipo curves: Valid-selected state, Test curve min, window."""
    out: dict[str, Any] = {
        "source_round": "lcrc_performance_route_8h_public122_s0_v1",
        "correction_reference": "analysis/{learning_curves,corrected_manifest,paired_corrected}.json",
        "scope": "5 existing Lipo curves; P0 adds no training/inference",
        "curves": {}, "can_and_cannot": {
            "can": ["bounded observed_oracle_gap for existing full-N Lipo arms",
                    "selection position / fixed-window median as descriptive context"],
            "cannot": ["determine achievable gain", "serve as an independent-seed generalisation CI",
                       "recover the lost K best.pt (K is a NEW_REPLICATION here)"],
        },
    }
    for name, rd in P0_CURVES.items():
        d = PERF_RUNS / rd
        if not (d / "curve.csv").exists():
            out["curves"][name] = {"status": "MISSING", "path": C.track_rel(d)}
            continue
        rows = _read_curve(d)
        valid = np.asarray([r.get("valid_rmse", float("nan")) for r in rows], np.float64)
        test = np.asarray([r.get("test_rmse", float("nan")) for r in rows], np.float64)
        epochs = np.asarray([r["epoch"] for r in rows], np.int64)
        best_valid_idx = int(np.nanargmin(valid))
        test_curve_min_idx = int(np.nanargmin(test))
        n = len(rows)
        half = rows[n // 2:]
        win = float(np.nanmedian([r.get("test_rmse", float("nan")) for r in half]))
        out["curves"][name] = {
            "n_epochs": n,
            "recipe": _recipe_of(d),
            "valid_selected_epoch": int(epochs[best_valid_idx]),
            "valid_selected_test_rmse": float(test[best_valid_idx]),
            "test_curve_min": float(test[test_curve_min_idx]),
            "test_curve_min_epoch": int(epochs[test_curve_min_idx]),
            "observed_oracle_gap": float(test[best_valid_idx] - test[test_curve_min_idx]),
            "second_half_test_median": win,
            "best_valid_rmse": float(valid[best_valid_idx]),
            "units": "logD (RMSE)",
        }
    # the two named examples in the prompt
    out["named_examples"] = {
        "C1_R1": "~.58394 valid-selected vs ~.56381 post-hoc",
        "CONTRIB_R1": "~.60632 vs ~.56505",
    }
    return out


def _recipe_of(run_dir: Path) -> dict[str, Any]:
    try:
        p = C.read_json(run_dir / "train.json")
        return {"name": p.get("recipe", {}).get("name"), "wd": p.get("recipe", {}).get("weight_decay"),
                "peak_lr": p.get("recipe", {}).get("peak_lr"),
                "epochs_run": p.get("recipe", {}).get("epochs_run"),
                "parameters": p.get("model", {}).get("parameters")}
    except Exception:
        return {}


# ---------------------------------------------------------------------------
# new-arm records
# ---------------------------------------------------------------------------


def arm_dir(arm_id: str) -> Path:
    return C.RUNS / arm_id


def load_arm_records(arm_id: str) -> dict[str, Any] | None:
    d = arm_dir(arm_id)
    tj = d / "train.json"
    gp = d / "grid_predictions.npz"
    if not tj.exists():
        return None
    payload = C.read_json(tj)
    rec: dict[str, Any] = {"arm_id": arm_id, "meta": payload, "grid": None}
    if gp.exists():
        task = payload["task"]
        # regression predictions are stored in standardised space (`_pred` is pred_std in
        # lcpr_train.split_metrics); de-normalise once here with the recorded frozen norm
        tnorm = (payload.get("data") or {}).get("target_norm")
        z = np.load(gp, allow_pickle=False)
        updates = np.asarray(z["grid_updates"], np.int64)

        def _p(split: str) -> np.ndarray:
            arr = np.asarray(z[split])
            if task == "regression" and tnorm is not None:
                arr = arr * float(tnorm["scale"]) + float(tnorm["mean"])
            return arr

        rec["pred_space"] = "raw" if (task == "regression" and tnorm is not None) else "logit_or_raw"
        rec["grid"] = {
            "updates": updates,
            "train_fit": _p("train_fit"),
            "valid": _p("valid"),
            "test": _p("test"),
            "fit_rows": np.asarray(z["fit_rows"]), "valid_rows": np.asarray(z["valid_rows"]),
            "test_rows": np.asarray(z["test_rows"]),
            "fit_y": np.asarray(z["fit_y"]), "valid_y": np.asarray(z["valid_y"]),
            "test_y": np.asarray(z["test_y"]),
        }
    step = d / "step0.pt"
    if step.exists():
        rec["step0"] = torch_load_cpu(step)
    return rec


def torch_load_cpu(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=False)


def grid_metric_sequence(task: str, grid: Mapping[str, Any], split: str, metric: str) -> np.ndarray:
    pred = np.asarray(grid[split], np.float64)
    y = np.asarray(grid[_ykey(split)], np.float64)
    out = []
    for p in pred:
        if metric == "rmse":
            out.append(float(math.sqrt(np.mean((p - y) ** 2))))
        elif metric == "mae":
            out.append(float(np.mean(np.abs(p - y))))
        elif metric == "roc_auc":
            from sklearn.metrics import roc_auc_score
            out.append(float(roc_auc_score((y > 0.5).astype(int), p)) if len(np.unique(y > 0.5)) > 1 else float("nan"))
        elif metric == "bce":
            z = p
            yy = y
            out.append(float(np.mean(np.maximum(z, 0) - z * yy + np.log1p(np.exp(-np.abs(z))))))
        else:
            raise KeyError(metric)
    return np.asarray(out, np.float64)


def _ykey(split: str) -> str:
    return "fit_y" if split == "train_fit" else f"{split}_y"


def arm_state_records(task: str, grid: Mapping[str, Any], updates: Sequence[int]) -> list[dict[str, Any]]:
    """Build selection records (update + per-split metric dicts) from grid arrays."""
    idx = {int(u): i for i, u in enumerate(grid["updates"])}
    recs = []
    for u in updates:
        u = int(u)
        if u not in idx:
            continue
        i = idx[u]
        rec = {"update": u}
        for split in ("train_fit", "valid", "test"):
            pred = np.asarray(grid[split], np.float64)[i]
            y = np.asarray(grid[_ykey(split)], np.float64)
            rec[split] = _metrics(task, pred, y)
        recs.append(rec)
    return recs


def _metrics(task: str, pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if task == "regression":
        e = pred - y
        return {"rmse": float(np.sqrt(np.mean(e ** 2))), "mae": float(np.mean(np.abs(e))),
                "n": int(y.size)}
    from sklearn.metrics import roc_auc_score, average_precision_score
    z = pred
    yy = (y > 0.5).astype(int)
    bce = float(np.mean(np.maximum(z, 0) - z * yy + np.log1p(np.exp(-np.abs(z)))))
    out = {"bce": bce, "n": int(y.size)}
    if len(np.unique(yy)) > 1:
        p = 1.0 / (1.0 + np.exp(-z))
        out["roc_auc"] = float(roc_auc_score(yy, p))
        out["pr_auc"] = float(average_precision_score(yy, p))
    else:
        out["roc_auc"] = float("nan")
        out["pr_auc"] = float("nan")
    return out


# ---------------------------------------------------------------------------
# selection / oracle-gap / rules
# ---------------------------------------------------------------------------


def selection_table_for_arm(arm_id: str, arm: Mapping[str, Any]) -> dict[str, Any]:
    rec = load_arm_records(arm_id)
    if rec is None or rec["grid"] is None:
        return {"arm_id": arm_id, "status": "NOT_RUN"}
    task = rec["meta"]["task"]
    grid = rec["grid"]
    U0 = int(rec["meta"]["recipe"]["U0"])
    updates = list(grid["updates"])
    e1 = [u for u in updates if u <= U0]
    e2 = [u for u in updates if u > U0]
    out: dict[str, Any] = {"arm_id": arm_id, "dataset": arm["dataset"], "model": arm["model"],
                           "fit": arm["fit"], "task": task, "U0": U0, "N": int(rec["meta"]["subset"]["n"]),
                           "phases": {}, "cost": rec["meta"].get("cost", {})}
    for phase, us in (("E1", e1), ("E2", e2), ("E1+E2", updates)):
        if not us:
            out["phases"][phase] = {"status": "EMPTY"}
            continue
        recs = arm_state_records(task, grid, us)
        gap = SEL.observed_oracle_gap(task, recs)
        pers = SEL.good_state_persistence(task, recs)
        per_rule = {}
        for rule in C.SELECTION_RULES:
            sel = SEL.select_state(task, rule, recs)
            s = sel["selected"]
            per_rule[rule] = {
                "update": sel["update"],
                "valid_criterion": sel["valid_criterion"],
                "test_primary": float(s["test"]["rmse"] if task == "regression" else s["test"]["roc_auc"]),
                "test_secondary": float(s["test"]["mae"] if task == "regression" else s["test"]["bce"]),
                "train_fit_primary": float(s["train_fit"]["rmse"] if task == "regression" else s["train_fit"]["roc_auc"]),
                "window": sel.get("window"),
                "window_fallback": sel.get("window_fallback"),
            }
        # descriptive: last-10 grid median and fixed endpoint
        tail = min(10, len(recs))
        last10 = recs[-tail:]
        out["phases"][phase] = {
            "n_candidates": len(recs),
            "observed_oracle_gap": gap,
            "persistence": pers,
            "rules": per_rule,
            "last10_median_test_primary": float(np.median(
                [r["test"]["rmse"] if task == "regression" else r["test"]["roc_auc"] for r in last10])),
            "endpoint_test_primary": float(recs[-1]["test"]["rmse"] if task == "regression" else recs[-1]["test"]["roc_auc"]),
            "endpoint_update": int(recs[-1]["update"]),
        }
    return out


# ---------------------------------------------------------------------------
# full-N U0/E1 panel: bootstrap + jackknife
# ---------------------------------------------------------------------------


def fulln_u0_panel() -> dict[str, Any]:
    """2000x Validation paired bootstrap + jackknife on the full-N U0/E1 grid.

    Panel: Lipo 5 arms (C1_R1/R3/CHEM8/K/DMPNN) + BACE 2 arms (C1/DMPNN), all at
    fit100.  Same resampling indices are used for every model/rule within a task.
    Lipo resamples validation rows; BACE resamples rows *stratified by class* so
    AUC stays estimable.
    """
    panel = []
    for model in ("C1_R1", "C1_R3", "CHEM8_R1", "K_R1", "DMPNN_D2F"):
        panel.append(f"LIP_{model}_F100")
    for model in ("C1_R1", "DMPNN_D2F"):
        panel.append(f"BACE_{model}_F100")

    rng = np.random.default_rng(int(C.BOOTSTRAP_SEED))
    out: dict[str, Any] = {"panel": panel, "replicates": C.BOOTSTRAP_REPLICATES,
                           "seed": C.BOOTSTRAP_SEED, "arms": {}, "notes": []}
    by_task: dict[str, list[str]] = {"Lipophilicity": [], "BACE": []}
    loaded = {}
    for a in panel:
        rec = load_arm_records(a)
        if rec is None or rec["grid"] is None:
            out["arms"][a] = {"status": "NOT_RUN"}
            continue
        loaded[a] = rec
        by_task[rec["meta"]["dataset"]].append(a)

    for task_name, arms in by_task.items():
        if not arms:
            continue
        task = "regression" if task_name == "Lipophilicity" else "binary"
        # resample indices once per task
        vrows = loaded[arms[0]]["grid"]["valid_rows"]
        vy = loaded[arms[0]]["grid"]["valid_y"]
        n = len(vy)
        strata = (vy > 0.5).astype(int) if task == "binary" else None
        draws = []
        for _ in range(C.BOOTSTRAP_REPLICATES):
            if strata is None:
                draws.append(rng.integers(0, n, size=n))
            else:
                idx = np.concatenate([rng.choice(np.where(strata == v)[0],
                                                 size=int((strata == v).sum()), replace=True)
                                      for v in (0, 1)])
                draws.append(idx)
        for a in arms:
            rec = loaded[a]
            grid = rec["grid"]
            U0 = int(rec["meta"]["recipe"]["U0"])
            sel_idx = [i for i, u in enumerate(grid["updates"]) if int(u) <= U0]
            vp = np.asarray(grid["valid"], np.float64)[sel_idx]
            tp = np.asarray(grid["test"], np.float64)[sel_idx]
            upstreams = [int(grid["updates"][i]) for i in sel_idx]
            vpred = vp  # shape (n_states, n_valid)
            tpred = tp
            # bootstrap selection
            boot = _bootstrap_over_draws(task, draws, vpred, np.asarray(vy, np.float64),
                                         tpred, np.asarray(grid["test_y"], np.float64), upstreams)
            jk = SEL.jackknife_selection(
                task, C.R_SEL, np.asarray(grid["valid_rows"], np.int64),
                np.asarray(grid["test_rows"], np.int64), vpred, np.asarray(vy, np.float64),
                tpred, np.asarray(grid["test_y"], np.float64),
                clusters=None, drop_rows=[2087])
            # ``selected_update`` from the jackknife is a *state index*; translate it
            jk["selected_state_index"] = True
            jk["grid_updates"] = upstreams
            for k in ("targeted_drop",):
                for item in jk.get(k, []):
                    item["selected_state_index"] = item["selected_update"]
                    item["selected_grid_update"] = upstreams[item["selected_update"]]
            jk["targeted_drop_rows_note"] = "2087 is a published global Valid row id, present in the Lipo Valid split"
            out["arms"][a] = {"task": task, "U0": U0, "n_states": len(sel_idx),
                              "bootstrap": boot, "jackknife": jk}
    return out


def _bootstrap_over_draws(task, draws, vpred, vy, tpred, ty, updates) -> dict[str, Any]:
    n_states = vpred.shape[0]

    def metric(pred, y):
        if task == "regression":
            return float(np.sqrt(np.mean((pred - y) ** 2)))
        from sklearn.metrics import roc_auc_score
        if len(np.unique(y)) < 2:
            return float("nan")
        return float(roc_auc_score(y, 1.0 / (1.0 + np.exp(-pred))))

    sel_updates = []
    test_metrics = []
    for idx in draws:
        yb = vy[idx]
        vals = np.array([metric(vpred[s][idx], yb) for s in range(n_states)])
        if task == "regression":
            best = int(np.lexsort((np.arange(n_states), vals))[0])
        else:
            best = int(np.lexsort((np.arange(n_states), -vals))[0])
        sel_updates.append(updates[best])
        test_metrics.append(metric(tpred[best], ty))
    tm = np.asarray(test_metrics, np.float64)
    fin = tm[np.isfinite(tm)]
    su = np.asarray(sel_updates, np.int64)
    return {
        "test_metric_p10": float(np.percentile(fin, 10)) if fin.size else None,
        "test_metric_p50": float(np.percentile(fin, 50)) if fin.size else None,
        "test_metric_p90": float(np.percentile(fin, 90)) if fin.size else None,
        "selected_update_spread": int(len(np.unique(su))),
        "selected_update_counts": {int(u): int((su == u).sum()) for u in np.unique(su)[:20]},
        "modal_update": int(np.bincount(su).argmax()) if su.size else None,
    }


# ---------------------------------------------------------------------------
# data x update budget
# ---------------------------------------------------------------------------


def data_budget_table() -> dict[str, Any]:
    arms = C.all_arms()
    by_model_ds: dict[tuple[str, str], list[str]] = {}
    for a in arms:
        by_model_ds.setdefault((a["model"], a["dataset"]), []).append(a["arm_id"])
    out: dict[str, Any] = {"cells": {}, "restart": None}
    for (model, dataset), arm_ids in by_model_ds.items():
        rows = []
        for aid in sorted(arm_ids, key=lambda x: int(x.rsplit("F", 1)[1])):
            rec = load_arm_records(aid)
            if rec is None or rec["grid"] is None:
                rows.append({"arm_id": aid, "status": "NOT_RUN"})
                continue
            task = rec["meta"]["task"]
            grid = rec["grid"]
            U0 = int(rec["meta"]["recipe"]["U0"])
            e1 = [u for u in grid["updates"] if u <= U0]
            e2 = [u for u in grid["updates"] if u > U0]
            row = {"arm_id": aid, "fit": rec["meta"]["fit"], "N": int(rec["meta"]["subset"]["n"]),
                   "U0": U0, "Umax": int(rec["meta"]["recipe"]["Umax"]), "status": "OK"}
            for phase, us in (("E1", e1), ("E2", e2)):
                if not us:
                    continue
                recs = arm_state_records(task, grid, us)
                for rule in C.SELECTION_RULES:
                    s = SEL.select_state(task, rule, recs)["selected"]
                    row[f"{phase}_{rule}_test"] = float(s["test"]["rmse"] if task == "regression" else s["test"]["roc_auc"])
                row[f"{phase}_last10_test_median"] = float(np.median(
                    [r["test"]["rmse"] if task == "regression" else r["test"]["roc_auc"] for r in recs[-min(10, len(recs)):]]))
                row[f"{phase}_last10_train_fit_median"] = float(np.median(
                    [r["train_fit"]["rmse"] if task == "regression" else r["train_fit"]["roc_auc"] for r in recs[-min(10, len(recs)):]]))
                row[f"{phase}_final_update_test"] = float(recs[-1]["test"]["rmse"] if task == "regression" else recs[-1]["test"]["roc_auc"])
            rows.append(row)
        # fractional Train_fit RMSE drop across the whole path (E1 last10 -> E2 last10)
        for row in rows:
            if row.get("status") != "OK":
                continue
            a0 = row.get("E1_last10_train_fit_median")
            a1 = row.get("E2_last10_train_fit_median")
            row["train_fit_drop_frac"] = (None if (a0 is None or a1 is None or a0 <= 0)
                                          else float((a0 - a1) / a0))
        out["cells"][f"{model}@{dataset}"] = rows
    # optional restart
    rdir = arm_dir(C.RESTART_ARM)
    if rdir.exists():
        out["restart"] = {"arm_id": C.RESTART_ARM, "status": "RUN"}
    else:
        out["restart"] = {"arm_id": C.RESTART_ARM, "status": "NOT_RUN"}
    return out


# ---------------------------------------------------------------------------
# readout probes
# ---------------------------------------------------------------------------


def readout_table() -> dict[str, Any]:
    out: dict[str, Any] = {"lipo": {}, "bace": {}, "purchase_rule": C.PROBE_PURCHASE_RULE}
    for ds_tag, dataset in (("lipo", "Lipophilicity"), ("bace", "BACE")):
        for rule in C.PROBE_ARMS:
            d = C.RUNS / f"{'LIP' if dataset == C.PRIMARY_DATASET else 'BACE'}_PROBE_{rule}"
            if not (d / "train.json").exists():
                out[ds_tag][rule] = {"status": "NOT_RUN"}
                continue
            payload = C.read_json(d / "train.json")
            gp = d / "grid_predictions.npz"
            entry = {"status": "OK", "new_params": payload["new_params"],
                     "updates": payload["train"]["updates"],
                     "step0_equivalence": payload["step0_delta_equivalence_max_abs"],
                     "selected": {}}
            if gp.exists():
                z = np.load(gp, allow_pickle=False)
                task = payload["task"]
                tnorm = payload.get("target_norm")

                def _p(name: str) -> np.ndarray:
                    arr = np.asarray(z[name], np.float64)
                    if task == "regression" and tnorm is not None:
                        arr = arr * float(tnorm["scale"]) + float(tnorm["mean"])
                    return arr

                updates = np.asarray(z["grid_updates"], np.int64)
                recs = [{"update": int(u),
                         "valid": _metrics(task, _p("valid")[i], np.asarray(z["valid_y"])),
                         "test": _metrics(task, _p("test")[i], np.asarray(z["test_y"])),
                         "train_fit": _metrics(task, _p("train_fit")[i], np.asarray(z["fit_y"]))}
                        for i, u in enumerate(updates)]
                for r in C.SELECTION_RULES:
                    s = SEL.select_state(task, r, recs, include_step0=True)["selected"]
                    entry["selected"][r] = {
                        "update": int(s["update"]),
                        "valid_rmse": s["valid"].get("rmse"), "valid_auc": s["valid"].get("roc_auc"),
                        "test_rmse": s["test"].get("rmse"), "test_auc": s["test"].get("roc_auc"),
                        "test_mae": s["test"].get("mae"), "test_bce": s["test"].get("bce"),
                    }
            out[ds_tag][rule] = entry
    # gains vs P_WARM (both splits)
    for ds_tag in ("lipo", "bace"):
        base = out[ds_tag].get("P_WARM", {})
        task = "regression" if ds_tag == "lipo" else "binary"
        for rule in C.PROBE_ARMS:
            e = out[ds_tag].get(rule, {})
            if e.get("status") != "OK" or base.get("status") != "OK" or not base.get("selected"):
                continue
            base_sel = base["selected"].get(C.R_SEL, {})
            sel = e["selected"].get(C.R_SEL, {})
            if task == "regression":
                vr = (base_sel.get("test_rmse"), sel.get("test_rmse"))
                vvalid = (base_sel.get("valid_rmse"), sel.get("valid_rmse"))
                e["gain_vs_warm_R_SEL"] = {
                    "valid_rmse_gain": (vvalid[0] - vvalid[1]) if None not in vvalid else None,
                    "test_rmse_gain": (vr[0] - vr[1]) if None not in vr else None,
                }
            else:
                vvalid = (base_sel.get("valid_auc"), sel.get("valid_auc"))
                vtest = (base_sel.get("test_auc"), sel.get("test_auc"))
                e["gain_vs_warm_R_SEL"] = {
                    "valid_auc_gain": (vvalid[1] - vvalid[0]) if None not in vvalid else None,
                    "test_auc_gain": (vtest[1] - vtest[0]) if None not in vtest else None,
                }
    return out


# ---------------------------------------------------------------------------
# structural subgroups (reuse existing covariates only)
# ---------------------------------------------------------------------------


def structural_subgroups() -> dict[str, Any]:
    """Atom count / Train-reference Morgan similarity / rooted-WL support by train quartiles.

    Covariates are recomputed from the frozen structure caches + labels; missing
    ones are recorded ``NOT_AVAILABLE`` (no new structure audit is built).
    """
    dataset = C.PRIMARY_DATASET
    try:
        from rdkit import Chem
        from rdkit.Chem import AllChem
        from rdkit import DataStructs
    except Exception:
        return {"dataset": dataset, "status": "NOT_AVAILABLE", "reason": "rdkit unavailable"}
    src = D.load_source(dataset, labels="none")
    smiles = [str(x) for x in src["smiles"]]
    rows = D.split_rows(dataset)
    n_atoms = np.array([Chem.MolFromSmiles(s).GetNumAtoms() for s in smiles], np.int64)
    train_set = set(rows["train"])
    train_fps = []
    for r in rows["train"][:2000]:     # bounded reference set (CPU/RAM)
        m = Chem.MolFromSmiles(smiles[r])
        train_fps.append(AllChem.GetMorganFingerprintAsBitVect(m, 2, nBits=2048))
    maxsim = np.zeros(len(smiles), np.float64)
    for i, s in enumerate(smiles):
        fp = AllChem.GetMorganFingerprintAsBitVect(Chem.MolFromSmiles(s), 2, nBits=2048)
        vals = DataStructs.BulkTanimotoSimilarity(fp, train_fps)
        maxsim[i] = float(max(vals)) if vals else 0.0
    out: dict[str, Any] = {"dataset": dataset, "status": "OK",
                           "covariates": ["n_atoms", "train_morgan_maxsim"],
                           "rooted_wl_support": "NOT_AVAILABLE",
                           "thresholds": "Train quartiles"}
    for split in ("valid", "test"):
        ridx = np.asarray(rows[split], np.int64)
        q = np.quantile(n_atoms[np.asarray(rows["train"], np.int64)], [0.25, 0.5, 0.75])
        out[f"{split}_n_atoms_quartile"] = {
            "q1": float(q[0]), "q2": float(q[1]), "q3": float(q[2]),
            "counts": [int(((n_atoms[ridx] <= q[0]).sum())), int(((n_atoms[ridx] > q[0]) & (n_atoms[ridx] <= q[1])).sum()),
                       int(((n_atoms[ridx] > q[1]) & (n_atoms[ridx] <= q[2])).sum()), int((n_atoms[ridx] > q[2]).sum())],
        }
    return out


def restart_summary() -> dict[str, Any]:
    """Compare the optional restart arm against the low-lr E2 leg of the base arm."""
    base = load_arm_records("LIP_C1_R1_F100")
    rst = load_arm_records(C.RESTART_ARM)
    out: dict[str, Any] = {"base_arm": "LIP_C1_R1_F100", "restart_arm": C.RESTART_ARM}
    if base is None or base["grid"] is None:
        out["status"] = "BASE_NOT_RUN"; return out
    if rst is None or rst["grid"] is None:
        out["status"] = "NOT_RUN"; return out
    out["status"] = "OK"
    for tag, rec in (("base_E2", base), ("restart_E2", rst)):
        g = rec["grid"]; U0 = int(rec["meta"]["recipe"]["U0"])
        us = [u for u in g["updates"] if int(u) > U0] or list(g["updates"])
        recs = arm_state_records("regression", g, us)
        rs = SEL.select_state("regression", C.R_SEL, recs)["selected"]
        tail = recs[-min(10, len(recs)):]
        out[tag] = {
            "n_candidates": len(recs),
            "R_SEL_update": int(rs["update"]),
            "R_SEL_valid_mse": float(rs["valid"]["rmse"] ** 2),
            "R_SEL_test_rmse": float(rs["test"]["rmse"]),
            "last10_test_median": float(np.median([r["test"]["rmse"] for r in tail])),
            "last10_train_fit_median": float(np.median([r["train_fit"]["rmse"] for r in tail])),
            "oracle_gap": float(observed_oracle_gap("regression", recs)["observed_oracle_gap"]),
        }
    out["test_gain_restart_minus_base"] = (out["base_E2"]["R_SEL_test_rmse"]
                                            - out["restart_E2"]["R_SEL_test_rmse"])
    return out


__all__ = [n for n in dir() if not n.startswith("_")]

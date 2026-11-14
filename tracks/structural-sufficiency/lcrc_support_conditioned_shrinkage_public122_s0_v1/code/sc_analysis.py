"""Analysis of the support-conditioned shrinkage panel (no new training).

Reads only this round's ``runs/*`` (train.json, grid_predictions.npz, curve.csv,
best_*.pt) plus the frozen support tables.  Produces:

``analysis/run_manifest.json``     one row per arm: selection, params, cost, hashes
``analysis/comparisons.json``      main table, four pre-registered comparisons,
                                   canonical-structure cluster bootstrap, scaffold
                                   sensitivity, auxiliary selection rules
``analysis/support_groups.json``   low/high support burden groups (Train-only OOF
                                   thresholds), unknown-key shares, self-count gap
``analysis/influence.json``        tail deletion (1 % / 5 %), top +/- dSSE rows
``analysis/intervention.json``     measured dose, a=1 counterfactual, alias pairs
``analysis/budget.json``           wall clock, GPU hours, job counts
``analysis/source_manifest.json``  read-only upstream sources + execution regime
``analysis/main_table.csv``        the headline table

All gains are reported with the sign "positive is better":
``RMSE_control - RMSE_candidate`` (regression) and ``AUC_candidate - AUC_control``.
"""

from __future__ import annotations

import csv
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import sc_common as C
import sc_support as S

import lcpr_train as PT


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def arm_dir(dataset: str, fit: str, arm: str) -> Path:
    tag = "LIP" if dataset == C.PRIMARY_DATASET else "BACE"
    return C.RUNS / f"{tag}_F{fit}_{arm}"


def load_run(dataset: str, fit: str, arm: str) -> dict[str, Any] | None:
    d = arm_dir(dataset, fit, arm)
    tj = d / "train.json"
    if not tj.exists():
        return None
    info = C.read_json(tj)
    task = info["task"]
    tnorm = info["data"]["target_norm"]
    z = np.load(d / "grid_predictions.npz", allow_pickle=False)
    grid = np.asarray(z["grid_updates"], np.int64)
    rule = info["selection"]["rules"][C.R_SEL]
    upd = rule["update"]
    out: dict[str, Any] = {"dir": d, "info": info, "task": task, "tnorm": tnorm,
                           "grid": grid, "selected_update": upd, "arm": arm, "fit": fit,
                           "dataset": dataset, "pred": {}, "y": {}, "rows": {}}
    for split, key in (("train_fit", "fit"), ("valid", "valid"), ("test", "test")):
        arr = np.asarray(z[split], dtype=np.float64)
        y = np.asarray(z[f"{key}_y"], dtype=np.float64)
        out["y"][split] = y
        out["rows"][split] = np.asarray(z[f"{key}_rows"], np.int64)
        if upd is None or len(grid) == 0:
            out["pred"][split] = None
            continue
        idx = int(np.where(grid == int(upd))[0][0])
        pred_std = arr[idx]
        if task == "regression":
            out["pred"][split] = pred_std * float(tnorm["scale"]) + float(tnorm["mean"])
        else:
            out["pred"][split] = pred_std
    return out


def metrics(task: str, pred: np.ndarray, y: np.ndarray) -> dict[str, float]:
    if task == "regression":
        return PT.regression_metrics(pred, y)
    return PT.binary_metrics(pred, y)


def primary(task: str, m: Mapping[str, float]) -> float:
    return float(m["rmse"] if task == "regression" else m["roc_auc"])


def _finite(x) -> float | None:
    return float(x) if x is not None and np.isfinite(x) else None


# ---------------------------------------------------------------------------
# main table
# ---------------------------------------------------------------------------


def main_table(panel: Mapping[str, Mapping[str, dict[str, Any]]]) -> dict[str, Any]:
    rows = []
    for fit, arms in panel.items():
        for arm, run in arms.items():
            if run is None:
                rows.append({"fit": fit, "arm": arm, "status": "NOT_RUN"})
                continue
            rec: dict[str, Any] = {
                "fit": fit, "arm": arm, "status": "COMPLETE",
                "selected_update": run["selected_update"],
                "parameters": run["info"]["model_info"]["parameters"],
                "wall_seconds": run["info"]["cost"]["wall_seconds"],
                "gpu_hours": run["info"]["cost"].get("gpu_hours_charged", 0.0),
                "support_key": run["info"].get("key") or "NONE(a=1 everywhere)",
            }
            for split in ("train_fit", "valid", "test"):
                if run["pred"][split] is None:
                    rec[f"{split}_status"] = "NO_SELECTED_STATE"
                    continue
                m = metrics(run["task"], run["pred"][split], run["y"][split])
                rec[f"{split}_rmse"] = _finite(m.get("rmse"))
                rec[f"{split}_mae"] = _finite(m.get("mae"))
                rec[f"{split}_roc_auc"] = _finite(m.get("roc_auc"))
                rec[f"{split}_pr_auc"] = _finite(m.get("pr_auc"))
                rec[f"{split}_bce"] = _finite(m.get("bce"))
            for rule in (C.A_SEL, C.W_SEL):
                r = run["info"]["selection"]["rules"].get(rule, {})
                rec[f"{rule.lower()}_update"] = r.get("update")
                rec[f"{rule.lower()}_criterion"] = r.get("criterion")
            rows.append(rec)
    return {"rows": rows}


def gains_table(panel: Mapping[str, Mapping[str, dict[str, Any]]],
                task: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    key = "rmse" if task == "regression" else "roc_auc"
    for fit, arms in panel.items():
        base = arms.get(C.ARM_BASE)
        if base is None or base["pred"]["valid"] is None:
            continue
        for cand_arm, ctrl_arm in list(C.SELECT_COMPARISONS) + [
            (a, C.ARM_BASE) for a in (C.ARM_UNIFORM, C.ARM_UNIFORM_OOF, C.ARM_REVERSE)
        ]:
            cand, ctrl = arms.get(cand_arm), arms.get(ctrl_arm)
            if cand is None or ctrl is None:
                continue
            if cand["pred"]["valid"] is None or ctrl["pred"]["valid"] is None:
                continue
            rec: dict[str, Any] = {"fit": fit, "candidate": cand_arm, "control": ctrl_arm,
                                   "pre_registered": (cand_arm, ctrl_arm) in C.SELECT_COMPARISONS}
            for split in ("train_fit", "valid", "test"):
                mc = metrics(task, cand["pred"][split], cand["y"][split])
                mk = metrics(task, ctrl["pred"][split], ctrl["y"][split])
                if task == "regression":
                    gain = mk["rmse"] - mc["rmse"]
                    rec[f"{split}_gain"] = float(gain)
                    rec[f"{split}_gain_rel"] = float(gain / mk["rmse"]) if mk["rmse"] else None
                    rec[f"{split}_mae_change"] = float(mc["mae"] - mk["mae"])
                    rec[f"{split}_mae_rel"] = float((mc["mae"] - mk["mae"]) / mk["mae"])
                else:
                    rec[f"{split}_gain"] = float(mc["roc_auc"] - mk["roc_auc"])
                    rec[f"{split}_gain_rel"] = None
                    rec[f"{split}_pr_auc_change"] = float(mc["pr_auc"] - mk["pr_auc"])
            out[f"F{fit}|{cand_arm}_vs_{ctrl_arm}"] = rec
    return out


# ---------------------------------------------------------------------------
# evaluation-row clusters (canonical structure, Murcko scaffold)
# ---------------------------------------------------------------------------


def row_clusters(dataset: str, split: str) -> dict[str, Any]:
    groups = S.load_groups(dataset)
    if split == "train_fit":
        raise ValueError("use row_clusters_for_rows for the fit split")
    rows = groups[split]["rows"]
    return {"rows": np.asarray(rows, np.int64),
            "group": np.asarray(groups[split]["group_id"], np.int64)}


def scaffold_ids(dataset: str) -> dict[str, Any]:
    """Murcko scaffold id per published row (local RDKit, cached in ``cache/``)."""
    path = C.CACHE / f"scaffolds_{dataset}.npz"
    if path.exists():
        z = np.load(path, allow_pickle=False)
        return {"scaffold": np.asarray(z["scaffold"], np.int64),
                "n_scaffolds": int(z["n_scaffolds"]), "sha256": C.sha256_file(path)}
    from rdkit import Chem
    from rdkit.Chem.Scaffolds import MurckoScaffold

    raw = np.load(C.REPO / "tracks/structural-sufficiency/lcrc_molnet_benchmark_pilot_v2/"
                  f"analysis/source/raw/lcrc_raw_v2/{dataset}.npz", allow_pickle=False)
    smiles = [str(x) for x in raw["canonical_smiles"]]
    vocab: dict[str, int] = {}
    ids = np.zeros(len(smiles), np.int64)
    for i, smi in enumerate(smiles):
        mol = Chem.MolFromSmiles(smi)
        scaf = MurckoScaffold.GetScaffoldForMol(mol)
        s = Chem.MolToSmiles(scaf) if scaf is not None else ""
        if not s:
            s = f"__acyclic__::{smi}"        # empty scaffold -> own canonical structure
        ids[i] = vocab.setdefault(s, len(vocab))
    C.CACHE.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, scaffold=ids, n_scaffolds=np.int64(len(vocab)))
    return {"scaffold": ids, "n_scaffolds": len(vocab), "sha256": C.sha256_file(path)}


def cluster_index(rows_eval: np.ndarray, rows_all: np.ndarray,
                  cluster_of_row: np.ndarray) -> np.ndarray:
    pos = {int(r): i for i, r in enumerate(rows_all.tolist())}
    return np.asarray([int(cluster_of_row[pos[int(r)]]) for r in rows_eval], np.int64)


# ---------------------------------------------------------------------------
# paired cluster bootstrap
# ---------------------------------------------------------------------------


def _paired_metric(pred: np.ndarray, y: np.ndarray, idx: np.ndarray, task: str) -> float:
    p, t = pred[idx], y[idx]
    if task == "regression":
        return float(np.sqrt(np.mean((p - t) ** 2)))
    from sklearn.metrics import roc_auc_score

    if len(np.unique(t)) < 2:
        return float("nan")
    return float(roc_auc_score(t, 1.0 / (1.0 + np.exp(-p))))


def cluster_bootstrap(pairs: Sequence[tuple[str, np.ndarray, np.ndarray]],
                      y: np.ndarray, clusters: np.ndarray, task: str, *,
                      replicates: int = C.BOOTSTRAP_REPLICATES,
                      seed: int = C.BOOTSTRAP_SEED) -> dict[str, Any]:
    """One shared resampling index per call (fixed weights, evaluation rows only)."""
    rng = np.random.default_rng(int(seed))
    uniq = np.unique(clusters)
    members = {int(c): np.where(clusters == c)[0] for c in uniq}
    draws: list[np.ndarray] = []
    for _ in range(int(replicates)):
        pick = rng.choice(uniq, size=len(uniq), replace=True)
        draws.append(np.concatenate([members[int(c)] for c in pick]))
    out: dict[str, Any] = {"n_clusters": int(len(uniq)),
                           "max_cluster_share": float(max(len(v) for v in members.values())
                                                      / len(clusters)),
                           "replicates": int(replicates)}
    for tag, cand, ctrl in pairs:
        point_c = _paired_metric(cand, y, np.arange(len(y)), task)
        point_k = _paired_metric(ctrl, y, np.arange(len(y)), task)
        vals = []
        for idx in draws:
            a = _paired_metric(cand, y, idx, task)
            b = _paired_metric(ctrl, y, idx, task)
            vals.append(b - a if task == "regression" else a - b)
        arr = np.asarray(vals, np.float64)
        arr = arr[np.isfinite(arr)]
        point = point_k - point_c if task == "regression" else point_c - point_k
        out[tag] = {
            "point_gain": float(point),
            "mean": float(arr.mean()) if arr.size else None,
            "p2.5": float(np.percentile(arr, 2.5)) if arr.size else None,
            "p97.5": float(np.percentile(arr, 97.5)) if arr.size else None,
            "frac_positive": float((arr > 0).mean()) if arr.size else None,
            "crosses_zero": bool(arr.size and np.percentile(arr, 2.5) <= 0 <= np.percentile(arr, 97.5)),
        }
    return out


# ---------------------------------------------------------------------------
# support burden groups (description, not a new selection rule)
# ---------------------------------------------------------------------------


def support_burden(dataset: str, fit: str, key: str, split: str) -> dict[str, Any]:
    """Per-molecule ``B_G = sum_v w_v (1 - a_SELECT(v))`` for one split."""
    sup = S.load_support(dataset, fit, key)
    cache = C.LD.load_cache(dataset, split)
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    alpha = np.asarray(sup[f"alpha_{C.ARM_SELECT}_{split}"], dtype=np.float64)
    r = np.asarray(sup[f"r_{split}"], dtype=np.float64)
    act = np.asarray(sup[f"active_{split}"])
    n = int(cache["n_molecules"])
    burden = np.zeros(n, dtype=np.float64)
    has_active = np.zeros(n, dtype=bool)
    for m in range(n):
        a0, a1 = int(atom_ptr[m]), int(atom_ptr[m + 1])
        act_m = act[a0:a1]
        if not np.any(act_m):
            continue
        has_active[m] = True
        rm = r[a0:a1][act_m]
        tot = rm.sum()
        if tot <= 0:
            continue
        w = rm / tot
        burden[m] = float(np.sum(w * (1.0 - alpha[a0:a1][act_m])))
    rows = [int(x) for x in np.asarray(cache["rows"]).tolist()]
    return {"burden": burden, "has_active": has_active, "rows": np.asarray(rows, np.int64)}


def support_groups(panel: Mapping[str, Mapping[str, dict[str, Any]]], dataset: str,
                   key: str, fit: str) -> dict[str, Any]:
    """Low/high burden groups with thresholds frozen on **Train-only OOF** burden."""
    oof = S.support_alpha_tables(dataset, fit, key, S.load_or_build_rootkeys(dataset),
                                 S.load_groups(dataset), oof=True)
    # OOF burden on the fit records (per-molecule, Train structures only)
    cache = C.LD.load_cache(dataset, "train")
    atom_ptr = np.asarray(cache["atom_ptr"], dtype=np.int64)
    positions = C.fit_positions(dataset, fit)
    w = np.asarray(oof["w"], dtype=np.float64)
    a_oof_root = np.asarray(oof["a"], dtype=np.float64)
    act = np.asarray(oof["active"], dtype=bool)
    ptr = np.concatenate([[0], np.cumsum([int(atom_ptr[p + 1] - atom_ptr[p])
                                          for p in positions])]).astype(np.int64)
    fit_burden = np.zeros(len(positions), dtype=np.float64)
    for i in range(len(positions)):
        sl = slice(ptr[i], ptr[i + 1])
        m = act[sl] & (w[sl] > 0)
        if np.any(m):
            fit_burden[i] = float(np.sum(w[sl][m] * (1.0 - a_oof_root[sl][m])))
    q25 = float(np.percentile(fit_burden[fit_burden > 0], 25)) if np.any(fit_burden > 0) else 0.0
    q75 = float(np.percentile(fit_burden[fit_burden > 0], 75)) if np.any(fit_burden > 0) else 0.0

    out: dict[str, Any] = {
        "dataset": dataset, "fit": fit, "key": key,
        "thresholds": {"q25": q25, "q75": q75,
                       "source": "Train-only OOF support burden of the fit structures"},
        "groups": {},
    }
    base = panel.get(fit, {}).get(C.ARM_BASE)
    if base is None:
        return out
    for split in ("valid", "test"):
        bur = support_burden(dataset, fit, key, split)
        rows_eval = base["rows"][split]
        pos = {int(r): i for i, r in enumerate(bur["rows"].tolist())}
        b = np.asarray([bur["burden"][pos[int(r)]] for r in rows_eval])
        act_row = np.asarray([bur["has_active"][pos[int(r)]] for r in rows_eval])
        low = (b <= q25)
        high = (b >= q75)
        none_active = ~act_row
        rec: dict[str, Any] = {"n_rows": int(len(b)), "n_low": int(low.sum()),
                               "n_high": int(high.sum()),
                               "n_no_active_root": int(none_active.sum()),
                               "frac_no_active_root": float(none_active.mean()),
                               "n_distinct_structures_low": int(len(set(rows_eval[low].tolist()))),
                               "n_distinct_structures_high": int(len(set(rows_eval[high].tolist())))}
        y = base["y"][split]
        for grp_name, mask in (("low", low), ("high", high)):
            rec[grp_name] = {}
            for arm in (C.ARM_BASE, C.ARM_UNIFORM, C.ARM_SELECT):
                run = panel.get(fit, {}).get(arm)
                if run is None or run["pred"][split] is None:
                    continue
                pred = run["pred"][split]
                m = metrics(run["task"], pred[mask], y[mask])
                rec[grp_name][arm] = {"rmse": m.get("rmse"), "mae": m.get("mae"),
                                      "n": int(mask.sum())}
            for ctrl_arm in (C.ARM_BASE, C.ARM_UNIFORM):
                run_s, run_c = panel[fit].get(C.ARM_SELECT), panel[fit].get(ctrl_arm)
                if run_s is None or run_c is None or run_s["pred"][split] is None:
                    continue
                d = (run_s["pred"][split][mask] - y[mask]) ** 2 - \
                    (run_c["pred"][split][mask] - y[mask]) ** 2
                rec[grp_name][f"mean_dsse_SELECT_vs_{ctrl_arm}"] = float(d.mean())
        out["groups"][split] = rec
    return out


# ---------------------------------------------------------------------------
# influence / tail
# ---------------------------------------------------------------------------


def influence(panel: Mapping[str, Mapping[str, dict[str, Any]]], fit: str) -> dict[str, Any]:
    arms = panel.get(fit, {})
    base = arms.get(C.ARM_BASE)
    if base is None:
        return {}
    out: dict[str, Any] = {"fit": fit}
    for split in ("valid", "test"):
        pred_b = base["pred"][split]
        y = base["y"][split]
        rows = base["rows"][split]
        ae = np.abs(pred_b - y)
        order = np.lexsort((rows, -ae))            # abs error desc, ties by public row id
        out[split] = {"headline": {}, "drop": {}}
        for arm, run in arms.items():
            if run is None or run["pred"][split] is None:
                continue
            m = metrics(run["task"], run["pred"][split], y)
            out[split]["headline"][arm] = {"rmse": m.get("rmse"), "mae": m.get("mae"),
                                           "roc_auc": m.get("roc_auc")}
        for frac in (0.01, 0.05):
            k = max(1, int(round(frac * len(y))))
            keep = np.zeros(len(y), bool)
            keep[order[k:]] = True
            rec = {"k": int(k), "n_kept": int(keep.sum())}
            for arm, run in arms.items():
                if run is None or run["pred"][split] is None:
                    continue
                m = metrics(run["task"], run["pred"][split][keep], y[keep])
                rec[arm] = {"rmse": m.get("rmse"), "mae": m.get("mae"),
                            "roc_auc": m.get("roc_auc")}
            out[split]["drop"][f"{frac:.2f}"] = rec
        sel = arms.get(C.ARM_SELECT)
        if sel is not None and sel["pred"][split] is not None:
            d = (sel["pred"][split] - y) ** 2 - (pred_b - y) ** 2
            ord_d = np.argsort(d)
            out[split]["top_improved"] = [
                {"row": int(rows[i]), "dsse": float(d[i]),
                 "base_abs_error": float(ae[i]), "select_abs_error": float(abs(sel["pred"][split][i] - y[i]))}
                for i in ord_d[:3]]
            out[split]["top_harmed"] = [
                {"row": int(rows[i]), "dsse": float(d[i]),
                 "base_abs_error": float(ae[i]), "select_abs_error": float(abs(sel["pred"][split][i] - y[i]))}
                for i in ord_d[-3:][::-1]]
            out[split]["frac_rows_improved"] = float((d < 0).mean())
            out[split]["sse_share_improved"] = float(
                -d[d < 0].sum() / max(float(((pred_b - y) ** 2).sum()), 1e-12))
    return out


# ---------------------------------------------------------------------------
# intervention / dose / alias pairs
# ---------------------------------------------------------------------------


def intervention(panel: Mapping[str, Mapping[str, dict[str, Any]]], dataset: str,
                 fit: str) -> dict[str, Any]:
    """Measured dose + the ``a=1`` counterfactual + the five existing alias pairs."""
    import torch
    import sc_model as M
    import sc_train as T

    sel = panel.get(fit, {}).get(C.ARM_SELECT)
    if sel is None:
        return {}
    spec = C.LD.frozen_spec()
    snorm, _ = C.shared_norms(dataset)
    support = S.load_support(dataset, fit, sel["info"]["key"])
    d = sel["dir"]
    ckpt = torch.load(d / "best_R_SEL.pt", map_location="cpu", weights_only=False)
    model = M.build_arm_model(spec, snorm, C.ARM_SELECT)
    model.load_state_dict({k: v for k, v in ckpt["state_dict"].items()}, strict=False)
    model.eval()
    device = torch.device("cpu")
    caches = {s: C.LD.load_cache(dataset, s) for s in C.SPLITS}
    runner = T.GateRunner(model, device, caches,
                          {s: (np.asarray(support[f"alpha_{C.ARM_SELECT}_{s}"])
                               if s in ("valid", "test") else None) for s in C.SPLITS})
    out: dict[str, Any] = {"fit": fit, "key": sel["info"]["key"],
                           "selected_update": int(ckpt["update"]),
                           "dose": sel["info"].get("dose"),
                           "counterfactual": {}}
    for split in ("valid", "test"):
        n = int(caches[split]["n_molecules"])
        atom_ptr = np.asarray(caches[split]["atom_ptr"], dtype=np.int64)
        alpha = np.asarray(support[f"alpha_{C.ARM_SELECT}_{split}"], dtype=np.float32)
        with torch.no_grad():
            p_gate = runner.predict(split)
            # a = 1 everywhere: the same weights, the gate switched off
            runner.alpha[split] = np.ones_like(alpha)
            p_one = runner.predict(split)
            runner.alpha[split] = alpha
        delta = p_one - p_gate
        if sel["task"] == "regression":
            scale = float(sel["tnorm"]["scale"])
            delta_raw = delta * scale
        else:
            delta_raw = delta
        out["counterfactual"][split] = {
            "n_molecules": int(n),
            "frac_molecules_changed": float((np.abs(delta) > 1e-7).mean()),
            "max_abs_delta_raw": float(np.abs(delta_raw).max()),
            "p50_abs_delta_raw": float(np.median(np.abs(delta_raw))),
            "p99_abs_delta_raw": float(np.percentile(np.abs(delta_raw), 99)),
        }
    # --- the five existing input-alias pairs (frozen upstream artefact) -------
    alias_path = (C.REPO / "tracks/structural-sufficiency/"
                  "lcrc_lipo_s3_locality_public122_s0_v1/analysis/alias_pairs.csv")
    pairs: list[dict[str, Any]] = []
    if alias_path.exists():
        src = C.LD.load_source(dataset, labels="none")
        row_pos = {}
        for split in C.SPLITS:
            for i, r in enumerate(src["rows"][split]):
                row_pos[int(r)] = (split, i)
        alpha_all = {}
        for split in C.SPLITS:
            alpha_all[split] = np.asarray(support[f"alpha_{C.ARM_SELECT}_{split}"], dtype=np.float64)
        with open(alias_path, encoding="utf-8") as fh:
            for rec in csv.DictReader(fh):
                a_row, b_row = int(rec["row_a"]), int(rec["row_b"])
                if any(int(r) not in row_pos for r in (a_row, b_row)):
                    continue
                pa, pb, da, db = [], [], 0.0, 0.0
                for row, store, astore in ((a_row, pa, "a"), (b_row, pb, "b")):
                    split, i = row_pos[row]
                    n = int(caches[split]["n_molecules"])
                    runner.alpha[split] = np.asarray(alpha_all[split], dtype=np.float32)
                    with torch.no_grad():
                        pred = runner.predict(split, [i])[0]
                    store.append(pred)
                    atom_ptr = np.asarray(caches[split]["atom_ptr"], dtype=np.int64)
                    astore_alpha = alpha_all[split][atom_ptr[i]: atom_ptr[i + 1]]
                    if astore == "a":
                        da = float(astore_alpha.mean())
                    else:
                        db = float(astore_alpha.mean())
                pairs.append({"row_a": a_row, "row_b": b_row,
                              "abs_pred_diff": float(abs(pa[0] - pb[0])),
                              "a_mean_a": da, "a_mean_b": db,
                              "abs_a_diff": float(abs(da - db)),
                              "pass_2e-5": bool(abs(pa[0] - pb[0]) <= 2e-5)})
        pass
    out["alias_pairs"] = {"n_pairs": len(pairs), "pairs": pairs,
                          "all_within_2e-5": bool(pairs and all(p["pass_2e-5"] for p in pairs))}
    return out


# ---------------------------------------------------------------------------
# entry
# ---------------------------------------------------------------------------


def build_panel(dataset: str, fits: Sequence[str]) -> dict[str, dict[str, Any]]:
    panel: dict[str, dict[str, Any]] = {}
    for fit in fits:
        panel[fit] = {arm: load_run(dataset, fit, arm) for arm in C.ARMS}
    return panel


def cluster_bootstrap_all(panel: Mapping[str, Mapping[str, dict[str, Any]]], dataset: str,
                         task: str) -> dict[str, Any]:
    out: dict[str, Any] = {}
    scaf = scaffold_ids(dataset)
    for fit, arms in panel.items():
        base = arms.get(C.ARM_BASE)
        if base is None or base["pred"]["valid"] is None:
            continue
        for split in ("valid", "test"):
            rows_eval = base["rows"][split]
            cl = row_clusters(dataset, split)
            pos = {int(r): i for i, r in enumerate(cl["rows"].tolist())}
            gidx = np.asarray([pos[int(r)] for r in rows_eval], np.int64)
            groups_cluster = cl["group"][gidx]
            cl_scaf = np.asarray([scaf["scaffold"][int(r)] for r in rows_eval], np.int64)
            pairs = []
            for cand_arm, ctrl_arm in C.SELECT_COMPARISONS:
                cand, ctrl = arms.get(cand_arm), arms.get(ctrl_arm)
                if cand is None or ctrl is None:
                    continue
                if cand["pred"][split] is None or ctrl["pred"][split] is None:
                    continue
                pairs.append((f"{cand_arm}_vs_{ctrl_arm}", cand["pred"][split],
                              ctrl["pred"][split]))
            if not pairs:
                continue
            y = base["y"][split]
            out[f"F{fit}|{split}"] = {
                "structure_cluster": cluster_bootstrap(pairs, y, groups_cluster, task),
                "scaffold_cluster": cluster_bootstrap(pairs, y, cl_scaf, task),
                "n_distinct_structures": int(len(np.unique(groups_cluster))),
                "n_scaffolds": int(len(np.unique(cl_scaf))),
            }
            out[f"F{fit}|{split}"]["scaffold_cluster"]["n_scaffolds_total_dataset"] = scaf["n_scaffolds"]
            out[f"F{fit}|{split}"]["scaffold_cluster"]["_rows_sha256"] = C.row_sha256(rows_eval)
    return out


def low_label_contrast(panel: Mapping[str, Mapping[str, dict[str, Any]]], task: str) -> dict[str, Any]:
    if not {"100", "25"} <= set(panel):
        return {}
    out: dict[str, Any] = {}
    for split in ("valid", "test"):
        base100 = panel["100"].get(C.ARM_BASE)
        if base100 is None or base100["pred"][split] is None:
            continue
        rows = base100["rows"][split]
        y = base100["y"][split]
        rec: dict[str, Any] = {}
        for ctrl_arm in (C.ARM_BASE, C.ARM_UNIFORM, C.ARM_UNIFORM_OOF):
            gains = {}
            ok = True
            for fit in ("100", "25"):
                s, c = panel[fit].get(C.ARM_SELECT), panel[fit].get(ctrl_arm)
                if s is None or c is None or s["pred"][split] is None:
                    ok = False
                    break
                ms = metrics(task, s["pred"][split], y)
                mc = metrics(task, c["pred"][split], y)
                gains[fit] = (mc["rmse"] - ms["rmse"]) if task == "regression" else \
                    (ms["roc_auc"] - mc["roc_auc"])
            if not ok:
                continue
            rng = np.random.default_rng(C.BOOTSTRAP_SEED + 7)
            n = len(y)
            vals = []
            preds = {}
            for fit in ("100", "25"):
                preds[fit] = (panel[fit][C.ARM_SELECT]["pred"][split],
                              panel[fit][ctrl_arm]["pred"][split])
            for _ in range(C.BOOTSTRAP_REPLICATES):
                idx = rng.integers(0, n, size=n)
                g = {}
                for fit in ("100", "25"):
                    ps, pc = preds[fit]
                    ms = _paired_metric(ps, y, idx, task)
                    mc = _paired_metric(pc, y, idx, task)
                    g[fit] = (mc - ms) if task == "regression" else (ms - mc)
                vals.append(g["25"] - g["100"])
            arr = np.asarray(vals, np.float64)
            arr = arr[np.isfinite(arr)]
            rec[f"SELECT_vs_{ctrl_arm}"] = {
                "gain_fit100": float(gains["100"]), "gain_fit25": float(gains["25"]),
                "diff_25_minus_100": float(gains["25"] - gains["100"]),
                "p2.5": float(np.percentile(arr, 2.5)) if arr.size else None,
                "p97.5": float(np.percentile(arr, 97.5)) if arr.size else None,
                "crosses_zero": bool(arr.size and np.percentile(arr, 2.5) <= 0 <= np.percentile(arr, 97.5)),
            }
        out[split] = rec
    out["note"] = ("two fit sizes of one seed; not a sample-complexity or noise-floor estimate")
    return out


def selection_sensitivity(panel: Mapping[str, Mapping[str, dict[str, Any]]], task: str) -> dict[str, Any]:
    """A_SEL / W_SEL direction and the last-10-grid median (secondary, never primary)."""
    out: dict[str, Any] = {}
    for fit, arms in panel.items():
        for arm, run in arms.items():
            if run is None:
                continue
            d = run["dir"]
            z = np.load(d / "grid_predictions.npz", allow_pickle=False)
            grid = np.asarray(z["grid_updates"], np.int64)
            if not len(grid):
                continue
            rec: dict[str, Any] = {}
            for split in ("valid", "test"):
                arr = np.asarray(z[split], np.float64)
                y = np.asarray(z[f"{'fit' if split == 'train_fit' else split}_y"], np.float64)
                tnorm = run["tnorm"]
                vals = []
                for i in range(arr.shape[0]):
                    p = arr[i]
                    if task == "regression":
                        p = p * float(tnorm["scale"]) + float(tnorm["mean"])
                        vals.append(float(np.sqrt(np.mean((p - y) ** 2))))
                    else:
                        from sklearn.metrics import roc_auc_score
                        vals.append(float(roc_auc_score(y, 1.0 / (1.0 + np.exp(-p)))))
                vals = np.asarray(vals)
                rec[split] = {"last10_median": float(np.median(vals[-10:])),
                              "best_grid": float(np.min(vals) if task == "regression"
                                                 else np.max(vals))}
            out[f"F{fit}|{arm}"] = rec
    return out


def write_main_csv(table: Mapping[str, Any], path: Path) -> None:
    rows = table["rows"]
    fields: list[str] = []
    for r in rows:
        for k in r:
            if k not in fields:
                fields.append(k)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def source_manifest() -> dict[str, Any]:
    src = {
        "round_id": C.ROUND_ID, "created_utc": C.utcnow(),
        "read_only_upstream": {},
        "materialised_here": {},
        "execution_regime": {},
    }
    for base, tag in ((C.PERF, "lcrc_performance_route_8h_public122_s0_v1"),
                      (C.DIAG, "lcrc_bottleneck_diagnosis_public122_s0_v1"),
                      (C.REPO / "tracks/structural-sufficiency/lcrc_lipo_model_dissection_v1",
                       "lcrc_lipo_model_dissection_v1")):
        files = {}
        for p in sorted((base / "code").glob("*.py")):
            files[C.track_rel(p)] = C.sha256_file(p)
        for extra in ("PROTOCOL.md", "configs/diagnosis.yaml"):
            q = base / extra
            if q.exists():
                files[C.track_rel(q)] = C.sha256_file(q)
        src["read_only_upstream"][tag] = files
    for d in ("Lipophilicity", "BACE"):
        src["read_only_upstream"].setdefault("frozen_structure_caches", {})[
            f"molnet_pilot/cache/{d}_{{split}}.pt"] = "gitignored structure caches, hash recorded per run"
    src["materialised_here"] = {
        "rootkeys": {d: C.sha256_file(C.rootkeys_path(d)) for d in C.DATASETS
                     if C.rootkeys_path(d).exists()},
        "groups": {d: C.sha256_file(C.groups_path(d)) for d in C.DATASETS
                   if C.groups_path(d).exists()},
        "support": {f"{d}|F{f}|{k}": C.sha256_file(C.support_path(d, f, k))
                    for d in C.DATASETS for f in (("100", "25") if d == C.PRIMARY_DATASET else ("100",))
                    for k in C.KEY_PRIORITY if C.support_path(d, f, k).exists()},
        "norms": {"structure": {d: C.sha256_file(C.STRUCT_NORM_REF[d]) for d in C.DATASETS},
                  "target": {d: C.sha256_file(p) for d, p in C.TARGET_NORM_REF.items()}},
    }
    return src


def main(argv: Sequence[str] | None = None) -> int:
    fits = [f for f in (argv or ["100", "25"]) if f]
    dataset = C.PRIMARY_DATASET
    task = C.dataset_meta(dataset)["task"]
    panel = build_panel(dataset, fits)
    missing = {fit: [a for a, r in arms.items() if r is None] for fit, arms in panel.items()}
    table = main_table(panel)
    table["missing"] = missing
    write_main_csv(table, C.ANALYSIS / "main_table.csv")

    manifest = {"round_id": C.ROUND_ID, "dataset": dataset, "fits": fits, "arms": {}}
    for fit, arms in panel.items():
        for arm, run in arms.items():
            if run is None:
                manifest["arms"][f"F{fit}|{arm}"] = {"status": "NOT_RUN"}
                continue
            info = run["info"]
            manifest["arms"][f"F{fit}|{arm}"] = {
                "status": info["status"], "arm": arm, "fit": fit,
                "out_dir": C.track_rel(run["dir"]),
                "selected_update": run["selected_update"],
                "selection_rules": {r: v for r, v in info["selection"]["rules"].items()},
                "retained": info["selection"]["retained"],
                "parameters": info["model_info"]["parameters"],
                "init_state_hash": info["model_info"]["init_state_hash"],
                "cost": info["cost"], "environment": info["environment"],
                "provenance": {"git_commit": info["provenance"]["git_commit"]},
                "support_sha256": (info.get("support") or {}).get("path"),
                "recipe": {k: v for k, v in info["recipe"].items()
                           if k in ("name", "peak_lr", "min_lr", "weight_decay", "batch_size",
                                    "dropout", "U0", "Umax", "steps_per_epoch")},
                "data": {"n_fit": info["data"]["n_fit"],
                         "subset_positions_sha256": info["subset"]["positions_sha256"],
                         "valid_rows_sha256": info["data"]["valid_rows_sha256"],
                         "test_rows_sha256": info["data"]["test_rows_sha256"]},
            }
    for key, rec in manifest["arms"].items():
        if rec.get("status") != "completed" and rec.get("status") != "COMPLETE":
            continue
        d = C.REPO / rec["out_dir"]
        files = {}
        for name in ("train.json", "curve.csv", "grid_predictions.npz",
                     "best_R_SEL.pt", "best_A_SEL.pt", "best_W_SEL.pt", "last.pt"):
            q = d / name
            if q.exists():
                files[name] = {"bytes": int(q.stat().st_size), "sha256": C.sha256_file(q)}
        rec["artefacts"] = files
        rec["artefacts_copy"] = ("local only: the `res` deployment was blocked by a full "
                                 "shared /home (infrastructure failure, see REPORT §7)")
    C.write_json(C.ANALYSIS / "run_manifest.json", manifest)

    comparisons = {
        "round_id": C.ROUND_ID, "dataset": dataset,
        "sign_convention": "positive = candidate better",
        "main_table": table,
        "gains": gains_table(panel, task),
        "bootstrap": cluster_bootstrap_all(panel, dataset, task),
        "low_label": low_label_contrast(panel, task),
        "selection_sensitivity": selection_sensitivity(panel, task),
    }
    C.write_json(C.ANALYSIS / "comparisons.json", comparisons)

    key = C.read_json(C.ANALYSIS / "key_selection.json")["chosen_key"]
    groups: dict[str, Any] = {"round_id": C.ROUND_ID, "key": key, "by_fit": {}}
    for fit in fits:
        if panel[fit].get(C.ARM_BASE) is None:
            continue
        groups["by_fit"][fit] = support_groups(panel, dataset, key, fit)
        rk = S.load_or_build_rootkeys(dataset)
        sup = S.load_support(dataset, fit, key)
        groups["by_fit"][fit]["unknown_key_shares"] = {
            "definition": "share of roots whose key is absent from the frozen fit table",
            "by_split": {s: float((np.asarray(sup[f"c_{s}"]) == 0).mean()) for s in C.SPLITS},
            "active_roots_only": {
                s: float((np.asarray(sup[f"c_{s}"])[np.asarray(sup[f"active_{s}"])] == 0).mean())
                for s in C.SPLITS},
        }
        groups["by_fit"][fit]["self_count"] = {
            "note": ("Train queries use the same full fit table, so a fit structure counts "
                     "itself; the OOF table (4/5 structures) is used only for gates, the "
                     "a_oof constant and the burden thresholds"),
            "fit_table_weighted_mean_a": float(sup["fit_mean_a_select"]),
            "oof_table_weighted_mean_a": float(sup["a_oof"]),
            "self_count_dose_gap": float(sup["fit_mean_a_select"] - sup["a_oof"]),
        }
    C.write_json(C.ANALYSIS / "support_groups.json", groups)

    infl = {"round_id": C.ROUND_ID, "by_fit": {fit: influence(panel, fit) for fit in fits}}
    C.write_json(C.ANALYSIS / "influence.json", infl)

    inter = {"round_id": C.ROUND_ID, "dataset": dataset, "by_fit": {}}
    for fit in fits:
        if panel[fit].get(C.ARM_SELECT) is not None:
            inter["by_fit"][fit] = intervention(panel, dataset, fit)
    C.write_json(C.ANALYSIS / "intervention.json", inter)

    C.write_json(C.ANALYSIS / "source_manifest.json", source_manifest())

    budget = {"round_id": C.ROUND_ID, "created_utc": C.utcnow(),
              "runs": {}, "totals": {}}
    tot_wall = tot_gpu = 0.0
    for fit, arms in panel.items():
        for arm, run in arms.items():
            if run is None:
                budget["runs"][f"F{fit}|{arm}"] = {"status": "NOT_RUN"}
                continue
            c = run["info"]["cost"]
            budget["runs"][f"F{fit}|{arm}"] = {
                "status": "COMPLETE", "wall_seconds": c["wall_seconds"],
                "process_wall_seconds": c["process_wall_seconds"],
                "train_seconds": c["train_seconds"], "eval_seconds": c["eval_seconds"],
                "gpu_hours": c.get("gpu_hours_charged", 0.0),
                "updates": c["updates_run"],
                "device": run["info"]["environment"]["device"],
            }
            tot_wall += c["process_wall_seconds"]
            tot_gpu += float(c.get("gpu_hours_charged", 0.0))
    budget["totals"] = {"sum_process_wall_seconds": tot_wall,
                        "sum_gpu_hours": tot_gpu,
                        "n_completed": int(sum(1 for v in budget["runs"].values()
                                               if v.get("status") == "COMPLETE")),
                        "caps": {"wall_target_hours": C.WALL_TARGET_HOURS,
                                 "wall_hard_hours": C.WALL_HARD_HOURS,
                                 "gpu_soft_hours": C.GPU_SOFT_CAP_HOURS,
                                 "gpu_hard_hours": C.GPU_HARD_CAP_HOURS}}
    C.write_json(C.ANALYSIS / "budget.json", budget)
    C.log(f"[analyze] completed={budget['totals']['n_completed']} "
          f"wall={tot_wall / 3600:.2f}h gpu={tot_gpu:.3f}A100h")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

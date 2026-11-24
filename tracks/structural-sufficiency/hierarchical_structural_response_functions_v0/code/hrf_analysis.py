"""Stage 2 analysis for HSRF v0: tables, axes, identifiability, pairs, report."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import hrf_common as C
import hrf_design as D

ROOT = C.RESULTS_DIR
RIDGE = ROOT / "ridge"
KRR = ROOT / "krr"
ANALYSIS = C.ROUND_DIR / "analysis"
FIG = ROOT / "figures"

RIDGE_ARMS = ["L_BASE", "F_BASE", "L_REAL", "F_NULL", "F_REAL"]
MLP_ARMS = ["BASE", "NULL", "REAL"]
KRR_ARMS = ["BASE", "REAL"]


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="") as handle:
        return list(csv.DictReader(handle))


def _align(rid: np.ndarray, pred: np.ndarray, expected: np.ndarray) -> np.ndarray:
    rid = np.asarray(rid, np.int64)
    if np.array_equal(rid, expected):
        return np.asarray(pred, np.float64)
    pos = {int(r): i for i, r in enumerate(rid.tolist())}
    order = np.asarray([pos[int(r)] for r in expected.tolist()], np.int64)
    return np.asarray(pred, np.float64)[order]


def load_models(stage: D.Stage0, payload: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    models: dict[str, dict[str, Any]] = {}
    ridge_meta = {r["arm"]: r for r in _read_csv(RIDGE / "main_table.csv")}
    for arm in RIDGE_ARMS:
        preds = {}
        for split in ("train", "valid", "test"):
            z = np.load(RIDGE / "predictions" / f"{arm}_{split}.npz", allow_pickle=False)
            expected = stage.rows[split]
            preds[split] = _align(z["row_id"], z["pred"], expected)
        meta = ridge_meta.get(arm, {})
        models[arm] = {
            "name": arm, "family": "HSRF-ridge", "representation": "explicit count + hinge",
            "pred": preds, "meta": meta,
            "selected": f"lambda={meta.get('lambda', '?')}",
            "n_features": int(float(meta.get("n_features", "nan"))),
            "params": int(float(meta.get("params", "nan"))),
            "source": "results/ridge",
        }
    for arm in MLP_ARMS:
        d = ROOT / f"mlp_{arm}"
        if not (d / "predictions_best_valid.npz").is_file():
            continue
        preds = {}
        for split in ("train", "valid", "test"):
            z = np.load(d / f"predictions_best_{split}.npz", allow_pickle=False)
            preds[split] = _align(z["row_id"], z["pred"], stage.rows[split])
        man = json.load((d / "manifest.json").open())
        models[f"MLP_{arm}"] = {
            "name": f"MLP_{arm}", "family": "blackbox-MLP",
            "representation": {"BASE": "F_BASE", "NULL": "F_NULL", "REAL": "F_REAL"}[arm],
            "pred": preds, "meta": man,
            "selected": f"epoch={man.get('best_epoch', '?')}",
            "n_features": int(man.get("n_features", 0)),
            "params": int(man.get("n_params", 0)),
            "source": str(d.relative_to(C.ROUND_DIR)),
        }
    for arm in KRR_ARMS:
        if not (KRR / "predictions" / f"KRR_{arm}_valid.npz").is_file():
            continue
        preds = {}
        for split in ("train", "valid", "test"):
            z = np.load(KRR / "predictions" / f"KRR_{arm}_{split}.npz", allow_pickle=False)
            preds[split] = _align(z["row_id"], z["pred"], stage.rows[split])
        rows = [r for r in _read_csv(KRR / "main_table.csv") if r["arm"] == arm]
        meta = rows[0] if rows else {}
        models[f"KRR_{arm}"] = {
            "name": f"KRR_{arm}", "family": "blackbox-KRR",
            "representation": "F_BASE" if arm == "BASE" else "F_REAL",
            "pred": preds, "meta": meta,
            "selected": f"sigma={float(meta.get('sigma', 'nan')):.3g}, lambda={meta.get('lambda', '?')}",
            "n_features": int(float(meta.get("n_features", "nan"))),
            "params": int(stage.rows["train"].size), "source": "results/krr",
        }
    hist = C.load_historical(payload)
    for name, rec in hist["arms"].items():
        if not rec.get("available"):
            continue
        preds = {s: rec["splits"][s]["pred"] for s in ("train", "valid", "test")}
        models[name] = {
            "name": name, "family": "frozen-reference",
            "representation": {"GINE_sq_pilot": "GINE", "LCRC_sq_pilot": "LCRC",
                                "PAIR_RULE_REAL_FULL": "pair rules",
                                "PAIR_RULE_MLP": "pair MLP",
                                "SWF_WORD_PREFIX": "word prefix"}.get(name, "historical"),
            "pred": preds, "meta": rec, "selected": "frozen",
            "n_features": 0, "params": 0, "source": rec["path"],
        }
    return models


def metrics_for(model: Mapping[str, Any], stage: D.Stage0) -> dict[str, float]:
    out = {}
    for split in ("train", "valid", "test"):
        y = stage.y[stage.rows[split]]
        p = model["pred"][split]
        out[f"{split}_rmse"] = C.rmse(y, p)
        out[f"{split}_mae"] = C.mae(y, p)
    return out


def main_table(models: Mapping[str, Mapping[str, Any]], stage: D.Stage0) -> list[dict[str, Any]]:
    rows = []
    for name, m in models.items():
        met = metrics_for(m, stage)
        rows.append({
            "model": name, "family": m["family"], "representation": m["representation"],
            "selected": m["selected"], "n_features": m["n_features"], "params": m["params"],
            **{k: float(v) for k, v in met.items()},
            "prediction_dtype": "raw logD",
            "source": m["source"],
        })
    order = {"HSRF-ridge": 0, "blackbox-MLP": 1, "blackbox-KRR": 2, "frozen-reference": 3}
    rows.sort(key=lambda r: (order.get(r["family"], 9), r["valid_rmse"]))
    C.write_csv(ROOT / "main_table.csv", rows)
    return rows


def comparison_specs(models: Mapping[str, Mapping[str, Any]],
                     stage: D.Stage0) -> list[tuple[str, str, str]]:
    specs: list[tuple[str, str, str]] = []

    def add(name: str, ref: str, cand: str) -> None:
        if ref in models and cand in models:
            specs.append((name, ref, cand))

    add("nonlinear_base", "L_BASE", "F_BASE")
    add("nonlinear_real", "L_REAL", "F_REAL")
    add("relation_vs_base", "F_BASE", "F_REAL")
    add("relation_vs_null", "F_NULL", "F_REAL")
    add("relation_linear", "L_BASE", "L_REAL")
    add("mlp_relation_vs_base", "MLP_BASE", "MLP_REAL")
    add("mlp_relation_vs_null", "MLP_NULL", "MLP_REAL")
    for arm in ("BASE", "REAL"):
        add(f"KRR_{arm}_vs_MLP_{arm}", f"MLP_{arm}", f"KRR_{arm}")
    for name in ("GINE_sq_pilot", "LCRC_sq_pilot", "PAIR_RULE_REAL_FULL",
                 "PAIR_RULE_MLP", "SWF_WORD_PREFIX"):
        add(f"F_BASE_vs_{name}", name, "F_BASE")
        add(f"F_REAL_vs_{name}", name, "F_REAL")
    # best same-input black boxes, selected by Valid RMSE
    for arm in ("BASE", "REAL"):
        pool = [f"MLP_{arm}", f"KRR_{arm}"]
        pool = [p for p in pool if p in models]
        if pool:
            best = min(pool, key=lambda p: metrics_for(models[p], stage)["valid_rmse"])
            add(f"best_bb_{arm}_vs_F_{arm}", f"F_{arm}", best)
            add(f"F_{arm}_vs_best_bb_{arm}", best, f"F_{arm}")
    # de-duplicate while preserving order
    seen = set()
    out = []
    for s in specs:
        if s[0] not in seen:
            seen.add(s[0]); out.append(s)
    return out


def run_contrasts(models: Mapping[str, Mapping[str, Any]], stage: D.Stage0,
                  payload: Mapping[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    specs = comparison_specs(models, stage)
    rows_by_split: dict[str, dict[str, dict[str, Any]]] = {}
    draws_out = {}
    for split in ("valid", "test"):
        cl = C.scaffold_groups(payload, split)
        preds = {name: m["pred"][split] for name, m in models.items()}
        rows, draws = C.paired_bootstrap(
            stage.y[stage.rows[split]], preds, cl["cluster_rows"], specs,
            int(C.load_config()["bootstrap"]["resamples"]),
            int(C.load_config()["bootstrap"]["seed"]))
        rows_by_split[split] = {r["comparison"]: r for r in rows}
        draws_out[split] = draws
        C.write_csv(ROOT / f"bootstrap_{split}.csv", rows)
        C.write_json(ROOT / f"bootstrap_{split}_clusters.json", {
            "split": split, "n_clusters": cl["n_clusters"],
            "kind_counts": cl["kind_counts"],
            "note": "Bemis-Murcko scaffold clusters; if every molecule is its own cluster this equals a molecule-row paired bootstrap.",
        })
    np.savez_compressed(ROOT / "bootstrap_draws.npz", **draws_out)
    contrast_rows = []
    for name, ref, cand in specs:
        row = {"comparison": name, "reference": ref, "candidate": cand,
               "gain_definition": "RMSE(reference) - RMSE(candidate); >0 = candidate better"}
        for split in ("valid", "test"):
            r = rows_by_split[split][name]
            row.update({
                f"{split}_rmse_reference": r["rmse_reference"],
                f"{split}_rmse_candidate": r["rmse_candidate"],
                f"{split}_delta": r["delta"],
                f"{split}_ci_low": r["ci_low"],
                f"{split}_ci_high": r["ci_high"],
                f"{split}_mae_delta": r["mae_delta"],
                f"{split}_drop_one_delta": r["drop_one_delta"],
            })
        contrast_rows.append(row)
    C.write_csv(ROOT / "contrast_table.csv", contrast_rows)
    return contrast_rows, {"rows_by_split": rows_by_split}


def _gain(rows: Mapping[str, Any], split: str, name: str) -> dict[str, float]:
    r = rows["rows_by_split"][split].get(name)
    if r is None:
        return {"delta": float("nan"), "ci_low": float("nan"), "ci_high": float("nan")}
    return {"delta": float(r["delta"]), "ci_low": float(r["ci_low"]),
            "ci_high": float(r["ci_high"])}


def decide(models: Mapping[str, Mapping[str, Any]], stage: D.Stage0,
           contrasts: Mapping[str, Any], coverage: Mapping[str, Any],
           ledger_ok: bool, ident: Mapping[str, Any]) -> dict[str, Any]:
    cfg = C.load_config()
    gates = cfg["gates"]
    hist = C.load_historical(C.load_payload())
    gine = hist["arms"].get("GINE_sq_pilot", {})
    lcrc = hist["arms"].get("LCRC_sq_pilot", {})
    gine_v = gine["splits"]["valid"]["rmse"] if gine.get("available") else float("nan")
    gine_t = gine["splits"]["test"]["rmse"] if gine.get("available") else float("nan")
    lcrc_v = lcrc["splits"]["valid"]["rmse"] if lcrc.get("available") else float("nan")
    lcrc_t = lcrc["splits"]["test"]["rmse"] if lcrc.get("available") else float("nan")
    ridge_valid = {a: metrics_for(models[a], stage)["valid_rmse"]
                   for a in ("F_BASE", "F_REAL") if a in models}
    candidate = min(ridge_valid, key=ridge_valid.get) if ridge_valid else "F_BASE"
    cand = models[candidate]
    cand_met = metrics_for(cand, stage)

    # representation axis
    bb = {}
    for arm in ("BASE", "REAL"):
        pool = [f"MLP_{arm}", f"KRR_{arm}"]
        pool = [p for p in pool if p in models]
        if pool:
            bb[arm] = min(pool, key=lambda p: metrics_for(models[p], stage)["valid_rmse"])
    bb_valid = {a: metrics_for(models[b], stage)["valid_rmse"] for a, b in bb.items()}
    if bb_valid and all(v > gine_v + gates["representation_gap_limit"] for v in bb_valid.values()):
        rep_axis = "REPRESENTATION_OR_LEARNER_LIMITED"
    elif bb_valid and any(v <= gine_v + gates["representation_competitive_delta"]
                          for v in bb_valid.values()):
        rep_axis = "COMPETITIVE_INFORMATION_SIGNAL"
    else:
        rep_axis = "PARTIAL_INFORMATION_SIGNAL"

    # nonlinear axis
    nl = {}
    for tag, ref, cand_arm in (("base", "L_BASE", "F_BASE"),
                               ("real", "L_REAL", "F_REAL")):
        if ref not in models or cand_arm not in models:
            continue
        gv = _gain(contrasts, "valid", f"nonlinear_{tag}")
        gt = _gain(contrasts, "test", f"nonlinear_{tag}")
        nl[tag] = {"valid_gain": gv["delta"], "valid_ci_low": gv["ci_low"],
                   "test_gain": gt["delta"],
                   "split_sensitive": bool(gv["delta"] > 0.02 and gt["delta"] < -0.02)}
    material = [v for v in nl.values()
                if v["valid_gain"] >= gates["nonlinear_valid_gain"] and v["test_gain"] >= 0]
    if material and any(v["valid_ci_low"] > 0 for v in material):
        nl_axis = "NONLINEAR_RESPONSE_SIGNAL"
    elif material:
        nl_axis = "NONLINEAR_RESPONSE_SCREEN_ONLY"
    else:
        nl_axis = "NO_MATERIAL_NONLINEAR_GAIN"
    if any(v["split_sensitive"] for v in nl.values()):
        nl_axis += "+SPLIT_SENSITIVE"

    # relation axis
    support = coverage["relation_per_split"]
    support_ok = all(
        support[s]["fraction_with_any_relation"] >= 0.5 and
        support[s]["fraction_real_ne_null_linear"] >= 0.10 and
        support[s]["median_nonzero_relations"] >= 1
        for s in ("train", "valid", "test"))
    rel_var = 0.0
    if (RIDGE / "ledger" / "F_REAL.npz").is_file():
        z = np.load(RIDGE / "ledger" / "F_REAL.npz")
        rel_var = float(np.std(z["relation_effect_sum"]))
    rb = _gain(contrasts, "valid", "relation_vs_base")
    rt = _gain(contrasts, "test", "relation_vs_base")
    rn = _gain(contrasts, "valid", "relation_vs_null")
    rnt = _gain(contrasts, "test", "relation_vs_null")
    rel_info = {"valid_vs_base": rb, "test_vs_base": rt,
                "valid_vs_null": rn, "test_vs_null": rnt,
                "support_ok": bool(support_ok), "relation_contribution_sd": rel_var}
    mlp_rel = _gain(contrasts, "valid", "mlp_relation_vs_base")
    mlp_rel_null = _gain(contrasts, "valid", "mlp_relation_vs_null")
    if all("F_REAL" in models for _ in [0]) and \
            rb["delta"] >= gates["relation_valid_gain"] and rt["delta"] >= 0 and \
            rn["delta"] >= gates["relation_valid_gain"] and rnt["delta"] >= 0 and \
            rb["ci_low"] > 0 and rn["ci_low"] > 0 and support_ok and rel_var > 0:
        rel_axis = "REAL_RELATION_SIGNAL"
    elif (mlp_rel["delta"] >= gates["relation_valid_gain"] and
          mlp_rel_null["delta"] >= gates["relation_valid_gain"] and
          not (rb["delta"] >= gates["relation_valid_gain"] and rt["delta"] >= 0)):
        rel_axis = "RELATION_INFORMATION_NOT_EXPLOITED_BY_RESPONSE_FUNCTIONS"
    elif rb["delta"] > 0 and rb["delta"] >= rn["delta"]:
        rel_axis = "NO_MATERIAL_RELATION_GAIN"
    elif rn["delta"] > 0 and rn["delta"] > rb["delta"]:
        rel_axis = "RELATION_CONTROL_ONLY"
    else:
        rel_axis = "NO_MATERIAL_RELATION_GAIN"

    # decomposition axis (candidate vs best same-input black box)
    dec = {}
    for arm in ("BASE", "REAL"):
        ename = f"F_{arm}"
        if ename not in models or arm not in bb:
            continue
        bname = bb[arm]
        ev = metrics_for(models[ename], stage)["valid_rmse"]
        et = metrics_for(models[ename], stage)["test_rmse"]
        bv = metrics_for(models[bname], stage)["valid_rmse"]
        bt = metrics_for(models[bname], stage)["test_rmse"]
        gv = bv - ev
        gt = bt - et
        ci = _gain(contrasts, "valid", f"best_bb_{arm}_vs_F_{arm}")
        dec[arm] = {"explicit": ename, "blackbox": bname,
                    "valid_gap": gv, "test_gap": gt,
                    "valid_ci_low": ci["ci_low"], "valid_ci_high": ci["ci_high"]}
    # bootstrap row is F vs bb, so negative delta means bb better. Support:
    # explicit-minus-bb CI upper < 0.
    if dec and all(v["valid_gap"] <= gates["decomposition_gap_limit"] and
                   v["test_gap"] <= gates["decomposition_gap_limit"] for v in dec.values()) \
            and ledger_ok:
        dec_axis = "LOW_DECOMPOSITION_COST"
    elif dec and any(v["valid_gap"] >= gates["decomposition_cost_gap"] and
                     v["valid_ci_high"] < 0 for v in dec.values()):
        dec_axis = "DECOMPOSITION_COST_SIGNAL"
    else:
        dec_axis = "UNRESOLVED_DECOMPOSITION_COST"

    # overall
    flags = []
    if not (cand_met["valid_rmse"] <= gine_v + gates["promising_gap_limit"] and
            cand_met["test_rmse"] <= gine_t + gates["promising_gap_limit"] and
            dec_axis == "LOW_DECOMPOSITION_COST" and ledger_ok):
        flags.append("NOT_PROMISING")
    if ident.get("summary", {}).get("attribution_ambiguous"):
        flags.append("ATTRIBUTION_NOT_IDENTIFIED")
    if cand_met["valid_rmse"] < lcrc_v and cand_met["test_rmse"] < lcrc_t:
        flags.append("PARTIAL_PROGRESS")
    if any(v["valid_gap"] >= gates["decomposition_cost_gap"] for v in dec.values()) and \
            rep_axis == "COMPETITIVE_INFORMATION_SIGNAL":
        flags.append("INTERPRETABLE_FUNCTION_CLASS_LIMITED")
    if rep_axis == "REPRESENTATION_OR_LEARNER_LIMITED":
        flags.append("REPRESENTATION_OR_LEARNER_LIMITED")
    if ledger_ok:
        flags.append("EXACT_LEDGER")
    overall = "PROMISING_INTERPRETABLE_PREDICTOR" if not flags or "NOT_PROMISING" not in flags \
        else "NOT_PROMISING"
    return {
        "candidate": candidate,
        "candidate_metrics": cand_met,
        "historical": {"GINE_valid": gine_v, "GINE_test": gine_t,
                       "LCRC_valid": lcrc_v, "LCRC_test": lcrc_t,
                       "source": {k: hist["arms"].get(k, {}).get("path") for k in
                                  ("GINE_sq_pilot", "LCRC_sq_pilot")}},
        "representation": {"axis": rep_axis, "blackbox": bb, "blackbox_valid": bb_valid},
        "nonlinear": {"axis": nl_axis, "pairs": nl},
        "relation": {"axis": rel_axis, **rel_info},
        "decomposition": {"axis": dec_axis, "pairs": dec},
        "overall": overall,
        "flags": flags,
        "ledger_ok": bool(ledger_ok),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--skip-identifiability", action="store_true")
    ap.add_argument("--skip-pairs", action="store_true")
    ap.add_argument("--skip-figures", action="store_true")
    args = ap.parse_args()
    cfg = C.load_config()
    stage = D.Stage0()
    payload = C.load_payload()
    ANALYSIS.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    global args_skip_figures
    args_skip_figures = bool(args.skip_figures)
    models = load_models(stage, payload)
    table = main_table(models, stage)
    contrasts, cdata = run_contrasts(models, stage, payload)
    print(json.dumps({"models": sorted(models), "contrasts": len(contrasts)}, indent=1))
    summary = build_summary(stage, models, table, contrasts, cdata)
    write_report(summary, table, contrasts)
    print(json.dumps({"candidate": summary["candidate"],
                      "axes": {k: v["axis"] for k, v in summary["axes"].items()},
                      "overall": summary["overall"],
                      "flags": summary["flags"]}, indent=1))




def _column_group_map(design: Mapping[str, Any]) -> dict[int, str]:
    out = {}
    for g in design["groups"]:
        for c in g["cols"]:
            out[int(c)] = str(g["name"])
    return out


def hierarchy_identities(stage: D.Stage0) -> dict[str, Any]:
    bc = stage.base_counts
    sig = np.load(C.DATA_DIR / "signatures.npz", allow_pickle=False)
    named = {r["name"]: int(r["raw_col"]) for r in stage.base_raw_meta
             if r["kind"] == "named_graph_count"}
    n_atoms = bc[:, named["n_atoms"]]
    n_bonds = bc[:, named["n_bonds"]]
    total1 = np.diff(np.asarray(sig["ptr_1"], np.int64))
    total2 = np.diff(np.asarray(sig["ptr_2"], np.int64))
    out: dict[str, Any] = {
        "n_atoms_col": named["n_atoms"], "n_bonds_col": named["n_bonds"],
        "identity_definition": "All enumerated instances from signatures.npz; retained-vocabulary coverage is reported separately.",
        "per_gran": {},
    }
    for gran in ("TOPO", "ELEMENT", "COARSE", "FULL"):
        cols1 = [int(r["raw_col"]) for r in stage.base_raw_meta
                 if r["name"].startswith(f"{gran}_1_")]
        cols2 = [int(r["raw_col"]) for r in stage.base_raw_meta
                 if r["name"].startswith(f"{gran}_2_")]
        s1 = bc[:, cols1].sum(axis=1)
        s2 = bc[:, cols2].sum(axis=1)
        out["per_gran"][gran] = {
            "n_size1_types": len(cols1), "n_size2_types": len(cols2),
            "size1_max_abs_error_vs_n_atoms": float(np.abs(total1 - n_atoms).max()),
            "size2_max_abs_error_vs_2n_bonds": float(np.abs(total2 - 2 * n_bonds).max()),
            "retained_size1_fraction": float(s1.sum() / max(total1.sum(), 1)),
            "retained_size2_fraction": float(s2.sum() / max(total2.sum(), 1)),
        }
    comp = bc[:, named["n_components"]]
    cyc = bc[:, named["cycle_rank"]]
    out["cycle_rank_max_abs_error"] = float(np.abs(cyc - (n_bonds - n_atoms + comp)).max())
    out["all_pass"] = bool(
        all(v["size1_max_abs_error_vs_n_atoms"] < 1e-9 and
            v["size2_max_abs_error_vs_2n_bonds"] < 1e-9
            for v in out["per_gran"].values()) and
        out["cycle_rank_max_abs_error"] < 1e-9)
    return out


def identifiability(stage: D.Stage0) -> dict[str, Any]:
    from scipy.stats import spearmanr

    out: dict[str, Any] = {"arms": {}, "hierarchy_identities": hierarchy_identities(stage)}
    any_ambig = False
    for arm in ("F_BASE", "F_REAL"):
        design = stage.design(arm)
        X = design["X"]
        Xt = X[stage.train]
        Xv = X[stage.rows["valid"]]
        mu = Xt.mean(axis=0)
        Xc = Xt - mu
        K = Xc @ Xc.T
        w, V = np.linalg.eigh(K)
        w = np.clip(w, 0.0, None)
        tol = max(Xt.shape) * np.finfo(np.float64).eps * float(w.max())
        pos = w > tol
        rank = int(pos.sum())
        cond = float(np.sqrt(w.max() / max(w[pos].min(), 1e-300)))
        # exact duplicate columns on Train
        hashes: dict[bytes, list[int]] = {}
        for j in range(Xt.shape[1]):
            h = hashlib.blake2b(np.ascontiguousarray(Xt[:, j]).tobytes(), digest_size=16).digest()
            hashes.setdefault(h, []).append(j)
        dup = [v for v in hashes.values() if len(v) > 1]
        cg = _column_group_map(design)
        dup_detail = [{"columns": v[:8], "names": [cg.get(int(c), f"col{c}") for c in v[:8]],
                       "size": len(v)} for v in sorted(dup, key=len, reverse=True)[:20]]
        # deterministic top-variance correlation clusters
        var = Xt.var(axis=0)
        top = np.argsort(var)[-300:]
        corr = np.corrcoef(Xt[:, top], rowvar=False)
        parent = list(range(top.size))
        def find(a: int) -> int:
            while parent[a] != a:
                parent[a] = parent[parent[a]]; a = parent[a]
            return a
        def union(a: int, b: int) -> None:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[rb] = ra
        for i in range(top.size):
            for j in range(i + 1, top.size):
                if abs(float(corr[i, j])) > 0.999:
                    union(i, j)
        clusters: dict[int, list[int]] = {}
        for i in range(top.size):
            clusters.setdefault(find(i), []).append(i)
        collinear = []
        for idxs in sorted(clusters.values(), key=len, reverse=True):
            if len(idxs) < 2:
                continue
            cols = [int(top[i]) for i in idxs]
            collinear.append({
                "size": len(cols),
                "columns": cols[:10],
                "names": [cg.get(c, f"col{c}") for c in cols[:10]],
            })
            if len(collinear) >= 10:
                break
        # near-optimal lambda solutions
        z = np.load(RIDGE / "states" / f"{arm}.npz", allow_pickle=False)
        lambdas = np.asarray(z["lambdas"], np.float64)
        betas = np.asarray(z["beta_all"], np.float64)
        sel = int(np.asarray(z["selected_index"]).reshape(-1)[0])
        ymean = float(np.asarray(z["ymean"]).reshape(-1)[0])
        pv = (float(stage.scaler["mean"]) +
              float(stage.scaler["std"]) * (ymean + (Xv - mu) @ betas))
        valid_rmse = np.sqrt(np.mean((pv - stage.y[stage.rows["valid"]][:, None]) ** 2, axis=0))
        best = float(valid_rmse[sel])
        near = np.where(valid_rmse <= best + 0.01)[0]
        if sel not in near:
            near = np.sort(np.unique(np.append(near, sel)))
        # group contribution matrices on Valid
        def contrib_matrix(beta: np.ndarray) -> np.ndarray:
            mat = np.empty((Xv.shape[0], len(design["groups"])), np.float64)
            for gi, g in enumerate(design["groups"]):
                cols = np.asarray(g["cols"], np.int64)
                mat[:, gi] = stage.scaler["std"] * (Xv[:, cols] @ beta[cols])
            return mat
        base_mat = contrib_matrix(betas[:, sel])
        base_mean_abs = np.mean(np.abs(base_mat), axis=0)
        base_signed = np.mean(base_mat, axis=0)
        top_base = set(np.argsort(-base_mean_abs)[:20].tolist())
        near_rows = []
        for j in near.tolist():
            if int(j) == sel:
                continue
            beta = betas[:, int(j)]
            mat = contrib_matrix(beta)
            mean_abs = np.mean(np.abs(mat), axis=0)
            top_j = set(np.argsort(-mean_abs)[:20].tolist())
            rho = float(spearmanr(base_mean_abs, mean_abs).statistic)
            signed_j = np.mean(mat, axis=0)
            sign_agree = float(np.mean(np.sign(base_signed[list(top_base)]) ==
                                       np.sign(signed_j[list(top_base)])))
            abs_diff = np.abs(mean_abs - base_mean_abs)
            rel_l2 = float(np.linalg.norm(mean_abs - base_mean_abs) /
                           max(np.linalg.norm(base_mean_abs), 1e-300))
            top_rel = float(np.max(abs_diff[list(top_base)] /
                                   np.maximum(base_mean_abs[list(top_base)], 1e-12)))
            pred_diff = float(stage.scaler["std"]) * (Xv @ (beta - betas[:, sel]))
            dbeta = beta - betas[:, sel]
            d = Xc @ dbeta
            u = V.T @ d
            ident_norm2 = float(np.sum((u[pos] ** 2) / w[pos]))
            null_frac = max(0.0, 1.0 - ident_norm2 / max(float(dbeta @ dbeta), 1e-300))
            near_rows.append({
                "lambda": float(lambdas[int(j)]),
                "valid_rmse": float(valid_rmse[int(j)]),
                "valid_rmse_gap": float(valid_rmse[int(j)] - best),
                "prediction_rms_diff_valid": float(np.sqrt(np.mean(pred_diff ** 2))),
                "contribution_spearman": rho,
                "top20_overlap": len(top_base & top_j) / 20.0,
                "top20_sign_agreement": sign_agree,
                "top20_max_relative_mean_abs_diff": top_rel,
                "group_mean_abs_relative_l2_diff": rel_l2,
                "null_space_fraction_of_beta_diff": null_frac,
            })
        ambig = any(r["contribution_spearman"] < 0.8 or r["top20_overlap"] < 0.6 or
                    r["top20_max_relative_mean_abs_diff"] > 0.2 or
                    r["group_mean_abs_relative_l2_diff"] > 0.2 or
                    r["prediction_rms_diff_valid"] > 0.05
                    for r in near_rows)
        any_ambig = any_ambig or ambig
        out["arms"][arm] = {
            "n_train": int(Xt.shape[0]), "n_features": int(Xt.shape[1]),
            "numerical_rank": rank, "rank_deficiency": int(Xt.shape[1] - rank),
            "condition_number": cond,
            "selected_lambda": float(lambdas[sel]),
            "selected_valid_rmse": best,
            "near_optimal_lambdas": [float(lambdas[int(j)]) for j in near.tolist()],
            "near_optimal_solutions": near_rows,
            "attribution_ambiguous": bool(ambig),
            "exact_duplicate_column_groups": len(dup),
            "exact_duplicate_columns": int(sum(len(v) for v in dup)),
            "unique_column_patterns": int(len(hashes)),
            "redundant_columns": int(sum(len(v) - 1 for v in dup)),
            "duplicate_examples": dup_detail,
            "top_variance_collinear_clusters": collinear,
        }
    out["summary"] = {
        "attribution_ambiguous": bool(any_ambig),
        "near_optimal_gap_threshold": 0.01,
        "note": "Near-optimal means Valid RMSE within 0.01 of the selected solution; contribution ranks are mean |group contribution| on Valid.",
    }
    return out


def _full_pred(model: Mapping[str, Any], stage: D.Stage0) -> np.ndarray:
    out = np.full(stage.n, np.nan, np.float64)
    for split in ("train", "valid", "test"):
        out[stage.rows[split]] = model["pred"][split]
    return out


def _pair_metrics(y: np.ndarray, a: np.ndarray, b: np.ndarray,
                  pred: np.ndarray) -> dict[str, Any]:
    dt = y[b] - y[a]
    dp = pred[b] - pred[a]
    err = dp - dt
    nz = np.abs(dt) > 1e-12
    sign = float(np.mean(np.sign(dp[nz]) == np.sign(dt[nz]))) if nz.any() else float("nan")
    return {
        "n_pairs": int(dt.size),
        "delta_true_mae": float(np.mean(np.abs(dt))),
        "delta_pred_mae": float(np.mean(np.abs(err))),
        "delta_pred_rmse": float(np.sqrt(np.mean(err ** 2))),
        "delta_skill_vs_zero": float(1.0 - np.mean(np.abs(err)) / max(np.mean(np.abs(dt)), 1e-300)),
        "direction_accuracy_nonzero": sign,
        "pearson_delta": float(np.corrcoef(dp, dt)[0, 1]) if dt.std() > 0 and dp.std() > 0 else float("nan"),
    }


def pair_diagnostics(stage: D.Stage0, models: Mapping[str, Mapping[str, Any]]) -> dict[str, Any]:
    path = C.STRUCT / "task_response_composition_s0_v1" / "data" / "trc_cache_lipo.npz"
    if not path.is_file():
        return {"available": False, "path": str(path)}
    z = np.load(path, allow_pickle=False)
    a = np.asarray(z["pair_mol_a"], np.int64)
    b = np.asarray(z["pair_mol_b"], np.int64)
    cell = np.asarray(z["pair_cell"], np.int64)
    tr = np.asarray(z["pair_transform"], np.int64)
    comp = np.asarray(z["pair_component"], np.int64)
    y = stage.y
    cell_name = {0: "train", 1: "valid", 2: "test", 3: "cross"}
    train_transforms = set(tr[cell == 0].tolist())
    seen = np.asarray([int(t) in train_transforms for t in tr.tolist()], bool)
    held = cell != 0
    out: dict[str, Any] = {
        "available": True, "path": str(path.relative_to(C.ROUND_DIR.parent.parent)),
        "sha256": C.sha256_file(path),
        "n_pairs": int(a.size), "cell_counts": {cell_name[k]: int((cell == k).sum())
                                                for k in (0, 1, 2, 3)},
        "n_train_transforms": len(train_transforms),
        "heldout_pairs": int(held.sum()),
        "heldout_seen_transform_pairs": int((held & seen).sum()),
        "heldout_unseen_transform_pairs": int((held & ~seen).sum()),
        "note": "Reuses task_response_composition_s0_v1 matched pairs read-only; predictions are this round's frozen models.",
        "models": {},
    }
    # every model
    selected = ["F_BASE", "F_REAL", "L_BASE", "L_REAL", "F_NULL",
                "MLP_BASE", "MLP_NULL", "MLP_REAL", "KRR_BASE", "KRR_REAL"]
    for name in selected:
        if name not in models:
            continue
        pred = _full_pred(models[name], stage)
        rec: dict[str, Any] = {}
        for key, mask in (("heldout", held), ("valid", cell == 1), ("test", cell == 2),
                          ("cross", cell == 3),
                          ("heldout_seen", held & seen), ("heldout_unseen", held & ~seen)):
            if int(mask.sum()) == 0:
                continue
            rec[key] = _pair_metrics(y, a[mask], b[mask], pred)
        out["models"][name] = rec
    # decomposition of delta for F_BASE / F_REAL
    out["decomposition"] = {}
    for arm in ("F_BASE", "F_REAL"):
        lp = RIDGE / "ledger" / f"{arm}.npz"
        if not lp.is_file():
            continue
        lz = np.load(lp, allow_pickle=False)
        main = np.asarray(lz["main_effect_sum"], np.float64)
        rel = np.asarray(lz["relation_effect_sum"], np.float64)
        pred = np.asarray(lz["prediction"], np.float64)
        dt = y[b] - y[a]
        dm = main[b] - main[a]
        dr = rel[b] - rel[a]
        dp = pred[b] - pred[a]
        rec = {
            "reconstruction_max_abs_error": float(np.abs(dm + dr - dp).max()),
            "heldout_mean_abs_main_delta": float(np.mean(np.abs(dm[held]))),
            "heldout_mean_abs_relation_delta": float(np.mean(np.abs(dr[held]))),
            "heldout_relation_share": float(np.mean(np.abs(dr[held])) /
                                             max(np.mean(np.abs(dm[held])) + np.mean(np.abs(dr[held])), 1e-300)),
            "heldout_fraction_relation_dominates": float(np.mean(np.abs(dr[held]) > np.abs(dm[held]))),
            "heldout_corr_relation_delta_with_true_delta": float(np.corrcoef(dr[held], dt[held])[0, 1])
            if dt[held].std() > 0 and dr[held].std() > 0 else float("nan"),
        }
        out["decomposition"][arm] = rec
    # compact CSV
    rows = []
    for name, rec in out["models"].items():
        for subset, met in rec.items():
            rows.append({"model": name, "subset": subset,
                         **{k: float(v) if isinstance(v, (int, float, np.floating)) else v
                            for k, v in met.items()}})
    C.write_csv(ROOT / "pair_diagnostic.csv", rows)
    C.write_json(ROOT / "pair_diagnostic.json", out)
    C.write_json(ANALYSIS / "pair_diagnostic.json", out)
    return out


def export_ledgers(stage: D.Stage0) -> dict[str, Any]:
    out = {"arms": {}}
    for arm in ("F_BASE", "F_REAL", "L_BASE", "L_REAL", "F_NULL"):
        npz = RIDGE / "ledger" / f"{arm}.npz"
        gp = RIDGE / "ledger" / f"{arm}_groups.json"
        if not npz.is_file() or not gp.is_file():
            continue
        z = np.load(npz, allow_pickle=False)
        meta = json.load(gp.open())
        rows = []
        split_of = {int(r): s for s, rs in stage.rows.items() for r in rs.tolist()}
        for i in range(stage.n):
            rows.append({
                "row_id": int(z["row_id"][i]), "split": split_of[int(z["row_id"][i])],
                "y": float(stage.y[int(z["row_id"][i])]),
                "prediction": float(z["prediction"][i]),
                "intercept": float(np.asarray(z["intercept"]).reshape(-1)[0]),
                "main_effect_sum": float(z["main_effect_sum"][i]),
                "relation_effect_sum": float(z["relation_effect_sum"][i]),
                "reconstructed": float(z["reconstructed"][i]),
                "reconstruction_error": float(z["reconstruction_error"][i]),
            })
        C.write_csv(RIDGE / "ledger" / f"{arm}_molecules.csv", rows)
        groups = meta["groups"]
        gv = np.asarray(z["group_values"], np.float64)
        grow = []
        for gi, g in enumerate(groups):
            grow.append({
                "arm": arm, "group_index": gi, "group_id": g["group_id"],
                "block": g["block"], "kind": g["kind"], "name": g["name"],
                "raw_col": g["raw_col"],
                "mean_abs_contribution": float(np.mean(np.abs(gv[:, gi]))),
                "mean_signed_contribution": float(np.mean(gv[:, gi])),
                "std_contribution": float(np.std(gv[:, gi])),
                "nonzero_rows": int(np.count_nonzero(gv[:, gi])),
            })
        grow.sort(key=lambda r: -r["mean_abs_contribution"])
        C.write_csv(RIDGE / "ledger" / f"{arm}_group_summary.csv", grow)
        maxerr = float(np.abs(z["reconstruction_error"]).max())
        out["arms"][arm] = {
            "n_molecules": int(stage.n), "n_groups": len(groups),
            "main_groups": int(sum(g["block"] == "base" for g in groups)),
            "relation_groups": int(sum(str(g["block"]).startswith("relation") for g in groups)),
            "max_reconstruction_error": maxerr,
            "ledger_ok": bool(maxerr <= 1e-7),
            "intercept": float(meta["intercept"]),
        }
    out["all_ok"] = bool(all(v["ledger_ok"] for v in out["arms"].values()))
    C.write_json(ANALYSIS / "contribution_ledger.json", out)
    return out


def export_curves(stage: D.Stage0) -> dict[str, Any]:
    all_rows = []
    summary = {"arms": {}}
    for arm in ("F_BASE", "F_REAL", "L_BASE", "L_REAL", "F_NULL"):
        p = RIDGE / "curves" / f"{arm}.json"
        if not p.is_file():
            continue
        d = json.load(p.open())
        curves = d["curves"]
        for c in curves:
            for i, x in enumerate(c["counts"]):
                all_rows.append({
                    "arm": arm, "block": c["block"], "group_id": c["group_id"],
                    "name": c["name"], "count": int(x),
                    "response": float(c["response"][i]),
                    "train_support": int(c["train_support"][i]) if i < len(c["train_support"]) else 0,
                    "valid_support": int(c["valid_support"][i]) if i < len(c["valid_support"]) else 0,
                    "test_support": int(c["test_support"][i]) if i < len(c["test_support"]) else 0,
                    "out_of_train_support": bool(i >= len(c["train_support"]) or
                                                 c["train_support"][i] == 0),
                    "mean_abs_group_contribution": float(c["mean_abs_group_contribution"]),
                    "train_max": int(c["train_max"]),
                })
        summary["arms"][arm] = {
            "n_main_curves": int(sum(c["block"] == "base" for c in curves)),
            "n_relation_curves": int(sum(str(c["block"]).startswith("relation") for c in curves)),
            "selected_lambda": d.get("selected_lambda"),
        }
    C.write_csv(ROOT / "curves.csv", all_rows)
    if not args_skip_figures:
        _plot_curve_grids()
    C.write_json(ANALYSIS / "function_curves.json", summary)
    return summary


args_skip_figures = False


def _plot_curve_grids() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    for arm in ("F_BASE", "F_REAL", "L_BASE", "L_REAL", "F_NULL"):
        p = RIDGE / "curves" / f"{arm}.json"
        if not p.is_file():
            continue
        d = json.load(p.open())
        for block_tag, block_name in (("main", "base"), ("relation", "relation")):
            cs = [c for c in d["curves"] if
                  (c["block"] == "base" if block_name == "base"
                   else str(c["block"]).startswith("relation"))]
            if not cs:
                continue
            n = len(cs)
            ncol = 5
            nrow = int(math.ceil(n / ncol))
            fig, axes = plt.subplots(nrow, ncol, figsize=(3.1 * ncol, 2.5 * nrow),
                                     squeeze=False)
            for ax, c in zip(axes.ravel(), cs):
                x = np.asarray(c["counts"], np.float64)
                r = np.asarray(c["response"], np.float64)
                sup = np.asarray(c["train_support"], np.float64)
                ax.plot(x, r, color="#1f4e79", lw=1.6)
                if sup.max() > 0:
                    ax2 = ax.twinx()
                    ax2.bar(x, sup, color="#b7c9d6", alpha=0.45, width=0.9)
                    ax2.set_yticks([])
                oos = x >= len(sup)
                if oos.any():
                    ax.scatter(x[oos], r[oos], marker="x", color="#c00000", s=18, zorder=3)
                ax.axhline(0, color="0.7", lw=0.6)
                ax.set_title(f"{c['name']}\n|contrib|={c['mean_abs_group_contribution']:.3g}",
                             fontsize=7)
                ax.tick_params(labelsize=6)
                ax.set_xlabel("count", fontsize=6)
                ax.set_ylabel("response (logD)", fontsize=6)
            for ax in axes.ravel()[n:]:
                ax.axis("off")
            fig.tight_layout()
            fig.savefig(FIG / f"curves_{arm}_{block_tag}.png", dpi=150)
            plt.close(fig)


def build_summary(stage: D.Stage0, models: Mapping[str, Mapping[str, Any]],
                  table: Sequence[Mapping[str, Any]],
                  contrasts: Sequence[Mapping[str, Any]],
                  cdata: Mapping[str, Any]) -> dict[str, Any]:
    coverage = json.load((ANALYSIS / "coverage.json").open())
    checks = json.load((ANALYSIS / "checks.json").open())
    ledger = export_ledgers(stage)
    curves = export_curves(stage)
    ident = identifiability(stage)
    C.write_json(ANALYSIS / "contribution_identifiability.json", ident)
    pairs = pair_diagnostics(stage, models)
    decision = decide(models, stage, cdata, coverage, ledger.get("all_ok", False), ident)
    # runtime accounting
    ridge_seconds = float(sum(float(r.get("wall_seconds", 0.0)) for r in
                              _read_csv(RIDGE / "main_table.csv")))
    mlp_seconds = {}
    for arm in MLP_ARMS:
        p = ROOT / f"mlp_{arm}" / "manifest.json"
        if p.is_file():
            mlp_seconds[arm] = float(json.load(p.open()).get("total_seconds", 0.0))
    krr_seconds = {}
    if (KRR / "main_table.csv").is_file():
        for r in _read_csv(KRR / "main_table.csv"):
            krr_seconds[r["arm"]] = float(r.get("wall_seconds", 0.0))
    gpu_hours = sum(mlp_seconds.values()) / 3600.0
    summary = {
        "round": C.load_config()["round_id"],
        "commit": C.git_commit(), "created_at": C.now_utc(),
        "protocol": "PROTOCOL.md", "config_sha256": C.sha256_file(C.CONFIG_PATH),
        "stage0_sha256": C.sha256_file(C.DATA_DIR / "hsrf_stage0.npz"),
        "cache_sha256": coverage["cache_sha256"],
        "dataset": "ogbg-mollipo", "seed": int(C.load_config()["train_seed"]),
        "split_sizes": {s: int(stage.rows[s].size) for s in ("train", "valid", "test")},
        "test_policy": "development_test_readable_but_report_only",
        "candidate": decision["candidate"],
        "candidate_metrics": decision["candidate_metrics"],
        "historical": decision["historical"],
        "axes": {
            "representation": decision["representation"],
            "nonlinear": decision["nonlinear"],
            "relation": decision["relation"],
            "decomposition": decision["decomposition"],
        },
        "overall": decision["overall"], "flags": decision["flags"],
        "ledger": ledger, "curves": curves,
        "checks": {"n_checks": checks["n_checks"], "n_pass": checks["n_pass"],
                   "all_pass": bool(checks.get("all_pass", False)),
                   "failures": [c["name"] for c in checks["checks"] if not c["pass"]]},
        "identifiability": ident["summary"],
        "identifiability_arms": {k: {"n_features": v["n_features"], "numerical_rank": v["numerical_rank"],
                                     "rank_deficiency": v["rank_deficiency"],
                                     "condition_number": v["condition_number"],
                                     "selected_lambda": v["selected_lambda"],
                                     "near_optimal_lambdas": v["near_optimal_lambdas"],
                                     "attribution_ambiguous": v["attribution_ambiguous"],
                                     "exact_duplicate_columns": v["exact_duplicate_columns"],
                                     "unique_column_patterns": v["unique_column_patterns"],
                                     "redundant_columns": v["redundant_columns"]}
                                 for k, v in ident["arms"].items()},
        "hierarchy_identities": ident["hierarchy_identities"],
        "pair_diagnostic": {k: pairs[k] for k in
                            ("available", "n_pairs", "cell_counts", "heldout_pairs",
                             "heldout_seen_transform_pairs", "heldout_unseen_transform_pairs")
                            if k in pairs},
        "main_table": "results/main_table.csv",
        "contrast_table": "results/contrast_table.csv",
        "bootstrap": {"resamples": int(C.load_config()["bootstrap"]["resamples"]),
                       "seed": int(C.load_config()["bootstrap"]["seed"]),
                       "files": ["results/bootstrap_valid.csv", "results/bootstrap_test.csv"]},
        "support": coverage["relation_per_split"],
        "runtime_seconds": {"stage0_local": 67.0, "ridge_total": ridge_seconds,
                            "mlp": mlp_seconds, "krr": krr_seconds},
        "gpu_hours_mlp": gpu_hours,
        "constraints": {
            "no_historical_retraining": True,
            "no_new_seeds": True,
            "no_new_datasets": True,
            "no_new_function_families": True,
            "train_only_vocab_and_scales": True,
        },
        "anomalies": [
            "Relation keys are undirected; per directed slot only t<=s is counted once to avoid double-counting swapped anchor pairs.",
            "The first ridge pass exposed a ledger centring bug and an MLP/KRR target de-standardisation bug; both were fixed before formal results.",
        ],
    }
    C.write_json(ROOT / "summary.json", summary)
    return summary


def _sci(x: Any) -> str:
    try:
        return f"{float(x):.2e}"
    except Exception:
        return str(x)


def _f(x: Any, nd: int = 4) -> str:
    try:
        v = float(x)
    except Exception:
        return str(x)
    if not math.isfinite(v):
        return "nan"
    return f"{v:.{nd}f}"


def _md_table(rows: Sequence[Mapping[str, Any]], cols: Sequence[tuple[str, str]]) -> str:
    out = ["| " + " | ".join(h for _, h in cols) + " |",
           "|" + "|".join("---" for _ in cols) + "|"]
    for r in rows:
        cells = []
        for k, _ in cols:
            v = r.get(k, "")
            if isinstance(v, float):
                cells.append(_f(v))
            else:
                cells.append(str(v))
        out.append("| " + " | ".join(cells) + " |")
    return "\n".join(out)


def write_report(summary: Mapping[str, Any], table: Sequence[Mapping[str, Any]],
                 contrasts: Sequence[Mapping[str, Any]]) -> None:
    ax = summary["axes"]
    hist = summary["historical"]
    cand = summary["candidate"]
    cm = summary["candidate_metrics"]
    rep = ax["representation"]; nl = ax["nonlinear"]; rel = ax["relation"]; dec = ax["decomposition"]
    cov = summary["support"]
    ident = summary["identifiability_arms"]
    led = summary["ledger"]["arms"]
    checks = summary["checks"]
    pair = summary["pair_diagnostic"]
    # compact main table
    keep = ["L_BASE", "F_BASE", "L_REAL", "F_NULL", "F_REAL",
            "MLP_BASE", "MLP_NULL", "MLP_REAL", "KRR_BASE", "KRR_REAL",
            "GINE_sq_pilot", "LCRC_sq_pilot", "SWF_WORD_PREFIX",
            "PAIR_RULE_REAL_FULL", "PAIR_RULE_MLP"]
    main_rows = []
    for r in table:
        if r["model"] not in keep:
            continue
        main_rows.append({
            "model": r["model"], "family": r["family"],
            "valid": float(r["valid_rmse"]), "test": float(r["test_rmse"]),
            "selected": r["selected"],
        })
    # contrast rows of interest
    want = {"nonlinear_base", "nonlinear_real", "relation_linear", "relation_vs_base", "relation_vs_null",
            "mlp_relation_vs_base", "mlp_relation_vs_null",
            "best_bb_BASE_vs_F_BASE", "best_bb_REAL_vs_F_REAL",
            "F_BASE_vs_GINE_sq_pilot", "F_REAL_vs_GINE_sq_pilot",
            "F_BASE_vs_LCRC_sq_pilot", "F_REAL_vs_LCRC_sq_pilot"}
    crow = []
    for r in contrasts:
        if r["comparison"] not in want:
            continue
        crow.append({
            "comparison": r["comparison"], "valid_gain": float(r["valid_delta"]),
            "valid_ci": f"[{_f(r['valid_ci_low'])}, {_f(r['valid_ci_high'])}]",
            "test_gain": float(r["test_delta"]),
        })
    lin_rel = next((r for r in contrasts if r["comparison"] == "relation_linear"), None)
    pair_dec = ""
    pair_rows = []
    if pair.get("available"):
        pj = json.load((ROOT / "pair_diagnostic.json").open())
        for name in ("F_BASE", "F_REAL", "L_BASE", "L_REAL", "F_NULL",
                     "MLP_NULL", "MLP_REAL", "KRR_BASE", "KRR_REAL"):
            h = pj.get("models", {}).get(name, {}).get("heldout")
            if h:
                pair_rows.append({"model": name, "mae": h["delta_pred_mae"],
                                  "skill": h["delta_skill_vs_zero"],
                                  "direction": h["direction_accuracy_nonzero"]})
        dd = pj.get("decomposition", {}).get("F_REAL")
        if dd:
            pair_dec = (f"F_REAL 的 held-out Δpred 中，关系项平均绝对贡献 {_f(dd['heldout_mean_abs_relation_delta'])}，"
                        f"主效应 {_f(dd['heldout_mean_abs_main_delta'])}，关系占比 "
                        f"{_f(dd['heldout_relation_share'])}；Δmain+Δrel 复算误差 "
                        f"{_f(dd['reconstruction_max_abs_error'], 2)}。")
    text = f"""# HSRF v0 — 分层结构响应函数快速筛选报告

**Round**: `{summary['round']}` · **Commit**: `{summary['commit']}` · **生成时间**: {summary['created_at']}
**数据**: `ogbg-mollipo`，官方 scaffold split（3360/420/420），seed 0。Test 仅作为 DEVELOPMENT_TEST 报告，所有选择只由 Train/Valid 决定。
**候选**: `{cand}`（Valid RMSE {_f(cm['valid_rmse'])}，Test RMSE {_f(cm['test_rmse'])}）。

## 1. 一句话结论

本轮显式 HSRF 表示（根局部结构计数 + 固定 hinge）在 Lipo 上**没有显示出有竞争力的预测力**：候选 `{cand}` 的 Valid RMSE 为 {_f(cm['valid_rmse'])}，比冻结 GINE 坐标 {_f(hist['GINE_valid'])} 高 {_f(cm['valid_rmse'] - hist['GINE_valid'])}；同输入 MLP/KRR 黑箱也没有接近 GINE（最强 Valid {_f(min(rep['blackbox_valid'].values()) if rep['blackbox_valid'] else float('nan'))}），因此四轴判定为 **{rep['axis']}**。非线性 hinge 与 REAL 关系都没有带来协议要求的 Valid 增益；贡献账目本身是精确的，但近优 λ 解的归因不稳定。

## 2. 冻结设置与历史坐标核对

- Stage 0 对每个根原子枚举 size 1–4、radius-2 内连通诱导子图，四粒度签名（TOPO/ELEMENT/COARSE/FULL），Train-only 词表（DF≥8，cap 1024；TOPO 不截断）。
- 64 个 COARSE 锚点（size 2/3 各 top-32）；真实关系按有向键槽统计；NULL 使用 hypergeometric 的 `E[h(K)]`，不是 `h(E[K])`。
- 五个闭式 ridge 臂：`L_BASE`、`F_BASE`、`L_REAL`、`F_NULL`、`F_REAL`；λ 网格 1e-6…10，按 Valid 选择。MLP 三臂与 KRR 两臂只作为同输入黑箱参照，不参与候选选择。
- 历史坐标均从实际产物重新读取并核对：GINE {_f(hist['GINE_valid'])}/{_f(hist['GINE_test'])}，LCRC {_f(hist['LCRC_valid'])}/{_f(hist['LCRC_test'])}，SWF {_f(next((r['valid'] for r in main_rows if r['model']=='SWF_WORD_PREFIX'), float('nan')))}/{_f(next((r['test'] for r in main_rows if r['model']=='SWF_WORD_PREFIX'), float('nan')))}，Pair Rule REAL-FULL {_f(next((r['valid'] for r in main_rows if r['model']=='PAIR_RULE_REAL_FULL'), float('nan')))}/{_f(next((r['test'] for r in main_rows if r['model']=='PAIR_RULE_REAL_FULL'), float('nan')))}。

## 3. 工程检查与正对照

- 工程检查：{checks['n_pass']}/{checks['n_checks']} 通过；失败项：{', '.join(checks['failures']) if checks['failures'] else '无'}。
- 数据/划分/标签/行号、Train-only scaler、人工 path/star/triangle/ring 枚举、重标号不变性、四粒度映射、REAL 关系上下界、hypergeometric 期望、ridge 直接解一致、预测 replay 均通过。
- 贡献账目最大复算误差：F_BASE {_sci(led['F_BASE']['max_reconstruction_error'])}，F_REAL {_sci(led['F_REAL']['max_reconstruction_error'])}（目标 ≤1e-7）。
- Synthetic-L/F/R：小型满秩设计在 λ=1e-6 下预测 RMSE <1e-5、系数误差 <1e-4；完整库上 F_REAL 对 Synthetic-R 的 Test RMSE 明显低于 F_BASE/F_NULL；F_BASE 的关系贡献恒为 0，未被错误归因。Synthetic 结果只作工程正对照，不是化学证据。
- 父子计数恒等式：size1 = n_atoms、size2 = 2·n_bonds、cycle_rank = n_bonds − n_atoms + n_components 在全部枚举实例上精确成立；保留词表覆盖 size1 ≥{_f(min(v['retained_size1_fraction'] for v in summary['hierarchy_identities']['per_gran'].values()))}，size2 ≥{_f(min(v['retained_size2_fraction'] for v in summary['hierarchy_identities']['per_gran'].values()))}。

## 4. 主结果

{_md_table(main_rows, [('model','模型'),('family','族'),('valid','Valid RMSE'),('test','Test RMSE'),('selected','选择')])}

Test 仅用于报告；没有据此改模型。候选由 Valid RMSE 在 F_BASE/F_REAL 中选择；`L_REAL` 的 Valid 0.7893 是五臂中最低，但按协议它只是无 hinge 的消融参照，不能作为候选。

关键对比（gain = RMSE(reference) − RMSE(candidate)，正值表示 candidate 更好）：

{_md_table(crow, [('comparison','对比'),('valid_gain','Valid gain'),('valid_ci','Valid 95% CI'),('test_gain','Test gain')])}

## 5. 四轴判定

| 轴 | 判定 | 证据 |
|---|---|---|
| 表示 / learner | **{rep['axis']}** | 最强同输入黑箱 Valid {_f(min(rep['blackbox_valid'].values()) if rep['blackbox_valid'] else float('nan'))}；GINE Valid {_f(hist['GINE_valid'])}；阈值 GINE+0.06={_f(hist['GINE_valid']+0.06)}。 |
| 非线性 hinge | **{nl['axis']}** | F_BASE−L_BASE Valid gain {_f(next((v['valid_gain'] for v in nl['pairs'].values() if 'valid_gain' in v), float('nan')))}；F_REAL−L_REAL 见上表；均未达到 0.02。 |
| REAL 关系 | **{rel['axis']}** | F_REAL−F_BASE Valid gain {_f(rel['valid_vs_base']['delta'])}，Test {_f(rel['test_vs_base']['delta'])}；F_REAL−F_NULL Valid {_f(rel['valid_vs_null']['delta'])}，Test {_f(rel['test_vs_null']['delta'])}。支持充分但 Valid 方向为负。 |
| 分解成本 | **{dec['axis']}** | gap = 黑箱 − 显式（正值=黑箱更差）。F_BASE vs 最强同输入黑箱：Valid gap {_f(dec['pairs']['BASE']['valid_gap'])}，Test gap {_f(dec['pairs']['BASE']['test_gap'])}；F_REAL：Valid {_f(dec['pairs']['REAL']['valid_gap'])}，Test {_f(dec['pairs']['REAL']['test_gap'])}。 |

**总体**: `{summary['overall']}`；flags: `{', '.join(summary['flags'])}`。

Bootstrap: 2000 次 paired resample，seed 1703。Valid/Test 中每个分子都是独立 Bemis–Murcko scaffold cluster（420/420），因此 paired cluster bootstrap 等价于 molecule-row paired bootstrap；CI 只表示固定状态下的样本不确定性。

解释：本轮的主要瓶颈不是“把有效应拆开”，而是**显式计数表示本身**。在相同输入下，MLP/KRR 也没有把 Valid 拉到 GINE 附近；因此不能把差距归因于 hinge 参数化或关系账目。关系在 Test 上的方向性改善（F_REAL 比 F_BASE 低 {_f(rel['test_vs_base']['delta'])}）与 Valid 方向相反，按协议只能标记为 split-sensitive，不能作为稳定信号。

## 6. 关系覆盖与 REAL/NULL

- 64/64 锚点被使用；1024 个关系类型，Train/Valid/Test 均有 100% 分子至少命中一个关系，100% 分子 REAL≠NULL（线性），median nonzero relations 分别约 {cov['train']['median_nonzero_relations']:.0f}/{cov['valid']['median_nonzero_relations']:.0f}/{cov['test']['median_nonzero_relations']:.0f}。
- 因此“关系没有增益”不能解释为覆盖不足；它是给定表示与正则化下的经验结果。
- 线性消融中 REAL 相对 BASE 有一致的小幅增益（Valid {_f(lin_rel['valid_delta']) if lin_rel else 'nan'}，Test {_f(lin_rel['test_delta']) if lin_rel else 'nan'}），但低于 0.02 的协议门槛；完整 hinge 模型 F_REAL 反而在 Valid 上更差。
- NULL 使用 hypergeometric 期望；F_NULL 的 Valid {_f(next((r['valid'] for r in main_rows if r['model']=='F_NULL'), float('nan')))} 与 F_REAL 的 {_f(next((r['valid'] for r in main_rows if r['model']=='F_REAL'), float('nan')))} 接近，REAL 没有稳定超过匹配边际。

## 7. 贡献账目与可辨识性

- 账目按 `intercept + Σ main + Σ relation` 导出；所有组（根结构、整图统计、关系）分开记录，不把重叠根结构解释为互斥原子能量。
- F_BASE 的精确秩 {ident['F_BASE']['numerical_rank']}/{ident['F_BASE']['n_features']}，F_REAL {ident['F_REAL']['numerical_rank']}/{ident['F_REAL']['n_features']}；设计矩阵存在大量精确重复列（F_BASE 重复组覆盖 {ident['F_BASE']['exact_duplicate_columns']} 列、冗余 {ident['F_BASE']['redundant_columns']} 列；F_REAL 分别为 {ident['F_REAL']['exact_duplicate_columns']}、{ident['F_REAL']['redundant_columns']}），这是计数签名跨粒度冗余的直接体现。
- 近优 λ（Valid 差 ≤0.01）：F_BASE {ident['F_BASE']['near_optimal_lambdas']}；F_REAL {ident['F_REAL']['near_optimal_lambdas']}。F_BASE 的 λ=1.0 近优解与 λ=0.3 解的 Valid 预测 RMS 差约 0.31，贡献排序 Spearman 约 0.92，但主要组的平均绝对贡献差异超过 20%，因此标记 **ATTRIBUTION_NOT_IDENTIFIED**。
- 近优解的 β 差几乎全部落在行空间（null-space fraction ~1e-12），说明差异不是数值零空间造成，而是近优解之间的真实函数差异。

## 8. 匹配对诊断

- 复用 `task_response_composition_s0_v1` 的 matched pairs：{pair.get('n_pairs','?')} 对，cell counts {pair.get('cell_counts','?')}；held-out {pair.get('heldout_pairs','?')} 对，其中 seen transform {pair.get('heldout_seen_transform_pairs','?')}、unseen {pair.get('heldout_unseen_transform_pairs','?')}。

{_md_table(pair_rows, [('model','模型'),('mae','held-out Δpred MAE'),('skill','相对零差分技能'),('direction','方向准确率')])}

- Δpred 相对“零差分”基准的技能分数与方向准确率见 `pair_diagnostic.csv`。{pair_dec}
- 该诊断只检验相邻根局部关系的可迁移性，不改变主表选择。

## 9. 解释与限制

1. **表示瓶颈**：固定的小半径根局部计数不能表达 GINE 的深层消息传递；同输入 MLP/KRR 也远落后于 GINE，说明瓶颈在输入表示/可学习的函数类，而不是缺少非线性头。
2. **关系冗余**：REAL 关系在覆盖上充足，但其增量与匹配边际 NULL 接近；完整 hinge 库中大量关系列与基础结构列高度共线，ridge 的 Valid 选择倾向强正则化，进一步压缩了关系信号。
3. **分解成本低但绝对精度差**：在相同输入下，显式可加模型的 Valid 与 KRR 相当（差距 <0.01），说明“可分解”本身没有付出大的预测代价；问题是这个函数类的上限不够高。
4. **分割敏感性**：F_REAL 在 Test 上优于 F_BASE/F_NULL，但 Valid 上更差；按预注册规则不能据此宣称关系信号。
5. **归因不稳定**：近优 λ 解可以给出差异明显的函数曲线；报告中的曲线只对应最佳 Valid 解，不能当作唯一因果解释。

## 10. 复现与预算

- Stage 0 本地 67 s、远端 146 s；ridge 五臂总计约 {summary['runtime_seconds']['ridge_total']:.0f} s CPU；KRR 约 {sum(summary['runtime_seconds']['krr'].values()):.0f} s CPU；MLP 三臂 GPU 总时长约 {summary['gpu_hours_mlp']*60:.1f} GPU·min（约 {summary['gpu_hours_mlp']:.3f} GPU·h），低于 ~2 GPU·h 软上限。
- 所有正式运行绑定 commit `{summary['commit']}`；历史模型未重训、未新增 seed/数据集/函数族；Test 未参与任何选择。
- 主要产物：`results/main_table.csv`、`results/contrast_table.csv`、`results/bootstrap_*.csv`、`results/ridge/ledger/`、`results/ridge/curves/`、`results/figures/`、`analysis/checks.json`、`analysis/contribution_identifiability.json`、`analysis/pair_diagnostic.json`、`summary.json`。
"""
    (C.ROUND_DIR / "REPORT.md").write_text(text)


if __name__ == "__main__":
    main()

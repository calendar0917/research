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
            "params": 0, "source": "results/krr",
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
    models = load_models(stage, payload)
    table = main_table(models, stage)
    contrasts, cdata = run_contrasts(models, stage, payload)
    print(json.dumps({"models": sorted(models), "contrasts": len(contrasts)}, indent=1))
    # placeholder; subsequent stages fill this in
    print(json.dumps({"candidate_ridge": {
        a: float(next(r["valid_rmse"] for r in table if r["model"] == a))
        for a in ("F_BASE", "F_REAL") if any(r["model"] == a for r in table)}}, indent=1))


if __name__ == "__main__":
    main()


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

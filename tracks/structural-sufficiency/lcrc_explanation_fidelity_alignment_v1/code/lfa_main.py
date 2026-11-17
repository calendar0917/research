"""Main aggregation — WC_LOGP capture, fidelity advantage and decisions (CPU).

Reads the shared pool file, the two select files and the frozen WC_LOGP
contribution table.  Never re-selects a candidate, never changes a dose.
"""

from __future__ import annotations

import argparse
import itertools
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import lfa_common as C
import lfa_data as D
import lfa_gates as G

METHOD_MODEL = {
    "A_LCRC_GRAD": "LCRC", "B_LCRC_LOCAL": "LCRC", "C_LCRC_MASK50": "LCRC",
    "D_TOPOLOGY": "BOTH", "E_GINE_GRAD": "GINE", "F_GINE_MASK50": "GINE",
}
SCOPE_COUNTS = {"valid": 420, "test": 420, "train_description": 128}


# ---------------------------------------------------------------------------
# inputs
# ---------------------------------------------------------------------------


def load_jsonl_map(path: Path | str, key_fields: Sequence[str] = ("split", "cache_row")) -> dict[tuple, dict]:
    out = {}
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = tuple(int(rec[f]) if isinstance(rec[f], int) else str(rec[f]) for f in key_fields)
            out[key] = rec
    return out


def _bond_atoms(cache, cache_row: int, bonds: Sequence[int]) -> list[int]:
    ptr = np.asarray(cache["bond_ptr"], dtype=np.int64)
    idx = np.asarray(cache["bond_index"], dtype=np.int64)[int(ptr[cache_row]):int(ptr[cache_row + 1])]
    ids = [int(b) for b in bonds]
    return sorted({int(v) for b in ids for v in idx[b].tolist()})


def _contracts_equal(a: Sequence[Sequence[int]], b: Sequence[Sequence[int]]) -> bool:
    return [[int(x) for x in rec] for rec in a] == [[int(x) for x in rec] for rec in b]


def build_rows(*, pool_path: Path | str, select_lcrc: Path | str, select_gine: Path | str) -> tuple[list[dict], dict]:
    pool = load_jsonl_map(pool_path)
    lcrc = load_jsonl_map(select_lcrc)
    gine = load_jsonl_map(select_gine)
    if set(pool) != set(lcrc) or set(pool) != set(gine):
        raise RuntimeError("pool and select files cover different molecules")
    b = D.bundle()
    payload = D.oracle_payload()
    ptr = np.asarray(payload["atom_ptr"], dtype=np.int64)
    atom = np.asarray(payload["WC_LOGP_atom"], dtype=np.float64)
    rows: list[dict] = []
    checks = {"n_molecules": 0, "n_missing_oracle": 0, "n_missing_controls": 0,
              "d_control_identical": True, "n_d_contract_mismatch": 0}
    for key in sorted(pool):
        split, cache_row = key
        prec, lrec, grec = pool[key], lcrc[key], gine[key]
        stratum = "train_description" if split == "train" else split
        row_id = int(prec["row_id"])
        a0, a1 = int(ptr[row_id]), int(ptr[row_id + 1])
        contrib = np.array(atom[a0:a1], dtype=np.float64)
        denom = float(np.abs(contrib).sum())
        cache = b.caches[split]
        checks["n_molecules"] += 1
        if not (denom > 0.0):
            checks["n_missing_oracle"] += 1
        for method in C.SELECTORS:
            models = ["LCRC", "GINE"] if METHOD_MODEL[method] == "BOTH" else [METHOD_MODEL[method]]
            if method in ("C_LCRC_MASK50", "F_GINE_MASK50"):
                sel_rec = lrec if method.startswith("C_") else grec
                top = sel_rec.get("mask50_selection")
                control_bonds = sel_rec.get("mask50_control_bonds", [])
                control_status = sel_rec.get("mask50_control_status")
                n_controls = int(sel_rec.get("mask50_n_controls", 0))
                hit_own = sel_rec.get("mask50_hit_own_top1")
                mask50_n_scanned = int(sel_rec.get("mask50_n_scanned", 0))
                s2 = (sel_rec.get("n_s2") or {}).get(method)
            else:
                top = prec["selections"][method]
                control_bonds = prec["controls"][method].get("controls", [])
                control_bonds = [rec["bonds"] for rec in control_bonds]
                control_status = prec["controls"][method].get("status")
                n_controls = int(prec["controls"][method].get("n_random", 0))
                hit_own = prec["hit_own_top1"][method]
                mask50_n_scanned = 0
                s2 = prec["n_s2"][method]
            if method == "D_TOPOLOGY":
                a_ctrl = [[int(x) for x in rec] for rec in lrec["methods"][method].get("control_bonds", [])]
                b_ctrl = [[int(x) for x in rec] for rec in grec["methods"][method].get("control_bonds", [])]
                if not _contracts_equal(a_ctrl, b_ctrl):
                    checks["n_d_contract_mismatch"] += 1
                    checks["d_control_identical"] = False
            for model in models:
                rec = lrec if model == "LCRC" else grec
                fit = rec["methods"].get(method, {})
                if top is None:
                    rows.append({"split": split, "stratum": stratum, "cache_row": int(cache_row),
                                 "row_id": row_id, "method": method, "fidelity_model": model,
                                 "status": "NO_CANDIDATE"})
                    continue
                bonds = [int(v) for v in top["bonds"]]
                atoms = _bond_atoms(cache, cache_row, bonds)
                top_capture = G.capture(contrib, atoms) if denom > 0 else None
                ctrl_captures = [G.capture(contrib, _bond_atoms(cache, cache_row, cb)) for cb in control_bonds] if denom > 0 else []
                ctrl_captures = [v for v in ctrl_captures if v is not None]
                med_ctrl_capture = float(np.median(ctrl_captures)) if ctrl_captures else None
                enrichment = (float(top_capture - med_ctrl_capture)
                              if (top_capture is not None and med_ctrl_capture is not None) else None)
                row = {
                    "split": split, "stratum": stratum, "cache_row": int(cache_row), "row_id": row_id,
                    "method": method, "fidelity_model": model, "status": "OK",
                    "k": int(len(bonds)), "n_atoms": int(len(atoms)),
                    "bonds": json.dumps(bonds),
                    "score": fit.get("score"),
                    "mask50_score": fit.get("mask50_score"),
                    "hit_own_top1": hit_own,
                    "mask50_n_scanned": mask50_n_scanned,
                    "capture_top1": top_capture,
                    "capture_random_median": med_ctrl_capture,
                    "capture_random_mean": float(np.mean(ctrl_captures)) if ctrl_captures else None,
                    "oracle_enrichment": enrichment,
                    "oracle_defined": bool(denom > 0),
                    "n_controls": int(len(ctrl_captures)),
                    "control_status": control_status,
                    "controls_missing": bool(len(ctrl_captures) < C.MIN_RANDOM),
                    "absolute_contrib_mass": denom,
                    "n_s2_internal": None if not s2 else int(s2["n_s2_internal"]),
                    "n_s2_boundary": None if not s2 else int(s2["n_s2_boundary"]),
                    "n_s2_affected": None if not s2 else int(s2["n_s2_affected"]),
                    "pred_zero": fit.get("pred_zero"),
                    "pred_half": fit.get("pred_half"),
                    "delta0": fit.get("delta0"),
                    "abs_delta0": fit.get("abs_delta0"),
                    "abs_delta0_std": fit.get("abs_delta0_std"),
                    "abs_delta_half": fit.get("abs_delta_half"),
                    "half_over_full": fit.get("half_over_full"),
                    "sign_consistent": fit.get("sign_consistent"),
                    "delta_local": fit.get("delta_local"),
                    "delta_rho": fit.get("delta_rho"),
                    "closure_error": fit.get("closure_error"),
                    "random_median_abs_delta0": fit.get("random_median_abs_delta0"),
                    "random_median_abs_delta0_std": fit.get("random_median_abs_delta0_std"),
                    "fidelity_advantage": fit.get("fidelity_advantage"),
                    "fidelity_advantage_std": fit.get("fidelity_advantage_std"),
                    "target_std": fit.get("target_std"),
                    "fidelity_missing": fit.get("fidelity_advantage_std") is None,
                    "n_fidelity_controls": len(fit.get("control_abs_delta0") or []),
                }
                rows.append(row)
    return rows, checks


# ---------------------------------------------------------------------------
# summaries
# ---------------------------------------------------------------------------


def _arr(rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([np.nan if r.get(key) is None else float(r[key]) for r in rows], dtype=np.float64)


def _support(n_ok: int, n_total: int) -> tuple[float | None, str]:
    frac = (n_ok / n_total) if n_total else None
    if not n_total or frac is None:
        return None, "NO_SUPPORT"
    return float(frac), ("OK" if frac >= C.MIN_EVALUABLE_FRACTION else "INSUFFICIENT_SUPPORT")


def in_split(row: Mapping[str, Any], split: str) -> bool:
    """``train_description`` is a stratum of the official Train split."""
    if split == "train_description":
        return str(row.get("stratum")) == "train_description"
    return str(row.get("split")) == split


def method_summary(rows: Sequence[Mapping[str, Any]], *, split: str, n_scope: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for method in C.SELECTORS:
        models = ["LCRC", "GINE"] if METHOD_MODEL[method] == "BOTH" else [METHOD_MODEL[method]]
        for model in models:
            sub = [r for r in rows if in_split(r, split) and r["method"] == method
                   and r["fidelity_model"] == model and r.get("status") == "OK"]
            key = method if METHOD_MODEL[method] != "BOTH" else f"{method}@{model}"
            if not sub:
                out[key] = {"method": method, "fidelity_model": model, "n": 0, "support": "NO_SUPPORT"}
                continue
            enr = _arr(sub, "oracle_enrichment")
            adv = _arr(sub, "fidelity_advantage_std")
            cap = _arr(sub, "capture_top1")
            abs0 = _arr(sub, "abs_delta0_std")
            half = _arr(sub, "half_over_full")
            mask_oracle = np.isfinite(enr)
            mask_fid = np.isfinite(adv)
            frac_oracle, support_oracle = _support(int(mask_oracle.sum()), n_scope)
            frac_fid, support_fid = _support(int(mask_fid.sum()), n_scope)
            boot_oracle = C.bootstrap_columns({"enrichment": enr, "capture": cap}, mask_oracle)
            boot_fid = C.bootstrap_columns({"advantage_std": adv, "abs_delta0_std": abs0}, mask_fid)
            conc_enr = C.positive_concentration(enr[mask_oracle].tolist())
            conc_adv = C.positive_concentration(adv[mask_fid].tolist())
            both = np.isfinite(enr) & np.isfinite(adv)
            both_ok = int(np.sum((enr > 0) & (adv > 0) & both))
            only_oracle = int(np.sum((enr > 0) & ~(adv > 0) & both))
            only_fid = int(np.sum((adv > 0) & ~(enr > 0) & both))
            neither = int(np.sum(~(enr > 0) & ~(adv > 0) & both))
            out[key] = {
                "method": method, "fidelity_model": model, "n": len(sub), "n_scope": n_scope,
                "n_oracle_evaluable": int(mask_oracle.sum()),
                "n_fidelity_evaluable": int(mask_fid.sum()),
                "oracle_evaluable_fraction": frac_oracle, "oracle_support": support_oracle,
                "fidelity_evaluable_fraction": frac_fid, "fidelity_support": support_fid,
                "n_controls_missing_oracle": int(np.sum([bool(r.get("controls_missing")) for r in sub])),
                "n_fidelity_missing": int(np.sum([bool(r.get("fidelity_missing")) for r in sub])),
                "oracle_capture": (boot_oracle["point"].get("capture") or {}).get("mean"),
                "oracle_enrichment": (boot_oracle["point"].get("enrichment") or {}).get("mean"),
                "oracle_enrichment_ci": boot_oracle["ci95"]["enrichment"],
                "oracle_capture_ci": boot_oracle["ci95"]["capture"],
                "oracle_enrichment_quantiles": C.quantiles(enr[mask_oracle].tolist()),
                "oracle_capture_quantiles": C.quantiles(cap[mask_oracle].tolist()),
                "enrichment_gt0_fraction": float(np.mean(enr[mask_oracle] > 0)) if mask_oracle.any() else None,
                "enrichment_eq0_fraction": float(np.mean(enr[mask_oracle] == 0)) if mask_oracle.any() else None,
                "enrichment_lt0_fraction": float(np.mean(enr[mask_oracle] < 0)) if mask_oracle.any() else None,
                "enrichment_concentration": conc_enr,
                "fidelity_advantage_std": (boot_fid["point"].get("advantage_std") or {}).get("mean"),
                "fidelity_advantage_std_ci": boot_fid["ci95"]["advantage_std"],
                "abs_delta0_std": (boot_fid["point"].get("abs_delta0_std") or {}).get("mean"),
                "abs_delta0_std_ci": boot_fid["ci95"]["abs_delta0_std"],
                "fidelity_advantage_quantiles": C.quantiles(adv[mask_fid].tolist()),
                "advantage_gt0_fraction": float(np.mean(adv[mask_fid] > 0)) if mask_fid.any() else None,
                "advantage_eq0_fraction": float(np.mean(adv[mask_fid] == 0)) if mask_fid.any() else None,
                "advantage_concentration": conc_adv,
                "half_over_full_median": float(np.nanmedian(half)) if np.isfinite(half).any() else None,
                "sign_consistent_fraction": float(np.mean([bool(r.get("sign_consistent")) for r in sub])) if sub else None,
                "n_both": int(both.sum()),
                "both_positive": both_ok, "only_oracle": only_oracle,
                "only_fidelity": only_fid, "neither": neither,
                "both_positive_fraction": (both_ok / int(both.sum())) if int(both.sum()) else None,
                "median_k": float(np.median([r["k"] for r in sub])) if sub else None,
                "mean_k": float(np.mean([r["k"] for r in sub])) if sub else None,
                "median_mask50_n_scanned": float(np.median([int(r["mask50_n_scanned"]) for r in sub
                                                            if int(r.get("mask50_n_scanned") or 0) > 0])) if any(
                    int(r.get("mask50_n_scanned") or 0) > 0 for r in sub) else None,
                "median_n_atoms": float(np.median([r["n_atoms"] for r in sub])) if sub else None,
                "median_n_s2_affected": float(np.median([r["n_s2_affected"] for r in sub if r.get("n_s2_affected") is not None])) if sub else None,
                "hit_own_top1_fraction": float(np.mean([bool(r["hit_own_top1"]) for r in sub if r.get("hit_own_top1") is not None])) if sub else None,
                "median_half_over_full": float(np.nanmedian(half)) if np.isfinite(half).any() else None,
            }
    return out


def stratified_by_k(rows: Sequence[Mapping[str, Any]], *, split: str) -> list[dict[str, Any]]:
    out = []
    for method in C.SELECTORS:
        models = ["LCRC", "GINE"] if METHOD_MODEL[method] == "BOTH" else [METHOD_MODEL[method]]
        for model in models:
            sub = [r for r in rows if in_split(r, split) and r["method"] == method
                   and r["fidelity_model"] == model and r.get("status") == "OK"]
            key = method if METHOD_MODEL[method] != "BOTH" else f"{method}@{model}"
            for k in range(C.SEARCH_MIN_K, C.SEARCH_MAX_K + 1):
                kk = [r for r in sub if int(r["k"]) == k]
                enr = _arr(kk, "oracle_enrichment")
                adv = _arr(kk, "fidelity_advantage_std")
                out.append({
                    "split": split, "method": key, "k": k, "n": len(kk),
                    "oracle_evaluable": int(np.isfinite(enr).sum()),
                    "oracle_enrichment_mean": float(np.nanmean(enr)) if np.isfinite(enr).any() else None,
                    "oracle_enrichment_median": float(np.nanmedian(enr)) if np.isfinite(enr).any() else None,
                    "oracle_gt0_fraction": float(np.mean(enr[np.isfinite(enr)] > 0)) if np.isfinite(enr).any() else None,
                    "fidelity_evaluable": int(np.isfinite(adv).sum()),
                    "fidelity_advantage_std_mean": float(np.nanmean(adv)) if np.isfinite(adv).any() else None,
                    "fidelity_advantage_std_median": float(np.nanmedian(adv)) if np.isfinite(adv).any() else None,
                    "fidelity_gt0_fraction": float(np.mean(adv[np.isfinite(adv)] > 0)) if np.isfinite(adv).any() else None,
                })
    return out


def paired_differences(rows: Sequence[Mapping[str, Any]], *, split: str) -> list[dict[str, Any]]:
    out = []
    labels = [m if METHOD_MODEL[m] != "BOTH" else f"{m}@LCRC" for m in C.SELECTORS]
    labels += [f"D_TOPOLOGY@GINE"]
    cols = ["oracle_enrichment"]
    for a, b in itertools.combinations(labels, 2):
        def model_of(key: str) -> str:
            return key.split("@")[1] if "@" in key else METHOD_MODEL[key]
        def method_of(key: str) -> str:
            return key.split("@")[0]
        sub_a = {int(r["cache_row"]): r for r in rows if in_split(r, split)
                 and r["method"] == method_of(a) and r["fidelity_model"] == model_of(a)
                 and r.get("status") == "OK"}
        sub_b = {int(r["cache_row"]): r for r in rows if in_split(r, split)
                 and r["method"] == method_of(b) and r["fidelity_model"] == model_of(b)
                 and r.get("status") == "OK"}
        common = sorted(set(sub_a) & set(sub_b))
        if not common:
            continue
        for col in cols:
            va = np.asarray([float(sub_a[i][col]) if sub_a[i].get(col) is not None else np.nan for i in common])
            vb = np.asarray([float(sub_b[i][col]) if sub_b[i].get(col) is not None else np.nan for i in common])
            diff = va - vb
            mask = np.isfinite(diff)
            boot = C.bootstrap_columns({"diff": diff}, mask)
            same_family = model_of(a) == model_of(b) or "@" in a or "@" in b
            out.append({
                "split": split, "metric": col, "method_a": a, "method_b": b,
                "n_pairs": int(mask.sum()),
                "mean_a": float(np.nanmean(va)) if np.isfinite(va).any() else None,
                "mean_b": float(np.nanmean(vb)) if np.isfinite(vb).any() else None,
                "mean_diff": boot["point"]["diff"]["mean"],
                "median_diff": boot["point"]["diff"]["median"],
                "ci95_mean": boot["ci95"]["diff"]["mean"],
                "ci95_low": (boot["ci95"]["diff"]["mean"] or [None, None])[0],
                "ci95_high": (boot["ci95"]["diff"]["mean"] or [None, None])[1],
                "gt0_fraction": float(np.mean(diff[mask] > 0)) if mask.any() else None,
                "same_fidelity_family": bool(model_of(a) == model_of(b)),
            })
    return out


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def _supported(entry: Mapping[str, Any], metric: str, floor: float) -> bool:
    if not entry or entry.get("n", 0) == 0:
        return False
    if metric.startswith("oracle"):
        point = entry.get("oracle_enrichment")
        ci = (entry.get("oracle_enrichment_ci") or {}).get("mean")
        conc = (entry.get("enrichment_concentration") or {}).get("top_share")
        support = entry.get("oracle_support")
    else:
        point = entry.get("fidelity_advantage_std")
        ci = (entry.get("fidelity_advantage_std_ci") or {}).get("mean")
        conc = (entry.get("advantage_concentration") or {}).get("top_share")
        support = entry.get("fidelity_support")
    if point is None or ci is None or ci[0] is None:
        return False
    return bool(ci[0] > 0.0 and point >= floor and support == "OK"
                and (conc is None or conc <= C.CONCENTRATION_MAX_SHARE))


def pair_diff(rows: Sequence[Mapping[str, Any]], a: str, b: str) -> dict[str, Any] | None:
    """Oriented paired difference ``a - b`` from the frozen pairwise table."""
    for row in rows:
        if {row.get("method_a"), row.get("method_b")} != {a, b}:
            continue
        if row.get("method_a") == a:
            return dict(row)
        flipped = dict(row)
        flipped["method_a"], flipped["method_b"] = a, b
        if row.get("mean_diff") is not None:
            flipped["mean_diff"] = -float(row["mean_diff"])
        if row.get("median_diff") is not None:
            flipped["median_diff"] = -float(row["median_diff"])
        ci = row.get("ci95_mean")
        if ci and ci[0] is not None:
            flipped["ci95_mean"] = [-float(ci[1]), -float(ci[0])]
        if row.get("ci95_low") is not None and row.get("ci95_high") is not None:
            flipped["ci95_low"] = -float(row["ci95_high"])
            flipped["ci95_high"] = -float(row["ci95_low"])
        if row.get("gt0_fraction") is not None:
            flipped["gt0_fraction"] = 1.0 - float(row["gt0_fraction"])
        return flipped
    return None


def decide(summary: Mapping[str, Mapping[str, Any]], paired: Sequence[Mapping[str, Any]],
           checks: Mapping[str, Any]) -> dict[str, Any]:
    valid = summary.get("valid", {})
    test = summary.get("test", {})
    o_sup = {key: _supported(valid.get(key, {}), "oracle", C.ORACLE_ENRICHMENT_MIN) for key in valid}
    f_sup = {key: _supported(valid.get(key, {}), "fidelity", C.FIDELITY_ADVANTAGE_STD_MIN) for key in valid}
    test_o = (test.get("B_LCRC_LOCAL") or {}).get("oracle_enrichment")
    test_f = (test.get("B_LCRC_LOCAL") or {}).get("fidelity_advantage_std")
    valid_b_o = (valid.get("B_LCRC_LOCAL") or {}).get("oracle_enrichment")
    valid_b_f = (valid.get("B_LCRC_LOCAL") or {}).get("fidelity_advantage_std")
    test_sign_ok = bool(
        test_o is not None and valid_b_o is not None and np.sign(test_o) == np.sign(valid_b_o)
        and test_f is not None and valid_b_f is not None and np.sign(test_f) == np.sign(valid_b_f)
    )
    labels: list[str] = []
    reasons: list[str] = []
    support_ok = all((valid.get(k) or {}).get("oracle_support") == "OK"
                     for k in ("A_LCRC_GRAD", "B_LCRC_LOCAL", "C_LCRC_MASK50",
                               "E_GINE_GRAD", "F_GINE_MASK50", "D_TOPOLOGY@LCRC"))
    if o_sup.get("B_LCRC_LOCAL") and f_sup.get("B_LCRC_LOCAL"):
        if test_sign_ok:
            labels.append("LOCAL_FAITHFUL_AND_RELEVANT")
        else:
            labels.append("UNRESOLVED")
            reasons.append("local_valid_supported_but_test_not_replicated")
    elif o_sup.get("B_LCRC_LOCAL") and not f_sup.get("B_LCRC_LOCAL"):
        if test_sign_ok:
            labels.append("LOCAL_RELEVANT_NOT_FAITHFUL")
        else:
            labels.append("UNRESOLVED")
            reasons.append("local_relevant_valid_only")
    else:
        labels.append("UNRESOLVED")
        reasons.append("local_oracle_not_supported")

    ca = pair_diff([row for row in paired if row.get("metric") == "oracle_enrichment"],
                   "C_LCRC_MASK50", "A_LCRC_GRAD")
    ca_ok = bool(ca and ca["ci95_mean"] and ca["ci95_mean"][0] is not None
                 and ca["ci95_mean"][0] > 0 and ca["mean_diff"] >= C.PAIRED_IMPROVEMENT_MIN)
    if ca_ok:
        labels.append("FINITE_MASK_IMPROVES_ALIGNMENT")
    test_ca = pair_diff([row for row in paired if row.get("split") == "test"
                         and row.get("metric") == "oracle_enrichment"],
                        "C_LCRC_MASK50", "A_LCRC_GRAD")
    if ca_ok and test_ca is not None and test_ca["mean_diff"] <= 0:
        reasons.append("finite_mask_valid_only")

    lcrc_best = max(("A_LCRC_GRAD", "B_LCRC_LOCAL", "C_LCRC_MASK50"),
                    key=lambda k: valid.get(k, {}).get("oracle_enrichment") or -1e9)
    gine_best = max(("E_GINE_GRAD", "F_GINE_MASK50"),
                    key=lambda k: valid.get(k, {}).get("oracle_enrichment") or -1e9)
    cross = pair_diff([row for row in paired if row.get("metric") == "oracle_enrichment"],
                      lcrc_best, gine_best)
    lrc_adv = bool(cross and cross["ci95_mean"] and cross["ci95_mean"][0] is not None
                   and cross["ci95_mean"][0] > 0 and cross["mean_diff"] >= C.PAIRED_IMPROVEMENT_MIN)
    if not lrc_adv and support_ok:
        labels.append("NO_LCRC_SPECIFIC_ADVANTAGE")
    if not support_ok:
        reasons.append("insufficient_support_for_at_least_one_method")
        if "UNRESOLVED" not in labels:
            labels.append("UNRESOLVED")
    return {
        "labels": labels,
        "reasons": reasons,
        "oracle_supported": o_sup,
        "fidelity_supported": f_sup,
        "test_replication_sign_ok": test_sign_ok,
        "finite_mask_paired_C_minus_A": ca,
        "finite_mask_supported": ca_ok,
        "lcrc_best_external": {"method": lcrc_best},
        "gine_best_external": {"method": gine_best},
        "cross_model_pair": cross,
        "lcrc_specific_advantage": lrc_adv,
        "checks": dict(checks),
        "thresholds": {
            "oracle_enrichment_min": C.ORACLE_ENRICHMENT_MIN,
            "fidelity_advantage_std_min": C.FIDELITY_ADVANTAGE_STD_MIN,
            "paired_improvement_min": C.PAIRED_IMPROVEMENT_MIN,
            "concentration_max_share": C.CONCENTRATION_MAX_SHARE,
            "min_evaluable_fraction": C.MIN_EVALUABLE_FRACTION,
        },
    }


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def run(*, pool_dir: Path | str = C.POOL_DIR, select_dir: Path | str = C.SELECT_DIR,
        out_dir: Path | str = C.ANALYSIS_DIR) -> dict[str, Any]:
    out_dir = Path(out_dir)
    C.ensure_dir(out_dir)
    pool_manifest = C.read_json(Path(pool_dir) / "pool_manifest.json")
    pool_path = Path(pool_dir) / "pool_main.jsonl"
    actual = C.sha256_file(pool_path)
    if actual != pool_manifest["reports"][C.SCOPE_MAIN]["sha256"]:
        raise RuntimeError("main pool sha256 does not match the manifest")
    rows, checks = build_rows(
        pool_path=pool_path,
        select_lcrc=Path(select_dir) / "select_lcrc_main.jsonl",
        select_gine=Path(select_dir) / "select_gine_main.jsonl",
    )
    columns = ["split", "stratum", "cache_row", "row_id", "method", "fidelity_model", "status",
               "k", "n_atoms", "bonds", "score", "mask50_score", "hit_own_top1", "mask50_n_scanned",
               "capture_top1", "capture_random_median", "capture_random_mean", "oracle_enrichment",
               "oracle_defined", "n_controls", "control_status", "controls_missing",
               "absolute_contrib_mass", "n_s2_internal", "n_s2_boundary", "n_s2_affected",
               "pred_zero", "pred_half", "delta0", "abs_delta0", "abs_delta0_std",
               "abs_delta_half", "half_over_full", "sign_consistent", "delta_local", "delta_rho",
               "closure_error", "random_median_abs_delta0", "random_median_abs_delta0_std",
               "fidelity_advantage", "fidelity_advantage_std", "target_std", "fidelity_missing",
               "n_fidelity_controls"]
    C.write_csv(out_dir / "main_molecule_results.csv", rows, columns)
    summary = {split: method_summary(rows, split=split, n_scope=SCOPE_COUNTS[split])
               for split in ("valid", "test", "train_description")}
    strata = {split: stratified_by_k(rows, split=split) for split in ("valid", "test", "train_description")}
    paired = {split: paired_differences(rows, split=split) for split in ("valid", "test", "train_description")}
    decision = decide(summary, paired["valid"], checks)
    payload = {
        "round_id": C.ROUND_ID,
        "n_rows": len(rows),
        "scope_counts": SCOPE_COUNTS,
        "summary": summary,
        "checks": checks,
        "pool_sha256": actual,
        "select_sha256": {
            "lcrc": C.sha256_file(Path(select_dir) / "select_lcrc_main.jsonl"),
            "gine": C.sha256_file(Path(select_dir) / "select_gine_main.jsonl"),
        },
        "created_utc": C.utc_now(),
    }
    C.write_json(out_dir / "main_summary.json", payload)
    C.write_csv(out_dir / "main_stratified_by_k.csv",
                [row for split in strata.values() for row in split])
    C.write_csv(out_dir / "main_paired_differences.csv",
                [row for split in paired.values() for row in split])
    concentration = []
    for split, entries in summary.items():
        for key, entry in entries.items():
            concentration.append({
                "split": split, "method": key, "n": entry.get("n"),
                "oracle_top5_share": (entry.get("enrichment_concentration") or {}).get("top_share"),
                "oracle_total_positive": (entry.get("enrichment_concentration") or {}).get("total_positive"),
                "oracle_positive": (entry.get("enrichment_concentration") or {}).get("n_positive"),
                "oracle_zero": (entry.get("enrichment_concentration") or {}).get("n_zero"),
                "fidelity_top5_share": (entry.get("advantage_concentration") or {}).get("top_share"),
                "fidelity_total_positive": (entry.get("advantage_concentration") or {}).get("total_positive"),
            })
    C.write_csv(out_dir / "main_concentration.csv", concentration)
    C.write_json(out_dir / "decision.json", decision)
    print(f"[main] rows={len(rows)} labels={decision['labels']}", flush=True)
    return {"rows": len(rows), "summary": summary, "decision": decision, "paired": paired, "stratified": strata}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--pool-dir", default=str(C.POOL_DIR))
    parser.add_argument("--select-dir", default=str(C.SELECT_DIR))
    parser.add_argument("--out-dir", default=str(C.ANALYSIS_DIR))
    args = parser.parse_args()
    out = run(pool_dir=args.pool_dir, select_dir=args.select_dir, out_dir=args.out_dir)
    print(json.dumps(out["decision"], indent=1)[:4000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

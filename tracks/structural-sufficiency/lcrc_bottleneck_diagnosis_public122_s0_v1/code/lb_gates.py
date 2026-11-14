"""Pre-registered gates and the route-verdict matrix.

Each gate prints the *numbers it used* and the threshold; every verdict carries an
evidence grade.  A gate that cannot be evaluated returns ``INCONCLUSIVE`` (never
``False``), because an unrun arm is not evidence of no effect.
"""

from __future__ import annotations

from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import lb_common as C


# ---------------------------------------------------------------------------
# shared helpers
# ---------------------------------------------------------------------------


def _gain(control: float | None, candidate: float | None, task: str) -> float | None:
    if control is None or candidate is None:
        return None
    if not (np.isfinite(control) and np.isfinite(candidate)):
        return None
    return float(control - candidate) if task == "regression" else float(candidate - control)


def _primary(entry: Mapping[str, Any], task: str) -> float | None:
    """Primary Test metric from either a rule record (``test_primary``) or a metrics dict."""
    for key in ("test_primary", "test_rmse", "test_auc"):
        v = entry.get(key)
        if v is not None:
            return float(v)
    return None


# ---------------------------------------------------------------------------
# 7.1 selection sensitivity
# ---------------------------------------------------------------------------


def selection_sensitivity(panel: Mapping[str, Any], sel_table: Mapping[str, Any]) -> dict[str, Any]:
    """SELECTION_SENSITIVITY: large observed oracle gap + real bootstrap/jackknife spread."""
    arms = panel.get("arms", {})
    rows = []
    gaps = []
    for a, v in arms.items():
        if v.get("status") == "NOT_RUN":
            rows.append({"arm": a, "status": "NOT_RUN"}); continue
        b = v["bootstrap"]
        spread = b.get("selected_update_spread")
        p10, p90 = b.get("test_metric_p10"), b.get("test_metric_p90")
        width = (p90 - p10) if (p10 is not None and p90 is not None) else None
        e1 = sel_table.get("arms", {}).get(a, {}).get("phases", {}).get("E1", {})
        gap = (e1.get("observed_oracle_gap") or {}).get("observed_oracle_gap")
        gaps.append(gap)
        rows.append({"arm": a, "observed_oracle_gap": gap, "bootstrap_p10_p90_width": width,
                     "selected_update_spread": spread})
    finite = [g for g in gaps if g is not None and np.isfinite(g)]
    judged = [r for r in rows if r.get("selected_update_spread") is not None]
    if not judged:
        return {"verdict": "INCONCLUSIVE", "reason": "no full-N panel results", "rows": rows}
    n_spread = sum(1 for r in judged if (r["selected_update_spread"] or 0) >= 3)
    frac = n_spread / len(judged)
    verdict = "SELECTION_SENSITIVITY" if frac >= 0.5 else "NO_CLEAR_AVOIDABLE_LIMIT"
    return {"verdict": verdict, "fraction_arms_with_spread_ge_3": frac,
            "median_observed_oracle_gap": float(np.median(finite)) if finite else None,
            "note": "observed gaps include Test-selection optimism; they are not achievable gains",
            "rows": rows}


def selector_utility(sel_table: Mapping[str, Any], cell_filter: str | None = None) -> dict[str, Any]:
    """SELECTOR_HAS_UTILITY: an alternative fixed rule beats R_SEL consistently across arms."""
    rows = []
    for arm_id, arm in sel_table.get("arms", {}).items():
        if not isinstance(arm, dict) or arm.get("status") == "NOT_RUN":
            continue
        if cell_filter and cell_filter not in arm_id:
            continue
        task = "regression" if arm.get("dataset") == C.PRIMARY_DATASET else "binary"
        ph = arm["phases"].get("E1")
        if not ph or ph.get("status") == "EMPTY":
            continue
        r = _primary(ph["rules"][C.R_SEL], task)
        row = {"arm": arm_id, "R_SEL": r}
        for alt in (C.A_SEL, C.W_SEL):
            row[f"{alt}_gain_vs_R_SEL"] = _gain(r, _primary(ph["rules"][alt], task), task)
        rows.append(row)
    if not rows:
        return {"verdict": "INCONCLUSIVE", "rows": []}
    out = {"rows": rows}
    for alt in (C.A_SEL, C.W_SEL):
        g = [r[f"{alt}_gain_vs_R_SEL"] for r in rows if r.get(f"{alt}_gain_vs_R_SEL") is not None]
        pos = sum(1 for x in g if x > 0)
        out[alt] = {"n": len(g), "n_positive": pos,
                    "median_gain": float(np.median(g)) if g else None}
    best_alt = max((C.A_SEL, C.W_SEL),
                   key=lambda a: (out[a]["n_positive"] / max(out[a]["n"], 1), out[a]["median_gain"] or -9e9))
    meta = out[best_alt]
    verdict = ("SELECTOR_HAS_UTILITY"
               if meta["n"] >= 4 and meta["n_positive"] >= 0.75 * meta["n"] and (meta["median_gain"] or 0) > 0
               else "NO_CLEAR_AVOIDABLE_LIMIT")
    out["verdict"] = verdict
    out["best_alternative"] = best_alt
    return out


# ---------------------------------------------------------------------------
# 7.2 optimization headroom
# ---------------------------------------------------------------------------


def optimization_headroom(data_budget: Mapping[str, Any]) -> dict[str, Any]:
    """OPTIMIZATION_HEADROOM vs FIT_ONLY, per arm, from the E1->E2 (and last10) records."""
    rows = []
    for cell, arms in data_budget.get("cells", {}).items():
        for r in arms:
            if r.get("status") != "OK" or "E2_R_SEL_test" not in r:
                continue
            task = "regression" if cell.endswith(C.PRIMARY_DATASET) else "binary"
            e1, e2 = r.get("E1_R_SEL_test"), r.get("E2_R_SEL_test")
            train_gain = _gain(r.get("E1_last10_train_fit_median"), r.get("E2_last10_train_fit_median"), task)
            test_gain = _gain(e1, e2, task)
            # a holdout gain only counts if it is bigger than the ruler
            ruler = C.RMSE_RULER if task == "regression" else C.AUC_RULER
            row = {"arm": r["arm_id"], "cell": cell,
                   "E1": e1, "E2": e2, "holdout_gain": test_gain,
                   "train_fit_gain": train_gain,
                   "last10_gain": _gain(r.get("E1_last10_test_median"), r.get("E2_last10_test_median"), task),
                   "ruler": ruler}
            if test_gain is not None and test_gain >= ruler:
                row["label"] = "OPTIMIZATION_HEADROOM"
            elif train_gain is not None and train_gain > 0:
                row["label"] = "FIT_ONLY"
            else:
                row["label"] = "NO_EFFECT"
            rows.append(row)
    if not rows:
        return {"verdict": "INCONCLUSIVE", "reason": "no arm ran the E2 leg", "rows": rows}
    labels = [r["label"] for r in rows]
    verdict = ("OPTIMIZATION_HEADROOM" if labels.count("OPTIMIZATION_HEADROOM") >= max(1, len(labels) // 2)
               else "FIT_ONLY" if labels.count("FIT_ONLY") >= max(1, len(labels) // 2)
               else "NO_CLEAR_AVOIDABLE_LIMIT")
    return {"verdict": verdict, "rows": rows,
            "note": "E1/E2 are the same continuous prefix; U0 is an observation budget, not a second run"}


# ---------------------------------------------------------------------------
# 7.3 sample efficiency
# ---------------------------------------------------------------------------


def sample_efficiency(data_budget: Mapping[str, Any]) -> dict[str, Any]:
    out: dict[str, Any] = {"cells": {}, "verdict": "INCONCLUSIVE"}
    any_gain = []
    for cell, arms in data_budget.get("cells", {}).items():
        task = "regression" if cell.endswith(C.PRIMARY_DATASET) else "binary"
        ruler = C.RMSE_RULER if task == "regression" else C.AUC_RULER
        ok = [r for r in arms if r.get("status") == "OK"]
        ok.sort(key=lambda r: r["N"])
        if len(ok) < 2:
            out["cells"][cell] = {"status": "INCONCLUSIVE", "reason": "needs >=2 fit sizes"}
            continue
        small, big = ok[0], ok[-1]
        per_rule = {}
        for rule in C.SELECTION_RULES:
            for phase in ("E1", "E2"):
                k = f"{phase}_{rule}_test"
                if k in small and k in big:
                    per_rule[f"{phase}_{rule}"] = _gain(small[k], big[k], task)
        vals = [v for v in per_rule.values() if v is not None]
        consistent = [v for v in vals if v >= ruler]
        out["cells"][cell] = {"N_small": small["N"], "N_big": big["N"],
                              "gains": per_rule, "ruler": ruler,
                              "n_rules_meeting_ruler": len(consistent), "n_comparisons": len(vals),
                              "verdict": ("SAMPLE_EFFICIENCY_LIMIT"
                                          if vals and len(consistent) >= max(1, int(np.ceil(0.6 * len(vals))))
                                          else "NO_CLEAR_EFFECT")}
        any_gain.extend(vals)
    cells = out["cells"]
    if cells and any(c.get("verdict") == "SAMPLE_EFFICIENCY_LIMIT" for c in cells.values()):
        out["verdict"] = "SAMPLE_EFFICIENCY_LIMIT"
    elif cells and all(c.get("verdict") == "NO_CLEAR_EFFECT" for c in cells.values()):
        out["verdict"] = "NO_CLEAR_EFFECT"
    return out


# ---------------------------------------------------------------------------
# 7.4 restart trigger
# ---------------------------------------------------------------------------


def restart_trigger(data_budget: Mapping[str, Any], arm_id: str = "LIP_C1_R1_F100") -> dict[str, Any]:
    arm = None
    for arms in data_budget.get("cells", {}).values():
        for r in arms:
            if r.get("arm_id") == arm_id and r.get("status") == "OK":
                arm = r
    if arm is None:
        return {"verdict": "CANNOT_EVALUATE", "reason": f"{arm_id} not run"}
    need = {"train_rmse_drop_min": C.RESTART_TRAIN_DROP, "test_gain_max": C.RESTART_TEST_GAIN}
    train_drop = arm.get("train_fit_drop_frac")
    test_gain = _gain(arm.get("E1_R_SEL_test"), arm.get("E2_R_SEL_test"), "regression")
    fire = (train_drop is not None and train_drop < need["train_rmse_drop_min"]
            and test_gain is not None and test_gain < need["test_gain_max"])
    return {"arm": arm_id, "rule": need, "train_rmse_drop_frac": train_drop,
            "test_gain_E1_to_E2": test_gain,
            "verdict": "FIRE_RESTART" if fire else "DO_NOT_RESTART",
            "note": "one restart max; failure is recorded as-is and never re-tuned"}


# ---------------------------------------------------------------------------
# 7.5 readout purchase
# ---------------------------------------------------------------------------


def readout_purchase(readout: Mapping[str, Any]) -> dict[str, Any]:
    lipo = readout.get("lipo", {})
    base = lipo.get("P_WARM", {})
    if base.get("status") != "OK" or not base.get("selected"):
        return {"verdict": "CANNOT_EVALUATE", "purchase_bace_probes": None}
    rows = []
    purchased = False
    for rule in ("P_POST", "P_PRE"):
        e = lipo.get(rule, {})
        if e.get("status") != "OK":
            rows.append({"probe": rule, "status": "NOT_RUN"}); continue
        g = e.get("gain_vs_warm_R_SEL", {})
        vgain, tgain = g.get("valid_rmse_gain"), g.get("test_rmse_gain")
        # purchase rule: >=1% relative on BOTH Valid and Test RMSE
        bv = base["selected"]["R_SEL"]["valid_rmse"]; bt = base["selected"]["R_SEL"]["test_rmse"]
        ok = (vgain is not None and tgain is not None
              and vgain / max(abs(bv), 1e-12) >= C.PROBE_MIN_RELATIVE_GAIN
              and tgain / max(abs(bt), 1e-12) >= C.PROBE_MIN_RELATIVE_GAIN)
        purchased = purchased or ok
        rows.append({"probe": rule, "valid_rmse_gain": vgain, "test_rmse_gain": tgain,
                     "relative_valid": None if vgain is None else vgain / max(abs(bv), 1e-12),
                     "relative_test": None if tgain is None else tgain / max(abs(bt), 1e-12),
                     "meets_purchase_rule": bool(ok)})
    verdict = "REPRESENTATION_USE_HEADROOM" if purchased else "READOUT_NO_PURCHASE"
    return {"verdict": verdict, "rows": rows, "purchase_bace_probes": bool(purchased)}


# ---------------------------------------------------------------------------
# final matrix
# ---------------------------------------------------------------------------


def verdict_matrix(synth: Mapping[str, Any]) -> dict[str, Any]:
    sel = selection_sensitivity(synth["selection"]["fulln_u0_panel"], synth["selection"])
    util = selector_utility(synth["selection"])
    opt = optimization_headroom(synth["data_budget"])
    data = sample_efficiency(synth["data_budget"])
    read = readout_purchase(synth["readout"])
    trig = restart_trigger(synth["data_budget"])

    labelled = {
        "H_SEL": {"evidence": sel["verdict"], "selector_utility": util["verdict"],
                  "grade": "direct" if sel["verdict"] == "SELECTION_SENSITIVITY" else "weak"},
        "H_OPT": {"evidence": opt["verdict"], "grade": "direct" if opt["verdict"] != "INCONCLUSIVE" else "none"},
        "H_DATA": {"evidence": data["verdict"], "grade": "direct" if data["verdict"] != "INCONCLUSIVE" else "none"},
        "H_USE": {"evidence": read["verdict"], "grade": "direct" if read["verdict"] != "CANNOT_EVALUATE" else "none"},
        "H_REP": {"evidence": "REPRESENTATION_DEFICIT_UNIDENTIFIED" if (
            sel["verdict"] != "SELECTION_SENSITIVITY" and opt["verdict"] in ("NO_EFFECT", "NO_CLEAR_AVOIDABLE_LIMIT")
            and data["verdict"] == "NO_CLEAR_EFFECT" and read["verdict"] == "READOUT_NO_PURCHASE")
            else "not_established", "grade": "suspected_only"},
    }
    drivers = [k for k, v in labelled.items() if k != "H_REP" and v["grade"] == "direct"]
    return {"restart_trigger": trig, "readout": read, "selection": sel, "selector_utility": util,
            "optimization": opt, "sample_efficiency": data, "labels": labelled,
            "primary_drivers": drivers,
            "caveat": ("Different intervention gains are not addable; low train error, block-deletion "
                       "sensitivity, finer signatures and a single failed probe each do not locate the bottleneck.")}


__all__ = ["selection_sensitivity", "selector_utility", "optimization_headroom", "sample_efficiency",
           "restart_trigger", "readout_purchase", "verdict_matrix"]

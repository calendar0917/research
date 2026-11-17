"""Aggregate the two formal workers into the round's data products.

All labels and oracle contributions are joined locally; the workers never used
them to choose a candidate.  Component bootstrap resamples shared-endpoint
connected components (seed 20261117, 2000 replicates).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

import xr_common as C
import xr_assets as A
import xr_models as M


# ---------------------------------------------------------------------------
# IO helpers
# ---------------------------------------------------------------------------


def _load_endpoint_records(path: Path) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    with open(path, "r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            rec = json.loads(line)
            key = str(rec["endpoint"])
            out[key] = rec
    return out


def _f(value: Any) -> float | None:
    if value is None:
        return None
    try:
        v = float(value)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _fmt(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, bool):
        return "1" if value else "0"
    if isinstance(value, (int, np.integer)):
        return str(int(value))
    if isinstance(value, (float, np.floating)):
        v = float(value)
        return "" if not math.isfinite(v) else f"{v:.10g}"
    return str(value)


def _write_csv(path: Path, rows: Sequence[Mapping[str, Any]], columns: Sequence[str]) -> dict[str, Any]:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(columns), extrasaction="ignore")
        writer.writeheader()
        for row in rows:
            writer.writerow({key: _fmt(row.get(key)) for key in columns})
    return {"path": C.relpath(path), "n": len(rows), "sha256": C.sha256_file(path)}


# ---------------------------------------------------------------------------
# stratum helpers
# ---------------------------------------------------------------------------


def stratum_of(pair: Mapping[str, Any]) -> str:
    sp = str(pair["split_pair"])
    if sp == "train-train":
        return "train_train"
    if sp == "train-valid":
        return "train_valid"
    if sp == "train-test":
        return "train_test"
    if pair["split_a"] != "train" and pair["split_b"] != "train":
        return "both_nontrain"
    return "at_least_one_nontrain"


STRATA = ("all", "train_train", "train_valid", "train_test", "at_least_one_nontrain", "both_nontrain")


def _subset(pairs: Sequence[Mapping[str, Any]], stratum: str) -> list[Mapping[str, Any]]:
    if stratum == "all":
        return list(pairs)
    return [p for p in pairs if stratum_of(p) == stratum]


def _support(components: np.ndarray, n: int) -> str:
    return C.support_label(int(np.unique(components).size), int(n))


def _mean_stat(values: Mapping[str, np.ndarray]) -> dict[str, float]:
    return {key: float(np.mean(np.asarray(val, dtype=np.float64))) for key, val in values.items()}


def _diff_stat(values: Mapping[str, np.ndarray]) -> dict[str, float]:
    l = np.asarray(values["l"], dtype=np.float64)
    g = np.asarray(values["g"], dtype=np.float64)
    return {"l": float(l.mean()), "g": float(g.mean()), "diff": float(l.mean() - g.mean())}


def _boot(
    values: Mapping[str, np.ndarray],
    components: np.ndarray,
    mask: np.ndarray,
    *,
    stat_fn: Callable[[Mapping[str, np.ndarray]], Mapping[str, float]] = _mean_stat,
    n_boot: int = C.BOOTSTRAP_REPLICATES,
) -> dict[str, Any]:
    return C.component_bootstrap(values, components, stat_fn, mask=mask, n_boot=n_boot, seed=C.BOOTSTRAP_SEED)


# ---------------------------------------------------------------------------
# pair/endpoint tables
# ---------------------------------------------------------------------------


def _sign(x: float, eps: float = 1e-12) -> int:
    if x > eps:
        return 1
    if x < -eps:
        return -1
    return 0


def _endpoint_prediction(rec: Mapping[str, Any] | None, model: str) -> float | None:
    if rec is None or rec.get("status") != "OK":
        return None
    return _f(rec.get("pred", {}).get(model, {}).get("pred_raw"))


def _endpoint_metric(rec: Mapping[str, Any] | None, model: str, path: Sequence[str]) -> Any:
    if rec is None or rec.get("status") != "OK":
        return None
    cur: Any = rec.get("localisation", {}).get(model)
    for key in path:
        if cur is None:
            return None
        cur = cur.get(key) if isinstance(cur, Mapping) else None
    return cur


def _pair_mean(ep_a: Mapping[str, Any] | None, ep_b: Mapping[str, Any] | None, model: str, path: Sequence[str]) -> float | None:
    vals = [_endpoint_metric(ep_a, model, path), _endpoint_metric(ep_b, model, path)]
    vals = [_f(v) for v in vals if _f(v) is not None]
    return float(np.mean(vals)) if vals else None


def build_tables(
    pairs: Sequence[Mapping[str, Any]],
    pair_records: Mapping[str, Mapping[str, Any]],
    measured: Mapping[str, Mapping[str, Any]],
    oracle: Mapping[str, Mapping[str, Any]],
    *,
    b: Any,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    pair_rows: list[dict[str, Any]] = []
    endpoint_rows: list[dict[str, Any]] = []
    oracle_rows: list[dict[str, Any]] = []

    for pair in pairs:
        pid = str(pair["pair_id"])
        pr = pair_records.get(pid)
        if pr is None:
            continue
        ea = measured.get(f"{pid}:a")
        eb = measured.get(f"{pid}:b")
        oa = oracle.get(f"{pid}:a")
        ob = oracle.get(f"{pid}:b")
        split_a, split_b = str(pair["split_a"]), str(pair["split_b"])
        row_a, row_b = int(pair["row_a"]), int(pair["row_b"])
        cache_a = A.cache_row_of(b, split_a, row_a)
        cache_b = A.cache_row_of(b, split_b, row_b)
        y_a = float(np.asarray(b.targets_raw[split_a], dtype=np.float64)[cache_a])
        y_b = float(np.asarray(b.targets_raw[split_b], dtype=np.float64)[cache_b])
        delta_y = y_b - y_a
        wc_a = float(pr["wc"]["wc_a"])
        wc_b = float(pr["wc"]["wc_b"])
        delta_wc = wc_b - wc_a
        direction_evaluable = bool(abs(delta_y) >= 0.1)

        pred_a_l = _endpoint_prediction(ea, "LCRC")
        pred_b_l = _endpoint_prediction(eb, "LCRC")
        pred_a_g = _endpoint_prediction(ea, "GINE")
        pred_b_g = _endpoint_prediction(eb, "GINE")
        preds = {}
        for name, pa, pb in (("LCRC", pred_a_l, pred_b_l), ("GINE", pred_a_g, pred_b_g)):
            if pa is None or pb is None:
                preds[name] = {"pred_a": None, "pred_b": None, "delta": None, "error": None, "direction": None, "accurate": None}
            else:
                dp = float(pb - pa)
                preds[name] = {
                    "pred_a": pa,
                    "pred_b": pb,
                    "delta": dp,
                    "error": abs(dp - delta_y),
                    "direction": bool(_sign(dp) == _sign(delta_y)) if direction_evaluable else None,
                    "accurate": bool(abs(dp - delta_y) <= 0.25),
                }

        loc: dict[str, dict[str, Any]] = {}
        for name, model in (("LCRC", "LCRC"), ("GINE", "GINE")):
            vals = {
                "atom_jaccard": _pair_mean(ea, eb, model, ("top1", "atom_jaccard")),
                "atom_recall": _pair_mean(ea, eb, model, ("top1", "atom_recall")),
                "bond_recall": _pair_mean(ea, eb, model, ("top1", "bond_recall")),
                "hit_attachment_boundary": _pair_mean(ea, eb, model, ("top1", "hit_attachment_boundary")),
                "atom_jaccard_random": _pair_mean(ea, eb, model, ("random", "mean_atom_jaccard")),
                "atom_recall_random": _pair_mean(ea, eb, model, ("random", "mean_atom_recall")),
                "bond_recall_random": _pair_mean(ea, eb, model, ("random", "mean_bond_recall")),
                "hit_boundary_random": _pair_mean(ea, eb, model, ("random", "mean_hit_attachment_boundary")),
                "atom_jaccard_adv": _pair_mean(ea, eb, model, ("advantage", "atom_jaccard")),
                "atom_recall_adv": _pair_mean(ea, eb, model, ("advantage", "atom_recall")),
                "bond_recall_adv": _pair_mean(ea, eb, model, ("advantage", "bond_recall")),
                "hit_boundary_adv": _pair_mean(ea, eb, model, ("advantage", "hit_attachment_boundary")),
                "top1_k": _pair_mean(ea, eb, model, ("top1", "n_bonds")),
            }
            vals["localisation_success"] = bool(
                (vals["atom_recall"] is not None and vals["atom_recall"] >= 0.5)
                or (vals["hit_attachment_boundary"] is not None and vals["hit_attachment_boundary"] >= 0.5)
            )
            loc[name] = vals

        oracle_pair: dict[str, Any] = {}
        for name in ("LCRC", "GINE", "LCRC_s2_matched", "LOCAL_LCRC"):
            caps = []
            meds = []
            ens = []
            for rec in (oa, ob):
                if rec is None or rec.get("status") != "OK":
                    continue
                c = rec.get("oracle", {}).get(name)
                if not c:
                    continue
                if _f(c.get("top1_capture")) is not None:
                    caps.append(float(c["top1_capture"]))
                if _f(c.get("random_median_capture")) is not None:
                    meds.append(float(c["random_median_capture"]))
                if _f(c.get("oracle_enrichment")) is not None:
                    ens.append(float(c["oracle_enrichment"]))
            oracle_pair[name] = {
                "capture": float(np.mean(caps)) if caps else None,
                "random_median": float(np.mean(meds)) if meds else None,
                "enrichment": float(np.mean(ens)) if ens else None,
                "n_endpoints": len(caps),
            }

        def fail_mode(name: str) -> str:
            p = preds[name]
            if p["accurate"] is None:
                return "not_evaluable"
            localized = loc[name]["localisation_success"]
            if p["accurate"] and localized:
                return "accurate_localized"
            if p["accurate"] and not localized:
                return "accurate_mislocalized"
            if (not p["accurate"]) and localized:
                return "inaccurate_localized"
            return "inaccurate_mislocalized"

        row = {
            "pair_id": pid,
            "pair_index": int(pair["pair_index"]),
            "component": int(pair["component"]),
            "split_pair": str(pair["split_pair"]),
            "split_a": split_a,
            "split_b": split_b,
            "stratum": stratum_of(pair),
            "in_H": bool(pair["in_H"]),
            "evaluable": bool(pr["evaluable"]),
            "y_a": y_a,
            "y_b": y_b,
            "delta_y": delta_y,
            "wc_a": wc_a,
            "wc_b": wc_b,
            "delta_wc": delta_wc,
            "direction_evaluable": direction_evaluable,
            "pred_a_LCRC": preds["LCRC"]["pred_a"],
            "pred_b_LCRC": preds["LCRC"]["pred_b"],
            "delta_pred_LCRC": preds["LCRC"]["delta"],
            "abs_delta_error_LCRC": preds["LCRC"]["error"],
            "direction_LCRC": preds["LCRC"]["direction"],
            "pair_accurate_LCRC": preds["LCRC"]["accurate"],
            "endpoint_abs_err_a_LCRC": None if pred_a_l is None else abs(pred_a_l - y_a),
            "endpoint_abs_err_b_LCRC": None if pred_b_l is None else abs(pred_b_l - y_b),
            "max_endpoint_abs_err_LCRC": None if (pred_a_l is None or pred_b_l is None) else max(abs(pred_a_l - y_a), abs(pred_b_l - y_b)),
            "pred_a_GINE": preds["GINE"]["pred_a"],
            "pred_b_GINE": preds["GINE"]["pred_b"],
            "delta_pred_GINE": preds["GINE"]["delta"],
            "abs_delta_error_GINE": preds["GINE"]["error"],
            "direction_GINE": preds["GINE"]["direction"],
            "pair_accurate_GINE": preds["GINE"]["accurate"],
            "endpoint_abs_err_a_GINE": None if pred_a_g is None else abs(pred_a_g - y_a),
            "endpoint_abs_err_b_GINE": None if pred_b_g is None else abs(pred_b_g - y_b),
            "max_endpoint_abs_err_GINE": None if (pred_a_g is None or pred_b_g is None) else max(abs(pred_a_g - y_a), abs(pred_b_g - y_b)),
            "failure_mode_LCRC": fail_mode("LCRC"),
            "failure_mode_GINE": fail_mode("GINE"),
        }
        for name in ("LCRC", "GINE"):
            for key, val in loc[name].items():
                row[f"{key}_{name}"] = val
            for key, val in oracle_pair[name].items():
                row[f"oracle_{key}_{name}"] = val
        row["oracle_capture_LCRC_s2"] = oracle_pair["LCRC_s2_matched"]["capture"]
        row["oracle_enrichment_LCRC_s2"] = oracle_pair["LCRC_s2_matched"]["enrichment"]
        row["oracle_capture_LOCAL_LCRC"] = oracle_pair["LOCAL_LCRC"]["capture"]
        row["oracle_enrichment_LOCAL_LCRC"] = oracle_pair["LOCAL_LCRC"]["enrichment"]
        pair_rows.append(row)

        # endpoint-level oracle table
        for side, ep, orec in (("a", ea, oa), ("b", eb, ob)):
            if orec is None or orec.get("status") != "OK":
                continue
            for model in ("LCRC", "GINE", "LOCAL_LCRC", "LCRC_s2_matched"):
                cap = orec.get("oracle", {}).get(model)
                if not cap:
                    continue
                loc_metric = orec.get("localisation", {}).get(model, {}).get("top1", {})
                oracle_rows.append(
                    {
                        "pair_id": pid,
                        "side": side,
                        "row_id": int(ep["meta"]["row_id"]) if ep else None,
                        "split": str(ep["meta"]["split"]) if ep else None,
                        "component": int(pair["component"]),
                        "split_pair": str(pair["split_pair"]),
                        "stratum": stratum_of(pair),
                        "model": model,
                        "candidate_k": loc_metric.get("n_bonds"),
                        "candidate_atoms": loc_metric.get("n_atoms"),
                        "top1_capture": cap.get("top1_capture"),
                        "random_median_capture": cap.get("random_median_capture"),
                        "oracle_enrichment": cap.get("oracle_enrichment"),
                        "denominator": cap.get("denominator"),
                        "n_random": cap.get("n_random"),
                        "atom_recall": loc_metric.get("atom_recall"),
                        "atom_jaccard": loc_metric.get("atom_jaccard"),
                        "hit_attachment_boundary": loc_metric.get("hit_attachment_boundary"),
                        "pred_raw": (ep or {}).get("pred", {}).get(model.replace("_s2_matched", ""), {}).get("pred_raw"),
                    }
                )
    return pair_rows, endpoint_rows, oracle_rows


# ---------------------------------------------------------------------------
# model comparison / decision
# ---------------------------------------------------------------------------


def _metric_values(pair_rows: Sequence[Mapping[str, Any]], key: str) -> np.ndarray:
    return np.asarray([_f(r.get(key)) if _f(r.get(key)) is not None else np.nan for r in pair_rows], dtype=np.float64)


def compare_models(
    pair_rows: Sequence[Mapping[str, Any]],
    oracle_rows: Sequence[Mapping[str, Any]],
    support: Mapping[str, Any],
    replay: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    out: list[dict[str, Any]] = []
    summary: dict[str, Any] = {"metrics": {}, "prediction": {}}

    def add_metric(metric: str, stratum: str, unit: str, l_vals: np.ndarray, g_vals: np.ndarray, comps: np.ndarray, mask: np.ndarray, note: str = "") -> None:
        mask = np.asarray(mask, dtype=bool) & np.isfinite(l_vals) & np.isfinite(g_vals)
        if int(mask.sum()) == 0:
            return
        bs = _boot({"l": l_vals, "g": g_vals}, comps, mask, stat_fn=_diff_stat)
        point = bs["point"]
        ci = bs["ci95"]
        sup = _support(comps[mask], int(mask.sum()))
        for model, key in (("LCRC", "l"), ("GINE", "g")):
            out.append(
                {
                    "metric": metric,
                    "stratum": stratum,
                    "model": model,
                    "unit": unit,
                    "n_units": int(mask.sum()),
                    "n_components": int(np.unique(comps[mask]).size),
                    "support": sup,
                    "point": point[key],
                    "ci_low": ci[key][0],
                    "ci_high": ci[key][1],
                    "diff_point": None,
                    "diff_ci_low": None,
                    "diff_ci_high": None,
                    "note": note,
                }
            )
        out.append(
            {
                "metric": metric,
                "stratum": stratum,
                "model": "LCRC_minus_GINE",
                "unit": unit,
                "n_units": int(mask.sum()),
                "n_components": int(np.unique(comps[mask]).size),
                "support": sup,
                "point": point["diff"],
                "ci_low": ci["diff"][0],
                "ci_high": ci["diff"][1],
                "diff_point": point["diff"],
                "diff_ci_low": ci["diff"][0],
                "diff_ci_high": ci["diff"][1],
                "note": note,
            }
        )
        summary["metrics"].setdefault(metric, {})[stratum] = {
            "point": bs["point"],
            "ci95": bs["ci95"],
            "n_pairs": bs["n_pairs"],
            "n_components": bs["n_components"],
            "n_boot": bs["n_boot"],
            "seed": bs["seed"],
        }

    for stratum in STRATA:
        subset = _subset(pair_rows, stratum)
        if not subset:
            continue
        comps = np.asarray([int(r["component"]) for r in subset], dtype=np.int64)
        n = len(subset)
        # prediction metrics
        add_metric("pair_abs_delta_error", stratum, "pairs", _metric_values(subset, "abs_delta_error_LCRC"), _metric_values(subset, "abs_delta_error_GINE"), comps, np.ones(n, dtype=bool), "lower is better")
        dirmask = np.asarray([bool(r["direction_evaluable"]) for r in subset], dtype=bool)
        add_metric(
            "direction_accuracy",
            stratum,
            "pairs",
            np.asarray([np.nan if r["direction_LCRC"] is None else float(bool(r["direction_LCRC"])) for r in subset], dtype=np.float64),
            np.asarray([np.nan if r["direction_GINE"] is None else float(bool(r["direction_GINE"])) for r in subset], dtype=np.float64),
            comps,
            dirmask,
            "only |Delta y| >= 0.1",
        )
        # localisation advantages (pair mean of the two endpoints)
        for metric in ("atom_jaccard_adv", "atom_recall_adv", "bond_recall_adv", "hit_boundary_adv"):
            add_metric(
                f"{metric}_advantage",
                stratum,
                "pairs",
                _metric_values(subset, f"{metric}_LCRC"),
                _metric_values(subset, f"{metric}_GINE"),
                comps,
                np.ones(n, dtype=bool),
                "Top-1 minus mean same-size random controls",
            )
        # oracle endpoint metrics
        osub = [r for r in oracle_rows if r["stratum"] == stratum and r["model"] in ("LCRC", "GINE")]
        if osub:
            ocomps = np.asarray([int(r["component"]) for r in osub], dtype=np.int64)
            by_key: dict[tuple[str, str], dict[str, Any]] = {}
            for r in osub:
                by_key.setdefault((str(r["pair_id"]), str(r["side"])), {})[str(r["model"])] = r
            keys = [k for k in sorted(by_key) if "LCRC" in by_key[k] and "GINE" in by_key[k]]
            for metric, key in (("oracle_enrichment", "oracle_enrichment"), ("oracle_top1_capture", "top1_capture")):
                comps2 = np.asarray([int(by_key[k]["LCRC"]["component"]) for k in keys], dtype=np.int64)
                lv2 = np.asarray([_f(by_key[k]["LCRC"].get(key)) for k in keys], dtype=np.float64)
                gv2 = np.asarray([_f(by_key[k]["GINE"].get(key)) for k in keys], dtype=np.float64)
                mask2 = np.isfinite(lv2) & np.isfinite(gv2)
                add_metric(metric, stratum, "endpoints", lv2, gv2, comps2, mask2, "component bootstrap over endpoints")
            # S2-matched LCRC only
            s2 = [r for r in oracle_rows if r["stratum"] == stratum and r["model"] == "LCRC_s2_matched" and _f(r.get("oracle_enrichment")) is not None]
            if s2:
                comps3 = np.asarray([int(r["component"]) for r in s2], dtype=np.int64)
                vals3 = np.asarray([_f(r["oracle_enrichment"]) for r in s2], dtype=np.float64)
                bs = _boot({"s2": vals3}, comps3, np.ones(len(s2), dtype=bool))
                out.append({
                    "metric": "oracle_enrichment_s2_matched",
                    "stratum": stratum,
                    "model": "LCRC",
                    "unit": "endpoints",
                    "n_units": len(s2),
                    "n_components": int(np.unique(comps3).size),
                    "support": _support(comps3, len(s2)),
                    "point": bs["point"]["s2"],
                    "ci_low": bs["ci95"]["s2"][0],
                    "ci_high": bs["ci95"]["s2"][1],
                    "diff_point": None,
                    "diff_ci_low": None,
                    "diff_ci_high": None,
                    "note": "supplementary LCRC-only n_s2_affected-matched control",
                })

    # full-split RMSE rows
    for split in ("valid", "test"):
        for model in ("LCRC", "GINE"):
            rec = next(c for c in replay["checkpoints"] if c["name"] == ("L_lipophilicity" if model == "LCRC" else "G0_lipophilicity"))
            rmse = rec["replay"]["splits"][split]["rmse_raw"]
            out.append({
                "metric": "full_split_rmse",
                "stratum": split,
                "model": model,
                "unit": "molecules",
                "n_units": int(rec["replay"]["splits"][split]["n_rows"]),
                "n_components": None,
                "support": "FULL_SPLIT",
                "point": rmse,
                "ci_low": None,
                "ci_high": None,
                "diff_point": None,
                "diff_ci_low": None,
                "diff_ci_high": None,
                "note": "official split; Test is DEVELOPMENT_TEST",
            })
    return out, summary


def decide(
    pair_rows: Sequence[Mapping[str, Any]],
    comparison_rows: Sequence[Mapping[str, Any]],
    replay: Mapping[str, Any],
    support: Mapping[str, Any],
) -> dict[str, Any]:
    def row(metric: str, stratum: str, model: str) -> Mapping[str, Any] | None:
        for rec in comparison_rows:
            if rec["metric"] == metric and rec["stratum"] == stratum and rec["model"] == model:
                return rec
        return None

    pred: dict[str, Any] = {}
    for model in ("LCRC", "GINE"):
        name = "L_lipophilicity" if model == "LCRC" else "G0_lipophilicity"
        rec = next(c for c in replay["checkpoints"] if c["name"] == name)
        valid = float(rec["replay"]["splits"]["valid"]["rmse_raw"])
        test = float(rec["replay"]["splits"]["test"]["rmse_raw"])
        d = row("direction_accuracy", "all", model)
        pred[model] = {
            "valid_rmse": valid,
            "test_rmse": test,
            "direction_accuracy_all": None if d is None else d["point"],
            "direction_accuracy_ci95": None if d is None else [d["ci_low"], d["ci_high"]],
        }
    prediction_accurate = bool(
        all(pred[m]["valid_rmse"] <= 0.80 and pred[m]["test_rmse"] <= 0.80 for m in pred)
        and all((pred[m]["direction_accuracy_all"] or 0.0) >= 0.75 for m in pred)
    )

    oracle_reliable: dict[str, bool] = {}
    oracle_detail: dict[str, Any] = {}
    for model in ("LCRC", "GINE"):
        r = row("oracle_enrichment", "all", model)
        ok = bool(r is not None and r["ci_low"] is not None and r["ci_low"] > 0 and r["point"] >= 0.02)
        oracle_reliable[model] = ok
        oracle_detail[model] = None if r is None else {"point": r["point"], "ci95": [r["ci_low"], r["ci_high"]], "n_components": r["n_components"], "support": r["support"]}

    real_supported: dict[str, bool] = {}
    real_detail: dict[str, Any] = {}
    for model in ("LCRC", "GINE"):
        checks = {}
        for metric in ("atom_jaccard_adv", "bond_recall_adv", "hit_boundary_adv"):
            r = row(f"{metric}_advantage", "all", model)
            checks[metric] = bool(r is not None and r["ci_low"] is not None and r["ci_low"] > 0)
        real_supported[model] = any(checks.values())
        real_detail[model] = {"checks": checks}

    # paired LCRC-GINE difference for the primary localisation metric
    primary = row("atom_jaccard_adv_advantage", "all", "LCRC_minus_GINE")
    if primary is None:
        primary = row("atom_jaccard_advantage", "all", "LCRC_minus_GINE")
    no_unique = bool(primary is not None and primary["ci_low"] is not None and primary["ci_low"] <= 0)
    unique_lcrc = bool(primary is not None and primary["ci_low"] is not None and primary["ci_low"] > 0)

    labels: list[str] = []
    if prediction_accurate and (oracle_reliable["LCRC"] or oracle_reliable["GINE"]) and (real_supported["LCRC"] or real_supported["GINE"]):
        labels.append("CONTINUE_STRUCTURAL_EXPLANATION")
    if prediction_accurate and not (oracle_reliable["LCRC"] or oracle_reliable["GINE"]):
        labels.append("EXPLANATION_METHOD_OR_REPRESENTATION_BOTTLENECK")
    if (oracle_reliable["LCRC"] or oracle_reliable["GINE"]) and not (real_supported["LCRC"] or real_supported["GINE"]):
        labels.append("CHEMICAL_RELEVANCE_UNRESOLVED")
    if not prediction_accurate:
        labels.append("PREDICTION_CONFOUNDED")
    if no_unique:
        labels.append("NO_UNIQUE_LCRC_ADVANTAGE_SHOWN")

    return {
        "labels": labels,
        "thresholds": {
            "prediction_accurate": "both models Valid/Test RMSE <= 0.80 and direction accuracy on |Delta y|>=0.1 >= 0.75",
            "oracle_localization_reliable": "component-bootstrap 95% lower bound of oracle_enrichment > 0 and point >= 0.02",
            "real_label_localization_supported": "at least one of atom-Jaccard advantage, bond-recall advantage, attachment-boundary hit-rate advantage has bootstrap 95% lower bound > 0",
            "no_unique_lcrc_advantage": "paired LCRC-GINE atom-Jaccard advantage interval includes 0 or is negative",
        },
        "prediction_accurate": prediction_accurate,
        "prediction": pred,
        "oracle_localization_reliable": oracle_reliable,
        "oracle_detail": oracle_detail,
        "real_label_localization_supported": real_supported,
        "real_label_detail": real_detail,
        "unique_lcrc_advantage": unique_lcrc,
        "no_unique_lcrc_advantage": no_unique,
        "primary_lcrc_minus_gine_atom_jaccard_advantage": None if primary is None else {"point": primary["point"], "ci95": [primary["ci_low"], primary["ci_high"]]},
        "support": support,
        "notes": [
            "Test is DEVELOPMENT_TEST, not a held-out final benchmark.",
            "The both-endpoints-non-Train stratum is empty: NO_SUPPORT.",
            "Component bootstrap resamples 231 shared-endpoint components, not 823 independent pairs.",
            "Prediction and localisation are reported separately and are not combined into a single score.",
        ],
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--measured-dir", default=str(C.RESULTS_DIR / "measured"))
    parser.add_argument("--oracle-dir", default=str(C.RESULTS_DIR / "oracle"))
    parser.add_argument("--n-boot", type=int, default=C.BOOTSTRAP_REPLICATES)
    args = parser.parse_args()
    C.install_source_paths()
    b = M.bundle(C.DATASET)
    pair_table = C.read_json(C.ANALYSIS_DIR / "pair_table.json")
    pairs = pair_table["pairs"]
    pair_records = {str(p["pair_id"]): p for p in pairs}
    measured = _load_endpoint_records(Path(args.measured_dir) / "endpoints.jsonl")
    oracle = _load_endpoint_records(Path(args.oracle_dir) / "endpoints.jsonl")
    support = C.read_json(C.ANALYSIS_DIR / "pair_support.json")
    replay = C.read_json(C.ANALYSIS_DIR / "model_replay.json")

    pair_rows, endpoint_rows, oracle_rows = build_tables(pairs, pair_records, measured, oracle, b=b)
    measured_cols = [
        "pair_id", "pair_index", "component", "split_pair", "split_a", "split_b", "stratum", "in_H", "evaluable",
        "y_a", "y_b", "delta_y", "wc_a", "wc_b", "delta_wc", "direction_evaluable",
        "pred_a_LCRC", "pred_b_LCRC", "delta_pred_LCRC", "abs_delta_error_LCRC", "direction_LCRC", "pair_accurate_LCRC",
        "endpoint_abs_err_a_LCRC", "endpoint_abs_err_b_LCRC", "max_endpoint_abs_err_LCRC",
        "pred_a_GINE", "pred_b_GINE", "delta_pred_GINE", "abs_delta_error_GINE", "direction_GINE", "pair_accurate_GINE",
        "endpoint_abs_err_a_GINE", "endpoint_abs_err_b_GINE", "max_endpoint_abs_err_GINE",
        "failure_mode_LCRC", "failure_mode_GINE",
        "top1_k_LCRC", "top1_k_GINE",
        "atom_jaccard_LCRC", "atom_jaccard_random_LCRC", "atom_jaccard_adv_LCRC",
        "atom_jaccard_GINE", "atom_jaccard_random_GINE", "atom_jaccard_adv_GINE",
        "atom_recall_LCRC", "atom_recall_random_LCRC", "atom_recall_adv_LCRC",
        "atom_recall_GINE", "atom_recall_random_GINE", "atom_recall_adv_GINE",
        "bond_recall_LCRC", "bond_recall_random_LCRC", "bond_recall_adv_LCRC",
        "bond_recall_GINE", "bond_recall_random_GINE", "bond_recall_adv_GINE",
        "hit_attachment_boundary_LCRC", "hit_boundary_random_LCRC", "hit_boundary_adv_LCRC",
        "hit_attachment_boundary_GINE", "hit_boundary_random_GINE", "hit_boundary_adv_GINE",
        "localisation_success_LCRC", "localisation_success_GINE",
        "oracle_capture_LCRC", "oracle_random_median_LCRC", "oracle_enrichment_LCRC",
        "oracle_capture_GINE", "oracle_random_median_GINE", "oracle_enrichment_GINE",
        "oracle_capture_LCRC_s2", "oracle_enrichment_LCRC_s2",
        "oracle_capture_LOCAL_LCRC", "oracle_enrichment_LOCAL_LCRC",
    ]
    measured_info = _write_csv(C.ANALYSIS_DIR / "measured_pair_results.csv", pair_rows, measured_cols)
    oracle_cols = [
        "pair_id", "side", "row_id", "split", "component", "split_pair", "stratum", "model",
        "candidate_k", "candidate_atoms", "top1_capture", "random_median_capture", "oracle_enrichment",
        "denominator", "n_random", "atom_recall", "atom_jaccard", "hit_attachment_boundary", "pred_raw",
    ]
    oracle_info = _write_csv(C.ANALYSIS_DIR / "oracle_localization_results.csv", oracle_rows, oracle_cols)
    comparison_rows, comparison_summary = compare_models(pair_rows, oracle_rows, support, replay)
    comparison_cols = [
        "metric", "stratum", "model", "unit", "n_units", "n_components", "support",
        "point", "ci_low", "ci_high", "diff_point", "diff_ci_low", "diff_ci_high", "note",
    ]
    comparison_info = _write_csv(C.ANALYSIS_DIR / "model_comparison.csv", comparison_rows, comparison_cols)
    decision = decide(pair_rows, comparison_rows, replay, support)
    decision["comparison_summary"] = comparison_summary
    decision["counts"] = {
        "n_pairs": len(pair_rows),
        "n_endpoints": len(measured),
        "n_oracle_rows": len(oracle_rows),
        "failure_modes": {
            model: dict(Counter(str(r[f"failure_mode_{model}"]) for r in pair_rows)) for model in ("LCRC", "GINE")
        },
        "direction_evaluable_pairs": int(sum(1 for r in pair_rows if r["direction_evaluable"])),
        "pair_accurate": {model: int(sum(1 for r in pair_rows if r[f"pair_accurate_{model}"] is True)) for model in ("LCRC", "GINE")},
    }
    C.write_json(C.ANALYSIS_DIR / "decision.json", decision)
    manifest = {
        "round_id": C.ROUND_ID,
        "created_utc": C.utc_now(),
        "protocol_sha256": C.sha256_file(C.ROUND_DIR / "PROTOCOL.md"),
        "code_git_commit": C.git_commit(),
        "inputs": {
            "data_identity_sha256": C.sha256_file(C.ANALYSIS_DIR / "data_identity.json"),
            "pair_support_sha256": C.sha256_file(C.ANALYSIS_DIR / "pair_support.json"),
            "model_replay_sha256": C.sha256_file(C.ANALYSIS_DIR / "model_replay.json"),
            "p1_gates_sha256": C.sha256_file(C.ANALYSIS_DIR / "p1_gates.json") if (C.ANALYSIS_DIR / "p1_gates.json").is_file() else None,
            "measured_records_sha256": C.sha256_file(Path(args.measured_dir) / "endpoints.jsonl"),
            "oracle_records_sha256": C.sha256_file(Path(args.oracle_dir) / "endpoints.jsonl"),
        },
        "outputs": {
            "measured_pair_results": measured_info,
            "oracle_localization_results": oracle_info,
            "model_comparison": comparison_info,
            "decision": {"path": C.relpath(C.ANALYSIS_DIR / "decision.json"), "sha256": C.sha256_file(C.ANALYSIS_DIR / "decision.json")},
        },
        "statistics": {"bootstrap_replicates": int(args.n_boot), "bootstrap_seed": C.BOOTSTRAP_SEED, "component_unit": "shared-endpoint connected component"},
        "status": "AGGREGATED",
    }
    C.write_json(C.ANALYSIS_DIR / "run_manifest.json", manifest)
    print(json.dumps({"measured": measured_info, "oracle": oracle_info, "comparison": comparison_info, "labels": decision["labels"]}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

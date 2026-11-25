"""Collect finished runs into the round's tables, comparisons and decisions.

Everything here is descriptive: it reads ``runs/*/run.json`` (written by the
trainers), the frozen preflight files and — when present — the frozen-checkpoint
predictions from ``esr_eval.py``.  No number in this file can influence a
training decision; the decision rules were frozen in
``analysis/phase_decisions_rules.json`` before the first PCBA arm finished.

    python esr_collect.py --runs runs --analysis analysis --preds analysis/preds
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

import esr_common as C

RNG_SEED = 20261125
N_BOOT = 2000
PRIMARY_KEY = {
    "LIPO": "rmse",
    "ESOL": "rmse",
    "TOX21": "macro_roc_auc",
    "PCBA": "ap",
}


# ---------------------------------------------------------------------------
# run records
# ---------------------------------------------------------------------------


def load_runs(root: Path) -> list[dict[str, Any]]:
    out = []
    for run_json in sorted(root.glob("*/run.json")):
        payload = json.loads(run_json.read_text())
        if payload.get("smoke"):
            continue
        if payload.get("status") != "completed":
            continue
        curve_path = run_json.parent / "curve.csv"
        curve = []
        if curve_path.exists():
            with open(curve_path, newline="", encoding="utf-8") as fh:
                curve = [{k: _num(v) for k, v in row.items()} for row in csv.DictReader(fh)]
        payload["_dir"] = run_json.parent
        payload["_curve"] = curve
        out.append(payload)
    return out


def _num(value: str) -> Any:
    try:
        f = float(value)
    except (TypeError, ValueError):
        return value
    return f


def hist_row(dataset: str, name: str) -> dict[str, Any]:
    if dataset == C.PCBA:
        hist = C.PCBA_HISTORY
        entries = {k: v for k, v in hist.items() if isinstance(v, Mapping)}
        if name not in entries:
            return {}
        return {
            "dataset": dataset,
            "arm": name,
            "source": "historical_reference",
            "parameters": entries[name].get("params"),
            "best_valid_primary": entries[name].get("valid_ap"),
            "test_primary_reference": entries[name].get("test_ap"),
            "epochs": entries[name].get("horizon"),
            "notes": "read-only historical reference, not trained in this round",
        }
    if dataset == C.LIPO:
        hist = C.SMALL_HISTORY["LIPO"]
        key = f"{name}_valid_rmse"
        if key not in hist:
            return {}
        return {
            "dataset": dataset,
            "arm": name,
            "source": "historical_reference",
            "best_valid_primary": hist[key],
            "test_primary_reference": hist.get(f"{name}_test_rmse"),
            "notes": "read-only historical reference",
        }
    return {}


# ---------------------------------------------------------------------------
# main table
# ---------------------------------------------------------------------------


def main_table(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for rec in records:
        dataset = rec["dataset"]
        key = PRIMARY_KEY[dataset]
        best = rec.get("best_valid_metrics") or {}
        final = rec.get("final_epoch_valid_metrics") or {}
        train_best = rec.get("train_metrics_at_best") or {}
        curve = rec.get("_curve") or []
        prefix_best = None
        prefix_final = None
        if dataset == C.PCBA:
            inside = [c for c in curve if float(c["epoch"]) <= 25]
            if inside:
                prefix_best = max(float(c["valid_ap"]) for c in inside)
                prefix_final = float(inside[-1]["valid_ap"])
        row = {
            "dataset": dataset,
            "arm": rec["arm"],
            "run_id": rec.get("run_id"),
            "source": "this_round",
            "seed": int(rec.get("seed", 0)),
            "parameters": int(rec.get("parameters") or 0),
            "epochs": int(rec.get("epochs") or rec.get("epochs_requested") or 0),
            "best_epoch": int(rec.get("best_epoch") or 0),
            "primary_best_valid": float(view_primary(dataset, best)),
            "primary_final_epoch": float(view_primary(dataset, final)),
            "primary_train_at_best": float(view_primary(dataset, train_best)),
            "gap_valid_minus_train": float(view_primary(dataset, best) - view_primary(dataset, train_best)),
            "valid_masked_bce": float(best.get("masked_bce", float("nan"))),
            "valid_mae": float(best.get("mae", float("nan"))),
            "wall_minutes": float(rec.get("wall_seconds", 0.0)) / 60.0,
            "gpu_hours": float(rec.get("gpu_hours", 0.0)),
            "grad_clip_events": int(rec.get("grad_clip_events") or 0),
            "best_within_25_prefix": prefix_best,
            "final_at_25_prefix": prefix_final,
            "notes": "",
        }
        if dataset == C.PCBA:
            row["n_tasks_scored"] = best.get("n_tasks_scored")
            row["n_tasks"] = best.get("n_tasks")
        rows.append(row)
    for dataset in (C.PCBA, C.LIPO):
        for name in ("W2", "FIXED1", "DYNAMIC", "C1", "C1_ADD", "C1_JOINT", "GINE"):
            ref = hist_row(dataset, name)
            if ref:
                rows.append(
                    {
                        "dataset": dataset,
                        "arm": name,
                        "run_id": None,
                        "seed": 0,
                        "parameters": ref.get("parameters"),
                        "epochs": ref.get("epochs"),
                        "primary_best_valid": ref.get("best_valid_primary"),
                        "test_primary_reference": ref.get("test_primary_reference"),
                        "source": ref.get("source"),
                        "notes": ref.get("notes"),
                    }
                )
    return rows


def view_primary(dataset: str, metrics: Mapping[str, Any]) -> float:
    if not metrics:
        return float("nan")
    key = PRIMARY_KEY[dataset]
    value = metrics.get(key)
    return float("nan") if value is None else float(value)


def write_csv(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    with open(path, "w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({k: _csv_value(row.get(k)) for k in fields})


def _csv_value(value: Any) -> Any:
    if isinstance(value, float):
        if math.isnan(value):
            return ""
        return f"{value:.8g}"
    if value is None:
        return ""
    return value


# ---------------------------------------------------------------------------
# paired comparisons from frozen-checkpoint predictions
# ---------------------------------------------------------------------------


def _load_preds(preds_dir: Path | None) -> dict[str, dict[str, Any]]:
    out: dict[str, dict[str, Any]] = {}
    if preds_dir is None or not preds_dir.exists():
        return out
    for npz in sorted(preds_dir.glob("*__valid.npz")):
        name = npz.name[: -len("__valid.npz")]
        with np.load(npz, allow_pickle=False) as data:
            out[name] = {
                "logits": np.asarray(data["logits"], dtype=np.float64),
                "y": np.asarray(data["y"], dtype=np.float64),
                "raw_pred": np.asarray(data["raw_pred"], dtype=np.float64) if "raw_pred" in data else None,
                "task_names": [str(x) for x in data["task_names"]],
            }
    return out


def _run_of(preds: Mapping[str, Any], records: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    for rec in records:
        if rec.get("run_id") == preds.get("_run"):
            return rec
    return {}


def _bootstrap_indices(n: int, rng: np.random.Generator, n_boot: int) -> np.ndarray:
    return rng.integers(0, n, size=(n_boot, n))


def _rmse(pred: np.ndarray, y: np.ndarray) -> float:
    return float(np.sqrt(np.mean((pred - y) ** 2)))


def paired_regression(a: Mapping[str, Any], b: Mapping[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    pa, pb = a["raw_pred"], b["raw_pred"]
    y = a["y"]
    n = y.size
    idx = _bootstrap_indices(n, rng, N_BOOT)
    diffs = np.empty(N_BOOT, dtype=np.float64)
    for i in range(N_BOOT):
        rows = idx[i]
        diffs[i] = _rmse(pa[rows], y[rows]) - _rmse(pb[rows], y[rows])
    point = _rmse(pa, y) - _rmse(pb, y)
    return {
        "kind": "paired_molecule_bootstrap_rmse_delta",
        "delta_point": point,
        "delta_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "share_a_better": float((diffs < 0).mean()),
        "n": int(n),
        "n_boot": int(N_BOOT),
    }


def _task_aucs(logits: np.ndarray, y: np.ndarray) -> np.ndarray:
    from sklearn.metrics import roc_auc_score

    out = np.full(logits.shape[1], np.nan)
    probs = 1.0 / (1.0 + np.exp(-logits))
    for t in range(logits.shape[1]):
        obs = np.isfinite(y[:, t])
        yy = y[obs, t]
        if yy.size == 0 or np.unique(yy).size != 2:
            continue
        out[t] = float(roc_auc_score(yy, probs[obs, t]))
    return out


def paired_multitask(a: Mapping[str, Any], b: Mapping[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    la, lb, y = a["logits"], b["logits"], a["y"]
    n = y.shape[0]
    auc_a, auc_b = _task_aucs(la, y), _task_aucs(lb, y)
    mask = np.isfinite(auc_a) & np.isfinite(auc_b)
    point = float(np.nanmean(auc_a[mask]) - np.nanmean(auc_b[mask]))
    idx = _bootstrap_indices(n, rng, 500)
    diffs = np.empty(idx.shape[0], dtype=np.float64)
    for i in range(idx.shape[0]):
        rows = idx[i]
        da = _task_aucs(la[rows], y[rows])
        db = _task_aucs(lb[rows], y[rows])
        m = np.isfinite(da) & np.isfinite(db)
        diffs[i] = float(np.nanmean(da[m]) - np.nanmean(db[m])) if m.any() else np.nan
    diffs = diffs[np.isfinite(diffs)]
    per_task = {
        a["task_names"][t]: float(auc_a[t] - auc_b[t]) for t in range(len(a["task_names"])) if mask[t]
    }
    return {
        "kind": "paired_molecule_bootstrap_macro_auc_delta",
        "delta_point": point,
        "delta_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "share_a_better": float((diffs > 0).mean()),
        "per_task_delta": per_task,
        "n_tasks": int(mask.sum()),
        "n_boot": int(idx.shape[0]),
    }


def paired_pcba(a: Mapping[str, Any], b: Mapping[str, Any], rng: np.random.Generator) -> dict[str, Any]:
    from pc_metrics import per_task_ap

    ya, yb = a["y"], b["y"]
    ta = per_task_ap(ya, 1.0 / (1.0 + np.exp(-a["logits"])))
    tb = per_task_ap(yb, 1.0 / (1.0 + np.exp(-b["logits"])))
    pa = np.asarray(ta["per_task"], dtype=np.float64)
    pb = np.asarray(tb["per_task"], dtype=np.float64)
    mask = np.isfinite(pa) & np.isfinite(pb)
    point = float(np.nanmean(pa[mask]) - np.nanmean(pb[mask]))
    idx = rng.integers(0, int(mask.sum()), size=(N_BOOT, int(mask.sum())))
    va, vb = pa[mask], pb[mask]
    diffs = va[idx].mean(axis=1) - vb[idx].mean(axis=1)
    def bce(logits: np.ndarray, y: np.ndarray) -> float:
        m = np.isfinite(y)
        loss = np.logaddexp(0.0, logits) - np.where(m, y, 0.0) * logits
        return float((loss * m).sum() / m.sum())

    rows = rng.integers(0, ya.shape[0], size=(200, ya.shape[0]))
    bce_deltas = np.array([bce(a["logits"][r], ya[r]) - bce(b["logits"][r], ya[r]) for r in rows])
    return {
        "kind": "paired_task_bootstrap_macro_ap_delta + molecule_bootstrap_bce_delta",
        "delta_point": point,
        "delta_ci95": [float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))],
        "share_a_better": float((diffs > 0).mean()),
        "n_tasks": int(mask.sum()),
        "improved_tasks": int((pa[mask] > pb[mask]).sum()),
        "degraded_tasks": int((pa[mask] < pb[mask]).sum()),
        "median_task_delta": float(np.median(va - vb)),
        "bce_delta_point": float(bce(a["logits"], ya) - bce(b["logits"], ya)),
        "bce_delta_ci95": [float(np.percentile(bce_deltas, 2.5)), float(np.percentile(bce_deltas, 97.5))],
        "n_boot": int(N_BOOT),
    }


def comparisons(
    records: Sequence[Mapping[str, Any]], preds: Mapping[str, Mapping[str, Any]]
) -> dict[str, Any]:
    rng = np.random.default_rng(RNG_SEED)
    out: dict[str, Any] = {"seed": RNG_SEED, "n_boot": N_BOOT, "datasets": {}}
    by_dataset: dict[str, list[str]] = {}
    name_of = {rec.get("run_id"): rec for rec in records}
    for name in preds:
        rec = name_of.get(name)
        if rec is None:
            continue
        by_dataset.setdefault(rec["dataset"], []).append(name)
    for dataset, names in by_dataset.items():
        names = sorted(names, key=lambda n: _arm_order(name_of[n]["arm"]))
        entries: dict[str, Any] = {}
        for i, a in enumerate(names):
            for b in names[i + 1 :]:
                ra, rb = name_of[a], name_of[b]
                pa = {
                    "logits": preds[a]["logits"],
                    "y": preds[a]["y"],
                    "raw_pred": preds[a]["raw_pred"],
                    "task_names": preds[a]["task_names"],
                }
                pb = {
                    "logits": preds[b]["logits"],
                    "y": preds[b]["y"],
                    "raw_pred": preds[b]["raw_pred"],
                    "task_names": preds[b]["task_names"],
                }
                if dataset in (C.LIPO, C.ESOL):
                    result = paired_regression(pa, pb, rng)
                elif dataset == C.TOX21:
                    result = paired_multitask(pa, pb, rng)
                else:
                    result = paired_pcba(pa, pb, rng)
                result.update(
                    {
                        "dataset": dataset,
                        "arm_a": ra["arm"],
                        "arm_b": rb["arm"],
                        "primary_a": view_primary(dataset, ra["best_valid_metrics"]),
                        "primary_b": view_primary(dataset, rb["best_valid_metrics"]),
                        "preferred": "a" if _prefers(dataset, result) else "b",
                    }
                )
                entries[f"{ra['arm']}__vs__{rb['arm']}"] = result
        out["datasets"][dataset] = entries
    return out


def _arm_order(arm: str) -> int:
    order = ["BASE", "SREN-ADD", "SREN-EVID", "SREN-BANK", "HIDDEN", "SREN-COMPOUND", "UNARY"]
    return order.index(arm) if arm in order else 99


def _prefers(dataset: str, result: Mapping[str, Any]) -> bool:
    """True when arm_a has the better primary point estimate (no CI used)."""
    a, b = float(result["primary_a"]), float(result["primary_b"])
    if dataset in (C.LIPO, C.ESOL):
        return a < b
    return a > b


# ---------------------------------------------------------------------------
# per-task table
# ---------------------------------------------------------------------------


def per_task_rows(
    records: Sequence[Mapping[str, Any]], preds: Mapping[str, Mapping[str, Any]]
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    name_of = {rec.get("run_id"): rec for rec in records}
    for name, payload in preds.items():
        rec = name_of.get(name)
        if rec is None:
            continue
        dataset = rec["dataset"]
        if dataset == C.PCBA:
            from pc_metrics import per_task_ap

            task = per_task_ap(payload["y"], 1.0 / (1.0 + np.exp(-payload["logits"])))
            metric = "ap"
            values = task["per_task"]
            observed = task["observed"]
            positive = task["positive"]
        elif dataset == C.TOX21:
            aucs = _task_aucs(payload["logits"], payload["y"])
            values = {payload["task_names"][i]: (None if not np.isfinite(aucs[i]) else float(aucs[i])) for i in range(aucs.size)}
            metric = "roc_auc"
            observed = [int(np.isfinite(payload["y"][:, i]).sum()) for i in range(len(payload["task_names"]))]
            positive = [int(np.nansum(np.where(np.isfinite(payload["y"][:, i]), payload["y"][:, i], 0.0))) for i in range(len(payload["task_names"]))]
        else:
            continue
        for i, (task_name, value) in enumerate(values.items()):
            rows.append(
                {
                    "dataset": dataset,
                    "arm": rec["arm"],
                    "task": task_name,
                    "metric": metric,
                    "value": value,
                    "n_observed": observed[i] if i < len(observed) else None,
                    "n_positive": positive[i] if i < len(positive) else None,
                }
            )
    return rows


# ---------------------------------------------------------------------------
# efficiency table
# ---------------------------------------------------------------------------


def efficiency_rows(records: Sequence[Mapping[str, Any]], analysis: Path) -> list[dict[str, Any]]:
    preflight: dict[str, dict[str, Any]] = {}
    for name, key in (("preflight_timing_pcba.json", C.PCBA), ("preflight_timing_lipo.json", C.LIPO)):
        path = analysis / name
        if path.exists():
            payload = json.loads(path.read_text())
            for arm, entry in payload.get("arms", {}).items():
                preflight[f"{key}:{arm}"] = entry
    rows = []
    for rec in records:
        key = f"{rec['dataset']}:{rec['arm']}"
        pf = preflight.get(key, {})
        rows.append(
            {
                "dataset": rec["dataset"],
                "arm": rec["arm"],
                "parameters": int(rec.get("parameters") or 0),
                "steps_per_epoch": int(rec.get("steps_per_epoch") or 0),
                "epochs": int(rec.get("epochs") or 0),
                "step_ms_p50": (pf.get("step_ms") or {}).get("p50"),
                "step_ms_p90": (pf.get("step_ms") or {}).get("p90"),
                "valid_eval_seconds": pf.get("valid_eval_seconds"),
                "peak_allocated_bytes": pf.get("peak_allocated_bytes"),
                "wall_minutes": float(rec.get("wall_seconds", 0.0)) / 60.0,
                "gpu_hours": float(rec.get("gpu_hours", 0.0)),
                "relation_pairs_per_molecule": (pf.get("n_relation_pairs_per_molecule") or {}).get("mean"),
            }
        )
    return rows


# ---------------------------------------------------------------------------
# decisions
# ---------------------------------------------------------------------------


def decisions(
    records: Sequence[Mapping[str, Any]], rules_path: Path, analysis: Path
) -> dict[str, Any]:
    rules = json.loads(rules_path.read_text())
    refs = rules["reference_points"]
    pcba = {rec["arm"]: rec for rec in records if rec["dataset"] == C.PCBA}
    small: dict[str, dict[str, Mapping[str, Any]]] = {}
    for rec in records:
        if rec["dataset"] != C.PCBA:
            small.setdefault(rec["dataset"], {})[rec["arm"]] = rec

    candidates = [arm for arm in rules["best_explicit"]["candidates"] if arm in pcba]
    prefix_best: dict[str, float] = {}
    for arm in candidates:
        rec = pcba[arm]
        inside = [float(c["valid_ap"]) for c in rec["_curve"] if float(c["epoch"]) <= 25]
        if inside:
            prefix_best[arm] = max(inside)
    best_arm = None
    if prefix_best:
        ranked = sorted(prefix_best.items(), key=lambda kv: (-kv[1], _arm_order(kv[0])))
        best_arm, best_value = ranked[0]
        for arm, value in ranked[1:]:
            if best_value - value <= 0.001 and _arm_order(arm) < _arm_order(best_arm):
                best_arm, best_value = arm, value

    out: dict[str, Any] = {
        "frozen_utc": rules["frozen_utc"],
        "reference_points": refs,
        "best_explicit": {"arm": best_arm, "prefix_best_ap": prefix_best},
        "levels": {},
        "triggers": {},
        "small_data": {},
    }
    if best_arm is not None:
        v = prefix_best[best_arm]
        l1 = v >= refs["W2_best_within_25"] + 0.005
        l2 = v >= refs["FIXED1_prefix25_final"] - 0.003
        out["levels"] = {
            "L1": {"reached": bool(l1), "value": v, "threshold": refs["W2_best_within_25"] + 0.005},
            "L2": {"reached": bool(l2), "value": v, "threshold": refs["FIXED1_prefix25_final"] - 0.003},
            "L3": {"reached": bool(v >= 0.280), "value": v, "threshold": 0.280},
            "L4": {"reached": bool(v >= 0.290), "value": v, "threshold": 0.290},
            "delta_vs_W2_best_within_25": v - refs["W2_best_within_25"],
            "delta_vs_W2_prefix25_final": v - refs["W2_prefix25_final"],
            "delta_vs_FIXED1_prefix25_final": v - refs["FIXED1_prefix25_final"],
        }
        curve = [float(c["valid_ap"]) for c in pcba[best_arm]["_curve"]]
        improving = bool(
            len(curve) >= 25
            and int(np.argmax(curve)) + 1 >= 20
            and float(np.mean(curve[20:25])) > float(np.mean(curve[15:20]))
        )
        hidden = pcba.get("HIDDEN")
        hidden_best = None
        if hidden is not None:
            inside_h = [float(c["valid_ap"]) for c in hidden["_curve"] if float(c["epoch"]) <= 25]
            hidden_best = max(inside_h) if inside_h else None
        out["triggers"] = {
            "D1_UNARY_MATCHED": {"triggered": bool(l1 or l2), "rule": rules["triggers"]["D1_UNARY_MATCHED"]},
            "D2_extension_40": {
                "triggered": bool(l1 and improving),
                "rule": rules["triggers"]["D2_extension_40"],
                "curve_improving": improving,
                "best_epoch": int(np.argmax(curve)) + 1 if curve else None,
                "mean_last5": float(np.mean(curve[20:25])) if len(curve) >= 25 else None,
                "mean_prev5": float(np.mean(curve[15:20])) if len(curve) >= 20 else None,
            },
            "D3_BANK_FLEX": {
                "triggered": bool(hidden_best is not None and hidden_best >= v + 0.005 and hidden_best >= refs["W2_best_within_25"] + 0.005),
                "rule": rules["triggers"]["D3_BANK_FLEX"],
                "hidden_prefix_best": hidden_best,
            },
            "D4_REL_BLIND": {"triggered": bool(l1 or l2), "rule": rules["triggers"]["D4_REL_BLIND"]},
            "D5_RANGE4": {"triggered": bool(l1), "rule": rules["triggers"]["D5_RANGE4"]},
        }
    for dataset, arms in small.items():
        base = arms.get("BASE")
        if base is None:
            continue
        base_primary = view_primary(dataset, base["best_valid_metrics"])
        entry = {"base_primary": base_primary, "arms": {}}
        for arm, rec in arms.items():
            if arm == "BASE":
                continue
            value = view_primary(dataset, rec["best_valid_metrics"])
            if dataset in (C.LIPO, C.ESOL):
                rel = (value - base_primary) / base_primary
                entry["arms"][arm] = {"primary": value, "relative_delta": rel, "signal": bool(rel <= -C.SMALL_SIGNAL[f"{dataset}_rel_rmse"])}
            else:
                entry["arms"][arm] = {"primary": value, "absolute_delta": value - base_primary, "signal": bool(value >= base_primary + C.SMALL_SIGNAL["TOX21_macro_auc"])}
        out["small_data"][dataset] = entry
    return out


# ---------------------------------------------------------------------------
# summary
# ---------------------------------------------------------------------------


def summary(
    records: Sequence[Mapping[str, Any]], dec: Mapping[str, Any], analysis: Path
) -> dict[str, Any]:
    out: dict[str, Any] = {
        "round_id": C.ROUND_ID,
        "t0_utc": C.T0_UTC,
        "elapsed_hours": C.elapsed_hours(),
        "n_runs": len(records),
        "gpu_hours_total": float(sum(float(r.get("gpu_hours") or 0.0) for r in records)),
        "budget": C.budget_state(),
        "datasets": {},
        "levels": dec.get("levels"),
        "triggers": dec.get("triggers"),
        "test_policy": "EXPOSED_TEST_ENGINEERING_COMPARISON_SET, one-shot after freeze; PCBA Test not scoreable",
    }
    for dataset in (C.LIPO, C.TOX21, C.ESOL, C.PCBA):
        subset = [r for r in records if r["dataset"] == dataset]
        if not subset:
            continue
        rows = []
        for rec in subset:
            rows.append(
                {
                    "arm": rec["arm"],
                    "primary": view_primary(dataset, rec["best_valid_metrics"]),
                    "parameters": int(rec.get("parameters") or 0),
                    "best_epoch": int(rec.get("best_epoch") or 0),
                    "wall_minutes": float(rec.get("wall_seconds", 0.0)) / 60.0,
                }
            )
        rows.sort(key=lambda r: r["primary"], reverse=dataset in (C.TOX21, C.PCBA))
        out["datasets"][dataset] = {"n_runs": len(subset), "ranking": rows}
    return out


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description="collect ESR results")
    ap.add_argument("--runs", default=str(C.RUNS))
    ap.add_argument("--analysis", default=str(C.ANALYSIS))
    ap.add_argument("--preds", default=str(C.ANALYSIS / "preds"))
    args = ap.parse_args(argv)
    analysis = Path(args.analysis)
    records = load_runs(Path(args.runs))
    preds = {}
    for name, payload in _load_preds(Path(args.preds)).items():
        payload["_run"] = name
        preds[name] = payload
    table = main_table(records)
    write_csv(analysis / "main_table.csv", table)
    write_csv(analysis / "per_task_metrics.csv", per_task_rows(records, preds))
    write_csv(analysis / "efficiency_table.csv", efficiency_rows(records, analysis))
    C.write_json(analysis / "paired_comparisons.json", comparisons(records, preds))
    dec = decisions(records, analysis / "phase_decisions_rules.json", analysis)
    C.write_json(analysis / "phase_decisions.json", dec)
    C.write_json(analysis / "summary.json", summary(records, dec, analysis))
    C.log(f"collected {len(records)} run(s), {len(preds)} prediction file(s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

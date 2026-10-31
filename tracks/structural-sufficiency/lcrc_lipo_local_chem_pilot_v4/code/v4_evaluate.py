"""Paired Fold-1 dev analysis and the pre-registered v4 decision rules.

``delta = metric(historical A-R3) - metric(v4 arm)``; positive favours the v4
arm.  The primary metric is raw (de-normalised) logD RMSE on the frozen Fold-1
dev molecules.  The tail diagnostic drops the top 2% of historical A-R3
absolute residuals (pre-registered, selected before any v4 GPU run).

P2 is a *locality control*: it receives the same four features after
per-molecule averaging and broadcasting.  It is not a causal proof.
"""

from __future__ import annotations

import csv
from typing import Any, Sequence

import numpy as np

import v4_common as C
import v4_descriptive as Q


# ---------------------------------------------------------------------------
# metrics / bootstrap
# ---------------------------------------------------------------------------


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean(np.abs(y - p)))


def bootstrap_delta(
    y: np.ndarray,
    hist: np.ndarray,
    new: np.ndarray,
    *,
    metric: str,
    n_boot: int = C.BOOTSTRAP_RESAMPLES,
    seed: int = C.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    a = np.asarray(hist, dtype=np.float64)
    b = np.asarray(new, dtype=np.float64)
    fn = rmse if metric == "rmse" else mae
    n = y.size
    rng = np.random.default_rng(int(seed))
    point = fn(y, a) - fn(y, b)
    vals = np.empty(int(n_boot), dtype=np.float64)
    for i in range(int(n_boot)):
        idx = rng.integers(0, n, n)
        vals[i] = fn(y[idx], a[idx]) - fn(y[idx], b[idx])
    return {
        "metric": metric,
        "delta_definition": f"{metric.upper()}(historical) - {metric.upper()}(v4); positive favours v4",
        "point": float(point),
        "mean": float(vals.mean()),
        "ci95_low": float(np.quantile(vals, 0.025)),
        "ci95_high": float(np.quantile(vals, 0.975)),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "n_molecules": int(n),
        "ci_includes_zero": bool(np.quantile(vals, 0.025) <= 0.0 <= np.quantile(vals, 0.975)),
    }


# ---------------------------------------------------------------------------
# loading
# ---------------------------------------------------------------------------


def _dev_rows_and_labels() -> tuple[list[int], np.ndarray]:
    folds = C.F.load_folds()
    positions = [int(p) for p in C.F.fold_positions(folds, C.FOLD)]
    y_all = np.asarray(C.load_fit_labels()["y"], dtype=np.float64)
    return positions, y_all[np.asarray(positions, dtype=np.int64)]


def load_run(key: str) -> dict[str, Any]:
    train_path = C.RUNS / key / "train.json"
    pred_path = C.RUNS / key / "predictions.npz"
    if not train_path.exists() or not pred_path.exists():
        raise FileNotFoundError(f"missing v4 run {key}")
    train = C.read_json(train_path)
    data = np.load(pred_path)
    rows = [int(x) for x in data["rows"].tolist()]
    pred = np.asarray(data["pred"], dtype=np.float64)
    expected_rows, y = _dev_rows_and_labels()
    if rows != expected_rows:
        raise ValueError(f"{key}: prediction rows do not match fold-1 dev positions")
    if not np.isfinite(pred).all():
        raise ValueError(f"{key}: non-finite dev predictions")
    return {
        "key": key,
        "train": train,
        "rows": rows,
        "pred": pred,
        "y": y,
        "train_rmse": float(train["metrics"]["train_rmse_raw"]),
        "train_mae": float(train["metrics"]["train_mae_raw"]),
        "dev_rmse": float(train["metrics"]["dev_rmse_raw"]),
        "dev_mae": float(train["metrics"]["dev_mae_raw"]),
        "gap": float(train["metrics"]["generalisation_gap"]),
    }


def load_historical() -> dict[str, Any]:
    key = C.HISTORICAL["key"]
    run_dir = C.V2_RUNS / key
    train = C.read_json(run_dir / "train.json")
    data = np.load(run_dir / "predictions.npz")
    rows = [int(x) for x in data["rows"].tolist()]
    pred = np.asarray(data["pred"], dtype=np.float64)
    expected_rows, y = _dev_rows_and_labels()
    if rows != expected_rows:
        raise ValueError(f"{key}: prediction rows do not match fold-1 dev positions")
    if abs(float(train["metrics"]["dev_rmse_raw"]) - C.HISTORICAL["dev_rmse"]) > 1e-9:
        raise ValueError(f"{key}: historical RMSE constant changed")
    if abs(float(train["metrics"]["dev_mae_raw"]) - C.HISTORICAL["dev_mae"]) > 1e-9:
        raise ValueError(f"{key}: historical MAE constant changed")
    return {
        "key": key,
        "train": train,
        "rows": rows,
        "pred": pred,
        "y": y,
        "train_rmse": float(train["metrics"]["train_rmse_raw"]),
        "train_mae": float(train["metrics"]["train_mae_raw"]),
        "dev_rmse": float(train["metrics"]["dev_rmse_raw"]),
        "dev_mae": float(train["metrics"]["dev_mae_raw"]),
        "gap": float(train["metrics"]["generalisation_gap"]),
    }


# ---------------------------------------------------------------------------
# tail diagnostic
# ---------------------------------------------------------------------------


def tail_analysis(
    y: np.ndarray,
    hist: np.ndarray,
    new: np.ndarray,
    *,
    fraction: float = C.TAIL_FRACTION,
    min_retained_fraction: float = C.TAIL_MIN_RETAINED_FRACTION,
) -> dict[str, Any]:
    y = np.asarray(y, dtype=np.float64)
    h = np.asarray(hist, dtype=np.float64)
    n = np.asarray(new, dtype=np.float64)
    err_h = h - y
    err_n = n - y
    abs_h = np.abs(err_h)
    order = np.argsort(-abs_h, kind="stable")
    n_drop = max(1, int(round(float(fraction) * y.size)))
    drop_idx = order[:n_drop]
    keep = np.ones(y.size, dtype=bool)
    keep[drop_idx] = False
    delta_full = rmse(y, h) - rmse(y, n)
    delta_keep = rmse(y[keep], h[keep]) - rmse(y[keep], n[keep])
    sq_improvement = err_h * err_h - err_n * err_n
    net = float(sq_improvement.sum())
    tail = float(sq_improvement[drop_idx].sum())
    top1 = float(sq_improvement.max())
    retained_fraction = (delta_keep / delta_full) if delta_full > 0 else None
    tail_share = (tail / net) if net != 0 else None
    top1_share = (max(0.0, top1) / net) if net > 0 else None
    not_tail_dependent = bool(
        delta_keep > 0
        and retained_fraction is not None
        and retained_fraction >= float(min_retained_fraction)
        and top1_share is not None
        and top1_share < 1.0
    )
    return {
        "fraction": float(fraction),
        "n_dropped": int(n_drop),
        "n_kept": int(keep.sum()),
        "exclusion_rule": "top absolute residual of the historical A-R3 model; row-id stable tie-break",
        "dropped_rows": [int(x) for x in drop_idx.tolist()],
        "delta_full": float(delta_full),
        "delta_drop_top_tail": float(delta_keep),
        "retained_fraction": None if retained_fraction is None else float(retained_fraction),
        "net_squared_error_improvement": net,
        "tail_squared_error_improvement": tail,
        "tail_share_of_net_gain": None if tail_share is None else float(tail_share),
        "top1_share_of_net_gain": None if top1_share is None else float(top1_share),
        "min_retained_fraction": float(min_retained_fraction),
        "direction_stable_after_tail_drop": bool(delta_keep > 0),
        "not_tail_dependent": not_tail_dependent,
    }


# ---------------------------------------------------------------------------
# paired comparison
# ---------------------------------------------------------------------------


def compare(reference: dict[str, Any], new: dict[str, Any], *, label: str) -> dict[str, Any]:
    y = reference["y"]
    h = reference["pred"]
    n = new["pred"]
    boot_rmse = bootstrap_delta(y, h, n, metric="rmse", seed=C.BOOTSTRAP_SEED)
    boot_mae = bootstrap_delta(y, h, n, metric="mae", seed=C.BOOTSTRAP_SEED + 1)
    sq = (h - y) ** 2 - (n - y) ** 2
    ab = np.abs(h - y) - np.abs(n - y)
    tail = tail_analysis(y, h, n)
    return {
        "label": label,
        "reference_key": reference["key"],
        "new_key": new["key"],
        "n": int(y.size),
        "reference": {
            "train_rmse": reference["train_rmse"],
            "train_mae": reference["train_mae"],
            "dev_rmse": reference["dev_rmse"],
            "dev_mae": reference["dev_mae"],
            "gap": reference["gap"],
        },
        "new": {
            "train_rmse": new["train_rmse"],
            "train_mae": new["train_mae"],
            "dev_rmse": new["dev_rmse"],
            "dev_mae": new["dev_mae"],
            "gap": new["gap"],
        },
        "delta_rmse": float(reference["dev_rmse"] - new["dev_rmse"]),
        "delta_mae": float(reference["dev_mae"] - new["dev_mae"]),
        "relative_rmse_change": float((new["dev_rmse"] - reference["dev_rmse"]) / reference["dev_rmse"]),
        "bootstrap_rmse": boot_rmse,
        "bootstrap_mae": boot_mae,
        "paired_molecule": {
            "mean_squared_improvement": float(sq.mean()),
            "median_squared_improvement": float(np.median(sq)),
            "mean_absolute_improvement": float(ab.mean()),
            "median_absolute_improvement": float(np.median(ab)),
            "win_rate_absolute": float(np.mean(ab > 0)),
            "win_rate_squared": float(np.mean(sq > 0)),
            "n_molecules_improved_absolute": int(np.sum(ab > 0)),
            "n_molecules_worsened_absolute": int(np.sum(ab < 0)),
        },
        "tail": {**tail, "source": C.HISTORICAL["key"]},
    }


# ---------------------------------------------------------------------------
# subgroup table
# ---------------------------------------------------------------------------


def subgroup_results(preds: dict[str, np.ndarray]) -> dict[str, Any]:
    table = Q.molecule_table()
    dev_pos = np.asarray(table["dev_positions"], dtype=np.int64)
    y = np.asarray(table["y"], dtype=np.float64)[dev_pos]
    train_pos = np.asarray(table["train_positions"], dtype=np.int64)
    charge = np.asarray(table["formal_charge"], dtype=np.int64)[dev_pos]
    hbd = np.asarray(table["hbd"], dtype=np.int64)[dev_pos]
    hba = np.asarray(table["hba"], dtype=np.int64)[dev_pos]
    tpsa = np.asarray(table["tpsa"], dtype=np.float64)[dev_pos]
    cuts = np.unique(np.quantile(np.asarray(table["tpsa"], dtype=np.float64)[train_pos], [0.25, 0.5, 0.75, 0.9]))
    defs: dict[str, list[tuple[str, Any]]] = {
        "formal_charge": [
            ("neutral", lambda v: v == 0),
            ("charged", lambda v: v != 0),
            ("positive", lambda v: v > 0),
            ("negative", lambda v: v < 0),
        ],
        "hbd_lipinski": [
            ("0", lambda v: v == 0),
            ("1", lambda v: v == 1),
            ("2", lambda v: v == 2),
            ("3", lambda v: v == 3),
            (">=4", lambda v: v >= 4),
        ],
        "hba_lipinski": [
            ("0-1", lambda v: v <= 1),
            ("2-3", lambda v: 2 <= v <= 3),
            ("4-5", lambda v: 4 <= v <= 5),
            (">=6", lambda v: v >= 6),
        ],
        "polarity_flags": [
            ("tpsa_ge_train_q75", lambda v: v >= float(cuts[2])),
            ("tpsa_ge_train_q90", lambda v: v >= float(cuts[3])),
            ("tpsa_below_train_median", lambda v: v <= float(cuts[1])),
        ],
    }
    tpsa_defs = []
    edges = [-np.inf, *cuts.tolist(), np.inf]
    for i in range(len(edges) - 1):
        lo, hi = float(edges[i]), float(edges[i + 1])
        if np.isneginf(lo):
            tpsa_defs.append((f"<= {hi:.4g}", lambda v, hi=hi: v <= hi))
        elif np.isposinf(hi):
            tpsa_defs.append((f"> {lo:.4g}", lambda v, lo=lo: v > lo))
        else:
            tpsa_defs.append((f"({lo:.4g}, {hi:.4g}]", lambda v, lo=lo, hi=hi: lo < v <= hi))
    defs["tpsa_train_quantiles"] = tpsa_defs
    values = {
        "formal_charge": charge,
        "hbd_lipinski": hbd,
        "hba_lipinski": hba,
        "polarity_flags": tpsa,
        "tpsa_train_quantiles": tpsa,
    }
    out: dict[str, Any] = {
        "n": int(y.size),
        "tpsa_train_cuts": [float(x) for x in cuts],
        "models": list(preds),
        "groups": {},
        "dev_molecule_counts": {
            "charged": int((charge != 0).sum()),
            "formal_charge_values": {str(int(x)): int((charge == x).sum()) for x in np.unique(charge)},
            "hbd_values": {str(int(x)): int((hbd == x).sum()) for x in np.unique(hbd)},
        },
    }
    for group_name, definitions in defs.items():
        block: dict[str, Any] = {}
        for label, fn in definitions:
            mask = np.asarray([bool(fn(v)) for v in values[group_name]])
            entry: dict[str, Any] = {"n": int(mask.sum())}
            for model, pred in preds.items():
                if mask.any():
                    entry[model] = {
                        "rmse": rmse(y[mask], pred[mask]),
                        "mae": mae(y[mask], pred[mask]),
                        "bias": float(np.mean(pred[mask] - y[mask])),
                    }
                else:
                    entry[model] = {"rmse": None, "mae": None, "bias": None}
            base = preds["historical_A_R3"]
            if mask.any():
                for model, pred in preds.items():
                    if model == "historical_A_R3":
                        continue
                    entry[f"delta_rmse_{model}"] = rmse(y[mask], base[mask]) - rmse(y[mask], pred[mask])
            block[label] = entry
        out["groups"][group_name] = block
    return out


# ---------------------------------------------------------------------------
# decision
# ---------------------------------------------------------------------------


def p1_gate(cmp: dict[str, Any]) -> dict[str, Any]:
    met = bool(
        cmp["delta_rmse"] >= C.MATERIALITY_RMSE
        and cmp["delta_mae"] >= 0.0
        and cmp["tail"]["not_tail_dependent"]
    )
    if cmp["delta_rmse"] < C.MATERIALITY_RMSE:
        status = "NO_CLEAR_LOCAL_CHEM_GAIN"
    elif not met:
        status = "INCONCLUSIVE"
    else:
        status = "P1_GATE_PASSED"
    return {
        "met": met,
        "status": status,
        "delta_rmse": cmp["delta_rmse"],
        "delta_mae": cmp["delta_mae"],
        "materiality_rmse": C.MATERIALITY_RMSE,
        "mae_not_worse": bool(cmp["delta_mae"] >= 0.0),
        "tail_not_dependent": bool(cmp["tail"]["not_tail_dependent"]),
        "tail": cmp["tail"],
        "rule": "delta_rmse >= 0.03 and delta_mae >= 0 and the gain is not tail-dependent",
    }


def decide(p1: dict[str, Any], p2: dict[str, Any] | None) -> dict[str, Any]:
    gate = p1_gate(p1)
    if not gate["met"]:
        verdict = gate["status"]
        reason = (
            "P1 did not reach the pre-registered 0.03 RMSE materiality with a non-worsening MAE and a "
            "non-tail-dependent gain; P2 was not run under the frozen rule"
        )
        gap = None
    else:
        if p2 is None:
            verdict = "P2_PENDING"
            reason = "P1 passed the gate; the broadcast control is required before a verdict"
            gap = None
        else:
            gap = float(p1["delta_rmse"] - p2["delta_rmse"])
            if p2["delta_rmse"] < C.MATERIALITY_RMSE or gap >= C.ALIGNMENT_GAP_MATERIAL:
                verdict = "LOCAL_CHEM_ALIGNMENT_PROMISING"
                reason = (
                    "P1 passed the gate and the broadcast control did not reproduce the gain "
                    "(P2 below materiality or the P1-P2 gap is at least the pre-registered 0.015)"
                )
            else:
                verdict = "CHEMICAL_FEATURE_INFORMATION_PROMISING"
                reason = (
                    "P1 passed the gate and the broadcast control improved similarly; the gain cannot be "
                    "attributed to atom-level correspondence"
                )
    return {
        "verdict": verdict,
        "reason": reason,
        "p1_gate": gate,
        "p1_delta_rmse": float(p1["delta_rmse"]),
        "p1_delta_mae": float(p1["delta_mae"]),
        "p2_delta_rmse": None if p2 is None else float(p2["delta_rmse"]),
        "p2_delta_mae": None if p2 is None else float(p2["delta_mae"]),
        "alignment_gap_p1_minus_p2": gap,
        "alignment_gap_material": C.ALIGNMENT_GAP_MATERIAL,
        "materiality_rmse": C.MATERIALITY_RMSE,
        "exploratory_only": True,
        "single_seed_single_fold": True,
        "mechanism_claim": "none; a single seed/fold pilot cannot establish a mechanism",
    }


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------


def analyse() -> dict[str, Any]:
    hist = load_historical()
    runs = {}
    import v4_registry as R

    for row in R.planned_registry()["runs"]:
        path = C.RUNS / row["key"] / "train.json"
        if path.exists():
            runs[row["key"]] = load_run(row["key"])
    if not runs:
        raise FileNotFoundError("no v4 run artifacts to analyse")
    p1_key = R.unconditional_keys()[0]
    if p1_key not in runs:
        raise FileNotFoundError(f"missing unconditional run {p1_key}")
    p1 = compare(hist, runs[p1_key], label="P1_vs_historical")
    p2_key = R.conditional_key()
    p2 = compare(hist, runs[p2_key], label="P2_vs_historical") if p2_key in runs else None
    p2_vs_p1 = compare(runs[p1_key], runs[p2_key], label="P2_vs_P1") if p2 else None
    decision = decide(p1, p2)

    table = [
        {
            "model": "A-R3 (v2 historical)",
            "run": hist["key"],
            "fold": 1,
            "seed": 0,
            "train_rmse": hist["train_rmse"],
            "train_mae": hist["train_mae"],
            "dev_rmse": hist["dev_rmse"],
            "dev_mae": hist["dev_mae"],
            "delta_rmse": 0.0,
            "delta_mae": 0.0,
            "gap": hist["gap"],
        }
    ]
    entries = [("P1 AtomChem4", p1_key, p1)]
    if p2_key in runs and p2 is not None:
        entries.append(("P2 BroadcastChem4", p2_key, p2))
    for label, key, cmp in entries:
        run = runs[key]
        table.append(
            {
                "model": label,
                "run": run["key"],
                "fold": int(run["train"]["fold"]),
                "seed": int(run["train"]["seed"]),
                "train_rmse": run["train_rmse"],
                "train_mae": run["train_mae"],
                "dev_rmse": run["dev_rmse"],
                "dev_mae": run["dev_mae"],
                "delta_rmse": cmp["delta_rmse"],
                "delta_mae": cmp["delta_mae"],
                "gap": run["gap"],
            }
        )

    preds = {"historical_A_R3": hist["pred"]}
    for key, run in runs.items():
        preds["P1_AtomChem4" if key == p1_key else "P2_BroadcastChem4"] = run["pred"]
    subgroups = subgroup_results(preds)

    results = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "COMPLETE" if p2 is not None or not decision["p1_gate"]["met"] else "P2_PENDING",
        "table": table,
        "runs": {
            key: {
                "train_rmse": r["train_rmse"],
                "train_mae": r["train_mae"],
                "dev_rmse": r["dev_rmse"],
                "dev_mae": r["dev_mae"],
                "gap": r["gap"],
                "train": C.track_rel(C.RUNS / key / "train.json"),
                "predictions": C.track_rel(C.RUNS / key / "predictions.npz"),
                "soup": C.track_rel(C.RUNS / key / "soup.pt"),
            }
            for key, r in runs.items()
        },
        "decision": decision,
        "subgroups_file": C.track_rel(C.ANALYSIS / "subgroup_results_v4.json"),
        "exploratory_only": True,
    }
    paired = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "P1_vs_historical": p1,
        "P2_vs_historical": p2,
        "P2_vs_P1": p2_vs_p1,
    }
    C.write_json(C.ANALYSIS / "results_v4.json", results)
    C.write_json(C.ANALYSIS / "paired_v4.json", paired)
    C.write_json(C.ANALYSIS / "decision_v4.json", decision)
    C.write_json(C.ANALYSIS / "subgroup_results_v4.json", subgroups)
    with (C.ANALYSIS / "table_v4.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=list(table[0].keys()))
        writer.writeheader()
        for row in table:
            writer.writerow(row)
    return {"status": results["status"], "decision": decision, "table": table, "paired": paired}


__all__ = [
    "rmse",
    "mae",
    "bootstrap_delta",
    "load_run",
    "load_historical",
    "tail_analysis",
    "compare",
    "subgroup_results",
    "p1_gate",
    "decide",
    "analyse",
]

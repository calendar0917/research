"""Pre-registered v2 CV/Monitor analysis and H1--H4 decisions.

Delta convention
----------------
Every "gain" is ``RMSE(comparator) - RMSE(candidate)`` so a positive value
favours the candidate:

* H1 candidate A, comparator E;
* H2 candidate selected recipe, comparator R0;
* H3 candidate SSL, comparator scratch.

The H4 gap is reported separately as ``RMSE(LCRC) - RMSE(CPU)``; positive means
the CPU baseline is better.  Monitor is evaluated exactly once, after the
prediction manifest is frozen, and is an external check rather than an
independent blind test.
"""

from __future__ import annotations

import hashlib
import math
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np

import v2_common as C
import v2_registry as R


# ---------------------------------------------------------------------------
# metrics and bootstrap
# ---------------------------------------------------------------------------


def rmse(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.sqrt(np.mean((y - p) ** 2)))


def mae(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return float(np.mean(np.abs(y - p)))


def r2(y: np.ndarray, p: np.ndarray) -> float:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    ss_res = float(np.sum((y - p) ** 2))
    ss_tot = float(np.sum((y - y.mean()) ** 2))
    return float(1.0 - ss_res / ss_tot) if ss_tot > 0 else float("nan")


def metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    y = np.asarray(y, dtype=np.float64)
    p = np.asarray(p, dtype=np.float64)
    return {
        "n": int(y.size),
        "rmse": rmse(y, p),
        "mae": mae(y, p),
        "r2": r2(y, p),
        "bias": float(np.mean(p - y)),
        "mean_label": float(y.mean()),
        "mean_prediction": float(p.mean()),
    }


def bootstrap_rmse_diff(
    y: np.ndarray,
    pred_a: np.ndarray,
    pred_b: np.ndarray,
    n_boot: int = C.BOOTSTRAP_N,
    seed: int = C.BOOTSTRAP_SEED,
) -> dict[str, Any]:
    """Paired molecule bootstrap of RMSE(a) - RMSE(b)."""
    y = np.asarray(y, dtype=np.float64)
    a = np.asarray(pred_a, dtype=np.float64)
    b = np.asarray(pred_b, dtype=np.float64)
    n = y.size
    rng = np.random.default_rng(seed)
    point = rmse(y, a) - rmse(y, b)
    vals = np.empty(n_boot, dtype=np.float64)
    for i in range(n_boot):
        idx = rng.integers(0, n, n)
        vals[i] = rmse(y[idx], a[idx]) - rmse(y[idx], b[idx])
    return {
        "point": float(point),
        "mean": float(vals.mean()),
        "ci95_low": float(np.quantile(vals, 0.025)),
        "ci95_high": float(np.quantile(vals, 0.975)),
        "n_boot": int(n_boot),
        "seed": int(seed),
        "n_molecules": int(n),
    }


# ---------------------------------------------------------------------------
# run loading
# ---------------------------------------------------------------------------


def _run(key: str) -> dict[str, Any] | None:
    path = C.RUNS / key / "train.json"
    return C.read_json(path) if path.exists() else None


def _require_run(key: str) -> dict[str, Any]:
    payload = _run(key)
    if payload is None:
        raise FileNotFoundError(f"missing run {key}")
    return payload


def _arm_recipe_mean(arm: str, recipe: str, fold: int, seeds: Iterable[int]) -> dict[str, Any] | None:
    rows = []
    for seed in seeds:
        key = f"F{fold}_{arm}_{recipe}_s{seed}"
        payload = _run(key)
        if payload is not None:
            rows.append(
                {
                    "key": key,
                    "seed": seed,
                    "dev_rmse": payload["metrics"]["dev_rmse_raw"],
                    "dev_mae": payload["metrics"]["dev_mae_raw"],
                    "train_rmse": payload["metrics"]["train_rmse_raw"],
                    "gap": payload["metrics"]["generalisation_gap"],
                }
            )
    if not rows:
        return None
    return {
        "arm": arm,
        "recipe": recipe,
        "fold": fold,
        "seeds": rows,
        "mean_dev_rmse": float(np.mean([x["dev_rmse"] for x in rows])),
        "mean_dev_mae": float(np.mean([x["dev_mae"] for x in rows])),
        "mean_gap": float(np.mean([x["gap"] for x in rows])),
    }


def _fold_dev_labels(fold: int) -> tuple[list[int], np.ndarray]:
    import v2_folds as F

    folds = F.load_folds()
    # Prediction files store cache positions (the row order of the frozen Fit
    # cache), not global Phase-1a row ids.  The Fit label vector is in the same
    # cache order, so index it with cache positions.
    positions = [int(p) for p in F.fold_positions(folds, fold)]
    fit = C.load_fit_labels()
    y_all = np.asarray(fit["y"], dtype=np.float64)
    return positions, y_all[np.asarray(positions, dtype=np.int64)]


def _prediction_vector(key: str, rows: list[int]) -> np.ndarray:
    path = C.RUNS / key / "predictions.npz"
    if not path.exists():
        raise FileNotFoundError(f"missing prediction vector for {key}")
    data = np.load(path)
    pred_rows = [int(x) for x in data["rows"].tolist()]
    if pred_rows != rows:
        raise ValueError(f"{key} prediction rows do not match the requested rows")
    return np.asarray(data["pred"], dtype=np.float64)


# ---------------------------------------------------------------------------
# internal CV analysis
# ---------------------------------------------------------------------------


def cv_analysis() -> dict[str, Any]:
    import v2_folds as F

    folds = F.load_folds()
    recipe_lock = C.read_json(C.RECIPE_LOCK) if C.RECIPE_LOCK.exists() else None
    selection = recipe_lock["selection"] if recipe_lock else {"A": "R0", "C": "R0"}
    table = []
    for fold in C.FOLDS:
        for arm in ("A", "E", "C"):
            recipes = ["R0"] if arm == "E" else list(C.RECIPE_ORDER)
            for recipe in recipes:
                row = _arm_recipe_mean(arm, recipe, fold, C.SEEDS)
                if row is not None:
                    table.append(row)

    # H1: A vs E, R0, per fold, paired by seed.
    h1_folds = []
    for fold in C.FOLDS:
        a = _arm_recipe_mean("A", "R0", fold, C.SEEDS)
        e = _arm_recipe_mean("E", "R0", fold, C.SEEDS)
        if a is None or e is None:
            continue
        seeds = sorted({x["seed"] for x in a["seeds"]} & {x["seed"] for x in e["seeds"]})
        a_by = {x["seed"]: x["dev_rmse"] for x in a["seeds"]}
        e_by = {x["seed"]: x["dev_rmse"] for x in e["seeds"]}
        deltas = {s: e_by[s] - a_by[s] for s in seeds}
        rows, y = _fold_dev_labels(fold)
        boot = None
        if seeds:
            # Mean prediction across seeds for a stable paired bootstrap.
            a_pred = np.mean([_prediction_vector(f"F{fold}_A_R0_s{s}", rows) for s in seeds], axis=0)
            e_pred = np.mean([_prediction_vector(f"F{fold}_E_R0_s{s}", rows) for s in seeds], axis=0)
            boot = bootstrap_rmse_diff(y, e_pred, a_pred, C.BOOTSTRAP_N, C.BOOTSTRAP_SEED + fold)
        h1_folds.append(
            {
                "fold": fold,
                "seeds": seeds,
                "A_mean_dev_rmse": float(np.mean(list(a_by.values()))),
                "E_mean_dev_rmse": float(np.mean(list(e_by.values()))),
                "delta_E_minus_A": float(np.mean(list(deltas.values()))),
                "per_seed_delta": deltas,
                "bootstrap": boot,
            }
        )
    h1_mean = float(np.mean([x["delta_E_minus_A"] for x in h1_folds])) if h1_folds else float("nan")
    h1_all_favor_a = bool(all(x["delta_E_minus_A"] > 0 for x in h1_folds)) if h1_folds else False
    h1 = {
        "hypothesis": "A retains an advantage over capacity-matched E",
        "candidate": "A-R0",
        "comparator": "E-R0",
        "delta": "RMSE(E) - RMSE(A); positive favours A",
        "folds": h1_folds,
        "mean_delta": h1_mean,
        "all_folds_favor_A": h1_all_favor_a,
        "materiality": C.H1_MATERIALITY,
        "promising": bool(h1_mean >= C.H1_MATERIALITY and h1_all_favor_a),
        "monitor_used": False,
    }

    # H2: selected recipe vs R0, per arm/fold where both exist.
    h2_arms = {}
    for arm in ("A", "C"):
        sel = selection.get(arm, "R0")
        fold_rows = []
        for fold in C.FOLDS:
            r0 = _arm_recipe_mean(arm, "R0", fold, C.SEEDS)
            sr = _arm_recipe_mean(arm, sel, fold, C.SEEDS)
            if r0 is None or sr is None:
                continue
            seeds = sorted({x["seed"] for x in r0["seeds"]} & {x["seed"] for x in sr["seeds"]})
            if not seeds:
                continue
            r0_by = {x["seed"]: x["dev_rmse"] for x in r0["seeds"]}
            sr_by = {x["seed"]: x["dev_rmse"] for x in sr["seeds"]}
            deltas = {s: r0_by[s] - sr_by[s] for s in seeds}
            fold_rows.append(
                {
                    "fold": fold,
                    "seeds": seeds,
                    "R0_mean_dev_rmse": float(np.mean(list(r0_by.values()))),
                    "selected_mean_dev_rmse": float(np.mean(list(sr_by.values()))),
                    "delta_R0_minus_selected": float(np.mean(list(deltas.values()))),
                    "per_seed_delta": deltas,
                }
            )
        mean_delta = float(np.mean([x["delta_R0_minus_selected"] for x in fold_rows])) if fold_rows else 0.0
        h2_arms[arm] = {
            "selected_recipe": sel,
            "folds": fold_rows,
            "mean_delta": mean_delta,
            "all_available_folds_favor_selected": bool(all(x["delta_R0_minus_selected"] > 0 for x in fold_rows))
            if fold_rows
            else False,
            "promising": bool(sel != "R0" and mean_delta >= C.H2_MATERIALITY and all(x["delta_R0_minus_selected"] > 0 for x in fold_rows))
            if fold_rows
            else False,
        }
    h2 = {
        "hypothesis": "fixed supervised regularization recipes improve Lipophilicity RMSE",
        "delta": "RMSE(R0) - RMSE(selected); positive favours the selected recipe",
        "arms": h2_arms,
        "materiality": C.H2_MATERIALITY,
        "any_promising": bool(any(v["promising"] for v in h2_arms.values())),
        "monitor_used": False,
        "note": "Fold-0 selection uses seeds 0/1; a non-R0 selection is cross-checked on every fold where R0 also exists.",
    }

    # H3: SSL vs scratch, same selected recipe and seed, per fold.
    ssl_gate = C.read_json(C.SSL_GATE) if C.SSL_GATE.exists() else {"gate_pass": False, "status": "NOT_RUN"}
    a_recipe = selection.get("A", "R0")
    h3_folds = []
    for fold in C.FOLDS:
        scratch_keys = []
        ssl_keys = []
        for seed in C.SEEDS:
            scratch_key = f"F{fold}_A_{a_recipe}_s{seed}" if a_recipe != "R0" else f"F{fold}_A_R0_s{seed}"
            ssl_key = f"F{fold}_A_SSL_sel_s{seed}"
            if (C.RUNS / scratch_key / "train.json").exists() and (C.RUNS / ssl_key / "train.json").exists():
                scratch_keys.append((seed, scratch_key))
                ssl_keys.append((seed, ssl_key))
        if not scratch_keys:
            continue
        scratch_by = {s: _require_run(k)["metrics"]["dev_rmse_raw"] for s, k in scratch_keys}
        ssl_by = {s: _require_run(k)["metrics"]["dev_rmse_raw"] for s, k in ssl_keys}
        deltas = {s: scratch_by[s] - ssl_by[s] for s in sorted(scratch_by)}
        rows, y = _fold_dev_labels(fold)
        ssl_pred = np.mean([_prediction_vector(k, rows) for _s, k in ssl_keys], axis=0)
        scratch_pred = np.mean([_prediction_vector(k, rows) for _s, k in scratch_keys], axis=0)
        boot = bootstrap_rmse_diff(y, scratch_pred, ssl_pred, C.BOOTSTRAP_N, C.BOOTSTRAP_SEED + 100 + fold)
        h3_folds.append(
            {
                "fold": fold,
                "seeds": sorted(scratch_by),
                "scratch_mean_dev_rmse": float(np.mean(list(scratch_by.values()))),
                "ssl_mean_dev_rmse": float(np.mean(list(ssl_by.values()))),
                "delta_scratch_minus_ssl": float(np.mean(list(deltas.values()))),
                "per_seed_delta": deltas,
                "bootstrap": boot,
            }
        )
    h3_mean = float(np.mean([x["delta_scratch_minus_ssl"] for x in h3_folds])) if h3_folds else float("nan")
    h3_consistent = bool(all(x["delta_scratch_minus_ssl"] > 0 for x in h3_folds)) if h3_folds else False
    h3 = {
        "hypothesis": "Fit-only SSL pretraining of LCRC-A local representations improves generalization",
        "candidate": "A-SSL",
        "comparator": "A-scratch",
        "delta": "RMSE(scratch) - RMSE(SSL); positive favours SSL",
        "pretext_gate": ssl_gate,
        "folds": h3_folds,
        "mean_delta": h3_mean,
        "consistent_direction": h3_consistent,
        "materiality": C.H3_MATERIALITY,
        "promising": bool(ssl_gate.get("gate_pass") is True and h3_mean >= C.H3_MATERIALITY and h3_consistent),
        "monitor_used": False,
    }

    # E vs A beyond H1: capacity audit and optional selected-recipe check.
    e_vs_a = {
        "R0": h1,
        "selected_A_recipe": None,
    }
    if a_recipe != "R0":
        rows = []
        for fold in C.FOLDS:
            a = _arm_recipe_mean("A", a_recipe, fold, C.SEEDS)
            e = _arm_recipe_mean("E", "R0", fold, C.SEEDS)
            if a is None or e is None:
                continue
            seeds = sorted({x["seed"] for x in a["seeds"]} & {x["seed"] for x in e["seeds"]})
            a_by = {x["seed"]: x["dev_rmse"] for x in a["seeds"]}
            e_by = {x["seed"]: x["dev_rmse"] for x in e["seeds"]}
            rows.append(
                {
                    "fold": fold,
                    "delta_E_minus_A": float(np.mean([e_by[s] - a_by[s] for s in seeds])),
                    "seeds": seeds,
                }
            )
        e_vs_a["selected_A_recipe"] = {
            "recipe": a_recipe,
            "folds": rows,
            "mean_delta": float(np.mean([x["delta_E_minus_A"] for x in rows])) if rows else float("nan"),
        }

    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "fold_summary": F.fold_summary(folds),
        "selection": selection,
        "table": table,
        "H1": h1,
        "H2": h2,
        "H3": h3,
        "E_vs_A": e_vs_a,
        "monitor_used": False,
        "bootstrap": {"n": C.BOOTSTRAP_N, "seed": C.BOOTSTRAP_SEED},
    }
    C.write_json(C.ANALYSIS / "cv_results_v2.json", payload)
    return payload


# ---------------------------------------------------------------------------
# Monitor analysis
# ---------------------------------------------------------------------------


def _load_manifest_predictions() -> tuple[dict[str, Any], list[int], np.ndarray, dict[str, np.ndarray]]:
    import v2_monitor_labels as L
    import v2_predict as P

    manifest = P.verify_manifest()
    labels = L.load_monitor_labels()
    rows = labels["rows"]
    y = labels["y"]
    preds = {}
    for key, row in manifest["predictions"].items():
        data = np.load(Path(C.REPO) / row["file"])
        if [int(x) for x in data["rows"].tolist()] != rows:
            raise ValueError(f"{key} row order changed")
        preds[key] = np.asarray(data["pred"], dtype=np.float64)
    return manifest, rows, y, preds


def monitor_analysis() -> dict[str, Any]:
    manifest, rows, y, preds = _load_manifest_predictions()
    per_prediction = {}
    for key, pred in preds.items():
        row = manifest["predictions"][key]
        per_prediction[key] = {
            "arm": row["arm"],
            "recipe": row["recipe"],
            "seed": row["seed"],
            "variant": row["variant"],
            **metrics(y, pred),
        }
    by_arm: dict[str, list[dict[str, Any]]] = {}
    for key, row in per_prediction.items():
        label = f"{row['arm']}:{row['variant']}"
        by_arm.setdefault(label, []).append({"key": key, **row})
    summaries = {}
    for label, items in by_arm.items():
        summaries[label] = {
            "n_predictions": len(items),
            "rmse_mean": float(np.mean([x["rmse"] for x in items])),
            "rmse_sd": float(np.std([x["rmse"] for x in items], ddof=1)) if len(items) > 1 else 0.0,
            "mae_mean": float(np.mean([x["mae"] for x in items])),
            "r2_mean": float(np.mean([x["r2"] for x in items])),
            "bias_mean": float(np.mean([x["bias"] for x in items])),
            "items": items,
        }

    def key_for(arm: str, variant: str, seed: int | None = None) -> str | None:
        candidates = [
            k
            for k, v in manifest["predictions"].items()
            if v["arm"] == arm and v["variant"] == variant and (seed is None or v["seed"] == seed)
        ]
        return candidates[0] if candidates else None

    def mean_pred(arm: str, variant: str) -> tuple[np.ndarray, list[int]]:
        keys = [k for k, v in manifest["predictions"].items() if v["arm"] == arm and v["variant"] == variant]
        if not keys:
            raise KeyError(f"no predictions for {arm}/{variant}")
        seeds = sorted(manifest["predictions"][k]["seed"] for k in keys)
        return np.mean([preds[k] for k in keys], axis=0), seeds

    monitor_h1 = None
    if key_for("A", "scratch") and key_for("E", "scratch"):
        a, a_seeds = mean_pred("A", "scratch")
        e, e_seeds = mean_pred("E", "scratch")
        monitor_h1 = {
            "A_rmse": rmse(y, a),
            "E_rmse": rmse(y, e),
            "delta_E_minus_A": rmse(y, e) - rmse(y, a),
            "bootstrap": bootstrap_rmse_diff(y, e, a),
            "seeds": {"A": a_seeds, "E": e_seeds},
        }
    monitor_h3 = None
    if key_for("A", "scratch") and key_for("A", "ssl"):
        a, a_seeds = mean_pred("A", "scratch")
        s, s_seeds = mean_pred("A", "ssl")
        monitor_h3 = {
            "scratch_rmse": rmse(y, a),
            "ssl_rmse": rmse(y, s),
            "delta_scratch_minus_ssl": rmse(y, a) - rmse(y, s),
            "bootstrap": bootstrap_rmse_diff(y, a, s),
            "seeds": {"scratch": a_seeds, "ssl": s_seeds},
        }
    cpu_keys = [k for k, v in manifest["predictions"].items() if v["arm"] == "CPU"]
    monitor_h4 = None
    if cpu_keys and key_for("A", "scratch"):
        cpu = preds[cpu_keys[0]]
        a, a_seeds = mean_pred("A", "scratch")
        e, _ = mean_pred("E", "scratch") if key_for("E", "scratch") else (None, [])
        c, _ = mean_pred("C", "scratch") if key_for("C", "scratch") else (None, [])
        monitor_h4 = {
            "CPU_key": cpu_keys[0],
            "CPU_rmse": rmse(y, cpu),
            "A_rmse": rmse(y, a),
            "A_minus_CPU": rmse(y, a) - rmse(y, cpu),
            "bootstrap_A_minus_CPU": bootstrap_rmse_diff(y, a, cpu),
            "E_rmse": rmse(y, e) if e is not None else None,
            "C_rmse": rmse(y, c) if c is not None else None,
            "A_seeds": a_seeds,
        }
    monitor_h2 = None
    c_keys = [k for k, v in manifest["predictions"].items() if v["arm"] == "C" and v["variant"] == "scratch"]
    a_keys = [k for k, v in manifest["predictions"].items() if v["arm"] == "A" and v["variant"] == "scratch"]
    if c_keys and a_keys:
        c = np.mean([preds[k] for k in c_keys], axis=0)
        a = np.mean([preds[k] for k in a_keys], axis=0)
        monitor_h2 = {
            "note": "H2 primary verdict uses internal CV; this is the absolute C-tuned Monitor result and a C-vs-A comparator.",
            "C_rmse": rmse(y, c),
            "A_rmse": rmse(y, a),
            "C_minus_A": rmse(y, c) - rmse(y, a),
            "C_bootstrap_minus_A": bootstrap_rmse_diff(y, c, a),
        }
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "status": "MONITOR_EVALUATED_ONCE",
        "n_molecules": len(rows),
        "rows_sha256": hashlib.sha256(np.asarray(rows, dtype=np.int64).tobytes()).hexdigest(),
        "label_vector_sha256": hashlib.sha256(np.ascontiguousarray(y, dtype="<f8").tobytes()).hexdigest(),
        "manifest_sha256": C.sha256_file(C.PREDICTIONS / "MANIFEST.json"),
        "manifest_lock_sha256": (C.PREDICTIONS / "MANIFEST.sha256").read_text().strip(),
        "per_prediction": per_prediction,
        "summaries": summaries,
        "H1": monitor_h1,
        "H2": monitor_h2,
        "H3": monitor_h3,
        "H4": monitor_h4,
        "monitor_is_independent_blind_test": False,
        "reserved_labels_read": False,
        "official_valid_test_read": False,
    }
    C.write_json(C.ANALYSIS / "monitor_metrics_v2.json", payload)
    return payload


def decision(cv: Mapping[str, Any], monitor: Mapping[str, Any]) -> dict[str, Any]:
    h1 = cv["H1"]
    h2 = cv["H2"]
    h3 = cv["H3"]
    h4 = monitor.get("H4")
    verdicts = {
        "H1": {
            "verdict": "PROMISING" if h1["promising"] else "NOT_PROMISING",
            "primary_cv_mean_delta_E_minus_A": h1["mean_delta"],
            "all_folds_favor_A": h1["all_folds_favor_A"],
            "monitor_delta_E_minus_A": (monitor.get("H1") or {}).get("delta_E_minus_A"),
            "rule": f"mean delta >= {C.H1_MATERIALITY} and all 3 folds favor A",
        },
        "H2": {
            "verdict": "PROMISING" if h2["any_promising"] else "NOT_PROMISING",
            "arms": {k: {"selected": v["selected_recipe"], "mean_delta": v["mean_delta"], "promising": v["promising"]} for k, v in h2["arms"].items()},
            "rule": f"a non-R0 recipe with mean delta >= {C.H2_MATERIALITY} on Fold 0 and consistent available folds",
        },
        "H3": {
            "verdict": "PROMISING" if h3["promising"] else ("PRETEXT_NOT_LEARNED" if h3["pretext_gate"].get("gate_pass") is not True else "NOT_PROMISING"),
            "primary_cv_mean_delta_scratch_minus_ssl": h3["mean_delta"],
            "consistent_direction": h3["consistent_direction"],
            "pretext_status": h3["pretext_gate"].get("status"),
            "monitor_delta_scratch_minus_ssl": (monitor.get("H3") or {}).get("delta_scratch_minus_ssl"),
            "rule": f"pretext gate pass, mean delta >= {C.H3_MATERIALITY}, consistent direction",
        },
        "H4": {
            "verdict": "CPU_BASELINE_BETTER" if h4 and h4["A_minus_CPU"] > 0 else "LCRC_BETTER_OR_EQUAL" if h4 else "NOT_EVALUATED",
            "CPU_rmse": h4["CPU_rmse"] if h4 else None,
            "A_rmse": h4["A_rmse"] if h4 else None,
            "A_minus_CPU": h4["A_minus_CPU"] if h4 else None,
            "bootstrap_A_minus_CPU": h4["bootstrap_A_minus_CPU"] if h4 else None,
            "note": "Positive gap means the CPU chemistry baseline is better; this is an honest descriptive comparison, not a significance gate.",
        },
    }
    payload = {
        "protocol_id": C.PROTOCOL_ID,
        "created_utc": C.utcnow(),
        "verdicts": verdicts,
        "headline": {
            "H1": verdicts["H1"]["verdict"],
            "H2": verdicts["H2"]["verdict"],
            "H3": verdicts["H3"]["verdict"],
            "H4": f"CPU gap {h4['A_minus_CPU']:+.4f} RMSE" if h4 else "NOT_EVALUATED",
        },
        "monitor_is_independent_blind_test": False,
        "reserved_official_sealed": True,
    }
    C.write_json(C.ANALYSIS / "decision_v2.json", payload)
    return payload


def evaluate() -> dict[str, Any]:
    # The prediction manifest must exist and verify before any Monitor label
    # is opened.  cv_analysis is label-blind and can run independently.
    import v2_predict as P

    if (C.ANALYSIS / "monitor_metrics_v2.json").exists():
        raise PermissionError("Monitor has already been evaluated once; refusing a second evaluation")
    P.verify_manifest()
    cv = cv_analysis()
    monitor = monitor_analysis()
    dec = decision(cv, monitor)
    # A compact machine-readable table for the report.
    tables = {
        "table1_folds": cv["fold_summary"],
        "table2_cv": cv["table"],
        "table3_recipe_selection": cv["selection"],
        "table4_monitor": monitor["summaries"],
    }
    C.write_json(C.ANALYSIS / "tables_v2.json", tables)
    return {"cv": cv, "monitor": monitor, "decision": dec}


__all__ = [
    "rmse",
    "mae",
    "r2",
    "metrics",
    "bootstrap_rmse_diff",
    "cv_analysis",
    "monitor_analysis",
    "decision",
    "evaluate",
]

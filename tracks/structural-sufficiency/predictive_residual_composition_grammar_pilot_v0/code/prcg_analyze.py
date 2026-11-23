"""Analysis of the three PRCG-v0 D0 seed-0 arms.

    uv run python .../code/prcg_analyze.py

Reads only the frozen result files, verifies that the three arms are comparable
(identical initialisation / code / split / Dev rows), computes the paired
bootstrap for the two contrasts and writes ``analysis/``.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

import prcg_common as C

BOOTSTRAP = 10000
BOOTSTRAP_SEED = 20261202
ARMS = ("FREQ_GRAMMAR", "PRED_GRAMMAR", "PRED_PORT_MEAN")


def _load(arm: str, seed: int = 0) -> dict[str, Any]:
    path = C.RESULTS_DIR / f"{arm}_seed{seed}.json"
    if not path.is_file():
        raise SystemExit(f"missing result {path}")
    return C.read_json(path)


def _rmse(p: np.ndarray, y: np.ndarray) -> np.ndarray:
    """RMSE over the last axis (vectorised over bootstrap replicates)."""
    return np.sqrt(np.mean((p - y) ** 2, axis=-1))


def main() -> int:
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    runs = {arm: _load(arm) for arm in ARMS}
    config = C.load_config()

    # ---- comparability gate -------------------------------------------------
    dev_rows = [np.asarray(runs[a]["dev_rows"], np.int64) for a in ARMS]
    dev_y = [np.asarray(runs[a]["dev_y"], np.float64) for a in ARMS]
    init = {json.dumps(runs[a]["init_hashes"], sort_keys=True) for a in ARMS}
    code = {json.dumps(runs[a]["code_hashes"], sort_keys=True) for a in ARMS}
    params = {runs[a]["parameters"]["total"] for a in ARMS}
    checks = {
        "same_dev_rows": all(np.array_equal(dev_rows[0], d) for d in dev_rows[1:]),
        "same_dev_targets": all(np.array_equal(dev_y[0], d) for d in dev_y[1:]),
        "identical_initialisation": len(init) == 1,
        "identical_code": len(code) == 1,
        "identical_parameter_count": len(params) == 1,
        "single_seed": len({runs[a]["seed"] for a in ARMS}) == 1,
        "dev_rows_are_504": dev_rows[0].size == 504,
        "git_commit_recorded": all(runs[a]["git_commit"] != "unknown" for a in ARMS),
    }
    if not all(checks.values()):
        raise SystemExit(f"arms are not comparable: {checks}")

    y = dev_y[0]
    pred = {a: np.asarray(runs[a]["dev_pred"], np.float64) for a in ARMS}
    metrics = {}
    for a in ARMS:
        p = pred[a]
        metrics[a] = {"rmse": C.rmse(p, y), "mae": C.mae(p, y),
                      "fit_rmse": float(runs[a]["fit_rmse"]),
                      "fit_mae": float(runs[a]["fit_mae"]),
                      "best_epoch": int(runs[a]["best_epoch"]),
                      "epochs_run": int(runs[a]["epochs_run"]),
                      "wall_minutes": float(runs[a]["wall_seconds"]) / 60.0,
                      "z_effective_rank_participation":
                          float(runs[a]["z_effective_rank_participation"]),
                      "z_variance_mean": float(runs[a]["z_variance_mean"]),
                      "words_per_molecule": None}

    stats_by_grammar = C.read_json(C.GRAMMAR_DIR / "stats.json")
    for name, arms in (("predictive", ("PRED_GRAMMAR", "PRED_PORT_MEAN")),
                       ("frequency", ("FREQ_GRAMMAR",))):
        fit_stats = stats_by_grammar[name]["fit"]
        for arm in arms:
            metrics[arm]["words_per_molecule"] = float(fit_stats["mean_words_per_molecule"])
            metrics[arm]["n_word_types"] = int(fit_stats["n_word_types"])
            metrics[arm]["multi_port_word_fraction"] = float(
                fit_stats["multi_port_word_fraction"])
            metrics[arm]["cross_scaffold_reuse"] = float(
                fit_stats["type_cross_scaffold_reuse_fraction"])

    # ---- paired bootstrap ---------------------------------------------------
    rng = np.random.default_rng(BOOTSTRAP_SEED)
    n = y.size
    idx = rng.integers(0, n, size=(BOOTSTRAP, n))
    yb = y[idx]
    pb = {a: pred[a][idx] for a in ARMS}
    rmse_b = {a: _rmse(pb[a], yb) for a in ARMS}
    mae_b = {a: np.mean(np.abs(pb[a] - yb), axis=-1) for a in ARMS}
    contrasts = {
        "delta_discovery": rmse_b["FREQ_GRAMMAR"] - rmse_b["PRED_GRAMMAR"],
        "delta_port": rmse_b["PRED_PORT_MEAN"] - rmse_b["PRED_GRAMMAR"],
        "delta_freq_vs_port_mean": rmse_b["FREQ_GRAMMAR"] - rmse_b["PRED_PORT_MEAN"],
    }
    point = {
        "delta_discovery": metrics["FREQ_GRAMMAR"]["rmse"] - metrics["PRED_GRAMMAR"]["rmse"],
        "delta_port": metrics["PRED_PORT_MEAN"]["rmse"] - metrics["PRED_GRAMMAR"]["rmse"],
        "delta_freq_vs_port_mean": metrics["FREQ_GRAMMAR"]["rmse"] -
        metrics["PRED_PORT_MEAN"]["rmse"],
    }

    def ci(values: np.ndarray) -> dict[str, float]:
        return {"lo": float(np.percentile(values, 2.5)),
                "hi": float(np.percentile(values, 97.5)),
                "mean": float(values.mean()),
                "frac_positive": float((values > 0).mean())}

    bootstrap = {
        "bootstrap": BOOTSTRAP, "seed": BOOTSTRAP_SEED, "n_dev": int(n),
        "point": point,
        "rmse_ci": {a: ci(rmse_b[a]) for a in ARMS},
        "mae_ci": {a: ci(mae_b[a]) for a in ARMS},
        "contrast_ci": {k: ci(v) for k, v in contrasts.items()},
    }

    # ---- descriptive diagnostics -------------------------------------------
    resid = {a: pred[a] - y for a in ARMS}
    corr = {}
    for a in ARMS:
        for b in ARMS:
            if a < b:
                corr[f"{a}~{b}"] = float(np.corrcoef(resid[a], resid[b])[0, 1])
    gaps = {a: {"dev_minus_fit_rmse": metrics[a]["rmse"] - metrics[a]["fit_rmse"],
                "fit_rmse": metrics[a]["fit_rmse"],
                "dev_over_fit": metrics[a]["rmse"] / metrics[a]["fit_rmse"]} for a in ARMS}
    # per-molecule paired diagnostics: how many Dev molecules each arm wins
    abs_err = {a: np.abs(pred[a] - y) for a in ARMS}
    paired = {}
    for x, z in (("FREQ_GRAMMAR", "PRED_GRAMMAR"), ("PRED_PORT_MEAN", "PRED_GRAMMAR")):
        d = abs_err[x] - abs_err[z]      # > 0  => the second arm is better
        paired[f"{x}_minus_{z}"] = {
            "mean_abs_err_diff": float(d.mean()),
            "median_abs_err_diff": float(np.median(d)),
            "fraction_molecules_second_arm_better": float((d > 0).mean()),
            "fraction_molecules_tied": float((d == 0).mean()),
        }
    sq = {a: (pred[a] - y) ** 2 for a in ARMS}
    paired_sq = {}
    for x, z in (("FREQ_GRAMMAR", "PRED_GRAMMAR"), ("PRED_PORT_MEAN", "PRED_GRAMMAR")):
        d = sq[x] - sq[z]
        paired_sq[f"{x}_minus_{z}"] = {
            "mean_sq_err_diff": float(d.mean()),
            "median_sq_err_diff": float(np.median(d)),
        }

    # ---- lexicon description ------------------------------------------------
    trace = C.read_json(C.GRAMMAR_DIR / "discovery_trace.json")
    stats = C.read_json(C.GRAMMAR_DIR / "stats.json")
    lexicon = {}
    for name, arm in (("predictive", "PRED_GRAMMAR"), ("frequency", "FREQ_GRAMMAR")):
        rounds = trace[name]
        rules = []
        for r in rounds:
            rules.append({
                "index": r["rule"]["index"],
                "parent_cert": r["rule"]["parent_cert"],
                "parent_size": r["rule"]["parent_size"],
                "pattern": r["rule"]["pattern"],
                "n_crossing_bonds": r["rule"]["n_crossing_bonds"],
                "selected_improvement": r.get("selected_improvement"),
                "selected_mean_improvement": r.get("selected_mean_improvement"),
                "selected_n_molecules": r.get("selected_n_molecules"),
                "selected_n_scaffolds": r.get("selected_n_scaffolds"),
                "applications_total": r.get("applications_total"),
            })
        by_gain = sorted(rules, key=lambda d: -(d["selected_mean_improvement"] or 0.0))
        lexicon[name] = {
            "n_rules": len(rules),
            "fit": stats[name]["fit"],
            "dev": stats[name]["dev"],
            "rules": rules,
            "top_rules_by_mean_improvement": by_gain[:8],
            "size_histogram": stats[name]["fit"]["size_histogram"],
            "multi_port_word_fraction": stats[name]["fit"]["multi_port_word_fraction"],
            "type_cross_scaffold_reuse_fraction":
                stats[name]["fit"]["type_cross_scaffold_reuse_fraction"],
            "median_type_support": stats[name]["fit"]["type_molecule_support_median"],
        }

    # ---- route label --------------------------------------------------------
    th = config["route_classification"]
    d_disc = point["delta_discovery"]
    d_port = point["delta_port"]
    best_rmse = metrics["PRED_GRAMMAR"]["rmse"]
    nondegenerate = (metrics["PRED_GRAMMAR"]["z_effective_rank_participation"] > 1.5 and
                     metrics["PRED_GRAMMAR"]["z_variance_mean"] > 1e-3)
    if best_rmse <= th["promising_max_rmse"] and d_disc >= th["promising_min_delta_discovery"] \
            and d_port >= th["promising_min_delta_port"] and nondegenerate:
        label = "PROMISING_PREDICTIVE_GRAMMAR"
    elif d_disc >= th["promising_min_delta_discovery"]:
        label = "LEXICON_SIGNAL_WITHOUT_PORT_GAIN"
    elif d_port >= th["promising_min_delta_port"]:
        label = "PORT_SIGNAL_WITHOUT_PREDICTIVE_LEXICON"
    else:
        label = "INCONCLUSIVE_SINGLE_SEED"

    historical = config["historical_d0_rmse"]
    route = {
        "label": label,
        "delta_discovery": d_disc,
        "delta_port": d_port,
        "rmse_PRED_GRAMMAR": best_rmse,
        "rmse_FREQ_GRAMMAR": metrics["FREQ_GRAMMAR"]["rmse"],
        "rmse_PRED_PORT_MEAN": metrics["PRED_PORT_MEAN"]["rmse"],
        "thresholds": th,
        "nondegenerate": bool(nondegenerate),
        "vs_R2_SENTENCE_s0": best_rmse - historical["R2_SENTENCE_BASE_S0"],
        "vs_R2_SENTENCE_s1": best_rmse - historical["R2_SENTENCE_BASE_S1"],
        "vs_GINE": best_rmse - historical["GINE_REF"],
        "competitive_with_R2_sentence": bool(
            best_rmse <= historical["R2_SENTENCE_BASE_S0"] + 0.01),
        "competitive_with_GINE": bool(best_rmse <= historical["GINE_REF"] + 0.01),
    }

    summary = {
        "protocol_id": config["protocol_id"],
        "git_commit": runs["PRED_GRAMMAR"]["git_commit"],
        "code_sha": runs["PRED_GRAMMAR"]["code_sha"],
        "split": runs["PRED_GRAMMAR"]["split"],
        "comparability": checks,
        "metrics": metrics,
        "train_dev_gap": gaps,
        "paired_dev_diagnostics": paired,
        "paired_dev_squared": paired_sq,
        "residual_correlation": corr,
        "route": route,
        "bootstrap": bootstrap,
        "lexicon": lexicon,
        "device": {a: runs[a]["device"] for a in ARMS},
        "wall_minutes_total": float(sum(runs[a]["wall_seconds"] for a in ARMS) / 60.0),
        "gpu_hours": 0.0,
        "finished_utc": {a: runs[a]["finished_utc"] for a in ARMS},
    }
    C.write_json(C.ANALYSIS_DIR / "summary.json", summary)
    C.write_json(C.ANALYSIS_DIR / "bootstrap.json", bootstrap)

    rows = []
    for a in ARMS:
        rows.append({"arm": a,
                     "dev_rmse": round(metrics[a]["rmse"], 4),
                     "dev_rmse_lo": round(bootstrap["rmse_ci"][a]["lo"], 4),
                     "dev_rmse_hi": round(bootstrap["rmse_ci"][a]["hi"], 4),
                     "dev_mae": round(metrics[a]["mae"], 4),
                     "fit_rmse": round(metrics[a]["fit_rmse"], 4),
                     "best_epoch": metrics[a]["best_epoch"],
                     "epochs_run": metrics[a]["epochs_run"],
                     "wall_minutes": round(metrics[a]["wall_minutes"], 2),
                     "words_per_molecule": metrics[a]["words_per_molecule"],
                     "z_participation": round(metrics[a]["z_effective_rank_participation"], 3)})
    C.write_csv(C.ANALYSIS_DIR / "arms.csv", rows)
    C.write_csv(C.ANALYSIS_DIR / "contrasts.csv", [
        {"contrast": k, "point": round(point[k], 5),
         "lo": round(bootstrap["contrast_ci"][k]["lo"], 5),
         "hi": round(bootstrap["contrast_ci"][k]["hi"], 5),
         "frac_positive": round(bootstrap["contrast_ci"][k]["frac_positive"], 3)}
        for k in point])

    print(json.dumps({"arms": rows, "contrasts": C.read_json(
        C.ANALYSIS_DIR / "contrasts.csv") if False else
        [{"contrast": k, "point": round(point[k], 4),
          "ci": [round(bootstrap["contrast_ci"][k]["lo"], 4),
                 round(bootstrap["contrast_ci"][k]["hi"], 4)],
          "frac_positive": round(bootstrap["contrast_ci"][k]["frac_positive"], 3)}
         for k in point],
        "route": {k: route[k] for k in
                  ("label", "delta_discovery", "delta_port", "rmse_PRED_GRAMMAR",
                   "nondegenerate", "vs_R2_SENTENCE_s0", "vs_GINE")}},
        indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
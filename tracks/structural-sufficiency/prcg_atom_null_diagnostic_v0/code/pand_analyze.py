"""PRCG-ATOM-NULL part C/D: compare ATOM_ONLY with the frozen PRCG-v0 arms.

Reads the committed PRCG-v0 result files and the one new ATOM_ONLY run, checks
that the four arms share split / Dev rows / targets / width / initialisation,
computes the paired bootstrap over the 504 Dev molecules, records the atom
coverage of each lexicon, and applies the pre-registered decision rule.

No model is trained here.  No Valid/Test label is read.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import pand_common as P

ARMS = ("FREQ_GRAMMAR", "PRED_GRAMMAR", "PRED_PORT_MEAN", "ATOM_ONLY")
PARENT_ARMS = ("FREQ_GRAMMAR", "PRED_GRAMMAR", "PRED_PORT_MEAN")


def _load(arm: str) -> dict[str, Any]:
    if arm == "ATOM_ONLY":
        path = P.RESULTS_DIR / "ATOM_ONLY_seed0.json"
    else:
        path = P.PARENT_DIR / "results" / f"{arm}_seed0.json"
    if not path.is_file():
        raise SystemExit(f"missing result {path}")
    return P.read_json(path)


def _rmse_vec(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.sqrt(np.mean((pred - true) ** 2, axis=-1))


def _mae_vec(pred: np.ndarray, true: np.ndarray) -> np.ndarray:
    return np.mean(np.abs(pred - true), axis=-1)


def _ci(values: np.ndarray) -> dict[str, float]:
    return {
        "lo": float(np.percentile(values, 2.5)),
        "hi": float(np.percentile(values, 97.5)),
        "mean": float(values.mean()),
        "median": float(np.median(values)),
        "frac_positive": float((values > 0).mean()),
        "frac_negative": float((values < 0).mean()),
    }


def _atom_counts(bundle: P.D.DataBundle, rows: np.ndarray) -> np.ndarray:
    ptr = np.asarray(bundle.raw["node_ptr"], np.int64)
    return ptr[rows + 1] - ptr[rows]


def _coverage_from_stats(stats: Mapping[str, Any], grammar: str, side: str,
                         expected_atoms: int) -> dict[str, Any]:
    d = stats[grammar][side]
    hist = {int(k): int(v) for k, v in d["size_histogram"].items()}
    n_atoms = int(sum(k * v for k, v in hist.items()))
    singleton_atoms = int(hist.get(1, 0))
    if n_atoms != int(expected_atoms):
        raise ValueError(f"{grammar}/{side}: {n_atoms} != {expected_atoms} atoms")
    return {
        "covered_atom_fraction": 1.0,
        "singleton_atom_fraction": float(singleton_atoms / max(1, n_atoms)),
        "singleton_atoms": singleton_atoms,
        "n_atoms": n_atoms,
        "n_words": int(d["n_words"]),
        "words_per_molecule": float(d["mean_words_per_molecule"]),
        "size_histogram": {str(k): v for k, v in sorted(hist.items())},
    }


def main() -> int:
    cfg = P.load_config()
    parent_cfg = P.parent_config()
    bundle = P.D.load_bundle()
    stats = P.read_json(P.PARENT_DIR / "grammar" / "stats.json")
    ridge = P.read_json(P.ANALYSIS_DIR / "ridge_audit.json")
    atom_checks = P.read_json(P.ANALYSIS_DIR / "atom_checks.json")

    runs = {a: _load(a) for a in ARMS}
    fit_rows = np.asarray(bundle.fit_rows, np.int64)
    dev_rows = np.asarray(bundle.eval_rows, np.int64)
    y = bundle.labels[dev_rows]
    pred = {a: np.asarray(runs[a]["dev_pred"], np.float64) for a in ARMS}

    # ---- comparability gate -------------------------------------------------
    dev_rows_equal = all(np.array_equal(np.asarray(runs[a]["dev_rows"], np.int64),
                                        dev_rows) for a in ARMS)
    dev_y_equal = all(np.array_equal(np.asarray(runs[a]["dev_y"], np.float64), y)
                      for a in ARMS)
    init = {json.dumps(runs[a]["init_hashes"], sort_keys=True) for a in ARMS}
    params = {int(runs[a]["parameters"]["total"]) for a in ARMS}
    code_hashes = {json.dumps(runs[a]["code_hashes"], sort_keys=True) for a in ARMS}
    checks = {
        "same_dev_rows": bool(dev_rows_equal),
        "same_dev_targets": bool(dev_y_equal),
        "same_split_sha256": len({runs[a]["split"]["sha256"] for a in ARMS}) == 1,
        "identical_initialisation_all_four_arms": len(init) == 1,
        "identical_parameter_count": len(params) == 1,
        "identical_parent_code_hash": len(code_hashes) == 1,
        "single_seed": len({int(runs[a]["seed"]) for a in ARMS}) == 1,
        "dev_rows_are_504": int(dev_rows.size) == 504,
        "atom_grammar_has_zero_rules": int(runs["ATOM_ONLY"]["grammar_rules"]) == 0,
        "atom_checks_all_pass": bool(atom_checks.get("all_pass", False)),
    }
    if not all(checks.values()):
        raise SystemExit(f"comparability gate failed: {checks}")

    metrics: dict[str, Any] = {}
    for a in ARMS:
        r = runs[a]
        metrics[a] = {
            "rmse": P.rmse(pred[a], y),
            "mae": P.mae(pred[a], y),
            "fit_rmse": float(r["fit_rmse"]),
            "fit_mae": float(r["fit_mae"]),
            "best_epoch": int(r["best_epoch"]),
            "epochs_run": int(r["epochs_run"]),
            "wall_minutes": float(r["wall_seconds"]) / 60.0,
            "z_effective_rank_participation": float(r["z_effective_rank_participation"]),
            "z_variance_mean": float(r["z_variance_mean"]),
            "words_per_molecule": None,
            "n_word_types": None,
            "singleton_atom_fraction": None,
        }

    # atom counts (exact, from the label-free raw structure pointer)
    n_atoms_fit = int(_atom_counts(bundle, fit_rows).sum())
    n_atoms_dev = int(_atom_counts(bundle, dev_rows).sum())
    coverage = {}
    for grammar, arm in (("frequency", "FREQ_GRAMMAR"),
                         ("predictive", "PRED_GRAMMAR"),
                         ("predictive", "PRED_PORT_MEAN")):
        coverage[arm] = {
            "fit": _coverage_from_stats(stats, grammar, "fit", n_atoms_fit),
            "dev": _coverage_from_stats(stats, grammar, "dev", n_atoms_dev),
        }
        metrics[arm]["words_per_molecule"] = coverage[arm]["fit"]["words_per_molecule"]
        metrics[arm]["n_word_types"] = int(stats[grammar]["fit"]["n_word_types"])
        metrics[arm]["singleton_atom_fraction"] = coverage[arm]["fit"]["singleton_atom_fraction"]
    coverage["ATOM_ONLY"] = {
        "fit": {"covered_atom_fraction": 1.0, "singleton_atom_fraction": 1.0,
                "singleton_atoms": n_atoms_fit, "n_atoms": n_atoms_fit,
                "n_words": n_atoms_fit,
                "words_per_molecule": float(n_atoms_fit / fit_rows.size),
                "size_histogram": {"1": n_atoms_fit}},
        "dev": {"covered_atom_fraction": 1.0, "singleton_atom_fraction": 1.0,
                "singleton_atoms": n_atoms_dev, "n_atoms": n_atoms_dev,
                "n_words": n_atoms_dev,
                "words_per_molecule": float(n_atoms_dev / dev_rows.size),
                "size_histogram": {"1": n_atoms_dev}},
    }
    metrics["ATOM_ONLY"]["words_per_molecule"] = coverage["ATOM_ONLY"]["fit"]["words_per_molecule"]
    metrics["ATOM_ONLY"]["singleton_atom_fraction"] = 1.0
    # atom-only has one type per atom core certificate; this is only descriptive
    # (the token count is what matters for the comparison)
    metrics["ATOM_ONLY"]["n_word_types"] = None

    # ---- paired bootstrap ---------------------------------------------------
    n = int(y.size)
    idx = P.bootstrap_indices(n)
    yb = y[idx]
    pb = {a: pred[a][idx] for a in ARMS}
    rmse_b = {a: _rmse_vec(pb[a], yb) for a in ARMS}
    mae_b = {a: _mae_vec(pb[a], yb) for a in ARMS}
    point = {
        "delta_merge": metrics["ATOM_ONLY"]["rmse"] - metrics["PRED_GRAMMAR"]["rmse"],
        "delta_atom_vs_freq": metrics["ATOM_ONLY"]["rmse"] - metrics["FREQ_GRAMMAR"]["rmse"],
        "delta_atom_vs_port_mean": metrics["ATOM_ONLY"]["rmse"]
        - metrics["PRED_PORT_MEAN"]["rmse"],
        "delta_discovery_reference": metrics["FREQ_GRAMMAR"]["rmse"]
        - metrics["PRED_GRAMMAR"]["rmse"],
    }
    boot_contrast = {
        "delta_merge": rmse_b["ATOM_ONLY"] - rmse_b["PRED_GRAMMAR"],
        "delta_atom_vs_freq": rmse_b["ATOM_ONLY"] - rmse_b["FREQ_GRAMMAR"],
        "delta_atom_vs_port_mean": rmse_b["ATOM_ONLY"] - rmse_b["PRED_PORT_MEAN"],
        "delta_discovery_reference": rmse_b["FREQ_GRAMMAR"] - rmse_b["PRED_GRAMMAR"],
    }
    mae_contrast = {
        "delta_merge": mae_b["ATOM_ONLY"] - mae_b["PRED_GRAMMAR"],
        "delta_atom_vs_freq": mae_b["ATOM_ONLY"] - mae_b["FREQ_GRAMMAR"],
        "delta_atom_vs_port_mean": mae_b["ATOM_ONLY"] - mae_b["PRED_PORT_MEAN"],
    }
    bootstrap = {
        "bootstrap": int(idx.shape[0]), "seed": P.BOOTSTRAP_SEED, "n_dev": n,
        "scheme": "paired over Dev molecules, common resample indices for all four arms",
        "point": point,
        "rmse_ci": {a: _ci(rmse_b[a]) for a in ARMS},
        "mae_ci": {a: _ci(mae_b[a]) for a in ARMS},
        "contrast_ci": {k: _ci(v) for k, v in boot_contrast.items()},
        "mae_contrast_ci": {k: _ci(v) for k, v in mae_contrast.items()},
        "residual_correlation": {
            f"{a}~{b}": float(np.corrcoef(pred[a] - y, pred[b] - y)[0, 1])
            for i, a in enumerate(ARMS) for b in ARMS[i + 1:]
        },
    }

    # per-molecule paired diagnostics
    abs_err = {a: np.abs(pred[a] - y) for a in ARMS}
    sq_err = {a: (pred[a] - y) ** 2 for a in ARMS}
    d_abs = abs_err["ATOM_ONLY"] - abs_err["PRED_GRAMMAR"]
    d_sq = sq_err["ATOM_ONLY"] - sq_err["PRED_GRAMMAR"]
    paired = {
        "atom_minus_pred": {
            "mean_abs_err_diff": float(d_abs.mean()),
            "median_abs_err_diff": float(np.median(d_abs)),
            "fraction_atom_better": float((d_abs < 0).mean()),
            "fraction_atom_worse": float((d_abs > 0).mean()),
            "fraction_tied": float((d_abs == 0).mean()),
            "mean_sq_err_diff": float(d_sq.mean()),
            "median_sq_err_diff": float(np.median(d_sq)),
        },
        "atom_minus_freq": {
            "mean_abs_err_diff": float((abs_err["ATOM_ONLY"] - abs_err["FREQ_GRAMMAR"]).mean()),
            "fraction_atom_better": float(((abs_err["ATOM_ONLY"]
                                            - abs_err["FREQ_GRAMMAR"]) < 0).mean()),
        },
        "atom_minus_port_mean": {
            "mean_abs_err_diff": float((abs_err["ATOM_ONLY"]
                                        - abs_err["PRED_PORT_MEAN"]).mean()),
            "fraction_atom_better": float(((abs_err["ATOM_ONLY"]
                                            - abs_err["PRED_PORT_MEAN"]) < 0).mean()),
        },
    }

    # ---- ridge-audit summary / caveat --------------------------------------
    rsum = ridge["summary"]
    rint = ridge["replay_integrity"]
    min_spearman = min(float(r["ranking"].get("spearman", 1.0)) for r in ridge["rounds"])
    top1_changes = sum(1 for r in ridge["rounds"] if r["ranking"].get("top1_changed"))
    top5_min = min(int(r["ranking"].get("top5_overlap", 5)) for r in ridge["rounds"])
    selected_den_ratios = [
        x["selected_denominator"][s]["den_ratio_strict_over_old"]
        for x in ridge["rounds"] for s in ("0", "1")
    ]
    ridge_summary = {
        "replay_rounds": int(rint["n_rounds"]),
        "replay_pattern_mismatches": int(rint["selected_pattern_mismatches"]),
        "replay_max_selected_gain_diff_vs_trace": float(
            rint["max_abs_selected_gain_diff_vs_trace"]),
        "n_selected_both_positive_original": int(rsum["n_both_positive_old"]),
        "n_selected_both_positive_strict": int(rsum["n_both_positive_strict"]),
        "n_selected_fold_sign_changes": int(rsum["n_selected_fold_sign_changes"]),
        "n_constraint_aware_selection_changes": int(
            rsum["n_constraint_aware_selection_changes"]),
        "max_abs_selected_mean_gain_diff": float(rsum["max_abs_selected_mean_gain_diff"]),
        "mean_abs_selected_mean_gain_diff": float(rsum["mean_abs_selected_mean_gain_diff"]),
        "min_spearman": float(min_spearman),
        "top1_changes": int(top1_changes),
        "min_top5_overlap": int(top5_min),
        "selected_denominator_ratio_min": float(min(selected_den_ratios)),
        "selected_denominator_ratio_median": float(np.median(selected_den_ratios)),
        "selected_denominator_ratio_max": float(max(selected_den_ratios)),
        "denominator_gap_abs_max_over_rounds": float(
            rsum["denominator_gap"]["abs_max_over_rounds"]),
        "denominator_gap_abs_median_over_rounds": float(
            rsum["denominator_gap"]["abs_median_over_rounds"]),
        "direct_vs_schur_max_abs_diff": float(
            rint["max_abs_direct_minus_schur_formula"]),
        "substantive_interpretation_change": bool(
            rsum["n_selected_fold_sign_changes"] > 0
            or rsum["n_constraint_aware_selection_changes"] > 0
            or rsum["max_abs_selected_mean_gain_diff"] > 1e-3
            or min_spearman < 0.99),
        "interpretation": (
            "The strict augmented-ridge denominator adds "
            "lambda*||S||_P^2 to the frozen denominator. The selected rules keep "
            "both fold gains positive, no selected-rule sign or selection changes, "
            "and the candidate ordering is essentially unchanged; the earlier "
            "predictive-rule discovery interpretation does not need contraction."
            if not (rsum["n_selected_fold_sign_changes"] > 0
                    or rsum["n_constraint_aware_selection_changes"] > 0
                    or rsum["max_abs_selected_mean_gain_diff"] > 1e-3
                    or min_spearman < 0.99)
            else
            "The strict augmented-ridge score materially changes the selected-rule "
            "screen; the earlier predictive-rule discovery interpretation must be "
            "contracted."
        ),
    }

    # ---- decision -----------------------------------------------------------
    delta = float(point["delta_merge"])
    ci = bootstrap["contrast_ci"]["delta_merge"]
    clear = bool(delta >= 0.01 and ci["lo"] > 0.0)
    not_worse = bool(delta <= 0.0)
    if not_worse:
        decision = "ATOM_ONLY_MATCHES_OR_BEATS_PRED"
        decision_text = (
            "ATOM_ONLY is not worse than PRED_GRAMMAR on Dev RMSE. There is no "
            "evidence in this round that merging into structural words has "
            "incremental predictive value. Stop the PRCG composition-model route."
        )
    elif clear:
        decision = "PRED_CLEAR_ADVANTAGE_OVER_ATOM_ONLY"
        decision_text = (
            "PRED_GRAMMAR is clearly better than ATOM_ONLY (delta_merge >= 0.01 "
            "and the paired 95% CI excludes zero). This is an initial signal that "
            "structural merging may be useful; it does not establish a word-level "
            "contribution law."
        )
    else:
        decision = "PRED_NUMERICALLY_BETTER_BUT_NOT_CLEAR"
        decision_text = (
            "PRED_GRAMMAR is numerically better than ATOM_ONLY, but the advantage "
            "does not meet the pre-registered clear-advantage rule (delta_merge "
            ">= 0.01 and CI lower bound > 0). Treat this as no clear evidence of "
            "incremental merge value in this round."
        )
    route = {
        "label": decision,
        "decision_text": decision_text,
        "delta_merge": delta,
        "delta_merge_ci": ci,
        "clear_advantage_rule": "delta_merge >= 0.01 and paired 95% CI lower > 0",
        "not_worse_rule": "RMSE(ATOM_ONLY) <= RMSE(PRED_GRAMMAR)",
        "rmse_atom_only": metrics["ATOM_ONLY"]["rmse"],
        "rmse_pred_grammar": metrics["PRED_GRAMMAR"]["rmse"],
        "stop_prcg_composition_route": bool(not_worse or not clear),
        "authorized_next_experiment": None,
        "no_automatic_next_step": [
            "PRCG-v1", "second seed", "D1", "Valid", "Test", "any other new network"
        ],
        "ridge_audit_caveat": ridge_summary,
    }

    summary = {
        "protocol_id": P.PROTOCOL_ID,
        "generated_utc": P.C.now_utc(),
        "git_commit_runs": {a: runs[a]["git_commit"] for a in ARMS},
        "new_code_hashes": P.new_code_hashes(),
        "parent_config_sha256": P.PARENT_CONFIG_SHA256,
        "compare_gate": checks,
        "split": runs["ATOM_ONLY"]["split"],
        "metrics": metrics,
        "atom_coverage": coverage,
        "bootstrap": bootstrap,
        "paired_dev_diagnostics": paired,
        "ridge_audit": ridge_summary,
        "route": route,
        "atom_checks": {
            "n_checks": int(atom_checks["n_checks"]),
            "n_passed": int(atom_checks["n_passed"]),
            "all_pass": bool(atom_checks["all_pass"]),
            "trained_checkpoint_loaded": bool(atom_checks.get("trained_checkpoint_loaded", False)),
            "wall_seconds": float(atom_checks["wall_seconds"]),
        },
        "resource": {
            "atom_only_wall_minutes": metrics["ATOM_ONLY"]["wall_minutes"],
            "atom_only_peak_rss_kb": (
                P.read_json(P.RESULTS_DIR / "ATOM_ONLY_seed0_manifest.json")["runtime"]
                ["peak_rss_kb"]
                if (P.RESULTS_DIR / "ATOM_ONLY_seed0_manifest.json").is_file() else None),
            "gpu_hours": 0.0,
        },
    }
    P.write_json(P.ANALYSIS_DIR / "summary.json", summary)
    P.write_json(P.ANALYSIS_DIR / "bootstrap.json", bootstrap)

    rows = []
    for a in ARMS:
        m = metrics[a]
        rows.append({
            "arm": a,
            "dev_rmse": round(m["rmse"], 4),
            "dev_rmse_lo": round(bootstrap["rmse_ci"][a]["lo"], 4),
            "dev_rmse_hi": round(bootstrap["rmse_ci"][a]["hi"], 4),
            "dev_mae": round(m["mae"], 4),
            "fit_rmse": round(m["fit_rmse"], 4),
            "best_epoch": m["best_epoch"],
            "epochs_run": m["epochs_run"],
            "wall_minutes": round(m["wall_minutes"], 2),
            "words_per_molecule": round(m["words_per_molecule"], 4),
            "singleton_atom_fraction": round(m["singleton_atom_fraction"], 4),
            "z_participation": round(m["z_effective_rank_participation"], 3),
        })
    P.write_csv(P.ANALYSIS_DIR / "arms.csv", rows)
    P.write_csv(P.ANALYSIS_DIR / "contrasts.csv", [
        {"contrast": k, "point": round(point[k], 6),
         "lo": round(bootstrap["contrast_ci"][k]["lo"], 6),
         "hi": round(bootstrap["contrast_ci"][k]["hi"], 6),
         "frac_positive": round(bootstrap["contrast_ci"][k]["frac_positive"], 3)}
        for k in point
    ])
    print(json.dumps({"arms": rows, "contrasts": [
        {"contrast": k, "point": round(point[k], 6),
         "ci": [round(bootstrap["contrast_ci"][k]["lo"], 6),
                round(bootstrap["contrast_ci"][k]["hi"], 6)],
         "frac_positive": round(bootstrap["contrast_ci"][k]["frac_positive"], 3)}
        for k in point], "route": route["label"], "decision": route["decision_text"],
        "ridge": {k: ridge_summary[k] for k in (
            "n_selected_both_positive_original", "n_selected_both_positive_strict",
            "n_selected_fold_sign_changes", "n_constraint_aware_selection_changes",
            "max_abs_selected_mean_gain_diff", "min_spearman",
            "substantive_interpretation_change")}}, indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
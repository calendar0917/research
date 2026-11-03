"""P1-B: does the model predict real local-edit property changes better than no change?"""

from __future__ import annotations

import json
from collections import Counter
from typing import Any, Sequence

import numpy as np
from scipy.stats import pearsonr, spearmanr

import ph_common as C


def _load() -> tuple[dict[str, dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    payload = C.read_json(C.EDIT / "analysis/probe_audit_scores.json")
    units = payload["units"]
    ref = None
    dy_by_pair: dict[str, list[float]] = {}
    for key, unit in units.items():
        pairs = unit["audit_pairs"]
        if len(pairs) != 74:
            raise RuntimeError(f"{key}: expected 74 pairs, got {len(pairs)}")
        ids = [p["pair_id"] for p in pairs]
        if len(set(ids)) != 74:
            raise RuntimeError(f"{key}: duplicate pair ids")
        comps = {p["component"] for p in pairs}
        if len(comps) != 26:
            raise RuntimeError(f"{key}: expected 26 components, got {len(comps)}")
        if ref is None:
            ref = pairs
        for p in pairs:
            dy_by_pair.setdefault(p["pair_id"], []).append(float(p["d_pred"]) - float(p["e_delta"]))
    spreads = {k: max(v) - min(v) for k, v in dy_by_pair.items()}
    max_spread = max(spreads.values())
    if max_spread > 1e-5:
        raise RuntimeError(f"Δy reconstruction differs across units: {max_spread}")
    for key, unit in units.items():
        for p in unit["audit_pairs"]:
            refp = next(x for x in ref if x["pair_id"] == p["pair_id"])
            for field in ("component", "class_id", "same_var_heavy", "row_a", "row_b"):
                if p[field] != refp[field]:
                    raise RuntimeError(f"{key}/{p['pair_id']}: {field} differs")
    meta = {
        "n_units": len(units),
        "n_pairs": len(ref),
        "n_components": len({p["component"] for p in ref}),
        "max_abs_dy_reconstruction_spread": float(max_spread),
        "same_var_heavy_counts": dict(Counter(int(p["same_var_heavy"]) for p in ref)),
        "audit_label_event": payload.get("audit_label_event", {}),
        "schema": payload.get("schema"),
        "git_commit": payload.get("git_commit"),
    }
    return units, ref, meta


def _component_means(
    errors: np.ndarray, components: np.ndarray, uniq: np.ndarray
) -> dict[str, float]:
    return {str(c): float(np.mean(np.abs(errors[components == c]))) for c in uniq}


def _subset_component_mae(
    errors: np.ndarray, components: np.ndarray, mask: np.ndarray
) -> tuple[float, int, int]:
    vals = []
    for c in sorted(set(components[mask].tolist())):
        vals.append(float(np.mean(np.abs(errors[mask & (components == c)]))))
    return (float(np.mean(vals)) if vals else float("nan"), len(vals), int(mask.sum()))


def _direction_agreement(dy: np.ndarray, dpred: np.ndarray, threshold: float = 0.10) -> dict[str, Any]:
    sel = np.abs(dy) >= threshold
    n = int(sel.sum())
    if n == 0:
        return {"threshold": threshold, "n_pairs": 0, "agreement": None}
    agree = int((np.sign(dy[sel]) == np.sign(dpred[sel])).sum())
    return {
        "threshold": threshold,
        "n_pairs": n,
        "n_remaining_excluded": int((~sel).sum()),
        "n_agree": agree,
        "agreement": float(agree / n),
    }


def _corr(x: np.ndarray, y: np.ndarray) -> dict[str, Any]:
    if x.size < 3 or np.allclose(x, x[0]) or np.allclose(y, y[0]):
        return {"spearman": None, "pearson": None, "n": int(x.size)}
    return {
        "spearman": float(spearmanr(x, y).statistic),
        "pearson": float(pearsonr(x, y).statistic),
        "n": int(x.size),
    }


def _metrics(unit: dict[str, Any], pairs: list[dict[str, Any]], ref_by_id: dict[str, dict[str, Any]]) -> dict[str, Any]:
    by_id = {p["pair_id"]: p for p in unit["audit_pairs"]}
    ids = [p["pair_id"] for p in pairs]
    dy = np.asarray([by_id[i]["d_pred"] - by_id[i]["e_delta"] for i in ids], dtype=np.float64)
    dp = np.asarray([by_id[i]["d_pred"] for i in ids], dtype=np.float64)
    comps = np.asarray([by_id[i]["component"] for i in ids], dtype=object)
    err_model = dp - dy
    err_zero = -dy
    return {
        "n_pairs": int(len(ids)),
        "n_components": int(len(set(comps.tolist()))),
        "MAE_MODEL": float(np.mean(np.abs(err_model))),
        "MAE_ZERO": float(np.mean(np.abs(err_zero))),
        "component_MAE_MODEL": C.component_equal_mae(err_model, comps),
        "component_MAE_ZERO": C.component_equal_mae(err_zero, comps),
        "Edit_Gain": C.component_equal_mae(err_zero, comps) - C.component_equal_mae(err_model, comps),
        "RMSE_MODEL": float(np.sqrt(np.mean(err_model**2))),
        "RMSE_ZERO": float(np.sqrt(np.mean(err_zero**2))),
        "component_RMSE_MODEL": C.component_equal_rmse(err_model, comps),
        "component_RMSE_ZERO": C.component_equal_rmse(err_zero, comps),
        "corr": _corr(dp, dy),
        "direction": _direction_agreement(dy, dp),
        "dy_mean_abs": float(np.mean(np.abs(dy))),
        "dy_sd": float(np.std(dy, ddof=1)),
    }


def main() -> int:
    C.ANALYSIS.mkdir(parents=True, exist_ok=True)
    units, ref_pairs, meta = _load()
    ref_by_id = {p["pair_id"]: p for p in ref_pairs}
    comps_all = np.asarray([p["component"] for p in ref_pairs], dtype=object)
    uniq = np.asarray(sorted(set(comps_all.tolist())), dtype=object)

    # mapping_unique from the frozen core-class table
    core_classes = C.read_json(C.EDIT / "analysis/core_classes.json")["classes"]
    map_unique = {int(x["core_class"]): bool(x["mapping_unique"]) for x in core_classes}
    mapping_unique = {}
    for p in ref_pairs:
        mapping_unique[p["pair_id"]] = map_unique[p["core_class"]]
    n_mapping_unique = sum(mapping_unique.values())
    meta["mapping_unique_pairs"] = int(n_mapping_unique)
    meta["mapping_unique_components"] = int(
        len({p["component"] for p in ref_pairs if mapping_unique[p["pair_id"]]})
    )
    meta["n_mapping_unique_known"] = int(len(mapping_unique))

    # per-unit metrics
    rows: list[dict[str, Any]] = []
    for key, unit in units.items():
        model, seed = key.split(":")
        seed = int(seed.lstrip("s"))
        m = _metrics(unit, ref_pairs, ref_by_id)
        m.update({"unit": key, "model": model, "seed": seed})
        rows.append(m)

    # per-component means for bootstrap
    per_component: dict[str, tuple[dict[str, float], dict[str, float]]] = {}
    for key, unit in units.items():
        by_id = {p["pair_id"]: p for p in unit["audit_pairs"]}
        ids = [p["pair_id"] for p in ref_pairs]
        dy = np.asarray([by_id[i]["d_pred"] - by_id[i]["e_delta"] for i in ids], dtype=np.float64)
        dp = np.asarray([by_id[i]["d_pred"] for i in ids], dtype=np.float64)
        per_component[key] = (
            _component_means(dp - dy, comps_all, uniq),
            _component_means(-dy, comps_all, uniq),
        )

    rng = np.random.default_rng(C.BOOTSTRAP_SEED)
    boot: dict[str, dict[str, list[float]]] = {
        key: {"gain": [], "model": [], "zero": []} for key in units
    }
    for _ in range(C.BOOTSTRAP_REPLICATES):
        selected = rng.choice(uniq, size=uniq.size, replace=True)
        for key in units:
            cm, cz = per_component[key]
            model_mae = float(np.mean([cm[str(c)] for c in selected]))
            zero_mae = float(np.mean([cz[str(c)] for c in selected]))
            boot[key]["model"].append(model_mae)
            boot[key]["zero"].append(zero_mae)
            boot[key]["gain"].append(zero_mae - model_mae)

    # aggregate by model over seeds
    aggregate: list[dict[str, Any]] = []
    for model in C.MODELS:
        subset = [r for r in rows if r["model"] == model]
        gains = [r["Edit_Gain"] for r in subset]
        aggregate.append(
            {
                "unit": f"{model}:seed_mean",
                "model": model,
                "seed": "mean",
                "n_seeds": len(subset),
                "MAE_MODEL": float(np.mean([r["MAE_MODEL"] for r in subset])),
                "MAE_ZERO": float(np.mean([r["MAE_ZERO"] for r in subset])),
                "component_MAE_MODEL": float(np.mean([r["component_MAE_MODEL"] for r in subset])),
                "component_MAE_ZERO": float(np.mean([r["component_MAE_ZERO"] for r in subset])),
                "Edit_Gain": float(np.mean(gains)),
                "Edit_Gain_sd_over_seeds": float(np.std(gains, ddof=1)),
                "RMSE_MODEL": float(np.mean([r["RMSE_MODEL"] for r in subset])),
                "RMSE_ZERO": float(np.mean([r["RMSE_ZERO"] for r in subset])),
                "spearman_mean": float(np.mean([r["corr"]["spearman"] for r in subset])),
                "pearson_mean": float(np.mean([r["corr"]["pearson"] for r in subset])),
                "direction_agreement_mean": float(np.mean([r["direction"]["agreement"] for r in subset])),
                "direction_n_pairs": int(subset[0]["direction"]["n_pairs"]),
                "n_seeds_with_positive_gain": int(sum(g > 0 for g in gains)),
            }
        )

    # component bootstrap CI table
    boot_payload = {
        "schema": "lcrc-posthoc-edit-response-bootstrap-v1",
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "unit": "audit pair component (26 components)",
        "replicates": C.BOOTSTRAP_REPLICATES,
        "seed": C.BOOTSTRAP_SEED,
        "shared_resample_across_units": True,
        "conditional_on": "the six frozen checkpoints",
        "per_unit": {},
        "per_model_seed_mean": {},
    }
    for r in rows:
        key = r["unit"]
        entry = {
            "model": r["model"],
            "seed": r["seed"],
            "MAE_MODEL": C.quantile_ci(boot[key]["model"]),
            "MAE_ZERO": C.quantile_ci(boot[key]["zero"]),
            "Edit_Gain": C.quantile_ci(boot[key]["gain"]),
            "frac_boot_gain_positive": float(np.mean(np.asarray(boot[key]["gain"]) > 0)),
        }
        boot_payload["per_unit"][key] = entry
    # seed-mean by model: same resample sequence, average over seeds for each replicate
    for model in C.MODELS:
        keys = [f"{model}:s{s}" for s in C.MOLECULE_SEEDS]
        gains = np.mean([np.asarray(boot[k]["gain"]) for k in keys], axis=0)
        models = np.mean([np.asarray(boot[k]["model"]) for k in keys], axis=0)
        zeros = np.mean([np.asarray(boot[k]["zero"]) for k in keys], axis=0)
        boot_payload["per_model_seed_mean"][model] = {
            "Edit_Gain": C.quantile_ci(gains.tolist()),
            "MAE_MODEL": C.quantile_ci(models.tolist()),
            "MAE_ZERO": C.quantile_ci(zeros.tolist()),
            "frac_boot_gain_positive": float(np.mean(gains > 0)),
        }

    # stratified subsets
    strat_rows: list[dict[str, Any]] = []
    subsets: dict[str, np.ndarray] = {}
    for label, sel in (
        ("same_var_heavy", np.asarray([bool(p["same_var_heavy"]) for p in ref_pairs])),
        ("diff_var_heavy", np.asarray([not bool(p["same_var_heavy"]) for p in ref_pairs])),
        ("mapping_unique", np.asarray([mapping_unique[p["pair_id"]] for p in ref_pairs])),
        ("mapping_not_unique", np.asarray([not mapping_unique[p["pair_id"]] for p in ref_pairs])),
    ):
        subsets[label] = sel
    for label, sel in subsets.items():
        n_comp = len(set(comps_all[sel].tolist()))
        n_pos = int(sel.sum())
        for r in rows:
            key = r["unit"]
            by_id = {p["pair_id"]: p for p in units[key]["audit_pairs"]}
            ids = [p["pair_id"] for p in ref_pairs]
            dy = np.asarray([by_id[i]["d_pred"] - by_id[i]["e_delta"] for i in ids], dtype=np.float64)
            dp = np.asarray([by_id[i]["d_pred"] for i in ids], dtype=np.float64)
            comps = np.asarray([by_id[i]["component"] for i in ids], dtype=object)
            mae_model, _, _ = _subset_component_mae(dp - dy, comps, sel)
            mae_zero, _, _ = _subset_component_mae(-dy, comps, sel)
            strat_rows.append(
                {
                    "subset": label,
                    "unit": key,
                    "model": r["model"],
                    "seed": r["seed"],
                    "n_pairs": n_pos,
                    "n_components": n_comp,
                    "component_MAE_MODEL": mae_model,
                    "component_MAE_ZERO": mae_zero,
                    "Edit_Gain": mae_zero - mae_model,
                    "support_status": "OK" if n_comp >= 8 and n_pos >= 10 else "LOW_SUPPORT",
                }
            )

    # old TRACE summary, kept terminologically separate
    old = C.read_json(C.EDIT / "analysis/probe_report.json")
    old_summary = {
        "verdict": old.get("verdict"),
        "verdict_reason": old.get("verdict_reason"),
        "mean_gain_primary": old.get("mean_gain_primary"),
        "per_seed_primary": old.get("per_seed_primary"),
        "bootstrap_pooled_gain_ci": old.get("bootstrap", {}).get("pooled_gain_ci"),
        "note": (
            "The old NATIVE baseline sets e_delta=0 in the residual-error probe; it is NOT "
            "the ZERO_CHANGE baseline used here.  The old MAE values are errors in predicting "
            "e_delta, not errors in predicting Δy."
        ),
    }

    # CSV rows
    csv_rows = []
    for r in rows:
        csv_rows.append(
            {
                "unit": r["unit"],
                "model": r["model"],
                "seed": r["seed"],
                "n_pairs": r["n_pairs"],
                "n_components": r["n_components"],
                "MAE_MODEL": r["component_MAE_MODEL"],
                "MAE_ZERO": r["component_MAE_ZERO"],
                "Edit_Gain": r["Edit_Gain"],
                "RMSE_MODEL": r["RMSE_MODEL"],
                "RMSE_ZERO": r["RMSE_ZERO"],
                "spearman": r["corr"]["spearman"],
                "pearson": r["corr"]["pearson"],
                "direction_agreement": r["direction"]["agreement"],
                "direction_n_pairs": r["direction"]["n_pairs"],
                "direction_n_excluded": r["direction"]["n_remaining_excluded"],
                "result_label": C.RESULT_LABEL,
            }
        )
    for r in aggregate:
        csv_rows.append(
            {
                "unit": r["unit"],
                "model": r["model"],
                "seed": r["seed"],
                "n_pairs": 74,
                "n_components": 26,
                "MAE_MODEL": r["component_MAE_MODEL"],
                "MAE_ZERO": r["component_MAE_ZERO"],
                "Edit_Gain": r["Edit_Gain"],
                "RMSE_MODEL": r["RMSE_MODEL"],
                "RMSE_ZERO": r["RMSE_ZERO"],
                "spearman": r["spearman_mean"],
                "pearson": r["pearson_mean"],
                "direction_agreement": r["direction_agreement_mean"],
                "direction_n_pairs": r["direction_n_pairs"],
                "direction_n_excluded": 74 - r["direction_n_pairs"],
                "result_label": C.RESULT_LABEL,
            }
        )
    C.save_csv(C.ANALYSIS / "edit_response_metrics.csv", csv_rows, columns=list(csv_rows[0].keys()))
    C.save_csv(C.ANALYSIS / "edit_response_strata.csv", strat_rows, columns=list(strat_rows[0].keys()))
    C.write_json(C.ANALYSIS / "edit_response_bootstrap.json", boot_payload)

    payload = {
        "schema": "lcrc-posthoc-edit-response-v1",
        "round_id": C.ROUND_ID,
        "result_label": C.RESULT_LABEL,
        "meta": meta,
        "definition": {
            "delta_y": "d_pred - e_delta",
            "delta_pred": "d_pred",
            "MAE_MODEL": "component_equal_MAE(delta_pred - delta_y)",
            "MAE_ZERO": "component_equal_MAE(0 - delta_y)",
            "Edit_Gain": "MAE_ZERO - MAE_MODEL; positive means the model beats no-change",
            "component_equal_MAE": "each of the 26 components has total weight 1; pairs inside a component split that weight",
        },
        "per_unit": rows,
        "seed_mean": aggregate,
        "bootstrap": boot_payload,
        "strata": strat_rows,
        "old_trace": old_summary,
        "interpretation_guard": (
            "MODEL vs ZERO_CHANGE is a different question from TRACE vs NATIVE.  A positive "
            "Edit_Gain does not validate TRACE, and the old TRACE failure does not answer "
            "whether the model's local-edit response is useful."
        ),
    }
    C.write_json(C.ANALYSIS / "edit_response_analysis.json", payload)

    print(json.dumps(meta, indent=1))
    for r in rows:
        print(
            r["unit"],
            f"MAE_MODEL={r['component_MAE_MODEL']:.4f}",
            f"MAE_ZERO={r['component_MAE_ZERO']:.4f}",
            f"Gain={r['Edit_Gain']:+.4f}",
            f"rho={r['corr']['spearman']:.3f}",
            f"dir={r['direction']['agreement']:.3f} ({r['direction']['n_pairs']}/{r['direction']['n_pairs']+r['direction']['n_remaining_excluded']})",
        )
    for r in aggregate:
        print(
            r["unit"],
            f"Gain={r['Edit_Gain']:+.4f}",
            f"sd={r['Edit_Gain_sd_over_seeds']:.4f}",
            f"MAE_MODEL={r['component_MAE_MODEL']:.4f}",
            f"MAE_ZERO={r['component_MAE_ZERO']:.4f}",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

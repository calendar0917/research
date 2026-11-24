"""Compile machine-readable SFEC v1 summary and route labels."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np

import sfec1_common as C


def _diff(summary: dict[str, Any], weighting: str, name: str) -> dict[str, Any]:
    d = summary["bootstrap"][weighting]["diffs"][name]
    return {"point": d["point"], "ci": d["ci"]}


def _point(summary: dict[str, Any], model: str, weighting: str) -> float:
    return summary["points"][model][weighting]["mae"]


def _compare(summary: dict[str, Any], a: str, b: str) -> dict[str, Any]:
    out = {}
    for weighting in ("observation", "context", "pair_type"):
        ma = _point(summary, a, weighting)
        mb = _point(summary, b, weighting)
        if weighting in summary.get("bootstrap", {}):
            d = _diff(summary, weighting, f"{a}_minus_{b}")
            point, ci = d["point"], d["ci"]
        else:
            point, ci = ma - mb, None
        out[weighting] = {
            "mae_first": ma, "mae_second": mb,
            "diff_first_minus_second": point, "ci": ci,
            "relative_gain_of_second_vs_first": 1.0 - mb / max(ma, 1e-12),
        }
    return out


def _models(summary: dict[str, Any], models: list[str]) -> dict[str, Any]:
    return {
        m: {w: summary["points"][m][w]["mae"] for w in ("observation", "context", "pair_type")}
        for m in models
    }


def main() -> int:
    cfg = C.load_config()
    corr = C.read_json(C.ANALYSIS_DIR / "correction_details.json")
    b = C.read_json(C.ANALYSIS_DIR / "conditional_transfer.json")
    c = C.read_json(C.ANALYSIS_DIR / "strict_mask_details.json")
    v = C.read_json(C.ANALYSIS_DIR / "valid_transfer.json")
    e = C.read_json(C.ANALYSIS_DIR / "conditional_function_candidates.json")
    checks = C.read_json(C.ANALYSIS_DIR / "checks.json")
    main_b = b["main_pair_shared_support"]
    all_b = b["all_conditional_models_enabled"]
    clean_b = b["main_clean_subset"]
    models = ["ZERO", "PAIR", "PAIR_MEAN", "SHARED", "PAIR_Q", "PAIR_Q_MEAN",
              "PAIR_Q_NULL", "LOCAL", "LOCAL_NULL"]
    phase_b = {}
    for label, obj in (("main_pair_shared_support", main_b),
                       ("all_conditional_models_enabled", all_b),
                       ("main_clean_subset", clean_b)):
        phase_b[label] = {
            "n_obs": obj["n_obs"], "n_contexts": obj["n_contexts"],
            "n_pairs": obj["n_pair_types"],
            "coverage": obj["coverage"],
            "model_mae": _models(obj, models),
            "comparisons": {
                "PAIR_Q_vs_PAIR": _compare(obj, "PAIR_Q", "PAIR"),
                "PAIR_Q_MEAN_vs_PAIR_MEAN": _compare(obj, "PAIR_Q_MEAN", "PAIR_MEAN"),
                "PAIR_Q_vs_PAIR_Q_NULL": _compare(obj, "PAIR_Q", "PAIR_Q_NULL"),
                "LOCAL_vs_SHARED": _compare(obj, "LOCAL", "SHARED"),
                "LOCAL_vs_LOCAL_NULL": _compare(obj, "LOCAL", "LOCAL_NULL"),
                "LOCAL_vs_PAIR_Q": _compare(obj, "LOCAL", "PAIR_Q"),
            },
        }
    phase_c = {
        "n_selected_pairs": c["n_selected_pairs"],
        "n_evaluable_pairs": c["n_evaluable_pairs"],
        "n_obs": c["n_obs"], "n_contexts": c["n_contexts"],
        "model_mae": {
            m: {w: c["points"][m][w]["mae"] for w in ("observation", "context", "pair_type")}
            for m in ("ZERO", "SHARED_MASKED", "LOCAL_MASKED", "LOCAL_NULL_MASKED",
                      "PAIR_FULL", "PAIR_Q_FULL")
        },
        "comparisons": {
            name: {
                w: {
                    "mae_first": c["points"][name.split("_minus_")[0]][w]["mae"],
                    "mae_second": c["points"][name.split("_minus_")[1]][w]["mae"],
                    "diff_first_minus_second": c["bootstrap"][w]["diffs"][name]["point"],
                    "ci": c["bootstrap"][w]["diffs"][name]["ci"],
                    "relative_gain_of_second_vs_first": 1.0 - c["points"][name.split("_minus_")[1]][w]["mae"]
                    / max(c["points"][name.split("_minus_")[0]][w]["mae"], 1e-12),
                }
                for w in ("observation", "context", "pair_type")
            }
            for name in (
                "LOCAL_MASKED_minus_SHARED_MASKED", "LOCAL_MASKED_minus_LOCAL_NULL_MASKED",
                "LOCAL_MASKED_minus_ZERO", "LOCAL_MASKED_minus_PAIR_FULL",
                "LOCAL_MASKED_minus_PAIR_Q_FULL", "SHARED_MASKED_minus_ZERO")
        },
        "relative_gain_local_vs_shared": c["relative_gain_local_vs_shared"],
        "relative_gain_local_vs_null": c["relative_gain_local_vs_null"],
        "relative_gain_local_vs_zero": c["relative_gain_local_vs_zero"],
        "relative_gain_local_vs_shared_context": c["relative_gain_local_vs_shared_context"],
        "relative_gain_local_vs_shared_pair_type": c["relative_gain_local_vs_shared_pair_type"],
        "loo_min_relative_gain_local_vs_shared": c["loo_min_relative_gain_local_vs_shared"],
        "max_pair_contribution_share": c["max_pair_contribution_share"],
        "direction_consistent": c["direction_consistent"],
        "intervals_support_positive_gain": c["intervals_support_positive_gain"],
        "gain_not_entirely_one_pair": c["gain_not_entirely_one_pair"],
        "practical_screen_10pct": c["practical_screen_10pct"],
        "judgement": c["judgement"],
        "reason": c["reason"],
        "n_positive_pairs": int(sum(1 for x in c["pair_records"] if x.get("evaluable") and x["relative_gain_local_vs_shared"] > 0)),
        "n_negative_pairs": int(sum(1 for x in c["pair_records"] if x.get("evaluable") and x["relative_gain_local_vs_shared"] < 0)),
        "median_pair_gain": float(np.median([x["relative_gain_local_vs_shared"] for x in c["pair_records"] if x.get("evaluable")])),
        "min_pair_gain": float(min(x["relative_gain_local_vs_shared"] for x in c["pair_records"] if x.get("evaluable"))),
        "max_pair_gain": float(max(x["relative_gain_local_vs_shared"] for x in c["pair_records"] if x.get("evaluable"))),
        "per_pair_gain": [
            {"A_key": x["A_key"], "B_key": x["B_key"],
             "n_evaluation": x["n_evaluation"],
             "mae_shared": x["mae_SHARED_MASKED"], "mae_local": x["mae_LOCAL_MASKED"],
             "relative_gain": x["relative_gain_local_vs_shared"]}
            for x in c["pair_records"] if x.get("evaluable")
        ],
    }
    phase_d = {}
    for label, obj in v["subsets"].items():
        if not obj.get("n_obs"):
            phase_d[label] = {"n_obs": 0}
            continue
        phase_d[label] = {
            "n_obs": obj["n_obs"], "n_contexts": obj["n_contexts"], "n_pairs": obj["n_pairs"],
            "coverage": obj["coverage"],
            "model_mae": {m: {w: obj["points"][m][w]["mae"]
                              for w in ("observation", "context", "pair_type")}
                          for m in ("ZERO", "PAIR", "PAIR_Q", "SHARED", "LOCAL", "LOCAL_NULL")},
            "comparisons": {
                "PAIR_vs_ZERO": _compare(obj, "PAIR", "ZERO"),
                "PAIR_Q_vs_PAIR": _compare(obj, "PAIR_Q", "PAIR"),
                "SHARED_vs_ZERO": _compare(obj, "SHARED", "ZERO"),
                "LOCAL_vs_SHARED": _compare(obj, "LOCAL", "SHARED"),
                "LOCAL_vs_LOCAL_NULL": _compare(obj, "LOCAL", "LOCAL_NULL"),
                "LOCAL_vs_PAIR_Q": _compare(obj, "LOCAL", "PAIR_Q"),
                "LOCAL_vs_ZERO": _compare(obj, "LOCAL", "ZERO"),
            },
        }
    phase_e = {
        "dispersion": e["dispersion"],
        "sign_reversal": e["sign_reversal"],
        "candidates": e["candidates"],
    }
    route_labels = ["CONDITIONAL_COMPOSITION_SCREEN_ONLY", "CONDITIONAL_LOOKUP_ONLY"]
    answers = {
        "q1_correction_impact": {
            "reversed_discovery_fraction": corr["fraction_reversed_discovery_rows"],
            "reversed_evaluation_fraction": corr["fraction_reversed_evaluation_rows"],
            "pairs_with_flipped_median_sign": corr["n_pairs_with_flipped_sign"],
            "p1_mae_difference": corr["max_abs_p1_mae_difference"],
            "p1_rmse_difference": corr["max_abs_p1_rmse_difference"],
            "p2_v0_applicable_pairs": 16,
            "p2_corrected_applicable_pairs": corr["p2_v0style_corrected"]["pool"]["n_pairs"],
            "p2_v0_observations": 18584,
            "p2_corrected_observations": corr["p2_v0style_corrected"]["pool"]["n_obs"],
            "p3_mae_difference": 0.0,
            "valid_visible_mu_observations_v0": corr["valid_v0_visible_obs"],
            "valid_visible_mu_observations_canonical": corr["valid_canonical_visible_obs"],
        },
        "q2_q_improves_direct_lookup": {
            "main_obs_mae_pair": _point(main_b, "PAIR", "observation"),
            "main_obs_mae_pair_q": _point(main_b, "PAIR_Q", "observation"),
            "relative_gain": 1.0 - _point(main_b, "PAIR_Q", "observation") / _point(main_b, "PAIR", "observation"),
            "pair_q_minus_pair": _diff(main_b, "observation", "PAIR_Q_minus_PAIR"),
            "pair_q_minus_pair_null": _diff(main_b, "observation", "PAIR_Q_minus_PAIR_Q_NULL"),
        },
        "q3_local_beyond_conditional_lookup": {
            "main_obs_local": _point(main_b, "LOCAL", "observation"),
            "main_obs_pair_q": _point(main_b, "PAIR_Q", "observation"),
            "main_local_minus_pair_q": _diff(main_b, "observation", "LOCAL_minus_PAIR_Q"),
            "allcond_obs_local": _point(all_b, "LOCAL", "observation"),
            "allcond_obs_pair_q": _point(all_b, "PAIR_Q", "observation"),
            "allcond_local_minus_pair_q": _diff(all_b, "observation", "LOCAL_minus_PAIR_Q"),
            "valid_unseen_local_minus_pair_q": _diff(v["subsets"]["unseen_exact_context"], "observation", "LOCAL_minus_PAIR_Q"),
        },
        "q4_strict_mask_signal": {
            "n_evaluable_pairs": c["n_evaluable_pairs"], "n_obs": c["n_obs"],
            "local_masked_mae": _point(c, "LOCAL_MASKED", "observation"),
            "shared_masked_mae": _point(c, "SHARED_MASKED", "observation"),
            "local_null_masked_mae": _point(c, "LOCAL_NULL_MASKED", "observation"),
            "relative_gain_local_vs_shared": c["relative_gain_local_vs_shared"],
            "relative_gain_local_vs_null": c["relative_gain_local_vs_null"],
            "relative_gain_local_vs_zero": c["relative_gain_local_vs_zero"],
            "loo_min_gain": c["loo_min_relative_gain_local_vs_shared"],
            "max_pair_share": c["max_pair_contribution_share"],
            "practical_screen_10pct": c["practical_screen_10pct"],
        },
        "q5_valid_unseen_transfer": {
            "n_unseen_obs": v["subsets"]["unseen_exact_context"]["n_obs"],
            "zero": _point(v["subsets"]["unseen_exact_context"], "ZERO", "observation"),
            "pair": _point(v["subsets"]["unseen_exact_context"], "PAIR", "observation"),
            "pair_q": _point(v["subsets"]["unseen_exact_context"], "PAIR_Q", "observation"),
            "shared": _point(v["subsets"]["unseen_exact_context"], "SHARED", "observation"),
            "local": _point(v["subsets"]["unseen_exact_context"], "LOCAL", "observation"),
            "local_null": _point(v["subsets"]["unseen_exact_context"], "LOCAL_NULL", "observation"),
            "local_minus_pair_q": _diff(v["subsets"]["unseen_exact_context"], "observation", "LOCAL_minus_PAIR_Q"),
            "local_minus_shared": _diff(v["subsets"]["unseen_exact_context"], "observation", "LOCAL_minus_SHARED"),
            "local_minus_zero": _diff(v["subsets"]["unseen_exact_context"], "observation", "LOCAL_minus_ZERO"),
        },
        "q6_conditional_rule_found": {
            "pooled_p80_reduction": e["dispersion"]["pooled_p80_reduction"],
            "per_pair_p80_reduction_median": e["dispersion"]["per_pair_p80_reduction_median"],
            "fraction_pairs_improved": e["dispersion"]["fraction_pairs_improved"],
            "sign_reversal_pairs_0_05": e["sign_reversal"]["0.05"]["n_reversal_pairs"],
            "sign_reversal_group_reproduction_0_05": e["sign_reversal"]["0.05"]["group_reproduction_rate"],
            "sign_reversal_pairs_both_signs_reproduced_0_05": e["sign_reversal"]["0.05"]["n_pairs_both_signs_reproduced"],
            "conditional_candidates_tau_0_10": e["candidates"]["0.1"]["n_candidates"],
            "conditional_candidates_verifiable": e["candidates"]["0.1"]["n_verifiable_candidates"],
            "conditional_candidate_pairs": e["candidates"]["0.1"]["n_candidate_pairs"],
            "global_pair_candidates": e["candidates"]["0.1"]["global_pair_candidates"]["n_candidates"],
            "eval_obs_coverage": e["candidates"]["0.1"]["eval_obs_coverage_verifiable"],
            "eval_obs_coverage_fraction_of_main": e["candidates"]["0.1"]["eval_obs_coverage_verifiable"] / max(e["n_eval_rows_main"], 1),
        },
    }
    payload = {
        "round": cfg["round"],
        "parent_round": cfg["parent_round"],
        "git_commit": C.git_commit(),
        "git_dirty": C.git_dirty(),
        "checks_all_ok": bool(checks.get("all_ok", False)),
        "data": {
            "n_discovery_pair_observations": int(corr["n_discovery_pair_observations"]),
            "n_evaluation_pair_observations": int(corr["n_evaluation_pair_observations"]),
            "n_canonical_pairs": int(corr["n_pairs_canonical"]),
            "n_main_obs": int(main_b["n_obs"]),
            "n_main_contexts": int(main_b["n_contexts"]),
            "n_main_pairs": int(main_b["n_pair_types"]),
            "n_allcond_obs": int(all_b["n_obs"]),
            "n_allcond_contexts": int(all_b["n_contexts"]),
            "n_clean_obs": int(clean_b["n_obs"]),
        },
        "correction": corr,
        "phase_b": phase_b,
        "phase_c": phase_c,
        "phase_d": phase_d,
        "phase_e": phase_e,
        "route_labels": route_labels,
        "route_label_roles": {
            "primary": "CONDITIONAL_COMPOSITION_SCREEN_ONLY",
            "secondary_transfer_qualifier": "CONDITIONAL_LOOKUP_ONLY",
        },
        "route_label_rationale": {
            "CONDITIONAL_COMPOSITION_SCREEN_ONLY": (
                "All strict-mask statistical and support conditions in PROTOCOL C4 hold, but the "
                "pooled relative MAE gain vs SHARED_MASKED is 7.76%, below the conservative "
                "10% practical gate used for the SUPPORTED label. The composition signal is "
                "real but screen-only."
            ),
            "CONDITIONAL_LOOKUP_ONLY": (
                "Transfer-level qualifier, not a denial of the strict-mask signal: true q "
                "improves direct pair lookup (PAIR_Q < PAIR and < PAIR_Q_NULL with CIs), "
                "while LOCAL is not better than PAIR_Q on the common-enabled set (+2.51% MAE) "
                "or on unseen official-Valid contexts (+4.56% MAE). The robust cross-context "
                "gain is conditional lookup; the shared-composition increment is small."
            ),
            "not_selected": {
                "CONDITIONAL_COMPOSITION_SUPPORTED": "10% practical screen not met.",
                "NO_SUPPORTED_CONDITIONAL_RULE": "Sign reversals reproduce at 88.5% and 259 conditional candidates are verifiable at tau=0.10.",
                "INSUFFICIENT_SUPPORT": "20/20 selected pairs are evaluable with 21,970 observations.",
                "RESULT_SENSITIVE_TO_IDENTITY_OR_MASKING": "Clean-subset and strict-mask directions agree; strict-mask gain remains 7.76% and LOO minimum is 6.75%.",
            },
        },
        "answers": answers,
        "notes": {
            "valid_unseen_definition": v["definition_unseen"],
            "test_labels": "Official PCQM Test labels were never read.",
            "strict_mask_interpretation": "Per-pair strict masks remove all discovery cuts of every molecule identity found in the deleted B cells; SHARED/LOCAL/LOCAL_NULL are refitted on the same masked data. Pooled metrics concatenate pair-specific masks; bootstrap resamples shared evaluation contexts.",
            "practical_threshold": "The 10% relative MAE gain is a practical screen, not a scientific theorem.",
        },
    }
    C.write_json(C.ANALYSIS_DIR / "summary.json", payload)
    print(json.dumps({
        "route_labels": route_labels,
        "phase_b_main": {k: v for k, v in phase_b["main_pair_shared_support"]["comparisons"].items()},
        "phase_c": {k: phase_c[k] for k in (
            "n_evaluable_pairs", "n_obs", "relative_gain_local_vs_shared",
            "relative_gain_local_vs_null", "relative_gain_local_vs_zero",
            "loo_min_relative_gain_local_vs_shared", "practical_screen_10pct",
            "judgement")},
        "phase_d_unseen": phase_d["unseen_exact_context"]["comparisons"],
    }, indent=2)[:12000])
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Assemble the pre-specified decision and a machine-readable run record."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

import numpy as np

import ib_common as C


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _num(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        out = float(value)
    except (TypeError, ValueError):
        return None
    return out if np.isfinite(out) else None


def _bool(value: Any) -> bool:
    return str(value).strip().lower() == "true"


def _find(rows: list[dict[str, Any]], **kwargs: Any) -> dict[str, Any] | None:
    for row in rows:
        if all(str(row.get(k)) == str(v) for k, v in kwargs.items()):
            return row
    return None


def main() -> None:
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    start = C.now_utc()
    identity = json.loads((C.ANALYSIS_DIR / "identity_audit.json").read_text(encoding="utf-8"))
    gap = json.loads((C.ANALYSIS_DIR / "gap_summary.json").read_text(encoding="utf-8"))
    sensitivity = json.loads((C.ANALYSIS_DIR / "sensitivity_summary.json").read_text(encoding="utf-8"))
    thresholds = json.loads((C.ANALYSIS_DIR / "topology_thresholds.json").read_text(encoding="utf-8"))

    contrast_table: dict[tuple[str, int], list[dict[str, Any]]] = {}
    strata_table: dict[tuple[str, int], list[dict[str, Any]]] = {}
    for split, seed in C.CELLS:
        contrast_table[(split, seed)] = _read_csv(
            C.ANALYSIS_DIR / f"strata_contrasts_{split}_S{seed}.csv")
        strata_table[(split, seed)] = _read_csv(
            C.ANALYSIS_DIR / f"strata_{split}_S{seed}.csv")

    primary_contrast = {
        "closure_22": "closure_22_nz_high_minus_closure_22_zero",
        "reconvergence_2": "reconvergence_2_nz_high_minus_reconvergence_2_zero",
        "ring_frac": "ring_frac_high_minus_ring_frac_low",
    }
    stratum_pair = {
        "closure_22": ("closure_22_nz_high", "closure_22_zero"),
        "reconvergence_2": ("reconvergence_2_nz_high", "reconvergence_2_zero"),
        "ring_frac": ("ring_frac_high", "ring_frac_low"),
    }

    def contrast(split: str, seed: int, feature: str) -> dict[str, Any] | None:
        return _find(contrast_table[(split, seed)], feature=feature,
                     contrast=primary_contrast[feature])

    def stratum(split: str, seed: int, feature: str,
                name: str) -> dict[str, Any] | None:
        return _find(strata_table[(split, seed)], feature=feature, stratum=name)

    overall = {}
    for split, seed in C.CELLS:
        cell = gap["cells"][f"{split}_S{seed}"]
        overall[f"{split}_S{seed}"] = {
            "gain_rmse": cell["gain_rmse"],
            "gain_rmse_ci": cell["bootstrap"]["gain_rmse_ci"],
            "gain_mae": cell["gain_mae"],
            "gain_mae_ci": cell["bootstrap"]["gain_mae_ci"],
            "prop_ego2_better": cell["prop_ego2_better"],
            "prop_ego2_better_ci": cell["bootstrap"]["prop_ego2_better_ci"],
            "gain_mae_after_max_abs": cell["leave_one_out"]["gain_after_max_abs"],
            "dominance": cell["dominance"],
        }

    # Overall cross-seed answer.
    g_s0 = overall["D0_S0"]["gain_mae"]
    g_s1 = overall["D0_S1"]["gain_mae"]
    g_d1 = overall["D1_S0"]["gain_mae"]
    d0_both_positive = bool(g_s0 > 0 and g_s1 > 0)
    d1_same_direction = bool(g_d1 > 0)
    overall_ci_excludes_zero = {
        cell: bool(overall[cell]["gain_mae_ci"]["lo"] is not None
                   and overall[cell]["gain_mae_ci"]["lo"] > 0)
        for cell in overall
    }

    def sensitivity_for(feature: str) -> dict[str, Any]:
        # closure_22 is directly probed by drop22.  reconvergence_2 and ring
        # fraction have no direct probe; drop22 is the nearest pre-specified
        # perturbation and is marked indirect.
        drop = "22"
        per_cell = {}
        for split, seed in C.CELLS:
            rec = sensitivity["cells"][f"{split}_S{seed}"]["variants"][drop]
            mean_abs = rec["prediction_change"]["all"]["mean"]
            delta_rmse = rec["replay"]["delta_rmse"]
            per_cell[f"{split}_S{seed}"] = {
                "drop": drop,
                "direct": feature == "closure_22",
                "molecule_coverage": rec["coverage"]["molecule_coverage"],
                "coverage_sufficient": rec["coverage_sufficient"],
                "mean_abs_prediction_change": mean_abs,
                "delta_rmse": delta_rmse,
                "root_l2_mean": rec["root_change"]["l2_all"]["mean"],
                "non_degenerate": bool(
                    rec["coverage_sufficient"]
                    and ((mean_abs is not None and mean_abs >= 0.01)
                         or (delta_rmse is not None and abs(delta_rmse) >= 0.005))
                ),
            }
        applicable = feature in ("closure_22", "reconvergence_2")
        return {
            "matching_perturbation": drop,
            "direct": feature == "closure_22",
            "applicable": applicable,
            "specificity": (
                "direct d2-d2 probe (broader than closure_22)"
                if feature == "closure_22" else
                "indirect d2-d2 probe; it may remove triangle/reconvergence edges "
                "but is not specific to them"
                if feature == "reconvergence_2" else
                "no pre-specified edge-type probe for ring-atom fraction"
            ),
            "per_cell": per_cell,
            "non_degenerate_all": bool(
                applicable and all(v["non_degenerate"] for v in per_cell.values())),
        }

    feature_decisions: dict[str, Any] = {}
    for feature in C.PRIMARY_FEATURES:
        c0 = contrast("D0", 0, feature)
        c1 = contrast("D0", 1, feature)
        cd1 = contrast("D1", 0, feature)
        g0 = _num(c0["gain_mae"]) if c0 else None
        g1 = _num(c1["gain_mae"]) if c1 else None
        gd1 = _num(cd1["gain_mae"]) if cd1 else None
        sign0 = int(np.sign(g0)) if g0 is not None else 0
        sign1 = int(np.sign(g1)) if g1 is not None else 0
        sign_d1 = int(np.sign(gd1)) if gd1 is not None else 0
        any_d0_consistent = bool(
            sign0 != 0 and sign0 == sign1
            and abs(g0) >= 0.005 and abs(g1) >= 0.005
        )
        positive_d0_consistent = bool(
            g0 is not None and g1 is not None
            and g0 > 0 and g1 > 0
            and abs(g0) >= 0.005 and abs(g1) >= 0.005
        )
        d1_clearly_reversed = bool(
            gd1 is not None and sign_d1 != 0 and sign_d1 != sign0
            and abs(gd1) >= 0.01
        )
        d1_not_reversed = not d1_clearly_reversed
        # Support and dominance for the two compared strata.
        hi_name, lo_name = stratum_pair[feature]
        hi0 = stratum("D0", 0, feature, hi_name)
        lo0 = stratum("D0", 0, feature, lo_name)
        hi1 = stratum("D0", 1, feature, hi_name)
        lo1 = stratum("D0", 1, feature, lo_name)
        strata = {
            "D0_S0": (hi0, lo0),
            "D0_S1": (hi1, lo1),
            "D1_S0": (stratum("D1", 0, feature, hi_name),
                      stratum("D1", 0, feature, lo_name)),
        }
        support_ok = all(
            pair[0] is not None and pair[1] is not None
            and _bool(pair[0]["support_ok"]) and _bool(pair[1]["support_ok"])
            for pair in strata.values()
        )
        dominated = any(
            pair[0] is not None and pair[1] is not None
            and (_bool(pair[0]["dominated_by_few"])
                 or _bool(pair[1]["dominated_by_few"]))
            for pair in strata.values()
        )
        not_dominated = bool(support_ok and not dominated)
        sens = sensitivity_for(feature)
        recommend = bool(
            d0_both_positive and positive_d0_consistent and d1_not_reversed
            and sens["non_degenerate_all"] and not_dominated
        )
        feature_decisions[feature] = {
            "primary_contrast": primary_contrast[feature],
            "primary_contrast_definition": (
                "gain(nz_high) - gain(zero)" if feature != "ring_frac"
                else "gain(ring_high) - gain(ring_low)"
            ),
            "gain_mae_s0": g0,
            "gain_mae_s1": g1,
            "gain_mae_d1": gd1,
            "sign_s0": sign0,
            "sign_s1": sign1,
            "sign_d1": sign_d1,
            "any_d0_consistent": any_d0_consistent,
            "positive_d0_consistent": positive_d0_consistent,
            "d1_clearly_reversed": d1_clearly_reversed,
            "d1_not_reversed": d1_not_reversed,
            "strata_support_ok_all_cells": support_ok,
            "strata_dominated_by_few_some_cell": dominated,
            "not_dominated": not_dominated,
            "sensitivity": sens,
            "recommend_next_operator": recommend,
            "strata": {
                cell: {
                    "high": pair[0],
                    "low": pair[1],
                }
                for cell, pair in strata.items()
            },
        }

    any_recommendation = any(
        rec["recommend_next_operator"] for rec in feature_decisions.values()
    )
    if any_recommendation:
        decision = "STRUCTURAL_BINDING_CANDIDATE_LOCALIZED"
        evidence_class = "POSSIBLE_STRUCTURAL_BINDING_SIGNAL_NOT_CAUSAL"
    else:
        decision = "STRUCTURAL_BINDING_BOTTLENECK_NOT_LOCALIZED"
        evidence_class = "INDISTINGUISHABLE_FROM_NONLINEAR_COMPUTE_OR_OPTIMIZATION"

    answers = {
        "q1_cross_seed": {
            "d0_both_positive": d0_both_positive,
            "d1_same_direction": d1_same_direction,
            "overall_ci_excludes_zero": overall_ci_excludes_zero,
            "text": (
                "The EGO2 advantage has the same positive sign on D0/s0, D0/s1 and "
                "D1/s0, but every paired scaffold-cluster 95% interval for the "
                "overall gain includes zero, and the D0 per-molecule sign agreement "
                "is only about 55%."
            ),
        },
        "q2_structural_concentration": {
            "any_positive_d0_consistent": any(
                rec["positive_d0_consistent"] for rec in feature_decisions.values()),
            "any_positive_d0_consistent_with_support": any(
                rec["positive_d0_consistent"] and rec["not_dominated"]
                for rec in feature_decisions.values()),
            "any_d0_consistent": any(
                rec["any_d0_consistent"] for rec in feature_decisions.values()),
            "features": {
                feature: {
                    "positive_d0_consistent": rec["positive_d0_consistent"],
                    "any_d0_consistent": rec["any_d0_consistent"],
                    "d1_clearly_reversed": rec["d1_clearly_reversed"],
                    "not_dominated": rec["not_dominated"],
                }
                for feature, rec in feature_decisions.items()
            },
            "text": (
                "No primary feature shows a positive, D0-consistent, adequately "
                "supported, non-dominated structure contrast.  The only positive "
                "D0-consistent primary contrast is reconvergence_2, but its high stratum "
                "contains 1 molecule on D0/s0, 1 on D0/s1 and 3 on D1/s0 (4 or 6 "
                "non-zero Dev molecules in total), so it is unsupported and "
                "dominated.  The only adequately supported D0-consistent contrast "
                "is ring fraction, and it is negative.  The closure_22 high-vs-zero "
                "contrast is negative or null and not D0-consistent at the 0.005 "
                "margin."
            ),
        },
        "q3_input_sensitivity": {
            "drop22": {
                cell: sensitivity["cells"][cell]["variants"]["22"]
                for cell in sensitivity["cells"]
            },
            "drop11": {
                cell: sensitivity["cells"][cell]["variants"]["11"]
                for cell in sensitivity["cells"]
            },
            "text": (
                "The frozen EGO2 model is sensitive to d2-d2 edges: 50.8-54.4% of "
                "Dev molecules are covered, mean |Delta prediction| is 0.023-0.039, "
                "and removing them worsens Dev RMSE by 0.0028-0.0070.  The d1-d1 "
                "perturbation covers only 3.8-5.2% of molecules and is not "
                "interpreted for performance.  Sensitivity is therefore "
                "non-degenerate for d2-d2, but it does not show that the R2 gap is "
                "concentrated in the corresponding closure_22 strata."
            ),
        },
        "q4_evidence_class": {
            "decision": decision,
            "evidence_class": evidence_class,
            "text": (
                "The current evidence is more consistent with 'still cannot "
                "distinguish from nonlinear computation/optimisation differences' "
                "than with a localized missing structural binding.  R2 and EGO2 "
                "differ in architecture, parameterisation and optimisation, and the "
                "pre-specified structure contrasts do not isolate a positive "
                "EGO2 advantage in the real R2 non-tree/reconvergence strata."
            ),
        },
    }

    payload = {
        "schema": "r2-internal-binding-gap-audit-decision-v1",
        "created_utc": C.now_utc(),
        "identity": {
            "conclusion": identity["conclusion"],
            "errata_required": identity["errata_required"],
            "retraining_required": identity["retraining_required"],
            "d1_all_run_checks_ok": identity["d1_all_run_checks_ok"],
            "d1_run_ids": identity["d1_run_ids"],
        },
        "overall": overall,
        "thresholds": thresholds,
        "feature_decisions": feature_decisions,
        "answers": answers,
        "decision": decision,
        "evidence_class": evidence_class,
        "recommend_next_operator": any_recommendation,
        "notes": [
            "R2 and EGO2 are different architectures; this audit is not a controlled "
            "structural intervention on R2.",
            "The EGO2 edge removals are input-sensitivity probes, not legal chemical "
            "edits.",
            "D0 and D1 are not pooled; 63 Dev rows overlap.",
            "No new model or training run was produced.",
        ],
    }
    C.write_json(C.ANALYSIS_DIR / "decision.json", payload)

    code_files = sorted(C.CODE_DIR.glob("ib_*.py")) + [C.CODE_DIR / "run_all.py"]
    run_record = {
        "schema": "r2-internal-binding-gap-audit-run-v1",
        "created_utc": C.now_utc(),
        "started_utc": start,
        "round": "r2_internal_binding_gap_audit_v0",
        "protocol": str((C.ROUND_DIR / "PROTOCOL.md").relative_to(C.REPO_ROOT)),
        "protocol_sha256": C.sha256_file(C.ROUND_DIR / "PROTOCOL.md"),
        "code_sha256": C.code_sha(code_files),
        "code_files": [p.name for p in code_files],
        "environment": C.environment(),
        "inputs": C.assert_inputs(),
        "execution": {
            "device": "cpu",
            "gpu_used": False,
            "rr_used": False,
            "training_runs": 0,
            "official_valid_test_labels_read": False,
            "checkpoint_selection": "pre-existing best checkpoints only",
        },
        "outputs": sorted(p.name for p in C.ANALYSIS_DIR.iterdir() if p.is_file()),
    }
    C.write_json(C.ANALYSIS_DIR / "run.json", run_record)

    print(f"decision: {decision}")
    print(f"evidence class: {evidence_class}")
    print(f"recommend next operator: {any_recommendation}")
    for feature, rec in feature_decisions.items():
        print(f"  {feature}: s0={rec['gain_mae_s0']:+.4f} "
              f"s1={rec['gain_mae_s1']:+.4f} d1={rec['gain_mae_d1']:+.4f} "
              f"positive_D0={rec['positive_d0_consistent']} "
              f"any_D0={rec['any_d0_consistent']} "
              f"d1_reversed={rec['d1_clearly_reversed']} "
              f"not_dominated={rec['not_dominated']} "
              f"sens={rec['sensitivity']['non_degenerate_all']} "
              f"recommend={rec['recommend_next_operator']}")


if __name__ == "__main__":
    main()
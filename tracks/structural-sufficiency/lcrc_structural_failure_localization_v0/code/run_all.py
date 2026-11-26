"""Round driver: writes every deliverable of ``lcrc-structural-failure-localization-v0``.

CPU only, no training, no Test data.  Order of operations:

1. identity + RMSE-alignment verification (hard gate),
2. Phase A case localisation (GAIN5 / LOSS5 / COMMON5 union) + figures,
3. B1 label-blind Train nearest neighbours,
4. B2 strict matched pairs, re-restricted to Train/Valid,
5. B3/C1 frozen candidate groups over the whole split,
6. C2 same-direction pair statistics, C3 Valid correspondence,
7. verdict labels derived *from* the numbers (criteria frozen in ``VERDICT_RULES``).
"""

from __future__ import annotations

import time
from typing import Any

import numpy as np
import pandas as pd

import sfl_cases as CA
import sfl_common as C
import sfl_data as D
import sfl_groups as G
import sfl_neighbors as N
import sfl_pairs as P

#: frozen evidence rules.  Nothing below is tuned after seeing a number.
VERDICT_RULES = {
    "REPEATED_STRUCTURAL_RESPONSE_SIGNAL": "n_pairs >= 6 and n_distinct_cores >= 3 and "
    "n_shared_endpoints_components >= 3 and |delta_y_median| >= 1.0 and "
    "delta_y_sign_consistency >= 0.80",
    "INPUT_INFORMATION_GAP_SIGNAL": "at least one frozen acidic family that the model "
    "response ignores is absent from the actual input representation; requires an "
    "explicit field-level demonstration, otherwise it is NOT claimed",
    "LIMITED_TRAIN_SUPPORT": "Train support of the key structure < 5 molecules or "
    "< 3 distinct scaffolds",
    "COMMON_FAILURE_UNRESOLVED": "both models fail the same molecules with no repeated "
    "structural response and no information gap",
    "CASE_LEVEL_HYPOTHESIS_ONLY": "single-molecule chemical explanation only",
}


def main() -> dict[str, Any]:
    t0 = time.time()
    C.setup_paths()
    C.ANALYSIS_DIR.mkdir(parents=True, exist_ok=True)
    C.FIG_DIR.mkdir(parents=True, exist_ok=True)

    smarts = C.smarts_audit()
    if not smarts["all_pass"]:
        raise RuntimeError(f"frozen SMARTS probe failed: {smarts}")

    base = D.build_base()
    hist = D.check_historical_1425(base)
    cases, case_sets = CA.case_rows(base)
    figures = CA.draw_cases(cases)
    fig_of = {f["row_id"]: f for f in figures}
    cases["figure"] = cases["row_id"].map(lambda r: fig_of[int(r)]["figure"])
    cases["figure_highlighted_atoms"] = cases["row_id"].map(
        lambda r: ",".join(str(x) for x in fig_of[int(r)]["highlighted_atoms"])
    )
    cases.to_csv(C.ANALYSIS_DIR / "cases.csv", index=False)

    nbr, nbr_audit = N.neighbours(base, cases)
    vn_audit = pd.DataFrame(nbr_audit["cases"])
    nbr.to_csv(C.ANALYSIS_DIR / "train_neighbors.csv", index=False)

    pairs, pair_checks = P.build(base)
    pairs.to_csv(C.ANALYSIS_DIR / "matched_pairs.csv", index=False)
    pair_groups = P.c2_groups(pairs)

    annotated = G.annotate(base)
    controls = G.train_median_controls(annotated)
    groups = G.group_summary(annotated, controls)
    groups.to_csv(C.ANALYSIS_DIR / "structural_group_summary.csv", index=False)
    c3 = G.c3_valid_cases(annotated, cases)

    # ---- derived evidence numbers -----------------------------------------
    valid = base.errors[base.errors.split == "valid"]
    total_gain = float(valid["gain"].sum())
    gain_sorted = valid.sort_values(["gain", "row_id"], ascending=[False, True])
    top3 = float(gain_sorted["gain"].head(3).sum())
    top5 = float(gain_sorted["gain"].head(5).sum())
    acid_pair = next(g for g in pair_groups if g["group"] == "ANY_ACID_APPEARS")
    ester_pair = next(g for g in pair_groups if g["group"] == "ESTER_TO_CARBOXYL")
    phospho_pair = next(g for g in pair_groups if g["group"] == "PHOSPHO_ACID_APPEARS")

    train_any = groups[(groups.split == "train") & (groups.group == "ANY_ACID")].iloc[0]
    train_phos = groups[(groups.split == "train") & (groups.group == "PHOSPHO_ACID")].iloc[0]
    valid_phos = groups[(groups.split == "valid") & (groups.group == "PHOSPHO_ACID")].iloc[0]
    valid_carb = groups[(groups.split == "valid") & (groups.group == "CARBOXYL_ACID")].iloc[0]
    valid_none = groups[(groups.split == "valid") & (groups.group == "NO_FROZEN_ACID")].iloc[0]
    train_carb = groups[(groups.split == "train") & (groups.group == "CARBOXYL_ACID")].iloc[0]

    repeated_ok = bool(
        acid_pair["status"] == "MATCHED_ANALOG"
        and acid_pair["n_pairs"] >= 6
        and acid_pair["n_distinct_cores"] >= 3
        and acid_pair["n_shared_endpoints_components"] >= 3
        and abs(acid_pair["delta_y_median"]) >= 1.0
        and acid_pair["delta_y_sign_consistency"] >= 0.80
    )
    limited_phos = bool(int(train_phos["n_molecules"]) < 5 or int(train_phos["n_distinct_scaffolds"]) < 3)

    labelled = cases[cases.candidate_families.ne("none")]
    n_labelled_cases = int(len(labelled))
    n_cases = int(len(cases))

    # size / lipophilicity controls: does the acid effect survive stratification?
    def _strat(group: str, split: str, lab: str) -> dict[str, Any]:
        r = groups[(groups.split == split) & (groups.group == group)]
        if r.empty:
            return {}
        r = r.iloc[0]
        return {
            "n": int(r.get(f"n_{lab}", 0) or 0),
            "errA_mean": r.get(f"errA_mean_{lab}"),
            "errB_mean": r.get(f"errB_mean_{lab}"),
        }

    controls_report = {
        "cuts": controls,
        "valid_CARBOXYL_small": _strat("CARBOXYL_ACID", "valid", "small"),
        "valid_CARBOXYL_large": _strat("CARBOXYL_ACID", "valid", "large"),
        "valid_CARBOXYL_low_logp": _strat("CARBOXYL_ACID", "valid", "low_logp"),
        "valid_CARBOXYL_high_logp": _strat("CARBOXYL_ACID", "valid", "high_logp"),
        "valid_NO_ACID_small": _strat("NO_FROZEN_ACID", "valid", "small"),
        "valid_NO_ACID_large": _strat("NO_FROZEN_ACID", "valid", "large"),
        "valid_NO_ACID_low_logp": _strat("NO_FROZEN_ACID", "valid", "low_logp"),
        "valid_NO_ACID_high_logp": _strat("NO_FROZEN_ACID", "valid", "high_logp"),
    }

    # input-representation facts (no hidden-vector inspection of any kind)
    def _n(split: str, col: str) -> int:
        return int((annotated[annotated.split == split][col] == 1).sum())

    input_facts = {
        "rows_with_nonzero_formal_charge": int((annotated["n_charged_atoms"] > 0).sum()),
        "rows_with_nonzero_formal_charge_train": int(
            ((annotated.split == "train") & (annotated.n_charged_atoms > 0)).sum()
        ),
        "rows_with_nonzero_formal_charge_valid": int(
            ((annotated.split == "valid") & (annotated.n_charged_atoms > 0)).sum()
        ),
        "train_rows_with_element_P": int((annotated[annotated.split == "train"]["n_P"] > 0).sum()),
        "valid_rows_with_element_P": int((annotated[annotated.split == "valid"]["n_P"] > 0).sum()),
        "train_rows_with_P_OH_or_P_O_minus": _n("train", "PHOSPHO"),
        "valid_rows_with_P_OH_or_P_O_minus": _n("valid", "PHOSPHO"),
        "train_rows_with_P_TRIESTER": _n("train", "P_TRIESTER"),
        "valid_rows_with_P_TRIESTER": _n("valid", "P_TRIESTER"),
        "train_rows_with_SULFONIC": _n("train", "SULFONIC"),
        "valid_rows_with_SULFONIC": _n("valid", "SULFONIC"),
        "train_acid_molecules_with_charged_representation": int(
            ((annotated.split == "train") & (annotated.ANY_ACID == 1) & (annotated.n_charged_atoms > 0)).sum()
        ),
        "encoder_A_input_fields": CA.ENCODER_INPUTS["A_FIXED1"],
        "encoder_B_input_fields": CA.ENCODER_INPUTS["B_GINE"],
        "declared_absent_in_both": [
            "pH / solution context",
            "pKa or any measured ionization constant",
            "microstate / protonation-state label",
        ],
        "representation_conflict_claim": "NOT_MADE",
        "representation_status": "REPRESENTATION_UNRESOLVED",
    }

    supported = []
    if repeated_ok:
        supported.append("REPEATED_STRUCTURAL_RESPONSE_SIGNAL")
    if limited_phos:
        supported.append("LIMITED_TRAIN_SUPPORT")
    supported.append("COMMON_FAILURE_UNRESOLVED")
    if n_labelled_cases <= 2 and n_cases > 5:
        supported.append("CASE_LEVEL_HYPOTHESIS_ONLY")
    # INPUT_INFORMATION_GAP_SIGNAL is only claimable with a field-level collapse proof
    input_gap_claimed = False

    summary = {
        "round_id": C.ROUND_ID,
        "schema": "sfl-summary-v1",
        "protocol": {
            "data": "official ogbg-mollipo scaffold split, Train 3360 / Valid 420",
            "test_used": False,
            "training": "none (no model was trained, fitted, tuned or distilled)",
            "external_services": "none (no pKa service, no external label)",
            "cpu_only": True,
            "gpu_hours": 0.0,
            "rdkit_version": __import__("rdkit").__version__,
            "frozen_smarts": C.FROZEN_SMARTS,
            "frozen_smarts_sha256": C.sha256_smarts(),
            "case_sets": case_sets,
            "top_k": C.TOP_K,
            "sim_thresholds": {"hi": C.SIM_HI, "vhi": C.SIM_VHI},
        },
        "verification": {
            "rmse_recomputed": base.checks["rmse"],
            "rmse_reference_alignment": base.checks["rmse_reference_alignment"],
            "identity_vs_raw_ogb_graph": base.checks["identity_vs_raw_ogb_graph"],
            "split_agreement_with_official_file": base.checks["split_agreement_with_official_file"],
            "max_label_gap_vs_mol_csv": base.checks["max_label_gap_vs_mol_csv"],
            "historical_case_1425": hist,
            "frozen_smarts_probe": smarts,
            "paired_errors_sha256": base.checks["paired_errors_sha256"],
        },
        "net_gain_decomposition": {
            "valid_sum_sq_err_A_minus_B": total_gain,
            "top1_row": int(gain_sorted.iloc[0]["row_id"]),
            "top1_gain": float(gain_sorted.iloc[0]["gain"]),
            "top3_gain": top3,
            "top3_share": top3 / total_gain,
            "top5_rows": [int(x) for x in gain_sorted["row_id"].head(5)],
            "top5_gain": top5,
            "top5_share": top5 / total_gain,
            "negatives_sum": float(gain_sorted[gain_sorted.gain < 0]["gain"].sum()),
            "n_rows_with_negative_gain": int((gain_sorted.gain < 0).sum()),
        },
        "phase_a": {
            "n_unique_cases": n_cases,
            "n_case_slots": 15,
            "n_cases_with_frozen_acid_family": n_labelled_cases,
            "n_cases_without_any_frozen_acid_family": n_cases - n_labelled_cases,
            "acid_cases": [
                {"row_id": int(r.row_id), "families": str(r.candidate_families)}
                for r in labelled.itertuples()
            ],
        },
        "sensitivity_without_top1": {
            "row_removed": 1425,
            "valid_rmse_A": C.rmse(valid[valid.row_id != 1425]["error_A"]),
            "valid_rmse_B": C.rmse(valid[valid.row_id != 1425]["error_B"]),
            "valid_sum_gain_remaining": float(valid[valid.row_id != 1425]["gain"].sum()),
            "valid_any_acid_rmse_A_without_1425": C.rmse(
                annotated[(annotated.split == "valid") & (annotated.ANY_ACID == 1)
                          & (annotated.row_id != 1425)]["error_A"]
            ),
            "valid_any_acid_rmse_B_without_1425": C.rmse(
                annotated[(annotated.split == "valid") & (annotated.ANY_ACID == 1)
                          & (annotated.row_id != 1425)]["error_B"]
            ),
            "valid_any_acid_n_without_1425": int(
                ((annotated.split == "valid") & (annotated.ANY_ACID == 1)
                 & (annotated.row_id != 1425)).sum()
            ),
            "valid_any_acid_errA_mean_without_1425": float(
                annotated[(annotated.split == "valid") & (annotated.ANY_ACID == 1)
                          & (annotated.row_id != 1425)]["error_A"].mean()
            ),
        },
        "phase_b1": {
            **nbr_audit,
            "case_support": [
                {
                    "case_row_id": int(r["case_row_id"]),
                    "max_sim_train": float(r["max_sim"]),
                    "n_sim_ge_0.6": int(r["n_sim_ge_0.6"]),
                    "n_sim_ge_0.8": int(r["n_sim_ge_0.8"]),
                    "top1_train_row": int(
                        nbr[(nbr.case_row_id == r["case_row_id"]) & (nbr["rank"] == 1)]["neighbor_row_id"].iloc[0]
                    ),
                    "top1_train_row_families": str(
                        nbr[(nbr.case_row_id == r["case_row_id"]) & (nbr["rank"] == 1)]["neighbor_candidate_families"].iloc[0]
                    ),
                }
                for _, r in vn_audit.iterrows()
            ],
        },
        "phase_b2": pair_checks,
        "phase_c2": pair_groups,
        "phase_c3": c3,
        "size_lipophilicity_control": controls_report,
        "train_support": {
            "CARBOXYL_ACID": {
                "n_molecules": int(train_carb["n_molecules"]),
                "n_distinct_scaffolds": int(train_carb["n_distinct_scaffolds"]),
                "y_minus_mol_logp_median": float(train_carb["y_minus_mol_logp_median"]),
            },
            "PHOSPHO_ACID": {
                "n_molecules": int(train_phos["n_molecules"]),
                "n_distinct_scaffolds": int(train_phos["n_distinct_scaffolds"]),
                "y_minus_mol_logp_median": float(train_phos["y_minus_mol_logp_median"]),
            },
            "VALID_PHOSPHO_ACID": {
                "n_molecules": int(valid_phos["n_molecules"]),
                "rows": c3["valid_phospho_rows"],
            },
            "VALID_CARBOXYL_ACID": {
                "n_molecules": int(valid_carb["n_molecules"]),
                "rmse_A": float(valid_carb["rmse_A"]),
                "rmse_B": float(valid_carb["rmse_B"]),
                "err_A_mean": float(valid_carb["err_A_mean"]),
                "err_B_mean": float(valid_carb["err_B_mean"]),
            },
            "VALID_NO_FROZEN_ACID": {
                "n_molecules": int(valid_none["n_molecules"]),
                "rmse_A": float(valid_none["rmse_A"]),
                "rmse_B": float(valid_none["rmse_B"]),
            },
        },
        "input_representation": input_facts,
        "verdict": {
            "supported_labels": supported,
            "rules": VERDICT_RULES,
            "input_information_gap_claimed": input_gap_claimed,
            "notes": [
                "Train-side model predictions are in-sample observations; no "
                "cross-molecule generalisation is claimed from them.",
                "Valid was already used in the discovery step; nothing here is an "
                "independent confirmation.",
                "No hidden-vector comparison was performed; representation questions "
                "are answered only at the level of the declared input fields.",
            ],
        },
        "wall_clock_seconds": None,
    }
    summary["wall_clock_seconds"] = float(time.time() - t0)
    C.write_json(C.ANALYSIS_DIR / "summary.json", summary)

    checks = {
        "round_id": C.ROUND_ID,
        "gates": {
            "identity_ok": base.checks["identity_vs_raw_ogb_graph"]["identity_ok"],
            "split_agreement": base.checks["split_agreement_with_official_file"],
            "label_gap_lt_1e-6": base.checks["max_label_gap_vs_mol_csv"] < 1e-6,
            "rmse_within_tolerance": all(
                v["within_1e-5"] for v in base.checks["rmse_reference_alignment"].values()
            ),
            "frozen_smarts_probe_all_pass": smarts["all_pass"],
            "historical_1425_identity": hist["identity_ok"],
            "pairs_orientation_consistent": pair_checks["orientation_mismatch"] == 0,
            "pairs_no_fragment_element_mismatch": len(pair_checks["fragment_element_mismatch"]) == 0,
            "test_split_file_opened": False,
            "no_model_fitted": True,
        },
        "counts": {
            "cases": n_cases,
            "neighbor_rows": int(len(nbr)),
            "matched_pairs_train_valid": int(len(pairs)),
            "matched_pairs_acid_direction": int(acid_pair["n_pairs"]),
            "matched_pairs_ester_to_acid": int(ester_pair.get("n_pairs", 0)),
            "matched_pairs_phospho": int(phospho_pair.get("n_pairs", 0)),
            "group_summary_rows": int(len(groups)),
        },
        "detail": {
            "pair_checks": pair_checks,
            "rmse_reference_alignment": base.checks["rmse_reference_alignment"],
        },
    }
    C.write_json(C.ANALYSIS_DIR / "checks.json", checks)
    return summary


if __name__ == "__main__":
    s = main()
    print("wall_clock_seconds", round(s["wall_clock_seconds"], 2))
    print("verdict", s["verdict"]["supported_labels"])
    print("gain decomposition", {k: v for k, v in s["net_gain_decomposition"].items() if k != "top5_rows"})

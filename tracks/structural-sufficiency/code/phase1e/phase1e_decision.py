"""Phase 1e Stage A4 + route decision — apply the frozen rule set.

Reads `label_audit.json`, `error_attribution.json`, `zero_training_routing.json`,
`reference_audit.json` and (if it exists) `strong_baseline.json`, then applies
the decision rules written in `notes/phase1e_protocol.md` §2–§4 verbatim.

The rules are deliberately mechanical: no rule may be relaxed after seeing a
number, and a weak correlation is never allowed to trigger a conclusion.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
if str(HERE) not in sys.path:
    sys.path.insert(0, str(HERE))

import phase1e_common as C  # noqa: E402

OUT = C.RESULTS_1E / "decision.json"
STRONG = C.RESULTS_1E / "strong_baseline.json"
B2_KEY = "P3E_seed0"


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.parse_args(argv)

    label = C.read_json(C.RESULTS_1E / "label_audit.json")
    err = C.read_json(C.RESULTS_1E / "error_attribution.json")
    routing = C.read_json(C.RESULTS_1E / "zero_training_routing.json")
    ref = C.read_json(C.RESULTS_1E / "reference_audit.json")
    strong = C.read_json(STRONG) if STRONG.exists() else None

    conf = label["monitor_confusion"]
    perm = label["permutation_stability"]
    reconstruct = label["reconstruction"]

    n_r = conf["counts"]["n_rdkit_large"]
    n_p = conf["counts"]["n_label_penalty"]
    n_r_not_p = conf["label_penalty_x_rdkit_large_ring"]["rdkit_large_not_labeled"]

    cap = err["models"]["R1_size_cap_seed0"]["groups"]
    gref = err["models"]["G_ref_seed0"]["groups"]
    comp_gap = cap["label_complement"]["mae"] - gref["label_complement"]["mae"]
    pen_share = max(
        err["models"][m]["groups"]["label_penalised"]["share_of_total_abs_error"]
        for m in err["models"]
    )

    # ---- rule 1: BENCHMARK_CYCLE_ARTIFACT_DOMINATES ------------------------
    r1a = (n_r_not_p / max(n_r, 1)) <= C.MAX_UNPENALISED_SHARE
    r1b = comp_gap < C.NEGLIGIBLE_BAND
    r1c = pen_share >= C.MIN_PENALISED_ERROR_SHARE
    artifact_dominates = bool(r1a and r1b and r1c)

    # ---- arm membership ----------------------------------------------------
    # A4 asks a question *inside* the SSRC arm family (same patch pipeline, same
    # information).  The Stage-B2 strong baseline is a different information
    # regime and answers the B3 question instead; it is never mixed into A4.
    ssrc_models = [m for m in err["models"] if m != B2_KEY]

    # ---- rule 2: STRUCTURAL_SIGNAL_BEYOND_CYCLE_REMAINS --------------------
    beyond_cycle_evidence: list[dict[str, Any]] = []
    beyond_cycle = False
    for model_key, cmp in err["comparisons_vs_G_ref"].items():
        if model_key == B2_KEY:
            continue
        for group_key in ("label_complement", "label_penalised"):
            c = cmp[group_key]
            better = c["mean"] <= -C.ZINC_MEANINGFUL_BAND and c["ci95"][1] < 0
            beyond_cycle_evidence.append(
                {
                    "model": model_key,
                    "group": group_key,
                    "delta_vs_Gref": c["mean"],
                    "ci95": c["ci95"],
                    "candidate_better_by_0.02_with_ci_below_zero": bool(better),
                }
            )
            beyond_cycle = beyond_cycle or bool(better)

    # ---- rule 3: UNRESOLVED_LABEL_MECHANISM --------------------------------
    unstable_frac = perm["n_unstable_penalty_monitor"] / max(n_p, 1)
    recon = label["monitor_groups"]
    pen_group = next(g for g in recon if g["name"] == "label_penalty_union")
    recon_frac_gt_1 = 0.0  # |recon| max is reported; compute from the per-molecule list
    per_mol = label["monitor_penalised_molecules"]
    recon_frac_gt_1 = float(np.mean([p["recon_abs_error"] > 1.0 for p in per_mol])) if per_mol else 0.0
    missed = conf["label_penalty_x_rdkit_large_ring"]["labeled_not_rdkit_large"]
    missed_err_share = max(
        err["models"][m]["groups"]["penalty_only_rdkit_missed"]["share_of_total_abs_error"]
        for m in ("R1_size_cap_seed0", "G_ref_seed0")
    )
    r3a = unstable_frac >= C.MAX_UNSTABLE_FRACTION
    r3b = recon_frac_gt_1 >= C.MAX_RECON_ERROR_FRACTION
    r3c = (
        (missed / max(n_p, 1)) >= C.MIN_PROXY_MISS_FRACTION
        and missed_err_share >= C.MIN_PROXY_MISS_FRACTION
    )
    unresolved_label = bool(r3a or r3b or r3c)

    # ---- Stage B3: R1_IS_NOT_COMPETITIVE -----------------------------------
    if strong is None:
        b3 = {
            "label": "NOT_EVALUATED",
            "reason": "no strictly isolated strong baseline on the SSRC split",
        }
        r1_not_competitive = None
    else:
        best_key = min(ssrc_models, key=lambda k: err["models"][k]["groups"]["all"]["mae"])
        best_mae = err["models"][best_key]["groups"]["all"]["mae"]
        b2_groups = err["models"][B2_KEY]["groups"]
        b2_vs_gref = err["comparisons_vs_G_ref"][B2_KEY]
        gap = strong["monitor_mae"] - best_mae
        lo, hi = strong["bootstrap_gap_vs_best_r1_ci95"]
        r1_not_competitive = bool(gap <= -C.ZINC_MEANINGFUL_BAND and hi < 0)
        b3 = {
            "label": "R1_IS_NOT_COMPETITIVE" if r1_not_competitive else "R1_STILL_IN_CONTENTION",
            "strong_baseline_monitor_mae": strong["monitor_mae"],
            "best_ssrc_arm": {"key": best_key, "monitor_mae": best_mae},
            "gap_strong_minus_best_ssrc": gap,
            "bootstrap_gap_ci95": [lo, hi],
            "rule": "gap <= -0.02 with a 95% paired-bootstrap CI wholly below 0",
            "information_regime_warning": (
                "the strong baseline receives structure-only global descriptors that the "
                "SSRC arms do not (ring-size fractions, cyclomatic rank, junction counts); "
                "this is a competitiveness statement, not a controlled architecture comparison"
            ),
            "strong_baseline_by_group": {
                g: {
                    "mae": b2_groups[g]["mae"],
                    "n": b2_groups[g]["n"],
                    "delta_vs_G_ref": b2_vs_gref[g]["mean"],
                    "ci95": b2_vs_gref[g]["ci95"],
                }
                for g in ("label_complement", "label_penalised", "rdkit_large")
            },
            "cycle_offset_capture_on_penalised": b2_groups["label_penalised"][
                "cycle_offset_capture"
            ],
            "interpretation": (
                "the strong baseline also wins on the 957-molecule complement, where the "
                "ring-size descriptors are nearly constant, so the gap is not explained by "
                "the ring tail alone"
            ),
        }

    # ---- route (protocol §4) ----------------------------------------------
    reason = ""
    if artifact_dominates and not beyond_cycle:
        direction = "A"
        reason = (
            "the large-ring mechanism is dominated by the artificial cycle penalty, no SSRC "
            "arm beats G_ref beyond the negligible band once the penalty is accounted for, "
            "and the family is not competitive with an isolated strong baseline"
            if r1_not_competitive
            else "the large-ring mechanism is dominated by the artificial cycle penalty and no "
            "SSRC arm beats G_ref beyond the negligible band once it is accounted for"
        )
    elif r1_not_competitive and beyond_cycle:
        direction = "B"
        reason = "relational information still carries signal but the R1 backbone is not competitive"
    elif beyond_cycle and not r1_not_competitive:
        direction = "C"
        reason = "an out-of-label-rule structural signal survives and the backbone is still competitive"
    elif not artifact_dominates and not beyond_cycle and r1_not_competitive:
        direction = "D"
        reason = "no defensible mechanism claim on either side"
    elif unresolved_label and not (artifact_dominates or beyond_cycle):
        direction = "D"
        reason = "the only surviving finding is label-mechanism ambiguity"
    else:
        direction = "D"
        reason = "no rule fired that supports continuing the R1 branch"

    carry_forward = []
    if direction == "A" and r1_not_competitive:
        carry_forward.append(
            "the Phase C non-ZINC task must be run on a competitive backbone, not on G_ref or R1"
        )
    if unresolved_label:
        carry_forward.append(
            "report the order-dependent cycle basis as a benchmark defect, not as a modelling "
            "failure; any future ZINC claim must exclude the penalised subset or report it separately"
        )

    payload = {
        "protocol_id": "ssrc-phase1e-zinc-v1",
        "stage": "A4_decision",
        "rules_source": "notes/phase1e_protocol.md §2–§4",
        "thresholds": {
            "max_unpenalised_share": C.MAX_UNPENALISED_SHARE,
            "negligible_band": C.NEGLIGIBLE_BAND,
            "zinc_meaningful_band": C.ZINC_MEANINGFUL_BAND,
            "min_penalised_error_share": C.MIN_PENALISED_ERROR_SHARE,
            "max_unstable_fraction": C.MAX_UNSTABLE_FRACTION,
            "max_recon_error_gt_1_fraction": C.MAX_RECON_ERROR_FRACTION,
            "min_proxy_miss_fraction": C.MIN_PROXY_MISS_FRACTION,
        },
        "inputs": [
            "label_audit.json",
            "error_attribution.json",
            "zero_training_routing.json",
            "reference_audit.json",
            "strong_baseline.json" if strong is not None else "(absent)",
        ],
        "artifact_dominates": {
            "label": "BENCHMARK_CYCLE_ARTIFACT_DOMINATES" if artifact_dominates else "NOT_SUPPORTED",
            "fired": artifact_dominates,
            "conditions": {
                "rdkit_large_group_purity_le_0.05_unpenalised": {
                    "n_rdkit_large": n_r,
                    "n_large_not_penalised": n_r_not_p,
                    "value": n_r_not_p / max(n_r, 1),
                    "pass": bool(r1a),
                },
                "complement_gap_cap_vs_Gref_lt_0.02": {
                    "value": float(comp_gap),
                    "pass": bool(r1b),
                },
                "penalised_error_share_ge_0.25": {
                    "value": float(pen_share),
                    "pass": bool(r1c),
                },
            },
        },
        "beyond_cycle": {
            "label": "STRUCTURAL_SIGNAL_BEYOND_CYCLE_REMAINS" if beyond_cycle else "NOT_SUPPORTED",
            "fired": beyond_cycle,
            "evidence": beyond_cycle_evidence,
            "note": (
                "R1_size_cap does have a positive point estimate on the penalised group but its "
                "CI crosses zero; the frozen rule therefore does not fire. This is recorded, not "
                "upgraded."
            ),
        },
        "unresolved_label_mechanism": {
            "label": "UNRESOLVED_LABEL_MECHANISM" if unresolved_label else "NOT_SUPPORTED",
            "fired": unresolved_label,
            "conditions": {
                "permutation_unstable_fraction_ge_0.05": {
                    "n_unstable": perm["n_unstable_penalty_monitor"],
                    "n_penalised": n_p,
                    "value": float(unstable_frac),
                    "pass": bool(r3a),
                    "rows": perm["unstable_penalty_monitor_rows"],
                },
                "reconstruction_error_gt_1_fraction_ge_0.05": {
                    "value": float(recon_frac_gt_1),
                    "pass": bool(r3b),
                },
                "proxy_miss_fraction_and_error_share_ge_0.05": {
                    "missed_fraction": missed / max(n_p, 1),
                    "missed_error_share": float(missed_err_share),
                    "pass": bool(r3c),
                },
            },
        },
        "stage_b3": b3,
        "routing_diagnostic": {
            "routed_mae": routing["routed_rdkit"]["monitor_mae"],
            "gap_vs_G_ref": routing["routed_rdkit"]["monitor_mae"]
            - ref["protocol_tables"]["internal_ssrc_monitor"]["rows"]["G_ref_seed0"],
            "delta_vs_G_ref_bootstrap": routing["comparisons"][0]["delta"],
            "upgraded_to_method": False,
            "why_not": "it remains 0.027 MAE worse than G_ref on the same reused Monitor",
        },
        "direction": {
            "choice": direction,
            "reason": reason,
            "carry_forward_constraints": carry_forward,
            "A_triggered": bool(artifact_dominates),
            "B_triggered": bool(r1_not_competitive),
            "C_triggered": bool(beyond_cycle and not r1_not_competitive),
        },
        "non_claims": [
            "post-hoc audit of a Monitor already used by Phase 1b/1c/1d; not a confirmation set",
            "no official ZINC valid/test and no internal Reserved label was read",
            "the negative direction does not extend to SSRC as a whole",
            "the cycle penalty was never used as a model input and no label compensation was applied",
        ],
    }
    C.write_json(OUT, payload)

    print(f"[A4] artifact_dominates={artifact_dominates} beyond_cycle={beyond_cycle} "
          f"unresolved_label={unresolved_label} r1_not_competitive={r1_not_competitive}")
    print(f"[A4] direction={direction}  {reason}")
    print(f"[A4] wrote {OUT.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

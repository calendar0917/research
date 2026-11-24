"""Build summary.json from the frozen per-stage JSON artifacts."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import sfec_common as C


def load(name: str):
    return json.loads((C.ANALYSIS_DIR / name).read_text())


def main() -> int:
    p1p, p1z = load("p1_pcqm.json"), load("p1_zinc.json")
    p2p, p2z = load("p2_pcqm.json"), load("p2_zinc.json")
    p3p, p3z = load("p3_pcqm.json"), load("p3_zinc.json")
    support = load("structural_support.json")
    checks = load("checks.json")
    valid = load("valid_transfer_pcqm.json")
    identity = load("data_identity.json")
    p1_sup = p1p["transfer"]["supported"]
    p1_ctx = p1_sup["context_equal"]
    p1_pair = p1_sup["pair_equal"]
    p2_pool = p2p["controlled_masking"]["pool"]
    p3_all = p3p["metrics"]["all"]
    p3_high = p3p["metrics"]["high_nonadditivity"]
    # examples: two stable candidates, two positive composition cases, two failures
    examples = [
        {
            "kind": "functional_equivalence_positive",
            "pair": "*OC -> *OCC",
            "discovery_n": 809, "mu": 0.0, "p80_abs_dev": 0.09523984767,
            "evaluation_n": 389, "fraction_within_tau_0.1": 0.7866323907,
            "median_eval_abs_residual": 0.01904796953,
            "reading": "near-zero stabilised replacement at tau=0.10; majority of evaluation residuals stay inside tau",
        },
        {
            "kind": "functional_equivalence_positive",
            "pair": "*CCCCC -> *CC1CC1",
            "discovery_n": 63, "mu": 0.002721138505, "p80_abs_dev": 0.07074960113,
            "evaluation_n": 17, "fraction_within_tau_0.1": 0.7647058824,
            "median_eval_abs_residual": 0.07347073963,
            "reading": "small stable candidate, but only 17 evaluation observations",
        },
        {
            "kind": "composition_positive",
            "pair": "*C -> *C#N",
            "evaluation_n": 442, "shared_pred": -0.2762, "pair_full_mu": -0.3374,
            "mae_zero": 0.4154, "mae_shared": 0.3470, "mae_pair_full": 0.3491,
            "relative_gain_shared_vs_zero": 0.1645,
            "reading": "one of the clearer additive-transfer successes; SHARED nearly matches direct PAIR-FULL",
        },
        {
            "kind": "composition_positive_but_local_helps_more",
            "pair": "*C -> *C(=O)O",
            "evaluation_n": 635, "shared_pred": -0.3939, "pair_full_mu": -0.2884,
            "mae_zero": 0.4979, "mae_shared": 0.4367, "mae_pair_full": 0.4295,
            "relative_gain_shared_vs_zero": 0.1229,
            "reading": "positive additive gain, but SHARED over-predicts the replacement; condition corrections reduce this pair's MAE",
        },
        {
            "kind": "composition_failure",
            "pair": "*N -> *O",
            "evaluation_n": 1452, "shared_pred": 0.2172, "pair_full_mu": 0.0789,
            "mae_zero": 0.1924, "mae_shared": 0.2166, "mae_pair_full": 0.1817,
            "relative_gain_shared_vs_zero": -0.1260,
            "reading": "additive SHARED is worse than ZERO; the direct pair effect is strongly context dependent",
        },
        {
            "kind": "composition_failure",
            "pair": "*Cl -> *F",
            "evaluation_n": 531, "shared_pred": 0.2061, "pair_full_mu": 0.0517,
            "mae_zero": 0.1282, "mae_shared": 0.1819, "mae_pair_full": 0.1154,
            "relative_gain_shared_vs_zero": -0.4190,
            "reading": "largest negative pooled example; b_F-b_Cl does not transfer to this replacement",
        },
    ]
    payload = {
        "round": "sfec_v0",
        "generated_utc": datetime.now(timezone.utc).isoformat(),
        "status": "completed",
        "protocol": "PROTOCOL.md",
        "config": "configs/sfec_v0.json",
        "headline": {
            "p1_pcqm": p1p["judgement"],
            "p2_pcqm": p2p["judgement"],
            "p3_pcqm": p3p["judgement"],
            "zinc": "INSUFFICIENT_CROSSED_SUPPORT",
            "one_sentence": (
                "PCQM-1M has enough crossed structure for partial response reuse, but additive SHARED composition "
                "and the overall LOCAL context rule stay below their pre-registered practical screens; "
                "ZINC is structurally acyclic and cannot test crossed response reuse."
            ),
        },
        "judgements": {
            "pcqm": {"P1": p1p["judgement"], "P2": p2p["judgement"], "P3": p3p["judgement"]},
            "zinc": {"P1": p1z["judgement"], "P2": p2z["judgement"], "P3": p3z["judgement"]},
        },
        "answers": [
            {
                "q": "Q1. Is there enough crossed structural support for transfer, and which dataset qualifies?",
                "answer": (
                    "PCQM-1M qualifies: 4,824,824 cells, 3,166,455 C contexts, 987,596 molecules; "
                    "primary 64-S graph has 3 components (largest 3,010,213), 2-core 419,353, cycle rank 777,293; "
                    "1,602 supported pairs and 96,991 disjoint evaluation pair observations. "
                    "ZINC does not: 0 supported pairs, primary graph 2-core 6 and cycle rank 1, 0 evaluation pair observations."
                ),
                "pcqm_supported_pairs": p1p["n_supported_pairs"],
                "zinc_supported_pairs": p1z["n_supported_pairs"],
            },
            {
                "q": "Q2. Does a discovery response reuse on evaluation contexts (P1)?",
                "answer": (
                    "PCQM: PARTIAL_RESPONSE_REUSE. Context-equal PAIR MAE {:.5f} vs ZERO {:.5f} "
                    "(gain {:.2f}%, paired 95% CI for ZERO-PAIR [{:.5f}, {:.5f}]); sign accuracy {:.3f}. "
                    "Pair-equal gives MAE {:.5f} vs {:.5f} (gain {:.2f}%). "
                    "ZINC: INSUFFICIENT_CROSSED_SUPPORT."
                ).format(
                    p1_ctx["PAIR"]["mae"], p1_ctx["ZERO"]["mae"],
                    100 * p1_ctx["PAIR"]["skill_vs_zero"],
                    p1_ctx["diff_ZERO_minus_PAIR"]["ci"][0], p1_ctx["diff_ZERO_minus_PAIR"]["ci"][1],
                    p1_ctx["PAIR"]["sign_accuracy"],
                    p1_pair["PAIR"]["mae"], p1_pair["ZERO"]["mae"],
                    100 * p1_pair["PAIR"]["skill_vs_zero"],
                ),
                "context_equal": {
                    "zero_mae": p1_ctx["ZERO"]["mae"], "pair_mae": p1_ctx["PAIR"]["mae"],
                    "gain": p1_ctx["PAIR"]["skill_vs_zero"],
                    "diff_ci": p1_ctx["diff_ZERO_minus_PAIR"]["ci"],
                    "sign_accuracy": p1_ctx["PAIR"]["sign_accuracy"],
                },
            },
            {
                "q": "Q3. Are stable functional-equivalence candidates useful, and how much do they cover (P1)?",
                "answer": (
                    "Only narrowly. At tau=0.10 there are 3 frozen candidates, covering 0.19% of supported pairs; "
                    "77.8% of their 415 evaluation observations stay within tau. At tau=0.20 there are 151 candidates "
                    "(9.43% coverage) and 83.1% stay within tau. Stable equivalence is real but sparse; "
                    "it is not a broad replacement dictionary."
                ),
                "tau_0.10": p1p["candidates"]["0.1"],
                "tau_0.20": p1p["candidates"]["0.2"],
            },
            {
                "q": "Q4. Does additive SHARED composition support unobserved replacement (P2)?",
                "answer": (
                    "No at the pre-registered 15% screen. Natural unobserved pairs: 0. Controlled masking: 16 applicable "
                    "pairs, 18,584 evaluation observations. SHARED MAE {:.5f} vs ZERO {:.5f}: gain {:.2f}% "
                    "(CI [{:.5f}, {:.5f}]); PAIR-FULL reference {:.5f}, so SHARED remains {:.5f} worse. "
                    "Macro pair-equal gain {:.2f}%. Verdict COMPOSITION_SCREEN_ONLY. "
                    "Supplementary condition corrections fitted after masking give MAE {:.5f} "
                    "(gain {:.2f}% vs SHARED), showing that context rules, not the global additive term, carry most of the signal."
                ).format(
                    p2_pool["mae_shared"], p2_pool["mae_zero"],
                    100 * p2_pool["relative_gain_shared_vs_zero"],
                    p2_pool["diff_zero_minus_shared"]["ci"][0], p2_pool["diff_zero_minus_shared"]["ci"][1],
                    p2_pool["mae_pair_full"], p2_pool["gap_shared_vs_pair_full"],
                    100 * p2_pool["macro_pair_equal"]["relative_gain"],
                    p2_pool["mae_local"], 100 * p2_pool["relative_gain_local_vs_shared"],
                ),
                "natural_unobserved_pairs": p2p["natural_unobserved"]["n_pairs"],
                "masked_pairs": p2_pool["n_pairs"],
                "masked_observations": p2_pool["n_obs"],
                "shared_gain": p2_pool["relative_gain_shared_vs_zero"],
                "shared_gain_ci": p2_pool["diff_zero_minus_shared"]["ci"],
                "pair_full_mae": p2_pool["mae_pair_full"],
                "local_after_masking_gain_vs_shared": p2_pool["relative_gain_local_vs_shared"],
            },
            {
                "q": "Q5. Does a LOCAL context rule beat SHARED and a matched NULL (P3)?",
                "answer": (
                    "Directionally yes, but below the overall 10% screen: 114 supported q groups, correction coverage "
                    "{:.2f}%; LOCAL MAE {:.5f} vs SHARED {:.5f} (gain {:.2f}%, CI [{:.5f}, {:.5f}]) and vs "
                    "LOCAL_NULL {:.5f} (gain {:.2f}%). On the discovery-frozen high-non-additivity quartile "
                    "({} pairs, {} observations), LOCAL gain is {:.2f}% (CI [{:.5f}, {:.5f}]). "
                    "Verdict CONTEXT_RULE_SCREEN_ONLY: real q signal, but the pre-registered overall 10% bar is not met."
                ).format(
                    100 * p3p["correction_coverage"],
                    p3_all["LOCAL"]["mae"], p3_all["SHARED"]["mae"],
                    100 * p3_all["relative_gain_local_vs_shared"],
                    p3_all["diff_SHARED_minus_LOCAL"]["ci"][0], p3_all["diff_SHARED_minus_LOCAL"]["ci"][1],
                    p3_all["LOCAL_NULL"]["mae"], 100 * p3_all["relative_gain_local_vs_null"],
                    p3_high["n_pairs"], p3_high["n_obs"],
                    100 * p3_high["relative_gain_local_vs_shared"],
                    p3_high["diff_SHARED_minus_LOCAL"]["ci"][0], p3_high["diff_SHARED_minus_LOCAL"]["ci"][1],
                ),
                "q_supported": p3p["n_q_supported"], "q_total": p3p["n_q_total"],
                "coverage": p3p["correction_coverage"],
                "local_gain_vs_shared": p3_all["relative_gain_local_vs_shared"],
                "local_gain_vs_null": p3_all["relative_gain_local_vs_null"],
                "high_nonadditivity_gain": p3_high["relative_gain_local_vs_shared"],
            },
            {
                "q": "Q6. Does ZINC reproduce or contradict the PCQM conclusions?",
                "answer": (
                    "ZINC is a structural negative control, not a contradiction: its primary bipartite graph is almost "
                    "a forest (44 components, largest 20,194, 2-core 6, cycle rank 1), so 0 pairs have both >=10 discovery "
                    "and >=5 evaluation contexts, and there are 0 evaluation pair observations. P1/P2/P3 are labelled "
                    "INSUFFICIENT_CROSSED_SUPPORT or CONTEXT_SUPPORT_INSUFFICIENT; no label-based claim is made."
                ),
                "zinc_bipartite_primary": support["zinc"]["bipartite_primary"],
                "zinc_p1_pass": support["zinc"]["support_gates"]["p1_pass"],
            },
            {
                "q": "Q7. Does the official PCQM Valid split confirm transfer without touching Test labels?",
                "answer": (
                    "Yes, descriptively. On 36,952 Valid pair observations with a frozen discovery median, PAIR MAE "
                    "{:.5f} vs ZERO {:.5f} (skill {:.2f}%, sign {:.3f}); unseen exact contexts: skill {:.2f}%; "
                    "seen exact contexts: skill {:.2f}%. 7,356 cells from 1,819 Valid molecules duplicate Train graphs "
                    "and were excluded. Official Test labels were never read; Test was not built."
                ).format(
                    valid["transfer"]["all_pairs_with_discovery_mu"]["pair_mae"],
                    valid["transfer"]["all_pairs_with_discovery_mu"]["zero_mae"],
                    100 * valid["transfer"]["all_pairs_with_discovery_mu"]["skill"],
                    valid["transfer"]["all_pairs_with_discovery_mu"]["sign_accuracy"],
                    100 * valid["transfer"]["unseen_exact_context"]["skill"],
                    100 * valid["transfer"]["seen_exact_context"]["skill"],
                ),
                "valid": valid["transfer"],
                "duplicate_graph_cells": valid["n_cells_duplicate_graph"],
                "duplicate_graph_molecules": valid["n_valid_molecules_duplicate_train_graph"],
            },
            {
                "q": "Q8. Are identity, reconstruction, labels, splits, and controls clean enough to trust the verdicts?",
                "answer": (
                    "Yes within the recorded limits. PCQM graph reassembly: 57,873 sampled cuts, 0 graph failures; "
                    "stereo exactness 91.1% (5,128 mismatches, reported separately). ZINC: 20,000 cuts, 0 exact-certificate "
                    "failures. 1,000/1,000 sampled PCQM SMILES and labels match the raw CSV; all 1M rows are unique and "
                    "inside official Train; discovery/evaluation molecule overlap is 0; primary evaluation contexts share "
                    "no molecule. 10,557 duplicate PCQM molecules carry labels differing by up to 4.94 eV; they are one identity "
                    "but the label conflict is a known noise source. All five FP64 controls pass. All checks pass."
                ),
                "checks_all_ok": checks["all_ok"],
                "pcqm_graph_reassembly": checks["pcqm"]["checks"]["reassembly_sampled"],
                "zinc_exact_reassembly": checks["zinc"]["checks"]["exact_reassembly"],
                "duplicate_molecule_labels": checks["pcqm"]["checks"]["duplicate_molecule_labels"],
                "controls": checks["controls"],
            },
        ],
        "data": {
            "pcqm_train_1m": {
                "cells": support["pcqm_train_1m"]["n_cells_total"],
                "contexts": support["pcqm_train_1m"]["n_C"],
                "S_types": support["pcqm_train_1m"]["n_S"],
                "molecules": support["pcqm_train_1m"]["n_molecules_total"],
                "q_types": support["pcqm_train_1m"]["n_q"],
                "coverage": support["pcqm_train_1m"]["coverage"],
                "bipartite_primary": support["pcqm_train_1m"]["bipartite_primary"],
                "support_gates": support["pcqm_train_1m"]["support_gates"],
            },
            "pcqm_train_100k": {
                "cells": support["pcqm_train_100k"]["n_cells_total"],
                "contexts": support["pcqm_train_100k"]["n_C"],
                "supported_pairs": support["pcqm_train_100k"]["support_gates"]["p1_supported_pairs"],
                "evaluation_pair_observations": support["pcqm_train_100k"]["support_gates"]["p1_evaluation_pair_observations"],
                "bipartite_primary": support["pcqm_train_100k"]["bipartite_primary"],
            },
            "zinc": {
                "cells": support["zinc"]["n_cells_total"],
                "contexts": support["zinc"]["n_C"],
                "S_types": support["zinc"]["n_S"],
                "molecules": support["zinc"]["n_molecules_total"],
                "q_types": support["zinc"]["n_q"],
                "coverage": support["zinc"]["coverage"],
                "bipartite_primary": support["zinc"]["bipartite_primary"],
                "support_gates": support["zinc"]["support_gates"],
            },
            "pcqm_valid": valid,
        },
        "p1": {
            "pcqm": p1p,
            "zinc": p1z,
        },
        "p2": {
            "pcqm": p2p,
            "zinc": p2z,
        },
        "p3": {
            "pcqm": p3p,
            "zinc": p3z,
        },
        "examples": examples,
        "checks": checks,
        "identity": {
            "git_commit": identity["git_commit"],
            "git_dirty": identity["git_dirty"],
            "protocol_sha256": identity["protocol"]["sha256"],
            "config_sha256": identity["config"]["sha256"],
            "pcqm_raw_sha256": identity["datasets"]["pcqm"]["raw_files"]["data.csv.gz"]["sha256"],
            "zinc_train_sha256": identity["datasets"]["zinc"]["raw_files"]["train.pickle"]["sha256"],
            "test_labels_read": False,
        },
        "limitations": [
            "PCQM-1M is a prefix of official Train, not the full 3.38M Train rows; support is ample, so no full-Train expansion was justified.",
            "Identity is graph-level; 8.9% of sampled PCQM stereo reassemblies differ and are tracked separately.",
            "Duplicate whole graphs can carry conflicting labels (max 4.94 eV); they share one molecule identity but add label noise.",
            "P2/P3 are observational screens with context-level bootstrap; they are not causal or population theorems.",
            "The 64-S vocabulary, tau, ridge lambda, and 10%/15% screens are fixed practical choices chosen before labels.",
            "ZINC is a structural negative control: almost no cycles means the crossed pair design is unavailable.",
        ],
        "deviations": [
            "ZINC exact-certificate reassembly was capped at 20,000 cuts for runtime; all 66,494 cut invariants and all PCQM graph reassembly checks passed.",
            "PCQM functional-equivalence candidate rows with zero evaluation observations are reported but excluded from fraction-within-tau aggregates.",
            "P2 condition corrections after masking are supplementary to the pre-registered SHARED judgment; the P2 verdict remains based on SHARED.",
            "The P1 scaffold-cluster bootstrap and the P2 post-masking LOCAL scoring were executed as protocol-specified secondary analyses after the primary P1/P2 runs; no threshold, vocabulary, split, or primary verdict was changed.",
        ],
    }
    C.write_json(C.ANALYSIS_DIR / "summary.json", payload)
    print(json.dumps(payload["headline"], indent=2))
    print("answers", len(payload["answers"]), "examples", len(payload["examples"]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
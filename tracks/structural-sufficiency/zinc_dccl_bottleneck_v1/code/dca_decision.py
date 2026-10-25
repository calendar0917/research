"""Final decision artifact: condenses phase A + phase B into the frozen numbers the
report quotes, plus the label-quantity extrapolation.

No new data is read here; only artifacts produced earlier in this round.
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

import dca_common as C

CHAMPION_VALID = 0.075990316172   # 10k DCC-L B-R0, official-valid seed mean (best-valid)
TEST_VALID_OFFSET = -0.0087       # historical B-R0: valid 0.07599 -> test 0.06732


def main() -> int:
    g = json.loads((C.ANALYSIS / "groups.json").read_text())
    p2 = json.loads((C.ANALYSIS / "analysis_part2.json").read_text())
    b = json.loads((C.ANALYSIS / "phase_b.json").read_text())
    pm = pd.read_csv(C.ANALYSIS / "per_molecule_features.csv")

    per = {}
    for k, v in b["per_run"].items():
        per[k] = {
            "locked": v["reads"]["best_monitor"]["locked_mae"],
            "locked_final": v["reads"]["final"]["locked_mae"],
            "monitor": v["reads"]["best_monitor"]["monitor_mae"],
            "official_valid": v["reads"]["best_monitor"]["official_valid_mae"],
            "gpu_hours": v["gpu_hours"],
            "epochs": v["epochs_run"],
            "tail10_train": v["tail10_median_online_train_mae"],
        }
    arm = {a: {m: float(np.mean([per[f"{a}_s{s}"][m] for s in (0, 1)]))
               for m in ("locked", "locked_final", "monitor", "official_valid", "tail10_train")}
           for a in ("fit4k", "fit8k")}
    arm["fit4k"]["gpu_hours"] = float(sum(per[f"fit4k_s{s}"]["gpu_hours"] for s in (0, 1)))
    arm["fit8k"]["gpu_hours"] = float(sum(per[f"fit8k_s{s}"]["gpu_hours"] for s in (0, 1)))

    v4, v8 = arm["fit4k"]["official_valid"], arm["fit8k"]["official_valid"]
    alpha_4_8 = float(np.log(v4 / v8) / np.log(2.0))
    alpha_8_10 = float(np.log(v8 / CHAMPION_VALID) / np.log(10000 / 8000))
    need_valid = 0.060 - TEST_VALID_OFFSET
    proj = {}
    for name, alpha in (("alpha_from_4k_8k", alpha_4_8), ("alpha_from_8k_10k", alpha_8_10),
                        ("alpha_mean", 0.5 * (alpha_4_8 + alpha_8_10))):
        proj[name] = {
            "alpha": alpha,
            "n_labels_for_valid_%.4f" % need_valid: float(
                10000 * (CHAMPION_VALID / need_valid) ** (1.0 / alpha)),
            "valid_at_12k": float(CHAMPION_VALID * (12000 / 10000) ** (-alpha)),
        }

    ae = pm.ae_mean.to_numpy()
    full = pm.fitset_supported_frac.to_numpy() >= 0.999
    dec = {
        "round": "zinc-dccl-bottleneck-v1",
        "official_test_read": False,
        "phase_a": {
            "mae_seed_mean_official_valid": g["overall"]["mae_seed_mean"],
            "unseen_n": int((pm.r2_unseen_atom_frac.to_numpy() > 0).sum()),
            "unseen_mae": p2["headroom"]["mae_unseen"],
            "seen_mae": p2["headroom"]["mae_seen"],
            "unseen_ae_share": p2["headroom"]["unseen_ae_share"],
            "fully_supported_n": int(full.sum()),
            "fully_supported_mae": float(ae[full].mean()),
            "fully_supported_ae_share": float(ae[full].sum() / ae.sum()),
            "under_supported_n": int((~full).sum()),
            "under_supported_mae": float(ae[~full].mean()),
            "under_supported_ae_share": float(ae[~full].sum() / ae.sum()),
            "dose_response": p2["dose_response_unseen_frac"],
            "confounder_partial_spearman": g["confounders"][
                "spearman_partial_natoms_cyclerank_absy"],
            "standardised_delta_unseen_minus_seen": g["confounders"][
                "delta_standardised_by_size_ring"],
            "per_seed_delta_unseen_minus_seen": g["confounders"][
                "per_seed_delta_unseen_minus_seen"],
            "max_bias_share_of_mse_cell": float(max(
                c["bias_share_of_mse"] if "bias_share_of_mse" in c else 0.0
                for c in g["tables_mece"]["U_x_T_x_R"])),
            "hard_core_top100": p2["hard_core"],
            "similarity_partial_simtopk": p2["similarity"]["wl2_idf_cosine"][
                "spearman_simtopk_ae_partial"],
            "similarity_partial_nnystd": p2["similarity"]["wl2_idf_cosine"][
                "spearman_nnystd_ae_partial"],
            "knn_reference_mae": p2["similarity"]["wl2_idf_cosine"]["knn_reference_mae"],
        },
        "phase_b": {
            "total_updates": 47400,
            "arms": arm,
            "delta_locked_seed0": b["contrast_fit8k_minus_fit4k"]["seed0"]["locked_delta_best_monitor"],
            "delta_locked_seed1": b["contrast_fit8k_minus_fit4k"]["seed1"]["locked_delta_best_monitor"],
            "ci_seed0": b["contrast_fit8k_minus_fit4k"]["seed0"]["ci95"],
            "ci_seed1": b["contrast_fit8k_minus_fit4k"]["seed1"]["ci95"],
            "relative_reduction_locked": float(1 - arm["fit8k"]["locked"] / arm["fit4k"]["locked"]),
            "tier_deltas": {
                s: {k: b["contrast_fit8k_minus_fit4k"][s][k]
                    for k in ("newly_supported", "still_unseen", "seen_at_4k")}
                for s in ("seed0", "seed1")},
            "frac_improved": {s: b["contrast_fit8k_minus_fit4k"][s]["frac_molecules_improved"]
                              for s in ("seed0", "seed1")},
            "budget_gpu_hours": arm["fit4k"]["gpu_hours"] + arm["fit8k"]["gpu_hours"],
        },
        "label_quantity_extrapolation": {
            "official_valid_by_fit_size": {"4000": v4, "8000": v8,
                                           "10000_historical_bestvalid": CHAMPION_VALID},
            "valid_needed_for_test_0.060_at_offset_%.4f" % TEST_VALID_OFFSET: need_valid,
            "projections": proj,
        },
        "other_mechanisms_measured_historically": {
            "seed_average_p3a_2seeds_valid_gain": 0.004456,
            "best_structure_mechanism_gain": 0.0048,
            "ema": "not measured (broken in zinc_performance_search_v1)",
        },
    }
    C.write_json(C.ANALYSIS / "decision.json", dec)
    print(json.dumps(dec["phase_b"] | {"label_quantity_extrapolation":
                                       dec["label_quantity_extrapolation"]}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

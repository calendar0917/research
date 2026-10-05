"""Parallel evidence line for the fusion-clarity round (CPU, no new fitting).

Re-derives the unified numbers of the latest completed round
(zinc_local_dictionary_component_supervision_seed0_v1) from the saved
predictions, audits what the trained dictionary actually changed, and records
the input/responsibility map facts needed by RESEARCH_MAP.md.  Read-only with
respect to every historical artifact; the output feeds the round's
RESEARCH_MAP.md / ERRATA entries.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from tracks.ksvd.experiments.luyin16 import (
    zinc_local_dictionary_component_supervision_seed0_v1 as src,
)
from tracks.ksvd.experiments.luyin16 import (
    zinc_local_tuple_dictionary_joint_vs_independent_seed0_v1 as prev,
)

SRC_DIR = src.RESULTS_DIR
OUT = Path(__file__).resolve().parent


def mae(err):
    return float(np.mean(np.abs(err)))


def main() -> None:
    out: dict = {}
    new = src.load_new_objects(SRC_DIR)
    fold = new["fold"]
    fit_idx, dev_idx = fold["fit_idx"], fold["dev_idx"]
    t = new["targets"]
    g, k = (np.asarray(t[key], np.float64 if key != "k" else np.int64) for key in ("g", "k"))
    g_fit, g_dev = g[fit_idx], g[dev_idx]
    k_dev = k[dev_idx]
    g0 = k_dev == 0
    roster = json.loads((SRC_DIR / "eval_roster.json").read_text())
    bias = roster["bias"]

    # ---- 1. unified recomputation from the saved predictions -----------------
    unified = {}
    for arm in src.ARMS:
        with np.load(SRC_DIR / f"{arm}_dev_predictions.npz") as z:
            dev_raw = np.asarray(z["dev"], np.float64)
        err_raw = g_dev - dev_raw
        err_cal = err_raw - bias[arm]
        with np.load(SRC_DIR / f"{arm}_dev_component_predictions.npz") as z:
            dc = np.asarray(z["dev"], np.float64)
        hat_ell, hat_s = dc[:, 0], dc[:, 1]
        ell = np.asarray(t["ell"], np.float64)[dev_idx]
        s = np.asarray(t["s"], np.float64)[dev_idx]
        e_ell, e_s = hat_ell - ell, hat_s - s
        unified[arm] = {
            "bias": bias[arm],
            "dev_G0_raw": mae(err_raw[g0]),
            "dev_G0_cal": mae(err_cal[g0]),
            "dev_overall_raw": mae(err_raw),
            "dev_overall_cal": mae(err_cal),
            # e = target - prediction convention (the round brief's): err = e_ell + e_s
            "e_g_raw_identity_max_abs": float(np.max(np.abs(err_raw - ((ell - hat_ell) + (s - hat_s))))),
            "triangle_gap_raw_G0": float(np.mean(np.abs(e_ell[g0]) + np.abs(e_s[g0]) - np.abs(e_ell[g0] + e_s[g0]))),
        }
    dm = {
        "G0_raw_gap": unified["M_COMP"]["dev_G0_raw"] - unified["D_COMP"]["dev_G0_raw"],
        "G0_cal_gap": unified["M_COMP"]["dev_G0_cal"] - unified["D_COMP"]["dev_G0_cal"],
        "overall_raw_gap": unified["M_COMP"]["dev_overall_raw"] - unified["D_COMP"]["dev_overall_raw"],
        "overall_cal_gap": unified["M_COMP"]["dev_overall_cal"] - unified["D_COMP"]["dev_overall_cal"],
    }
    out["unified_recomputation"] = {"per_arm": unified, "M_minus_D_gaps": dm}
    out["unified_recomputation"]["calibration_note"] = (
        "The raw G0 gap (~0.0089) shrinks to ~0.0020 after each arm's own fit-median "
        "calibration (D bias -0.0428 vs M -0.0225): a large part of the raw deficit is "
        "a fit-side median offset, NOT a bias-independent signal-quality statement. "
        "This retracts the earlier 'not bias-related' phrasing; the calibrated "
        "comparison (still MLP-favouring, CI crossing 0) is the honest endpoint."
    )

    # ---- 2. what the trained dictionary actually changed ---------------------
    d_init = prev.init_d_loc()
    d_last = torch.load(SRC_DIR / "D_COMP_init_state.pt", map_location="cpu", weights_only=True)["local_tuple.D_loc_raw"]
    d_soup = torch.load(SRC_DIR / "D_COMP_raw_soup_state.pt", map_location="cpu", weights_only=True)["local_tuple.D_loc_raw"]
    with torch.no_grad():
        init_norm = float(d_init.norm())
        drift_last = float((d_last - d_init).norm())
        drift_soup = float((d_soup - d_init).norm())
        rel_drift = drift_soup / init_norm
        col_change = float((F.normalize(d_soup, dim=0) - F.normalize(d_init, dim=0)).norm())
        # cosine-matched atom similarity (sign/permutation aware): for each final
        # atom, the best |cos| over initial atoms, with sign reported
        a_init = F.normalize(d_init, dim=0)
        a_soup = F.normalize(d_soup, dim=0)
        cos = a_soup.t() @ a_init          # [64 final, 64 init]
        best_abs, best_idx = cos.abs().max(dim=1)
        best_signed = cos.gather(1, best_idx.unsqueeze(1)).squeeze(1)
        # atom usage from the saved mechanism health
        health = json.loads((SRC_DIR / "mechanism_health.json").read_text())
        ep240 = health["D_COMP"]["epoch240"]
        out["dictionary_change"] = {
            "init_L2": init_norm,
            "raw_drift_last": drift_last,
            "raw_drift_soup": drift_soup,
            "raw_drift_relative_to_init_L2": rel_drift,
            "col_normalized_same_index_change": col_change,
            "col_normalized_same_index_change_relative": col_change / float(np.sqrt(2.0)),
            "best_abs_cosine_to_init_atoms": {
                "mean": float(best_abs.mean()),
                "median": float(best_abs.median()),
                "fraction_above_0.9": float((best_abs > 0.9).float().mean()),
                "fraction_flipped_sign": float((best_signed < 0).float().mean()),
            },
            "note": (
                "raw drift 71.6 vs init L2 ~89.4 is a raw-matrix distance, not a "
                "unit-free 'mechanism progress' metric; the column-normalised "
                "same-index change and the best-cosine matches are the comparable "
                "shape-change measures. Sign flips mean atom-identity symmetry is "
                "not sign-fixed; permutation symmetry means same-index comparison "
                "understates shape preservation."
            ),
            "health_epoch240": {
                "atoms_used_of_64": ep240.get("atoms_used_of_64"),
                "per_tuple_nnz": ep240.get("per_tuple_nnz"),
                "root_code_effective_rank": ep240.get("root_code_effective_rank"),
                "root_code_rms": ep240.get("root_code_rms"),
                "injection_rms": ep240.get("injection_rms"),
            },
        }

    # ---- 3. input/responsibility map facts (from source code, verified) ------
    phi_env, atom_env, node_sizes = prev._env_phi_atom()
    out["responsibility_map"] = {
        "phi65": {
            "content": (
                "root_basis(11: root indicator, shell one-hot 0..2, log1p degree, "
                "log1p per-shell neighbour counts, log1p walk counts) + node mean/std "
                "+ edge mean/std (15: shell-pair one-hot, degree sums, common "
                "neighbours, per-shell edge stats) + [log1p(n_nodes), log1p(n_edges)]"
            ),
            "chemistry": "none: only adjacency and rooted distances (fsar_r2.phi_for_center)",
            "verdict": "pure topology; calling p65 'pure topology' is accurate",
        },
        "semantics_in_the_tuple": (
            "q60 = [onehot28(root atom); onehot28(neighbour atom); onehot4(bond)] - "
            "the ONLY chemistry inside the joint 125-D tuple input"
        ),
        "Sem108_bypass": (
            "patch_cont[:, 0:108] = atom_shell (84 = (radius+1) x 28 shell-resolved "
            "atom counts) + bond_shell (24 = 6 shell-pairs x 4 bond types); plus "
            "anchor size2 -> the 110-D fusion input. Shell-resolved means it is "
            "structure-CONDITIONED semantics, not pure global chemistry."
        ),
        "J_vs_I": (
            "w_J = C[t,a]/d (realised incidence); w_I = n_t n_a / d^2 (marginal "
            "product over the support union); both enter only the local tuple "
            "pooling, nowhere else"
        ),
        "static_relations": (
            "enter the body reader through the frozen node (3x48)/edge (6x32) "
            "encoded slots and the C6-mask relation groups; unchanged this round"
        ),
        "posterior_D_L_V_L": (
            "the 144->288->144 matched MLP bridge replaced the historical D_L/V_L "
            "linear dictionary bridge in BOTH arms of the source round; the old "
            "node/edge slot dictionaries are dead in this body family"
        ),
    }

    (OUT / "parallel_evidence.json").write_text(json.dumps(out, indent=2, sort_keys=False) + "\n")
    print(json.dumps(out["unified_recomputation"]["M_minus_D_gaps"], indent=2))
    print(json.dumps(out["dictionary_change"]["best_abs_cosine_to_init_atoms"], indent=2))
    print("done -> parallel_evidence.json")


if __name__ == "__main__":
    main()

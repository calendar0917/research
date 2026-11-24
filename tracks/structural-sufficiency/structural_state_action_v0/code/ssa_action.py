"""Section 6: what the environment still does when the edit record is identical.

Filter (structure only -- no label, no prediction, no error):

* same canonical replacement ``A -> B``;
* two different exact contexts ``C1 != C2`` with four real, clean, distinct
  Valid molecules;
* ``E_AB(C1) = E_AB(C2)`` as *signed multisets* of root records.

The subset is then split by whether the edit is non-empty, and a stricter
subset additionally requires ``dg(C1) = dg(C2)`` bitwise on the raw graph
descriptor.  For that strict subset the input-state increment is identical, so
any remaining model difference is a nonlinear readout of two *different initial
states* -- not a different input increment.  Nothing synthetic is constructed:
every state belongs to a real Valid molecule.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

import ssa_common as C
import ssa_interaction as IA
import ssa_records as REC

MODELS = ("ZERO", "PAIR_Q", "LOCAL", "LCRC")


def _metrics_block(d_true: np.ndarray, preds: Mapping[str, np.ndarray],
                   pair_index: np.ndarray, n_pairs: int, seed: int) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for model in MODELS:
        dp = preds[model]
        obs_m = IA.dd_metrics(d_true, dp)
        mac = IA.macro_metrics(d_true, dp, pair_index, n_pairs, seed=seed)
        out[model] = {
            "observation": obs_m,
            "pair_type_macro": {k: mac[k] for k in ("n_pairs", "mae_eV", "rmse_eV",
                                                    "dir_acc", "dir_n", "ci_mae")},
        }
    out["contrasts"] = {}
    for other in ("ZERO", "PAIR_Q", "LOCAL"):
        c = IA.macro_contrast(np.abs(preds["LCRC"] - d_true),
                              np.abs(preds[other] - d_true), pair_index, n_pairs, seed=seed + 1)
        out["contrasts"][f"LCRC_minus_{other}"] = c
    return out


def analyse(obs: Mapping[str, Any], sel: Mapping[str, Any], sample: Mapping[str, Any],
            states: Mapping[str, Any], U: np.ndarray, machine: Any,
            frozen_pred_eV_mol: np.ndarray, interaction: Mapping[str, Any]) -> dict[str, Any]:
    rect = sel["rect"]
    rect_idx = np.asarray(sample["rect_index"], np.int64)
    mol_of = sample["molecule_cols"]
    mol_index = {int(m): i for i, m in enumerate(np.asarray(states["mol_ids"]).tolist())}
    atom_start = np.asarray(states["atom_start"], np.int64)
    atom_stop = np.asarray(states["atom_stop"], np.int64)
    obs_mol_cache_row = np.asarray(obs["mol_cache_row"], np.int64)
    pair_types = np.asarray(sel["pair_types"], np.int64)
    n_pairs = int(pair_types.size)
    pair_index_all = np.searchsorted(pair_types, rect["pair_key"])

    # one vectorised reconstruction of every sample molecule's prediction
    pred_state = machine.readout(U, states["g"])["pred_eV"]

    rows: list[dict[str, Any]] = []
    n = len(rect_idx)
    same_edit = np.zeros(n, bool)
    same_g = np.zeros(n, bool)
    edit_empty = np.zeros(n, bool)
    edit_size = np.zeros(n, np.int64)
    du_gap = np.zeros(n, np.float64)
    du_gap_rel = np.zeros(n, np.float64)
    distinct = np.zeros(n, bool)
    d_lcrc_recon_err = np.zeros(n, np.float64)
    d_lcrc_state = np.zeros(n, np.float64)
    for pos, i in enumerate(rect_idx.tolist()):
        m = [int(mol_of[c][i]) for c in ("m1", "m2", "m3", "m4")]
        distinct[pos] = bool(m[0] != m[1] and m[2] != m[3] and m[0] != m[2] and m[1] != m[3])
        ia, ib, ja, jb = (mol_index[x] for x in m)
        e1 = REC.multiset_edit(
            REC.multiset_counts(states["rec_hi"][atom_start[ib]:atom_stop[ib]],
                                states["rec_lo"][atom_start[ib]:atom_stop[ib]]),
            REC.multiset_counts(states["rec_hi"][atom_start[ia]:atom_stop[ia]],
                                states["rec_lo"][atom_start[ia]:atom_stop[ia]]),
        )
        e2 = REC.multiset_edit(
            REC.multiset_counts(states["rec_hi"][atom_start[jb]:atom_stop[jb]],
                                states["rec_lo"][atom_start[jb]:atom_stop[jb]]),
            REC.multiset_counts(states["rec_hi"][atom_start[ja]:atom_stop[ja]],
                                states["rec_lo"][atom_start[ja]:atom_stop[ja]]),
        )
        same = REC.edit_equal(e1, e2)
        same_edit[pos] = same
        edit_size[pos] = int(np.abs(e1["count"]).sum()) if e1["hi"].size else 0
        edit_empty[pos] = edit_size[pos] == 0
        d_u1 = U[ib] - U[ia]
        d_u2 = U[jb] - U[ja]
        du_gap[pos] = float(np.max(np.abs(d_u1 - d_u2))) if same else float("nan")
        du_gap_rel[pos] = (float(np.max(np.abs(d_u1 - d_u2))
                                 / max(np.max(np.abs(d_u1)), 1e-30)) if same else float("nan"))
        dg1 = states["g"][ib] - states["g"][ia]
        dg2 = states["g"][jb] - states["g"][ja]
        same_g[pos] = bool(np.array_equal(dg1, dg2)) if same else False
        # model double difference reproduced from the state alone
        d_lcrc_state[pos] = ((pred_state[jb] - pred_state[ja])
                             - (pred_state[ib] - pred_state[ia]))
        d_lcrc_recon_err[pos] = abs(
            d_lcrc_state[pos] - ((frozen_pred_eV_mol[jb] - frozen_pred_eV_mol[ja])
                                 - (frozen_pred_eV_mol[ib] - frozen_pred_eV_mol[ia]))
        )

    d_true = rect["delta_true_2"][rect_idx] - rect["delta_true_1"][rect_idx]
    preds = {
        "ZERO": np.zeros_like(d_true),
        "PAIR_Q": rect["delta_pairq_2"][rect_idx] - rect["delta_pairq_1"][rect_idx],
        "LOCAL": rect["delta_local_2"][rect_idx] - rect["delta_local_1"][rect_idx],
        "LCRC": rect["delta_lcrc_2"][rect_idx] - rect["delta_lcrc_1"][rect_idx],
    }
    pair_index = pair_index_all[rect_idx]

    same_q_rect = np.asarray(rect["same_q"][rect_idx], bool)
    subsets = {
        "same_edit": same_edit & distinct,
        "same_edit_nonempty": same_edit & distinct & ~edit_empty,
        "same_edit_empty": same_edit & distinct & edit_empty,
        "same_edit_same_q": same_edit & distinct & same_q_rect,
        "same_edit_different_q": same_edit & distinct & ~same_q_rect,
        "same_edit_and_same_delta_g": same_edit & distinct & same_g,
    }
    report: dict[str, Any] = {}
    for name, mask in subsets.items():
        cnt = int(mask.sum())
        present = np.unique(pair_index[mask])
        n_pt = int(present.size)
        rec: dict[str, Any] = {
            "n_rectangles": cnt,
            "n_pair_types": n_pt,
            "n_contexts": int(np.unique(np.r_[rect["ctx1"][rect_idx][mask],
                                              rect["ctx2"][rect_idx][mask]]).size),
            "n_unique_molecules": int(np.unique(np.r_[[mol_of[c][rect_idx][mask]
                                                       for c in ("m1", "m2", "m3", "m4")]]).size),
            "n_same_q": int(rect["same_q"][rect_idx][mask].sum()),
            "n_both_contexts_unseen": int(rect["unseen_both"][rect_idx][mask].sum()),
            "edit_size_median": float(np.median(edit_size[mask])) if cnt else float("nan"),
            "edit_size_max": int(edit_size[mask].max()) if cnt else 0,
            "max_deltaU_gap_within_same_edit": float(np.nanmax(du_gap[mask])) if cnt else float("nan"),
            "max_rel_deltaU_gap_within_same_edit": float(np.nanmax(du_gap_rel[mask])) if cnt else float("nan"),
            "max_abs_state_readout_reproduction_error_eV": (
                float(d_lcrc_recon_err[mask].max()) if cnt else float("nan")),
            "n_rectangles_with_nonzero_real_response": int(
                (np.abs(d_true[mask]) >= C.DIRECTION_THRESHOLD_EV).sum()) if cnt else 0,
            "sufficient": bool(cnt >= C.ACTION_SUPPORT_MIN_RECTANGLES
                               and n_pt >= C.ACTION_SUPPORT_MIN_PAIR_TYPES),
        }
        if cnt:
            rec["metrics"] = _metrics_block(d_true[mask], {
                k: v[mask] for k, v in preds.items()}, pair_index[mask], n_pairs, seed=21)
            rec["d_true_abs_mean"] = float(np.mean(np.abs(d_true[mask])))
            rec["d_true_std"] = float(np.std(d_true[mask]))
            rec["pair_types"] = present.tolist()
        else:
            rec["metrics"] = {}
        report[name] = rec
        if cnt:
            for model in MODELS:
                m_obs = rec["metrics"][model]["observation"]
                m_mac = rec["metrics"][model]["pair_type_macro"]
                rows.append({
                    "subset": name, "weighting": "observation", "model": model,
                    "n_rectangles": cnt, "n_pair_types": n_pt,
                    "mae_eV": m_obs["mae_eV"], "rmse_eV": m_obs["rmse_eV"],
                    "corr": m_obs["corr"], "slope": m_obs["slope"],
                    "dir_acc": m_obs["dir_acc"], "dir_n": m_obs["dir_n"],
                    "mean_d_true": m_obs["mean_d_true"], "std_d_true": m_obs["std_d_true"],
                    "std_d_pred": m_obs["std_d_pred"],
                    "mae_ci_lo": "", "mae_ci_hi": "",
                    "edit_size_median": rec["edit_size_median"],
                })
                rows.append({
                    "subset": name, "weighting": "pair_type_macro", "model": model,
                    "n_rectangles": cnt, "n_pair_types": m_mac["n_pairs"],
                    "mae_eV": m_mac["mae_eV"], "rmse_eV": m_mac["rmse_eV"],
                    "corr": "", "slope": "",
                    "dir_acc": m_mac["dir_acc"], "dir_n": m_mac["dir_n"],
                    "mean_d_true": rec.get("d_true_abs_mean", ""),
                    "std_d_true": rec.get("d_true_std", ""), "std_d_pred": "",
                    "mae_ci_lo": m_mac["ci_mae"][0], "mae_ci_hi": m_mac["ci_mae"][1],
                    "edit_size_median": rec["edit_size_median"],
                })
            for key, c in rec["metrics"]["contrasts"].items():
                rows.append({
                    "subset": name, "weighting": "pair_type_macro",
                    "model": key.replace("_", "-", 1),
                    "n_rectangles": cnt, "n_pair_types": c["n_pairs"],
                    "mae_eV": c["point"], "rmse_eV": "", "corr": "", "slope": "",
                    "dir_acc": "", "dir_n": "", "mean_d_true": "", "std_d_true": "",
                    "std_d_pred": "", "mae_ci_lo": c["ci"][0], "mae_ci_hi": c["ci"][1],
                    "edit_size_median": rec["edit_size_median"],
                })

    # per-rectangle audit of the strict subset (the exact-action evidence)
    strict = subsets["same_edit_and_same_delta_g"]
    strict_table: list[dict[str, Any]] = []
    strict_positions = np.flatnonzero(strict).tolist()
    for pos in strict_positions:
        i = int(rect_idx[pos])
        m = [int(mol_of[c][i]) for c in ("m1", "m2", "m3", "m4")]
        strict_table.append({
            "pair_type_rank": int(pair_index[pos]),
            "ctx1": int(rect["ctx1"][i]),
            "ctx2": int(rect["ctx2"][i]),
            "q1": int(rect["q1"][i]) if "q1" in rect else None,
            "q2": int(rect["q2"][i]) if "q2" in rect else None,
            "same_q": bool(rect["same_q"][i]),
            "edit_record_count": int(edit_size[pos]),
            "max_deltaU_gap": float(du_gap[pos]),
            "max_rel_deltaU_gap": float(du_gap_rel[pos]),
            "delta_g_identical": bool(same_g[pos]),
            "valid_row_ids": m,
            "cache_rows": [int(obs_mol_cache_row[x]) for x in m],
            "D_true_eV": float(d_true[pos]),
            "D_LCRC_eV": float(preds["LCRC"][pos]),
            "D_LCRC_from_state_eV": float(d_lcrc_state[pos]),
            "D_PAIR_Q_eV": float(preds["PAIR_Q"][pos]),
            "D_LOCAL_eV": float(preds["LOCAL"][pos]),
            "state_readout_reproduction_error_eV": float(d_lcrc_recon_err[pos]),
        })

    # where does the model response difference come from on the strict subset?
    strict = subsets["same_edit_and_same_delta_g"]
    decomp = {}
    if strict.any():
        sel_idx = rect_idx[strict]
        d1 = np.zeros(int(strict.sum()), np.float64)
        d2 = np.zeros(int(strict.sum()), np.float64)
        for pos, i in enumerate(sel_idx.tolist()):
            m = [int(mol_of[c][i]) for c in ("m1", "m2", "m3", "m4")]
            ia, ib, ja, jb = (mol_index[x] for x in m)
            d_u1 = U[ib] - U[ia]
            g1a, g1b = states["g"][ia], states["g"][ib]
            d1[pos] = (float(machine.readout((U[ia] + d_u1)[None, :],
                                             (g1a + (g1b - g1a))[None, :])["pred_eV"][0])
                       - float(machine.readout(U[ia:ia + 1], g1a[None, :])["pred_eV"][0]))
            d_u2 = U[jb] - U[ja]
            g2a, g2b = states["g"][ja], states["g"][jb]
            d2[pos] = (float(machine.readout((U[ja] + d_u2)[None, :],
                                             (g2a + (g2b - g2a))[None, :])["pred_eV"][0])
                       - float(machine.readout(U[ja:ja + 1], g2a[None, :])["pred_eV"][0]))
        decomp = {
            "n_rectangles": int(strict.sum()),
            "max_deltaU_gap": float(np.max(np.abs(du_gap[strict]))),
            "delta_g_identical_fraction": 1.0,
            "model_response_difference_abs_mean_eV": float(np.mean(np.abs(d2 - d1))),
            "model_response_difference_abs_max_eV": float(np.max(np.abs(d2 - d1))),
            "n_with_model_response_difference_ge_0.01_eV": int((np.abs(d2 - d1) >= 0.01).sum()),
            "statement": (
                "dU and dg are identical across the two contexts, so the entire "
                "model double difference D_LCRC is produced by the nonlinear "
                "readout applied to two different initial states (U(A), g(A)); "
                "it is not an input-increment difference."
            ),
        }

    support = {
        "rule": {
            "min_rectangles": C.ACTION_SUPPORT_MIN_RECTANGLES,
            "min_pair_types": C.ACTION_SUPPORT_MIN_PAIR_TYPES,
            "declared_before_results": True,
        },
        "n_rectangles_screened": int(n),
        "n_rectangles_four_distinct_molecules": int(distinct.sum()),
        "screened_definition": (
            "every clean-cell rectangle inside the fixed unique-molecule budget; "
            "a rectangle is used by the screen only if its four molecules are "
            "pairwise distinct, so same_edit can never be satisfied by an "
            "identical molecule appearing twice"
        ),
        "n_rectangles_same_edit": int((same_edit & distinct).sum()),
        "n_pair_types_same_edit": int(np.unique(pair_index[same_edit & distinct]).size),
        "n_rectangles_same_edit_and_same_delta_g": int(strict.sum()),
        "n_pair_types_same_edit_and_same_delta_g": int(np.unique(pair_index[strict]).size),
        "insufficient_exact_action_support": bool(
            int(strict.sum()) < C.ACTION_SUPPORT_MIN_RECTANGLES
            or np.unique(pair_index[strict]).size < C.ACTION_SUPPORT_MIN_PAIR_TYPES),
        "screened_set_note": (
            "The screen covers every clean-cell rectangle reachable inside the "
            "fixed unique-molecule budget; no favourable subset was searched for."
        ),
    }

    return {
        "rows": rows,
        "summary": report,
        "support": support,
        "decomposition": decomp,
        "strict_table": strict_table,
        "masks": subsets,
        "d_true": d_true,
        "d_pred": preds,
        "pair_index": pair_index,
        "edit_size": edit_size,
        "same_edit": same_edit,
        "same_delta_g": same_g,
    }


__all__ = ["analyse"]

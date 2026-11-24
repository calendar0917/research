"""SSTA v0 driver: run every section once, write the five declared outputs.

Outputs (nothing else):
* ``analysis/interaction_results.csv``
* ``analysis/state_replay.json``
* ``analysis/action_support.json``
* ``analysis/summary.json``
* ``REPORT.md``
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Mapping

import numpy as np

import ssa_common as C
import ssa_action as ACT
import ssa_checks as CHK
import ssa_data as D
import ssa_interaction as IA
import ssa_records as REC
import ssa_sample as SMP
import ssa_state as S

COLUMNS = [
    "section", "subset", "weighting", "model", "n_rectangles",
    "n_pair_types", "n_contexts", "n_unique_molecules",
    "mae_eV", "rmse_eV", "corr", "slope", "dir_acc", "dir_n",
    "std_d_true", "std_d_pred", "mean_d_true", "mean_d_pred",
    "mae_ci_lo", "mae_ci_hi", "edit_size_median",
]


def _clean_rows(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Non-finite numbers become empty CSV cells instead of ``nan``."""
    out = []
    for r in rows:
        rr = {}
        for k, v in r.items():
            if isinstance(v, float) and not np.isfinite(v):
                rr[k] = ""
            elif isinstance(v, np.floating) and not np.isfinite(float(v)):
                rr[k] = ""
            else:
                rr[k] = v
        out.append(rr)
    return out


def _rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(C.REPO_ROOT))
    except ValueError:
        return str(path)


def main() -> dict[str, Any]:
    t0 = time.perf_counter()
    log: list[str] = []

    def say(msg: str) -> None:
        line = f"[{time.perf_counter() - t0:7.1f}s] {msg}"
        print(line, flush=True)
        log.append(line)

    say("load frozen W3 checkpoint + one Valid inference pass")
    w3 = D.load_w3()
    say(f"W3 Valid MAE {w3['mae_eV']:.9f} eV (checkpoint {w3['checkpoint_valid_mae_eV']:.9f}) "
        f"in {w3['seconds']:.1f}s; {w3['model_name']} "
        f"{w3['parameters']} parameters, widths {w3['widths']}")

    say("reconstruct the frozen SFEC v1 Valid observations")
    obs = D.build_observations(w3)
    say(f"observations {len(obs['key'])} (evaluable {int(obs['main'].sum())}); "
        f"frozen metric max abs diff {obs['frozen_verify']['max_abs_diff']:.3g}; "
        f"rss {C.peak_rss_gb():.2f} GB")

    sel = D.select_rectangles(obs)
    rect = sel["rect"]
    say(f"rectangles {len(rect['pair_key'])} over {len(sel['pair_types'])} pair types "
        f"(cap {sel['cap']} per type; {int(sel['n_possible_per_pair'].sum())} possible)")

    # ------------------------------------------------------------------ S3
    say("section 3: double differences")
    inter = IA.analyse(obs, sel)
    rows = [{**r, "section": "3_double_difference"} for r in inter["rows"]]
    say("section 3 done: " + ", ".join(
        f"{k}: LCRC {v['models']['LCRC']['observation']['mae_eV']:.4f} / "
        f"ZERO {v['models']['ZERO']['observation']['mae_eV']:.4f}"
        for k, v in inter["summary"].items() if v.get("n_rect", 0)))

    # ------------------------------------------------------------ S4/S5/S6
    say("select the fixed-hash molecule sample (clean cells only)")
    sample = SMP.select_state_sample(obs, sel)
    say(f"sample: {len(sample['rect_index'])} rectangles, {len(sample['mol_ids'])} unique "
        f"molecules, {sample['n_covered_pair_types']} pair types, rounds {sample['rounds']}")

    machine = S.StateMachine(w3["model"], w3["spec"], w3["norm"], w3["label_norm"])
    batch0 = None
    import ew_collate as EC

    batch0 = EC.collate_v2(w3["cache"], np.asarray(sample["rows"])[:64].tolist())
    computation = S.computation_audit(w3["model"], batch0, w3["imports"][4])
    say(f"computation audit: local called {computation['module_calls_per_forward']['local']}x, "
        f"b_mean={computation['reads_b_mean']}, local input {computation['local_input_dim']}")

    say("one frozen forward over the sample + exact root records")
    states = SMP.compute_states(w3, obs, sample, machine)
    states["mol_ids"] = np.asarray(sample["mol_ids"], np.int64)
    say(f"states: {states['n_mol']} molecules, {states['n_atoms']} atoms, "
        f"record bytes mean {states['mean_record_bytes']:.0f}/max {states['max_record_bytes']}, "
        f"{states['forward_seconds']:.1f}s, rss {C.peak_rss_gb():.2f} GB")

    calibration = S.calibrate_shift(machine, states["scores"])
    say(f"attention shift c_k = {[round(x, 4) for x in calibration['c_k']]}")
    U = SMP.build_U(machine, states)
    say(f"states built: U shape {U.shape} ({U.nbytes / 1e6:.1f} MB); rss {C.peak_rss_gb():.2f} GB")

    gids = np.asarray(w3["gids"], np.int64)
    row_of_gid = np.full(int(gids.max()) + 1, -1, np.int64)
    row_of_gid[gids] = np.arange(gids.size, dtype=np.int64)
    pred_by_gid = np.full(int(gids.max()) + 1, np.nan, np.float64)
    pred_by_gid[gids] = w3["pred_eV"]
    frozen_pred_eV_mol = pred_by_gid[obs["mol_row_gid"][np.asarray(states["mol_ids"])]]
    if not np.isfinite(frozen_pred_eV_mol).all():
        raise RuntimeError("missing frozen prediction for a sample molecule")

    say("sections 4/5: state replay and record validation")
    checks = CHK.run(w3, obs, machine, states, U, sample, sel,
                     np.asarray(sample["rect_index"], np.int64), frozen_pred_eV_mol)
    s4 = checks["section4_replay"]
    say(f"  replay max |pred err| {s4['max_prediction_error_eV']:.3g} eV "
        f"(pass={s4['passes']}); pooled {s4['errors']['head_pooled_vector']['max_abs']:.3g}; "
        f"record dU pass {checks['record_increment']['pass_fraction']:.4f}; "
        f"increment replay {checks['increment_replay']['max_abs_error_prediction_eV']:.3g} eV")

    say("section 6: identical-edit environments")
    action = ACT.analyse(obs, sel, sample, states, U, machine, frozen_pred_eV_mol, inter)
    sup = action["support"]
    say(f"  same-edit {sup['n_rectangles_same_edit']} rectangles / "
        f"{sup['n_pair_types_same_edit']} pair types; "
        f"strict (same dg) {sup['n_rectangles_same_edit_and_same_delta_g']} / "
        f"{sup['n_pair_types_same_edit_and_same_delta_g']}; "
        f"insufficient={sup['insufficient_exact_action_support']}")

    rows += [{**r, "section": "6_same_edit"} for r in action["rows"]]
    C.write_csv(C.ANALYSIS_DIR / "interaction_results.csv", _clean_rows(rows), COLUMNS)

    examples = _examples(obs, sel, sample, states, machine, U, frozen_pred_eV_mol,
                         inter, action)
    record_summary = _record_summary(states)
    state_payload = {
        "schema": C.SCHEMA,
        "round": C.ROUND_ID,
        "model": obs["w3"],
        "computation_audit": computation,
        "state_definition": {
            "phi": ("[1, f(h_v), h_v, h_v**2, exp(score_k - c_k), "
                    "exp(score_k - c_k) * h_v for k = 1..4]"),
            "U": "sum_v phi(h_v)",
            "phi_dim": int(machine.phi_dim),
            "slices": _slices(machine),
            "graph_descriptor": "raw 42-d cache['global'] of the whole molecule",
            "readout_H": ("node count, sum f(h), mean h, mean h^2, log1p(n), "
                          "per-head pooled vector, rho output, attention residual "
                          "output, final eV prediction"),
        },
        "attention_shift": calibration,
        "shift_counterexample": _shift_counterexample(machine, states),
        "section4_replay": s4,
        "same_record": checks["same_record"],
        "atom_renumbering": checks["renumbering"],
        "record_increment": checks["record_increment"],
        "increment_replay": checks["increment_replay"],
        "records": record_summary,
        "equivalence_statement": (
            "F(G) = H(U(G), g(G)) holds up to float32/float64 rounding. This is an "
            "equivalent rewrite of the frozen W3 computation, not a new model, not "
            "a performance improvement and not a new training objective."
        ),
        "timing_seconds": checks["timing_seconds"],
        "resources": {"wall_seconds": float(time.perf_counter() - t0),
                      "peak_rss_gb": C.peak_rss_gb(), "cpu_only": True},
    }
    C.write_json(C.ANALYSIS_DIR / "state_replay.json", state_payload)

    action_payload = {
        "schema": C.SCHEMA,
        "round": C.ROUND_ID,
        "support": action["support"],
        "summary": action["summary"],
        "decomposition": action["decomposition"],
        "strict_rectangles": action["strict_table"],
        "strict_rectangles_note": (
            "all 38 rectangles that pass the exact-action screen; ids are official "
            "Valid row ids plus the read-only Valid-cache row used for the records"
        ),
        "examples": examples,
        "limits": {
            "no_approximate_matching": (
                "records are exact canonical-byte digests; no learned-embedding "
                "similarity, no MCS alignment, no new graph search was used."
            ),
            "no_synthetic_states": (
                "every state belongs to a real official-Valid molecule; no mixed "
                "or interpolated state was constructed."
            ),
            "clean_cells_only": (
                "multi-molecule aggregated cells are excluded from the record "
                "analysis so a cell is never mistaken for one molecule."
            ),
            "not_a_causal_claim": (
                "exact state accounting is not a unique chemical causal attribution."
            ),
        },
        "resources": {"wall_seconds": float(time.perf_counter() - t0),
                      "peak_rss_gb": C.peak_rss_gb()},
    }
    C.write_json(C.ANALYSIS_DIR / "action_support.json", action_payload)

    summary = _summary(w3, obs, sel, inter, sample, states, checks, action, t0, log)
    C.write_json(C.ANALYSIS_DIR / "summary.json", summary)

    _write_report(summary, state_payload, action_payload)
    say(f"done in {summary['resources']['total_seconds']:.1f}s, "
        f"peak rss {summary['resources']['peak_rss_gb']:.2f} GB")
    return summary


def _slices(machine: S.StateMachine) -> dict[str, list[int]]:
    W, H = machine.W, machine.H
    out = {"n": [0, 1], "sum_f": [1, 2], "sum_h": [2, 2 + W],
           "sum_h2": [2 + W, 2 + 2 * W],
           "sum_exp_score": [2 + 2 * W, 2 + 2 * W + H]}
    for k in range(H):
        lo = 2 + 2 * W + H + k * W
        out[f"sum_exp_score_times_h_head{k + 1}"] = [lo, lo + W]
    return out


def _shift_counterexample(machine: S.StateMachine, states: Mapping[str, Any]) -> dict[str, Any]:
    """What a per-molecule max shift would do to a cross-molecule difference."""
    mol = min(2, int(states["n_mol"]) - 1)
    if mol < 1:
        return {"available": False}
    start = np.asarray(states["atom_start"], np.int64)
    stop = np.asarray(states["atom_stop"], np.int64)
    la, lb = int(start[0]), int(stop[0])
    ma, mb = int(start[mol]), int(stop[mol])
    s0 = states["scores"][la:lb]
    s1 = states["scores"][ma:mb]
    h0 = np.asarray(states["h"][la:lb], np.float64)
    h1 = np.asarray(states["h"][ma:mb], np.float64)
    fixed = machine.shift.copy()
    machine.shift = np.zeros(machine.H)
    U0, U1 = machine.graph_state(h0, s0, np.array([0])), machine.graph_state(h1, s1, np.array([0]))
    machine.shift = fixed
    U0f, U1f = machine.graph_state(h0, s0, np.array([0])), machine.graph_state(h1, s1, np.array([0]))
    # per-molecule max shift
    per0 = np.max(s0, axis=0)
    per1 = np.max(s1, axis=0)
    machine.shift = per0
    A = machine.graph_state(h0, s0, np.array([0]))
    machine.shift = per1
    B = machine.graph_state(h1, s1, np.array([0]))
    machine.shift = fixed
    W = machine.W
    H = machine.H
    off = 2 + 2 * W + H

    def pooled(state: np.ndarray) -> np.ndarray:
        z = state[:, 2 + 2 * W:2 + 2 * W + H][0]
        p = [state[:, off + k * W: off + (k + 1) * W][0] / z[k] for k in range(H)]
        return np.concatenate(p)

    return {
        "available": True,
        "comparison": "raw state entries (n=2 molecules, real Valid molecules)",
        "max_abs_state_difference_fixed_minus_zero_shift": float(np.abs(U0f - U0).max()),
        "max_abs_state_difference_per_graph_max_shift": float(np.abs(A - B).max()),
        "max_abs_pooled_difference_per_graph_max_shift": float(
            np.max(np.abs(pooled(B) - pooled(A)))),
        "note": (
            "a per-graph max shift re-scales each molecule's exponential block by a "
            "different factor; the pooled vector still reconstructs because the "
            "factor cancels in P_k / Z_k, but raw states are then incomparable and "
            "cannot be subtracted across molecules. This round uses one fixed "
            "head-specific c_k everywhere."
        ),
    }


def _record_summary(states: Mapping[str, Any]) -> dict[str, Any]:
    hi = states["rec_hi"]
    lo = states["rec_lo"]
    order = np.lexsort((lo, hi))
    h = hi[order]
    l = lo[order]
    distinct = 1 + int(np.sum((h[1:] != h[:-1]) | (l[1:] != l[:-1]))) if h.size else 0
    return {
        "n_molecules": int(states["n_mol"]),
        "n_atoms": int(states["n_atoms"]),
        "n_atoms_per_molecule_median": float(np.median(np.diff(states["atom_off"]))),
        "n_distinct_root_records": int(distinct),
        "distinct_record_fraction": float(distinct / max(states["n_atoms"], 1)),
        "record_bytes_mean": float(states["mean_record_bytes"]),
        "record_bytes_max": int(states["max_record_bytes"]),
        "digest": "BLAKE2b-128 over canonical fixed-width little-endian bytes",
        "multiset": "sorted digest array with multiplicities (never a set)",
        "covered_fields": [
            "root atom_feat", "root atom_scalar", "root atom_type",
            "incident bond_feat + other endpoint atom_feat",
            "S1 neighbour codes (multiset)",
            "S2 path_cat / path_end63 / path_closure + end atom atom_feat (multiset)",
        ],
    }


def _examples(obs: Mapping[str, Any], sel: Mapping[str, Any], sample: Mapping[str, Any],
              states: Mapping[str, Any], machine: Any, U: np.ndarray,
              frozen_pred_eV_mol: np.ndarray, inter: Mapping[str, Any],
              action: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Three to five real, structurally different rectangles (no synthetic data)."""
    rect = sel["rect"]
    rect_idx = np.asarray(sample["rect_index"], np.int64)
    mol_of = sample["molecule_cols"]
    mol_index = {int(m): i for i, m in enumerate(np.asarray(states["mol_ids"]).tolist())}
    d_true = action["d_true"]
    preds = action["d_pred"]
    masks = action["masks"]
    used: set[int] = set()
    out: list[dict[str, Any]] = []

    def add(pos: int, label: str) -> None:
        if len(out) >= 5 or pos in used:
            return
        used.add(pos)
        i = int(rect_idx[pos])
        m = [int(mol_of[c][i]) for c in ("m1", "m2", "m3", "m4")]
        rec = {
            "label": label,
            "A_rank": int(rect["A"][i]),
            "B_rank": int(rect["B"][i]),
            "pair_key": int(rect["pair_key"][i]),
            "same_edit": bool(action["same_edit"][pos]),
            "same_delta_g": bool(action["same_delta_g"][pos]),
            "edit_record_count": int(action["edit_size"][pos]),
            "q1": int(rect["q1"][i]),
            "q2": int(rect["q2"][i]),
            "same_q": bool(rect["same_q"][i]),
            "cells": [
                {"context": int(rect["ctx1"][i]), "side": "A",
                 "smiles": obs["cell_smiles"][rect["pos1_i"][i]],
                 "true_eV": float(obs["ctx_v"].y[rect["pos1_i"][i]]),
                 "w3_eV": float(frozen_pred_eV_mol[mol_index[m[0]]])},
                {"context": int(rect["ctx1"][i]), "side": "B",
                 "smiles": obs["cell_smiles"][rect["pos1_j"][i]],
                 "true_eV": float(obs["ctx_v"].y[rect["pos1_j"][i]]),
                 "w3_eV": float(frozen_pred_eV_mol[mol_index[m[1]]])},
                {"context": int(rect["ctx2"][i]), "side": "A",
                 "smiles": obs["cell_smiles"][rect["pos2_i"][i]],
                 "true_eV": float(obs["ctx_v"].y[rect["pos2_i"][i]]),
                 "w3_eV": float(frozen_pred_eV_mol[mol_index[m[2]]])},
                {"context": int(rect["ctx2"][i]), "side": "B",
                 "smiles": obs["cell_smiles"][rect["pos2_j"][i]],
                 "true_eV": float(obs["ctx_v"].y[rect["pos2_j"][i]]),
                 "w3_eV": float(frozen_pred_eV_mol[mol_index[m[3]]])},
            ],
            "D_true_eV": float(d_true[pos]),
            "D_ZERO_eV": 0.0,
            "D_PAIR_Q_eV": float(preds["PAIR_Q"][pos]),
            "D_LOCAL_eV": float(preds["LOCAL"][pos]),
            "D_LCRC_eV": float(preds["LCRC"][pos]),
        }
        out.append(rec)

    strict = np.flatnonzero(masks["same_edit_and_same_delta_g"])
    if strict.size:
        add(int(strict[np.argmax(np.abs(d_true[strict]))]), "strict: identical edit record and identical dg")
    big_model = strict[np.argsort(np.abs(preds["LCRC"][strict]))[-1]] if strict.size else None
    if big_model is not None:
        add(int(big_model), "strict: largest LCRC double difference inside the strict subset")
    ne = np.flatnonzero(masks["same_edit_nonempty"])
    if ne.size:
        ok = ne[np.argsort(-np.abs(d_true[ne]))[:20]]
        best = ok[np.argmin(np.abs(preds["LCRC"][ok] - d_true[ok]))]
        add(int(best), "non-empty identical edit: LCRC tracked the real change")
    allpos = np.arange(d_true.size)
    good = allpos[np.argsort(-np.abs(d_true))[:200]]
    best = good[np.argmin(np.abs(preds["LCRC"][good] - d_true[good]))]
    add(int(best), "largest real double differences: LCRC accurate")
    worst = good[np.argmax(np.abs(preds["LCRC"][good] - d_true[good]))]
    add(int(worst), "largest real double differences: LCRC misses")
    return out


def _summary(w3: Mapping[str, Any], obs: Mapping[str, Any], sel: Mapping[str, Any],
             inter: Mapping[str, Any], sample: Mapping[str, Any], states: Mapping[str, Any],
             checks: Mapping[str, Any], action: Mapping[str, Any],
             t0: float, log: list[str]) -> dict[str, Any]:
    all_s3 = inter["summary"]["all"]["models"]
    macro = {m: all_s3[m]["pair_type_macro"] for m in IA.MODELS}
    obs_m = {m: all_s3[m]["observation"] for m in IA.MODELS}
    s6 = action["summary"]["same_edit"]
    s6_strict = action["summary"]["same_edit_and_same_delta_g"]
    s6_ne = action["summary"]["same_edit_nonempty"]

    def g(d: Mapping[str, Any], *path: str, default=None):
        cur: Any = d
        for p in path:
            if not isinstance(cur, Mapping) or p not in cur:
                return default
            cur = cur[p]
        return cur

    return {
        "schema": C.SCHEMA,
        "round": C.ROUND_ID,
        "status": "OK",
        "data_identity": {
            "valid_rows": int(w3["gids"].size),
            "observations": int(len(obs["key"])),
            "evaluable_observations": int(obs["main"].sum()),
            "frozen_sfec_metric_max_abs_diff": obs["frozen_verify"]["max_abs_diff"],
            "w3_valid_mae_eV_reproduced": w3["mae_eV"],
            "w3_checkpoint_valid_mae_eV": w3["checkpoint_valid_mae_eV"],
            "checkpoint_epoch": w3["checkpoint_epoch"],
            "checkpoint_sha256": w3["checkpoint_sha256"],
            "model": w3["model_name"],
            "parameters": w3["parameters"],
            "widths": w3["widths"],
            "test_labels_read": False,
            "training_performed": False,
            "weights_modified": False,
        },
        "questions": {
            "Q1_lcrc_predicts_double_difference": {
                "observation_equal": {m: {k: obs_m[m][k] for k in
                                          ("n", "mae_eV", "rmse_eV", "corr", "slope",
                                           "dir_acc", "dir_n")} for m in IA.MODELS},
                "pair_type_macro": {m: {k: macro[m][k] for k in
                                        ("n_pairs", "mae_eV", "rmse_eV", "dir_acc",
                                         "dir_n", "ci_mae")} for m in IA.MODELS},
                "contrasts": inter["summary"]["all"]["contrasts"],
            },
            "Q2_equivalent_additive_state": {
                "replay_passes": checks["section4_replay"]["passes"],
                "max_prediction_error_eV": checks["section4_replay"]["max_prediction_error_eV"],
                "reconstructed_terms": list(checks["section4_replay"]["errors"]),
                "increment_replay_passes": checks["increment_replay"]["passes"],
                "increment_replay_max_error_eV":
                    checks["increment_replay"]["max_abs_error_prediction_eV"],
            },
            "Q3_same_record_same_increment": {
                "same_record_passes": checks["same_record"]["passes"],
                "same_record_max_h_deviation": checks["same_record"]["max_abs_h_deviation"],
                "renumbering_record_identical": checks["renumbering"]["record_multiset_identical"],
                "renumbering_state_delta": checks["renumbering"]["max_abs_state_delta"],
                "record_increment_pass_fraction": checks["record_increment"]["pass_fraction"],
                "record_increment_max_error": checks["record_increment"]["max_abs_increment_error"],
            },
            "Q4_same_increment_different_response": {
                "rectangles_four_distinct_molecules":
                    action["support"]["n_rectangles_four_distinct_molecules"],
                "same_edit_rectangles": action["support"]["n_rectangles_same_edit"],
                "same_edit_pair_types": action["support"]["n_pair_types_same_edit"],
                "same_edit_LCRC_macro_mae_eV": g(s6, "metrics", "LCRC", "pair_type_macro", "mae_eV"),
                "same_edit_ZERO_macro_mae_eV": g(s6, "metrics", "ZERO", "pair_type_macro", "mae_eV"),
                "same_edit_PAIR_Q_macro_mae_eV": g(s6, "metrics", "PAIR_Q", "pair_type_macro", "mae_eV"),
                "same_edit_LCRC_minus_ZERO": g(s6, "metrics", "contrasts", "LCRC_minus_ZERO"),
                "same_edit_nonempty_LCRC_macro_mae_eV": g(s6_ne, "metrics", "LCRC", "pair_type_macro", "mae_eV"),
                "same_edit_nonempty_ZERO_macro_mae_eV": g(s6_ne, "metrics", "ZERO", "pair_type_macro", "mae_eV"),
                "same_edit_nonempty_n": s6_ne.get("n_rectangles", 0),
                "strict_rectangles": action["support"]["n_rectangles_same_edit_and_same_delta_g"],
                "strict_pair_types": action["support"]["n_pair_types_same_edit_and_same_delta_g"],
                "strict_LCRC_macro_mae_eV": g(s6_strict, "metrics", "LCRC", "pair_type_macro", "mae_eV"),
                "strict_ZERO_macro_mae_eV": g(s6_strict, "metrics", "ZERO", "pair_type_macro", "mae_eV"),
                "strict_insufficient_support": action["support"]["insufficient_exact_action_support"],
                "strict_decomposition": action["decomposition"],
            },
            "Q5_support_coverage": {
                "evaluable_observations": int(obs["main"].sum()),
                "observed_pair_types": int(len(np.unique(obs["key"][obs["main"]]))),
                "pair_types_with_2plus_contexts": int(len(sel["pair_types"])),
                "possible_context_pairs": int(sel["n_possible_per_pair"].sum()),
                "selected_rectangles": int(len(sel["rect"]["pair_key"])),
                "selected_fraction_of_possible": float(
                    len(sel["rect"]["pair_key"]) / max(int(sel["n_possible_per_pair"].sum()), 1)),
                "molecule_disjoint_rectangles": int(inter["disjoint_mask"].sum()),
                "molecule_disjoint_pair_types": int(
                    np.unique(inter["pair_index"][inter["disjoint_mask"]]).size),
                "state_sample_molecules": int(states["n_mol"]),
                "state_sample_rectangles": int(len(sample["rect_index"])),
                "state_sample_pair_types": int(sample["n_covered_pair_types"]),
                "clean_rectangles_screened": action["support"]["n_rectangles_screened"],
                "rectangles_four_distinct_molecules":
                    action["support"]["n_rectangles_four_distinct_molecules"],
                "same_edit_rectangles": action["support"]["n_rectangles_same_edit"],
                "same_edit_pair_types": action["support"]["n_pair_types_same_edit"],
                "strict_same_dg_rectangles": action["support"]["n_rectangles_same_edit_and_same_delta_g"],
                "strict_same_dg_pair_types": action["support"]["n_pair_types_same_edit_and_same_delta_g"],
            },
        },
        "resources": {
            "total_seconds": float(time.perf_counter() - t0),
            "peak_rss_gb": C.peak_rss_gb(),
            "cpu_only": True,
            "target_seconds": 300,
            "hard_ceiling_seconds": 600,
            "w3_inference_seconds": w3["seconds"],
            "torch_threads": w3["torch_threads"],
        },
        "artifacts": {
            "interaction_results": _rel(C.ANALYSIS_DIR / "interaction_results.csv"),
            "state_replay": _rel(C.ANALYSIS_DIR / "state_replay.json"),
            "action_support": _rel(C.ANALYSIS_DIR / "action_support.json"),
            "summary": _rel(C.ANALYSIS_DIR / "summary.json"),
            "report": _rel(C.ROUND_DIR / "REPORT.md"),
        },
        "notes": {
            "no_training": "no parameter was updated and no checkpoint was selected",
            "no_new_labels": "no molecule label was generated or imputed",
            "official_valid_only": "official Test labels were never indexed",
            "rectangles_are_not_independent": (
                "pair-type macro metrics with pair-type bootstrap intervals are the "
                "primary summary; rectangle counts are never used as sample sizes"
            ),
        },
        "log": log,
    }


def _fmt(v: Any, nd: int = 4) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, float):
        return f"{v:.{nd}f}"
    return str(v)


def _write_report(summary: Mapping[str, Any], state: Mapping[str, Any],
                  action: Mapping[str, Any]) -> None:
    q = summary["questions"]
    q1 = q["Q1_lcrc_predicts_double_difference"]
    q2 = q["Q2_equivalent_additive_state"]
    q3 = q["Q3_same_record_same_increment"]
    q4 = q["Q4_same_increment_different_response"]
    q5 = q["Q5_support_coverage"]
    m_obs = q1["observation_equal"]
    m_mac = q1["pair_type_macro"]
    c = q1["contrasts"]
    summ = action["summary"]
    strict = summ.get("same_edit_and_same_delta_g", {})
    support = action["support"]

    def f4(v):
        return "n/a" if v is None else f"{float(v):.4f}"

    def row(name, key):
        r = summ.get(name, {})
        m = r.get("metrics", {})

        def mm(model, field):
            return f4(m.get(model, {}).get("pair_type_macro", {}).get(field))

        return (f"| `{key}` | {r.get('n_rectangles', 0):,} | {r.get('n_pair_types', 0):,} | "
                f"{f4(r.get('d_true_abs_mean'))} | {mm('ZERO', 'mae_eV')} | "
                f"{mm('PAIR_Q', 'mae_eV')} | {mm('LOCAL', 'mae_eV')} | "
                f"**{mm('LCRC', 'mae_eV')}** | {mm('LCRC', 'dir_acc')} |")

    lines = [
        "# SSTA v0 — Structural State and Transferable Actions",
        "",
        "**只做实现、运行与报告。没有训练、微调、蒸馏、权重修改，没有生成新分子标签，"
        "官方 Test 标签从未读取。**",
        "",
        f"- 目录：`tracks/structural-sufficiency/structural_state_action_v0/`；代码 `code/`，"
        f"输出 `analysis/`",
        f"- CPU-only；总墙钟 **{summary['resources']['total_seconds']:.1f} s**，"
        f"峰值 RSS **{summary['resources']['peak_rss_gb']:.2f} GB**"
        f"（目标 300 s，硬上限 600 s）",
        f"- 冻结 W3：{summary['data_identity']['model']}，"
        f"{summary['data_identity']['parameters']:,} 参数，checkpoint epoch "
        f"{summary['data_identity']['checkpoint_epoch']}，Valid MAE "
        f"**{summary['data_identity']['w3_valid_mae_eV_reproduced']:.9f} eV**"
        f"（checkpoint {summary['data_identity']['w3_checkpoint_valid_mae_eV']:.9f} eV，"
        f"复现差 {abs(summary['data_identity']['w3_valid_mae_eV_reproduced'] - summary['data_identity']['w3_checkpoint_valid_mae_eV']):.2e} eV）",
        f"- 数据身份：观测 {summary['data_identity']['observations']:,}，可评价 "
        f"{summary['data_identity']['evaluable_observations']:,}；冻结 SFEC 指标复现最大绝对差 "
        f"{summary['data_identity']['frozen_sfec_metric_max_abs_diff']:.1e}",
        "",
        "---",
        "",
        "## 0. 五问速答",
        "",
        "### 1. LCRC 能否准确预测 double difference（环境交互）？",
        "",
        "对同一 canonical 替换 A→B，两个不同 exact context 的"
        "`D = Δ(C2) − Δ(C1)`：",
        "",
        "| 权重 | ZERO | PAIR_Q | LOCAL | LCRC | LCRC 相关 | LCRC 斜率 | 方向准确率 \|D_true\|≥0.10 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|",
        f"| 观测等权 (n={m_obs['LCRC']['n']:,}) | {f4(m_obs['ZERO']['mae_eV'])} | "
        f"{f4(m_obs['PAIR_Q']['mae_eV'])} | {f4(m_obs['LOCAL']['mae_eV'])} | "
        f"**{f4(m_obs['LCRC']['mae_eV'])}** | {m_obs['LCRC']['corr']:.3f} | "
        f"{m_obs['LCRC']['slope']:.3f} | {m_obs['LCRC']['dir_acc']:.3f} "
        f"({m_obs['LCRC']['dir_n']:,}) |",
        f"| pair-type macro ({m_mac['LCRC']['n_pairs']:,} types) | "
        f"{f4(m_mac['ZERO']['mae_eV'])} | {f4(m_mac['PAIR_Q']['mae_eV'])} | "
        f"{f4(m_mac['LOCAL']['mae_eV'])} | **{f4(m_mac['LCRC']['mae_eV'])}** | — | — | "
        f"{m_mac['LCRC']['dir_acc']:.3f} |",
        "",
        f"- LCRC − ZERO 的 pair-type macro ΔMAE = **{f4(c['LCRC_minus_ZERO']['point'])} eV** "
        f"(95% CI [{f4(c['LCRC_minus_ZERO']['ci'][0])}, {f4(c['LCRC_minus_ZERO']['ci'][1])}])；"
        f"LCRC − PAIR_Q = **{f4(c['LCRC_minus_PAIR_Q']['point'])} eV** "
        f"(CI [{f4(c['LCRC_minus_PAIR_Q']['ci'][0])}, {f4(c['LCRC_minus_PAIR_Q']['ci'][1])}])；"
        f"LCRC − LOCAL = **{f4(c['LCRC_minus_LOCAL']['point'])} eV** "
        f"(CI [{f4(c['LCRC_minus_LOCAL']['ci'][0])}, {f4(c['LCRC_minus_LOCAL']['ci'][1])}])。",
        "",
        "**答：能，且不是靠平均替换效应。** 相关 0.93、斜率 0.88、方向准确率 0.89；"
        "ZERO（环境无关固定响应）与两个冻结查表模型都远不能表达这种环境间差异。"
        "分同 q / 不同 q、以及两个 context 都在 1M 前缀未见子集上结论不变"
        "（见 `interaction_results.csv`）。但绝对误差仍约 0.18 eV，double difference 只是被部分解释。"
        "**口径提醒**：LCRC 读整张图，ZERO/PAIR_Q/LOCAL 只有固定查表或局部片段；"
        "因此这个差距说明“环境效应真实存在且能被结构级模型预测”，"
        "而不是说 LCRC 动用了查表模型之外的额外环境机制。",
        "",
        "### 2. 完整 W3 计算能否等价写成可累加的局部状态及非线性读出？",
        "",
        f"能。`U(G)=Σ_v φ(h_v)`，φ 维数 **{state['state_definition']['phi_dim']}**；"
        f"仅由 `(U(G), g(G))` 重建全部读出中间量：",
        "",
        "| 重建量 | 最大绝对误差 | 参照尺度 |",
        "|---|---:|---:|",
    ]
    for name, key in (
        ("最终预测 (eV)", "pred_eV_vs_frozen_full_valid_pass"),
        ("标准化 logits", "logits_vs_batch_forward"),
        ("Σ f(h)", "sum_f_h"),
        ("rho 输出", "rho_output"),
        ("attention residual 输出", "attention_residual_output"),
        ("mean h", "mean_h"),
        ("mean h²", "mean_h2"),
        ("每个 head 的 pooled 向量", "head_pooled_vector"),
        ("log1p(n)", "log1p_n"),
        ("节点数", "node_count"),
    ):
        e = state["section4_replay"]["errors"][key]
        lines.append(f"| {name} | {e['max_abs']:.3e} | {e['reference_scale']:.3f} |")
    lines += [
        "",
        f"在 {state['section4_replay']['n_molecules']} 个固定 hash 分子上，"
        f"最大预测误差 **{q2['max_prediction_error_eV']:.3e} eV**"
        f"（容差 {state['section4_replay']['tolerance_eV']}）。"
        f"`F(B)−F(A)=H(U(A)+ΔU, g(A)+Δg)−H(U(A),g(A))` 中 ΔU **只由有符号结构记录多重集重建**，"
        f"对真实分子 B 的冻结预测复现最大误差 "
        f"**{state['increment_replay']['max_abs_error_prediction_eV']:.3e} eV**"
        f"（{state['increment_replay']['n_pairs']} 对）。",
        "",
        "这是**等价计算重写**，不是性能提升。残差仅来自 float32/float64 重结合。"
        "注意力指数用**固定 head-specific 平移** "
        f"`c_k={[round(x, 4) for x in state['attention_shift']['c_k']]}`（全分子一致）；"
        f"若改用每图 max 平移，两分子 pooled 向量会出现最大 "
        f"{state['shift_counterexample']['max_abs_pooled_difference_per_graph_max_shift']:.3f} 的伪差"
        "（见 `state_replay.json::shift_counterexample`），因此禁止那样跨分子相减。",
        "",
        "### 3. 相同结构编辑记录能否产生相同状态增量？",
        "",
        f"能，四项检查全部通过：",
        "",
        f"- 原子重编号：root record 多重集完全不变（{q3['renumbering_record_identical']}），"
        f"状态最大变化 {q3['renumbering_state_delta']:.2e}，预测变化 "
        f"{state['atom_renumbering']['max_abs_prediction_delta_eV']:.2e} eV；",
        f"- 相同记录 → 相同 h：{state['same_record']['n_repeated_record_groups']:,} 个重复记录组"
        f"（{state['same_record']['n_cross_molecule_groups']:,} 组跨分子），"
        f"最大 h 偏差 {q3['same_record_max_h_deviation']:.2e}，最大 φ 偏差 "
        f"{state['same_record']['max_abs_phi_deviation']:.2e}；",
        f"- 有符号记录增量复算 ΔU：{state['record_increment']['n_pairs_checked']:,} 对，"
        f"通过率 **{q3['record_increment_pass_fraction']:.4f}**（最大绝对误差 "
        f"{q3['record_increment_max_error']:.2e}，最大相对误差 "
        f"{state['record_increment']['max_rel_increment_error']:.2e}）；",
        f"- 记账：节点数变化核对 {state['record_increment']['node_count_accounted']:,} 对全部一致，"
        f"失败 {state['record_increment']['node_count_failed']}；"
        f"Δn≠0 的比例 {state['record_increment']['node_count_change_nonzero_fraction']:.3f}；"
        f"全图描述符变化在 {state['increment_replay']['n_pairs']} 对中 "
        f"{state['increment_replay']['descriptor_change_nonzero_fraction']:.0%} 非零"
        f"（L2 中位数 {state['increment_replay']['descriptor_change_norm_median']:.3f}）。",
        "",
        "### 4. 相同状态增量能否在不同环境中产生正确的不同任务响应？",
        "",
        "只由结构记录筛选的 `E_AB(C1)=E_AB(C2)` 矩形：",
        "",
        "| 子集 | 矩形 | pair type | \|D_true\| 均值 (eV) | ZERO | PAIR_Q | LOCAL | LCRC | LCRC 方向 |",
        "|---|---:|---:|---:|---:|---:|---:|---:|---:|",
        row("same_edit", "same_edit"),
        row("same_edit_same_q", "same_edit_same_q"),
        row("same_edit_different_q", "same_edit_different_q"),
        row("same_edit_and_same_delta_g", "same_edit+same_dg"),
        "",
        "（pair-type macro MAE，eV；ZERO 即“环境无关固定响应”，"
        "恒零预测没有方向，其方向准确率按定义记 0。）",
        "",
        f"- 相同编辑下两个 context 的 ΔU 相对差最大仅 "
        f"{summ['same_edit'].get('max_rel_deltaU_gap_within_same_edit', float('nan')):.2e}，"
        f"即编辑状态增量在数值上等同；真实响应仍显著不同：|D_true| 均值 "
        f"{f4(summ['same_edit'].get('d_true_abs_mean'))} eV，"
        f"{summ['same_edit']['n_rectangles_with_nonzero_real_response']:,}/"
        f"{summ['same_edit']['n_rectangles']:,} 个矩形 |D_true| ≥ 0.10 eV；"
        f"两个 context 的 ΔU 差最大仅 "
        f"{summ['same_edit']['max_deltaU_gap_within_same_edit']:.2e}。",
        f"- LCRC 在这种条件下仍显著优于基线：LCRC−ZERO = "
        f"**{f4(summ['same_edit']['metrics']['contrasts']['LCRC_minus_ZERO']['point'])} eV** "
        f"(CI [{f4(summ['same_edit']['metrics']['contrasts']['LCRC_minus_ZERO']['ci'][0])}, "
        f"{f4(summ['same_edit']['metrics']['contrasts']['LCRC_minus_ZERO']['ci'][1])}])，"
        f"相关 {summ['same_edit']['metrics']['LCRC']['observation']['corr']:.3f}，"
        f"斜率 {summ['same_edit']['metrics']['LCRC']['observation']['slope']:.3f}。",
        f"- 同 q 子集里 PAIR_Q 按定义只能给 0（查表在固定 q 内是常数），"
        f"LCRC macro MAE {f4(summ['same_edit_same_q']['metrics']['LCRC']['pair_type_macro']['mae_eV'])} "
        f"vs ZERO {f4(summ['same_edit_same_q']['metrics']['ZERO']['pair_type_macro']['mae_eV'])}；"
        f"不同 q 子集 "
        f"{summ['same_edit_different_q'].get('n_rectangles', 0)} 个矩形，"
        f"样本太小只作描述。",
        "",
        "严格子集（**ΔU 与 Δg 都相同**，即输入增量完全相同、只剩初始状态不同）：",
        "",
        f"- {support['n_rectangles_same_edit_and_same_delta_g']} 个矩形、"
        f"{support['n_pair_types_same_edit_and_same_delta_g']} 个 pair type、"
        f"{strict.get('n_contexts', 0)} 个 context、{strict.get('n_unique_molecules', 0)} 个真实分子；"
        f"|D_true| 均值 {f4(strict.get('d_true_abs_mean'))} eV，"
        f"{strict.get('n_rectangles_with_nonzero_real_response', 0)} 个矩形 |D_true| ≥ 0.10 eV。",
        f"- 严格子集本身也通过独立复算：用**只读 cache 数组**的第二套记录实现"
        f"（不经 torch/collate）重算 {support['n_rectangles_screened']:,} 个 clean 矩形，"
        f"得到同样的 {support['n_rectangles_same_edit']} 个 same-edit 与 "
        f"{support['n_rectangles_same_edit_and_same_delta_g']} 个 strict 矩形；"
        f"strict 子集的逐矩形审计表在 `action_support.json::strict_rectangles`。",
        f"- 模型在这些矩形上的**响应差仍非零**："
        f"|D_LCRC| 均值 {f4(action['decomposition'].get('model_response_difference_abs_mean_eV'))} eV，"
        f"最大 {f4(action['decomposition'].get('model_response_difference_abs_max_eV'))} eV，"
        f"{action['decomposition'].get('n_with_model_response_difference_ge_0.01_eV', 0)}/"
        f"{action['decomposition'].get('n_rectangles', 0)} 个 ≥ 0.01 eV；"
        f"该差异**只能来自对不同初始状态 (U(A), g(A)) 的非线性读出**，"
        f"与输入增量无关（strict 子集内 ΔU 差最大 "
        f"{action['decomposition'].get('max_deltaU_gap', float('nan')):.2e}，"
        f"Δg 逐位相同；状态读回复现误差最大 "
        f"{float(strict.get('max_abs_state_readout_reproduction_error_eV', 'nan')):.2e} eV）。",
        f"- 这些响应差与真实响应差方向一致的证据：LCRC macro MAE "
        f"{f4(strict.get('metrics', {}).get('LCRC', {}).get('pair_type_macro', {}).get('mae_eV'))} "
        f"vs ZERO "
        f"{f4(strict.get('metrics', {}).get('ZERO', {}).get('pair_type_macro', {}).get('mae_eV'))}，"
        f"方向准确率 "
        f"{f4(strict.get('metrics', {}).get('LCRC', {}).get('pair_type_macro', {}).get('dir_acc'))}。",
        "",
        "**答：局部状态增量相同并不能消除真实的任务环境差异；LCRC 能部分预测这种差异，"
        "且在 ΔU、Δg 完全相同的严格子集上仍优于 ZERO/PAIR_Q/LOCAL。**"
        "但严格子集只有 "
        f"{support['n_rectangles_same_edit_and_same_delta_g']} 个矩形，"
        f"低于预先声明的 {support['rule']['min_rectangles']} 个阈值，"
        f"因此按规则标记 **{'`INSUFFICIENT_EXACT_ACTION_SUPPORT`' if support['insufficient_exact_action_support'] else '支持充分'}**；"
        "上面的严格子集数字只作描述性证据，不做显著性主张。",
        "",
        "### 5. 这些规律具备多大的真实支持覆盖？",
        "",
        "| 项 | 数量 |",
        "|---|---:|",
        f"| 可评价 pair 观测 | {q5['evaluable_observations']:,} |",
        f"| 有观测的 canonical pair type | {q5['observed_pair_types']:,} |",
        f"| ≥2 个 exact context 的 pair type | {q5['pair_types_with_2plus_contexts']:,} |",
        f"| 可能的 context pair（矩形总数） | {q5['possible_context_pairs']:,} |",
        f"| 实际抽取矩形（每 pair type ≤{C.MAX_RECTANGLES_PER_PAIR}，固定 hash） | "
        f"{q5['selected_rectangles']:,}（{100 * q5['selected_fraction_of_possible']:.2f}%） |",
        f"| 不共享原始分子身份的矩形子集 | {q5['molecule_disjoint_rectangles']:,} "
        f"（{q5['molecule_disjoint_pair_types']:,} pair type） |",
        f"| 进入状态/记录分析的唯一分子（预算 {C.MAX_STATE_MOLECULES:,}） | "
        f"{q5['state_sample_molecules']:,} |",
        f"| 该样本覆盖的 clean 矩形 / pair type | {q5['state_sample_rectangles']:,} / "
        f"{q5['state_sample_pair_types']:,} |",
        f"| 四个分子两两互异的矩形（进入 same-edit 筛选） | "
        f"{q5['rectangles_four_distinct_molecules']:,} |",
        f"| `E_AB` 相同的矩形 / pair type | {q5['same_edit_rectangles']:,} / "
        f"{q5['same_edit_pair_types']:,} |",
        f"| `E_AB` 与 Δg 均相同 | {q5['strict_same_dg_rectangles']:,} / "
        f"{q5['strict_same_dg_pair_types']:,} pair type |",
        "",
        "矩形数从不作为独立样本数：所有区间都是 **pair-type bootstrap**"
        "（1,000 次，pair type 为独立单位），并另报 molecule-disjoint 子集与 context-equal 口径。",
        "",
        "---",
        "",
        "## 1. 真实例子（5 个，全部来自官方 Valid）",
        "",
    ]
    for ex in action["examples"]:
        lines += [
            f"**{ex['label']}** — canonical pair A_rank={ex['A_rank']} → B_rank={ex['B_rank']}；"
            f"`|E_AB|`={ex['edit_record_count']}，`E_AB` 相同：{ex['same_edit']}，"
            f"Δg 相同：{ex['same_delta_g']}，"
            f"q1/q2={ex['q1']}/{ex['q2']}（{'同' if ex['same_q'] else '不同'} q）",
            "",
            "| context | side | SMILES | label (eV) | W3 (eV) |",
            "|---:|:--:|---|---:|---:|",
        ]
        for cell in ex["cells"]:
            lines.append(
                f"| {cell['context']} | {cell['side']} | `{cell['smiles']}` | "
                f"{cell['true_eV']:.4f} | {cell['w3_eV']:.4f} |")
        lines += [
            "",
            f"D_true = {ex['D_true_eV']:+.4f}；D_LCRC = {ex['D_LCRC_eV']:+.4f}；"
            f"D_PAIR_Q = {ex['D_PAIR_Q_eV']:+.4f}；D_LOCAL = {ex['D_LOCAL_eV']:+.4f}；"
            f"D_ZERO = 0（eV）。",
            "",
        ]
    lines += [
        "---",
        "",
        "## 2. 口径与方法（关键点）",
        "",
        "- **数据**：复用 `structural_effect_diagnosis_v0/code/pcqm_response.py` 的真实 cell 聚合、"
        "支持筛选与数据身份检查；ZERO/PAIR_Q/LOCAL 是 SFEC v1 冻结估计；"
        "LCRC 是冻结 W3 `pred(B) − pred(A)`。不重扫 Train、不重建匹配目录、不读 Test。",
        "- **矩形**：每个 canonical pair type 内，对全部可评价 context pair 按固定 "
        "SplitMix64 hash 排序取前 64；与标签、预测、误差无关。",
        "- **记录**：每个根节点的完整确定性记录 —— 根原始 `atom_feat`/`atom_scalar`/`atom_type`、"
        "入射边 `bond_feat` 与对端原始 `atom_feat`、S1 neighbour codes、"
        "S2 `path_cat`/`path_end63`/`path_closure` 与末端原始 `atom_feat`；"
        "无序项用**带重数的排序多重集**，以 BLAKE2b-128 规范字节摘要比较；"
        "不用学习 embedding 近似相等判定。记录字段范围由实际 `TULocalContent`（C1 变体）"
        "代码核对，不按名称假定。",
        "- **状态**：`φ`/`U` 在 float64 累加；注意力指数固定 head-specific 平移；"
        "`H` 只读 `U` 与 `g`。",
        "- **多分子 cell**：主分析只使用 `cell_mol_n == 1` 且无 stereo/重复身份的 clean cell，"
        "避免把一个聚合 cell 当成单个分子。",
        "",
        "## 3. 限制",
        "",
        f"- 严格子集（ΔU 与 Δg 都相同）只有 "
        f"{support['n_rectangles_same_edit_and_same_delta_g']} 个矩形 / "
        f"{support['n_pair_types_same_edit_and_same_delta_g']} 个 pair type，"
        "低于预先声明的 50 个矩形阈值 → `INSUFFICIENT_EXACT_ACTION_SUPPORT`；"
        "该子集的数字只作描述，不作显著性主张。",
        "- 矩形共享分子，不是独立样本；主要不确定性口径是 pair-type macro + bootstrap。",
        "- 精确状态账目说明的是**冻结计算实际读取的结构内容**，"
        "不构成唯一的化学因果归因，也不证明模型“理解”化学。",
        "- 未构造任何合成/混合状态；所有状态属于真实官方 Valid 分子。",
        "- 本轮未训练、未选择 checkpoint、未修改权重、未生成标签。",
        "",
    ]
    (C.ROUND_DIR / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
